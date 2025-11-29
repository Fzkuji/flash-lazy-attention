# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified to use LazyAttention with Triton kernel

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from adasplash import lazy_attention_triton
    HAS_LAZY_ATTENTION = True
except ImportError:
    warnings.warn(
        "AdaSplash is not installed. Please install it via `pip install adasplash`",
        category=ImportWarning,
    )
    lazy_attention_triton = None

logger = logging.get_logger(__name__)


class LazyAttentionLayer(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int = 512,  # Required for Lazy Attention
        rope_theta: float | None = 10000.,
        max_position_embeddings: int | None = None,
        layer_idx: int = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        if lazy_attention_triton is None:
            raise ImportError("Please install AdaSplash via `pip install adasplash` first")

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

        # === Lazy Attention specific parameters ===
        # Learnable head-wise attention biases [H, window_size]
        self.bias = nn.Parameter(torch.zeros(self.num_heads, self.window_size))
        # Learnable sparsity parameter [H]
        self.tau = nn.Parameter(torch.full((self.num_heads,), -1.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get('cu_seqlens')

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        # === Convert to [B, H, L, D] format for Lazy Attention ===
        q = rearrange(q, 'b l h d -> b h l d')
        k = rearrange(k, 'b l h d -> b h l d')
        v = rearrange(v, 'b l h d -> b h l d')

        # === Handle varlen: convert attention_mask or cu_seqlens to varlen format ===
        varlen = None
        if attention_mask is not None:
            # attention_mask: [B, L] with 1 for valid tokens, 0 for padding
            # varlen: [B] with actual sequence lengths
            varlen = attention_mask.sum(dim=1).to(torch.int32)
        elif cu_seqlens is not None:
            # cu_seqlens: cumulative sequence lengths [0, len1, len1+len2, ...]
            # Convert to varlen: [len1, len2, ...]
            varlen = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)

        # === Call Lazy Attention Triton kernel ===
        o = lazy_attention_triton(
            q, k, v,
            bias=self.bias,
            tau=self.tau,
            window_size=self.window_size,
            varlen=varlen
        )

        # === Convert back to [B, L, H*D] ===
        o = rearrange(o, 'b h l d -> b l (h d)')
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values


# ==========================================
# Simple Usage Example
# ==========================================

if __name__ == "__main__":
    # Initialize layer
    layer = LazyAttentionLayer(
        hidden_size=512,
        num_heads=8,
        window_size=512,
        rope_theta=10000
    ).cuda()

    # Example 1: Without padding (no attention_mask)
    batch_size, seq_len, hidden_size = 2, 128, 512
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device='cuda')

    output, _, _ = layer(hidden_states)
    print(f"Output shape: {output.shape}")  # [2, 128, 512]

    # Example 2: With padding (attention_mask)
    attention_mask = torch.ones(batch_size, seq_len, device='cuda', dtype=torch.long)
    attention_mask[0, 64:] = 0  # Batch 0: only first 64 tokens are valid
    attention_mask[1, 100:] = 0  # Batch 1: only first 100 tokens are valid

    output, _, _ = layer(hidden_states, attention_mask=attention_mask)
    print(f"Output shape with varlen: {output.shape}")  # [2, 128, 512]

    # Verify padding positions are masked
    print(f"Padding positions are zero: {torch.allclose(output[0, 64:, :], torch.zeros_like(output[0, 64:, :]))}")
