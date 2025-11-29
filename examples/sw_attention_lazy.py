# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified to use Lazy Attention

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.utils import logging

# ========== 关键修改 1: 导入 lazy_attention_triton ==========
try:
    from adasplash import lazy_attention_triton
    HAS_LAZY_ATTENTION = True
except ImportError:
    warnings.warn(
        "AdaSplash is not installed. Please install it via `pip install adasplash`",
        category=ImportWarning
    )
    lazy_attention_triton = None
    HAS_LAZY_ATTENTION = False


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

logger = logging.get_logger(__name__)


class SWAttention(nn.Module):

    def __init__(
            self,
            hidden_size: int = 2048,
            num_heads: int = 32,
            num_kv_heads: Optional[int] = None,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            window_size: Optional[int] = None,  # 滑动窗口大小（可选）
            rope_theta: Optional[float] = 10000.,
            max_position_embeddings: Optional[int] = None,
            layer_idx: int = None,
            use_learnable_bias: bool = True,  # 兼容原始接口，Lazy Attention 总是使用可学习 bias
            max_bias_length: int = 512,       # ========== 关键修改 2: 可学习bias的最大长度 ==========
    ):
        super().__init__()

        if not HAS_LAZY_ATTENTION:
            raise ImportError("Please install AdaSplash via `pip install adasplash` first")

        # Note: use_learnable_bias is kept for compatibility with original interface
        # Lazy Attention always uses learnable bias with max_bias_length

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

        self.window_size = window_size  # 滑动窗口大小（可选）
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx
        self.max_bias_length = max_bias_length  # 可学习bias的最大长度

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # ========== 关键修改 3: Lazy Attention 的可学习参数 ==========
        # 每个 head 的位置 bias: [num_heads, max_bias_length]
        self.bias = nn.Parameter(torch.zeros(self.num_heads, self.max_bias_length))

        # 每个 head 的稀疏性参数: [num_heads]
        self.tau = nn.Parameter(torch.full((self.num_heads,), -1.0))

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            output_attentions: bool = False,  # Lazy Attention 不返回 attention weights
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get('cu_seqlens', None)

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

        # Apply RoPE
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        # ========== 关键修改 4: 处理 GQA (Grouped Query Attention) ==========
        # Lazy Attention 需要所有 heads 有相同数量,所以需要 repeat k, v
        if self.num_kv_groups > 1:
            # [B, L, num_kv_heads, D] -> [B, L, num_heads, D]
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)

        # ========== 关键修改 5: 转换 shape [B, L, H, D] -> [B, H, L, D] ==========
        q = rearrange(q, 'b l h d -> b h l d')
        k = rearrange(k, 'b l h d -> b h l d')
        v = rearrange(v, 'b l h d -> b h l d')

        # ========== 关键修改 6: 处理 attention_mask -> varlen ==========
        varlen = None
        if attention_mask is not None:
            # attention_mask: [B, 1, L, L] 或 [B, L]
            if attention_mask.dim() == 4:
                # 从 causal mask 中提取实际长度
                # 对于每个样本,找到最后一个 1 的位置
                varlen = (attention_mask[:, 0, 0, :] != 0).sum(dim=-1).to(torch.int32)
            elif attention_mask.dim() == 2:
                # [B, L] - 直接求和得到实际长度
                varlen = attention_mask.sum(dim=1).to(torch.int32)

        # ========== 关键修改 7: 调用 Lazy Attention ==========
        attn_output = lazy_attention_triton(
            q, k, v,
            bias=self.bias,
            tau=self.tau,
            window_size=self.max_bias_length,
            varlen=varlen
        )

        # ========== 关键修改 8: 转回 [B, H, L, D] -> [B, L, H*D] ==========
        attn_output = rearrange(attn_output, 'b h l d -> b l (h d)')
        o = self.o_proj(attn_output)

        # Lazy Attention 不返回 attention weights
        attentions = None

        return o, attentions, past_key_values
