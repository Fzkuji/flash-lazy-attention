import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.inv_freq.shape[0]: # simplistic cache check
             self.seq_len_cached = None
        
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]

        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LazyAttention(nn.Module):
    """
    Lazy Attention Implementation.
    
    Key components:
    1. Positional Discrimination:
       - RoPE (Dimension-wise)
       - Learnable Attention Biases (Head-wise, distance dependent)
    2. Elastic-Softmax:
       - ReLU(Softmax(scores) + tau/i)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 512,
        rope_base: int = 10000,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        # 1. RoPE
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)

        # 2. Learnable Attention Biases
        # Shape: [num_heads, window_size + 1] to cover distance 0 to window_size
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, window_size + 1))
        
        # 3. Elastic-Softmax Tau
        # Initialize to -1.0 as per paper
        self.tau = nn.Parameter(torch.full((num_heads,), -1.0))

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, dim]
        mask: [batch_size, seq_len, seq_len] or None (causal implied if None usually, but explicit is better)
        """
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, L, D_h]
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(v, seq_len=L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Compute Scores
        # s_{ij} = (R_i q_i)^T (R_j k_j) / sqrt(d)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # [B, H, L, L]

        # Add Learnable Attention Biases
        # Distance |i-j|
        # We need a matrix of distances.
        indices = torch.arange(L, device=x.device)
        dist = torch.abs(indices[:, None] - indices[None, :]) # [L, L]
        
        # Clip to window size. Entries > window_size get 0 bias (effective bias)
        # Implementation: Gather biases for dist <= W. For dist > W, bias is 0.
        # We can use embedding lookup or gather.
        
        # Mask for within window
        is_within_window = (dist <= self.window_size)
        
        # We clamp dist to valid range for lookup, then mask result
        safe_dist = dist.clamp(max=self.window_size)
        
        # bias: [H, W+1] -> lookup -> [H, L, L]
        # We want to broadcast bias [H, L, L] to [B, H, L, L]
        
        # biases: [num_heads, window_size + 1]
        # safe_dist: [L, L]
        # gathered_bias: [num_heads, L, L]
        gathered_bias = self.attention_biases[:, safe_dist] 
        
        # Apply window mask: where dist > window_size, bias is 0
        gathered_bias = gathered_bias * is_within_window.unsqueeze(0).type_as(gathered_bias)
        
        scores = scores + gathered_bias.unsqueeze(0)

        # Causal Mask (if needed, usually yes for LLM)
        # Assuming causal for now as per paper context (LLM generation)
        causal_mask = torch.tril(torch.ones((L, L), device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        
        if mask is not None:
             scores = scores.masked_fill(mask == 0, float('-inf'))

        # Elastic-Softmax
        # 1. Softmax
        attn_probs = F.softmax(scores, dim=-1) # [B, H, L, L]

        # 2. Add tau / i
        # i is the number of attended tokens. 
        # For causal attention at position i (0-indexed), we attend to i+1 tokens (0..i).
        # So denominator is (row_index + 1).
        
        row_indices = torch.arange(1, L + 1, device=x.device).view(1, 1, L, 1) # [1, 1, L, 1]
        tau_term = self.tau.view(1, self.num_heads, 1, 1) / row_indices # [1, H, L, 1] -> broadcasts to [1, H, L, L]
        
        attn_elastic = F.relu(attn_probs + tau_term)
        
        # Note: attn_elastic sums do NOT necessarily equal 1.
        
        output = torch.matmul(attn_elastic, v) # [B, H, L, D_h]
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        
        return self.o_proj(output)

