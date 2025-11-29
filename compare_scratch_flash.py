#!/usr/bin/env python3
"""Compare scratch (PyTorch) vs flash (Triton) implementations"""
import torch
import torch.nn.functional as F
from einops import rearrange

# Test parameters
torch.manual_seed(42)
B, H, L, D = 2, 4, 32, 64
max_bias_length = 512

# Create test inputs
q = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
k = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
bias = torch.randn(H, max_bias_length, device='cuda', dtype=torch.bfloat16)
tau = torch.full((H,), -1.0, device='cuda', dtype=torch.bfloat16)

print("="*80)
print("Comparing Scratch (PyTorch) vs Flash (Triton) Forward Pass")
print("="*80)
print(f"\nInput shapes:")
print(f"  q: {q.shape}  (B, H, L, D)")
print(f"  k: {k.shape}")
print(f"  v: {v.shape}")
print(f"  bias: {bias.shape}  (H, max_bias_length)")
print(f"  tau: {tau.shape}  (H,)")

# ========== Scratch Implementation (PyTorch) ==========
print(f"\n{'='*80}")
print("Running Scratch (PyTorch) Implementation")
print(f"{'='*80}")

# Convert to scratch format: [B, H, L, D] -> [B, L, H, D]
q_scratch = rearrange(q, 'b h l d -> b l h d')
k_scratch = rearrange(k, 'b h l d -> b l h d')
v_scratch = rearrange(v, 'b h l d -> b l h d')

# Transpose to [B, H, L, D]
q_s = q_scratch.transpose(1, 2)
k_s = k_scratch.transpose(1, 2)
v_s = v_scratch.transpose(1, 2)

# Compute attention scores
scaling = D ** -0.5
attn_scores = torch.matmul(q_s, k_s.transpose(2, 3)) * scaling

# Apply learnable bias
seq_len_q, seq_len_k = attn_scores.shape[-2:]
rel_pos = torch.arange(seq_len_q, device=attn_scores.device)[:, None] - \
          torch.arange(seq_len_k, device=attn_scores.device)[None, :]
valid_mask = (0 <= rel_pos) & (rel_pos < max_bias_length)
indices = rel_pos.clamp(0, max_bias_length - 1)
bias_mat = bias[:, indices] * valid_mask.to(attn_scores.dtype)
attn_scores = attn_scores + bias_mat[None, :, :, :]

# Apply causal mask
causal_mask = torch.tril(torch.ones(L, L, device='cuda', dtype=torch.bool))
attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

# Softmax
attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)

# Elastic-Softmax
tau_s = tau.view(1, H, 1, 1)
i_positions = torch.arange(1, seq_len_q + 1, device=attn_weights.device, dtype=attn_weights.dtype)
i_positions = i_positions.view(1, 1, -1, 1)
attn_weights = F.relu(attn_weights + tau_s / i_positions)

# Apply to values
out_scratch = torch.matmul(attn_weights, v_s)
out_scratch = rearrange(out_scratch, 'b h l d -> b l h d')
out_scratch = rearrange(out_scratch, 'b l h d -> b h l d')  # Back to [B, H, L, D]

print(f"\nScratch output shape: {out_scratch.shape}")
print(f"Scratch output mean: {out_scratch.mean().item():.6f}")
print(f"Scratch output std: {out_scratch.std().item():.6f}")
print(f"Scratch output min: {out_scratch.min().item():.6f}")
print(f"Scratch output max: {out_scratch.max().item():.6f}")

# ========== Flash Implementation (Triton) ==========
print(f"\n{'='*80}")
print("Running Flash (Triton) Implementation")
print(f"{'='*80}")

from adasplash import lazy_attention_triton

out_flash = lazy_attention_triton(
    q, k, v,
    bias=bias,
    tau=tau,
    window_size=max_bias_length,
    varlen=None
)

print(f"\nFlash output shape: {out_flash.shape}")
print(f"Flash output mean: {out_flash.mean().item():.6f}")
print(f"Flash output std: {out_flash.std().item():.6f}")
print(f"Flash output min: {out_flash.min().item():.6f}")
print(f"Flash output max: {out_flash.max().item():.6f}")

# ========== Compare ==========
print(f"\n{'='*80}")
print("Comparison")
print(f"{'='*80}")

diff = (out_scratch - out_flash).abs()
print(f"\nAbsolute difference:")
print(f"  Mean: {diff.mean().item():.6f}")
print(f"  Max: {diff.max().item():.6f}")
print(f"  Median: {diff.median().item():.6f}")

rel_diff = diff / (out_scratch.abs() + 1e-8)
print(f"\nRelative difference:")
print(f"  Mean: {rel_diff.mean().item():.6f}")
print(f"  Max: {rel_diff.max().item():.6f}")

# Check if close
is_close = torch.allclose(out_scratch, out_flash, atol=1e-2, rtol=1e-2)
print(f"\nOutputs match (atol=1e-2, rtol=1e-2): {is_close}")

if not is_close:
    print("\n⚠️  WARNING: Outputs do NOT match!")
    print("This indicates a bug in the Triton kernel implementation.")
else:
    print("\n✅ Outputs match - forward pass is correct!")
