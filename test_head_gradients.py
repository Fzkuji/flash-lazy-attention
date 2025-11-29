#!/usr/bin/env python3
"""Test if all heads have gradients for bias and tau"""
import torch
from adasplash import lazy_attention_triton

print("="*80)
print("Testing if all heads receive gradients")
print("="*80)

B, H, L, D = 1, 4, 8, 16
window_size = 16

# Create test inputs
q = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
k = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
v = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
bias = torch.randn(H, window_size, device='cuda', requires_grad=True)
tau = torch.full((H,), -1.0, device='cuda', requires_grad=True)

print(f"\nInput shapes:")
print(f"  q: {q.shape}")
print(f"  k: {k.shape}")
print(f"  v: {v.shape}")
print(f"  bias: {bias.shape}")
print(f"  tau: {tau.shape}")

# Forward pass
out = lazy_attention_triton(q, k, v, bias, tau, window_size=window_size)
print(f"\nOutput shape: {out.shape}")

# Backward pass
loss = out.sum()
loss.backward()

# Check which heads have gradients
print(f"\n{'='*80}")
print("Gradient Check by Head:")
print(f"{'='*80}")

all_heads_ok = True
for h in range(H):
    has_bias = (bias.grad[h] != 0).any().item()
    has_tau = (tau.grad[h] != 0).item()
    status = '✅' if (has_bias and has_tau) else '❌'
    print(f"Head {h}: bias_grad={'✅' if has_bias else '❌'}, tau_grad={'✅' if has_tau else '❌'} {status}")

    if not (has_bias and has_tau):
        all_heads_ok = False

print(f"\n{'='*80}")
if all_heads_ok:
    print("✅ SUCCESS: All heads have gradients!")
else:
    print("❌ FAILURE: Some heads are missing gradients!")
    print("\nThis indicates a bug in the Triton backward kernel.")
    print("The kernel is not properly computing gradients for all heads.")
print(f"{'='*80}")

# Show actual gradient values
print(f"\nbias.grad:")
print(bias.grad)
print(f"\ntau.grad:")
print(tau.grad)
