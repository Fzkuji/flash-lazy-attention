#!/usr/bin/env python3
"""Test actual backward pass with same setup as test_head_gradients.py"""
import torch
from adasplash import lazy_attention_triton

print("="*80)
print("Testing ACTUAL Backward Pass (Same as test_head_gradients.py)")
print("="*80)

B, H, L, D = 1, 4, 8, 16
window_size = 16

# Create test inputs - EXACT same as test_head_gradients.py
q = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
k = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
v = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
bias = torch.randn(H, window_size, device='cuda', requires_grad=True)
tau = torch.full((H,), -1.0, device='cuda', requires_grad=True)

print(f"\nInput shapes:")
print(f"  q: {q.shape}, dtype: {q.dtype}")
print(f"  k: {k.shape}, dtype: {k.dtype}")
print(f"  v: {v.shape}, dtype: {v.dtype}")
print(f"  bias: {bias.shape}, dtype: {bias.dtype}")
print(f"  tau: {tau.shape}, dtype: {tau.dtype}")

print(f"\nTensor strides:")
print(f"  q.stride: {q.stride()}")
print(f"  k.stride: {k.stride()}")
print(f"  v.stride: {v.stride()}")
print(f"  bias.stride: {bias.stride()}")
print(f"  tau.stride: {tau.stride()}")

# Forward pass
print("\nRunning forward pass...")
out = lazy_attention_triton(q, k, v, bias, tau, window_size=window_size)
print(f"Output shape: {out.shape}, dtype: {out.dtype}")

# Backward pass
print("Running backward pass...")
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
    print("\n🔴 BUG CONFIRMED: Only head 0 has gradients")
print(f"{'='*80}")

# Show gradient details
print(f"\nDetailed gradient values:")
print(f"\nbias.grad:")
for h in range(H):
    grad_sum = bias.grad[h].sum().item()
    grad_nonzero = (bias.grad[h] != 0).sum().item()
    print(f"  Head {h}: sum={grad_sum:10.4f}, nonzero_count={grad_nonzero}/16")

print(f"\ntau.grad:")
for h in range(H):
    print(f"  Head {h}: {tau.grad[h].item():10.4f}")

# Additional debug: Check if gradients for q, k, v are correct
print(f"\n{'='*80}")
print("Q/K/V Gradient Check:")
print(f"{'='*80}")
for h in range(H):
    q_has_grad = (q.grad[0, h] != 0).any().item()
    k_has_grad = (k.grad[0, h] != 0).any().item()
    v_has_grad = (v.grad[0, h] != 0).any().item()
    status = '✅' if (q_has_grad and k_has_grad and v_has_grad) else '❌'
    print(f"Head {h}: dq={'✅' if q_has_grad else '❌'}, "
          f"dk={'✅' if k_has_grad else '❌'}, "
          f"dv={'✅' if v_has_grad else '❌'} {status}")
