#!/usr/bin/env python3
"""Test if mask_relu is False for heads 1,2,3 due to bias/tau values"""
import torch
from adasplash import lazy_attention_triton

print("="*80)
print("Testing mask_relu Hypothesis")
print("="*80)

B, H, L, D = 1, 4, 8, 16
window_size = 16

# Create test inputs - use same setup as test_head_gradients.py
q = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
k = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
v = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
bias = torch.randn(H, window_size, device='cuda', requires_grad=True)
tau = torch.full((H,), -1.0, device='cuda', requires_grad=True)

print(f"\nInitial values:")
print(f"tau: {tau}")
print(f"\nbias values per head:")
for h in range(H):
    print(f"  Head {h}: mean={bias[h].mean():.4f}, std={bias[h].std():.4f}, "
          f"min={bias[h].min():.4f}, max={bias[h].max():.4f}")

# Forward pass
print(f"\nRunning forward pass...")
out = lazy_attention_triton(q, k, v, bias, tau, window_size=window_size)

# Backward pass
print(f"Running backward pass...")
loss = out.sum()
loss.backward()

# Check gradients
print(f"\n{'='*80}")
print("Gradient Results:")
print(f"{'='*80}")
for h in range(H):
    has_bias = (bias.grad[h] != 0).any().item()
    has_tau = (tau.grad[h] != 0).item()
    status = '✅' if (has_bias and has_tau) else '❌'

    bias_sum = bias.grad[h].sum().item() if has_bias else 0.0
    tau_val = tau.grad[h].item() if has_tau else 0.0

    print(f"Head {h}: bias_grad_sum={bias_sum:8.4f}, tau_grad={tau_val:8.4f} {status}")

print(f"\n{'='*80}")
print("Analysis:")
print(f"{'='*80}")

# Check if the issue is related to bias magnitudes
print("\nHypothesis: Heads with larger negative bias → smaller p_norm → p_term<0 → no gradients")
print("If this is true, we should see correlation between bias values and gradient presence.\n")

# Let's also test with all tau=0 to see if the issue persists
print("\n" + "="*80)
print("Testing with tau=0 (no elastic term):")
print("="*80)

q2 = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
k2 = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
v2 = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
bias2 = bias.detach().clone().requires_grad_(True)
tau2 = torch.full((H,), 0.0, device='cuda', requires_grad=True)  # tau=0 instead of -1

out2 = lazy_attention_triton(q2, k2, v2, bias2, tau2, window_size=window_size)
loss2 = out2.sum()
loss2.backward()

print(f"\nWith tau=0:")
for h in range(H):
    has_bias = (bias2.grad[h] != 0).any().item()
    has_tau = (tau2.grad[h] != 0).item()
    status = '✅' if has_bias else '❌'
    print(f"Head {h}: bias_grad={'✅' if has_bias else '❌'}, tau_grad={'✅' if has_tau else '❌'} {status}")
