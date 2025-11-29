#!/usr/bin/env python3
"""Direct test - load the actual file to ensure fix is used"""
import sys
import os

# Force import from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import
from lazy_attention_triton import lazy_attention_triton
import torch

print("="*80)
print("Direct Fix Test - Loading from LOCAL file")
print("="*80)

B, H, L, D = 1, 4, 8, 16
window_size = 16

# Create test inputs
q = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
k = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
v = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
bias = torch.randn(H, window_size, device='cuda', requires_grad=True)
tau = torch.full((H,), -1.0, device='cuda', requires_grad=True)

print("\nRunning forward...")
out = lazy_attention_triton(q, k, v, bias, tau, window_size=window_size)

print("Running backward...")
loss = out.sum()
loss.backward()

# Check gradients
print(f"\n{'='*80}")
print("Gradient Results:")
print(f"{'='*80}")

all_ok = True
for h in range(H):
    has_bias = (bias.grad[h] != 0).any().item()
    has_tau = (tau.grad[h] != 0).item()
    has_q = (q.grad[0, h] != 0).any().item()

    status = '✅' if (has_bias and has_tau and has_q) else '❌'
    print(f"Head {h}: bias={'✅' if has_bias else '❌'}, "
          f"tau={'✅' if has_tau else '❌'}, "
          f"q={'✅' if has_q else '❌'} {status}")

    if not (has_bias and has_tau):
        all_ok = False

print(f"\n{'='*80}")
if all_ok:
    print("✅ SUCCESS: All heads have gradients!")
    print("The fix WORKS!")
else:
    print("❌ FAILURE: Some heads missing gradients")
    print("The fix did NOT work or was not applied")
print(f"{'='*80}")
