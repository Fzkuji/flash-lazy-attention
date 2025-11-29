#!/usr/bin/env python3
"""Test if tau and bias parameters can be trained"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from adasplash import lazy_attention_triton

print("="*80)
print("Testing if tau and bias parameters can be updated")
print("="*80)

B, H, L, D = 1, 4, 8, 16
window_size = 16

# Create test inputs
q = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
k = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
v = torch.randn(B, H, L, D, device='cuda', requires_grad=True)

# Create parameters
bias = torch.randn(H, window_size, device='cuda', requires_grad=True)
tau = torch.full((H,), -1.0, device='cuda', requires_grad=True)

print(f"\nInitial values:")
print(f"  bias[0, 0] = {bias[0, 0].item():.6f}")
print(f"  tau = {tau.tolist()}")

# Forward + backward
out = lazy_attention_triton(q, k, v, bias, tau)
loss = out.sum()
loss.backward()

print(f"\nGradients computed:")
print(f"  bias.grad[0, 0] = {bias.grad[0, 0].item():.6f}")
print(f"  tau.grad = {tau.grad.tolist()}")

# Simulate optimizer step
lr = 0.01
with torch.no_grad():
    bias -= lr * bias.grad
    tau -= lr * tau.grad

print(f"\nAfter optimizer step (lr={lr}):")
print(f"  bias[0, 0] = {bias[0, 0].item():.6f}")
print(f"  tau = {tau.tolist()}")

print(f"\n{'='*80}")
if all(tau[i].item() != -1.0 for i in range(H)):
    print("✅ SUCCESS: tau values changed after training step!")
else:
    print("❌ FAILURE: Some tau values still -1.0")
    for i in range(H):
        if tau[i].item() == -1.0:
            print(f"  Head {i}: tau={tau[i].item()}, grad={tau.grad[i].item()}")
print(f"{'='*80}")
