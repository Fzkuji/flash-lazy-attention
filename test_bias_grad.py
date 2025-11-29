#!/usr/bin/env python3
"""Test script to verify bias gradient computation"""
import torch
from adasplash import lazy_attention_triton

# Set random seed for reproducibility
torch.manual_seed(42)

# Small test case
B, H, L, D = 1, 2, 8, 16
window_size = 4

# Create inputs
q = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
k = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
v = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
bias = torch.randn(H, window_size, device='cuda', requires_grad=True)
tau = torch.full((H,), -1.0, device='cuda', requires_grad=True)

print(f"Input shapes:")
print(f"  q: {q.shape}")
print(f"  k: {k.shape}")
print(f"  v: {v.shape}")
print(f"  bias: {bias.shape}")
print(f"  tau: {tau.shape}")
print(f"  window_size: {window_size}")

# Forward pass
out = lazy_attention_triton(q, k, v, bias, tau, window_size=window_size)
print(f"\nOutput shape: {out.shape}")

# Backward pass
loss = out.sum()
loss.backward()

# Check gradients
print("\n" + "="*60)
print("Gradient Check:")
print("="*60)

print(f"\nq.grad is not None: {q.grad is not None}")
if q.grad is not None:
    print(f"  q.grad shape: {q.grad.shape}")
    print(f"  q.grad mean: {q.grad.mean().item():.6f}")
    print(f"  q.grad std: {q.grad.std().item():.6f}")
    print(f"  q.grad abs max: {q.grad.abs().max().item():.6f}")

print(f"\nk.grad is not None: {k.grad is not None}")
if k.grad is not None:
    print(f"  k.grad shape: {k.grad.shape}")
    print(f"  k.grad mean: {k.grad.mean().item():.6f}")
    print(f"  k.grad std: {k.grad.std().item():.6f}")
    print(f"  k.grad abs max: {k.grad.abs().max().item():.6f}")

print(f"\nv.grad is not None: {v.grad is not None}")
if v.grad is not None:
    print(f"  v.grad shape: {v.grad.shape}")
    print(f"  v.grad mean: {v.grad.mean().item():.6f}")
    print(f"  v.grad std: {v.grad.std().item():.6f}")
    print(f"  v.grad abs max: {v.grad.abs().max().item():.6f}")

print(f"\nbias.grad is not None: {bias.grad is not None}")
if bias.grad is not None:
    print(f"  bias.grad shape: {bias.grad.shape}")
    print(f"  bias.grad mean: {bias.grad.mean().item():.6f}")
    print(f"  bias.grad std: {bias.grad.std().item():.6f}")
    print(f"  bias.grad abs max: {bias.grad.abs().max().item():.6f}")
    print(f"  bias.grad non-zero count: {(bias.grad != 0).sum().item()} / {bias.grad.numel()}")
    print(f"\n  bias.grad values:")
    print(bias.grad)
else:
    print("  WARNING: bias.grad is None!")

print(f"\ntau.grad is not None: {tau.grad is not None}")
if tau.grad is not None:
    print(f"  tau.grad shape: {tau.grad.shape}")
    print(f"  tau.grad mean: {tau.grad.mean().item():.6f}")
    print(f"  tau.grad std: {tau.grad.std().item():.6f}")
    print(f"  tau.grad abs max: {tau.grad.abs().max().item():.6f}")

print("\n" + "="*60)
print("Test completed!")
print("="*60)
