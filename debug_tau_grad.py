#!/usr/bin/env python3
"""Debug script to diagnose tau gradient issues"""
import torch
from adasplash import lazy_attention_triton

# Test with different precisions
torch.manual_seed(42)

B, H, L, D = 2, 4, 32, 64
window_size = 512

print("="*80)
print("Testing tau gradient with different dtypes")
print("="*80)

for dtype_name, dtype in [("float32", torch.float32), ("bfloat16", torch.bfloat16)]:
    print(f"\n{'='*80}")
    print(f"Testing with dtype: {dtype_name}")
    print(f"{'='*80}")

    # Create inputs
    q = torch.randn(B, H, L, D, device='cuda', dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, L, D, device='cuda', dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, L, D, device='cuda', dtype=dtype, requires_grad=True)
    bias = torch.randn(H, window_size, device='cuda', dtype=dtype, requires_grad=True)
    tau = torch.full((H,), -1.0, device='cuda', dtype=dtype, requires_grad=True)

    print(f"Initial tau: {tau.detach().cpu().numpy()}")

    # Forward pass
    out = lazy_attention_triton(q, k, v, bias, tau, window_size=window_size)

    # Backward pass
    loss = out.sum()
    loss.backward()

    # Check tau gradient
    print(f"\ntau.grad exists: {tau.grad is not None}")
    if tau.grad is not None:
        print(f"tau.grad dtype: {tau.grad.dtype}")
        print(f"tau.grad values: {tau.grad.detach().cpu().numpy()}")
        print(f"tau.grad mean: {tau.grad.mean().item():.10f}")
        print(f"tau.grad abs mean: {tau.grad.abs().mean().item():.10f}")
        print(f"tau.grad abs max: {tau.grad.abs().max().item():.10f}")
        print(f"tau.grad non-zero: {(tau.grad != 0).sum().item()} / {tau.grad.numel()}")

        # Simulate an optimizer step
        lr = 0.001
        with torch.no_grad():
            tau_new = tau - lr * tau.grad
        print(f"\nAfter simulated optimizer step (lr={lr}):")
        print(f"tau_new values: {tau_new.detach().cpu().numpy()}")
        print(f"tau change: {(tau_new - tau).abs().mean().item():.10f}")

    print(f"\nother gradients:")
    print(f"  q.grad abs mean: {q.grad.abs().mean().item():.10f}")
    print(f"  k.grad abs mean: {k.grad.abs().mean().item():.10f}")
    print(f"  v.grad abs mean: {v.grad.abs().mean().item():.10f}")
    print(f"  bias.grad abs mean: {bias.grad.abs().mean().item():.10f}")

print(f"\n{'='*80}")
print("Diagnosis Summary:")
print(f"{'='*80}")
print("\nIf tau.grad is zero or very small with bfloat16:")
print("  → Problem: bfloat16 precision causes gradient underflow")
print("  → Solution: Use float32 for tau parameter or increase gradient scale")
print("\nIf tau.grad is zero with both dtypes:")
print("  → Problem: Bug in backward kernel")
print("  → Check mask_relu logic and dtau computation")
