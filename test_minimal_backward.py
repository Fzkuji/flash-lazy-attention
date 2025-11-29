#!/usr/bin/env python3
"""Minimal test for backward gradient computation across heads"""
import torch
import triton
import triton.language as tl

@triton.jit
def minimal_bwd_kernel(
    DBias,  # [H, W] - output
    DTau,  # [H] - output
    stride_bh, stride_bw,
    H: tl.constexpr,
    W: tl.constexpr
):
    """Ultra-simplified kernel - just test atomic_add with grid parallelization"""
    h_idx = tl.program_id(0)
    l_idx = tl.program_id(1)  # Simulate sequence position

    # Simple pattern: each (h, l) thread adds 1.0 to different bias positions
    for w in range(min(W, l_idx + 1)):  # Add to positions 0..min(W, l_idx)
        tl.atomic_add(DBias + h_idx * stride_bh + w * stride_bw, 1.0)

    # Each thread adds 0.1 to its head's tau
    tl.atomic_add(DTau + h_idx, 0.1)


def test_minimal_backward():
    """Test minimal backward pass"""
    print("="*80)
    print("Testing Minimal Backward Kernel")
    print("="*80)

    H, L, W = 4, 8, 16

    # Create outputs
    dbias = torch.zeros(H, W, device='cuda')
    dtau = torch.zeros(H, device='cuda')

    print(f"\nOutput shapes:")
    print(f"  dbias: {dbias.shape}")
    print(f"  dtau: {dtau.shape}")

    # Launch kernel with grid (H, L) to simulate backward pass parallelization
    # Each (h_idx, l_idx) thread will add gradients
    grid = (H, L)
    minimal_bwd_kernel[grid](
        dbias, dtau,
        dbias.stride(0), dbias.stride(1),
        H=H, W=W
    )

    # Check results
    print(f"\nResults:")
    all_ok = True
    for h in range(H):
        has_bias = (dbias[h] != 0).any().item()
        has_tau = (dtau[h] != 0).item()
        status = '✅' if (has_bias and has_tau) else '❌'
        print(f"  Head {h}: dbias_sum={dbias[h].sum().item():.4f}, dtau={dtau[h].item():.4f} {status}")
        if not (has_bias and has_tau):
            all_ok = False

    print(f"\n{'='*80}")
    if all_ok:
        print("✅ SUCCESS: All heads have gradients")
    else:
        print("❌ FAILURE: Some heads missing gradients")
    print(f"{'='*80}\n")

    return all_ok


if __name__ == "__main__":
    test_minimal_backward()
