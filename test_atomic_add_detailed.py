#!/usr/bin/env python3
"""Test atomic_add with 2D indexing like in the backward kernel"""
import torch
import triton
import triton.language as tl

@triton.jit
def test_dbias_atomic_kernel(
    DBias,  # [H, W]
    stride_bh,
    stride_bw,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    h_idx = tl.program_id(0)

    # Simulate adding values to different positions in the bias window
    # Similar to: tl.atomic_add(DBias + h_idx * stride_bh + dist_clamped * stride_bw, dbias_val)
    offs = tl.arange(0, BLOCK_SIZE)
    dist_clamped = offs % W  # Wrap around window size

    # Create some test values
    values = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)

    # Atomic add with 2D indexing
    tl.atomic_add(DBias + h_idx * stride_bh + dist_clamped * stride_bw, values)


@triton.jit
def test_dtau_atomic_kernel(
    DTau,  # [H]
    H: tl.constexpr
):
    h_idx = tl.program_id(0)

    # Simulate: tl.atomic_add(DTau + h_idx, tl.sum(dtau_acc))
    value = 1.0
    tl.atomic_add(DTau + h_idx, value)


def test_bias_atomic():
    """Test atomic_add for bias gradient (2D indexing)"""
    H, W = 4, 16
    BLOCK_SIZE = 8

    dbias = torch.zeros(H, W, device='cuda')
    print(f"\n{'='*80}")
    print(f"Testing dBias atomic_add (shape=[{H}, {W}])")
    print(f"{'='*80}")
    print(f"Before: all zeros")
    print(f"stride_bh={dbias.stride(0)}, stride_bw={dbias.stride(1)}")

    grid = (H,)
    test_dbias_atomic_kernel[grid](
        dbias,
        dbias.stride(0),
        dbias.stride(1),
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE
    )

    print(f"\nAfter atomic_add:")
    for h in range(H):
        has_grad = (dbias[h] != 0).any().item()
        status = '✅' if has_grad else '❌'
        print(f"  Head {h}: sum={dbias[h].sum().item():.1f}, has_nonzero={has_grad} {status}")

    all_heads_ok = all((dbias[h] != 0).any() for h in range(H))
    print(f"\n{'='*80}")
    if all_heads_ok:
        print("✅ SUCCESS: dBias atomic_add works for all heads")
    else:
        print("❌ FAILURE: dBias atomic_add doesn't work for all heads")
    print(f"{'='*80}\n")

    return all_heads_ok


def test_tau_atomic():
    """Test atomic_add for tau gradient (1D indexing)"""
    H = 4

    dtau = torch.zeros(H, device='cuda')
    print(f"\n{'='*80}")
    print(f"Testing dTau atomic_add (shape=[{H}])")
    print(f"{'='*80}")
    print(f"Before: {dtau}")

    grid = (H,)
    test_dtau_atomic_kernel[grid](
        dtau,
        H=H
    )

    print(f"After: {dtau}")

    all_heads_ok = all(dtau[h] != 0 for h in range(H))
    print(f"\n{'='*80}")
    if all_heads_ok:
        print("✅ SUCCESS: dTau atomic_add works for all heads")
    else:
        print("❌ FAILURE: dTau atomic_add doesn't work for all heads")
        for h in range(H):
            status = '✅' if dtau[h] != 0 else '❌'
            print(f"  Head {h}: {dtau[h].item()} {status}")
    print(f"{'='*80}\n")

    return all_heads_ok


if __name__ == "__main__":
    bias_ok = test_bias_atomic()
    tau_ok = test_tau_atomic()

    print(f"\n{'='*80}")
    print("FINAL RESULTS:")
    print(f"  dBias: {'✅ PASS' if bias_ok else '❌ FAIL'}")
    print(f"  dTau:  {'✅ PASS' if tau_ok else '❌ FAIL'}")
    print(f"{'='*80}")
