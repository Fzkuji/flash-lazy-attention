#!/usr/bin/env python3
"""Minimal test for backward gradient computation across heads"""
import torch
import triton
import triton.language as tl

@triton.jit
def minimal_bwd_kernel(
    DBias,  # [H, W] - output
    DTau,  # [H] - output
    DO,  # [B, H, L, D] - input gradient
    V,  # [B, H, L, D] - input
    stride_doh, stride_dol, stride_dod,
    stride_vh, stride_vl, stride_vd,
    stride_bh, stride_bw,
    stride_dob, stride_vb,
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    BLOCK: tl.constexpr
):
    """Simplified backward kernel - just accumulate gradients"""
    h_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    # Load a block of DO and V
    offs_l = tl.arange(0, BLOCK)
    offs_d = tl.arange(0, D)

    DO_ptr = DO + b_idx * stride_dob + h_idx * stride_doh + offs_l[:, None] * stride_dol + offs_d[None, :] * stride_dod
    V_ptr = V + b_idx * stride_vb + h_idx * stride_vh + offs_l[:, None] * stride_vl + offs_d[None, :] * stride_vd

    mask = offs_l[:, None] < L
    do_block = tl.load(DO_ptr, mask=mask, other=0.0)
    v_block = tl.load(V_ptr, mask=mask, other=0.0)

    # Compute gradient (simplified)
    dp = tl.dot(do_block, tl.trans(v_block))  # [BLOCK, BLOCK]

    # dBias: accumulate to different positions based on distance
    for i in range(BLOCK):
        if offs_l[i] < L:
            for j in range(BLOCK):
                if offs_l[j] < L and offs_l[i] >= offs_l[j]:
                    dist = offs_l[i] - offs_l[j]
                    if dist < W:
                        val = dp[i, j]
                        tl.atomic_add(DBias + h_idx * stride_bh + dist * stride_bw, val)

    # dTau: simple sum
    dtau_val = tl.sum(dp)
    tl.atomic_add(DTau + h_idx, dtau_val)


def test_minimal_backward():
    """Test minimal backward pass"""
    print("="*80)
    print("Testing Minimal Backward Kernel")
    print("="*80)

    B, H, L, D, W = 1, 4, 8, 16, 16
    BLOCK = 8

    # Create inputs
    do = torch.randn(B, H, L, D, device='cuda')
    v = torch.randn(B, H, L, D, device='cuda')

    # Create outputs
    dbias = torch.zeros(H, W, device='cuda')
    dtau = torch.zeros(H, device='cuda')

    print(f"\nInput shapes:")
    print(f"  do: {do.shape}")
    print(f"  v: {v.shape}")
    print(f"  dbias: {dbias.shape}")
    print(f"  dtau: {dtau.shape}")

    # Launch kernel
    grid = (H, B)
    minimal_bwd_kernel[grid](
        dbias, dtau,
        do, v,
        do.stride(1), do.stride(2), do.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        dbias.stride(0), dbias.stride(1),
        do.stride(0), v.stride(0),
        B=B, H=H, L=L, D=D, W=W,
        BLOCK=BLOCK
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
