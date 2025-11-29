#!/usr/bin/env python3
"""Test if preprocess kernel computes delta correctly for all heads"""
import torch
from adasplash.lazy_attention_triton import _lazy_attention_forward_return_lse, _lazy_bwd_preprocess_kernel

def test_delta_computation():
    """Test delta computation in preprocess kernel"""
    print("="*80)
    print("Testing Delta Computation in Preprocess Kernel")
    print("="*80)

    B, H, L, D = 1, 4, 8, 16
    window_size = 16
    BLOCK_M = 64
    BLOCK_N = 64

    # Create inputs
    q = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
    bias = torch.randn(H, window_size, device='cuda', dtype=torch.bfloat16)
    tau = torch.full((H,), -1.0, device='cuda', dtype=torch.bfloat16)

    print("\nRunning forward pass to get LSE...")
    out, lse = _lazy_attention_forward_return_lse(q, k, v, bias, tau, window_size, varlen=None)

    # Create DO
    do = torch.randn_like(out)

    # Create delta output
    delta = torch.empty_like(lse)

    print("Running preprocess kernel to compute delta...")

    grid_m = (1, H, B)  # Only 1 block since L=8 < BLOCK_M=64

    _lazy_bwd_preprocess_kernel[grid_m](
        q, k, v, bias, tau, lse, do, delta, q,  # dummy varlen
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        bias.stride(0), bias.stride(1),
        lse.stride(0), do.stride(0), do.stride(2), do.stride(3),
        q.stride(0), k.stride(0), v.stride(0), delta.stride(0),
        H, L, window_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_VARLEN=False
    )

    # Check delta values
    print("\n" + "="*80)
    print("Delta Statistics by Head:")
    print("="*80)
    print(f"\n{'Head':<6} {'mean':<12} {'std':<12} {'min':<12} {'max':<12} {'has_values':<12}")
    print("-" * 80)

    all_ok = True
    for h in range(H):
        delta_h = delta[0, h, :]
        has_values = (delta_h != 0).any().item()
        status = '✅' if has_values else '❌'

        print(f"{h:<6} {delta_h.mean().item():<12.6f} {delta_h.std().item():<12.6f} "
              f"{delta_h.min().item():<12.6f} {delta_h.max().item():<12.6f} {status}")

        if not has_values:
            all_ok = False

    print("\n" + "="*80)
    print("Analysis:")
    print("="*80)

    if not all_ok:
        print("\n🔴 FOUND THE BUG!")
        print("   Some heads have delta=0 everywhere!")
        print("   This would cause ds=0 in backward kernel, leading to zero gradients!")
        for h in range(H):
            if (delta[0, h, :] == 0).all():
                print(f"\n   Head {h}: delta is ALL ZEROS")
                print(f"     → ds = p_norm * (term - delta) = p_norm * (term - 0)")
                print(f"     → If term is also wrong, ds could be 0")
    else:
        print("\n✅ All heads have non-zero delta values")
        print("   Delta computation in preprocess kernel is correct.")
        print("   The bug must be somewhere else...")

    # Also print delta values for inspection
    print(f"\nFull delta tensor:")
    print(delta)

    return all_ok


if __name__ == "__main__":
    test_delta_computation()
