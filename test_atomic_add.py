#!/usr/bin/env python3
"""Test atomic_add behavior across different heads"""
import torch
import triton
import triton.language as tl

@triton.jit
def test_atomic_kernel(
    Output,  # [H]
    stride_h,
    H: tl.constexpr
):
    h_idx = tl.program_id(0)

    # Simple test: each head should add 1.0
    tl.atomic_add(Output + h_idx * stride_h, 1.0)

def test_atomic_add_multi_head():
    """Test if atomic_add works for all heads"""
    H = 4
    output = torch.zeros(H, device='cuda')

    print(f"Before: {output}")
    print(f"output.stride(0) = {output.stride(0)}")

    grid = (H,)
    test_atomic_kernel[grid](
        output,
        output.stride(0),
        H=H
    )

    print(f"After: {output}")

    # Check if all heads got the value
    expected = torch.ones(H, device='cuda')
    if torch.allclose(output, expected):
        print("✅ SUCCESS: atomic_add works for all heads")
    else:
        print("❌ FAILURE: atomic_add doesn't work for all heads")
        for h in range(H):
            status = '✅' if output[h] == 1.0 else '❌'
            print(f"  Head {h}: {output[h].item()} {status}")

if __name__ == "__main__":
    test_atomic_add_multi_head()
