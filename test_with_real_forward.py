#!/usr/bin/env python3
"""Test backward kernel with REAL LSE from forward pass"""
import torch
import triton
import triton.language as tl
from adasplash.lazy_attention_triton import _lazy_attention_forward_return_lse

@triton.jit
def extract_values_with_real_lse_kernel(
    Q, K, V, Bias, Tau, LSE, DO,
    # Output tensors for debugging
    Out_lse_mean, Out_p_norm_mean, Out_relu_count,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_bh, stride_bw,
    stride_lseb, stride_om,
    stride_qb, stride_kb,
    n_heads, seq_len, window_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr
):
    """Extract values using REAL LSE from forward pass"""
    m_block_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)

    # Only process first block
    if m_block_idx != 0 or b_idx != 0:
        return

    tau = tl.load(Tau + h_idx)

    # Load Q for this head
    offs_m = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)

    Q_ptr = Q + b_idx * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(Q_ptr, mask=offs_m[:, None] < seq_len, other=0.0)

    # Load LSE - THIS IS THE REAL LSE FROM FORWARD!
    LSE_ptr = LSE + b_idx * stride_lseb + h_idx * seq_len + offs_m
    lse = tl.load(LSE_ptr, mask=offs_m < seq_len, other=0.0)
    lse_mean = tl.sum(lse) / BLOCK_M

    # Load K for first block
    offs_n = tl.arange(0, BLOCK_N)
    K_ptr_base = K + b_idx * stride_kb + h_idx * stride_kh
    K_ptr = K_ptr_base + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
    k = tl.load(K_ptr, mask=offs_n[None, :] < seq_len, other=0.0)

    # Compute attention scores
    sm_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
    s = tl.dot(q, k) * sm_scale

    # Load bias
    dist = offs_m[:, None] - offs_n[None, :]
    dist_clamped = tl.minimum(dist, window_size - 1)
    dist_clamped = tl.maximum(dist_clamped, 0)
    valid_mask = (dist >= 0)

    Bias_ptr_base = Bias + h_idx * stride_bh
    bias_ptrs = Bias_ptr_base + dist_clamped * stride_bw
    bias_val = tl.load(bias_ptrs, mask=valid_mask, other=0.0)

    # Add bias to scores
    in_window = (dist >= 0) & (dist < window_size)
    bias_val = tl.where(in_window, bias_val, 0.0)
    s += bias_val
    s = tl.where(dist >= 0, s, float("-inf"))

    # Compute p_norm - THIS USES REAL LSE!
    p_norm = tl.exp(s - lse[:, None])
    p_norm_mean = tl.sum(tl.where(valid_mask, p_norm, 0.0)) / tl.sum(valid_mask.to(tl.float32))

    # Compute p_term and mask_relu
    idx_i = offs_m + 1
    tau_term = tau / idx_i
    p_term = p_norm + tau_term[:, None]
    mask_relu = p_term > 0
    relu_count = tl.sum(mask_relu.to(tl.int32))

    # Store results
    tl.store(Out_lse_mean + h_idx, lse_mean)
    tl.store(Out_p_norm_mean + h_idx, p_norm_mean)
    tl.store(Out_relu_count + h_idx, relu_count.to(tl.float32))


def test_with_real_forward():
    """Test using REAL LSE from forward pass"""
    print("="*80)
    print("Test with REAL LSE from Forward Pass")
    print("="*80)

    B, H, L, D = 1, 4, 32, 16
    window_size = 16
    BLOCK_M = 16
    BLOCK_N = 16

    # Create test inputs - same as in actual test
    q = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
    bias = torch.randn(H, window_size, device='cuda', dtype=torch.bfloat16)
    tau = torch.full((H,), -1.0, device='cuda', dtype=torch.bfloat16)

    print("\nRunning forward pass to get REAL LSE...")
    # Run actual forward pass to get LSE
    out, lse = _lazy_attention_forward_return_lse(q, k, v, bias, tau, window_size, varlen=None)

    print(f"Forward pass complete. LSE shape: {lse.shape}")
    print(f"\nLSE statistics by head:")
    for h in range(H):
        lse_h = lse[0, h, :]
        print(f"  Head {h}: mean={lse_h.mean().item():.4f}, std={lse_h.std().item():.4f}, "
              f"min={lse_h.min().item():.4f}, max={lse_h.max().item():.4f}")

    # Create DO
    do = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)

    # Create output tensors for debug values
    out_lse_mean = torch.zeros(H, device='cuda')
    out_p_norm_mean = torch.zeros(H, device='cuda')
    out_relu_count = torch.zeros(H, device='cuda')

    print("\nExtracting values with REAL LSE...")

    grid = (1, H, B)  # Only first block

    extract_values_with_real_lse_kernel[grid](
        q, k, v, bias, tau, lse, do,
        out_lse_mean, out_p_norm_mean, out_relu_count,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        bias.stride(0), bias.stride(1),
        lse.stride(0), do.stride(2),
        q.stride(0), k.stride(0),
        H, L, window_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D
    )

    # Print results
    print("\n" + "="*80)
    print("Results with REAL LSE:")
    print("="*80)
    print(f"\n{'Head':<6} {'lse_mean':<15} {'p_norm_mean':<15} {'relu_count':<12}")
    print("-" * 80)
    for h in range(H):
        print(f"{h:<6} {out_lse_mean[h].item():<15.4f} {out_p_norm_mean[h].item():<15.6f} "
              f"{int(out_relu_count[h].item()):<12}")

    # Analysis
    print("\n" + "="*80)
    print("CRITICAL Analysis:")
    print("="*80)

    # Check relu counts
    has_relu = out_relu_count > 0
    if not has_relu.all():
        print(f"\n🔴 FOUND THE BUG!")
        for h in range(H):
            if out_relu_count[h] == 0:
                print(f"   Head {h}: relu_count=0 with REAL LSE!")
                print(f"   → lse_mean={out_lse_mean[h].item():.4f}")
                print(f"   → p_norm_mean={out_p_norm_mean[h].item():.6f}")
                print(f"   → This explains why gradients are zero for this head!")
    else:
        print(f"\n✅ All heads have positive relu_count even with REAL LSE")
        print(f"   → The bug is NOT in LSE computation!")
        print(f"   → The bug must be elsewhere in the backward kernel")

    # Check p_norm values
    print(f"\nP_norm analysis:")
    for h in range(H):
        if out_p_norm_mean[h] < 1e-6:
            print(f"  🔴 Head {h}: p_norm ≈ 0 (={out_p_norm_mean[h].item():.2e})")
        else:
            print(f"  ✅ Head {h}: p_norm = {out_p_norm_mean[h].item():.6f}")


if __name__ == "__main__":
    test_with_real_forward()
