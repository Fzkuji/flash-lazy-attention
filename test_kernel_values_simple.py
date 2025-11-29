#!/usr/bin/env python3
"""Simple test to check kernel intermediate values without excessive printing"""
import torch
import triton
import triton.language as tl

@triton.jit
def extract_values_kernel(
    Q, K, V, Bias, Tau, LSE, DO,
    # Output tensors for debugging
    Out_q_sum, Out_k_sum, Out_bias_mean, Out_lse_mean,
    Out_s_mean, Out_p_norm_mean, Out_relu_count,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_bh, stride_bw,
    stride_lseb, stride_om, stride_ok,
    stride_qb, stride_kb, stride_vb,
    n_heads, seq_len, window_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr
):
    """Extract intermediate values for debugging"""
    m_block_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)

    # Only process first block
    if m_block_idx != 0 or b_idx != 0:
        return

    # Load tau
    tau = tl.load(Tau + h_idx)

    # Load Q for this head
    offs_m = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)

    Q_ptr = Q + b_idx * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(Q_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
    q_sum = tl.sum(q)

    # Load LSE
    LSE_ptr = LSE + b_idx * stride_lseb + h_idx * seq_len + offs_m
    lse = tl.load(LSE_ptr, mask=offs_m < seq_len, other=0.0)
    lse_mean = tl.sum(lse) / BLOCK_M

    # Load K for first block
    offs_n = tl.arange(0, BLOCK_N)
    K_ptr_base = K + b_idx * stride_kb + h_idx * stride_kh
    K_ptr = K_ptr_base + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
    k = tl.load(K_ptr, mask=offs_n[None, :] < seq_len, other=0.0)
    k_sum = tl.sum(k)

    # Compute attention scores
    sm_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
    s = tl.dot(q, k) * sm_scale
    s_mean_raw = tl.sum(s) / (BLOCK_M * BLOCK_N)

    # Load bias
    dist = offs_m[:, None] - offs_n[None, :]
    dist_clamped = tl.minimum(dist, window_size - 1)
    dist_clamped = tl.maximum(dist_clamped, 0)
    valid_mask = (dist >= 0)

    Bias_ptr_base = Bias + h_idx * stride_bh
    bias_ptrs = Bias_ptr_base + dist_clamped * stride_bw
    bias_val = tl.load(bias_ptrs, mask=valid_mask, other=0.0)
    bias_mean = tl.sum(bias_val) / tl.sum(valid_mask.to(tl.float32))

    # Add bias to scores
    s += bias_val
    s = tl.where(dist >= 0, s, float("-inf"))
    s_with_bias_mean = tl.sum(tl.where(valid_mask, s, 0.0)) / tl.sum(valid_mask.to(tl.float32))

    # Compute p_norm
    p_norm = tl.exp(s - lse[:, None])
    p_norm_mean = tl.sum(tl.where(valid_mask, p_norm, 0.0)) / tl.sum(valid_mask.to(tl.float32))

    # Compute p_term and mask_relu
    idx_i = offs_m + 1
    tau_term = tau / idx_i
    p_term = p_norm + tau_term[:, None]
    mask_relu = p_term > 0
    relu_count = tl.sum(mask_relu.to(tl.int32))

    # Store results
    tl.store(Out_q_sum + h_idx, q_sum)
    tl.store(Out_k_sum + h_idx, k_sum)
    tl.store(Out_bias_mean + h_idx, bias_mean)
    tl.store(Out_lse_mean + h_idx, lse_mean)
    tl.store(Out_s_mean + h_idx, s_with_bias_mean)
    tl.store(Out_p_norm_mean + h_idx, p_norm_mean)
    tl.store(Out_relu_count + h_idx, relu_count.to(tl.float32))


def test_simple_debug():
    """Test with cleaner output"""
    print("="*80)
    print("Kernel Values Debug Test (Clean Output)")
    print("="*80)

    B, H, L, D = 1, 4, 32, 16
    window_size = 16
    BLOCK_M = 16
    BLOCK_N = 16

    # Create test inputs
    q = torch.randn(B, H, L, D, device='cuda')
    k = torch.randn(B, H, L, D, device='cuda')
    v = torch.randn(B, H, L, D, device='cuda')
    bias = torch.randn(H, window_size, device='cuda')
    tau = torch.full((H,), 0.0, device='cuda')  # Use tau=0

    # Create LSE (zeros for simplicity - in real case this comes from forward)
    lse = torch.zeros(B, H, L, device='cuda')

    # Create DO
    do = torch.randn(B, H, L, D, device='cuda')

    # Create output tensors for debug values
    out_q_sum = torch.zeros(H, device='cuda')
    out_k_sum = torch.zeros(H, device='cuda')
    out_bias_mean = torch.zeros(H, device='cuda')
    out_lse_mean = torch.zeros(H, device='cuda')
    out_s_mean = torch.zeros(H, device='cuda')
    out_p_norm_mean = torch.zeros(H, device='cuda')
    out_relu_count = torch.zeros(H, device='cuda')

    print("\nLaunching kernel to extract values...")

    grid = (1, H, B)  # Only first block

    extract_values_kernel[grid](
        q, k, v, bias, tau, lse, do,
        out_q_sum, out_k_sum, out_bias_mean, out_lse_mean,
        out_s_mean, out_p_norm_mean, out_relu_count,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        bias.stride(0), bias.stride(1),
        lse.stride(0), do.stride(2), do.stride(3),
        q.stride(0), k.stride(0), v.stride(0),
        H, L, window_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D
    )

    # Print results
    print("\n" + "="*80)
    print("Results by Head:")
    print("="*80)
    print(f"\n{'Head':<6} {'q_sum':<12} {'k_sum':<12} {'bias_mean':<12} {'lse_mean':<12}")
    print("-" * 80)
    for h in range(H):
        print(f"{h:<6} {out_q_sum[h].item():<12.4f} {out_k_sum[h].item():<12.4f} "
              f"{out_bias_mean[h].item():<12.4f} {out_lse_mean[h].item():<12.4f}")

    print(f"\n{'Head':<6} {'s_mean':<12} {'p_norm_mean':<12} {'relu_count':<12}")
    print("-" * 80)
    for h in range(H):
        print(f"{h:<6} {out_s_mean[h].item():<12.4f} {out_p_norm_mean[h].item():<12.6f} "
              f"{int(out_relu_count[h].item()):<12}")

    # Analysis
    print("\n" + "="*80)
    print("Analysis:")
    print("="*80)

    # Check if values are consistent across heads
    q_sums_equal = torch.allclose(out_q_sum, out_q_sum[0].expand(H), rtol=1e-3)
    k_sums_equal = torch.allclose(out_k_sum, out_k_sum[0].expand(H), rtol=1e-3)
    lse_equal = torch.allclose(out_lse_mean, out_lse_mean[0].expand(H), rtol=1e-3)

    if not q_sums_equal:
        print("⚠️  WARNING: q_sum differs between heads - Q loading might be wrong!")
    if not k_sums_equal:
        print("⚠️  WARNING: k_sum differs between heads - K loading might be wrong!")
    if not lse_equal:
        print("⚠️  WARNING: lse_mean differs between heads - LSE loading might be wrong!")

    # Check relu counts
    has_relu = out_relu_count > 0
    if not has_relu.all():
        print(f"\n🔴 CRITICAL: Some heads have relu_count=0!")
        for h in range(H):
            if out_relu_count[h] == 0:
                print(f"   Head {h}: NO RELU activations - this causes zero gradients!")
    else:
        print(f"\n✅ All heads have positive relu_count")

    # Check p_norm
    p_norm_zero = out_p_norm_mean < 1e-10
    if p_norm_zero.any():
        print(f"\n🔴 CRITICAL: Some heads have p_norm ≈ 0!")
        for h in range(H):
            if p_norm_zero[h]:
                print(f"   Head {h}: p_norm_mean={out_p_norm_mean[h].item():.2e}")


if __name__ == "__main__":
    test_simple_debug()
