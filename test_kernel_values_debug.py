#!/usr/bin/env python3
"""Debug kernel values to find where gradients become zero for h_idx>0"""
import torch
import triton
import triton.language as tl

@triton.jit
def debug_bwd_kernel(
    Q, K, V, Bias, Tau, LSE, DO,
    DBias, DTau,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_bh, stride_bw,
    stride_lseb, stride_om, stride_ok,
    stride_qb, stride_kb, stride_vb,
    n_heads, seq_len, window_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr
):
    """Debug version with extensive printing"""
    m_block_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)

    # Load tau first (outside if to avoid duplicate loads)
    tau = tl.load(Tau + h_idx)

    # Only print for first block to reduce output
    # Use a single print per head by checking offs_m[0] == 0
    offs_m = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)

    if m_block_idx == 0 and b_idx == 0 and offs_m[0] == 0:
        tl.device_print("=== Head", h_idx, "===")

        # Load Q for this head
        offs_m = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, HEAD_DIM)

        Q_ptr = Q + b_idx * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        q = tl.load(Q_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
        q_sum = tl.sum(q)
        tl.device_print("q_sum:", q_sum)

        # Load LSE
        LSE_ptr = LSE + b_idx * stride_lseb + h_idx * seq_len + offs_m
        lse = tl.load(LSE_ptr, mask=offs_m < seq_len, other=0.0)
        lse_mean = tl.sum(lse) / BLOCK_M
        tl.device_print("lse_mean:", lse_mean)

        # Load K for first block
        offs_n = tl.arange(0, BLOCK_N)
        K_ptr_base = K + b_idx * stride_kb + h_idx * stride_kh
        K_ptr = K_ptr_base + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
        k = tl.load(K_ptr, mask=offs_n[None, :] < seq_len, other=0.0)
        k_sum = tl.sum(k)
        tl.device_print("k_sum:", k_sum)

        # Compute attention scores
        sm_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
        s = tl.dot(q, k) * sm_scale
        s_mean = tl.sum(s) / (BLOCK_M * BLOCK_N)
        tl.device_print("s_mean (raw QK score):", s_mean)

        # Load bias
        dist = offs_m[:, None] - offs_n[None, :]
        dist_clamped = tl.minimum(dist, window_size - 1)
        dist_clamped = tl.maximum(dist_clamped, 0)
        valid_mask = (dist >= 0)

        Bias_ptr_base = Bias + h_idx * stride_bh
        bias_ptrs = Bias_ptr_base + dist_clamped * stride_bw
        bias_val = tl.load(bias_ptrs, mask=valid_mask, other=0.0)
        bias_mean = tl.sum(bias_val) / tl.sum(valid_mask.to(tl.float32))
        tl.device_print("bias_mean:", bias_mean)

        # Add bias to scores
        s += bias_val
        s = tl.where(dist >= 0, s, float("-inf"))
        s_with_bias_mean = tl.sum(tl.where(valid_mask, s, 0.0)) / tl.sum(valid_mask.to(tl.float32))
        tl.device_print("s_mean (after bias):", s_with_bias_mean)

        # Compute p_norm
        p_norm = tl.exp(s - lse[:, None])
        p_norm_mean = tl.sum(tl.where(valid_mask, p_norm, 0.0)) / tl.sum(valid_mask.to(tl.float32))
        tl.device_print("p_norm_mean:", p_norm_mean)

        # Compute p_term
        idx_i = offs_m + 1
        tau_term = tau / idx_i
        p_term = p_norm + tau_term[:, None]

        # Check mask_relu
        mask_relu = p_term > 0
        relu_count = tl.sum(mask_relu.to(tl.int32))
        tl.device_print("relu_count (positions > 0):", relu_count)

        if relu_count > 0:
            tl.device_print("mask_relu has True values")
        else:
            tl.device_print("mask_relu is ALL FALSE - this is the problem!")

        # Check p_term values
        p_term_min = tl.min(tl.where(valid_mask, p_term, 1e10))
        p_term_max = tl.max(tl.where(valid_mask, p_term, -1e10))
        tl.device_print("p_term_min:", p_term_min)
        tl.device_print("p_term_max:", p_term_max)


def test_debug_kernel():
    """Test with debug kernel"""
    print("="*80)
    print("Debug Kernel Values Test")
    print("="*80)

    B, H, L, D = 1, 4, 32, 16  # Increased L to 32 for larger blocks
    window_size = 16
    BLOCK_M = 16  # Must be ≥16 for tl.dot
    BLOCK_N = 16  # Must be ≥16 for tl.dot

    # Create test inputs
    q = torch.randn(B, H, L, D, device='cuda')
    k = torch.randn(B, H, L, D, device='cuda')
    v = torch.randn(B, H, L, D, device='cuda')
    bias = torch.randn(H, window_size, device='cuda')
    tau = torch.full((H,), 0.0, device='cuda')  # Use tau=0 to isolate

    # Compute LSE (simplified - just use zeros for now)
    lse = torch.zeros(B, H, L, device='cuda')

    # Create DO
    do = torch.randn(B, H, L, D, device='cuda')

    # Create outputs
    dbias = torch.zeros(H, window_size, device='cuda')
    dtau = torch.zeros(H, device='cuda')

    print("\nLaunching debug kernel...")
    print("This will print values for each head to identify where the issue occurs.\n")

    grid = (1, H, B)  # Only first block

    debug_bwd_kernel[grid](
        q, k, v, bias, tau, lse, do,
        dbias, dtau,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        bias.stride(0), bias.stride(1),
        lse.stride(0), do.stride(2), do.stride(3),
        q.stride(0), k.stride(0), v.stride(0),
        H, L, window_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D
    )

    print("\n" + "="*80)
    print("Check the debug output above to see where values differ between heads")
    print("="*80)


if __name__ == "__main__":
    test_debug_kernel()
