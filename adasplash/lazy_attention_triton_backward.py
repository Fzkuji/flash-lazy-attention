import torch
import triton
import triton.language as tl

# ... (Existing Forward Kernels) ...

@triton.jit
def _lazy_bwd_preprocess_kernel(
    Q, K, V, Bias, Tau, LSE, DO, Delta,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_bh, stride_bw,
    stride_lseb, stride_dob, stride_om, stride_ok, # DO strides
    stride_qb, stride_kb, stride_vb, stride_deltab,
    n_heads, seq_len, window_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr
):
    m_block_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    
    offs_m = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    
    # Pointers
    Q_ptr = Q + b_idx * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    K_ptr_base = K + b_idx * stride_kb + h_idx * stride_kh
    V_ptr_base = V + b_idx * stride_vb + h_idx * stride_vh
    Bias_ptr_base = Bias + h_idx * stride_bh
    DO_ptr = DO + b_idx * stride_dob + h_idx * stride_qh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok # Reuse stride_qh/qm/qk for DO if same layout
    LSE_ptr = LSE + b_idx * stride_lseb + h_idx * seq_len + offs_m
    Delta_ptr = Delta + b_idx * stride_deltab + h_idx * seq_len + offs_m
    
    # Load Q, LSE, DO, Tau
    q = tl.load(Q_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
    lse = tl.load(LSE_ptr, mask=offs_m < seq_len, other=0.0)
    do = tl.load(DO_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
    tau = tl.load(Tau + h_idx)
    
    delta = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    n_end = (m_block_idx + 1) * BLOCK_M
    for n_start in range(0, n_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        
        # Load K, V
        K_ptr = K_ptr_base + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
        V_ptr = V_ptr_base + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        k = tl.load(K_ptr, mask=offs_n[None, :] < seq_len, other=0.0)
        v = tl.load(V_ptr, mask=offs_n[:, None] < seq_len, other=0.0)
        
        # Recompute Score
        sm_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
        s = tl.dot(q, k) * sm_scale
        
        dist = offs_m[:, None] - offs_n[None, :]
        in_window = (dist >= 0) & (dist <= window_size)
        valid_mask = (dist >= 0)
        dist_clamped = tl.minimum(dist, window_size)
        dist_clamped = tl.maximum(dist_clamped, 0)
        bias_ptrs = Bias_ptr_base + dist_clamped * stride_bw
        bias_val = tl.load(bias_ptrs, mask=valid_mask, other=0.0)
        bias_val = tl.where(in_window, bias_val, 0.0)
        s += bias_val
        s = tl.where(dist >= 0, s, float("-inf"))
        
        # Recompute P
        p_norm = tl.exp(s - lse[:, None])
        idx_i = offs_m + 1
        tau_term = tau / idx_i
        p_term = p_norm + tau_term[:, None]
        mask_relu = p_term > 0
        
        # dP_elastic = DO dot V^T
        dp_elastic = tl.dot(do, tl.trans(v))
        
        # Delta calculation: sum(P_norm * dP_elastic * mask)
        term = p_norm * dp_elastic
        term = tl.where(mask_relu, term, 0.0)
        delta += tl.sum(term, 1)

    tl.store(Delta_ptr, delta, mask=offs_m < seq_len)

@triton.jit
def _lazy_bwd_kernel_dq(
    Q, K, V, Bias, Tau, LSE, DO, Delta,
    DQ, DBias, DTau,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_bh, stride_bw,
    stride_lseb, stride_dob, stride_om, stride_ok,
    stride_qb, stride_kb, stride_vb, stride_deltab,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    n_heads, seq_len, window_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr
):
    # Compute dQ
    # Also accumulate dBias and dTau using atomic adds (simplification)
    m_block_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    
    offs_m = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    
    # Pointers
    Q_ptr = Q + b_idx * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    K_ptr_base = K + b_idx * stride_kb + h_idx * stride_kh
    V_ptr_base = V + b_idx * stride_vb + h_idx * stride_vh
    Bias_ptr_base = Bias + h_idx * stride_bh
    DO_ptr = DO + b_idx * stride_dob + h_idx * stride_qh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    LSE_ptr = LSE + b_idx * stride_lseb + h_idx * seq_len + offs_m
    Delta_ptr = Delta + b_idx * stride_deltab + h_idx * seq_len + offs_m
    DQ_ptr = DQ + b_idx * stride_dqb + h_idx * stride_dqh + offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk
    
    q = tl.load(Q_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
    lse = tl.load(LSE_ptr, mask=offs_m < seq_len, other=0.0)
    delta = tl.load(Delta_ptr, mask=offs_m < seq_len, other=0.0)
    do = tl.load(DO_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
    tau = tl.load(Tau + h_idx)
    
    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    dtau_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    n_end = (m_block_idx + 1) * BLOCK_M
    for n_start in range(0, n_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        
        K_ptr = K_ptr_base + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
        V_ptr = V_ptr_base + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        k = tl.load(K_ptr, mask=offs_n[None, :] < seq_len, other=0.0)
        v = tl.load(V_ptr, mask=offs_n[:, None] < seq_len, other=0.0)
        
        sm_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
        s = tl.dot(q, k) * sm_scale
        
        dist = offs_m[:, None] - offs_n[None, :]
        in_window = (dist >= 0) & (dist <= window_size)
        valid_mask = (dist >= 0)
        dist_clamped = tl.minimum(dist, window_size)
        dist_clamped = tl.maximum(dist_clamped, 0)
        
        bias_ptrs = Bias_ptr_base + dist_clamped * stride_bw
        bias_val = tl.load(bias_ptrs, mask=valid_mask, other=0.0)
        bias_val = tl.where(in_window, bias_val, 0.0)
        s += bias_val
        s = tl.where(dist >= 0, s, float("-inf"))
        
        p_norm = tl.exp(s - lse[:, None])
        idx_i = offs_m + 1
        tau_term = tau / idx_i
        p_term = p_norm + tau_term[:, None]
        mask_relu = p_term > 0
        
        dp_elastic = tl.dot(do, tl.trans(v))
        
        # dP_norm = dP_elastic * mask
        # dS = P_norm * (dP_norm - Delta)
        #    = P_norm * (dP_elastic * mask - Delta)
        # If mask is 0, dP_norm is 0. 
        # Wait, formula for dS: P_norm * (dP_norm_effective - sum(...))
        # Here dP_norm_effective coming from ReLU is dP_elastic if mask else 0.
        
        ds = p_norm * (tl.where(mask_relu, dp_elastic, 0.0) - delta[:, None])
        
        # dQ = dS * K
        dq += tl.dot(ds.to(k.dtype), tl.trans(k)) * sm_scale
        
        # dBias = dS (if in window)
        # We use atomic add to global memory
        dbias_val = tl.where(in_window, ds, 0.0)
        tl.atomic_add(DBias + h_idx * stride_bh + dist_clamped * stride_bw, dbias_val, mask=valid_mask)
        
        # dTau
        # dL/dTau = sum( dP_elastic * mask * (1/i) )
        # dP_elastic is dp_elastic
        term_tau = dp_elastic * (1.0 / idx_i[:, None])
        term_tau = tl.where(mask_relu, term_tau, 0.0)
        dtau_acc += tl.sum(term_tau, 1)

    # Atomic add dTau
    tl.atomic_add(DTau + h_idx, tl.sum(dtau_acc))
    
    # Store dQ
    tl.store(DQ_ptr, dq.to(DQ.dtype.element_ty), mask=offs_m[:, None] < seq_len)

# Need separate kernel for DK/DV or implement in same pass?
# FlashAttention usually does separate pass. 
# For Task 1 MVP, we can use a simpler approach: 
# _lazy_bwd_kernel_dk_dv iterating over K/V blocks (outer loop N), inner loop M.

@triton.jit
def _lazy_bwd_kernel_dk_dv(
    Q, K, V, Bias, Tau, LSE, DO, Delta,
    DK, DV,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_bh, stride_bw,
    stride_lseb, stride_dob, stride_om, stride_ok,
    stride_qb, stride_kb, stride_vb, stride_deltab,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_dvb, stride_dvh, stride_dvn, stride_dvk,
    n_heads, seq_len, window_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr
):
    # Compute dK, dV
    # Grid: (seq_len // BLOCK_N, n_heads, batch)
    n_block_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    
    offs_n = n_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    
    # Pointers
    K_ptr = K + b_idx * stride_kb + h_idx * stride_kh + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    V_ptr = V + b_idx * stride_vb + h_idx * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
    
    DK_ptr = DK + b_idx * stride_dkb + h_idx * stride_dkh + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk
    DV_ptr = DV + b_idx * stride_dvb + h_idx * stride_dvh + offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk
    
    k = tl.load(K_ptr, mask=offs_n[:, None] < seq_len, other=0.0)
    v = tl.load(V_ptr, mask=offs_n[:, None] < seq_len, other=0.0)
    tau = tl.load(Tau + h_idx)
    
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    
    # Loop over M (Queries)
    # We need to iterate M such that i >= j (causal). 
    # i = offs_m, j = offs_n.
    # Start M block where offs_m max >= offs_n min.
    m_start_block = (n_block_idx * BLOCK_N) // BLOCK_M
    
    num_m_blocks = tl.cdiv(seq_len, BLOCK_M)
    
    for m_block in range(m_start_block, num_m_blocks):
        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        
        Q_ptr = Q + b_idx * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        DO_ptr = DO + b_idx * stride_dob + h_idx * stride_qh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
        LSE_ptr = LSE + b_idx * stride_lseb + h_idx * seq_len + offs_m
        Delta_ptr = Delta + b_idx * stride_deltab + h_idx * seq_len + offs_m
        Bias_ptr_base = Bias + h_idx * stride_bh
        
        q = tl.load(Q_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
        do = tl.load(DO_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
        lse = tl.load(LSE_ptr, mask=offs_m < seq_len, other=0.0)
        delta = tl.load(Delta_ptr, mask=offs_m < seq_len, other=0.0)
        
        # Recompute Score (Transposed view: we are iterating M for fixed N)
        # s[m, n] = q[m] . k[n]
        sm_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
        s = tl.dot(q, tl.trans(k)) * sm_scale
        
        dist = offs_m[:, None] - offs_n[None, :]
        in_window = (dist >= 0) & (dist <= window_size)
        valid_mask = (dist >= 0)
        dist_clamped = tl.minimum(dist, window_size)
        dist_clamped = tl.maximum(dist_clamped, 0)
        
        bias_ptrs = Bias_ptr_base + dist_clamped * stride_bw
        bias_val = tl.load(bias_ptrs, mask=valid_mask, other=0.0)
        bias_val = tl.where(in_window, bias_val, 0.0)
        s += bias_val
        s = tl.where(dist >= 0, s, float("-inf"))
        
        p_norm = tl.exp(s - lse[:, None])
        idx_i = offs_m + 1
        tau_term = tau / idx_i
        p_term = p_norm + tau_term[:, None]
        mask_relu = p_term > 0
        
        # dV += P_elastic^T * DO
        # P_elastic [M, N], DO [M, D] -> [N, D]
        p_elastic = tl.maximum(p_term, 0.0)
        dv += tl.dot(tl.trans(p_elastic.to(do.dtype)), do)
        
        # dS calculation for dK
        # dS = P_norm * (dP_elastic*mask - Delta)
        # dP_elastic = DO . V^T
        # Here we need dP_elastic[m, n]
        # But we don't have full V here to compute dot product efficiently?
        # Wait, dP_elastic_mn = DO_m . V_n
        # We have V_n (it's `v` loaded outside loop). DO_m is loaded.
        # So we can compute dP_elastic elementwise? No, dot product.
        # dP_elastic[m, n] = sum_d (DO[m, d] * V[n, d])
        
        dp_elastic = tl.dot(do, tl.trans(v))
        
        ds = p_norm * (tl.where(mask_relu, dp_elastic, 0.0) - delta[:, None])
        
        # dK += dS^T * Q
        # dS [M, N], Q [M, D] -> [N, D]
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q) * sm_scale
        
    tl.store(DK_ptr, dk.to(DK.dtype.element_ty), mask=offs_n[:, None] < seq_len)
    tl.store(DV_ptr, dv.to(DV.dtype.element_ty), mask=offs_n[:, None] < seq_len)

