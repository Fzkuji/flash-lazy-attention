import torch
import triton
import triton.language as tl

# ==========================================
# Forward Kernels
# ==========================================

@triton.jit
def _get_lse_kernel_batch(
    Q, K, Bias, LSE, VARLEN,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_bh, stride_bw,
    stride_qb, stride_kb, stride_lseb, # Batch strides
    n_heads, seq_len_max,
    window_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    m_block_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    
    # Get actual sequence length
    if IS_VARLEN:
        seq_len = tl.load(VARLEN + b_idx).to(tl.int32)
    else:
        seq_len = seq_len_max

    offs_m = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    
    # Batch Offsets
    Q_ptr = Q + b_idx * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    K_ptr_base = K + b_idx * stride_kb + h_idx * stride_kh
    Bias_ptr_base = Bias + h_idx * stride_bh 
    LSE_ptr = LSE + b_idx * stride_lseb + h_idx * seq_len_max + offs_m
    
    # Load Q
    q = tl.load(Q_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
    
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    
    # Loop bounds
    n_end = (m_block_idx + 1) * BLOCK_M
    if IS_VARLEN:
        n_end = tl.minimum(n_end, seq_len)
        
    for n_start in range(0, n_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        
        # Load K
        K_ptr = K_ptr_base + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
        k = tl.load(K_ptr, mask=offs_n[None, :] < seq_len, other=0.0)
        
        sm_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
        s = tl.dot(q, k) * sm_scale
        
        dist = offs_m[:, None] - offs_n[None, :]
        in_window = (dist >= 0) & (dist < window_size)

        dist_clamped = tl.minimum(dist, window_size - 1)
        dist_clamped = tl.maximum(dist_clamped, 0)
        
        bias_ptrs = Bias_ptr_base + dist_clamped * stride_bw
        
        valid_mask = (dist >= 0)
        bias_val = tl.load(bias_ptrs, mask=valid_mask, other=0.0)
        bias_val = tl.where(in_window, bias_val, 0.0)
        
        s += bias_val
        s = tl.where(dist >= 0, s, float("-inf"))
        
        # Apply varlen mask to scores
        if IS_VARLEN:
            s = tl.where(offs_n[None, :] < seq_len, s, float("-inf"))
        
        m_block = tl.max(s, 1)
        new_m_i = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - new_m_i)
        l_i = l_i * alpha + tl.sum(tl.exp(s - new_m_i[:, None]), 1)
        m_i = new_m_i

    # Compute final LSE
    lse = m_i + tl.log(l_i)
    tl.store(LSE_ptr, lse, mask=offs_m < seq_len)

@triton.jit
def _lazy_fwd_kernel_batch(
    Q, K, V, Bias, Tau, LSE, Out, VARLEN,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_oh, stride_om, stride_ok,
    stride_bh, stride_bw,
    stride_qb, stride_kb, stride_vb, stride_ob, stride_lseb,
    n_heads, seq_len_max,
    window_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    m_block_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    
    if IS_VARLEN:
        seq_len = tl.load(VARLEN + b_idx).to(tl.int32)
    else:
        seq_len = seq_len_max
    
    offs_m = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    
    Q_ptr = Q + b_idx * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    K_ptr_base = K + b_idx * stride_kb + h_idx * stride_kh
    V_ptr_base = V + b_idx * stride_vb + h_idx * stride_vh
    Bias_ptr_base = Bias + h_idx * stride_bh
    Out_ptr = Out + b_idx * stride_ob + h_idx * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    LSE_ptr = LSE + b_idx * stride_lseb + h_idx * seq_len_max + offs_m
    
    lse = tl.load(LSE_ptr, mask=offs_m < seq_len, other=0.0)
    tau = tl.load(Tau + h_idx)

    q = tl.load(Q_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
    
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    n_end = (m_block_idx + 1) * BLOCK_M
    if IS_VARLEN:
        n_end = tl.minimum(n_end, seq_len)
        
    for n_start in range(0, n_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        
        K_ptr = K_ptr_base + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
        k = tl.load(K_ptr, mask=offs_n[None, :] < seq_len, other=0.0)
        
        sm_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
        s = tl.dot(q, k) * sm_scale
        
        dist = offs_m[:, None] - offs_n[None, :]
        in_window = (dist >= 0) & (dist < window_size)
        valid_mask = (dist >= 0)
        dist_clamped = tl.minimum(dist, window_size - 1)
        dist_clamped = tl.maximum(dist_clamped, 0)
        
        bias_ptrs = Bias_ptr_base + dist_clamped * stride_bw
        bias_val = tl.load(bias_ptrs, mask=valid_mask, other=0.0)
        bias_val = tl.where(in_window, bias_val, 0.0)
        s += bias_val
        s = tl.where(dist >= 0, s, float("-inf"))
        
        if IS_VARLEN:
            s = tl.where(offs_n[None, :] < seq_len, s, float("-inf"))
        
        p_norm = tl.exp(s - lse[:, None])
        idx_i = offs_m + 1
        idx_i_float = idx_i.to(tl.float32)
        tau_term = tau / idx_i_float
        # 用 float32 计算 ReLU 避免 bf16 精度问题（tau=-1.0 时 p_term 接近 0）
        p_norm_f32 = p_norm.to(tl.float32)
        p_term_f32 = p_norm_f32 + tau_term[:, None]
        p_elastic = tl.maximum(p_term_f32, 0.0).to(p_norm.dtype)
        # Re-apply causal mask: when tau > 0, upper triangle would have tau/i > 0
        # This prevents information leakage from future tokens
        p_elastic = tl.where(valid_mask, p_elastic, 0.0)

        V_ptr = V_ptr_base + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(V_ptr, mask=offs_n[:, None] < seq_len, other=0.0)
        acc += tl.dot(p_elastic.to(v.dtype), v)

    tl.store(Out_ptr, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seq_len)

# ==========================================
# Backward Kernels
# ==========================================

@triton.jit
def _lazy_bwd_preprocess_kernel(
    Q, K, V, Bias, Tau, LSE, DO, Delta, VARLEN,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_bh, stride_bw,
    stride_lseb, stride_dob, stride_doh, stride_om, stride_ok,
    stride_qb, stride_kb, stride_vb, stride_deltab,
    n_heads, seq_len_max, window_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    m_block_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)

    # Load actual sequence length
    if IS_VARLEN:
        seq_len = tl.load(VARLEN + b_idx).to(tl.int32)
    else:
        seq_len = seq_len_max

    offs_m = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)

    # Pointers
    Q_ptr = Q + b_idx * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    K_ptr_base = K + b_idx * stride_kb + h_idx * stride_kh
    V_ptr_base = V + b_idx * stride_vb + h_idx * stride_vh
    Bias_ptr_base = Bias + h_idx * stride_bh
    DO_ptr = DO + b_idx * stride_dob + h_idx * stride_doh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    LSE_ptr = LSE + b_idx * stride_lseb + h_idx * seq_len_max + offs_m
    Delta_ptr = Delta + b_idx * stride_deltab + h_idx * seq_len_max + offs_m

    q = tl.load(Q_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
    lse = tl.load(LSE_ptr, mask=offs_m < seq_len, other=0.0)
    do = tl.load(DO_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
    tau = tl.load(Tau + h_idx)
    
    delta = tl.zeros([BLOCK_M], dtype=tl.float32)

    n_end = (m_block_idx + 1) * BLOCK_M
    if IS_VARLEN:
        n_end = tl.minimum(n_end, seq_len)

    for n_start in range(0, n_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        
        K_ptr = K_ptr_base + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
        V_ptr = V_ptr_base + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        k = tl.load(K_ptr, mask=offs_n[None, :] < seq_len, other=0.0)
        v = tl.load(V_ptr, mask=offs_n[:, None] < seq_len, other=0.0)
        
        sm_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
        s = tl.dot(q, k) * sm_scale
        
        dist = offs_m[:, None] - offs_n[None, :]
        in_window = (dist >= 0) & (dist < window_size)
        valid_mask = (dist >= 0)
        dist_clamped = tl.minimum(dist, window_size - 1)
        dist_clamped = tl.maximum(dist_clamped, 0)
        
        bias_ptrs = Bias_ptr_base + dist_clamped * stride_bw
        bias_val = tl.load(bias_ptrs, mask=valid_mask, other=0.0)
        bias_val = tl.where(in_window, bias_val, 0.0)
        s += bias_val
        s = tl.where(dist >= 0, s, float("-inf"))

        # Apply varlen mask
        if IS_VARLEN:
            s = tl.where(offs_n[None, :] < seq_len, s, float("-inf"))

        p_norm = tl.exp(s - lse[:, None])
        idx_i = offs_m + 1
        idx_i_float = idx_i.to(tl.float32)
        tau_term = tau / idx_i_float
        p_term = p_norm + tau_term[:, None]
        # 用 float32 计算 mask_relu 避免 bf16 精度问题
        p_norm_f32 = p_norm.to(tl.float32)
        p_term_f32 = p_norm_f32 + tau_term[:, None]
        # mask_relu also needs causal mask: when tau > 0, upper triangle would pass p_term > 0
        mask_relu = (p_term_f32 > 0) & valid_mask

        dp_elastic = tl.dot(do, tl.trans(v))

        term = p_norm * dp_elastic
        term = tl.where(mask_relu, term, 0.0)
        delta += tl.sum(term, 1)

    tl.store(Delta_ptr, delta, mask=offs_m < seq_len)

@triton.jit
def _lazy_bwd_kernel_dq(
    Q, K, V, Bias, Tau, LSE, DO, Delta, VARLEN,
    DQ, DBias_blocks, DTau_blocks,  # Per-block buffers to eliminate atomic contention
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_bh, stride_bw,
    stride_lseb, stride_dob, stride_doh, stride_om, stride_ok,
    stride_qb, stride_kb, stride_vb, stride_deltab,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_dbias_blk, stride_dbias_w,  # Strides for per-block dbias buffer
    n_heads, seq_len_max, window_size, num_m_blocks,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    COMPUTE_LAZY_GRAD: tl.constexpr = True  # Skip dbias/dtau when False (frozen params)
):
    """
    Optimized dQ kernel with per-block dBias/dTau buffers.

    Key optimization: Each thread block writes to its own buffer slice,
    eliminating cross-block atomic contention. Final reduction is done in PyTorch.
    """
    m_block_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)

    # Compute global block index for per-block buffer access
    global_block_idx = b_idx * n_heads * num_m_blocks + h_idx * num_m_blocks + m_block_idx

    # Load actual sequence length
    if IS_VARLEN:
        seq_len = tl.load(VARLEN + b_idx).to(tl.int32)
    else:
        seq_len = seq_len_max

    offs_m = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)

    # Pointers
    Q_ptr = Q + b_idx * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    K_ptr_base = K + b_idx * stride_kb + h_idx * stride_kh
    V_ptr_base = V + b_idx * stride_vb + h_idx * stride_vh
    Bias_ptr_base = Bias + h_idx * stride_bh
    DO_ptr = DO + b_idx * stride_dob + h_idx * stride_doh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    LSE_ptr = LSE + b_idx * stride_lseb + h_idx * seq_len_max + offs_m
    Delta_ptr = Delta + b_idx * stride_deltab + h_idx * seq_len_max + offs_m
    DQ_ptr = DQ + b_idx * stride_dqb + h_idx * stride_dqh + offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk

    # Per-block buffer pointers (no cross-block contention!)
    DBias_block_ptr = DBias_blocks + global_block_idx * stride_dbias_blk
    DTau_block_ptr = DTau_blocks + global_block_idx

    q = tl.load(Q_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
    lse = tl.load(LSE_ptr, mask=offs_m < seq_len, other=0.0)
    delta = tl.load(Delta_ptr, mask=offs_m < seq_len, other=0.0)
    do = tl.load(DO_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
    tau = tl.load(Tau + h_idx)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    dtau_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    n_end = (m_block_idx + 1) * BLOCK_M
    if IS_VARLEN:
        n_end = tl.minimum(n_end, seq_len)

    sm_scale = 1.0 / tl.sqrt(float(HEAD_DIM))

    for n_start in range(0, n_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)

        K_ptr = K_ptr_base + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
        V_ptr = V_ptr_base + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        k = tl.load(K_ptr, mask=offs_n[None, :] < seq_len, other=0.0)
        v = tl.load(V_ptr, mask=offs_n[:, None] < seq_len, other=0.0)

        s = tl.dot(q, k) * sm_scale

        dist = offs_m[:, None] - offs_n[None, :]
        in_window = (dist >= 0) & (dist < window_size)
        valid_mask = (dist >= 0)
        dist_clamped = tl.minimum(dist, window_size - 1)
        dist_clamped = tl.maximum(dist_clamped, 0)

        bias_ptrs = Bias_ptr_base + dist_clamped * stride_bw
        bias_val = tl.load(bias_ptrs, mask=valid_mask, other=0.0)
        bias_val = tl.where(in_window, bias_val, 0.0)
        s += bias_val
        s = tl.where(dist >= 0, s, float("-inf"))

        # Apply varlen mask
        if IS_VARLEN:
            s = tl.where(offs_n[None, :] < seq_len, s, float("-inf"))

        p_norm = tl.exp(s - lse[:, None])
        idx_i = offs_m + 1
        idx_i_float = idx_i.to(tl.float32)
        tau_term = tau / idx_i_float
        # 用 float32 计算 mask_relu 避免 bf16 精度问题
        p_norm_f32 = p_norm.to(tl.float32)
        p_term_f32 = p_norm_f32 + tau_term[:, None]
        # mask_relu also needs causal mask: when tau > 0, upper triangle would pass p_term > 0
        mask_relu = (p_term_f32 > 0) & valid_mask

        dp_elastic = tl.dot(do, tl.trans(v))
        ds = p_norm * (tl.where(mask_relu, dp_elastic, 0.0) - delta[:, None])

        dq += tl.dot(ds.to(k.dtype), tl.trans(k)) * sm_scale

        # dBias/dTau - skip when frozen (COMPUTE_LAZY_GRAD=False)
        if COMPUTE_LAZY_GRAD:
            # dBias - write to per-block buffer (no cross-block atomic contention!)
            dbias_ptrs = DBias_block_ptr + dist_clamped * stride_dbias_w
            dbias_val = tl.where(in_window, ds, 0.0)
            tl.atomic_add(dbias_ptrs, dbias_val, mask=valid_mask)

            # dTau
            term_tau = dp_elastic * (1.0 / idx_i_float[:, None])
            term_tau = tl.where(mask_relu, term_tau, 0.0)
            dtau_acc += tl.sum(term_tau, 1)

    # Store results
    if COMPUTE_LAZY_GRAD:
        tl.store(DTau_block_ptr, tl.sum(dtau_acc))
    tl.store(DQ_ptr, dq.to(DQ.dtype.element_ty), mask=offs_m[:, None] < seq_len)

@triton.jit
def _lazy_bwd_kernel_dk_dv(
    Q, K, V, Bias, Tau, LSE, DO, Delta, VARLEN,
    DK, DV,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_bh, stride_bw,
    stride_lseb, stride_dob, stride_doh, stride_om, stride_ok,
    stride_qb, stride_kb, stride_vb, stride_deltab,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_dvb, stride_dvh, stride_dvn, stride_dvk,
    n_heads, seq_len_max, window_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    n_block_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)

    # Load actual sequence length
    if IS_VARLEN:
        seq_len = tl.load(VARLEN + b_idx).to(tl.int32)
    else:
        seq_len = seq_len_max

    offs_n = n_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    K_ptr = K + b_idx * stride_kb + h_idx * stride_kh + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    V_ptr = V + b_idx * stride_vb + h_idx * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
    DK_ptr = DK + b_idx * stride_dkb + h_idx * stride_dkh + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk
    DV_ptr = DV + b_idx * stride_dvb + h_idx * stride_dvh + offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk

    k = tl.load(K_ptr, mask=offs_n[:, None] < seq_len, other=0.0)
    v = tl.load(V_ptr, mask=offs_n[:, None] < seq_len, other=0.0)
    tau = tl.load(Tau + h_idx)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # Start M block where offs_m max >= offs_n min.
    m_start_block = (n_block_idx * BLOCK_N) // BLOCK_M
    num_m_blocks = tl.cdiv(seq_len, BLOCK_M)
    
    for m_block in range(m_start_block, num_m_blocks):
        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)

        Q_ptr = Q + b_idx * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        DO_ptr = DO + b_idx * stride_dob + h_idx * stride_doh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
        LSE_ptr = LSE + b_idx * stride_lseb + h_idx * seq_len_max + offs_m
        Delta_ptr = Delta + b_idx * stride_deltab + h_idx * seq_len_max + offs_m
        Bias_ptr_base = Bias + h_idx * stride_bh

        q = tl.load(Q_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
        do = tl.load(DO_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
        lse = tl.load(LSE_ptr, mask=offs_m < seq_len, other=0.0)
        delta = tl.load(Delta_ptr, mask=offs_m < seq_len, other=0.0)

        sm_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
        s = tl.dot(q, tl.trans(k)) * sm_scale

        dist = offs_m[:, None] - offs_n[None, :]
        in_window = (dist >= 0) & (dist < window_size)
        valid_mask = (dist >= 0)
        dist_clamped = tl.minimum(dist, window_size - 1)
        dist_clamped = tl.maximum(dist_clamped, 0)

        bias_ptrs = Bias_ptr_base + dist_clamped * stride_bw
        bias_val = tl.load(bias_ptrs, mask=valid_mask, other=0.0)
        bias_val = tl.where(in_window, bias_val, 0.0)
        s += bias_val
        s = tl.where(dist >= 0, s, float("-inf"))

        # Apply varlen mask
        if IS_VARLEN:
            s = tl.where(offs_n[None, :] < seq_len, s, float("-inf"))

        p_norm = tl.exp(s - lse[:, None])
        idx_i = offs_m + 1
        idx_i_float = idx_i.to(tl.float32)
        tau_term = tau / idx_i_float
        p_term = p_norm + tau_term[:, None]
        # 用 float32 计算 mask_relu 避免 bf16 精度问题
        p_norm_f32 = p_norm.to(tl.float32)
        p_term_f32 = p_norm_f32 + tau_term[:, None]
        # mask_relu also needs causal mask: when tau > 0, upper triangle would pass p_term > 0
        mask_relu = (p_term_f32 > 0) & valid_mask

        p_elastic = tl.maximum(p_term, 0.0)
        # Re-apply causal mask: when tau > 0, upper triangle would have tau/i > 0
        p_elastic = tl.where(valid_mask, p_elastic, 0.0)
        dv += tl.dot(tl.trans(p_elastic.to(do.dtype)), do)

        dp_elastic = tl.dot(do, tl.trans(v))
        ds = p_norm * (tl.where(mask_relu, dp_elastic, 0.0) - delta[:, None])
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q) * sm_scale

    tl.store(DK_ptr, dk.to(DK.dtype.element_ty), mask=offs_n[:, None] < seq_len)
    tl.store(DV_ptr, dv.to(DV.dtype.element_ty), mask=offs_n[:, None] < seq_len)

# ==========================================
# Python Wrapper
# ==========================================

def lazy_attention_triton(q, k, v, bias, tau, window_size=None, varlen=None):
    # Auto-infer window_size from bias shape if not provided
    if window_size is None:
        window_size = bias.shape[1]
    return LazyAttentionTritonFunc.apply(q, k, v, bias, tau, window_size, varlen)

class LazyAttentionTritonFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, tau, window_size, varlen=None):
        out, lse = _lazy_attention_forward_return_lse(q, k, v, bias, tau, window_size, varlen)
        ctx.save_for_backward(q, k, v, bias, tau, lse, varlen)
        ctx.window_size = window_size
        return out
        
    @staticmethod
    def backward(ctx, do):
        q, k, v, bias, tau, lse, varlen = ctx.saved_tensors
        window_size = ctx.window_size

        # Check which inputs need gradients
        needs_dbias = ctx.needs_input_grad[3]  # bias is 4th input (index 3)
        needs_dtau = ctx.needs_input_grad[4]   # tau is 5th input (index 4)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # Only compute dbias/dtau if needed (skip atomic_add overhead when frozen)
        if needs_dbias or needs_dtau:
            dbias = torch.zeros_like(bias, dtype=torch.float32)
            dtau = torch.zeros_like(tau, dtype=torch.float32)
            _lazy_attention_backward(do, q, k, v, bias, tau, lse, dq, dk, dv, dbias, dtau, window_size, varlen)
        else:
            # Fast path: skip dbias/dtau computation entirely (no atomic_add!)
            dbias = None
            dtau = None
            _lazy_attention_backward_fast(do, q, k, v, bias, tau, lse, dq, dk, dv, window_size, varlen)

        return dq, dk, dv, dbias, dtau, None, None

def _lazy_attention_forward_return_lse(q, k, v, bias, tau, window_size, varlen=None):
    B, H, L, D = q.shape
    BLOCK_M = 64
    BLOCK_N = 64
    
    lse = torch.empty((B, H, L), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    
    grid = (triton.cdiv(L, BLOCK_M), H, B)
    
    IS_VARLEN = varlen is not None
    if not IS_VARLEN:
        varlen_ptr = q # dummy
    else:
        varlen_ptr = varlen
    
    _get_lse_kernel_batch[grid](
        q, k, bias, lse, varlen_ptr,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        bias.stride(0), bias.stride(1),
        q.stride(0), k.stride(0), lse.stride(0),
        H, L, window_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_VARLEN=IS_VARLEN
    )
    
    _lazy_fwd_kernel_batch[grid](
        q, k, v, bias, tau, lse, out, varlen_ptr,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        out.stride(1), out.stride(2), out.stride(3),
        bias.stride(0), bias.stride(1),
        q.stride(0), k.stride(0), v.stride(0), out.stride(0), lse.stride(0),
        H, L, window_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_VARLEN=IS_VARLEN
    )
    
    return out, lse

def _lazy_attention_backward(do, q, k, v, bias, tau, lse, dq, dk, dv, dbias, dtau, window_size, varlen=None):
    """
    Optimized backward pass using per-block buffers for dBias/dTau.

    Key optimization: Instead of all thread blocks atomically competing for
    the same dBias[H, W] array, each block writes to its own buffer slice.
    This eliminates cross-block atomic contention and provides ~1.6x speedup.
    """
    B, H, L, D = q.shape
    BLOCK_M = 64
    BLOCK_N = 64

    num_m_blocks = triton.cdiv(L, BLOCK_M)
    total_blocks = B * H * num_m_blocks

    delta = torch.empty_like(lse)
    grid_m = (num_m_blocks, H, B)

    # Allocate per-block buffers (eliminates atomic contention)
    dbias_blocks = torch.zeros((total_blocks, window_size), device=q.device, dtype=torch.float32)
    dtau_blocks = torch.zeros((total_blocks,), device=q.device, dtype=torch.float32)

    # Handle varlen
    IS_VARLEN = varlen is not None
    if not IS_VARLEN:
        varlen_ptr = q  # dummy
    else:
        varlen_ptr = varlen

    _lazy_bwd_preprocess_kernel[grid_m](
        q, k, v, bias, tau, lse, do, delta, varlen_ptr,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        bias.stride(0), bias.stride(1),
        lse.stride(0), do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        q.stride(0), k.stride(0), v.stride(0), delta.stride(0),
        H, L, window_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_VARLEN=IS_VARLEN
    )

    _lazy_bwd_kernel_dq[grid_m](
        q, k, v, bias, tau, lse, do, delta, varlen_ptr,
        dq, dbias_blocks, dtau_blocks,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        bias.stride(0), bias.stride(1),
        lse.stride(0), do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        q.stride(0), k.stride(0), v.stride(0), delta.stride(0),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        dbias_blocks.stride(0), dbias_blocks.stride(1),
        H, L, window_size, num_m_blocks,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_VARLEN=IS_VARLEN,
        COMPUTE_LAZY_GRAD=True
    )

    # Reduce per-block buffers to final gradients (fast PyTorch operations)
    dbias_blocks = dbias_blocks.view(B, H, num_m_blocks, window_size)
    dbias.copy_(dbias_blocks.sum(dim=(0, 2)))  # [H, W]

    dtau_blocks = dtau_blocks.view(B, H, num_m_blocks)
    dtau.copy_(dtau_blocks.sum(dim=(0, 2)))  # [H]

    grid_n = (triton.cdiv(L, BLOCK_N), H, B)

    _lazy_bwd_kernel_dk_dv[grid_n](
        q, k, v, bias, tau, lse, do, delta, varlen_ptr,
        dk, dv,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        bias.stride(0), bias.stride(1),
        lse.stride(0), do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        q.stride(0), k.stride(0), v.stride(0), delta.stride(0),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        H, L, window_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_VARLEN=IS_VARLEN
    )


def _lazy_attention_backward_fast(do, q, k, v, bias, tau, lse, dq, dk, dv, window_size, varlen=None):
    """
    Fast backward pass that skips dbias/dtau computation.

    Used when bias/tau are frozen (requires_grad=False).
    Saves memory allocation and atomic_add overhead.
    """
    B, H, L, D = q.shape
    BLOCK_M = 64
    BLOCK_N = 64

    num_m_blocks = triton.cdiv(L, BLOCK_M)

    delta = torch.empty_like(lse)
    grid_m = (num_m_blocks, H, B)

    # Dummy buffers (not used when COMPUTE_LAZY_GRAD=False)
    dbias_dummy = torch.empty((1, 1), device=q.device, dtype=torch.float32)
    dtau_dummy = torch.empty((1,), device=q.device, dtype=torch.float32)

    # Handle varlen
    IS_VARLEN = varlen is not None
    if not IS_VARLEN:
        varlen_ptr = q  # dummy
    else:
        varlen_ptr = varlen

    _lazy_bwd_preprocess_kernel[grid_m](
        q, k, v, bias, tau, lse, do, delta, varlen_ptr,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        bias.stride(0), bias.stride(1),
        lse.stride(0), do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        q.stride(0), k.stride(0), v.stride(0), delta.stride(0),
        H, L, window_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_VARLEN=IS_VARLEN
    )

    _lazy_bwd_kernel_dq[grid_m](
        q, k, v, bias, tau, lse, do, delta, varlen_ptr,
        dq, dbias_dummy, dtau_dummy,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        bias.stride(0), bias.stride(1),
        lse.stride(0), do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        q.stride(0), k.stride(0), v.stride(0), delta.stride(0),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        dbias_dummy.stride(0), 1,  # dummy strides
        H, L, window_size, num_m_blocks,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_VARLEN=IS_VARLEN,
        COMPUTE_LAZY_GRAD=False  # Skip dbias/dtau computation!
    )

    # No reduction needed - dbias/dtau are not computed

    grid_n = (triton.cdiv(L, BLOCK_N), H, B)

    _lazy_bwd_kernel_dk_dv[grid_n](
        q, k, v, bias, tau, lse, do, delta, varlen_ptr,
        dk, dv,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        bias.stride(0), bias.stride(1),
        lse.stride(0), do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        q.stride(0), k.stride(0), v.stride(0), delta.stride(0),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        H, L, window_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_VARLEN=IS_VARLEN
    )
