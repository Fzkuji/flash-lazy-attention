import torch
import triton
import triton.language as tl

@triton.jit
def _get_lse_kernel_batch(
    Q, K, Bias, LSE,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_bh, stride_bw,
    stride_qb, stride_kb, stride_lseb, # Batch strides
    n_heads, seq_len,
    window_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    m_block_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    
    offs_m = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    
    # Batch Offsets
    Q_ptr = Q + b_idx * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    K_ptr_base = K + b_idx * stride_kb + h_idx * stride_kh
    Bias_ptr_base = Bias + h_idx * stride_bh # Bias is shared across batch
    LSE_ptr = LSE + b_idx * stride_lseb + h_idx * seq_len + offs_m
    
    # Load Q
    # Handle simplified causal masking where seq_len is multiple of BLOCK
    q = tl.load(Q_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
    
    # Initialize stats
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    
    # Loop over K blocks
    # Causal: only up to current block
    # We loop n from 0 to (m_block_idx + 1) * BLOCK_M
    n_end = (m_block_idx + 1) * BLOCK_M
    
    for n_start in range(0, n_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        
        # Load K
        K_ptr = K_ptr_base + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
        k = tl.load(K_ptr, mask=offs_n[None, :] < seq_len, other=0.0)
        
        # Compute QK^T
        # scaling
        sm_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
        s = tl.dot(q, k) * sm_scale
        
        # Add Distance Bias
        # dist = i - j (since causal i >= j)
        # i = offs_m[:, None], j = offs_n[None, :]
        dist = offs_m[:, None] - offs_n[None, :]
        
        # Check window
        in_window = (dist >= 0) & (dist <= window_size)
        valid_mask = (dist >= 0)
        
        # Load Bias: need to clamp dist for load safety, then mask result
        dist_clamped = tl.minimum(dist, window_size)
        dist_clamped = tl.maximum(dist_clamped, 0)
        
        # Bias ptr: [H, W+1]
        # Using pointer arithmetic for gathering bias
        # bias_val = Bias[h, dist_clamped]
        bias_ptrs = Bias_ptr_base + dist_clamped * stride_bw
        bias_val = tl.load(bias_ptrs, mask=valid_mask, other=0.0)
        
        # Apply window mask to bias: bias is 0 if outside window
        bias_val = tl.where(in_window, bias_val, 0.0)
        
        # Add bias to score
        s += bias_val
        
        # Causal Masking
        s = tl.where(dist >= 0, s, float("-inf"))
        
        # --- Online Softmax Logic ---
        m_block = tl.max(s, 1)
        new_m_i = tl.maximum(m_i, m_block)
        
        # update l_i
        # l_new = l_old * exp(m_old - m_new) + exp(s - m_new).sum()
        alpha = tl.exp(m_i - new_m_i)
        l_i = l_i * alpha + tl.sum(tl.exp(s - new_m_i[:, None]), 1)
        
        m_i = new_m_i

    # Compute final LSE = m_i + log(l_i)
    lse = m_i + tl.log(l_i)
    
    # Store LSE
    tl.store(LSE_ptr, lse, mask=offs_m < seq_len)

@triton.jit
def _lazy_fwd_kernel_batch(
    Q, K, V, Bias, Tau, LSE, Out,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_oh, stride_om, stride_ok,
    stride_bh, stride_bw,
    stride_qb, stride_kb, stride_vb, stride_ob, stride_lseb,
    n_heads, seq_len,
    window_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    # Pass 2: Compute Output using LSE
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
    Out_ptr = Out + b_idx * stride_ob + h_idx * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    LSE_ptr = LSE + b_idx * stride_lseb + h_idx * seq_len + offs_m
    
    # Load LSE: [H, L] (assuming batch handled via pointer offset outside)
    # LSE is [L] for this head/batch
    lse_ptr = LSE + b_idx * stride_lseb + h_idx * seq_len + offs_m
    lse = tl.load(lse_ptr, mask=offs_m < seq_len, other=0.0)
    
    # Load Tau for this head: [H]
    tau = tl.load(Tau + h_idx)
    
    # Load Q
    q = tl.load(Q_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
    
    # Accumulator
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # Loop k/v
    n_end = (m_block_idx + 1) * BLOCK_M
    
    for n_start in range(0, n_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        
        # Load K
        K_ptr = K_ptr_base + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
        k = tl.load(K_ptr, mask=offs_n[None, :] < seq_len, other=0.0)
        
        # Score = QK^T
        sm_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
        s = tl.dot(q, k) * sm_scale
        
        # Add Bias (Same logic as Pass 1)
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
        
        # --- Elastic Softmax Logic ---
        # P_norm = exp(s - lse)
        p_norm = tl.exp(s - lse[:, None])
        
        # Tau offset: tau / i
        # i is the number of attended tokens. For causal, i = current_query_position + 1
        # offs_m is 0-based index. So i = offs_m + 1.
        idx_i = offs_m + 1
        tau_term = tau / idx_i
        
        # Apply ReLU(P + tau/i)
        p_elastic = tl.maximum(p_norm + tau_term[:, None], 0.0)
        
        # Load V
        V_ptr = V_ptr_base + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(V_ptr, mask=offs_n[:, None] < seq_len, other=0.0)
        
        # Accumulate O = P * V
        # p_elastic: [M, N], v: [N, D] -> [M, D]
        acc += tl.dot(p_elastic.to(v.dtype), v)
        
    # Store Output
    tl.store(Out_ptr, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seq_len)

def lazy_attention_triton(q, k, v, bias, tau, window_size=512):
    # Updating Kernel Code Below (Dynamic overwrite)
    return LazyAttentionTritonFunc.apply(q, k, v, bias, tau, window_size)

class LazyAttentionTritonFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, tau, window_size):
        return _lazy_attention_forward(q, k, v, bias, tau, window_size)
        
    @staticmethod
    def backward(ctx, do):
        # Placeholder for backward pass implementation
        # Implementing backward for Two-Pass Flash + Elastic Softmax is complex.
        # For now, we only support forward pass for inference/validation.
        raise NotImplementedError("Backward pass for Lazy Attention Triton kernel is not yet implemented.")

def _lazy_attention_forward(q, k, v, bias, tau, window_size):
    B, H, L, D = q.shape
    BLOCK_M = 64
    BLOCK_N = 64
    
    lse = torch.empty((B, H, L), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    
    grid = (triton.cdiv(L, BLOCK_M), H, B)
    
    # Pass 1
    _get_lse_kernel_batch[grid](
        q, k, bias, lse,
        q.stride(1), q.stride(2), q.stride(3), # stride_qh, qm, qk
        k.stride(1), k.stride(2), k.stride(3),
        bias.stride(0), bias.stride(1),
        q.stride(0), k.stride(0), lse.stride(0), # Batch strides
        H, L, window_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D
    )
    
    # Pass 2
    _lazy_fwd_kernel_batch[grid](
        q, k, v, bias, tau, lse, out,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        out.stride(1), out.stride(2), out.stride(3),
        bias.stride(0), bias.stride(1),
        q.stride(0), k.stride(0), v.stride(0), out.stride(0), lse.stride(0), # Batch strides
        H, L, window_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D
    )
    
    return out
