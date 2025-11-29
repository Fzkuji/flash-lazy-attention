# Bug Fix Proposal

## Root Cause Found

**Issue**: DO (output gradient) pointer calculation uses wrong stride for head dimension.

### Current Code (WRONG)
In all three kernels (preprocess, dq, dk_dv), line 204, 293, 410:
```python
DO_ptr = DO + b_idx * stride_dob + h_idx * stride_qh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
                                            ^^^^^^^^^ WRONG! Using Q's head stride for DO
```

### Kernel Parameters (line 268)
```python
stride_lseb, stride_dob, stride_om, stride_ok,
```
Missing: stride for DO's head dimension!

### Kernel Call (line 563)
```python
lse.stride(0), do.stride(0), do.stride(2), do.stride(3),
```
Missing: `do.stride(1)` for head dimension!

## Why This Causes the Bug

1. DO pointer for h_idx=0 uses: `DO + b_idx * stride_dob + 0 * stride_qh + ...` ✅ Works
2. DO pointer for h_idx>0 uses: `DO + b_idx * stride_dob + h_idx * stride_qh + ...` ❌ WRONG ADDRESS!

If `stride_qh != do.stride(1)`, h_idx>0 reads/writes to wrong memory locations!

## Fix

### Step 1: Add DO head stride parameter to all three kernels

**_lazy_bwd_preprocess_kernel** (line 174):
```python
def _lazy_bwd_preprocess_kernel(
    Q, K, V, Bias, Tau, LSE, DO, Delta, VARLEN,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_bh, stride_bw,
    stride_lseb, stride_dob, stride_doh, stride_om, stride_ok,  # ADD stride_doh
    stride_qb, stride_kb, stride_vb, stride_deltab,
    n_heads, seq_len_max, window_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
```

**_lazy_bwd_kernel_dq** (line 261):
```python
def _lazy_bwd_kernel_dq(
    Q, K, V, Bias, Tau, LSE, DO, Delta, VARLEN,
    DQ, DBias, DTau,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_bh, stride_bw,
    stride_lseb, stride_dob, stride_doh, stride_om, stride_ok,  # ADD stride_doh
    stride_qb, stride_kb, stride_vb, stride_deltab,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    n_heads, seq_len_max, window_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
```

**_lazy_bwd_kernel_dk_dv** (line 362):
```python
def _lazy_bwd_kernel_dk_dv(
    Q, K, V, Bias, Tau, LSE, DO, Delta, VARLEN,
    DK, DV,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_bh, stride_bw,
    stride_lseb, stride_dob, stride_doh, stride_om, stride_ok,  # ADD stride_doh
    stride_qb, stride_kb, stride_vb, stride_deltab,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_dvb, stride_dvh, stride_dvn, stride_dvk,
    n_heads, seq_len_max, window_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
```

### Step 2: Update DO pointer calculation

Change in all three kernels (lines 204, 293, 410):
```python
# OLD (WRONG):
DO_ptr = DO + b_idx * stride_dob + h_idx * stride_qh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok

# NEW (CORRECT):
DO_ptr = DO + b_idx * stride_dob + h_idx * stride_doh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
```

### Step 3: Update kernel calls

**_lazy_bwd_preprocess_kernel call** (line 543):
```python
_lazy_bwd_preprocess_kernel[grid_m](
    q, k, v, bias, tau, lse, do, delta, varlen_ptr,
    q.stride(1), q.stride(2), q.stride(3),
    k.stride(1), k.stride(2), k.stride(3),
    v.stride(1), v.stride(2), v.stride(3),
    bias.stride(0), bias.stride(1),
    lse.stride(0), do.stride(0), do.stride(1), do.stride(2), do.stride(3),  # ADD do.stride(1)
    q.stride(0), k.stride(0), v.stride(0), delta.stride(0),
    H, L, window_size,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
    IS_VARLEN=IS_VARLEN
)
```

**_lazy_bwd_kernel_dq call** (line 556):
```python
_lazy_bwd_kernel_dq[grid_m](
    q, k, v, bias, tau, lse, do, delta, varlen_ptr,
    dq, dbias, dtau,
    q.stride(1), q.stride(2), q.stride(3),
    k.stride(1), k.stride(2), k.stride(3),
    v.stride(1), v.stride(2), v.stride(3),
    bias.stride(0), bias.stride(1),
    lse.stride(0), do.stride(0), do.stride(1), do.stride(2), do.stride(3),  # ADD do.stride(1)
    q.stride(0), k.stride(0), v.stride(0), delta.stride(0),
    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
    H, L, window_size,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
    IS_VARLEN=IS_VARLEN
)
```

**_lazy_bwd_kernel_dk_dv call** (line 573):
```python
_lazy_bwd_kernel_dk_dv[grid_n](
    q, k, v, bias, tau, lse, do, delta, varlen_ptr,
    dk, dv,
    q.stride(1), q.stride(2), q.stride(3),
    k.stride(1), k.stride(2), k.stride(3),
    v.stride(1), v.stride(2), v.stride(3),
    bias.stride(0), bias.stride(1),
    lse.stride(0), do.stride(0), do.stride(1), do.stride(2), do.stride(3),  # ADD do.stride(1)
    q.stride(0), k.stride(0), v.stride(0), delta.stride(0),
    dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
    dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
    H, L, window_size,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
    IS_VARLEN=IS_VARLEN
)
```

## Expected Result

After this fix, all heads should receive gradients correctly!
