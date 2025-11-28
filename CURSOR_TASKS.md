# Cursor Tasks: Lazy Attention Improvements

## Task 1: Implement Backward Pass for Training Support 🔥 HIGH PRIORITY

### Goal
Make Lazy Attention trainable by implementing the backward pass for the Triton kernel.

### Current Status
- ✅ Forward pass works correctly
- ❌ Backward pass raises `NotImplementedError`
- File: `adasplash/lazy_attention_triton.py:204-208`

### Requirements

#### 1.1 Implement `_get_lse_kernel_batch` Backward
Create a new kernel `_get_lse_kernel_batch_backward` that computes:
- `dQ` from `dO` (gradient w.r.t queries)
- `dK` from `dO` (gradient w.r.t keys)

**Key considerations:**
- Reuse LSE from forward pass (saved in `ctx`)
- Handle softmax gradient: `dS = P * (dP - sum(dP * P))`
- Handle Elastic-Softmax gradient through ReLU

#### 1.2 Implement `_lazy_fwd_kernel_batch` Backward
Create a new kernel `_lazy_fwd_kernel_batch_backward` that computes:
- `dV` from `dO` (gradient w.r.t values)
- `d_bias` from `dO` (gradient w.r.t attention biases)
- `d_tau` from `dO` (gradient w.r.t tau parameters)

**Key considerations:**
- Elastic-Softmax backward: `d_elastic = dO * (elastic_out > 0)` (ReLU gradient)
- Accumulate `d_bias` across all queries at same distance
- Accumulate `d_tau` with `1/i` weighting

#### 1.3 Update `LazyAttentionTritonFunc.backward()`
```python
@staticmethod
def backward(ctx, do):
    # Retrieve saved tensors from ctx
    q, k, v, bias, tau, lse, window_size = ctx.saved_tensors

    # Call backward kernels
    dq, dk = _get_lse_kernel_batch_backward(...)
    dv, d_bias, d_tau = _lazy_fwd_kernel_batch_backward(...)

    return dq, dk, dv, d_bias, d_tau, None  # None for window_size
```

#### 1.4 Update `LazyAttentionTritonFunc.forward()`
Add `ctx.save_for_backward(q, k, v, bias, tau, lse)` to save tensors needed for backward pass.

### Reference Implementations
- FlashAttention-2 backward: https://github.com/Dao-AILab/flash-attention
- Triton attention tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
- AdaSplash backward pass: `adasplash/adasplash_block_mask.py` (search for `_bwd` kernels)

### Testing
Create `tests/test_lazy_attention_triton_backward.py`:
```python
def test_backward_pass():
    """Test that gradients are computed correctly."""
    q = torch.randn(2, 4, 128, 64, device='cuda', requires_grad=True)
    k = torch.randn(2, 4, 128, 64, device='cuda', requires_grad=True)
    v = torch.randn(2, 4, 128, 64, device='cuda', requires_grad=True)
    bias = torch.randn(4, 33, device='cuda', requires_grad=True)
    tau = torch.full((4,), -1.0, device='cuda', requires_grad=True)

    # Forward pass
    out = lazy_attention_triton(q, k, v, bias, tau, window_size=32)

    # Backward pass
    loss = out.sum()
    loss.backward()

    # Check gradients exist
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert bias.grad is not None
    assert tau.grad is not None

def test_gradcheck():
    """Validate gradients using torch.autograd.gradcheck."""
    # Use small sizes for gradcheck (it's slow)
    q = torch.randn(1, 2, 32, 16, device='cuda', dtype=torch.float64, requires_grad=True)
    # ... similar for k, v, bias, tau

    assert torch.autograd.gradcheck(
        lambda q, k, v, b, t: lazy_attention_triton(q, k, v, b, t, window_size=8),
        (q, k, v, bias, tau),
        atol=1e-3,
        rtol=1e-3
    )
```

### Success Criteria
- ✅ `loss.backward()` completes without errors
- ✅ All gradients (`dq`, `dk`, `dv`, `d_bias`, `d_tau`) are non-None
- ✅ Gradients match PyTorch reference within tolerance (1e-2)
- ✅ `torch.autograd.gradcheck` passes

---

## Task 2: Add Variable Length Support 🚀 MEDIUM PRIORITY

### Goal
Support batches with different sequence lengths without padding overhead.

### Current Status
- ❌ Assumes all sequences have same length `L`
- ❌ Wastes computation on padding tokens

### Requirements

#### 2.1 Update Function Signature
```python
def lazy_attention_triton(q, k, v, bias, tau, window_size=512, varlen=None):
    """
    Args:
        ...
        varlen: Optional[torch.Tensor] - Actual sequence lengths [B] (int32)
                If None, assumes all sequences have length L.
    """
```

#### 2.2 Update Kernel to Handle Per-Sequence Lengths
In both `_get_lse_kernel_batch` and `_lazy_fwd_kernel_batch`:

```python
@triton.jit
def _get_lse_kernel_batch(
    Q, K, Bias, LSE, VARLEN,  # Add VARLEN parameter
    ...,
    IS_VARLEN: tl.constexpr,  # Add compile-time flag
):
    # Load actual sequence length for this batch
    if IS_VARLEN:
        seq_len = tl.load(VARLEN + b_idx).to(tl.int32)
    else:
        seq_len = seq_len  # Use provided constant

    # Adjust loop bounds
    if IS_VARLEN:
        n_end = min((m_block_idx + 1) * BLOCK_M, seq_len)
    else:
        n_end = (m_block_idx + 1) * BLOCK_M

    # Use seq_len in all masking operations
    q = tl.load(Q_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
    k = tl.load(K_ptr, mask=offs_n[None, :] < seq_len, other=0.0)
```

#### 2.3 Update Python Wrapper
```python
def _lazy_attention_forward(q, k, v, bias, tau, window_size, varlen=None):
    B, H, L, D = q.shape

    if varlen is not None:
        assert varlen.shape == (B,), f"varlen shape must be [B], got {varlen.shape}"
        assert varlen.device == q.device
        IS_VARLEN = True
    else:
        varlen = torch.full((B,), L, device=q.device, dtype=torch.int32)
        IS_VARLEN = False

    # Pass varlen and IS_VARLEN to kernels
    _get_lse_kernel_batch[grid](
        q, k, bias, lse, varlen,
        ...,
        IS_VARLEN=IS_VARLEN,
        ...
    )
```

### Reference Implementation
- AdaSplash varlen: `adasplash/adasplash_block_mask.py:100-102` and similar sections

### Testing
```python
def test_varlen_support():
    """Test variable length sequences."""
    B, H, L, D = 2, 4, 128, 64

    # Create inputs with different actual lengths
    varlen = torch.tensor([64, 100], device='cuda', dtype=torch.int32)

    q = torch.randn(B, H, L, D, device='cuda')
    k = torch.randn(B, H, L, D, device='cuda')
    v = torch.randn(B, H, L, D, device='cuda')
    bias = torch.randn(H, 33, device='cuda')
    tau = torch.full((H,), -1.0, device='cuda')

    # Should only compute attention for actual lengths
    out = lazy_attention_triton(q, k, v, bias, tau, window_size=32, varlen=varlen)

    # Verify no cross-contamination
    # First sequence should not attend beyond position 64
    # Second sequence should not attend beyond position 100

def test_varlen_vs_padded():
    """Verify varlen gives same results as padded version."""
    # Test that varlen=[64, 64] gives same output as no varlen with L=64
```

### Success Criteria
- ✅ Accepts `varlen` parameter
- ✅ Computes correct attention for each sequence length
- ✅ No cross-contamination between sequences
- ✅ Matches padded results when lengths are equal

---

## Task 3: Add Block Sparsity Masking for Speedup ⚡ LOW PRIORITY

### Goal
Skip computation for blocks where all attention weights are zero after Elastic-Softmax.

### Current Status
- ✅ Elastic-Softmax produces sparse attention weights
- ❌ All blocks are computed even if weights are zero
- ❌ No actual speedup from sparsity

### Requirements

#### 3.1 Add Block Mask Tensor
```python
def _lazy_attention_forward(q, k, v, bias, tau, window_size, varlen=None):
    B, H, L, D = q.shape
    BLOCK_M = 64
    BLOCK_N = 64

    # Allocate block mask: [B, H, num_blocks_m, num_blocks_n]
    num_blocks_m = triton.cdiv(L, BLOCK_M)
    num_blocks_n = triton.cdiv(L, BLOCK_N)
    bmask = torch.zeros((B, H, num_blocks_m, num_blocks_n),
                        device=q.device, dtype=torch.bool)
```

#### 3.2 Update Forward Pass to Compute Block Mask
In `_lazy_fwd_kernel_batch`:
```python
# After computing p_elastic
# Check if any weight in this block is non-zero
block_has_nonzero = tl.sum(p_elastic > 0.0) > 0

# Store mask (needs atomic operation or reduction)
if block_has_nonzero:
    # Store to BMASK[b_idx, h_idx, m_block_idx, n_block_idx]
    pass

# Only compute matmul if block is non-zero
if block_has_nonzero:
    acc += tl.dot(p_elastic.to(v.dtype), v)
```

#### 3.3 Add Sparsity Metrics
```python
def get_sparsity_stats(bmask):
    """Calculate sparsity statistics from block mask.

    Returns:
        density: Fraction of non-zero blocks
        sink_ratio: Attention mass on first token
    """
    density = bmask.float().mean().item()
    return {
        'block_density': density,
        'theoretical_speedup': 1.0 / max(density, 0.01)
    }
```

### Reference Implementation
- AdaSplash block masking: `adasplash/adasplash_block_mask.py:220-230`

### Testing
```python
def test_block_sparsity():
    """Test that block masking skips zero blocks."""
    # Use very negative tau to force high sparsity
    tau = torch.full((4,), -10.0, device='cuda')

    out, bmask = lazy_attention_triton(q, k, v, bias, tau,
                                        window_size=32,
                                        return_mask=True)

    # Verify blocks are actually sparse
    assert bmask.float().mean() < 0.5, "Expected >50% sparsity with tau=-10"

def test_speedup_with_sparsity():
    """Benchmark speedup from sparsity."""
    import time

    # Dense attention (tau=0)
    t0 = time.time()
    out_dense = lazy_attention_triton(q, k, v, bias, tau_dense, window_size=32)
    time_dense = time.time() - t0

    # Sparse attention (tau=-5)
    t0 = time.time()
    out_sparse = lazy_attention_triton(q, k, v, bias, tau_sparse, window_size=32)
    time_sparse = time.time() - t0

    speedup = time_dense / time_sparse
    print(f"Speedup from sparsity: {speedup:.2f}x")
```

### Success Criteria
- ✅ Block mask correctly identifies zero/nonzero blocks
- ✅ Zero blocks are skipped during computation
- ✅ Achieves measurable speedup (>1.2x) at >50% sparsity
- ✅ Results match non-masked version

---

## Implementation Priority

1. **Task 1 (Backward Pass)** - Required for training, highest impact
2. **Task 2 (Variable Length)** - Improves efficiency, moderate complexity
3. **Task 3 (Block Masking)** - Performance optimization, optional

## Tips for Cursor

### Debugging Triton Kernels
```python
# Enable debug mode
import os
os.environ['TRITON_INTERPRET'] = '1'

# Print values inside kernel
tl.device_print("Debug:", value)
```

### Testing Individual Kernels
```python
# Test kernel in isolation
@triton.testing.perf_report(...)
def benchmark_kernel():
    # Benchmark specific kernel
```

### Reference Existing Code
- Look at AdaSplash implementations for inspiration
- FlashAttention repo has excellent backward pass examples
- Triton tutorials cover attention patterns

### Common Pitfalls
- ⚠️ Ensure all pointers are correctly offset for batch dimension
- ⚠️ Watch out for integer overflow in stride calculations
- ⚠️ Use `tl.load(..., mask=..., other=0.0)` to avoid reading out of bounds
- ⚠️ Remember to add `HEAD_DIM: tl.constexpr` for any `tl.arange(0, HEAD_DIM)`

Good luck! 🚀
