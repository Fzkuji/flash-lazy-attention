import torch
import pytest
import math

try:
    from adasplash.lazy_attention_triton import lazy_attention_triton
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

@pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed")
def test_backward_gradients_exist():
    """Test that backward pass runs and produces gradients."""
    B, H, L, D = 2, 4, 64, 32
    window_size = 16
    
    q = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
    k = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
    v = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
    bias = torch.randn(H, window_size + 1, device='cuda', requires_grad=True)
    tau = torch.full((H,), -1.0, device='cuda', requires_grad=True)
    
    out = lazy_attention_triton(q, k, v, bias, tau, window_size=window_size)
    loss = out.sum()
    loss.backward()
    
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert bias.grad is not None
    assert tau.grad is not None
    
    # Check for NaN
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()

@pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed")
def test_backward_vs_pytorch():
    """Compare Triton backward gradients against PyTorch reference."""
    torch.manual_seed(42)
    
    B, H, L, D = 1, 1, 32, 16 # Small size for easier debug
    window_size = 8
    
    # Inputs
    q_ref = torch.randn(B, H, L, D, device='cuda', dtype=torch.float64, requires_grad=True)
    k_ref = torch.randn(B, H, L, D, device='cuda', dtype=torch.float64, requires_grad=True)
    v_ref = torch.randn(B, H, L, D, device='cuda', dtype=torch.float64, requires_grad=True)
    bias_ref = torch.randn(H, window_size + 1, device='cuda', dtype=torch.float64, requires_grad=True)
    tau_ref = torch.full((H,), -1.0, device='cuda', dtype=torch.float64, requires_grad=True)
    
    # Clone for Triton (FP32)
    q_tri = q_ref.clone().detach().float().requires_grad_(True)
    k_tri = k_ref.clone().detach().float().requires_grad_(True)
    v_tri = v_ref.clone().detach().float().requires_grad_(True)
    bias_tri = bias_ref.clone().detach().float().requires_grad_(True)
    tau_tri = tau_ref.clone().detach().float().requires_grad_(True)
    
    # --- PyTorch Reference Forward ---
    # Re-implement functional logic in high precision
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale
    
    # Distance Bias
    indices = torch.arange(L, device='cuda')
    dist = indices[:, None] - indices[None, :]
    bias_expanded = torch.zeros_like(scores)
    for h in range(H):
        for i in range(L):
            for j in range(L):
                d = i - j
                if 0 <= d <= window_size:
                    bias_expanded[:, h, i, j] = bias_ref[h, d]
    scores = scores + bias_expanded
    
    # Causal Mask
    causal_mask = torch.tril(torch.ones((L, L), device='cuda', dtype=torch.bool))
    scores = scores.masked_fill(~causal_mask, float('-inf'))
    
    # Elastic Softmax
    probs = torch.softmax(scores, dim=-1)
    row_indices = torch.arange(1, L + 1, device='cuda').view(1, 1, L, 1)
    tau_term = tau_ref.view(1, H, 1, 1) / row_indices
    probs_elastic = torch.relu(probs + tau_term)
    out_ref = torch.matmul(probs_elastic, v_ref)
    
    # --- Triton Forward ---
    out_tri = lazy_attention_triton(q_tri, k_tri, v_tri, bias_tri, tau_tri, window_size=window_size)
    
    # --- Backward ---
    grad_out = torch.randn_like(out_ref)
    
    out_ref.backward(grad_out)
    out_tri.backward(grad_out.float())
    
    # --- Compare Gradients ---
    # Relaxed tolerance for Backward pass (accumulation errors)
    atol, rtol = 1e-2, 1e-2
    
    assert torch.allclose(q_tri.grad, q_ref.grad.float(), atol=atol, rtol=rtol), "dQ mismatch"
    assert torch.allclose(k_tri.grad, k_ref.grad.float(), atol=atol, rtol=rtol), "dK mismatch"
    assert torch.allclose(v_tri.grad, v_ref.grad.float(), atol=atol, rtol=rtol), "dV mismatch"
    
    # Bias gradients might be sparse, check sum or non-zero parts?
    # Bias grad check:
    # Only indices [0..window_size] are used.
    # PyTorch grad handles this automatically.
    assert torch.allclose(bias_tri.grad, bias_ref.grad.float(), atol=atol, rtol=rtol), "dBias mismatch"
    assert torch.allclose(tau_tri.grad, tau_ref.grad.float(), atol=atol, rtol=rtol), "dTau mismatch"

if __name__ == "__main__":
    test_backward_gradients_exist()
    test_backward_vs_pytorch()

