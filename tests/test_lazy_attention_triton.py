import torch
import pytest
import math

# Try to import Triton version
try:
    from adasplash.lazy_attention_triton import lazy_attention_triton
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

from adasplash.lazy_attention import LazyAttention

@pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed")
def test_lazy_attention_triton_vs_pytorch():
    """
    Compare Triton implementation output against PyTorch reference implementation.
    Note: PyTorch implementation includes RoPE and projection layers.
    Triton implementation expects pre-rotated Q/K and raw tensors.
    """
    
    torch.manual_seed(42)
    
    B, H, L, D = 2, 4, 128, 64
    window_size = 32
    
    # Setup inputs
    q = torch.randn(B, H, L, D, device='cuda', dtype=torch.float32)
    k = torch.randn(B, H, L, D, device='cuda', dtype=torch.float32)
    v = torch.randn(B, H, L, D, device='cuda', dtype=torch.float32)
    
    # Bias: [H, W+1]
    bias = torch.randn(H, window_size + 1, device='cuda', dtype=torch.float32)
    
    # Tau: [H]
    tau = torch.full((H,), -1.0, device='cuda', dtype=torch.float32)
    
    # --- Run Triton Implementation ---
    out_triton = lazy_attention_triton(q, k, v, bias, tau, window_size=window_size)
    
    # --- Run PyTorch Reference ---
    # We need to replicate the exact math of the Triton kernel in PyTorch
    # The LazyAttention module puts everything together. 
    # Here we implement the "Functional" equivalent in PyTorch for direct comparison.
    
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale # [B, H, L, L]
    
    # Distance Bias
    indices = torch.arange(L, device='cuda')
    dist = indices[:, None] - indices[None, :] # i - j
    # Causal mask in kernel: dist >= 0
    
    # Kernel Logic for bias:
    # dist_clamped = min(dist, window_size), max(0)
    # bias_val = Bias[h, dist_clamped]
    # if in_window (0 <= dist <= window_size): s += bias_val
    
    bias_expanded = torch.zeros_like(scores)
    for h in range(H):
        for i in range(L):
            for j in range(L):
                d = i - j
                if 0 <= d <= window_size:
                    bias_expanded[:, h, i, j] = bias[h, d]
                    
    scores = scores + bias_expanded
    
    # Causal Mask
    causal_mask = torch.tril(torch.ones((L, L), device='cuda', dtype=torch.bool))
    scores = scores.masked_fill(~causal_mask, float('-inf'))
    
    # Elastic Softmax
    # P_norm = exp(S - LSE) = softmax(S)
    probs = torch.softmax(scores, dim=-1)
    
    # ReLU(P + tau/i)
    # i = row_idx + 1
    row_indices = torch.arange(1, L + 1, device='cuda').view(1, 1, L, 1)
    tau_term = tau.view(1, H, 1, 1) / row_indices
    
    probs_elastic = torch.relu(probs + tau_term)
    
    out_ref = torch.matmul(probs_elastic, v)
    
    # Comparison
    # Tolerances might need to be loose due to float32 vs potential accumulations
    assert torch.allclose(out_triton, out_ref, atol=1e-3, rtol=1e-3)

if __name__ == "__main__":
    test_lazy_attention_triton_vs_pytorch()

