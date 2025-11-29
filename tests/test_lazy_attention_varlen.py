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
def test_varlen_correctness():
    """
    Test that varlen correctly masks out tokens beyond sequence length.
    """
    torch.manual_seed(42)
    B, H, L, D = 2, 4, 128, 64
    window_size = 32
    
    # Create inputs
    q = torch.randn(B, H, L, D, device='cuda')
    k = torch.randn(B, H, L, D, device='cuda')
    v = torch.randn(B, H, L, D, device='cuda')
    bias = torch.randn(H, window_size + 1, device='cuda')
    tau = torch.full((H,), -1.0, device='cuda')
    
    # Define variable lengths
    # Batch 0: Length 64
    # Batch 1: Length 100
    varlen = torch.tensor([64, 100], device='cuda', dtype=torch.int32)
    
    # Run Triton kernel with varlen
    out_varlen = lazy_attention_triton(q, k, v, bias, tau, window_size=window_size, varlen=varlen)
    
    # Check output masking
    # For batch 0, output[0, :, 64:, :] should be irrelevant (or ideally 0 if store masked properly, 
    # but kernel stores 'acc' which is accumulated 0 if inputs masked).
    # In our implementation, we load q masked, so accumulation should be 0.
    
    # Check Batch 0
    assert torch.allclose(out_varlen[0, :, 64:, :], torch.zeros_like(out_varlen[0, :, 64:, :]), atol=1e-5)
    
    # Check Batch 1
    assert torch.allclose(out_varlen[1, :, 100:, :], torch.zeros_like(out_varlen[1, :, 100:, :]), atol=1e-5)

@pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed")
def test_varlen_vs_padded_reference():
    """
    Verify that varlen implementation matches standard padded implementation (masked manually).
    """
    torch.manual_seed(42)
    B, H, L, D = 2, 2, 64, 32
    window_size = 16
    
    q = torch.randn(B, H, L, D, device='cuda')
    k = torch.randn(B, H, L, D, device='cuda')
    v = torch.randn(B, H, L, D, device='cuda')
    bias = torch.randn(H, window_size + 1, device='cuda')
    tau = torch.full((H,), -1.0, device='cuda')
    
    varlen = torch.tensor([32, 48], device='cuda', dtype=torch.int32)
    
    # 1. Run Varlen Kernel
    out_varlen = lazy_attention_triton(q, k, v, bias, tau, window_size=window_size, varlen=varlen)
    
    # 2. Run Standard Kernel (Padded)
    # We pass full length, kernel computes everything.
    # But the result for valid positions should be identical to varlen result.
    # Note: Standard kernel without varlen computes attention over padding tokens too.
    # BUT, if Q/K/V contain garbage in padding, result differs.
    # We must zero-out padding in inputs to compare fairly if standard kernel doesn't take mask.
    # Our standard kernel (without varlen arg) treats seq_len=L.
    # So it will attend to padding tokens.
    # To match, we need to ensure padding tokens don't contribute.
    # Since standard kernel doesn't support attention_mask argument, we can't easily make it ignore padding 
    # unless we manually mask scores inside (which we can't from outside).
    
    # Actually, we can just verify that out_varlen valid parts match what we expect.
    # Let's run standard kernel on a sliced input? No, batching makes it hard.
    
    # Let's compare against PyTorch Reference with manual masking
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Distance Bias
    indices = torch.arange(L, device='cuda')
    dist = indices[:, None] - indices[None, :]
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
    
    # --- KEY: Apply Padding Mask for Reference ---
    # Mask out keys that are padding
    # mask[b, i, j] is True if j < varlen[b]
    # Actually, we also need to mask out Queries that are padding to match output=0
    
    for b in range(B):
        vl = varlen[b]
        # Mask Keys: scores[:, :, :, vl:] = -inf
        scores[b, :, :, vl:] = float('-inf')
        
        # Mask Queries (optional, affects output)
        # We want output[b, :, vl:, :] = 0
    
    # Elastic Softmax
    probs = torch.softmax(scores, dim=-1)
    row_indices = torch.arange(1, L + 1, device='cuda').view(1, 1, L, 1)
    tau_term = tau.view(1, H, 1, 1) / row_indices
    probs_elastic = torch.relu(probs + tau_term)
    
    out_ref = torch.matmul(probs_elastic, v)
    
    # Zero out padding in reference output
    for b in range(B):
        vl = varlen[b]
        out_ref[b, :, vl:, :] = 0.0
        
    # Compare
    # Only compare valid regions
    for b in range(B):
        vl = varlen[b]
        valid_out_triton = out_varlen[b, :, :vl, :]
        valid_out_ref = out_ref[b, :, :vl, :]
        assert torch.allclose(valid_out_triton, valid_out_ref, atol=1e-2, rtol=1e-2)

@pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed")
def test_varlen_backward():
    """Test that backward pass works correctly with varlen."""
    torch.manual_seed(42)
    B, H, L, D = 2, 4, 64, 32
    window_size = 16

    q = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
    k = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
    v = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
    bias = torch.randn(H, window_size, device='cuda', requires_grad=True)
    tau = torch.full((H,), -1.0, device='cuda', requires_grad=True)

    varlen = torch.tensor([32, 48], device='cuda', dtype=torch.int32)

    # Forward
    out = lazy_attention_triton(q, k, v, bias, tau, window_size=window_size, varlen=varlen)

    # Backward
    loss = out.sum()
    loss.backward()

    # Check gradients exist
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None

    # Check padding positions have zero gradient
    assert torch.allclose(q.grad[0, :, 32:, :], torch.zeros_like(q.grad[0, :, 32:, :]), atol=1e-5)
    assert torch.allclose(q.grad[1, :, 48:, :], torch.zeros_like(q.grad[1, :, 48:, :]), atol=1e-5)
    assert torch.allclose(k.grad[0, :, 32:, :], torch.zeros_like(k.grad[0, :, 32:, :]), atol=1e-5)
    assert torch.allclose(k.grad[1, :, 48:, :], torch.zeros_like(k.grad[1, :, 48:, :]), atol=1e-5)
    assert torch.allclose(v.grad[0, :, 32:, :], torch.zeros_like(v.grad[0, :, 32:, :]), atol=1e-5)
    assert torch.allclose(v.grad[1, :, 48:, :], torch.zeros_like(v.grad[1, :, 48:, :]), atol=1e-5)

if __name__ == "__main__":
    test_varlen_correctness()
    test_varlen_vs_padded_reference()
    test_varlen_backward()

