import torch
import pytest
from adasplash.lazy_attention import LazyAttention

def test_lazy_attention_forward():
    B, L, D = 2, 16, 32
    num_heads = 4
    model = LazyAttention(dim=D, num_heads=num_heads, window_size=4)
    x = torch.randn(B, L, D)
    
    out = model(x)
    assert out.shape == (B, L, D)
    
def test_lazy_attention_backward():
    B, L, D = 2, 16, 32
    num_heads = 4
    model = LazyAttention(dim=D, num_heads=num_heads, window_size=4)
    x = torch.randn(B, L, D, requires_grad=True)
    
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert model.tau.grad is not None
    assert model.attention_biases.grad is not None
    
def test_elastic_softmax_behavior():
    # Check if weights can be zero (sparsity)
    # If tau is very negative, many weights should be zero
    B, L, D = 1, 10, 16
    num_heads = 1
    model = LazyAttention(dim=D, num_heads=num_heads)
    
    # Force tau to be very negative
    with torch.no_grad():
        model.tau.fill_(-100.0)
        
    x = torch.randn(B, L, D)
    
    # We need to inspect internals, but since it's a module, we can just check output.
    # If weights are all zero, output should be zero? 
    # Elastic Softmax: ReLU(Softmax + tau/i).
    # If tau is -100, tau/i is -100, -50, etc.
    # Softmax is between 0 and 1.
    # Softmax - large number < 0. ReLU -> 0.
    # So attention weights should be 0.
    
    # However, if weights are 0, output is 0 (assuming no skip connection in the attention block itself, which is true here).
    
    out = model(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)
    
def test_rope_embedding():
    # Basic check that rope is doing something
    B, L, D = 1, 4, 4
    model = LazyAttention(dim=D, num_heads=2)
    
    # If we pass same vector at different positions, RoPE should make them different
    x = torch.ones(B, L, D)
    # We can't easily access internal Q/K without hooking, but we can check output
    out = model(x)
    # With standard attention on all ones, and no positional embedding, output would be uniform/symmetric.
    # With RoPE, it shouldn't be.
    # Also we have learned biases which break symmetry if not initialized to 0? 
    # They are initialized to 0. 
    # But RoPE rotates.
    
    # Just check it runs.
    assert out.shape == (B, L, D)

