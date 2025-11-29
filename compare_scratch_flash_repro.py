
import torch
import torch.nn.functional as F
import math
from adasplash import lazy_attention_triton

def lazy_attention_pytorch(q, k, v, bias, tau, window_size):
    """
    PyTorch reference implementation of Lazy Attention.
    """
    B, H, L, D = q.shape
    scale = 1.0 / math.sqrt(D)
    
    # [B, H, L, L]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Add Learnable Bias
    # Bias: [H, window_size]
    # We need to construct the full [H, L, L] bias matrix
    bias_full = torch.zeros(H, L, L, device=q.device, dtype=q.dtype)
    
    # This loop is slow but correct
    for i in range(L):
        for j in range(L):
            dist = i - j
            if 0 <= dist < window_size:
                bias_full[:, i, j] = bias[:, dist]
            else:
                pass # 0 for outside window, but we have causal mask anyway
                
    scores = scores + bias_full.unsqueeze(0) # [1, H, L, L]
    
    # Causal Mask
    causal_mask = torch.tril(torch.ones((L, L), device=q.device, dtype=torch.bool))
    scores = scores.masked_fill(~causal_mask, float('-inf'))
    
    # Elastic Softmax
    # 1. Softmax(scores)
    probs = torch.softmax(scores, dim=-1)
    
    # 2. Add tau/i term
    # i is 1-based index of the row (query index)
    row_indices = torch.arange(1, L + 1, device=q.device).view(1, 1, L, 1)
    tau_term = tau.view(1, H, 1, 1) / row_indices
    
    probs_elastic = torch.relu(probs + tau_term)
    
    # 3. Matmul with V
    output = torch.matmul(probs_elastic, v)
    
    return output

def test_forward_backward_match():
    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.float32 # Test with float32 first to isolate logic bugs from precision issues
    
    B, H, L, D = 2, 4, 64, 32
    window_size = 16
    
    print(f"Config: B={B}, H={H}, L={L}, D={D}, window_size={window_size}, dtype={dtype}")
    
    # Inputs
    q = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(H, window_size, device=device, dtype=dtype, requires_grad=True)
    tau = torch.full((H,), -1.0, device=device, dtype=dtype, requires_grad=True)
    
    # --- Path 1: Triton ---
    # Clone inputs to ensure gradients don't mix
    q_t = q.clone().detach().requires_grad_(True)
    k_t = k.clone().detach().requires_grad_(True)
    v_t = v.clone().detach().requires_grad_(True)
    bias_t = bias.clone().detach().requires_grad_(True)
    tau_t = tau.clone().detach().requires_grad_(True)
    
    out_triton = lazy_attention_triton(q_t, k_t, v_t, bias_t, tau_t, window_size=window_size)
    loss_triton = out_triton.sum()
    loss_triton.backward()
    
    # --- Path 2: PyTorch ---
    q_ref = q.clone().detach().requires_grad_(True)
    k_ref = k.clone().detach().requires_grad_(True)
    v_ref = v.clone().detach().requires_grad_(True)
    bias_ref = bias.clone().detach().requires_grad_(True)
    tau_ref = tau.clone().detach().requires_grad_(True)
    
    out_ref = lazy_attention_pytorch(q_ref, k_ref, v_ref, bias_ref, tau_ref, window_size)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    
    # --- Comparison ---
    print("\n--- Forward Comparison ---")
    if torch.allclose(out_triton, out_ref, atol=1e-3, rtol=1e-3):
        print("Forward Pass MATCH ✅")
    else:
        print("Forward Pass MISMATCH ❌")
        diff = (out_triton - out_ref).abs().max()
        print(f"Max Diff: {diff}")
        # Debug Heads
        for h in range(H):
            if not torch.allclose(out_triton[:, h], out_ref[:, h], atol=1e-3):
                print(f"Mismatch in Head {h}")
    
    print("\n--- Backward Comparison ---")
    
    grads = {
        "q": (q_t.grad, q_ref.grad),
        "k": (k_t.grad, k_ref.grad),
        "v": (v_t.grad, v_ref.grad),
        "bias": (bias_t.grad, bias_ref.grad),
        "tau": (tau_t.grad, tau_ref.grad)
    }
    
    for name, (g_t, g_ref) in grads.items():
        if g_t is None or g_ref is None:
            print(f"{name}: Gradients Missing ❌")
            continue
            
        if torch.allclose(g_t, g_ref, atol=1e-2, rtol=1e-2):
            print(f"{name}.grad MATCH ✅")
        else:
            print(f"{name}.grad MISMATCH ❌")
            diff = (g_t - g_ref).abs().max()
            print(f"Max Diff: {diff}")
            if name in ["bias", "tau"]:
                print(f"Triton {name}.grad:\n{g_t}")
                print(f"Ref {name}.grad:\n{g_ref}")

if __name__ == "__main__":
    test_forward_backward_match()

