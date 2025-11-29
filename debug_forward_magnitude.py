import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

def debug_lazy_attention():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print("CUDA not available, skipping Triton test.")
        return

    # Try bfloat16 if supported, else float32
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    
    B, H, L, D = 2, 4, 128, 64
    window_size = 64
    
    print(f"--- Debugging Lazy Attention with {dtype} on {device} ---")
    
    q = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    
    # Bias initialized small
    bias = torch.zeros(H, window_size, device=device, dtype=torch.float32, requires_grad=True)
    torch.nn.init.normal_(bias, std=1e-3)
    
    # Tau = -1.0
    tau = torch.full((H,), -1.0, device=device, dtype=torch.float32, requires_grad=True)
    
    # Forward
    print("Running Forward Pass...")
    try:
        # Note: converting tau to input dtype as per user usage
        out = lazy_attention_triton(q, k, v, bias, tau.to(dtype), window_size=window_size)
    except Exception as e:
        print(f"Forward failed: {e}")
        return

    print(f"Output shape: {out.shape}")
    print(f"Output mean: {out.float().mean().item()}")
    print(f"Output max: {out.float().max().item()}")
    print(f"Output min: {out.float().min().item()}")
    
    if out.abs().max() == 0:
        print("❌ Forward Output is ALL ZEROS! Attention is completely dead.")
    else:
        print("✅ Forward Output is NOT all zeros.")
        
    # Backward
    print("Running Backward Pass...")
    try:
        loss = out.sum()
        loss.backward()
    except Exception as e:
        print(f"Backward failed: {e}")
        return
    
    print("\n--- Gradients ---")
    print(f"Bias grad exists: {bias.grad is not None}")
    if bias.grad is not None:
        g_bias = bias.grad.float()
        print(f"Bias grad max: {g_bias.abs().max().item()}")
        print(f"Bias grad mean: {g_bias.abs().mean().item()}")
        if g_bias.abs().max() == 0:
            print("❌ Bias grad is ALL ZEROS!")
        else:
            print("✅ Bias grad looks OK.")
            
    print(f"Tau grad exists: {tau.grad is not None}")
    if tau.grad is not None:
        g_tau = tau.grad.float()
        print(f"Tau grad: {g_tau}")
        if g_tau.abs().max() == 0:
            print("❌ Tau grad is ALL ZEROS!")
        else:
            print("✅ Tau grad looks OK.")

if __name__ == "__main__":
    debug_lazy_attention()

