#!/usr/bin/env python3
"""Check if we're importing the correct version with the fix"""
import inspect
import sys

# Import the module (not the function)
import adasplash.lazy_attention_triton

print("="*80)
print("Import Path Check")
print("="*80)

# Check where the module is imported from
lat_module = sys.modules['adasplash.lazy_attention_triton']
print(f"\nModule file location:")
print(f"  {lat_module.__file__}")

# Check if the fix is present by inspecting the function signature
print(f"\nChecking if stride_doh fix is present...")

# Get the source code of _lazy_bwd_kernel_dq
try:
    source = inspect.getsource(adasplash.lazy_attention_triton._lazy_bwd_kernel_dq)
    if 'stride_doh' in source:
        print("  ✅ stride_doh FOUND in _lazy_bwd_kernel_dq source!")
    else:
        print("  ❌ stride_doh NOT FOUND in _lazy_bwd_kernel_dq source!")
        print("  This means the fix is not being used!")
except Exception as e:
    print(f"  ⚠️  Could not inspect source: {e}")
    print("  (This is normal for JIT-compiled Triton kernels)")

# Alternative check: look at the backward function
print(f"\nChecking _lazy_attention_backward function...")
try:
    source = inspect.getsource(adasplash.lazy_attention_triton._lazy_attention_backward)
    # Check if do.stride(1) is passed in the kernel calls
    if 'do.stride(1)' in source:
        print("  ✅ do.stride(1) FOUND in kernel calls!")
    else:
        print("  ❌ do.stride(1) NOT FOUND in kernel calls!")
        print("  The fix is NOT applied!")

    # Count occurrences
    count = source.count('do.stride(1)')
    print(f"  Found {count} occurrences of 'do.stride(1)' (should be 3)")

except Exception as e:
    print(f"  Error: {e}")

print("\n" + "="*80)
print("If the fix is NOT found, you need to:")
print("1. Make sure test is using LOCAL adasplash, not system install")
print("2. Try: cd /path/to/adasplash && python test_head_gradients.py")
print("="*80)
