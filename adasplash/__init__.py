try:
    from .adasplash_block_mask import sparse_attn as adasplash
    from .adasplash_no_block_mask import sparse_attn as adasplash_no_block_mask
    from .triton_entmax import triton_entmax
    from .lazy_attention_triton import lazy_attention_triton
except ImportError:
    adasplash = None
    adasplash_no_block_mask = None
    triton_entmax = None
    lazy_attention_triton = None

from .lazy_attention import LazyAttention
