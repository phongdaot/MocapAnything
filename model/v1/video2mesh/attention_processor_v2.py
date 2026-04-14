# attention_processor_0821
#
# Re-export shim. The original module has been split into per-class files:
#   - attention_rope_utils.py         : precompute_freqs_cis, apply_rotary_emb_1d_on_frame
#   - flash_triposg_attn_processor.py : FlashTripoSGAttnProcessor2_0
#   - triposg_attn_processor.py       : TripoSGAttnProcessor2_0
#   - fused_triposg_attn_processor.py : FusedTripoSGAttnProcessor2_0
#
# This module preserves the previous public API so existing
# ``from .attention_processor_v2 import ...`` imports keep working.

from .attention_rope_utils import (
    apply_rotary_emb_1d_on_frame,
    precompute_freqs_cis,
)
from .flash_triposg_attn_processor import FlashTripoSGAttnProcessor2_0
from .fused_triposg_attn_processor import FusedTripoSGAttnProcessor2_0
from .triposg_attn_processor import TripoSGAttnProcessor2_0


__all__ = [
    "apply_rotary_emb_1d_on_frame",
    "precompute_freqs_cis",
    "FlashTripoSGAttnProcessor2_0",
    "FusedTripoSGAttnProcessor2_0",
    "TripoSGAttnProcessor2_0",
]
