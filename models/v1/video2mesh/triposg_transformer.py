# triposg_transformer_0821
#
# Re-export shim. The original module has been split into per-class files:
#   - dit_block.py          : DiTBlock
#   - triposg_dit_model.py  : TripoSGDiTModel4D
#
# This module preserves the previous public API so existing
# ``from .triposg_transformer import ...`` imports keep working.

from .dit_block import DiTBlock
from .triposg_dit_model import TripoSGDiTModel4D


__all__ = [
    "DiTBlock",
    "TripoSGDiTModel4D",
]
