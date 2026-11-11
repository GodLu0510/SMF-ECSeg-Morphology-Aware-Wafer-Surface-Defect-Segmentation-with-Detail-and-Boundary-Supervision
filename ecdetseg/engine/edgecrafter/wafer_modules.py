"""Compatibility imports; preserve original graph and checkpoint names."""
from .modules.sdc import WaferDetailAdapter
from .modules.mabg import WaferMorphologyBlock
__all__ = ["WaferDetailAdapter", "WaferMorphologyBlock"]
