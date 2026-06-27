"""模块注册表 — 同时支持 v1 (MINSNet) 与 v3 (TFSNet)"""

# v1 既有模块（保留）
from .encoder import PyramidEncoder
from .mins import MINSBlock
from .ispn import ISPN
from .mspn import MSPN
from .reconstruction import FinalReconstruction

# v3 新增模块
from .tfsi import TFSI
from .igrf import IGRF
from .lff import RadialBasisFilter, LFFFeatureAdapter
from .sace import SACE, DeformableCrossAttention, OffsetMaskHead
from .ifpn import IFPN, IllumExtract
from .ndpn import NDPN
from .mrpn import MRPN

# v5.6 新增 block
from .blocks import ConvBlock, ResBlock, NAFBlock, LayerNorm2d

# v5.9 新增
from .amp_enhance import AmpEnhance, AmpNet

# v6 新增
from .cross_rwkv import CrossRWKVGate

__all__ = [
    # v1
    "PyramidEncoder", "MINSBlock", "ISPN", "MSPN", "FinalReconstruction",
    # v3
    "TFSI", "IGRF",
    "RadialBasisFilter", "LFFFeatureAdapter",
    "SACE", "DeformableCrossAttention", "OffsetMaskHead",
    "IFPN", "IllumExtract",
    "NDPN", "MRPN",
    # v5.6
    "ConvBlock", "ResBlock", "NAFBlock", "LayerNorm2d",
    # v5.9
    "AmpEnhance", "AmpNet",
]

