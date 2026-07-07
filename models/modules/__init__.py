"""模块注册表 — Mark4 重命名"""

# v1 既有模块（保留）
from .encoder import PyramidEncoder
from .mins import MINSBlock
from .ispn import ISPN as ISPNv1
from .mspn import MSPN
from .reconstruction import FinalReconstruction

# Mark4 模块
from .tfsi import TFDE
from .tfsi_v2 import DIE
from .igrf import SGRF
from .lff import RadialBasisFilter, LFFFeatureAdapter
from .ifpn import ISPN as ISPN_legacy, IllumExtract
from .ispn_v2 import ISPN
from .ndpn import NDPN
from .mrpn import MCPN
from .pure_rwkv_sace import TCA
from .swd import WSD

# v5.6 block
from .blocks import ConvBlock, ResBlock, NAFBlock, LayerNorm2d

# v5.9 新增
from .amp_enhance import AmpEnhance, AmpNet

__all__ = [
    # v1
    "PyramidEncoder", "MINSBlock", "MSPN", "FinalReconstruction",
    # Mark4
    "TFDE", "DIE", "SGRF",
    "RadialBasisFilter", "LFFFeatureAdapter",
    "TCA",
    "ISPN", "ISPN_legacy", "IllumExtract",
    "NDPN", "MCPN",
    "WSD",
    # blocks
    "ConvBlock", "ResBlock", "NAFBlock", "LayerNorm2d",
    # v5.9
    "AmpEnhance", "AmpNet",
]
