"""
TSD-Net (Foxtrot): TSDR分解的三分支LLVE框架
====================================================
基于 docs/TSD-Foxtrot/TSD-Foxtort.md 设计

核心创新:
  1. TSDR分解理论: 时间独立性 × 空间选择性 → 三源分解 (成像噪声/光照扰动/运动伪影)
  2. TCA-RWKV: 三路结构化查询 + RWKV空间注意力 → 显式特征解耦
  3. 三分支并行处理: Branch-N(噪声)/Branch-L(光照)/Branch-M(运动) + 正交约束
  4. 自适应融合: 空间门控 + 中心帧残差锚定

作者: OpenCode
日期: 2026-09-13
"""

from .tsdnet import TSDNet
from .encoder import SharedEncoder
from .tca_rwkv import TCA_RWKV
from .branch_n import BranchN
from .branch_l import BranchL
from .branch_m import BranchM
from .fusion import AdaptiveFusion

__all__ = [
    'TSDNet',
    'SharedEncoder', 
    'TCA_RWKV',
    'BranchN',
    'BranchL', 
    'BranchM',
    'AdaptiveFusion',
]
