"""
Shared Encoder for TSD-Net (Foxtrot)
======================================
轻量级 U-Net 编码器，NAFBlock 构建，输出三层金字塔特征

设计原则:
  1. 共享权重，逐帧独立提取空间特征 (不做时序融合)
  2. 三层金字塔: {H, H/2, H/4} × {C, 2C, 4C}
  3. 避免过早耦合时序信息 — TCA 之后再引入时序建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from models.modules.blocks import NAFBlock, LayerNorm2d


class SharedEncoder(nn.Module):
    """Shared Encoder — NAFBlock-based 3-scale pyramid
    
    Args:
        in_channels: 输入通道数 (默认3 for RGB)
        out_channels_list: 三层输出通道数 (默认[32, 64, 128])
        num_blocks_per_scale: 每层的 NAFBlock 数量 (默认[2, 2, 2])
    
    forward(x): x is (B, 3, H, W), returns (F1, F2, F3):
        F1: (B, C1, H, W)       — 原分辨率
        F2: (B, C2, H/2, W/2)   — 1/2 分辨率  ← TCA 在此尺度工作
        F3: (B, C3, H/4, W/4)   — 1/4 分辨率
    """
    
    def __init__(self, in_channels: int = 3,
                 out_channels_list: List[int] = None,
                 num_blocks_per_scale: List[int] = None):
        super().__init__()
        if out_channels_list is None:
            out_channels_list = [32, 64, 128]
        if num_blocks_per_scale is None:
            num_blocks_per_scale = [2, 2, 2]
        
        C1, C2, C3 = out_channels_list
        self.C1, self.C2, self.C3 = C1, C2, C3
        
        # 入口 conv: RGB → C1
        self.stem = nn.Conv2d(in_channels, C1, 3, 1, 1, bias=True)
        
        # Scale 1: H × C1
        self.scale1_blocks = nn.Sequential(*[
            NAFBlock(C1) for _ in range(num_blocks_per_scale[0])
        ])
        self.norm1 = LayerNorm2d(C1)
        
        # Downsample 1: H → H/2, C1 → C2
        self.down1 = nn.Conv2d(C1, C2, 3, 2, 1, bias=True)
        
        # Scale 2: H/2 × C2
        self.scale2_blocks = nn.Sequential(*[
            NAFBlock(C2) for _ in range(num_blocks_per_scale[1])
        ])
        self.norm2 = LayerNorm2d(C2)
        
        # Downsample 2: H/2 → H/4, C2 → C3
        self.down2 = nn.Conv2d(C2, C3, 3, 2, 1, bias=True)
        
        # Scale 3: H/4 × C3
        self.scale3_blocks = nn.Sequential(*[
            NAFBlock(C3) for _ in range(num_blocks_per_scale[2])
        ])
        self.norm3 = LayerNorm2d(C3)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, 3, H, W) — 单帧输入
        
        Returns: (F1, F2, F3)
        """
        x = self.stem(x)
        
        # Scale 1
        f1 = self.norm1(self.scale1_blocks(x))
        
        # Scale 2
        f2 = self.norm2(self.scale2_blocks(self.down1(f1)))
        
        # Scale 3
        f3 = self.norm3(self.scale3_blocks(self.down2(f2)))
        
        return f1, f2, f3
