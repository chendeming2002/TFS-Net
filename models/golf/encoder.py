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
    """Shared Encoder — 3-scale pyramid

    Args:
        in_channels: 输入通道数 (默认3 for RGB)
        out_channels_list: 三层输出通道数 (默认[32, 64, 128])
        num_blocks_per_scale: 每层的 block 数量 (默认[2, 2, 2])
        block_type: 每层使用的 block 类型, "naf" (默认, 历史行为) 或 "rwkv"。
            "rwkv" 时改用空间 RWKV block (v8 B1 实验, 见 docs/v8/01-设计.md §四)。
            两种 block 接口一致 (同形状进出), 因此可以逐尺度互换。

    forward(x): x is (B, 3, H, W), returns (F1, F2, F3):
        F1: (B, C1, H, W)       — 原分辨率
        F2: (B, C2, H/2, W/2)   — 1/2 分辨率  ← TCA 在此尺度工作
        F3: (B, C3, H/4, W/4)   — 1/4 分辨率
    """

    def __init__(self, in_channels: int = 3,
                 out_channels_list: List[int] = None,
                 num_blocks_per_scale: List[int] = None,
                 block_type: str = "naf",
                 wkv_mode: str = "legacy",
                 block_loops: int = 1):
        super().__init__()
        if out_channels_list is None:
            out_channels_list = [32, 64, 128]
        if num_blocks_per_scale is None:
            num_blocks_per_scale = [2, 2, 2]
        if block_type not in ("naf", "rwkv"):
            raise ValueError(f"block_type must be 'naf' or 'rwkv', got {block_type!r}")

        C1, C2, C3 = out_channels_list
        self.C1, self.C2, self.C3 = C1, C2, C3
        self.block_type = block_type

        def make_blocks(channels: int, count: int) -> nn.Sequential:
            """按 block_type 生成指定数量的 block。"""
            if block_type == "naf":
                return nn.Sequential(*[NAFBlock(channels) for _ in range(count)])
            # block_loops: 单个 RWKV block 内部循环的次数 (Re-WKV 的做法, 参数不变)。
            # 与 num_blocks_per_scale (堆几个独立 block, 参数 ×N) 是两个不同维度。
            from models.golf_v7r.rwkv_encoder_block import RWKVEncoderBlock
            return nn.Sequential(*[RWKVEncoderBlock(channels, wkv_mode=wkv_mode,
                                                    num_loops=block_loops)
                                   for _ in range(count)])

        # 入口 conv: RGB → C1
        self.stem = nn.Conv2d(in_channels, C1, 3, 1, 1, bias=True)

        # Scale 1: H × C1
        self.scale1_blocks = make_blocks(C1, num_blocks_per_scale[0])
        self.norm1 = LayerNorm2d(C1)

        # Downsample 1: H → H/2, C1 → C2
        self.down1 = nn.Conv2d(C1, C2, 3, 2, 1, bias=True)

        # Scale 2: H/2 × C2
        self.scale2_blocks = make_blocks(C2, num_blocks_per_scale[1])
        self.norm2 = LayerNorm2d(C2)

        # Downsample 2: H/2 → H/4, C2 → C3
        self.down2 = nn.Conv2d(C2, C3, 3, 2, 1, bias=True)

        # Scale 3: H/4 × C3
        self.scale3_blocks = make_blocks(C3, num_blocks_per_scale[2])
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
