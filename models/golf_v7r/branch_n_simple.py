#!/usr/bin/env python3
"""
Branch-N (简化版): 成像噪声去除

与 Golf R2 的差异:
  - 去掉 var_map 通路 (矩阵 RWKV 的 Noise 头已捕捉噪声统计)
  - 2×NAFBlock (vs 3)
  - Upsample3x3 (vs UpsampleBlock + F1 skip)
  - 保留 ChannelAttention (SE-like) 提升通道选择能力

Input:  F_N  [B, 128, H/2, W/2]  (矩阵 RWKV 注入的噪声上下文)
Output: Y_N  [B, 3, H, W]
"""
import torch
import torch.nn as nn

from models.modules.blocks import NAFBlock
from models.golf_v7r.upsample import Upsample3x3


class ChannelAttention(nn.Module):
    """通道注意力 (SE-like)"""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class BranchNSimple(nn.Module):
    """Branch-N (简化): Imaging Noise Denoising"""

    def __init__(self, channels: int = 128, num_blocks: int = 2,
                 out_channels: int = 3):
        super().__init__()
        self.channels = channels

        self.denoise_blocks = nn.Sequential(*[
            NAFBlock(channels) for _ in range(num_blocks)
        ])
        self.channel_attn = ChannelAttention(channels)

        self.upsample = Upsample3x3(in_ch=channels, out_ch=64, scale=2)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, out_channels, 3, 1, 1, bias=True),
        )

    def forward(self, F_N: torch.Tensor) -> dict:
        x = self.denoise_blocks(F_N)
        x = self.channel_attn(x)
        x = self.upsample(x)          # [B, 64, H, W]
        Y_N = self.to_rgb(x)
        return {"Y_N": Y_N}


if __name__ == '__main__':
    m = BranchNSimple(channels=128, num_blocks=2)
    x = torch.randn(2, 128, 32, 32)
    out = m(x)
    print(f"BranchNSimple: {x.shape} → {out['Y_N'].shape}, "
          f"params={sum(p.numel() for p in m.parameters())/1e3:.1f}K")
