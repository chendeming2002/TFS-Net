#!/usr/bin/env python3
"""
Branch-N (简化版): 成像噪声去除

与 Golf R2 的差异:
  - 去掉 var_map 通路 (矩阵 RWKV 的 Noise 头已捕捉噪声统计)
  - 2×NAFBlock (vs 3)
  - Upsample3x3 (仅 at_f1=False 时使用; v8 的 at_f1=True 下不建立)
  - 保留 ChannelAttention (SE-like) 提升通道选择能力

Input:  F_N  [B, C, H, W]   (矩阵 RWKV 注入的噪声上下文; C = channels, v8 用 32)
Output: Y_N  [B, 3, H, W]   (at_f1=True 时与输入同分辨率, 不做上采样)
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


class NBranch(nn.Module):
    """Branch-N (简化): Imaging Noise Denoising"""

    def __init__(self, channels: int = 128, num_blocks: int = 2,
                 out_channels: int = 3, at_f1: bool = False):
        super().__init__()
        self.channels = channels
        # at_f1=True: 输入已经是全分辨率 (F1 尺度), 不再做 2 倍上采样。
        # 这是 v8 U-Net 式重构的一部分 (见 docs/v8/05-UNet重构设计.md)。
        self.at_f1 = bool(at_f1)

        self.denoise_blocks = nn.Sequential(*[
            NAFBlock(channels) for _ in range(num_blocks)
        ])
        self.channel_attn = ChannelAttention(channels)

        if self.at_f1:
            self.upsample = None
            self.to_rgb = nn.Sequential(
                nn.Conv2d(channels, 32, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv2d(32, out_channels, 3, 1, 1, bias=True),
            )
        else:
            self.upsample = Upsample3x3(in_ch=channels, out_ch=64, scale=2)
            self.to_rgb = nn.Sequential(
                nn.Conv2d(64, 32, 3, 1, 1, bias=True),
                nn.GELU(),
                nn.Conv2d(32, out_channels, 3, 1, 1, bias=True),
            )

    def forward(self, F_N: torch.Tensor) -> dict:
        x = self.denoise_blocks(F_N)
        x = self.channel_attn(x)
        if not self.at_f1:
            x = self.upsample(x)      # [B, 64, H, W]
        Y_N = self.to_rgb(x)
        return {"Y_N": Y_N}


if __name__ == '__main__':
    m = NBranch(channels=128, num_blocks=2)
    x = torch.randn(2, 128, 32, 32)
    out = m(x)
    print(f"NBranch: {x.shape} → {out['Y_N'].shape}, "
          f"params={sum(p.numel() for p in m.parameters())/1e3:.1f}K")


# 旧名别名: 历史 run 与文档记为 BranchNSimple。
BranchNSimple = NBranch
