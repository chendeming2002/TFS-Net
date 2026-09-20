"""
UpsampleBlock — 消除棋盘格伪影的上采样模块 (Golf 核心修复)
============================================================
Foxtrot 的棋盘格根源:
  三分支上采样均为 Conv2d(C, C*4, 1×1) → PixelShuffle(2)
  1×1 卷积无感受野, PixelShuffle 把 4 个通道机械拆到 2×2 子像素网格,
  相邻输出像素之间零信息交流 → 2px 周期棋盘格 (实测自相关峰=2px)
  这是结构性缺陷, 再训练无法消除.

Golf 方案 (Odena et al. 2016 resize-convolution, 反棋盘格标准做法):
  1. F.interpolate(bilinear, ×2) — 空间平滑, 无网格伪影
  2. 3×3 conv (接 skip concat) — 提供感受野/通道混合
  3. 3×3 conv — 细化

关于 anti-aliasing: 双线性上采样后接卷积在频域上是低通+可学习增强,
不会在 Nyquist 频率产生能量尖峰 (即棋盘格).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.blocks import LayerNorm2d


class UpsampleBlock(nn.Module):
    """Resize-conv 上采样: bilinear ×2 → [concat skip] → 3×3 conv → GELU → 3×3 conv.

    Args:
        in_channels:   输入特征通道
        out_channels:  输出特征通道
        skip_channels: 高分辨率 skip 特征通道 (Golf: F1 编码器特征; 0=无 skip)
        norm:          是否加 LayerNorm2d

    Input:
        x:    (B, in_channels, H, W)
        skip: (B, skip_channels, 2H, 2W) 可选 — 高分辨率细节锚点

    Output: (B, out_channels, 2H, 2W)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 skip_channels: int = 0, norm: bool = True):
        super().__init__()
        self.skip_channels = skip_channels
        conv_in = in_channels + skip_channels
        self.conv1 = nn.Conv2d(conv_in, out_channels, 3, 1, 1, bias=True)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)
        self.norm = LayerNorm2d(out_channels) if norm else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        # 1. 双线性上采样 (无棋盘格)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # 2. 拼接高分辨率 skip (F1 提供原始分辨率细节)
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        # 3. 两次 3×3 卷积 (感受野混合, 消除子像素隔离)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.norm(x)
