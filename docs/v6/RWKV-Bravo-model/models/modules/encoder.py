"""
PyramidEncoder — 3级多尺度金字塔编码器 (Bravo: 固定3级, 无4级)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, ResBlock, LayerNorm2d


class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, act=True),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True),
        )

    def forward(self, x):
        return self.block(x)


class PyramidEncoder(nn.Module):
    """
    3级金字塔编码器 [32, 64, 96] → fused_channels.
    Level 3 (96ch, H/4) 作为最粗尺度.
    """

    def __init__(self, in_channels=3, level_channels=(32, 64, 96), fused_channels=64,
                 num_bottleneck_blocks: int = 0):
        super().__init__()
        c1, c2, c3 = level_channels

        self.stage1 = EncoderStage(in_channels, c1, stride=1)
        self.stage2 = EncoderStage(c1, c2, stride=2)
        self.stage3 = EncoderStage(c2, c3, stride=2)

        self.num_bottleneck_blocks = num_bottleneck_blocks
        if num_bottleneck_blocks > 0:
            self.bottleneck = nn.Sequential(
                *[ResBlock(c3) for _ in range(num_bottleneck_blocks)]
            )
        else:
            self.bottleneck = None

        self.lateral3 = nn.Conv2d(c3, fused_channels, 1, 1, 0)
        self.lateral2 = nn.Conv2d(c2, fused_channels, 1, 1, 0)
        self.lateral1 = nn.Conv2d(c1, fused_channels, 1, 1, 0)
        self.fuse_norm = LayerNorm2d(fused_channels)
        self.fuse = nn.Sequential(
            ConvBlock(fused_channels, fused_channels, 3, 1, 1, act=True),
            ConvBlock(fused_channels, fused_channels, 3, 1, 1, act=True),
        )

    def forward_single(self, x, return_coarse=False):
        l1 = self.stage1(x)
        l2 = self.stage2(l1)
        l3 = self.stage3(l2)

        if self.bottleneck is not None:
            l3 = self.bottleneck(l3)

        p3 = self.lateral3(l3)
        p2 = self.lateral2(l2) + F.interpolate(p3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral1(l1) + F.interpolate(p2, size=l1.shape[-2:], mode="bilinear", align_corners=False)
        p1_normed = self.fuse_norm(p1)
        fused = self.fuse(p1_normed)

        if return_coarse:
            return fused, l3
        return fused

    def forward(self, x, return_coarse=False):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        if return_coarse:
            fused, coarse = self.forward_single(x, return_coarse=True)
            _, cf, hf, wf = fused.shape
            _, cc, hc, wc = coarse.shape
            return fused.view(b, t, cf, hf, wf), coarse.view(b, t, cc, hc, wc)
        else:
            feat = self.forward_single(x, return_coarse=False)
            _, cf, hf, wf = feat.shape
            return feat.view(b, t, cf, hf, wf)
