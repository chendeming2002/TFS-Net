"""
PyramidEncoder — 多尺度金字塔编码器
=====================================

v3 修改：新增 return_coarse 参数，支持返回最粗尺度特征 F_t^(L) 供 IFPN 双流使用。

当前假设（待确认，详见 v3quest.md § 1.4.4）：
  - coarse_feat = l3（stage3 直接输出，96 通道，H/4×W/4）
  - 注：当前 PyramidEncoder 两次 stride=2 下采样，l3 分辨率为 H/4×W/4
  - v3 设计文档写 H/8×W/8，与当前编码器结构不一致（需额外 stage 或修改下采样）
  - 而非 p3（lateral 投影后，48 通道，H×W 全分辨率）

向后兼容：return_coarse=False 时保持与 v1 MINSNet 完全一致的接口。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock


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
    def __init__(self, in_channels=3, level_channels=(32, 64, 96), fused_channels=48):
        super().__init__()
        c1, c2, c3 = level_channels
        self.stage1 = EncoderStage(in_channels, c1, stride=1)
        self.stage2 = EncoderStage(c1, c2, stride=2)
        self.stage3 = EncoderStage(c2, c3, stride=2)

        self.lateral3 = nn.Conv2d(c3, fused_channels, 1, 1, 0)
        self.lateral2 = nn.Conv2d(c2, fused_channels, 1, 1, 0)
        self.lateral1 = nn.Conv2d(c1, fused_channels, 1, 1, 0)
        self.fuse = nn.Sequential(
            ConvBlock(fused_channels, fused_channels, 3, 1, 1, act=True),
            ConvBlock(fused_channels, fused_channels, 3, 1, 1, act=True),
        )

    def forward_single(self, x, return_coarse=False):
        """
        Args:
            x             : (B, C, H, W) 单帧图像
            return_coarse : 是否同时返回最粗尺度特征 l3

        Returns:
            return_coarse=False: fused_feat (B, C_f, H, W)
            return_coarse=True : (fused_feat, coarse_feat) 其中 coarse_feat = l3 (B, c3, H/8, W/8)
        """
        l1 = self.stage1(x)       # (B, c1, H, W)
        l2 = self.stage2(l1)      # (B, c2, H/2, W/2)
        l3 = self.stage3(l2)      # (B, c3, H/4, W/4)  ← 注：两次 stride=2 → H/4

        p3 = self.lateral3(l3)
        p2 = self.lateral2(l2) + F.interpolate(p3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral1(l1) + F.interpolate(p2, size=l1.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(p1)     # (B, C_f, H, W)

        if return_coarse:
            # 当前假设：coarse_feat = l3（最粗尺度，未经 lateral 投影）
            # 待 IFPN 设计澄清后可能改为 p3 或其他尺度
            return fused, l3
        return fused

    def forward(self, x, return_coarse=False):
        """
        Args:
            x             : (B, T, C, H, W) 多帧序列
            return_coarse : 是否同时返回最粗尺度特征

        Returns:
            return_coarse=False: feats (B, T, C_f, H_f, W_f)
            return_coarse=True : (feats, coarse_feats)
                                 feats        : (B, T, C_f, H, W)
                                 coarse_feats : (B, T, c3, H/4, W/4)
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        if return_coarse:
            fused, coarse = self.forward_single(x, return_coarse=True)
            _, cf, hf, wf = fused.shape
            _, cc, hc, wc = coarse.shape
            fused = fused.view(b, t, cf, hf, wf)
            coarse = coarse.view(b, t, cc, hc, wc)
            return fused, coarse
        else:
            feat = self.forward_single(x, return_coarse=False)
            _, cf, hf, wf = feat.shape
            return feat.view(b, t, cf, hf, wf)
