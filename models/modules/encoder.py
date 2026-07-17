"""
PyramidEncoder — 多尺度金字塔编码器
====================================

v5.5: 默认 3 级金字塔 [32, 64, 96]，移除 4 级配置。
      最粗尺度 l3 (96ch, H/4×W/4)，coarse_channels=96。
      4 级配置仍通过 level_channels 长度 4 向后兼容，但不再推荐。

v3 修改：新增 return_coarse 参数，支持返回最粗尺度特征 F_t^(L) 供 IFPN 双流使用。

向后兼容：return_coarse=False 时保持与 v1 MINSNet 完全一致的接口。
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
    def __init__(self, in_channels=3, level_channels=(32, 64, 96), fused_channels=64,
                 num_bottleneck_blocks: int = 0):
        super().__init__()
        if len(level_channels) == 4:
            c1, c2, c3, c4 = level_channels
        else:
            c1, c2, c3 = level_channels
            c4 = None

        self.stage1 = EncoderStage(in_channels, c1, stride=1)
        self.stage2 = EncoderStage(c1, c2, stride=2)
        self.stage3 = EncoderStage(c2, c3, stride=2)
        self.has_stage4 = c4 is not None
        if self.has_stage4:
            self.stage4 = EncoderStage(c3, c4, stride=2)
            self.lateral4 = nn.Conv2d(c4, fused_channels, 1, 1, 0)

        # v5.8: 瓶颈块 — 在最粗层 (c3, H/4) 增加深度, 最省显存
        self.num_bottleneck_blocks = num_bottleneck_blocks
        if num_bottleneck_blocks > 0:
            bottleneck_ch = c4 if c4 is not None else c3
            self.bottleneck = nn.Sequential(
                *[ResBlock(bottleneck_ch) for _ in range(num_bottleneck_blocks)]
            )
        else:
            self.bottleneck = None

        self.lateral3 = nn.Conv2d(c3, fused_channels, 1, 1, 0)
        self.lateral2 = nn.Conv2d(c2, fused_channels, 1, 1, 0)
        self.lateral1 = nn.Conv2d(c1, fused_channels, 1, 1, 0)
        # v5.9.1: fuse 前加 LayerNorm2d 控制值域
        # 根因: lateral 累加导致 p1 值域 ±1800 → fuse conv 输出 ±12000 → GELU(全负)=0 → 特征死亡
        # LayerNorm2d 把 p1 归一化到均值 0 方差 1 → fuse conv 正常工作 → GELU 不饱和
        self.fuse_norm = LayerNorm2d(fused_channels)
        self.fuse = nn.Sequential(
            ConvBlock(fused_channels, fused_channels, 3, 1, 1, act=True),
            ConvBlock(fused_channels, fused_channels, 3, 1, 1, act=True),
        )

    def forward_single(self, x, return_coarse=False):
        """
        Args:
            x             : (B, C, H, W) 单帧图像
            return_coarse : 是否同时返回最粗尺度特征

        Returns:
            return_coarse=False: fused_feat (B, C_f, H, W)
            return_coarse=True : (fused_feat, coarse_feat)
                                 4-stage: coarse = l4 (B, c4, H/8, W/8)
                                 3-stage: coarse = l3 (B, c3, H/4, W/4)
        """
        l1 = self.stage1(x)       # (B, c1, H, W)
        l2 = self.stage2(l1)      # (B, c2, H/2, W/2)
        l3 = self.stage3(l2)      # (B, c3, H/4, W/4)

        if self.has_stage4:
            l4 = self.stage4(l3)  # (B, c4, H/8, W/8)
            # v5.8: 瓶颈块在最粗层
            if self.bottleneck is not None:
                l4 = self.bottleneck(l4)
            p4 = self.lateral4(l4)
            p3 = self.lateral3(l3) + F.interpolate(p4, size=l3.shape[-2:], mode="bilinear", align_corners=False)
        else:
            l4 = None
            # v5.8: 瓶颈块在最粗层 (3级编码器时在 l3)
            if self.bottleneck is not None:
                l3 = self.bottleneck(l3)
            p3 = self.lateral3(l3)

        p2 = self.lateral2(l2) + F.interpolate(p3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral1(l1) + F.interpolate(p2, size=l1.shape[-2:], mode="bilinear", align_corners=False)
        # v5.9.1: fuse 前归一化, 防止 lateral 累加值域爆炸导致 GELU 饱和
        p1_normed = self.fuse_norm(p1)
        fused = self.fuse(p1_normed)     # (B, C_f, H, W)

        if return_coarse:
            coarse = l4 if self.has_stage4 else l3
            return fused, coarse
        return fused

    def forward_single_lateral(self, x):
        """Flight8: return l1_lat, l2_lat, l3_lat — skip FPN fusion for downstream modules."""
        l1 = self.stage1(x)
        l2 = self.stage2(l1)
        l3 = self.stage3(l2)
        if self.bottleneck is not None:
            l3 = self.bottleneck(l3)
        return self.lateral1(l1), self.lateral2(l2), self.lateral3(l3)

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
