"""
Golf Branch-N: 成像噪声去除 (修复版)
======================================
与 Foxtrot 的区别:
  1. 上采样改为 resize-conv (消除棋盘格)
  2. 可选接收 F1 高分辨率 skip (原始分辨率细节锚点)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.blocks import NAFBlock, LayerNorm2d
from .upsample import UpsampleBlock


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


class BranchN(nn.Module):
    """Branch-N: Imaging Noise Denoising (Golf)

    Input:
        F_N:     (B, C, H/2, W/2) — TCA 噪声分量
        var_map: (B, 1, H/2, W/2) — 帧间方差图
        F1:      (B, C1, H, W)    — 可选, 编码器全分辨率 skip

    Output dict: Y_N (B,3,H,W), sigma_map (B,1,H,W)
    """

    def __init__(self, channels: int = 128, num_blocks: int = 3,
                 out_channels: int = 3, skip_channels: int = 32):
        super().__init__()
        self.channels = channels
        self.skip_channels = skip_channels

        # 方差图投影
        self.var_proj = nn.Sequential(
            nn.Conv2d(1, channels // 4, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 3, 1, 1, bias=True),
        )

        # 融合 F_N + var_map
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=True),
            LayerNorm2d(channels),
        )

        # NAFBlock 去噪
        self.denoise_blocks = nn.Sequential(*[
            NAFBlock(channels) for _ in range(num_blocks)
        ])

        # Channel Attention
        self.channel_attn = ChannelAttention(channels)

        # 上采样: resize-conv + F1 skip (修复棋盘格 + 恢复细节)
        self.upsample = UpsampleBlock(
            in_channels=channels,
            out_channels=channels,
            skip_channels=skip_channels,
            norm=True,
        )

        # 输出投影
        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels // 2, out_channels, 3, 1, 1, bias=True),
        )

    def forward(self, F_N: torch.Tensor, var_map: torch.Tensor = None,
                skip: torch.Tensor = None) -> dict:
        B, C, H_ds, W_ds = F_N.shape
        if var_map is None:
            var_map = torch.zeros(B, 1, H_ds, W_ds, device=F_N.device, dtype=F_N.dtype)

        var_feat = self.var_proj(var_map)
        fused = torch.cat([F_N, var_feat], dim=1)
        x = self.fuse(fused)

        x = self.denoise_blocks(x)
        x = self.channel_attn(x)

        # resize-conv 上采样 + F1 skip
        x = self.upsample(x, skip)

        Y_N = self.to_rgb(x)
        sigma_map = F.interpolate(var_map, scale_factor=2, mode='bilinear', align_corners=False)
        return {"Y_N": Y_N, "sigma_map": sigma_map}
