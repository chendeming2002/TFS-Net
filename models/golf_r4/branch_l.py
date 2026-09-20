"""
Golf Branch-L: 光照校正 (修复版)
==================================
与 Foxtrot 的区别:
  1. 上采样改为 resize-conv (消除棋盘格)
  2. F1 skip 提供原分辨率细节，光照图在全分辨率精化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.blocks import NAFBlock, LayerNorm2d
from .upsample import UpsampleBlock


class BranchL(nn.Module):
    """Branch-L: Illumination Correction via Retinex (Golf)

    Input:
        F_L:  (B, C, H/2, W/2) — TCA 光照分量
        X_t:  (B, 3, H, W)     — 中心帧原图 [0,1]
        skip: (B, C1, H, W)    — 可选 F1 skip

    Output dict: Y_L, L_t, R_t
    """

    def __init__(self, channels: int = 128, num_blocks: int = 2,
                 gamma_init: float = 2.0, out_channels: int = 3,
                 skip_channels: int = 32):
        super().__init__()
        self.channels = channels

        # 特征细化
        self.refine_blocks = nn.Sequential(*[
            NAFBlock(channels) for _ in range(num_blocks)
        ])
        self.refine_norm = LayerNorm2d(channels)

        # 低分辨率光照图预测
        self.illum_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 3, 1, 1, bias=True),
        )
        nn.init.zeros_(self.illum_head[-1].weight)
        nn.init.zeros_(self.illum_head[-1].bias)

        # 上采样: resize-conv + F1 skip
        self.upsample = UpsampleBlock(
            in_channels=channels,
            out_channels=channels,
            skip_channels=skip_channels,
            norm=True,
        )

        # 全分辨率光照图修正
        self.illum_refine = nn.Conv2d(channels, 1, 3, 1, 1, bias=True)
        nn.init.zeros_(self.illum_refine.weight)
        nn.init.zeros_(self.illum_refine.bias)

        # 可学习 gamma
        self.log_gamma = nn.Parameter(torch.tensor(float(gamma_init)).log())

        # 残差补偿
        self.residual_conv = nn.Conv2d(channels, out_channels, 3, 1, 1, bias=True)
        nn.init.zeros_(self.residual_conv.weight)
        nn.init.zeros_(self.residual_conv.bias)

    def forward(self, F_L: torch.Tensor, X_t: torch.Tensor,
                skip: torch.Tensor = None) -> dict:
        B, C, H_ds, W_ds = F_L.shape
        H, W = X_t.shape[-2:]

        x = self.refine_norm(self.refine_blocks(F_L))

        # 低分辨率光照图
        L_ds = torch.sigmoid(self.illum_head(x))  # (B,1,H/2,W/2)

        # resize-conv 上采样 + F1 skip
        x_up = self.upsample(x, skip)             # (B,C,H,W)

        # 全分辨率光照融合
        L_res = self.illum_refine(x_up)
        L_raw = F.interpolate(L_ds, size=(H, W), mode='bilinear', align_corners=False)
        L_t = torch.sigmoid(L_raw + L_res)

        # Retinex
        eps = 1e-4
        R_t = (X_t / L_t.clamp(min=eps)).clamp(max=10.0)
        gamma = self.log_gamma.exp()
        Y_L = R_t * L_t.pow(gamma)
        Y_L = Y_L + self.residual_conv(x_up)
        Y_L = Y_L.clamp(0.0, 1.0)

        return {"Y_L": Y_L, "L_t": L_t, "R_t": R_t}
