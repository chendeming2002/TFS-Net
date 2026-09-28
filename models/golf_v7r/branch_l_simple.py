#!/usr/bin/env python3
"""
Branch-L (简化版): 光照校正 (Retinex)

与 Golf R2 的差异:
  - 去掉 illum_refine 全分辨率修正 (低分辨率 L 直接 bilinear 到全分辨率)
  - 2×NAFBlock (保持)
  - Upsample3x3 (vs UpsampleBlock + F1 skip)
  - 保留 Retinex 物理先验 (L_t, R_t, learnable gamma) ⭐
  - 保留 residual_conv 残差补偿

Input:  F_L  [B, 128, H/2, W/2]  (矩阵 RWKV 注入的光照上下文)
        X_t  [B, 3, H, W]        (中心帧原图 [0,1])
Output: Y_L, L_t, R_t
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.blocks import NAFBlock, LayerNorm2d
from models.golf_v7r.upsample import Upsample3x3


class BranchLSimple(nn.Module):
    """Branch-L (简化): Illumination Correction via Retinex"""

    def __init__(self, channels: int = 128, num_blocks: int = 2,
                 gamma_init: float = 2.0, out_channels: int = 3):
        super().__init__()
        self.channels = channels

        self.refine_blocks = nn.Sequential(*[
            NAFBlock(channels) for _ in range(num_blocks)
        ])
        self.refine_norm = LayerNorm2d(channels)

        # 低分辨率光照图预测 (零初始化 → 初始 sigmoid(0)=0.5)
        self.illum_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 3, 1, 1, bias=True),
        )
        nn.init.zeros_(self.illum_head[-1].weight)
        nn.init.zeros_(self.illum_head[-1].bias)

        # 上采样
        self.upsample = Upsample3x3(in_ch=channels, out_ch=64, scale=2)

        # 可学习 gamma
        self.log_gamma = nn.Parameter(torch.tensor(float(gamma_init)).log())

        # 残差补偿 (零初始化)
        self.residual_conv = nn.Conv2d(64, out_channels, 3, 1, 1, bias=True)
        nn.init.zeros_(self.residual_conv.weight)
        nn.init.zeros_(self.residual_conv.bias)

    def forward(self, F_L: torch.Tensor, X_t: torch.Tensor) -> dict:
        B, C, H_ds, W_ds = F_L.shape
        H, W = X_t.shape[-2:]

        x = self.refine_norm(self.refine_blocks(F_L))

        # 低分辨率光照图
        L_ds = torch.sigmoid(self.illum_head(x))  # [B,1,H/2,W/2]

        # 上采样特征
        x_up = self.upsample(x)                    # [B,64,H,W]

        # 全分辨率光照 (直接 bilinear, 无 illum_refine)
        L_t = F.interpolate(L_ds, size=(H, W), mode='bilinear', align_corners=False)

        # Retinex
        eps = 1e-4
        R_t = (X_t / L_t.clamp(min=eps)).clamp(max=10.0)
        gamma = self.log_gamma.exp()
        Y_L = R_t * L_t.pow(gamma)
        Y_L = Y_L + self.residual_conv(x_up)
        Y_L = Y_L.clamp(0.0, 1.0)

        return {"Y_L": Y_L, "L_t": L_t, "R_t": R_t}


if __name__ == '__main__':
    m = BranchLSimple(channels=128, num_blocks=2)
    F_L = torch.randn(2, 128, 32, 32)
    X_t = torch.rand(2, 3, 64, 64)
    out = m(F_L, X_t)
    print(f"BranchLSimple: F_L={F_L.shape}, X_t={X_t.shape}")
    print(f"  Y_L={out['Y_L'].shape}, L_t={out['L_t'].shape}, R_t={out['R_t'].shape}")
    print(f"  params={sum(p.numel() for p in m.parameters())/1e3:.1f}K")
