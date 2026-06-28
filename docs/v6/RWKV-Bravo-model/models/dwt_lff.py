"""
HaarDWT2D + SpatialDWTLFFAdapter — RWKV-Bravo 空域小波光场适配器
============================================================
Bravo 设计:
  - 空间域 Conv3x3(LL) → α soft 分配 (非 FFT 域)
  - IDWT(LL_ref/LL_deg, 0.5·LH/HL/HH) → 完美重构
  - alpha_init 参数化: 中心帧 α=0.6, 邻居帧 α=0.4
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2D(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        x01, x02 = x[:, :, 0::2, :], x[:, :, 1::2, :]
        L = (x01 + x02) * 0.5
        H_ = (x01 - x02) * 0.5
        LL = (L[:, :, :, 0::2] + L[:, :, :, 1::2]) * 0.5
        LH = (L[:, :, :, 0::2] - L[:, :, :, 1::2]) * 0.5
        HL = (H_[:, :, :, 0::2] + H_[:, :, :, 1::2]) * 0.5
        HH = (H_[:, :, :, 0::2] - H_[:, :, :, 1::2]) * 0.5
        return LL, LH, HL, HH

    def inverse(self, LL, LH, HL, HH):
        B, C, H2, W2 = LL.shape
        L = torch.zeros(B, C, H2, W2 * 2, device=LL.device, dtype=LL.dtype)
        H_ = torch.zeros(B, C, H2, W2 * 2, device=LL.device, dtype=LL.dtype)
        L[:, :, :, 0::2] = (LL + LH) * 2
        L[:, :, :, 1::2] = (LL - LH) * 2
        H_[:, :, :, 0::2] = (HL + HH) * 2
        H_[:, :, :, 1::2] = (HL - HH) * 2
        x = torch.zeros(B, C, H2 * 2, W2 * 2, device=LL.device, dtype=LL.dtype)
        x[:, :, 0::2, :] = (L + H_) * 0.5
        x[:, :, 1::2, :] = (L - H_) * 0.5
        return x


class SpatialDWTLFFAdapter(nn.Module):
    def __init__(self, in_channels: int, alpha_init: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.alpha_init = alpha_init
        self.dwt = HaarDWT2D()

        self.illum_alpha = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.Sigmoid(),
        )

        self.norm_sace = nn.LayerNorm(in_channels)
        self.norm_tfsi = nn.LayerNorm(in_channels)

        for m in self.illum_alpha.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
                nn.init.constant_(m.weight, 0.0)
                init_val = math.log(max(alpha_init / max(1.0 - alpha_init, 1e-8), 1e-8))
                if m.bias is not None:
                    nn.init.constant_(m.bias, init_val)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        LL, LH, HL, HH = self.dwt(x)
        alpha = self.illum_alpha(LL)
        LL_ref = alpha * LL
        LL_deg = (1.0 - alpha) * LL
        LH_half, HL_half, HH_half = LH * 0.5, HL * 0.5, HH * 0.5

        feat_sace = self.dwt.inverse(LL_ref, LH_half, HL_half, HH_half)
        feat_tfsi = self.dwt.inverse(LL_deg, LH_half, HL_half, HH_half)

        feat_sace = self.norm_sace(feat_sace.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat_tfsi = self.norm_tfsi(feat_tfsi.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return {
            "x_out": feat_sace,
            "feat_tfsi": feat_tfsi,
            "feat_sace": feat_sace,
        }
