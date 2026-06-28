"""
Spatial-Domain DWT-LFF Adapter (v6.4)
======================================
Haar DWT → 空间域低频光照分离 → IDWT 重构.

Bravo P1: alpha_init 参数化, 支持 center (α=0.6) vs neighbor (α=0.4).
"""

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
    """
    Bravo P1: alpha_init 控制初始光照分离倾向.
    center (α=0.6): 更多光照 → 作为干净参考锚点.
    neighbor (α=0.4): 更多退化 → 保留退化信号供 SACE 对齐.
    """

    def __init__(self, in_channels: int, alpha_init: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.dwt = HaarDWT2D()

        self.illum_alpha = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.Sigmoid(),
        )

        self.norm_sace = nn.LayerNorm(in_channels)
        self.norm_tfsi = nn.LayerNorm(in_channels)

        # 偏置初始化为 alpha_init 的控制值
        # Sigmoid(x + b) ≈ Sigmoid(b) at init; b = 0 → α ≈ 0.5
        # 通过 sigmoid bias 控制: α_init 由 illim_alpha 的 Conv 权重偏置决定
        for m in self.illum_alpha.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
                # 用偏置控制初始 α: conv1x1(LL) →  sigmoid(x + bias)
                # bias = log(alpha_init / (1 - alpha_init)) 使初始 sigmoid ≈ alpha_init
                bias_val = torch.tensor(alpha_init).log() - torch.tensor(1.0 - alpha_init).log()
                if m.bias is not None:
                    nn.init.constant_(m.bias, float(bias_val))
                else:
                    nn.init.zeros_(m.weight)
                    nn.init.constant_(m.weight, 0.0)

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
            "feat_tfsi": feat_tfsi,
            "feat_sace": feat_sace,
        }
