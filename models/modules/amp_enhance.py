"""
AmpEnhance — 图像级频域幅度增强模块 (v5.9)
=============================================
参照 FourLLIE (CVPR 2024) 的 Stage 1 设计：在输入图像层做频域幅度增强，
保留相位（结构/噪声指纹），逆转 γ_t 的幅度压缩。

数学形式:
    curve_amps = clamp(sigmoid(AmpNet(y)), 0.1, 1.0)   # 估计光照衰减图
    F = fft2(y, norm='ortho')
    mag, pha = |F|, ∠F
    mag_new = mag / curve_amps                           # 幅度除法提亮
    ỹ = ifft2(polar(mag_new, pha), norm='ortho').real   # 保留相位

物理含义:
    y = γ · (x*k + n)
    ỹ ≈ x*k + n    (逆转 γ 的幅度压缩, SNR 不变, 相位不变)

参考: reference_repos/FourLLIE/models/archs/FourLLIE.py L57-69
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.fft as fft


class AmpNet(nn.Module):
    """从输入图像估计幅度变换图 curve_amps ∈ (0, 1]。

    轻量设计：3→16→16→3，两层 3×3 conv + GELU + 1×1 conv + sigmoid。
    参数量约 5K，远小于 Encoder/IFPN。
    """

    def __init__(self, in_channels: int = 3, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, in_channels, 1, 1, 0, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class AmpEnhance(nn.Module):
    """图像级频域幅度增强 — 保留相位，幅度除以 curve_amps。

    Args:
        in_channels: 图像通道数 (默认 3)
        hidden: AmpNet 隐藏通道 (默认 16)
        min_amps: curve_amps 下界，限制最大放大倍数 (默认 0.1 → 10×)

    Forward:
        Input:  y (B, C, H, W) 低光输入图像
        Output: ỹ (B, C, H, W) 幅度增强后的图像
    """

    def __init__(self, in_channels: int = 3, hidden: int = 16, min_amps: float = 0.1):
        super().__init__()
        self.min_amps = min_amps
        self.amp_net = AmpNet(in_channels=in_channels, hidden=hidden)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        B, C, H, W = y.shape

        # Step 1: 从原图估计 curve_amps
        curve_amps = self.amp_net(y)
        curve_amps = curve_amps.clamp(min=self.min_amps, max=1.0)

        # Step 2: FFT
        F = fft.fft2(y, dim=(-2, -1), norm='ortho')
        mag = torch.abs(F)
        pha = torch.angle(F)

        # Step 3: 幅度除法提亮 (保留相位)
        mag_new = mag / curve_amps

        # Step 4: IFFT (保留相位)
        F_new = torch.polar(mag_new, pha)
        y_enhanced = fft.ifft2(F_new, dim=(-2, -1), norm='ortho').real

        return y_enhanced
