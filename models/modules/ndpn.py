"""
NDPN (Noise-Denoising Pyramid Network) — TFS-Net v3
=====================================================
基于 SNR 自适应聚合的多帧降噪。

实现状态: ✅ 完整实现

数据流:
    1. SNR 估计: s_SNR = sigmoid((SNR_hat - τ_mid) / τ_scale)
    2. 双因素权重: α_i = sigmoid(Conv(残差)) * (1 - s_SNR), α_t = s_SNR
    3. 加权聚合: F_denoised = Σ w_i * F_i^aligned
    4. 强度门控: f_noise_out = s_noise * F_denoised + (1-s_noise) * F_t
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.blocks import ConvBlock


class NDPN(nn.Module):
    """
    Args:
        channels       : 特征通道数 (默认 64)
        tau_mid_init   : SNR 归一化中心初始值
        tau_scale_init : SNR 归一化尺度初始值
    """

    def __init__(
        self,
        channels: int = 64,
        tau_mid_init: float = 1.0,
        tau_scale_init: float = 1.0,
    ):
        super().__init__()
        self.channels = channels

        self.tau_mid = nn.Parameter(torch.tensor(tau_mid_init))
        self.log_tau_scale = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(tau_scale_init)))))

        self.alpha_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.alpha_conv[-1].weight)
        nn.init.zeros_(self.alpha_conv[-1].bias)

        self.refine = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1, act=True),
            ConvBlock(channels, channels, kernel_size=1, stride=1, padding=0, act=False),
        )

    def forward(
        self,
        feats: torch.Tensor,
        F_aligned_list: List[torch.Tensor],
        mu_t_clean: torch.Tensor,
        sigma_t: torch.Tensor,
        s_noise: torch.Tensor,
        center_idx: int,
    ) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = feats.shape
        assert len(F_aligned_list) == T
        assert C == self.channels

        # Step 1: SNR 估计
        eps = 1e-6
        signal = mu_t_clean.abs().mean(dim=1, keepdim=True)
        noise = sigma_t.mean(dim=1, keepdim=True)
        snr_hat = signal / (noise + eps)

        tau_scale = torch.exp(self.log_tau_scale).clamp(min=1e-2)
        s_snr = torch.sigmoid((snr_hat - self.tau_mid) / tau_scale)

        # Step 2: 各帧权重
        F_t = feats[:, center_idx]
        alphas: List[torch.Tensor] = []

        for i in range(T):
            F_i_aligned = F_aligned_list[i]
            if i == center_idx:
                alpha_i = s_snr
            else:
                resid = (F_i_aligned - F_t).abs()
                alpha_raw = torch.sigmoid(self.alpha_conv(resid))
                alpha_i = alpha_raw * (1.0 - s_snr)
            alphas.append(alpha_i)

        # Step 3: 归一化加权聚合
        alpha_sum = torch.stack(alphas, dim=1).sum(dim=1) + eps

        F_denoised = torch.zeros_like(F_t)
        for i in range(T):
            w_i = alphas[i] / alpha_sum
            F_denoised = F_denoised + w_i * F_aligned_list[i]

        F_denoised = self.refine(F_denoised)

        # 移除双重门控：直接输出去噪特征（由 IGRF 统一做强度加权）
        f_noise_out = F_denoised

        return {
            "f_noise_out": f_noise_out,
            "s_snr":       s_snr,
            "snr_hat":     snr_hat,
        }
