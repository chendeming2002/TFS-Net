"""
NDPN (Noise-Denoising Pyramid Network) — TFS-Net v3 → v6 Charlie
==================================================================
基于 SNR 自适应聚合的多帧降噪。

Charlie P0-2: s_noise 条件输入 (已实施)
    通过 noise_proj (Conv2d) 投影 s_noise 到特征空间,
    以加法方式注入去噪特征, 替代 IGRF 中的直接注入.
    IGRF Stage1 在 charlie_mode 下不再接收 s_noise.
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
        num_frames: int = 5,
    ):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames

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
        self.noise_proj = nn.Conv2d(1, channels, 1, 1, 0)
        nn.init.zeros_(self.noise_proj.weight)
        nn.init.zeros_(self.noise_proj.bias)

        # Delta: correspondence confidence projector (from C_omega diagonals)
        self.conf_proj = nn.Sequential(
            nn.Linear(num_frames - 1, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, 1),
            nn.Sigmoid(),
        )

        # Delta: noise extraction (encoder feat vs aligned feat difference)
        self.noise_extract = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

        # Delta: confidence-guided denoising strength
        self.denoise_strength = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(
        self,
        feats: torch.Tensor,
        F_aligned_list: List[torch.Tensor],
        mu_t_clean: torch.Tensor,
        sigma_t_clean: torch.Tensor,
        s_noise: torch.Tensor,
        center_idx: int,
        C_omega_list: list = None,
        F_t_aligned: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = feats.shape
        assert len(F_aligned_list) == T
        assert C == self.channels

        # Step 1: SNR 估计
        eps = 1e-6
        signal = mu_t_clean.abs().mean(dim=1, keepdim=True)
        noise = sigma_t_clean.mean(dim=1, keepdim=True)
        snr_hat = signal / (noise + eps)
        tau_scale = torch.exp(self.log_tau_scale).clamp(min=1e-2)
        s_snr = torch.sigmoid((snr_hat - self.tau_mid) / tau_scale)

        # Delta: correspondence confidence from C_omega_list
        conf_map = None
        if C_omega_list is not None and len(C_omega_list) > 0:
            diag_scores = []
            for C_t in C_omega_list:
                diag = C_t.diagonal(dim1=-2, dim2=-1)
                diag_scores.append(diag)
            diag_stack = torch.stack(diag_scores, dim=-1)  # (B, N, T-1)
            conf_raw = self.conf_proj(diag_stack).squeeze(-1)  # (B, N)
            ds = int(conf_raw.shape[-1] ** 0.5)
            conf_map = conf_raw.reshape(B, 1, ds, ds)
            conf_map = F.interpolate(conf_map, size=(H, W), mode='bilinear',
                                     align_corners=False)

        # Step 2: 各帧权重
        F_ref = F_t_aligned if F_t_aligned is not None else feats[:, center_idx]
        alphas: List[torch.Tensor] = []

        for i in range(T):
            F_i_aligned = F_aligned_list[i]
            if i == center_idx:
                alpha_i = s_snr
            else:
                resid = (F_i_aligned - F_ref).abs()
                alpha_raw = torch.sigmoid(self.alpha_conv(resid))
                alpha_i = alpha_raw * (1.0 - s_snr)
            alphas.append(alpha_i)

        alpha_sum = torch.stack(alphas, dim=1).sum(dim=1) + eps

        F_denoised = torch.zeros_like(F_ref)
        for i in range(T):
            w_i = alphas[i] / alpha_sum
            F_denoised = F_denoised + w_i * F_aligned_list[i]

        F_denoised = self.refine(F_denoised)

        # Delta: noise extraction from encoder vs aligned reference
        f_enc = feats[:, center_idx]
        noise_feat = self.noise_extract(torch.cat([f_enc, F_ref], dim=1))

        # Delta: confidence-guided denoising strength
        if conf_map is not None:
            strength = self.denoise_strength(torch.cat([noise_feat, conf_map], dim=1))
        else:
            strength = self.denoise_strength(torch.cat([noise_feat, torch.zeros_like(noise_feat[:, :1])], dim=1))

        # Delta: gamma-scaled denoising
        f_noise_out = f_enc - noise_feat * strength * self.gamma + self.noise_proj(s_noise)

        return {
            "f_noise_out": f_noise_out,
            "s_snr":       s_snr,
            "snr_hat":     snr_hat,
        }
