"""
NDPN (Noise-Denoising Pyramid Network) — TFS-Net v3 → Flight10m3
==================================================================
基于 SNR 自适应聚合 + 细节保留的多帧降噪。

Flight10m3 核心变更:
  1. 提升 F_denoised (时序聚合) 为主去噪源——多帧加权已充分去噪
  2. 废除 per-frame feature subtraction（原 noise_feat * strength），改为
     conf-gated 自适应残差校正（轻量、细节保留）
  3. 新增 detail_proj：从图像梯度构建细节 map，控制残差强度
  4. 纹理区（高梯度）→ 弱校正保留细节；平坦区（低梯度）→ 允许校正
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.blocks import ConvBlock


class NDPN(nn.Module):
    def __init__(self, channels: int = 64, tau_mid_init: float = 1.0,
                 tau_scale_init: float = 1.0, num_frames: int = 5):
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

        # Correspondence confidence from C_omega diagonals
        self.conf_proj = nn.Sequential(
            nn.Linear(num_frames - 1, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.conf_proj[-2].weight)
        nn.init.zeros_(self.conf_proj[-2].bias)

        # Flight10m3: light adaptive correction — 3×3 spatial (denoise) + 1×1 pointwise (detail)
        # The 1×1 path acts as an information highway: pointwise corrections pass through
        # without spatial mixing, preserving fine textures that 3×3 would blur.
        self.corr_spatial = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        nn.init.zeros_(self.corr_spatial[-1].weight)
        nn.init.zeros_(self.corr_spatial[-1].bias)

        self.corr_pointwise = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, 1, 0),
        )
        nn.init.zeros_(self.corr_pointwise[-1].weight)
        nn.init.zeros_(self.corr_pointwise[-1].bias)

        # Flight10m4: multi-scale coarse guidance projector — l2_lat (H/2) upsampled → coarse hint
        self.coarse_proj = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels, 3, 1, 1),
        )
        nn.init.zeros_(self.coarse_proj[-1].weight)
        nn.init.zeros_(self.coarse_proj[-1].bias)

        # Flight10m3: detail projector — image gradient → (B, 1, H, W) detail map
        # Learns which gradient patterns are "texture to preserve" vs "noise to remove"
        self.detail_proj = nn.Sequential(
            nn.Conv2d(1, channels // 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

        self.gamma_raw = nn.Parameter(torch.full((1, channels, 1, 1), 0.01))

    @property
    def gamma(self):
        return self.gamma_raw.clamp(max=0.03)

    @staticmethod
    def _image_gradient(img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) → gradient magnitude (B, 1, H, W)."""
        gray = img.mean(dim=1, keepdim=True)
        gx = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()
        gy = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs()
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return gx + gy

    def forward(self, feats: torch.Tensor, F_aligned_list: List[torch.Tensor],
                mu_t_clean: torch.Tensor, sigma_t_clean: torch.Tensor,
                s_noise: torch.Tensor, center_idx: int,
                C_omega_list: list = None, F_t_aligned: torch.Tensor = None,
                image_center: torch.Tensor = None,
                l2_feats: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = feats.shape
        eps = 1e-6

        # Step 1: SNR estimation
        signal = mu_t_clean.abs().mean(dim=1, keepdim=True)
        noise = sigma_t_clean.mean(dim=1, keepdim=True)
        snr_hat = signal / (noise + eps)
        tau_scale = torch.exp(self.log_tau_scale).clamp(min=1e-2)
        s_snr = torch.sigmoid((snr_hat - self.tau_mid) / tau_scale)

        # Step 2: C_omega confidence map
        conf_map = None
        if C_omega_list is not None and len(C_omega_list) > 0:
            diag_scores = []
            for C_t in C_omega_list:
                diag = C_t.diagonal(dim1=-2, dim2=-1)
                diag_scores.append(diag)
            diag_stack = torch.stack(diag_scores, dim=-1)
            conf_raw = self.conf_proj(diag_stack).squeeze(-1)
            ds = int(conf_raw.shape[-1] ** 0.5)
            conf_map = conf_raw.reshape(B, 1, ds, ds)
            conf_map = F.interpolate(conf_map, size=(H, W), mode='bilinear', align_corners=False)

        # Step 3: Multi-frame SNR-weighted temporal aggregation
        F_ref = F_t_aligned if F_t_aligned is not None else feats[:, center_idx]
        alphas = []
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
            F_denoised = F_denoised + (alphas[i] / alpha_sum) * F_aligned_list[i]
        F_denoised = self.refine(F_denoised)

        # Flight10m3: F_denoised IS the primary denoised output.
        # The correction below only makes small, detail-aware adjustments.

        # Step 4: Detail map from image gradient
        f_enc = feats[:, center_idx]
        if image_center is not None:
            detail_raw = self._image_gradient(image_center)
            detail_map = self.detail_proj(detail_raw)
        else:
            detail_map = torch.zeros(B, 1, H, W, device=feats.device)

        # Step 5: Multi-scale correction — spatial (3×3) + pointwise (1×1) + coarse (l2)
        correction_input = torch.cat([f_enc, F_denoised, detail_map], dim=1)
        corr_spatial = self.corr_spatial(correction_input)
        corr_pointwise = self.corr_pointwise(correction_input)
        correction = (corr_spatial + corr_pointwise) * self.gamma * (1.0 - detail_map)

        # Multi-scale coarse hint from l2_lat (H/2)
        if l2_feats is not None:
            l2_center = l2_feats[:, center_idx, :, :, :]
            coarse_hint = self.coarse_proj(F.interpolate(l2_center, size=(H, W),
                                      mode='bilinear', align_corners=False))
            correction = correction + coarse_hint * self.gamma * 0.5

        if conf_map is not None:
            correction = correction * conf_map

        # Step 6: Final output — temporal base + light correction + s_noise injection
        f_noise_out = F_denoised + correction + self.noise_proj(s_noise)

        return {
            "f_noise_out": f_noise_out,
            "s_snr":       s_snr,
            "snr_hat":     snr_hat,
        }
