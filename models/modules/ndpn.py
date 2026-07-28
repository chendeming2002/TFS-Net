"""
NDPN (Noise-Denoising Pyramid Network) — TFS-Net → Flight10m5
==================================================================
基于 SNR 自适应聚合 + detail-residual highway 的多帧降噪。

Flight10m5 核心变更 (vs m4):
  P0: corr_pointwise → detail_residual gate: f_enc - F_denoised → 1×1 gate → preserved
  P1: coarse_proj 输入源 → F_denoised avgpool(替代 l2_lat), +detail_map 门控
  P1: detail_map 源 → F_denoised gradient(替代 image_center, 避免噪声误判)
  P2: gamma clamp 0.03→0.1 + refine 加 skip connection
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

        # P0: spatial correction (3×3 only — 1×1 bypass removed)
        self.corr_spatial = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        nn.init.zeros_(self.corr_spatial[-1].weight)
        nn.init.zeros_(self.corr_spatial[-1].bias)

        # P0: detail residual highway — from f_enc - F_denoised
        self.detail_gate_1x1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, 1, 0),
        )
        nn.init.zeros_(self.detail_gate_1x1[-1].weight)
        nn.init.zeros_(self.detail_gate_1x1[-1].bias)

        # P1: coarse prior from F_denoised avgpool (replaces l2_lat)
        self.coarse_prior = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        nn.init.zeros_(self.coarse_prior[-1].weight)
        nn.init.zeros_(self.coarse_prior[-1].bias)

        # P1: detail projector from F_denoised gradient (replaces image_center)
        self.detail_proj = nn.Sequential(
            nn.Conv2d(1, channels // 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

        # P2: gamma clamp relaxed to 0.1
        self.gamma_raw = nn.Parameter(torch.full((1, channels, 1, 1), 0.01))

    @property
    def gamma(self):
        return self.gamma_raw.clamp(max=0.1)

    @staticmethod
    def _feature_gradient(feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, C, H, W) → gradient magnitude (B, 1, H, W)."""
        gray = feat.mean(dim=1, keepdim=True)
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

        # P2: refine with skip connection
        F_denoised = F_denoised + self.refine(F_denoised)

        # Step 4: Detail map from F_denoised gradient (P1: noiseless, not image_center)
        f_enc = feats[:, center_idx]
        detail_raw = self._feature_gradient(F_denoised)
        detail_map = self.detail_proj(detail_raw)

        # Step 5: Spatial correction + detail residual highway (P0)
        correction_input = torch.cat([f_enc, F_denoised, detail_map], dim=1)
        corr_spatial = self.corr_spatial(correction_input)
        correction = corr_spatial * self.gamma * (1.0 - detail_map)

        # P0: detail residual highway — preserve what temporal aggregation removed
        detail_residual = f_enc - F_denoised
        detail_gate = torch.sigmoid(self.detail_gate_1x1(detail_residual))
        preserved_detail = detail_residual * detail_gate * detail_map

        # P1: coarse prior from F_denoised avgpool (not l2_lat), properly gated
        coarse_pool = F.avg_pool2d(F_denoised, 2)
        coarse_hint = self.coarse_prior(F.interpolate(coarse_pool, size=(H, W),
                                        mode='bilinear', align_corners=False))
        correction = correction + coarse_hint * self.gamma * 0.5 * (1.0 - detail_map)

        if conf_map is not None:
            correction = correction * conf_map
            preserved_detail = preserved_detail * conf_map

        # Step 6: Final output
        f_noise_out = F_denoised + correction + preserved_detail + self.noise_proj(s_noise)

        return {
            "f_noise_out": f_noise_out,
            "s_snr":       s_snr,
            "snr_hat":     snr_hat,
        }
