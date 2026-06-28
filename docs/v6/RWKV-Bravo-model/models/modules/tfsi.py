"""
TFSI — Temporal-Frequency Source Indicator

输出 s_illum / s_noise 双源强度图.
Bravo P3: 新增 phase_conf_head 相位置信度估计, 调制 s_noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, LayerNorm2d
from .dwt_lff import SpatialDWTLFFAdapter


class SpatialBranch(nn.Module):
    def __init__(self, channels: int, fused_channels: int, eps: float = 1e-6,
                 soft_median_tau: float = 0.1, use_soft_median: bool = True):
        super().__init__()
        self.eps = eps
        self.use_soft_median = use_soft_median
        self.soft_median_tau = soft_median_tau
        self.conv = nn.Sequential(
            ConvBlock(channels * 3, fused_channels, kernel_size=3, stride=1, padding=1, act=True),
            ConvBlock(fused_channels, fused_channels, kernel_size=3, stride=1, padding=1, act=True),
        )

    @staticmethod
    def _soft_median(x: torch.Tensor, dim: int = 1, tau: float = 0.1) -> torch.Tensor:
        with torch.no_grad():
            med = x.median(dim=dim).values.unsqueeze(dim)
        dist = (x - med).abs()
        weights = F.softmax(-dist / tau, dim=dim)
        return (weights * x).sum(dim=dim)

    def forward(self, feats: torch.Tensor) -> dict:
        if self.use_soft_median:
            mu_t = self._soft_median(feats, dim=1, tau=self.soft_median_tau)
        else:
            mu_t = feats.median(dim=1).values

        sigma_t_sq = feats.var(dim=1, unbiased=False)
        sigma_t = torch.sqrt(sigma_t_sq + self.eps)
        snr = mu_t / (sigma_t + self.eps)
        stats = torch.cat([mu_t, sigma_t, snr], dim=1)
        f_s = self.conv(stats)

        return {"F_s": f_s, "mu_t": mu_t, "sigma_t": sigma_t, "snr": snr}


class FrequencyBranch(nn.Module):
    def __init__(self, channels: int, fused_channels: int,
                 dwt_lff: SpatialDWTLFFAdapter = None):
        super().__init__()
        self.channels = channels
        self.fused_channels = fused_channels
        self.dwt_lff = dwt_lff

        if channels == fused_channels:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
        else:
            self.in_proj = nn.Conv2d(channels, fused_channels, kernel_size=1, bias=True)
            self.out_proj = nn.Identity()

        if dwt_lff is None:
            from .lff import LFFFeatureAdapter
            self.lff = LFFFeatureAdapter(channels=fused_channels, K=10, n_ang_freq=1,
                                          per_channel_rbf=False, phase_preserving=True)
        else:
            self.lff = None

    def forward(self, feats: torch.Tensor, center_idx: int) -> torch.Tensor:
        f_center = feats[:, center_idx]
        f_center = self.in_proj(f_center)
        if self.dwt_lff is not None:
            out = self.dwt_lff(f_center)
            f_f = out["feat_tfsi"]
        else:
            f_f = self.lff(f_center)
        f_f = self.out_proj(f_f)
        return f_f


class ConcatFusion(nn.Module):
    def __init__(self, fused_channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBlock(fused_channels * 2, fused_channels, kernel_size=3, stride=1, padding=1, act=True),
            ConvBlock(fused_channels, fused_channels, kernel_size=3, stride=1, padding=1, act=True),
        )

    def forward(self, f_s: torch.Tensor, f_f: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([f_s, f_f], dim=1))


class IntensityHead(nn.Module):
    """
    Bravo P3: phase_conf 调制 s_noise.
    低光视频相位受结构+运动模糊+噪声扭曲 (FDN 2025 + FourierDiff 2024).
    相位置信度越低 → s_noise 越大 (增加去噪).
    """

    def __init__(self, fused_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(fused_channels + 1, 2, 1, 1, 0)

    def forward(self, f_fused: torch.Tensor, phase_conf: torch.Tensor = None) -> dict:
        if phase_conf is not None:
            f_input = torch.cat([f_fused, phase_conf], dim=1)
        else:
            f_input = torch.cat([f_fused, torch.zeros_like(f_fused[:, :1])], dim=1)
        raw = self.conv(f_input)
        intensities = torch.sigmoid(raw)
        s_illum = intensities[:, 0:1]
        s_noise = intensities[:, 1:2]
        if phase_conf is not None:
            s_noise = s_noise * (1.0 + 0.5 * (1.0 - phase_conf))
            s_noise = s_noise.clamp(0.0, 1.0)
        return {"s_illum": s_illum, "s_noise": s_noise}


class TFSI(nn.Module):
    """
    TFSI: SpatialBranch → FrequencyBranch → ConcatFusion → IntensityHead(s_illum, s_noise)

    Bravo P3: phase_conf_head 从频域特征估计相位可靠度.
    """

    def __init__(self, channels: int = 64, fused_channels: int = 64, eps: float = 1e-6,
                 use_soft_median: bool = True, dwt_lff: SpatialDWTLFFAdapter = None):
        super().__init__()
        self.channels = channels
        self.fused_channels = fused_channels
        self.eps = eps

        self.norm = LayerNorm2d(channels)
        self.spatial_branch = SpatialBranch(channels, fused_channels, eps=eps,
                                            use_soft_median=use_soft_median)
        self.freq_branch = FrequencyBranch(
            channels=channels, fused_channels=fused_channels, dwt_lff=dwt_lff,
        )
        self.concat_fusion = ConcatFusion(fused_channels)
        self.intensity_head = IntensityHead(fused_channels)

        # Bravo P3: 相位置信度估计 — 从频域特征 (F_f) 预测相位噪声水平
        self.phase_conf_head = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(fused_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feats: torch.Tensor) -> dict:
        b, t, c, h, w = feats.shape
        center_idx = t // 2

        feats_norm = feats.view(b * t, c, h, w)
        feats_norm = self.norm(feats_norm).view(b, t, c, h, w)

        spatial_out = self.spatial_branch(feats_norm)
        f_s = spatial_out["F_s"]

        f_f = self.freq_branch(feats_norm, center_idx)

        f_fused = self.concat_fusion(f_s, f_f)

        # Bravo P3: 从频域特征估计相位置信度
        phase_conf = self.phase_conf_head(f_f)

        intensities = self.intensity_head(f_fused, phase_conf=phase_conf)

        return {
            "F_fused": f_fused,
            "F_s": f_s,
            "F_f": f_f,
            "mu_t": spatial_out["mu_t"],
            "sigma_t": spatial_out["sigma_t"],
            "snr": spatial_out["snr"],
            "s_illum": intensities["s_illum"],
            "s_noise": intensities["s_noise"],
        }
