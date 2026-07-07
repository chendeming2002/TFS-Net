"""
DPE (Degradation Prior Estimator) — Mark4 simplified TFDE.
Replaces complex FrequencyBranch/LFF/phase_conf with multi-scale dilated convolutions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSpatialBranch(nn.Module):
    """Multi-scale spatial feature extraction via dilated convolutions.

    3×3 local (d=1) → captures noise/texture grain
    3×3 mid (d=2) → captures medium-scale illumination gradients
    3×3 wide (d=4) → captures large-scale illumination regions
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid = out_channels // 3
        wide_ch = out_channels - 2 * mid

        self.branch_local = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, 1, 1),
            nn.GELU(),
        )
        self.branch_mid = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, 1, padding=2, dilation=2),
            nn.GELU(),
        )
        self.branch_wide = nn.Sequential(
            nn.Conv2d(in_channels, wide_ch, 3, 1, padding=4, dilation=4),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_local = self.branch_local(x)
        f_mid = self.branch_mid(x)
        f_wide = self.branch_wide(x)
        return self.fuse(torch.cat([f_local, f_mid, f_wide], dim=1))


class DPE(nn.Module):
    """DPE (Degradation Prior Estimator): simplified TFDE — pure spatial.

    Args:
        channels: feature channels (default 64)
        fused_channels: internal fused feature channels
    """

    def __init__(self, channels: int = 64, fused_channels: int = 64, eps: float = 1e-6,
                 use_soft_median: bool = True, soft_median_tau: float = 0.1):
        super().__init__()
        self.eps = eps
        self.use_soft_median = use_soft_median
        self.soft_median_tau = soft_median_tau

        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)

        self.ms_branch = MultiScaleSpatialBranch(
            in_channels=channels * 3,
            out_channels=fused_channels,
        )

        self.head = nn.Conv2d(fused_channels, 2, 1)

    @staticmethod
    def _soft_median(x: torch.Tensor, dim: int = 1, tau: float = 0.1) -> torch.Tensor:
        with torch.no_grad():
            med = x.median(dim=dim).values.unsqueeze(dim)
        dist = (x - med).abs()
        weights = F.softmax(-dist / tau, dim=dim)
        return (weights * x).sum(dim=dim)

    def forward(self, feats: torch.Tensor) -> dict:
        B, T, C, H, W = feats.shape

        feats_norm = self.norm(feats.reshape(B * T, C, H, W)).reshape(B, T, C, H, W)

        if self.use_soft_median:
            mu_t = self._soft_median(feats_norm, dim=1, tau=self.soft_median_tau)
        else:
            mu_t = feats_norm.median(dim=1).values

        sigma_t_sq = feats_norm.var(dim=1, unbiased=False)
        sigma_t = torch.sqrt(sigma_t_sq + self.eps)
        snr = mu_t / (sigma_t + self.eps)

        stats = torch.cat([mu_t, sigma_t, snr], dim=1)
        f_fused = self.ms_branch(stats)

        raw = torch.sigmoid(self.head(f_fused))
        s_illum = raw[:, 0:1]
        s_noise = raw[:, 1:2]

        return {
            "F_fused": f_fused,
            "F_s": f_fused,
            "F_f": f_fused,
            "mu_t": mu_t,
            "sigma_t": sigma_t,
            "snr": snr,
            "s_illum": s_illum,
            "s_noise": s_noise,
        }
