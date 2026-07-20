"""
DPE (Degradation Prior Estimator) — Flight9: single-scale softplus + spatial anti-collapse.

Flight9 key changes vs Flight8:
  1. Sigmoid → Softplus for s_illum (physics: illumination is linear/continuous, not [0,1])
  2. Cancel 3-stage cascade → single-scale on l3_lat (H/4, global lighting)
  3. Add L_illum_spatial loss in TFSNetLoss to enforce spatial variance
  4. s_noise stays sigmoid (noise confidence ∈ [0,1])

References:
  - IllumFlow (arXiv 2511.02411): illumination ≈ linear parametric function
  - QRetinex-Net (2025): frequency-aware illumination regularization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import LayerNorm2d


class IllumHead(nn.Module):
    """Softplus-based illumination head — no [0,1] constraint, natural gradient."""

    def __init__(self, in_channels: int, base_init: float = 0.3, max_val: float = 3.0):
        super().__init__()
        self.head = nn.Conv2d(in_channels, 1, 1)
        self.base = nn.Parameter(torch.tensor(base_init))
        self.max_val = max_val
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        raw = self.head(x)
        s_illum = F.softplus(raw) + self.base
        s_illum = s_illum / (1.0 + s_illum / self.max_val)
        return s_illum


class DPE(nn.Module):
    """Flight9: single-scale degradation prior estimator on l3_lat (H/4)."""

    def __init__(self, channels: int = 64, eps: float = 1e-6,
                 use_soft_median: bool = True, soft_median_tau: float = 0.1):
        super().__init__()
        self.eps = eps
        self.use_soft_median = use_soft_median
        self.soft_median_tau = soft_median_tau
        self.norm = nn.GroupNorm(8, channels)

        # Input: temporal stats(3C) + gray(1) + lum(1) = 3C+2 = 194 for C=64
        in_ch = channels * 3 + 2
        self.proj  = nn.Conv2d(in_ch, channels, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, padding=1, groups=channels // 4), nn.GELU(),
        )
        self.norm_out = LayerNorm2d(channels)
        self.illum_head = IllumHead(channels)       # softplus, no [0,1]
        self.noise_head = nn.Conv2d(channels, 1, 1)  # sigmoid in forward
        nn.init.zeros_(self.noise_head.weight)
        nn.init.zeros_(self.noise_head.bias)

    @staticmethod
    def _soft_median(x, dim=1, tau=0.1):
        with torch.no_grad():
            med = x.median(dim=dim).values.unsqueeze(dim)
        return (F.softmax(-(x - med).abs() / tau, dim=dim) * x).sum(dim=dim)

    def _compute_gray_lum(self, center_frame, H_ds, W_ds):
        gray = center_frame.mean(dim=1, keepdim=True)
        lum  = center_frame.max(dim=1, keepdim=True)[0]
        gray_ds = F.adaptive_avg_pool2d(gray, (H_ds, W_ds))
        lum_ds  = F.adaptive_avg_pool2d(lum,  (H_ds, W_ds))
        return gray_ds, lum_ds

    def forward(self, l3_lat: torch.Tensor, center_frame: torch.Tensor = None) -> dict:
        """
        l3_lat: (B, T, 64, H/4, W/4)
        center_frame: (B, 3, H, W) for gray/lum priors
        """
        B, T, C, H3, W3 = l3_lat.shape

        # Temporal statistics on l3
        fn = self.norm(l3_lat.reshape(B * T, C, H3, W3)).reshape(B, T, C, H3, W3)
        mu  = self._soft_median(fn, dim=1, tau=self.soft_median_tau) if self.use_soft_median else fn.median(dim=1).values
        sigma = torch.sqrt(fn.var(dim=1, unbiased=False) + self.eps)
        snr = mu / (sigma + self.eps)

        # Physics priors
        if center_frame is not None:
            gray_ds, lum_ds = self._compute_gray_lum(center_frame, H3, W3)
            if gray_ds.shape[0] == 1 and mu.shape[0] > 1:
                gray_ds = gray_ds.expand(mu.shape[0], -1, -1, -1)
                lum_ds  = lum_ds.expand(mu.shape[0], -1, -1, -1)
        else:
            gray_ds = torch.zeros(B, 1, H3, W3, device=l3_lat.device)
            lum_ds  = torch.zeros(B, 1, H3, W3, device=l3_lat.device)

        x = torch.cat([mu, sigma, snr, gray_ds, lum_ds], dim=1)  # (B, 3C+2, H3, W3)
        x = self.proj(x)
        x = self.refine(x)
        x = self.norm_out(x)

        s_illum  = self.illum_head(x)
        s_noise = torch.sigmoid(self.noise_head(x))

        return {
            "F_fused": x, "F_s": x, "F_f": x,
            "mu_t": mu, "sigma_t": sigma, "snr": snr,
            "s_illum": s_illum, "s_noise": s_noise,
            "s_illum_raw": s_illum,   # for L_illum_spatial
        }
