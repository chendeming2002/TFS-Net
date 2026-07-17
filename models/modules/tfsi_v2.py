"""
DPE (Degradation Prior Estimator) — Flight8: 3-stage coarse-to-fine scan + physics priors.

Stage 3 (coarse): l3_lat(H/4) + gray↓/lum↓ → MSbranch(d=1,2,4)
Stage 2 (mid):    l2_lat(H/2) + gray↓/lum↓ + s3↑ → MSbranch(d=1,2)
Stage 1 (fine):   l1_lat(H)   + gray/lum     + s2↑ → Conv → head → s_illum, s_noise

Gray = RGB.mean (overall brightness), Lum = RGB.max (peak brightness).
These provide direct physical illumination priors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSpatialBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid = out_channels // 3
        wide_ch = out_channels - 2 * mid
        self.branch_local = nn.Sequential(nn.Conv2d(in_channels, mid, 3, 1, 1), nn.GELU())
        self.branch_mid   = nn.Sequential(nn.Conv2d(in_channels, mid, 3, 1, padding=2, dilation=2), nn.GELU())
        self.branch_wide  = nn.Sequential(nn.Conv2d(in_channels, wide_ch, 3, 1, padding=4, dilation=4), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.GELU(), nn.Conv2d(out_channels, out_channels, 3, 1, 1))

    def forward(self, x):
        return self.fuse(torch.cat([self.branch_local(x), self.branch_mid(x), self.branch_wide(x)], dim=1))


class DPE(nn.Module):
    def __init__(self, channels: int = 64, fused_channels: int = 64, eps: float = 1e-6,
                 use_soft_median: bool = True, soft_median_tau: float = 0.1):
        super().__init__()
        self.eps = eps
        self.use_soft_median = use_soft_median
        self.soft_median_tau = soft_median_tau
        self.norm = nn.GroupNorm(8, channels)

        # Stage 3 (coarse, H/4): full MS branch d=1,2,4 for global lighting
        self.ms3 = MultiScaleSpatialBranch(channels * 3 + 2, fused_channels)
        # Stage 2 (mid, H/2): smaller MS branch d=1,2
        self.ms2 = MultiScaleSpatialBranch(channels * 3 + 2 + fused_channels, fused_channels)
        # Stage 1 (fine, H): single conv for local texture
        self.refine1 = nn.Sequential(nn.Conv2d(channels * 3 + 2 + fused_channels, fused_channels, 3, 1, 1), nn.GELU())

        self.pre_head_norm = nn.LayerNorm(fused_channels)
        self.head = nn.Conv2d(fused_channels, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _soft_median(x, dim=1, tau=0.1):
        with torch.no_grad():
            med = x.median(dim=dim).values.unsqueeze(dim)
        return (F.softmax(-(x - med).abs() / tau, dim=dim) * x).sum(dim=dim)

    def _compute_gray_lum(self, center_frame):
        """center_frame: (B, 3, H, W) in [0,1]. Returns gray(B,1,H,W), lum(B,1,H,W)."""
        gray = center_frame.mean(dim=1, keepdim=True)
        lum  = center_frame.max(dim=1, keepdim=True)[0]
        return gray, lum

    def _temporal_stats(self, feats):
        B, T, C, H, W = feats.shape
        fn = self.norm(feats.reshape(B*T, C, H, W)).reshape(B, T, C, H, W)
        mu = self._soft_median(fn, dim=1, tau=self.soft_median_tau) if self.use_soft_median else fn.median(dim=1).values
        sigma = torch.sqrt(fn.var(dim=1, unbiased=False) + self.eps)
        snr = mu / (sigma + self.eps)
        return mu, sigma, snr

    def forward(self, l1_lat: torch.Tensor, l2_lat: torch.Tensor, l3_lat: torch.Tensor,
                center_frame: torch.Tensor = None) -> dict:
        """
        l1_lat: (B,T,64,H,W), l2_lat: (B,T,64,H/2,W/2), l3_lat: (B,T,64,H/4,W/4)
        center_frame: (B,3,H,W) in [0,1] for gray/lum priors
        """
        B, T, C1, H, W = l1_lat.shape
        _, _, C2, H2, W2 = l2_lat.shape
        _, _, C3, H3, W3 = l3_lat.shape

        # Physics priors at each scale
        if center_frame is not None:
            gray, lum = self._compute_gray_lum(center_frame)  # (B,1,H,W)
            gray3 = F.adaptive_avg_pool2d(gray, (H3, W3)); lum3 = F.adaptive_avg_pool2d(lum, (H3, W3))
            gray2 = F.adaptive_avg_pool2d(gray, (H2, W2)); lum2 = F.adaptive_avg_pool2d(lum, (H2, W2))
        else:
            gray3 = lum3 = torch.zeros(B, 1, H3, W3, device=l3_lat.device)
            gray2 = lum2 = torch.zeros(B, 1, H2, W2, device=l2_lat.device)
            gray  = lum  = torch.zeros(B, 1, H,  W,  device=l1_lat.device)

        # Stage 3 (coarse)
        mu3, sigma3, snr3 = self._temporal_stats(l3_lat)
        if gray3.shape[0] == 1 and mu3.shape[0] > 1:
            gray3 = gray3.expand(mu3.shape[0], -1, -1, -1)
            lum3  = lum3.expand(mu3.shape[0], -1, -1, -1)
        stats3 = torch.cat([mu3, sigma3, snr3, gray3, lum3], dim=1)
        s3 = self.ms3(stats3)

        # Stage 2 (mid)
        mu2, sigma2, snr2 = self._temporal_stats(l2_lat)
        if gray2.shape[0] == 1 and mu2.shape[0] > 1:
            gray2 = gray2.expand(mu2.shape[0], -1, -1, -1)
            lum2  = lum2.expand(mu2.shape[0], -1, -1, -1)
        stats2 = torch.cat([mu2, sigma2, snr2, gray2, lum2, F.interpolate(s3, size=(H2, W2), mode='bilinear', align_corners=False)], dim=1)
        s2 = self.ms2(stats2)

        # Stage 1 (fine)
        mu1, sigma1, snr1 = self._temporal_stats(l1_lat)
        if gray.shape[0] == 1 and mu1.shape[0] > 1:
            gray = gray.expand(mu1.shape[0], -1, -1, -1)
            lum  = lum.expand(mu1.shape[0], -1, -1, -1)
        stats1 = torch.cat([mu1, sigma1, snr1, gray, lum, F.interpolate(s2, size=(H, W), mode='bilinear', align_corners=False)], dim=1)
        s1 = self.refine1(stats1)

        f_normed = self.pre_head_norm(s1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        raw = torch.sigmoid(self.head(f_normed))
        return {"F_fused": s1, "F_s": s1, "F_f": s1,
                "mu_t": mu1, "sigma_t": sigma1, "snr": snr1,
                "s_illum": raw[:, 0:1], "s_noise": raw[:, 1:2]}
