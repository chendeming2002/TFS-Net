"""
ISPN (Illumination-Source Processing Network) — Mod6: spatial curve + gain (no bias).

Retinex-inspired: enhanced = curve(img) × gain_map
- curve_branch: ZeroDCE-style spatial iterative curve (pixel-wise α, from refine feat h)
- gain_map: multiplicative brightening (spatial, ≈ 1 / L_estimated)

Mod6: bias_map removed — additive correction provided by ZeroDCE curve.
Mod6: MLP global α replaced by SpatialCurveBranch (2-conv, per-pixel α, 8 iterations).
Zero-init for Phase 1 compatibility: curve≈identity, gain≈1 at startup.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ISPN(nn.Module):
    def __init__(self, channels: int = 64, img_channels: int = 3,
                 max_gain: float = 10.0, curve_iter: int = 8):
        super().__init__()
        self.max_gain = max_gain
        self.curve_iter = curve_iter

        self.refine = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GELU(),
        )

        self.gain_head = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1),
        )
        nn.init.zeros_(self.gain_head[-1].weight)
        nn.init.constant_(self.gain_head[-1].bias, 0.8)

        self.curve_conv1 = nn.Conv2d(channels, 32, 3, 1, 1)
        self.curve_conv2 = nn.Conv2d(32, curve_iter * img_channels, 3, 1, 1)
        nn.init.zeros_(self.curve_conv2.weight)
        nn.init.zeros_(self.curve_conv2.bias)

    def set_max_gain(self, max_gain: float):
        self.max_gain = max_gain

    def apply_curve(self, img: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """ZeroDCE pixel-wise curve: LE_n = LE_{n-1} + A_n * LE_{n-1} * (1 - LE_{n-1})
        A: (B, n_iter, 3, H, W) per-pixel per-channel alpha"""
        enhanced = img
        for i in range(self.curve_iter):
            alpha = A[:, i]  # (B, 3, H, W)
            enhanced = enhanced + alpha * enhanced * (1.0 - enhanced)
        return enhanced

    def forward(self, f_enc_center: torch.Tensor, s_illum: torch.Tensor) -> dict:
        h = self.refine(torch.cat([f_enc_center, s_illum], dim=1))

        raw_gain = self.gain_head(h)
        gain_map = 0.5 + F.softplus(raw_gain) / F.softplus(torch.tensor(4.0, device=raw_gain.device, dtype=raw_gain.dtype)) * (self.max_gain - 0.5)

        x = F.gelu(self.curve_conv1(h))
        A = torch.tanh(self.curve_conv2(x))
        A = A.reshape(A.shape[0], self.curve_iter, 3, A.shape[2], A.shape[3])

        return {
            "gain_map": gain_map,
            "curve_alpha": A,
            "f_illum_feat": h,
        }
