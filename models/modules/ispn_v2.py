"""
ISPN (Illumination-Source Processing Network) — Mark4 Retinex head + Flight3 Mod4 ZeroDCE curve.

Retinex-inspired: enhanced = curve_gain × input × spatial_gain + bias
- curve_branch: ZeroDCE-style iterative curve (global per-image, from s_illum)
- gain_map: multiplicative brightening (spatial, ≈ 1 / L_estimated)
- bias_map: additive correction (dark current, color bias)

Zero-init for Phase 1 compatibility: curve≈identity, gain≈1, bias≈0 at startup.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ISPN(nn.Module):
    def __init__(self, channels: int = 64, img_channels: int = 3,
                 max_gain: float = 10.0, bias_range: float = 0.1,
                 curve_iter: int = 3):
        super().__init__()
        self.max_gain = max_gain
        self.bias_range = bias_range
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

        self.bias_head = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, img_channels, 1),
        )
        nn.init.zeros_(self.bias_head[-1].weight)
        nn.init.zeros_(self.bias_head[-1].bias)

        self.curve_alpha = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, curve_iter * img_channels),
            nn.Tanh(),
        )
        nn.init.zeros_(self.curve_alpha[2].weight)
        nn.init.zeros_(self.curve_alpha[2].bias)
        nn.init.zeros_(self.curve_alpha[4].weight)
        nn.init.zeros_(self.curve_alpha[4].bias)

    def set_max_gain(self, max_gain: float):
        self.max_gain = max_gain

    def apply_curve(self, img: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """ZeroDCE iterative curve: LE_n(I) = LE_{n-1} + A_n * LE_{n-1} * (1 - LE_{n-1})"""
        B = img.shape[0]
        alpha = alpha.reshape(B, self.curve_iter, 3, 1, 1)
        enhanced = img
        for i in range(self.curve_iter):
            A = alpha[:, i]
            enhanced = enhanced + A * enhanced * (1.0 - enhanced)
        return enhanced

    def forward(self, f_enc_center: torch.Tensor, s_illum: torch.Tensor) -> dict:
        h = self.refine(torch.cat([f_enc_center, s_illum], dim=1))

        raw_gain = self.gain_head(h)
        gain_map = 1.0 + F.softplus(raw_gain) * (self.max_gain - 1.0) / F.softplus(torch.tensor(4.0, device=raw_gain.device, dtype=raw_gain.dtype))

        bias_map = torch.tanh(self.bias_head(h)) * self.bias_range

        alpha = self.curve_alpha(s_illum)

        return {
            "gain_map": gain_map,
            "bias_map": bias_map,
            "curve_alpha": alpha,
            "f_illum_feat": h,
        }
