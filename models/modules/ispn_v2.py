"""
ISPN (Illumination-Source Processing Network) — Mark4 simplified Retinex head.

Retinex-inspired: enhanced = input × gain + bias
- gain_map: multiplicative brightening (≈ 1 / L_estimated)
- bias_map: additive correction (dark current, color bias)

Zero-init for Phase 1 compatibility: gain≈1, bias≈0 at startup.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ISPN(nn.Module):
    def __init__(self, channels: int = 64, img_channels: int = 3,
                 max_gain: float = 10.0, bias_range: float = 0.1):
        super().__init__()
        self.max_gain = max_gain
        self.bias_range = bias_range

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
        nn.init.constant_(self.gain_head[-1].bias, 0.8)  # log(2.2)→gain≈2.2 initial: moderate brightening

        self.bias_head = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, img_channels, 1),
        )
        nn.init.zeros_(self.bias_head[-1].weight)
        nn.init.zeros_(self.bias_head[-1].bias)

    def forward(self, f_enc_center: torch.Tensor, s_illum: torch.Tensor) -> dict:
        h = self.refine(torch.cat([f_enc_center, s_illum], dim=1))

        log_gain = self.gain_head(h)
        gain_map = torch.exp(log_gain).clamp(1.0, self.max_gain)

        bias_map = torch.tanh(self.bias_head(h)) * self.bias_range

        return {
            "gain_map": gain_map,
            "bias_map": bias_map,
            "f_illum_feat": h,
        }
