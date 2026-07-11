"""
ISPN (Illumination-Source Processing Network) — Flight5: Target-Convergent Curve + gain.

Flight5: replaces ZeroDCE LE-curve with Target-Convergent Curve (TCC).
TCC formula: LE_n = LE_{n-1} + A_n * LE_{n-1} * (1-LE_{n-1}) * (α_target - LE_{n-1})
Converges to α_target instead of saturating at 1.0.
A ∈ [-4, 4] (4×tanh), 6 iterations, per-pixel.
gain_head bias=0.0 → initial gain≈1.1 (identity bias).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ISPN(nn.Module):
    def __init__(self, channels: int = 64, img_channels: int = 3,
                 max_gain: float = 10.0, curve_iter: int = 6):
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
        nn.init.zeros_(self.gain_head[-1].bias)

        # Flight5: Target-Convergent Curve — A maps in [-4, 4], 6 iter, learns α_target
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.curve_conv1 = nn.Conv2d(channels, 32, 3, 1, 1)
        self.curve_conv2 = nn.Conv2d(32, curve_iter * img_channels, 3, 1, 1)
        nn.init.zeros_(self.curve_conv2.weight)
        nn.init.zeros_(self.curve_conv2.bias)

    @property
    def alpha_target(self):
        return torch.sigmoid(self.alpha_raw)

    def set_max_gain(self, max_gain: float):
        self.max_gain = max_gain

    def forward(self, f_enc_center: torch.Tensor, s_illum: torch.Tensor) -> dict:
        h = self.refine(torch.cat([f_enc_center, s_illum], dim=1))

        raw_gain = self.gain_head(h)
        gain_map = 0.5 + F.softplus(raw_gain) / F.softplus(torch.tensor(4.0, device=raw_gain.device, dtype=raw_gain.dtype)) * (self.max_gain - 0.5)

        x = F.gelu(self.curve_conv1(h))
        A_raw = self.curve_conv2(x)  # (B, n_iter*3, H, W)
        A_maps = 4.0 * torch.tanh(A_raw)  # [-4, 4]
        A_maps = A_maps.reshape(A_maps.shape[0], self.curve_iter, 3, A_maps.shape[2], A_maps.shape[3])

        return {
            "gain_map": gain_map,
            "curve_A": A_maps,
            "alpha_target": self.alpha_target,
            "f_illum_feat": h,
        }
