"""
ISPN (Illumination-Source Processing Network) — Flight6: DOF-rebalanced curve + pixel-wise gain.

Flight6: Zero-DCE++ inspired DOF rebalancing.
Curve: 3ch params (downsampled 4×, reused 6 iter) — DOF = 0.1875HW
Gain:  pixel-wise conv+sigmoid — DOF = 1HW → gain has 5× DOF advantage

Gain range: [0.5, 2.0] via sigmoid, initial≈1.25.
Curve bias=0.25 → A≈0.98 at init → curve starts working immediately.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ISPN(nn.Module):
    def __init__(self, channels: int = 64, img_channels: int = 3,
                 curve_iter: int = 6, ds_factor: int = 8,
                 gain_min: float = 0.5, gain_max: float = 2.0):
        super().__init__()
        self.curve_iter = curve_iter
        self.ds_factor = ds_factor
        self.gain_min = gain_min
        self.gain_max = gain_max

        self.refine = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GELU(),
        )

        # Flight6: pixel-wise gain — conv+sigmoid, no GAP, no softplus
        self.gain_net = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        nn.init.zeros_(self.gain_net[-1].weight)
        nn.init.zeros_(self.gain_net[-1].bias)

        # Flight6: TCC with parameter sharing — 3ch output, 4× downsampled
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.curve_net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, img_channels, 3, padding=1),
        )
        nn.init.zeros_(self.curve_net[-1].weight)
        nn.init.constant_(self.curve_net[-1].bias, 0.25)

    @property
    def alpha_target(self):
        return torch.sigmoid(self.alpha_raw)

    def forward(self, f_enc_center: torch.Tensor, s_illum: torch.Tensor) -> dict:
        h = self.refine(torch.cat([f_enc_center, s_illum], dim=1))

        # Gain: pixel-wise, sigmoid → [gain_min, gain_max]
        gain_raw = self.gain_net(h)
        gain_map = self.gain_min + (self.gain_max - self.gain_min) * torch.sigmoid(gain_raw)

        # Curve: downsampled feature → 1 A_map → reused n_iter times
        B, _, H, W = h.shape
        H_ds, W_ds = H // self.ds_factor, W // self.ds_factor
        feat_ds = F.adaptive_avg_pool2d(h, (H_ds, W_ds))
        A_map_ds = self.curve_net(feat_ds)
        A_map = F.interpolate(A_map_ds, size=(H, W), mode='bilinear', align_corners=False)
        A_map = 4.0 * torch.tanh(A_map)  # (B, 3, H, W)

        # Repeat A_map for curve_iter times
        A_maps = A_map.unsqueeze(1).repeat(1, self.curve_iter, 1, 1, 1)

        return {
            "gain_map": gain_map,
            "curve_A": A_maps,
            "alpha_target": self.alpha_target,
            "f_illum_feat": h,
        }
