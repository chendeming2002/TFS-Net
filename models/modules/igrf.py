"""
IGRF v4.3 - Sequential Cascade Fusion with Bounded Multiplicative Brightening
===============================================================================

v4.3 key changes vs v4.2:
  1. BrightenStage uses bounded delta: lit_up_map = raw * (1 + tanh(delta) * max_delta)
  2. Eliminates clamp gradient dead zone from v4.2's exp(log(x) + delta)
  3. L_inter only supervises img_s2 (theoretically well-posed, img_s1 still has blur)

References:
  - Cai et al., Retinexformer (ICCV 2023): lit_up_map multiplicative brightening
  - Zamir et al., MPRNet (CVPR 2021): .detach() cross-stage gradient isolation
  - Feijoo et al., DarkIR (CVPR 2025): denoise-before-brighten design
"""

import torch
import torch.nn as nn

from .blocks import ResBlock


class StageBlock(nn.Module):
    """Single-stage restoration block: branch feature + current image -> delta -> restored image"""

    def __init__(self, channels: int, img_channels: int = 3):
        super().__init__()
        self.img_proj = nn.Conv2d(img_channels, channels, 3, 1, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0),
            nn.GELU(),
            ResBlock(channels),
            ResBlock(channels),
            nn.Conv2d(channels, img_channels, 3, 1, 1),
        )

    def forward(self, f_branch: torch.Tensor, img_current: torch.Tensor):
        img_feat = self.img_proj(img_current)
        combined = torch.cat([f_branch, img_feat], dim=1)
        delta = self.fuse(combined)
        img_next = torch.clamp(img_current + delta, 0.0, 1.0)
        return img_next, delta


class BrightenStage(nn.Module):
    """
    Bounded multiplicative brightening stage (v4.3).

    lit_up_map = lit_up_map_raw * (1 + tanh(delta) * max_delta)
    res_t = clamp(img_dark * lit_up_map, 0, 1)

    Bounded delta prevents the exp gradient dead zone from v4.2.
    tanh limits adjustment range to +/- max_delta (default 50%).
    """

    def __init__(self, channels: int, img_channels: int = 3, max_delta: float = 0.5):
        super().__init__()
        self.max_delta = max_delta
        self.feat_proj = nn.Conv2d(channels, img_channels, 3, 1, 1)
        self.img_proj = nn.Conv2d(img_channels, img_channels, 3, 1, 1)
        self.delta_refine = nn.Sequential(
            nn.Conv2d(img_channels * 2, img_channels, 1, 1, 0),
            nn.GELU(),
            ResBlock(img_channels),
            nn.Conv2d(img_channels, img_channels, 3, 1, 1),
        )

    def forward(self, lit_up_map_raw: torch.Tensor, f_illum_feat: torch.Tensor,
                img_dark: torch.Tensor):
        feat_cond = self.feat_proj(f_illum_feat)
        img_cond = self.img_proj(img_dark)
        delta = self.delta_refine(torch.cat([feat_cond, img_cond], dim=1))

        # Bounded delta: adjustment limited to +/- max_delta (default +/-50%)
        lit_up_map = lit_up_map_raw * (1.0 + torch.tanh(delta) * self.max_delta)
        lit_up_map = lit_up_map.clamp(min=0.5)

        res_t = torch.clamp(img_dark * lit_up_map, 0.0, 1.0)
        return res_t, lit_up_map


class IGRF(nn.Module):
    """
    IGRF v4.3 - Denoise -> Motion -> Brighten (sequential cascade)

    Stage 1 (denoise):   img_s1 = image_center + delta_noise
    Stage 2 (motion):    img_s2 = img_s1 + delta_motion
    Stage 3 (brighten):  res_t  = img_s2 * bounded_lit_up_map
                          (NO .detach(): L_recon gradient flows through to NDPN/MRPN)
    """

    def __init__(self, channels: int = 64, out_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels

        self.stage_noise = StageBlock(channels, out_channels)    # denoise
        self.stage_motion = StageBlock(channels, out_channels)   # motion deblur
        self.brighten = BrightenStage(channels, out_channels)    # multiplicative brighten

    def forward(
        self,
        f_illum_feat: torch.Tensor,
        f_noise_out: torch.Tensor,
        f_motion_out: torch.Tensor,
        lit_up_map_raw: torch.Tensor,
        image_center: torch.Tensor,
    ) -> dict:
        # Stage 1: denoise (in dark domain, noise amplitude is small)
        img_s1, delta_s1 = self.stage_noise(f_noise_out, image_center)

        # Stage 2: motion deblur
        img_s2, delta_s2 = self.stage_motion(f_motion_out, img_s1)

        # Stage 3: bounded multiplicative brightening
        # NO .detach() on img_s2: allow L_recon gradient to flow through to NDPN/MRPN
        res_t, lit_up_map = self.brighten(lit_up_map_raw, f_illum_feat, img_s2)

        return {
            "res_t":       res_t,
            "img_s1":      img_s1,
            "img_s2":      img_s2,
            "lit_up_map":  lit_up_map,
            "delta_s1":    delta_s1,
            "delta_s2":    delta_s2,
        }
