"""
IGRF v4.2 - Sequential Cascade Fusion with Multiplicative Brightening
=====================================================================

v4.2 key changes vs v4.1:
  1. Stage order: noise -> motion -> brighten (denoise first, brighten last)
  2. BrightenStage uses Retinexformer-style img * exp(lit_up_feat) multiplication
  3. .detach() between stages to prevent gradient interference
  4. Gradient stability: d(res_t)/d(lit_up_feat) = res_t (proportional to output)

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
    Retinexformer-style multiplicative brightening stage.

    Converts lit_up_map_raw to log space, adds a delta correction (conditioned
    on illumination features and current image), then exponentiates.

    res_t = img_dark * exp(log(lit_up_map_raw) + delta)
    Gradient: d(res_t)/d(delta) = res_t (proportional to output, not dark input)
    """

    def __init__(self, channels: int, img_channels: int = 3):
        super().__init__()
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
        lit_up_log = torch.log(lit_up_map_raw.clamp(min=1e-6))
        lit_up_map = torch.exp(lit_up_log + delta)
        res_t = torch.clamp(img_dark * lit_up_map, 0.0, 1.0)
        return res_t, lit_up_map


class IGRF(nn.Module):
    """
    IGRF v4.2 - Denoise -> Motion -> Brighten (sequential cascade with .detach())

    Stage 1 (denoise):     img_s1 = image_center + delta_noise
    Stage 2 (motion):      img_s2 = img_s1.detach() + delta_motion
    Stage 3 (brighten):    res_t  = img_s2.detach() * exp(log(lit_up_map_raw) + delta)
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

        # Stage 2: motion deblur (.detach() prevents gradient interference)
        img_s2, delta_s2 = self.stage_motion(f_motion_out, img_s1.detach())

        # Stage 3: Retinexformer-style multiplicative brightening (.detach())
        res_t, lit_up_map = self.brighten(
            lit_up_map_raw, f_illum_feat, img_s2.detach()
        )

        return {
            "res_t":       res_t,
            "img_s1":      img_s1,
            "img_s2":      img_s2,
            "lit_up_map":  lit_up_map,
            "delta_s1":    delta_s1,
            "delta_s2":    delta_s2,
        }
