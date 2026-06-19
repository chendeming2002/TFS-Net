"""
IGRF v5.5 - Sequential Cascade Fusion with Hybrid Brightening + Direct Intensity Injection
==========================================================================================

v5.5 key changes vs v4.3:
  1. s_illum direct injection into BrightenStage (additive correction, no channel dilution)
  2. s_noise direct injection into Stage 1 (additive correction, symmetric with s_illum)
  3. Hybrid brightening: res_t = img_s2 * lit_up_map + s_illum * corr_mag(f_illum_feat)
     - Multiplicative base preserves Retinex physics
     - Additive s_illum correction gets unattenuated L_pix gradient (no 100x decay)
  4. Zero-init all new paths: initial behavior = v4.3 behavior

v4.3 key changes (preserved):
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
    """Single-stage restoration block: branch feature + current image + optional intensity -> delta -> restored image

    v5.5: When use_intensity=True, s_intensity (s_noise) is injected as additive correction
    to delta via intensity_corr (Conv2d 1->img_channels, zero-initialized).
    """

    def __init__(self, channels: int, img_channels: int = 3, use_intensity: bool = False):
        super().__init__()
        self.use_intensity = use_intensity
        self.img_proj = nn.Conv2d(img_channels, channels, 3, 1, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0),
            nn.GELU(),
            ResBlock(channels),
            ResBlock(channels),
            nn.Conv2d(channels, img_channels, 3, 1, 1),
        )
        if use_intensity:
            # v5.5: s_noise direct injection — zero-init so initial delta unchanged
            self.intensity_corr = nn.Conv2d(1, img_channels, kernel_size=3, padding=1, bias=True)
            nn.init.zeros_(self.intensity_corr.weight)
            nn.init.zeros_(self.intensity_corr.bias)

    def forward(self, f_branch: torch.Tensor, img_current: torch.Tensor,
                s_intensity: torch.Tensor = None):
        img_feat = self.img_proj(img_current)
        combined = torch.cat([f_branch, img_feat], dim=1)
        delta = self.fuse(combined)
        # v5.5: additive intensity correction (zero-init -> initial behavior unchanged)
        if self.use_intensity and s_intensity is not None:
            delta = delta + self.intensity_corr(s_intensity)
        img_next = torch.clamp(img_current + delta, 0.0, 1.0)
        return img_next, delta


class BrightenStage(nn.Module):
    """
    Hybrid brightening stage (v5.5).

    Multiplicative base (Retinex, preserved from v4.3):
        lit_up_map = lit_up_map_raw * (1 + tanh(delta) * max_delta)
        brighten_base = img_dark * lit_up_map

    Additive s_illum correction (v5.5 new):
        corr_mag = illum_corr(f_illum_feat)           # 64->3, zero-init
        illum_residual = s_illum * corr_mag
        res_t = clamp(brighten_base + illum_residual, 0, 1)

    Gradient benefit:
        d(res_t)/d(s_illum) = corr_mag  (no img_s2 decay, no 65x channel dilution)
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
        # v5.5: s_illum direct injection — zero-init so initial behavior = v4.3
        self.illum_corr = nn.Conv2d(channels, img_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.illum_corr.weight)
        nn.init.zeros_(self.illum_corr.bias)

    def forward(self, lit_up_map_raw: torch.Tensor, f_illum_feat: torch.Tensor,
                img_dark: torch.Tensor, s_illum: torch.Tensor = None):
        feat_cond = self.feat_proj(f_illum_feat)
        img_cond = self.img_proj(img_dark)
        delta = self.delta_refine(torch.cat([feat_cond, img_cond], dim=1))

        # Bounded delta: adjustment limited to +/- max_delta (default +/-50%)
        lit_up_map = lit_up_map_raw * (1.0 + torch.tanh(delta) * self.max_delta)
        lit_up_map = lit_up_map.clamp(min=0.5)

        # Multiplicative base (Retinex)
        brighten_base = img_dark * lit_up_map

        # v5.5: additive s_illum correction (zero-init -> initial res_t = brighten_base)
        if s_illum is not None:
            corr_mag = self.illum_corr(f_illum_feat)
            illum_residual = s_illum * corr_mag
            res_t = torch.clamp(brighten_base + illum_residual, 0.0, 1.0)
        else:
            res_t = torch.clamp(brighten_base, 0.0, 1.0)

        return res_t, lit_up_map


class IGRF(nn.Module):
    """
    IGRF v5.5 - Denoise -> Motion -> Brighten (sequential cascade)

    Stage 1 (denoise):   img_s1 = clamp(img_center + delta_noise(f_noise, img, s_noise), 0, 1)
    Stage 2 (motion):    img_s2 = clamp(img_s1 + delta_motion(f_motion, img_s1), 0, 1)
    Stage 3 (brighten):  res_t = clamp(img_s2 * lit_up_map + s_illum * corr_mag, 0, 1)
                          (NO .detach(): L_recon gradient flows through to NDPN/MRPN/IFPN)
    """

    def __init__(self, channels: int = 64, out_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels

        self.stage_noise = StageBlock(channels, out_channels, use_intensity=True)   # denoise + s_noise
        self.stage_motion = StageBlock(channels, out_channels, use_intensity=False)  # motion (no intensity)
        self.brighten = BrightenStage(channels, out_channels)    # hybrid brighten + s_illum

    def forward(
        self,
        f_illum_feat: torch.Tensor,
        f_noise_out: torch.Tensor,
        f_motion_out: torch.Tensor,
        lit_up_map_raw: torch.Tensor,
        image_center: torch.Tensor,
        s_illum: torch.Tensor = None,
        s_noise: torch.Tensor = None,
    ) -> dict:
        # Stage 1: denoise (in dark domain, noise amplitude is small)
        # v5.5: s_noise directly injected as additive correction
        img_s1, delta_s1 = self.stage_noise(f_noise_out, image_center, s_intensity=s_noise)

        # Stage 2: motion deblur (no intensity prior)
        img_s2, delta_s2 = self.stage_motion(f_motion_out, img_s1)

        # Stage 3: hybrid brightening
        # v5.5: s_illum directly injected as additive correction (no channel dilution)
        # NO .detach() on img_s2: allow L_recon gradient to flow through to NDPN/MRPN
        res_t, lit_up_map = self.brighten(lit_up_map_raw, f_illum_feat, img_s2, s_illum=s_illum)

        return {
            "res_t":       res_t,
            "img_s1":      img_s1,
            "img_s2":      img_s2,
            "lit_up_map":  lit_up_map,
            "delta_s1":    delta_s1,
            "delta_s2":    delta_s2,
        }
