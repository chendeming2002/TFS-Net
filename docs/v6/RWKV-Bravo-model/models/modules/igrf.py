"""
IGRF — Inverse-Grained Restoration Fusion (v5.5)
=================================================
Denoise → Motion → Hybrid Brighten 三级联恢复.
"""

import torch
import torch.nn as nn

from .blocks import ResBlock, NAFBlock


def soft_clamp(x: torch.Tensor, sharpness: float = 20.0) -> torch.Tensor:
    return torch.sigmoid(sharpness * (x - 0.5))


def _make_res_blocks(channels: int, n: int, use_nafblock: bool = False):
    Block = NAFBlock if use_nafblock else ResBlock
    return nn.Sequential(*[Block(channels) for _ in range(n)])


class StageBlock(nn.Module):
    def __init__(self, channels: int, img_channels: int = 3, use_intensity: bool = False,
                 use_soft_clamp: bool = False, use_nafblock: bool = False, num_res_blocks: int = 2):
        super().__init__()
        self.use_intensity = use_intensity
        self.use_soft_clamp = use_soft_clamp
        self.img_proj = nn.Conv2d(img_channels, channels, 3, 1, 1)
        Block = NAFBlock if use_nafblock else ResBlock
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0),
            nn.GELU(),
            *[Block(channels) for _ in range(num_res_blocks)],
            nn.Conv2d(channels, img_channels, 3, 1, 1),
        )
        if use_intensity:
            self.intensity_corr = nn.Conv2d(1, img_channels, kernel_size=3, padding=1, bias=True)
            nn.init.zeros_(self.intensity_corr.weight)
            nn.init.zeros_(self.intensity_corr.bias)

    def forward(self, f_branch, img_current, s_intensity=None):
        img_feat = self.img_proj(img_current)
        combined = torch.cat([f_branch, img_feat], dim=1)
        delta = self.fuse(combined)
        if self.use_intensity and s_intensity is not None:
            delta = delta + self.intensity_corr(s_intensity)
        if self.use_soft_clamp:
            img_next = soft_clamp(img_current + delta)
        else:
            img_next = torch.clamp(img_current + delta, 0.0, 1.0)
        return img_next, delta


class BrightenStage(nn.Module):
    def __init__(self, channels: int, img_channels: int = 3, max_delta: float = 0.5,
                 use_nafblock: bool = False):
        super().__init__()
        self.max_delta = max_delta
        self.feat_proj = nn.Conv2d(channels, img_channels, 3, 1, 1)
        self.img_proj = nn.Conv2d(img_channels, img_channels, 3, 1, 1)
        Block = NAFBlock if use_nafblock else ResBlock
        self.delta_refine = nn.Sequential(
            nn.Conv2d(img_channels * 2, img_channels, 1, 1, 0),
            nn.GELU(),
            Block(img_channels),
            nn.Conv2d(img_channels, img_channels, 3, 1, 1),
        )
        self.illum_corr = nn.Conv2d(channels, img_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.illum_corr.weight)
        nn.init.zeros_(self.illum_corr.bias)

    def forward(self, lit_up_map_raw, f_illum_feat, img_dark, s_illum=None):
        feat_cond = self.feat_proj(f_illum_feat)
        img_cond = self.img_proj(img_dark)
        delta = self.delta_refine(torch.cat([feat_cond, img_cond], dim=1))
        lit_up_map = lit_up_map_raw * (1.0 + torch.tanh(delta) * self.max_delta)
        lit_up_map = lit_up_map.clamp(min=0.5)
        brighten_base = img_dark * lit_up_map
        if s_illum is not None:
            corr_mag = self.illum_corr(f_illum_feat)
            illum_residual = s_illum * corr_mag
            res_t = torch.clamp(brighten_base + illum_residual, 0.0, 1.0)
        else:
            res_t = torch.clamp(brighten_base, 0.0, 1.0)
        return res_t, lit_up_map


class IGRF(nn.Module):
    def __init__(self, channels: int = 64, out_channels: int = 3, use_soft_clamp: bool = False,
                 use_nafblock: bool = False, num_res_blocks: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels

        self.stage_noise = StageBlock(channels, out_channels, use_intensity=True,
                                       use_soft_clamp=use_soft_clamp,
                                       use_nafblock=use_nafblock, num_res_blocks=num_res_blocks)
        self.stage_motion = StageBlock(channels, out_channels, use_intensity=False,
                                        use_soft_clamp=use_soft_clamp,
                                        use_nafblock=use_nafblock, num_res_blocks=num_res_blocks)
        self.brighten = BrightenStage(channels, out_channels, use_nafblock=use_nafblock)

    def forward(self, f_illum_feat, f_noise_out, f_motion_out, lit_up_map_raw,
                image_center, s_illum=None, s_noise=None):
        img_s1, delta_s1 = self.stage_noise(f_noise_out, image_center, s_intensity=s_noise)
        img_s2, delta_s2 = self.stage_motion(f_motion_out, img_s1)
        res_t, lit_up_map = self.brighten(lit_up_map_raw, f_illum_feat, img_s2, s_illum=s_illum)
        return {"res_t": res_t, "img_s1": img_s1, "img_s2": img_s2, "lit_up_map": lit_up_map,
                "delta_s1": delta_s1, "delta_s2": delta_s2}
