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
import torch.nn.functional as F
from .blocks import ResBlock, NAFBlock


def soft_clamp(x: torch.Tensor, sharpness: float = 20.0) -> torch.Tensor:
    """v5.6 P0-4: soft clamp with non-zero gradient everywhere.

    sigmoid(sharpness * (x - 0.5)) maps R -> (0, 1), gradient = sharpness * s * (1-s) > 0.
    At x=0 or x=1, gradient is tiny but non-zero (vs hard clamp's 0 gradient).
    """
    return torch.sigmoid(sharpness * (x - 0.5))


def _make_res_blocks(channels: int, n: int, use_nafblock: bool = False):
    """创建 n 个残差块，可选 NAFBlock 或 ResBlock。"""
    Block = NAFBlock if use_nafblock else ResBlock
    return nn.Sequential(*[Block(channels) for _ in range(n)])


class StageBlock(nn.Module):
    """Single-stage restoration block: branch feature + current image + optional intensity -> delta -> restored image

    v5.5: When use_intensity=True, s_intensity (s_noise) is injected as additive correction
    to delta via intensity_corr (Conv2d 1->img_channels, zero-initialized).
    v5.6 P0-4: use_soft_clamp controls whether intermediate stages use soft clamp (default True).
    v5.6 P1-7: use_nafblock controls whether to use NAFBlock instead of ResBlock in fuse.
    """

    def __init__(self, channels: int, img_channels: int = 3, use_intensity: bool = False,
                 use_soft_clamp: bool = False, use_nafblock: bool = False,
                 num_res_blocks: int = 2):
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
        # v5.6 P0-4: soft clamp for intermediate stages (gradient non-zero in dark/bright regions)
        if self.use_soft_clamp:
            img_next = soft_clamp(img_current + delta)
        else:
            img_next = torch.clamp(img_current + delta, 0.0, 1.0)
        return img_next, delta


class BrightenStage(nn.Module):
    """
    Hybrid brightening stage (v5.5, restored from v5.7 gating experiment).

    v5.7 gating (lit_up_map = 1 + s_illum * (lit_up_map_full - 1)) solved s_illum collapse
    but reduced PSNR by 0.6 dB (constrains per-channel freedom). Reverted to v5.5 additive.

    Multiplicative base (Retinex):
        lit_up_map = lit_up_map_raw * (1 + tanh(delta) * max_delta)
        brighten_base = img_dark * lit_up_map

    Additive s_illum correction:
        corr_mag = illum_corr(f_illum_feat)           # 64->3, zero-init
        illum_residual = s_illum * corr_mag
        res_t = clamp(brighten_base + illum_residual, 0, 1)
    """

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
        # Delta: unified_illu 已移除 — A_illu 改由 IFPN 生成并传入

    def forward(self, lit_up_map_raw: torch.Tensor, f_illum_feat: torch.Tensor,
                img_dark: torch.Tensor, A_illu: torch.Tensor = None):
        """Delta: A_illu 由 IFPN 传入，不再内部生成"""
        feat_cond = self.feat_proj(f_illum_feat)
        img_cond = self.img_proj(img_dark)
        delta = self.delta_refine(torch.cat([feat_cond, img_cond], dim=1))
        lit_up_map = lit_up_map_raw * (1.0 + torch.tanh(delta) * self.max_delta)
        lit_up_map = lit_up_map.clamp(min=0.5)

        if A_illu is not None:
            A_resized = F.interpolate(A_illu, size=lit_up_map.shape[-2:],
                                      mode='bilinear', align_corners=False)
            lit_up_map = lit_up_map * (1.0 + A_resized)

        res_t = torch.clamp(img_dark * lit_up_map, 0.0, 1.0)
        return res_t, lit_up_map


class IGRF(nn.Module):
    """
    IGRF v5.7 - Denoise -> Motion -> Brighten (sequential cascade)

    Stage 1 (denoise):   img_s1 = clamp(img_center + delta_noise(f_noise, img, s_noise))
                          s_noise 作为 additive correction 直接参与 delta
    Stage 2 (motion):    img_s2 = clamp(img_s1 + delta_motion(f_motion, img_s1))
    Stage 3 (brighten):  lit_up_map = 1 + s_illum * (lit_up_map_full - 1)   (v5.7 乘法门控)
                          res_t = clamp(img_s2 * lit_up_map, 0, 1)
                          s_illum 门控 lit_up_map: s_illum=0→无提亮, s_illum=1→完全提亮
                          (NO .detach(): L_recon gradient flows through to NDPN/MRPN/IFPN)
    v5.7: 移除加法修正路径, 改用乘法门控, 消除 s_illum 功能冗余
    """

    def __init__(self, channels: int = 64, out_channels: int = 3, use_soft_clamp: bool = False,
                 use_nafblock: bool = False, num_res_blocks: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels

        self.stage_noise = StageBlock(channels, out_channels, use_intensity=True,
                                       use_soft_clamp=use_soft_clamp,
                                       use_nafblock=use_nafblock,
                                       num_res_blocks=num_res_blocks)
        self.stage_motion = StageBlock(channels, out_channels, use_intensity=False,
                                       use_soft_clamp=use_soft_clamp,
                                       use_nafblock=use_nafblock,
                                       num_res_blocks=num_res_blocks)
        self.brighten = BrightenStage(channels, out_channels,
                                      use_nafblock=use_nafblock)    # hybrid brighten + s_illum (final hard clamp)

    def forward(
        self,
        f_illum_feat: torch.Tensor,
        f_noise_out: torch.Tensor,
        f_motion_out: torch.Tensor,
        lit_up_map_raw: torch.Tensor,
        image_center: torch.Tensor,
        A_illu: torch.Tensor = None,
        s_noise: torch.Tensor = None,
    ) -> dict:
        """Delta: A_illu 由 IFPN 生成传入"""
        img_s1, delta_s1 = self.stage_noise(f_noise_out, image_center, s_intensity=s_noise)
        img_s2, delta_s2 = self.stage_motion(f_motion_out, img_s1)
        res_t, lit_up_map = self.brighten(lit_up_map_raw, f_illum_feat, img_s2, A_illu=A_illu)

        return {
            "res_t":       res_t,
            "img_s1":      img_s1,
            "img_s2":      img_s2,
            "lit_up_map":  lit_up_map,
            "delta_s1":    delta_s1,
            "delta_s2":    delta_s2,
        }
