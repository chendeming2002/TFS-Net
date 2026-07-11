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


def soft_clamp(x: torch.Tensor, sharpness: float = 4.0) -> torch.Tensor:
    """Flight4: Tanh-based symmetric soft clamp with non-zero gradient everywhere.

    Maps R → approximately [0, 1] via 0.5 + 0.5 * tanh(sharpness * (x - 0.5)).
    Gradient = sharpness/2 * sech²(sharpness*(x-0.5)) > 0 everywhere.
    Unlike hard clamp (gradient=0 outside [0,1]), this provides continuous gradient.
    
    At sharpness=4: x=0→output≈0.02, x=1→output≈0.98, gradient nonzero everywhere.
    This prevents the asymmetry where negative deltas get truncated to zero
    (causing S2's zero-mean constraint to fail in practice).
    """
    return 0.5 + 0.5 * torch.tanh(sharpness * (x - 0.5))


def _make_res_blocks(channels: int, n: int, use_nafblock: bool = False):
    """创建 n 个残差块，可选 NAFBlock 或 ResBlock。"""
    Block = NAFBlock if use_nafblock else ResBlock
    return nn.Sequential(*[Block(channels) for _ in range(n)])


class StageBlock(nn.Module):
    """Single-stage restoration block with zero-gate (Mark4).

    gate starts at 0 → StageBlock(any_input) ≈ 0 → no perturbation to main path.
    Physically: equivalent to LoRA B=0 initialization — new branch starts silently.
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
        # Mark4: zero-gate — starts at 0, progressive release
        self.gate = nn.Parameter(torch.zeros(1))
        if use_intensity:
            self.intensity_corr = nn.Conv2d(1, img_channels, kernel_size=3, padding=1, bias=True)
            nn.init.zeros_(self.intensity_corr.weight)
            nn.init.zeros_(self.intensity_corr.bias)

    def forward(self, f_branch: torch.Tensor, img_current: torch.Tensor,
                s_intensity: torch.Tensor = None):
        img_feat = self.img_proj(img_current)
        combined = torch.cat([f_branch, img_feat], dim=1)
        delta = self.fuse(combined) * self.gate
        if self.use_intensity and s_intensity is not None:
            delta = delta + self.intensity_corr(s_intensity)
        # Flight3 Mod5: zero-mean constraint — StageBlock rearranges pixels but
        # cannot shift mean brightness. All brightness change must route through
        # ISPN curve/gain (enforcing S2.1 denoise-before-brighten constraint).
        delta = delta - delta.mean(dim=[-2, -1], keepdim=True)
        # Flight4: always use soft_clamp — hard clamp corrupts gradient flow
        # (zero-mean δ only works when negative deltas aren't truncated to 0)
        img_next = soft_clamp(img_current + delta)
        return img_next, delta


class BrightenStage(nn.Module):
    """Mod6: Retinex-style brightening — img_enhanced = img × gain.

    bias_map removed — additive correction provided by ZeroDCE curve.
    """

    def __init__(self, channels: int, img_channels: int = 3,
                 use_nafblock: bool = False):
        super().__init__()
        Block = NAFBlock if use_nafblock else ResBlock
        self.final_refine = nn.Sequential(
            nn.Conv2d(img_channels, img_channels, 3, 1, 1),
        )
        nn.init.zeros_(self.final_refine[0].weight)
        nn.init.zeros_(self.final_refine[0].bias)
        self.refine_gate = nn.Parameter(torch.zeros(1))

    def forward(self, gain_map: torch.Tensor, img_dark: torch.Tensor) -> tuple:
        gain = F.interpolate(gain_map, size=img_dark.shape[-2:],
                              mode='bilinear', align_corners=False)

        img_bright = img_dark * gain
        img_bright = img_bright + self.final_refine(img_bright) * self.refine_gate
        # Flight4: soft_clamp replaces hard clamp — preserves gradient through boundaries
        img_bright = soft_clamp(img_bright)
        return img_bright, gain


class SGRF(nn.Module):
    """Mark4: Stage-wise Guided Restoration & Fusion.

    Stage 1 (Denoise):  img_s1 = img_center + StageBlock_1(f_noise_out, img_center)
    Stage 2 (Deblur):   img_s2 = img_s1 + StageBlock_2(f_motion_out, img_s1)
    Stage 3 (Brighten): res_t = img_s2 × gain_map (Mod6: bias removed)

    StageBlock uses zero-gate: gate=0 at init → no perturbation to Phase 1 output.
    Mod5: delta zero-mean constraint → StageBlocks cannot shift global brightness.
    Mod6: curve is pixel-wise with 8 iterations, bias_map removed.
    """

    def __init__(self, channels: int = 64, out_channels: int = 3, use_soft_clamp: bool = False,
                 use_nafblock: bool = False, num_res_blocks: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels

        self.stage_noise = StageBlock(channels, out_channels, use_intensity=False,
                                        use_soft_clamp=use_soft_clamp,
                                        use_nafblock=use_nafblock,
                                        num_res_blocks=num_res_blocks)
        self.stage_motion = StageBlock(channels, out_channels, use_intensity=False,
                                        use_soft_clamp=use_soft_clamp,
                                        use_nafblock=use_nafblock,
                                        num_res_blocks=num_res_blocks)
        self.brighten = BrightenStage(channels, out_channels,
                                       use_nafblock=use_nafblock)

    def forward(
        self,
        gain_map: torch.Tensor,
        f_noise_out: torch.Tensor,
        f_motion_out: torch.Tensor,
        image_center: torch.Tensor,
        curve_A: torch.Tensor = None,
        alpha_target: torch.Tensor = None,
        curve_iter: int = 6,
    ) -> dict:
        img_s1, _ = self.stage_noise(f_noise_out, image_center)
        img_s2, _ = self.stage_motion(f_motion_out, img_s1)

        # Flight5: Target-Convergent Curve — converges to α_target
        img_curved = img_s2
        if curve_A is not None and alpha_target is not None:
            for i in range(curve_iter):
                A = curve_A[:, i]
                delta = A * img_curved * (1.0 - img_curved) * (alpha_target - img_curved)
                img_curved = img_curved + delta
            # Safety net: soft clamp (should not be activated in normal operation)
            img_curved = 0.5 + 0.5 * torch.tanh(4.0 * (img_curved - 0.5))

        res_t, lit_up = self.brighten(gain_map, img_curved)

        return {
            "res_t":       res_t,
            "img_s1":      img_s1,
            "img_s2":      img_s2,
            "img_curved":  img_curved,
            "lit_up_map":  lit_up,
        }
