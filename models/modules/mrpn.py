"""
MRPN (Motion-Refining Pyramid Network) — TFS-Net v5 → v6 Charlie
=================================================================
基于 MSPN (MINS-Net) 结构的窗口相关 + 门控融合运动聚合。

Charlie P1 (σ→MRPN): sigma_t_clean 作为模糊感知输入, 高方差 → 运动 → 增加去模糊强度
Charlie P2 (blur_mask): sigma_t_clean + corr → blur_estimator → 门控自适应聚合
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    ResBlock,
    pad_to_window,
    unpad_from_window,
    window_partition_2d,
    window_partition_video,
    window_reverse_2d,
)


class MCPN(nn.Module):
    def __init__(self, channels=64, window_size=8, num_frames=5):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.num_frames = num_frames
        self.gate = nn.Conv2d(channels * 2, channels, 1, 1, 0)
        self.refine = ResBlock(channels)

        self.blur_estimator = nn.Sequential(
            nn.Conv2d(channels + 1, channels // 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.blur_estimator[-2].weight)
        nn.init.zeros_(self.blur_estimator[-2].bias)

        # Delta: motion magnitude estimator from C_omega diagonals
        self.motion_estimator = nn.Sequential(
            nn.Conv2d(num_frames - 1, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid(),
        )

        # Delta: sigma projection (channel-wise, not frame-wise)
        self.sigma_proj = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

        # Delta: compensation gate (motion magnitude gated)
        self.comp_gate = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

        # Delta: motion refinement (F_t_aligned vs center delta)
        self.motion_refine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

        # Flight3: γ=0.001 → gradient flows through motion branch without visible output
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 0.05))

        # Mark3: smooth startup — bias g_t toward 1.0 (favor f_center)
        #         and suppress outer residual at Phase 2 entry
        self.startup_gate = nn.Parameter(torch.ones(1))
        self.out_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # Zero-init refine conv2 → pass-through at init
        nn.init.zeros_(self.refine.conv2.weight)
        nn.init.zeros_(self.refine.conv2.bias)

    def reset_startup(self):
        """Mark3: 从旧 checkpoint 续训时重置 MCPN 平滑启动参数"""
        self.gamma.data.fill_(0.05)  # Flight7: larger initial for gradient-isolated Stage B
        self.startup_gate.data.fill_(1.0)
        self.out_scale.data.zero_()
        nn.init.zeros_(self.refine.conv2.weight)
        nn.init.zeros_(self.refine.conv2.bias)

    def _aggregate_neighbors(self, f_t, f_omega):
        """窗口 dot-product 相关聚合相邻帧（不含中心帧）。"""
        b, t, c, h, w = f_omega.shape
        feat = f_omega.reshape(b * t, c, h, w)
        feat, pad_hw = pad_to_window(feat, self.window_size)
        hp, wp = feat.shape[-2:]
        feat = feat.reshape(b, t, c, hp, wp)

        f_t_padded, _ = pad_to_window(f_t, self.window_size)

        center_windows = window_partition_2d(f_t_padded, self.window_size)
        feat_windows = window_partition_video(feat, self.window_size)

        corr = torch.matmul(center_windows, feat_windows.transpose(-1, -2)) / math.sqrt(c)
        corr = torch.softmax(corr, dim=-1)

        aligned_windows = torch.matmul(corr, feat_windows)
        aligned = window_reverse_2d(aligned_windows, self.window_size, hp, wp)
        aligned = unpad_from_window(aligned, pad_hw)
        return aligned

    def forward(self, F_aligned_list, center_idx, sigma_t_clean=None,
                C_omega_list=None, F_t_aligned=None):
        f_t_aligned = F_aligned_list[center_idx]
        f_neighbors = torch.stack(
            [F_aligned_list[i] for i in range(len(F_aligned_list)) if i != center_idx],
            dim=1,
        )

        # Delta: use F_t_aligned as alignment reference
        f_center = F_t_aligned if F_t_aligned is not None else f_t_aligned

        f_omega_aligned = self._aggregate_neighbors(f_center, f_neighbors)

        # Delta: motion magnitude from C_omega_list (full motion_estimator)
        motion_mag = None
        if C_omega_list is not None and len(C_omega_list) > 0:
            diag_vals = []
            for C_t in C_omega_list:
                diag = C_t.diagonal(dim1=-2, dim2=-1)  # (B, N)
                diag_vals.append(diag)
            diag_stack = torch.stack(diag_vals, dim=1)  # (B, T-1, N)
            B_val = diag_stack.shape[0]
            ds = int(diag_stack.shape[-1] ** 0.5)
            H_ref, W_ref = f_t_aligned.shape[-2:]
            diag_2d = diag_stack.reshape(B_val, len(C_omega_list), ds, ds)
            motion_mag = self.motion_estimator(diag_2d)  # (B, 1, ds, ds)
            motion_mag = F.interpolate(motion_mag, size=(H_ref, W_ref),
                                       mode='bilinear', align_corners=False)

        # Delta: use F_t_aligned as alignment reference
        f_center = F_t_aligned if F_t_aligned is not None else f_t_aligned

        # Delta: sigma projection (frame-level → per-channel)
        sigma_feat = None
        if sigma_t_clean is not None:
            B_val = sigma_t_clean.shape[0]
            sigma_flat = sigma_t_clean.mean(dim=(2, 3))  # (B, C)
            sigma_feat = self.sigma_proj(sigma_flat).view(B_val, 1, self.channels, 1, 1)

        # blur_mask gate
        blur_mask = None
        if sigma_t_clean is not None:
            sigma_1ch = sigma_t_clean.mean(dim=1, keepdim=True)
            frame_diff = (f_omega_aligned - f_center).abs()
            blur_input = torch.cat([sigma_1ch, frame_diff], dim=1)
            blur_mask = self.blur_estimator(blur_input)

        # Delta: motion refinement from F_t_aligned vs center delta
        motion_delta = self.motion_refine(
            torch.cat([f_center, f_omega_aligned], dim=1)
        )

        z_t = torch.cat([f_center, f_omega_aligned], dim=1)
        g_t = torch.sigmoid(self.gate(z_t))

        # Delta: compensation gate modulated by motion magnitude
        comp = self.comp_gate(torch.cat([f_center, motion_mag], dim=1)) \
            if motion_mag is not None else torch.sigmoid(self.gate(z_t))

        # Delta: blur + motion modulated gate
        if blur_mask is not None:
            g_t = g_t * (1.0 - blur_mask) + blur_mask * 0.3
        if motion_mag is not None:
            g_t = g_t * (1.0 - motion_mag) + motion_mag * 0.3

        # Mark3: startup gate — bias toward f_center at Phase 2 entry
        startup = self.startup_gate.clamp(0, 1)
        g_t = g_t * (1 - startup) + startup

        # Delta: gamma-scaled motion refinement + gate fusion
        f_t_fuse = g_t * f_center + (1.0 - g_t) * f_omega_aligned \
                   + motion_delta * comp * self.gamma
        # Mark3: out_scale removes outer +f_center at startup (refine already has skip)
        hat_f_t = self.refine(f_t_fuse) + f_center * self.out_scale

        return {
            "f_omega_aligned": f_omega_aligned,
            "z_t": z_t,
            "G_t": g_t,
            "f_t_fuse": f_t_fuse,
            "f_motion_out": hat_f_t,
        }
