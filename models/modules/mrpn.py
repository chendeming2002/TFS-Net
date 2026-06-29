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

from .blocks import (
    ResBlock,
    pad_to_window,
    unpad_from_window,
    window_partition_2d,
    window_partition_video,
    window_reverse_2d,
)


class MRPN(nn.Module):
    def __init__(self, channels=64, window_size=8):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.gate = nn.Conv2d(channels * 2, channels, 1, 1, 0)
        self.refine = ResBlock(channels)

        # Charlie P2: blur_mask 估计器 (sigma_t_clean + 帧间差异 → 模糊感知)
        self.blur_estimator = nn.Sequential(
            nn.Conv2d(channels + 1, channels // 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        # 零初始化最后一个 Conv2d 权重 → 初始 blur_mask=0.5, 等同原始 MRPN
        nn.init.zeros_(self.blur_estimator[-2].weight)
        nn.init.zeros_(self.blur_estimator[-2].bias)

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

    def forward(self, F_aligned_list, center_idx, sigma_t_clean=None):
        """
        Args:
            F_aligned_list: List[Tensor], SACE 对齐后的全帧特征, 每项 (B, C, H, W)
            center_idx: int, 中心帧索引
            sigma_t_clean: (B, C, H, W) or None — Charlie P1: 帧间方差 (运动感知)
        """
        f_t_aligned = F_aligned_list[center_idx]  # (B, C, H, W)

        f_neighbors = torch.stack(
            [F_aligned_list[i] for i in range(len(F_aligned_list)) if i != center_idx],
            dim=1,
        )  # (B, T-1, C, H, W)

        f_omega_aligned = self._aggregate_neighbors(f_t_aligned, f_neighbors)

        # Charlie P1+P2: blur_mask 门控 (sigma_t_clean + 帧间差异 → 模糊感知)
        blur_mask = None
        if sigma_t_clean is not None:
            sigma_1ch = sigma_t_clean.mean(dim=1, keepdim=True)  # (B, 1, H, W)
            frame_diff = (f_omega_aligned - f_t_aligned).abs()   # (B, C, H, W)
            blur_input = torch.cat([sigma_1ch, frame_diff], dim=1)  # (B, C+1, H, W)
            blur_mask = self.blur_estimator(blur_input)

        # 门控融合: gate 决定信任对齐中心帧 vs 聚合邻帧
        z_t = torch.cat([f_t_aligned, f_omega_aligned], dim=1)
        g_t = torch.sigmoid(self.gate(z_t))

        # Charlie P2: blur_mask 调制 — 模糊区域优先使用邻帧补偿
        if blur_mask is not None:
            g_t = g_t * (1.0 - blur_mask) + blur_mask * 0.3  # 模糊区 → 偏向邻帧聚合

        f_t_fuse = g_t * f_t_aligned + (1.0 - g_t) * f_omega_aligned

        hat_f_t = self.refine(f_t_fuse) + f_t_aligned

        return {
            "f_omega_aligned": f_omega_aligned,
            "z_t": z_t,
            "G_t": g_t,
            "f_t_fuse": f_t_fuse,
            "f_motion_out": hat_f_t,
        }
