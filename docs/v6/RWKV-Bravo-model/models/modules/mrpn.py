"""
MRPN — Motion-Refining Pyramid Network
=======================================
窗口点积相关 + 门控融合的运动聚合.
"""

import math

import torch
import torch.nn as nn

from .blocks import ResBlock, pad_to_window, unpad_from_window, window_partition_2d, window_partition_video, window_reverse_2d


class MRPN(nn.Module):
    def __init__(self, channels=64, window_size=8):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.gate = nn.Conv2d(channels * 2, channels, 1, 1, 0)
        self.refine = ResBlock(channels)

    def _aggregate_neighbors(self, f_t, f_omega):
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

    def forward(self, F_aligned_list, center_idx):
        f_t_aligned = F_aligned_list[center_idx]
        f_neighbors = torch.stack(
            [F_aligned_list[i] for i in range(len(F_aligned_list)) if i != center_idx], dim=1,
        )

        f_omega_aligned = self._aggregate_neighbors(f_t_aligned, f_neighbors)
        z_t = torch.cat([f_t_aligned, f_omega_aligned], dim=1)
        g_t = torch.sigmoid(self.gate(z_t))
        f_t_fuse = g_t * f_t_aligned + (1.0 - g_t) * f_omega_aligned
        hat_f_t = self.refine(f_t_fuse) + f_t_aligned

        return {
            "f_omega_aligned": f_omega_aligned,
            "G_t": g_t,
            "f_t_fuse": f_t_fuse,
            "f_motion_out": hat_f_t,
        }
