"""
MRPN (Motion-Refining Pyramid Network) — TFS-Net v5
=====================================================
基于 MSPN (MINS-Net) 结构的窗口相关 + 门控融合运动聚合。

核心设计:
    使用 SACE 对齐后的特征（而非原始 Encoder 特征），确保 f_t 与 f_omega 同域。
    聚合仅来自对齐后的相邻帧（排除中心帧），避免中心帧自身参与聚合。

与 MSPN 的差异:
    - corr 由模块内部从 SACE 对齐特征计算（MSPN 由 MINS 外部提供）
    - 输入为 SACE 预对齐的 F_aligned_list（MSPN 为原始邻帧 f_omega_m）

数据流:
    f_t_aligned = F_aligned_list[center_idx]          (SACE 对齐后的中心帧)
    f_neighbors = F_aligned_list \\ {center_idx}       (仅对齐后的相邻帧)
    1. corr = softmax(f_t_aligned_win · f_neighbors_win^T / √C)  (窗口内相关性)
    2. f_agg = corr · f_neighbors_win                (加权聚合邻帧)
    3. g = σ(Conv1×1([f_t_aligned ∥ f_agg]))          (门控融合)
    4. f_fuse = g ⊙ f_t_aligned + (1-g) ⊙ f_agg
    5. hat_f_t = Refine(f_fuse) + f_t_aligned         (残差精炼)
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

    def _aggregate_neighbors(self, f_t, f_omega):
        """窗口 dot-product 相关聚合相邻帧（不含中心帧）。"""
        b, t, c, h, w = f_omega.shape
        feat = f_omega.view(b * t, c, h, w)
        feat, pad_hw = pad_to_window(feat, self.window_size)
        hp, wp = feat.shape[-2:]
        feat = feat.view(b, t, c, hp, wp)

        # center frame: pad + window partition
        f_t_padded, _ = pad_to_window(f_t, self.window_size)

        # window partition
        center_windows = window_partition_2d(f_t_padded, self.window_size)
        # (b, n_w, ws², c)
        feat_windows = window_partition_video(feat, self.window_size)
        # (b, n_w, t*ws², c)

        # dot-product correlation (replaces external corr from MINS)
        corr = torch.matmul(center_windows, feat_windows.transpose(-1, -2)) / math.sqrt(c)
        corr = torch.softmax(corr, dim=-1)

        aligned_windows = torch.matmul(corr, feat_windows)
        aligned = window_reverse_2d(aligned_windows, self.window_size, hp, wp)
        aligned = unpad_from_window(aligned, pad_hw)
        return aligned

    def forward(self, F_aligned_list, center_idx):
        """
        Args:
            F_aligned_list: List[Tensor], SACE 对齐后的全帧特征, 每项 (B, C, H, W)
            center_idx: int, 中心帧索引
        """
        f_t_aligned = F_aligned_list[center_idx]  # (B, C, H, W)

        # 仅保留相邻帧（排除中心帧），避免中心帧自聚合
        f_neighbors = torch.stack(
            [F_aligned_list[i] for i in range(len(F_aligned_list)) if i != center_idx],
            dim=1,
        )  # (B, T-1, C, H, W)

        f_omega_aligned = self._aggregate_neighbors(f_t_aligned, f_neighbors)

        # 门控融合: gate 决定信任对齐中心帧 vs 聚合邻帧
        z_t = torch.cat([f_t_aligned, f_omega_aligned], dim=1)
        g_t = torch.sigmoid(self.gate(z_t))
        f_t_fuse = g_t * f_t_aligned + (1.0 - g_t) * f_omega_aligned

        # 残差精炼 (ResBlock 内部有残差, +f_t_aligned 为第二层恒等跳跃)
        hat_f_t = self.refine(f_t_fuse) + f_t_aligned

        return {
            "f_omega_aligned": f_omega_aligned,
            "z_t": z_t,
            "G_t": g_t,
            "f_t_fuse": f_t_fuse,
            "f_motion_out": hat_f_t,
        }
