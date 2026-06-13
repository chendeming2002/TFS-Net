"""
MRPN (Motion-Refining Pyramid Network) — TFS-Net v3
=====================================================
基于残差驱动隐式遮挡的运动鲁棒聚合。

实现状态: ✅ 完整实现

数据流:
    1. R_i = |F_i^aligned - F_t| (运动残差)
    2. w_i_raw = sigmoid(-Conv(R_i)) (大残差→小权重)
       w_t_raw = 1.0 (中心帧固定)
    3. F_motion_agg = Σ (w_i/Σw) * F_i^aligned
    4. f_motion_out = s_motion * Refine(F_motion_agg) + (1-s_motion) * F_t
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.blocks import ConvBlock


class MRPN(nn.Module):
    """
    Args:
        channels: 特征通道数 (默认 64)
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.channels = channels

        self.weight_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.weight_conv[-1].weight)
        nn.init.zeros_(self.weight_conv[-1].bias)

        self.refine = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1, act=True),
            ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1, act=False),
        )

    def forward(
        self,
        feats: torch.Tensor,
        F_aligned_list: List[torch.Tensor],
        s_motion: torch.Tensor,
        center_idx: int,
    ) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = feats.shape
        assert C == self.channels

        F_t = feats[:, center_idx]

        # Step 1-2: 计算每帧权重
        weights_raw: List[torch.Tensor] = []

        for i in range(T):
            F_i_aligned = F_aligned_list[i]
            if i == center_idx:
                w_i = torch.ones(B, 1, H, W, device=F_t.device, dtype=F_t.dtype)
            else:
                R_i = (F_i_aligned - F_t).abs()
                logits_i = self.weight_conv(R_i)
                w_i = torch.sigmoid(-logits_i)
            weights_raw.append(w_i)

        # Step 3: 归一化
        eps = 1e-6
        w_sum = torch.stack(weights_raw, dim=1).sum(dim=1) + eps

        F_motion_agg = torch.zeros_like(F_t)
        weights_norm: List[torch.Tensor] = []
        for i in range(T):
            w_i_norm = weights_raw[i] / w_sum
            weights_norm.append(w_i_norm)
            F_motion_agg = F_motion_agg + w_i_norm * F_aligned_list[i]

        # Step 4: refine + 直接输出（移除双重门控，由 IGRF 统一做强度加权）
        F_motion_refined = self.refine(F_motion_agg)
        f_motion_out = F_motion_refined

        motion_weights = torch.cat(
            [weights_norm[i] for i in range(T) if i != center_idx], dim=1
        )

        return {
            "f_motion_out":   f_motion_out,
            "motion_weights": motion_weights,
        }
