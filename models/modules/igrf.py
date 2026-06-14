"""
IGRF v4 — 拼接融合重建 (Concatenation Fusion & Reconstruction)
================================================================

v3 问题：归一化加权融合 (w_base≈0.44, branches≈0.12-0.32) 严重衰减分支梯度。
v4 修复：移除加权融合，改为 concat → Conv → 重建。
        所有分支获得等强度梯度，由卷积自动学习最优组合。

数据流：
    [F_base, F_illum, F_noise, F_motion] → Conv(4C→C) → 4×ResBlock → delta
    res_t = clamp(image_center + delta, 0, 1)
"""

import torch
import torch.nn as nn

from .blocks import ResBlock


class IGRF(nn.Module):
    """
    IGRF v4 — 拼接融合重建模块

    输入：
        - f_t_base     : (B, C_f, H, W) base 特征
        - f_illum_out  : (B, C_f, H, W) IFPN 输出
        - f_noise_out  : (B, C_f, H, W) NDPN 输出
        - f_motion_out : (B, C_f, H, W) MRPN 输出
        - image_center : (B, 3, H, W)   中心帧原始图像

    注：s_illum/s_noise/s_motion 不再用于融合加权，
        仅作为 TFSI 的中间产物供 loss 监督。
    """

    def __init__(self, channels: int = 64, out_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels

        # 拼接投影：4C → C（学习最优融合权重）
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, 1, 0),
            nn.GELU(),
        )

        # 深化重建头：4个 ResBlock + 输出卷积
        self.recon_head = nn.Sequential(
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels),
            nn.Conv2d(channels, out_channels, 3, 1, 1),
        )

    def forward(
        self,
        f_t_base: torch.Tensor,
        f_illum_out: torch.Tensor,
        f_noise_out: torch.Tensor,
        f_motion_out: torch.Tensor,
        s_illum: torch.Tensor = None,
        s_noise: torch.Tensor = None,
        s_motion: torch.Tensor = None,
        image_center: torch.Tensor = None,
    ) -> dict:
        # 拼接融合：所有分支等权参与，梯度无衰减
        f_cat = torch.cat(
            [f_t_base, f_illum_out, f_noise_out, f_motion_out], dim=1
        )  # (B, 4*C_f, H, W)
        f_fused = self.fuse_proj(f_cat)  # (B, C_f, H, W)

        # 残差重建
        delta = self.recon_head(f_fused)  # (B, 3, H, W)
        res_t = torch.clamp(image_center + delta, 0.0, 1.0)

        return {
            "res_t": res_t,
            "delta": delta,
            "f_fused": f_fused,
        }
