"""
IGRF — 强度引导融合 (Intensity-Guided Reconstruction Fusion)
==============================================================

v3.1 改进：
  - 融合公式改为归一化加权（解决不对称问题）
  - 重建头加4个 ResBlock（解决过浅问题）

公式：
    w_total = s_illum + s_noise + s_motion + 1
    F_fused = (s_illum / w_total) * F_illum
            + (s_noise / w_total) * F_noise
            + (s_motion / w_total) * F_motion
            + (1 / w_total) * F_base

    Î_t = clamp(I_t + ReconHead(F_fused), 0, 1)
"""

import torch
import torch.nn as nn

from .blocks import ResBlock


class IGRF(nn.Module):
    """
    IGRF 强度引导融合模块 (v3.1)

    输入：
        - f_t_base     : (B, C_f, H, W) base 特征
        - f_illum_out  : (B, C_f, H, W) IFPN 输出
        - f_noise_out  : (B, C_f, H, W) NDPN 输出
        - f_motion_out : (B, C_f, H, W) MRPN 输出
        - s_illum      : (B, 1, H, W)   光照强度
        - s_noise      : (B, 1, H, W)   噪声强度
        - s_motion     : (B, 1, H, W)   运动强度
        - image_center : (B, 3, H, W)   中心帧原始图像
    """

    def __init__(self, channels: int = 64, out_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels

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
        s_illum: torch.Tensor,
        s_noise: torch.Tensor,
        s_motion: torch.Tensor,
        image_center: torch.Tensor,
    ) -> dict:
        # 归一化加权融合：保证所有权重加和为1
        w_total = s_illum + s_noise + s_motion + 1.0  # (B, 1, H, W)
        w_illum = s_illum / w_total
        w_noise = s_noise / w_total
        w_motion = s_motion / w_total
        w_base = 1.0 / w_total

        f_fused = (
            w_illum * f_illum_out
            + w_noise * f_noise_out
            + w_motion * f_motion_out
            + w_base * f_t_base
        )  # (B, C_f, H, W)

        # 残差重建
        delta = self.recon_head(f_fused)  # (B, 3, H, W)
        res_t = torch.clamp(image_center + delta, 0.0, 1.0)

        return {
            "res_t": res_t,
            "delta": delta,
            "f_fused": f_fused,
        }
