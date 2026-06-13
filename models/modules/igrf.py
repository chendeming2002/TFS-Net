"""
IGRF — 强度引导融合 (Intensity-Guided Reconstruction Fusion)
==============================================================

v3 设计：替代 FinalReconstruction，将三分支输出按 TFSI 强度加权融合后重建。

当前实现状态：
  - 融合公式：✅ 已实现
  - F_t^base 假设：⚠️ 暂用编码器融合特征 F_t 作为 base（详见 v3quest.md § 1.7）
  - 三分支通道一致性：假设 IFPN/NDPN/MRPN 输出均为 C_f 通道

设计参考：TFSv3-result.md § 4.6

公式：
    F_fused = s_illum · F_t^{illum_out}
            + s_noise · F_t^{noise_out}
            + s_motion · F_t^{motion_out}
            + F_t^{base}

    Î_t = clamp(I_t + Conv3x3(GELU(Conv3x3(F_fused))), 0, 1)
"""

import torch
import torch.nn as nn


class IGRF(nn.Module):
    """
    IGRF 强度引导融合模块

    输入：
        - F_t_base     : (B, C_f, H, W) base 特征（当前假设 = 编码器融合特征 F_t）
        - F_illum_out  : (B, C_f, H, W) IFPN 输出
        - F_noise_out  : (B, C_f, H, W) NDPN 输出
        - F_motion_out : (B, C_f, H, W) MRPN 输出
        - s_illum      : (B, 1, H, W)   光照强度，∈ [0,1]
        - s_noise      : (B, 1, H, W)   噪声强度，∈ [0,1]
        - s_motion     : (B, 1, H, W)   运动强度，∈ [0,1]
        - image_center : (B, 3, H, W)   中心帧原始图像

    输出：
        - res_t : (B, 3, H, W) 增强帧，clamp 到 [0,1]
        - delta : (B, 3, H, W) 残差修正量
        - f_fused : (B, C_f, H, W) 融合后特征（供调试/可视化）
    """

    def __init__(self, channels: int = 48, out_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        # 两层卷积将融合特征映射到残差修正
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, out_channels, 3, 1, 1)

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
        """
        注：所有强度 s_* shape 为 (B, 1, H, W)，通过广播与 (B, C_f, H, W) 特征相乘。
        """
        # 强度加权融合
        # F_fused = s_illum·F_illum + s_noise·F_noise + s_motion·F_motion + F_base
        f_fused = (
            s_illum * f_illum_out
            + s_noise * f_noise_out
            + s_motion * f_motion_out
            + f_t_base
        )  # (B, C_f, H, W)

        # 残差重建
        u_t = self.act(self.conv1(f_fused))   # (B, C_f, H, W)
        delta = self.conv2(u_t)               # (B, 3, H, W)
        res_t = torch.clamp(image_center + delta, 0.0, 1.0)  # (B, 3, H, W)

        return {
            "res_t": res_t,
            "delta": delta,
            "f_fused": f_fused,
        }
