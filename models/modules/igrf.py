"""
IGRF v4.1 — 顺序级联融合重建 (Sequential Cascade Fusion & Reconstruction)
=========================================================================

v4 问题：concat 一次性融合缺乏物理语义，各分支梯度被稀释。
v4.1 修复：三源分支仍并行计算特征，但在 IGRF 中顺序应用到图像上。
          物理顺序：光照修复 → 去噪 → 运动修复

参考文献：
    - FDN (Tu et al., TIP 2025): 先修复光照(振幅)再修复模糊(相位)
    - MPRNet (Zamir et al., CVPR 2021): 多阶段渐进式修复 + 每阶段独立监督
    - DarkIR (Feijoo et al., CVPR 2025): encoder修复光照(频域), decoder修复模糊(空间域)

数据流：
    image_center
      → StageBlock_illum(f_illum_out, img) → img_s1 (光照修复)
      → StageBlock_noise(f_noise_out, img_s1) → img_s2 (去噪)
      → StageBlock_motion(f_motion_out, img_s2) → res_t (运动修复, 最终输出)
"""

import torch
import torch.nn as nn

from .blocks import ResBlock


class StageBlock(nn.Module):
    """
    单阶段修复块：分支特征 + 当前图像 → 残差 delta → 修复后图像

    设计思路：
        将当前图像编码为条件特征，与分支特征拼接后预测残差。
        网络知道"当前图像长什么样"从而决定"还需要做什么修复"。
    """

    def __init__(self, channels: int, img_channels: int = 3):
        super().__init__()
        # 将当前图像编码为条件特征
        self.img_proj = nn.Conv2d(img_channels, channels, 3, 1, 1)
        # 融合分支特征 + 图像条件 → 残差 delta
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0),
            nn.GELU(),
            ResBlock(channels),
            ResBlock(channels),
            nn.Conv2d(channels, img_channels, 3, 1, 1),
        )

    def forward(self, f_branch: torch.Tensor, img_current: torch.Tensor):
        """
        Args:
            f_branch    : (B, C, H, W) 分支输出特征
            img_current : (B, 3, H, W) 当前阶段图像

        Returns:
            img_next : (B, 3, H, W) 修复后的图像 (clamped to [0,1])
            delta    : (B, 3, H, W) 残差
        """
        img_feat = self.img_proj(img_current)  # (B, C, H, W)
        combined = torch.cat([f_branch, img_feat], dim=1)  # (B, 2C, H, W)
        delta = self.fuse(combined)  # (B, 3, H, W)
        img_next = torch.clamp(img_current + delta, 0.0, 1.0)
        return img_next, delta


class IGRF(nn.Module):
    """
    IGRF v4.1 — 顺序级联融合重建模块

    三个独立的 StageBlock 按物理修复顺序依次处理：
        Stage 1: 光照修复 (IFPN 特征)
        Stage 2: 去噪修复 (NDPN 特征)
        Stage 3: 运动修复 (MRPN 特征)

    每个阶段输出中间图像，可供独立监督（梯度直接回传到对应分支）。
    """

    def __init__(self, channels: int = 64, out_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels

        self.stage_illum = StageBlock(channels, out_channels)   # 光照修复
        self.stage_noise = StageBlock(channels, out_channels)   # 去噪修复
        self.stage_motion = StageBlock(channels, out_channels)  # 运动修复

    def forward(
        self,
        f_illum_out: torch.Tensor,
        f_noise_out: torch.Tensor,
        f_motion_out: torch.Tensor,
        image_center: torch.Tensor,
        # 保留接口兼容（不使用）
        f_t_base: torch.Tensor = None,
        s_illum: torch.Tensor = None,
        s_noise: torch.Tensor = None,
        s_motion: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            f_illum_out  : (B, C, H, W) IFPN 输出特征
            f_noise_out  : (B, C, H, W) NDPN 输出特征
            f_motion_out : (B, C, H, W) MRPN 输出特征
            image_center : (B, 3, H, W) 中心帧原始图像

        Returns:
            dict with keys: res_t, img_s1, img_s2, delta_s1, delta_s2, delta_s3
        """
        # Stage 1: 光照修复
        img_s1, delta_s1 = self.stage_illum(f_illum_out, image_center)

        # Stage 2: 去噪（在已修复光照的图像上）
        img_s2, delta_s2 = self.stage_noise(f_noise_out, img_s1)

        # Stage 3: 运动修复（在已去噪的图像上）
        res_t, delta_s3 = self.stage_motion(f_motion_out, img_s2)

        return {
            "res_t": res_t,
            "img_s1": img_s1,
            "img_s2": img_s2,
            "delta_s1": delta_s1,
            "delta_s2": delta_s2,
            "delta_s3": delta_s3,
        }
