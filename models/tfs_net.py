"""
TFS-Net v3 — Three-source Fusion & Synthesis Network
======================================================
端到端多帧低光增强网络。

整体结构 (5 stages):
    Stage 0: PyramidEncoder        多帧 → 多尺度特征 + 全分辨率融合
    Stage 1: TFSI                  时序光照/噪声/运动强度场估计 + 频/空双分支
    Stage 2: SACE                  可变形跨帧对齐 (与 TFSI 共享 LFF)
    Stage 3: IFPN/NDPN/MRPN        三源恢复分支 (光照/噪声/运动)
    Stage 4: IGRF                  强度引导残差融合 → 输出
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.encoder import PyramidEncoder
from models.modules.tfsi import TFSI
from models.modules.sace import SACE
from models.modules.ifpn import IFPN
from models.modules.ndpn import NDPN
from models.modules.mrpn import MRPN
from models.modules.igrf import IGRF


class TFSNet(nn.Module):
    """
    Args:
        in_channels    : 输入图像通道 (默认 3)
        level_channels : 编码器三尺度通道 (默认 (32, 64, 96))
        fused_channels : 编码器融合输出通道 (默认 48)
        eps            : TFSI 数值稳定项
        n_groups       : SACE 可变形分组数
        kernel_size    : SACE 可变形核大小
        share_lff      : SACE 是否与 TFSI 共享 LFF (默认 True)
    """

    def __init__(
        self,
        in_channels: int = 3,
        level_channels: Tuple[int, ...] = (32, 64, 96, 128),
        fused_channels: int = 64,
        eps: float = 1e-6,
        n_groups: int = 4,
        kernel_size: int = 3,
        share_lff: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.fused_channels = fused_channels
        self.share_lff = share_lff
        # 支持 3 或 4 级编码器
        coarse_channels = level_channels[-1]  # 最粗层通道数

        # Stage 0: PyramidEncoder
        self.encoder = PyramidEncoder(
            in_channels=in_channels,
            level_channels=level_channels,
            fused_channels=fused_channels,
        )

        # Stage 1: TFSI
        self.tfsi = TFSI(
            channels=fused_channels,
            fused_channels=fused_channels,
            eps=eps,
        )

        # Stage 2: SACE (共享 LFF)
        shared_lff = self.tfsi.freq_branch.lff if share_lff else None
        self.sace = SACE(
            channels=fused_channels,
            n_groups=n_groups,
            kernel_size=kernel_size,
            use_optimized=True,
            lff_module=shared_lff,
        )

        # Stage 3: 三源恢复分支
        self.ifpn = IFPN(
            fused_channels=fused_channels,
            coarse_channels=coarse_channels,
            img_channels=in_channels,
        )
        self.ndpn = NDPN(channels=fused_channels)
        self.mrpn = MRPN(channels=fused_channels)

        # Stage 4: IGRF
        self.igrf = IGRF(channels=fused_channels, out_channels=in_channels)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, 3, H, W) 多帧低光输入

        Returns:
            dict with keys: res_t, delta, f_fused_igrf, s_illum, s_noise, s_motion, ...
        """
        B, T, C_in, H, W = x.shape
        center_idx = T // 2

        # Stage 0: 编码
        feats, coarse_feats = self.encoder(x, return_coarse=True)
        # feats        : (B, T, 48, H, W)
        # coarse_feats : (B, T, 96, H/4, W/4)

        # Stage 1: TFSI
        tfsi_out = self.tfsi(feats)
        F_fused  = tfsi_out["F_fused"]
        s_illum  = tfsi_out["s_illum"]
        s_noise  = tfsi_out["s_noise"]
        s_motion = tfsi_out["s_motion"]
        sigma_t  = tfsi_out["sigma_t"]

        # Stage 2: SACE
        sace_out       = self.sace(feats, tfsi_out)
        mu_t_clean     = sace_out["mu_t_clean"]
        F_aligned_list = sace_out["F_aligned_list"]
        attn_maps      = sace_out["attn_maps"]

        # Stage 3: 三源恢复
        image_center = x[:, center_idx]

        _, _, _, hc, wc = coarse_feats.shape
        image_down = F.interpolate(
            image_center, size=(hc, wc), mode='bicubic', align_corners=False
        )
        imgs_down = F.interpolate(
            x.view(B * T, C_in, H, W),
            size=(hc, wc), mode='bicubic', align_corners=False,
        ).view(B, T, C_in, hc, wc)

        F_t_L = coarse_feats[:, center_idx]

        ifpn_out = self.ifpn(
            I_t_down=image_down,
            F_t_L=F_t_L,
            s_illum=s_illum,
            feats=feats,
            coarse_feats=coarse_feats,
            center_idx=center_idx,
            imgs_down=imgs_down,
        )

        ndpn_out = self.ndpn(
            feats=feats,
            F_aligned_list=F_aligned_list,
            mu_t_clean=mu_t_clean,
            sigma_t=sigma_t,
            s_noise=s_noise,
            center_idx=center_idx,
        )

        mrpn_out = self.mrpn(
            feats=feats,
            F_aligned_list=F_aligned_list,
            s_motion=s_motion,
            center_idx=center_idx,
        )

        # Stage 4: IGRF v4.1 — 顺序级联融合（光照→去噪→运动）
        igrf_out = self.igrf(
            f_illum_out=ifpn_out["f_illum_out"],
            f_noise_out=ndpn_out["f_noise_out"],
            f_motion_out=mrpn_out["f_motion_out"],
            image_center=image_center,
        )

        return {
            "res_t":          igrf_out["res_t"],
            "img_s1":         igrf_out["img_s1"],
            "img_s2":         igrf_out["img_s2"],
            "image_center":   image_center,
            "s_illum":        s_illum,
            "s_noise":        s_noise,
            "s_motion":       s_motion,
            "f_illum_out":    ifpn_out["f_illum_out"],
            "f_noise_out":    ndpn_out["f_noise_out"],
            "f_motion_out":   mrpn_out["f_motion_out"],
            "L_t":            ifpn_out["L_t"],
            "L_ref":          ifpn_out["L_ref"],
            "L_ratio":        ifpn_out["L_ratio"],
            "attn_maps":      attn_maps,
            "mu_t_clean":     mu_t_clean,
            "s_snr":          ndpn_out["s_snr"],
            "motion_weights": mrpn_out["motion_weights"],
            "tfsi_out":       tfsi_out,
        }
