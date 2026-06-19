"""
TFS-Net v3 — Three-source Fusion & Synthesis Network
======================================================
端到端多帧低光增强网络。

整体结构 (5 stages):
    Stage 0: PyramidEncoder        多帧 → 多尺度特征 + 全分辨率融合
    Stage 1: TFSI                  时序光照/噪声强度场估计 + 频/空双分支
    Stage 2: SACE                  可变形跨帧对齐 (与 TFSI 共享 LFF)
    Stage 3: IFPN/NDPN/MRPN        三源恢复分支 (光照/噪声/运动)
    Stage 4: IGRF                  强度引导残差融合 → 输出
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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

        # 逐帧特征缓存 (推理时滑动窗口复用)
        self.frame_cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def clear_frame_cache(self):
        """清空逐帧特征缓存（切换序列或释放显存时调用）。"""
        self.frame_cache.clear()

    def forward(
        self,
        x: torch.Tensor,
        frame_indices: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, 3, H, W) 多帧低光输入

        Returns:
            dict with keys: res_t, delta, f_fused_igrf, s_illum, s_noise, ...
        """
        B, T, C_in, H, W = x.shape
        center_idx = T // 2

        # Stage 0: 编码（支持逐帧缓存）
        feats_list: List[torch.Tensor] = []
        for i in range(T):
            gidx = frame_indices[i] if frame_indices else None
            if gidx is not None and gidx in self.frame_cache:
                feats_list.append(self.frame_cache[gidx]["feat"])
            else:
                f = self.encoder.forward_single(x[:, i], return_coarse=False)
                feats_list.append(f)
                if gidx is not None:
                    self.frame_cache[gidx] = {"feat": f}
        feats = torch.stack(feats_list, dim=1)

        # Stage 1: TFSI
        tfsi_out = self.tfsi(feats)
        F_fused  = tfsi_out["F_fused"]
        s_illum  = tfsi_out["s_illum"]
        s_noise  = tfsi_out["s_noise"]
        sigma_t  = tfsi_out["sigma_t"]

        # Stage 2: SACE（LFF 支持逐帧缓存）
        cached_lff: Dict[int, torch.Tensor] = {}
        if frame_indices:
            for i, gidx in enumerate(frame_indices):
                if gidx in self.frame_cache and "lff" in self.frame_cache[gidx]:
                    cached_lff[i] = self.frame_cache[gidx]["lff"]

        sace_out = self.sace(
            feats, tfsi_out,
            cached_lff=cached_lff if cached_lff else None,
        )

        # 将 SACE 返回的 lff_feats 存入缓存
        if frame_indices and "lff_feats" in sace_out:
            for i, gidx in enumerate(frame_indices):
                if gidx in self.frame_cache:
                    self.frame_cache[gidx]["lff"] = sace_out["lff_feats"][i]

        mu_t_clean     = sace_out["mu_t_clean"]
        F_aligned_list = sace_out["F_aligned_list"]
        attn_maps      = sace_out["attn_maps"]

        # Stage 3: 三源恢复
        image_center = x[:, center_idx]

        # v5.3: IFPN 改用 SACE 对齐特征（不再使用 Encoder 粗特征）
        aligned_feats = torch.stack(F_aligned_list, dim=1)  # (B, T, C_f, H, W)

        # 下采样图像到粗特征分辨率（H/4, W/4 由编码器结构决定）
        h_c, w_c = H // 4, W // 4
        image_down = F.interpolate(
            image_center, size=(h_c, w_c), mode='bicubic', align_corners=False
        )
        imgs_down = F.interpolate(
            x.view(B * T, C_in, H, W),
            size=(h_c, w_c), mode='bicubic', align_corners=False,
        ).view(B, T, C_in, h_c, w_c)

        ifpn_out = self.ifpn(
            I_t_down=image_down,
            s_illum=s_illum,
            aligned_feats=aligned_feats,
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
            F_aligned_list=F_aligned_list,
            center_idx=center_idx,
        )

        # Stage 4: IGRF v4.3 - Denoise -> Motion -> Bounded Brighten
        igrf_out = self.igrf(
            f_illum_feat=ifpn_out["f_illum_feat"],
            f_noise_out=ndpn_out["f_noise_out"],
            f_motion_out=mrpn_out["f_motion_out"],
            lit_up_map_raw=ifpn_out["lit_up_map_raw"],
            image_center=image_center,
        )

        return {
            "res_t":          igrf_out["res_t"],
            "img_s1":         igrf_out["img_s1"],
            "img_s2":         igrf_out["img_s2"],
            "lit_up_map":     igrf_out["lit_up_map"],
            "image_center":   image_center,
            "s_illum":        s_illum,
            "s_noise":        s_noise,
            "f_illum_feat":   ifpn_out["f_illum_feat"],
            "f_noise_out":    ndpn_out["f_noise_out"],
            "f_motion_out":   mrpn_out["f_motion_out"],
            "L_t":            ifpn_out["L_t"],
            "L_ref":          ifpn_out["L_ref"],
            "L_ratio":        ifpn_out["L_ratio"],
            "attn_maps":      attn_maps,
            "mu_t_clean":     mu_t_clean,
            "s_snr":          ndpn_out["s_snr"],
            "motion_weights": mrpn_out["G_t"],
            "tfsi_out":       tfsi_out,
        }
