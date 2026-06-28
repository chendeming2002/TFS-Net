from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.encoder import PyramidEncoder
from models.modules.tfsi import TFSI
from models.modules.pure_rwkv_sace import PureRWKVSACE
from models.modules.ifpn import IFPN
from models.modules.ndpn import NDPN
from models.modules.mrpn import MRPN
from models.modules.igrf import IGRF
from models.modules.dwt_lff import SpatialDWTLFFAdapter


class TFSNet(nn.Module):
    """
    TFSNet — RWKV-Bravo Variant (v6.5 Bravo)
    ==========================================
    RWKV-Bravo 是 v6.5 的架构精简版:

      Stage 0: PyramidEncoder        多帧 → 多尺度特征 + 全分辨率融合
      Stage 1: TFSI                  时序光照/噪声强度场估计 (Bravo: phase_conf)
      Stage 2: PureRWKVSACE          纯 RWKV 多尺度帧间对齐 (替代 DAT)
      Stage 3: IFPN/NDPN/MRPN        三源恢复分支 (光照/噪声/运动)
      Stage 4: IGRF                  强度引导残差融合 → 输出

    Bravo 移除组件:
      - AmpEnhance (v5.9 前置幅度增强, 废弃, 节省 91K)
      - DeformableCrossAttention / OffsetMaskHead (v3 可变形对齐, 节省 130K)
      - CrossRWKVGate (v6 中间门控, 由 PureRWKVSACE 直接替代)
      - SACE / share_lff / sace_* 参数

    Bravo 新增:
      - TFSI.phase_conf_head: 相位置信度调制 s_noise (FDN 2025)
      - 双实例 DWT-LFF: center α_init=0.6 / neighbor α_init=0.4
      - PureRWKVSACE 三尺度双向 RWKV + 边缘门控

    API:
        forward(x: (B,T,3,H,W)) → dict
    """

    def __init__(
        self,
        in_channels: int = 3,
        level_channels: Tuple[int, ...] = (32, 64, 96),
        fused_channels: int = 64,
        eps: float = 1e-6,
        use_soft_clamp: bool = True,
        use_soft_median: bool = True,
        use_nafblock: bool = False,
        num_bottleneck_blocks: int = 0,
        num_igrf_res_blocks: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.fused_channels = fused_channels
        coarse_channels = level_channels[-1]

        # Stage 0: PyramidEncoder
        self.encoder = PyramidEncoder(
            in_channels=in_channels,
            level_channels=level_channels,
            fused_channels=fused_channels,
            num_bottleneck_blocks=num_bottleneck_blocks,
        )

        # Bravo P1: 双实例 DWT-LFF — center (α=0.6) vs neighbor (α=0.4)
        self.dwt_lff_center = SpatialDWTLFFAdapter(
            in_channels=fused_channels, alpha_init=0.6
        )
        self.dwt_lff_neighbor = SpatialDWTLFFAdapter(
            in_channels=fused_channels, alpha_init=0.4
        )

        # Stage 1: TFSI — 使用 center DWT-LFF (feat_tfsi 流向 TFSI 频域分支)
        self.tfsi = TFSI(
            channels=fused_channels,
            fused_channels=fused_channels,
            eps=eps,
            use_soft_median=use_soft_median,
            dwt_lff=self.dwt_lff_center,
        )

        # Stage 2: PureRWKVSACE — 使用 dual DWT-LFF 传入
        self.sace = PureRWKVSACE(
            channels=fused_channels,
            lff_center=self.dwt_lff_center,
            lff_neighbor=self.dwt_lff_neighbor,
            n_layer=1,
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
        self.igrf = IGRF(
            channels=fused_channels,
            out_channels=in_channels,
            use_soft_clamp=use_soft_clamp,
            use_nafblock=use_nafblock,
            num_res_blocks=num_igrf_res_blocks,
        )

        self.frame_cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def clear_frame_cache(self):
        self.frame_cache.clear()

    def forward(
        self,
        x: torch.Tensor,
        frame_indices: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, C_in, H, W = x.shape
        center_idx = T // 2

        # Stage 0: 编码
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

        # Stage 1: TFSI (使用 center DWT-LFF)
        tfsi_out = self.tfsi(feats)
        F_fused  = tfsi_out["F_fused"]
        s_illum  = tfsi_out["s_illum"]
        s_noise  = tfsi_out["s_noise"]

        # Stage 2: PureRWKVSACE (使用 dual DWT-LFF)
        sace_out = self.sace(feats, tfsi_out)
        mu_t_clean     = sace_out["mu_t_clean"]
        F_aligned_list = sace_out["F_aligned_list"]

        # Stage 3: 三源恢复
        image_center = x[:, center_idx]
        aligned_feats = torch.stack(F_aligned_list, dim=1)

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
            aligned_feats=aligned_feats,
            center_idx=center_idx,
            imgs_down=imgs_down,
        )

        ndpn_out = self.ndpn(
            feats=feats,
            F_aligned_list=F_aligned_list,
            mu_t_clean=mu_t_clean,
            sigma_t_clean=sace_out["sigma_t_clean"],
            s_noise=s_noise,
            center_idx=center_idx,
        )

        mrpn_out = self.mrpn(
            F_aligned_list=F_aligned_list,
            center_idx=center_idx,
        )

        # Stage 4: IGRF
        igrf_out = self.igrf(
            f_illum_feat=ifpn_out["f_illum_feat"],
            f_noise_out=ndpn_out["f_noise_out"],
            f_motion_out=mrpn_out["f_motion_out"],
            lit_up_map_raw=ifpn_out["lit_up_map_raw"],
            image_center=image_center,
            s_illum=s_illum,
            s_noise=s_noise,
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
            "ifpn_side":      ifpn_out.get("ifpn_side"),
            "attn_maps":      [],
            "mu_t_clean":     mu_t_clean,
            "s_snr":          ndpn_out["s_snr"],
            "motion_weights": mrpn_out["G_t"],
            "tfsi_out":       tfsi_out,
        }
