"""
PureRWKVSACE — v6.5 纯 RWKV 帧间对齐
======================================
Bravo P1 修正:
  - 双 DWT-LFF: center (α=0.6, VSRELL CVPR 2026) / neighbor (α=0.4)
  - V 投影使用 Encoder 原始中心帧 (STCD IJCAI 2025)
  - 三尺度双向 RWKV (ABMamba AHBS + Otter TRM)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

from .blocks import LayerNorm2d
from .dwt_lff import SpatialDWTLFFAdapter
from .cross_rwkv import VRWKVStyleSpatialMix


class PureRWKVSACE(nn.Module):
    """
    Args:
        channels: 特征通道数
        lff_center: center frame DWT-LFF (α≈0.6, 干净光照锚点)
        lff_neighbor: neighbor frame DWT-LFF (α≈0.4, 退化信号)
        n_layer: RWKV 层数
    """

    def __init__(
        self,
        channels: int = 64,
        lff_center: SpatialDWTLFFAdapter = None,
        lff_neighbor: SpatialDWTLFFAdapter = None,
        n_layer: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.lff_center = lff_center
        self.lff_neighbor = lff_neighbor

        self.rwkv_full = VRWKVStyleSpatialMix(channels, num_frames=5, layer_id=0.0, n_layer=n_layer)
        self.rwkv_half = VRWKVStyleSpatialMix(channels, num_frames=5, layer_id=0.33, n_layer=n_layer)
        self.rwkv_quarter = VRWKVStyleSpatialMix(channels, num_frames=5, layer_id=0.67, n_layer=n_layer)

        # Bravo: 边缘门控在 Encoder 原始中心帧上作边缘检测 (STCD 保留原始内容)
        self.edge_prompt = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.Sigmoid(),
        )

        self.norm_out = LayerNorm2d(channels)

    @staticmethod
    def _bidirectional_scan(rwkv_module, x_flat, feat_shape):
        fwd = rwkv_module(x_flat, feat_shape)
        rev = rwkv_module(torch.flip(x_flat, dims=[1]), feat_shape)
        bwd = torch.flip(rev, dims=[1])
        return (fwd + bwd) / 2

    def forward(self, feats: torch.Tensor, tfsi_out: Dict = None) -> Dict:
        B, T, C, H, W = feats.shape
        center_idx = T // 2

        # Bravo P1: 双 DWT-LFF — center (α=0.6 干净锚点) / neighbor (α=0.4 退化)
        lff_feats: List[torch.Tensor] = []
        for t in range(T):
            if t == center_idx:
                out = self.lff_center(feats[:, t])
            else:
                out = self.lff_neighbor(feats[:, t])
            lff_feats.append(out["feat_sace"])
        lff_stack = torch.stack(lff_feats, dim=1)

        mu_t_clean = lff_stack[:, center_idx]
        sigma_t_clean = lff_stack.std(dim=1, unbiased=False)
        s_noise = tfsi_out.get("s_noise") if tfsi_out else None

        # 三尺度双向 RWKV
        x_flat = lff_stack.permute(0, 1, 3, 4, 2).reshape(B, T, H * W, C)

        out_full = self._bidirectional_scan(self.rwkv_full, x_flat, (H, W))

        h2, w2 = H // 2, W // 2
        x_half_3d = lff_stack.reshape(B, T, C, H, W)
        x_half = F.avg_pool3d(x_half_3d, (1, 2, 2)).permute(0, 1, 3, 4, 2).reshape(B, T, h2 * w2, C)
        out_half_flat = self._bidirectional_scan(self.rwkv_half, x_half, (h2, w2))
        out_half = out_half_flat.reshape(B, T, h2, w2, C).permute(0, 1, 4, 2, 3)
        out_half = F.interpolate(
            out_half.reshape(B * T, C, h2, w2), size=(H, W), mode='bilinear', align_corners=False
        ).reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2).reshape(B, T, H * W, C)

        h4, w4 = H // 4, W // 4
        x_quarter = F.avg_pool3d(x_half_3d, (1, 4, 4)).permute(0, 1, 3, 4, 2).reshape(B, T, h4 * w4, C)
        out_qtr_flat = self._bidirectional_scan(self.rwkv_quarter, x_quarter, (h4, w4))
        out_qtr = out_qtr_flat.reshape(B, T, h4, w4, C).permute(0, 1, 4, 2, 3)
        out_qtr = F.interpolate(
            out_qtr.reshape(B * T, C, h4, w4), size=(H, W), mode='bilinear', align_corners=False
        ).reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2).reshape(B, T, H * W, C)

        out_flat = (out_full + out_half + out_qtr) / 3
        out = out_flat.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)

        # Bravo P1: V 投影使用 Encoder 原始中心帧 (STCD IJCAI 2025)
        f_raw_center = feats[:, center_idx]
        edge_weight = self.edge_prompt(f_raw_center)

        F_aligned_list: List[torch.Tensor] = []
        for t in range(T):
            f_t = out[:, t] + (1.0 - edge_weight) * f_raw_center
            if s_noise is not None:
                f_t = f_t + (1.0 - s_noise) * f_raw_center
            else:
                f_t = f_t + f_raw_center
            f_t = self.norm_out(f_t)
            F_aligned_list.append(f_t)

        return {
            "mu_t_clean":      mu_t_clean,
            "sigma_t_clean":   sigma_t_clean,
            "F_aligned_list":  F_aligned_list,
            "lff_feats":       lff_feats,
        }
