"""
PureRWKVSACE — v6.5 纯 RWKV 帧间对齐 (2026-06-27)
===================================================
基于 pureRWKV.md 三大范式:
  范式 1: 双向扫描对齐 (ABMamba AHBS + Otter TRM)
  范式 2: 帧间门控交互 (Video RWKV LCR edge prompt)
  范式 3: 多尺度时序建模 (ABMamba M=3 stride=2)

设计: 移除 DeformableCrossAttention (130K 参数), 用多尺度双向 RWKV + 边缘门控替代

API 兼容原 SACE: 输入 feats/tfsi_out, 输出 F_aligned_list
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

from models.modules.lff import LFFFeatureAdapter
from .dwt_lff import SpatialDWTLFFAdapter
from .cross_rwkv import VRWKVStyleSpatialMix
from .blocks import LayerNorm2d


class PureRWKVSACE(nn.Module):
    """纯 RWKV 多尺度帧间注意力 (v6.5)

    移除 DeformableCrossAttention, 用 3 尺度双向 RWKV 替换。

    Args:
        channels: 特征通道数 (默认 64)
        lff_module: 外部传入的 DWT-LFF 或传统 LFF (None 则内部新建)
        n_layer: RWKV 层数 (用于 fancy init)
    """

    def __init__(
        self,
        channels: int = 64,
        lff_module=None,
        n_layer: int = 1,
    ):
        super().__init__()
        self.channels = channels

        # DWT-LFF (共享, 传入)
        if lff_module is not None:
            self.lff = lff_module
        else:
            self.lff = LFFFeatureAdapter(channels=channels, K=10, n_ang_freq=1,
                                         per_channel_rbf=False, phase_preserving=True)

        # 1️⃣ 多尺度 RWKV (ABMamba AHBS: M=3, stride=2)
        self.rwkv_full = VRWKVStyleSpatialMix(channels, num_frames=5,
                                                layer_id=0.0, n_layer=n_layer)
        self.rwkv_half = VRWKVStyleSpatialMix(channels, num_frames=5,
                                                layer_id=0.33, n_layer=n_layer)
        self.rwkv_quarter = VRWKVStyleSpatialMix(channels, num_frames=5,
                                                   layer_id=0.67, n_layer=n_layer)

        # 2️⃣ 边缘门控 (Video RWKV LCR edge prompt)
        self.edge_prompt = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.Sigmoid(),
        )

        # LayerNorm
        self.norm_out = LayerNorm2d(channels)

    @staticmethod
    def _bidirectional_scan(rwkv_module, x_flat, feat_shape):
        """Otter TRM 风格双向扫描: 前向 + 反向 → 平均"""
        fwd = rwkv_module(x_flat, feat_shape)
        rev = rwkv_module(torch.flip(x_flat, dims=[1]), feat_shape)
        bwd = torch.flip(rev, dims=[1])
        return (fwd + bwd) / 2

    def forward(
        self,
        feats: torch.Tensor,       # (B, T, C, H, W) 编码器特征
        tfsi_out: Dict = None,    # TFSI 输出 (取 s_noise)
        cached_lff: Dict = None,   # 推理缓存
    ) -> Dict:
        B, T, C, H, W = feats.shape
        device = feats.device

        # === Step 1: 逐帧 DWT-LFF ===
        lff_feats: List[torch.Tensor] = []
        for t in range(T):
            if cached_lff and t in cached_lff:
                lff_feats.append(cached_lff[t])
            else:
                lff_out = self.lff(feats[:, t])
                if isinstance(lff_out, dict):
                    lff_feats.append(lff_out["feat_sace"])
                else:
                    lff_feats.append(lff_out)
        lff_stack = torch.stack(lff_feats, dim=1)  # (B, T, C, H, W)

        # 中心帧参考
        mu_t_clean = lff_stack[:, T // 2]
        sigma_t_clean = lff_stack.std(dim=1, unbiased=False)
        s_noise = tfsi_out.get("s_noise") if tfsi_out else None

        # === Step 2: 展平为 token ===
        x_flat = lff_stack.permute(0, 1, 3, 4, 2).reshape(B, T, H * W, C)

        # === Step 3: 多尺度双向 RWKV (ABMamba Eq.1) ===
        out_full = self._bidirectional_scan(self.rwkv_full, x_flat, (H, W))

        # 1/2 尺度
        h2, w2 = H // 2, W // 2
        x_half_3d = lff_stack.reshape(B, T, C, H, W)
        x_half = F.avg_pool3d(x_half_3d, (1, 2, 2)).permute(0, 1, 3, 4, 2).reshape(B, T, h2 * w2, C)
        out_half_flat = self._bidirectional_scan(self.rwkv_half, x_half, (h2, w2))
        out_half = out_half_flat.reshape(B, T, h2, w2, C).permute(0, 1, 4, 2, 3)
        out_half = F.interpolate(
            out_half.reshape(B * T, C, h2, w2), size=(H, W), mode='bilinear', align_corners=False
        ).reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2).reshape(B, T, H * W, C)

        # 1/4 尺度
        h4, w4 = H // 4, W // 4
        x_quarter = F.avg_pool3d(x_half_3d, (1, 4, 4)).permute(0, 1, 3, 4, 2).reshape(B, T, h4 * w4, C)
        out_qtr_flat = self._bidirectional_scan(self.rwkv_quarter, x_quarter, (h4, w4))
        out_qtr = out_qtr_flat.reshape(B, T, h4, w4, C).permute(0, 1, 4, 2, 3)
        out_qtr = F.interpolate(
            out_qtr.reshape(B * T, C, h4, w4), size=(H, W), mode='bilinear', align_corners=False
        ).reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2).reshape(B, T, H * W, C)

        # === Step 4: 跨尺度聚合 ===
        out_flat = (out_full + out_half + out_qtr) / 3
        out = out_flat.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)

        # === Step 5: 边缘门控残差 (Video RWKV LCR) ===
        edge_weight = self.edge_prompt(lff_stack[:, T // 2])  # (B, C, H, W)

        F_aligned_list: List[torch.Tensor] = []
        for t in range(T):
            # 边缘门控: 高边缘区信任 RWKV 输出, 低边缘区回退到原始特征
            f_t = out[:, t] + (1.0 - edge_weight) * lff_stack[:, t]
            # 噪声感知残差
            if s_noise is not None:
                f_t = f_t + (1.0 - s_noise) * lff_stack[:, t]
            else:
                f_t = f_t + lff_stack[:, t]
            f_t = self.norm_out(f_t)
            F_aligned_list.append(f_t)

        return {
            "mu_t_clean":      mu_t_clean,
            "sigma_t_clean":   sigma_t_clean,
            "F_aligned_list":  F_aligned_list,
            "lff_feats":       lff_feats,
            "attn_maps":       [],  # v6.5: 无 DAT, 兼容旧接口
        }
