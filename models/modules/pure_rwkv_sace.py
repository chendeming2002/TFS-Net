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

        # v6 Bravo P1-2: DWT-LFF 拆为中心/邻居双实例 (VSRELL + STCD)
        # 中心帧 α=0.6 (偏 LL_ref → 干净锚定)
        # 邻居帧 α=0.4 (偏 LL_deg → 退化诊断)
        self.lff_center = SpatialDWTLFFAdapter(in_channels=channels, alpha_init=0.6)
        self.lff_neighbor = SpatialDWTLFFAdapter(in_channels=channels, alpha_init=0.4)

        # 兼容旧接口: lff_module 传入时仍可用
        self._lff_external = lff_module is not None
        if lff_module is not None:
            self.lff_center = lff_module
            self.lff_neighbor = lff_module

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

        # 3️⃣ 多尺度融合 (Charlie P1-1: concat + channel_mix 替代等权平均)
        self.channel_mix = nn.Sequential(
            nn.Linear(channels * 3, channels),
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

        # === Step 1: 逐帧 DWT-LFF (中心/邻居分支分离) ===
        lff_feats: List[torch.Tensor] = []
        center_idx = T // 2
        for t in range(T):
            if cached_lff and t in cached_lff:
                lff_feats.append(cached_lff[t])
            else:
                # v6 Bravo P1-2: 中心帧走 lff_center, 邻居帧走 lff_neighbor
                if t == center_idx:
                    lff_out = self.lff_center(feats[:, t])
                else:
                    lff_out = self.lff_neighbor(feats[:, t])
                if isinstance(lff_out, dict):
                    lff_feats.append(lff_out["feat_sace"])
                else:
                    lff_feats.append(lff_out)
        lff_stack = torch.stack(lff_feats, dim=1)  # (B, T, C, H, W)

        # 中心帧参考
        mu_t_clean = lff_stack[:, center_idx]
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

        # === Step 4: 多尺度融合 (Charlie P1-1: concat+channel_mix 替代 /3) ===
        out_cat = torch.cat([out_full, out_half, out_qtr], dim=-1)  # (B, T, L, 3C)
        out_flat = self.channel_mix(out_cat)  # (B, T, L, C)
        out = out_flat.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)

        # === Step 5: 边缘门控残差 (Video RWKV LCR) ===
        # v6 Bravo P1: V 使用 Encoder 原始中心帧 (而非 DWT-LFF 归一化后)
        # STCD: 对齐用归一化空间, 内容融合用原始空间
        f_raw_center = feats[:, T // 2]  # (B, C, H, W) Encoder 原始中心帧
        edge_weight = self.edge_prompt(f_raw_center)  # ★ 在原始特征上做边缘检测

        F_aligned_list: List[torch.Tensor] = []
        for t in range(T):
            # RWKV 对齐输出 + 原始内容残差 (V source = encoder raw)
            f_t = out[:, t] + (1.0 - edge_weight) * f_raw_center
            # 噪声感知残差: 也在原始空间
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
            "attn_maps":       [],  # v6.5: 无 DAT, 兼容旧接口
        }
