"""
IFPN (Illumination-Filtering Pyramid Network) - TFS-Net v5.5
============================================================
v4.3: Hybrid lit_up_map (L_ratio anchor + feat_proj delta), sigmoid-bounded.
v5.5: s_illum removed from IFPN — directly injected into IGRF BrightenStage.
      IFPN now purely data-driven (aligned_feats + image), no intensity prior.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.blocks import ConvBlock, pairwise_cosine_logits


class IllumExtract(nn.Module):
    """
    Retinexformer 风格的光照图提取器。

    Args:
        img_channels       : 输入图像通道 (默认 3)
        feat_channels      : 输入特征通道 (默认 96)
        feat_proj_channels : 特征压缩后通道 (默认 16)
        n_fea_middle       : 中间通道数 (默认 32, 能被 groups=4 整除)
        n_fea_in           : groups 数 (默认 4)
    """

    def __init__(
        self,
        img_channels: int = 3,
        feat_channels: int = 96,
        feat_proj_channels: int = 16,
        n_fea_middle: int = 32,
        n_fea_in: int = 4,
    ):
        super().__init__()
        assert n_fea_middle % n_fea_in == 0

        n_fea_total = img_channels + 1 + feat_proj_channels

        self.feat_proj = nn.Conv2d(feat_channels, feat_proj_channels, kernel_size=1, bias=True)
        self.conv1 = nn.Conv2d(n_fea_total, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle,
            kernel_size=5, padding=2, bias=True,
            groups=n_fea_in,
        )
        self.conv2 = nn.Conv2d(n_fea_middle, img_channels, kernel_size=1, bias=True)

    def forward(self, img_down: torch.Tensor, feat_L: torch.Tensor) -> torch.Tensor:
        img_mean = img_down.mean(dim=1, keepdim=True)
        feat_proj = self.feat_proj(feat_L)
        x = torch.cat([img_down, img_mean, feat_proj], dim=1)
        x = self.conv1(x)
        x = self.depth_conv(x)
        L = self.conv2(x)
        return L


class IFPN(nn.Module):
    """
    Illumination-Filtering Pyramid Network.

    v5.3: 用 SACE 对齐特征替代 Encoder 粗特征，通过 coarse_adapter 适配维度。

    Args:
        fused_channels   : 中心帧特征通道数 (默认 64)
        aligned_channels : SACE 对齐特征通道数 (默认 = fused_channels)
        coarse_channels  : 内部粗特征通道 (默认 128, IllumExtract 输入)
        img_channels     : 图像通道 (默认 3)
    """

    def __init__(
        self,
        fused_channels: int = 64,
        aligned_channels: int = None,
        coarse_channels: int = 128,
        img_channels: int = 3,
        feat_proj_channels: int = 16,
        n_fea_middle: int = 32,
        n_fea_in: int = 4,
        sim_temperature: float = 1.0,
        max_bright: float = 4.0,
    ):
        super().__init__()
        self.fused_channels = fused_channels
        self.aligned_channels = aligned_channels or fused_channels
        self.coarse_channels = coarse_channels
        self.img_channels = img_channels
        self.sim_temperature = sim_temperature
        self.max_bright = max_bright

        # v5.3: SACE 对齐特征 → 粗特征适配器 (1×1 Conv + 空间下采样)
        self.coarse_adapter = nn.Sequential(
            nn.Conv2d(self.aligned_channels, coarse_channels, 1, 1, 0),
            nn.GELU(),
        )

        # Charlie3 P0: s_illum 先验注入 (零初始化，渐进学习)
        # s_illum 作为光照先验投影到粗特征空间，与 coarse_adapter 输出相加
        self.s_illum_proj = nn.Sequential(
            nn.Conv2d(1, self.aligned_channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.aligned_channels, coarse_channels, 3, 1, 1),
        )
        nn.init.zeros_(self.s_illum_proj[0].weight)
        nn.init.zeros_(self.s_illum_proj[2].weight)
        nn.init.zeros_(self.s_illum_proj[2].bias)

        # Delta: A_illu 生成 (从 IGRF 移入)
        self.illu_conv = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, 3, 1, 1, groups=fused_channels),
            nn.Conv2d(fused_channels, 1, 1),
            nn.Sigmoid(),
        )

        # Delta: F_t_aligned 锚定光照 (参考 FRESCO spatial correspondence)
        self.illu_anchor = nn.Sequential(
            nn.Conv2d(fused_channels * 2, fused_channels, 1),
            nn.GELU(),
            nn.Conv2d(fused_channels, fused_channels, 1),
        )
        self.illu_anchor_gate = nn.Parameter(torch.zeros(1, fused_channels, 1, 1))

        self.illum_extract = IllumExtract(
            img_channels=img_channels,
            feat_channels=coarse_channels,
            feat_proj_channels=feat_proj_channels,
            n_fea_middle=n_fea_middle,
            n_fea_in=n_fea_in,
        )

        self.img_estimator = nn.Sequential(
            nn.Conv2d(coarse_channels, 32, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, img_channels, kernel_size=1, bias=True),
        )

        self.ratio_proj = nn.Conv2d(img_channels, fused_channels, kernel_size=1, bias=True)

        # v5.5: s_illum removed from IFPN (directly injected into IGRF instead)
        # illum_cond_proj (65->64) no longer needed

        self.feat_refine = nn.Sequential(
            ConvBlock(fused_channels, fused_channels, kernel_size=3, stride=1, padding=1, act=True),
            ConvBlock(fused_channels, fused_channels, kernel_size=1, stride=1, padding=0, act=False),
        )

        # v4.3: hybrid estimation - feature space delta projected to image space
        self.lit_up_proj = nn.Conv2d(fused_channels, img_channels, kernel_size=1, bias=True)

        # v5.9.2: side_head 用于中间感知监督 (DarkIR EnhanceLoss 风格)
        # 把 f_illum_feat 投影为低分辨率图像, 监督 GT 下采样, 强制光照特征有意义
        self.side_head = nn.Sequential(
            nn.Conv2d(fused_channels, 32, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, img_channels, 3, 1, 1, bias=True),
        )

    def forward(
        self,
        I_t_down: torch.Tensor,
        aligned_feats: torch.Tensor,
        center_idx: int,
        imgs_down: torch.Tensor = None,
        s_illum: torch.Tensor = None,
        F_t_aligned: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            I_t_down      : (B, 3, h, w) 下采样后的中心帧图像
            aligned_feats : (B, T, C_a, H, W) SACE 对齐后的多帧特征
            center_idx    : 中心帧索引
            imgs_down     : (B, T, 3, h, w) 下采样后的多帧图像（可选）
            s_illum       : (B, 1, H, W) TFSI 光照退化强度
            F_t_aligned   : (B, C_coarse, H, W) Delta SACE 对齐增强特征
        """
        B, T, C_a, H, W = aligned_feats.shape

        # Charlie3 P0: s_illum 先验注入 (替换原 aligned_feats 调制)
        # 注入位置: coarse_adapter 之后，作为光照先验加到粗特征上
        if s_illum is not None:
            # s_illum_proj: (B,C_a,H,W) → (B,coarse_c,H,W)
            illum_prior = self.s_illum_proj(s_illum)
            # broadcast 到每一帧: (B,1,C_c,h,w) → (B,T,C_c,h,w)
            BT = B * T
            aligned_flat = aligned_feats.reshape(BT, *aligned_feats.shape[2:])
            projected = self.coarse_adapter(aligned_flat)
            h, w = I_t_down.shape[-2:]
            coarse_flat = F.adaptive_avg_pool2d(projected, (h, w))
            # 注入 s_illum 先验 (每帧共享)
            illum_prior_2d = F.interpolate(illum_prior, size=(h, w), mode='bilinear',
                                           align_corners=False)
            coarse_flat = coarse_flat + illum_prior_2d.repeat(T, 1, 1, 1)
            coarse_feats = coarse_flat.reshape(B, T, self.coarse_channels, h, w)
        else:
            BT = B * T
            aligned_flat = aligned_feats.reshape(BT, *aligned_feats.shape[2:])
            projected = self.coarse_adapter(aligned_flat)
            h, w = I_t_down.shape[-2:]
            coarse_flat = F.adaptive_avg_pool2d(projected, (h, w))
            coarse_feats = coarse_flat.reshape(B, T, self.coarse_channels, h, w)

        # F_t_L: 中心帧粗特征（内部生成）
        F_t_L = coarse_feats[:, center_idx]

        # Step 1: 中心帧光照
        L_t = self.illum_extract(I_t_down, F_t_L)

        # Step 2: 逐帧光照
        L_list = []
        for i in range(T):
            F_i_L = coarse_feats[:, i]
            if imgs_down is not None:
                I_i_down = imgs_down[:, i]
            else:
                I_i_down = self.img_estimator(F_i_L)
            L_i = self.illum_extract(I_i_down, F_i_L)
            L_list.append(L_i)

        # Step 3: 帧间相似度
        f_t_coarse = coarse_feats[:, center_idx]
        neighbor_indices = [i for i in range(T) if i != center_idx]
        neighbors = coarse_feats[:, neighbor_indices]
        sim_logits = pairwise_cosine_logits(f_t_coarse, neighbors)
        weights = F.softmax(sim_logits / self.sim_temperature, dim=-1)
        weights = weights.reshape(B, T - 1, 1, 1, 1)

        # Step 4: 邻帧光照加权
        L_neighbors = torch.stack([L_list[i] for i in neighbor_indices], dim=1)
        L_ref = (weights * L_neighbors).sum(dim=1)

        # Step 5: Illumination ratio estimation
        eps = 1e-3
        L_ratio_lr = (L_ref / (L_t.abs() + eps)).clamp(0.5, 8.0)
        L_ratio = F.interpolate(L_ratio_lr, size=(H, W), mode='bilinear', align_corners=False)
        ratio_feat = self.ratio_proj(L_ratio)

        # v5.5: s_illum no longer concatenated here — directly injected into IGRF
        f_illum_feat = self.feat_refine(ratio_feat)

        # Delta: F_t_aligned 锚定光照特征 (防止帧间闪烁)
        if F_t_aligned is not None:
            F_t_resized = F.interpolate(F_t_aligned, size=f_illum_feat.shape[-2:],
                                        mode='bilinear', align_corners=False)
            anchor_feat = self.illu_anchor(torch.cat([f_illum_feat, F_t_resized], dim=1))
            f_illum_feat = f_illum_feat + anchor_feat * self.illu_anchor_gate.tanh()

        # Delta: A_illu 生成 (从 IGRF 移入)
        A_illu = self.illu_conv(f_illum_feat)  # (B, C_coarse, h, w)

        # v4.3: hybrid estimation - L_ratio anchor + feature-space delta
        lit_up_delta = self.lit_up_proj(f_illum_feat)              # 64ch -> 3ch
        lit_up_feat = L_ratio + lit_up_delta                        # physical anchor + feature correction
        lit_up_map_raw = 1.0 + self.max_bright * torch.sigmoid(lit_up_feat)  # bounded [1, 1+max_bright]

        # v5.9.2: 中间监督输出 — 把 f_illum_feat 投影为图像, 供增强损失用
        ifpn_side = self.side_head(f_illum_feat)

        return {
            "lit_up_map_raw": lit_up_map_raw,
            "f_illum_feat":   f_illum_feat,
            "A_illu":         A_illu,         # Delta: 从 IGRF 移入
            "L_t":            L_t,
            "L_ref":          L_ref,
            "L_ratio":        L_ratio,
            "ifpn_side":      ifpn_side,
        }
