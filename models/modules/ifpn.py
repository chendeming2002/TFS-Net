"""
IFPN (Illumination-Filtering Pyramid Network) - TFS-Net v4.3
============================================================
v4.3: Hybrid lit_up_map (L_ratio anchor + feat_proj delta), sigmoid-bounded,
      s_illum conditioning for illumination prior integration.
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

    Args:
        fused_channels   : 中心帧特征通道数 (默认 48)
        coarse_channels  : 粗尺度特征通道 (默认 96)
        img_channels     : 图像通道 (默认 3)
    """

    def __init__(
        self,
        fused_channels: int = 64,
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
        self.coarse_channels = coarse_channels
        self.img_channels = img_channels
        self.sim_temperature = sim_temperature
        self.max_bright = max_bright

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

        # v4.3: s_illum (1ch) is expanded and concatenated with ratio_feat
        self.illum_cond_proj = nn.Conv2d(fused_channels + 1, fused_channels, kernel_size=1, bias=True)

        self.feat_refine = nn.Sequential(
            ConvBlock(fused_channels, fused_channels, kernel_size=3, stride=1, padding=1, act=True),
            ConvBlock(fused_channels, fused_channels, kernel_size=1, stride=1, padding=0, act=False),
        )

        # v4.3: hybrid estimation - feature space delta projected to image space
        self.lit_up_proj = nn.Conv2d(fused_channels, img_channels, kernel_size=1, bias=True)

    def forward(
        self,
        I_t_down: torch.Tensor,
        F_t_L: torch.Tensor,
        s_illum: torch.Tensor,
        feats: torch.Tensor,
        coarse_feats: torch.Tensor,
        center_idx: int,
        imgs_down: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, C_f, H, W = feats.shape
        _, _, _, h, w = coarse_feats.shape

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
        weights = weights.view(B, T - 1, 1, 1, 1)

        # Step 4: 邻帧光照加权
        L_neighbors = torch.stack([L_list[i] for i in neighbor_indices], dim=1)
        L_ref = (weights * L_neighbors).sum(dim=1)

        # Step 5: Illumination ratio estimation
        eps = 1e-3
        L_ratio_lr = (L_ref / (L_t.abs() + eps)).clamp(0.5, 8.0)
        L_ratio = F.interpolate(L_ratio_lr, size=(H, W), mode='bilinear', align_corners=False)
        ratio_feat = self.ratio_proj(L_ratio)

        # v4.3: integrate s_illum prior into illumination features
        if s_illum is not None:
            s_illum_up = F.interpolate(s_illum, size=(H, W), mode='bilinear', align_corners=False)
            illum_cond = torch.cat([ratio_feat, s_illum_up], dim=1)  # (B, C+1, H, W)
            ratio_feat = self.illum_cond_proj(illum_cond)              # (B, C, H, W)

        f_illum_feat = self.feat_refine(ratio_feat)

        # v4.3: hybrid estimation - L_ratio anchor + feature-space delta
        lit_up_delta = self.lit_up_proj(f_illum_feat)              # 64ch -> 3ch
        lit_up_feat = L_ratio + lit_up_delta                        # physical anchor + feature correction
        lit_up_map_raw = 1.0 + self.max_bright * torch.sigmoid(lit_up_feat)  # bounded [1, 1+max_bright]

        return {
            "lit_up_map_raw": lit_up_map_raw,
            "f_illum_feat":  f_illum_feat,
            "L_t":           L_t,
            "L_ref":         L_ref,
            "L_ratio":       L_ratio,
        }
