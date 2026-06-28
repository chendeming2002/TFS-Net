"""
IFPN — Illumination-Filtering Pyramid Network
==============================================
基于 SACE 对齐特征的光照估计与提亮图生成.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.blocks import ConvBlock, pairwise_cosine_logits


class IllumExtract(nn.Module):
    def __init__(self, img_channels=3, feat_channels=96, feat_proj_channels=16,
                 n_fea_middle=32, n_fea_in=4):
        super().__init__()
        assert n_fea_middle % n_fea_in == 0
        n_fea_total = img_channels + 1 + feat_proj_channels
        self.feat_proj = nn.Conv2d(feat_channels, feat_proj_channels, kernel_size=1, bias=True)
        self.conv1 = nn.Conv2d(n_fea_total, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle, img_channels, kernel_size=1, bias=True)

    def forward(self, img_down, feat_L):
        img_mean = img_down.mean(dim=1, keepdim=True)
        feat_proj = self.feat_proj(feat_L)
        x = torch.cat([img_down, img_mean, feat_proj], dim=1)
        x = self.conv1(x)
        x = self.depth_conv(x)
        L = self.conv2(x)
        return L


class IFPN(nn.Module):
    def __init__(self, fused_channels=64, aligned_channels=None, coarse_channels=128,
                 img_channels=3, feat_proj_channels=16, n_fea_middle=32, n_fea_in=4,
                 sim_temperature=1.0, max_bright=4.0):
        super().__init__()
        self.fused_channels = fused_channels
        self.aligned_channels = aligned_channels or fused_channels
        self.coarse_channels = coarse_channels
        self.img_channels = img_channels
        self.sim_temperature = sim_temperature
        self.max_bright = max_bright

        self.coarse_adapter = nn.Sequential(
            nn.Conv2d(self.aligned_channels, coarse_channels, 1, 1, 0),
            nn.GELU(),
        )

        self.illum_extract = IllumExtract(
            img_channels=img_channels, feat_channels=coarse_channels,
            feat_proj_channels=feat_proj_channels, n_fea_middle=n_fea_middle, n_fea_in=n_fea_in,
        )

        self.img_estimator = nn.Sequential(
            nn.Conv2d(coarse_channels, 32, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, img_channels, kernel_size=1, bias=True),
        )

        self.ratio_proj = nn.Conv2d(img_channels, fused_channels, kernel_size=1, bias=True)

        self.feat_refine = nn.Sequential(
            ConvBlock(fused_channels, fused_channels, kernel_size=3, stride=1, padding=1, act=True),
            ConvBlock(fused_channels, fused_channels, kernel_size=1, stride=1, padding=0, act=False),
        )

        self.lit_up_proj = nn.Conv2d(fused_channels, img_channels, kernel_size=1, bias=True)

        self.side_head = nn.Sequential(
            nn.Conv2d(fused_channels, 32, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, img_channels, 3, 1, 1, bias=True),
        )

    def forward(self, I_t_down, aligned_feats, center_idx, imgs_down=None):
        B, T, C_a, H, W = aligned_feats.shape
        BT = B * T
        aligned_flat = aligned_feats.reshape(BT, *aligned_feats.shape[2:])
        projected = self.coarse_adapter(aligned_flat)
        h, w = I_t_down.shape[-2:]
        coarse_flat = F.adaptive_avg_pool2d(projected, (h, w))
        coarse_feats = coarse_flat.reshape(B, T, self.coarse_channels, h, w)

        F_t_L = coarse_feats[:, center_idx]
        L_t = self.illum_extract(I_t_down, F_t_L)

        L_list = []
        for i in range(T):
            F_i_L = coarse_feats[:, i]
            if imgs_down is not None:
                I_i_down = imgs_down[:, i]
            else:
                I_i_down = self.img_estimator(F_i_L)
            L_i = self.illum_extract(I_i_down, F_i_L)
            L_list.append(L_i)

        f_t_coarse = coarse_feats[:, center_idx]
        neighbor_indices = [i for i in range(T) if i != center_idx]
        neighbors = coarse_feats[:, neighbor_indices]
        sim_logits = pairwise_cosine_logits(f_t_coarse, neighbors)
        weights = F.softmax(sim_logits / self.sim_temperature, dim=-1)
        weights = weights.reshape(B, T - 1, 1, 1, 1)

        L_neighbors = torch.stack([L_list[i] for i in neighbor_indices], dim=1)
        L_ref = (weights * L_neighbors).sum(dim=1)

        eps = 1e-3
        L_ratio_lr = (L_ref / (L_t.abs() + eps)).clamp(0.5, 8.0)
        L_ratio = F.interpolate(L_ratio_lr, size=(H, W), mode='bilinear', align_corners=False)
        ratio_feat = self.ratio_proj(L_ratio)

        f_illum_feat = self.feat_refine(ratio_feat)

        lit_up_delta = self.lit_up_proj(f_illum_feat)
        lit_up_feat = L_ratio + lit_up_delta
        lit_up_map_raw = 1.0 + self.max_bright * torch.sigmoid(lit_up_feat)

        ifpn_side = self.side_head(f_illum_feat)

        return {
            "lit_up_map_raw": lit_up_map_raw,
            "f_illum_feat":  f_illum_feat,
            "L_t":           L_t,
            "L_ref":         L_ref,
            "L_ratio":       L_ratio,
            "ifpn_side":     ifpn_side,
        }
