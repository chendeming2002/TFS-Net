"""
TFS-Net v6 Bravo — 纯 RWKV 多帧低光增强网络
=============================================
RWKV-Bravo 相对 v6 的架构精简 (P4):
  - 移除 AmpEnhance (91K): 不再需前置幅度增强
  - 移除 DeformableCrossAttention (130K): 由 PureRWKVSACE 替代
  - 移除 LFF (FFT 频域 RBF): 由 SpatialDWTLFFAdapter (空域 DWT) 替代
  - TFSNet 直接实例化 PureRWKVSACE (无 Fallback SACE 分支)

数据流 (5 stages):
  0: PyramidEncoder          多帧 → 多尺度特征 + 全分辨率融合
  1: TFSI + DWT-LFF (center) 时序/频域 → s_illum, s_noise, F_fused
  2: PureRWKVSACE (dual DWT) 多尺度双向 RWKV + 边缘门控对齐
  3: IFPN / NDPN / MRPN      光照/噪声/运动三源恢复
  4: IGRF                    强度引导残差融合 → 输出
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    ConvBlock, ResBlock, NAFBlock, LayerNorm2d,
    pad_to_window, unpad_from_window,
    window_partition_2d, window_partition_video, window_reverse_2d,
    pairwise_cosine_logits, safe_divide,
)
from .cross_rwkv import VRWKVStyleSpatialMix
from .dwt_lff import SpatialDWTLFFAdapter, HaarDWT2D


# =====================================================================
# Stage 0: PyramidEncoder
# =====================================================================

class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, act=True),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True),
        )

    def forward(self, x):
        return self.block(x)


class PyramidEncoder(nn.Module):
    def __init__(self, in_channels=3, level_channels=(32, 64, 96), fused_channels=64,
                 num_bottleneck_blocks: int = 0):
        super().__init__()
        if len(level_channels) == 4:
            c1, c2, c3, c4 = level_channels
        else:
            c1, c2, c3 = level_channels
            c4 = None

        self.stage1 = EncoderStage(in_channels, c1, stride=1)
        self.stage2 = EncoderStage(c1, c2, stride=2)
        self.stage3 = EncoderStage(c2, c3, stride=2)
        self.has_stage4 = c4 is not None
        if self.has_stage4:
            self.stage4 = EncoderStage(c3, c4, stride=2)
            self.lateral4 = nn.Conv2d(c4, fused_channels, 1, 1, 0)

        self.num_bottleneck_blocks = num_bottleneck_blocks
        if num_bottleneck_blocks > 0:
            bottleneck_ch = c4 if c4 is not None else c3
            self.bottleneck = nn.Sequential(
                *[ResBlock(bottleneck_ch) for _ in range(num_bottleneck_blocks)]
            )
        else:
            self.bottleneck = None

        self.lateral3 = nn.Conv2d(c3, fused_channels, 1, 1, 0)
        self.lateral2 = nn.Conv2d(c2, fused_channels, 1, 1, 0)
        self.lateral1 = nn.Conv2d(c1, fused_channels, 1, 1, 0)
        self.fuse_norm = LayerNorm2d(fused_channels)
        self.fuse = nn.Sequential(
            ConvBlock(fused_channels, fused_channels, 3, 1, 1, act=True),
            ConvBlock(fused_channels, fused_channels, 3, 1, 1, act=True),
        )

    def forward_single(self, x, return_coarse=False):
        l1 = self.stage1(x)
        l2 = self.stage2(l1)
        l3 = self.stage3(l2)

        if self.has_stage4:
            l4 = self.stage4(l3)
            if self.bottleneck is not None:
                l4 = self.bottleneck(l4)
            p4 = self.lateral4(l4)
            p3 = self.lateral3(l3) + F.interpolate(p4, size=l3.shape[-2:], mode="bilinear", align_corners=False)
        else:
            l4 = None
            if self.bottleneck is not None:
                l3 = self.bottleneck(l3)
            p3 = self.lateral3(l3)

        p2 = self.lateral2(l2) + F.interpolate(p3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral1(l1) + F.interpolate(p2, size=l1.shape[-2:], mode="bilinear", align_corners=False)
        p1_normed = self.fuse_norm(p1)
        fused = self.fuse(p1_normed)

        if return_coarse:
            coarse = l4 if self.has_stage4 else l3
            return fused, coarse
        return fused

    def forward(self, x, return_coarse=False):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        if return_coarse:
            fused, coarse = self.forward_single(x, return_coarse=True)
            _, cf, hf, wf = fused.shape
            _, cc, hc, wc = coarse.shape
            return fused.view(b, t, cf, hf, wf), coarse.view(b, t, cc, hc, wc)
        else:
            feat = self.forward_single(x, return_coarse=False)
            _, cf, hf, wf = feat.shape
            return feat.view(b, t, cf, hf, wf)


# =====================================================================
# Stage 1: TFSI + 子模块
# =====================================================================

class SpatialBranch(nn.Module):
    def __init__(self, channels: int, fused_channels: int, eps: float = 1e-6,
                 soft_median_tau: float = 0.1, use_soft_median: bool = True):
        super().__init__()
        self.eps = eps
        self.use_soft_median = use_soft_median
        self.soft_median_tau = soft_median_tau
        self.conv = nn.Sequential(
            ConvBlock(channels * 3, fused_channels, kernel_size=3, stride=1, padding=1, act=True),
            ConvBlock(fused_channels, fused_channels, kernel_size=3, stride=1, padding=1, act=True),
        )

    @staticmethod
    def _soft_median(x: torch.Tensor, dim: int = 1, tau: float = 0.1) -> torch.Tensor:
        with torch.no_grad():
            med = x.median(dim=dim).values.unsqueeze(dim)
        dist = (x - med).abs()
        weights = F.softmax(-dist / tau, dim=dim)
        return (weights * x).sum(dim=dim)

    def forward(self, feats: torch.Tensor) -> dict:
        if self.use_soft_median:
            mu_t = self._soft_median(feats, dim=1, tau=self.soft_median_tau)
        else:
            mu_t = feats.median(dim=1).values

        sigma_t_sq = feats.var(dim=1, unbiased=False)
        sigma_t = torch.sqrt(sigma_t_sq + self.eps)
        snr = mu_t / (sigma_t + self.eps)

        stats = torch.cat([mu_t, sigma_t, snr], dim=1)
        f_s = self.conv(stats)

        return {"F_s": f_s, "mu_t": mu_t, "sigma_t": sigma_t, "snr": snr}


class FrequencyBranch(nn.Module):
    """Bravo: 仅使用 DWT-LFF 模式 (无传统 FFT-LFF)"""

    def __init__(self, channels: int, fused_channels: int,
                 dwt_lff: SpatialDWTLFFAdapter = None):
        super().__init__()
        self.channels = channels
        self.fused_channels = fused_channels
        self.dwt_lff = dwt_lff

        if channels == fused_channels:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
        else:
            self.in_proj = nn.Conv2d(channels, fused_channels, kernel_size=1, bias=True)
            self.out_proj = nn.Identity()

        self.lff = None  # Bravo: 无传统 FFT-LFF

    def forward(self, feats: torch.Tensor, center_idx: int) -> torch.Tensor:
        f_center = feats[:, center_idx]
        f_center = self.in_proj(f_center)

        if self.dwt_lff is not None:
            out = self.dwt_lff(f_center)
            f_f = out["feat_tfsi"]
        else:
            f_f = f_center

        f_f = self.out_proj(f_f)
        return f_f


class ConcatFusion(nn.Module):
    def __init__(self, fused_channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBlock(fused_channels * 2, fused_channels, kernel_size=3, stride=1, padding=1, act=True),
            ConvBlock(fused_channels, fused_channels, kernel_size=3, stride=1, padding=1, act=True),
        )

    def forward(self, f_s: torch.Tensor, f_f: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([f_s, f_f], dim=1))


class IntensityHead(nn.Module):
    """Bravo: phase_conf 调制 s_noise — 相位不可靠区域增强去噪"""
    def __init__(self, fused_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(fused_channels + 1, 2, 1, 1, 0)

    def forward(self, f_fused: torch.Tensor, phase_conf: torch.Tensor = None) -> dict:
        if phase_conf is not None:
            f_input = torch.cat([f_fused, phase_conf], dim=1)
        else:
            f_input = torch.cat([f_fused, torch.zeros_like(f_fused[:, :1])], dim=1)
        raw = self.conv(f_input)
        intensities = torch.sigmoid(raw)
        s_illum = intensities[:, 0:1]
        s_noise = intensities[:, 1:2]

        if phase_conf is not None:
            s_noise = s_noise * (1.0 + 0.5 * (1.0 - phase_conf))
            s_noise = s_noise.clamp(0.0, 1.0)

        return {"s_illum": s_illum, "s_noise": s_noise}


class TFSI(nn.Module):
    """TFSI 时频源指示器 — Bravo: 内建 phase_conf_head 相位置信度"""
    def __init__(self, channels: int = 64, fused_channels: int = 64, eps: float = 1e-6,
                 use_soft_median: bool = True, dwt_lff: SpatialDWTLFFAdapter = None):
        super().__init__()
        self.channels = channels
        self.fused_channels = fused_channels
        self.eps = eps

        self.norm = LayerNorm2d(channels)
        self.spatial_branch = SpatialBranch(channels, fused_channels, eps=eps,
                                            use_soft_median=use_soft_median)
        self.freq_branch = FrequencyBranch(channels, fused_channels, dwt_lff=dwt_lff)
        self.concat_fusion = ConcatFusion(fused_channels)
        self.intensity_head = IntensityHead(fused_channels)

        # Bravo: 相位置信度 — 从频域特征估计
        self.phase_conf_head = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(fused_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feats: torch.Tensor) -> dict:
        b, t, c, h, w = feats.shape
        center_idx = t // 2

        feats_norm = feats.view(b * t, c, h, w)
        feats_norm = self.norm(feats_norm).view(b, t, c, h, w)

        spatial_out = self.spatial_branch(feats_norm)
        f_s = spatial_out["F_s"]
        f_f = self.freq_branch(feats_norm, center_idx)
        f_fused = self.concat_fusion(f_s, f_f)

        # Bravo: 相位置信度
        phase_conf = self.phase_conf_head(f_f)
        intensities = self.intensity_head(f_fused, phase_conf=phase_conf)

        return {
            "F_fused": f_fused, "F_s": f_s, "F_f": f_f,
            "mu_t": spatial_out["mu_t"], "sigma_t": spatial_out["sigma_t"],
            "snr": spatial_out["snr"],
            "s_illum": intensities["s_illum"], "s_noise": intensities["s_noise"],
        }


# =====================================================================
# Stage 2: PureRWKVSACE (纯 RWKV 对齐, 替代 SACE + DAT)
# =====================================================================

class PureRWKVSACE(nn.Module):
    """
    Bravo: 多尺度双向 RWKV 帧间对齐 + 双 DWT-LFF 实例 (STCD + VSRELL)
      - lff_center (α=0.6): 中心帧光照锚定
      - lff_neighbor (α=0.4): 邻居帧退化诊断
      - 3 尺度 RWKV (full/half/quarter) + 边缘门控残差
    """

    def __init__(self, channels: int = 64, lff_module=None, n_layer: int = 1):
        super().__init__()
        self.channels = channels

        # Bravo P1: 双 DWT-LFF 实例
        self.lff_center = SpatialDWTLFFAdapter(in_channels=channels, alpha_init=0.6)
        self.lff_neighbor = SpatialDWTLFFAdapter(in_channels=channels, alpha_init=0.4)

        self._lff_external = lff_module is not None
        if lff_module is not None:
            self.lff_center = lff_module
            self.lff_neighbor = lff_module

        # 3 尺度 RWKV
        self.rwkv_full = VRWKVStyleSpatialMix(channels, num_frames=5, layer_id=0.0, n_layer=n_layer)
        self.rwkv_half = VRWKVStyleSpatialMix(channels, num_frames=5, layer_id=0.33, n_layer=n_layer)
        self.rwkv_quarter = VRWKVStyleSpatialMix(channels, num_frames=5, layer_id=0.67, n_layer=n_layer)

        # 边缘门控 (Video RWKV LCR edge prompt)
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

    def forward(self, feats: torch.Tensor, tfsi_out: Dict = None,
                cached_lff: Dict = None) -> Dict:
        B, T, C, H, W = feats.shape
        device = feats.device

        # Step 1: 逐帧 DWT-LFF (中心/邻居分支分离)
        lff_feats: List[torch.Tensor] = []
        center_idx = T // 2
        for t in range(T):
            if cached_lff and t in cached_lff:
                lff_feats.append(cached_lff[t])
            else:
                if t == center_idx:
                    lff_out = self.lff_center(feats[:, t])
                else:
                    lff_out = self.lff_neighbor(feats[:, t])
                if isinstance(lff_out, dict):
                    lff_feats.append(lff_out["feat_sace"])
                else:
                    lff_feats.append(lff_out)
        lff_stack = torch.stack(lff_feats, dim=1)

        mu_t_clean = lff_stack[:, center_idx]
        sigma_t_clean = lff_stack.std(dim=1, unbiased=False)
        s_noise = tfsi_out.get("s_noise") if tfsi_out else None

        # Step 2: 展平 token
        x_flat = lff_stack.permute(0, 1, 3, 4, 2).reshape(B, T, H * W, C)

        # Step 3: 多尺度双向 RWKV
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

        # Step 5: 边缘门控残差
        f_raw_center = feats[:, T // 2]
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
            "mu_t_clean": mu_t_clean, "sigma_t_clean": sigma_t_clean,
            "F_aligned_list": F_aligned_list, "lff_feats": lff_feats,
            "attn_maps": [],
        }


# =====================================================================
# Stage 3a: IFPN — Illumination-Filtering Pyramid Network
# =====================================================================

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
            nn.Conv2d(self.aligned_channels, coarse_channels, 1, 1, 0), nn.GELU(),
        )
        self.illum_extract = IllumExtract(img_channels, coarse_channels, feat_proj_channels, n_fea_middle, n_fea_in)
        self.img_estimator = nn.Sequential(
            nn.Conv2d(coarse_channels, 32, kernel_size=1, bias=True), nn.GELU(),
            nn.Conv2d(32, img_channels, kernel_size=1, bias=True),
        )
        self.ratio_proj = nn.Conv2d(img_channels, fused_channels, kernel_size=1, bias=True)
        self.feat_refine = nn.Sequential(
            ConvBlock(fused_channels, fused_channels, 3, 1, 1, act=True),
            ConvBlock(fused_channels, fused_channels, 1, 1, 0, act=False),
        )
        self.lit_up_proj = nn.Conv2d(fused_channels, img_channels, kernel_size=1, bias=True)

        # Bravo: side_head 用于 IFPN 中间感知监督
        self.side_head = nn.Sequential(
            nn.Conv2d(fused_channels, 32, 3, 1, 1, bias=True), nn.GELU(),
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

        neighbor_indices = [i for i in range(T) if i != center_idx]
        neighbors = coarse_feats[:, neighbor_indices]
        sim_logits = pairwise_cosine_logits(F_t_L, neighbors)
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
            "lit_up_map_raw": lit_up_map_raw, "f_illum_feat": f_illum_feat,
            "L_t": L_t, "L_ref": L_ref, "L_ratio": L_ratio, "ifpn_side": ifpn_side,
        }


# =====================================================================
# Stage 3b: NDPN — Noise-Denoising Pyramid Network
# =====================================================================

class NDPN(nn.Module):
    def __init__(self, channels: int = 64, tau_mid_init: float = 1.0, tau_scale_init: float = 1.0):
        super().__init__()
        self.channels = channels
        self.tau_mid = nn.Parameter(torch.tensor(tau_mid_init))
        self.log_tau_scale = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(tau_scale_init)))))
        self.alpha_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True), nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.alpha_conv[-1].weight)
        nn.init.zeros_(self.alpha_conv[-1].bias)
        self.refine = nn.Sequential(
            ConvBlock(channels, channels, 3, 1, 1, act=True),
            ConvBlock(channels, channels, 1, 1, 0, act=False),
        )

    def forward(self, feats, F_aligned_list, mu_t_clean, sigma_t_clean, s_noise, center_idx):
        B, T, C, H, W = feats.shape
        eps = 1e-6
        signal = mu_t_clean.abs().mean(dim=1, keepdim=True)
        noise = sigma_t_clean.mean(dim=1, keepdim=True)
        snr_hat = signal / (noise + eps)
        tau_scale = torch.exp(self.log_tau_scale).clamp(min=1e-2)
        s_snr = torch.sigmoid((snr_hat - self.tau_mid) / tau_scale)

        F_t = feats[:, center_idx]
        alphas: List[torch.Tensor] = []
        for i in range(T):
            F_i_aligned = F_aligned_list[i]
            if i == center_idx:
                alpha_i = s_snr
            else:
                resid = (F_i_aligned - F_t).abs()
                alpha_raw = torch.sigmoid(self.alpha_conv(resid))
                alpha_i = alpha_raw * (1.0 - s_snr)
            alphas.append(alpha_i)

        alpha_sum = torch.stack(alphas, dim=1).sum(dim=1) + eps
        F_denoised = torch.zeros_like(F_t)
        for i in range(T):
            w_i = alphas[i] / alpha_sum
            F_denoised = F_denoised + w_i * F_aligned_list[i]

        F_denoised = self.refine(F_denoised)
        f_noise_out = F_denoised

        return {"f_noise_out": f_noise_out, "s_snr": s_snr, "snr_hat": snr_hat}


# =====================================================================
# Stage 3c: MRPN — Motion-Refining Pyramid Network
# =====================================================================

class MRPN(nn.Module):
    def __init__(self, channels=64, window_size=8):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.gate = nn.Conv2d(channels * 2, channels, 1, 1, 0)
        self.refine = ResBlock(channels)

    def _aggregate_neighbors(self, f_t, f_omega):
        b, t, c, h, w = f_omega.shape
        feat = f_omega.reshape(b * t, c, h, w)
        feat, pad_hw = pad_to_window(feat, self.window_size)
        hp, wp = feat.shape[-2:]
        feat = feat.reshape(b, t, c, hp, wp)

        f_t_padded, _ = pad_to_window(f_t, self.window_size)
        center_windows = window_partition_2d(f_t_padded, self.window_size)
        feat_windows = window_partition_video(feat, self.window_size)

        corr = torch.matmul(center_windows, feat_windows.transpose(-1, -2)) / math.sqrt(c)
        corr = torch.softmax(corr, dim=-1)

        aligned_windows = torch.matmul(corr, feat_windows)
        aligned = window_reverse_2d(aligned_windows, self.window_size, hp, wp)
        aligned = unpad_from_window(aligned, pad_hw)
        return aligned

    def forward(self, F_aligned_list, center_idx):
        f_t_aligned = F_aligned_list[center_idx]
        f_neighbors = torch.stack(
            [F_aligned_list[i] for i in range(len(F_aligned_list)) if i != center_idx], dim=1,
        )
        f_omega_aligned = self._aggregate_neighbors(f_t_aligned, f_neighbors)

        z_t = torch.cat([f_t_aligned, f_omega_aligned], dim=1)
        g_t = torch.sigmoid(self.gate(z_t))
        f_t_fuse = g_t * f_t_aligned + (1.0 - g_t) * f_omega_aligned

        hat_f_t = self.refine(f_t_fuse) + f_t_aligned

        return {
            "f_omega_aligned": f_omega_aligned, "z_t": z_t, "G_t": g_t,
            "f_t_fuse": f_t_fuse, "f_motion_out": hat_f_t,
        }


# =====================================================================
# Stage 4: IGRF — Intensity-Guided Residual Fusion
# =====================================================================

def soft_clamp(x: torch.Tensor, sharpness: float = 20.0) -> torch.Tensor:
    return torch.sigmoid(sharpness * (x - 0.5))


def _make_res_blocks(channels: int, n: int, use_nafblock: bool = False):
    Block = NAFBlock if use_nafblock else ResBlock
    return nn.Sequential(*[Block(channels) for _ in range(n)])


class StageBlock(nn.Module):
    def __init__(self, channels: int, img_channels: int = 3, use_intensity: bool = False,
                 use_soft_clamp: bool = False, use_nafblock: bool = False,
                 num_res_blocks: int = 2):
        super().__init__()
        self.use_intensity = use_intensity
        self.use_soft_clamp = use_soft_clamp
        self.img_proj = nn.Conv2d(img_channels, channels, 3, 1, 1)
        Block = NAFBlock if use_nafblock else ResBlock
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0), nn.GELU(),
            *[Block(channels) for _ in range(num_res_blocks)],
            nn.Conv2d(channels, img_channels, 3, 1, 1),
        )
        if use_intensity:
            self.intensity_corr = nn.Conv2d(1, img_channels, kernel_size=3, padding=1, bias=True)
            nn.init.zeros_(self.intensity_corr.weight)
            nn.init.zeros_(self.intensity_corr.bias)

    def forward(self, f_branch, img_current, s_intensity=None):
        img_feat = self.img_proj(img_current)
        combined = torch.cat([f_branch, img_feat], dim=1)
        delta = self.fuse(combined)
        if self.use_intensity and s_intensity is not None:
            delta = delta + self.intensity_corr(s_intensity)
        if self.use_soft_clamp:
            img_next = soft_clamp(img_current + delta)
        else:
            img_next = torch.clamp(img_current + delta, 0.0, 1.0)
        return img_next, delta


class BrightenStage(nn.Module):
    def __init__(self, channels: int, img_channels: int = 3, max_delta: float = 0.5,
                 use_nafblock: bool = False):
        super().__init__()
        self.max_delta = max_delta
        self.feat_proj = nn.Conv2d(channels, img_channels, 3, 1, 1)
        self.img_proj = nn.Conv2d(img_channels, img_channels, 3, 1, 1)
        Block = NAFBlock if use_nafblock else ResBlock
        self.delta_refine = nn.Sequential(
            nn.Conv2d(img_channels * 2, img_channels, 1, 1, 0), nn.GELU(),
            Block(img_channels),
            nn.Conv2d(img_channels, img_channels, 3, 1, 1),
        )
        self.illum_corr = nn.Conv2d(channels, img_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.illum_corr.weight)
        nn.init.zeros_(self.illum_corr.bias)

    def forward(self, lit_up_map_raw, f_illum_feat, img_dark, s_illum=None):
        feat_cond = self.feat_proj(f_illum_feat)
        img_cond = self.img_proj(img_dark)
        delta = self.delta_refine(torch.cat([feat_cond, img_cond], dim=1))

        lit_up_map = lit_up_map_raw * (1.0 + torch.tanh(delta) * self.max_delta)
        lit_up_map = lit_up_map.clamp(min=0.5)
        brighten_base = img_dark * lit_up_map

        if s_illum is not None:
            corr_mag = self.illum_corr(f_illum_feat)
            illum_residual = s_illum * corr_mag
            res_t = torch.clamp(brighten_base + illum_residual, 0.0, 1.0)
        else:
            res_t = torch.clamp(brighten_base, 0.0, 1.0)

        return res_t, lit_up_map


class IGRF(nn.Module):
    def __init__(self, channels: int = 64, out_channels: int = 3, use_soft_clamp: bool = False,
                 use_nafblock: bool = False, num_res_blocks: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels

        self.stage_noise = StageBlock(channels, out_channels, use_intensity=True,
                                       use_soft_clamp=use_soft_clamp,
                                       use_nafblock=use_nafblock,
                                       num_res_blocks=num_res_blocks)
        self.stage_motion = StageBlock(channels, out_channels, use_intensity=False,
                                        use_soft_clamp=use_soft_clamp,
                                        use_nafblock=use_nafblock,
                                        num_res_blocks=num_res_blocks)
        self.brighten = BrightenStage(channels, out_channels, use_nafblock=use_nafblock)

    def forward(self, f_illum_feat, f_noise_out, f_motion_out,
                lit_up_map_raw, image_center, s_illum=None, s_noise=None):
        img_s1, delta_s1 = self.stage_noise(f_noise_out, image_center, s_intensity=s_noise)
        img_s2, delta_s2 = self.stage_motion(f_motion_out, img_s1)
        res_t, lit_up_map = self.brighten(lit_up_map_raw, f_illum_feat, img_s2, s_illum=s_illum)

        return {"res_t": res_t, "img_s1": img_s1, "img_s2": img_s2,
                "lit_up_map": lit_up_map, "delta_s1": delta_s1, "delta_s2": delta_s2}


# =====================================================================
# TFSNet — 主网络 (Bravo 版)
# =====================================================================

class TFSNet(nn.Module):
    """
    TFSNet v6 Bravo — 纯 RWKV 多帧低光增强

    Bravo 精简:
      - AmpEnhance (v5.9 前置幅度增强, 废弃, 节省 91K)          → 移除
      - DeformableCrossAttention / OffsetMaskHead (可变形对齐)   → 移除 (130K)
      - CrossRWKVGate (v6 中间门控)                              → 移除
      - SACE / share_lff / sace_* 参数                           → 移除
      - LFFFeatureAdapter (FFT 频域 RBF)                         → 移除
      + PureRWKVSACE 三尺度双向 RWKV + 边缘门控                  → 新增
      + SpatialDWTLFFAdapter 双实例 (中心 α=0.6 / 邻居 α=0.4)   → 新增
      + TFSI.phase_conf_head 相位置信度 → s_noise 调制           → 新增
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
            in_channels=in_channels, level_channels=level_channels,
            fused_channels=fused_channels, num_bottleneck_blocks=num_bottleneck_blocks,
        )

        # Bravo: DWT-LFF for TFSI (center α=0.6)
        dwt_lff_tfsi = SpatialDWTLFFAdapter(in_channels=fused_channels, alpha_init=0.6)

        # Stage 1: TFSI
        self.tfsi = TFSI(
            channels=fused_channels, fused_channels=fused_channels,
            eps=eps, use_soft_median=use_soft_median, dwt_lff=dwt_lff_tfsi,
        )

        # Stage 2: PureRWKVSACE (内部自建双 DWT-LFF, 无 Fallback)
        self.sace = PureRWKVSACE(channels=fused_channels, n_layer=1)

        # Stage 3: 三源恢复
        self.ifpn = IFPN(fused_channels=fused_channels, coarse_channels=coarse_channels, img_channels=in_channels)
        self.ndpn = NDPN(channels=fused_channels)
        self.mrpn = MRPN(channels=fused_channels)

        # Stage 4: IGRF
        self.igrf = IGRF(
            channels=fused_channels, out_channels=in_channels,
            use_soft_clamp=use_soft_clamp, use_nafblock=use_nafblock,
            num_res_blocks=num_igrf_res_blocks,
        )

        # 逐帧特征缓存
        self.frame_cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def clear_frame_cache(self):
        self.frame_cache.clear()

    def forward(self, x: torch.Tensor, frame_indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
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

        # Stage 1: TFSI
        tfsi_out = self.tfsi(feats)
        F_fused = tfsi_out["F_fused"]
        s_illum = tfsi_out["s_illum"]
        s_noise = tfsi_out["s_noise"]

        # Stage 2: PureRWKVSACE
        cached_lff: Dict[int, torch.Tensor] = {}
        if frame_indices:
            for i, gidx in enumerate(frame_indices):
                if gidx in self.frame_cache and "lff" in self.frame_cache[gidx]:
                    cached_lff[i] = self.frame_cache[gidx]["lff"]

        sace_out = self.sace(feats, tfsi_out, cached_lff=cached_lff if cached_lff else None)

        if frame_indices and "lff_feats" in sace_out:
            for i, gidx in enumerate(frame_indices):
                if gidx in self.frame_cache:
                    self.frame_cache[gidx]["lff"] = sace_out["lff_feats"][i]

        mu_t_clean = sace_out["mu_t_clean"]
        F_aligned_list = sace_out["F_aligned_list"]
        attn_maps = sace_out["attn_maps"]

        # Stage 3: 三源恢复
        image_center = x[:, center_idx]
        aligned_feats = torch.stack(F_aligned_list, dim=1)

        h_c, w_c = H // 4, W // 4
        image_down = F.interpolate(image_center, size=(h_c, w_c), mode='bicubic', align_corners=False)
        imgs_down = F.interpolate(
            x.view(B * T, C_in, H, W), size=(h_c, w_c), mode='bicubic', align_corners=False,
        ).view(B, T, C_in, h_c, w_c)

        ifpn_out = self.ifpn(image_down, aligned_feats, center_idx, imgs_down=imgs_down)

        ndpn_out = self.ndpn(
            feats, F_aligned_list, mu_t_clean,
            sigma_t_clean=sace_out["sigma_t_clean"],
            s_noise=s_noise, center_idx=center_idx,
        )

        mrpn_out = self.mrpn(F_aligned_list, center_idx=center_idx)

        # Stage 4: IGRF
        igrf_out = self.igrf(
            f_illum_feat=ifpn_out["f_illum_feat"],
            f_noise_out=ndpn_out["f_noise_out"],
            f_motion_out=mrpn_out["f_motion_out"],
            lit_up_map_raw=ifpn_out["lit_up_map_raw"],
            image_center=image_center,
            s_illum=s_illum, s_noise=s_noise,
        )

        return {
            "res_t": igrf_out["res_t"],
            "img_s1": igrf_out["img_s1"],
            "img_s2": igrf_out["img_s2"],
            "lit_up_map": igrf_out["lit_up_map"],
            "image_center": image_center,
            "s_illum": s_illum,
            "s_noise": s_noise,
            "f_illum_feat": ifpn_out["f_illum_feat"],
            "f_noise_out": ndpn_out["f_noise_out"],
            "f_motion_out": mrpn_out["f_motion_out"],
            "L_t": ifpn_out["L_t"],
            "L_ref": ifpn_out["L_ref"],
            "L_ratio": ifpn_out["L_ratio"],
            "ifpn_side": ifpn_out.get("ifpn_side"),
            "attn_maps": attn_maps,
            "mu_t_clean": mu_t_clean,
            "s_snr": ndpn_out["s_snr"],
            "motion_weights": mrpn_out["G_t"],
            "tfsi_out": tfsi_out,
        }
