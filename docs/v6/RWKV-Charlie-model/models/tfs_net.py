"""
TFSNet — RWKV-Charlie 完整模型
==============================

Charlie 核心架构改进 (Bravo2):
  P0 - Loss: lambda_perc=0.04, lambda_pix=1.0, lambda_ssim=0.2
  P1 - RWKV: post_norm + decay/first clamp
  P2 - DWT: 全高频, 取消 0.5 共享
  Charlie:
    1. TFSI.FrequencyBranch — 多帧 temporal_fusion
    2. PureRWKVSACE — 多尺度 concat + channel_mix 融合
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    ConvBlock, ResBlock, NAFBlock, LayerNorm2d,
    window_partition_video, window_reverse_video,
    pad_to_window, unpad_from_window,
)
from .cross_rwkv import VRWKVStyleSpatialMix
from .dwt_lff import SpatialDWTLFFAdapter


class ChannelNorm(nn.Module):
    """可学习的 channel-wise 仿射变换"""
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-6
        return (x - mean) / std * self.gamma + self.beta


class PixelShuffleUpsampler(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int = 4):
        super().__init__()
        self.scale = scale
        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels * scale, 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvResBlock(nn.Module):
    def __init__(self, channels: int, n_blocks: int = 2):
        super().__init__()
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                LayerNorm2d(channels),
                ConvBlock(channels, channels, 3, 1, 1),
            )
            for _ in range(n_blocks)
        ])
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return x + self.blocks(x) * self.gamma


class PureRWKVSACE(nn.Module):
    """PureRWKVSACE — v6 Charlie: 多尺度 concat + channel_mix 融合
    
    Charlie 改动: 去掉 /3 平均, 改用 concat + channel_mix
    """
    def __init__(self, channels: int = 32, num_frames: int = 5,
                 num_saces: int = 3, layer_id: float = 1.0, n_layer: int = 2):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.num_saces = num_saces
        self.sace_in = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
                nn.Conv2d(channels, channels, 1),
                nn.GELU(),
            )
            for _ in range(num_saces)
        ])
        proj_out_dim = channels * 3 * num_saces
        self.channel_mix = nn.Sequential(
            nn.Conv2d(proj_out_dim, channels, 1, groups=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.mid_convs = nn.Sequential(
            ConvBlock(channels, channels, 3, 1, 1),
            ConvBlock(channels, channels, 3, 1, 1),
        )
        self.sace_out = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
                nn.Conv2d(channels, channels, 1),
            )
            for _ in range(num_saces)
        ])
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B, T, C, H, W = x.shape
        xs = []
        B_flat = B * T
        feat_2d = x.view(B_flat, C, H, W)
        for i, proj in enumerate(self.sace_in):
            xs_i = proj(feat_2d).view(B, T, 1, C, H, W)
            xs.append(xs_i.expand(-1, -1, self.num_saces, -1, -1, -1))
        cat_3 = torch.cat(xs, dim=3)
        cat_flat = cat_3.permute(0, 1, 3, 4, 5, 2).reshape(B, T, C * 3 * self.num_saces, H, W)
        B_clip, T_clip, C_clip, H_clip, W_clip = cat_flat.shape
        flat_2d = cat_flat.reshape(B * T_clip, C_clip, H_clip, W_clip)
        fused = self.channel_mix(flat_2d)
        fused = self.mid_convs(fused)
        out_parts = []
        for i, out_proj in enumerate(self.sace_out):
            start = i * C
            end = (i + 1) * C
            part = fused[:, start:end, :, :]
            part = out_proj(part)
            out_parts.append(part.view(B, T, C, H, W))
        out = torch.stack(out_parts, dim=2).mean(dim=2)
        return out


class LearnableSigmoidScheduler(nn.Module):
    def __init__(self, channels: int, alpha_max: float = 8.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1) * 2.0)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.alpha_max = alpha_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x * self.alpha + self.beta) * self.alpha_max


class MRPN(nn.Module):
    """Charlie: σ→MRPN — sigma_t 作为额外输入
    
    融合 sigma_t 到 gate, 用 flip+corr 估计双向运动
    """
    def __init__(self, channels: int, num_frames: int = 5):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.sigma_proj = nn.Sequential(
            nn.Linear(num_frames, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.corr_block = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def forward(self, x: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        sigma_feat = self.sigma_proj(sigma_t).view(B, 1, C, 1, 1)
        g = torch.sigmoid(x + sigma_feat)
        x_fwd = x[:, 1:] - x[:, :-1]
        x_bwd = x[:, :-1] - x[:, 1:]
        x_fwd = F.pad(x_fwd, (0, 0, 0, 0, 0, 0, 0, 1))
        x_bwd = F.pad(x_bwd, (0, 0, 0, 0, 0, 0, 1, 0))
        diff = torch.cat([x_fwd.abs(), x_bwd.abs()], dim=2)
        diff_flat = diff.view(B * T, C * 2, H, W)
        w = self.corr_block(diff_flat).view(B, T, 1, H, W)
        gated = g * x * (1 + x * w * self.gamma)
        return gated + x


class SpatialFreqInteraction(nn.Module):
    """TFSI_v2 空间频域交互"""
    def __init__(self, channels: int, num_frames: int = 5):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_perm = x.permute(0, 1, 3, 4, 2)
        normed = self.norm(x_perm)
        freq = torch.fft.rfft2(normed, norm='ortho')
        mag = freq.abs()
        phase = freq.angle()
        mag_shifted = torch.fft.fftshift(mag, dim=(-2, -1))
        mag_enhanced = mag_shifted * (1 + self.gamma.view(1, 1, 1, 1, C).tanh())
        mag_ishifted = torch.fft.ifftshift(mag_enhanced, dim=(-2, -1))
        enhanced_freq = torch.polar(mag_ishifted, phase)
        enhanced = torch.fft.irfft2(enhanced_freq, norm='ortho')
        enhanced_perm = enhanced.permute(0, 1, 4, 2, 3)
        return x + enhanced_perm * self.gamma


class FrequencyBranch(nn.Module):
    """Charlie: 多帧 FrequencyBranch — temporal_fuse 融合相邻帧频域特征"""
    def __init__(self, channels: int, num_frames: int = 5, use_temporal_fusion: bool = True):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.use_temporal_fusion = use_temporal_fusion

        self.conv_norm = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )

        # Charlie: 多帧 temporal_fuse
        if use_temporal_fusion:
            self.temporal_fuse = nn.Conv3d(channels, channels, (3, 1, 1),
                                            padding=(1, 0, 0), bias=False)

        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.omega = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape

        if self.use_temporal_fusion:
            x_3d = x.permute(0, 2, 1, 3, 4)
            x_3d = self.temporal_fuse(x_3d)
            x_t = x_3d.permute(0, 2, 1, 3, 4)
        else:
            x_t = x

        x_flat = x_t.view(B * T, C, H, W)
        x_norm = self.conv_norm(x_flat)
        x_f = torch.fft.rfft2(x_norm, norm='ortho')
        mag = x_f.abs()
        phase = x_f.angle()
        mag_shifted = torch.fft.fftshift(mag, dim=(-2, -1))
        mag_enhanced = mag_shifted * (1 + self.alpha.tanh())
        mag_ishifted = torch.fft.ifftshift(mag_enhanced, dim=(-2, -1))
        enhanced_freq = torch.polar(mag_ishifted, phase)
        enhanced = torch.fft.irfft2(enhanced_freq, norm='ortho')
        enhanced = enhanced.view(B, T, C, enhanced.shape[-2], enhanced.shape[-1])
        enhanced = enhanced.permute(0, 1, 3, 4, 2)
        enhanced = self.omega(enhanced).permute(0, 1, 4, 2, 3)
        return x + enhanced * self.alpha.view(1, 1, -1, 1, 1)


class TFBI(nn.Module):
    """TFS-Block (TFBI): 时频频交互 (Bravo 原生)"""
    def __init__(self, channels: int, num_frames: int = 5):
        super().__init__()
        self.enhance = SpatialFreqInteraction(channels, num_frames)
        self.freq_branch = FrequencyBranch(channels, num_frames)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        spatial = self.enhance(x)
        freq = self.freq_branch(x)
        return spatial + freq * self.gamma


class PCDwAlign(nn.Module):
    """PCD 对齐 (光流驱动)"""
    def __init__(self, channels: int, num_frames: int = 5):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.conv_offset = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, 2, 3, 1, 1),
        )
        self.conv_weight = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    @staticmethod
    def apply_offset(x, offset):
        B, C, H, W = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing='ij',
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float().unsqueeze(0).repeat(B, 1, 1, 1)
        offset = offset.permute(0, 2, 3, 1)
        grid = grid + offset
        grid = grid * 2 / torch.tensor([W - 1, H - 1], device=x.device) - 1
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)

    def forward(self, ref_feat, neigh_feat, ref_img=None, neigh_img=None):
        B, C, H, W = ref_feat.shape
        if neigh_feat is None:
            return ref_feat
        inp = torch.cat([ref_feat, neigh_feat], dim=1)
        offset = self.conv_offset(inp)
        warped = self.apply_offset(neigh_feat, offset)
        raw_weight = self.conv_weight(torch.cat([ref_feat, warped], dim=1))
        gate_weight = torch.sigmoid(raw_weight)
        weighted = warped * gate_weight + warped * self.gamma.view(1, -1, 1, 1)
        return weighted


class CrossScaleAttention(nn.Module):
    def __init__(self, channels, num_frames=5):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(B, -1, C)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = self.norm(x_flat + self.gamma * attn_out)
        x_out = x_flat.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)
        return x_out


class TFSNet(nn.Module):
    """TFSNet v6 — RWKV-Charlie 完整模型
    
    Charlie 特有的配置参数:
      use_temporal_fusion (bool): TFSI.FrequencyBranch 多帧 temporal_fusion
      use_channel_mix (bool): PureRWKVSACE concat + channel_mix 融合
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_frames: int = 5,
        width: int = 32,
        middle_channel: int = 32,
        use_rwkv: bool = True,
        deep_supervision: bool = False,
        use_temporal_fusion: bool = True,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.use_rwkv = use_rwkv
        self.deep_supervision = deep_supervision

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, 1, 1),
            ConvBlock(width, width, 3, 1, 1),
        )

        self.conv_down2 = ConvBlock(width, width * 2, 3, 2, 1)
        self.conv_down4 = ConvBlock(width * 2, width * 4, 3, 2, 1)

        self.dwt_lff_4 = SpatialDWTLFFAdapter(width * 4, alpha_init=0.5)
        self.dwt_lff_2 = SpatialDWTLFFAdapter(width * 2, alpha_init=0.5)

        self.tfsi = TFBI(width * 4, num_frames)

        self.sace = PureRWKVSACE(
            channels=width * 4,
            num_frames=num_frames,
            num_saces=3,
            layer_id=1.0,
            n_layer=2,
        )

        self.conv_up2 = nn.Sequential(
            ConvBlock(width * 8, width * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            ConvBlock(width, width * 2, 3, 1, 1),
        )

        self.conv_up4_pre = nn.Sequential(
            nn.Conv2d(width * 4, width * 2, 3, 1, 1),
            nn.PixelShuffle(2),
        )

        self.conv_up4 = nn.Sequential(
            ConvBlock(width * 2 + width * 2, width * 2, 3, 1, 1),
            nn.PixelShuffle(2),
            ConvBlock(width * 2, width, 3, 1, 1),
        )

        if self.use_rwkv:
            self.cross_rwkv = VRWKVStyleSpatialMix(
                channels=width * 4,
                num_frames=num_frames,
                layer_id=1.0,
                n_layer=2,
            )

        self.csa = CrossScaleAttention(width * 4, num_frames)

        self.tail = nn.Sequential(
            ConvBlock(width * 2 + width, in_channels, 3, 1, 1),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        )

        if deep_supervision:
            self.ds_tail_1 = nn.Conv2d(width * 4, in_channels, 1)
            self.ds_tail_2 = nn.Conv2d(width * 2, in_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        f1 = self.head(x_flat)
        f2 = self.conv_down2(f1)
        f4 = self.conv_down4(f2)

        f4 = f4.view(B, T, -1, f4.shape[-2], f4.shape[-1])

        # DWT-LFF
        f4_flat = f4.view(B * T, -1, f4.shape[-2], f4.shape[-1])
        f4_dwt = self.dwt_lff_4(f4_flat)
        f4_sace = f4_dwt["feat_sace"].view(B, T, -1, f4.shape[-2], f4.shape[-1])
        f4_tfsi = f4_dwt["feat_tfsi"].view(B, T, -1, f4.shape[-2], f4.shape[-1])

        # TFSI
        f4_tfsi = self.tfsi(f4_tfsi)

        # SACE
        f4_sace = self.sace(f4_sace)

        # Fusion
        f4_fused = f4_sace + f4_tfsi

        # Cross-RWKV
        if self.use_rwkv:
            B_f, T_f, C_f, H_f, W_f = f4_fused.shape
            f4_flat_rwkv = f4_fused.permute(0, 1, 3, 4, 2).reshape(B_f, T_f, H_f * W_f, C_f)
            f4_rwkv = self.cross_rwkv(f4_flat_rwkv, (H_f, W_f))
            f4_fused = f4_rwkv.reshape(B_f, T_f, H_f, W_f, C_f).permute(0, 1, 4, 2, 3)

        f4_fused = self.csa(f4_fused)

        f4_out = f4_fused.view(B * T, -1, H_f, W_f)

        # Deep supervision
        if self.deep_supervision:
            ds1 = self.ds_tail_1(f4_out).view(B, T, C, H_f, W_f)
            ds1 = F.interpolate(ds1.view(B, C, H_f, W_f), size=(H, W), mode='bilinear', align_corners=False)

        # Decoder
        f2_flat = f2.view(B * T, -1, f2.shape[-2], f2.shape[-1])
        f2_dwt = self.dwt_lff_2(f2_flat)
        f2_feat = f2_dwt["x_out"].view(B, T, -1, f2.shape[-2], f2.shape[-1])
        f2_feat = f2_feat.view(B * T, -1, f2.shape[-2], f2.shape[-1])

        f2_up = self.conv_up2(torch.cat([f4_out, f4_out], dim=1))
        f2_up = f2_up + F.interpolate(f2_feat, size=f2_up.shape[-2:], mode='bilinear', align_corners=False)

        f1_up_pre = self.conv_up4_pre(f4_out)
        f1_up = self.conv_up4(torch.cat([f1_up_pre, f2_up], dim=1))

        out = self.tail(torch.cat([f1_up, f1], dim=1))

        if self.deep_supervision:
            ds2 = self.ds_tail_2(f2_up)
            ds2 = F.interpolate(ds2, size=(H, W), mode='bilinear', align_corners=False)
            return [out, ds1, ds2]

        return out
