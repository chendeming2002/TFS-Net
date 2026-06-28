"""
VRWKVStyleSpatialMix — RWKV-Bravo 帧间扫描核心
===============================================
严格参照 Vision-RWKV (ICLR 2025) VRWKV_SpatialMix 实现:
  - Q-Shift: 通道空间位移 + time mix 混合
  - Bi-WKV: 含 spatial_first 当前帧加权, 严格匹配官方公式
  - fancy init: 指数衰减分布 + 层级相关

Bravo: 移除 CrossRWKVGate（由 PureRWKVSACE 直接调用本模块）
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VRWKVStyleSpatialMix(nn.Module):
    def __init__(self, channels: int, num_frames: int = 5,
                 layer_id: float = 0.0, n_layer: int = 1):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.shift_pixel = 1
        self.channel_gamma = 0.25

        ratio_0_to_1 = layer_id / max(n_layer - 1, 1)
        ratio_1_to_almost0 = 1.0 - layer_id / max(n_layer, 1)

        decay_speed = torch.ones(channels)
        for h in range(channels):
            decay_speed[h] = -5 + 8 * (h / (channels - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
        self.spatial_decay = nn.Parameter(decay_speed)

        zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(channels)]) * 0.5
        self.spatial_first = nn.Parameter(torch.ones(channels) * math.log(0.3) + zigzag)

        x = torch.ones(1, 1, channels)
        for i in range(channels):
            x[0, 0, i] = i / channels
        self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
        self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.receptance = nn.Linear(channels, channels, bias=False)
        self.output = nn.Linear(channels, channels, bias=False)

        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.key.weight)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.receptance.weight)
        for mod in [self.output, self.key, self.value, self.receptance]:
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)

    @staticmethod
    def q_shift_2d(x: torch.Tensor, shift_pixel: int = 1,
                   gamma: float = 0.25) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C >= 4
        g = int(C * gamma)
        out = x.clone()
        if g > 0:
            out[:, :g, :, :W - shift_pixel] = x[:, :g, :, shift_pixel:W]
        if 2 * g <= C:
            out[:, g:2 * g, :, shift_pixel:W] = x[:, g:2 * g, :, :W - shift_pixel]
        if 3 * g <= C:
            out[:, 2 * g:3 * g, :H - shift_pixel, :] = x[:, 2 * g:3 * g, shift_pixel:H, :]
        if 4 * g <= C:
            out[:, 3 * g:4 * g, shift_pixel:H, :] = x[:, 3 * g:4 * g, :H - shift_pixel, :]
        return out

    def _bi_wkv_scan(self, w: torch.Tensor, u: torch.Tensor,
                     k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, T, L, C = k.shape
        device, dtype = k.device, k.dtype
        ew = torch.exp(w).view(1, 1, 1, C)
        u_coeff = u.view(1, 1, 1, C)
        k_weighted = k * ew
        kv_all = (k_weighted * v).sum(dim=1)
        boost = u_coeff * k * v
        k_all = k_weighted.sum(dim=1)
        k_boost = u_coeff * k
        out = torch.zeros(B, T, L, C, device=device, dtype=dtype)
        for t in range(T):
            num_t = kv_all + boost[:, t]
            den_t = k_all + k_boost[:, t]
            out[:, t] = num_t / (den_t + 1e-8)
        return out

    def forward(self, x_flat: torch.Tensor, feat_2d_shape: tuple) -> torch.Tensor:
        B, T, L, C = x_flat.shape
        device = x_flat.device
        H, W = feat_2d_shape
        assert L == H * W

        x_2d = x_flat.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
        x_shifted_2d = self.q_shift_2d(x_2d, self.shift_pixel, self.channel_gamma)
        x_shifted = x_shifted_2d.permute(0, 2, 3, 1).reshape(B, T, L, C)

        xk = x_flat * self.spatial_mix_k + x_shifted * (1 - self.spatial_mix_k)
        xv = x_flat * self.spatial_mix_v + x_shifted * (1 - self.spatial_mix_v)
        xr = x_flat * self.spatial_mix_r + x_shifted * (1 - self.spatial_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        w = self.spatial_decay / T
        u = self.spatial_first / T
        wkv_out = self._bi_wkv_scan(w, u, k, v)

        rwkv = sr * wkv_out
        out = self.output(rwkv)
        return out
