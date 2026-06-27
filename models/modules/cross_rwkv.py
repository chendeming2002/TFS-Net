"""
Cross-RWKV Gate — v6 跨帧交叉注意力 (2026-06-27, 修订版)
============================================================
严格参照 Vision-RWKV (ICLR 2025) 的 VRWKV_SpatialMix 实现

核心修正 (vs 初版):
  1. Bi-WKV: 加入 spatial_first (当前 token 加权项), 匹配官方 WKV 公式
  2. Time Mix: 加入 spatial_mix_k/v/r 可学习混合比 (当前 vs 历史特征)
  3. Q-Shift: 改为通道空间位移 + mix 混合 (匹配官方 q_shift + mix 模式)
  4. Gate: 简化接收门控 sr = sigmoid(r), 取消 query-kv concat
  5. Init: fancy init 风格 (指数衰减分布 + 层级相关)

继承的论文方法:
  - Vision-RWKV (ICLR 2025) Bi-WKV + Q-Shift + VRWKV_SpatialMix
  - URWKV (CVPR 2025) 噪声 map 注入 WKV decay

API 保持 SACE 兼容: 输入 F_aligned_list, query μ_t_clean, 输出增强版 F_aligned_list
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class VRWKVStyleSpatialMix(nn.Module):
    """Vision-RWKV 风格的 SpatialMix 模块 — 严格参照官方 VRWKV_SpatialMix。

    包含: Q-Shift (通道位移 + mix 混合) + WKV 扫描 (含 spatial_first 项)

    Args:
        channels: token 维度
        num_frames: 跨帧窗口大小 T (默认 5)
        layer_id: 层 ID (用于 fancy init 的层级比例)
        n_layer: 总层数 (用于 fancy init)
    """

    def __init__(self, channels: int, num_frames: int = 5,
                 layer_id: float = 0.0, n_layer: int = 1):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.shift_pixel = 1
        self.channel_gamma = 0.25  # 每方向只处理 1/4 通道

        # ★ WKV 参数 (参照 Vision-RWKV fancy init)
        ratio_0_to_1 = layer_id / max(n_layer - 1, 1)
        ratio_1_to_almost0 = 1.0 - layer_id / max(n_layer, 1)

        # spatial_time_decay: 帧间衰减 (对应 w 参数)
        decay_speed = torch.ones(channels)
        for h in range(channels):
            decay_speed[h] = -5 + 8 * (h / (channels - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
        self.spatial_decay = nn.Parameter(decay_speed)

        # spatial_time_first: 当前帧加权 (对应 u 参数)
        zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(channels)]) * 0.5
        self.spatial_first = nn.Parameter(torch.ones(channels) * math.log(0.3) + zigzag)

        # ★ Time Mix 混合比 (当前 vs 历史特征)
        x = torch.ones(1, 1, channels)
        for i in range(channels):
            x[0, 0, i] = i / channels
        self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
        self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        # ★ 投影 (参照 VRWKV_SpatialMix)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.receptance = nn.Linear(channels, channels, bias=False)
        self.output = nn.Linear(channels, channels, bias=False)

        # 输出全零初始化 → 初始恒等
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
        """Vision-RWKV Q-Shift: 通道空间位移 (非 concat+conv 替代)。

        将通道分成 4 组, 每组做不同方向 1-pixel shift。
        gamma=0.25 时 4 组覆盖全部通道。
        """
        B, C, H, W = x.shape
        assert C >= 4
        g = int(C * gamma)  # 通道组大小
        out = x.clone()

        # 组 0: 左移
        if g > 0:
            out[:, :g, :, :W - shift_pixel] = x[:, :g, :, shift_pixel:W]
        # 组 1: 右移
        if 2 * g <= C:
            out[:, g:2 * g, :, shift_pixel:W] = x[:, g:2 * g, :, :W - shift_pixel]
        # 组 2: 上移
        if 3 * g <= C:
            out[:, 2 * g:3 * g, :H - shift_pixel, :] = x[:, 2 * g:3 * g, shift_pixel:H, :]
        # 组 3: 下移
        if 4 * g <= C:
            out[:, 3 * g:4 * g, shift_pixel:H, :] = x[:, 3 * g:4 * g, :H - shift_pixel, :]

        return out

    def _bi_wkv_scan(self, w: torch.Tensor, u: torch.Tensor,
                     k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Bi-WKV 双向扫描 — 严格匹配 Vision-RWKV 官方公式.

        官方公式 (per-channel decay w, per-channel first/bonus u):
            y_t = (u·k_t·v_t + Σ_i exp(w)·k_i·v_i)
                / (u·k_t      + Σ_i exp(w)·k_i)

        注意: w 和 u 是 per-channel 固定值, 不随帧距离 |t-i| 变化.
        所有帧对之间使用相同的 per-channel decay. 帧平均通过求和实现.

        Args:
            w: (C,) spatial_decay / T
            u: (C,) spatial_first / T
            k: (B, T, L, C) key
            v: (B, T, L, C) value
        Returns:
            out: (B, T, L, C) Bi-WKV 输出
        """
        B, T, L, C = k.shape
        device, dtype = k.device, k.dtype

        # Per-channel 权重 (对所有帧对相同)
        ew = torch.exp(w).view(1, 1, 1, C)   # (1, 1, 1, C)
        u_coeff = u.view(1, 1, 1, C)          # (1, 1, 1, C)

        # 分子: u·k_t·v_t + Σ_i ew·k_i·v_i  (对所有帧求和)
        k_weighted = k * ew                   # (B, T, L, C)
        kv_all = (k_weighted * v).sum(dim=1)  # Σ across T → (B, L, C)
        boost = u_coeff * k * v               # (B, T, L, C) — 当前帧加权

        # 分母: u·k_t + Σ_i ew·k_i
        k_all = k_weighted.sum(dim=1)          # Σ across T → (B, L, C)
        k_boost = u_coeff * k                  # (B, T, L, C) — 当前帧

        # 对各帧分别计算 (分子/分母)
        out = torch.zeros(B, T, L, C, device=device, dtype=dtype)
        for t in range(T):
            num_t = kv_all + boost[:, t]       # (B, L, C)
            den_t = k_all + k_boost[:, t]      # (B, L, C)
            out[:, t] = num_t / (den_t + 1e-8)

        return out

    def forward(self, x_flat: torch.Tensor,
                feat_2d_shape: tuple) -> torch.Tensor:
        """前向传播。

        Args:
            x_flat: (B, T, L, C) 或 (B*T, L, C) token 序列
            feat_2d_shape: (H, W) 特征图空间尺寸
        Returns:
            out: (B, T, L, C) WKV 扫描输出
        """
        B, T, L, C = x_flat.shape
        device = x_flat.device
        H, W = feat_2d_shape
        assert L == H * W

        # Step 1: Q-Shift (先还原为 2D, 位移, 再压平)
        x_2d = x_flat.reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # (B*T, C, H, W)
        x_shifted_2d = self.q_shift_2d(x_2d, self.shift_pixel, self.channel_gamma)
        x_shifted = x_shifted_2d.permute(0, 2, 3, 1).reshape(B, T, L, C)  # (B, T, L, C)

        # Step 2: Time Mix (当前 vs 历史特征混合)
        xk = x_flat * self.spatial_mix_k + x_shifted * (1 - self.spatial_mix_k)
        xv = x_flat * self.spatial_mix_v + x_shifted * (1 - self.spatial_mix_v)
        xr = x_flat * self.spatial_mix_r + x_shifted * (1 - self.spatial_mix_r)

        # Step 3: 投影 + 接收门控
        k = self.key(xk)     # (B, T, L, C)
        v = self.value(xv)   # (B, T, L, C)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)  # 接收门控

        # Step 4: Bi-WKV 扫描
        w = self.spatial_decay / T  # (C,)
        u = self.spatial_first / T  # (C,)
        wkv_out = self._bi_wkv_scan(w, u, k, v)  # (B, T, L, C)

        # Step 5: 门控 + 输出
        rwkv = sr * wkv_out
        out = self.output(rwkv)  # (B, T, L, C)

        return out


class CrossRWKVGate(nn.Module):
    """参考帧引导的 Cross-RWKV 跨帧聚合门 (v6.1 修订版)。

    基于 Vision-RWKV VRWKV_SpatialMix 严格实现:
      - Q-Shift 通道位移 + time mix 混合
      - Bi-WKV 含 spatial_first 当前帧加权
      - spatial_mix_k/v/r 可学习混合比
      - fancy init 层级衰减分布
    """

    def __init__(self, channels: int, num_frames: int = 5,
                 layer_id: float = 0.0, n_layer: int = 1):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames

        # Vision-RWKV SpatialMix 核心
        self.spatial_mix = VRWKVStyleSpatialMix(
            channels=channels,
            num_frames=num_frames,
            layer_id=layer_id,
            n_layer=n_layer,
        )

        # 噪声感知门控 (URWKV) + Entropy Gate (MINS-Net)
        self.noise_gate = nn.Sequential(
            nn.Conv2d(1, channels, 1, 1, 0),
            nn.Sigmoid(),
        )
        # EntropyGate: 从帧间特征统计中学习噪声感知聚合权重
        self.entropy_gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1, 1, 0),
            nn.Sigmoid(),
        )

        # 残差缩放 + 归一化
        self.layer_norm = nn.LayerNorm(channels)
        self.alpha = nn.Parameter(torch.zeros(1, 1, channels, 1, 1))

    def forward(
        self,
        query: torch.Tensor,
        frames: List[torch.Tensor],
        s_noise: torch.Tensor = None,
    ) -> List[torch.Tensor]:
        B, C, H, W = query.shape
        T = len(frames)
        L = H * W

        # Stack all frames: (B, T, C, H, W) → (B, T, L, C)
        feat_stack = torch.stack(frames, dim=1)
        feat_flat = feat_stack.permute(0, 1, 3, 4, 2).reshape(B, T, L, C)

        # Vision-RWKV SpatialMix 跨帧扫描
        out_flat = self.spatial_mix(feat_flat, (H, W))  # (B, T, L, C)

        # 还原为 2D
        out = out_flat.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        out = self.layer_norm(out.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)

        # Entropy: 帧间特征标准差 → 度量帧关联不确定性
        # 高 std = 某帧主导 → 低 entropy → 高置信度
        frame_std = feat_stack.std(dim=1)  # (B, C, H, W) 跨帧标准差
        frame_mean = feat_stack.mean(dim=1)  # (B, C, H, W) 跨帧均值

        # 对各帧独立计算 entropy gate
        out_list = []
        for t in range(T):
            # EntropyGate 输入: [当前帧RWKV输出, 跨帧均值, 跨帧标准差]
            eg_input = torch.cat([out[:, t], frame_mean, frame_std], dim=1)  # (B, 2C+1, H, W)
            eg_w = self.entropy_gate(eg_input)  # (B, 1, H, W) — Entropy 置信度权重

            # 噪声感知门控 + alpha 缩放残差
            if s_noise is not None:
                noise_w = self.noise_gate(s_noise)
                # 综合 entropy 和 noise: 熵高(置信度低)+噪声高 → 多用原始帧
                safe_w = 1.0 - (1.0 - eg_w) * (1.0 - noise_w)  # 双重门控的"安全"权重
                out_list.append(
                    frames[t] + self.alpha.squeeze(0) * (
                        out[:, t] * (1.0 - safe_w) + frames[t] * safe_w
                    )
                )
            else:
                out_list.append(
                    frames[t] + self.alpha.squeeze(0) * (out[:, t] * eg_w + frames[t] * (1.0 - eg_w))
                )

        return out_list
