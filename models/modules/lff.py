"""
LFF (Local Frequency Feature) Module — TFS-Net v3
====================================================
基于 FRBNet (Frequency-domain Radial Basis Network) 的频率域特征适配器。

实现状态:
    ✅ RadialBasisFilter: 共享 basis 的单实例双输出 RBF (修正 N2)
    ✅ LFFFeatureAdapter: 频率域特征整形 (修正 N1: r_hat 归一化)

设计原则:
    1. 一个 RBF 实例同时输出 mag/phase 响应，共享 mu/log_bwh/basis (N2)
    2. r_hat 归一化到 [0, 1]，与 mu ∈ [0, 1] 对齐 (N1)
    3. 角度调制: 1 + 0.1*cos(n*θ)，值域 [0.9, 1.1] (FRBNet 一致)
    4. 频率域整形不改变特征维度: (B, C, H, W) → (B, C, H, W)

参考: reference_repos/FRBNet/frbnet_utils.py L7-47
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.fft as fft


class RadialBasisFilter(nn.Module):
    """
    径向基频率滤波器 — 一个实例同时输出幅度响应和相位响应 (N2 修正)。

    所有基函数 (mu, bwh, angular_mod, basis) 在 mag/phase 之间共享，
    仅 coeff 和 gate 独立。

    Args:
        K (int): 径向基函数个数 (默认 10, 与 FRBNet 论文一致)
        n_ang_freq (int): 角度调制频率 (默认 1, 与 FRBNet 论文一致)

    Forward:
        Input:  H (int), W (int), device, dtype
        Output: (diff_mag, diff_phase), 各 shape (1, H, W)
    """

    def __init__(self, K: int = 10, n_ang_freq: int = 1):
        super().__init__()
        self.K = K
        self.n_ang_freq = n_ang_freq

        # 共享: 径向基中心 (固定，均匀分布在归一化频率轴上)
        mu = torch.linspace(0.0, 1.0, steps=K)
        self.register_buffer('mu', mu)

        # 共享: 带宽 (可学习)
        # 初始 bwh ≈ 0.01，FRBNet 实际 bwh ≈ 0.002，此处折中
        self.log_bwh = nn.Parameter(torch.tensor(math.log(0.01)))

        # 独立: 幅度通路
        self.coeff_mag = nn.Parameter(torch.zeros(K))
        self.raw_gate_mag = nn.Parameter(torch.ones(K))

        # 独立: 相位通路
        self.coeff_phase = nn.Parameter(torch.zeros(K))
        self.raw_gate_phase = nn.Parameter(torch.ones(K))

    @torch.no_grad()
    def _build_freq_grid(self, H: int, W: int, device, dtype):
        """构造归一化频率网格 r_hat 和角度 theta"""
        fy = fft.fftfreq(H, device=device, dtype=dtype).view(H, 1).expand(H, W)
        fx = fft.fftfreq(W, device=device, dtype=dtype).view(1, W).expand(H, W)
        r_grid = torch.sqrt(fx ** 2 + fy ** 2)
        # N1 修正: 归一化到 [0, 1]
        r_hat = r_grid / (r_grid.max() + 1e-8)
        theta = torch.atan2(fy, fx)
        return r_hat, theta

    def forward(self, H: int, W: int, device, dtype):
        r_hat, theta = self._build_freq_grid(H, W, device, dtype)

        bwh = torch.exp(self.log_bwh).clamp(min=1e-3, max=1.0)
        basis = torch.exp(
            -((r_hat.unsqueeze(0) - self.mu.view(-1, 1, 1)) ** 2)
            / (2 * bwh ** 2)
        )  # (K, H, W)

        # 角度调制: 1 + 0.1*cos(n*θ)，值域 [0.9, 1.1]
        if self.n_ang_freq > 0:
            angular_mod = 1.0 + 0.1 * torch.cos(self.n_ang_freq * theta)
            basis = basis * angular_mod.unsqueeze(0)

        # 幅度通路
        gate_mag = torch.sigmoid(self.raw_gate_mag)
        diff_mag = (
            gate_mag.view(-1, 1, 1) * self.coeff_mag.view(-1, 1, 1) * basis
        ).sum(dim=0, keepdim=True)  # (1, H, W)

        # 相位通路
        gate_phase = torch.sigmoid(self.raw_gate_phase)
        diff_phase = (
            gate_phase.view(-1, 1, 1) * self.coeff_phase.view(-1, 1, 1) * basis
        ).sum(dim=0, keepdim=True)  # (1, H, W)

        return diff_mag, diff_phase


class LFFFeatureAdapter(nn.Module):
    """
    Local Frequency Feature 特征适配器。

    将输入特征通过 2D FFT 转到频率域，
    用 RBF 对幅度和相位做加性整形（残差形式），再 IFFT 回空间域。

    数学形式:
        F = FFT2(x)
        mag, phase = |F|, ∠F
        mag'   = mag * (1 + diff_mag)
        phase' = phase + diff_phase
        F' = mag' * exp(i * phase')
        x' = Real(IFFT2(F'))

    Args:
        channels (int): 输入特征通道数 (不变)
        K (int): RBF 基函数个数
        n_ang_freq (int): 角度调制频率
        per_channel_rbf (bool): 是否每通道独立 RBF

    Forward:
        Input:  x (B, C, H, W)
        Output: x' (B, C, H, W)
    """

    def __init__(
        self,
        channels: int,
        K: int = 10,
        n_ang_freq: int = 1,
        per_channel_rbf: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.per_channel_rbf = per_channel_rbf

        if per_channel_rbf:
            self.rbf_bank = nn.ModuleList([
                RadialBasisFilter(K=K, n_ang_freq=n_ang_freq)
                for _ in range(channels)
            ])
        else:
            self.rbf = RadialBasisFilter(K=K, n_ang_freq=n_ang_freq)

        # 频域整形后的通道融合 (1x1 conv)
        self.post_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        # 初始化为恒等映射
        nn.init.eye_(self.post_conv.weight.squeeze(-1).squeeze(-1))
        nn.init.zeros_(self.post_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.channels, f"通道数不匹配: {C} vs {self.channels}"

        device, dtype = x.device, x.dtype

        # Step 1: 2D FFT
        F = fft.fft2(x, dim=(-2, -1), norm='ortho')
        mag = torch.abs(F)
        phase = torch.angle(F)

        # Step 2: 计算频域响应
        if self.per_channel_rbf:
            diff_mag_list, diff_phase_list = [], []
            for c in range(C):
                dm, dp = self.rbf_bank[c](H, W, device, dtype)
                diff_mag_list.append(dm)
                diff_phase_list.append(dp)
            diff_mag = torch.stack(diff_mag_list, dim=0).unsqueeze(0)
            diff_phase = torch.stack(diff_phase_list, dim=0).unsqueeze(0)
        else:
            diff_mag, diff_phase = self.rbf(H, W, device, dtype)
            diff_mag = diff_mag.unsqueeze(0)      # (1, 1, H, W)
            diff_phase = diff_phase.unsqueeze(0)  # (1, 1, H, W)

        # Step 3: 残差整形
        mag_new = mag * (1.0 + diff_mag)
        phase_new = phase + diff_phase

        # Step 4: IFFT
        F_new = torch.polar(mag_new, phase_new)
        x_freq = fft.ifft2(F_new, dim=(-2, -1), norm='ortho').real

        # Step 5: 后处理
        x_out = self.post_conv(x_freq)

        return x_out
