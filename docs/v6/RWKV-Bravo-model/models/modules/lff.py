"""
LFF — Local Frequency Feature Adapter (FRBNet-based)
====================================================
FFT → RBF 幅度整形 → IFFT. 用于 IFPN 光照特征提取.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.fft as fft


class RadialBasisFilter(nn.Module):
    def __init__(self, K: int = 10, n_ang_freq: int = 1):
        super().__init__()
        self.K = K
        self.n_ang_freq = n_ang_freq

        mu = torch.linspace(0.0, 1.0, steps=K)
        self.register_buffer('mu', mu)

        self.log_bwh = nn.Parameter(torch.tensor(math.log(0.01)))

        self.coeff_mag = nn.Parameter(torch.zeros(K))
        self.raw_gate_mag = nn.Parameter(torch.ones(K))
        self.coeff_phase = nn.Parameter(torch.zeros(K))
        self.raw_gate_phase = nn.Parameter(torch.ones(K))

    @torch.no_grad()
    def _build_freq_grid(self, H: int, W: int, device, dtype):
        fy = fft.fftfreq(H, device=device, dtype=dtype).view(H, 1).expand(H, W)
        fx = fft.fftfreq(W, device=device, dtype=dtype).view(1, W).expand(H, W)
        r_grid = torch.sqrt(fx ** 2 + fy ** 2)
        r_hat = r_grid / (r_grid.max() + 1e-8)
        theta = torch.atan2(fy, fx)
        return r_hat, theta

    def forward(self, H: int, W: int, device, dtype):
        r_hat, theta = self._build_freq_grid(H, W, device, dtype)
        bwh = torch.exp(self.log_bwh).clamp(min=1e-3, max=1.0)
        basis = torch.exp(
            -((r_hat.unsqueeze(0) - self.mu.view(-1, 1, 1)) ** 2) / (2 * bwh ** 2)
        )
        if self.n_ang_freq > 0:
            angular_mod = 1.0 + 0.1 * torch.cos(self.n_ang_freq * theta)
            basis = basis * angular_mod.unsqueeze(0)

        gate_mag = torch.sigmoid(self.raw_gate_mag)
        diff_mag = (gate_mag.view(-1, 1, 1) * self.coeff_mag.view(-1, 1, 1) * basis).sum(dim=0, keepdim=True)
        gate_phase = torch.sigmoid(self.raw_gate_phase)
        diff_phase = (gate_phase.view(-1, 1, 1) * self.coeff_phase.view(-1, 1, 1) * basis).sum(dim=0, keepdim=True)
        return diff_mag, diff_phase


class LFFFeatureAdapter(nn.Module):
    def __init__(self, channels: int, K: int = 10, n_ang_freq: int = 1,
                 per_channel_rbf: bool = False, phase_preserving: bool = True):
        super().__init__()
        self.channels = channels
        self.per_channel_rbf = per_channel_rbf
        self.phase_preserving = phase_preserving

        if per_channel_rbf:
            self.rbf_bank = nn.ModuleList([
                RadialBasisFilter(K=K, n_ang_freq=n_ang_freq) for _ in range(channels)
            ])
        else:
            self.rbf = RadialBasisFilter(K=K, n_ang_freq=n_ang_freq)

        self.post_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.eye_(self.post_conv.weight.squeeze(-1).squeeze(-1))
        nn.init.zeros_(self.post_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        F = fft.fft2(x, dim=(-2, -1), norm='ortho')
        mag = torch.abs(F)
        phase = torch.angle(F)

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
            diff_mag = diff_mag.unsqueeze(0)
            diff_phase = diff_phase.unsqueeze(0)

        mag_new = mag * (1.0 + diff_mag)
        if self.phase_preserving:
            phase_new = phase
        else:
            phase_new = phase + diff_phase

        F_new = torch.polar(mag_new, phase_new)
        x_freq = fft.ifft2(F_new, dim=(-2, -1), norm='ortho').real
        x_out = self.post_conv(x_freq)
        return x_out
