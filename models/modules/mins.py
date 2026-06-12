import math

import torch
import torch.nn as nn

from .blocks import LayerNorm2d, pad_to_window, unpad_from_window, window_partition_2d, window_partition_video, window_reverse_2d, window_reverse_video


class EntropyGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1, 1, 0),
        )

    def forward(self, feat, entropy):
        return torch.sigmoid(self.net(torch.cat([feat, entropy], dim=1)))


class MINSBlock(nn.Module):
    def __init__(self, channels=48, window_size=8):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.norm = LayerNorm2d(channels)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.q_proj = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k_proj = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v_proj = nn.Conv2d(channels, channels, 1, 1, 0)
        self.entropy_gate = EntropyGate(channels)

    def _pad_video(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x, pad_hw = pad_to_window(x, self.window_size)
        hp, wp = x.shape[-2:]
        x = x.view(b, t, c, hp, wp)
        return x, pad_hw

    def _forward_attention(self, q_feat, k_feat, v_feat):
        q_windows = window_partition_2d(q_feat, self.window_size)
        k_windows = window_partition_video(k_feat, self.window_size)
        v_windows = window_partition_video(v_feat, self.window_size)

        attn_logits = torch.matmul(q_windows, k_windows.transpose(-1, -2)) / math.sqrt(self.channels)
        corr = torch.softmax(attn_logits, dim=-1)
        aggregated = torch.matmul(corr, v_windows)

        entropy = -(corr * torch.log(corr.clamp_min(1e-6))).sum(dim=-1)
        entropy = entropy.unsqueeze(-1)

        return aggregated, corr, entropy

    def _project_neighbor_prior(self, corr, prior_center, num_neighbors, padded_hw):
        hp, wp = padded_hw
        prior_windows = window_partition_2d(prior_center, self.window_size)
        neighbor_prior = torch.matmul(corr.transpose(-1, -2), prior_windows)
        return window_reverse_video(neighbor_prior, self.window_size, num_neighbors, hp, wp)

    def forward(self, f_center, f_neighbors):
        num_neighbors = f_neighbors.shape[1]

        f_center = self.norm(f_center)
        b, c, h, w = f_center.shape
        f_neighbors = f_neighbors.view(b * num_neighbors, c, h, w)
        f_neighbors = self.norm(f_neighbors).view(b, num_neighbors, c, h, w)

        center_pad, pad_hw = pad_to_window(f_center, self.window_size)
        neighbors_pad, _ = self._pad_video(f_neighbors)
        hp, wp = center_pad.shape[-2:]

        q_input = center_pad - self.avg_pool(center_pad)
        k_input = neighbors_pad.view(b * num_neighbors, c, hp, wp)
        k_input = k_input - self.avg_pool(k_input)
        k_input = k_input.view(b, num_neighbors, c, hp, wp)

        q_feat = self.q_proj(q_input)
        k_feat = self.k_proj(k_input.view(b * num_neighbors, c, hp, wp)).view(b, num_neighbors, c, hp, wp)
        v_feat = self.v_proj(neighbors_pad.view(b * num_neighbors, c, hp, wp)).view(b, num_neighbors, c, hp, wp)

        aggregated, corr, entropy = self._forward_attention(q_feat, k_feat, v_feat)

        aggregated = window_reverse_2d(aggregated, self.window_size, hp, wp)
        entropy = window_reverse_2d(entropy, self.window_size, hp, wp)

        f_center_c = aggregated + center_pad
        p_center_m = self.entropy_gate(f_center_c, entropy)
        p_center_i = 1.0 - p_center_m

        p_neighbor_m = self._project_neighbor_prior(corr, p_center_m, num_neighbors, (hp, wp))
        p_neighbor_m = p_neighbor_m.clamp(0.0, 1.0)
        p_neighbor_i = 1.0 - p_neighbor_m

        f_center_m = p_center_m * f_center_c
        f_center_i = p_center_i * f_center_c

        f_neighbors_m = p_neighbor_m * neighbors_pad
        f_neighbors_i = p_neighbor_i * neighbors_pad

        f_center_m = unpad_from_window(f_center_m, pad_hw)
        f_center_i = unpad_from_window(f_center_i, pad_hw)
        p_center_m = unpad_from_window(p_center_m, pad_hw)
        p_center_i = unpad_from_window(p_center_i, pad_hw)
        entropy = unpad_from_window(entropy, pad_hw)

        f_neighbors_m = unpad_from_window(f_neighbors_m.view(b * num_neighbors, c, hp, wp), pad_hw)
        f_neighbors_i = unpad_from_window(f_neighbors_i.view(b * num_neighbors, c, hp, wp), pad_hw)
        p_neighbor_m = unpad_from_window(p_neighbor_m.view(b * num_neighbors, 1, hp, wp), pad_hw)
        p_neighbor_i = unpad_from_window(p_neighbor_i.view(b * num_neighbors, 1, hp, wp), pad_hw)

        f_neighbors_m = f_neighbors_m.view(b, num_neighbors, c, h, w)
        f_neighbors_i = f_neighbors_i.view(b, num_neighbors, c, h, w)
        p_neighbor_m = p_neighbor_m.view(b, num_neighbors, 1, h, w)
        p_neighbor_i = p_neighbor_i.view(b, num_neighbors, 1, h, w)

        return {
            "f_t_c": unpad_from_window(f_center_c, pad_hw),
            "f_t_m": f_center_m,
            "f_t_i": f_center_i,
            "f_omega_m": f_neighbors_m,
            "f_omega_i": f_neighbors_i,
            "P_t_m": p_center_m,
            "P_t_i": p_center_i,
            "P_omega_m": p_neighbor_m,
            "P_omega_i": p_neighbor_i,
            "C_t_omega": corr,
            "H_t": entropy,
        }
