import torch
import torch.nn as nn

from .blocks import ResBlock, pad_to_window, unpad_from_window, window_partition_2d, window_partition_video, window_reverse_2d


class MSPN(nn.Module):
    def __init__(self, channels=48, window_size=8):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.gate = nn.Conv2d(channels * 2, channels, 1, 1, 0)
        self.refine = ResBlock(channels)

    def _aggregate_neighbors(self, corr, f_omega_m):
        b, t, c, h, w = f_omega_m.shape
        feat = f_omega_m.view(b * t, c, h, w)
        feat, pad_hw = pad_to_window(feat, self.window_size)
        hp, wp = feat.shape[-2:]
        feat = feat.view(b, t, c, hp, wp)
        feat_windows = window_partition_video(feat, self.window_size)
        aligned_windows = torch.matmul(corr, feat_windows)
        aligned = window_reverse_2d(aligned_windows, self.window_size, hp, wp)
        aligned = unpad_from_window(aligned, pad_hw)
        return aligned

    def forward(self, f_t_m, f_omega_m, corr):
        f_omega_aligned = self._aggregate_neighbors(corr, f_omega_m)
        z_t_m = torch.cat([f_t_m, f_omega_aligned], dim=1)
        g_t_m = torch.sigmoid(self.gate(z_t_m))
        f_t_m_fuse = g_t_m * f_t_m + (1.0 - g_t_m) * f_omega_aligned
        hat_f_t_m = self.refine(f_t_m_fuse) + f_t_m
        return {
            "f_omega_aligned": f_omega_aligned,
            "z_t_m": z_t_m,
            "G_t_m": g_t_m,
            "f_t_m_fuse": f_t_m_fuse,
            "hat_f_t_m": hat_f_t_m,
        }

