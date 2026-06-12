import torch
import torch.nn as nn


class FinalReconstruction(nn.Module):
    def __init__(self, channels=48, out_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, out_channels, 3, 1, 1)

    def forward(self, image_center, hat_f_t_i, hat_f_t_m, p_t_i, p_t_m):
        f_t_rec = p_t_i * hat_f_t_i + p_t_m * hat_f_t_m
        u_t = self.act(self.conv1(f_t_rec))
        delta = self.conv2(u_t)
        res_t = torch.clamp(image_center + delta, 0.0, 1.0)
        return {
            "f_t_rec": f_t_rec,
            "u_t": u_t,
            "delta": delta,
            "res_t": res_t,
        }

