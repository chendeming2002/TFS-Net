"""
RWKV-Only LLVE — No-TCA ablation baseline
==========================================
Encoder(l1/l2/l3) → 中心帧三尺度融合 → head → res_t
无 TCA、无运动门控、纯单帧多尺度增强（方案 5 baseline）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.encoder import PyramidEncoder


class RWKVOnlyNoTCA(nn.Module):
    def __init__(self, in_channels=3, level_channels=(32, 64, 96),
                 fused_channels=64):
        super().__init__()
        self.encoder = PyramidEncoder(
            in_channels=in_channels,
            level_channels=level_channels,
            fused_channels=fused_channels,
            num_bottleneck_blocks=0,
        )

        self.proj_l1 = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, 3, 1, 1), nn.GELU())
        self.proj_l2 = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, 3, 1, 1), nn.GELU())
        self.proj_l3 = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, 3, 1, 1), nn.GELU())

        self.head = nn.Sequential(
            nn.Conv2d(fused_channels * 3, fused_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(fused_channels * 2, fused_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(fused_channels, 3, 3, 1, 1),
        )

        self.hf_alpha = nn.Parameter(torch.tensor(0.05))

    def forward(self, x, phase='phase2'):
        B, T, C, H, W = x.shape
        center_idx = T // 2
        img_center = x[:, center_idx]

        x_flat = x.reshape(B * T, C, H, W)
        l1_flat, l2_flat, l3_flat = self.encoder.forward_single_lateral(x_flat)
        l1_lat = l1_flat.reshape(B, T, -1, H,      W)
        l2_lat = l2_flat.reshape(B, T, -1, H // 2, W // 2)
        l3_lat = l3_flat.reshape(B, T, -1, H // 4, W // 4)

        f1 = self.proj_l1(l1_lat[:, center_idx])
        f2 = self.proj_l2(l2_lat[:, center_idx])
        f3 = self.proj_l3(l3_lat[:, center_idx])

        f2_up = F.interpolate(f2, size=(H, W), mode='bilinear', align_corners=False)
        f3_up = F.interpolate(f3, size=(H, W), mode='bilinear', align_corners=False)

        fused = torch.cat([f1, f2_up, f3_up], dim=1)
        res_t = self.head(fused)

        hf = img_center - F.avg_pool2d(img_center, 5, 1, 2)
        res_t = res_t + self.hf_alpha * hf

        if not self.training:
            res_t = torch.clamp(res_t, 0, 1)

        return {"res_t": res_t}
