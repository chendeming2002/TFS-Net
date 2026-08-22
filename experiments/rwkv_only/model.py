"""
RWKV-Only LLVE — 三尺度融合验证模型
======================================
Encoder(l1/l2/l3) → TCA(l2) + l1/l3直连 → 三尺度合并 → head → res_t

数据流:
  Encoder → l1(H), l2(H/2), l3(H/4)
    l2 → TCA [HaarDWT + WKV + C_omega] → F_aligned(H/2) → up→H
    l3(H/4) → Conv → up→H/2 → up→H
    l1(H) → Conv → H
  concat[l1_feat, TCA↑, l3↑] → Conv→3ch → res_t
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.encoder import PyramidEncoder
from models.modules.pure_rwkv_sace import TCA


class RWKVOnlyLLVE(nn.Module):
    def __init__(self, in_channels=3, level_channels=(32, 64, 96),
                 fused_channels=64):
        super().__init__()
        self.encoder = PyramidEncoder(
            in_channels=in_channels,
            level_channels=level_channels,
            fused_channels=fused_channels,
            num_bottleneck_blocks=0,
        )
        self.tca = TCA(channels=fused_channels)

        # Light per-scale projectors
        self.proj_l1 = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, 3, 1, 1),
            nn.GELU(),
        )
        self.proj_l3 = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, 3, 1, 1),
            nn.GELU(),
        )

        # Fusion + decode head (3 channels × fused_channels → 3ch RGB)
        self.head = nn.Sequential(
            nn.Conv2d(fused_channels * 3, fused_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(fused_channels * 2, fused_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(fused_channels, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, phase='phase2'):
        B, T, C, H, W = x.shape

        # Encoder: all three scales
        x_flat = x.reshape(B * T, C, H, W)
        l1_flat, l2_flat, l3_flat = self.encoder.forward_single_lateral(x_flat)
        l1_lat = l1_flat.reshape(B, T, -1, H,      W)
        l2_lat = l2_flat.reshape(B, T, -1, H // 2, W // 2)
        l3_lat = l3_flat.reshape(B, T, -1, H // 4, W // 4)

        # TCA on l2 (H/2) — RWKV backbone
        tca_out = self.tca(l2_lat)
        tca_feat = tca_out["F_t_aligned"]  # (B, 64, H/2, W/2)
        tca_up = F.interpolate(tca_feat, size=(H, W), mode='bilinear',
                               align_corners=False)

        # l1 (H) — fine detail, direct
        l1_c = l1_lat[:, T // 2]  # center frame (B, 64, H, W)
        l1_feat = self.proj_l1(l1_c)

        # l3 (H/4) — global context → up H/2 → up H
        l3_c = l3_lat[:, T // 2]  # (B, 64, H/4, W/4)
        l3_feat = self.proj_l3(l3_c)
        l3_up = F.interpolate(l3_feat, size=(H, W), mode='bilinear',
                              align_corners=False)

        # Fuse three scales → decode to RGB
        fused = torch.cat([l1_feat, tca_up, l3_up], dim=1)  # (B, 192, H, W)
        res_t = self.head(fused)  # (B, 3, H, W)

        return {"res_t": res_t}
