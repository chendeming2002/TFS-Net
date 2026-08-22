"""
RWKV-Only LLVE v2 — 运动感知门控增强
=====================================
Encoder(l1/l2/l3) → TCA(l2) + C_omega→motion_map → 运动门控 + 三尺度融合 → head → res_t

v2 改进 (vs v1):
  - C_omega 对角线 → conf_proj → conf_map → motion_map = 1 - conf_map
  - 运动区域(motion→1): tca_gated ≈ l1_center (尖锐单帧)
  - 静止区域(motion→0): tca_gated ≈ tca_up (多帧降噪)
  - 无 gamma clamp, 无多 loss, 无梯度隔离
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.encoder import PyramidEncoder
from models.modules.pure_rwkv_sace import TCA


class RWKVOnlyV2(nn.Module):
    def __init__(self, in_channels=3, level_channels=(32, 64, 96),
                 fused_channels=64, num_frames=5):
        super().__init__()
        self.num_frames = num_frames
        self.fused_channels = fused_channels

        self.encoder = PyramidEncoder(
            in_channels=in_channels,
            level_channels=level_channels,
            fused_channels=fused_channels,
            num_bottleneck_blocks=0,
        )
        self.tca = TCA(channels=fused_channels)

        # C_omega confidence projector (from NDPN)
        self.conf_proj = nn.Sequential(
            nn.Linear(num_frames - 1, fused_channels // 4),
            nn.GELU(),
            nn.Linear(fused_channels // 4, 1),
            nn.Sigmoid(),
        )

        # l1 projector — two copies: one for head, one for tca gating
        self.proj_l1 = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, 3, 1, 1), nn.GELU())
        self.proj_l1_gate = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, 3, 1, 1), nn.GELU())

        self.proj_l3 = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, 3, 1, 1), nn.GELU())

        self.head = nn.Sequential(
            nn.Conv2d(fused_channels * 3, fused_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(fused_channels * 2, fused_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(fused_channels, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def _build_motion_map(self, C_omega_list, target_size, device):
        if not C_omega_list:
            return torch.zeros(1, 1, *target_size, device=device)

        B = C_omega_list[0].shape[0]
        diag_scores = []
        for C_t in C_omega_list:
            diag = C_t.diagonal(dim1=-2, dim2=-1)
            diag_scores.append(diag)
        diag_stack = torch.stack(diag_scores, dim=-1)  # (B, N, 4)

        conf_raw = self.conf_proj(diag_stack).squeeze(-1)  # (B, N)
        ds = int(conf_raw.shape[-1] ** 0.5)
        conf_map = conf_raw.reshape(B, 1, ds, ds)
        conf_map = F.interpolate(conf_map, size=target_size,
                                 mode='bilinear', align_corners=False)
        return 1.0 - conf_map  # motion_map

    def forward(self, x, phase='phase2'):
        B, T, C, H, W = x.shape
        center_idx = T // 2

        x_flat = x.reshape(B * T, C, H, W)
        l1_flat, l2_flat, l3_flat = self.encoder.forward_single_lateral(x_flat)
        l1_lat = l1_flat.reshape(B, T, -1, H,      W)
        l2_lat = l2_flat.reshape(B, T, -1, H // 2, W // 2)
        l3_lat = l3_flat.reshape(B, T, -1, H // 4, W // 4)

        tca_out = self.tca(l2_lat)
        tca_feat = tca_out["F_t_aligned"]
        tca_up = F.interpolate(tca_feat, size=(H, W), mode='bilinear',
                               align_corners=False)

        motion_map = self._build_motion_map(
            tca_out.get("C_omega_list", []), (H, W), x.device)

        l1_c = l1_lat[:, center_idx]
        l1_feat = self.proj_l1(l1_c)
        l1_gate_feat = self.proj_l1_gate(l1_c)

        # Motion-aware gating
        tca_gated = tca_up * (1 - motion_map) + l1_gate_feat * motion_map

        l3_c = l3_lat[:, center_idx]
        l3_feat = self.proj_l3(l3_c)
        l3_up = F.interpolate(l3_feat, size=(H, W), mode='bilinear',
                              align_corners=False)

        fused = torch.cat([l1_feat, tca_gated, l3_up], dim=1)
        res_t = self.head(fused)

        return {"res_t": res_t, "motion_map": motion_map}
