"""
RWKV-Only LLVE — Ablation 模型
===============================
基于 v2 架构，可单独开关三个组件:
  --ablation A: sigmoid移除 (方案3)
  --ablation B: HF残差 (方案4)
  --ablation C: conf_scale (方案2A)
  --ablation ABC: 三者全开 (等价 v3)
  --ablation none: 纯v2 (对照)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.encoder import PyramidEncoder
from models.modules.pure_rwkv_sace import TCA


class RWKVOnlyAblation(nn.Module):
    def __init__(self, in_channels=3, level_channels=(32, 64, 96),
                 fused_channels=64, num_frames=5,
                 remove_sigmoid=False, use_hf_residual=False, use_conf_scale=False):
        super().__init__()
        self.remove_sigmoid = remove_sigmoid
        self.use_hf_residual = use_hf_residual
        self.use_conf_scale = use_conf_scale
        self.num_frames = num_frames
        self.fused_channels = fused_channels

        self.encoder = PyramidEncoder(
            in_channels=in_channels,
            level_channels=level_channels,
            fused_channels=fused_channels,
            num_bottleneck_blocks=0,
        )
        self.tca = TCA(channels=fused_channels)

        if use_conf_scale:
            self.conf_proj = nn.Sequential(
                nn.Linear(num_frames - 1, fused_channels // 4),
                nn.GELU(),
                nn.Linear(fused_channels // 4, 1),
            )
            self.conf_scale = nn.Parameter(torch.tensor(3.0))
        else:
            self.conf_proj = nn.Sequential(
                nn.Linear(num_frames - 1, fused_channels // 4),
                nn.GELU(),
                nn.Linear(fused_channels // 4, 1),
                nn.Sigmoid(),
            )

        self.proj_l1 = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, 3, 1, 1), nn.GELU())
        self.proj_l1_gate = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, 3, 1, 1), nn.GELU())
        self.proj_l3 = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, 3, 1, 1), nn.GELU())

        head_layers = [
            nn.Conv2d(fused_channels * 3, fused_channels * 2, 3, 1, 1), nn.GELU(),
            nn.Conv2d(fused_channels * 2, fused_channels, 3, 1, 1), nn.GELU(),
            nn.Conv2d(fused_channels, 3, 3, 1, 1),
        ]
        if not remove_sigmoid:
            head_layers.append(nn.Sigmoid())
        self.head = nn.Sequential(*head_layers)

        if use_hf_residual:
            self.hf_alpha = nn.Parameter(torch.tensor(0.05))

    def _build_motion_map(self, C_omega_list, target_size, device):
        if not C_omega_list:
            return torch.zeros(1, 1, *target_size, device=device)

        B = C_omega_list[0].shape[0]
        diag_scores = []
        for C_t in C_omega_list:
            diag = C_t.diagonal(dim1=-2, dim2=-1)
            diag_scores.append(diag)
        diag_stack = torch.stack(diag_scores, dim=-1)

        if self.use_conf_scale:
            conf_raw = self.conf_proj(diag_stack).squeeze(-1)
            conf_map = torch.sigmoid(conf_raw * self.conf_scale)
        else:
            conf_map = self.conf_proj(diag_stack).squeeze(-1)

        ds = int(conf_map.shape[-1] ** 0.5)
        conf_map = conf_map.reshape(B, 1, ds, ds)
        conf_map = F.interpolate(conf_map, size=target_size,
                                 mode='bilinear', align_corners=False)
        return 1.0 - conf_map

    def forward(self, x, phase='phase2'):
        B, T, C, H, W = x.shape
        center_idx = T // 2
        img_center = x[:, center_idx]

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

        tca_gated = tca_up * (1 - motion_map) + l1_gate_feat * motion_map

        l3_c = l3_lat[:, center_idx]
        l3_feat = self.proj_l3(l3_c)
        l3_up = F.interpolate(l3_feat, size=(H, W), mode='bilinear',
                              align_corners=False)

        fused = torch.cat([l1_feat, tca_gated, l3_up], dim=1)
        res_t = self.head(fused)

        if self.use_hf_residual:
            hf = img_center - F.avg_pool2d(img_center, 5, 1, 2)
            res_t = res_t + self.hf_alpha * hf

        if self.remove_sigmoid and not self.training:
            res_t = torch.clamp(res_t, 0, 1)

        return {"res_t": res_t, "motion_map": motion_map}
