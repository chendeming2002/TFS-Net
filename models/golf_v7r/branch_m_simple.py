#!/usr/bin/env python3
"""
Branch-M (简化版): 运动补偿

与 Golf R2 的差异:
  - FlowEstimator 只用中心帧 ±1 两帧 (而非全部 4 相邻帧)
  - 2×NAFBlock (vs 3)
  - Upsample3x3 (vs UpsampleBlock + F1 skip)
  - 保留 confidence gating (已验证有效) ⭐
  - 保留 flow-based deformable alignment ⭐

Input:  F_M     [B, 128, H/2, W/2]          (矩阵 RWKV 注入的运动上下文)
        F2_seq  [B, T, 64, H/2, W/2]        (encoder 全部帧特征)
Output: Y_M, flow_vis, conf_map
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.blocks import NAFBlock, LayerNorm2d
from models.golf_v7r.upsample import Upsample3x3


class FlowEstimator(nn.Module):
    """轻量级光流估计 (特征域)"""

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.corr_channels = 32
        self.corr_proj = nn.Sequential(
            nn.Conv2d(in_channels * 2, self.corr_channels, 1, bias=True),
            nn.GELU(),
        )
        self.flow_decoder = nn.Sequential(
            nn.Conv2d(self.corr_channels, 64, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, 2, 3, 1, 1, bias=True),
        )
        # 零初始化 → 初始光流为 0
        nn.init.zeros_(self.flow_decoder[-1].weight)
        nn.init.zeros_(self.flow_decoder[-1].bias)

    def forward(self, f_center, f_neigh):
        corr = self.corr_proj(torch.cat([f_center, f_neigh], dim=1))
        return self.flow_decoder(corr)


class DeformableAlign(nn.Module):
    """基于 flow field 的可变形采样"""

    def forward(self, feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feat.device),
            torch.linspace(-1, 1, W, device=feat.device),
            indexing='ij',
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        flow_perm = flow.permute(0, 2, 3, 1)
        scale = torch.tensor([W / 2.0, H / 2.0], device=flow.device, dtype=flow.dtype)
        sample_grid = grid + flow_perm / scale
        return F.grid_sample(feat, sample_grid, mode='bilinear',
                             padding_mode='border', align_corners=True)


class ConfidenceEstimator(nn.Module):
    """遮挡/对齐失败置信度检测"""

    def __init__(self, channels: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 3, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        # bias=2.0 → 初始置信度 ≈ 0.88 (默认信任相邻帧)
        nn.init.zeros_(self.conv[-2].weight)
        nn.init.constant_(self.conv[-2].bias, 2.0)

    def forward(self, warped, center):
        return self.conv(torch.cat([warped, center], dim=1))


class BranchMSimple(nn.Module):
    """Branch-M (简化): Motion Compensation, 仅用 ±1 两帧"""

    def __init__(self, channels: int = 128, enc_channels: int = 64,
                 num_frames: int = 5, num_blocks: int = 2,
                 out_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.center_idx = num_frames // 2

        # 只用中心帧两侧的相邻帧 (t-1, t+1)
        self.neighbor_offsets = [-1, 1]

        self.neigh_proj = nn.Sequential(
            nn.Conv2d(enc_channels, channels, 1, bias=True),
            LayerNorm2d(channels),
        )
        self.flow_estimator = FlowEstimator(channels)
        self.deform_align = DeformableAlign()
        self.conf_estimator = ConfidenceEstimator(channels)

        # fusion: center + 2 邻居 = 3 × channels
        num_inputs = 1 + len(self.neighbor_offsets)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * num_inputs, channels, 1, bias=True),
            LayerNorm2d(channels),
        )
        self.refine_blocks = nn.Sequential(*[
            NAFBlock(channels) for _ in range(num_blocks)
        ])
        self.upsample = Upsample3x3(in_ch=channels, out_ch=64, scale=2)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, out_channels, 3, 1, 1, bias=True),
        )

    def _align_and_aggregate(self, F_M, F2_seq):
        B, T, C_enc, H, W = F2_seq.shape
        center = F_M
        aligned_list = [center]
        flow_list, conf_list = [], []

        for off in self.neighbor_offsets:
            t = self.center_idx + off
            if t < 0 or t >= T:
                continue
            neigh = self.neigh_proj(F2_seq[:, t])
            flow = self.flow_estimator(center, neigh)
            flow_list.append(flow)
            warped = self.deform_align(neigh, flow)
            conf = self.conf_estimator(warped, center)
            conf_list.append(conf)
            aligned_list.append(warped * conf)

        aggregated = self.fusion_conv(torch.cat(aligned_list, dim=1))
        flow_center = (torch.stack(flow_list, dim=1).mean(dim=1)
                       if flow_list else torch.zeros(B, 2, H, W, device=F_M.device))
        conf_avg = (torch.stack(conf_list, dim=1).mean(dim=1)
                    if conf_list else torch.ones(B, 1, H, W, device=F_M.device))
        return aggregated, flow_center, conf_avg

    def forward(self, F_M: torch.Tensor, F2_seq: torch.Tensor) -> dict:
        aggregated, flow_vis, conf_map = self._align_and_aggregate(F_M, F2_seq)
        x = self.refine_blocks(aggregated)
        x = self.upsample(x)
        Y_M = self.to_rgb(x)
        return {"Y_M": Y_M, "flow_vis": flow_vis, "conf_map": conf_map}


if __name__ == '__main__':
    m = BranchMSimple(channels=128, enc_channels=64, num_frames=5, num_blocks=2)
    F_M = torch.randn(2, 128, 32, 32)
    F2_seq = torch.randn(2, 5, 64, 32, 32)
    out = m(F_M, F2_seq)
    print(f"BranchMSimple: F_M={F_M.shape}, F2_seq={F2_seq.shape}")
    print(f"  Y_M={out['Y_M'].shape}, flow={out['flow_vis'].shape}, conf={out['conf_map'].shape}")
    print(f"  params={sum(p.numel() for p in m.parameters())/1e3:.1f}K")
