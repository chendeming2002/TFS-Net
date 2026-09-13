"""
Branch-M: Motion Compensation Branch
======================================
处理运动伪影 (Type IV: 帧间结构位移 + 遮挡)

设计原则:
  1. 特征域光流估计 (轻量级 coarse-to-fine)
  2. Deformable 对齐 + 置信度门控 (遮挡感知)
  3. 多帧聚合 + 残差融合
  4. 与 Branch-L 解耦 (M 处理几何位移，L 处理光度变化)

输入: F_M (TCA 解耦的运动分量) + F1/F2/F3 (多尺度编码特征)
输出: Y_M (运动补偿后的图像)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.blocks import NAFBlock, LayerNorm2d
from typing import List, Tuple


class FlowEstimator(nn.Module):
    """轻量级光流估计网络 (特征域，coarse-to-fine)
    
    输入两帧特征，输出 2-channel flow field (dx, dy)
    """
    
    def __init__(self, in_channels: int = 128):
        super().__init__()
        # Cost volume correlation
        self.corr_channels = 32
        self.corr_proj = nn.Sequential(
            nn.Conv2d(in_channels * 2, self.corr_channels, 1, bias=True),
            nn.GELU(),
        )
        
        # Flow decoder
        self.flow_decoder = nn.Sequential(
            nn.Conv2d(self.corr_channels, 64, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, 2, 3, 1, 1, bias=True),  # 2-channel (dx, dy)
        )
        # 零初始化 (初始无位移)
        nn.init.zeros_(self.flow_decoder[-1].weight)
        nn.init.zeros_(self.flow_decoder[-1].bias)
    
    def forward(self, f_center: torch.Tensor, f_neigh: torch.Tensor) -> torch.Tensor:
        """
        f_center, f_neigh: (B, C, H, W)
        Returns: flow (B, 2, H, W) — (dx, dy) 位移场
        """
        # 拼接做 cost volume
        corr = torch.cat([f_center, f_neigh], dim=1)
        corr = self.corr_proj(corr)
        flow = self.flow_decoder(corr)
        return flow


class DeformableAlign(nn.Module):
    """Deformable 对齐模块 (基于 flow field 的可变形采样)
    
    使用 grid_sample 实现 differentiable warping
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, C, H, W) — 待对齐的特征
        flow: (B, 2, H, W) — 位移场 (dx, dy)
        
        Returns: warped_feat (B, C, H, W)
        """
        B, C, H, W = feat.shape
        
        # 构造采样 grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feat.device),
            torch.linspace(-1, 1, W, device=feat.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        grid = grid.repeat(B, 1, 1, 1)
        
        # flow 归一化到 [-1, 1] (out-of-place, 保证 autograd 安全)
        flow_perm = flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
        scale = torch.tensor([W / 2.0, H / 2.0], device=flow.device, dtype=flow.dtype)
        flow_norm = flow_perm / scale
        
        # 采样 grid + flow
        sample_grid = grid + flow_norm
        warped = F.grid_sample(feat, sample_grid, mode='bilinear',
                               padding_mode='border', align_corners=True)
        return warped


class ConfidenceEstimator(nn.Module):
    """置信度估计 (遮挡/对齐失败检测)
    
    输入 warped 特征 + 中心帧特征，输出置信度图
    """
    
    def __init__(self, channels: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 3, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        # 初始化为高置信度 (大部分区域对齐成功)
        nn.init.zeros_(self.conv[-2].weight)
        nn.init.constant_(self.conv[-2].bias, 2.0)
    
    def forward(self, warped: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """
        Returns: conf (B, 1, H, W) ∈ [0, 1]
        """
        x = torch.cat([warped, center], dim=1)
        return self.conv(x)


class BranchM(nn.Module):
    """Branch-M: Motion Compensation
    
    Args:
        channels: 输入特征通道数 (默认128)
        num_frames: 时序窗口大小 (默认5)
        num_blocks: 聚合后的细化 block 数 (默认3)
        out_channels: 输出通道数 (默认3)
    
    Input:
        F_M: (B, C, H/2, W/2) — TCA 解耦的运动分量 (中心帧)
        F_M_seq: (B, T, C, H/2, W/2) — 全时序特征 (用于对齐聚合)
        pyramid_feats: List[(B, C_i, H_i, W_i)] — 多尺度编码特征 (可选，用于多尺度对齐)
    
    Output:
        Y_M: (B, 3, H, W) — 运动补偿后的图像
        flow_vis: (B, 2, H/2, W/2) — 中心帧光流 (供可视化/损失)
        conf_map: (B, 1, H/2, W/2) — 置信度图
    """
    
    def __init__(self, channels: int = 128, num_frames: int = 5,
                 num_blocks: int = 3, out_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.center_idx = num_frames // 2
        
        # 光流估计器
        self.flow_estimator = FlowEstimator(channels)
        
        # Deformable 对齐
        self.deform_align = DeformableAlign()
        
        # 置信度估计
        self.conf_estimator = ConfidenceEstimator(channels)
        
        # 多帧聚合融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * num_frames, channels, 1, bias=True),
            LayerNorm2d(channels),
        )
        
        # 运动细化 blocks
        self.refine_blocks = nn.Sequential(*[
            NAFBlock(channels) for _ in range(num_blocks)
        ])
        
        # 上采样到原分辨率
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1, bias=True),
            nn.PixelShuffle(2),
            LayerNorm2d(channels),
        )
        
        # 输出投影
        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels // 2, out_channels, 3, 1, 1, bias=True),
        )
    
    def _align_and_aggregate(self, F_M: torch.Tensor,
                             F_M_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对齐并聚合邻帧
        
        Returns:
            aggregated: (B, C, H, W) — 聚合后的特征
            flow_center: (B, 2, H, W) — 中心帧的平均光流 (可视化用)
            conf_avg: (B, 1, H, W) — 平均置信度
        """
        B, T, C, H, W = F_M_seq.shape
        center = F_M  # (B, C, H, W)
        
        aligned_list = [center]  # 中心帧不需要对齐
        flow_list = []
        conf_list = []
        
        for t in range(T):
            if t == self.center_idx:
                continue
            
            neigh = F_M_seq[:, t]  # (B, C, H, W)
            
            # 光流估计
            flow = self.flow_estimator(center, neigh)  # (B, 2, H, W)
            flow_list.append(flow)
            
            # Deformable 对齐
            warped = self.deform_align(neigh, flow)
            
            # 置信度估计
            conf = self.conf_estimator(warped, center)
            conf_list.append(conf)
            
            # 置信度门控
            warped_gated = warped * conf
            aligned_list.append(warped_gated)
        
        # 聚合
        aggregated_cat = torch.cat(aligned_list, dim=1)  # (B, C*T, H, W)
        aggregated = self.fusion_conv(aggregated_cat)
        
        # 平均光流和置信度 (用于监督/可视化)
        flow_center = torch.stack(flow_list, dim=1).mean(dim=1) if flow_list else torch.zeros(B, 2, H, W, device=F_M.device)
        conf_avg = torch.stack(conf_list, dim=1).mean(dim=1) if conf_list else torch.ones(B, 1, H, W, device=F_M.device)
        
        return aggregated, flow_center, conf_avg
    
    def forward(self, F_M: torch.Tensor, F_M_seq: torch.Tensor,
                pyramid_feats: List[torch.Tensor] = None) -> dict:
        """
        F_M: (B, C, H/2, W/2)
        F_M_seq: (B, T, C, H/2, W/2)
        pyramid_feats: 暂时未使用，预留多尺度对齐接口
        
        Returns dict: Y_M, flow_vis, conf_map
        """
        # 对齐并聚合
        aggregated, flow_vis, conf_map = self._align_and_aggregate(F_M, F_M_seq)
        
        # 细化
        x = self.refine_blocks(aggregated)
        
        # 上采样到原分辨率
        x = self.upsample(x)  # (B, C, H, W)
        
        # 输出 RGB
        Y_M = self.to_rgb(x)  # (B, 3, H, W)
        
        return {
            "Y_M": Y_M,
            "flow_vis": flow_vis,
            "conf_map": conf_map,
        }
