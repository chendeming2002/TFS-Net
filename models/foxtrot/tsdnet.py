"""
TSDNet: Tripartite Structured Decomposition Network (Foxtrot)
==============================================================
三分支结构化分解低光增强网络 (TSD-Foxtort.md 实现)

架构概览:
  Input: X (B, T, 3, H, W)
  ┌─────────────────────────────────────────────────┐
  │ 1. Shared Encoder (3-scale pyramid)             │
  │    F1, F2, F3 = Encoder(X_seq)                  │
  └─────────────────────────────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────┐
  │ 2. TCA-RWKV (temporal context aggregation)      │
  │    F_N, F_L, F_M = TCA_RWKV(F2, F_seq)          │
  │    (三查询结构化解耦)                             │
  └─────────────────────────────────────────────────┘
                           ↓
  ┌──────────────┬──────────────┬──────────────────┐
  │  Branch-N    │  Branch-L    │  Branch-M        │
  │  (噪声)      │  (光照)      │  (运动)           │
  │  Y_N, σ²     │  Y_L, L, R   │  Y_M, flow, conf │
  └──────────────┴──────────────┴──────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────┐
  │ 3. Adaptive Fusion                               │
  │    Ô_t = Fusion(Y_N, Y_L, Y_M, X_t)             │
  └─────────────────────────────────────────────────┘

关键设计决策:
  - 编码器共享: 降低参数量，统一特征表达
  - TCA-RWKV: 替代传统 Transformer，线性复杂度 + 结构化三查询
  - 分支独立: 噪声/光照/运动物理解耦，各自监督
  - 自适应融合: 空间变化权重，无先验假设
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .encoder import SharedEncoder
from .tca_rwkv import TCA_RWKV
from .branch_n import BranchN
from .branch_l import BranchL
from .branch_m import BranchM
from .fusion import AdaptiveFusion


class TSDNet(nn.Module):
    """TSDNet: Tripartite Structured Decomposition Network
    
    Args:
        in_channels: 输入通道 (默认3 for RGB)
        num_frames: 时序窗口 (默认5)
        encoder_channels: 编码器通道数 [32, 64, 128] for 3 scales
        tca_channels: TCA 输出通道数 (默认128)
        tca_heads: TCA head 数 (默认4)
        tca_num_blocks: TCA block 层数 (默认6)
        branch_blocks: 各分支的 NAFBlock 数 (默认 [2, 2, 3] for N/L/M)
        fusion_blocks: 融合后精化 block 数 (默认2)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_frames: int = 5,
        encoder_channels: List[int] = [32, 64, 128],
        tca_channels: int = 128,
        tca_heads: int = 4,
        tca_num_blocks: int = 6,
        branch_blocks: List[int] = [2, 2, 3],
        fusion_blocks: int = 2,
        ema_alpha: float = 0.7,
        gamma: float = 2.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.center_idx = num_frames // 2
        
        # 1. Shared Encoder
        self.encoder = SharedEncoder(
            in_channels=in_channels,
            out_channels_list=encoder_channels,
        )
        
        # 2. TCA-RWKV (在 F2 尺度，channels=encoder_channels[1])
        self.tca = TCA_RWKV(
            channels=encoder_channels[1],
            tca_channels=tca_channels,
            num_heads=tca_heads,
            num_blocks=tca_num_blocks,
            num_frames=num_frames,
        )
        
        # 3. 三分支
        self.branch_n = BranchN(
            channels=tca_channels,
            num_blocks=branch_blocks[0],
        )
        
        self.branch_l = BranchL(
            channels=tca_channels,
            num_blocks=branch_blocks[1],
            ema_alpha=ema_alpha,
            gamma=gamma,
        )
        
        self.branch_m = BranchM(
            channels=tca_channels,
            num_frames=num_frames,
            num_blocks=branch_blocks[2],
        )
        
        # 4. Adaptive Fusion
        self.fusion = AdaptiveFusion(
            in_channels=in_channels,
            hidden_channels=64,
            num_blocks=fusion_blocks,
        )
    
    def forward(self, x: torch.Tensor, return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, C, H, W) — 输入视频序列
            return_intermediate: 是否返回中间结果 (用于损失计算)
        
        Returns:
            dict with keys:
              - O_t: (B, 3, H, W) — 最终输出 (中心帧增强结果)
              - [若 return_intermediate=True]:
                  Y_N, Y_L, Y_M: 各分支输出
                  sigma_map, L_t, R_t, flow_vis, conf_map: 辅助输出
                  fusion_weights: 融合权重
        """
        B, T, C, H, W = x.shape
        assert T == self.num_frames, f"Expected {self.num_frames} frames, got {T}"
        
        # 中心帧
        X_t = x[:, self.center_idx]  # (B, C, H, W)
        
        # ==================== Stage 1: Encoding ====================
        # 编码所有帧
        pyramid_list = []  # List of [(B, C1, H1, W1), (B, C2, H2, W2), (B, C3, H3, W3)]
        for t in range(T):
            F1, F2, F3 = self.encoder(x[:, t])
            pyramid_list.append((F1, F2, F3))
        
        # 提取各尺度时序特征
        F1_seq = torch.stack([p[0] for p in pyramid_list], dim=1)  # (B, T, C1, H, W)
        F2_seq = torch.stack([p[1] for p in pyramid_list], dim=1)  # (B, T, C2, H/2, W/2)
        F3_seq = torch.stack([p[2] for p in pyramid_list], dim=1)  # (B, T, C3, H/4, W/4)
        
        # 中心帧特征
        F2_center = F2_seq[:, self.center_idx]  # (B, C2, H/2, W/2)
        
        # ==================== Stage 2: TCA-RWKV ====================
        tca_output = self.tca(F2_center, F2_seq)
        F_N = tca_output["F_N"]  # (B, tca_channels, H/2, W/2)
        F_L = tca_output["F_L"]  # (B, tca_channels, H/2, W/2)
        F_M = tca_output["F_M"]  # (B, tca_channels, H/2, W/2)
        var_map = tca_output["var_map"]  # (B, 1, H/2, W/2)
        ortho_loss = tca_output["ortho_loss"]  # scalar
        
        # 将 F_seq 转换为 TCA 通道数 (用于分支时序处理)
        # 这里简化处理: 使用 F_N/F_L/F_M 的时序版本
        # 完整实现应对 F2_seq 全序列做 TCA，这里为简化只传中心帧特征
        F_N_seq = F_N.unsqueeze(1).repeat(1, T, 1, 1, 1)  # 简化: 复制中心帧
        F_L_seq = F_L.unsqueeze(1).repeat(1, T, 1, 1, 1)
        F_M_seq = F_M.unsqueeze(1).repeat(1, T, 1, 1, 1)
        
        # ==================== Stage 3: 三分支处理 ====================
        # Branch-N
        branch_n_out = self.branch_n(F_N, var_map)
        Y_N = branch_n_out["Y_N"]
        sigma_map = branch_n_out["sigma_map"]
        
        # Branch-L
        branch_l_out = self.branch_l(F_L, X_t, F_L_seq)
        Y_L = branch_l_out["Y_L"]
        L_t = branch_l_out["L_t"]
        R_t = branch_l_out["R_t"]
        
        # Branch-M
        branch_m_out = self.branch_m(F_M, F_M_seq)
        Y_M = branch_m_out["Y_M"]
        flow_vis = branch_m_out["flow_vis"]
        conf_map = branch_m_out["conf_map"]
        
        # ==================== Stage 4: Adaptive Fusion ====================
        fusion_out = self.fusion(Y_N, Y_L, Y_M, X_t)
        O_t = fusion_out["O_t"]
        fusion_weights = fusion_out["weights"]
        
        # ==================== Output ====================
        output = {"O_t": O_t}
        
        if return_intermediate:
            output.update({
                "Y_N": Y_N,
                "Y_L": Y_L,
                "Y_M": Y_M,
                "sigma_map": sigma_map,
                "L_t": L_t,
                "R_t": R_t,
                "flow_vis": flow_vis,
                "conf_map": conf_map,
                "fusion_weights": fusion_weights,
                "Y_fused": fusion_out["Y_fused"],
                "ortho_loss": ortho_loss,
            })
        
        return output


if __name__ == "__main__":
    # 简单测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TSDNet(
        in_channels=3,
        num_frames=5,
        encoder_channels=[32, 64, 128],
        tca_channels=128,
        tca_heads=4,
        tca_num_blocks=6,
    ).to(device)
    
    x = torch.randn(2, 5, 3, 256, 256).to(device)
    
    print("Testing forward pass...")
    with torch.no_grad():
        out = model(x, return_intermediate=True)
    
    print(f"O_t shape: {out['O_t'].shape}")
    print(f"Y_N shape: {out['Y_N'].shape}")
    print(f"Y_L shape: {out['Y_L'].shape}")
    print(f"Y_M shape: {out['Y_M'].shape}")
    print(f"sigma_map shape: {out['sigma_map'].shape}")
    print(f"L_t shape: {out['L_t'].shape}")
    print(f"flow_vis shape: {out['flow_vis'].shape}")
    print(f"fusion_weights shape: {out['fusion_weights'].shape}")
    print("Forward pass successful!")
    
    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
