"""
GolfNet: Golf 版低光视频增强网络
=================================
相比 Foxtrot 的核心改动:

1. [棋盘格修复] 三分支上采样全部替换为 resize-conv (UpsampleBlock)
   - Foxtrot: Conv2d(C,C*4,1×1) → PixelShuffle(2) → 2px棋盘格
   - Golf:    bilinear×2 → 3×3conv → GELU → 3×3conv → 无格伪影

2. [F1 skip 接入] 编码器全分辨率特征 F1 直连三分支上采样
   - F1 携带原始分辨率边缘/纹理信息
   - 补偿 TCA 在 H/2 尺度工作导致的高频信息丢失
   - 三分支独立接收各自 F1_center (中心帧 F1)

3. [权重网络加深] Fusion 权重网络从 2 层增至 4 层
   - 更精细的空间自适应分支选择

4. [残差 gamma 上界] 0.5 → 0.9
   - Foxtrot sigmoid 限制过强, 限制了中心帧细节贡献

其余结构 (TCA-RWKV, 三查询解耦, NAFBlock, Retinex, 光流对齐) 不变.
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


class GolfNet(nn.Module):
    """GolfNet — Foxtrot 棋盘格修复 + F1 skip 版本

    Args:
        in_channels:      输入通道 (默认3)
        num_frames:       时序窗口 (默认5)
        encoder_channels: 编码器三层通道 [C1, C2, C3] (默认[32,64,128])
        tca_channels:     TCA 输出通道 (默认128)
        tca_heads:        TCA head 数 (默认4)
        tca_num_blocks:   TCA block 层数 (默认6)
        branch_blocks:    各分支 NAFBlock 数 [N,L,M] (默认[3,2,3])
        fusion_blocks:    融合精化 block 数 (默认2)
        gamma:            Retinex gamma 初始值 (默认2.0)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_frames: int = 5,
        encoder_channels: List[int] = None,
        tca_channels: int = 128,
        tca_heads: int = 4,
        tca_num_blocks: int = 6,
        branch_blocks: List[int] = None,
        fusion_blocks: int = 2,
        gamma: float = 2.0,
        use_f3_film: bool = True,       # R3-C
        use_hires_warp: bool = True,    # R3-B
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [32, 64, 128]
        if branch_blocks is None:
            branch_blocks = [3, 2, 3]

        self.in_channels = in_channels
        self.num_frames = num_frames
        self.center_idx = num_frames // 2
        self.use_f3_film = use_f3_film
        C1, C2, C3 = encoder_channels

        # 1. Shared Encoder (输出三层金字塔)
        self.encoder = SharedEncoder(
            in_channels=in_channels,
            out_channels_list=encoder_channels,
        )

        # 2. TCA-RWKV (H/2 尺度, C2 通道)
        self.tca = TCA_RWKV(
            channels=C2,
            tca_channels=tca_channels,
            num_heads=tca_heads,
            num_blocks=tca_num_blocks,
            num_frames=num_frames,
            f3_channels=C3,              # R3-C: F3 的通道数
            use_f3_film=use_f3_film,     # R3-C: FiLM 开关
        )

        # 3. 三分支 — 各接 F1 skip
        self.branch_n = BranchN(
            channels=tca_channels,
            num_blocks=branch_blocks[0],
            skip_channels=C1,
        )
        self.branch_l = BranchL(
            channels=tca_channels,
            num_blocks=branch_blocks[1],
            gamma_init=gamma,
            skip_channels=C1,
        )
        self.branch_m = BranchM(
            channels=tca_channels,
            enc_channels=C2,
            num_frames=num_frames,
            num_blocks=branch_blocks[2],
            skip_channels=C1,
            use_hires_warp=use_hires_warp,  # R3-B
        )

        # 4. Adaptive Fusion
        self.fusion = AdaptiveFusion(
            in_channels=in_channels,
            hidden_channels=64,
            num_blocks=fusion_blocks,
        )

    def forward(self, x: torch.Tensor, return_intermediate: bool = True,
                phase: str = None, frame_indices=None, **_) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            dict 含 "res_t" (兼容 tiled_forward) + 中间态
        """
        B, T, C, H, W = x.shape
        assert T == self.num_frames

        X_t = x[:, self.center_idx]  # (B,3,H,W) 中心帧

        # ── Stage 1: 编码所有帧 ──────────────────────────────────────
        F1_seq, F2_seq, F3_seq = [], [], []
        for t in range(T):
            f1, f2, f3 = self.encoder(x[:, t])
            F1_seq.append(f1)
            F2_seq.append(f2)
            F3_seq.append(f3)

        F1_seq = torch.stack(F1_seq, dim=1)  # (B,T,C1,H,W)
        F2_seq = torch.stack(F2_seq, dim=1)  # (B,T,C2,H/2,W/2)
        F3_seq = torch.stack(F3_seq, dim=1)  # (B,T,C3,H/4,W/4)

        # 中心帧编码特征
        F1_center = F1_seq[:, self.center_idx]  # (B,C1,H,W) ← skip 源
        F2_center = F2_seq[:, self.center_idx]  # (B,C2,H/2,W/2)

        # R3-C: F3 全局描述子 — 时序均值再空间全局 avg pool → (B, C3)
        f3_ctx = None
        if self.use_f3_film:
            f3_ctx = F3_seq.mean(dim=1)                    # (B,C3,H/4,W/4) 时序均值
            f3_ctx = f3_ctx.mean(dim=[-2, -1])             # (B,C3) 空间全局 pool

        # ── Stage 2: TCA-RWKV ─────────────────────────────────────────
        tca_out = self.tca(F2_center, F2_seq, f3_ctx=f3_ctx)  # R3-C 传入 f3_ctx
        F_N = tca_out["F_N"]
        F_L = tca_out["F_L"]
        F_M = tca_out["F_M"]
        var_map = tca_out["var_map"]
        ortho_loss = tca_out["ortho_loss"]

        # ── Stage 3: 三分支 (各接 F1_center skip) ────────────────────
        out_n = self.branch_n(F_N, var_map, skip=F1_center)
        Y_N = out_n["Y_N"]
        sigma_map = out_n["sigma_map"]

        out_l = self.branch_l(F_L, X_t, skip=F1_center)
        Y_L = out_l["Y_L"]
        L_t = out_l["L_t"]
        R_t = out_l["R_t"]

        out_m = self.branch_m(F_M, F2_seq, skip=F1_center)
        Y_M = out_m["Y_M"]
        flow_vis = out_m["flow_vis"]
        conf_map = out_m["conf_map"]

        # ── Stage 4: Adaptive Fusion ──────────────────────────────────
        fuse_out = self.fusion(Y_N, Y_L, Y_M, X_t)
        O_t = fuse_out["O_t"]

        output = {"O_t": O_t, "res_t": O_t, "image_center": X_t}

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
                "fusion_weights": fuse_out["weights"],
                "Y_fused": fuse_out["Y_fused"],
                "ortho_loss": ortho_loss,
                "var_map": var_map,
            })

        return output
