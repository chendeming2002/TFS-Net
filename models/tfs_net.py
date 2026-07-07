"""
TFS-Net v6 Delta Mark1 — Three-source Fusion & Synthesis Network (2026-07-01)
============================================================================
端到端多帧低光增强网络。

整体结构:
    Stage 0: PyramidEncoder → F_stack
    Stage 1: SWD (Spatial Wavelet Diverter) → feat_tfde + feat_tca
    Stage 2: TFDE (时频退化估计) → s_illum, s_noise
    Stage 3: TCA (时序对应对齐) → tca_out, C_omega_list, F_t_aligned
    Stage 4: ISPN/NDPN/MCPN 三源恢复
    Stage 5: CXG + SGRF → res_t
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.encoder import PyramidEncoder
from models.modules.tfsi_v2 import DPE
from models.modules.swd import WFR
from models.modules.pure_rwkv_sace import TCA
from models.modules.ispn_v2 import ISPN
from models.modules.ndpn import NDPN
from models.modules.mrpn import MCPN
from models.modules.igrf import SGRF
from models.modules.amp_enhance import AmpEnhance


class CXG(nn.Module):
    """CXG (Cross-eXcitation Gate): NDPN/MCPN 交叉激励门 (结构性重参数化)"""

    def __init__(self, channels: int, deploy: bool = False):
        super().__init__()
        self.deploy = deploy
        self.channels = channels

        if not deploy:
            self.gate_noise = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, 1),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.gate_noise[-2].weight)
            nn.init.ones_(self.gate_noise[-2].bias)
            self.gate_motion = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, 1),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.gate_motion[-2].weight)
            nn.init.ones_(self.gate_motion[-2].bias)
        else:
            self.scale_noise = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.scale_motion = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, f_noise: torch.Tensor, f_motion: torch.Tensor):
        if self.deploy:
            return f_noise * self.scale_noise, f_motion * self.scale_motion
        g_n = self.gate_noise(f_motion)
        g_m = self.gate_motion(f_noise)
        return f_noise * g_n, f_motion * g_m

    def get_deploy(self, avg_gate_n: torch.Tensor, avg_gate_m: torch.Tensor):
        deploy_mod = CXG(self.channels, deploy=True)
        deploy_mod.scale_noise.data = avg_gate_n.view(1, -1, 1, 1)
        deploy_mod.scale_motion.data = avg_gate_m.view(1, -1, 1, 1)
        return deploy_mod


class TFSNet(nn.Module):
    """
    Args:
        in_channels    : 输入图像通道 (默认 3)
        level_channels : 编码器三尺度通道 (默认 (32, 64, 96))
        fused_channels : 编码器融合输出通道 (默认 48)
        eps            : TFSI 数值稳定项
        n_groups       : SACE 可变形分组数
        kernel_size    : SACE 可变形核大小
        share_lff      : SACE 是否与 TFSI 共享 LFF (默认 True)
    """

    def __init__(
        self,
        in_channels: int = 3,
        level_channels: Tuple[int, ...] = (32, 64, 96),
        fused_channels: int = 64,
        eps: float = 1e-6,
        n_groups: int = 4,
        kernel_size: int = 3,
        share_lff: bool = True,
        sace_phase_preserving: bool = True,
        use_soft_clamp: bool = True,
        sace_offset_use_norm: bool = True,
        sace_offset_kaiming_init: bool = True,
        use_soft_median: bool = True,
        use_cross_rwkv: bool = False,
        use_dwt_lff: bool = False,
        use_pure_rwkv: bool = False,
        use_nafblock: bool = False,
        num_bottleneck_blocks: int = 0,
        num_igrf_res_blocks: int = 2,
        use_amp_enhance: bool = False,
        charlie_mode: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.fused_channels = fused_channels
        self.share_lff = share_lff
        # v5.5: 默认 3 级编码器, coarse_channels=96
        coarse_channels = level_channels[-1]  # 最粗层通道数

        # v6 Charlie: 数据流路径重改 (Charlie-plan P0)
        self.charlie_mode = charlie_mode

        # v5.9: AmpEnhance — 图像级频域幅度增强 (Encoder 前处理)
        self.use_amp_enhance = use_amp_enhance
        if use_amp_enhance:
            self.amp_enhance = AmpEnhance(in_channels=in_channels, hidden=16, min_amps=0.1)
        else:
            self.amp_enhance = None

        # Stage 0: PyramidEncoder
        self.encoder = PyramidEncoder(
            in_channels=in_channels,
            level_channels=level_channels,
            fused_channels=fused_channels,
            num_bottleneck_blocks=num_bottleneck_blocks,
        )

        # Mark1: ISPN 输入投影 (encoder 特征 → 3ch 图像)
        # Mark1: SWD 空域小波分流 (替代 DWT-LFF)
        self.wfr = WFR(channels=fused_channels, alpha_init=0.6)

        self.dpe = DPE(
            channels=fused_channels, fused_channels=fused_channels, eps=eps,
            use_soft_median=use_soft_median,
        )

        self.tca = TCA(channels=fused_channels)

        self.ispn = ISPN(channels=fused_channels, img_channels=in_channels)
        self.ndpn = NDPN(channels=fused_channels)
        self.mcpn = MCPN(channels=fused_channels)

        # Stage 4: CXG 交叉激励门
        self.cxg = CXG(channels=fused_channels)

        # Stage 5: SGRF 阶段式修复融合
        self.sgrf = SGRF(channels=fused_channels, out_channels=in_channels,
                         use_soft_clamp=use_soft_clamp, use_nafblock=use_nafblock,
                         num_res_blocks=num_igrf_res_blocks)

        # 逐帧特征缓存
        self.frame_cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def clear_frame_cache(self):
        """清空逐帧特征缓存（切换序列或释放显存时调用）。"""
        self.frame_cache.clear()

    def forward(
        self,
        x: torch.Tensor,
        frame_indices: Optional[List[int]] = None,
        phase: str = 'phase2',
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, 3, H, W) 多帧低光输入
            phase: 'phase1_warmup' | 'phase1' | 'phase1_5' | 'phase2' (Mark3)
        """
        self._phase = phase
        B, T, C_in, H, W = x.shape
        center_idx = T // 2
        image_center = x[:, center_idx]

        # Stage 0: Encoder → F_stack (B,T,C,H,W)
        feats_list: List[torch.Tensor] = []
        for i in range(T):
            f = self.encoder.forward_single(x[:, i], return_coarse=False)
            feats_list.append(f)
        feats = torch.stack(feats_list, dim=1)

        # Stage 1: SWD 小波分流 (逐帧 → B*T,C,H,W → DWT → feat_tfde/feat_tca)
        feats_flat = feats.reshape(B * T, -1, H, W)
        wfr_out = self.wfr(feats_flat)
        feat_tfde = wfr_out["feat_tfde"].reshape(B, T, -1, H // 2, W // 2)
        feat_tca = wfr_out["feat_tca"].reshape(B, T, -1, H // 2, W // 2)

        dpe_out = self.dpe(feat_tfde)
        s_illum = dpe_out["s_illum"]
        s_noise_orig = dpe_out["s_noise"]
        # 上采样到 H×W (NDPN/ISPN 需要全分辨率)
        s_illum = F.interpolate(s_illum, size=(H, W), mode='bilinear', align_corners=False)
        s_noise = F.interpolate(s_noise_orig, size=(H, W), mode='bilinear', align_corners=False)

        # Stage 3: TCA 时序对应对齐
        tca_out = self.tca(feat_tca, dpe_out)
        F_aligned_list = [tca_out["tca_out"][:, t] for t in range(T)]
        C_omega_list = tca_out.get("C_omega_list", [])
        F_t_aligned = tca_out["F_t_aligned"]
        mu_t_clean = tca_out["mu_t_clean"]
        sigma_t_clean = tca_out["sigma_t_clean"]

        aligned_feats = torch.stack(F_aligned_list, dim=1)

        f_enc_center = feats[:, center_idx]
        ispn_out = self.ispn(f_enc_center, s_illum)
        gain_map = ispn_out["gain_map"]
        bias_map = ispn_out["bias_map"]
        f_illum_feat = ispn_out["f_illum_feat"]

        # --- Mark3: Phase-dependent NDPN/MCPN/CXG ---
        phase = getattr(self, '_phase', 'phase2')
        if phase in ('phase1', 'phase1_warmup'):
            f_noise_out = torch.zeros_like(F_t_aligned)
            f_motion_out = torch.zeros_like(F_t_aligned)
            ndpn_out = {"f_noise_out": f_noise_out, "s_snr": torch.zeros(B, 1, H, W, device=x.device)}
            mcpn_out = {"f_motion_out": f_motion_out, "G_t": torch.zeros(B, 64, H, W, device=x.device)}
            f_noise_gated, f_motion_gated = f_noise_out, f_motion_out
        elif phase == 'phase1_5':
            unlock = getattr(self, '_unlock_ratio', 0.0)
            ndpn_out = self.ndpn(feats=feats, F_aligned_list=F_aligned_list, mu_t_clean=mu_t_clean,
                sigma_t_clean=sigma_t_clean, s_noise=s_noise, center_idx=center_idx,
                C_omega_list=C_omega_list, F_t_aligned=F_t_aligned)
            mcpn_out = self.mcpn(F_aligned_list=F_aligned_list, center_idx=center_idx,
                sigma_t_clean=sigma_t_clean, C_omega_list=C_omega_list, F_t_aligned=F_t_aligned)
            f_noise_out = ndpn_out["f_noise_out"] * unlock
            f_motion_out = mcpn_out["f_motion_out"] * unlock
            if unlock > 0.3:
                f_noise_gated, f_motion_gated = self.cxg(f_noise_out, f_motion_out)
            else:
                f_noise_gated, f_motion_gated = f_noise_out, f_motion_out
        else:  # phase2
            ndpn_out = self.ndpn(feats=feats, F_aligned_list=F_aligned_list, mu_t_clean=mu_t_clean,
                sigma_t_clean=sigma_t_clean, s_noise=s_noise, center_idx=center_idx,
                C_omega_list=C_omega_list, F_t_aligned=F_t_aligned)
            mcpn_out = self.mcpn(F_aligned_list=F_aligned_list, center_idx=center_idx,
                sigma_t_clean=sigma_t_clean, C_omega_list=C_omega_list, F_t_aligned=F_t_aligned)
            f_noise_gated, f_motion_gated = self.cxg(ndpn_out["f_noise_out"], mcpn_out["f_motion_out"])

        # SGRF: 阶段式修复融合 (Mark4: gain/bias interface)
        sgrf_out = self.sgrf(
            gain_map=gain_map,
            bias_map=bias_map,
            f_noise_out=ndpn_out["f_noise_out"],
            f_motion_out=mcpn_out["f_motion_out"],
            image_center=image_center,
        )

        return {
            "res_t":          sgrf_out["res_t"],
            "img_s1":         sgrf_out["img_s1"],
            "img_s2":         sgrf_out["img_s2"],
            "lit_up_map":     sgrf_out["lit_up_map"],
            "gain_map":       gain_map,
            "bias_map":       bias_map,
            "image_center":   image_center,
            "s_illum":        s_illum,
            "s_noise":        s_noise,
            "f_illum_feat":   f_illum_feat,
            "f_noise_out":    ndpn_out["f_noise_out"],
            "f_motion_out":   mcpn_out["f_motion_out"],
            "mu_t_clean":     mu_t_clean,
            "s_snr":          ndpn_out["s_snr"],
            "motion_weights": mcpn_out["G_t"],
            "C_omega":        C_omega_list,
            "F_out_list":     F_aligned_list,
            "F_hat":          F_t_aligned,
            "dpe_out":        dpe_out,
            "phase":          phase,
        }
