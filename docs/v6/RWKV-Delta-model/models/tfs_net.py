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
from models.modules.pure_rwkv_sace import TCA
from models.modules.ispn_v2 import ISPN
from models.modules.ndpn import NDPN
from models.modules.mrpn import MCPN
from models.modules.igrf import SGRF


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
        # Stage 0: PyramidEncoder
        self.encoder = PyramidEncoder(
            in_channels=in_channels,
            level_channels=level_channels,
            fused_channels=fused_channels,
            num_bottleneck_blocks=num_bottleneck_blocks,
        )

        self.dpe = DPE(
            channels=fused_channels, eps=eps,
            use_soft_median=use_soft_median,
        )

        self.tca = TCA(channels=fused_channels)

        self.ispn = ISPN(channels=fused_channels, img_channels=in_channels)
        self.ndpn = NDPN(channels=fused_channels)
        self.mcpn = MCPN(channels=fused_channels)

        self.cxg = CXG(channels=fused_channels)

        self.sgrf = SGRF(channels=fused_channels, out_channels=in_channels,
                         use_soft_clamp=use_soft_clamp, use_nafblock=use_nafblock,
                         num_res_blocks=num_igrf_res_blocks)

        # 逐帧特征缓存 (key: frame_global_index → {l1_lat, l2_lat, l3_lat})
        self.frame_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._cache_max_size = 64
        self._cache_access_order: List[int] = []

    def clear_frame_cache(self):
        self.frame_cache.clear()
        self._cache_access_order.clear()

    def _cache_evict_lru(self):
        while len(self._cache_access_order) > self._cache_max_size:
            oldest = self._cache_access_order.pop(0)
            if oldest in self.frame_cache:
                del self.frame_cache[oldest]

    def _cache_put(self, gidx: int, key: str, tensor: torch.Tensor):
        if gidx not in self.frame_cache:
            self.frame_cache[gidx] = {}
        self.frame_cache[gidx][key] = tensor.detach()
        if gidx not in self._cache_access_order:
            self._cache_access_order.append(gidx)
        self._cache_evict_lru()

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

        # Stage 0: Encoder → lateral features with frame cache
        use_cache = frame_indices is not None
        if use_cache:
            l1_list, l2_list, l3_list = [], [], []
            for i in range(T):
                gidx = frame_indices[i]
                if gidx in self.frame_cache and "l1_lat" in self.frame_cache[gidx]:
                    l1_list.append(self.frame_cache[gidx]["l1_lat"])
                    l2_list.append(self.frame_cache[gidx]["l2_lat"])
                    l3_list.append(self.frame_cache[gidx]["l3_lat"])
                    self._cache_access_order.remove(gidx)
                    self._cache_access_order.append(gidx)
                else:
                    l1, l2, l3 = self.encoder.forward_single_lateral(x[:, i])
                    l1_list.append(l1)
                    l2_list.append(l2)
                    l3_list.append(l3)
                    self._cache_put(gidx, "l1_lat", l1)
                    self._cache_put(gidx, "l2_lat", l2)
                    self._cache_put(gidx, "l3_lat", l3)
            l1_lat = torch.stack(l1_list, dim=1)
            l2_lat = torch.stack(l2_list, dim=1)
            l3_lat = torch.stack(l3_list, dim=1)
        else:
            x_flat = x.reshape(B * T, C_in, H, W)
            l1_flat, l2_flat, l3_flat = self.encoder.forward_single_lateral(x_flat)
            l1_lat = l1_flat.reshape(B, T, *l1_flat.shape[1:])
            l2_lat = l2_flat.reshape(B, T, *l2_flat.shape[1:])
            l3_lat = l3_flat.reshape(B, T, *l3_flat.shape[1:])
        feats = l1_lat

        # Stage 1: DPE — single-scale on l3 (H/4) + softplus illumination head
        dpe_out = self.dpe(l3_lat, center_frame=image_center)
        s_illum = dpe_out["s_illum"]
        s_noise = dpe_out["s_noise"]
        s_illum = F.interpolate(s_illum, size=(H, W), mode='bilinear', align_corners=False)
        s_noise = F.interpolate(s_noise, size=(H, W), mode='bilinear', align_corners=False)

        # Stage 2: TCA — WKV @ H/2 on l2_lat directly (no WFR pre-processing)
        tca_out = self.tca(l2_lat)
        F_aligned_list_half = [tca_out["tca_out"][:, t] for t in range(T)]
        F_aligned_list = [F.interpolate(f, size=(H, W), mode='bilinear', align_corners=False) for f in F_aligned_list_half]
        C_omega_list = tca_out.get("C_omega_list", [])
        F_t_aligned = F.interpolate(tca_out["F_t_aligned"], size=(H, W), mode='bilinear', align_corners=False)
        mu_t_clean = F.interpolate(tca_out["mu_t_clean"], size=(H, W), mode='bilinear', align_corners=False)
        sigma_t_clean = F.interpolate(tca_out["sigma_t_clean"], size=(H, W), mode='bilinear', align_corners=False)

        f_enc_center = feats[:, center_idx]
        ispn_out = self.ispn(f_enc_center, s_illum)
        gain_map = ispn_out["gain_map"]
        f_illum_feat = ispn_out["f_illum_feat"]
        curve_A = ispn_out.get("curve_A", None)
        alpha_target = ispn_out.get("alpha_target", None)

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

        # SGRF: 阶段式修复融合 (Flight5: TCC curve)
        sgrf_out = self.sgrf(
            gain_map=gain_map,
            f_noise_out=f_noise_gated,
            f_motion_out=f_motion_gated,
            image_center=image_center,
            curve_A=curve_A,
            alpha_target=alpha_target,
            curve_iter=self.ispn.curve_iter,
        )

        return {
            "res_t":          sgrf_out["res_t"],
            "img_s1":         sgrf_out["img_s1"],
            "img_s2":         sgrf_out["img_s2"],
            "img_curved":     sgrf_out.get("img_curved", sgrf_out["img_s2"]),
            "img_lit":        sgrf_out.get("img_lit", sgrf_out.get("img_curved", sgrf_out["img_s2"])),
            "residual":       sgrf_out.get("residual", torch.zeros_like(sgrf_out["res_t"])),
            "lit_up_map":     sgrf_out["lit_up_map"],
            "gain_map":       gain_map,
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
            "curve_A":        curve_A,
            "alpha_target":   alpha_target,
            "phase":          phase,
        }
