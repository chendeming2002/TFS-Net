"""
TFS-Net v3 / v6 Charlie — Three-source Fusion & Synthesis Network
==================================================================
端到端多帧低光增强网络。

整体结构 (5 stages):
    Stage 0: PyramidEncoder        多帧 → 多尺度特征 + 全分辨率融合
    Stage 1: TFSI                  时序光照/噪声强度场估计 + 频/空双分支
    Stage 2: SACE                  可变形跨帧对齐 (与 TFSI 共享 LFF)
    Stage 3: IFPN/NDPN/MRPN        三源恢复分支 (光照/噪声/运动)
    Stage 4: IGRF                  强度引导残差融合 → 输出

v6 Charlie 数据流 (charlie_mode=True):
    TFSI → s_noise → NDNP (条件输入, Charlie P0)
    TFSI → s_illum → IGRF Stage3 (保留)
    SACE → sigma_t_clean → MRPN (Charlie P1: σ→MRPN)
    SACE → sigma_sace → s_noise 调制 (Charlie P2: 噪声图来源)
    IGRF ≠ s_noise (Charlie P0: IGRF 仅接收 s_illum)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.encoder import PyramidEncoder
from models.modules.tfsi import TFSI
from models.modules.sace import SACE
from models.modules.ifpn import IFPN
from models.modules.ndpn import NDPN
from models.modules.mrpn import MRPN
from models.modules.igrf import IGRF
from models.modules.amp_enhance import AmpEnhance
from models.modules.dwt_lff import SpatialDWTLFFAdapter
from models.modules.pure_rwkv_sace import PureRWKVSACE


class CrossFusionGate(nn.Module):
    """Charlie3 P1 + Delta: NDPN/MRPN 交叉门控 (结构性重参数化)

    训练时: 动态交叉门控协调梯度冲突
    推理时: 融合为静态逐通道缩放 (deploy=True)
    参考 DRNet (CVPR 2026) DRMLP 重参数化范式
    """

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
            nn.init.ones_(self.gate_noise[-2].bias)  # sigmoid(1)≈0.73, stable start
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
        """训练结束后调用: 用统计平均门控权重创建推理融合模块"""
        deploy_mod = CrossFusionGate(self.channels, deploy=True)
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

        # Charlie2 D2: IFPN 图像输入 → encoder 特征投影 (64→3)
        self.feat_to_img = nn.Conv2d(fused_channels, in_channels, 3, 1, 1)

        # v6 Bravo: DWT-LFF — TFSI 用 center (α=0.6 锚定), SACE 自建双实例
        self.use_dwt_lff = use_dwt_lff
        if use_dwt_lff:
            dwt_lff_tfsi = SpatialDWTLFFAdapter(in_channels=fused_channels, alpha_init=0.6)
        else:
            dwt_lff_tfsi = None

        # Stage 1: TFSI
        self.tfsi = TFSI(
            channels=fused_channels,
            fused_channels=fused_channels,
            eps=eps,
            use_soft_median=use_soft_median,
            dwt_lff=dwt_lff_tfsi,
        )

        # Stage 2: SACE / PureRWKVSACE
        if use_dwt_lff and not use_pure_rwkv:
            shared_lff = dwt_lff_tfsi
        else:
            shared_lff = self.tfsi.freq_branch.lff if share_lff else None

        if use_pure_rwkv:
            # v6.5: 纯 RWKV (内部自建双 DWT-LFF 实例 + Bravo V raw)
            self.sace = PureRWKVSACE(channels=fused_channels, n_layer=1)
        else:
            self.sace = SACE(
                channels=fused_channels,
                n_groups=n_groups,
                kernel_size=kernel_size,
                use_optimized=True,
                lff_module=shared_lff,
                phase_preserving=sace_phase_preserving,
                offset_use_norm=sace_offset_use_norm,
                offset_kaiming_init=sace_offset_kaiming_init,
                use_cross_rwkv=use_cross_rwkv,
            )

        # Stage 3: 三源恢复分支
        self.ifpn = IFPN(
            fused_channels=fused_channels,
            coarse_channels=coarse_channels,
            img_channels=in_channels,
        )
        self.ndpn = NDPN(channels=fused_channels)
        self.mrpn = MRPN(channels=fused_channels)

        # Charlie3 P1: NDPN/MRPN 输出端交叉门控 (VSRELL cross-modulation 风格)
        self.cross_fuse = CrossFusionGate(channels=fused_channels)

        # Stage 4: IGRF
        self.igrf = IGRF(channels=fused_channels, out_channels=in_channels,
                         use_soft_clamp=use_soft_clamp, use_nafblock=use_nafblock,
                         num_res_blocks=num_igrf_res_blocks)

        # 逐帧特征缓存 (推理时滑动窗口复用)
        self.frame_cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def clear_frame_cache(self):
        """清空逐帧特征缓存（切换序列或释放显存时调用）。"""
        self.frame_cache.clear()

    def forward(
        self,
        x: torch.Tensor,
        frame_indices: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, 3, H, W) 多帧低光输入

        Returns:
            dict with keys: res_t, delta, f_fused_igrf, s_illum, s_noise, ...
        """
        B, T, C_in, H, W = x.shape
        center_idx = T // 2

        # v5.9: AmpEnhance — 图像级频域幅度增强 (Encoder 前处理)
        # 全帧共用 center frame 的 curve_amps，保证 SACE 对齐一致性
        if self.amp_enhance is not None:
            # 对 center frame 估计 curve_amps
            img_center = x[:, center_idx]  # (B, 3, H, W)
            with torch.no_grad():
                curve_amps = self.amp_enhance.amp_net(img_center).clamp(min=0.1, max=1.0)
            # 对每帧用同一 curve_amps 做幅度增强
            x_enhanced = torch.empty_like(x)
            for i in range(T):
                y_i = x[:, i]  # (B, 3, H, W)
                F_i = torch.fft.fft2(y_i, dim=(-2, -1), norm='ortho')
                mag_i = torch.abs(F_i) / curve_amps
                pha_i = torch.angle(F_i)
                F_new = torch.polar(mag_i, pha_i)
                x_enhanced[:, i] = torch.fft.ifft2(F_new, dim=(-2, -1), norm='ortho').real
            x = x_enhanced

        # Stage 0: 编码（支持逐帧缓存）
        feats_list: List[torch.Tensor] = []
        for i in range(T):
            gidx = frame_indices[i] if frame_indices else None
            if gidx is not None and gidx in self.frame_cache:
                feats_list.append(self.frame_cache[gidx]["feat"])
            else:
                f = self.encoder.forward_single(x[:, i], return_coarse=False)
                feats_list.append(f)
                if gidx is not None:
                    self.frame_cache[gidx] = {"feat": f}
        feats = torch.stack(feats_list, dim=1)

        # Stage 1: TFSI
        tfsi_out = self.tfsi(feats)
        F_fused  = tfsi_out["F_fused"]
        s_illum  = tfsi_out["s_illum"]
        s_noise  = tfsi_out["s_noise"]

        # Stage 2: SACE（LFF 支持逐帧缓存）
        cached_lff: Dict[int, torch.Tensor] = {}
        if frame_indices:
            for i, gidx in enumerate(frame_indices):
                if gidx in self.frame_cache and "lff" in self.frame_cache[gidx]:
                    cached_lff[i] = self.frame_cache[gidx]["lff"]

        sace_out_dict = self.sace(
            feats, tfsi_out,
            cached_lff=cached_lff if cached_lff else None,
        )

        # Delta: 新 SACE 输出接口
        sace_feats      = sace_out_dict["sace_out"]   # (B, T, C, H, W)
        C_omega_list    = sace_out_dict.get("C_omega_list", [])
        F_t_aligned     = sace_out_dict["F_t_aligned"] # (B, C, H, W)
        mu_t_clean      = sace_out_dict["mu_t_clean"]
        sigma_t_clean   = sace_out_dict["sigma_t_clean"]

        # 兼容旧接口: F_aligned_list = list of (B, C, H, W) per frame
        F_aligned_list = [sace_feats[:, t] for t in range(T)]

        # Stage 3: 三源恢复
        image_center = x[:, center_idx]

        # v5.3: IFPN 改用 SACE 对齐特征（不再使用 Encoder 粗特征）
        aligned_feats = torch.stack(F_aligned_list, dim=1)  # (B, T, C_f, H, W)

        # Charlie2 D2: IFPN 输入用 Encoder 浅层特征 (避免原始噪声泄漏)
        h_c, w_c = H // 4, W // 4
        f_center_for_ifpn = feats[:, center_idx]  # (B, 64, H, W) encoder 中心帧
        image_down = F.interpolate(
            self.feat_to_img(f_center_for_ifpn),
            size=(h_c, w_c), mode='bilinear', align_corners=False
        )
        # 邻帧图像: 对 encoder 邻帧做同样投影
        imgs_flat = feats.reshape(B * T, feats.shape[2], H, W)
        imgs_proj = self.feat_to_img(imgs_flat)
        imgs_down = F.interpolate(
            imgs_proj, size=(h_c, w_c), mode='bilinear', align_corners=False
        ).view(B, T, imgs_proj.shape[1], h_c, w_c)

        ifpn_out = self.ifpn(
            I_t_down=image_down,
            aligned_feats=aligned_feats,
            center_idx=center_idx,
            imgs_down=imgs_down,
            s_illum=s_illum,
            F_t_aligned=F_t_aligned,
        )

        ndpn_out = self.ndpn(
            feats=feats,
            F_aligned_list=F_aligned_list,
            mu_t_clean=mu_t_clean,
            sigma_t_clean=sigma_t_clean,
            s_noise=s_noise,
            center_idx=center_idx,
            C_omega_list=C_omega_list,         # Delta: correspondence confidence
            F_t_aligned=F_t_aligned,            # Delta: aligned reference
        )

        # Charlie P2: sigma_sace 调制 s_noise
        if self.charlie_mode and s_noise is not None:
            sigma_sace = sigma_t_clean.mean(dim=1, keepdim=True)
            scale_sigma = torch.sigmoid(-sigma_sace * 2.0)
            s_noise = s_noise * scale_sigma

        mrpn_out = self.mrpn(
            F_aligned_list=F_aligned_list,
            center_idx=center_idx,
            sigma_t_clean=sigma_t_clean if self.charlie_mode else None,
            C_omega_list=C_omega_list,         # Delta: motion magnitude
            F_t_aligned=F_t_aligned,            # Delta: alignment baseline
        )

        # Extract outputs for IGRF
        lit_up_map_raw = ifpn_out["lit_up_map_raw"]
        f_illum_feat   = ifpn_out["f_illum_feat"]
        A_illu         = ifpn_out.get("A_illu")        # Delta: from IFPN
        f_noise_out_raw = ndpn_out["f_noise_out"]
        f_motion_out_raw = mrpn_out["f_motion_out"]

        # Charlie3 P1: 交叉门控 — 互补置信度交换
        f_noise_out, f_motion_out = self.cross_fuse(f_noise_out_raw, f_motion_out_raw)

        # Stage 4: IGRF (Delta: A_illu 由 IFPN 传入)
        igrf_out = self.igrf(
            f_illum_feat=f_illum_feat,
            f_noise_out=f_noise_out,
            f_motion_out=f_motion_out,
            lit_up_map_raw=lit_up_map_raw,
            image_center=image_center,
            A_illu=A_illu,
        )

        return {
            "res_t":          igrf_out["res_t"],
            "img_s1":         igrf_out["img_s1"],
            "img_s2":         igrf_out["img_s2"],
            "lit_up_map":     igrf_out["lit_up_map"],
            "image_center":   image_center,
            "s_illum":        s_illum,
            "s_noise":        s_noise,
            "f_illum_feat":   ifpn_out["f_illum_feat"],
            "f_noise_out":    ndpn_out["f_noise_out"],
            "f_motion_out":   mrpn_out["f_motion_out"],
            "L_t":            ifpn_out["L_t"],
            "L_ref":          ifpn_out["L_ref"],
            "L_ratio":        ifpn_out["L_ratio"],
            "A_illu":         A_illu,                       # Delta: from IFPN
            "ifpn_side":      ifpn_out.get("ifpn_side"),
            "mu_t_clean":     mu_t_clean,
            "s_snr":          ndpn_out["s_snr"],
            "motion_weights": mrpn_out["G_t"],
            "tfsi_out":       tfsi_out,
        }
