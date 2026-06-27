"""
SACE (Spatial-Adaptive Cross-frame Enhancement) Module — TFS-Net v3
====================================================================
基于可变形跨帧注意力的多帧对齐与光照归一化模块。

实现状态:
    ✅ DeformableCrossAttention: 简单版 + 优化版双实现
    ✅ SACE 主类: 完整 forward 数据流
    ✅ offset reshape 修正 (B.3 P0 bug fix)

数据流:
    1. 对每帧 f_t 做 LFF 频域整形 → f_t_lff
    2. 沿时域取中位值 → μ_t_clean (作为参考帧)
    3. 用 μ_t_clean 作为 query，逐帧 f_t 作为 key/value 做可变形注意力
    4. 输出 list[F_aligned_t]，每帧已对齐到中心帧的空间结构
"""

from __future__ import annotations

from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.blocks import ConvBlock, LayerNorm2d
from models.modules.lff import LFFFeatureAdapter
from models.modules.dwt_lff import DWTLFFAdapter
from models.modules.cross_rwkv import CrossRWKVGate


class DeformableCrossAttention(nn.Module):
    """
    可变形跨帧注意力 — 用 query 引导对 key_value 的空间自适应采样。

    Args:
        channels      : 输入通道数
        n_groups      : 分组数
        kernel_size   : 每组采样的邻域核大小
        use_optimized : 是否使用优化版 (单次 grid_sample)
    """

    def __init__(
        self,
        channels: int,
        n_groups: int = 4,
        kernel_size: int = 3,
        use_optimized: bool = True,
    ):
        super().__init__()
        assert channels % n_groups == 0

        self.channels = channels
        self.n_groups = n_groups
        self.kernel_size = kernel_size
        self.n_points = kernel_size * kernel_size
        self.total_points = n_groups * self.n_points
        self.group_channels = channels // n_groups
        self.use_optimized = use_optimized

        self.value_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def _build_base_grid(self, H: int, W: int, device, dtype):
        k = self.kernel_size
        center = (k - 1) / 2.0
        ys, xs = torch.meshgrid(
            torch.arange(k, device=device, dtype=dtype) - center,
            torch.arange(k, device=device, dtype=dtype) - center,
            indexing='ij',
        )
        base = torch.stack([xs, ys], dim=-1).reshape(-1, 2)
        return base

    def _build_reference_grid(self, B: int, H: int, W: int, device, dtype):
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        ref = torch.stack([grid_x, grid_y], dim=-1)
        ref = ref.unsqueeze(0).expand(B, H, W, 2)
        return ref

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = key_value.shape
        G, K, P = self.n_groups, self.n_points, self.total_points
        Cg = self.group_channels
        device, dtype = key_value.device, key_value.dtype

        # Step 1: 值投影
        value = self.value_proj(key_value)

        # Step 2: 解析 offset 和 mask
        # offset 通道布局: [g0_x_all, g0_y_all, g1_x_all, g1_y_all, ...]
        # 正确解析: view(B, G, 2, K, H, W) → permute → (B, G, K, 2, H, W)
        offset = offset.view(B, G, 2, K, H, W).permute(0, 1, 3, 2, 4, 5).contiguous()
        # offset[:, g, k, 0, h, w] = x_offset, offset[:, g, k, 1, h, w] = y_offset

        mask = mask.view(B, G, K, H, W)
        mask = F.softmax(mask, dim=2)

        # Step 3: 构建采样位置
        base_offset = self._build_base_grid(H, W, device, dtype)
        ref_grid = self._build_reference_grid(B, H, W, device, dtype)

        combined_pixel = offset + base_offset.view(1, 1, K, 2, 1, 1)

        norm_scale = torch.tensor([2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)],
                                  device=device, dtype=dtype).view(1, 1, 1, 2, 1, 1)
        combined_norm = combined_pixel * norm_scale

        ref_expand = ref_grid.permute(0, 3, 1, 2).unsqueeze(1).unsqueeze(1)
        sample_loc = ref_expand + combined_norm

        if self.use_optimized:
            out = self._sample_optimized(value, sample_loc, mask, B, C, H, W, G, K, Cg)
        else:
            out = self._sample_simple(value, sample_loc, mask, B, C, H, W, G, K, Cg)

        out = self.output_proj(out)
        return out

    def _sample_simple(self, value, sample_loc, mask, B, C, H, W, G, K, Cg):
        out = torch.zeros(B, C, H, W, device=value.device, dtype=value.dtype)
        value_groups = value.view(B, G, Cg, H, W)

        for g in range(G):
            v_g = value_groups[:, g]
            for k in range(K):
                loc = sample_loc[:, g, k].permute(0, 2, 3, 1)
                sampled = F.grid_sample(
                    v_g, loc,
                    mode='bilinear', padding_mode='zeros', align_corners=True,
                )
                w_gk = mask[:, g, k].unsqueeze(1)
                out[:, g * Cg:(g + 1) * Cg] += sampled * w_gk
        return out

    def _sample_optimized(self, value, sample_loc, mask, B, C, H, W, G, K, Cg):
        value_grouped = value.view(B, G, Cg, H, W).reshape(B * G, Cg, H, W)

        loc = sample_loc.permute(0, 1, 2, 4, 5, 3)
        loc = loc.reshape(B * G, K, H, W, 2)
        loc = loc.reshape(B * G, K * H, W, 2)

        sampled = F.grid_sample(
            value_grouped, loc,
            mode='bilinear', padding_mode='zeros', align_corners=True,
        )
        sampled = sampled.view(B * G, Cg, K, H, W)

        m = mask.reshape(B * G, K, H, W).unsqueeze(1)
        out_grouped = (sampled * m).sum(dim=2)
        out = out_grouped.view(B, G, Cg, H, W).reshape(B, C, H, W)
        return out


class OffsetMaskHead(nn.Module):
    """从 [query, key_value] 拼接特征生成 offset 和 mask。

    v5.6 P1-1: 内部加 LayerNorm2d（depthwise conv 后、GELU 前），参照 DAT conv_offset。
    v5.6 P1-2: offset_head/mask_head 改为 kaiming weight + zero bias（不全零），参照 DAT。
    """

    def __init__(self, channels: int, n_groups: int, kernel_size: int, hidden: int = 64,
                 use_norm: bool = True, kaiming_init: bool = True):
        super().__init__()
        self.n_groups = n_groups
        self.n_points = kernel_size * kernel_size
        n_off = n_groups * self.n_points * 2
        n_msk = n_groups * self.n_points

        layers = [nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=True)]
        if use_norm:
            layers.append(LayerNorm2d(hidden))
        layers.append(nn.GELU())
        layers.append(nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=True))
        if use_norm:
            layers.append(LayerNorm2d(hidden))
        layers.append(nn.GELU())
        self.shared = nn.Sequential(*layers)

        self.offset_head = nn.Conv2d(hidden, n_off, kernel_size=1, bias=True)
        self.mask_head = nn.Conv2d(hidden, n_msk, kernel_size=1, bias=True)

        if kaiming_init:
            # v5.6 P1-2: kaiming weight + zero bias (DAT 风格，梯度尺度健康)
            nn.init.kaiming_normal_(self.offset_head.weight, mode='fan_in')
            nn.init.zeros_(self.offset_head.bias)
            nn.init.kaiming_normal_(self.mask_head.weight, mode='fan_in')
            nn.init.zeros_(self.mask_head.bias)
        else:
            # v5.5 旧行为: 全零初始化
            nn.init.zeros_(self.offset_head.weight)
            nn.init.zeros_(self.offset_head.bias)
            nn.init.zeros_(self.mask_head.weight)
            nn.init.zeros_(self.mask_head.bias)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor):
        x = torch.cat([query, key_value], dim=1)
        h = self.shared(x)
        offset = self.offset_head(h)
        mask = self.mask_head(h)
        return offset, mask


class SACE(nn.Module):
    """
    Spatial-Adaptive Cross-frame Enhancement.

    Args:
        channels      : 输入特征通道数
        n_groups      : 可变形注意力分组数
        kernel_size   : 可变形采样核大小
        use_optimized : 采样器是否使用优化版
        lff_module    : 外部传入的 LFFFeatureAdapter 或 DWTLFFAdapter (None 则内部新建)
                         v6.2: 支持 DWTLFFAdapter — 取 feat_sace 输出 (相位差+高频结构)
        K, n_ang_freq : 内部 LFF 的参数
    """

    def __init__(
        self,
        channels: int = 64,
        n_groups: int = 4,
        kernel_size: int = 3,
        use_optimized: bool = True,
        lff_module: LFFFeatureAdapter = None,
        K: int = 10,
        n_ang_freq: int = 1,
        phase_preserving: bool = True,
        offset_use_norm: bool = False,
        offset_kaiming_init: bool = False,
        use_cross_rwkv: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.n_groups = n_groups
        self.kernel_size = kernel_size

        self._lff_external = lff_module is not None
        if lff_module is not None:
            # 共享模式: 使用外部 LFF (其 phase_preserving 由创建方决定)
            self.lff = lff_module
        else:
            # 独立模式 (M4 消融): 内部创建, 可启用相位整形辅助对齐
            self.lff = LFFFeatureAdapter(
                channels=channels, K=K, n_ang_freq=n_ang_freq, per_channel_rbf=False,
                phase_preserving=phase_preserving,
            )

        self.offset_mask_head = OffsetMaskHead(
            channels=channels, n_groups=n_groups, kernel_size=kernel_size, hidden=64,
            use_norm=offset_use_norm, kaiming_init=offset_kaiming_init,
        )

        self.deform_attn = DeformableCrossAttention(
            channels=channels,
            n_groups=n_groups,
            kernel_size=kernel_size,
            use_optimized=use_optimized,
        )

        self.norm_q = LayerNorm2d(channels)
        self.norm_kv = LayerNorm2d(channels)

        # v6: Cross-RWKV Gate — deformable 对齐后做多帧长程聚合
        self.use_cross_rwkv = use_cross_rwkv
        if use_cross_rwkv:
            self.cross_rwkv = CrossRWKVGate(channels=channels, num_frames=5)
        else:
            self.cross_rwkv = None

    @staticmethod
    def _soft_median(x: torch.Tensor, dim: int = 1, tau: float = 0.1) -> torch.Tensor:
        """
        梯度友好的软中位值近似。
        所有帧都获得梯度回传，距中位值近的帧权重更大。
        """
        with torch.no_grad():
            med = x.median(dim=dim).values.unsqueeze(dim)
        dist = (x - med).abs()
        weights = F.softmax(-dist / tau, dim=dim)
        return (weights * x).sum(dim=dim)

    def forward(
        self,
        feats: torch.Tensor,
        tfsi_out: Dict = None,
        cached_lff: Dict = None,
    ) -> Dict:
        B, T, C, H, W = feats.shape
        assert C == self.channels

        # Step 1: 逐帧 LFF（支持缓存：cached_lff[local_idx] → 预计算特征）
        lff_feats: List[torch.Tensor] = []
        for t in range(T):
            if cached_lff and t in cached_lff:
                lff_feats.append(cached_lff[t])
            else:
                lff_out = self.lff(feats[:, t])
                # v6.2: DWTLFFAdapter 返回 dict, SACE 取 feat_sace
                if isinstance(lff_out, dict):
                    lff_feats.append(lff_out["feat_sace"])
                else:
                    lff_feats.append(lff_out)
        lff_stack = torch.stack(lff_feats, dim=1)

        # Step 2: 中心帧直接作参考（v6.2: soft-median 移除）
        # soft-median 与 DAT 风格注意力绑定, RWKV 不需要参考帧引导
        # 改用中心帧 LFF 特征直接作为对齐 query
        mu_t_clean = lff_stack[:, T // 2]  # (B, C, H, W) — 中心帧 LFF
        # 时域标准差 — 供 NDPN 计算 SNR
        sigma_t_clean = lff_stack.std(dim=1, unbiased=False)

        # M3: 从 TFSI 输出提取 s_noise, 用于噪声感知残差门控
        s_noise = tfsi_out.get("s_noise") if tfsi_out else None

        # Step 3: 逐帧可变形对齐
        attn_maps: List[Tuple[torch.Tensor, torch.Tensor]] = []
        F_aligned_list: List[torch.Tensor] = []

        q_norm = self.norm_q(mu_t_clean)

        for t in range(T):
            kv = lff_feats[t]
            kv_norm = self.norm_kv(kv)

            offset, mask = self.offset_mask_head(q_norm, kv_norm)
            f_aligned = self.deform_attn(q_norm, kv_norm, offset, mask)
            # M3: 噪声感知残差门控 — 高噪区抑制残差(噪声), 低噪区保留信息流
            if s_noise is not None:
                f_aligned = f_aligned + (1.0 - s_noise) * kv
            else:
                f_aligned = f_aligned + kv  # 残差 (向后兼容: tfsi_out 未提供时)

            attn_maps.append((offset, mask))
            F_aligned_list.append(f_aligned)

        # v6: Cross-RWKV Gate — deformable 对齐后做多帧长程聚合
        if self.cross_rwkv is not None:
            F_aligned_list = self.cross_rwkv(
                query=mu_t_clean,
                frames=F_aligned_list,
                s_noise=s_noise,
            )

        return {
            "attn_maps":       attn_maps,
            "mu_t_clean":      mu_t_clean,
            "sigma_t_clean":   sigma_t_clean,
            "F_aligned_list":  F_aligned_list,
            "lff_feats":       lff_feats,
        }
