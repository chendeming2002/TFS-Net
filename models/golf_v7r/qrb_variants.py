#!/usr/bin/env python3
"""QRB 的三个改进方向 (见 docs/v8/12-QRB改进方向评估.md)。

三个变体各自只改一处, 与 `cross_rwkv` 的单变量对照保持一致:

- **方案 A (K 调制)**: Q 生成逐通道的缩放与偏移, 在进入扫描前调制 K。
  只调制 K, 不动 V, 因为 K 决定检索的权重分布, 不影响数值稳定性; V 直接进入
  clamp(-8, 8), 若在扫描前再调制容易把它推到边界。
- **方案 B (Q 内容支路)**: 在现有输出之外加一条受控的加性支路 λ·ψ(Q),
  λ 严格零初始化。这条支路与 additive 融合里已冻结的 residual_gamma 是同类结构,
  只作为诊断性消融, 不与 frozen residual 叠加使用。
- **方案 C (门控局部上下文)**: R/G 的投影之前各加一次 3×3 深度卷积的局部上下文,
  测试门控表达能力是否是瓶颈。

三个变体都只继承 `CrossRWKVHeadV8` 的扫描与初始化, 不重复实现 WKV 逻辑。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.golf_v7r.cross_rwkv_head_v8 import CrossRWKVHeadV8


class CrossRWKVHeadKvMod(CrossRWKVHeadV8):
    """方案 A: Q 条件化 K 调制。

    Q 经全局平均池化 + 线性得到两组逐通道参数 (a, b), 在进入扫描前调制 K:
        K' = K ⊙ (1 + α·tanh(a)) + β·tanh(b)
    α、β 是逐通道的可学习尺度, 零初始化 → 初始时 K' = K, 与原结构完全一致。
    """

    def __init__(self, channels: int, **kwargs):
        super().__init__(channels, **kwargs)
        # Q 的通道级条件: 池化 + 两个线性投影 (缩放 a / 偏移 b)
        self.q_mod_pool = nn.AdaptiveAvgPool2d(1)
        self.q_mod_r = nn.Linear(channels, channels, bias=True)
        self.q_mod_g = nn.Linear(channels, channels, bias=True)
        self.q_scale_a = nn.Parameter(torch.zeros(channels))
        self.q_scale_b = nn.Parameter(torch.zeros(channels))

    def _modulate_k(self, center_feat: torch.Tensor,
                    k_tokens: torch.Tensor) -> torch.Tensor:
        """K' = K ⊙ (1 + α·tanh(a)) + β·tanh(b); α、β 零初始化 → 初始不变。

        监控 (只记录, 不改计算): scale/shift 的实际幅度, 供训练日志读取。
        """
        q_vec = self.q_mod_pool(center_feat).flatten(1)      # (B, C)
        a = torch.tanh(self.q_mod_r(q_vec))                  # (B, C)
        b = torch.tanh(self.q_mod_g(q_vec))
        scale = 1.0 + self.q_scale_a * a                      # (B, C)
        shift = self.q_scale_b * b
        # 供诊断: 幅度与调制后 K 的范围
        self._last_k_mod = {
            "scale_mean": scale.detach().mean(), "scale_absmax": scale.detach().abs().max(),
            "shift_absmax": shift.detach().abs().max(),
            "k_absmax_post": (k_tokens * scale.unsqueeze(1) + shift.unsqueeze(1)).detach().abs().max(),
        }
        return k_tokens * scale.unsqueeze(1) + shift.unsqueeze(1)

    def forward(self, center_feat: torch.Tensor,
                context_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = center_feat.shape
        r = self._project(center_feat, self.shift_r, self.proj_r)
        g = self._project(center_feat, self.shift_g, self.proj_g)
        k = self._project(context_feat, self.shift_k, self.proj_k)
        k = self._modulate_k(center_feat, k)                # 方案 A 的唯一改动
        v = self._project(context_feat, self.shift_v, self.proj_v)

        D, hd = self.num_dirs, self.head_dim
        r = r.view(B, H * W, D, hd)
        g = torch.nn.functional.silu(g.view(B, H * W, D, hd))

        scan_fns = self._scan_fns()
        k_list, v_list = [], []
        for i, scan_fn in enumerate(scan_fns):
            k_2d = k[:, :, i * hd:(i + 1) * hd].transpose(1, 2).reshape(B, hd, H, W)
            v_2d = v[:, :, i * hd:(i + 1) * hd].transpose(1, 2).reshape(B, hd, H, W)
            k_list.append(scan_fn(k_2d))
            v_list.append(scan_fn(v_2d))
        from models.modules.batched_wkv import batched_scan
        wkv_list = batched_scan(self.bi_wkv_list, k_list, v_list, total_tokens=H * W)

        heads = []
        for i, scan_fn in enumerate(scan_fns):
            wkv_i = wkv_list[i][:, self._inv_index(scan_fn, B, hd, H, W,
                                                   center_feat.device)]
            heads.append(g[:, :, i] * self.head_norm(r[:, :, i] * wkv_i))
        out = torch.cat(heads, dim=-1)
        out = self.proj_out(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        if self.use_channel_mix:
            a = self.post_norm(out.flatten(2).transpose(1, 2))
            r2 = self.cm_r(a)
            k2 = self.cm_k(a)
            if self.cm_act == "relu_sq":
                v2 = torch.relu(k2).pow(2)
            else:
                v2 = torch.nn.functional.leaky_relu(k2, 0.01).pow(2)
            a2 = torch.sigmoid(r2) * self.cm_v(v2)
            out = out + self.cm_out(a2).transpose(1, 2).reshape(B, C, H, W)
        return out


class CrossRWKVHeadQPath(CrossRWKVHeadV8):
    """方案 B: 受控的 Q 内容支路。

    输出 = SiLU(G)·LN(R·WKV) + λ·ψ(Q)
    ψ 是 1×1 卷积, λ 零初始化 → 初始时与原结构完全一致。

    ⚠️ 与 additive 融合里已冻结的 residual_gamma 是同类直通结构, 只作消融诊断,
    不与 frozen residual 叠加使用; 训练后必须检查 λ 的幅度。
    """

    def __init__(self, channels: int, **kwargs):
        super().__init__(channels, **kwargs)
        self.q_path = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.q_lambda = nn.Parameter(torch.zeros(1))
        # ψ 的权重较小初始化, λ 严格零初始化, 起步与原结构一致
        nn.init.uniform_(self.q_path.weight,
                         -0.05 / (channels ** 0.5), 0.05 / (channels ** 0.5))

    def forward(self, center_feat: torch.Tensor,
                context_feat: torch.Tensor) -> torch.Tensor:
        main = super().forward(center_feat, context_feat)
        side = self.q_path(center_feat)
        # 监控 (只记录, 不改计算): 旁路与主路径的 RMS 比。λ 本身是标量,
        # 其幅度由训练决定, 本监控用于事后判断旁路是否被使用。
        with torch.no_grad():
            main_rms = main.pow(2).mean().sqrt()
            side_rms = side.pow(2).mean().sqrt()
            self._last_q_path = {
                "lambda": float(self.q_lambda.detach()),
                "side_rms": float(side_rms),
                "main_rms": float(main_rms),
                "side_to_main": float(side_rms / (main_rms + 1e-8)),
            }
        return main + self.q_lambda * side


class CrossRWKVHeadGateCtx(CrossRWKVHeadV8):
    """方案 C: R/G 投影前加局部上下文, 测试门控表达能力。

    R/G 的 Q-Shift 之后, 各经一次 3×3 深度卷积再进线性投影。其他一切不变。
    """

    def __init__(self, channels: int, **kwargs):
        super().__init__(channels, **kwargs)
        if self.use_omni_shift:
            self.ctx_r = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels,
                                   bias=False)
            self.ctx_g = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels,
                                   bias=False)
            # 零初始化: 起步时 ctx(x) = 0, 与原版一致; 审查实测初始残差 RMS 比 0.596,
            # 过大, 会在训练一开始就显著改变 R/G 投影的输入分布。这里先归零,
            # 让门控从原版状态起步, 再让 ctx 通过训练逐步加入。
            nn.init.zeros_(self.ctx_r.weight)
            nn.init.zeros_(self.ctx_g.weight)
        else:
            self.ctx_r = self.ctx_g = None

    def _project(self, x_2d, shift, proj, ctx=None):
        """Q-Shift → 局部上下文 (可选) → 归一化 → 线性投影。"""
        if shift is not None:
            x_2d = shift(x_2d)
        if ctx is not None:
            x_2d = x_2d + ctx(x_2d)          # 残差注入, 保留原路径
        tokens = self.pre_norm(x_2d.flatten(2).transpose(1, 2))
        return proj(tokens)

    def forward(self, center_feat: torch.Tensor,
                context_feat: torch.Tensor) -> torch.Tensor:
        r = self._project(center_feat, self.shift_r, self.proj_r, self.ctx_r)
        g = self._project(center_feat, self.shift_g, self.proj_g, self.ctx_g)
        k = self._project(context_feat, self.shift_k, self.proj_k)
        v = self._project(context_feat, self.shift_v, self.proj_v)
        return self._finish(center_feat, r, g, k, v)

    def _finish(self, center_feat, r, g, k, v):
        """与原 forward 相同的后半段, 从扫描开始。"""
        B, C, H, W = center_feat.shape
        D, hd = self.num_dirs, self.head_dim
        r = r.view(B, H * W, D, hd)
        g = torch.nn.functional.silu(g.view(B, H * W, D, hd))

        scan_fns = self._scan_fns()
        k_list, v_list = [], []
        for i, scan_fn in enumerate(scan_fns):
            k_2d = k[:, :, i * hd:(i + 1) * hd].transpose(1, 2).reshape(B, hd, H, W)
            v_2d = v[:, :, i * hd:(i + 1) * hd].transpose(1, 2).reshape(B, hd, H, W)
            k_list.append(scan_fn(k_2d))
            v_list.append(scan_fn(v_2d))
        from models.modules.batched_wkv import batched_scan
        wkv_list = batched_scan(self.bi_wkv_list, k_list, v_list, total_tokens=H * W)

        heads = []
        for i, scan_fn in enumerate(scan_fns):
            wkv_i = wkv_list[i][:, self._inv_index(scan_fn, B, hd, H, W,
                                                   center_feat.device)]
            heads.append(g[:, :, i] * self.head_norm(r[:, :, i] * wkv_i))
        out = torch.cat(heads, dim=-1)
        out = self.proj_out(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        if self.use_channel_mix:
            a = self.post_norm(out.flatten(2).transpose(1, 2))
            r2 = self.cm_r(a)
            k2 = self.cm_k(a)
            if self.cm_act == "relu_sq":
                v2 = torch.relu(k2).pow(2)
            else:
                v2 = torch.nn.functional.leaky_relu(k2, 0.01).pow(2)
            a2 = torch.sigmoid(r2) * self.cm_v(v2)
            out = out + self.cm_out(a2).transpose(1, 2).reshape(B, C, H, W)
        return out
