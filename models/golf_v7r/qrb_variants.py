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

三个变体都只重写基类的钩子 (`_make_r` / `_make_g` / `_modulate_k`), 不复制前向;
基类后续的结构演进 (残差、SPAB 等) 自动作用于全部变体。
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

    # 前向完全继承基类: 基类 forward 在 K 投影后调用 _modulate_k 钩子。
    # 不允许在此复制前向 —— 1027 之后基类加了输入残差与 R 路径 SPAB,
    # 拷贝会让变体静默落后于基线结构, 单变量对照失效 (审查发现的回归, 已删除)。


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

    def _make_r(self, center_feat: torch.Tensor) -> torch.Tensor:
        """基类 R 路径 + 局部上下文残差 (ctx 零初始化 → 起步与基类一致)。"""
        if self.spab_r is not None:
            r_in = self.shift_r(center_feat)
            r_in = self.spab_r(r_in)[0]
            r_in = r_in + self.ctx_r(r_in)          # 局部上下文, 保留原路径
            r_in = self.pre_norm(r_in.flatten(2).transpose(1, 2))
            return self.proj_r(r_in)
        return super()._make_r(center_feat)         # use_omni_shift=False 时 ctx 本为 None

    def _make_g(self, center_feat: torch.Tensor) -> torch.Tensor:
        if self.ctx_g is not None:
            g_in = self.shift_g(center_feat)
            g_in = g_in + self.ctx_g(g_in)
            tokens = self.pre_norm(g_in.flatten(2).transpose(1, 2))
            return self.proj_g(tokens)
        return super()._make_g(center_feat)

