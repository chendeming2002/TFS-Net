#!/usr/bin/env python3
"""v8 WKV 实现 (单归一化双向 + CUDA 算子) 的单元测试。

对应 `models/modules/cuda_wkv.py` 与 `docs/v8/04-训练加速调研.md`。

核心验证:
  1. 官方定义 (单归一化双向) 的数学性质: 均匀权重下输出等于 v 的平均;
  2. 与暴力参考逐元素一致 (暴力参考是判定"哪种写法正确"的依据);
  3. 与旧实现 (双比值平均) **不等价** —— 固化这一差异, 防止误以为可以互换;
  4. 反传正常, 参数有梯度;
  5. CUDA 版与纯 PyTorch 回退版一致 (若算子可用)。

运行:
    CUDA_HOME=/usr/local/cuda CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9 \
    PATH=$CONDA_PREFIX/bin:$PATH PYTHONPATH=. \
    python tests/test_cuda_wkv.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.golf_r4.tca_rwkv import FastBiWKV
from models.modules.cuda_wkv import (
    CudaBiWKV, TorchBiWKVSingleNorm, cuda_available)


def _brute_force(k, v, w_kernel, u_kernel):
    """单归一化双向 WKV 的暴力参考 (对数域稳定)。

    y_t = [Σ_{j≠t} exp(k_j − w_kernel·|t−j|) v_j + exp(u_kernel + k_t) v_t]
        / [Σ_{j≠t} exp(k_j − w_kernel·|t−j|)       + exp(u_kernel + k_t)    ]
    """
    L, C = k.shape
    idx = np.arange(L)
    out = np.zeros((L, C))
    for t in range(L):
        d = np.abs(t - idx).astype(np.float64)
        logits = k - w_kernel * d[:, None]
        mask = np.ones(L, bool)
        mask[t] = False
        logits = np.concatenate([logits[mask], (u_kernel + k[t])[None, :]], axis=0)
        vals = np.concatenate([v[mask], v[t][None, :]], axis=0)
        m = logits.max(axis=0)
        e = np.exp(logits - m)
        out[t] = (e * vals).sum(axis=0) / e.sum(axis=0)
    return out


def test_torch_fallback_matches_brute_force():
    """纯 PyTorch 回退版与暴力参考一致。"""
    torch.manual_seed(0)
    L, C = 32, 4
    k = torch.randn(1, L, C)
    v = torch.randn(1, L, C)
    m = TorchBiWKVSingleNorm(C)
    with torch.no_grad():
        m.spatial_decay.fill_(0.2)
        m.spatial_first.fill_(0.1)
        y = m(k, v, total_tokens=L)[0].numpy().astype(np.float64)

    w_kernel = float(F.softplus(torch.tensor(0.2))) / L
    u_kernel = 0.1 / L
    ref = _brute_force(k[0].numpy().astype(np.float64),
                       v[0].numpy().astype(np.float64), w_kernel, u_kernel)
    rel = np.abs(y - ref).max() / (np.abs(ref).max() + 1e-12)
    assert rel < 1e-5, f"回退版与暴力参考不符, 相对误差 {rel:.2e}"
    print(f"  [ok] 纯 PyTorch 回退版 == 暴力参考 (相对误差 {rel:.2e})")


def test_uniform_weight_limit():
    """衰减与奖励都接近 0 时, 输出应接近 v 的等权平均。"""
    L, C = 6, 1
    k = torch.zeros(1, L, C)
    v = torch.arange(1.0, L + 1).view(1, L, 1)
    m = TorchBiWKVSingleNorm(C)
    with torch.no_grad():
        m.spatial_decay.fill_(-20.0)   # softplus(-20) ≈ 0
        m.spatial_first.fill_(-20.0)
        y = m(k, v, total_tokens=L)[0, :, 0].numpy()
    expect = float(v.mean())
    assert abs(y.mean() - expect) < 0.5, f"均匀权重下应接近 {expect:.2f}, 实为 {y.mean():.2f}"
    print(f"  [ok] 均匀权重极限: 输出均值 {y.mean():.2f} ≈ v 均值 {expect:.2f}")


def test_single_norm_differs_from_legacy():
    """单归一化双向与旧实现 (双比值平均) 不等价 —— 这是本轮修正的依据。

    两者若被当成可互换, 会让"换实现"与"改模型"混为一谈, 因此固化为断言。
    """
    torch.manual_seed(1)
    L, C = 64, 4
    k = torch.randn(1, L, C)
    v = torch.randn(1, L, C)
    legacy = FastBiWKV(C)
    new = TorchBiWKVSingleNorm(C)
    new.load_state_dict(legacy.state_dict())
    with torch.no_grad():
        y_legacy = legacy(k, v, total_tokens=L)
        y_new = new(k, v, total_tokens=L)
    rel = (y_legacy - y_new).abs().max().item() / (y_new.abs().max().item() + 1e-12)
    assert rel > 0.1, f"两种定义差异应显著, 实测 {rel:.3f}"
    print(f"  [ok] 单归一化 vs 旧双比值平均 差异 {rel:.3f} (确认不等价)")


def test_backward():
    """反传正常: k、v 与两个参数都有梯度。"""
    torch.manual_seed(2)
    L, C = 32, 4
    k = torch.randn(1, L, C, requires_grad=True)
    v = torch.randn(1, L, C, requires_grad=True)
    m = TorchBiWKVSingleNorm(C)
    m(k, v, total_tokens=L).sum().backward()
    for name, t in (("k", k), ("v", v),
                    ("spatial_decay", m.spatial_decay), ("spatial_first", m.spatial_first)):
        assert t.grad is not None and t.grad.abs().sum() > 0, f"{name} 无梯度"
    print("  [ok] 反传正常 (k/v/decay/first 均有梯度)")


def test_cuda_matches_torch_fallback():
    """CUDA 版与纯 PyTorch 回退版一致 (算子不可用时跳过)。"""
    if not cuda_available():
        print("  [skip] CUDA 算子不可用")
        return
    torch.manual_seed(3)
    L, C, B = 32, 32, 2
    k = torch.randn(B, L, C, device="cuda")
    v = torch.randn(B, L, C, device="cuda")
    a = CudaBiWKV(C).cuda()
    b = TorchBiWKVSingleNorm(C).cuda()
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        ya = a(k, v, total_tokens=L)
        yb = b(k, v, total_tokens=L)
    rel = (ya - yb).abs().max().item() / (ya.abs().max().item() + 1e-12)
    assert rel < 5e-3, f"CUDA 版与回退版不一致, 相对误差 {rel:.2e}"
    print(f"  [ok] CUDA 版 == 纯 PyTorch 回退版 (相对误差 {rel:.2e})")


def test_cuda_matches_brute_force():
    """CUDA 版与暴力参考一致 (判定官方核定义是否被正确复现)。"""
    if not cuda_available():
        print("  [skip] CUDA 算子不可用")
        return
    torch.manual_seed(4)
    L, C = 32, 32          # C 必须是 16 的倍数才会走 CUDA 内核 (否则回退到 PyTorch 版)
    k = torch.randn(1, L, C, device="cuda")
    v = torch.randn(1, L, C, device="cuda")
    m = CudaBiWKV(C).cuda()
    with torch.no_grad():
        m.spatial_decay.fill_(0.2)
        m.spatial_first.fill_(0.1)
        y = m(k, v, total_tokens=L)[0].cpu().numpy().astype(np.float64)
    w_kernel = float(F.softplus(torch.tensor(0.2))) / L
    ref = _brute_force(k[0].cpu().numpy().astype(np.float64),
                       v[0].cpu().numpy().astype(np.float64), w_kernel, 0.1 / L)
    rel = np.abs(y - ref).max() / (np.abs(ref).max() + 1e-12)
    assert rel < 1e-5, f"CUDA 版与暴力参考不符, 相对误差 {rel:.2e}"
    print(f"  [ok] CUDA 版 == 暴力参考 (相对误差 {rel:.2e})")


def test_mode_switch_in_head():
    """空间头的 wkv_mode 开关: legacy 与 single_norm 都能前向, 且参数名一致。"""
    from models.golf_v7r.spatial_rwkv_head_v8 import SpatialRWKVHeadV8
    torch.manual_seed(5)
    x = torch.randn(1, 32, 8, 8)
    legacy = SpatialRWKVHeadV8(32, wkv_mode="legacy")
    new = SpatialRWKVHeadV8(32, wkv_mode="single_norm")
    for name, m in (("legacy", legacy), ("single_norm", new)):
        with torch.no_grad():
            y = m(x, x)
        assert y.shape == x.shape, f"{name}: 形状 {y.shape}"
    # 两类 WKV 模块的参数名相同, 便于直接对照
    a = set(n.split(".")[-1] for n, _ in legacy.bi_wkv_list[0].named_parameters())
    b = set(n.split(".")[-1] for n, _ in new.bi_wkv_list[0].named_parameters())
    assert a == b, f"参数名不一致: {a} vs {b}"
    print("  [ok] wkv_mode 开关可用 (两种实现参数名一致)")


def main():
    print("v8 WKV 实现单测 (单归一化双向):")
    test_torch_fallback_matches_brute_force()
    test_uniform_weight_limit()
    test_single_norm_differs_from_legacy()
    test_backward()
    test_cuda_matches_torch_fallback()
    test_cuda_matches_brute_force()
    test_mode_switch_in_head()
    print("\n全部通过 ✅")


if __name__ == '__main__':
    main()
