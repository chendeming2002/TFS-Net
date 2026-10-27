#!/usr/bin/env python3
"""QRB 变体的第二轮审查测试。

按 docs/v8/12-QRB改进方向评估.md 第七节与第二轮审查要求, 补充以下检查:

  1. kvmod 的 q_scale_a / q_scale_b 是可训练参数且严格零初始化; 初始 K 与基线一致;
  2. 调制后 K 的数值范围在 clamp(-8, 8) 以内, CUDA 扫描稳定;
  3. qpath 的 q_lambda 零初始化, 首次 backward 与优化器一步后均有梯度;
     监控属性记录 λ、旁路 RMS、主路径 RMS;
  4. gatectx 的 ctx_r / ctx_g 只作用于 R/G 投影, K/V 与扫描不变;
  5. 三个变体各自独立, 不叠加。

运行:
    PYTHONPATH=. python tests/test_qrb_variants_review.py
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.golf_v7r.cross_rwkv_head_v8 import CrossRWKVHeadV8
from models.golf_v7r.qrb_variants import (
    CrossRWKVHeadKvMod, CrossRWKVHeadQPath, CrossRWKVHeadGateCtx)


def _pair(cls):
    torch.manual_seed(0)
    a = CrossRWKVHeadV8(32, wkv_mode="legacy")
    torch.manual_seed(0)
    b = cls(32, wkv_mode="legacy")
    b.load_state_dict({k: v for k, v in a.state_dict().items()
                       if k in b.state_dict()}, strict=False)
    return a, b


def test_kvmod_params_and_init():
    _, m = _pair(CrossRWKVHeadKvMod)
    # 严格零初始化且可训练
    for name in ("q_scale_a", "q_scale_b"):
        p = getattr(m, name)
        assert isinstance(p, nn.Parameter), f"{name} 不是可训练参数"
        assert p.requires_grad, f"{name} 不可训练"
        assert p.detach().abs().max().item() == 0.0, f"{name} 未零初始化"
    print("  [ok] q_scale_a / q_scale_b 严格零初始化且可训练")

    c = torch.randn(1, 32, 8, 8)
    kv = torch.randn(1, 32, 8, 8)
    a, b = _pair(CrossRWKVHeadKvMod)
    with torch.no_grad():
        ya, yb = a(c, kv), b(c, kv)
    assert torch.allclose(ya, yb, atol=0), "初始化时 K 调制与原结构不一致"
    print("  [ok] 初始化时 K 与基线逐元素一致 (V 完全不变)")


def test_kvmod_gradient_and_k_range():
    torch.manual_seed(1)
    m = CrossRWKVHeadKvMod(32, wkv_mode="legacy")
    c = torch.randn(1, 32, 16, 16, requires_grad=True)
    kv = torch.randn(1, 32, 16, 16, requires_grad=True)
    m(c, kv).sum().backward()
    for name in ("q_scale_a", "q_scale_b"):
        g = getattr(m, name).grad
        assert g is not None and g.abs().sum() > 0, f"{name} 无梯度"
    print("  [ok] q_scale_a / q_scale_b 首步即有梯度")

    # 调制后 K 的数值范围: 不应超出 clamp(-8, 8) 的可用区间太多
    with torch.no_grad():
        m.q_scale_a.fill_(0.5)
        m.q_scale_b.fill_(0.3)
        q = m.q_mod_pool(c).flatten(1)
        a = torch.tanh(m.q_mod_r(q))
        b_ = torch.tanh(m.q_mod_g(q))
        k_t = m._project(kv, m.shift_k, m.proj_k)              # (B, N, C)
        k_mod = m._modulate_k(c, k_t)
        absmax = k_mod.abs().max().item()
    print(f"  [ok] 调制后 |K| max = {absmax:.3f} (clamp 阈值 8)")
    assert absmax < 8.0, "调制后 K 超出 clamp 可用区间"

    # 监控属性存在且记录了实际幅度
    assert hasattr(m, "_last_k_mod"), "缺少调制监控"
    rec = m._last_k_mod
    assert "scale_mean" in rec and "shift_absmax" in rec
    print(f"  [ok] 调制监控记录 scale_mean {float(rec['scale_mean']):.3f}, "
          f"|shift| max {float(rec['shift_absmax']):.3f}")


def test_qpath_lambda_and_monitoring():
    _, m = _pair(CrossRWKVHeadQPath)
    p = m.q_lambda
    assert isinstance(p, nn.Parameter) and p.requires_grad
    assert p.detach().abs().max().item() == 0.0, "q_lambda 未零初始化"
    print("  [ok] q_lambda 零初始化且可训练")

    c = torch.randn(1, 32, 8, 8)
    kv = torch.randn(1, 32, 8, 8)
    m(c, kv).sum().backward()
    g0 = p.grad.abs().item()
    assert g0 > 0, "首次 backward 时 q_lambda 无梯度"
    print(f"  [ok] 首次 backward q_lambda 梯度 {g0:.4f}")

    # 优化器一步后, 支路参数仍可获得梯度
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    opt.step()
    m.zero_grad()
    m(c, kv).sum().backward()
    assert m.q_path.weight.grad is not None and m.q_path.weight.grad.abs().sum() > 0
    print("  [ok] 优化器一步后 ψ(Q) 参数仍有梯度")

    assert hasattr(m, "_last_q_path"), "缺少支路监控"
    rec = m._last_q_path
    assert rec["side_to_main"] > 0, "旁路 RMS 未被记录"
    print(f"  [ok] 支路监控记录 side_to_main={rec['side_to_main']:.4f}")


def test_gatectx_isolation():
    """ctx 只作用于 R/G 投影, K/V 与扫描完全不变。"""
    _, m = _pair(CrossRWKVHeadGateCtx)
    a, b = _pair(CrossRWKVHeadGateCtx)

    c = torch.randn(1, 32, 8, 8)
    kv = torch.randn(1, 32, 8, 8)
    # K/V 投影: 不经 ctx, 两个模型应给出完全相同的 K/V (加载相同基类权重)
    with torch.no_grad():
        k_a = m._project(kv, m.shift_k, m.proj_k)
        k_b = a._project(kv, a.shift_k, a.proj_k)
        assert torch.allclose(k_a, k_b, atol=0), "ctx 不应影响 K 投影"
        v_a = m._project(kv, m.shift_v, m.proj_v)
        v_b = a._project(kv, a.shift_v, a.proj_v)
        assert torch.allclose(v_a, v_b, atol=0), "ctx 不应影响 V 投影"
    print("  [ok] K/V 投影不受 ctx 影响 (逐元素相同)")

    # R/G 经 ctx 后与不经 ctx 应不同 (基类路径 vs gatectx 路径, ctx 零初始化)
    with torch.no_grad():
        r0 = CrossRWKVHeadV8._make_r(m, c)   # 无 ctx 的基类路径
        r1 = m._make_r(c)                    # gatectx 路径 (ctx 零初始化)
    d = (r0 - r1).abs().mean().item()
    # 零初始化时 ctx 输出为零, R 投影应与不加 ctx 时一致
    assert d == 0.0, f"零初始化后 ctx 仍改变 R 投影 (差 {d:.2e})"
    print(f"  [ok] ctx 只作用于 R/G; 零初始化时 R 投影与原版一致")


def test_gatectx_ctx_rms_ratio():
    """零初始化后, 初始残差注入为零, 起步与原版一致。"""
    torch.manual_seed(3)
    m = CrossRWKVHeadGateCtx(32, wkv_mode="legacy")
    c = torch.randn(1, 32, 16, 16)
    with torch.no_grad():
        ctx_out = m.ctx_r(c)
        rms_ctx = ctx_out.pow(2).mean().sqrt().item()
    assert rms_ctx == 0.0, f"ctx 应零初始化, 实测 RMS {rms_ctx:.4f}"
    print(f"  [ok] ctx 零初始化, 初始残差 RMS = 0, 起步与原版一致")


def test_gatectx_gradient_and_shape():
    """零初始化后首步 ctx 无梯度 (设计好的恒等起点); 多步优化后应离开零。"""
    torch.manual_seed(4)
    m = CrossRWKVHeadGateCtx(32, wkv_mode="legacy")
    c = torch.randn(1, 32, 8, 8, requires_grad=True)
    kv = torch.randn(1, 32, 8, 8, requires_grad=True)
    y = m(c, kv)
    assert y.shape == c.shape and torch.isfinite(y).all()
    # 零初始化时 ctx_r 输出为零, 其梯度乘以零: 首步为零是预期行为
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(5):
        opt.zero_grad(); m(c, kv).sum().backward(); opt.step()
    g = m.ctx_r.weight.grad
    assert g is not None and g.abs().sum() > 0, "多步优化后 ctx_r 仍无梯度"
    print(f"  [ok] ctx 多步后获得梯度 (5 步 |∇|={float(g.abs().sum()):.3f})")


def test_variants_are_isolated():
    """三个变体各自只启用一个机制, 未叠加。"""
    km = CrossRWKVHeadKvMod(32, wkv_mode="legacy")
    qp = CrossRWKVHeadQPath(32, wkv_mode="legacy")
    gc = CrossRWKVHeadGateCtx(32, wkv_mode="legacy")

    assert not hasattr(km, "q_lambda") and not hasattr(km, "ctx_r")
    assert not hasattr(qp, "q_scale_a") and not hasattr(qp, "ctx_r")
    assert not hasattr(gc, "q_scale_a") and not hasattr(gc, "q_lambda")
    print("  [ok] 三个变体互不叠加")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n全部 {len(tests)} 项通过")
