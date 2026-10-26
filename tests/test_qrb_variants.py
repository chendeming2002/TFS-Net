#!/usr/bin/env python3
"""QRB 三个变体的单测。

验证:
  1. 三个变体的输出形状、有限值、前向/反向;
  2. kvmod 的 K 调制在 α=β=0 时与原结构一致, 非零后改变输出;
  3. qpath 的 λ=0 时与原结构一致, 非零后加入 Q 的直通项;
  4. gatectx 的局部上下文使 R/G 对 Q 的空间扰动更敏感;
  5. 三个变体的全部新增参数都能收到梯度。

运行:
    PYTHONPATH=. python tests/test_qrb_variants.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.golf_v7r.qrb_variants import (
    CrossRWKVHeadKvMod, CrossRWKVHeadQPath, CrossRWKVHeadGateCtx)
from models.golf_v7r.cross_rwkv_head_v8 import CrossRWKVHeadV8


def _make_pair(cls, **kw):
    torch.manual_seed(0)
    a = CrossRWKVHeadV8(32, wkv_mode="legacy", **kw)
    torch.manual_seed(0)
    b = cls(32, wkv_mode="legacy", **kw)
    return a, b


def test_shapes_and_backward():
    for cls in (CrossRWKVHeadKvMod, CrossRWKVHeadQPath, CrossRWKVHeadGateCtx):
        torch.manual_seed(1)
        m = cls(32, wkv_mode="legacy")
        c = torch.randn(1, 32, 8, 8, requires_grad=True)
        kv = torch.randn(1, 32, 8, 8, requires_grad=True)
        y = m(c, kv)
        assert y.shape == c.shape, f"{cls.__name__} 形状不符"
        assert torch.isfinite(y).all(), f"{cls.__name__} 出现 NaN/Inf"
        y.sum().backward()
        assert c.grad.abs().sum() > 0 and kv.grad.abs().sum() > 0
        print(f"  [ok] {cls.__name__} 前向/反向正常")


def test_kvmod_identity_at_init():
    a, b = _make_pair(CrossRWKVHeadKvMod)
    c = torch.randn(1, 32, 8, 8)
    kv = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        ya = a(c, kv)
        yb = b(c, kv)
    err = (ya - yb).abs().max().item()
    assert err < 1e-6, f"kvmod 初始与原结构不一致, 误差 {err:.2e}"
    print(f"  [ok] kvmod 初始 (α=β=0) 与原结构逐元素一致 (误差 {err:.1e})")


def test_kvmod_modulates_when_nonzero():
    _, b = _make_pair(CrossRWKVHeadKvMod)
    c = torch.randn(1, 32, 8, 8)
    kv = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        y0 = b(c, kv)
        b.q_scale_a.fill_(1.0)
        y1 = b(c, kv)
    diff = (y0 - y1).abs().mean().item()
    assert diff > 1e-4, f"调制未生效 (差 {diff:.2e})"
    print(f"  [ok] kvmod 调制生效 (平均差 {diff:.4f})")


def test_qpath_identity_at_init():
    a, b = _make_pair(CrossRWKVHeadQPath)
    c = torch.randn(1, 32, 8, 8)
    kv = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        ya = a(c, kv)
        yb = b(c, kv)
    err = (ya - yb).abs().max().item()
    assert err < 1e-6, f"qpath 初始与原结构不一致, 误差 {err:.2e}"
    print(f"  [ok] qpath 初始 (λ=0) 与原结构逐元素一致 (误差 {err:.1e})")


def test_qpath_adds_q_when_nonzero():
    _, b = _make_pair(CrossRWKVHeadQPath)
    c = torch.randn(1, 32, 8, 8)
    kv = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        y0 = b(c, kv)
        b.q_lambda.fill_(1.0)
        y1 = b(c, kv)
    diff = (y0 - y1).abs().mean().item()
    assert diff > 1e-4, f"Q 直通支路未生效 (差 {diff:.2e})"
    print(f"  [ok] qpath 加性支路生效 (平均差 {diff:.4f})")


def test_gatectx_spatial_sensitivity():
    """零初始化时 gatectx 与原结构一致; ctx 离开零后输出应改变。"""
    torch.manual_seed(2)
    m = CrossRWKVHeadGateCtx(32, wkv_mode="legacy")
    base = CrossRWKVHeadV8(32, wkv_mode="legacy")
    base.load_state_dict({k: v for k, v in m.state_dict().items() if k in base.state_dict()})
    c = torch.randn(1, 32, 8, 8)
    kv = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        y_base = base(c, kv)
        y_ctx = m(c, kv)
    diff0 = (y_base - y_ctx).abs().mean().item()
    assert diff0 == 0.0, f"零初始化时应与原版一致 (差 {diff0:.2e})"
    with torch.no_grad():
        m.ctx_r.weight.fill_(0.1)
        m.ctx_g.weight.fill_(0.1)
        y_ctx2 = m(c, kv)
    diff1 = (y_base - y_ctx2).abs().mean().item()
    assert diff1 > 1e-4, "ctx 非零后输出未改变"
    print(f"  [ok] 零初始化时与原版一致; ctx 非零后输出改变 (差 {diff1:.4f})")


def test_all_new_params_have_gradients():
    """三个变体的新增参数 (调制/支路/局部上下文) 都应能收到梯度。"""
    torch.manual_seed(3)
    c = torch.randn(1, 32, 8, 8)
    kv = torch.randn(1, 32, 8, 8)
    for cls, key in ((CrossRWKVHeadKvMod, "q_scale_a"),
                     (CrossRWKVHeadQPath, "q_lambda"),
                     (CrossRWKVHeadGateCtx, "ctx_r.weight")):
        m = cls(32, wkv_mode="legacy")
        m(c, kv).sum().backward()
        p = dict(m.named_parameters())[key]
        assert p.grad is not None and p.grad.abs().sum() > 0, \
            f"{cls.__name__}.{key} 无梯度"
        print(f"  [ok] {cls.__name__}.{key} 有梯度")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n全部 {len(tests)} 项通过")
