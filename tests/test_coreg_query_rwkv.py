#!/usr/bin/env python3
"""CoReg-QRB 的结构、兼容性与梯度测试。

运行:
    PYTHONPATH=. python tests/test_coreg_query_rwkv.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.golf_v7r.coreg_query_rwkv import CoRegisteredQueryRWKVHead
from models.golf_v7r.cross_rwkv_head_v8 import CrossRWKVHeadV8
from models.golf_v7r.triple_query_tca_v9 import TaskQueryAttention


def _make_baseline_pair():
    torch.manual_seed(17)
    baseline = CrossRWKVHeadV8(32, wkv_mode="legacy")
    torch.manual_seed(17)
    coreg = CoRegisteredQueryRWKVHead(32, wkv_mode="legacy")
    return baseline, coreg


def test_identity_at_initialization():
    """四个新尺度为 0 时, 新块应与基线逐元素一致。"""
    baseline, coreg = _make_baseline_pair()
    q = torch.randn(1, 32, 9, 11)
    x = torch.randn(1, 32, 9, 11)
    baseline.eval()
    coreg.eval()
    with torch.no_grad():
        y0 = baseline(q, x)
        y1 = coreg(q, x)
    err = (y0 - y1).abs().max().item()
    assert err < 1e-6, f"初始行为不一致: {err:.3e}"
    print(f"  [ok] 初始与 CrossRWKVHeadV8 逐元素一致 ({err:.1e})")


def test_zero_scales_can_unlock_from_first_step():
    """零扰动初始化不能把新路径永久锁死：四个尺度首步都必须有梯度。"""
    torch.manual_seed(9)
    head = CoRegisteredQueryRWKVHead(32, wkv_mode="legacy")
    q = torch.randn(2, 32, 8, 8)
    x = torch.randn(2, 32, 8, 8)
    head(q, x).square().mean().backward()
    for name in ("read_scale", "gate_scale", "write_scale", "aligned_scale"):
        grad = getattr(head, name).grad
        assert grad is not None and grad.abs() > 0, f"{name} 首步无梯度"
    print("  [ok] 四个零初始化尺度首步均可学习")


def test_relation_roles_change_output():
    """Write/Read/Gate/Aligned 打开后必须改变输出。"""
    torch.manual_seed(1)
    head = CoRegisteredQueryRWKVHead(32, wkv_mode="legacy").eval()
    q = torch.randn(1, 32, 8, 8)
    x = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        y0 = head(q, x)
        head.write_scale.fill_(0.25)
        head.read_scale.fill_(0.20)
        head.gate_scale.fill_(0.15)
        head.aligned_scale.fill_(0.30)
        # 初始 aligned 打分为 0.5; 非零 scale 已能激活双路修正。
        y1 = head(q, x)
    diff = (y1 - y0).abs().mean().item()
    assert diff > 1e-4, f"关系条件化未生效: {diff:.3e}"
    assert torch.isfinite(y1).all()
    print(f"  [ok] K/R/G/Aligned 打开后输出变化 ({diff:.4f})")


def test_pair_path_receives_gradients_after_unlock():
    """尺度解锁后, 成对编码器、写入与对齐路由都应有梯度。"""
    torch.manual_seed(2)
    head = CoRegisteredQueryRWKVHead(32, wkv_mode="legacy")
    with torch.no_grad():
        head.write_scale.fill_(0.1)
        head.read_scale.fill_(0.1)
        head.gate_scale.fill_(0.1)
        head.aligned_scale.fill_(0.1)
    q = torch.randn(1, 32, 8, 8, requires_grad=True)
    x = torch.randn(1, 32, 8, 8, requires_grad=True)
    head(q, x).square().mean().backward()
    names = ("pair_in.weight", "pair_local.weight", "pair_to_k.weight",
             "pair_to_aligned.weight")
    params = dict(head.named_parameters())
    for name in names:
        grad = params[name].grad
        assert grad is not None and grad.abs().sum() > 0, f"{name} 无梯度"
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0
    print("  [ok] 成对关系、写入、对齐路由与两路输入梯度正常")


def test_query_pairing_is_spatially_sensitive():
    """保持 Context 不变并打乱 Query 空间位置, 关系路由应响应。"""
    torch.manual_seed(3)
    head = CoRegisteredQueryRWKVHead(32, wkv_mode="legacy").eval()
    with torch.no_grad():
        head.write_scale.fill_(0.3)
        head.aligned_scale.fill_(0.3)
    q = torch.randn(1, 32, 8, 8)
    x = torch.randn(1, 32, 8, 8)
    q_shuffle = q.flatten(2).flip(-1).reshape_as(q)
    with torch.no_grad():
        y0 = head(q, x)
        y1 = head(q_shuffle, x)
    diff = (y0 - y1).abs().mean().item()
    assert diff > 1e-3, f"空间成对关系不敏感: {diff:.3e}"
    print(f"  [ok] Query 空间打乱能被成对路由感知 ({diff:.4f})")


def test_rejects_mismatched_grids():
    head = CoRegisteredQueryRWKVHead(32, wkv_mode="legacy")
    try:
        head(torch.randn(1, 32, 8, 8), torch.randn(1, 32, 7, 8))
    except ValueError as exc:
        assert "同形" in str(exc)
    else:
        raise AssertionError("异形 Query/Context 未被拒绝")
    print("  [ok] 明确拒绝非同网格输入")


def test_tca_coreg_mode_forward_backward():
    torch.manual_seed(4)
    tca = TaskQueryAttention(
        32, spatial_head="v8", wkv_mode="legacy", q_mode="coreg",
        q_shift_share="by_source", cm_shift=True)
    multi = torch.randn(1, 32, 12, 12, requires_grad=True)
    single = torch.randn(1, 32, 12, 12, requires_grad=True)
    f_n, f_l, f_m, ortho = tca(multi, single)
    assert isinstance(tca.attn_N, CoRegisteredQueryRWKVHead)
    for feat in (f_n, f_l, f_m):
        assert feat.shape == multi.shape and torch.isfinite(feat).all()
    (f_n.mean() + f_l.mean() + f_m.mean() + ortho).backward()
    assert multi.grad is not None and multi.grad.abs().sum() > 0
    assert single.grad is not None and single.grad.abs().sum() > 0
    assert set(tca._last_coreg_stats) == {"N", "L", "M"}
    print("  [ok] TaskQueryAttention coreg 模式前向/反向正常")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n全部 {len(tests)} 项通过")
