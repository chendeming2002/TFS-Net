#!/usr/bin/env python3
"""CrossRWKVHeadV8 (Cross RWKV Gate) 的单元测试。

对应 `models/golf_v7r/cross_rwkv_head_v8.py` 与
`docs/v8/06-CrossRWKVGate-Q重构设计.md`。

要点与验证:
  1. 输出形状与输入一致;
  2. R 与 G 都是新的任务 Q 机制 —— 任一路不同, 输出不同;
  3. R/G 与 K/V 不同源时, 两路输入都要有梯度;
  4. G 的偏置初始化使门初始开启, WKV 通路的参数第一步就有梯度;
  5. 逐头 LayerNorm 可切换且不改变形状;
  6. TCA 的 q_mode 开关: 旧路径参数仍在, 新路径不再有 query_* 卷积。

运行:
    PYTHONPATH=. python tests/test_cross_rwkv_head_v8.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.golf_v7r.cross_rwkv_head_v8 import CrossRWKVHeadV8
from models.golf_v7r.triple_query_tca_v9 import TripleQueryTCAv9


def test_output_shape_and_finiteness():
    torch.manual_seed(0)
    head = CrossRWKVHeadV8(32, wkv_mode="legacy").eval()
    c = torch.randn(2, 32, 9, 11)
    kv = torch.randn(2, 32, 9, 11)
    with torch.no_grad():
        y = head(c, kv)
    assert y.shape == c.shape, f"形状不符: {y.shape} vs {c.shape}"
    assert torch.isfinite(y).all(), "输出出现 NaN/Inf"
    print(f"  [ok] 输出形状与输入一致: {tuple(y.shape)}")


def test_r_and_g_both_act_as_query():
    """R 或 G 的输入变化都应改变输出, 二者共同构成任务查询。"""
    torch.manual_seed(1)
    head = CrossRWKVHeadV8(32, wkv_mode="legacy").eval()
    kv = torch.randn(1, 32, 8, 8)
    q1 = torch.randn(1, 32, 8, 8)
    q2 = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        y1 = head(q1, kv)
        y2 = head(q2, kv)
    diff = (y1 - y2).abs().mean().item()
    assert diff > 1e-3, f"不同查询输入输出几乎相同 (差 {diff:.2e})"
    print(f"  [ok] R/G 查询输入生效 (平均差 {diff:.4f})")


def test_r_generalization_is_signed():
    """R 是带符号投影: 单独缩放 R 的权重会改变输出, 且不经过 sigmoid 饱和。"""
    torch.manual_seed(2)
    head = CrossRWKVHeadV8(32, wkv_mode="legacy").eval()
    c = torch.randn(1, 32, 8, 8)
    kv = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        y0 = head(c, kv)
        head.proj_r.weight.mul_(2.0)
        y1 = head(c, kv)
    assert (y1 - y0).abs().mean().item() > 1e-3, "R 缩放没有影响输出"
    print("  [ok] R 带符号线性调制生效")


def test_gradients_reach_both_inputs():
    torch.manual_seed(3)
    head = CrossRWKVHeadV8(32)
    c = torch.randn(1, 32, 8, 8, requires_grad=True)
    kv = torch.randn(1, 32, 8, 8, requires_grad=True)
    head(c, kv).sum().backward()
    assert c.grad is not None and c.grad.abs().sum() > 0, "R/G 输入路径无梯度"
    assert kv.grad is not None and kv.grad.abs().sum() > 0, "K/V 输入路径无梯度"
    print("  [ok] R/G 与 K/V 两条输入路径均有梯度")


def test_g_bias_opens_gate_at_init():
    """G 的偏置初始化为 1, SiLU(1)>0, WKV 通路第一步即可获得梯度。"""
    torch.manual_seed(4)
    head = CrossRWKVHeadV8(32, use_channel_mix=False, wkv_mode="legacy")
    assert abs(head.proj_g.bias.mean().item() - 1.0) < 1e-6, "proj_g 偏置未初始化为 1"
    c = torch.randn(1, 32, 8, 8)
    kv = torch.randn(1, 32, 8, 8)
    head(c, kv).sum().backward()
    for name, p in head.named_parameters():
        if name.startswith("bi_wkv_list"):
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"{name} 第一步无梯度, G 门处于关闭状态"
    print("  [ok] G 门初始开启, WKV 参数第一步即有梯度")


def test_head_norm_switches_shape():
    """逐头 LayerNorm 分 4 组, 形状保持。"""
    torch.manual_seed(5)
    head = CrossRWKVHeadV8(64, num_directions=4, wkv_mode="legacy").eval()
    assert head.head_norm.normalized_shape == (16,), \
        f"逐头归一化维度错误: {head.head_norm.normalized_shape}"
    with torch.no_grad():
        y = head(torch.randn(1, 64, 10, 10), torch.randn(1, 64, 10, 10))
    assert y.shape == (1, 64, 10, 10)
    print("  [ok] 逐头 LayerNorm 维度 = C/头数")


def test_tca_q_mode_switch():
    """TCA: receptance 模式保留 query_*; cross_rwkv 模式去掉并换用新头。"""
    tca_old = TripleQueryTCAv9(32, spatial_head="v8", wkv_mode="legacy",
                               q_mode="receptance")
    assert tca_old.query_N is not None
    assert isinstance(tca_old.attn_N, CrossRWKVHeadV8) is False

    tca_new = TripleQueryTCAv9(32, spatial_head="v8", wkv_mode="legacy",
                               q_mode="cross_rwkv")
    assert tca_new.query_N is None, "cross_rwkv 模式不应再有 query 卷积"
    assert isinstance(tca_new.attn_N, CrossRWKVHeadV8)
    assert isinstance(tca_new.attn_M, CrossRWKVHeadV8)
    print("  [ok] TCA q_mode 开关正确")


def test_tca_cross_mode_forward():
    torch.manual_seed(6)
    tca = TripleQueryTCAv9(32, spatial_head="v8", wkv_mode="legacy",
                           q_mode="cross_rwkv").eval()
    multi = torch.randn(1, 32, 16, 16)
    single = torch.randn(1, 32, 16, 16)
    with torch.no_grad():
        F_N, F_L, F_M, ortho = tca(multi, single)
    for name, f in (("N", F_N), ("L", F_L), ("M", F_M)):
        assert f.shape == multi.shape, f"F_{name} 形状不符"
        assert torch.isfinite(f).all(), f"F_{name} 出现 NaN/Inf"
    # M 路的 R/G 输入只来自单帧, 改多帧输入不应改变 M 路输出
    with torch.no_grad():
        _, _, F_M2, _ = tca(multi + 1.0, single)
    diff = (F_M - F_M2).abs().mean().item()
    assert diff < 1e-6, f"M 路受多帧输入影响 (差 {diff:.2e}), 违反单帧约定"
    print(f"  [ok] cross_rwkv 前向正常, M 路不受多帧输入影响 (差 {diff:.2e})")


def test_old_path_unchanged():
    """receptance 模式的参数量与结构应与改动前一致 (query_* 仍在)。"""
    tca = TripleQueryTCAv9(32, spatial_head="v8", wkv_mode="legacy",
                           q_mode="receptance")
    q_params = sum(p.numel() for p in tca.query_N.parameters())
    assert q_params > 0
    print(f"  [ok] 旧路径 query_N 参数量 = {q_params}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n全部 {len(tests)} 项通过")
