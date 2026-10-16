#!/usr/bin/env python3
"""v8 B1 (RWKV 编码器) 与 C (RWKV-7 / Value Residual) 的单元测试。

对应:
    models/golf_v7r/rwkv_encoder_block.py   (B1, docs/v8/01-设计.md §四)
    models/golf_v7r/rwkv7_smoke.py          (C,  docs/v8/01-设计.md §3.4)

运行:
    PYTHONPATH=. python tests/test_v8_encoder_and_rwkv7.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.golf.encoder import SharedEncoder
from models.golf_v7r.rwkv_encoder_block import RWKVEncoderBlock
from models.golf_v7r.rwkv7_smoke import (
    RWKV7ChannelMixSmoke, RWKV7TimeMixSmoke, rwkv7_op)


def test_encoder_block_shape_and_interface():
    """RWKVEncoderBlock 与 NAFBlock 接口一致: 同形状进出, 可直接替换。"""
    for c in (32, 64):
        blk = RWKVEncoderBlock(c)
        x = torch.randn(2, c, 16, 16)
        y = blk(x)
        assert y.shape == x.shape, f"通道 {c}: 形状不符 {y.shape} vs {x.shape}"
    print("  [ok] RWKVEncoderBlock 与 NAFBlock 接口一致 (同形状进出)")


def test_encoder_block_loops():
    """num_loops 循环次数: 同为 1 次与 2 次输出应不同 (确认循环真的执行)。"""
    torch.manual_seed(0)
    x = torch.randn(1, 32, 8, 8)
    blk1 = RWKVEncoderBlock(32, num_loops=1).eval()
    blk2 = RWKVEncoderBlock(32, num_loops=2).eval()
    with torch.no_grad():
        y1 = blk1(x)
        y2 = blk2(x)
    assert not torch.allclose(y1, y2), "num_loops=2 与 1 输出相同, 循环未生效"
    print("  [ok] num_loops 循环次数生效")


def test_shared_encoder_block_type_switch():
    """SharedEncoder 的 block_type 开关: 两种类型都能跑通且输出三层金字塔形状一致。"""
    for bt in ("naf", "rwkv"):
        enc = SharedEncoder(block_type=bt)
        x = torch.randn(2, 3, 32, 32)
        f1, f2, f3 = enc(x)
        assert f1.shape == (2, 32, 32, 32), f"{bt}: F1 形状 {f1.shape}"
        assert f2.shape == (2, 64, 16, 16), f"{bt}: F2 形状 {f2.shape}"
        assert f3.shape == (2, 128, 8, 8), f"{bt}: F3 形状 {f3.shape}"
    print("  [ok] SharedEncoder 两种 block_type 输出形状一致")


def test_shared_encoder_bad_block_type():
    """非法 block_type 应报错。"""
    try:
        SharedEncoder(block_type="bogus")
    except ValueError:
        print("  [ok] 非法 block_type 被拒绝")
        return
    raise AssertionError("非法 block_type 未报错")


def test_encoder_rwkv_backward():
    """RWKV 编码器反传正常, 每层的主要投影都拿到梯度。"""
    torch.manual_seed(1)
    enc = SharedEncoder(block_type="rwkv")
    x = torch.randn(1, 3, 16, 16)
    f1, f2, f3 = enc(x)
    (f1.sum() + f2.sum() + f3.sum()).backward()
    # 主路径投影 (proj_q/k/v/out) 与 OmniShift 必须全部有梯度
    key_names = [n for n, _ in enc.named_parameters()
                 if any(k in n for k in ("proj_q", "proj_k", "proj_v", "proj_out",
                                         "omni_shift"))]
    assert key_names, "未找到预期的编码器参数"
    dead = [n for n in key_names
            if enc.get_parameter(n).grad is None
            or enc.get_parameter(n).grad.abs().sum() == 0]
    assert not dead, f"主路径参数无梯度: {dead}"
    print(f"  [ok] RWKV 编码器反传正常 ({len(key_names)} 个主路径参数均有梯度)")


def test_rwkv7_op_shape_and_state_recurrence():
    """rwkv7_op 的输出形状正确, 且随序列推进依赖历史 (不是逐位置独立)。"""
    torch.manual_seed(2)
    B, T, C = 2, 16, 64
    hs = 32
    r = torch.randn(B, T, C)
    w = torch.randn(B, T, C)
    k = torch.randn(B, T, C)
    v = torch.randn(B, T, C)
    a = torch.randn(B, T, C)
    b = torch.randn(B, T, C)
    out = rwkv7_op(r, w, k, v, a, b, hs)
    assert out.shape == (B, T, C), f"形状不符: {out.shape}"

    # 改末位的 k 不应影响前几位 (因果性/递推性): 状态是逐步积累的
    k2 = k.clone()
    k2[:, -1] += 5.0
    out2 = rwkv7_op(r, w, k2, v, a, b, hs)
    assert torch.allclose(out[:, :-1], out2[:, :-1], atol=1e-5), \
        "改动末位影响了前面的输出, 递推性被破坏"
    assert not torch.allclose(out[:, -1], out2[:, -1]), "末位改动未影响末位输出"
    print("  [ok] rwkv7_op 形状正确且具递推性 (改动末位不影响前位)")


def test_rwkv7_value_residual_first_layer_stores_v():
    """第一层保存 v_first, 不应用值残差; 其 v0/v1/v2 因此无梯度 (与参考一致)。"""
    torch.manual_seed(3)
    blk = RWKV7TimeMixSmoke(64, head_size=32, layer_id=0)
    x = torch.randn(1, 8, 64)
    out, v_first = blk(x, None)
    assert v_first is not None, "第一层未保存 v_first"
    out.sum().backward()
    for name in ("v0", "v1", "v2"):
        g = blk.get_parameter(name).grad
        assert g is None or g.abs().sum() == 0, f"第一层 {name} 不应有梯度"
    print("  [ok] 第一层保存 v_first, 值残差参数无梯度 (与参考实现一致)")


def test_rwkv7_value_residual_second_layer_receives_gradient():
    """第二层应用值残差, 其 v0/v1/v2 应获得梯度。"""
    torch.manual_seed(4)
    blk0 = RWKV7TimeMixSmoke(64, head_size=32, layer_id=0)
    blk1 = RWKV7TimeMixSmoke(64, head_size=32, layer_id=1)
    x = torch.randn(1, 8, 64)
    h, v_first = blk0(x, None)
    out, _ = blk1(h, v_first)
    out.sum().backward()
    for name in ("v0", "v1", "v2"):
        g = blk1.get_parameter(name).grad
        assert g is not None and g.abs().sum() > 0, f"第二层 {name} 无梯度"
    print("  [ok] 第二层值残差参数有梯度")


def test_rwkv7_value_residual_gate_initial_half():
    """v0 零初始化 → sigmoid(0) = 0.5, 即初始时首层与当前层 value 各占一半。"""
    torch.manual_seed(5)
    blk = RWKV7TimeMixSmoke(64, head_size=32, layer_id=1)
    gate = torch.sigmoid(blk.v0)
    assert abs(gate.mean().item() - 0.5) < 1e-6, f"初始门控应为 0.5, 实为 {gate.mean():.4f}"
    print("  [ok] 值残差初始门控 = 0.5")


def test_rwkv7_channel_mix_shape_and_backward():
    """Channel Mix 形状正确且全部参数有梯度。"""
    torch.manual_seed(6)
    cm = RWKV7ChannelMixSmoke(64)
    x = torch.randn(2, 16, 64)
    y = cm(x)
    assert y.shape == x.shape, f"形状不符: {y.shape}"
    y.sum().backward()
    dead = [n for n, p in cm.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"无梯度参数: {dead}"
    print("  [ok] Channel Mix 形状正确且全部参数有梯度")


def main():
    print("v8 B1 (RWKV 编码器) 单测:")
    test_encoder_block_shape_and_interface()
    test_encoder_block_loops()
    test_shared_encoder_block_type_switch()
    test_shared_encoder_bad_block_type()
    test_encoder_rwkv_backward()
    print("\nv8 C (RWKV-7 / Value Residual) 单测:")
    test_rwkv7_op_shape_and_state_recurrence()
    test_rwkv7_value_residual_first_layer_stores_v()
    test_rwkv7_value_residual_second_layer_receives_gradient()
    test_rwkv7_value_residual_gate_initial_half()
    test_rwkv7_channel_mix_shape_and_backward()
    print("\n全部通过 ✅")


if __name__ == '__main__':
    main()
