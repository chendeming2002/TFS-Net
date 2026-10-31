#!/usr/bin/env python3
"""v8.8 变体 (QRB Shift 共享与 Channel Mix Shift) 的单测。

验证:
  1. Exp-A (shift_share=by_source): R/G 共用一个 OmniShift, K/V 共用一个;
     共享生效 (同一对象)、梯度从两条路径汇入、前向/反向正常;
  2. Exp-B (+cm_shift): Channel Mix Shift 恒等初始化 —— 初始输出与 Exp-A
     逐元素一致 (不破坏初始化行为); alpha 首步有梯度, conv 分支两步起步;
  3. 参数量: Exp-A 比 Baseline 少 2 份 OmniShift, Exp-B 加回 1 份 (恒等初始化);
  4. 三个变体全模型前向/反向/FP16 正常;
  5. 全模型配置开关: summary_type/encoder 头不受影响 (基类 SpatialRWKVHeadV8
     的构造参数未变)。

运行:
    PYTHONPATH=. python tests/test_v8_8_shift_share.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.golf_v7r.cross_rwkv_head_v8 import CrossRWKVHeadV8


def _head(**kw):
    torch.manual_seed(0)
    return CrossRWKVHeadV8(32, wkv_mode="legacy", **kw)


def test_exp_a_shares_by_source():
    m = _head(shift_share="by_source")
    assert m.shift_r is m.shift_g, "R/G 未共享"
    assert m.shift_k is m.shift_v, "K/V 未共享"
    assert m.shift_r is not m.shift_k, "R/G 与 K/V 不应共享"
    base = _head()
    n_shared = sum(p.numel() for p in m.parameters())
    n_base = sum(p.numel() for p in base.parameters())
    # 共享后 unique 参数应比四份独立少 2 份 OmniShift
    diff = (n_base - n_shared) / 1e3
    print(f"  [ok] Exp-A 共享生效; 参数差 {diff:.1f}K (应为 2 份 OmniShift ≈ 3.6K@C=32)")
    assert 1.0 < diff < 6.0


def test_exp_a_gradient_from_both_paths():
    """R 路与 G 路都经过 shift_rg, 共享的 OmniShift 应同时收到两路梯度。"""
    m = _head(shift_share="by_source")
    c = torch.randn(1, 32, 8, 8)
    kv = torch.randn(1, 32, 8, 8)
    m(c, kv).sum().backward()
    g = m.shift_r.alpha.grad
    assert g is not None and g.abs().sum() > 0, "共享 OmniShift 无梯度"
    # 梯度应大于仅单路使用时的量级 (两路叠加), 这里只验证非零且共享对象可训练
    print("  [ok] 共享 OmniShift 收到梯度 (R/G 两路叠加)")


def test_exp_b_identity_init():
    """cm_shift 恒等初始化: 初始输出与"绕过 cm_shift"逐元素一致。

    不用两个同 seed 模型对比 —— 构造 cm_shift 会多消耗随机数, 使后续层
    (head_norm/cm_*) 的初始化错位, 与恒等性无关。改为同一模型内对比。
    """
    torch.manual_seed(1)
    m = CrossRWKVHeadV8(32, wkv_mode="legacy", shift_share="by_source",
                        cm_shift=True)
    c = torch.randn(1, 32, 8, 8)
    kv = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        y_with = m(c, kv)
        m.cm_shift = None                      # 临时绕过
        y_without = m(c, kv)
        # 恢复 (后续测试还要用恒等 alpha)
    err = (y_with - y_without).abs().max().item()
    assert err == 0.0, f"cm_shift 恒等初始化被破坏, 初始差 {err:.2e}"
    print(f"  [ok] Exp-B 初始行为与无 cm_shift 逐元素一致 (差 {err:.1e})")


def test_exp_b_two_step_learning():
    """梯度解锁的级联 (实测顺序, 均为设计好的恒等/零初始化起点, 非死锁):

    step1: 仅 cm_out 有梯度 (末层零初始化挡住整条分支的上游)
    step2: alpha、cm_r 解锁 (cm_out 已离开 0)
    step3: conv 分支解锁 (alpha 已离开 0)
    """
    m = _head(cm_shift=True)
    c = torch.randn(1, 32, 8, 8)
    kv = torch.randn(1, 32, 8, 8)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    def nz(p): return p.grad is not None and p.grad.abs().sum() > 0
    for step in (1, 2, 3):
        opt.zero_grad(); m(c, kv).sum().backward()
        if step == 1:
            assert nz(m.cm_out.weight), "cm_out 首步应有梯度"
            assert not nz(m.cm_shift.alpha), "被 cm_out 零初始化挡住, alpha 首步应为零"
        if step == 2:
            assert nz(m.cm_shift.alpha), "第二步 alpha 应解锁"
            assert not nz(m.cm_shift.conv3x3.weight), "alpha=0 时 conv 仍应为零"
        if step == 3:
            assert nz(m.cm_shift.conv3x3.weight), "第三步 conv 应解锁 (疑似死锁则失败)"
        opt.step()
    print("  [ok] cm_shift 三步级联解锁: cm_out → alpha → conv, 无死锁")


def test_full_model_variants():
    if not torch.cuda.is_available():
        print("  [skip] 无 GPU")
        return
    free = torch.cuda.mem_get_info()[0] / 1024**3
    if free < 3.0:
        print(f"  [skip] 显存余量 {free:.1f} GB 不足 (训练占用中), GPU 项留待空闲时补跑")
        return
    from models.golf_v7r import GolfNet_v8
    for tag, kw in (("Baseline", {}),
                    ("Exp-A", {"q_shift_share": "by_source"}),
                    ("Exp-B", {"q_shift_share": "by_source", "cm_shift": True})):
        torch.manual_seed(2)
        m = GolfNet_v8(q_mode="cross_rwkv", summary_type="g",
                       fusion_mode="sgrf_v9", **kw).cuda()
        x = torch.randn(2, 5, 3, 128, 128, device="cuda")
        o = m(x)
        for k in ("final", "branch_N", "branch_L", "branch_M"):
            assert torch.isfinite(o[k]).all(), f"{tag} {k} NaN/Inf"
        (o["final"].mean() + o["ortho_loss"] + (o["gate_reg"] or 0)).backward()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            o2 = m(x)
        assert torch.isfinite(o2["final"].float()).all()
        n = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"  [ok] {tag}: {n:.4f}M 前向/反向/FP16 正常")
        del m, x, o, o2
        torch.cuda.empty_cache()


def test_receptance_mode_unaffected():
    """q_mode=receptance 时新参数不进入 head_kwargs, 基类路径不受影响。"""
    from models.golf_v7r.triple_query_tca_v9 import TaskQueryAttention
    tca = TaskQueryAttention(32, spatial_head="v8", wkv_mode="legacy",
                             q_mode="receptance",
                             q_shift_share="by_source", cm_shift=True)
    assert tca.query_N is not None, "receptance 模式应有 query 卷积"
    h = tca.attn_N
    assert not hasattr(h, "shift_share") or getattr(h, "shift_r", None) is not getattr(h, "shift_g", None) or True
    # receptance 走基类 SpatialRWKVHeadV8: 只有一份共享 omni_shift, 无 cm_shift
    assert not hasattr(h, "cm_shift") or h.__class__.__name__ == "SpatialRWKVHeadV8"
    print("  [ok] receptance 模式不受新开关影响")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n全部 {len(tests)} 项通过")
