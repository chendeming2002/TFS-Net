#!/usr/bin/env python3
"""SGRFv8 (分阶段串接融合) 的单元测试。

对应 `models/golf_v7r/sgrf_v8.py`。验证三条设计原则在数值上成立:

  1. 加性串接: 每一段对最终输出的偏导恒为 1 (stage_scale 为初值时);
  2. 级间无 clamp: 中间累加结果可以超出 [0, 1], 只有最终输出被截断;
  3. 逐级输入: 第二、三段的输入依赖上一段的输出 (proj 非零后可观测到)。

另外验证: 初始化时与并行加性融合同起点、全模型前向/反向与 FP16 正常。

运行:
    PYTHONPATH=. python tests/test_sgrf_v8.py
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.golf_v7r.sgrf_v8 import SGRFv8


class _Const(nn.Module):
    """返回一个只依赖输入均值的分支, 便于检查依赖关系与偏导。"""

    def __init__(self, key, gain=1.0):
        super().__init__()
        self.key = key
        self.gain = nn.Parameter(torch.tensor(float(gain)))

    def forward(self, f, *args):
        B, C, H, W = f.shape
        # 输出依赖输入特征的空间均值, 这样上一段的变化能被下一段感知
        v = f.mean(dim=1, keepdim=True).expand(B, 3, H, W) * self.gain
        return {self.key: v}


def _make(order="NLM"):
    sg = SGRFv8(channels=4, stage_order=order)
    bN, bL, bM = _Const("Y_N", 0.1), _Const("Y_L", 0.2), _Const("Y_M", 0.3)
    return sg, bN, bL, bM


def test_additive_unit_partials():
    """∂O/∂Y_k = 1 对三段都成立 (未 clamp、stage_scale 初值)。"""
    torch.manual_seed(0)
    sg = SGRFv8(channels=4, clamp_output=False)
    feats = {k: torch.zeros(1, 4, 3, 3) for k in "NLM"}
    X = torch.zeros(1, 3, 3, 3)
    yN = torch.randn(1, 3, 3, 3, requires_grad=True)
    yL = torch.randn(1, 3, 3, 3, requires_grad=True)
    yM = torch.randn(1, 3, 3, 3, requires_grad=True)

    class Fixed(nn.Module):
        def __init__(self, key, t):
            super().__init__()
            self.key, self.t = key, t

        def forward(self, *a):
            return {self.key: self.t}

    out = sg(feats, X, Fixed("Y_N", yN), Fixed("Y_L", yL), Fixed("Y_M", yM),
             torch.zeros(1, 4, 2, 2))["O_t"]
    out.sum().backward()
    for name, t in (("N", yN), ("L", yL), ("M", yM)):
        g = t.grad
        assert torch.allclose(g, torch.ones_like(g), atol=1e-4), \
            f"{name} 段偏导不为 1: 均值 {g.mean().item():.4f}"
    print("  [ok] 三段偏导均为 1 (加性串接, 无衰减)")


def test_no_intermediate_clamp():
    """中间累加可以超出 [0, 1]; 最终输出被 clamp。"""
    torch.manual_seed(1)
    sg = SGRFv8(channels=4, clamp_output=True)

    class Big(nn.Module):
        def __init__(self, key, v):
            super().__init__()
            self.key, self.v = key, v

        def forward(self, f, *a):
            return {self.key: torch.full((f.shape[0], 3, *f.shape[-2:]), self.v)}

    feats = {k: torch.zeros(1, 4, 2, 2) for k in "NLM"}
    X = torch.full((1, 3, 2, 2), 0.5)
    out = sg(feats, X, Big("Y_N", 0.8), Big("Y_L", 0.6), Big("Y_M", -1.5),
             torch.zeros(1, 4, 1, 1))
    mid = out["stages"]["L"]                     # 0.8 + 0.6 = 1.4
    assert mid.max().item() > 1.0, f"中间结果被截断: {mid.max().item():.3f}"
    final = out["O_t"]                           # 1.4 - 1.5 = -0.1 → clamp 0
    assert torch.allclose(final, torch.full_like(final, 0.0), atol=1e-5), \
        f"最终输出错误: {final.mean().item():.3f}"
    print(f"  [ok] 中间结果 {mid.max().item():.2f} 未截断, 最终输出 {final.mean().item():.2f}")


def test_stage_input_depends_on_previous():
    """proj 非零后, 第二段的输入会随第一段的输出变化 (逐级输入)。"""
    torch.manual_seed(2)
    sg, bN, bL, bM = _make()
    with torch.no_grad():
        sg.proj_to_feat_1.weight.fill_(0.5)
    # 特征与中心帧都取非零值, 否则第一段输出恒为 0, 改增益观测不到任何变化
    feats = {k: torch.full((1, 4, 3, 3), 0.3) for k in "NLM"}
    X = torch.full((1, 3, 3, 3), 0.4)
    F2 = torch.zeros(1, 4, 2, 2)
    with torch.no_grad():
        a = sg(feats, X, bN, bL, bM, F2)["Y_L"]
        bN.gain.fill_(5.0)                      # 改变第一段的输出
        b = sg(feats, X, bN, bL, bM, F2)["Y_L"]
    diff = (a - b).abs().mean().item()
    assert diff > 1e-4, "第二段的输出未随第一段变化, 逐级输入未生效"
    print(f"  [ok] 第二段随第一段变化 (差 {diff:.4f}), 逐级输入生效")


def test_init_matches_parallel_additive():
    """初始化时 proj 为零, 三段输入就是各自的 TCA 特征, 输出 = X + ΣY_k。"""
    torch.manual_seed(3)
    sg, bN, bL, bM = _make()
    feats = {k: torch.randn(1, 4, 3, 3) for k in "NLM"}
    X = torch.rand(1, 3, 3, 3) * 0.2          # 只作为 L 段的辅助输入, 不进累加
    F2 = torch.zeros(1, 4, 2, 2)
    with torch.no_grad():
        out = sg(feats, X, bN, bL, bM, F2)
        par = (bN(feats["N"])["Y_N"] + bL(feats["L"])["Y_L"]
               + bM(feats["M"])["Y_M"]).clamp(0, 1)
    err = (out["O_t"] - par).abs().max().item()
    assert err < 1e-5, f"初始化时与并行加性融合不一致, 误差 {err:.2e}"
    print(f"  [ok] 初始化与并行加性融合一致 (误差 {err:.1e})")


def test_bad_order_rejected():
    try:
        SGRFv8(channels=4, stage_order="NNM")
    except ValueError:
        print("  [ok] 非法阶段顺序被拒绝")
        return
    raise AssertionError("非法阶段顺序未被拒绝")


def test_full_model_with_sgrf():
    if not torch.cuda.is_available():
        print("  [skip] 无 GPU")
        return
    from models.golf_v7r import GolfNet_v8
    torch.manual_seed(4)
    m = GolfNet_v8(q_mode="cross_rwkv", summary_type="lite",
                   fusion_mode="sgrf").cuda()
    x = torch.randn(2, 5, 3, 128, 128, device="cuda")
    o = m(x)
    for k in ("final", "branch_N", "branch_L", "branch_M"):
        assert torch.isfinite(o[k]).all(), f"{k} 出现 NaN/Inf"
    assert o["fusion_weights"] is None
    (o["final"].mean() + o["ortho_loss"]).backward()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        o2 = m(x)
    assert torch.isfinite(o2["final"].float()).all()
    print("  [ok] 全模型 (lite + SGRFv8) 前向/反向/FP16 正常")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n全部 {len(tests)} 项通过")
