#!/usr/bin/env python3
"""SGRFv10 (逐段约束) 的结构检查。

验证:
  1. 三段限幅各自独立, L 段不限幅;
  2. 三段幅度范围各自独立, L 更宽;
  3. 门控初值 0.5, 无饱和;
  4. 限幅统计按段记录, L 段压缩比例恒为 0;
  5. 全模型 (g + v10) 前向/反向/FP16 正常。

运行:
    PYTHONPATH=. python tests/test_sgrf_v10.py
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.golf_v7r.sgrf_v9 import SGRFv9


class _Fixed(nn.Module):
    def __init__(self, key, v):
        super().__init__()
        self.key, self.v = key, v

    def forward(self, f, *args):
        B, _, H, W = f.shape
        return {self.key: torch.full((B, 3, H, W), self.v, device=f.device)}


def test_per_stage_bounds():
    sg = SGRFv9(feat_channels=4)
    assert sg.delta_bound["N"] == 0.60
    assert sg.delta_bound["M"] == 0.16
    assert sg.delta_bound["L"] is None, "L 段不应有限幅"
    assert sg.amp_range["L"] == 0.8
    assert sg.amp_range["N"] == 0.5
    assert sg.amp_range["M"] == 0.5
    print("  [ok] 三段限幅与幅度范围独立")


def test_amp_respects_per_stage_range():
    sg = SGRFv9(feat_channels=4)
    with torch.no_grad():
        sg.stage_amp.fill_(10.0)
        a = sg._amp_all()
    n, l, m = [float(a[sg.stage_order.index(k)]) for k in "NLM"]
    assert abs(l - 1.4) < 1e-5, f"L 幅度上界应为 1.4, 实测 {l}"
    assert abs(n - 1.25) < 1e-5, f"N 幅度上界应为 1.25, 实测 {n}"
    print(f"  [ok] 幅度上界 L {l:.3f}, N/M {n:.3f}")


def test_l_stage_not_clipped():
    torch.manual_seed(1)
    sg = SGRFv9(feat_channels=4)
    feats = {k: torch.randn(1, 4, 6, 6) for k in "NLM"}
    X = torch.rand(1, 3, 6, 6)
    # L 段输出大增量, 观察它是否被限幅
    bL = _Fixed("Y_L", 1.2)
    with torch.no_grad():
        o = sg(feats, X, _Fixed("Y_N", 0.0), bL, _Fixed("Y_M", 0.0),
               torch.zeros(1, 4, 3, 3))
    st = o["stage_stats"]
    assert float(st["frac_compress_L"]) == 0.0, "L 段不应进入压缩统计"
    # L 段不限幅: 输出的最大幅度应等于 L 分支的输出幅度
    assert abs(float(o["Y_L"].abs().max()) - 1.2) < 1e-5
    print(f"  [ok] L 段未限幅 (输出幅度 {float(o['Y_L'].abs().max()):.2f}, 压缩比例 0)")


def test_gate_init():
    torch.manual_seed(2)
    sg = SGRFv9(feat_channels=4)
    feats = {k: torch.randn(1, 4, 6, 6) for k in "NLM"}
    with torch.no_grad():
        o = sg(feats, torch.rand(1, 3, 6, 6), _Fixed("Y_N", 0.1),
               _Fixed("Y_L", 0.1), _Fixed("Y_M", 0.1), torch.zeros(1, 4, 3, 3))
    for n in "NLM":
        g = float(o["stage_stats"][f"gate_{n}"])
        sat = float(o["stage_stats"][f"gate_{n}_sat"])
        assert abs(g - 0.5) < 1e-4 and sat == 0.0
    print("  [ok] 门控初值 0.5, 无饱和")


def test_full_model_g_plus_v10():
    if not torch.cuda.is_available():
        print("  [skip] 无 GPU")
        return
    from models.golf_v7r import GolfNet_v8
    torch.manual_seed(3)
    m = GolfNet_v8(q_mode="cross_rwkv", summary_type="g",
                   fusion_mode="sgrf_v9").cuda()
    x = torch.randn(2, 5, 3, 128, 128, device="cuda")
    o = m(x)
    for k in ("final", "branch_N", "branch_L", "branch_M"):
        assert torch.isfinite(o[k]).all(), f"{k} 出现 NaN/Inf"
    (o["final"].mean() + o["ortho_loss"] + (o["gate_reg"] or 0)).backward()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        o2 = m(x)
    assert torch.isfinite(o2["final"].float()).all()
    print("  [ok] 全模型 (G + v10) 前向/反向/FP16 正常")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n全部 {len(tests)} 项通过")
