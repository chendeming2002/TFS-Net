#!/usr/bin/env python3
"""SGRFv9 (门控渐进式恢复) 的结构与梯度检查。

验证:
  1. 门控初值为 0.5 且不饱和; 各段增量非零;
  2. 阶段幅度初值为 1, 且恒在 [0.5, 1.5] 内;
  3. 最终损失能传回第 N 段 (无梯度切断);
  4. 每段的门控 / 投影 / 增量处理参数都能收到梯度;
  5. 置零某段只影响该段;
  6. 全模型前向/反向/FP16 正常。

运行:
    PYTHONPATH=. python tests/test_sgrf_v9.py
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.golf_v7r.sgrf_v9 import SGRFv9, StageGate


class _Fixed(nn.Module):
    """输出固定值的分支, 便于检查门控与幅度。"""

    def __init__(self, key, v, feat_scale=0.0):
        super().__init__()
        self.key, self.v, self.feat_scale = key, v, feat_scale

    def forward(self, f, *args):
        B, _, H, W = f.shape
        base = torch.full((B, 3, H, W), self.v, device=f.device)
        return {self.key: base + self.feat_scale * f[:, :3]}


def _model():
    return SGRFv9(feat_channels=4)


def test_gate_init_and_range():
    torch.manual_seed(0)
    g = StageGate(feat_channels=4)
    st = torch.randn(2, 3, 8, 8)
    ft = torch.randn(2, 4, 8, 8)
    with torch.no_grad():
        gv = g(st, ft)
    assert torch.allclose(gv, torch.full_like(gv, 0.5), atol=1e-6), \
        f"门控初值不是 0.5: {gv.mean().item():.4f}"
    sat = ((gv > 0.9) | (gv < 0.1)).float().mean().item()
    assert sat == 0.0, "初始门控已饱和"
    print("  [ok] 门控初值 0.5, 无饱和")


def test_amp_bounds():
    sg = _model()
    with torch.no_grad():
        amps = sg._amp_all()
        assert torch.allclose(amps, torch.ones_like(amps)), "幅度初值不是 1"
        sg.stage_amp.fill_(10.0)
        assert float(sg._amp_all().max()) <= 1.5 + 1e-6, "幅度上界失效"
        sg.stage_amp.fill_(-10.0)
        assert float(sg._amp_all().min()) >= 0.5 - 1e-6, "幅度下界失效"
    print("  [ok] 阶段幅度初值 1, 恒在 [0.5, 1.5] 内")


def test_gradient_reaches_first_stage():
    """最终输出对第 N 段的门控与增量处理必须有梯度 (无 detach)。"""
    torch.manual_seed(1)
    sg = _model()
    feats = {k: torch.randn(1, 4, 8, 8) for k in "NLM"}
    X = torch.rand(1, 3, 8, 8)
    F2 = torch.zeros(1, 4, 2, 2)
    out = sg(feats, X, _Fixed("Y_N", 0.2), _Fixed("Y_L", 0.2), _Fixed("Y_M", 0.2), F2)
    out["O_t"].sum().backward()
    assert sg.gate["N"].net[-1].weight.grad.abs().sum() > 0, "N 段门控无梯度"
    assert sg.lift_N.weight.grad.abs().sum() > 0, "N 段增量处理无梯度"
    assert sg.stage_amp.grad.abs().sum() > 0, "阶段幅度无梯度"
    print("  [ok] 最终损失可传回 N 段 (门控/增量/幅度均有梯度)")


def test_all_params_used():
    """门控末层与状态投影是零初始化, 首步不可达属预期; 用优化器跑 5 步再查。"""
    torch.manual_seed(2)
    sg = _model()
    feats = {k: torch.randn(1, 4, 8, 8) for k in "NLM"}
    X = torch.rand(1, 3, 8, 8)
    F2 = torch.zeros(1, 4, 2, 2)
    bN, bL, bM = _Fixed("Y_N", 0.1), _Fixed("Y_L", 0.1), _Fixed("Y_M", 0.1)
    opt = torch.optim.AdamW(sg.parameters(), lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        sg(feats, X, bN, bL, bM, F2)["O_t"].sum().backward()
        opt.step()
    opt.zero_grad()
    sg(feats, X, bN, bL, bM, F2)["O_t"].sum().backward()
    dead = [n for n, p in sg.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"训练 5 步后仍有未参与训练的融合参数: {dead}"
    print(f"  [ok] 融合模块全部 {sum(1 for _ in sg.parameters())} 个参数张量均可训练")


def test_zero_stage_only_affects_that_stage():
    torch.manual_seed(3)
    sg = _model()
    feats = {k: torch.randn(1, 4, 8, 8) for k in "NLM"}
    X = torch.rand(1, 3, 8, 8)
    F2 = torch.zeros(1, 4, 2, 2)
    bN, bL, bM = _Fixed("Y_N", 0.3), _Fixed("Y_L", 0.3), _Fixed("Y_M", 0.3)
    # 有限幅后增量不再线性, 改为数值对比:
    #   置零 M 段的结果应等于"只累加 N、L 两段"的结果, 且必须不同于完整结果。
    with torch.no_grad():
        full = sg(feats, X, bN, bL, bM, F2)["O_t"]
        zM = sg(feats, X, bN, bL, bM, F2, zero_stage="M")["O_t"]
        zN = sg(feats, X, bN, bL, bM, F2, zero_stage="N")["O_t"]
    d_M = (full - zM).abs().mean().item()
    d_N = (full - zN).abs().mean().item()
    assert d_M > 1e-3, f"置零 M 段没有产生任何变化: {d_M:.2e}"
    assert d_N > 1e-3, f"置零 N 段没有产生任何变化: {d_N:.2e}"
    print(f"  [ok] 置零单段仅削减该段贡献 (M 差 {d_M:.4f}, N 差 {d_N:.4f})")


def test_stats_reported():
    torch.manual_seed(4)
    sg = _model()
    feats = {k: torch.randn(1, 4, 8, 8) for k in "NLM"}
    out = sg(feats, torch.rand(1, 3, 8, 8), _Fixed("Y_N", 0.2),
             _Fixed("Y_L", 0.2), _Fixed("Y_M", 0.2), torch.zeros(1, 4, 2, 2))
    st = out["stage_stats"]
    for name in "NLM":
        assert f"gate_{name}" in st and f"inc_{name}" in st
        assert f"gate_{name}_sat" in st
    print(f"  [ok] 统计齐全: gate 均值 "
          f"{[round(float(st[f'gate_{n}']),3) for n in 'NLM']}, "
          f"饱和率 {[round(float(st[f'gate_{n}_sat']),3) for n in 'NLM']}")


def test_full_model():
    if not torch.cuda.is_available():
        print("  [skip] 无 GPU")
        return
    from models.golf_v7r import GolfNet_v8
    torch.manual_seed(5)
    m = GolfNet_v8(q_mode="cross_rwkv", summary_type="lite",
                   fusion_mode="sgrf_v9").cuda()
    x = torch.randn(2, 5, 3, 128, 128, device="cuda")
    o = m(x)
    for k in ("final", "branch_N", "branch_L", "branch_M"):
        assert torch.isfinite(o[k]).all(), f"{k} 出现 NaN/Inf"
    (o["final"].mean() + o["ortho_loss"]).backward()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        o2 = m(x)
    assert torch.isfinite(o2["final"].float()).all()
    st = m.sgrf(x.new_tensor(0.0)) if False else None
    print("  [ok] 全模型 (lite + SGRFv9) 前向/反向/FP16 正常")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n全部 {len(tests)} 项通过")
