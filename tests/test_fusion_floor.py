#!/usr/bin/env python3
"""V7RFusion 融合算子梯度病理的回归测试 (§6.16)。

验证:
  1. `floored` 与 `softmax` 的**初始输出逐位相同** (受控对照前提);
  2. `floored` 的权重满足 **g_k ≥ ε 且 Σg = 1** (凸组合 + 硬下界);
  3. 塌缩场景下 `floored` 把分支梯度占比从 ~0 抬到 ε 决定的水平;
  4. 默认构造仍是 `softmax` → 历史 ckpt 可直接加载 (向后兼容);
  5. 两模式参数量相同 (纯算子替换, 不引入新参数)。

运行:
    PYTHONPATH=. python tests/test_fusion_floor.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.golf_v7r.fusion_v7r import V7RFusion
from models.golf_v7r import GolfNet_v7r_v3


def test_same_init():
    torch.manual_seed(0)
    YN, YL, YM, Xc = [torch.rand(1, 3, 32, 32) for _ in range(4)]
    mean = (YN + YL + YM) / 3.0
    a = V7RFusion(mode="softmax")(YN, YL, YM, Xc)
    b = V7RFusion(mode="floored", weight_floor=0.1)(YN, YL, YM, Xc)
    assert torch.allclose(a["O_t"], mean, atol=1e-5)
    assert torch.allclose(b["O_t"], mean, atol=1e-5)
    assert torch.allclose(a["O_t"], b["O_t"], atol=1e-6), "两模式初始输出必须一致"
    assert torch.allclose(a["weights"], b["weights"], atol=1e-6), "初始权重必须都是 1/3"
    print("  [ok] 两模式初始输出/权重逐位相同 (可做受控对照)")


def test_weight_floor_property():
    torch.manual_seed(1)
    YN, YL, YM, Xc = [torch.rand(1, 3, 32, 32) for _ in range(4)]
    for eps in (0.05, 0.10, 0.20):
        f = V7RFusion(mode="floored", weight_floor=eps)
        with torch.no_grad():  # 极端随机 logits, 逼出下界
            f.weight_net[-1].weight.normal_(0, 3.0)
            f.weight_net[-1].bias.normal_(0, 3.0)
        w = f(YN, YL, YM, Xc)["weights"]
        assert float(w.min()) >= eps - 1e-6, f"ε={eps}: min={float(w.min())}"
        assert abs(float(w.sum(dim=1).mean()) - 1.0) < 1e-4, "Σg 必须为 1 (凸组合)"
        print(f"  [ok] ε={eps}: min(w)={float(w.min()):.4f} ≥ ε, Σw={float(w.sum(dim=1).mean()):.4f}")


def _grad_share(mode, eps, logits):
    """用同一张量作三路输入, 返回 (权重, 最小路梯度/最大路梯度)。"""
    H = torch.rand(1, 3, 8, 8)
    Xc = torch.rand(1, 3, 8, 8)
    f = V7RFusion(mode=mode, weight_floor=eps)
    with torch.no_grad():
        f.weight_net[-1].weight.zero_()
        f.weight_net[-1].bias.copy_(torch.tensor(logits))
    YN, YL, YM = [H.clone().requires_grad_(True) for _ in range(3)]
    o = f(YN, YL, YM, Xc)
    w = o["weights"].mean(dim=(0, 2, 3)).detach()
    o["Y_fused"].sum().backward()
    g = [abs(float(t.grad.sum())) for t in (YN, YL, YM)]
    return [float(x) for x in w], min(g) / max(g)


def test_gradient_floor_under_collapse():
    """L 路被压到极低时, softmax 梯度占比→0, floored 保底。"""
    logits = [2.0, -6.0, 2.0]
    _, r_sm = _grad_share("softmax", 0.1, logits)
    assert r_sm < 0.01, f"softmax 应塌缩, 实测 {r_sm}"
    prev = 0.0
    for eps in (0.05, 0.10, 0.20):
        _, r = _grad_share("floored", eps, logits)
        assert r > r_sm * 10, f"ε={eps} 未有效抬升 (softmax={r_sm:.5f}, floored={r:.5f})"
        assert r > prev, "ε 越大保底应越强"
        prev = r
        print(f"  [ok] 塌缩场景: softmax L占比={r_sm:.5f} → floored(ε={eps}) {r:.4f}")
    # ε=1/3 时退化为固定均值 (梯度完全相等)
    _, r_max = _grad_share("floored", 1 / 3 - 1e-6, logits)
    assert r_max > 0.99, f"ε→1/3 应使梯度均衡, 实测 {r_max}"
    print(f"  [ok] ε→1/3 极限: L占比={r_max:.4f} (退化为等权平均)")


def test_backward_compat():
    """默认必须是 softmax, 且参数量与 floored 相同。"""
    m = GolfNet_v7r_v3()
    assert m.fusion.mode == "softmax", "默认 mode 必须保持 softmax 以兼容历史 ckpt"
    n_sm = sum(p.numel() for p in GolfNet_v7r_v3(fusion_mode="softmax").parameters())
    n_fl = sum(p.numel() for p in GolfNet_v7r_v3(fusion_mode="floored").parameters())
    assert n_sm == n_fl, "floored 不应引入额外参数"
    print(f"  [ok] 默认 softmax; 两模式参数量相同 ({n_sm/1e6:.4f}M)")

    # 历史 ckpt 可直接加载
    ckpt = "outputs/golf_v7r_v3_phaseA_l/best.pth"
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        r = GolfNet_v7r_v3().load_state_dict(ck["model_state_dict"])
        assert not r.missing_keys and not r.unexpected_keys, (r.missing_keys, r.unexpected_keys)
        print("  [ok] phaseA_l/best.pth 可直接加载 (无 missing/unexpected)")
    else:
        print("  [skip] 历史 ckpt 不存在, 跳过加载验证")


def test_bad_args():
    for bad in ("additive", "mean", ""):
        try:
            V7RFusion(mode=bad)
            raise AssertionError(f"mode={bad!r} 应被拒绝")
        except ValueError:
            pass
    for bad in (-0.1, 0.5, 1 / 3):
        try:
            V7RFusion(mode="floored", weight_floor=bad)
            raise AssertionError(f"weight_floor={bad} 应被拒绝")
        except ValueError:
            pass
    print("  [ok] 非法 mode / weight_floor 被拒绝")


def main():
    print("V7RFusion 权重下限单测 (§6.16):")
    test_same_init()
    test_weight_floor_property()
    test_gradient_floor_under_collapse()
    test_backward_compat()
    test_bad_args()
    print("\n全部通过 ✅")


if __name__ == "__main__":
    main()
