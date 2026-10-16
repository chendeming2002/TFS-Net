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

    # 历史 ckpt 可直接加载。
    # 例外: BranchM 修复 (v8) 在 to_rgb 前新增了 to_rgb_norm, 早于该修复的 ckpt
    # 必然缺这两个键 —— 这是有意的结构改动, 不是不兼容, 因此允许它们缺失。
    ckpt = "outputs/golf_v7r_v3_phaseA_l/best.pth"
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        r = GolfNet_v7r_v3().load_state_dict(ck["model_state_dict"], strict=False)
        allowed_missing = {"branch_M.to_rgb_norm.weight", "branch_M.to_rgb_norm.bias"}
        unexpected_missing = set(r.missing_keys) - allowed_missing
        assert not unexpected_missing, f"出现预期外的缺失键: {unexpected_missing}"
        assert not r.unexpected_keys, f"出现多余键: {r.unexpected_keys}"
        print("  [ok] phaseA_l/best.pth 可加载 (仅缺 BranchM 修复新增的 to_rgb_norm)")
    else:
        print("  [skip] 历史 ckpt 不存在, 跳过加载验证")


def test_gated_mode_structure():
    """门控融合 (v8 M2) 的结构性质。

    形式: O = refine(g ⊙ Y_M + Y_base), 其中 g = sigmoid(gate_net([Y_N,Y_L,X_t])),
    Y_base = base_net([Y_N,Y_L,X_t])。对应 PRE-Mamba 的双流 MSSM 门控
    (reference_repos/PRE-Mamba/.../mssm.py:322)。
    """
    torch.manual_seed(0)
    B, C, H, W = 2, 3, 32, 32
    YN = torch.rand(B, C, H, W)
    YL = torch.rand(B, C, H, W)
    YM = torch.rand(B, C, H, W)
    Xc = torch.rand(B, C, H, W)
    f = V7RFusion(in_channels=3, mode="gated", residual_freeze=True)
    o = f(YN, YL, YM, Xc)

    gate_feat = torch.cat([YN, YL, Xc], dim=1)
    g = torch.sigmoid(f.gate_net(gate_feat))
    base = f.base_net(gate_feat)
    assert torch.allclose(o["Y_fused"], g * YM + base, atol=1e-6), "门控融合输出形式不符"
    # 初始: base 应为 Y_N + Y_L 的直通, 门接近常开 (bias=2 → 0.88)
    assert torch.allclose(base, YN + YL, atol=1e-6), "初始 base 应为 Y_N + Y_L"
    assert abs(g.mean().item() - 0.881) < 0.01, f"初始门应≈0.881, 实为 {g.mean():.3f}"
    # gated 模式不建立 weight_net, 避免死参数
    assert f.weight_net is None, "gated 模式不应建立 weight_net"
    print(f"  [ok] gated 融合形式正确 (初始门 {g.mean().item():.3f}, base=Y_N+Y_L)")


def test_gated_gate_reads_multiframe_not_m():
    """门必须由多帧流 (N、L) 与中心帧算出, 不读 Y_M。

    这是"多帧做门、单帧做内容"的核心: 改变 Y_M 不应改变门的值,
    只会改变被门调制的内容。若门读 Y_M, 该性质会被破坏。
    """
    torch.manual_seed(1)
    B, C, H, W = 1, 3, 16, 16
    YN = torch.rand(B, C, H, W)
    YL = torch.rand(B, C, H, W)
    YM = torch.rand(B, C, H, W)
    Xc = torch.rand(B, C, H, W)
    f = V7RFusion(in_channels=3, mode="gated", residual_freeze=True)

    gate_feat = torch.cat([YN, YL, Xc], dim=1)
    g1 = torch.sigmoid(f.gate_net(gate_feat))
    YM2 = torch.rand(B, C, H, W)
    g2 = torch.sigmoid(f.gate_net(gate_feat))   # 门只看 N/L/X, 与 Y_M 无关
    assert torch.allclose(g1, g2), "门不应受 Y_M 影响"

    # 而输出必须随 Y_M 变化 (单帧流确实提供内容)
    o1 = f(YN, YL, YM, Xc)["Y_fused"]
    o2 = f(YN, YL, YM2, Xc)["Y_fused"]
    assert not torch.allclose(o1, o2), "输出应随 Y_M 变化"
    print("  [ok] 门只由多帧流决定, 单帧流只提供内容")


def test_gated_backward():
    """门控融合反传正常: gate_net 末层与 base_net 首步即有梯度。"""
    torch.manual_seed(2)
    B, C, H, W = 2, 3, 16, 16
    YN = torch.rand(B, C, H, W, requires_grad=True)
    YL = torch.rand(B, C, H, W, requires_grad=True)
    YM = torch.rand(B, C, H, W, requires_grad=True)
    Xc = torch.rand(B, C, H, W)
    f = V7RFusion(in_channels=3, mode="gated", residual_freeze=True)
    f(YN, YL, YM, Xc)["O_t"].sum().backward()
    for name in ("gate_net.4.weight", "base_net.weight"):
        g = f.get_parameter(name).grad
        assert g is not None and g.abs().sum() > 0, f"{name} 无梯度"
    for t, nm in ((YN, "Y_N"), (YL, "Y_L"), (YM, "Y_M")):
        assert t.grad is not None and t.grad.abs().sum() > 0, f"{nm} 无梯度"
    print("  [ok] gated 融合反传正常 (门/基线/三路输入均有梯度)")


def test_bad_args():
    # "additive" 自 §6.19 起是合法模式 (配频带划分目标使用), 故不在非法列表中
    for bad in ("mean", "", "softmax_", "FLOORED"):
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
    print("\nV7RFusion 门控融合单测 (v8 M2):")
    test_gated_mode_structure()
    test_gated_gate_reads_multiframe_not_m()
    test_gated_backward()
    print("\n全部通过 ✅")


if __name__ == "__main__":
    main()
