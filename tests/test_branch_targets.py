#!/usr/bin/env python3
"""分支专属监督目标的单元测试 (docs/v7/04-model-architecture.md 第四节)。

验证:
  1. 三个目标互不相同 —— 这是整个方案的前提, 若目标相同则等于什么都没改;
  2. 目标的物理性质符合设计 (时序均值比单帧噪声小; 光照归一化去掉了低频差异);
  3. 对齐搜索能找到正确位移 (用人工平移的合成片段验证);
  4. 损失函数在开关打开时真的用了三个不同目标, 关闭时与历史行为逐位一致;
  5. 目标缺失时回退到 GT 并告警, 不崩溃;
  6. 反传正常。

运行:
    PYTHONPATH=. python tests/test_branch_targets.py
"""
import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.branch_targets import (_best_global_shift, _integer_shift,
                                     build_branch_targets)
from models.golf_v7r.loss import GolfV7RLoss


def test_targets_are_distinct():
    """三个目标必须互不相同, 否则方案等价于什么都没改。"""
    torch.manual_seed(0)
    clip = torch.rand(5, 3, 64, 64)
    t = build_branch_targets(clip)

    for name in ('noise_target', 'illum_target', 'motion_target'):
        assert name in t, f"缺少 {name}"
        assert t[name].shape == (3, 64, 64), t[name].shape

    pairs = [('noise_target', 'illum_target'),
             ('noise_target', 'motion_target'),
             ('illum_target', 'motion_target')]
    for a, b in pairs:
        diff = (t[a] - t[b]).abs().mean().item()
        assert diff > 1e-3, f"{a} 与 {b} 几乎相同 (平均差 {diff:.2e})"
    print("  [ok] 三目标互不相同: " + ", ".join(
        f"{a[0]}-{b[0]}={(t[a] - t[b]).abs().mean().item():.4f}" for a, b in pairs))


def test_noise_target_reduces_variance():
    """时序均值目标应当比单帧噪声更小 —— 这是 N 路该学它的理由。"""
    torch.manual_seed(1)
    clean = torch.rand(1, 3, 48, 48).repeat(5, 1, 1, 1)
    noisy = (clean + 0.1 * torch.randn_like(clean)).clamp(0, 1)
    t = build_branch_targets(noisy)

    center_err = (noisy[2] - clean[2]).abs().mean().item()
    avg_err = (t['noise_target'] - clean[2]).abs().mean().item()
    assert avg_err < center_err, f"时序均值未降噪: {avg_err:.4f} >= {center_err:.4f}"
    print(f"  [ok] 时序均值降噪: 中心帧误差 {center_err:.4f} → 均值误差 {avg_err:.4f}")


def test_illum_target_removes_lowfreq():
    """光照归一化目标应当去掉整体亮度差异 —— 这是 L 路该学它的理由。

    两个只差一个全局亮度系数的片段, 归一化后的目标应当非常接近。
    """
    torch.manual_seed(2)
    base = torch.rand(5, 3, 64, 64) * 0.5 + 0.25
    dim = (base * 0.5).clamp(0, 1)

    t_base = build_branch_targets(base)
    t_dim = build_branch_targets(dim)

    raw_gap = (base[2] - dim[2]).abs().mean().item()
    illum_gap = (t_base['illum_target'] - t_dim['illum_target']).abs().mean().item()
    assert illum_gap < raw_gap, f"归一化未去掉亮度差: {illum_gap:.4f} >= {raw_gap:.4f}"
    print(f"  [ok] 光照归一化: 原始亮度差 {raw_gap:.4f} → 归一化后 {illum_gap:.4f}")


def test_shift_search_finds_known_offset():
    """对齐搜索应当能找到人工注入的位移。"""
    torch.manual_seed(3)
    ref = torch.rand(3, 48, 48)
    for want in [(0, 0), (2, 0), (0, -2), (-2, 2), (4, -4)]:
        moved = _integer_shift(ref, want[0], want[1])
        # 反向找: moved 需要被平移 -want 才能回到 ref
        found = _best_global_shift(ref, moved, max_shift=4, step=2)
        assert found == (-want[0], -want[1]), f"want {want}, got {found}"
    print("  [ok] 位移搜索在 ±4 (step=2) 内全部命中")


def test_loss_uses_distinct_targets():
    """开关打开时, 三路的分支损失必须对各自目标计算。"""
    torch.manual_seed(4)
    gt = torch.rand(2, 3, 32, 32)
    y = torch.rand(2, 3, 32, 32)
    out = {'final': y, 'branch_N': y.clone(), 'branch_L': y.clone(),
           'branch_M': y.clone(), 'ortho_loss': torch.tensor(0.1)}

    targets = {
        'noise_target': torch.zeros(2, 3, 32, 32),
        'illum_target': torch.ones(2, 3, 32, 32),
        'motion_target': torch.full((2, 3, 32, 32), 0.5),
    }
    crit = GolfV7RLoss(branch_specific_targets=True, branch_warmup_epochs=0)
    d = crit(out, gt, targets)

    # 三路分支损失应当互不相同 (因为目标是 0 / 1 / 0.5 三个不同常数)
    vals = [d['branch_N'].item(), d['branch_L'].item(), d['branch_M'].item()]
    assert len(set(round(v, 6) for v in vals)) == 3, f"三路损失未分化: {vals}"
    # 逐项核对: 对常数目标, L1 = mean|y - c|
    for key, name in (('branch_N', 'noise_target'),
                      ('branch_L', 'illum_target'),
                      ('branch_M', 'motion_target')):
        want = (y - targets[name]).abs().mean().item()
        assert abs(d[key].item() - want) < 1e-6, (key, d[key].item(), want)
    print(f"  [ok] 三路分别回归各自目标: N/L/M = "
          f"{vals[0]:.4f} / {vals[1]:.4f} / {vals[2]:.4f}")


def test_loss_default_matches_history():
    """开关默认关闭时, 损失必须与历史行为逐位一致。"""
    torch.manual_seed(5)
    gt = torch.rand(2, 3, 32, 32)
    out = {'final': torch.rand(2, 3, 32, 32),
           'branch_N': torch.rand(2, 3, 32, 32),
           'branch_L': torch.rand(2, 3, 32, 32),
           'branch_M': torch.rand(2, 3, 32, 32),
           'ortho_loss': torch.tensor(0.1)}

    legacy = GolfV7RLoss()(out, gt)
    # 即使传入 targets, 开关关闭时也应当忽略
    with_targets = GolfV7RLoss()(out, gt, {
        'noise_target': torch.zeros(2, 3, 32, 32),
        'illum_target': torch.ones(2, 3, 32, 32),
        'motion_target': torch.full((2, 3, 32, 32), 0.5),
    })
    assert torch.equal(legacy['loss'], with_targets['loss']), "关闭时不应受 targets 影响"
    print(f"  [ok] 默认关闭时行为不变 (loss={legacy['loss'].item():.6f})")


def test_missing_targets_fall_back_with_warning():
    """开关打开但目标缺失时, 应当回退到 GT 并告警, 不崩溃。"""
    torch.manual_seed(6)
    gt = torch.rand(2, 3, 32, 32)
    y = torch.rand(2, 3, 32, 32)
    out = {'final': y, 'branch_N': y.clone(), 'branch_L': y.clone(),
           'branch_M': y.clone(), 'ortho_loss': torch.tensor(0.1)}

    crit = GolfV7RLoss(branch_specific_targets=True, branch_warmup_epochs=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = crit(out, gt, None)
        assert any("branch_specific_targets" in str(w.message) for w in caught), \
            "目标缺失时未告警"
    # 回退到 GT 后三路损失应当相同
    vals = [d['branch_N'].item(), d['branch_L'].item(), d['branch_M'].item()]
    assert max(vals) - min(vals) < 1e-9, f"回退后三路应一致: {vals}"
    print("  [ok] 目标缺失时回退到 GT 并告警")


def test_backward():
    """反传必须正常, 且梯度确实经过三路分支。"""
    torch.manual_seed(7)
    gt = torch.rand(2, 3, 32, 32)
    final = torch.rand(2, 3, 32, 32, requires_grad=True)
    y_n = torch.rand(2, 3, 32, 32, requires_grad=True)
    y_l = torch.rand(2, 3, 32, 32, requires_grad=True)
    y_m = torch.rand(2, 3, 32, 32, requires_grad=True)
    out = {'final': final, 'branch_N': y_n, 'branch_L': y_l, 'branch_M': y_m,
           'ortho_loss': torch.tensor(0.1)}
    targets = {
        'noise_target': torch.rand(2, 3, 32, 32),
        'illum_target': torch.rand(2, 3, 32, 32),
        'motion_target': torch.rand(2, 3, 32, 32),
    }
    crit = GolfV7RLoss(branch_specific_targets=True, branch_warmup_epochs=0)
    crit(out, gt, targets)['loss'].backward()
    for name, tensor in (('final', final), ('branch_N', y_n),
                         ('branch_L', y_l), ('branch_M', y_m)):
        assert tensor.grad is not None and tensor.grad.abs().sum() > 0, f"{name} 无梯度"
    print("  [ok] 反传正常, 三路梯度均非零")


def main():
    print("分支专属监督目标单测:")
    test_targets_are_distinct()
    test_noise_target_reduces_variance()
    test_illum_target_removes_lowfreq()
    test_shift_search_finds_known_offset()
    test_loss_uses_distinct_targets()
    test_loss_default_matches_history()
    test_missing_targets_fall_back_with_warning()
    test_backward()
    print("\n全部通过 ✅")


if __name__ == '__main__':
    main()
