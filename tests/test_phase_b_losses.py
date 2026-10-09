#!/usr/bin/env python3
"""Phase B 新损失项与极暗增广的单元测试 (§6.6 / §6.9)。

验证:
  1. 所有新开关默认 0 → `GolfV7RLoss` 输出与历史行为逐位一致 (回归保护);
  2. `dark_weight_alpha` 与裸 L1 同量级 (是权重【重分配】而非额外叠加一份 L1);
  3. `w_freq` / `w_perceptual` 可独立开关, 且能正常反传;
  4. `random_dark_gamma` 只压暗、不改变形状/值域, prob=0 时恒等。

运行:
    PYTHONPATH=. python tests/test_phase_b_losses.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.golf_v7r.loss import GolfV7RLoss
from datasets.transforms import random_dark_gamma


def dummy_batch(seed=0):
    torch.manual_seed(seed)
    out = {
        'final': torch.rand(2, 3, 64, 64),
        'branch_N': torch.rand(2, 3, 64, 64),
        'branch_L': torch.rand(2, 3, 64, 64),
        'branch_M': torch.rand(2, 3, 64, 64),
        'ortho_loss': torch.tensor(0.1),
    }
    gt = torch.rand(2, 3, 64, 64)
    return out, gt


def test_default_unchanged():
    out, gt = dummy_batch()
    d = GolfV7RLoss()(out, gt)
    expected_keys = {'loss', 'l1', 'branch_N', 'branch_L', 'branch_M', 'ortho'}
    assert set(d.keys()) == expected_keys, f"默认输出项变了: {sorted(d.keys())}"
    assert abs(d['loss'].item() - 0.353228) < 1e-5, d['loss'].item()
    print(f"  [ok] 默认行为不变 (loss={d['loss'].item():.6f}, 项={sorted(d.keys())})")


def test_dark_weight_scale():
    """核心语义: 加权项应与裸 L1 同量级, 而不是额外叠加一份 L1。"""
    out, gt = dummy_batch()
    d0 = GolfV7RLoss()(out, gt)
    for alpha in (0.5, 1.0):
        d = GolfV7RLoss(dark_weight_alpha=alpha)(out, gt)
        l1 = d['l1'].item()
        dl = d['dark_l1'].item()
        # 量级约束: 不应超过裸 L1 (若超过说明是"再叠一份 L1")
        assert dl < l1, f"alpha={alpha}: dark_l1={dl} 未与裸 L1 ({l1}) 同量级"
        # 线性约束
        assert abs(dl - alpha * 0.112034) < 2e-3, (alpha, dl)
        print(f"  [ok] alpha={alpha}: dark_l1={dl:.6f} < l1={l1:.6f} (线性)")


def test_dark_weight_orientation():
    """暗 GT 下, 加权项相对裸 L1 的比值应高于亮 GT 下的比值。"""
    out, bright = dummy_batch()
    dark = bright * 0.05
    crit = GolfV7RLoss(dark_weight_alpha=1.0)
    r_bright = crit(out, bright)['dark_l1'].item() / crit(out, bright)['l1'].item()
    r_dark = crit(out, dark)['dark_l1'].item() / crit(out, dark)['l1'].item()
    assert r_dark > r_bright, (r_bright, r_dark)
    print(f"  [ok] 暗区权重正确: 比值 亮={r_bright:.4f} < 暗={r_dark:.4f}")


def test_terms_independent_and_backprop():
    out, gt = dummy_batch()
    base = GolfV7RLoss()(out, gt)['loss'].item()
    for kw in ({'dark_weight_alpha': 1.0}, {'w_freq': 0.02}, {'w_perceptual': 0.05}):
        o = dict(out)
        o['final'] = torch.rand(2, 3, 64, 64, requires_grad=True)
        d = GolfV7RLoss(**kw)(o, gt)
        d['loss'].backward()
        assert o['final'].grad is not None and torch.isfinite(o['final'].grad).all()
        print(f"  [ok] {kw} → Δloss={d['loss'].item() - base:+.6f}, 反传正常")


def test_dark_gamma_aug():
    torch.manual_seed(0)
    clip = torch.rand(5, 3, 32, 32)
    # prob=0 → 恒等
    assert torch.equal(random_dark_gamma(clip, prob=0.0), clip), "prob=0 应为恒等"
    # prob=1 → 必然变暗 (gamma>1), 且形状/值域不变
    dark = random_dark_gamma(clip, prob=1.0, gamma_range=(2.0, 2.0))
    assert dark.shape == clip.shape
    assert dark.min() >= 0.0 and dark.max() <= 1.0
    assert dark.mean() < clip.mean(), f"未变暗: {dark.mean()} vs {clip.mean()}"
    # 期望亮度关系: E[x^2] ≈ E[x]^2 只在均匀分布下成立, 这里只查单调压暗
    print(f"  [ok] dark_gamma(prob=1, gamma=2): mean {clip.mean():.4f} → {dark.mean():.4f}")


def main():
    print("Phase B 损失/增广单测:")
    test_default_unchanged()
    test_dark_weight_scale()
    test_dark_weight_orientation()
    test_terms_independent_and_backprop()
    test_dark_gamma_aug()
    print("\n全部通过 ✅")


if __name__ == '__main__':
    main()
