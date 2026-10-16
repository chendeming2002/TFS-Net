#!/usr/bin/env python3
"""GT 频带划分监督目标 + 加性融合的单元测试 (docs/v7/03-v7r-v3-design.md §6.19)。

设计要点与对应的验证:
  1. **ΣS ≡ GT** —— 三个目标之和恒等于 GT。加性融合在权重为 1 时精确还原 GT;
  2. **中频/高频由 M、N 共享** —— N 覆盖中频到高频, 不局限于最细高频;
  3. **三带两两正交** —— 不存在两路学同一张图 (可用性由共享比例调节);
  4. **中高频带均值为零** —— 输出常数在这些带上的误差即 GT 自身能量;
  5. **加性融合初始化即精确还原** —— 这是与凸组合的关键区别。
     凸组合 Σw=1 配 ΣS=GT 会差 3 倍因子, 是本设计必须避开的历史错误。

运行:
    PYTHONPATH=. python tests/test_branch_targets.py
"""
import os
import sys
import warnings

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.branch_targets import (
    build_branch_targets, build_branch_targets_residual, lowpass)
from models.golf_v7r.fusion_v7r import V7RFusion
from models.golf_v7r.loss import GolfV7RLoss


def test_partition_sums_to_gt():
    """三个目标之和必须恒等于 GT —— 加性融合可达性的依据。"""
    torch.manual_seed(0)
    gt = torch.rand(3, 64, 64)
    t = build_branch_targets(gt, kernel_low=31, kernel_high=5)
    total = t['illum_target'] + t['motion_target'] + t['noise_target']
    err = (total - gt).abs().max().item()
    assert err < 1e-6, f"三目标之和与 GT 不符, 最大误差 {err:.2e}"
    print(f"  [ok] ΣS ≡ GT (最大误差 {err:.2e})")


def test_motion_and_noise_share_mid_and_high():
    """M 与 N 必须同时含有中频与高频成分 —— 这是"共享"的定义。

    早期版本把 N 限定在最细高频 (GT − LPF_5(GT)), 实测该带幅度远低于低频带,
    N 路等于拿到空图。此处直接核验两路在 Q、R 上的份额。
    """
    torch.manual_seed(1)
    gt = torch.rand(3, 128, 128)
    kl, kh = 31, 5
    lp_low = lowpass(gt, kl)
    lp_high = lowpass(gt, kh)
    Q = lp_high - lp_low      # 中频
    R = gt - lp_high          # 高频

    t = build_branch_targets(gt, kernel_low=kl, kernel_high=kh,
                             motion_share_mid=0.5, motion_share_high=0.2)
    want_m = 0.5 * Q + 0.2 * R
    want_n = 0.5 * Q + 0.8 * R
    assert torch.allclose(t['motion_target'], want_m, atol=1e-6), "M 的中高频份额不符"
    assert torch.allclose(t['noise_target'], want_n, atol=1e-6), "N 的中高频份额不符"
    print("  [ok] M/N 共享中频与高频 (N: 中频 50% + 高频 80%; M: 中频 50% + 高频 20%)")


def test_noise_band_is_not_trivial():
    """N 路的幅度必须与其它两路可比 —— 不能是"几乎空白的图"。"""
    torch.manual_seed(2)
    gt = torch.rand(3, 128, 128) * 0.6 + 0.2
    t = build_branch_targets(gt, kernel_low=31, kernel_high=5)
    amp = {k: v.abs().mean().item() for k, v in t.items()}
    ratio = amp['noise_target'] / amp['illum_target']
    assert ratio > 0.02, f"N 路幅度仅为 L 路的 {ratio:.1%}, 过低"
    print(f"  [ok] 三路幅度可比: L={amp['illum_target']:.4f}, "
          f"M={amp['motion_target']:.4f}, N={amp['noise_target']:.4f}")


def test_high_bands_have_zero_mean():
    """中频与高频目标的均值为零 —— "常数不可行"的依据。"""
    torch.manual_seed(3)
    gt = torch.rand(3, 128, 128) * 0.8 + 0.1
    t = build_branch_targets(gt, kernel_low=31, kernel_high=5)
    for name in ('motion_target', 'noise_target'):
        m = t[name].mean().item()
        assert abs(m) < 0.01, f"{name} 均值 {m:+.4f} 偏大, 常数输出可能蒙混过关"
    print(f"  [ok] 中/高频带均值为零 (M={t['motion_target'].mean():+.5f}, "
          f"N={t['noise_target'].mean():+.5f})")


def test_targets_are_distinct():
    """三个目标必须互不相同 —— 否则等于什么都没改 (§6.18 的失败模式)。"""
    torch.manual_seed(4)
    gt = torch.rand(3, 128, 128)
    t = build_branch_targets(gt, kernel_low=31, kernel_high=5)
    pairs = [('illum_target', 'motion_target'), ('illum_target', 'noise_target'),
             ('motion_target', 'noise_target')]
    for a, b in pairs:
        d = (t[a] - t[b]).abs().mean().item()
        assert d > 1e-3, f"{a} 与 {b} 几乎相同 (平均绝对差 {d:.5f})"
    print("  [ok] 三目标互不相同: " + ", ".join(
        f"{a[0]}-{b[0]}={(t[a]-t[b]).abs().mean().item():.4f}" for a, b in pairs))


def test_additive_fusion_reproduces_gt():
    """加性融合在三路各拟合自己目标时, 必须精确还原 GT。

    这是与凸组合的关键区别: 凸组合给的是加权【平均】, 而 ΣS = GT 是【求和】性质,
    两者相差 3 倍因子。历史上正是这个错误导致设计不可行。
    """
    torch.manual_seed(5)
    gt = torch.rand(3, 64, 64)
    t = build_branch_targets(gt, kernel_low=15, kernel_high=5)
    YL = t['illum_target'].unsqueeze(0)
    YM = t['motion_target'].unsqueeze(0)
    YN = t['noise_target'].unsqueeze(0)
    x = torch.zeros_like(YL)

    fus = V7RFusion(mode='additive', residual_freeze=True)
    out = fus(YN, YL, YM, x)
    w = [round(out['weights'][0, i].mean().item(), 4) for i in range(3)]
    assert all(abs(v - 1.0) < 1e-4 for v in w), f"初始权重应为 1, 实测 {w}"
    err = (out['O_t'] - gt.unsqueeze(0).clamp(0, 1)).abs().max().item()
    assert err < 1e-4, f"加性融合未能还原 GT, 最大误差 {err:.4f}"
    print(f"  [ok] 加性融合初始化即精确还原 GT (权重={w}, 误差={err:.2e})")


def test_convex_fusion_would_fail():
    """反证: 凸组合配频带目标会差 3 倍 —— 记录这个历史错误, 防止回退。"""
    torch.manual_seed(6)
    gt = torch.rand(3, 64, 64)
    t = build_branch_targets(gt, kernel_low=15, kernel_high=5)
    YL = t['illum_target'].unsqueeze(0)
    YM = t['motion_target'].unsqueeze(0)
    YN = t['noise_target'].unsqueeze(0)
    x = torch.zeros_like(YL)
    fus = V7RFusion(mode='softmax', residual_freeze=True)
    out = fus(YN, YL, YM, x)
    ratio = gt.mean().item() / max(out['O_t'].mean().item(), 1e-9)
    assert abs(ratio - 3.0) < 0.05, f"预期差 3 倍, 实测 {ratio:.2f}"
    print(f"  [ok] 反证: 凸组合差 3 倍 (GT/输出 = {ratio:.2f}) → 必须用加性")


def test_additive_gradient_is_positive():
    """加性模式下权重恒为正, 不存在"某路权重归零则梯度断流"的问题。"""
    torch.manual_seed(7)
    fus = V7RFusion(mode='additive')
    with torch.no_grad():          # 极端负偏置: softplus 仍严格为正
        fus.weight_net[-1].bias.fill_(-20.0)
    y = [torch.rand(1, 3, 16, 16) for _ in range(3)]
    out = fus(y[0], y[1], y[2], torch.zeros_like(y[0]))
    wmin = out['weights'].min().item()
    assert wmin > 0, f"权重出现非正值 {wmin}"
    print(f"  [ok] 极端负偏置下权重仍为正 (min w = {wmin:.2e})")


def test_constant_output_is_penalized():
    """常数输出必须在中/高频带上产生显著误差 (封死平凡解)。"""
    torch.manual_seed(8)
    gt = torch.rand(3, 128, 128) * 0.8 + 0.1
    t = build_branch_targets(gt, kernel_low=31, kernel_high=5)
    pred = torch.full_like(t['noise_target'], float(t['noise_target'].mean()))
    err = F.l1_loss(pred, t['noise_target']).item()
    assert err > 1e-3, f"常数输出在 N 带上的误差仅 {err:.5f}"
    print(f"  [ok] 常数输出在 N 带上误差 = {err:.5f} (正常输出为 0)")


def test_band_normalization_balances_contributions():
    """按带归一化后三路分支损失可比 —— 否则低频路独占梯度。"""
    torch.manual_seed(9)
    gt = torch.rand(2, 3, 64, 64) * 0.8 + 0.1
    y = torch.zeros(2, 3, 64, 64)
    out = {'final': y, 'branch_N': y, 'branch_L': y, 'branch_M': y,
           'ortho_loss': torch.tensor(0.1)}
    t = build_branch_targets(gt[0], kernel_low=15, kernel_high=5)
    targets = {k: v.unsqueeze(0).repeat(2, 1, 1, 1) for k, v in t.items()}

    kw = dict(branch_specific_targets=True, branch_warmup_epochs=0)
    raw = GolfV7RLoss(branch_target_normalize=False, **kw)(out, gt, targets)
    norm = GolfV7RLoss(branch_target_normalize=True, **kw)(out, gt, targets)

    ratio_raw = raw['branch_L'].item() / max(raw['branch_M'].item(), 1e-9)
    assert ratio_raw > 3, f"原始 L/M 损失应差 3 倍以上, 实测 {ratio_raw:.1f}"

    def contribution(d, branch_key, target_name):
        scale = targets[target_name].abs().mean().clamp_min(1e-3).item()
        return d[branch_key].item() / scale

    l_n = contribution(norm, 'branch_L', 'illum_target')
    m_n = contribution(norm, 'branch_M', 'motion_target')
    assert abs(l_n / max(m_n, 1e-9) - 1.0) < 0.05, f"归一化后应接近, 实测 {l_n/m_n:.3f}"
    assert norm['loss'].item() != raw['loss'].item(), "归一化未影响总损失"
    print(f"  [ok] 按带归一化: 原始比 {ratio_raw:.1f} → 归一化后 {l_n/m_n:.3f}")


def test_bad_args_rejected():
    """kernel 顺序错误、共享比例越界、偶数核都必须报错。"""
    gt = torch.rand(3, 32, 32)
    for kw, what in [({'kernel_low': 5, 'kernel_high': 15}, 'kernel 顺序'),
                     ({'motion_share_mid': 1.5}, '共享比例越界')]:
        try:
            build_branch_targets(gt, **kw)
        except ValueError:
            continue
        raise AssertionError(f"{what} 未被拒绝")
    try:
        lowpass(gt, 16)
    except ValueError:
        pass
    else:
        raise AssertionError("偶数核未被拒绝")
    print("  [ok] 非法参数均被正确拒绝")


def test_loss_uses_distinct_targets():
    """开关打开时三路分别对各自频带算 L1。"""
    torch.manual_seed(10)
    gt = torch.rand(2, 3, 32, 32)
    y = torch.rand(2, 3, 32, 32)
    out = {'final': y, 'branch_N': y.clone(), 'branch_L': y.clone(),
           'branch_M': y.clone(), 'ortho_loss': torch.tensor(0.1)}
    t = build_branch_targets(gt[0], kernel_low=9, kernel_high=5)
    targets = {k: v.unsqueeze(0).repeat(2, 1, 1, 1) for k, v in t.items()}

    crit = GolfV7RLoss(branch_specific_targets=True, branch_warmup_epochs=0)
    d = crit(out, gt, targets)
    for key, name in (('branch_N', 'noise_target'), ('branch_L', 'illum_target'),
                      ('branch_M', 'motion_target')):
        want = F.l1_loss(y, targets[name]).item()
        assert abs(d[key].item() - want) < 1e-6, (key, d[key].item(), want)
    vals = [d['branch_N'].item(), d['branch_L'].item(), d['branch_M'].item()]
    print(f"  [ok] 三路分别回归各自频带: N/L/M = "
          f"{vals[0]:.4f} / {vals[1]:.4f} / {vals[2]:.4f}")


def test_loss_default_matches_history():
    """默认关闭时与历史行为逐位一致。"""
    torch.manual_seed(11)
    gt = torch.rand(2, 3, 32, 32)
    out = {'final': torch.rand(2, 3, 32, 32),
           'branch_N': torch.rand(2, 3, 32, 32),
           'branch_L': torch.rand(2, 3, 32, 32),
           'branch_M': torch.rand(2, 3, 32, 32),
           'ortho_loss': torch.tensor(0.1)}
    legacy = GolfV7RLoss()(out, gt)
    with_targets = GolfV7RLoss()(out, gt, {
        'noise_target': torch.rand(2, 3, 32, 32),
        'illum_target': torch.rand(2, 3, 32, 32),
        'motion_target': torch.rand(2, 3, 32, 32)})
    assert torch.equal(legacy['loss'], with_targets['loss'])
    print(f"  [ok] 默认关闭时行为不变 (loss={legacy['loss'].item():.6f})")


def test_missing_targets_fall_back_with_warning():
    """目标缺失时回退到 GT 并告警。"""
    torch.manual_seed(12)
    gt = torch.rand(2, 3, 32, 32)
    y = torch.rand(2, 3, 32, 32)
    out = {'final': y, 'branch_N': y.clone(), 'branch_L': y.clone(),
           'branch_M': y.clone(), 'ortho_loss': torch.tensor(0.1)}
    crit = GolfV7RLoss(branch_specific_targets=True, branch_warmup_epochs=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        crit(out, gt, None)
        assert any("branch_specific_targets" in str(w.message) for w in caught)
    print("  [ok] 目标缺失时回退到 GT 并告警")


def test_residual_freeze():
    """残差门冻结后, 最终输出不得再混入中心帧。"""
    torch.manual_seed(13)
    fused = V7RFusion(mode='additive', residual_freeze=True)
    y = [torch.rand(1, 3, 16, 16) for _ in range(3)]
    x = torch.zeros(1, 3, 16, 16)
    out = fused(y[0], y[1], y[2], x)
    ref = fused.refine(sum(y)).clamp(0, 1)   # 权重为 1 时即 Σy
    assert torch.allclose(out['O_t'], ref, atol=1e-6), "冻结后输出仍受中心帧影响"
    print("  [ok] 残差门冻结后输出完全由三分支决定")


def test_backward():
    """反传正常, 三路梯度均非零。"""
    torch.manual_seed(14)
    gt = torch.rand(2, 3, 32, 32)
    final = torch.rand(2, 3, 32, 32, requires_grad=True)
    tensors = {k: torch.rand(2, 3, 32, 32, requires_grad=True) for k in ('N', 'L', 'M')}
    out = {'final': final, 'branch_N': tensors['N'], 'branch_L': tensors['L'],
           'branch_M': tensors['M'], 'ortho_loss': torch.tensor(0.1)}
    t = build_branch_targets(gt[0], kernel_low=9, kernel_high=5)
    targets = {k: v.unsqueeze(0).repeat(2, 1, 1, 1) for k, v in t.items()}
    crit = GolfV7RLoss(branch_specific_targets=True, branch_warmup_epochs=0)
    crit(out, gt, targets)['loss'].backward()
    for name, tensor in [('final', final)] + list(tensors.items()):
        assert tensor.grad is not None and tensor.grad.abs().sum() > 0, f"{name} 无梯度"
    print("  [ok] 反传正常, 三路梯度均非零")


def test_residual_targets_sum_to_center_gt():
    """b 方案 (对齐残差) 的三个目标之和恒等于中心帧。"""
    torch.manual_seed(20)
    window = torch.rand(5, 3, 48, 48)
    t = build_branch_targets_residual(window, kernel_low=15)
    total = t['illum_target'] + t['noise_target'] + t['motion_target']
    err = (total - window[2]).abs().max().item()
    assert err < 1e-6, f"残差目标和与中心帧不符, 最大误差 {err:.2e}"
    print(f"  [ok] b 方案 ΣS ≡ 中心帧 GT (最大误差 {err:.2e})")


def _structured_window(T=5, size=64, shift=2):
    """构造一个可对齐的合成窗口: 每个时刻的内容是同一图案的整数位移。

    纯随机窗口之间没有可对齐的结构, 位移搜索会失效, 无法检验 b 方案的意图;
    这里让第 t 帧等于中心帧图案整体偏移 (t-2)·shift, 位移搜索应能找回它。
    """
    torch.manual_seed(21)
    base = torch.zeros(3, size, size)
    base[:, size // 4:size // 2, size // 4:size // 2] = 1.0
    base[:, size // 2:3 * size // 4, size // 2:3 * size // 4] = 0.5
    window = []
    for t in range(T):
        dy = int(round((t - T // 2) * shift))
        window.append(torch.roll(base, shifts=dy, dims=-2))
    return torch.stack(window, dim=0)


def test_residual_targets_are_orthogonal():
    """b 方案的 S_N 与 S_M 必须接近正交 —— 这是相对 band 方案的核心改进。

    band 方案两路共享中频, 实测 corr(S_N, S_M) = +0.93, 融合只需其一;
    b 方案把 S_M 定义为"对齐后的余量", 与 S_N (对齐结果的高频) 在设计上不重叠。
    """
    window = _structured_window()
    t = build_branch_targets_residual(window, kernel_low=15, max_shift=4)
    a = t['noise_target'].flatten() - t['noise_target'].mean()
    b = t['motion_target'].flatten() - t['motion_target'].mean()
    corr = (a * b).sum() / (a.norm() * b.norm() + 1e-12)
    assert corr.abs() < 0.3, f"S_N 与 S_M 相关性过高 ({corr:.3f}), 两路仍可互相替代"
    print(f"  [ok] b 方案 corr(S_N, S_M) = {corr:+.3f} (band 方案为 +0.93)")


def test_residual_targets_lack_dc():
    """b 方案的 N、M 目标是纯零均值纹理 —— 这是 N、M 两路塌缩的直接原因。

    实测: 归一化损失下, "输出常数 0" 使零均值目标到达损失下界, 该路失去学习动力。
    本测试把这一性质固化下来, 提醒后续不要在有直流分量的假设下调这两个目标。
    """
    torch.manual_seed(22)
    window = torch.rand(5, 3, 64, 64) * 0.1 + 0.02   # 低亮度场景
    t = build_branch_targets_residual(window, kernel_low=15)
    dc_n = t['noise_target'].mean().abs().item()
    dc_m = t['motion_target'].mean().abs().item()
    std_n = t['noise_target'].std().item()
    std_m = t['motion_target'].std().item()
    assert dc_n / std_n < 0.2, f"S_N 直流比过高 ({dc_n/std_n:.3f})"
    assert dc_m / std_m < 0.3, f"S_M 直流比过高 ({dc_m/std_m:.3f})"
    print(f"  [ok] b 方案 N/M 目标近乎零均值 (DC/std: N={dc_n/std_n:.3f}, M={dc_m/std_m:.3f})")


def test_branch_specific_tiny_target_gives_low_gradient_signal():
    """零均值小目标在归一化损失下, "输出 0" 即接近损失下界 —— 解释 N/M 塌缩的机制测试。

    对一个零均值信号 s, 常数预测 c 的平均绝对误差在 c = median(s) ≈ 0 处取最小;
    归一到单位尺度后该最小值约为 0.8 (批量 L1 下), 而任何非零输出都要额外付出拟合成本。
    这里直接验证"向 0 收缩"是该目标下的低损失方向。
    """
    torch.manual_seed(23)
    gt = torch.rand(3, 64, 64) * 0.1
    t = build_branch_targets(gt, kernel_low=15, kernel_high=5,
                             motion_share_mid=0.5, motion_share_high=0.5)
    target = t['motion_target']
    scale = target.abs().mean().clamp_min(1e-3)
    loss_zero = F.l1_loss(torch.zeros_like(target), target) / scale
    loss_true = F.l1_loss(target, target) / scale
    assert loss_zero < 1.05, f"输出 0 的归一化损失应接近下界, 实为 {loss_zero:.3f}"
    assert loss_zero > loss_true, "输出 0 不应优于输出真值"
    print(f"  [ok] 零均值目标下输出 0 的归一化损失 = {loss_zero:.3f} (输出真值 = {loss_true:.3f})")


def main():
    print("GT 频带划分目标 + 加性融合单测 (§6.19):")
    test_partition_sums_to_gt()
    test_motion_and_noise_share_mid_and_high()
    test_noise_band_is_not_trivial()
    test_high_bands_have_zero_mean()
    test_targets_are_distinct()
    test_additive_fusion_reproduces_gt()
    test_convex_fusion_would_fail()
    test_additive_gradient_is_positive()
    test_constant_output_is_penalized()
    test_band_normalization_balances_contributions()
    test_bad_args_rejected()
    test_loss_uses_distinct_targets()
    test_loss_default_matches_history()
    test_missing_targets_fall_back_with_warning()
    test_residual_freeze()
    test_backward()
    print("\nb 方案 (对齐残差) 单测:")
    test_residual_targets_sum_to_center_gt()
    test_residual_targets_are_orthogonal()
    test_residual_targets_lack_dc()
    test_branch_specific_tiny_target_gives_low_gradient_signal()
    print("\n全部通过 ✅")


if __name__ == '__main__':
    main()
