#!/usr/bin/env python3
"""分支专属监督目标: GT 的频带划分, 中频/高频由运动与噪声两路共享 (§6.19 定稿)。

## 为什么是频带划分

三个分支原本都对同一张 GT 做 L1, 最优解因此重合 —— 三路输出相同是该目标的正确解,
不是缺陷 (§6.2)。让三路各回归不同频带是正确方向。

## 为什么三个频带中, 中频与高频要由 M、N 共享

物理上三个源对频带的占用并不互斥:
  · 光照是低频、全局、慢变          → 低频带独属于 BranchL
  · 运动表现为物体边界位移, 边界在中高频 → BranchM 主导中频
  · 加性噪声全频段存在、高频最显著    → BranchN 覆盖中频到高频

早期版本把 BranchN 限制在 5 像素以下的最细高频, 实测该带平均绝对值仅 0.0057,
而低频带为 0.5271 —— 差 92 倍, N 路等于拿到一张空图, 学不到去噪。这与 TSDR
"噪声全频段、高频显著"的描述直接矛盾。

因此中频带 Q 与高频带 R 由 M、N 按比例分摊:

    S_L = P                                (光照: 低频带)
    S_M = a·Q + b·R                        (运动: 中频为主, 含部分高频)
    S_N = (1-a)·Q + (1-b)·R                (噪声: 覆盖中频到高频)

a 是运动路占中频的比例, b 是运动路占高频的比例。取 a=0.5、b=0.2 时, N 含高频 80%、
中频 50%。这样两路都真的含有中高频成分, 而 ΣS ≡ GT 仍精确成立。

## 与融合算子的配套要求 (关键)

ΣS ≡ GT 是【求和】性质, 因此融合必须是**加性**的:

    O_t = Σ_k w_k · Y_k        (w_k 不归一化, 零初始化时 w_k = 1)

若用凸组合 (Σw_k = 1 的 softmax), 则三路都完美拟合时输出是 (ΣS)/3 = GT/3 ——
差一个 3 倍因子, 数学上不可能还原 GT。这一点由 `V7RFusion(mode="additive")` 保证。

## 数值性质 (全部有单测覆盖)

  1. ΣS ≡ GT            —— 加性融合在 w=1 时精确还原 GT;
  2. 三带两两正交        —— 不存在两路学同一张图;
  3. 中高频带均值为零    —— 输出常数在带上的误差即 GT 自身能量, 封死平凡解;
  4. 只用 GT 构造        —— 不需要输入片段、块匹配或光流, 每样本约 3 ms。
"""
import torch
import torch.nn.functional as F


def lowpass(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """反射填充的平均池化低通。kernel <= 1 时返回原图。

    只支持**奇数**核: 偶数核配对称填充会把空间尺寸变成 H+1, 破坏"ΣS = GT"。
    x: [..., C, H, W] -> 同形状
    """
    k = int(kernel)
    if k <= 1:
        return x
    if k % 2 == 0:
        raise ValueError(f"lowpass 只支持奇数核, 收到 {k}; 偶数核配对称填充会改变空间尺寸")
    c, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
    lead = x.shape[:-3]
    # 统一成 [N, C, H, W] 再过池化, 最后按原形状还原 (支持 [C,H,W] 与 [B,C,H,W])
    flat = x.reshape(-1, c, h, w)
    pad = k // 2
    out = F.avg_pool2d(F.pad(flat, (pad, pad, pad, pad), mode="reflect"),
                       kernel_size=k, stride=1)
    return out.reshape(*lead, c, h, w) if lead else out.squeeze(0)


def build_branch_targets(gt: torch.Tensor, kernel_low: int = 31,
                         kernel_high: int = 5, motion_share_mid: float = 0.5,
                         motion_share_high: float = 0.2) -> dict:
    """把 GT 划分为三个频带, 中频/高频由运动与噪声两路共享。

    gt: [C, H, W] 或 [B, C, H, W], 值域 [0, 1]
    kernel_low:  低频截止核 (越大越平滑), 默认 31。需为奇数。
    kernel_high: 中高频分界核, 默认 5。需为奇数且小于 kernel_low。
    motion_share_mid:  运动路占中频带的比例, 其余归噪声路。
    motion_share_high: 运动路占高频带的比例, 其余归噪声路。

    Returns: dict with keys illum_target / motion_target / noise_target,
        三者之和恒等于 gt。
    """
    if kernel_high >= kernel_low:
        raise ValueError(
            f"kernel_high ({kernel_high}) 必须小于 kernel_low ({kernel_low}); "
            "否则中频带 lowpass_high - lowpass_low 恒为负频带")
    if not (0.0 <= motion_share_mid <= 1.0) or not (0.0 <= motion_share_high <= 1.0):
        raise ValueError("motion_share_mid / motion_share_high 必须在 [0, 1] 内")

    gt = gt.float()
    lp_low = lowpass(gt, kernel_low)
    lp_high = lowpass(gt, kernel_high)
    band_low = lp_low                      # P: 低频 (光照)
    band_mid = lp_high - lp_low            # Q: 中频
    band_high = gt - lp_high               # R: 高频

    a, b = float(motion_share_mid), float(motion_share_high)
    return {
        "illum_target": band_low,
        "motion_target": a * band_mid + b * band_high,
        "noise_target": (1.0 - a) * band_mid + (1.0 - b) * band_high,
    }


def describe_bands(gt: torch.Tensor, **kwargs) -> dict:
    """诊断用: 返回三路的能量占比与幅度, 供确认分配是否合理。"""
    t = build_branch_targets(gt, **kwargs)
    energy = {k: v.pow(2).sum().item() for k, v in t.items()}
    total = sum(energy.values())
    return {
        "energy_fraction": {k: v / total for k, v in energy.items()},
        "mean_abs": {k: v.abs().mean().item() for k, v in t.items()},
    }


# ---------------------------------------------------------------------------
# 历史版本 (保留仅供复现, 请勿用于新训练)
# ---------------------------------------------------------------------------

def build_branch_targets_partition(gt: torch.Tensor, kernel_low: int = 15,
                                   kernel_high: int = 5) -> dict:
    """已废弃: 严格三分频带 (L=低频, M=中频, N=高频)。

    问题: N 只有最细高频, 实测平均绝对值 0.0057 vs 低频 0.5271, N 路学不到东西;
    且该版本曾配 softmax 凸组合使用, 存在 Σw=1 与 ΣS=GT 相差 3 倍的错误。
    """
    gt = gt.float()
    lp_low = lowpass(gt, kernel_low)
    lp_high = lowpass(gt, kernel_high)
    return {
        "illum_target": lp_low,
        "motion_target": lp_high - lp_low,
        "noise_target": gt - lp_high,
    }


def _integer_shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """按整数像素平移 (边界复制填充)。x: [C, H, W]"""
    if dy == 0 and dx == 0:
        return x
    pad = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))
    padded = F.pad(x.unsqueeze(0), pad, mode="replicate").squeeze(0)
    h, w = x.shape[-2:]
    top = max(-dy, 0)
    left = max(-dx, 0)
    return padded[:, top:top + h, left:left + w]


def _best_global_shift(ref: torch.Tensor, src: torch.Tensor,
                       max_shift: int = 4, step: int = 2) -> tuple:
    """在 ±max_shift 内网格搜索使 src 最接近 ref 的整数平移。ref/src: [C,H,W]"""
    best = (0, 0)
    best_err = float("inf")
    offsets = list(range(-max_shift, max_shift + 1, step))
    for dy in offsets:
        for dx in offsets:
            err = (ref - _integer_shift(src, dy, dx)).abs().mean().item()
            if err < best_err:
                best_err = err
                best = (dy, dx)
    return best


def build_branch_targets_from_clip(clip: torch.Tensor, max_shift: int = 4,
                                   lowfreq_kernel: int = 31) -> dict:
    """已废弃 (§6.18): 从低光输入片段构造三个目标。

    实测三目标全部落在低光域 (均值 0.107 vs GT 0.534), 且 noise 与 motion 目标
    平均绝对差仅 0.0028 (SDSD 帧间位移多在 0-4 px, 对齐后两者退化为同一张图)。
    训练结果比共享目标差 1.828 dB (7 胜 / 63 负)。保留仅供复现。
    """
    T = clip.shape[0]
    center = T // 2
    ref = clip[center]

    aligned = []
    for t in range(T):
        if t == center:
            continue
        dy, dx = _best_global_shift(ref, clip[t], max_shift=max_shift)
        aligned.append(_integer_shift(clip[t], dy, dx))

    if aligned:
        aligned_stack = torch.stack(aligned, dim=0)
        noise = (ref + aligned_stack.sum(dim=0)) / float(T)
        far = torch.stack([aligned_stack[0], aligned_stack[-1]], dim=0)
        motion = far.mean(dim=0)
    else:
        noise = ref.clone()
        motion = ref.clone()

    field = lowpass(noise.unsqueeze(0), lowfreq_kernel).squeeze(0)
    illum = (noise / field.clamp_min(1e-3)).clamp(0.0, 1.0)

    return {
        "noise_target": noise,
        "illum_target": illum,
        "motion_target": motion,
    }
