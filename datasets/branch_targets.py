#!/usr/bin/env python3
"""分支专属监督目标 (对应 docs/v7/04-model-architecture.md 第四节)。

背景: 三个分支原本都对同一张 GT 做 L1 回归, 最优解因此是同一张图 —— 三路输出相同
不是缺陷而是该目标下的正确解。实测三路输出余弦 0.996–0.999, 消融任一路后指标不变
(docs/v7/03-v7r-v3-design.md §6.17)。

做法: 让每一路回归【不同的】目标, 使"三路输出相同"在目标层面就是错的。
三个目标都从输入片段自身构造, 不需要额外标注:

  · noise_target    时序均值: 邻帧对齐后与中心帧取均值 —— 噪声帧间独立, 平均是其最优估计
  · illum_target    光照归一化: 时序均值图除以自身低频分量 —— 光照是低频慢变的全局量
  · motion_target   对齐参考: 邻帧按位移 warp 到中心帧时刻 —— 运动退化的修复结果即对齐后的帧

设计取舍 (为什么在数据侧用块匹配而非模型的 FlowEstimator):
  目标必须在前向之前就已确定, 否则它会随模型参数一起变化, 失去"固定监督信号"的性质
  —— 那又会退化成一个可被优化器绕过的软约束。块匹配是纯函数, 对同一输入恒定输出。
"""
import torch
import torch.nn.functional as F


def _lowfreq(x: torch.Tensor, kernel: int = 31) -> torch.Tensor:
    """大核均值池化近似低通, 用于估计光照场。

    x: [..., C, H, W] -> 同形状
    """
    shape = x.shape
    flat = x.reshape(-1, shape[-3], shape[-2], shape[-1])
    pad = kernel // 2
    out = F.avg_pool2d(F.pad(flat, (pad, pad, pad, pad), mode="reflect"),
                       kernel_size=kernel, stride=1)
    return out.reshape(shape)


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
    """在 ±max_shift 范围内网格搜索使 src 与 ref 最接近的整数平移。

    用全局平移而非逐像素光流: 目标图只需"大致对齐", 过细的对齐会把噪声也对上,
    反而削弱时序平均的去噪效果。step=2 把搜索量从 81 降到 25 次。

    ref / src: [C, H, W]
    Returns: (dy, dx)
    """
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


def build_branch_targets(clip: torch.Tensor, max_shift: int = 4,
                         lowfreq_kernel: int = 31) -> dict:
    """为单个样本构造三个分支专属监督目标。

    clip: [T, 3, H, W]  低光输入片段 (值域 [0,1])
    Returns: dict with keys noise_target / illum_target / motion_target, 各 [3, H, W]
    """
    assert clip.dim() == 4, f"expect [T,3,H,W], got {tuple(clip.shape)}"
    T = clip.shape[0]
    center = T // 2
    ref = clip[center]

    # 邻帧对齐到中心帧
    aligned = []
    for t in range(T):
        if t == center:
            continue
        dy, dx = _best_global_shift(ref, clip[t], max_shift=max_shift)
        aligned.append(_integer_shift(clip[t], dy, dx))

    if aligned:
        aligned_stack = torch.stack(aligned, dim=0)          # [T-1, 3, H, W]
        # 时序均值: 帧间独立噪声被平均抑制, 低频结构保留
        noise_target = (ref + aligned_stack.sum(dim=0)) / float(T)
        # 对齐参考: 只取【最远的两帧】的对齐结果。最远帧的位移最大,
        # 所以这张图承载的位移校正信息最强, 与"全帧平均"区分得开。
        # 若取全部邻帧平均, 它会退化成与 noise_target 几乎相同的图 (实测差 0.012),
        # 使 M 路和 N 路的目标重合, 分工又回到原点。
        far = torch.stack([aligned_stack[0], aligned_stack[-1]], dim=0)
        motion_target = far.mean(dim=0)
    else:
        noise_target = ref.clone()
        motion_target = ref.clone()

    # 光照归一化: 除以自身低频分量, 去掉绝对曝光、保留相对反射
    illum_field = _lowfreq(noise_target.unsqueeze(0), kernel=lowfreq_kernel).squeeze(0)
    illum_target = (noise_target / illum_field.clamp_min(1e-3)).clamp(0.0, 1.0)

    return {
        "noise_target": noise_target,
        "illum_target": illum_target,
        "motion_target": motion_target,
    }
