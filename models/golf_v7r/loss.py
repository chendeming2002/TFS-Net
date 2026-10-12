#!/usr/bin/env python3
"""
Golf v7r Loss: 多分支监督 + 正交约束

Loss = L1(final, GT)
     + w_N * L1(Y_N, GT) + w_L * L1(Y_L, GT) + w_M * L1(Y_M, GT)
     + lambda_ortho * ortho_loss(ctx_N, ctx_L, ctx_M)

Phase B 追加项 (§6.6-B / §6.9, 全部默认 0 即关闭):
     + dark_weight_alpha * 加权L1(final, GT)   (§6.6-B 暗区加权)
     + w_perceptual * VGG 多层感知损失          (§6.9)
     + w_freq * FFT 幅值(+相位) L1              (§6.9)
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class GolfV7RLoss(nn.Module):
    """多分支 + 正交约束 loss

    Phase A.3 开关 (§6.2-B, 按退化类型分配监督空间):
        use_lowfreq_L=False (默认/历史行为):
            Branch-L 与 N/M 一样对整幅 GT 做 L1 → 被强信号拉向同一解 (RGB 余弦 0.99).
        use_lowfreq_L=True:
            Branch-L 只对【低频分量】做 L1 (先 avg_pool 降采样再上采样),
            强制它学光照/亮度, 而非高频纹理 → 从标签层面制造分支分工.
    """

    def __init__(self,
                 w_branch_N: float = 0.1,
                 w_branch_L: float = 0.1,
                 w_branch_M: float = 0.1,
                 lambda_ortho: float = 0.01,
                 w_ssim: float = 0.0,
                 branch_warmup_epochs: int = 5,
                 use_lowfreq_L: bool = False,
                 lowfreq_kernel: int = 16,
                 w_output_decorr: float = 0.0,
                 decorr_per_image: bool = True,
                 w_perceptual: float = 0.0,
                 perceptual_multilayer: bool = True,
                 w_freq: float = 0.0,
                 freq_phase_weight: float = 1.0,
                 dark_weight_alpha: float = 0.0,
                 dark_weight_gamma: float = 1.0,
                 branch_specific_targets: bool = False):
        """Phase B 开关 (§6.6-B 暗区加权 + §6.9 感知/频率损失)。

        w_perceptual / w_freq / dark_weight_alpha:
          - 默认全 0.0 → 历史行为完全不变 (可逐项消融)。
        dark_weight_alpha (§6.6-B):
          - >0 时对 L1 做亮度倒数加权: w(x) = (1/(lum(x)+eps))^gamma 归一化到均值 1。
            动机: L1 在亮区数值大, 暗区梯度被亮区主导 → 极暗序列学不动。
            归一化保证总 loss 尺度不变, 不因加权而实质改变有效学习率。
        branch_specific_targets (docs/v7/04-model-architecture.md 第四节):
          - False (默认) → 三路分支都对同一张 GT 做 L1 (历史行为)。
          - True  → 三路各自回归【不同】目标 (由数据侧 build_branch_targets 提供):
                    N 路 → noise_target  (时序均值)
                    L 路 → illum_target  (光照归一化)
                    M 路 → motion_target (对齐参考帧)
            动机: 共享目标下 Y_N = Y_L = Y_M = GT 就是全局最优, 三路同质化是该目标的
            正确解而非缺陷。换成互不相同的目标后最优解不再重合, 分工在目标层面成立。
            目标缺失时自动回退到 GT, 并在首次发生时告警。
        """
        super().__init__()
        self.w_branch_N = w_branch_N
        self.w_branch_L = w_branch_L
        self.w_branch_M = w_branch_M
        self.lambda_ortho = lambda_ortho
        self.w_ssim = w_ssim
        self.branch_warmup_epochs = branch_warmup_epochs
        self.use_lowfreq_L = use_lowfreq_L
        self.lowfreq_kernel = int(lowfreq_kernel) if lowfreq_kernel else 16
        self.branch_specific_targets = bool(branch_specific_targets)
        self._warned_missing_targets = False

        self.current_epoch = 0
        self._use_ssim = w_ssim > 0
        if self._use_ssim:
            try:
                from pytorch_msssim import ms_ssim
                self._ms_ssim = ms_ssim
            except ImportError:
                self._use_ssim = False

        # §6.14 复核: 输出层去相关 (默认关闭)
        self.w_output_decorr = float(w_output_decorr or 0.0)
        self.decorr_per_image = bool(decorr_per_image)

        # §6.9 感知/频率项 (默认关闭)
        self.w_perceptual = float(w_perceptual or 0.0)
        self.w_freq = float(w_freq or 0.0)
        self.freq_phase_weight = float(freq_phase_weight)
        self._perceptual = None
        if self.w_perceptual > 0:
            try:
                from losses.losses import PerceptualLoss
                self._perceptual = PerceptualLoss(multilayer=perceptual_multilayer)
            except Exception as exc:  # torchvision 缺失等
                warnings.warn(f"PerceptualLoss unavailable, w_perceptual ignored: {exc}")
                self.w_perceptual = 0.0

        # §6.6-B 暗区加权 (默认关闭)
        self.dark_weight_alpha = float(dark_weight_alpha or 0.0)
        self.dark_weight_gamma = float(dark_weight_gamma or 1.0)

    def _dark_weight_map(self, gt: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """§6.6-B 亮度倒数权重图, 缩放使加权 L1 的期望量级与裸 L1 一致。

        权重主体 w = (1/(lum+eps))^gamma 做均值归一; 再整体乘 `ref` (= 裸 L1 的标量值),
        使 Σ w*|pred-gt| 的均值与裸 L1 同量级 —— 即本项只做【误差在空间上的重分配】
        (暗区权重 >1, 亮区 <1), 而非额外叠加一份 L1。由此不改变有效学习率, alpha 可直接
        作为"混合比例"解读 (0=纯 L1, 1=完全按暗度重新加权)。
        """
        lum = gt.mean(dim=1, keepdim=True)                     # [B,1,H,W]
        w = (1.0 / (lum + 1e-2)) ** self.dark_weight_gamma
        w = w / (w.mean() + 1e-8)
        return self.dark_weight_alpha * w * ref

    def _freq_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """§6.9 FFT 幅值 (+相位) L1。"""
        fft_pred = torch.fft.rfft2(pred.float(), norm='ortho')
        fft_gt = torch.fft.rfft2(gt.float(), norm='ortho')
        loss = F.l1_loss(fft_pred.abs(), fft_gt.abs())
        if self.freq_phase_weight > 0:
            loss = loss + self.freq_phase_weight * F.l1_loss(fft_pred.angle(), fft_gt.angle())
        return loss

    @staticmethod
    def _decorr_loss(y_a: torch.Tensor, y_b: torch.Tensor,
                     per_image: bool = True) -> torch.Tensor:
        """输出层去相关: |cos(Y_a, Y_b)| 的均值 (逐图计算后平均)。

        与 `TripleQueryTCA._ortho_loss` 的区别是作用层不同: 后者约束特征层 `F_*`,
        本项约束输出层 `Y_*` (最终 RGB)。

        ⚠️ 训练结论 (2026-10-12, 详见 docs/v7/03-v7r-v3-design.md §6.17):
        本损失有平凡解, 两个 seed 的训练都退化了 —— 让某一路输出常数图即可使余弦恒为 0
        (常数减去自身均值后是全零向量), 代价远低于"学出不同的结构"。退化解也不会被
        融合层屏蔽, 因为融合权重受 Σw=1 约束, 无法把该路归零。

        已加 `valid` 掩码堵住"精确常数"这条捷径: 任一路能量趋零时不计入本项。
        但这不足以让本损失变得有用 —— 实验表明它压低的是"相似度", 而不是建立"分工"。
        若要继续使用, 必须同时保证各分支各自有效 (例如各自有独立的、互不相同的监督目标)。

        去均值后再算余弦的原因: 三路输出的共同 DC 分量会人为抬高余弦,
        而分支差异体现在结构/纹理而非整体亮度。
        """
        a = y_a.flatten(2).float()   # [B, 3, H*W]
        b = y_b.flatten(2).float()
        if not per_image:
            a = a - a.mean()
            b = b - b.mean()
            return (a * b).sum().abs() / (a.norm() * b.norm() + 1e-8)

        a = a - a.mean(dim=2, keepdim=True)
        b = b - b.mean(dim=2, keepdim=True)
        denom = a.norm(dim=2) * b.norm(dim=2)      # [B]
        valid = denom > 1e-4                        # 任一路近似常数则不计入
        if not bool(valid.any()):
            return y_a.new_zeros(())
        cos = (a * b).sum(dim=2) / denom.clamp_min(1e-8)
        return (cos.abs() * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

    def _output_decorr(self, output_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """三路输出两两去相关 (NL/NM/LM 平均)。缺失的键自动跳过。"""
        keys = [k for k in ('branch_N', 'branch_L', 'branch_M')
                if output_dict.get(k) is not None]
        if len(keys) < 2:
            return output_dict['final'].new_tensor(0.0)
        vals = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                vals.append(self._decorr_loss(output_dict[keys[i]],
                                              output_dict[keys[j]],
                                              per_image=self.decorr_per_image))
        return torch.stack(vals).mean()

    def _lowfreq(self, x: torch.Tensor) -> torch.Tensor:
        """低通: avg_pool 下采样 → 上采样回原分辨率 (保留低频亮度结构)"""
        k = self.lowfreq_kernel
        h, w = x.shape[-2:]
        pooled = F.avg_pool2d(x, kernel_size=k, stride=k,
                              ceil_mode=True, count_include_pad=False)
        return F.interpolate(pooled, size=(h, w), mode='bilinear',
                             align_corners=False)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _branch_weight(self, base: float) -> float:
        """分支监督权重 warmup: 前若 epoch 较少, 线性增长"""
        if base == 0:
            return 0.0
        if self.branch_warmup_epochs <= 0:
            return base
        frac = min(1.0, (self.current_epoch + 1) / self.branch_warmup_epochs)
        return base * frac

    _BRANCH_TARGET_KEYS = {
        'branch_N': 'noise_target',
        'branch_L': 'illum_target',
        'branch_M': 'motion_target',
    }

    def _resolve_branch_target(self, key: str, gt: torch.Tensor,
                               targets: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """取该分支应当回归的目标。关闭开关或目标缺失时回退到 GT。"""
        if not self.branch_specific_targets:
            return gt
        name = self._BRANCH_TARGET_KEYS[key]
        if targets is not None and targets.get(name) is not None:
            return targets[name].to(device=gt.device, dtype=gt.dtype)
        if not self._warned_missing_targets:
            warnings.warn(
                "branch_specific_targets=True 但未收到 "
                f"{sorted(self._BRANCH_TARGET_KEYS.values())}; 已回退到 GT。"
                " 请确认数据集已开启 branch_targets 且训练脚本透传了这些键。")
            self._warned_missing_targets = True
        return gt

    def forward(self, output_dict: Dict[str, torch.Tensor],
                gt: torch.Tensor,
                branch_targets: Optional[Dict[str, torch.Tensor]] = None
                ) -> Dict[str, torch.Tensor]:
        pred = output_dict['final']
        loss_l1 = F.l1_loss(pred, gt)
        loss = loss_l1

        loss_dict = {'loss': loss, 'l1': loss_l1}

        # 分支监督
        for key, base_w in [('branch_N', self.w_branch_N),
                            ('branch_L', self.w_branch_L),
                            ('branch_M', self.w_branch_M)]:
            if key in output_dict and output_dict[key] is not None:
                w = self._branch_weight(base_w)
                if w > 0:
                    pred_b = output_dict[key]
                    gt_b = self._resolve_branch_target(key, gt, branch_targets)
                    # Phase A.3 开关: Branch-L 只监督低频分量 (从标签层面强制分工)
                    if key == 'branch_L' and self.use_lowfreq_L:
                        pred_b = self._lowfreq(pred_b)
                        gt_b = self._lowfreq(gt_b)
                    bl = F.l1_loss(pred_b, gt_b)
                    loss = loss + w * bl
                    loss_dict[key] = bl

        # 正交约束 (作用在【特征层】F_*)
        if 'ortho_loss' in output_dict and self.lambda_ortho > 0:
            ol = output_dict['ortho_loss']
            loss = loss + self.lambda_ortho * ol
            loss_dict['ortho'] = ol

        # §6.14 复核: 输出层去相关 (作用在【输出层】Y_*)
        # 动机: 特征层正交无法阻止输出层同质化 —— 主 L1 让三路各自回归同一 GT,
        #       最优解本身就是同一张图。必须在输出层直接激励分化。
        if self.w_output_decorr > 0:
            dc = self._output_decorr(output_dict)
            loss = loss + self.w_output_decorr * dc
            loss_dict['out_decorr'] = dc.detach()

        # MS-SSIM
        if self._use_ssim:
            ssim_val = self._ms_ssim(pred.clamp(0, 1), gt.clamp(0, 1),
                                     data_range=1.0, size_average=True)
            loss = loss + self.w_ssim * (1 - ssim_val)
            loss_dict['ms_ssim'] = ssim_val

        # §6.6-B 暗区加权 L1 (与裸 L1 同量级, 只重分配空间权重, 不改变有效学习率)
        if self.dark_weight_alpha > 0:
            wmap = self._dark_weight_map(gt, loss_l1.detach())
            l_dark = (wmap * (pred - gt).abs()).mean()
            loss = loss + l_dark
            loss_dict['dark_l1'] = l_dark.detach()

        # §6.9 VGG 多层感知损失
        if self.w_perceptual > 0 and self._perceptual is not None:
            l_perc = self._perceptual(pred.clamp(0, 1), gt.clamp(0, 1))
            loss = loss + self.w_perceptual * l_perc
            loss_dict['perceptual'] = l_perc.detach()

        # §6.9 FFT 频率损失
        if self.w_freq > 0:
            l_freq = self._freq_loss(pred, gt)
            loss = loss + self.w_freq * l_freq
            loss_dict['freq'] = l_freq.detach()

        loss_dict['loss'] = loss
        return loss_dict


class SimpleLoss(nn.Module):
    """基线 loss (仅 L1), 用于对照"""

    def forward(self, output_dict, gt):
        pred = output_dict['final']
        return {'loss': F.l1_loss(pred, gt)}


if __name__ == '__main__':
    criterion = GolfV7RLoss()
    out = {
        'final': torch.rand(2, 3, 64, 64),
        'branch_N': torch.rand(2, 3, 64, 64),
        'branch_L': torch.rand(2, 3, 64, 64),
        'branch_M': torch.rand(2, 3, 64, 64),
        'ortho_loss': torch.tensor(0.1),
    }
    gt = torch.rand(2, 3, 64, 64)
    d = criterion(out, gt)
    print("Loss dict:", {k: f"{v.item():.4f}" for k, v in d.items()})
