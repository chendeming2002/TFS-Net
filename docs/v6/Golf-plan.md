# Golf 版本设计方案 (Foxtrot 修复版)

> 基于 Foxtrot ep70 实验 + 论文调研 (CVPR2026/ICCV2025/NeurIPS2025)
> 目标: 修复棋盘格/分格伪影 + 结构缺陷, 提升 PSNR 与感知质量
> 创建: 2026-09-16
> 训练: `outputs/golf_r1` (40ep, 2026-09-16 启动)

---

## 一、Foxtrot 存在的问题（诊断结论）

### 1.1 ⚠ 分格/棋盘格伪影 — 两个独立来源

**来源 A（主因）：`tiled_forward` 均匀平均导致缝线伪影**

排查过程：
- 用 Read 工具直接查看推理帧 `outputs/foxtrot_r1/pair45_vis_ep70.jpg`，发现中间列（模型输出）有**约 224px 间隔的大尺度方块边界**，块内亮度不连续
- 检查 `utils/inference.py:tiled_forward` 发现：overlap 区权重=2.0、非重叠区权重=1.0，直接 `sum/weight`
- 1920×1080 图像按 tile=256/overlap=32 切分 → stride=224 → **重叠带只有 32px 且没有渐变混合**，直接平均使重叠带亮度被稀释
- 结论：**这是 tile 缝合伪影，与模型无关，推理时即可复现**

**来源 B（次因）：三分支 PixelShuffle 上采样**

```
Foxtrot 三分支统一结构:
  nn.Conv2d(C, C*4, 1×1, bias=True)  →  nn.PixelShuffle(2)
```
- 1×1 卷积**无感受野**，PixelShuffle 把 4 个输出通道机械拆到 2×2 子像素网格
- 每个子像素位置只由单一通道决定，相邻输出像素之间零信息交流
- 训练后各子像素通道的梯度分布不均 → **2px 周期棋盘格**（实测自相关第一峰=2px）
- 这是 Odena et al. 2016 明确指出的 sub-pixel 棋盘格问题

### 1.2 结构缺陷（代码审计）

| # | 问题 | 证据 | 影响 |
|---|------|------|------|
| P1 | F1/F3 编码器特征完全闲置 | `tsdnet.py:148-150` 构造后无引用 | ~60% 编码器算力浪费 |
| P2 | 全流程 H/2 处理，无 skip | TCA/三分支全在 128×128 | 结构细节依赖 PixelShuffle 外推 |
| P3 | 三分支同监督同一 GT | `loss.py` L_N/L_L/L_M 都对 gt | 功能冗余，参数效率低 |
| P4 | 时序一致性损失缺失 | `lambda_temp=0`，代码里是 `pass` | 帧间闪烁未约束 |
| P5 | Branch-M 光流分辨率不足 | 光流在 H/2 估计 | SDSD ±4px 运动量化误差大 |

### 1.3 实验结果回顾（Foxtrot ep70）

| 指标 | ep60 | ep70 | 判读 |
|------|:---:|:---:|------|
| PSNR | **19.716** | 19.645 | ep60 峰值 |
| SSIM | 0.7390 | **0.7402** | ep70 最优 |
| LPIPS | 0.3216 | **0.3199** | ep70 最优 |
| pair45 PSNR | 14.76 | 14.37 | 落后 TBC1B 15.69 |

**三指标错位**：PSNR 先达峰，感知指标后达峰（感知-保真权衡）。

---

## 二、Golf 设计方案

### 2.1 核心改动总览

| # | 改动 | 对应问题 | 实现 |
|---|------|---------|------|
| G1 | **resize-conv 上采样** | 来源B 棋盘格 | `models/golf/upsample.py` |
| G2 | **F1 skip 接入三分支** | P1 + P2 | 三分支 upsample 时 concat F1 |
| G3 | **tiled_forward 余弦窗口混合** | 来源A 缝线 | `utils/inference.py` 重写 |
| G4 | **棋盘格频域损失** | 兜底 | `GolfLoss._checkerboard_loss` |
| G5 | **分支差异化弱监督** | P3 | `GolfLoss._branch_divergence` |
| G6 | **高频稳定损失** | P4 | `GolfLoss._temporal_highfreq_stability` |
| G7 | **Fusion 权重网络加深** | 融合表达力 | 2层→4层 |
| G8 | **残差 gamma 上界 0.5→0.9** | 中心帧细节受限 | `fusion.py` |

### 2.2 G1: resize-conv 上采样（消除棋盘格）

**原理**（Odena et al. 2016 "Deconvolution and Checkerboard Artifacts"）：
- 转置卷积/PixelShuffle 的棋盘格源于**子像素位置的滤波器不重叠或权重不均**
- 解决方案：先做平滑插值（bilinear），再用普通卷积精化

```python
class UpsampleBlock(nn.Module):
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.act(self.conv1(x))   # 3×3 感受野混合
        x = self.conv2(x)             # 3×3 细化
        return self.norm(x)
```

**为什么有效**：双线性插值在频域是低通，不会在 Nyquist 频率（周期2px）产生能量尖峰；后续 3×3 卷积提供可学习增强但保持空间连续性。

### 2.3 G2: F1 skip 接入

```
编码器输出:
  F1: (B,T,32,H,W)      ← 原分辨率, 之前完全浪费
  F2: (B,T,64,H/2,W/2)  ← 唯一使用
  F3: (B,T,128,H/4,W/4) ← 之前完全浪费

Golf 改动: F1_center 直连三分支的上采样
  Branch-N.upsample(F_N, skip=F1_center)
  Branch-L.upsample(F_L, skip=F1_center)
  Branch-M.upsample(F_M, skip=F1_center)
```

**收益**：补偿 TCA 在 H/2 工作导致的高频信息丢失，F1 携带原始分辨率的边缘/纹理。

### 2.4 G3: tiled_forward 余弦窗口混合

```python
# 修复前: output += tile_pred;  weight += 1.0  → 重叠带权重2.0, 亮度跳变
# 修复后:
tile_w = raised_cosine_2d(tile_size)   # 中心1.0, 边缘0.1
output += tile_pred * tile_w
weight += tile_w
output /= weight.clamp_min(1e-8)
```

**关键细节**：窗口下限设 0.1（不是 0），避免图像边界单 tile 覆盖时权重归零导致黑边。
边界处 `0.1*pred/0.1 == pred` 保持精确。

### 2.5 G4-G6: 新增损失项

```python
L_total = L_final                          # 主监督 (L1 + 0.3·SSIM)
        + 0.3·L_N + 0.3·L_L + 0.3·L_M      # 三分支监督
        + 0.01·L_ortho                     # 正交解耦
        + 0.05·L_chess                     # [新] FFT Nyquist 频段惩罚
        + 0.02·L_temp                      # [新] 高频稳定
        + 0.05·L_div                       # [新] 分支差异化
```

**L_chess**：对输出做 FFT，统计 Nyquist 十字带（减去中心低频）的能量占比。棋盘格在此频段有特征尖峰。

**L_temp**：约束输出高频能量不超过输入高频的 1.5 倍（防止噪声放大）。

**L_div**：用输入构造三个空间 mask（暗区/亮区/高梯度），引导 Branch-N/L/M 各自在对应区域与输入差异更大——强制功能分工，不需要额外 GT。

### 2.6 G7-G8: 融合模块改进

- 权重网络 2层→4层（`64→64→32→3`），更精细的空间自适应
- 残差 gamma 从 `sigmoid(·)`∈(0,0.5) 放宽到 `0.9·sigmoid(·)`∈(0,0.9)，让中心帧细节贡献更大

---

## 三、Golf 架构总览

```
输入 (B,5,3,H,W)
    ↓
[SharedEncoder] 逐帧编码
    ├─ F1 (32ch, H)      ──────────────┐
    ├─ F2 (64ch, H/2) ──┐              │ (skip)
    └─ F3 (128ch,H/4)   │              │
                        ↓              │
              [TCA-RWKV 三查询解耦]     │
                        ↓              │
              F_N / F_L / F_M (128ch, H/2)
                        ↓              │
    ┌───────────────────┼──────────────┼──────────────┐
    ↓                   ↓              ↓              ↓
[Branch-N]         [Branch-L]     [Branch-M]     (F1_center)
 去噪+var_map       Retinex       光流对齐+conf      │
    │                   │              │              │
    └── UpsampleBlock(x, skip=F1) ─────┘──────────────┘
         bilinear×2 → concat(F1) → 3×3conv → GELU → 3×3conv
                        ↓
              Y_N / Y_L / Y_M (3ch, H)
                        ↓
              [AdaptiveFusion 4层权重网络]
                        ↓
                    O_t (3ch, H)
```

**参数量**：3.50M（Foxtrot 2.57M，增加来自：F1 skip 卷积 + Fusion 加深）

---

## 四、训练配置

```yaml
# configs/golf_r1.yaml
model:
  type: GolfNet
  encoder_channels: [32, 64, 128]
  tca_channels: 128
  branch_blocks: [3, 2, 3]     # N/L/M (N 从2增到3)
  fusion_blocks: 2
  gamma: 2.0

train:
  batch_size: 2
  epochs: 40
  lr: 0.0008
  warmup_epochs: 5
  grad_accum_steps: 8

loss:
  lambda_N: 0.3, lambda_L: 0.3, lambda_M: 0.3
  lambda_ortho: 0.01
  lambda_chess: 0.05      # [新]
  lambda_temp: 0.02       # [新]
  lambda_div: 0.05        # [新]
```

**LR schedule**（继承 train.py 的 phase 设计）：
- ep1-5 (warmup): 0.01→1.0 ×base
- ep6-10: 0.75×base
- ep11-25: 0.75→0.5×base
- ep26-40: 0.5×base

---

## 五、验证计划

| 阶段 | 检查项 | 判据 |
|------|--------|------|
| 启动 | 前 200 step loss 下降 | 1.77→0.89 ✓ (已验证) |
| ep10 | val PSNR | 对照 Foxtrot ep10 (18.10) |
| ep20 | val PSNR/SSIM | 关键判读点 |
| ep40 | 终判 + pair45 | vs Foxtrot ep70 (19.645/0.740/pair45 14.37) |
| 终判后 | **棋盘格复检** | FFT Nyquist 能量 + 视觉 |

**成功标准**：
1. pair45 PSNR ≥ 15.0（超越 Foxtrot 14.37）
2. val PSNR ≥ 19.7（持平/超越 Foxtrot ep60）
3. 推理帧无可见分格纹理（视觉） + 自相关第一峰 ≠ 2px（数值）

---

## 六、风险评估

| 风险 | 可能性 | 缓解 |
|------|:---:|------|
| F1 skip 引入噪声（F1 含低光噪声）| 中 | UpsampleBlock 的 3×3 conv 可学习抑制 |
| L_div 误导分支 | 中 | 权重小 (0.05)，且是弱引导 |
| L_chess 惩罚正常高频 | 低 | 只惩罚 Nyquist 十字带，非全频 |
| 参数量增加导致过拟合 | 低 | 仅 +0.9M，且有 SSIM/chess 正则 |
| 训练更慢 | 确定 | 实测 1.2 it/s (Foxtrot 1.23)，几乎无差异 |

---

## 七、与 Foxtrot 的对照总结

| 维度 | Foxtrot | Golf | 依据 |
|------|:---:|:---:|------|
| 上采样 | PixelShuffle(2) | resize-conv | 消除棋盘格 |
| F1 利用 | 闲置 | skip 接入三分支 | 恢复细节 |
| F3 利用 | 闲置 | 仍闲置（预留）| 本轮不动 |
| 推理缝合 | 均匀平均 | 余弦窗口 | 消除缝线 |
| 分支监督 | 同 GT | 同 GT + 差异化弱引导 | 强制分工 |
| 时序约束 | 无 | 高频稳定 | 防噪声放大 |
| Fusion 权重网 | 2层 | 4层 | 更强表达 |
| 残差 gamma | ≤0.5 | ≤0.9 | 中心帧贡献 |
| 参数量 | 2.57M | 3.50M | +36% |

---

## 八、后续候选（本轮不实现）

来自 `docs/paper_kb/ideas_top_transferable.md`：

- **S2 Morton 扫描**：替换 4 方向扫描，算力↓50%
- **S3 高分辨率 warp**：Branch-M 光流提升到 H（G2 已部分缓解）
- **A3 双向一致性**：GT→合成低光 循环一致性
- **F3 coarse context**：全局语义注入 TCA

**原则**：任何改动先做 10ep 快速验证，避免重蹈 Flight11 覆辙（未验证直接全量训练）。

---

## 附：工程产物

| 文件 | 说明 |
|------|------|
| `models/golf/upsample.py` | resize-conv 上采样 (G1) |
| `models/golf/encoder.py` | 共享编码器（沿用 Foxtrot）|
| `models/golf/tca_rwkv.py` | TCA-RWKV（沿用 Foxtrot）|
| `models/golf/branch_n/l/m.py` | 三分支 + F1 skip (G2) |
| `models/golf/fusion.py` | 融合（G7+G8）|
| `models/golf/loss.py` | GolfLoss（G4-G6）|
| `models/golf/golfnet.py` | 主网络 |
| `train_golf.py` | 训练脚本（与基础设施同构）|
| `configs/golf_r1.yaml` | 训练配置 |
| `utils/inference.py` | 余弦窗口 tiled_forward（G3，全局生效）|
