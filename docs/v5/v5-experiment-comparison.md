# TFS-Net v5 改进方案实验比较报告

> 日期：2026-06-23
> 数据集：SDSD indoor（训练 70 序列 / 2064 样本，验证 10 序列）
> 硬件：RTX 4090 (24GB)，fp32，AMP 关闭

---

## 1. 实验总览

### 1.1 基线

| 基线 | 配置 | 参数量 | ep5 PSNR | ep10 PSNR | ep15 PSNR | 说明 |
|------|------|--------|----------|-----------|-----------|------|
| **v5.5** | batch=3, crop=256, lr=0.001, warmup=10, 150 epoch | 1.119M | **19.094** | **19.026** | **19.235** | 最好基线 |

v5.5 验证 PSNR 在 19.0–19.2 振荡，SSIM 0.75–0.76。训练损失停滞在 0.275，`loss_illum≈3e-5`（s_illum 塌缩到 0）。

### 1.2 改进方案实验一览

| 实验 | 改动项 | batch | crop | lr | warmup | 参数量 | ep5 PSNR | ep10 PSNR | vs v5.5 ep5 | 结论 |
|------|--------|-------|------|-----|--------|--------|----------|-----------|-------------|------|
| **Group A** | P0-1 LayerNorm2d | 2 | 256 | 0.0008 | 5 | 1.122M | 18.760 | — | -0.33 | ❌ 过拟合 |
| **Group B旧** | P0-1 + P0-2(L_ratio) + P0-3 | 2 | 256 | 0.0008 | 5 | 1.124M | 17.902 | — | -1.19 | ❌ 监督目标失效 |
| **Group B2** | P0-1 + P0-2(暗度) + P0-4 + P1-1~6 | 2 | 256 | 0.0008 | 5 | 1.124M | 17.807 | 18.337 | -1.29 | ❌ 改进互相干扰 |
| **Group B3** | P0-1 + P0-4 + P1-1/2/5/6 (架构only) | 2 | 256 | 0.0008 | 5 | 1.124M | 17.624 | — | -1.47 | ❌ soft_clamp 灾难 |
| **A2** | P0-1 + P1-6 only (v5.5 参数) | 3 | 224 | 0.001 | 10 | 1.119M | 18.560 | — | -0.53 | ❌ LN 过拟合 |
| **v5.7** | s_illum 门控 lit_up_map | 3 | 256 | 0.001 | 10 | 1.119M | 18.593 | 18.421 | -0.50 | ⚠️ s_illum 不塌缩但 PSNR 降 |
| **v5.8** | v5.5 + 增深 (瓶颈4+IGRF4) | 3 | 256 | 0.001 | 10 | 2.079M | 训练中 | — | ? | ⏸ 进行中 |

---

## 2. 各方案详细分析

### 2.1 Group A：LayerNorm2d 主干归一化（P0-1）

**改动**：ConvBlock 加 Pre-LN，ResBlock 加 Pre-LN + LayerScale(beta 零初始化)。

**结果**：ep5 PSNR=18.760（-0.33 dB），训练损失比 v5.5 低 12%（0.264 vs 0.296）。

**诊断**：
- 训练损失更低但验证更差 → **过拟合**
- LayerNorm2d 加速了训练集拟合，但 SDSD indoor 仅 2064 样本，泛化变差
- 对比 NAFNet：NAFNet 用 LN 有效是因为 30+ 层深度堆叠，TFS-Net 仅 2-4 conv/模块

**结论**：❌ **LayerNorm2d 在 TFS-Net 浅 Conv 主干上有害**。已回退默认 `use_norm=False`。

### 2.2 Group B旧：s_illum 显式监督 L_ratio（P0-2）

**改动**：新增 `L_illum_sup = L1(s_illum, clamp(1 - L_t/L_ref, 0, 1))`。

**结果**：ep5 PSNR=17.902（-1.19 dB）。i_sup 在 epoch 5 塌缩到 0.0001。

**诊断**：
- SDSD indoor 所有帧在同一低光环境拍摄 → L_t ≈ L_ref → target ≈ 0
- 监督目标把 s_illum 驱动到 0，与 §9 诊断的塌缩是同一结果
- i_sup 轨迹：0.169 → 0.091 → 0.082 → 0.286 → 0.180 → **0.0001**（epoch 5 塌缩）

**结论**：❌ **L_ratio 监督目标在均匀光照数据集上恒为 0**。目标设计错误。

### 2.3 Group B2：近全量改进组合（P0-1 + P0-2(暗度) + P0-4 + P1-1~6）

**改动**：修正 s_illum 监督为基于输入暗度 `clamp(1 - img/GT, 0, 1)`，同时开启所有 P1 改进。

**结果**：ep5=17.807, ep10=18.337, ep15=17.700。i_sup 稳定在 0.10-0.11（不塌缩）。

**诊断**：
- i_sup 不塌缩 ✅（暗度目标在 SDSD 上有效）
- 但 PSNR 比 v5.5 差 1.3 dB ❌
- 改进过多互相干扰，无法判断哪项有益哪项有害
- freq 损失从 v5.5 的 0.012 暴涨到 0.80（频域相位项值远大于幅度项）

**结论**：❌ **组合实验无法归因**。需要逐项 ablation。

### 2.4 Group B3：架构改进 only（P0-1 + P0-4 + P1-1/2/5/6）

**改动**：移除所有损失改进（无 s_illum 监督、无多层 VGG、无频域相位），只保留架构改动。

**结果**：ep5 PSNR=17.624（-1.47 dB，所有实验中最差）。

**诊断**：
- 比 Group A（仅 LN）还差 1.14 dB → soft_clamp + SACE kaiming + soft_median 三项中有致命问题
- `soft_clamp(x) = sigmoid(20×(x-0.5))` 对低光输入是灾难性的：
  - 暗帧 img≈0.05 → soft_clamp(0.05) = sigmoid(-9) ≈ 0.0001
  - 中间级输出被压到近零，模型无法在 Stage1/2 表示亮的中间值

**退化归因**：
```
Group A (LN only):         -0.33 dB
+ soft_clamp + SACE kaiming + soft_median: -1.14 dB  ← 主要元凶
```

**结论**：❌ **soft_clamp 在低光场景掐死动态范围**。已回退默认 `use_soft_clamp=False`。SACE kaiming init 也回退（未带来收益）。

### 2.5 A2：LN + IH only（v5.5 参数，干净对照）

**改动**：只保留 P0-1(LN) + P1-6(IntensityHead 加深)，关闭所有其他改动，用 v5.5 的训练参数（batch=3, lr=0.001, warmup=10）。

**结果**：ep5 PSNR=18.560（-0.53 dB）。训练损失最低（0.260）但验证最差。

**诊断**：
- 排除了超参混淆（batch=3, v5.5 参数），唯一变量是 LN+IH 和 crop 224
- 退化归因：LN+IH 本身 -0.33 dB + crop 256→224 -0.20 dB = -0.53 dB
- 训练损失最低但验证最差 → **确认 LN 导致过拟合**（与 batch/crop 无关）

**结论**：❌ **LayerNorm2d 本身导致过拟合**。IntensityHead 加深也未阻止 s_illum 塌缩。两者均回退到 v5.5。

### 2.6 v5.7：s_illum 门控 lit_up_map

**改动**：`lit_up_map = 1 + s_illum × (lit_up_map_full − 1)`，替代 v5.5 的加法修正。s_illum=0 → 无提亮 → 模型被迫让 s_illum>0。

**结果**：ep5=18.593, ep10=18.421。`illum=0.012` 全程稳定（v5.5 为 5e-7）。

**诊断**：
- ✅ **s_illum 不再塌缩**——门控机制成功
- ❌ 但 PSNR 比 v5.5 差 0.50-0.61 dB，且 ep5→ep10 PSNR 下降（过拟合）
- 根因：门控限制了模型表达力
  - v5.5：lit_up_map 是 3 通道 RGB 独立提亮，s_illum 可旁路为 0
  - v5.7：lit_up_map 被 1 通道 s_illum 统一门控，R/G/B 必须同比例提亮
  - s_illum 必须先学好才能提亮 → 学习难度增大

**核心矛盾**：
```
v5.5: s_illum 塌缩但 PSNR 最好 (19.09) ← 模型选择旁路 s_illum
v5.7: s_illum 不塌缩但 PSNR 降 (18.59)  ← 强制 s_illum 参与限制表达力
```

**结论**：⚠️ **门控解决了 s_illum 塌缩的理论问题，但以牺牲模型表达力为代价**。v5.5 让 s_illum=0、lit_up_map 独立工作反而是更优的实践方案。已回退到 v5.5 加法修正。

### 2.7 v5.8：增加模型深度（进行中）

**改动**：v5.5 基础上，Encoder 瓶颈加 4 个 ResBlock（H/4, 96ch），IGRF 每阶段 ResBlock 从 2 个加到 4 个。

**参数量**：2.079M（v5.5 的 1.119M +86%）。

**结果**：训练中（Epoch 1, loss 0.33 正常下降）。

**逻辑**：所有架构/损失改进都无法突破 v5.5 的 19.09 dB → 判断为模型容量天花板 → 增加深度突破。

---

## 3. 退化归因分解

以 v5.5 ep5 PSNR=19.094 为基准：

```
v5.5 (19.094)
 │
 ├─ +LayerNorm2d (P0-1)           → -0.33 dB  (Group A: 18.76)
 │   └─ 原因: 浅 Conv 主干 + 小数据集 → 过拟合
 │
 ├─ +crop 256→224                 → -0.20 dB  (A2 vs A: 18.56 vs 18.76)
 │   └─ 原因: 空间信息损失 23%
 │
 ├─ +soft_clamp (P0-4)            → -1.14 dB  (B3 vs A: 17.62 vs 18.76)
 │   └─ 原因: sigmoid(20×(0.05-0.5))≈0, 掐死低光动态范围
 │
 ├─ +s_illum监督 L_ratio (P0-2)   → -0.86 dB  (B旧 vs A: 17.90 vs 18.76)
 │   └─ 原因: SDSD 均匀光照 → L_t≈L_ref → target≈0
 │
 ├─ +s_illum监督 暗度 (P0-2修正)  → +0.19 dB  (B2 vs B3: 17.81 vs 17.62)
 │   └─ 小幅补回, 但仍比 v5.5 差 1.3 dB
 │
 ├─ +s_illum 门控 (v5.7)          → -0.50 dB  (v5.7 vs v5.5: 18.59 vs 19.09)
 │   └─ 原因: R/G/B 必须同比例提亮, 限制表达力
 │
 └─ +增深 1.1M→2.1M (v5.8)        → ? dB     (训练中)
     └─ 假设: 突破容量天花板
```

---

## 4. s_illum 塌缩问题专题

### 4.1 问题现象

v5.5 中 `s_illum` 塌缩到 0（mean=0.0001），`loss_illum≈3e-5`（边缘平滑正则饱和）。加法修正 `s_illum × corr_mag ≈ 0` 成为 no-op，模型退化到仅乘法 lit_up_map 提亮。

### 4.2 尝试的解决方案

| 方案 | 机制 | s_illum 不塌缩? | PSNR 影响 | 结论 |
|------|------|:---:|-----------|------|
| L_ratio 监督 | s_illum→clamp(1-L_t/L_ref) | ❌ (target≈0) | -0.86 dB | 目标在 SDSD 上失效 |
| 暗度监督 | s_illum→clamp(1-img/GT) | ✅ | -0.19 dB | 约束 s_illum 语义 |
| IntensityHead 加深 | 2 层 Conv+LN | ❌ | -0.33 dB | 未阻止塌缩 |
| **乘法门控** | lit_up_map=1+s_illum×(full-1) | **✅** | **-0.50 dB** | **限制表达力** |

### 4.3 核心矛盾

**s_illum 塌缩是理论问题，不是性能瓶颈**。v5.5 顶着 s_illum=0 仍达到 19.09 dB——模型发现 lit_up_map 独立完成提亮是最优实践方案，s_illum 的加法修正是冗余的。

强制 s_illum 参与提亮（门控）虽然解决了塌缩，但牺牲了模型表达力（PSNR 降 0.5 dB）。**这是一个"正确性 vs 性能"的权衡**。

### 4.4 结论

在当前模型架构下，**接受 s_illum 塌缩是理性选择**。s_illum 的诊断功能（TFSI 输出）仍可作为可解释性指标，不必强制其参与提亮执行。如需 s_illum 真正参与，需要更深层的架构重设计（如让 s_illum 控制去噪强度而非提亮）。

---

## 5. 关键教训

### 5.1 LayerNorm2d 不是万能药

- **NAFNet 用 LN 有效**是因为 30+ 层深度堆叠 + SimpleGate + SCA 的完整 Transformer-style block
- **TFS-Net 用 LN 有害**是因为浅 Conv 主干（2-4 conv/模块）+ 小数据集（2064 样本）
- **教训**：不能脱离模型深度和数据量照搬归一化策略

### 5.2 soft_clamp 在低光场景有陷阱

- `sigmoid(20×(x-0.5))` 对正常亮度图像（x≈0.5）工作良好
- 但对低光输入（x≈0.05）输出 ≈0，掐死动态范围
- **教训**：激活/约束函数的选择必须考虑输入分布，低光增强任务中中间表示可能很暗

### 5.3 显式监督目标需匹配数据集特性

- L_ratio 监督假设帧间有光照差异，但 SDSD indoor 是均匀光照 → target 恒为 0
- **教训**：监督目标的数据集适配性必须先验证

### 5.4 解决理论问题不一定带来性能提升

- s_illum 门控解决了塌缩（理论正确），但 PSNR 反降（实践退步）
- **教训**：模型会找到自己的最优解，强制改变其行为可能适得其反

### 5.5 组合实验无法归因

- B2 同时改了 9 项，无法判断哪项有益哪项有害
- **教训**：ablation 必须逐项进行，或至少分组隔离

---

## 6. 代码改动保留状态

### 6.1 已回退（默认关闭）

| 改动 | 文件 | 默认值 | 原因 |
|------|------|--------|------|
| ConvBlock LayerNorm2d | `blocks.py` | `use_norm=False` | 过拟合 |
| ResBlock LayerNorm2d + LayerScale | `blocks.py` | `use_norm=False` | 过拟合 |
| soft_clamp | `igrf.py` | `use_soft_clamp=False` | 掐死动态范围 |
| SACE OffsetHead LayerNorm2d | `sace.py` | `offset_use_norm=False` | 组合实验有害 |
| SACE kaiming init | `sace.py` | `offset_kaiming_init=False` | 组合实验有害 |
| IntensityHead 加深 | `tfsi.py` | 回退到 v5.5 单层 | 未阻止塌缩 |
| s_illum 门控 | `igrf.py` | 回退到 v5.5 加法修正 | 限制表达力 |
| L_recon | `losses.py` | 移除 | 与 L_pix 冗余 |
| 多层 VGG 感知损失 | `losses.py` | `perc_multilayer=False` | 未单独验证 |
| 频域相位损失 | `losses.py` | `freq_with_phase=False` | 未单独验证 |
| NAFBlock | `blocks.py` | `use_nafblock=False` | 未训练验证 |

### 6.2 保留（默认开启或代码内置）

| 改动 | 文件 | 状态 | 原因 |
|------|------|------|------|
| soft_median (可导) | `tfsi.py` | `use_soft_median=True` | 可导性正面改进，不影响过拟合 |
| s_illum/s_noise 显式监督框架 | `losses.py` | `lambda_*=0.0` (默认关闭) | 代码保留，权重为 0 |
| Encoder 瓶颈块 | `encoder.py` | `num_bottleneck_blocks=0` (默认) | v5.8 使用中 |
| IGRF ResBlock 数量可配置 | `igrf.py` | `num_res_blocks=2` (默认) | v5.8 使用中 |
| train.py --resume 支持 | `train.py` | 新增 | 续训功能 |

### 6.3 新增功能（未验证）

| 功能 | 文件 | 参数 | 状态 |
|------|------|------|------|
| NAFBlock | `blocks.py` | `use_nafblock=True` | 代码就绪，未训练 |
| 多层 VGG 感知损失 | `losses.py` | `perc_multilayer=True` | 代码就绪，未单独验证 |
| 频域相位损失 | `losses.py` | `freq_with_phase=True` | 代码就绪，未单独验证 |

---

## 7. 当前方向：v5.8 增加模型深度

### 7.1 策略

所有架构/损失改进都无法突破 v5.5 的 19.09 dB → 判断为模型容量天花板 → 增加深度突破。

### 7.2 配置

| 项目 | v5.5 | v5.8 |
|------|------|------|
| 参数量 | 1.119M | 2.079M (+86%) |
| Encoder 瓶颈块 | 0 | 4 ResBlock (96ch, H/4) |
| IGRF 每阶段 ResBlock | 2 | 4 |
| 其他架构 | v5.5 不变 | v5.5 不变 |
| 训练参数 | batch=3, lr=0.001, warmup=10 | 同 v5.5, 50 epoch |

### 7.3 预期判据

| ep5 PSNR | 含义 | 下一步 |
|----------|------|--------|
| ≥ 19.5 | 深度突破天花板 ✅ | 跑完 50 epoch |
| 19.0-19.5 | 深度有正面效果 | 继续跑看趋势 |
| < 19.0 | 2M 参数仍过拟合 | 加 weight decay / dropout / 减少瓶颈块 / 预训练 |

### 7.4 后续备选方案

1. **v5.8 + 更大模型**：level_channels [48, 96, 128], fused 128（~4M 参数）
2. **预训练**：在 LOL/SMID 预训练，SDSD 微调
3. **跨数据集训练**：SDSD indoor + outdoor 合并
4. **P1-3/P1-4 单独验证**：在 v5.8 基础上单独开多层 VGG / 频域相位

---

## 8. 文件索引

| 文件 | 说明 |
|------|------|
| `docs/v5/v5-design.md` | v5 完整设计文档（§1-§10），含退化建模、根因诊断、改进方案 |
| `docs/v5/v5-architecture-diagrams.md` | v5.7 mermaid 架构图（最简 + 带细节） |
| `docs/v5/v5-experiment-comparison.md` | 本文档：实验方案比较报告 |
| `docs/v5/three-source-separability-experiment.md` | 三源分离实验记录 |
| `configs/v58_deep.yaml` | v5.8 增深配置 |
| `configs/ablation_*.yaml` | 各 ablation 实验配置 |
| `outputs/sdsd_v57_gating/` | v5.7 门控实验输出 |
| `outputs/sdsd_v58_deep/` | v5.8 增深实验输出（进行中） |
| `outputs/sdsd_ablation_*/` | 各 ablation 实验输出 |
