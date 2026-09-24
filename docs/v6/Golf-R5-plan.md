# Golf-R5 设计方案

> 日期：2026-09-24
> 动机：R4 技术修复全成功但性能未达标（19.80 vs R2 的 20.14）+ tile 边界伪影 + 泛化分裂
> 策略：**保守改进**——在 R4 基础上做**最小必要修改**，避免"过度工程化"

---

## 零、设计原则（吸取 R4 教训）

### R4 失败的核心教训

1. **修复≠提升**：R4 修复了 6 项技术缺陷（双零死锁/FiLM/prior/KV/L_temp/NaN），但 val PSNR 反降 -0.34 dB
2. **R2 的"缺陷"可能是隐式正则**：proj_out 零初始化可能非永久死锁，而是"延迟激活"机制
3. **泛化分裂未解决**：pair45 ep20=16.58 最优，ep60=15.89 退步（val 过拟合 + 运动场景欠拟合）
4. **tile 边界伪影**：256² patch 训练，全分辨率 tiled 推理，时序聚合（TCA）在 tile 间不连续

### R5 的设计约束

**DO：最小修改，直击核心问题**
1. 解决 tile 伪影（时序聚合保守化）
2. 解决泛化分裂（防 val 过拟合）
3. 保留 R4 的数值修复（NaN fix / loss 尺度不变 / 工程卫生）

**DON'T：大改架构**
1. 不再动 TCA 初始化（可能破坏 R2 的隐式正则）
2. 不再改 FiLM/KV（可能引入新瓶颈）
3. 不扩大模型（3.69M → 保持）

---

## 一、核心问题诊断

### 问题 1：tile 边界块状伪影（新问题）

**现象**：
- 推理输出 tile 边界处亮度跳变 ±5~15（0-255）
- 224px stride 边界，但 FFT 无周期峰（非全局网格）
- 内容相关的块间不连续

**根因**：
```
训练：256² patch，每个 patch 独立处理 5 帧 TCA 聚合
推理：1080×1920 tiled，每个 tile 独立处理 5 帧
→ tile 边界处的时序特征不连续（TCA 聚合依赖局部 5 帧窗口）
→ 相邻 tile 的融合输出不一致
```

**TOP idea 对应**：
- **DWTA-Net 动态权重融合**：静态区多用时序（降噪），动态区少用时序（保细节）
- **RetinexMCNet 两阶段**：单帧充分处理 → 时序一致性精调

**教训**：多帧聚合在 tiled 推理下不安全，需要"以中心帧为主，时序为辅"的保守策略。

### 问题 2：泛化分裂（R2/R4 共有）

**现象**：
- val PSNR 持续上升，pair45 在 ep20 达峰后下降
- ep60 val=19.80，pair45=15.89（比 ep20 的 16.58 差 -0.69 dB）

**根因**：
- 静态 val 集过拟合（539 clips × 6 次验证）
- 运动场景（pair45）欠拟合（训练集静态场景占比高）
- 256² patch 训练 vs 全分辨率验证的分布错配

**教训**：单一 val 集监控不可靠，需 pair45 早停或数据增强。

### 问题 3：R4 的性能未达 R2（核心矛盾）

**现象**：R4 解锁 8.4% 参数后，val PSNR 反降

**假设**：
1. FiLM/KV 共享引入新瓶颈
2. R2 的"双零死锁"可能非永久（延迟激活机制）
3. L_temp 噪声去相关过强，压制 M 分支高频

**策略**：R5 **回退部分 R4 修改**，保留工程修复，放弃可能有害的语义修复。

---

## 二、R5 核心改动（3 项）

### R5-1：动态时序权重门控（解决 tile 伪影）

**原理**：借鉴 DWTA-Net，在融合模块加入**基于运动残差的动态门控**：

```python
# 当前 R4 融合
O_t = refine(w_N·Y_N + w_L·Y_L + w_M·Y_M) + γ·residual

# R5 改为动态时序门控
residual_M = |Y_M - X_t|  # 运动残差（高→动态区，低→静态区）
ω_temporal = sigmoid(-α·residual_M)  # 静态→1（信任时序），动态→0（回退中心帧）
O_t_fused = refine(w_N·Y_N + w_L·Y_L + w_M·Y_M)
O_t = ω_temporal·O_t_fused + (1-ω_temporal)·X_t + γ·residual
```

**效果**：
- **tile 边界**（运动估计不准）→ residual 大 → ω↓ → 多用中心帧，减少跨 tile 时序不连续
- **静态区**（降噪主战场）→ residual 小 → ω↑ → 最大化时序平均
- **动态区**（细节保留）→ residual 大 → ω↓ → 保留中心帧，防鬼影

**改动量**：`fusion.py` 加 10 行，新增参数 `α`（可学习或固定=2.0）

**验证**：10-epoch 快速验证，目视检查 tile 伪影是否减轻，pair45 PSNR 是否持平。

---

### R5-2：pair45 早停 + 数据增强（解决泛化分裂）

**策略 A：早停策略调整**
```python
# 当前：val PSNR 最优存 best.pth
# R5：双指标早停
if pair45_psnr > best_pair45:
    save('best_pair45.pth')
if val_psnr > best_val:
    save('best_val.pth')
# 最终取 best_pair45.pth
```

**策略 B：数据增强（轻量）**
```python
# 训练时随机时序增强
- 帧序随机翻转（[t-2,t-1,t,t+1,t+2] → [t+2,t+1,t,t-1,t-2]）
- 帧随机 dropout（5 帧 → 随机丢 1 帧，复制中心帧填充）
```

**效果**：
- 早停避免 val 过拟合后继续训练
- 数据增强增加时序多样性，缓解静态场景偏置

**改动量**：`train_golf_r5.py` 加 20 行，`dataset.py` 加 15 行

---

### R5-3：选择性回退 R4 修改（避免过度工程化）

**回退清单**：

| R4 修改 | R5 决策 | 理由 |
|---------|---------|------|
| R4-P0 双零死锁修复（proj_out 非零初始化）| **回退 → R2 方案** | R2 可能非永久死锁，强行改可能破坏隐式正则 |
| R4-P1 FiLM LN 后生效 | **回退 → 删除 FiLM** | 表达力可能不足，成为瓶颈 |
| R4-P2 prior_L 低通门控 | **保留**（但降权重 0.01→0.005）| 非恒等是对的，但可能过强 |
| R4-P3 KV 共享全时序 | **回退 → R2 聚合统计量** | 全时序可能引入噪声，聚合量是有效先验 |
| R4-P4 L_temp 噪声去相关 | **回退 → R3 空间 HF 上限** | 去相关可能压制 M 分支，简单上限更稳健 |
| R4-P5 结构化先验正则 | **保留**（降权重）| 物理意义正确，但不要太强 |
| NaN 修复（upsample fp32）| **保留** | 工程必要，无副作用 |
| 损失尺度不变（k⁰）| **保留** | 工程必要，防发散 |

**关键回退**：
1. **TCA 初始化恢复 R2**：`proj_out=0, scale=0`（允许双零）
2. **删除 FiLM**：`use_f3_film=False`
3. **KV 恢复聚合**：`kv_proj(concat_time)` → `kv = mean(ctx_N/L/M)`
4. **L_temp 恢复空间 HF**：`|cos(HF(Y), HF(x_c-x̄))|` → `relu(|HF(Y)| - τ|HF(X_t)|)`

**哲学**：**"如果修复让性能变差，那不是 bug 而是 feature"**。R5 承认 R2 的"缺陷"可能是隐式正则，回退到 R2 基础上只加保守的工程修复。

---

## 三、R5 架构总览（vs R4 对比）

```
输入 (B,5,3,H,W)
    ↓ SharedEncoder（不变）
    ├─ F1 (32ch,H)     → 三分支 skip（不变）
    ├─ F2 (64ch,H/2)   → TCA 主输入（不变）
    └─ F3 (128ch,H/4)  → [R5: 删除 FiLM，F3 闲置或删除]
    ↓
TCA-RWKV:
  [R5: 恢复 R2 初始化] proj_out=0, scale=0（允许双零）
  [R5: 恢复 R2 KV 聚合] KV = proj(mean_ctx)（非全时序）
  Q_N/L/M = query(F2_center)          差异化查询（不变）
  attn_k  = RWKV(Q_k, KV)
  raw_k   = center + attn_k · scale_k
  F_k     = prior_k(raw_k)            [R5: prior 降权 λ=0.005]
    ↓
Branch-N/L/M（不变，除 loss）
    ↓
[R5-1] AdaptiveFusion + 动态时序门控:
  Y_fused = w_N·Y_N + w_L·Y_L + w_M·Y_M
  ω_temporal = sigmoid(-2.0·|Y_M - X_t|)
  O_t = ω_temporal·refine(Y_fused) + (1-ω_temporal)·X_t
```

**参数量**：3.69M → **3.55M**（删除 FiLM 的 6C 投影，≈40K 参数）

---

## 四、损失函数调整

| 项 | R4 | R5 | 变化 |
|----|----|----|------|
| L_final | recon(O_t, GT) | 同 | — |
| L_N/L/M | 0.3 × recon | 同 | — |
| L_ortho | 0.01 × ortho | 同 | — |
| L_temp | 0.02 × 噪声去相关余弦 | **0.02 × 空间 HF 上限**（R3 版本）| 回退 |
| L_prior | 0.01 × 尺度不变 | **0.005** × 尺度不变 | 降权 |
| L_div | 0.05 × diversity | 同 | — |

**L_temp 公式变化**：
```python
# R4（噪声去相关）
noise_hf = hf(x_c - x̄)
L_temp = |cos(HF(Y_N), noise_hf)| + |cos(HF(L_t), noise_hf)|

# R5（空间 HF 上限，R3 版本）
L_temp_N = relu(|HF(Y_N)| - 1.2·|HF(X_t)|).mean()
L_temp_L = relu(|LF(Y_L)| - 1.2·|LF(x̄)|).mean()  # L 用低频
L_temp = L_temp_N + L_temp_L
```

---

## 五、实验设计

### 5.1 快速验证（10 epoch）

**目标**：验证动态时序门控对 tile 伪影的改善

**配置**：
- 基线：R4 best.pth（ep60）推理 pair45
- R5：从 R2 ep40 checkpoint（pair45 最优点 16.06）finetune 10 epoch
- 对比：tile 伪影（人工检查）+ pair45 PSNR

**成功标准**：
- tile 伪影视觉减轻
- pair45 PSNR ≥ R2 ep40（16.06）

### 5.2 完整训练（60 epoch）

**起点**：R2 checkpoint（不从头训，利用 R2 的隐式正则）

**监控**：
- val PSNR（参考）
- **pair45 PSNR（主指标）**
- conv1_max（>1500 预警）
- NaN 计数

**早停**：pair45 连续 3 次验证不提升

**验收标准**：

| 指标 | 目标 | 理由 |
|------|------|------|
| pair45 PSNR | **≥ 17.0** | 原始目标，R2/R4 均未达 |
| val PSNR | ≥ 19.5 | 允许略低于 R4（19.80），但不能崩 |
| tile 伪影 | 视觉可接受 | 主观评估，无定量标准 |
| NaN | 0 | 保留 R4 修复 |

---

## 六、风险评估

| 风险 | 概率 | 缓解 |
|------|:---:|------|
| 回退双零死锁后真的永久冻结 | 中 | 监控梯度，若冻结立即终止 |
| 删除 FiLM 后表达力不足 | 低 | R2 无 FiLM 达 20.14，证明非必要 |
| 动态门控过强，抑制时序降噪 | 中 | α=2.0 可调，消融验证 |
| pair45 早停过早，val 性能差 | 低 | 保存双 checkpoint（best_val + best_pair45）|

---

## 七、实现优先级

### P0（立即实现，10-epoch 验证）

1. `models/golf_r5/fusion.py`：动态时序门控（R5-1）
2. `configs/golf_r5.yaml`：从 R2 config 复制，关闭 FiLM
3. `train_golf_r5.py`：pair45 早停逻辑
4. 烟雾测试：单步 forward + loss 计算

### P1（快速验证通过后）

5. `models/golf_r5/tca_rwkv.py`：恢复 R2 KV 聚合 + 双零初始化
6. `models/golf_r5/loss.py`：L_temp 回退 R3 版本
7. 完整训练 60 epoch

### P2（可选，按需）

8. 数据增强（帧序翻转 / dropout）
9. 高分辨率 warp（S3）

---

## 八、成功标准与止损线

### 成功（进入 R6）

- pair45 PSNR ≥ 17.0
- tile 伪影视觉可接受
- val PSNR ≥ 19.5

### 止损（转向 Flight11）

- 10-epoch 验证 pair45 < 16.0（R2 基线）
- 或完整训练 val PSNR < 19.0（性能崩溃）

**止损判断**：若 R5 仍失败，说明 **Golf 系列（3.7M 轻量三分支）已达极限**，需转向 Flight11 的 T-BC1b 主干（6.8M，更强归纳偏置）。

---

## 九、与长期计划的关系

**TSDR 框架契合度**：
- R5 的动态时序门控 = TSDR "动态源不可平均"的显式实现
- 回退 KV 聚合 = 承认"统计量先验"比"全时序"更鲁棒（TSDR 的 DPE 职责）
- pair45 早停 = 承认 val 集无法代表运动场景（TSDR Type IV 欠拟合）

**与 Flight11 的分界**：
- Golf-R5：保守修复，保持 3.7M 轻量
- Flight11 T-BC1b：若 R5 失败，承认需要 6.8M + 更强时序建模

---

## 十、下一步行动

1. **创建 R5 目录结构**：`models/golf_r5/`（复制 R4，修改 fusion + tca + loss）
2. **实现 R5-1 动态门控**：`fusion.py` 10 行
3. **配置文件**：`configs/golf_r5.yaml`（关闭 FiLM，降 prior 权重）
4. **烟雾测试**：单步 forward + loss，确认无语法错误
5. **10-epoch 快速验证**：从 R2 ep40 finetune
6. **评审决策点**：快速验证通过 → 完整训练；失败 → 转 Flight11

---

**设计完成时间**：2026-09-24  
**预计实现时间**：2h（P0）+ 4h（P1）= 6h  
**预计训练时间**：10-epoch 验证（10h）+ 完整训练（60h，若通过验证）
