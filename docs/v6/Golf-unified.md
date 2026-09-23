# Golf 系列统一文档（Foxtrot → R1/R2 → R3 → R4）

> 整合原 `Golf-plan.md` / `Golf-R2-analysis.md` / `Golf-R3-plan.md` / `Golf-R3-implementation-summary.md` / `Golf-R3-Ltemp-analysis.md` / `Golf-R4-loss-audit.md` / `Golf-R4-NaN-diagnosis.md` 七份文档
> 更新时间：2026-09-23
> 当前状态：**Golf-R4 重启训练中**（outputs/golf_r4，从 ep30 恢复，已修复三项损失缺陷 + NaN 溢出，keepalive pid 37746）

---

## 零、一句话总览

Golf 是 **Foxtrot 的轻量化修复线**：主干从 Foxtrot 的 2.57M 三分支升级到 3.69M（含 F3 FiLM/KV 投影），修复了 Foxtrot 的棋盘格伪影与结构缺陷；R2 首次突破 20 dB；R3 引入 5 项改进但暴露出**乘法双零死锁**（8.4% 参数永久冻结）；R4 系统性修复后重新训练。

| 版本 | 参数 | val PSNR | pair45 | 状态 |
|------|:---:|:---:|:---:|------|
| Foxtrot ep70 | 2.57M | 19.72@60 | 14.37 | 有棋盘格伪影 |
| **Golf-R2 ep60** | 3.50M | **20.14** | 12.80（ep60）| 突破 20 dB，但存在双零死锁 |
| Golf-R3（ep6 停止）| 3.55M | — | — | 双零死锁未修复，归档 `golf_r3_deadlock_ep6` |
| **Golf-R4** | 3.69M | 训练中（ep31+）| 训练中 | 全部修复 + NaN 修复，从 ep30 重启 |

---

## 一、Foxtrot 的问题诊断（Golf 的动机）

### 1.1 棋盘格/分格伪影 — 两个独立来源

**来源 A（主因）：`tiled_forward` 均匀平均**
- overlap 区权重=2.0、非重叠区权重=1.0，直接 `sum/weight`
- 1920×1080 按 tile=256/overlap=32 → stride=224 → 每 224px 一条亮度接缝
- **推理期问题，与模型无关**

**来源 B（次因）：三分支 PixelShuffle 上采样**
- `Conv2d(C, C*4, 1×1) → PixelShuffle(2)`，1×1 卷积无感受野
- 子像素位置由单一通道决定，相邻像素零通信 → 2px 周期棋盘格（自相关第一峰=2px）
- Odena et al. 2016 经典问题

### 1.2 结构缺陷

| # | 问题 | 证据 |
|---|------|------|
| P1 | F1/F3 编码器特征闲置 | `tsdnet.py:148-150` 构造后无引用，~60% 编码器算力浪费 |
| P2 | 全流程 H/2 处理，无 skip | TCA/三分支全在 128×128 |
| P3 | 三分支同监督同一 GT | 功能冗余 |
| P4 | 时序一致性缺失 | `lambda_temp=0` |
| P5 | Branch-M 光流分辨率不足 | H/2 估计，SDSD ±4px→±2px |

---

## 二、Golf 的核心改动（R1→R2）

| # | 改动 | 对应问题 |
|---|------|---------|
| G1 | **resize-conv 上采样**（bilinear + 3×3 conv）| 消除 2px 棋盘格（来源B）|
| G2 | **F1 skip 接入三分支** | P1+P2 |
| G3 | **余弦窗口 tiled_forward**（窗口下限 0.1）| 消除 224px 缝线（来源A）|
| G4 | Fusion 权重网络 2层→4层 | 表达力 |
| G5 | 残差 gamma 上界 0.5→0.9 | 中心帧贡献 |

### 2.1 R2 终判（outputs/golf_r2，ep60）

| 指标 | ep10 | ep20 | ep30 | ep40 | ep50 | **ep60** |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| PSNR | 17.650 | 18.063 | 17.798 | 18.868 | 17.980 | **20.140** |
| SSIM | 0.695 | 0.715 | 0.720 | 0.726 | 0.720 | **0.740** |
| LPIPS | 0.348 | 0.340 | 0.340 | 0.329 | 0.329 | **0.314** |

- **首次突破 20 dB**；ep50→60 跃升 +2.16 dB（LR 降档 1e-4 触发）
- 训练损失 ep41-60 持续下降 0.0208

### 2.2 R2 的泛化分裂（重要教训）

| 权重 | val PSNR | pair45 PSNR |
|------|:---:|:---:|
| ep40 | 18.868 | **16.063**（前30帧）|
| ep60 | **20.140** | 12.797（全131帧）|

**val 提升但 pair45 退步** = 分布内 vs 分布外泛化分离：ep51-60 小 LR 精修过拟合静态 val 分布。**ep40 才是 pair45 最优点**。
→ 教训：单一静态 val 集不足以作停止标准，需 pair45 专项监控。

---

## 三、R3 的 5 项改进（已实施，但暴露死锁）

| ID | 名称 | 目标 |
|----|------|------|
| R3-A | conf 中性起点（bias 2.0→0.0）| 修复 conf_map 饱和（R2 ep20=1.000）|
| R3-B | Branch-M 高分辨率 warp（H/2→H 域）| 修复 warp_t 贡献崩溃（R2 ep20=0.03）|
| R3-C | F3 FiLM 退化感知调制 | 增强分支专属性 |
| R3-D | pair45 双验证集 | 防止 val 过拟合 |
| R3-E | L_temp 分支选择性 + 阈值 1.2 | 解放 M 分支 |

### 3.1 R3 用户驱动的 L_temp 分析（问题 2 的前身）

原 L_temp 作用于全局 O_t，会无差别惩罚所有分支高频，**包括 M 分支合理的细节恢复**。分析结论：
- N 分支（去噪）应约束高频
- L 分支（光照）应约束高频
- **M 分支（运动对齐）应豁免**

R3 实施了"仅约束 N+L"，但 **L_temp 本身仍是空间 HF 上限，非真时序**（详见 §五问题2）。

---

## 四、审查发现的系统性问题（R4 的动机）

2026-09-20 对 R3 的独立数值审查，发现 **比原审计报告更严重** 的问题：

### 4.1 [P0] 乘法双零死锁（原审计未发现根因，影响面 6× 于审计报告）

**现象**：TCA 注意力核心参数梯度精确为 0，永久冻结。

**真实根因**（非审计所述 LayerNorm）：
```
attn 输出 = proj_out(...) × scale_N
  proj_out 零初始化 (blocks 传统)
  scale_N  零初始化 (LayerScale 传统)
→ 梯度链: ∂L/∂proj_out 需 scale≠0; ∂L/∂scale 需 proj_out≠0
→ 两者互为条件, 精确为 0, 永久死锁
```

**数值实锤**：

| 配置 | proj_out.grad | scale.grad | 结果 |
|------|:---:|:---:|------|
| proj=0, scale=0（Golf 实际）| 0 | 0 | **双零死锁 ✗** |
| proj≠0, scale=0（ConvNeXt 标准）| 0 | ≠0 | 可解锁 ✓ |
| NAFBlock（conv≠0, gamma=0）| ≠0 | ≠0 | 单零可解锁 ✓ |

**影响面**：`attn_N/L/M` + `query_N/L/M` + `scale_N/L/M` + `f3_film` = **298,752 参数（8.4%）全死**。

**全项目扫描**：

| 模型 | 判读 |
|------|------|
| TBC1B / TBC / TA / BC / A / v2 / v3 / sdsd_f10m5 / sdsd_f11_simple2 | 健康 ✓ |
| **foxtrot_r1 / golf_r1 / golf_r2 / golf_r3 / sdsd_f11 / sdsd_f11_simple** | **双零死锁 ✗** |

**为何 TBC1B 逃逸**：`channel_mix` 有非零 bias（旁路），提供了一条不经过 proj_out×scale 的梯度通路。

### 4.2 [P1] R3-C FiLM 死模块（审计指出，但根因不同）

- 审计归因：LayerNorm 对正标量缩放不变（LN(αx)≡LN(x)）
- **实测根因**：双零死锁（即使随机初始化 f3_film，梯度仍精确 0）
- 且原实现 gate 乘在 LN 之前，per-channel 缩放本可在 LN 后生效
- 验证：`两个不同 f3_ctx → F_N 差异 = 0.00e+00`，`f3_film.weight.grad = 0.000e+00`

### 4.3 [P2] prior_L 代数恒等（原审计未发现）

```python
enhanced = feat * gate + feat * (1 - gate) ≡ feat   # 完全 no-op
```
实测差异 2.384e-07（浮点噪声级）。

### 4.4 [P3] K/V 设计偏离 Foxtrot 原设计

- 原设计（TSD-Foxtrot.md §3.2）：`K = V = Concat_time(F_{t±i})` — 三路共享全时序
- Golf 实现：用 `ctx_N/L/M`（mean/smooth/diff 聚合统计量）作 KV，**丢失逐帧细节**

### 4.5 [P4] L_temp 非真时序（用户问题 2）

原实现 `relu(|HF(Y)| - 1.2|HF(X_t)|)` 比较的是 **Y_branch vs 同帧 X_t**，**无任何跨帧输入**。命名 "temporal" 但实现是 "spatial HF cap"。

---

## 五、用户三个设计问题的回答

### 问题 2：L_temp 对 N/L 分支的物理意义？能否覆盖两种噪声？

**当前 R3 实现（空间 HF 上限）不符合物理意义** —— 它比较同帧 Y vs X_t，没有跨帧信息，既不能约束"噪声形成的帧间稳定"，也不能约束"光照的帧间不突变"。

**R4 的真时序重设计**：

**N 分支（去噪）— 覆盖 Read + Shot 两类噪声**：
```
成像噪声 = Read Noise (信号无关) + Shot Noise (信号相关, Poisson)
两者共享【跨帧 i.i.d.】性质 → 时间平均是最优估计器 (Foi 2008)
   - Read: 高斯, 时间平均后方差/N
   - Shot: 泊松, 时间平均后方差/N
→ 同一策略覆盖两类 (正是 TSDR 合并 I+II 的依据)
实现: L_temp_N = |HF(Y_N) - HF(x̄)|   (x̄ = 5帧窗口时间均值)
```

**L 分支（光照）— 帧间缓变一致性**：
```
光照扰动 = 帧间强相关、慢变 → 相邻帧照度几乎相同
→ 时间均值 x̄ 的低频 ≈ 每帧低频
→ L_temp_L = |LF(Y_L) - LF(x̄)|   (大核15近似更低频带)
```

**回答**：
1. R3 实现**不符合**物理意义（无跨帧）
2. R4 的 `|HF(Y_N)-HF(x̄)|` **能覆盖两种细分噪声**（Read+Shot 共享 i.i.d. 时间平均性质）
3. R3 的方法对 N/L 的"帧间一致性约束"**无效**（本质是空间能量上限）；R4 用时间均值作参考才真正实现

### 问题 3：K/V 与差异化先验为何不实现？

**（1）K/V 来源：三路共享全时序拼接 `K=Concat_time(F_{t±i})`**
- 原 Golf 实现用了 `ctx_N/L/M` 聚合统计量作 KV → 信息坍缩
- **R4 已实现**：`kv_proj(Concat_time(feats_shifted))` 共享 KV + 差异化 Q
- 未在 R1-R3 实现的原因：早期实现图省算力用了聚合量，R4 补上

**（2）分支差异化手段：N 熵正则 / L DCT低频+全局池化 / M top-k稀疏+位置偏置**
- R3 的 prior_N/L/M 全部退化（prior_L 是恒等，prior_N 只是加常数，prior_M 无稀疏）
- **R4 已实现物理化版本**：
  - N: `feat + scale·mean_ctx` + MSE(feat, mean) 一致正则（熵最大化的近似）
  - L: `feat + scale·lowpass(feat)` + 空间 TV 正则（低频约束）
  - M: `feat + scale·diff_proj(center-context)` + L1 稀疏正则
- DCT/top-k 未逐字实现的原因：DCT 与"大核低通"在效果上等价但计算重；top-k 在 RWKV 的 O(N) 线性扫描上不自然（RWKV 无显式注意力矩阵可 top-k）。R4 用等价的正则项（TV/L1）实现相同物理意图。

---

## 六、Golf-R4 完整修复清单

| # | 修复 | 文件 | 验证 |
|---|------|------|------|
| R4-P0 | proj_out 零初始化 → 标准小初始化（解双零死锁）| tca_rwkv.py | 8 步后全部参数移动 ✓ |
| R4-P1 | FiLM 改为 LN 后 per-channel scale+shift（6C 输出）| tca_rwkv.py | ΔF_N=2.9e-2（原 0）✓ |
| R4-P2 | prior_L no-op → 低通门控 | tca_rwkv.py | 非恒等 ✓ |
| R4-P3 | 三路共享 KV = Concat_time 投影 | tca_rwkv.py | kv_proj 有梯度 ✓ |
| R4-P4 | L_temp → 真时序（N: HF一致 / L: LF一致）| loss.py | temp=0.245 激活 ✓ |
| R4-P5 | 三路结构化先验正则（N均值/L TV/M L1）| tca_rwkv.py + loss.py | L_prior=2.4 ✓ |

### 6.1 R4 架构总览

```
输入 (B,5,3,H,W)
    ↓ SharedEncoder
    ├─ F1 (32ch,H)     → 三分支 skip
    ├─ F2 (64ch,H/2)   → TCA 主输入
    └─ F3 (128ch,H/4)  → global pool → f3_ctx → FiLM(6C) 调制三路注意力输出
    ↓
TCA-RWKV:
  Q_N/L/M = query(F2_center)          差异化查询
  KV      = kv_proj(Concat_time(F_seq))  共享全时序 (R4-P3)
  attn_k  = RWKV(Q_k, KV) · (1+γ_k) + β_k   per-channel FiLM (R4-P1)
  raw_k   = center + attn_k · scale_k       LayerScale (零初始化, 单零可解锁)
  F_k     = prior_k(raw_k)                  结构化先验 (R4-P2/P5)
    ↓
Branch-N (去噪) / Branch-L (Retinex) / Branch-M (光流对齐)
    ↓ AdaptiveFusion → O_t
```

### 6.2 启动验证（outputs/golf_r4, ep1）

```
step 50:  loss=1.2134  temp=0.0361  div=-0.1646
step 900: loss=0.6196  temp=0.2914  div=-0.4223
diag: conf=0.507 (中性) / wstd 渐增
GPU: 12.3GB / 65% / 63°C   速度 1.17 it/s
```

**关键信号**：`temp` 从 0.036 增长到 0.291（R3 为恒 0.0078 未激活）→ **真时序损失生效**。

---

## 七、实验设计对比

| 组 | 描述 | 状态 |
|----|------|------|
| Foxtrot | 原始（棋盘格）| ✅ ep70 |
| Golf-R1 | R1 原型（buggy L_div）| ✅ 归档 |
| Golf-R2 | R2（突破 20 dB，双零死锁）| ✅ ep60 |
| Golf-R3 | R3（5 改进，死锁未修）| ⛔ ep6 停止，归档 `golf_r3_deadlock_ep6` |
| **Golf-R4** | 全部修复 | 🔄 训练中 |

### 7.1 验收标准

| 指标 | R2 基线 | R4 目标 |
|------|:---:|:---:|
| val PSNR | 20.14 | ≥ 20.1 |
| pair45 | 16.06（ep40）| ≥ 17.0 |
| TCA 参数量活跃 | 91.6% | **100%** |
| temp 损失激活 | 未激活 | > 0 ✓ |

**核心假设**：R2 的 20.14 是在 8.4% 参数冻结下取得的；R4 解锁后应能进一步释放容量。

---

## 八、R4 训练期损失空间修复（2026-09-21）

### 8.1 问题触发

ep1-4 训练期观察到 **`total_loss` 持续上升而 `final_loss` 持续下降**：
```
total: 1.21 → 6.24 → 8.94   (上升)
final: 0.58 → 0.19          (下降)
```

### 8.2 三项独立缺陷

| # | 缺陷 | 机制 | 修复 |
|---|------|------|------|
| D1 | **L_prior 平方发散**（主因）| `MSE(feat, src)` 随特征量级 k² 增长 → ep4 爆炸至 587 | 改为余弦/归一化 TV/归一化 L1（尺度不变 k⁰）|
| D2 | **L_ortho 卡死鞍点** | `prior` scale=0 + TCA `scale=0` → 三路恒等 → ortho=3.0（理论最大）| `prior` scale 恢复非零初始化（0.1/0.1/0.5）|
| D3 | **L_temp 语义倒挂** | 旧 `\|HF(Y)-HF(x̄)\|` 被亮度差异主导，**惩罚正确提亮**（理想 GT=0.41 > 不处理=0.00）| 改为余弦去相关 `\|cos(HF(Y), HF(x_c-x̄))\|`（亮度不变）|

**验证**：
- 修复前：ep4 `total=6.24, final=0.19, prior=587`
- 修复后：ep1 `total=1.27, final=0.63, prior=1.92` → 300步后 `total=0.71, final=0.31, prior=0.62`，loss 与 final 同步下降 ✓

---

## 九、R4 NaN 崩溃修复（2026-09-23）

### 9.1 NaN 时间线（关键证据）

| Epoch | NaN 次数 | 占比 | 判读 |
|:---:|:---:|:---:|------|
| 1-40 | **0** | 0% | 完全健康（165,000 步）|
| 41 | 181 | 4.4% | 首次爆发（step 1194）|
| 42-43 | 1617-2039 | 39-49% | 密度上升 |
| 44-55 | 3400-3900 | 92-94% | **接近全量失败** |

**总 NaN：49,236 次**。ep40→41 边界突然爆发，**滞后触发非代码 bug**。

### 9.2 根因：AMP fp16 下 Branch-M upsample 数值溢出

**诊断路径**（推翻了两个假设）：
1. ❌ **cosine_similarity 除零**：HF norm 在 28-30 量级健康，**norm 在增长而非趋零**
2. ❌ **训练集特定样本**：fp32 推理全部正常，**AMP fp16 推理直接 NaN**
3. ✓ **AMP fp16 溢出**：Hook 追踪定位到 `branch_m.upsample.conv2`

**数值证据**：
```
upsample.conv1 输出: max=1941 (fp32, GELU 后)
upsample.conv2 输出: NaN (fp16, 128ch × 3×3 = 1152 乘积累加)
```

**触发机制**：
- ep1-40：特征量级小，fp16 安全
- ep40+：训练推进，Branch-M 表示能力增强 → 特征量级增大
- ep41 step 1194：首次触发 `conv1_out > 1500` → NaN
- ep42+：**选择效应**（非权重污染）：NaN batch skip 不更新参数，存活 batch 继续训练推高量级 → 92-94% 全量崩溃

**为何 R2/R3 没有这个问题**：
- R4 的三项损失修复让训练更高效 → 特征表示能力更强 → 量级更大 → 触发 fp16 边界
- R2/R3 可能在到达溢出阈值前就收敛（R2 最佳在 ep40）
- **本质**：R4 训练得"太好"，突破了 fp16 数值限制

### 9.3 修复与验证

**修复**：`models/golf_r4/upsample.py` 强制 fp32 计算
```python
def forward(self, x, skip=None):
    with autocast(enabled=False):
        x = x.float()
        # ... bilinear upsample + conv1 + conv2
```

**验证**：ep55 checkpoint，AMP fp16 推理，8/8 batch 全部正常，0 NaN ✓

**性能代价**：预计 +4% 训练时间，+12.5% 内存，换取数值稳定。

**额外修复**（2026-09-23）：
- `_conv1_out` eval 残留清理：eval 时 `self._conv1_out = None`，避免引用 256² tensor (~33MB)
- GradScaler 与 bf16 兼容：`GradScaler(enabled=use_amp and amp_dtype==torch.float16)`，bf16 时自动禁用
- `tiled_forward` 签名增加 `amp_dtype` 参数，支持 bf16 切换
- conv1_max 监控：每 `log_interval` 步记录，>1500 预警，>1800 建议切 bf16

**重启训练**：从 ep30 (best.pth) 恢复，已运行至 ep31+，conv1_max=73 健康，无 NaN。

---

## 十、关键工程资产

| 文件 | 说明 |
|------|------|
| `models/golf_r4/upsample.py` | resize-conv 上采样（G1）|
| `models/golf_r4/tca_rwkv.py` | TCA-RWKV（R4-P0/P1/P2/P3/P5）|
| `models/golf_r4/branch_n/l/m.py` | 三分支（R3-A/B）|
| `models/golf_r4/fusion.py` | 融合（G4/G5）|
| `models/golf_r4/loss.py` | GolfLoss（R4-P4/P5）|
| `models/golf_r4/golfnet.py` | 主网络（F3 提取 + prior_loss 透传）|
| `train_golf_r4.py` | 训练脚本（pair45 双验证）|
| `configs/golf_r4.yaml` | 配置 |
| `utils/inference.py` | 余弦窗口 tiled_forward（G3，全局）|
| `outputs/golf_r3_deadlock_ep6/` | R3 死锁证据归档 |

## 十、关键工程资产

| 文件 | 说明 |
|------|------|
| `models/golf_r4/upsample.py` | resize-conv 上采样（G1）+ NaN 修复（fp32 强制）|
| `models/golf_r4/tca_rwkv.py` | TCA-RWKV（R4-P0/P1/P2/P3/P5）|
| `models/golf_r4/branch_n/l/m.py` | 三分支（R3-A/B）|
| `models/golf_r4/fusion.py` | 融合（G4/G5）|
| `models/golf_r4/loss.py` | GolfLoss（R4-P4/P5 + 三项缺陷修复）|
| `models/golf_r4/golfnet.py` | 主网络（F3 提取 + prior_loss 透传）|
| `train_golf_r4.py` | 训练脚本（pair45 双验证 + amp_dtype 支持 + conv1_max 监控）|
| `configs/golf_r4.yaml` | 配置（`amp_dtype: fp16`, bf16 备案已注释）|
| `utils/inference.py` | 余弦窗口 tiled_forward（G3，全局）+ `amp_dtype` 参数 |
| `scripts/monitor.sh` | 训练监控脚本（conv1_max 预警 + NaN 统计）|
| `outputs/golf_r3_deadlock_ep6/` | R3 死锁证据归档 |
| `outputs/golf_r4/best.pth` | ep30 健康 checkpoint（重启点）|
| `outputs/golf_r4/latest_ep55_nan.pth` | ep55 NaN 污染归档 |

---

## 十一、教训总结

1. **两个零相乘 = 永久死锁**：LayerScale（门控零初始化）+ 内部投影零初始化 会互相锁死。标准做法是**只保留一个零初始化**（ConvNeXt 保留 LayerScale 零，内部 conv 不零）。本项目的 NAFBlock 正确（单零），TCA-RWKV 错误（双零）。

2. **TBC1B 的侥幸**：若没有非零旁路（channel_mix bias），概念模型也会死。这说明**架构冗余有时会掩盖 bug**。

3. **损失命名 ≠ 损失语义**：`L_temp` 叫"时序"，实现却是"空间 HF 上限"。审查必须核对实现而非名称。

4. **泛化监控必要性**：R2 的 val/pair45 分裂证明单一 val 集不可靠。

5. **审查的价值**：本次用户驱动的审查发现了 **6× 于原审计报告** 的问题（双零死锁根因 + prior_L no-op + K/V 偏离 + L_temp 语义错误）。

6. **损失项必须有量级上界**（新）：任何无界的正则（MSE 平方、无界 L1）在长训练中都会被特征量级放大。设计时先做**尺度不变性测试**（feat×k → loss×?，应为 k⁰）。

7. **零初始化要成对检查**（新）：`proj_out=0` + `scale=0` 是乘法双零死锁；`prior scale=0` + `scale=0` 是分支恒等鞍点。**只保留一个零初始化**。

8. **用真实数据验证**（新）：合成数据（GT=x×3）导致多次误判；只有真实 SDSD 配对数据能暴露亮度主导问题。

9. **AMP 的隐藏风险**（新）：fp16 不是简单的"开关"，而是**数值精度与性能的权衡**。全分辨率 (1080×1920) + 高维特征 (128 通道) + 多层卷积 → fp16 累积误差可能爆炸。**关键路径必须做 fp16 压力测试**（不仅测试 ep1，还要测试 ep40+）。

10. **滞后触发的调试方法**（新）：ep1-40 健康 → 容易误判"代码没问题"。正确方法：对比 fp32 vs fp16 推理、Hook 追踪每个模块、检查激活值量级演化、长训练监控数值健康指标。

11. **"训练太好"的代价**（新）：R4 的三项损失修复让训练更高效 → 特征表示能力更强 → 特征量级更大 → 触发 fp16 数值边界。**数值稳定 > 性能优化 > 语义正确性**（训练阶段）。

---

## 十二、后续路线

### R4 完成后
- 消融：FiLM 开关 / KV 共享 vs 聚合 / 真时序 vs 空间 的独立贡献
- 外部测试：SMID、DRV
- 部署：TensorRT int8

### 若 R4 仍未达标
- 承认 256² 裁剪需要更强归纳偏置，回归 Flight11 T-BC1b 主干
- 或加深分支 block [3,2,3]→[4,3,4] + 扩通道

---

**文档整合完成**：原 7 份 Golf 文档（Golf-plan / R2-analysis / R3-plan / R3-implementation-summary / R3-Ltemp-analysis / R4-loss-audit / R4-NaN-diagnosis）合并为本文件。
