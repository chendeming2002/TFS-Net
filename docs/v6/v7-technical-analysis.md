# v7 技术路线分析与设计

**日期**: 2026-09-27  
**背景**: Golf R2/R4/R5 系列失败，需要根本性架构创新

---

## 一、三个核心技术问题深度分析

### Q1: RWKV 能否实现多头 Transformer 能力？

#### 1.1 RWKV 的本质限制

**RWKV (Receptance Weighted Key Value)** 是线性复杂度的序列模型：
```python
# 标准 Attention: O(T²D)
attn = softmax(Q @ K^T / √d) @ V  # T×T 交互矩阵

# RWKV: O(TD)
o_t = Σ_{i=1}^t exp(w_{t-i}) * (k_i ⊙ v_i)  # 递归聚合，无 T×T 矩阵
```

**关键差异**:
| 维度 | Multi-head Attention | RWKV (单头) |
|------|---------------------|-------------|
| **复杂度** | O(T²D) | O(TD) |
| **交互方式** | 全局成对交互 (pairwise) | 线性加权累积 |
| **表达能力** | 二次项 (quadratic) | 线性项 (linear) |
| **位置依赖** | 显式 (通过 QK^T) | 隐式 (通过衰减权重 w) |

#### 1.2 已知的 RWKV 多头化尝试

**方案A: 简单分组多头 (Naive Multi-head RWKV)**
```python
# 将 D 维特征分为 H 个头
for h in range(H):
    d_h = D // H
    o_h = RWKV(x[:, h*d_h:(h+1)*d_h])
o = concat(o_1, ..., o_H)
```
- ✅ 实现简单
- ❌ **头间无通信** — 每个头独立处理，无法捕捉跨头特征交互
- ❌ 仍是线性聚合，无法恢复二次交互能力

**方案B: RetNet / GLA (Gated Linear Attention)**
- **RetNet** (Sun et al., 2023): 引入 retention 机制，多头并行
- **GLA** (Yang et al., 2024): 门控线性注意力，保持 O(TD) 但增强表达
- ✅ 多头间有门控交互
- ⚠️ 仍受限于线性复杂度约束，**无法完全等价于 Transformer**

**方案C: 混合 RWKV-Transformer**
```python
# 低频全局: RWKV (高效)
global_feat = RWKV(x_downsampled)

# 高频局部: Transformer (精确)
local_feat = MultiheadAttention(x, window=W)

# 融合
out = Fusion(global_feat, local_feat)
```
- ✅ 兼顾效率与表达能力
- ✅ 已有成功案例 (e.g., Nyströmformer 的思路)
- ⚠️ 工程复杂度高

#### 1.3 答案：RWKV 本质上无法等价多头 Transformer

**理论限制**:
- **线性模型无法表达二次交互** — RWKV 的 O(TD) 复杂度意味着必须舍弃 T×T 的成对交互
- **多头分组不增加交互阶数** — H 个独立 RWKV 头仍是 H 个线性模型的并行

**实践证据**:
- Golf R2/R4/R5 的单头 TCA-RWKV 在 pair45 泛化上持续失败
- 即使 R5 回退到 R2 架构，泛化差仍无法解决
- **推测**: 低光视频增强需要**全局时序成对比较**（如检测运动一致性），而 RWKV 的线性聚合无法捕捉

**工程折衷**:
1. **短序列 (T≤5)** — 直接用标准 Transformer，O(25D) 可接受
2. **长序列 (T>10)** — 混合架构：RWKV 做粗粒度 + Window Attention 做细粒度
3. **资源受限** — 保留 RWKV 但接受性能上限

**推荐**: v7 应**放弃 RWKV**，短序列视频 (T=5) 用标准 Multi-head Attention。

---

### Q2: 棋盘伪影的根因 — Conv / RWKV / 数据流？

#### 2.1 伪影类型分类

Golf 系列观察到的伪影有两种：

##### 类型A: **Tile 边界伪影** (Tile Boundary Artifacts)
- **表现**: 1080×1920 推理时，每隔 224px (tile stride) 出现 ±5-15 亮度跳变
- **出现时机**: 仅在 tiled inference，训练/验证时无
- **来源**: `utils/inference.py:tiled_forward` 的拼接策略

**根因分析**:
```python
# 当前实现 (Foxtrot/Golf 共同问题)
# overlap 区域简单平均，非重叠区权重=1.0
weight[overlap] = 2.0
weight[non_overlap] = 1.0
output = sum_tiles / weight
```

**问题**:
1. **独立 tile TCA 状态** — 每个 tile 的 TCA 从零初始化，导致边界处时序特征不连续
2. **硬拼接** — 即使用余弦窗口，tile 间的语义鸿沟仍存在 (如一个 tile 认为是静态，相邻 tile 认为是动态)

**R5 的失败尝试**:
- 动态门控 (temporal_gate) 设计用于在 tile 边界回退中心帧
- 但门控未激活 (100% 高权重) → 未解决

**真正需要的解决方案**:
1. **跨 tile TCA 状态传递** — 相邻 tile 共享部分 RWKV 隐状态
2. **全局一致性约束** — 训练时添加 tile 边界平滑损失
3. **后处理修复** — 推理后用轻量 CNN 修复边界 (类似 deblocking filter)

##### 类型B: **棋盘格伪影** (Checkerboard Artifacts)
- **表现**: 2px 周期的细微亮度振荡，自相关第一峰=2px
- **出现时机**: 训练/推理均有，但不总是明显
- **来源**: **PixelShuffle 上采样** (Odena et al., 2016 经典问题)

**根因分析**:
```python
# Golf 的 upsample 模块
Conv2d(C, C*4, kernel=1)  # 1×1 卷积，无感受野
PixelShuffle(upscale=2)    # 重排为 2×H, 2×W
```

**问题**:
- 1×1 卷积生成 4 个子像素位置，**相邻像素间零通信**
- 子像素 (0,0) 和 (0,1) 完全由不同通道决定，无空间连续性约束

**为什么 Golf R2-R5 仍有此问题**:
- Golf 修复了 Foxtrot 的 tile 边界伪影 (通过余弦窗口)
- 但**未修复 PixelShuffle 伪影** — R2-R5 均保留了 `upsample.py` 的 1×1 Conv

**解决方案**:
1. **改用 3×3 Conv + PixelShuffle** (Shi et al., 2016 推荐)
   ```python
   Conv2d(C, C*4, kernel=3, padding=1)  # 3×3 引入邻域信息
   PixelShuffle(2)
   ```
2. **改用转置卷积** (可能引入其他伪影)
3. **改用 Nearest/Bilinear + Conv 细化**

#### 2.2 RWKV 是否引入伪影？

**分析**:
- RWKV 是**时序模块**，不直接处理空间结构
- Golf 的 TCA 在 H/2 × W/2 分辨率工作，输出特征无空间伪影
- **结论**: RWKV 本身**不引入棋盘伪影**

**但 RWKV 可能加剧 tile 边界问题**:
- RWKV 的递归状态在每个 tile 独立初始化 → 边界处时序特征不连续
- 如果改用 Transformer，可以通过 positional encoding 实现全局一致性

#### 2.3 数据流的影响

**Golf 的数据流**:
```
Input (T, H, W) 
  → Encoder: H/2 × W/2 
  → TCA-RWKV: 时序聚合 
  → 三分支 (N/L/M): 各自处理 H/2 × W/2
  → Upsample: PixelShuffle(2) → H × W
  → Fusion → Output
```

**关键观察**:
1. **所有处理在 H/2 分辨率** — 训练时 128×128，推理时 540×960
2. **上采样是最后一步** — 伪影在此引入，后续无修复机会
3. **三分支独立上采样** — 可能引入三倍伪影量

**改进方向**:
- **延迟上采样** — 在 Fusion 后统一上采样 (减少伪影源)
- **多尺度融合** — 保留 H 分辨率特征，用 skip connection

#### 2.4 总结

| 伪影类型 | 根因 | RWKV 相关? | Conv 相关? | 数据流相关? |
|---------|------|-----------|-----------|------------|
| **Tile 边界** | 独立 tile TCA 状态 | ✅ 是 | ❌ 否 | ⚠️ 部分 (拼接策略) |
| **棋盘格** | 1×1 Conv + PixelShuffle | ❌ 否 | ✅ 是 | ⚠️ 部分 (上采样位置) |

**答案**:
- **棋盘伪影**: **Conv (1×1 PixelShuffle) 是主因**，改用 3×3 Conv 可解决
- **Tile 边界伪影**: **RWKV 递归状态不连续** + **数据流拼接策略**共同导致
- **v7 必须修复**: 
  1. Upsample: 1×1 → 3×3 Conv
  2. TCA: RWKV → Transformer (或实现跨 tile 状态传递)
  3. 数据流: 添加全局一致性约束

---

### Q3: 根据长期计划和 R5 教训，v7 设计方案

#### 3.1 长期计划回顾

**原计划** (Golf-R5-plan.md §五.2):
- Flight11: 多尺度时序金字塔 + 显式边界处理
- 目标: 解决 pair45 泛化 + tile 伪影

**R5 失败教训**:
1. 单头 RWKV 表达瓶颈
2. 256² 裁剪训练的泛化鸿沟
3. 保守修改策略失败
4. R2 的"隐式正则"无法复现

#### 3.2 v7 设计约束

**硬约束**:
- GPU: RTX 4090 (24GB)
- 数据: SDSD indoor (256² 裁剪训练)
- 目标: pair45 PSNR ≥ 18.0 dB (超越 R2 的 17.22)
- 推理: 1080×1920 tiled inference，无明显伪影

**软约束**:
- 参数量 ≤ 5M (实时推理考虑)
- 训练时长 ≤ 48 小时

#### 3.3 三条技术路线

---

### 路线1: **Transformer-TCA (保守升级)**

**核心思想**: 最小化架构变动，仅替换 TCA-RWKV → Transformer

#### 架构
```
Input (T=5, H, W)
  → Encoder: Conv → H/2 × W/2, C=128
  → TCA-Transformer: Multi-head Attention (H=8, D=128)
  → 三分支 (N/L/M): 保留 Golf 设计
  → Upsample: 3×3 Conv + PixelShuffle (修复棋盘伪影)
  → Fusion → Output
```

#### 关键改动
| 模块 | Golf R2 | v7-Route1 | 改动理由 |
|------|---------|-----------|---------|
| TCA | RWKV (O(TD)) | Transformer (O(T²D)) | 恢复二次交互 |
| 头数 | 单头 | 8 头 | 增强表达能力 |
| Upsample | 1×1 Conv | **3×3 Conv** | 修复棋盘伪影 |
| 参数量 | 3.50M | ~4.2M | +0.7M (TCA Transformer) |

#### 复杂度分析
- T=5, D=128, H=8
- TCA: O(T²D) = O(25×128) = 3200 ops/pixel (vs RWKV 的 640)
- **5× 计算量**，但 T=5 时仍可接受

#### 优点
- ✅ **最小化风险** — 仅替换 TCA，其他保留
- ✅ 二次交互恢复，理论上应改善泛化
- ✅ 工程实现简单

#### 缺点
- ❌ **未解决 tile 边界问题** — Transformer 仍是局部处理每个 tile
- ❌ 256² 训练泛化鸿沟未解决
- ⚠️ 可能仍过拟合 val (需要更强正则)

#### 风险评估
- **成功概率**: 60% (改善泛化，但可能不足以达到 18.0 dB)
- **失败模式**: 
  1. Transformer 过参数化 → 更严重的 val 过拟合
  2. Tile 边界伪影依然存在
  3. 训练不稳定 (Transformer 需要更careful的学习率)

---

### 路线2: **多尺度时序金字塔 (Flight11)**

**核心思想**: 弃用单一 TCA，改用金字塔结构捕捉不同时空尺度

#### 架构
```
Input (T=5, H, W)
  ↓
Encoder: 多尺度编码
  Level-1: H/2 × W/2 (粗粒度, 长时依赖)
  Level-2: H/4 × W/4 (中等)
  Level-3: H/8 × W/8 (细粒度, 短时依赖)
  ↓
时序金字塔 TCA:
  L3: Window Attention (T=5, window=全局)
  L2: Window Attention (T=5, window=全局)
  L1: Window Attention (T=3, window=局部)  # 减少计算
  ↓
多尺度融合:
  L3 → upsample → L2 → upsample → L1
  ↓
三分支 + Fusion → Output
```

#### 关键设计
1. **金字塔时序建模** — 不同层级用不同时序窗口
2. **渐进式融合** — coarse-to-fine，低层级提供全局先验
3. **自适应时序窗口** — L3 用 T=5，L1 用 T=3 节省计算

#### 参数量估算
- Encoder: ~1.0M
- 金字塔 TCA (3 层): ~2.5M
- 三分支: ~0.8M
- Fusion: ~0.3M
- **Total: ~4.6M**

#### 优点
- ✅ **多尺度捕捉不同运动模式** — 解决 pair45 快速运动场景
- ✅ 渐进式融合减少伪影 (低层提供全局一致性)
- ✅ 架构创新，突破 Golf 单一尺度瓶颈

#### 缺点
- ❌ **工程复杂度高** — 需要重新设计整个架构
- ❌ 训练不稳定风险 (多层级需要careful的损失设计)
- ❌ 调试困难 (多层级难以定位问题)

#### 风险评估
- **成功概率**: 50% (高收益但高风险)
- **失败模式**:
  1. 多尺度融合不稳定 (层级间特征不匹配)
  2. 训练时间过长 (多层级增加计算量)
  3. 过拟合风险更高 (参数量增加)

---

### 路线3: **混合架构 + 全分辨率训练**

**核心思想**: 根本性解决泛化鸿沟 — 直接在 1080×1920 训练

#### 架构
```
Input (T=5, 1080×1920)
  ↓
Encoder: Lightweight CNN (MobileNet-like)
  → H/4 × W/4 (270×480), C=64
  ↓
TCA: Efficient Attention (Linformer / Performer)
  → 近似 O(TD) 但保留部分二次交互
  ↓
Decoder: 渐进式上采样
  H/4 → H/2 (skip from encoder)
  H/2 → H (skip from input)
  ↓
Output
```

#### 关键设计
1. **全分辨率训练** — 消除训练-测试分布差异
2. **高效 Attention** — Linformer (O(TD)) 或 Performer (kernel approximation)
3. **轻量 backbone** — 减少显存占用
4. **跳跃连接** — 保留高频细节

#### 显存估算 (batch=1, T=5, FP16)
- Input: 5×3×1080×1920 × 2 bytes = 62 MB
- Features (H/4): 5×64×270×480 × 2 bytes = 82 MB
- Gradients: ~3× features = 246 MB
- **Total: ~400 MB** (单样本)
- **Batch size = 4 可行** (1.6 GB features + overhead ≈ 4 GB)

#### 优点
- ✅ **根本性解决泛化鸿沟** — 训练=测试分辨率
- ✅ 自然消除 tile 边界伪影 (无 tiled inference)
- ✅ 高效 Attention 保持可训练性

#### 缺点
- ❌ **训练速度慢** — 全分辨率计算量是 256² 的 18× (1080×1920 / 256²)
- ❌ Batch size 受限 (≤4) → 训练不稳定
- ❌ 需要高效 Attention 实现 (Linformer/Performer 工程复杂)

#### 风险评估
- **成功概率**: 40% (理论最优但工程挑战大)
- **失败模式**:
  1. 显存不足 (即使 batch=1)
  2. 训练时间过长 (>7 天)
  3. 小 batch 导致训练不稳定

---

## 二、三条路线对比矩阵

| 维度 | 路线1: Transformer-TCA | 路线2: 多尺度金字塔 | 路线3: 全分辨率混合 |
|------|----------------------|-------------------|-------------------|
| **核心思想** | 最小化改动 | 多尺度架构创新 | 根治泛化鸿沟 |
| **参数量** | 4.2M | 4.6M | 3.8M |
| **计算量** | 5× vs R2 | 3× vs R2 | 18× vs R2 (训练) |
| **工程复杂度** | 低 ⭐ | 高 ⭐⭐⭐ | 中 ⭐⭐ |
| **成功概率** | 60% | 50% | 40% |
| **预期 pair45** | 17.5-18.5 | 18.0-19.0 | 18.5-19.5 |
| **解决 tile 伪影** | ❌ | ⚠️ 部分 | ✅ 完全 |
| **解决棋盘伪影** | ✅ (3×3 Conv) | ✅ (3×3 Conv) | ✅ (3×3 Conv) |
| **训练稳定性** | ⚠️ | ❌ | ⚠️ (小batch) |
| **训练时长** | 30-40h | 40-50h | 60-80h |
| **调试难度** | 低 | 高 | 中 |
| **可解释性** | 高 | 中 | 高 |
| **失败后退路** | 可回退 R2 | 难回退 | 可降分辨率 |

---

## 三、推荐决策树

```
用户需求评估
├─ 优先稳妥 (60天内需要成果)
│   → 路线1: Transformer-TCA
│   → 风险低，收益中等
│
├─ 追求性能突破 (可接受失败风险)
│   → 路线2: 多尺度金字塔
│   → 风险高，收益高
│
└─ 长期研究 (可投入3个月)
    → 路线3: 全分辨率混合
    → 根治问题但工程量大
```

---

## 四、我的推荐: **路线1 (Transformer-TCA) + 路线2 (金字塔) 的渐进式方案**

### 4.1 Why 渐进式？

**经验教训**: Golf R5 的保守修改失败，但 R4 的激进修复也失败 → 需要**可控的创新**

### 4.2 阶段1: v7a - Transformer-TCA (4 周)

**目标**: 验证 Transformer 能否改善泛化

**架构**: Golf R2 + TCA RWKV → Transformer (8 heads) + 3×3 Upsample

**预期结果**:
- 成功: pair45 ≥ 17.5 dB → 进入阶段2
- 失败: pair45 < 17.0 dB → 放弃 Golf 架构，直接跳到路线3

**验证指标**:
- Ep10 pair45 是否超过 R2 (17.22)
- Val-pair45 差距是否缩小 (R2 是 2.92)
- Tile 边界伪影幅度 (目标 <5 亮度跳变)

### 4.3 阶段2: v7b - 添加多尺度 (4 周)

**前提**: 阶段1 成功 (pair45 ≥ 17.5)

**架构**: v7a + 多尺度 encoder (2 层金字塔)

**改动**:
- Encoder 输出 H/2 和 H/4 两个尺度
- H/4 用更大的 TCA 感受野 (或更长时序窗口 T=7)
- 多尺度融合

**预期结果**:
- 成功: pair45 ≥ 18.5 dB → 项目成功
- 失败: 17.5 ≤ pair45 < 18.5 → v7a 已足够，优化超参数

### 4.4 阶段3 (可选): 全分辨率微调

**前提**: v7b 成功，且有充足时间

**策略**: 在 v7b 基础上，用 512² 或 768² 裁剪微调 10-20 epoch

**目标**: 进一步缩小训练-测试分辨率差距

---

## 五、下一步行动

### 立即需要用户决策:

**问题1**: 选择路线？
- A. 渐进式 (推荐): v7a (Transformer) → v7b (多尺度)
- B. 激进式: 直接路线2 (多尺度金字塔)
- C. 根治式: 路线3 (全分辨率)

**问题2**: 是否保留三分支？
- Golf 的 Branch-N/L/M 三分支在 R2-R5 均存在
- 但可能是冗余设计 (R4 证明三分支功能重叠)
- 选项:
  - A. 保留 (降低风险)
  - B. 简化为单分支 (减少参数，提升泛化)

**问题3**: TCA 的时序窗口？
- Golf 用 T=5
- 选项:
  - A. 保持 T=5 (与 R2 对比)
  - B. 扩展 T=7 (更长时序，但计算量 +96%)

### 实施计划 (若选择渐进式路线1):

#### Week 1-2: v7a 实现
- [ ] 实现 TCA-Transformer (8 heads, D=128)
- [ ] 修改 Upsample: 1×1 → 3×3 Conv
- [ ] 从 R2 初始化，仅随机初始化 TCA Transformer
- [ ] Smoke test: forward + loss

#### Week 3-4: v7a 训练
- [ ] 训练 60 epoch (预计 40h)
- [ ] 监控: val/pair45/tgate (如果保留)/tile伪影
- [ ] Ep10 验证: 若 pair45 < 17.0 → 提前终止

#### Week 5-6 (若 v7a 成功): v7b 设计
- [ ] 添加多尺度 encoder
- [ ] 设计多尺度融合策略
- [ ] 重新训练

---

**等待用户决策**: 请选择技术路线 + 回答问题2/3，然后开始实现。
