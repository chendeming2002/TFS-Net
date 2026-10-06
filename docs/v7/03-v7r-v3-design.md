# v7r-v3: Triple Query TCA + Matrix RWKV

**日期**: 2026-09-28
**状态**: ✅ 训练完成 (60 epoch)，**超越 Golf R2 目标**
**定位**: 修正 v7r-v2 的「单路 RWKV + 事后解耦」缺陷，回接 Golf-R4 的有效设计

---

## 〇、最终结果 (2026-10-03)

| 指标 | v7r-v3 ep60 | Golf R2 | Golf v7 ep10 | 结论 |
|------|-------------|---------|--------------|------|
| Val PSNR | **20.53** | 20.14 | 19.42 | ✅ +0.39 vs R2 |
| Val SSIM | **0.7615** | — | 0.7525 | ✅ |
| Val LPIPS | **0.4732** | — | 0.4859 | ✅ |

**成功判据达成**:
- ep5 ≥ 19.0 ✅ (19.73)
- ep30 ≥ 20.14 ✅ (20.41)
- ep60 ≥ 21.0 ❌ (20.53，未达激进目标，但稳定超越 R2)

> ⚠️ 口径提示：R2 的 20.14 采用 `tiled_forward(256/32) + max_val_seqs=5 + alex-LPIPS`；
> v7r-v3 的 20.53 采用 `全1080p整帧 + 10序列 + vgg-LPIPS`。二者**非严格同口径**，
> 需以 `eval_checkpoint.py` 统一协议复测 R2 后方可作最终定论 (进行中)。

---

## 一、设计动机

### 1.1 问题溯源

前序 v7r-v2 的多源解耦路径为：

```
PixelTemporal → SpatialSummary(帧级token) → MatrixRWKV ×2
              → ContextDecomposition(线性投影) → FiLM → 三分支
```

**核心缺陷**：v7r-v2 把三源解耦交给**事后线性投影 + 正交约束**，丢失了 R4 已验证的两个关键机制：

1. **三路差异化 Query**（显式退化先验）—— v2 只有单路 MatrixRWKV 输出
2. **共享 KV 中的逐帧/统计细节** —— v2 的 KV 是帧级 token `[B,T,D]`，空间结构被 `AdaptiveAvgPool2d` 抹平

### 1.2 Golf-R4 实验的教训

| 版本 | KV 设计 | Pair45 PSNR | 结论 |
|------|---------|-------------|------|
| R2 | 聚合统计量 | **17.22** | 有效先验 |
| R4 | `Concat_time(F_{t±i})` 1×1 投影 | 15.89 | 全时序噪声放大 🔴 |
| R5 | 回退聚合统计量 | 15.78 | 仍差于 R2 |

**关键结论**（`Golf-R5-plan.md` §R5-3）：
> 全时序可能引入噪声，聚合量是有效先验

即：**R4 的「三路 Query」思路正确，但「Concat 全时序 KV」实现错误**。正确组合应是：

> **三路差异化 Query（继承 R4） + 统计先验 KV（继承 R2/R5） + 逐像素对齐（继承 v7r-v2 PixelTemporal）**

这就是 v7r-v3。

---

## 二、架构总览

```
输入: [B, T=5, 3, H, W]
  │
  ├─ Stage 1: SharedEncoder (逐帧独立)
  │    F2_seq = [B, T, 64, H/2, W/2]
  │
  ├─ Stage 2: feature_proj + PixelTemporalAttentionSimple (window=3)
  │    feat_aligned = [B, 128, H/2, W/2]      ← 逐像素时序对齐
  │
  ├─ Stage 3: Matrix RWKV + Triple Query TCA ⭐
  │    │
  │    ├─ (辅助) SpatialSummary → MatrixRWKVBlock ×2 → ctx [B,T,192]
  │    │
  │    └─ TripleQueryTCA(feat_aligned, feats_proj):
  │         ┌─ 三路查询 (从像素对齐特征) ─┐
  │         │  Q_N = query_N(feat_aligned)  │
  │         │  Q_L = query_L(feat_aligned)  │
  │         │  Q_M = query_M(feat_aligned)  │
  │         └───────────────────────────────┘
  │
  │         ┌─ 共享 KV (统计先验拼接) ──────────┐
  │         │  ctx_mean   = mean(F2_seq)         │  ← 噪声: i.i.d. 均值
  │         │  ctx_smooth = LowPass(ctx_mean)    │  ← 光照: 低频慢变
  │         │  ctx_diff   = max|F_c - F_t|       │  ← 运动: 位移差分
  │         │  KV = kv_proj([mean, smooth, diff])│  ← 3C→C 1×1 投影
  │         └────────────────────────────────────┘
  │
  │         ┌─ 三路 RWKV 空间注意力 (共享 KV) ─┐
  │         │  attn_N = RWKVSpatialHead(Q_N, KV)│
  │         │  attn_L = RWKVSpatialHead(Q_L, KV)│
  │         │  attn_M = RWKVSpatialHead(Q_M, KV)│
  │         └───────────────────────────────────┘
  │
  │         F_k = out_norm_k(feat_aligned + attn_k · scale_k)
  │
  ├─ Stage 4: Branch-N/L/M (复用 v7r-v2 简化分支)
  │
  └─ Stage 5: V7RFusion
      O_t = final
```

---

## 三、模块细节

### 3.1 TripleQueryTCA

位置: `models/golf_v7r/triple_query_tca.py`

| 组件 | 实现 | 参数量 |
|------|------|--------|
| `query_N/L/M` | `LayerNorm2d + Conv2d(1×1)` | 3 × (128×128 + 256) ≈ 49.5K |
| `kv_proj` | `Conv2d(3C→C, 1×1) + LayerNorm2d` | 3×128×128 + 256 ≈ 49.4K |
| `rwkv_inject` | `MatrixRWKVInjector` (Linear 192→3C) | 192×384+384 ≈ 74.1K |
| `attn_N/L/M` | `RWKVSpatialHead` (BiWKV×4方向) | 3 × 66.8K ≈ 200.4K |
| `scale_N/L/M` | LayerScale (零初始化) | 3 × 128 |
| **合计** | | **≈ 0.374M** |

**KV 统计先验**（对齐 R2/R5）：

```python
def _ctx_mean(feats_seq):     # 成像噪声: i.i.d. → 帧间平均最优估计
    return feats_seq.mean(dim=1)

def _ctx_smooth(feats_seq):   # 光照衰减: 低频慢变 → 大核池化低通
    return F.avg_pool2d(feats_seq.mean(dim=1), kernel_size=7, padding=3)

def _ctx_diff(feats_seq):     # 运动位移: 中心帧 vs 邻帧最大绝对差
    diffs = [(center - feats_seq[:,t]).abs() for t != center]
    return stack(diffs).max(dim=0).values
```

### 3.1b MatrixRWKV 注入（方案 B：三路统计门控）

> **背景（审查修正）**：v3 初版存在「死计算」缺陷——`SpatialSummary + 2×MatrixRWKVBlock`
> 算出 `ctx` 却未接入 `TripleQueryTCA`，0.95M 参数每步空转，且使 "Matrix RWKV" 名不副实。
> 本方案 B 将 `ctx` 作为共享统计 KV 的门控，恢复梯度路径。

**注入点**：MatrixRWKV 的时序判断 → 调制统计 KV。二者信息形态互补：

| 路径 | 空间结构 | 时序动力学 | 分辨率 |
|------|----------|-----------|--------|
| 统计 KV (`mean/smooth/diff`) | ✅ 完整 H×W | ❌ 对时间顺序不变 | H/2 |
| MatrixRWKV `ctx` | ⚠️ 仅 2×2 | ✅ T 帧递推 | 帧级 |

因此注入方向是**用时序态势选择退化假设**，而非补充空间细节。

```python
# MatrixRWKVInjector: to_gate 零初始化
gate = to_gate(ctx_center)              # [B, 3C]
gate = gate.view(B, 3, C, 1, 1)
s_mean   = ctx_mean   * (1 + gate[:,0]) # 三路逐通道门控
s_smooth = ctx_smooth * (1 + gate[:,1])
s_diff   = ctx_diff   * (1 + gate[:,2])
kv_shared = kv_proj(concat([s_mean, s_smooth, s_diff]))
```

**恒等性与梯度**：
- `to_gate=0` → `gate=0` → `s_k = ctx_k`，初始不扰动既有表示（恒等）
- 梯度 `∂L/∂W_gate ∝ ∂L/∂kv · ctx_k ≠ 0`（统计量非零）
- 属**单零**（门控层零、被门控量非零），从 step1 起可达 MatrixRWKV，**避开 R4 双零死锁**

**实测验证**（5 步 AdamW lr=4e-4）：

| step | gate_grad | rwkv_block_grad | gate_val | ctx_drift |
|------|-----------|-----------------|----------|-----------|
| 0 | 0 | 0 | 0.000 | 0 |
| 1 | 3.3e-7 | 0 | 0.000 | — |
| 2 | 2.9e-7 | 8.5e-11 | 0.037 | — |
| 5 | 1.5e-7 | 1.6e-10 | 0.096 | 0.277 |
| 59 | 4.9e-8 | 2.0e-10 | 0.208 | 1.025 |

- ✅ Injector 自 step1 解锁，`gate_val` 从 0 增至 0.21
- ✅ MatrixRWKV 梯度非零（step2 起），`ctx_drift` 从 0 增至 1.025 → **真正在学习**
- ℹ️ `rwkv_block_grad ≈ 1e-10` 看似极小，但被门控乘子 `(1+g)`（g≈0.1）衰减，与 LayerScale
  初始 ~1e-10 同量级，Adam 可正常处理；关键是**非零**（对比 v3 初版的**精确 0**）

**参数量**：3.67M → **3.75M**（+74.1K）

### 3.2 关键设计选择

#### ✅ 为什么用统计先验 KV 而非 Concat 全时序？

| 方案 | 优点 | 缺点 | 来源 |
|------|------|------|------|
| Concat 全时序 (R4) | 保留逐帧细节 | 噪声放大、tile 边界不连续 | ❌ 实验失败 |
| 聚合统计量 (R2/R5) | 天然去噪、物理对应 | 丢逐帧细节 | ✅ 实验有效 |
| **统计先验拼接 (v3)** | **保留三源物理语义 + 空间结构** | 仍丢部分逐帧细节 | **本设计** |

**关键**：v3 的统计先验**保留空间结构**（`[B,C,H,W]`），而 v2 的帧级 token 是 `[B,D]`。这使 KV 具有空间选择性 —— 运动区域和静态区域可使用不同的 KV 响应。

#### ✅ 为什么 LayerScale 零初始化？

对称性：三路 `scale_N/L/M = 0` 时，`F_k = out_norm(feat_aligned)`，三路初始完全相同。这看似"坍塌风险"，但：

1. **out_norm 逐通道仿射不同** → 三路归一化参数独立学习
2. **梯度非零**：实测 scale 梯度 `1e-3` 量级（见 §五 验证）
3. **对比 R4 双零死锁**：R4 的 `proj_out=0` + `scale=0` 形成乘法双零链，梯度精确为 0；v3 的 `proj_out` 非零（Xavier 初始化），只有 scale=0 是单零，梯度可达 ✅

---

## 四、与相关版本对比

| 维度 | Golf R2 | Golf R4 | v7r-v2 | **v7r-v3** |
|------|---------|---------|--------|------------|
| 逐像素对齐 | ❌ | ❌ | ✅ PixelTemporal | ✅ PixelTemporal |
| 三路 Query | ✅ | ✅ | ❌ | ✅ |
| KV 来源 | 聚合统计 | Concat 全时序 | 帧级 token | **统计先验拼接** |
| KV 空间结构 | ✅ | ✅ | ❌ | ✅ |
| RWKV 状态 | 向量 | 向量 | 矩阵 (RWKV-6) | 矩阵 (RWKV-6) |
| RWKV 位置 | 空间注意力内 | 空间注意力内 | 帧级（辅助） | **帧级（门控 KV）** |
| MatrixRWKV 梯度路径 | — | — | ✅ FiLM 注入 | ✅ 三路门控注入 |
| 解耦时机 | 查询前 | 查询前 | 查询后 | **查询前** |
| 参数量 | 3.50M | 3.69M | 3.46M | **3.75M** |

---

## 五、验证记录

### 5.0 全量评估 (eval_checkpoint.py, 全 1080p / 10 序列 / VGG-LPIPS)

2026-10-03 用独立脚本 `eval_checkpoint.py` 复测 `best.pth` (ep60)，并**首次以同一协议复测
Golf R2**（此前 20.14 出自训练脚本的 tiled+5序列+alex 口径，不可直接比）：

| 模型 | ckpt | 验证协议 | PSNR | SSIM | LPIPS | 参数量 |
|------|------|---------|------|------|-------|--------|
| Golf v7 | ep10 | 全1080p/10seq/VGG | 19.4195 | 0.7525 | 0.4859 | 1.04M |
| **Golf v7r-v3** | **ep60** | 全1080p/10seq/VGG | **20.5341** | 0.7615 | **0.4732** | 3.75M |
| Golf R2 | ep60 | 全1080p/10seq/VGG | 20.2041 | **0.7760** | 0.4863 | 3.50M |

**结论**：同口径下 v7r-v3 比 R2 **PSNR +0.33 dB**、LPIPS 更优，但 **SSIM 低于 R2**
（0.7615 vs 0.7760）——即 v7r-v3 保真度更高、R2 结构相似度更高，二者存在保真-结构权衡。
逐序列差异集中在 offset 大的序列（pair19 +1.63）；pair45 两模型都极低（R2 11.88 / v3 10.78）。

> ⚠️ 上表所有数字仍受下方 **5.4 数据配对缺陷** 影响，需在修正后重测。

### 5.4 关键数据缺陷：SDSD LQ/GT 配对错位 (2026-10-03)

**发现**：SDSD 每个序列的 LQ 与 GT **帧数相等，但文件名区间整体偏移**，且偏移量随序列不同：

| 序列 | LQ 区间 | GT 区间 | 文件名偏移 |
|------|---------|---------|-----------|
| pair19 | 0156–0277 | 0161–0282 | +5 |
| pair40 | 0036–0174 | 0050–0188 | +14 |
| **pair45** | 0047–0177 | 0107–0237 | **+60** |
| pair60 | 0032–0144 | 0054–0166 | +22 |

训练集同样如此（70 序列中 58 个 |偏移|>3）。

**缺陷**：`datasets/sdsd_dataset.py` 采用「LQ/GT **文件名交集**」配对，例如 pair45
把 LQ `0047`(第0帧) 与 GT `0047`... 实际只保留交集 71 帧，并把 LQ 第 60 帧与 GT 第 0 帧
配成一对——**时间错位 60 帧**。

**证据**：
1. 训练日志总步数 = **8253**，恰等于文件名交集帧数（总帧数 9259），证实训练用了错位子集；
2. 亮度归一化 LQ-vs-GT 结构相关：10 个 val 序列**全部** position 配对优于 name 配对，
   平均 **+1.71 dB** (pair40 +1.71, pair60 +3.31, pair55 +3.17)；
3. 用 v7r-v3 模型自身输出对比：position 配对平均比 name 配对 **+0.78 dB** PSNR；
4. 官方 SDSD 参考实现 `reference_repos/LLVE_STCD`（指向同一 `indoor_np` 数据）
   明确按 **sorted 位置** 配对（`img_paths_LQ[0:30]` vs `img_paths_GT[0:30]`，断言帧数相等），
   而非文件名；
5. 视觉核验 pair45：LQ pos60 与 GT pos60 的同一暗色人形/植物一致，而 GT 同名(读作 pos0)
   对应另一场景（白色头盔）。

**修复**：`SDSDDataset` 新增 `pairing` 参数（`"name"` 保留历史行为 / `"position"` 官方约定），
训练/评估/推理脚本均可切换。position 配对后 train=9259 / val=1244 帧（此前 8253 / 1120）。

**验证完成（2026-10-04）**：`configs/golf_v7r_v3_pospair_quick.yaml` 以 position 配对从头训练 5 epoch
（LR 调度 T_max=60 与原运行一致），用于在同等前 5 epoch 下直接对照原始 name-paired 结果。

**受控对照 A（同一 checkpoint，只换评估配对）** —— 证明「评估端」错位效应：

| checkpoint | 评估配对 | n | PSNR | SSIM | LPIPS |
|-----------|---------|---|------|------|-------|
| v7r-v3 **ep5** (name 训练) | name | 1120 | 19.71 | 0.7524 | 0.5012 |
| v7r-v3 **ep5** (name 训练) | **position** | 1244 | **21.81** | **0.7861** | **0.4686** |

**仅换评估配对即 +2.10 dB / +0.034 SSIM / −0.033 LPIPS**，且 per-sequence 极端值被修正：
pair45 10.78→15.15、pair55 16.94→22.56、pair60 21.02→25.34、pair50 22.06→24.35。
这**直接证明**此前所有 PSNR 都被配对错位系统性压低；且错位越重的序列受损越大，
是 pair45 长期垫底的主因。

**受控对照 B（ep5 同期，name 训练 vs position 训练，统一 position 评估）** —— 证明「训练端」错位效应：

| ep5 checkpoint | 训练配对 | 评估配对 | PSNR | SSIM | LPIPS |
|---------------|---------|---------|------|------|-------|
| v7r-v3 ep5 | name | position | 21.81 | 0.7861 | 0.4686 |
| v7r-v3 ep5 | **position** | position | **21.92** | **0.8145** | **0.3904** |

position 训练使 **SSIM +0.028、LPIPS −0.078**（PSNR +0.11 基本持平），
说明正确配对训练主要改善**结构保真**而非逐像素亮度。训练 loss 也系统性更低
（ep1–ep5 avg loss: position 0.0918/0.0870/0.0807/0.0816/0.0823 vs name 0.1343/0.1283/0.1285/0.1293/0.1327）。

**受控对照 C（ep60 名训最优 vs ep5，统一 position 评估）** —— 揭示错位训练的「负迁移」：

| checkpoint | 训练配对 | 评估配对 | PSNR | SSIM | LPIPS |
|-----------|---------|---------|------|------|-------|
| v7r-v3 ep5 | name | position | 21.81 | 0.7861 | 0.4686 |
| v7r-v3 **ep60** (原 best) | name | position | 21.52 | 0.7743 | 0.4541 |

在统一正确协议下，**原 ep60「最优」反而比 ep5 低 0.29 dB** —— 即此前 60 epoch 训练
在错位目标上做的后期优化是**有害的**（模型在最大化一个错误配对下的指标，而非真实保真度）。

**统一 position 协议下的最终排序（1244 帧）**：

| 方法 | 训练配对 | PSNR | SSIM | LPIPS |
|------|---------|------|------|-------|
| **v7r-v3 ep5 (position 训练)** | position | **21.92** | **0.8145** | **0.3904** |
| v7r-v3 ep5 (name 训练) | name | 21.81 | 0.7861 | 0.4686 |
| v7r-v3 ep60 (name 训练, 原 best) | name | 21.52 | 0.7743 | 0.4541 |
| R2 ep60 | name | 21.33 | 0.7850 | 0.4707 |

> **结论**：仅 5 epoch 的 position-paired 训练已同时超过原 60-epoch 模型与 R2（PSNR/SSIM/LPIPS 三优）。
> 必须**用 position 配对全量重训 60 epoch**才能得到公平的最终结论；此前所有跨版本比较
> （v7 vs v7r vs v7r-v3 vs R2）均建立在错位协议上，需作废并重做。

### 5.1 前向传播

```
Input:       [1, 5, 3, 64, 64]
Output:      [1, 3, 64, 64]
branch_N/L/M: [1, 3, 64, 64]
ortho_loss:  1.0 (初始, 三路未解耦)
```

### 5.2 参数量分解

```
encoder             0.4095M
pixel_temporal      0.0243M
spatial_summary     0.0989M
matrix_rwkv         0.8529M
triple_query_tca    0.3740M   ← 含 rwkv_inject 0.0741M
branch_N            0.5564M
branch_L            0.6093M
branch_M            0.8043M
fusion              0.0129M
─────────────────────────────
total               3.75M
```

### 5.3 梯度流验证

5 步 AdamW (lr=4e-4) 训练后：

| 参数 | step0 | step4 | 状态 |
|------|-------|-------|------|
| `scale_N` | 1.63e-3 | 2.01e-3 | ✅ 非零 |
| `scale_L` | 2.52e-13 | 1.66e-3 | ✅ 解锁（step1 后） |
| `scale_M` | 1.81e-3 | 4.00e-3 | ✅ 非零 |

**注意**：`scale_L` 在 step0 梯度极低（`2.5e-13`），这是因为初始时三路输出完全相同、正交约束对 L 的梯度经对称性抵消；step1 优化后对称性打破，梯度恢复正常。**不是双零死锁**（双零死锁梯度永久为 0）。

---

## 五甲、MatrixRWKV 注入方案矩阵（实验计划）

审查发现 v3 初版的 MatrixRWKV 是「死计算」。本版实现**方案 B**（三路统计门控），
并保留 **方案 C** 作为后续消融，构成一条完整的注入强度梯度：

| 变体 | 机制 | 注入形式 | 参数 | 状态 |
|------|------|----------|------|------|
| **B**（当前） | 三路统计门控 | `s_k = ctx_k · (1 + g_k)`，`g = to_gate(ctx_center)` | +74.1K | ✅ **已实现** |
| **B+C** | 门控 + per-channel FiLM | `kv = kv_proj(s) ; kv = kv·(1+γ) + β` | +123.5K | 📋 计划 |

**方案 C 定义**（`to_film: Linear(rwkv_dim → 2C)`，零初始化）：

```python
gamma, beta = to_film(ctx_center).chunk(2, dim=-1)   # [B, C] each
kv_shared = kv_shared * (1 + gamma[..., None, None]) + beta[..., None, None]
```

**对比设计要点**：
- B 与 C 作用点不同（B 在统计量上、C 在投影后 KV 上），可叠加
- 两者均零初始化 → 均从恒等起步，不冲突、不破坏训练稳定性
- 消融时只看 `inject_stat['gate']` 与 `inject_stat['film']` 的幅度演化

**判读标准**（训练后）：
- 若 B 的 `gate` 幅度显著增长且指标优于"无注入" → MatrixRWKV 的时序门控有增益
- 若 B+C 优于 B → per-channel FiLM 精化有额外增益
- 若均收敛到 → 0 → 诚实结论：MatrixRWKV 对该 KV 无增益（考虑改为方案 A 或删除）

---

## 六、待验证问题

1. **正交约束是否有效解耦** —— 需观察训练后 `ortho_loss` 是否下降（v2 降到 0.00014）
2. **统计先验 KV vs Concat KV** —— 若 v3 优于 R4，则验证「三路 Q + 统计 KV」组合假设
3. **MatrixRWKV 门控增益** —— `gate` 幅度是否增长、是否带来指标提升（见 §五甲）
4. **Pair45 泛化** —— R2/R4/R5 均存在的泛化分裂是否缓解
5. **掩码可视化** —— 三路 Query 是否学到不同的退化响应区域

---

## 七、文件清单

| 文件 | 说明 |
|------|------|
| `models/golf_v7r/triple_query_tca.py` | TripleQueryTCA + MatrixRWKVInjector (方案 B) |
| `models/golf_v7r/golfnet_v7r_v3.py` | GolfNet_v7r_v3 主网络 (传入 ctx, 透出 inject_stat) |
| `models/golf_v7r/__init__.py` | 导出更新 |
| `configs/golf_v7r_v3.yaml` | 训练配置 |
| `train_golf_v7r_v3.py` | 训练脚本 (含 gate 诊断日志) |
| `scripts/monitor_golf_v7r_v3.sh` | 监视终端脚本 (含门控诊断) |
| `docs/v7/03-v7r-v3-design.md` | 本文档 |

---

## 八、运行命令

```bash
# 训练
python train_golf_v7r_v3.py --config configs/golf_v7r_v3.yaml

# 单测
python -m models.golf_v7r.triple_query_tca
python -m models.golf_v7r.golfnet_v7r_v3
```

---

**参考**:
- `docs/v7/02-v7-architecture-design.md` — v7 总体设计
- `docs/TSD-Foxtrot/TSD-Foxtrot.md` §3.2 — 三路查询原始定义
- `docs/v6/Golf-R5-plan.md` §R5-3 — R4 回退决策与理由
- `docs/v6/Golf-R5-postmortem.md` — R5 失败分析
