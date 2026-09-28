# v7r-v3: Triple Query TCA + Matrix RWKV

**日期**: 2026-09-28
**状态**: 实现完成，待训练验证
**定位**: 修正 v7r-v2 的「单路 RWKV + 事后解耦」缺陷，回接 Golf-R4 的有效设计

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
| `attn_N/L/M` | `RWKVSpatialHead` (BiWKV×4方向) | 3 × 66.8K ≈ 200.4K |
| `scale_N/L/M` | LayerScale (零初始化) | 3 × 128 |
| **合计** | | **≈ 0.30M** |

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
| RWKV 位置 | 空间注意力内 | 空间注意力内 | 帧级（辅助） | 帧级（辅助） |
| 解耦时机 | 查询前 | 查询前 | 查询后 | **查询前** |
| 参数量 | 3.50M | 3.69M | 3.46M | **3.67M** |

---

## 五、验证记录

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
triple_query_tca    0.2995M   ← 核心新增
branch_N            0.5564M
branch_L            0.6093M
branch_M            0.8043M
fusion              0.0129M
─────────────────────────────
total               3.67M
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

## 六、待验证问题

1. **正交约束是否有效解耦** —— 需观察训练后 `ortho_loss` 是否下降（v2 降到 0.00014）
2. **统计先验 KV vs Concat KV** —— 若 v3 优于 R4，则验证「三路 Q + 统计 KV」组合假设
3. **Pair45 泛化** —— R2/R4/R5 均存在的泛化分裂是否缓解
4. **掩码可视化** —— 三路 Query 是否学到不同的退化响应区域

---

## 七、文件清单

| 文件 | 说明 |
|------|------|
| `models/golf_v7r/triple_query_tca.py` | TripleQueryTCA 核心模块 |
| `models/golf_v7r/golfnet_v7r_v3.py` | GolfNet_v7r_v3 主网络 |
| `models/golf_v7r/__init__.py` | 导出更新 |
| `configs/golf_v7r_v3.yaml` | 训练配置 |
| `train_golf_v7r_v3.py` | 训练脚本 |
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
