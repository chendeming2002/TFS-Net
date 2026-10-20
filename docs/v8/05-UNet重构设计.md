# v8 U-Net 式重构: 把 SpatialSummary + TripleQueryTCA 重写成编码器-解码器结构

> 日期: 2026-10-17
> 前置: `01-设计.md` (归因结论)、`04-训练加速调研.md` (WKV 修正)
> 实验: `configs/v8/unet_refactor_15ep.yaml`, 10–15 epoch

## 一、要解决的问题

原结构的四个阶段里, 有三段在做同一件事 —— "决定多帧该怎么汇总":

| 阶段 | 做了什么 | 产物去哪了 |
|------|---------|-----------|
| `SpatialSummary` | 把 F2 池化成 4×4 → 帧级 token [B,T,D] | 只喂给下一段 |
| `MatrixRWKVBlock` ×2 | 在帧 token 上做时序建模 | 只用来给 KV 做门控 |
| `PixelTemporalAttentionSimple` | 在 F2 上做逐像素的邻帧对齐 | 作为 `TripleQueryTCA` 的输入 |
| `TripleQueryTCA` | 重新构建统计 KV (均值/低频/差分) | **三段里唯一真正进入分支的** |

问题在于: 前三段的产物要么被池化抹掉了空间信息, 要么只用来调制一个门控,
而真正决定三路分工的 `TripleQueryTCA` 又工作在 **F2 尺度**上。

这带来两个具体后果:

1. **信息流失**: 所有细节要先被下采样到 H/2 再用上采样还原。而 M 支路要建模的
   "局部结构偏移"恰恰在最高频带上 —— 它在 F2 上就已经不完整了。
2. **三个分支的输入完全相同**: 三路 Query 都从同一张 `feat_spatial` 生成,
   共享同一个 `kv_shared`。分工只能靠 Query 权重与注意力头参数的差异去"逼"。
   前几轮实验反复证明这条路走不通 (见 `01-设计.md` §六·补·续)。

## 二、新结构

### 2.1 总体数据流

```
x [B,T,3,H,W]
 ├─ FrameEncoder (逐帧, 每尺度 1 个 RWKV block, block 内循环 M 次)
 │     → f1_seq [B,T,C1,H,W]   f2_seq [B,T,C2,H/2,W/2]   f3_seq [B,T,C3,H/4,W/4]
 │
 ├─ TemporalAggregator  (自上而下聚合, F3 → F2 → F1)
 │     · 逐帧 RWKV block 处理 F3 → 每帧描述子
 │     · 帧级 RWKV (循环 M 次) → 每帧权重 (softmax)
 │     · 按权重聚合 F3 → F3_agg
 │     · F3_agg ↑ F2_center → 融合 → F2_agg ↑ F1_center → 融合 → F1_multi
 │     · 旁路: F1_single = f1_center,  F2_single = f2_center
 │
 ├─ TaskQueryAttention (F1 尺度)
 │     Q_N, Q_L ← F1_multi        KV_N ← 低通(F1_multi)        KV_L ← 更强低通(F1_multi)
 │     Q_M     ← F1_single       KV_M ← 高频残差(F1_single)
 │
 ├─ BranchN / BranchL / BranchM (F1 直出, 不再上采样)
 └─ V7RFusion (additive)
```

### 2.2 三处关键变化

**变化一: 单模块循环 M 次, 而不是堆 M 个 block。**

Re-WKV (Restore-RWKV) 的做法是把同一个模块串行循环 M 次: 参数不变, 计算 ×M。
本项目的 `RWKVEncoderBlock` 早就支持 `num_loops`, 但之前的编码器是"每尺度堆 2 个
独立 block"(参数与计算都 ×2), 二者不是一回事。本次统一为循环:

- `encoder_loops=2` → 每尺度 1 个 block, 内部循环 2 次;
- `temporal_loops=2` → U-Net 里的帧级 RWKV 也是单模块循环 2 次。

**实现中踩过的坑**: 最初把 `encoder_loops` 传给 `num_blocks_per_scale`, 结果参数量
从 1.10M 涨到 1.63M —— 那又变回"堆 block"了。单测 `test_loops_change_compute_not_params`
专门断言 M 不改变参数量, 防止再错。

**变化二: SpatialSummary 变成自上而下的聚合链。**

旧的 `SpatialSummary` 把 F2 池化成 4×4 的帧 token, 空间信息几乎全丢。
新实现直接在空间特征上做逐帧 RWKV, 再用帧级 RWKV 给出每帧权重, 按权重聚合:

```
weights = softmax( scorer( RWKV_loop_M(frame_tokens) ) )
F3_agg  = Σ_t weights_t · f3_t
```

权重是 softmax 归一化的, 因此聚合结果是各帧的**凸组合** —— 不会像直接相加那样放大噪声,
这与"时序平均抑制 i.i.d. 噪声"的物理直觉一致。实测初始权重接近均匀 (每帧 0.20)。

**变化三: TripleQueryTCA 移到 F1, 且按支路来源分流。**

| 支路 | Query 来源 | KV 来源 | 残差基 |
|------|-----------|---------|--------|
| N (噪声) | 多帧聚合 F1_multi | 低通(F1_multi) | F1_multi |
| L (光照) | 多帧聚合 F1_multi | 更强低通(F1_multi) | F1_multi |
| M (运动) | **单帧 F1_single** | **高频残差(F1_single)** | **F1_single** |

单帧高频残差 = `F1_single − avg_pool(F1_single)`。取高频是因为 M 要建模的
"局部结构偏移"表现为边缘与纹理的变化, 低频光照变化不属于它。

这样分工发生在**数据流**上: M 的整条路径 (Query、KV、残差基) 都不含跨帧信息,
而 N、L 的整条路径都来自多帧聚合。不需要损失函数去"逼"三路变得不同。

### 2.3 删除的冗余结构

| 组件 | 为什么删 |
|------|---------|
| `SpatialSummary` (池化 token) | 被逐帧 RWKV + 时序加权取代 |
| `PixelTemporalAttentionSimple` | 被 U-Net 的加权聚合取代; 它在 F2 上的邻帧对齐在 SDSD 上本就作用有限 |
| `feature_proj` | 只为把 F2 投到 128 维给 TCA 用, 新结构在 F1 直接工作 |
| `MatrixRWKVInjector` | 它给 KV 加的逐通道门控, 已被 U-Net 的帧间权重与支路分流取代 |

保留: `V7RFusion`、三个 `Branch*Simple` (通过 `at_f1=True` 切全分辨率直出)。

## 三、代价与收益

| 项 | v7r-v3 | v8 | 说明 |
|----|:--:|:--:|------|
| 总参数 | 4.19M | **1.10M** | 删掉冗余结构 |
| 编码器参数 | 0.63M (RWKV block 时) | 0.36M | 单 block 循环, 不堆叠 |
| 前向 (b1, 256) | 149 ms | 149 ms | 序列长度从 16384 增到 65536, 但 CUDA 算子抵消了增量 |
| 单步时间 | 0.49 s | 约 0.55 s | 见训练日志 |

参数减少 74%, 而前向耗时不变 —— 省下的是"计算但不产出"的部分。

## 四、必须说明的两处限制

**限制一: 训练时的序列长度是硬上限。**

CUDA WKV 的**反向**内核在栈上开了三个 `IDEAL_T_LEN` 数组 (`Tmax/32 = 2048` 个 float),
因此训练时 L 不能超过 65536。crop=256 对应 L=65536, 恰好卡在上限;
若要用 384 或 512 的裁剪训练, 必须把 `Tmax` 调大 —— 但调大会让每个线程的栈数组变大,
**实测 2097152 时反向直接报 "invalid argument"**。前向内核按 token 分块, 没有这个限制,
这也是整图评估 (L=207万) 能跑通的原因。

**限制二: 融合用 additive, 不是 gated。**

合并实验 (`outputs/v8_combined_B1M1M2M3A`) 已经证明 gated 模式存在结构性自锁:
门 `g` 降到 0.000, 而 `∂O/∂Y_M = g`, 门一关 M 路就再也拿不到梯度。
本次的目标是验证新聚合结构本身, 因此用 additive (权重恒正、导数恒 ≥ ∂O/∂Y_M = w_M > 0),
保证三路都有梯度。门控是否值得保留, 应在结构稳定之后单独验证。

## 五、判据

| # | 判据 | 达标线 |
|:--:|------|--------|
| 1 | 三路输出的空间方差 | 都明显高于常数图水平 (约 1e-4 量级; 常数图约 1e-9) |
| 2 | 三路与各自频带目标的误差 | 随训练下降 |
| 3 | 消融: 去掉任一路 | 有明显代价 (additive 口径: 直接置零, 不归一化) |
| 4 | 帧权重 | 不塌成单帧 (任一路权重接近 1) |

**判据 1 是主判据。** 前几轮的教训是"输出余弦低"与"消融掉指标"都能被常数图伪造,
只有空间方差能直接区分"这一路在输出内容"还是"在输出常数"。

## 六、实施产物

| 文件 | 内容 |
|------|------|
| `models/golf_v7r/spatial_summary_unet.py` | `TemporalAggregator` + `TemporalWeighting` |
| `models/golf_v7r/triple_query_tca_v9.py` | `TaskQueryAttention` (F1 尺度, 按支路分流) |
| `models/golf_v7r/golfnet_v8.py` | `GolfNet_v8` 主网络 |
| `models/golf/encoder.py` | 新增 `block_loops` (单模块循环) |
| `models/golf_v7r/branch_{n,l,m}_simple.py` | 新增 `at_f1` (全分辨率直出) |
| `train_golf_v7r_v3.py` | 新增 `model.arch = "v8"` 选择器 |
| `configs/v8/unet_refactor_15ep.yaml` | 实验配置 |
| `tests/test_v8_unet.py` | 7 项单测 |
| `scripts/diag_branch_health.py` | 判据 1 的测量脚本 (256 裁剪 + 空间方差) |
| `scripts/diag_v7r_v3_branch_ablation.py` | 判据 3 的测量脚本 (按配置 `arch` 选择模型类) |

## 七、判据测量工具与口径验证

判据 1 用 `scripts/diag_branch_health.py` 测量: 取每段序列中心 5 帧, 在图像中心裁
256×256 的窗口送入模型, 对 `branch_N/L/M` 各自逐通道求空间方差再平均。裁剪的原因
是全分辨率会把大面积的均匀暗区算进方差, 稀释局部结构, 使常数图与有结构图的差距
被压低。

方差判读分三档: 低于 1e-7 判为常数图, 1e-7 至 1e-5 判为弱结构, 高于 1e-5 判为有结构。

脚本先在上一轮 10ep 合并实验的 checkpoint 上验证过口径, 结果与当时的结论一致:

```
空间方差   N=2.105e-09  L=1.408e-02  M=7.854e-10  Final=9.695e-02
单独 PSNR  N=5.08  L=13.46  M=5.08 dB
最终 PSNR=14.69 dB   GT mean=0.410 std=0.179
门 g mean=0.000
```

N 与 M 的方差比常数图判据低两个数量级以上, L 的方差高出七个数量级, 与此前
"N、M 塌成常数图、L 正常"的归因一致。注意这里的 PSNR 是 256 裁剪口径, 与全图
评估的数字不可直接比较, 只用于同一实验内部的横向对照。

## 八、明确不做的事

| 项 | 原因 |
|----|------|
| 把 gated 融合一并启用 | 已知自锁; 先单独验证结构, 避免两个问题混在一起 |
| 同时调 Tmax 与 crop_size | 反向内核有栈数组约束, 需要单独评估显存与稳定性 |
| 在本次实验里做单变量拆分 | 用户的要求是先合并实施看整体, 拆分放到之后 |
