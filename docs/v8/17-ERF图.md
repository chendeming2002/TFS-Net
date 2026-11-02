# ERF 图: 两种 RWKV 块 Spatial Mix 的感受野分解

> 创建于 2026-10-30
> 脚本: `scripts/diag_erf.py`
> 输出: `outputs/erf_random/erf.png` (随机初始化)、`outputs/erf_v86/erf.png` (v8.6 best, epoch 25)
> 背景: Vision-RWKV / Restore-RWKV 论文用 ERF (Luo et al. 2016) 展示块级感受野;
> 本文档用同一方法比较 EncoderRWKVBlock 与 QueryRWKVBlock 的 Q 路 / KV 路。

## 一、方法

取输出的中心位置对输入求梯度, 每个输入位置的梯度幅值构成 ERF 图;
每组随机输入先按总能量归一再平均 (各 16 组, 64×64, C=32)。

两个头的签名都是 `forward(center_feat, context_feat)`, Q 路与 KV 路是独立的
张量参数, 因此不改模型代码即可分解:

- **ERF_Q** = |d(final 中心)/d(center_feat)| — 经 R/G、OmniShift/SPAB、输入残差;
- **ERF_KV** = |d(final 中心)/d(context_feat)| — 经 proj_k/v 与双向 WKV 扫描;
- **ERF_Spatial** = |d(Spatial Mix 输出中心)/d(x)|, x 同送两路 (两偏导之和即全导数)。
  Spatial Mix 输出用 forward pre-hook 截取 (encoder 头: channel_mix 的输入;
  query 头: post_norm 的输入)。

运行环境说明: GPU 被 v8.7-nl 训练占用, 本轮在 CPU 上跑, WKV 用 O(L²) 的
纯 PyTorch 回退实现, 因此尺寸限 64×64。训练结束后可在 GPU 上以 256×256
重跑出论文级图。

## 二、结果

### 2.1 定量 (中心 r≤4 能量占比 E(r≤4), 覆盖 90% 能量的半径 R90)

随机初始化:

| 块 | 路 | E(r≤4) | R90 (px) |
|---|---|---:|---:|
| EncoderRWKVBlock | Q | 100.0% | 0 |
| EncoderRWKVBlock | KV | 1.4% | 30 |
| EncoderRWKVBlock | Spatial | 90.7% | 0 |
| QueryRWKVBlock | Q | 99.7% | 0 |
| QueryRWKVBlock | KV | 1.4% | 37 |
| QueryRWKVBlock | Spatial | 28.2% | 25 |

v8.6 训练权重 (epoch 25):

| 块 | 路 | E(r≤4) | R90 (px) |
|---|---|---:|---:|
| EncoderRWKVBlock | Q | 100.0% | 0 |
| EncoderRWKVBlock | KV | 1.4% | 36 |
| EncoderRWKVBlock | Spatial | 72.3% | 22 |
| QueryRWKVBlock | Q | 86.7% | **4** |
| QueryRWKVBlock | KV | 1.5% | 26 |
| QueryRWKVBlock | Spatial | 66.7% | 34 |

### 2.2 KV 路 ERF 的方向性检验 (v8.6, 64×64)

四方向扫描若表现为"十字加对角"的星形, 四条中心线应集中远超均匀水平的能量。
实测 (均匀分布的期望是每线 1.56%):

| 块 | 中行 | 中列 | 主对角 | 反对角 |
|---|---:|---:|---:|---:|
| EncoderRWKVBlock | 1.7% | 1.6% | 1.5% | 1.6% |
| QueryRWKVBlock | 1.9% | 1.5% | 1.5% | 1.9% |

四线各约 1.6%, 与均匀期望一致 —— **KV 路 ERF 近乎均匀地铺满全图, 不是星形**。

## 三、发现

1. **Q 路是逐点到局部的, 不含长程机制。** encoder 头的 Q (sigmoid(q) 门) 100%
   能量在中心单像素 (proj_q 是 1×1 投影, 且基类的 OmniShift 只注入 KV 输入);
   query 头的 Q 训练后 R90 从 0 变为 4 px —— 这 4 px 正是 SPAB 学到的局部
   邻域 (随机初始化时 SPAB 输出为零, 训练后权重离开零, 见 v8.6 的多步测试)。
   两条 Q 路都没有超出局部的感受野。

2. **KV 路是全局的, 且当前权重下近乎均匀。** 中心 r≤4 只占 1.4–1.5%, R90 达
   26–37 px (64 px 图上约为全图)。方向性检验显示没有十字/星形结构 —— 四方向
   扫描的各向异性被极弱的衰减抹平了。这与 v8.5 的发现一致: 实测各块衰减的
   有效距离大于序列长度 (docs/v8/09 §衰减参数), WKV 在当前训练下趋近"全局
   加权平均"。ERF 图为该结论提供了直接证据。

3. **Spatial Mix 的总 ERF = 恒等脉冲 + 全局底噪。** 输入残差 (v8.6) 在中心
   单像素贡献主要能量, KV 路铺开全图。训练后 encoder 头的中心占比从 90.7%
   降到 72.3%, 说明扫描通路的相对贡献随训练上升; query 头的对应变化方向
   相反 (28.2% → 66.7%), 两者都由各通路在具体工作点的幅度决定, 不宜过度解读。

4. **"传统 R 参数"与"QueryRWKV 的 Q 路"的 ERF 差别不在空间范围。** 两者都是
   逐点/局部; 差别在形式 (sigmoid 非负门 vs 带符号 R + SiLU(G) + SPAB 局部
   上下文) 与局部宽度 (0 vs 4 px)。长程聚合完全由 KV 路承担 —— 这正是
   "Q 管逐点调制、KV 管长程聚合"分工的直接证据, 也说明用 ERF 评价 Q 机制时,
   合格标准就是"局部且不过分稀疏", 而不是"看它是否全局"。

## 四、局限

- CPU 回退实现限 64×64; 更大尺寸可复跑 (GPU 空闲后 `--size 256`)。
- ERF 是工作点量: 乘法结构下梯度依赖输入值, 已用多组随机输入平均;
  真实数据分布下的 ERF 可再加一版 (输入改为数据样本)。
- 若想看到论文式的"星形"ERF, 需人为调大 `spatial_decay` 做扫描对照 ——
  这同时可以定量回答"衰减要多大才能让扫描退化为局部"。本轮未做。

## 五、跨版本对比: v8.6 / v8.8 / v8.8N (2026-11-02)

脚本 `scripts/diag_erf_compare.py`, 图 `outputs/erf_compare/erf_compare.png`
(5 行 × 3 列)。与第二节同口径 (64×64, n=16, 同 seed, CPU 回退 WKV)。
注意: v8.6 / v8.8 用最终 best (ep25), **v8.8N 用的是训练中途的 ep10 best
(25.31, 训练尚未结束)** —— ERF 是工作点量, 中途权重只作方向性参考。

### 5.1 定量

| 块 (版本) | 路径 | E(r≤4) | R90 (px) |
|---|---|---:|---:|
| EncoderRWKVBlock (v8.6) | Q | 100.0% | 0 |
| EncoderRWKVBlock (v8.6) | KV | 1.4% | 36 |
| EncoderRWKVBlock (v8.6) | Spatial | 72.3% | 22 |
| QueryRWKVBlock (v8.6) | Q | 86.7% | 4 |
| QueryRWKVBlock (v8.6) | KV | 1.5% | 26 |
| QueryRWKVBlock (v8.6) | Spatial | 66.7% | 34 |
| EncoderRWKVBlock (v8.8) | Q | 100.0% | 0 |
| EncoderRWKVBlock (v8.8) | KV | 1.5% | 37 |
| EncoderRWKVBlock (v8.8) | Spatial | 61.4% | 34 |
| QueryRWKVBlock (v8.8) | Q | 86.0% | 4 |
| QueryRWKVBlock (v8.8) | KV | 1.5% | 37 |
| QueryRWKVBlock (v8.8) | Spatial | 72.2% | 12 |
| QueryRWKVBlock (v8.8N, ep10) | Q | 85.6% | 4 |
| QueryRWKVBlock (v8.8N, ep10) | KV | 1.7% | 34 |
| QueryRWKVBlock (v8.8N, ep10) | Spatial | 70.0% | 30 |

### 5.2 KV 路方向能量 (均匀期望 1.56%/线)

| 块 | 中行 | 中列 | 主对角 | 反对角 |
|---|---:|---:|---:|---:|
| v8.6 Encoder | 1.7% | 1.6% | 1.6% | 1.7% |
| v8.6 Query | 1.8% | 1.6% | 1.5% | 1.8% |
| v8.8 Encoder | 1.8% | 1.7% | 1.5% | 1.6% |
| v8.8 Query | 1.6% | 1.7% | 1.7% | 1.6% |
| v8.8N Query | 1.8% | 1.8% | 1.6% | 1.8% |

### 5.3 结论

1. **Q 路的局部性跨版本稳定**: 编码器 Q 恒为逐点 (100%, R90=0); 查询 Q 恒为
   R90=4px、约 86% 中心能量 —— v8.8N 把 OmniShift 换成 RLK-TS 后 Q 路的
   ERF 几乎不变 (85.6%/4px 对 86.0%/4px)。这说明查询 Q 路的 4px 局部邻域
   主要来自 SPAB (R 路径的 3×3×3), Shift 算子的更替 (OmniShift ↔ RLK-TS)
   不改变 Q 路的空间范围。
2. **"KV 路全局近均匀"在三个版本里都成立**: 中心 1.4–1.7%, R90 26–37px,
   方向能量全部贴着均匀期望 (1.5–1.8% 对 1.56%)。衰减过弱使 WKV 趋近全局
   平均的问题不是某一版的特例, 而是 Shift 改动未触及的底层性质 —— 要看到
   各向异性的 ERF 仍需调大 `spatial_decay` (见第四节待办)。
3. **Spatial 总 ERF 的中心占比随版本漂移** (72.3→61.4% enc; 66.7→72.2% query),
   反映各工作点上"残差脉冲 vs 全局底噪"的幅度平衡, 不宜跨版本直接比较;
   Q/KV 两路的分解量才是可比的稳定量。

## 六、随机初始化的结构对比: RWKV 三变体 vs VRT TMSA (2026-11-02)

脚本 `scripts/diag_erf_structures.py`, 图 `outputs/erf_structures/erf_structures.png`。
全部随机初始化 (同 seed), 隔离结构先验 —— 没有训练痕迹, 差异只来自数据流拓扑。
VRT 源码归档于 `reference_repos/VRT/` (官方仓库, BSD), 用其 TMSA 块
(窗口 2×8×8, 不 shift, 互注意开) 经适配器接入 (center, context) 双输入:
center 为第 0 帧、context 为第 1 帧。口径说明: RWKV 头的 Q 路只含查询与残差
(K/V 来自另一张量); VRT 的自注意让 center 帧同时充当 K/V, 其 "Q 路" 是
中心帧的总贡献 —— 两者都是"各自输入流对输出的梯度", 但语义不完全等价。

### 6.1 定量 (64×64, n=16)

| 块 | 路径 | E(r≤4) | R90 (px) |
|---|---|---:|---:|
| RWKV Encoder (v8.6+) | Q | 100.0% | 0 |
| RWKV Encoder (v8.6+) | KV | 1.4% | 30 |
| RWKV Query omni (v8.8) | Q | 99.7% | 0 |
| RWKV Query omni (v8.8) | KV | 1.4% | 32 |
| RWKV Query RLK-TS (v8.8N) | Q | 99.9% | 0 |
| RWKV Query RLK-TS (v8.8N) | KV | 1.4% | 31 |
| VRT TMSA (窗口 8×8) | Q | 84.9% | 9 |
| VRT TMSA (窗口 8×8) | KV | 38.1% | 7 |

KV 路方向能量 (均匀期望 1.56%/线): RWKV 三变体全部 1.6–1.7% (均匀);
VRT 为 行 26.2% / 列 25.8% / 主对角 25.6% / **反对角 0.0%**。

### 6.2 VRT 反对角为零的成因 (已验证非伪影)

逐单元核查: VRT 的 KV 路 ERF 严格落在中心点所属的 8×8 窗口内。64×64 网格上
中心 (32,32) 恰为其窗口的左上角 (32 mod 8 = 0), 足迹呈偏移 (0..+7, 0..+7)
的象限分布; 反对角线 (xx+yy=63) 不经过该窗口的任何单元, 因此精确为零。
行/列/主对角的 26% 也是这个象限足迹与三条线的相交结果。**这是窗口划分的
边界效应, 不是注意力本身的各向异性** —— 把探针移到窗口中心会得到对称的
8×8 方块。RWKV 的方向能量度量 (第二节) 不受此影响, 因为它的足迹是全图。

### 6.3 结构对比结论

1. **RWKV 与 VRT 的本质差别是全局支撑 vs 硬局部**: 随机初始化下 RWKV 三变体
   的 KV 路都是全图近均匀 (R90 30–32px, 方向能量贴均匀期望), VRT 严格锁在
   8×8 窗口内 (R90 7px)。RWKV 的"全局"是结构先验 —— 初始的 spatial_decay
   (randn·0.1) 给出的衰减距离已大于序列长度, WKV 生来就接近全局加权平均;
   这与训练后权重的 ERF (第二节) 一致, 说明衰减过弱不是训练学坏的, 是
   初始化就没给局部性。
2. **Shift 算子的更替不改变结构先验**: omni 与 RLK-TS 的 Q/KV 路指标逐项
   相同 (Q: 99.7%/0px 对 99.9%/0px; KV: 1.4%/32px 对 1.4%/31px)。
   随机初始化下 Q 路都是逐点的 —— OmniShift 的卷积分支与 SPAB 的卷积
   (后者零初始化) 在初始时都不贡献空间混合, 输出由恒等/残差通路主导。
3. **四种块的 ERF 可以一眼区分**: RWKV = 中心脉冲 + 全图底噪; VRT = 中心
   脉冲 + 窗口方块。若论文需要一张"结构差异"图, 这张随机初始化版比训练
   权重版更干净 (没有工作点幅度的干扰)。
