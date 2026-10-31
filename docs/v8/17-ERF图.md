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
