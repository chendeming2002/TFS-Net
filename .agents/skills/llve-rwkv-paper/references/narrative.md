# 核心叙事（TSDR / TSD-Net）

> 论文：TSD-Net: Triple Source Decoupled Restoration for Low-Light Video Enhancement
> 当前草稿：v7（`docs/paper/写作-论文/TSD-Net论文-IntroDraftv7.md`），按 v8.5 实现（TemporalAggregator_G + SGRFv10 + kvmod）
> 命名唯一口径：`docs/v8/08-命名对照.md`

## 一、TSDR 三源分解框架

**观测模型**（sRGB 域，Eq. 1–3）：

$$I_t(p) = L_t(p)\cdot X_t(p) + n_t(p),\quad n_t(p)\sim\mathcal{N}\big(0,\ \sigma_s^2 L_t(p)X_t(p)+\sigma_r^2\big)$$

- 物理依据：Wei et al. [1]（ELD）、Foi et al. [12]、Feng et al. [13]（PNNP）。**注意**：三篇均为 RAW 域模型，sRGB 域只是一阶近似，措辞见 `papers-noise.md`。
- 运动：$I_{t'}(p)\approx L_{t'}(p)X_t(p+\mathbf d_{t\to t'}(p))+n_{t'}(p)$。
- 概念分解（非精确等式）：$I_t=\mathcal D(\mathcal I(X_t),\mathcal N_t,\mathcal M_t)$。

**两维度分类**（跨帧持续性 × 空间全局/局部）→ 四类型 → 三源：

| 源 | 组成 | 时序行为 | 空间行为 | 对应操作 |
|---|---|---|---|---|
| 成像噪声 | read noise（全局、帧独立）+ shot noise（局部、帧独立、强度随信号） | 帧独立、零均值 | 全局+局部混合 | 时域平均 |
| 光照 | 衰减 $L_t$ | 短窗内近似恒定 | 全局、缓变 | 平滑全局估计 |
| 运动 | 位移 $\mathbf d$ | 持续、结构化 | 局部 | 单帧读取（不可平均） |

**为什么两源不够**（§3.1 论证核心）：噪声+运动同池 → 平均错位帧产生鬼影；光照+运动同池 → 用仍在运动的内容估计偏置；光照+噪声同池 → 从含噪信号估计增益，校正放大噪声。

**依赖顺序**：光照决定 shot noise 强度，两者都作用在运动位移的内容上 → 修复顺序 **噪声 → 光照 → 运动**（SGRF 逐段组合的依据）。

## 二、五部件架构（v8.5 / v8.8 口径）

```
x [B,T,3,H,W]
 └─ FrameEncoder        逐帧共享权重；3 尺度；每尺度 1 个 EncoderRWKVBlock 循环 2 次
 └─ TemporalAggregator  唯一跨帧部件：F3 再编码 → 每尺度分两流（全帧均值 / 中心帧）→ 共享参数 top-down
                         输出 F1_multi, F1_single, F2_single
 └─ TaskQueryAttention  F1 尺度 3 个 QueryRWKVBlock → F_N, F_L, F_M（+ ortho 损失）
 └─ NBranch / LBranch / MBranch → Y_N, Y_L, Y_M
 └─ SGRF（代码 V7RFusion）逐段门控组合 → O_t
```

**TaskQueryAttention 的三源查询分配**（Table 1）：

| 支路 | 源 | 依据 | Query | Key-value | 残差基 |
|---|---|---|---|---|---|
| NBranch | 噪声 | 零均值、帧独立、内容对齐 | $F^1_{single}-F^1_{multi}$ | $F^1_{multi}$ | $F^1_{multi}$ |
| LBranch | 光照 | $L_t$ 短窗近似恒定 | $F^1_{multi}$ | $F^1_{single}$ | $F^1_{single}$ |
| MBranch | 运动 | 位移不可平均 | $F^1_{single}$ | $F^1_{single}$ | $F^1_{single}$ |

**SGRF 逐段组合**（Eq. 5）：$X_k=X_{k-1}+s_k G_k\odot b_k(\Delta_k)$，$k\in\{N,L,M\}$；$\tilde F_k=F_k+\mathrm{proj}_k(X_{k-1})$（零初始化 1×1 卷积把当前状态映射回特征空间）；门 $G_k=\sigma(\mathrm{gate}_k([X_{k-1},\tilde F_k]))$ 零初始化（起点 0.5）；$b_k=\delta_k\tanh(\Delta_k/\delta_k)$ 软限幅；$s_k=1+(r_k/2)\tanh(a_k)$ 有界幅度。初始化时退化为"半分支输出顺序相加"。

**损失**（Eq. 6–7）：L1 重建 + 分支监督（频带目标 $S_L=\mathrm{LP}_{31}(GT)$，$S_N=\frac12 Q+\frac45 R$，$S_M=\frac12 Q+\frac15 R$，$Q$/$R$ 为中/高频带，按 $\overline{|S_k|}$ 归一化，权重 0.3 线性爬坡 5 轮）+ 特征正交惩罚（λ=0.01）+ 门正则（λ=0.05）。

## 三、RWKV 设计谱系（创新点边界，写 §2.3 与 §3.4 前必读）

| 来源 | 借用的机制 | 注意 |
|---|---|---|
| Vision-RWKV [16] | Bi-WKV 双向一维扫描（衰减 w 为**可学习常量**，非输入相关） | 四方向是 Q-Shift（token 混合）不是扫描；四方向 2D 扫描出自 RSRWKV |
| Restore-RWKV [15] | Re-WKV 同块循环 M 次（深度与参数量解耦）；OmniShift 多分支深度卷积（1×1/3×3/5×5/恒等，测试时重参数化为单个 5×5） | 论文自称 "first RWKV-based model for image restoration"（医学图像） |
| Video RWKV/LCR [14] | Cross RWKV gate：R/G ← 查询路（主干+历史态），K/V ← 另一路（边缘）；次序 R⊙WKV → 逐头 LN → SiLU(G) | 任务是动作识别，非修复 |
| EvRWKV | Cross-RWKV：K 取**对侧流**、V/R 取本路流；CS-Shift；连续跨模态交互 | 与 LCR 的分工不同；同组（Dong Xingbo） |
| **本仓库自有** | kvmod：查询池化描述子预测 K 的 per-channel scale/shift（零初始化）；QRB 三源任务门控；四方向 2D 扫描（来自 RSRWKV，非 Vision-RWKV） | §3.4 的贡献表述核心 |

## 四、贡献声明（v7 三条，措辞已定）

1. TSDR 理论：按时序行为把低光视频退化分解为光照/噪声/运动三源，各配专门支路。
2. TSD-Net：三源专门支路 + 逐段（N→L→M）组合。
3. task-gated query RWKV block：每源经 cross RWKV gate 读 clip 表示，全局空间上下文线性复杂度；"the first RWKV-based model for low-light video enhancement"（范围限定，见红线 2）。

## 五、命名表（论文用名 ↔ 代码）

| 论文名 | 代码符号 | 说明 |
|---|---|---|
| FrameEncoder | `FrameEncoder` | 部件 |
| TemporalAggregator | `TemporalAggregator` | 部件；内部成员 `summary.frame_encoder` 是块 |
| TaskQueryAttention | `TaskQueryAttention` | 部件 |
| NBranch / LBranch / MBranch | 同名 | 部件 |
| SGRF | `V7RFusion` | **论文名与代码名不同**（旧 `SGRF` 类是另一物） |
| QueryRWKVBlock | `CrossRWKVHeadV8` | 功能名；`q_mode="receptance"` 时退化为 `SpatialRWKVHeadV8` |
| EncoderRWKVBlock | `RWKVEncoderBlock` | 功能名 |
| TemporalMatrixRWKVBlock | `MatrixRWKVBlock` | v7 草稿 §3.3 描述的是它；v8.5 G 方案已改为均值聚合（草稿 §3.3 按 G 写的） |

## 六、当前最佳数字（引用前核对 `docs/v8/13-实验结果汇总.md`）

| 版本 | PSNR | SSIM | LPIPS | 参数量 | 备注 |
|---|---:|---:|---:|---:|---|
| v8.8-B | **26.05** | 0.8432 | 0.3426 | 1.327M | 当前最佳；Shift 共享 + 双侧 CM Shift |
| v9 CoReg-QRB | 26.01 | 0.8432 | 略差 | — | 与 v8.8-B 持平；代价 +1.9 GFLOPs / +2.5 ms |

- 验证集：`SDSD/test/low-light` 10 段序列，全帧评估。
- v7 草稿 Table 2 的 TSD-Net 行（1.175M / 99.51G / 0.467GB / 31.0ms / 32.2 FPS）是旧 fullkv 配置，**须按当前配置重测**。
- 分支置零代价（kvmod 配置）：N −0.62 dB / L −17.4 dB / M −0.57 dB —— 说明融合几乎全走 L 路，N/M 监督未达标（相对误差 0.67/0.58），这是 §4 消融与局限部分必须诚实交代的问题。
