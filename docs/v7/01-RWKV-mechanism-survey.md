# RWKV 架构演进调研与 LLVE 适用性分析

> 日期: 2026-09-28 | 仓库: TFS-Net/docs/v7/01-RWKV-mechanism-survey.md  
> 目的: 为 Golf v7 低光视频增强（LLVE）网络的重新设计提供准确的技术基础

---

## 一、RWKV 版本核心机制对照

### 1.1 状态更新公式演进

下表以单头为例，展示 RWKV 各版本状态（state）的递推公式。$D$ = 模型维度，$h$ = 头数，$d = D/h$ = 每头维度。

| 版本 | 代号 | 状态形状 | 状态更新 $S_t = \cdots$ | 特征 |
|------|------|---------|----------------------|------|
| **V4** (Dove) | — | 向量 $\mathbb{R}^D$ | $s_t = e^{-w} \odot s_{t-1} + e^{k_t} \odot v_t$ | 逐通道标量状态；归一化分母；无多头 |
| **V5** (Eagle) | RWKV-5 | 矩阵 $\mathbb{R}^{d \times d}$ | $S_t = S_{t-1} \cdot \text{diag}(w) + v_t^T \cdot k_t$ | 多头矩阵值状态；外积 $v^T k$；静态对角衰减；去归一化 |
| **V6** (Finch) | RWKV-6 | 矩阵 $\mathbb{R}^{d \times d}$ | $S_t = S_{t-1} \cdot \text{diag}(w_t) + v_t^T \cdot k_t$ | $w_t$ 数据相关（LoRA + ddlerp）；token shift 也数据相关 |
| **V7** (Goose) | RWKV-7 | 矩阵 $\mathbb{R}^{d \times d}$ | $S_t = S_{t-1} (\text{diag}(w_t) - \hat{\kappa}_t^T (a_t \odot \hat{\kappa}_t)) + v_t^T \cdot \tilde{k}_t$ | 广义 delta 规则；对角+秩1转移矩阵；向量值学习率 $a_t$；解耦移除键/替换键 |
| **V8** (Heron) | RWKV-8 | 同 V7 基座 | V7 + DeepEmbed + DEA + ROSA | DeepEmbed: 稀疏词嵌入调制 FFN；DEA: 紧凑 KV 缓存（混合架构用）；ROSA: 后缀自动机符号推理 |

### 1.2 关键差异深入解析

#### RWKV-4 → RWKV-5: 向量→矩阵

- **V4**: 状态 $s \in \mathbb{R}^D$，更新 $s_t = e^{-w} \odot s_{t-1} + e^{k_t} \odot v_t$，有归一化分母。head_size = 1。
- **V5**: 状态 $S \in \mathbb{R}^{d \times d}$，更新 $S_t = S_{t-1} \text{diag}(w) + v_t^T k_t$（外积）。head_size = 64。
  - 状态容量从 $D$ 标量扩大到 $h \times d \times d = D \times d$ 标量（例如 D=512, d=64 → 总状态 512×64=32K，是 V4 的 64 倍）
  - 去掉归一化分母，改用 LayerNorm/GroupNorm 稳定输出
  - 引入 SiLU 门控 $g_t$

**来源**: Eagle and Finch 论文 (arXiv:2404.05892) §4.1，RWKV Wiki §RWKV-V5

#### RWKV-5 → RWKV-6: 静态→动态衰减

- $w$ 从可学习但静态的逐头向量 → $w_t$ 数据相关向量
  - $d_t = \text{lora}_d(\text{ddlerp}_d(x_t, x_{t-1}))$
  - $w_t = \exp(-\exp(d_t))$
- Token Shift 从固定 lerp → 数据相关 ddlerp（借用 LoRA 机制）
- 每个 token 可独立决定每个通道的记忆衰减速度

**来源**: Eagle and Finch 论文 §4.2

#### RWKV-6 → RWKV-7: 广义 Delta 规则

**V7 的核心创新是将转移矩阵从对角矩阵扩展为对角+秩1矩阵**:

$$S_t = S_{t-1} \underbrace{(\text{diag}(w_t) - \hat{\kappa}_t^T (a_t \odot \hat{\kappa}_t))}_{G_t: \text{转移矩阵}} + v_t^T \tilde{k}_t$$

具体改进:

1. **向量值在线学习率 $a_t$**: 原始 DeltaNet 用标量 $a$；V7 用向量，允许逐通道决定"替换多少旧记忆"
2. **解耦移除键 $\kappa$ / 替换键 $\tilde{k}$**: V7 引入可学习参数 $\xi$ 和 $\alpha$，分别调节移除键和替换键
   - $\kappa_t = k_t \odot \xi$（移除键 = 原始键 × 学到的变换）
   - $\tilde{k}_t = k_t \odot \text{lerp}(1, a_t, \alpha)$（替换键 = 原始键 × 学习率调节）
3. **值残差学习**: $v_t = \text{lerp}(v_{t,0}', v_{t,l}', \nu_t)$，跨层共享第0层的 value
4. **转移矩阵可产生负特征值**: 理论上超越 $\text{TC}^0$，可用常数层识别所有正则语言
5. **Token Shift 简化**: 去掉 V6 的 ddlerp，回退到简单 lerp（提速）
6. **MLP 简化**: 去掉 Channel Mixing 的门控 $W_r$，改为标准 2 层 MLP + ReLU²，hidden = 4D

**来源**: RWKV-7 论文 (arXiv:2503.14456) §3-4, RWKV Wiki §RWKV-V7

**V7 参考代码 (来自论文 Appendix H)**:
```python
def ref_fwd(r, w, k, v, a, b):  # a = -κ̂, b = κ̂ ⊙ a_t
    state = torch.zeros((B, H, N, N), device=DEVICE)
    for t in range(T):
        sab = torch.einsum('bhik,bhk,bhj->bhij', state, a[:,t], b[:,t])
        state = state * w[:,t,:,None,:] + sab + torch.einsum('bhj,bhi->bhij', k[:,t], v[:,t])
        out[:,t] = torch.einsum('bhj,bhij->bhi', r[:,t], state)
    return out
```

#### RWKV-8: 实验性特征（2025-2026）

- **DeepEmbed**: 为词表中每个 token 在每层 FFN 中训练高维嵌入向量，用于 channelwise 乘法调制。推理时可卸载到 RAM/SSD，不占 VRAM。本质是稀疏 MoE 的极端形式。
- **DeepEmbedAttention (DEA)**: 为混合架构设计的紧凑 KV 缓存（仅 1/9 MLA 大小）
- **ROSA**: Rapid Online Suffix Automaton，离散符号推理组件，无需浮点计算，可在 CPU 并行运行。用于精确上下文检索、算术推理等。

**LLVE 适用性评估**: DeepEmbed 和 ROSA 面向 NLP 词汇级任务，与像素级视觉任务距离较远。DEA 的混合注意力思想有一定参考价值，但当前不建议在 LLVE 中引入。

---

## 二、各机制在 LLVE 任务中的角色分析

### 2.1 LLVE 任务特性回顾

低光视频增强的核心退化模型:
$$I_t = X_t \odot \ell_t + n_t$$

| 退化分量 | 帧间特性 | 频域 | 对应操作 |
|---------|---------|------|---------|
| 成像噪声 $n_t$ (Poisson+Gaussian) | i.i.d. | 全频段,HF显著 | 时间平均去噪 |
| 光照衰减 $\ell_t$ | 帧间强相关,慢变 | 低频 | 光照校正 |
| 运动位移 | 帧间结构变化 | 稀疏局部 | 对齐后融合 |

**关键约束**:
- T = 5 帧（短序列）
- 训练 crop = 256², 推理 1080×1920 (tiled)
- GPU = RTX 4090 (24GB)
- 参数量 ≤ 5M

### 2.2 机制-角色映射表

| RWKV 机制 | 来源 | LLVE 中的角色 | 具体用途 | 价值评估 |
|-----------|------|-------------|---------|---------|
| **矩阵值状态** $S \in \mathbb{R}^{d \times d}$ | V5+ | 帧级时序记忆 | 存储帧间光照/噪声/运动的关联模式 | ★★★★☆ 核心能力，容量远大于 V4 向量状态 |
| **多头分解** | V5+ | 噪声分量分离 | 不同头关注不同退化分量（N/L/M） | ★★★★★ 直接对应三分支结构 |
| **数据相关衰减 $w_t$** | V6+ | 自适应时序加权 | 运动大→衰减快（忘旧帧）;静止→衰减慢（多帧平均） | ★★★★★ 天然适合运动自适应去噪 |
| **广义 delta 规则** ($\kappa, a_t$) | V7 | 精准记忆更新 | 用 $\kappa$ 移除过时的帧特征，用 $\tilde{k}$ 写入新帧；$a_t$ 控制替换力度 | ★★★★☆ 对 T=5 短序列的增益需要验证 |
| **解耦移除/替换键** | V7 | 选择性记忆管理 | 移除键可定位"被遮挡区域的旧值"，替换键写入"新曝光区域" | ★★★☆☆ 概念有吸引力，但 T=5 场景下记忆冲突不严重 |
| **值残差学习** | V7 | 跨层特征复用 | 第0层的 value 在所有层间共享，防止深层遗忘浅层纹理 | ★★★★☆ 有利于保持空间细节 |
| **ReLU² MLP** | V7 | 通道混合 | 替代 V4-V6 的门控 FFN，更简洁高效 | ★★★☆☆ 微调级别改进 |
| **Token Shift** | V3+ | 帧间信息混合 | 当前帧与前一帧特征的可学习混合 | ★★★★☆ 在 T=5 帧级 token 上很自然 |
| **DeepEmbed** | V8 | 不适用 | 面向词表级调制，无对应视觉概念 | ☆☆☆☆☆ |
| **ROSA** | V8 | 不适用 | 离散符号推理，与连续像素特征不兼容 | ☆☆☆☆☆ |

### 2.3 核心洞察

1. **RWKV-6 级别的机制已足够 LLVE**: T=5 是极短序列，V7 的广义 delta 规则在长上下文中优势显著（解决 $O(N)$ 的记忆瓶颈），但 T=5 时状态几乎不会溢出。**V6 的数据相关衰减 + 矩阵状态 + 多头** 是性价比最高的组合。
2. **V7 的值残差学习可以跨层保留空间细节**，这在 LLVE 中尤其重要（避免深层网络丢失高频纹理）。建议采用。
3. **不应在像素级使用 RWKV**（此前 Golf R2-R5 的教训）: RWKV 的 token shift 和递归聚合设计面向序列 token，在 2D 空间像素上效果差。正确的分工是: **像素级用 CNN/Attention，帧级用 RWKV**。
4. **多头自然对应多分支噪声分割**: 可以将 h 个 RWKV 头显式分配给 N/L/M 三类退化分量，通过头间正交约束实现噪声分离。这是 RWKV 多头与三分支架构的自然融合点。

---

## 三、Vision-RWKV 相关工作

| 工作 | 年份 | 关键设计 | 与 LLVE 的关系 |
|------|------|---------|--------------|
| **Vision-RWKV (VRWKV)** | 2024 | 双向空间扫描 + RWKV-4 在图像 token 上 | 证明 RWKV 可用于 2D 视觉，但使用 V4（无矩阵状态）|
| **Restore-RWKV** | 2026 | RWKV-6 用于医学图像修复，1.16M 参数 SOTA | 低参数量 RWKV 在修复任务上的成功案例 |
| **URWKV** | 2026 | 多状态 RWKV 用于低光图像（非视频） | 直接相关：低光 + RWKV，但是单帧 |
| **DarkIR** | 2025 | RWKV 用于低光去模糊 + 去噪，SOTA on LOL-Blur | 证明 RWKV 在低光修复上有竞争力 |
| **RD-Fusion** | 2025 | RWKV 用于红外-可见光融合 | RWKV 在多模态融合中的局部/全局建模 |
| **Restore-RWKV (Medical)** | 2026 | RWKV-6 + token shift 强化局部特征 | "token shift 被其他架构忽略" 的观察值得借鉴 |

**关键结论**: RWKV 已在图像修复/低光增强/融合任务中证明有效，但均为**单帧**或**空间维度**应用。**时间维度的帧级 RWKV** 尚无先例，这是我们的创新点。

---

## 四、推荐采用的 RWKV 机制清单

基于 LLVE 任务需求和模型约束，推荐以下组合:

| 机制 | 来源 | 采用理由 |
|------|------|---------|
| 矩阵值状态 $S \in \mathbb{R}^{d \times d}$ | V5 | 核心能力，帧间关系需要矩阵容量 |
| 多头 (h=4-8) | V5 | 对应噪声分割的 N/L/M 分支 |
| 数据相关衰减 $w_t$ | V6 | 运动自适应是 LLVE 的核心需求 |
| Token Shift (简单 lerp) | V7 | V7 版本更简洁（去掉 ddlerp），适合短序列 |
| 值残差学习 | V7 | 跨层保留空间纹理 |
| LayerNorm per head | V5 | 稳定多头输出 |
| SiLU 门控 | V5 | 控制信息流 |
| ReLU² MLP (Channel Mix) | V7 | 简洁高效的通道混合 |

**不采用**:
- V7 广义 delta 规则（$\kappa, a_t$ 解耦）: T=5 太短，收益不确定，增加实现复杂度
- V8 DeepEmbed / ROSA: 面向 NLP，与 LLVE 不相关
- V6 ddlerp Token Shift: V7 实验表明简单 lerp 速度更快且效果相当

---

## 五、参考文献

1. Peng et al., "RWKV: Reinventing RNNs for the Transformer Era", EMNLP 2023 (arXiv:2305.13048)
2. Peng et al., "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence", COLM 2024 (arXiv:2404.05892)
3. Peng et al., "RWKV-7 'Goose' with Expressive Dynamic State Evolution", 2025 (arXiv:2503.14456)
4. RWKV Wiki, "RWKV Architecture History", https://wiki.rwkv.com/basic/architecture.html
5. Yang et al., "Parallelizing DeltaNet", NeurIPS 2024 (arXiv:2406.06484)
6. Schlag et al., "Linear Transformers Are Secretly Fast Weight Programmers", ICML 2021
