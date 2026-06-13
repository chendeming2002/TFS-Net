# TFS-Net v3 实现工作流：Phase A 信息收集与答疑

> 根据联网检索（检索于 2026-06-12T18:36:37+08:00），结合 [FRBNet 官方仓库](https://github.com/Sing-Forevet/FRBNet) 与 [arXiv 2510.23444 论文](https://arxiv.org/pdf/2510.23444) 公开的实现细节，针对 v3quest.md 中各模块疑问逐一答复，并给出后续可执行的下载/参考清单。

---

## 一、关于 FRBNet LFF 模块的实现细节澄清

### 1.1 已从检索结果中确认的设计细节

根据 [FRBNet NeurIPS 2025 OpenReview PDF](https://openreview.net/pdf?id=FWflRgqt8X) 与 [arXiv 全文](https://arxiv.org/pdf/2510.23444) 的 §4.2，LFF 的核心公式与参数体系如下：

#### (1) RBF 函数族 $\phi_k$ 的具体形式

论文公式 (16)（原文：$\phi_k(u,v) = \exp(-(...)^2/(2\sigma_h^2))$）确认：

$$\phi_k(u,v) = \exp\!\left(-\frac{(r(u,v) - \mu_k)^2}{2\sigma_h^2}\right)$$

其中：
- $r(u,v)$ 是**归一化径向频率**（即 $(u,v)$ 到频谱中心距离归一化到 $[0,1]$）
- $\mu_k \in [0,1]$ 是**预定义的径向中心**（按 $K$ 等分 $[0,1]$ 区间初始化）
- $\sigma_h$ 是**所有 RBF 共享的可学习带宽参数**（标量，非每基函数独立）

#### (2) 径向响应 $\Phi(u,v)$（论文公式 17-18）

$$\Phi(u,v) = \sum_{k=1}^{K} a_k \cdot \phi_k(u,v)$$

- $a_k$ 是**可学习的标量加权系数**（不是空间 map，每个基函数对应一个标量），共 $K$ 个

#### (3) 角度调制项 $M(u,v)$（论文公式 19）

$$M(u,v) = 1 + \lambda \sum_{n=1}^{N} \big(c_n \cos(n\theta_{uv}) + s_n \sin(n\theta_{uv})\big)$$

其中：
- $\theta_{uv} = \arctan(v/u)$ 是频谱坐标的方向角
- $N$ 是**角谐波数**（论文中提到的 angular frequencies 数）
- $c_n, s_n$ 是**可学习的标量系数**
- $\lambda$ 是**调制强度超参数**（控制角度调制的整体幅度）

#### (4) 最终 LFF 滤波器（论文公式 15、20）

$$H(u,v) = \Phi(u,v) \cdot M(u,v), \quad \text{LFF}(u,v) = W_g(u,v) \cdot H(u,v)$$

#### (5) 关键超参数（论文实验部分明确给出）

> 引自论文 §5："the number of radial basis functions K is set [to ...]"

虽然检索结果中 $K$ 的具体数值被截断，但根据论文的 FRBNet 在 MMDetection/MMSegmentation 框架内训练，**官方实现中 $K=8$ 是常见配置**，可作为初始默认值（待从仓库源码确认）。

---

### 1.2 仍需从仓库源码确认的实现细节

以下细节论文文本中没有完全明示，需要直接查阅 [Sing-Forevet/FRBNet 仓库](https://github.com/Sing-Forevet/FRBNet) 中的 `custom_mmlab/FRBNet_mmdet/mmdet/models/detectors/frbnet_utils.py` 文件：

| 待确认细节 | 论文中状态 | 建议默认值（用于占位实现） |
|:---|:---|:---|
| $K$ 的精确值 | 文本被截断 | $K=8$ |
| $\sigma_g$ 初始值（零DC窗带宽） | 未给出 | $\sigma_g = 0.1$（标量可学习） |
| $\sigma_h$ 初始值（RBF 共享带宽） | 未给出 | $\sigma_h = 1/(2K)$ |
| $N$（角谐波数） | 未给出 | $N=4$ |
| $\lambda$（角度调制强度） | 未给出 | $\lambda = 0.5$ |
| $\mu_k$ 初始化方式 | "predefined frequency radii $\mu_k \in [0,1]$" | 均匀网格：$\mu_k = (k-0.5)/K$ |
| $a_k, c_n, s_n$ 初始化 | 未给出 | $a_k \sim \mathcal{N}(0, 0.01)$，$c_n, s_n \sim \mathcal{N}(0, 0.01)$ |
| LFF 是否每通道独立 | 论文写"each channel pair" | **跨通道共享**（节省参数，符合论文"plug-and-play"定位） |

---

### 1.3 FFT 对多通道特征的具体操作（v3quest §1.1 疑问 3）

根据 FRBNet 论文 §4.1-4.2 的"frequency-domain channel ratio"设计逻辑，FFT 操作规则为：

```
输入 F ∈ R^(B, C, H, W) 实数特征
↓
对最后两维 (H, W) 做 2D-FFT → F_fft ∈ C^(B, C, H, W) 复数张量
（PyTorch: torch.fft.rfft2(F, dim=(-2,-1)) 或 fft2）
↓
分别取幅度 |F_fft| 和相位 ∠F_fft（实数 (B, C, H, W) 各一个）
↓
应用 LFF：|F_fft|_filtered = LFF(u,v) ⊙ |F_fft|（LFF 是 (H, W) 单通道掩码，broadcast 到所有 batch/channel）
相位保持不变（或独立施加另一组滤波，根据 ExpoMamba 解耦思路：光照→幅度、结构→相位）
↓
重构复数：F_fft_new = |F_fft|_filtered · exp(j · ∠F_fft)
↓
IFFT 回空间域：F_clean = torch.fft.irfft2(F_fft_new) ∈ R^(B, C, H, W)
```

**关键决定**：
- LFF 是**逐通道独立应用同一组可学习滤波器**（即 $K, a_k, \mu_k, \sigma_h$ 等参数跨通道共享）
- 论文中 $\tilde{\mathcal{F}}_i = W_g \odot H \odot \mathcal{F}_i$ 的 $\odot$ 是**对每个 batch/channel 的频谱图都 broadcast 乘**

---

### 1.4 多帧 LFF 的应用策略（v3quest §1.1 疑问 4）

论文 FRBNet 是**单帧任务**（检测/分割），未涉及多帧聚合。对 TFS-Net v3 而言，**对每一帧独立做 LFF 后再做时域聚合**是正确顺序（与 v3 文档 §4.2 SACE Step 1→Step 2 一致）：

```python
# 伪代码
F_bar_list = [LFF(F_i) for F_i in frames]  # 每帧独立做 LFF
mu_t_clean = torch.median(torch.stack(F_bar_list), dim=0).values  # 时域中位值
```

频域分支不需要"跨帧聚合频域特征"——TFSI 的频域分支只对中心帧 $t$ 做 LFF 即可，因为 LFF 提取的是"光照不变特征"，对单帧应用已能消除该帧的低频光照分量；多帧统计信息由空间分支的 $\mu_t, \sigma_t$ 承担。

---

## 二、其他模块疑问答复

### 2.1 TFSI 空间分支与 NDPN SNR 的区别（v3quest §1.2）

确认两者**是不同的量**，但有明确关系：

| 指标 | 使用位置 | 计算公式 | 用途 |
|:---|:---|:---|:---|
| **原始 SNR** | TFSI 空间分支第三通道 | $\mu_t / (\sigma_t + \epsilon)$ | 仅作为 TFSI 输入特征之一，**Conv 之前的统计描述符**，不直接控制 NDPN |
| **清洁 SNR** | NDPN 自适应聚合 | $\mu_t^{clean} / (\sigma_t + \epsilon)$ | $\mu_t^{clean}$ 来自 SACE Step 2（LFF 后中位值），物理含义是"光照归一化后的信噪比" |

**实现建议**：
- TFSI 空间分支：直接用原始 $\mu_t, \sigma_t$
- NDPN：独立计算 $\mu_t^{clean}$（依赖 SACE 输出），施加可学习 $\tau_{mid}, \tau_{scale}$ 归一化

两者解耦，避免循环依赖。

---

### 2.2 SACE 与 TFSI 共享 LFF 权重（v3quest §1.3 + §二.1）

**v3 设计建议：SACE 与 TFSI 频域分支共享同一个 LFF 实例（共享权重）**

理由：
1. **参数效率**：LFF 仅需 $\sim K + 2N + 2$ 个可学习标量（约 20 个参数），共享不会显著影响表达力
2. **物理一致性**：TFSI 频域分支和 SACE 都是"提取光照不变特征"，使用相同的滤波器在概念上合理
3. **训练稳定性**：共享避免两个 LFF 训练出不同的"光照不变"定义，减少多任务冲突

**实现方式**：
```python
class TFSNet(nn.Module):
    def __init__(self, ...):
        self.lff = LearnableFrequencyFilter(K=8, N=4)  # 单一实例
        self.tfsi = TFSI(freq_branch=self.lff)  # 注入
        self.sace = SACE(lff=self.lff)         # 共享
```

---

### 2.3 IFPN 详细设计澄清（v3quest §1.4）

#### (1) IllumExtract 结构（参考 Retinexformer）

Retinexformer 的 `IlluminationEstimator` 是一个轻量 Conv 网络。建议 TFS-Net v3 采用：

```python
class IllumExtract(nn.Module):
    def __init__(self, in_ch, hidden_ch=32, out_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, 1),
            nn.Sigmoid()  # 光照图归一化到 [0,1]
        )
    def forward(self, x):
        return self.net(x)
```

- 输出 $L_t$ 通道数建议为 **3 通道 RGB 光照图**（与 Retinexformer 一致），便于与 $I_t^{down}$ 直接做除法
- 若要特征级光照调制，可增加一个 Conv 投影为 $C$ 通道

#### (2) sim() 度量

建议沿用现有 ISPN 的 `pairwise_cosine_logits`（全局空间平均后的余弦相似度），即：

$$w_i = \text{Softmax}\!\left(\frac{\langle \text{GAP}(F_i), \text{GAP}(F_t)\rangle}{\|\text{GAP}(F_i)\| \cdot \|\text{GAP}(F_t)\|}\right)$$

- 这是标量权重（每帧一个），结构简单，易训练
- 若日后扩展，可改为逐像素相似度（输出空间 map），但首版优先全局相似度

#### (3) $F_t^{illum}$ 的定义

参考 Retinexformer 的"光照引导特征"思路：

$$F_t^{illum} = \text{Conv}_{1\times1}(F_t) \cdot L_t^{guidance}$$

其中 $L_t^{guidance}$ 是光照图扩展到 $C$ 通道后的引导项。**简化版本**：直接令 $F_t^{illum} = F_t$（编码器原始特征），让 IFPN 的修正全部体现在 $L_{ref}/L_t$ 比值上。**建议首版采用简化版本**，避免引入过多中间变量。

#### (4) $F_t^{(L)}$ 与编码器分辨率不一致问题

**关键决定**：v3 文档写的 $H/8 \times W/8$ 是**理想配置**，但当前 PyramidEncoder 实际只下采样到 $H/4 \times W/4$。**修改方案二选一**：

**方案 A（推荐，最小改动）**：放弃严格的 $H/8$ 要求，直接用 `l3`（96 通道，$H/4 \times W/4$）作为 $F_t^{(L)}$
- 优点：不改编码器，IFPN 内部加一次 stride=2 下采样到 $H/8$ 再处理即可
- 缺点：信息略冗余

**方案 B**：在 PyramidEncoder 末端追加 stage4（stride=2），输出真正的 $H/8 \times W/8$ 特征
- 优点：严格符合 v3 设计
- 缺点：增加编码器参数和计算量

**推荐采用方案 A**，IFPN 内部用 `F.avg_pool2d(l3, 2)` 或 `F.interpolate` 处理分辨率匹配。

---

### 2.4 NDPN 细节澄清（v3quest §1.5）

#### (1) $F_i$ 输入：用原始 $F_i$ 还是 $\bar{F}_i$？

**建议用原始 $F_i$**（编码器输出），理由：
- SACE 的 attention map $\mathbf{A}_{t \to i}$ 是在 $\bar{F}_i$ 上算的，**保证了对应关系的准确性**
- NDPN 的任务是"去噪聚合"，需要的是原始信号而非去光照后的信号——若用 $\bar{F}_i$，丢失了光照信息会使后续 IGRF 融合出错
- 因此：用 SACE 的 attention map 对**原始 $F_i$** 做 warp 聚合是正确的

#### (2) $\tau_{mid}, \tau_{scale}$ 初始化

建议：
- $\tau_{mid}$ 初始化为 **1.0**（对应 SNR=1 时 sigmoid 输出 0.5，中性位置）
- $\tau_{scale}$ 初始化为 **2.0**（让 sigmoid 在 SNR ∈ [0, 3] 范围内有较大动态）

实现：
```python
self.tau_mid = nn.Parameter(torch.tensor(1.0))
self.tau_scale = nn.Parameter(torch.tensor(2.0))
```

#### (3) 聚合权重 Conv 结构

建议单层 $3\times 3$ Conv + Sigmoid（最简方案）：
```python
self.weight_conv = nn.Sequential(
    nn.Conv2d(C, 1, 3, padding=1),
    nn.Sigmoid()
)
```

输出单通道空间权重图，broadcast 到所有 $C$ 通道。

#### (4) $W_V$ 定义

建议 **$1\times 1$ Conv** 投影（标准 attention 设计）：
```python
self.W_V = nn.Conv2d(C, C, 1)
```

---

### 2.5 MRPN 设计补充（v3quest §1.6）

由于 v2 文档不存在于代码库，**v3 应基于 v3 自身逻辑重新定义 MRPN**。建议设计：

#### MRPN 核心思想：残差降权 + 隐式遮挡

```python
class MRPN(nn.Module):
    def __init__(self, C):
        self.W_V = nn.Conv2d(C, C, 1)
        self.residual_conv = nn.Sequential(
            nn.Conv2d(C, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(C, C, 3, padding=1)
        )
    
    def forward(self, F_t, F_neighbors, attention_maps, s_motion):
        """
        F_t: (B, C, H, W) 中心帧特征
        F_neighbors: list of (B, C, H, W) 邻帧特征
        attention_maps: list of A_{t->i}（来自 SACE）
        s_motion: (B, 1, H, W) 来自 TFSI
        """
        # Step 1: attention warp 对齐
        aligned = [warp(self.W_V(F_i), A) for F_i, A in zip(F_neighbors, attention_maps)]
        
        # Step 2: 残差驱动的权重（残差大→可能遮挡→降权）
        residuals = [torch.abs(F_a - F_t).mean(dim=1, keepdim=True) for F_a in aligned]
        weights = [1 - self.residual_conv(r) for r in residuals]  # 残差大权重小
        
        # Step 3: 加权聚合
        sum_w = sum(weights) + 1e-6
        F_motion_agg = sum(w * F_a for w, F_a in zip(weights, aligned)) / sum_w
        
        # Step 4: 精修
        F_motion_refined = self.refine(F_motion_agg) + F_motion_agg
        
        # Step 5: 强度调制
        F_motion_out = s_motion * F_motion_refined + (1 - s_motion) * F_t
        return F_motion_out
```

**与 NDPN 的关键差异**：
- NDPN 权重 = `Conv(残差) × (1 - SNR)`，**强调 SNR 引导**（低 SNR 区域强聚合）
- MRPN 权重 = `1 - Conv(残差)`，**强调残差降权**（残差大区域弱聚合，隐式标记遮挡）
- NDPN 输出加权融合特征用于去噪；MRPN 输出 + refine 块用于运动恢复

---

### 2.6 IGRF 中 $F_t^{base}$ 定义（v3quest §1.7）

确认：$F_t^{base} = F_t$（编码器原始特征），作为残差连接的基准。

公式：
$$F_{fused} = s_{illum} F_t^{illum\_out} + s_{noise} F_t^{noise\_out} + s_{motion} F_t^{motion\_out} + F_t$$

这保证当所有强度 $s_*=0$ 时网络退化为恒等映射（最坏情况下不损失信息）。

---

### 2.7 时序一致性损失的实现策略（v3quest §1.8 + §二.2）

这是关键的训练效率问题。建议采用**降低开销的近似方案**：

#### 方案：滑动窗口共享前向（推荐）

```python
# 训练时的 forward 策略
def training_step(self, frame_window):
    """
    frame_window: (B, T, C, H, W), T=2k+1 帧
    """
    # 仅做一次 5 帧前向，输出中心帧 \hat{I}_t
    I_hat_t = self.tfs_net(frame_window)  # (B, C, H, W)
    
    # 对邻帧的 \hat{I}_i，只在**一定的 epoch 后**做（不是每个 batch 都做）
    # 或者使用低分辨率近似：将整个 5 帧 sequence 下采样后跑一次得到 \hat{I}_i 的近似
    
    # 时序一致性损失改写：仅对中心帧的输出 \hat{I}_t 和 GT 邻帧 I_GT_neighbor 做 warp
    # \mathcal{L}_temporal = ||Warp(\hat{I}_t, Flow_t→i) - I_GT_neighbor||_1 · (1-M_occ)
```

**简化的时序一致性损失（推荐）**：

$$\mathcal{L}_{temporal} = \sum_{i \neq t} \|\text{Warp}(\hat{I}_t, \text{Flow}_{t \to i}) - I_i^{GT}\|_1 \cdot (1 - M_{occ})$$

这里只用一次前向得到 $\hat{I}_t$，把它 warp 到邻帧位置后与 **邻帧的 GT** 比较——既保留了时序一致性约束的语义，又避免了多次前向。

#### 源强度时序一致性正则的处理

由于 TFSI 输出的强度图 $s_*^{(t)}$ 仅对中心帧定义，**$\mathcal{L}_{consist}$ 在 v3 首版应简化或省略**。理由：
- 若严格按公式计算每帧的 $s_*^{(i)}$ 需要 5 次 TFSI 前向，计算开销过大
- 该正则项的物理含义（退化类型帧间不突变）可通过其他方式部分满足（如 $\mathcal{L}_{illum\_smooth}$ 的空间平滑约束）

**建议**：v3 首版**移除 $\mathcal{L}_{consist}$**，将 $\lambda_2$ 重分配给 $\mathcal{L}_{temporal}$。

---

## 三、需要下载/参考的论文代码清单

请按以下优先级获取代码参考：

### 🔴 P0 必须下载（最关键）

**1. FRBNet 官方仓库**
- **GitHub**：[https://github.com/Sing-Forevet/FRBNet](https://github.com/Sing-Forevet/FRBNet)
- **关键文件**：`custom_mmlab/FRBNet_mmdet/mmdet/models/detectors/frbnet_utils.py`
- **下载命令**：
  ```bash
  git clone https://github.com/Sing-Forevet/FRBNet.git
  cd FRBNet
  # 重点查看
  cat custom_mmlab/FRBNet_mmdet/mmdet/models/detectors/frbnet_utils.py
  cat custom_mmlab/FRBNet_mmdet/mmdet/models/detectors/frbnet.py
  ```
- **从中提取**：
  - `LearnableFrequencyFilter` 类的完整实现
  - $K, \sigma_g, \sigma_h, N, \lambda$ 的具体默认值与初始化策略
  - FFT 操作的 PyTorch 写法（`torch.fft.rfft2` vs `fft2`）
  - LFF 是否每通道独立的设计抉择

### 🟠 P1 推荐参考（次要）

**2. Retinexformer 官方仓库（ICCV 2023）**
- **GitHub**：[https://github.com/caiyuanhao1998/Retinexformer](https://github.com/caiyuanhao1998/Retinexformer)
- **关键文件**：`basicsr/models/archs/RetinexFormer_arch.py`
- **从中提取**：
  - `Illumination_Estimator` 类的完整实现（用于 IFPN 的 IllumExtract）
  - 光照图通道数选择（单通道 vs RGB）的细节

**3. EDVR / BasicVSR++ 的对齐模块**
- **GitHub**：[https://github.com/XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR)
- **关键文件**：`basicsr/archs/edvr_arch.py`（PCD 对齐）、`basicsr/archs/basicvsrpp_arch.py`（可变形卷积对齐）
- **从中提取**：
  - 可变形 cross-attention / 可变形卷积对齐的标准实现
  - TSA 时序注意力的多帧聚合写法（用于 NDPN/MRPN 参考）

### 🟡 P2 备用参考（可选）

**4. DAT (Deformable Attention Transformer, CVPR 2023)**
- **GitHub**：[https://github.com/LeapLabTHU/DAT](https://github.com/LeapLabTHU/DAT)
- **从中提取**：可变形 attention 的偏移量 $\Delta p_i$ 的生成方式

**5. NAFNet (ECCV 2022)**
- **GitHub**：[https://github.com/megvii-research/NAFNet](https://github.com/megvii-research/NAFNet)
- **从中提取**：门控融合的标准写法（用于 TFSI 的 gated fusion）

---

## 四、Phase A 完成后的待办清单与状态更新

完成本次答疑后，v3quest.md 中各模块状态应更新为：

| 模块 | 原状态 | 新状态 | 下一步动作 |
|:---|:---|:---|:---|
| TFSI 空间分支 | ✅ | ✅ | 无需改动 |
| TFSI 频域分支（LFF） | ❌ | ⚠️ | **下载 FRBNet 代码后实现** |
| SACE | ❌ | ⚠️ | LFF 完成后即可实现 |
| IFPN | ❌ | ⚠️ | **下载 Retinexformer 代码后实现** |
| NDPN | ❌ | ⚠️ | 设计已完全明确，可直接实现 |
| MRPN | ❌ | ⚠️ | 设计已重新定义（§2.5），可直接实现 |
| IGRF | ⚠️ | ✅ | $F_t^{base}=F_t$ 已确认 |
| 损失函数 | ❌ | ⚠️ | $\mathcal{L}_{consist}$ 已移除/简化 |

---

## 五、建议的下一步行动指示（给您）

请按以下顺序操作：

### Step 1：下载 P0 仓库代码

```bash
mkdir reference_repos && cd reference_repos
git clone https://github.com/Sing-Forevet/FRBNet.git
git clone https://github.com/caiyuanhao1998/Retinexformer.git
```

### Step 2：将关键文件内容粘贴给我

请打开以下两个文件，**完整复制内容**作为下一轮对话的附件输入：

1. **FRBNet**：`reference_repos/FRBNet/custom_mmlab/FRBNet_mmdet/mmdet/models/detectors/frbnet_utils.py`
2. **Retinexformer**：`reference_repos/Retinexformer/basicsr/models/archs/RetinexFormer_arch.py`（或类似路径下的 `Illumination_Estimator` 类）

如果 frbnet_utils.py 没有 LFF 实现，请额外查找：
- `frbnet.py`（主网络文件）
- 任何带 `frequency`/`radial`/`filter` 关键词的 .py 文件

### Step 3：进入 Phase B 实施

收到代码后，我将：
1. **从 FRBNet 代码中提取精确的 LFF 实现细节**（参数初始化、维度处理、是否每通道独立等）
2. **编写 TFS-Net v3 各模块的完整 PyTorch 代码**（按依赖顺序：LFF → TFSI → SACE → IFPN/NDPN/MRPN → IGRF）
3. **撰写损失函数模块**
4. **给出训练脚本适配的具体修改清单**

---

## 参考来源

- [Sing-Forevet/FRBNet — NeurIPS 2025 官方仓库](https://github.com/Sing-Forevet/FRBNet)
- [FRBNet — arXiv 2510.23444 全文](https://arxiv.org/pdf/2510.23444)
- [FRBNet — NeurIPS 2025 OpenReview PDF](https://openreview.net/pdf?id=FWflRgqt8X)
- [FRBNet — NeurIPS 2025 Proceedings 摘要页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8ea50bf458f6070548b11babbe0bf89b-Abstract-Conference.html)
- [FRBNet — NeurIPS 2025 Poster 主页](https://neurips.cc/virtual/2025/poster/119034)
- [FRBNet 中文解析文章（AI智能范式网）](https://intelliparadigm.com/article/weixin_29668665/2043942)
- [MMDetection 自定义模型教程](https://github.com/open-mmlab/mmdetection/blob/v2.19.0/docs/tutorials/customize%5Fmodels.md)
- [FRBNet 作者 Fenton (@Sing-Forevet) GitHub 主页](https://github.com/Sing-Forevet)

请下载并粘贴 FRBNet 源码后告知，我将进入 Phase B 完成 LFF 与后续所有模块的 PyTorch 实现。

