# TFS-Net 三源退化特征空间可分离性实验设计

> **目标**：在本模型 `PyramidEncoder` 特征空间中，证明 TFS-Net/MINS-Net 建模的三源退化（光照 γ / 传感噪声 n / 运动模糊 k）可分离。
>
> **方法论来源**（双参考）：
> - **DPMambaIR (arXiv 2504.17732v3, 2026)**：通过重建解耦（reconstruction-based disentanglement）证明退化嵌入与内容嵌入可分离——退化嵌入 E_d 从退化图提取，内容嵌入 E_c 从干净 GT 提取，重建器 R(E_c, E_d) 重建退化图。论文 Fig.9 t-SNE 可视化 + Table VII 线性探针分类准确率（Cls.Acc.=92%）实证可分性。
> - **Condformer (IJCV 2025)**：通过局部 patch 可估计性证明噪声先验与内容可分离——LoNPE 从 32×32 patch 估计全局噪声参数，8 patch median 抑制内容偏差。
>
> **与现有 `Experience/SDSD-Trid/Visual_diff_noises.py` 的根本区别**：现有实验用 ImageNet 预训练 ResNet50 特征 + 手工特征（循环论证）+ 2 类分类 + 单次 silhouette。本实验用**本模型自己的 PyramidEncoder 特征** + **DPMambaIR 风格重建解耦 + Condformer 风格局部先验估计** + **线性探针 + 置换检验**（有公信力）。

---

## 一、核心论证链（双参考：DPMambaIR + Condformer）

### 1.1 DPMambaIR 的可分性论证（论文 Eq.4-7, Fig.9, Table VII）

DPMambaIR 证明退化嵌入与内容嵌入可分离的三个支柱（论文 §III-C, lines 423-446）：

1. **重建解耦**（Eq.6-7）：
   - 退化嵌入：`E_d = E(I^D)`（从退化图提取）
   - 内容嵌入：`E_c = Restormer(O)`（从干净 GT 提取，参数冻结）
   - 重建：`I^D_hat = R(E_c, E_d)`（组合两者重建退化图）
   - **关键论证（line 439）**："This disentanglement ensures that E_d captures pure degradation information necessary to reconstruct the corruption, independent of the image content."
   - **为什么 E_d 不含内容**：内容由 E_c 提供，E_d 只需补充"退化差异"。若 E_d 编码了内容，则 R 会冗余，损失不会驱动 E_d 学习内容。

2. **连续退化流形**（line 424-427）：不用分类标签（离散），用重建目标（连续），强制 E_d 学习退化类型+强度的连续流形。

3. **实证可分性**（Fig.9 + Table VII）：
   - Fig.9 t-SNE：不同退化类型形成明显分离的簇。关键发现（line 778-781）："JPEG 和 noise 共享相同内容，但 t-SNE 可分性由退化信息而非图像内容驱动"。
   - Table VII 线性探针：在冻结 extractor 上加 MLP 头，1000 iterations 训练，7 类 Cls.Acc.=92.0%（Demb=512）。
   - Fig.5 物理可解释性：Δdp 分布显示 noise/jpeg→小步长（高惯性低通），blur/low-light→大步长（高增益放大）。

### 1.2 Condformer 的可分性论证（`reference_repos/Condformer/`）

Condformer 从另一角度证明噪声先验可分离（`model.py:58-93`, `train.py:436-506`）：

1. **LoNPE 局部可估计性**：`PlainCNN(32×32 patch) → GAP → Linear(64→2)` 从局部 patch 估计全局噪声 `[σ_s, σ_r]`。8 patch median（`train.py:46`）抑制内容偏差。成立前提：噪声 i.i.d. → 局部统计量 = 全局统计量。
2. **CondAttention 调制**（`basic_module.py:190-226`）：`LFM_layer` 把 2D 先验广播到 H×W，调制 Q/K。先验低维全局，内容高维空间，子空间不交叠。
3. **GT 监督**（`train.py:39`）：`loss_NE = L1(prior, img_sigma)`，强制先验只编码噪声。

### 1.3 两者的互补性

| 维度 | DPMambaIR | Condformer | 本设计融合 |
|:---|:---|:---|:---|
| 解耦机制 | 重建解耦（E_c 从 GT，E_d 从退化图） | 局部可估计性（patch median 去内容） | **双轨**：重建解耦 + 局部 patch |
| 退化维度 | 7 类（噪声/模糊/低光/雨/雪/雾/JPEG） | 2D `[σ_s, σ_r]` | **三源 6D** `[α,γ,σ_s,σ_r,θ,L]` |
| 验证方法 | t-SNE + 线性探针 Cls.Acc. | L1 估计误差 | t-SNE + 线性探针 + 先验回归 R² |
| 物理可解释 | Δdp 分布 | — | **三源参数回归** |
| 内容无关性 | 架构保证（E_c 提供内容） | patch median 统计保证 | **两种都测** |

### 1.4 迁移到 TFS-Net 三源

TFS-Net 退化模型（`v5-design.md` §1.1）：`y_t = γ_t · (x_t * k_t + n_t)`

对应 DPMambaIR Eq.5 的 `I(x) = Q(α(x)J(x) + β(x)γ(x)) ⊗ K`：

| DPMambaIR 项 | TFS-Net 对应 | 物理本质 | 低维参数 | 局部可估计性 |
|:---|:---|:---|:---|:---|
| α(x) 光照 | γ_t 光照衰减 | 空间平滑乘性场 | `[α, γ_gamma]` | ✅ 空间平滑→局部均值 |
| η/γ(x) 传输噪声 | n_t 传感噪声 | i.i.d. 加性 | `[σ_s, σ_r]` | ✅ i.i.d.→局部方差 |
| K 模糊核 | k_t 运动模糊 | 方向性卷积 | `[θ, L]` | ✅ 方向性→局部梯度方向 |
| Q 压缩 | — | TFS-Net 无 | — | — |

**核心假设 H1**（DPMambaIR 重建解耦推广）：三源退化参数可被重建解耦的退化嵌入 E_d 编码（E_d 从退化图提取，E_c 从 GT 提取，R(E_c, E_d) 重建退化图），且 E_d 与内容嵌入 E_c 在特征空间中可分——即 E_d 捕获纯退化信息，独立于图像内容（DPMambaIR line 439）。

**核心假设 H2**（Condformer 局部可估计性推广）：三源参数都是**局部可估计的低维条件**（γ 空间平滑→局部均值可估，n i.i.d.→局部方差可估，k 方向性→局部梯度方向可估），与内容（空间高频全向结构）在"可估计性 + 频域签名"两个维度上分离。

**核心假设 H3**（本模型特有）：TFS-Net 的 `PyramidEncoder` 训练后特征空间中，三源退化线性可分性强于随机初始化——证明 TFSI 的归纳偏置诱导了可分性。这是 DPMambaIR/Condformer 论证能否迁移到本模型的关键判据。

**DPMambaIR 与 Condformer 的同构性**：两者都把退化建模为低维条件（DPMambaIR: E_d 嵌入向量；Condformer: `[σ_s, σ_r]` 先验），内容建模为高维空间结构（DPMambaIR: SSM 隐状态；Condformer: Transformer 特征）。可分性论证同构：退化低维可估，内容高维不可由局部估出 → 子空间不交叠。本设计双轨并用，互为佐证。

---

## 二、三源退化合成方法（从 GT 受控合成）

### 2.1 合成原理

从 SDSD/DID 的 GT 干净帧 `I_clean` 出发，按退化模型 `y = γ·(B_k * I + n)` 的物理顺序合成。**不叠加在已退化图像上**（修正 `Experience/SDSD-Trid/generateTriD.py:93` 在 `lq` 上叠加的缺陷）。

### 2.2 三源参数化与 GT 标签

每个合成样本携带 5D 先验向量作为 GT 标签：

```python
prior_gt = [alpha, gamma, sigma_s, sigma_r, theta_k, L_k]  # 6D
# alpha ∈ [0.15, 0.6]: 光照线性缩放
# gamma ∈ [1.8, 4.0]: Gamma 暗化指数
# sigma_s ∈ [0, 0.3]: Poisson 散粒噪声强度 (Condformer 同范围, train.py:446)
# sigma_r ∈ [0, 0.2]: Gaussian 读出噪声强度 (Condformer 同范围, train.py:447)
# theta_k ∈ [0, 180): 运动方向角度
# L_k ∈ [11, 25]: 运动核长度
```

参数范围与 Condformer `train.py:446-447` 的 `[σ_s∈0~0.3, σ_r∈0~0.2]` 对齐，保证噪声合成与公信方法一致。

### 2.3 合成函数

```python
def synth_three_source(I_clean, alpha, gamma, sigma_s, sigma_r, theta_k, L_k):
    """
    物理顺序: B_k(I) → +n → ·γ
    与 v5-design.md §1.1 退化模型 y = γ·(x*k + n) 一致
    """
    # Step 1: 运动模糊 (曝光期间运动)
    kernel = make_motion_kernel(L_k, theta_k)  # 方向核
    I_blur = cv2.filter2D(I_clean, -1, kernel)
    I_blur = cv2.filter2D(I_blur, -1, kernel)  # 二次卷积 (增强拖影)

    # Step 2: 传感噪声 (读出阶段) — Poisson-Gaussian, 同 Condformer train.py:449-451
    I_float = I_blur.astype(np.float32) / 255.0
    noise = np.random.normal(0, np.sqrt((sigma_s**2) * I_float + sigma_r**2))
    I_noisy = I_float + noise

    # Step 3: 光照衰减 (ISP 后处理)
    I_illum = alpha * (np.clip(I_noisy, 0, 1) ** gamma)
    return np.clip(I_illum * 255, 0, 255).astype(np.uint8)
```

**噪声合成与 Condformer 完全一致**（`train.py:449-451`：`img_noisy = img_clean + normal(0, sqrt(σ_s²·img + σ_r²))`），保证公信力。

### 2.4 五类样本设计

| 类别 | id | 合成方式 | GT 先验 | 用途 |
|:---|:---|:---|:---|:---|
| Clean | 0 | `I_clean` 原图 | 全 0 | 基线 |
| Illum_Only | 1 | 仅 `α·(I)^γ`，无噪声无模糊 | `[α,γ,0,0,0,0]` | 光照可分性 |
| Noise_Only | 2 | 仅 `I + n`，无光照无模糊 | `[1,1,σ_s,σ_r,0,0]` | 噪声可分性 |
| Motion_Only | 3 | 仅 `B_k * I`，无噪声无光照 | `[1,1,0,0,θ,L]` | 运动可分性 |
| All_Three | 4 | `γ·(B_k*I + n)` 完整退化 | `[α,γ,σ_s,σ_r,θ,L]` | 混合可分性 |

每类 N=300 张，从 SDSD GT 采样（DID 同理作泛化验证）。

---

## 三、三源退化嵌入提取器（双轨设计）

### 3.1 设计动机：DPMambaIR 重建解耦 + Condformer 局部估计

两条路线分别证明可分性的不同方面：

| 路线 | 方法 | 证明什么 | 参考来源 |
|:---|:---|:---|:---|
| **A. 重建解耦** | E_d 从退化图，E_c 从 GT，R(E_c,E_d) 重建退化图 | E_d 编码纯退化，内容无关 | DPMambaIR Eq.6-7, line 439 |
| **B. 局部先验估计** | 从 32×32 patch 估计 6D 先验，8 patch median | 三源参数局部可估，与内容统计无关 | Condformer `model.py:58-93` |

两条路线互补：A 证明"嵌入不含内容"，B 证明"参数可从局部估计"。

### 3.2 路线 A：重建解耦退化提取器（DPMambaIR 风格）

#### 3.2.1 架构（复刻 DPMambaIR Fig.3 + Eq.6-7）

```python
class DegradationExtractor(nn.Module):
    """
    DPMambaIR 风格退化提取器 (论文 Fig.3b, Eq.6)
    
    结构: Central Difference Convolution (CDC) 多尺度 + 标准卷积
    - CDC 捕获梯度相关退化线索 (边缘/纹理模糊, 噪声纹理)
    - 多尺度 4 分支不同核大小, 感知不同频带退化
    - 标准卷积分支捕获全局亮度/光照分布
    
    输出: E_d ∈ R^{B × D_emb}  退化嵌入 (D_emb=512, 同 DPMambaIR Table VII 最优)
    """
    def __init__(self, in_ch=3, embed_dim=512):
        super().__init__()
        # CDC 多尺度分支 (4 个核大小: 3,5,7,9)
        self.cdc_branches = nn.ModuleList([
            CentralDiffConv(in_ch, 64, k) for k in [3, 5, 7, 9]
        ])
        # 标准卷积分支 (全局亮度)
        self.std_conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1), nn.GELU()
        )
        # 融合 + 编码
        self.fuse = nn.Sequential(
            nn.Conv2d(64 * 5, 128, 1, 1, 0), nn.GELU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
            nn.Linear(128, embed_dim)
        )
    
    def forward(self, I_deg):
        feats = [branch(I_deg) for branch in self.cdc_branches]
        feats.append(self.std_conv(I_deg))
        return self.fuse(torch.cat(feats, dim=1))  # (B, D_emb)


class ContentExtractor(nn.Module):
    """
    DPMambaIR Eq.7: E_c = Restormer(O), 参数冻结
    本设计用本模型的 PyramidEncoder 替代 Restormer (因为 TFS-Net 已有)
    参数冻结, 只提供内容嵌入
    """
    def __init__(self, frozen_encoder):
        super().__init__()
        self.encoder = frozen_encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
    
    def forward(self, I_clean):
        with torch.no_grad():
            feat = self.encoder.forward_single(I_clean)
            return F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, 64)


class DegradationReconstructor(nn.Module):
    """
    DPMambaIR Eq.7: I^D_hat = R(E_c, E_d)
    组合内容嵌入 + 退化嵌入, 重建退化图
    
    论证: 若 E_d 含内容, 则 R 冗余, L1 损失不驱动 E_d 学内容
          若 E_d 只含退化, 则 R 需要 E_d 补充退化差异, E_d 被迫学退化
    """
    def __init__(self, content_dim=64, embed_dim=512, out_ch=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(content_dim + embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256 * 8 * 8),
            nn.GELU(),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.GELU(),  # 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),   # 32
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),    # 64
            nn.ConvTranspose2d(32, out_ch, 4, 2, 1),           # 128
        )
    
    def forward(self, E_c, E_d):
        x = torch.cat([E_c, E_d], dim=1)
        x = self.decoder(x).view(-1, 256, 8, 8)
        return self.upsample(x)  # (B, 3, 128, 128) 重建的退化图
```

#### 3.2.2 训练（复刻 DPMambaIR §IV-A, line 550-560）

```python
def train_reconstruction_disentanglement(dataloader, extractor, content_enc, reconstructor):
    """
    DPMambaIR 预训练策略 (论文 §IV-A):
    - Content Extractor (Restormer) 参数冻结, 只优化 Extractor + Reconstructor
    - 100,000 iterations, batch=1, patch=256×256
    - AdamW, lr=1e-4 → 1e-6 cosine annealing
    - L1 + LPIPS 损失 (论文 Eq.后文 L = ||I^D - Î^D||_1 + λ·LPIPS)
    """
    optimizer = AdamW(
        list(extractor.parameters()) + list(reconstructor.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=100000, eta_min=1e-6)
    criterion_l1 = nn.L1Loss()
    
    for iter, (I_clean, I_deg, prior_gt) in enumerate(dataloader):
        # Eq.6: E_d = E(I^D)
        E_d = extractor(I_deg)
        # Eq.7: E_c = Restormer(O) [冻结], Î^D = R(E_c, E_d)
        E_c = content_enc(I_clean)
        I_deg_hat = reconstructor(E_c, E_d)
        # 论文损失: L = ||I^D - Î^D||_1 + λ·LPIPS
        loss = criterion_l1(I_deg_hat, I_deg) + 0.1 * lpips_loss(I_deg_hat, I_deg)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
```

#### 3.2.3 关键判据（DPMambaIR Table VII 同款）

| 指标 | 计算 | DPMambaIR 最优值 | 本设计目标 |
|:---|:---|:---|:---|
| Rec. PSNR | 重建退化图 vs 真实退化图 | 25.67 dB (Demb=512) | >20 dB |
| Cls. Acc. | 冻结 extractor + MLP, 1000 iter | 92.0% (7类) | >85% (5类) |
| t-SNE 可分性 | E_d 降维可视化 | Fig.9 明显分簇 | 5 类明显分簇 |

### 3.3 路线 B：局部先验估计器（Condformer 风格）

#### 3.3.1 架构（复刻 LoNPE 结构，输出 6D）

```python
class ThreeSourcePriorEstimator(nn.Module):
    """
    Condformer LoNPE 的三源推广
    
    与 LoNPE (model.py:58-83) 的结构差异:
    - LoNPE: PlainCNN(3ch, patch=32, n_ch=32, max=256) → Linear(256→64→2)
    - TSPE:  PlainCNN(3ch, patch=32, n_ch=32, max=256) → Linear(256→64→6)
    唯一区别是输出维度 2→6 (三源 6 个参数)
    
    保持 patch_size=32, 与 LoNPE 一致, 保证局部可估计性假设相同
    """
    def __init__(self, num_conditions=6, patch_size=32):
        super().__init__()
        self.features = PlainCNN(n_colors=3, patch_size=patch_size,
                                 n_channels=32, max_chn=256)
        self.classifier = nn.Sequential(
            nn.Linear(self.features.out_channels, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, num_conditions)
        )

    def forward(self, x):
        y = self.features(x)
        y = F.adaptive_avg_pool2d(y, 1).flatten(1)
        return self.classifier(y)  # (B, 6) 三源先验
```

#### 3.3.2 训练（复刻 Condformer `train.py:436-506`）

```python
def train_tspe(dataloader, model, optimizer, epoch):
    """
    复刻 Condformer train_DF2K 的 LoNPE 训练逻辑:
    1. 随机裁剪 8 个 32×32 patch (train.py:458)
    2. TSPE 前向预测 6D 先验 (train.py:459)
    3. L1 loss 对 GT 先验 (train.py:460)
    4. median 聚合 8 个 patch 的预测 (train.py:469)
    
    GT 先验来自合成时的参数记录, 等价于 Condformer 的 img_sigma
    """
    criterion = nn.L1Loss()
    for img_clean, prior_gt in dataloader:
        # 随机裁剪 8 个 patch (同 Condformer)
        patches = random_cropping(img_noisy, patch_size=32, number=8)
        pred = model(patches)  # (8*B, 6)
        # GT 复制 8 份 (同 train.py:460)
        gt = prior_gt.repeat(8, 1)
        loss = criterion(pred, gt)
        (lambda_NE * loss).backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 3.4 路线 B 成功判据

| 指标 | 判据 | 含义 |
|:---|:---|:---|
| 先验估计误差 | 各维 MAE < 0.05 (归一化后) | 三源参数可从局部 patch 准确估计 |
| 内容无关性 | 同一图不同 patch 估计方差 < 跨图方差 | 估计的是退化而非内容 |
| Per-source 准确率 | 各源参数 R² > 0.8 | 三源各自可估 |

**若 TSPE 训练成功**：三源在先验空间可分离（Condformer 论证成立）。**若失败**：某源不可局部估计（可能是 γ 空间变化太大或 k 方向不明显），需分析是哪源失败。

### 3.5 双轨对照判据

| 判据组合 | 含义 |
|:---|:---|
| A 成功 + B 成功 | **强可分**：退化嵌入不含内容（A），且参数局部可估（B） |
| A 成功 + B 失败 | 退化嵌入可分但参数不可局部估——可能三源是全局耦合的（如 γ 改变噪声分布） |
| A 失败 + B 成功 | 参数可估但嵌入含内容——重建解耦不充分，需增大 D_emb 或改进 R |
| A 失败 + B 失败 | **不可分**：三源在当前建模下无法从单帧分离，需多帧时序信号 |

**DPMambaIR 的关键发现（line 778-781）对本设计的启示**：论文证明"JPEG 和 noise 共享相同内容但 t-SNE 可分性由退化信息驱动"——这正好对应本设计的 `Noise_Only` vs `Motion_Only`（同一 GT 不同退化），若 t-SNE 可分则证明可分性来自退化而非内容。

### 3.6 内容解耦验证测试（DPMambaIR line 778-781 专项）

DPMambaIR 论文最关键的可分性证据之一（line 778-781）：

> "JPEG 和 noise 共享相同内容（合成时用同一 GT），但 t-SNE 可分性由退化信息而非图像内容驱动。"

这排除了"可分性来自内容差异"的循环论证。本设计复刻此测试：

```python
def content_disentanglement_test(extractor, synth_data):
    """
    DPMambaIR line 778-781 的复刻测试
    
    构造: 同一 GT, 不同退化类型
    若 E_d 可分 → 可分性来自退化, 不是内容
    
    对比: 不同 GT, 同一退化类型  
    若 E_d 不可分 → 印证 E_d 不编码内容
    """
    # 测试 1: same-content, different-degradation (应可分)
    same_gt = synth_data['GT'][0]  # 固定一张 GT
    illum = synth_illum(same_gt, ...)
    noise = synth_noise(same_gt, ...)
    motion = synth_motion(same_gt, ...)
    
    E_illum = extractor(illum)
    E_noise = extractor(noise)
    E_motion = extractor(motion)
    
    # 同内容三源 E_d 距离应大 (退化驱动)
    d_degradation = pairwise_dist([E_illum, E_noise, E_motion])
    
    # 测试 2: different-content, same-degradation (应不可分)
    same_illum_params = {...}
    E_diff_contents = [extractor(synth_illum(gt_i, **same_illum_params)) 
                       for gt_i in different_gts]
    
    # 不同内容同退化 E_d 距离应小 (内容不驱动)
    d_content = pairwise_dist(E_diff_contents)
    
    # 判据: d_degradation >> d_content → 可分性由退化驱动
    separability_ratio = d_degradation.mean() / d_content.mean()
    return separability_ratio  # > 2.0 → 退化驱动可分
```

**判据**：`separability_ratio > 2.0` → 可分性由退化驱动，非内容。这正是 DPMambaIR Fig.9 的核心论证。

### 3.7 物理可解释性分析（DPMambaIR Fig.5 迁移）

DPMambaIR Fig.5（line 462-510）展示了**物理可解释的参数分布**：

> "noise/jpeg → 较小 Δdp（高惯性低通滤波器，抑制高频方差）；blur/low-light → 较大 Δdp（高增益放大，增强弱信号）。"

这证明退化嵌入不仅是可分的，还携带**物理可解释的退化类型信息**。本设计将此迁移到 TFSI 的 `s_illum/s_noise` 输出：

```python
def physical_interpretability_analysis(tfsi_outputs, degradation_labels):
    """
    DPMambaIR Fig.5 的 TFS-Net 迁移
    
    DPMambaIR: Δdp 分布按退化类型分化 (noise→小, blur→大)
    TFS-Net:   s_illum/s_noise 分布应按退化类型分化
    
    预期:
    - Illum_Only → s_illum 高, s_noise 低
    - Noise_Only → s_illum 低, s_noise 高  
    - Motion_Only → s_illum 低, s_noise 低 (v5 移除 s_motion, 运动由 MRPN 隐式处理)
    - All_Three → s_illum 高, s_noise 高
    - Clean → 两者都低
    """
    import seaborn as sns
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, s_name in zip(axes, ['s_illum', 's_noise']):
        for label_id, label_name in enumerate(['Clean','Illum','Noise','Motion','AllThree']):
            mask = degradation_labels == label_id
            values = tfsi_outputs[s_name][mask].mean(dim=[1,2,3])  # 每样本空间均值
            sns.kdeplot(values, ax=ax, label=label_name, fill=True, alpha=0.3)
        ax.set_title(f'{s_name} distribution by degradation type')
        ax.set_xlabel(f'{s_name} (spatial mean)')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('fig_physical_interpretability.png', dpi=300)
```

**判据矩阵**（对应 DPMambaIR Fig.5 的物理可解释性）：

| 退化类型 | 预期 s_illum | 预期 s_noise | 物理依据 |
|:---|:---|:---|:---|
| Clean | 低 | 低 | 无退化 |
| Illum_Only | **高** | 低 | γ 衰减 → s_illum 诊断 |
| Noise_Only | 低 | **高** | n 加性 → s_noise 诊断 |
| Motion_Only | 低 | 低 | v5 移除 s_motion，运动由 MRPN 隐式处理 |
| All_Three | **高** | **高** | 多源叠加 |

**关键连接到 v5-design.md §4.10/§9.6**：若 TFSI 的 `s_illum/s_noise` 塌缩到 0（§9 实证），则此分析会显示所有退化类型的 s 分布重叠——**这本身就是诊断 TFSI 失效的直接证据**。反之，若分布按退化类型分化，则证明 TFSI 在可分性意义上是有效的。

**这是本设计相对 DPMambaIR 的增量**：DPMambaIR 只分析 Δdp（SSM 参数），本设计分析 TFSI 的强度图输出——直接对接 TFS-Net 的核心模块，验证其设计意图是否实现。

---

## 四、本模型特征空间可分性验证（核心实验）

### 4.1 实验组设计（4 组对照）

| 组 | Encoder | TFSI | 训练 | 证明什么 |
|:---|:---|:---|:---|:---|
| A | PyramidEncoder 随机初始化 | 无 | 无 | 基线：随机特征空间可分性 |
| B | PyramidEncoder 训练后 | **无**（消融） | SDSD | Encoder 自学是否诱导可分性 |
| C | PyramidEncoder 训练后 | **有** | SDSD | TFSI 归纳偏置是否增强可分性 |
| D | ImageNet 预训练 ResNet50 | — | ImageNet | 公信力对照（现有实验方法） |

**关键对比**：
- C vs A：训练后是否比随机更可分 → 证明训练诱导了可分性
- C vs B：有 TFSI 是否比无 TFSI 更可分 → **证明 TFSI 的归纳偏置作用**（这是把 DPMambaIR/Condformer 论证迁移到本模型的关键证据）
- C vs D：本模型特征是否比 ImageNet 特征更可分 → 证明领域特异性

### 4.2 特征提取（本模型 PyramidEncoder）

```python
class TFSNetFeatureExtractor:
    """
    提取本模型 PyramidEncoder 的多层特征
    
    与现有 Visual_diff_noises.py 用 ImageNet ResNet50 的区别:
    - 本模型: PyramidEncoder 3 级 [32,64,96], 随机初始化, SDSD 训练
    - 现有: ResNet50 4 级, ImageNet 预训练, 与退化无关
    
    PyramidEncoder 结构 (encoder.py:33-89):
    - stage1: (B,32,H,W)     → lateral1 → p1
    - stage2: (B,64,H/2,W/2) → lateral2 → p2  
    - stage3: (B,96,H/4,W/4) → lateral3 → p3
    - fused:  (B,64,H,W)     FPN 融合后
    
    提取点:
    - l1 (stage1 输出): 32ch, H×W, 低级视觉
    - l2 (stage2 输出): 64ch, H/2, 中级纹理  
    - l3 (stage3 输出): 96ch, H/4, 中级结构
    - fused (FPN 输出): 64ch, H×W, 多尺度融合
    - TFSI.F_fused (TFSI 融合特征): 64ch, H×W, 时频联合 (仅 C 组)
    """
    def __init__(self, checkpoint_path=None, use_tfsi=False):
        from models.modules.encoder import PyramidEncoder
        from models.modules.tfsi import TFSI
        
        self.encoder = PyramidEncoder(in_channels=3, level_channels=[32,64,96], fused_channels=64)
        self.tfsi = TFSI(channels=64, fused_channels=64) if use_tfsi else None
        
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(extract_encoder_state(ckpt))
            if use_tfsi and self.tfsi:
                self.tfsi.load_state_dict(extract_tfsi_state(ckpt))
    
    def extract(self, img_uint8, return_tfsi=False):
        """返回各层 GAP 后的特征向量 dict"""
        x = preprocess(img_uint8)  # (1,3,H,W) normalized
        l1 = self.encoder.stage1(x)
        l2 = self.encoder.stage2(l1)
        l3 = self.encoder.stage3(l2)
        fused = self.encoder.forward_single(x, return_coarse=False)
        
        feats = {
            'l1': gap(l1).flatten(),     # 32-dim
            'l2': gap(l2).flatten(),     # 64-dim
            'l3': gap(l3).flatten(),     # 96-dim
            'fused': gap(fused).flatten() # 64-dim
        }
        
        if return_tfsi and self.tfsi is not None:
            # TFSI 需要多帧, 这里用单帧复制成 T 帧
            x_t = x.unsqueeze(1).repeat(1, 5, 1, 1, 1)  # (1,5,3,H,W)
            enc_feats = self.encoder(x_t)  # (1,5,64,H,W)
            tfsi_out = self.tfsi(enc_feats)
            feats['tfsi_fused'] = gap(tfsi_out['F_fused']).flatten()  # 64-dim
            feats['s_illum'] = tfsi_out['s_illum'].mean().flatten()  # 1-dim
            feats['s_noise'] = tfsi_out['s_noise'].mean().flatten()   # 1-dim
        
        return feats
```

**重要**：TFSI 需要多帧输入（`tfsi.py:256-271`），但分离性实验用单帧。处理方式：单帧复制成 T=5 帧（模拟静止场景），此时时域方差 σ_t≈0、SNR→∞，TFSI 退化为纯频域分支 + 空间分支的静态特征。这恰好测试 TFSI 对**单帧退化签名**的敏感度，与时频联合分离假设一致。

### 4.3 残差特征 vs 直接特征

Condformer 用 LoNPE 估计先验，不用残差。但 `Visual_diff_noises.py:167` 用 `|F(deg)-F(GT)|` 残差特征。两种都测：

| 特征类型 | 计算 | 含义 |
|:---|:---|:---|
| 直接特征 `F(deg)` | Encoder 前向 | 退化图的特征（含内容+退化） |
| 残差特征 `|F(deg)-F(GT)|` | 差值绝对值 | 纯退化特征（去除内容） |

残差特征是更强的可分性测试（已去除内容干扰），直接特征是更现实的测试（无 GT 时只有这个）。

---

## 五、统计验证方法（有公信力）

### 5.1 主方法：线性探针分类（Linear Probe）

**公信力来源**：SimCLR (ICML 2020)、CLIP (ICML 2021)、BYOL (NeurIPS 2020) 评估表征质量的金标准。Condformer 隐含使用（LoNPE 的 Linear 分类头即线性探针）。

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict

# 任务 1: 5 类分类 (Clean/Illum/Noise/Motion/AllThree)
def linear_probe_classification(features, labels, n_folds=5, n_seeds=10):
    """
    线性探针 5-fold CV, 10 seeds
    如果线性分类器就能高精度分类 → 特征空间中三源线性可分
    """
    all_acc = []
    for seed in range(n_seeds):
        clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        acc = cross_val_score(clf, features, labels, cv=cv, scoring='accuracy')
        all_acc.extend(acc)
    return {'acc_mean': np.mean(all_acc), 'acc_std': np.std(all_acc)}

# 任务 2: 先验回归 (连续 6D 先验预测)
def linear_probe_regression(features, priors_gt, n_folds=5):
    """
    线性回归预测 6D 先验向量
    R² > 0.7 → 先验信息在线性可解码范围内
    这是比分类更强的判据: 分类只需类别边界, 回归需要数值准确
    """
    from sklearn.linear_model import Ridge
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2_per_dim = []
    for dim in range(6):
        scores = cross_val_score(Ridge(alpha=1.0), features, priors_gt[:, dim],
                                 cv=cv, scoring='r2')
        r2_per_dim.append(scores.mean())
    return r2_per_dim  # 6 个 R² 值
```

**成功判据**：
- 5 类 accuracy > 0.80 → 强可分；> 0.60 → 中等
- 先验回归 R² > 0.7 (各维) → 先验线性可解码
- C 组 > A 组 20+ 个百分点 → TFSI 诱导了可分性

### 5.2 统计显著性：置换检验

```python
def permutation_test(features, labels, n_permutations=1000):
    """
    置换检验: 验证线性探针 accuracy 显著高于随机
    neuroscience 标准方法 (Nichols & Holmes 2002)
    """
    clf = LogisticRegression(max_iter=2000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    real_acc = cross_val_score(clf, features, labels, cv=cv, scoring='accuracy').mean()
    
    null_accs = []
    for _ in range(n_permutations):
        shuffled = np.random.permutation(labels)
        null_accs.append(cross_val_score(clf, features, shuffled, cv=cv, 
                                          scoring='accuracy').mean())
    p_value = (np.sum(null_accs >= real_acc) + 1) / (n_permutations + 1)
    return {'real_accuracy': real_acc, 'p_value': p_value, 'null_dist': null_accs}
```

**判据**：`p < 0.001` → 三源可分性统计高度显著。

### 5.3 无监督聚类指标（高维计算，不在 2D t-SNE 上）

**修正 `Visual_diff_noises.py:316` 在 2D 上算 silhouette 的错误**：

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def cluster_validity(features, labels):
    # 在原始高维特征上计算, 不降维
    return {
        'silhouette': silhouette_score(features, labels),       # >0.3 好
        'calinski_harabasz': calinski_harabasz_score(features, labels),  # 越大越好
        'davies_bouldin': davies_bouldin_score(features, labels)  # <1.0 好
    }
```

### 5.4 两两 ROC-AUC（成对判别）

```python
def pairwise_auc(features, labels):
    """
    5x5 两两 AUC 矩阵
    关注: Noise vs Motion (最易混淆), AllThree vs 各单源
    """
    classes = np.unique(labels)
    n = len(classes)
    auc_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            mask = np.isin(labels, [classes[i], classes[j]])
            X, y = features[mask], (labels[mask] == classes[j]).astype(int)
            clf = LogisticRegression(max_iter=2000)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y, scores)
            auc_mat[i, j] = auc_mat[j, i] = auc
    return auc_mat
```

### 5.5 信息泄露修正

**修正 `Visual_diff_noises.py:291` SelectKBest 全数据 fit 的泄露**：

```python
from sklearn.pipeline import Pipeline
# 正确: Pipeline 内嵌, 每折独立 fit
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(k=50)),
    ('clf', LogisticRegression(max_iter=2000))
])
acc = cross_val_score(pipeline, features, labels, cv=5, scoring='accuracy')
```

---

## 六、可视化方法（论文级图表）

### 6.1 图表清单

| 图 | 方法 | 证明什么 | DPMambaIR 对应 | 公信力 |
|:---|:---|:---|:---|:---|
| Fig 1 | 退化嵌入 t-SNE 5 类散点（DPMambaIR Fig.9 复刻） | E_d 按退化类型分簇 | Fig.9 | 高 |
| Fig 2 | 内容解耦验证：同内容 vs 同退化的 E_d 距离对比 | 可分性由退化驱动非内容 | line 778-781 | 高 |
| Fig 3 | Rec.PSNR + Cls.Acc vs D_emb 曲线 | 嵌入质量随维度变化 | Table VII | 高 |
| Fig 4 | 4 组 linear probe accuracy 柱状图 | C>A, C>B, C vs D | — | 高 |
| Fig 5 | 先验回归 R² 雷达图（6 维） | 先验线性可解码 | — | 高 |
| Fig 6 | 混淆矩阵热力图（C 组，5 类） | 各类正确识别 | — | 高 |
| Fig 7 | 层级 accuracy 曲线（l1/l2/l3/fused/tfsi） | 可分性峰值层级 → 指导 TFSI 接入 | — | 高 |
| Fig 8 | TFSI s_illum/s_noise 分布按退化类型分化 | 物理可解释性 | Fig.5 (Δdp 分布) | 高 |
| Fig 9 | 置换检验 null 分布直方图 | 统计显著性 p<0.001 | — | 高 |
| Fig 10 | 两两 AUC 矩阵热力图 | Noise vs Motion 关键对 | — | 高 |
| Fig 11 | 4 组 t-SNE 并排（A/B/C/D） | 训练诱导的可分性可视化 | — | 中 |
| Fig 12 | DID 数据集重复 Fig 4-6 | 跨数据集泛化 | — | 高 |

### 6.2 关键图实现

```python
# Fig 2: 4 组 linear probe 对比
def plot_group_comparison(results, save_path):
    """
    results: {group_name: {layer: {'acc_mean':..., 'acc_std':...}}}
    4 组 × N 层的柱状图
    """
    groups = ['A(随机)', 'B(训练无TFSI)', 'C(训练有TFSI)', 'D(ImageNet ResNet)']
    layers = ['l1', 'l2', 'l3', 'fused', 'tfsi_fused']
    # 分组柱状图, 误差棒, chance level 线
    ...

# Fig 5: 层级 accuracy 曲线
def plot_layer_accuracy(results, save_path):
    """
    X: l1, l2, l3, fused, tfsi_fused
    Y: 5-fold CV accuracy ± std
    4 条曲线 (A/B/C/D)
    指导 TFSI 接入位置: 可分性峰值层
    """

# Fig 1: TSPE 先验估计散点
def plot_tspe_scatter(preds, gts, dim_names, save_path):
    """
    6 个子图, 每个一维先验
    X: GT 值, Y: TSPE 预测值
    理想: y=x 对角线
    每子图标注 R² 和 MAE
    """
```

---

## 七、完整实验流程

```
Phase 0: 数据合成 [~30 min]
  ├── 加载 SDSD GT (N=300) + DID GT (N=300)
  ├── 每张 GT 生成 5 类退化 (共 3000 张/数据集)
  ├── 记录每张的 6D GT 先验
  └── 保存 synth_sdsd/ 和 synth_did/

Phase 1A: 退化嵌入提取器预训练（DPMambaIR 重建解耦）[~4 hours]
  ├── DegradationExtractor (CDC 多尺度, D_emb=512)
  ├── ContentExtractor (冻结 PyramidEncoder)
  ├── DegradationReconstructor: R(E_c, E_d) → Î_D
  ├── L1 + LPIPS 损失, 100K iter, batch=1, patch=256
  ├── 评估: Rec.PSNR (>20dB), Cls.Acc (>85%, 冻结+MLP 1K iter)
  └── 保存 E_d 嵌入用于 t-SNE

Phase 1B: 局部先验估计器训练（Condformer LoNPE 推广）[~2 hours]
  ├── TSPE: PlainCNN(32×32 patch) → Linear(256→64→6)
  ├── L1 loss 对 6D GT 先验, 8 patch median 聚合
  ├── 100 epoch, batch=64
  └── 评估: 各维 MAE, R², 内容无关性

Phase 2: 可分性验证 [~1 hour]
  ├── 2A: E_d t-SNE 5 类散点 (DPMambaIR Fig.9 复刻)
  ├── 2B: 内容解耦验证 (同内容 vs 同退化距离比, §3.6)
  ├── 2C: 物理可解释性 (s_illum/s_noise 分布, §3.7)
  └── 2D: 双轨对照 (A成功+B成功 → 强可分)

Phase 3: 本模型特征空间 4 组对照 [~1.5 hours]
  ├── 组 A: 随机 PyramidEncoder
  ├── 组 B: 训练后 PyramidEncoder (无 TFSI 消融)
  ├── 组 C: 训练后 PyramidEncoder + TFSI (v5.5 ckpt)
  ├── 组 D: ImageNet ResNet50 (对照)
  ├── 每组每层提取 GAP 特征 (l1/l2/l3/fused/tfsi_fused)
  ├── 线性探针 5 类分类 (× 10 seeds)
  ├── 线性探针先验回归 (6 维 R²)
  ├── 置换检验 (n=1000, C 组 fused)
  ├── 聚类指标 + 两两 AUC
  └── 保存 features + results.json

Phase 4: 可视化 [~30 min]
  ├── 12 张图 (§6.1)
  └── 保存 figures/

Phase 5: DID 泛化验证 [~1 hour]
  ├── 在 synth_did 上重复 Phase 1A-3
  └── 对比 SDSD vs DID 结果

总计: ~10 hours
```

---

## 八、成功判据与风险应对

### 8.1 成功判据矩阵

| 指标 | 随机基线 | 可分 | 强可分 | DPMambaIR 参考值 | 公信力 |
|:---|:---|:---|:---|:---|:---|
| Rec. PSNR (重建解耦) | <10 dB | >15 dB | >20 dB | 25.67 dB (Table VII) | 高（DPMambaIR 同方法） |
| Cls. Acc. (冻结+MLP) | 20% | >70% | >85% | 92.0% (Table VII) | 高（DPMambaIR 同方法） |
| 内容解耦比 (§3.6) | ~1.0 | >1.5 | >2.0 | Fig.9 定性证据 | 高（DPMambaIR line 778-781） |
| TSPE 各维 R² | <0.1 | >0.6 | >0.8 | — | 高（Condformer 同方法） |
| 5 类 Linear Probe Acc (C组) | 0.20 | >0.60 | >0.80 | — | 高（金标准） |
| C vs A accuracy 提升 | 0 | >15pp | >30pp | — | 高（证明 TFSI 作用） |
| C vs B accuracy 提升 | 0 | >5pp | >15pp | — | 高（TFSI 归纳偏置） |
| Permutation p-value | — | <0.05 | <0.001 | — | 高 |
| Noise vs Motion AUC | 0.50 | >0.75 | >0.90 | — | 高（关键对） |
| s_illum/s_noise 分布分化 | 无分化 | 部分分化 | 明确分化 | Fig.5 Δdp 分化 | 高（DPMambaIR Fig.5 迁移） |

### 8.2 风险与应对

**风险 1: TSPE 某源估计失败（如运动）**
- 诊断：看 6 维 R² 哪维低
- 应对：增大运动核强度范围；或承认"运动需多帧时序信号，单帧 patch 不足以估计"——这恰好支持 TFS-Net 用时频联合而非纯空间的设计动机

**风险 2: 组 C 不比组 A 好（TFSI 没诱导可分性）**
- 诊断：检查 v5-design.md §9 的 `s_illum/s_noise 塌缩`问题——TFSI 输出全 0 时，组 C≈组 B
- 应对：用 §10.3 P0-2 的显式监督重训 TFSI（`L_illum_sup = L1(s_illum, clamp(1-L_t/L_ref,0,1))`），再测组 C
- **这本身就是有价值的发现**：证明 TFSI 当前实现未实现设计意图，需要显式监督

**风险 3: 组 D（ImageNet ResNet）比组 C（本模型）更可分**
- 含义：通用预训练特征比领域特化特征更可分
- 应对：这说明小数据集（SDSD 70 序列）不足以训出可分特征空间；可考虑用 DID + SDSD 联合训练，或迁移学习
- **这也是有价值的发现**：指出本模型需要更多数据或预训练

**风险 4: Noise vs Motion AUC < 0.7**
- 两者都是高频退化，ResNet/Encoder 可能混淆
- 应对：检查频域签名（`v5-design.md` §1.2：噪声=宽带相位扰动，运动=方向带状幅度衰减）——如果频域签名可分但空间特征不可分，说明需要频域分支（TFSI LFF），恰好是本模型设计动机

---

## 九、与现有实验的核心差异总结

| 维度 | `Visual_diff_noises.py` | 本设计 | DPMambaIR 依据 |
|:---|:---|:---|:---|
| 特征空间 | ImageNet 预训练 ResNet50（与退化无关） | **本模型 PyramidEncoder**（A/B/C/D 4 组对照） | — |
| 退化提取 | 无（直接用分类特征） | **DPMambaIR 重建解耦**（E_d 从退化图，E_c 从 GT） | Eq.6-7, line 439 |
| 提取器结构 | 无 | **CDC 多尺度 + 标准卷积** | Fig.3 |
| 训练目标 | 无 | **L1 + LPIPS 重建**（非分类） | line 444-446 |
| 内容解耦 | 无 | **同内容 vs 同退化距离比** | line 778-781 |
| 嵌入质量 | 无 | **Rec.PSNR + Cls.Acc 双指标** | Table VII |
| 物理可解释 | 无 | **s_illum/s_noise 分布按退化类型分化** | Fig.5 Δdp 分布 |
| 局部可估计性 | 无 | **TSPE（Condformer LoNPE 推广）** | — |
| 类别数 | 2（illum vs motion） | 5（clean + 3 纯源 + allthree） | — |
| 混合退化 | 未参与 | 明确参与 | — |
| 特征设计 | 手工特征（循环论证） | 无手工特征，纯学习 | — |
| 主指标 | 单次 silhouette（2D） | 线性探针 + 先验回归（高维） | Table VII Cls.Acc |
| 显著性 | 无 | 置换检验 p-value | — |
| 信息泄露 | SelectKBest 全数据 fit | Pipeline 修复 | — |
| 多 seed | 单 seed | 10 seeds 均值±标准差 | — |
| TFSI 作用 | 未测 | C vs B 对照 | — |
| 跨数据集 | 仅 SDSD-TriDeg | SDSD + DID | — |
| 噪声模型 | 简单叠加 | Poisson-Gaussian（同 Condformer） | — |
| t-SNE 验证 | 2 类手工特征 | **5 类学习嵌入**（DPMambaIR Fig.9 复刻） | Fig.9 |

---

## 十、实现优先级

```
P0 (核心证明 — DPMambaIR 路线):
  [1] 三源合成器 (§2.3, 从 GT 合成 5 类, 6D GT 先验)
  [2] 退化嵌入提取器预训练 (§3.2, DPMambaIR 重建解耦, CDC+多尺度)
  [3] Rec.PSNR + Cls.Acc 双指标评估 (§3.2.3, DPMambaIR Table VII)
  [4] E_d t-SNE 5 类散点 (DPMambaIR Fig.9 复刻)
  [5] 内容解耦验证测试 (§3.6, 同内容 vs 同退化距离比)

P0' (核心证明 — 本模型特征空间):
  [6] 4 组特征提取 (§4.1, A/B/C/D)
  [7] 线性探针分类 + 先验回归 (§5.1)
  [8] 置换检验 (§5.2)
  [9] TFSI s_illum/s_noise 物理可解释性分析 (§3.7, DPMambaIR Fig.5 迁移)

P1 (强补充 — Condformer 路线):
  [10] TSPE 局部先验估计器训练 (§3.3, Condformer LoNPE 推广)
  [11] 层级 accuracy 曲线 (§6.1 Fig 7)
  [12] 混淆矩阵 + 两两 AUC (§5.4)
  [13] 双轨对照判据 (§3.5, A+B 成功 → 强可分)

P2 (泛化):
  [14] DID 数据集重复实验
  [15] 4 组 t-SNE 并排 (§6.1 Fig 11)

P3 (可选):
  [16] DPMambaIR 风格动态参数调制测试
       (用 E_d 调制本模型 SSM/Conv 参数, 
        对比有无退化先验的恢复 PSNR, 验证 E_d 携带可分信息)
  [17] D_emb 维度消融 (§3.2.3, DPMambaIR Table VII 复刻,
       测 256/384/512/1024 的 Rec.PSNR + Cls.Acc 曲线)
```

---

## 十一、文件结构建议

```
experiments/three_source_separability/
├── configs/
│   ├── synth.yaml          # 合成参数
│   └── tspe.yaml           # TSPE 训练参数
├── data/
│   ├── synthesizer.py      # 三源合成器 (§2.3)
│   └── dataset.py          # 合成数据集 loader
├── models/
│   └── tspe.py             # ThreeSourcePriorEstimator (§3.2)
├── feature_extractors/
│   ├── tfsnet_extractor.py # 本模型 PyramidEncoder 特征 (§4.2)
│   └── resnet_extractor.py # ImageNet ResNet50 对照
├── analyzers/
│   ├── linear_probe.py     # 线性探针 (§5.1)
│   ├── permutation.py      # 置换检验 (§5.2)
│   ├── cluster_metrics.py  # 聚类指标 (§5.3)
│   └── pairwise_auc.py     # 两两 AUC (§5.4)
├── visualizers/
│   └── plots.py            # 10 张图 (§6)
├── scripts/
│   ├── run_synth.py
│   ├── run_tspe_train.py
│   ├── run_feature_extract.py
│   ├── run_analysis.py
│   └── run_all.py
└── results/
    ├── features/
    ├── metrics/
    └── figures/
```

---

## 十二、与 v5 设计的对接

本实验直接对接 `v5-design.md` 的多个未决问题：

| v5 未决问题 | 本实验回答 |
|:---|:---|
| §4.10 `s_illum/s_noise 塌缩` | 组 C vs B 测试 TFSI 是否真的诱导了可分性 |
| §9.6 "目标缺失+功能冗余" | TSPE 提供独立于 TFSI 的先验估计基线 |
| §10.3 P0-2 显式监督 | 若 C 组可分性弱，证明需要显式监督重训 |
| §1.2 频域退化签名 | Noise vs Motion AUC 测试频域签名是否可分 |
| §4.11.2 TFSI 诊断职能 | 先验回归 R² 测试 TFSI 输出是否对应真实退化参数 |

**最重要的一点**：本实验不是"用外部模型证明可分性"，而是"用本模型自己的特征空间证明可分性"——这才是 TFS-Net/MINS-Net 设计合理性的直接证据。
