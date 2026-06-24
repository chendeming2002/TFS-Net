# TFS-Net v5.5 三源退化特征空间解耦可视化实验报告

> **日期**：2026-06-24
> **权重**：`outputs/sdsd_stage2/best.pth`（v5.5，epoch 16，PSNR=19.23dB）
> **数据**：SDSD indoor GT，从干净帧合成 5 类退化（Clean / Illum_Only / Noise_Only / Motion_Only / All_Three）
> **方法**：CPU 推理，3 帧窗口，128×128，每类 10 张样本
> **脚本**：`diagnose.py`（诊断）、`visualize_separability.py`（特征提取+t-SNE）
> **原始特征**：`features.npz`

---

## 一、实验目的

验证 TFS-Net v5.5 的 PyramidEncoder 特征空间中，三源退化（光照 γ / 传感噪声 n / 运动模糊 k）是否可分离。若可分，则 TFSI 的归纳偏置成功诱导了退化-内容解耦；若不可分，则定位根因。

## 二、实验设置

### 2.1 三源退化合成（从 SDSD GT 出发）

| 类别 | id | 合成方式 | GT 先验 |
|:---|:---|:---|:---|
| Clean | 0 | GT 原图 | 无退化 |
| Illum_Only | 1 | `α·(I/255)^γ·255`，α∈[0.15,0.6], γ∈[1.8,4.0] | 仅光照 |
| Noise_Only | 2 | Poisson-Gaussian：`I + N(0,√(σ_s²·I+σ_r²))`，σ_s∈[0.05,0.3], σ_r∈[0.02,0.2] | 仅噪声 |
| Motion_Only | 3 | 方向核二次卷积，ksize∈[11,17], angle∈[0,180] | 仅运动 |
| All_Three | 4 | `γ·(B_k*I + n)`（物理顺序：motion→noise→illum） | 三源叠加 |

### 2.2 特征提取

- 模型：TFSNet v5.5（`level_channels=[32,64,96]`，`fused_channels=64`，`use_amp_enhance=False`）
- 输入：单帧复制成 T=3 帧（模拟静止场景）
- 提取点：Encoder 各层（l1/l2/l3/p3/p2/p1/fuse0/fused）、TFSI（F_fused/F_s/F_f/s_illum/s_noise）、三分支（f_illum/f_noise/f_motion）

### 2.3 计算资源

- **GPU**：RTX 4090，被 v59 训练占用 22GB/24GB，无法使用
- **推理**：纯 CPU（torch 2.0.0，fp32），每样本约 6 秒
- **内存**：62GB 总量，训练占用约 6GB，推理峰值约 2GB

---

## 三、核心发现：Encoder fuse 层特征死亡

### 3.1 逐层激活 norm 追踪（fig1_encoder_layer_death.png）

| 层 | norm | 说明 |
|:---|:---|:---|
| l1 (stage1) | ≈470 | 正常 |
| l2 (stage2) | ≈7890 | 值域增长 |
| l3 (stage3) | ≈173084 | 值域爆炸 |
| p3 (lateral3) | ≈151071 | 正常 |
| p2 (lateral2+up) | ≈302124 | 累加增长 |
| p1 (lateral1+up) | ≈604017 | 值域 ±1800，**无归一化** |
| fuse[0] (Conv+GELU) | ≈215万 | conv 输出 ±12000，GELU 后非零 |
| **fuse[1] (Conv+GELU)** | **= 0** | **conv 输出全负（min=-36259, max=-119）→ GELU(负)=0** |

**根因链**：
```
Encoder 无归一化 → lateral 累加导致 p1 值域爆炸（±1800）
→ fuse[0] conv 输出 ±12000 → fuse[1] conv 映射到全负空间
→ GELU(全负) = 0 → fused 特征恒为 0
```

### 3.2 fuse[0] 负值占比（fig2_fuse_neg_frac.png）

fuse[0] 输出中负值占比 > 50%，且 fuse[1] conv 将其进一步推到全负 → GELU 完全饱和。

### 3.3 fp64 验证

用 `model.double()` + fp64 输入测试，fused norm 仍为 0。排除精度问题，确认是**权重+架构导致的确定性死亡**。

---

## 四、三源强度图塌缩（fig3_intensity_collapse.png）

| 类别 | s_illum | s_noise |
|:---|:---|:---|
| Clean | 0.0001 | 0.0000 |
| Illum_Only | 0.0001 | 0.0000 |
| Noise_Only | 0.0001 | 0.0000 |
| Motion_Only | 0.0001 | 0.0000 |
| All_Three | 0.0001 | 0.0000 |

**所有 5 类退化的 s_illum/s_noise 完全相同**——TFSI 输出不随输入退化类型变化。

### 4.1 与 v5-design.md §9.6 的关系

§9.6 记载"s_illum/s_noise 塌缩到 0"，归因于"监督缺失+功能冗余"。本实验揭示了**更深层的根因**：

| 诊断层级 | §9.6 诊断 | 本实验诊断 |
|:---|:---|:---|
| 表象 | s_illum ≈ 0 | ✓ 一致 |
| 中层 | 通路权重非零但模型选择不走 | 通路权重非零 ✓ |
| **根因** | 监督缺失+功能冗余 | **Encoder fuse 层死亡 → 特征根本没流到 TFSI** |

§9.6 的诊断在"通路权重非零"上是正确的，但遗漏了**fuse 层死亡**这一更上游的瓶颈。即使有完美的监督，特征也流不过去。

---

## 五、三源分支特征无退化特异性（fig4_branch_norms.png）

| 类别 | f_illum norm | f_noise norm | f_motion norm |
|:---|:---|:---|:---|
| Clean | 702.13 | 11.22 | 98.04 |
| Illum_Only | 701.76 | 11.22 | 98.04 |
| Noise_Only | 701.85 | 11.22 | 98.04 |
| Motion_Only | 702.10 | 11.22 | 98.04 |
| All_Three | 701.76 | 11.22 | 98.04 |

**5 类退化下三分支特征 norm 完全相同**（差异 < 0.4）。分支只接收到"死亡的常数特征"，没有学习到退化特异性响应。

---

## 六、模型为何还能输出 PSNR=19.2？

IGRF 的输出 `res_t = img_s2 × lit_up_map + s_illum × corr_mag` 中：
- `img_s2` 来自 `image_center` 的 skip connection（不依赖 encoder fused）
- `lit_up_map` 由 IFPN 的 `I_t_down`（低分辨率原图）路径驱动（不依赖 encoder fused）
- `s_illum × corr_mag` 项因 s_illum≈0 而为 no-op

**模型退化为纯图像域提亮滤波器**，特征空间完全不参与恢复。这解释了为什么 PSNR 停在 19.2dB 基线带（v5-design.md §0.2）且无法突破。

---

## 七、对三源可分性实验的启示

### 7.1 用 v5.5 权重做 t-SNE 可分性可视化无意义

因为 encoder fused 特征恒为 0，所有退化类型的特征完全相同，t-SNE 会显示所有点重叠在一个位置。`visualize_separability.py` 已提取 `features.npz`，其中 `enc_fused` 和 `tfsi_fused` 全为 0 或常数，无法用于可分性分析。

### 7.2 三源可分性验证需要先修复 encoder

| 修复方向 | 具体措施 | 预期效果 |
|:---|:---|:---|
| **A. fuse 前加 LayerNorm** | `fuse: LN → Conv → GELU → LN → Conv → GELU` | 防止值域爆炸导致 GELU 饱和 |
| **B. lateral 后加值域裁剪** | `p1 = clamp(p1, -10, 10)` | 简单但有效，可能损失信息 |
| **C. 用 ResBlock 替代 ConvBlock** | fuse 层用残差连接，即使 GELU 饱和也有 skip | 保证特征流 |
| **D. 降低 lateral 累加幅度** | 用 `0.5×lateral + 0.5×up` 替代直接相加 | 减缓值域增长 |

### 7.3 建议的实验顺序

1. **修复 encoder fuse 层**（方向 A 最直接）
2. **重训 10-20 epoch**，确认 fused norm 非零
3. **用修复后权重重做本可视化**：预期 5 类退化在 t-SNE 上形成分离簇
4. **对比修复前后**：修复后可分性应显著提升，直接证明 encoder 是瓶颈

---

## 八、文件清单

| 文件 | 内容 |
|:---|:---|
| `fig1_encoder_layer_death.png` | Encoder 各层激活 norm（log scale），显示 fuse 层死亡 |
| `fig2_fuse_neg_frac.png` | fuse[0] 输出负值占比分布 |
| `fig3_intensity_collapse.png` | s_illum/s_noise 散点（5 类退化），显示塌缩 |
| `fig4_branch_norms.png` | 三源分支特征 norm 柱状图，显示无退化特异性 |
| `fig5_summary_table.png` | 汇总表格 |
| `features.npz` | 原始特征矩阵（50 样本 × 10 特征类型） |
| `diagnose.py` | 诊断可视化脚本 |
| `visualize_separability.py` | 特征提取 + t-SNE 脚本（因特征死亡未产出 t-SNE 图） |
| `experiment_report.md` | 本报告 |

---

## 九、结论

**v5.5 权重的 PyramidEncoder fuse 层存在确定性特征死亡**：lateral 累加导致值域爆炸 → fuse[1] conv 映射到全负空间 → GELU 饱和到 0。这使得：

1. **encoder fused 特征恒为 0**（所有输入、所有退化类型）
2. **TFSI 的 s_illum/s_noise 塌缩**（特征没流到 TFSI，与 §9.6 现象一致但根因更深）
3. **三分支特征无退化特异性**（只接收到死亡常数）
4. **模型仅靠图像域 skip connection 工作**（退化为纯提亮滤波器）
5. **三源可分性无法在当前特征空间验证**（特征空间已死亡）

**下一步**：修复 encoder fuse 层（加 LayerNorm 或 ResBlock），重训后重做可视化，才能验证三源可分性。
