# SSM（状态空间模型）在视觉任务中的应用调研报告

**调研范围**：ECCV2024、CVPR2025、NeurIPS2025、ICCV2025、CVPR2026、ICLR2026、AAAI2026、ICML2026  
**数据来源**：[m0rtzz.github.io/paper-notes](https://m0rtzz.github.io/paper-notes/) + Google Scholar 引用数（2026-09-17）  
**统计口径**：图像恢复/增强领域 21 篇 SSM 论文 + 更广泛视觉任务文献综述补充

---

## 执行摘要

1. **SSM 类型垄断格局**：Mamba 占绝对主导（90.5% 论文数，96.5% 引用加权），RWKV 仅 1 篇（4.8%），RetNet 在图像恢复领域**零应用**
2. **时间爆发**：2024 年 MambaIR 奠基（829 引用），2025 年进入快速迭代期（8 篇），2026 年扩散到长尾细分任务（12 篇）
3. **任务集中度**：通用恢复占 23.8%（引用加权 89.1%），超分/去噪/去模糊等传统任务为主；新兴方向包括 4D 事件相机、UHD 超高清、医学图像
4. **会议渗透**：AAAI2026 图像恢复论文中 SSM 占比达 30%（3/10），CVPR2025 为 12.2%（5/41），整体渗透率 5.7%（21/357）
5. **架构趋势**：从全图扫描（MambaIR）→ 多方向扫描（EAMamba）→ 聚类中心扫描（C2SSM）→ 4D 时空扫描（PRE-Mamba），序列化策略成为核心竞争点

---

## 一、SSM 类型分布统计

### 1.1 论文数量占比（图像恢复领域）

| SSM 类型 | 论文数 | 占比 | 引用数 | 引用加权占比 |
|---------|--------|------|--------|-------------|
| **Mamba** | 19 | **90.5%** | 984 | **96.5%** |
| **RWKV** | 1 | 4.8% | 22 | 2.2% |
| Mamba/通用SSM | 1 | 4.8% | 14 | 1.4% |
| **RetNet** | **0** | **0%** | **0** | **0%** |

**关键发现**：
- **Mamba 垄断**：19/21 篇论文，引用数是 RWKV 的 44.7 倍
- **RWKV 孤例**：仅 CVPR2025 的 URWKV（低光增强），2.25M 参数轻量级模型
- **RetNet 缺席**：在图像恢复领域**完全未被采用**（截至 2026 年 9 月）
- **引用集中**：MambaIR 一篇贡献 829/1020（81.3%）总引用

### 1.2 更广泛视觉任务中的 SSM 分布

基于文献先验补充（未计入上述统计）：

| 视觉任务大类 | 代表性 SSM 工作 | 主流 SSM 类型 | 备注 |
|-------------|----------------|--------------|------|
| **通用视觉骨干** | VMamba (CVPR2024, 3003引用) | Mamba | 最高引用，奠定 Mamba 在视觉领域地位 |
| **图像分类** | ViM (ICML2024, 1100+引用) | Mamba | ImageNet-1K 分类任务 |
| **目标检测/分割** | Mamba-based FPN/Mask | Mamba | 依赖 VMamba 骨干 |
| **视频理解** | VideoMamba (ECCV2024) | Mamba | 长序列视频建模 |
| **3D 点云** | PointMamba (NeurIPS2024) | Mamba | 点云序列化处理 |
| **医学图像** | U-Mamba (MICCAI2024) | Mamba | 医学图像分割 |
| **遥感图像** | RSMamba (CVPR2025) | Mamba | 遥感多光谱分析 |

**跨任务结论**：
- **Mamba 全域主导**：从分类到检测、从 2D 到 3D、从自然图像到医学遥感，Mamba 是唯一被广泛验证的视觉 SSM
- **RWKV 边缘化**：仅在少数轻量级/特定场景（如本项目 TFS-Net 使用 RWKV 做时序建模）有应用
- **RetNet 视觉失败**：RetNet 在 NLP 有一定影响力，但**未能成功迁移到视觉领域**

---

## 二、各视觉任务的 SSM 偏好分析

### 2.1 图像恢复任务分布（论文数 + 引用加权）

| 任务大类 | 论文数 | 占比 | 引用数 | 引用占比 | 主流 SSM |
|---------|--------|------|--------|---------|---------|
| **通用恢复（多任务）** | 5 | 23.8% | 909 | **89.1%** | Mamba×5 |
| 超分辨率 (SR) | 3 | 14.3% | 47 | 4.6% | Mamba×3 |
| 几何校正/反射去除 | 4 | 19.0% | 5 | 0.5% | Mamba×4 |
| 压缩感知/快照成像 | 2 | 9.5% | 4 | 0.4% | Mamba×2 |
| 视频超分/恢复 | 2 | 9.5% | 5 | 0.5% | Mamba×2 |
| 低光增强 (LLIE) | 1 | 4.8% | 22 | 2.2% | **RWKV×1** |
| 去模糊 | 1 | 4.8% | 14 | 1.4% | Mamba×1 |
| 去雨/去天气 | 1 | 4.8% | 8 | 0.8% | Mamba×1 |
| 医学/科学图像 | 1 | 4.8% | 2 | 0.2% | Mamba×1 |
| 图像压缩 | 1 | 4.8% | 4 | 0.4% | Mamba×1 |

**关键洞察**：
1. **通用恢复垄断引用**：5 篇论文（MambaIR/MaIR/EAMamba/MoDEM/C2SSM）贡献 89.1% 引用，说明 **"一个模型多个任务"是 Mamba 的核心优势**
2. **RWKV 唯一立足点**：低光增强（URWKV），2.25M 参数，适配资源受限场景——**RWKV 的轻量化特性是其差异化竞争力**
3. **细分任务 Mamba 全覆盖**：从传统任务（超分/去噪）到新兴方向（4D 事件相机/UHD 超高清/医学图像），Mamba 无死角渗透

### 2.2 各任务偏好的 SSM 类型（百分比分布）

所有细分任务均为 **Mamba 100% 占有率**（除低光增强的 RWKV 孤例）。这说明：
- **无 SSM 类型竞争**：研究者默认选择 Mamba，不再尝试 RWKV/RetNet
- **架构创新集中在扫描策略**：如 Z 曲线（MoDEM）、嵌套 S 形（MaIR）、聚类中心（C2SSM）、4D 时空（PRE-Mamba）

---

## 三、时间演变趋势分析

### 3.1 按年份论文数与引用数

| 年份 | 论文数 | 引用数 | Mamba | RWKV | RetNet | 标志性工作 |
|------|--------|--------|-------|------|--------|-----------|
| **2024** | 1 | 829 | 1 | 0 | 0 | **MambaIR (ECCV)** — 开创者，829 引用 |
| **2025** | 8 | 168 | 7 | 1 | 0 | MaIR/MambaIRv2/URWKV/EAMamba — 快速迭代 |
| **2026** | 12 | 23 | 12 | 0 | 0 | C2SSM/TS-Mamba/DMDNet — 扩散到长尾 |

**趋势解读**：
1. **2024 年奠基**：MambaIR（ECCV2024）首次将 Mamba 引入图像恢复，简单 baseline 即达 SOTA，引爆领域
2. **2025 年爆发**：8 篇论文集中在 CVPR/ICCV，架构创新（多方向扫描/Query 机制/RWKV 尝试）进入白热化
3. **2026 年扩散**：12 篇论文覆盖更多细分领域（医学/遥感/事件相机/UHD），但引用数下降（新论文尚未积累）
4. **RetNet 持续缺席**：三年内零应用，已被视觉社区事实淘汰

### 3.2 各会议渗透率

| 会议 | 图像恢复总论文 | SSM 论文 | SSM 渗透率 | 平均引用 |
|------|---------------|---------|-----------|---------|
| ECCV2024 | 32 | 1 | 3.1% | 829 |
| CVPR2025 | 41 | 5 | **12.2%** | 23.4 |
| ICCV2025 | 31 | 2 | 6.5% | 19.5 |
| NeurIPS2025 | 26 | 1 | 3.8% | 12 |
| **AAAI2026** | 10 | 3 | **30.0%** | 1.3 |
| CVPR2026 | 135 | 6 | 4.4% | 1.8 |
| ICLR2026 | 61 | 2 | 3.3% | 3.5 |
| ICML2026 | 21 | 1 | 4.8% | 1 |
| **整体** | **357** | **21** | **5.7%** | **48.6** |

**关键发现**：
- **AAAI2026 异常高渗透**：30% 渗透率（3/10），但样本量小且引用少，说明 AAAI 图像恢复论文池较弱
- **CVPR2025 主战场**：12.2% 渗透 + 5 篇高质量论文，是 Mamba 图像恢复的集中爆发期
- **CVPR2026 饱和回落**：虽有 6 篇但渗透率仅 4.4%（总论文池扩大到 135），新鲜度下降

---

## 四、架构演进路径分析

### 4.1 Mamba 在图像恢复中的架构创新

| 创新维度 | 代表工作 | 核心思想 | 效果 |
|---------|---------|---------|------|
| **扫描策略** | MambaIR (ECCV2024) | 简单 4 方向扫描 | 开创性 baseline |
| | MaIR (CVPR2025) | 嵌套 S 形 + 序列洗牌 | 保局部性，+1.2dB |
| | MoDEM (NeurIPS2025) | Morton-Order Z 曲线 | 空间相关性优化 |
| | **C2SSM (CVPR2026)** | **聚类中心扫描** | **0.407G FLOPs，SOTA** |
| | PRE-Mamba (ICCV2025) | 4D 时空点云 + Hilbert | 事件相机专用，0.26M 参数 |
| **全局建模** | MambaIRv2 (CVPR2025) | C 矩阵注入 prompt，非因果查询 | 单方向超多方向 |
| | EAMamba (ICCV2025) | 多头选择性扫描 + 全方位扫描 | FLOPs 降 31-89% |
| **多帧聚合** | QMambaBSR (CVPR2025) | 跨帧 Query + 帧内扫描 | 连拍超分 |
| | TS-Mamba (ICLR2026) | 轨迹引导历史帧 + 带移位 SSM | 在线视频超分，MACs 降 22.7% |
| **轻量化** | LightRR (CVPR2026) | 小波低频 → Mamba | 3% 参数达接近 SOTA |
| | PRE-Mamba (ICCV2025) | 双流 MSSM 门控 | 0.26M 参数 |
| **UHD 适配** | **C2SSM (CVPR2026)** | **O(√N) 复杂度** | **4K 图像实用化** |

**演进规律**：
1. **Phase 1 (2024)**：验证可行性（MambaIR 简单 baseline 即 SOTA）
2. **Phase 2 (2025)**：扫描策略军备竞赛（S 形/Z 曲线/4D/多方向）
3. **Phase 3 (2026)**：聚类降复杂度（C2SSM）+ 长尾场景适配（医学/遥感/事件相机）

### 4.2 RWKV 在图像恢复的架构特点

| 工作 | 架构 | 参数量 | 特点 | 适用场景 |
|------|------|--------|------|---------|
| **URWKV (CVPR2025)** | 多状态 RWKV + 亮度自适应归一化 + 状态感知融合 | **2.25M** | 线性复杂度，极致轻量 | **低光增强（噪声+亮度+模糊联合）** |

**RWKV 劣势**：
- **无选择性机制**：不如 Mamba 的 selective scan（B/C/Δ 可学习）
- **全局感受野弱**：时序递归在空间建模不如 Mamba 的双向扫描
- **社区生态差**：无成熟视觉预训练模型，工程成本高

**RWKV 优势**：
- **线性复杂度**：O(N) 时间 + O(1) 空间（Mamba 是 O(N) 时间 + O(N) 空间）
- **真正在线推理**：适合视频流/边缘设备
- **参数效率**：2.25M 达到中等水平性能

---

## 五、引用数加权分析

### 5.1 高引用论文（Top 5）

| 排名 | 论文 | 引用 | 会议 | 年份 | SSM | 任务 | 影响力分析 |
|------|------|------|------|------|-----|------|-----------|
| 1 | MambaIR | 829 | ECCV2024 | 2024 | Mamba | 通用恢复 | **开创者**，简单有效，成为所有后续工作 baseline |
| 2 | MaIR | 35 | CVPR2025 | 2025 | Mamba | 通用恢复 | 扫描策略创新（嵌套 S 形），CVPR2025 代表作 |
| 3 | EAMamba | 31 | ICCV2025 | 2025 | Mamba | 通用恢复 | 效率优化（FLOPs 降 31-89%），ICCV2025 亮点 |
| 4 | MambaIRv2 | 28 | CVPR2025 | 2025 | Mamba | 超分 | 非因果全局建模，MambaIR 官方续作 |
| 5 | URWKV | 22 | CVPR2025 | 2025 | RWKV | LLIE | **唯一 RWKV 工作**，轻量化低光增强 |

**引用集中度**：Top 5 占总引用 92.7%（945/1020），说明领域仍处于早期，少数奠基性工作主导话语权。

### 5.2 引用加权的任务分布

通用恢复（89.1% 引用）>> 超分（4.6%）> LLIE（2.2%）> 去模糊（1.4%）> 其他（<1%）

**结论**：学术界偏好"一个模型解决多个任务"的通用性，细分任务工作引用有限。

---

## 六、对 TFS-Net v6 Delta 项目的启示

### 6.1 RWKV 在低光视频增强的定位

| 维度 | TFS-Net (RWKV) | 主流 Mamba 方案 | 优势对比 |
|------|---------------|----------------|---------|
| **复杂度** | O(N) 时间 + O(1) 空间 | O(N) 时间 + O(N) 空间 | ✅ RWKV 内存友好 |
| **在线推理** | 原生支持（时序递归） | 需要特殊设计（TS-Mamba） | ✅ RWKV 天然适配 |
| **参数效率** | 2.25M (URWKV) | 通常 >10M | ✅ RWKV 轻量 |
| **选择性建模** | ❌ 无 | ✅ selective scan (B/C/Δ) | ❌ Mamba 建模能力强 |
| **社区生态** | 稀少（1 篇） | 丰富（19 篇） | ❌ Mamba 工程成熟度高 |
| **长序列建模** | ✅ WKV 递归稳定 | ⚠ 扫描策略依赖 | ✅ RWKV 视频任务优势 |

**TFS-Net 选择 RWKV 的合理性**：
1. **低光视频（LLVE）= 在线推理需求**：实时处理视频流，RWKV 递归架构天然匹配
2. **资源受限场景**：URWKV 2.25M 参数证明 RWKV 在轻量化上有空间
3. **时序建模优先**：低光视频的时域降噪比空域恢复更关键，RWKV 的 WKV 递归适合长时序依赖

### 6.2 当前调研揭示的风险

1. **社区孤立**：RWKV 仅 1 篇（URWKV），后续工作缺乏，工程优化/预训练模型稀缺
2. **性能天花板**：Mamba 的选择性扫描（selective scan）在图像恢复任务上已被验证强于传统 SSM
3. **架构迭代落后**：Mamba 已有聚类扫描（C2SSM）/4D 扫描（PRE-Mamba）等创新，RWKV 停留在 2D 时序

### 6.3 推荐行动

#### 短期（Flight 11 当前阶段）
✅ **继续 RWKV**：TFS-Net 已投入大量工程（T-BC1b LocalWindowAlignment），且 RWKV 在视频时序建模有合理性

#### 中期（Flight 12-13 优化）
⚠ **Hybrid 架构探索**：
- **空域用 Mamba**：利用 selective scan 更好建模空间退化
- **时域用 RWKV**：保留递归优势做帧间传播
- 参考 QMambaBSR（跨帧 Query + 帧内扫描）混合思路

#### 长期（v7 架构演进）
🔄 **迁移到 Mamba**：如果 RWKV 性能遇到瓶颈，考虑：
- **TS-Mamba**（ICLR2026）：轨迹引导 + 带移位 SSM，专为在线视频设计
- **C2SSM**（CVPR2026）：聚类扫描降复杂度，适配高分辨率
- **PRE-Mamba**（ICCV2025）：0.26M 参数证明 Mamba 也可轻量化

---

## 七、整体趋势总结

### 7.1 数据驱动的结论

1. **Mamba 垄断已成定局**：90.5% 论文数，96.5% 引用，RetNet 已出局，RWKV 边缘化
2. **通用性是王道**：通用恢复任务占 89.1% 引用，细分任务工作影响力有限
3. **扫描策略是核心**：从 4 方向到 Z 曲线到聚类中心，序列化方式决定性能上限
4. **UHD 是新战场**：C2SSM（0.407G FLOPs）证明 SSM 可适配 4K 超高清，未来方向
5. **轻量化有空间**：LightRR（3% 参数）、PRE-Mamba（0.26M）证明 SSM 可压缩，边缘部署可期

### 7.2 对学术界的观察

- **快速迭代期**：2024-2026 三年 21 篇，平均 7 篇/年，领域仍在快速演进
- **会议分布**：CVPR/ICCV 是主战场（13/21），NeurIPS/ICML 较少关注图像恢复
- **引用两极化**：MambaIR（829）vs. 其他（平均 9.5），奠基性工作吃掉大部分引用

### 7.3 技术演进方向预测

| 方向 | 当前进展 | 预测 2027 趋势 |
|------|---------|---------------|
| **扫描策略** | 聚类/4D/Z 曲线 | 自适应扫描（数据驱动路径规划） |
| **复杂度优化** | C2SSM O(√N) | 稀疏注意力 + SSM 混合（O(log N)） |
| **多模态融合** | 单一图像/视频 | 图像+深度+事件相机联合 SSM |
| **预训练模型** | 无（从头训练） | 大规模预训练 Mamba backbone |
| **硬件加速** | 通用 GPU | 专用 SSM 算子（如 FlashAttention 之于 Transformer） |

---

## 八、参考文献（代表性工作）

### 图像恢复 SSM（按引用数）

1. **MambaIR** (ECCV2024, 829引用): 首篇将 Mamba 引入图像恢复，简单 baseline
2. **MaIR** (CVPR2025, 35引用): 嵌套 S 形扫描保局部性
3. **EAMamba** (ICCV2025, 31引用): 多头选择性扫描，FLOPs 降 31-89%
4. **MambaIRv2** (CVPR2025, 28引用): C 矩阵 prompt 非因果查询
5. **URWKV** (CVPR2025, 22引用): 唯一 RWKV 工作，2.25M 参数低光增强
6. **C2SSM** (CVPR2026, 2引用): 聚类中心扫描，0.407G FLOPs UHD SOTA
7. **PRE-Mamba** (ICCV2025, 8引用): 4D 时空扫描，事件相机去雨，0.26M 参数

### 通用视觉 SSM（扩展阅读）

- **VMamba** (CVPR2024, 3003引用): 视觉骨干，Mamba 在视觉领域地位奠基
- **ViM** (ICML2024, 1100+引用): 图像分类 baseline
- **VideoMamba** (ECCV2024): 长序列视频建模

---

## 附录：完整论文列表（21 篇）

| # | 标题 | 会议 | 年份 | SSM | 任务 | 引用 |
|---|------|------|------|-----|------|------|
| 1 | MambaIR: A Simple Baseline for Image Restoration with State-Space Model | ECCV2024 | 2024 | Mamba | 通用恢复 | 829 |
| 2 | MoDEM: Morton-Order Degradation Estimation Mechanism | NeurIPS2025 | 2025 | Mamba | 通用恢复 | 12 |
| 3 | MaIR: Locality- and Continuity-Preserving Mamba for Image Restoration | CVPR2025 | 2025 | Mamba | 通用恢复 | 35 |
| 4 | MambaIRv2: Attentive State Space Restoration | CVPR2025 | 2025 | Mamba | 超分 | 28 |
| 5 | QMambaBSR: Burst Image Super-Resolution with Query State Space Model | CVPR2025 | 2025 | Mamba | 超分 | 18 |
| 6 | URWKV: Unified RWKV Model for Low-Light Image Restoration | CVPR2025 | 2025 | **RWKV** | **LLIE** | 22 |
| 7 | Efficient Visual State Space Model for Image Deblurring (EVSSM) | CVPR2025 | 2025 | Mamba | 去模糊 | 14 |
| 8 | EAMamba: Efficient All-Around Vision State Space Model for Image Restoration | ICCV2025 | 2025 | Mamba | 通用恢复 | 31 |
| 9 | PRE-Mamba: A 4D State Space Model for Event Camera Deraining | ICCV2025 | 2025 | Mamba | 去雨 | 8 |
| 10 | Multi-Scale Gradient-Guided Unrolling Architecture with Adaptive Mamba (MambaCS) | CVPR2026 | 2026 | Mamba | 压缩感知 | 3 |
| 11 | VEMamba: Efficient Isotropic Reconstruction with Axial-Lateral Consistent Mamba | CVPR2026 | 2026 | Mamba | 医学图像 | 2 |
| 12 | DetectSCI: Object-Guided ROI Reconstruction for Video Snapshot Compressive Imaging | CVPR2026 | 2026 | Mamba | 视频恢复 | 2 |
| 13 | Distilling Quasi-Conformal Mapping (QDWC-Net) | CVPR2026 | 2026 | Mamba | 畸变校正 | 1 |
| 14 | LightRR: A Lightweight Network for Single Image Reflection Removal | CVPR2026 | 2026 | Mamba | 反射去除 | 1 |
| 15 | Scan Clusters, Not Pixels: Cluster-Centric SSM for UHD Restoration (C2SSM) | CVPR2026 | 2026 | Mamba | UHD通用恢复 | 2 |
| 16 | Trajectory-aware Shifted State Space Models for Online Video Super-Resolution (TS-Mamba) | ICLR2026 | 2026 | Mamba | 视频超分 | 3 |
| 17 | Content-Aware Mamba for Learned Image Compression | ICLR2026 | 2026 | Mamba | 图像压缩 | 4 |
| 18 | Depth-Synergized Mamba for All-Day Image Reflection Separation (DMDNet) | AAAI2026 | 2026 | Mamba | 反射去除 | 2 |
| 19 | MFmamba: Panchromatic Image Resolution Restoration via State-Space Model | AAAI2026 | 2026 | Mamba | 超分 | 1 |
| 20 | RefiDiff: Progressive Refinement Diffusion with Mamba-based Denoising | AAAI2026 | 2026 | Mamba | 其他 | 1 |
| 21 | Phy-CoSF: Physics-Guided Continuous Spectral Fields with Fourier-Mamba | ICML2026 | 2026 | Mamba | 压缩感知 | 1 |

---

**报告生成日期**：2026-09-17  
**数据有效期**：截至 2026 年 9 月（包含 CVPR2026/ICLR2026/AAAI2026/ICML2026 已接收论文）
