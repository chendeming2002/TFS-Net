# LLIE/LLVE/图像恢复领域近期研究动向（2024-2026）

> 基于 CVPR2026(135篇)、ICCV2025(31篇)、NeurIPS2025(26篇) 图像恢复论文调研
> 所有关键代码经直接阅读，非望文生义
> 更新：2026-09-16

---

## 一、总体趋势综述

### 1.1 三大主流方向

**方向一：扩散模型 All-in-One（主流，约占 30%）**

- 将扩散模型从图像生成迁移到图像恢复，配合 ControlNet/LoRA/一致性模型
- 代表：Bi-Bridge（低光双向扩散桥）、ZeroIDIR（零参考扩散）、UniLDiff
- **共同特征**：训练解耦（去噪 + 条件），单步/少步推理
- **对 LLVE 的适配**：推理延迟大（10-50×），视频时序一致性未内置，但感知质量高

**方向二：极轻量化专用网络（4K-1M参数，移动端向）**

- 代表：MobileIE（4K参数，ICCV2025）、Multinex（45K参数）、UCAN
- **关键技术**：结构重参数化（MBRConv：5分支conv训练→单conv推理）、Hedgehog线性注意力
- **对 LLVE 的适配**：直接嵌入视频流水线的帧级预处理，但无时序建模

**方向三：结构化多尺度 Mamba/SSM（快速崛起，约占 15%）**

- 代表：C2SSM（聚类中心扫描）、PRE-Mamba（4D时空SSM）、MoDEM（Morton-SSM退化估计）
- **关键技术**：Mamba的O(N)状态空间替代O(N²)Transformer，配合多尺度设计
- **对 LLVE 的适配**：PRE-Mamba的双时间尺度（帧内微秒级+帧间宏观级）直接对应多帧融合需求

### 1.2 新兴设计哲学

**频域先验注入**（2025-2026爆发）
- SSGformer：Sobel高频边缘 + SVD低频退化纹理 → 退化感知空间分组
  - 实际实现：`sobel(x)` 计算边缘幅度`edge`+方向`theta`，`svd`做11×11大核平滑→提取低频，两路linear attention融合生成退化感知掩码
  - 不是简单的频带分割，而是**频谱特征作为空间分组依据**
- MoDEM：Morton Z-order curve编码保2D局部性→1D SSM序列，同时估计全局/局部退化先验

**物理先验回归**（反扩散模型的声音）
- CWNet（ICCV2025）：结构因果模型（SCM）将低光退化解构为因果因子（内容语义）+ 非因果因子（亮度/颜色退化），因果解耦后再处理，比纯数据驱动更可解释
- LASQ（NeurIPS2025）：基于自然图像亮度满足幂律分布的物理先验，MCMC采样生成层次化亮度算子，无监督训练

**退化感知路由/分组**（从All-in-One借鉴）
- 不再用"一个网络处理所有退化"，而是"按退化类型动态路由"
- SSGformer：按Sobel+SVD光谱特征做组内注意力+跨组注意力，避免干净区域和退化区域信息混合

**超越像素GT监督**（新兴）
- edit-aware loss（CVPR2026）：用可微ISP渲染后计算误差，解决"像素GT ≠ 感知质量"的错配
- IQPIR（CVPR2026）：预训练NR-IQA模型的质量先验作条件信号，直接对最终感知质量优化

---

## 二、LLIE 专项动向（图像低光增强）

### 2.1 当前最优方法技术指标（公开数据）

| 方法 | LOLv1 PSNR | 参数量 | 时序 | 代码 |
|------|:---:|:---:|:---:|:---:|
| MobileIE (ICCV25) | 23.62 | **4K** | ✗ | [GitHub](https://github.com/AVC2-UESTC/MobileIE) |
| Multinex (CVPR26) | 23.19 | 45K | ✗ | 推断有 |
| CWNet (ICCV25) | 不详 | 1.23M | ✗ | arXiv 2507.10689 |
| LASQ (NeurIPS25) | SOTA无监督 | 不详 | ✗ | [GitHub](https://github.com/XYLGroup/LASQ) |

### 2.2 MobileIE 核心技术细节（代码已读 `/tmp/MobileIE/`）

**MBRConv（多分支重参数化卷积）**：
```python
# 训练时 5个并行分支：5×5+1×1+3×3+竖条+横条，均附BN副本，合10路concat→1×1压缩
# 推理时 slim() 方法将BN折叠并用矩阵乘法合并所有权重到单个5×5卷积
x = torch.cat([x1,x2,x3,x4,x5, bn(x1),bn(x2),bn(x3),bn(x4),bn(x5)], dim=1)
out = conv_out(x)   # 1×1压缩到目标通道
```

**FST（Feature Self-Interaction）**：
```python
# 实际是二次项: (w1*x) * (w2*x) + bias
# 即 x²的可学习缩放，捕获高阶非线性，运行时零额外FLOPs（可与MBRConv合并）
return self.weight1 * x1 * self.weight2 * x1 + self.bias
```

**双路注意力（HDPA）**：
- 全局通道：`AdaptiveAvgPool2d(1) → Conv1×1 → Sigmoid` (SE风格)
- 局部空间：对通道注意力的最大响应取max_out，再做1×1空间注意力
- 两路相乘：`x4 = torch.mul(x2, x3) * x1` — 通道门控×空间门控×特征

**可借鉴点**：MBRConv的训练-推理分离策略可直接用于我们的 SharedEncoder 轻量化改造，推理时合并为单卷积无额外开销。

### 2.3 LLIE 技术演进脉络

```
2023: Retinex + 扩散（Diff-Retinex ICCV23）
2024: 纯数据驱动轻量网络（SNR-Aware等）
2025: 因果解耦（CWNet）/ 幂律先验无监督（LASQ）/ 多模态事件辅助
2026: 双向扩散桥（Bi-Bridge）/ 极轻量45K（Multinex）/ 零参考扩散（ZeroIDIR）
趋势: 物理先验+极轻量+无/弱监督，感知质量优先，扩散模型成本过高被质疑
```

---

## 三、LLVE 专项动向（低光视频增强）

### 3.1 公开可查方法（截至2026-09）

**直接 LLVE 方法**（按发表时间）：
| 方法 | 会议 | 帧数 | 对齐方式 | 时序一致性 |
|------|------|:---:|---------|----------|
| VLLVE/LLVE_STCD | IJCAI2025 | 2 | Cross-attention 隐式 | 双向监督 |
| STA-SUNet | ICIP2024 | 5 | PCD 金字塔DCN | ✓ |
| RetinexMCNet | ICCV2025 | 多帧记忆 | key-value记忆池 | ✓ |
| VSRELL | CVPR2026 | 7 | 光流Warp+光照感知DCN | ISFP时序传播 |
| DWTA-Net | 2025 | Stage1:5帧/Stage2:递归 | PCD+GMFlow | 像素域递归累积 |

### 3.2 LLVE 领域关键挑战（未解决的问题）

1. **帧间闪烁**：单帧处理忽略时序约束，帧间光照估计不一致导致 flicker
2. **暗区光流失效**：传统光流在极暗区域估计噪声大（噪声被误判为运动）
3. **噪声-运动解耦**：相机抖动和帧间噪声难以区分，易产生双重伪影
4. **长程时序建模**：短窗口（5帧）捕获的时序范围有限，但长窗口显存爆炸

### 3.3 2026新方向：跨帧对齐+稠密引导（DGAF-VSR策略）

从视频超分领域迁移：**在特征域做upsampled warp而非RGB域**
- 核心发现：高分辨率特征域warp比低分辨率更少丢失高频
- FTCM：整个U-Net的稠密时序引导（不只是encoder或decoder端）
- **迁移到LLVE**：将upscale换成brightness enhancement，warp策略直接复用

### 3.4 AFUNet 跨帧对齐展开框架（ICCV2025，代码：github.com/eezkni/AFUNet）

多曝光HDR→LLVE迁移策略：
- HQS展开：对齐子问题（窗口交叉注意力SAM）+ 融合子问题（通道注意力Transformer CFM）交替迭代T=4阶段
- 窗口交叉注意力：不需要显式光流，在局部窗口内做Q-K匹配完成软对齐
- **直接迁移**：将曝光度不同的帧替换为低光+时序邻帧，对齐子问题换成亮度自适应对齐

---

## 四、一般图像恢复趋势（对LLVE有参考价值）

### 4.1 结构化退化感知（最高价值）

**SSGformer核心：Sobel+SVD光谱先验驱动空间分组**（代码已读 `/tmp/SSGformer/`）

实际实现（`SSGformer_arch.py:523-606`）：
```python
# Sobel特征：边缘幅度edge + 方向theta
Kx = [[-1,0,1],[-2,0,2],[-1,0,1]]  # 标准Sobel
Ky = [[1,2,1],[0,0,0],[-1,-2,-1]]
edge = hypot(conv(x,Kx), conv(x,Ky))   # 高频边缘信号
theta = atan2(conv(x,Ky), conv(x,Kx)) / pi  # 方向信息

# SVD特征：大核(11×11)平滑→提取低频退化模式
svd_feat = conv11x11(img_svd_cat) + dwconv3x3(img_svd_cat)
# 两路Linear Attention融合→生成退化感知掩码
mask = conv_mask(cat(sobel_feat, svd_feat))
```

分组注意力机制（`FGA_C:162-290`）：
- 组内注意力（in_group_attention）：同一退化组内精细建模
- 跨组注意力（cross_group_attention）：组间余弦相似度驱动，高相似组互补融合

**对LLVE的价值**：将Sobel边缘图作为Branch-M的辅助输入（运动=高频+方向变化），SVD低频作为Branch-L的先验（光照=低频全局变化），天然对应TSDR分解。

### 4.2 超越像素监督

| 技术 | 实现 | 在LLVE的用法 |
|------|------|------------|
| edit-aware loss | 可微ISP渲染后计算L1/SSIM | 替换我们的L_final，考虑后处理链路 |
| IQA-guided | NR-IQA分数作条件信号 | 无GT真实低光视频上的训练正则 |
| 双向一致性（Bi-Bridge） | Low→Normal + Normal→Low同时训练 | 视频帧间一致性的无配对正则 |

### 4.3 轻量化注意力替代

**Hedgehog线性注意力（UCAN）**：
- 解决传统线性注意力秩坍缩的核心问题
- 保持O(N)复杂度的同时保留全rank能力
- **对RWKV的比较**：RWKV本身就是线性复杂度，但在空间2D扫描上的方向感知能力更强

**Morton-SSM（MoDEM）**：
- Z-order curve将2D展平为1D，保局部性
- 比水平/垂直扫描更好地保持像素邻域关系
- **直接可用**：替换我们TCA-RWKV中的4方向扫描策略

---

## 五、与我们任务（LLVE/Foxtrot）的直接关联总结

| 发现 | 来源 | 对Foxtrot的影响 | 优先级 |
|------|------|----------------|:---:|
| F1/F3特征被编码器计算但完全闲置 | 代码审计 | 直接影响结构有效性 | 🔴 高 |
| 所有分支在H/2处理，无skip | 代码审计 | 细节恢复瓶颈 | 🔴 高 |
| Branch-M flow从F2_seq计算（32×32=1024 token，极小） | 代码审计 | 光流分辨率不足 | 🟡 中 |
| MBRConv重参数化可用于Encoder轻量化 | MobileIE代码 | 推理加速潜力 | 🟢 低 |
| Sobel+SVD频谱先验可驱动三分支路由 | SSGformer代码 | 增强分支解耦 | 🟡 中 |
| AFUNet交叉注意力软对齐 | 论文调研 | 替代显式光流对齐 | 🟡 中 |
| Morton/4D-SSM时空建模 | PRE-Mamba | 时序建模更完整 | 🟢 低 |
| 双向一致性训练无需GT | Bi-Bridge | 真实数据半监督 | 🟢 低 |
