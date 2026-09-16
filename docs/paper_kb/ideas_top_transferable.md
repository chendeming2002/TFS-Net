# 可迁移创新点 Idea 汇总（LLIE/LLVE 方向）

> 筛选标准：① 与我们的任务场景适配（视频/多帧/低光）② 创新时间近（2025-2026）③ 有明确落地路径
> 用途：作为后续改进的候选池，非下一轮必选
> 更新时间：2026-09-16

---

## 评级说明

- **S 级**：直接可用，与我们现有架构天然兼容，改动小
- **A 级**：需中等改造，但收益明确
- **B 级**：需大改造或与其他方向竞争，收益不确定

---

## S 级 Idea（直接可用）

### S1. 频谱先验驱动的三分支路由（SSGformer 迁移）

**来源**：SSGformer (ICCV2025) — 代码已读 `/tmp/SSGformer/basicsr/models/archs/SSGformer_arch.py:484-607`

**原始机制**：
```python
# Sobel 提取高频边缘（幅度+方向）
edge  = hypot(conv(x, Kx), conv(x, Ky))     # 高频信号
theta = atan2(conv(x, Ky), conv(x, Kx))     # 方向信号
# SVD 大核(11×11)提取低频退化模式
svd_low = conv11x11(x) + dwconv3x3(x)
# 两路 Linear Attention 融合 → 退化感知分组掩码
mask = conv_mask(cat(sobel_feat, svd_feat))
```

**迁移到 Foxtrot**：
- 我们当前的 TCA-RWKV 三路查询是**隐式学习**的（MLP 生成 Q_N/Q_L/Q_M），没有任何频域先验引导
- 可改为：用 Sobel 高频图引导 Q_M（运动=高频+方向变化），SVD 低频图引导 Q_L（光照=低频全局），残差引导 Q_N
- **收益**：把"结构化先验"从损失层（正交约束）提前到特征入口，可能比当前"事后正交"更有效

**改动量**：在 `tca_rwkv.py` 的 `_temporal_diff_context` / `_temporal_smooth_context` 中注入 Sobel/SVD 计算，约 30 行

**风险**：Sobel/SVD 在暗区噪声上会放大噪声，需要先做去噪预处理（我们的 var_map 可复用）

---

### S2. Morton/蛇形扫描替代 4 方向扫描（MoDEM 迁移）

**来源**：MoDEM (NeurIPS2025) — Morton Z-order curve

**原始洞察**：SSM 需要 1D 序列，但朴素展平破坏 2D 局部性。4 方向扫描（我们的方案）是对角线+水平+垂直，仍有跨行不连续。Morton 编码保证相邻 2D 点映射后距离小。

**当前 Foxtrot 的问题**（代码 `tca_rwkv.py:138-183`）：
```python
_scan_h   = flatten(2).transpose       # 水平
_scan_v   = permute.flatten             # 垂直
_scan_d1  = 对角主                     # 手动坐标索引
_scan_d2  = 对角副
```
4 方向已经不错，但 Morton 在理论上更保局部性，且可能减少 1 次扫描（省算力）。

**迁移**：替换 `RWKVSpatialHead` 的扫描策略，2 路（Morton + 逆Morton）替代 4 路

**改动量**：约 40 行（Morton 编码需要预计算索引表）

**收益**：算力↓50%（扫描次数减半），局部性↑

---

### S3. 特征域 upsample-warp-downsample 对齐（DGAF-VSR 核心）

**来源**：DGAF-VSR (CVPR2026)

**原始发现**：特征域跨帧相关性比像素域强；高分辨率 warp 比低分辨率少丢高频。

**当前 Foxtrot 的问题**（代码 `branch_m.py:_align_and_aggregate`）：
- 光流在 H/2 分辨率估计，warp 也在 H/2
- 对于 SDSD 的 ±1-4px 小运动，H/2 分辨率下仅 ±0.5-2px，量化误差大

**迁移方案**：
```python
# 当前: flow估计(H/2) → warp(H/2)
# 改为: flow估计(H/2) → 上采样flow到H → warp(H) → 下采样回H/2
flow_h2 = self.flow_estimator(center, neigh)      # H/2
flow_h  = F.interpolate(flow_h2, scale_factor=2)  # H
warped_h = self.deform_align(neigh_up, flow_h)    # 在H做warp
warped   = F.avg_pool2d(warped_h, 2)              # 回H/2
```
但注意：neigh 是 H/2 特征，若要 H 域 warp 需先上采样特征——这会增加算力。

**折中方案**：仅在最终聚合后做一次高分辨率 warp 校正

**改动量**：中等（涉及 branch_m 数据流重构）

---

## A 级 Idea（需中等改造）

### A1. 双时间尺度 SSM 时序建模（PRE-Mamba 迁移）

**来源**：PRE-Mamba (ICCV2025) — 4D event cloud + 双时间尺度

**核心思想**：
- 帧内时间尺度（微秒级）：捕获帧内运动模糊
- 帧间时间尺度（宏观）：捕获跨帧位移
- 两个尺度并行 SSM，STDF 模块让时间信息优先调制空间信息

**当前 Foxtrot 的对应**：
- 我们的 TCA 是单尺度（H/2）
- 帧内尺度可以用 Branch-M 内的局部处理
- 帧间尺度用现有 TCA

**迁移**：TCA 输出后加一个并行的"帧内细节 SSM"分支，专门处理单帧内的模糊/噪声

**创新点包装**："我们首次将双时间尺度 SSM 引入 LLVE，同时建模帧内运动模糊和帧间位移"

---

### A2. 因果解耦的退化建模（CWNet 迁移）

**来源**：CWNet (ICCV2025) — 结构因果模型

**核心思想**：
```
低光图 = f(因果因子: 场景语义, 非因果因子: 亮度/颜色退化)
```
用因果推断分离"内容"和"退化"，避免网络混淆二者。

**当前 Foxtrot 的对应**：
- 我们的三分支是**类型解耦**（噪声/光照/运动）
- CWNet 是**因果解耦**（内容/退化）
- 两者是正交的维度！可以叠加

**迁移**：在 Branch-L 中加入因果解耦头，输出（内容分量, 光照分量），用反事实损失约束

**创新点包装**："TSDR 类型解耦 × 因果解耦的双重解耦框架"

---

### A3. 双向一致性半监督训练（Bi-Bridge 迁移）

**来源**：Bi-Bridge (CVPR2026)

**核心思想**：训练时同时学 Low→Normal 和 Normal→Low，用对称扩散桥的端点对称性作隐式正则。

**当前 Foxtrot 的对应**：
- 我们有 SDSD 配对数据，可以**构造逆退化对**（GT → 合成低光）
- 加强约束：`Enhance(Low) ≈ GT` 且 `Degrade(GT) ≈ Low`（退化模型用简单 gamma+noise）

**迁移**：
```python
# 现有损失
L_final = recon(O_t, GT)
# 新增双向一致性
low_synth = degrade(GT)              # GT → 合成低光
L_cycle = recon(GT, enhance(low_synth))  # 循环一致性
```

**收益**：额外正则，尤其在没有真实配对数据的场景

**改动量**：小（loss.py 加一项 + 一个简单 degrade 函数）

---

## B 级 Idea（大改造，需评估）

### B1. 多模态事件辅助

**来源**：EIC-LIE、BiEvLight、RetinEV

**问题**：需要事件相机硬件，SDSD 数据集没有

**可能的替代**：用**合成事件流**（从帧间亮度差生成伪事件），作为时序运动的额外监督信号

**风险**：合成事件质量存疑

---

### B2. 扩散模型生成式增强

**来源**：Bi-Bridge、ZeroIDIR、MR. Illuminate

**问题**：
- 推理延迟 10-50×（5帧窗口视频不可接受）
- 时序一致性未内置
- SDSD 的 PSNR 导向与扩散的感知导向不完全对齐

**可能的轻量版**：单步一致性模型（如 ExpoCM 的思路），但仍需大量改造

---

### B3. 张量低秩分解时序压缩

**来源**：Self-Attention Driven Tensor Representation (CVPR2026)

**思想**：用低秩张量表示压缩时序特征，降低长窗口显存

**适用场景**：如果要扩展窗口从5帧到9帧/15帧

**风险**：当前5帧窗口显存不是瓶颈（10.4GB/24GB），暂不需要

---

## 六、组合创新建议（按创新强度排序）

### 组合1（最强）：频谱路由 + 双时间尺度 SSM + 因果解耦
```
L_final + λ_ortho·L_ortho + λ_freq·L_freq + λ_causal·L_causal
TCA-RWKV(三路查询←Sobel/SVD先验) + 帧内SSM分支 + 因果解耦头
```
**优势**：三个维度正交（频域×时间尺度×因果），论文故事完整
**劣势**：改动大，需要多次消融验证每一部分

### 组合2（性价比最高）：S1频谱路由 + S3高分辨率warp
```
在现有 Foxtrot 上做两个小改动：
  ① TCA 三路查询注入 Sobel/SVD 先验（30行）
  ② Branch-M 的 warp 提升到 H 分辨率（100行）
```
**优势**：改动小，直击我们当前的两个弱项（隐式查询 + 低分辨率对齐）
**预期**：pair45 可能 +0.5-1.0dB

### 组合3（稳健）：A3双向一致性 + S2 Morton扫描
```
低成本正则 + 算力优化，适合作为"工程改进"而非"创新点"
```

---

## 七、与本任务适配度的关键判断

| Idea | 输入适配 | 场景适配 | 算力适配 | 数据适配 | 综合 |
|------|:---:|:---:|:---:|:---:|:---:|
| S1 频谱路由 | ✓ RGB | ✓ | ✓ | ✓ | **强推** |
| S2 Morton扫描 | ✓ | ✓ | ✓↑ | ✓ | 推荐 |
| S3 高分辨率warp | ✓ | ✓ | △ | ✓ | **强推** |
| A1 双时间尺度 | ✓ | ✓ | △ | ✓ | 可选 |
| A2 因果解耦 | ✓ | ✓ | ✓ | △需改造 | 可选 |
| A3 双向一致性 | ✓ | ✓ | ✓ | ✓合成 | 推荐 |
| B1 事件 | ✗需硬件 | ✗ | — | ✗ | 不建议 |
| B2 扩散 | ✓ | △ | ✗ | △ | 不建议 |

---

## 八、立即可做的验证实验（低成本）

1. **S1 快速验证**：在 `_temporal_diff_context` 中加入 Sobel 计算，跑 10 epoch 看 pair45
2. **S3 快速验证**：Branch-M warp 分辨率翻倍，跑 10 epoch
3. **A3 快速验证**：loss 加 cycle 项，跑 10 epoch

三个实验可并行准备代码，串行训练验证（每个约 10h）。

---

## 附：论文代码仓库清单（已验证可克隆）

| 论文 | 仓库 | 状态 |
|------|------|:---:|
| MobileIE | github.com/AVC2-UESTC/MobileIE | ✅ 已克隆 |
| SSGformer | github.com/jeongyh98/SSGformer | ✅ 已克隆 |
| LASQ | github.com/XYLGroup/LASQ | ⟳ 待重试 |
| AFUNet | github.com/eezkni/AFUNet | ⟳ 待重试 |
| PRE-Mamba | github.com/softword-tt/PRE-Mamba | ⟳ 待重试 |
| Multinex | 需从 CVPR2026 页面查找 | ⏳ |
| DGAF-VSR | 需从 CVPR2026 页面查找 | ⏳ |

---

**说明**：本文档仅为 idea 整理，不代表下一轮改进方向。任何 idea 落地前必须先做 10-epoch 快速验证，避免重蹈 Flight11 的覆辙（未经小实验验证直接全量训练）。
