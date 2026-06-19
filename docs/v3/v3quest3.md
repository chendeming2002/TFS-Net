# TFS-Net v3 — Phase B 启动指令（给 Claude 的自包含提示）

> 本文件是发给 Claude（对话接口）的完整提示。所有必要代码片段均已内嵌，无需额外文件。
>
> **日期**：2026-06-12
> **前置文档**：v3answer.md, v3code3.md（你的 Phase A 答复），v3quest2.md（代码审查结果）

---

## 〇、用户决策确认

**R1-R10 全部接受你的方案。** 但代码审查发现 v3code3.md 中有 **3 个 P0 级阻塞错误** 和 **3 个 P1/P2 补充需求**，需在 Phase B 编码前修正。

### 用户对 B.1 的追加决策

1. **K=10**（非你默认的 16），**n_ang_freq=1**（非你默认的 0）— 与 FRBNet 原始论文保持一致
2. **角度调制公式修正**：应使用 `1 + λ * cos(n*θ)` 形式（FRBNet 原始方式，始终为正的小扰动），而非你代码中的纯 `cos(n*θ)`（会产生负值导致基函数符号反转）。λ 取 0.1（FRBNet 默认）。
3. **B.1 代码审查通过**（N1 r_hat 归一化 ✅、N2 共享 basis ✅），仅需调整上述默认值和角度调制公式后继续

### 用户对 B.2/B.3 的追加决策

1. **B.2 FrequencyBranch**：代码审查通过 ✅
2. **B.3 SACE**：发现 P0 级 offset reshape bug — `offset.view(B, G, K, 2, H, W)` 应为 `offset.view(B, G, 2, K, H, W).permute(0, 1, 3, 2, 4, 5)`（offset 通道内存布局是 [all_x, all_y]，非 [x0,y0,x1,y1,...]）
3. SACE 数据流（LFF→median→逐帧DCA+残差）确认正确，OffsetMaskHead 拼接输入保留，默认 use_optimized=True

### 用户对 B.4-B.7 的追加决策

1. **B.4 IFPN**: 代码审查通过 ✅（IllumExtract groups=4, n_fea_middle=32 合理）
2. **B.5 NDPN**: 代码审查通过 ✅（SNR 估计 + 双因素权重正确）
3. **B.6 MRPN**: 代码审查通过 ✅（残差降权 + 中心帧固定权重正确）
4. **B.7 TFSNet**: 发现 3 个 P0 接口错误：
   - PyramidEncoder 构造函数参数名错误（应用 `level_channels` 而非 `c1,c2,c3`）
   - PyramidEncoder.forward() 返回值当 dict 解析（实际返回 tuple）
   - IGRF 构造函数 `img_channels` 不存在（应为 `out_channels`）

### 用户对 B.7-B.9 的追加决策（第四轮审查）

1. **B.7 TFSNet 修正版**: 代码审查通过 ✅（PyramidEncoder/IGRF/TFSI API 全部正确）
   - 依赖提醒：`self.tfsi.freq_branch.lff` 需确保 FrequencyBranch 内部保存 `self.lff`
2. **B.8 TFSNetLoss**: 代码审查通过 ✅（FFT/边缘感知平滑/PerceptualLoss 复用均正确）
3. **B.9 config YAML**: ✅ 通过
4. **B.9 `__init__.py`**: ✅ 通过（modules/models/losses 三个文件）
5. **B.9 train.py**: ❌ P0 + 不完整
   - import 路径错误：`from datasets.sdsd` → 应为 `from datasets`
   - 代码截断：只到 import 区（约 L33），缺少完整函数实现
6. **B.9 infer.py**: ❌ 完全缺失（v3answer7 未包含）

---

## 一、P0 阻塞修正（3 项，必须在编码时纠正）

### N1: r_hat 归一化遗漏 — FRBNet 对频率坐标做了归一化到 [0,1]

你 v3code3 §6.2 的 `RadialBasisFilter.forward()` 直接用未归一化的 `r_grid`：
```python
# 你的写法（v3code3 §6.2）
phi = torch.exp(-((r_grid.unsqueeze(0) - self.mu.view(-1, 1, 1)) / bwh) ** 2)
```

**但 FRBNet 原始代码**（`frbnet_utils.py` L27-28）对 `r_hat` 做了归一化：
```python
r_hat = torch.sqrt(fx ** 2 + fy ** 2)
r_hat = r_hat / r_hat.max()  # normalize to 0..1  ← 关键！
```

**影响**：`mu = linspace(0, 1, K)` 范围 [0,1]，但 `r_grid = sqrt(fx²+fy²)` 值域约 [0, 0.707]（Nyquist 频率）。不匹配会导致 **高频端基函数永远无法激活**（mu=0.8~1.0 的基函数找不到对应的 r_grid 值）。

**修正要求**：在 LFF 的 forward 中加 `r_grid = r_grid / (r_grid.max() + 1e-8)` 归一化，或在 RBF forward 内部归一化。

---

### N2: RadialBasisFilter 共享 basis 被错误拆分

你 v3code3 §1.2 创建了两个独立 `RadialBasisFilter` 实例：
```python
self.lff_mag = RadialBasisFilter(K=K, n_ang_freq=n_ang_freq)   # 独立的 mu/log_bwh
self.lff_phase = RadialBasisFilter(K=K, n_ang_freq=n_ang_freq) # 独立的 mu/log_bwh
```

**但 FRBNet 原始代码**（`frbnet_utils.py` L7-47）是 **一个实例内共享 basis**：
```python
class RadialBasisFilter(nn.Module):
    def __init__(self, n_coeff, lamda):
        # 共享参数
        self.coeff_mag   = nn.Parameter(torch.zeros(n_coeff))
        self.coeff_phase = nn.Parameter(torch.zeros(n_coeff))
        self.raw_gate_mag = nn.Parameter(torch.ones(n_coeff))    
        self.raw_gate_phase = nn.Parameter(torch.ones(n_coeff))  
        mu = torch.linspace(0.0, 1.0, steps=n_coeff)
        self.register_buffer('mu', mu)
        self.log_bwh = nn.Parameter(torch.tensor(0.0))  # 共享带宽

    def forward(self, H, W, device, dtype):
        # 共享 basis 计算
        basis = torch.exp(-((r_hat.unsqueeze(0) - self.mu[:, None, None]) ** 2) / (2 * bwh ** 2))
        # 共享角度调制
        basis = basis * angular_mod.unsqueeze(0)
        # 仅 coeff 和 gate 独立
        diff_mag = (gate_mag * self.coeff_mag[:, None, None] * basis).sum(0, keepdim=True)
        diff_phase = (gate_phase * self.coeff_phase[:, None, None] * basis).sum(0, keepdim=True)
        return diff_mag, diff_phase  # 同时返回两个响应
```

**修正要求**：使用**一个 RBF 实例**同时输出 `diff_mag` 和 `diff_phase`，共享 `mu/log_bwh/angular_mod`。物理含义：同一组频带，不同的幅度/相位处理方式。

---

### N3: Retinexformer depth_conv 的 groups 参数不应修改

你 v3code3 §4.3 声称："depth_conv 的 groups 改为 n_fea_middle，纠正 v3answer 错误"。

**但 Retinexformer 原始代码**（`RetinexFormer_arch.py` L102-103）：
```python
self.depth_conv = nn.Conv2d(
    n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
```
原始代码用 `groups=n_fea_in`（默认 4）。这是**分组卷积**（31 通道分 4 组，每组约 8 通道），是有意设计，组间有信息交流。

**修正要求**：IllumExtract 中 `depth_conv` 保留 `groups=4`（与原始一致），**不要**改为 `groups=n_fea_middle`。

---

## 二、P1 补充需求（2 项）

### N4: DeformableCrossAttention 效率优化

你 v3code3 §3.3 的 `_deform_sample` 使用 `for g in range(G) for k in range(Ks)` 双重 Python 循环，G=4, Ks=9 → **36 次 grid_sample 调用/帧**。

**要求**：
1. 首版提供简单版（当前循环写法，确保正确性）
2. 同时提供**优化版**：将所有采样点拼接为一次 `grid_sample` 调用（将 G×Ks 维度合入 batch 维度），或使用 `torchvision.ops.deform_conv2d` C++ 后端
3. 用 `self.use_optimized = True/False` 开关切换

---

### N5: SACE 和 IFPN 主类完整 forward() 设计

v3code3 仅给出子模块代码（`DeformableCrossAttention` 和 `IllumExtract`），但**缺少主类的完整数据流**。

**要求**：在实现 SACE 和 IFPN 时，提供完整的 `forward()` 方法，包含：

**SACE 主类**：
- 输入：`feats (B,T,C,H,W)` + `lff_module`（共享 LFF 实例）
- 数据流：逐帧 LFF → 时域中位值 μ_t_clean → 逐帧 DeformableCrossAttention → 输出
- 输出 dict：`attn_maps`（list of offset/mask）、`mu_t_clean (B,C,H,W)`、`F_aligned_list`（list of (B,C,H,W)）

**IFPN 主类**：
- 输入：`I_t_down (B,3,H/4,W/4)` + `F_t_L (B,c3,H/4,W/4)` + `s_illum (B,1,H,W)` + `feats (B,T,C,H,W)` + `center_idx`
- 数据流：IllumExtract → sim() → L_ref → 强度调制 → 上采样
- 输出 dict：`f_illum_out (B,C_f,H,W)`

---

## 三、P2 补充需求（1 项）

### N6: 训练配置和脚本适配

请在 Phase B 的最后阶段提供：
1. **完整的 v3 版 `configs/sdsd_stage1.yaml`**（替换 v1 的 model/loss 配置）
2. **`train.py` 修改清单**（具体行号和代码，当前 train.py 使用 `MINSNet` 和 `MINSLoss`）
3. **`infer.py` 修改清单**
4. **`models/modules/__init__.py` 更新**（当前导出 v1 模块）

当前 v1 配置供参考：
```yaml
# configs/sdsd_stage1.yaml (当前 v1 版本)
seed: 42
output_dir: outputs/sdsd_stage1
dataset:
  train_input_root: F:/DatasetDL/SDSD/indoor/input
  train_target_root: F:/DatasetDL/SDSD/indoor/GT
  val_input_root: F:/DatasetDL/SDSD/test/low-light
  val_target_root: F:/DatasetDL/SDSD/test/GT
  window_size: 5
  crop_size: 256
  num_workers: 4
model:
  in_channels: 3
  level_channels: [32, 64, 96]
  fused_channels: 48
  mins_window_size: 8        # v1 专用，v3 不需要
train:
  batch_size: 2
  epochs: 200
  lr: 0.0002
  weight_decay: 0.0001
  amp: false
loss:
  lambda_pix: 1.0            # v1 专用
  lambda_ssim: 0.2           # v1 专用
  lambda_perc: 0.05
  lambda_tv: 0.01            # v1 专用
  perceptual_pretrained: false
```

---

## 四、现有代码库上下文

> 以下是你编码时需要了解的现有代码结构。**不可修改的文件**标注了 [锁定]。

### 4.1 项目文件结构

```
e:/TFS-Net/
├── models/
│   ├── __init__.py          # 导出 MINSNet + TFSNet
│   ├── tfs_net.py           # TFSNet 主网络骨架（你需补全）
│   ├── mins_net.py          # [锁定] v1 网络（保留向后兼容）
│   └── modules/
│       ├── __init__.py      # 导出所有模块
│       ├── blocks.py        # [锁定] 基础构建块（ConvBlock, ResBlock, LayerNorm2d, 等）
│       ├── encoder.py       # PyramidEncoder（已适配 return_coarse）
│       ├── tfsi.py          # TFSI 骨架（频域分支待补 LFF）
│       ├── igrf.py          # IGRF 已实现
│       ├── mins.py          # [锁定] v1 MINSBlock（待 Phase D 删除）
│       ├── ispn.py          # [锁定] v1 ISPN（待 Phase D 删除）
│       ├── mspn.py          # [锁定] v1 MSPN（待 Phase D 删除）
│       └── reconstruction.py # [锁定] v1 FinalReconstruction（待 Phase D 删除）
├── losses/
│   └── losses.py            # 现有 MINSLoss + PerceptualLoss（你需新增 TFSNetLoss）
├── train.py                 # 训练脚本（你需适配 model/loss import）
├── infer.py                 # 推理脚本（你需适配 model import）
├── configs/sdsd_stage1.yaml # 训练配置（你需更新为 v3 版本）
└── reference_repos/         # 参考论文代码（已在 Phase A 分析完毕）
```

### 4.2 关键接口约定

**TFSNet.forward()** 输入输出（`tfs_net.py` 已定义）：
```python
# 输入：x (B, T, C, H, W)，T=5 帧，中心帧 t=2
# 输出：dict { res_t, s_illum, s_noise, s_motion, delta, f_fused_igrf, tfsi_out }
```

**PyramidEncoder** 输出（`encoder.py` 已实现）：
```python
feats, coarse_feats = encoder(x, return_coarse=True)
# feats:        (B, T, C_f=48, H, W)  全分辨率融合特征
# coarse_feats: (B, T, c3=96, H/4, W/4)  最粗尺度特征
```

**TFSI** 输出（`tfsi.py` 已实现，频域分支待补）：
```python
tfsi_out = tfsi(feats)
# tfsi_out = { F_fused, F_s, F_f, mu_t, sigma_t, snr, s_illum, s_noise, s_motion }
# mu_t (B, C, H, W), sigma_t (B, C, H, W), snr (B, C, H, W)
# s_* (B, 1, H, W) ∈ [0,1]
```

**IGRF** 输入（`igrf.py` 已实现）：
```python
igrf_out = igrf(f_t_base, f_illum_out, f_noise_out, f_motion_out,
                s_illum, s_noise, s_motion, image_center)
# f_t_base = feats[:, center_idx]  (B, C_f=48, H, W)
# f_*_out: (B, C_f=48, H, W)
# s_*: (B, 1, H, W)
# image_center: (B, 3, H, W)
```

### 4.3 blocks.py 可复用工具 [锁定]

```python
ConvBlock(in_ch, out_ch, kernel_size=3, stride=1, padding=1, act=True)  # Conv2d + GELU
ResBlock(channels)                    # 残差块
LayerNorm2d(channels, eps=1e-6)       # 通道级 LayerNorm
safe_divide(x, y, eps=1e-6)           # 安全除法
pairwise_cosine_logits(center, neighbors)  # 全局余弦相似度（供 IFPN sim() 使用）
```

### 4.4 当前 losses.py 中可复用的部分

```python
class PerceptualLoss(nn.Module):  # VGG16 感知损失（已有，可复用）
    def __init__(self, pretrained=False)
    def forward(self, pred, target)  # → scalar

def ssim_map(x, y, window_size=11, sigma=1.5)  # SSIM 图（已有，可复用）
```

---

## 五、Phase B 执行清单

请按以下顺序实现，每完成一项给出完整代码 + shape 验证伪代码。

```
B.1: models/modules/lff.py
     - RadialBasisFilter（修正 N1: r_hat 归一化, N2: 共享 basis）
     - LFFFeatureAdapter（修正 N2: 单实例输出 mag+phase）
     - 验证: (2, 48, 64, 64) → (2, 48, 64, 64)

B.2: models/modules/tfsi.py 补全
     - 将 FrequencyBranch 替换为 LFFFeatureAdapter
     - 验证: TFSI 整体前向 shape 正确

B.3: models/modules/sace.py
     - DeformableCrossAttention（修正 N4: 提供优化版+简单版）
     - SACE 主类（修正 N5: 完整 forward 数据流）
     - 验证: 5 帧输入 → attn_maps + mu_t_clean

B.4: models/modules/ifpn.py
     - IllumExtract（修正 N3: groups=4 不改）
     - IFPN 主类（修正 N5: 完整 forward 数据流）
     - 验证: 双流输入 → f_illum_out

B.5: models/modules/ndpn.py
     - SNR 自适应聚合
     - 验证: feats + attn_maps + snr → f_noise_out

B.6: models/modules/mrpn.py
     - 残差降权 + refine
     - 验证: feats + attn_maps → f_motion_out

B.7: models/tfs_net.py 整合
     - 取消 NotImplementedError，接入所有模块
     - 端到端验证: (2, 5, 3, 256, 256) → dict{res_t: (2, 3, 256, 256), ...}

B.8: losses/losses.py 新增 TFSNetLoss
     - 首版 use_temporal=False
     - 包含: L_recon(空间+频域), L_perc(复用现有), L_illum_smooth

B.9: 配置与脚本适配（N6）
     - configs/sdsd_stage1.yaml v3 版本
     - train.py 修改清单
     - infer.py 修改清单
     - models/modules/__init__.py 更新
```

---

## 六、约束条件（重申）

1. **不修改** `blocks.py`、`encoder.py`、`igrf.py`、`datasets/*`、`utils/*`
2. **不删除** v1 旧文件（mins.py, ispn.py, mspn.py, mins_net.py, reconstruction.py）
3. 所有 `forward()` 方法的中间张量必须标注 shape 注释
4. 每个模块文件头部 docstring 标注当前实现状态（✅/⚠️/❌）
5. TFSNet 总参数量目标 < 2M
6. 每模块实现后提供参数量统计

---

**请确认你理解了 N1-N3 的修正要求，然后从 B.1 开始实现。**
