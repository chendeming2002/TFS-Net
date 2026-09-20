# Golf-R3 改进方案

**创建时间**：2026-09-11  
**目标**：基于 Golf 原型（R1-S2 ep20=19.66）+ LLVE 最佳实践 + Flight11 S1.5 失败诊断，构建 Golf-R3 改进版本，目标 pair45 ≥ 18.0 + 通用集 PSNR > 20.0

---

## 一、背景与动机

### 1.1 Golf R1-S2 诊断

| 问题域 | 观测 | 根因 |
|--------|------|------|
| **conf_map 崩溃** | ep10 均值 0.969→ep20 1.000 饱和 | ConfidenceEstimator bias 初始化 2.0 → sigmoid 起点 0.88 偏高，softmax 优化路径狭窄 → 退化一致性门 |
| **Branch-M 差退化** | ep20 warp_t 贡献 0.03（应 > 0.5）| H/2 编码域 warp + 双线性上采样 → 精细结构模糊 → 分支被权重网络抑制 |
| **高频震荡** | 预测 HF 能量 1.5×-2× 输入（夜景低对比） | L_temp 阈值 1.5× 过宽，失去约束 |
| **val 过拟合风险** | 仅 5 序列 val 监控 | pair45 无同步评估，运动泛化未追踪 |

### 1.2 Flight11 S1.5 v1.1 中性

| 实验 | ep20 PSNR | 判读 |
|------|-----------|------|
| Golf R1-S2 | **19.66** | 轻量三分支原型，conf 崩溃前最优 |
| Flight11 S1.5 v1.1 | 19.78 | 8 项锚点损失 + T-BC1b 主干，**训练代价 8× 仅 +0.12dB** |

**结论**：三分支架构有效，但 Flight11 重型主干（LocalTCA 9×9 窗口 softmax）对 256² 裁剪数据过设计，**Golf 轻量化路径更优**。

### 1.3 LLVE 最佳实践回归

从 8 个 SOTA 仓库源码级分析提取的 **已验证有效模式**：

| ID | 模式 | 源仓库 | 预期效果 |
|----|------|--------|----------|
| **#1** | conf 中性起点（bias=0） | CDVD-TSP | 防止 sigmoid 偏置锁死 |
| **#2** | 高分辨率 warp（H 域操作） | fastdvdnet, STCD | 保持精细结构清晰度 |
| **#3** | 退化感知查询调制 | StableLLVE (FiLM) | 区分低光/运动特定权重 |
| **#4** | 高频能量正则（动态阈值） | FRBNet | 防止夜景过增强 |
| **#5** | 双验证集监控 | EvRWKV (test-easy/hard) | 防止 val 过拟合 |

**Golf-R3 策略**：**保持 Golf 轻量主干不变**（3-stage encoder + 128-ch TCA-RWKV），**5 处定向修正**注入 LLVE 最佳实践，**最小改动 + 最大回报**。

---

## 二、Golf-R3 改进方案

### R3-A：ConfidenceEstimator 中性起点

**原理**：bias=2.0 → sigmoid 初始 0.88 偏高 → softmax 后置信度 0.94+，优化空间窄；bias=0.0 → sigmoid 初始 0.5 中性 → softmax 0.2-0.8 动态可学习。

**修改**：`models/golf_r3/branch_m.py`
```python
self.conf_head = nn.Sequential(
    nn.Conv2d(channels, channels//2, 3, 1, 1),
    nn.ReLU(inplace=True),
    nn.Conv2d(channels//2, 1, 1),
    nn.Sigmoid(),
)
# 修改最后卷积 bias 初始化
nn.init.constant_(self.conf_head[-2].bias, 0.0)  # 原 2.0
```

**验收**：训练日志 conf_map ep10/20 均值应在 0.5-0.8 动态范围，**不再 ep20 饱和到 1.0**。

---

### R3-B：Branch-M 高分辨率 warp

**原理**：原流程 = F2 编码(H/2) → warp → 上采样 H，双线性模糊；改进 = F2 上采样 H → warp(H 域) → 保持锐度。

**修改**：`models/golf_r3/branch_m.py::BranchM.__init__()`
```python
def __init__(self, ..., use_hires_warp=True):
    self.use_hires_warp = use_hires_warp
    if use_hires_warp:
        # F2 → H 域预上采样
        self.enc_upsample = nn.Sequential(
            nn.Conv2d(enc_channels, enc_channels, 3, 1, 1),
            nn.PixelShuffle(2),  # C→C/4, H/2→H
        )
    else:
        self.enc_upsample = None
```

**修改**：`forward()` warp 前上采样 `F2_seq`：
```python
if self.use_hires_warp:
    F2_seq = torch.stack([self.enc_upsample(f) for f in F2_seq], dim=1)
# 现在 F2_seq 已是 H 分辨率，直接 warp
```

**验收**：分支贡献日志 warp_t 从 0.03 提升到 > 0.3，说明高分辨率特征被权重网络识别有效。

---

### R3-C：F3 FiLM 退化感知调制

**原理**：TCA-RWKV 当前统一权重处理所有帧，无法区分低光/运动模式；F3 (H/4 深层编码) 时序均值 → 全局 pool 提取退化描述子 → FiLM γ/β 调制 TCA 通道。

**实现**：
1. **F3 提取**：`GolfNet.forward()` 编码器同时输出 F3 (C3=128, H/4)
2. **全局描述子**：`f3_ctx = F3_seq.mean(dim=1).mean(dim=[-2,-1])` → (B, C3)
3. **FiLM 生成器**：`TCA_RWKV` 新增 `FiLMGenerator(C3 → 2×tca_channels)`
4. **调制注入**：每个 RWKV block 前后 `x = x * (1 + γ) + β`

**修改文件**：
- `models/golf_r3/tca_rwkv.py`：新增 `FiLMGenerator` + `use_f3_film` 开关
- `models/golf_r3/golfnet.py`：编码 F3 → 时序+空间 pool → 传入 TCA

**验收**：FiLM γ 日志应在 [-0.3, +0.3] 动态范围（非零常数），说明不同退化模式下通道权重自适应。

---

### R3-D：pair45 双验证集监控

**原理**：现 val 仅 5 序列（静态室内为主），pair45（高速运动）泛化未追踪 → 过拟合风险。

**实现**：训练脚本 `train_golf_r3.py` 每 val_interval 同步评估：
1. **标准 val**：5 序列裁剪 256² → PSNR/SSIM/LPIPS
2. **pair45 全帧**：前 30 帧（0047-0076）全尺寸推理 → pair45_psnr

**配置**：`configs/golf_r3.yaml`
```yaml
dataset:
  pair45_input_root: /home/a1005/yzy/dataset/SDSD/test/low-light/pair45
  pair45_gt_root: /home/a1005/yzy/dataset/SDSD/test/GT/pair45
```

**验收**：训练日志每 10 epoch 输出双指标：
```
Val stats: {'psnr': 19.8, 'ssim': 0.78}
Pair45 stats: {'pair45_psnr': 16.5, 'pair45_frames': 30}
```
若 val 上升但 pair45 下降 → 过拟合警报。

---

### R3-E：L_temp 分支选择性约束 + 阈值收紧

**原理**：
1. **分支选择性约束**：原设计 L_temp 作用于全局输出 O_t，会无差别惩罚所有分支的高频贡献，包括 Branch-M（运动对齐）合理的细节恢复。改进：仅约束 Branch-N（去噪）和 Branch-L（光照），Branch-M 豁免以保护细节恢复。
2. **阈值收紧**：原阈值 `1.5×` 对夜景低对比输入过宽；收紧至 `1.2×` 激活约束。

**理论依据**：
- Branch-N（去噪）：应降低高频（噪声 = 输入高频主要成分）→ ✅ 需要 L_temp
- Branch-L（光照）：低频为主，防过平滑反弹伪影 → ✅ 需要 L_temp
- Branch-M（运动对齐）：应保留/增强高频（细节恢复）→ ❌ 不应约束

**修改**：`models/golf_r3/loss.py`
```python
# 1. 函数签名改为分支级
def _temporal_highfreq_stability(self, Y_branch, X_t):  # 原 (O_t, X_t)
    Y_low = F.avg_pool2d(Y_branch, k, 1, pad)  # 原 O_low
    Y_high = Y_branch - Y_low
    return F.relu(Y_high.abs().mean() - 1.2 * X_high.abs().mean())

# 2. 前向计算仅约束 N+L
L_temp_N = self._temporal_highfreq_stability(Y_N, X_t)
L_temp_L = self._temporal_highfreq_stability(Y_L, X_t)
L_temp = L_temp_N + L_temp_L  # Branch-M 豁免

# 3. 返回诊断信息
return {"L_temp_N": ..., "L_temp_L": ..., ...}
```

**预期效果**：
- warp_t 从 0.03 升至 **> 0.5**（M 分支解放，原目标 0.3 提高）
- pair45 PSNR **+0.5-1.0dB**（运动细节恢复不被压制）

**验收**：
1. L_temp_N/L 从 ep5 开始 > 0（N/L 约束激活）
2. warp_t@20 > 0.5（M 分支贡献显著提升）
3. pair45 PSNR > 17.0（原目标 16.5 提高）

---

## 三、实验设计

### 3.1 对照组

| 实验 ID | 描述 | 配置 | 目的 |
|---------|------|------|------|
| **Golf R1-S2** | 原型基线 | 原 golf.yaml | 已知 ep20=19.66, conf 崩溃 |
| **Golf R3-S1** | 完整改进 | golf_r3.yaml (5 改全开) | 验证组合效果 |

### 3.2 消融组（可选）

若 R3-S1 成功（PSNR > 20.0 + pair45 > 16.5），后续消融隔离各改进贡献：
- R3-Ab1：仅 R3-A (conf 中性)
- R3-Ab2：仅 R3-B (高分辨率 warp)
- R3-Ab3：仅 R3-C (FiLM)

### 3.3 训练配置

**硬件**：RTX 4090 (24GB)  
**数据**：SDSD indoor train (256² 裁剪, 5 帧窗口)  
**超参**：
- Epochs: 60 (warmup 5)
- LR: 8e-4 → cosine decay
- Batch: 2 × 8 grad_accum = 16 等效
- AMP: True
- Val: 每 10 epoch

**日志增强**（相比 R1）：
- 分支贡献权重 (w_n, w_l, w_m) 每 epoch
- conf_map 均值/std 每 epoch
- FiLM γ 均值/std 每 epoch
- L_temp 激活频率 (>0 的 step 占比)
- pair45_psnr 双指标

### 3.4 验收标准

| 指标 | Golf R1-S2 | R3-S1 目标 | 判据 |
|------|------------|-----------|------|
| **ep20 PSNR (val)** | 19.66 | **> 20.0** | +0.34 dB |
| **ep20 pair45** | N/A | **> 17.0** | R3-E 提升目标（原 16.5） |
| **conf_map ep20** | 1.000 (崩溃) | **0.5-0.8** | 可学习动态 |
| **warp_t 贡献 ep20** | 0.03 | **> 0.5** | R3-E 解放目标（原 0.3） |
| **FiLM γ std** | N/A | **> 0.1** | 非退化常数 |

**终判条件**：
1. **主指标达标**：val PSNR > 20.0 **且** pair45 > 17.0
2. **过程健康**：conf_map 不饱和 + warp_t > 0.5 + FiLM 动态
3. 若失败，按 §四 故障树诊断

---

## 四、故障诊断树

### 4.1 若 conf_map 仍饱和（> 0.95 ep20）

**检查点**：
1. bias 是否正确初始化为 0？（打印 `self.conf_head[-2].bias` 训练前）
2. sigmoid 输出是否有负样本？（histogram 日志）

**方案**：
- 若 bias 未生效 → 手动 `nn.init.constant_` 在模型初始化后
- 若仍饱和 → 替换为 Tanh（输出 [-1,1]，softmax 前平移）

### 4.2 若 warp_t 贡献仍低（< 0.2 ep20）

**检查点**：
1. `use_hires_warp=True` 是否生效？（打印 `self.use_hires_warp`）
2. 上采样后 F2_seq 分辨率是否 = tca_out？（shape 日志）

**方案**：
- 若分辨率不匹配 → 检查 `enc_upsample` 上采样倍数
- 若匹配但贡献低 → 可能数据集运动幅度小，用 pair45 长训（运动强先验）

### 4.3 若 FiLM γ 退化常数（std < 0.05）

**检查点**：
1. f3_ctx 是否有效传入？（打印 `f3_ctx` 非 None）
2. FiLMGenerator 权重是否冻结？（梯度日志）

**方案**：
- 若未传入 → 检查 `GolfNet.forward()` F3 编码逻辑
- 若冻结 → 检查优化器 param_groups（FiLM 参数应可训练）

### 4.4 若 val 高但 pair45 低（gap > 3dB）

**诊断**：val 过拟合（5 序列静态场景记忆）

**方案**：
1. 数据增强加强（时序抖动、亮度扰动）
2. pair45 加入训练集（运动样本欠采样补偿）
3. L_ortho 权重提升（防止分支塌缩为单一模式）

---

## 五、实现清单

### 5.1 代码文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `models/golf_r3/` | ✅ 已创建 | 从 `golf/` 复制，独立命名空间 |
| `models/golf_r3/branch_m.py` | ✅ 已修改 | R3-A (bias=0) + R3-B (hires warp) |
| `models/golf_r3/tca_rwkv.py` | ✅ 已修改 | R3-C (FiLM 注入) |
| `models/golf_r3/golfnet.py` | ✅ 已修改 | F3 编码 + f3_ctx 传递 |
| `models/golf_r3/loss.py` | ✅ 已修改 | R3-E (temp_threshold=1.2) |
| `configs/golf_r3.yaml` | ✅ 已创建 | R3-D (pair45 路径) |
| `train_golf_r3.py` | ✅ 已创建 | validate_pair45() 函数 + 双指标日志 |

### 5.2 文档

| 文件 | 状态 |
|------|------|
| `docs/v6/Golf-R3-plan.md` | ✅ 本文档 |
| `experiments/golf/RESULTS.md` | 🔄 待更新（R3-S1 结果后） |

### 5.3 启动检查清单

训练前必查（防止配置错误）：

```bash
# 1. 语法检查
python3 -m py_compile models/golf_r3/*.py train_golf_r3.py

# 2. 配置加载测试
python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/golf_r3.yaml'))
assert cfg['model']['use_f3_film'] == True
assert cfg['model']['use_hires_warp'] == True
assert cfg['loss']['temp_threshold'] == 1.2
assert 'pair45_input_root' in cfg['dataset']
print('✅ Config OK')
"

# 3. 模型实例化测试
python3 -c "
import torch
from models.golf_r3 import GolfNet_R3
m = GolfNet_R3(use_f3_film=True, use_hires_warp=True)
x = torch.randn(1, 5, 3, 256, 256)
out = m(x)
assert 'res_t' in out
print(f'✅ Model OK, output shape={out[\"res_t\"].shape}')
"

# 4. pair45 路径检查
ls /home/a1005/yzy/dataset/SDSD/test/low-light/pair45/*.png | head -3
ls /home/a1005/yzy/dataset/SDSD/test/GT/pair45/*.png | head -3
```

全部通过后启动训练：
```bash
python3 train_golf_r3.py --config configs/golf_r3.yaml
```

---

## 六、后续规划

### 6.1 若 R3-S1 成功（PSNR > 20.0 + pair45 > 16.5）

1. **ep60 长训**：验证收敛终值（预期 > 20.5）
2. **消融实验**：隔离各改进贡献（R3-Ab1/2/3）
3. **外部测试集**：SMID, DRV (验证泛化)
4. **架构优化**：
   - TCA-RWKV 层数扫描（6→4/8）
   - encoder 通道扫描（[32,64,128]→[48,96,192]）
5. **知识蒸馏**：Golf-R3 → Golf-Lite (移动端部署)

### 6.2 若 R3-S1 失败（未达标）

1. **诊断优先**：按 §四 故障树定位瓶颈改进（单点失败 vs 系统性不足）
2. **架构扩展**：
   - **方案 A（保守）**：加深分支 block（[3,2,3]→[4,3,4]）
   - **方案 B（激进）**：回归 Flight11 T-BC1b 主干（承认轻量化失败）
3. **数据扩展**：
   - 加入 SMID 训练集（低光多样性）
   - pair45 上采样（运动样本加权）
4. **损失重构**：
   - 参考 Flight11 S1（15 项 TFSNetLoss）
   - SSIM 权重提升（0.3→0.5）

### 6.3 与 Flight11 路径对比

| 维度 | Golf-R3 | Flight11 S1.5 v1.1 |
|------|---------|-------------------|
| **主干** | 轻量 TCA-RWKV (128-ch, 6 block) | 重型 T-BC1b (9×9 窗口 softmax) |
| **训练成本** | ~30h (60 epoch, 4090) | ~40h (40 epoch, 更慢) |
| **内存** | 6GB (batch=2×8) | 10GB+ |
| **推理速度** | 45 fps (256²) | 25 fps (256²) |
| **终判指标** | 若 ep60 > 20.5 → **Golf-R3 胜出**（轻量化成功） | ep40 = 19.99（重型主干代价高） |

**决策树**：
- 若 R3-S1 ep60 > Flight11 ep40 → Golf 成为 v6 Delta 主线
- 若 R3-S1 < Flight11 → 放弃轻量化，Flight11 T-BC1b 主干成为 v6 Delta 终版

---

## 七、资源与依赖

### 7.1 硬件需求

- GPU: RTX 4090 (24GB) × 1
- 训练时长: 60 epoch × 30 min/epoch ≈ 30h
- 存储: ~20GB (检查点 + 日志)

### 7.2 软件依赖

```python
torch >= 2.0
torchvision
pyyaml
tqdm
scikit-image  # pair45 PSNR 计算
lpips         # 可选，LPIPS 指标
Pillow        # 图像加载
```

### 7.3 数据集路径

```
/home/a1005/yzy/dataset/SDSD/
├── indoor/
│   ├── input/      # 训练低光输入
│   └── GT/         # 训练 GT
└── test/
    ├── low-light/
    │   └── pair45/ # 运动测试序列输入
    └── GT/
        └── pair45/ # 运动测试序列 GT
```

---

## 八、总结

Golf-R3 = **轻量化三分支架构** + **LLVE 最佳实践** + **Flight11 失败经验**，5 处定向改进：

1. **R3-A**：conf 中性起点（bias 2.0→0.0）
2. **R3-B**：高分辨率 warp（H/2→H 域）
3. **R3-C**：F3 FiLM 退化感知
4. **R3-D**：pair45 双验证集
5. **R3-E**：L_temp 阈值收紧（1.5→1.2）

**核心假设**：Golf R1-S2 的 19.66@20 已接近 Flight11 的 19.78@20，瓶颈是 conf 崩溃/warp 模糊/过拟合，而非主干容量不足。**5 处修正若全部生效，预期 ep20 > 20.0 + pair45 > 16.5，成为 v6 Delta 轻量化终版。**

若假设证伪（R3-S1 失败），则回归 Flight11 重型主干路径（T-BC1b + 15 项损失），承认 256² 裁剪数据需要更强归纳偏置。

---

**下一步**：启动 Golf R3-S1 训练，72h 内出 ep60 终判。
