# TCA 与 RWKV 注意力实现 — 独立分析交付包

> **交付目的**: 本包包含 TSD-Net v6 / Flight 11 中 TCA（时序对应对齐）模块与基于 RWKV 的空间注意力的**最新完整实现**，供独立代码审查与设计分析。代码为逐字复制（import 路径保持原样，供阅读而非直接运行）。
> **版本基线**: Flight 11 (T-BC1b 主干), 2026-09-11
> **上游证据**: 概念模型消融系列 (v1→v2→TBC1B) 与完整模型 Flight11 训练, 详见 §7

---

## 0. 给分析者的一段话背景

TSD-Net 是多帧低光视频增强网络（5 帧滑窗 → 中心帧增强），按 TSDR 框架将退化分为三源（成像噪声/光照/运动），由三个并行分支处理。**TCA 是 Encoder 与三分支之间的"多源分割"核心组件**，承担两个职责：
1. **空间增强**: RWKV 式线性注意力（SpatialWKV2D）在 H/2 分辨率增强特征
2. **时序聚合**: 将 4 个邻帧特征对齐并聚合到中心帧坐标系（Type I/II 时间平均降噪的材料来源）

本包代码经历了完整模型 (F10m5) → 概念模型消融 (定位问题) → Flight 11 (T-BC1b 主干转正) 三个阶段。**当前生产形态 = `local_tca.py` 的 `LocalTCA`**（继承 `pure_rwkv_sace.py` 的 `TCA` 空间路径，替换其时序路径）。

## 1. 文件清单

| 文件 | 行数 | 角色 |
|------|:---:|------|
| `code/local_tca.py` | 181 | **Flight 11 主干（最新）**: `LocalTCA` + `LocalWindowAlignment`（9×9 窗口对齐 + aux 接口）|
| `code/pure_rwkv_sace.py` | 403 | **共享 TCA（空间路径被继承）**: `MVCShift` / `SpatialWKV2D` / `BiWKV`(chunk-cumsum) / `TemporalCorrespondence`(旧 C_omega，已被 LocalWindowAlignment 取代) / HaarDWT anchor |
| `code/blocks.py` | 197 | 依赖: `LayerNorm2d` 等 |
| `code/encoder.py` | 148 | 上下文: `PyramidEncoder`（TCA 的输入 l2_lat 生产者，H/2 分辨率 64ch）|

### encoder.py 血统注记（供审查）

`PyramidEncoder` **直接演化自本项目前身的 MINS-Net 脚手架**（`reference_repos/MINS-Net/models/modules/encoder.py`，"PyTorch implementation scaffold for the first runnable version of MINS-Net on SDSD"——MINS-Net 是项目自己的第一版实现，非外部论文）。逐字继承 `EncoderStage`（每级 2×Conv3×3+GELU，stride 1/2/2）与三级金字塔 + 1×1 lateral 结构。后续演化：fuse 前 LayerNorm2d（v5.9.1，防 lateral 累加值域爆炸）、可选 bottleneck、Flight8 新增 `forward_single_lateral`（跳过 FPN 融合直出三尺度 lateral——**当前生产路径**）。

架构模式归源：stride-2 conv-conv 级联 = U-Net 编码半支的通用形态；lateral 1×1 + 自顶向下相加（仅存于 `forward_single` 的 FPN 路径，生产路径未用）= FPN (Lin et al. 2017) 式。构件（Conv+GELU、标准残差、可选 Pre-LN+LayerScale）为通用件，无论文级特异设计；Pre-LN/LayerScale 变体受 NAFNet 启发但**默认关闭**（实证 LN 在浅 Conv 主干导致过拟合）。

## 2. 数据流（LocalTCA.forward 全景）

```
输入: l2_lat (B, T=5, C=64, H/2, W/2)   ← encoder.py 产出

── 空间路径（继承自 TCA, 纯逐帧 2D 处理）────────────────────
feats_flat (B·T, 64, H/2, W/2)
  → _haar_dwt: LL(H/4) + [LH,HL,HH]           # Flight10m1 "minimal WFR"
  → dwt_anchor(LL) = InstanceNorm + 1×1        # LL 去光照偏置
  → dwt_hf_proj(cat(LH,HL,HH)↑)                # HF 边缘先验
  → anchor_fuse(cat[feats, anchor↑, hf↑])      # 3C→C 1×1 → x_enhanced
  → mvc_shift(x_enhanced)                      # 膨胀 1/2/3 DW-conv + 1×1
  → spatial_wkv(x_shifted)                     # ★ 4 方向 RWKV 注意力 (§3)
  → channel_mix(1×1→4C→C)
  → sace_out = x_enhanced + x_cm × spatial_gamma
  → sace_4d (B, T, C, H/2, W/2)
  → mu_t_clean = sace_4d[:,2];  sigma_t_clean = sace_4d.std(dim=1)

── 时序路径（LocalWindowAlignment, 替换旧 C_omega warp）────
center = sace_4d[:,2], neighbors = sace_4d[[0,1,3,4]]
  → embed(1×1, 64→16) + L2 归一                # 共享匹配嵌入
  → 81 个 (dx,dy)∈±4px 偏移逐个 pad-crop       # _pad_crop
  → logits = cos(center_k, neighbor_k)/temp    # temp=softplus+0.02 可学习
  → + identity_bias(5.8 可学习) @ (0,0) 偏移   # ★ bootstrap (§4.2)
  → softmax 逐帧归一 (每帧 81 内, 非联合)       # ★ 上限修复 (§4.2)
  → pass2: warp_t = Σ prob·neighbor_t          # 局部凸组合
  → g_t = sigmoid(frame_gate[center, warp_t])  # per-frame 可弃权门
  → F_agg = Σ g_t·warp_t / Σ g_t
  → conf = mean_t max_d prob_t (3×3 平滑)      # 匹配唯一性先验
  → disp_field = Σ prob·(dx,dy) (conf 加权帧均) # soft-argmax 位移场
  → F_out = LN(conf·F_agg + (1−conf)·center)

输出 dict: F_t_aligned / conf_map / warped_list(逐帧) / disp_field / mu / sigma
（conf_map 替换 NDPN 的退化 conf_proj; disp_field→MCPN 运动门;
  warped_list→NDPN 全分辨率平均材料; 外部另有 luckiness 残差门乘入 NDPN α）
```

## 3. RWKV 注意力算法细节（pure_rwkv_sace.py）

### 3.1 SpatialWKV2D — 四方向扫描

```python
scan_fns = [水平 row-major, 垂直 col-major, 主对角线, 反对角线]
# 每方向: k/v 头(16ch) → 扫描重排 → BiWKV(因果+反因果) → 逆重排
# 4 头拼接 → sigmoid(r)·wkv → proj_out(零初始化) → post-LN
```

**与参考实现的偏离（有意设计，供审查）**：
- 参考库（URWKV/DRWKV/EvRWKV，本地 reference_repos/ 已核）全部用**单序列 1D 双向 CUDA kernel**；我们的 4 方向（尤其两条对角线）**无先例**，且对角线拼接处空间不连续，与 RWKV decay 的局部连续假设存在张力
- 对角线索引用纯 Python 循环逐 forward 重建（`_scan_diag_*`/`_inv_scan`），性能开销显著
- 参数化：`w = -softplus(spatial_decay)` 强制衰减为负；k/v clamp(±8) 替代参考 kernel 的 max-subtraction 稳定化

### 3.2 BiWKV._scan_cumsum — chunk-cumsum 递推（含 2026-09-05 数学修复）

数学语义（因果方向）：

```
out(t) = (u·ekv_t + Σ_{i≤t} ekv_i·ew^{t-i}) / (u·ek_t + Σ_{i≤t} ek_i·ew^{t-i})
其中 ek=e^k, ekv=e^k·v, ew=e^{w/T}, u=e^{u/T}
```

CHUNK=256 分块实现。**修复记录（潜伏一年的 bug，naive 对照实锤）**：

```python
# 旧 (错误): decay_state = ew^{cs-1-j} — 上一 chunk 贡献随 j 递增(方向反)
#            state 更新 off-by-one: ew^{cs-1}
# 新 (正确): decay_state = ew^{j'+1}  — 旧状态到当前位 j' 距离 j'+1, 物理正确
#            state 更新: ew^{cs}
```

**可复现验证**（我们已跑通，欢迎独立复核）：

```python
# naive 参考 (闭区间 i≤t, 双重循环) vs _scan_cumsum:
#   L=600(3 chunk), 随机 w<0: 修复后 max diff = 4.77e-07 (PASS)
#                            旧版    max diff = 2.40e-02 (BUG 实锤)
```

修复后实测对 PSNR 中性（误差在 S/D 比值中大部分相消）——修复理由是正确性底线而非性能。

### 3.3 MVC-Shift

膨胀率 {1,2,3} 的 DW-conv(3×3) + 1×1 三支路残差累加。**与参考偏离**：参考的 q_shift/lerp 是"前序 token 注入"（与 WKV 递推耦合）；MVC-Shift 是并行局部特征增强，无"前序状态"概念。参考 EvRWKV 的 OmniShift（ID+1×1+3×3+5×5 可重参数，恒等分支可学习权重）是更强的形态。

## 4. LocalWindowAlignment 关键设计（local_tca.py）

### 4.1 为什么替换旧 C_omega（TemporalCorrespondence）

旧路径：特征 avg_pool→32² → N×N **全局** softmax 矩阵 → warp = 全帧凸组合 → 上采样。三重结构性缺陷（均有实测证据）：
1. 对齐粒度每格 8×8px；2. 全帧凸组合 = TSDR 理论警告的"对动态源朴素平均产生鬼影"（最难场景 pair45 实测模糊）；3. 其对角线置信信号**退化为全局常数**（std=0.0000）。

### 4.2 两个实测驱动的修复（重要工程教训）

**bootstrap=bias（自举死锁修复）**: T-BC1 原版 conf=max-prob 初始≈1/81=0.013，而 conf 是**乘性自门控**（out = conf·F_agg + ...）——对齐支路梯度被 conf 自身缩放 ~75×衰减 → 20 epoch 训练 conf 纹丝不动（实测 0.0133 恒定，静态/运动场景无分化）= 冷启动死锁。修复：恒等偏移 (0,0) 的 logit 加可学习偏置（init 5.8，初始 conf≈0.78）——"先信任对齐，学会在错误处不信任"。

**逐帧 softmax（上限钳制修复）**: 原实现 Tn·R²=324 联合 softmax——4 帧的恒等偏移互相竞争，**conf 上限被钳在 1/Tn=0.25**。修复为逐帧 81 内归一后，上限恢复 1.0 且"帧内对齐置信 × 帧间门控"语义解耦。

### 4.3 aux 接口（Flight 11 三分支对接）

| 输出 | 消费方 | 语义 |
|------|--------|------|
| `warped_list` | NDPN F_aligned_list | 全分辨率逐帧对齐（"对齐后残差 i.i.d."前提成立，时间平均红利实测 +0.65dB@运动场景）|
| `disp_field` (B,2,H,W) | MCPN motion_mag=1−exp(−\|·\|) | soft-argmax 亚像素位移（首次 Type IV 直接测量）|
| `conf_map` | NDPN（替换退化 conf_proj）| 匹配唯一性**先验**——注意：实测静态/运动无分化（0.698≈0.699），**不可作运动判别器** |

## 5. 已知权衡与遗留问题（诚清单）

1. **conf 非判别性**: SDSD 室内运动仅 ±1-4px，窗口内真假匹配高度相似 → max-prob 天然不区分。两次实现（C_omega 对角线 / 窗口 max-prob）均证伪"运动感知路由"。有效形态 = 始终开启聚合 + 可学习帧门
2. **窗口半径 ±4px 封顶**: 大位移场景 conf→低→回退单帧（设计内降级），但无补偿能力（MCPN 职责）
3. **单层 16ch/方向**: 参考实现为 15-32 层全通道堆叠；我们单层 64ch÷4 方向——TA 实验（算子修复+gamma=ones）表明该容量下 WKV 净贡献为小正（+0.13dB vs 无TCA），未复现参考模型的深层收益
4. **数值策略**: clamp(±8) 无 max-subtraction；identity_bias 会轻微偏置 soft-argmax 位移估计（向 (0,0) 收缩）
5. **Python 对角扫描性能**: 每次 forward 重建索引；H/2=540×960 推理时显著

## 6. 关键实验证据（消融摘要，同预算 20ep 配对）

| 实验 | 配置 | 全局 PSNR@40 | pair45(最难运动场景, 127帧) |
|------|------|:--:|:--:|
| v2（旧 C_omega + motion gate）| 基线 | 20.02 | 15.04 |
| 无TCA | 移除 TCA | — | 19.56@20（单帧过拟合下跌）|
| T-BC1（窗口对齐, 死锁）| conf 恒 0.013 | 19.69@20 | — |
| **T-BC1b（当前形态）** | **bootstrap+bias+逐帧softmax** | **19.99** | **15.69 (+0.65)** |

完整模型 Flight11（T-BC1b 主干 + 三分支 + disp_field/conf_map/luckiness 对接）训练中（`outputs/sdsd_f11_simple2/`，v1.1 简化损失）。

## 7. 希望分析的问题（按优先级）

1. **`_scan_cumsum` 修复后数学的独立复核**：跨 chunk 状态传播（`decay_state=ew^{j'+1}`、state 更新 `ew^{cs}`）是否与因果 WKV 闭式精确一致？闭区间（含 i=t）+ u-bonus 的语义下是否有更优实现？
2. **对角扫描与 decay 局部连续性**：对角线拼接序列在交界处空间跳变，静态 per-channel 指数衰减假设是否被系统性破坏？有无低成本缓解（如交界 reset / 方向专属 decay）？
3. **decay 参数化**：单层架构下，静态 `-softplus` vs fancy init（层深 ramp）vs input-dependent（LoRA 生成，RWKV v7 式）哪个收益/成本比最高？
4. **disp_field 的 identity_bias 偏置**：soft-argmax 在 bias=5.2（训练后）下向 (0,0) 收缩多少？如何无偏化（如 bias 仅作用于 conf 路径、warp 用无偏 softmax）？
5. **窗口对齐的上限**：81 位置局部凸组合 vs 光流 grid_sample（CDVD-TSP 式）在 ±4px 内的精度/鲁棒性权衡？嵌入维度 16 是否足够？
6. **luckiness 残差门与 conf 的互补性**（NDPN 内实现，`exp(−‖warp−center‖²/2δ²)` @4×降采样+5×5 mean filter）：先验（warp 前）×后验（warp 后）双门设计是否合理？
7. **单层容量瓶颈**：若保持参数预算（~0.5M），4 方向×16ch 单层 vs 2 方向×32ch vs 参考式深层堆叠的浅化版，哪个更有希望？

## 8. 复现环境备注

- PyTorch 2.1 + CUDA 11.8, RTX 4090；模型 ~1.5M 参数（含三分支）
- 训练: SDSD-indoor, 5 帧窗, 256² crop, batch 2×accum 8, AdamW 8e-4, AMP
- `code/` 内文件为 `models/modules/` 逐字副本；import（`models.modules.*`）在包外不可直接运行，分析以阅读为主

---

*包生成: 2026-09-11, 基线 commit: aa78696 之后的工作树*
