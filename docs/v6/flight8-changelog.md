# Flight8 更改记录 (Changelog)

> 日期：2026-07-17  
> 基线：Flight7.2+WFR (PSNR=17.10@ep80, dpe_si=0.893/0.063)  
> 目标：解决 DPE s_illum 饱和 + 提升 WKV 空间建模能力  
> 状态：训练中 (ep22, PSNR=14.68@ep20)

---

## 1. 更改动机

### 1.1 DPE s_illum 饱和问题 (Flight7.2 遗留)

Flight7.2 的 DPE 在经过全部训练后 `dpe_si=0.893/0.063`，均值接近 1.0 且空间方差小。根本原因：
- DPE 仅接收 WFR feat_tfde（已损失低频信息）+ Enc center 直连
- 单尺度 multi-dilation conv 无法有效构建从粗到细的光照估计
- 缺少像素域物理先验（原始亮度、最大亮度）辅助定位暗区

**Flight8 方案**: 3-stage 渐进扫描 + gray/lum 先验

### 1.2 TCA WKV 分辨率不足 (Flight7.2 限制)

Flight7.2 的 WKV 扫描在半分辨率 H/2=128×128 进行，序列长 L=16384。升级到全分辨率 H=256×256 (L=65536) 可指数提升空间依赖建模。

**Flight8 方案**: Internal FPN 聚合三尺度 → 全分辨率 WKV, batch=2+accum=8 补偿显存

### 1.3 编码器信息损失

Flight7.2 的 PyramidEncoder 通过 FPN 融合将三尺度坍缩为单尺度输出。下游模块无法利用多尺度上下文——粗尺度 l3 适合全局光照估计，细尺度 l1 适合细节结构。

**Flight8 方案**: 取消 FPN 融合，输出 l1_lat/l2_lat/l3_lat 三尺度 lateral features，各模块自主选择所需尺度

### 1.4 推理无效计算

滑动窗口推理每帧重算全部 5 帧 encoder+WFR，4/5 帧重叠。无缓存机制。

**Flight8 方案**: LRU frame_cache (64帧上限)，配合 `frame_indices` 参数

### 1.5 Sigmoid 末端初始化不确定

NDPN/MCPN 中 4 个 sigmoid 末端层使用默认 kaiming init，训练初期可能饱和导致确定性不足。

**Flight8 方案**: 全部 zero-init (weight+bias=0 → sigmoid(0)=0.5)

---

## 2. 具体更改项目

### 2.1 Encoder — 多尺度输出

**文件**: `models/modules/encoder.py:114-121`

| 项 | 旧 (Flight7.2) | 新 (Flight8) |
|----|---------------|-------------|
| 输出方法 | `forward_single()` → FPN fused (64ch, H×W) | `forward_single_lateral()` → l1(64,H,W), l2(64,H/2), l3(64,H/4) |
| FPN 融合 | 强制 bottom-up 聚合 | 取消，三个 1×1 lateral conv 独立映射 |
| 瓶颈块 | 在最粗层 (H/4) | 保留，位置不变 |

训练路径使用 batched encoder：`x.reshape(B*T, C, H, W)` 一次性编码所有帧 (单 CUDA launch vs T 次 launch)。

### 2.2 DPE — 3-Stage Progressive Scan + 物理先验

**文件**: `models/modules/tfsi_v2.py`

```
旧 (单尺度):
  WFR feat_tfde → GroupNorm → 时域统计(μ,σ,SNR) → MultiScaleDilation(d=1,2,4) → head

新 (3-stage):
  Stage1: l3@H/4 → refine + gray(I↓H/4) + lum(I↓H/4) → s_c3
  Stage2: l2@H/2 → refine + upsample(s_c3) + gray(I↓H/2) + lum(I↓H/2) → s_c2
  Stage3: l1@H   → refine + upsample(s_c2) + gray(I) + lum(I) + cls_token → head
```

新增项:
- **gray prior**: `I_t.mean(dim=1, keepdim=True)` — 原始亮度
- **lum prior**: `I_t.max(dim=1, keepdim=True).values` — 最大亮度  
- **cls_token**: `nn.Parameter(torch.zeros(1,1,C,1,1))` — 全局统计偏差
- L_illum_sup 监督目标从 gain→GT/img_s2 改为直接监督 s_illum 的空间分布

缩并: 输入从 WFR feat_tfde 改为 encoder l1/l2/l3 (直连，避免 WFR 低频分流导致信息损失)

### 2.3 TCA — Internal FPN + Full-Res WKV

**文件**: `models/modules/pure_rwkv_sace.py:335-397`

```
旧 (Flight7.2):
  Encoder fused feats → WKV @ H/2(128×128) → upsampled → tca_out

新 (Flight8):
  l3_lat → l2_lat → l1_lat (bottom-up FPN aggregation) → WKV @ H(256×256) → ChannelMix
  + WFR feat_tca @ H/2 (residual injection, λ zero-init)
```

具体变更:
- 输入从 Encoder fused feats 改为 encoder_lats tuple
- Internal FPN: p3→p2→p1 自底向上聚合（与旧 encoder FPN 同结构，但内嵌在 TCA 中）
- WKV 分辨率：128×128 → **256×256** (4× 序列长，4× 空间建模能力)
- WFR 残差注入：`sace_out += wfr_feat_tca × wfr_lambda` (λ 零初始化)
- C_omega 计算：改为使用 WFR feat_tca @ H/2（对齐精度足够 + 省显存）
- TemporalAggregation：在 full-res sace_out 上进行（原来在降采样后）

### 2.4 训练超参数

**文件**: `configs/delta_flight8.yaml`, `train.py`

| 参数 | 旧 | 新 |
|------|-----|-----|
| batch_size | 4 | **2** |
| grad_accum_steps | 2 | **8** |
| eff batch | 8 | **16** |
| epochs | 85 | **100** |
| Phase 1 | 0-10 | 0-10 (不变) |
| Phase 1.5 | 11-30 (20ep) | **11-35 (25ep)** |
| Phase 2 | 31-85 (55ep) | **36-100 (65ep)** |
| val_interval | 10 | 10 (不变) |

### 2.5 帧缓存系统

**文件**: `models/tfs_net.py:162-183`

| 项 | 功能 |
|----|------|
| `frame_cache` | `Dict[int, Dict[str, Tensor]]` — 按全局帧索引缓存 |
| `clear_frame_cache()` | 清空缓存 (ep boundary + val before) |
| `_cache_put(gidx, key, tensor)` | 存入 (自动 detach) |
| `_cache_evict_lru()` | LRU 淘汰 (上限 64 帧 ≈ 1.6GB) |
| `_cache_max_size` | 64 |
| 推理复用率 | 滑动窗口 4/5 帧命中 (~80%) |

缓存内容: `l1_lat(64,H,W)`, `l2_lat(64,H/2)`, `l3_lat(64,H/4)`, `feat_tca(64,H/2)`

### 2.6 sigmoid 末端零初始化

**文件**: `models/modules/ndpn.py`, `models/modules/mrpn.py`

| 模块 | 层 | 旧 init | 新 init |
|------|-----|---------|---------|
| NDPN | `conf_proj[-2]` (Linear→Sigmoid) | kaiming default | **zeros** |
| NDPN | `denoise_strength[-2]` (Conv→Sigmoid) | kaiming default | **zeros** |
| MCPN | `motion_estimator[-2]` (Conv→Sigmoid) | kaiming default | **zeros** |
| MCPN | `comp_gate[-2]` (Conv→Sigmoid) | kaiming default | **zeros** |

`blur_estimator` 已有正确 zero-init (从 v5 起)。

### 2.7 训练循环显存管理

**文件**: `train.py`

| 新增项 | 触发条件 |
|--------|---------|
| `del outputs, loss, loss_dict, clip, target` | 每步 (防止 grad_accum 期间计算图泄露) |
| `torch.cuda.empty_cache()` | 每 50×grad_accum_steps |
| `model.clear_frame_cache()` | 每个 epoch 开始 + validate 前 |

### 2.8 移除死代码

- `tfs_net.py`: 移除未使用的 `feat_tfde` 变量 (DPE 不再通过 WFR 路由)
- `pure_rwkv_sace.py`: 移除 TCA.forward() 中未使用的 `cached_lff` 参数
- WFR: 训练路径跳过 feat_tfde 计算 (仅限训练模式，推理缓存路径仍保留)

---

## 3. 预期 vs 实际结果

### 3.1 DPE 饱和 (预期失败)

| 指标 | 预期 | Flight8 (ep22) | Flight7.2 (ep85) |
|------|------|---------------|-------------------|
| dpe_si | <0.7, std>0.1 | **0.920/0.000** ✗ | 0.893/0.063 |
| s_illum spatial var | 高 | **零方差** | 略有方差 |

**根因未解决**: gray/lum priors 提供的是像素域亮度信息，但 DPE 内部通过 Conv-GELU 后这些信号被归一化到了窄分布。cls_token 初始为零，需要长时间学习才能产生有效的全局偏置。3 个 stage 的级联导致早期 stage 的饱和传播到所有 stage。

### 3.2 PSNR 对比

| 阶段 | Flight7.2 | Flight8 | Δ |
|------|-----------|---------|-----|
| ep10 (P1) | 17.19 | 16.33 | **-0.86** |
| ep20 (P1.5) | 15.13 | 14.68 | **-0.45** |
| P1.5 dip | ↓2.06 | ↓1.65 | 稍好 |

Flight8 的 ep10 基线显著低于 Flight7.2，说明 multi-scale encoder + 3-stage DPE 的设计本身存在性能退化。

### 3.3 gn 超常增长

gn 从 0.01↗0.049 (5×) vs Flight7.2 的 0.01↗0.020 (2×)。可能原因:
- dpe_si 饱和导致 s_noise 也失去空间分布 → NDPN 通过 gamma 暴走补偿
- full-res WKV 产生的 tca_out 噪声特性与 H/2 版本不同 → NDPN 需要更大的 gamma

### 3.4 正向指标

- LPIPS 在 ep10→20 从 0.352→0.343 (改善) — 感知质量提升
- Phase 1.5 dip 幅度 (1.65 dB) 小于 Flight7.2 (2.06 dB) — 启锁更平滑
- 训练稳定，无 NaN/collapse
- LRU frame_cache 推理正常工作
- Batched encoder 训练无性能退化

---

## 4. 经验教训

1. **3-stage DPE 未解决饱和**: 多尺度级联设计没有内置反饱和机制（如 LayerNorm + 零初始化 head），仅凭 gray/lum 先验和 cls_token 不够。应用显式的 L_illum_spatial 损失强制空间方差。

2. **Encoder 多尺度 vs FPN**: 多尺度输出增加了信息（l1/l2/l3 路由到不同模块），但下游模块 (DPE, TCA) 内部聚合这些尺度的方式存在效率损失。可能 FPN 在 encoder 端做比在 decoder 各模块内做更优。

3. **Full-res WKV 性价比低**: 4× 计算量换 0.86dB 退化 —— 要么 batch/accum 配置不足 (eff=16 可能 < 原 eff=8 因为 batch=2 噪声大)，要么 full-res WKV 引入了额外优化难度。

4. **gn 暴走**: 当 DPE 输出退化为常数时，NDPN 会通过 gamma 过度补偿。应添加 gamma 上限 clamp 或更强的 L_gamma_reg。
