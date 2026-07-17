# Flight 8 设计: Encoder 多尺度解耦 + DPE 三级渐进 + TCA 内置 FPN

> 日期: 2026-07-17
> 基于: Flight7.2+WFR (ep80, PSNR=17.10, gn=0.020, gm=0.017 — 三源激活验证成功)
> 核心目标: 打破 PSNR 平台期(17.1) + 解决 DPE 饱和(s_illum≥0.9)

---

## 一、Flight7.2+WFR 仍然未解决的问题

| 问题 | 证据 | 根因 |
|------|------|------|
| PSNR 平台期 17.1 | ep70→80: Δ=+0.03 | Encoder 单一 FPN 混合特征对三类模块都次优 |
| DPE s_illum 持续饱和 | si=0.88-0.97, std=0.02-0.12 | DPE 缺乏空间光照先验, 收敛到"全亮" |
| S1/S2 OOD 翻转 (DID) | S1 mean=0.98 (过曝) | NDPN/MCPN 缺乏跨数据集鲁棒性 |

---

## 二、三项核心改动

### 改动 1: Encoder 输出多尺度 lateral 特征

**当前**: Encoder 内部做 FPN top-down fusion → 输出单一 `fused (H, 64ch)`
**Flight8**: Encoder 跳过 FPN fusion, 直接输出 3 个 lateral 特征

```python
# Encoder.forward_single → 返回 (l1_lat, l2_lat, l3_lat)
l1_lat = self.lateral1(l1)  # (B, 64, H, W)
l2_lat = self.lateral2(l2)  # (B, 64, H/2, W/2)  
l3_lat = self.lateral3(l3)  # (B, 64, H/4, W/4)
# 不执行 FPN fuse, lateral1/2/3 是 1×1 Conv 投影
return l1_lat, l2_lat, l3_lat
```

效果: WFR 拿 `l1_lat` (原始分辨率, 无 fusion 污染), TCA 拿 `l1_lat+l2_lat+l3_lat` 内部聚合, DPE 拿 `l1_lat+l2_lat+l3_lat` 渐进扫描

### 改动 2: TCA 内部 FPN + 全分辨率 WKV

```python
# TCA.forward:
# Step 0: 内部 FPN 聚合
l1_lat(B,T,64,H,W), l2_lat(B,T,64,H/2), l3_lat(B,T,64,H/4) → FPN → fused(B×T,64,H,W)

# Step 1-3: WKV@H×W (不降采样!)
MVC-Shift → SpatialWKV2D(fused) → ChannelMix → sace_out = fused + cm×γ + wfr×λ

# Step 4-5: 时序对应与聚合 (不变)
C_omega_list, F_t_aligned
```

**WKV 计算量**: 序列长度 256×256=65536 tokens, CHUNK=256, chunk 数=256. 比 H/2 分辨率增加 4× tokens 但仍在 RTX 4090 承受范围内 (~200ms/frame).

### 改动 3: DPE 三级渐进扫描 + 物理先验

```
l1_lat(B,T,64,H)  l2_lat(B,T,64,H/2)  l3_lat(B,T,64,H/4)
                                          ↓
                                      Stage 3 (coarse, H/4):
                                        f3 → [μ,σ,SNR] → cat(gray↓,lum↓) → MSbranch(d=1,2,4) → s3
                                          ↓ bilinear↑2×
                                      Stage 2 (mid, H/2):  
                                        f2 → [μ,σ,SNR] → cat(gray↓,lum↓) → +s3 → MSbranch(d=1,2) → s2
                                          ↓ bilinear↑2×
                                      Stage 1 (fine, H):
                                        f1 → [μ,σ,SNR] → cat(gray,lum) → +s2 → Conv(GELU) → s1
                                          ↓ LN → head
                                      s_illum, s_noise
```

**关键设计**:
- 每级 concat **灰度图(RGB.mean) + 明度图(RGB.max)** 作为物理光照先验
- Stage 3 用 d=4 感受野覆盖全局照明 (64×64→4×4 effective), 估计整体暗度
- Stage 2 用 d=2 做中等梯度精细化, 融合 coarse 估计
- Stage 1 仅需 1 个 Conv——全分辨率下局部纹理即可分辨噪声
- s_noise 可用物理噪声模型: σ_n = √(a·s_illum + b), a/b 可学习 [Foi et al., 2008]

### 改动 4 (可选): DSM 阶段零重置 (仅 Phase 1)

Phase 1(0-10): NDPN/MCPN γ=0, S1/S2 gate=0, residual_scale=0 — 完全隔离
Phase 1.5(11-30): γ 从 0 渐进到 0.01, S1/S2 gate 0→1, residual_scale 0→1
Phase 2(31-85): 全开, 正常训练

---

## 三、实施清单

| # | 文件 | 改动 | 行数 |
|---|------|------|:---:|
| 1 | encoder.py | 新增 `self.lateral1/2/3`, 移除 FPN fuse convolutions, `forward_single` 返回 3 个 lateral | ~15 |
| 2 | pure_rwkv_sace.py | TCA.forward 增加 FPN 聚合 (l1/l2/l3→fused), 移除 H/2 降采样 | ~20 |
| 3 | tfsi_v2.py | DPE 三级渐进扫描 + gray/lum concat + s_noise 物理模型 | ~40 |
| 4 | tfs_net.py | Encoder 路由: l1_lat→WFR, (l1,l2,l3)→TCA, (l1,l2,l3)→DPE | ~10 |
| 5 | losses.py | 同上 (Flight7.2 losses, 无需改动) | 0 |
| 6 | train.py | 同上 (Phase 调度不变) | 0 |

**总改动**: ~85 行, 4 个文件

---

## 四、预期效果

| 指标 | Flight7.2 ep80 | Flight8 预期 | 理由 |
|------|:-------------:|:-----------:|------|
| PSNR (SDSD) | 17.10 (平台期) | **17.5-18.0** | DPE 饱和解决 → 更精准的 s_illum → ISPN 提亮更准确 |
| dpe_si std | 0.02-0.12 | **0.10-0.30** | 物理先验(gray/lum) + 三级渐进 → 空间光照分化 |
| NDPN γ | 0.020 | 0.015-0.025 | 稳定激活, 不再需要过度补偿 |
| DID PSNR (OOD) | 25.24 (img_lit) | 23-26 | 保持 Stage A 鲁棒性 |
| S1 OOD 翻转 | 0.072→0.984 | **0.07→0.07±0.1** | γ 渐进初始化 + 物理先验 → NDPN 不会误判未见的噪声模式 |
| TCA 对齐精度 | 5.5 ifpn | **4.5-5.0** | 全分辨率 WKV → 更精确的空间对应 |

---

## 五、参数变化

| 模块 | 当前 | Flight8 | 变化 |
|------|:---:|:------:|:---:|
| Encoder | ~320K | **~325K** | +5K (lateral convs) |
| TCA | ~300K | **~305K** | +5K (内部 FPN fuse convs) |
| DPE | ~50K | **~65K** | +15K (2 级 lateral + refine convs) |
| 总计 | 1.55M | **~1.58M** | +30K (+2%) |
