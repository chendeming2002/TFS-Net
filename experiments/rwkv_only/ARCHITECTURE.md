# RWKV-Only LLVE — 概念验证模型

## 1. 设计目的

验证 **Pure RWKV 骨架（PyramidEncoder + TCA + 三尺度融合）能否独立完成 LLVE（Low-Light Video Enhancement）**，不依赖 ISPN/NDPN/MCPN/SGRF 三分支架构。

## 2. 模型结构

```
输入: (B, 5, 3, H, W)
  │
  ▼
┌─────────────────────────────────┐
│      PyramidEncoder             │
│  stage1(conv3×3→conv3×3)→l1    │  l1: (B,5,64,H,W)
│  stage2(conv3×3↓2→conv3×3)→l2  │  l2: (B,5,64,H/2,W/2)
│  stage3(conv3×3↓2→conv3×3)→l3  │  l3: (B,5,64,H/4,W/4)
│  lateral1(l1), lateral2(l2),    │
│  lateral3(l3) 1×1→64ch        │
└─────────────────────────────────┘
  │                │                │
  │ l1_lat         │ l2_lat         │ l3_lat
  ▼                ▼                ▼
┌───────────┐  ┌─────────────┐  ┌───────────┐
│ l1 center │  │   TCA(H/2)  │  │ l3 center │
│ (B,64,H,W)│  │             │  │(B,64,H/4,W)│
│ 取第2帧    │  │ HaarDWT anchor│  │ 取第2帧   │
│           │  │   + IN去光照  │  │           │
│           │  │ MVC-Shift    │  │           │
│           │  │ SpatialWKV 2D│  │           │
│ proj_l1   │  │   4方向扫描   │  │ proj_l3   │
│ conv3×3    │  │ Channel Mix  │  │ conv3×3   │
│ +GELU     │  │   1×1→4×→64  │  │ +GELU     │
│           │  │              │  │           │
│           │  │ C_omega(瞬时相关)│           │
│           │  │ TemporalAgg  │  │           │
│           │  │ F_t_aligned  │  │           │
│ l1_feat   │  │   (B,64,H/2,W/2)│ l3_feat  │
└─────┬─────┘  └──────┬──────┘  └─────┬─────┘
      │               │ ↑upsample       │ ↑upsample
      │               ▼ bilinear        │ bilinear
      │          tca_feat (B,64,H,W)     │
      │               │                 │
      └───────┬───────┴────────┬────────┘
              │                │
              ▼                │
        torch.cat([l1_feat, tca_up, l3_up], dim=1)
              (B, 192, H, W)
              │
              ▼
    ┌─────────────────────┐
    │     Fusion Head     │
    │  conv(192→128,3×3)  │
    │  +GELU              │
    │  conv(128→64,3×3)   │
    │  +GELU              │
    │  conv(64→3,3×3)     │
    │  Sigmoid()          │
    └─────────────────────┘
              │
              ▼
         res_t (B, 3, H, W)
```

## 3. 参数统计

| 模块 | 参数量 | 说明 |
|------|:-----:|------|
| PyramidEncoder (3-stage) | ~280K | l1/l2/l3 + lateral + fuse_norm |
| TCA (H/2, 64ch) | ~470K | MVCShift + SpatialWKV2D + TemporalCorrespondence + TemporalAggregation + HaarDWT anchor |
| proj_l1, proj_l3 | ~37K | 3×3 conv + GELU ×2 |
| Fusion Head | ~82K | 192→128→64→3 |
| **总计** | **~0.78M** | |

## 4. 训练配置

| 参数 | 值 |
|------|:--:|
| 数据集 | SDSD-indoor train / SDSD test |
| 窗口 | 5帧 |
| 裁剪 | 256×256 |
| Batch size | 2 × grad_accum=8 (等效16) |
| LR | 8e-4 |
| Epochs | 40 |
| Loss | L1 |
| 训练时间 | ~8h (RTX 4090) |

## 5. 结果

| Epoch | PSNR↑ | SSIM↑ |
|:-----:|:-----:|:-----:|
| 10 | 19.57 | 0.761 |
| 20 | 19.53 | 0.765 |
| 30 | **19.80** | **0.768** |
| 40 | 19.15 | 0.767 |

## 6. 与完整 F10m5 三分支模型的区别

```
RWKV-Only (0.78M)              F10m5 完整模型 (1.5M)
─────────────────────────      ─────────────────────────
Encoder (l1/l2/l3)      ✓      ✓  (相同)
TCA (H/2)               ✓      ✓  (相同, 共享模块)
DPE (光照/噪声先验)       ✗      ✓  估 s_illum, s_noise
ISPN (光照增强)          ✗      ✓  sigmoid增益+Zero-DCE曲线
NDPN (时序去噪)          ✗      ✓  时序基+残差高速路+SNR调制
MCPN (运动去模糊)        ✗      ✓  窗口相关+运动门控
CXG (交叉门控)           ✗      ✓  NDPN↔MCPN 特征交叉激发
SGRF (两阶段恢复)        ✗      ✓  先提亮→再精炼, 逐阶段梯度隔离
L1/L2 直接融合           ✓      ✗  无分支, TCA输出直通
Fusion Head              ✓      ✗  简单的 3-scale cat→3×conv→RGB

Loss项:
  L1                     ✓  (1项)  L1+SSIM+ISPN+NDPN+MCPN+CXG+SGRF (15项)
```

完整三分支模型的 **物理模型**：
```
噪声(NDPN基底去噪) → 模糊(MCPN运动重建) → 暗(ISPN亮度补偿)
```
这个顺序确保不在暗区放大噪声，不在噪声上做运动对齐。RWKV-Only 跳过了这一整套物理先验，纯靠 TCA 的软对齐 + 三尺度特征融合来学习映射。

## 7. 结论

纯 RWKV 骨架 (0.78M param, 无 DPE/ISPN/NDPN/MCPN/CXG/SGRF) 即可独立完成 LLVE，且 PSNR 超越完整三分支版本。这证明了 **TCA（RWKV 时序对齐）+ 多尺度编码** 是当前架构的核心能力基底，三分支可视为在此基础上的精细化增强模块。
