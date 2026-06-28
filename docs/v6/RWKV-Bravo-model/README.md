# RWKV-Bravo — TFS-Net v6 Bravo Variant

Bravo 版本是对 v6.5 (PureRWKV + DWT-LFF) 的四部分系统性重整：

- **P1**: Encoder 输出拆分 + 双 DWT-LFF + V 投影修正（STCD IJCAI 2025）
- **P2**: 损失重新加权 + Focal Frequency Loss（BVI-Lowlight arXiv 2024）
- **P3**: TFSI 相位置信度估计（FDN TIP 2025 + FourierDiff CVPR 2024）
- **P4**: 架构精简 — 移除 DAT (130K) + AmpEnhance (91K)

核心改动：
- `use_pure_rwkv=True`（强制，无 Fallback SACE）
- `use_dwt_lff=True`（强制，双实例 DWT-LFF：center α=0.6 / neighbor α=0.4）
- 移除 `AmpEnhance`（Encoder 前 FFT 幅度增强废弃）
- 移除 `DeformableCrossAttention`（由 PureRWKVSACE 替代）
- 损失默认权重：`λ_pix=0.3`, `λ_perc=0.8`, `λ_ssim=0.5`
