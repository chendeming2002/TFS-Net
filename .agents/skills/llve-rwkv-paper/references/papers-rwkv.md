# RWKV 参考论文精读笔记（4 篇）

> 精读层级：四篇均为 arXiv HTML 全文精读（2026-11-05）。引用核对针对 v6/v7 草稿 [14][15][16] 与本仓库设计文档。
> 延伸：`docs/v8/21-CrossAttention-RWKV论文全景.md`（跨流 RWKV 论文动态审计，类型①–④分类）。

## 1. Vision-RWKV [16]

**Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-Like Architectures**
Yuchen Duan, Weiyun Wang, Zhe Chen, Xizhou Zhu, Lewei Lu, Tong Lu, Yu Qiao, Hongsheng Li, Jifeng Dai, Wenhai Wang（OpenGVLab 等）· arXiv:2403.02308（v3, 2025-03）· 代码开源（github.com/OpenGVLab/Vision-RWKV）· 页面无 venue 标注

**方法核心**
- **Q-Shift（四方向 token 混合）**：通道切 4 段，分别取上/下/左/右邻居做线性插值：$X^\dagger[h,w]=\mathrm{Concat}(X[h-1,w,0{:}C/4],\ X[h+1,w,C/4{:}C/2],\ X[h,w-1,C/2{:}3C/4],\ X[h,w+1,3C/4{:}C])$；插值系数 $\mu^{(*)}$ 对 R/K/V 三组独立可学：$\mathrm{QShift}^{(*)}(X)=X+(1-\mu^{(*)})X^\dagger$。
- **Bi-WKV（双向一维扫描，求和形式）**：$wkv_t=\frac{\sum_i e^{-(|t-i|-1)/T\cdot w+k_i}v_i+e^{u+k_t}v_t}{\sum_i e^{-(|t-i|-1)/T\cdot w+k_i}+e^{u+k_t}}$；两方向经 4 个 RNN 隐状态（分子 a/b、分母 c/d）合并成一次除法。前向衰减 $e^{-w/T}$、反向增长 $e^{w/T}$。
- **衰减 w 与偏置 u 都是 C 维可学习向量（逐通道常量），不是输入相关**——与 Mamba 的选择性机制不同。写论文时不要写成 data-dependent decay。
- **Block**：spatial-mix（Q-Shift → Bi-WKV → $\sigma(R)\odot wkv$ → $W_O$ → LayerNorm，post-norm）+ channel-mix（$V_c=\mathrm{squaredReLU}(K_c)W_{V_c}$，$\sigma(R_c)\odot V_c$）。

**关键数字**：FLOPs(Bi-WKV)=13·T·C，O(TC)；2048² 输入下比 ViT-T 快 10×、内存降 80%。ImageNet：T 75.1%（DeiT-T 72.2）/ S 80.1 / B 82.0 / L 86.0；COCO 50.6 box AP；ADE20K S 47.2 / L 53.5 mIoU。

**引用核对（v7 [16]）**："a bidirectional WKV scan [16] carries global context at a cost linear in the number of pixels" ✅ 准确。
**注意**：论文的四方向是 Q-Shift（注意力前的 token 混合），Bi-WKV 本身仍是一维展平扫描。四方向 2D 扫描（SpatialWKV2D）出自 RSRWKV，本仓库从那里借。不要在论文里把四方向扫描归给 Vision-RWKV。

## 2. Restore-RWKV [15]

**Restore-RWKV: Efficient and Effective Medical Image Restoration with RWKV**
Zhiwen Yang, Jiayin Li, Hui Zhang, Dan Zhao, Bingzheng Wei, Yan Xu · arXiv:2407.11087（v3, 2025-01）· 预印本（无 venue 标注）· 代码开源（github.com/Yaziwel/Restore-RWKV）

**方法核心**
- **Omni-Shift**：4 分支深度卷积（1×1、3×3、5×5、恒等）可学习加权融合；测试时**结构重参数化**为单个 5×5 卷积。动机：Uni-Shift/Quad-Shift 只从有限方向移位，未利用 2D 空间关系；k=5 经消融最优（32.09 vs k=3 的 31.99 dB）。
- **Re-WKV**：Bi-WKV 沿**交替扫描方向**（H-Scan/V-Scan）递归 M 次（默认 M=2），同一套参数复用 → 零额外参数换 2D 全局感受野；M=1→2 提升 0.15 dB，M≥2 饱和。论文的论证是**表达效率**（"enhances global token interactions ... recurrently performing attention along different scan directions"），不是直接以参数量为由。
- **URWKV**：4 级 U 形 encoder-decoder（C=48，N=[4,6,6,8]，refinement 4 块）；light 版 1.16M 参数。
- 损失：L1，Adam，30K 迭代，patch 128。

**关键数字**：PET 37.33 / CT 去噪 33.80 / MRI SR 32.09 dB；light 版（1.16M/1.52G）CT 去噪 33.64 dB 超过 DenoMamba（112.62M/110.24G，约 97× 参数）。

**引用核对（v7 [15]）**："as in Re-WKV [15] the block is looped a fixed number of times, so the encoding depth is decoupled from the parameter count" ✅ 与论文一致（同块循环、参数不变、深度解耦）。
**注意**：论文 Comments 自称 "the first RWKV-based model for image restoration"（医学图像修复）。TSD-Net 的 "first ... for low-light video enhancement" 与之不冲突，但 §2.3 必须提及并区分范围。

## 3. Video RWKV / LCR [14]

**Video RWKV: Video Action Recognition Based RWKV**
Zhuowen Yin, Chengru Li, Xingbo Dong · arXiv:2411.05636（2024-11）· 预印本 · 代码公开

**方法核心（Cross RWKV gate，本仓库 QRB 的直接来源）**
- $R,G=\mathrm{QShift}(Cauconv(x_t,h_{t-1}))W_{R,G}$：来自**当前主干特征 + 历史隐状态**；$K,V=\mathrm{QShift}(x_t^e)W_{K,V}$：来自**当前帧边缘特征**。
- 输出：$a_t=\mathrm{concat}\big(\mathrm{SiLU}(G_t)\odot \mathrm{LN}(R_t\cdot \mathrm{Bi\text{-}wKV}_t)\big)W_a$。次序：**R 门控 WKV 输出 → 逐头 LayerNorm → SiLU(G) 调制 → 投影**。
- 边缘特征：Canny（Otsu 自适应阈值）+ 零初始化卷积；边缘还作为 LSTM 遗忘门调制（$C_t=x_t^e\times(\tanh(A_t)+C_{t-1})+C_{t-1}$），抑制背景。
- 整体：3D patch embedding（1×16×16）+ L 个 LCR block（逐帧处理）+ CLS 分类。

**关键数字**：Jester 90.83% Top-1（5.14M / 0.022T FLOPs；TimeSformer-L 89.94% / 4.8M / 43.1T）。K400/SSv2 表格未在抓取段落中。

**引用核对（v7 [14]）**："the query source produces the receptance and gate signals, while the key-value source produces the key and value" ✅ 方向正确（R/G←查询路，K/V←另一路）。
**注意**：任务是**动作识别**，不是修复/低光。§2.3 引用时须写明 "video action recognition"。

## 4. EvRWKV（v6/v7 未引，§2.3 建议补充）

**EvRWKV: A Continuous Interactive RWKV Framework for Effective Event-Guided Low-Light Image Enhancement**
Wenjie Cai, Qingguo Meng, Zhenyu Wang, **Xingbo Dong**, Zhe Jin · arXiv:2507.03184 · IEEE TCSVT 2026, 36(7):10201–10216

**方法核心**
- **Cross-RWKV**：$\mathrm{wkv}^{img}=\mathrm{Re\text{-}WKV}(K_s^{ev},V_s^{img})$，$\mathrm{wkv}^{ev}=\mathrm{Re\text{-}WKV}(K_s^{img},V_s^{ev})$——**K 取对侧流，V/R 取本路流**（与 LCR 的 R/G←查询路、K/V←另一路分工不同）。
- 融合门控：$X_1=\sigma(\alpha_{img})\mathrm{wkv}^{img}+(1-\sigma(\alpha_{img}))X^{ev}$；输出 $O_{img}=(\sigma(R_{img})\odot X_1)W_o^{img}$。
- Re-WKV = Bi-WKV 沿 H/V 交替扫描 2 次；CS-Shift = 多分支深度卷积（1×1/3×3/5×5 + 恒等）+ 测试时重参数化。
- **EISFE**：频域支（每通道可学高斯核 FFT 滤波）+ 空域支（可变形卷积），空间/通道注意力融合。
- "连续交互"：在 4 级 U 形每一级都做跨流交换，反对 early/late fusion 二分。

**关键数字**：SDE-in 23.09 dB（+1.79）/ SDSD-in 28.96 / RELED 32.18；24.2M 参数；下游分割 mIoU 30.84→41.77。

**与本仓库关系**：OmniShift 的代码出处（`reference_repos/EvRWKV/.../Cross_Restore_RWKV.py`）；与 Video RWKV 同组（Dong Xingbo）。§2.3 可作为"跨流 RWKV 在低光增强的最近邻"引用，并说明 QRB 与之的差异（同模态统计源 → 任务分解查询 + 查询条件化 K）。

## §2.3 组织建议（RWKV-style linear recurrence for vision）

按"机制演进"叙述：Vision-RWKV（视觉 RWKV 基线：Bi-WKV + Q-Shift）→ Restore-RWKV（循环复用 + OmniShift，修复域）→ Video RWKV/LCR（跨流门控：R/G 与 K/V 异源）→ EvRWKV（K 跨流、V 本路；低光增强）→ TSD-Net QRB（同模态统计源 → 三任务查询 + 查询条件化 K，首个低光视频增强）。每篇一句机制 + 一句与我们的差异。避免把四方向扫描归给 Vision-RWKV；避免与 Restore-RWKV 的 "first" 冲突。
