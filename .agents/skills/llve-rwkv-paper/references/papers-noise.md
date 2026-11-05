# 噪声建模/分解理论论文精读笔记（5 篇）

> 精读层级：**全文**＝arXiv HTML 或作者官方 PDF 全文精读（Agent 精读，2026-11-05）。
> 引用核对针对 v6/v7 草稿 [1][10][11][12][13] 与观测模型 Eq. (1)。

## 1. Foi et al. 2008 [12]（全文精读，Tampere 大学官方 PDF）

**Practical Poissonian-Gaussian Noise Modeling and Fitting for Single-Image Raw-Data** · Alessandro Foi, Mejdi Trimeche, Vladimir Katkovnik, Karen Egiazarian · IEEE TIP 17(10): 1737–1754, 2008

- 模型：$z(x)=y(x)+\sigma(y(x))\xi(x)$，噪声拆为信号相关 Poisson 项（$\mathrm{var}\{\eta_p\}=y/\chi$）+ 信号无关 Gaussian 项（$\eta_g\sim\mathcal N(0,b)$）；总方差仿射 **$\sigma^2(y)=a\cdot y+b$**——"shot 方差随信号线性增长、read 方差恒定"的原始出处。
- 拟合：小波域分解 → 平滑区水平集分割 → 每段 (均值, 标准差) → 加权最小二乘 + 极大似然。
- **注意**：2008 原文**没有 Anscombe 变换**（方差稳定化是配套工作）。若论文写 "Foi proposed the Anscombe transform" 是错的。
- **作者只有 4 位，无 Astola**——若参考文献列出 Astola 需删。

**引用核对**：支持 Eq. (1) 的 $\sigma_s^2\cdot$信号$+\sigma_r^2$ 形式 ✅（三篇中唯一给出精确闭式的）。但 **不支持 "four canonical types" 四分类** ❌——原文无任何按"跨帧持续 × 空间全局"维度的退化分类。v7 的 "[12, 2]" 中 [12] 只能作物理噪声背景，四分类本体靠 [2] 或改写。

## 2. ELD [1]（全文精读，arXiv:2003.12751）

**A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising** · Kaixuan Wei, Ying Fu, Jiaolong Yang, Hua Huang · CVPR 2020, pp. 2758–2767

- **arXiv 编号是 2003.12751，不是 2004.14068**（后者是数学论文）。TPAMI 扩展版：*Physics-based Noise Modeling for Extreme Low-light Photography*，arXiv:2108.02158。
- 噪声源：$D=K\cdot I+N$，$N=K\cdot N_p+N_{read}+N_r+N_q$——散粒噪声信号相关；读噪（暗电流/热噪/源跟随器，Tukey lambda 长尾）信号无关；行噪声信号无关且空域相关；量化噪声信号无关。
- 标定→采样 5 步：平场帧估增益 K → 暗帧估读噪 → 对数域最小二乘 → 除以低光系数 f∈[100,300] 加噪 → **乘回 f（"expose bright but excessively noisy contents"）**。
- ELD 数据集：4 相机 × 10 场景 × 3 ISO × 2 系数 = 240 对；SID Sony 上完整模型 39.27/37.13/36.30 dB（×100/250/300），×250/×300 追平或超过真实配对数据。

**引用核对**："brightening the frames amplifies the embedded noise in proportion [1]" ✅（合成流程第 5 步同比放大信号与噪声）；Eq. (1) 混合模型 ✅（结构同源）。

## 3. PNNP [13]（全文精读，arXiv:2310.09126）

**Learning Physics-Informed Noise Models from Dark Frames for Low-Light Raw Image Denoising** · Hansen Feng, Lizhi Wang, Yiqi Huang, Yuzhi Wang, Lin Zhu, Hua Huang · IEEE TPAMI 48(4) 2026: 3952–3969

- 沿用 ELD 模型 $D=K(I+N_p)+N_{indep}$；**物理引导噪声解耦（PND）**：暗帧分解为帧级 FPN + 带级行/列噪声 + 逐像素 i.i.d.（只有最后一项交给网络）；**物理感知代理模型（PPM）**仅 1×1 卷积 + ISO 相关增益层，约 2.30K 参数。
- **不是"每类噪声一个分支"**——是双分支（ISO-dependent × g(iso) + ISO-agnostic）。
- 数字：ELD 数据集平均 46.39 dB（ELD 44.44）；SID 平均 40.83（ELD 39.05）。

**引用核对**：支持 Eq. (1) ✅（模型同 $D=K(I+N_p)+N_{indep}$）；但它是"物理 + 学习混合"，引用时建议写 "physics-informed" 而非纯 physics-based。

## 4. DualDn [10]（全文精读，arXiv:2409.18783）

**DualDn: Dual-domain Denoising via Differentiable ISP** · **Ruikang Li**（第一作者，非 "Zhang, Ruikang"）等 · ECCV 2024

- 双域 = **RAW + sRGB**；可微 ISP（黑电平/白平衡/颜色校正/改进 AHD 去马赛克/可变 tone curve，训练随机化）作桥梁端到端联合训练，**推理时丢弃**。
- RAW 域噪声模型即 Poisson-Gaussian；sRGB 域噪声不做解析建模，而是把 RAW 噪声图随图像一起过可微 ISP。
- 数字：双域 31.48 dB vs 仅 RAW 29.44 vs 仅 sRGB 30.59；DND 49.26 dB。

**引用核对**：v7 "decompositions are built on the imaging domain [10]" ⚠️ 基本准确但需小心——DualDn 是**按成像域分别建模噪声再联合去噪**的去噪架构，不是显式"退化分解"方法。建议措辞："noise is separately modeled and removed across the RAW and sRGB domains [10]"。TSD-Net 批评它"依赖不可得的 RAW 中间表示" ✅ 成立（推理需 RAW 输入 + EXIF 元数据）。

## 5. RDDM [11]（全文精读，arXiv:2308.13712）

**Residual Denoising Diffusion Models** · Jiawei Liu 等 · CVPR 2024 · **arXiv 编号是 2308.13712，不是 2303.08530**（后者是神经科学论文）

- 单一去噪扩散解耦为双扩散：前向 $I_t=I_0+\bar\alpha_t I_{res}+\bar\beta_t\varepsilon$，$I_{res}=I_{in}-I_0$；$\bar\alpha_t,\bar\beta_t$ 是**两组独立 schedule**；逆向确定性采样。
- 残差承载"确定性"（语义跃迁）、噪声承载"多样性"；与 DDIM 系数变换一致（CelebA 5 步 FID 同为 69.60）。
- **纯单帧**：ISTD 阴影去除 30.91 / LOL 低光 25.39 / RainDrop 32.51；无视频/时序建模。

**引用核对**：v7 "the diffusion timestep [11]" ✅ 准确——分解索引正是扩散时间步 $t$；"缺时序设计" ✅ 成立。

## sRGB/RAW gap 措辞模板（写 §3.1 或 limitation 用）

三篇物理模型（Foi/ELD/PNNP）全部在 **RAW/传感器域**（光电子→增益→量化），TSD-Net 的 Eq. (1) 用在 **sRGB 域**；非线性 ISP（gamma/tone mapping）会重塑噪声分布，仿射形式只在局部一阶近似意义下成立。建议两句式：

> Our observation model adopts the affine signal-dependent variance established for RAW sensor noise [1, 12, 13], where the shot-noise variance grows linearly with the expected signal and the read-noise variance is constant. Although formulated for the RAW domain, this affine form serves as a first-order approximation in the sRGB domain, preserving the signal-dependent structure after ISP processing.

补充：ELD/PNNP 含行噪声/FPN 等空域相关项，TSD-Net 的各向同性高斯是进一步简化（sRGB 视频还叠加压缩噪声），可在 limitation 一句带过。
