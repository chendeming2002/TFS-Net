# TSDR 框架 — TSD-Net 三源分解的动机与理论解释（7-16 版本存档）

> 记录时间: 2026-09-06
> 来源: TSD-Net 2026-07-16 版本论文章节（原文英文，逐字存档）
> 状态: 当前版本的三源分解理论基础

---

## 原文存档 (verbatim)

Central to TSD-Net is the TSDR framework, which provides a principled basis for decomposing the complex, coupled degradation in low-light video into three independently manageable sources. Our formulation is grounded in two orthogonal dimensions of noise characterization—**temporal independence** (whether the degradation is statistically independent across frames) and **spatial selectivity** (whether the degradation is globally distributed or concentrated at specific locations)—yielding four canonical degradation types:

**(I) globally distributed, frame-independent noise** such as read noise;

**(II) spatially selective, frame-independent noise** such as photon shot noise concentrated in dark regions;

**(III) globally distributed, temporally correlated degradation**, i.e., illumination fluctuation;

**(IV) spatially selective, temporally structured degradation**, i.e., motion artifacts localized at object and camera motion boundaries.

We show that these four types naturally reduce to three operational sources. Types I and II, despite differing in spatial distribution, share the defining property of temporal independence—their realizations are i.i.d. across frames—making them amenable to the same temporal-averaging-based denoising strategy; moreover, they co-occur at every pixel as the Poisson-Gaussian mixture inherent to sensor noise and cannot be analytically separated in a single frame. They thus form a unified **imaging noise source**.

Type III constitutes the **illumination source**, a slowly varying global bias correctable through spatial smoothness priors and temporal anchoring.

Type IV constitutes the **dynamic source**, a structured temporal displacement that requires explicit alignment or compensation rather than averaging. Crucially, no further merging is admissible: the illumination source and the dynamic source, though both temporally correlated, demand fundamentally incompatible processing—temporal smoothing for the former versus spatial compensation for the latter, with naive averaging of the dynamic source producing ghosting artifacts. Likewise, the imaging noise source cannot be merged with either temporal source, as averaging eliminates i.i.d. variance but not systematic bias or structured displacement. This irreducibility guarantees that three is both sufficient and necessary for a complete, non-redundant decomposition.

Guided by this framework, TSD-Net instantiates the tri-source split through a Wavelet Feature Router (WFR) for frequency-level routing, TCA for temporal correspondence-level separation, and a Degradation Prior Estimator (DPE) for source prior estimation, feeding three parallel specialized branches (ISPN, NDPN, MCPN) whose outputs are re-integrated by SGRF in the physically motivated order.

---

## 结构化摘要

### 二维表征矩阵

| | 全局分布 (globally distributed) | 空间选择性 (spatially selective) |
|---|---|---|
| **帧独立 (i.i.d.)** | (I) 读噪声 | (II) 暗区散粒噪声 |
| **时间相关** | (III) 光照波动 | (IV) 运动伪影（运动边界局部化）|

### 四型 → 三源归并

- (I)+(II) → **成像噪声源**：共享"帧独立/i.i.d."本质 → 时间平均降噪有效；泊松-高斯混合共存于每个像素，单帧内不可解析分离
- (III) → **光照源**：缓变全局偏置 → 空间平滑先验 + 时间锚定
- (IV) → **动态源**：结构化时间位移 → 显式对齐/补偿，**不可平均**（平均产生鬼影）

### 不可再归并论证

- 光照 vs 动态：同为时间相关，但**处理不相容**（平滑 vs 补偿）
- 成像噪声 vs 两个时间源：平均只消 i.i.d. 方差，不消系统偏置/结构位移

### 架构映射

WFR（频率路由）+ TCA（时间对应级分离）+ DPE（源先验估计）→ ISPN/NDPN/MCPN 三分支 → SGRF 物理序重构

---

## ⚠️ 现状注记（2026-09-06，含 git 时间线）

上文的 WFR 描述对应 Flight8 时代架构。git 提交时间线（`pure_rwkv_sace.py` / 决策文档）：

| 日期 | 提交 | 事件 |
|------|------|------|
| 2026-07-16 | 2facccb "Flight VII MK2 newWFR" | **716 论文版本写作同日**——WFR (newWFR) 仍在架构中，故 716 文本提及 WFR |
| 2026-07-20 | d0dcd3f "Flight IX FSD" | **WFR 正式取消**（决策与依据：`docs/Delta-Flight9-plan.md` §五）——TCA 改为直连 encoder l2_lat，删除 wfr_lambda |
| 2026-07-23 | 6abd3f8 "Flight X Mark 1" | 频率路由思想部分回收：TCA 内部引入 HaarDWT LL anchor + HF 边缘先验（自称 "minimal WFR"）|

取消依据（Flight9 计划 §五）：

- WFR 在 Flight8 中唯一剩余作用是为 TCA 提供零初始化残差（wfr_lambda=0），从未被有效利用
- InterLight / NID-LLIE / EvLIR（2026 SOTA）均无显式小波分流；多尺度 Encoder 本身就是天然频率分离器（l3 低频全局 / l1 高频局部），各模块直接选尺度

**当前状态**：WFR 类代码仍存于 `models/modules/swd.py`（死代码，无实例化，`losses.py` 的 L_wfr_reg 经 hasattr 守卫恒为 0）。当前实际架构（概念模型与 F10m5 一致）：**Encoder 多尺度隐式路由 + DPE + TCA + 三分支 + SGRF**，无独立 WFR 模块。**716 论文文本的架构枚举（含 WFR）已相对代码过时，论文下一版本需同步更新。**

---

## 三分支输入信息清单与 TSDR 契合性分析（基于 F10m5 实现，2026-09-06）

> 依据 `models/tfs_net.py` forward 实测调用签名 + `ispn_v2.py`/`ndpn.py`/`mrpn.py` 定义。
> 上游组件（Encoder 与三分支之间）实际产出：**DPE**(取 l3, H/4) → `s_illum`, `s_noise`；**TCA**(取 l2, H/2) → `F_aligned_list`(逐帧对齐特征, 上采样 H)、`F_t_aligned`(聚合)、`mu_t_clean`(中心帧均值)、`sigma_t_clean`(帧间标准差)、`C_omega_list`(对应矩阵)；Encoder 直连：`l1_lat`/`l2_lat`/`image_center`(原始 RGB)。

### 逐分支输入清单

**ISPN（光照分支）** — `ispn(f_enc_center, s_illum)`

| 输入 | 来源 | TSDR 语义 |
|------|------|----------|
| `f_enc_center` | Encoder l1 中心帧 (H) | 单帧空间细节基底 |
| `s_illum` | DPE ← l3 (H/4)↑ | Type III 先验：全局低频光照图 |

**NDPN（噪声分支）** — `ndpn(feats, F_aligned_list, mu_t_clean, sigma_t_clean, s_noise, center_idx, C_omega_list, F_t_aligned, image_center, l2_feats)`

| 输入 | 来源 | TSDR 语义 |
|------|------|----------|
| `feats` (l1 全帧) | Encoder | 逐帧原始特征（含 i.i.d. 噪声实现）|
| `F_aligned_list` | TCA | **对齐后的多帧材料**——时间平均降噪的原料 |
| `F_t_aligned` | TCA | 聚合基准 (F_ref) |
| `mu_t_clean` / `sigma_t_clean` | TCA | 时序统计 → SNR 自适应加权 (α) |
| `s_noise` | DPE ← l3 | Type II 空间选择性先验（暗区集中）|
| `C_omega_list` | TCA | 对应质量 → conf_map 门控 |
| `image_center` | 原始 RGB 直连 | 单帧泊松-高斯混合的不可分离基底（细节保留）|
| `l2_feats` | Encoder | 中尺度补充 |

**MCPN（运动分支）** — `mcpn(F_aligned_list, center_idx, sigma_t_clean, C_omega_list, F_t_aligned)`

| 输入 | 来源 | TSDR 语义 |
|------|------|----------|
| `F_aligned_list` / `F_t_aligned` | TCA | 补偿材料 |
| `sigma_t_clean` | TCA | 运动幅度的**间接**代理（帧间方差）|
| `C_omega_list` | TCA | 理论上应提供运动边界空间定位 |

### 与 TSDR 理论建模的逐源契合度

**NDPN ↔ 成像噪声源 (I+II)：结构契合，两处实现折扣**

理论要求"时间平均消 i.i.d. 方差"。`F_aligned_list` 提供平均材料 ✓、`s_noise` 编码 Type II 空间选择性 ✓、mu/sigma 支持 SNR 自适应 ✓。折扣：(a) 旧 C_omega warp 的模糊使"对齐后残差 i.i.d."的前提不成立（T-BC 修复）；(b) C_omega→conf 的前提检验退化为常数（T-BC 修复）。

**ISPN ↔ 光照源 (III)：空间维度契合，时间锚定缺位 ★ 新发现的契合缺口**

理论要求光照源用 **"spatial smoothness priors and temporal anchoring"** 双机制修正。现状：空间平滑 ✓（s_illum 源自 l3 全局感受野）；**时间锚定 ✗**——ISPN 不接收任何时序统计，尽管 `mu_t_clean` 就在 TCA 输出中。716 文本写作时即如此，属理论-结构的最清晰缺口。候选修复：向 ISPN 注入 `mu_t_clean`（或多帧低频均值）作时间锚定项。

**MCPN ↔ 动态源 (IV)：补偿材料就位，测量信号缺位**

理论要求处理 "structured temporal displacement"——按语义 MCPN 理应获得**显式位移/运动幅度测量**。现状仅有 `sigma_t_clean`（间接代理，混入光照变化与噪声方差）与已退化的 `C_omega`（无法定位运动边界）。理论设想的"运动边界空间选择性"无测量支撑。→ **T-BC 直接补齐**：位移场 d*(p)（soft-argmax）+ conf_map（运动边界图）。

**TCA ↔ "temporal correspondence-level separation"：职责关键，分离信号失效**

716 文本赋予 TCA 的职责是把"可平均域"与"需补偿域"**分离**。分离的判据是对应质量——C_omega 对角线退化为常数后，三分支实际拿到的是**未分离的混合材料**，TSDR 的归并禁令（动态源不可平均）在实现层失去执行依据。→ T-BC 的 conf_map 即"分离信号"的正确实现。

**DPE ↔ "source prior estimation"：完全契合**

s_illum（Type III 先验）+ s_noise（Type I/II 先验）与理论一一对应；Type IV 的测量职责理论归于 TCA 而非 DPE——DPE 无运动先验输出是**自洽**的设计。

### 总判与缺口清单

| 源 | 理论机制 | 结构实现 | 契合 |
|----|---------|---------|:--:|
| 噪声 (I+II) | 对齐后时间平均 + SNR 加权 | NDPN 全套输入就位；对齐质量/前提检验两处退化 | ◐ |
| 光照 (III) | 空间平滑 + **时间锚定** | 仅空间平滑；时间锚定未实现 | ✗ |
| 动态 (IV) | 显式对齐/补偿 + 边界定位 | 补偿材料就位；位移场/边界图缺位 | ◐ |
| 分离 (TCA 职责) | 对应质量判据 | C_omega 退化 → 未分离 | ✗ |

**结论**：架构骨架与 TSDR 理论自洽（DPE=先验估计、TCA=对应分离、三分支=三源处理、SGRF=物理序归并），三处实现级缺口中两处（动态源测量、分离信号）由 T-BC 补齐，一处（ISPN 时间锚定）为新识别的待办。另注意：716 论文文本的架构枚举（WFR）与代码不同步（见上文注记），论文改版时应一并更新。
