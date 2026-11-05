# 逐章写作手册（Abstract → Conclusion）

> 以 v7 草稿（`docs/paper/写作-论文/TSD-Net论文-IntroDraftv7.md`）为当前底稿。写作顺序：**§3 部件 → §3.1–3.2 → §1 → §2 → §4**（部件细节先定，框架与动机后写）。
> 每节给出：段落逻辑、必须包含的要素、措辞禁忌、对应引用。

## Abstract（~150–200 词）

**结构**：任务与难度（1–2 句）→ 现有分解准则的局限（1 句）→ TSDR 三源分解（2 句）→ TSD-Net 实例化 + RWKV 线性复杂度（2 句）→ 结果与贡献（1–2 句，数字用当前最佳）。

**禁忌**：不写未核实的 "first"；不写 "[待核实]"；数字必须与 `docs/v8/13-实验结果汇总.md` 一致。

## §1 Introduction（六段式，v7 已定）

| 段 | 功能 | 要点 | 引用 |
|---|---|---|---|
| 1 | 任务与耦合难度 | 低光子数 → 低 SNR；提亮同比放大噪声；帧独立噪声 → 闪烁；亮度/噪声/时序三者耦合 | [1][2] |
| 2 | 三类方法组织 | Sequential / Implicit / Parallel 各一句机制 | [3,4][5,6][7,8,9] |
| 3 | 各自局限 + 分解准则问题 | 级联误差累积；隐式法依赖数据；parallel 分解按信号表示（成像域/频率/SNR/时间步）而非时序行为 → 不兼容的源进同一支路 | [10][8][11] |
| 4 | TSDR 提案 | 三源（光照/噪声/运动）按时序行为分解；一源一支路 | — |
| 5 | TSD-Net 实例化 | FrameEncoder → TemporalAggregator（均值/中心帧双流）→ TaskQueryAttention → N/L/M 支路 → SGRF 逐段组合；全局上下文由双向 WKV 扫描承载，线性复杂度 | [14][15][16] |
| 6 | 贡献三条 | 见 `narrative.md` 第四节 | — |

**措辞禁忌**：
- 第 3 段批评要"对事不对人"：写分解准则的局限（"partition the degradation by signal representation rather than by its temporal behavior"），不写方法本身的优劣。
- 第 5 段 "All global context in the network is carried by bidirectional WKV scans ... cross-frame combination is a plain average" 是 v7 已定稿句，保留。
- 贡献第 3 条 "the first RWKV-based model for low-light video enhancement" —— 范围限定已含 "low-light video enhancement"，保留；但 §2.3 必须交代 Restore-RWKV 的 "first ... for image restoration"（见红线 2）。

## §2 Related Work（三节）

**§2.1 Low-light video enhancement**：按 Sequential / Implicit / Parallel 三类综述，与 §1 第 2–3 段呼应。每类 2–3 句：机制 → 代表工作 → 局限。注意 VRT/RVRT 是通用视频修复（写明任务范围，见 papers-llve.md）。

**§2.2 Degradation decomposition in low-light enhancement**：三条已有准则 —— 成像域 [10]（依赖 RAW 中间表示）、频率/SNR 统计 [8]（按信号表示分区）、扩散时间步 [11]（与真实时序无关）。每条一句机制 + 一句代价。收尾句引出 TSDR 的"按时序行为分解"。

**§2.3 RWKV-style linear recurrence for vision**：按 `papers-rwkv.md` 的组织建议（Vision-RWKV → Restore-RWKV → LCR → EvRWKV → QRB）。每篇一句机制 + 一句与我们的差异；末尾一段写 QRB 的独特性（同模态统计源 → 三任务查询 + 查询条件化 K）。

## §3 Proposed Method

**§3.1 Tri-Source Decoupled Restoration Framework**
- 观测模型 Eq. (1)(2)(3) + 两维度分类表 + "两源不够"论证（ghosting / 偏置 / 增益放大）+ 依赖顺序段。
- 引用：[1][12][13]（模型）、[2]（时序独立性维度）。
- **sRGB/RAW gap 措辞**用 `papers-noise.md` 的模板。

**§3.2 Overall Architecture**
- 五部件流水线一段 + "TemporalAggregator 是唯一跨帧部件" + 双流设计意图（"compares like with like"）。

**§3.3 Frame Encoding and Temporal Aggregation**
- FrameEncoder：共享权重的物理依据（同传感器同场景模型）；Re-WKV 循环 [15]；EncoderRWKVBlock = OmniShift [15] + 双向 WKV [16] + channel mixing。
- TemporalAggregator：F3 再编码 → 每尺度双流（均值/中心帧）→ 共享参数 top-down（Eq. 4）；均值含中心帧、逐尺度独立。

**§3.4 Task-Gated Query Attention**
- Table 1（三源查询分配）+ 每源一段（假设 → 分配 → 理由）。
- QueryRWKVBlock 机制：cross RWKV gate [14]（R/G←查询路，K/V←另一路）+ kvmod（查询条件化 K 的 per-channel scale/shift，零初始化）+ 双向 WKV [16] + OmniShift [15] + 逐头 LN + SiLU(G)。
- **注意**：kvmod 是本仓库自有（v8.5 配置），写作时明确其为 "query-conditioned keys, zero-initialized so that training starts from the unmodulated gate"。

**§3.5 Source-Specific Branches**
- NBranch（2×NAF + channel attention；目标=中高频噪声份额）、LBranch（Retinex 形式，$Y_L=R_t L_t^\gamma$，零初始化残差补偿，clamp[0,1]；目标=低频）、MBranch（单帧、直接修复；目标=运动中高频份额）。
- 引用 NAFNet/Retinex 系时核对 `reference_repos`（NAFNet、Retinexformer 已在本地）。

**§3.6 Output Composition**
- Eq. (5) + 三个因子（correction/gate/bound-amplitude）+ 零初始化退化性质（"training starts from a plain additive composition"）+ 梯度路径不断。
- 引用：SGRF 的顺序依赖来自 §3.1 的依赖顺序段。

**§3.7 Loss Function**
- Eq. (6)(7) + 频带目标（$\mathrm{LP}_{31}$/$\mathrm{LP}_5$）+ 归一化 + 权重爬坡 + 正交惩罚 + 门正则。

## §4 Experiments

**结构**：实现细节（数据集/训练/指标）→ 主结果（Table 2 复杂度 + 主对比表）→ 消融（分支必要性/QRB 变体/门控/损失）→ 可视化。

**要点**：
- 数字全部来自 `docs/v8/13-实验结果汇总.md`；当前最佳 v8.8-B 26.05/0.8432/0.3426（1.327M）。
- Table 2 的 TSD-Net 行须按当前配置重测（v7 旧行 1.175M 作废）。
- 诚实交代：分支置零代价 N −0.62 / L −17.4 / M −0.57 dB（融合几乎全走 L），N/M 相对误差 0.67/0.58 未达 0.5 判据——放在消融或 limitation，不要隐藏。
- 详见 `experiments-design.md`。

## §5 Conclusion

3–4 句：方法总结（TSDR + TSD-Net + QRB）→ 结果一句 → 局限与未来工作（N/M 支路监督未达标、动态运动数据集验证、真实场景泛化）。

## 通用风格

- 方法用一般现在时，实验用过去时；避免 "we can see that" 类填充。
- 每段主题句先行；公式首次出现给直觉解释。
- 引用编号全文一致（以 v7 的 [1]–[16] 为基础，新增文献顺延）。
- 表格标题在上、图标题在下；Table 1 已定稿（查询分配），Table 2 待重测。
