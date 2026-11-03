# CoReg-QRB：面向同尺寸配准特征的 QueryRWKV

## 1. 设计目标

TFS-Net 的三路查询不是一般的跨模态 token 检索，而是同一场景、同一尺度、同一像素网格上的任务交互：

- N 路：`Q = F1_single - F1_multi`，`X = F1_multi`；
- L 路：`Q = F1_multi`，`X = F1_single`；
- M 路：`Q = F1_single`，`X = F1_single`。

因此核心先验是 `(q[h,w], x[h,w])` 有确定的成对意义。普通 CrossRWKV 只让 `Q` 生成 `R/G`、让 `X` 生成 `K/V`，没有显式建模这对同位置特征之间的乘性一致性与差异性。CoReg-QRB（Co-Registered QueryRWKV Block）专门补足这一点。

这里的“两阶段调制”并不是由 `C > 1` 导致。通道数只决定投影的维度；真正要求两阶段处理的是：

1. 先从同位置的 Query/Context 对中提取关系；
2. 再用这个关系分别控制“写什么、如何读、是否通过”。

必要条件是两路特征的 `B/C/H/W` 对齐，尤其是 `H×W` 一一对应。实现会显式拒绝形状不一致的输入，避免用广播或插值悄悄破坏配准关系。

## 2. 两阶段关系调制

### 2.1 阶段一：构造同位置关系状态

先分别归一化两路特征，再构造有方向和无方向的关系项：

\[
\bar q=\operatorname{LN}(q),\qquad \bar x=\operatorname{LN}(x),
\]

\[
p=\phi\left([\bar q,\bar x,\bar q\odot\bar x,|\bar q-\bar x|]\right)
  +\operatorname{DWConv}_{3\times3}(\phi(\cdot)).
\]

其中 `q` 与 `x` 保留角色顺序；乘积描述一致性，绝对差描述冲突，深度卷积只在配准关系图上补充局部邻域证据。`4C → C/4 → C` 的低秩瓶颈控制参数和显存开销。

### 2.2 阶段二：把关系分别注入 K/R/G

基线投影仍保留：

\[
r_0=W_r q,\quad g_0=W_g q,\quad k_0=W_k x,\quad v=W_v x.
\]

关系状态不直接替代它们，而是产生有界增量：

\[
r=r_0+\tanh(\alpha_r)W_{pr}p,
\]

\[
g=g_0+\tanh(\alpha_g)W_{pg}p,
\]

\[
k=k_0+\tanh(\alpha_w)\tanh(W_{pk}p).
\]

这三条路径有清晰分工：

- `K` 增量是关系条件写入（write）：决定 Context 的哪些内容进入 WKV 状态；
- `R` 增量是关系条件读取（read）：决定 Query 如何从该状态取回信息；
- `G` 增量是关系条件输出（gate）：抑制不可靠或任务无关的交互。

这比把 Query 连续做两次卷积更有意义：两步分别对应“形成关系证据”和“按 RWKV 内部职责分发关系证据”。

## 3. 同位置读取与全局状态读取的自适应融合

同网格特征还有一个普通序列 CrossRWKV 没有利用的优势：当前位置的 `q[h,w]` 和 `v[h,w]` 本来就是配准的。每个扫描方向 `d` 同时计算：

\[
s_d=\operatorname{LN}_h(r_d\odot\operatorname{WKV}_d(k_d,v_d)),
\]

\[
\ell_d=\operatorname{LN}_h(r_d\odot v_d),
\]

并由关系状态预测方向相关的对齐置信度

\[
\rho_d=\sigma(W_{\rho,d}p).
\]

最终读取为

\[
y_d=g_d\odot\left[s_d+\tanh(\alpha_a)\rho_d(\ell_d-s_d)\right].
\]

`s_d` 提供跨位置的长程状态读取，`ℓ_d` 保留严格同位置的细节证据。这样做不是额外堆一个局部注意力模块，而是在同一个 RWKV head 内显式选择“配准读取”与“扫描状态读取”。这对低层视觉尤其重要：恢复细节需要同位置证据，噪声/运动判断又需要较大感受野。

## 4. 稳定迁移策略

四个调制强度 `αr/αg/αw/αa` 均从 0 初始化。因此初始时：

\[
\operatorname{CoRegQRB}(q,x)=\operatorname{CrossRWKVHeadV8}(q,x)
\]

逐元素完全一致。已有 v8.8-B 权重可以作为初始化来源，新路径再由训练逐步解锁。这一点避免在已有 26.05 dB 基线上突然改变特征分布，也让消融结果可归因。

## 5. 工程接入

- 实现：`models/golf_v7r/coreg_query_rwkv.py`
- 统一入口：`TaskQueryAttention(q_mode="coreg")`
- 实验配置：`configs/v8_5/v9_coreg_30ep.yaml`
- 验证：`python tests/test_coreg_query_rwkv.py`
- 训练：`python train_golf_v7r_v3.py --config configs/v8_5/v9_coreg_30ep.yaml`

基类新增的是兼容扩展钩子；`q_mode: cross_rwkv` 和现有 `kvmod/qpath/gatectx` 的数值路径保持不变。
在 32 通道设置下，每个 CoReg head 增加 4,840 个参数，三个任务头合计增加 14,520，整网参数量约增加 1.1%。训练日志会自动输出三路均值形式的 `K/R/G/A` 有效尺度与 `rho`，完整的分支级状态保存在 `output_dict["coreg_stats"]`。

## 6. 建议消融顺序

为避免一次实验无法定位增益，建议在 v8.8-B 上按以下顺序比较：

1. `cross_rwkv`：原基线；
2. 仅开启关系到 `K`：验证关系条件写入；
3. `K + R/G`：验证读写分离；
4. 完整 CoReg-QRB：加入 aligned/global 双读取；
5. 将 Query 在空间上随机打乱：验证提升确实来自像素配准，而非单纯增加参数。

除 PSNR/SSIM/LPIPS 外，应记录 `tanh(αr/αg/αw/αa)`、各方向 `ρ` 均值与方差，以及 N/L/M 三路分支消融结果。若 `α` 始终接近 0，说明基线已足够；若 `ρ` 饱和而性能没有提升，应优先检查分支监督与融合门，而不是继续增加关系模块深度。

## 7. 创新边界

CoReg-QRB 的可主张创新不是“第一次用两层调制”或“第一次把 Query 输入 RWKV”，而是以下组合与任务先验的统一：

1. 面向同尺寸配准恢复特征的显式成对关系状态；
2. 同一个关系状态对 K/R/G 执行职责分离的写入、读取和门控；
3. 在 RWKV head 内自适应融合同位置读取与全局扫描状态读取；
4. 对已有 CrossRWKV 严格零扰动初始化，可直接迁移稳定基线。

最终是否形成论文贡献仍必须由上述消融、复杂度对照和跨数据集结果共同证明。
