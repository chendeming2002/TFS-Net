# Core-QRB：删除 G 后的最小交叉 RWKV

> 日期：2026-11-04
> 状态：代码、配置和单元测试已完成；Core-current 30 epoch 训练进行中
> 历史基线：v9 CoReg-QRB（26.01 dB / 0.8433 / 0.3504）
> 主实验配置：`configs/v9/v9_core_qrb_current_30ep.yaml`

## 1. 结论

CoReg-QRB 证明了“Query 应参与 WKV 写入并修正同位置读取”是有效方向，但它先构造
`[Q,X,Q·X,|Q-X|]` 成对描述子，再分别注入 K/R/G/A，概念和实现都偏重。30 epoch 终值显示：

| 作用 | K | R | G | A | ρ |
|---|---:|---:|---:|---:|---:|
| 终值 | 0.318 | 0.093 | −0.049 | −0.264 | 0.711 |

K 和 A 被明显使用，R 只有温和修正，G 的关系注入接近零。因此新结构只保留三个根本问题：

1. **读什么**：Q 产生 R；
2. **写什么**：Q 直接产生 K 的增量；
3. **全局状态与同位置内容如何修正**：Q/X 联合产生一个有符号 A。

Spatial Mix 的 G 路被完整删除，不保留 `shift_g`、`proj_g`、`SiLU(G)`、`pair_to_g` 或
`gate_scale`。Channel Mix 内部原有的 `sigmoid(r2)` 是逐位置通道变换的 receptance，不是
Spatial Mix 的 G，因此保持 v9 已验证实现不变。

## 2. 为什么 A 必须改为直接有符号

旧 CoReg 的单头读出为：

\[
Z=U+\tanh(s_A)\,\rho\,(V-U),\qquad \rho=\sigma(f_A(P)).
\]

训练终值中 `aligned_scale=-0.264`、`rho=0.711`。用两项均值估计，其平均有效系数约为
`−0.264×0.711≈−0.188`。因此不能把 `rho>0.5` 单独解释成“模型偏向同位置 V”；实际作用是
沿 `V-U` 的负方向做修正。旧结构用一个有符号全局尺度再乘一个非负位置权重，既难解释，
也不能让不同位置独立选择正负方向。

Core-QRB 直接定义：

\[
A=\tanh\left(W_A[\bar Q,\bar X]\right),
\qquad A\in[-1,1]^{B\times HW\times D}.
\]

每个像素、每个扫描方向都有自己的有符号系数，不再使用第二个 `aligned_scale` 或 sigmoid ρ。

## 3. 最小结构

Q 和 X 分别只执行一次局部混合与归一化：

\[
\bar Q=\operatorname{LN}(T_Q(Q)),
\qquad
\bar X=\operatorname{LN}(T_X(X)).
\]

然后生成：

\[
R=W_R\bar Q,
\]

\[
K=W_{KX}\bar X+W_{KQ}\bar Q,
\qquad
V=W_V\bar X,
\]

\[
A=\tanh(W_A[\bar Q,\bar X]).
\]

对第 d 个方向：

\[
U_d=\operatorname{BiWKV}_d(K_d,V_d),
\]

\[
Z_d=U_d+A_d\odot(V_d-U_d),
\]

\[
Y_d=\operatorname{LN}_{head}(R_d\odot Z_d).
\]

四个方向拼接、线性投影并加回 Q 残差，随后进入未修改的 Channel Mix。

这个定义仍是交叉 RWKV：Q 不提供被搬运的内容 V，但决定状态读取 R、状态写入 K 和同位置
修正 A；X 提供 K 的主体和 V 内容。

## 4. 初始化与可训练性

- `W_KQ=0`：初始 K 只由 X 构造，不在第一步突然改变 WKV 状态分布；
- `W_A=0, b_A=0`：初始 A=0，读出严格为 recurrent state U；
- R、KX、V、输出投影沿用现有 v9 初始化；
- Q 残差和 Channel Mix 保持不变。

`W_KQ` 和 `W_A` 虽然零初始化，但都在第一次反向传播获得非零梯度。新增测试同时验证了
Query/Context、Kq 和 A 的首步梯度。Core-QRB 的初始行为不是旧 CoReg 的逐元素复刻，因为已
有意删除整个 G 输出门；它只保证新增 Kq/A 修正从零平稳打开。

## 5. 删除的 CoReg 组件

Core-QRB 不再包含：

1. `[Q,X,Q·X,|Q-X|]` 手工成对描述子；
2. `pair_in/pair_out` 低秩 MLP；
3. `pair_local` 3×3 局部关系卷积；
4. `pair_to_r/pair_to_g/pair_to_k/pair_to_aligned` 四投影；
5. `read_scale/gate_scale/write_scale/aligned_scale` 四个全局尺度；
6. Spatial Mix 的 `shift_g/proj_g/SiLU(G)`；
7. sigmoid ρ 与 signed scale 的两阶段 A 调制。

R 不再接收额外关系残差。Q 已经直接生成 R，再为 R 添加一条关系 MLP不属于最根本需求。

## 6. Q 侧局部 Mixer

K/V 的 Context Mixer 继续使用现有 OmniShift（或现有 `shift_kind` 指定的 RLK-TS），且 K/V
共享一次结果。Q 侧提供四个严格互斥的开关：

| `q_mixer` | Q 侧结构 | 用途 |
|---|---|---|
| `current` | OmniShift → SPAB | 主实验；先隔离 QRB 关系重构，不同时改变既有 Q 局部路径 |
| `omni` | OmniShift | 判断 SPAB 是否必要，并作为纯 TokenShift 对照 |
| `span` | SPAB | 用 SPAN 卷积注意力直接承担 TokenShift 的局部混合职责 |
| `wt` | WTConvLite | 用一级 Haar DWT、子带深度 3×3 卷积和 IWT 提供多频局部混合 |

四种方案都只改变 `T_Q`，R/Kq/A 共用同一份 Q tokens，不再为不同角色重复卷积和 LayerNorm。

### 6.1 SPAN 方案

`span` 只保留一份 SPAB：

```text
Q -> SPAB -> LN -> R / Kq / A
```

它不是在 OmniShift 后额外堆注意力，而是完整替代 Q 侧 OmniShift。这样才能回答“参数无关注意力
卷积能否承担 TokenShift 的局部混合职责”，避免两个局部块叠加后无法归因。

### 6.2 WTConv 方案

`wt` 的查询路径为：

```text
Q -> Haar DWT(level=1) -> DWConv 3x3 on 4C sub-bands -> Haar IWT
  -> zero-initialized residual scale -> LN -> R / Kq / A
```

Haar 滤波器固定且正交，不学习 LL/LH/HL/HH 的手工权重，不增加频域损失或子带约束。残差尺度
`gamma=0`，所以 WTConvLite 初始严格为恒等；首步 `gamma` 有梯度，之后子带卷积再开始学习。
实现支持奇数高宽，DWT 前补齐，IWT 后裁回原尺寸。

## 7. 参数量

以下为当前代码实测参数量，网络口径与 v9 CoReg 一致；WKV 使用 legacy 或 CUDA 不影响这里的
结构比较：

| 模型 | 单个 QRB 参数 | 整网参数 | 相对 v9 CoReg |
|---|---:|---:|---:|
| CrossRWKV v8.8-B | 65,148 | 1,327,085 | −14,520 |
| v9 CoReg | 69,988 | 1,341,605 | 基线 |
| Core `current` | 65,376 | 1,327,769 | **−13,836** |
| Core `span` | 64,252 | 1,324,397 | **−17,208** |
| Core `omni` | 37,632 | 1,244,537 | **−97,068** |
| Core `wt` | 37,661 | 1,244,624 | **−96,981** |

`current` 的目标是先验证概念简化，不追求比 CrossRWKV 更少参数；它相对 CrossRWKV 多出的
Kq/A 被删除的 G 抵消后仅增加 684 个整网参数。若主实验成立，`span/wt/omni` 才用于进一步
判断局部混合器的精简空间。

## 8. 诊断量

训练日志新增：

- `A`：有符号 A 的均值；
- `|A|`：A 的平均绝对值，判断修正是否真正打开；
- `Kq:Kx`：K 的查询增量 RMS 与 Context 主体 RMS 之比。

模块内部还记录 A 的最小值、最大值、正值比例及 Kq/Kx 各自 RMS。它们只用于诊断，不进入
损失函数。若 A 均值接近零但 `|A|` 非零，表示不同位置执行了方向相反的修正，不能判定 A 冗余。

## 9. 实验顺序

所有实验继续使用 v9 CoReg 的数据、损失、SGRFv9 和训练超参数，只改变 QRB：

1. **Core-current**：首先与 v9 CoReg 做 30 epoch 主对照，判断删除 G/关系 MLP 后能否保留
   干净指标与 σ=0.05 鲁棒性；
2. **Core-span**：若主对照成立，测试单 SPAB 替代 Q OmniShift；
3. **Core-wt**：测试无手工子带约束的 WTConvLite；
4. **Core-omni**：作为最轻纯 TokenShift 对照。

不同时修改 SGRF、分支容量或损失，否则无法判断收益来自 QRB 还是恢复后端。

## 10. 配置与代码

| 文件 | 作用 |
|---|---|
| `models/golf_v7r/core_query_rwkv.py` | Core-QRB、WTConvLite 和四种 Q Mixer |
| `configs/v9/v9_core_qrb_current_30ep.yaml` | 主实验 |
| `configs/v9/v9_core_qrb_span_30ep.yaml` | SPAN 替换 |
| `configs/v9/v9_core_qrb_wt_30ep.yaml` | WTConv 替换 |
| `configs/v9/v9_core_qrb_omni_30ep.yaml` | OmniShift-only 对照 |
| `tests/test_core_query_rwkv.py` | 无 G、初始化、符号 A、梯度、Mixer 与 TCA 接线测试 |

旧 `q_mode: coreg` 和原配置不变，可继续加载旧 checkpoint；新结构使用 `q_mode: core`，避免
把结构不兼容的 checkpoint 误载入同一模式。

六组初步实验的完整启动顺序和 SGRF-S0/S1 配置见
[04-六组实验执行说明.md](./04-六组实验执行说明.md)。当前 Core-current 完成前不启动其余组。

## 11. Core-QRB 30ep 最终结果 (2026-11-06)

### 11.1 验证曲线

| 轮次 | PSNR | SSIM | LPIPS |
|:--:|:--:|:--:|:--:|
| 5 | 23.83 | 0.8231 | 0.3719 |
| 10 | 25.43 | 0.8325 | 0.3433 |
| 15 | 25.56 | 0.8385 | 0.3459 |
| 20 | 25.71 | 0.8388 | 0.3473 |
| 25 | 25.81 | 0.8390 | 0.3530 |
| 30 | 25.81 | 0.8390 | 0.3530 |

第 20 轮后基本平台化 (+0.10 dB / 10 轮)。best 在 ep25。

### 11.2 与基线对照 (v8.8-B 26.05/0.8432/0.3426; CoReg 26.01/0.8433/0.3504)

| 项 | v8.8-B | CoReg | Core-QRB | 差值 (对 CoReg) |
|---|---:|---:|---:|---:|
| PSNR | 26.05 | 26.01 | 25.81 | −0.20 |
| SSIM | 0.8432 | 0.8433 | 0.8390 | −0.0043 |
| LPIPS | 0.3426 | 0.3504 | 0.3530 | +0.0026 |
| 加噪 σ=0.05 | 14.70 | 16.52 | **16.99** | **+0.47** |
| 加噪 σ=0.02 | — | 21.96 | 22.34 | +0.38 |
| M 置零代价 | −0.31 | −0.45 | −0.17 | 更弱 |
| Param (M) | 1.327 | 1.342 | **1.328** | −0.014 |
| FLOPs (G) | 118.73 | 120.61 | 118.83 | −1.78 |
| Latency (ms) | 50.9 | 53.4 | 50.2 | −3.2 |

### 11.3 判读

1. **验证主指标略低** (−0.20 dB 对 CoReg)。第 20 轮后平台化，曲线形状正常。
2. **噪声鲁棒性最好** (σ=0.05: 16.99，全部版本最高；σ=0.02: 22.34 同样最高)。
3. **效率最好**：参数 1.328M (与 v8.8-B 持平，比 CoReg 少 14K)、延迟 50.2 ms (比 CoReg 快 3.2 ms)。
4. **M 路贡献下降**：融合权重 0.771 (CoReg 1.047)，置零代价 −0.17 (CoReg −0.45)。
   这与 CoReg 的 "G 冗余" 结论部分矛盾——删除 G 后 M 反而更弱。但在单变量比较中
   (CoReg→Core)，M 贡献确实下降，说明 G 路对 M 有支撑作用。
5. 门控全部未饱和 (最高 2.1%)，ρ 未使用 (Core 无 sigmoid 两阶段)，Kq/Kx 比值监控
   显示查询增量活跃 (终值约 1.2–1.5)。

### 11.4 结论

Core-QRB 定位：**效率与鲁棒性版本**。干净集 PSNR 比最优版本低 0.20 dB，但噪声鲁棒性
最高、延迟最低、参数最少。与 CoReg (鲁棒性次之、干净指标持平) 和 v8.8-B (干净指标与
效率最优、鲁棒性最弱) 形成三足。

是否作为主版本取决于对鲁棒性的权重；当前默认仍保留 v8.8-B 为干净指标基准、CoReg 为
综合基准。Core-QRB 的 span/wt/omni 变体 (第 9 节顺序) 暂不启动，等待 SGRFv10 结果
一并裁决。
