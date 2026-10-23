# 11-TemporalAggregator_G: 双流共享 top-down 的多尺度聚合

> 创建于 2026-10-23 (同日按六项裁决修订)
> 代码: `models/golf_v7r/temporal_aggregator_g.py` (已实现, 冒烟测试通过)
> 命名: G = 改进 (裁决 5); 接入后登记进 `08-命名对照.md`
> 状态: **已接入并开始训练** (2026-10-22)。`golfnet_v8.py` 的 `summary_type` 新增
> `"g"` 分支, 配置 `configs/v8_5/g_sgrf10_30ep.yaml`, 与 SGRFv10 (逐段约束) 组合,
> 输出 `outputs/v8_5_g_sgrf10_30ep`。

## 一、动机: lite 版的三个结构问题

本设计直接回应 2026-10-23 理论建模审查中确认的事实:

1. **邻帧信息只在一处进入。** lite 唯一的跨帧操作是 F3 尺度 (H/4) 上 2K 个非中心帧的
   算术平均; F2、F1 尺度只做"上采样 + 与中心帧特征拼接融合"。因此 `F1_multi` 是
   中心帧锚定 + 粗尺度邻帧均值调制的学习式融合, 不是任何意义上的时序平均 ——
   论文 §3.4 "F1_multi approximates the stable content" 的前提没有结构兑现。
2. **编码器为邻帧算出的 F1/F2 特征是死计算。** lite 只消费邻帧的 F3 特征。
3. **F1_single 不经 top-down 路径。** lite 与 full 的 `F1_single = f1_seq[:, center]`
   都是编码器原始输出, 与论文 §3.3 "A bypass carries the center frame through the
   same top-down path" 不符。

## 二、规格 (用户指导, 2026-10-23)

1. 记号: 每尺度 T=2K+1 帧特征 `fi_seq`; 中心帧特征 `fi_t`; 邻帧集合 (2K 帧) `fi_Ω`。
2. 上采样 Conv+PixelShuffle 打包成 **`Up` 组件**。
3. 每一个尺度, 邻帧特征集合加和平均得 `fi_agg`。
4. 在 F1、F2 尺度, `fi_agg`、`fi_t` 分别与上采样过的上一尺度 `_agg`、`_t` 做
   fuse (cat + Conv), 最终在 F1 尺度输出 `F1_single` 与 `F1_multi`。

## 三、六项裁决 (用户, 2026-10-23)

| # | 问题 | 裁决 | 实现落点 |
|---|---|---|---|
| 1 | fuse 是否加非线性 | **搬运 lite 的 fuse 非线性** | fuse = Conv3x3 + LayerNorm2d + SiLU |
| 2 | frame_encoder 去留 | **不动 frame_encoder**; 重构范围 = frame_encoder 与 F1_single/F1_multi 产生之间的部分 | frame_encoder (F3 逐帧 RWKV, 全部 T 帧) 原样保留, 两股都从它的输出分叉 |
| 3 | 双路参数独立还是共享 | **共享** | 同一组 Up/fuse, agg 股与 t 股各前向一次 |
| 4 | multi 路均值是否含中心帧 | **含中心帧** | `fi_agg = fi_seq.mean(dim=1)`, 全 T 帧 |
| 5 | G 的含义 | **改进** | 命名登记待接入后写入 08-命名对照.md |
| 6 | 接入与训练 | **只实现不启动** | 不改 golfnet_v8.py, 不建 config |

注 (裁决 2 的答复): frame_encoder 本来就是 TemporalAggregatorLite 的成员
(`self.frame_encoder = RWKVEncoderBlock(C3)`), 在 F3 尺度对全部 T 帧逐帧编码、
位于聚合之前; G 中位置与行为不变。

## 四、结构

```
f1_seq [B,T,C1,H,W]   f2_seq [B,T,C2,H/2,W/2]   f3_seq [B,T,C3,H/4,W/4]
                                                  │ frame_encoder (逐帧 RWKV, 原样保留)
                                                  ▼
                                               f3_enc [B,T,C3,H/4,W/4]
每尺度拆两股 (split_streams):
   fi_t   = fi[:, center]            (F3 取 f3_enc 的中心帧)
   fi_agg = fi.mean(dim=1)           (全 T 帧均值, 含中心, 裁决 4)

共享 top-down (同一组参数走两遍, 裁决 3):
   _top_down(f3, f2, f1):
     s = Up3to2(f3)                          Up = Conv3x3(→4×) + PixelShuffle(2)
     s = fuse2(cat[s, f2])                   fuse = Conv3x3 + LN + SiLU (裁决 1)
     mid = s                                 (F2 级中间量)
     s = Up2to1(s)
     s = fuse1(cat[s, f1])
     return s, mid

   F1_multi,  _           = _top_down(f3_agg, f2_agg, f1_agg)
   F1_single, F2_single   = _top_down(f3_t,   f2_t,   f1_t)
```

cat 顺序为 `[上采样上一级, 本级特征]`, 与 lite 一致。

## 五、结构性质与冒烟结果 (CPU, 2026-10-23, 全部通过)

| 检查 | 结果 |
|---|---|
| 输出形状 | F1_multi/F1_single [B,C1,H,W], F2_single [B,C2,H/2,W/2] ✓ |
| **参数量 = lite** | 662,948 == 662,948 ✓ (裁决 1+2+3 的推论: G 与 lite 的模块集合完全相同, 只是接线不同) |
| 性质 1: 扰动邻帧 | F1_single / F2_single **逐位不变**; F1_multi 响应 ✓ |
| 性质 2: 扰动中心帧 | F1_multi **也响应** (均值含中心, 与旧稿"multi 不含中心"相反) ✓ |
| 性质 3: T 帧相同输入 | \|F1_multi − F1_single\|max = 1.07e-06 ≈ 0 (同路同输入, 共享参数的直接推论) ✓ |
| 反传 | 正常。首步 backward 有 3 个参数梯度为零 (`frame_encoder.head.post_norm.*`, `channel_mix.0.weight`) —— **预期行为, 非死参数**: channel_mix 末层零初始化 (`_init_weights`) 挡住上游首步梯度; 末层自身有梯度, 模拟一步更新后第二步三个参数梯度全部恢复 (实测 4.26 / 4.60 / 112.3)。真实 checkpoint (best.pth, ep25) 佐证: post_norm.weight 偏离初始 1.0 达 0.95, bias 偏离 0 达 0.52, channel_mix.2.weight 从全零长到 \|max\|=0.199, 训练中被积极更新。lite 上行为完全相同 |
| checkpoint 兼容 | state_dict 键名与 lite 逐一对齐 (frame_encoder / up3_to_2.proj / fuse2.* / up2_to_1.proj / fuse1.*) → 可从已训练 lite checkpoint **热启动**聚合器权重 |

## 六、与 lite / full 对照

| 项 | full | lite | **G** |
|----|------|------|-------|
| frame_encoder (F3 逐帧 RWKV) | 有 | 有 | **有 (原样)** |
| 跨帧聚合 | 仅 F3, 学习式帧权重加权 (含中心) | 仅 F3, 非中心帧均值 | **每尺度, 全 T 帧均值 (含中心)** |
| multi 路是否拼接中心帧 | 每级拼接 | 每级拼接 | **否 (中心只经均值进入)** |
| top-down | 1 次 | 1 次 | **同一组参数 2 次 (agg 股 + t 股)** |
| F1_single | 原始 f1_t | 原始 f1_t | **t 股 top-down 输出** (论文 §3.3 描述成立) |
| 邻帧 F1/F2 特征 | 死计算 | 死计算 | **被消费** |
| 聚合器参数 | — | 662,948 | **662,948 (相同)** |

## 七、参数与计算预算

- **参数**: 与 lite 完全相同 (662,948), 全模型仍 ≈1.2354M; Table 2 的参数量问题
  (1.175M vs 实测 1.2354M) 与本设计无关, 另行处理。
- **计算**: top-down 四个卷积模块从走 1 次变为走 2 次 → 聚合器卷积 FLOPs 约 ×2;
  新增 F1/F2 两处全帧均值 (reduction, 可忽略); frame_encoder 不变。整模型延迟影响
  待接入后实测 (lite 基线: 编码器合批后整网前向 36ms@b1-256)。
- 邻帧的 F1/F2 编码特征从死计算变为被消费 —— 编码器 FLOPs 不变, 利用率提高。

## 八、语义后果 (对 TCA 与论文的影响)

1. **Q_N 的语义首次获得结构支撑。** F1_multi = P(全帧均值股), F1_single = P(中心帧股),
   P 为同一非线性映射 (共享参数) → `Q_N = F1_single − F1_multi = P(center) − P(mean)`
   是"同一镜头下的中心帧 vs 窗口均值"之差, 论文 §3.4 的 instantaneous residual
   论证从"假设"变为"结构事实 + 近似误差"。
2. **仍是近似, 三个误差源要在论文如实写**: (a) 特征空间差值不继承像素空间零均值性
   (P 非线性); (b) 无对齐, 位移信号项 d·|∇X| 仍进入差值 —— 但均值含中心后差值中
   该项被稀释约 (1−1/T) 倍 (T=5 时 ×0.8); (c) 各尺度独立均值, 无跨尺度时序交互。
3. **F1_single 语义变化**: 不再是"编码器原始中心帧特征", 而是"中心帧经同一 top-down
   路径的输出"。下游 TCA 中 M 路的 Q/KV/残差基、L 路的 KV/残差基内容随之变化
   (Table 1 的路由关系不变)。M 路受此影响最大 —— 它原本拿到的是最"生"的中心帧特征。
4. **F2_single 语义变化**: 从原始 f2_t 变为 t 股 F2 级中间量; `single_frame=true`
   下仍无消费者 (接口兼容保留)。

## 九、待办 (全部等用户指令, 裁决 6)

1. **接入**: `golfnet_v8.py` 的 `summary_type` 增加 `"g"` 分支 (构造参数与 lite 同签名,
   一行分支即可); 补单测 (分离性质 1–3)。
2. **对照实验编排** (接入后): 单变量 = 聚合器, 基线 `a_lite_15ep` (24.12 / 24.65 dB
   @ep10/15)。可选热启动臂 (从 lite checkpoint 载聚合器权重, 键名已对齐) 与随机初始化臂,
   两臂基线必须配对, 避免初始化差异混入归因。
3. **数值检查** (独立于接入): GT 相邻帧实测 d·|∇X| vs σ/√T, 为 §3.4 的条件表述定量。

已完成的接入事项 (2026-10-23):
- `golfnet_v8.py` 的 `summary_type` 新增 `"g"` 分支, 默认行为不变;
- 配置 `configs/v8_5/g_sgrf10_30ep.yaml`, 与 SGRFv10 (逐段约束) 组合;
- 冒烟通过, 热启动 lite 聚合器权重检查通过 (键名逐一对齐);
- 30 轮训练已启动, 输出 `outputs/v8_5_g_sgrf10_30ep`。
4. **文档联动**: 采纳后登记 `08-命名对照.md`; 论文 §3.3 按 G 重写 (frame_encoder +
   每尺度均值 + 共享 top-down 双股), §3.4 补近似误差源; 不采纳则本文档存档为否决记录。
