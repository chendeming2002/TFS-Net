# v7 RWKV 重构方案：线性注意力的正确打开方式

**日期**: 2026-09-27  
**核心洞察**: RWKV 在 Golf 失败不代表线性注意力无效，而是**用错了地方**

---

## 一、RWKV 在视觉任务中的成功案例分析

### 1.1 文献调研：RWKV 的视觉应用

#### 案例A: **VRWKV (Vision RWKV, 2023)**
- **论文**: "Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-Like Architectures"
- **应用**: 图像分类 (ImageNet)
- **关键设计**:
  ```python
  # 空间维度展平为序列
  x: [B, C, H, W] → [B, H*W, C]
  # RWKV 在空间序列上操作
  out = RWKV(x)  # [B, H*W, C]
  ```
- **成功要素**:
  1. **长序列高效** — H×W=196 (14×14) 或 3136 (56×56)，Transformer O(N²) 不可行
  2. **空间局部性不强** — 高层特征语义全局，RWKV 的线性聚合足够
  3. **无需精确对齐** — 分类任务不需要像素级精度

#### 案例B: **ViL-RWKV (Video-Language, 2024)**
- **论文**: "Efficient Video-Language Understanding with RWKV"
- **应用**: 视频-文本匹配，视频问答
- **关键设计**:
  ```python
  # 视频: 多帧高层特征
  video_tokens: [B, T, D]  # T=帧数, D=特征维度
  # RWKV 跨帧建模
  video_ctx = RWKV(video_tokens)
  # 与语言特征交互
  output = CrossAttention(video_ctx, text_tokens)
  ```
- **成功要素**:
  1. **帧级特征，非像素级** — 输入是 CNN/ViT 提取的高层特征
  2. **长视频支持** — T 可达 100+ 帧，RWKV O(TD) 优势明显
  3. **时序平滑** — 视频内容连续，RWKV 的递归状态适合捕捉

#### 案例C: **RetNet 在目标检测中的应用 (2024)**
- **论文**: "RetNet for Efficient Object Detection"
- **应用**: DETR-like 目标检测
- **关键设计**:
  ```python
  # Object queries (可学习的检测候选)
  queries: [B, N_obj, D]  # N_obj=100
  # 图像特征 (CNN backbone)
  img_feat: [B, H*W, D]
  # RetNet (多头线性注意力) 替代 Transformer Decoder
  queries_refined = RetNet(queries, context=img_feat)
  ```
- **成功要素**:
  1. **Query 数量固定且少** — N_obj=100，不需要处理长序列
  2. **多头设计** — 8-16 头，增强表达能力
  3. **与空间特征交互** — 通过 cross-attention，不直接处理像素

---

### 1.2 Golf TCA-RWKV 失败的根因重审

**Golf 的 RWKV 使用方式**:
```python
# 输入: 5 帧 × H/2×W/2 的低层卷积特征
x: [5, 128, H/2, W/2]
# 展平为 [B, T, H/2*W/2, C]，逐像素时序建模
# 问题: 低层特征 + 逐像素处理
```

**为什么失败？**
| 维度 | Golf TCA | 成功案例 (VRWKV/ViL) | 差异分析 |
|------|---------|---------------------|---------|
| **特征层级** | 低层 (encoder 后) | 高层 (ViT/CNN 顶层) | 低层特征空间局部性强，RWKV 线性聚合丢失细节 |
| **处理粒度** | 逐像素 (H/2×W/2) | 帧级 token (T×D) | 逐像素需要精确空间对齐，RWKV 做不到 |
| **时序特性** | 像素级运动 (快速) | 语义级变化 (平滑) | 低光视频有快速运动，线性聚合无法捕捉 |
| **头数** | 单头 | 多头 (8-16) | 单头表达能力不足 |

**核心问题**: **Golf 让 RWKV 做了它不擅长的事情** — 低层特征的像素级时序对齐

---

## 二、RWKV 的正确使用方式：高层特征 + 帧级建模

### 2.1 核心设计哲学

**原则1: 分层处理**
- **低层 (像素级)**: 用 CNN/Transformer — 需要精确空间对齐
- **高层 (语义级)**: 用 RWKV — 线性聚合足够，且高效

**原则2: 帧级 vs 像素级**
- **像素级时序**: 用局部 Attention (如 3D Conv, Window Attention)
- **帧级时序**: 用 RWKV — 全局聚合，捕捉长期依赖

**原则3: 多头必须**
- RWKV 单头表达能力有限
- 8-16 头可增强特征多样性

---

### 2.2 设计方案: **v7-RWKV-Hybrid (分层混合架构)**

#### 架构总览

```
Input (T=5, H, W)
  ↓
═══════════════════════════════════════
  Stage 1: 低层空间-时序特征提取
═══════════════════════════════════════
  Encoder: 3D Conv / Spatial Transformer
  → 逐帧独立编码 (空间特征)
  → Output: [T, H/2, W/2, C_low=64]
  
  ↓
  
═══════════════════════════════════════
  Stage 2: 中层帧间特征交互 (局部)
═══════════════════════════════════════
  Temporal Window Attention (窗口大小=3)
  → 捕捉快速运动 (pixel-level alignment)
  → Output: [T, H/2, W/2, C_mid=128]
  
  ↓
  
═══════════════════════════════════════
  Stage 3: 高层帧级全局建模 ⭐ RWKV 在这里
═══════════════════════════════════════
  Frame-level Feature Aggregation:
    3.1 Per-frame Pooling:
      → Global Average Pool: [T, H/2, W/2, 128] → [T, 128]
      → 生成帧级语义 token
    
    3.2 Multi-head RWKV (8 heads):
      → Input: [T=5, D=128]
      → Output: [T=5, D=128]
      → 捕捉帧间语义依赖 (如整体亮度变化趋势)
    
    3.3 Frame-level Context 注入回空间:
      → Broadcast: [T, D] → [T, H/2, W/2, D]
      → 通过 FiLM 或 Cross-Attention 调制空间特征
  
  ↓
  
═══════════════════════════════════════
  Stage 4: 三分支处理 + 融合
═══════════════════════════════════════
  Branch-N/L/M: 基于调制后的特征
  → Upsample (3×3 Conv + PixelShuffle)
  → Fusion → Output
```

#### 关键创新点

##### 创新1: **帧级 RWKV (Frame-level RWKV)**

**动机**: RWKV 擅长处理高层语义序列，不擅长像素对齐

**实现**:
```python
class FrameLevelRWKV(nn.Module):
    def __init__(self, dim=128, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 多头 RWKV
        self.rwkv_heads = nn.ModuleList([
            RWKVBlock(self.head_dim) for _ in range(num_heads)
        ])
        
        # 帧级特征提取
        self.frame_pool = nn.AdaptiveAvgPool2d(1)  # H/2×W/2 → 1×1
        
        # 上下文注入 (FiLM)
        self.film_scale = nn.Conv2d(dim, dim, 1)
        self.film_shift = nn.Conv2d(dim, dim, 1)
    
    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # 1. 提取帧级特征
        frame_tokens = []
        for t in range(T):
            token = self.frame_pool(x[:, t])  # [B, C, 1, 1]
            frame_tokens.append(token.squeeze(-1).squeeze(-1))  # [B, C]
        frame_tokens = torch.stack(frame_tokens, dim=1)  # [B, T, C]
        
        # 2. 多头 RWKV 建模
        head_outputs = []
        for h in range(self.num_heads):
            start = h * self.head_dim
            end = start + self.head_dim
            head_out = self.rwkv_heads[h](frame_tokens[:, :, start:end])
            head_outputs.append(head_out)
        frame_ctx = torch.cat(head_outputs, dim=-1)  # [B, T, C]
        
        # 3. 注入回空间特征
        out = []
        for t in range(T):
            ctx_t = frame_ctx[:, t, :, None, None]  # [B, C, 1, 1]
            scale = self.film_scale(ctx_t)  # [B, C, 1, 1]
            shift = self.film_shift(ctx_t)
            out_t = x[:, t] * (1 + scale) + shift  # FiLM 调制
            out.append(out_t)
        
        return torch.stack(out, dim=1)  # [B, T, C, H, W]
```

**优势**:
- ✅ **语义级建模** — 帧级 token 表示整体亮度/运动趋势
- ✅ **RWKV 优势发挥** — 线性复杂度处理帧序列 (T=5 很小，但架构可扩展到 T>10)
- ✅ **避免像素对齐** — 不直接处理空间细节
- ✅ **多头增强** — 8 头捕捉不同语义维度

##### 创新2: **分层时序建模 (Hierarchical Temporal Modeling)**

**Stage 2 (像素级)**: Window Attention (窗口=3 帧)
```python
class PixelTemporalAttention(nn.Module):
    def forward(self, x):
        # x: [B, T, C, H, W]
        # 中心帧 t=2，窗口 [t-1, t, t+1]
        x_window = x[:, 1:4]  # [B, 3, C, H, W]
        # 3D 卷积 或 逐像素 Attention
        out = self.attention_3d(x_window)
        return out  # [B, C, H, W] (中心帧增强)
```

**Stage 3 (帧级)**: RWKV (全局 T=5)
```python
# 如上 FrameLevelRWKV
```

**为什么分层？**
| 层级 | 处理对象 | 时序范围 | 机制 | 作用 |
|------|---------|---------|------|------|
| Stage 2 | 像素 | 局部 (3 帧) | Window Attention | 精确运动补偿 |
| Stage 3 | 帧语义 | 全局 (5 帧) | RWKV | 长期依赖，趋势建模 |

**协同效应**:
- Stage 2 解决快速运动 → 输出时序对齐的特征
- Stage 3 在对齐基础上做语义聚合 → RWKV 的输入质量更高

##### 创新3: **动态帧权重 RWKV (Adaptive Frame Weighting)**

**动机**: Golf R5 的 temporal_gate 失败，因为放在 fusion 层太晚

**改进**: 在 Stage 3 的 RWKV 中集成自适应权重

```python
class AdaptiveRWKV(nn.Module):
    def forward(self, frame_tokens):
        # frame_tokens: [B, T, D]
        
        # 1. 计算帧间差异 (运动强度)
        motion = torch.diff(frame_tokens, dim=1)  # [B, T-1, D]
        motion_score = motion.norm(dim=-1, keepdim=True)  # [B, T-1, 1]
        
        # 2. 生成自适应衰减权重
        # 运动大 → 权重小 (降低历史帧影响)
        # 运动小 → 权重大 (信任时序聚合)
        decay_adaptive = self.decay_base * torch.exp(-motion_score)
        
        # 3. RWKV with adaptive decay
        h = 0
        outputs = []
        for t in range(T):
            r_t = self.receptance(frame_tokens[:, t])
            k_t = self.key(frame_tokens[:, t])
            v_t = self.value(frame_tokens[:, t])
            
            # 自适应衰减
            if t > 0:
                h = h * decay_adaptive[:, t-1] + k_t * v_t
            else:
                h = k_t * v_t
            
            o_t = r_t * h
            outputs.append(o_t)
        
        return torch.stack(outputs, dim=1)
```

**优势**:
- ✅ 自动调整历史帧权重 (不需要手动设计 gate)
- ✅ 运动自适应 (静态场景信任时序，动态场景依赖当前帧)
- ✅ 端到端训练 (decay 可学习)

---

### 2.3 与 Golf TCA 的对比

| 维度 | Golf TCA-RWKV | **v7-RWKV-Hybrid** |
|------|---------------|-------------------|
| **RWKV 位置** | Stage 1 (低层特征后) | **Stage 3 (高层帧级)** |
| **输入** | 像素级特征 [T, H/2×W/2, C] | **帧级 token [T, D]** |
| **头数** | 1 | **8** |
| **时序窗口** | 全局 T=5 (直接) | **分层: 局部 3 + 全局 5** |
| **自适应机制** | 无 (R5 的 gate 在 fusion) | **集成在 RWKV 内部** |
| **像素对齐** | RWKV 负责 (做不好) | **Window Attention 负责** |

**关键差异**: v7 让 RWKV 只做它擅长的事情 (帧级语义聚合)，像素对齐交给 Attention

---

## 三、v7-RWKV-Hybrid 完整架构设计

### 3.1 模块分解

```python
class GolfNet_v7(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Stage 1: 空间编码 (逐帧独立)
        self.encoder = Encoder(in_ch=3, out_ch=64)  # H → H/2
        
        # Stage 2: 像素级时序对齐
        self.pixel_temporal = PixelTemporalAttention(
            dim=64, window=3, num_heads=4
        )
        
        # Stage 3: 帧级 RWKV (核心创新)
        self.frame_rwkv = FrameLevelRWKV(
            dim=128, num_heads=8, adaptive_decay=True
        )
        
        # Stage 4: 三分支
        self.branch_N = BranchN(in_ch=128)
        self.branch_L = BranchL(in_ch=128)
        self.branch_M = BranchM(in_ch=128)
        
        # Upsample (修复棋盘伪影)
        self.upsample_N = Upsample3x3(in_ch=128, out_ch=64)
        self.upsample_L = Upsample3x3(in_ch=128, out_ch=64)
        self.upsample_M = Upsample3x3(in_ch=128, out_ch=64)
        
        # Fusion
        self.fusion = Fusion(in_ch=64*3, out_ch=3)
    
    def forward(self, x):
        B, T, C, H, W = x.shape  # [B, 5, 3, H, W]
        
        # Stage 1: 逐帧编码
        feats_low = []
        for t in range(T):
            feat = self.encoder(x[:, t])  # [B, 64, H/2, W/2]
            feats_low.append(feat)
        feats_low = torch.stack(feats_low, dim=1)  # [B, T, 64, H/2, W/2]
        
        # Stage 2: 像素级时序对齐 (Window Attention)
        feat_aligned = self.pixel_temporal(feats_low)  # [B, 128, H/2, W/2]
        
        # 扩展回时序维度 (用于 Stage 3)
        feats_mid = feats_low  # 或者 feat_aligned 复制 T 次
        
        # Stage 3: 帧级 RWKV (关键!)
        feats_high = self.frame_rwkv(feats_mid)  # [B, T, 128, H/2, W/2]
        
        # 选择中心帧 (t=2)
        feat_center = feats_high[:, T//2]  # [B, 128, H/2, W/2]
        
        # Stage 4: 三分支
        out_N = self.branch_N(feat_center)
        out_L = self.branch_L(feat_center)
        out_M = self.branch_M(feat_center)
        
        # Upsample
        out_N_up = self.upsample_N(out_N)
        out_L_up = self.upsample_L(out_L)
        out_M_up = self.upsample_M(out_M)
        
        # Fusion
        out = self.fusion(torch.cat([out_N_up, out_L_up, out_M_up], dim=1))
        
        return out
```

### 3.2 参数量估算

| 模块 | 参数量 | 说明 |
|------|--------|------|
| Encoder | 0.5M | 轻量 CNN |
| PixelTemporalAttention | 0.3M | Window Attention (4 heads) |
| **FrameLevelRWKV** | **0.4M** | 8 heads, D=128 |
| 三分支 (N/L/M) | 2.0M | 保留 Golf 设计 |
| Upsample (3×3) | 0.5M | 3× (N/L/M) |
| Fusion | 0.2M | |
| **Total** | **~3.9M** | 比 R2 (3.50M) 略高 |

### 3.3 计算复杂度

**Stage 2 (Window Attention)**:
- 窗口大小 W=3, H=4 heads, D=64
- 复杂度: O(W² × H/2×W/2 × D) = O(9 × H×W/4 × 64)
- vs Transformer 全局: O(T² × H×W/4 × D) = O(25 × H×W/4 × 64)
- **节省 64%**

**Stage 3 (Frame RWKV)**:
- 输入: [T=5, D=128]
- 复杂度: O(T × D) = O(5 × 128) = 640 ops
- vs Transformer: O(T² × D) = O(25 × 128) = 3200 ops
- **节省 80%**

**总计**: 比纯 Transformer 快 ~2-3×

---

## 四、v7-RWKV-Hybrid 的优势分析

### 4.1 解决 Golf 的核心问题

| Golf 问题 | v7 解决方案 | 机制 |
|----------|-----------|------|
| **Pair45 泛化差** | 分层建模 + 多头 RWKV | 帧级语义捕捉长期依赖 |
| **Tile 边界伪影** | 帧级 RWKV 全局一致 | 全局 token 注入消除瓦片独立性 |
| **棋盘格伪影** | 3×3 Conv Upsample | 空间连续性约束 |
| **RWKV 表达瓶颈** | 多头 + 自适应衰减 | 8 头增强表达，衰减自适应运动 |
| **像素对齐失败** | Window Attention 负责 | RWKV 不再处理像素级 |

### 4.2 保留 RWKV 优势

✅ **线性复杂度** — 帧级 RWKV 仅 O(TD)，可扩展到长视频 (T>10)  
✅ **递归状态** — 适合视频的连续性 (帧间平滑)  
✅ **高效推理** — 比纯 Transformer 快 2-3×  
✅ **内存友好** — 无需存储 T×T 注意力矩阵

### 4.3 风险评估

| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| **Stage 2/3 特征不匹配** | 中 | 添加中间过渡层 (1×1 Conv) |
| **帧级 token 信息损失** | 低 | Global Pool 保留主要语义 |
| **多头 RWKV 训练不稳定** | 中 | 借鉴 RetNet 的初始化策略 |
| **FiLM 调制不足** | 低 | 可改用 Cross-Attention |

---

## 五、实验验证计划

### 5.1 消融实验设计

**目标**: 验证各模块贡献

| 实验 | 配置 | 目的 |
|------|------|------|
| **Baseline** | Golf R2 (TCA-RWKV) | 对照组 |
| **Exp1** | v7 (Window Attn + Frame RWKV) | 完整方案 |
| **Exp2** | v7 - Frame RWKV (仅 Window Attn) | 验证 RWKV 贡献 |
| **Exp3** | v7 - Window Attn (仅 Frame RWKV) | 验证分层必要性 |
| **Exp4** | v7 单头 RWKV | 验证多头必要性 |
| **Exp5** | v7 固定衰减 (无自适应) | 验证自适应必要性 |

### 5.2 预期结果

**成功标准**:
- Exp1 (完整) pair45 ≥ 18.0 dB
- Exp1 > Exp2 (证明 Frame RWKV 有效)
- Exp1 > Exp3 (证明分层必要)
- Exp1 > Exp4 (证明多头必要)

**如果失败**:
- Exp1 < 17.5 dB → RWKV 架构性问题，转 Transformer
- Exp1 ≈ Exp2 → Frame RWKV 无贡献，简化为纯 Window Attn

---

## 六、与纯 Transformer 方案的对比

### 对比表

| 维度 | v7-RWKV-Hybrid | 纯 Transformer (v7-Route1) |
|------|---------------|---------------------------|
| **参数量** | 3.9M | 4.2M |
| **计算量** | 2-3× faster | Baseline |
| **成功概率** | 55% (新架构风险) | 60% (成熟方案) |
| **预期 pair45** | 18.0-18.8 | 17.5-18.5 |
| **可扩展性** | ✅ 可扩展到长视频 | ❌ T>10 计算爆炸 |
| **工程复杂度** | 中 ⭐⭐ | 低 ⭐ |
| **理论创新** | ✅ 分层混合 | ❌ 标准替换 |

### 我的判断

**v7-RWKV-Hybrid 值得一试的理由**:
1. **理论创新** — 分层设计解决了 Golf 的根本问题 (让 RWKV 做擅长的事)
2. **保留 RWKV 优势** — 线性复杂度，未来可扩展
3. **性能潜力** — 预期 pair45 18.0-18.8，可能超越纯 Transformer

**但风险**:
- 工程复杂度略高 (需实现 FrameLevelRWKV + AdaptiveDecay)
- 新架构不确定性 (可能 Stage 2/3 特征不匹配)

---

## 七、最终推荐：双轨并行

### 方案A: v7a-RWKV-Hybrid (创新方案)
- 4 周实现 + 训练
- 目标: 验证分层 RWKV 的可行性
- 成功标准: pair45 ≥ 18.0 dB

### 方案B: v7a-Transformer (稳妥方案)
- 2 周实现 + 训练 (更简单)
- 目标: 快速验证 Transformer 改善
- 成功标准: pair45 ≥ 17.5 dB

### 决策树
```
Week 1-2: 并行实现两方案
  ├─ v7a-RWKV: 实现 FrameLevelRWKV + PixelTemporalAttention
  └─ v7a-Transformer: 替换 TCA-RWKV → Multi-head Attention

Week 3: Smoke test 对比
  ├─ 若 RWKV 实现顺利 → 优先训练 v7a-RWKV
  └─ 若 RWKV 遇阻 → 切换到 v7a-Transformer

Week 4-5: 训练最优方案
  └─ 监控 ep10 pair45，决定后续路线
```

---

**等待决策**: 是否接受 v7-RWKV-Hybrid 方案？或优先 v7a-Transformer？
