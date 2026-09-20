"""
TCA-RWKV: Temporal Cross-frame Attention with RWKV backend
===========================================================
Foxtrot 的核心创新模块。三路结构化查询从共享特征中显式解耦出三种退化子空间。

设计理论 (来自 TSDR 分解):
  - 成像噪声: 帧间 i.i.d. → 均值分量 (高熵/均匀注意力)
  - 光照扰动: 帧间强相关慢变 → 低频时序趋势 (全局池化/低温)
  - 运动伪影: 帧间结构位移 → 对齐残差 (稀疏局部注意)

RWKV 后端 (注意力机制的 RWKV 实现):
  - 继承 pure_rwkv_sace.py 的 SpatialWKV2D + BiWKV
  - 三路查询 Q_N / Q_L / Q_M 各自生成专属 key/value
  - 配合先验约束 (entropy reg / global pool / top-k sparse)

正交约束:
  L_ortho = ||F_N^T F_L||_F² + ||F_L^T F_M||_F² + ||F_N^T F_M||_F²
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List

from models.modules.blocks import LayerNorm2d, NAFBlock
from models.modules.pure_rwkv_sace import BiWKV, MVCShift


# ============================================================
# FastBiWKV — 大 chunk 变体 (python 循环开销优化)
# ============================================================
class FastBiWKV(BiWKV):
    """BiWKV with larger chunk size: 16384 tokens / 256 = 64 次 python 迭代 → /1024 = 16 次.
    数学完全一致 (chunk 递推公式与长度无关), 仅减少 E-core taskset 下的 python 开销.
    实测 Foxtrot 3 头 × 4 方向 × 双向 = 24 次调用/前向, 1536 → 384 次迭代."""

    CHUNK = 1024

    @staticmethod
    def _scan_cumsum(ek, ekv, u_coef, ew_pow):
        CHUNK = FastBiWKV.CHUNK
        B, L, C = ek.shape
        out = torch.zeros(B, L, C, device=ek.device)
        state_num = torch.zeros(B, 1, C, device=ek.device)
        state_den = torch.zeros(B, 1, C, device=ek.device)
        # 末尾补一位避免 ew_pow[:, 1:cs+1] 越界 (当 cs == L 时 1:cs+1 越界)
        ew_pow_ext = torch.cat([ew_pow, ew_pow[:, -1:]], dim=1)  # (1, L+1, C)
        for s in range(0, L, CHUNK):
            e = min(s + CHUNK, L)
            cs = e - s
            ek_c, ekv_c = ek[:, s:e], ekv[:, s:e]
            S_loc = (ekv_c / ew_pow[:, :cs].clamp(min=1e-12)).cumsum(dim=1) * ew_pow[:, :cs]
            D_loc = (ek_c  / ew_pow[:, :cs].clamp(min=1e-12)).cumsum(dim=1) * ew_pow[:, :cs]
            decay_state = ew_pow_ext[:, 1:cs+1]   # 用 ext 避免越界
            S = S_loc + state_num * decay_state
            D = D_loc + state_den * decay_state
            out[:, s:e] = (u_coef * ekv_c + S) / (u_coef * ek_c + D + 1e-8)
            state_num = ew_pow_ext[:, cs:cs+1] * state_num + S_loc[:, -1:]
            state_den = ew_pow_ext[:, cs:cs+1] * state_den + D_loc[:, -1:]
        return out


# ============================================================
# RWKV-based 空间注意力头 (单路)
# ============================================================
class RWKVSpatialHead(nn.Module):
    """单路 RWKV 空间注意力，用于 TCA 三路查询之一。
    
    基于 BiWKV (双向 WKV) + 4 方向扫描
    每路独立 K/V 投影，共享 BiWKV 参数
    """
    
    def __init__(self, channels: int, num_directions: int = 4):
        super().__init__()
        assert channels % num_directions == 0
        self.channels = channels
        self.num_dirs = num_directions
        self.head_dim = channels // num_directions
        
        # 查询/键/值投影
        self.proj_q = nn.Linear(channels, channels, bias=False)
        self.proj_k = nn.Linear(channels, channels, bias=False)
        self.proj_v = nn.Linear(channels, channels, bias=False)
        self.proj_out = nn.Linear(channels, channels, bias=False)
        
        self.pre_norm = nn.LayerNorm(channels)
        self.post_norm = nn.LayerNorm(channels)
        
        # BiWKV for each direction (FastBiWKV: chunk=1024 降低 python 循环开销)
        self.bi_wkv_list = nn.ModuleList([
            FastBiWKV(self.head_dim) for _ in range(num_directions)
        ])
        
        # R4-FIX (双零死锁): 原 `nn.init.zeros_(self.proj_out.weight)` 与
        # TCA_RWKV 的 scale_N/L/M (零初始化) 形成【乘法双零死锁】:
        #   attn 输出 = proj_out(...) × scale ;  scale=0 且 proj_out=0
        #   → 梯度链需要 scale≠0 才能流入 proj_out, 又需要 proj_out≠0 才能流入 scale
        #   → 两者梯度精确为 0, 永久冻结 (实测 298,752 参数 / 8.4% 全死)
        #   对比: NAFBlock 的 gamma=0 是【单零】(conv 非零), 可正常解锁 ✓
        #   对比: TBC1B 因 channel_mix 有非零 bias 旁路, 侥幸逃逸 ✓
        # 修复: 改为标准小初始化 (与 proj_q/k/v 同量级), 保留训练稳定性同时解锁梯度
        nn.init.uniform_(self.proj_out.weight,
                         -0.5 / math.sqrt(channels),
                          0.5 / math.sqrt(channels))
        nn.init.uniform_(self.proj_k.weight, 
                         -0.05 / math.sqrt(channels), 
                          0.05 / math.sqrt(channels))
        nn.init.uniform_(self.proj_q.weight, 
                         -0.5 / math.sqrt(channels), 
                          0.5 / math.sqrt(channels))
        nn.init.uniform_(self.proj_v.weight, 
                         -0.5 / math.sqrt(channels), 
                          0.5 / math.sqrt(channels))
        # 扫描索引缓存: 对角扫描坐标与逆索引按 (H,W,device) 缓存, 避免 python 循环逐帧重建
        self._idx_cache = {}
    
    def _scan_indices(self, H: int, W: int, anti: bool, device: torch.device) -> torch.Tensor:
        key = (H, W, anti, str(device))
        if key not in self._idx_cache:
            coords, used = [], set()
            if anti:
                for s in range(H + W - 1):
                    for i in range(max(0, s - W + 1), min(s + 1, H)):
                        j = W - 1 - (s - i)
                        if 0 <= j < W:
                            ij = i * W + j
                            if ij not in used:
                                coords.append(ij)
                                used.add(ij)
                for ij in range(H * W):
                    if ij not in used:
                        coords.append(ij)
                coords = coords[:H * W]
            else:
                for s in range(H + W - 1):
                    for i in range(max(0, s - W + 1), min(s + 1, H)):
                        coords.append(i * W + (s - i))
            self._idx_cache[key] = torch.tensor(coords, dtype=torch.long, device=device)
        return self._idx_cache[key]
    
    def _inv_indices(self, H: int, W: int, anti: bool, device: torch.device) -> torch.Tensor:
        key = (H, W, anti, str(device), 'inv')
        if key not in self._idx_cache:
            idx = self._scan_indices(H, W, anti, device)
            inv = torch.zeros(H * W, dtype=torch.long, device=device)
            inv.scatter_(0, idx, torch.arange(H * W, device=device))
            self._idx_cache[key] = inv
        return self._idx_cache[key]
    
    @staticmethod
    def _scan_h(x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)
    
    @staticmethod
    def _scan_v(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 3, 2).flatten(2).transpose(1, 2)
    
    def _scan_d1(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        idx = self._scan_indices(H, W, anti=False, device=x.device)
        return x.flatten(2)[:, :, idx].transpose(1, 2)
    
    def _scan_d2(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        idx = self._scan_indices(H, W, anti=True, device=x.device)
        return x.flatten(2)[:, :, idx].transpose(1, 2)
    
    def _get_scan_fns(self):
        return [self._scan_h, self._scan_v, self._scan_d1, self._scan_d2]
    
    def _inv_scan(self, scan_fn, B, C, H, W, device):
        # h/v 扫描是恒等排列的 flatten; 对角扫描走缓存逆索引
        name = scan_fn.__name__
        if name in ('_scan_h', '_scan_v'):
            key = (H, W, name, str(device), 'inv')
            if key not in self._idx_cache:
                self._idx_cache[key] = torch.arange(H * W, dtype=torch.long, device=device)
            return self._idx_cache[key]
        anti = (name == '_scan_d2')
        return self._inv_indices(H, W, anti=anti, device=device)
    
    def forward(self, center_feat: torch.Tensor,
                context_feat: torch.Tensor,
                film: tuple = None) -> torch.Tensor:
        """
        center_feat:  (B, C, H, W) — 中心帧特征，生成 Q
        context_feat: (B, C, H, W) — K/V 来源 (R4: 全时序拼接投影)
        film:         (gamma, beta) 可选, 均为 (B, C, 1, 1), R4-C 退化感知调制

        R4-C FiLM 修复说明:
          原 R3-C 把 per-channel gate 乘在 Q 上、且在 pre_norm(LN) 之前:
            1) 与 proj_out 的双零死锁叠加 → 梯度精确为 0 (死模块)
            2) LN 对"全局标量"缩放严格不变 (LN(αx)≡LN(x)), 即使解锁也会被部分抵消
          修复: 标准 FiLM per-channel scale+shift 施加在【注意力输出上、post_norm 之前】:
            out = out * (1 + γ) + β   (γ,β 形状 (B,C,1,1), 零初始化 → 初始恒等)
          该位置无 LN 阻隔、无乘零链, 梯度必然非零。

        Returns: (B, C, H, W)
        """
        B, C, H, W = center_feat.shape
        N = H * W

        # Flatten → token sequence
        q_tokens = center_feat.flatten(2).transpose(1, 2)   # (B, N, C)
        kv_tokens = context_feat.flatten(2).transpose(1, 2) # (B, N, C)

        q_tokens = self.pre_norm(q_tokens)
        kv_tokens = self.pre_norm(kv_tokens)

        q = self.proj_q(q_tokens)  # (B, N, C)
        k = self.proj_k(kv_tokens)
        v = self.proj_v(kv_tokens)

        scan_fns = self._get_scan_fns()
        heads = []
        for i, scan_fn in enumerate(scan_fns):
            c0, c1 = i * self.head_dim, (i + 1) * self.head_dim
            k_head = k[:, :, c0:c1]
            v_head = v[:, :, c0:c1]
            k_2d = k_head.transpose(1, 2).reshape(B, self.head_dim, H, W)
            v_2d = v_head.transpose(1, 2).reshape(B, self.head_dim, H, W)
            k_seq = scan_fn(k_2d)
            v_seq = scan_fn(v_2d)
            wkv_seq = self.bi_wkv_list[i](k_seq, v_seq, total_tokens=N)
            inv_idx = self._inv_scan(scan_fn, B, self.head_dim, H, W, center_feat.device)
            heads.append(wkv_seq[:, inv_idx])

        wkv = torch.cat(heads, dim=-1)  # (B, N, C)
        out = torch.sigmoid(q) * wkv
        out = self.proj_out(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)   # (B,C,H,W)

        # R4-C: FiLM (per-channel scale+shift), 施加在注意力输出上、post_norm 之前
        if film is not None:
            gamma, beta = film
            out = out * (1.0 + gamma) + beta

        out = out.flatten(2).transpose(1, 2)   # back to (B,N,C)
        out = self.post_norm(out)
        return out.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# 三路结构化注意力先验 (R4 重写: 物理意义明确 + 非退化)
# ============================================================
class StructuredPriorWrapper(nn.Module):
    """为三路注意力注入退化统计先验 (对应 TSD-Foxtrot.md §3.2 设计).

    R4 重写动机 (Golf-R3 的三个先验全部退化/无效):
      - prior_N: feat + 0.1*mean_ctx 只是"加常数", 无熵正则语义
      - prior_L: feat*gate + feat*(1-gate) ≡ feat (代数恒等, 完全 no-op)
      - prior_M: 仅 plus 差分, 无 top-k 稀疏/位置偏置

    物理对应 (TSDR 三源分解):
      N (成像噪声): 帧间 i.i.d. → 时序均值是最优估计器 → 熵正则鼓励注意力均匀
      L (光照扰动): 帧间强相关慢变 → 时间轴 DCT 仅保留低频 + 空间平滑
      M (运动伪影): 帧间结构位移、局部 → top-k 稀疏 + 位置偏置 (相邻帧同位置)

    每个 forward 返回 (enhanced_feat, prior_loss):
      prior_loss 是可在总损失中最小化的正则项 (N: 负熵, L: 时间 TV, M: L1 稀疏)
    """

    def __init__(self, channels: int, branch_type: str):
        super().__init__()
        assert branch_type in ('N', 'L', 'M')
        self.branch_type = branch_type

        if branch_type == 'N':
            # 噪声: 时序均值投影 (可学习残差权重)
            # 非零初始化 (R2 风格 0.1): 提供对称破缺, 否则三分支初始恒等 →
            #   ortho 落在鞍点 (grad≈1e-8) 无法下降 (实测 R4 卡在 3.000)
            self.mean_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        elif branch_type == 'L':
            # 光照: 全局长程池化门控 (空间低频) — 用大核深度可分离卷积近似低通
            self.lowpass = nn.Sequential(
                nn.Conv2d(channels, channels, 7, 1, 3, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.Sigmoid(),
            )
            self.lp_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        else:  # M
            # 运动: 差分 (位移) 投影 (非零初始化 0.5, 同 R2)
            self.diff_proj = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
                nn.GELU(),
            )
            self.diff_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.5))

    def forward(self, feat: torch.Tensor,
                center: torch.Tensor,
                context: torch.Tensor,
                seq_mean: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feat:      (B,C,H,W) — 该路注意力输出
        center:    (B,C,H,W) — 中心帧特征
        context:   (B,C,H,W) — 该路时序上下文
        seq_mean:  (B,C,H,W) — 全时序均值 (仅 N 路用, 提供"平均器"信号)

        Returns: (enhanced_feat, prior_loss)
        """
        if self.branch_type == 'N':
            # 噪声: feat + scale * (时序均值)  — 强化 i.i.d. 平均分量
            src = seq_mean if seq_mean is not None else context
            enhanced = feat + self.mean_scale * src
            # 负熵正则 (尺度不变): 用【余弦相似度】而非 MSE
            #   原 F.mse_loss(feat, src) 随 feat 量级【平方】增长 → 正反馈发散
            #   (实测 L_prior 1.15 → 587, total_loss 1.2 → 8.9)
            #   余弦相似度 ∈ [-1,1] 且对尺度不变 (k^0), 有界 → 无发散风险
            f_flat = feat.flatten(1)
            s_flat = src.detach().flatten(1)
            cos = F.cosine_similarity(f_flat, s_flat, dim=1, eps=1e-6)
            prior_loss = (1.0 - cos).mean()   # ∈ [0,2], 一致时→0
            return enhanced, prior_loss

        elif self.branch_type == 'L':
            # 光照: feat + scale * 低通(feat) — 强化低频慢变分量
            lp = self.lowpass(feat)
            enhanced = feat + self.lp_scale * lp
            # 低频约束 (尺度不变): 归一化 TV = TV(feat) / rms(feat)
            #   rms 分母 detach, 避免梯度通过分母"刷分"
            gx = (feat[:, :, :, 1:] - feat[:, :, :, :-1]).abs().mean()
            gy = (feat[:, :, 1:, :] - feat[:, :, :-1, :]).abs().mean()
            rms = feat.pow(2).mean().sqrt().detach() + 1e-6
            prior_loss = (gx + gy) / rms
            return enhanced, prior_loss

        else:  # M
            # 运动: feat + scale * diff_proj(center-context) — 强化位移分量
            diff = center - context
            enhanced = feat + self.diff_scale * self.diff_proj(diff)
            # 稀疏正则 (尺度不变): L1/rms 比值 ∈ (0,1]
            #   稀疏信号(少数大峰) → L1/rms 小; 稠密信号 → 趋近 1
            #   最小化 → 鼓励运动响应稀疏化 (运动是局部现象)
            d = self.diff_proj(diff)
            l1 = d.abs().mean()
            rms = d.pow(2).mean().sqrt().detach() + 1e-6
            prior_loss = l1 / rms
            return enhanced, prior_loss


# ============================================================
# TCA-RWKV 主模块
# ============================================================
class TCA_RWKV(nn.Module):
    """TCA-RWKV: Temporal Cross-frame Attention with RWKV backend
    
    三路结构化查询 + RWKV 空间注意力 + 先验约束
    
    Args:
        channels: 特征通道数 (对应 SharedEncoder.base_channels * 2, 即 H/2 分辨率)
        num_frames: 输入帧数 (默认5)
    
    Returns dict:
        F_N: (B, C, H, W) — 噪声分量特征
        F_L: (B, C, H, W) — 光照分量特征
        F_M: (B, C, H, W) — 运动分量特征
        var_map: (B, 1, H, W) — 帧间方差图 (供 Branch-N 使用)
        ortho_loss: scalar — 正交约束损失
    """
    
    def __init__(self, channels: int = 64, tca_channels: int = 128,
                 num_heads: int = 4, num_blocks: int = 6, num_frames: int = 5,
                 f3_channels: int = 128, use_f3_film: bool = True):
        super().__init__()
        self.channels = channels
        self.tca_channels = tca_channels
        self.num_frames = num_frames
        self.center_idx = num_frames // 2
        self.use_f3_film = use_f3_film
        
        # 输入投影: encoder 通道 → TCA 通道
        self.in_proj = nn.Sequential(
            nn.Conv2d(channels, tca_channels, 1, bias=True),
            LayerNorm2d(tca_channels),
        )
        channels = tca_channels  # 内部使用 tca_channels
        
        # R4-C (FiLM 修复): F3 全局描述子 → per-channel FiLM (scale+shift) × 3 路
        # 输出 6×channels: [γ_N, β_N, γ_L, β_L, γ_M, β_M], 每路 (B,C)
        # 零初始化 → 初始恒等 (out*(1+0)+0 = out), 但梯度非零 (施加在注意力输出上)
        if use_f3_film:
            self.f3_film = nn.Linear(f3_channels, 6 * channels, bias=True)
            nn.init.zeros_(self.f3_film.weight)
            nn.init.zeros_(self.f3_film.bias)
        
        # MVC-Shift 预处理 (时序 token shift)
        self.mvc_shift = MVCShift(channels)
        
        # 三路查询 MLP (共享输入，独立投影)
        self.query_N = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.query_L = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.query_M = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        
        # 三路 RWKV 空间注意力头
        self.attn_N = RWKVSpatialHead(channels)
        self.attn_L = RWKVSpatialHead(channels)
        self.attn_M = RWKVSpatialHead(channels)

        # R4: 共享 K/V 投影 — Concat_time(F_{t±i}) (T*C → C)
        # 对应 TSD-Foxtrot.md §3.2 "K = V = Concat_time(F_{t±i})"
        self.kv_proj = nn.Sequential(
            nn.Conv2d(num_frames * channels, channels, 1, bias=True),
            LayerNorm2d(channels),
        )
        
        # 三路先验约束包装
        self.prior_N = StructuredPriorWrapper(channels, 'N')
        self.prior_L = StructuredPriorWrapper(channels, 'L')
        self.prior_M = StructuredPriorWrapper(channels, 'M')
        
        # 各路 LayerScale (零初始化，训练稳定)
        self.scale_N = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.scale_L = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.scale_M = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
        # 时序特征聚合 (得到上下文 context 送入 RWKV)
        self.temporal_agg = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GELU(),
        )
        
        # 输出归一化
        self.out_norm_N = LayerNorm2d(channels)
        self.out_norm_L = LayerNorm2d(channels)
        self.out_norm_M = LayerNorm2d(channels)
    
    def _temporal_mean_context(self, feats: torch.Tensor) -> torch.Tensor:
        """帧间均值上下文 (供 Branch-N 使用)
        
        feats: (B, T, C, H, W)
        Returns: (B, C, H, W)
        """
        return feats.mean(dim=1)
    
    def _temporal_smooth_context(self, feats: torch.Tensor) -> torch.Tensor:
        """低频时序上下文 (供 Branch-L 使用): 全局均值 + 低频滤波
        
        feats: (B, T, C, H, W)
        Returns: (B, C, H, W)
        """
        mean_feat = feats.mean(dim=1)  # (B, C, H, W)
        # 空间大核池化模拟低通滤波
        B, C, H, W = mean_feat.shape
        smooth = F.avg_pool2d(mean_feat, kernel_size=7, stride=1, padding=3)
        return smooth
    
    def _temporal_diff_context(self, feats: torch.Tensor, center_idx: int) -> torch.Tensor:
        """运动差分上下文 (供 Branch-M 使用)
        
        feats: (B, T, C, H, W)
        Returns: (B, C, H, W) — 中心帧 vs 邻帧的最大差分
        """
        center = feats[:, center_idx]  # (B, C, H, W)
        diffs = []
        for t in range(feats.shape[1]):
            if t != center_idx:
                diff = (center - feats[:, t]).abs()
                diffs.append(diff)
        # 取最大差分 (运动区域响应最强)
        max_diff = torch.stack(diffs, dim=0).max(dim=0).values
        return max_diff
    
    def _compute_var_map(self, feats: torch.Tensor) -> torch.Tensor:
        """帧间方差图 (暗区噪声大)
        
        feats: (B, T, C, H, W)
        Returns: (B, 1, H, W)
        """
        var = feats.var(dim=1, unbiased=False)  # (B, C, H, W)
        return var.mean(dim=1, keepdim=True)    # (B, 1, H, W) — 通道均值方差
    
    def _ortho_loss(self, F_N: torch.Tensor, 
                    F_L: torch.Tensor, 
                    F_M: torch.Tensor) -> torch.Tensor:
        """正交约束损失
        
        L_ortho = ||F_N^T F_L||_F² + ||F_L^T F_M||_F² + ||F_N^T F_M||_F²
        在下采样的空间分辨率上计算，控制计算量
        """
        B, C, H, W = F_N.shape
        # 下采样到 8×8 再计算正交性
        ds = min(8, H, W)
        fn = F.adaptive_avg_pool2d(F_N, (ds, ds)).flatten(1)  # (B, C*ds*ds)
        fl = F.adaptive_avg_pool2d(F_L, (ds, ds)).flatten(1)
        fm = F.adaptive_avg_pool2d(F_M, (ds, ds)).flatten(1)
        
        # 归一化
        fn = F.normalize(fn, dim=1)
        fl = F.normalize(fl, dim=1)
        fm = F.normalize(fm, dim=1)
        
        loss = (fn * fl).sum(dim=1).pow(2).mean() + \
               (fl * fm).sum(dim=1).pow(2).mean() + \
               (fn * fm).sum(dim=1).pow(2).mean()
        return loss
    
    def forward(self, center_feat: torch.Tensor, feats_seq: torch.Tensor,
                f3_ctx: torch.Tensor = None) -> Dict:
        """
        center_feat: (B, C_enc, H/2, W/2)
        feats_seq:   (B, T, C_enc, H/2, W/2)
        f3_ctx:      (B, 128) — F3 全局描述子 [R4-C FiLM]

        Returns dict: F_N, F_L, F_M, var_map, ortho_loss,
                      prior_loss (N/L/M 结构化先验正则之和)
        """
        B, T, C_enc, H, W = feats_seq.shape

        # 投影到 tca_channels
        center = self.in_proj(center_feat)
        feats = self.in_proj(feats_seq.reshape(B * T, C_enc, H, W)).reshape(B, T, self.tca_channels, H, W)

        # Step 1: MVC-Shift 预处理
        feats_flat = feats.reshape(B * T, self.tca_channels, H, W)
        feats_shifted = self.mvc_shift(feats_flat)
        feats_shifted = feats_shifted.reshape(B, T, self.tca_channels, H, W)
        center_shifted = feats_shifted[:, self.center_idx]

        # Step 2: 三路时序上下文 (供先验约束使用, 物理语义: 均值/低频/差分)
        ctx_N = self.temporal_agg(self._temporal_mean_context(feats_shifted))
        ctx_L = self.temporal_agg(self._temporal_smooth_context(feats_shifted))
        ctx_M = self.temporal_agg(self._temporal_diff_context(feats_shifted, self.center_idx))

        # Step 2b [R4-结构化 K/V, 对应 TSD-Foxtrot.md §3.2]:
        #   K = V = Concat_time(F_{t±i})  —— 三路共享, 保留全部逐帧信息
        #   (原 Golf-R3 用 ctx_N/L/M 聚合统计量作 KV, 丢失逐帧细节)
        #   Concat 后 1×1 投影回 C 通道, 避免 T*C 维度爆炸
        concat_kv = feats_shifted.reshape(B, T * self.tca_channels, H, W)
        kv_shared = self.kv_proj(concat_kv)   # (B, C, H, W)

        # Step 3: 三路查询生成
        Q_N = self.query_N(center_shifted)
        Q_L = self.query_L(center_shifted)
        Q_M = self.query_M(center_shifted)

        # Step 4 [R4-C FiLM]: F3 退化感知 per-channel 调制 (施加在注意力输出上)
        film_N = film_L = film_M = None
        if self.use_f3_film and f3_ctx is not None:
            fp = self.f3_film(f3_ctx)                        # (B, 6C)
            gN, bN, gL, bL, gM, bM = fp.chunk(6, dim=-1)
            u = (-1, -1)
            film_N = (gN.unsqueeze(-1).unsqueeze(-1), bN.unsqueeze(-1).unsqueeze(-1))
            film_L = (gL.unsqueeze(-1).unsqueeze(-1), bL.unsqueeze(-1).unsqueeze(-1))
            film_M = (gM.unsqueeze(-1).unsqueeze(-1), bM.unsqueeze(-1).unsqueeze(-1))

        # Step 5: RWKV 空间注意力 (共享 K/V, 差异化 Q + FiLM)
        attn_N = self.attn_N(Q_N, kv_shared, film=film_N)
        attn_L = self.attn_L(Q_L, kv_shared, film=film_L)
        attn_M = self.attn_M(Q_M, kv_shared, film=film_M)

        # Step 6: LayerScale 残差 + 结构化先验
        raw_N = center + attn_N * self.scale_N
        raw_L = center + attn_L * self.scale_L
        raw_M = center + attn_M * self.scale_M

        F_N_prior, pl_N = self.prior_N(raw_N, center, ctx_N, seq_mean=ctx_N)
        F_L_prior, pl_L = self.prior_L(raw_L, center, ctx_L)
        F_M_prior, pl_M = self.prior_M(raw_M, center, ctx_M)
        prior_loss = pl_N + pl_L + pl_M

        # Step 7: 输出归一化
        F_N = self.out_norm_N(F_N_prior)
        F_L = self.out_norm_L(F_L_prior)
        F_M = self.out_norm_M(F_M_prior)

        # Step 8: 方差图 + 正交约束
        var_map = self._compute_var_map(feats)
        ortho_loss = self._ortho_loss(F_N, F_L, F_M)

        return {
            "F_N": F_N,
            "F_L": F_L,
            "F_M": F_M,
            "var_map": var_map,
            "ortho_loss": ortho_loss,
            "prior_loss": prior_loss,
        }
