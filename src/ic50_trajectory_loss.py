# -*- coding: utf-8 -*-
"""
IC50-guided Trajectory Regularization (IC50-TR)
================================================
药物特异性 IC50 引导的轨迹正则模块。

核心思想：CCLE 每个药物有连续 IC50 值（比二元敏感/耐药标签更丰富）。
在 latent 空间里，按 IC50 排序的相邻样本应该特征相近——形成一条
从敏感到耐药的平滑轨迹。这条轨迹对跨域（CCLE→TCGA）有用：
分类器学会沿轨迹连续插值，而非只记住离散标签的决策边界。

用法（在 finetuning 训练循环里）：
    from ic50_trajectory_loss import IC50TrajectoryLoss
    criterion_tr = IC50TrajectoryLoss(temperature=0.07)
    loss_tr = criterion_tr(drug_gene_features, ic50_values, drug_indices)
    loss = loss_predic + 0.1*coral + 0.1*supcon + 0.1*loss_tr

只作用于 CCLE（源域）。IC50 值需要预先归一化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IC50TrajectoryLoss(nn.Module):
    """IC50 加权特征平滑损失

    对同药同癌种内所有 pair，用高斯核 w = exp(-(ΔIC50)²/σ²) 做权重：
    IC50 越近 → 特征越应相似（高权重拉近）
    IC50 越远 → 权重趋零（不强制）

    不需要硬排序、不需要正负样本拆分，3 个样本的组也能稳定工作。
    """
    def __init__(self, sigma=0.5):
        super().__init__()
        self.sigma = sigma

    def forward(self, features, ic50, drug_idx, tissue=None):
        device = features.device
        N, D = features.shape
        n_drugs = ic50.shape[1]
        total_loss = 0.0
        valid_groups = 0

        if tissue is None:
            tissue = torch.zeros(N, dtype=torch.long, device=device)

        for d in range(n_drugs):
            mask_d = (drug_idx == d)
            if mask_d.sum() < 2:
                continue

            feat_d = features[mask_d]
            ic50_d = ic50[mask_d, d]
            tissue_d = tissue[mask_d]

            valid = ~torch.isnan(ic50_d)
            if valid.sum() < 2:
                continue
            feat_d, ic50_d, tissue_d = feat_d[valid], ic50_d[valid], tissue_d[valid]

            for t in tissue_d.unique():
                mask_t = (tissue_d == t)
                if mask_t.sum() < 2:
                    continue
                f = feat_d[mask_t]
                ic = ic50_d[mask_t]

                # 特征距离矩阵 [K, K]
                f_norm = F.normalize(f, dim=1)
                feat_dist = 1.0 - torch.matmul(f_norm, f_norm.T)  # cosine dist

                # IC50 距离 → 高斯权重 [K, K]
                ic_dist = (ic[:, None] - ic[None, :]).pow(2)
                weight = torch.exp(-ic_dist / (2 * self.sigma ** 2))
                weight.fill_diagonal_(0)                           # 排除自己

                # 加权平滑：IC50 相近的 pair 特征距离应小
                if weight.sum() == 0:
                    continue
                total_loss += (weight * feat_dist).sum() / weight.sum()
                valid_groups += 1

        if valid_groups == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / valid_groups


def load_and_normalize_ic50(csv_path='data/preprocessed/ccle_labeled_ic50.csv'):
    """加载 IC50 数据并做 log 归一化，返回 {DepMap_ID: [9]} 字典"""
    import pandas as pd
    import numpy as np
    df = pd.read_csv(csv_path, index_col=0)
    drugs = [c for c in df.columns if c != 'Tissue']

    ic50_dict = {}
    for idx, row in df.iterrows():
        vals = []
        for d in drugs:
            v = row[d]
            vals.append(np.log1p(v) if not pd.isna(v) else np.nan)
        ic50_dict[idx] = np.array(vals, dtype=np.float32)

    # 全局归一化：每个药独立做 z-score
    all_vals = {d: [] for d in range(len(drugs))}
    for arr in ic50_dict.values():
        for d in range(len(drugs)):
            if not np.isnan(arr[d]):
                all_vals[d].append(arr[d])
    stats = {}
    for d in range(len(drugs)):
        a = np.array(all_vals[d])
        stats[d] = (a.mean(), a.std())

    # 应用归一化
    for k, arr in ic50_dict.items():
        for d in range(len(drugs)):
            if not np.isnan(arr[d]):
                mu, std = stats[d]
                arr[d] = (arr[d] - mu) / (std + 1e-8)

    return ic50_dict, stats



class PairwiseRankingLoss(nn.Module):
    """IC50 成对排序损失 — 只惩罚序反了的 pair

    同药同癌种内，IC50_A < IC50_B 但模型判了 pred_A < pred_B：
        loss = relu(pred_B - pred_A) = relu(-pred_diff)

    不拉大差距，不设 margin，不设 min_gap。
    序对了就 loss=0，序反了按反的程度罚。
    """
    def __init__(self):
        super().__init__()

    def forward(self, predictions, ic50, drug_idx, tissue=None):
        device = predictions.device
        N = predictions.shape[0]
        n_drugs = ic50.shape[1]
        total_loss = 0.0
        valid_pairs = 0

        if tissue is None:
            tissue = torch.zeros(N, dtype=torch.long, device=device)

        for d in range(n_drugs):
            mask_d = (drug_idx == d)
            pred_d = predictions[mask_d]
            ic50_d = ic50[mask_d, d]
            t_d = tissue[mask_d]

            valid = ~torch.isnan(ic50_d)
            if valid.sum() < 2:
                continue
            pred_d, ic50_d, t_d = pred_d[valid], ic50_d[valid], t_d[valid]

            for t_val in t_d.unique():
                mask_t = (t_d == t_val)
                if mask_t.sum() < 2:
                    continue
                p = pred_d[mask_t]
                ic = ic50_d[mask_t]

                ic_diff = ic[:, None] - ic[None, :]           # [K,K]
                pred_diff = p[:, None] - p[None, :]           # [K,K]

                # ic_i < ic_j → pred_i 应该 > pred_j
                # 序反了才罚：pred_i < pred_j 即 pred_diff < 0
                wrong_order = (ic_diff < 0) & (pred_diff < 0)
                if wrong_order.sum() == 0:
                    continue

                losses = F.relu(-pred_diff[wrong_order])/10      # 序反了多少
                total_loss += losses.mean()
                valid_pairs += 1

        if valid_pairs == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / valid_pairs


class SoftKendallTauLoss(nn.Module):
    """Soft Kendall's τ 排序一致性损失

    输入 logits（sigmoid 之前的值），用 tanh 做软 τ。
    pred_diff 在 logit 空间可达 ±5，tanh 能有效区分序对/序反。

    τ ≈ +1: 预测序与 IC50 序一致
    τ ≈ -1: 预测序与 IC50 序相反
    loss = (1 - τ) / 2  ∈ [0, 1]
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, ic50, drug_idx, tissue=None):
        """
        Args:
            logits:  [N] — 模型输出的 logits（sigmoid 之前）
            ic50:    [N, 9]
            drug_idx: [N]
            tissue:  [N]
        """
        device = logits.device
        N = logits.shape[0]
        n_drugs = ic50.shape[1]
        total_tau = 0.0
        valid_groups = 0

        if tissue is None:
            tissue = torch.zeros(N, dtype=torch.long, device=device)

        for d in range(n_drugs):
            mask_d = (drug_idx == d)
            pred_d = logits[mask_d]
            ic50_d = ic50[mask_d, d]
            t_d = tissue[mask_d]

            valid = ~torch.isnan(ic50_d)
            if valid.sum() < 2:
                continue
            pred_d, ic50_d, t_d = pred_d[valid], ic50_d[valid], t_d[valid]

            for t_val in t_d.unique():
                mask_t = (t_d == t_val)
                if mask_t.sum() < 2:
                    continue
                p = pred_d[mask_t]
                ic = ic50_d[mask_t]

                ic_diff = ic[:, None] - ic[None, :]
                pred_diff = p[:, None] - p[None, :]

                valid_pair = (ic_diff < 0)
                if valid_pair.sum() == 0:
                    continue

                # tanh(logit_diff): 序对→+1，序反→-1
                tau_pairs = F.softplus(-pred_diff[valid_pair])
                total_tau += tau_pairs.mean()
                valid_groups += 1

        if valid_groups == 0:
            return torch.tensor(0.0, device=device)
        return  total_tau / valid_groups / 10
    

# class PairwiseRankingLoss(nn.Module):
#     """IC50 成对排序损失 — 作用在预测值上，与轨迹 loss 互补

#     轨迹 loss 约束 latent 空间连续性（特征层），
#     排序 loss 约束预测单调性（输出层）。

#     对于同药同癌种的样本对 (A, B)，IC50_A < IC50_B：
#         margin ∝ |IC50_B - IC50_A|  — 差距越大，序约束越强
#         loss = relu(margin - (pred_A - pred_B))
#     """
#     def __init__(self, base_margin=0.5, min_gap=0 ):
#         super().__init__()
#         self.base_margin = base_margin
#         self.min_gap = min_gap

#     def forward(self, predictions, ic50, drug_idx, tissue=None):
#         device = predictions.device
#         N = predictions.shape[0]
#         n_drugs = ic50.shape[1]
#         total_loss = 0.0
#         valid_pairs = 0

#         if tissue is None:
#             tissue = torch.zeros(N, dtype=torch.long, device=device)

#         for d in range(n_drugs):
#             mask_d = (drug_idx == d)
#             pred_d = predictions[mask_d]
#             ic50_d = ic50[mask_d, d]
#             t_d = tissue[mask_d]

#             valid = ~torch.isnan(ic50_d)
#             if valid.sum() < 2:
#                 continue
#             pred_d, ic50_d, t_d = pred_d[valid], ic50_d[valid], t_d[valid]

#             for t_val in t_d.unique():
#                 mask_t = (t_d == t_val)
#                 if mask_t.sum() < 2:
#                     continue
#                 p = pred_d[mask_t]
#                 ic = ic50_d[mask_t]

#                 ic_diff = ic[:, None] - ic[None, :]           # [K,K]
#                 pred_diff = p[:, None] - p[None, :]           # [K,K]

#                 # ic_i < ic_j - gap → A 比 B 更敏感
#                 valid_pair = (ic_diff < -self.min_gap)
#                 if valid_pair.sum() == 0:
#                     continue

#                 # margin 正比于 IC50 差距：差 100 倍比差 1.1 倍要求更严
#                 gap = ic_diff[valid_pair].abs()
#                 margin = self.base_margin * (gap / gap.max().clamp(min=1e-8))

#                 losses = F.relu(margin - pred_diff[valid_pair])
#                 total_loss += losses.mean()
#                 valid_pairs += 1

#         if valid_pairs == 0:
#             return torch.tensor(0.0, device=device)
#         return total_loss / valid_pairs


class DrugFingerprintLoss(nn.Module):
    """药物指纹 SupCon 对齐 MOA 嵌入

    从 labels [B, 9] 构建药物指纹相似度 → 正样本 mask，
    在 MOA 嵌入上做对比学习。与 TCGA 指纹 loss 完全对称。
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, drug_labels, drug_moa_embeddings):
        """
        Args:
            drug_labels:          [B, 9] — CCLE 标签矩阵
            drug_moa_embeddings:  [9, D] — 9 药的 MOA 嵌入
        Returns:
            scalar
        """
        # 药物指纹 = 每药在该 batch 中的 0/1 敏感谱
        drug_fp = F.normalize(drug_labels.T.float(), dim=1)  # [9, B]
        fp_sim = torch.matmul(drug_fp, drug_fp.T)             # [9, 9]

        # 指纹相似 > 0.5 → 正样本对
        pos_mask = (fp_sim > 0.5).float()
        pos_mask.fill_diagonal_(0)

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=drug_moa_embeddings.device)

        # SupCon 对比
        moa_norm = F.normalize(drug_moa_embeddings, dim=1)
        sim = torch.matmul(moa_norm, moa_norm.T) / self.temperature
        exp_sim = torch.exp(sim)
        denom = exp_sim.sum(dim=1, keepdim=True) - torch.diag(exp_sim).unsqueeze(1)
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        valid = pos_mask.sum(dim=1) > 0

        log_prob = torch.log(pos_sum[valid] / denom[valid].squeeze())
        return -log_prob.mean()
