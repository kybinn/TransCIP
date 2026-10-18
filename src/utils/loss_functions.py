import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import (
    jaccard_score,
    roc_auc_score,
    precision_score,
    f1_score,
    accuracy_score,
    average_precision_score,
    recall_score,
)
import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")


class SupConLossPro(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLossPro, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None):
        """
        features: 提取出的特征 [Batch, Dim]
        labels: 类别标签 [Batch] (药物级对比用这个)
        mask: 自定义的正样本对比矩阵 [Batch, Batch] (患者本底对比用这个)
        """
        features = F.normalize(features, dim=1) 
        sim_matrix = torch.matmul(features, features.T) / self.temperature
         
        if mask is None:
            if labels is None:
                raise ValueError("Must provide either labels or mask")
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float()
        else: 
            mask = mask.float()
        
 
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(features.shape[0]).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask
         
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
        pos_count = mask.sum(1)
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(1)[valid] / (pos_count[valid] + 1e-9)
        loss = -mean_log_prob_pos.mean()
        
        return loss
    

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
         
        features = F.normalize(features, dim=1) 
        
 
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
 
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
 
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(features.shape[0]).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask
        
 
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
 
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos.mean()
        
        return loss


def bina_metric(y_gt, y_pred, y_prob):

    def f1(y_gt, y_pred):
        """F1分数"""
        return f1_score(y_gt, y_pred, average='binary')

    def roc_auc(y_gt, y_prob):
        """ROC AUC分数"""
        return roc_auc_score(y_gt, y_prob)

    def precision_auc(y_gt, y_prob):
        """平均精确率 (Average Precision)"""
        return average_precision_score(y_gt, y_prob)

    def precision(y_gt, y_pred):
        """精确率"""
        return precision_score(y_gt, y_pred, zero_division=0)

    def recall(y_gt, y_pred):
        """召回率"""
        return recall_score(y_gt, y_pred, zero_division=0)
    

    
    precision = precision(y_gt, y_pred)
    recall = recall(y_gt, y_pred)
    f1 = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    roc_auc = roc_auc(y_gt, y_prob)
    accuracy = accuracy_score(y_gt, y_pred)

    return  precision, recall, f1, prauc, roc_auc,accuracy


def Adversarial_loss(domain_s, domain_t):
    # Initialize the pseudo-labels of the domain confrontation
    domain_labels = torch.cat((domain_s, domain_t), dim=0) 
    bs_s = domain_s.size(0)
    bs_t = domain_t.size(0)
    device = domain_s.device
    target_labels = torch.from_numpy(np.array([[0]]*bs_s + [[1]]*bs_t).astype('float32')).to(device)  
    domain_loss = nn.BCEWithLogitsLoss()(domain_labels, target_labels) 
    
    return domain_loss


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def semi_loss(z1: torch.Tensor, z2: torch.Tensor):
    f = lambda x: torch.exp(x / 1)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

def instanceloss(z1: torch.Tensor, z2: torch.Tensor, mean: bool = True):
    l1 = semi_loss(z1, z2)
    l2 = semi_loss(z2, z1)
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()
    return ret


def coral(source, target):

    d = source.size(1)  # dim vector

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    # loss = loss / (4 * d * d)
    return loss

def ortho_loss(source, target):
        n_source = source.size()[0]
        n_target = target.size()[0]
        if n_source < n_target:
            source = source.repeat(int(n_target / n_source) + 1, 1)[:n_target, :]
        elif n_target < n_source:
            target = target.repeat(int(n_source / n_target) + 1, 1)[:n_source, :]
        
        source_norm = torch.norm(source, p=2, dim=1, keepdim=True).detach()
        source_emb = source.div(source_norm.expand_as(source) + 1e-8)

        target_norm = torch.norm(target, p=2, dim=1, keepdim=True).detach()
        target_emb = target.div(target_norm.expand_as(target) + 1e-8)

        loss = torch.mean((source_emb.t().mm(target_emb)).pow(2))
        return loss

def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = input_data.device
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c   

