# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import torch.nn as nn
import config
from models import DRPmodel, DRPmodel1, DRPmodel2, DRPmodel3
from torch_geometric.data import Batch
from utils.loss_functions import bina_metric
import pandas as pd
from itertools import cycle
from utils.loss_functions import coral,ortho_loss,SupConLoss,SupConLossPro
from torch.nn import functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

IC50_FILE = os.path.join(PROJECT_ROOT, 'data', 'preprocessed', 'ccle_labeled_ic50.csv')
DRUG_PHI_FILE = os.path.join(PROJECT_ROOT, 'data', 'drug_phi_global.npy')

from ic50_trajectory_loss import (IC50TrajectoryLoss, PairwiseRankingLoss,
    load_and_normalize_ic50)



class SoftF1Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(SoftF1Loss, self).__init__()
        self.epsilon = epsilon  
        

    def forward(self, logits, targets):
        
        probs = torch.sigmoid(logits)
        targets = targets.float()

         
        intersection = torch.sum(probs * targets)
        denominator = torch.sum(probs) + torch.sum(targets)

         
        soft_f1 = (2.0 * intersection + self.epsilon) / (denominator + self.epsilon)
        
         
        return 1.0 - soft_f1


def build_batch_ic50(sample_ids, ic50_dict, device):
    batch_ic50 = torch.full((len(sample_ids), 9), float('nan'), device=device)
    for i, sid in enumerate(sample_ids):
        if sid in ic50_dict:
            batch_ic50[i] = torch.tensor(ic50_dict[sid], device=device)
    return batch_ic50


 

def alignment_context(model, gene_all, ccc_features=None):
    cancer_embed = model.cancer_encoder(gene_all)
    if (
        ccc_features is None
        or getattr(model, "ccc_encoder", None) is None
        or getattr(model, "ccc_gate_layer", None) is None
        or not getattr(model, "use_base_ccc_gate_for_alignment", True)
    ):
        return cancer_embed

    ccc_embed = model.ccc_encoder(ccc_features)
    ccc_gate = torch.sigmoid(model.ccc_gate_layer(ccc_embed))
    return cancer_embed * ccc_gate


def tcga_fingerprint_loss_ablation(model, tcga_gene_all, tcga_ccc_features,
                                   drug_embedding_smiles, drug_embedding_MOA,
                                   prototypes, t_cancer, t_ccc_embed, t_contextual_gene, supcon_criterion,
                                   device):
    all_tcga_features = []
    drug_smiles_tensor = torch.stack([torch.tensor(s) for s in drug_embedding_smiles]).to(device)
    drug_moa_tensor = torch.stack([torch.tensor(m) for m in drug_embedding_MOA]).to(device)

    for d_idx in range(drug_embedding_smiles.shape[0]):
        tcga_drug_inputs_smiles = drug_smiles_tensor[d_idx].repeat(tcga_gene_all.shape[0], 1)
        tcga_drug_inputs_MOA = drug_moa_tensor[d_idx].repeat(tcga_gene_all.shape[0], 1)
        if tcga_ccc_features is None:
            _, tcga_drug_gene_features  = model(
                tcga_gene_all, tcga_drug_inputs_smiles, tcga_drug_inputs_MOA
            )
        else:
            _, tcga_drug_gene_features  = model(
                tcga_gene_all, tcga_ccc_features, tcga_drug_inputs_smiles, tcga_drug_inputs_MOA
            )
        all_tcga_features.append(tcga_drug_gene_features)
    # print(drug_embedding_smiles.shape)
    T_Batch = tcga_gene_all.shape[0]
    tcga_features_by_drug = torch.stack(all_tcga_features, dim=0)
    valid_drug_mask = prototypes.norm(dim=1) > 1e-8
    if valid_drug_mask.sum() < 2:
        return torch.tensor(0.0, requires_grad=True, device=device)

    with torch.no_grad():
        tcga_fingerprints = []
        for d_idx in range(drug_embedding_smiles.shape[0]):
            if not valid_drug_mask[d_idx]:
                continue
            sim_d = F.cosine_similarity(
                tcga_features_by_drug[d_idx],
                prototypes[d_idx].unsqueeze(0),
                dim=1
            )
            tcga_fingerprints.append(sim_d)

        tcga_fingerprints = torch.stack(tcga_fingerprints, dim=1)
        norm_fingerprints = F.normalize(tcga_fingerprints, p=2, dim=1)
        tcga_patient_sim_matrix = torch.matmul(norm_fingerprints, norm_fingerprints.T)
        sim_matrix_no_diag = tcga_patient_sim_matrix.clone()
        sim_matrix_no_diag.fill_diagonal_(-1.0)
        k_neighbors = min(7, T_Batch - 1)
        if k_neighbors <= 0:
            return torch.tensor(0.0, requires_grad=True, device=device)

        topk_values, topk_indices = torch.topk(sim_matrix_no_diag, k=k_neighbors, dim=1)
        topk_weights = torch.softmax(topk_values / 0.2, dim=1)
        tcga_fingerprint_mask = torch.zeros_like(tcga_patient_sim_matrix)
        tcga_fingerprint_mask.scatter_(1, topk_indices, topk_weights)
        tcga_fingerprint_mask = 0.5 * (tcga_fingerprint_mask + tcga_fingerprint_mask.T)
        tcga_fingerprint_mask.fill_diagonal_(0.0)

  
    if t_ccc_embed is None:
        return 0.5*supcon_criterion(t_contextual_gene, mask=tcga_fingerprint_mask)+0.5*(supcon_criterion(t_cancer, mask=tcga_fingerprint_mask))
    else:
        return 0.5*supcon_criterion(t_contextual_gene, mask=tcga_fingerprint_mask)+0.5*(supcon_criterion(t_cancer, mask=tcga_fingerprint_mask))
        
        


def multi_eval_epoch(substruct_graphs,drug_embedding_smiles,drug_embedding_MOA,model, loader, device):
    model.eval()
    total_loss = 0
    alpha = 0
    y_true, y_pred = [], []
    auc_list, aupr_list, acc_list, f1_list = [], [], [], []

    result_per_epoch = []
    labels_per_epoch = []

    for gene_all, ccc_features, labels, mask, *_ in loader:
        gene_all = gene_all.to(device)
        ccc_features = ccc_features.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        
         
        cancergene_inputs = []
        ccc_inputs = []
        labels_real = []
        drug_inputs_smiles,drug_inputs_MOA,substruct_graphs_input = [], [],[]
        drug_indices=[]  
        for n in range(mask.shape[1]):
            for m in range(mask.shape[0]):
                if mask[m][n] > 0:
                    cancergene_inputs.append(gene_all[m])
                    ccc_inputs.append(ccc_features[m])
                    drug_inputs_smiles.append(drug_embedding_smiles[n])
                    drug_inputs_MOA.append(drug_embedding_MOA[n])
                    
                    labels_real.append(labels[m][n])
                    drug_indices.append(n) 
            
        
        cancergene_inputs = torch.stack(cancergene_inputs, dim=0)
        ccc_inputs = torch.stack(ccc_inputs, dim=0)
        labels_real = torch.stack(labels_real, dim=0)
        drug_indices = torch.tensor(drug_indices, dtype=torch.long) 
        

        with torch.no_grad():
            predictions, drug_gene_features = model(cancergene_inputs,ccc_inputs,drug_inputs_smiles,drug_inputs_MOA,substruct_graphs,config.substru_mask,drug_indices)
            probabilities = torch.sigmoid(predictions)
            loss  = nn.BCEWithLogitsLoss()(predictions,labels_real) 
            
            total_loss += loss

            y_true.extend(labels_real.cpu().detach().numpy().tolist())
            y_pred.extend(probabilities.cpu().detach().numpy().tolist())

        
        
        result_in_batch = torch.zeros_like(labels)
        i = 0
                 
        for n in range(mask.shape[1]):
            for m in range(mask.shape[0]):
                if mask[m][n] > 0:
                    result_in_batch[m][n] = torch.sigmoid(predictions[i])
                    i+=1
        
        
        result_per_epoch.append(result_in_batch.cpu().detach().numpy())
        labels_per_epoch.append(labels.cpu().detach().numpy())  
        
    result_per_epoch = np.vstack(result_per_epoch) 
    labels_per_epoch = np.vstack(labels_per_epoch)
         
    y_true = np.array(y_true)   
    y_pred = np.array(y_pred)

         
    
    df_metrics = pd.DataFrame(columns=['Drug','Precision','F1','PRAUC','Accuracy','ROC_AUC'])
    for drug_idx in range(result_per_epoch.shape[1]):
        drug_results = result_per_epoch[:, drug_idx]
        drug_labels = labels_per_epoch[:, drug_idx]
        
        valid_indices = np.where(drug_labels != -1)[0]
        if(len(valid_indices) == 0):
            precision, recall, f1, prauc, roc_auc,accuracy = -1, -1, -1, -1, -1,-1
        else:
            drug_results = drug_results[valid_indices]
            drug_labels = drug_labels[valid_indices]
            # print(f"drug{drug_idx} total samples:", len(drug_labels))
            # print(f"drug{drug_idx} positive samples:", sum(drug_labels))
            precision, recall, f1, prauc, roc_auc,accuracy = bina_metric(drug_labels, (drug_results>=0.5).astype(int), drug_results)
        
        new_row = {'Drug': drug_idx,
                    'Precision': precision,
                    'F1': f1,
                    'PRAUC': prauc,
                    'Accuracy': accuracy,
                    'ROC_AUC': roc_auc}
        df_metrics = pd.concat([df_metrics, pd.DataFrame([new_row])], ignore_index=True)

    
    return total_loss, df_metrics , y_true, y_pred


def testing(substruct_graphs,drug_embedding_smiles,drug_embedding_MOA,model, t_dataloader, device):
    model.train(False)  
    test_loss, results, y_true, y_pred = multi_eval_epoch(substruct_graphs,drug_embedding_smiles=drug_embedding_smiles,drug_embedding_MOA= drug_embedding_MOA,model=model,
                                          loader=t_dataloader,
                                          device=device) 
    precision, recall, f1, prauc, roc_auc,accuracy = bina_metric(y_true, (y_pred>=0.5).astype(int), y_pred)
    
    print(
        f"** TESTING on TCGA data**   "
        f"loss: {test_loss:.5f}, "
        f"Precision: {precision:.5f}, "
        f"Recall: {recall:.5f}, "
        f"F1: {f1:.5f}, "
        f"PRAUC: {prauc:.5f}, "
        f"ROC_AUC: {roc_auc:.5f}. "
    )
    return test_loss, results, y_true, y_pred


def multi_eval_epoch1(drug_embedding_smiles, drug_embedding_MOA, model, loader, device):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    result_per_epoch = []
    labels_per_epoch = []

    for gene_all, ccc_features, labels, mask, *_ in loader:
        gene_all = gene_all.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        cancergene_inputs = []
        labels_real = []
        drug_inputs_smiles = []
        drug_inputs_MOA = []
        for n in range(mask.shape[1]):
            for m in range(mask.shape[0]):
                if mask[m][n] > 0:
                    cancergene_inputs.append(gene_all[m])
                    drug_inputs_smiles.append(drug_embedding_smiles[n])
                    drug_inputs_MOA.append(drug_embedding_MOA[n])
                    labels_real.append(labels[m][n])

        cancergene_inputs = torch.stack(cancergene_inputs, dim=0)
        labels_real = torch.stack(labels_real, dim=0)

        with torch.no_grad():
            predictions, drug_gene_features  = model(
                cancergene_inputs, drug_inputs_smiles, drug_inputs_MOA
            )
            probabilities = torch.sigmoid(predictions)
            loss = nn.BCEWithLogitsLoss()(predictions, labels_real)
            total_loss += loss

            y_true.extend(labels_real.cpu().detach().numpy().tolist())
            y_pred.extend(probabilities.cpu().detach().numpy().tolist())

        result_in_batch = torch.zeros_like(labels)
        i = 0
        for n in range(mask.shape[1]):
            for m in range(mask.shape[0]):
                if mask[m][n] > 0:
                    result_in_batch[m][n] = torch.sigmoid(predictions[i])
                    i += 1
        result_per_epoch.append(result_in_batch.cpu().detach().numpy())
        labels_per_epoch.append(labels.cpu().detach().numpy())

    result_per_epoch = np.vstack(result_per_epoch)
    labels_per_epoch = np.vstack(labels_per_epoch)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    df_metrics = pd.DataFrame(columns=['Drug', 'Precision', 'F1', 'PRAUC', 'Accuracy', 'ROC_AUC'])
    for drug_idx in range(result_per_epoch.shape[1]):
        drug_results = result_per_epoch[:, drug_idx]
        drug_labels = labels_per_epoch[:, drug_idx]
        valid_indices = np.where(drug_labels != -1)[0]
        if len(valid_indices) == 0:
            precision, recall, f1, prauc, roc_auc, accuracy = -1, -1, -1, -1, -1, -1
        else:
            drug_results = drug_results[valid_indices]
            drug_labels = drug_labels[valid_indices]
            precision, recall, f1, prauc, roc_auc, accuracy = bina_metric(
                drug_labels, (drug_results >= 0.5).astype(int), drug_results
            )

        new_row = {'Drug': drug_idx,
                   'Precision': precision,
                   'F1': f1,
                   'PRAUC': prauc,
                   'Accuracy': accuracy,
                   'ROC_AUC': roc_auc}
        df_metrics = pd.concat([df_metrics, pd.DataFrame([new_row])], ignore_index=True)

    return total_loss, df_metrics, y_true, y_pred


def testing1(drug_embedding_smiles, drug_embedding_MOA, model, t_dataloader, device):
    model.train(False)
    test_loss, results, y_true, y_pred = multi_eval_epoch1(
        drug_embedding_smiles=drug_embedding_smiles,
        drug_embedding_MOA=drug_embedding_MOA,
        model=model,
        loader=t_dataloader,
        device=device
    )
    precision, recall, f1, prauc, roc_auc, accuracy = bina_metric(y_true, (y_pred >= 0.5).astype(int), y_pred)

    print(
        f"** TESTING on TCGA data**   "
        f"loss: {test_loss:.5f}, "
        f"Precision: {precision:.5f}, "
        f"Recall: {recall:.5f}, "
        f"F1: {f1:.5f}, "
        f"PRAUC: {prauc:.5f}, "
        f"ROC_AUC: {roc_auc:.5f}. "
    )
    return test_loss, results, y_true, y_pred


def multi_eval_epoch2(drug_embedding_smiles, drug_embedding_MOA, model, loader, device):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    result_per_epoch = []
    labels_per_epoch = []

    for gene_all, ccc_features, labels, mask, *_ in loader:
        gene_all = gene_all.to(device)
        ccc_features = ccc_features.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        cancergene_inputs = []
        ccc_inputs = []
        labels_real = []
        drug_inputs_smiles = []
        drug_inputs_MOA = []
        for n in range(mask.shape[1]):
            for m in range(mask.shape[0]):
                if mask[m][n] > 0:
                    cancergene_inputs.append(gene_all[m])
                    ccc_inputs.append(ccc_features[m])
                    drug_inputs_smiles.append(drug_embedding_smiles[n])
                    drug_inputs_MOA.append(drug_embedding_MOA[n])
                    labels_real.append(labels[m][n])

        cancergene_inputs = torch.stack(cancergene_inputs, dim=0)
        ccc_inputs = torch.stack(ccc_inputs, dim=0)
        labels_real = torch.stack(labels_real, dim=0)

        with torch.no_grad():
            predictions, drug_gene_features = model(
                cancergene_inputs, ccc_inputs, drug_inputs_smiles, drug_inputs_MOA
            )
            probabilities = torch.sigmoid(predictions)
            loss = nn.BCEWithLogitsLoss()(predictions, labels_real)
            total_loss += loss

            y_true.extend(labels_real.cpu().detach().numpy().tolist())
            y_pred.extend(probabilities.cpu().detach().numpy().tolist())

        result_in_batch = torch.zeros_like(labels)
        i = 0
        for n in range(mask.shape[1]):
            for m in range(mask.shape[0]):
                if mask[m][n] > 0:
                    result_in_batch[m][n] = torch.sigmoid(predictions[i])
                    i += 1
        result_per_epoch.append(result_in_batch.cpu().detach().numpy())
        labels_per_epoch.append(labels.cpu().detach().numpy())

    result_per_epoch = np.vstack(result_per_epoch)
    labels_per_epoch = np.vstack(labels_per_epoch)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    df_metrics = pd.DataFrame(columns=['Drug', 'Precision', 'F1', 'PRAUC', 'Accuracy', 'ROC_AUC'])
    for drug_idx in range(result_per_epoch.shape[1]):
        drug_results = result_per_epoch[:, drug_idx]
        drug_labels = labels_per_epoch[:, drug_idx]
        valid_indices = np.where(drug_labels != -1)[0]
        if len(valid_indices) == 0:
            precision, recall, f1, prauc, roc_auc, accuracy = -1, -1, -1, -1, -1, -1
        else:
            drug_results = drug_results[valid_indices]
            drug_labels = drug_labels[valid_indices]
            precision, recall, f1, prauc, roc_auc, accuracy = bina_metric(
                drug_labels, (drug_results >= 0.5).astype(int), drug_results
            )

        new_row = {'Drug': drug_idx,
                   'Precision': precision,
                   'F1': f1,
                   'PRAUC': prauc,
                   'Accuracy': accuracy,
                   'ROC_AUC': roc_auc}
        df_metrics = pd.concat([df_metrics, pd.DataFrame([new_row])], ignore_index=True)

    return total_loss, df_metrics, y_true, y_pred


def testing2(drug_embedding_smiles, drug_embedding_MOA, model, t_dataloader, device):
    model.train(False)
    test_loss, results, y_true, y_pred = multi_eval_epoch2(
        drug_embedding_smiles=drug_embedding_smiles,
        drug_embedding_MOA=drug_embedding_MOA,
        model=model,
        loader=t_dataloader,
        device=device
    )
    precision, recall, f1, prauc, roc_auc, accuracy = bina_metric(y_true, (y_pred >= 0.5).astype(int), y_pred)

    print(
        f"** TESTING on TCGA data**   "
        f"loss: {test_loss:.5f}, "
        f"Precision: {precision:.5f}, "
        f"Recall: {recall:.5f}, "
        f"F1: {f1:.5f}, "
        f"PRAUC: {prauc:.5f}, "
        f"ROC_AUC: {roc_auc:.5f}. "
    )
    return test_loss, results, y_true, y_pred



def training2(drug_embedding_smiles,drug_embedding_MOA,cancer_encoder,ccc_encoder,ccc_gate_layer,s_dataloader,train_ids,test_ids,val_dataloader,tcga_dataloader,
              model_cls=DRPmodel2, disable_supcon=False, disable_tcga_fingerprint=False, disable_rank=False, log_dir="logs/stage2_ablation2", **kwargs):
    model = model_cls(cancer_encoder, ccc_encoder, ccc_gate_layer, kwargs['latent_dim']).to(kwargs['device'])

    if getattr(model, 'ccc_encoder', None) is not None:
        for p in model.ccc_encoder.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=kwargs['lr']
    )

    print("\n============Training for CCLE data================")
    supcon_criterion = SupConLoss(temperature=0.07).to(kwargs['device'])
    supcon_criterion2 = SupConLossPro(temperature=0.07).to(kwargs['device'])
    ic50_dict, _ = load_and_normalize_ic50(IC50_FILE)
    criterion_rank = PairwiseRankingLoss()
    f1_criterion = SoftF1Loss()
    best_epoch = 0
    val_rocauc = 0

    for epoch in range(int(kwargs['train_num_epochs'])):
        result_per_epoch = []
        labels_per_epoch = []
        y_true, y_pred = [], []
        training_loss_sum = 0
        coral_loss_sum = 0
        loss_predic_sum = 0

        model.train(True)
        optimizer.zero_grad()

        for i, batch in enumerate(zip(tcga_dataloader, cycle(s_dataloader))):
            s_batch = batch[1]
            tcga_batch = batch[0]
            s_gene_all = s_batch[0].to(kwargs['device'])
            ccc_features = s_batch[1].to(kwargs['device'])
            labels = s_batch[2].to(kwargs['device'])
            mask = s_batch[3].to(kwargs['device'])
            s_tissue = s_batch[4].to(kwargs['device'])
            s_ids_idx = s_batch[5]

            tcga_gene_all = tcga_batch[0].to(kwargs['device'])
            tcga_ccc_features = tcga_batch[1].to(kwargs['device'])

            s_contextual_gene = alignment_context(model, s_gene_all, ccc_features)
            t_contextual_gene = alignment_context(model, tcga_gene_all, tcga_ccc_features)
            t_cancer = cancer_encoder(tcga_gene_all)
            if (
                ccc_features is None
                or getattr(model, "ccc_encoder", None) is None
                or getattr(model, "ccc_gate_layer", None) is None
                or not getattr(model, "use_base_ccc_gate_for_alignment", True)
            ):
                t_ccc_embed = None
            else:
                t_ccc_embed = ccc_encoder(tcga_ccc_features)

            coral_loss = kwargs['lam1'] * coral(s_contextual_gene, t_contextual_gene)
            coral_loss_sum += coral_loss.cpu().detach().item()

            drug_inputs_smiles = []
            drug_inputs_MOA = []
            cancergene_inputs = []
            ccc_inputs = []
            labels_real = []
            labels_contrastive = []
            drug_indices = []
            sample_ids = []
            tissue_ids = []

            for n in range(mask.shape[1]):
                for m in range(mask.shape[0]):
                    if mask[m][n] > 0:
                        cancergene_inputs.append(s_gene_all[m])
                        ccc_inputs.append(ccc_features[m])
                        drug_inputs_smiles.append(drug_embedding_smiles[n])
                        drug_inputs_MOA.append(drug_embedding_MOA[n])
                        labels_real.append(labels[m][n])
                        labels_contrastive.append(n * 2 + labels[m][n])
                        drug_indices.append(n)
                        sample_ids.append(train_ids[s_ids_idx[m].item()])
                        tissue_ids.append(s_tissue[m].item())

            cancergene_inputs = torch.stack(cancergene_inputs, dim=0).to(kwargs['device'])
            ccc_inputs = torch.stack(ccc_inputs, dim=0).to(kwargs['device'])
            labels_real = torch.stack(labels_real, dim=0).to(kwargs['device'])
            labels_contrastive = torch.stack(labels_contrastive, dim=0).to(kwargs['device'])
            drug_indices_tensor = torch.tensor(drug_indices, dtype=torch.long).to(kwargs['device'])

            predictions, drug_gene_features = model(
                cancergene_inputs, ccc_inputs, drug_inputs_smiles, drug_inputs_MOA
            )

            loss_predic = 0.0
            valid_drug_count = 0
            for d_idx in range(mask.shape[1]):
                drug_mask = (drug_indices_tensor == d_idx)
                current_drug_samples = drug_mask.sum().item()

                if current_drug_samples > 0:
                    pred_d = predictions[drug_mask]
                    label_d = labels_real[drug_mask]
                    pos_num = label_d.sum()
                    neg_num = current_drug_samples - pos_num

                    if pos_num > 0 and neg_num > 0:
                        pos_weight = neg_num / pos_num
                        bce_d = F.binary_cross_entropy_with_logits(pred_d, label_d, pos_weight=pos_weight)
                    else:
                        bce_d = F.binary_cross_entropy_with_logits(pred_d, label_d, reduction='mean')

                    loss_predic += bce_d
                    valid_drug_count += 1

            if valid_drug_count > 0:
                loss_predic = loss_predic / valid_drug_count
            else:
                loss_predic = torch.tensor(0.0, requires_grad=True).to(kwargs['device'])

            loss_predic_sum += loss_predic.cpu().detach().item()
            if disable_supcon:
                loss_supcon = torch.tensor(0.0, requires_grad=True, device=kwargs['device'])
            else:
                loss_supcon = supcon_criterion(drug_gene_features, labels_contrastive)
            batch_ic50 = build_batch_ic50(sample_ids, ic50_dict, kwargs['device'])
            tissue_tensor = torch.tensor(tissue_ids, dtype=torch.long, device=kwargs['device'])
            if disable_rank:
                loss_rank = torch.tensor(0.0, requires_grad=True, device=kwargs['device'])
            else:
                loss_rank = criterion_rank(predictions, batch_ic50, drug_indices_tensor, tissue=tissue_tensor)
            loss_f1 = f1_criterion(predictions, labels_real)

            prototypes = []
            for d_idx in range(mask.shape[1]):
                mask_d = (drug_indices_tensor == d_idx) & (labels_real == 1)
                if mask_d.sum() > 0:
                    proto = drug_gene_features[mask_d].mean(dim=0)
                else:
                    proto = torch.zeros(drug_gene_features.shape[1]).to(kwargs['device'])
                prototypes.append(proto)
            prototypes = torch.stack(prototypes)
            if disable_tcga_fingerprint:
                loss_tcga_fingerprint = torch.tensor(0.0, requires_grad=True, device=kwargs['device'])
            else:
                loss_tcga_fingerprint = tcga_fingerprint_loss_ablation(
                    model, tcga_gene_all, tcga_ccc_features, drug_embedding_smiles, drug_embedding_MOA,
                    prototypes, t_cancer, t_ccc_embed, t_contextual_gene, supcon_criterion2, kwargs['device']
                )
   
            
            loss = 1 * (loss_predic + coral_loss  + loss_f1 )+ 0.4 * (
               1*loss_supcon +  1*loss_tcga_fingerprint + loss_rank
            )#                                             #700 
            loss.backward()
            optimizer.step()
            training_loss_sum += loss.cpu().detach().item()

            y_true.extend(labels_real.cpu().detach().numpy().tolist())
            probabilities = torch.sigmoid(predictions)
            y_pred.extend(probabilities.cpu().detach().numpy().tolist())

            result_in_batch = torch.zeros_like(labels)
            j = 0
            for n in range(mask.shape[1]):
                for m in range(mask.shape[0]):
                    if mask[m][n] > 0:
                        result_in_batch[m][n] = torch.sigmoid(predictions[j])
                        j += 1
            result_per_epoch.append(result_in_batch.cpu().detach().numpy())
            labels_per_epoch.append(labels.cpu().detach().numpy())

        result_per_epoch = np.vstack(result_per_epoch)
        labels_per_epoch = np.vstack(labels_per_epoch)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        for drug_idx in range(result_per_epoch.shape[1]):
            drug_results = result_per_epoch[:, drug_idx]
            drug_labels = labels_per_epoch[:, drug_idx]
            precision, recall, f1, prauc, roc_auc, accuracy = bina_metric(
                drug_labels, (drug_results >= 0.5).astype(int), drug_results
            )

        precision, recall, f1, prauc, roc_auc, accuracy = bina_metric(y_true, (y_pred >= 0.5).astype(int), y_pred)
        print(
            f"Epoch {epoch+1}: "
            f"training loss: {training_loss_sum:.5f}, "
            f"Precision: {precision:.5f}, "
            f"Recall: {recall:.5f}, "
            f"F1: {f1:.5f}, "
            f"PRAUC: {prauc:.5f}, "
            f"ROC_AUC: {roc_auc:.5f}. "
        )


        model.eval()
        val_loss, _, val_y_true, val_y_pred = multi_eval_epoch2(
            drug_embedding_smiles=drug_embedding_smiles,
            drug_embedding_MOA=drug_embedding_MOA,
            model=model,
            loader=val_dataloader,
            device=kwargs['device']
        )
        val_precision, val_recall, val_f1, val_prauc, val_roc_auc, val_accuracy = bina_metric(
            val_y_true, (val_y_pred >= 0.5).astype(int), val_y_pred
        )
        print(
            f"validate loss: {val_loss:.5f}, "
            f"Precision: {val_precision:.5f}, "
            f"Recall: {val_recall:.5f}, "
            f"F1: {val_f1:.5f}, "
            f"PRAUC: {val_prauc:.5f}, "
            f"ROC_AUC: {val_roc_auc:.5f}. "
        )

        if val_roc_auc > val_rocauc:
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(kwargs['model_save_folder'], 'DRPmodel.pt'))
            val_rocauc = val_roc_auc



        if epoch - best_epoch > 15:
            break

    model.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'DRPmodel.pt')))

    return model


def training1(drug_embedding_smiles,drug_embedding_MOA,cancer_encoder,ccc_encoder,ccc_gate_layer,s_dataloader,train_ids,test_ids,val_dataloader,tcga_dataloader, **kwargs):
    return training2(
        drug_embedding_smiles, drug_embedding_MOA, cancer_encoder, ccc_encoder, ccc_gate_layer,
        s_dataloader, train_ids, test_ids, val_dataloader, tcga_dataloader,
        model_cls=DRPmodel1, log_dir="logs/stage2_ablation1", **kwargs
    )


def training3(drug_embedding_smiles,drug_embedding_MOA,cancer_encoder,ccc_encoder,ccc_gate_layer,s_dataloader,train_ids,test_ids,val_dataloader,tcga_dataloader, **kwargs):
    return training2(
        drug_embedding_smiles, drug_embedding_MOA, cancer_encoder, ccc_encoder, ccc_gate_layer,
        s_dataloader, train_ids, test_ids, val_dataloader, tcga_dataloader,
        model_cls=DRPmodel3, log_dir="logs/stage2_ablation3", **kwargs
    )




def training7(drug_embedding_smiles,drug_embedding_MOA,cancer_encoder,ccc_encoder,ccc_gate_layer,s_dataloader,train_ids,test_ids,val_dataloader,tcga_dataloader, **kwargs):
    return training2(
        drug_embedding_smiles, drug_embedding_MOA, cancer_encoder, ccc_encoder, ccc_gate_layer,
        s_dataloader, train_ids, test_ids, val_dataloader, tcga_dataloader,
        model_cls=DRPmodel, disable_supcon=True, log_dir="logs/stage2_ablation7", **kwargs
    )


def training8(drug_embedding_smiles,drug_embedding_MOA,cancer_encoder,ccc_encoder,ccc_gate_layer,s_dataloader,train_ids,test_ids,val_dataloader,tcga_dataloader, **kwargs):
    return training2(
        drug_embedding_smiles, drug_embedding_MOA, cancer_encoder, ccc_encoder, ccc_gate_layer,
        s_dataloader, train_ids, test_ids, val_dataloader, tcga_dataloader,
        model_cls=DRPmodel, disable_tcga_fingerprint=True, log_dir="logs/stage2_ablation8", **kwargs
    )

 
def training9(drug_embedding_smiles,drug_embedding_MOA,cancer_encoder,ccc_encoder,ccc_gate_layer,s_dataloader,train_ids,test_ids,val_dataloader,tcga_dataloader, **kwargs):
    """Compatibility entry: old Ablation 9 is now the no-rank full baseline."""
    return training2(
        drug_embedding_smiles, drug_embedding_MOA, cancer_encoder, ccc_encoder, ccc_gate_layer,
        s_dataloader, train_ids, test_ids, val_dataloader, tcga_dataloader,
        model_cls=DRPmodel, disable_rank=True, log_dir="logs/stage2_ablation9", **kwargs
    )


def training(subdrug_Encoder=None, substruct_graphs=None, drug_embedding_smiles=None, drug_embedding_MOA=None,
             cancer_encoder=None, ccc_encoder=None, ccc_gate_layer=None, s_dataloader=None,
             train_ids=None, test_ids=None, val_dataloader=None, tcga_dataloader=None, **kwargs):
    """Complete ablation3 baseline: MOA -> main branch, SMILES -> CCC/TME gate, no rank loss."""
    return training2(
        drug_embedding_smiles, drug_embedding_MOA, cancer_encoder, ccc_encoder, ccc_gate_layer,
        s_dataloader, train_ids, test_ids, val_dataloader, tcga_dataloader,
        model_cls=DRPmodel, disable_rank=False, log_dir="logs/stage2_full_model10_with_rank", **kwargs
    )


def testing(substruct_graphs=None, drug_embedding_smiles=None, drug_embedding_MOA=None,
            model=None, t_dataloader=None, device=None):
    """Complete model testing entry matching DRPmodel10."""
    return testing2(
        drug_embedding_smiles=drug_embedding_smiles,
        drug_embedding_MOA=drug_embedding_MOA,
        model=model,
        t_dataloader=t_dataloader,
        device=device,
    )











