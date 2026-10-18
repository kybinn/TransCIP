import torch
import torch.nn as nn
from torch.nn import functional as F
from copy import deepcopy
import numpy as np
from torch.autograd import Function

from torch.utils.checkpoint import checkpoint
def random_zero(tensor, probability):
    mask = torch.bernoulli(torch.full_like(tensor, probability))
    return tensor * mask


def get_weight(att_mat,device):
    att_mat = torch.stack(att_mat).squeeze(1)
    #print(att_mat.size())
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=2)
    #print(att_mat.size())
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(3))
    aug_att_mat = att_mat.to(device) + residual_att.to(device)
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    #print(aug_att_mat.size())
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]
    
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    #print(joint_attentions.size())
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    #print(v.size())
    v = v[:,0,1:]
    #print(v.size())
    return v

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)  


################ PART2 
class extractionForCancer(nn.Module):
    def __init__(self, dim_input, dim_hid1, dim_hid2, dim_final):
        super(extractionForCancer, self).__init__()
        mlp_layers = [
            nn.Linear(dim_input, dim_hid1),
            nn.ReLU(),
            nn.Linear(dim_hid1, dim_hid2),
            nn.ReLU(),
            nn.Linear(dim_hid2, dim_final)
        ]
        self.mlp = nn.Sequential(*mlp_layers)
        # self.geneTransformer = Transformer( 
        #                 mask = mask,
        #                 embed_dim=embed_dim,
        #                 depth=depth,
        #                 num_heads=num_heads,
        #                 drop_ratio=0.5, attn_drop_ratio=0.5, drop_path_ratio=0.5,
        #                 representation_size=embed_dim if has_logits else None)
        
    def forward(self, RNAseq):
        # x_r = self.RNAseq(RNAseq, adj_1)
        RNA_embedding = self.mlp(RNAseq)
        
        return RNA_embedding



################ PART3: Used for CCC features 
class extractionForCCC(nn.Module):
    def __init__(self, dim_input=386, dim_hid=128, dim_final=64):
        super(extractionForCCC, self).__init__()
        mlp_layers = [
            nn.Linear(dim_input, dim_hid),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
            nn.Linear(dim_hid, dim_final)
        ]
        self.mlp = nn.Sequential(*mlp_layers)
        
        
    def forward(self, ccc_features):
        ccc_embedding = self.mlp(ccc_features)
        return ccc_embedding




class BipartiteCCC(nn.Module):
    def __init__(self, ligand_gene_ids, receptor_gene_ids, num_genes=208,
                 d_model=96, output_dim=64):
        super().__init__()
        n_pairs = len(ligand_gene_ids)
        self.n_pairs = n_pairs
        self.num_genes = num_genes

        self.register_buffer('lig_ids',
                             torch.tensor(ligand_gene_ids, dtype=torch.long))
        self.register_buffer('rec_ids',
                             torch.tensor(receptor_gene_ids, dtype=torch.long))

        self.lig_embed = nn.Embedding(num_genes, d_model) 
        self.rec_embed = nn.Embedding(num_genes, d_model) 
        nn.init.xavier_uniform_(self.lig_embed.weight)
        nn.init.xavier_uniform_(self.rec_embed.weight)

        self.score_scale = nn.Parameter(torch.tensor(1.0))

        
        self.gene_update = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        self.pair_update = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

        
        gene_deg = torch.zeros(num_genes)
        gene_deg.scatter_add_(0, self.lig_ids,
                              torch.ones(n_pairs, dtype=torch.float))
        gene_deg.scatter_add_(0, self.rec_ids,
                              torch.ones(n_pairs, dtype=torch.float))
        self.register_buffer('gene_deg', gene_deg.clamp(min=1))

        
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, ccc_features):
        B, E = ccc_features.shape
        D = self.lig_embed.embedding_dim
        device = ccc_features.device

        # Step 0: pair tokens = (lig_emb + rec_emb) × score
        lig = self.lig_embed(self.lig_ids)                    # [386, D]
        rec = self.rec_embed(self.rec_ids)                    # [386, D]
        pair = (lig + rec).unsqueeze(0)                        # [1, 386, D] 
        score = (F.softplus(self.score_scale.abs() * ccc_features)
                 - 0.6931).unsqueeze(-1)                       # [B, 386, 1]
        pair = pair * score                                    # [B, 386, D]

        # Step 1: Pairs → Genes
        batch_off = (torch.arange(B, device=device)
                     * self.num_genes).view(B, 1)
        lig_flat = (self.lig_ids.unsqueeze(0) + batch_off).reshape(-1)
        rec_flat = (self.rec_ids.unsqueeze(0) + batch_off).reshape(-1)

        gene = torch.zeros(B * self.num_genes, D, device=device)
        gene.index_add_(0, lig_flat, pair.reshape(-1, D))
        gene.index_add_(0, rec_flat, pair.reshape(-1, D))
        gene = self.gene_update(
            gene.reshape(B, self.num_genes, D)
            / self.gene_deg.view(1, -1, 1))                   # [B, G, D]

        # Step 2: Genes → Pairs（
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, 386)
        g_lig = gene[b_idx, self.lig_ids.unsqueeze(0).expand(B, -1)] 
        g_rec = gene[b_idx, self.rec_ids.unsqueeze(0).expand(B, -1)]  
        pair = pair + self.pair_update(
            torch.cat([g_lig, g_rec], dim=-1))               

        # Step 3: pooling
        w = torch.softmax(
            self.attn_pool(pair).squeeze(-1), dim=-1)          # [B, 386]
        out = (pair * w.unsqueeze(-1)).sum(dim=1)              # [B, D]

        return self.output_proj(out)                           # [B, output_dim]
    


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, drop=0.1, act_fn=nn.SELU, **kwargs):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.drop = drop
        hidden_dims = deepcopy(hidden_dims)
        hidden_dims.insert(0, input_dim)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    act_fn(),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.Dropout(self.drop))
            )
        self.module = nn.Sequential(*modules)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dims[-1], output_dim, bias=True))
        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, inputs):
        embed = self.module(inputs)
        output = self.output_layer(embed)
        return output



class DRPmodel10(nn.Module):
    """
    full/base model.
 
        MOA    -> main response branch
        SMILES -> CCC/TME gate branch

    Prediction:
        prediction = pred_real_tme
    """

    def __init__(self, *args, emb_dim=64, dropout=0.1, max_tme_scale=1, max_gate_delta_scale=1):
        super(DRPmodel10, self).__init__()

        if len(args) == 5:
            self.subdrug_Encoder = args[0]
            cancer_encoder, ccc_encoder, ccc_gate_layer, emb_dim_from_args = args[1:]
            emb_dim = emb_dim_from_args
        elif len(args) == 4:
            self.subdrug_Encoder = None
            cancer_encoder, ccc_encoder, ccc_gate_layer, emb_dim_from_args = args
            emb_dim = emb_dim_from_args
        else:
            self.subdrug_Encoder = kwargs.pop("subdrug_Encoder", None)
            cancer_encoder = kwargs.pop("cancer_encoder")
            ccc_encoder = kwargs.pop("ccc_encoder")
            ccc_gate_layer = kwargs.pop("ccc_gate_layer")
            emb_dim = kwargs.pop("emb_dim", emb_dim)

        self.cancer_encoder = cancer_encoder
        self.ccc_encoder = ccc_encoder
        self.ccc_gate_layer = ccc_gate_layer
        self.emb_dim = emb_dim
        self.max_tme_scale = max_tme_scale
        self.max_gate_delta_scale = max_gate_delta_scale

        self.log_vars = nn.Parameter(torch.zeros(9))

        self.drug_mapping1 = nn.Sequential(
            nn.Linear(384, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
        )

        self.moa_ln = nn.LayerNorm(768)
        self.drug_mapping2 = nn.Sequential(
            nn.Linear(768, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
        )

        self.base_cancer_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.base_head = nn.Linear(emb_dim, 1)

        fusion_dim = emb_dim * 4

        self.drug_delta_fuser = nn.Sequential(
            nn.Linear(fusion_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )
        self.drug_delta_head = nn.Linear(emb_dim, 1)

        self.moa_gate_mapper = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.gate_delta_fuser = nn.Sequential(
            nn.Linear(fusion_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )
        self.gate_delta_head = nn.Linear(emb_dim, emb_dim)

        # Small residual scales protect small-sample stability.
        self.max_drug_residual_scale = 0.10
        self.drug_residual_scale_logit = nn.Parameter(torch.tensor(-3.0))
        self.gate_delta_scale_logit = nn.Parameter(torch.tensor(-3.0))
        self.tme_scale_logit = nn.Parameter(torch.tensor(-3.0))

        nn.init.zeros_(self.drug_delta_head.weight)
        nn.init.zeros_(self.drug_delta_head.bias)
        nn.init.zeros_(self.gate_delta_head.weight)
        nn.init.zeros_(self.gate_delta_head.bias)

    def _to_tensor(self, x, device):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=torch.float32)

        if isinstance(x, (list, tuple)) and len(x) > 0 and torch.is_tensor(x[0]):
            return torch.stack(
                [v.to(device=device, dtype=torch.float32) for v in x],
                dim=0,
            )

        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def _prepare_embedding(self, x, batch_size, device, mapper):
        x = self._to_tensor(x, device)

        if x.dim() == 1:
            x = x.unsqueeze(0).expand(batch_size, -1)
        elif x.dim() == 2 and x.size(0) == 1 and batch_size > 1:
            x = x.expand(batch_size, -1)
        elif x.dim() == 2 and x.size(0) != batch_size:
            raise ValueError(
                f"Embedding batch size {x.size(0)} does not match sample batch size {batch_size}."
            )

        return mapper(x)

    def _pair_features(self, x, d):
        return torch.cat(
            [x, d, x * d, torch.abs(x - d)],
            dim=-1,
        )

    def _map_smiles(self, drug_embedding_simle, batch_size, device):
        return self._prepare_embedding(
            drug_embedding_simle,
            batch_size,
            device,
            self.drug_mapping1,
        )

    def _map_moa(self, drug_embedding_MOA, batch_size, device):
        return self._prepare_embedding(
            drug_embedding_MOA,
            batch_size,
            device,
            lambda x: self.drug_mapping2(self.moa_ln(x)),
        )

    def _main_branch(self, C, drug_embedding):
        base_context = self.base_cancer_proj(C)
        e_base = self.base_head(base_context).squeeze(-1)

        drug_delta_context = self.drug_delta_fuser(
            self._pair_features(C, drug_embedding)
        )
        drug_delta_raw = self.drug_delta_head(drug_delta_context).squeeze(-1)

        drug_scale = self.max_drug_residual_scale * torch.sigmoid(
            self.drug_residual_scale_logit
        )

        main_logit = e_base + drug_scale * drug_delta_raw
        response_context = base_context + drug_scale * drug_delta_context

        return main_logit, response_context, e_base, drug_delta_raw, drug_scale

    def _drug_specific_gate(self, B1, base_gate_logit, condition_embedding):
        B1_norm = F.layer_norm(B1, B1.shape[-1:]) #700 720
        # B1_norm = self.ccc_gate_layer(B1)       # 720  
        condition_gate = self.moa_gate_mapper(condition_embedding)

        gate_context = self.gate_delta_fuser(
            self._pair_features(B1_norm, condition_gate)
        )
        gate_delta_logit = self.gate_delta_head(gate_context)

        gate_delta_scale = self.max_gate_delta_scale * torch.sigmoid(
            self.gate_delta_scale_logit
        )

        drug_specific_gate = torch.sigmoid(
            base_gate_logit + gate_delta_scale * gate_delta_logit
        )

        return drug_specific_gate, gate_delta_logit, gate_delta_scale, condition_gate, gate_context

    def forward_components(
        self,
        gene_all,
        ccc_features,
        drug_embedding_simle,
        drug_embedding_MOA,
        *args,
        **kwargs,
    ):
        device = gene_all.device
        batch_size = gene_all.size(0)

        gene_all = gene_all.to(device=device, dtype=torch.float32)
        ccc_features = ccc_features.to(device=device, dtype=torch.float32)

        A = self.cancer_encoder(gene_all)
        B1 = self.ccc_encoder(ccc_features)

        base_gate_logit = self.ccc_gate_layer(B1)
        base_gate = torch.sigmoid(base_gate_logit)
        C_base = A * base_gate

        smiles_embedding = self._map_smiles(drug_embedding_simle, batch_size, device)
        raw_moa_embedding = self._map_moa(drug_embedding_MOA, batch_size, device)
        main_drug_embedding = raw_moa_embedding 
        gate_drug_embedding = smiles_embedding 

         
        drug_specific_gate, gate_delta_logit, gate_delta_scale, condition_gate, gate_context = self._drug_specific_gate(
            B1,
            base_gate_logit,
            gate_drug_embedding,
        )

        C_real_tme = A * drug_specific_gate

        pred_real_tme, real_response_context, _, _, _ = self._main_branch(
            C_real_tme,
            main_drug_embedding,
        )

         

        prediction = pred_real_tme 

         
        return {
            "prediction": prediction,
            "real_response_context": real_response_context,
        }

    def forward(
        self,
        gene_all,
        ccc_features,
        drug_embedding_simle,
        drug_embedding_MOA,
        *args,
        **kwargs,
    ):
        out = self.forward_components(
            gene_all,
            ccc_features,
            drug_embedding_simle,
            drug_embedding_MOA,
            *args,
            **kwargs,
        )
        return out["prediction"], out["real_response_context"] 
    

class DRPmodel(DRPmodel10):
    """Alias for the current complete model. Loss-only ablations use this."""

    pass


class DRPmodel1(DRPmodel10):
    """Ablation 1: no CCC, no TME gate.

    This is a clean no-TME baseline: no BipartiteCCC, no ccc_gate_layer, no
    pseudo gene-derived gate. The prediction uses only the gene encoder and the
    new full model's main drug input, i.e. MOA.
    """

    def __init__(self, cancer_encoder, ccc_encoder, ccc_gate_layer, emb_dim):
        super(DRPmodel1, self).__init__(cancer_encoder, None, None, emb_dim)

    def forward_components(self, gene_all, *inputs, **kwargs):
        if len(inputs) == 2:
            drug_embedding_simle, drug_embedding_MOA = inputs
        elif len(inputs) >= 3:
            _, drug_embedding_simle, drug_embedding_MOA = inputs[:3]
        else:
            raise ValueError("DRPmodel1 expects gene_all plus SMILES/MOA embeddings.")

        device = gene_all.device
        batch_size = gene_all.size(0)
        gene_all = gene_all.to(device=device, dtype=torch.float32) 
        raw_moa_embedding = self._map_moa(drug_embedding_MOA, batch_size, device)
        main_drug_embedding = raw_moa_embedding

        A = self.cancer_encoder(gene_all) 
        C_base = A
        main_logit, response_context, e_base, drug_delta_raw, drug_scale = self._main_branch(
            C_base,
            main_drug_embedding,
        ) 
        return {
            "prediction": main_logit,
            "real_response_context": response_context,
        }

    def forward(self, gene_all, *inputs, **kwargs):
        out = self.forward_components(gene_all, *inputs, **kwargs)
        return out["prediction"], out["real_response_context"] 


class DRPmodel2(DRPmodel10):
    """Ablation 2: replace BipartiteCCC with a plain MLP CCC encoder.

    The forward graph is identical to the complete model. ablation_common.py
    provides an MLP-pretrained ccc_encoder for this class.
    """

    pass


class DRPmodel3(nn.Module):
    

    def __init__(self, *args, emb_dim=64, dropout=0.1, max_tme_scale=1, max_gate_delta_scale=1):
        super(DRPmodel3, self).__init__()

        if len(args) == 5:
            self.subdrug_Encoder = args[0]
            cancer_encoder, ccc_encoder, ccc_gate_layer, emb_dim_from_args = args[1:]
            emb_dim = emb_dim_from_args
        elif len(args) == 4:
            self.subdrug_Encoder = None
            cancer_encoder, ccc_encoder, ccc_gate_layer, emb_dim_from_args = args
            emb_dim = emb_dim_from_args
        else:
            self.subdrug_Encoder = kwargs.pop("subdrug_Encoder", None)
            cancer_encoder = kwargs.pop("cancer_encoder")
            ccc_encoder = kwargs.pop("ccc_encoder")
            ccc_gate_layer = kwargs.pop("ccc_gate_layer")
            emb_dim = kwargs.pop("emb_dim", emb_dim)

        self.cancer_encoder = cancer_encoder
        self.ccc_encoder = ccc_encoder
        self.ccc_gate_layer = ccc_gate_layer
        self.emb_dim = emb_dim
        self.max_tme_scale = max_tme_scale
        self.max_gate_delta_scale = max_gate_delta_scale

        self.log_vars = nn.Parameter(torch.zeros(9))

        self.drug_mapping1 = nn.Sequential(
            nn.Linear(384, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
        )

        self.moa_ln = nn.LayerNorm(768)
        self.drug_mapping2 = nn.Sequential(
            nn.Linear(768, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
        )

        self.base_cancer_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.base_head = nn.Linear(emb_dim, 1)

        fusion_dim = emb_dim * 4

        self.drug_delta_fuser = nn.Sequential(
            nn.Linear(fusion_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )
        self.drug_delta_head = nn.Linear(emb_dim, 1)

        self.moa_gate_mapper = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.gate_delta_fuser = nn.Sequential(
            nn.Linear(fusion_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )
        self.gate_delta_head = nn.Linear(emb_dim, emb_dim)

        # Small residual scales protect small-sample stability.
        self.max_drug_residual_scale = 0.10
        self.drug_residual_scale_logit = nn.Parameter(torch.tensor(-3.0))
        self.gate_delta_scale_logit = nn.Parameter(torch.tensor(-3.0))
        self.tme_scale_logit = nn.Parameter(torch.tensor(-3.0))

        nn.init.zeros_(self.drug_delta_head.weight)
        nn.init.zeros_(self.drug_delta_head.bias)
        nn.init.zeros_(self.gate_delta_head.weight)
        nn.init.zeros_(self.gate_delta_head.bias)

    def _to_tensor(self, x, device):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=torch.float32)

        if isinstance(x, (list, tuple)) and len(x) > 0 and torch.is_tensor(x[0]):
            return torch.stack(
                [v.to(device=device, dtype=torch.float32) for v in x],
                dim=0,
            )

        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def _prepare_embedding(self, x, batch_size, device, mapper):
        x = self._to_tensor(x, device)

        if x.dim() == 1:
            x = x.unsqueeze(0).expand(batch_size, -1)
        elif x.dim() == 2 and x.size(0) == 1 and batch_size > 1:
            x = x.expand(batch_size, -1)
        elif x.dim() == 2 and x.size(0) != batch_size:
            raise ValueError(
                f"Embedding batch size {x.size(0)} does not match sample batch size {batch_size}."
            )

        return mapper(x)

    def _pair_features(self, x, d):
        return torch.cat(
            [x, d, x * d, torch.abs(x - d)],
            dim=-1,
        )

    def _map_smiles(self, drug_embedding_simle, batch_size, device):
        return self._prepare_embedding(
            drug_embedding_simle,
            batch_size,
            device,
            self.drug_mapping1,
        )

    def _map_moa(self, drug_embedding_MOA, batch_size, device):
        return self._prepare_embedding(
            drug_embedding_MOA,
            batch_size,
            device,
            lambda x: self.drug_mapping2(self.moa_ln(x)),
        )

    def _main_branch(self, C, drug_embedding):
       
        base_context = self.base_cancer_proj(C)
        e_base = self.base_head(base_context).squeeze(-1)

         
        main_logit = self.base_head(self.base_cancer_proj(C+drug_embedding)).squeeze(-1)

        return main_logit, base_context, e_base 

    def _drug_specific_gate(self, B1, base_gate_logit, condition_embedding):
        B1_norm = F.layer_norm(B1, B1.shape[-1:]) #700 720
        # B1_norm = self.ccc_gate_layer(B1)       # 720  
        condition_gate = self.moa_gate_mapper(condition_embedding)

        gate_delta_logit = self.gate_delta_head(B1_norm+condition_gate)

        drug_specific_gate = torch.sigmoid(
            gate_delta_logit
        )

        return drug_specific_gate, gate_delta_logit

    def forward_components(
        self,
        gene_all,
        ccc_features,
        drug_embedding_simle,
        drug_embedding_MOA,
        *args,
        **kwargs,
    ):
        device = gene_all.device
        batch_size = gene_all.size(0)

        gene_all = gene_all.to(device=device, dtype=torch.float32)
        ccc_features = ccc_features.to(device=device, dtype=torch.float32)

        A = self.cancer_encoder(gene_all)
        B1 = self.ccc_encoder(ccc_features)

        base_gate_logit = self.ccc_gate_layer(B1)
        base_gate = torch.sigmoid(base_gate_logit)
        C_base = A * base_gate

        smiles_embedding = self._map_smiles(drug_embedding_simle, batch_size, device)
        raw_moa_embedding = self._map_moa(drug_embedding_MOA, batch_size, device)
        main_drug_embedding = raw_moa_embedding#smiles_embedding
        gate_drug_embedding = smiles_embedding#raw_moa_embedding

         
        drug_specific_gate, gate_delta_logit  = self._drug_specific_gate(
            B1,
            base_gate_logit,
            gate_drug_embedding,
        )

        C_real_tme = A * drug_specific_gate

        pred_real_tme, real_response_context, _ = self._main_branch(
            C_real_tme,
            main_drug_embedding,
        )
 
        prediction = pred_real_tme 

        return {
            "prediction": prediction,

            "real_response_context": real_response_context,
        }

    def forward(
        self,
        gene_all,
        ccc_features,
        drug_embedding_simle,
        drug_embedding_MOA,
        *args,
        **kwargs,
    ):
        out = self.forward_components(
            gene_all,
            ccc_features,
            drug_embedding_simle,
            drug_embedding_MOA,
            *args,
            **kwargs,
        )
        return out["prediction"], out["real_response_context"] 
    


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential()
        self.layer.add_module('fc1', nn.Linear(input_dim, input_dim//4))
        self.layer.add_module('bn1', nn.BatchNorm1d(input_dim//4))
        self.layer.add_module('relu1', nn.SELU(True))
        self.layer.add_module('fc2', nn.Linear(input_dim//4, input_dim//8))
        self.layer.add_module('bn2', nn.BatchNorm1d(input_dim//8))
        self.layer.add_module('relu2', nn.SELU(True))
        self.layer.add_module('fc3', nn.Linear(input_dim//8, 1))

    def forward(self, input_data):
        output = self.layer(input_data)
        return output


