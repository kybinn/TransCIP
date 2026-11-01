from collections import OrderedDict
from ogb.utils import smiles2graph
import numpy as np
import torch
from torch_geometric.data import Data


from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Descriptors
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import config
import re

def process_substructure_batch(substru_list):
    
    num_drugs = len(substru_list) 
    total_substructures = sum(len(sub_list) for sub_list in substru_list)

    all_substructures = []
    mask_np = np.zeros((num_drugs, total_substructures), dtype=np.float32)

    current_idx = 0
    for drug_idx, sub_smiles_list in enumerate(substru_list):
        for smiles in sub_smiles_list:
            all_substructures.append(smiles)
            mask_np[drug_idx, current_idx] = 1.0
            current_idx += 1

    drug_mask_matrix = torch.tensor(mask_np, dtype=torch.float32)
    return all_substructures, drug_mask_matrix



def clean_fragment_smiles(frag_smiles: str) -> str:
   
    mol = Chem.MolFromSmiles(frag_smiles)

    if mol is None:
        return ""

    try:
        cleaned_mol = Chem.ReplaceAtoms(
            mol,
            [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"],
            [Chem.Atom(1) for atom in mol.GetAtoms() if atom.GetSymbol() == "*"],
        )

        cleaned_smiles = Chem.MolToSmiles(cleaned_mol)
        return cleaned_smiles

    except Exception as e:
        cleaned = re.sub(r"\d*\*+", "H", frag_smiles)
        mol_fallback = Chem.MolFromSmiles(cleaned)
        if mol_fallback is not None:
            return Chem.MolToSmiles(mol_fallback)

        return frag_smiles
    

def get_substructure_smiles(main_smiles: str) -> list:
    mol = Chem.MolFromSmiles(main_smiles)
    if mol is None:
        return []
    fragment_mols = BRICS.BreakBRICSBonds(mol)
    sub_mol_objects = Chem.GetMolFrags(fragment_mols, asMols=True)

    sub_smiles_list = []
    for sub_mol in sub_mol_objects:
        frag_smiles = Chem.MolToSmiles(sub_mol)
        sub_smiles_list.append(frag_smiles)
    return sub_smiles_list



def batch_process_smiles(smiles_list: list) -> list:
    batch_cleaned_fragments = []
    for main_smiles in smiles_list:
        raw_fragments = get_substructure_smiles(main_smiles)
        cleaned_fragments_for_this_drug = []

        for frag in raw_fragments:

            cleaned_frag = clean_fragment_smiles(frag)
            if cleaned_frag:
                cleaned_fragments_for_this_drug.append(cleaned_frag)

        batch_cleaned_fragments.append(cleaned_fragments_for_this_drug)
    return batch_cleaned_fragments


def get_drug_MOA(text_list, device="cuda", batch_size=1, pooling="cls"):

    device = torch.device(device if torch.cuda.is_available() else "cpu")


    model_name = "utils/BiomedBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_embeddings = []


    with torch.no_grad():
        for i in tqdm(
            range(0, len(text_list), batch_size), desc="PubMedBERT MoA embedding"
        ):
            batch = text_list[i : i + batch_size]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  


            if pooling == "cls":
                emb = hidden_states[:, 0, :]  
            else:
               
                emb = hidden_states.mean(dim=1)

            all_embeddings.append(emb.cpu().numpy())


    all_embeddings = np.vstack(all_embeddings)
    print(
        f"[PubMedBERT] Finished encoding {len(text_list)} mechanisms | dim={all_embeddings.shape[1]}"
    )

    return all_embeddings

def smiles_to_chemberta_embeddings(
    smiles_list, device="cuda", batch_size=1, pooling="cls"
):
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model_name = "utils/ChemBERTa-77M-MTR"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(
            range(0, len(smiles_list), batch_size), desc="ChemBERTa embedding"
        ):
            batch = smiles_list[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=256,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # (B, L, H)

            if pooling == "cls":
                emb = hidden_states[:, 0, :]  # [CLS] token
            else:
                emb = hidden_states.mean(dim=1)

            all_embeddings.append(emb.cpu().numpy())

    all_embeddings = np.vstack(all_embeddings)
    print(
        f"[ChemBERTa] Finished encoding {len(smiles_list)} molecules | dim={all_embeddings.shape[1]}"
    )
    return all_embeddings



def get_drug_ecfp(smiles, radius=2, n_bits=2048):

    _morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, 
        fpSize=2048, 
        includeChirality=True
    )
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = _morgan_gen.GetFingerprint(mol)
        return np.array(fp).astype('float32')
    except:
        return None

def get_drug_tme_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        
        features = {
            "MolLogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "MolWt": Descriptors.MolWt(mol),
    
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "HAcceptors": Descriptors.NumHAcceptors(mol),
            "HDonors": Descriptors.NumHDonors(mol)
        }
        return features
    except:
        return None


def graph_from_smile(smiles):
    graph = smiles2graph(smiles) 
    result = {
        'edge_index': graph['edge_index'],
        'edge_attr': graph['edge_feat'],
        'x': graph['node_feat']
    }
    result = {k: torch.from_numpy(v) for k, v in result.items()}
    
    return Data(**result)



def graph_batch_from_smile(smiles_list):
    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    graphs = [smiles2graph(x) for x in smiles_list]
    for idx, graph in enumerate(graphs):
        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])
        lstnode += graph['num_nodes']
        batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0)
    }
    result = {k: torch.from_numpy(v) for k, v in result.items()}
    result['num_nodes'] = lstnode
    return Data(**result)


def read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    
    """
    Read GMT file into dictionary of gene_module:genes.\n
    min_g and max_g are optional gene set size filters.

    Args:
        fname (str): Path to gmt file
        sep (str): Separator used to read gmt file.
        min_g (int): Minimum of gene members in gene module.
        max_g (int): Maximum of gene members in gene module.
    Returns:
        OrderedDict: Dictionary of gene_module:genes.
    """
    dict_pathway = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_pathway[val[0]] = val[2:]
    return dict_pathway

def create_pathway_mask(feature_list, dict_pathway, add_missing=1, fully_connected=True, to_tensor=False):
   
    """
    Creates a mask of shape [genes,pathways] where (i,j) = 1 if gene i is in pathway j, 0 else.

    Expects a list of genes and pathway dict.
    Note: dict_pathway should be an Ordered dict so that the ordering can be later interpreted.

    Args:
        feature_list (list): List of genes in single-cell dataset.
        dict_pathway (OrderedDict): Dictionary of gene_module:genes.
        add_missing (int): Number of additional, fully connected nodes.
        fully_connected (bool): Whether to fully connect additional nodes or not.
        to_tensor (False): Whether to convert mask to tensor or not.
    Returns:
        torch.tensor/np.array: Gene module mask.
    """
    assert type(dict_pathway) == OrderedDict
    p_mask = np.zeros((len(feature_list), len(dict_pathway)))
    pathway = list()
    for j, k in enumerate(dict_pathway.keys()):
        pathway.append(k)
        for i in range(p_mask.shape[0]):
            if feature_list[i] in dict_pathway[k]:
                p_mask[i,j] = 1.
    if add_missing:
        n = 1 if type(add_missing)==bool else add_missing
        # Get non connected genes
        if not fully_connected:
            idx_0 = np.where(np.sum(p_mask, axis=1)==0)
            vec = np.zeros((p_mask.shape[0],n))
            vec[idx_0,:] = 1.
        else:
            vec = np.ones((p_mask.shape[0], n))
        p_mask = np.hstack((p_mask, vec))
        for i in range(n):
            x = 'node %d' % i
            pathway.append(x)
    if to_tensor:
        p_mask = torch.Tensor(p_mask)
    return p_mask,np.array(pathway)

def get_mask(gmt_path,mask_col,genes):
    
    mask_ratio = 0.015
    if gmt_path is None:
        mask = np.random.binomial(1,mask_ratio,size=(genes, mask_col)) 
        
        pathway = list()
        for i in range(mask_col):
            x = 'node %d' % i
            pathway.append(x)
        print('Full connection!')
    else:
        
        print('Loading mask from gmt file...')
        path_dict = read_gmt(gmt_path) 
        
        mask,pathway = create_pathway_mask(feature_list=genes,
                                          dict_pathway=path_dict)
                                          
        pathway = pathway[np.sum(mask,axis=0)>4]
        mask = mask[:,np.sum(mask,axis=0)>4]
        
        
        pathway = pathway[sorted(np.argsort(np.sum(mask,axis=0))[-min(mask_col,mask.shape[1]):])]
        mask = mask[:,sorted(np.argsort(np.sum(mask,axis=0))[-min(mask_col,mask.shape[1]):])]
        
        index_nonzero = np.where(np.sum(mask, axis=1)>0)[0]
        mask = mask[index_nonzero,:]
        
        print('Mask loaded!')
    return torch.Tensor(mask), pathway

def set_seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

