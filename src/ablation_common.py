import argparse
import itertools
import json
import os
import config
os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_ID

import time
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

import dataload
import finetuning
import pretraining
from utils.tools import get_drug_MOA, set_seed_all, smiles_to_chemberta_embeddings


ABLATION_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ABLATION_DIR, os.pardir))


DRUGS_9 = [
    "5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel",
    "Doxorubicin", "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"
]


def wrap_params(training_params, type="unlabeled"):
    aux_dict = {k: v for k, v in training_params.items() if k not in ["unlabeled", "labeled"]}
    aux_dict.update(**training_params[type])
    return aux_dict


def make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)


def dict_to_str(d):
    return "_".join(["_".join([k, str(v)]) for k, v in d.items()])


def experiment_name(ablation_id):
    return "full" if ablation_id == 0 else f"ablation{ablation_id}"


def build_lr_index(target_ccc):
    lr_pairs = target_ccc.columns.tolist()
    genes = set()
    edges = []
    for pair in lr_pairs:
        ligand, receptor = pair.split("_")
        genes.add(ligand)
        genes.add(receptor)
        edges.append((ligand, receptor))

    gene_to_id = {g: i for i, g in enumerate(sorted(genes))}
    edge_index_list = [[], []]
    ligand_gene_ids = [gene_to_id[p.split("_")[0]] for p in lr_pairs]
    receptor_gene_ids = [gene_to_id[p.split("_")[1]] for p in lr_pairs]
    for ligand, receptor in edges:
        edge_index_list[0].append(gene_to_id[ligand])
        edge_index_list[1].append(gene_to_id[receptor])
    base_edge_index = torch.tensor(edge_index_list, dtype=torch.long)
    return base_edge_index, ligand_gene_ids, receptor_gene_ids


def build_parser():
    parser = argparse.ArgumentParser("Ablation pretraining and fine-tuning")
    parser.add_argument("--metric", dest="metric", nargs="?", default="auroc", choices=["auroc", "auprc"])
    parser.add_argument("--measurement", dest="measurement", nargs="?", default="Z_SCORE", choices=["Z_SCORE", "LN_IC50", "AUC"])
    parser.add_argument("--thres_gdsc", dest="thres_g", nargs="?", type=float, default=0.0)
    parser.add_argument("--thres_label", dest="thres_s", nargs="?", type=float, default=0.1)
    parser.add_argument("--n", dest="n", nargs="?", type=int, default=5)
    parser.add_argument("--alph", dest="alph", nargs="?", type=float, default=0.005)
    parser.add_argument("--beta", dest="beta", nargs="?", type=float, default=0.005)

    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument("--train", dest="retrain_flag", action="store_true")
    train_group.add_argument("--no-train", dest="retrain_flag", action="store_false")
    parser.set_defaults(retrain_flag=False)

    norm_group = parser.add_mutually_exclusive_group(required=False)
    norm_group.add_argument("--norm", dest="norm_flag", action="store_true")
    norm_group.add_argument("--no-norm", dest="norm_flag", action="store_false")
    parser.set_defaults(norm_flag=True)
    return parser


def run_ablation(ablation_id, training_fn_name, params_grid=None, drug=None, force_retrain=None):
    args = build_parser().parse_args()
    if force_retrain is not None:
        args.retrain_flag = bool(force_retrain)
    if params_grid is None:
        params_grid = {"pretrain_num_epochs": [90], "train_num_epochs": [700]}# 
    if drug is None:
        drug = DRUGS_9

    keys, values = zip(*params_grid.items())
    params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for params_dict in params_list:
        main(args, drug, params_dict, ablation_id, training_fn_name)


def main(args, drug, params_dict, ablation_id, training_fn_name):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed_all(2024)

    gex_features_df = pd.read_csv(config.gex_feature_file, index_col=0)
    soure_ccc = pd.read_csv(config.soure_ccc_file, index_col=0)
    target_ccc = pd.read_csv(config.target_ccc_file, index_col=0)
    base_edge_index, ligand_gene_ids, receptor_gene_ids = build_lr_index(target_ccc)

    with open(os.path.join(ABLATION_DIR, "train_params.json"), "r") as f:
        training_params = json.load(f)
    training_params["unlabeled"].update(params_dict)
    training_params["labeled"].update(params_dict)
    param_str = dict_to_str(params_dict)
    exp_name = experiment_name(ablation_id)
    model_save_folder = os.path.join(ABLATION_DIR, "model_save", exp_name)

    ccc_encoder_type = "mlp" if ablation_id == 2 else "bipartite"
    pretrained_model_folder = os.path.join(PROJECT_ROOT, "src/model_save", "pretrain_num_epochs_90_train_num_epochs_700")
    if ablation_id == 2:
        pretrained_model_folder = os.path.join(model_save_folder, param_str)

    training_params.update({
        "device": device,
        "total_genes": gex_features_df.shape[1] - 1,
        "model_save_folder": os.path.join(model_save_folder, param_str),
        "pretrained_model_folder": pretrained_model_folder,
        "retrain_flag": args.retrain_flag,
        "ligand_gene_ids": ligand_gene_ids,
        "receptor_gene_ids": receptor_gene_ids,
        "ccc_encoder_type": ccc_encoder_type,
    })
    make_dir(training_params["model_save_folder"])
    if ablation_id == 1 and not training_params["retrain_flag"]:
        ablation1_encoder = os.path.join(training_params["model_save_folder"], "cancer_encoder.pt")
        if not os.path.exists(ablation1_encoder):
            print("Ablation 1 no-CCC pretrained encoder not found; retraining pretraining stage.")
            training_params["retrain_flag"] = True
    if ablation_id == 2 and not training_params["retrain_flag"]:
        required_files = ["cancer_encoder.pt", "ccc_encoder.pt", "ccc_gate_layer.pt"]
        missing_files = [
            name for name in required_files
            if not os.path.exists(os.path.join(training_params["model_save_folder"], name))
        ]
        if missing_files:
            print("Ablation 2 MLP-CCC pretrained files not found; retraining pretraining stage.")
            training_params["retrain_flag"] = True

    s_dataloaders, t_dataloaders = dataload.get_unlabeled_dataloaders(
        gex_features_df=gex_features_df,
        soure_ccc=soure_ccc,
        target_ccc=target_ccc,
        seed=2024,
        test_ratio=0.1,
        batch_size=training_params["unlabeled"]["batch_size"],
    )
    s_train = s_dataloaders[0]
    training_params.update({
        "cancer_genes": next(iter(s_train))[0].size(1),
        "ccc_num": next(iter(s_train))[1].size(1),
    })

    if ablation_id == 1:
        cancer_encoder = pretraining.training1(
            base_edge_index,
            s_dataloaders=s_dataloaders,
            t_dataloaders=t_dataloaders,
            **wrap_params(training_params, type="unlabeled"),
        )
        ccc_encoder = None
        ccc_gate_layer = None
    else:
        cancer_encoder, ccc_encoder, ccc_gate_layer = pretraining.training2(
            base_edge_index,
            s_dataloaders=s_dataloaders,
            t_dataloaders=t_dataloaders,
            **wrap_params(training_params, type="unlabeled"),
        )

    labeled_dataloader = dataload.get_multi_labeled_dataloader(
        gex_features_df=gex_features_df,
        soure_ccc=soure_ccc,
        target_ccc=target_ccc,
        seed=2024,
        batch_size=training_params["labeled"]["batch_size"],
        drug=drug,
        ccle_measurement=args.measurement,
        threshold_gdsc=args.thres_g,
        threshold_label=args.thres_s,
        n_splits=args.n,
    )

    training_fn = getattr(finetuning, training_fn_name)
    fold = 0
    all_results = []
    for train_labeled_ccle, test_labeled_ccle, labeled_tcga, train_ids, test_ids in labeled_dataloader:
        print(f"\n################ Dataset Fold: {fold} ################")
        ft_cancer_encoder = deepcopy(cancer_encoder)
        ft_ccc_encoder = deepcopy(ccc_encoder)
        ft_ccc_gate_layer = deepcopy(ccc_gate_layer)
        print("================Drugs:================\n", drug)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            smiles_strs = []
            moa_list = []
            for i in range(len(config.drug_smiles)):
                smiles_strs.append(config.drug_smiles[i])
                moa_list.append(config.MOA_list[i])

            drug_embedding_smiles = smiles_to_chemberta_embeddings(smiles_strs)
            drug_embedding_MOA = get_drug_MOA(moa_list)

            network = training_fn(
                drug_embedding_smiles=drug_embedding_smiles,
                drug_embedding_MOA=drug_embedding_MOA,
                cancer_encoder=ft_cancer_encoder,
                ccc_encoder=ft_ccc_encoder,
                ccc_gate_layer=ft_ccc_gate_layer,
                s_dataloader=train_labeled_ccle,
                train_ids=train_ids,
                test_ids=test_ids,
                val_dataloader=test_labeled_ccle,
                tcga_dataloader=labeled_tcga,
                **wrap_params(training_params, type="labeled"),
            )
            torch.save(network.state_dict(), os.path.join(training_params["model_save_folder"], f"fold{fold}_DRPmodel.pt"))

            print("\n================Transfer testing for TCGA data================")
            testing_fn = finetuning.testing2
            test_loss, results, y_true, y_pred = testing_fn(
                drug_embedding_smiles=drug_embedding_smiles,
                drug_embedding_MOA=drug_embedding_MOA,
                model=network,
                t_dataloader=labeled_tcga,
                device=training_params["device"],
            )
            print("testing loss : {:.4f}".format(test_loss))
            print("test-avg (TCGA) metrics:\n\t {}".format(np.around(results, 4)))

        results = results.set_index("Drug").T
        task_save_folder = os.path.join(ABLATION_DIR, f"{exp_name}-results", "fold-" + str(fold))
        make_dir(task_save_folder)
        file_name = os.path.join(task_save_folder, param_str)
        with open(f"{file_name}.csv", "w") as f:
            results.to_csv(f)
        fold += 1
        all_results.append(results)

    avg_result = np.mean(np.array(all_results), 0)
    avg_result = pd.DataFrame(avg_result)
    avg_result.columns = drug
    avg_result.index = ["Precision", "F1", "PRAUC", "Accuracy", "ROC_AUC"]
    file_name = os.path.join(ABLATION_DIR, f"{exp_name}-results", param_str)
    with open(f"{file_name}.csv", "w") as f:
        avg_result.to_csv(f)

    elapsed = time.time() - start_time
    print("9-drug elapsed time: ", round(elapsed, 4))
