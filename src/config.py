import os
import numpy
# import torch
"""
configuration file includes all related datasets 
"""
CUDA_ID = os.environ.get('CUDA_ID', '3') #'cuda:0' if torch.cuda.is_available() else 'cpu'

ABLATION_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ABLATION_DIR, os.pardir))

root_data_folder = os.path.join(PROJECT_ROOT, 'data')
raw_data_folder = os.path.join(root_data_folder, 'raw_dat')
preprocessed_data_folder = os.path.join(root_data_folder, 'preprocessed')


#TCGA_datasets
tcga_multi_label_file = os.path.join(root_data_folder, 'TCGA_labels.csv')
tcga_folder = os.path.join(raw_data_folder, 'TCGA')
tcga_sample_file = os.path.join(tcga_folder, 'TCGA_phenotype_denseDataOnlyDownload.tsv.gz')

#PDX_datasets
PDX_folder = os.path.join(root_data_folder, 'PDX')
PDX_multi_label_file = os.path.join(PDX_folder, 'PDX_labels.csv')
PDX_gex_file = os.path.join(PDX_folder, 'gene_expression_filtered_1000genes.csv')


#CCLE datasets
ccle_folder = os.path.join(raw_data_folder, 'CCLE')
ccle_gex_file = os.path.join(ccle_folder, 'zscore.csv')
ccle_preprocessed_gex_file = os.path.join(preprocessed_data_folder, 'CCLE_expression.csv')
ccle_sample_file = os.path.join(ccle_folder, 'sample_info.csv')

#gex features
gex_feature_file = os.path.join(preprocessed_data_folder, '1000gene_features.csv')
soure_ccc_file = os.path.join(preprocessed_data_folder, 'CCLE_LRFeatures386.csv')
target_ccc_file = os.path.join(preprocessed_data_folder, 'TCGA_LRFeatures386.csv')

#GDSC datasets
gdsc_folder = os.path.join(raw_data_folder, 'GDSC')
gdsc_target_file1 = os.path.join(gdsc_folder, 'GDSC1_fitted_dose_response_25Feb20.csv')
gdsc_target_file2 = os.path.join(gdsc_folder, 'GDSC2_fitted_dose_response_25Feb20.csv')
gdsc_raw_target_file = os.path.join(gdsc_folder, 'gdsc_ic50flag.csv')
gdsc_sample_file = os.path.join(gdsc_folder, 'gdsc_cell_line_annotation.csv')
gdsc_preprocessed_target_file = os.path.join(preprocessed_data_folder, 'gdsc_ic50flag.csv')
gdsc_drugs = os.path.join(gdsc_folder, 'drug_smiles.csv')
drugs_MOAtext = os.path.join(root_data_folder, 'simplify_MOA.csv')
# drugs_MOAtext = os.path.join(root_data_folder, 'pubchem_chemo_mechanisms.csv')#'simplify_MOA.csv')

label_graph = numpy.array(object=object)
label_graph_norm = numpy.array(object=object)

tissue_map = numpy.array(object=object)
drug_feat = numpy.array(object=object)
drug_smiles = numpy.array(object=object)
MOA_list = numpy.array(object=object)
substru_list = numpy.array(object=object)
substru_mask = numpy.array(object=object)
best_roc_threshold = 0.5
