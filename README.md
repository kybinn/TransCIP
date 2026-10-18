# TransCIP

**Communication-Associated Molecular Interaction Learning for Cross-Domain Drug Response Prediction**

TransCIP is a transfer learning framework for predicting clinical drug response by transferring knowledge from labeled cancer cell lines to unlabeled patient tumors. It integrates communication-associated molecular interaction priors with molecular representation learning and prototype-guided cross-domain adaptation.

This repository provides the implementation of TransCIP, including the **pre-training code and pre-trained models**, **adaptation code and adapted models**, and a Conda environment configuration.

## Overview

> **Interpretation of the interaction priors:** Communication-associated gene pairs are prior-informed molecular relationships used to construct features from gene expression. They are **not** direct measurements of cell-to-cell signaling activity or therapeutic drug–target binding.

## Repository contents

The released repository includes:

- The main entry point, `full.py`.
- Code and model weights for the pre-training stage.
- Code and model weights for the adaptation stage.
- A Conda environment YAML file.
- Data-processing resources and preprocessed inputs.

Some large resources must be downloaded separately (see [Required external files](#required-external-files)).

The paths below are those relevant to the basic setup; they are **not** intended as an exhaustive listing of every file in the repository:

```text
TransCIP/
├── full.py
├── ...                         # environment YAML and released code/checkpoints
├── src/
│   └── utils/
│       └── BiomedBERT/         # download separately
└── data/
    └── preprocessed/
        └── 1000gene_features.csv  # download separately
```

## Installation

Clone the repository:

```bash
git clone https://github.com/kybinn/TransCIP.git
cd TransCIP
```

Create the Conda environment using the YAML file **included in this repository**:

```bash
conda env create -f <environment-file>.yaml
conda activate <environment-name>
```

Replace the two placeholders above with the actual YAML filename and the environment `name` defined inside that file. 

## Required external files

Two resources are too large to be hosted directly in this repository. Download them before running the pipeline.

### 1. BiomedBERT

Download the model files from:

**[microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext (Hugging Face)](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/tree/main)**

Place the local model directory at:

```text
src/utils/BiomedBERT/
```

Retain the original Hugging Face model configuration, tokenizer files, vocabulary, and PyTorch model weights so that the local directory can be loaded using the Transformers API.

### 2. Preprocessed gene features

Download `1000gene_features.csv` from:

**[1000gene_features.csv (Google Drive)](https://drive.google.com/file/d/1H_QZlwEzYrxkl5Kjqo3Vke3Av5Hotdty/view?usp=sharing)**

Save it as:

```text
data/preprocessed/1000gene_features.csv
```

Please keep the original filename. Confirm that both resources are in place before starting the program.

## Running TransCIP

From the repository root, run:

```bash
python full.py
```

`full.py` is the provided entry point for the released workflow. The repository includes the pre-training and adaptation implementations and corresponding model checkpoints.

### Reusing released models

For reproduction with the provided models, verify the checkpoint and data paths configured in `full.py` and its referenced modules. Use the supplied pre-trained and adapted checkpoints rather than retraining unnecessarily.

### Training from scratch

The full pre-training implementation is also provided. To re-run pre-training, **edit the relevant settings in the Python code** (such as whether to load an existing pre-trained checkpoint, training settings, and model-save paths), then run:

```bash
python full.py
```

After pre-training, ensure that the adaptation stage points to the newly generated checkpoint before proceeding. The exact parameter names and checkpoint locations are defined in the released code; this README does not assume unverified command-line switches.

