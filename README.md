# CGRL

**CGRL: Concept-Guided Pruning and Representation Learning for Whole-Slide Image Classification**.



---

## Overview

CGRL is a concept-guided pruning and representation learning framework for computational pathology. The method introduces disease-related concept information into multiple instance learning models to prune less informative concepts and improve slide-level representation learning for whole-slide image classification.

In this repository, CGRL is implemented with several MIL backbones:

- `cgrl_abmil`
- `cgrl_clam_sb`
- `cgrl_trans_mil`
- `cgrl_dsmil`



---

## Repository

```bash
git clone https://github.com/ThucHuynh44/CGRL.git
cd CGRL
```
---
## Installation

Create a Python environment:

```bash
conda create -n cgrl python=3.9
conda activate cgrl
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The project was developed and tested with:

```text
Python 3.9
PyTorch 1.12.0
CUDA 11.x
```

---

## Data Preparation

This repository uses pre-extracted WSI features as input. Raw whole-slide images are not included.

A typical feature directory is organized as:

```bash
DATA_ROOT/
├── slide_001.pt
├── slide_002.pt
├── slide_003.pt
└── ...
```

The split directory should contain the dataset split files:

```bash
splits/
└── task_2_tumor_subtyping_BRCA/
    ├── splits_0.csv
    ├── splits_1.csv
    ├── ...
    └── splits_4.csv
```

The concept embeddings should be precomputed and stored as `.pt` files, for example:

```bash
models/
└── prompt_feats_titan/
    └── brca_concepts_titan.pt
```

---

## Checkpoints

Pretrained checkpoints can be downloaded from:

```text
https://drive.google.com/drive/folders/12CWcEZ2Tt4VX3kPwunzJQt4k32SMPp1N?usp=sharing
```

After downloading, place the checkpoints in an appropriate directory, for example:

```bash
checkpoints/
└── CGRL/
```

---

## Training

Before running training, replace the following placeholders:

| Placeholder | Description |
|------------|-------------|
| `<GPU_ID>` | GPU index used for training. |
| `<DATA_ROOT>` | Root directory containing extracted WSI features. |
| `<SPLIT_DIR>` | Directory containing dataset split files. |
| `<CONCEPT_PATH>` | Path to the precomputed concept embedding file. |
| `<NUM_SELECTED>` | Number of selected concepts. |

Example environment variables:

```bash
export GPU_ID=0
export DATA_ROOT=/path/to/BRCA
export SPLIT_DIR=/path/to/CGRL/splits/task_2_tumor_subtyping_BRCA
export CONCEPT_PATH=/path/to/CGRL/models/prompt_feats_titan/brca_concepts_titan.pt
export NUM_SELECTED=300
```

---

## Train CGRL-ABMIL

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    --drop_out 0.25 \
    --early_stopping \
    --lr 2e-4 \
    --k 5 \
    --exp_code BRCA_cgrl_abmil_compression_k300 \
    --weighted_sample \
    --bag_loss ce \
    --inst_loss svm \
    --task tcga_brca \
    --model_type cgrl_abmil \
    --log_data \
    --subtyping \
    --data_root_dir ${DATA_ROOT} \
    --embed_dim 768 \
    --split_dir ${SPLIT_DIR} \
    --concept_path ${CONCEPT_PATH} \
    --num_selected ${NUM_SELECTED}
```

---

## Train CGRL-CLAM-SB

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    --drop_out 0.25 \
    --early_stopping \
    --lr 2e-4 \
    --k 5 \
    --exp_code BRCA_cgrl_clam_sb_k300 \
    --weighted_sample \
    --bag_loss ce \
    --task tcga_brca \
    --model_type cgrl_clam_sb \
    --log_data \
    --subtyping \
    --data_root_dir ${DATA_ROOT} \
    --embed_dim 768 \
    --split_dir ${SPLIT_DIR} \
    --concept_path ${CONCEPT_PATH} \
    --num_selected ${NUM_SELECTED}
```

---

## Train CGRL-TransMIL

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    --drop_out 0.25 \
    --early_stopping \
    --lr 2e-4 \
    --k 5 \
    --exp_code BRCA_cgrl_transmil_k300 \
    --weighted_sample \
    --bag_loss ce \
    --task tcga_brca \
    --model_type cgrl_trans_mil \
    --log_data \
    --subtyping \
    --data_root_dir ${DATA_ROOT} \
    --embed_dim 768 \
    --split_dir ${SPLIT_DIR} \
    --concept_path ${CONCEPT_PATH} \
    --num_selected ${NUM_SELECTED}
```

---

## Train CGRL-DSMIL

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    --drop_out 0.25 \
    --early_stopping \
    --lr 2e-4 \
    --k 5 \
    --exp_code BRCA_cgrl_dsmil_k300 \
    --weighted_sample \
    --bag_loss ce \
    --task tcga_brca \
    --model_type cgrl_dsmil \
    --log_data \
    --subtyping \
    --data_root_dir ${DATA_ROOT} \
    --embed_dim 768 \
    --split_dir ${SPLIT_DIR} \
    --concept_path ${CONCEPT_PATH} \
    --num_selected ${NUM_SELECTED}
```

---

## Example: BRCA Tumor Subtyping

The following example trains CGRL with the TransMIL backbone on the BRCA tumor subtyping task:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --drop_out 0.25 \
    --early_stopping \
    --lr 2e-4 \
    --k 5 \
    --exp_code BRCA_cgrl_transmil_k300 \
    --weighted_sample \
    --bag_loss ce \
    --task tcga_brca \
    --model_type cgrl_trans_mil \
    --log_data \
    --subtyping \
    --data_root_dir /path/to/BRCA \
    --embed_dim 768 \
    --split_dir /path/to/CGRL/splits/task_2_tumor_subtyping_BRCA \
    --concept_path /path/to/CGRL/models/prompt_feats_titan/brca_concepts_titan.pt \
    --num_selected 300
```

---

## Command Arguments

| Argument | Description |
|----------|-------------|
| `--task` | Classification task, for example `tcga_brca`. |
| `--model_type` | MIL backbone. Supported values: `cgrl_abmil`, `cgrl_clam_sb`, `cgrl_trans_mil`, and `cgrl_dsmil`. |
| `--data_root_dir` | Root directory containing extracted slide features. |
| `--split_dir` | Directory containing dataset split files. |
| `--concept_path` | Path to the precomputed concept embeddings. |
| `--exp_code` | Experiment name used for saving logs, checkpoints, and results. |
| `--embed_dim` | Feature embedding dimension. Default: `768`. |
| `--num_selected` | Number of selected disease-related concepts. |
| `--drop_out` | Dropout rate. |
| `--early_stopping` | Enable early stopping. |
| `--lr` | Learning rate. |
| `--k` | Number of folds or split setting used by the training pipeline. |
| `--weighted_sample` | Enable weighted sampling for class imbalance. |
| `--bag_loss` | Bag-level loss function. Example: `ce`. |
| `--log_data` | Enable experiment logging. |
| `--subtyping` | Enable tumor subtyping setting. |

---





## Citation

```bash
@inproceedings{huynh2026cgrl,
  title={CGRL: Concept-Guided Pruning and Representation Learning for Whole-Slide Image Classification},
  author={Huynh, Thuc and Le, Tuan and Bui, Doanh C},
  booktitle={2026 International Conference on Multimedia Analysis and Pattern Recognition (MAPR)},
  pages={286--291},
  year={2026},
  organization={IEEE}
}
```
---

## Contact

For questions or issues, please open an issue in this repository.

---

## Acknowledgements

This research was supported by The VNUHCM-University of Information Technology's Scientific Research Support Fund
