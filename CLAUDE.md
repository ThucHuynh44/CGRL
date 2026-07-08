# DISC-MIL — Distributed Instance Selection for Multiple Instance Learning

> Phân loại Whole Slide Images (WSI) histopathology sử dụng Multiple Instance Learning (MIL) với concept-aware instance selection.

## Project Overview

DISC-MIL là một framework nghiên cứu thực hiện phân loại WSI (Whole Slide Images) bằng MIL. Điểm nổi bật là module **CGRL (Concept-Gated Representation Learning)** — chọn lọc các patch instances dựa trên độ tương đồng với concept prototypes trước khi đưa vào MIL aggregator, giúp giảm đáng kể FLOPs mà vẫn giữ độ chính xác.

## Quick Start

```bash
# Training với base MIL model
python main.py \
    --model_type trans_mil \
    --task tcga_rcc \
    --k 5 \
    --exp_code my_experiment \
    --data_root_dir /path/to/dataset \
    --split_dir splits/tcga_rcc_cross_site_mode_fraction

# Training với CGRL wrapper (prefix "cgrl_")
python main.py \
    --model_type cgrl_trans_mil \
    --task tcga_rcc \
    --k 5 \
    --exp_code my_experiment_cgrl \
    --num_selected 100 \
    --alpha 0.01 \
    --data_root_dir /path/to/dataset \
    --split_dir splits/tcga_rcc_cross_site_mode_fraction
```

## Supported Model Types

| `--model_type` | Base Architecture | CGRL-wrappable |
|---|---|---|
| `clam_sb` | CLAM Single-Branch (gated attention) | ✅ (`cgrl_clam_sb`) |
| `clam_mb` | CLAM Multi-Branch | ✅ (`cgrl_clam_mb`) |
| `trans_mil` | TransMIL (Nyström attention) | ✅ (`cgrl_trans_mil`) |
| `mamba_mil` | MambaMIL (SRMamba/Mamba/BiMamba) | ✅ (`cgrl_mamba_mil`) |
| `abmil` | Attention-based MIL | ✅ (`cgrl_abmil`) |
| `mean_mil` | Mean Pooling MIL | ✅ (`cgrl_mean_mil`) |
| `max_mil` | Max Pooling MIL | ✅ (`cgrl_max_mil`) |
| `dsmil` | Dual-Stream MIL | ✅ (`cgrl_dsmil`) |
| `cate_abmil` | CATE + ABMIL (concept distillation) | ❌ |
| `cate_dl` | CATE + Deep Learning (concept distillation) | ❌ |

## Supported Tasks

| `--task` | Dataset | Classes |
|---|---|---|
| `tcga_rcc` | TCGA RCC | 3 (CCRCC, PRCC, CHRCC) |
| `tcga_nsclc` | TCGA NSCLC | 2 (LUAD, LUSC) |
| `tcga_brca` | TCGA BRCA | 2 (IDC, ILC) |
| `brcas` | Internal BRCA | 3 (Group_BT, Group_AT, Group_MT) |
| `herohe` | HEROHE | 2 (Negative, Positive) |

## Project Structure

```
DISC-MIL/
├── main.py                     # Entry point — k-fold training orchestration
├── models/                     # Tất cả model definitions
│   ├── cgrl_mil.py            # CGRLMIL wrapper + ConceptSelector (MAIN)
│   ├── model_clam.py          # CLAM_SB, CLAM_MB
│   ├── TransMIL.py            # TransMIL + NystromAttention
│   ├── MambaMIL.py           # MambaMIL variants
│   ├── abmil.py               # Attention-based MIL
│   ├── dsmil.py               # Dual-Stream MIL
│   ├── cate.py                # CATE concept module
│   ├── catemil.py             # CATE + DAttention
│   ├── infonce.py             # InfoNCE contrastive loss
│   └── prompts.py             # Text prompts cho TITAN encoder
├── utils/                      # Training utilities
│   ├── core_utils.py          # Main train() + all train_loop functions (>900 lines)
│   ├── utils.py               # DataLoader helpers, metrics
│   ├── eval_utils.py          # Model loading + evaluation
│   └── constants.py           # IMAGENET constants
├── dataset_modules/            # Dataset loaders
│   └── dataset_generic.py      # Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
├── dataset_csv/               # CSV annotations (5 datasets)
├── splits/                    # Pre-computed train/val/test splits
├── results_selectK/           # Experiment results output
└── mamba/                     # Mamba SSM implementation (forked, external)
```

## Training Architecture

```
main.py::main(args)
  ├── wandb.init()
  ├── dataset = Generic_MIL_Dataset(csv_path, data_dir, label_dict)
  └── for fold i in [k_start, k_end):
        dataset.return_splits(csv_path='splits_{i}.csv')
        train(datasets, i, args) [utils/core_utils.py]
          ├── Model init:
          │   ├── Nếu model_type có prefix "cgrl_" → wrap base với CGRLMIL
          │   └── Else → init base model trực tiếp
          ├── optimizer = Adam/SGD
          ├── for epoch:
          │   ├── CLAM model  → train_loop_clam / validate_clam
          │   ├── CGRL model  → train_loop_cgrl / validate_cgrl
          │   └── Others      → train_loop / validate
          ├── measure_expected_flops()
          └── summary() hoặc summary_cgrl()
```

## Key Hyperparameters

| Parameter | Default | Mô tả |
|---|---|---|
| `--num_selected` | 100 | Số patches chọn sau CGRL selection |
| `--alpha` | 0.01 | Trọng số sim_loss (InfoNCE) |
| `--beta` | 0.1 | Trọng số cac_loss (CAC compression, old version) |
| `--mambamil_rate` | 10 | Sequence reduction rate (SRMamba) |
| `--mambamil_layer` | 2 | Số Mamba layers |
| `--embed_dim` | 1024 | Feature dimension |
| `--bag_weight` | 0.7 | CLAM: trọng số bag_loss vs instance_loss |
| `-B` (k_sample) | 8 | CLAM: # positive/negative patches |

## CGRL Concept Selection

CGRL hoạt động theo cơ chế:

1. **Input**: Bag gồm N patches, mỗi patch có feature vector [D]
2. **Projection**: `sim_projector` map features → concept space
3. **Similarity**: Cosine similarity với concept bank [num_classes, concept_dim]
4. **Selection**: Top-K patches theo max similarity per instance
5. **InfoNCE Loss**: `L_y + λ_sym L_sym + λ_cross L_cross` (chỉ khi có label)
6. **MIL Forward**: Selected K patches → base MIL model → classification

**Tại test time**: `label=None` → skip sim_loss → inference nhanh hơn.

## Dependencies chính

- `torch`, `torchvision`
- `einops` — tensor reshaping
- `wandb` — experiment tracking
- `thop` — FLOPs profiling
- `topk.svm` — SmoothTop1SVM cho CLAM instance loss
- `conch` — pathology encoder (pathology-specific)
- `transformers` — AutoModel/AutoTokenizer (TITAN encoder)
- `causal_conv1d`, `triton` — Mamba GPU kernels
- `scikit-learn`, `scipy`, `pandas`, `numpy`
