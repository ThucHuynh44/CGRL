## Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd DISC-MIL
```

Create a Python environment (recommended):

```bash
conda create -n discmil python=3.9
conda activate discmil
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** The project was developed and tested with **Python 3.9**, **PyTorch 1.12.0**, and **CUDA 11.x**.

---

## Checkpoints

Pretrained checkpoints can be downloaded from:

[https://drive.google.com/drive/folders/1NaHEmRs1bDeiUEKapw4pBpepU0-pQ4Nw?usp=sharing](https://drive.google.com/drive/folders/12CWcEZ2Tt4VX3kPwunzJQt4k32SMPp1N?usp=sharing)

---

## Training

Replace the placeholders below before running:

- `<GPU_ID>`: GPU index.
- `<TASK_NAME>`: Classification task (e.g., `tcga_brca`, `tcga_lung`).
- `<DATA_ROOT>`: Root directory of extracted features.
- `<SPLIT_DIR>`: Dataset split directory.
- `<CONCEPT_PATH>`: Path to the concept embedding file.
- `<EXP_NAME>`: Experiment name.

### Train CGRL-ABMIL

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py \
    ...
```

### Train CGRL-CLAM-SB

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py \
    ...
```

### Train CGRL-TransMIL

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py \
    ...
```

### Train CGRL-DSMIL

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py \
    ...
```

---

## Command Arguments

| Argument | Description |
|----------|-------------|
| `--task` | Classification task (e.g., `tcga_brca`, `tcga_lung`). |
| `--model_type` | MIL backbone (`cgrl_abmil`, `cgrl_clam_sb`, `cgrl_trans_mil`, or `cgrl_dsmil`). |
| `--data_root_dir` | Root directory containing extracted slide features. |
| `--split_dir` | Directory containing the dataset split files. |
| `--concept_path` | Path to the precomputed concept embeddings. |
| `--exp_code` | Experiment name used for saving logs and checkpoints. |
| `--embed_dim` | Feature embedding dimension (default: `768`). |
| `--num_selected` | Number of selected concepts. |

Training logs, checkpoints, and evaluation results are automatically saved under the directory corresponding to `--exp_code`.
