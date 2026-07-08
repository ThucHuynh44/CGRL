# Utils Module

Utilities cho training, evaluation, data loading và metrics.

## File Map

| File | Contents | Lines |
|---|---|---|
| `core_utils.py` | Main `train()` + all train/validate loops + FLOPs profiling | 1–912 |
| `utils.py` | DataLoader helpers, split generation, metrics | 1–167 |
| `eval_utils.py` | Model loading + evaluation utilities | 1–119 |
| `file_utils.py` | pickle/hdf5 I/O | 1–34 |
| `transform_utils.py` | Image transforms | 1–12 |
| `constants.py` | IMAGENET/OPENAI mean-std + MODEL2CONSTANTS | 1–26 |
| `utils_contrastive.py` | Placeholder (empty) | 1 |
| `core_utils_old.py` | Old version (CGRL code commented out) | — |

---

## `core_utils.py` — Main Training Engine (QUAN TRỌNG)

> Hơn 900 lines. Đây là file xử lý chính cho training loop.

### Classes

#### `Accuracy_Logger` (dòng 17–50)
Track per-class accuracy:
- `log(Y_hat, Y)`: single sample
- `log_batch(Y_hat, Y)`: batch
- `get_summary(c)`: returns `(accuracy, correct, count)` for class `c`

#### `EarlyStopping` (dòng 52–93)
```python
class EarlyStopping:
    def __init__(self, patience=20, stop_epoch=50, verbose=False)
    def __call__(val_loss) → bool  # True = stop training
```
Patience-based stopping: stops if validation loss doesn't improve after `patience` epochs.

### `measure_expected_flops()` (dòng 116–192)
Đo FLOPs của baseline vs CGRL model:
```python
def measure_expected_flops(model, loader, args, True)
```
- Computes average N (input patches) từ dataset
- Uses `thop.profile` on dummy tensors `[1, N_avg, D]` vs `[1, K_avg, D]`
- Logs to wandb: baseline_flops, cgrl_flops, flops_reduction (%)

---

### Main Training Flow

#### `train(datasets, cur, args)` (dòng 195–411)
Entry point được gọi từ `main.py`:
```python
def train(datasets, cur, args):
    # cur = current fold index
    # datasets = (train_dataset, val_dataset, test_dataset)
```

**Logic model routing**:
```python
if args.model_type.startswith('cgrl_'):
    # Wraps base model với CGRLMIL
    base_model_type = args.model_type.replace('cgrl_', '')
    base_model = create_base_mil(base_model_type, args)
    model = CGRLMIL(base_model, ...)
elif args.model_type.startswith('clam'):
    model = CLAM_SB or CLAM_MB
else:
    model = create_base_mil(args.model_type, args)
```

Optimizer/SCHEDULER routing:
- `bag_loss='svm'` → `nn.CrossEntropyLoss()`
- `bag_loss='ce'` → also `nn.CrossEntropyLoss()`
- `inst_loss='svm'` → `SmoothTop1SVM()`
- lr_scheduler: `optim.lr_scheduler.ReduceLROnPlateau`

---

### Training Loops

#### `train_loop_cgrl()` (dòng 409–476) — CGRL Models
```python
total_loss = cls_loss + args.alpha * sim_loss + args.beta * cac_loss
```
Logs per epoch: `cls_loss`, `sim_loss`, `total_loss`, AUC, accuracy, selected_K.

#### `validate_cgrl()` (dòng 479–556)
- `label=None` → skip sim_loss (test-time behavior)
- Uses `cls_loss` cho EarlyStopping

#### `summary_cgrl()` (dòng 559–604)
Test-time evaluation:
- `label=None` → fast inference
- Computes: test_auc, test_acc, test_error
- Logs per-sample results (slide_id, Y_hat, Y, probability)

#### `train_loop_clam()` (dòng 607–748)
CLAM-specific: `total_loss = bag_weight * bag_loss + (1 - bag_weight) * instance_loss`

#### `train_loop()` (dòng 674–793) — Standard MIL
Default training loop:
- `loss = criterion(logits, labels)`
- Backpropagation

#### `validate()` (dòng 719–851)
Default validation loop.

#### `summary()` (dòng 853–941)
Default test-time evaluation.

---

## `utils.py` — Data & Metrics Helpers

### `get_split_loader()` (dòng ~50)
```python
def get_split_loader(split_dataset, training=False, weighted=False, args=None)
```
Creates `DataLoader` từ split dataset. Hỗ trợ weighted sampling.

### `generate_split()` (dòng ~80)
```python
def generate_split(dataset, split_dir, val_frac=0.1, test_frac=0.1, seed=7,
                   bin_exist=False, num_splits=None)
```
Creates train/val/test splits.

### `accuracy()` (dòng ~120)
```python
def accuracy(output, labels)
```
Computes batch accuracy.

---

## `eval_utils.py` — Evaluation Utilities

### `summary_model()` (dòng ~30)
Load checkpoint + evaluate model on test set.

---

## `constants.py` — Normalization Constants

```python
IMAGENET = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
OPENAI = {'mean': [0.48145466, 0.4578275, 0.40821073],
          'std': [0.26862954, 0.26130258, 0.27577711]}
MODEL2CONSTANTS = {'conch': OPENAI, 'default': IMAGENET}
```

---

## `file_utils.py` — I/O

```python
save_pkl(path, obj)
load_pkl(path)
```
Dùng `pickle` và `h5py` để lưu/load results và features.

---

## Key Training Patterns

### WandB Logging
Tất cả train loops log metrics via `wandb.log()`:
```python
wandb.log({'train_loss': loss, 'train_auc': auc, 'epoch': epoch})
```

### Checkpoint Loading
- Best model được load sau khi training kết thúc
- `results_dir` lưu `split_{i}_results.pkl` cho mỗi fold

### Model Creation Pattern
```python
# 1. Base model factory
def create_base_mil(model_type, args):
    if model_type == 'trans_mil': return TransMIL(args)
    if model_type == 'mamba_mil': return MambaMIL(args)
    ...

# 2. CGRL wrap
if args.model_type.startswith('cgrl_'):
    base = create_base_mil(...)
    model = CGRLMIL(base, ...)
```
