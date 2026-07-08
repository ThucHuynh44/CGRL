# Dataset Modules

Data loading cho WSI classification: CSV annotations, patient-level aggregation, MIL bag creation.

## File Map

| File | Classes / Functions | Lines |
|---|---|---|
| `dataset_generic.py` | `Generic_WSI_Classification_Dataset`, `Generic_MIL_Dataset`, `Generic_Split`, `save_splits` | 31–367 |
| `wsi_dataset.py` | `Whole_Slide_Bag`, `Whole_Slide_Bag_FP`, `Wsi_Region` | 10–114 |
| `dataset_h5.py` | Legacy HDF5 dataset | — |
| `__init__.py` | Module init | — |

---

## `dataset_generic.py` — Main Dataset Module ⭐

### `save_splits()` (dòng 16–28)

```python
def save_splits(split_datasets, column_keys, filename, boolean_style=False)
```

Exports train/val/test splits to CSV:
- `boolean_style=False`: columns = `['train', 'val', 'test']`, rows = slide_ids
- `boolean_style=True`: one-hot encoded matrix

### `Generic_WSI_Classification_Dataset` (dòng 31–79)

Base dataset class cho WSI classification:

```python
class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self, csv_path='dataset_csv/ccrcc_clean.csv', shuffle=False,
                 seed=7, print_info=True, label_dict={}, filter_dict={},
                 ignore=[], patient_strat=False, patient_view=False)
```

**Key attributes**:
- `slide_data`: pandas DataFrame loaded from CSV
- `label_dict`: maps string labels → integer indices
- `filter_dict`: filter slides by site/label
- `slide_ids`, `labels`, `patient_ids`: lists aligned with slide_data

**Key methods**:
- `label_encoder()`: convert string labels → int using `label_dict`
- `filter_dict()`: remove slides by criteria (e.g., `{'site': 'SiteA'}`)
- `patient_data_prep()`: aggregate slide-level → patient-level labels via max voting
- `return_splits()`: returns `Generic_Split` objects from pre-generated CSV splits

**CSV format**:
```
slide_id, patient_id, label, site, ...
TCGA_ABC_001, TCGA_ABC, 0, site1, ...
```
- `slide_id`: unique WSI identifier (matches `.pt` filename)
- `patient_id`: patient identifier (for patient-level splits)
- `label`: integer (mapped via `label_dict`)
- `site`: institution/source site

### `Generic_MIL_Dataset` (dòng 315–353)

```python
class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    def __getitem__(self, index)
```

Extends base class cho MIL paradigm:
- Mỗi sample = 1 bag (1 slide = N patches)
- Load patch features từ `.pt` files: `{data_dir}/pt_files/{slide_id}.pt`
- **Feature format**: tensor `[num_patches, feature_dim]` (e.g., `[~3000, 1024]`)
- Feature dim phụ thuộc encoder (Conch = 768, default = 1024)

### `Generic_Split` (dòng 355–367)

Lightweight container cho một split:
```python
class Generic_Split():
    def __init__(self, slide_data, data_dir, num_classes, label_dict)
```

### `cls_ids_prep()` (dòng 81–86)
```python
def cls_ids_prep(patient_cls)
```
Precomputes per-class patient/slide ID lists for class-balanced sampling.

### `patient_data_prep()` (dòng 92–108)
Aggregates slide labels to patient level using **max voting**:
```python
def patient_data_prep(patient_labels, patient_ids, labels)
```

---

## `wsi_dataset.py` — Low-level WSI Bag Classes

### `Whole_Slide_Bag` (dòng 10–40)
```python
class Whole_Slide_Bag(Dataset):
    def __init__(self, file_path, pretrained_weights=False, target_size=224,
                 custom_transforms=None, flip='NONE', rotate='NONE')
```
Loads raw image patches từ `.h5` or `.pt` file. Áp dụng augmentation transforms.

### `Whole_Slide_Bag_FP` (dòng 42–80)
Feature pre-extraction mode: load patches + compute features using pretrained encoder.

### `Wsi_Region` (dòng 82–114)
Single WSI region container.

---

## Supported Dataset CSVs

| CSV File | Task | Classes |
|---|---|---|
| `dataset_csv/brcas.csv` | Internal BRCA subtypes | Group_BT, Group_AT, Group_MT (3) |
| `dataset_csv/herohe.csv` | HEROHE breast cancer | Negative, Positive (2) |
| `dataset_csv/tcga_brca_subset.csv` | TCGA BRCA subtypes | IDC, ILC (2) |
| `dataset_csv/tcga_nsclc_subset.csv` | TCGA NSCLC subtypes | LUAD, LUSC (2) |
| `dataset_csv/tcga_rcc_subset.csv` | TCGA RCC subtypes | CCRCC, PRCC, CHRCC (3) |
| `dataset_csv/tcga_cesc.csv` | TCGA CESC | Multiple subtypes (2–3) |

---

## Splits Directory

Pre-computed CV splits stored in `splits/`:
```
splits/tcga_rcc_cross_site_mode_fraction/
  splits_0.csv, splits_1.csv, splits_2.csv, splits_3.csv, splits_4.csv
```

**Naming convention**: `{task}_{cross_site_val_mode}_{fraction}.csv`

**Split usage**: Passed via `--split_dir` argument:
```bash
--split_dir splits/tcga_rcc_cross_site_mode_fraction
```

---

## Data Loading Flow

```
Generic_MIL_Dataset(csv_path, data_dir, label_dict)
  └── Generic_WSI_Classification_Dataset(csv_path, ...)
        └── slide_data loaded from CSV
  └── __getitem__(index):
        slide_id = self.slide_ids[index]
        features = torch.load(f"{data_dir}/pt_files/{slide_id}.pt")
        label = self.labels[index]
        return (features, label)
  └── return_splits():
        reads {split_dir}/splits_{fold}.csv
        returns (train_split, val_split, test_split) as Generic_Split
```

---

## Label Dictionaries

Standard label_dict mappings (used in `main.py`):

```python
tcga_rcc:   {'CCRCC': 0, 'PRCC': 1, 'CHRCC': 2}
tcga_nsclc: {'LUAD': 0, 'LUSC': 1}
tcga_brca:  {'IDC': 0, 'ILC': 1}
brcas:      {'Group_BT': 0, 'Group_AT': 1, 'Group_MT': 2}
herohe:     {'Negative': 0, 'Positive': 1}
```

---

## Key Notes

- **Feature files**: `.pt` files trong `pt_files/` directory, filename = `slide_id.pt`
- **Patient-level splits**: `patient_strat=True` ensures no patient overlap between train/val/test
- **Cross-site validation**: `--subtyping` flag enables cross-site evaluation mode
- **Balance**: `weighted_sample=True` enables class-weighted sampling during training
