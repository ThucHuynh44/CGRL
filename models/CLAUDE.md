# Models Module

Tất cả định nghĩa neural network models cho DISC-MIL.

**Input format chuẩn**: Tất cả models nhận input `[B, N, D]` — batch size × num_patches × feature_dim.

## File Map

| File | Classes / Functions | Dòng |
|---|---|---|
| `cgrl_mil.py` | `ConceptSelector`, `CGRLMIL` | 27–309 |
| `model_clam.py` | `Attn_Net`, `Attn_Net_Gated`, `CLAM_SB`, `CLAM_MB` | 15–253 |
| `TransMIL.py` | `NystromAttention`, `Nystromformer`, `TransLayer`, `PPEG`, `TransMIL` | 33–327 |
| `MambaMIL.py` | `MambaMIL` | 23–130 |
| `abmil.py` | `DAttention` | 82–142 |
| `abmil_cluster.py` | `DAttention` (variant với concept + InfoNCE) | 27–194 |
| `dsmil.py` | `FCLayer`, `IClassifier`, `BClassifier`, `MILNet` | 1–75 |
| `Mean_Max_MIL.py` | `MeanMIL`, `MaxMIL` | 22–117 |
| `cate.py` | `CATE` | 8–69 |
| `catemil.py` | `DAttention` (CATE + DAttention integration) | 84–147 |
| `infonce.py` | `InfoNCE`, `info_nce()` | 5–115 |
| `prompts.py` | `brca_prompts`, `nsclc_prompts`, `rcc_prompts`, `cesc_prompts` | 1–300 |
| `generate_concept_anchor.py` | Script generate concept embeddings từ TITAN | — |

---

## `cgrl_mil.py` — CGRLMIL & ConceptSelector ⭐ CORE MODEL

Đây là model chính của project. CGRLMIL wrap một base MIL model với một ConceptSelector.

### `ConceptSelector` (dòng 27–144)

```python
class ConceptSelector(nn.Module):
    def __init__(self, input_dim=512, concept_dim_hidden=512, n_classes=2,
                 concept_path=None, device='cuda', num_selected=100)
```

**Pipeline trong `forward(x, label)`**:
1. `sim_projector` (MLP): map `[B, N, D]` → `[B, N, concept_dim_hidden]`
2. Cosine similarity: `S = X_norm @ T_norm.T` → similarity matrix `[B, N, n_classes]`
3. Top-K selection: mỗi instance giữ lại top-K patches theo max similarity
4. `_calculate_sim_loss()`: InfoNCE contrastive loss (chỉ khi `label is not None`)

**Key attributes**:
- `sim_projector`: MLP `[input_dim → concept_dim_hidden → concept_dim_hidden]`
- `concept_bank`: tensor `[n_classes, n_sub_concepts, concept_dim_hidden]`
- `num_selected`: số patches giữ lại (default 100)

**Important**: Tại test time, truyền `label=None` để skip sim_loss → inference nhanh.

### `_calculate_sim_loss()` (dòng 146–211)

InfoNCE loss gồm 3 terms:
- `L_y`: positive class term
- `L_sym`: symmetric other-classes term
- `L_cross`: cross-class mixing term

### `CGRLMIL` (dòng 217–247)

```python
class CGRLMIL(nn.Module):
    def __init__(self, mil_model, input_dim=512, concept_dim_hidden=512,
                 n_classes=2, concept_path=None, device='cuda',
                 num_selected=100, lambda_sym=0.1, lambda_cross=0.1)
```

Wrapper: khởi tạo `self.selector = ConceptSelector(...)` và `self.mil = mil_model`.

**`forward(x, label)`**:
1. `x_selected, results_dict = self.selector(x, label)`
2. `logits, Y_prob, Y_hat, _, feature = self.mil(x_selected)`
3. Returns: `(logits, Y_prob, Y_hat, results_dict, feature)`

---

## `model_clam.py` — CLAM (Contrastive Learning for Attention-based MIL)

### `Attn_Net` (dòng 15–31)
Simple attention: `Linear(L, D) → Tanh → [Linear(D, n_classes)]`. Output: `[B, N, n_classes]`.

### `Attn_Net_Gated` (dòng 41–64)
Gated attention: 3 FC layers với sigmoid gate. Cải thiện expressiveness so với `Attn_Net`.

### `CLAM_SB` (dòng 77–182) — Single Branch

```python
class CLAM_SB(nn.Module):
    def __init__(self, gate=True, size_arg='big', dropout=True, k_sample=8,
                 n_classes=2, instance_loss_fn=None, instance_weight=0.5)
```

**Forward modes** (controlled by `return_features`, `attention_only`, `instance_eval`):
- `instance_eval=True`: chọn k_sample positive + k_sample negative patches
- `return_features=True`: trả về deep features thay vì logits
- `attention_only=True`: chỉ trả về attention weights

**Instance evaluation** (dòng 107–136):
- Chọn top-B và bottom-B patches theo attention scores
- Dùng `SmoothTop1SVM` để tạo pseudo instance labels
- Tính instance-level loss

### `CLAM_MB` (dòng 183–253) — Multi Branch

Multi-branch version: separate attention network và classifier cho **mỗi class**. Tốt cho sub-typing tasks.

---

## `TransMIL.py` — TransMIL với Nyström Attention

### `NystromAttention` (dòng 33–151)
Landmark-based attention approximation dùng Nyström method:
- Chọn `n_landmarks` landmarks (default = seq_len // 4)
- Moore-Penrose pseudo-inverse với 6 iterations
- Complexity: O(n) thay vì O(n²)

### `PPEG` (dòng 241–255) — Positional Pog Pooling Equipped Group
Multi-kernel depthwise convolution cho positional encoding.

### `TransMIL` (dòng 258–327)
Full architecture:
1. `_fc1`: project `[B, N, D] → [B, N, 512]`
2. 2 × `TransLayer` (Nyström + FFN)
3. `PPEG` positional encoding
4. Learned CLS token → attention pooling
5. `self.classifier`: linear `[512 → n_classes]`

---

## `MambaMIL.py` — Mamba-based MIL

```python
class MambaMIL(nn.Module):
    def __init__(self, n_classes=2, mambamil_layer=2, mambamil_rate=10,
                 mambamil_type='SRMamba', dropout=0.25)
```

**Supports 3 variants** (via `mambamil_type`):
- `'SRMamba'`: Sequence Reduction Mamba — giảm N patches theo rate
- `'Mamba'`: Unidirectional SSM
- `'BiMamba'`: Bidirectional SSM

**Forward flow**:
```
[B, N, D] → _fc1 → [B, N, 512] → Mamba layers → attention pooling → classifier
```

---

## `abmil.py` — Attention-based MIL

### `DAttention` (dòng 82–142)
Simple attention-based MIL aggregator:
1. FC projection: `[B, N, D] → [B, N, 512]`
2. Gated attention mechanism
3. Attention-weighted sum pooling
4. Linear classifier

---

## `abmil_cluster.py` — ABMIL với Concept & InfoNCE

`DAttention` (dòng 27–194): mở rộng `abmil.py` với:
- Concept projector (tương tự CGRL)
- InfoNCE loss integration
- KL divergence (VAE-style reparameterization)

---

## `dsmil.py` — Dual-Stream MIL

### `IClassifier` (dòng 14–26)
Per-instance feature extractor + classifier. Outputs per-instance predictions.

### `BClassifier` (dòng 28–56)
Bag-level classifier:
1. Sort instances by criticality scores
2. 1D convolution over sorted sequence
3. Output: `[1, n_classes]` bag prediction

### `MILNet` (dòng 58–72)
Combines both:
```python
logits = 0.5 * bag_pred + 0.5 * max_instance_pred
```

---

## `Mean_Max_MIL.py` — Baseline Aggregators

### `MeanMIL` (dòng 22–63)
```python
features.mean(dim=1) → classifier
```

### `MaxMIL` (dòng 71–112)
```python
features.max(dim=1)[0] → classifier
```

---

## `cate.py` — Concept-Aware Token Embedding

```python
class CATE(nn.Module):
    def __init__(self, concept_path=None, n_classes=2, embed_dim=1024,
                 device='cuda', num_select=10, concept_dim=768)
```

**Key components**:
- Load concept features từ `.pt` file (generated by TITAN model)
- `encoder_IB`: VAE reparameterization (z_mean, z_logvar, z)
- InfoNCE contrastive loss: chọn top-10 similar instances per class
- `intervention_layer`: transform concept-matched features

---

## `catemil.py` — CATE + DAttention

`DAttention` (dòng 84–147): kết hợp basic DAttention với embedded CATE module. Concatenates:
- Reparameterized features (z)
- Concept-intervened features

---

## `infonce.py` — InfoNCE Contrastive Loss

### `info_nce()` (dòng 60–115)

```python
def info_nce(query, positive_key, negative_keys, temperature=0.07,
             collect_negative_from='all', unpaired=False)
```

**Parameters**:
- `query`: anchor representations
- `positive_key`: positive samples
- `negative_keys`: negative samples
- `collect_negative_from='all'`: negative keys pooled together
- `collect_negative_from='paired'`: per-sample negatives
- `unpaired=False`: whether negatives are paired with queries

**Output**: Scalar InfoNCE loss (temperature-scaled cosine similarity).

---

## `prompts.py` — Text Prompt Templates

| Function | Task | Classes |
|---|---|---|
| `brca_prompts()` | BRCA | IDC, ILC + 3 tissue types |
| `nsclc_prompts()` | NSCLC | LUAD, LUSC + 3 tissue types |
| `rcc_prompts()` | RCC | CCRCC, PRCC, CHRCC + 3 tissue types |
| `cesc_prompts()` | CESC | Adenocarcinoma, Squamous + 4 tissue types |

Mỗi prompt function trả về list of class templates (labels + tissue context descriptions). Dùng bởi `generate_concept_anchor.py` để tạo concept embeddings với TITAN model.

---

## `generate_concept_anchor.py` — Concept Embedding Generator

Script standalone:
1. Load TITAN model + tokenizer
2. Encode text prompts từ `prompts.py`
3. Save `.pt` file: list of tensors `[n_templates_per_class, 768]`

Output dùng cho `ConceptSelector` (`concept_path`) và `CATE` (`concept_path`).
