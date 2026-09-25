# KPlaceNet — Complete Implementation Walkthrough

## 1. The Problem

Given a single photograph (no GPS, no EXIF), predict its geographic coordinates (latitude, longitude). This is a classification problem where we discretize the Earth into cells, not a regression problem predicting raw lat/lon.

**Why classification over regression:** Regression on lat/lon is hard because the loss landscape is non-Euclidean (longitude wraps at ±180°, latitude is bounded ±90°). Classification sidesteps this by learning to predict which "bin" a photo belongs to, then mapping each bin's centroid as the prediction. The haversine distance between the predicted centroid and ground truth becomes the evaluation metric, which is the real-world-relevant measure.

---

## 2. Cell Construction — The Geographic Discretization

### 2.1 Quad-Tree (L1 Baseline)

The Earth is partitioned into approximately 300 rectangular cells using a density-driven recursive quad-tree:

```text
Start: one cell covering the entire world [-90,90] × [-180,180]
While len(cells) < K:
    Find the leaf cell with the most training points
    Split it at its midpoint (lat_mid, lon_mid) into up to 4 quadrants
    Keep only non-empty quadrants (no empty classes)
    Stop when K cells reached or nothing is splittable
```

**Why K ≈ 300:** This is the 4050 GPU constraint. The head is `Linear(2048, K)`. With K=300, the head has 614K params. With K=3000 it would be 6M, and the class imbalance would be extreme (10k images / 3000 classes = ~3 per class). K=300 gives ~33 images per class on average, which is enough to learn something.

**Why density-driven:** Dense urban areas (London, Tokyo, New York) get small cells; sparse rural areas (Sahara, Siberia) get large cells. This matches the data distribution — you can't learn a cell with 1 training image.

**Determinism:** The split order is sorted by `(-count, index)`, so ties are broken by position. The same training CSV always produces the same cells.

**Empty class avoidance:** If a split yields fewer than 2 non-empty children, it's rejected. This prevents the degenerate case where duplicate coordinates (common in OSV-5M) create empty classes that the classifier can never learn.

### 2.2 K-Means (L4 Adaptive)

Uses `sklearn.MiniBatchKMeans` with an equal-area projection:

```python
# Latitude scaled by cos(mean_lat) to approximate equal-area
mean_lat_rad = radians(coords[:, 0].mean())
cos_lat = max(cos(mean_lat_rad), 1e-6)
coords_proj[:, 1] *= cos_lat  # scale longitude
```

**Why cosine projection:** At the equator, 1° of longitude ≈ 111 km. At 60°N, 1° of longitude ≈ 55 km. Without scaling, k-means treats a 1°×1° box at the equator and at 60°N as equal-sized, but they're physically very different. Multiplying longitude by `cos(lat)` corrects this, making the Euclidean distance in projected space approximately proportional to actual ground distance.

**Why MiniBatchKMeans:** Regular KMeans on 10k points takes ~5 seconds. MiniBatchKMeans takes ~0.5 seconds. Both converge to the same solution for this problem size.

**Empty cluster fix:** MiniBatchKMeans can occasionally produce empty clusters (a cluster with 0 assigned points). The fix: iteratively find the largest cluster, take its farthest point from the centroid, and reassign it to the empty slot. This guarantees exactly K nonempty cells.

**Bounding boxes:** Each cell gets the axis-aligned bounding box of its assigned points. For k-means cells, boxes can overlap (two cells can both claim a point is inside their box). This is why assignment uses nearest-centroid fallback.

### 2.3 DBSCAN (L4 Adaptive)

The most complex builder:

1. **Binary search for eps:** DBSCAN's cluster count depends on `eps` (neighborhood radius). Smaller eps → more fragmentation → more clusters. Larger eps → more merging → fewer clusters. We binary-search 60 iterations for the eps closest to K=300.
2. **Noise reassignment:** DBSCAN labels outlier points as `label=-1` (noise). These get reassigned to the nearest non-noise cluster centroid via haversine distance.
3. **Merge/split to exactly K:** If the final cluster count after noise reassignment is ≠ K, we either merge the smallest cluster into its nearest neighbor, or split the largest cluster at its centroid median along the longest axis.
4. **Re-contiguify:** Cluster IDs are remapped to 0..K-1 contiguous integers.

**Why DBSCAN is the worst performer (Gini 0.559):** DBSCAN creates mega-clusters in dense urban areas (one cluster for all of London, one for all of Tokyo) while creating many tiny clusters in sparse areas. This extreme imbalance (min 3, max 1864 images per cell) makes the classifier's job harder — it sees very few examples for some classes and way too many for others.

---

## 3. Cell Assignment — `assign_cells()`

```python
def assign_cells(coords, cells):
    # 1. Collect all cells whose bounding box contains the point
    # 2. Among containing cells, pick nearest centroid (haversine)
    # 3. If no cell contains the point, pick nearest centroid overall
```

**Why this dual logic:** Quad-tree cells are non-overlapping boxes — containment check is sufficient. But k-means and DBSCAN cells can have overlapping bounding boxes (two Voronoi cells can have overlapping AABBs). The nearest-centroid fallback resolves this ambiguity.

**Why haversine for centroid distance:** Euclidean distance on raw lat/lon is wrong. Two points at (0, 0) and (0, 179) are 1° apart in longitude but ~20,000 km apart on Earth. Haversine computes the great-circle distance correctly.

---

## 4. The Model — `GeoClassifier`

```text
Input: (B, 3, 224, 224) — RGB image, ImageNet-normalized
    ↓
ResNet-50 backbone (torchvision, pretrained on ImageNet)
    ↓
avgpool → (B, 2048, 1, 1) → flatten → (B, 2048)
    ↓
Dropout (optional, 0.0 default)
    ↓
Linear(2048, K) → logits (B, K)
```

**Why ResNet-50:** It's the sweet spot for a 6GB GPU. ResNet-18 (11M params) is too weak for geographic features. ResNet-101 (44M params) OOMs with batch 32. ResNet-50 (25M params) fits comfortably with AMP.

**Why `nn.Sequential(*list(backbone.children())[:-1])`:** This strips the original `fc` layer from torchvision's ResNet-50, keeping everything up to `avgpool`. The original `fc` maps 2048→1000 (ImageNet classes). We replace it with our own `Linear(2048, K)`.

**Why Xavier initialization on the head:** The head is randomly initialized (Kaiming is the default, but Xavier gives slightly better initial distributions for classification heads). The bias is initialized to zeros.

**Freeze/unfreeze strategy:**

- **L1 (frozen):** All backbone parameters frozen, only the head is trainable. ~614K trainable params out of 25M total. This is the baseline.
- **L2 (layer4 fine-tuned):** The last block (layer4, 7M params) is unfrozen. Total trainable: ~7.6M params. This lets the backbone adapt its high-level features to geographic content.

**Why only layer4, not layer3 or earlier:** Lower layers detect edges/textures (universal). Layer4 detects high-level semantic features (buildings, landscapes, landmarks). Geographic photos have distinct high-level features (Eiffel Tower vs. Mount Fuji), so unfreezing layer4 lets the backbone learn these. Unfreezing layer3+ risks overfitting on 10k images.

---

## 5. Training Pipeline — `train.py`

### 5.1 Data Loading

```python
ds = GeoDataset(csv_path, image_size=224, train=True)
```

**CSV format:** `image_path,lat,lon` — three columns. No metadata, no country labels, no timestamps. Minimal and clean.

**Transforms (train):**

```python
RandomResizedCrop(224, scale=(0.8, 1.0))  # random crop 80-100% of image
RandomHorizontalFlip(p=0.5)               # 50% chance flip
ToTensor()                                 # PIL → tensor [0,1]
Normalize(mean=[0.485, 0.456, 0.406],     # ImageNet stats
           std=[0.229, 0.224, 0.225])
```

**Why these augmentations:** `RandomResizedCrop` with `scale=(0.8, 1.0)` means the model sees 80-100% of the image at a random crop position. This teaches invariance to framing/composition. Horizontal flip is safe because geographic scenes (buildings, landscapes) are generally left-right symmetric. We don't use color jitter, rotation, or perspective transforms because they could destroy geographic cues (e.g., rotating a photo of the Eiffel Tower 180° makes it point down).

**Transforms (eval):**

```python
Resize(256)          # shortest side to 256
CenterCrop(224)      # center crop to 224×224
ToTensor()
Normalize(...)
```

**Why Resize+CenterCrop:** Deterministic. Every eval run produces the same result. No randomness.

### 5.2 Cell Label Assignment

```python
coords = ds.get_coords()            # Nx2 tensor of (lat, lon)
cells = build_cells(coords, K=300)  # or load from JSON
cell_ids = assign_cells(coords_np, cells)
ds.cell_ids = cell_ids              # assign labels to dataset
```

**Why assign before DataLoader:** The dataset's `__getitem__` returns `(image, cell_id)`. The cell_id is computed once and stored. This is faster than computing cell assignment on-the-fly during training.

**Why fixed cells (JSON) for L2:** In L2, we compare 1%, 10%, and 100% of data. If each fraction built its own cells, the classifier head dimensions would differ, making comparisons unfair. Fixed cells ensure all fractions use the same 300 classes.

### 5.3 Subset Selection

```python
rng = np.random.RandomState(seed)
subset_indices = rng.choice(full_len, size=max_samples, replace=False)
```

**Why RandomState not torch.Generator:** Deterministic across Python/NumPy versions. `torch.Generator` behavior can change between PyTorch versions.

### 5.4 Optimizer and Loss

```python
optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
```

**Why AdamW:** Standard choice for fine-tuning. Learning rate 1e-3 is aggressive but works because only the head (or head + layer4) is being trained — the pretrained backbone features are already good.

**Why CrossEntropyLoss:** Standard for multi-class classification. No label smoothing, no focal loss. Keep it simple for the baseline.

### 5.5 AMP (Automatic Mixed Precision)

```python
use_amp = args.amp and device.type == "cuda"
scaler = GradScaler("cuda")
with autocast("cuda"):
    logits = model(images)
    loss = criterion(logits, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Why AMP:** ResNet-50 with batch 32 at 224px uses ~3.5GB VRAM in FP32. AMP uses FP16 for most ops, reducing memory to 2GB and giving ~1.5x speedup on RTX 4050 (which has Tensor Cores for FP16).

**Why `torch.amp` not `torch.cuda.amp`:** Torch 2.x moved AMP to `torch.amp`. The code has a fallback for older PyTorch.

### 5.6 Checkpointing

Every epoch saves `last.pt` and `best.pt` (best by training loss):

```python
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "args": train_args,
    "cells": [{"cell_id": c.cell_id,
               "centroid_lat": c.centroid_lat,
               "centroid_lon": c.centroid_lon} for c in cells],
    "loss": epoch_loss,
    "num_cells": num_classes,
}, ckpt_path)
```

**Why store cells in the checkpoint:** At eval time, you need the cell centroids to map predicted cell_id → (lat, lon). Storing them in the checkpoint makes it self-contained — you don't need the training CSV or cell builder to evaluate.

### 5.7 OOM Handling

```python
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("[OOM] ... re-run with --batch-size 16 or 8 ...")
    raise
```

**Why not auto-retry:** Automatic batch-size reduction changes the training dynamics (effective batch size affects gradient noise). Better to let the user decide.

---

## 6. Evaluation — `eval.py`

```python
pred_ids = logits.argmax(dim=1)         # (B,) predicted cell IDs
pred_lats = centroids[pred_ids, 0]      # map to centroid lat
pred_lons = centroids[pred_ids, 1]      # map to centroid lon
dists = haversine_km_vec(pred_lats, pred_lons, true_lats, true_lons)
```

**Haversine distance:**

```text
a = sin(dphi/2)² + cos(φ1)·cos(φ2)·sin(dlambda/2)²
c = 2·arcsin(√a)
distance = R · c    # R = 6371 km
```

**Why haversine not Vincenty:** Haversine is simpler, sufficient for distances > 1km, and doesn't need convergence iteration. Vincenty is more accurate for very short distances, but we don't care about sub-100m accuracy.

**Metrics:** `within_1km`, `within_25km`, `within_200km` (percentage of predictions within that distance of ground truth). This is the standard in geographic estimation literature (PlaNet, GeoCLIP, etc.).

---

## 7. Uncertainty — L3 `src/uncertainty.py`

### 7.1 Temperature Scaling

```python
class TemperatureScaler(nn.Module):
    def __init__(self):
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, logits):
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits, labels):
        # LBFGS optimizer on T only
        # Loss = CrossEntropy(logits / T, labels)
        # Minimize over calibration set
```

**Why temperature scaling:** Neural networks are overconfident. The raw softmax outputs are not calibrated — when the model says 90% confidence, it's actually correct only 70% of the time. Temperature scaling divides all logits by T > 1, which flattens the softmax distribution, reducing overconfidence.

**Why LBFGS:** It's a second-order optimizer that converges in ~20-30 iterations for a single scalar parameter. Adam would also work but takes more iterations.

**Why T = 3.94 (the measured result):** The model is extremely overconfident. T=3.94 means the effective logits are divided by 4, which massively softens the distribution. This makes sense — 300 coarse cells on 3000 global test images means the model rarely sees the true class, so its raw confidence is misleading.

**Clamp to [0.01, 100]:** Prevents degenerate solutions where T→0 (infinite confidence) or T→∞ (uniform distribution).

### 7.2 Expected Calibration Error (ECE)

```python
def expected_calibration_error(probs, labels, n_bins=15):
    conf, pred = probs.max(dim=1)    # confidence = max probability
    correct = pred.eq(labels)         # 1 if correct, 0 if wrong
    for each bin [lo, hi):
        frac = fraction of samples in bin
        bin_acc = accuracy in bin
        bin_conf = mean confidence in bin
        ece += frac * |bin_acc - bin_conf|
```

**Why 15 bins:** Standard in calibration literature (Guo et al. 2017). Too few bins (< 5) miss calibration structure. Too many bins (> 25) are noisy with small datasets.

**Target ECE < 0.1 (stretch < 0.05):** ECE = 0 means perfectly calibrated. ECE = 0.1 means on average the confidence is off by 10%. We achieved 0.022 — well below the stretch target.

### 7.3 Conformal Prediction

```python
def fit_conformal_quantile(probs_cal, labels_cal, alpha=0.1):
    true_probs = probs_cal[arange(n), labels_cal]
    scores = 1.0 - true_probs          # nonconformity: 1 - P(correct)
    q = quantile(scores, ceil((n+1)(1-alpha)) / n)  # finite-sample quantile
    return q

def conformal_prediction_set(probs, alpha=0.1, quantile=q):
    nonconformity = 1.0 - probs
    mask = nonconformity <= q           # include class k if 1-p_k <= q
    # guarantee at least one class per sample
    mask[arange(n), probs.argmax(1)] = True
```

**Why conformal prediction:** Instead of "the answer is cell 47", conformal says "the answer is in {cell 47, cell 123, cell 201}" with 90% coverage guarantee. This is more honest — the model tells you when it's uncertain.

**Why `ceil((n+1)(1-alpha))/n`:** This is the finite-sample correction from Angelopoulos & Bates 2023. With 1000 calibration samples and alpha=0.1, the quantile level is `ceil(1001 * 0.9) / 1000 = 0.901`. This ensures the coverage guarantee holds even for finite calibration sets.

**Why mean set size = 188/300:** The model is uncertain about most predictions. To achieve 90% coverage, you need to include 188 out of 300 classes on average. This reflects the low base accuracy — the model's top-1 is correct only ~0.25% of the time, so you need a huge net to catch 90% of true labels.

### 7.4 Abstention

```python
def abstention_mask(probs, threshold=0.5):
    max_prob = probs.max(dim=1).values
    return max_prob >= threshold
```

**Why 0.5 threshold:** This is a "confident only" filter. After temperature scaling (T=3.94), almost no prediction has max probability > 0.5, so the abstention rate is 99.95%. This is correct behavior — the model genuinely doesn't know, and it's better to say "I don't know" than give a wrong answer.

---

## 8. The Training Runner — `run_l2_experiments.py`

The matrix runner automates the 12-run L2 experiment:

```text
2 inits × 3 fractions × 2 regimes = 12 runs
```

**Plan-only default:** Every script defaults to printing what it would do without executing. This is critical for safety — you can review the commands before committing GPU time.

**Subset CSV generation:** The runner creates deterministic subset CSVs (e.g., `metadata.csv_sub0.01_s42.csv`) so each fraction is reproducible.

**Skip-existing:** If a checkpoint already exists, the runner skips that run. This enables safe resume after crashes.

---

## 9. The L4 Runner — `run_l4_experiments.py`

Same pattern as L2 but for cell methods:

```text
3 methods × same K × same regime = 3 runs
```

**Eval-only mode:** If checkpoints already exist (from a previous run), the runner can skip training and only evaluate. This is useful for quick re-evaluation.

**Cell balance metrics:** Measured on the training distribution:

- **Gini coefficient:** 0 = perfectly balanced, 1 = maximally imbalanced
- **Min/Max:** Range of images per cell
- **Median:** Middle value (robust to outliers)

**Urban/rural proxy:** Since OSV-5M has no urban/rural labels, we use cell count as a proxy. Dense cells (> median count) are "urban", sparse cells (≤ median) are "rural". This is transparently documented as an approximation.

---

## 10. The Complete Data Flow

```text
Raw OSV-5M images + metadata.csv (image_path, lat, lon)
    ↓
download_subset.py — download 10k images, 4 shards, shard-balanced
    ↓
build_cells() — quad-tree / kmeans / dbscan → cells JSON
    ↓
assign_cells() — map each (lat,lon) to a cell ID
    ↓
GeoDataset — returns (image_tensor, cell_id)
    ↓
DataLoader — batches of (B, 3, 224, 224) + (B,)
    ↓
GeoClassifier — ResNet-50 backbone → Linear(2048, K) → logits (B, K)
    ↓
CrossEntropyLoss(logits, labels) → loss
    ↓
AdamW + AMP + GradScaler → optimizer step
    ↓
Checkpoint: model_state + cells + args + loss
    ↓
eval.py — argmax → cell_id → centroid (lat,lon) → haversine distance
```

---

## 11. All Decisions and Why

| Decision | Choice | Why |
|---|---|---|
| Task framing | Classification, not regression | Haversine loss is non-Euclidean; classification sidesteps |
| Backbone | ResNet-50 | Fits 6GB GPU with batch 32; ResNet-18 too weak, ResNet-101 OOMs |
| Cell count | K ≈ 300 | 4050 constraint; 3000 too few images per class; 30 too coarse |
| Cell construction | Quad-tree L1, then kmeans L4 | Quad-tree is deterministic and fast; kmeans is more balanced |
| Initialization | ImageNet > Places365 | Measured: ImageNet wins at 1% and 100% data fractions |
| Training regime | Layer4 fine-tuned > frozen | Measured: 3.33% vs 2.53% within-200km |
| Optimizer | AdamW, lr=1e-3 | Standard for fine-tuning; only head/layer4 trained |
| Loss | CrossEntropy | Standard; no focal loss needed at K=300 |
| AMP | Yes, FP16 | ~1.5x speedup, ~40% memory reduction on RTX 4050 |
| Augmentation | RandomResizedCrop + HFlip | Minimal; too much augmentation destroys geographic cues |
| Evaluation | Haversine within 1/25/200km | Standard in geo-estimation literature |
| Calibration | Temperature scaling | Post-hoc, no retraining, ECE 0.43→0.02 |
| Conformal | Split conformal | Finite-sample coverage guarantee; honest uncertainty |
| Abstention | Max-probability threshold | Simple, interpretable, works with calibrated probs |
| Default cell method | K-means | Best within-200km (4.07%), best balance (Gini 0.324) |
| Windows safety | pathlib, num_workers=0 | Developed on Windows 4050; no multiprocessing forks |
| Plan-only default | All scripts | Prevents accidental GPU use; review before execution |
