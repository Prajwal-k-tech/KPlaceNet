# KPlaceNet / GeoNet

Evolutionary, layer-by-layer — PlaNet-style coarse geolocation on RTX 4050 (6GB VRAM).

> Locked plan: `docs/implementation_plan.md` — L0→L4, solo + 4050 constraints. Do not add new datasets/backbones without re-plan.

## Setup (RTX 4050, Windows + Python 3.10+, torch 2.x)

```powershell
# 1. Create env (example with venv)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install deps (no heavy extras)
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify imports (no GPU needed for import check)
python -c "import src.dataset, src.cells, src.model, src.train, src.eval, src.uncertainty; print('imports ok')"
```

> No `pip install` is run by the scaffold — install manually when ready.

## Data Layout (gitignored subsets)

```
data/
  README.md          # explains gitignored subsets
  flickr_geo_tiny/   # 5-10k debug — via scripts/download_subset.py --dataset flickr_geo_tiny
  osv5m_subset_10k/  # train-split subset — via --dataset osv5m --split train (plan first, then --yes)
  osv5m_test/        # official OSV test, eval-only — via --split test (never train)
  im2gps3k/          # 3k eval only — never train — via --dataset im2gps3k
  .gitkeep
```

- CSV format expected by `src/dataset.py`: `image_path,lat,lon` (header required). Paths may be absolute or relative to CSV location.
- OSV-5M spatial separation (1km) is respected by not mixing splits; IM2GPS3k is eval-only.
- Large datasets are **not** committed — see `data/README.md`.

## How to Run — Smoke Test (L0 exit gate)

```powershell
# 1) Create a tiny dummy CSV (50 rows) for overfit check
python scripts/download_subset.py --dataset flickr_geo_tiny --output-dir data/flickr_geo_tiny --max-samples 50 --dry-run

# 2) Real OSV-5M subset — SAFE downloader (never fetches the ~259GB repo).
#    WARNING: osv5m/osv5m is about 259GB in full and the selected shards are multi-GB.
#    Always inspect the plan first (no network, no files), then re-run with --yes.
python scripts/download_subset.py --dataset osv5m --split train --output-dir data/osv5m_subset_10k --max-samples 10000 --plan-only
python scripts/download_subset.py --dataset osv5m --split train --output-dir data/osv5m_subset_10k --max-samples 10000 --yes
#    Official OSV test (eval-only, separate dir — never train on it):
python scripts/download_subset.py --dataset osv5m --split test --output-dir data/osv5m_test --max-samples 3000 --plan-only
#    IM2GPS3k is manual-only (authors' download) — the script prints instructions.

# 3) Train smoke — overfit 50 images, 224px, AMP, batch 32->16 fallback note
python -m src.train --csv data/flickr_geo_tiny/metadata.csv --num-cells 300 --epochs 2 --batch-size 32 --image-size 224 --amp

# 4) Eval — within 1/25/200 km via haversine
python -m src.eval --csv data/im2gps3k/metadata.csv --checkpoint checkpoints/last.pt --num-cells 300
```

- `src/train.py` uses AMP (`torch.cuda.amp`) and CE loss; checkpoint saved to `checkpoints/last.pt`.
- If OOM on 4050: re-run with `--batch-size 16` or `--batch-size 8` (plan Section 3/8).
- `src/cells.py` provides `build_cells` / `assign_cells` (quad-tree K~300 in L1; k-means/DBSCAN in L4).
- `src/uncertainty.py` provides L3 uncertainty: `TemperatureScaler`, `expected_calibration_error`, `conformal_prediction_set`, `abstention_mask`.

## Repo Layout

```
KPlaceNet/
  docs/implementation_plan.md
  src/dataset.py  cells.py  model.py  train.py  eval.py  uncertainty.py
  scripts/download_subset.py
  notebooks/01_data_explore.ipynb
  data/  # gitignored
  requirements.txt
```

## Notes

- Windows paths safe — uses `pathlib.Path`, no hardcoded absolute paths.
- Imports use `src.*` absolute form; `src` is a package (`src/__init__.py`).
- L0, L1, and the full L2 matrix are complete. Results are recorded in `docs/results/l1_baseline.md` and `docs/results/l2_data_efficiency.md`; generated datasets/checkpoints remain gitignored.

## L2 — Data-Efficient Training (Gap 4)

L2 varies data fraction (1% / 10% / 100%), transfer init (ImageNet vs Places365), and training regime (frozen vs layer4 fine-tune) to test the 10x data-saving hypothesis.

### Quick Start — Plan-Only (no downloads, no GPU)

```powershell
# Print the full 12-run matrix plan (no execution)
python scripts/run_l2_experiments.py --plan-only

# Print a single fraction/init/regime combination
python scripts/run_l2_experiments.py --plan-only --fractions 0.01 --inits imagenet --regimes frozen
```

### One-Run Smoke Test (ImageNet only, no Places365 download needed)

```powershell
# 1% fraction, ImageNet init, frozen backbone, 1 epoch — fast smoke check
python scripts/run_l2_experiments.py --run --fractions 0.01 --inits imagenet --regimes frozen --epochs 1

# Or run train.py directly with fixed cells
python scripts/download_subset.py --dataset flickr_geo_tiny --output-dir data/flickr_geo_tiny --dry-run
python -m src.train --csv data/flickr_geo_tiny/metadata.csv --num-cells 300 --epochs 1 --batch-size 32 --amp --save-cells checkpoints/cells_smoke.json
python -m src.train --csv data/flickr_geo_tiny/metadata.csv --num-cells 300 --epochs 1 --batch-size 32 --amp --cells-json checkpoints/cells_smoke.json --max-samples 25
```

### Full L2 Runs (requires 10k CSV + optional Places365 checkpoint)

```powershell
# Download official Places365 checkpoint (~97 MB, plan-only by default)
python scripts/download_places365.py                          # prints plan
python scripts/download_places365.py --yes                    # downloads to checkpoints/places365/

# Run all 12 combinations (uses sys.executable, safe subprocess)
python scripts/run_l2_experiments.py --run --epochs 10

# Run specific subset
python scripts/run_l2_experiments.py --run --fractions 0.01 0.10 --inits imagenet places365 --regimes frozen layer4 --skip-existing
```

### L2 Checkpoint Structure

Each run saves to `checkpoints/l2_<init>_<fraction>_<regime>/`:
- `last.pt` / `best.pt` — model + optimizer + cells + `num_cells` (effective)
- `metrics_<run_tag>.json` — per-epoch loss, cell accuracy, trainable params, device, elapsed time, args

### Fixed-Cell Fairness

All data fractions in an L2 experiment use the same cells (built once from the full 10k CSV, saved as JSON). This ensures the classifier head has identical architecture across fractions — the only variable is training data size. Pass `--cells-json` to train.py to load fixed cells; otherwise cells are built from the CSV.

### Cell-Balanced Sampling Comparison

Training remains uniform by default. To compare that baseline with inverse-
frequency cell sampling on the same data fractions, model setup, and seed:

```powershell
# Print the paired commands without downloading data or starting training
python scripts/run_l2_experiments.py --plan-only --fractions 1.0 --inits imagenet --regimes frozen --sampling-modes uniform cell-balanced

# Execute both runs once the training subset and dependencies are available
python scripts/run_l2_experiments.py --run --fractions 1.0 --inits imagenet --regimes frozen --sampling-modes uniform cell-balanced --epochs 10

# Evaluate both checkpoints on the same untouched OSV-5M test subset
python -m src.eval --csv data/osv5m_test/metadata.csv --checkpoint checkpoints/l2_imagenet_1.0_frozen/last.pt
python -m src.eval --csv data/osv5m_test/metadata.csv --checkpoint checkpoints/l2_imagenet_1.0_frozen_cell-balanced/last.pt

# Repeat the paired experiment with three training seeds (separate checkpoint dirs)
python scripts/run_l2_experiments.py --run --fractions 1.0 --inits imagenet --regimes frozen --sampling-modes uniform cell-balanced --seeds 42 43 44 --epochs 10

# Evaluate every paired checkpoint and write a hash/provenance-bearing JSON report
python scripts/evaluate_sampling_comparison.py --csv data/osv5m_test/metadata.csv --seeds 42 43 44
```

`--sampling-power` controls the inverse-frequency exponent: 0 gives each
example equal probability under replacement, while 1 gives each observed
training cell equal expected sampling mass. Both weighted modes sample with
replacement, so some examples repeat and others may be skipped within an
epoch for any weighted-sampling power, including 0. Use `--sampling-mode
uniform` for the original shuffled, no-replacement baseline. This intentionally
changes the training distribution; compare the
untouched test set and report both overall and geographic-stratum metrics.
Use multiple seeds before treating a difference as a robust performance claim.
The comparison script records per-checkpoint metrics and hashes, test-CSV hash,
runtime versions, and balanced-minus-uniform deltas for each shared seed.

## L3 — Uncertainty-Aware Geolocation (Gap 3)

L3 adds post-hoc uncertainty calibration to the L2 winner (`l2_imagenet_1.0_layer4`):
temperature scaling (Guo et al. 2017), split-conformal prediction sets, and
confidence-based abstention.

### Fork contribution

This fork corrects the finite-sample conformal cutoff to use the discrete
order statistic `k = ceil((n + 1) * (1 - alpha))`; when `k > n`, the
conservative threshold is infinity. L3 now uses three disjoint seeded
partitions: temperature fitting, conformal calibration, and evaluation. Its
JSON artifact records split indices and sample IDs, CSV/checkpoint hashes,
all inference configuration, relevant software versions, repository
revision, and dirty-tree status. Empty abstention subsets are represented as
missing distances rather than nonstandard JSON `NaN`. These changes make the
protocol auditable; they do not establish exchangeability or regional
coverage under geographic shift. The upstream L3 numbers below remain
historical and are not measurements of this fork's changes.

### Evaluation integrity

`src.eval` requires a trained checkpoint containing both `model_state` and the
cell centroids used by its classifier. It rejects missing or inconsistent
checkpoint metadata, loads weights strictly, and never builds classes from the
evaluation CSV or reports scores from random weights. This keeps evaluation
labels out of model/class construction; use a training checkpoint produced by
`src.train`.

See [`docs/FORK_STATUS.md`](docs/FORK_STATUS.md) for the fork's current
contribution scope, validation evidence, and real-data reproduction status.

### Quick Start — Plan-Only (no GPU, no images loaded)

```powershell
# Print the full L3 experiment plan (default: plan-only)
python scripts/run_l3_uncertainty.py

# Custom calibration size and alpha
python scripts/run_l3_uncertainty.py --calibration-size 500 --alpha 0.05
```

### Full L3 Run (requires OSV test images + L2 checkpoint)

```powershell
# Run L3 on the selected checkpoint with default settings
#   - Temperature fit: 500 seeded random OSV-test rows
#   - Conformal calibration: separate 1000 seeded random rows
#   - Evaluation: remaining 1500 rows
#   - Temperature scaling + conformal + abstention analysis
python scripts/run_l3_uncertainty.py --run

# Custom settings
python scripts/run_l3_uncertainty.py --run --alpha 0.05 --threshold 0.3 0.5 0.7
```

> **Calibration caveat:** The L3 run uses three disjoint seeded partitions of
> the official OSV test subset (`data/osv5m_test/metadata.csv`): 500 rows fit
> temperature, a separate 1000 rows fit the conformal threshold, and the
> remaining 1500 rows are evaluation-only. The generated results JSON records
> exact indices, stable sample IDs, CSV and checkpoint SHA-256 hashes, library
> versions, and a split hash. The finite-sample conformal statement is marginal
> and requires calibration and future examples to be exchangeable. Random row
> splitting does not establish geographic exchangeability, guarantee regional
> coverage, or provide a guarantee under geographic distribution shift.

The run also reports empirical coverage, mean prediction-set size, top-1
accuracy, and geodesic-distance thresholds by occupied equal-area grid region.
Latitude bands are uniform in `sin(latitude)` and longitude bands are uniform,
so grid cells have approximately equal spherical area. Regions below the
minimum count remain in the JSON artifact with `metrics: null`; reported
coverage includes a 95% Wilson interval. These are descriptive diagnostics,
not conditional coverage guarantees, and the intervals do not correct for
spatial dependence. Configure the grid with `--region-lat-bands`,
`--region-lon-bands`, and `--region-min-count`.

## L4 — Adaptive Cell Construction (Gap 1)

L4 compares three cell construction methods at K=300 using the L2 winner backbone
(ResNet-50, ImageNet init, layer4 fine-tune, 100% data, 10 epochs):

- **quad-tree** — density-driven recursive spatial bisection (axis-aligned boxes)
- **kmeans** — MiniBatchKMeans on (lat, lon) with cos-mean-lon equal-area scaling
- **dbscan** — DBSCAN density clustering with eps binary search to hit ~K clusters,
  noise reassignment to nearest centroid, merge/split normalization to exactly K

### Quick Start — Plan-Only (no GPU, no images loaded)

```powershell
# Print the full L4 experiment plan (default: plan-only)
python scripts/run_l4_experiments.py

# Custom methods
python scripts/run_l4_experiments.py --methods kmeans dbscan
```

### Full L4 Run (requires 10k training CSV + OSV test images)

```powershell
# Run all 3 methods (builds cells, trains, evaluates)
python scripts/run_l4_experiments.py --run

# Resume with existing checkpoints
python scripts/run_l4_experiments.py --run --skip-existing

# Single method
python scripts/run_l4_experiments.py --run --methods kmeans
```

### Adaptive Assignment

`assign_cells` handles both quad-tree boxes and adaptive/overlapping clusters:
1. Collect all cells whose bounding box contains the point
2. Among containing cells, pick the nearest centroid (haversine)
3. If no cell contains the point, pick the nearest centroid overall

This resolves ambiguity when k-means/DBSCAN clusters overlap spatially.

### Urban/Rural Proxy

OSV-5M has no urban/rural labels.  L4 uses a **cell-count proxy**: cells with
training point count > median are "urban" (dense), count <= median are "rural"
(sparse).  This is reported transparently as an approximation — see
`docs/results/l4_adaptive_cells.md` for caveats.

### Results

| Method | @1km | @25km | @200km | Mean Dist | Gini |
|--------|------|-------|--------|-----------|------|
| quad_tree | 0.00% | 0.27% | 3.33% | 6025 km | 0.370 |
| kmeans | 0.00% | 0.17% | 4.07% | 6058 km | 0.324 |
| dbscan | 0.00% | 0.07% | 3.47% | 6084 km | 0.559 |

**Recommended default: kmeans** (highest within-200km, best cell balance).

Full results in `docs/results/l4_adaptive_cells.md`.

## Demo

The Gradio demo accepts an image and displays the checkpoint's coarse cell
centroid on a map. It uses the default L2 checkpoint path unless overridden;
the optional calibration artifact is shown as absent when it is missing, and
invalid calibration files fail clearly instead of silently reverting to
uncalibrated predictions. When the calibration artifact includes a checkpoint
SHA-256 (as the L3 runner does), the demo checks it against the selected model
and rejects a mismatch.

```sh
# Optional: point the demo at any trained checkpoint and matching L3 artifact
export KPLACENET_CHECKPOINT=checkpoints/l2_imagenet_1.0_frozen_cell-balanced/last.pt
export KPLACENET_CALIBRATION=checkpoints/l3_uncertainty_results.json
python demo/app.py
```

If no calibration file is available, the UI labels the output as uncalibrated
softmax confidence with top-1 only; it does not imply a conformal guarantee.

The dependency-free checks run with Python 3.11 and the standard library:

```sh
python -m unittest discover -s tests -p 'test_eval_contract.py' -v
python -m unittest discover -s tests -p 'test_regional_eval.py' -v
python -m unittest discover -s tests -p 'test_demo_config.py' -v
python -m unittest discover -s tests -p 'test_sampling*.py' -v
```

GitHub Actions runs these checks on pushes and pull requests. The full test
suite still needs the ML dependencies listed in `requirements.txt`.

```powershell
# Launch from the repository root
python demo/app.py
# or: python -m demo.app
```

Upload a photo → global + zoomed (+/-5 deg) matplotlib map of the predicted cell, plus calibrated vs raw confidence, temperature T, conformal set size, and an ACCEPT / ABSTAIN verdict. Haversine distance is only measurable on the labeled test set (no ground truth for uploads).
