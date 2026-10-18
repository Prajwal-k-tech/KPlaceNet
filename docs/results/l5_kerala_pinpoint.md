# L5 — Kerala Pinpoint Mode

## Hypothesis

Warm-starting from the selected global backbone and fine-tuning on a single
region with region-specific cells can turn near-total failure in that region
into useful coarse localization, without retraining from scratch.

## Methodology

### Setup

L5 ("Kerala pinpoint mode") is an additive, region-specific model. It is
warm-started from the L2 winner (`l2_imagenet_1.0_layer4`) and does not replace
the global model.

- **Backbone:** ResNet-50, warm-started from `l2_imagenet_1.0_layer4`
- **Regime:** layer4 + head fine-tuned (not from scratch)
- **Epochs:** 10
- **Batch size:** 16 (AMP on)
- **LR:** 3e-4
- **Image size:** 224px
- **Seed:** 42
- **Device:** single RTX 4050 (~49 s/epoch)

### Cells

300 cells are built with equal-area k-means (seed 42) from **train-only** Kerala
coordinates, so val/test coordinates do not influence cell geometry. These cells
are separate from the global model's 300 quad-tree cells and are not
interchangeable with them.

| Property | Value |
|---|---|
| Nearest-centroid distance p10 / median / p90 | 3.29 / 5.71 / 14.98 km |
| Cell size mean / median | 53.1 / 46.5 |
| Cell size min / max | 6 / 253 |
| Cell size Gini | 0.3406 |

### Data

OSV-5M Kerala slice (CC-BY-SA-4.0): 15,939 train / 1,000 seeded random val
holdout / 357 official test.

### Training Curve

Train loss and train top-1 cell accuracy, epoch 1 vs epoch 10:

| Epoch | Train loss | Train top-1 cell acc |
|---:|---:|---:|
| 1 | 3.614 | 23.9% |
| 10 | 0.109 | 97.6% |

The curve is monotonic with no instability.

## Evaluation Results

Before = L2 global model, after = L5 Kerala mode. `—` marks a metric that was
not part of the before-model evaluation.

### VAL — 1,000 Seeded Random Holdout

| Metric | Before (L2 global) | After (L5 Kerala) |
|---|---:|---:|
| Mean dist | 7,632.8 km | 47.6 km |
| Median dist | 6,253.1 km | 7.0 km |
| p90 dist | — | 165.7 km |
| @1km | — | 8.4% |
| @25km | 0.0% | 64.6% |
| @200km | 14.7% | 92.9% |
| Cell top-1 | 20.4% | 43.1% |

### TEST — 357 Official

| Metric | Before (L2 global) | After (L5 Kerala) |
|---|---:|---:|
| Mean dist | 7,644.5 km | 102.8 km |
| Median dist | 6,330.0 km | 42.1 km |
| p90 dist | — | 291.6 km |
| @1km | — | 0.6% |
| @25km | 0.0% | 41.2% |
| @200km | 7.6% | 80.1% |
| Cell top-1 | 16.2% | 17.9% |

### Per-Town (After)

| Town | VAL mean km | VAL @25km | VAL n | TEST mean km | TEST @25km | TEST n |
|---|---:|---:|---:|---:|---:|---:|
| kochi | 36.4 | 68.7% | 150 | 14.1 | 77.8% | 9 |
| kottayam | 39.7 | 64.1% | 195 | 85.6 | 38.9% | 18 |
| munnar | 20.3 | 69.2% | 26 | 47.2 | 61.5% | 13 |
| other_kerala | 50.9 | 64.1% | 612 | 106.5 | 39.9% | 306 |
| trivandrum | 158.4 | 47.1% | 17 | 168.5 | 27.3% | 11 |

### Chance Baseline

With 300 cells, top-1 chance is 0.33%. A random cell guess lands roughly
150–200 km from the true location, so any Kerala-only model looks good on mean
distance by construction; the discriminative metrics are @1km/@25km, median, and
top-1. The before model's mean of ~7,600 km means its predictions were far
outside Kerala, so the mean is not what shows the improvement here.

## Files Produced

```
checkpoints/l5_kerala_kmeans300/
  results.json            # full results JSON
  samples_*.csv           # per-sample predictions
  metrics_l5_kerala.json  # per-epoch training metrics (loss, train acc)
```

## Reproduction

```powershell
# Evaluation
python scripts/run_l5_kerala.py --phase eval
```

## Caveats

- **The random val holdout is optimistic.** OSV-5M images cluster along streets,
  so a random split can share streets/locations between train and val. The
  official test split (median 42.1 km vs 7.0 km val) is the honest number.
- **Far south is the weakest cluster.** Both splits show the Trivandrum region
  at the bottom. The observed failure mode is far-south images (lat ≈ 8.1–8.2)
  predicted roughly 500 km north.
- **No calibration.** No L3 calibration was fitted for this model; the demo
  reports raw confidence only for Kerala mode.
- **Domain.** OSV-5M is street-level imagery; phone photos will differ.

## Context

Global L2/L3/L4 numbers are untouched. Kerala mode is an additive, separate
model used by the demo's "Kerala · regional model" toggle.

## Findings

1. Fine-tuning a global model on one region with region-specific cells turns
   near-total failure (0% @25km) into useful coarse localization: 41.2% @25km
   on the official test split, median 42.1 km.
2. The gap between the random holdout (64.6% @25km) and the official test split
   (41.2% @25km) quantifies spatial autocorrelation inflation in the
   random-split metric.
3. Remaining headroom is the far south (Trivandrum) plus finer cells.
