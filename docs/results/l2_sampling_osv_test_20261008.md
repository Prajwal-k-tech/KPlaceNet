# Cell-balanced sampling on a fixed OSV-5M subset

**Status:** This report records the initial three-seed sampling comparison on one fixed 3,000-row test sample. A separate 10k-versus-20k experiment is in progress and will use a second, ID-disjoint sample from the same official test split; that follow-up will be documented separately. The follow-up was motivated after examining this report, so it is exploratory rather than a pre-registered confirmation.

## Question and protocol

Does inverse-frequency cell-balanced sampling improve geolocation over ordinary shuffled sampling when the dataset, cell definitions, model, and training budget are held fixed?

Both conditions use ImageNet-initialized ResNet-50, the same 300 quadtree cells, layer4 fine-tuning, batch size 16, 224-pixel images, AdamW (`lr=0.001`, `weight_decay=0.0001`), AMP, and ten epochs. The uniform arm shuffles without replacement. The balanced arm samples with replacement using inverse training-cell frequency, so it changes the per-epoch example distribution while keeping the epoch length fixed. Seeds 42, 43, and 44 use the same 10,000 training examples and cell definitions within each pair. The selected checkpoint is the final epoch; the test scores do not select checkpoints.

Training used 10,000 official OSV-5M training rows, 5,000 each from archive shards 00 and 01. Evaluation used the same 3,000-row subset drawn across all five official test archives. The train and test sample identifiers are unique and disjoint, and all sampled images decoded successfully. Training metadata SHA-256: `dd0d70ffc916aece79602ed9e623905e5e8a81dc2002d1a7b47e894c0d6617be`. Test metadata SHA-256: `d0ae3837e459d3171fbee3e2882e0c2c7634f6ffd5f7442d11274b80c5ec69ac`. The train subset comes from only two of the 98 training archives; their representativeness is unknown. This is a fixed-subset experiment, not a full OSV-5M reproduction. OSV-5M documents its own spatially separated train/test protocol; this work preserves those split labels. See the [OSV-5M paper](https://arxiv.org/abs/2404.18873) and [dataset card](https://huggingface.co/datasets/osv5m/osv5m).

## Held-out results

Values below are mean ± sample standard deviation across three training seeds. Distance is mean geodesic distance in kilometers; within-threshold values are percentages.

| Sampling | Within 1 km | Within 25 km | Within 200 km | Mean distance | Median distance |
|---|---:|---:|---:|---:|---:|
| Uniform | 0.00 ± 0.00% | 0.20 ± 0.03% | 3.38 ± 0.38% | 5,849.9 ± 183.9 km | 4,653.6 ± 391.8 km |
| Cell-balanced | 0.00 ± 0.00% | 0.16 ± 0.02% | 3.38 ± 0.16% | 5,859.0 ± 143.5 km | 4,759.2 ± 308.7 km |

Paired deltas are cell-balanced minus uniform. The three per-seed mean-distance deltas were `+243.3`, `−16.3`, and `−199.8` km; their mean was `+9.1 km` (sample SD `222.7 km`). The paired within-200-km delta averaged `0.00` percentage points. These results show no consistent improvement from cell-balanced sampling under this setup. The balanced runs also had lower final training cell accuracy (93.49–93.80%) than uniform (96.11–96.52%); that is a training diagnostic, not evidence of better geolocation.

For context, always predicting the most frequent training-cell centroid gives 9,585.5 km mean error, 10,509.2 km median error, and 0.13% within 200 km on the same test rows. The uniform models improve over this simple class-prior baseline, but their absolute geolocation accuracy remains weak. The baseline is computed from the hash-verified fixed-cell artifact and is included in the JSON report.

The evaluator reports descriptive distances in a 6 × 12 equal-area latitude/longitude grid. Thirty-seven of the 52 occupied regions had at least 20 test rows and received metrics; sparse regions remain in the machine-readable report without scores. Regional slices are descriptive only and do not imply conditional accuracy or coverage guarantees.

## Reproduction and artifacts

Run from the repository root after downloading the same dataset subsets and ImageNet weights:

```bash
python scripts/run_l2_experiments.py \
  --csv data/osv5m_subset_10k/metadata.csv \
  --fractions 1.0 --inits imagenet --regimes layer4 \
  --sampling-modes uniform cell-balanced --seeds 42 43 44 \
  --epochs 10 --device cuda --num-workers 4 --run --skip-existing

python scripts/evaluate_sampling_comparison.py \
  --csv data/osv5m_test/metadata.csv \
  --checkpoint-root checkpoints \
  --output docs/results/l2_sampling_osv_test_20261008.json \
  --init imagenet --fraction 1.0 --regime layer4 \
  --modes uniform cell-balanced --seeds 42 43 44 \
  --batch-size 32 --image-size 224 --num-workers 4 --device cuda \
  --region-lat-bands 6 --region-lon-bands 12 --region-min-count 20
```

The JSON report records the training manifest and metrics hashes, selected sample-ID hashes, cell and CSV hashes, checkpoint hashes, code fingerprints, runtime versions, hardware, per-seed results, and regional deltas. Checkpoints and image data are intentionally not committed. The run used Python 3.12.14, PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130, NumPy 2.5.2, and an NVIDIA GeForce RTX 3050 6GB Laptop GPU.

## Limitations and interpretation

- Three seeds provide a descriptive estimate, not a strong basis for statistical significance or broad superiority claims.
- The train subset samples only two training archives. It spans 166 countries, but archive representativeness and full-dataset behavior are not established.
- The model has 300 classes for only 10,000 training images and reaches high training cell accuracy. The much weaker held-out geolocation scores motivate testing more training data; they do not by themselves identify the cause.
- OSV-5M spatially separates train and test. Results should not be generalized to other image sources, time periods, or deployment settings.
- The data-scale follow-up uses a fresh, ID-disjoint row sample from the same official test split, rather than reusing these test rows. It is still exploratory because the follow-up question was chosen after inspecting these results; the new sample is not a new data source or independent geography.

KPlaceNet is PlaNet-inspired, not a faithful reproduction of PlaNet's adaptive S2 cells, Inception architecture, dataset scale, or reported benchmark results. See the [PlaNet paper](https://arxiv.org/abs/1602.05314).
