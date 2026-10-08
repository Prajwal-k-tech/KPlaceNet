# Training-set size comparison on OSV-5M

**Status:** Exploratory paired experiment. The follow-up question was selected
after inspecting the earlier sampling report. The 3,000 evaluation IDs are new
and disjoint from that earlier sample and from all training rows, but they come
from the same official OSV-5M test split; this is not an independent data
source or geography.

## Question and protocol

Does doubling the bounded training sample from 10,000 to 20,000 improve
geolocation when the classifier, geographic cells, and training budget stay
fixed? Three paired seeds (42, 43, 44) used ImageNet-initialized ResNet-50,
layer4 fine-tuning, 300 quadtree cells built once from the full 20,000-row
training pool, uniform shuffled sampling, AdamW (`lr=0.001`,
`weight_decay=0.0001`), batch size 16, 224-pixel inputs, AMP, and ten epochs.
For each seed, the 10k sample is a strict subset of its 20k sample. The final
epoch checkpoint was evaluated; test scores did not select checkpoints or
change training settings.

The training pool has 20,000 official train rows, 5,000 from each of archive
shards 00–03. The comparison used a second 3,000-row test sample drawn across
all five official test archives. It has 169 countries, unique identifiers,
zero overlap with the earlier test sample and training IDs, and all images
decoded. The source training CSV SHA-256 is
`3db0b1c9d6819beb4ee07f26895620c71b0cbf237e4df23b042c762cccb3d5fa`; the
evaluation CSV SHA-256 is
`92753639fa943c6f76ad2331651dabd593e80faf0714c3b66b450a3a2e5f2934`.
The [training-pool manifest](osv5m_train_pool_20k_manifest_20261008.json)
records the exact opaque row IDs and per-seed 10k source indices; it contains
no image paths or image data.
Exact evaluation IDs and selection details are in the
[held-out sample manifest](osv5m_test_confirmatory_sample_20261008.json).
Machine-readable per-run provenance is in the [10k report](l2_data_scale_osv_test_20261009_10k.json),
[20k report](l2_data_scale_osv_test_20261009_20k.json), and
[paired comparison](l2_data_scale_osv_test_20261009_comparison.json).

## Results

Metrics are mean ± sample standard deviation across three training seeds. The
paired column is 20k minus 10k; distance differences are in kilometers and
accuracy differences are percentage points.

| Metric | 10k | 20k | Paired change |
|---|---:|---:|---:|
| Within 1 km | 0.00 ± 0.00% | 0.00 ± 0.00% | 0.00 ± 0.00 pp |
| Within 25 km | 0.044 ± 0.019% | 0.067 ± 0.058% | +0.022 ± 0.069 pp |
| Within 200 km | 3.756 ± 0.158% | 5.300 ± 0.240% | +1.544 ± 0.383 pp |
| Mean geodesic distance | 5,977.3 ± 77.0 km | 5,521.6 ± 129.7 km | −455.7 ± 110.7 km |
| Median geodesic distance | 4,658.5 ± 292.6 km | 3,460.2 ± 165.1 km | −1,198.3 ± 275.2 km |

All three paired seeds improved in mean and median distance and within-200-km
accuracy. The most-frequent training-cell centroid baseline on these same test
rows has 9,713.3 km mean distance, 10,666.8 km median distance, and 0.067%
within 200 km. The 20k models are better than that baseline, but their absolute
geolocation errors remain large. Final training cell accuracy fell from
96.12–96.58% at 10k to 92.61–92.97% at 20k; this is a training diagnostic, not
a test result.

The consistent direction across three paired seeds is encouraging evidence
that the larger sample helped under this setup. It is not a significance test:
there are only three seeds and one inspected test sample, so the paired
standard deviations describe this run rather than uncertainty over OSV-5M.

## Limits and interpretation

- The 20k training pool covers only four of the 98 official train archives.
  The result does not establish performance on the full training corpus.
- OSV-5M spatially separates train and test. A random split within the sampled
  test rows does not establish exchangeability across places or geographic
  regions; the result does not imply regional or distribution-shift guarantees.
- The follow-up is exploratory because its question followed inspection of the
  earlier test report. ID disjointness prevents row reuse, but does not make
  this a preregistered confirmation or independent geography.
- KPlaceNet is PlaNet-inspired, not a reproduction of PlaNet's adaptive S2
  cells, Inception architecture, data scale, or benchmark results. See the
  [PlaNet paper](https://arxiv.org/abs/1602.05314) and
  [OSV-5M paper](https://arxiv.org/abs/2404.18873).

## Reproduction

After obtaining the same 20k train subset and the test sample described by the
committed ID manifest, run from the repository root:

```bash
python scripts/run_l2_experiments.py \
  --csv data/osv5m_subset_20k/metadata_20k.csv \
  --fractions 0.5 1.0 --inits imagenet --regimes layer4 \
  --sampling-modes uniform --seeds 42 43 44 --epochs 10 \
  --device cuda --num-workers 4 --run --skip-existing \
  --checkpoint-root checkpoints/data_scale_20k \
  --cells-json checkpoints/data_scale_20k/l2_cells_20k.json

python scripts/evaluate_sampling_comparison.py \
  --csv data/osv5m_confirmatory_test/metadata.csv \
  --sample-manifest docs/results/osv5m_test_confirmatory_sample_20261008.json \
  --checkpoint-root checkpoints/data_scale_20k \
  --output docs/results/l2_data_scale_osv_test_20261009_10k.json \
  --init imagenet --fraction 0.5 --regime layer4 \
  --modes uniform --seeds 42 43 44 --batch-size 32 \
  --image-size 224 --num-workers 4 --device cuda \
  --region-lat-bands 6 --region-lon-bands 12 --region-min-count 20

python scripts/evaluate_sampling_comparison.py \
  --csv data/osv5m_confirmatory_test/metadata.csv \
  --sample-manifest docs/results/osv5m_test_confirmatory_sample_20261008.json \
  --checkpoint-root checkpoints/data_scale_20k \
  --output docs/results/l2_data_scale_osv_test_20261009_20k.json \
  --init imagenet --fraction 1.0 --regime layer4 \
  --modes uniform --seeds 42 43 44 --batch-size 32 \
  --image-size 224 --num-workers 4 --device cuda \
  --region-lat-bands 6 --region-lon-bands 12 --region-min-count 20

python scripts/compare_data_scale_reports.py \
  docs/results/l2_data_scale_osv_test_20261009_10k.json \
  docs/results/l2_data_scale_osv_test_20261009_20k.json \
  --output docs/results/l2_data_scale_osv_test_20261009_comparison.json
```

The runner records each seed's sample IDs and source-dataset manifest beside
the local checkpoints. Checkpoints, images, and downloaded dataset CSVs are
not committed.
