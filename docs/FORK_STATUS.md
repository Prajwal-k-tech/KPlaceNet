# Fork work status

**Checked:** 2026-10-09 (Asia/Kolkata)
**Fork:** [Prajwal-k-tech/KPlaceNet](https://github.com/Prajwal-k-tech/KPlaceNet)  
**Work branch:** `prajwal/eval-integrity`  
**Work:** evaluation integrity and provenance, controlled OSV-5M data-scale and
sampling studies, split-conformal calibration evaluation, and a runnable demo

## What this project implements

KPlaceNet is a small, PlaNet-inspired image geolocation project. It has code
for a ResNet-50 classifier over roughly 300 geographic cells, training and
evaluation scripts, data-efficiency and cell-partition experiment runners,
and post-hoc uncertainty tools. This differs materially from the original
[PlaNet paper](https://arxiv.org/abs/1602.05314), which uses adaptive S2 cells,
an Inception model, and a 126-million-photo corpus. KPlaceNet is not a faithful
reproduction of PlaNet's model, scale, or reported benchmark results.

The repository has L0-L4 code and historical result reports. At the start of
this fork work, the data and checkpoints needed to reproduce them were absent.
Those historical results are not results of this contribution. A bounded
10k comparison of uniform and cell-balanced sampling and a paired
10k-versus-20k follow-up are complete. Their code, sample manifests, and
evaluation artifacts are recorded under `docs/results/`. The measured results
are limited to one held-out OSV-5M test sample and are reported as exploratory.

## Fork contributions

The standard evaluator previously allowed three invalid evaluation paths: it
could run with random weights when the checkpoint was absent, reconstruct
class centroids from evaluation coordinates, or partially load mismatched
weights and clip predictions. It now requires a checkpoint with model weights,
contiguous cell IDs, valid centroids, and matching classifier dimensions; it
loads the model state strictly. This prevents evaluation labels from defining
the model's output classes and prevents invalid scores from being reported as
model results.

Cell-balanced training is an explicit optional mode. It uses seeded
inverse-frequency weights over the active training subset, leaves the default
uniform baseline unchanged, and records the mode in checkpoint arguments and
metrics. The L2 runner can plan or execute paired uniform/cell-balanced runs
across explicit seeds, with separate checkpoint directories for non-default
seeds. A companion evaluator runs every selected checkpoint against one fixed
test CSV, then saves sample metrics, per-seed paired deltas, and dataset and
checkpoint hashes in a JSON report.
The Gradio demo now shares the evaluator's checkpoint contract, loads model
weights strictly, and rejects a calibration artifact whose recorded checkpoint
hash differs from the selected model. Checkpoint and calibration paths are
configurable; missing calibration is labeled as raw softmax/top-1 mode rather
than silently implying a calibrated prediction.
The fork runs dependency-free contract/reporting checks and a separate CPU
model-test job in GitHub Actions. The CPU job runs the full unittest suite,
including a generated-image training/evaluation smoke; it downloads no dataset
or pretrained model.
Training writes a `run_manifest.json` beside each checkpoint and metrics file.
It records the source CSV hash, selected row positions and opaque sample IDs,
cell-definition hash, initialization source (and Places365 checkpoint hash),
seed, training arguments, and runtime versions. The checkpoint and metrics
record the manifest hash for cross-artifact verification.
The L2 runner validates generated fraction-CSV and cell-definition caches
against the source CSV hash, settings, generator source hash, and artifact hash
before reusing them; stale or modified caches are regenerated. Fraction CSVs
retain source columns. The experiment runner supports loader-worker tuning and
has a regression test for running its plan from a clean checkout. The paired
evaluator verifies each checkpoint's training-manifest content hash and checks
that the manifest seed and sampling mode match the requested comparison. It
records the manifest/configuration, training CSV hash, selected sample-ID hash,
fixed test CSV hash, checkpoint hash, and paired seed deltas. Evaluation also
reports descriptive geodesic error by equal-area region, keeps sparse bins
visible without metrics, and summarizes paired regional deltas.
Temperature fitting now moves the trainable temperature parameter along with
calibration logits to the selected device, so CUDA-backed fitting does not mix
CPU parameters with GPU tensors. A CUDA-conditional regression test checks
both fitting and scaled-logit placement; this checkout has CPU-only PyTorch, so
the test is skipped locally. The configured CI also uses CPU-only PyTorch, so
CUDA execution remains unverified.

## Validation and limitations

- Full `python -m unittest discover -s tests -v` in an isolated Python 3.12
  CPU environment: 55 tests ran, 54 passed, and the CUDA-only device-placement
  test was skipped. This includes the synthetic pipeline, stale-cache,
  provenance-integrity, and geographic-strata tests.
- The synthetic integration test generated 8 training and 4 evaluation images,
  trained one CPU epoch with random initialization, loaded the saved checkpoint,
  and emitted valid evaluator JSON for all 4 held-out synthetic images. Its
  accuracy and distance values are not geolocation evidence.
- `python -m py_compile tests/test_synthetic_pipeline.py` and `git diff --check`
  passed.
- The isolated test environment used CPU PyTorch 2.14.1 and torchvision 0.29.1;
  its temporary training artifacts were removed automatically. No dataset or
  pretrained weights were downloaded.
- GitHub Actions passed on commit `a692d88`: [run 37746197271](https://github.com/Prajwal-k-tech/KPlaceNet/actions/runs/37746197271), including both the standard-library and CPU model-test jobs.
- GitHub Actions passed on commit `ae25b44`: [run 37782064966](https://github.com/Prajwal-k-tech/KPlaceNet/actions/runs/37782064966). The full suite ran 49 tests, with 48 passing and the CUDA-only test skipped.
- GitHub Actions passed on commits `5a52890`, `9b7d775`, and `b99fae5`: [run 37790574684](https://github.com/Prajwal-k-tech/KPlaceNet/actions/runs/37790574684), [run 37790980822](https://github.com/Prajwal-k-tech/KPlaceNet/actions/runs/37790980822), and [run 37791670958](https://github.com/Prajwal-k-tech/KPlaceNet/actions/runs/37791670958). The latest CPU suite ran 55 tests, with 54 passing and the CUDA-only test skipped.
- The L2 cache-integrity changes passed locally (48 tests) and in GitHub Actions on `bbb1362`: [run 37749946875](https://github.com/Prajwal-k-tech/KPlaceNet/actions/runs/37749946875), including both workflow jobs.
- The initial real-data L2 experiment used 10,000 train images from shards 00–01
  and a fixed 3,000-row OSV-5M test sample. Across three seeds, uniform sampling
  reached 5,849.9 km mean geodesic error (sample SD 183.9 km); cell-balanced
  reached 5,859.0 km (SD 143.5 km). The paired mean-distance difference was
  +9.1 km (balanced worse; paired SD 222.7 km), with no mean change in within-
  200-km accuracy. The model improves over the most-frequent-cell baseline but
  its absolute localization performance remains weak; the sampler showed no
  consistent gain. Since this test sample has been inspected, those results and
  the data-scale follow-up on it are exploratory.
- The paired data-scale study completed six uniform-sampling runs: 10k vs 20k
  train rows, one shared 300-cell map, seeds 42/43/44, and ten epochs. On the
  fresh 3,000-row test sample, mean geodesic distance was 5,977.3 km (seed SD
  77.0) at 10k and 5,521.6 km (SD 129.7) at 20k. The paired 20k-minus-10k
  change was −455.7 km (SD 110.7); within-200-km accuracy increased from
  3.756% to 5.300%, a paired +1.544 percentage points (SD 0.383). All three
  seeds improved both metrics. This is exploratory, not statistically
  conclusive; absolute error remains high and the pool covers only four of 98
  train archives. See `docs/results/l2_data_scale_osv_test_20261009.md` and
  its per-run JSON reports.
- The L3 analysis used the seed-42 20k checkpoint with disjoint temperature-fit
  (500), conformal-calibration (1,000), and evaluation (1,500) partitions from
  that fresh sample. ECE fell from 0.4171 to 0.0453 after temperature scaling;
  the empirical 90% conformal target had 90.8% evaluation coverage but mean
  set size 159.7 of 300 cells. Thresholds 0.5 and 0.7 abstained on every row;
  threshold 0.3 retained only 12 rows. This improves confidence calibration,
  not geolocation accuracy, and does not establish coverage under geographic
  shift. Details and the exact splits are in the L3 JSON/Markdown report.
- The test sample IDs, metadata hash, split, and replacement sampling details
  are in `docs/results/osv5m_test_confirmatory_sample_20261008.json`. The
  training pool's opaque IDs and each 10k subset's source indices are in
  `docs/results/osv5m_train_pool_20k_manifest_20261008.json`; both manifests
  contain identifiers only, not image data or paths.
- The Gradio prediction function loaded the seed-42 20k checkpoint on CPU and
  rendered a finite-coordinate prediction for a synthetic image. This confirms
  the inference/rendering path only; it is not geolocation evidence.
- Source-manifest validation now binds each generated fraction CSV to its
  shared source dataset and recognizes that per-seed CSV hashes differ. A real
  CLI run caught and fixed this assumption; the data-scale comparator also now
  works when invoked as a script. Twenty focused evaluation/comparator tests
  pass.

## Next steps

1. Re-run the fixed L3 calibration evaluation after the L2 reports and docs are
   committed, so its provenance snapshot is clean and verifiable.
2. Run the complete local unittest suite, validate the demo with the generated
   calibration artifact, update final CI/status receipts, and push the reports
   directly to the fork branch.
3. Remove only the task-specific virtualenv and package cache after validation;
   retain the bounded dataset and checkpoints needed to reproduce and demo the
   contribution.
