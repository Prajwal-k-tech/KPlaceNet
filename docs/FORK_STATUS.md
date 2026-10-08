# Fork work status

**Checked:** 2026-10-08 (Asia/Kolkata)
**Fork:** [Prajwal-k-tech/KPlaceNet](https://github.com/Prajwal-k-tech/KPlaceNet)  
**Work branch:** `prajwal/eval-integrity`  
**Work:** evaluation-integrity guard, cell-balanced sampling, paired L2 evaluation,
CPU-tested synthetic train/evaluate smoke, and per-run training provenance

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
OSV-5M training and held-out evaluation is now running locally; see the
validation section for its status.

## Fork change in progress

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
- A verified local data subset contains 10,000 official OSV-5M train images and
  3,000 sampled official test images. Sample IDs are unique within each split,
  there is no train/test ID overlap, every image decodes, and the official
  train/test split labels are preserved. The source metadata hashes are recorded
  with the experiment artifacts; image data and checkpoints are not committed.
- A paired three-seed comparison of uniform and inverse-cell-frequency
  cell-balanced sampling is in progress (ImageNet initialization, ResNet-50
  layer4 fine-tuning, 10 epochs, 300 fixed geographic cells). Four of six runs
  have completed; the fifth is in progress. Training accuracy differs between
  samplers but is only diagnostic. No real-data evaluation result or performance
  claim is available until all six runs finish and are evaluated on the
  untouched test subset.

## Next steps

1. Finish the six-run matrix and evaluate all checkpoints on the fixed,
   untouched official test subset; report paired seed deltas and limitations.
2. Package a verified checkpoint and matching calibration artifact for the
   Gradio demo only after real-data inference is checked against the evaluator.
