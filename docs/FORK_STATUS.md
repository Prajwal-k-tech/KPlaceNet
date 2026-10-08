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

The repository has L0-L4 code and historical result reports. That makes it a
substantial prototype, but its reported experiments are not reproducible from
this checkout alone: `data/` contains only tracked placeholders/documentation,
and there is no `checkpoints/` directory. The reports are recorded results;
they have not been independently rerun for this fork contribution.

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
retain source columns.

## Validation and limitations

- Full `python -m unittest discover -s tests -v` in an isolated Python 3.12
  CPU environment: all 48 tests passed, including the synthetic pipeline and
  stale-cache tests.
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
- The L2 cache-integrity changes passed locally (48 tests) and in GitHub Actions on `b2e94f3`: [run 37748946520](https://github.com/Prajwal-k-tech/KPlaceNet/actions/runs/37748946520), including both workflow jobs.
- No real-data evaluation was run. `data/` contains no required OSV-5M test
  subset or evaluation images, and the repository contains no trained checkpoint.
  No real-data accuracy, calibration, or regional result is claimed.

## Next steps

1. Before running a real experiment, confirm dataset and checkpoint download
   sizes and available storage. Then run paired uniform and cell-balanced
   experiments on a fixed training subset and untouched spatially separated
   test set; repeat seeds before making any performance claim.
2. Package a verified checkpoint and matching calibration artifact for the
   Gradio demo only after real-data inference is checked against the evaluator.
