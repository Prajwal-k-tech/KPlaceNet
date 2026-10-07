# Fork work status

**Checked:** 2026-10-07 (Asia/Kolkata)  
**Fork:** [Prajwal-k-tech/KPlaceNet](https://github.com/Prajwal-k-tech/KPlaceNet)  
**Work branch:** `prajwal/eval-integrity`  
**Latest implementation:** `21a9cbe` (`feat: report paired geolocation sampling results`)
**Work:** evaluation-contract guard, cell-balanced sampling, and paired L2 runner mode

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
The Gradio demo now shares the evaluator's checkpoint contract and loads model
weights strictly, so a missing or incompatible checkpoint cannot silently
produce demo predictions.

## Validation and limitations

- `python -m unittest discover -s tests -p 'test_eval_contract.py' -v`: 5 tests
  passed.
- `python -m py_compile src/eval.py src/eval_contract.py tests/test_eval_contract.py`:
  passed.
- `git diff --check`: passed.
- `python -m unittest discover -s tests -p 'test_sampling.py' -v`: 3
  dependency-free tests passed for uniform, softened, and fully balanced
  inverse-frequency weights plus invalid inputs.
- The paired-runner tests also pass for unchanged uniform paths, distinct
  balanced paths, command arguments, and seed-isolated checkpoints.
- The comparison-report tests pass for JSON metric parsing, mode summaries,
  paired deltas, and unpaired seeds.
- Python byte-compilation passed for the sampler, dataset, trainer, L2 runner,
  comparison evaluator, demo, and new tests.
- Full test discovery enumerated 21 tests: 18 passed; three modules could not
  import because NumPy and `typing_extensions` are absent in the current Python
  3.14 environment. A temporary install attempt selected large CUDA packages,
  so it was canceled before installation and its temporary environment removed.
- No training or real-data evaluation was run. The required OSV-5M subset,
  evaluation images, and trained checkpoint are absent. No real-data accuracy,
  calibration, or regional result is claimed for this change. PyTorch import
  fails because `typing_extensions` is missing; torchvision, NumPy, and pandas
  are also unavailable, so the PyTorch sampler and training path could not be
  executed here.

## Next steps

1. In a supported project environment, install the declared ML dependencies
   and run the full test suite plus a tiny generated-image training smoke test.
2. Inspect the downloader plan and storage before fetching data. Run paired
   uniform and cell-balanced experiments on the same fixed train subset and
   untouched spatially separated test set; repeat seeds before claiming a
   performance change.
3. Package the verified checkpoint and calibration artifact for the Gradio
   demo, record hashes and setup instructions, and capture a short demo only
   after inference is checked against the evaluator.

No upstream pull request is being opened or updated as part of this work.
