# Fork work status

**Checked:** 2026-10-07 (Asia/Kolkata)  
**Fork:** [Prajwal-k-tech/KPlaceNet](https://github.com/Prajwal-k-tech/KPlaceNet)  
**Work branch:** `prajwal/eval-integrity`  
**Evaluator change:** `d3b3dd6` (`fix(eval): require trained checkpoint metadata`)

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

## Validation and limitations

- `python -m unittest discover -s tests -p 'test_eval_contract.py' -v`: 5 tests
  passed.
- `python -m py_compile src/eval.py src/eval_contract.py tests/test_eval_contract.py`:
  passed.
- `git diff --check`: passed.
- Full test discovery ran 12 tests: 8 passed; the L3 metrics, reproducibility,
  and uncertainty suites could not import because NumPy and `typing_extensions`
  are absent in the current Python 3.14 environment. No dependencies were
  installed for this check.
- No training or real-data evaluation was run. The required OSV-5M subset,
  evaluation images, and trained checkpoint are absent. No real-data accuracy,
  calibration, or regional result is claimed for this change.

## Next steps

1. Keep this evaluation guard on the fork branch and validate it in a supported
   project environment with the declared ML dependencies.
2. Inspect the dataset downloader's plan and required storage before fetching
   any OSV-5M shards. Then run the recorded baseline with its exact checkpoint
   and split, if the data and compute requirements are practical.
3. Choose any further modeling or calibration work only after that baseline is
   verified. Report real-data results separately from synthetic tests and
   preserve the geographic-shift limitations of split conformal prediction.

No upstream pull request is being opened or updated as part of this work.
