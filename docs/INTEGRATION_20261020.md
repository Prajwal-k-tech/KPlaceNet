# Contribution integration on 20 October 2026

This integration combines Asterisk-Hunter main `bab61f9` with Prajwal Kumar K's complete contribution branch `c02d320`. Both parent histories are retained. It preserves Kerala L5, the Plotly demo and GeoGuessr alongside conformal correctness, three-way fitting splits, provenance, strict evaluation, sampling comparisons, and the recorded 10k versus 20k follow-up.

README and demo conflicts were resolved to retain both modes. The global demo retains configurable checkpoint and calibration paths, verifies a supplied calibration checkpoint hash, and labels missing calibration as raw confidence. `KPLACENET_CELLS` can override the global sidecar. Sidecar bounds are applied only to matching centroids. Kerala loading now validates class metadata, matches sidecar centroids to the checkpoint, and loads weights strictly. Kerala still has no fitted calibration and reports raw confidence.

The cross-platform report path formatter now recognizes foreign Windows and POSIX absolute paths. A regression assertion covers Windows paths, and three demo integration tests cover global raw fallback, verified global calibration, and uncalibrated Kerala output. Gradio and Plotly are included in requirements because both are needed by the demos.

## Validation

- Full unit test discovery: 73 tests run, 72 passed, one CUDA-only test skipped on CPU. Includes a synthetic image train-checkpoint-evaluate smoke test.
- L2 plan-only and L3 plan-only commands pass without image data or model downloads.
- Python byte compilation passes for src, scripts, demo, and tests.
- Gradio UI builds with 22 blocks without launching a server. Gradio 6 reports a theme/CSS constructor deprecation warning; model inference was not tested without the original artifacts.
- Synthetic conformal diagnostic: 20,000 trials, seed 2026, n=9, alpha=.01. Correct rank 10 returns infinity and coverage 1.0000; historical clamped maximum gives .9037. This is a mathematical software diagnostic, not geolocation evidence.

These CPU checks use Python 3.12 and PyTorch 2.14.1+cpu. They do not rerun or independently reproduce the dated CUDA image experiments. Images and trained checkpoints remain local, gitignored prerequisites.

The classroom report and viva guide are in `docs/classroom/`. The report preserves separate sample identities and attribution for historical global, controlled fork, and Kerala studies. Title page institution, instructor, department, and roll numbers still need confirmation.

## Demo preparation follow-up

The fork main and both contribution branches were published after Git Credential Manager browser authentication. Direct upstream push is denied to Prajwal-k-tech; PR 1 is updated for maintainer merge. The upload UI was opened in a browser, mode switching and research details inspected, and the Gradio 6 launcher now preserves theme and CSS. `scripts/check_demo_ready.py` reports missing model and data files before rehearsal. The seven-page live demo playbook documents controls, inference, design rationale, an exact artifact transfer list, and a timed script.

Full follow-up validation: 74 tests run, 73 passed, one CUDA-only skip. No original checkpoints or demonstration images were found in the Windows project locations checked or Ubuntu home. File readiness is not verified inference; real-image demo verification remains pending those artifacts.
