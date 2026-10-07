"""Evaluate uniform and cell-balanced checkpoints on one fixed test CSV.

This script does not train models. It runs ``src.eval`` for every selected
seed/mode pair, captures its metrics, and writes a provenance-bearing JSON
report with paired per-seed deltas.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_l2_experiments import checkpoint_dir_name

METRIC_KEYS = ("within_1km", "within_25km", "within_200km", "mean_km", "median_km")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_eval_metrics(stdout: str) -> dict[str, float | int]:
    """Extract and validate the evaluator's machine-readable final metrics."""
    for line in reversed(stdout.splitlines()):
        if line.startswith("[json]"):
            metrics = json.loads(line[len("[json]"):])
            if not isinstance(metrics, dict) or not isinstance(metrics.get("n"), int) or metrics["n"] < 1:
                raise ValueError("evaluator returned invalid or empty metrics")
            for key in METRIC_KEYS:
                value = metrics.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"evaluator returned invalid metric {key!r}")
            return metrics
    raise ValueError("evaluator output did not contain a [json] metrics record")


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-mode metrics and paired balanced-minus-uniform deltas."""
    modes = sorted({run["sampling_mode"] for run in runs})
    summary: dict[str, Any] = {"by_mode": {}, "paired_delta_cell_balanced_minus_uniform": {}}
    for mode in modes:
        selected = [run for run in runs if run["sampling_mode"] == mode]
        summary["by_mode"][mode] = {
            "runs": len(selected),
            "metrics": {
                key: {
                    "mean": statistics.mean(run["metrics"][key] for run in selected),
                    "stdev": statistics.stdev(run["metrics"][key] for run in selected)
                    if len(selected) > 1 else 0.0,
                }
                for key in METRIC_KEYS
            },
        }

    uniform = {run["seed"]: run for run in runs if run["sampling_mode"] == "uniform"}
    balanced = {run["seed"]: run for run in runs if run["sampling_mode"] == "cell-balanced"}
    for key in METRIC_KEYS:
        deltas = [
            balanced[seed]["metrics"][key] - uniform[seed]["metrics"][key]
            for seed in sorted(uniform.keys() & balanced.keys())
        ]
        summary["paired_delta_cell_balanced_minus_uniform"][key] = {
            "per_seed": deltas,
            "mean": statistics.mean(deltas) if deltas else None,
            "stdev": statistics.stdev(deltas) if len(deltas) > 1 else (0.0 if deltas else None),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data/osv5m_test/metadata.csv",
                        help="Fixed, untouched evaluation CSV")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/sampling_comparison.json")
    parser.add_argument("--init", choices=["imagenet", "places365"], default="imagenet")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--regime", choices=["frozen", "layer4"], default="frozen")
    parser.add_argument("--modes", nargs="+", choices=["uniform", "cell-balanced"],
                        default=["uniform", "cell-balanced"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    args = parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"Evaluation CSV not found: {args.csv}")
    if args.output.resolve() == args.csv.resolve():
        raise SystemExit("--output must not overwrite the evaluation CSV")
    if "uniform" not in args.modes:
        raise SystemExit("--modes must include uniform as the comparison baseline")
    if len(set(args.modes)) != len(args.modes) or len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("--modes and --seeds must not contain duplicates")
    if not 0.0 < args.fraction <= 1.0:
        raise SystemExit("--fraction must be in (0, 1]")

    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        for mode in args.modes:
            name = checkpoint_dir_name(args.init, args.fraction, args.regime, mode, seed)
            checkpoint = args.checkpoint_root / name / "last.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint for seed={seed}, mode={mode}: {checkpoint}")
            command = [
                sys.executable, "-m", "src.eval", "--csv", str(args.csv),
                "--checkpoint", str(checkpoint), "--batch-size", str(args.batch_size),
                "--image-size", str(args.image_size), "--num-workers", str(args.num_workers),
                "--device", args.device,
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            if result.returncode:
                raise RuntimeError(
                    f"Evaluation failed for seed={seed}, mode={mode} (exit {result.returncode}):\n"
                    f"{result.stderr[-3000:]}"
                )
            runs.append({
                "seed": seed,
                "sampling_mode": mode,
                "checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "metrics": parse_eval_metrics(result.stdout),
            })

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    report = {
        "schema_version": 1,
        "comparison": {
            "init": args.init,
            "fraction": args.fraction,
            "regime": args.regime,
            "modes": args.modes,
            "seeds": args.seeds,
        },
        "evaluation_csv": str(args.csv),
        "evaluation_csv_sha256": sha256_file(args.csv),
        "evaluator_revision": revision,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
            "numpy": package_version("numpy"),
        },
        "runs": runs,
        "summary": summarize_runs(runs),
        "interpretation": (
            "Descriptive paired evaluation on the specified test CSV. It does not imply "
            "causal superiority or generalization beyond this dataset."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved comparison report: {args.output}")
    print(json.dumps(report["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
