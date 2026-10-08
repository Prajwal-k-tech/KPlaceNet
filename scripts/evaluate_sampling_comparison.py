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


def portable_path(path: Path | str) -> str:
    """Render paths relative to the repository without exposing host paths."""
    value = Path(path)
    if not value.is_absolute():
        return value.as_posix()
    try:
        return value.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return value.name


def load_training_provenance(checkpoint: Path) -> dict[str, Any]:
    """Load and verify the manifest paired with a training checkpoint."""
    manifest_path = checkpoint.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing training manifest for checkpoint: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid training manifest: {manifest_path}")
    declared_hash = manifest.get("manifest_sha256")
    unsigned_manifest = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    canonical = json.dumps(unsigned_manifest, sort_keys=True, separators=(",", ":"))
    if not isinstance(declared_hash, str) or hashlib.sha256(canonical.encode("utf-8")).hexdigest() != declared_hash:
        raise ValueError(f"Training manifest hash mismatch: {manifest_path}")
    required = ("dataset_csv_sha256", "dataset_sample_count", "selected_sample_ids",
                "cells_sha256", "seed", "config", "initialization", "software")
    if any(key not in manifest for key in required):
        raise ValueError(f"Training manifest is missing required provenance: {manifest_path}")
    config = manifest["config"]
    if not isinstance(config, dict) or not isinstance(manifest["initialization"], dict) or not isinstance(manifest["software"], dict):
        raise ValueError(f"Training manifest has invalid configuration metadata: {manifest_path}")
    if (
        not isinstance(manifest["dataset_csv_sha256"], str)
        or not isinstance(manifest["cells_sha256"], str)
        or isinstance(manifest["dataset_sample_count"], bool)
        or not isinstance(manifest["dataset_sample_count"], int)
        or manifest["dataset_sample_count"] < 1
        or isinstance(manifest["seed"], bool)
        or not isinstance(manifest["seed"], int)
        or manifest["seed"] < 0
    ):
        raise ValueError(f"Training manifest has invalid identity metadata: {manifest_path}")
    sample_ids = manifest["selected_sample_ids"]
    if not isinstance(sample_ids, list) or not all(isinstance(value, str) for value in sample_ids):
        raise ValueError(f"Training manifest has invalid selected sample IDs: {manifest_path}")
    if not sample_ids or len(sample_ids) > manifest["dataset_sample_count"]:
        raise ValueError(f"Training manifest has an invalid selected sample count: {manifest_path}")
    sample_ids_payload = json.dumps(sample_ids, separators=(",", ":"))
    return {
        "manifest_path": portable_path(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_sha256": declared_hash,
        "training_csv_sha256": manifest["dataset_csv_sha256"],
        "training_dataset_sample_count": manifest["dataset_sample_count"],
        "selected_sample_count": len(sample_ids),
        "selected_sample_ids_sha256": hashlib.sha256(sample_ids_payload.encode("utf-8")).hexdigest(),
        "cells_sha256": manifest["cells_sha256"],
        "seed": manifest["seed"],
        "config": {
            key: portable_path(value) if key in {"csv", "checkpoint_dir", "cells_json", "places365_checkpoint"}
            and value is not None else value
            for key, value in manifest["config"].items()
        },
        "initialization": manifest["initialization"],
        "software": manifest["software"],
    }


def validate_training_comparison(runs: list[dict[str, Any]]) -> None:
    """Reject comparisons whose checkpoints do not share a controlled setup."""
    if not runs:
        raise ValueError("comparison must contain at least one training run")
    variable_config = {"csv", "checkpoint_dir", "run_tag", "seed", "sampling_mode", "sampling_power"}
    reference = runs[0]["training"]
    reference_config = {
        key: value for key, value in reference["config"].items() if key not in variable_config
    }
    invariant_fields = (
        "training_csv_sha256",
        "training_dataset_sample_count",
        "cells_sha256",
        "initialization",
        "software",
    )
    for run in runs[1:]:
        training = run["training"]
        if any(training[field] != reference[field] for field in invariant_fields):
            raise ValueError("training runs do not share the same data, cells, initialization, or software")
        config = {key: value for key, value in training["config"].items() if key not in variable_config}
        if config != reference_config:
            raise ValueError("training runs do not share the same model and optimization configuration")

    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    balanced_powers: set[Any] = set()
    for run in runs:
        by_seed.setdefault(run["seed"], {})[run["sampling_mode"]] = run["training"]
        if run["sampling_mode"] == "cell-balanced":
            balanced_powers.add(run["training"]["config"].get("sampling_power"))
    if len(balanced_powers) > 1:
        raise ValueError("cell-balanced runs do not share the same sampling power")
    for seed, modes in by_seed.items():
        uniform = modes.get("uniform")
        balanced = modes.get("cell-balanced")
        if uniform is None or balanced is None:
            continue
        if (
            uniform["selected_sample_ids_sha256"] != balanced["selected_sample_ids_sha256"]
            or uniform["selected_sample_count"] != balanced["selected_sample_count"]
        ):
            raise ValueError(f"paired sampling runs for seed={seed} used different training samples")


def parse_eval_metrics(stdout: str) -> dict[str, Any]:
    """Extract and validate the evaluator's machine-readable final metrics."""
    for line in reversed(stdout.splitlines()):
        if line.startswith("[json]"):
            metrics = json.loads(line[len("[json]"):])
            if (not isinstance(metrics, dict) or isinstance(metrics.get("n"), bool)
                    or not isinstance(metrics.get("n"), int) or metrics["n"] < 1):
                raise ValueError("evaluator returned invalid or empty metrics")
            for key in METRIC_KEYS:
                value = metrics.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"evaluator returned invalid metric {key!r}")
            regional = metrics.get("regional_distance")
            if not isinstance(regional, dict) or not isinstance(regional.get("regions"), dict):
                raise ValueError("evaluator returned no geographic-strata metrics")
            return metrics
    raise ValueError("evaluator output did not contain a [json] metrics record")


def parse_checkpoint_manifest_hash(stdout: str) -> str:
    """Read the training-manifest hash embedded inside an evaluated checkpoint."""
    for line in stdout.splitlines():
        if line.startswith("[run-manifest]"):
            value = line[len("[run-manifest]"):].strip()
            if len(value) == 64 and all(char in "0123456789abcdef" for char in value):
                return value
            raise ValueError("evaluator returned an invalid checkpoint training-manifest hash")
    raise ValueError("evaluator output did not include the checkpoint training-manifest hash")


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
    region_keys = sorted({
        region
        for run in runs
        for region in run.get("metrics", {}).get("regional_distance", {}).get("regions", {})
    })
    regional_delta: dict[str, Any] = {}
    for region in region_keys:
        per_metric: dict[str, Any] = {}
        for key in METRIC_KEYS:
            deltas = []
            sample_counts = set()
            for seed in sorted(uniform.keys() & balanced.keys()):
                baseline = uniform[seed]["metrics"].get("regional_distance", {}).get("regions", {}).get(region, {})
                treatment = balanced[seed]["metrics"].get("regional_distance", {}).get("regions", {}).get(region, {})
                baseline_metrics = baseline.get("metrics")
                treatment_metrics = treatment.get("metrics")
                if baseline_metrics is None or treatment_metrics is None:
                    continue
                if key not in baseline_metrics or key not in treatment_metrics:
                    continue
                deltas.append(treatment_metrics[key] - baseline_metrics[key])
                sample_counts.add(baseline["n"])
            per_metric[key] = {
                "per_seed": deltas,
                "mean": statistics.mean(deltas) if deltas else None,
                "stdev": statistics.stdev(deltas) if len(deltas) > 1 else (0.0 if deltas else None),
                "n": next(iter(sample_counts)) if len(sample_counts) == 1 else None,
            }
        regional_delta[region] = per_metric
    summary["regional_paired_delta_cell_balanced_minus_uniform"] = regional_delta
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
    parser.add_argument("--region-lat-bands", type=int, default=6)
    parser.add_argument("--region-lon-bands", type=int, default=12)
    parser.add_argument("--region-min-count", type=int, default=20)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def cuda_device_name() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


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

    selected_runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        for mode in args.modes:
            name = checkpoint_dir_name(args.init, args.fraction, args.regime, mode, seed)
            checkpoint = args.checkpoint_root / name / "last.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint for seed={seed}, mode={mode}: {checkpoint}")
            training = load_training_provenance(checkpoint)
            if training["seed"] != seed or training["config"].get("seed") != seed:
                raise ValueError(f"Training manifest seed does not match requested seed={seed}: {checkpoint}")
            if training["config"].get("sampling_mode") != mode:
                raise ValueError(f"Training manifest sampling mode does not match {mode}: {checkpoint}")
            expected_initialization = "torchvision_imagenet" if args.init == "imagenet" else "places365"
            if training["initialization"].get("kind") != expected_initialization:
                raise ValueError(f"Training manifest initialization does not match {args.init}: {checkpoint}")
            selected_runs.append({
                "seed": seed,
                "sampling_mode": mode,
                "checkpoint_path": checkpoint,
                "checkpoint": portable_path(checkpoint),
                "training": training,
            })

    validate_training_comparison(selected_runs)

    runs: list[dict[str, Any]] = []
    for selected in selected_runs:
        seed = selected["seed"]
        mode = selected["sampling_mode"]
        checkpoint = selected["checkpoint_path"]
        training = selected["training"]
        assert isinstance(checkpoint, Path)
        assert isinstance(training, dict)
        command = [
            sys.executable, "-m", "src.eval", "--csv", str(args.csv),
            "--checkpoint", str(checkpoint), "--batch-size", str(args.batch_size),
            "--image-size", str(args.image_size), "--num-workers", str(args.num_workers),
            "--region-lat-bands", str(args.region_lat_bands),
            "--region-lon-bands", str(args.region_lon_bands),
            "--region-min-count", str(args.region_min_count),
            "--device", args.device,
        ]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(
                f"Evaluation failed for seed={seed}, mode={mode} (exit {result.returncode}):\n"
                f"{result.stderr[-3000:]}"
            )
        checkpoint_manifest_hash = parse_checkpoint_manifest_hash(result.stdout)
        if checkpoint_manifest_hash != training["manifest_sha256"]:
            raise ValueError(
                f"Checkpoint and training manifest disagree for seed={seed}, mode={mode}: {checkpoint}"
            )
        runs.append({
            "seed": seed,
            "sampling_mode": mode,
            "checkpoint": selected["checkpoint"],
            "checkpoint_sha256": sha256_file(checkpoint),
            "training": training,
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
            "checkpoint_policy": "last.pt from the final configured epoch; no test-set selection",
        },
        "evaluation": {
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "num_workers": args.num_workers,
            "device": args.device,
            "region_lat_bands": args.region_lat_bands,
            "region_lon_bands": args.region_lon_bands,
            "region_min_count": args.region_min_count,
        },
        "evaluation_csv": portable_path(args.csv),
        "evaluation_csv_sha256": sha256_file(args.csv),
        "evaluator_revision": revision,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
            "numpy": package_version("numpy"),
            "cuda_device_name": cuda_device_name(),
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
