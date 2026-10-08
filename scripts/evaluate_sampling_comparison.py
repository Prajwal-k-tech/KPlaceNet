"""Evaluate uniform and cell-balanced checkpoints on one fixed test CSV.

This script does not train models. It runs ``src.eval`` for every selected
seed/mode pair, captures its metrics, and writes a provenance-bearing JSON
report with paired per-seed deltas.
"""

from __future__ import annotations

import argparse
import csv
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
    if (
        isinstance(manifest.get("cell_count"), bool)
        or not isinstance(manifest.get("cell_count"), int)
        or manifest["cell_count"] < 1
        or not isinstance(manifest.get("cells_sha256"), str)
    ):
        raise ValueError(f"Training manifest has invalid cell metadata: {manifest_path}")
    if not isinstance(sample_ids, list) or not all(isinstance(value, str) for value in sample_ids):
        raise ValueError(f"Training manifest has invalid selected sample IDs: {manifest_path}")
    if not sample_ids or len(sample_ids) > manifest["dataset_sample_count"]:
        raise ValueError(f"Training manifest has an invalid selected sample count: {manifest_path}")
    sample_ids_payload = json.dumps(sample_ids, separators=(",", ":"))
    training_csv = Path(config["csv"]).expanduser() if isinstance(config.get("csv"), str) else None
    if training_csv is not None and not training_csv.is_absolute():
        training_csv = ROOT / training_csv
    source_csv_sha256 = manifest["dataset_csv_sha256"]
    source_sample_count = manifest["dataset_sample_count"]
    source_manifest_info: dict[str, Any] | None = None
    if training_csv is not None and training_csv.is_file():
        if sha256_file(training_csv) != manifest["dataset_csv_sha256"]:
            raise ValueError(f"Training CSV hash mismatch: {training_csv}")
        cache_manifest_path = training_csv.with_name(training_csv.name + ".manifest.json")
        if cache_manifest_path.is_file():
            cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
            indices = cache_manifest.get("selected_source_indices")
            fraction = cache_manifest.get("fraction")
            source_count = cache_manifest.get("source_row_count")
            if (
                cache_manifest.get("artifact_sha256") != manifest["dataset_csv_sha256"]
                or cache_manifest.get("row_count") != manifest["dataset_sample_count"]
                or cache_manifest.get("seed") != manifest["seed"]
                or isinstance(fraction, bool)
                or not isinstance(fraction, (int, float))
                or not math.isfinite(fraction)
                or not 0 < fraction <= 1
                or isinstance(source_count, bool)
                or not isinstance(source_count, int)
                or source_count < manifest["dataset_sample_count"]
                or not isinstance(indices, list)
                or len(indices) != manifest["dataset_sample_count"]
                or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
                or len(indices) != len(set(indices))
                or any(index < 0 or index >= source_count for index in indices)
                or len(indices) != max(1, int(source_count * fraction))
                or not isinstance(cache_manifest.get("source_csv_sha256"), str)
                or len(cache_manifest["source_csv_sha256"]) != 64
            ):
                raise ValueError(f"Generated training CSV manifest is invalid: {cache_manifest_path}")
            source_csv_sha256 = cache_manifest["source_csv_sha256"]
            source_sample_count = source_count
            source_manifest_info = {
                "path": portable_path(cache_manifest_path),
                "sha256": sha256_file(cache_manifest_path),
                "fraction": fraction,
                "selected_source_indices_sha256": hashlib.sha256(
                    json.dumps(indices, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
    return {
        "manifest_path": portable_path(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_sha256": declared_hash,
        "training_csv_sha256": manifest["dataset_csv_sha256"],
        "training_dataset_sample_count": manifest["dataset_sample_count"],
        "training_source_csv_sha256": source_csv_sha256,
        "training_source_sample_count": source_sample_count,
        "training_effective_fraction": len(sample_ids) / source_sample_count,
        "training_source_manifest": source_manifest_info,
        "selected_sample_count": len(sample_ids),
        "selected_sample_ids_sha256": hashlib.sha256(sample_ids_payload.encode("utf-8")).hexdigest(),
        "cells_sha256": manifest["cells_sha256"],
        "cell_count": manifest["cell_count"],
        "seed": manifest["seed"],
        "config": {
            key: portable_path(value) if key in {"csv", "checkpoint_dir", "cells_json", "places365_checkpoint"}
            and value is not None else value
            for key, value in manifest["config"].items()
        },
        "initialization": manifest["initialization"],
        "software": manifest["software"],
    }


def load_training_metrics(checkpoint: Path, training: dict[str, Any]) -> dict[str, Any]:
    """Verify and summarize the training metrics paired with a checkpoint."""
    run_tag = training["config"].get("run_tag")
    if not isinstance(run_tag, str) or not run_tag:
        raise ValueError(f"Training manifest has no run tag for checkpoint: {checkpoint}")
    metrics_path = checkpoint.parent / f"metrics_{run_tag}.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing training metrics for checkpoint: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_meta = metrics.get("run") if isinstance(metrics, dict) else None
    epochs = metrics.get("epochs") if isinstance(metrics, dict) else None
    if not isinstance(run_meta, dict) or not isinstance(epochs, list) or not epochs:
        raise ValueError(f"Invalid training metrics artifact: {metrics_path}")
    if (
        run_meta.get("run_manifest_sha256") != training["manifest_sha256"]
        or run_meta.get("seed") != training["seed"]
        or run_meta.get("sampling_mode") != training["config"].get("sampling_mode")
        or run_meta.get("epochs") != training["config"].get("epochs")
    ):
        raise ValueError(f"Training metrics and manifest disagree: {metrics_path}")
    final_epoch = epochs[-1]
    if (
        not isinstance(final_epoch, dict)
        or final_epoch.get("epoch") != run_meta["epochs"]
        or any(
            isinstance(final_epoch.get(key), bool)
            or not isinstance(final_epoch.get(key), (int, float))
            or not math.isfinite(final_epoch[key])
            for key in ("loss", "cell_accuracy_pct", "elapsed_sec")
        )
    ):
        raise ValueError(f"Training metrics have an invalid final epoch: {metrics_path}")
    return {
        "path": portable_path(metrics_path),
        "sha256": sha256_file(metrics_path),
        "final_epoch": final_epoch["epoch"],
        "final_training_loss": final_epoch["loss"],
        "final_training_cell_accuracy_pct": final_epoch["cell_accuracy_pct"],
        "final_epoch_elapsed_sec": final_epoch["elapsed_sec"],
        "run_elapsed_sec": run_meta.get("total_elapsed_sec"),
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
        "training_dataset_sample_count",
        "training_source_csv_sha256",
        "training_source_sample_count",
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
            uniform["training_csv_sha256"] != balanced["training_csv_sha256"]
            or
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


def majority_cell_baseline(
    cells: list[dict[str, Any]], true_lats: list[float], true_lons: list[float]
) -> dict[str, Any]:
    """Score the training-majority cell as a geography-aware trivial baseline."""
    if not cells:
        raise ValueError("training cells must not be empty")
    if not true_lats or len(true_lats) != len(true_lons):
        raise ValueError("evaluation coordinates must be non-empty and aligned")
    normalized = []
    for cell in cells:
        cell_id, count = cell.get("cell_id"), cell.get("count")
        lat, lon = cell.get("centroid_lat"), cell.get("centroid_lon")
        if (
            isinstance(cell_id, bool) or not isinstance(cell_id, int)
            or isinstance(count, bool) or not isinstance(count, int) or count < 0
            or not isinstance(lat, (int, float)) or not math.isfinite(lat) or not -90 <= lat <= 90
            or not isinstance(lon, (int, float)) or not math.isfinite(lon) or not -180 <= lon <= 180
        ):
            raise ValueError("training cells contain invalid ids, counts, or centroids")
        normalized.append((cell_id, count, float(lat), float(lon)))
    normalized.sort()
    if [cell[0] for cell in normalized] != list(range(len(normalized))):
        raise ValueError("training cell ids must be contiguous from zero")
    cell_id, train_count, pred_lat, pred_lon = max(
        normalized, key=lambda cell: (cell[1], -cell[0])
    )

    distances = []
    for lat, lon in zip(true_lats, true_lons, strict=True):
        if (
            isinstance(lat, bool) or isinstance(lon, bool)
            or not isinstance(lat, (int, float)) or not math.isfinite(lat) or not -90 <= lat <= 90
            or not isinstance(lon, (int, float)) or not math.isfinite(lon) or not -180 <= lon <= 180
        ):
            raise ValueError("evaluation coordinates must be finite latitude/longitude values")
        phi1, phi2 = math.radians(pred_lat), math.radians(lat)
        dphi = phi2 - phi1
        dlam = math.radians(lon - pred_lon)
        hav = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        distances.append(2 * 6371.0 * math.asin(math.sqrt(min(1.0, max(0.0, hav)))))

    total = sum(cell[1] for cell in normalized)
    return {
        "baseline": "always predict the most frequent training cell centroid",
        "cell_id": cell_id,
        "training_cell_count": train_count,
        "training_cell_share_pct": 100 * train_count / total if total else None,
        "n": len(distances),
        "within_1km": 100 * sum(distance <= 1 for distance in distances) / len(distances),
        "within_25km": 100 * sum(distance <= 25 for distance in distances) / len(distances),
        "within_200km": 100 * sum(distance <= 200 for distance in distances) / len(distances),
        "mean_km": statistics.mean(distances),
        "median_km": statistics.median(distances),
    }


def load_majority_cell_baseline(training: dict[str, Any], evaluation_csv: Path) -> dict[str, Any]:
    """Load hash-verified training cells and score their majority-cell prior."""
    cells_name = training["config"].get("cells_json")
    if not isinstance(cells_name, str) or not cells_name:
        raise ValueError("training manifest has no fixed-cell artifact path")
    cells_path = Path(cells_name)
    if not cells_path.is_absolute():
        cells_path = ROOT / cells_path
    artifact = json.loads(cells_path.read_text(encoding="utf-8"))
    cells = artifact.get("cells") if isinstance(artifact, dict) else None
    if not isinstance(cells, list) or len(cells) != training["cell_count"]:
        raise ValueError(f"invalid fixed-cell artifact: {cells_path}")
    cell_hash = hashlib.sha256(
        json.dumps(cells, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if cell_hash != training["cells_sha256"]:
        raise ValueError(f"fixed-cell artifact hash mismatch: {cells_path}")
    with evaluation_csv.open("r", newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        if not rows.fieldnames or not {"lat", "lon"}.issubset(rows.fieldnames):
            raise ValueError(f"evaluation CSV must contain lat and lon columns: {evaluation_csv}")
        coordinates = [(float(row["lat"]), float(row["lon"])) for row in rows]
    baseline = majority_cell_baseline(
        cells, [point[0] for point in coordinates], [point[1] for point in coordinates]
    )
    baseline["cells"] = portable_path(cells_path)
    baseline["cells_sha256"] = cell_hash
    baseline["evaluation_csv_sha256"] = sha256_file(evaluation_csv)
    return baseline


def load_evaluation_sample_manifest(evaluation_csv: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify sample identity and split metadata for an evaluation CSV."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("split") != "test":
        raise ValueError("evaluation sample manifest must describe the test split")
    ids = []
    splits = set()
    with evaluation_csv.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"id", "split"}.issubset(reader.fieldnames):
            raise ValueError("evaluation CSV must contain id and split columns")
        for row in reader:
            ids.append(row["id"])
            splits.add(row["split"])
    id_hash = hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()
    saved_ids = manifest.get("sample_ids")
    if (
        not ids
        or len(ids) != len(set(ids))
        or not isinstance(saved_ids, list)
        or not all(isinstance(value, str) for value in saved_ids)
        or sorted(saved_ids) != sorted(ids)
        or splits != {"test"}
        or manifest.get("final_sample_count") != len(ids)
        or manifest.get("metadata_sha256") != sha256_file(evaluation_csv)
        or manifest.get("final_id_set_sha256") != id_hash
    ):
        raise ValueError("evaluation sample manifest does not match its CSV")
    return {**manifest, "path": portable_path(manifest_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data/osv5m_test/metadata.csv",
                        help="Fixed, untouched evaluation CSV")
    parser.add_argument("--sample-manifest", type=Path,
                        help="Optional manifest that identifies the held-out CSV rows and split")
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
    sample_manifest = None
    if args.sample_manifest:
        sample_manifest = load_evaluation_sample_manifest(args.csv, args.sample_manifest)

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
                "training_metrics": load_training_metrics(checkpoint, training),
            })

    validate_training_comparison(selected_runs)
    majority_baseline = load_majority_cell_baseline(selected_runs[0]["training"], args.csv)

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
            "training_metrics": selected["training_metrics"],
            "metrics": parse_eval_metrics(result.stdout),
        })

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    try:
        working_tree_dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        working_tree_dirty = None
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
        "evaluation_sample_manifest": sample_manifest,
        "evaluator_revision": revision,
        "evaluator_working_tree_dirty": working_tree_dirty,
        "training_code_sha256": {
            name: sha256_file(ROOT / path)
            for name, path in {
                "train": "src/train.py",
                "model": "src/model.py",
                "dataset": "src/dataset.py",
                "cells": "src/cells.py",
                "reproducibility": "src/reproducibility.py",
                "l2_runner": "scripts/run_l2_experiments.py",
            }.items()
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
            "numpy": package_version("numpy"),
            "cuda_device_name": cuda_device_name(),
        },
        "runs": runs,
        "majority_cell_baseline": majority_baseline,
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
