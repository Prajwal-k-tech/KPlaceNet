"""Compare two provenance-bearing reports from nested data-scale runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_sampling_comparison import load_training_provenance  # noqa: E402

METRICS = ("within_1km", "within_25km", "within_200km", "mean_km", "median_km")


def compare_reports(smaller: dict[str, Any], larger: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """Verify controlled, nested runs and report descriptive paired deltas."""
    small_comparison = smaller.get("comparison")
    large_comparison = larger.get("comparison")
    if not isinstance(small_comparison, dict) or not isinstance(large_comparison, dict):
        raise ValueError("reports must contain comparison metadata")
    small_fraction = small_comparison.get("fraction")
    large_fraction = large_comparison.get("fraction")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in (small_fraction, large_fraction)) or not (0 < small_fraction < large_fraction <= 1):
        raise ValueError("reports must describe increasing data fractions")
    for field in ("evaluation_csv_sha256", "evaluation_sample_manifest", "evaluation", "training_code_sha256"):
        if smaller.get(field) != larger.get(field):
            raise ValueError(f"reports use different {field}")
    sample_manifest = smaller.get("evaluation_sample_manifest")
    if not isinstance(sample_manifest, dict) or sample_manifest.get("split") != "test":
        raise ValueError("data-scale reports require a verified test sample manifest")
    sample_ids = sample_manifest.get("sample_ids")
    if (
        not isinstance(sample_ids, list)
        or not sample_ids
        or not all(isinstance(value, str) for value in sample_ids)
        or len(sample_ids) != len(set(sample_ids))
        or len(sample_ids) != sample_manifest.get("final_sample_count")
        or sample_manifest.get("metadata_sha256") != smaller["evaluation_csv_sha256"]
        or hashlib.sha256("\n".join(sorted(sample_ids)).encode("utf-8")).hexdigest()
        != sample_manifest.get("final_id_set_sha256")
    ):
        raise ValueError("data-scale reports have an invalid test sample manifest")
    for field in ("init", "regime", "modes", "seeds"):
        if small_comparison.get(field) != large_comparison.get(field):
            raise ValueError(f"reports use different comparison {field}")
    if small_comparison.get("modes") != ["uniform"]:
        raise ValueError("data-scale reports must use uniform sampling only")

    def by_seed(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
        runs = report.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError("report has no runs")
        indexed: dict[int, dict[str, Any]] = {}
        for run in runs:
            seed = run.get("seed")
            if seed in indexed or run.get("sampling_mode") != "uniform":
                raise ValueError("report has duplicate seeds or non-uniform runs")
            training = run.get("training")
            if not isinstance(training, dict):
                raise ValueError("run has no training provenance")
            checkpoint = root / run["checkpoint"]
            verified = load_training_provenance(checkpoint)
            manifest_path = checkpoint.parent / "run_manifest.json"
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            if verified["seed"] != seed or verified["selected_sample_count"] != training["selected_sample_count"]:
                raise ValueError("run manifest does not match its report")
            if hashlib.sha256(manifest_bytes).hexdigest() != training["manifest_file_sha256"]:
                raise ValueError("run manifest file hash differs from report")
            if training["training_dataset_sample_count"] != verified["training_dataset_sample_count"]:
                raise ValueError("training dataset count differs from manifest")
            for field in (
                "training_source_csv_sha256",
                "training_source_sample_count",
                "training_effective_fraction",
            ):
                if training.get(field) != verified[field]:
                    raise ValueError(f"training source provenance differs for {field}")
            if (
                training["training_csv_sha256"] != verified["training_csv_sha256"]
                or training["cells_sha256"] != verified["cells_sha256"]
                or training["selected_sample_ids_sha256"] != verified["selected_sample_ids_sha256"]
            ):
                raise ValueError("training provenance differs from report")
            expected_count = int(verified["training_source_sample_count"] * report["comparison"]["fraction"])
            if (
                verified["selected_sample_count"] != expected_count
                or verified["training_dataset_sample_count"] != expected_count
            ):
                raise ValueError("selected sample count does not match the reported fraction")
            if training.get("manifest_sha256") != verified["manifest_sha256"]:
                raise ValueError("run manifest hash differs from report")
            if run.get("metrics") is None or any(
                isinstance(run["metrics"].get(key), bool)
                or not isinstance(run["metrics"].get(key), (int, float))
                or not math.isfinite(run["metrics"][key])
                for key in METRICS
            ):
                raise ValueError("run has invalid evaluation metrics")
            if manifest["config"].get("sampling_mode") != "uniform":
                raise ValueError("run manifest is not uniform sampling")
            indexed[seed] = {"run": run, "manifest": manifest}
        if set(indexed) != set(report["comparison"].get("seeds", [])):
            raise ValueError("report seeds do not match its run manifests")
        for item in indexed.values():
            if item["manifest"]["config"].get("seed") != item["run"]["seed"]:
                raise ValueError("run configuration seed mismatch")
            if item["manifest"]["config"].get("csv") is None:
                raise ValueError("run manifest has no training CSV")
        return indexed

    small_runs = by_seed(smaller)
    large_runs = by_seed(larger)
    if set(small_runs) != set(large_runs):
        raise ValueError("reports do not contain matching seeds")
    deltas: dict[str, list[float]] = {key: [] for key in METRICS}
    per_seed = []
    for seed in sorted(small_runs):
        small = small_runs[seed]
        large = large_runs[seed]
        left, right = small["manifest"], large["manifest"]
        if (
            small["run"]["training"]["training_source_csv_sha256"]
            != large["run"]["training"]["training_source_csv_sha256"]
            or small["run"]["training"]["training_source_sample_count"]
            != large["run"]["training"]["training_source_sample_count"]
        ):
            raise ValueError("paired runs do not use the same source training CSV")
        if left["cells_sha256"] != right["cells_sha256"]:
            raise ValueError("paired runs do not use the same geographic cells")
        if left["initialization"] != right["initialization"] or left["software"] != right["software"]:
            raise ValueError("paired runs use different initialization or software")
        config_fields = set(left["config"]) | set(right["config"])
        variable = {"csv", "checkpoint_dir", "run_tag", "seed", "max_samples"}
        if any(left["config"].get(key) != right["config"].get(key) for key in config_fields - variable):
            raise ValueError("paired runs use different training configurations")
        small_ids, large_ids = set(left["selected_sample_ids"]), set(right["selected_sample_ids"])
        if not small_ids < large_ids:
            raise ValueError("smaller run's selected training IDs are not a strict subset of larger run")
        row = {"seed": seed, "small_sample_count": len(small_ids), "large_sample_count": len(large_ids)}
        for key in METRICS:
            delta = large["run"]["metrics"][key] - small["run"]["metrics"][key]
            deltas[key].append(delta)
            row[f"delta_{key}"] = delta
        per_seed.append(row)

    return {
        "schema_version": 1,
        "small_fraction": small_fraction,
        "large_fraction": large_fraction,
        "evaluation_csv_sha256": smaller["evaluation_csv_sha256"],
        "training_source_csv_sha256": small_runs[next(iter(small_runs))]["run"]["training"]["training_source_csv_sha256"],
        "training_source_sample_count": small_runs[next(iter(small_runs))]["run"]["training"]["training_source_sample_count"],
        "cells_sha256": small_runs[next(iter(small_runs))]["manifest"]["cells_sha256"],
        "delta_direction": "larger data fraction minus smaller data fraction",
        "per_seed": per_seed,
        "summary": {
            key: {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0}
            for key, values in deltas.items()
        },
        "interpretation": (
            "Exploratory paired results: the data-scale question followed inspection of an earlier test sample. "
            "This sample is ID-disjoint but comes from the same official OSV-5M test split; three seeds do not "
            "establish statistical significance or generalization beyond these rows."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smaller", type=Path, help="report for the smaller training fraction")
    parser.add_argument("larger", type=Path, help="report for the larger training fraction")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root for report artifacts")
    parser.add_argument("--output", type=Path, help="write JSON comparison here")
    args = parser.parse_args()
    smaller = json.loads(args.smaller.read_text(encoding="utf-8"))
    larger = json.loads(args.larger.read_text(encoding="utf-8"))
    result = compare_reports(smaller, larger, args.root)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
