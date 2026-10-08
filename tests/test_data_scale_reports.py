"""Tests for paired data-scale report validation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_data_scale_reports import compare_reports


def write_manifest(root: Path, name: str, seed: int, ids: list[str], max_samples: int) -> tuple[str, str]:
    directory = root / "checkpoints" / name
    directory.mkdir(parents=True)
    manifest = {
        "dataset_csv_sha256": "same-source",
        "dataset_sample_count": 4,
        "selected_sample_ids": ids,
        "cells_sha256": "same-cells",
        "cell_count": 300,
        "seed": seed,
        "config": {
            "csv": f"data/{name}.csv",
            "checkpoint_dir": str(directory),
            "run_tag": name,
            "seed": seed,
            "sampling_mode": "uniform",
            "max_samples": max_samples,
            "epochs": 10,
            "batch_size": 16,
        },
        "initialization": {"kind": "imagenet"},
        "software": {"torch": "test"},
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["manifest_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    path = directory / "run_manifest.json"
    rendered = json.dumps(manifest)
    path.write_text(rendered, encoding="utf-8")
    return f"checkpoints/{name}/last.pt", manifest["manifest_sha256"], hashlib.sha256(rendered.encode()).hexdigest()


def report(root: Path, fraction: float, ids_for_seed: dict[int, list[str]], offset: float = 0.0) -> dict:
    runs = []
    for seed, ids in ids_for_seed.items():
        name = f"{fraction}_{seed}"
        checkpoint, manifest_hash, file_hash = write_manifest(root, name, seed, ids, len(ids))
        runs.append({
            "seed": seed,
            "sampling_mode": "uniform",
            "checkpoint": checkpoint,
            "training": {
                "manifest_sha256": manifest_hash,
                "manifest_file_sha256": file_hash,
                "training_csv_sha256": "same-source",
                "training_dataset_sample_count": 4,
                "selected_sample_count": len(ids),
                "selected_sample_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                "cells_sha256": "same-cells",
            },
            "metrics": {
                "within_1km": 1 + offset,
                "within_25km": 10 + offset,
                "within_200km": 30 + offset,
                "mean_km": 1000 - offset,
                "median_km": 900 - offset,
            },
        })
    return {
        "comparison": {"fraction": fraction, "init": "imagenet", "regime": "layer4", "modes": ["uniform"], "seeds": [42, 43, 44]},
        "evaluation_csv_sha256": "same-test",
        "evaluation": {"image_size": 224},
        "training_code_sha256": {"train": "same-code"},
        "runs": runs,
    }


class DataScaleReportTests(unittest.TestCase):
    def test_compares_nested_seed_pairs_and_reports_sample_sd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids10 = {seed: [f"{seed}-a", f"{seed}-b"] for seed in (42, 43, 44)}
            ids20 = {seed: [*ids10[seed], f"{seed}-c", f"{seed}-d"] for seed in ids10}
            result = compare_reports(report(root, 0.5, ids10), report(root, 1.0, ids20, 2.0), root)
        self.assertEqual([row["delta_mean_km"] for row in result["per_seed"]], [-2.0] * 3)
        self.assertEqual(result["summary"]["mean_km"], {"mean": -2.0, "sample_sd": 0.0})
        self.assertIn("do not establish statistical significance", result["interpretation"])

    def test_rejects_test_set_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids10 = {seed: [f"{seed}-a", f"{seed}-b"] for seed in (42, 43, 44)}
            ids20 = {seed: [*ids10[seed], f"{seed}-c", f"{seed}-d"] for seed in ids10}
            small, large = report(root, 0.5, ids10), report(root, 1.0, ids20)
            large["evaluation_csv_sha256"] = "different-test"
            with self.assertRaisesRegex(ValueError, "evaluation_csv_sha256"):
                compare_reports(small, large, root)

    def test_rejects_non_nested_training_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids10 = {seed: [f"{seed}-a", f"{seed}-b"] for seed in (42, 43, 44)}
            ids20 = {seed: [f"{seed}-a", f"{seed}-c", f"{seed}-d", f"{seed}-e"] for seed in ids10}
            with self.assertRaisesRegex(ValueError, "strict subset"):
                compare_reports(report(root, 0.5, ids10), report(root, 1.0, ids20), root)


if __name__ == "__main__":
    unittest.main()
