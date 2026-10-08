"""Tests for paired data-scale report validation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_data_scale_reports import compare_reports


def write_manifest(root: Path, name: str, seed: int, ids: list[str], max_samples: int) -> tuple[str, str, str, str]:
    directory = root / "checkpoints" / name
    directory.mkdir(parents=True)
    csv_path = root / "data" / f"{name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_content = ("id\n" + "\n".join(ids) + "\n").encode()
    csv_path.write_bytes(csv_content)
    csv_hash = hashlib.sha256(csv_content).hexdigest()
    source_hash = hashlib.sha256(b"shared source csv").hexdigest()
    fraction = max_samples / 4
    (csv_path.with_name(csv_path.name + ".manifest.json")).write_text(json.dumps({
        "format_version": 1,
        "source_csv_sha256": source_hash,
        "fraction": fraction,
        "seed": seed,
        "sampling_algorithm": "test fixture",
        "generator_sha256": hashlib.sha256(b"generator").hexdigest(),
        "source_row_count": 4,
        "row_count": len(ids),
        "selected_source_indices": list(range(len(ids))),
        "artifact_sha256": csv_hash,
    }), encoding="utf-8")
    manifest = {
        "dataset_csv_sha256": csv_hash,
        "dataset_sample_count": len(ids),
        "selected_sample_ids": ids,
        "cells_sha256": "same-cells",
        "cell_count": 300,
        "seed": seed,
        "config": {
            "csv": str(csv_path),
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
    return (f"checkpoints/{name}/last.pt", manifest["manifest_sha256"],
            hashlib.sha256(rendered.encode()).hexdigest(), csv_hash)


def report(root: Path, fraction: float, ids_for_seed: dict[int, list[str]], offset: float = 0.0) -> dict:
    runs = []
    for seed, ids in ids_for_seed.items():
        name = f"{fraction}_{seed}"
        checkpoint, manifest_hash, file_hash, csv_hash = write_manifest(root, name, seed, ids, len(ids))
        runs.append({
            "seed": seed,
            "sampling_mode": "uniform",
            "checkpoint": checkpoint,
            "training": {
                "manifest_sha256": manifest_hash,
                "manifest_file_sha256": file_hash,
                "training_csv_sha256": csv_hash,
                "training_dataset_sample_count": len(ids),
                "training_source_csv_sha256": hashlib.sha256(b"shared source csv").hexdigest(),
                "training_source_sample_count": 4,
                "training_effective_fraction": len(ids) / 4,
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
    test_ids = ["test-a", "test-b"]
    return {
        "comparison": {"fraction": fraction, "init": "imagenet", "regime": "layer4", "modes": ["uniform"], "seeds": [42, 43, 44]},
        "evaluation_csv_sha256": "same-test",
        "evaluation_sample_manifest": {
            "split": "test",
            "final_sample_count": len(test_ids),
            "metadata_sha256": "same-test",
            "sample_ids": test_ids,
            "final_id_set_sha256": hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
        },
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
