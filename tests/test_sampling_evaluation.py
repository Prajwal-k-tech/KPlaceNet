"""Tests for the auditable paired-sampling evaluation report."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_sampling_comparison import (
    ROOT,
    load_training_provenance,
    parse_checkpoint_manifest_hash,
    parse_eval_metrics,
    portable_path,
    summarize_runs,
    validate_training_comparison,
)


def run(seed, mode, offset=0.0):
    return {
        "seed": seed,
        "sampling_mode": mode,
        "metrics": {
            "within_1km": 1.0 + offset,
            "within_25km": 10.0 + offset,
            "within_200km": 30.0 + offset,
            "mean_km": 1000.0 - offset,
            "median_km": 900.0 - offset,
        },
    }


def training(seed, mode, sample_hash="same-samples", cell_hash="same-cells"):
    return {
        "seed": seed,
        "training_csv_sha256": "same-dataset",
        "training_dataset_sample_count": 100,
        "selected_sample_count": 100,
        "selected_sample_ids_sha256": sample_hash,
        "cells_sha256": cell_hash,
        "initialization": {"kind": "torchvision_imagenet"},
        "software": {"torch": "test"},
        "config": {
            "seed": seed,
            "sampling_mode": mode,
            "sampling_power": 1.0,
            "csv": f"train_{seed}.csv",
            "checkpoint_dir": f"checkpoints/{seed}_{mode}",
            "run_tag": f"run_{seed}_{mode}",
            "epochs": 10,
            "batch_size": 16,
        },
    }


class SamplingEvaluationTests(unittest.TestCase):
    def test_portable_paths_omit_host_absolute_prefixes(self):
        self.assertEqual(portable_path("data/osv5m_test/metadata.csv"), "data/osv5m_test/metadata.csv")
        self.assertEqual(
            portable_path(ROOT / "checkpoints/model/last.pt"),
            "checkpoints/model/last.pt",
        )
        self.assertEqual(portable_path("/tmp/external/model.pt"), "model.pt")

    def test_parses_json_line_and_rejects_empty_or_missing_metrics(self):
        metrics = {"within_1km": 1.0, "within_25km": 10.0, "within_200km": 30.0,
                   "mean_km": 1000.0, "median_km": 900.0, "n": 5,
                   "regional_distance": {"regions": {}}}
        self.assertEqual(parse_eval_metrics("progress\n[json]" + json.dumps(metrics)), metrics)
        with self.assertRaises(ValueError):
            parse_eval_metrics("no metrics")
        with self.assertRaises(ValueError):
            parse_eval_metrics('[json]{"n": 0}')
        missing_regions = {key: value for key, value in metrics.items() if key != "regional_distance"}
        with self.assertRaisesRegex(ValueError, "no geographic-strata"):
            parse_eval_metrics("[json]" + json.dumps(missing_regions))

    def test_parses_and_validates_embedded_checkpoint_manifest_hash(self):
        digest = "a" * 64
        self.assertEqual(parse_checkpoint_manifest_hash("[run-manifest]" + digest), digest)
        with self.assertRaisesRegex(ValueError, "invalid"):
            parse_checkpoint_manifest_hash("[run-manifest]not-a-hash")
        with self.assertRaisesRegex(ValueError, "did not include"):
            parse_checkpoint_manifest_hash("no provenance")

    def test_reports_mode_means_and_seed_paired_deltas(self):
        report = summarize_runs([
            run(42, "uniform"), run(42, "cell-balanced", 2.0),
            run(43, "uniform"), run(43, "cell-balanced", 4.0),
        ])
        delta = report["paired_delta_cell_balanced_minus_uniform"]["within_25km"]
        self.assertEqual(delta["per_seed"], [2.0, 4.0])
        self.assertEqual(delta["mean"], 3.0)
        self.assertEqual(report["by_mode"]["cell-balanced"]["runs"], 2)

    def test_unpaired_seed_does_not_enter_paired_delta(self):
        report = summarize_runs([run(42, "uniform"), run(43, "cell-balanced", 2.0)])
        delta = report["paired_delta_cell_balanced_minus_uniform"]["within_200km"]
        self.assertIsNone(delta["mean"])
        self.assertEqual(delta["per_seed"], [])

    def test_reports_paired_geographic_deltas_and_omits_sparse_bins(self):
        baseline = run(42, "uniform")
        balanced = run(42, "cell-balanced")
        baseline["metrics"]["regional_distance"] = {"regions": {
            "lat00_lon00": {"n": 30, "metrics": {"mean_km": 500.0, "median_km": 400.0,
                                                        "within_1km": 0.0, "within_25km": 0.1,
                                                        "within_200km": 0.4}},
            "lat00_lon01": {"n": 2, "metrics": None},
        }}
        balanced["metrics"]["regional_distance"] = {"regions": {
            "lat00_lon00": {"n": 30, "metrics": {"mean_km": 450.0, "median_km": 350.0,
                                                        "within_1km": 0.0, "within_25km": 0.2,
                                                        "within_200km": 0.5}},
            "lat00_lon01": {"n": 2, "metrics": None},
        }}
        summary = summarize_runs([baseline, balanced])
        regional = summary["regional_paired_delta_cell_balanced_minus_uniform"]
        self.assertEqual(regional["lat00_lon00"]["mean_km"]["mean"], -50.0)
        self.assertAlmostEqual(regional["lat00_lon00"]["within_200km"]["mean"], 0.1)
        self.assertEqual(regional["lat00_lon00"]["within_200km"]["n"], 30)
        self.assertEqual(regional["lat00_lon01"]["within_200km"]["mean"], None)

    def test_training_provenance_validates_manifest_and_hashes_selected_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "run" / "last.pt"
            checkpoint.parent.mkdir()
            manifest = {
                "schema_version": 1,
                "dataset_csv_sha256": "dataset-hash",
                "dataset_sample_count": 2,
                "selected_indices": [0, 1],
                "selected_sample_ids": ["sha256:first", "sha256:second"],
                "seed": 42,
                "config": {"sampling_mode": "uniform"},
                "initialization": {"kind": "torchvision_imagenet"},
                "cells_sha256": "cells-hash",
                "cell_count": 2,
                "software": {"torch": "test"},
            }
            canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            manifest["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
            (checkpoint.parent / "run_manifest.json").write_text(json.dumps(manifest))

            provenance = load_training_provenance(checkpoint)

            expected_ids = json.dumps(manifest["selected_sample_ids"], separators=(",", ":"))
            self.assertEqual(provenance["selected_sample_count"], 2)
            self.assertEqual(provenance["training_csv_sha256"], "dataset-hash")
            self.assertEqual(
                provenance["selected_sample_ids_sha256"],
                hashlib.sha256(expected_ids.encode()).hexdigest(),
            )

    def test_training_provenance_rejects_tampered_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "run" / "last.pt"
            checkpoint.parent.mkdir()
            (checkpoint.parent / "run_manifest.json").write_text(
                json.dumps({"manifest_sha256": "wrong", "seed": 42})
            )
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_training_provenance(checkpoint)

    def test_training_comparison_requires_paired_samples_and_shared_setup(self):
        validate_training_comparison([
            {"seed": 42, "sampling_mode": "uniform", "training": training(42, "uniform")},
            {"seed": 42, "sampling_mode": "cell-balanced", "training": training(42, "cell-balanced")},
            {"seed": 43, "sampling_mode": "uniform", "training": training(43, "uniform", "seed-43-samples")},
            {"seed": 43, "sampling_mode": "cell-balanced", "training": training(43, "cell-balanced", "seed-43-samples")},
        ])

        bad_samples = [
            {"seed": 42, "sampling_mode": "uniform", "training": training(42, "uniform")},
            {"seed": 42, "sampling_mode": "cell-balanced", "training": training(42, "cell-balanced", "other-samples")},
        ]
        with self.assertRaisesRegex(ValueError, "different training samples"):
            validate_training_comparison(bad_samples)

        bad_cells = [
            {"seed": 42, "sampling_mode": "uniform", "training": training(42, "uniform")},
            {"seed": 42, "sampling_mode": "cell-balanced", "training": training(42, "cell-balanced", cell_hash="other-cells")},
        ]
        with self.assertRaisesRegex(ValueError, "same data, cells"):
            validate_training_comparison(bad_cells)


if __name__ == "__main__":
    unittest.main()
