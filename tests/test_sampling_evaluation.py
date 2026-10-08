"""Tests for the auditable paired-sampling evaluation report."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_sampling_comparison import (
    load_training_provenance,
    parse_eval_metrics,
    summarize_runs,
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


class SamplingEvaluationTests(unittest.TestCase):
    def test_parses_json_line_and_rejects_empty_or_missing_metrics(self):
        metrics = {"within_1km": 1.0, "within_25km": 10.0, "within_200km": 30.0,
                   "mean_km": 1000.0, "median_km": 900.0, "n": 5}
        self.assertEqual(parse_eval_metrics("progress\n[json]" + json.dumps(metrics)), metrics)
        with self.assertRaises(ValueError):
            parse_eval_metrics("no metrics")
        with self.assertRaises(ValueError):
            parse_eval_metrics('[json]{"n": 0}')

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


if __name__ == "__main__":
    unittest.main()
