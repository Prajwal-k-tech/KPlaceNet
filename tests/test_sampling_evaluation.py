"""Tests for the auditable paired-sampling evaluation report."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_sampling_comparison import (
    ROOT,
    load_evaluation_sample_manifest,
    load_training_provenance,
    load_training_metrics,
    load_majority_cell_baseline,
    majority_cell_baseline,
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


def training(seed, mode, sample_hash="same-samples", cell_hash="same-cells", csv_hash="same-dataset"):
    return {
        "seed": seed,
        "training_csv_sha256": csv_hash,
        "training_dataset_sample_count": 100,
        "training_source_csv_sha256": "same-source-dataset",
        "training_source_sample_count": 100,
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
    def test_comparison_allows_seed_specific_csvs_from_same_source(self):
        runs = [
            {"seed": seed, "sampling_mode": "uniform", "training": training(
                seed, "uniform", sample_hash=f"samples-{seed}", csv_hash=f"csv-{seed}"
            )}
            for seed in (42, 43, 44)
        ]
        validate_training_comparison(runs)

    def test_training_provenance_verifies_generated_subset_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "train.csv"
            csv_bytes = b"id\na\nb\n"
            csv_path.write_bytes(csv_bytes)
            csv_hash = hashlib.sha256(csv_bytes).hexdigest()
            source_hash = hashlib.sha256(b"source").hexdigest()
            (root / "train.csv.manifest.json").write_text(json.dumps({
                "format_version": 1,
                "source_csv_sha256": source_hash,
                "fraction": 0.5,
                "seed": 42,
                "sampling_algorithm": "test",
                "generator_sha256": "0" * 64,
                "source_row_count": 4,
                "row_count": 2,
                "selected_source_indices": [0, 2],
                "artifact_sha256": csv_hash,
            }), encoding="utf-8")
            run_dir = root / "checkpoints" / "run"
            run_dir.mkdir(parents=True)
            manifest = {
                "dataset_csv_sha256": csv_hash,
                "dataset_sample_count": 2,
                "selected_sample_ids": ["a", "b"],
                "cells_sha256": "c" * 64,
                "cell_count": 1,
                "seed": 42,
                "config": {"csv": str(csv_path), "seed": 42, "sampling_mode": "uniform"},
                "initialization": {"kind": "imagenet"},
                "software": {"torch": "test"},
            }
            canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            manifest["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            provenance = load_training_provenance(run_dir / "last.pt")

        self.assertEqual(provenance["training_csv_sha256"], csv_hash)
        self.assertEqual(provenance["training_source_csv_sha256"], source_hash)
        self.assertEqual(provenance["training_source_sample_count"], 4)
        self.assertEqual(provenance["training_effective_fraction"], 0.5)
        self.assertEqual(provenance["training_source_manifest"]["fraction"], 0.5)

    def test_evaluation_sample_manifest_binds_csv_rows_and_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "test.csv"
            csv_bytes = b"id,split\na,test\nb,test\n"
            csv_path.write_bytes(csv_bytes)
            ids_hash = hashlib.sha256(b"a\nb").hexdigest()
            manifest_path = root / "sample.json"
            manifest_path.write_text(json.dumps({
                "split": "test",
                "final_sample_count": 2,
                "metadata_sha256": hashlib.sha256(csv_bytes).hexdigest(),
                "final_id_set_sha256": ids_hash,
                "sample_ids": ["a", "b"],
            }), encoding="utf-8")

            manifest = load_evaluation_sample_manifest(csv_path, manifest_path)
            self.assertEqual(manifest["final_id_set_sha256"], ids_hash)

            tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
            tampered["sample_ids"] = ["a", "c"]
            tampered["final_id_set_sha256"] = hashlib.sha256(b"a\nc").hexdigest()
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_evaluation_sample_manifest(csv_path, manifest_path)

            tampered["sample_ids"] = ["a", "b"]
            tampered["final_id_set_sha256"] = ids_hash
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            csv_path.write_text("id,split\na,train\nb,test\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_evaluation_sample_manifest(csv_path, manifest_path)

    def test_majority_cell_baseline_uses_highest_count_and_reports_geodesic_metrics(self):
        cells = [
            {"cell_id": 0, "count": 2, "centroid_lat": 0.0, "centroid_lon": 0.0},
            {"cell_id": 1, "count": 4, "centroid_lat": 20.0, "centroid_lon": 20.0},
        ]

        result = majority_cell_baseline(cells, [20.0, 20.0], [20.0, 20.0])

        self.assertEqual(result["cell_id"], 1)
        self.assertEqual(result["training_cell_count"], 4)
        self.assertEqual(result["training_cell_share_pct"], 100.0 * 4 / 6)
        self.assertEqual(result["within_1km"], 100.0)
        self.assertEqual(result["within_25km"], 100.0)
        self.assertEqual(result["within_200km"], 100.0)
        self.assertEqual(result["mean_km"], 0.0)

    def test_majority_cell_baseline_rejects_invalid_coordinates(self):
        cells = [{"cell_id": 0, "count": 1, "centroid_lat": 0.0, "centroid_lon": 0.0}]
        with self.assertRaisesRegex(ValueError, "latitude/longitude"):
            majority_cell_baseline(cells, [91.0], [0.0])

    def test_majority_baseline_verifies_cells_and_uses_evaluation_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cells = [
                {"cell_id": 0, "count": 1, "centroid_lat": 0.0, "centroid_lon": 0.0},
                {"cell_id": 1, "count": 3, "centroid_lat": 10.0, "centroid_lon": 10.0},
            ]
            cells_path = root / "cells.json"
            cells_path.write_text(json.dumps({"num_cells": 2, "cells": cells}))
            evaluation_csv = root / "eval.csv"
            evaluation_csv.write_text("image_path,lat,lon\na.jpg,10,10\n")
            cell_hash = hashlib.sha256(
                json.dumps(cells, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            training = {
                "cell_count": 2,
                "cells_sha256": cell_hash,
                "config": {"cells_json": str(cells_path)},
            }

            result = load_majority_cell_baseline(training, evaluation_csv)

            self.assertEqual(result["cell_id"], 1)
            self.assertEqual(result["n"], 1)
            self.assertEqual(result["mean_km"], 0.0)
            self.assertEqual(result["evaluation_csv_sha256"], hashlib.sha256(evaluation_csv.read_bytes()).hexdigest())
            training["cells_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_majority_cell_baseline(training, evaluation_csv)

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
                                                        "within_1km": 0.0, "within_25km": 10.0,
                                                        "within_200km": 40.0}},
            "lat00_lon01": {"n": 2, "metrics": None},
        }}
        balanced["metrics"]["regional_distance"] = {"regions": {
            "lat00_lon00": {"n": 30, "metrics": {"mean_km": 450.0, "median_km": 350.0,
                                                        "within_1km": 0.0, "within_25km": 20.0,
                                                        "within_200km": 50.0}},
            "lat00_lon01": {"n": 2, "metrics": None},
        }}
        summary = summarize_runs([baseline, balanced])
        regional = summary["regional_paired_delta_cell_balanced_minus_uniform"]
        self.assertEqual(regional["lat00_lon00"]["mean_km"]["mean"], -50.0)
        self.assertAlmostEqual(regional["lat00_lon00"]["within_200km"]["mean"], 10.0)
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

    def test_training_metrics_are_bound_to_the_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "run" / "last.pt"
            checkpoint.parent.mkdir()
            manifest_hash = "a" * 64
            training_provenance = {
                "manifest_sha256": manifest_hash,
                "seed": 42,
                "config": {"run_tag": "demo", "sampling_mode": "uniform", "epochs": 2},
            }
            metrics = {
                "run": {
                    "run_manifest_sha256": manifest_hash,
                    "seed": 42,
                    "sampling_mode": "uniform",
                    "epochs": 2,
                    "total_elapsed_sec": 12.0,
                },
                "epochs": [
                    {"epoch": 1, "loss": 1.0, "cell_accuracy_pct": 50.0, "elapsed_sec": 6.0},
                    {"epoch": 2, "loss": 0.5, "cell_accuracy_pct": 75.0, "elapsed_sec": 6.0},
                ],
            }
            path = checkpoint.parent / "metrics_demo.json"
            path.write_text(json.dumps(metrics))

            result = load_training_metrics(checkpoint, training_provenance)

            self.assertEqual(result["final_epoch"], 2)
            self.assertEqual(result["final_training_cell_accuracy_pct"], 75.0)
            self.assertEqual(len(result["sha256"]), 64)
            metrics["run"]["run_manifest_sha256"] = "b" * 64
            path.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, "disagree"):
                load_training_metrics(checkpoint, training_provenance)

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
