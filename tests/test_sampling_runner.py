"""Tests for paired L2 sampling experiment command construction."""

from __future__ import annotations

import csv
import contextlib
import io
import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scripts import run_l2_experiments
from scripts.run_l2_experiments import (
    checkpoint_dir_name,
    make_subset_csv,
    make_train_cmd,
)


class SamplingRunnerTests(unittest.TestCase):
    def test_uniform_checkpoint_name_preserves_existing_l2_path(self):
        self.assertEqual(
            checkpoint_dir_name("imagenet", 1.0, "frozen", "uniform"),
            "l2_imagenet_1.0_frozen",
        )

    def test_balanced_run_uses_separate_checkpoint_and_records_mode(self):
        args = Namespace(epochs=2, image_size=224, lr=0.001, seed=42,
                         device="cpu", sampling_power=0.75,
                         places365_checkpoint="places.pt", num_workers=4)
        cmd = make_train_cmd(args, Path("train.csv"), Path("balanced"),
                             "imagenet", 1.0, "frozen", "cell-balanced",
                             32, "cells.json", 300, 1000)
        self.assertIn("l2_imagenet_1.0_frozen_cell-balanced", cmd)
        self.assertEqual(cmd[cmd.index("--sampling-mode") + 1], "cell-balanced")
        self.assertEqual(cmd[cmd.index("--sampling-power") + 1], "0.75")
        self.assertEqual(cmd[cmd.index("--num-workers") + 1], "4")

    def test_non_default_seed_gets_an_isolated_checkpoint_directory(self):
        name = checkpoint_dir_name("imagenet", 1.0, "frozen", "cell-balanced", 43)
        self.assertEqual(name, "l2_imagenet_1.0_frozen_s43_cell-balanced")

    def test_plan_only_creates_manifest_directory_on_clean_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "train.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image_path", "lat", "lon"])
                writer.writeheader()
                writer.writerow({"image_path": "one.jpg", "lat": 1, "lon": 2})

            checkpoints = root / "missing" / "checkpoints"
            cells = checkpoints / "cells.json"
            argv = [
                "run_l2_experiments.py", "--csv", str(source),
                "--fractions", "1.0", "--inits", "imagenet",
                "--regimes", "frozen", "--sampling-modes", "uniform",
                "--epochs", "1", "--plan-only",
            ]
            with patch.object(run_l2_experiments, "CHECKPOINTS_DIR", checkpoints), \
                    patch.object(run_l2_experiments, "CELLS_JSON", cells), \
                    patch.object(sys, "argv", argv), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_l2_experiments.main()

            manifest = json.loads(
                (checkpoints / "l2_experiment_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["plan_only"])
            self.assertEqual(manifest["results"][0]["sampling_mode"], "uniform")

    def test_subset_generation_rejects_invalid_fraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "train.csv"
            with self.assertRaises(ValueError):
                make_subset_csv(source, 0.0, 42, 2)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "requires NumPy")
    def test_subset_cache_tracks_source_and_preserves_source_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "train.csv"

            def write_source(revision: str) -> None:
                with source.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=["image_path", "lat", "lon", "id", "region"]
                    )
                    writer.writeheader()
                    for index in range(10):
                        writer.writerow({
                            "image_path": f"images/{index}.jpg",
                            "lat": index,
                            "lon": index + 1,
                            "id": f"{revision}-{index}",
                            "region": revision,
                        })

            write_source("first")
            subset, count = make_subset_csv(source, 0.5, 42, 2)
            self.assertEqual(count, 5)
            with subset.open(newline="", encoding="utf-8") as stream:
                first_rows = list(csv.DictReader(stream))
            self.assertEqual(len(first_rows), 5)
            self.assertEqual(first_rows[0]["region"], "first")
            first_manifest = json.loads(
                Path(str(subset) + ".manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(first_manifest["generator_sha256"])

            # Same inputs reuse a validated artifact.
            self.assertEqual(make_subset_csv(source, 0.5, 42, 2), (subset, 5))

            # A changed source invalidates and regenerates the cached subset.
            write_source("second")
            self.assertEqual(make_subset_csv(source, 0.5, 42, 2), (subset, 5))
            with subset.open(newline="", encoding="utf-8") as stream:
                second_rows = list(csv.DictReader(stream))
            second_manifest = json.loads(
                Path(str(subset) + ".manifest.json").read_text(encoding="utf-8")
            )
            self.assertNotEqual(
                first_manifest["source_csv_sha256"],
                second_manifest["source_csv_sha256"],
            )
            self.assertTrue(all(row["region"] == "second" for row in second_rows))

            # A modified generated file is also detected and rebuilt.
            subset.write_text("corrupt cache\n", encoding="utf-8")
            self.assertEqual(make_subset_csv(source, 0.5, 42, 2), (subset, 5))
            with subset.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 5)

    @unittest.skipUnless(
        all(importlib.util.find_spec(module) for module in ("torch", "torchvision", "numpy")),
        "requires the CPU model-test dependencies",
    )
    def test_cells_cache_rebuilds_when_source_changes(self):
        from scripts.run_l2_experiments import build_cells_from_csv

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "train.csv"
            cells_path = Path(temporary) / "cells.json"

            def write_source(offset: float) -> None:
                with source.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["image_path", "lat", "lon"])
                    writer.writeheader()
                    for index, (lat, lon) in enumerate(((-40, -100), (-20, -80), (20, 80), (40, 100))):
                        writer.writerow({
                            "image_path": f"{index}.jpg",
                            "lat": lat + offset,
                            "lon": lon + offset,
                        })

            write_source(0.0)
            build_cells_from_csv(source, 2, cells_path, 42)
            first_manifest = json.loads(
                Path(str(cells_path) + ".manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(first_manifest["generator_sha256"])
            first_artifact = cells_path.read_bytes()

            # Unchanged inputs validate and reuse the same cell artifact.
            build_cells_from_csv(source, 2, cells_path, 42)
            self.assertEqual(cells_path.read_bytes(), first_artifact)

            # Changed source coordinates invalidate and rebuild the partition.
            write_source(5.0)
            build_cells_from_csv(source, 2, cells_path, 42)
            second_manifest = json.loads(
                Path(str(cells_path) + ".manifest.json").read_text(encoding="utf-8")
            )
            self.assertNotEqual(
                first_manifest["source_csv_sha256"],
                second_manifest["source_csv_sha256"],
            )
            self.assertNotEqual(cells_path.read_bytes(), first_artifact)


if __name__ == "__main__":
    unittest.main()
