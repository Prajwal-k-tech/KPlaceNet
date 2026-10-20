"""Exercise one tiny CPU train/evaluate run without external data or weights."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODULES = ("torch", "torchvision", "numpy", "pandas", "PIL", "tqdm")
HAS_MODEL_DEPENDENCIES = all(
    importlib.util.find_spec(module) is not None for module in REQUIRED_MODULES
)


@unittest.skipUnless(HAS_MODEL_DEPENDENCIES, "install the CPU model-test dependencies")
class SyntheticTrainEvalTests(unittest.TestCase):
    def test_training_checkpoint_runs_through_evaluator(self) -> None:
        from PIL import Image, ImageDraw

        randomizer = random.Random(20261008)
        with tempfile.TemporaryDirectory(prefix="kplacenet-train-eval-") as temporary:
            root = Path(temporary)
            for split, count in (("train", 8), ("eval", 4)):
                with (root / f"{split}.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["image_path", "lat", "lon"])
                    for index in range(count):
                        group = index % 2
                        lat, lon = ((-35.0, -120.0), (35.0, 120.0))[group]
                        lat += randomizer.uniform(-1.0, 1.0)
                        lon += randomizer.uniform(-1.0, 1.0)
                        image_name = f"{split}_{index}.png"
                        image = Image.new(
                            "RGB",
                            (80, 80),
                            (240, 40, 40) if group == 0 else (40, 40, 240),
                        )
                        draw = ImageDraw.Draw(image)
                        x = randomizer.randrange(0, 50)
                        draw.rectangle((x, 5, 75, 75), outline="white", width=2)
                        image.save(root / image_name)
                        writer.writerow([image_name, lat, lon])

            checkpoint_dir = root / "checkpoints"
            env = os.environ | {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
            training = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.train",
                    "--csv",
                    str(root / "train.csv"),
                    "--num-cells",
                    "2",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--image-size",
                    "64",
                    "--no-pretrained",
                    "--no-amp",
                    "--device",
                    "cpu",
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                    "--seed",
                    "13",
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            self.assertEqual(training.returncode, 0, training.stdout + training.stderr)
            checkpoint = checkpoint_dir / "last.pt"
            self.assertTrue(checkpoint.is_file())
            manifest_path = checkpoint_dir / "run_manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset_sample_count"], 8)
            self.assertEqual(manifest["selected_indices"], list(range(8)))
            self.assertEqual(len(manifest["selected_sample_ids"]), 8)
            self.assertEqual(manifest["initialization"]["kind"], "random_initialization")
            metrics_path = checkpoint_dir / "metrics_run.json"
            self.assertTrue(metrics_path.is_file())
            training_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(
                training_metrics["run"]["run_manifest_sha256"],
                manifest["manifest_sha256"],
            )

            import torch

            saved_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(
                saved_checkpoint["run_manifest_sha256"], manifest["manifest_sha256"]
            )

            evaluation = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.eval",
                    "--csv",
                    str(root / "eval.csv"),
                    "--checkpoint",
                    str(checkpoint),
                    "--num-cells",
                    "2",
                    "--batch-size",
                    "2",
                    "--image-size",
                    "64",
                    "--device",
                    "cpu",
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            self.assertEqual(evaluation.returncode, 0, evaluation.stdout + evaluation.stderr)
            metrics_line = next(
                (
                    line[len("[json]") :]
                    for line in evaluation.stdout.splitlines()
                    if line.startswith("[json]")
                ),
                None,
            )
            self.assertIsNotNone(metrics_line, evaluation.stdout)
            metrics = json.loads(metrics_line)
            self.assertEqual(metrics["n"], 4)
            for metric in (
                "within_1km",
                "within_25km",
                "within_200km",
                "mean_km",
                "median_km",
            ):
                self.assertTrue(math.isfinite(metrics[metric]), metric)


if __name__ == "__main__":
    unittest.main()
