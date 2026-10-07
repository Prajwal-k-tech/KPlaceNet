"""Tests for paired L2 sampling experiment command construction."""

from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path

from scripts.run_l2_experiments import checkpoint_dir_name, make_train_cmd


class SamplingRunnerTests(unittest.TestCase):
    def test_uniform_checkpoint_name_preserves_existing_l2_path(self):
        self.assertEqual(
            checkpoint_dir_name("imagenet", 1.0, "frozen", "uniform"),
            "l2_imagenet_1.0_frozen",
        )

    def test_balanced_run_uses_separate_checkpoint_and_records_mode(self):
        args = Namespace(epochs=2, image_size=224, lr=0.001, seed=42,
                         device="cpu", sampling_power=0.75,
                         places365_checkpoint="places.pt")
        cmd = make_train_cmd(args, Path("train.csv"), Path("balanced"),
                             "imagenet", 1.0, "frozen", "cell-balanced",
                             32, "cells.json", 300, 1000)
        self.assertIn("l2_imagenet_1.0_frozen_cell-balanced", cmd)
        self.assertEqual(cmd[cmd.index("--sampling-mode") + 1], "cell-balanced")
        self.assertEqual(cmd[cmd.index("--sampling-power") + 1], "0.75")

    def test_non_default_seed_gets_an_isolated_checkpoint_directory(self):
        name = checkpoint_dir_name("imagenet", 1.0, "frozen", "cell-balanced", 43)
        self.assertEqual(name, "l2_imagenet_1.0_frozen_s43_cell-balanced")


if __name__ == "__main__":
    unittest.main()
