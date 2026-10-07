"""Tests for demo calibration artifact parsing without ML dependencies."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.demo_config import (
    load_calibration,
    verify_calibration_checkpoint,
)


class DemoCalibrationConfigTests(unittest.TestCase):
    def test_missing_artifact_selects_explicit_raw_top1_mode(self):
        config = load_calibration(Path("missing-calibration.json"))
        self.assertEqual(config.temperature, 1.0)
        self.assertIsNone(config.conformal_quantile)
        self.assertFalse(config.temperature_fitted)
        self.assertFalse(config.conformal_fitted)

    def test_loads_fitted_temperature_and_infinite_conformal_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps({
                "temperature": 2.5,
                "conformal_quantile": "Infinity",
            }), encoding="utf-8")
            config = load_calibration(path)
        self.assertEqual(config.temperature, 2.5)
        self.assertEqual(config.conformal_quantile, math.inf)
        self.assertTrue(config.temperature_fitted)
        self.assertTrue(config.conformal_fitted)

    def test_checkpoint_hash_is_checked_when_present(self):
        expected = "a" * 64
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps({
                "temperature": 1.5,
                "provenance": {"checkpoint_sha256": expected},
            }), encoding="utf-8")
            config = load_calibration(path)
        self.assertTrue(verify_calibration_checkpoint(config, expected.upper()))
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            verify_calibration_checkpoint(config, "b" * 64)

    def test_checkpoint_hash_must_be_sha256_hex(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps({"provenance": {"checkpoint_sha256": "bad"}}),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "64-character"):
                load_calibration(path)

    def test_hashless_artifact_is_explicitly_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps({"temperature": 1.25}), encoding="utf-8")
            config = load_calibration(path)
        self.assertFalse(verify_calibration_checkpoint(config, "a" * 64))

    def test_rejects_malformed_or_out_of_range_calibration(self):
        invalid = (
            "not-json",
            json.dumps({"temperature": 0.0}),
            json.dumps({"temperature": 1.0, "conformal_quantile": -0.1}),
            json.dumps({"temperature": 1.0, "conformal_quantile": 1.1}),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            for content in invalid:
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_calibration(path)


if __name__ == "__main__":
    unittest.main()
