"""Dependency-free tests for geographic-cell sampling weights."""

from __future__ import annotations

import math
import unittest

from src.sampling import cell_sampling_weights


class CellSamplingWeightTests(unittest.TestCase):
    def test_power_one_gives_equal_total_weight_per_observed_cell(self):
        weights = cell_sampling_weights([0, 0, 0, 1, 2, 2])
        totals = [sum(w for w, label in zip(weights, [0, 0, 0, 1, 2, 2]) if label == c)
                  for c in (0, 1, 2)]
        self.assertEqual(totals, [1.0, 1.0, 1.0])

    def test_power_zero_is_uniform_and_fractional_power_softens(self):
        labels = [0, 0, 0, 0, 1]
        self.assertEqual(cell_sampling_weights(labels, power=0), [1.0] * 5)
        weights = cell_sampling_weights(labels, power=0.5)
        self.assertEqual(weights, [0.5, 0.5, 0.5, 0.5, 1.0])

    def test_rejects_empty_invalid_labels_and_invalid_power(self):
        for labels, power in (([], 1), ([0, -1], 1), ([0, True], 1), ([0], math.nan),
                              ([0], -0.1), ([0], 1.1), ([0], True)):
            with self.subTest(labels=labels, power=power), self.assertRaises(ValueError):
                cell_sampling_weights(labels, power=power)


if __name__ == "__main__":
    unittest.main()
