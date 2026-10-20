"""Tests for fail-closed geolocation checkpoint validation."""

from __future__ import annotations

import unittest

from src.eval_contract import validate_eval_checkpoint


def checkpoint(cells=None, num_cells=2):
    if cells is None:
        cells = [
            {"cell_id": 0, "centroid_lat": 10.0, "centroid_lon": 20.0},
            {"cell_id": 1, "centroid_lat": -10.0, "centroid_lon": -20.0},
        ]
    return {
        "model_state": {
            "head.weight": type("Shape", (), {"shape": (num_cells, 2048)})(),
            "head.bias": type("Shape", (), {"shape": (num_cells,)})(),
        },
        "cells": cells,
        "num_cells": num_cells,
        "args": {"num_cells": num_cells},
    }


class EvalCheckpointContractTests(unittest.TestCase):
    def test_valid_cells_are_returned_in_class_id_order(self):
        data = checkpoint(cells=[
            {"cell_id": 1, "centroid_lat": -10.0, "centroid_lon": -20.0},
            {"cell_id": 0, "centroid_lat": 10.0, "centroid_lon": 20.0},
        ])
        num_cells, cells = validate_eval_checkpoint(data)
        self.assertEqual(num_cells, 2)
        self.assertEqual([cell["cell_id"] for cell in cells], [0, 1])

    def test_requires_trained_weights_and_saved_cells(self):
        for data in (
            {"cells": checkpoint()["cells"], "num_cells": 2},
            {**checkpoint(), "cells": []},
            {**checkpoint(), "model_state": {}},
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_eval_checkpoint(data)

    def test_rejects_mismatched_class_dimensions(self):
        cases = (
            {**checkpoint(), "num_cells": 3},
            {**checkpoint(), "args": {"num_cells": 3}},
            {
                **checkpoint(),
                "model_state": {
                    "head.weight": type("Shape", (), {"shape": (3, 2048)})(),
                    "head.bias": type("Shape", (), {"shape": (3,)})(),
                },
            },
        )
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_eval_checkpoint(data)

    def test_rejects_duplicate_or_noncontiguous_cell_ids(self):
        cells = checkpoint()["cells"]
        cases = (
            [cells[0], {**cells[1], "cell_id": 0}],
            [cells[0], {**cells[1], "cell_id": 2}],
        )
        for invalid_cells in cases:
            with self.subTest(invalid_cells=invalid_cells), self.assertRaises(ValueError):
                validate_eval_checkpoint(checkpoint(cells=invalid_cells))

    def test_rejects_invalid_centroids(self):
        for key, value in (("centroid_lat", 91.0), ("centroid_lon", float("nan"))):
            cells = checkpoint()["cells"]
            cells[0] = {**cells[0], key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_eval_checkpoint(checkpoint(cells=cells))


if __name__ == "__main__":
    unittest.main()
