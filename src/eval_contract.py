"""Validation helpers for evaluation checkpoints.

This module intentionally uses only the standard library so the checkpoint
contract can be tested without installing the ML stack or loading images.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any


def validate_eval_checkpoint(checkpoint: Any) -> tuple[int, list[dict[str, Any]]]:
    """Return the model class count and ordered cells for a valid checkpoint.

    Evaluation must use the model weights and geographic class definitions
    saved together during training. Inferring either from evaluation labels
    would leak information into the benchmark; silently evaluating random or
    partially loaded weights would produce meaningless scores.
    """
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")

    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("checkpoint is missing a non-empty 'model_state'")

    cells = checkpoint.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("checkpoint is missing its non-empty 'cells' metadata")

    num_cells = checkpoint.get("num_cells")
    if isinstance(num_cells, bool) or not isinstance(num_cells, int) or num_cells <= 0:
        raise ValueError("checkpoint 'num_cells' must be a positive integer")
    if len(cells) != num_cells:
        raise ValueError(
            f"checkpoint has {len(cells)} cells but declares num_cells={num_cells}"
        )

    args = checkpoint.get("args")
    if isinstance(args, Mapping) and "num_cells" in args:
        arg_num_cells = args["num_cells"]
        if (
            isinstance(arg_num_cells, bool)
            or not isinstance(arg_num_cells, int)
            or arg_num_cells != num_cells
        ):
            raise ValueError("checkpoint 'args.num_cells' disagrees with 'num_cells'")

    head_weight = model_state.get("head.weight")
    head_shape = getattr(head_weight, "shape", None)
    if head_shape is None or len(head_shape) != 2 or head_shape[0] != num_cells:
        raise ValueError(
            "checkpoint 'head.weight' output dimension does not match num_cells"
        )

    head_bias = model_state.get("head.bias")
    bias_shape = getattr(head_bias, "shape", None)
    if bias_shape is None or len(bias_shape) != 1 or bias_shape[0] != num_cells:
        raise ValueError(
            "checkpoint 'head.bias' output dimension does not match num_cells"
        )

    ordered_cells: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            raise ValueError(f"checkpoint cell at index {index} must be a mapping")

        cell_id = cell.get("cell_id")
        if isinstance(cell_id, bool) or not isinstance(cell_id, int):
            raise ValueError(f"checkpoint cell at index {index} has an invalid cell_id")
        if cell_id in seen_ids:
            raise ValueError(f"checkpoint contains duplicate cell_id {cell_id}")
        seen_ids.add(cell_id)

        lat = cell.get("centroid_lat")
        lon = cell.get("centroid_lon")
        if (
            isinstance(lat, bool)
            or not isinstance(lat, Real)
            or not math.isfinite(lat)
            or not -90 <= lat <= 90
        ):
            raise ValueError(f"cell {cell_id} has an invalid centroid_lat")
        if (
            isinstance(lon, bool)
            or not isinstance(lon, Real)
            or not math.isfinite(lon)
            or not -180 <= lon <= 180
        ):
            raise ValueError(f"cell {cell_id} has an invalid centroid_lon")

        ordered_cells.append(dict(cell))

    if seen_ids != set(range(num_cells)):
        raise ValueError("checkpoint cell_id values must be contiguous from 0 to num_cells-1")

    ordered_cells.sort(key=lambda cell: cell["cell_id"])
    return num_cells, ordered_cells
