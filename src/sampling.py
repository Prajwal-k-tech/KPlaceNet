"""Sampling weights for geographic-cell-aware training."""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


def cell_sampling_weights(cell_ids: Sequence[int], power: float = 1.0) -> list[float]:
    """Return inverse-frequency sample weights for cell labels.

    ``power=0`` gives uniform per-example weights. ``power=1`` gives each
    observed cell equal total weight; intermediate powers soften rebalancing.
    The returned weights are used by replacement sampling; ``power=0`` is not
    the same epoch policy as a shuffled, no-replacement loader. Evaluation data
    is not changed.
    """
    if len(cell_ids) == 0:
        raise ValueError("cell_ids must contain at least one label")
    if isinstance(power, bool) or not isinstance(power, (int, float)):
        raise ValueError("power must be a finite number in [0, 1]")
    power = float(power)
    if not math.isfinite(power) or not 0.0 <= power <= 1.0:
        raise ValueError("power must be a finite number in [0, 1]")

    labels: list[int] = []
    for label in cell_ids:
        if isinstance(label, bool) or not isinstance(label, int) or label < 0:
            raise ValueError("cell_ids must be non-negative integers")
        labels.append(label)

    counts = Counter(labels)
    return [counts[label] ** (-power) for label in labels]


def make_cell_weighted_sampler(cell_ids: Sequence[int], power: float = 1.0,
                               seed: int = 42):
    """Build a seeded PyTorch sampler drawing one epoch of examples with replacement."""
    import torch
    from torch.utils.data import WeightedRandomSampler

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    weights = torch.as_tensor(cell_sampling_weights(cell_ids, power), dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True, generator=generator
    )
