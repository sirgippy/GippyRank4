"""Presentation-only probability encoding for compact static artifacts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DISPLAY_PROBABILITY_SCALE = 1000


def quantize_display_probabilities(
    probabilities: Sequence[float] | np.ndarray,
    *,
    scale: int = DISPLAY_PROBABILITY_SCALE,
) -> list[int]:
    """Convert probabilities to deterministic integer weights summing to ``scale``."""
    if isinstance(scale, bool) or scale < 1:
        raise ValueError("display probability scale must be a positive integer")
    values = np.asarray(probabilities, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError("display probabilities must be a finite nonnegative vector")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("display probabilities must have positive mass")
    scaled = values / total * scale
    weights = np.floor(scaled).astype(np.int64)
    remainder = int(scale - weights.sum())
    fractional = scaled - weights
    if remainder > 0:
        order = np.argsort(-fractional, kind="stable")
        weights[order[:remainder]] += 1
    elif remainder < 0:
        order = np.argsort(fractional, kind="stable")
        for index in order:
            if remainder == 0:
                break
            if weights[index] > 0:
                weights[index] -= 1
                remainder += 1
    if int(weights.sum()) != scale or np.any(weights < 0):
        raise FloatingPointError("display probability quantization failed")
    return [int(value) for value in weights]
