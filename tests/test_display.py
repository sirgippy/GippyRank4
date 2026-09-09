from __future__ import annotations

from gippyrank.posterior.display import (
    DISPLAY_PROBABILITY_SCALE,
    quantize_display_probabilities,
)


def test_display_probability_encoding_is_deterministic_and_conserves_weight() -> None:
    first = quantize_display_probabilities([0.1, 0.2, 0.3, 0.4])
    second = quantize_display_probabilities([0.1, 0.2, 0.3, 0.4])

    assert first == second
    assert all(isinstance(value, int) for value in first)
    assert sum(first) == DISPLAY_PROBABILITY_SCALE
