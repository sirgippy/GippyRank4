"""Invariants for the research-only partial DB transfer-impact policies."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from study_partial_db_transfer_impact import (
    candidate_values,
    mask_sets,
    shift_pmf,
)


def test_partial_policies_preserve_known_contribution() -> None:
    values = candidate_values([2.0, -0.5, 1.0], 1, 0.25)
    assert values == {
        "all_or_nothing": 0.0,
        "observed": 2.5,
        "coverage_scaled": 10.0 / 3.0,
        "missing_mean": 2.75,
    }
    assert candidate_values([], 2, 0.25)["observed"] == 0.0


def test_high_coverage_masks_hide_each_player_once() -> None:
    assert mask_sets(8, 7) == [(i,) for i in range(8)]
    assert mask_sets(12, 6) == mask_sets(12, 6)


def test_downstream_translation_preserves_probability() -> None:
    original = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    np.testing.assert_allclose(shift_pmf(original, 0.0), original)
    shifted = shift_pmf(original, 0.25)
    np.testing.assert_allclose(shifted.sum(), 1.0)
    assert np.dot(np.arange(1, 6), shifted) > np.dot(np.arange(1, 6), original)
