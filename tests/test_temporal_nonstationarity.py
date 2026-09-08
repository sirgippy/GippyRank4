from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from gippyrank.research.temporal_nonstationarity import (
    CANDIDATE_NAMES,
    candidate_grid,
    game_age_days,
    recency_weight,
    select_recency_candidate,
    temper_factor,
)


def test_game_age_is_strictly_cutoff_safe() -> None:
    cutoff = datetime(2024, 9, 15, tzinfo=UTC)
    game = cutoff - timedelta(days=28)
    assert game_age_days(cutoff, game) == pytest.approx(28.0)
    with pytest.raises(ValueError, match="strictly before"):
        game_age_days(cutoff, cutoff)
    with pytest.raises(ValueError, match="strictly before"):
        game_age_days(cutoff, cutoff + timedelta(seconds=1))


def test_half_life_is_exact_and_monotone() -> None:
    cutoff = datetime(2024, 9, 15, tzinfo=UTC)
    assert recency_weight(cutoff, cutoff - timedelta(days=28), 28) == pytest.approx(0.5)
    values = [
        recency_weight(cutoff, cutoff - timedelta(days=age), 28)
        for age in (1, 7, 14, 28, 56)
    ]
    assert values == sorted(values, reverse=True)
    assert recency_weight(cutoff, cutoff - timedelta(days=28), None) == 1.0


def test_tempering_preserves_likelihood_support() -> None:
    factor = np.asarray([[0.0, 0.25], [1.0, 4.0]])
    tempered = temper_factor(factor, 0.5)
    assert tempered[0, 0] == 0.0
    assert np.all(tempered[factor > 0] > 0)
    assert np.array_equal(temper_factor(factor, 1.0), factor)


def test_candidate_grid_is_frozen() -> None:
    assert tuple(candidate.name for candidate in candidate_grid()) == CANDIDATE_NAMES
    assert tuple(candidate.half_life_days for candidate in candidate_grid()) == (
        None,
        14.0,
        28.0,
        56.0,
        112.0,
    )


def test_selection_uses_the_development_gate_only() -> None:
    metrics = {
        "Static V1": {
            "next_game_nll": 1.0,
            "next_game_mae": 10.0,
            "next_game_brier": 0.20,
        }
    }
    for index, name in enumerate(CANDIDATE_NAMES[1:], start=1):
        metrics[name] = {
            "next_game_nll": 0.985 - index * 0.001,
            "next_game_mae": 9.8 - index * 0.01,
            "next_game_brier": 0.2005,
            "seasons_nll_improved": 3,
            "worst_season_nll_delta": 0.01,
        }
    assert select_recency_candidate(metrics) == "R112"
