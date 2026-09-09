from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t as student_t

from gippyrank.posterior.engine import LikelihoodV1, Team
from gippyrank.posterior.predictive import (
    ScheduledGame,
    predict_game,
    predictive_components,
    win_probabilities,
)


def _teams(
    home_pmf: list[float],
    away_pmf: list[float],
    *,
    home_subdivision: str = "fbs",
    away_subdivision: str = "fbs",
) -> tuple[Team, Team]:
    return (
        Team("home", "Home", home_subdivision, np.asarray(home_pmf, dtype=float)),
        Team("away", "Away", away_subdivision, np.asarray(away_pmf, dtype=float)),
    )


def _game(
    home_subdivision: str = "fbs",
    away_subdivision: str = "fbs",
    *,
    neutral: bool = False,
) -> ScheduledGame:
    return ScheduledGame(
        "game",
        "home",
        "away",
        home_subdivision,
        away_subdivision,
        neutral,
    )


def test_delta_pmfs_reduce_to_the_corresponding_student_t_game_distribution() -> None:
    beta = np.zeros(34)
    beta[0] = 6.0
    likelihood = LikelihoodV1(beta, 2.0, 15.0)
    home, away = _teams([1.0, 0.0], [0.0, 1.0])

    summary = predict_game(_game(), home, away, likelihood)
    location = 6.0 * (0.25 - 0.75)

    assert summary.expected_home_margin == pytest.approx(location)
    assert summary.median_home_margin == pytest.approx(location)
    assert summary.home_win_probability == pytest.approx(
        1.0 - student_t.cdf((0.0 - location) / 2.0, 15.0)
    )
    assert summary.away_win_probability == pytest.approx(
        1.0 - summary.home_win_probability
    )
    assert summary.margin_interval_80 == pytest.approx(
        (
            location + 2.0 * student_t.ppf(0.10, 15.0),
            location + 2.0 * student_t.ppf(0.90, 15.0),
        ),
        abs=1e-9,
    )


def test_posterior_uncertainty_is_marginalized_not_replaced_by_expected_rank() -> None:
    beta = np.zeros(34)
    beta[3] = 100.0  # nonlinear d * abs(d) same-subdivision surface term
    likelihood = LikelihoodV1(beta, 1.0, 15.0)
    home, away = _teams([0.8, 0.2], [0.2, 0.8])
    game = _game()

    locations, weights = predictive_components(game, home, away, likelihood)
    summary = predict_game(game, home, away, likelihood)
    expected_rank_location = 100.0 * (-0.3) * abs(-0.3)

    assert summary.expected_home_margin == pytest.approx(
        float(np.dot(weights, locations))
    )
    assert summary.expected_home_margin == pytest.approx(-15.0)
    assert summary.expected_home_margin != pytest.approx(expected_rank_location)
    assert len(locations) == 4
    assert weights.sum() == pytest.approx(1.0)


def test_win_probability_uses_explicit_half_tie_convention() -> None:
    home, away = win_probabilities(0.2, 0.1)

    assert home == pytest.approx(0.75)
    assert away == pytest.approx(0.25)
    assert home + away == pytest.approx(1.0)


def test_canonical_home_orientation_reorients_for_the_away_page() -> None:
    beta = np.zeros(34)
    beta[0] = 10.0
    likelihood = LikelihoodV1(beta, 2.0, 15.0)
    home, away = _teams([1.0, 0.0], [0.0, 1.0])
    summary = predict_game(_game(neutral=True), home, away, likelihood)

    assert summary.home_win_probability + summary.away_win_probability == pytest.approx(
        1.0
    )
    away_expected_margin = -summary.expected_home_margin
    away_interval = (-summary.margin_interval_80[1], -summary.margin_interval_80[0])
    assert away_expected_margin == pytest.approx(5.0)
    assert away_interval[0] < away_interval[1]


def test_home_away_neutral_and_fbs_fcs_site_semantics() -> None:
    beta = np.zeros(34)
    beta[8] = 4.0  # same-subdivision non-neutral home effect
    beta[23] = 6.0  # FBS-home cross-subdivision effect
    beta[24] = -2.0  # FCS-home model-oriented effect
    likelihood = LikelihoodV1(beta, 1.0, 15.0)

    fbs_home, fbs_away = _teams([1.0], [1.0])
    assert predict_game(
        _game(), fbs_home, fbs_away, likelihood
    ).expected_home_margin == pytest.approx(4.0)
    assert predict_game(
        _game(neutral=True), fbs_home, fbs_away, likelihood
    ).expected_home_margin == pytest.approx(0.0)

    fbs, fcs = _teams([1.0], [1.0], away_subdivision="fcs")
    assert predict_game(
        _game(away_subdivision="fcs"), fbs, fcs, likelihood
    ).expected_home_margin == pytest.approx(6.0)
    fcs_home, fbs_away = _teams(
        [1.0], [1.0], home_subdivision="fcs", away_subdivision="fbs"
    )
    assert predict_game(
        _game("fcs", "fbs"), fcs_home, fbs_away, likelihood
    ).expected_home_margin == pytest.approx(2.0)
    assert predict_game(
        _game("fcs", "fbs", neutral=True), fcs_home, fbs_away, likelihood
    ).expected_home_margin == pytest.approx(0.0)
