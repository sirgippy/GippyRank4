from __future__ import annotations

import numpy as np
import pytest

from gippyrank.posterior.display import DISPLAY_PROBABILITY_SCALE
from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    Team,
    game_evidence_pmf,
    game_factor,
    infer_posterior,
)
from gippyrank.posterior.game_evidence import (
    _display_pmf,
    game_evidence_summary,
    performance_grade,
    performance_percentile,
)


def likelihood() -> LikelihoodV1:
    beta = np.zeros(34)
    beta[0] = -50.0
    beta[10] = -50.0
    beta[11] = 50.0
    return LikelihoodV1(beta, scale=7.0, degrees_of_freedom=15.0)


def test_single_game_evidence_is_the_factor_to_focal_message() -> None:
    teams = [
        Team("focal", "Focal", "fbs", np.array([0.9, 0.1])),
        Team("opponent", "Opponent", "fbs", np.array([0.2, 0.8])),
    ]
    game = Game("g", "focal", "opponent", "fbs", "fbs", 31, 14)
    result = infer_posterior(teams, [game], likelihood(), tolerance=1e-12)
    evidence = game_evidence_pmf(game, "focal", teams, likelihood(), result)
    factor = game_factor(game, *teams, likelihood())
    expected = factor @ teams[1].prior
    expected /= expected.sum()
    assert np.allclose(evidence, expected)

    reversed_prior = [
        teams[0],
        Team("opponent", "Opponent", "fbs", np.array([0.8, 0.2])),
    ]
    reversed_result = infer_posterior(reversed_prior, [game], likelihood(), tolerance=1e-12)
    assert not np.allclose(
        evidence,
        game_evidence_pmf(game, "focal", reversed_prior, likelihood(), reversed_result),
    )


def test_focal_prior_is_not_a_direct_game_evidence_factor() -> None:
    game = Game("g", "focal", "opponent", "fbs", "fbs", 31, 14)
    opponent = Team("opponent", "Opponent", "fbs", np.array([0.2, 0.8]))
    first_teams = [Team("focal", "Focal", "fbs", np.array([0.95, 0.05])), opponent]
    second_teams = [Team("focal", "Focal", "fbs", np.array([0.05, 0.95])), opponent]
    first = infer_posterior(first_teams, [game], likelihood(), tolerance=1e-12)
    second = infer_posterior(second_teams, [game], likelihood(), tolerance=1e-12)
    assert np.allclose(
        game_evidence_pmf(game, "focal", first_teams, likelihood(), first),
        game_evidence_pmf(game, "focal", second_teams, likelihood(), second),
    )


def test_rematches_share_pair_cavity_but_receive_individual_ratings() -> None:
    teams = [
        Team("a", "A", "fbs", np.array([0.5, 0.5])),
        Team("b", "B", "fbs", np.array([0.5, 0.5])),
        Team("c", "C", "fbs", np.array([0.1, 0.9])),
    ]
    first = Game("first", "a", "b", "fbs", "fbs", 28, 14)
    second = Game("second", "a", "b", "fbs", "fbs", 7, 24)
    outside = Game("outside", "b", "c", "fbs", "fbs", 35, 7)
    result = infer_posterior(teams, [first, second, outside], likelihood(), tolerance=1e-12)
    assert result.bp_state is not None
    index = result.bp_state.pair_indices[tuple(sorted(("a", "b")))]
    assert result.bp_state.factors[index].game_ids == ("first", "second")
    cavity = result.bp_state.cavity_beliefs[index, "b"]
    assert np.isclose(cavity.sum(), 1.0)
    first_rating = game_evidence_pmf(first, "a", teams, likelihood(), result)
    second_rating = game_evidence_pmf(second, "a", teams, likelihood(), result)
    assert not np.allclose(first_rating, second_rating)


def test_cross_subdivision_focal_support_and_orientation() -> None:
    fbs = Team("fbs", "FBS", "fbs", np.full(3, 1 / 3))
    fcs = Team("fcs", "FCS", "fcs", np.full(4, 1 / 4))
    away_game = Game("away", "fbs", "fcs", "fbs", "fcs", 35, 7)
    home_game = Game("home", "fcs", "fbs", "fcs", "fbs", 7, 35)
    away_result = infer_posterior([fbs, fcs], [away_game], likelihood(), tolerance=1e-12)
    home_result = infer_posterior([fbs, fcs], [home_game], likelihood(), tolerance=1e-12)
    away = game_evidence_pmf(away_game, "fbs", [fbs, fcs], likelihood(), away_result)
    home = game_evidence_pmf(home_game, "fbs", [fbs, fcs], likelihood(), home_result)
    assert away.shape == home.shape == (3,)
    assert away[0] > away[-1]
    assert home[0] > home[-1]


def test_performance_display_and_percentile_are_snapshot_derived() -> None:
    summary = game_evidence_summary(np.array([0.7, 0.2, 0.1]))

    assert len(summary["display_pmf"]) == 40
    assert sum(summary["display_pmf"]) == 1000
    assert performance_percentile(1.0, [1.0, 2.0, 3.0]) == pytest.approx(83.3333333333)
    assert performance_percentile(2.0, [1.0, 2.0, 3.0]) == pytest.approx(50.0)


@pytest.mark.parametrize("rank_count", [1, 3, 40, 80, 138, 160])
def test_equal_width_display_bins_keep_uniform_rank_pmf_uniform(rank_count: int) -> None:
    display = _display_pmf(np.full(rank_count, 1 / rank_count))

    assert display == [DISPLAY_PROBABILITY_SCALE // 40] * 40


def test_equal_width_display_bins_handle_concentrated_endpoint_mass() -> None:
    first_rank = np.zeros(138)
    first_rank[0] = 1.0
    last_rank = np.zeros(138)
    last_rank[-1] = 1.0

    first_display = _display_pmf(first_rank)
    last_display = _display_pmf(last_rank)

    assert sum(first_display) == sum(last_display) == DISPLAY_PROBABILITY_SCALE
    assert first_display == list(reversed(last_display))
    assert first_display[-1] == last_display[0] == 0


def test_equal_width_display_bins_preserve_smooth_unimodal_shape() -> None:
    ranks = np.arange(138, dtype=float)
    source = np.exp(-0.5 * ((ranks - 67.5) / 24.0) ** 2)
    display = np.asarray(_display_pmf(source), dtype=float)
    peak = int(np.argmax(display))

    assert 0 < peak < len(display) - 1
    assert np.all(np.diff(display[: peak + 1]) >= 0)
    assert np.all(np.diff(display[peak:]) <= 0)
    assert np.isclose(display.sum() / DISPLAY_PROBABILITY_SCALE, 1.0)


def test_equal_width_display_bins_are_deterministic_for_random_pmf() -> None:
    pmf = np.random.default_rng(55).random(138)

    assert _display_pmf(pmf) == _display_pmf(pmf)


@pytest.mark.parametrize(
    ("percentile", "grade"),
    [(0, "F"), (9.99, "F"), (10, "D"), (29.99, "D"), (30, "C"), (70, "B"), (90, "A"), (100, "A")],
)
def test_performance_grade_uses_frozen_boundaries(percentile: float, grade: str) -> None:
    assert performance_grade(percentile) == grade
