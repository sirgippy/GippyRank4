from __future__ import annotations

import numpy as np

from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    Team,
    game_evidence_pmf,
    game_factor,
    infer_posterior,
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
