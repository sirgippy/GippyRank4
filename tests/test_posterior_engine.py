from __future__ import annotations

import itertools

import numpy as np
import pytest

from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    Team,
    game_factor,
    infer_posterior,
)


def likelihood() -> LikelihoodV1:
    # In the frozen same-subdivision basis beta[0] multiplies rank-percentile
    # difference. Lower rank is better, hence its deliberately negative sign.
    beta = np.zeros(34)
    beta[0] = -50.0
    # The cross-subdivision block starts after the nine FBS/FBS columns and
    # keeps stable (FBS, FCS) coordinates regardless of the schedule order.
    beta[10] = -50.0
    beta[11] = 50.0
    return LikelihoodV1(beta, scale=7.0, degrees_of_freedom=15.0)


def test_two_team_message_passing_matches_exact_enumeration() -> None:
    teams = [
        Team("a", "A", "fbs", np.array([0.8, 0.2])),
        Team("b", "B", "fbs", np.array([0.3, 0.7])),
    ]
    game = Game("g", "a", "b", "fbs", "fbs", 31, 14)
    factor = game_factor(game, *teams, likelihood())
    joint = factor * teams[0].prior[:, None] * teams[1].prior[None, :]
    joint /= joint.sum()
    result = infer_posterior(teams, [game], likelihood(), tolerance=1e-12)
    assert result.converged
    assert np.allclose(result.pmfs["a"], joint.sum(axis=1), atol=1e-10)
    assert np.allclose(result.pmfs["b"], joint.sum(axis=0), atol=1e-10)


def test_three_team_round_robin_matches_exact_enumeration() -> None:
    teams = [Team(str(i), str(i), "fbs", np.array([0.5, 0.5])) for i in range(3)]
    games = [
        Game("ab", "0", "1", "fbs", "fbs", 49, 0),
        Game("bc", "1", "2", "fbs", "fbs", 49, 0),
        Game("ca", "2", "0", "fbs", "fbs", 49, 0),
    ]
    factors = [
        game_factor(g, teams[int(g.home_id)], teams[int(g.away_id)], likelihood())
        for g in games
    ]
    exact = np.zeros((3, 2))
    weights = []
    for state in itertools.product(range(2), repeat=3):
        weight = np.prod([team.prior[state[i]] for i, team in enumerate(teams)])
        weight *= (
            factors[0][state[0], state[1]]
            * factors[1][state[1], state[2]]
            * factors[2][state[2], state[0]]
        )
        weights.append((state, weight))
    total = sum(weight for _, weight in weights)
    for state, weight in weights:
        for team, rank in enumerate(state):
            exact[team, rank] += weight / total
    result = infer_posterior(
        teams, games, likelihood(), tolerance=1e-6, max_iterations=2_000
    )
    assert result.converged
    # The three-node cycle is deliberately an approximation; its predeclared
    # toy tolerance is TV <= 0.03, much tighter than useful weekly uncertainty.
    for team in range(3):
        assert 0.5 * np.abs(result.pmfs[str(team)] - exact[team]).sum() <= 0.03


def test_opponent_evidence_propagates_through_chain() -> None:
    teams = [
        Team("a", "A", "fbs", np.array([0.5, 0.5])),
        Team("b", "B", "fbs", np.array([0.5, 0.5])),
        Team("c", "C", "fbs", np.array([0.05, 0.95])),
    ]
    ab = Game("ab", "a", "b", "fbs", "fbs", 28, 7)
    bc = Game("bc", "b", "c", "fbs", "fbs", 28, 7)
    one = infer_posterior(teams, [ab], likelihood())
    chain = infer_posterior(teams, [ab, bc], likelihood())
    assert chain.pmfs["a"][0] > one.pmfs["a"][0]


def test_disconnected_team_retains_its_prior() -> None:
    teams = [
        Team("a", "A", "fbs", np.array([0.5, 0.5])),
        Team("b", "B", "fbs", np.array([0.5, 0.5])),
        Team("c", "C", "fbs", np.array([0.2, 0.8])),
    ]
    result = infer_posterior(
        teams, [Game("ab", "a", "b", "fbs", "fbs", 30, 10)], likelihood()
    )
    assert np.allclose(result.pmfs["c"], teams[2].prior)
    assert all(np.isclose(pmf.sum(), 1.0) for pmf in result.pmfs.values())


def test_cross_subdivision_game_is_a_real_factor() -> None:
    fbs = Team("fbs", "FBS", "fbs", np.array([0.5, 0.5]))
    fcs = Team("fcs", "FCS", "fcs", np.array([0.5, 0.5]))
    result = infer_posterior(
        [fbs, fcs], [Game("x", "fbs", "fcs", "fbs", "fcs", 49, 0)], likelihood()
    )
    assert result.pmfs["fbs"][0] > 0.5


def test_invalid_factor_references_are_rejected() -> None:
    with pytest.raises(ValueError, match="without a prior"):
        infer_posterior(
            [Team("a", "A", "fbs", np.array([0.5, 0.5]))],
            [Game("x", "a", "missing", "fbs", "fbs", 1, 0)],
            likelihood(),
        )
