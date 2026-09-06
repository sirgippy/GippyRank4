from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gippyrank.performance_v1 import (
    explicit_neutralized_target,
    infer_performance,
    performance_pmf_summaries,
    pmf_tv_distance,
    remove_focal_prior,
    uniform_pmf,
)
from gippyrank.posterior.engine import Game, LikelihoodV1, Team, game_factor
from gippyrank.posterior.snapshots import load_likelihood


def likelihood() -> LikelihoodV1:
    beta = np.zeros(34)
    beta[0] = -50.0
    beta[10], beta[11] = -50.0, 50.0
    return LikelihoodV1(beta, scale=7.0, degrees_of_freedom=15.0)


def frozen_likelihood() -> LikelihoodV1:
    return load_likelihood(
        Path("data/processed/posterior/historical_likelihood_v1.json")
    )


def expected(pmf: np.ndarray) -> float:
    return float(np.dot(np.arange(1, len(pmf) + 1), pmf))


def one_game(
    margin: int,
    *,
    focal_prior: np.ndarray | None = None,
    opponent_prior: np.ndarray | None = None,
    focal_home: bool = True,
) -> np.ndarray:
    focal_prior = uniform_pmf(30) if focal_prior is None else focal_prior
    opponent_prior = uniform_pmf(30) if opponent_prior is None else opponent_prior
    focal = Team("focal", "Focal", "fbs", focal_prior)
    opponent = Team("opponent", "Opponent", "fbs", opponent_prior)
    game = (
        Game("g", "focal", "opponent", "fbs", "fbs", 30 + margin, 30)
        if focal_home
        else Game("g", "opponent", "focal", "fbs", "fbs", 30, 30 + margin)
    )
    return infer_performance(
        [focal, opponent], [game], likelihood(), anchor_family="context"
    ).pmfs["focal"]


def test_prior_removal_is_the_likelihood_profile() -> None:
    teams = [
        Team("a", "A", "fbs", np.array([0.8, 0.2])),
        Team("b", "B", "fbs", np.array([0.3, 0.7])),
    ]
    game = Game("g", "a", "b", "fbs", "fbs", 31, 14)
    factor = game_factor(game, *teams, likelihood())
    posterior = factor * teams[0].prior[:, None] * teams[1].prior[None, :]
    posterior = posterior.sum(axis=1)
    posterior /= posterior.sum()
    expected_profile = factor @ teams[1].prior
    expected_profile /= expected_profile.sum()
    assert np.allclose(remove_focal_prior(posterior, teams[0].prior), expected_profile)


def test_explicit_neutralization_matches_stripping_on_a_tree() -> None:
    teams = [
        Team("a", "A", "fbs", np.array([0.8, 0.2])),
        Team("b", "B", "fbs", np.array([0.3, 0.7])),
    ]
    game = Game("g", "a", "b", "fbs", "fbs", 31, 14)
    inferred = infer_performance(
        teams, [game], likelihood(), anchor_family="context", tolerance=1e-12
    )
    explicit, result = explicit_neutralized_target(
        teams, [game], likelihood(), "a", tolerance=1e-12
    )
    assert result.converged
    assert pmf_tv_distance(inferred.pmfs["a"], explicit) <= 1e-12


def test_focal_prior_independence_holds_for_final_performance() -> None:
    focal_a_prior = np.linspace(1, 30, 30)
    focal_a_prior /= focal_a_prior.sum()
    focal_b_prior = focal_a_prior[::-1]
    first = one_game(7, focal_prior=focal_a_prior)
    second = one_game(7, focal_prior=focal_b_prior)
    assert pmf_tv_distance(first, second) <= 1e-12


def test_explicit_neutralization_is_prior_independent_on_a_cycle() -> None:
    first_prior = np.linspace(1, 30, 30)
    first_prior /= first_prior.sum()
    second_prior = first_prior[::-1]
    games = [
        Game("ab", "a", "b", "fbs", "fbs", 31, 24),
        Game("bc", "b", "c", "fbs", "fbs", 21, 17),
        Game("ca", "c", "a", "fbs", "fbs", 28, 27),
    ]
    common = [
        Team("b", "B", "fbs", uniform_pmf(30)),
        Team("c", "C", "fbs", uniform_pmf(30)),
    ]
    first, _ = explicit_neutralized_target(
        [Team("a", "A", "fbs", first_prior), *common],
        games,
        likelihood(),
        "a",
        tolerance=1e-12,
    )
    second, _ = explicit_neutralized_target(
        [Team("a", "A", "fbs", second_prior), *common],
        games,
        likelihood(),
        "a",
        tolerance=1e-12,
    )
    assert pmf_tv_distance(first, second) <= 1e-12


def test_zero_game_performance_is_uniform() -> None:
    focal_prior = np.array([0.8, 0.2])
    result = infer_performance(
        [Team("focal", "Focal", "fbs", focal_prior)],
        [],
        likelihood(),
        anchor_family="context",
    )
    assert np.allclose(result.pmfs["focal"], [0.5, 0.5])


def test_margin_monotonicity_and_win_loss_continuity() -> None:
    pmfs = [one_game(margin) for margin in (-1, 0, 1)]
    expected_ranks = [expected(pmf) for pmf in pmfs]
    assert expected_ranks[0] >= expected_ranks[1] >= expected_ranks[2]
    assert (
        max(
            abs(expected_ranks[index + 1] - expected_ranks[index]) for index in range(2)
        )
        < 1.0
    )


def test_opponent_strength_monotonicity() -> None:
    strong = np.exp(-np.arange(30) / 7.0)
    strong /= strong.sum()
    weak = strong[::-1]
    assert expected(one_game(7, opponent_prior=strong)) <= expected(
        one_game(7, opponent_prior=weak)
    )


def test_road_performance_is_at_least_as_strong_as_home_performance() -> None:
    home = one_game(7, focal_home=True)
    road = one_game(7, focal_home=False)
    assert expected(road) <= expected(home)


def test_summaries_include_mode_and_all_requested_intervals() -> None:
    summary = performance_pmf_summaries(np.array([0.1, 0.6, 0.3]))
    assert summary["mode_rank"] == 2
    for level in (50, 80, 95):
        assert f"interval_{level}_low" in summary
        assert f"interval_{level}_high" in summary


def test_frozen_likelihood_is_student_t_v1() -> None:
    value = json.loads(
        Path("data/processed/posterior/historical_likelihood_v1.json").read_text()
    )
    assert frozen_likelihood().degrees_of_freedom == 15.0
    assert value["artifact_kind"] == "historical_likelihood_v1_coefficients"
