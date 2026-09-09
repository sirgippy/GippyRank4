from __future__ import annotations

import json

import numpy as np
import pytest

from gippyrank.posterior.engine import LikelihoodV1, Team
from gippyrank.posterior.predictive import ScheduledGame
from gippyrank.posterior.season_simulation import (
    CompletedRecord,
    SeasonSimulationConfig,
    poisson_binomial_pmf,
    sample_latent_qualities,
    simulate_season,
    simulate_shared_game_outcomes,
)


def _teams(
    home_pmf: list[float], away_pmf: list[float]
) -> tuple[Team, Team]:
    return (
        Team("home", "Home", "fbs", np.asarray(home_pmf, dtype=float)),
        Team("away", "Away", "fbs", np.asarray(away_pmf, dtype=float)),
    )


def _games(*, season_type: str = "regular") -> list[ScheduledGame]:
    return [
        ScheduledGame(
            "game-1",
            "home",
            "away",
            "fbs",
            "fbs",
            season_type=season_type,
            date="2026-09-10T00:00:00Z",
        ),
        ScheduledGame(
            "game-2",
            "home",
            "away",
            "fbs",
            "fbs",
            season_type=season_type,
            date="2026-09-17T00:00:00Z",
        ),
    ]


def _likelihood(coefficient: float = 0.0, scale: float = 1.0) -> LikelihoodV1:
    beta = np.zeros(34)
    beta[0] = coefficient
    return LikelihoodV1(beta, scale, 15.0)


def test_poisson_binomial_pmf_is_normalized_and_exact() -> None:
    pmf = poisson_binomial_pmf([0.2, 0.7])

    assert pmf == pytest.approx([0.24, 0.62, 0.14])
    assert pmf.sum() == pytest.approx(1.0)


def test_latent_quality_is_sampled_once_per_team_and_replay_is_deterministic() -> None:
    teams = _teams([0.25, 0.75], [1.0, 0.0])
    posterior = {team.team_id: team.prior for team in teams}

    first = sample_latent_qualities(teams, posterior, outer_draw_count=500, seed=17)
    second = sample_latent_qualities(teams, posterior, outer_draw_count=500, seed=17)

    assert first.team_ids == ("away", "home")
    assert first.ranks.tobytes() == second.ranks.tobytes()
    assert np.all(first.for_team("away") == 1)
    assert set(first.for_team("home")) == {1, 2}
    assert first.outer_draw_count == 500


def test_shared_latent_quality_induces_cross_game_dependence() -> None:
    teams = _teams([0.5, 0.5], [1.0, 0.0])
    posterior = {team.team_id: team.prior for team in teams}
    _, outcomes = simulate_shared_game_outcomes(
        teams,
        posterior,
        _games(),
        _likelihood(20.0),
        outer_draw_count=3000,
        inner_rollout_count=8,
        seed=11,
    )

    correlation = float(
        np.corrcoef(outcomes["game-1"].ravel(), outcomes["game-2"].ravel())[0, 1]
    )
    assert correlation > 0.15


def test_degenerate_quality_leaves_only_independent_game_noise() -> None:
    teams = _teams([1.0, 0.0], [1.0, 0.0])
    posterior = {team.team_id: team.prior for team in teams}
    _, outcomes = simulate_shared_game_outcomes(
        teams,
        posterior,
        _games(),
        _likelihood(),
        outer_draw_count=3000,
        inner_rollout_count=8,
        seed=11,
    )

    correlation = float(
        np.corrcoef(outcomes["game-1"].ravel(), outcomes["game-2"].ravel())[0, 1]
    )
    assert abs(correlation) < 0.05


def test_season_summary_shifts_final_record_from_fixed_completed_record() -> None:
    teams = _teams([1.0], [1.0])
    posterior = {team.team_id: team.prior for team in teams}
    artifact = simulate_season(
        teams,
        posterior,
        _games(),
        {
            "home": CompletedRecord(wins=2, losses=1),
            "away": CompletedRecord(wins=1, losses=2),
        },
        _likelihood(),
        config=SeasonSimulationConfig(outer_draw_count=100, seed=4),
    )

    summary = artifact["teams"]["home"]
    assert summary["completed_wins"] == 2
    assert summary["remaining_games"] == 2
    assert summary["final_win_distribution"] == {
        "2": pytest.approx(0.25),
        "3": pytest.approx(0.5),
        "4": pytest.approx(0.25),
    }
    assert sum(summary["record_probabilities"].values()) == pytest.approx(1.0)
    assert summary["final_win_interval_95"] == [2, 4]
    variance = summary["variance_decomposition"]
    assert variance["team_quality"] == pytest.approx(0.0)
    assert variance["game_randomness"] == pytest.approx(0.5)
    assert variance["total"] == pytest.approx(
        variance["team_quality"] + variance["game_randomness"]
    )
    assert variance["team_quality_fraction"] + variance["game_randomness_fraction"] == pytest.approx(1.0)


def test_single_game_outer_marginal_matches_exact_v1_mixture() -> None:
    teams = _teams([0.2, 0.8], [0.7, 0.3])
    posterior = {team.team_id: team.prior for team in teams}
    artifact = simulate_season(
        teams,
        posterior,
        [_games()[0]],
        {},
        _likelihood(12.0),
        config=SeasonSimulationConfig(outer_draw_count=20_000, seed=8),
    )

    marginal = artifact["game_marginals"]["game-1"]
    assert marginal["outer_home_win_probability"] == pytest.approx(
        marginal["exact_home_win_probability"], abs=0.01
    )
    assert marginal["outer_expected_home_margin"] == pytest.approx(
        marginal["exact_expected_home_margin"], abs=0.15
    )


def test_unsupported_future_game_fails_closed_for_affected_team() -> None:
    teams = _teams([1.0], [1.0])
    posterior = {team.team_id: team.prior for team in teams}
    unsupported = ScheduledGame(
        "unsupported",
        "home",
        "unknown-fcs",
        "fbs",
        "fcs",
        date="2026-09-10T00:00:00Z",
    )
    artifact = simulate_season(
        teams,
        posterior,
        [unsupported],
        {},
        _likelihood(),
        config=SeasonSimulationConfig(outer_draw_count=10, seed=1),
    )

    summary = artifact["teams"]["home"]
    assert summary["forecast_status"] == "unavailable"
    assert "expected_final_wins" not in summary
    assert summary["unsupported_games"][0]["reason"] == "missing_posterior_team"
    assert artifact["season_scope"]["unsupported_behavior"] == "fail_closed"


def test_supported_fcs_game_contributes_to_the_fbs_summary() -> None:
    fbs = Team("fbs", "FBS", "fbs", np.array([1.0]))
    fcs = Team("fcs", "FCS", "fcs", np.array([1.0]))
    game = ScheduledGame("fcs-game", "fbs", "fcs", "fbs", "fcs")

    artifact = simulate_season(
        [fbs, fcs],
        {"fbs": fbs.prior, "fcs": fcs.prior},
        [game],
        {},
        _likelihood(),
        config=SeasonSimulationConfig(outer_draw_count=10, seed=1),
    )

    assert artifact["teams"]["fbs"]["remaining_games"] == 1
    assert artifact["teams"]["fbs"]["forecast_status"] == "available"
    assert "fcs" not in artifact["teams"]


def test_postseason_schedule_is_outside_v1_regular_season_scope() -> None:
    teams = _teams([1.0], [1.0])
    posterior = {team.team_id: team.prior for team in teams}
    artifact = simulate_season(
        teams,
        posterior,
        _games(season_type="postseason"),
        {},
        _likelihood(),
        config=SeasonSimulationConfig(outer_draw_count=10, seed=1),
    )

    assert artifact["season_scope"]["future_game_count"] == 0
    assert artifact["teams"]["home"]["remaining_games"] == 0
    assert artifact["teams"]["home"]["final_win_distribution"] == {"0": 1.0}


def test_season_artifact_is_byte_deterministic() -> None:
    teams = _teams([0.4, 0.6], [0.6, 0.4])
    posterior = {team.team_id: team.prior for team in teams}
    config = SeasonSimulationConfig(outer_draw_count=250, seed=29)
    first = simulate_season(teams, posterior, _games(), {}, _likelihood(8.0), config=config)
    second = simulate_season(teams, posterior, _games(), {}, _likelihood(8.0), config=config)

    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
