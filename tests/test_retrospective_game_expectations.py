from __future__ import annotations

import numpy as np
import pytest

from gippyrank.posterior.engine import Game, LikelihoodV1, Team, infer_posterior
from gippyrank.posterior.predictive import (
    ScheduledGame,
    posterior_prediction_team,
    predict_game,
)
from gippyrank.posterior.retrospective import (
    RETROSPECTIVE_GAME_EXPECTATIONS_VERSION,
    build_retrospective_game_expectations,
)


def _likelihood() -> LikelihoodV1:
    beta = np.zeros(34)
    beta[0] = -50.0
    beta[10] = -50.0
    beta[11] = 50.0
    return LikelihoodV1(beta, scale=7.0, degrees_of_freedom=15.0)


def _metadata(game_ids: list[str]) -> dict[str, object]:
    return {
        "snapshot_id": "2026-weekly-example-context",
        "season": 2026,
        "snapshot_type": "weekly",
        "effective_cutoff": "2026-09-27T12:00:00+00:00",
        "included_game_ids": game_ids,
        "historical_likelihood_version": "V1",
    }


def _expectations(
    teams: list[Team], games: list[Game]
) -> dict[str, object]:
    posterior = infer_posterior(teams, games, _likelihood(), tolerance=1e-12)
    return build_retrospective_game_expectations(
        metadata=_metadata([game.game_id for game in games]),
        teams=teams,
        games=games,
        posterior=posterior,
        likelihood=_likelihood(),
        max_iterations=500,
        tolerance=1e-12,
        damping=0.35,
        workers=1,
    )


def test_held_out_score_changes_only_the_observed_margin() -> None:
    teams = [
        Team("a", "A", "fbs", np.array([0.7, 0.3])),
        Team("b", "B", "fbs", np.array([0.4, 0.6])),
        Team("c", "C", "fbs", np.array([0.2, 0.8])),
    ]
    original = [
        Game("target", "a", "b", "fbs", "fbs", 31, 14),
        Game("other", "b", "c", "fbs", "fbs", 28, 10),
    ]
    changed = [
        Game("target", "a", "b", "fbs", "fbs", 7, 35),
        original[1],
    ]

    first = _expectations(teams, original)["games"]["target"]
    second = _expectations(teams, changed)["games"]["target"]

    assert first["actual_home_margin"] == 17
    assert second["actual_home_margin"] == -28
    for field in (
        "expected_home_margin",
        "median_home_margin",
        "margin_interval_50",
        "margin_interval_80",
        "margin_interval_95",
        "display_distribution",
    ):
        assert second[field] == pytest.approx(first[field])
    assert second["observed_margin_percentile"] != pytest.approx(
        first["observed_margin_percentile"]
    )


def test_held_out_rematch_retains_the_other_pair_game() -> None:
    teams = [
        Team("a", "A", "fbs", np.array([0.5, 0.5])),
        Team("b", "B", "fbs", np.array([0.5, 0.5])),
        Team("c", "C", "fbs", np.array([0.1, 0.9])),
    ]
    first = Game("first", "a", "b", "fbs", "fbs", 35, 10)
    second = Game("second", "b", "a", "fbs", "fbs", 7, 31)
    outside = Game("outside", "b", "c", "fbs", "fbs", 28, 10)
    artifact = _expectations(teams, [first, second, outside])

    retained_posterior = infer_posterior(
        teams, [second, outside], _likelihood(), tolerance=1e-12
    )
    expected = predict_game(
        ScheduledGame("first", "a", "b", "fbs", "fbs"),
        posterior_prediction_team(teams[0], retained_posterior.pmfs),
        posterior_prediction_team(teams[1], retained_posterior.pmfs),
        _likelihood(),
    )
    record = artifact["games"]["first"]

    assert record["expected_home_margin"] == pytest.approx(
        expected.expected_home_margin
    )
    assert record["median_home_margin"] == pytest.approx(expected.median_home_margin)
    assert artifact["inference"]["rematch_handling"].startswith("retain every other")


def test_artifact_is_home_oriented_and_contains_exact_and_display_summaries() -> None:
    fbs = Team("fbs", "FBS", "fbs", np.array([1.0]))
    fcs = Team("fcs", "FCS", "fcs", np.array([1.0]))
    game = Game("cross", "fcs", "fbs", "fcs", "fbs", 7, 35)
    likelihood = _likelihood()
    posterior = infer_posterior([fbs, fcs], [game], likelihood, tolerance=1e-12)
    artifact = build_retrospective_game_expectations(
        metadata=_metadata([game.game_id]),
        teams=[fbs, fcs],
        games=[game],
        posterior=posterior,
        likelihood=likelihood,
        max_iterations=500,
        tolerance=1e-12,
        damping=0.35,
        workers=1,
    )
    record = artifact["games"]["cross"]

    assert artifact["retrospective_game_expectations_version"] == (
        RETROSPECTIVE_GAME_EXPECTATIONS_VERSION
    )
    assert artifact["margin_orientation"] == record["margin_orientation"] == "home_minus_away"
    assert record["actual_home_margin"] == -28
    assert record["lower_tail_probability"] == pytest.approx(
        record["observed_margin_percentile"]
    )
    assert record["lower_tail_probability"] + record["upper_tail_probability"] == pytest.approx(1)
    assert len(record["display_distribution"]["masses"]) == 40
    assert sum(record["display_distribution"]["masses"]) + record["display_distribution"]["lower_tail_probability"] + record["display_distribution"]["upper_tail_probability"] == 1000


def test_publishing_expectations_does_not_mutate_production_posterior() -> None:
    teams = [
        Team("a", "A", "fbs", np.array([0.6, 0.4])),
        Team("b", "B", "fbs", np.array([0.4, 0.6])),
        Team("c", "C", "fbs", np.array([0.3, 0.7])),
    ]
    games = [
        Game("ab", "a", "b", "fbs", "fbs", 28, 14),
        Game("bc", "b", "c", "fbs", "fbs", 17, 21),
    ]
    posterior = infer_posterior(teams, games, _likelihood(), tolerance=1e-12)
    before = {team_id: pmf.copy() for team_id, pmf in posterior.pmfs.items()}

    build_retrospective_game_expectations(
        metadata=_metadata([game.game_id for game in games]),
        teams=teams,
        games=games,
        posterior=posterior,
        likelihood=_likelihood(),
        max_iterations=500,
        tolerance=1e-12,
        damping=0.35,
        workers=1,
    )

    for team_id, pmf in before.items():
        assert np.array_equal(posterior.pmfs[team_id], pmf)
