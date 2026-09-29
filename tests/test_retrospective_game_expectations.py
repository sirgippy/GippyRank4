from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from gippyrank.methodology import (
    HISTORICAL_LIKELIHOOD_SHA256,
    RETROSPECTIVE_CONDITIONING,
)
from gippyrank.posterior import predictive, retrospective
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


def _expectations(teams: list[Team], games: list[Game]) -> dict:
    posterior = infer_posterior(teams, games, _likelihood(), tolerance=1e-12)
    return build_retrospective_game_expectations(
        metadata=_metadata([game.game_id for game in games]),
        teams=teams,
        games=games,
        posterior=posterior,
        likelihood=_likelihood(),
    )


def test_completed_game_uses_the_selected_full_posterior() -> None:
    teams = [
        Team("a", "A", "fbs", np.array([0.7, 0.3])),
        Team("b", "B", "fbs", np.array([0.4, 0.6])),
        Team("c", "C", "fbs", np.array([0.2, 0.8])),
    ]
    games = [
        Game("target", "a", "b", "fbs", "fbs", 31, 14),
        Game("other", "b", "c", "fbs", "fbs", 28, 10),
    ]
    posterior = infer_posterior(teams, games, _likelihood(), tolerance=1e-12)
    before = {team_id: pmf.copy() for team_id, pmf in posterior.pmfs.items()}
    artifact = build_retrospective_game_expectations(
        metadata=_metadata([game.game_id for game in games]),
        teams=teams,
        games=games,
        posterior=posterior,
        likelihood=_likelihood(),
    )
    expected = predict_game(
        ScheduledGame("target", "a", "b", "fbs", "fbs"),
        posterior_prediction_team(teams[0], posterior.pmfs),
        posterior_prediction_team(teams[1], posterior.pmfs),
        _likelihood(),
    )
    record = artifact["games"]["target"]
    assert record["expected_home_margin"] == pytest.approx(
        expected.expected_home_margin
    )
    assert record["median_home_margin"] == pytest.approx(expected.median_home_margin)
    assert (
        artifact["inference"]["implementation"]
        == "selected_snapshot_full_posterior_pmfs"
    )
    assert artifact["inference"]["games_evaluated"] == 2
    assert artifact["historical_likelihood_sha256"] == HISTORICAL_LIKELIHOOD_SHA256
    assert artifact["conditioning"] == RETROSPECTIVE_CONDITIONING
    assert "excluded_evidence" not in artifact["inference"]
    for team_id, pmf in before.items():
        assert np.array_equal(posterior.pmfs[team_id], pmf)


def test_game_result_can_change_its_own_hindsight_distribution() -> None:
    teams = [
        Team("a", "A", "fbs", np.array([0.7, 0.3])),
        Team("b", "B", "fbs", np.array([0.4, 0.6])),
        Team("c", "C", "fbs", np.array([0.2, 0.8])),
    ]
    other = Game("other", "b", "c", "fbs", "fbs", 28, 10)
    first = _expectations(
        teams, [Game("target", "a", "b", "fbs", "fbs", 31, 14), other]
    )["games"]["target"]
    second = _expectations(
        teams, [Game("target", "a", "b", "fbs", "fbs", 7, 35), other]
    )["games"]["target"]
    assert first["actual_home_margin"] == 17
    assert second["actual_home_margin"] == -28
    assert first["expected_home_margin"] != pytest.approx(
        second["expected_home_margin"]
    )
    assert first["observed_margin_percentile"] != pytest.approx(
        second["observed_margin_percentile"]
    )


def test_artifact_is_home_oriented_and_contains_exact_and_display_summaries() -> None:
    fbs = Team("fbs", "FBS", "fbs", np.array([1.0]))
    fcs = Team("fcs", "FCS", "fcs", np.array([1.0]))
    game = Game("cross", "fcs", "fbs", "fcs", "fbs", 7, 35)
    artifact = _expectations([fbs, fcs], [game])
    record = artifact["games"]["cross"]
    assert (
        artifact["retrospective_game_expectations_version"]
        == RETROSPECTIVE_GAME_EXPECTATIONS_VERSION
    )
    assert (
        artifact["margin_orientation"]
        == record["margin_orientation"]
        == "home_minus_away"
    )
    assert record["actual_home_margin"] == -28
    assert type(record["actual_home_margin"]) is int
    assert record["lower_tail_probability"] == pytest.approx(
        record["observed_margin_percentile"]
    )
    assert record["lower_tail_probability"] + record[
        "upper_tail_probability"
    ] == pytest.approx(1)
    display = record["display_distribution"]
    assert len(display["masses"]) == 40
    assert (
        sum(display["masses"])
        + display["lower_tail_probability"]
        + display["upper_tail_probability"]
        == 1000
    )


def test_full_posterior_mixture_is_built_once_per_game() -> None:
    teams = [
        Team("a", "A", "fbs", np.array([0.7, 0.3])),
        Team("b", "B", "fbs", np.array([0.4, 0.6])),
    ]
    games = [Game("target", "a", "b", "fbs", "fbs", 31, 14)]
    posterior = infer_posterior(teams, games, _likelihood(), tolerance=1e-12)
    with (
        patch.object(
            retrospective,
            "predictive_components",
            wraps=retrospective.predictive_components,
        ) as components,
        patch.object(
            predictive,
            "predictive_components",
            wraps=predictive.predictive_components,
        ) as repeated_components,
    ):
        build_retrospective_game_expectations(
            metadata=_metadata(["target"]),
            teams=teams,
            games=games,
            posterior=posterior,
            likelihood=_likelihood(),
        )
    assert components.call_count == 1
    repeated_components.assert_not_called()
