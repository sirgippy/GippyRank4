from __future__ import annotations

import csv
import gzip
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gippyrank.performance_v1 import infer_performance
from gippyrank.posterior.engine import Game, LikelihoodV1, Team, game_factor
from gippyrank.quality_of_record import binary_game_factor, infer_quality_of_record
from gippyrank.research.issue_242_quality_of_results import (
    MARGIN_WEIGHTS,
    _published_inputs,
    _write_pmf_artifact,
)
from gippyrank.research.quality_of_results import (
    blended_game_factor,
    infer_quality_of_results,
    validate_margin_weight,
)


def _likelihood() -> LikelihoodV1:
    beta = np.zeros(34)
    beta[0] = -30.0
    beta[10], beta[11] = -30.0, 30.0
    return LikelihoodV1(beta, scale=7.0, degrees_of_freedom=15.0)


def _teams() -> list[Team]:
    return [
        Team("a", "A", "fbs", np.array([0.6, 0.3, 0.1])),
        Team("b", "B", "fbs", np.array([0.2, 0.4, 0.4])),
        Team("c", "C", "fbs", np.array([0.3, 0.4, 0.3])),
        Team("idle", "Idle", "fbs", np.array([0.7, 0.2, 0.1])),
        Team("fcs", "FCS", "fcs", np.array([0.5, 0.5])),
    ]


def _games() -> list[Game]:
    return [
        Game("ab", "a", "b", "fbs", "fbs", 24, 21),
        Game("bc", "b", "c", "fbs", "fbs", 17, 10),
        Game("ca", "c", "a", "fbs", "fbs", 7, 10, neutral_site=True),
        Game("fcs-a", "fcs", "a", "fcs", "fbs", 10, 17, neutral_site=True),
    ]


def test_blended_factor_endpoints_reuse_existing_factors_exactly() -> None:
    teams = {team.team_id: team for team in _teams()}
    likelihood = _likelihood()
    for game in _games():
        home, away = teams[game.home_id], teams[game.away_id]
        assert np.array_equal(
            blended_game_factor(game, home, away, likelihood, 0.0),
            binary_game_factor(game, home, away, likelihood),
        )
        assert np.array_equal(
            blended_game_factor(game, home, away, likelihood, 1.0),
            game_factor(game, home, away, likelihood),
        )


def test_intermediate_factor_is_geometric_blend_of_existing_factors() -> None:
    teams = {team.team_id: team for team in _teams()}
    game = _games()[0]
    likelihood = _likelihood()
    outcome = binary_game_factor(
        game, teams[game.home_id], teams[game.away_id], likelihood
    )
    margin = game_factor(game, teams[game.home_id], teams[game.away_id], likelihood)
    expected = np.sqrt(outcome * margin)
    expected /= expected.max()
    actual = blended_game_factor(
        game, teams[game.home_id], teams[game.away_id], likelihood, 0.5
    )
    assert np.allclose(actual, expected, atol=1e-14, rtol=0)
    assert np.isclose(actual.max(), 1.0)


@pytest.mark.parametrize("weight", [-0.01, 1.01, float("nan"), float("inf")])
def test_weight_must_be_finite_and_in_unit_interval(weight: float) -> None:
    with pytest.raises(ValueError, match="margin_weight"):
        validate_margin_weight(weight)


def test_tempered_endpoint_networks_match_quality_of_record_and_play() -> None:
    teams = _teams()
    games = _games()
    likelihood = _likelihood()
    qor = infer_quality_of_record(teams, games, likelihood)
    qop = infer_performance(
        teams,
        games,
        likelihood,
        anchor_family="context",
    )

    result_qor = infer_quality_of_results(teams, games, likelihood, 0.0)
    result_qop = infer_quality_of_results(teams, games, likelihood, 1.0)
    assert result_qor.blended_context_posterior.converged
    assert result_qop.blended_context_posterior.converged
    for team in teams:
        if team.subdivision != "fbs":
            continue
        assert np.allclose(
            result_qor.pmfs[team.team_id], qor.pmfs[team.team_id], atol=1e-12
        )
        assert np.allclose(
            result_qop.pmfs[team.team_id], qop.pmfs[team.team_id], atol=1e-12
        )
    assert result_qor.eligible_game_counts["idle"] == 0
    assert np.array_equal(result_qor.pmfs["idle"], np.full(3, 1 / 3))
    assert np.array_equal(result_qop.pmfs["idle"], np.full(3, 1 / 3))


def test_tie_only_rejects_models_that_use_binary_winner_evidence() -> None:
    teams = {team.team_id: team for team in _teams()}
    tied = replace(_games()[0], home_points=21, away_points=21)
    likelihood = _likelihood()
    assert np.array_equal(
        blended_game_factor(tied, teams["a"], teams["b"], likelihood, 1.0),
        game_factor(tied, teams["a"], teams["b"], likelihood),
    )
    with pytest.raises(ValueError, match="no binary winner"):
        blended_game_factor(tied, teams["a"], teams["b"], likelihood, 0.5)


def test_official_weekly_inputs_use_repaired_campbell_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    snapshots = _published_inputs(root)
    assert len(snapshots) == 8
    week_two = next(
        snapshot for snapshot in snapshots if snapshot.slot_id == "2026-09-08"
    )
    metadata = json.loads((week_two.context_path / "metadata.json").read_text())
    correction = metadata["historical_evidence_correction"]
    assert correction["issue"] == "238"
    assert correction["excluded_games"][0]["game_id"] == "401866625"
    for snapshot in snapshots:
        with (snapshot.context_path / "included_games.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            game_ids = {row["id"] for row in csv.DictReader(handle)}
        assert "401866625" not in game_ids
        assert "401917058" in game_ids


def test_pmf_gzip_artifacts_have_deterministic_headers_and_bytes(
    tmp_path: Path,
) -> None:
    pmf = np.array([0.25, 0.75])
    per_weight = {weight: {"pmfs": {"team-a": pmf}} for weight in MARGIN_WEIGHTS}
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"

    _write_pmf_artifact(first, per_weight)
    _write_pmf_artifact(second, per_weight)

    first_bytes = first.read_bytes()
    assert first_bytes == second.read_bytes()
    assert first_bytes[4:8] == b"\0\0\0\0"
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        assert handle.readline() == "margin_weight,team_id,rank,probability\n"
