from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

import gippyrank.research.temporal_nonstationarity as temporal
from gippyrank.research.temporal_nonstationarity import (
    CANDIDATE_NAMES,
    HistoricalGame,
    PriorInput,
    TemporalCandidate,
    candidate_grid,
    development_gate_metrics,
    evaluate_candidates,
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


def test_selection_prefers_longer_half_life_within_nll_tie_band() -> None:
    metrics = {
        "Static V1": {
            "next_game_nll": 1.0,
            "next_game_mae": 10.0,
            "next_game_brier": 0.20,
        },
        "R14": {
            "next_game_nll": 1.0,
            "next_game_mae": 10.0,
            "next_game_brier": 0.20,
            "seasons_nll_improved": 0,
            "worst_season_nll_delta": 0.0,
        },
        "R28": {
            "next_game_nll": 1.0,
            "next_game_mae": 10.0,
            "next_game_brier": 0.20,
            "seasons_nll_improved": 0,
            "worst_season_nll_delta": 0.0,
        },
        "R56": {
            "next_game_nll": 0.980,
            "next_game_mae": 9.8,
            "next_game_brier": 0.2005,
            "seasons_nll_improved": 3,
            "worst_season_nll_delta": 0.01,
        },
        "R112": {
            "next_game_nll": 0.981,
            "next_game_mae": 9.8,
            "next_game_brier": 0.2005,
            "seasons_nll_improved": 3,
            "worst_season_nll_delta": 0.01,
        },
    }
    assert select_recency_candidate(metrics) == "R112"


def _synthetic_game(
    game_id: str, week: int, day: int, home_id: str, away_id: str
) -> HistoricalGame:
    return HistoricalGame(
        game_id=game_id,
        season=2020,
        start=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=day - 1),
        week=week,
        home_id=home_id,
        away_id=away_id,
        home_name=home_id.upper(),
        away_name=away_id.upper(),
        home_subdivision="fbs",
        away_subdivision="fbs",
        home_population=2,
        away_population=2,
        home_points=21,
        away_points=14,
        margin=7.0,
        neutral_site=False,
        home_ranks=np.array([1]),
        away_ranks=np.array([1]),
        rank_pairs=np.array([[1, 1]]),
    )


def test_evaluate_candidates_enforces_leakage_and_scoring_contracts(monkeypatch) -> None:
    games = [
        _synthetic_game("g1", 1, 1, "a", "b"),
        _synthetic_game("g2", 2, 8, "b", "c"),
        _synthetic_game("g3", 3, 15, "a", "c"),
        _synthetic_game("g4", 4, 22, "a", "b"),
        _synthetic_game("g5", 5, 29, "b", "c"),
        _synthetic_game("g6", 6, 36, "a", "c"),
    ]
    context_pmfs = {
        "a": np.array([0.8, 0.2]),
        "b": np.array([0.3, 0.7]),
        "c": np.array([0.4, 0.6]),
    }
    priors = {
        (2020, team_id): PriorInput(
            2020, team_id, team_id.upper(), "fbs", 2, pmf, "Context"
        )
        for team_id, pmf in context_pmfs.items()
    }
    candidates = (
        TemporalCandidate("Static V1", None),
        TemporalCandidate("R14", 14.0),
    )
    calls = []

    monkeypatch.setattr(temporal, "game_factor", lambda *args: np.ones((2, 2)))

    def fake_infer(teams, engine_games, likelihood, **kwargs):
        calls.append(
            {
                "game_ids": [game.game_id for game in engine_games],
                "teams": {team.team_id: team.prior.copy() for team in teams},
            }
        )
        return SimpleNamespace(
            converged=True,
            pmfs={team.team_id: team.prior.copy() for team in teams},
        )

    monkeypatch.setattr(temporal, "infer_posterior", fake_infer)
    monkeypatch.setattr(
        temporal,
        "predictive_scores",
        lambda game, posterior, likelihood, surface_cache: {
            "margin_nll": 1.0,
            "margin_mae": 1.0,
            "win_brier": 0.1,
            "expected_margin": 1.0,
            "win_probability": 0.5,
            "actual_margin": game.margin,
        },
    )

    rows, _ = evaluate_candidates(
        games,
        priors,
        likelihood=SimpleNamespace(),
        candidates=candidates,
        seasons=(2020,),
    )

    cutoffs = temporal._cutoffs(games)
    assert calls
    for index, call in enumerate(calls):
        cutoff = cutoffs[index // len(candidates)]
        assert all(
            games_by_id < cutoff
            for games_by_id in [
                next(game.start for game in games if game.game_id == game_id)
                for game_id in call["game_ids"]
            ]
        )
        assert np.array_equal(call["teams"]["a"], context_pmfs["a"])

    game_by_id = {game.game_id: game for game in games}
    keys_by_candidate = {}
    for candidate in candidates:
        candidate_rows = [row for row in rows if row["candidate"] == candidate.name]
        keys_by_candidate[candidate.name] = {
            (row["season"], row["cutoff"], row["target_kind"], row["target_game_id"])
            for row in candidate_rows
        }
        for row in candidate_rows:
            cutoff = datetime.fromisoformat(str(row["cutoff"]))
            assert game_by_id[str(row["target_game_id"])].start > cutoff
    assert len(set(map(frozenset, keys_by_candidate.values()))) == 1


def test_development_selection_ignores_evaluation_seasons() -> None:
    rows = []
    for season in (*range(2018, 2022), *range(2022, 2026)):
        for candidate in CANDIDATE_NAMES:
            is_r56 = candidate == "R56"
            rows.append(
                {
                    "candidate": candidate,
                    "target_kind": "next_game",
                    "season": season,
                    "margin_nll": 0.98 if is_r56 else 1.0,
                    "margin_mae": 9.8 if is_r56 else 10.0,
                    "win_brier": 0.2005 if is_r56 else 0.2,
                }
            )
    development = [row for row in rows if int(row["season"]) < 2022]
    selected = select_recency_candidate(development_gate_metrics(development))
    evaluation = [
        {**row, "margin_nll": 0.1, "margin_mae": 1.0}
        for row in rows
        if int(row["season"]) >= 2022 and row["candidate"] == "R14"
    ]
    selected_with_evaluation = select_recency_candidate(
        development_gate_metrics([*development, *evaluation])
    )
    assert selected == selected_with_evaluation == "R56"
