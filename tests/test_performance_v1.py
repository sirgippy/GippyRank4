from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gippyrank.performance_v1 import (
    PerformanceInference,
    explicit_neutralized_target,
    infer_performance,
    performance_pmf_summaries,
    pmf_tv_distance,
    remove_focal_prior,
    uniform_pmf,
)
from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    PosteriorResult,
    Team,
    game_factor,
)
from gippyrank.posterior.snapshots import load_likelihood

_BUILD_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "performance_build_script_for_tests",
    Path(__file__).resolve().parents[1] / "scripts/build_performance_v1.py",
)
assert _BUILD_SCRIPT_SPEC is not None and _BUILD_SCRIPT_SPEC.loader is not None
_BUILD_SCRIPT = importlib.util.module_from_spec(_BUILD_SCRIPT_SPEC)
sys.modules[_BUILD_SCRIPT_SPEC.name] = _BUILD_SCRIPT
_BUILD_SCRIPT_SPEC.loader.exec_module(_BUILD_SCRIPT)
Corpus = _BUILD_SCRIPT.Corpus
_artifact_inventory = _BUILD_SCRIPT._artifact_inventory
_prepare_cutoff = _BUILD_SCRIPT._prepare_cutoff
_validation_case_targets = _BUILD_SCRIPT._validation_case_targets
aggregate_future_rows = _BUILD_SCRIPT.aggregate_future_rows
loopy_cycle_diagnostics = _BUILD_SCRIPT.loopy_cycle_diagnostics
validate_common_future_keys = _BUILD_SCRIPT.validate_common_future_keys


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


def _corpus_row(
    game_id: str,
    start_date: str,
    *,
    completed: str = "True",
    home_classification: str = "fbs",
    away_classification: str = "fbs",
) -> dict[str, str]:
    return {
        "id": game_id,
        "season": "2024",
        "week": "1",
        "seasonType": "regular",
        "startDate": start_date,
        "completed": completed,
        "neutralSite": "False",
        "conferenceGame": "False",
        "homeId": "1",
        "homeTeam": "Home",
        "homeClassification": home_classification,
        "homeConference": "",
        "homePoints": "24",
        "awayId": "2",
        "awayTeam": "Away",
        "awayClassification": away_classification,
        "awayConference": "",
        "awayPoints": "17",
    }


def test_cutoff_partitions_completed_games_by_source_date() -> None:
    rows = (
        _corpus_row("1", "2024-09-01T12:00:00+00:00"),
        _corpus_row("2", "2024-09-02T23:59:59+00:00"),
        _corpus_row("3", "2024-09-03T12:00:00+00:00"),
        _corpus_row("4", "2024-09-04T12:00:00+00:00", completed="False"),
        _corpus_row(
            "5",
            "2024-09-05T12:00:00+00:00",
            home_classification="fbs",
            away_classification="other",
        ),
    )
    corpus = Corpus({2024: rows}, {2024: ()})
    prepared = _prepare_cutoff(
        Path("/tmp/no-performance-provenance"),
        corpus,
        2024,
        datetime(2024, 9, 2, 23, 59, 59, tzinfo=UTC),
    )
    assert [row["id"] for row in prepared.included_rows] == ["1", "2"]
    assert [row["id"] for row in prepared.future_rows] == ["3"]
    assert [game.game_id for game in prepared.games] == ["1", "2"]
    assert [game.game_id for game in prepared.future_games] == ["3"]
    assert {game.game_id for game in prepared.games}.isdisjoint(
        game.game_id for game in prepared.future_games
    )


def _future_stub(
    model: str, game_id: str = "g", focal_team_id: str = "a", is_next: bool = True
) -> dict[str, object]:
    return {
        "season": 2024,
        "cutoff": "2024-09-02T23:59:59+00:00",
        "season_phase": "early",
        "model": model,
        "game_id": game_id,
        "focal_team_id": focal_team_id,
        "is_next_game": is_next,
        "games_played_bucket": "1-3",
        "win_probability": 0.6,
        "actual_win": 1,
        "brier": 0.16,
        "marginalized_nll": 1.0,
        "margin_absolute_error": 2.0,
    }


def test_future_validation_requires_strict_common_keys() -> None:
    models = ("Predictive_C", "Predictive_H", "Performance_C", "Performance_H")
    rows = [
        row
        for model in models
        for row in (
            _future_stub(model),
            _future_stub(model, game_id="h", focal_team_id="b", is_next=False),
        )
    ]
    audit = validate_common_future_keys(rows)
    assert audit["common_key_count"] == 2
    assert audit["all_models_share_identical_keys"] is True
    with pytest.raises(ValueError, match="strict common scoring keys"):
        validate_common_future_keys(rows[:-1])


def _fake_validation_run(season: int, cutoff_index: int) -> SimpleNamespace:
    support = 8
    team_ids = ("a", "b", "c", "d", "e")
    teams = tuple(
        Team(team_id, team_id.upper(), "fbs", uniform_pmf(support))
        for team_id in team_ids
    )
    shapes = (
        np.ones(support),
        np.arange(1, support + 1, dtype=float),
        np.arange(support, 0, -1, dtype=float),
        np.array([8.0, 1, 1, 1, 1, 1, 1, 1]),
        np.array([4.0, 1, 1, 1, 1, 1, 1, 4]),
    )
    context_anchor = {
        team_id: shape / shape.sum() for team_id, shape in zip(team_ids, shapes)
    }
    history_anchor = {
        team_id: pmf[::-1] for team_id, pmf in context_anchor.items()
    }
    games = (
        []
        if cutoff_index == 0
        else [
            Game("ab", "a", "b", "fbs", "fbs", 31, 24),
            Game("bc", "b", "c", "fbs", "fbs", 28, 24),
            Game("ca", "c", "a", "fbs", "fbs", 27, 24),
        ]
    )
    counts_by_index = {
        0: [0, 0, 0, 0, 0],
        1: [1, 2, 3, 1, 2],
        3: [4, 5, 6, 4, 5],
        6: [7, 8, 9, 7, 8],
    }
    counts = dict(zip(team_ids, counts_by_index[cutoff_index], strict=True))
    context_result = PosteriorResult(
        context_anchor, True, 1, 0.0, 0.0, len(games), len(games), 2
    )
    history_result = PosteriorResult(
        history_anchor, True, 1, 0.0, 0.0, len(games), len(games), 2
    )
    context = PerformanceInference(
        "context",
        "prior_stripping",
        teams,
        tuple(games),
        context_result,
        context_anchor,
        counts,
    )
    history = PerformanceInference(
        "history",
        "prior_stripping",
        teams,
        tuple(games),
        history_result,
        history_anchor,
        counts,
    )
    prepared = SimpleNamespace(
        season=season,
        effective_cutoff=datetime(2024, 1, cutoff_index + 1, tzinfo=UTC),
        games=tuple(games),
    )
    return SimpleNamespace(
        prepared=prepared,
        context=context,
        history=history,
        cutoff_index=cutoff_index,
        cutoff_count=7,
    )


def test_prior_removal_selection_is_fixed_and_deterministic() -> None:
    runs = [
        _fake_validation_run(season, cutoff_index)
        for season in (2022, 2023, 2024, 2025)
        for cutoff_index in (0, 1, 3, 6)
    ]
    first = _validation_case_targets(runs)
    second = _validation_case_targets(runs)
    assert [
        (label, run.prepared.season, run.cutoff_index, team_id)
        for label, run, team_id in first
    ] == [
        (label, run.prepared.season, run.cutoff_index, team_id)
        for label, run, team_id in second
    ]
    roles = {
        role for label, _run, _team_id in first for role in label.split(";")
    }
    assert {
        "zero_games",
        "elite_early",
        "middle_early",
        "weak_early",
        "narrow_mid",
        "broad_mid",
        "irregular_mid",
        "anchor_disagreement_late",
        "dense_graph_late",
        "cycle_exposure_late",
    } <= roles


def test_all_future_is_inclusive_and_next_game_is_a_subset() -> None:
    models = ("Predictive_C", "Predictive_H", "Performance_C", "Performance_H")
    rows = [
        row
        for model in models
        for row in (
            _future_stub(model),
            _future_stub(model, game_id="h", focal_team_id="b", is_next=False),
        )
    ]
    aggregate = aggregate_future_rows(rows)
    counts = {(row["model"], row["horizon"]): row["prediction_count"] for row in aggregate}
    assert counts["Predictive_C", "all_future"] == 2
    assert counts["Predictive_C", "next_game"] == 1


def test_loopy_diagnostic_separates_tree_feedback_from_cycle_feedback() -> None:
    diagnostics = loopy_cycle_diagnostics(likelihood())
    baseline = diagnostics["baseline"]
    tree_rows = [
        row for row in diagnostics["topology_rows"] if row["topology"] == "tree"
    ]
    cycle_rows = [
        row for row in diagnostics["topology_rows"] if row["topology"] == "cycle"
    ]
    assert float(baseline["stripping_tv_distance"]) > 0.01
    assert max(float(row["stripping_tv_distance"]) for row in tree_rows) < 1e-8
    assert max(float(row["stripping_tv_distance"]) for row in cycle_rows) > 0.01
    assert max(
        float(row["explicit_neutralized_tv_distance"])
        for row in diagnostics["topology_rows"]
    ) < 1e-10
    one_sweep = next(
        row for row in diagnostics["settings_rows"] if row["label"] == "one_message_sweep"
    )
    assert float(one_sweep["stripping_tv_distance"]) < float(
        baseline["stripping_tv_distance"]
    )


def test_artifact_inventory_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "sample.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "sample.txt").write_text("stable\n", encoding="utf-8")
    assert _artifact_inventory(tmp_path) == _artifact_inventory(tmp_path)
    (tmp_path / "report.md").write_text("stale\n", encoding="utf-8")
    assert all(
        item["artifact"] != "report.md"
        for item in _artifact_inventory(
            tmp_path, exclude=frozenset({"report.md"})
        )
    )


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
    assert expected(road) <= expected(home) + 1e-12


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
