from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np

from gippyrank.posterior.engine import LikelihoodV1
from gippyrank.research.offense_defense import (
    ODFit,
    build_score_environment,
    candidate_grid,
    development_gate,
    evaluate_candidates,
    expected_score_decomposition,
    fit_od_model,
    leakage_safe_residual_rows,
    od_predictive_scores,
    residual_rows,
    select_candidate,
    stage0_component_persistence,
)
from gippyrank.research.temporal_nonstationarity import HistoricalGame, PriorInput


def _likelihood() -> LikelihoodV1:
    return LikelihoodV1(np.zeros(34), scale=7.0, degrees_of_freedom=15.0)


def _game(
    game_id: str,
    start: datetime,
    home: str,
    away: str,
    home_points: int,
    away_points: int,
    *,
    week: int,
    season: int = 2008,
    home_subdivision: str = "fbs",
    away_subdivision: str = "fbs",
) -> HistoricalGame:
    return HistoricalGame(
        game_id=game_id,
        season=season,
        start=start,
        week=week,
        home_id=home,
        away_id=away,
        home_name=home.upper(),
        away_name=away.upper(),
        home_subdivision=home_subdivision,
        away_subdivision=away_subdivision,
        home_population=3 if home_subdivision == "fbs" else 2,
        away_population=3 if away_subdivision == "fbs" else 2,
        home_points=home_points,
        away_points=away_points,
        margin=float(home_points - away_points)
        if home_subdivision == away_subdivision or home_subdivision == "fbs"
        else float(away_points - home_points),
        neutral_site=False,
        home_ranks=np.array([1, 2, 3][: 3 if home_subdivision == "fbs" else 2]),
        away_ranks=np.array([1, 2, 3][: 3 if away_subdivision == "fbs" else 2]),
        rank_pairs=np.array([[1, 1], [2, 2]], dtype=float),
    )


def _environment() -> tuple[list[HistoricalGame], object]:
    start = datetime(2008, 8, 30, tzinfo=UTC)
    games = [
        _game("train", start, "a", "b", 20, 20, week=1),
        _game("future", start + timedelta(days=7), "a", "b", 30, 10, week=2),
    ]
    return games, build_score_environment(games, (2008,))


def _prior(season: int, team_id: str, population: int = 3) -> PriorInput:
    return PriorInput(
        season,
        team_id,
        team_id.upper(),
        "fbs",
        population,
        np.full(population, 1.0 / population),
        "test",
    )


def test_expected_score_decomposition_has_symmetric_offense_and_defense_orientation() -> (
    None
):
    games, environment = _environment()
    decomposition = expected_score_decomposition(games[1], _likelihood(), environment)
    assert decomposition.expected_home_points == 20
    assert decomposition.expected_away_points == 20
    assert decomposition.home_offensive_residual == 10
    assert decomposition.away_offensive_residual == -10
    assert decomposition.home_defensive_residual == 10
    assert decomposition.away_defensive_residual == -10


def test_same_subdivision_home_away_swap_preserves_team_residual_meanings() -> None:
    start = datetime(2008, 8, 30, tzinfo=UTC)
    original = _game("original", start, "a", "b", 30, 10, week=1)
    swapped = _game("swapped", start, "b", "a", 10, 30, week=1)
    environment = build_score_environment([original], (2008,))
    first = expected_score_decomposition(original, _likelihood(), environment)
    second = expected_score_decomposition(swapped, _likelihood(), environment)
    assert first.expected_margin == -second.expected_margin
    assert first.home_offensive_residual == second.away_offensive_residual
    assert first.away_offensive_residual == second.home_offensive_residual
    assert first.home_defensive_residual == second.away_defensive_residual
    assert first.away_defensive_residual == second.home_defensive_residual


def test_cross_subdivision_expected_score_uses_fbs_first_orientation() -> None:
    start = datetime(2008, 8, 30, tzinfo=UTC)
    games = [
        _game(
            "cross",
            start,
            "fcs",
            "fbs",
            7,
            28,
            week=1,
            home_subdivision="fcs",
            away_subdivision="fbs",
        ),
    ]
    environment = build_score_environment(games, (2008,))
    decomposition = expected_score_decomposition(games[0], _likelihood(), environment)
    assert (
        decomposition.expected_home_points == decomposition.expected_away_points == 17.5
    )
    assert decomposition.home_offensive_residual == -10.5
    assert decomposition.away_offensive_residual == 10.5


def test_residual_cutoff_excludes_future_games() -> None:
    games, environment = _environment()
    cutoff = games[1].start
    rows = residual_rows(games, _likelihood(), environment, cutoff=cutoff)
    assert {row["game_id"] for row in rows} == {"train"}


def test_leakage_safe_residuals_ignore_later_outcomes_and_rank_pairs() -> None:
    training = [
        _game(
            "environment",
            datetime(2007, 8, 30, tzinfo=UTC),
            "a",
            "b",
            20,
            20,
            week=1,
            season=2007,
        )
    ]
    target = _game(
        "target", datetime(2008, 8, 30, tzinfo=UTC), "a", "b", 20, 10, week=1
    )
    later = _game("later", datetime(2008, 9, 6, tzinfo=UTC), "a", "b", 30, 10, week=2)
    environment = build_score_environment(training, (2007,))
    priors = {(2008, team): _prior(2008, team) for team in ("a", "b")}
    before = leakage_safe_residual_rows(
        [target, later], priors, _likelihood(), environment
    )
    changed = [
        replace(target, rank_pairs=np.array([[3, 1], [3, 2]], dtype=float)),
        replace(later, home_points=7, away_points=42, margin=-35.0),
    ]
    after = leakage_safe_residual_rows(changed, priors, _likelihood(), environment)
    before_target = next(
        row for row in before if row["game_id"] == "target" and row["team_id"] == "a"
    )
    after_target = next(
        row for row in after if row["game_id"] == "target" and row["team_id"] == "a"
    )
    assert before_target["expected_margin"] == after_target["expected_margin"]
    assert before_target["offensive_residual"] == after_target["offensive_residual"]
    assert before_target["defensive_residual"] == after_target["defensive_residual"]


def test_stage0_constructs_same_and_cross_component_relations_and_nulls() -> None:
    rows = []
    for index in range(4):
        rows.append(
            {
                "season": 2018,
                "team_id": "a",
                "game_id": str(index),
                "game_date": (
                    datetime(2018, 9, 1, tzinfo=UTC) + timedelta(days=7 * index)
                ).isoformat(),
                "offensive_residual": float(index),
                "defensive_residual": float(index * 2),
            }
        )
    metrics, early_late, summary = stage0_component_persistence(rows, permutations=2)
    relations = {str(row["relation"]) for row in metrics}
    controls = {str(row["control"]) for row in metrics}
    assert {
        "offense->offense",
        "defense->defense",
        "offense->defense",
        "defense->offense",
    } <= relations
    assert {"observed", "shuffle_order"} <= controls
    assert summary["n_team_seasons_with_at_least_3_games"] == 1
    assert early_late


def test_od_fit_is_centered_and_uses_opponent_defense_in_score_prediction() -> None:
    start = datetime(2018, 8, 30, tzinfo=UTC)
    games = [
        _game("g1", start, "a", "b", 30, 10, week=1, season=2018),
        _game("g2", start + timedelta(days=7), "b", "c", 20, 10, week=2, season=2018),
    ]
    training = [_game("t", datetime(2008, 8, 30, tzinfo=UTC), "a", "b", 20, 20, week=1)]
    environment = build_score_environment(training, (2008,))
    priors = {(2018, team): _prior(2018, team) for team in ("a", "b", "c")}
    metadata = {team: (team.upper(), "fbs", 3) for team in ("a", "b", "c")}
    fit = fit_od_model(
        games[:1], priors, environment, candidate_grid()[0], team_metadata=metadata
    )
    assert np.isclose(np.mean([state.offense for state in fit.states.values()]), 0.0)
    assert np.isclose(np.mean([state.defense for state in fit.states.values()]), 0.0)
    prediction = od_predictive_scores(games[1], fit, environment)
    changed = dict(fit.states)
    changed["c"] = type(changed["c"])(
        **{**changed["c"].__dict__, "defense": changed["c"].defense + 5}
    )
    changed_fit = ODFit(
        fit.candidate, changed, fit.score_scale, fit.cutoff, fit.training_game_ids
    )
    changed_prediction = od_predictive_scores(games[1], changed_fit, environment)
    assert changed_prediction["expected_margin"] < prediction["expected_margin"]


def test_candidate_grid_and_development_selection_are_frozen() -> None:
    assert [candidate.name for candidate in candidate_grid()] == ["OD0", "OD1"]
    rows = []
    for season in range(2018, 2022):
        rows.extend(
            [
                {
                    "candidate": "Posterior V1",
                    "target_kind": "next_game",
                    "season": season,
                    "margin_nll": 4.0,
                    "margin_mae": 10.0,
                    "win_brier": 0.20,
                },
                {
                    "candidate": "Scalar scoreboard",
                    "target_kind": "next_game",
                    "season": season,
                    "margin_nll": 3.95,
                    "margin_mae": 9.9,
                    "win_brier": 0.20,
                },
                {
                    "candidate": "OD0",
                    "target_kind": "next_game",
                    "season": season,
                    "margin_nll": 3.9,
                    "margin_mae": 9.8,
                    "win_brier": 0.20,
                },
                {
                    "candidate": "OD1",
                    "target_kind": "next_game",
                    "season": season,
                    "margin_nll": 3.91,
                    "margin_mae": 9.8,
                    "win_brier": 0.20,
                },
            ]
        )
    gate = development_gate(rows)
    assert select_candidate(gate) == "OD0"


def test_evaluation_uses_identical_future_keys_and_strict_cutoffs() -> None:
    start = datetime(2018, 8, 30, tzinfo=UTC)
    games = [
        _game(
            f"g{index}",
            start + timedelta(days=7 * index),
            "a",
            "b",
            20 + index,
            10,
            week=index + 1,
            season=2018,
        )
        for index in range(5)
    ]
    train = [_game("t", datetime(2008, 8, 30, tzinfo=UTC), "a", "b", 20, 20, week=1)]
    environment = build_score_environment(train, (2008,))
    priors = {(2018, team): _prior(2018, team) for team in ("a", "b")}
    rows, _diagnostics, _profiles = evaluate_candidates(
        games,
        priors,
        _likelihood(),
        environment,
        candidate_grid(),
        (2018,),
    )
    by_candidate = {
        str(candidate): {
            (row["season"], row["cutoff"], row["target_kind"], row["target_game_id"])
            for row in rows
            if row["candidate"] == candidate
        }
        for candidate in ("Posterior V1", "Scalar scoreboard", "OD0", "OD1")
    }
    assert (
        by_candidate["Posterior V1"]
        == by_candidate["Scalar scoreboard"]
        == by_candidate["OD0"]
        == by_candidate["OD1"]
    )
    assert all(row["target_date"] > row["cutoff"] for row in rows)
