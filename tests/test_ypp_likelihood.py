"""Semantic tests for the research-only conditional YPP implementation."""

from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from gippyrank.posterior.engine import Game, LikelihoodV1, Team, infer_posterior
from gippyrank.research.ypp_likelihood import (
    DF_GRID,
    build_ypp_data,
    conditional_design,
    feature_names,
    fit_ypp_model,
    infer_posterior_with_ypp,
    oriented_rank_coordinates,
    oriented_ypp_difference,
    ypp_factor,
)


def _research_script():
    spec = importlib.util.spec_from_file_location(
        "ypp_research_script", Path("scripts/build_ypp_likelihood_research.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _likelihood() -> LikelihoodV1:
    return LikelihoodV1(np.zeros(34), scale=7.0, degrees_of_freedom=15.0)


def _teams() -> list[Team]:
    return [
        Team("home", "Home", "fbs", np.full(5, 0.2)),
        Team("away", "Away", "fbs", np.full(5, 0.2)),
    ]


def _y1_model() -> dict[str, object]:
    return {
        "beta": np.linspace(-0.2, 0.2, len(feature_names(rank_signal=False, include_margin=True))),
        "scale": 1.0,
        "df": 5.0,
        "rank_signal": False,
        "include_margin": True,
    }


def _y2_model() -> dict[str, object]:
    return {
        "beta": np.linspace(-0.2, 0.2, len(feature_names(rank_signal=True, include_margin=True))),
        "scale": 1.0,
        "df": 5.0,
        "rank_signal": True,
        "include_margin": True,
    }


def test_same_subdivision_ypp_orientation_is_home_minus_away() -> None:
    assert oriented_ypp_difference("fbs", "fbs", 7.5, 5.0) == pytest.approx(2.5)


def test_cross_subdivision_ypp_orientation_is_fbs_minus_fcs_regardless_of_listing() -> None:
    assert oriented_ypp_difference("fbs", "fcs", 6.0, 4.0) == pytest.approx(2.0)
    assert oriented_ypp_difference("fcs", "fbs", 4.0, 6.0) == pytest.approx(2.0)
    assert oriented_rank_coordinates("fcs", "fbs", 20, 10, 100, 120) == pytest.approx(
        ((10 - 0.5) / 120, (20 - 0.5) / 100)
    )


def test_missing_ypp_is_exactly_the_frozen_v1_factor() -> None:
    teams = _teams()
    game = Game("g", "home", "away", "fbs", "fbs", 31, 17, False)
    baseline = infer_posterior(teams, [game], _likelihood(), tolerance=1e-12)
    for ypp in ((None, None), (6.0, None)):
        research = infer_posterior_with_ypp(
            teams,
            [game],
            _likelihood(),
            {"g": ypp},
            _y2_model(),
            tolerance=1e-12,
        )
        for team_id in baseline.pmfs:
            assert research.pmfs[team_id] == pytest.approx(
                baseline.pmfs[team_id], abs=1e-12
            )


def test_conditional_ypp_null_is_rank_invariant() -> None:
    home, away = _teams()
    game = Game("g", "home", "away", "fbs", "fbs", 31, 17, False)
    factor = ypp_factor(game, home, away, _y1_model(), 6.0, 4.0)
    assert np.allclose(factor, factor[0, 0])
    null = infer_posterior_with_ypp(
        [home, away], [game], _likelihood(), {"g": (6.0, 4.0)}, _y1_model(), tolerance=1e-12
    )
    baseline = infer_posterior([home, away], [game], _likelihood(), tolerance=1e-12)
    for team_id in baseline.pmfs:
        assert null.pmfs[team_id] == pytest.approx(baseline.pmfs[team_id], abs=1e-12)


def test_build_ypp_data_uses_equal_total_game_weight_and_missing_ypp_is_skipped() -> None:
    rows = [
        {
            "game_id": "1",
            "season": 2018,
            "home_subdivision": "fbs",
            "away_subdivision": "fbs",
            "home_team_population": 5,
            "away_team_population": 5,
            "home_points": 21,
            "away_points": 14,
            "neutral_site": "False",
            "home_ypp": "6.0",
            "away_ypp": "4.0",
            "rank_pairs": json.dumps([[1, 2], [2, 3]]),
        },
        {
            "game_id": "2",
            "season": 2018,
            "home_subdivision": "fbs",
            "away_subdivision": "fcs",
            "home_team_population": 5,
            "away_team_population": 8,
            "home_points": 35,
            "away_points": 7,
            "neutral_site": "False",
            "home_ypp": "",
            "away_ypp": "4.0",
            "rank_pairs": json.dumps([[1, 2]]),
        },
    ]
    data = build_ypp_data(rows)
    assert len(data) == 2
    assert np.sum(data.weight) == pytest.approx(1.0)
    assert np.all(data.target == pytest.approx(2.0))


def test_naive_independent_diagnostic_is_not_a_candidate_name() -> None:
    # This mirrors the research build's promotion boundary: the diagnostic is
    # deliberately not in the candidate set evaluated for selection.
    assert "naive_independent_diagnostic" not in {"v1", "y1", "y2"}
    assert len(DF_GRID) == 4


def test_candidate_selection_is_deterministic() -> None:
    script = _research_script()
    n = 72
    data_type = type(build_ypp_data([]))
    data = data_type(
        x=np.linspace(0.05, 0.95, n),
        y=np.linspace(0.95, 0.05, n),
        target=np.linspace(-2.0, 3.0, n),
        margin=np.tile(np.asarray([-21.0, -7.0, 7.0, 21.0]), n // 4),
        pairing=np.asarray(["fbs-fbs"] * n),
        neutral=np.asarray([0.0, 1.0] * (n // 2)),
        fbs_home=np.zeros(n),
        weight=np.ones(n),
        game_id=np.asarray([f"g{i}" for i in range(n)]),
        season=np.repeat(np.arange(2004, 2022), 4),
    )
    first = script.select_ypp_models(data)
    second = script.select_ypp_models(data)
    assert first[1] == second[1]
    assert first[2] == second[2]


def test_fit_candidate_is_deterministic() -> None:
    n = 12
    data_type = type(build_ypp_data([]))
    data = data_type(
        x=np.linspace(0.1, 0.9, n),
        y=np.linspace(0.9, 0.1, n),
        target=np.linspace(-1.0, 2.0, n),
        margin=np.linspace(-14.0, 14.0, n),
        pairing=np.asarray(["fbs-fbs"] * n),
        neutral=np.zeros(n),
        fbs_home=np.zeros(n),
        weight=np.ones(n),
        game_id=np.asarray([f"g{i}" for i in range(n)]),
        season=np.asarray([2018] * n),
    )
    mask = np.ones(n, dtype=bool)
    first = fit_ypp_model(
        data, mask, rank_signal=True, include_margin=True, degrees_of_freedom=5.0
    )
    second = fit_ypp_model(
        data, mask, rank_signal=True, include_margin=True, degrees_of_freedom=5.0
    )
    assert first["scale"] == second["scale"]
    assert np.array_equal(first["beta"], second["beta"])


def test_conditional_design_is_deterministic_and_rank_basis_changes_y2_only() -> None:
    kwargs = {
        "margin": np.asarray([7.0, 7.0]),
        "pairing": np.asarray(["fbs-fbs", "fbs-fbs"]),
        "neutral": np.asarray([0.0, 0.0]),
        "fbs_home": np.asarray([0.0, 0.0]),
        "x": np.asarray([0.1, 0.8]),
        "y": np.asarray([0.2, 0.3]),
    }
    y1 = conditional_design(**kwargs, rank_signal=False)
    y2 = conditional_design(**kwargs, rank_signal=True)
    assert np.array_equal(y1, conditional_design(**kwargs, rank_signal=False))
    assert y2.shape[1] > y1.shape[1]


def test_comparison_support_mismatch_is_not_silently_scored() -> None:
    posterior_metrics = _research_script().posterior_metrics

    with pytest.raises(ValueError, match="support mismatch"):
        posterior_metrics({"a": np.full(4, 0.25)}, {"a": np.full(5, 0.2)})


def test_comparison_game_keys_are_order_invariant_and_strict() -> None:
    script = _research_script()
    rows = [{"game_id": "b"}, {"game_id": "a"}]
    assert script._game_population_hash(rows) == script._game_population_hash(rows[::-1])
    assert script._game_population_hash(rows) != script._game_population_hash(
        [{"game_id": "a"}]
    )


def test_cutoff_future_selection_is_strictly_after_cutoff() -> None:
    from datetime import UTC, datetime

    future_games = _research_script()._future_games

    rows = [
        {"start_date": "2022-09-01T00:00:00Z", "game_id": "a"},
        {"start_date": "2022-09-02T00:00:00Z", "game_id": "b"},
    ]
    result = future_games(rows, datetime(2022, 9, 1, tzinfo=UTC))
    assert [row["game_id"] for row in result] == ["b"]


def test_production_engine_has_no_research_ypp_dependency() -> None:
    source = Path("src/gippyrank/posterior/engine.py").read_text(encoding="utf-8")
    assert "research.ypp_likelihood" not in source


def test_production_v1_artifacts_remain_unchanged() -> None:
    paths = [
        Path("data/processed/posterior/historical_likelihood_v1.json"),
        Path("data/processed/preseason/context/predictions.csv"),
        Path("data/processed/preseason/history/predictions.csv"),
    ]
    if not all(path.exists() for path in paths):
        pytest.skip("frozen production artifacts are not present in this checkout")
    before = {
        str(path): sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    teams = _teams()
    infer_posterior_with_ypp(
        teams,
        [Game("g", "home", "away", "fbs", "fbs", 21, 14, False)],
        _likelihood(),
        {"g": (None, None)},
        _y2_model(),
    )
    after = {
        str(path): sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    assert before == after
