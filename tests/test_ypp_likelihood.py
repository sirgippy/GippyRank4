"""Semantic tests for the conditional YPP implementation."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from gippyrank.posterior.engine import Game, LikelihoodV1, Team, infer_posterior
from gippyrank.research.ypp_likelihood import (
    SUPPORTED_YPP_PAIRINGS,
    build_ypp_data,
    conditional_design,
    feature_names,
    fit_ypp_model,
    infer_posterior_with_ypp,
    oriented_rank_coordinates,
    oriented_ypp_difference,
    pairing_for,
    ypp_factor,
    ypp_pairing_supported,
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


def _fcs_teams() -> list[Team]:
    return [
        Team("home", "Home", "fcs", np.asarray([0.05, 0.15, 0.25, 0.25, 0.30])),
        Team("away", "Away", "fcs", np.asarray([0.30, 0.25, 0.20, 0.15, 0.10])),
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


def _population_row(
    game_id: str,
    home_subdivision: str,
    away_subdivision: str,
    ypp_diff: float | None,
    *,
    home_points: int = 45,
    away_points: int = 17,
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "season": 2022,
        "start_date": "2022-09-10T17:00:00Z",
        "home_subdivision": home_subdivision,
        "away_subdivision": away_subdivision,
        "home_points": home_points,
        "away_points": away_points,
        "home_ypp": None if ypp_diff is None else 6.0,
        "away_ypp": None if ypp_diff is None else 6.0 - ypp_diff,
        "ypp_diff": ypp_diff,
        "neutral_site": False,
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


@pytest.mark.parametrize(
    ("home_subdivision", "away_subdivision"),
    [("fbs", "fbs"), ("fbs", "fcs")],
)
def test_supported_pairings_with_usable_ypp_get_a_y2_factor(
    home_subdivision: str, away_subdivision: str
) -> None:
    home = Team("home", "Home", home_subdivision, np.full(5, 0.2))
    away = Team("away", "Away", away_subdivision, np.full(5, 0.2))
    game = Game(
        "g",
        "home",
        "away",
        home_subdivision,
        away_subdivision,
        31,
        17,
        False,
    )
    factor = ypp_factor(game, home, away, _y2_model(), 6.0, 4.0)
    assert factor.shape == (5, 5)
    assert not np.allclose(factor, np.ones_like(factor))
    assert pairing_for(home_subdivision, away_subdivision) in SUPPORTED_YPP_PAIRINGS
    assert ypp_pairing_supported(home_subdivision, away_subdivision)


def test_fcs_fcs_ypp_is_always_an_all_ones_factor_and_v1_posterior() -> None:
    teams = _fcs_teams()
    game = Game("g", "home", "away", "fcs", "fcs", 31, 17, False)
    factor = ypp_factor(game, teams[0], teams[1], _y2_model(), 6.0, 4.0)
    assert np.array_equal(factor, np.ones((5, 5)))

    baseline = infer_posterior(teams, [game], _likelihood(), tolerance=1e-12)
    corrected = infer_posterior_with_ypp(
        teams,
        [game],
        _likelihood(),
        {"g": (6.0, 4.0)},
        _y2_model(),
        tolerance=1e-12,
    )
    for team_id in baseline.pmfs:
        assert corrected.pmfs[team_id] == pytest.approx(
            baseline.pmfs[team_id], abs=1e-12
        )


def test_missing_supported_pairing_ypp_factor_is_exactly_ones() -> None:
    teams = _teams()
    game = Game("g", "home", "away", "fbs", "fbs", 31, 17, False)
    factor = ypp_factor(game, teams[0], teams[1], _y2_model(), None, 4.0)
    assert np.array_equal(factor, np.ones((5, 5)))


def test_corrected_fit_excludes_unsupported_pairing_rows() -> None:
    data_type = type(build_ypp_data([]))
    supported = data_type(
        x=np.linspace(0.1, 0.9, 16),
        y=np.linspace(0.9, 0.1, 16),
        target=np.linspace(-1.0, 2.0, 16),
        margin=np.linspace(-14.0, 14.0, 16),
        pairing=np.asarray(["fbs-fbs"] * 16),
        neutral=np.zeros(16),
        fbs_home=np.zeros(16),
        weight=np.ones(16),
        game_id=np.asarray([f"s{i}" for i in range(16)]),
        season=np.asarray([2010] * 16),
    )
    unsupported = data_type(
        x=np.asarray([0.2, 0.8]),
        y=np.asarray([0.8, 0.2]),
        target=np.asarray([100.0, -100.0]),
        margin=np.asarray([7.0, -7.0]),
        pairing=np.asarray(["fcs-fcs", "fcs-fcs"]),
        neutral=np.zeros(2),
        fbs_home=np.zeros(2),
        weight=np.ones(2),
        game_id=np.asarray(["u1", "u2"]),
        season=np.asarray([2010, 2010]),
    )
    all_data = data_type(
        **{
            field: np.concatenate([getattr(supported, field), getattr(unsupported, field)])
            for field in (
                "x",
                "y",
                "target",
                "margin",
                "pairing",
                "neutral",
                "fbs_home",
                "weight",
                "game_id",
                "season",
            )
        }
    )
    mask = np.ones(len(all_data), dtype=bool)
    filtered_fit = fit_ypp_model(
        all_data, mask, rank_signal=True, degrees_of_freedom=5.0
    )
    direct_fit = fit_ypp_model(
        supported,
        np.ones(len(supported), dtype=bool),
        rank_signal=True,
        degrees_of_freedom=5.0,
    )
    assert filtered_fit["fit_pairings"] == tuple(sorted(SUPPORTED_YPP_PAIRINGS))
    assert filtered_fit["fit_game_count"] == direct_fit["fit_game_count"]
    assert filtered_fit["beta"] == pytest.approx(direct_fit["beta"])


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


def test_disagreement_artifact_selector_restricts_pairings(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _research_script()
    rows = [
        _population_row("fbs-fbs", "fbs", "fbs", -1.0),
        _population_row("fbs-fcs", "fbs", "fcs", -1.0),
        _population_row("fcs-fcs", "fcs", "fcs", -1.0),
    ]

    def fake_local_effect(row, _all_rows, _targets, _models, _likelihood):
        return {"game_id": row["game_id"], "pairing": row["pairing"]}

    monkeypatch.setattr(script, "_local_disagreement_effect", fake_local_effect)
    artifact_rows = script.build_disagreement_table(rows, {}, {}, None)

    assert {row["pairing"] for row in artifact_rows} <= set(
        script.SUPPORTED_YPP_PAIRINGS
    )
    assert {row["game_id"] for row in artifact_rows} == {"fbs-fbs", "fbs-fcs"}


def test_disagreement_artifact_selector_requires_usable_ypp() -> None:
    script = _research_script()
    rows = [
        _population_row("missing-ypp", "fbs", "fbs", None),
        _population_row("usable-ypp", "fbs", "fbs", -1.0),
    ]

    selected = script._choose_disagreement_rows(rows)

    assert [row["game_id"] for row in selected] == ["usable-ypp"]


def test_supported_ypp_observed_population_excludes_fcs_fcs_with_ypp() -> None:
    script = _research_script()
    rows = [
        _population_row("supported-same", "fbs", "fbs", -1.0),
        _population_row("supported-cross", "fbs", "fcs", -1.0),
        _population_row("unsupported-with-ypp", "fcs", "fcs", -1.0),
        _population_row("supported-missing", "fbs", "fbs", None),
    ]

    selected = script.select_population_rows(
        rows,
        season=2022,
        cutoff=datetime(2022, 12, 31, tzinfo=UTC),
        view=script.SUPPORTED_YPP_OBSERVED_VIEW,
    )

    assert [row["game_id"] for row in selected] == [
        "supported-same",
        "supported-cross",
    ]


def test_supported_ypp_observed_v1_and_y2_comparison_keys_are_identical() -> None:
    script = _research_script()
    rows = [
        _population_row("supported", "fbs", "fbs", -1.0),
        _population_row("unsupported", "fcs", "fcs", -1.0),
    ]
    selected_v1 = script.select_population_rows(
        rows,
        season=2022,
        cutoff=datetime(2022, 12, 31, tzinfo=UTC),
        view=script.SUPPORTED_YPP_OBSERVED_VIEW,
    )
    selected_y2 = script.select_population_rows(
        rows,
        season=2022,
        cutoff=datetime(2022, 12, 31, tzinfo=UTC),
        view=script.SUPPORTED_YPP_OBSERVED_VIEW,
    )
    assert [row["game_id"] for row in selected_v1] == [row["game_id"] for row in selected_y2]

    common = {
        "season": 2022,
        "cutoff_index": 6,
        "prior_family": "context",
        "population_view": script.SUPPORTED_YPP_OBSERVED_VIEW,
        "game_key_sha256": script._game_population_hash(selected_v1),
        "team_key_sha256": "teams",
        "matched_fbs_teams": 1,
    }
    comparison_rows = [
        {**common, "candidate": candidate}
        for candidate in ("v1", script.Y2_SUPPORTED_VARIANT)
    ]
    assert script.validate_common_comparison_keys(comparison_rows)["group_count"] == 1


def test_full_population_keeps_fcs_fcs_games() -> None:
    script = _research_script()
    rows = [
        _population_row("supported", "fbs", "fbs", -1.0),
        _population_row("unsupported-with-ypp", "fcs", "fcs", -1.0),
        _population_row("unsupported-missing", "fcs", "fcs", None),
    ]

    selected = script.select_population_rows(
        rows,
        season=2022,
        cutoff=datetime(2022, 12, 31, tzinfo=UTC),
        view="full",
    )

    assert [row["game_id"] for row in selected] == [
        "supported",
        "unsupported-with-ypp",
        "unsupported-missing",
    ]


def test_promotion_assessment_uses_full_view(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _research_script()
    views: list[str] = []

    def fake_metric_delta(*_args, **kwargs):
        views.append(kwargs["view"])
        return []

    monkeypatch.setattr(script, "_metric_delta", fake_metric_delta)
    script._promotion_assessment([], [], [], [])

    assert views == ["full"]


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


def test_candidate_selection_excludes_2022_2025_from_fit_and_selection() -> None:
    script = _research_script()
    n = 88
    data_type = type(build_ypp_data([]))
    data = data_type(
        x=np.linspace(0.05, 0.95, n),
        y=np.linspace(0.95, 0.05, n),
        target=np.concatenate(
            [np.linspace(-2.0, 3.0, 72), np.asarray([100.0] * 16)]
        ),
        margin=np.tile(np.asarray([-21.0, -7.0, 7.0, 21.0]), 22),
        pairing=np.asarray(["fbs-fbs"] * n),
        neutral=np.asarray([0.0, 1.0] * (n // 2)),
        fbs_home=np.zeros(n),
        weight=np.ones(n),
        game_id=np.asarray([f"g{i}" for i in range(n)]),
        season=np.repeat(np.arange(2004, 2026), 4),
    )
    _models, selection_rows, selected = script.select_ypp_models(data)
    assert {int(row["train_games"]) for row in selection_rows} == {56}
    assert {int(row["development_games"]) for row in selection_rows} == {16}
    for value in selected.values():
        assert value["training_seasons"] == "2004-2017"
        assert value["development_seasons"] == "2018-2021"
        assert value["final_fit_seasons"] == "2004-2021"
        assert value["final_fit_game_count"] == 72


def test_comparison_fbs_keys_are_required_to_match() -> None:
    script = _research_script()
    common = {
        "season": 2022,
        "cutoff_index": 6,
        "prior_family": "context",
        "population_view": "full",
        "game_key_sha256": "game",
        "team_key_sha256": "teams",
        "matched_fbs_teams": 10,
    }
    rows = [
        {**common, "candidate": "v1"},
        {**common, "candidate": "y2_supported"},
    ]
    assert script.validate_common_comparison_keys(rows)["group_count"] == 1
    with pytest.raises(ValueError, match="support mismatch"):
        script.validate_common_comparison_keys(
            [rows[0], {**rows[1], "team_key_sha256": "different"}]
        )


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
