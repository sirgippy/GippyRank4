"""Semantic tests for the primitive box-score likelihood implementation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gippyrank.posterior.engine import Game, LikelihoodV1, Team, infer_posterior
from gippyrank.research.primitive_box_score_likelihood import (
    PRIMITIVE_FIELDS,
    PrimitiveData,
    _model_locations,
    build_primitive_data,
    feature_names,
    infer_posterior_with_primitives,
    oriented_difference,
    oriented_rank_coordinates,
    pairing_for,
    primitive_factor,
)


def _script():
    spec = importlib.util.spec_from_file_location(
        "primitive_research_script", Path("scripts/build_primitive_box_score_likelihood.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _likelihood() -> LikelihoodV1:
    return LikelihoodV1(np.linspace(-1.0, 1.0, 34), scale=7.0, degrees_of_freedom=15.0)


def _teams(home_subdivision: str = "fbs", away_subdivision: str = "fbs") -> list[Team]:
    return [
        Team("home", "Home", home_subdivision, np.full(5, 0.2)),
        Team("away", "Away", away_subdivision, np.full(5, 0.2)),
    ]


def _game(home_subdivision: str = "fbs", away_subdivision: str = "fbs") -> Game:
    return Game("g", "home", "away", home_subdivision, away_subdivision, 24, 21, False)


def _evidence(
    *,
    home_yards: object = 400,
    away_yards: object = 300,
    home_plays: object = 70,
    away_plays: object = 60,
    home_int: object = 0,
    away_int: object = 1,
    home_fumbles: object = 0,
    away_fumbles: object = 0,
) -> tuple[dict[str, object], dict[str, object]]:
    return (
        {
            "total_yards": home_yards,
            "offensive_plays_derived": home_plays,
            "interceptions_thrown": home_int,
            "fumbles_lost": home_fumbles,
        },
        {
            "total_yards": away_yards,
            "offensive_plays_derived": away_plays,
            "interceptions_thrown": away_int,
            "fumbles_lost": away_fumbles,
        },
    )


def _model(response_kind: str, *, turnover_context: bool = False, rank_signal: bool = True) -> dict[str, object]:
    names = feature_names(
        response_kind=response_kind,
        rank_signal=rank_signal,
        turnover_context=turnover_context,
    )
    return {
        "beta": np.linspace(-0.15, 0.15, len(names)),
        "scale": 1.0,
        "df": 5.0,
        "response_kind": response_kind,
        "turnover_context": turnover_context,
        "rank_signal": rank_signal,
    }


def test_pairing_and_fbs_fcs_orientation_are_canonical() -> None:
    assert pairing_for("fbs", "fcs") == "fbs-fcs"
    assert pairing_for("fcs", "fbs") == "fbs-fcs"
    assert oriented_difference("fcs", "fbs", 300, 450) == pytest.approx(150)
    assert oriented_difference("fbs", "fcs", 450, 300) == pytest.approx(150)
    assert oriented_rank_coordinates("fcs", "fbs", 20, 10, 100, 120) == pytest.approx(
        ((10 - 0.5) / 120, (20 - 0.5) / 100)
    )


def test_build_primitive_data_preserves_diff_and_total_in_fbs_first_orientation() -> None:
    rows = [
        {
            "game_id": "g",
            "season": 2018,
            "home_subdivision": "fcs",
            "away_subdivision": "fbs",
            "home_team_population": 8,
            "away_team_population": 5,
            "home_points": 10,
            "away_points": 17,
            "neutral_site": "False",
            "rank_pairs": json.dumps([[2, 1], [3, 2]]),
            "home_total_yards": 300,
            "away_total_yards": 450,
            "home_offensive_plays_derived": 60,
            "away_offensive_plays_derived": 70,
            "home_interceptions_thrown": 2,
            "away_interceptions_thrown": 0,
            "home_fumbles_lost": 1,
            "away_fumbles_lost": 0,
        }
    ]
    data = build_primitive_data(rows)
    assert np.all(data.yards_diff == 150)
    assert np.all(data.yards_total == 750)
    assert np.all(data.plays_diff == 10)
    assert np.all(data.plays_total == 130)
    assert np.all(data.interceptions_diff == -2)
    assert np.all(data.fumbles_lost_diff == -1)
    assert data.x == pytest.approx(np.asarray([(1 - 0.5) / 5, (2 - 0.5) / 5]))


def test_fcs_fcs_is_exact_v1_fallback() -> None:
    teams = _teams("fcs", "fcs")
    game = _game("fcs", "fcs")
    home, away = _evidence()
    factor = primitive_factor(
        game,
        teams[0],
        teams[1],
        _model("yards"),
        component="yards",
        home_evidence=home,
        away_evidence=away,
    )
    assert np.array_equal(factor, np.ones((5, 5)))
    baseline = infer_posterior(teams, [game], _likelihood(), tolerance=1e-12)
    research = infer_posterior_with_primitives(
        teams,
        [game],
        _likelihood(),
        {"g": (home, away)},
        {"a_yards": _model("yards")},
        variant="a",
        tolerance=1e-12,
    )
    for team_id in baseline.pmfs:
        assert research.pmfs[team_id] == pytest.approx(baseline.pmfs[team_id], abs=1e-12)


def test_research_v1_variant_matches_production_v1() -> None:
    teams = _teams()
    game = _game()
    baseline = infer_posterior(teams, [game], _likelihood(), tolerance=1e-12)
    research = infer_posterior_with_primitives(
        teams,
        [game],
        _likelihood(),
        {},
        {},
        variant="v1",
        tolerance=1e-12,
    )
    for team_id in baseline.pmfs:
        assert research.pmfs[team_id] == pytest.approx(baseline.pmfs[team_id], abs=1e-12)


def test_missing_a_evidence_falls_back_to_v1() -> None:
    teams = _teams()
    game = _game()
    home, away = _evidence(home_yards=None)
    baseline = infer_posterior(teams, [game], _likelihood(), tolerance=1e-12)
    research = infer_posterior_with_primitives(
        teams,
        [game],
        _likelihood(),
        {"g": (home, away)},
        {"a_yards": _model("yards")},
        variant="a",
        tolerance=1e-12,
    )
    for team_id in baseline.pmfs:
        assert research.pmfs[team_id] == pytest.approx(baseline.pmfs[team_id], abs=1e-12)


def test_missing_b_play_evidence_disables_both_conditional_components() -> None:
    teams = _teams()
    game = _game()
    home, away = _evidence(home_plays=None)
    play_factor = primitive_factor(
        game,
        teams[0],
        teams[1],
        _model("plays"),
        component="plays",
        home_evidence=home,
        away_evidence=away,
    )
    yard_factor = primitive_factor(
        game,
        teams[0],
        teams[1],
        _model("yards"),
        component="yards",
        home_evidence=home,
        away_evidence=away,
    )
    assert np.array_equal(play_factor, np.ones((5, 5)))
    assert np.array_equal(yard_factor, np.ones((5, 5)))


def test_missing_c_turnover_context_falls_back_exactly_to_b() -> None:
    teams = _teams()
    game = _game()
    home, away = _evidence(home_int=None)
    models = {
        "b_plays": _model("plays"),
        "b_yards": _model("yards"),
        "c_plays": _model("plays", turnover_context=True),
        "c_yards": _model("yards", turnover_context=True),
    }
    b = infer_posterior_with_primitives(
        teams,
        [game],
        _likelihood(),
        {"g": (home, away)},
        models,
        variant="b",
        tolerance=1e-12,
    )
    c = infer_posterior_with_primitives(
        teams,
        [game],
        _likelihood(),
        {"g": (home, away)},
        models,
        variant="c",
        tolerance=1e-12,
    )
    for team_id in b.pmfs:
        assert c.pmfs[team_id] == pytest.approx(b.pmfs[team_id], abs=1e-12)


def test_int_and_fumbles_lost_are_separate_context_variables() -> None:
    assert PRIMITIVE_FIELDS == (
        "total_yards",
        "offensive_plays_derived",
        "interceptions_thrown",
        "fumbles_lost",
    )
    home, away = _evidence(home_int=2, away_int=0, home_fumbles=0, away_fumbles=1)
    assert home["interceptions_thrown"] != home["fumbles_lost"]
    assert away["interceptions_thrown"] != away["fumbles_lost"]


def test_rank_neutral_yard_context_is_rank_invariant() -> None:
    teams = _teams()
    home, away = _evidence()
    factor = primitive_factor(
        _game(),
        teams[0],
        teams[1],
        _model("yards", rank_signal=False),
        component="yards",
        home_evidence=home,
        away_evidence=away,
    )
    assert np.allclose(factor, factor[0, 0])


def test_turnover_context_alone_is_rank_invariant_when_rank_neutralized() -> None:
    teams = _teams()
    home, away = _evidence(home_int=2, away_int=0, home_fumbles=0, away_fumbles=1)
    factor = primitive_factor(
        _game(),
        teams[0],
        teams[1],
        _model("yards", turnover_context=True, rank_signal=False),
        component="yards",
        home_evidence=home,
        away_evidence=away,
    )
    assert np.allclose(factor, factor[0, 0])


def test_rank_location_cache_preserves_single_point_coordinates() -> None:
    model = _model("yards")
    common = {
        "margin": np.asarray([3.0]),
        "pairing": np.asarray(["fbs-fbs"]),
        "neutral": np.asarray([0.0]),
        "fbs_home": np.asarray([0.0]),
        "plays_diff": np.asarray([4.0]),
        "plays_total": np.asarray([140.0]),
        "yards_total": np.asarray([900.0]),
        "interceptions_diff": np.asarray([0.0]),
        "interceptions_total": np.asarray([2.0]),
        "fumbles_lost_diff": np.asarray([0.0]),
        "fumbles_lost_total": np.asarray([1.0]),
    }
    first = _model_locations(model, x=np.asarray([0.1]), y=np.asarray([0.2]), **common)
    second = _model_locations(model, x=np.asarray([0.8]), y=np.asarray([0.9]), **common)
    assert first[0] != pytest.approx(second[0])


def test_development_selection_does_not_read_final_metrics() -> None:
    script = _script()
    rows = []
    for season in range(2018, 2022):
        for candidate, nll in (("v1", 1.0), ("a", 0.9), ("b", 0.8), ("c", 0.7)):
            rows.append(
                {
                    "season": season,
                    "is_final_cutoff": True,
                    "candidate": candidate,
                    "nll": nll,
                    "crps": 0.1,
                    "interval_80_coverage": 0.8,
                }
            )
    selected, _comparisons, summary = script.select_candidate_from_development(rows)
    assert selected == "c"
    assert summary["final_metrics_used"] is False
    assert summary["final_seasons_used"] == []


def test_comparison_keys_are_strict_and_identical() -> None:
    script = _script()
    common = {
        "season": 2022,
        "cutoff_index": 0,
        "prior_family": "context",
        "population_view": "full",
        "game_key_sha256": "games",
        "team_key_sha256": "teams",
    }
    rows = [{**common, "candidate": candidate} for candidate in ("v1", "a")]
    assert script.validate_common_comparison_keys(rows, ("v1", "a"))["group_count"] == 1
    with pytest.raises(ValueError, match="comparison support mismatch"):
        script.validate_common_comparison_keys(
            [rows[0], {**rows[1], "game_key_sha256": "different"}],
            ("v1", "a"),
        )


def test_reconstructed_teams_use_supplied_preseason_pmfs_and_fixed_support() -> None:
    script = _script()
    targets = {
        (2018, "fbs", "home"): {"pmf": np.zeros(3)},
        (2018, "fbs", "away"): {"pmf": np.zeros(3)},
    }
    rows = [
        {
            "season": 2018,
            "home_team_id": "home",
            "away_team_id": "away",
            "home_subdivision": "fbs",
            "away_subdivision": "fbs",
        }
    ]
    supplied = {"home": np.asarray([0.1, 0.2, 0.7]), "away": np.asarray([0.6, 0.3, 0.1])}
    teams = script.make_reconstructed_teams(
        2018, "history_reconstructed", rows, targets, supplied
    )
    assert {
        team.team_id: team.prior.tolist() for team in teams
    } == {
        "home": pytest.approx([0.1, 0.2, 0.7]),
        "away": pytest.approx([0.6, 0.3, 0.1]),
    }
    altered_labels = {key: {"pmf": np.ones(3)} for key in targets}
    altered = script.make_reconstructed_teams(
        2018, "history_reconstructed", rows, altered_labels, supplied
    )
    for left, right in zip(altered, teams, strict=True):
        assert left.prior == pytest.approx(right.prior)


def test_historical_prior_audit_declares_pre_target_boundary() -> None:
    script = _script()
    for season in range(2018, 2022):
        row = script.reconstructed_prior_audit_row(season, "history_reconstructed", 10, [])
        assert row["trained_through_season"] == season - 1
        assert row["target_outcomes_used"] is False
        assert row["supported"] is True
    excluded = script.reconstructed_prior_audit_row(
        2012,
        "history_reconstructed",
        0,
        [],
        supported=False,
        reason="required FCS-to-FBS promotion fallback has zero pre-target training rows",
        cold_start_reasons={"fcs_to_fbs_transition": 4},
        eligible_promotion_rows=0,
        eligible_generic_rows=120,
    )
    assert excluded["supported"] is False
    assert excluded["target_outcomes_used"] is False
    assert excluded["cold_start_reasons"] == {"fcs_to_fbs_transition": 4}
    assert excluded["eligible_promotion_rows"] == 0
    assert excluded["eligible_generic_rows"] == 120


@pytest.mark.parametrize(
    ("reasons", "expected_promotion", "expected_generic"),
    [
        ({"no_prior_rank_distribution": 1}, False, True),
        ({"fcs_to_fbs_transition": 1}, True, False),
        ({"fcs_to_fbs_transition": 1, "no_prior_rank_distribution": 1}, True, True),
        ({}, False, False),
    ],
)
def test_cold_start_requirements_are_target_driven(
    reasons: dict[str, int], expected_promotion: bool, expected_generic: bool
) -> None:
    script = _script()
    inference = [SimpleNamespace(cold_start_reason=reason) for reason, count in reasons.items() for _ in range(count)]
    cold = [
        SimpleNamespace(reason="fcs_to_fbs_transition", season=2011, subdivision="fbs"),
        SimpleNamespace(reason="no_prior_rank_distribution", season=2011, subdivision="fbs"),
    ]
    actual, promotion, generic = script.cold_start_requirements(
        inference, cold, cold, 2011
    )
    assert actual == reasons
    assert (actual.get("fcs_to_fbs_transition", 0) > 0) is expected_promotion
    assert (actual.get("no_prior_rank_distribution", 0) > 0) is expected_generic
    assert promotion and generic


def test_reconstructed_rolling_consumer_excludes_unsupported_targets() -> None:
    script = _script()
    teams = {(season, "history_reconstructed"): [] for season in range(2008, 2026) if season != 2012}
    supported = script.supported_reconstructed_target_seasons(teams)
    assert len(supported) == 17
    assert 2012 not in supported


def test_historical_promotion_adapter_returns_teamseason_and_preserves_cross_lag() -> None:
    script = _script()
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import build_preseason_prior as history

    cold = history.ColdStartSeason(
        2013, "fbs", "team", "Team", 3,
        np.asarray([0.0]), np.asarray([2.0]),
        np.asarray([-0.4, 0.2]), "fcs_to_fbs_transition",
    )
    adapted = history.cold_start_teams([cold])
    assert isinstance(adapted[0], history.TeamSeason)
    assert adapted[0].lag1_z is cold.cross_subdivision_lag_z
    assert adapted[0].features == {}
    assert script.RECONSTRUCTED_PRIOR_FAMILIES == (
        "context_reconstructed",
        "history_reconstructed",
    )


def test_fit_is_deterministic() -> None:
    from gippyrank.research.primitive_box_score_likelihood import (
        component_mask,
        fit_primitive_model,
    )

    rows = []
    for index in range(12):
        rows.append(
            {
                "game_id": str(index),
                "season": 2010,
                "home_subdivision": "fbs",
                "away_subdivision": "fbs",
                "home_team_population": 5,
                "away_team_population": 5,
                "home_points": 21 + index % 3,
                "away_points": 14,
                "neutral_site": "False",
                "rank_pairs": json.dumps([[1, 2], [2, 1], [3, 3]]),
                "home_total_yards": 300 + index,
                "away_total_yards": 250,
                "home_offensive_plays_derived": 65,
                "away_offensive_plays_derived": 60,
                "home_interceptions_thrown": 1,
                "away_interceptions_thrown": 0,
                "home_fumbles_lost": 0,
                "away_fumbles_lost": 0,
            }
        )
    data = build_primitive_data(rows)
    mask = component_mask(data, (2010,), response_kind="yards")
    first = fit_primitive_model(
        data,
        mask,
        response_kind="yards",
        turnover_context=False,
        degrees_of_freedom=5.0,
    )
    second = fit_primitive_model(
        data,
        mask,
        response_kind="yards",
        turnover_context=False,
        degrees_of_freedom=5.0,
    )
    assert first["scale"] == second["scale"]
    assert np.array_equal(first["beta"], second["beta"])
def test_primitive_data_type_is_explicitly_structured() -> None:
    assert set(PrimitiveData.__dataclass_fields__) >= {
        "yards_diff",
        "yards_total",
        "plays_diff",
        "plays_total",
        "interceptions_diff",
        "interceptions_total",
        "fumbles_lost_diff",
        "fumbles_lost_total",
    }
