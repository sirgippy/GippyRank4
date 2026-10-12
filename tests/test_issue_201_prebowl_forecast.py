from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import t as student_t

from gippyrank.posterior.engine import LikelihoodV1, Team
from gippyrank.posterior.predictive import (
    ScheduledGame,
    predict_game,
    predictive_components,
)
from gippyrank.research.issue_201_prebowl_forecast import (
    MODEL_ARMS,
    _predictive_summary_from_components,
    _validate_unique_forecasts,
    build_schedule_plan,
    filter_evidence_rows,
    frozen_panel_identity,
    game_phase,
    mixture_margin_crps,
    paired_game_differences,
    score_rank_pmf,
    summarize_forecast_scores,
)


def _schedule_row(
    game_id: int,
    kickoff: str,
    *,
    home_id: int = 1,
    away_id: int = 2,
    home_team: str = "Home",
    away_team: str = "Away",
    home_classification: str = "fbs",
    away_classification: str = "fbs",
    season_type: str = "regular",
) -> dict[str, object]:
    return {
        "id": game_id,
        "season": 2022,
        "week": 1,
        "seasonType": season_type,
        "startDate": kickoff,
        "neutralSite": False,
        "conferenceGame": True,
        "homeId": home_id,
        "homeTeam": home_team,
        "homeClassification": home_classification,
        "homeConference": "A",
        "awayId": away_id,
        "awayTeam": away_team,
        "awayClassification": away_classification,
        "awayConference": "B",
        # The schedule plan must not copy these score-bearing fields.
        "completed": True,
        "homePoints": 45,
        "awayPoints": 0,
    }


def test_batched_predictive_summary_matches_canonical_api() -> None:
    beta = np.zeros(34)
    beta[0] = 20.0
    likelihood = LikelihoodV1(beta, 11.0, 15.0)
    home_pmf = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    away_pmf = np.array([0.3, 0.1, 0.1, 0.2, 0.3])
    home = Team("h", "Home", "fbs", home_pmf)
    away = Team("a", "Away", "fbs", away_pmf)
    game = ScheduledGame("g", "h", "a", "fbs", "fbs")
    locations, weights = predictive_components(game, home, away, likelihood)

    canonical = predict_game(game, home, away, likelihood)
    batched = _predictive_summary_from_components(locations, weights, likelihood)

    assert batched.expected_home_margin == pytest.approx(
        canonical.expected_home_margin, abs=1e-12
    )
    assert batched.median_home_margin == pytest.approx(
        canonical.median_home_margin, abs=1e-10
    )
    assert batched.home_win_probability == pytest.approx(
        canonical.home_win_probability, abs=1e-12
    )
    for field in ("margin_interval_50", "margin_interval_80", "margin_interval_95"):
        assert getattr(batched, field) == pytest.approx(
            getattr(canonical, field), abs=1e-10
        )


def test_mixture_margin_crps_matches_independent_cdf_integral() -> None:
    locations = np.array([-5.0, 7.5, 15.0])
    weights = np.array([0.2, 0.5, 0.3])
    observation = 3.0
    scale = 8.0
    degrees = 7.0

    def cdf(value: float) -> float:
        return float(
            np.dot(
                weights,
                student_t.cdf((value - locations) / scale, df=degrees),
            )
        )

    reference = quad(lambda value: cdf(value) ** 2, -np.inf, observation)[0]
    reference += quad(lambda value: (1.0 - cdf(value)) ** 2, observation, np.inf)[0]
    actual, error, order, tail_multiple = mixture_margin_crps(
        locations, weights, observation, scale, degrees
    )

    assert actual == pytest.approx(reference, abs=2e-7)
    assert error <= 2e-6
    assert order in {256, 512}
    assert tail_multiple >= 40


def test_mixture_margin_crps_is_deterministic_translation_and_reflection_invariant() -> (
    None
):
    locations = np.array([-8.0, 1.5, 11.0])
    weights = np.array([0.15, 0.55, 0.30])
    first = mixture_margin_crps(locations, weights, 4.0, 9.0, 7.0)
    second = mixture_margin_crps(locations, weights, 4.0, 9.0, 7.0)
    translated = mixture_margin_crps(locations + 100.0, weights, 104.0, 9.0, 7.0)
    reflected = mixture_margin_crps(-locations, weights, -4.0, 9.0, 7.0)

    assert first == second
    assert translated[0] == pytest.approx(first[0], abs=1e-10)
    assert reflected[0] == pytest.approx(first[0], abs=1e-10)
    assert first[1] <= 2e-6


def test_student_t_mixture_crps_approaches_point_mass_limit() -> None:
    score, error, _order, _tail_multiple = mixture_margin_crps(
        np.array([1.0]), np.array([1.0]), 3.0, 1e-3, 5.0
    )

    assert score == pytest.approx(2.0, abs=1e-3)
    assert error <= 2e-6


def test_rank_pmf_score_validates_support_and_normalization() -> None:
    forecast = np.array([0.6, 0.3, 0.1])
    target = np.array([0.2, 0.5, 0.3])
    result = score_rank_pmf(forecast, target)
    assert result["nll"] == pytest.approx(-np.dot(target, np.log(forecast)))
    assert result["crps"] >= 0
    with pytest.raises(ValueError, match="identical"):
        score_rank_pmf(forecast, np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="normalized"):
        score_rank_pmf(np.array([0.5, 0.5, 0.5]), target)


def test_army_navy_phase_overrides_calendar_and_frozen_identity_is_content_bound() -> (
    None
):
    game = {
        "season": 2022,
        "home_team": "Army",
        "away_team": "Navy",
        "kickoff_utc": "2022-12-10T20:00:00+00:00",
    }
    assert game_phase(game, date(2022, 12, 3)) == "army_navy"

    panel = [
        {
            "season": 2022,
            "checkpoint": 1,
            "subdivision": "fbs",
            "team_id": "1",
            "context_prior_pmf": [0.5, 0.5],
            "context_posterior_pmf": [0.6, 0.4],
            "history_prior_pmf": [0.4, 0.6],
            "history_posterior_pmf": [0.3, 0.7],
            "crossed_message_pmfs": {},
        }
    ]
    identity = frozen_panel_identity(panel)
    assert frozen_panel_identity(panel) == identity
    changed = [{**panel[0], "context_posterior_pmf": [0.5, 0.5]}]
    assert frozen_panel_identity(changed) != identity


def test_score_blind_schedule_plan_assigns_each_game_once_before_kickoff(
    tmp_path: Path,
) -> None:
    rows = [
        _schedule_row(101, "2022-09-01T23:00:00Z"),
        _schedule_row(102, "2022-09-10T18:00:00Z", home_id=3, away_id=4),
        _schedule_row(
            103,
            "2022-09-10T20:00:00Z",
            home_id=5,
            away_id=6,
            away_classification="fcs",
        ),
    ]
    (tmp_path / "2022.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "2022-fcs.json").write_text("[]", encoding="utf-8")

    plan, _sources = build_schedule_plan(
        tmp_path, protocol_sha256="a" * 64, seasons=(2022,)
    )
    games = plan["seasons"][0]["games"]
    ids = [row["game_id"] for row in games]

    assert ids == ["101", "102", "103"]
    assert len(ids) == len(set(ids))
    for row in games:
        cutoff = datetime.fromisoformat(row["assigned_cutoff"])
        kickoff = datetime.fromisoformat(row["kickoff_utc"])
        assert cutoff < kickoff
        assert "home_points" not in row
        assert "completed" not in row
    assert {row["forecast_population"] for row in games} == {
        "fbs_fbs",
        "fbs_fcs",
    }
    assert len(plan["seasons"][0]["cutoffs_utc"]) == 2


def _evidence_row(
    game_id: str,
    kickoff: str,
    *,
    season_type: str = "regular",
    home_subdivision: str = "fbs",
    away_subdivision: str = "fbs",
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "kickoff_utc": kickoff,
        "completed": True,
        "home_points": 21,
        "away_points": 14,
        "season_type": season_type,
        "home_subdivision": home_subdivision,
        "away_subdivision": away_subdivision,
    }


def test_evidence_applies_48_hour_lag_and_preserves_fcs_postseason() -> None:
    cutoff = datetime(2022, 12, 20, tzinfo=UTC)
    rows = [
        _evidence_row("old_regular", "2022-12-15T18:00:00+00:00"),
        _evidence_row("near_lag", "2022-12-19T12:00:00+00:00"),
        _evidence_row(
            "fbs_postseason",
            "2022-12-16T18:00:00+00:00",
            season_type="postseason",
            away_subdivision="fcs",
        ),
        _evidence_row(
            "fcs_postseason",
            "2022-12-16T19:00:00+00:00",
            season_type="postseason",
            home_subdivision="fcs",
            away_subdivision="fcs",
        ),
    ]

    existing, existing_excluded = filter_evidence_rows(rows, cutoff, "existing")
    prebowl, prebowl_excluded = filter_evidence_rows(rows, cutoff, "pre_bowl")

    assert {row["game_id"] for row in existing} == {
        "old_regular",
        "fbs_postseason",
        "fcs_postseason",
    }
    assert {row["game_id"] for row in prebowl} == {"old_regular", "fcs_postseason"}
    assert any(row["game_id"] == "near_lag" for row in existing_excluded)
    assert any(row["game_id"] == "fbs_postseason" for row in prebowl_excluded)
    assert all(row["game_id"] != "near_lag" for row in prebowl)


def test_forecast_population_validator_rejects_duplicate_game_anchor() -> None:
    forecasts = []
    for arm, _family, _policy in MODEL_ARMS:
        forecasts.append(
            {
                "forecast_id": f"{arm}-1",
                "forecast_stage": "rolling",
                "forecast_population": "fbs_fbs",
                "model_arm": arm,
                "game_id": "same-game",
                "cutoff_utc": "2022-09-06T00:00:00+00:00",
            }
        )
    assert _validate_unique_forecasts(forecasts)["unique_forecast_ids"]
    duplicate = {
        **forecasts[0],
        "forecast_id": "new-id",
        "cutoff_utc": "2022-09-13T00:00:00+00:00",
    }
    with pytest.raises(ValueError, match="duplicate game forecast"):
        _validate_unique_forecasts(forecasts + [duplicate])


def test_paired_game_differences_include_both_aggregate_summaries() -> None:
    rows = []
    for arm, _, _ in MODEL_ARMS:
        for season, value in (("2022", 1.0), ("2023", 3.0)):
            rows.append(
                {
                    "forecast_stage": "rolling",
                    "forecast_population": "fbs_fbs",
                    "model_arm": arm,
                    "season": season,
                    "phase": "all_phases",
                    "games": 10 if season == "2022" else 30,
                    "crps": value,
                    "margin_nll": value + 1.0,
                    "median_absolute_error": value + 2.0,
                    "home_win_brier": value / 10.0,
                }
            )
        for season in ("pooled_equal_game", "equal_season_sensitivity"):
            rows.append(
                {
                    "forecast_stage": "rolling",
                    "forecast_population": "fbs_fbs",
                    "model_arm": arm,
                    "season": season,
                    "phase": "all_phases",
                    "games": 40,
                    "crps": 2.0,
                    "margin_nll": 3.0,
                    "median_absolute_error": 4.0,
                    "home_win_brier": 0.2,
                }
            )
    paired = paired_game_differences(rows, "fbs_fbs")
    assert len(paired) == 4 * 6
    assert {
        row["aggregation"]
        for row in paired
        if row["season"] in {"pooled_equal_game", "equal_season_sensitivity"}
    } == {
        "pooled_equal_game",
        "equal_season_sensitivity",
    }


def test_forecast_summaries_include_pooled_and_equal_season_interval_widths() -> None:
    widths_by_game = (
        (2022, (4.0, 8.0, 12.0)),
        (2022, (8.0, 12.0, 16.0)),
        (2023, (10.0, 14.0, 18.0)),
    )
    rows = []
    for season, widths in widths_by_game:
        rows.append(
            {
                "forecast_stage": "rolling",
                "forecast_population": "fbs_fbs",
                "model_arm": "Context_existing",
                "season": season,
                "phase": "pre_thanksgiving",
                "crps": 1.0,
                "margin_nll": 2.0,
                "median_absolute_error": 3.0,
                "home_win_brier": 0.1,
                "coverage_50": 0.5,
                "coverage_80": 0.8,
                "coverage_95": 0.95,
                **{
                    field: bound
                    for level, width in zip((50, 80, 95), widths, strict=True)
                    for field, bound in (
                        (f"margin_interval_{level}_low", 0.0),
                        (f"margin_interval_{level}_high", width),
                    )
                },
            }
        )

    summaries = summarize_forecast_scores(rows, stage="rolling", population="fbs_fbs")
    by_season = {
        row["season"]: row for row in summaries if row["phase"] == "all_phases"
    }
    assert by_season["2022"]["margin_interval_50_width"] == pytest.approx(6.0)
    assert by_season["pooled_equal_game"]["margin_interval_50_width"] == pytest.approx(
        22 / 3
    )
    assert by_season["equal_season_sensitivity"]["margin_interval_50_width"] == (
        pytest.approx(8.0)
    )
    assert by_season["pooled_equal_game"]["margin_interval_80_width"] == pytest.approx(
        34 / 3
    )
    assert by_season["equal_season_sensitivity"]["margin_interval_95_width"] == (
        pytest.approx(16.0)
    )
