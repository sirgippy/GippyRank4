from __future__ import annotations

import numpy as np
import pytest

from gippyrank.research import issue_201_prebowl_forecast as issue201
from gippyrank.research.issue_256_predictive_calibration import (
    _bootstrap_draws,
    _verify_prior_history,
    build_calibration_observations,
    interval_coverage,
    orient_interval,
    progression_group,
    strength_quartile,
    strength_quartile_reference,
)


def _score(family: str) -> dict[str, str]:
    return {
        "forecast_id": f"forecast-{family}",
        "forecast_stage": "rolling",
        "season": "2022",
        "cutoff_utc": "2022-09-27T00:00:00+00:00",
        "game_id": "game-1",
        "phase": "pre_thanksgiving",
        "forecast_population": "fbs_fbs",
        "home_id": "1",
        "home_team": "Home Team",
        "home_subdivision": "fbs",
        "away_id": "2",
        "away_team": "Away Team",
        "away_subdivision": "fbs",
        "kickoff_utc": "2022-10-01T20:00:00+00:00",
        "policy": "existing",
        "prior_family": family,
        "model_arm": f"{family}_existing",
        "evidence_game_count": "0",
        "evidence_ids_sha256": issue201.sha256_json([]),
        "actual_home_margin": "3",
        "expected_home_margin": "2",
        "median_home_margin": "2",
        "margin_interval_50_low": "-1",
        "margin_interval_50_high": "5",
        "margin_interval_80_low": "-5",
        "margin_interval_80_high": "11",
        "margin_interval_95_low": "-10",
        "margin_interval_95_high": "16",
        "coverage_50": "1",
        "coverage_80": "1",
        "coverage_95": "1",
        "crps": "1.3",
        "margin_nll": "2.1",
    }


def _surprise_row(score: dict[str, str], *, is_home: bool) -> dict[str, str]:
    team_id = score["home_id"] if is_home else score["away_id"]
    opponent_id = score["away_id"] if is_home else score["home_id"]
    team = score["home_team"] if is_home else score["away_team"]
    opponent = score["away_team"] if is_home else score["home_team"]
    sign = 1 if is_home else -1
    return {
        "season": score["season"],
        "model_arm": score["model_arm"],
        "prior_family": score["prior_family"],
        "policy": score["policy"],
        "team_id": team_id,
        "team": team,
        "opponent_id": opponent_id,
        "opponent": opponent,
        "game_id": score["game_id"],
        "forecast_id": score["forecast_id"],
        "kickoff_utc": score["kickoff_utc"],
        "cutoff_utc": score["cutoff_utc"],
        "is_home": str(is_home).lower(),
        "forecast_population": score["forecast_population"],
        "team_game_number": "1",
        "actual_margin": str(3 * sign),
        "predicted_margin": str(2 * sign),
        "next_signed_error": "1",
        "margin_80_interval_half_width": "8",
        "next_standardized_error": "0.125",
        "candidate_previous_game_count": "0",
        "eligible_prior_game_count": "0",
        "excluded_previous_game_count": "0",
        "eligible_previous_game_ids": "",
        "accumulated_surprise": "",
        "surprise_group": "",
        "primary_eligible": "false",
        "exclusion_reason": "fewer_than_three_evidence_eligible_prior_games",
        "preseason_expected_rank": "25",
        "preseason_rank_sd": "20",
        "preseason_rank_strength_percentile": "0.75",
        "preseason_rank_uncertainty_fraction": "0.2",
        "preseason_rank_population": "100",
        "preseason_rank_top10_probability": "0.1",
        "preseason_rank_pmf_sha256": "pmf-hash",
    }


def test_coverage_and_away_orientation_keep_bounds_ordered() -> None:
    assert interval_coverage(1.0, 1.0, 2.0)
    assert not interval_coverage(0.99, 1.0, 2.0)
    assert orient_interval(-5.0, 11.0, is_home=False) == (-11.0, 5.0)
    with pytest.raises(ValueError, match="lower bound"):
        interval_coverage(0.0, 2.0, 1.0)


def test_progression_and_strength_groups_use_frozen_boundaries() -> None:
    assert progression_group(0) == "early_0_2"
    assert progression_group(2) == "early_0_2"
    assert progression_group(3) == "middle_3_6"
    assert progression_group(6) == "middle_3_6"
    assert progression_group(7) == "late_7_plus"
    cut_points = (0.2, 0.45, 0.8)
    assert strength_quartile(0.19, cut_points) == "q1_low"
    assert strength_quartile(0.2, cut_points) == "q1_low"
    assert strength_quartile(0.21, cut_points) == "q2_lower_middle"
    assert strength_quartile(0.45, cut_points) == "q2_lower_middle"
    assert strength_quartile(0.46, cut_points) == "q3_upper_middle"
    assert strength_quartile(0.8, cut_points) == "q3_upper_middle"
    assert strength_quartile(0.81, cut_points) == "q4_high"


def test_strength_quartile_reference_counts_distinct_consistent_team_seasons() -> None:
    rows = [
        {
            "model_family": family,
            "forecast_population": "fbs_fbs",
            "season": season,
            "team_id": team_id,
            "preseason_strength": strength,
        }
        for family in ("History", "Context")
        for season, team_id, strength in (
            (2022, "1", 0.1),
            (2022, "2", 0.2),
            (2023, "1", 0.3),
            (2023, "2", 0.4),
        )
    ]
    rows.append(rows[0].copy())

    reference = strength_quartile_reference(rows)

    assert reference["History"]["team_seasons"] == 4
    assert reference["History"]["q25"] == pytest.approx(0.175)
    assert reference["History"]["q50"] == pytest.approx(0.25)
    assert reference["History"]["q75"] == pytest.approx(0.325)
    assert reference["Context"] == reference["History"]

    inconsistent = rows + [{**rows[0], "preseason_strength": 0.9}]
    with pytest.raises(ValueError, match="inconsistent preseason strength"):
        strength_quartile_reference(inconsistent)


def test_join_builds_one_game_and_two_correctly_oriented_team_rows() -> None:
    scores = [_score(family) for family in ("History", "Context")]
    surprise = [
        _surprise_row(score, is_home=is_home)
        for score in scores
        for is_home in (True, False)
    ]
    evidence = {
        (2022, scores[0]["cutoff_utc"], "existing"): {
            "included_game_ids": [],
            "included_game_ids_sha256": issue201.sha256_json([]),
            "included_game_id_set": set(),
        }
    }

    forecast_rows, team_rows, audit = build_calibration_observations(
        scores, surprise, [], evidence
    )

    assert len(forecast_rows) == 2
    assert len(team_rows) == 4
    away = next(row for row in team_rows if row["team_id"] == "2")
    assert away["actual_margin"] == -3.0
    assert away["margin_interval_80_low"] == -11.0
    assert away["margin_interval_80_high"] == 5.0
    assert away["coverage_80"] is True
    assert audit["target_game_excluded_from_prior_histories"] is True


def test_prior_history_rejects_target_game_outcome() -> None:
    row = {
        "season": "2022",
        "model_arm": "History_existing",
        "team_id": "1",
        "game_id": "target",
        "cutoff_utc": "2022-10-10T00:00:00+00:00",
        "policy": "existing",
        "forecast_id": "target-forecast",
        "eligible_previous_game_ids": "prior|target",
        "eligible_prior_game_count": "2",
        "primary_eligible": "false",
        "accumulated_surprise": "0.1",
        "surprise_group": "",
    }
    included_ids = ["prior", "target"]
    score = {"evidence_ids_sha256": issue201.sha256_json(included_ids)}
    evidence = {
        (2022, row["cutoff_utc"], "existing"): {
            "included_game_ids_sha256": score["evidence_ids_sha256"],
            "included_game_id_set": set(included_ids),
        }
    }

    with pytest.raises(ValueError, match="target result appears"):
        _verify_prior_history(row, score, evidence, {})


def test_game_cluster_bootstrap_is_deterministic() -> None:
    rows = [
        {
            "season": 2022,
            "game_id": f"game-{index}",
            "home_id": str(index % 2),
            "away_id": str((index + 1) % 2),
        }
        for index in range(8)
    ]
    values = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    first = _bootstrap_draws(
        rows,
        values,
        mode="game",
        repetitions=30,
        seed=12,
        label="test",
    )
    second = _bootstrap_draws(
        rows,
        values,
        mode="game",
        repetitions=30,
        seed=12,
        label="test",
    )

    assert first.tolist() == second.tolist()
