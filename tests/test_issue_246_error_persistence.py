from __future__ import annotations

from typing import Any

import pytest

from gippyrank.research.issue_246_error_persistence import (
    build_lagged_pairs,
    build_team_game_errors,
    lag_statistics,
    preseason_favorite_group,
    season_progress_bin,
    signed_error_from_home_margin,
)


def _score_row(
    game_id: str,
    *,
    home_id: str,
    away_id: str,
    home_subdivision: str,
    away_subdivision: str,
    kickoff: str,
    cutoff: str,
    home_margin: float,
    predicted_margin: float,
    evidence_hash: str,
    population: str,
) -> dict[str, str]:
    return {
        "season": "2022",
        "model_arm": "Context_existing",
        "prior_family": "Context",
        "policy": "existing",
        "forecast_stage": "rolling",
        "forecast_population": population,
        "game_id": game_id,
        "forecast_id": f"forecast-{game_id}",
        "home_id": home_id,
        "home_team": f"team-{home_id}",
        "home_subdivision": home_subdivision,
        "away_id": away_id,
        "away_team": f"team-{away_id}",
        "away_subdivision": away_subdivision,
        "kickoff_utc": kickoff,
        "cutoff_utc": cutoff,
        "actual_home_points": str(30 + home_margin),
        "actual_away_points": "30",
        "actual_home_margin": str(home_margin),
        "expected_home_margin": str(predicted_margin),
        "evidence_game_count": "1",
        "evidence_ids_sha256": evidence_hash,
    }


def _preseason_row(rolling: dict[str, str]) -> dict[str, str]:
    return {
        **rolling,
        "forecast_stage": "preseason",
        "cutoff_utc": "preseason_no_evidence",
        "forecast_id": f"preseason-{rolling['game_id']}",
        "home_win_probability": "0.85",
        "away_win_probability": "0.15",
    }


def test_signed_error_is_observed_minus_expected_in_team_orientation() -> None:
    assert signed_error_from_home_margin(10.0, 6.0, is_home=True) == 4.0
    assert signed_error_from_home_margin(10.0, 6.0, is_home=False) == -4.0


def test_preseason_favorite_and_progress_boundaries_are_fixed() -> None:
    assert preseason_favorite_group(0.80) == "strong_preseason_favorite"
    assert preseason_favorite_group(0.799) == "other"
    assert season_progress_bin(2) == "early_games_2_4"
    assert season_progress_bin(4) == "early_games_2_4"
    assert season_progress_bin(5) == "middle_games_5_8"
    assert season_progress_bin(8) == "middle_games_5_8"
    assert season_progress_bin(9) == "late_game_9_plus"


def test_lag_statistics_measure_correlation_and_directional_persistence() -> None:
    rows = [
        {"previous_signed_error": -4.0, "next_signed_error": -2.0},
        {"previous_signed_error": 1.0, "next_signed_error": 0.5},
        {"previous_signed_error": 3.0, "next_signed_error": 1.5},
    ]
    result = lag_statistics(rows)
    assert result["transitions"] == 3
    assert result["lag1_correlation"] == pytest.approx(1.0)
    assert result["directional_persistence"] == pytest.approx(1.0)


def test_fbs_fcs_game_keeps_only_fbs_orientation_and_lag_requires_evidence() -> None:
    evidence = {
        (2022, "2022-09-06T00:00:00+00:00", "existing"): {
            "included_game_ids": ["g1"],
            "included_game_id_set": {"g1"},
        },
        (2022, "2022-09-13T00:00:00+00:00", "existing"): {
            "included_game_ids": ["g2"],
            "included_game_id_set": {"g2"},
        },
        (2022, "2022-09-20T00:00:00+00:00", "existing"): {
            "included_game_ids": [],
            "included_game_id_set": set(),
        },
    }
    rolling: list[dict[str, str]] = [
        _score_row(
            "g1",
            home_id="1",
            away_id="2",
            home_subdivision="fbs",
            away_subdivision="fbs",
            kickoff="2022-09-03T20:00:00+00:00",
            cutoff="2022-09-02T00:00:00+00:00",
            home_margin=8,
            predicted_margin=5,
            evidence_hash="empty",
            population="fbs_fbs",
        ),
        _score_row(
            "g2",
            home_id="900",
            away_id="1",
            home_subdivision="fcs",
            away_subdivision="fbs",
            kickoff="2022-09-10T20:00:00+00:00",
            cutoff="2022-09-06T00:00:00+00:00",
            home_margin=-7,
            predicted_margin=-3,
            evidence_hash="g1-hash",
            population="fbs_fcs",
        ),
        _score_row(
            "g3",
            home_id="1",
            away_id="3",
            home_subdivision="fbs",
            away_subdivision="fbs",
            kickoff="2022-09-17T20:00:00+00:00",
            cutoff="2022-09-13T00:00:00+00:00",
            home_margin=4,
            predicted_margin=2,
            evidence_hash="g2-hash",
            population="fbs_fbs",
        ),
    ]
    for record, evidence_hash in zip(rolling, ("empty", "g1-hash", "g2-hash")):
        evidence[(2022, record["cutoff_utc"], "existing")] = {
            "included_game_ids": [],
            "included_game_id_set": set(),
            "included_game_ids_sha256": evidence_hash,
        }
    evidence[(2022, "2022-09-06T00:00:00+00:00", "existing")][
        "included_game_id_set"
    ] = {"g1"}
    evidence[(2022, "2022-09-13T00:00:00+00:00", "existing")][
        "included_game_id_set"
    ] = {"g2"}

    team_games = build_team_game_errors(
        rolling,
        [_preseason_row(row) for row in rolling],
        evidence,
    )
    team_one = [row for row in team_games if row["team_id"] == "1"]
    assert [row["game_id"] for row in team_one] == ["g1", "g2", "g3"]
    assert len(team_games) == 5

    pairs = build_lagged_pairs(team_games, evidence)
    team_one_pairs = [row for row in pairs if row["team_id"] == "1"]
    assert len(team_one_pairs) == 2
    assert all(row["eligible"] for row in team_one_pairs)
    assert team_one_pairs[0]["previous_game_id"] == "g1"
    assert team_one_pairs[1]["previous_game_id"] == "g2"


def test_missing_previous_game_from_next_evidence_excludes_transition() -> None:
    team_rows: list[dict[str, Any]] = []
    for ordinal, game_id in enumerate(("g1", "g2"), start=1):
        team_rows.append(
            {
                "season": 2022,
                "model_arm": "Context_existing",
                "prior_family": "Context",
                "policy": "existing",
                "team_id": "1",
                "team": "A",
                "game_id": game_id,
                "kickoff_utc": f"2022-09-0{ordinal * 3}T20:00:00+00:00",
                "cutoff_utc": f"2022-09-0{ordinal * 3 - 1}T00:00:00+00:00",
                "team_game_number": ordinal,
                "actual_margin": 1.0,
                "predicted_margin": 0.0,
                "signed_error": 1.0,
                "preseason_win_probability": 0.6,
                "preseason_favorite_group": "other",
            }
        )
    evidence = {
        (2022, "2022-09-05T00:00:00+00:00", "existing"): {"included_game_id_set": set()}
    }
    pair = build_lagged_pairs(team_rows, evidence)[0]
    assert not pair["eligible"]
    assert (
        pair["exclusion_reason"] == "previous_game_absent_from_next_forecast_evidence"
    )
