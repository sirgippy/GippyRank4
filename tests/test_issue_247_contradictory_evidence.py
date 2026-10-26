from __future__ import annotations

import pytest

from gippyrank.research.issue_246_error_persistence import rank_pmf_summary
from gippyrank.research.issue_247_contradictory_evidence import (
    build_observations,
    standardized_margin_error,
    surprise_group,
)


def _team_game(number: int, surprise: float) -> dict[str, object]:
    game_id = f"g{number}"
    return {
        "season": 2022,
        "model_arm": "History_existing",
        "prior_family": "History",
        "policy": "existing",
        "team_id": "team-1",
        "team": "Team One",
        "opponent_id": f"opponent-{number}",
        "opponent": f"Opponent {number}",
        "game_id": game_id,
        "forecast_id": f"forecast-{game_id}",
        "kickoff_utc": f"2022-10-{number * 3:02}T20:00:00+00:00",
        "cutoff_utc": f"2022-10-{number * 3 - 1:02}T00:00:00+00:00",
        "is_home": True,
        "forecast_population": "fbs_fbs",
        "team_game_number": number,
        "actual_margin": 10.0,
        "predicted_margin": 10.0 - surprise,
        "signed_error": surprise,
        "margin_80_interval_half_width": 10.0,
        "standardized_surprise": surprise,
        "preseason_expected_rank": 50.0,
        "preseason_rank_sd": 20.0,
        "preseason_rank_strength_percentile": 0.5,
        "preseason_rank_uncertainty_fraction": 0.2,
        "preseason_rank_population": 100,
        "preseason_rank_top10_probability": 0.1,
        "preseason_rank_pmf_sha256": f"pmf-{game_id}",
    }


def test_standardized_surprise_uses_interval_half_width_units() -> None:
    assert standardized_margin_error(14.0, 7.0) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="positive"):
        standardized_margin_error(1.0, 0.0)


def test_surprise_groups_use_frozen_symmetric_thresholds() -> None:
    assert surprise_group(-0.5) == "sustained_negative"
    assert surprise_group(-0.499) == "approximately_neutral"
    assert surprise_group(0.499) == "approximately_neutral"
    assert surprise_group(0.5) == "sustained_positive"


def test_preseason_rank_summary_retains_continuous_location_and_uncertainty() -> None:
    summary = rank_pmf_summary([0.5, 0.0, 0.5, 0.0])
    assert summary["preseason_expected_rank"] == pytest.approx(2.0)
    assert summary["preseason_rank_sd"] == pytest.approx(1.0)
    assert summary["preseason_rank_strength_percentile"] == pytest.approx(2 / 3)
    assert summary["preseason_rank_uncertainty_fraction"] == pytest.approx(1 / 3)
    assert summary["preseason_rank_population"] == 4


def test_history_uses_only_prior_games_in_the_future_forecasts_evidence() -> None:
    team_games = [_team_game(number, number / 10) for number in range(1, 6)]
    evidence = {
        (2022, row["cutoff_utc"], "existing"): {
            "included_game_id_set": set(previous_ids)
        }
        for row, previous_ids in zip(
            team_games,
            ([], ["g1"], ["g1", "g2"], ["g1", "g2", "g3"], ["g1", "g3", "g4"]),
            strict=True,
        )
    }

    observations = build_observations(team_games, evidence)
    target = next(row for row in observations if row["game_id"] == "g5")

    assert target["candidate_previous_game_count"] == 4
    assert target["eligible_prior_game_count"] == 3
    assert target["excluded_previous_game_count"] == 1
    assert target["eligible_previous_game_ids"] == "g1|g3|g4"
    assert target["accumulated_surprise"] == pytest.approx(0.8 / 3)
    assert target["primary_eligible"] is True
    assert target["surprise_group"] == "approximately_neutral"
