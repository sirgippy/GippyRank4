from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from gippyrank.research.context_incremental_information import cross_message_arms
from gippyrank.research.issue_194_part_a import (
    EXPECTED_SAFE_CUTOFFS,
    PAIR_FEATURE_NAMES,
    _crossed_effect_rows,
    _match_schedule_pairs,
    _mean_by_checkpoint,
    _summarize_checkpoint_scores,
    signed_midpoint_cdf_residual,
)


def test_signed_midpoint_cdf_residual_uses_target_weighted_midpoints() -> None:
    forecast = np.array([0.5, 0.5])

    assert signed_midpoint_cdf_residual(forecast, [1.0, 0.0]) == pytest.approx(-0.25)
    assert signed_midpoint_cdf_residual(forecast, [0.0, 1.0]) == pytest.approx(0.25)


def test_signed_midpoint_cdf_residual_rejects_invalid_pmfs() -> None:
    with pytest.raises(ValueError, match="normalized PMFs"):
        signed_midpoint_cdf_residual([0.25, 0.25], [0.5, 0.5])


def test_crossed_effects_decompose_each_checkpoint_metric() -> None:
    fields = {
        "HH": (1.0, 2.0, 0.8, 2.0, 0.5),
        "CH": (2.0, 3.0, 0.7, 3.0, 0.4),
        "HC": (4.0, 5.0, 0.6, 4.0, 0.3),
        "CC": (8.0, 8.0, 0.5, 7.0, 0.2),
    }
    score_rows = []
    for arm, values in fields.items():
        nll, crps, mass, width, hhi = values
        score_rows.append(
            {
                "season": 2024,
                "cutoff_utc": "2024-09-19T23:59:59+00:00",
                "checkpoint_phase": "other_checkpoint",
                "team_id": "1",
                "arm": arm,
                "nll": nll,
                "crps": crps,
                "interval_80_target_mass": mass,
                "interval_80_width": width,
                "hhi_concentration": hhi,
            }
        )

    effects = _crossed_effect_rows(score_rows)
    nll = next(row for row in effects if row["metric"] == "rank_nll")

    assert nll["direct_prior"] == pytest.approx(1.0)
    assert nll["transmitted"] == pytest.approx(3.0)
    assert nll["interaction"] == pytest.approx(3.0)
    assert nll["native_context_minus_history"] == pytest.approx(7.0)


def test_pair_matching_is_deterministic_and_one_to_one() -> None:
    pair_keys = [
        ("100", "200"),
        ("100", "300"),
        ("100", "400"),
        ("200", "300"),
        ("200", "400"),
        ("300", "400"),
    ]
    feature_matrix = np.array(
        [
            [0.10, 0.20, 0.30, 0.40, 10.0, 2.0],
            [0.12, 0.18, 0.31, 0.35, 11.0, 1.0],
            [0.80, 0.75, 0.10, 0.12, 12.0, 3.0],
            [0.30, 0.25, 0.70, 0.55, 9.0, 2.0],
            [0.50, 0.50, 0.50, 0.50, 8.0, 4.0],
            [0.60, 0.70, 0.20, 0.25, 7.0, 3.0],
        ]
    )
    base_rows = [
        {
            "season": 2024,
            "pair_id": f"2024:{left}:{right}",
            "team_id_1": left,
            "team_id_2": right,
            "is_scheduled_opponent_pair": pair in {("100", "200"), ("300", "400")},
        }
        for pair in pair_keys
        for left, right in [pair]
    ]
    first_rows = deepcopy(base_rows)
    second_rows = deepcopy(base_rows)

    _match_schedule_pairs(first_rows, pair_keys, feature_matrix, season=2024)
    _match_schedule_pairs(second_rows, pair_keys, feature_matrix, season=2024)

    first_matches = {
        row["pair_id"]: row["matched_control_pair_id"]
        for row in first_rows
        if row["match_status"] == "matched_opponent"
    }
    second_matches = {
        row["pair_id"]: row["matched_control_pair_id"]
        for row in second_rows
        if row["match_status"] == "matched_opponent"
    }
    assert first_matches == second_matches
    assert len(first_matches) == 2
    assert len(set(first_matches.values())) == len(first_matches)
    assert not set(first_matches.values()) & set(first_matches)
    match_ids = {
        row["match_id"]
        for row in first_rows
        if row["match_status"] == "matched_opponent"
    }
    assert all(match_id.startswith("match-2024-") for match_id in match_ids)
    assert len(PAIR_FEATURE_NAMES) == feature_matrix.shape[1]


def test_checkpoint_summary_weights_checkpoint_means_equally() -> None:
    rows = [
        {"cutoff_utc": "2024-08-24T23:59:59+00:00", "nll": 1.0},
        {"cutoff_utc": "2024-08-24T23:59:59+00:00", "nll": 3.0},
        {"cutoff_utc": "2024-09-19T23:59:59+00:00", "nll": 10.0},
    ]

    assert _mean_by_checkpoint(rows, "nll") == pytest.approx(6.0)


def test_checkpoint_summary_skips_pooled_future_game_rows() -> None:
    phase_rows, season_rows = _summarize_checkpoint_scores(
        [],
        [],
        [
            {
                "forecast_stage": "rolling",
                "forecast_population": "fbs_fbs",
                "season": "pooled_equal_game",
                "phase": "other_regular_season",
                "model_arm": "History_existing",
            }
        ],
    )

    assert phase_rows == []
    assert season_rows == []


def test_cross_message_support_guard_rejects_different_positive_support() -> None:
    with pytest.raises(ValueError, match="different positive support"):
        cross_message_arms(
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.4, 0.6, 0.0],
            [0.6, 0.0, 0.4],
        )


def test_safe_checkpoint_plan_has_six_cutoffs_per_season() -> None:
    assert set(EXPECTED_SAFE_CUTOFFS) == {2022, 2023, 2024, 2025}
    assert all(len(cutoffs) == 6 for cutoffs in EXPECTED_SAFE_CUTOFFS.values())
    assert sum(map(len, EXPECTED_SAFE_CUTOFFS.values())) == 24
