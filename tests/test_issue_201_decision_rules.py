from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_issue_201_prebowl_comparison import _result_classification

ARMS = (
    "History_existing",
    "Context_existing",
    "History_pre_bowl",
    "Context_pre_bowl",
)


def _result(
    context_policy_crps: tuple[float, float, float, float],
    history_policy_crps: tuple[float, float, float, float],
    *,
    context_nll_delta: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    context_coverage_delta: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for index, season in enumerate((2022, 2023, 2024, 2025)):
        bases = {
            "History_existing": 10.0,
            "Context_existing": 9.0,
            "History_pre_bowl": 10.0 + history_policy_crps[index],
            "Context_pre_bowl": 9.0 + context_policy_crps[index],
        }
        for arm in ARMS:
            row = {
                "forecast_stage": "rolling",
                "forecast_population": "fbs_fbs",
                "model_arm": arm,
                "season": str(season),
                "phase": "all_phases",
                "games": 100,
                "crps": bases[arm],
                "margin_nll": 4.0,
                "coverage_50": 0.50,
                "coverage_80": 0.80,
                "coverage_95": 0.95,
            }
            if arm == "Context_pre_bowl":
                row["margin_nll"] = 4.0 + context_nll_delta[index]
                row["coverage_80"] = 0.80 + context_coverage_delta[index]
            rows.append(row)
    for arm in ARMS:
        c = [row for row in rows if row["model_arm"] == arm]
        existing = [
            row for row in c if row["season"] in {"2022", "2023", "2024", "2025"}
        ]
        mean = sum(float(row["crps"]) for row in existing) / len(existing)
        for season_label in ("pooled_equal_game", "equal_season_sensitivity"):
            rows.append(
                {
                    "forecast_stage": "rolling",
                    "forecast_population": "fbs_fbs",
                    "model_arm": arm,
                    "season": season_label,
                    "phase": "all_phases",
                    "games": 400,
                    "crps": mean,
                    "margin_nll": 4.0,
                    "coverage_50": 0.50,
                    "coverage_80": 0.80,
                    "coverage_95": 0.95,
                }
            )
    return {"forecast_summaries": rows}


def test_result_rule_marks_improvement_only_when_all_crps_conditions_hold() -> None:
    result = _result(
        (-1.0, -1.0, -1.0, 1.0),
        (0.0, 0.0, 0.0, 0.0),
    )
    decision = _result_classification(result)
    assert decision["classification"] == "improve"
    assert decision["recommendation"].startswith("proceed")


def test_result_rule_marks_no_useful_improvement_when_both_aggregates_and_wins_fail() -> (
    None
):
    result = _result(
        (1.0, 1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0, 0.0),
    )
    decision = _result_classification(result)
    assert decision["classification"] == "no useful improvement"
    assert decision["recommendation"].startswith("defer")


def test_opposite_nll_or_calibration_direction_downgrades_an_improvement() -> None:
    result = _result(
        (-1.0, -1.0, -1.0, 1.0),
        (-0.5, -0.5, -0.5, 0.0),
        context_nll_delta=(0.1, 0.1, 0.1, 0.0),
    )
    decision = _result_classification(result)
    assert decision["classification"] == "inconclusive"
    assert decision["opposite_margin_nll_seasons"] == 3
