from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_issue_201_prebowl_comparison import _result_classification, write_results

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
    context_coverage_error_delta: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
    ),
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
                "home_win_brier": 0.1,
                "coverage_50": 0.50,
                "coverage_80": 0.90,
                "coverage_95": 0.95,
            }
            if arm == "Context_pre_bowl":
                row["margin_nll"] = 4.0 + context_nll_delta[index]
                row["coverage_80"] = 0.90 + context_coverage_error_delta[index]
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
                    "home_win_brier": 0.1,
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


def test_three_worse_nll_seasons_downgrade_an_improvement() -> None:
    result = _result(
        (-1.0, -1.0, -1.0, 1.0),
        (-0.5, -0.5, -0.5, 0.0),
        context_nll_delta=(0.1, 0.1, -0.1, 0.1),
    )
    decision = _result_classification(result)
    assert decision["classification"] == "inconclusive"
    assert decision["worse_margin_nll_seasons"] == 3


def test_nll_improvement_in_crps_losing_season_does_not_count_as_a_regression() -> None:
    result = _result(
        (-1.0, -1.0, -1.0, 1.0),
        (-0.5, -0.5, -0.5, 0.0),
        context_nll_delta=(0.1, -0.1, -0.1, -0.1),
    )
    decision = _result_classification(result)
    assert decision["classification"] == "improve"
    assert decision["worse_margin_nll_seasons"] == 1


def test_calibration_regressions_are_counted_independently_of_crps_direction() -> None:
    result = _result(
        (-1.0, -1.0, -1.0, 1.0),
        (-0.5, -0.5, -0.5, 0.0),
        context_coverage_error_delta=(0.01, 0.01, -0.01, 0.01),
    )
    decision = _result_classification(result)
    assert decision["classification"] == "inconclusive"
    assert decision["worse_calibration_error_seasons"] == 3


def _fresh_result_for_serialization() -> dict[str, object]:
    result = _result(
        (-1.0, -1.0, -1.0, 1.0),
        (-0.5, -0.5, -0.5, 0.0),
    )
    summaries = result["forecast_summaries"]
    preseason_summary = []
    for arm in ARMS:
        summaries.append(
            {
                "forecast_stage": "rolling",
                "forecast_population": "fbs_fcs",
                "model_arm": arm,
                "season": "pooled_equal_game",
                "phase": "all_phases",
                "games": 10,
                "crps": 1.0,
                "margin_nll": 2.0,
                "home_win_brier": 0.1,
                "coverage_50": 0.5,
                "coverage_80": 0.8,
                "coverage_95": 0.95,
            }
        )
        preseason_summary.append(
            {
                "forecast_stage": "preseason",
                "forecast_population": "fbs_fbs",
                "model_arm": arm,
                "season": "pooled_equal_game",
                "phase": "all_phases",
                "games": 10,
                "crps": 1.5,
                "margin_nll": 2.5,
                "home_win_brier": 0.1,
                "coverage_50": 0.5,
                "coverage_80": 0.8,
                "coverage_95": 0.95,
            }
        )

    def score_row(
        stage: str, population: str, arm: str, season: str, crps: float
    ) -> dict[str, object]:
        return {
            "forecast_stage": stage,
            "forecast_population": population,
            "model_arm": arm,
            "season": season,
            "phase": "pre_thanksgiving",
            "crps": crps,
            "margin_nll": 2.0,
            "median_absolute_error": 3.0,
            "home_win_brier": 0.1,
            "coverage_50": 0.5,
            "coverage_80": 0.8,
            "coverage_95": 0.95,
            "margin_interval_50_low": -5.0,
            "margin_interval_50_high": 5.0,
            "margin_interval_80_low": -10.0,
            "margin_interval_80_high": 10.0,
            "margin_interval_95_low": -15.0,
            "margin_interval_95_high": 15.0,
            "crps_numerical_error_estimate": 1e-6,
            "crps_quadrature_order": 256,
            "crps_tail_multiple": 40.0,
        }

    forecast_scores = []
    for arm in ARMS:
        for season in ("2022", "2023", "2024", "2025"):
            primary = next(
                row
                for row in result["forecast_summaries"]
                if row["forecast_population"] == "fbs_fbs"
                and row["model_arm"] == arm
                and row["season"] == season
            )
            forecast_scores.append(
                score_row("rolling", "fbs_fbs", arm, season, float(primary["crps"]))
            )
        forecast_scores.extend(
            (
                score_row("rolling", "fbs_fcs", arm, "2022", 1.0),
                score_row("preseason", "fbs_fbs", arm, "2022", 1.5),
                score_row("preseason", "fbs_fcs", arm, "2022", 1.6),
            )
        )
    ranking_scores = [
        {
            "model_arm": arm,
            "nll": 1.0,
            "crps": 0.1,
            "expected_rank_error": 2.0,
            "interval_80_target_mass": 0.8,
        }
        for arm in ARMS
    ]
    result.update(
        {
            "preseason_summary": preseason_summary,
            "forecast_scores": forecast_scores,
            "forecast_exclusions": [],
            "paired_game_differences": [],
            "calendar_mismatches": [],
            "ranking_scores": ranking_scores,
            "ranking_summary": [],
            "ranking_exclusions": [],
            "frozen_scores": [],
            "frozen_summary": [],
            "frozen_paired_summary": [],
            "frozen_exclusions": [],
            "frozen_checkpoint_safety": [],
            "replay_evidence": [],
            "frozen_parity": {
                "frozen_pmf_identity_before": "before",
                "frozen_pmf_identity_after": "before",
                "score_rows_compared": 0,
                "max_absolute_metric_difference": 0.0,
            },
            "model_arms": list(ARMS),
            "protocol_sha256": "protocol",
            "input_artifacts": {},
            "issue200_provenance": {},
            "pre_bowl_target_provenance": {},
            "fit_origin_by_season_policy": {},
            "target_snapshot_dates": {},
            "schedule_window_dates": {},
            "ranking_common_population": {},
            "replay_consistency": {"forecast_row_count": len(forecast_scores)},
        }
    )
    return result


def test_fresh_score_result_serializes_preseason_report_and_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fresh-results"
    write_results(_fresh_result_for_serialization(), output)

    with (output / "forecast_summary.csv").open(newline="", encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    preseason_rows = [row for row in summaries if row["forecast_stage"] == "preseason"]
    assert len(preseason_rows) == 32
    assert {row["forecast_population"] for row in preseason_rows} == {
        "fbs_fbs",
        "fbs_fcs",
    }

    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Preseason-only FBS–FBS CRPS diagnostic" in report
    assert "quadrature orders [256]" in report
    assert "tail multiples [40.0]" in report

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"]["forecast_summary.csv"]["bytes"] > 0
    assert manifest["outputs"]["report.md"]["bytes"] > 0
