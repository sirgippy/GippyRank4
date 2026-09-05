"""Fixed-specification helpers for the Lean Context validation study.

This module intentionally contains no feature search.  Keeping the feature
contract and the prospective decision rule here makes the research script and
its semantic tests auditably small.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from gippyrank.preseason import TeamSeason

H_FEATURES = ("lag2_z_mean", "lag3_z_mean", "long_run_z_mean")
RECRUITING_FEATURES = (
    "recruiting_class_rank",
    "recruiting_class_points",
    "recruiting_points_2y_mean",
    "recruiting_points_3y_mean",
    "recruiting_points_4y_mean",
    "recruiting_points_trend",
)
RETURNING_FEATURES = (
    "returning_pct_ppa",
    "returning_pct_passing_ppa",
    "returning_pct_receiving_ppa",
    "returning_pct_rushing_ppa",
)
LEAN_CONTEXT_FEATURES = (*RECRUITING_FEATURES, *RETURNING_FEATURES)
FORBIDDEN_LEAN_FEATURES = frozenset(
    {
        "talent_composite",
        "coach_tenure_seasons",
        "coach_change",
        "transfer_count",
        "ol_continuity",
        "defensive_continuity",
    }
)

# These constants are declared before fitting or reading any evaluation output.
MIN_PRIOR_TARGETS = 3
MATERIAL_NLL_GAIN = 0.005
MAX_COVERAGE_REGRESSION = 0.05
PARSIMONY_TIE_NLL = 0.002
CLEAR_C_NLL_GAIN = 0.005
INTERVAL_80_COVERAGE_METRIC = "interval_80_coverage"


def annual_delta_field(left: str, right: str, metric: str) -> str:
    """Return the shared annual-record name for a model-pair metric delta."""
    return f"{left.lower()}_minus_{right.lower()}_{metric}"


def assert_lean_specification(features: Iterable[str]) -> None:
    """Reject any alteration to the candidate fixed by the validation brief."""
    supplied = tuple(features)
    if supplied != LEAN_CONTEXT_FEATURES:
        raise ValueError("Lean Context features must exactly match the predeclared set")
    if set(supplied) & FORBIDDEN_LEAN_FEATURES:
        raise ValueError("forbidden feature entered Lean Context")
    if any("interaction" in name for name in supplied):
        raise ValueError("interactions are forbidden in Lean Context")


def observed_common(rows: Iterable[TeamSeason]) -> list[TeamSeason]:
    """Choose the scientific comparison population before any imputation."""
    return [
        row
        for row in rows
        if all(row.features.get(name) is not None for name in LEAN_CONTEXT_FEATURES)
    ]


def prior_training(rows: Iterable[TeamSeason], target_season: int) -> list[TeamSeason]:
    """Return completed history only; target outcomes cannot leak into fitting."""
    result = [row for row in rows if row.season < target_season]
    if any(row.season >= target_season for row in result):
        raise ValueError("target season entered training")
    return result


def choose_nested_model(prior_rows: list[dict[str, float | int | str]]) -> str:
    """Apply the simple, predeclared prospective selection hierarchy.

    ``prior_rows`` contains one observed-common annual metric record per prior
    target and is deliberately the only input.  The returned label is selected
    before the current target's scores are examined.
    """
    if len(prior_rows) < MIN_PRIOR_TARGETS:
        return "H"

    def qualifies(label: str) -> bool:
        delta = np.asarray(
            [float(row[annual_delta_field(label, "H", "nll")]) for row in prior_rows]
        )
        crps = np.asarray(
            [float(row[annual_delta_field(label, "H", "crps")]) for row in prior_rows]
        )
        coverage = np.asarray(
            [
                float(row[annual_delta_field(label, "H", INTERVAL_80_COVERAGE_METRIC)])
                for row in prior_rows
            ]
        )
        return bool(
            delta.mean() <= -MATERIAL_NLL_GAIN
            and crps.mean() <= 0.0
            and coverage.mean() >= -MAX_COVERAGE_REGRESSION
        )

    eligible = [label for label in ("C", "L") if qualifies(label)]
    if not eligible:
        return "H"
    if len(eligible) == 1:
        return eligible[0]
    l_minus_c = np.mean(
        [float(row[annual_delta_field("L", "C", "nll")]) for row in prior_rows]
    )
    if abs(l_minus_c) <= PARSIMONY_TIE_NLL:
        return "L"
    return "L" if l_minus_c < 0 else "C"


def recommendation_outcome(
    *,
    l_minus_h_nll: float,
    l_minus_h_crps: float,
    l_minus_h_interval_80_coverage: float,
    l_minus_c_nll: float,
) -> str:
    """Classify L using the predeclared A/B/C/D promotion outcomes.

    A clear C win uses the same 0.005 NLL material-gain threshold as the H
    criterion. Borderline non-ties default to C rather than promoting L.
    """
    h_criteria_pass = (
        l_minus_h_nll <= -MATERIAL_NLL_GAIN
        and l_minus_h_crps <= 0.0
        and l_minus_h_interval_80_coverage >= -MAX_COVERAGE_REGRESSION
    )
    if not h_criteria_pass:
        return "D"
    if l_minus_c_nll <= -CLEAR_C_NLL_GAIN:
        return "A"
    if abs(l_minus_c_nll) <= PARSIMONY_TIE_NLL:
        return "B"
    return "C"
