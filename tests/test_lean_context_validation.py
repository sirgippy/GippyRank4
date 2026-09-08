"""Semantic guards for the fixed Lean Context research candidate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from gippyrank.lean_context_validation import (
    LEAN_CONTEXT_FEATURES,
    annual_delta_field,
    assert_lean_specification,
    choose_nested_model,
    observed_common,
    prior_training,
    recommendation_outcome,
)
from gippyrank.preseason import TeamSeason

ROOT = Path(__file__).parents[1]

def row(season: int, team: str, complete: bool = True) -> TeamSeason:
    features = {name: 1.0 for name in LEAN_CONTEXT_FEATURES}
    if not complete:
        features[LEAN_CONTEXT_FEATURES[0]] = None
    return TeamSeason(
        season,
        "fbs",
        team,
        team,
        2,
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([1]),
        features,
    )


def decision_row(
    *,
    l_h: float,
    c_h: float,
    l_c: float,
    l_crps: float = 0,
    c_crps: float = 0,
    l_coverage: float = 0,
    c_coverage: float = 0,
) -> dict[str, float]:
    return {
        annual_delta_field("L", "H", "nll"): l_h,
        annual_delta_field("C", "H", "nll"): c_h,
        annual_delta_field("L", "C", "nll"): l_c,
        annual_delta_field("L", "H", "crps"): l_crps,
        annual_delta_field("C", "H", "crps"): c_crps,
        annual_delta_field("L", "H", "interval_80_coverage"): l_coverage,
        annual_delta_field("C", "H", "interval_80_coverage"): c_coverage,
    }


def test_lean_features_are_exact_and_forbidden_additions_fail() -> None:
    assert_lean_specification(LEAN_CONTEXT_FEATURES)
    with pytest.raises(ValueError, match="exactly"):
        assert_lean_specification((*LEAN_CONTEXT_FEATURES, "talent_composite"))
    with pytest.raises(ValueError, match="exactly"):
        assert_lean_specification(LEAN_CONTEXT_FEATURES[:-1])


def test_observed_population_is_set_before_imputation() -> None:
    assert [
        value.team_id
        for value in observed_common([row(2020, "a"), row(2020, "b", False)])
    ] == ["a"]


def test_training_rows_never_contain_target_or_future() -> None:
    result = prior_training([row(2020, "a"), row(2021, "b"), row(2022, "c")], 2021)
    assert [value.season for value in result] == [2020]


def test_nested_selector_uses_only_prior_metrics_and_prefers_parsimony_tie() -> None:
    prior = [decision_row(l_h=-0.01, c_h=-0.011, l_c=0.001) for _ in range(3)]
    assert choose_nested_model(prior) == "L"
    assert choose_nested_model(prior[:2]) == "H"


def test_nested_selector_rejects_candidate_with_bad_calibration() -> None:
    prior = [
        decision_row(l_h=-0.02, c_h=0.0, l_c=-0.02, l_coverage=-0.06) for _ in range(3)
    ]
    assert choose_nested_model(prior) == "H"


def test_nested_selector_consumes_actual_annual_record_schema_after_minimum_history() -> (
    None
):
    prior = [decision_row(l_h=-0.01, c_h=-0.011, l_c=0.001) for _ in range(3)]
    assert all("interval_80_coverage" in key for key in prior[0] if "coverage" in key)
    assert choose_nested_model(prior) == "L"


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            {
                "l_minus_h_nll": -0.01,
                "l_minus_h_crps": -0.001,
                "l_minus_h_interval_80_coverage": 0.0,
                "l_minus_c_nll": -0.01,
            },
            "A",
        ),
        (
            {
                "l_minus_h_nll": -0.01,
                "l_minus_h_crps": -0.001,
                "l_minus_h_interval_80_coverage": 0.0,
                "l_minus_c_nll": 0.001,
            },
            "B",
        ),
        (
            {
                "l_minus_h_nll": -0.01,
                "l_minus_h_crps": -0.001,
                "l_minus_h_interval_80_coverage": 0.0,
                "l_minus_c_nll": 0.01,
            },
            "C",
        ),
        (
            {
                "l_minus_h_nll": -0.001,
                "l_minus_h_crps": 0.001,
                "l_minus_h_interval_80_coverage": 0.0,
                "l_minus_c_nll": -0.01,
            },
            "D",
        ),
    ],
)
def test_recommendation_outcomes_are_exhaustive(
    values: dict[str, float], expected: str
) -> None:
    assert recommendation_outcome(**values) == expected


def test_synthetic_smoke_reaches_nested_selection_and_report(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from validate_lean_context import run_smoke

        result = run_smoke(tmp_path)
    finally:
        sys.path.pop(0)
    assert result["synthetic_only_not_evidence"] is True
    assert result["nested_selection"][-1]["n_prior_target_seasons"] == 3
    assert result["nested_selection"][-1]["selected_model"] == "L"
    assert result["recommendation_outcome"] == "B"
    assert (tmp_path / "smoke_report.md").exists()
