"""Semantic guards for the fixed Lean Context research candidate."""

from __future__ import annotations

import hashlib
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


def test_validation_runner_preserves_frozen_2026_artifacts_and_excludes_2026_outcomes() -> (
    None
):
    frozen = {
        "history/annual/2026/predictions.csv": "0b3454a09288019e17739869c42aed3123fdda2163694f52f63bca65baf37f90",
        "context/annual/2026/predictions.csv": "641182890ec88ea8bc6150cc97047ddc688d0c680486d9fcc63a58d7bfae9132",
    }
    for relative, digest in frozen.items():
        assert (
            hashlib.sha256(
                (ROOT / "data/processed/preseason" / relative).read_bytes()
            ).hexdigest()
            == digest
        )
    source = (ROOT / "scripts/validate_lean_context.py").read_text()
    assert "TARGET_MAX = 2025" in source
    assert "annual/2026" not in source
