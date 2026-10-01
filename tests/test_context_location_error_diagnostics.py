"""Frozen-baseline and rolling-origin invariants for issue 152."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_context_location_error_diagnostics import (
    BASE,
    OUT,
    baseline_rows,
    db_coverage_status,
    feature_diagnostics,
    inventory,
    nearest_support,
    percentile_and_range,
    transfer_diagnostics,
    write_csv,
    write_json,
)

from gippyrank.context_prior_v1_3 import LOCATION_FEATURE_NAMES
from gippyrank.preseason import Preprocessor


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_feature_inventory_matches_fitting_contract_and_dataset_columns() -> None:
    features = inventory()
    assert tuple(row["feature_name"] for row in features) == LOCATION_FEATURE_NAMES
    emitted = rows(OUT / "feature_inventory.csv")
    assert [row["feature_name"] for row in emitted] == list(LOCATION_FEATURE_NAMES)
    columns = set(rows(OUT / "team_seasons.csv")[0])
    for name in LOCATION_FEATURE_NAMES:
        for suffix in (
            "raw",
            "transformed",
            "standardized",
            "missing",
            "coefficient",
            "contribution",
            "missing_contribution",
            "training_percentile",
            "training_range",
        ):
            assert f"feature_{name}_{suffix}" in columns


def test_147_population_prior_and_final_checkpoint_parity() -> None:
    prior, final, checkpoints, _ = baseline_rows()
    diagnostic = rows(OUT / "team_seasons.csv")
    original = {(int(r["season"]), r["team_id"]): r for r in prior}
    assert len(diagnostic) == len(original) == len(final) == 534
    assert {(int(r["season"]), r["team_id"]) for r in diagnostic} == set(original)
    for current in diagnostic:
        key = int(current["season"]), current["team_id"]
        old = original[key]
        assert int(current["final_shared_checkpoint_index"]) == checkpoints[key[0]]
        assert current["final_shared_checkpoint"] == final[key]["context"]["cutoff"]
        for family in ("context", "history"):
            assert float(current[f"{family}_preseason_nll"]) == float(
                old[f"{family}_prior_nll"]
            )
            assert float(current[f"{family}_preseason_crps"]) == float(
                old[f"{family}_prior_crps"]
            )
            assert float(current[f"{family}_final_nll"]) == float(
                final[key][family]["posterior_nll"]
            )
            assert float(current[f"{family}_final_expected_rank"]) == float(
                final[key][family]["posterior_expected_rank"]
            )


def test_contributions_signs_and_cutoffs() -> None:
    diagnostic = rows(OUT / "team_seasons.csv")
    fitted = [r for r in diagnostic if r["component_status"] == "fitted"]
    assert len(fitted) == 528
    for row in diagnostic:
        assert int(row["context_training_cutoff"]) == int(row["season"]) - 1
        assert int(row["history_training_cutoff"]) == int(row["season"]) - 1
        target = float(row["target_expected_rank"])
        context = float(row["context_preseason_expected_rank"])
        history = float(row["history_preseason_expected_rank"])
        assert float(row["context_preseason_signed_rank_error"]) == pytest.approx(
            context - target
        )
        assert float(row["context_minus_history_abs_rank_error"]) == pytest.approx(
            abs(context - target) - abs(history - target)
        )
        assert float(row["context_minus_history_final_nll"]) == pytest.approx(
            float(row["context_final_nll"]) - float(row["history_final_nll"])
        )
    for row in fitted:
        contributions = sum(
            float(row[f"feature_{name}_contribution"])
            + float(row[f"feature_{name}_missing_contribution"])
            for name in LOCATION_FEATURE_NAMES
        )
        rebuilt = (
            float(row["context_location_intercept"])
            + float(row["lag1_contribution"])
            + contributions
        )
        assert rebuilt == pytest.approx(
            float(row["context_fitted_location_center"]), abs=1e-10
        )
        assert abs(float(row["context_location_reconstruction_residual"])) < 1e-10
        assert float(row["context_fitted_location_center"]) == pytest.approx(
            float(row["context_location_center"]), abs=1e-8
        )


def test_training_only_percentile_and_nearest_support() -> None:
    training = np.array([-1.0, 0.0, 0.0, 1.0])
    assert percentile_and_range(0.0, training) == (50.0, "within training range")
    assert percentile_and_range(2.0, training) == (100.0, "above training maximum")
    assert percentile_and_range(-2.0, training) == (0.0, "below training minimum")
    points = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    assert nearest_support(np.array([0.2, 0.0]), points, k=2) == pytest.approx(
        (0.2, 0.5)
    )


def test_rolling_preprocessing_does_not_use_target_value() -> None:
    preprocessor = Preprocessor.fit([{"x": 0.0}, {"x": 2.0}], ["x"])
    assert preprocessor.medians["x"] == 1.0
    assert preprocessor.means["x"] == 1.0
    assert preprocessor.scales["x"] == 1.0
    assert preprocessor.transform([{"x": 100.0}]).tolist() == [[99.0, 0.0]]
    assert preprocessor.transform([{"x": None}]).tolist() == [[0.0, 1.0]]


def test_feature_diagnostics_rejects_target_season_training() -> None:
    class Row:
        season = 2022

    with pytest.raises(ValueError, match="training includes target"):
        feature_diagnostics(None, Row(), [Row()])


def test_transfer_coverage_and_zero_are_distinct_from_missing() -> None:
    assert db_coverage_status(0, 0, True) == "no incoming DB players"
    assert db_coverage_status(2, 2, True) == "complete"
    assert db_coverage_status(2, 1, True) == "partial"
    assert db_coverage_status(2, 0, True) == "unavailable"
    assert db_coverage_status(0, 0, False) == "unavailable"
    evidence = {
        "provenance_class": "retrospective_research_reconstruction",
        "checkpoint_status": "historical_timing_unverified",
        "repair_class": "none",
        "availability_status": "partial",
        "incoming_transfers": "1",
        "db_incoming": "0",
        "db_resolved": "0",
        "offensive_resolved": "0",
        "offensive_join_failed": "0",
        "offensive_applicability_unknown": "1",
        "post_repair_observed_usage_sum": "",
        "source_coverage_gap": "False",
        "primary_reason": "offense_applicability_unproven",
    }
    fixed = {
        name: "0.0"
        for name in (
            "transfer_in_prior_usage_sum",
            "transfer_in_prior_defensive_impact_db_sum",
            "transfer_in_prior_defensive_impact_db_available",
        )
    }
    output = transfer_diagnostics(
        2022,
        "1",
        {"transfer_in_prior_usage_sum": None},
        {(2022, "1"): evidence},
        {(2022, "1"): fixed},
        {},
        {},
    )
    assert output["incoming_offensive_observed_usage_sum"] == ""
    assert output["incoming_offensive_applicability_unknown_count"] == 1
    assert output["db_coverage_status"] == "no incoming DB players"
    assert output["transfer_in_prior_usage_sum_model_input_value"] == ""
    assert output["transfer_in_prior_usage_sum_corrected_diagnostic_value"] == 0.0


def test_retained_db_observations_preserve_coverage_states() -> None:
    diagnostic = rows(OUT / "team_seasons.csv")
    for row in diagnostic:
        status = row["db_coverage_status"]
        observed_sum = row["observed_db_impact_sum"]
        if status == "unavailable":
            assert observed_sum == ""
        elif status == "no incoming DB players":
            assert observed_sum == "0.0"
        else:
            assert observed_sum != ""


def test_artifact_summary_and_provenance_are_deterministic() -> None:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((OUT / "provenance.json").read_text(encoding="utf-8"))
    assert summary["team_seasons_included"] == 534
    assert summary["fitted_decompositions"] == 528
    assert summary["baseline_reproduction_checks"] == 700
    assert summary["missing_diagnostic_fields"]["fitted_context_location_center"] == 6
    assert "timestamp" not in provenance
    assert all(len(value) == 64 for value in provenance["source_hashes"].values())
    assert (BASE / "provenance.json").is_file()


def test_machine_readable_writer_is_byte_deterministic(tmp_path: Path) -> None:
    artifact = [{"season": 2022, "team_id": "1", "value": 0.0}]
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    write_csv(first, artifact)
    write_csv(second, artifact)
    assert first.read_bytes() == second.read_bytes()
    write_json(tmp_path / "a.json", {"b": 2, "a": 1})
    write_json(tmp_path / "b.json", {"a": 1, "b": 2})
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
