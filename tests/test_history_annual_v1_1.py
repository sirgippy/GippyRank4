"""History-owned annual provenance, semantic, and retained V1.1 golden checks."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import gippyrank.history_annual_v1_1 as history
import gippyrank.preseason as fit_semantics
from gippyrank.preseason import (
    DirectRankModel,
    GenericRankPrior,
    Preprocessor,
    TeamSeason,
    conditional_rank_mixture_pmf,
    normal_pmf,
    rank_bin_edges,
    rank_to_z,
)

ROOT = Path(__file__).resolve().parents[1]
HISTORY_PREDICTIONS = ROOT / "data/processed/preseason/history/predictions.csv"
HISTORY_REPORT = ROOT / "data/processed/preseason/history/model_report.json"
ORIGINAL_PREDICTIONS = ROOT / "data/processed/preseason/rank_prior_predictions.csv"
RETAINED_INPUTS = ROOT / "data/processed/context_v1_4_candidate/development_model_inputs.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _retained_model(metadata: dict[str, object]) -> DirectRankModel:
    preprocessing = metadata["preprocessing"]
    assert isinstance(preprocessing, dict)
    return DirectRankModel(
        feature_names=list(metadata["feature_names"]),
        preprocessor=Preprocessor(
            tuple(preprocessing["feature_names"]),
            dict(preprocessing["medians"]),
            dict(preprocessing["means"]),
            dict(preprocessing["scales"]),
        ),
        beta=np.asarray(metadata["location_coefficients"], dtype=float),
        gamma=np.asarray(metadata["log_scale_coefficients"], dtype=float),
        minimum_scale=float(metadata["minimum_scale"]),
        penalty=float(metadata["penalty"]),
        optimizer=dict(metadata["optimizer"]),
        lag_count=int(metadata["lag_count"]),
        family=str(metadata["family"]),
        degrees_of_freedom=metadata["degrees_of_freedom"],
    )


def test_history_import_does_not_import_context14_candidate() -> None:
    script = (
        "import sys; import gippyrank.history_annual_v1_1; "
        "assert 'gippyrank.context_prior_v1_4_candidate' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert history.HistoryAnnualArtifactSource.__module__ == (
        "gippyrank.history_annual_v1_1"
    )


def test_retained_2026_history_artifacts_keep_their_legacy_identity() -> None:
    annual = ROOT / "data/processed/preseason/history/annual/2026"
    predictions = annual / "predictions.csv"
    fitted_instance = annual / "fitted_instance.json"
    assert _sha256(predictions) == (
        "0b3454a09288019e17739869c42aed3123fdda2163694f52f63bca65baf37f90"
    )
    assert _sha256(fitted_instance) == (
        "583b2583ae30e9056266bc07a4cb034cd9f2a4c36ac4d217a35603c1aa1ece91"
    )
    source = history.load_validated_history_annual_artifact(
        predictions, fitted_instance, target_season=2026, trained_through_season=2025
    )
    assert source.provenance_class == "retained_legacy_history_artifact"
    assert source.population == 138
    assert source.model_metadata_sha256 == (
        "159423c81d5f9bc5d12b8ccf65c1e185512a1a5ba73e23fad86108fdebef314d"
    )
    ordinary = source.prediction_row("2")["producer"]
    transition = source.prediction_row("16")["producer"]
    assert ordinary["producer_kind"] == "direct_rank_model"
    assert ordinary["producer_model_metadata_sha256"] == source.model_metadata_sha256
    assert transition["producer_identity_status"] == "unavailable_in_retained_legacy_artifact"
    assert transition["producer_model_metadata_sha256"] is None
    assert transition["producer_identity_sha256"] is None
    rows = list(csv.DictReader(predictions.open()))
    assert Counter(row["prior_method"] for row in rows) == {
        "same_subdivision_lag1": 136,
        "learned_fcs_to_fbs_transition": 2,
    }


def test_history11_retained_2022_2025_golden_parity() -> None:
    """An older V1.1 fit and input fixture independently pin the annual predictor.

    The original FCS lag distributions are unavailable: six transition rows
    are checked against two retained pre-builder artifacts, while the 528
    ordinary rows are recalculated through the new annual prediction arm.
    No retained generic FBS output exists in this panel.
    """
    assert _sha256(HISTORY_PREDICTIONS) == (
        "474f5587b8db44847bc375931e4f4937e7e8029077d5814af3f6b6e55056a706"
    )
    assert _sha256(ORIGINAL_PREDICTIONS) == (
        "52c574c159eff0e9c85da240383648a160e01c28996670e7e4d0e617459010cc"
    )
    assert _sha256(RETAINED_INPUTS) == (
        "8014a8dc0bd260740224f227a5f80a412a843d7205ae758b131669501bada601"
    )
    metadata = json.loads(HISTORY_REPORT.read_text())["final_test"][
        "selected_fit_metadata"
    ]
    assert _json_sha256(metadata) == (
        "339323e015683da2c58fa0e6c3f7c2f624accb2ed3da8114098351fa3dc78447"
    )
    model = _retained_model(metadata)
    assert json.loads(json.dumps(model.metadata())) == metadata
    reference_rows = list(csv.DictReader(HISTORY_PREDICTIONS.open()))
    original_rows = {
        (row["season"], row["team_id"]): row
        for row in csv.DictReader(ORIGINAL_PREDICTIONS.open())
        if row["model"] == "V1_1_long_run_baseline"
    }
    reference = {(row["season"], row["team_id"]): row for row in reference_rows}
    assert len(reference_rows) == len(reference) == len(original_rows) == 534
    assert set(reference) == set(original_rows)
    counts = Counter(row["prior_method"] for row in reference_rows)
    assert counts == {
        "same_subdivision_lag1": 528,
        "learned_fcs_to_fbs_transition": 6,
    }
    for key, row in reference.items():
        prior = original_rows[key]
        assert row["subdivision"] == prior["subdivision"] == "fbs"
        assert row["team_name"] == prior["team_name"]
        assert row["prior_method"] == prior["prior_method"]
        assert row["pmf"] == prior["pmf"]

    seasons = json.loads(RETAINED_INPUTS.read_text())["seasons"]
    generated: dict[tuple[str, str], dict[str, object]] = {}
    max_absolute_difference = 0.0
    for season, fixture in seasons.items():
        for row in fixture["fitted_rows"]:
            target = history._TargetInput(
                row["team_id"],
                row["team_name"],
                row["population"],
                np.asarray(row["lag1_z"], dtype=float),
                row["features"],
                None,
                None,
            )
            predicted = history._predict_annual_team(
                target, int(season), model, None, None
            )
            key = (season, row["team_id"])
            retained = reference[key]
            assert predicted["team_name"] == retained["team_name"]
            assert predicted["prior_method"] == retained["prior_method"]
            assert predicted["model_family"] == retained["model_family"]
            assert predicted["spec_version"] == retained["spec_version"]
            actual_pmf = np.asarray(json.loads(predicted["pmf"]), dtype=float)
            retained_pmf = np.asarray(json.loads(retained["pmf"]), dtype=float)
            assert len(actual_pmf) == len(retained_pmf) == row["population"]
            max_absolute_difference = max(
                max_absolute_difference,
                float(np.max(np.abs(actual_pmf - retained_pmf))),
            )
            assert predicted["pmf"] == retained["pmf"]
            generated[key] = predicted
    expected_fitted_keys = {
        key for key, row in reference.items()
        if row["prior_method"] == "same_subdivision_lag1"
    }
    assert set(generated) == expected_fitted_keys
    assert len(generated) == 528
    assert max_absolute_difference == 0.0


@pytest.mark.parametrize(
    ("module", "constant", "changed"),
    [
        (history, "HISTORY_1_1_FEATURES", ("lag2_z_mean",)),
        (history, "HISTORY_1_1_TRANSITION_FEATURES", ("lag2_z_mean",)),
        (history, "HISTORY_1_1_PENALTY", 0.3),
        (history, "HISTORY_1_1_MINIMUM_SCALE", 0.11),
        (history, "HISTORY_1_1_ROW_WEIGHT", 2.0),
        (history, "HISTORY_1_1_PMF_DECIMALS", 11),
        (history, "HISTORY_1_1_HISTORY_START_SEASON", 2003),
        (history, "HISTORY_1_1_CONSTITUENT_RANK_FILTER_SEMANTICS_VERSION", 2),
        (fit_semantics, "DIRECT_RANK_OPTIMIZER_METHOD", "BFGS"),
        (fit_semantics, "DIRECT_RANK_INITIAL_LAG_BETA", 0.56),
        (fit_semantics, "DIRECT_RANK_GAMMA_COEFFICIENT_BOUNDS", (-4.0, 4.0)),
        (fit_semantics, "DIRECT_RANK_LOG_SCALE_CLIP_BOUNDS", (-5.0, 3.9)),
        (fit_semantics, "DIRECT_RANK_OPTIMIZER_FTOL", 1e-9),
        (fit_semantics, "DIRECT_RANK_OPTIMIZER_GTOL", 1e-5),
        (fit_semantics, "PREPROCESSOR_SCALE_FLOOR", 1e-7),
        (fit_semantics, "PREPROCESSOR_STD_DDOF", 1),
        (fit_semantics, "RANK_TRANSFORM_EPSILON", 1e-5),
        (fit_semantics, "RANK_PERCENTILE_MIDPOINT_OFFSET", 0.4),
        (fit_semantics, "RANK_TRANSFORM_SEMANTICS_VERSION", 2),
        (fit_semantics, "RANK_BIN_LOWER_ENDPOINT", -20.0),
        (fit_semantics, "RANK_BIN_UPPER_ENDPOINT", 20.0),
        (fit_semantics, "RANK_PMF_MASS_FLOOR", 1e-12),
        (fit_semantics, "QUADRATURE_POINTS", 11),
        (fit_semantics, "DETERMINISTIC_QUADRATURE_METHOD", "different method"),
        (fit_semantics, "GENERIC_RANK_PRIOR_MOMENTS_VERSION", 2),
        (fit_semantics, "RANK_PMF_INTEGRATION_SEMANTICS_VERSION", 2),
    ],
)
def test_history_semantic_identity_tracks_material_behavior(
    monkeypatch: pytest.MonkeyPatch, module: object, constant: str, changed: object
) -> None:
    original = history.history11_semantic_specification_sha256()
    monkeypatch.setattr(module, constant, changed)
    assert history.history11_semantic_specification_sha256() != original


def test_history_semantic_identity_excludes_paths_and_retained_attestation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = history.history11_semantic_specification_sha256()
    monkeypatch.setattr(history, "_CANONICAL_HISTORY_ROOT", tmp_path)
    monkeypatch.setattr(history, "_HISTORY_2026_MODEL_METADATA_SHA256", "a" * 64)
    assert history.history11_semantic_specification_sha256() == original


def test_history_annual_fit_enables_abnormal_line_search_retry(monkeypatch) -> None:
    training = [
        TeamSeason(
            2004 + index,
            "fbs",
            str(index),
            str(index),
            20,
            np.asarray([-0.2, 0.1 + index / 20]),
            np.asarray([-0.1, 0.2 + index / 25]),
            np.asarray([5, 8]),
            {
                "lag2_z_mean": index / 10,
                "lag3_z_mean": -index / 20,
                "long_run_z_mean": index / 30,
            },
        )
        for index in range(5)
    ]
    target = history._TargetInput(
        "target",
        "Target",
        20,
        np.asarray([-0.1, 0.2]),
        {
            "lag2_z_mean": 0.1,
            "lag3_z_mean": -0.05,
            "long_run_z_mean": 0.08,
        },
        None,
        None,
    )
    monkeypatch.setattr(
        history,
        "_load_inputs",
        lambda *_args, **_kwargs: (None, training, [], [], [target]),
    )
    original_fit = DirectRankModel.fit
    fit_calls = []

    def track_fit(cls, rows, features, **kwargs):
        fit_calls.append(kwargs.copy())
        return original_fit(rows, features, **kwargs)

    monkeypatch.setattr(DirectRankModel, "fit", classmethod(track_fit))
    history.reproduce_canonical_history_annual(2027, from_snapshot=False)

    assert len(fit_calls) == 1
    assert fit_calls[0]["abnormal_retry_maxls"] == (
        fit_semantics.DIRECT_RANK_OPTIMIZER_ABNORMAL_RETRY_MAXLS
    )


def test_history_constituent_rank_filter_matches_frozen_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = history.history11_semantic_specification()
    assert spec["completed_history_start_season"] == 2002
    assert spec["constituent_rank_filter"] == {
        "semantics_version": 1,
        "numeric_conversion": "finite observed ranks converted to integer by truncation toward zero",
        "retained_bounds": "inclusive integer ranks from 1 through team_population",
        "out_of_population": "discard observation",
        "empty_outcome": "skip team-season row when no usable ranks remain",
    }
    assert history._rank_values({"team_population": "3", "rank_observations": "[0,1,3,4]"}).tolist() == [1, 3]
    assert len(history._rank_values({"team_population": "3", "rank_observations": "[0,4]"})) == 0
    monkeypatch.setattr(history, "HISTORY_1_1_CONSTITUENT_RANK_FILTER_SEMANTICS_VERSION", 2)
    with pytest.raises(ValueError, match="unsupported History constituent-rank filter"):
        history._rank_values({"team_population": "3", "rank_observations": "[1]"})


@pytest.mark.parametrize(("location", "scale"), [(0.3, 0.7), (0.2, 0.8)])
def test_generic_producer_identity_binds_each_fitted_parameter(
    location: float, scale: float
) -> None:
    semantic = history.history11_semantic_specification_sha256()
    baseline = history._generic_prior_producer(0.2, 0.7, semantic)
    changed = history._generic_prior_producer(location, scale, semantic)
    assert changed["producer_parameters_sha256"] != baseline["producer_parameters_sha256"]
    assert changed["producer_identity_sha256"] != baseline["producer_identity_sha256"]


def test_generic_history_moments_and_rank_coordinate_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        TeamSeason(2024, "fbs", "a", "A", 4, np.asarray([0.0]),
                   np.asarray([0.0]), np.asarray([1]), {}),
        TeamSeason(2025, "fbs", "b", "B", 4, np.asarray([0.0]),
                   np.asarray([2.0, 4.0, 6.0]), np.asarray([2, 3, 4]), {}),
    ]
    prior = GenericRankPrior.fit(rows)
    assert prior.location == 2.0
    assert prior.scale == pytest.approx(np.sqrt(16.0 / 3.0))
    np.testing.assert_array_equal(
        prior.pmf(4), normal_pmf(2.0, np.sqrt(16.0 / 3.0), 4)
    )
    assert rank_to_z(np.asarray([1, 2, 3, 4]), 4)[0] == pytest.approx(
        np.log(0.125) - np.log1p(-0.125)
    )
    np.testing.assert_allclose(
        rank_bin_edges(4),
        [-np.inf, np.log(0.25 / 0.75), 0.0, np.log(0.75 / 0.25), np.inf],
    )
    single = conditional_rank_mixture_pmf(np.asarray([2.0]), prior.scale, 4)
    np.testing.assert_array_equal(single, prior.pmf(4))
    monkeypatch.setattr(fit_semantics, "GENERIC_RANK_PRIOR_MOMENTS_VERSION", 2)
    with pytest.raises(ValueError, match="moment semantics"):
        GenericRankPrior.fit(rows)
