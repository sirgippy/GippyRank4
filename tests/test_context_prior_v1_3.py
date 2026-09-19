from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import gippyrank.context_prior_v1_3 as candidate_module
from gippyrank.context_prior import InferenceRow
from gippyrank.context_prior_v1_3 import (
    CONTEXT_PRIOR_CANDIDATE_VERSION,
    D5_CONTEXT_FEATURES,
    LOCATION_FEATURE_NAMES,
    MODEL_FEATURE_NAMES,
    RETROSPECTIVE_2026_PROVENANCE,
    SCALE_FEATURE_NAMES,
    attach_transfer_features,
    attach_transfer_features_to_inference_rows,
    candidate_guard,
    load_validated_reconstructed_transfer_features,
    model_specification,
    model_specification_metadata,
    validate_feature_contract,
)
from gippyrank.preseason import TeamSeason
from gippyrank.preseason_transfer import (
    ManifestValidationError,
    SnapshotManifest,
    SnapshotRecord,
)

ROOT = Path(__file__).parents[1]
CANDIDATE = ROOT / "data/processed/preseason/context_v1_3_candidate"


def _team() -> TeamSeason:
    return TeamSeason(
        2022,
        "fbs",
        "alpha",
        "Alpha",
        2,
        np.asarray([0.0]),
        np.asarray([0.1]),
        np.asarray([1]),
        {name: 0.0 for name in MODEL_FEATURE_NAMES},
    )


def _transfer_row() -> dict[str, object]:
    return {
        "season": 2022,
        "subdivision": "fbs",
        "team_id": "alpha",
        "team_name": "Alpha",
        "transfer_in_prior_usage_sum": 1.5,
        "transfer_in_prior_defensive_impact_db_sum": -0.25,
        "transfer_in_prior_defensive_impact_db_available": 1.0,
        "audit_unresolved_count": 4,
    }


def test_frozen_contract_has_exact_features_and_equation_placement() -> None:
    assert validate_feature_contract(MODEL_FEATURE_NAMES) == MODEL_FEATURE_NAMES
    assert validate_feature_contract(D5_CONTEXT_FEATURES, exact=False) == D5_CONTEXT_FEATURES
    assert LOCATION_FEATURE_NAMES == MODEL_FEATURE_NAMES
    assert SCALE_FEATURE_NAMES == (
        "lag2_z_mean",
        "lag3_z_mean",
        "long_run_z_mean",
    )
    assert "returning_pct_passing_ppa" not in MODEL_FEATURE_NAMES
    with pytest.raises(ValueError):
        validate_feature_contract((*MODEL_FEATURE_NAMES, "transfer_out_count"))

    spec = model_specification()
    metadata = model_specification_metadata()
    assert spec.spec_version == CONTEXT_PRIOR_CANDIDATE_VERSION == "1.3"
    assert metadata["context_features_affect"] == "location_only"
    assert metadata["history_features_affect"] == ["location", "scale"]
    assert metadata["active_production_version"] == "1.3"


def test_transfer_attachment_is_attach_only_and_preserves_history() -> None:
    row = _team()
    attached = attach_transfer_features([row], [_transfer_row()])
    assert attached[0].features["lag2_z_mean"] == 0.0
    assert attached[0].features["transfer_in_prior_usage_sum"] == 1.5
    assert attached[0].features[
        "transfer_in_prior_defensive_impact_db_sum"
    ] == -0.25
    assert "audit_unresolved_count" not in attached[0].features

    with pytest.raises(ValueError, match="duplicate transfer feature row"):
        attach_transfer_features([row], [_transfer_row(), _transfer_row()])
    with pytest.raises(ValueError, match="no transfer features"):
        attach_transfer_features([row], [], require_all=True)
    leaked = TeamSeason(
        **{
            **row.__dict__,
            "features": {**row.features, "transfer_in_rating_sum": 1.0},
        }
    )
    with pytest.raises(ValueError, match="unapproved feature columns"):
        attach_transfer_features([leaked], [_transfer_row()])


def test_inference_attachment_keeps_target_outcomes_out() -> None:
    row = InferenceRow(
        2022,
        "fbs",
        "alpha",
        "Alpha",
        2,
        (0.0,),
        (),
        {name: 0.0 for name in MODEL_FEATURE_NAMES},
    )
    attached = attach_transfer_features_to_inference_rows([row], [_transfer_row()])
    assert attached[0].features["transfer_in_prior_usage_sum"] == 1.5
    attached[0].require_no_target()


def test_activated_guard_allows_2026_without_relabeling_its_provenance() -> None:
    candidate_guard(2025)
    candidate_guard(2026)
    with pytest.raises(ManifestValidationError):
        from gippyrank.context_prior_v1_3 import (
            load_validated_production_transfer_features,
        )

        load_validated_production_transfer_features(
            Path("missing.csv"),
            Path("missing-manifest.json"),
            target_season=2026,
            expected_team_keys=[(2026, "fbs", "alpha")],
            provenance={},
        )


def test_production_transfer_validation_requires_manifest_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = ("portal", "usage", "stats", "roster", "games_players")
    records = tuple(
        SnapshotRecord(
            snapshot_id=f"2027:{source}",
            target_season=2027,
            source=source,
            source_season=2027 if source == "portal" else 2026,
            path=f"{source}.json",
            source_filename=f"{source}.json",
            endpoint=f"/{source}",
            query_parameters={},
            retrieval_timestamp="2027-08-01T00:00:00+00:00",
            target_cutoff="2027-08-15",
            captured_on_or_before_cutoff=True,
            sha256=f"{len(source):064x}",
            record_count=1,
            canonical=True,
        )
        for source in sources
    )
    manifest = SnapshotManifest(tmp_path / "manifest.json", tmp_path, records)
    monkeypatch.setattr(candidate_module, "load_snapshot_manifest", lambda *args, **kwargs: manifest)
    feature_path = tmp_path / "features.csv"
    feature_path.write_text(
        "season,subdivision,team_id,transfer_in_prior_usage_sum,"
        "transfer_in_prior_defensive_impact_db_sum,"
        "transfer_in_prior_defensive_impact_db_available\n"
        "2027,fbs,alpha,1.0,0.0,1.0\n",
        encoding="utf-8",
    )
    provenance = {
        "provenance_class": "production_preseason_immutable_snapshot",
        "target_season": 2027,
        "cutoff": "2027-08-15",
        "snapshot_ids": [record.snapshot_id for record in records],
        "snapshot_sha256": [record.sha256 for record in records],
        "all_snapshots_on_or_before_cutoff": True,
    }
    loaded, metadata = candidate_module.load_validated_production_transfer_features(
        feature_path,
        manifest.path,
        target_season=2027,
        expected_team_keys=[(2027, "fbs", "alpha")],
        provenance=provenance,
    )
    assert loaded[0]["team_id"] == "alpha"
    assert metadata["snapshot_ids"] == provenance["snapshot_ids"]

    with pytest.raises(ManifestValidationError, match="canonical snapshot set"):
        candidate_module.load_validated_production_transfer_features(
            feature_path,
            manifest.path,
            target_season=2027,
            expected_team_keys=[(2027, "fbs", "alpha")],
            provenance={**provenance, "snapshot_ids": []},
        )


def test_2026_reconstruction_accepts_late_inputs_only_with_explicit_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = ("portal", "usage", "stats", "roster", "games_players")
    records = tuple(
        SnapshotRecord(
            snapshot_id=f"2026:{source}",
            target_season=2026,
            source=source,
            source_season=2026 if source == "portal" else 2025,
            path=f"{source}.json",
            source_filename=f"{source}.json",
            endpoint=f"/{source}",
            query_parameters={},
            retrieval_timestamp="2026-09-01T00:00:00+00:00",
            target_cutoff="2026-08-15",
            captured_on_or_before_cutoff=False,
            sha256=f"{len(source) + 100:064x}",
            record_count=1,
            canonical=True,
        )
        for source in sources
    )
    manifest = SnapshotManifest(tmp_path / "manifest.json", tmp_path, records)
    monkeypatch.setattr(candidate_module, "load_snapshot_manifest", lambda *args, **kwargs: manifest)
    feature_path = tmp_path / "features.csv"
    feature_path.write_text(
        "season,subdivision,team_id,transfer_in_prior_usage_sum,"
        "transfer_in_prior_defensive_impact_db_sum,"
        "transfer_in_prior_defensive_impact_db_available\n"
        "2026,fbs,alpha,1.0,0.0,1.0\n",
        encoding="utf-8",
    )
    provenance = {
        "provenance_class": RETROSPECTIVE_2026_PROVENANCE,
        "target_season": 2026,
        "cutoff": "2026-08-15",
        "snapshot_ids": [record.snapshot_id for record in records],
        "snapshot_sha256": [record.sha256 for record in records],
        "raw_source_hashes": [record.sha256 for record in records],
        "retrieval_timestamps": [record.retrieval_timestamp for record in records],
        "source_endpoints": [record.endpoint for record in records],
        "derivation_timestamp": "2026-09-01T00:00:00+00:00",
        "known_absence_of_archived_august_15_transfer_snapshot": True,
        "provenance_statement": "retrospective reconstruction",
    }
    loaded, metadata = load_validated_reconstructed_transfer_features(
        feature_path,
        manifest.path,
        target_season=2026,
        expected_team_keys=[(2026, "fbs", "alpha")],
        provenance=provenance,
    )
    assert loaded[0]["team_id"] == "alpha"
    assert metadata["provenance_class"] == RETROSPECTIVE_2026_PROVENANCE


def test_candidate_artifacts_are_separate_and_parity_validated() -> None:
    expected_files = {
        "candidate_report.md",
        "coefficients.csv",
        "context_coverage.csv",
        "evaluation.json",
        "feature_provenance.json",
        "fitted_model.json",
        "historical_transfer_features.csv",
        "historical_transfer_features.provenance.json",
        "model_report.json",
        "model_spec.json",
        "parity_report.json",
        "predictions.csv",
        "rolling_metrics.csv",
    }
    assert expected_files <= {path.name for path in CANDIDATE.iterdir()}
    spec = json.loads((CANDIDATE / "model_spec.json").read_text())
    assert spec["spec_version"] == "1.3"
    assert spec["active_production_version"] == "1.3"
    assert spec["features"] == list(MODEL_FEATURE_NAMES)
    report = json.loads((CANDIDATE / "model_report.json").read_text())
    assert all(value["passed"] for value in report["parity"].values())
    assert report["production_activation"]["ready_for_first_future_season_with_valid_on_time_snapshot"]
    assert "retrospective reconstruction" in report["production_activation"]["2026_guardrail"]

    with (CANDIDATE / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 534
    assert {row["spec_version"] for row in rows} == {"1.3"}
    assert {row["candidate_status"] for row in rows} == {"active_production"}
    assert all(
        str(value) != "" for row in rows for value in json.loads(row["pmf"])
    )

    provenance = json.loads(
        (CANDIDATE / "historical_transfer_features.provenance.json").read_text()
    )
    assert provenance["provenance_class"] == "retrospective_research_reconstruction"
    assert "/home/" not in json.dumps(report["source_artifacts"])
    assert hashlib.sha256(
        (ROOT / "data/processed/preseason/context/annual/2026/predictions.csv").read_bytes()
    ).hexdigest() == "641182890ec88ea8bc6150cc97047ddc688d0c680486d9fcc63a58d7bfae9132"
