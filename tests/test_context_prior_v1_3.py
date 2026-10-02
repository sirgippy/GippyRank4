from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
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
    PRODUCTION_TRANSFER_PROVENANCE,
    RETROSPECTIVE_2026_PROVENANCE,
    SCALE_FEATURE_NAMES,
    attach_transfer_features,
    attach_transfer_features_to_inference_rows,
    candidate_guard,
    load_validated_production_transfer_features,
    load_validated_reconstructed_transfer_features,
    model_specification,
    model_specification_metadata,
    validate_feature_contract,
)
from gippyrank.preseason import TeamSeason
from gippyrank.preseason_transfer import (
    ManifestValidationError,
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


def _write_transfer_validation_inputs(
    root: Path, *, season: int, late: bool = False, usage: float = 1.0
) -> tuple[Path, Path, dict[str, object], list[tuple[int, str, str]]]:
    raw_root = root / "raw"
    raw_root.mkdir(parents=True)
    sources = ("portal", "usage", "stats", "roster", "games_players")
    retrieval = f"{season}-09-01T00:00:00+00:00" if late else f"{season}-08-01T00:00:00+00:00"
    records = []
    for source in sources:
        relative = Path("snapshots") / f"{source}.json"
        raw_path = raw_root / relative
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_content = f'[{{"source":"{source}"}}]'.encode()
        raw_path.write_bytes(raw_content)
        records.append(
            SnapshotRecord(
                snapshot_id=f"{season}:{source}",
                target_season=season,
                source=source,
                source_season=season if source in {"portal", "usage", "stats"} else season - 1,
                path=relative.as_posix(),
                source_filename=relative.name,
                endpoint=f"/{source}",
                query_parameters={},
                retrieval_timestamp=retrieval,
                target_cutoff=f"{season}-08-15",
                captured_on_or_before_cutoff=not late,
                sha256=hashlib.sha256(raw_content).hexdigest(),
                record_count=1,
                canonical=True,
            ).as_dict()
        )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "raw_root": "raw",
                "cutoff": {"month": 8, "day": 15},
                "snapshots": records,
            }
        ),
        encoding="utf-8",
    )
    feature_path = root / "features.csv"
    feature_path.write_text(
        "season,subdivision,team_id,team_name,transfer_in_prior_usage_sum,"
        "transfer_in_prior_defensive_impact_db_sum,"
        "transfer_in_prior_defensive_impact_db_available\n"
        f"{season},fbs,alpha,Alpha,{usage},0.0,1.0\n",
        encoding="utf-8",
    )
    snapshot_ids = [str(record["snapshot_id"]) for record in records]
    snapshot_hashes = [str(record["sha256"]) for record in records]
    provenance: dict[str, object] = {
        "provenance_class": (
            RETROSPECTIVE_2026_PROVENANCE if late else PRODUCTION_TRANSFER_PROVENANCE
        ),
        "target_season": season,
        "cutoff": f"{season}-08-15",
        "snapshot_ids": snapshot_ids,
        "snapshot_sha256": snapshot_hashes,
        "all_snapshots_on_or_before_cutoff": not late,
    }
    if late:
        provenance.update(
            {
                "raw_source_hashes": snapshot_hashes,
                "retrieval_timestamps": [retrieval] * len(records),
                "source_endpoints": [str(record["endpoint"]) for record in records],
                "derivation_timestamp": f"{season}-09-02T00:00:00+00:00",
                "known_absence_of_archived_august_15_transfer_snapshot": True,
                "provenance_statement": "retrospective reconstructed state from late source retrievals",
            }
        )
    provenance["source_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    return feature_path, manifest_path, provenance, [(season, "fbs", "alpha")]


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


def test_production_transfer_provenance_is_authoritative_and_content_addressed(
    tmp_path: Path,
) -> None:
    feature_path, manifest_path, provenance, expected = _write_transfer_validation_inputs(
        tmp_path / "checkout-a", season=2027
    )
    loaded, identity = candidate_module.load_validated_production_transfer_features(
        feature_path,
        manifest_path,
        target_season=2027,
        expected_team_keys=expected,
        provenance=provenance,
    )
    assert loaded[0]["team_id"] == "alpha"
    assert identity.provenance_class == PRODUCTION_TRANSFER_PROVENANCE
    assert identity.cutoff_state == "on_time"
    assert identity.transfer_feature_artifact_sha256 == hashlib.sha256(
        feature_path.read_bytes()
    ).hexdigest()
    assert identity.source_manifest_sha256 == provenance["source_manifest_sha256"]
    assert identity.canonical_snapshot_ids
    assert identity.to_metadata()["feature_artifact_id"] == (
        "gippyrank.context1_3.transfer_features.season_2027"
    )

    with pytest.raises(ManifestValidationError, match="canonical snapshot IDs and hashes"):
        candidate_module.load_validated_production_transfer_features(
            feature_path,
            manifest_path,
            target_season=2027,
            expected_team_keys=expected,
            provenance={**provenance, "snapshot_ids": []},
        )
    with pytest.raises(ManifestValidationError, match="source manifest hash is mismatched"):
        candidate_module.load_validated_production_transfer_features(
            feature_path,
            manifest_path,
            target_season=2027,
            expected_team_keys=expected,
            provenance={**provenance, "source_manifest_sha256": "f" * 64},
        )
    with pytest.raises(ManifestValidationError, match="feature artifact hash is mismatched"):
        candidate_module.load_validated_production_transfer_features(
            feature_path,
            manifest_path,
            target_season=2027,
            expected_team_keys=expected,
            provenance={**provenance, "transfer_feature_artifact_sha256": "e" * 64},
        )
    with pytest.raises(ManifestValidationError, match="cutoff state"):
        candidate_module.load_validated_production_transfer_features(
            feature_path,
            manifest_path,
            target_season=2027,
            expected_team_keys=expected,
            provenance={**provenance, "all_snapshots_on_or_before_cutoff": False},
        )
    with pytest.raises(ManifestValidationError, match="population differs"):
        candidate_module.load_validated_production_transfer_features(
            feature_path,
            manifest_path,
            target_season=2027,
            expected_team_keys=[(2027, "fbs", "different-team")],
            provenance=provenance,
        )

    malformed_root = tmp_path / "malformed-canonical-flag"
    malformed_feature, malformed_manifest, malformed_provenance, malformed_expected = (
        _write_transfer_validation_inputs(malformed_root, season=2027)
    )
    manifest_payload = json.loads(malformed_manifest.read_text(encoding="utf-8"))
    manifest_payload["snapshots"][0]["canonical"] = "false"
    malformed_manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    malformed_provenance["source_manifest_sha256"] = hashlib.sha256(
        malformed_manifest.read_bytes()
    ).hexdigest()
    with pytest.raises(ManifestValidationError, match="canonical flag is not boolean"):
        candidate_module.load_validated_production_transfer_features(
            malformed_feature,
            malformed_manifest,
            target_season=2027,
            expected_team_keys=malformed_expected,
            provenance=malformed_provenance,
        )

    second_root = tmp_path / "checkout-b"
    second_feature, second_manifest, second_provenance, second_expected = (
        _write_transfer_validation_inputs(second_root, season=2027)
    )
    _, second_identity = candidate_module.load_validated_production_transfer_features(
        second_feature,
        second_manifest,
        target_season=2027,
        expected_team_keys=second_expected,
        provenance=second_provenance,
    )
    assert second_identity.source_identity_sha256 == identity.source_identity_sha256
    assert second_feature.resolve() != feature_path.resolve()

    second_feature.write_text(
        second_feature.read_text(encoding="utf-8").replace(",Alpha,1.0,", ",Alpha,2.0,"),
        encoding="utf-8",
    )
    _, changed_identity = candidate_module.load_validated_production_transfer_features(
        second_feature,
        second_manifest,
        target_season=2027,
        expected_team_keys=second_expected,
        provenance=second_provenance,
    )
    assert changed_identity.transfer_feature_artifact_sha256 != identity.transfer_feature_artifact_sha256
    assert changed_identity.source_identity_sha256 != identity.source_identity_sha256
    bad_raw_source = second_root / "raw/snapshots/portal.json"
    bad_raw_source.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="snapshot hash mismatch"):
        candidate_module.load_validated_production_transfer_features(
            second_feature,
            second_manifest,
            target_season=2027,
            expected_team_keys=second_expected,
            provenance=second_provenance,
        )


def test_2026_reconstruction_requires_retrospective_class_and_complete_attestation(
    tmp_path: Path,
) -> None:
    feature_path, manifest_path, provenance, expected = _write_transfer_validation_inputs(
        tmp_path / "retrospective-2026", season=2026, late=True
    )
    loaded, identity = load_validated_reconstructed_transfer_features(
        feature_path,
        manifest_path,
        target_season=2026,
        expected_team_keys=expected,
        provenance=provenance,
    )
    assert loaded[0]["team_id"] == "alpha"
    assert identity.provenance_class == RETROSPECTIVE_2026_PROVENANCE
    assert identity.cutoff_state == "retrospective_reconstruction"
    assert identity.archived_august_15_snapshot_absent is True
    assert identity.retrieval_timestamps
    assert identity.source_endpoints
    assert identity.reconstructed_state_declaration
    assert identity.to_metadata()["provenance_class"] == (
        "retrospective_2026_reconstruction"
    )
    with pytest.raises(TypeError):
        replace(identity, provenance_class=PRODUCTION_TRANSFER_PROVENANCE)
    with pytest.raises(ManifestValidationError, match="must acknowledge the absent"):
        load_validated_reconstructed_transfer_features(
            feature_path,
            manifest_path,
            target_season=2026,
            expected_team_keys=expected,
            provenance={
                **provenance,
                "known_absence_of_archived_august_15_transfer_snapshot": False,
            },
        )
    with pytest.raises(ManifestValidationError, match="must be"):
        load_validated_reconstructed_transfer_features(
            feature_path,
            manifest_path,
            target_season=2026,
            expected_team_keys=expected,
            provenance={**provenance, "provenance_class": PRODUCTION_TRANSFER_PROVENANCE},
        )
    with pytest.raises(ManifestValidationError, match="on-time production snapshot"):
        load_validated_production_transfer_features(
            feature_path,
            manifest_path,
            target_season=2026,
            expected_team_keys=expected,
            provenance={**provenance, "provenance_class": PRODUCTION_TRANSFER_PROVENANCE},
        )


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
