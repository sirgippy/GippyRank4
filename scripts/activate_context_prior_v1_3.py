"""Materialize the active Context 1.3 2026 reconstruction.

This is an offline activation step.  Acquisition is intentionally separate:
``fetch_preseason_transfer_snapshots.py`` stores immutable raw responses and
the manifest, while this command derives the three frozen model fields and
fits the 1.3 annual starting distribution.  The 2026 result is explicitly
retrospective and never overwrites the retained Context 1.2 artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import build_preseason_context_prior_v1_2 as c12
import build_preseason_context_prior_v1_3 as c13
import build_preseason_prior as v1
import numpy as np

from gippyrank.context_prior_v1_3 import (
    CONTEXT_1_3_FEATURES,
    CONTEXT_PRIOR_CANDIDATE_VERSION,
    MODEL_FEATURE_NAMES,
    RETROSPECTIVE_2026_PROVENANCE,
    attach_transfer_features_to_inference_rows,
    load_validated_reconstructed_transfer_features,
    model_specification_metadata,
)
from gippyrank.preseason_transfer import (
    CANONICAL_FEATURE_COLUMNS,
    derive_preseason_transfer_features,
    read_player_aliases,
    read_team_aliases,
)

ROOT = Path(__file__).resolve().parents[1]
PRESEASON = ROOT / "data/processed/preseason"
TEAM_FEATURES = PRESEASON / "team_season_features.csv"
DEFAULT_MANIFEST = ROOT / "data/raw/cfbd/preseason/transfers/manifest.json"
DEFAULT_OUTPUT = PRESEASON / "context_v1_3_2026_reconstruction"
ACTIVE_PRIOR = PRESEASON / "context_v1_3"
HISTORICAL_TRANSFER_FEATURES = (
    PRESEASON / "context_v1_3_candidate/historical_transfer_features.csv"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def build_transfer_artifact(
    *, manifest_path: Path, output: Path
) -> dict[str, Any]:
    """Derive 2026 features with the frozen #114 semantics and late provenance."""
    team_rows = _read_csv(TEAM_FEATURES)
    team_aliases_path = ROOT / "data/reference/preseason_team_aliases.csv"
    player_aliases_path = ROOT / "data/reference/preseason_player_aliases.csv"
    result = derive_preseason_transfer_features(
        manifest_path,
        team_rows,
        team_aliases=read_team_aliases(team_aliases_path),
        player_aliases=read_player_aliases(player_aliases_path),
        required_seasons=[2026],
        allow_late_snapshots=True,
    )
    records = result["manifest"].for_target(2026)
    if not records:
        raise ValueError("2026 transfer manifest has no canonical snapshots")
    provenance: dict[str, Any] = {
        "provenance_class": RETROSPECTIVE_2026_PROVENANCE,
        "target_season": 2026,
        "cutoff": "2026-08-15",
        "snapshot_ids": [record.snapshot_id for record in records],
        "snapshot_sha256": [record.sha256 for record in records],
        "raw_source_hashes": [record.sha256 for record in records],
        "retrieval_timestamps": [record.retrieval_timestamp for record in records],
        "source_endpoints": [record.endpoint for record in records],
        "all_snapshots_on_or_before_cutoff": False,
        "known_absence_of_archived_august_15_transfer_snapshot": True,
        "derivation_timestamp": datetime.now(UTC).isoformat(),
        "source_manifest": _repo_relative(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "provenance_statement": (
            "These features reproduce the Context 1.3 feature definitions using "
            "retrospective source data retrieved after the 2026 preseason cutoff. "
            "They are valid for the current model lineage but are not evidence of "
            "the exact information state on August 15, 2026."
        ),
    }
    feature_fields = [*CANONICAL_FEATURE_COLUMNS, "provenance_class"]
    feature_rows = [
        {**{field: row.get(field) for field in CANONICAL_FEATURE_COLUMNS},
         "provenance_class": RETROSPECTIVE_2026_PROVENANCE}
        for row in result["features"]
    ]
    quality_report = dict(result["quality_report"])
    quality_report["manifest"] = _repo_relative(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "transfer_features.csv", feature_rows, feature_fields)
    _write_csv(output / "transfer_team_audit.csv", result["audit"])
    _write_csv(output / "transfer_player_audit.csv", result["player_audit"])
    _write_csv(output / "identity_mapping.csv", result["identity_mapping"])
    _write_json(output / "data_quality_report.json", quality_report)
    _write_json(output / "feature_provenance.json", provenance)
    _write_json(
        output / "source_manifest.json",
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    _write_json(
        output / "raw_source_inventory.json",
        {
            "raw_payloads_unchanged": True,
            "target_season": 2026,
            "snapshots": [record.as_dict() for record in records],
            "retrieval_timestamps": provenance["retrieval_timestamps"],
            "sha256": provenance["raw_source_hashes"],
        },
    )
    return {"features": feature_rows, "provenance": provenance, "quality": quality_report}


def _prediction_from_inference(model: Any, row: Any) -> v1.PriorPrediction:
    row.require_no_target()
    if row.lag1_z is None:
        raise ValueError(
            "Context 1.3 inference row lacks History fallback: "
            f"{(row.season, row.subdivision, row.team_id)}"
        )
    lag1 = np.asarray(row.lag1_z, dtype=float)
    lag_zs = tuple(np.asarray(values, dtype=float) for values in row.lag_zs)
    pmf = model.pmf(row.features, lag1, row.population, lag_zs)
    locations, scale = model.conditional_parameters(row.features, lag1, lag_zs)
    return v1.PriorPrediction(
        row.season,
        row.subdivision,
        row.team_id,
        row.team_name,
        row.population,
        np.asarray([], dtype=int),
        "context_prior_v1_3",
        "same_subdivision_lag1",
        pmf,
        float(np.mean(locations)),
        scale,
    )


def build_starting_prior(
    *,
    transfer_output: Path,
    manifest_path: Path,
    output: Path,
    transfer_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Fit through 2025 and produce an independent reconstructed 2026 prior."""
    rows, _cold, _coverage = c13.load_candidate_rows(HISTORICAL_TRANSFER_FEATURES)
    model, fitted = c13.fit_model(
        rows,
        target_season=2026,
        trained_through_season=2025,
        context_features=CONTEXT_1_3_FEATURES,
    )
    team_rows = _read_csv(TEAM_FEATURES)
    index = {
        (int(row["season"]), row["subdivision"], row["team_id"]): row
        for row in team_rows
    }
    base = c12.inference_rows(2026, 2025, index, c12.cached_tenures())
    base = [
        row.__class__(
            row.season,
            row.subdivision,
            row.team_id,
            row.team_name,
            row.population,
            row.lag1_z,
            row.lag_zs,
            {name: row.features.get(name) for name in MODEL_FEATURE_NAMES},
            row.cold_start_reason,
        )
        for row in base
    ]
    feature_path = transfer_output / "transfer_features.csv"
    feature_rows, validated = load_validated_reconstructed_transfer_features(
        feature_path,
        manifest_path,
        target_season=2026,
        expected_team_keys={(2026, "fbs", row.team_id) for row in base},
        provenance=transfer_provenance,
    )
    validated = {
        **validated,
        "feature_artifact": _repo_relative(feature_path),
        "manifest": _repo_relative(manifest_path),
    }
    fallback = {
        item.key: item
        for item in c13.parse_prediction_artifact(
            PRESEASON / "history/annual/2026/predictions.csv"
        )
    }
    # Preserve the established History cold-start fallback without borrowing
    # any Context 1.2 posterior state.
    attached = attach_transfer_features_to_inference_rows(base, feature_rows)
    predictions = []
    for row in attached:
        key = (row.season, row.subdivision, row.team_id)
        if row.lag1_z is None:
            if key not in fallback:
                raise ValueError(f"Context 1.3 inference row lacks History fallback: {key}")
            predictions.append(fallback[key])
        else:
            predictions.append(_prediction_from_inference(model, row))
    prediction_rows = [
        {
            **item.csv_row(),
            "model_family": "context_prior",
            "spec_version": CONTEXT_PRIOR_CANDIDATE_VERSION,
            "artifact_kind": "reconstructed_preseason_forecast",
            "starting_point_status": "retrospective_2026_reconstruction",
            "transfer_provenance_class": RETROSPECTIVE_2026_PROVENANCE,
        }
        for item in predictions
    ]
    annual = ACTIVE_PRIOR / "annual/2026"
    _write_csv(annual / "predictions.csv", prediction_rows)
    _write_json(annual / "fitted_instance.json", fitted.metadata())
    _write_json(
        annual / "fitted_model.json",
        {
            "model_family": "context_prior",
            "spec_version": CONTEXT_PRIOR_CANDIDATE_VERSION,
            "model": model.metadata(),
            "prior_artifact_kind": "reconstructed_preseason_forecast",
            "transfer_provenance_class": RETROSPECTIVE_2026_PROVENANCE,
        },
    )
    _write_json(annual / "model_spec.json", model_specification_metadata())
    _write_json(
        annual / "feature_provenance.json",
        {"context_1_3": c13.feature_provenance(), "transfer": transfer_provenance},
    )
    _write_json(
        annual / "model_report.json",
        {
            "model_family": "context_prior",
            "spec_version": CONTEXT_PRIOR_CANDIDATE_VERSION,
            "status": "active_production",
            "target_season": 2026,
            "trained_through_season": 2025,
            "starting_point_status": "retrospective_2026_reconstruction",
            "target_outcomes_used": False,
            "n_fbs": len(predictions),
            "transfer_provenance": transfer_provenance,
            "validated_transfer_metadata": validated,
            "prior_artifact_sha256": _sha256(annual / "predictions.csv"),
        },
    )
    return {
        "prediction_count": len(predictions),
        "prior_path": _repo_relative(annual / "predictions.csv"),
        "prior_sha256": _sha256(annual / "predictions.csv"),
    }


def run(*, manifest_path: Path = DEFAULT_MANIFEST, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    transfer = build_transfer_artifact(manifest_path=manifest_path, output=output)
    prior = build_starting_prior(
        transfer_output=output,
        manifest_path=manifest_path,
        output=output,
        transfer_provenance=transfer["provenance"],
    )
    comparison = {
        "manifest_version": "context-1.2-vs-1.3-2026-v1",
        "status": "awaiting_official_week_4",
        "target_season": 2026,
        "artifacts": {
            "context_1_2_preseason": "data/processed/snapshots/2026/2026-preseason-context/predictive/context",
            "context_1_3_reconstructed_preseason": "data/processed/snapshots/2026/2026-preseason-context-v1.3/predictive/context",
            "context_1_2_official_week_4": None,
            "context_1_3_official_week_4": None,
        },
        "week_4_requirements": {
            "same_team_population": True,
            "same_included_game_ids": True,
            "same_effective_cutoff": True,
            "same_historical_likelihood_version": "V1",
            "distinct_prior_model_versions": ["1.2", "1.3"],
        },
        "transfer_reconstruction": transfer["provenance"],
    }
    _write_json(output / "comparison_quartet.json", comparison)
    _write_json(output / "activation_report.json", {"transfer": transfer["quality"], "prior": prior})
    return {"transfer": transfer["provenance"], "prior": prior, "comparison": comparison}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(manifest_path=args.manifest, output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
