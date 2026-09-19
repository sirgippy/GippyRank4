"""Frozen Context 1.3 candidate contract.

Context 1.3 is an implemented-but-not-active production candidate.  This
module owns the small contract that separates the candidate from the active
Context 1.2 publication path:

* the feature list is explicit and ordered;
* transfer values arrive through the attach-only #114 boundary;
* target-season transfer artifacts are validated against an immutable,
  on-time snapshot manifest before inference;
* Context features are location-only while History features remain in both
  equations of :class:`gippyrank.preseason.DirectRankModel`.

Historical research rows may use the retrospective representation documented
by the candidate artifacts.  They are deliberately not treated as archived
preseason production snapshots.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from gippyrank.context_prior import (
    AnnualFittedInstance,
    InferenceRow,
    ModelSpecification,
)
from gippyrank.preseason import DirectRankModel, TeamSeason
from gippyrank.preseason_transfer import (
    MODEL_FEATURE_COLUMNS as TRANSFER_FEATURE_COLUMNS,
)
from gippyrank.preseason_transfer import (
    ManifestValidationError,
    SnapshotManifest,
    load_snapshot_manifest,
    merge_preseason_transfer_features,
)

CONTEXT_PRIOR_CANDIDATE_VERSION = "1.3"
ACTIVE_CONTEXT_PRIOR_VERSION = "1.2"
FROZEN_PENALTY = 0.25
PRODUCTION_CUTOFF_MONTH = 8
PRODUCTION_CUTOFF_DAY = 15
INELIGIBLE_CANDIDATE_SEASONS = frozenset({2026})

H_FEATURES = (
    "lag2_z_mean",
    "lag3_z_mean",
    "long_run_z_mean",
)
COACHING_FEATURES = ("coach_tenure_seasons",)
RECRUITING_FEATURES = (
    "recruiting_class_rank",
    "recruiting_class_points",
    "recruiting_points_2y_mean",
    "recruiting_points_3y_mean",
    "recruiting_points_4y_mean",
    "recruiting_points_trend",
)
TALENT_FEATURES = ("talent_composite",)
RETURNING_FEATURES = ("returning_pct_ppa",)
INCOMING_OFFENSE_FEATURES = ("transfer_in_prior_usage_sum",)
INCOMING_DB_FEATURES = (
    "transfer_in_prior_defensive_impact_db_sum",
    "transfer_in_prior_defensive_impact_db_available",
)

CONTEXT_1_3_FEATURES = (
    *COACHING_FEATURES,
    *RECRUITING_FEATURES,
    *TALENT_FEATURES,
    *RETURNING_FEATURES,
    *INCOMING_OFFENSE_FEATURES,
    *INCOMING_DB_FEATURES,
)
MODEL_FEATURE_NAMES = (*H_FEATURES, *CONTEXT_1_3_FEATURES)
LOCATION_FEATURE_NAMES = MODEL_FEATURE_NAMES
SCALE_FEATURE_NAMES = H_FEATURES

D5_CONTEXT_FEATURES = (
    *COACHING_FEATURES,
    *RECRUITING_FEATURES,
    *TALENT_FEATURES,
    *RETURNING_FEATURES,
    *INCOMING_OFFENSE_FEATURES,
)

REMOVED_CONTEXT_1_2_FEATURES = (
    "returning_pct_passing_ppa",
    "returning_pct_receiving_ppa",
    "returning_pct_rushing_ppa",
)
REJECTED_TRANSFER_FEATURES = frozenset(
    {
        "transfer_out_prior_usage_sum",
        "transfer_out_prior_usage_mean",
        "transfer_out_prior_usage_max",
        "transfer_out_prior_usage_qb",
        "transfer_net_prior_usage",
        "transfer_in_count",
        "transfer_in_count_qb",
        "transfer_out_count",
        "transfer_out_count_qb",
        "transfer_net_count",
        "transfer_in_prior_usage_mean",
        "transfer_in_prior_usage_max",
        "transfer_in_prior_usage_qb",
        "transfer_in_prior_usage_sum_available",
        "transfer_in_rating_sum",
        "transfer_in_rating_mean",
        "transfer_in_rating_max",
        "transfer_in_weighted_rating_sum",
        "transfer_out_rating_sum",
        "transfer_out_rating_mean",
        "transfer_out_rating_max",
        "transfer_out_weighted_rating_sum",
        "transfer_net_rating_sum",
        "transfer_net_weighted_rating_sum",
        "transfer_in_stars_4_count",
        "transfer_in_stars_5_count",
        "transfer_out_stars_4_count",
        "transfer_out_stars_5_count",
        "transfer_in_prior_defensive_experience_sum",
        "transfer_in_prior_defensive_experience_available",
        "transfer_in_prior_defensive_experience_db_sum",
        "transfer_in_prior_defensive_impact_sum",
        "transfer_in_prior_defensive_impact_available",
        "transfer_in_prior_defensive_impact_dl_edge_sum",
        "transfer_in_prior_defensive_impact_dl_edge_available",
        "transfer_in_prior_defensive_impact_lb_sum",
        "transfer_in_prior_defensive_impact_lb_available",
    }
)


def validate_feature_contract(
    feature_names: Sequence[str], *, exact: bool = True
) -> tuple[str, ...]:
    """Validate the frozen 1.3 model feature contract.

    ``exact=False`` is useful for validating a predeclared subset such as D5;
    production Context 1.3 fits use the default exact check.
    """
    names = tuple(feature_names)
    if len(names) != len(set(names)):
        raise ValueError("Context 1.3 feature names must be unique")
    unknown = set(names) - set(MODEL_FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unapproved Context 1.3 features requested: {sorted(unknown)}")
    if exact and names != MODEL_FEATURE_NAMES:
        raise ValueError(
            "Context 1.3 feature order does not match the frozen contract: "
            f"expected {list(MODEL_FEATURE_NAMES)}, got {list(names)}"
        )
    if set(names) & set(REMOVED_CONTEXT_1_2_FEATURES):
        raise ValueError("Context 1.3 cannot fit decomposed returning-production features")
    if set(names) & REJECTED_TRANSFER_FEATURES:
        raise ValueError("Context 1.3 cannot fit rejected transfer research features")
    return names


def model_specification() -> ModelSpecification:
    """Return the durable candidate specification, independent of a fit."""
    validate_feature_contract(MODEL_FEATURE_NAMES)
    return ModelSpecification(
        "context_prior",
        CONTEXT_PRIOR_CANDIDATE_VERSION,
        MODEL_FEATURE_NAMES,
        "normal",
        FROZEN_PENALTY,
        "full t-1 empirical quadrature; t-2/t-3 summaries",
    )


def model_specification_metadata() -> dict[str, object]:
    """Return an artifact-ready specification with equation placement."""
    return {
        **model_specification().metadata(),
        "candidate": True,
        "active_production_version": ACTIVE_CONTEXT_PRIOR_VERSION,
        "location_feature_names": list(LOCATION_FEATURE_NAMES),
        "scale_feature_names": list(SCALE_FEATURE_NAMES),
        "context_features_affect": "location_only",
        "history_features_affect": ["location", "scale"],
        "optimizer_retry": "retry with maxiter=2000 only after iteration-limit failure",
        "target_season_outcomes_forbidden": True,
    }


def fit_model(
    rows: list[TeamSeason],
    *,
    target_season: int,
    trained_through_season: int,
    context_features: Sequence[str] = CONTEXT_1_3_FEATURES,
) -> tuple[DirectRankModel, AnnualFittedInstance]:
    """Fit a frozen candidate or D5 control with the production optimizer."""
    if trained_through_season >= target_season:
        raise ValueError("training must end before target")
    context = validate_feature_contract(context_features, exact=False)
    if context != CONTEXT_1_3_FEATURES and context != D5_CONTEXT_FEATURES:
        raise ValueError("only the frozen P3 candidate and D5 control are supported")
    features = [*H_FEATURES, *context]
    location = [*H_FEATURES, *context]
    scale = list(SCALE_FEATURE_NAMES)
    training = [row for row in rows if row.season <= trained_through_season]
    if not training:
        raise ValueError("cannot fit Context 1.3 without training rows")

    def fit(options: dict[str, float | int] | None = None) -> DirectRankModel:
        return DirectRankModel.fit(
            training,
            features,
            penalty=FROZEN_PENALTY,
            location_feature_names=location,
            scale_feature_names=scale,
            optimizer_options=options,
        )

    try:
        model = fit()
    except RuntimeError as error:
        if "ITERATIONS REACHED LIMIT" not in str(error):
            raise
        model = fit({"maxiter": 2000})
    return model, AnnualFittedInstance(
        "context_prior",
        CONTEXT_PRIOR_CANDIDATE_VERSION
        if context == CONTEXT_1_3_FEATURES
        else "D5",
        trained_through_season,
        target_season,
        None,
    )


def _key(row: Mapping[str, Any] | TeamSeason | InferenceRow) -> tuple[int, str, str]:
    if isinstance(row, Mapping):
        return (int(row["season"]), str(row["subdivision"]), str(row["team_id"]))
    return (int(row.season), str(row.subdivision), str(row.team_id))


def _feature_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _checked_transfer_rows(
    feature_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only the three model-facing fields from the #114 boundary."""
    result: list[dict[str, Any]] = []
    for source in feature_rows:
        missing = [name for name in TRANSFER_FEATURE_COLUMNS if name not in source]
        if missing:
            raise ValueError(f"transfer feature row is missing columns: {missing}")
        result.append(
            {
                "season": int(source["season"]),
                "subdivision": str(source["subdivision"]),
                "team_id": str(source["team_id"]),
                **{
                    name: _feature_value(source[name])
                    for name in TRANSFER_FEATURE_COLUMNS
                },
            }
        )
    return result


def _attach_feature_values(
    rows: Iterable[TeamSeason | InferenceRow],
    feature_rows: Iterable[Mapping[str, Any]],
    *,
    require_all: bool,
) -> list[TeamSeason | InferenceRow]:
    source_rows = list(rows)
    transfers = _checked_transfer_rows(feature_rows)
    # Use the shared attach-only boundary for duplicate, missing-column, and
    # exact team-season checks.  The returned mappings intentionally contain no
    # audit columns that could leak into the model feature dictionary.
    base_rows = [
        {
            "season": row.season,
            "subdivision": row.subdivision,
            "team_id": row.team_id,
        }
        for row in source_rows
    ]
    attached = merge_preseason_transfer_features(
        base_rows, transfers, require_all=require_all
    )
    by_key = {_key(row): row for row in attached}
    result: list[TeamSeason | InferenceRow] = []
    for row in source_rows:
        leaked = {
            name
            for name in row.features
            if name.startswith("transfer_") and name not in TRANSFER_FEATURE_COLUMNS
        }
        if leaked:
            raise ValueError(
                "transfer attachment received unapproved feature columns: "
                f"{sorted(leaked)}"
            )
        values = by_key[_key(row)]
        updated = {
            **row.features,
            **{
                name: values.get(name)
                for name in TRANSFER_FEATURE_COLUMNS
            },
        }
        result.append(replace(row, features=updated))
    return result


def attach_transfer_features(
    rows: Iterable[TeamSeason],
    feature_rows: Iterable[Mapping[str, Any]],
    *,
    require_all: bool = True,
) -> list[TeamSeason]:
    """Attach only the canonical transfer fields to historical TeamSeason rows."""
    return [
        row
        for row in _attach_feature_values(
            rows, feature_rows, require_all=require_all
        )
        if isinstance(row, TeamSeason)
    ]


def attach_transfer_features_to_inference_rows(
    rows: Iterable[InferenceRow],
    feature_rows: Iterable[Mapping[str, Any]],
) -> list[InferenceRow]:
    """Attach target-season transfer fields without adding target outcomes."""
    result = [
        row
        for row in _attach_feature_values(rows, feature_rows, require_all=True)
        if isinstance(row, InferenceRow)
    ]
    for row in result:
        row.require_no_target()
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError as error:
        raise ManifestValidationError(f"transfer feature artifact is missing: {path}") from error


def _manifest_provenance(
    manifest: SnapshotManifest, target_season: int
) -> dict[str, object]:
    records = manifest.for_target(target_season)
    return {
        "provenance_class": "production_preseason_immutable_snapshot",
        "target_season": target_season,
        "cutoff": f"{target_season}-{manifest.cutoff_month:02d}-{manifest.cutoff_day:02d}",
        "snapshot_ids": [record.snapshot_id for record in records],
        "snapshot_sha256": [record.sha256 for record in records],
        "all_snapshots_on_or_before_cutoff": all(
            record.captured_on_or_before_cutoff for record in records
        ),
    }


def load_validated_production_transfer_features(
    feature_path: Path,
    manifest_path: Path,
    *,
    target_season: int,
    expected_team_keys: Iterable[tuple[int, str, str]],
    provenance: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Load a target artifact only after validating its immutable input chain."""
    if target_season in INELIGIBLE_CANDIDATE_SEASONS:
        raise ManifestValidationError(
            "Context 1.3 candidate is ineligible for the existing 2026 "
            "publication lineage; keep Context 1.2 for 2026"
        )
    manifest = load_snapshot_manifest(
        manifest_path, required_seasons=[target_season], verify_hashes=True
    )
    records = manifest.for_target(target_season)
    if not records or any(not record.captured_on_or_before_cutoff for record in records):
        raise ManifestValidationError(
            f"target season {target_season} lacks an on-time production snapshot"
        )
    by_source: dict[str, int] = {}
    for record in records:
        by_source[record.source] = by_source.get(record.source, 0) + 1
    for source in ("portal", "usage", "stats"):
        if by_source.get(source) != 1:
            raise ManifestValidationError(
                f"target season {target_season} needs exactly one canonical {source} snapshot"
            )
    for source in ("roster", "games_players"):
        if by_source.get(source, 0) < 1:
            raise ManifestValidationError(
                f"target season {target_season} has no canonical {source} snapshot"
            )

    expected = set(expected_team_keys)
    if not expected:
        raise ManifestValidationError("canonical target FBS population is empty")
    rows = _read_csv(feature_path)
    target_rows = [row for row in rows if int(row.get("season", "-1")) == target_season]
    actual: dict[tuple[int, str, str], dict[str, str]] = {}
    for row in target_rows:
        key = _key(row)
        if key in actual:
            raise ManifestValidationError(f"duplicate transfer feature row: {key}")
        missing = [name for name in TRANSFER_FEATURE_COLUMNS if name not in row]
        if missing:
            raise ManifestValidationError(
                f"transfer feature row is missing model columns: {missing}"
            )
        actual[key] = row
    if set(actual) != expected:
        raise ManifestValidationError(
            "target transfer feature population differs from canonical FBS table: "
            f"missing={sorted(expected - set(actual))} "
            f"extra={sorted(set(actual) - expected)}"
        )
    if provenance.get("provenance_class") != "production_preseason_immutable_snapshot":
        raise ManifestValidationError("target transfer provenance is not production-safe")
    if int(provenance.get("target_season", -1)) != target_season:
        raise ManifestValidationError("transfer provenance target season is inconsistent")
    expected_cutoff = f"{target_season}-{manifest.cutoff_month:02d}-{manifest.cutoff_day:02d}"
    if provenance.get("cutoff") != expected_cutoff:
        raise ManifestValidationError("transfer provenance cutoff is inconsistent")
    expected_snapshot_ids = [record.snapshot_id for record in records]
    expected_snapshot_hashes = [record.sha256 for record in records]
    if provenance.get("snapshot_ids") != expected_snapshot_ids:
        raise ManifestValidationError(
            "transfer provenance does not identify the canonical snapshot set"
        )
    if provenance.get("snapshot_sha256") != expected_snapshot_hashes:
        raise ManifestValidationError(
            "transfer provenance snapshot hashes do not match the manifest"
        )
    if provenance.get("all_snapshots_on_or_before_cutoff") is not True:
        raise ManifestValidationError(
            "transfer provenance does not certify the cutoff boundary"
        )
    metadata = {
        **_manifest_provenance(manifest, target_season),
        "feature_artifact": str(feature_path),
        "manifest": str(manifest_path),
    }
    return [actual[key] for key in sorted(actual)], metadata


def candidate_guard(target_season: int) -> None:
    """Reject accidental reinterpretation of the currently published 2026 prior."""
    if target_season in INELIGIBLE_CANDIDATE_SEASONS:
        raise ValueError(
            "Context 1.3 is a future-activation candidate; 2026 remains the "
            "published Context 1.2 artifact"
        )


__all__ = [
    "ACTIVE_CONTEXT_PRIOR_VERSION",
    "CONTEXT_1_3_FEATURES",
    "CONTEXT_PRIOR_CANDIDATE_VERSION",
    "D5_CONTEXT_FEATURES",
    "FROZEN_PENALTY",
    "H_FEATURES",
    "INCOMING_DB_FEATURES",
    "INCOMING_OFFENSE_FEATURES",
    "LOCATION_FEATURE_NAMES",
    "MODEL_FEATURE_NAMES",
    "REJECTED_TRANSFER_FEATURES",
    "REMOVED_CONTEXT_1_2_FEATURES",
    "RETURNING_FEATURES",
    "SCALE_FEATURE_NAMES",
    "attach_transfer_features",
    "attach_transfer_features_to_inference_rows",
    "candidate_guard",
    "fit_model",
    "load_validated_production_transfer_features",
    "model_specification",
    "model_specification_metadata",
    "validate_feature_contract",
]
