"""Frozen Context 1.3 production contract.

Context 1.3 is the active production prior.  This module owns the small
contract that keeps the promoted model separate from retained Context 1.2
publication artifacts:

* the feature list is explicit and ordered;
* transfer values arrive through the attach-only #114 boundary;
* target-season transfer artifacts are validated against an immutable snapshot
  manifest before inference; reconstructed 2026 inputs use a distinct,
  explicit provenance class;
* Context features are location-only while History features remain in both
  equations of :class:`gippyrank.preseason.DirectRankModel`.

Historical research rows may use the retrospective representation documented
by the candidate artifacts.  They are deliberately not treated as archived
preseason production snapshots.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from gippyrank.context_prior import (
    AnnualFittedInstance,
    InferenceRow,
    ModelSpecification,
)
from gippyrank.preseason import (
    QUADRATURE_POINTS,
    DirectRankModel,
    TeamSeason,
    product_quadrature,
)
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
ACTIVE_CONTEXT_PRIOR_VERSION = "1.3"
RETROSPECTIVE_2026_PROVENANCE = "retrospective_2026_transfer_reconstruction"
RETROSPECTIVE_RESEARCH_PROVENANCE = "retrospective_research_transfer_reconstruction"
PRODUCTION_TRANSFER_PROVENANCE = "production_preseason_immutable_transfer_snapshot"
FROZEN_PENALTY = 0.25
PRODUCTION_CUTOFF_MONTH = 8
PRODUCTION_CUTOFF_DAY = 15
# The 2026 artifact is allowed only through the retrospective validator below.
# Future seasons continue to require the production-safe validator.
RETROSPECTIVE_RECONSTRUCTION_SEASONS = frozenset({2026})

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

_CONTEXT_TRANSFER_PROVENANCE_CLASSES = frozenset(
    {
        PRODUCTION_TRANSFER_PROVENANCE,
        RETROSPECTIVE_2026_PROVENANCE,
        RETROSPECTIVE_RESEARCH_PROVENANCE,
    }
)
_LOCATION_RECONSTRUCTION_TOLERANCE = 1e-8
_CONTEXT_TRANSFER_PROVENANCE_TOKEN = object()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: object) -> str:
    """Hash a JSON-compatible model or inference representation."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _team_feature_sha256(values: Mapping[str, float | None]) -> str:
    if set(values) != set(TRANSFER_FEATURE_COLUMNS):
        raise ValueError("transfer feature identity must use the frozen feature columns")
    normalized: dict[str, float | None] = {}
    for name in TRANSFER_FEATURE_COLUMNS:
        value = values[name]
        if value is None:
            normalized[name] = None
            continue
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"transfer feature {name!r} must be finite")
        normalized[name] = numeric
    availability = normalized["transfer_in_prior_defensive_impact_db_available"]
    if availability not in (0.0, 1.0):
        raise ValueError("defensive transfer availability must be binary")
    if (
        availability == 0.0
        and normalized["transfer_in_prior_defensive_impact_db_sum"] not in (None, 0.0)
    ):
        raise ValueError("unavailable defensive transfer impact must be neutral")
    return sha256_json(normalized)


def _team_transfer_feature_hashes(
    values_by_team: Mapping[str, Mapping[str, float | None]],
) -> tuple[tuple[str, str], ...]:
    if not values_by_team:
        raise ValueError("transfer provenance requires a non-empty target FBS population")
    return tuple(
        (team_id, _team_feature_sha256(values_by_team[team_id]))
        for team_id in sorted(values_by_team)
    )


@dataclass(frozen=True, init=False)
class ContextTransferInputProvenance:
    """Content identity for transfer inputs only, not the full Context row.

    Production and retrospective objects are minted by the corresponding
    transfer validation loaders. The explicit ``research_only`` constructor is
    for checked-in development fixtures and can only create research-class
    identities.
    """

    target_season: int
    provenance_class: str
    transfer_feature_artifact_sha256: str
    source_manifest_sha256: str | None
    canonical_snapshot_ids: tuple[str, ...]
    canonical_snapshot_sha256: tuple[str, ...]
    cutoff_state: str
    feature_artifact_id: str
    manifest_artifact_id: str | None
    target_fbs_team_ids: tuple[str, ...]
    target_fbs_population_sha256: str
    target_team_feature_sha256: tuple[tuple[str, str], ...]
    retrieval_timestamps: tuple[str, ...]
    source_endpoints: tuple[str, ...]
    reconstructed_state_declaration: str | None
    archived_august_15_snapshot_absent: bool | None
    research_source_artifact_sha256: str | None
    _diagnostic_paths: tuple[str, str] | None

    def __init__(
        self,
        *,
        target_season: int,
        provenance_class: str,
        transfer_feature_artifact_sha256: str,
        source_manifest_sha256: str | None,
        canonical_snapshot_ids: tuple[str, ...],
        canonical_snapshot_sha256: tuple[str, ...],
        cutoff_state: str,
        feature_artifact_id: str,
        manifest_artifact_id: str | None,
        target_fbs_team_ids: tuple[str, ...],
        target_fbs_population_sha256: str,
        target_team_feature_sha256: tuple[tuple[str, str], ...],
        retrieval_timestamps: tuple[str, ...] = (),
        source_endpoints: tuple[str, ...] = (),
        reconstructed_state_declaration: str | None = None,
        archived_august_15_snapshot_absent: bool | None = None,
        research_source_artifact_sha256: str | None = None,
        diagnostic_paths: tuple[str, str] | None = None,
        _construction_token: object = None,
    ) -> None:
        if _construction_token is not _CONTEXT_TRANSFER_PROVENANCE_TOKEN:
            raise TypeError(
                "validated transfer provenance must come from a transfer loader; "
                "use research_only() for development fixtures"
            )
        for name, value in (
            ("target_season", target_season),
            ("provenance_class", provenance_class),
            ("transfer_feature_artifact_sha256", transfer_feature_artifact_sha256),
            ("source_manifest_sha256", source_manifest_sha256),
            ("canonical_snapshot_ids", tuple(canonical_snapshot_ids)),
            ("canonical_snapshot_sha256", tuple(canonical_snapshot_sha256)),
            ("cutoff_state", cutoff_state),
            ("feature_artifact_id", feature_artifact_id),
            ("manifest_artifact_id", manifest_artifact_id),
            ("target_fbs_team_ids", tuple(target_fbs_team_ids)),
            ("target_fbs_population_sha256", target_fbs_population_sha256),
            ("target_team_feature_sha256", tuple(target_team_feature_sha256)),
            ("retrieval_timestamps", tuple(retrieval_timestamps)),
            ("source_endpoints", tuple(source_endpoints)),
            ("reconstructed_state_declaration", reconstructed_state_declaration),
            ("archived_august_15_snapshot_absent", archived_august_15_snapshot_absent),
            ("research_source_artifact_sha256", research_source_artifact_sha256),
            ("_diagnostic_paths", diagnostic_paths),
        ):
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_season, bool)
            or not isinstance(self.target_season, int)
            or self.target_season < 1
        ):
            raise ValueError("transfer provenance requires a valid target season")
        if self.provenance_class not in _CONTEXT_TRANSFER_PROVENANCE_CLASSES:
            raise ValueError("unsupported Context transfer provenance class")
        if not self.feature_artifact_id or not self.cutoff_state:
            raise ValueError("transfer provenance requires stable artifact and cutoff identities")
        for label, digest in (
            ("derived transfer artifact", self.transfer_feature_artifact_sha256),
            ("target FBS population", self.target_fbs_population_sha256),
            *(
                (("source manifest", self.source_manifest_sha256),)
                if self.source_manifest_sha256 is not None
                else ()
            ),
        ):
            if not _is_sha256(digest):
                raise ValueError(f"{label} identity must be SHA-256")
        if len(self.canonical_snapshot_ids) != len(self.canonical_snapshot_sha256):
            raise ValueError("canonical transfer snapshot IDs and hashes differ in length")
        if (
            not self.target_fbs_team_ids
            or tuple(sorted(set(self.target_fbs_team_ids))) != self.target_fbs_team_ids
            or self.target_fbs_population_sha256 != sha256_json(list(self.target_fbs_team_ids))
        ):
            raise ValueError("transfer provenance target FBS population identity is invalid")
        if not set(dict(self.target_team_feature_sha256)) <= set(self.target_fbs_team_ids):
            raise ValueError("transfer team feature values exceed the target FBS population")
        if any(not _is_sha256(value) for value in self.canonical_snapshot_sha256):
            raise ValueError("canonical transfer snapshot hashes must be SHA-256")
        for team_id, digest in self.target_team_feature_sha256:
            if not team_id or not _is_sha256(digest):
                raise ValueError("target team transfer feature identities are invalid")
        if self.provenance_class == PRODUCTION_TRANSFER_PROVENANCE:
            if (
                self.cutoff_state != "on_time"
                or self.target_season in RETROSPECTIVE_RECONSTRUCTION_SEASONS
                or not self.source_manifest_sha256
                or not self.manifest_artifact_id
                or not self.canonical_snapshot_ids
                or self.reconstructed_state_declaration is not None
            ):
                raise ValueError("production transfer provenance is incomplete or retrospective")
        elif self.provenance_class == RETROSPECTIVE_2026_PROVENANCE:
            if (
                self.target_season != 2026
                or self.cutoff_state != "retrospective_reconstruction"
                or not self.source_manifest_sha256
                or not self.manifest_artifact_id
                or not self.canonical_snapshot_ids
                or self.archived_august_15_snapshot_absent is not True
                or not self.retrieval_timestamps
                or not self.source_endpoints
                or len(self.retrieval_timestamps) != len(self.canonical_snapshot_ids)
                or len(self.source_endpoints) != len(self.canonical_snapshot_ids)
                or not self.reconstructed_state_declaration
            ):
                raise ValueError("2026 transfer reconstruction provenance is incomplete")
        elif (
            self.cutoff_state != "research_only_unvalidated"
            or self.source_manifest_sha256 is not None
            or self.manifest_artifact_id is not None
            or self.canonical_snapshot_ids
            or self.archived_august_15_snapshot_absent is not None
        ):
            raise ValueError("research transfer provenance cannot claim production validation")

    @classmethod
    def research_only(
        cls,
        target_season: int,
        *,
        feature_artifact_id: str,
        transfer_feature_values_by_team: Mapping[str, Mapping[str, float | None]],
        source_artifact_payload: object,
        target_team_ids: Iterable[str] | None = None,
    ) -> ContextTransferInputProvenance:
        """Bind retrospective development data without validating production lineage."""
        team_feature_hashes = _team_transfer_feature_hashes(
            transfer_feature_values_by_team
        )
        keys = sorted(set(target_team_ids) if target_team_ids is not None else {
            team_id for team_id, _ in team_feature_hashes
        })
        if not keys or any(not team_id for team_id in keys):
            raise ValueError("research transfer provenance requires target team IDs")
        if not set(dict(team_feature_hashes)) <= set(keys):
            raise ValueError("research transfer feature rows exceed the declared target population")
        return cls(
            target_season=target_season,
            provenance_class=RETROSPECTIVE_RESEARCH_PROVENANCE,
            transfer_feature_artifact_sha256=sha256_json(source_artifact_payload),
            source_manifest_sha256=None,
            canonical_snapshot_ids=(),
            canonical_snapshot_sha256=(),
            cutoff_state="research_only_unvalidated",
            feature_artifact_id=feature_artifact_id,
            manifest_artifact_id=None,
            target_fbs_team_ids=tuple(keys),
            target_fbs_population_sha256=sha256_json(keys),
            target_team_feature_sha256=team_feature_hashes,
            research_source_artifact_sha256=sha256_json(source_artifact_payload),
            _construction_token=_CONTEXT_TRANSFER_PROVENANCE_TOKEN,
        )

    @property
    def source_identity_sha256(self) -> str:
        return sha256_json(self.identity_payload())

    def identity_payload(self) -> dict[str, object]:
        """Return checkout-independent identity fields; paths are excluded."""
        return {
            "archived_august_15_snapshot_absent": self.archived_august_15_snapshot_absent,
            "canonical_snapshot_ids": list(self.canonical_snapshot_ids),
            "canonical_snapshot_sha256": list(self.canonical_snapshot_sha256),
            "cutoff_state": self.cutoff_state,
            "feature_artifact_id": self.feature_artifact_id,
            "manifest_artifact_id": self.manifest_artifact_id,
            "provenance_class": self.provenance_class,
            "reconstructed_state_declaration": self.reconstructed_state_declaration,
            "research_source_artifact_sha256": self.research_source_artifact_sha256,
            "retrieval_timestamps": list(self.retrieval_timestamps),
            "source_endpoints": list(self.source_endpoints),
            "source_manifest_sha256": self.source_manifest_sha256,
            "target_fbs_population_sha256": self.target_fbs_population_sha256,
            "target_fbs_team_ids": list(self.target_fbs_team_ids),
            "target_season": self.target_season,
            "target_team_feature_sha256": dict(self.target_team_feature_sha256),
            "transfer_feature_artifact_sha256": self.transfer_feature_artifact_sha256,
        }

    def to_metadata(self, *, include_diagnostic_paths: bool = False) -> dict[str, object]:
        metadata = {
            **self.identity_payload(),
            "source_identity_sha256": self.source_identity_sha256,
        }
        if include_diagnostic_paths and self._diagnostic_paths is not None:
            metadata["diagnostic_paths"] = {
                "transfer_feature_artifact": self._diagnostic_paths[0],
                "snapshot_manifest": self._diagnostic_paths[1],
            }
        return metadata

    def validate_inference_row(self, row: InferenceRow) -> None:
        if row.season != self.target_season:
            raise ValueError("transfer provenance and Context inference row seasons differ")
        expected = dict(self.target_team_feature_sha256).get(row.team_id)
        if expected is None:
            raise ValueError("Context inference team is absent from validated transfer population")
        values = {name: row.features.get(name) for name in TRANSFER_FEATURE_COLUMNS}
        if _team_feature_sha256(values) != expected:
            raise ValueError("Context inference transfer values differ from their source artifact")


@dataclass(frozen=True)
class Context13LocationDecomposition:
    """Canonical outcome-free decomposition of one fitted Context 1.3 prior."""

    target_season: int
    trained_through_season: int
    team_id: str
    team_name: str
    population: int
    context_model_sha256: str
    inference_inputs_sha256: str
    transfer_provenance: ContextTransferInputProvenance
    intercept: float
    history_derived_subtotal: float
    context_only_subtotal: float
    conditional_location_points: np.ndarray
    residual_scale: float

    def __post_init__(self) -> None:
        if self.transfer_provenance.target_season != self.target_season:
            raise ValueError("decomposition source provenance has the wrong season")
        if not self.team_id or not self.team_name:
            raise ValueError("decomposition team identity must be non-empty")
        if isinstance(self.population, bool) or self.population < 1:
            raise ValueError("decomposition population must be positive")
        for name, value in (
            ("Context model", self.context_model_sha256),
            ("inference inputs", self.inference_inputs_sha256),
        ):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"decomposition {name} identity must be SHA-256")
        scalars = (
            self.intercept,
            self.history_derived_subtotal,
            self.context_only_subtotal,
            self.residual_scale,
        )
        if not all(np.isfinite(value) for value in scalars) or self.residual_scale <= 0:
            raise ValueError("decomposition values must be finite with positive scale")
        points = np.asarray(self.conditional_location_points, dtype=float)
        if points.ndim != 1 or not len(points) or not np.isfinite(points).all():
            raise ValueError("conditional location points must be a finite vector")
        center = self.intercept + self.history_derived_subtotal + self.context_only_subtotal
        if not np.isclose(
            points.mean(), center, rtol=0, atol=_LOCATION_RECONSTRUCTION_TOLERANCE
        ):
            raise ValueError("Context 1.3 location terms do not reconstruct the center")
        points = points.copy()
        points.setflags(write=False)
        object.__setattr__(self, "conditional_location_points", points)

    @property
    def location_center(self) -> float:
        return float(self.conditional_location_points.mean())


def validate_context13_fitted_model(
    model: DirectRankModel, instance: AnnualFittedInstance
) -> str:
    """Validate the fitted model against the full frozen Context 1.3 contract."""
    if (
        instance.model_family != "context_prior"
        or instance.spec_version != CONTEXT_PRIOR_CANDIDATE_VERSION
        or instance.target_season <= 1
        or instance.trained_through_season != instance.target_season - 1
        or instance.context_effective_cutoff is not None
    ):
        raise ValueError("candidate input must be a rolling-origin Context 1.3 fit")
    if (
        tuple(model.feature_names) != MODEL_FEATURE_NAMES
        or tuple(model.preprocessor.feature_names) != MODEL_FEATURE_NAMES
        or model.location_feature_names != list(LOCATION_FEATURE_NAMES)
        or model.scale_feature_names != list(SCALE_FEATURE_NAMES)
        or model.family != "normal"
        or model.lag_count != 1
        or model.degrees_of_freedom is not None
        or model.penalty != FROZEN_PENALTY
        or model.minimum_scale != 0.10
        or not isinstance(model.optimizer, dict)
        or model.optimizer.get("success") is not True
    ):
        raise ValueError("fitted model does not match the Context 1.3 model specification")
    expected_beta_size = 1 + 2 * len(MODEL_FEATURE_NAMES) + model.lag_count
    if model.beta.shape != (expected_beta_size,) or model.gamma.shape != (
        1 + 2 * len(MODEL_FEATURE_NAMES),
    ):
        raise ValueError("fitted model coefficient shape does not match Context 1.3")
    if not np.isfinite(model.beta).all() or not np.isfinite(model.gamma).all():
        raise ValueError("fitted model coefficients must be finite")
    # Context-only covariates are location-only by the frozen C1.3 contract.
    # The design is [intercept, numeric values..., missingness indicators...].
    context_only = tuple(name for name in MODEL_FEATURE_NAMES if name not in H_FEATURES)
    forbidden_scale_indices = [
        1 + index
        for index, name in enumerate(MODEL_FEATURE_NAMES)
        if name in context_only
    ] + [
        1 + len(MODEL_FEATURE_NAMES) + index
        for index, name in enumerate(MODEL_FEATURE_NAMES)
        if name in context_only
    ]
    if not np.allclose(model.gamma[forbidden_scale_indices], 0.0, rtol=0, atol=1e-14):
        raise ValueError(
            "Context 1.3 Context-only scale coefficients must be zero"
        )
    for values in (
        model.preprocessor.medians,
        model.preprocessor.means,
        model.preprocessor.scales,
    ):
        if set(values) != set(MODEL_FEATURE_NAMES) or not all(
            np.isfinite(value) for value in values.values()
        ):
            raise ValueError("fitted model preprocessing does not match Context 1.3")
    if any(value <= 0 for value in model.preprocessor.scales.values()):
        raise ValueError("Context 1.3 preprocessing scales must be positive")
    return sha256_json(model.metadata())


def _inference_input_sha256(row: InferenceRow) -> str:
    row.require_no_target()
    values: dict[str, float | None] = {}
    for name, value in row.features.items():
        numeric = None if value is None else float(value)
        if numeric is not None and not np.isfinite(numeric):
            raise ValueError(f"inference feature {name!r} must be finite or missing")
        values[name] = numeric
    return sha256_json(
        {
            "season": row.season,
            "subdivision": row.subdivision,
            "team_id": row.team_id,
            "team_name": row.team_name,
            "population": row.population,
            "lag1_z": None if row.lag1_z is None else list(row.lag1_z),
            "lag_zs": [list(values) for values in row.lag_zs],
            "features": values,
        }
    )


def decompose_context13_location(
    model: DirectRankModel,
    instance: AnnualFittedInstance,
    row: InferenceRow,
    transfer_provenance: ContextTransferInputProvenance,
) -> Context13LocationDecomposition:
    """Derive fitted Context 1.3 location terms from a validated input row.

    The result is bound to the actual model metadata, input values, team, target
    season, and the source provenance returned by the inference-input validator.
    """
    model_sha256 = validate_context13_fitted_model(model, instance)
    if (
        row.season != instance.target_season
        or transfer_provenance.target_season != instance.target_season
    ):
        raise ValueError("Context decomposition team input and model seasons differ")
    if transfer_provenance.provenance_class not in _CONTEXT_TRANSFER_PROVENANCE_CLASSES:
        raise ValueError("Context decomposition input provenance is invalid")
    if row.subdivision != "fbs":
        raise ValueError("Context 1.3 candidate decomposition requires an FBS team")
    if not row.team_id or not row.team_name:
        raise ValueError("decomposition team identity must be non-empty")
    if row.population < 1 or row.lag1_z is None or not len(row.lag1_z):
        raise ValueError("fitted Context 1.3 inference requires a lag-1 rank distribution")
    if row.lag_zs:
        raise ValueError("Context 1.3 inference accepts only its frozen lag-1 distribution")
    if set(row.features) != set(MODEL_FEATURE_NAMES):
        raise ValueError("inference row features do not match the Context 1.3 specification")
    transfer_provenance.validate_inference_row(row)

    features = {name: row.features[name] for name in MODEL_FEATURE_NAMES}
    lag1 = np.asarray(row.lag1_z, dtype=float)
    if not np.isfinite(lag1).all():
        raise ValueError("lag-1 inference values must be finite")
    locations, scale = model.conditional_parameters(features, lag1)
    design = model.design_vector(features)
    base_coefficients = model.beta[model.lag_count :]
    intercept = float(base_coefficients[0])
    context_subtotal = 0.0
    history_subtotal = float(
        np.mean(product_quadrature((lag1,), QUADRATURE_POINTS) @ model.beta[:1])
    )
    n_features = len(model.feature_names)
    for index, name in enumerate(model.feature_names):
        value_and_missing = float(
            design[1 + index] * base_coefficients[1 + index]
            + design[1 + n_features + index]
            * base_coefficients[1 + n_features + index]
        )
        if name in H_FEATURES:
            history_subtotal += value_and_missing
        else:
            context_subtotal += value_and_missing
    return Context13LocationDecomposition(
        target_season=row.season,
        trained_through_season=instance.trained_through_season,
        team_id=row.team_id,
        team_name=row.team_name,
        population=row.population,
        context_model_sha256=model_sha256,
        inference_inputs_sha256=_inference_input_sha256(row),
        transfer_provenance=transfer_provenance,
        intercept=intercept,
        history_derived_subtotal=history_subtotal,
        context_only_subtotal=context_subtotal,
        conditional_location_points=np.asarray(locations, dtype=float),
        residual_scale=float(scale),
    )


def moderate_positive_net(value: float, alpha: float) -> float:
    """Keep nonpositive terms and scale only their positive portion."""
    if not np.isfinite(value) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("net contribution must be finite and alpha within [0, 1]")
    return min(value, 0.0) + alpha * max(value, 0.0)


def moderated_location_points(
    decomposition: Context13LocationDecomposition, alpha: float
) -> np.ndarray:
    """Shift a fitted Context location mixture using its context-only subtotal."""
    moderated = moderate_positive_net(decomposition.context_only_subtotal, alpha)
    points = decomposition.conditional_location_points + (
        moderated - decomposition.context_only_subtotal
    )
    center = decomposition.intercept + decomposition.history_derived_subtotal + moderated
    if not np.isclose(
        points.mean(), center, rtol=0, atol=_LOCATION_RECONSTRUCTION_TOLERANCE
    ):
        raise ValueError("moderated Context location does not reconstruct the center")
    return points

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
        "candidate": False,
        "status": "active_production",
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


def _read_csv_bytes(content: bytes, label: str) -> list[dict[str, str]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ManifestValidationError(f"transfer feature artifact is not UTF-8: {label}") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ManifestValidationError(f"transfer feature artifact has invalid columns: {label}")
    return list(reader)


def _manifest_provenance(
    manifest: SnapshotManifest,
    target_season: int,
    *,
    provenance_class: str,
) -> dict[str, object]:
    records = manifest.for_target(target_season)
    return {
        "provenance_class": provenance_class,
        "target_season": target_season,
        "cutoff": f"{target_season}-{manifest.cutoff_month:02d}-{manifest.cutoff_day:02d}",
        "snapshot_ids": [record.snapshot_id for record in records],
        "snapshot_sha256": [record.sha256 for record in records],
        "retrieval_timestamps": [record.retrieval_timestamp for record in records],
        "source_endpoints": [record.endpoint for record in records],
        "all_snapshots_on_or_before_cutoff": all(
            record.captured_on_or_before_cutoff for record in records
        ),
    }


def _load_validated_transfer_features(
    feature_path: Path,
    manifest_path: Path,
    *,
    target_season: int,
    expected_team_keys: Iterable[tuple[int, str, str]],
    provenance: Mapping[str, Any],
    expected_provenance_class: str,
    require_on_time_cutoff: bool,
) -> tuple[list[dict[str, str]], ContextTransferInputProvenance]:
    """Load a target artifact only after validating its immutable input chain."""
    try:
        feature_bytes = feature_path.read_bytes()
        manifest_sha_before = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    except FileNotFoundError as error:
        raise ManifestValidationError(f"transfer source artifact is missing: {error.filename}") from error
    manifest = load_snapshot_manifest(
        manifest_path, required_seasons=[target_season], verify_hashes=True
    )
    try:
        manifest_sha_after = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    except FileNotFoundError as error:
        raise ManifestValidationError("snapshot manifest disappeared during validation") from error
    if manifest_sha_before != manifest_sha_after:
        raise ManifestValidationError("snapshot manifest changed during validation")
    manifest_sha256 = manifest_sha_after
    feature_sha256 = hashlib.sha256(feature_bytes).hexdigest()
    claimed_feature_sha256 = provenance.get(
        "transfer_feature_artifact_sha256",
        provenance.get("feature_artifact_sha256"),
    )
    if claimed_feature_sha256 is not None and claimed_feature_sha256 != feature_sha256:
        raise ManifestValidationError("transfer feature artifact hash is mismatched")
    claimed_manifest_sha256 = provenance.get("source_manifest_sha256")
    if claimed_manifest_sha256 is not None and claimed_manifest_sha256 != manifest_sha256:
        raise ManifestValidationError("transfer provenance source manifest hash is mismatched")
    records = tuple(
        sorted(
            manifest.for_target(target_season),
            key=lambda record: (record.source, record.snapshot_id),
        )
    )
    if not records:
        raise ManifestValidationError(
            f"target season {target_season} has no canonical transfer snapshots"
        )
    if target_season < 1 or manifest.cutoff_month != 8 or manifest.cutoff_day != 15:
        raise ManifestValidationError("Context transfer snapshots require an August 15 cutoff")
    if any(record.target_season != target_season for record in records):
        raise ManifestValidationError("snapshot manifest contains the wrong target season")
    if len({record.snapshot_id for record in records}) != len(records):
        raise ManifestValidationError("canonical transfer snapshot IDs must be unique")
    if require_on_time_cutoff and any(
        not record.captured_on_or_before_cutoff for record in records
    ):
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
    expected_sources = {"portal", "usage", "stats", "roster", "games_players"}
    if set(by_source) != expected_sources:
        raise ManifestValidationError(
            "canonical transfer snapshot set contains unexpected source classes"
        )

    expected_items = list(expected_team_keys)
    if any(
        season != target_season or subdivision != "fbs" or not team_id
        for season, subdivision, team_id in expected_items
    ):
        raise ManifestValidationError("expected target population must contain only target FBS teams")
    if len(set(expected_items)) != len(expected_items):
        raise ManifestValidationError("expected target FBS population contains duplicate teams")
    expected = set(expected_items)
    if not expected:
        raise ManifestValidationError("canonical target FBS population is empty")
    rows = _read_csv_bytes(feature_bytes, str(feature_path))
    if not rows:
        raise ManifestValidationError("transfer feature artifact contains no data rows")
    header = set(rows[0])
    required_columns = {"season", "subdivision", "team_id", "team_name", *TRANSFER_FEATURE_COLUMNS}
    if not required_columns <= header:
        raise ManifestValidationError(
            f"transfer feature artifact is missing columns: {sorted(required_columns - header)}"
        )
    try:
        target_rows = [row for row in rows if int(row.get("season", "-1")) == target_season]
    except (TypeError, ValueError) as error:
        raise ManifestValidationError("transfer feature artifact has an invalid season value") from error
    actual: dict[tuple[int, str, str], dict[str, str]] = {}
    for row in target_rows:
        try:
            key = _key(row)
        except (KeyError, TypeError, ValueError) as error:
            raise ManifestValidationError("transfer feature artifact has an invalid team key") from error
        if key in actual:
            raise ManifestValidationError(f"duplicate transfer feature row: {key}")
        if not row.get("team_name", "").strip():
            raise ManifestValidationError(f"transfer feature row has no team name: {key}")
        if row.get("provenance_class") and row["provenance_class"] != expected_provenance_class:
            raise ManifestValidationError(
                "derived transfer feature row provenance class does not match its loader"
            )
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
    if provenance.get("provenance_class") != expected_provenance_class:
        raise ManifestValidationError(
            f"target transfer provenance must be {expected_provenance_class}"
        )
    if provenance.get("target_season") != target_season:
        raise ManifestValidationError("transfer provenance target season is inconsistent")
    expected_cutoff = f"{target_season}-{manifest.cutoff_month:02d}-{manifest.cutoff_day:02d}"
    if provenance.get("cutoff") != expected_cutoff:
        raise ManifestValidationError("transfer provenance cutoff is inconsistent")
    expected_snapshot_ids = [record.snapshot_id for record in records]
    expected_snapshot_hashes = [record.sha256 for record in records]
    supplied_ids = provenance.get("snapshot_ids")
    supplied_hashes = provenance.get("snapshot_sha256")
    if (
        not isinstance(supplied_ids, list)
        or not isinstance(supplied_hashes, list)
        or len(supplied_ids) != len(supplied_hashes)
        or set(zip(supplied_ids, supplied_hashes))
        != set(zip(expected_snapshot_ids, expected_snapshot_hashes))
    ):
        raise ManifestValidationError(
            "transfer provenance does not identify the canonical snapshot IDs and hashes"
        )
    records_by_id = {record.snapshot_id: record for record in records}
    on_time = all(record.captured_on_or_before_cutoff for record in records)
    if provenance.get("all_snapshots_on_or_before_cutoff") is not on_time:
        raise ManifestValidationError(
            "transfer provenance cutoff state disagrees with the validated snapshots"
        )
    retrieval_timestamps = [record.retrieval_timestamp for record in records]
    source_endpoints = [record.endpoint for record in records]
    reconstructed_declaration: str | None = None
    archived_snapshot_absent: bool | None = None
    if not require_on_time_cutoff:
        if target_season not in RETROSPECTIVE_RECONSTRUCTION_SEASONS:
            raise ManifestValidationError(
                "retrospective transfer provenance is only supported for 2026"
            )
        required = {
            "derivation_timestamp",
            "retrieval_timestamps",
            "source_endpoints",
            "raw_source_hashes",
            "known_absence_of_archived_august_15_transfer_snapshot",
            "provenance_statement",
        }
        missing = sorted(field for field in required if field not in provenance)
        if missing:
            raise ManifestValidationError(
                f"retrospective transfer provenance is missing {missing}"
            )
        if provenance["known_absence_of_archived_august_15_transfer_snapshot"] is not True:
            raise ManifestValidationError(
                "retrospective transfer provenance must acknowledge the absent archived cutoff snapshot"
            )
        if provenance["retrieval_timestamps"] != [
            records_by_id[snapshot_id].retrieval_timestamp for snapshot_id in supplied_ids
        ]:
            raise ManifestValidationError(
                "retrospective transfer retrieval timestamps do not match the manifest"
            )
        if provenance["source_endpoints"] != [
            records_by_id[snapshot_id].endpoint for snapshot_id in supplied_ids
        ]:
            raise ManifestValidationError(
                "retrospective transfer source endpoints do not match the manifest"
            )
        raw_hashes = provenance["raw_source_hashes"]
        if (
            not isinstance(raw_hashes, list)
            or len(raw_hashes) != len(expected_snapshot_hashes)
            or set(raw_hashes) != set(expected_snapshot_hashes)
        ):
            raise ManifestValidationError(
                "retrospective transfer raw source hashes do not match the manifest"
            )
        reconstructed_declaration = provenance["provenance_statement"]
        if not isinstance(reconstructed_declaration, str) or not reconstructed_declaration.strip():
            raise ManifestValidationError("retrospective reconstruction declaration is empty")
        archived_snapshot_absent = True
    transfer_values_by_team: dict[str, dict[str, float | None]] = {}
    for (_season, _subdivision, team_id), row in actual.items():
        values: dict[str, float | None] = {}
        for name in TRANSFER_FEATURE_COLUMNS:
            try:
                values[name] = None if not row[name].strip() else float(row[name])
            except (TypeError, ValueError) as error:
                raise ManifestValidationError(
                    f"target transfer feature {name!r} is not numeric for team {team_id}"
                ) from error
        try:
            _team_feature_sha256(values)
        except ValueError as error:
            raise ManifestValidationError(str(error)) from error
        transfer_values_by_team[team_id] = values
    team_hashes = _team_transfer_feature_hashes(transfer_values_by_team)
    population_sha256 = sha256_json([team_id for team_id, _ in team_hashes])
    provenance_object = ContextTransferInputProvenance(
        target_season=target_season,
        provenance_class=expected_provenance_class,
        transfer_feature_artifact_sha256=feature_sha256,
        source_manifest_sha256=manifest_sha256,
        canonical_snapshot_ids=tuple(expected_snapshot_ids),
        canonical_snapshot_sha256=tuple(expected_snapshot_hashes),
        cutoff_state="on_time" if on_time else "retrospective_reconstruction",
        feature_artifact_id=f"gippyrank.context1_3.transfer_features.season_{target_season}",
        manifest_artifact_id=f"gippyrank.preseason_transfer.snapshot_manifest.season_{target_season}",
        target_fbs_team_ids=tuple(team_id for team_id, _ in team_hashes),
        target_fbs_population_sha256=population_sha256,
        target_team_feature_sha256=team_hashes,
        retrieval_timestamps=tuple(retrieval_timestamps),
        source_endpoints=tuple(source_endpoints),
        reconstructed_state_declaration=reconstructed_declaration,
        archived_august_15_snapshot_absent=archived_snapshot_absent,
        diagnostic_paths=(str(feature_path), str(manifest_path)),
        _construction_token=_CONTEXT_TRANSFER_PROVENANCE_TOKEN,
    )
    return [actual[key] for key in sorted(actual)], provenance_object


def load_validated_production_transfer_features(
    feature_path: Path,
    manifest_path: Path,
    *,
    target_season: int,
    expected_team_keys: Iterable[tuple[int, str, str]],
    provenance: Mapping[str, Any],
) -> tuple[list[dict[str, str]], ContextTransferInputProvenance]:
    """Validate the normal on-time immutable production transfer chain."""
    return _load_validated_transfer_features(
        feature_path,
        manifest_path,
        target_season=target_season,
        expected_team_keys=expected_team_keys,
        provenance=provenance,
        expected_provenance_class=PRODUCTION_TRANSFER_PROVENANCE,
        require_on_time_cutoff=True,
    )


def load_validated_reconstructed_transfer_features(
    feature_path: Path,
    manifest_path: Path,
    *,
    target_season: int,
    expected_team_keys: Iterable[tuple[int, str, str]],
    provenance: Mapping[str, Any],
) -> tuple[list[dict[str, str]], ContextTransferInputProvenance]:
    """Validate the explicit retrospective 2026 reconstruction chain.

    This path intentionally accepts late retrievals only for 2026 and requires
    metadata that prevents the artifact from masquerading as an August 15
    production snapshot.
    """
    return _load_validated_transfer_features(
        feature_path,
        manifest_path,
        target_season=target_season,
        expected_team_keys=expected_team_keys,
        provenance=provenance,
        expected_provenance_class=RETROSPECTIVE_2026_PROVENANCE,
        require_on_time_cutoff=False,
    )


def candidate_guard(target_season: int) -> None:
    """Retain the old call boundary while allowing the activated 2026 path."""
    if target_season < 2003:
        raise ValueError("Context 1.3 target seasons must be >= 2003")


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
    "PRODUCTION_TRANSFER_PROVENANCE",
    "REJECTED_TRANSFER_FEATURES",
    "REMOVED_CONTEXT_1_2_FEATURES",
    "RETROSPECTIVE_2026_PROVENANCE",
    "RETROSPECTIVE_RESEARCH_PROVENANCE",
    "RETURNING_FEATURES",
    "SCALE_FEATURE_NAMES",
    "Context13LocationDecomposition",
    "ContextTransferInputProvenance",
    "attach_transfer_features",
    "attach_transfer_features_to_inference_rows",
    "candidate_guard",
    "decompose_context13_location",
    "fit_model",
    "load_validated_production_transfer_features",
    "load_validated_reconstructed_transfer_features",
    "model_specification",
    "model_specification_metadata",
    "moderate_positive_net",
    "moderated_location_points",
    "sha256_json",
    "validate_context13_fitted_model",
    "validate_feature_contract",
]
