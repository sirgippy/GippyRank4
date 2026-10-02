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
    Preprocessor,
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
RETROSPECTIVE_2026_PROVENANCE = "retrospective_2026_reconstruction"
RETROSPECTIVE_RESEARCH_PROVENANCE = "retrospective_research_transfer_reconstruction"
PRODUCTION_TRANSFER_PROVENANCE = "production_preseason_immutable_snapshot"
FROZEN_PENALTY = 0.25
PRODUCTION_CUTOFF_MONTH = 8
PRODUCTION_CUTOFF_DAY = 15
# The 2026 artifact is allowed only through the retrospective validator below.
# Future seasons continue to require the production-safe validator.
RETROSPECTIVE_RECONSTRUCTION_SEASONS = frozenset({2026})
_CONTEXT13_2026_MODEL_METADATA_SHA256 = (
    "6a6c759c250a9b4845bce563072dcf146fd66dca8264017f215aff1580bc9dce"
)
_CONTEXT13_2026_FITTED_SOURCE_ARTIFACT_SHA256 = (
    "2756bee8545e0ebf143438dc15fed3a2208f1eb4379c60d4b559be3ec3e58db1"
)
_COMMITTED_2026_TRANSFER_FEATURE_SHA256 = (
    "db8fad38bb81e437059b8cf44c85be5760f3426466b53c5cf649ea5c0718fef0"
)
_COMMITTED_2026_TRANSFER_MANIFEST_SHA256 = (
    "574e705ec9e3c51658ee83cb1bbb1c1792a326ab1a89333219e9b27e7a8b00c9"
)
_COMMITTED_2026_TRANSFER_PROVENANCE_SHA256 = (
    "2c4725d18ad9dbadbc02c6d08dcb754492409173d924f606ffd7c1e6c4c1ca16"
)

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


_CONTEXT13_FITTED_SOURCE_TOKEN = object()


@dataclass(frozen=True, init=False)
class Context13FittedModelSource:
    """Typed identity for a Context 1.3 fit and its exact training inputs.

    Canonical objects are created by :func:`fit_model_with_source` or by the
    validating annual-artifact loader below. Development fixtures must use the
    explicitly research-only constructor. Candidate validation prevents a
    research-only fit source from being paired with production transfer
    provenance or being represented as a canonical fit.
    """

    model_family: str
    spec_version: str
    target_season: int
    trained_through_season: int
    model_metadata_sha256: str
    fitted_instance_identity_sha256: str
    frozen_model_spec_identity_sha256: str
    training_corpus_input_sha256: str
    training_row_count: int
    provenance_class: str
    source_identity_sha256: str

    def __init__(
        self,
        *,
        model_family: str,
        spec_version: str,
        target_season: int,
        trained_through_season: int,
        model_metadata_sha256: str,
        fitted_instance_identity_sha256: str,
        frozen_model_spec_identity_sha256: str,
        training_corpus_input_sha256: str,
        training_row_count: int,
        provenance_class: str,
        _construction_token: object = None,
    ) -> None:
        if _construction_token is not _CONTEXT13_FITTED_SOURCE_TOKEN:
            raise TypeError("Context 1.3 fit sources must come from the canonical fit or loader")
        values = {
            "model_family": model_family,
            "spec_version": spec_version,
            "target_season": target_season,
            "trained_through_season": trained_through_season,
            "model_metadata_sha256": model_metadata_sha256,
            "fitted_instance_identity_sha256": fitted_instance_identity_sha256,
            "frozen_model_spec_identity_sha256": frozen_model_spec_identity_sha256,
            "training_corpus_input_sha256": training_corpus_input_sha256,
            "training_row_count": training_row_count,
            "provenance_class": provenance_class,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source_identity_sha256", sha256_json(values))
        self.__post_init__()

    def __post_init__(self) -> None:
        if (
            self.model_family != "context_prior"
            or self.spec_version != CONTEXT_PRIOR_CANDIDATE_VERSION
            or self.target_season <= 1
            or self.trained_through_season != self.target_season - 1
            or self.training_row_count < 0
            or self.provenance_class not in {"canonical_context13_fit", "research_only"}
        ):
            raise ValueError("Context 1.3 fitted-model source identity is invalid")
        for digest in (
            self.model_metadata_sha256,
            self.fitted_instance_identity_sha256,
            self.frozen_model_spec_identity_sha256,
            self.training_corpus_input_sha256,
            self.source_identity_sha256,
        ):
            if not _is_sha256(digest):
                raise ValueError("Context 1.3 fitted-model source hashes must be SHA-256")
        if sha256_json(self.identity_payload()) != self.source_identity_sha256:
            raise ValueError("Context 1.3 fitted-model source identity hash is inconsistent")

    def identity_payload(self) -> dict[str, object]:
        return {
            "model_family": self.model_family,
            "spec_version": self.spec_version,
            "target_season": self.target_season,
            "trained_through_season": self.trained_through_season,
            "model_metadata_sha256": self.model_metadata_sha256,
            "fitted_instance_identity_sha256": self.fitted_instance_identity_sha256,
            "frozen_model_spec_identity_sha256": self.frozen_model_spec_identity_sha256,
            "training_corpus_input_sha256": self.training_corpus_input_sha256,
            "training_row_count": self.training_row_count,
            "provenance_class": self.provenance_class,
        }

    def to_metadata(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "source_identity_sha256": self.source_identity_sha256,
        }

    def validate_instance(self, instance: AnnualFittedInstance) -> None:
        if (
            instance.model_family != self.model_family
            or instance.spec_version != self.spec_version
            or instance.target_season != self.target_season
            or instance.trained_through_season != self.trained_through_season
            or instance.context_effective_cutoff is not None
            or sha256_json(instance.metadata()) != self.fitted_instance_identity_sha256
        ):
            raise ValueError("fitted-model source does not match the annual instance")

    def validate_model(
        self,
        model: DirectRankModel,
        instance: AnnualFittedInstance,
        *,
        allow_research_only: bool = False,
        training_rows: Sequence[TeamSeason] | None = None,
    ) -> None:
        if sha256_json(self.identity_payload()) != self.source_identity_sha256:
            raise ValueError("fitted-model source identity hash is inconsistent")
        validate_context13_fitted_model(model, instance)
        self.validate_instance(instance)
        if sha256_json(model.metadata()) != self.model_metadata_sha256:
            raise ValueError("fitted-model source metadata hash does not match the model")
        if self.frozen_model_spec_identity_sha256 != sha256_json(
            model_specification_metadata()
        ):
            raise ValueError("fitted-model source uses a different frozen specification")
        if self.provenance_class == "research_only" and not allow_research_only:
            raise ValueError("research-only Context 1.3 fit provenance is not promotable")
        if training_rows is not None and (
            self.training_row_count != len(training_rows)
            or self.training_corpus_input_sha256
            != _training_corpus_input_sha256(training_rows)
        ):
            raise ValueError("fitted-model source training-corpus identity does not match its rows")

    @classmethod
    def research_only(
        cls,
        model: DirectRankModel,
        instance: AnnualFittedInstance,
        *,
        fixture_id: str,
    ) -> Context13FittedModelSource:
        """Bind a retained development fixture without claiming a canonical fit."""
        if not fixture_id.strip():
            raise ValueError("research-only fit provenance needs a fixture identifier")
        validate_context13_fitted_model(model, instance)
        instance_identity = sha256_json(instance.metadata())
        model_sha = sha256_json(model.metadata())
        training_identity = sha256_json(
            {
                "fixture_id": fixture_id,
                "model_metadata_sha256": model_sha,
                "fitted_instance_identity_sha256": instance_identity,
                "provenance_class": "research_only",
            }
        )
        return cls(
            model_family=instance.model_family,
            spec_version=instance.spec_version,
            target_season=instance.target_season,
            trained_through_season=instance.trained_through_season,
            model_metadata_sha256=model_sha,
            fitted_instance_identity_sha256=instance_identity,
            frozen_model_spec_identity_sha256=sha256_json(model_specification_metadata()),
            training_corpus_input_sha256=training_identity,
            training_row_count=0,
            provenance_class="research_only",
            _construction_token=_CONTEXT13_FITTED_SOURCE_TOKEN,
        )


def _training_row_payload(row: TeamSeason) -> dict[str, object]:
    def vector(values: object) -> list[float]:
        result = np.asarray(values, dtype=float)
        if result.ndim != 1 or not np.isfinite(result).all():
            raise ValueError("Context 1.3 training inputs must be finite vectors")
        return [float(value) for value in result]

    features: dict[str, float | None] = {}
    for name in MODEL_FEATURE_NAMES:
        value = row.features.get(name)
        if value is None:
            features[name] = None
        else:
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError("Context 1.3 training features must be finite or missing")
            features[name] = numeric
    return {
        "season": row.season,
        "subdivision": row.subdivision,
        "team_id": row.team_id,
        "team_name": row.team_name,
        "population": row.population,
        "lag1_z": vector(row.lag1_z),
        "lag_zs": [vector(values) for values in row.lag_zs],
        "target_z": vector(row.target_z),
        "target_ranks": [int(value) for value in np.asarray(row.target_ranks)],
        "features": features,
    }


def _training_corpus_input_sha256(rows: Sequence[TeamSeason]) -> str:
    payload = [_training_row_payload(row) for row in rows]
    keys = [(row["season"], row["subdivision"], row["team_id"]) for row in payload]
    if len(keys) != len(set(keys)):
        raise ValueError("Context 1.3 training corpus has duplicate team-season rows")
    payload.sort(key=lambda row: (row["season"], row["subdivision"], row["team_id"]))
    return sha256_json(payload)


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


def fit_model_with_source(
    rows: Sequence[TeamSeason],
    *,
    target_season: int,
    trained_through_season: int,
) -> tuple[DirectRankModel, AnnualFittedInstance, Context13FittedModelSource]:
    """Fit the frozen Context 1.3 model and bind its exact historical corpus.

    This canonical source-producing path accepts only FBS team-season rows at
    or before the rolling-origin cutoff. It rejects future rows rather than
    silently filtering them, so a source identity cannot hide target-season
    training inputs.
    """
    if trained_through_season != target_season - 1:
        raise ValueError("Context 1.3 annual fits must train through target season - 1")
    if not rows:
        raise ValueError("Context 1.3 fit provenance requires a non-empty training corpus")
    if any(row.season > trained_through_season for row in rows):
        raise ValueError("Context 1.3 training corpus includes rows beyond its cutoff")
    if any(row.subdivision != "fbs" for row in rows):
        raise ValueError("Context 1.3 training corpus must contain only FBS rows")
    model, instance = fit_model(
        list(rows),
        target_season=target_season,
        trained_through_season=trained_through_season,
        context_features=CONTEXT_1_3_FEATURES,
    )
    model_sha = sha256_json(model.metadata())
    instance_sha = sha256_json(instance.metadata())
    corpus_sha = _training_corpus_input_sha256(rows)
    source = Context13FittedModelSource(
        model_family=instance.model_family,
        spec_version=instance.spec_version,
        target_season=instance.target_season,
        trained_through_season=instance.trained_through_season,
        model_metadata_sha256=model_sha,
        fitted_instance_identity_sha256=instance_sha,
        frozen_model_spec_identity_sha256=sha256_json(model_specification_metadata()),
        training_corpus_input_sha256=corpus_sha,
        training_row_count=len(rows),
        provenance_class="canonical_context13_fit",
        _construction_token=_CONTEXT13_FITTED_SOURCE_TOKEN,
    )
    source.validate_model(model, instance, training_rows=rows)
    return model, instance, source


def load_validated_context13_fitted_model(
    model_artifact_path: str | Path,
    fitted_instance_path: str | Path,
    *,
    training_rows: Sequence[TeamSeason] | None = None,
    fitted_source_path: str | Path | None = None,
) -> tuple[DirectRankModel, AnnualFittedInstance, Context13FittedModelSource]:
    """Load an annual model only after validating its authoritative fit source.

    When the corpus is available, refit through the canonical code and require
    exact model and instance parity. The retained 2026 production artifact has
    a separately committed, content-pinned source attestation because its
    generated training table is not part of every checkout. Neither path trusts
    the caller's filesystem path as source identity.
    """
    try:
        model_artifact = json.loads(Path(model_artifact_path).read_text(encoding="utf-8"))
        instance_artifact = json.loads(
            Path(fitted_instance_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Context 1.3 fitted artifacts are unreadable") from error
    if not isinstance(model_artifact, dict) or not isinstance(instance_artifact, dict):
        raise TypeError("Context 1.3 fitted artifacts must contain JSON objects")
    if (
        model_artifact.get("model_family") != "context_prior"
        or model_artifact.get("spec_version") != CONTEXT_PRIOR_CANDIDATE_VERSION
        or not isinstance(model_artifact.get("model"), dict)
    ):
        raise ValueError("annual model artifact does not identify Context 1.3")
    try:
        instance = AnnualFittedInstance(
            model_family=str(instance_artifact["model_family"]),
            spec_version=str(instance_artifact["spec_version"]),
            trained_through_season=int(instance_artifact["trained_through_season"]),
            target_season=int(instance_artifact["target_season"]),
            context_effective_cutoff=instance_artifact.get("context_effective_cutoff"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("annual fitted-instance artifact has an invalid identity") from error
    if training_rows is not None:
        model, expected_instance, source = fit_model_with_source(
            training_rows,
            target_season=instance.target_season,
            trained_through_season=instance.trained_through_season,
        )
        if instance.metadata() != expected_instance.metadata():
            raise ValueError("annual fitted-instance artifact differs from the canonical fit")
        if sha256_json(model_artifact["model"]) != sha256_json(model.metadata()):
            raise ValueError("annual model metadata differs from the canonical Context 1.3 fit")
        recorded_source = model_artifact.get("fit_provenance")
        if recorded_source is not None and recorded_source != source.to_metadata():
            raise ValueError(
                "annual model fit provenance differs from the canonical training corpus"
            )
        source.validate_model(model, instance, training_rows=training_rows)
        return model, instance, source

    if instance.target_season != 2026:
        raise ValueError("an authoritative fit-source attestation is required for this season")
    source_path = Path(fitted_source_path) if fitted_source_path else Path(
        model_artifact_path
    ).with_name("fitted_model_source.json")
    try:
        source_bytes = source_path.read_bytes()
        source_metadata = json.loads(source_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Context 1.3 fitted-source attestation is unreadable") from error
    if hashlib.sha256(source_bytes).hexdigest() != _CONTEXT13_2026_FITTED_SOURCE_ARTIFACT_SHA256:
        raise ValueError("Context 1.3 fitted-source attestation is not the committed source record")
    if (
        instance.model_family != "context_prior"
        or instance.spec_version != CONTEXT_PRIOR_CANDIDATE_VERSION
        or instance.trained_through_season != instance.target_season - 1
        or instance.context_effective_cutoff is not None
        or model_artifact.get("transfer_provenance_class")
        != RETROSPECTIVE_2026_PROVENANCE
        or sha256_json(model_artifact["model"])
        != _CONTEXT13_2026_MODEL_METADATA_SHA256
    ):
        raise ValueError("annual artifact does not match the retained Context 1.3 fit")
    if not isinstance(source_metadata, dict):
        raise TypeError("Context 1.3 fit-source attestation must be a JSON object")
    try:
        source = Context13FittedModelSource(
            model_family=str(source_metadata["model_family"]),
            spec_version=str(source_metadata["spec_version"]),
            target_season=int(source_metadata["target_season"]),
            trained_through_season=int(source_metadata["trained_through_season"]),
            model_metadata_sha256=str(source_metadata["model_metadata_sha256"]),
            fitted_instance_identity_sha256=str(
                source_metadata["fitted_instance_identity_sha256"]
            ),
            frozen_model_spec_identity_sha256=str(
                source_metadata["frozen_model_spec_identity_sha256"]
            ),
            training_corpus_input_sha256=str(
                source_metadata["training_corpus_input_sha256"]
            ),
            training_row_count=int(source_metadata["training_row_count"]),
            provenance_class=str(source_metadata["provenance_class"]),
            _construction_token=_CONTEXT13_FITTED_SOURCE_TOKEN,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Context 1.3 fit-source attestation has an invalid identity") from error
    if source_metadata != source.to_metadata():
        raise ValueError("Context 1.3 fit-source attestation identity does not match its fields")
    model = _direct_rank_model_from_metadata(model_artifact["model"])
    if sha256_json(model_artifact["model"]) != sha256_json(model.metadata()):
        raise ValueError("Context 1.3 annual model metadata cannot be reconstructed")
    source.validate_model(model, instance)
    return model, instance, source


def _direct_rank_model_from_metadata(metadata: Mapping[str, Any]) -> DirectRankModel:
    try:
        preprocessing = metadata["preprocessing"]
        if not isinstance(preprocessing, dict):
            raise TypeError("model preprocessing metadata must be an object")
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
            degrees_of_freedom=(
                None
                if metadata["degrees_of_freedom"] is None
                else float(metadata["degrees_of_freedom"])
            ),
            location_feature_names=list(metadata["location_feature_names"]),
            scale_feature_names=list(metadata["scale_feature_names"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("annual Context 1.3 model metadata is malformed") from error


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


def load_validated_committed_2026_reconstruction(
    artifact_directory: str | Path,
) -> tuple[list[dict[str, str]], ContextTransferInputProvenance]:
    """Validate the retained 2026 reconstruction from its committed attestations.

    The original raw snapshot payloads are intentionally not checked into the
    repository. This path accepts only the exact committed derived CSV,
    processed source manifest, and provenance attestation whose content hashes
    were recorded when the reconstruction was validated. It preserves the
    historical provenance value and creates the same typed identity as the
    raw-backed loader, without relying on checkout-specific paths.
    """
    directory = Path(artifact_directory)
    feature_path = directory / "transfer_features.csv"
    manifest_path = directory / "source_manifest.json"
    provenance_path = directory / "feature_provenance.json"
    try:
        feature_bytes = feature_path.read_bytes()
        manifest_bytes = manifest_path.read_bytes()
        provenance_bytes = provenance_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        provenance = json.loads(provenance_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestValidationError(
            "committed 2026 transfer reconstruction artifacts are unreadable"
        ) from error
    feature_sha = hashlib.sha256(feature_bytes).hexdigest()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    provenance_sha = hashlib.sha256(provenance_bytes).hexdigest()
    if (
        feature_sha != _COMMITTED_2026_TRANSFER_FEATURE_SHA256
        or manifest_sha != _COMMITTED_2026_TRANSFER_MANIFEST_SHA256
        or provenance_sha != _COMMITTED_2026_TRANSFER_PROVENANCE_SHA256
    ):
        raise ManifestValidationError(
            "committed 2026 transfer artifacts do not match the retained attestation"
        )
    if not isinstance(manifest, dict) or not isinstance(provenance, dict):
        raise ManifestValidationError("committed 2026 transfer metadata must be JSON objects")
    records = manifest.get("snapshots")
    if not isinstance(records, list) or len(records) != 37:
        raise ManifestValidationError("committed 2026 source manifest has an invalid snapshot set")
    snapshot_ids: list[str] = []
    snapshot_hashes: list[str] = []
    retrieval_timestamps: list[str] = []
    source_endpoints: list[str] = []
    source_kinds: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ManifestValidationError("committed 2026 source manifest has an invalid record")
        snapshot_id = record.get("snapshot_id")
        digest = record.get("sha256")
        timestamp = record.get("retrieval_timestamp")
        endpoint = record.get("endpoint")
        if (
            record.get("target_season") != 2026
            or record.get("target_cutoff") != "2026-08-15"
            or record.get("canonical") is not True
            or record.get("captured_on_or_before_cutoff") is not False
            or not isinstance(snapshot_id, str)
            or not snapshot_id
            or not _is_sha256(digest)
            or not isinstance(timestamp, str)
            or not timestamp
            or not isinstance(endpoint, str)
            or not endpoint
            or not isinstance(record.get("source"), str)
        ):
            raise ManifestValidationError("committed 2026 snapshot metadata is incomplete")
        snapshot_ids.append(snapshot_id)
        snapshot_hashes.append(digest)
        retrieval_timestamps.append(timestamp)
        source_endpoints.append(endpoint)
        source_kinds.add(str(record["source"]))
    expected_sources = {"games_players", "portal", "roster", "stats", "usage"}
    if (
        len(snapshot_ids) != len(set(snapshot_ids))
        or source_kinds != expected_sources
        or provenance.get("provenance_class") != RETROSPECTIVE_2026_PROVENANCE
        or provenance.get("target_season") != 2026
        or provenance.get("cutoff") != "2026-08-15"
        or provenance.get("all_snapshots_on_or_before_cutoff") is not False
        or provenance.get("known_absence_of_archived_august_15_transfer_snapshot") is not True
        or provenance.get("source_manifest_sha256") != manifest_sha
        or provenance.get("snapshot_ids") != snapshot_ids
        or provenance.get("snapshot_sha256") != snapshot_hashes
        or provenance.get("raw_source_hashes") != snapshot_hashes
        or provenance.get("retrieval_timestamps") != retrieval_timestamps
        or provenance.get("source_endpoints") != source_endpoints
        or not isinstance(provenance.get("provenance_statement"), str)
        or not provenance["provenance_statement"].strip()
    ):
        raise ManifestValidationError("committed 2026 provenance does not match its source manifest")
    feature_rows = _read_csv_bytes(feature_bytes, str(feature_path))
    expected_columns = {
        "season",
        "subdivision",
        "team_id",
        "team_name",
        *TRANSFER_FEATURE_COLUMNS,
        "provenance_class",
    }
    if not feature_rows or set(feature_rows[0]) != expected_columns:
        raise ManifestValidationError("committed 2026 transfer feature schema is invalid")
    actual: dict[str, dict[str, float | None]] = {}
    for row in feature_rows:
        if (
            row.get("season") != "2026"
            or row.get("subdivision") != "fbs"
            or row.get("provenance_class") != RETROSPECTIVE_2026_PROVENANCE
            or not row.get("team_id")
            or not row.get("team_name")
        ):
            raise ManifestValidationError("committed 2026 transfer feature row is invalid")
        team_id = row["team_id"]
        if team_id in actual:
            raise ManifestValidationError("committed 2026 transfer features duplicate a team")
        values: dict[str, float | None] = {}
        try:
            for name in TRANSFER_FEATURE_COLUMNS:
                values[name] = None if not row[name].strip() else float(row[name])
            _team_feature_sha256(values)
        except (KeyError, TypeError, ValueError) as error:
            raise ManifestValidationError(
                "committed 2026 transfer feature value is invalid"
            ) from error
        actual[team_id] = values
    if len(actual) != 138:
        raise ManifestValidationError("committed 2026 transfer population must contain 138 FBS teams")
    team_hashes = _team_transfer_feature_hashes(actual)
    identity = ContextTransferInputProvenance(
        target_season=2026,
        provenance_class=RETROSPECTIVE_2026_PROVENANCE,
        transfer_feature_artifact_sha256=feature_sha,
        source_manifest_sha256=manifest_sha,
        canonical_snapshot_ids=tuple(snapshot_ids),
        canonical_snapshot_sha256=tuple(snapshot_hashes),
        cutoff_state="retrospective_reconstruction",
        feature_artifact_id="gippyrank.context1_3.transfer_features.season_2026",
        manifest_artifact_id="gippyrank.preseason_transfer.snapshot_manifest.season_2026",
        target_fbs_team_ids=tuple(team_id for team_id, _ in team_hashes),
        target_fbs_population_sha256=sha256_json(
            [team_id for team_id, _ in team_hashes]
        ),
        target_team_feature_sha256=team_hashes,
        retrieval_timestamps=tuple(retrieval_timestamps),
        source_endpoints=tuple(source_endpoints),
        reconstructed_state_declaration=str(provenance["provenance_statement"]),
        archived_august_15_snapshot_absent=True,
        diagnostic_paths=(str(feature_path), str(manifest_path)),
        _construction_token=_CONTEXT_TRANSFER_PROVENANCE_TOKEN,
    )
    return feature_rows, identity


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
    "Context13FittedModelSource",
    "Context13LocationDecomposition",
    "ContextTransferInputProvenance",
    "attach_transfer_features",
    "attach_transfer_features_to_inference_rows",
    "candidate_guard",
    "decompose_context13_location",
    "fit_model",
    "fit_model_with_source",
    "load_validated_committed_2026_reconstruction",
    "load_validated_context13_fitted_model",
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
