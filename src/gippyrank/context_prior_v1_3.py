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
RETROSPECTIVE_2026_PROVENANCE = "retrospective_2026_reconstruction"
RETROSPECTIVE_RESEARCH_PROVENANCE = "retrospective_research_reconstruction"
PRODUCTION_TRANSFER_PROVENANCE = "production_preseason_immutable_snapshot"
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

_CONTEXT_INPUT_PROVENANCE_CLASSES = frozenset(
    {
        PRODUCTION_TRANSFER_PROVENANCE,
        RETROSPECTIVE_2026_PROVENANCE,
        RETROSPECTIVE_RESEARCH_PROVENANCE,
    }
)
_LOCATION_RECONSTRUCTION_TOLERANCE = 1e-8


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: object) -> str:
    """Hash a JSON-compatible model or inference representation."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ContextInputProvenance:
    """Identity of the validated source state behind target inference inputs."""

    target_season: int
    provenance_class: str
    source_metadata_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_season, bool)
            or not isinstance(self.target_season, int)
            or self.target_season < 1
        ):
            raise ValueError("Context input provenance requires a valid target season")
        if self.provenance_class not in _CONTEXT_INPUT_PROVENANCE_CLASSES:
            raise ValueError("unsupported Context input provenance class")
        digest = self.source_metadata_sha256
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Context input provenance must carry a SHA-256 identity")

    @classmethod
    def from_validated_metadata(
        cls, target_season: int, metadata: Mapping[str, object]
    ) -> ContextInputProvenance:
        """Bind the exact metadata returned by a validated transfer loader."""
        metadata_season = metadata.get("target_season")
        if isinstance(metadata_season, bool) or not isinstance(metadata_season, int):
            raise TypeError("validated Context input metadata has an invalid target season")
        if metadata_season != target_season:
            raise ValueError("validated Context input metadata has the wrong season")
        provenance_class = metadata.get("provenance_class")
        if not isinstance(provenance_class, str):
            raise TypeError("validated Context input metadata lacks its provenance class")
        return cls(target_season, provenance_class, sha256_json(dict(metadata)))

    @classmethod
    def from_research_metadata(
        cls, target_season: int, metadata: Mapping[str, object]
    ) -> ContextInputProvenance:
        """Bind a retrospective research panel without implying archived inputs."""
        if metadata.get("provenance_class") != RETROSPECTIVE_RESEARCH_PROVENANCE:
            raise ValueError("research metadata must be labeled retrospective reconstruction")
        return cls.from_validated_metadata(target_season, metadata)


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
    input_provenance: ContextInputProvenance
    intercept: float
    history_derived_subtotal: float
    context_only_subtotal: float
    conditional_location_points: np.ndarray
    residual_scale: float

    def __post_init__(self) -> None:
        if self.input_provenance.target_season != self.target_season:
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
        object.__setattr__(self, "conditional_location_points", points.copy())

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
    input_provenance: ContextInputProvenance,
) -> Context13LocationDecomposition:
    """Derive fitted Context 1.3 location terms from a validated input row.

    The result is bound to the actual model metadata, input values, team, target
    season, and the source provenance returned by the inference-input validator.
    """
    model_sha256 = validate_context13_fitted_model(model, instance)
    if (
        row.season != instance.target_season
        or input_provenance.target_season != instance.target_season
    ):
        raise ValueError("Context decomposition team input and model seasons differ")
    if input_provenance.provenance_class not in _CONTEXT_INPUT_PROVENANCE_CLASSES:
        raise ValueError("Context decomposition input provenance is invalid")
    if row.subdivision != "fbs":
        raise ValueError("Context 1.3 candidate decomposition requires an FBS team")
    if row.population < 1 or row.lag1_z is None or not len(row.lag1_z):
        raise ValueError("fitted Context 1.3 inference requires a lag-1 rank distribution")
    if row.lag_zs:
        raise ValueError("Context 1.3 inference accepts only its frozen lag-1 distribution")
    if set(row.features) != set(MODEL_FEATURE_NAMES):
        raise ValueError("inference row features do not match the Context 1.3 specification")

    features = {name: row.features[name] for name in MODEL_FEATURE_NAMES}
    lag1 = np.asarray(row.lag1_z, dtype=float)
    if not np.isfinite(lag1).all():
        raise ValueError("lag-1 inference values must be finite")
    locations, scale = model.conditional_parameters(features, lag1)
    design = model._matrix(features)
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
        input_provenance=input_provenance,
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
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Load a target artifact only after validating its immutable input chain."""
    manifest = load_snapshot_manifest(
        manifest_path, required_seasons=[target_season], verify_hashes=True
    )
    records = manifest.for_target(target_season)
    if not records:
        raise ManifestValidationError(
            f"target season {target_season} has no canonical transfer snapshots"
        )
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
    if provenance.get("provenance_class") != expected_provenance_class:
        raise ManifestValidationError(
            f"target transfer provenance must be {expected_provenance_class}"
        )
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
    if require_on_time_cutoff and provenance.get("all_snapshots_on_or_before_cutoff") is not True:
        raise ManifestValidationError(
            "transfer provenance does not certify the cutoff boundary"
        )
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
            record.retrieval_timestamp for record in records
        ]:
            raise ManifestValidationError(
                "retrospective transfer retrieval timestamps do not match the manifest"
            )
        if provenance["source_endpoints"] != [record.endpoint for record in records]:
            raise ManifestValidationError(
                "retrospective transfer source endpoints do not match the manifest"
            )
        raw_hashes = provenance["raw_source_hashes"]
        if not isinstance(raw_hashes, list) or sorted(raw_hashes) != sorted(
            expected_snapshot_hashes
        ):
            raise ManifestValidationError(
                "retrospective transfer raw source hashes do not match the manifest"
            )
    metadata = {
        **_manifest_provenance(
            manifest, target_season, provenance_class=expected_provenance_class
        ),
        "feature_artifact": str(feature_path),
        "manifest": str(manifest_path),
    }
    return [actual[key] for key in sorted(actual)], metadata


def load_validated_production_transfer_features(
    feature_path: Path,
    manifest_path: Path,
    *,
    target_season: int,
    expected_team_keys: Iterable[tuple[int, str, str]],
    provenance: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, object]]:
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
) -> tuple[list[dict[str, str]], dict[str, object]]:
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
    "ContextInputProvenance",
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
