"""Canonical, reproducible History 1.1 annual build from processed inputs.

The public build and verification paths always reconstruct their own rows from
the repository's historical artifacts. Neither accepts fitted models, PMFs,
prediction files, or caller-created attestations as a source of authority.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gippyrank import preseason as fit_semantics
from gippyrank.context_prior import AnnualFittedInstance
from gippyrank.preseason import (
    DirectRankModel,
    GenericRankPrior,
    TeamSeason,
    historical_rank_features,
    rank_to_z,
)

_CANONICAL_HISTORY_ROOT = Path(__file__).resolve().parents[2]
HISTORY_1_1_FEATURES = ("lag2_z_mean", "lag3_z_mean", "long_run_z_mean")
HISTORY_1_1_TRANSITION_FEATURES: tuple[str, ...] = ()
HISTORY_1_1_PENALTY = 0.25
HISTORY_1_1_MINIMUM_SCALE = 0.10
HISTORY_1_1_LAG_COUNT = 1
HISTORY_1_1_FAMILY = "normal"
HISTORY_1_1_DEGREES_OF_FREEDOM = None
HISTORY_1_1_ROW_WEIGHT = 1.0
HISTORY_1_1_PMF_DECIMALS = 12
_INPUT_TOKEN = object()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def history11_semantic_specification() -> dict[str, object]:
    """The annual History fit, input, and fallback construction contract."""
    preprocessing = {
        "missing_value_imputation": "training median, or zero when all missing",
        "standardization": "training mean and population standard deviation",
        "scale_floor": fit_semantics.PREPROCESSOR_SCALE_FLOOR,
        "std_ddof": fit_semantics.PREPROCESSOR_STD_DDOF,
        "missingness_indicators": True,
    }
    optimizer = {
        "method": fit_semantics.DIRECT_RANK_OPTIMIZER_METHOD,
        "maxiter": fit_semantics.DIRECT_RANK_OPTIMIZER_MAXITER,
        "ftol": fit_semantics.DIRECT_RANK_OPTIMIZER_FTOL,
        "gtol": fit_semantics.DIRECT_RANK_OPTIMIZER_GTOL,
        "initial_lag_beta": fit_semantics.DIRECT_RANK_INITIAL_LAG_BETA,
        "initial_scale": fit_semantics.DIRECT_RANK_INITIAL_SCALE,
        "gamma_bounds": list(fit_semantics.DIRECT_RANK_GAMMA_COEFFICIENT_BOUNDS),
        "log_scale_clip": list(fit_semantics.DIRECT_RANK_LOG_SCALE_CLIP_BOUNDS),
        "beta_regularization_weight": fit_semantics.DIRECT_RANK_BETA_REGULARIZATION_WEIGHT,
        "gamma_regularization_weight": fit_semantics.DIRECT_RANK_GAMMA_REGULARIZATION_WEIGHT,
        "gamma_intercept_penalized": False,
    }
    return {
        "model_family": "history_prior",
        "spec_version": "1.1",
        "annual_build_semantics_version": 2,
        "ordered_features": list(HISTORY_1_1_FEATURES),
        "distribution_family": HISTORY_1_1_FAMILY,
        "degrees_of_freedom": HISTORY_1_1_DEGREES_OF_FREEDOM,
        "lag_count": HISTORY_1_1_LAG_COUNT,
        "penalty": HISTORY_1_1_PENALTY,
        "minimum_scale": HISTORY_1_1_MINIMUM_SCALE,
        "rolling_origin_cutoff": "trained_through_season == target_season - 1",
        "quadrature_points": fit_semantics.QUADRATURE_POINTS,
        "quadrature_method": fit_semantics.DETERMINISTIC_QUADRATURE_METHOD,
        "team_season_weight": HISTORY_1_1_ROW_WEIGHT,
        "preprocessing": preprocessing,
        "optimizer": optimizer,
        "rank_coordinate_transform": {
            "semantics_version": fit_semantics.RANK_TRANSFORM_SEMANTICS_VERSION,
            "percentile": "(rank - midpoint_offset) / population",
            "midpoint_offset": fit_semantics.RANK_PERCENTILE_MIDPOINT_OFFSET,
            "clip_epsilon": fit_semantics.RANK_TRANSFORM_EPSILON,
            "link": "log(percentile) - log1p(-percentile)",
        },
        "pmf_integration": {
            "semantics_version": fit_semantics.RANK_PMF_INTEGRATION_SEMANTICS_VERSION,
            "rank_bin_interior": "logit(k / population), k=1..population-1",
            "lower_endpoint": str(fit_semantics.RANK_BIN_LOWER_ENDPOINT),
            "upper_endpoint": str(fit_semantics.RANK_BIN_UPPER_ENDPOINT),
            "family": HISTORY_1_1_FAMILY,
            "conditional_mass": "Normal CDF difference at adjacent rank-bin edges",
            "negative_mass_floor": fit_semantics.RANK_PMF_MASS_FLOOR,
            "mixture_weighting": "equal weight for each empirical lag-1 coordinate",
            "normalization": "divide all nonnegative masses by their sum",
        },
        "training": "all eligible FBS outcome rows through target season minus one",
        "historical_features": "same-subdivision prior ranks and completed rank history",
        "target_population": "exact FBS rows in the target team-season feature artifact",
        "target_outcomes": "forbidden in annual build inputs",
        "input_retention": "copy exact canonical input bytes into the annual artifact for later reproduction",
        "cold_start": {
            "fcs_transition": {
                "eligible_training_rows": "historical FBS outcomes through T-1 with prior-season FCS ranks and no prior-season FBS ranks",
                "ordered_features": list(HISTORY_1_1_TRANSITION_FEATURES),
                "lag_source": "prior-season empirical FCS rank-coordinate distribution",
                "penalty": HISTORY_1_1_PENALTY,
                "minimum_scale": HISTORY_1_1_MINIMUM_SCALE,
                "lag_count": HISTORY_1_1_LAG_COUNT,
                "distribution_family": HISTORY_1_1_FAMILY,
                "degrees_of_freedom": HISTORY_1_1_DEGREES_OF_FREEDOM,
                "team_season_weight": HISTORY_1_1_ROW_WEIGHT,
                "optimizer": optimizer,
                "preprocessing": preprocessing,
            },
            "no_prior": {
                "eligible_training_rows": "historical FBS outcomes through T-1 with neither prior-season FBS nor FCS ranks",
                "moments_semantics_version": fit_semantics.GENERIC_RANK_PRIOR_MOMENTS_VERSION,
                "team_season_weight": HISTORY_1_1_ROW_WEIGHT,
                "location_estimator": "mean of per-team empirical target-coordinate means",
                "variance_estimator": "mean of per-team empirical squared deviations from pooled location",
                "scale": "max(sqrt(variance), minimum_scale)",
                "minimum_scale": HISTORY_1_1_MINIMUM_SCALE,
                "pmf": "single Normal CDF rank-bin integration with clipping and renormalization",
            },
        },
        "pmf_serialization": {
            "decimal_places": HISTORY_1_1_PMF_DECIMALS,
            "method": "Python round each rank probability before compact JSON encoding",
        },
    }


def history11_semantic_specification_sha256() -> str:
    return _sha256_json(history11_semantic_specification())


@dataclass(frozen=True, init=False)
class History11AnnualBuildInputSource:
    """Identity of the canonical processed inputs and exact derived annual rows."""

    target_season: int
    trained_through_season: int
    historical_rank_artifact_sha256: str
    team_feature_artifact_sha256: str
    training_rows_sha256: str
    training_row_count: int
    target_input_rows_sha256: str
    target_team_ids_sha256: str
    target_population: int
    historical_cold_rows_sha256: str
    source_identity_sha256: str

    def __init__(self, payload: dict[str, object], *, _token: object = None) -> None:
        if _token is not _INPUT_TOKEN:
            raise TypeError("History annual build inputs require the canonical loader")
        for key, value in payload.items():
            object.__setattr__(self, key, value)
        object.__setattr__(self, "source_identity_sha256", _sha256_json(payload))

    def to_metadata(self) -> dict[str, object]:
        return {
            "target_season": self.target_season,
            "trained_through_season": self.trained_through_season,
            "historical_rank_artifact_sha256": self.historical_rank_artifact_sha256,
            "team_feature_artifact_sha256": self.team_feature_artifact_sha256,
            "training_rows_sha256": self.training_rows_sha256,
            "training_row_count": self.training_row_count,
            "target_input_rows_sha256": self.target_input_rows_sha256,
            "target_team_ids_sha256": self.target_team_ids_sha256,
            "target_population": self.target_population,
            "historical_cold_rows_sha256": self.historical_cold_rows_sha256,
            "source_identity_sha256": self.source_identity_sha256,
        }


@dataclass(frozen=True)
class _TargetInput:
    team_id: str
    team_name: str
    population: int
    lag1_z: np.ndarray | None
    features: dict[str, float | None]
    cold_start_reason: str | None
    cross_subdivision_lag_z: np.ndarray | None


def _csv_rows(raw: bytes, label: str) -> list[dict[str, str]]:
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
        if not reader.fieldnames or len(reader.fieldnames) != len(
            set(reader.fieldnames)
        ):
            raise ValueError(f"{label} has invalid CSV columns")
        return list(reader)
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"{label} is unreadable") from error


def _rank_values(row: dict[str, str]) -> np.ndarray:
    try:
        population = int(row["team_population"])
        ranks = np.asarray(json.loads(row["rank_observations"]), dtype=float)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("historical rank row is invalid") from error
    if population < 1 or ranks.ndim != 1 or not np.isfinite(ranks).all():
        raise ValueError("historical rank row has invalid population or ranks")
    # A small number of constituent ranks lie outside the composite roster.
    # History 1.1 excludes those observations and skips rows with no usable
    # outcome, matching the retained historical row builder.
    ranks = ranks.astype(int)
    return ranks[(ranks >= 1) & (ranks <= population)]


def _row_payload(row: TeamSeason) -> dict[str, object]:
    return {
        "season": row.season,
        "team_id": row.team_id,
        "team_name": row.team_name,
        "population": row.population,
        "lag1_z": row.lag1_z.tolist(),
        "target_z": row.target_z.tolist(),
        "target_ranks": row.target_ranks.tolist(),
        "features": row.features,
    }


def _load_inputs(
    target_season: int,
    *,
    from_snapshot: bool = False,
) -> tuple[
    History11AnnualBuildInputSource,
    list[TeamSeason],
    list[TeamSeason],
    list[TeamSeason],
    list[_TargetInput],
]:
    if target_season <= 2026:
        raise ValueError("canonical History annual builds begin after retained 2026")
    trained_through = target_season - 1
    if from_snapshot:
        input_dir = (
            _CANONICAL_HISTORY_ROOT
            / f"data/processed/preseason/history/annual/{target_season}/source_inputs"
        )
        rank_path = input_dir / "team_season_rank_distributions.csv"
        feature_path = input_dir / "team_season_features.csv"
    else:
        rank_path = (
            _CANONICAL_HISTORY_ROOT
            / "data/processed/modeling/team_season_rank_distributions.csv"
        )
        feature_path = (
            _CANONICAL_HISTORY_ROOT
            / "data/processed/preseason/team_season_features.csv"
        )
    try:
        rank_raw, feature_raw = rank_path.read_bytes(), feature_path.read_bytes()
    except OSError as error:
        raise ValueError(
            "canonical History historical inputs are unavailable"
        ) from error
    rank_rows = _csv_rows(rank_raw, "historical rank artifact")
    feature_rows = _csv_rows(feature_raw, "team feature artifact")
    outcomes: dict[tuple[int, str, str], dict[str, str]] = {}
    for row in rank_rows:
        try:
            key = (int(row["season"]), row["subdivision"], row["team_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "historical rank artifact has invalid team keys"
            ) from error
        if key in outcomes or not key[2] or key[0] >= target_season:
            raise ValueError(
                "historical ranks have duplicate or target-season outcomes"
            )
        _rank_values(row)
        outcomes[key] = row
    target_features: dict[str, dict[str, str]] = {}
    for row in feature_rows:
        try:
            season = int(row["season"])
            division, team_id = row["subdivision"], row["team_id"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("team feature artifact has invalid team keys") from error
        if season > target_season:
            raise ValueError("team feature artifact contains future seasons")
        if season == target_season and division == "fbs":
            if not team_id or team_id in target_features or not row.get("team_name"):
                raise ValueError("target FBS feature population has invalid team IDs")
            target_features[team_id] = row
    if (
        not outcomes
        or not target_features
        or not any(
            season == trained_through and division == "fbs"
            for season, division, _team_id in outcomes
        )
    ):
        raise ValueError("canonical History inputs lack historical or target FBS rows")

    def z(key: tuple[int, str, str]) -> np.ndarray | None:
        row = outcomes.get(key)
        if row is None:
            return None
        ranks = _rank_values(row)
        return None if not len(ranks) else rank_to_z(ranks, int(row["team_population"]))

    def history_features(
        season: int, team_id: str, lag1: np.ndarray
    ) -> dict[str, float | None]:
        prior = tuple(
            values
            for year in range(2002, season)
            if (values := z((year, "fbs", team_id))) is not None
        )
        features = historical_rank_features(lag1, prior)
        lag2 = z((season - 2, "fbs", team_id))
        lag3 = z((season - 3, "fbs", team_id))
        return {
            "lag2_z_mean": None if lag2 is None else float(np.mean(lag2)),
            "lag3_z_mean": None if lag3 is None else float(np.mean(lag3)),
            "long_run_z_mean": features["long_run_z_mean"],
        }

    training: list[TeamSeason] = []
    promotion: list[TeamSeason] = []
    generic: list[TeamSeason] = []
    for (season, division, team_id), row in sorted(outcomes.items()):
        if division != "fbs":
            continue
        target_ranks = _rank_values(row)
        if not len(target_ranks):
            continue
        target_z = rank_to_z(target_ranks, int(row["team_population"]))
        name = row.get("team_name") or team_id
        lag1 = z((season - 1, "fbs", team_id))
        if lag1 is not None:
            training.append(
                TeamSeason(
                    season,
                    "fbs",
                    team_id,
                    name,
                    int(row["team_population"]),
                    lag1,
                    target_z,
                    target_ranks,
                    history_features(season, team_id, lag1),
                )
            )
            continue
        cross = z((season - 1, "fcs", team_id))
        cold_row = TeamSeason(
            season,
            "fbs",
            team_id,
            name,
            int(row["team_population"]),
            cross if cross is not None else np.asarray([0.0]),
            target_z,
            target_ranks,
            {},
        )
        (promotion if cross is not None else generic).append(cold_row)
    if not training:
        raise ValueError("canonical History training corpus is empty")
    population = len(target_features)
    target: list[_TargetInput] = []
    for team_id, row in sorted(target_features.items()):
        lag1 = z((trained_through, "fbs", team_id))
        cross = z((trained_through, "fcs", team_id)) if lag1 is None else None
        target.append(
            _TargetInput(
                team_id,
                row["team_name"],
                population,
                lag1,
                history_features(target_season, team_id, lag1)
                if lag1 is not None
                else {},
                None
                if lag1 is not None
                else (
                    "fcs_to_fbs_transition"
                    if cross is not None
                    else "no_prior_rank_distribution"
                ),
                cross,
            )
        )
    input_payload = {
        "target_season": target_season,
        "trained_through_season": trained_through,
        "historical_rank_artifact_sha256": hashlib.sha256(rank_raw).hexdigest(),
        "team_feature_artifact_sha256": hashlib.sha256(feature_raw).hexdigest(),
        "training_rows_sha256": _sha256_json([_row_payload(row) for row in training]),
        "training_row_count": len(training),
        "target_input_rows_sha256": _sha256_json(
            [
                {
                    "team_id": row.team_id,
                    "team_name": row.team_name,
                    "population": row.population,
                    "lag1_z": None if row.lag1_z is None else row.lag1_z.tolist(),
                    "features": row.features,
                    "cold_start_reason": row.cold_start_reason,
                    "cross_subdivision_lag_z": None
                    if row.cross_subdivision_lag_z is None
                    else row.cross_subdivision_lag_z.tolist(),
                }
                for row in target
            ]
        ),
        "target_team_ids_sha256": _sha256_json(sorted(target_features)),
        "target_population": population,
        "historical_cold_rows_sha256": _sha256_json(
            [_row_payload(row) for row in [*promotion, *generic]]
        ),
    }
    return (
        History11AnnualBuildInputSource(input_payload, _token=_INPUT_TOKEN),
        training,
        promotion,
        generic,
        target,
    )


def load_canonical_history_annual_build_inputs(
    target_season: int,
) -> History11AnnualBuildInputSource:
    """Return the typed identity established by the canonical input loader."""
    return _load_inputs(target_season)[0]


def _predict_annual_team(
    row: _TargetInput,
    target_season: int,
    model: DirectRankModel,
    promotion_model: DirectRankModel | None,
    generic_prior: GenericRankPrior | None,
) -> dict[str, object]:
    """Apply the three History 1.1 annual prediction arms in one place."""
    if row.lag1_z is not None:
        pmf = model.pmf(row.features, row.lag1_z, row.population)
        locations, scale = model.conditional_parameters(row.features, row.lag1_z)
        method = "same_subdivision_lag1"
    elif row.cold_start_reason == "fcs_to_fbs_transition":
        if promotion_model is None or row.cross_subdivision_lag_z is None:
            raise ValueError("History transition source is unavailable")
        pmf = promotion_model.pmf({}, row.cross_subdivision_lag_z, row.population)
        locations, scale = promotion_model.conditional_parameters(
            {}, row.cross_subdivision_lag_z
        )
        method = "learned_fcs_to_fbs_transition"
    elif row.cold_start_reason == "no_prior_rank_distribution":
        if generic_prior is None:
            raise ValueError("History generic source is unavailable")
        pmf = generic_prior.pmf(row.population)
        locations = np.asarray([generic_prior.location])
        scale = generic_prior.scale
        method = "generic_fbs_cold_start"
    else:
        raise ValueError("History annual inference row has an unknown prior source")
    return {
        "season": target_season,
        "subdivision": "fbs",
        "team_id": row.team_id,
        "team_name": row.team_name,
        "model_family": "history_prior",
        "spec_version": "1.1",
        "trained_through_season": target_season - 1,
        "pmf": json.dumps(
            [round(float(value), HISTORY_1_1_PMF_DECIMALS) for value in pmf],
            separators=(",", ":"),
        ),
        "prior_method": method,
        "conditional_location_mean": float(np.mean(locations)),
        "predictive_scale": scale,
    }


def reproduce_canonical_history_annual(
    target_season: int,
    *,
    from_snapshot: bool = True,
) -> tuple[bytes, dict[str, object], History11AnnualBuildInputSource]:
    """Rebuild a future annual output; used by both writer and verifier."""
    source, training, promotion, generic, target = _load_inputs(
        target_season, from_snapshot=from_snapshot
    )

    def fit(rows: list[TeamSeason], features: list[str]) -> DirectRankModel:
        return DirectRankModel.fit(
            rows,
            features,
            penalty=HISTORY_1_1_PENALTY,
            minimum_scale=HISTORY_1_1_MINIMUM_SCALE,
            lag_count=HISTORY_1_1_LAG_COUNT,
            family=HISTORY_1_1_FAMILY,
            degrees_of_freedom=HISTORY_1_1_DEGREES_OF_FREEDOM,
            row_weights=np.full(len(rows), HISTORY_1_1_ROW_WEIGHT),
            optimizer_options={
                "maxiter": fit_semantics.DIRECT_RANK_OPTIMIZER_MAXITER,
                "ftol": fit_semantics.DIRECT_RANK_OPTIMIZER_FTOL,
                "gtol": fit_semantics.DIRECT_RANK_OPTIMIZER_GTOL,
            },
            preprocessor_scale_floor=fit_semantics.PREPROCESSOR_SCALE_FLOOR,
            preprocessor_std_ddof=fit_semantics.PREPROCESSOR_STD_DDOF,
        )

    model = fit(training, list(HISTORY_1_1_FEATURES))
    needs_promotion = any(
        row.cold_start_reason == "fcs_to_fbs_transition" for row in target
    )
    needs_generic = any(
        row.cold_start_reason == "no_prior_rank_distribution" for row in target
    )
    if needs_promotion and not promotion:
        raise ValueError("History transition fallback lacks historical training rows")
    if needs_generic and not generic:
        raise ValueError("History generic fallback lacks historical training rows")
    promotion_model = (
        fit(promotion, list(HISTORY_1_1_TRANSITION_FEATURES))
        if needs_promotion
        else None
    )
    generic_prior = (
        GenericRankPrior.fit(generic, minimum_scale=HISTORY_1_1_MINIMUM_SCALE)
        if needs_generic
        else None
    )
    predictions = [
        _predict_annual_team(row, target_season, model, promotion_model, generic_prior)
        for row in target
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=list(predictions[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(predictions)
    instance = AnnualFittedInstance(
        "history_prior", "1.1", target_season - 1, target_season
    )
    fitted = {**instance.metadata(), "model": model.metadata()}
    return output.getvalue().encode("utf-8"), fitted, source


def build_canonical_history_annual(target_season: int, output_dir: Path) -> object:
    """Emit predictions, fitted instance, and its authoritative source together."""
    if target_season <= 2026:
        raise ValueError("retained 2026 History artifacts must not be rewritten")
    expected_dir = (
        _CANONICAL_HISTORY_ROOT
        / f"data/processed/preseason/history/annual/{target_season}"
    )
    if output_dir.resolve() != expected_dir.resolve():
        raise ValueError(
            "canonical History annual output requires its fixed repository directory"
        )
    if any(
        (output_dir / name).exists()
        for name in (
            "predictions.csv",
            "fitted_instance.json",
            "fitted_model_source.json",
        )
    ):
        raise FileExistsError("canonical History annual artifacts already exist")
    live_rank_path = (
        _CANONICAL_HISTORY_ROOT
        / "data/processed/modeling/team_season_rank_distributions.csv"
    )
    live_feature_path = (
        _CANONICAL_HISTORY_ROOT / "data/processed/preseason/team_season_features.csv"
    )
    try:
        rank_bytes = live_rank_path.read_bytes()
        feature_bytes = live_feature_path.read_bytes()
    except OSError as error:
        raise ValueError(
            "canonical History historical inputs are unavailable"
        ) from error
    input_dir = output_dir / "source_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "team_season_rank_distributions.csv").write_bytes(rank_bytes)
    (input_dir / "team_season_features.csv").write_bytes(feature_bytes)
    prediction_bytes, fitted, _inputs = reproduce_canonical_history_annual(
        target_season
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.csv"
    instance_path = output_dir / "fitted_instance.json"
    source_path = output_dir / "fitted_model_source.json"
    prediction_path.write_bytes(prediction_bytes)
    instance_path.write_text(
        json.dumps(fitted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    annual_source = _history_annual_source_from_files(
        prediction_path,
        instance_path,
        target_season=target_season,
        trained_through_season=target_season - 1,
        provenance_class="canonical_history_1_1_annual_output",
    )
    source_path.write_text(
        json.dumps(annual_source.to_metadata(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return annual_source

# History annual provenance is owned and verified here. Context 1.4 imports
# only the public source type and loader, never a History construction helper.
_HISTORY_ANNUAL_SOURCE_TOKEN = object()
_CANONICAL_HISTORY_AUTHORITY_TOKEN = object()
_RETAINED_HISTORY_AUTHORITY_TOKEN = object()
_RESEARCH_HISTORY_AUTHORITY_TOKEN = object()
_HISTORY_2026_PREDICTIONS_SEMANTIC_SHA256 = (
    "12ab4ee6ba4d75b0cdd5855d9a13a99f9769919683f3d3aee6ecb2a7485ae9b1"
)
_HISTORY_2026_MODEL_METADATA_SHA256 = (
    "159423c81d5f9bc5d12b8ccf65c1e185512a1a5ba73e23fad86108fdebef314d"
)
_HISTORY_ANNUAL_LEGACY_SCHEMA_VERSION = 1
_HISTORY_ANNUAL_CANONICAL_SCHEMA_VERSION = 2
_HISTORY_1_1_PRIOR_METHODS = frozenset({
    "same_subdivision_lag1", "learned_fcs_to_fbs_transition", "generic_fbs_cold_start"
})


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_history_pmf(pmf: np.ndarray, population: int) -> None:
    values = np.asarray(pmf, dtype=float)
    if (
        values.ndim != 1
        or len(values) != population
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.isclose(values.sum(), 1.0, rtol=0, atol=1e-8)
    ):
        raise ValueError("History annual PMF is invalid")


@dataclass(frozen=True, init=False)
class HistoryAnnualArtifactSource:
    """Content-addressed History 1.1 annual source for any rolling-origin season."""

    model_family: str
    spec_version: str
    target_season: int
    trained_through_season: int
    fitted_instance_identity_sha256: str
    model_metadata_sha256: str
    prediction_artifact_sha256: str
    prediction_semantic_sha256: str
    population: int
    team_ids_sha256: str
    team_rows_sha256: str
    prior_methods: tuple[str, ...]
    artifact_id: str
    provenance_class: str
    provenance_schema_version: int
    history_semantic_spec_identity_sha256: str | None
    training_input_source_identity_sha256: str | None
    _team_rows_json: str
    _authority_token: object
    source_identity_sha256: str

    def __init__(
        self,
        *,
        model_family: str,
        spec_version: str,
        target_season: int,
        trained_through_season: int,
        fitted_instance_identity_sha256: str,
        model_metadata_sha256: str,
        prediction_artifact_sha256: str,
        prediction_semantic_sha256: str,
        population: int,
        team_ids_sha256: str,
        team_rows_sha256: str,
        prior_methods: tuple[str, ...],
        artifact_id: str,
        provenance_class: str,
        provenance_schema_version: int,
        history_semantic_spec_identity_sha256: str | None,
        training_input_source_identity_sha256: str | None,
        team_rows_json: str,
        _construction_token: object = None,
    ) -> None:
        if _construction_token is not _HISTORY_ANNUAL_SOURCE_TOKEN:
            raise TypeError("History annual sources must come from the validated History loader")
        for name, value in (
            ("model_family", model_family),
            ("spec_version", spec_version),
            ("target_season", target_season),
            ("trained_through_season", trained_through_season),
            ("fitted_instance_identity_sha256", fitted_instance_identity_sha256),
            ("model_metadata_sha256", model_metadata_sha256),
            ("prediction_artifact_sha256", prediction_artifact_sha256),
            ("prediction_semantic_sha256", prediction_semantic_sha256),
            ("population", population),
            ("team_ids_sha256", team_ids_sha256),
            ("team_rows_sha256", team_rows_sha256),
            ("prior_methods", tuple(prior_methods)),
            ("artifact_id", artifact_id),
            ("provenance_class", provenance_class),
            ("provenance_schema_version", provenance_schema_version),
            ("history_semantic_spec_identity_sha256", history_semantic_spec_identity_sha256),
            ("training_input_source_identity_sha256", training_input_source_identity_sha256),
            ("_team_rows_json", team_rows_json),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "_authority_token",
            {
                "canonical_history_1_1_annual_output": _CANONICAL_HISTORY_AUTHORITY_TOKEN,
                "retained_legacy_history_artifact": _RETAINED_HISTORY_AUTHORITY_TOKEN,
                "research_history_fixture": _RESEARCH_HISTORY_AUTHORITY_TOKEN,
            }.get(provenance_class),
        )
        object.__setattr__(self, "source_identity_sha256", _sha256_json(self.identity_payload()))
        self.__post_init__()

    def __post_init__(self) -> None:
        if (
            self.model_family != "history_prior"
            or self.spec_version != "1.1"
            or self.trained_through_season != self.target_season - 1
            or self.target_season <= 1
            or self.artifact_id
            != f"gippyrank.history.annual_predictions.season_{self.target_season}"
            or self.provenance_class
            not in {
                "retained_legacy_history_artifact",
                "canonical_history_1_1_annual_output",
                "research_history_fixture",
            }
        ):
            raise ValueError("History annual source has an invalid rolling-origin identity")
        expected_schema = (
            _HISTORY_ANNUAL_LEGACY_SCHEMA_VERSION
            if self.provenance_class == "retained_legacy_history_artifact"
            else _HISTORY_ANNUAL_CANONICAL_SCHEMA_VERSION
        )
        if self.provenance_schema_version != expected_schema:
            raise ValueError("History annual provenance schema does not match its class")
        if (
            self.provenance_class == "retained_legacy_history_artifact"
            and self.target_season != 2026
        ) or (
            self.provenance_class in {
                "canonical_history_1_1_annual_output",
                "research_history_fixture",
            }
            and self.target_season <= 2026
        ):
            raise ValueError("History annual provenance class does not match its source season")
        if self.provenance_class == "canonical_history_1_1_annual_output":
            if self._authority_token is not _CANONICAL_HISTORY_AUTHORITY_TOKEN or not _is_sha256(self.history_semantic_spec_identity_sha256) or not _is_sha256(
                self.training_input_source_identity_sha256
            ):
                raise ValueError("canonical History output requires verified build lineage")
        elif self.provenance_class == "retained_legacy_history_artifact":
            if self._authority_token is not _RETAINED_HISTORY_AUTHORITY_TOKEN:
                raise ValueError("retained History authority does not match its class")
        elif self._authority_token is not _RESEARCH_HISTORY_AUTHORITY_TOKEN:
            raise ValueError("research History authority does not match its class")
        if self.provenance_class != "canonical_history_1_1_annual_output" and (
            self.history_semantic_spec_identity_sha256 is not None
            or self.training_input_source_identity_sha256 is not None
        ):
            raise ValueError("legacy and research History sources cannot claim canonical build lineage")
        for digest in (
            self.fitted_instance_identity_sha256,
            self.model_metadata_sha256,
            self.prediction_artifact_sha256,
            self.prediction_semantic_sha256,
            self.team_ids_sha256,
            self.team_rows_sha256,
            self.source_identity_sha256,
        ):
            if not _is_sha256(digest):
                raise ValueError("History annual source hashes must be SHA-256")
        try:
            rows = json.loads(self._team_rows_json)
        except json.JSONDecodeError as error:
            raise ValueError("History annual source team rows are invalid") from error
        if (
            not isinstance(rows, dict)
            or not rows
            or len(rows) != self.population
            or tuple(sorted({str(row.get("prior_method")) for row in rows.values()}))
            != self.prior_methods
            or _sha256_json(sorted(rows)) != self.team_ids_sha256
            or _sha256_json([rows[key] for key in sorted(rows)]) != self.team_rows_sha256
        ):
            raise ValueError("History annual source requires team rows")
        if _sha256_json(self.identity_payload()) != self.source_identity_sha256:
            raise ValueError("History annual source identity hash is inconsistent")

    def identity_payload(self) -> dict[str, object]:
        return {
            "model_family": self.model_family,
            "spec_version": self.spec_version,
            "target_season": self.target_season,
            "trained_through_season": self.trained_through_season,
            "fitted_instance_identity_sha256": self.fitted_instance_identity_sha256,
            "model_metadata_sha256": self.model_metadata_sha256,
            "prediction_artifact_sha256": self.prediction_artifact_sha256,
            "prediction_semantic_sha256": self.prediction_semantic_sha256,
            "population": self.population,
            "team_ids_sha256": self.team_ids_sha256,
            "team_rows_sha256": self.team_rows_sha256,
            "prior_methods": list(self.prior_methods),
            "artifact_id": self.artifact_id,
            "provenance_class": self.provenance_class,
            "provenance_schema_version": self.provenance_schema_version,
            **(
                {
                    "history_semantic_spec_identity_sha256": self.history_semantic_spec_identity_sha256,
                    "training_input_source_identity_sha256": self.training_input_source_identity_sha256,
                }
                if self.provenance_class == "canonical_history_1_1_annual_output"
                else {}
            ),
        }

    def to_metadata(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "source_identity_sha256": self.source_identity_sha256,
        }

    def prediction_row(self, team_id: str) -> dict[str, object]:
        rows = json.loads(self._team_rows_json)
        try:
            return dict(rows[team_id])
        except KeyError as error:
            raise ValueError("History annual source has no row for the requested team") from error

    def cold_start_prediction(
        self,
        *,
        target_season: int,
        trained_through_season: int,
        team_id: str,
        team_name: str,
        population: int,
        cold_start_reason: str,
    ) -> dict[str, object]:
        """Validate and select a production-shaped History cold-start row."""
        self.__post_init__()
        if (
            self.provenance_class
            not in {"retained_legacy_history_artifact", "canonical_history_1_1_annual_output"}
            or self.target_season != target_season
            or self.trained_through_season != trained_through_season
            or trained_through_season != target_season - 1
            or self.population != population
        ):
            raise ValueError("History source does not match the requested rolling-origin identity")
        row = self.prediction_row(team_id)
        if row.get("team_name") != team_name:
            raise ValueError("History source team identity does not match the requested team")
        required_method = {
            "fcs_to_fbs_transition": "learned_fcs_to_fbs_transition",
            "no_prior_rank_distribution": "generic_fbs_cold_start",
        }.get(cold_start_reason)
        if required_method is None or row.get("prior_method") != required_method:
            raise ValueError("History fallback method does not match its cold-start reason")
        self._validated_prediction_pmf(row, population)
        return row

    @staticmethod
    def _validated_prediction_pmf(
        row: dict[str, object], population: int
    ) -> np.ndarray:
        try:
            pmf = np.asarray(row["pmf"], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("History source row has an invalid PMF") from error
        _validate_history_pmf(pmf, population)
        return pmf


def _validate_history_1_1_model_metadata(model: Mapping[str, object]) -> None:
    preprocessing = model.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise TypeError("History 1.1 model preprocessing is missing")
    if (
        model.get("family") != HISTORY_1_1_FAMILY
        or model.get("feature_names") != list(HISTORY_1_1_FEATURES)
        or model.get("lag_count") != HISTORY_1_1_LAG_COUNT
        or model.get("degrees_of_freedom") != HISTORY_1_1_DEGREES_OF_FREEDOM
        or model.get("penalty") != HISTORY_1_1_PENALTY
        or model.get("minimum_scale") != HISTORY_1_1_MINIMUM_SCALE
        or model.get("quadrature_points") != fit_semantics.QUADRATURE_POINTS
        or model.get("quadrature_method")
        != fit_semantics.DETERMINISTIC_QUADRATURE_METHOD
        or preprocessing.get("feature_names") != list(HISTORY_1_1_FEATURES)
    ):
        raise ValueError("History model identity does not match the frozen 1.1 contract")
    for key in ("medians", "means", "scales"):
        values = preprocessing.get(key)
        if not isinstance(values, dict) or set(values) != set(HISTORY_1_1_FEATURES):
            raise ValueError("History 1.1 preprocessing does not match its model features")
        try:
            numeric = np.asarray([values[name] for name in HISTORY_1_1_FEATURES], dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("History 1.1 preprocessing values are invalid") from error
        if not np.isfinite(numeric).all() or (key == "scales" and np.any(numeric <= 0)):
            raise ValueError("History 1.1 preprocessing values are invalid")
    try:
        beta = np.asarray(model.get("location_coefficients"), dtype=float)
        gamma = np.asarray(model.get("log_scale_coefficients"), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("History 1.1 model coefficients are invalid") from error
    if beta.shape != (8,) or gamma.shape != (7,) or not np.isfinite(beta).all() or not np.isfinite(gamma).all():
        raise ValueError("History 1.1 model coefficients do not match its frozen layout")
    optimizer = model.get("optimizer")
    if not isinstance(optimizer, dict) or optimizer.get("success") is not True:
        raise ValueError("History 1.1 model was not successfully fitted")


def _history_fitted_instance_identity(
    fitted_instance: Mapping[str, object], model_sha256: str
) -> str:
    return _sha256_json(
        {
            "model_family": fitted_instance.get("model_family"),
            "spec_version": fitted_instance.get("spec_version"),
            "target_season": fitted_instance.get("target_season"),
            "trained_through_season": fitted_instance.get("trained_through_season"),
            "context_effective_cutoff": fitted_instance.get("context_effective_cutoff"),
            "model_metadata_sha256": model_sha256,
        }
    )


def _history_annual_source_from_files(
    prediction_artifact_path: str | Path,
    fitted_instance_path: str | Path,
    *,
    target_season: int,
    trained_through_season: int,
    provenance_class: str,
) -> HistoryAnnualArtifactSource:
    history_semantic_spec_identity_sha256: str | None = None
    training_input_source_identity_sha256: str | None = None
    if provenance_class == "canonical_history_1_1_annual_output":
        expected_predictions, expected_instance, inputs = reproduce_canonical_history_annual(
            target_season
        )
        try:
            actual_predictions = Path(prediction_artifact_path).read_bytes()
            actual_instance = json.loads(
                Path(fitted_instance_path).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("future History annual outputs are unreadable") from error
        if actual_predictions != expected_predictions or actual_instance != json.loads(
            json.dumps(expected_instance)
        ):
            raise ValueError("History annual outputs differ from the canonical History build")
        history_semantic_spec_identity_sha256 = history11_semantic_specification_sha256()
        training_input_source_identity_sha256 = inputs.source_identity_sha256
    try:
        prediction_bytes = Path(prediction_artifact_path).read_bytes()
        fitted_instance = json.loads(Path(fitted_instance_path).read_text(encoding="utf-8"))
        reader = csv.DictReader(io.StringIO(prediction_bytes.decode("utf-8"), newline=""))
        prediction_rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error, json.JSONDecodeError) as error:
        raise ValueError("History annual artifacts are unreadable") from error
    required_columns = {
        "season",
        "subdivision",
        "team_id",
        "team_name",
        "model_family",
        "spec_version",
        "trained_through_season",
        "pmf",
        "prior_method",
    }
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("History annual prediction artifact has invalid columns")
    if not required_columns <= set(reader.fieldnames):
        raise ValueError("History annual prediction artifact is missing required columns")
    if not isinstance(fitted_instance, dict) or not isinstance(
        fitted_instance.get("model"), dict
    ):
        raise TypeError("History annual fitted-instance artifact is invalid")
    model = fitted_instance["model"]
    _validate_history_1_1_model_metadata(model)
    model_sha256 = _sha256_json(model)
    if (
        fitted_instance.get("model_family") != "history_prior"
        or fitted_instance.get("spec_version") != "1.1"
        or fitted_instance.get("target_season") != target_season
        or fitted_instance.get("trained_through_season") != trained_through_season
        or fitted_instance.get("context_effective_cutoff") is not None
    ):
        raise ValueError("History fitted instance does not match the requested 1.1 fit")
    prediction_rows.sort(
        key=lambda row: (row.get("season"), row.get("subdivision"), row.get("team_id"))
    )
    semantic_sha256 = _sha256_json(prediction_rows)
    by_team: dict[str, dict[str, object]] = {}
    population = len(prediction_rows)
    if population < 1:
        raise ValueError("History annual predictions must contain the target FBS population")
    for row in prediction_rows:
        if (
            row.get("season") != str(target_season)
            or row.get("subdivision") != "fbs"
            or row.get("model_family") != "history_prior"
            or row.get("spec_version") != "1.1"
            or row.get("trained_through_season") != str(trained_through_season)
        ):
            raise ValueError("History prediction row has mismatched model or season identity")
        team_id = str(row.get("team_id", ""))
        team_name = str(row.get("team_name", ""))
        method = str(row.get("prior_method", ""))
        if not team_id or not team_name or team_id in by_team:
            raise ValueError("History annual predictions contain duplicate or empty team IDs")
        if method not in _HISTORY_1_1_PRIOR_METHODS:
            raise ValueError("History annual prediction has an unknown prior method")
        try:
            pmf = np.asarray(json.loads(str(row["pmf"])), dtype=float)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("History prediction row contains an invalid PMF") from error
        _validate_history_pmf(pmf, population)
        by_team[team_id] = {
            "team_id": team_id,
            "team_name": team_name,
            "prior_method": method,
            "pmf": [float(value) for value in pmf],
        }
    if target_season == 2026:
        if (
            provenance_class != "retained_legacy_history_artifact"
            or population != 138
            or model_sha256 != _HISTORY_2026_MODEL_METADATA_SHA256
            or semantic_sha256 != _HISTORY_2026_PREDICTIONS_SEMANTIC_SHA256
        ):
            raise ValueError("History predictions differ from the pinned 2026 annual artifact")
    elif provenance_class not in {
        "canonical_history_1_1_annual_output",
        "research_history_fixture",
    }:
        raise ValueError("future History annual output provenance is invalid")
    team_ids = sorted(by_team)
    team_rows = [by_team[team_id] for team_id in team_ids]
    return HistoryAnnualArtifactSource(
        model_family="history_prior",
        spec_version="1.1",
        target_season=target_season,
        trained_through_season=trained_through_season,
        fitted_instance_identity_sha256=_history_fitted_instance_identity(
            fitted_instance, model_sha256
        ),
        model_metadata_sha256=model_sha256,
        prediction_artifact_sha256=hashlib.sha256(prediction_bytes).hexdigest(),
        prediction_semantic_sha256=semantic_sha256,
        population=population,
        team_ids_sha256=_sha256_json(team_ids),
        team_rows_sha256=_sha256_json(team_rows),
        prior_methods=tuple(sorted({str(row["prior_method"]) for row in team_rows})),
        artifact_id=f"gippyrank.history.annual_predictions.season_{target_season}",
        provenance_class=provenance_class,
        provenance_schema_version=(
            _HISTORY_ANNUAL_LEGACY_SCHEMA_VERSION
            if provenance_class == "retained_legacy_history_artifact"
            else _HISTORY_ANNUAL_CANONICAL_SCHEMA_VERSION
        ),
        history_semantic_spec_identity_sha256=history_semantic_spec_identity_sha256,
        training_input_source_identity_sha256=training_input_source_identity_sha256,
        team_rows_json=json.dumps(by_team, sort_keys=True, separators=(",", ":")),
        _construction_token=_HISTORY_ANNUAL_SOURCE_TOKEN,
    )


def load_research_history_annual_fixture(
    prediction_artifact_path: str | Path,
    fitted_instance_path: str | Path,
    *,
    target_season: int,
    trained_through_season: int,
) -> HistoryAnnualArtifactSource:
    """Inspect synthetic History files as a non-promotable research fixture."""
    if target_season <= 2026 or trained_through_season != target_season - 1:
        raise ValueError("research History fixture requires a future T-1 season")
    return _history_annual_source_from_files(
        prediction_artifact_path,
        fitted_instance_path,
        target_season=target_season,
        trained_through_season=trained_through_season,
        provenance_class="research_history_fixture",
    )


def load_validated_history_annual_artifact(
    prediction_artifact_path: str | Path,
    fitted_instance_path: str | Path,
    *,
    target_season: int,
    trained_through_season: int,
    source_attestation_path: str | Path | None = None,
) -> HistoryAnnualArtifactSource:
    """Load pinned 2026 History or reproduce a future canonical annual build."""
    if target_season <= 1 or trained_through_season != target_season - 1:
        raise ValueError("History annual source must use the rolling-origin T-1 cutoff")
    if target_season == 2026:
        return _history_annual_source_from_files(
            prediction_artifact_path,
            fitted_instance_path,
            target_season=target_season,
            trained_through_season=trained_through_season,
            provenance_class="retained_legacy_history_artifact",
        )
    source = _history_annual_source_from_files(
        prediction_artifact_path,
        fitted_instance_path,
        target_season=target_season,
        trained_through_season=trained_through_season,
        provenance_class="canonical_history_1_1_annual_output",
    )
    sidecar_path = (
        Path(source_attestation_path)
        if source_attestation_path is not None
        else Path(fitted_instance_path).with_name("fitted_model_source.json")
    )
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("future History annual source attestation is unreadable") from error
    if sidecar != source.to_metadata():
        raise ValueError("History annual outputs differ from their canonical source attestation")
    return source


__all__ = [
    "History11AnnualBuildInputSource",
    "HistoryAnnualArtifactSource",
    "build_canonical_history_annual",
    "history11_semantic_specification",
    "history11_semantic_specification_sha256",
    "load_canonical_history_annual_build_inputs",
    "load_research_history_annual_fixture",
    "load_validated_history_annual_artifact",
    "reproduce_canonical_history_annual",
]
