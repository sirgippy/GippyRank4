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
HISTORY_1_1_PENALTY = 0.25
HISTORY_1_1_MINIMUM_SCALE = 0.10
HISTORY_1_1_LAG_COUNT = 1
HISTORY_1_1_FAMILY = "normal"
HISTORY_1_1_DEGREES_OF_FREEDOM = None
_INPUT_TOKEN = object()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def history11_semantic_specification() -> dict[str, object]:
    """The annual History fit, input, and fallback construction contract."""
    return {
        "model_family": "history_prior",
        "spec_version": "1.1",
        "annual_build_semantics_version": 1,
        "ordered_features": list(HISTORY_1_1_FEATURES),
        "distribution_family": HISTORY_1_1_FAMILY,
        "degrees_of_freedom": HISTORY_1_1_DEGREES_OF_FREEDOM,
        "lag_count": HISTORY_1_1_LAG_COUNT,
        "penalty": HISTORY_1_1_PENALTY,
        "minimum_scale": HISTORY_1_1_MINIMUM_SCALE,
        "rolling_origin_cutoff": "trained_through_season == target_season - 1",
        "quadrature_points": fit_semantics.QUADRATURE_POINTS,
        "team_season_weight": 1.0,
        "preprocessing": {
            "missing_value_imputation": "training median, or zero when all missing",
            "standardization": "training mean and population standard deviation",
            "scale_floor": fit_semantics.PREPROCESSOR_SCALE_FLOOR,
            "std_ddof": fit_semantics.PREPROCESSOR_STD_DDOF,
            "missingness_indicators": True,
        },
        "optimizer": {
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
        },
        "training": "all eligible FBS outcome rows through target season minus one",
        "historical_features": "same-subdivision prior ranks and completed rank history",
        "target_population": "exact FBS rows in the target team-season feature artifact",
        "target_outcomes": "forbidden in annual build inputs",
        "input_retention": "copy exact canonical input bytes into the annual artifact for later reproduction",
        "cold_start": {
            "fcs_transition": "fit direct rank model from historical FCS-to-FBS outcomes",
            "no_prior": "generic rank prior from historical FBS cold starts",
        },
        "pmf_serialization": "round each rank probability to 12 decimal places",
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
            row_weights=np.ones(len(rows)),
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
    promotion_model = fit(promotion, []) if needs_promotion else None
    generic_prior = (
        GenericRankPrior.fit(generic, minimum_scale=HISTORY_1_1_MINIMUM_SCALE)
        if needs_generic
        else None
    )
    predictions: list[dict[str, object]] = []
    for row in target:
        if row.lag1_z is not None:
            pmf = model.pmf(row.features, row.lag1_z, row.population)
            locations, scale = model.conditional_parameters(row.features, row.lag1_z)
            method = "same_subdivision_lag1"
        elif row.cold_start_reason == "fcs_to_fbs_transition":
            assert (
                promotion_model is not None and row.cross_subdivision_lag_z is not None
            )
            pmf = promotion_model.pmf({}, row.cross_subdivision_lag_z, row.population)
            locations, scale = promotion_model.conditional_parameters(
                {}, row.cross_subdivision_lag_z
            )
            method = "learned_fcs_to_fbs_transition"
        else:
            assert generic_prior is not None
            pmf = generic_prior.pmf(row.population)
            locations = np.asarray([generic_prior.location])
            scale = generic_prior.scale
            method = "generic_fbs_cold_start"
        predictions.append(
            {
                "season": target_season,
                "subdivision": "fbs",
                "team_id": row.team_id,
                "team_name": row.team_name,
                "model_family": "history_prior",
                "spec_version": "1.1",
                "trained_through_season": target_season - 1,
                "pmf": json.dumps(
                    [round(float(value), 12) for value in pmf], separators=(",", ":")
                ),
                "prior_method": method,
                "conditional_location_mean": float(np.mean(locations)),
                "predictive_scale": scale,
            }
        )
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
    from gippyrank.context_prior_v1_4_candidate import _history_annual_source_from_files

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
