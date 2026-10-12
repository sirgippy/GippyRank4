"""Shared mechanics for issue 200's matched History/Context pipelines.

Response targets are kept as a separate input from the final-rank predictor
rows. The four arms therefore share identical rank lags and preseason feature
values while training against the selected FBS response policy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Literal, Protocol

import numpy as np

from gippyrank import preseason
from gippyrank.context_db_repair import MODEL_FEATURES
from gippyrank.context_db_repair import fit_model as fit_context14
from gippyrank.context_db_repair import production_pmf as context14_pmf
from gippyrank.context_prior import InferenceRow
from gippyrank.history_annual_v1_1 import (
    HISTORY_1_1_DEGREES_OF_FREEDOM,
    HISTORY_1_1_FAMILY,
    HISTORY_1_1_FEATURES,
    HISTORY_1_1_LAG_COUNT,
    HISTORY_1_1_MINIMUM_SCALE,
    HISTORY_1_1_PENALTY,
    HISTORY_1_1_ROW_WEIGHT,
)
from gippyrank.modeling import validate_distribution_row
from gippyrank.preseason import DirectRankModel, GenericRankPrior, TeamSeason, rank_to_z

Policy = Literal["existing", "pre_bowl"]
TeamKey = tuple[int, str, str]


class ColdStartRow(Protocol):
    season: int
    subdivision: str
    team_id: str
    team_name: str
    population: int
    target_z: np.ndarray
    target_ranks: np.ndarray
    cross_subdivision_lag_z: np.ndarray | None
    reason: str


@dataclass(frozen=True)
class ResponseTarget:
    """One validated constituent-rank response distribution."""

    season: int
    subdivision: str
    team_id: str
    team_name: str
    population: int
    ranks: tuple[int, ...]
    rank_z: tuple[float, ...]
    row_sha256: str

    @property
    def key(self) -> TeamKey:
        return (self.season, self.subdivision, self.team_id)


@dataclass(frozen=True)
class HistoryColdStartModels:
    transition: DirectRankModel | None
    generic: GenericRankPrior | None


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_response_targets(
    rows: list[dict[str, str]], *, label: str
) -> dict[TeamKey, ResponseTarget]:
    """Validate and index a corrected target corpus by canonical identity."""
    result: dict[TeamKey, ResponseTarget] = {}
    for row in rows:
        validate_distribution_row(row)
        key = (int(row["season"]), row["subdivision"], row["team_id"])
        if key in result:
            raise ValueError(f"{label}: duplicate response target {key}")
        ranks = tuple(int(value) for value in json.loads(row["rank_observations"]))
        population = int(row["team_population"])
        result[key] = ResponseTarget(
            season=key[0],
            subdivision=key[1],
            team_id=key[2],
            team_name=row["team_name"],
            population=population,
            ranks=ranks,
            rank_z=tuple(
                float(value) for value in rank_to_z(np.asarray(ranks), population)
            ),
            row_sha256=sha256_json(row),
        )
    return result


def validate_fcs_targets_unchanged(
    existing: dict[TeamKey, ResponseTarget], pre_bowl: dict[TeamKey, ResponseTarget]
) -> None:
    """Require the agreed final FCS response corpus in both policies."""
    final_fcs = {key: value for key, value in existing.items() if key[1] == "fcs"}
    pre_fcs = {key: value for key, value in pre_bowl.items() if key[1] == "fcs"}
    if set(final_fcs) != set(pre_fcs):
        raise ValueError("FCS response team-seasons differ between corrected corpora")
    for key, final in final_fcs.items():
        pre = pre_fcs[key]
        if (
            final.population != pre.population
            or final.ranks != pre.ranks
            or final.row_sha256 != pre.row_sha256
        ):
            raise ValueError(f"{key}: FCS response policy changed between corpora")


def common_fbs_keys(
    existing: dict[TeamKey, ResponseTarget], pre_bowl: dict[TeamKey, ResponseTarget]
) -> set[TeamKey]:
    """Return matched FBS response opportunities across target policies."""
    final_keys = {key for key in existing if key[1] == "fbs"}
    pre_keys = {key for key in pre_bowl if key[1] == "fbs"}
    return final_keys & pre_keys


def attach_repaired_context_rows(
    rows: list[TeamSeason],
    repaired_features: dict[tuple[int, str], dict[str, float | None]],
) -> tuple[list[TeamSeason], list[tuple[int, str]]]:
    """Attach Context 1.4 transfer fields and preserve uncovered rows as missing."""
    attached = []
    missing: list[tuple[int, str]] = []
    for row in rows:
        key = (row.season, row.team_id)
        features = {name: row.features.get(name) for name in MODEL_FEATURES}
        repaired = repaired_features.get(key)
        if repaired is None:
            # Missing source coverage is not evidence of a zero transfer impact.
            for name in (
                "transfer_in_prior_usage_sum",
                "observed_db_impact_sum",
                "db_impact_coverage_fraction",
            ):
                features[name] = None
            missing.append(key)
        else:
            features.update(repaired)
        attached.append(replace(row, features=features))
    return attached, missing


def predictor_payload(row: TeamSeason) -> dict[str, object]:
    """Values used to predict the next rank distribution, excluding responses."""
    return {
        "season": row.season,
        "subdivision": row.subdivision,
        "team_id": row.team_id,
        "team_name": row.team_name,
        "lag1_z": np.asarray(row.lag1_z, dtype=float).tolist(),
        "lag_zs": [np.asarray(value, dtype=float).tolist() for value in row.lag_zs],
        "features": row.features,
    }


def inference_payload(row: InferenceRow) -> dict[str, object]:
    """Hashable outcome-free prediction row used by both policy arms."""
    row.require_no_target()
    return {
        "season": row.season,
        "subdivision": row.subdivision,
        "team_id": row.team_id,
        "team_name": row.team_name,
        "lag1_z": row.lag1_z,
        "lag_zs": row.lag_zs,
        "features": row.features,
        "cold_start_reason": row.cold_start_reason,
    }


def apply_fbs_response_targets(
    rows: list[TeamSeason],
    targets: dict[TeamKey, ResponseTarget],
    *,
    matched_keys: set[TeamKey],
) -> list[TeamSeason]:
    """Substitute FBS response labels while preserving final-rank predictors."""
    result = []
    for row in rows:
        if row.subdivision != "fbs":
            continue
        key = (row.season, row.subdivision, row.team_id)
        if key not in matched_keys:
            continue
        target = targets.get(key)
        if target is None:
            raise ValueError(f"matched FBS predictor row lacks a response: {key}")
        before = sha256_json(predictor_payload(row))
        updated = replace(
            row,
            population=target.population,
            target_z=np.asarray(target.rank_z, dtype=float),
            target_ranks=np.asarray(target.ranks, dtype=int),
        )
        if sha256_json(predictor_payload(updated)) != before:
            raise AssertionError(f"{key}: response substitution changed predictors")
        result.append(updated)
    return result


def apply_fbs_cold_response_targets(
    rows: list[ColdStartRow],
    targets: dict[TeamKey, ResponseTarget],
    *,
    matched_keys: set[TeamKey],
) -> list[ColdStartRow]:
    """Substitute response arrays in the History cold-start training rows."""
    result = []
    for row in rows:
        if row.subdivision != "fbs":
            continue
        key = (int(row.season), "fbs", str(row.team_id))
        if key not in matched_keys:
            continue
        target = targets.get(key)
        if target is None:
            raise ValueError(f"matched FBS cold-start row lacks a response: {key}")
        result.append(
            replace(
                row,
                population=target.population,
                target_z=np.asarray(target.rank_z, dtype=float),
                target_ranks=np.asarray(target.ranks, dtype=int),
            )
        )
    return result


def fit_history11(rows: list[TeamSeason]) -> DirectRankModel:
    """Fit the verified rolling History 1.1 recipe on the supplied window."""
    if not rows:
        raise ValueError("History 1.1 cannot fit an empty training window")
    return DirectRankModel.fit(
        rows,
        list(HISTORY_1_1_FEATURES),
        penalty=HISTORY_1_1_PENALTY,
        minimum_scale=HISTORY_1_1_MINIMUM_SCALE,
        lag_count=HISTORY_1_1_LAG_COUNT,
        family=HISTORY_1_1_FAMILY,
        degrees_of_freedom=HISTORY_1_1_DEGREES_OF_FREEDOM,
        row_weights=np.full(len(rows), HISTORY_1_1_ROW_WEIGHT),
        optimizer_options={
            "maxiter": preseason.DIRECT_RANK_OPTIMIZER_MAXITER,
            "ftol": preseason.DIRECT_RANK_OPTIMIZER_FTOL,
            "gtol": preseason.DIRECT_RANK_OPTIMIZER_GTOL,
        },
        preprocessor_scale_floor=preseason.PREPROCESSOR_SCALE_FLOOR,
        preprocessor_std_ddof=preseason.PREPROCESSOR_STD_DDOF,
    )


def fit_history_cold_starts(
    rows: list[ColdStartRow], *, trained_through_season: int
) -> HistoryColdStartModels:
    """Fit the retained transition and no-prior History fallbacks."""
    eligible = [
        row
        for row in rows
        if int(row.season) <= trained_through_season and row.subdivision == "fbs"
    ]
    transition_rows = []
    generic_rows = []
    for row in eligible:
        key = (int(row.season), "fbs", str(row.team_id))
        lag = row.cross_subdivision_lag_z
        if row.reason == "fcs_to_fbs_transition" and lag is not None:
            transition_rows.append(
                TeamSeason(
                    season=key[0],
                    subdivision="fbs",
                    team_id=key[2],
                    team_name=str(row.team_name),
                    population=int(row.population),
                    lag1_z=np.asarray(lag, dtype=float),
                    target_z=np.asarray(row.target_z, dtype=float),
                    target_ranks=np.asarray(row.target_ranks, dtype=int),
                    features={},
                )
            )
        elif row.reason == "no_prior_rank_distribution":
            generic_rows.append(
                TeamSeason(
                    season=key[0],
                    subdivision="fbs",
                    team_id=key[2],
                    team_name=str(row.team_name),
                    population=int(row.population),
                    lag1_z=np.asarray([0.0]),
                    target_z=np.asarray(row.target_z, dtype=float),
                    target_ranks=np.asarray(row.target_ranks, dtype=int),
                    features={},
                )
            )
    transition = (
        DirectRankModel.fit(
            transition_rows,
            [],
            penalty=HISTORY_1_1_PENALTY,
            minimum_scale=HISTORY_1_1_MINIMUM_SCALE,
            lag_count=HISTORY_1_1_LAG_COUNT,
            family=HISTORY_1_1_FAMILY,
            degrees_of_freedom=HISTORY_1_1_DEGREES_OF_FREEDOM,
            row_weights=np.full(len(transition_rows), HISTORY_1_1_ROW_WEIGHT),
            optimizer_options={
                "maxiter": preseason.DIRECT_RANK_OPTIMIZER_MAXITER,
                "ftol": preseason.DIRECT_RANK_OPTIMIZER_FTOL,
                "gtol": preseason.DIRECT_RANK_OPTIMIZER_GTOL,
            },
            preprocessor_scale_floor=preseason.PREPROCESSOR_SCALE_FLOOR,
            preprocessor_std_ddof=preseason.PREPROCESSOR_STD_DDOF,
        )
        if transition_rows
        else None
    )
    generic = (
        GenericRankPrior.fit(generic_rows, minimum_scale=HISTORY_1_1_MINIMUM_SCALE)
        if generic_rows
        else None
    )
    return HistoryColdStartModels(transition, generic)


def fit_context14_rolling(
    rows: list[TeamSeason], *, trained_through_season: int
) -> DirectRankModel:
    """Fit repaired Context 1.4 with the existing optimizer and retry rule."""
    return fit_context14(rows, trained_through_season)


def predict_history11(
    model: DirectRankModel,
    cold_models: HistoryColdStartModels,
    row: InferenceRow,
    *,
    cross_subdivision_lag_z: np.ndarray | None,
) -> tuple[np.ndarray, str, float, float]:
    """Predict one FBS PMF using the History 1.1 cutoff and cold-start rules."""
    row.require_no_target()
    if row.lag1_z is not None:
        lag = np.asarray(row.lag1_z, dtype=float)
        pmf = model.pmf(row.features, lag, row.population)
        locations, scale = model.conditional_parameters(row.features, lag)
        return pmf, "same_subdivision_lag1", float(np.mean(locations)), scale
    if row.cold_start_reason == "fcs_to_fbs_transition":
        if cold_models.transition is None or cross_subdivision_lag_z is None:
            raise ValueError(
                f"{row.team_id}: History transition fallback is unavailable"
            )
        pmf = cold_models.transition.pmf({}, cross_subdivision_lag_z, row.population)
        locations, scale = cold_models.transition.conditional_parameters(
            {}, cross_subdivision_lag_z
        )
        return (
            pmf,
            "learned_fcs_to_fbs_transition",
            float(np.mean(locations)),
            scale,
        )
    if row.cold_start_reason == "no_prior_rank_distribution":
        if cold_models.generic is None:
            raise ValueError(f"{row.team_id}: History generic fallback is unavailable")
        pmf = cold_models.generic.pmf(row.population)
        return (
            pmf,
            "generic_fbs_cold_start",
            cold_models.generic.location,
            cold_models.generic.scale,
        )
    raise ValueError(f"{row.team_id}: unknown History cold-start state")


def predict_context14(
    model: DirectRankModel,
    row: InferenceRow,
    *,
    history_pmf: np.ndarray,
) -> tuple[np.ndarray, str, float | None, float | None]:
    """Predict Context 1.4, preserving the matched History fallback."""
    row.require_no_target()
    if row.lag1_z is None:
        return history_pmf.copy(), "history_cold_start", None, None
    if set(row.features) != set(MODEL_FEATURES):
        raise ValueError(f"{row.team_id}: Context 1.4 feature contract differs")
    pmf, location, scale = context14_pmf(
        model, row.features, np.asarray(row.lag1_z, dtype=float), row.population
    )
    return pmf, "same_subdivision_lag1", location, scale


def validate_prior_pmf(pmf: np.ndarray, population: int, *, context: str) -> None:
    values = np.asarray(pmf, dtype=float)
    if (
        values.ndim != 1
        or len(values) != population
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-10)
    ):
        raise ValueError(f"{context}: PMF is not normalized on support 1..{population}")
