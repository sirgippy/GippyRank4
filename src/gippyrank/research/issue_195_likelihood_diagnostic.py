"""Ex-post likelihood adequacy diagnostic for issue 195.

The runner conditions the frozen issue-200 existing-policy rank-only Student-t
likelihood on corrected final-rank pair measurements. It does not produce an
operational forecast or alter production inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp
from scipy.stats import t as student_t

from gippyrank.context_db_repair import MODEL_FEATURES
from gippyrank.history_annual_v1_1 import HISTORY_1_1_FEATURES
from gippyrank.modeling import design_matrix

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_historical_modeling as historical_builder

TARGET_SEASONS = (2023, 2024, 2025)
RESIDUAL_SEASONS = (2022, 2023, 2024)
MODEL_ARMS = (
    "History_existing",
    "Context_existing",
    "History_pre_bowl",
    "Context_pre_bowl",
)
REGULARIZATION_GRID = (0.01, 0.1, 1.0)
DEFAULT_REGULARIZATION = 0.1
CONTEXT_FEATURES = tuple(
    name for name in MODEL_FEATURES if name not in HISTORY_1_1_FEATURES
)
MISSINGNESS_FEATURES = tuple(f"missing__{name}" for name in CONTEXT_FEATURES)
DIAGNOSTIC_ARMS = ("rank_only", "rank_site_calibration", "rank_site_context")
COMPARISONS = (
    ("rank_site_calibration_minus_rank_only", "rank_site_calibration", "rank_only"),
    (
        "rank_site_context_minus_rank_site_calibration",
        "rank_site_context",
        "rank_site_calibration",
    ),
)
STUDENT_T_DF = 15.0
OPTIMIZER_OPTIONS = {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 50}
PARAMETER_BOUNDS = ((-4.0, 4.0), (-100.0, 100.0), (-3.0, 3.0))


@dataclass(frozen=True)
class ContextPreprocessor:
    """Training-only imputation, scaling, and side-difference transform."""

    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    missing_means: np.ndarray
    missing_scales: np.ndarray

    def transform(self, home: np.ndarray, away: np.ndarray) -> np.ndarray:
        home_values = np.asarray(home, dtype=float)
        away_values = np.asarray(away, dtype=float)
        if home_values.shape != away_values.shape:
            raise ValueError("home and away Context feature shapes differ")
        if home_values.ndim != 2 or home_values.shape[1] != len(self.feature_names):
            raise ValueError("Context feature matrix has the wrong schema")
        home_missing = ~np.isfinite(home_values)
        away_missing = ~np.isfinite(away_values)
        home_filled = np.where(home_missing, self.means, home_values)
        away_filled = np.where(away_missing, self.means, away_values)
        home_z = (home_filled - self.means) / self.scales
        away_z = (away_filled - self.means) / self.scales
        home_missing_z = (
            home_missing.astype(float) - self.missing_means
        ) / self.missing_scales
        away_missing_z = (
            away_missing.astype(float) - self.missing_means
        ) / self.missing_scales
        return np.column_stack((home_z - away_z, home_missing_z - away_missing_z))

    def metadata(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "value_means": self.means.tolist(),
            "value_scales": self.scales.tolist(),
            "missingness_means": self.missing_means.tolist(),
            "missingness_scales": self.missing_scales.tolist(),
            "fit_population": "both team sides of training games only",
        }


@dataclass(frozen=True)
class LikelihoodData:
    """Game-level outcomes and complete per-game rank-pair mixture locations."""

    game_ids: tuple[str, ...]
    seasons: np.ndarray
    margins: np.ndarray
    home_field: np.ndarray
    pair_game: np.ndarray
    base_locations: np.ndarray
    context_home: np.ndarray
    context_away: np.ndarray
    metadata: tuple[dict[str, object], ...]

    @property
    def game_count(self) -> int:
        return len(self.game_ids)


@dataclass(frozen=True)
class LikelihoodSlice:
    margins: np.ndarray
    home_field: np.ndarray
    pair_game: np.ndarray
    base_locations: np.ndarray
    context: np.ndarray


@dataclass(frozen=True)
class MixtureEvaluation:
    rows: tuple[dict[str, object], ...]
    mean_nll: float


@dataclass(frozen=True)
class FitResult:
    log_multiplier: float
    home_field_adjustment: float
    log_scale_correction: float
    context_coefficients: np.ndarray
    scale: float
    objective: float
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_iterations: int
    optimizer_evaluations: int
    gradient_norm: float
    regularization: float

    @property
    def multiplier(self) -> float:
        return float(np.exp(self.log_multiplier))

    def metadata(self, feature_names: Sequence[str]) -> dict[str, object]:
        return {
            "overall_mean_multiplier": self.multiplier,
            "additional_non_neutral_home_field_points": self.home_field_adjustment,
            "common_log_scale_correction": self.log_scale_correction,
            "scale": self.scale,
            "context_coefficients": {
                name: float(value)
                for name, value in zip(
                    feature_names, self.context_coefficients, strict=True
                )
            },
            "context_ridge_penalty": self.regularization,
            "objective_sum_game_nll_plus_half_ridge_norm": self.objective,
            "optimizer": {
                "method": "L-BFGS-B with analytic gradient",
                "success": self.optimizer_success,
                "status": self.optimizer_status,
                "message": self.optimizer_message,
                "iterations": self.optimizer_iterations,
                "evaluations": self.optimizer_evaluations,
                "final_gradient_infinity_norm": self.gradient_norm,
                "gradient_tolerance": OPTIMIZER_OPTIONS["gtol"],
                "parameter_bounds": {
                    "log_multiplier": list(PARAMETER_BOUNDS[0]),
                    "home_field_adjustment": list(PARAMETER_BOUNDS[1]),
                    "log_scale_correction": list(PARAMETER_BOUNDS[2]),
                    "context_coefficients": "unbounded; L2 regularized",
                },
            },
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path, fields: Sequence[str], rows: Iterable[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def fit_context_preprocessor(
    home: np.ndarray,
    away: np.ndarray,
    train_indices: np.ndarray,
    feature_names: Sequence[str],
) -> ContextPreprocessor:
    """Fit side-value and missingness standardization on training games only."""
    home = np.asarray(home, dtype=float)
    away = np.asarray(away, dtype=float)
    train_indices = np.asarray(train_indices, dtype=int)
    if home.shape != away.shape or home.ndim != 2:
        raise ValueError("home and away Context features must be matching matrices")
    if home.shape[1] != len(feature_names):
        raise ValueError("feature name count does not match Context values")
    if train_indices.size == 0:
        raise ValueError("Context preprocessing requires at least one training game")
    side_values = np.vstack((home[train_indices], away[train_indices]))
    missing = ~np.isfinite(side_values)
    means = np.zeros(side_values.shape[1], dtype=float)
    scales = np.ones(side_values.shape[1], dtype=float)
    missing_means = np.zeros(side_values.shape[1], dtype=float)
    missing_scales = np.ones(side_values.shape[1], dtype=float)
    for column in range(side_values.shape[1]):
        observed = side_values[~missing[:, column], column]
        if observed.size:
            means[column] = float(np.mean(observed))
            scale = float(np.std(observed))
            if scale > 1e-12:
                scales[column] = scale
        miss_values = missing[:, column].astype(float)
        missing_means[column] = float(np.mean(miss_values))
        missing_scale = float(np.std(miss_values))
        if missing_scale > 1e-12:
            missing_scales[column] = missing_scale
    return ContextPreprocessor(
        tuple(feature_names), means, scales, missing_means, missing_scales
    )


def identity_parameters(context_count: int = 0) -> np.ndarray:
    """Identity baseline start: multiplier 1, no extra home term or scale change."""
    if context_count < 0:
        raise ValueError("context_count cannot be negative")
    return np.zeros(3 + context_count, dtype=float)


def _group_log_mixture(
    pair_log_densities: np.ndarray, pair_game: np.ndarray, game_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stable per-game log mixtures and their complete pair responsibilities."""
    pair_log_densities = np.asarray(pair_log_densities, dtype=float)
    pair_game = np.asarray(pair_game, dtype=int)
    if pair_log_densities.ndim != 1 or pair_log_densities.shape != pair_game.shape:
        raise ValueError("pair log densities and game indices must be matching vectors")
    if game_count < 1 or pair_game.size == 0:
        raise ValueError("a likelihood mixture requires at least one game and pair")
    if np.any(pair_game < 0) or np.any(pair_game >= game_count):
        raise ValueError("rank-pair game index is out of range")
    counts = np.bincount(pair_game, minlength=game_count)
    if np.any(counts == 0):
        raise ValueError("every game must retain at least one rank pair")
    maxima = np.full(game_count, -np.inf, dtype=float)
    np.maximum.at(maxima, pair_game, pair_log_densities)
    sums = np.bincount(
        pair_game,
        weights=np.exp(pair_log_densities - maxima[pair_game]),
        minlength=game_count,
    )
    log_mix = maxima + np.log(sums) - np.log(counts)
    responsibilities = np.exp(pair_log_densities - maxima[pair_game]) / sums[pair_game]
    return log_mix, responsibilities, counts


def marginalized_nll_gradient(
    parameters: np.ndarray,
    view: LikelihoodSlice,
    *,
    baseline_scale: float,
    degrees_of_freedom: float,
    regularization: float,
) -> tuple[float, np.ndarray]:
    """Sum marginalized per-game NLL plus an L2 penalty, with analytic gradient."""
    parameters = np.asarray(parameters, dtype=float)
    context_count = view.context.shape[1]
    if parameters.shape != (3 + context_count,):
        raise ValueError("calibration parameter vector has the wrong size")
    if baseline_scale <= 0 or degrees_of_freedom <= 0 or regularization < 0:
        raise ValueError("scale/df must be positive and regularization nonnegative")
    log_multiplier, home_adjustment, log_scale = parameters[:3]
    multiplier = float(np.exp(log_multiplier))
    scale = baseline_scale * float(np.exp(log_scale))
    beta = parameters[3:]
    game_context_shift = view.context @ beta
    locations = (
        multiplier * view.base_locations
        + home_adjustment * view.home_field[view.pair_game]
        + game_context_shift[view.pair_game]
    )
    residual = view.margins[view.pair_game] - locations
    pair_log_density = student_t.logpdf(
        residual / scale, degrees_of_freedom
    ) - math.log(scale)
    log_mix, responsibilities, _ = _group_log_mixture(
        pair_log_density, view.pair_game, len(view.margins)
    )
    objective = -float(np.sum(log_mix)) + 0.5 * regularization * float(beta @ beta)
    denominator = degrees_of_freedom * scale * scale + residual * residual
    location_score = (degrees_of_freedom + 1.0) * residual / denominator
    weighted_location_score = -responsibilities * location_score
    group_location_score = np.bincount(
        view.pair_game,
        weights=weighted_location_score,
        minlength=len(view.margins),
    )
    log_scale_score = -1.0 + (degrees_of_freedom + 1.0) * residual**2 / denominator
    gradient = np.empty_like(parameters)
    gradient[0] = float(
        np.sum(weighted_location_score * multiplier * view.base_locations)
    )
    gradient[1] = float(group_location_score @ view.home_field)
    gradient[2] = -float(np.sum(responsibilities * log_scale_score))
    if context_count:
        gradient[3:] = view.context.T @ group_location_score + regularization * beta
    return objective, gradient


def make_likelihood_slice(
    data: LikelihoodData,
    game_indices: np.ndarray,
    context_design: np.ndarray,
) -> LikelihoodSlice:
    game_indices = np.asarray(game_indices, dtype=int)
    context_design = np.asarray(context_design, dtype=float)
    if game_indices.ndim != 1 or game_indices.size == 0:
        raise ValueError("likelihood slice needs at least one game")
    if context_design.shape[0] != game_indices.size:
        raise ValueError("Context design rows do not match selected games")
    if np.any(np.diff(game_indices) < 0):
        raise ValueError("game indices must be in ascending order")
    local_index = np.full(data.game_count, -1, dtype=int)
    local_index[game_indices] = np.arange(game_indices.size)
    old_pair_game = data.pair_game
    pair_mask = local_index[old_pair_game] >= 0
    pair_game = local_index[old_pair_game[pair_mask]]
    if pair_game.size == 0:
        raise ValueError("selected games have no rank-pair observations")
    return LikelihoodSlice(
        margins=data.margins[game_indices],
        home_field=data.home_field[game_indices],
        pair_game=pair_game,
        base_locations=data.base_locations[pair_mask],
        context=context_design,
    )


def fit_residual_calibration(
    view: LikelihoodSlice,
    *,
    baseline_scale: float,
    degrees_of_freedom: float = STUDENT_T_DF,
    regularization: float = 0.0,
) -> FitResult:
    """Fit shared rank/site recalibration, optionally with the linear Context term."""
    initial = identity_parameters(view.context.shape[1])
    bounds = [*PARAMETER_BOUNDS, *([(None, None)] * view.context.shape[1])]

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        return marginalized_nll_gradient(
            parameters,
            view,
            baseline_scale=baseline_scale,
            degrees_of_freedom=degrees_of_freedom,
            regularization=regularization,
        )

    result = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options=OPTIMIZER_OPTIONS,
    )
    gradient_norm = float(
        np.linalg.norm(np.asarray(result.jac, dtype=float), ord=np.inf)
    )
    if not result.success and gradient_norm > 1e-4:
        raise RuntimeError(
            "calibration optimizer did not converge: "
            f"status={result.status} message={result.message} gradient={gradient_norm:.6g}"
        )
    theta = np.asarray(result.x, dtype=float)
    return FitResult(
        log_multiplier=float(theta[0]),
        home_field_adjustment=float(theta[1]),
        log_scale_correction=float(theta[2]),
        context_coefficients=theta[3:].copy(),
        scale=baseline_scale * float(np.exp(theta[2])),
        objective=float(result.fun),
        optimizer_success=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        optimizer_iterations=int(result.nit),
        optimizer_evaluations=int(result.nfev),
        gradient_norm=gradient_norm,
        regularization=float(regularization),
    )


def _locations_from_fit(view: LikelihoodSlice, fit: FitResult) -> np.ndarray:
    context_shift = view.context @ fit.context_coefficients
    return (
        fit.multiplier * view.base_locations
        + fit.home_field_adjustment * view.home_field[view.pair_game]
        + context_shift[view.pair_game]
    )


def _mixture_cdf(value: float, locations: np.ndarray, scale: float, df: float) -> float:
    return float(np.mean(student_t.cdf((value - locations) / scale, df)))


def _mixture_quantile(
    probability: float, locations: np.ndarray, scale: float, df: float
) -> float:
    lower = float(np.min(locations) - 100.0 * scale)
    upper = float(np.max(locations) + 100.0 * scale)
    return float(
        brentq(
            lambda value: _mixture_cdf(value, locations, scale, df) - probability,
            lower,
            upper,
            xtol=1e-9,
        )
    )


def score_likelihood_slice(
    view: LikelihoodSlice,
    *,
    fit: FitResult | None,
    baseline_scale: float,
    degrees_of_freedom: float = STUDENT_T_DF,
) -> MixtureEvaluation:
    """Score equal-game mixtures and integer-margin midpoint PIT/central intervals."""
    if fit is None:
        locations = view.base_locations
        scale = baseline_scale
    else:
        locations = _locations_from_fit(view, fit)
        scale = fit.scale
    if scale <= 0 or not np.isfinite(locations).all():
        raise ValueError("invalid Student-t mixture locations or scale")
    counts = np.bincount(view.pair_game, minlength=len(view.margins))
    if np.any(counts == 0) or not np.array_equal(
        view.pair_game, np.repeat(np.arange(len(view.margins)), counts)
    ):
        raise ValueError("rank-pair mixtures must be complete and grouped by game")
    rows: list[dict[str, object]] = []
    game_nll: list[float] = []
    pair_offsets = np.concatenate(([0], np.cumsum(counts)))
    for game_index in range(len(view.margins)):
        first, last = int(pair_offsets[game_index]), int(pair_offsets[game_index + 1])
        game_locations = locations[first:last]
        if len(game_locations) != last - first:
            raise AssertionError(
                "rank-pair mixture group counts changed during scoring"
            )
        margin = float(view.margins[game_index])
        logs = student_t.logpdf(
            (margin - game_locations) / scale, degrees_of_freedom
        ) - math.log(scale)
        log_density = float(logsumexp(logs) - math.log(len(game_locations)))
        midpoint_low = _mixture_cdf(
            margin - 0.5, game_locations, scale, degrees_of_freedom
        )
        midpoint_high = _mixture_cdf(
            margin + 0.5, game_locations, scale, degrees_of_freedom
        )
        result: dict[str, object] = {
            "margin_log_density": log_density,
            "mixture_nll": -log_density,
            "predicted_margin_mean": float(np.mean(game_locations)),
            "signed_margin_residual": margin - float(np.mean(game_locations)),
            "mixture_pit_midpoint": 0.5 * (midpoint_low + midpoint_high),
            "rank_pair_count": len(game_locations),
        }
        for level in (0.50, 0.80, 0.95):
            tail = (1.0 - level) / 2.0
            lower = _mixture_quantile(tail, game_locations, scale, degrees_of_freedom)
            upper = _mixture_quantile(
                1.0 - tail, game_locations, scale, degrees_of_freedom
            )
            label = int(level * 100)
            result[f"interval_{label}_low"] = lower
            result[f"interval_{label}_high"] = upper
            result[f"interval_{label}_width"] = upper - lower
            result[f"interval_{label}_covered"] = lower <= margin <= upper
        rows.append(result)
        game_nll.append(-log_density)
    return MixtureEvaluation(tuple(rows), float(np.mean(game_nll)))


def training_indices(data: LikelihoodData, *, target_season: int) -> np.ndarray:
    """Return only residual seasons strictly earlier than the target season."""
    indices = np.flatnonzero(
        (data.seasons < target_season) & (data.seasons >= min(RESIDUAL_SEASONS))
    )
    if indices.size == 0:
        raise ValueError(f"{target_season}: no earlier residual training games")
    if np.any(data.seasons[indices] >= target_season):
        raise AssertionError("target season leaked into residual fitting")
    return indices


def _fit_context_preprocessor(
    data: LikelihoodData, indices: np.ndarray, feature_names: Sequence[str]
) -> tuple[ContextPreprocessor, np.ndarray]:
    preprocessor = fit_context_preprocessor(
        data.context_home, data.context_away, indices, feature_names
    )
    return preprocessor, preprocessor.transform(
        data.context_home[indices], data.context_away[indices]
    )


def _score_nll_for_indices(
    data: LikelihoodData,
    indices: np.ndarray,
    context_design: np.ndarray,
    *,
    fit: FitResult,
    baseline_scale: float,
    degrees_of_freedom: float,
) -> tuple[float, int]:
    view = make_likelihood_slice(data, indices, context_design)
    locations = _locations_from_fit(view, fit)
    scale = fit.scale
    pair_logs = student_t.logpdf(
        (view.margins[view.pair_game] - locations) / scale, degrees_of_freedom
    ) - math.log(scale)
    log_mix, _, _ = _group_log_mixture(pair_logs, view.pair_game, len(view.margins))
    return float(-np.mean(log_mix)), len(indices)


def select_context_regularization(
    data: LikelihoodData,
    train_indices: np.ndarray,
    *,
    baseline_scale: float,
    degrees_of_freedom: float = STUDENT_T_DF,
    feature_names: Sequence[str] = CONTEXT_FEATURES,
    grid: Sequence[float] = REGULARIZATION_GRID,
) -> tuple[float, dict[str, object]]:
    """Choose ridge only by forward validation within the earlier seasons."""
    train_indices = np.asarray(train_indices, dtype=int)
    train_seasons = sorted({int(value) for value in data.seasons[train_indices]})
    if len(train_seasons) == 1:
        return DEFAULT_REGULARIZATION, {
            "selection": "default_single_earlier_season",
            "training_seasons": train_seasons,
            "selected_regularization": DEFAULT_REGULARIZATION,
            "grid": list(grid),
            "validation_folds": [],
        }
    validation_seasons = train_seasons[1:]
    scores: dict[float, dict[str, object]] = {
        float(value): {"nll_sum": 0.0, "game_count": 0, "folds": []} for value in grid
    }
    for validation_season in validation_seasons:
        fold_train = train_indices[data.seasons[train_indices] < validation_season]
        fold_validation = train_indices[
            data.seasons[train_indices] == validation_season
        ]
        if fold_train.size == 0 or fold_validation.size == 0:
            raise ValueError("forward regularization fold has an empty season")
        train_preprocessor, train_context = _fit_context_preprocessor(
            data, fold_train, feature_names
        )
        validation_context = train_preprocessor.transform(
            data.context_home[fold_validation], data.context_away[fold_validation]
        )
        train_view = make_likelihood_slice(data, fold_train, train_context)
        for regularization in grid:
            fit = fit_residual_calibration(
                train_view,
                baseline_scale=baseline_scale,
                degrees_of_freedom=degrees_of_freedom,
                regularization=float(regularization),
            )
            nll, game_count = _score_nll_for_indices(
                data,
                fold_validation,
                validation_context,
                fit=fit,
                baseline_scale=baseline_scale,
                degrees_of_freedom=degrees_of_freedom,
            )
            aggregate = scores[float(regularization)]
            aggregate["nll_sum"] = float(aggregate["nll_sum"]) + nll * game_count
            aggregate["game_count"] = int(aggregate["game_count"]) + game_count
            aggregate["folds"].append(  # type: ignore[union-attr]
                {
                    "training_seasons": sorted(
                        {int(value) for value in data.seasons[fold_train]}
                    ),
                    "validation_season": validation_season,
                    "n_games": game_count,
                    "mean_game_nll": nll,
                    "optimizer": fit.metadata(
                        [
                            *feature_names,
                            *[f"missing__{name}" for name in feature_names],
                        ]
                    )["optimizer"],
                }
            )
    for aggregate in scores.values():
        aggregate["mean_game_nll"] = float(aggregate["nll_sum"]) / int(
            aggregate["game_count"]
        )
    selected = min(
        scores,
        key=lambda value: (
            float(scores[value]["mean_game_nll"]),
            abs(value - DEFAULT_REGULARIZATION),
        ),
    )
    return selected, {
        "selection": "forward_validation_on_earlier_seasons_only",
        "training_seasons": train_seasons,
        "validation_seasons": validation_seasons,
        "selected_regularization": selected,
        "grid": list(grid),
        "validation_scores": {
            str(value): {
                "mean_game_nll": scores[value]["mean_game_nll"],
                "n_games": scores[value]["game_count"],
                "folds": scores[value]["folds"],
            }
            for value in grid
        },
    }


def _read_issue201_games(path: Path) -> list[dict[str, object]]:
    score_rows = read_csv(path)
    selected = [
        row
        for row in score_rows
        if row.get("forecast_stage") == "preseason"
        and row.get("forecast_population") == "fbs_fbs"
        and int(row.get("season") or 0) in (*RESIDUAL_SEASONS, 2025)
    ]
    by_game: dict[str, dict[str, dict[str, str]]] = {}
    for row in selected:
        game_id = row.get("game_id", "").strip()
        arm = row.get("model_arm", "").strip()
        if not game_id or arm not in MODEL_ARMS:
            raise ValueError("#201 preseason FBS-FBS row has missing id or unknown arm")
        if arm in by_game.setdefault(game_id, {}):
            raise ValueError(f"#201 has duplicate primary row for {game_id}/{arm}")
        by_game[game_id][arm] = row
    games: list[dict[str, object]] = []
    for game_id in sorted(by_game, key=int):
        arms = by_game[game_id]
        if set(arms) != set(MODEL_ARMS):
            raise ValueError(
                f"#201 primary cohort is not four-arm common for game {game_id}"
            )
        reference = arms["History_existing"]
        for arm_name, row in arms.items():
            for field in (
                "season",
                "phase",
                "home_id",
                "away_id",
                "home_team",
                "away_team",
                "actual_home_points",
                "actual_away_points",
                "actual_home_margin",
                "neutral_site",
            ):
                if row.get(field) != reference.get(field):
                    raise ValueError(
                        f"#201 {game_id}: {field} differs in {arm_name} from History_existing"
                    )
        try:
            season = int(reference["season"])
            home_points = int(reference["actual_home_points"])
            away_points = int(reference["actual_away_points"])
            margin = int(reference["actual_home_margin"])
            home_id = str(reference["home_id"])
            away_id = str(reference["away_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"#201 game {game_id}: malformed score or team id"
            ) from exc
        if not home_id or not away_id or home_id == away_id:
            raise ValueError(f"#201 game {game_id}: canonical FBS team ids are invalid")
        if margin != home_points - away_points:
            raise ValueError(f"#201 game {game_id}: margin disagrees with the score")
        games.append(
            {
                "id": game_id,
                "season": season,
                "week": "",
                "startDate": reference.get("kickoff_utc", ""),
                "homeId": home_id,
                "awayId": away_id,
                "homeClassification": "fbs",
                "awayClassification": "fbs",
                "homePoints": str(home_points),
                "awayPoints": str(away_points),
                "neutralSite": reference.get("neutral_site", "False"),
                "homeTeam": reference.get("home_team", ""),
                "awayTeam": reference.get("away_team", ""),
                "phase": reference.get("phase", ""),
                "army_navy": reference.get("phase", "") == "army_navy",
                "conference_game": reference.get("conference_game", ""),
                "kickoff_utc": reference.get("kickoff_utc", ""),
                "home_points": home_points,
                "away_points": away_points,
                "margin": margin,
            }
        )
    if not games:
        raise ValueError("#201 has no preseason FBS-FBS games in 2022-2025")
    return games


def _read_targets(path: Path) -> list[dict[str, str]]:
    result = []
    for row in read_csv(path):
        if (
            int(row.get("season") or 0) in (*RESIDUAL_SEASONS, 2025)
            and row.get("subdivision") == "fbs"
        ):
            result.append(row)
    return result


def _target_index(
    rows: Sequence[dict[str, str]],
) -> dict[tuple[int, str, str], dict[str, str]]:
    result: dict[tuple[int, str, str], dict[str, str]] = {}
    for row in rows:
        key = (int(row["season"]), row["subdivision"], row["team_id"])
        if key in result:
            raise ValueError(f"duplicate corrected final-rank target {key}")
        result[key] = row
    return result


def _read_rank_pair_lookup(
    path: Path,
    candidate_game_ids: set[str],
    target_rows: dict[tuple[int, str, str], dict[str, str]],
) -> tuple[
    dict[tuple[int, tuple[int, str, str], tuple[int, str, str]], list[tuple[int, int]]],
    dict[str, list[dict[str, object]]],
]:
    lookup: dict[
        tuple[int, tuple[int, str, str], tuple[int, str, str]], list[tuple[int, int]]
    ] = {}
    manifest: dict[str, list[dict[str, object]]] = {
        game_id: [] for game_id in candidate_game_ids
    }
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        expected_fields = (
            "game_id",
            "season",
            "system_code",
            "home_team_id",
            "home_subdivision",
            "home_rank",
            "home_team_population",
            "home_rank_percentile",
            "away_team_id",
            "away_subdivision",
            "away_rank",
            "away_team_population",
            "away_rank_percentile",
        )
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError(
                "#204 final rank-pair schema differs from modeling builder contract"
            )
        for row in reader:
            game_id = row["game_id"]
            if game_id not in candidate_game_ids:
                continue
            season = int(row["season"])
            home_key = (season, row["home_subdivision"], row["home_team_id"])
            away_key = (season, row["away_subdivision"], row["away_team_id"])
            home_target = target_rows.get(home_key)
            away_target = target_rows.get(away_key)
            if home_target is None or away_target is None:
                raise ValueError(
                    f"#204 rank pair {game_id}: corrected target row is missing"
                )
            home_population = int(home_target["team_population"])
            away_population = int(away_target["team_population"])
            if (
                int(row["home_team_population"]) != home_population
                or int(row["away_team_population"]) != away_population
            ):
                raise ValueError(
                    f"#204 rank pair {game_id}: target population mismatch"
                )
            home_rank, away_rank = int(row["home_rank"]), int(row["away_rank"])
            if (
                not 1 <= home_rank <= home_population
                or not 1 <= away_rank <= away_population
            ):
                raise ValueError(
                    f"#204 rank pair {game_id}: rank is outside corrected support"
                )
            key = (int(game_id), home_key, away_key)
            lookup.setdefault(key, []).append((home_rank, away_rank))
            manifest[game_id].append(
                {
                    "system_code": row["system_code"],
                    "home_rank": home_rank,
                    "away_rank": away_rank,
                }
            )
    return lookup, manifest


def _read_context_features(
    feature_path: Path,
    feature_manifest_path: Path,
    *,
    target_seasons: Sequence[int],
) -> tuple[dict[tuple[int, str], dict[str, object]], dict[str, object]]:
    rows = read_csv(feature_path)
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    result: dict[tuple[int, str], dict[str, object]] = {}
    for row in rows:
        season = int(row.get("season") or 0)
        if season not in target_seasons or row.get("subdivision") != "fbs":
            continue
        team_id = row.get("team_id", "")
        key = (season, team_id)
        if not team_id or key in result:
            raise ValueError(
                f"#200 Context feature input has invalid/duplicate key {key}"
            )
        try:
            features = json.loads(row["features_json"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"#200 Context feature row {key} lacks valid features_json"
            ) from exc
        if not isinstance(features, dict) or set(features) != set(MODEL_FEATURES):
            raise ValueError(
                f"#200 Context feature row {key} differs from Context 1.4 schema"
            )
        if sha256_json(features) != row.get("context_feature_sha256"):
            raise ValueError(f"#200 Context feature row {key} hash mismatch")
        season_manifest = feature_manifest.get(str(season), {})
        predictor_rows = {
            str(item.get("team_id")): item
            for item in season_manifest.get("predictor_rows", [])
        }
        predictor = predictor_rows.get(team_id)
        if predictor is None:
            raise ValueError(f"#200 feature manifest is missing predictor row {key}")
        if predictor.get("predictor_input_sha256") != row.get("predictor_input_sha256"):
            raise ValueError(f"#200 feature row {key} predictor hash mismatch")
        if predictor.get("context_feature_sha256") != row.get("context_feature_sha256"):
            raise ValueError(
                f"#200 feature manifest {key} predates exact feature-row export; rebuild #200"
            )
        values = {
            name: (np.nan if features[name] is None else float(features[name]))
            for name in CONTEXT_FEATURES
        }
        result[key] = {
            "features": values,
            "features_json": row["features_json"],
            "context_feature_sha256": row["context_feature_sha256"],
            "predictor_input_sha256": row["predictor_input_sha256"],
            "team_name": row.get("team_name", ""),
            "repaired_transfer_feature_available": row.get(
                "repaired_transfer_feature_available", ""
            ),
        }
    return result


def _build_likelihood_data(
    game_rows: list[dict[str, object]],
    joined_rows: list[dict[str, str]],
    feature_rows: dict[tuple[int, str], dict[str, object]],
    baseline_beta: np.ndarray,
) -> LikelihoodData:
    joined_by_id = {str(row["game_id"]): row for row in joined_rows}
    game_rows = sorted(game_rows, key=lambda row: int(str(row["id"])))
    game_ids = tuple(str(row["id"]) for row in game_rows)
    if len(set(game_ids)) != len(game_ids):
        raise ValueError("duplicate game ids in the common issue-201 cohort")
    pseudo = historical_builder.pseudo_data(joined_rows)
    design = historical_builder.design_matrix(
        pseudo.x,
        pseudo.y,
        pseudo.pairing,
        pseudo.home,
        pseudo.neutral,
        surface=True,
        fbs_home=pseudo.fbs_home,
    )
    if design.shape[1] != len(baseline_beta):
        raise ValueError(
            f"selected #200 likelihood has {len(baseline_beta)} coefficients but the "
            f"historical modeling builder produced {design.shape[1]} features"
        )
    base_locations = design @ baseline_beta
    game_index = {game_id: index for index, game_id in enumerate(game_ids)}
    pair_game = np.asarray(
        [game_index[str(int(game_id))] for game_id in pseudo.game_id], dtype=int
    )
    metadata: list[dict[str, object]] = []
    seasons: list[int] = []
    margins: list[float] = []
    home_field: list[float] = []
    home_context: list[list[float]] = []
    away_context: list[list[float]] = []
    for game in game_rows:
        game_id = str(game["id"])
        joined = joined_by_id.get(game_id)
        if joined is None:
            raise ValueError(
                f"game {game_id}: historical builder did not retain the game"
            )
        season = int(game["season"])
        home_id, away_id = str(game["homeId"]), str(game["awayId"])
        home_features = feature_rows[(season, home_id)]["features"]
        away_features = feature_rows[(season, away_id)]["features"]
        context = {
            **game,
            "pairing_method": joined["pairing_method"],
            "rank_pair_count": len(json.loads(joined["rank_pairs"])),
            "rank_pairs": json.loads(joined["rank_pairs"]),
            "home_team_population": int(joined["home_team_population"]),
            "away_team_population": int(joined["away_team_population"]),
            "home_context_feature_sha256": feature_rows[(season, home_id)][
                "context_feature_sha256"
            ],
            "away_context_feature_sha256": feature_rows[(season, away_id)][
                "context_feature_sha256"
            ],
        }
        metadata.append(context)
        seasons.append(season)
        margins.append(float(game["margin"]))
        home_field.append(
            0.0 if historical_builder._as_bool(game["neutralSite"]) else 1.0
        )
        home_context.append([float(home_features[name]) for name in CONTEXT_FEATURES])
        away_context.append([float(away_features[name]) for name in CONTEXT_FEATURES])
    if set(joined_by_id) != set(game_ids):
        raise ValueError("rank-pair join changed the common game cohort")
    return LikelihoodData(
        game_ids=game_ids,
        seasons=np.asarray(seasons, dtype=int),
        margins=np.asarray(margins, dtype=float),
        home_field=np.asarray(home_field, dtype=float),
        pair_game=pair_game,
        base_locations=np.asarray(base_locations, dtype=float),
        context_home=np.asarray(home_context, dtype=float),
        context_away=np.asarray(away_context, dtype=float),
        metadata=tuple(metadata),
    )


def _evaluate_arm(
    data: LikelihoodData,
    indices: np.ndarray,
    context_design: np.ndarray,
    *,
    fit: FitResult | None,
    scale: float,
    degrees_of_freedom: float,
    arm: str,
) -> list[dict[str, object]]:
    view = make_likelihood_slice(data, indices, context_design)
    result = score_likelihood_slice(
        view,
        fit=fit,
        baseline_scale=scale,
        degrees_of_freedom=degrees_of_freedom,
    )
    rows = []
    for local_index, game_index in enumerate(indices):
        metadata = data.metadata[int(game_index)]
        rows.append(
            {
                "game_id": data.game_ids[int(game_index)],
                "season": int(data.seasons[int(game_index)]),
                "phase": metadata.get("phase", ""),
                "army_navy": bool(metadata.get("army_navy", False)),
                "home_team_id": metadata["homeId"],
                "away_team_id": metadata["awayId"],
                "home_team": metadata.get("homeTeam", ""),
                "away_team": metadata.get("awayTeam", ""),
                "neutral_site": historical_builder._as_bool(metadata["neutralSite"]),
                "actual_home_margin": int(data.margins[int(game_index)]),
                "pairing_method": metadata["pairing_method"],
                "rank_pair_count": metadata["rank_pair_count"],
                "arm": arm,
                **result.rows[local_index],
            }
        )
    return rows


def _comparison_outputs(
    per_game_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    by_arm_game = {(str(row["game_id"]), str(row["arm"])): row for row in per_game_rows}
    games = sorted({str(row["game_id"]) for row in per_game_rows}, key=int)
    season_summary: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    phase_summaries: list[dict[str, object]] = []
    for season in TARGET_SEASONS:
        season_ids = [
            game_id
            for game_id in games
            if int(by_arm_game[(game_id, "rank_only")]["season"]) == season
        ]
        if not season_ids:
            continue
        for arm in DIAGNOSTIC_ARMS:
            arm_rows = [by_arm_game[(game_id, arm)] for game_id in season_ids]
            summary: dict[str, object] = {
                "season": season,
                "role": "held_out_evaluation",
                "arm": arm,
                "n_games": len(arm_rows),
                "mean_game_mixture_nll": float(
                    np.mean([row["mixture_nll"] for row in arm_rows])
                ),
                "mean_mixture_pit_midpoint": float(
                    np.mean([row["mixture_pit_midpoint"] for row in arm_rows])
                ),
                "mean_signed_margin_residual": float(
                    np.mean([row["signed_margin_residual"] for row in arm_rows])
                ),
            }
            for level in (50, 80, 95):
                summary[f"coverage_{level}"] = float(
                    np.mean([row[f"interval_{level}_covered"] for row in arm_rows])
                )
                summary[f"mean_width_{level}"] = float(
                    np.mean([row[f"interval_{level}_width"] for row in arm_rows])
                )
            season_summary.append(summary)
        for name, candidate, reference in COMPARISONS:
            differences = [
                float(by_arm_game[(game_id, candidate)]["mixture_nll"])
                - float(by_arm_game[(game_id, reference)]["mixture_nll"])
                for game_id in season_ids
            ]
            comparison_rows.append(
                {
                    "comparison": name,
                    "aggregation": "season",
                    "season": season,
                    "n_games": len(differences),
                    "mean_candidate_minus_reference_nll": float(np.mean(differences)),
                    "interpretation": "negative favors candidate",
                }
            )
    for name, candidate, reference in COMPARISONS:
        differences = [
            float(by_arm_game[(game_id, candidate)]["mixture_nll"])
            - float(by_arm_game[(game_id, reference)]["mixture_nll"])
            for game_id in games
        ]
        comparison_rows.append(
            {
                "comparison": name,
                "aggregation": "pooled_equal_game",
                "season": "all_held_out",
                "n_games": len(differences),
                "mean_candidate_minus_reference_nll": float(np.mean(differences)),
                "interpretation": "negative favors candidate",
            }
        )
        per_season = [
            float(row["mean_candidate_minus_reference_nll"])
            for row in comparison_rows
            if row["comparison"] == name and row["aggregation"] == "season"
        ]
        if per_season:
            comparison_rows.append(
                {
                    "comparison": name,
                    "aggregation": "equal_season",
                    "season": "all_held_out",
                    "n_games": len(differences),
                    "mean_candidate_minus_reference_nll": float(np.mean(per_season)),
                    "interpretation": "negative favors candidate",
                }
            )
        for direction, selected in (
            ("positive_gain", [value for value in differences if value < 0]),
            ("negative_loss", [value for value in differences if value > 0]),
        ):
            magnitudes = sorted((abs(value) for value in selected), reverse=True)
            total = float(sum(magnitudes))
            concentration: dict[str, object] = {
                "comparison": name,
                "direction": direction,
                "n_games": len(differences),
                "contributing_games": len(magnitudes),
                "total_absolute_nll_change": total,
            }
            for pct in (1, 5, 10):
                top_count = (
                    max(1, math.ceil(len(magnitudes) * pct / 100)) if magnitudes else 0
                )
                top_sum = float(sum(magnitudes[:top_count]))
                concentration[f"top_{pct}pct_share"] = top_sum / total if total else 0.0
                concentration[f"top_{pct}pct_game_count"] = top_count
            team_totals: dict[str, float] = {}
            for game_id, delta in zip(games, differences, strict=True):
                if (direction == "positive_gain" and delta >= 0) or (
                    direction == "negative_loss" and delta <= 0
                ):
                    continue
                game = by_arm_game[(game_id, reference)]
                magnitude = abs(delta)
                for team_id in (str(game["home_team_id"]), str(game["away_team_id"])):
                    team_totals[team_id] = (
                        team_totals.get(team_id, 0.0) + magnitude / 2.0
                    )
            ordered_team = sorted(team_totals.values(), reverse=True)
            team_total = float(sum(ordered_team))
            count = max(1, math.ceil(len(ordered_team) * 0.10)) if ordered_team else 0
            concentration["largest_team_absolute_share"] = (
                ordered_team[0] / team_total if team_total else 0.0
            )
            concentration["top_10pct_team_share"] = (
                sum(ordered_team[:count]) / team_total if team_total else 0.0
            )
            phase_summaries.append(concentration)
    phase_rows: list[dict[str, object]] = []
    phase_groups: dict[tuple[int, str], list[str]] = {}
    for game_id in games:
        row = by_arm_game[(game_id, "rank_only")]
        key = (int(row["season"]), str(row["phase"]))
        phase_groups.setdefault(key, []).append(game_id)
    for (season, phase), ids in sorted(phase_groups.items()):
        for arm in DIAGNOSTIC_ARMS:
            arm_rows = [by_arm_game[(game_id, arm)] for game_id in ids]
            phase_rows.append(
                {
                    "season": season,
                    "phase": phase,
                    "arm": arm,
                    "n_games": len(arm_rows),
                    "mean_game_mixture_nll": float(
                        np.mean([row["mixture_nll"] for row in arm_rows])
                    ),
                    "mean_signed_margin_residual": float(
                        np.mean([row["signed_margin_residual"] for row in arm_rows])
                    ),
                }
            )
    return season_summary, comparison_rows, phase_rows + phase_summaries


def _decision(comparison_rows: list[dict[str, object]]) -> dict[str, object]:
    lookup = {
        (str(row["comparison"]), str(row["aggregation"])): float(
            row["mean_candidate_minus_reference_nll"]
        )
        for row in comparison_rows
    }
    seasons = [
        lookup[("rank_site_context_minus_rank_site_calibration", "season")]
        for row in comparison_rows
        if row["comparison"] == "rank_site_context_minus_rank_site_calibration"
        and row["aggregation"] == "season"
    ]
    context_pooled = lookup.get(
        ("rank_site_context_minus_rank_site_calibration", "pooled_equal_game"), math.nan
    )
    context_equal_season = lookup.get(
        ("rank_site_context_minus_rank_site_calibration", "equal_season"), math.nan
    )
    calibration_pooled = lookup.get(
        ("rank_site_calibration_minus_rank_only", "pooled_equal_game"), math.nan
    )
    calibration_equal_season = lookup.get(
        ("rank_site_calibration_minus_rank_only", "equal_season"), math.nan
    )
    context_consistent = (
        np.isfinite(context_pooled)
        and np.isfinite(context_equal_season)
        and context_pooled < 0
        and context_equal_season < 0
        and sum(value < 0 for value in seasons) >= 2
    )
    calibration_consistent = (
        np.isfinite(calibration_pooled)
        and np.isfinite(calibration_equal_season)
        and calibration_pooled < 0
        and calibration_equal_season < 0
        and sum(
            float(row["mean_candidate_minus_reference_nll"]) < 0
            for row in comparison_rows
            if row["comparison"] == "rank_site_calibration_minus_rank_only"
            and row["aggregation"] == "season"
        )
        >= 2
    )
    if context_consistent:
        conclusion = "Context association beyond calibration"
    elif (
        calibration_consistent
        and np.isfinite(context_pooled)
        and np.isfinite(context_equal_season)
        and context_pooled >= 0
        and context_equal_season >= 0
    ):
        conclusion = "calibration explains the association"
    else:
        conclusion = "inconclusive/no detectable Context association"
    return {
        "status": "complete",
        "classification": conclusion,
        "primary_comparison": "rank_site_context_minus_rank_site_calibration",
        "generic_calibration_comparison": "rank_site_calibration_minus_rank_only",
        "direction_rule": "candidate-minus-reference NLL must be negative in pooled equal-game and equal-season summaries and at least two of three held-out seasons",
        "context_pooled_equal_game_delta_nll": context_pooled,
        "context_equal_season_delta_nll": context_equal_season,
        "context_negative_seasons": sum(value < 0 for value in seasons),
        "calibration_pooled_equal_game_delta_nll": calibration_pooled,
        "calibration_equal_season_delta_nll": calibration_equal_season,
        "limitations": [
            "This is an ex-post game-margin mechanism diagnostic, not an operational forecast.",
            "Final ranking measurements include regular-season and postseason outcomes and may include the diagnosed game.",
            "Rank measurements are noisy and endogenous; the result does not test conditional independence given true ranks.",
            "2022-2025 informed prior research and remain exploratory development seasons.",
            "Repeated teams and seasons preclude treating games as independent confirmation.",
            "A negative result addresses only the fixed linear Context mean correction.",
        ],
    }


def _file_records(
    paths: dict[str, tuple[Path, Path]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for logical_name, (path, root) in paths.items():
        try:
            relative_path = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            relative_path = f"external-input/{path.name}"
        row: dict[str, object] = {
            "logical_name": logical_name,
            "path": relative_path,
            "exists": path.is_file(),
        }
        if path.is_file():
            row["bytes"] = path.stat().st_size
            row["sha256"] = sha256_file(path)
            records.append(row)
        else:
            missing.append(row)
    return records, missing


def _required_paths(args: argparse.Namespace) -> dict[str, tuple[Path, Path]]:
    corpus = args.corpus_root
    issue200 = args.issue200_root
    issue201 = args.issue201_root
    return {
        "issue204_corrected_final_observations": (
            corpus / "data/processed/massey_issue_204_v1/final_observations.csv",
            corpus,
        ),
        "issue204_corrected_final_targets": (
            corpus
            / "data/processed/massey_prebowl_issue_204_v1/final_target_distributions.csv",
            corpus,
        ),
        "issue204_corrected_final_rank_pairs": (
            corpus
            / "data/processed/massey_prebowl_issue_204_v1/final_constituent_rank_pairs.csv",
            corpus,
        ),
        "issue200_existing_likelihood": (
            issue200 / "historical_likelihood_existing.json",
            issue200,
        ),
        "issue200_manifest": (issue200 / "manifest.json", issue200),
        "issue200_pipeline_bundle": (issue200 / "pipeline_bundle.json", issue200),
        "issue200_preseason_feature_manifest": (
            issue200 / "preseason_feature_manifest.json",
            issue200,
        ),
        "issue200_context_feature_inputs": (
            issue200 / "context_feature_inputs.csv",
            issue200,
        ),
        "issue201_manifest": (issue201 / "manifest.json", issue201),
        "issue201_forecast_scores": (issue201 / "forecast_scores.csv", issue201),
    }


def _input_gate_report(
    args: argparse.Namespace,
    input_records: list[dict[str, object]],
    missing: list[dict[str, object]],
    synthetic_controls: dict[str, object] | None,
) -> dict[str, object]:
    status = "blocked_missing_inputs" if missing else "ready"
    gate = {
        "artifact_kind": "issue_195_input_gate",
        "status": status,
        "missing_required_inputs": missing,
        "available_inputs": input_records,
        "execution": "synthetic_controls_only" if missing else "not_started",
        "scientific_conclusion": None,
        "synthetic_controls": (
            "synthetic_controls.json" if synthetic_controls is not None else None
        ),
        "root_arguments": {
            "corpus_root": args.corpus_root.name or ".",
            "issue200_root": args.issue200_root.name or ".",
            "issue201_root": args.issue201_root.name or ".",
        },
    }
    return gate


def _write_blocked_report(
    output_root: Path,
    gate: dict[str, object],
    synthetic_controls: dict[str, object] | None,
) -> None:
    missing = gate["missing_required_inputs"]
    lines = [
        "# Issue 195: rank-only likelihood Context diagnostic",
        "",
        "**Run status: blocked by missing inputs. No historical comparison was scored and no scientific decision was made.**",
        "",
        "The available #200 likelihood and #201 prospective comparison artifacts are retained with recorded hashes. The corrected #204 final-rank target/pair corpus and a new exact-row export of #200 Context inputs are required to build the same-game mixtures. The legacy `data/processed/massey` corpus and the pre-bowl pair file are not substitutes.",
        "",
        "## Missing required inputs",
        "",
    ]
    for row in missing:
        lines.append(f"- `{row['path']}` ({row['logical_name']})")
    lines.extend(
        [
            "",
            "## Available checks",
            "",
            "- The deterministic synthetic null and positive-control simulations use the frozen #200 existing-policy likelihood. They validate the likelihood mixture, gradient, and control fitting behavior only; they do not stand in for the historical cohort.",
            "- The 2022–2025 result is exploratory development, and final ranks are ex-post measurements that may include each diagnosed regular-season game and postseason information.",
            "- No production likelihood, rankings, model defaults, or published outputs were changed.",
            "",
            "## Regeneration and rerun",
            "",
            "Restore the preserved #204 sources under `data/raw/` and run the repository's existing commands from `data/research/massey_rank_population_issue_204/report.md`:",
            "",
            "```sh",
            "uv run python scripts/normalize_massey.py --raw-root data/raw --output-root data/processed/massey_issue_204_v1",
            "uv run python scripts/build_massey_datasets.py --source data/processed/massey_issue_204_v1/observations.csv --output-root data/processed/massey_issue_204_v1",
            "uv run python scripts/build_massey_prebowl_targets.py --raw-root data/raw --final-observations data/processed/massey_issue_204_v1/final_observations.csv --checklist data/research/massey_prebowl_issue_198/collection_checklist.csv --output-root data/processed/massey_prebowl_issue_204_v1 --report-root data/research/massey_rank_population_issue_204 --coverage-threshold 0.90 --games data/raw/cfbd/games",
            "uv run python scripts/build_matched_history_context_issue_200.py --corpus-root CORPUS_ROOT --source-root SOURCE_ROOT --game-root GAME_ROOT --reference-root REFERENCE_ROOT --output-root ISSUE200_OUTPUT_ROOT",
            "uv run python scripts/run_issue_195_likelihood_diagnostic.py --corpus-root CORPUS_ROOT --issue200-root ISSUE200_OUTPUT_ROOT --issue201-root data/research/issue_201_prebowl_forecast --output-root data/research/issue_195_likelihood_diagnostic",
            "```",
            "",
            "Issue #204's audit command is `uv run python scripts/audit_massey_rank_population_issue_204.py --help`; use its recorded original-artifact and corrected-artifact paths after regeneration.",
            "",
            "## Synthetic controls",
            "",
        ]
    )
    if synthetic_controls is None:
        lines.append(
            "Synthetic controls were skipped because the frozen #200 likelihood was unavailable."
        )
    else:
        for name, control in synthetic_controls["cases"].items():
            lines.append(
                f"- **{name}:** {control['status']}; held-out NLL change for Context versus generic calibration is {float(control['context_minus_generic_nll']):.5f} nats/game; known Context effect is {control['known_context_effect_points']} points."
            )
    lines.extend(
        [
            "",
            "The synthetic results do not satisfy issue 195's historical acceptance criteria. No per-game or season summaries are published until the common #201 cohort, corrected #204 rank pairs, and exact #200 Context feature rows are available.",
            "",
        ]
    )
    (output_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _load_and_validate_provenance(
    paths: dict[str, tuple[Path, Path]], records: list[dict[str, object]]
) -> dict[str, object]:
    record_by_name = {str(row["logical_name"]): row for row in records}
    issue200_manifest = json.loads(
        paths["issue200_manifest"][0].read_text(encoding="utf-8")
    )
    issue201_manifest = json.loads(
        paths["issue201_manifest"][0].read_text(encoding="utf-8")
    )
    issue200_input_records = {
        str(row["logical_name"]): row for row in issue200_manifest["input_artifacts"]
    }
    issue200_output_records = issue200_manifest.get("outputs", {})
    for logical_name, key in (
        ("issue204_corrected_final_observations", "corrected_final_observations"),
        ("issue204_corrected_final_targets", "corrected_final_targets"),
        ("issue204_corrected_final_rank_pairs", "corrected_final_rank_pairs"),
    ):
        current = record_by_name[logical_name]
        expected = issue200_input_records.get(key)
        if expected is None or current["sha256"] != expected["sha256"]:
            raise ValueError(
                f"{logical_name}: hash differs from the #200 likelihood inputs"
            )
    likelihood_record = record_by_name["issue200_existing_likelihood"]
    expected_likelihood = issue200_output_records.get(
        "historical_likelihood_existing.json"
    )
    if (
        expected_likelihood is None
        or likelihood_record["sha256"] != expected_likelihood["sha256"]
    ):
        raise ValueError("#200 existing likelihood file differs from its manifest")
    bundle_record = record_by_name["issue200_pipeline_bundle"]
    expected_bundle = issue201_manifest["input_artifacts"]["issue200_pipeline_bundle"][
        "sha256"
    ]
    if bundle_record["sha256"] != expected_bundle:
        raise ValueError("#200 bundle differs from the bundle used by #201")
    forecast_record = record_by_name["issue201_forecast_scores"]
    expected_forecasts = issue201_manifest["outputs"]["forecast_scores.csv"]["sha256"]
    if forecast_record["sha256"] != expected_forecasts:
        raise ValueError("#201 forecast score file differs from its manifest")
    feature_output = issue200_output_records.get("context_feature_inputs.csv")
    feature_record = record_by_name["issue200_context_feature_inputs"]
    if feature_output is None or feature_record["sha256"] != feature_output["sha256"]:
        raise ValueError(
            "#200 exact Context feature export differs from its manifest; rebuild #200"
        )
    feature_manifest_output = issue200_output_records.get(
        "preseason_feature_manifest.json"
    )
    feature_manifest_record = record_by_name["issue200_preseason_feature_manifest"]
    if (
        feature_manifest_output is None
        or feature_manifest_record["sha256"] != feature_manifest_output["sha256"]
    ):
        raise ValueError("#200 feature manifest differs from its output manifest")
    return issue200_input_records


def _load_baseline(path: Path) -> tuple[np.ndarray, float, float, dict[str, object]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("artifact_kind") != "matched_historical_likelihood_issue_200":
        raise ValueError("selected likelihood is not the frozen issue-200 artifact")
    if (
        artifact.get("policy") != "existing"
        or artifact.get("training_seasons") != "2003-2021"
    ):
        raise ValueError("selected likelihood is not the existing-policy 2003-2021 fit")
    beta = np.asarray(artifact["beta"], dtype=float)
    scale = float(artifact["scale"])
    df = float(artifact["degrees_of_freedom"])
    if beta.ndim != 1 or beta.size != int(artifact["feature_count"]) or beta.size != 34:
        raise ValueError("frozen existing-policy likelihood coefficient schema changed")
    if not np.isfinite(beta).all() or not math.isfinite(scale) or scale <= 0:
        raise ValueError("frozen likelihood contains invalid parameters")
    if df != STUDENT_T_DF or float(artifact["student_t_df"]) != STUDENT_T_DF:
        raise ValueError("frozen likelihood degrees of freedom changed")
    return beta, scale, df, artifact


def run_synthetic_controls(
    beta: np.ndarray,
    baseline_scale: float,
    degrees_of_freedom: float = STUDENT_T_DF,
    *,
    seed: int = 195,
    training_games: int = 240,
    evaluation_games: int = 96,
    rank_pairs_per_game: int = 4,
    known_context_effect: float = 5.0,
) -> dict[str, object]:
    """Bounded deterministic null/positive controls from the frozen likelihood."""
    if training_games < 20 or evaluation_games < 10 or rank_pairs_per_game < 2:
        raise ValueError(
            "synthetic control dimensions are below the bounded fixture size"
        )
    results: dict[str, object] = {
        "artifact_kind": "issue_195_synthetic_controls",
        "seed": seed,
        "training_games_per_case": training_games,
        "evaluation_games_per_case": evaluation_games,
        "rank_pairs_per_game": rank_pairs_per_game,
        "rank_likelihood": "frozen issue-200 existing-policy rank/site Student-t",
        "known_positive_context_effect_points_per_training_standard_deviation": known_context_effect,
        "cases": {},
    }
    for scenario, effect, scenario_seed in (
        ("null", 0.0, seed),
        ("positive_control", known_context_effect, seed + 1),
    ):
        rng = np.random.default_rng(scenario_seed)
        game_count = training_games + evaluation_games
        neutral = rng.random(game_count) < 0.12
        game_pair = np.repeat(np.arange(game_count), rank_pairs_per_game)
        home_ranks = rng.integers(1, 132, size=game_pair.size)
        away_ranks = rng.integers(1, 132, size=game_pair.size)
        x = (home_ranks - 0.5) / 131.0
        y = (away_ranks - 0.5) / 131.0
        pairing = np.full(game_pair.size, "fbs-fbs", dtype=object)
        home = np.repeat((~neutral).astype(float), rank_pairs_per_game)
        neutral_pair = np.repeat(neutral.astype(float), rank_pairs_per_game)
        matrix = design_matrix(x, y, pairing, home, neutral_pair, True)
        if matrix.shape[1] != len(beta):
            raise ValueError(
                "frozen likelihood design does not match synthetic FBS fixture"
            )
        base_locations = matrix @ beta
        context_home = rng.normal(size=(game_count, 1))
        context_away = rng.normal(size=(game_count, 1))
        raw_preprocessor = fit_context_preprocessor(
            context_home,
            context_away,
            np.arange(training_games),
            ("synthetic_context",),
        )
        all_context = raw_preprocessor.transform(context_home, context_away)
        true_context = all_context[:, 0] * effect
        locations = base_locations + true_context[game_pair]
        selected_pair = rng.integers(0, rank_pairs_per_game, size=game_count)
        latent_location = locations[
            np.arange(game_count) * rank_pairs_per_game + selected_pair
        ]
        margins = np.rint(
            latent_location
            + student_t.rvs(
                degrees_of_freedom,
                size=game_count,
                random_state=rng,
            )
            * baseline_scale
        )
        data = LikelihoodData(
            game_ids=tuple(
                f"synthetic-{scenario}-{index}" for index in range(game_count)
            ),
            seasons=np.full(game_count, 2022, dtype=int),
            margins=np.asarray(margins, dtype=float),
            home_field=(~neutral).astype(float),
            pair_game=game_pair,
            base_locations=base_locations,
            context_home=context_home,
            context_away=context_away,
            metadata=tuple({} for _ in range(game_count)),
        )
        train = np.arange(training_games)
        evaluation = np.arange(training_games, game_count)
        train_preprocessor, train_context = _fit_context_preprocessor(
            data, train, ("synthetic_context",)
        )
        evaluation_context = train_preprocessor.transform(
            context_home[evaluation], context_away[evaluation]
        )
        generic = fit_residual_calibration(
            make_likelihood_slice(data, train, np.zeros((training_games, 0))),
            baseline_scale=baseline_scale,
            degrees_of_freedom=degrees_of_freedom,
        )
        context_fit = fit_residual_calibration(
            make_likelihood_slice(data, train, train_context),
            baseline_scale=baseline_scale,
            degrees_of_freedom=degrees_of_freedom,
            regularization=DEFAULT_REGULARIZATION,
        )
        generic_nll, _ = _score_nll_for_indices(
            data,
            evaluation,
            np.zeros((evaluation_games, 0)),
            fit=generic,
            baseline_scale=baseline_scale,
            degrees_of_freedom=degrees_of_freedom,
        )
        context_nll, _ = _score_nll_for_indices(
            data,
            evaluation,
            evaluation_context,
            fit=context_fit,
            baseline_scale=baseline_scale,
            degrees_of_freedom=degrees_of_freedom,
        )
        coefficient = float(context_fit.context_coefficients[0])
        passed = (
            context_fit.optimizer_success
            and generic.optimizer_success
            and (effect == 0.0 or (coefficient > 0 and context_nll < generic_nll))
        )
        results["cases"][scenario] = {
            "status": "passed" if passed else "failed",
            "seed": scenario_seed,
            "known_context_effect_points": effect,
            "fitted_context_effect_points": coefficient,
            "generic_calibration_held_out_nll": generic_nll,
            "context_calibration_held_out_nll": context_nll,
            "context_minus_generic_nll": context_nll - generic_nll,
            "generic_optimizer": generic.metadata(())["optimizer"],
            "context_optimizer": context_fit.metadata(
                ("synthetic_context", "missing__synthetic_context")
            )["optimizer"],
            "selection": "fixed lambda 0.1; no simulation tuning",
            "interpretation": "implementation validation only; no historical or causal claim",
        }
    results["status"] = (
        "passed"
        if all(case["status"] == "passed" for case in results["cases"].values())
        else "failed"
    )
    return results


def _feature_missingness_summary(data: LikelihoodData) -> list[dict[str, object]]:
    rows = []
    for index, name in enumerate(CONTEXT_FEATURES):
        home_missing = ~np.isfinite(data.context_home[:, index])
        away_missing = ~np.isfinite(data.context_away[:, index])
        rows.append(
            {
                "feature": name,
                "included_games": data.game_count,
                "home_missing_count": int(np.sum(home_missing)),
                "away_missing_count": int(np.sum(away_missing)),
                "any_side_missing_game_count": int(np.sum(home_missing | away_missing)),
                "missingness_indicator": f"missing__{name}",
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _required_paths(args)
    input_records, missing = _file_records(paths)
    baseline_path = paths["issue200_existing_likelihood"][0]
    synthetic_controls = None
    if baseline_path.is_file():
        beta, baseline_scale, df, _baseline_artifact = _load_baseline(baseline_path)
        synthetic_controls = run_synthetic_controls(beta, baseline_scale, df)
        write_json(output_root / "synthetic_controls.json", synthetic_controls)
    if missing:
        gate = _input_gate_report(args, input_records, missing, synthetic_controls)
        write_json(output_root / "input_gate.json", gate)
        manifest = {
            "artifact_kind": "issue_195_likelihood_diagnostic_manifest",
            "status": "blocked_missing_inputs",
            "input_records": input_records,
            "missing_required_inputs": missing,
            "historical_scoring_performed": False,
            "scientific_decision": None,
            "synthetic_controls_status": (
                synthetic_controls["status"]
                if synthetic_controls
                else "skipped_missing_likelihood"
            ),
            "outputs": {},
        }
        _write_blocked_report(output_root, gate, synthetic_controls)
        output_paths = sorted(path for path in output_root.iterdir() if path.is_file())
        manifest["outputs"] = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in output_paths
            if path.name != "manifest.json"
        }
        write_json(output_root / "manifest.json", manifest)
        return manifest

    issue200_input_records = _load_and_validate_provenance(paths, input_records)
    beta, baseline_scale, degrees_of_freedom, _baseline_artifact = _load_baseline(
        paths["issue200_existing_likelihood"][0]
    )
    feature_rows = _read_context_features(
        paths["issue200_context_feature_inputs"][0],
        paths["issue200_preseason_feature_manifest"][0],
        target_seasons=(*RESIDUAL_SEASONS, 2025),
    )
    common_games = _read_issue201_games(paths["issue201_forecast_scores"][0])
    candidate_ids = {str(row["id"]) for row in common_games}
    target_rows = _read_targets(paths["issue204_corrected_final_targets"][0])
    targets = _target_index(target_rows)
    rank_lookup, pair_rows_by_game = _read_rank_pair_lookup(
        paths["issue204_corrected_final_rank_pairs"][0], candidate_ids, targets
    )
    exclusions: list[dict[str, object]] = []
    eligible_games: list[dict[str, object]] = []
    for game in common_games:
        game_id, season = str(game["id"]), int(game["season"])
        missing_reason = ""
        for side, team_id in (
            ("home", str(game["homeId"])),
            ("away", str(game["awayId"])),
        ):
            if (season, "fbs", team_id) not in targets:
                missing_reason = f"missing_corrected_final_rank_target_{side}"
                break
            if (season, team_id) not in feature_rows:
                missing_reason = f"missing_issue200_context_feature_row_{side}"
                break
        if missing_reason:
            exclusions.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "phase": game["phase"],
                    "included": False,
                    "exclusion_reason": missing_reason,
                }
            )
        else:
            eligible_games.append(game)
    eligible_ids = {str(row["id"]) for row in eligible_games}
    # Keep pair lookup and game records on the same common population before invoking
    # the exact historical join/fallback implementation.
    filtered_rank_lookup = {
        key: value for key, value in rank_lookup.items() if str(key[0]) in eligible_ids
    }
    distributions = [
        row for row in target_rows if int(row["season"]) in (*RESIDUAL_SEASONS, 2025)
    ]
    joined, _missing_rank = historical_builder.build_joined(
        [
            {
                "id": game["id"],
                "season": game["season"],
                "week": game["week"],
                "startDate": game["startDate"],
                "homeId": game["homeId"],
                "awayId": game["awayId"],
                "homeClassification": "fbs",
                "awayClassification": "fbs",
                "homePoints": game["homePoints"],
                "awayPoints": game["awayPoints"],
                "neutralSite": game["neutralSite"],
                "homeTeam": game["homeTeam"],
                "awayTeam": game["awayTeam"],
            }
            for game in eligible_games
        ],
        distributions,
        filtered_rank_lookup,
        [],
    )
    joined_ids = {str(row["game_id"]) for row in joined}
    if joined_ids != eligible_ids:
        for game in eligible_games:
            game_id = str(game["id"])
            if game_id not in joined_ids:
                exclusions.append(
                    {
                        "game_id": game_id,
                        "season": int(game["season"]),
                        "phase": game["phase"],
                        "included": False,
                        "exclusion_reason": "missing_rank_or_score_after_historical_join",
                    }
                )
        eligible_games = [
            game for game in eligible_games if str(game["id"]) in joined_ids
        ]
        if not eligible_games:
            raise ValueError(
                "no #201 common games remain after corrected rank-pair join"
            )
    joined = [
        row
        for row in joined
        if str(row["game_id"]) in {str(g["id"]) for g in eligible_games}
    ]
    data = _build_likelihood_data(eligible_games, joined, feature_rows, beta)
    if data.game_count != len(eligible_games):
        raise AssertionError("diagnostic arms do not share one game population")

    fit_records: list[dict[str, object]] = [
        {
            "season": 2022,
            "role": "residual_training_only_not_held_out_evaluation",
            "target_season": 2023,
            "n_games": int(np.sum(data.seasons == 2022)),
        }
    ]
    per_game_rows: list[dict[str, object]] = []
    model_coefficients: list[dict[str, object]] = []
    for target_season in TARGET_SEASONS:
        train = training_indices(data, target_season=target_season)
        evaluate = np.flatnonzero(data.seasons == target_season)
        if evaluate.size == 0:
            raise ValueError(f"{target_season}: no eligible #201 games remain")
        train_preprocessor, train_context = _fit_context_preprocessor(
            data, train, CONTEXT_FEATURES
        )
        evaluation_context = train_preprocessor.transform(
            data.context_home[evaluate], data.context_away[evaluate]
        )
        regularization, selection = select_context_regularization(
            data,
            train,
            baseline_scale=baseline_scale,
            degrees_of_freedom=degrees_of_freedom,
        )
        generic_view = make_likelihood_slice(data, train, np.zeros((train.size, 0)))
        context_view = make_likelihood_slice(data, train, train_context)
        generic_fit = fit_residual_calibration(
            generic_view,
            baseline_scale=baseline_scale,
            degrees_of_freedom=degrees_of_freedom,
        )
        context_fit = fit_residual_calibration(
            context_view,
            baseline_scale=baseline_scale,
            degrees_of_freedom=degrees_of_freedom,
            regularization=regularization,
        )
        evaluation_context = make_likelihood_slice(data, evaluate, evaluation_context)
        baseline_rows = _evaluate_arm(
            data,
            evaluate,
            np.zeros((evaluate.size, 0)),
            fit=None,
            scale=baseline_scale,
            degrees_of_freedom=degrees_of_freedom,
            arm="rank_only",
        )
        generic_rows = _evaluate_arm(
            data,
            evaluate,
            np.zeros((evaluate.size, 0)),
            fit=generic_fit,
            scale=baseline_scale,
            degrees_of_freedom=degrees_of_freedom,
            arm="rank_site_calibration",
        )
        context_rows = _evaluate_arm(
            data,
            evaluate,
            evaluation_context.context,
            fit=context_fit,
            scale=baseline_scale,
            degrees_of_freedom=degrees_of_freedom,
            arm="rank_site_context",
        )
        if not (
            {str(row["game_id"]) for row in baseline_rows}
            == {str(row["game_id"]) for row in generic_rows}
            == {str(row["game_id"]) for row in context_rows}
        ):
            raise AssertionError(
                f"{target_season}: diagnostic arms have different games"
            )
        per_game_rows.extend(baseline_rows + generic_rows + context_rows)
        training_seasons = sorted({int(value) for value in data.seasons[train]})
        fit_record = {
            "target_season": target_season,
            "role": "held_out_evaluation",
            "training_seasons": training_seasons,
            "n_training_games": int(train.size),
            "n_evaluation_games": int(evaluate.size),
            "target_season_excluded_from_fit_and_regularization_selection": bool(
                np.all(data.seasons[train] < target_season)
                and np.all(data.seasons[evaluate] == target_season)
            ),
            "regularization_selection": selection,
            "context_preprocessor": train_preprocessor.metadata(),
            "rank_site_calibration_fit": generic_fit.metadata(()),
            "rank_site_context_fit": context_fit.metadata(
                (*CONTEXT_FEATURES, *MISSINGNESS_FEATURES)
            ),
            "frozen_baseline": {
                "artifact": "issue-200 existing-policy likelihood",
                "scale": baseline_scale,
                "degrees_of_freedom": degrees_of_freedom,
                "feature_count": len(beta),
                "refit": False,
            },
        }
        fit_records.append(fit_record)
        for arm, fit in (
            ("rank_site_calibration", generic_fit),
            ("rank_site_context", context_fit),
        ):
            for name, coefficient in zip(
                (*CONTEXT_FEATURES, *MISSINGNESS_FEATURES),
                fit.context_coefficients,
                strict=False,
            ):
                model_coefficients.append(
                    {
                        "target_season": target_season,
                        "trained_through_season": max(training_seasons),
                        "arm": arm,
                        "feature": name,
                        "coefficient_points_per_training_standard_deviation": float(
                            coefficient
                        ),
                        "regularization": fit.regularization,
                    }
                )

    season_summary, comparison_rows, phase_and_concentration = _comparison_outputs(
        per_game_rows
    )
    phase_rows = [row for row in phase_and_concentration if "phase" in row]
    concentration_rows = [row for row in phase_and_concentration if "direction" in row]
    decision = _decision(comparison_rows)
    all_candidate_ids = {str(row["id"]) for row in common_games}
    included_ids = {str(row["id"]) for row in eligible_games}
    for game in common_games:
        game_id = str(game["id"])
        if game_id in included_ids:
            joined_row = next(row for row in joined if str(row["game_id"]) == game_id)
            rank_pair_rows = pair_rows_by_game.get(game_id, [])
            rank_pairs_json = json.loads(joined_row["rank_pairs"])
            exclusions.append(
                {
                    "game_id": game_id,
                    "season": int(game["season"]),
                    "phase": game["phase"],
                    "included": True,
                    "pairing_method": joined_row["pairing_method"],
                    "rank_pair_count": len(rank_pairs_json),
                    "observed_same_system_pair_count": len(rank_pair_rows),
                    "rank_pairs_sha256": sha256_json(rank_pairs_json),
                    "army_navy": game["army_navy"],
                    "exclusion_reason": "",
                }
            )
    game_manifest = sorted(exclusions, key=lambda row: int(str(row["game_id"])))
    if len(all_candidate_ids) != len(common_games):
        raise AssertionError("duplicate #201 common game ids")
    manifest = {
        "artifact_kind": "issue_195_likelihood_diagnostic_manifest",
        "status": "complete",
        "interpretation": "exploratory ex-post likelihood mechanism diagnostic",
        "input_records": input_records,
        "baseline": {
            "artifact": "issue-200 existing-policy likelihood",
            "artifact_sha256": sha256_file(paths["issue200_existing_likelihood"][0]),
            "policy": "existing",
            "training_seasons": "2003-2021",
            "scale": baseline_scale,
            "degrees_of_freedom": degrees_of_freedom,
            "historical_v1_reference": "retained by #200 provenance only; not scored in the primary comparisons",
        },
        "population": {
            "candidate_common_issue201_games": len(common_games),
            "included_games": len(eligible_games),
            "additional_exclusions": len(common_games) - len(eligible_games),
            "seasons": [*RESIDUAL_SEASONS, 2025],
            "evaluation_seasons": list(TARGET_SEASONS),
            "2022_role": "residual training only, not held-out evaluation",
            "game_id_sha256": sha256_json(sorted(included_ids, key=int)),
            "arms_use_identical_game_ids": True,
            "issue201_phase_values": sorted(
                {str(row["phase"]) for row in common_games}
            ),
            "army_navy_game_count": sum(bool(row["army_navy"]) for row in common_games),
            "pairing_method_counts": {
                method: sum(
                    1 for game in data.metadata if game["pairing_method"] == method
                )
                for method in sorted(
                    {str(game["pairing_method"]) for game in data.metadata}
                )
            },
        },
        "features": {
            "source": "issue-200 repaired Context 1.4 predictor inputs",
            "feature_names": list(CONTEXT_FEATURES),
            "missingness_indicator_names": list(MISSINGNESS_FEATURES),
            "preprocessing": "fit location/scale and missingness rates on both sides of earlier training games; impute missing values to training means; score home-minus-away standardized differences",
            "missingness_summary": _feature_missingness_summary(data),
            "raw_input_feature_file_sha256": sha256_file(
                paths["issue200_context_feature_inputs"][0]
            ),
            "preseason_feature_manifest_sha256": sha256_file(
                paths["issue200_preseason_feature_manifest"][0]
            ),
        },
        "likelihood": {
            "objective": "sum of marginalized per-game mixture NLL plus half-lambda L2 Context coefficient norm",
            "rank_pair_weighting": "each game contributes one log-mean-density regardless of constituent pair count",
            "student_t_df": degrees_of_freedom,
            "site_term": "additional home term multiplied by 1-neutral; zero on neutral sites",
            "scale_term": "one common exp(log_scale_correction) multiplier",
            "mean_term": "positive overall multiplier on frozen conditional mean plus home term and Context home-minus-away correction",
            "selection_grid": list(REGULARIZATION_GRID),
            "decision": decision,
            "convergence_checks": "analytic gradients, L-BFGS-B termination state, gradient infinity norm, finite positive scale, and finite output scores are recorded",
        },
        "issue200_manifest_sha256": sha256_file(paths["issue200_manifest"][0]),
        "issue201_manifest_sha256": sha256_file(paths["issue201_manifest"][0]),
        "historical_modeling_builder_sha256": sha256_file(
            SCRIPT_DIR / "build_historical_modeling.py"
        ),
        "original_v1_baseline_reference_sha256": issue200_input_records.get(
            "baseline_historical_likelihood", {}
        ).get("sha256"),
        "outputs": {},
    }
    write_csv(
        args.output_root / "game_manifest.csv",
        (
            "game_id",
            "season",
            "phase",
            "included",
            "pairing_method",
            "rank_pair_count",
            "observed_same_system_pair_count",
            "rank_pairs_sha256",
            "army_navy",
            "exclusion_reason",
        ),
        game_manifest,
    )
    write_csv(
        args.output_root / "per_game_scores.csv",
        (
            "game_id",
            "season",
            "phase",
            "army_navy",
            "home_team_id",
            "away_team_id",
            "home_team",
            "away_team",
            "neutral_site",
            "actual_home_margin",
            "pairing_method",
            "rank_pair_count",
            "arm",
            "margin_log_density",
            "mixture_nll",
            "predicted_margin_mean",
            "signed_margin_residual",
            "mixture_pit_midpoint",
            "interval_50_low",
            "interval_50_high",
            "interval_50_width",
            "interval_50_covered",
            "interval_80_low",
            "interval_80_high",
            "interval_80_width",
            "interval_80_covered",
            "interval_95_low",
            "interval_95_high",
            "interval_95_width",
            "interval_95_covered",
        ),
        per_game_rows,
    )
    write_csv(
        args.output_root / "season_summary.csv",
        tuple(dict.fromkeys(key for row in season_summary for key in row)),
        season_summary,
    )
    write_csv(
        args.output_root / "comparison_summary.csv",
        tuple(dict.fromkeys(key for row in comparison_rows for key in row)),
        comparison_rows,
    )
    write_csv(
        args.output_root / "phase_summary.csv",
        tuple(dict.fromkeys(key for row in phase_rows for key in row)),
        phase_rows,
    )
    write_csv(
        args.output_root / "gain_concentration.csv",
        tuple(dict.fromkeys(key for row in concentration_rows for key in row)),
        concentration_rows,
    )
    write_csv(
        args.output_root / "feature_missingness.csv",
        (
            "feature",
            "included_games",
            "home_missing_count",
            "away_missing_count",
            "any_side_missing_game_count",
            "missingness_indicator",
        ),
        _feature_missingness_summary(data),
    )
    write_csv(
        args.output_root / "fitted_context_coefficients.csv",
        (
            "target_season",
            "trained_through_season",
            "arm",
            "feature",
            "coefficient_points_per_training_standard_deviation",
            "regularization",
        ),
        model_coefficients,
    )
    write_json(args.output_root / "fit_records.json", fit_records)
    write_json(args.output_root / "decision.json", decision)
    write_json(
        args.output_root / "results.json",
        {"decision": decision, "comparisons": comparison_rows},
    )
    report = [
        "# Issue 195: rank-only likelihood Context diagnostic",
        "",
        f"**Decision: {decision['classification']}.**",
        "",
        "This is a fixed ex-post game-margin mechanism diagnostic. The primary comparison is the rank/site/Context arm against the same rank/site generic calibration control; generic calibration versus the frozen rank-only likelihood is reported separately.",
        "",
        f"- Corrected #204 final-rank/common #201 cohort: {len(eligible_games):,} games after {len(common_games) - len(eligible_games):,} additional exclusions.",
        f"- Held-out evaluation seasons: {', '.join(map(str, TARGET_SEASONS))}; 2022 is residual training only.",
        f"- Context versus generic calibration: pooled equal-game ΔNLL {decision['context_pooled_equal_game_delta_nll']:.6f}, equal-season ΔNLL {decision['context_equal_season_delta_nll']:.6f}; negative favors Context.",
        f"- Generic calibration versus frozen likelihood: pooled equal-game ΔNLL {decision['calibration_pooled_equal_game_delta_nll']:.6f}, equal-season ΔNLL {decision['calibration_equal_season_delta_nll']:.6f}; negative favors recalibration.",
        "",
        "## Interpretation limits",
        "",
        "Corrected final ranking measurements incorporate regular-season outcomes, potentially including the game being diagnosed, and postseason information. They are noisy measurements rather than latent rank truth. A Context association may reflect missing game-result structure, ordinary calibration, or rank measurement error/endogeneity; it does not directly test conditional independence given true ranks. These 2022–2025 seasons have informed earlier research and remain exploratory development data. Repeated teams and seasons are not independent confirmation. The negative result, if any, addresses only this fixed linear mean correction.",
        "",
        "The ex-post margin diagnostic does not replace future-game regular-season CRPS as the operational comparison measure or common pre-bowl ranking NLL as a secondary prospective measure. No likelihood, ranking, inference setting, website default, or published snapshot is changed.",
        "",
        "## Reproduction",
        "",
        "```sh",
        "uv run python scripts/run_issue_195_likelihood_diagnostic.py --corpus-root CORPUS_ROOT --issue200-root ISSUE200_OUTPUT_ROOT --issue201-root data/research/issue_201_prebowl_forecast --output-root data/research/issue_195_likelihood_diagnostic",
        "```",
        "",
        "The existing `scripts/build_historical_modeling.py` join and pseudo-data builders construct the corrected final-rank mixtures. `game_manifest.csv` binds the #201 game IDs, final-rank pair mixtures, phase labels, pair counts, and exclusions. `fit_records.json` records forward training seasons, regularization selection, preprocessing, coefficients, objective, and convergence checks.",
        "",
    ]
    (args.output_root / "report.md").write_text("\n".join(report), encoding="utf-8")
    output_paths = sorted(path for path in args.output_root.iterdir() if path.is_file())
    manifest["outputs"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in output_paths
        if path.name != "manifest.json"
    }
    write_json(args.output_root / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=ROOT)
    parser.add_argument(
        "--issue200-root",
        type=Path,
        default=ROOT / "data/research/matched_history_context_issue_200",
    )
    parser.add_argument(
        "--issue201-root",
        type=Path,
        default=ROOT / "data/research/issue_201_prebowl_forecast",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/research/issue_195_likelihood_diagnostic",
    )
    return parser.parse_args()
