"""Research-only score-margin likelihood candidates.

The production Historical Likelihood V1 is deliberately not changed by this
module.  It provides the small, auditable pieces needed to ask whether the
same V1 mean surface should use a non-constant observation scale or a smooth
blowout transform.

All rank coordinates are the V1 coordinates: same-subdivision games use the
home-minus-away orientation and cross-subdivision games use FBS-minus-FCS.
Total points is accepted only by :func:`scale_design`; it can therefore alter
uncertainty but never the location surface.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import gammaln
from scipy.stats import t as student_t

from gippyrank.modeling import PAIRINGS, design_matrix, fit_robust_surface

DF = 15.0
BLOWOUT_KS = (14.0, 21.0, 28.0, 42.0)
CANDIDATES = ("v1", "a", "b", "c14", "c21", "c28", "c42")
_SCALE_PAIRINGS = PAIRINGS


@dataclass(frozen=True)
class MarginData:
    """Rank-pair pseudo-observations with equal total weight per game."""

    x: np.ndarray
    y: np.ndarray
    margin: np.ndarray
    total_points: np.ndarray
    pairing: np.ndarray
    home: np.ndarray
    neutral: np.ndarray
    fbs_home: np.ndarray
    weight: np.ndarray
    game_id: np.ndarray
    season: np.ndarray

    def __len__(self) -> int:
        return len(self.margin)


def _as_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _json_pairs(value: object) -> list[tuple[float, float]]:
    if isinstance(value, str):
        value = json.loads(value)
    return [(float(pair[0]), float(pair[1])) for pair in (value or [])]


def pairing_for(home_subdivision: str, away_subdivision: str) -> str:
    home = home_subdivision.casefold()
    away = away_subdivision.casefold()
    if home not in {"fbs", "fcs"} or away not in {"fbs", "fcs"}:
        raise ValueError(f"unsupported subdivisions: {home_subdivision}, {away_subdivision}")
    return "fbs-fcs" if home != away else f"{home}-{away}"


def oriented_margin(
    home_subdivision: str,
    away_subdivision: str,
    home_points: int,
    away_points: int,
) -> float:
    """Return the V1 response orientation, including FBS-first cross games."""

    home = home_subdivision.casefold()
    away = away_subdivision.casefold()
    if home == away or home == "fbs":
        return float(home_points - away_points)
    if away == "fbs":
        return float(away_points - home_points)
    raise ValueError("cross-subdivision games must contain one FBS team")


def oriented_rank_coordinates(
    home_subdivision: str,
    away_subdivision: str,
    home_rank: float,
    away_rank: float,
    home_population: int,
    away_population: int,
) -> tuple[float, float]:
    """Return percentile coordinates in stable FBS/FCS orientation."""

    home_coordinate = (float(home_rank) - 0.5) / int(home_population)
    away_coordinate = (float(away_rank) - 0.5) / int(away_population)
    if (
        home_subdivision.casefold() != away_subdivision.casefold()
        and home_subdivision.casefold() == "fcs"
    ):
        return away_coordinate, home_coordinate
    return home_coordinate, away_coordinate


def build_margin_data(rows: Sequence[Mapping[str, object]]) -> MarginData:
    """Expand historical games into the existing V1 rank-pair machinery."""

    values: list[list[object]] = [[] for _ in range(11)]
    for row in rows:
        pairing = pairing_for(
            str(row["home_subdivision"]), str(row["away_subdivision"])
        )
        pairs = _json_pairs(row.get("rank_pairs"))
        if not pairs:
            continue
        home_subdivision = str(row["home_subdivision"]).casefold()
        away_subdivision = str(row["away_subdivision"]).casefold()
        neutral = float(_as_bool(row.get("neutral_site")))
        cross = home_subdivision != away_subdivision
        fbs_home = float(cross and home_subdivision == "fbs" and not neutral)
        margin = oriented_margin(
            home_subdivision,
            away_subdivision,
            int(row["home_points"]),
            int(row["away_points"]),
        )
        n = len(pairs)
        for home_rank, away_rank in pairs:
            x, y = oriented_rank_coordinates(
                home_subdivision,
                away_subdivision,
                home_rank,
                away_rank,
                int(row["home_team_population"]),
                int(row["away_team_population"]),
            )
            item = [
                x,
                y,
                margin,
                int(row["home_points"]) + int(row["away_points"]),
                pairing,
                1.0 - neutral,
                neutral,
                fbs_home,
                1.0 / n,
                str(row["game_id"]),
                int(row["season"]),
            ]
            for output, value in zip(values, item):
                output.append(value)
    arrays = tuple(np.asarray(value) for value in values)
    return MarginData(*arrays)


def candidate_name(kind: str, k: float | None = None) -> str:
    kind = kind.casefold()
    if kind == "v1" or kind in {"a", "b"}:
        return kind
    if kind == "c":
        if k not in BLOWOUT_KS:
            raise ValueError(f"Candidate C k must be one of {BLOWOUT_KS}")
        return f"c{int(k)}"
    if kind in CANDIDATES:
        return kind
    raise ValueError(f"unknown margin-likelihood candidate: {kind}")


def blowout_transform(margin: np.ndarray | float, k: float) -> np.ndarray:
    """Apply the declared smooth, symmetric, invertible blowout transform."""

    if k <= 0:
        raise ValueError("blowout transform scale k must be positive")
    return k * np.arcsinh(np.asarray(margin, dtype=float) / k)


def blowout_inverse(value: np.ndarray | float, k: float) -> np.ndarray:
    if k <= 0:
        raise ValueError("blowout transform scale k must be positive")
    return k * np.sinh(np.asarray(value, dtype=float) / k)


def blowout_log_jacobian(margin: np.ndarray | float, k: float) -> np.ndarray:
    """Return log |d g_k(margin) / d margin| for original-scale scoring."""

    if k <= 0:
        raise ValueError("blowout transform scale k must be positive")
    return -0.5 * np.log1p((np.asarray(margin, dtype=float) / k) ** 2)


def student_t_logpdf(
    value: np.ndarray | float,
    location: np.ndarray,
    scale: np.ndarray | float,
    degrees_of_freedom: float = DF,
) -> np.ndarray:
    scale_array = np.asarray(scale, dtype=float)
    if np.any(scale_array <= 0) or degrees_of_freedom <= 0:
        raise ValueError("Student-t scale and df must be positive")
    value_array = np.asarray(value, dtype=float)
    z = (value_array - location) / scale_array
    constant = (
        gammaln((degrees_of_freedom + 1) / 2)
        - gammaln(degrees_of_freedom / 2)
        - 0.5 * np.log(degrees_of_freedom * np.pi)
        - np.log(scale_array)
    )
    return constant - (degrees_of_freedom + 1) / 2 * np.log1p(
        z * z / degrees_of_freedom
    )


def mean_design(data: MarginData) -> np.ndarray:
    """Return exactly the frozen V1 34-column mean-feature family."""

    return design_matrix(
        data.x,
        data.y,
        data.pairing,
        data.home,
        data.neutral,
        surface=True,
        fbs_home=data.fbs_home,
    )


def _one_hot_pairing(pairing: np.ndarray) -> np.ndarray:
    return np.column_stack([pairing == name for name in _SCALE_PAIRINGS]).astype(float)


def standardize_log_total_points(
    total_points: np.ndarray,
    *,
    mean: float | None = None,
    scale: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Standardize log1p(total points), returning the fitted parameters."""

    values = np.log1p(np.asarray(total_points, dtype=float))
    fitted_mean = float(np.mean(values) if mean is None else mean)
    fitted_scale = float(np.std(values, ddof=0) if scale is None else scale)
    if not np.isfinite(fitted_mean) or not np.isfinite(fitted_scale):
        raise ValueError("total-points normalization must be finite")
    fitted_scale = max(fitted_scale, 1.0e-12)
    return (values - fitted_mean) / fitted_scale, {
        "mean": fitted_mean,
        "scale": fitted_scale,
        "source": "training_rows_only",
    }


def scale_design(
    pairing: np.ndarray,
    expected_margin: np.ndarray,
    *,
    total_points: np.ndarray | None = None,
    total_points_normalization: Mapping[str, float] | None = None,
    include_total_points: bool = False,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, float] | None]:
    """Build Candidate A/B log-scale features.

    ``expected_margin`` is a V1 rank/site hypothesis location.  Its absolute
    value is used only as a mismatch magnitude, so reversing an FBS/FCS game
    cannot reverse the scale.  The response/location is never included here.
    """

    pairing = np.asarray(pairing)
    expected_margin = np.asarray(expected_margin, dtype=float)
    if len(pairing) != len(expected_margin):
        raise ValueError("pairing and expected-margin arrays must have equal lengths")
    base = _one_hot_pairing(pairing)
    mismatch = np.log1p(np.abs(expected_margin))
    columns = [base, mismatch[:, None]]
    names = tuple([f"pairing:{name}:intercept" for name in _SCALE_PAIRINGS] + [
        "log1p_abs_v1_expected_margin"
    ])
    normalization: dict[str, float] | None = None
    if include_total_points:
        if total_points is None:
            raise ValueError("total points are required for Candidate B/C scale context")
        normalized, normalization = standardize_log_total_points(
            total_points,
            mean=None
            if total_points_normalization is None
            else float(total_points_normalization["mean"]),
            scale=None
            if total_points_normalization is None
            else float(total_points_normalization["scale"]),
        )
        columns.append(normalized[:, None])
        names = (*names, "standardized_log1p_total_points")
    return np.column_stack(columns), names, normalization


def scale_values(
    parameters: np.ndarray,
    scale_features: np.ndarray,
    *,
    minimum: float = 1.0e-8,
) -> np.ndarray:
    """Exponentiate log-scale parameters and enforce strictly positive scale."""

    values = np.exp(np.asarray(scale_features, dtype=float) @ np.asarray(parameters))
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise FloatingPointError("candidate scale is non-finite or non-positive")
    return np.maximum(values, minimum)


def fit_variable_scale(
    X: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    scale_features: np.ndarray,
    *,
    fixed_beta: np.ndarray | None = None,
    degrees_of_freedom: float = DF,
    maxiter: int = 120,
) -> dict[str, object]:
    """Fit a positive log-scale model around a fixed or refit mean surface.

    Candidate A/B pass the leakage-safe V1 beta through ``fixed_beta`` so that
    their comparison tests heteroskedastic scale only.  Candidate C leaves it
    unset and refits the same mean-feature family on its transformed response
    with the production-style Student-t IRLS fit.
    """

    X = np.asarray(X, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)
    scale_features = np.asarray(scale_features, dtype=float)
    if X.ndim != 2 or scale_features.ndim != 2 or len(X) != len(target):
        raise ValueError("candidate fit arrays have incompatible shapes")
    if len(weights) != len(target) or len(scale_features) != len(target):
        raise ValueError("candidate fit arrays have incompatible lengths")
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("candidate weights must be finite and non-negative")

    if fixed_beta is None:
        mean_fit = fit_robust_surface(
            X,
            target,
            weights,
            iterations=8,
            df=degrees_of_freedom,
        )
        beta = np.asarray(mean_fit["beta"], dtype=float)
        mean_fit_description = "student_t_irls_refit_on_candidate_response"
    else:
        beta = np.asarray(fixed_beta, dtype=float)
        if beta.ndim != 1 or len(beta) != X.shape[1] or not np.isfinite(beta).all():
            raise ValueError("fixed beta must be a finite vector matching the mean design")
        mean_fit_description = "leakage_safe_v1_beta_fixed"
    total_weight = float(weights.sum())
    residual = target - X @ beta
    initial_scale = max(float(np.sqrt(np.average(residual * residual, weights=weights))), 1.0)
    alpha = np.zeros(scale_features.shape[1], dtype=float)
    alpha[: len(_SCALE_PAIRINGS)] = np.log(initial_scale)

    def objective_gradient(current_alpha: np.ndarray) -> tuple[float, np.ndarray]:
        log_scale = scale_features @ current_alpha
        log_scale = np.clip(log_scale, -20.0, 20.0)
        scale = np.exp(log_scale)
        denominator = degrees_of_freedom * scale * scale + residual * residual
        nll = (
            log_scale
            + 0.5 * np.log(degrees_of_freedom * np.pi)
            + gammaln(degrees_of_freedom / 2)
            - gammaln((degrees_of_freedom + 1) / 2)
            + (degrees_of_freedom + 1)
            / 2
            * np.log1p(residual * residual / (degrees_of_freedom * scale * scale))
        )
        objective_value = float(np.sum(weights * nll) / total_weight)
        scale_gradient = 1.0 - (degrees_of_freedom + 1) * residual * residual / denominator
        gradient = scale_features.T @ (weights * scale_gradient) / total_weight
        return objective_value, gradient

    result = minimize(
        lambda current_alpha: objective_gradient(current_alpha)[0],
        alpha,
        jac=lambda current_alpha: objective_gradient(current_alpha)[1],
        method="L-BFGS-B",
        bounds=[(-20.0, 20.0)] * scale_features.shape[1],
        options={"maxiter": maxiter, "ftol": 1.0e-9},
    )
    alpha = result.x
    return {
        "beta": beta,
        "scale_parameters": alpha,
        "df": float(degrees_of_freedom),
        "scale_model": (
            "pairing_intercepts_plus_log1p_abs_v1_expected_margin"
            + (
                "+standardized_log1p_total_points"
                if scale_features.shape[1] == 5
                else ""
            )
        ),
        "scale_feature_names": list(
            [f"pairing:{name}:intercept" for name in _SCALE_PAIRINGS]
            + ["log1p_abs_v1_expected_margin"]
            + (["standardized_log1p_total_points"] if scale_features.shape[1] == 5 else [])
        ),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "mean_fit": mean_fit_description,
        "objective": "weighted_pseudo_student_t",
    }


def fit_constant_scale(
    X: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    degrees_of_freedom: float = DF,
) -> dict[str, object]:
    """Fit a constant-scale comparison model without modifying production V1."""

    weights = np.asarray(weights, dtype=float)
    weighted_X = np.asarray(X, dtype=float) * np.sqrt(weights)[:, None]
    weighted_target = np.asarray(target, dtype=float) * np.sqrt(weights)
    beta = np.linalg.lstsq(weighted_X, weighted_target, rcond=None)[0]
    residual = np.asarray(target, dtype=float) - np.asarray(X, dtype=float) @ beta
    scale = max(float(np.sqrt(np.average(residual * residual, weights=weights))), 1.0)
    return {
        "beta": beta,
        "scale": scale,
        "df": float(degrees_of_freedom),
        "scale_model": "constant",
        "scale_feature_names": [],
        "objective": "weighted_pseudo_student_t",
    }


def _model_kind_and_k(name: str) -> tuple[str, float | None]:
    name = candidate_name(name)
    if name == "v1" or name in {"a", "b"}:
        return name, None
    return "c", float(name[1:])


def fit_candidate(
    name: str,
    data: MarginData,
    X: np.ndarray,
    baseline_location: np.ndarray,
    fit_mask: np.ndarray,
    *,
    baseline_beta: np.ndarray | None = None,
    degrees_of_freedom: float = DF,
    total_points_normalization: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Fit one predeclared candidate using only ``fit_mask`` rows."""

    kind, k = _model_kind_and_k(name)
    target = data.margin if k is None else blowout_transform(data.margin, k)
    include_total = kind in {"b", "c"}
    features, feature_names, normalization = scale_design(
        data.pairing[fit_mask],
        baseline_location[fit_mask],
        total_points=data.total_points[fit_mask],
        total_points_normalization=total_points_normalization,
        include_total_points=include_total,
    ) if kind != "v1" else (None, (), None)
    if kind == "v1":
        frozen_beta = np.linalg.lstsq(
            X[fit_mask] * np.sqrt(data.weight[fit_mask])[:, None],
            baseline_location[fit_mask] * np.sqrt(data.weight[fit_mask]),
            rcond=None,
        )[0]
        return {
            "name": "v1",
            "kind": "v1",
            "beta": frozen_beta,
            "scale": 1.0,
            "df": float(degrees_of_freedom),
            "scale_model": "frozen_v1",
            "scale_feature_names": [],
        }
    fixed_beta = baseline_beta if kind in {"a", "b"} else None
    if kind in {"a", "b"} and fixed_beta is None:
        raise ValueError("Candidate A/B require the leakage-safe V1 beta")
    fit = fit_variable_scale(
        X[fit_mask],
        target[fit_mask],
        data.weight[fit_mask],
        features,
        fixed_beta=fixed_beta,
        degrees_of_freedom=degrees_of_freedom,
    )
    fit.update(
        {
            "name": candidate_name(kind, k),
            "kind": kind,
            "transform_k": k,
            "mean_feature_count": int(X.shape[1]),
            "scale_feature_names": list(feature_names),
            "total_points_normalization": normalization
            if include_total
            else None,
            "fit_rows": int(np.count_nonzero(fit_mask)),
            "fit_games": len(np.unique(data.game_id[fit_mask])),
            "fit_seasons": [int(value) for value in sorted(set(data.season[fit_mask]))],
        }
    )
    return fit


def model_location(model: Mapping[str, object], X: np.ndarray) -> np.ndarray:
    """Return a candidate location; scale context cannot affect it."""

    beta = np.asarray(model["beta"], dtype=float)
    if len(beta) != X.shape[1]:
        raise ValueError("candidate beta does not match the V1 mean design")
    return X @ beta


def model_scales(
    model: Mapping[str, object],
    data: MarginData,
    baseline_location: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    kind = str(model["kind"])
    if kind == "v1":
        return np.full(np.count_nonzero(mask), float(model["scale"]), dtype=float)
    normalization = model.get("total_points_normalization")
    features, _names, _ = scale_design(
        data.pairing[mask],
        baseline_location[mask],
        total_points=data.total_points[mask],
        total_points_normalization=normalization if isinstance(normalization, Mapping) else None,
        include_total_points=kind in {"b", "c"},
    )
    return scale_values(np.asarray(model["scale_parameters"], dtype=float), features)


def model_log_density(
    model: Mapping[str, object],
    raw_margin: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    kind = str(model["kind"])
    k = model.get("transform_k")
    if kind == "c":
        if k is None:
            raise ValueError("Candidate C is missing transform k")
        transformed = blowout_transform(raw_margin, float(k))
        return student_t_logpdf(transformed, locations, scales, float(model["df"])) + blowout_log_jacobian(raw_margin, float(k))
    return student_t_logpdf(raw_margin, locations, scales, float(model["df"]))


def mixture_interval(
    locations: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    df: float,
    level: float,
    *,
    transform_k: float | None = None,
) -> tuple[float, float]:
    """Central mixture interval, with optional mapping back to raw margins."""

    locations = np.asarray(locations, dtype=float)
    scales = np.asarray(scales, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    low = float(np.min(locations) - 40.0 * np.max(scales))
    high = float(np.max(locations) + 40.0 * np.max(scales))

    def cdf(value: float) -> float:
        return float(
            np.sum(weights * student_t.cdf((value - locations) / scales, df))
        )

    lower = brentq(lambda value: cdf(value) - (1.0 - level) / 2.0, low, high)
    upper = brentq(lambda value: cdf(value) - (1.0 + level) / 2.0, low, high)
    if transform_k is not None:
        return tuple(float(value) for value in blowout_inverse(np.array([lower, upper]), transform_k))  # type: ignore[return-value]
    return lower, upper


def mixture_interval_approx(
    locations: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    df: float,
    level: float,
    *,
    transform_k: float | None = None,
) -> tuple[float, float]:
    """Fast deterministic central interval for repeated research summaries.

    The exact mixture interval above is appropriate for small targeted
    checks.  Large historical panels use this moment-matched approximation:
    it preserves rank-pair dispersion and Student-t tail quantiles without
    doing three numerical roots for every game.  Original-scale Candidate C
    intervals are mapped through the exact inverse transform.
    """

    locations = np.asarray(locations, dtype=float)
    scales = np.asarray(scales, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    mean = float(np.dot(weights, locations))
    variance = float(
        np.dot(
            weights,
            (locations - mean) ** 2 + (df / (df - 2.0)) * scales * scales,
        )
    ) if df > 2.0 else float(np.dot(weights, scales * scales))
    effective_scale = np.sqrt(max(variance * (df - 2.0) / df, 1.0e-12))
    quantiles = student_t.ppf([(1.0 - level) / 2.0, (1.0 + level) / 2.0], df)
    interval = mean + effective_scale * quantiles
    if transform_k is not None:
        interval = blowout_inverse(interval, transform_k)
    return float(interval[0]), float(interval[1])


def _group_indices(game_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, inverse = np.unique(game_ids, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    return unique, inverse, order


def score_model(
    model: Mapping[str, object],
    data: MarginData,
    X: np.ndarray,
    baseline_location: np.ndarray,
    mask: np.ndarray,
    *,
    intervals: bool = True,
) -> dict[str, float | int | str]:
    """Score equal-weight whole-game likelihood and calibration metrics."""

    mask = np.asarray(mask, dtype=bool)
    locations = model_location(model, X[mask])
    scales = model_scales(model, data, baseline_location, mask)
    raw_margin = data.margin[mask]
    log_density = model_log_density(model, raw_margin, locations, scales)
    weights = data.weight[mask]
    unique, inverse, _order = _group_indices(data.game_id[mask])
    group_weight = np.bincount(inverse, weights=weights, minlength=len(unique))
    maxima = np.full(len(unique), -np.inf)
    np.maximum.at(maxima, inverse, log_density)
    marginal = maxima + np.log(
        np.bincount(
            inverse,
            weights=weights * np.exp(log_density - maxima[inverse]),
            minlength=len(unique),
        )
    ) - np.log(group_weight)
    prediction = np.bincount(inverse, weights=weights * (
        blowout_inverse(locations, float(model["transform_k"]))
        if str(model["kind"]) == "c"
        else locations
    ), minlength=len(unique)) / group_weight
    actual = np.bincount(inverse, weights=weights * raw_margin, minlength=len(unique)) / group_weight
    errors = prediction - actual
    output: dict[str, float | int | str] = {
        "marginalized_nll": float(-np.mean(marginal)),
        "expected_margin_mae": float(np.mean(np.abs(errors))),
        "n_games": len(unique),
        "n_pseudo_observations": int(np.count_nonzero(mask)),
    }
    if not intervals:
        return output
    interval_values: dict[float, list[bool]] = {0.5: [], 0.8: [], 0.95: []}
    widths: dict[float, list[float]] = {0.5: [], 0.8: [], 0.95: []}
    pair_weights = weights
    for group_index in range(len(unique)):
        selected = inverse == group_index
        transform_k = float(model["transform_k"]) if str(model["kind"]) == "c" else None
        for level, _values in interval_values.items():
            lower, upper = mixture_interval_approx(
                locations[selected],
                scales[selected],
                pair_weights[selected],
                float(model["df"]),
                level,
                transform_k=transform_k,
            )
            _values.append(lower <= actual[group_index] <= upper)
            widths[level].append(upper - lower)
    for level, values in interval_values.items():
        label = int(level * 100)
        output[f"coverage_{label}"] = float(np.mean(values))
        output[f"interval_width_{label}"] = float(np.mean(widths[level]))
    return output


def game_key_sha256(game_ids: Iterable[object]) -> str:
    value = "\n".join(sorted({str(game_id) for game_id in game_ids})).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def strict_comparison_key_audit(
    keys_by_candidate: Mapping[str, Iterable[object]],
) -> dict[str, object]:
    normalized = {
        str(candidate): sorted({str(value) for value in values})
        for candidate, values in keys_by_candidate.items()
    }
    unique_sets = {tuple(values) for values in normalized.values()}
    return {
        "identical": len(unique_sets) <= 1,
        "candidate_count": len(normalized),
        "game_counts": {candidate: len(values) for candidate, values in normalized.items()},
        "hashes": {
            candidate: game_key_sha256(values) for candidate, values in normalized.items()
        },
    }


def _coverage_error(metrics: Mapping[str, object]) -> float:
    return abs(float(metrics["coverage_80"]) - 0.8)


def development_gate(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    candidate_seasons: Sequence[Mapping[str, object]],
    baseline_seasons: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Apply the issue's frozen development pass and calibration gates."""

    candidate_map = {int(row["season"]): row for row in candidate_seasons}
    baseline_map = {int(row["season"]): row for row in baseline_seasons}
    common = sorted(set(candidate_map) & set(baseline_map))
    deltas = [
        float(candidate_map[season]["marginalized_nll"])
        - float(baseline_map[season]["marginalized_nll"])
        for season in common
    ]
    mae_delta = float(candidate["expected_margin_mae"]) - float(baseline["expected_margin_mae"])
    primary = (
        float(candidate["marginalized_nll"]) - float(baseline["marginalized_nll"]) <= -0.005
        and sum(delta < 0.0 for delta in deltas) >= 3
        and max(deltas, default=0.0) <= 0.010
        and mae_delta <= 0.10
        and _coverage_error(candidate) - _coverage_error(baseline) <= 0.01
    )
    alternative = (
        float(candidate["marginalized_nll"]) - float(baseline["marginalized_nll"]) <= 0.002
        and _coverage_error(baseline) - _coverage_error(candidate) >= 0.02
        and mae_delta <= 0.10
    )
    return {
        "primary_pass": bool(primary),
        "alternative_calibration_pass": bool(alternative),
        "qualifies": bool(primary or alternative),
        "development_nll_delta": float(candidate["marginalized_nll"])
        - float(baseline["marginalized_nll"]),
        "season_nll_deltas": {str(season): delta for season, delta in zip(common, deltas)},
        "seasons_with_nll_improvement": int(sum(delta < 0.0 for delta in deltas)),
        "expected_margin_mae_delta": mae_delta,
        "coverage_80_error_delta": _coverage_error(candidate) - _coverage_error(baseline),
    }


def _alternative_calibration_move(
    candidate: Mapping[str, object],
    simpler: Mapping[str, object],
) -> bool:
    """Return whether calibration justifies a complexity step."""

    return bool(
        float(candidate["marginalized_nll"])
        - float(simpler["marginalized_nll"])
        <= 0.002
        and _coverage_error(simpler) - _coverage_error(candidate) >= 0.02
        and float(candidate["expected_margin_mae"])
        - float(simpler["expected_margin_mae"])
        <= 0.10
    )


def select_development_candidate(
    aggregate: Mapping[str, Mapping[str, object]],
    seasons: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[str | None, dict[str, object]]:
    """Qualify every candidate against V1, then apply complexity preferences."""

    candidate_names = tuple(name for name in CANDIDATES if name != "v1")
    qualification: dict[str, dict[str, object]] = {}
    for candidate in candidate_names:
        gate = development_gate(
            aggregate[candidate],
            aggregate["v1"],
            seasons[candidate],
            seasons["v1"],
        )
        qualification[candidate] = {
            **gate,
            "qualifies_vs_v1": bool(gate["qualifies"]),
        }

    decisions: dict[str, object] = {}
    current = "v1"

    def consider(candidate: str, *, requires_complexity: bool) -> None:
        nonlocal current
        gate = qualification[candidate]
        simpler = aggregate[current]
        gain = float(simpler["marginalized_nll"]) - float(
            aggregate[candidate]["marginalized_nll"]
        )
        calibration_move = _alternative_calibration_move(aggregate[candidate], simpler)
        complexity_ok = not requires_complexity or gain >= 0.003 or calibration_move
        accepted = bool(gate["qualifies_vs_v1"] and complexity_ok)
        decisions[candidate] = {
            **gate,
            "comparison_baseline": current,
            "complexity_gain_over_current": gain,
            "alternative_calibration_move_over_current": calibration_move,
            "complexity_requirement_applied": requires_complexity,
            "accepted": accepted,
        }
        if accepted:
            current = candidate

    # A is the simplest research extension.  Once it qualifies against V1 it
    # becomes the starting point for the complexity comparison; its
    # qualification itself does not need the later .003 complexity hurdle.
    consider("a", requires_complexity=False)
    consider("b", requires_complexity=True)

    c_candidates = [name for name in candidate_names if name.startswith("c")]
    qualifying_c = [name for name in c_candidates if qualification[name]["qualifies_vs_v1"]]
    for candidate in c_candidates:
        if candidate not in qualifying_c:
            decisions[candidate] = {
                **qualification[candidate],
                "comparison_baseline": current,
                "complexity_gain_over_current": None,
                "alternative_calibration_move_over_current": False,
                "complexity_requirement_applied": True,
                "accepted": False,
            }
    if qualifying_c:
        best_c = min(
            qualifying_c,
            key=lambda name: (float(aggregate[name]["marginalized_nll"]), float(name[1:])),
        )
        c_comparison_baseline = current
        consider(best_c, requires_complexity=True)
        for candidate in qualifying_c:
            if candidate == best_c:
                continue
            decisions[candidate] = {
                **qualification[candidate],
                "comparison_baseline": c_comparison_baseline,
                "complexity_gain_over_current": None,
                "alternative_calibration_move_over_current": False,
                "complexity_requirement_applied": True,
                "accepted": False,
                "selected_as_c_variant": False,
            }
        decisions["c"] = {
            **decisions[best_c],
            "chosen_k": float(best_c[1:]),
            "candidate": best_c,
            "selected_as_c_variant": True,
        }
    else:
        decisions["c"] = {
            "candidate": None,
            "chosen_k": None,
            "qualifies_vs_v1": False,
            "accepted": False,
        }
    return (None if current == "v1" else current), {
        "selected_candidate": None if current == "v1" else current,
        "simplest_current_after_gate": current,
        "decisions": decisions,
        "qualification_baseline": "v1",
        "selection_metric_period": "development_2018_2021_only",
        "final_2022_2025_used": False,
    }
