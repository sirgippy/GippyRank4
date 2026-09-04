"""Direct, leakage-aware preseason distributions over final rank.

This module models a final-rank coordinate, not latent team strength. A
team's prior constituent ranks are quadrature points for an uncertain observed
conditioning variable; current constituent ranks are an empirical outcome.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm, t

EPSILON = 1e-6
QUADRATURE_POINTS = 12
MULTI_LAG_QUADRATURE_POINTS = 2


def deterministic_quadrature(
    values: np.ndarray, points: int = QUADRATURE_POINTS
) -> np.ndarray:
    """Deterministically retain evenly spaced empirical support points.

    This limits fitting cost without replacing an empirical rank distribution
    with a location/scale summary. Evaluation and emitted PMFs still use all
    constituent observations.
    """
    values = np.asarray(values, dtype=float)
    values = np.sort(values)
    if len(values) <= points:
        return values
    return values[np.linspace(0, len(values) - 1, points, dtype=int)]


def product_quadrature(
    distributions: tuple[np.ndarray, ...],
    points: int = MULTI_LAG_QUADRATURE_POINTS,
) -> np.ndarray:
    """Equal-weight deterministic product quadrature for lag distributions.

    Each input is reduced independently by permutation-invariant empirical
    order statistics, then the Cartesian product marginalizes their joint
    conditioning uncertainty.  Consequently each retained product point has
    equal mass and every team-season retains total weight one.
    """
    if not distributions:
        raise ValueError("at least one lag distribution is required")
    supports = [deterministic_quadrature(values, points) for values in distributions]
    if any(not len(values) for values in supports):
        raise ValueError("lag distributions must be non-empty")
    return np.asarray(list(product(*supports)), dtype=float)


def historical_rank_features(
    lag1_z: np.ndarray, history: tuple[np.ndarray, ...]
) -> dict[str, float | None]:
    """Descriptive preseason features from already-observed rank distributions.

    ``history`` is supplied by the caller as seasons strictly before its target;
    this pure helper deliberately has no target or future-season argument.
    """
    season_means = np.asarray([np.mean(values) for values in history if len(values)])
    lag1_mean = float(np.mean(lag1_z))
    lag2_mean = float(np.mean(history[-2])) if len(history) >= 2 else None
    previous_means = season_means[:-1]
    return {
        "long_run_z_mean": float(np.mean(season_means)) if len(season_means) else None,
        "history_z_std": float(np.std(season_means)) if len(season_means) > 1 else None,
        "history_seasons": float(len(season_means)),
        "lag1_disagreement": float(np.std(lag1_z)),
        "trajectory_z": lag1_mean - lag2_mean if lag2_mean is not None else None,
        "recent_shock_z": lag1_mean - float(np.mean(previous_means))
        if len(previous_means)
        else None,
    }


def clean_name(value: str) -> str:
    """Make API team names joinable without changing stored source values."""
    return " ".join(value.casefold().replace("&", "and").split())


def rank_sample(row: dict[str, str]) -> np.ndarray:
    """Return usable ordinary-constituent ranks for one team-season."""
    return np.asarray(json.loads(row["rank_observations"]), dtype=float)


def rank_to_z(ranks: np.ndarray, population: int) -> np.ndarray:
    """Map actual ranks to an unbounded representation of rank percentile."""
    percentile = np.clip(
        (np.asarray(ranks, dtype=float) - 0.5) / population, EPSILON, 1 - EPSILON
    )
    return np.log(percentile) - np.log1p(-percentile)


def rank_bin_edges(population: int) -> np.ndarray:
    """Transformed bin boundaries for ranks 1..N, including infinite edges."""
    if population < 1:
        raise ValueError("population must be positive")
    interior = np.arange(1, population, dtype=float) / population
    return np.r_[-np.inf, np.log(interior) - np.log1p(-interior), np.inf]


def normal_pmf(location: float, scale: float, population: int, **_: Any) -> np.ndarray:
    """Integrate a Normal coordinate distribution over discrete rank bins."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    edges = rank_bin_edges(population)
    pmf = np.maximum(np.diff(norm.cdf((edges - location) / scale)), 0.0)
    return pmf / pmf.sum()


def pmf_summaries(pmf: np.ndarray) -> dict[str, float]:
    ranks = np.arange(1, len(pmf) + 1)
    cdf = np.cumsum(pmf)
    quantile = lambda p: float(ranks[np.searchsorted(cdf, p, side="left")])
    return {
        "expected_rank": float(np.dot(ranks, pmf)),
        "median_rank": quantile(0.5),
        "interval_80_low": quantile(0.1),
        "interval_80_high": quantile(0.9),
        "top5_probability": float(pmf[:5].sum()),
        "top10_probability": float(pmf[:10].sum()),
        "top25_probability": float(pmf[:25].sum()),
    }


def crps_discrete(pmf: np.ndarray, rank: int) -> float:
    """CRPS on a normalized rank coordinate, comparable across subdivisions."""
    cdf = np.cumsum(pmf)
    outcome = np.arange(1, len(pmf) + 1) >= rank
    return float(np.mean((cdf - outcome) ** 2))


def team_log_score(pmf: np.ndarray, ranks: np.ndarray) -> float:
    """Equal-team empirical log score; duplicate constituent rows change nothing."""
    probabilities = np.maximum(pmf[np.asarray(ranks, dtype=int) - 1], 1e-15)
    return float(-np.mean(np.log(probabilities)))


@dataclass(frozen=True)
class Preprocessor:
    """Training-only median imputation, standardization, and indicators."""

    feature_names: tuple[str, ...]
    medians: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]

    @classmethod
    def fit(
        cls, rows: list[dict[str, float | None]], feature_names: list[str]
    ) -> Preprocessor:
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for name in feature_names:
            observed = np.asarray(
                [r[name] for r in rows if r.get(name) is not None], dtype=float
            )
            medians[name] = float(np.median(observed)) if len(observed) else 0.0
            imputed = np.asarray(
                [medians[name] if r.get(name) is None else r[name] for r in rows],
                dtype=float,
            )
            means[name] = float(np.mean(imputed))
            scales[name] = max(float(np.std(imputed)), 1e-8)
        return cls(tuple(feature_names), medians, means, scales)

    def transform(self, rows: list[dict[str, float | None]]) -> np.ndarray:
        values = []
        for row in rows:
            numeric, missing = [], []
            for name in self.feature_names:
                value = row.get(name)
                missing.append(float(value is None))
                numeric.append(
                    (
                        (self.medians[name] if value is None else float(value))
                        - self.means[name]
                    )
                    / self.scales[name]
                )
            values.append([*numeric, *missing])
        return np.asarray(values, dtype=float)

    def metadata(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TeamSeason:
    """One equally weighted historical target distribution and its inputs."""

    season: int
    subdivision: str
    team_id: str
    team_name: str
    population: int
    lag1_z: np.ndarray
    target_z: np.ndarray
    target_ranks: np.ndarray
    features: dict[str, float | None]
    lag_zs: tuple[np.ndarray, ...] = ()


@dataclass
class DirectRankModel:
    """Heteroscedastic Normal regression marginalized over prior-rank samples."""

    feature_names: list[str]
    preprocessor: Preprocessor
    beta: np.ndarray
    gamma: np.ndarray
    minimum_scale: float = 0.10
    penalty: float = 0.25
    optimizer: dict[str, object] | None = None
    lag_count: int = 1
    family: str = "normal"
    degrees_of_freedom: float | None = None

    @classmethod
    def fit(
        cls,
        rows: list[TeamSeason],
        feature_names: list[str],
        penalty: float = 0.25,
        minimum_scale: float = 0.10,
        optimizer_options: dict[str, float | int] | None = None,
        lag_count: int = 1,
        family: str = "normal",
        degrees_of_freedom: float | None = None,
    ) -> DirectRankModel:
        if not rows:
            raise ValueError("cannot fit without rows")
        preprocessor = Preprocessor.fit([r.features for r in rows], feature_names)
        x = np.column_stack(
            [np.ones(len(rows)), preprocessor.transform([r.features for r in rows])]
        )
        if family not in {"normal", "student_t"}:
            raise ValueError(f"unsupported distribution family: {family}")
        if family == "student_t" and (
            degrees_of_freedom is None or degrees_of_freedom <= 2
        ):
            raise ValueError("Student-t degrees of freedom must exceed 2")
        if lag_count < 1:
            raise ValueError("lag_count must be positive")
        lag_samples = []
        for row in rows:
            distributions = (row.lag1_z, *row.lag_zs[: lag_count - 1])
            if len(distributions) != lag_count:
                raise ValueError("row lacks required lag distributions")
            points = (
                QUADRATURE_POINTS if lag_count == 1 else MULTI_LAG_QUADRATURE_POINTS
            )
            lag_samples.append(product_quadrature(distributions, points))
        target_samples = [deterministic_quadrature(row.target_z) for row in rows]
        max_lag = max(map(len, lag_samples))
        max_target = max(map(len, target_samples))
        lags = np.zeros((len(rows), max_lag, lag_count))
        targets = np.zeros((len(rows), max_target))
        lag_mask = np.zeros((len(rows), max_lag), dtype=bool)
        target_mask = np.zeros((len(rows), max_target), dtype=bool)
        for index, (lag, target) in enumerate(
            zip(lag_samples, target_samples, strict=True)
        ):
            lags[index, : len(lag)] = lag
            lag_mask[index, : len(lag)] = True
            targets[index, : len(target)] = target
            target_mask[index, : len(target)] = True
        lag_counts = lag_mask.sum(axis=1)
        target_counts = target_mask.sum(axis=1)

        def objective_gradient(theta: np.ndarray) -> tuple[float, np.ndarray]:
            beta, gamma = (
                theta[: x.shape[1] + lag_count],
                theta[x.shape[1] + lag_count :],
            )
            base_locations = x @ beta[lag_count:]
            eta = x @ gamma
            exp_eta = np.exp(np.clip(eta, -5, 4))
            scales = minimum_scale + exp_eta
            locations = base_locations[:, None] + lags @ beta[:lag_count]
            if family == "normal":
                densities = norm.logpdf(
                    targets[:, :, None], locations[:, None, :], scales[:, None, None]
                )
            else:
                densities = t.logpdf(
                    (targets[:, :, None] - locations[:, None, :])
                    / scales[:, None, None],
                    degrees_of_freedom,
                ) - np.log(scales[:, None, None])
            densities = np.where(lag_mask[:, None, :], densities, -np.inf)
            log_mixture = logsumexp(densities, axis=2)
            target_log_probability = log_mixture - np.log(lag_counts[:, None])
            losses = -(target_log_probability * target_mask).sum(axis=1) / target_counts
            regularizer = penalty * (np.sum(beta**2) + 0.25 * np.sum(gamma[1:] ** 2))
            objective = float(np.mean(losses) + regularizer / len(rows))
            responsibilities = np.exp(densities - log_mixture[:, :, None])
            residual = targets[:, :, None] - locations[:, None, :]
            target_weight = (
                target_mask[:, :, None] / target_counts[:, None, None] / len(rows)
            )
            location_score = responsibilities * residual / scales[:, None, None] ** 2
            weighted_location_score = location_score * target_weight
            beta_gradient = np.empty_like(beta)
            beta_gradient[:lag_count] = -np.einsum(
                "rtq,rqk->k", weighted_location_score, lags
            )
            base_gradient = -np.sum(weighted_location_score, axis=(1, 2))
            beta_gradient[lag_count:] = x.T @ base_gradient
            if family == "normal":
                scale_score = responsibilities * (
                    -1 / scales[:, None, None]
                    + residual**2 / scales[:, None, None] ** 3
                )
                location_score = (
                    responsibilities * residual / scales[:, None, None] ** 2
                )
            else:
                denominator = (
                    degrees_of_freedom * scales[:, None, None] ** 2 + residual**2
                )
                location_score = responsibilities * (
                    (degrees_of_freedom + 1) * residual / denominator
                )
                standardized_squared = residual**2 / scales[:, None, None] ** 2
                scale_score = (
                    responsibilities
                    * (
                        -1
                        + (degrees_of_freedom + 1)
                        * standardized_squared
                        / (degrees_of_freedom + standardized_squared)
                    )
                    / scales[:, None, None]
                )
                weighted_location_score = location_score * target_weight
                beta_gradient[:lag_count] = -np.einsum(
                    "rtq,rqk->k", weighted_location_score, lags
                )
                base_gradient = -np.sum(weighted_location_score, axis=(1, 2))
                beta_gradient[lag_count:] = x.T @ base_gradient
            gamma_gradient = x.T @ (
                -np.sum(
                    scale_score * exp_eta[:, None, None] * target_weight, axis=(1, 2)
                )
            )
            beta_gradient += 2 * penalty * beta / len(rows)
            gamma_gradient[1:] += 0.5 * penalty * gamma[1:] / len(rows)
            return objective, np.r_[beta_gradient, gamma_gradient]

        def objective(theta: np.ndarray) -> float:
            return objective_gradient(theta)[0]

        def gradient(theta: np.ndarray) -> np.ndarray:
            return objective_gradient(theta)[1]

        initial_beta = np.zeros(x.shape[1] + lag_count)
        initial_beta[0] = 0.55
        initial_gamma = np.zeros(x.shape[1])
        initial_gamma[0] = np.log(0.7)
        options = {"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6}
        options.update(optimizer_options or {})
        result = minimize(
            objective,
            np.r_[initial_beta, initial_gamma],
            method="L-BFGS-B",
            jac=gradient,
            bounds=[(None, None)] * len(initial_beta)
            + [(-5.0, 4.0)] * len(initial_gamma),
            options=options,
        )
        diagnostics = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "objective": float(result.fun),
        }
        if not result.success:
            raise RuntimeError(f"preseason optimizer failed: {result.message}")
        return cls(
            feature_names,
            preprocessor,
            result.x[: x.shape[1] + lag_count],
            result.x[x.shape[1] + lag_count :],
            minimum_scale,
            penalty,
            diagnostics,
            lag_count,
            family,
            degrees_of_freedom,
        )

    def _matrix(self, features: dict[str, float | None]) -> np.ndarray:
        return np.r_[1.0, self.preprocessor.transform([features])[0]]

    def conditional_parameters(
        self,
        features: dict[str, float | None],
        lag1_z: np.ndarray,
        lag_zs: tuple[np.ndarray, ...] = (),
    ) -> tuple[np.ndarray, float]:
        x = self._matrix(features)
        distributions = (lag1_z, *lag_zs[: self.lag_count - 1])
        if len(distributions) != self.lag_count:
            raise ValueError("prediction lacks required lag distributions")
        points = (
            QUADRATURE_POINTS if self.lag_count == 1 else MULTI_LAG_QUADRATURE_POINTS
        )
        lags = product_quadrature(distributions, points)
        base_location = float(x @ self.beta[self.lag_count :])
        locations = base_location + lags @ self.beta[: self.lag_count]
        scale = float(self.minimum_scale + np.exp(np.clip(x @ self.gamma, -5, 4)))
        return locations, scale

    def pmf(
        self,
        features: dict[str, float | None],
        lag1_z: np.ndarray,
        population: int,
        lag_zs: tuple[np.ndarray, ...] = (),
    ) -> np.ndarray:
        locations, scale = self.conditional_parameters(features, lag1_z, lag_zs)
        edges = rank_bin_edges(population)
        standardized_edges = (edges[None, :] - locations[:, None]) / scale
        cdf = (
            norm.cdf(standardized_edges)
            if self.family == "normal"
            else t.cdf(standardized_edges, self.degrees_of_freedom)
        )
        masses = np.maximum(np.diff(cdf, axis=1), 0.0)
        pmf = np.mean(masses, axis=0)
        return pmf / pmf.sum()

    def metadata(self) -> dict[str, object]:
        return {
            "feature_names": self.feature_names,
            "preprocessing": self.preprocessor.metadata(),
            "location_coefficients": self.beta.tolist(),
            "log_scale_coefficients": self.gamma.tolist(),
            "minimum_scale": self.minimum_scale,
            "penalty": self.penalty,
            "optimizer": self.optimizer,
            "conditioning": "equal-weight deterministic quadrature over prior constituent ranks",
            "lag_count": self.lag_count,
            "family": self.family,
            "degrees_of_freedom": self.degrees_of_freedom,
            "quadrature_points": QUADRATURE_POINTS
            if self.lag_count == 1
            else MULTI_LAG_QUADRATURE_POINTS,
            "quadrature_method": "sort empirical values then retain evenly spaced order statistics",
            "outcome_weighting": "equal team-season weight; empirical target log score averages outcomes within team-season",
        }


@dataclass(frozen=True)
class GenericRankPrior:
    """Broad analytical fallback for teams lacking any prior rank observation."""

    location: float
    scale: float
    n_team_seasons: int

    @classmethod
    def fit(
        cls, rows: list[TeamSeason], minimum_scale: float = 0.10
    ) -> GenericRankPrior:
        if not rows:
            raise ValueError("cannot fit cold-start prior without historical rows")
        # Equal team-season weight: every team's empirical constituent outcomes
        # are averaged before pooling its contribution to the moments.
        means = np.asarray([np.mean(row.target_z) for row in rows], dtype=float)
        location = float(np.mean(means))
        variance = float(
            np.mean([np.mean((row.target_z - location) ** 2) for row in rows])
        )
        return cls(location, max(float(np.sqrt(variance)), minimum_scale), len(rows))

    def pmf(self, population: int) -> np.ndarray:
        return normal_pmf(self.location, self.scale, population)

    def metadata(self) -> dict[str, object]:
        return {
            "fit_method": "analytical equal-team empirical moments",
            "location": self.location,
            "scale": self.scale,
            "n_team_seasons": self.n_team_seasons,
            "optimizer": None,
        }
