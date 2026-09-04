"""Direct, leakage-aware preseason distributions over final rank.

This module models a final-rank coordinate, not latent team strength. A
team's prior constituent ranks are quadrature points for an uncertain observed
conditioning variable; current constituent ranks are an empirical outcome.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

EPSILON = 1e-6
QUADRATURE_POINTS = 12


def deterministic_quadrature(
    values: np.ndarray, points: int = QUADRATURE_POINTS
) -> np.ndarray:
    """Deterministically retain evenly spaced empirical support points.

    This limits fitting cost without replacing an empirical rank distribution
    with a location/scale summary. Evaluation and emitted PMFs still use all
    constituent observations.
    """
    values = np.asarray(values, dtype=float)
    if len(values) <= points:
        return values
    return values[np.linspace(0, len(values) - 1, points, dtype=int)]


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


@dataclass
class DirectRankModel:
    """Heteroscedastic Normal regression marginalized over prior-rank samples."""

    feature_names: list[str]
    preprocessor: Preprocessor
    beta: np.ndarray
    gamma: np.ndarray
    minimum_scale: float = 0.10
    penalty: float = 0.25

    @classmethod
    def fit(
        cls,
        rows: list[TeamSeason],
        feature_names: list[str],
        penalty: float = 0.25,
        minimum_scale: float = 0.10,
    ) -> DirectRankModel:
        if not rows:
            raise ValueError("cannot fit without rows")
        preprocessor = Preprocessor.fit([r.features for r in rows], feature_names)
        x = np.column_stack(
            [np.ones(len(rows)), preprocessor.transform([r.features for r in rows])]
        )
        lag_samples = [deterministic_quadrature(row.lag1_z) for row in rows]
        target_samples = [deterministic_quadrature(row.target_z) for row in rows]
        max_lag = max(map(len, lag_samples))
        max_target = max(map(len, target_samples))
        lags = np.zeros((len(rows), max_lag))
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

        def objective(theta: np.ndarray) -> float:
            beta, gamma = theta[: x.shape[1] + 1], theta[x.shape[1] + 1 :]
            base_locations = x @ beta[1:]
            scales = minimum_scale + np.exp(np.clip(x @ gamma, -5, 4))
            locations = base_locations[:, None] + beta[0] * lags
            densities = norm.logpdf(
                targets[:, :, None], locations[:, None, :], scales[:, None, None]
            )
            densities = np.where(lag_mask[:, None, :], densities, -np.inf)
            target_log_probability = logsumexp(densities, axis=2) - np.log(
                lag_counts[:, None]
            )
            losses = -(target_log_probability * target_mask).sum(axis=1) / target_counts
            regularizer = penalty * (np.sum(beta**2) + 0.25 * np.sum(gamma[1:] ** 2))
            return float(np.mean(losses) + regularizer / len(rows))

        initial_beta = np.zeros(x.shape[1] + 1)
        initial_beta[0] = 0.55
        initial_gamma = np.zeros(x.shape[1])
        initial_gamma[0] = np.log(0.7)
        result = minimize(
            objective,
            np.r_[initial_beta, initial_gamma],
            method="L-BFGS-B",
            options={"maxiter": 4, "ftol": 1e-7},
        )
        if not result.success and result.status != 1:
            raise RuntimeError(f"preseason optimizer failed: {result.message}")
        return cls(
            feature_names,
            preprocessor,
            result.x[: x.shape[1] + 1],
            result.x[x.shape[1] + 1 :],
            minimum_scale,
            penalty,
        )

    def _matrix(self, features: dict[str, float | None]) -> np.ndarray:
        return np.r_[1.0, self.preprocessor.transform([features])[0]]

    def conditional_parameters(
        self, features: dict[str, float | None], lag1_z: np.ndarray
    ) -> tuple[np.ndarray, float]:
        x = self._matrix(features)
        base_location = float(x @ self.beta[1:])
        locations = base_location + self.beta[0] * np.asarray(lag1_z)
        scale = float(self.minimum_scale + np.exp(np.clip(x @ self.gamma, -5, 4)))
        return locations, scale

    def pmf(
        self, features: dict[str, float | None], lag1_z: np.ndarray, population: int
    ) -> np.ndarray:
        locations, scale = self.conditional_parameters(features, lag1_z)
        edges = rank_bin_edges(population)
        masses = np.maximum(
            np.diff(norm.cdf((edges[None, :] - locations[:, None]) / scale), axis=1),
            0.0,
        )
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
            "conditioning": "equal-weight deterministic quadrature over lag-1 constituent ranks",
            "quadrature_points": QUADRATURE_POINTS,
            "outcome_weighting": "equal team-season weight; empirical target log score averages outcomes within team-season",
        }
