"""Leakage-aware preseason feature normalization and direct rank PMFs."""

from __future__ import annotations

import json
from collections.abc import Iterable

import numpy as np
from scipy.special import expit, logit


def clean_name(value: str) -> str:
    """Make API team names joinable without changing the source values."""
    return " ".join(value.casefold().replace("&", "and").split())


def rank_sample(row: dict[str, str]) -> np.ndarray:
    """Return constituent ranks, preserving the source disagreement."""
    return np.asarray(json.loads(row["rank_observations"]), dtype=float)


def rank_summary(row: dict[str, str], population: int) -> dict[str, float]:
    values = rank_sample(row)
    percentile = (values - 0.5) / population
    return {
        "mean": float(np.mean(percentile)),
        "sd": float(np.std(percentile, ddof=1)) if len(values) > 1 else 0.0,
        "n": float(len(values)),
        "rank_mean": float(np.mean(values)),
    }


def normal_pmf(
    location: float,
    scale: float,
    population: int,
    draws: int = 200_000,
    seed: int = 20260903,
) -> np.ndarray:
    """Map a smooth percentile distribution to a discrete rank PMF.

    Lower ranks are better. Monte Carlo is deterministic and only represents
    final rank outcomes; it is not a latent team-strength state.
    """
    rng = np.random.default_rng(seed)
    samples = np.clip(rng.normal(location, max(scale, 1e-4), draws), 1e-5, 1 - 1e-5)
    ranks = np.minimum(population, np.maximum(1, np.floor(samples * population + 0.5))).astype(int)
    return np.bincount(ranks, minlength=population + 1)[1:] / draws


def pmf_summaries(pmf: np.ndarray) -> dict[str, float]:
    ranks = np.arange(1, len(pmf) + 1)
    cdf = np.cumsum(pmf)
    q = lambda p: float(ranks[np.searchsorted(cdf, p, side="left")])
    return {
        "expected_rank": float(np.dot(ranks, pmf)),
        "median_rank": q(0.5),
        "interval_80_low": q(0.1),
        "interval_80_high": q(0.9),
        "top5_probability": float(pmf[:5].sum()),
        "top10_probability": float(pmf[:10].sum()),
        "top25_probability": float(pmf[:25].sum()),
    }


def weighted_ridge(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, penalty: float = 2.0
) -> np.ndarray:
    """Fit a deterministic weighted ridge model with an intercept column."""
    w = weights / np.mean(weights)
    gram = x.T @ (w[:, None] * x)
    gram += np.eye(x.shape[1]) * penalty
    gram[0, 0] -= penalty
    return np.linalg.solve(gram, x.T @ (w * y))


def pseudo_targets(rows: Iterable[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Expand each team-season to constituent outcomes with total weight one."""
    values: list[float] = []
    weights: list[float] = []
    for row in rows:
        population = int(row["team_population"])
        ranks = rank_sample(row)
        values.extend(((ranks - 0.5) / population).tolist())
        weights.extend([1.0 / len(ranks)] * len(ranks))
    return np.asarray(values), np.asarray(weights)


def empirical_nll(pmf: np.ndarray, ranks: np.ndarray) -> float:
    probabilities = np.maximum(pmf[ranks.astype(int) - 1], 1e-12)
    return float(-np.mean(np.log(probabilities)))


def crps_discrete(pmf: np.ndarray, rank: int) -> float:
    """CRPS for a discrete rank CDF, using the integer-rank convention."""
    cdf = np.cumsum(pmf)
    outcome = np.arange(1, len(pmf) + 1) >= rank
    return float(np.mean((cdf - outcome) ** 2))


def logistic_percentile(value: float) -> float:
    return float(logit(np.clip(value, 1e-5, 1 - 1e-5)))


def inverse_logistic(value: float) -> float:
    return float(expit(value))
