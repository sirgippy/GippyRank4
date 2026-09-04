"""Historical rank-distribution and score-margin modeling utilities."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

PAIRINGS = ("fbs-fbs", "fbs-fcs", "fcs-fcs")
_KNOTS = np.array([0.2, 0.4, 0.6, 0.8])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def coverage_table(observations: list[dict[str, str]]) -> list[dict[str, object]]:
    populations: dict[tuple[int, str], set[str]] = defaultdict(set)
    groups: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in observations:
        key = (int(row["season"]), row["subdivision"])
        if row["is_composite"] == "True":
            populations[key].add(row["team_id"])
        elif row["system_code"] != "CMP":
            groups[(*key, row["system_code"])].append(row)
    result = []
    for (season, subdivision, code), rows in sorted(groups.items()):
        ranks = [int(row["ordinal_rank"]) for row in rows]
        teams = {row["team_id"] for row in rows}
        rank_values = set(ranks)
        population = len(populations[(season, subdivision)])
        result.append(
            {
                "season": season,
                "subdivision": subdivision,
                "system_code": code,
                "distinct_teams_ranked": len(teams),
                "team_population": population,
                "coverage_ratio": len(teams) / population if population else 0.0,
                "minimum_rank": min(ranks),
                "maximum_rank": max(ranks),
                "has_ties": len(ranks) != len(rank_values),
                "has_rank_gaps": rank_values != set(range(min(ranks), max(ranks) + 1)),
            }
        )
    return result


def build_distributions(
    observations: list[dict[str, str]], depth: list[dict[str, object]], threshold: float
) -> tuple[list[dict[str, object]], dict[tuple[int, str, str], list[tuple[str, int]]]]:
    selected = {
        (int(row["season"]), row["subdivision"], row["system_code"])
        for row in depth
        if float(row["coverage_ratio"]) >= threshold
    }
    population = {
        (int(row["season"]), row["subdivision"]): int(row["team_population"])
        for row in depth
    }
    groups: dict[tuple[int, str, str], list[tuple[str, int, str]]] = defaultdict(list)
    for row in observations:
        key = (int(row["season"]), row["subdivision"], row["system_code"])
        if row["is_composite"] != "True" and key in selected:
            groups[(int(row["season"]), row["subdivision"], row["team_id"])].append(
                (row["team_name"], int(row["ordinal_rank"]), row["system_code"])
            )
    rows = []
    by_team: dict[tuple[int, str, str], list[tuple[str, int]]] = {}
    for (season, subdivision, team_id), values in sorted(groups.items()):
        ranks = [rank for _, rank, _ in values]
        names = [name for name, _, _ in values]
        n = population[(season, subdivision)]
        counts: dict[int, int] = defaultdict(int)
        for rank in ranks:
            counts[rank] += 1
        pmf = [
            {
                "rank": rank,
                "count": counts[rank],
                "probability": counts[rank] / len(ranks),
            }
            for rank in sorted(counts)
        ]
        mean = float(np.mean(ranks))
        q25, median, q75 = np.percentile(ranks, [25, 50, 75])
        rows.append(
            {
                "season": season,
                "subdivision": subdivision,
                "team_id": team_id,
                "team_name": max(set(names), key=names.count),
                "team_population": n,
                "usable_systems": len(ranks),
                "rank_observations": json.dumps(ranks),
                "pmf": json.dumps(pmf, separators=(",", ":")),
                "rank_percentile_mean": (mean - 0.5) / n,
                "rank_mean": mean,
                "rank_median": float(median),
                "rank_sd": float(np.std(ranks, ddof=1)) if len(ranks) > 1 else 0.0,
                "rank_iqr": float(q75 - q25),
            }
        )
        by_team[(season, subdivision, team_id)] = [
            (code, rank) for _, rank, code in values
        ]
    return rows, by_team


def _same_basis(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Smooth odd basis: swapping teams changes every column's sign."""
    d = x - y
    s = (x + y) / 2
    return np.column_stack(
        [
            d,
            d * s,
            d * s * s,
            d * np.abs(d),
            *(d * np.maximum(s - k, 0) for k in _KNOTS),
        ]
    )


def _cross_basis(fbs: np.ndarray, fcs: np.ndarray) -> np.ndarray:
    """Smooth surface whose first two coordinates are always FBS and FCS."""
    return np.column_stack(
        [
            np.ones(len(fbs)),
            fbs,
            fcs,
            fbs * fbs,
            fcs * fcs,
            fbs * fcs,
            *(np.maximum(fbs - k, 0) for k in _KNOTS),
            *(np.maximum(fcs - k, 0) for k in _KNOTS),
        ]
    )


def design_matrix(
    x: np.ndarray,
    y: np.ndarray,
    pairing: np.ndarray,
    home: np.ndarray,
    neutral: np.ndarray,
    surface: bool = True,
    fbs_home: np.ndarray | None = None,
) -> np.ndarray:
    """Build identifiable pairing blocks; cross rows use stable FBS/FCS coordinates.

    ``home`` is non-neutral home-field for same-subdivision rows. For cross rows,
    ``fbs_home`` is 1 when FBS is home, while ``1-fbs_home-home`` identifies FCS
    home; neutral is the reference site and has neither indicator.
    """
    fbs_home = (
        np.zeros(len(x), dtype=float)
        if fbs_home is None
        else np.asarray(fbs_home, dtype=float)
    )
    fcs_home = (1 - np.asarray(neutral, dtype=float)) * (1 - fbs_home)
    if surface:
        same = np.column_stack([_same_basis(x, y), home])
        cross = np.column_stack([_cross_basis(x, y), fbs_home, fcs_home])
    else:
        same = np.column_stack([x - y, home])
        cross = np.column_stack([np.ones(len(x)), x - y, fbs_home, fcs_home])
    return np.column_stack(
        [
            (same if name != "fbs-fcs" else cross) * (pairing == name)[:, None]
            for name in PAIRINGS
        ]
    )


def _log_student_t(
    y: np.ndarray, location: np.ndarray, scale: float, df: float
) -> np.ndarray:
    z = (y - location) / scale
    constant = (
        gammaln((df + 1) / 2)
        - gammaln(df / 2)
        - 0.5 * np.log(df * np.pi)
        - np.log(scale)
    )
    return constant - (df + 1) / 2 * np.log1p(z * z / df)


def student_t_nll(
    y: np.ndarray, location: np.ndarray, scale: float, df: float
) -> float:
    return float(-np.mean(_log_student_t(y, location, scale, df)))


def _initial_fit(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, float]:
    weighted = np.maximum(weights, 0)
    gram = X.T @ (weighted[:, None] * X)
    rhs = X.T @ (weighted * y)
    try:
        beta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    scale = max(float(np.sqrt(np.average((y - X @ beta) ** 2, weights=weights))), 1.0)
    return beta, scale


def fit_robust_surface(
    X: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    iterations: int = 8,
    df: float = 5.0,
) -> dict[str, object]:
    """Fit a weighted pseudo-observation Student-t regression by IRLS."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum() * len(weights)
    beta, scale = _initial_fit(X, target, weights)
    for _ in range(iterations):
        residual = target - X @ beta
        robust = (df + 1) / (df + (residual / scale) ** 2)
        beta, scale = _initial_fit(X, target, weights * robust)
    return {
        "beta": beta,
        "scale": scale,
        "df": float(df),
        "n_features": X.shape[1],
        "objective": "weighted_pseudo",
    }


def fit_marginalized(
    X: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    df: float = 5.0,
    maxiter: int = 250,
) -> dict[str, object]:
    """Fit equal-game -log(mean density), with stable log-sum-exp."""
    beta, scale = _initial_fit(X, target, np.ones(len(target)))
    unique, inverse = np.unique(groups, return_inverse=True)
    n_groups = len(unique)
    counts = np.bincount(inverse, minlength=n_groups)

    def objective_gradient(theta: np.ndarray) -> tuple[float, np.ndarray]:
        loc = X @ theta[:-1]
        sc = np.exp(theta[-1])
        residual = target - loc
        logs = _log_student_t(target, loc, sc, df)
        maxima = np.full(n_groups, -np.inf)
        np.maximum.at(maxima, inverse, logs)
        normalizers = np.bincount(
            inverse, weights=np.exp(logs - maxima[inverse]), minlength=n_groups
        )
        log_marginal = maxima + np.log(normalizers) - np.log(counts)
        probabilities = np.exp(logs - maxima[inverse]) / normalizers[inverse]
        denominator = df * sc * sc + residual**2
        loc_gradient = (df + 1) * residual / denominator
        beta_gradient = -X.T @ (probabilities * loc_gradient)
        scale_gradient = -np.sum(
            probabilities * (-1 + (df + 1) * residual**2 / denominator)
        )
        return float(-np.mean(log_marginal)), np.r_[
            beta_gradient, scale_gradient
        ] / n_groups

    def objective(theta: np.ndarray) -> float:
        return objective_gradient(theta)[0]

    def gradient(theta: np.ndarray) -> np.ndarray:
        return objective_gradient(theta)[1]

    result = minimize(
        objective,
        np.r_[beta, np.log(scale)],
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(None, None)] * X.shape[1] + [(np.log(0.1), np.log(1000.0))],
        options={"maxiter": maxiter, "ftol": 1e-9},
    )
    return {
        "beta": result.x[:-1],
        "scale": float(np.exp(result.x[-1])),
        "df": float(df),
        "n_features": X.shape[1],
        "objective": "marginalized",
        "optimizer_success": bool(result.success),
        "optimizer_message": result.message,
    }


def game_log_scores(
    target: np.ndarray,
    locations: np.ndarray,
    groups: np.ndarray,
    scale: float,
    df: float,
) -> dict[str, float]:
    """Return equal-game expected conditional and marginalized NLLs."""
    values = []
    for group in np.unique(groups):
        logs = _log_student_t(
            target[groups == group], locations[groups == group], scale, df
        )
        values.append(
            (float(-np.mean(logs)), float(-(logsumexp(logs) - np.log(len(logs)))))
        )
    values = np.asarray(values)
    return {
        "expected_conditional_nll": float(np.mean(values[:, 0])),
        "marginalized_nll": float(np.mean(values[:, 1])),
        "n_games": len(values),
    }
