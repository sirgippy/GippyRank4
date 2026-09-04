"""Historical rank-distribution and score-margin modeling utilities."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy.special import gammaln

PAIRINGS = ("fbs-fbs", "fbs-fcs", "fcs-fcs")


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
        result.append({
            "season": season, "subdivision": subdivision, "system_code": code,
            "distinct_teams_ranked": len(teams), "team_population": population,
            "coverage_ratio": len(teams) / population if population else 0.0,
            "minimum_rank": min(ranks), "maximum_rank": max(ranks),
            "has_ties": len(ranks) != len(rank_values),
            "has_rank_gaps": rank_values != set(range(min(ranks), max(ranks) + 1)),
        })
    return result


def build_distributions(
    observations: list[dict[str, str]], depth: list[dict[str, object]], threshold: float
) -> tuple[list[dict[str, object]], dict[tuple[int, str, str], list[tuple[str, int]]]]:
    selected = {
        (int(row["season"]), row["subdivision"], row["system_code"])
        for row in depth if float(row["coverage_ratio"]) >= threshold
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
        pmf = [{"rank": rank, "count": counts[rank], "probability": counts[rank] / len(ranks)}
               for rank in sorted(counts)]
        mean = float(np.mean(ranks))
        q25, median, q75 = np.percentile(ranks, [25, 50, 75])
        rows.append({
            "season": season, "subdivision": subdivision, "team_id": team_id,
            "team_name": max(set(names), key=names.count), "team_population": n,
            "usable_systems": len(ranks), "rank_observations": json.dumps(ranks),
            "pmf": json.dumps(pmf, separators=(",", ":")),
            "rank_percentile_mean": (mean - 0.5) / n,
            "rank_mean": mean, "rank_median": float(median),
            "rank_sd": float(np.std(ranks, ddof=1)) if len(ranks) > 1 else 0.0,
            "rank_iqr": float(q75 - q25),
        })
        by_team[(season, subdivision, team_id)] = [(code, rank) for _, rank, code in values]
    return rows, by_team


def _basis(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A compact tensor-product hinge surface plus context terms."""
    knots = np.array([0.15, 0.30, 0.45, 0.60, 0.75, 0.90])
    bx = np.column_stack([x, x * x, np.maximum(x[:, None] - knots, 0.0)])
    by = np.column_stack([y, y * y, np.maximum(y[:, None] - knots, 0.0)])
    terms = [np.ones(len(x)), x, y, x * y, x * x, y * y]
    terms.extend([bx[:, i] * by[:, j] for i in range(bx.shape[1]) for j in range(by.shape[1])])
    return np.column_stack(terms)


def design_matrix(x: np.ndarray, y: np.ndarray, pairing: np.ndarray, home: np.ndarray, neutral: np.ndarray, surface: bool = True) -> np.ndarray:
    base = _basis(x, y) if surface else np.column_stack([np.ones(len(x)), x - y])
    context = np.column_stack([home, neutral])
    blocks = [np.column_stack([base, context])]
    for pair in PAIRINGS:
        mask = (pairing == pair).astype(float)[:, None]
        blocks.append(np.column_stack([base * mask, context * mask]))
    return np.column_stack(blocks)


def fit_robust_surface(X: np.ndarray, target: np.ndarray, weights: np.ndarray, iterations: int = 6) -> dict[str, object]:
    """Fit a fixed-df Student-t regression by deterministic IRLS."""
    weights = weights / weights.sum() * len(weights)
    beta = np.linalg.solve(X.T @ (weights[:, None] * X) + np.eye(X.shape[1]) * 1e-6, X.T @ (weights * target))
    scale = max(float(np.median(np.abs(target - X @ beta)) / 0.6745), 1.0)
    df = 5.0
    for _ in range(iterations):
        residual = target - X @ beta
        robust = (df + 1) / (df + (residual / scale) ** 2)
        total = weights * robust
        beta = np.linalg.solve(X.T @ (total[:, None] * X) + np.eye(X.shape[1]) * 1e-6, X.T @ (total * target))
        scale = max(float(np.sqrt(np.sum(total * (target - X @ beta) ** 2) / np.sum(total))), 1.0)
    return {"beta": beta, "scale": scale, "df": df, "n_features": X.shape[1]}


def student_t_nll(y: np.ndarray, location: np.ndarray, scale: float, df: float) -> float:
    z = (y - location) / scale
    constant = gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(df * np.pi) - np.log(scale)
    return float(-np.mean(constant - (df + 1) / 2 * np.log1p(z * z / df)))
