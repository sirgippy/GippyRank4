"""Historical rank-distribution and score-margin modeling utilities."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import t as student_t

PAIRINGS = ("fbs-fbs", "fbs-fcs", "fcs-fcs")
CONSTITUENT_RANK_PAIR_FIELDS = (
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
_KNOTS = np.array([0.2, 0.4, 0.6, 0.8])


def mixture_central_interval(
    locations: np.ndarray, scale: float, df: float, level: float = 0.8
) -> tuple[float, float]:
    """Central interval of an equally weighted Student-t location mixture."""
    from scipy.optimize import brentq

    locations = np.asarray(locations, dtype=float)
    low = float(np.min(locations) - 20 * scale)
    high = float(np.max(locations) + 20 * scale)

    def cdf(value: float) -> float:
        return float(np.mean(student_t.cdf((value - locations) / scale, df)))

    return (
        brentq(lambda value: cdf(value) - (1 - level) / 2, low, high),
        brentq(lambda value: cdf(value) - (1 + level) / 2, low, high),
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_rank_observations(row: dict[str, object]) -> list[int]:
    """Validate source ranks against the declared population before transforms."""
    context = (
        f"{row.get('season')} {row.get('subdivision')} team "
        f"{row.get('team_id')} ({row.get('team_name')})"
    )
    try:
        population = int(row["team_population"])
        ranks = json.loads(str(row["rank_observations"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context}: malformed rank observations: {exc}") from exc
    if not str(row.get("team_id", "")).strip() or not str(
        row.get("team_name", "")
    ).strip():
        raise ValueError(f"{context}: canonical team identity is required")
    if population < 1:
        raise ValueError(f"{context}: team population must be positive")
    if not isinstance(ranks, list) or not ranks:
        raise ValueError(f"{context}: rank observations must be a non-empty list")
    for rank in ranks:
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError(f"{context}: constituent rank must be an integer: {rank!r}")
        if not 1 <= rank <= population:
            raise ValueError(
                f"{context}: constituent rank {rank} is outside 1..{population}"
            )
    return ranks


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


def validate_distribution_row(row: dict[str, object]) -> None:
    """Validate a rank PMF against its retained constituent observations."""
    context = (
        f"{row.get('season')} {row.get('subdivision')} team "
        f"{row.get('team_id')} ({row.get('team_name')})"
    )
    try:
        population = int(row["team_population"])
        usable_systems = int(row["usable_systems"])
        ranks = json.loads(str(row["rank_observations"]))
        pmf = json.loads(str(row["pmf"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context}: malformed rank distribution: {exc}") from exc
    if not str(row.get("team_id", "")).strip() or not str(
        row.get("team_name", "")
    ).strip():
        raise ValueError(f"{context}: canonical team identity is required")
    if population < 1:
        raise ValueError(f"{context}: team population must be positive")
    if not isinstance(ranks, list) or not ranks:
        raise ValueError(f"{context}: rank observations must be a non-empty list")
    rank_counts: Counter[int] = Counter()
    for rank in ranks:
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError(f"{context}: constituent rank must be an integer: {rank!r}")
        if not 1 <= rank <= population:
            raise ValueError(
                f"{context}: constituent rank {rank} is outside 1..{population}"
            )
        rank_counts[rank] += 1
    if usable_systems != len(ranks):
        raise ValueError(
            f"{context}: usable_systems={usable_systems} but retained "
            f"{len(ranks)} rank observations"
        )
    if not isinstance(pmf, list) or not pmf:
        raise ValueError(f"{context}: PMF must be a non-empty list")
    observed_points: dict[int, tuple[int, float]] = {}
    for point in pmf:
        try:
            rank = point["rank"]
            count = point["count"]
            probability = point["probability"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{context}: malformed PMF point {point!r}") from exc
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
        ):
            raise TypeError(f"{context}: PMF rank/count must be integers: {point!r}")
        if not 1 <= rank <= population:
            raise ValueError(
                f"{context}: PMF rank {rank} is outside 1..{population}"
            )
        if rank in observed_points:
            raise ValueError(f"{context}: duplicate PMF rank {rank}")
        try:
            probability = float(probability)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: invalid PMF probability {point!r}") from exc
        if count < 1 or not math.isfinite(probability) or probability < 0:
            raise ValueError(f"{context}: invalid PMF count/probability {point!r}")
        observed_points[rank] = (count, probability)
    if set(observed_points) != set(rank_counts):
        raise ValueError(
            f"{context}: PMF support does not match retained rank observations"
        )
    total = 0.0
    for rank, count in rank_counts.items():
        pmf_count, probability = observed_points[rank]
        expected_probability = count / len(ranks)
        if pmf_count != count or not math.isclose(
            probability, expected_probability, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError(
                f"{context}: PMF rank {rank} disagrees with retained observations"
            )
        total += probability
    if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"{context}: PMF mass is {total}, expected 1")


def audit_population_coverage(
    observations: list[dict[str, str]],
    depth: list[dict[str, object]],
    distributions: list[dict[str, object]],
    threshold: float,
) -> list[dict[str, object]]:
    """Check that every CMP population team has a selected target distribution."""
    population_teams: dict[tuple[int, str], set[str]] = defaultdict(set)
    selected = {
        (int(row["season"]), str(row["subdivision"]), str(row["system_code"]))
        for row in depth
        if float(row["coverage_ratio"]) >= threshold
    }
    for row in observations:
        if row["is_composite"] == "True":
            population_teams[(int(row["season"]), row["subdivision"])].add(
                str(row["team_id"])
            )
    covered: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in distributions:
        covered[(int(row["season"]), str(row["subdivision"]))].add(
            str(row["team_id"])
        )
    result = []
    for key, teams in sorted(population_teams.items()):
        missing = sorted(teams - covered[key])
        result.append(
            {
                "season": key[0],
                "subdivision": key[1],
                "team_population": len(teams),
                "teams_with_selected_constituents": len(teams) - len(missing),
                "teams_missing_selected_constituents": len(missing),
                "coverage_ratio": (len(teams) - len(missing)) / len(teams),
                "selected_system_count": sum(
                    1
                    for item in selected
                    if item[:2] == key
                ),
                "missing_team_ids": json.dumps(missing, separators=(",", ":")),
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
    population_teams: dict[tuple[int, str], set[str]] = defaultdict(set)
    seen_composites: set[tuple[int, str, str]] = set()
    for row in observations:
        if row["is_composite"] == "True":
            team_id = str(row["team_id"]).strip()
            context = f"{row['season']} {row['subdivision']} CMP team {team_id}"
            if not team_id:
                raise ValueError(f"{context}: canonical team identity is required")
            if row["system_code"] != "CMP":
                raise ValueError(f"{context}: composite observation must use system_code CMP")
            identity = (int(row["season"]), str(row["subdivision"]), team_id)
            if identity in seen_composites:
                raise ValueError(f"{context}: duplicate CMP population observation")
            seen_composites.add(identity)
            population_teams[(int(row["season"]), row["subdivision"])].add(team_id)
        elif row["system_code"] == "CMP":
            raise ValueError(
                f"{row['season']} {row['subdivision']} team {row['team_id']}: "
                "CMP system observation is not marked composite"
            )
    population = {key: len(teams) for key, teams in population_teams.items()}
    for row in depth:
        key = (int(row["season"]), str(row["subdivision"]))
        declared = int(row["team_population"])
        if declared != population.get(key, 0):
            raise ValueError(
                f"{key[0]} {key[1]} {row['system_code']}: depth population "
                f"{declared} disagrees with {population.get(key, 0)} distinct CMP teams"
            )
    seen_observations: set[tuple[int, str, str, str]] = set()
    groups: dict[tuple[int, str, str], list[tuple[str, int, str]]] = defaultdict(list)
    for row in observations:
        key = (int(row["season"]), row["subdivision"], row["system_code"])
        if row["is_composite"] != "True" and key in selected:
            context = (
                f"{key[0]} {key[1]} team {row.get('team_id')} system {key[2]}"
            )
            team_id = str(row.get("team_id", "")).strip()
            if not team_id or not str(row.get("team_name", "")).strip():
                raise ValueError(f"{context}: canonical team identity is required")
            if team_id not in population_teams[(key[0], key[1])]:
                raise ValueError(
                    f"{context}: constituent observation has no CMP population member"
                )
            try:
                rank = int(row["ordinal_rank"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{context}: rank must be an integer") from exc
            if str(rank) != str(row["ordinal_rank"]).strip():
                raise ValueError(f"{context}: rank must be a canonical integer")
            n = population[(key[0], key[1])]
            if not 1 <= rank <= n:
                raise ValueError(
                    f"{context}: source rank {rank} is outside model population 1..{n}"
                )
            observation_key = (key[0], key[1], team_id, key[2])
            if observation_key in seen_observations:
                raise ValueError(f"{context}: duplicate team/system observation")
            seen_observations.add(observation_key)
            groups[(int(row["season"]), row["subdivision"], row["team_id"])].append(
                (row["team_name"], rank, row["system_code"])
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
        validate_distribution_row(rows[-1])
        by_team[(season, subdivision, team_id)] = [
            (code, rank) for _, rank, code in values
        ]
    return rows, by_team


def build_constituent_rank_pairs(
    games: list[dict[str, str]],
    by_team: dict[tuple[int, str, str], list[tuple[str, int]]],
    depth: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Join same-system rank observations for games without fitting a model."""
    populations: dict[tuple[int, str], int] = {}
    for row in depth:
        populations[(int(row["season"]), str(row["subdivision"]))] = int(
            row["team_population"]
        )
    result = []
    for game in games:
        home_subdivision = str(game.get("homeClassification", ""))
        away_subdivision = str(game.get("awayClassification", ""))
        if home_subdivision not in {"fbs", "fcs"} or away_subdivision not in {
            "fbs",
            "fcs",
        }:
            continue
        season = int(game["season"])
        home_key = (season, home_subdivision, str(game["homeId"]))
        away_key = (season, away_subdivision, str(game["awayId"]))
        home_ranks = dict(by_team.get(home_key, []))
        away_ranks = dict(by_team.get(away_key, []))
        for system_code in sorted(set(home_ranks) & set(away_ranks)):
            home_rank = home_ranks[system_code]
            away_rank = away_ranks[system_code]
            home_population = populations.get((season, home_subdivision))
            away_population = populations.get((season, away_subdivision))
            context = f"game {game.get('id')} season {season} system {system_code}"
            if home_population is None or away_population is None:
                raise ValueError(f"{context}: missing cohort population")
            for rank, population, side in (
                (home_rank, home_population, "home"),
                (away_rank, away_population, "away"),
            ):
                if not 1 <= rank <= population:
                    raise ValueError(
                        f"{context}: {side} rank {rank} is outside "
                        f"1..{population}"
                    )
            result.append(
                {
                    "game_id": game["id"],
                    "season": season,
                    "system_code": system_code,
                    "home_team_id": home_key[2],
                    "home_subdivision": home_subdivision,
                    "home_rank": home_rank,
                    "home_team_population": home_population,
                    "home_rank_percentile": (home_rank - 0.5) / home_population,
                    "away_team_id": away_key[2],
                    "away_subdivision": away_subdivision,
                    "away_rank": away_rank,
                    "away_team_population": away_population,
                    "away_rank_percentile": (away_rank - 0.5) / away_population,
                }
            )
    return result


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
    maxiter: int = 1000,
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
        "optimizer_status": int(result.status),
        "optimizer_nit": int(result.nit),
        "optimizer_nfev": int(result.nfev),
    }


def game_log_scores(
    target: np.ndarray,
    locations: np.ndarray,
    groups: np.ndarray,
    scale: float,
    df: float,
) -> dict[str, float]:
    """Return equal-game expected conditional and marginalized NLLs."""
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    logs = _log_student_t(target, locations, scale, df)
    expected = np.bincount(inverse, weights=logs) / counts
    maxima = np.full(len(counts), -np.inf)
    np.maximum.at(maxima, inverse, logs)
    sums = np.bincount(inverse, weights=np.exp(logs - maxima[inverse]))
    marginalized = maxima + np.log(sums) - np.log(counts)
    return {
        "expected_conditional_nll": float(-np.mean(expected)),
        "marginalized_nll": float(-np.mean(marginalized)),
        "n_games": len(counts),
    }
