"""Research-only offense/defense latent-state investigation.

This module deliberately does not participate in Posterior V1 or publication.
It uses the frozen V1 score-margin surface to construct a symmetric expected
score decomposition, tests whether the two residual components persist, and
provides a small, predeclared scoreboard-only OD candidate family.

The OD candidates are simple Gaussian MAP diagnostics, not a production
posterior.  For a game between ``i`` and ``j`` the score means are::

    E[points_i] = environment_i + O_i - D_j
    E[points_j] = environment_j + O_j - D_i

The environment is a training-only pairing/site score baseline.  Offense and
defense are separately sum-to-zero centered, so the usual ``O + c, D + c``
location non-identifiability is removed explicitly.  The predictive score
distribution is an independent Normal diagnostic with a frozen training
scale; V1 remains the Student-t margin baseline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import norm, t

from gippyrank.modeling import design_matrix
from gippyrank.posterior.engine import (
    LikelihoodV1,
    Team,
    game_margin_parameters,
    infer_posterior,
)
from gippyrank.research.margin_likelihood import pairing_for
from gippyrank.research.temporal_nonstationarity import (
    DEVELOPMENT_SEASONS,
    EVALUATION_SEASONS,
    TRAIN_SEASONS,
    HistoricalGame,
    PriorInput,
)

PERIODS = {
    "training": set(TRAIN_SEASONS),
    "development": set(DEVELOPMENT_SEASONS),
    "evaluation": set(EVALUATION_SEASONS),
}
MAX_CUTOFFS_PER_SEASON = 4
NULL_PERMUTATIONS = 10
QUALITY_SCALE = 20.0
PRIOR_SD = 8.0
EPSILON = np.finfo(float).tiny


@dataclass(frozen=True)
class ScoreEnvironment:
    """Frozen training-era scoring baselines keyed by pairing and site."""

    score_means: Mapping[tuple[str, bool, bool], tuple[float, float]]
    fallback_mean: tuple[float, float]
    scale: float
    training_seasons: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.scale <= 0 or not math.isfinite(self.scale):
            raise ValueError("score environment scale must be finite and positive")
        if any(
            len(value) != 2 or not np.isfinite(value).all()
            for value in self.score_means.values()
        ):
            raise ValueError("score environment means must be finite home/away pairs")

    def key(self, game: HistoricalGame) -> tuple[str, bool, bool]:
        cross = game.home_subdivision != game.away_subdivision
        return (
            pairing_for(game.home_subdivision, game.away_subdivision),
            bool(game.neutral_site),
            bool(cross and game.home_subdivision == "fbs" and not game.neutral_site),
        )

    def baseline_scores(self, game: HistoricalGame) -> tuple[float, float]:
        return tuple(self.score_means.get(self.key(game), self.fallback_mean))  # type: ignore[return-value]


@dataclass(frozen=True)
class ScoreDecomposition:
    """Symmetric V1 expected-score and team-level residual decomposition."""

    expected_home_points: float
    expected_away_points: float
    home_offensive_residual: float
    away_offensive_residual: float
    home_defensive_residual: float
    away_defensive_residual: float
    expected_margin: float
    expected_total: float


@dataclass(frozen=True)
class ODCandidate:
    """One frozen scoreboard-only candidate."""

    name: str
    prior_correlation: float
    prior_sd: float = PRIOR_SD


@dataclass(frozen=True)
class ODState:
    team_id: str
    team_name: str
    subdivision: str
    offense: float
    defense: float
    scalar_quality: float
    games_seen: int


@dataclass(frozen=True)
class ODFit:
    candidate: str
    states: Mapping[str, ODState]
    score_scale: float
    cutoff: str | None
    training_game_ids: tuple[str, ...]


def candidate_grid() -> tuple[ODCandidate, ...]:
    """Return the predeclared candidate family in stable order."""

    return (
        ODCandidate("OD0", prior_correlation=0.0),
        ODCandidate("OD1", prior_correlation=0.25),
    )


def _environment_key(game: HistoricalGame) -> tuple[str, bool, bool]:
    cross = game.home_subdivision != game.away_subdivision
    return (
        pairing_for(game.home_subdivision, game.away_subdivision),
        bool(game.neutral_site),
        bool(cross and game.home_subdivision == "fbs" and not game.neutral_site),
    )


def build_score_environment(
    games: Sequence[HistoricalGame],
    training_seasons: Iterable[int] = TRAIN_SEASONS,
) -> ScoreEnvironment:
    """Fit only fixed pairing/site score means from the training seasons."""

    seasons = tuple(sorted({int(value) for value in training_seasons}))
    selected = [game for game in games if game.season in seasons]
    if not selected:
        raise ValueError("score environment requires at least one training game")
    values: dict[tuple[str, bool, bool], list[tuple[float, float]]] = defaultdict(list)
    for game in selected:
        values[_environment_key(game)].append(
            (float(game.home_points), float(game.away_points))
        )
    means = {
        key: (float(np.mean([row[0] for row in rows])), float(np.mean([row[1] for row in rows])))
        for key, rows in sorted(values.items())
    }
    all_scores = np.asarray(
        [(float(game.home_points), float(game.away_points)) for game in selected],
        dtype=float,
    )
    fallback = (float(all_scores[:, 0].mean()), float(all_scores[:, 1].mean()))
    residuals = []
    for game in selected:
        baseline = means[_environment_key(game)]
        residuals.extend(
            [float(game.home_points) - baseline[0], float(game.away_points) - baseline[1]]
        )
    scale = max(float(np.std(residuals, ddof=0)), 5.0)
    return ScoreEnvironment(means, fallback, scale, seasons)


def _v1_locations_for_rank_pairs(
    game: HistoricalGame, likelihood: LikelihoodV1
) -> np.ndarray:
    pairs = np.asarray(game.rank_pairs, dtype=float)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs):
        raise ValueError(f"game {game.game_id} has no valid rank pairs")
    home_coordinate = (pairs[:, 0] - 0.5) / game.home_population
    away_coordinate = (pairs[:, 1] - 0.5) / game.away_population
    if game.home_subdivision != game.away_subdivision and game.home_subdivision == "fcs":
        x, y = away_coordinate, home_coordinate
    else:
        x, y = home_coordinate, away_coordinate
    cross = game.home_subdivision != game.away_subdivision
    pairing = np.full(len(pairs), pairing_for(game.home_subdivision, game.away_subdivision))
    matrix = design_matrix(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        pairing,
        np.full(len(pairs), float(not game.neutral_site)),
        np.full(len(pairs), float(game.neutral_site)),
        surface=True,
        fbs_home=np.full(
            len(pairs), float(cross and game.home_subdivision == "fbs" and not game.neutral_site)
        ),
    )
    if len(likelihood.beta) != matrix.shape[1]:
        raise ValueError(
            f"V1 beta has {len(likelihood.beta)} coefficients; expected {matrix.shape[1]}"
        )
    return matrix @ likelihood.beta


def v1_expected_margin(game: HistoricalGame, likelihood: LikelihoodV1) -> float:
    """Return the V1 margin location averaged over the game's rank pairs."""

    return float(np.mean(_v1_locations_for_rank_pairs(game, likelihood)))


def expected_score_decomposition(
    game: HistoricalGame,
    likelihood: LikelihoodV1,
    environment: ScoreEnvironment,
) -> ScoreDecomposition:
    """Split a V1 expected margin around a training-only total environment.

    For same-subdivision games the oriented margin is home minus away.  For a
    cross-subdivision game it is FBS minus FCS, independent of schedule order.
    The expected total is the sum of the training-only home/away environment
    means.  This gives both sides exactly half of the V1 margin and therefore
    never silently assigns all of a margin to one component.
    """

    expected_margin = v1_expected_margin(game, likelihood)
    base_home, base_away = environment.baseline_scores(game)
    expected_total = base_home + base_away
    fbs_home_orientation = (
        game.home_subdivision == game.away_subdivision
        or game.home_subdivision == "fbs"
    )
    first_points = (expected_total + expected_margin) / 2.0
    second_points = (expected_total - expected_margin) / 2.0
    if fbs_home_orientation:
        expected_home, expected_away = first_points, second_points
    else:
        expected_home, expected_away = second_points, first_points
    return ScoreDecomposition(
        expected_home_points=float(expected_home),
        expected_away_points=float(expected_away),
        home_offensive_residual=float(game.home_points - expected_home),
        away_offensive_residual=float(game.away_points - expected_away),
        home_defensive_residual=float(expected_away - game.away_points),
        away_defensive_residual=float(expected_home - game.home_points),
        expected_margin=expected_margin,
        expected_total=expected_total,
    )


def residual_rows(
    games: Sequence[HistoricalGame],
    likelihood: LikelihoodV1,
    environment: ScoreEnvironment,
    *,
    cutoff: datetime | date | None = None,
) -> list[dict[str, object]]:
    """Construct focal-team residuals without using games at/after ``cutoff``."""

    cutoff_value = _as_utc(cutoff) if cutoff is not None else None
    rows: list[dict[str, object]] = []
    for game in sorted(games, key=lambda item: (item.start, item.game_id)):
        if cutoff_value is not None and _as_utc(game.start) >= cutoff_value:
            continue
        decomposition = expected_score_decomposition(game, likelihood, environment)
        common = {
            "season": game.season,
            "game_id": game.game_id,
            "game_date": game.start.isoformat(),
            "week": game.week,
            "pairing": pairing_for(game.home_subdivision, game.away_subdivision),
            "neutral_site": game.neutral_site,
            "expected_margin": decomposition.expected_margin,
            "expected_total": decomposition.expected_total,
        }
        rows.extend(
            [
                {
                    **common,
                    "team_id": game.home_id,
                    "team_name": game.home_name,
                    "subdivision": game.home_subdivision,
                    "opponent_id": game.away_id,
                    "opponent_name": game.away_name,
                    "points_for": game.home_points,
                    "points_against": game.away_points,
                    "offensive_residual": decomposition.home_offensive_residual,
                    "defensive_residual": decomposition.home_defensive_residual,
                },
                {
                    **common,
                    "team_id": game.away_id,
                    "team_name": game.away_name,
                    "subdivision": game.away_subdivision,
                    "opponent_id": game.home_id,
                    "opponent_name": game.home_name,
                    "points_for": game.away_points,
                    "points_against": game.home_points,
                    "offensive_residual": decomposition.away_offensive_residual,
                    "defensive_residual": decomposition.away_defensive_residual,
                },
            ]
        )
    return rows


def _as_utc(value: datetime | date) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return datetime.combine(value, datetime.min.time(), tzinfo=UTC)


def _period(season: int) -> str:
    for name, values in PERIODS.items():
        if season in values:
            return name
    return "other"


def _correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    x_values, y_values = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(x_values) == 0 or np.std(y_values) == 0:
        return None
    return float(np.corrcoef(x_values, y_values)[0, 1])


def _metric(
    pairs: Sequence[tuple[float, float]],
    *,
    period: str,
    season: int | str,
    scale: str,
    relation: str,
    control: str,
    lag: int,
    bin_name: str,
) -> dict[str, object]:
    x = [pair[0] for pair in pairs]
    y = [pair[1] for pair in pairs]
    correlation = _correlation(x, y)
    slope = None
    if len(x) >= 2 and np.std(x):
        slope = float(np.polyfit(x, y, 1)[0])
    return {
        "period": period,
        "season": season,
        "scale": scale,
        "relation": relation,
        "control": control,
        "lag": lag,
        "bin": bin_name,
        "n_pairs": len(pairs),
        "correlation": correlation,
        "slope": slope,
    }


def _elapsed_bin(days: float) -> str:
    if days <= 7:
        return "0-7d"
    if days <= 21:
        return "8-21d"
    if days <= 56:
        return "22-56d"
    if days <= 112:
        return "57-112d"
    return "113d+"


def stage0_component_persistence(
    residuals: Sequence[Mapping[str, object]],
    *,
    seed: int = 40,
    permutations: int = NULL_PERMUTATIONS,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Measure same/cross-component persistence and deterministic nulls."""

    groups: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in residuals:
        groups[(int(row["season"]), str(row["team_id"]))].append(row)
    for values in groups.values():
        values.sort(key=lambda row: (str(row["game_date"]), str(row["game_id"])))

    observed: dict[tuple[object, ...], list[tuple[float, float]]] = defaultdict(list)
    nulls: dict[tuple[object, ...], list[tuple[float, float]]] = defaultdict(list)
    elapsed: dict[tuple[object, ...], list[tuple[float, float]]] = defaultdict(list)
    early_late: list[dict[str, object]] = []
    relations = (
        ("offense", "offense"),
        ("defense", "defense"),
        ("offense", "defense"),
        ("defense", "offense"),
    )
    rng = np.random.default_rng(seed)

    def values_for(rows: Sequence[Mapping[str, object]], component: str, scale: str) -> np.ndarray:
        field = {
            "offense": "offensive_residual",
            "defense": "defensive_residual",
        }[component]
        values = np.asarray([float(row[field]) for row in rows])
        return values if scale == "raw" else values - values.mean()

    for (season, team_id), rows in sorted(groups.items()):
        if len(rows) < 3:
            continue
        period = _period(season)
        for scale in ("raw", "demeaned"):
            for source, target in relations:
                for lag in (1, 2):
                    source_values = values_for(rows, source, scale)
                    target_values = values_for(rows, target, scale)
                    pairs = [
                        (float(source_values[i]), float(target_values[i + lag]))
                        for i in range(len(rows) - lag)
                    ]
                    observed[(period, season, scale, f"{source}->{target}", lag)].extend(pairs)
                    for i in range(len(rows) - lag):
                        days = (
                            _as_utc(datetime.fromisoformat(str(rows[i + lag]["game_date"])))
                            - _as_utc(datetime.fromisoformat(str(rows[i]["game_date"])))
                        ).total_seconds() / 86400.0
                        elapsed[(period, scale, f"{source}->{target}", lag, _elapsed_bin(days))].append(
                            (float(source_values[i]), float(target_values[i + lag]))
                        )
            # Within-team-season order shuffling is a deterministic null.  The
            # same component values are retained; only temporal ordering moves.
            for _ in range(permutations):
                order = rng.permutation(len(rows))
                shuffled = [rows[index] for index in order]
                for source, target in relations:
                    source_values = values_for(shuffled, source, scale)
                    target_values = values_for(shuffled, target, scale)
                    for lag in (1, 2):
                        nulls[(period, scale, f"{source}->{target}", "shuffle_order", lag)].extend(
                            (float(source_values[i]), float(target_values[i + lag]))
                            for i in range(len(rows) - lag)
                        )
        raw_offense = values_for(rows, "offense", "raw")
        raw_defense = values_for(rows, "defense", "raw")
        split = len(rows) // 2
        if split and len(rows) - split:
            early_late.append(
                {
                    "period": period,
                    "season": season,
                    "team_id": team_id,
                    "early_offense_mean": float(raw_offense[:split].mean()),
                    "late_offense_mean": float(raw_offense[split:].mean()),
                    "early_defense_mean": float(raw_defense[:split].mean()),
                    "late_defense_mean": float(raw_defense[split:].mean()),
                    "n_games": len(rows),
                }
            )

    # Pair unrelated teams within each season after a deterministic rotation.
    by_season: dict[int, list[list[Mapping[str, object]]]] = defaultdict(list)
    for (season, _team_id), values in groups.items():
        if len(values) >= 3:
            by_season[season].append(values)
    for season, sequences in sorted(by_season.items()):
        period = _period(season)
        for permutation in range(max(1, permutations)):
            ordered = list(sequences)
            if len(ordered) > 1:
                shift = (permutation + 1) % len(ordered)
                ordered = ordered[shift:] + ordered[:shift]
            shuffled = [sequences[index] for index in rng.permutation(len(sequences))]
            for left, right in zip(sequences, ordered, strict=True):
                if left is right:
                    continue
                for scale in ("raw", "demeaned"):
                    for source, target in relations:
                        source_values = values_for(left, source, scale)
                        target_values = values_for(right, target, scale)
                        for lag in (1, 2):
                            n = min(len(source_values), len(target_values)) - lag
                            if n > 0:
                                nulls[(period, scale, f"{source}->{target}", "unrelated_team", lag)].extend(
                                    (float(source_values[i]), float(target_values[i + lag]))
                                    for i in range(n)
                                )
            for left, right in zip(sequences, shuffled, strict=True):
                if left is right:
                    continue
                for scale in ("raw", "demeaned"):
                    for source, target in relations:
                        source_values = values_for(left, source, scale)
                        target_values = values_for(right, target, scale)
                        for lag in (1, 2):
                            n = min(len(source_values), len(target_values)) - lag
                            if n > 0:
                                nulls[(period, scale, f"{source}->{target}", "shuffle_components", lag)].extend(
                                    (float(source_values[i]), float(target_values[i + lag]))
                                    for i in range(n)
                                )

    metrics: list[dict[str, object]] = []
    for (period, season, scale, relation, lag), pairs in sorted(observed.items(), key=str):
        metrics.append(
            _metric(
                pairs,
                period=period,
                season=season,
                scale=scale,
                relation=relation,
                control="observed",
                lag=lag,
                bin_name="all",
            )
        )
    combined_observed: dict[tuple[object, ...], list[tuple[float, float]]] = defaultdict(list)
    for (period, _season, scale, relation, lag), pairs in observed.items():
        combined_observed[(period, scale, relation, lag)].extend(pairs)
    for (period, scale, relation, lag), pairs in sorted(combined_observed.items(), key=str):
        metrics.append(
            _metric(
                pairs,
                period=period,
                season="all",
                scale=scale,
                relation=relation,
                control="observed",
                lag=lag,
                bin_name="all",
            )
        )
    for (period, scale, relation, control, lag), pairs in sorted(nulls.items(), key=str):
        metrics.append(
            _metric(
                pairs,
                period=period,
                season="all",
                scale=scale,
                relation=relation,
                control=control,
                lag=lag,
                bin_name="all",
            )
        )
    for (period, scale, relation, lag, bin_name), pairs in sorted(elapsed.items(), key=str):
        metrics.append(
            _metric(
                pairs,
                period=period,
                season="all",
                scale=scale,
                relation=relation,
                control="observed",
                lag=lag,
                bin_name=bin_name,
            )
        )
    early_late_summary = {}
    for period in sorted({str(row["period"]) for row in early_late}):
        values = [row for row in early_late if row["period"] == period]
        early_late_summary[period] = {
            "n": len(values),
            "offense_correlation": _correlation(
                [float(row["early_offense_mean"]) for row in values],
                [float(row["late_offense_mean"]) for row in values],
            ),
            "defense_correlation": _correlation(
                [float(row["early_defense_mean"]) for row in values],
                [float(row["late_defense_mean"]) for row in values],
            ),
        }
    summary = {
        "n_focal_team_games": len(residuals),
        "n_team_seasons_with_at_least_3_games": sum(len(rows) >= 3 for rows in groups.values()),
        "null_seed": seed,
        "null_permutations": permutations,
        "early_late_by_period": early_late_summary,
    }
    return metrics, early_late, summary


def _team_quality(prior: PriorInput | None, population: int) -> float:
    if prior is None or len(prior.pmf) == 0:
        return 0.0
    pmf = np.asarray(prior.pmf, dtype=float)
    pmf = pmf / pmf.sum()
    ranks = np.arange(1, len(pmf) + 1, dtype=float)
    percentile = float(np.sum(pmf * (ranks - 0.5) / max(population, len(pmf))))
    return QUALITY_SCALE * (0.5 - percentile)


def _team_metadata(
    games: Sequence[HistoricalGame],
) -> dict[str, tuple[str, str, int]]:
    result: dict[str, tuple[str, str, int]] = {}
    for game in games:
        result[game.home_id] = (game.home_name, game.home_subdivision, game.home_population)
        result[game.away_id] = (game.away_name, game.away_subdivision, game.away_population)
    return result


def _centered_reduction(n: int) -> np.ndarray:
    if n < 2:
        return np.zeros((n, 0))
    result = np.zeros((n, n - 1))
    result[: n - 1, :] = np.eye(n - 1)
    result[-1, :] = -1.0
    return result


def _od_design(
    first: str,
    second: str,
    team_index: Mapping[str, int],
    reduction: np.ndarray,
) -> np.ndarray:
    n = len(team_index)
    row = np.zeros(2 * max(n - 1, 0))
    if n < 2:
        return row
    for team_id, sign, offset in (
        (first, 1.0, 0),
        (second, -1.0, n - 1),
    ):
        index = team_index[team_id]
        row[offset : offset + n - 1] += sign * reduction[index]
    return row


def fit_od_model(
    games: Sequence[HistoricalGame],
    priors: Mapping[tuple[int, str], PriorInput],
    environment: ScoreEnvironment,
    candidate: ODCandidate,
    *,
    team_metadata: Mapping[str, tuple[str, str, int]] | None = None,
    cutoff: datetime | date | str | None = None,
) -> ODFit:
    """Fit a centered scoreboard-only OD MAP model using only ``games``."""

    metadata = dict(team_metadata or _team_metadata(games))
    for game in games:
        metadata.setdefault(game.home_id, (game.home_name, game.home_subdivision, game.home_population))
        metadata.setdefault(game.away_id, (game.away_name, game.away_subdivision, game.away_population))
    team_ids = sorted(metadata)
    if len(team_ids) < 2:
        states = {
            team_id: ODState(team_id, *metadata[team_id][:2], 0.0, 0.0, 0.0)
            for team_id in team_ids
        }
        return ODFit(candidate.name, states, environment.scale, _cutoff_string(cutoff), ())
    index = {team_id: position for position, team_id in enumerate(team_ids)}
    reduction = _centered_reduction(len(team_ids))
    design_rows: list[np.ndarray] = []
    targets: list[float] = []
    seen: dict[str, int] = defaultdict(int)
    for game in games:
        base_home, base_away = environment.baseline_scores(game)
        design_rows.append(_od_design(game.home_id, game.away_id, index, reduction))
        targets.append(float(game.home_points) - base_home)
        design_rows.append(_od_design(game.away_id, game.home_id, index, reduction))
        targets.append(float(game.away_points) - base_away)
        seen[game.home_id] += 1
        seen[game.away_id] += 1
    matrix = np.asarray(design_rows, dtype=float)
    target = np.asarray(targets, dtype=float)
    n = len(team_ids)
    prior_full = np.zeros(2 * n)
    for team_id, position in index.items():
        name, subdivision, population = metadata[team_id]
        prior = priors.get((games[0].season, team_id)) if games else None
        quality = _team_quality(prior, population)
        # A scalar quality prior is deliberately split symmetrically.  OD
        # evidence, rather than a hidden offseason decomposition, separates it.
        prior_full[position] = quality / 2.0
        prior_full[n + position] = quality / 2.0
    prior_full[:n] -= prior_full[:n].mean()
    prior_full[n:] -= prior_full[n:].mean()
    precision_full = np.zeros((2 * n, 2 * n))
    rho = float(candidate.prior_correlation)
    if abs(rho) >= 1:
        raise ValueError("OD prior correlation must lie strictly between -1 and 1")
    block = np.asarray([[1.0, -rho], [-rho, 1.0]]) / (
        candidate.prior_sd**2 * (1.0 - rho**2)
    )
    for position in range(n):
        indexes = [position, n + position]
        precision_full[np.ix_(indexes, indexes)] = block
    reduction_full = np.zeros((2 * n, 2 * (n - 1)))
    reduction_full[:n, : n - 1] = reduction
    reduction_full[n:, n - 1 :] = reduction
    prior_reduced = np.concatenate([prior_full[: n - 1], prior_full[n : 2 * n - 1]])
    precision = reduction_full.T @ precision_full @ reduction_full
    weighted_gram = matrix.T @ matrix / environment.scale**2
    rhs = matrix.T @ target / environment.scale**2 + precision @ prior_reduced
    normal = weighted_gram + precision
    try:
        fitted = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        fitted = np.linalg.lstsq(normal, rhs, rcond=None)[0]
    full = reduction_full @ fitted
    states = {}
    for team_id, position in index.items():
        name, subdivision, _population = metadata[team_id]
        prior = priors.get((games[0].season, team_id)) if games else None
        quality = _team_quality(prior, metadata[team_id][2])
        states[team_id] = ODState(
            team_id,
            name,
            subdivision,
            float(full[position]),
            float(full[n + position]),
            quality,
            seen[team_id],
        )
    if not np.isclose(np.mean([state.offense for state in states.values()]), 0.0, atol=1e-8):
        raise AssertionError("offensive centering constraint failed")
    if not np.isclose(np.mean([state.defense for state in states.values()]), 0.0, atol=1e-8):
        raise AssertionError("defensive centering constraint failed")
    return ODFit(
        candidate.name,
        states,
        environment.scale,
        _cutoff_string(cutoff),
        tuple(game.game_id for game in games),
    )


def _cutoff_string(value: datetime | date | str | None) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else _as_utc(value).isoformat()


def _actual_oriented_margin(game: HistoricalGame) -> float:
    return float(game.margin)


def od_predictive_scores(
    game: HistoricalGame,
    fit: ODFit,
    environment: ScoreEnvironment,
) -> dict[str, float]:
    """Return honest Normal score/margin predictive metrics for an OD fit."""

    base_home, base_away = environment.baseline_scores(game)
    home = fit.states.get(game.home_id)
    away = fit.states.get(game.away_id)
    home_offense = home.offense if home else 0.0
    home_defense = home.defense if home else 0.0
    away_offense = away.offense if away else 0.0
    away_defense = away.defense if away else 0.0
    expected_home = base_home + home_offense - away_defense
    expected_away = base_away + away_offense - home_defense
    if game.home_subdivision == game.away_subdivision or game.home_subdivision == "fbs":
        expected_margin = expected_home - expected_away
    else:
        expected_margin = expected_away - expected_home
    margin_scale = math.sqrt(2.0) * fit.score_scale
    actual_margin = _actual_oriented_margin(game)
    actual_win = float(actual_margin > 0)
    win_probability = float(norm.cdf(expected_margin / margin_scale))
    total_expected = expected_home + expected_away
    actual_total = float(game.home_points + game.away_points)
    total_scale = margin_scale
    return {
        "margin_nll": float(-norm.logpdf(actual_margin, expected_margin, margin_scale)),
        "margin_mae": abs(expected_margin - actual_margin),
        "win_brier": (win_probability - actual_win) ** 2,
        "home_score_nll": float(-norm.logpdf(game.home_points, expected_home, fit.score_scale)),
        "away_score_nll": float(-norm.logpdf(game.away_points, expected_away, fit.score_scale)),
        "home_score_mae": abs(expected_home - float(game.home_points)),
        "away_score_mae": abs(expected_away - float(game.away_points)),
        "total_points_nll": float(-norm.logpdf(actual_total, total_expected, total_scale)),
        "total_points_mae": abs(total_expected - actual_total),
        "expected_margin": float(expected_margin),
        "win_probability": win_probability,
        "actual_margin": actual_margin,
        "expected_home_points": float(expected_home),
        "expected_away_points": float(expected_away),
        "actual_total_points": actual_total,
    }


def scalar_predictive_scores(
    game: HistoricalGame,
    posterior: Mapping[str, np.ndarray],
    likelihood: LikelihoodV1,
) -> dict[str, float]:
    """Score a target with the unchanged Posterior V1 Student-t margin model."""

    home = Team(game.home_id, game.home_name, game.home_subdivision, posterior[game.home_id])  # type: ignore[arg-type]
    away = Team(game.away_id, game.away_name, game.away_subdivision, posterior[game.away_id])  # type: ignore[arg-type]
    locations, _margin = game_margin_parameters(game.as_engine_game(), home, away, likelihood)
    joint = home.prior[:, None] * away.prior[None, :]
    actual = _actual_oriented_margin(game)
    density = t.pdf((actual - locations) / likelihood.scale, likelihood.degrees_of_freedom) / likelihood.scale
    predictive_density = float(np.sum(joint * density))
    expected = float(np.sum(joint * locations))
    win_probability = float(np.sum(joint * t.cdf(locations / likelihood.scale, likelihood.degrees_of_freedom)))
    return {
        "margin_nll": float(-np.log(max(predictive_density, EPSILON))),
        "margin_mae": abs(expected - actual),
        "win_brier": (win_probability - float(actual > 0)) ** 2,
        "expected_margin": expected,
        "win_probability": win_probability,
        "actual_margin": actual,
    }


def _cutoffs(season_games: Sequence[HistoricalGame]) -> list[datetime]:
    by_week: dict[int, list[datetime]] = defaultdict(list)
    for game in season_games:
        by_week[game.week].append(_as_utc(game.start))
    weeks = sorted(by_week)
    eligible = weeks[1:-1]
    if len(eligible) > MAX_CUTOFFS_PER_SEASON:
        indexes = np.linspace(0, len(eligible) - 1, MAX_CUTOFFS_PER_SEASON, dtype=int)
        eligible = [eligible[index] for index in indexes]
    return [max(by_week[week]) + timedelta(microseconds=1) for week in eligible]


def _prior_for_team(
    priors: Mapping[tuple[int, str], PriorInput],
    season: int,
    team_id: str,
    population: int,
) -> np.ndarray:
    prior = priors.get((season, team_id))
    if prior is None:
        return np.full(population, 1.0 / population)
    values = np.asarray(prior.pmf, dtype=float)
    if len(values) != population:
        values = np.resize(values, population)
    values = np.maximum(values, 0.0)
    return values / values.sum()


def _scalar_posterior(
    season_games: Sequence[HistoricalGame],
    past: Sequence[HistoricalGame],
    priors: Mapping[tuple[int, str], PriorInput],
    likelihood: LikelihoodV1,
) -> dict[str, np.ndarray]:
    metadata = _team_metadata(season_games)
    teams = [
        Team(
            team_id,
            metadata[team_id][0],
            metadata[team_id][1],  # type: ignore[arg-type]
            _prior_for_team(priors, season_games[0].season, team_id, metadata[team_id][2]),
        )
        for team_id in sorted(metadata)
    ]
    result = infer_posterior(
        teams,
        [game.as_engine_game() for game in past],
        likelihood,
        max_iterations=75,
        tolerance=1e-3,
    )
    if not result.converged:
        raise RuntimeError(f"Posterior V1 did not converge for {season_games[0].season}")
    return result.pmfs


def _phase(season_games: Sequence[HistoricalGame], cutoff: datetime) -> str:
    first = min(_as_utc(game.start) for game in season_games)
    last = max(_as_utc(game.start) for game in season_games)
    fraction = (cutoff - first).total_seconds() / max((last - first).total_seconds(), 1.0)
    return "early" if fraction < 1 / 3 else "mid" if fraction < 2 / 3 else "late"


def evaluate_candidates(
    games: Sequence[HistoricalGame],
    priors: Mapping[tuple[int, str], PriorInput],
    likelihood: LikelihoodV1,
    environment: ScoreEnvironment,
    candidates: Sequence[ODCandidate],
    seasons: Iterable[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate identical next-game keys for V1 and the supplied OD candidates."""

    season_set = {int(value) for value in seasons}
    by_season: dict[int, list[HistoricalGame]] = defaultdict(list)
    for game in games:
        if game.season in season_set:
            by_season[game.season].append(game)
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    for season, season_games in sorted(by_season.items()):
        season_games = sorted(season_games, key=lambda item: (_as_utc(item.start), item.game_id))
        metadata = _team_metadata(season_games)
        by_team: dict[str, list[HistoricalGame]] = defaultdict(list)
        for game in season_games:
            by_team[game.home_id].append(game)
            by_team[game.away_id].append(game)
        for cutoff_index, cutoff in enumerate(_cutoffs(season_games), start=1):
            past = [game for game in season_games if _as_utc(game.start) < cutoff]
            if not past:
                continue
            targets: dict[str, HistoricalGame] = {}
            for team_games in by_team.values():
                future = [game for game in team_games if _as_utc(game.start) > cutoff]
                if future:
                    targets[future[0].game_id] = future[0]
            if not targets:
                continue
            posterior = _scalar_posterior(season_games, past, priors, likelihood)
            baseline_rows = []
            for target in sorted(targets.values(), key=lambda item: (_as_utc(item.start), item.game_id)):
                scores = scalar_predictive_scores(target, posterior, likelihood)
                baseline_rows.append(
                    {
                        "season": season,
                        "cutoff": cutoff.isoformat(),
                        "cutoff_index": cutoff_index,
                        "phase": _phase(season_games, cutoff),
                        "candidate": "Posterior V1",
                        "target_kind": "next_game",
                        "target_game_id": target.game_id,
                        "target_date": target.start.isoformat(),
                        "home_team_id": target.home_id,
                        "away_team_id": target.away_id,
                        **scores,
                    }
                )
            rows.extend(baseline_rows)
            for candidate in candidates:
                fit = fit_od_model(
                    past,
                    priors,
                    environment,
                    candidate,
                    team_metadata=metadata,
                    cutoff=cutoff,
                )
                state_values = list(fit.states.values())
                scalar_values = np.asarray([state.scalar_quality for state in state_values])
                offenses = np.asarray([state.offense for state in state_values])
                defenses = np.asarray([state.defense for state in state_values])
                diagnostics.append(
                    {
                        "season": season,
                        "cutoff": cutoff.isoformat(),
                        "candidate": candidate.name,
                        "n_teams": len(state_values),
                        "n_games": len(past),
                        "offense_defense_correlation": _correlation(offenses, defenses),
                        "offense_scalar_correlation": _correlation(offenses, scalar_values),
                        "defense_scalar_correlation": _correlation(defenses, scalar_values),
                        "offense_minus_defense_variance": float(np.var(offenses - defenses)),
                        "center_offense": float(np.mean(offenses)),
                        "center_defense": float(np.mean(defenses)),
                    }
                )
                for state in state_values:
                    profiles.append(
                        {
                            "season": season,
                            "cutoff": cutoff.isoformat(),
                            "candidate": candidate.name,
                            "team_id": state.team_id,
                            "team_name": state.team_name,
                            "subdivision": state.subdivision,
                            "scalar_quality": state.scalar_quality,
                            "offense": state.offense,
                            "defense": state.defense,
                            "offense_minus_defense": state.offense - state.defense,
                            "games_seen": state.games_seen,
                        }
                    )
                for target in sorted(targets.values(), key=lambda item: (_as_utc(item.start), item.game_id)):
                    scores = od_predictive_scores(target, fit, environment)
                    rows.append(
                        {
                            "season": season,
                            "cutoff": cutoff.isoformat(),
                            "cutoff_index": cutoff_index,
                            "phase": _phase(season_games, cutoff),
                            "candidate": candidate.name,
                            "target_kind": "next_game",
                            "target_game_id": target.game_id,
                            "target_date": target.start.isoformat(),
                            "home_team_id": target.home_id,
                            "away_team_id": target.away_id,
                            **scores,
                        }
                    )
    keys_by_candidate: dict[str, set[tuple[object, ...]]] = defaultdict(set)
    for row in rows:
        keys_by_candidate[str(row["candidate"])].add(
            (row["season"], row["cutoff"], row["target_kind"], row["target_game_id"])
        )
    if keys_by_candidate:
        expected = next(iter(keys_by_candidate.values()))
        if any(keys != expected for keys in keys_by_candidate.values()):
            raise RuntimeError("V1 and OD candidate scoring keys differ")
    return rows, diagnostics, profiles


def aggregate_future_metrics(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str | None], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["candidate"]), str(row["target_kind"]), int(row["season"]), None)].append(row)
    result = []
    for (candidate, target_kind, season, _phase_name), values in sorted(grouped.items()):
        result.append(
            {
                "candidate": candidate,
                "target_kind": target_kind,
                "season": season,
                "n": len(values),
                "margin_nll": float(np.mean([float(row["margin_nll"]) for row in values])),
                "margin_mae": float(np.mean([float(row["margin_mae"]) for row in values])),
                "win_brier": float(np.mean([float(row["win_brier"]) for row in values])),
                "home_score_nll": _mean_optional(values, "home_score_nll"),
                "away_score_nll": _mean_optional(values, "away_score_nll"),
                "home_score_mae": _mean_optional(values, "home_score_mae"),
                "away_score_mae": _mean_optional(values, "away_score_mae"),
                "total_points_nll": _mean_optional(values, "total_points_nll"),
                "total_points_mae": _mean_optional(values, "total_points_mae"),
            }
        )
    return result


def aggregate_phase_metrics(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["candidate"]), str(row["target_kind"]), str(row["phase"]))].append(row)
    result = []
    for (candidate, target_kind, phase), values in sorted(grouped.items()):
        result.append(
            {
                "candidate": candidate,
                "target_kind": target_kind,
                "phase": phase,
                "n": len(values),
                "margin_nll": float(np.mean([float(row["margin_nll"]) for row in values])),
                "margin_mae": float(np.mean([float(row["margin_mae"]) for row in values])),
                "win_brier": float(np.mean([float(row["win_brier"]) for row in values])),
            }
        )
    return result


def _mean_optional(values: Sequence[Mapping[str, object]], key: str) -> float | None:
    observed = [float(row[key]) for row in values if key in row]
    return float(np.mean(observed)) if observed else None


def development_gate(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Apply the issue's frozen development gate using only 2018–2021."""

    dev = [row for row in rows if int(row["season"]) in DEVELOPMENT_SEASONS and row["target_kind"] == "next_game"]
    aggregate = aggregate_future_metrics(dev)
    deltas: list[dict[str, object]] = []
    for candidate in [item.name for item in candidate_grid()]:
        all_candidate = [item for item in aggregate if item["candidate"] == candidate]
        base_all = [item for item in aggregate if item["candidate"] == "Posterior V1"]
        n = sum(int(item["n"]) for item in all_candidate)
        baseline_n = sum(int(item["n"]) for item in base_all)
        candidate_nll = float(np.average([item["margin_nll"] for item in all_candidate], weights=[item["n"] for item in all_candidate]))
        candidate_mae = float(np.average([item["margin_mae"] for item in all_candidate], weights=[item["n"] for item in all_candidate]))
        candidate_brier = float(np.average([item["win_brier"] for item in all_candidate], weights=[item["n"] for item in all_candidate]))
        base_nll = float(np.average([item["margin_nll"] for item in base_all], weights=[item["n"] for item in base_all]))
        base_mae = float(np.average([item["margin_mae"] for item in base_all], weights=[item["n"] for item in base_all]))
        base_brier = float(np.average([item["win_brier"] for item in base_all], weights=[item["n"] for item in base_all]))
        by_season = {int(item["season"]): item for item in all_candidate}
        base_by_season = {int(item["season"]): item for item in base_all}
        nll_deltas = [float(by_season[season]["margin_nll"]) - float(base_by_season[season]["margin_nll"]) for season in DEVELOPMENT_SEASONS if season in by_season and season in base_by_season]
        deltas.append(
            {
                "candidate": candidate,
                "n": n,
                "baseline_n": baseline_n,
                "margin_nll": candidate_nll,
                "margin_mae": candidate_mae,
                "win_brier": candidate_brier,
                "baseline_margin_nll": base_nll,
                "baseline_margin_mae": base_mae,
                "baseline_win_brier": base_brier,
                "nll_improvement": base_nll - candidate_nll,
                "mae_improvement": base_mae - candidate_mae,
                "brier_delta": candidate_brier - base_brier,
                "seasons_nll_improved": sum(delta < 0 for delta in nll_deltas),
                "worst_season_nll_delta": max(nll_deltas, default=float("inf")),
                "passes_gate": bool(
                    base_nll - candidate_nll >= 0.010
                    and base_mae - candidate_mae >= 0.10
                    and candidate_brier - base_brier <= 0.001
                    and sum(delta < 0 for delta in nll_deltas) >= 3
                    and max(nll_deltas, default=float("inf")) <= 0.015
                ),
            }
        )
    return deltas


def select_candidate(gate_rows: Sequence[Mapping[str, object]]) -> str | None:
    qualifying = [row for row in gate_rows if bool(row["passes_gate"])]
    if not qualifying:
        return None
    best_nll = min(float(row["margin_nll"]) for row in qualifying)
    tied = [row for row in qualifying if float(row["margin_nll"]) - best_nll <= 0.002]
    complexity = {candidate.name: index for index, candidate in enumerate(candidate_grid())}
    return min(tied, key=lambda row: (complexity[str(row["candidate"])], float(row["margin_nll"])))["candidate"]  # type: ignore[return-value]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def production_hashes(root: Path) -> dict[str, str]:
    paths = {
        "historical_likelihood_v1": root / "data/processed/posterior/historical_likelihood_v1.json",
        "context_prior": root / "data/processed/preseason/context/predictions.csv",
        "history_prior": root / "data/processed/preseason/history/predictions.csv",
        "context_prior_2026": root / "data/processed/preseason/context/annual/2026/predictions.csv",
        "history_prior_2026": root / "data/processed/preseason/history/annual/2026/predictions.csv",
    }
    return {name: sha256(path) for name, path in paths.items() if path.exists()}


def load_likelihood(root: Path) -> LikelihoodV1:
    value = json.loads((root / "data/processed/posterior/historical_likelihood_v1.json").read_text())
    return LikelihoodV1(
        np.asarray(value["beta"], dtype=float),
        float(value["scale"]),
        float(value["degrees_of_freedom"]),
        value.get("fit_kind", "weighted_pseudo"),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
