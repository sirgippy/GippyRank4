"""Research utilities for the Issue #34 within-season drift study.

This module intentionally keeps the experiment outside the production ranking
path.  It consumes the frozen Historical Likelihood V1 surface and historical
rank/evidence artifacts, then applies a factor-strength probe only inside the
research runner.  A half-life is therefore a diagnostic approximation, not a
new production posterior.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.special import gammaln
from scipy.stats import t

from gippyrank.modeling import design_matrix
from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    Team,
    game_factor,
    game_margin_parameters,
    infer_posterior,
)
from gippyrank.preseason import (
    DirectRankModel,
    Preprocessor,
    TeamSeason,
    historical_rank_features,
    rank_to_z,
)

TRAIN_SEASONS = tuple(range(2008, 2018))
DEVELOPMENT_SEASONS = tuple(range(2018, 2022))
EVALUATION_SEASONS = tuple(range(2022, 2026))
CONTEXT_FEATURES = (
    "coach_tenure_seasons",
    "recruiting_class_rank",
    "recruiting_class_points",
    "recruiting_points_2y_mean",
    "recruiting_points_3y_mean",
    "recruiting_points_4y_mean",
    "recruiting_points_trend",
    "talent_composite",
    "returning_pct_ppa",
    "returning_pct_passing_ppa",
    "returning_pct_receiving_ppa",
    "returning_pct_rushing_ppa",
)
HISTORY_FEATURES = ("lag2_z_mean", "lag3_z_mean", "long_run_z_mean")
ALL_CONTEXT_FEATURES = (*HISTORY_FEATURES, *CONTEXT_FEATURES)
HALF_LIVES = (None, 14.0, 28.0, 56.0, 112.0)
CANDIDATE_NAMES = ("Static V1", "R14", "R28", "R56", "R112")
MAX_CUTOFFS_PER_SEASON = 4
EPSILON = np.finfo(float).tiny


@dataclass(frozen=True)
class TemporalCandidate:
    """One frozen factor-strength diagnostic candidate."""

    name: str
    half_life_days: float | None


@dataclass(frozen=True)
class HistoricalGame:
    """A normalized row from the existing historical modeling-game artifact."""

    game_id: str
    season: int
    start: datetime
    week: int
    home_id: str
    away_id: str
    home_name: str
    away_name: str
    home_subdivision: str
    away_subdivision: str
    home_population: int
    away_population: int
    home_points: int
    away_points: int
    margin: float
    neutral_site: bool
    home_ranks: np.ndarray
    away_ranks: np.ndarray
    rank_pairs: np.ndarray

    def as_engine_game(self) -> Game:
        return Game(
            self.game_id,
            self.home_id,
            self.away_id,
            self.home_subdivision,  # type: ignore[arg-type]
            self.away_subdivision,  # type: ignore[arg-type]
            self.home_points,
            self.away_points,
            self.neutral_site,
        )


@dataclass(frozen=True)
class PriorInput:
    season: int
    team_id: str
    team_name: str
    subdivision: str
    population: int
    pmf: np.ndarray
    source: str


def candidate_grid() -> tuple[TemporalCandidate, ...]:
    """Return the issue's immutable candidate grid in declaration order."""
    return tuple(
        TemporalCandidate(name, half_life)
        for name, half_life in zip(CANDIDATE_NAMES, HALF_LIVES, strict=True)
    )


def _as_utc(value: datetime | date) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return datetime.combine(value, datetime.min.time(), tzinfo=UTC)


def game_age_days(cutoff: datetime | date, game_time: datetime | date) -> float:
    """Return elapsed calendar time from a completed game to a cutoff.

    A game at or after the cutoff is rejected.  This makes the no-future-data
    contract explicit instead of relying on callers to remember a filter.
    """
    age = (_as_utc(cutoff) - _as_utc(game_time)).total_seconds() / 86400.0
    if age <= 0:
        raise ValueError("game must be strictly before the cutoff")
    return age


def recency_weight(
    cutoff: datetime | date, game_time: datetime | date, half_life_days: float | None
) -> float:
    """Return ``2 ** (-age / half_life)`` with exact static semantics."""
    if half_life_days is None:
        return 1.0
    if half_life_days <= 0 or not math.isfinite(half_life_days):
        raise ValueError("half_life_days must be finite and positive")
    return float(2.0 ** (-game_age_days(cutoff, game_time) / half_life_days))


def temper_factor(factor: np.ndarray, weight: float) -> np.ndarray:
    """Raise a nonnegative likelihood factor to a positive strength.

    Zero support remains zero and positive support remains positive.  Returning
    a copy for weight one also prevents a research call from mutating a cached
    production factor.
    """
    values = np.asarray(factor, dtype=float)
    if weight <= 0 or not math.isfinite(weight):
        raise ValueError("factor weight must be finite and positive")
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("factor must be finite and non-negative")
    if weight == 1:
        return values.copy()
    result = np.zeros_like(values)
    positive = values > 0
    result[positive] = values[positive] ** weight
    return result


def select_recency_candidate(
    development_metrics: dict[str, dict[str, object]],
) -> str | None:
    """Apply the frozen Issue #34 gate using development results only."""
    static = development_metrics["Static V1"]
    qualifying: list[tuple[str, float]] = []
    for candidate in candidate_grid()[1:]:
        metrics = development_metrics[candidate.name]
        if (
            float(static["next_game_nll"]) - float(metrics["next_game_nll"]) >= 0.010
            and float(static["next_game_mae"]) - float(metrics["next_game_mae"])
            >= 0.10
            and float(metrics["next_game_brier"])
            - float(static["next_game_brier"])
            <= 0.001
            and int(metrics["seasons_nll_improved"]) >= 3
            and float(metrics["worst_season_nll_delta"]) <= 0.015
        ):
            qualifying.append((candidate.name, float(metrics["next_game_nll"])))
    if not qualifying:
        return None
    # First identify the best aggregate NLL.  The issue's frozen rule treats
    # candidates within 0.002 NLL of that best value as practically tied and
    # prefers the longer half-life (weaker decay) within that band.
    by_name = {candidate.name: candidate for candidate in candidate_grid()}
    best_nll = min(nll for _name, nll in qualifying)
    within_tie = [item for item in qualifying if item[1] - best_nll <= 0.002]
    return max(
        within_tie,
        key=lambda item: by_name[item[0]].half_life_days or float("inf"),
    )[0]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_bool(value: str) -> bool:
    return value.strip().casefold() in {"true", "1", "yes"}


def _parse_start(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _rank_sample(row: dict[str, str], field: str) -> np.ndarray:
    values = np.asarray(json.loads(row[field]), dtype=int)
    population = int(row["home_team_population"] if field.startswith("home") else row["away_team_population"])
    return values[(values >= 1) & (values <= population)]


def load_historical_games(root: Path, seasons: Iterable[int]) -> list[HistoricalGame]:
    wanted = set(seasons)
    rows = _read_csv(root / "data/processed/modeling/historical_modeling_games.csv")
    games = []
    for row in rows:
        if int(row["season"]) not in wanted:
            continue
        home_ranks = _rank_sample(row, "home_rank_observations")
        away_ranks = _rank_sample(row, "away_rank_observations")
        rank_pairs = np.asarray(json.loads(row["rank_pairs"]), dtype=float)
        rank_pairs = rank_pairs[
            (rank_pairs[:, 0] >= 1)
            & (rank_pairs[:, 0] <= int(row["home_team_population"]))
            & (rank_pairs[:, 1] >= 1)
            & (rank_pairs[:, 1] <= int(row["away_team_population"]))
        ]
        if not len(home_ranks) or not len(away_ranks) or not len(rank_pairs):
            continue
        games.append(
            HistoricalGame(
                game_id=str(row["game_id"]),
                season=int(row["season"]),
                start=_parse_start(row["start_date"]),
                week=int(row["week"]),
                home_id=str(row["home_team_id"]),
                away_id=str(row["away_team_id"]),
                home_name="",
                away_name="",
                home_subdivision=row["home_subdivision"],
                away_subdivision=row["away_subdivision"],
                home_population=int(row["home_team_population"]),
                away_population=int(row["away_team_population"]),
                home_points=int(row["home_points"]),
                away_points=int(row["away_points"]),
                margin=float(row["margin"]),
                neutral_site=_parse_bool(row["neutral_site"]),
                home_ranks=home_ranks,
                away_ranks=away_ranks,
                rank_pairs=rank_pairs,
            )
        )
    games.sort(key=lambda item: (item.season, item.start, item.game_id))
    names = {}
    distributions = _read_csv(
        root / "data/processed/modeling/team_season_rank_distributions.csv"
    )
    for row in distributions:
        names[(int(row["season"]), row["team_id"])] = row["team_name"]
    return [
        HistoricalGame(
            **{
                **game.__dict__,
                "home_name": names.get((game.season, game.home_id), game.home_id),
                "away_name": names.get((game.season, game.away_id), game.away_id),
            }
        )
        for game in games
    ]


def _valid_ranks(row: dict[str, str]) -> np.ndarray:
    ranks = np.asarray(json.loads(row["rank_observations"]), dtype=int)
    population = int(row["team_population"])
    return ranks[(ranks >= 1) & (ranks <= population)]


def _maybe_float(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def _context_rows(root: Path) -> list[TeamSeason]:
    outcomes = {
        (int(row["season"]), row["subdivision"], row["team_id"]): row
        for row in _read_csv(
            root / "data/processed/modeling/team_season_rank_distributions.csv"
        )
    }
    feature_rows = _read_csv(root / "data/processed/preseason/team_season_features.csv")
    features = {
        (int(row["season"]), row["subdivision"], row["team_id"]): row
        for row in feature_rows
    }
    result = []
    for (season, subdivision, team_id), target in sorted(outcomes.items()):
        if subdivision != "fbs" or season < 2004:
            continue
        prior = outcomes.get((season - 1, subdivision, team_id))
        target_ranks = _valid_ranks(target)
        if prior is None or not len(target_ranks):
            continue
        prior_ranks = _valid_ranks(prior)
        if not len(prior_ranks):
            continue
        lag1_z = rank_to_z(prior_ranks, int(prior["team_population"]))
        lag_distributions = []
        history = []
        values: dict[str, float | None] = {}
        for lag in (2, 3):
            old = outcomes.get((season - lag, subdivision, team_id))
            old_ranks = _valid_ranks(old) if old else np.asarray([], dtype=int)
            old_z = (
                rank_to_z(old_ranks, int(old["team_population"]))
                if old is not None and len(old_ranks)
                else np.asarray([], dtype=float)
            )
            values[f"lag{lag}_z_mean"] = float(np.mean(old_z)) if len(old_z) else None
            lag_distributions.append(old_z)
        for old_season in range(2002, season):
            old = outcomes.get((old_season, subdivision, team_id))
            old_ranks = _valid_ranks(old) if old else np.asarray([], dtype=int)
            if old is not None and len(old_ranks):
                history.append(rank_to_z(old_ranks, int(old["team_population"])))
        values.update(historical_rank_features(lag1_z, tuple(history)))
        current = features.get((season, subdivision, team_id), {})
        for name in (
            "recruiting_class_rank",
            "recruiting_class_points",
            "talent_composite",
            "returning_pct_ppa",
            "returning_pct_passing_ppa",
            "returning_pct_receiving_ppa",
            "returning_pct_rushing_ppa",
        ):
            values[name] = _maybe_float(current.get(name))
        points = [
            _maybe_float(
                features.get((season - lag, subdivision, team_id), {}).get(
                    "recruiting_class_points"
                )
            )
            for lag in range(4)
        ]
        for years in (2, 3, 4):
            observed = [point for point in points[:years] if point is not None]
            values[f"recruiting_points_{years}y_mean"] = (
                float(np.mean(observed)) if observed else None
            )
        values["recruiting_points_trend"] = (
            points[0] - points[2]
            if points[0] is not None and points[2] is not None
            else None
        )
        coach = current.get("head_coach") or None
        tenure = 0
        if coach:
            for prior_season in range(season, 2001, -1):
                item = features.get((prior_season, subdivision, team_id), {})
                if item.get("head_coach") != coach:
                    break
                tenure += 1
        values["coach_tenure_seasons"] = float(tenure) if tenure else None
        result.append(
            TeamSeason(
                season,
                subdivision,
                team_id,
                current.get("team_name", target["team_name"]),
                int(target["team_population"]),
                lag1_z,
                rank_to_z(target_ranks, int(target["team_population"])),
                target_ranks,
                values,
                tuple(lag_distributions),
            )
        )
    return result


def _uniform(population: int) -> np.ndarray:
    return np.full(population, 1.0 / population)


def _load_context_artifact(root: Path) -> dict[tuple[int, str], PriorInput]:
    path = root / "data/processed/preseason/context/predictions.csv"
    if not path.exists():
        return {}
    result = {}
    for row in _read_csv(path):
        result[(int(row["season"]), row["team_id"])] = PriorInput(
            int(row["season"]),
            row["team_id"],
            row["team_name"],
            row["subdivision"],
            len(json.loads(row["pmf"])),
            np.asarray(json.loads(row["pmf"]), dtype=float),
            "frozen_context_prior_artifact",
        )
    return result


def _load_frozen_context_development_model(root: Path) -> DirectRankModel:
    """Load the repository's already-fitted Context development candidate."""
    report_path = root / "data/processed/preseason/context/model_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = report["selection"]["selected"]
    candidate = report["development"]["candidates"].get(selected)
    if candidate is None:
        raise ValueError(f"Context report has no selected development candidate: {selected}")
    metadata = candidate["fit_metadata"]
    preprocessing = metadata["preprocessing"]
    return DirectRankModel(
        list(metadata["feature_names"]),
        Preprocessor(
            tuple(preprocessing["feature_names"]),
            {key: float(value) for key, value in preprocessing["medians"].items()},
            {key: float(value) for key, value in preprocessing["means"].items()},
            {key: float(value) for key, value in preprocessing["scales"].items()},
        ),
        np.asarray(metadata["location_coefficients"], dtype=float),
        np.asarray(metadata["log_scale_coefficients"], dtype=float),
        float(metadata["minimum_scale"]),
        float(metadata["penalty"]),
        metadata.get("optimizer"),
        int(metadata.get("lag_count", 1)),
        metadata.get("family", "normal"),
        metadata.get("degrees_of_freedom"),
        metadata.get("location_feature_names"),
        metadata.get("scale_feature_names"),
    )


def build_context_priors(
    root: Path, seasons: Iterable[int]
) -> dict[tuple[int, str], PriorInput]:
    """Build missing development Context PMFs without target-season outcomes."""
    wanted = set(seasons)
    result = {
        key: value
        for key, value in _load_context_artifact(root).items()
        if key[0] in wanted
    }
    rows = _context_rows(root)
    missing_seasons = sorted(season for season in wanted if season < 2022)
    if missing_seasons:
        # The repository's Context investigation already freezes a selected
        # development fit through 2017.  Reusing its coefficients is both
        # deterministic and materially cheaper than an unrelated optimizer
        # refit here; applying it to these rows uses no target-season outcome.
        model = _load_frozen_context_development_model(root)
    else:
        model = None
    for season in missing_seasons:
        target = [row for row in rows if row.season == season]
        if not target or model is None:
            continue
        for row in target:
            locations, _scale = model.conditional_parameters(
                row.features, row.lag1_z, row.lag_zs
            )
            result[(season, row.team_id)] = PriorInput(
                season,
                row.team_id,
                row.team_name,
                row.subdivision,
                row.population,
                model.pmf(row.features, row.lag1_z, row.population, row.lag_zs),
                "annual_context_prior_fit_through_previous_season",
            )
            if not np.isfinite(locations).all():
                raise FloatingPointError("non-finite Context prior location")
    return result


def _team_pmf_from_ranks(ranks: np.ndarray, population: int) -> np.ndarray:
    pmf = np.bincount(ranks.astype(int) - 1, minlength=population).astype(float)
    return pmf / pmf.sum()


def _teams_for_season(
    season_games: Sequence[HistoricalGame],
    priors: dict[tuple[int, str], PriorInput],
) -> list[Team]:
    team_info: dict[str, tuple[str, str, int]] = {}
    for game in season_games:
        team_info[game.home_id] = (
            game.home_name,
            game.home_subdivision,
            game.home_population,
        )
        team_info[game.away_id] = (
            game.away_name,
            game.away_subdivision,
            game.away_population,
        )
    teams = []
    for team_id, (name, subdivision, population) in sorted(team_info.items()):
        prior = priors.get((season_games[0].season, team_id))
        if prior is not None:
            pmf = prior.pmf
            population = len(pmf)
        else:
            pmf = _uniform(population)
        teams.append(Team(team_id, name, subdivision, pmf))  # type: ignore[arg-type]
    return teams


def _oriented_expected_margin(
    game: HistoricalGame, likelihood: LikelihoodV1
) -> float:
    pairs = game.rank_pairs
    home_percentile = (pairs[:, 0] - 0.5) / game.home_population
    away_percentile = (pairs[:, 1] - 0.5) / game.away_population
    cross = game.home_subdivision != game.away_subdivision
    if cross and game.home_subdivision == "fcs":
        x, y = away_percentile, home_percentile
    else:
        x, y = home_percentile, away_percentile
    pairing = "fbs-fcs" if cross else f"{game.home_subdivision}-{game.away_subdivision}"
    matrix = design_matrix(
        x,
        y,
        np.full(len(pairs), pairing),
        np.full(len(pairs), float(not game.neutral_site)),
        np.full(len(pairs), float(game.neutral_site)),
        surface=True,
        fbs_home=np.full(
            len(pairs), float(cross and game.home_subdivision == "fbs" and not game.neutral_site)
        ),
    )
    return float(np.mean(matrix @ likelihood.beta))


def residual_rows(
    games: Sequence[HistoricalGame], likelihood: LikelihoodV1
) -> list[dict[str, object]]:
    """Create focal-team residuals with positive meaning better than expected."""
    result = []
    for game in games:
        expected = _oriented_expected_margin(game, likelihood)
        residual = game.margin - expected
        for focal, opponent, sign in (
            (game.home_id, game.away_id, 1.0),
            (game.away_id, game.home_id, -1.0),
        ):
            result.append(
                {
                    "season": game.season,
                    "team_id": focal,
                    "opponent_id": opponent,
                    "game_id": game.game_id,
                    "game_date": game.start.isoformat(),
                    "week": game.week,
                    "game_order": 0,
                    "observed_oriented_margin": sign * game.margin,
                    "expected_oriented_margin": sign * expected,
                    "residual": sign * residual,
                    "home_id": game.home_id,
                    "away_id": game.away_id,
                    "home_name": game.home_name,
                    "away_name": game.away_name,
                    "pairing": f"{game.home_subdivision}-{game.away_subdivision}",
                }
            )
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in result:
        grouped[(int(row["season"]), str(row["team_id"]))].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: (str(row["game_date"]), str(row["game_id"])))
        for index, row in enumerate(values, start=1):
            row["game_order"] = index
    return sorted(result, key=lambda row: (row["season"], row["game_date"], row["game_id"], row["team_id"]))


def _period(season: int) -> str:
    if season in TRAIN_SEASONS:
        return "training"
    if season in DEVELOPMENT_SEASONS:
        return "development"
    if season in EVALUATION_SEASONS:
        return "evaluation"
    return "other"


def _correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    x_values, y_values = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(x_values) == 0 or np.std(y_values) == 0:
        return None
    return float(np.corrcoef(x_values, y_values)[0, 1])


def _metric_row(
    rows: list[dict[str, object]],
    *,
    period: str,
    dimension: str,
    bin_name: str,
    control: str,
    lag: int,
) -> dict[str, object]:
    x = [float(row[0]) for row in rows]
    y = [float(row[1]) for row in rows]
    corr = _correlation(x, y)
    return {
        "period": period,
        "dimension": dimension,
        "bin": bin_name,
        "control": control,
        "lag": lag,
        "n_pairs": len(rows),
        "correlation": corr,
        "mean_product": float(np.mean(np.asarray(x) * np.asarray(y))) if rows else None,
    }


def stage0_persistence(
    residuals: Sequence[dict[str, object]], seed: int = 340
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Measure raw/demeaned lag persistence and deterministic null controls."""
    groups: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in residuals:
        groups[(int(row["season"]), str(row["team_id"]))].append(row)
    lag_pairs: dict[tuple[str, str, str, int], list[tuple[float, float]]] = defaultdict(list)
    elapsed_pairs: dict[tuple[str, str, str, int], list[tuple[float, float]]] = defaultdict(list)
    game_bins: dict[tuple[str, str, str, int], list[tuple[float, float]]] = defaultdict(list)
    early_late: list[dict[str, object]] = []
    predictor_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    rng = np.random.default_rng(seed)
    n_team_seasons = 0
    for (season, _team), values in sorted(groups.items()):
        values.sort(key=lambda row: int(row["game_order"]))
        if len(values) < 3:
            continue
        n_team_seasons += 1
        raw = np.asarray([float(row["residual"]) for row in values])
        demeaned = raw - raw.mean()
        dates = [_parse_start(str(row["game_date"])) for row in values]
        period = _period(season)
        for lag in (1, 2):
            for index in range(len(values) - lag):
                elapsed = (dates[index + lag] - dates[index]).total_seconds() / 86400
                pair = (float(raw[index]), float(raw[index + lag]))
                centered = (float(demeaned[index]), float(demeaned[index + lag]))
                lag_pairs[(period, "raw", "observed", lag)].append(pair)
                lag_pairs[(period, "demeaned", "observed", lag)].append(centered)
                elapsed_bin = (
                    "0-7d" if elapsed <= 7 else
                    "8-21d" if elapsed <= 21 else
                    "22-56d" if elapsed <= 56 else
                    "57-112d" if elapsed <= 112 else "113d+"
                )
                elapsed_pairs[(period, "raw", elapsed_bin, lag)].append(pair)
                elapsed_pairs[(period, "demeaned", elapsed_bin, lag)].append(centered)
                game_bin = "lag1" if lag == 1 else "lag2"
                game_bins[(period, "raw", game_bin, lag)].append(pair)
                game_bins[(period, "demeaned", game_bin, lag)].append(centered)
        split = len(values) // 2
        if split and len(values) - split:
            early_late.append(
                {
                    "period": period,
                    "season": season,
                    "team_id": values[0]["team_id"],
                    "early_mean_residual": float(raw[:split].mean()),
                    "late_mean_residual": float(raw[split:].mean()),
                    "early_late_delta": float(raw[split:].mean() - raw[:split].mean()),
                    "n_games": len(values),
                }
            )
        for index in range(1, len(values)):
            previous = raw[:index]
            predictor_values[period].append((float(previous[-1]), float(raw[index])))
            predictor_values[f"{period}:rolling"].append(
                (float(previous.mean()), float(raw[index]))
            )
            ewma = float(previous[0])
            alpha = 1.0 - 2.0 ** (-1.0 / 3.0)
            for item in previous[1:]:
                ewma = alpha * float(item) + (1 - alpha) * ewma
            predictor_values[f"{period}:ewma"].append((ewma, float(raw[index])))
        for permutation in range(25):
            shuffled = rng.permutation(raw)
            for lag in (1, 2):
                pairs = [
                    (float(shuffled[index]), float(shuffled[index + lag]))
                    for index in range(len(shuffled) - lag)
                ]
                centered_shuffled = shuffled - shuffled.mean()
                centered_pairs = [
                    (
                        float(centered_shuffled[index]),
                        float(centered_shuffled[index + lag]),
                    )
                    for index in range(len(shuffled) - lag)
                ]
                lag_pairs[(period, "raw", "shuffle_order", lag)].extend(pairs)
                lag_pairs[(period, "demeaned", "shuffle_order", lag)].extend(
                    centered_pairs
                )
    # Same-season unrelated-team null: matching sequence positions are paired
    # after a deterministic season-level permutation, preserving residual scale.
    by_season: dict[int, list[np.ndarray]] = defaultdict(list)
    for (season, _team), values in groups.items():
        by_season[season].append(np.asarray([float(row["residual"]) for row in values]))
    for season, sequences in by_season.items():
        if len(sequences) < 2:
            continue
        period = _period(season)
        for _ in range(25):
            shuffled = rng.permutation(len(sequences))
            for left_index, left in enumerate(sequences):
                right = sequences[shuffled[left_index]]
                if len(left) < 3 or len(right) < 3:
                    continue
                for lag in (1, 2):
                    count = min(len(left), len(right)) - lag
                    pairs = [
                        (float(left[i]), float(right[i + lag])) for i in range(count)
                    ]
                    centered_left = left - left.mean()
                    centered_right = right - right.mean()
                    centered_pairs = [
                        (
                            float(centered_left[i]),
                            float(centered_right[i + lag]),
                        )
                        for i in range(count)
                    ]
                    lag_pairs[(period, "raw", "unrelated_team", lag)].extend(pairs)
                    lag_pairs[(period, "demeaned", "unrelated_team", lag)].extend(
                        centered_pairs
                    )
    metrics = []
    for key, pairs in sorted(lag_pairs.items()):
        period, kind, control, lag = key
        metrics.append(
            _metric_row(
                pairs,
                period=period,
                dimension=kind,
                bin_name="all",
                control=control,
                lag=lag,
            )
        )
    for key, pairs in sorted(elapsed_pairs.items()):
        period, kind, elapsed_bin, lag = key
        metrics.append(
            _metric_row(
                pairs,
                period=period,
                dimension=f"{kind}_elapsed_days",
                bin_name=elapsed_bin,
                control="observed",
                lag=lag,
            )
        )
    for key, pairs in sorted(game_bins.items()):
        period, kind, game_bin, lag = key
        metrics.append(
            _metric_row(
                pairs,
                period=period,
                dimension=f"{kind}_game_count",
                bin_name=game_bin,
                control="observed",
                lag=lag,
            )
        )
    predictor_metrics = []
    for name, pairs in sorted(predictor_values.items()):
        x = [pair[0] for pair in pairs]
        y = [pair[1] for pair in pairs]
        predictor_metrics.append(
            {
                "predictor": name,
                "n": len(pairs),
                "correlation": _correlation(x, y),
                "slope": float(np.polyfit(x, y, 1)[0]) if len(pairs) > 1 and np.std(x) else None,
                "mean_next_residual": float(np.mean(y)) if y else None,
            }
        )
    early_late_summary = {}
    for period in sorted({_period(int(row["season"])) for row in residuals}):
        values = [row for row in early_late if row["period"] == period]
        early_late_summary[period] = {
            "n": len(values),
            "correlation": _correlation(
                [float(row["early_mean_residual"]) for row in values],
                [float(row["late_mean_residual"]) for row in values],
            ),
            "mean_late_minus_early": float(
                np.mean([float(row["early_late_delta"]) for row in values])
            )
            if values
            else None,
        }
    summary = {
        "n_focal_team_games": len(residuals),
        "n_team_seasons_with_at_least_3_games": n_team_seasons,
        "predictor_metrics": predictor_metrics,
        "early_late_by_period": early_late_summary,
        "early_late_rows": len(early_late),
        "null_seed": seed,
        "null_permutations_per_group": 25,
    }
    return metrics, early_late, summary


def _student_t_density(value: float, locations: np.ndarray, scale: float, df: float) -> np.ndarray:
    z = (value - locations) / scale
    constant = (
        gammaln((df + 1) / 2)
        - gammaln(df / 2)
        - 0.5 * np.log(df * np.pi)
        - np.log(scale)
    )
    return np.exp(constant - (df + 1) / 2 * np.log1p(z * z / df))


def predictive_scores(
    game: HistoricalGame,
    posterior: dict[str, np.ndarray],
    likelihood: LikelihoodV1,
    surface_cache: dict[tuple[str, int, int], np.ndarray] | None = None,
) -> dict[str, float]:
    teams = {
        game.home_id: Team(
            game.home_id,
            game.home_name,
            game.home_subdivision,  # type: ignore[arg-type]
            posterior[game.home_id],
        ),
        game.away_id: Team(
            game.away_id,
            game.away_name,
            game.away_subdivision,  # type: ignore[arg-type]
            posterior[game.away_id],
        ),
    }
    cache_key = (
        game.game_id,
        len(teams[game.home_id].prior),
        len(teams[game.away_id].prior),
    )
    if surface_cache is not None and cache_key in surface_cache:
        locations = surface_cache[cache_key]
    else:
        locations, _margin = game_margin_parameters(
            game.as_engine_game(), teams[game.home_id], teams[game.away_id], likelihood
        )
        if surface_cache is not None:
            surface_cache[cache_key] = locations
    joint = teams[game.home_id].prior[:, None] * teams[game.away_id].prior[None, :]
    actual = float(game.margin)
    densities = _student_t_density(actual, locations, likelihood.scale, likelihood.degrees_of_freedom)
    predictive_density = float(np.sum(joint * densities))
    expected = float(np.sum(joint * locations))
    win_probability = float(
        np.sum(joint * t.cdf(locations / likelihood.scale, likelihood.degrees_of_freedom))
    )
    return {
        "margin_nll": float(-np.log(max(predictive_density, EPSILON))),
        "margin_mae": abs(expected - actual),
        "win_brier": (win_probability - float(actual > 0)) ** 2,
        "expected_margin": expected,
        "win_probability": win_probability,
        "actual_margin": actual,
    }


def _phase(season_games: Sequence[HistoricalGame], cutoff: datetime) -> str:
    first = min(game.start for game in season_games)
    last = max(game.start for game in season_games)
    span = max((last - first).total_seconds(), 1.0)
    fraction = (cutoff - first).total_seconds() / span
    return "early" if fraction < 1 / 3 else "mid" if fraction < 2 / 3 else "late"


def _cutoffs(season_games: Sequence[HistoricalGame]) -> list[datetime]:
    by_week: dict[int, list[datetime]] = defaultdict(list)
    for game in season_games:
        by_week[game.week].append(game.start)
    weeks = sorted(by_week)
    # Four deterministic, evenly-spaced completed-week cutoffs keep the staged
    # research tractable while still covering early, middle, and late season.
    eligible = weeks[1:-1]
    if len(eligible) > MAX_CUTOFFS_PER_SEASON:
        indexes = np.linspace(
            0, len(eligible) - 1, MAX_CUTOFFS_PER_SEASON, dtype=int
        )
        eligible = [eligible[index] for index in indexes]
    return [max(by_week[week]) + timedelta(microseconds=1) for week in eligible]


def evaluate_candidates(
    games: Sequence[HistoricalGame],
    priors: dict[tuple[int, str], PriorInput],
    likelihood: LikelihoodV1,
    candidates: Sequence[TemporalCandidate],
    seasons: Iterable[int],
    *,
    max_iterations: int = 75,
    tolerance: float = 1e-3,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run identical cutoff/target keys for all requested candidates."""
    season_set = set(seasons)
    by_season: dict[int, list[HistoricalGame]] = defaultdict(list)
    for game in games:
        if game.season in season_set:
            by_season[game.season].append(game)
    prediction_rows: list[dict[str, object]] = []
    example_rows: list[dict[str, object]] = []
    for season, season_games in sorted(by_season.items()):
        print(f"evaluating temporal candidates for {season}", flush=True)
        season_games = sorted(season_games, key=lambda item: (item.start, item.game_id))
        teams = _teams_for_season(season_games, priors)
        teams_by_id = {team.team_id: team for team in teams}
        # Fill the engine's immutable V1 factor cache once per season.  The
        # same matrices are then reused by Static V1 and every recency probe.
        for game in season_games:
            game_factor(
                game.as_engine_game(),
                teams_by_id[game.home_id],
                teams_by_id[game.away_id],
                likelihood,
            )
        by_team = defaultdict(list)
        surface_cache: dict[tuple[str, int, int], np.ndarray] = {}
        for game in season_games:
            by_team[game.home_id].append(game)
            by_team[game.away_id].append(game)
        for cutoff_index, cutoff in enumerate(_cutoffs(season_games), start=1):
            past = [game for game in season_games if game.start < cutoff]
            if not past:
                continue
            next_games = {}
            for team_games in by_team.values():
                after = [game for game in team_games if game.start > cutoff]
                if after:
                    next_games[after[0].game_id] = after[0]
            # The primary next-game panel is the predeclared question.  An
            # all-future panel would score the same games repeatedly at every
            # cutoff and is deliberately omitted because it is not inexpensive
            # on this loopy graph.
            target_sets = {"next_game": list(next_games.values())}
            engine_games = [game.as_engine_game() for game in past]
            for candidate in candidates:
                weights = {
                    game.game_id: recency_weight(cutoff, game.start, candidate.half_life_days)
                    for game in past
                }
                result = infer_posterior(
                    teams,
                    engine_games,
                    likelihood,
                    factor_weights=None
                    if candidate.half_life_days is None
                    else weights,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                )
                if not result.converged:
                    raise RuntimeError(
                        f"non-converged {candidate.name} posterior at {season} cutoff {cutoff.isoformat()}"
                    )
                for target_kind, targets in target_sets.items():
                    for target in targets:
                        if target.home_id not in result.pmfs or target.away_id not in result.pmfs:
                            continue
                        scores = predictive_scores(
                            target, result.pmfs, likelihood, surface_cache
                        )
                        prediction_rows.append(
                            {
                                "season": season,
                                "cutoff": cutoff.isoformat(),
                                "cutoff_index": cutoff_index,
                                "phase": _phase(season_games, cutoff),
                                "candidate": candidate.name,
                                "target_kind": target_kind,
                                "target_game_id": target.game_id,
                                "target_date": target.start.isoformat(),
                                "home_team_id": target.home_id,
                                "home_team_name": target.home_name,
                                "away_team_id": target.away_id,
                                "away_team_name": target.away_name,
                                **scores,
                            }
                        )
                    if target_kind == "next_game" and targets and candidate.name != "Static V1":
                        for target in targets:
                            if target.game_id not in next_games or target.home_id not in result.pmfs:
                                continue
                            scores = predictive_scores(
                                target, result.pmfs, likelihood, surface_cache
                            )
                            static_key = (
                                season,
                                cutoff.isoformat(),
                                target.game_id,
                                "Static V1",
                            )
                            # The pool is materialized after all candidates by a
                            # separate join; this row records the candidate side.
                            example_rows.append(
                                {
                                    "season": season,
                                    "cutoff": cutoff.isoformat(),
                                    "target_game_id": target.game_id,
                                    "candidate": candidate.name,
                                    "home_team_id": target.home_id,
                                    "home_team_name": target.home_name,
                                    "recent_residuals": "",
                                    "candidate_expected_margin": scores["expected_margin"],
                                    "candidate_actual_margin": scores["actual_margin"],
                                    "static_join_key": "|".join(map(str, static_key)),
                                }
                            )
    # A candidate is allowed to change only inference weights, never which
    # cutoff or target is scored.  Keep this as an executable contract so a
    # future refactor cannot silently compare different scoring keys.
    keys_by_candidate: dict[str, set[tuple[object, ...]]] = defaultdict(set)
    for row in prediction_rows:
        keys_by_candidate[str(row["candidate"])].add(
            (
                row["season"],
                row["cutoff"],
                row["target_kind"],
                row["target_game_id"],
            )
        )
    if keys_by_candidate:
        reference = next(iter(keys_by_candidate.values()))
        if any(keys != reference for keys in keys_by_candidate.values()):
            raise RuntimeError("candidate scoring keys differ")
    return prediction_rows, example_rows


def aggregate_metrics(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Aggregate next-game and all-future metrics by candidate and season."""
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["candidate"]), str(row["target_kind"]), str(row["season"]))].append(row)
    result = {}
    for (candidate, target_kind, season), values in sorted(grouped.items()):
        result[f"{candidate}|{target_kind}|{season}"] = {
            "candidate": candidate,
            "target_kind": target_kind,
            "season": int(season),
            "n": len(values),
            "margin_nll": float(np.mean([float(row["margin_nll"]) for row in values])),
            "margin_mae": float(np.mean([float(row["margin_mae"]) for row in values])),
            "win_brier": float(np.mean([float(row["win_brier"]) for row in values])),
        }
    return result


def aggregate_phase_metrics(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Aggregate prediction metrics by candidate and early/mid/late phase."""
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["candidate"]),
                str(row["target_kind"]),
                str(row["phase"]),
            )
        ].append(row)
    result = {}
    for (candidate, target_kind, phase), values in sorted(grouped.items()):
        result[f"{candidate}|{target_kind}|{phase}"] = {
            "candidate": candidate,
            "target_kind": target_kind,
            "phase": phase,
            "n": len(values),
            "margin_nll": float(np.mean([float(row["margin_nll"]) for row in values])),
            "margin_mae": float(np.mean([float(row["margin_mae"]) for row in values])),
            "win_brier": float(np.mean([float(row["win_brier"]) for row in values])),
        }
    return result


def development_gate_metrics(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    aggregate = aggregate_metrics(
        [row for row in rows if row["target_kind"] == "next_game"]
    )
    static_by_season = {
        int(item["season"]): item
        for key, item in aggregate.items()
        if item["candidate"] == "Static V1"
    }
    result = {}
    for candidate in CANDIDATE_NAMES:
        values = [
            item
            for item in aggregate.values()
            if item["candidate"] == candidate
        ]
        if not values:
            continue
        season_deltas = [
            float(item["margin_nll"])
            - float(static_by_season[int(item["season"])] ["margin_nll"])
            for item in values
            if int(item["season"]) in DEVELOPMENT_SEASONS
        ]
        overall = [
            item for item in values if int(item["season"]) in DEVELOPMENT_SEASONS
        ]
        result[candidate] = {
            "next_game_nll": float(np.average([float(item["margin_nll"]) for item in overall], weights=[int(item["n"]) for item in overall])),
            "next_game_mae": float(np.average([float(item["margin_mae"]) for item in overall], weights=[int(item["n"]) for item in overall])),
            "next_game_brier": float(np.average([float(item["win_brier"]) for item in overall], weights=[int(item["n"]) for item in overall])),
            "seasons_nll_improved": sum(delta < 0 for delta in season_deltas),
            "worst_season_nll_delta": max(season_deltas, default=0.0),
            "season_deltas": season_deltas,
        }
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
