"""Frozen synthetic schedule-equivalence study for issue 252.

This module is research-only. It generates observations from the frozen
Historical Likelihood V1 model, then evaluates the existing Quality of Results
posterior without changing the production model or publication artifacts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t as student_t

from gippyrank.methodology import HISTORICAL_LIKELIHOOD_SHA256
from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    Team,
    game_margin_parameters,
)
from gippyrank.posterior.snapshots import load_pinned_likelihood, load_teams
from gippyrank.research.quality_of_results import infer_quality_of_results

MARGIN_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
PILOT_SEED_START = 2_510_000
PILOT_DRAWS = 200
SITE_SLOTS = (
    "home",
    "away",
    "home",
    "away",
    "neutral",
    "home",
    "away",
    "home",
    "away",
    "neutral",
)
SCHEDULE_TARGETS: dict[str, tuple[float, ...]] = {
    "strong_opponents": (0.06, 0.09, 0.12, 0.15, 0.18, 0.22, 0.25, 0.28, 0.31, 0.34),
    "approximately_even": (0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68),
    "weaker_opponents": (0.66, 0.69, 0.72, 0.75, 0.78, 0.82, 0.85, 0.88, 0.91, 0.94),
}
UNCONDITIONAL_SEED_START = 2_520_000
UNCONDITIONAL_DRAWS = 500
CONDITIONED_SEED_START = 2_530_000
CONDITIONED_DRAWS = 200
CONDITIONED_SEARCH_LIMIT = 20_000
SENSITIVITY_SEED_START = 2_540_000
SENSITIVITY_SEED_COUNT = 1_000
EXPECTED_PROBABILITY_TOLERANCE = 0.03
_LOCATION_CACHE: dict[tuple[object, ...], dict[str, np.ndarray]] = {}


@dataclass(frozen=True)
class OpponentSlot:
    """A frozen opponent prior and its assigned venue in one schedule."""

    team: Team
    venue: str
    expected_win_probability: float
    selection_target: float


@dataclass(frozen=True)
class ComponentTrial:
    """One focal team's synthetic schedule and observations."""

    focal: Team
    teams: tuple[Team, ...]
    games: tuple[Game, ...]
    game_rows: tuple[dict[str, object], ...]
    wins: int
    residual_score: float


@dataclass(frozen=True)
class Trial:
    """Matched synthetic season comparison across the four study arms."""

    seed: int
    focal_rank: int
    groups: dict[str, ComponentTrial]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_rank(pmf: np.ndarray, uniform: float) -> int:
    """Draw a one-based rank from a PMF using an explicit uniform variate."""
    return min(
        int(np.searchsorted(np.cumsum(pmf), uniform, side="right")) + 1, len(pmf)
    )


def _site_location_matrices(
    rank_count: int, likelihood: LikelihoodV1
) -> dict[str, np.ndarray]:
    """Build the frozen same-subdivision V1 location surface for each venue."""
    cache_key = (
        rank_count,
        hashlib.sha256(likelihood.beta.tobytes()).hexdigest(),
        likelihood.scale,
        likelihood.degrees_of_freedom,
    )
    if cache_key in _LOCATION_CACHE:
        return _LOCATION_CACHE[cache_key]
    uniform = np.full(rank_count, 1.0 / rank_count)
    focal = Team("surface_focal", "Focal", "fbs", uniform)
    opponent = Team("surface_opponent", "Opponent", "fbs", uniform)
    result: dict[str, np.ndarray] = {}
    for venue in ("home", "away", "neutral"):
        if venue == "away":
            home, away = opponent, focal
        else:
            home, away = focal, opponent
        game = Game(
            f"surface-{venue}",
            home.team_id,
            away.team_id,
            "fbs",
            "fbs",
            0.0,
            0.0,
            venue == "neutral",
        )
        locations, _ = game_margin_parameters(game, home, away, likelihood)
        result[venue] = locations
    _LOCATION_CACHE[cache_key] = result
    return result


def expected_focal_win_probability(
    focal_prior: np.ndarray,
    opponent_prior: np.ndarray,
    venue: str,
    likelihood: LikelihoodV1,
    location_matrices: dict[str, np.ndarray] | None = None,
) -> float:
    """Integrate the V1 Student-t win chance over both discrete rank PMFs."""
    focal_pmf = np.asarray(focal_prior, dtype=float)
    opponent_pmf = np.asarray(opponent_prior, dtype=float)
    if venue not in {"home", "away", "neutral"}:
        raise ValueError(f"unsupported venue: {venue}")
    if focal_pmf.ndim != 1 or opponent_pmf.shape != focal_pmf.shape:
        raise ValueError("focal and opponent PMFs must share a one-dimensional support")
    surfaces = location_matrices or _site_location_matrices(len(focal_pmf), likelihood)
    location = surfaces[venue]
    if venue == "away":
        win_probability = student_t.cdf(
            (0.0 - location) / likelihood.scale, likelihood.degrees_of_freedom
        )
        joint_prior = np.multiply.outer(opponent_pmf, focal_pmf)
    else:
        win_probability = student_t.sf(
            (0.0 - location) / likelihood.scale, likelihood.degrees_of_freedom
        )
        joint_prior = np.multiply.outer(focal_pmf, opponent_pmf)
    value = float(np.sum(joint_prior * win_probability))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise FloatingPointError("integrated V1 win probability is invalid")
    return value


def _select_focal_team(teams: list[Team]) -> Team:
    fbs = [team for team in teams if team.subdivision == "fbs"]
    if not fbs:
        raise ValueError("2026 Context prior has no FBS teams")
    expected = {
        team.team_id: float(np.dot(np.arange(1, len(team.prior) + 1), team.prior))
        for team in fbs
    }
    ordered = sorted(fbs, key=lambda team: (expected[team.team_id], team.team_id))
    return ordered[len(ordered) // 2]


def build_schedules(
    teams: list[Team], likelihood: LikelihoodV1
) -> tuple[Team, dict[str, tuple[OpponentSlot, ...]]]:
    """Select the three preregistered slates from independent 2026 prior PMFs."""
    fbs = [team for team in teams if team.subdivision == "fbs"]
    supports = {len(team.prior) for team in fbs}
    if len(supports) != 1:
        raise ValueError("the 2026 FBS priors do not share one rank support")
    focal = _select_focal_team(fbs)
    surfaces = _site_location_matrices(len(focal.prior), likelihood)
    candidates = [team for team in fbs if team.team_id != focal.team_id]
    probabilities = {
        team.team_id: {
            venue: expected_focal_win_probability(
                focal.prior, team.prior, venue, likelihood, surfaces
            )
            for venue in ("home", "away", "neutral")
        }
        for team in candidates
    }
    average_probability = {
        team.team_id: float(
            np.mean([probabilities[team.team_id][venue] for venue in SITE_SLOTS])
        )
        for team in candidates
    }
    schedules: dict[str, tuple[OpponentSlot, ...]] = {}
    for role, targets in SCHEDULE_TARGETS.items():
        unused = {team.team_id: team for team in candidates}
        selected: list[tuple[Team, float]] = []
        for target in targets:
            team = min(
                unused.values(),
                key=lambda item: (
                    abs(average_probability[item.team_id] - target),
                    item.team_id,
                ),
            )
            selected.append((team, target))
            del unused[team.team_id]
        slots = tuple(
            OpponentSlot(
                team=team,
                venue=venue,
                expected_win_probability=probabilities[team.team_id][venue],
                selection_target=target,
            )
            for (team, target), venue in zip(selected, SITE_SLOTS, strict=True)
        )
        actual_mean = float(np.mean([slot.expected_win_probability for slot in slots]))
        target_mean = float(np.mean(targets))
        if abs(actual_mean - target_mean) > EXPECTED_PROBABILITY_TOLERANCE:
            raise ValueError(
                f"{role} schedule mean {actual_mean:.4f} misses preregistered "
                f"target {target_mean:.4f} by more than "
                f"{EXPECTED_PROBABILITY_TOLERANCE:.2f}"
            )
        schedules[role] = slots
    return focal, schedules


def _component_trial(
    *,
    role: str,
    schedule: tuple[OpponentSlot, ...],
    focal_prior: np.ndarray,
    focal_rank: int,
    opponent_uniforms: np.ndarray,
    margin_shocks: np.ndarray,
    likelihood: LikelihoodV1,
    locations: dict[str, np.ndarray],
    seed: int,
) -> ComponentTrial:
    focal = Team(f"{role}_focal", f"Synthetic {role}", "fbs", focal_prior)
    teams: list[Team] = [focal]
    games: list[Game] = []
    game_rows: list[dict[str, object]] = []
    standardized_residual_squares: list[float] = []
    wins = 0
    for index, (slot, opponent_uniform, shock) in enumerate(
        zip(schedule, opponent_uniforms, margin_shocks, strict=True), start=1
    ):
        opponent = Team(
            f"{role}_opponent_{index:02d}",
            slot.team.name,
            "fbs",
            slot.team.prior,
        )
        teams.append(opponent)
        is_home = slot.venue != "away"
        home, away = (focal, opponent) if is_home else (opponent, focal)
        game_id = f"{seed}-{role}-{index:02d}"
        template = Game(
            game_id,
            home.team_id,
            away.team_id,
            "fbs",
            "fbs",
            0.0,
            0.0,
            slot.venue == "neutral",
        )
        location_surface = locations[slot.venue]
        opponent_rank = _sample_rank(slot.team.prior, float(opponent_uniform))
        if is_home:
            location = float(location_surface[focal_rank - 1, opponent_rank - 1])
        else:
            location = float(location_surface[opponent_rank - 1, focal_rank - 1])
        home_margin = location + likelihood.scale * float(shock)
        game = replace(template, home_points=home_margin, away_points=0.0)
        games.append(game)
        focal_margin = home_margin if is_home else -home_margin
        focal_win = focal_margin > 0.0
        wins += int(focal_win)
        standardized_residual_squares.append(float(shock) ** 2)
        game_rows.append(
            {
                "seed": seed,
                "focal_group": role,
                "game_number": index,
                "focal_team_id": focal.team_id,
                "opponent_source_team_id": slot.team.team_id,
                "opponent_name": slot.team.name,
                "venue": slot.venue,
                "focal_true_rank": focal_rank,
                "opponent_true_rank": opponent_rank,
                "opponent_expected_win_probability": slot.expected_win_probability,
                "v1_expected_home_margin": location,
                "synthetic_home_margin": home_margin,
                "synthetic_focal_margin": focal_margin,
                "focal_won": focal_win,
                "standardized_margin_residual": shock,
            }
        )
    return ComponentTrial(
        focal=focal,
        teams=tuple(teams),
        games=tuple(games),
        game_rows=tuple(game_rows),
        wins=wins,
        residual_score=float(np.median(standardized_residual_squares)),
    )


def generate_trial(
    seed: int,
    focal_prior: np.ndarray,
    schedules: dict[str, tuple[OpponentSlot, ...]],
    likelihood: LikelihoodV1,
    *,
    include_control: bool = True,
    include_strong: bool = False,
) -> Trial:
    """Generate one matched season realization before candidate inference."""
    rng = np.random.default_rng(seed)
    focal_rank = _sample_rank(focal_prior, float(rng.random()))
    opponent_uniforms = rng.random(len(SITE_SLOTS))
    matched_shocks = rng.standard_t(likelihood.degrees_of_freedom, size=len(SITE_SLOTS))
    groups: dict[str, ComponentTrial] = {}
    locations = _site_location_matrices(len(focal_prior), likelihood)
    groups["approximately_even"] = _component_trial(
        role="approximately_even",
        schedule=schedules["approximately_even"],
        focal_prior=focal_prior,
        focal_rank=focal_rank,
        opponent_uniforms=opponent_uniforms,
        margin_shocks=matched_shocks,
        likelihood=likelihood,
        locations=locations,
        seed=seed,
    )
    groups["weaker_opponents"] = _component_trial(
        role="weaker_opponents",
        schedule=schedules["weaker_opponents"],
        focal_prior=focal_prior,
        focal_rank=focal_rank,
        opponent_uniforms=opponent_uniforms,
        margin_shocks=matched_shocks,
        likelihood=likelihood,
        locations=locations,
        seed=seed,
    )
    if include_strong:
        groups["strong_opponents"] = _component_trial(
            role="strong_opponents",
            schedule=schedules["strong_opponents"],
            focal_prior=focal_prior,
            focal_rank=focal_rank,
            opponent_uniforms=opponent_uniforms,
            margin_shocks=matched_shocks,
            likelihood=likelihood,
            locations=locations,
            seed=seed,
        )
    if include_control:
        control_a_shocks = rng.standard_t(
            likelihood.degrees_of_freedom, size=len(SITE_SLOTS)
        )
        control_b_shocks = rng.standard_t(
            likelihood.degrees_of_freedom, size=len(SITE_SLOTS)
        )
        groups["same_schedule_a"] = _component_trial(
            role="same_schedule_a",
            schedule=schedules["approximately_even"],
            focal_prior=focal_prior,
            focal_rank=focal_rank,
            opponent_uniforms=opponent_uniforms,
            margin_shocks=control_a_shocks,
            likelihood=likelihood,
            locations=locations,
            seed=seed,
        )
        groups["same_schedule_b"] = _component_trial(
            role="same_schedule_b",
            schedule=schedules["approximately_even"],
            focal_prior=focal_prior,
            focal_rank=focal_rank,
            opponent_uniforms=opponent_uniforms,
            margin_shocks=control_b_shocks,
            likelihood=likelihood,
            locations=locations,
            seed=seed,
        )
    return Trial(seed, focal_rank, groups)


def _pmf_summary(pmf: np.ndarray) -> dict[str, float | int]:
    ranks = np.arange(1, len(pmf) + 1, dtype=float)
    expected_rank = float(np.dot(ranks, pmf))
    variance = float(np.dot((ranks - expected_rank) ** 2, pmf))
    positive = pmf > 0
    entropy = float(-np.sum(pmf[positive] * np.log2(pmf[positive])))
    low = int(np.searchsorted(np.cumsum(pmf), 0.10, side="left") + 1)
    high = int(np.searchsorted(np.cumsum(pmf), 0.90, side="left") + 1)
    median = int(np.searchsorted(np.cumsum(pmf), 0.50, side="left") + 1)
    return {
        "expected_rank": expected_rank,
        "median_rank": median,
        "rank_sd": math.sqrt(max(variance, 0.0)),
        "entropy_bits": entropy,
        "interval_80_low": low,
        "interval_80_high": high,
        "interval_80_width": high - low + 1,
    }


def _pmf_distances(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    total_variation = float(0.5 * np.sum(np.abs(left - right)))
    wasserstein = float(np.sum(np.abs(np.cumsum(left)[:-1] - np.cumsum(right)[:-1])))
    return total_variation, wasserstein


def _infer_trial(
    trial: Trial,
    likelihood: LikelihoodV1,
    *,
    max_iterations: int = 500,
    tolerance: float = 1e-9,
    damping: float = 0.35,
) -> tuple[
    list[dict[str, object]], list[dict[str, object]], dict[float, dict[str, np.ndarray]]
]:
    teams = [team for group in trial.groups.values() for team in group.teams]
    games = [game for group in trial.groups.values() for game in group.games]
    focus_ids = {name: group.focal.team_id for name, group in trial.groups.items()}
    team_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    per_weight: dict[float, dict[str, np.ndarray]] = {}
    factor_surfaces: dict[tuple[object, ...], dict[str, np.ndarray]] = {}
    for weight in MARGIN_WEIGHTS:
        inference = infer_quality_of_results(
            teams,
            games,
            likelihood,
            weight,
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
            factor_surface_cache=factor_surfaces,
        )
        posterior = inference.blended_context_posterior
        if not posterior.converged:
            raise RuntimeError(f"seed {trial.seed}: λ={weight:g} did not converge")
        pmfs = {name: inference.pmfs[team_id] for name, team_id in focus_ids.items()}
        if any(len(pmf) != len(next(iter(pmfs.values()))) for pmf in pmfs.values()):
            raise ValueError("focal PMFs do not share one rank support")
        per_weight[weight] = pmfs
        summaries = {name: _pmf_summary(pmf) for name, pmf in pmfs.items()}
        for name, summary in summaries.items():
            team_rows.append(
                {
                    "seed": trial.seed,
                    "margin_weight": weight,
                    "focal_group": name,
                    "wins": trial.groups[name].wins,
                    "expected_win_rate": float(
                        trial.groups[name].wins / len(SITE_SLOTS)
                    ),
                    **summary,
                    "converged": posterior.converged,
                    "iterations": posterior.iterations,
                }
            )
        for comparison, left_name, right_name in (
            ("schedule_pair", "approximately_even", "weaker_opponents"),
            ("same_schedule_control", "same_schedule_a", "same_schedule_b"),
        ):
            if left_name not in summaries or right_name not in summaries:
                continue
            left, right = summaries[left_name], summaries[right_name]
            tv, wasserstein = _pmf_distances(pmfs[left_name], pmfs[right_name])
            pair_rows.append(
                {
                    "seed": trial.seed,
                    "margin_weight": weight,
                    "comparison": comparison,
                    "left_group": left_name,
                    "right_group": right_name,
                    "rank_difference": float(
                        right["expected_rank"] - left["expected_rank"]
                    ),
                    "median_rank_difference": int(
                        right["median_rank"] - left["median_rank"]
                    ),
                    "right_outranks_left": bool(
                        right["expected_rank"] < left["expected_rank"]
                    ),
                    "rank_sd_difference": float(right["rank_sd"] - left["rank_sd"]),
                    "entropy_difference_bits": float(
                        right["entropy_bits"] - left["entropy_bits"]
                    ),
                    "interval_80_width_difference": int(
                        right["interval_80_width"] - left["interval_80_width"]
                    ),
                    "pmf_total_variation": tv,
                    "pmf_wasserstein_rank_distance": wasserstein,
                }
            )
    return team_rows, pair_rows, per_weight


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) <= 1:
        return mean, mean, mean
    half_width = (
        1.959963984540054 * float(np.std(array, ddof=1)) / math.sqrt(len(array))
    )
    return mean, mean - half_width, mean + half_width


def _wilson_interval(successes: int, count: int) -> tuple[float, float, float]:
    if count == 0:
        return 0.0, 0.0, 1.0
    z = 1.959963984540054
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    half = (
        z
        * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count))
        / denominator
    )
    return rate, max(0.0, center - half), min(1.0, center + half)


def summarize_pairs(
    pair_rows: list[dict[str, object]], panel: str
) -> list[dict[str, object]]:
    """Aggregate per-trial comparisons with Monte Carlo intervals."""
    output: list[dict[str, object]] = []
    comparisons = sorted({str(row["comparison"]) for row in pair_rows})
    for comparison in comparisons:
        for weight in MARGIN_WEIGHTS:
            rows = [
                row
                for row in pair_rows
                if row["comparison"] == comparison and row["margin_weight"] == weight
            ]
            if not rows:
                continue
            fields = (
                "rank_difference",
                "rank_sd_difference",
                "entropy_difference_bits",
                "interval_80_width_difference",
                "pmf_total_variation",
                "pmf_wasserstein_rank_distance",
            )
            summary: dict[str, object] = {
                "panel": panel,
                "comparison": comparison,
                "margin_weight": weight,
                "draw_count": len(rows),
            }
            for field in fields:
                values = [float(row[field]) for row in rows]
                mean, low, high = _mean_ci(values)
                prefix = {
                    "rank_difference": "rank",
                    "rank_sd_difference": "rank_sd",
                    "entropy_difference_bits": "entropy_bits",
                    "interval_80_width_difference": "interval_80_width",
                    "pmf_total_variation": "pmf_total_variation",
                    "pmf_wasserstein_rank_distance": "pmf_wasserstein",
                }[field]
                summary[f"{prefix}_mean"] = mean
                summary[f"{prefix}_mean_ci95_low"] = low
                summary[f"{prefix}_mean_ci95_high"] = high
            differences = np.asarray([float(row["rank_difference"]) for row in rows])
            summary.update(
                {
                    "rank_difference_median": float(np.median(differences)),
                    "rank_difference_p05": float(np.quantile(differences, 0.05)),
                    "rank_difference_p25": float(np.quantile(differences, 0.25)),
                    "rank_difference_p75": float(np.quantile(differences, 0.75)),
                    "rank_difference_p95": float(np.quantile(differences, 0.95)),
                }
            )
            successes = sum(bool(row["right_outranks_left"]) for row in rows)
            rate, low, high = _wilson_interval(successes, len(rows))
            summary["right_outranks_left_probability"] = rate
            summary["right_outranks_left_ci95_low"] = low
            summary["right_outranks_left_ci95_high"] = high
            output.append(summary)
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_gzip_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw_handle,
        gzip.GzipFile(
            filename="", fileobj=raw_handle, mode="wb", compresslevel=9, mtime=0
        ) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle,
    ):
        fields = list(rows[0]) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_prior_profiles(
    path: Path,
    focal: Team,
    schedules: dict[str, tuple[OpponentSlot, ...]],
) -> None:
    unique = {focal.team_id: focal}
    for slots in schedules.values():
        for slot in slots:
            unique[slot.team.team_id] = slot.team
    rows = [
        {
            "team_id": team.team_id,
            "team_name": team.name,
            "rank": rank,
            "probability": float(probability),
        }
        for team in sorted(unique.values(), key=lambda item: item.team_id)
        for rank, probability in enumerate(team.prior, start=1)
    ]
    _write_gzip_csv(path, rows)


def _schedule_rows(
    schedules: dict[str, tuple[OpponentSlot, ...]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role, slots in schedules.items():
        for index, slot in enumerate(slots, start=1):
            rows.append(
                {
                    "schedule_role": role,
                    "game_number": index,
                    "opponent_team_id": slot.team.team_id,
                    "opponent_name": slot.team.name,
                    "venue": slot.venue,
                    "selection_target_win_probability": slot.selection_target,
                    "v1_expected_focal_win_probability": slot.expected_win_probability,
                }
            )
    return rows


def _representative_index(trials: list[Trial], group: str | None = None) -> int:
    scores = np.asarray(
        [
            trial.groups[group].residual_score
            if group is not None
            else float(
                np.median(
                    [
                        trial.groups["approximately_even"].residual_score,
                        trial.groups["weaker_opponents"].residual_score,
                    ]
                )
            )
            for trial in trials
        ],
        dtype=float,
    )
    median = float(np.median(scores))
    return min(
        range(len(trials)),
        key=lambda index: (abs(float(scores[index]) - median), trials[index].seed),
    )


def _component_from_trial(trial: Trial, group: str) -> ComponentTrial:
    return trial.groups[group]


def _set_focal_margin(game: Game, focal_is_home: bool, focal_margin: float) -> Game:
    home_margin = focal_margin if focal_is_home else -focal_margin
    return replace(game, home_points=home_margin, away_points=0.0)


def _infer_component(
    component: ComponentTrial,
    games: tuple[Game, ...],
    likelihood: LikelihoodV1,
) -> dict[float, tuple[dict[str, float | int], np.ndarray]]:
    output: dict[float, tuple[dict[str, float | int], np.ndarray]] = {}
    cache: dict[tuple[object, ...], dict[str, np.ndarray]] = {}
    for weight in MARGIN_WEIGHTS:
        inference = infer_quality_of_results(
            list(component.teams),
            list(games),
            likelihood,
            weight,
            factor_surface_cache=cache,
        )
        result = inference.blended_context_posterior
        if not result.converged:
            raise RuntimeError(f"counterfactual λ={weight:g} did not converge")
        pmf = inference.pmfs[component.focal.team_id]
        output[weight] = (_pmf_summary(pmf), pmf)
    return output


def run_outcome_sensitivity(
    focal_prior: np.ndarray,
    schedules: dict[str, tuple[OpponentSlot, ...]],
    likelihood: LikelihoodV1,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, Trial]]:
    """Run frozen one-game result and margin counterfactuals."""
    trials = [
        generate_trial(
            seed,
            focal_prior,
            schedules,
            likelihood,
            include_control=False,
            include_strong=True,
        )
        for seed in range(
            SENSITIVITY_SEED_START, SENSITIVITY_SEED_START + SENSITIVITY_SEED_COUNT
        )
    ]
    selected: dict[str, Trial] = {}
    for role in SCHEDULE_TARGETS:
        index = _representative_index(trials, role)
        selected[role] = trials[index]

    case_pairs: list[tuple[str, str, float, float, tuple[str, ...]]] = [
        ("win_loss_boundary", "narrow_loss_vs_win", -3.0, 3.0, tuple(SCHEDULE_TARGETS)),
        (
            "within_win_margin",
            "ordinary_vs_decisive_win",
            10.0,
            35.0,
            tuple(SCHEDULE_TARGETS),
        ),
        (
            "within_loss_margin",
            "ordinary_vs_decisive_loss",
            -10.0,
            -35.0,
            tuple(SCHEDULE_TARGETS),
        ),
        (
            "weak_opponent_large_win",
            "ordinary_vs_decisive_win",
            14.0,
            35.0,
            ("weaker_opponents",),
        ),
    ]
    contrast_rows: list[dict[str, object]] = []
    pmf_rows: list[dict[str, object]] = []
    for role in SCHEDULE_TARGETS:
        trial = selected[role]
        component = _component_from_trial(trial, role)
        probs = [slot.expected_win_probability for slot in schedules[role]]
        target_probability = {
            "strong_opponents": 0.20,
            "approximately_even": 0.50,
            "weaker_opponents": 0.80,
        }[role]
        game_index = min(
            range(len(probs)),
            key=lambda index: (abs(probs[index] - target_probability), index),
        )
        original = component.games[game_index]
        focal_is_home = original.home_id == component.focal.team_id
        for panel, pair_name, margin_a, margin_b, roles in case_pairs:
            if role not in roles:
                continue
            game_a = _set_focal_margin(original, focal_is_home, margin_a)
            game_b = _set_focal_margin(original, focal_is_home, margin_b)
            games_a = tuple(
                game_a if index == game_index else game
                for index, game in enumerate(component.games)
            )
            games_b = tuple(
                game_b if index == game_index else game
                for index, game in enumerate(component.games)
            )
            result_a = _infer_component(component, games_a, likelihood)
            result_b = _infer_component(component, games_b, likelihood)
            for weight in MARGIN_WEIGHTS:
                summary_a, pmf_a = result_a[weight]
                summary_b, pmf_b = result_b[weight]
                tv, wasserstein = _pmf_distances(pmf_a, pmf_b)
                benefit = float(summary_a["expected_rank"] - summary_b["expected_rank"])
                contrast_rows.append(
                    {
                        "schedule_role": role,
                        "target_game_number": game_index + 1,
                        "opponent_team_id": schedules[role][game_index].team.team_id,
                        "opponent_expected_win_probability": probs[game_index],
                        "counterfactual_family": panel,
                        "comparison": pair_name,
                        "focal_margin_a": margin_a,
                        "focal_margin_b": margin_b,
                        "margin_weight": weight,
                        "expected_rank_a": summary_a["expected_rank"],
                        "expected_rank_b": summary_b["expected_rank"],
                        "rank_benefit_of_b_over_a": benefit,
                        "posterior_total_variation": tv,
                        "posterior_wasserstein_rank_distance": wasserstein,
                        "rank_sd_change_b_minus_a": summary_b["rank_sd"]
                        - summary_a["rank_sd"],
                        "entropy_change_bits_b_minus_a": summary_b["entropy_bits"]
                        - summary_a["entropy_bits"],
                        "interval_80_width_change_b_minus_a": summary_b[
                            "interval_80_width"
                        ]
                        - summary_a["interval_80_width"],
                    }
                )
                for outcome, summary, pmf in (
                    ("a", summary_a, pmf_a),
                    ("b", summary_b, pmf_b),
                ):
                    for rank, probability in enumerate(pmf, start=1):
                        pmf_rows.append(
                            {
                                "schedule_role": role,
                                "counterfactual_family": panel,
                                "comparison": pair_name,
                                "outcome": outcome,
                                "focal_margin": margin_a
                                if outcome == "a"
                                else margin_b,
                                "margin_weight": weight,
                                "rank": rank,
                                "probability": float(probability),
                                "expected_rank": summary["expected_rank"],
                                "rank_sd": summary["rank_sd"],
                                "entropy_bits": summary["entropy_bits"],
                            }
                        )
    return contrast_rows, pmf_rows, selected


def _markdown_table(rows: list[dict[str, object]], fields: tuple[str, ...]) -> str:
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    return "\n".join(lines)


def _render_report(
    summaries: list[dict[str, object]],
    schedule_probabilities: dict[str, tuple[float, ...]],
    conditioned_accepted: int,
    conditioned_attempts: int,
    representative_rows: list[dict[str, object]],
    sensitivity: list[dict[str, object]],
) -> str:
    primary = [row for row in summaries if row["comparison"] == "schedule_pair"]
    control = [row for row in summaries if row["comparison"] == "same_schedule_control"]
    lines = [
        "# Issue 252: Schedule equivalence and outcome sensitivity",
        "",
        "## Scope and frozen method",
        "",
        "This is research only. It evaluates the issue 242 Quality of Results inference at λ = 0, 0.25, 0.50, 0.75, and 1; it does not change or publish a ranking. Synthetic margins are drawn from the byte-pinned Historical Likelihood V1 Student-t model with site effects. Focal and opponent beliefs are discrete 2026 Context 1.4 rank PMFs, and the issue 242 inference strips each focal Context prior after posterior inference. The shared focal prior is Colorado's preseason PMF (expected rank 63.14) from the complete 138-team FBS support.",
        "",
        f"The unconditional panel has 500 paired seasons. The record-conditioned panel has {conditioned_accepted} exact 5–5 / 8–2 pairs, found in {conditioned_attempts} independent seed attempts. Acceptance is based only on the approximately-even team's 5–5 record and the weaker-opponent team's 8–2 record; read this panel separately from the unconditional panel. Same-schedule control realizations saved at these accepted seeds are not selected on their own records.",
        "",
        "The schedule-control conclusion applies to the median-prior focal strength in the 2026 FBS population. Additional focal-strength strata were not run.",
        "",
        "## Computed schedules",
        "",
        "Each slate has 10 games with 4 home, 4 away, and 2 neutral sites. Opponent probabilities integrate the frozen V1 rank-pair margin distribution over the complete opponent and focal prior PMFs.",
        "",
        _markdown_table(
            [
                {
                    "role": role,
                    "target": f"{np.mean(SCHEDULE_TARGETS[role]):.3f}",
                    "computed": f"{np.mean(probabilities):.3f}",
                    "range": f"{min(probabilities):.3f}–{max(probabilities):.3f}",
                }
                for role, probabilities in schedule_probabilities.items()
            ],
            ("role", "target", "computed", "range"),
        ),
        "",
        "The strong-opponent slate is used for outcome sensitivity only; the primary schedule-equivalence comparison is approximately-even versus weaker opponents.",
        "",
        "## Representative season comparisons",
        "",
        "One comparison per panel is selected by distance to the median generated-model residual score. The record-conditioned panel therefore shows the motivating 5–5 / 8–2 pair. No inferred rank or candidate output was used to select either example.",
        "",
        _markdown_table(
            [
                {
                    "panel": row["panel"],
                    "λ": row["margin_weight"],
                    "even record": f"{row['left_wins']}-{10 - int(row['left_wins'])}",
                    "weaker record": f"{row['right_wins']}-{10 - int(row['right_wins'])}",
                    "even E[rank]": f"{float(row['left_expected_rank']):.1f}",
                    "weaker E[rank]": f"{float(row['right_expected_rank']):.1f}",
                    "Δ rank": f"{float(row['rank_difference']):+.1f}",
                    "rank SD Δ": f"{float(row['right_rank_sd']) - float(row['left_rank_sd']):+.1f}",
                    "entropy Δ": f"{float(row['right_entropy_bits']) - float(row['left_entropy_bits']):+.2f}",
                    "PMF TV": f"{float(row['pmf_total_variation']):.3f}",
                }
                for row in representative_rows
                if row["comparison"] == "schedule_pair"
            ],
            (
                "panel",
                "λ",
                "even record",
                "weaker record",
                "even E[rank]",
                "weaker E[rank]",
                "Δ rank",
                "rank SD Δ",
                "entropy Δ",
                "PMF TV",
            ),
        ),
        "",
        "## Unconditional schedule equivalence",
        "",
        "The rank difference is weaker-schedule expected rank minus approximately-even expected rank; a negative value favors the weaker-schedule team. Intervals are 95% Monte Carlo intervals for means. The outrank probability uses a 95% Wilson interval. TV and Wasserstein measure paired posterior PMF differences.",
        "",
        _markdown_table(
            [
                {
                    "λ": row["margin_weight"],
                    "mean Δ rank": f"{float(row['rank_mean']):.2f} [{float(row['rank_mean_ci95_low']):.2f}, {float(row['rank_mean_ci95_high']):.2f}]",
                    "median": f"{float(row['rank_difference_median']):.2f}",
                    "5–95%": f"{float(row['rank_difference_p05']):.1f} to {float(row['rank_difference_p95']):.1f}",
                    "P(weaker outranks)": f"{float(row['right_outranks_left_probability']):.3f} [{float(row['right_outranks_left_ci95_low']):.3f}, {float(row['right_outranks_left_ci95_high']):.3f}]",
                    "mean SD Δ": f"{float(row['rank_sd_mean']):+.2f}",
                    "mean entropy Δ": f"{float(row['entropy_bits_mean']):+.2f}",
                    "mean 80% width Δ": f"{float(row['interval_80_width_mean']):+.2f}",
                    "mean TV": f"{float(row['pmf_total_variation_mean']):.3f}",
                    "n": row["draw_count"],
                }
                for row in primary
                if row["panel"] == "unconditional"
            ],
            (
                "λ",
                "mean Δ rank",
                "median",
                "5–95%",
                "P(weaker outranks)",
                "mean SD Δ",
                "mean entropy Δ",
                "mean 80% width Δ",
                "mean TV",
                "n",
            ),
        ),
        "",
        "The λ=0.25 mean advantage is about one expected-rank place and its interval spans zero; that is uncertainty about the mean, not evidence that schedules are equivalent. At λ=0.50, 0.75, and 1.0, the mean advantages are smaller than one place, but their intervals exclude zero. These intervals distinguish the estimated mean from zero under this Monte Carlo analysis; they do not measure whether a difference matters for ranking decisions. Practical importance depends on the season-accomplishment tradeoff and cannot be decided from statistical distinguishability alone. No post-hoc equivalence threshold is applied.",
        "",
        "The increasing probability that the weaker-schedule team outranks the approximately-even-schedule team (0.508, 0.544, and 0.576 at λ=0.50, 0.75, and 1.0) is not contradictory to its shrinking mean advantage. The mean reflects both how often and by how many places one team outranks the other; the probability records only which team has the better expected rank in each pair. The 5th–95th percentile spread also remains much wider than the mean shifts.",
        "",
        "## Record-conditioned schedule comparison",
        "",
        "| λ | mean Δ rank (95% MC interval) | median | 5–95% | P(weaker outranks; 95% Wilson interval) | mean rank SD Δ | mean entropy Δ | mean 80% width Δ |",
        "| ---: | --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for weight in MARGIN_WEIGHTS:
        record = next(
            (
                row
                for row in primary
                if row["panel"] == "record_conditioned"
                and row["margin_weight"] == weight
            ),
            None,
        )
        if record is None:
            continue
        lines.append(
            f"| {weight:g} | {float(record['rank_mean']):.2f} [{float(record['rank_mean_ci95_low']):.2f}, {float(record['rank_mean_ci95_high']):.2f}] | {float(record['rank_difference_median']):.2f} | {float(record['rank_difference_p05']):.1f} to {float(record['rank_difference_p95']):.1f} | {float(record['right_outranks_left_probability']):.3f} [{float(record['right_outranks_left_ci95_low']):.3f}, {float(record['right_outranks_left_ci95_high']):.3f}] | {float(record['rank_sd_mean']):+.2f} | {float(record['entropy_bits_mean']):+.2f} | {float(record['interval_80_width_mean']):+.2f} |"
        )
    lines.extend(
        [
            "",
            "The same-schedule control estimates outcome-driven variation when opponent priors, venue slots, and latent ranks are shared. Its unconditional distribution is a reference for interpreting schedule-pair variation; it is not subtracted from the schedule comparison. The artifact also retains control rows labeled `record_conditioned`. Their seeds were selected because the primary approximately-even and weaker-opponent schedules produced 5–5 and 8–2 records, respectively; the two control outcomes were not filtered on their own records. Thus those control rows are conditioned on the primary pair's seed selection, not on control records, and their variability should not be directly compared with the record-conditioned primary comparison as if both used the same outcome selection.",
            "",
            "### Same-schedule control",
            "",
            "| λ | mean Δ rank (95% MC interval) | 5–95% | P(right team outranks; 95% Wilson interval) | mean PMF TV |",
            "| ---: | --- | --- | --- | ---: |",
            *[
                f"| {float(row['margin_weight']):g} | {float(row['rank_mean']):.2f} [{float(row['rank_mean_ci95_low']):.2f}, {float(row['rank_mean_ci95_high']):.2f}] | {float(row['rank_difference_p05']):.1f} to {float(row['rank_difference_p95']):.1f} | {float(row['right_outranks_left_probability']):.3f} [{float(row['right_outranks_left_ci95_low']):.3f}, {float(row['right_outranks_left_ci95_high']):.3f}] | {float(row['pmf_total_variation_mean']):.3f} |"
                for row in control
                if row["panel"] == "unconditional"
            ],
            "",
            "## Outcome sensitivity",
            "",
            "The contrast reports rank benefit of outcome B over A as expected rank A minus expected rank B, so positive values mean B receives a better expected rank. Each row reuses the same opponents, venue, and other nine game results. A zero effect at λ=0 for same-winner margin changes is expected from the binary endpoint and serves as a consistency check. All five weights are shown.",
            "",
            "Within each schedule, each contrast changes one selected game's margin while holding its other nine games fixed. Comparisons across strong, approximately-even, and weaker-opponent schedules also change the surrounding nine-game context. Differences in sensitivity across those schedules therefore cannot be attributed exclusively to the strength of the selected opponent.",
            "",
            _markdown_table(
                [
                    {
                        "schedule": row["schedule_role"],
                        "case": f"{row['counterfactual_family']}: {row['comparison']}",
                        "λ": row["margin_weight"],
                        "focal p(win)": f"{float(row['opponent_expected_win_probability']):.2f}",
                        "rank benefit": f"{float(row['rank_benefit_of_b_over_a']):.2f}",
                        "rank SD Δ": f"{float(row['rank_sd_change_b_minus_a']):+.2f}",
                        "entropy Δ": f"{float(row['entropy_change_bits_b_minus_a']):+.2f}",
                        "PMF TV": f"{float(row['posterior_total_variation']):.3f}",
                    }
                    for row in sensitivity
                    if row["comparison"]
                    in {
                        "narrow_loss_vs_win",
                        "ordinary_vs_decisive_win",
                        "ordinary_vs_decisive_loss",
                    }
                ],
                (
                    "schedule",
                    "case",
                    "λ",
                    "focal p(win)",
                    "rank benefit",
                    "rank SD Δ",
                    "entropy Δ",
                    "PMF TV",
                ),
            ),
            "",
            "These fixed counterfactuals characterize the candidates' relative response to a winner change and to margins within the same outcome. They do not establish a universal exchange rate between a win and additional points.",
            "",
            "## Interpretation and next step",
            "",
            "The unconditional λ=0.25 estimate is a 1.03-place mean advantage for the weaker-schedule team (95% Monte Carlo interval −2.09 to +0.03; outrank probability 0.500). The interval includes zero, so the estimated mean is uncertain; that is not evidence of equivalence. At λ=0.50, 0.75, and 1.0, the smaller mean advantages (0.78, 0.61, and 0.54 places) have intervals excluding zero. Those statistically distinguishable mean shifts are small in expected-rank units, but practical importance depends on the ranking's intended season-accomplishment tradeoff. The probability the weaker-schedule team outranks rises from 0.508 at λ=0.50 to 0.544 at λ=0.75 and 0.576 at λ=1.0; this can coexist with smaller mean advantages because the probability ignores the size of each pair's rank difference. In the exact-record-conditioned primary panel, the weaker-minus-even mean difference is +0.30 places (95% interval +0.09 to +0.52; weaker team outranks in 45.0% of pairs), a small mean favoring the approximately-even schedule. The weaker schedule also has greater mean rank SD (+2.07), entropy (+0.13 bits), and 80% interval width (+5.87 places) in this panel. The paired differences remain broad at every weight.",
            "",
            "At λ=0.25, changing a narrow loss to a narrow win improves expected rank by 8.40 places against the approximately-even opponent and 5.44 against the weaker opponent; changing an ordinary +10 win to +35 improves rank by 1.69 and 1.73 places in those same cases. The weak-opponent +14 to +35 comparison improves rank by 1.38. By λ=0.50, decisive-margin changes approach the win/loss change for a weaker opponent. These counterfactuals make λ=0.25 the clearest existing candidate in this experiment for preserving the win/loss distinction while retaining margin evidence.",
            "",
            "**Recommendation:** retain λ=0.25 as a promising candidate for a preregistered follow-up with additional focal-strength strata. This recommendation reflects the combined pattern: a small, uncertain unconditional schedule mean shift alongside broad paired variation, a slightly stronger-schedule advantage in the exact-record-conditioned primary comparison, and a clearer winner-versus-margin distinction in the outcome-sensitivity contrasts. It does not follow simply from the λ=0.25 interval spanning zero and is not a claim of schedule equivalence. Review it against the intended season-accomplishment tradeoff before any separate model or publication proposal; this study supports further evaluation, not activation.",
            "",
            "## Reproducibility",
            "",
            "`manifest.json` records source hashes, candidate weights, prior population, seeds, sample counts, generator and inference settings, and artifact hashes. `schedule_definitions.csv`, `prior_profiles.csv.gz`, and `historical_likelihood_v1.json` retain the frozen model inputs. `simulated_games.csv.gz` preserves unconditional and accepted conditional observations. `representative_comparison.csv` and `representative_pmfs.csv.gz` retain selected season comparisons; `counterfactual_pmfs.csv.gz` retains the outcome-sensitivity distributions. Per-trial rank and uncertainty summaries allow the aggregate tables to be reconstructed.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_report_csv(path: Path) -> list[dict[str, object]]:
    """Read a saved report input CSV while restoring its numeric values."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        result: list[dict[str, object]] = []
        for row in rows:
            converted: dict[str, object] = {}
            for key, value in row.items():
                if value is None or value == "":
                    converted[key] = value
                    continue
                try:
                    if any(marker in value.lower() for marker in (".", "e")):
                        numeric: float | int = float(value)
                    else:
                        numeric = int(value)
                except ValueError:
                    converted[key] = value
                else:
                    converted[key] = numeric
            result.append(converted)
    return result


def refresh_report_from_artifacts(
    root: Path, output_dir: Path | None = None
) -> dict[str, object]:
    """Re-render the report from saved summaries without rerunning inference."""
    root = root.resolve()
    output_dir = output_dir or root / "data/research/issue_252_quality_of_results"
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_hashes = manifest["artifact_sha256"]
    for name, expected_hash in artifact_hashes.items():
        artifact_path = output_dir / name
        if sha256_file(artifact_path) != expected_hash:
            raise ValueError(f"saved artifact hash does not match manifest: {name}")

    schedule_probabilities: dict[str, list[float]] = {}
    for row in _read_report_csv(output_dir / "schedule_definitions.csv"):
        role = str(row["schedule_role"])
        schedule_probabilities.setdefault(role, []).append(
            float(row["v1_expected_focal_win_probability"])
        )
    summaries = _read_report_csv(output_dir / "simulation_summary.csv")
    representative_rows = _read_report_csv(output_dir / "representative_comparison.csv")
    sensitivity = _read_report_csv(output_dir / "outcome_sensitivity.csv")
    simulation = manifest["simulation"]
    report = _render_report(
        summaries,
        {role: tuple(values) for role, values in schedule_probabilities.items()},
        int(simulation["conditioned_accepted_draw_count"]),
        int(simulation["conditioned_attempt_count"]),
        representative_rows,
        sensitivity,
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    source_path = root / "src/gippyrank/research/issue_252_quality_of_results.py"
    source_hash = sha256_file(source_path)
    simulation_provenance = manifest.get(
        "simulation_provenance",
        {
            "source_revision": manifest["source_revision"],
            "source_code_sha256": manifest["source_code_sha256"],
        },
    )
    try:
        source_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        source_revision = "unavailable"
    manifest["simulation_provenance"] = simulation_provenance
    manifest["source_revision"] = source_revision
    manifest["source_code_sha256"] = source_hash
    manifest["report_generation"] = {
        "method": "deterministic_render_from_saved_artifacts",
        "simulation_rerun": False,
        "source_revision": source_revision,
        "source_code_sha256": source_hash,
        "input_artifacts": [
            "schedule_definitions.csv",
            "simulation_summary.csv",
            "representative_comparison.csv",
            "outcome_sensitivity.csv",
        ],
    }
    artifact_hashes["report.md"] = sha256_file(report_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def run_experiment(root: Path, output_dir: Path | None = None) -> dict[str, object]:
    """Generate the frozen issue 252 research artifacts."""
    root = root.resolve()
    output_dir = output_dir or root / "data/research/issue_252_quality_of_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    teams, team_rows, prior_path = load_teams(root, 2026, "context", "1.4")
    fbs = [team for team in teams if team.subdivision == "fbs"]
    if not fbs:
        raise ValueError("the 2026 Context prior has no FBS population")
    likelihood_path = root / "data/processed/posterior/historical_likelihood_v1.json"
    likelihood = load_pinned_likelihood(likelihood_path)
    focal, schedules = build_schedules(teams, likelihood)
    if sha256_file(likelihood_path) != HISTORICAL_LIKELIHOOD_SHA256:
        raise ValueError("Historical Likelihood V1 does not match the pinned bytes")

    prior_copy = output_dir / "prior_profiles.csv.gz"
    _write_prior_profiles(prior_copy, focal, schedules)
    likelihood_copy = output_dir / "historical_likelihood_v1.json"
    likelihood_copy.write_bytes(likelihood_path.read_bytes())
    schedule_rows = _schedule_rows(schedules)
    _write_csv(output_dir / "schedule_definitions.csv", schedule_rows)

    unconditional_trials = [
        generate_trial(seed, focal.prior, schedules, likelihood)
        for seed in range(
            UNCONDITIONAL_SEED_START, UNCONDITIONAL_SEED_START + UNCONDITIONAL_DRAWS
        )
    ]
    conditioned_trials: list[Trial] = []
    conditioned_attempts = 0
    for offset in range(CONDITIONED_SEARCH_LIMIT):
        seed = CONDITIONED_SEED_START + offset
        trial = generate_trial(seed, focal.prior, schedules, likelihood)
        conditioned_attempts += 1
        if (
            trial.groups["approximately_even"].wins == 5
            and trial.groups["weaker_opponents"].wins == 8
        ):
            conditioned_trials.append(trial)
            if len(conditioned_trials) == CONDITIONED_DRAWS:
                break
    if len(conditioned_trials) != CONDITIONED_DRAWS:
        raise RuntimeError(
            f"only {len(conditioned_trials)} of {CONDITIONED_DRAWS} conditioned pairs "
            f"found within {CONDITIONED_SEARCH_LIMIT} seeds"
        )

    team_summary_rows: list[dict[str, object]] = []
    pair_trial_rows: dict[str, list[dict[str, object]]] = {
        "unconditional": [],
        "record_conditioned": [],
    }
    game_artifact_rows: list[dict[str, object]] = []
    representative_pmfs: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []
    representative_seeds: dict[str, int] = {}
    for panel, trials in (
        ("unconditional", unconditional_trials),
        ("record_conditioned", conditioned_trials),
    ):
        representative = trials[_representative_index(trials)]
        representative_seeds[panel] = representative.seed
        for draw_index, trial in enumerate(trials, start=1):
            team_rows_for_trial, pair_rows_for_trial, per_weight = _infer_trial(
                trial, likelihood
            )
            for row in team_rows_for_trial:
                row["panel"] = panel
            for row in pair_rows_for_trial:
                row["panel"] = panel
            team_summary_rows.extend(team_rows_for_trial)
            pair_trial_rows[panel].extend(pair_rows_for_trial)
            for group in trial.groups.values():
                for row in group.game_rows:
                    game_artifact_rows.append({"panel": panel, **row})
            if trial.seed == representative.seed:
                for weight, group_pmfs in per_weight.items():
                    for group, pmf in group_pmfs.items():
                        for rank, probability in enumerate(pmf, start=1):
                            representative_pmfs.append(
                                {
                                    "panel": panel,
                                    "seed": trial.seed,
                                    "focal_group": group,
                                    "margin_weight": weight,
                                    "rank": rank,
                                    "probability": float(probability),
                                }
                            )
                    for comparison, left_name, right_name in (
                        ("schedule_pair", "approximately_even", "weaker_opponents"),
                        ("same_schedule_control", "same_schedule_a", "same_schedule_b"),
                    ):
                        left_pmf = group_pmfs.get(left_name)
                        right_pmf = group_pmfs.get(right_name)
                        if left_pmf is None or right_pmf is None:
                            continue
                        left_summary = _pmf_summary(left_pmf)
                        right_summary = _pmf_summary(right_pmf)
                        tv, wasserstein = _pmf_distances(left_pmf, right_pmf)
                        representative_rows.append(
                            {
                                "panel": panel,
                                "seed": trial.seed,
                                "comparison": comparison,
                                "margin_weight": weight,
                                "left_group": left_name,
                                "right_group": right_name,
                                "left_wins": trial.groups[left_name].wins,
                                "right_wins": trial.groups[right_name].wins,
                                "left_expected_rank": left_summary["expected_rank"],
                                "right_expected_rank": right_summary["expected_rank"],
                                "rank_difference": right_summary["expected_rank"]
                                - left_summary["expected_rank"],
                                "left_rank_sd": left_summary["rank_sd"],
                                "right_rank_sd": right_summary["rank_sd"],
                                "left_entropy_bits": left_summary["entropy_bits"],
                                "right_entropy_bits": right_summary["entropy_bits"],
                                "pmf_total_variation": tv,
                                "pmf_wasserstein_rank_distance": wasserstein,
                            }
                        )
            if draw_index % 25 == 0 or draw_index == len(trials):
                print(
                    f"Completed {panel} inference trials: {draw_index}/{len(trials)}",
                    flush=True,
                )

    summaries = summarize_pairs(pair_trial_rows["unconditional"], "unconditional")
    summaries += summarize_pairs(
        pair_trial_rows["record_conditioned"], "record_conditioned"
    )
    _write_csv(output_dir / "team_trial_summaries.csv", team_summary_rows)
    _write_csv(
        output_dir / "pair_trial_summaries.csv",
        pair_trial_rows["unconditional"] + pair_trial_rows["record_conditioned"],
    )
    _write_csv(output_dir / "simulation_summary.csv", summaries)
    _write_gzip_csv(output_dir / "simulated_games.csv.gz", game_artifact_rows)
    _write_gzip_csv(output_dir / "representative_pmfs.csv.gz", representative_pmfs)
    _write_csv(output_dir / "representative_comparison.csv", representative_rows)

    sensitivity_rows, counterfactual_pmfs, sensitivity_trials = run_outcome_sensitivity(
        focal.prior, schedules, likelihood
    )
    _write_csv(output_dir / "outcome_sensitivity.csv", sensitivity_rows)
    _write_gzip_csv(output_dir / "counterfactual_pmfs.csv.gz", counterfactual_pmfs)
    sensitivity_game_rows = [
        {"seed": trial.seed, **row}
        for group, trial in sensitivity_trials.items()
        for row in trial.groups[group].game_rows
    ]
    _write_gzip_csv(
        output_dir / "sensitivity_representative_games.csv.gz", sensitivity_game_rows
    )

    # The manifest captures both the local input artifact and all outputs needed
    # to reconstruct the deterministic study.
    source_prior_hash = sha256_file(prior_path)
    source_team_rows = [
        row for row in team_rows.values() if row.get("subdivision") == "fbs"
    ]
    focal_mean = float(np.dot(np.arange(1, len(focal.prior) + 1), focal.prior))
    schedule_means = {
        role: float(np.mean([slot.expected_win_probability for slot in slots]))
        for role, slots in schedules.items()
    }
    report = _render_report(
        summaries,
        {
            role: tuple(slot.expected_win_probability for slot in slots)
            for role, slots in schedules.items()
        },
        len(conditioned_trials),
        conditioned_attempts,
        representative_rows,
        sensitivity_rows,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    artifact_paths = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    artifact_hashes = {path.name: sha256_file(path) for path in artifact_paths}
    try:
        source_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        source_revision = "unavailable"
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "issue": 252,
        "study": "research_only_quality_of_results_schedule_equivalence",
        "source_revision": source_revision,
        "source_code_sha256": sha256_file(
            root / "src/gippyrank/research/issue_252_quality_of_results.py"
        ),
        "protocol_sha256": sha256_file(
            root / "docs/issue_252_quality_of_results_protocol.md"
        ),
        "candidate_margin_weights": list(MARGIN_WEIGHTS),
        "inference": {
            "implementation": "issue242_quality_of_results_v1",
            "max_iterations": 500,
            "tolerance": 1e-9,
            "damping": 0.35,
            "prior_stripping": "normalize(blended_context_posterior / focal_context_prior)",
            "rank_support": len(focal.prior),
            "convergence_required": True,
        },
        "inputs": {
            "season": 2026,
            "prior_family": "context",
            "prior_model_version": "1.4",
            "prior_path": str(prior_path.relative_to(root)),
            "prior_sha256": source_prior_hash,
            "prior_fbs_team_count": len(fbs),
            "prior_population_row_count": len(source_team_rows),
            "focal_team_id": focal.team_id,
            "focal_team_name": focal.name,
            "focal_prior_expected_rank": focal_mean,
            "historical_likelihood_path": "data/processed/posterior/historical_likelihood_v1.json",
            "historical_likelihood_sha256": sha256_file(likelihood_path),
            "historical_likelihood_version": "V1",
        },
        "schedule_design": {
            "venue_slots": list(SITE_SLOTS),
            "venue_counts": {
                venue: SITE_SLOTS.count(venue) for venue in ("home", "away", "neutral")
            },
            "target_probability_tolerance": EXPECTED_PROBABILITY_TOLERANCE,
            "computed_schedule_mean_probabilities": schedule_means,
            "opponent_beliefs": "independent_context_preseason_pmfs_with_uncertainty",
            "selection": "nearest_unused_opponent_probability_to_sorted_targets_then_team_id",
            "outcome_generation": "continuous_v1_student_t_margin_at_sampled_rank_pair_and_site",
        },
        "simulation": {
            "pilot": {
                "seed_start": PILOT_SEED_START,
                "draw_count": PILOT_DRAWS,
                "exact_record_matches": 9,
                "record_match_rate": 0.045,
                "approximately_even_mean_wins": 5.165,
                "approximately_even_sd_wins": 2.4653936458800505,
                "weaker_schedule_mean_wins": 7.735,
                "weaker_schedule_sd_wins": 1.9554780674714325,
                "three_trial_five_weight_inference_seconds": 6.089,
                "candidate_rank_outputs_inspected": False,
            },
            "unconditional_seed_start": UNCONDITIONAL_SEED_START,
            "unconditional_draw_count": len(unconditional_trials),
            "conditioned_seed_start": CONDITIONED_SEED_START,
            "conditioned_accepted_draw_count": len(conditioned_trials),
            "conditioned_attempt_count": conditioned_attempts,
            "representative_seed_by_panel": representative_seeds,
            "conditioned_accepted_seeds": [trial.seed for trial in conditioned_trials],
            "conditioned_search_limit": CONDITIONED_SEARCH_LIMIT,
            "conditioned_records": {
                "approximately_even_wins": 5,
                "weaker_schedule_wins": 8,
                "games_per_season": 10,
            },
            "sensitivity_seed_start": SENSITIVITY_SEED_START,
            "sensitivity_seed_count": SENSITIVITY_SEED_COUNT,
            "sensitivity_representative_seed_by_role": {
                role: trial.seed for role, trial in sensitivity_trials.items()
            },
            "random_generator": f"numpy.random.default_rng / NumPy {np.__version__}",
            "scipy_version": scipy.__version__,
            "matched_draws": "shared focal rank, opponent rank quantiles, and Student-t quantiles across schedule pair; independent shocks for same-schedule control",
            "monte_carlo_intervals": "normal 95% interval for means; Wilson 95% interval for outrank probabilities",
        },
        "artifact_sha256": artifact_hashes,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="regenerate the report and manifest from saved artifacts without inference",
    )
    args = parser.parse_args()
    if args.report_only:
        manifest = refresh_report_from_artifacts(args.root, args.output)
    else:
        manifest = run_experiment(args.root, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
