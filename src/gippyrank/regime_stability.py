"""Leakage-safe primitives for the regime-stability research experiment.

These helpers deliberately know nothing about a production specification or a
future-season artifact.  A caller supplies completed team-seasons and a target
year; every returned training row is strictly older than that target.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log

import numpy as np

from gippyrank.preseason import TeamSeason


@dataclass(frozen=True)
class TrainingPlan:
    """A leakage-safe training subset and its relative likelihood weights."""

    target_season: int
    rows: list[TeamSeason]
    weights: np.ndarray


def exponential_weights(
    seasons: np.ndarray, target_season: int, half_life: float | None
) -> np.ndarray:
    """Return positive exponential recency weights relative to a target year.

    ``None`` is the exact equal-weight limit.  Passing a target or future
    season is rejected rather than silently assigning it a non-causal weight.
    """
    seasons = np.asarray(seasons, dtype=int)
    ages = target_season - seasons
    if np.any(ages <= 0):
        raise ValueError("recency weights require seasons strictly before target")
    if half_life is None:
        return np.ones(len(seasons), dtype=float)
    if half_life <= 0:
        raise ValueError("half_life must be positive or None")
    return np.exp(-log(2.0) * ages / half_life)


def training_plan(
    rows: list[TeamSeason],
    target_season: int,
    *,
    half_life: float | None = None,
    window: int | None = None,
) -> TrainingPlan:
    """Select completed rows, optionally within a backward-looking window."""
    selected = [row for row in rows if row.season < target_season]
    if window is not None:
        if window < 1:
            raise ValueError("window must be positive")
        selected = [row for row in selected if row.season >= target_season - window]
    if not selected:
        raise ValueError("training plan has no rows")
    return TrainingPlan(
        target_season,
        selected,
        exponential_weights(
            np.asarray([row.season for row in selected]), target_season, half_life
        ),
    )


def nested_family_prior_years(
    candidates: Sequence[str],
    prior_scores: Mapping[int, Mapping[str, float]],
    target: int,
) -> list[int]:
    """Return earlier targets with scores for every member of one family."""
    return sorted(
        year
        for year, scores in prior_scores.items()
        if year < target and all(candidate in scores for candidate in candidates)
    )


def nested_family_choice(
    candidates: Sequence[str],
    prior_scores: Mapping[int, Mapping[str, float]],
    target: int,
    *,
    default_candidate: str,
) -> str:
    """Choose prospectively within one explicit candidate family.

    Scores from the target being chosen, and candidates outside ``candidates``,
    are deliberately unavailable to this calculation.  The configured static
    candidate is the deterministic default before any common prior target.
    """
    if not candidates:
        raise ValueError("nested family choice requires at least one candidate")
    if default_candidate not in candidates:
        raise ValueError("nested family default must be a family candidate")
    available = nested_family_prior_years(candidates, prior_scores, target)
    if not available:
        return default_candidate
    return min(
        candidates,
        key=lambda name: (
            float(np.mean([prior_scores[year][name] for year in available])),
            candidates.index(name),
        ),
    )


def nested_choice(
    candidates: list[str], prior_scores: dict[int, dict[str, float]], target: int
) -> str:
    """Backward-compatible generic prospective choice for a supplied family."""
    if not candidates:
        raise ValueError("nested choice requires at least one candidate")
    return nested_family_choice(
        candidates, prior_scores, target, default_candidate=candidates[0]
    )


def assert_same_keys(
    left: set[tuple[int, str, str]], right: set[tuple[int, str, str]]
) -> None:
    """Guard every paired H/C comparison against population drift."""
    if left != right:
        raise ValueError("paired models must predict identical target-team keys")


def decomposition_total(contributions: np.ndarray) -> float:
    """Aggregate paired per-team score differences without changing weights."""
    values = np.asarray(contributions, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("score decomposition must be finite")
    return float(values.sum())
