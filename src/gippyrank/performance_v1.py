"""Research-only Performance V1 inference and PMF utilities.

Performance removes the focal team's preseason prior as a direct factor from
the shared Historical Likelihood V1 posterior. It deliberately keeps the
predictive anchor for the rest of the schedule network, so opponent quality
remains informed by the selected Context or History preseason model. Under
approximate loopy BP, a small indirect feedback residue can remain when prior
information travels through opponents and returns through schedule cycles;
explicit focal-prior neutralization is the correctness baseline.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np

from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    PosteriorResult,
    Team,
    infer_posterior,
)

AnchorFamily = Literal["context", "history"]
PerformanceMethod = Literal["prior_stripping", "explicit_neutralized"]


@dataclass(frozen=True)
class PerformanceInference:
    """A shared-network Performance result for one anchor family and cutoff."""

    anchor_family: AnchorFamily
    method: PerformanceMethod
    teams: tuple[Team, ...]
    games: tuple[Game, ...]
    anchor_result: PosteriorResult
    pmfs: dict[str, np.ndarray]
    eligible_game_counts: dict[str, int]


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("PMF must be a finite, nonempty vector")
    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("PMF must contain positive mass")
    return values / total


def uniform_pmf(size: int) -> np.ndarray:
    """Return the neutral focal prior on ranks 1..size."""
    if size < 1:
        raise ValueError("uniform PMF support must be positive")
    return np.full(size, 1.0 / size)


def remove_focal_prior(posterior: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Compute ``normalize(posterior / prior)`` with explicit support checks."""
    posterior = _normalise(posterior)
    prior = _normalise(prior)
    if posterior.shape != prior.shape:
        raise ValueError("posterior and prior must have identical rank support")
    positive = prior > 0
    if np.any(~positive & (posterior > 1e-14)):
        raise ValueError("posterior has mass outside the focal prior support")
    ratio = np.zeros_like(posterior)
    ratio[positive] = posterior[positive] / prior[positive]
    return _normalise(ratio)


def _neutralized_teams(teams: tuple[Team, ...], target_id: str) -> list[Team]:
    found = False
    result = []
    for team in teams:
        if team.team_id == target_id:
            found = True
            result.append(
                Team(
                    team.team_id,
                    team.name,
                    team.subdivision,
                    uniform_pmf(len(team.prior)),
                )
            )
        else:
            result.append(team)
    if not found:
        raise KeyError(f"target team is absent from inference: {target_id}")
    return result


def explicit_neutralized_target(
    teams: list[Team] | tuple[Team, ...],
    games: list[Game] | tuple[Game, ...],
    likelihood: LikelihoodV1,
    target_id: str,
    *,
    max_iterations: int = 500,
    tolerance: float = 1e-9,
    damping: float = 0.35,
) -> tuple[np.ndarray, PosteriorResult]:
    """Run ordinary BP with only ``target_id`` replaced by a uniform prior."""
    frozen_teams = tuple(teams)
    result = infer_posterior(
        _neutralized_teams(frozen_teams, target_id),
        list(games),
        likelihood,
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
    )
    return result.pmfs[target_id], result


def evidence_counts(
    teams: list[Team] | tuple[Team, ...], games: list[Game]
) -> dict[str, int]:
    counts = Counter({team.team_id: 0 for team in teams})
    for game in games:
        counts[game.home_id] += 1
        counts[game.away_id] += 1
    return dict(counts)


def infer_performance(
    teams: list[Team] | tuple[Team, ...],
    games: list[Game] | tuple[Game, ...],
    likelihood: LikelihoodV1,
    *,
    anchor_family: AnchorFamily,
    method: PerformanceMethod = "prior_stripping",
    max_iterations: int = 500,
    tolerance: float = 1e-9,
    damping: float = 0.35,
) -> PerformanceInference:
    """Infer Performance PMFs for FBS teams on the complete game network.

    ``prior_stripping`` is the efficient implementation of the exact
    factorized identity. ``explicit_neutralized`` is the correctness baseline
    that makes the focal prior uniform before the ordinary BP run.
    """
    if anchor_family not in {"context", "history"}:
        raise ValueError(f"unsupported anchor family: {anchor_family}")
    if method not in {"prior_stripping", "explicit_neutralized"}:
        raise ValueError(f"unsupported Performance method: {method}")
    frozen_teams = tuple(teams)
    frozen_games = tuple(games)
    anchor_result = infer_posterior(
        list(frozen_teams),
        list(frozen_games),
        likelihood,
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
    )
    counts = evidence_counts(frozen_teams, list(frozen_games))
    pmfs: dict[str, np.ndarray] = {}
    for team in frozen_teams:
        if team.subdivision != "fbs":
            continue
        if method == "prior_stripping":
            pmfs[team.team_id] = remove_focal_prior(
                anchor_result.pmfs[team.team_id], team.prior
            )
        else:
            pmfs[team.team_id], _ = explicit_neutralized_target(
                frozen_teams,
                frozen_games,
                likelihood,
                team.team_id,
                max_iterations=max_iterations,
                tolerance=tolerance,
                damping=damping,
            )
    return PerformanceInference(
        anchor_family,
        method,
        frozen_teams,
        frozen_games,
        anchor_result,
        pmfs,
        counts,
    )


def pmf_quantile(pmf: np.ndarray, probability: float) -> int:
    if not 0 < probability <= 1:
        raise ValueError("probability must be in (0, 1]")
    pmf = _normalise(pmf)
    ranks = np.arange(1, len(pmf) + 1)
    return int(ranks[np.searchsorted(np.cumsum(pmf), probability, side="left")])


def performance_pmf_summaries(pmf: np.ndarray) -> dict[str, float | int]:
    """Return display and uncertainty summaries for a discrete rank PMF."""
    pmf = _normalise(pmf)
    ranks = np.arange(1, len(pmf) + 1)
    result: dict[str, float | int] = {
        "expected_rank": float(np.dot(ranks, pmf)),
        "median_rank": pmf_quantile(pmf, 0.5),
        "mode_rank": int(np.argmax(pmf) + 1),
    }
    for level in (50, 80, 95):
        tail = (1 - level / 100) / 2
        result[f"interval_{level}_low"] = pmf_quantile(pmf, tail)
        result[f"interval_{level}_high"] = pmf_quantile(pmf, 1 - tail)
    for threshold in (5, 10, 25):
        result[f"top{threshold}_probability"] = float(pmf[:threshold].sum())
    return result


def pmf_tv_distance(left: np.ndarray, right: np.ndarray) -> float:
    left, right = _normalise(left), _normalise(right)
    if left.shape != right.shape:
        raise ValueError("PMFs must have identical rank support")
    return float(0.5 * np.abs(left - right).sum())
