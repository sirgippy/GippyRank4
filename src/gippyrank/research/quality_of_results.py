"""Research-only inference combining binary outcomes and scoring margins.

The tempered factor keeps the full binary winner evidence and raises the
conditional margin evidence to a configurable power. At the endpoints it uses
the existing Quality of Record and Quality of Play factors verbatim.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from gippyrank.performance_snapshot import strip_context_prior
from gippyrank.performance_v1 import evidence_counts, uniform_pmf
from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    PosteriorResult,
    Team,
    game_factor,
    infer_posterior,
)
from gippyrank.quality_of_record import binary_game_factor


@dataclass(frozen=True)
class QualityOfResultsInference:
    """One tempered posterior and its prior-stripped FBS rank distributions."""

    margin_weight: float
    blended_context_posterior: PosteriorResult
    pmfs: dict[str, np.ndarray]
    eligible_game_counts: dict[str, int]


def validate_margin_weight(margin_weight: float) -> float:
    """Return a finite margin weight in [0, 1]."""
    value = float(margin_weight)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("margin_weight must be finite and in [0, 1]")
    return value


def blended_game_factor(
    game: Game,
    home: Team,
    away: Team,
    likelihood: LikelihoodV1,
    margin_weight: float,
    *,
    factor_surface_cache: dict[tuple[object, ...], dict[str, np.ndarray]] | None = None,
) -> np.ndarray:
    """Return the research likelihood factor for one completed game.

    For 0 < weight < 1 the factor is
    P(sign(m)|theta) ** (1-weight) * p(m|theta) ** weight. The engine's
    Quality of Play factor may be multiplied by a positive game-specific
    constant; that constant cancels from inference and preserves exact endpoint
    compatibility with the existing implementation.
    """
    weight = validate_margin_weight(margin_weight)
    surfaces: dict[str, np.ndarray] | None = None
    if factor_surface_cache is not None:
        cache_key = (
            game.game_id,
            game.home_id,
            game.away_id,
            game.home_points,
            game.away_points,
            game.neutral_site,
            home.subdivision,
            away.subdivision,
            len(home.prior),
            len(away.prior),
            hashlib.sha256(likelihood.beta.tobytes()).hexdigest(),
            likelihood.scale,
            likelihood.degrees_of_freedom,
        )
        surfaces = factor_surface_cache.setdefault(cache_key, {})
    if weight == 0.0:
        if surfaces is None:
            return binary_game_factor(game, home, away, likelihood)
        if "outcome" not in surfaces:
            surfaces["outcome"] = binary_game_factor(game, home, away, likelihood)
        return surfaces["outcome"]
    if weight == 1.0:
        if surfaces is None:
            return game_factor(game, home, away, likelihood)
        if "margin" not in surfaces:
            surfaces["margin"] = game_factor(game, home, away, likelihood)
        return surfaces["margin"]

    if surfaces is None:
        outcome = binary_game_factor(game, home, away, likelihood)
        margin = game_factor(game, home, away, likelihood)
    else:
        if "outcome" not in surfaces:
            surfaces["outcome"] = binary_game_factor(game, home, away, likelihood)
        if "margin" not in surfaces:
            surfaces["margin"] = game_factor(game, home, away, likelihood)
        outcome, margin = surfaces["outcome"], surfaces["margin"]
    tiny = np.finfo(float).tiny
    log_factor = (1.0 - weight) * np.log(np.maximum(outcome, tiny))
    log_factor += weight * np.log(np.maximum(margin, tiny))
    log_factor -= float(np.max(log_factor))
    factor = np.exp(log_factor)
    if not np.isfinite(factor).all() or not np.any(factor > 0):
        raise FloatingPointError(f"{game.game_id}: blended game factor is invalid")
    return factor


def infer_quality_of_results(
    teams: list[Team] | tuple[Team, ...],
    games: list[Game] | tuple[Game, ...],
    likelihood: LikelihoodV1,
    margin_weight: float,
    *,
    max_iterations: int = 500,
    tolerance: float = 1e-9,
    damping: float = 0.35,
    factor_surface_cache: dict[tuple[object, ...], dict[str, np.ndarray]] | None = None,
) -> QualityOfResultsInference:
    """Infer one blended network posterior and strip each FBS Context prior.

    This runs one belief-propagation model for the requested weight. It does
    not combine completed ranking outputs. Non-convergence fails closed, and
    FBS teams without eligible games retain the existing uniform Performance
    distribution.
    """
    weight = validate_margin_weight(margin_weight)
    frozen_teams = tuple(teams)
    frozen_games = tuple(games)
    beta_hash = hashlib.sha256(likelihood.beta.tobytes()).hexdigest()
    factor_cache_key = (
        "issue242_quality_of_results_v1",
        beta_hash,
        likelihood.scale.hex(),
        likelihood.degrees_of_freedom.hex(),
        weight.hex(),
    )
    result = infer_posterior(
        list(frozen_teams),
        list(frozen_games),
        likelihood,
        factor_builder=lambda game, home, away: blended_game_factor(
            game,
            home,
            away,
            likelihood,
            weight,
            factor_surface_cache=factor_surface_cache,
        ),
        factor_cache_key=factor_cache_key,
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
    )
    if not result.converged:
        raise RuntimeError(
            "Quality of Results posterior did not converge "
            f"after {result.iterations} iterations at margin_weight={weight:g}"
        )

    counts = evidence_counts(frozen_teams, list(frozen_games))
    pmfs = {
        team.team_id: (
            strip_context_prior(result.pmfs[team.team_id], team.prior)
            if counts[team.team_id]
            else uniform_pmf(len(team.prior))
        )
        for team in frozen_teams
        if team.subdivision == "fbs"
    }
    return QualityOfResultsInference(weight, result, pmfs, counts)
