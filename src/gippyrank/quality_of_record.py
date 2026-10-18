"""Context-anchored Quality of Record inference from binary game outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import stdtr

from gippyrank.performance_snapshot import strip_context_prior
from gippyrank.performance_v1 import evidence_counts, uniform_pmf
from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    PosteriorResult,
    Team,
    game_margin_parameters,
    infer_posterior,
)


@dataclass(frozen=True)
class QualityOfRecordInference:
    """Binary network posterior and prior-stripped FBS rank distributions."""

    binary_context_posterior: PosteriorResult
    pmfs: dict[str, np.ndarray]
    eligible_game_counts: dict[str, int]


def binary_game_factor(
    game: Game, home: Team, away: Team, likelihood: LikelihoodV1
) -> np.ndarray:
    """Return the frozen V1 probability of the observed winner for each rank pair.

    ``game_margin_parameters`` supplies the original site and subdivision
    orientation. Only the sign of its observed margin is used here.
    """
    locations, margin = game_margin_parameters(game, home, away, likelihood)
    if margin == 0:
        raise ValueError(f"game {game.game_id} has no binary winner")
    sign = 1.0 if margin > 0 else -1.0
    return stdtr(likelihood.degrees_of_freedom, sign * locations / likelihood.scale)


def infer_quality_of_record(
    teams: list[Team],
    games: list[Game],
    likelihood: LikelihoodV1,
    *,
    max_iterations: int = 500,
    tolerance: float = 1e-9,
    damping: float = 0.35,
) -> QualityOfRecordInference:
    """Run binary-only BP on the complete network and strip each FBS prior."""
    result = infer_posterior(
        teams,
        games,
        likelihood,
        factor_builder=lambda game, home, away: binary_game_factor(
            game, home, away, likelihood
        ),
        factor_cache_key=("quality_of_record_v1", likelihood.beta.tobytes()),
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
    )
    if not result.converged:
        raise RuntimeError(
            "Quality of Record binary posterior did not converge "
            f"after {result.iterations} iterations"
        )
    counts = evidence_counts(teams, games)
    pmfs = {
        team.team_id: (
            strip_context_prior(result.pmfs[team.team_id], team.prior)
            if counts[team.team_id]
            else uniform_pmf(len(team.prior))
        )
        for team in teams
        if team.subdivision == "fbs"
    }
    return QualityOfRecordInference(result, pmfs, counts)
