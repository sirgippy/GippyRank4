"""Exact small-graph validation utilities for posterior approximation audits."""

from __future__ import annotations

from itertools import product

import numpy as np

from gippyrank.posterior.engine import Game, LikelihoodV1, Team, game_factor


def exact_marginals(
    teams: list[Team], games: list[Game], likelihood: LikelihoodV1
) -> dict[str, np.ndarray]:
    """Enumerate a tiny joint rank space exactly; intended only for tests/audits."""
    by_id = {team.team_id: team for team in teams}
    factors = [
        (game, game_factor(game, by_id[game.home_id], by_id[game.away_id], likelihood))
        for game in games
    ]
    marginals = {team.team_id: np.zeros(len(team.prior)) for team in teams}
    total = 0.0
    index = {team.team_id: position for position, team in enumerate(teams)}
    for state in product(*(range(len(team.prior)) for team in teams)):
        weight = float(
            np.prod(
                [team.prior[state[position]] for position, team in enumerate(teams)]
            )
        )
        for game, factor in factors:
            weight *= factor[
                state[index[game.home_id]], state[index[game.away_id]]
            ]
        total += weight
        for position, team in enumerate(teams):
            marginals[team.team_id][state[position]] += weight
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("exact posterior has zero or non-finite mass")
    return {team_id: values / total for team_id, values in marginals.items()}
