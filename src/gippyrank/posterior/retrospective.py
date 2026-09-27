"""Retrospective leave-one-game-out expected-outcome distributions.

This module answers a deliberately different question from a historical
forecast: given all *other* evidence included by one selected snapshot, how
unusual was a completed game's observed home-oriented margin?  The held-out
game is removed before belief propagation is rerun, so its score cannot enter
the posterior PMFs used in its own Historical Likelihood mixture.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from gippyrank.methodology import HISTORICAL_LIKELIHOOD_VERSION
from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    PosteriorResult,
    Team,
    infer_posterior,
)
from gippyrank.posterior.predictive import (
    FUTURE_MARGIN_DISPLAY_BINS,
    FUTURE_MARGIN_DISPLAY_MAX,
    FUTURE_MARGIN_DISPLAY_MIN,
    ScheduledGame,
    margin_display_distribution,
    mixture_cdf,
    posterior_prediction_team,
    predict_game,
    predictive_components,
)

RETROSPECTIVE_GAME_EXPECTATIONS_VERSION = "1.0"
RETROSPECTIVE_INFERENCE_IMPLEMENTATION = (
    "per_game_leave_one_out_component_bp_recompute"
)
RETROSPECTIVE_INTERPRETATION = (
    "This is a retrospective leave-one-game-out posterior predictive "
    "distribution. It describes how the observed result compares with what "
    "GippyRank would expect given all other evidence available at the "
    "selected snapshot. It is not the prediction GippyRank would have made "
    "before the game was played."
)


@dataclass(frozen=True)
class _LeaveOneOutResult:
    """One held-out posterior plus compact recomputation diagnostics."""

    game: Game
    posterior: PosteriorResult
    component_team_count: int
    component_game_count: int


def _scheduled_game(game: Game) -> ScheduledGame:
    """Convert observed evidence to the home-oriented predictive contract."""

    return ScheduledGame(
        game_id=game.game_id,
        home_id=game.home_id,
        away_id=game.away_id,
        home_subdivision=game.home_subdivision,
        away_subdivision=game.away_subdivision,
        neutral_site=game.neutral_site,
        schedule_status="completed",
    )


def _participant_component(
    game: Game, retained_games: Sequence[Game]
) -> tuple[set[str], list[Game]]:
    """Return only the retained factor-graph components touching a game.

    Components disconnected from both participants cannot influence either
    posterior marginal.  Recomputing just this union is equivalent to a
    season-wide replay for the two required PMFs and avoids needless work for
    separately connected schedule regions.
    """

    adjacency: dict[str, set[str]] = defaultdict(set)
    for retained in retained_games:
        adjacency[retained.home_id].add(retained.away_id)
        adjacency[retained.away_id].add(retained.home_id)
    component = {game.home_id, game.away_id}
    pending = deque(component)
    while pending:
        team_id = pending.popleft()
        for neighbor in adjacency[team_id]:
            if neighbor not in component:
                component.add(neighbor)
                pending.append(neighbor)
    return component, [
        retained
        for retained in retained_games
        if retained.home_id in component and retained.away_id in component
    ]


def _leave_one_out_result(
    game: Game,
    *,
    teams_by_id: Mapping[str, Team],
    games: Sequence[Game],
    likelihood: LikelihoodV1,
    max_iterations: int,
    tolerance: float,
    damping: float,
) -> _LeaveOneOutResult:
    """Recompute the exact selected-model BP problem with one game absent."""

    retained_games = [candidate for candidate in games if candidate.game_id != game.game_id]
    component_ids, component_games = _participant_component(game, retained_games)
    component_teams = [teams_by_id[team_id] for team_id in sorted(component_ids)]
    posterior = infer_posterior(
        component_teams,
        component_games,
        likelihood,
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
    )
    if not posterior.converged:
        raise RuntimeError(
            "leave-one-game-out posterior did not converge for "
            f"{game.game_id} after {posterior.iterations} iterations "
            f"(max delta {posterior.max_message_delta})"
        )
    return _LeaveOneOutResult(
        game=game,
        posterior=posterior,
        component_team_count=len(component_teams),
        component_game_count=len(component_games),
    )


def _expectation_record(
    leave_one_out: _LeaveOneOutResult,
    *,
    teams_by_id: Mapping[str, Team],
    likelihood: LikelihoodV1,
    source_snapshot_id: str,
) -> dict[str, Any]:
    """Summarize a held-out posterior using the canonical margin machinery."""

    game = leave_one_out.game
    scheduled = _scheduled_game(game)
    home = posterior_prediction_team(
        teams_by_id[game.home_id], leave_one_out.posterior.pmfs
    )
    away = posterior_prediction_team(
        teams_by_id[game.away_id], leave_one_out.posterior.pmfs
    )
    summary = predict_game(scheduled, home, away, likelihood)
    locations, weights = predictive_components(scheduled, home, away, likelihood)
    actual_home_margin = float(game.home_points - game.away_points)
    lower_tail = mixture_cdf(
        actual_home_margin,
        locations,
        weights,
        likelihood.scale,
        likelihood.degrees_of_freedom,
    )
    # The V1 Student-t mixture is continuous, so <= and >= tails meet at the
    # observed margin with no point mass. Clamp tiny floating error only.
    lower_tail = min(max(lower_tail, 0.0), 1.0)
    upper_tail = min(max(1.0 - lower_tail, 0.0), 1.0)
    return {
        "game_id": game.game_id,
        "source_snapshot_id": source_snapshot_id,
        "home_team_id": game.home_id,
        "away_team_id": game.away_id,
        "home_subdivision": game.home_subdivision,
        "away_subdivision": game.away_subdivision,
        "neutral_site": game.neutral_site,
        "margin_orientation": "home_minus_away",
        "actual_home_margin": actual_home_margin,
        "observed_margin_percentile": lower_tail,
        "lower_tail_probability": lower_tail,
        "upper_tail_probability": upper_tail,
        "display_distribution": margin_display_distribution(
            scheduled, home, away, likelihood
        ),
        **summary.as_dict(),
    }


def build_retrospective_game_expectations(
    *,
    metadata: Mapping[str, Any],
    teams: Sequence[Team],
    games: Sequence[Game],
    posterior: PosteriorResult,
    likelihood: LikelihoodV1,
    max_iterations: int,
    tolerance: float,
    damping: float,
    workers: int | None = None,
) -> dict[str, Any]:
    """Build true LOO expected outcomes for every modeled completed game.

    Each target is removed from the retained game list before inference,
    including when the target shares a grouped pair factor with another
    rematch. The replay starts from score-independent messages; pair cavities
    from the full posterior could retain indirect feedback from the target.
    """

    source_snapshot_id = str(metadata["snapshot_id"])
    team_by_id = {team.team_id: team for team in teams}
    if len(team_by_id) != len(teams):
        raise ValueError("retrospective expectations require unique team IDs")
    game_ids = [game.game_id for game in games]
    if len(game_ids) != len(set(game_ids)):
        raise ValueError("retrospective expectations require unique game IDs")
    if any(
        game.home_id not in team_by_id or game.away_id not in team_by_id
        for game in games
    ):
        raise ValueError("retrospective game references a team without a prior")
    if not posterior.converged:
        raise ValueError(
            "retrospective expectations require a converged production posterior"
        )
    requested_workers = workers if workers is not None else min(8, len(games))
    worker_count = max(1, min(int(requested_workers), len(games) or 1))
    started = perf_counter()

    def replay(game: Game) -> _LeaveOneOutResult:
        return _leave_one_out_result(
            game,
            teams_by_id=team_by_id,
            games=games,
            likelihood=likelihood,
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
        )

    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            replays = list(executor.map(replay, games))
    else:
        replays = [replay(game) for game in games]

    records = {
        replay_result.game.game_id: _expectation_record(
            replay_result,
            teams_by_id=team_by_id,
            likelihood=likelihood,
            source_snapshot_id=source_snapshot_id,
        )
        for replay_result in replays
    }
    component_team_counts = [result.component_team_count for result in replays]
    component_game_counts = [result.component_game_count for result in replays]
    iteration_counts = [result.posterior.iterations for result in replays]
    return {
        "retrospective_game_expectations_version": RETROSPECTIVE_GAME_EXPECTATIONS_VERSION,
        "artifact_kind": "retrospective_game_expectations",
        "source_snapshot_id": source_snapshot_id,
        "season": metadata["season"],
        "snapshot_type": metadata["snapshot_type"],
        "effective_cutoff": metadata.get("effective_cutoff"),
        "included_game_ids": list(metadata.get("included_game_ids", [])),
        "historical_likelihood_version": metadata.get(
            "historical_likelihood_version", HISTORICAL_LIKELIHOOD_VERSION
        ),
        "margin_orientation": "home_minus_away",
        "interpretation": RETROSPECTIVE_INTERPRETATION,
        "inference": {
            "implementation": RETROSPECTIVE_INFERENCE_IMPLEMENTATION,
            "excluded_evidence": "one exact game ID per distribution",
            "rematch_handling": "retain every other eligible game, including other games between the same teams",
            "initialization": (
                "uniform deterministic BP messages independent of the held-out score"
            ),
            "component_scope": "connected components containing either game participant after exclusion",
            "configured_workers": worker_count,
            "runtime_seconds": perf_counter() - started,
            "games_evaluated": len(replays),
            "component_team_count_min": min(component_team_counts, default=0),
            "component_team_count_max": max(component_team_counts, default=0),
            "component_game_count_min": min(component_game_counts, default=0),
            "component_game_count_max": max(component_game_counts, default=0),
            "recompute_iteration_count_min": min(iteration_counts, default=0),
            "recompute_iteration_count_max": max(iteration_counts, default=0),
        },
        "margin_axis": {
            "min_margin": FUTURE_MARGIN_DISPLAY_MIN,
            "max_margin": FUTURE_MARGIN_DISPLAY_MAX,
            "bins": FUTURE_MARGIN_DISPLAY_BINS,
            "direction": "home_minus_away",
            "unit": "points",
            "tail_handling": "tail mass is retained separately from visible bins",
            "probability_encoding": {
                "type": "fixed_scale_integer",
                "scale": 1000,
                "normalization": "divide weights by scale",
                "total_weight": 1000,
            },
        },
        "games": records,
    }
