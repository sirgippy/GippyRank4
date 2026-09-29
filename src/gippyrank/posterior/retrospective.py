"""Completed-game hindsight distributions from a selected snapshot's posterior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from gippyrank.methodology import (
    HISTORICAL_LIKELIHOOD_VERSION,
    RETROSPECTIVE_CONDITIONING,
    RETROSPECTIVE_GAME_EXPECTATIONS_VERSION,
)
from gippyrank.posterior.engine import Game, LikelihoodV1, PosteriorResult, Team
from gippyrank.posterior.predictive import (
    FUTURE_MARGIN_DISPLAY_BINS,
    FUTURE_MARGIN_DISPLAY_MAX,
    FUTURE_MARGIN_DISPLAY_MIN,
    ScheduledGame,
    _margin_display_distribution_from_components,
    _predict_game_from_components,
    mixture_cdf,
    posterior_prediction_team,
    predictive_components,
)

RETROSPECTIVE_INFERENCE_IMPLEMENTATION = "selected_snapshot_full_posterior_pmfs"
RETROSPECTIVE_INTERPRETATION = (
    "This retrospective distribution compares the observed result with what "
    "GippyRank expects given everything known at the selected snapshot, "
    "including this game. It is not a prediction made before the game."
)


def _expectation_record(
    game: Game,
    *,
    teams_by_id: Mapping[str, Team],
    posterior: PosteriorResult,
    likelihood: LikelihoodV1,
    source_snapshot_id: str,
) -> dict[str, Any]:
    """Use the canonical Historical Likelihood mixture and home orientation."""
    scheduled = ScheduledGame(
        game_id=game.game_id,
        home_id=game.home_id,
        away_id=game.away_id,
        home_subdivision=game.home_subdivision,
        away_subdivision=game.away_subdivision,
        neutral_site=game.neutral_site,
        schedule_status="completed",
    )
    home = posterior_prediction_team(teams_by_id[game.home_id], posterior.pmfs)
    away = posterior_prediction_team(teams_by_id[game.away_id], posterior.pmfs)
    locations, weights = predictive_components(scheduled, home, away, likelihood)
    components = (locations, weights)
    summary = _predict_game_from_components(likelihood, components)
    actual_home_margin = game.home_points - game.away_points
    lower_tail = min(
        max(
            mixture_cdf(
                float(actual_home_margin),
                locations,
                weights,
                likelihood.scale,
                likelihood.degrees_of_freedom,
            ),
            0.0,
        ),
        1.0,
    )
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
        "upper_tail_probability": min(max(1.0 - lower_tail, 0.0), 1.0),
        "display_distribution": _margin_display_distribution_from_components(
            likelihood, components
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
    likelihood_sha256: str | None = None,
) -> dict[str, Any]:
    """Summarize completed games using already-computed full posterior PMFs."""
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
    records = {
        game.game_id: _expectation_record(
            game,
            teams_by_id=team_by_id,
            posterior=posterior,
            likelihood=likelihood,
            source_snapshot_id=source_snapshot_id,
        )
        for game in games
    }
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
        "historical_likelihood_sha256": likelihood_sha256,
        "posterior_pmfs_sha256": metadata.get("posterior_pmfs_sha256"),
        "conditioning": RETROSPECTIVE_CONDITIONING,
        "margin_orientation": "home_minus_away",
        "interpretation": RETROSPECTIVE_INTERPRETATION,
        "inference": {
            "implementation": RETROSPECTIVE_INFERENCE_IMPLEMENTATION,
            "games_evaluated": len(records),
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
