"""Hierarchical posterior-predictive simulation for the remaining season.

The season simulator has two deliberately separate layers.  An outer draw
chooses one latent ordinal rank for every modeled team from that team's
snapshot posterior PMF.  Those ranks stay fixed for the whole draw.  Given a
fixed draw, future games are conditionally independent Historical Likelihood
V1 outcomes.  Regular-season win totals use the exact conditional
Poisson-binomial distribution, which avoids adding avoidable inner Monte
Carlo noise while retaining the same generative semantics as game rollouts.

The production V1 approximation samples team marginal PMFs independently.
The full joint posterior is not yet exposed by the inference layer, so this
is an explicit approximation rather than an implicit permutation of ranks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from gippyrank.posterior.engine import LikelihoodV1, Team
from gippyrank.posterior.predictive import (
    ScheduledGame,
    conditional_margin_location_surface,
    mixture_cdf,
    predictive_components,
    win_probabilities,
)

SEASON_SIMULATION_SCHEMA_VERSION = "1.0"
SEASON_SIMULATION_VERSION = "hierarchical_latent_state_v1"
DEFAULT_OUTER_DRAW_COUNT = 2_000
DEFAULT_INNER_ROLLOUT_COUNT = 0
DEFAULT_SEASON_SIMULATION_SEED = 49_049
SEASON_WIN_THRESHOLDS = (6, 8, 10, 11, 12)
_PMF_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class CompletedRecord:
    """A team's fixed regular-season record at the selected snapshot."""

    wins: int = 0
    losses: int = 0
    ties: int = 0

    def __post_init__(self) -> None:
        for field in ("wins", "losses", "ties"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"completed record {field} must be an integer")
            if value < 0:
                raise ValueError(f"completed record {field} must be non-negative")
            object.__setattr__(self, field, int(value))


@dataclass(frozen=True)
class SeasonSimulationConfig:
    """Explicit reproducibility and budget controls for Season Simulation V1."""

    outer_draw_count: int = DEFAULT_OUTER_DRAW_COUNT
    inner_rollout_count: int = DEFAULT_INNER_ROLLOUT_COUNT
    seed: int = DEFAULT_SEASON_SIMULATION_SEED
    simulation_version: str = SEASON_SIMULATION_VERSION
    likelihood_version: str = "V1"

    def __post_init__(self) -> None:
        for field in ("outer_draw_count", "inner_rollout_count", "seed"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"season simulation {field} must be an integer")
            value = int(value)
            if field == "outer_draw_count" and value < 1:
                raise ValueError("season simulation needs at least one outer draw")
            if field == "inner_rollout_count" and value < 0:
                raise ValueError("season simulation inner rollout count is negative")
            if field == "seed" and not 0 <= value < 2**63:
                raise ValueError("season simulation seed must fit a non-negative int64")
            object.__setattr__(self, field, value)
        if not self.simulation_version or not self.likelihood_version:
            raise ValueError("season simulation versions must be non-empty")

    @property
    def n_outer(self) -> int:
        """Compatibility spelling for the outer universe count."""

        return self.outer_draw_count

    @property
    def n_inner(self) -> int:
        """Compatibility spelling for the optional inner rollout count."""

        return self.inner_rollout_count

    def as_dict(self) -> dict[str, Any]:
        """Return the stable, JSON-ready configuration contract."""

        return {
            "simulation_version": self.simulation_version,
            "seed": self.seed,
            "outer_draw_count": self.outer_draw_count,
            "inner_rollout_count": self.inner_rollout_count,
            "latent_sampling_method": "independent_marginal_pmf",
            "joint_posterior_approximation": (
                "sample_each_team_marginal_independently; no permutation enforcement"
            ),
            "conditional_distribution_method": "exact_poisson_binomial",
            "likelihood_version": self.likelihood_version,
        }


@dataclass(frozen=True)
class LatentQualitySamples:
    """One fixed latent rank per team and outer universe."""

    team_ids: tuple[str, ...]
    ranks: np.ndarray

    def __post_init__(self) -> None:
        ranks = np.asarray(self.ranks, dtype=np.int64)
        if ranks.ndim != 2 or ranks.shape[1] != len(self.team_ids):
            raise ValueError("latent rank samples must be outer draws by team")
        if np.any(ranks < 1):
            raise ValueError("latent rank samples must use one-based ranks")
        ranks.setflags(write=False)
        object.__setattr__(self, "ranks", ranks)

    @property
    def outer_draw_count(self) -> int:
        return int(self.ranks.shape[0])

    def for_team(self, team_id: str) -> np.ndarray:
        """Return the fixed rank draw for one team across outer universes."""

        try:
            index = self.team_ids.index(team_id)
        except ValueError as error:
            raise KeyError(f"latent rank samples are missing team {team_id}") from error
        return self.ranks[:, index].copy()


@dataclass(frozen=True)
class _PreparedSimulation:
    teams: tuple[Team, ...]
    teams_by_id: dict[str, Team]
    fbs_team_ids: tuple[str, ...]
    candidate_games_by_team: dict[str, tuple[ScheduledGame, ...]]
    supported_games: tuple[ScheduledGame, ...]
    unsupported_games: tuple[dict[str, Any], ...]
    unsupported_by_team: dict[str, tuple[dict[str, Any], ...]]


def _validated_pmf(value: object, label: str) -> np.ndarray:
    pmf = np.asarray(value, dtype=float)
    if pmf.ndim != 1 or not len(pmf) or not np.isfinite(pmf).all():
        raise ValueError(f"{label} must be a finite, nonempty PMF")
    if np.any(pmf < 0):
        raise ValueError(f"{label} contains negative probability")
    total = float(pmf.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError(f"{label} must have positive finite mass")
    if not np.isclose(total, 1.0, atol=_PMF_TOLERANCE, rtol=0.0):
        raise ValueError(f"{label} must sum to one")
    return pmf / total


def _ordered_teams(
    teams: Sequence[Team], posterior_pmfs: Mapping[str, np.ndarray]
) -> tuple[Team, ...]:
    by_id = {team.team_id: team for team in teams}
    if len(by_id) != len(teams):
        raise ValueError("season simulation team IDs must be unique")
    missing = sorted(set(by_id) - set(posterior_pmfs))
    if missing:
        raise KeyError(f"season simulation posterior PMFs are missing teams: {missing}")
    for team_id, team in by_id.items():
        pmf = _validated_pmf(posterior_pmfs[team_id], f"posterior PMF for {team_id}")
        if len(pmf) != len(team.prior):
            raise ValueError(
                f"posterior PMF for {team_id} has {len(pmf)} ranks; "
                f"team metadata has {len(team.prior)}"
            )
    return tuple(by_id[team_id] for team_id in sorted(by_id))


def sample_latent_qualities(
    teams: Sequence[Team],
    posterior_pmfs: Mapping[str, np.ndarray],
    *,
    outer_draw_count: int = DEFAULT_OUTER_DRAW_COUNT,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> LatentQualitySamples:
    """Sample one posterior latent rank per team and outer universe.

    The random draw is made here, before any game is visited.  Consumers must
    reuse the returned matrix for every future game in the universe; sampling
    this function again for each matchup would violate the season invariant.
    """

    if isinstance(outer_draw_count, bool) or not isinstance(
        outer_draw_count, (int, np.integer)
    ):
        raise TypeError("outer_draw_count must be an integer")
    outer_draw_count = int(outer_draw_count)
    if outer_draw_count < 1:
        raise ValueError("outer_draw_count must be positive")
    if rng is not None and seed is not None:
        raise ValueError("provide rng or seed, not both")
    generator = rng if rng is not None else np.random.default_rng(
        DEFAULT_SEASON_SIMULATION_SEED if seed is None else seed
    )
    ordered = _ordered_teams(teams, posterior_pmfs)
    ranks = np.empty((outer_draw_count, len(ordered)), dtype=np.int64)
    for index, team in enumerate(ordered):
        pmf = _validated_pmf(posterior_pmfs[team.team_id], f"posterior PMF for {team.team_id}")
        cumulative = np.cumsum(pmf)
        draws = generator.random(outer_draw_count)
        rank_indices = np.searchsorted(cumulative, draws, side="right")
        # Floating-point normalization can leave the final cumulative value a
        # few ulps below one.  Generator.random never returns one, but clamp
        # defensively so a valid PMF can never produce an out-of-support rank.
        ranks[:, index] = np.minimum(rank_indices, len(pmf) - 1) + 1
    return LatentQualitySamples(tuple(team.team_id for team in ordered), ranks)


def _completed_record(value: object) -> CompletedRecord:
    if value is None:
        return CompletedRecord()
    if isinstance(value, CompletedRecord):
        return value
    if isinstance(value, Mapping):
        return CompletedRecord(
            wins=value.get("wins", value.get("completed_wins", 0)),
            losses=value.get("losses", value.get("completed_losses", 0)),
            ties=value.get("ties", value.get("completed_ties", 0)),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) not in {2, 3}:
            raise ValueError("completed record sequences must contain wins, losses, ties")
        return CompletedRecord(*value, ties=0) if len(value) == 2 else CompletedRecord(*value)
    raise ValueError("completed records must be CompletedRecord, mapping, or sequence")


def _is_regular_game(game: ScheduledGame) -> bool:
    season_type = str(getattr(game, "season_type", "regular") or "regular").casefold()
    return season_type in {"", "regular"}


def _game_descriptor(game: ScheduledGame, reason: str) -> dict[str, Any]:
    return {
        "game_id": str(game.game_id),
        "home_team_id": str(game.home_id),
        "away_team_id": str(game.away_id),
        "home_subdivision": str(game.home_subdivision).casefold(),
        "away_subdivision": str(game.away_subdivision).casefold(),
        "neutral_site": bool(game.neutral_site),
        "season_type": str(getattr(game, "season_type", "regular") or "regular"),
        "reason": reason,
    }


def _prepare_simulation(
    teams: Sequence[Team], future_games: Sequence[ScheduledGame]
) -> _PreparedSimulation:
    ordered = tuple(sorted(teams, key=lambda team: team.team_id))
    teams_by_id = {team.team_id: team for team in ordered}
    if len(teams_by_id) != len(ordered):
        raise ValueError("season simulation team IDs must be unique")
    fbs_team_ids = tuple(
        team_id
        for team_id, team in teams_by_id.items()
        if str(team.subdivision).casefold() == "fbs"
    )
    fbs_set = set(fbs_team_ids)
    candidate_games_by_team: dict[str, list[ScheduledGame]] = {
        team_id: [] for team_id in fbs_team_ids
    }
    unsupported_by_team: dict[str, list[dict[str, Any]]] = {
        team_id: [] for team_id in fbs_team_ids
    }
    supported: list[ScheduledGame] = []
    unsupported: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for game in sorted(
        future_games,
        key=lambda item: (str(getattr(item, "date", "") or ""), str(item.game_id)),
    ):
        game_id = str(game.game_id)
        if not game_id:
            raise ValueError("future schedule games need stable IDs")
        if game_id in seen_ids:
            raise ValueError(f"duplicate future schedule game ID: {game_id}")
        seen_ids.add(game_id)
        if not _is_regular_game(game):
            continue

        home_subdivision = str(game.home_subdivision).casefold()
        away_subdivision = str(game.away_subdivision).casefold()
        participant_ids = [
            team_id
            for team_id, subdivision in (
                (str(game.home_id), home_subdivision),
                (str(game.away_id), away_subdivision),
            )
            if team_id in fbs_set or subdivision == "fbs"
        ]
        known_fbs_participants = [team_id for team_id in participant_ids if team_id in fbs_set]
        if not known_fbs_participants:
            continue
        for team_id in known_fbs_participants:
            candidate_games_by_team[team_id].append(game)

        reason: str | None = None
        if game.home_id == game.away_id:
            reason = "same_team_on_both_sides"
        elif home_subdivision not in {"fbs", "fcs"} or away_subdivision not in {
            "fbs",
            "fcs",
        }:
            reason = "unsupported_subdivision"
        elif game.home_id not in teams_by_id or game.away_id not in teams_by_id:
            reason = "missing_posterior_team"
        elif str(teams_by_id[game.home_id].subdivision).casefold() != home_subdivision or str(
            teams_by_id[game.away_id].subdivision
        ).casefold() != away_subdivision:
            reason = "schedule_subdivision_mismatch"

        if reason is not None:
            descriptor = _game_descriptor(game, reason)
            unsupported.append(descriptor)
            for team_id in known_fbs_participants:
                unsupported_by_team[team_id].append(descriptor)
        else:
            supported.append(game)

    return _PreparedSimulation(
        teams=ordered,
        teams_by_id=teams_by_id,
        fbs_team_ids=fbs_team_ids,
        candidate_games_by_team={
            team_id: tuple(games) for team_id, games in candidate_games_by_team.items()
        },
        supported_games=tuple(supported),
        unsupported_games=tuple(unsupported),
        unsupported_by_team={
            team_id: tuple(games) for team_id, games in unsupported_by_team.items()
        },
    )


def poisson_binomial_pmf(probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the exact PMF of a sum of independent Bernoulli outcomes."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(
        (values < 0) | (values > 1)
    ):
        raise ValueError("Poisson-binomial probabilities must be finite in [0, 1]")
    pmf = np.zeros(len(values) + 1, dtype=float)
    pmf[0] = 1.0
    for index, probability in enumerate(values):
        previous = pmf[: index + 1].copy()
        updated = np.zeros(index + 2, dtype=float)
        updated[: index + 1] += previous * (1.0 - probability)
        updated[1:] += previous * probability
        pmf[: index + 2] = updated
    total = float(pmf.sum())
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("Poisson-binomial PMF normalization failed")
    return pmf / total


def _vectorized_poisson_binomial(probabilities: np.ndarray) -> np.ndarray:
    """Calculate one exact conditional PMF for every outer universe."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(
        (values < 0) | (values > 1)
    ):
        raise ValueError("conditional probabilities must be finite in [0, 1]")
    outer_count, game_count = values.shape
    pmfs = np.zeros((outer_count, game_count + 1), dtype=float)
    pmfs[:, 0] = 1.0
    for index in range(game_count):
        previous = pmfs[:, : index + 1].copy()
        updated = np.zeros((outer_count, index + 2), dtype=float)
        updated[:, : index + 1] += previous * (1.0 - values[:, index, None])
        updated[:, 1:] += previous * values[:, index, None]
        pmfs[:, : index + 2] = updated
    totals = pmfs.sum(axis=1)
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise FloatingPointError("conditional Poisson-binomial normalization failed")
    return pmfs / totals[:, None]


def _pmf_quantile(pmf: np.ndarray, probability: float, offset: int = 0) -> int:
    index = int(np.searchsorted(np.cumsum(pmf), probability, side="left"))
    return offset + min(index, len(pmf) - 1)


def _pmf_interval(pmf: np.ndarray, level: float, offset: int = 0) -> list[int]:
    tail = (1.0 - level) / 2.0
    return [
        _pmf_quantile(pmf, tail, offset),
        _pmf_quantile(pmf, 1.0 - tail, offset),
    ]


def _continuous_interval(values: np.ndarray, level: float) -> list[float]:
    tail = (1.0 - level) / 2.0
    return [
        float(np.quantile(values, tail)),
        float(np.quantile(values, 1.0 - tail)),
    ]


def _variance_decomposition(quality: float, game_randomness: float) -> dict[str, float]:
    quality = max(float(quality), 0.0)
    game_randomness = max(float(game_randomness), 0.0)
    total = quality + game_randomness
    quality_fraction = quality / total if total > 0 else 0.0
    game_fraction = game_randomness / total if total > 0 else 0.0
    return {
        "total": total,
        "quality_uncertainty_variance": quality,
        "game_randomness_variance": game_randomness,
        "team_quality": quality,
        "game_randomness": game_randomness,
        "team_quality_fraction": quality_fraction,
        "game_randomness_fraction": game_fraction,
    }


def _event_summary(probabilities: np.ndarray) -> dict[str, Any]:
    values = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    if values.ndim != 1 or not len(values):
        raise ValueError("event conditional probabilities must be nonempty")
    overall = float(values.mean())
    quality = float(np.var(values))
    game = float(np.mean(values * (1.0 - values)))
    variance = _variance_decomposition(quality, game)
    return {
        "probability": overall,
        "conditional_probability_median": float(np.median(values)),
        "conditional_probability_interval_50": _continuous_interval(values, 0.50),
        "conditional_probability_interval_80": _continuous_interval(values, 0.80),
        "conditional_probability_interval_95": _continuous_interval(values, 0.95),
        "quality_uncertainty_variance": variance["quality_uncertainty_variance"],
        "game_randomness_variance": variance["game_randomness_variance"],
        "total_variance": variance["total"],
        "quality_uncertainty_fraction": variance["team_quality_fraction"],
        "game_randomness_fraction": variance["game_randomness_fraction"],
        "monte_carlo_standard_error": float(
            np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        ),
    }


def _record_key(wins: int, losses: int, ties: int) -> str:
    return f"{wins}-{losses}" + (f"-{ties}" if ties else "")


def _conditional_game_samples(
    prepared: _PreparedSimulation,
    latent: LatentQualitySamples,
    posterior_pmfs: Mapping[str, np.ndarray],
    likelihood: LikelihoodV1,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """Return conditional home-win probabilities, locations, and exact marginals."""

    prediction_teams = {
        team.team_id: Team(
            team.team_id,
            team.name,
            team.subdivision,
            _validated_pmf(posterior_pmfs[team.team_id], f"posterior PMF for {team.team_id}"),
        )
        for team in prepared.teams
    }
    positions = {team_id: index for index, team_id in enumerate(latent.team_ids)}
    probabilities: dict[str, np.ndarray] = {}
    locations_by_game: dict[str, np.ndarray] = {}
    exact: dict[str, dict[str, float]] = {}
    for game in prepared.supported_games:
        home = prediction_teams[game.home_id]
        away = prediction_teams[game.away_id]
        surface = conditional_margin_location_surface(game, home, away, likelihood)
        locations = surface[
            latent.ranks[:, positions[game.home_id]] - 1,
            latent.ranks[:, positions[game.away_id]] - 1,
        ]
        probability = np.clip(
            student_t.sf(
                (0.0 - locations) / likelihood.scale,
                likelihood.degrees_of_freedom,
            ),
            0.0,
            1.0,
        )
        locations_by_game[game.game_id] = np.asarray(locations, dtype=float)
        probabilities[game.game_id] = np.asarray(probability, dtype=float)

        mixture_locations, weights = predictive_components(game, home, away, likelihood)
        exact_cdf = mixture_cdf(
            0.0,
            mixture_locations,
            weights,
            likelihood.scale,
            likelihood.degrees_of_freedom,
        )
        exact_home, _exact_away = win_probabilities(exact_cdf)
        exact[game.game_id] = {
            "home_win_probability": float(exact_home),
            "expected_home_margin": float(np.dot(weights, mixture_locations)),
        }
    return probabilities, locations_by_game, exact


def _shared_game_outcomes(
    prepared: _PreparedSimulation,
    locations_by_game: Mapping[str, np.ndarray],
    likelihood: LikelihoodV1,
    rng: np.random.Generator,
    inner_rollout_count: int,
) -> dict[str, np.ndarray]:
    if inner_rollout_count <= 0:
        return {}
    result: dict[str, np.ndarray] = {}
    for game in prepared.supported_games:
        locations = locations_by_game[game.game_id]
        margins = locations[:, None] + likelihood.scale * rng.standard_t(
            likelihood.degrees_of_freedom,
            size=(len(locations), inner_rollout_count),
        )
        # There is no point mass at zero in Student-t V1.  A hypothetical tie
        # convention remains in the exact single-game predictor, but season
        # rollouts use the continuous game draw directly.
        result[game.game_id] = margins > 0.0
    return result


def _dependence_diagnostics(
    prepared: _PreparedSimulation,
    outcomes: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    if not outcomes:
        return {
            "method": "shared_game_result_per_outer_universe_and_inner_rollout",
            "status": "not_sampled; production win totals use exact conditional PMFs",
            "teams": {},
        }
    game_ids_by_team: dict[str, list[str]] = {team_id: [] for team_id in prepared.fbs_team_ids}
    for game in prepared.supported_games:
        for team_id in (game.home_id, game.away_id):
            if team_id in game_ids_by_team:
                game_ids_by_team[team_id].append(game.game_id)
    teams: dict[str, Any] = {}
    for team_id, game_ids in game_ids_by_team.items():
        pairs: dict[str, float] = {}
        for left_index, left_id in enumerate(game_ids):
            for right_id in game_ids[left_index + 1 :]:
                # Orient every result as a focal-team win before comparing;
                # the away side is the complement of the canonical home result.
                left_game = next(
                    game for game in prepared.supported_games if game.game_id == left_id
                )
                right_game = next(
                    game for game in prepared.supported_games if game.game_id == right_id
                )
                left = outcomes[left_id]
                right = outcomes[right_id]
                if left_game.away_id == team_id:
                    left = ~left
                if right_game.away_id == team_id:
                    right = ~right
                left_flat, right_flat = left.ravel(), right.ravel()
                if np.std(left_flat) == 0 or np.std(right_flat) == 0:
                    correlation = 0.0
                else:
                    correlation = float(np.corrcoef(left_flat, right_flat)[0, 1])
                pairs[f"{left_id}:{right_id}"] = correlation
        teams[team_id] = {
            "game_pair_count": len(pairs),
            "mean_pairwise_correlation": (
                float(np.mean(list(pairs.values()))) if pairs else 0.0
            ),
            "pairwise_correlations": pairs,
        }
    return {
        "method": "shared_game_result_per_outer_universe_and_inner_rollout",
        "status": "sampled",
        "teams": teams,
    }


def _team_summary(
    *,
    team: Team,
    record: CompletedRecord,
    candidate_games: Sequence[ScheduledGame],
    unsupported_games: Sequence[dict[str, Any]],
    team_probabilities: np.ndarray,
    outer_count: int,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "team_id": team.team_id,
        "team_name": team.name,
        "subdivision": str(team.subdivision).casefold(),
        "completed_wins": record.wins,
        "completed_losses": record.losses,
        "completed_ties": record.ties,
        "remaining_games": len(candidate_games),
        "simulated_remaining_games": int(team_probabilities.shape[1]),
        "forecast_status": "unavailable" if unsupported_games else "available",
    }
    if unsupported_games:
        base["forecast_unavailable_reason"] = "unsupported_future_game"
        base["unsupported_games"] = list(unsupported_games)
        return base

    conditional_pmfs = _vectorized_poisson_binomial(team_probabilities)
    remaining_pmf = np.mean(conditional_pmfs, axis=0)
    remaining_pmf = np.maximum(remaining_pmf, 0.0)
    remaining_pmf /= remaining_pmf.sum()
    final_wins = record.wins + np.arange(len(remaining_pmf))
    final_pmf = remaining_pmf
    conditional_remaining_means = team_probabilities.sum(axis=1)
    conditional_final_means = record.wins + conditional_remaining_means
    game_variances = np.sum(
        team_probabilities * (1.0 - team_probabilities), axis=1
    )
    quality_variance = float(np.var(conditional_final_means))
    game_variance = float(np.mean(game_variances))
    variance = _variance_decomposition(quality_variance, game_variance)
    max_wins = int(final_wins[-1])

    remaining_distribution = {
        str(index): float(probability)
        for index, probability in enumerate(remaining_pmf)
    }
    final_distribution = {
        str(int(wins)): float(probability)
        for wins, probability in zip(final_wins, final_pmf, strict=True)
    }
    records = {
        _record_key(int(wins), record.losses + len(remaining_pmf) - 1 - index, record.ties): float(
            probability
        )
        for index, (wins, probability) in enumerate(
            zip(final_wins, final_pmf, strict=True)
        )
    }
    thresholds: dict[str, float] = {}
    for threshold in SEASON_WIN_THRESHOLDS:
        if threshold <= max_wins:
            index = max(0, threshold - record.wins)
            thresholds[f"wins_{threshold}_plus"] = float(final_pmf[index:].sum())
    win_out = float(final_pmf[-1])
    lose_out = float(final_pmf[0])
    thresholds.update({"win_out": win_out, "lose_out": lose_out})
    events: dict[str, dict[str, Any]] = {}
    for threshold in SEASON_WIN_THRESHOLDS:
        if threshold > max_wins:
            continue
        name = f"wins_{threshold}_plus"
        index = max(0, threshold - record.wins)
        events[name] = _event_summary(conditional_pmfs[:, index:].sum(axis=1))
    events["win_out"] = _event_summary(np.prod(team_probabilities, axis=1))
    events["lose_out"] = _event_summary(np.prod(1.0 - team_probabilities, axis=1))

    conditional_summary = {
        "expected_remaining_wins": float(conditional_remaining_means.mean()),
        "expected_final_wins": float(conditional_final_means.mean()),
        "median": float(np.median(conditional_final_means)),
        "interval_50": _continuous_interval(conditional_final_means, 0.50),
        "interval_80": _continuous_interval(conditional_final_means, 0.80),
        "interval_95": _continuous_interval(conditional_final_means, 0.95),
        "remaining_median": float(np.median(conditional_remaining_means)),
        "remaining_interval_50": _continuous_interval(conditional_remaining_means, 0.50),
        "remaining_interval_80": _continuous_interval(conditional_remaining_means, 0.80),
        "remaining_interval_95": _continuous_interval(conditional_remaining_means, 0.95),
    }
    return {
        **base,
        "expected_final_wins": float(np.dot(final_wins, final_pmf)),
        "median_final_wins": _pmf_quantile(final_pmf, 0.50, record.wins),
        "final_win_interval_50": _pmf_interval(final_pmf, 0.50, record.wins),
        "final_win_interval_80": _pmf_interval(final_pmf, 0.80, record.wins),
        "final_win_interval_95": _pmf_interval(final_pmf, 0.95, record.wins),
        "expected_remaining_wins": float(remaining_pmf @ np.arange(len(remaining_pmf))),
        "median_remaining_wins": _pmf_quantile(remaining_pmf, 0.50),
        "remaining_win_interval_50": _pmf_interval(remaining_pmf, 0.50),
        "remaining_win_interval_80": _pmf_interval(remaining_pmf, 0.80),
        "remaining_win_interval_95": _pmf_interval(remaining_pmf, 0.95),
        "final_win_distribution": final_distribution,
        "remaining_win_distribution": remaining_distribution,
        "record_probabilities": records,
        "threshold_probabilities": thresholds,
        "conditional_expected_wins": conditional_summary,
        "variance_decomposition": variance,
        "event_probability_decomposition": events,
        "monte_carlo": {
            "outer_draw_count": outer_count,
            "conditional_expected_final_wins_standard_error": float(
                np.std(conditional_final_means, ddof=1) / np.sqrt(outer_count)
                if outer_count > 1
                else 0.0
            ),
        },
    }


def simulate_season(
    teams: Sequence[Team],
    posterior_pmfs: Mapping[str, np.ndarray],
    future_games: Sequence[ScheduledGame],
    completed_records: Mapping[str, object],
    likelihood: LikelihoodV1,
    *,
    config: SeasonSimulationConfig | None = None,
    provenance: Mapping[str, Any] | None = None,
    prediction_source: str | None = None,
) -> dict[str, Any]:
    """Build one canonical regular-season forecast artifact for a snapshot.

    Individual team win distributions are exact conditional
    Poisson-binomial mixtures over outer latent-quality universes.  If
    ``inner_rollout_count`` is positive, shared Student-t game outcomes are
    additionally sampled for marginal/dependence diagnostics; they are never
    used to update posterior beliefs or to redraw latent ranks.
    """

    config = config or SeasonSimulationConfig()
    if config.likelihood_version != "V1":
        raise ValueError("Season Simulation V1 requires Historical Likelihood V1")
    prepared = _prepare_simulation(teams, future_games)
    generator = np.random.default_rng(config.seed)
    latent = sample_latent_qualities(
        prepared.teams,
        posterior_pmfs,
        outer_draw_count=config.outer_draw_count,
        rng=generator,
    )
    probabilities, locations_by_game, exact_marginals = _conditional_game_samples(
        prepared, latent, posterior_pmfs, likelihood
    )
    outcomes = _shared_game_outcomes(
        prepared,
        locations_by_game,
        likelihood,
        generator,
        config.inner_rollout_count,
    )

    team_probabilities: dict[str, list[np.ndarray]] = {
        team_id: [] for team_id in prepared.fbs_team_ids
    }
    for game in prepared.supported_games:
        home_probability = probabilities[game.game_id]
        if game.home_id in team_probabilities:
            team_probabilities[game.home_id].append(home_probability)
        if game.away_id in team_probabilities:
            team_probabilities[game.away_id].append(1.0 - home_probability)

    summaries: dict[str, dict[str, Any]] = {}
    for team in prepared.teams:
        if str(team.subdivision).casefold() != "fbs":
            continue
        values = team_probabilities[team.team_id]
        conditional = (
            np.column_stack(values)
            if values
            else np.empty((config.outer_draw_count, 0), dtype=float)
        )
        summaries[team.team_id] = _team_summary(
            team=team,
            record=_completed_record(completed_records.get(team.team_id)),
            candidate_games=prepared.candidate_games_by_team[team.team_id],
            unsupported_games=prepared.unsupported_by_team[team.team_id],
            team_probabilities=conditional,
            outer_count=config.outer_draw_count,
        )

    game_marginals: dict[str, dict[str, Any]] = {}
    for game in prepared.supported_games:
        probabilities_for_game = probabilities[game.game_id]
        locations = locations_by_game[game.game_id]
        exact = exact_marginals[game.game_id]
        game_marginals[game.game_id] = {
            "game_id": game.game_id,
            "home_team_id": game.home_id,
            "away_team_id": game.away_id,
            "outer_draw_count": config.outer_draw_count,
            "exact_home_win_probability": exact["home_win_probability"],
            "outer_home_win_probability": float(probabilities_for_game.mean()),
            "home_win_probability_error": float(
                probabilities_for_game.mean() - exact["home_win_probability"]
            ),
            "outer_home_win_standard_error": float(
                np.std(probabilities_for_game, ddof=1) / np.sqrt(config.outer_draw_count)
                if config.outer_draw_count > 1
                else 0.0
            ),
            "exact_expected_home_margin": exact["expected_home_margin"],
            "outer_expected_home_margin": float(locations.mean()),
            "expected_home_margin_error": float(
                locations.mean() - exact["expected_home_margin"]
            ),
        }
        if game.game_id in outcomes:
            frequency = float(outcomes[game.game_id].mean())
            game_marginals[game.game_id].update(
                {
                    "inner_home_win_probability": frequency,
                    "inner_home_win_probability_error": frequency
                    - exact["home_win_probability"],
                    "inner_rollout_count": config.inner_rollout_count,
                }
            )

    candidate_ids = [
        game.game_id
        for games in prepared.candidate_games_by_team.values()
        for game in games
    ]
    candidate_ids = sorted(set(candidate_ids))
    supported_ids = [game.game_id for game in prepared.supported_games]
    unavailable_team_ids = [
        team_id for team_id, games in prepared.unsupported_by_team.items() if games
    ]
    return {
        "schema_version": SEASON_SIMULATION_SCHEMA_VERSION,
        "artifact_kind": "season_simulation",
        "simulation_version": config.simulation_version,
        "forecast_status": "partial" if unavailable_team_ids else "available",
        "prediction_source": prediction_source,
        "provenance": dict(provenance or {}),
        "configuration": config.as_dict(),
        "season_scope": {
            "season_type": "regular",
            "future_rule": "strictly after selected snapshot cutoff",
            "completed_games_are_fixed": True,
            "future_game_count": len(candidate_ids),
            "supported_future_game_count": len(supported_ids),
            "future_game_ids": candidate_ids,
            "supported_future_game_ids": sorted(supported_ids),
            "unsupported_future_games": list(prepared.unsupported_games),
            "unsupported_team_ids": unavailable_team_ids,
            "unsupported_behavior": "fail_closed",
        },
        "latent_quality": {
            "draw_unit": "one rank per modeled team per outer universe",
            "held_fixed_throughout_universe": True,
            "posterior_sampling": "independent marginal PMFs",
            "joint_posterior_approximation": (
                "marginal-independence V1 approximation; sampled ranks need not be a permutation"
            ),
        },
        "conditional_game_model": {
            "likelihood_version": config.likelihood_version,
            "distribution": "Student-t residual conditional on fixed latent ranks",
            "games_conditionally_independent": True,
            "shared_game_result": (
                "one canonical game outcome is shared by home and away in each rollout"
            ),
            "posterior_updates_from_simulated_games": False,
        },
        "teams": summaries,
        "game_marginals": game_marginals,
        "cross_game_dependence": _dependence_diagnostics(prepared, outcomes),
        "validation": {
            "single_game_marginal_equivalence": {
                "game_count": len(game_marginals),
                "max_abs_home_win_probability_error": max(
                    (
                        abs(float(item["home_win_probability_error"]))
                        for item in game_marginals.values()
                    ),
                    default=0.0,
                ),
                "max_abs_expected_home_margin_error": max(
                    (
                        abs(float(item["expected_home_margin_error"]))
                        for item in game_marginals.values()
                    ),
                    default=0.0,
                ),
                "interpretation": "outer integration error; exact values are the existing V1 mixture",
            },
            "variance_decomposition": {
                "law": "Var(W)=Var_r(E[W|r])+E_r(Var(W|r))",
                "conditional_game_method": "exact Poisson-binomial",
            },
        },
        "monte_carlo": {
            "outer_draw_count": config.outer_draw_count,
            "inner_rollout_count": config.inner_rollout_count,
            "seed": config.seed,
            "inner_rollouts_used_for": (
                "shared game/dependence diagnostics only"
                if config.inner_rollout_count
                else "none; exact conditional win-total PMFs used"
            ),
        },
    }


def simulate_shared_game_outcomes(
    teams: Sequence[Team],
    posterior_pmfs: Mapping[str, np.ndarray],
    future_games: Sequence[ScheduledGame],
    likelihood: LikelihoodV1,
    *,
    outer_draw_count: int = DEFAULT_OUTER_DRAW_COUNT,
    inner_rollout_count: int = 100,
    seed: int = DEFAULT_SEASON_SIMULATION_SEED,
) -> tuple[LatentQualitySamples, dict[str, np.ndarray]]:
    """Return shared game-result matrices for dependence/invariant tests.

    Each matrix is shaped ``(outer_universe, inner_rollout)`` and contains
    the canonical home-win result.  The away-team result for the same cell is
    its complement; callers must not draw a second result for that game.
    Unsupported games raise rather than receiving an invented probability.
    """

    if inner_rollout_count < 1:
        raise ValueError("shared game outcomes need at least one inner rollout")
    prepared = _prepare_simulation(teams, future_games)
    if prepared.unsupported_games:
        raise ValueError(
            "cannot simulate unsupported future games: "
            + ", ".join(item["game_id"] for item in prepared.unsupported_games)
        )
    generator = np.random.default_rng(seed)
    latent = sample_latent_qualities(
        prepared.teams,
        posterior_pmfs,
        outer_draw_count=outer_draw_count,
        rng=generator,
    )
    _probabilities, locations, _exact = _conditional_game_samples(
        prepared, latent, posterior_pmfs, likelihood
    )
    return latent, _shared_game_outcomes(
        prepared,
        locations,
        likelihood,
        generator,
        inner_rollout_count,
    )
