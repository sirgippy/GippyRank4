"""Deterministic loopy belief propagation for the frozen margin likelihood.

The latent variable for each team is its *final ordinal rank*, rather than a
separate strength rating.  A game is a pairwise factor over the two teams'
rank coordinates.  Sum-product messages therefore let later evidence about an
opponent alter the interpretation of an earlier result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import gammaln

from gippyrank.modeling import design_matrix

Subdivision = Literal["fbs", "fcs"]
_GAME_FACTOR_CACHE: dict[tuple[object, ...], np.ndarray] = {}


@dataclass(frozen=True)
class Team:
    """A ranked team's discrete preseason marginal distribution."""

    team_id: str
    name: str
    subdivision: Subdivision
    prior: np.ndarray

    def __post_init__(self) -> None:
        prior = np.asarray(self.prior, dtype=float)
        if len(prior) == 0 or np.any(prior < 0) or not np.isfinite(prior).all():
            raise ValueError("prior must be a finite, non-negative nonempty PMF")
        if not np.isclose(prior.sum(), 1.0, atol=1e-8):
            raise ValueError("prior must sum to one")
        object.__setattr__(self, "prior", prior / prior.sum())


@dataclass(frozen=True)
class Game:
    """A completed game, oriented exactly as the frozen likelihood expects."""

    game_id: str
    home_id: str
    away_id: str
    home_subdivision: Subdivision
    away_subdivision: Subdivision
    home_points: int
    away_points: int
    neutral_site: bool = False


@dataclass(frozen=True)
class LikelihoodV1:
    """The serialized Historical Likelihood V1 surface parameters."""

    beta: np.ndarray
    scale: float
    degrees_of_freedom: float
    fit_kind: str = "weighted_pseudo"

    def __post_init__(self) -> None:
        beta = np.asarray(self.beta, dtype=float)
        if beta.ndim != 1 or not len(beta) or not np.isfinite(beta).all():
            raise ValueError("beta must be a finite vector")
        if self.scale <= 0 or self.degrees_of_freedom <= 0:
            raise ValueError("Student-t scale and df must be positive")
        object.__setattr__(self, "beta", beta)


@dataclass(frozen=True)
class PosteriorResult:
    pmfs: dict[str, np.ndarray]
    converged: bool
    iterations: int
    max_message_delta: float
    objective: float
    raw_game_factor_count: int
    unique_pair_factor_count: int
    max_team_degree: int
    warnings: tuple[str, ...] = ()
    bp_state: BeliefPropagationState | None = None

    @property
    def state(self) -> BeliefPropagationState | None:
        """Compatibility alias for callers that refer to the BP state plainly."""
        return self.bp_state


@dataclass(frozen=True)
class PairFactor:
    """A numerically stabilized product of games between one pair of teams."""

    first_id: str
    second_id: str
    values: np.ndarray
    game_ids: tuple[str, ...]


@dataclass(frozen=True)
class BeliefPropagationState:
    """Read-only converged state used by explanatory downstream features.

    ``cavity_beliefs[(factor_index, team_id)]`` is the belief for ``team_id``
    with the whole grouped pair factor removed. Keeping this state alongside
    the posterior makes it possible to interpret one member of a rematch
    without accidentally using the other member to establish opponent quality.
    """

    factors: tuple[PairFactor, ...]
    messages: dict[tuple[int, str], np.ndarray]
    cavity_beliefs: dict[tuple[int, str], np.ndarray]
    pair_indices: dict[tuple[str, str], int]

    def pair_cavity(
        self, first_id: str, second_id: str, team_id: str
    ) -> np.ndarray:
        """Return the normalized cavity belief for a grouped team pair."""
        key = tuple(sorted((first_id, second_id)))
        try:
            index = self.pair_indices[key]
            return self.cavity_beliefs[index, team_id].copy()
        except KeyError as error:
            raise KeyError(f"No cavity belief for pair {key} and team {team_id}") from error


def _student_t_logpdf(
    value: float, location: np.ndarray, scale: float, df: float
) -> np.ndarray:
    z = (value - location) / scale
    constant = (
        gammaln((df + 1) / 2)
        - gammaln(df / 2)
        - 0.5 * np.log(df * np.pi)
        - np.log(scale)
    )
    return constant - (df + 1) / 2 * np.log1p(z * z / df)


def game_factor(
    game: Game, home: Team, away: Team, likelihood: LikelihoodV1
) -> np.ndarray:
    """Return p(observed margin | home rank, away rank), with V1 orientation.

    Same-subdivision outcomes are home-minus-away. Cross-subdivision outcomes
    are always FBS-minus-FCS, matching ``build_historical_modeling.py``.
    """
    home_rank = np.arange(1, len(home.prior) + 1, dtype=float)
    away_rank = np.arange(1, len(away.prior) + 1, dtype=float)
    hp = (home_rank - 0.5) / len(home_rank)
    ap = (away_rank - 0.5) / len(away_rank)
    cross = home.subdivision != away.subdivision
    if cross and home.subdivision == "fcs":
        x, y = np.meshgrid(ap, hp, indexing="ij")
        margin = game.away_points - game.home_points
    else:
        x, y = np.meshgrid(hp, ap, indexing="ij")
        margin = game.home_points - game.away_points
    pairing = "fbs-fcs" if cross else f"{home.subdivision}-{away.subdivision}"
    neutral = float(game.neutral_site)
    fbs_home = float(cross and home.subdivision == "fbs" and not game.neutral_site)
    matrix = design_matrix(
        x.ravel(),
        y.ravel(),
        np.full(x.size, pairing),
        np.full(x.size, 1.0 - neutral),
        np.full(x.size, neutral),
        surface=True,
        fbs_home=np.full(x.size, fbs_home),
    )
    if len(likelihood.beta) != matrix.shape[1]:
        raise ValueError(
            f"Likelihood V1 beta has {len(likelihood.beta)} coefficients; "
            f"surface requires {matrix.shape[1]}"
        )
    factor = np.exp(
        _student_t_logpdf(
            margin,
            matrix @ likelihood.beta,
            likelihood.scale,
            likelihood.degrees_of_freedom,
        )
    )
    # A common positive multiplier does not affect BP and avoids underflow.
    factor /= factor.max()
    if cross and home.subdivision == "fcs":
        return factor.reshape(len(away_rank), len(home_rank)).T
    return factor.reshape(len(home_rank), len(away_rank))


def game_margin_parameters(
    game: Game, home: Team, away: Team, likelihood: LikelihoodV1
) -> tuple[np.ndarray, float]:
    """Return raw V1 Student-t locations and the observed oriented margin.

    The factor used by BP is intentionally rescaled by a common positive
    multiplier.  Research-side predictive scoring needs the unscaled density,
    so this helper exposes the same design semantics without changing the BP
    factor contract.  The returned matrix is always indexed as home rank by
    away rank, even for a cross-subdivision game whose model coordinates are
    always FBS first and FCS second.
    """
    home_rank = np.arange(1, len(home.prior) + 1, dtype=float)
    away_rank = np.arange(1, len(away.prior) + 1, dtype=float)
    hp = (home_rank - 0.5) / len(home_rank)
    ap = (away_rank - 0.5) / len(away_rank)
    cross = home.subdivision != away.subdivision
    if cross and home.subdivision == "fcs":
        x, y = np.meshgrid(ap, hp, indexing="ij")
        margin = float(game.away_points - game.home_points)
    else:
        x, y = np.meshgrid(hp, ap, indexing="ij")
        margin = float(game.home_points - game.away_points)
    pairing = "fbs-fcs" if cross else f"{home.subdivision}-{away.subdivision}"
    neutral = float(game.neutral_site)
    fbs_home = float(cross and home.subdivision == "fbs" and not game.neutral_site)
    matrix = design_matrix(
        x.ravel(),
        y.ravel(),
        np.full(x.size, pairing),
        np.full(x.size, 1.0 - neutral),
        np.full(x.size, neutral),
        surface=True,
        fbs_home=np.full(x.size, fbs_home),
    )
    if len(likelihood.beta) != matrix.shape[1]:
        raise ValueError(
            f"Likelihood V1 beta has {len(likelihood.beta)} coefficients; "
            f"surface requires {matrix.shape[1]}"
        )
    locations = matrix @ likelihood.beta
    if cross and home.subdivision == "fcs":
        locations = locations.reshape(len(away_rank), len(home_rank)).T
    else:
        locations = locations.reshape(len(home_rank), len(away_rank))
    return locations, margin


def game_margin_density(
    game: Game, home: Team, away: Team, likelihood: LikelihoodV1
) -> tuple[np.ndarray, float]:
    """Return the unscaled frozen V1 density at the observed oriented margin."""
    _locations, density, margin = game_margin_surface(game, home, away, likelihood)
    return density, margin


def game_margin_surface(
    game: Game, home: Team, away: Team, likelihood: LikelihoodV1
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return V1 locations, observed margin, and raw density in one pass."""
    locations, margin = game_margin_parameters(game, home, away, likelihood)
    return (
        locations,
        np.exp(
            _student_t_logpdf(
                margin,
                locations,
                likelihood.scale,
                likelihood.degrees_of_freedom,
            )
        ),
        margin,
    )


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 0.0)
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("zero or non-finite belief-propagation message")
    return values / total


def infer_posterior(
    teams: list[Team],
    games: list[Game],
    likelihood: LikelihoodV1,
    *,
    factor_weights: dict[str, float] | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-9,
    damping: float = 0.35,
) -> PosteriorResult:
    """Approximate marginal posterior with deterministic, damped sum-product BP.

    Trees are exact. Cyclic schedules use the usual loopy approximation; a
    non-converged run is returned as invalid so the snapshot layer cannot mark
    it publishable.
    """
    by_id = {team.team_id: team for team in teams}
    if len(by_id) != len(teams):
        raise ValueError("team IDs must be unique")
    if factor_weights is not None:
        unknown = set(factor_weights) - {game.game_id for game in games}
        if unknown:
            raise ValueError(f"factor weights reference unknown games: {sorted(unknown)}")
        if any(
            not np.isfinite(weight) or weight <= 0
            for weight in factor_weights.values()
        ):
            raise ValueError("factor weights must be finite and strictly positive")
    grouped: dict[tuple[str, str], tuple[np.ndarray, list[str]]] = {}
    # Canonical orientation means rematches do not introduce a trivial
    # two-variable cycle. Individual IDs remain available for provenance.
    for game in games:
        if game.home_id not in by_id or game.away_id not in by_id:
            raise ValueError(f"game {game.game_id} references a team without a prior")
        if game.home_id == game.away_id:
            raise ValueError("a game cannot have the same team twice")
        key = tuple(sorted((game.home_id, game.away_id)))
        home, away = by_id[game.home_id], by_id[game.away_id]
        cache_key = (
            likelihood.beta.tobytes(),
            likelihood.scale,
            likelihood.degrees_of_freedom,
            home.subdivision,
            away.subdivision,
            len(home.prior),
            len(away.prior),
            game.home_points,
            game.away_points,
            game.neutral_site,
        )
        values = _GAME_FACTOR_CACHE.get(cache_key)
        if values is None:
            values = game_factor(game, home, away, likelihood)
            _GAME_FACTOR_CACHE[cache_key] = values
        if (game.home_id, game.away_id) != key:
            values = values.T
        log_values = np.log(np.maximum(values, np.finfo(float).tiny))
        if factor_weights is not None:
            log_values *= factor_weights.get(game.game_id, 1.0)
        if key in grouped:
            prior_log_values, game_ids = grouped[key]
            grouped[key] = (prior_log_values + log_values, [*game_ids, game.game_id])
        else:
            grouped[key] = (log_values, [game.game_id])
    factors = [
        PairFactor(
            first, second, np.exp(log_values - log_values.max()), tuple(game_ids)
        )
        for (first, second), (log_values, game_ids) in sorted(grouped.items())
    ]
    adjacency: dict[str, list[int]] = {team.team_id: [] for team in teams}
    for index, factor in enumerate(factors):
        adjacency[factor.first_id].append(index)
        adjacency[factor.second_id].append(index)
    messages: dict[tuple[int, str], np.ndarray] = {}
    for index, factor in enumerate(factors):
        messages[index, factor.first_id] = np.ones(len(by_id[factor.first_id].prior))
        messages[index, factor.second_id] = np.ones(len(by_id[factor.second_id].prior))
    if not factors:
        state = BeliefPropagationState((), {}, {}, {})
        return PosteriorResult(
            {team.team_id: team.prior.copy() for team in teams},
            True,
            0,
            0.0,
            0.0,
            0,
            0,
            0,
            bp_state=state,
        )

    max_delta = np.inf
    for iteration in range(1, max_iterations + 1):
        updates: dict[tuple[int, str], np.ndarray] = {}
        max_delta = 0.0
        for index, factor in enumerate(factors):
            for target, other, transpose in (
                (factor.first_id, factor.second_id, False),
                (factor.second_id, factor.first_id, True),
            ):
                belief = by_id[other].prior.copy()
                for neighbor in adjacency[other]:
                    if neighbor != index:
                        belief *= messages[neighbor, other]
                belief = _normalise(belief)
                raw = (
                    (factor.values.T @ belief)
                    if transpose
                    else (factor.values @ belief)
                )
                proposal = _normalise(raw)
                old = messages[index, target]
                update = _normalise(damping * old + (1.0 - damping) * proposal)
                max_delta = max(max_delta, float(np.max(np.abs(update - old))))
                updates[index, target] = update
        messages.update(updates)
        if max_delta <= tolerance:
            break
    cavity_beliefs: dict[tuple[int, str], np.ndarray] = {}
    for index, factor in enumerate(factors):
        for team_id in (factor.first_id, factor.second_id):
            belief = by_id[team_id].prior.copy()
            for neighbor in adjacency[team_id]:
                if neighbor != index:
                    belief *= messages[neighbor, team_id]
            cavity_beliefs[index, team_id] = _normalise(belief)

    pmfs = {}
    for team in teams:
        belief = team.prior.copy()
        for index in adjacency[team.team_id]:
            belief *= messages[index, team.team_id]
        pmfs[team.team_id] = _normalise(belief)
    # A finite surrogate objective useful for diagnostics, not a Bethe free energy.
    objective = float(sum(np.log(pmf.max()) for pmf in pmfs.values()))
    state = BeliefPropagationState(
        tuple(factors),
        {key: value.copy() for key, value in messages.items()},
        {key: value.copy() for key, value in cavity_beliefs.items()},
        {
            (factor.first_id, factor.second_id): index
            for index, factor in enumerate(factors)
        },
    )
    return PosteriorResult(
        pmfs,
        max_delta <= tolerance,
        iteration,
        max_delta,
        objective,
        len(games),
        len(factors),
        max(map(len, adjacency.values()), default=0),
        bp_state=state,
    )


def game_evidence_pmf(
    game: Game,
    focal_team_id: str,
    teams: list[Team] | tuple[Team, ...],
    likelihood: LikelihoodV1,
    posterior: PosteriorResult,
) -> np.ndarray:
    """Return a single game's normalized factor-to-focal-team evidence.

    The opponent belief comes from the grouped pair cavity in ``posterior``.
    Therefore every rematch shares a cavity that excludes the entire grouped
    head-to-head factor, while each individual game likelihood is applied
    separately. The focal team's prior is never multiplied into this PMF.
    """
    state = posterior.bp_state
    if state is None:
        raise ValueError("posterior does not expose BP cavity state")
    by_id = {team.team_id: team for team in teams}
    if game.home_id not in by_id or game.away_id not in by_id:
        raise ValueError(f"game {game.game_id} references a team without a prior")
    if focal_team_id not in {game.home_id, game.away_id}:
        raise ValueError("focal team must participate in the game")
    key = tuple(sorted((game.home_id, game.away_id)))
    factor_index = state.pair_indices.get(key)
    if factor_index is None:
        raise ValueError(f"game {game.game_id} is absent from the posterior pair factors")
    factor = state.factors[factor_index]
    if game.game_id not in factor.game_ids:
        raise ValueError(f"game {game.game_id} is absent from its posterior pair factor")
    opponent_id = game.away_id if focal_team_id == game.home_id else game.home_id
    opponent_cavity = state.cavity_beliefs[factor_index, opponent_id]
    values = game_factor(game, by_id[game.home_id], by_id[game.away_id], likelihood)
    raw = (
        values @ opponent_cavity
        if focal_team_id == game.home_id
        else values.T @ opponent_cavity
    )
    return _normalise(raw)
