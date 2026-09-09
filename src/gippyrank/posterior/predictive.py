"""Exact posterior-predictive margin summaries for scheduled games.

The production Historical Likelihood V1 is a Student-t margin model whose
location is a function of the two ordinal-rank coordinates, the pairing, and
the site.  A scheduled game's predictive distribution is therefore a finite
mixture of those Student-t components, weighted by the two teams' posterior
PMFs.  This module keeps that calculation separate from the static-site
exporter so the same canonical result can be used by publication and tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import gammaln, stdtr
from scipy.stats import t as student_t

from gippyrank.modeling import design_matrix
from gippyrank.posterior.display import (
    DISPLAY_PROBABILITY_SCALE,
    quantize_display_probabilities,
)
from gippyrank.posterior.engine import LikelihoodV1, Team

PREDICTION_SCHEMA_VERSION = "1.0"
PREDICTION_SOURCE_CONTEXT = "predictive_context"
PREDICTION_SOURCE_HISTORY = "predictive_history"
FUTURE_MARGIN_DISPLAY_MIN = -40.0
FUTURE_MARGIN_DISPLAY_MAX = 40.0
FUTURE_MARGIN_DISPLAY_BINS = 40
FUTURE_MARGIN_DISPLAY_COMPONENT_BINS = 256
_LOCATION_SURFACE_CACHE: dict[tuple[object, ...], np.ndarray] = {}


@dataclass(frozen=True)
class ScheduledGame:
    """A scheduled matchup without an observed score."""

    game_id: str
    home_id: str
    away_id: str
    home_subdivision: str
    away_subdivision: str
    neutral_site: bool = False
    season_type: str = "regular"
    date: str | None = None


@dataclass(frozen=True)
class PredictiveMarginSummary:
    """Compact canonical summaries of a home-oriented margin mixture."""

    expected_home_margin: float
    median_home_margin: float
    home_win_probability: float
    away_win_probability: float
    tie_probability: float
    margin_interval_50: tuple[float, float]
    margin_interval_80: tuple[float, float]
    margin_interval_95: tuple[float, float]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready summary with the canonical home orientation."""

        return {
            "expected_home_margin": self.expected_home_margin,
            "median_home_margin": self.median_home_margin,
            "home_win_probability": self.home_win_probability,
            "away_win_probability": self.away_win_probability,
            "tie_probability": self.tie_probability,
            "margin_interval_50": list(self.margin_interval_50),
            "margin_interval_80": list(self.margin_interval_80),
            "margin_interval_95": list(self.margin_interval_95),
        }


def _pairing(home_subdivision: str, away_subdivision: str) -> tuple[str, bool]:
    home = home_subdivision.casefold()
    away = away_subdivision.casefold()
    if home not in {"fbs", "fcs"} or away not in {"fbs", "fcs"}:
        raise ValueError(
            f"unsupported subdivisions: {home_subdivision}, {away_subdivision}"
        )
    cross = home != away
    return ("fbs-fcs" if cross else f"{home}-{away}"), cross


def posterior_prediction_team(
    team: Team, posterior_pmfs: Mapping[str, np.ndarray]
) -> Team:
    """Return prediction metadata paired with an explicit posterior PMF.

    ``Team.prior`` stores whichever rank PMF was used to construct a snapshot,
    so prediction callers must replace it explicitly with the snapshot
    posterior rather than relying on the field's historical name.
    """

    try:
        posterior_pmf = posterior_pmfs[team.team_id]
    except KeyError as error:
        raise KeyError(f"posterior PMF is missing team {team.team_id}") from error
    return Team(team.team_id, team.name, team.subdivision, posterior_pmf)


def posterior_prediction_teams(
    teams: Sequence[Team], posterior_pmfs: Mapping[str, np.ndarray]
) -> dict[str, Team]:
    """Build prediction-side teams with metadata and posterior PMFs."""

    return {
        team.team_id: posterior_prediction_team(team, posterior_pmfs) for team in teams
    }


def _location_surface(
    game: ScheduledGame,
    home: Team,
    away: Team,
    likelihood: LikelihoodV1,
) -> np.ndarray:
    """Return V1 locations indexed by home rank, then away rank."""

    if home.team_id != game.home_id or away.team_id != game.away_id:
        raise ValueError("team objects do not match the scheduled game orientation")
    if (
        home.subdivision != game.home_subdivision
        or away.subdivision != game.away_subdivision
    ):
        raise ValueError("team subdivisions do not match the scheduled game")

    cache_key = (
        len(home.prior),
        len(away.prior),
        game.home_subdivision.casefold(),
        game.away_subdivision.casefold(),
        bool(game.neutral_site),
        likelihood.beta.tobytes(),
    )
    cached = _LOCATION_SURFACE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    home_percentiles = (np.arange(1, len(home.prior) + 1, dtype=float) - 0.5) / len(
        home.prior
    )
    away_percentiles = (np.arange(1, len(away.prior) + 1, dtype=float) - 0.5) / len(
        away.prior
    )
    pairing, cross = _pairing(game.home_subdivision, game.away_subdivision)
    neutral = float(game.neutral_site)
    home_non_neutral = 1.0 - neutral
    fbs_home = float(
        cross and game.home_subdivision.casefold() == "fbs" and not game.neutral_site
    )

    if cross and game.home_subdivision.casefold() == "fcs":
        # Historical Likelihood V1 always receives FBS coordinates first for
        # cross-subdivision games, but the returned surface remains home×away.
        x, y = np.meshgrid(away_percentiles, home_percentiles, indexing="ij")
    else:
        x, y = np.meshgrid(home_percentiles, away_percentiles, indexing="ij")

    design = design_matrix(
        x.ravel(),
        y.ravel(),
        np.full(x.size, pairing),
        np.full(x.size, home_non_neutral),
        np.full(x.size, neutral),
        surface=True,
        fbs_home=np.full(x.size, fbs_home),
    )
    if len(likelihood.beta) != design.shape[1]:
        raise ValueError(
            f"Likelihood V1 beta has {len(likelihood.beta)} coefficients; "
            f"surface requires {design.shape[1]}"
        )
    locations = design @ likelihood.beta
    if cross and game.home_subdivision.casefold() == "fcs":
        # Historical Likelihood V1's cross-subdivision response is FBS-minus-
        # FCS.  The published predictive margin is home-minus-away, so an FCS
        # home team reverses that model-oriented coordinate.
        result = -locations.reshape(len(away.prior), len(home.prior)).T
    else:
        result = locations.reshape(len(home.prior), len(away.prior))
    result.setflags(write=False)
    _LOCATION_SURFACE_CACHE[cache_key] = result
    return result


def predictive_components(
    game: ScheduledGame,
    home: Team,
    away: Team,
    likelihood: LikelihoodV1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return component locations and rank-PMF mixture weights.

    The returned arrays are flattened in home-rank/away-rank order.  No
    expected-rank plug-in occurs: every supported pair of posterior rank
    states contributes its own Historical Likelihood component.
    """

    locations = _location_surface(game, home, away, likelihood).ravel()
    weights = np.multiply.outer(home.prior, away.prior).ravel()
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            "posterior rank PMFs must produce positive finite mixture weight"
        )
    return locations, weights / total


def conditional_margin_location_surface(
    game: ScheduledGame,
    home: Team,
    away: Team,
    likelihood: LikelihoodV1,
) -> np.ndarray:
    """Return the frozen V1 home-margin surface for fixed rank states.

    The matrix is indexed by the canonical home and away rank coordinates.
    Unlike :func:`predictive_components`, this helper does not average over
    either team's posterior PMF.  It is the primitive used by season
    simulation after one latent rank has been sampled for each team.
    """

    return _location_surface(game, home, away, likelihood)


def conditional_margin_location(
    game: ScheduledGame,
    home: Team,
    away: Team,
    likelihood: LikelihoodV1,
    home_rank: int,
    away_rank: int,
) -> float:
    """Return the V1 expected home margin for one fixed rank pair."""

    if isinstance(home_rank, bool) or isinstance(away_rank, bool):
        raise TypeError("rank states must be integers")
    locations = conditional_margin_location_surface(game, home, away, likelihood)
    if not 1 <= home_rank <= locations.shape[0] or not 1 <= away_rank <= locations.shape[1]:
        raise ValueError("rank states are outside the teams' supported coordinates")
    return float(locations[home_rank - 1, away_rank - 1])


def conditional_home_win_probability(
    game: ScheduledGame,
    home: Team,
    away: Team,
    likelihood: LikelihoodV1,
    home_rank: int,
    away_rank: int,
) -> float:
    """Return ``P(home wins | fixed latent ranks)`` under Historical V1."""

    location = conditional_margin_location(
        game, home, away, likelihood, home_rank, away_rank
    )
    return float(
        student_t.sf(
            (0.0 - location) / likelihood.scale,
            likelihood.degrees_of_freedom,
        )
    )


def mixture_cdf(
    value: float,
    locations: np.ndarray,
    weights: np.ndarray,
    scale: float,
    degrees_of_freedom: float,
) -> float:
    """Evaluate the exact finite Student-t location-mixture CDF."""

    if scale <= 0 or degrees_of_freedom <= 0:
        raise ValueError("Student-t scale and degrees of freedom must be positive")
    locations = np.asarray(locations, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if locations.ndim != 1 or weights.ndim != 1 or len(locations) != len(weights):
        raise ValueError("mixture locations and weights must be equally sized vectors")
    if (
        not np.isfinite(locations).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError(
            "mixture locations and weights must be finite and non-negative"
        )
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("mixture weights must have positive mass")
    return float(
        np.dot(
            weights / total,
            stdtr(degrees_of_freedom, (float(value) - locations) / scale),
        )
    )


def mixture_quantile(
    probability: float,
    locations: np.ndarray,
    weights: np.ndarray,
    scale: float,
    degrees_of_freedom: float,
) -> float:
    """Find a deterministic quantile of the finite Student-t mixture."""

    if not 0 < probability < 1:
        raise ValueError(
            "mixture quantile probability must be strictly between zero and one"
        )
    if scale <= 0 or degrees_of_freedom <= 0:
        raise ValueError("Student-t scale and degrees of freedom must be positive")
    locations = np.asarray(locations, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if locations.ndim != 1 or not len(locations):
        raise ValueError("mixture locations must be a nonempty vector")
    if weights.ndim != 1 or len(weights) != len(locations):
        raise ValueError("mixture locations and weights must be equally sized vectors")
    if (
        not np.isfinite(locations).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError(
            "mixture locations and weights must be finite and non-negative"
        )
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("mixture weights must have positive mass")
    weights = weights / total
    tail = max(float(student_t.ppf(1.0 - 1.0e-12, degrees_of_freedom)), 32.0)
    span = max(float(scale) * tail, 1.0)
    lower = float(locations.min() - span)
    upper = float(locations.max() + span)
    log_constant = (
        gammaln((degrees_of_freedom + 1.0) / 2.0)
        - gammaln(degrees_of_freedom / 2.0)
        - 0.5 * np.log(degrees_of_freedom * np.pi)
        - np.log(scale)
    )

    # The bracket is wider than the numerical tail used by the CDF, so every
    # component has negligible mass outside it and no per-quantile bracketing
    # evaluations are needed.
    def cdf_and_density(value: float) -> tuple[float, float]:
        standardized = (float(value) - locations) / scale
        cdf = float(np.dot(weights, stdtr(degrees_of_freedom, standardized)))
        density = float(
            np.dot(
                weights,
                np.exp(
                    log_constant
                    - (degrees_of_freedom + 1.0)
                    / 2.0
                    * np.log1p(standardized * standardized / degrees_of_freedom)
                ),
            )
        )
        return cdf, density

    # A safeguarded Newton solve uses the exact mixture CDF and its analytic
    # Student-t density.  Bisection remains the fallback, so the result is
    # deterministic and cannot leave the bracket even for a flat tail.
    location_mean = float(np.dot(weights, locations))
    location_variance = float(np.dot(weights, (locations - location_mean) ** 2))
    effective_scale = float(scale)
    if degrees_of_freedom > 2.0:
        effective_scale = float(
            np.sqrt(
                scale * scale
                + location_variance * (degrees_of_freedom - 2.0) / degrees_of_freedom
            )
        )
    value = float(
        location_mean + effective_scale * student_t.ppf(probability, degrees_of_freedom)
    )
    value = min(max(value, lower), upper)
    for _ in range(64):
        cdf, density = cdf_and_density(value)
        if abs(cdf - probability) <= 1.0e-12:
            return value
        if cdf < probability:
            lower = value
        else:
            upper = value
        proposal = value - (cdf - probability) / density if density > 0 else np.nan
        value = (
            float(proposal)
            if np.isfinite(proposal) and lower < proposal < upper
            else (lower + upper) / 2.0
        )
        if upper - lower <= 1.0e-10:
            return (lower + upper) / 2.0
    raise FloatingPointError("predictive mixture quantile did not converge")


def _display_components(
    locations: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate a large exact mixture into deterministic display components."""
    if len(locations) <= FUTURE_MARGIN_DISPLAY_COMPONENT_BINS:
        return locations, weights
    location_min = float(locations.min())
    location_max = float(locations.max())
    location_span = location_max - location_min
    if location_span <= 0:
        return locations, weights
    component_bins = np.minimum(
        (
            (locations - location_min)
            / location_span
            * FUTURE_MARGIN_DISPLAY_COMPONENT_BINS
        ).astype(int),
        FUTURE_MARGIN_DISPLAY_COMPONENT_BINS - 1,
    )
    component_weights = np.bincount(
        component_bins,
        weights=weights,
        minlength=FUTURE_MARGIN_DISPLAY_COMPONENT_BINS,
    )
    component_locations = np.bincount(
        component_bins,
        weights=weights * locations,
        minlength=FUTURE_MARGIN_DISPLAY_COMPONENT_BINS,
    )
    populated = component_weights > 0
    return (
        component_locations[populated] / component_weights[populated],
        component_weights[populated],
    )


def _display_cdf(
    edges: np.ndarray,
    locations: np.ndarray,
    weights: np.ndarray,
    likelihood: LikelihoodV1,
) -> np.ndarray:
    """Evaluate a vector of CDF edges for a normalized location mixture."""
    standardized = (edges[:, None] - locations[None, :]) / likelihood.scale
    return np.asarray(
        np.dot(
            stdtr(likelihood.degrees_of_freedom, standardized),
            weights,
        ),
        dtype=float,
    )


def margin_display_approximation_metrics(
    game: ScheduledGame,
    home: Team,
    away: Team,
    likelihood: LikelihoodV1,
    *,
    minimum: float = FUTURE_MARGIN_DISPLAY_MIN,
    maximum: float = FUTURE_MARGIN_DISPLAY_MAX,
    bins: int = FUTURE_MARGIN_DISPLAY_BINS,
) -> dict[str, Any]:
    """Compare exact and display-approximation CDFs at fixed chart edges."""
    if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum >= maximum:
        raise ValueError("margin display bounds must be finite and ordered")
    if bins < 1:
        raise ValueError("margin display needs at least one bin")
    locations, weights = predictive_components(game, home, away, likelihood)
    edges = np.linspace(float(minimum), float(maximum), bins + 1)
    display_locations, display_weights = _display_components(locations, weights)
    exact_cdf = _display_cdf(edges, locations, weights, likelihood)
    display_cdf = _display_cdf(
        edges, display_locations, display_weights, likelihood
    )
    exact_masses = np.diff(exact_cdf)
    display_masses = np.diff(display_cdf)
    cdf_error = np.abs(exact_cdf - display_cdf)
    mass_error = np.abs(exact_masses - display_masses)
    return {
        "exact_component_count": len(locations),
        "display_component_count": len(display_locations),
        "max_absolute_cdf_error": float(cdf_error.max()),
        "mean_absolute_cdf_error": float(cdf_error.mean()),
        "max_absolute_bin_mass_error": float(mass_error.max()),
        "mean_absolute_bin_mass_error": float(mass_error.mean()),
    }


def margin_display_distribution(
    game: ScheduledGame,
    home: Team,
    away: Team,
    likelihood: LikelihoodV1,
    *,
    minimum: float = FUTURE_MARGIN_DISPLAY_MIN,
    maximum: float = FUTURE_MARGIN_DISPLAY_MAX,
    bins: int = FUTURE_MARGIN_DISPLAY_BINS,
) -> dict[str, Any]:
    """Return deterministic fixed-grid masses for the exact margin mixture.

    The finite grid is presentation data only.  Mass outside the visible
    domain is retained in explicit tail fields, so the browser can show a
    clipped distribution without treating the endpoints as hard limits.
    Exact summaries must continue to come from :func:`predict_game`.
    """
    if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum >= maximum:
        raise ValueError("margin display bounds must be finite and ordered")
    if bins < 1:
        raise ValueError("margin display needs at least one bin")
    locations, weights = predictive_components(game, home, away, likelihood)
    edges = np.linspace(float(minimum), float(maximum), bins + 1)
    locations, weights = _display_components(locations, weights)
    cdf = _display_cdf(edges, locations, weights, likelihood)
    masses = np.diff(cdf)
    lower_tail = float(cdf[0])
    upper_tail = float(1.0 - cdf[-1])
    if (
        not np.isfinite(masses).all()
        or np.any(masses < -1.0e-12)
        or not 0 <= lower_tail <= 1
        or not 0 <= upper_tail <= 1
        or not np.isclose(
            float(masses.sum()) + lower_tail + upper_tail,
            1.0,
            atol=1.0e-10,
            rtol=0.0,
        )
    ):
        raise FloatingPointError("predictive margin display masses are invalid")
    encoded = quantize_display_probabilities(
        np.concatenate((np.maximum(masses, 0.0), [lower_tail, upper_tail])),
        scale=DISPLAY_PROBABILITY_SCALE,
    )
    return {
        "masses": encoded[:-2],
        "lower_tail_probability": encoded[-2],
        "upper_tail_probability": encoded[-1],
    }


def win_probabilities(
    negative_probability: float,
    zero_probability: float = 0.0,
) -> tuple[float, float]:
    """Return home/away win probabilities with half credit for ties.

    ``negative_probability`` is ``P(margin < 0)`` for a home-oriented margin;
    ``zero_probability`` is ``P(margin = 0)``.  Historical Likelihood V1 is a
    continuous Student-t mixture, so production predictions have zero point
    mass at exactly zero, but the explicit convention remains part of the
    artifact contract and is tested independently here.
    """

    if not 0 <= negative_probability <= 1 or not 0 <= zero_probability <= 1:
        raise ValueError("win-probability inputs must be between zero and one")
    if negative_probability + zero_probability > 1 + 1.0e-12:
        raise ValueError("negative and zero margin probabilities exceed one")
    away = negative_probability + 0.5 * zero_probability
    home = 1.0 - negative_probability - 0.5 * zero_probability
    return home, away


def predict_game(
    game: ScheduledGame,
    home: Team,
    away: Team,
    likelihood: LikelihoodV1,
) -> PredictiveMarginSummary:
    """Compute the exact posterior-predictive summary for one scheduled game."""

    locations, weights = predictive_components(game, home, away, likelihood)
    cdf_at_zero = mixture_cdf(
        0.0,
        locations,
        weights,
        likelihood.scale,
        likelihood.degrees_of_freedom,
    )
    home_win, away_win = win_probabilities(cdf_at_zero)
    quantiles = {
        level: (
            mixture_quantile(
                (1.0 - level) / 2.0,
                locations,
                weights,
                likelihood.scale,
                likelihood.degrees_of_freedom,
            ),
            mixture_quantile(
                (1.0 + level) / 2.0,
                locations,
                weights,
                likelihood.scale,
                likelihood.degrees_of_freedom,
            ),
        )
        for level in (0.50, 0.80, 0.95)
    }
    return PredictiveMarginSummary(
        expected_home_margin=float(np.dot(weights, locations)),
        median_home_margin=mixture_quantile(
            0.50,
            locations,
            weights,
            likelihood.scale,
            likelihood.degrees_of_freedom,
        ),
        home_win_probability=home_win,
        away_win_probability=away_win,
        tie_probability=0.0,
        margin_interval_50=quantiles[0.50],
        margin_interval_80=quantiles[0.80],
        margin_interval_95=quantiles[0.95],
    )
