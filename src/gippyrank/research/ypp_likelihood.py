"""Conditional YPP research components.

This module deliberately keeps the production Historical Likelihood V1
untouched.  It supplies the oriented YPP semantics, a small conditional
Student-t family, and a research-only pairwise BP wrapper that multiplies

    p(margin | ranks, site) * p(YPP | margin, ranks, site)

when YPP is observed.  Missing YPP and historically unsupported pairings are
represented by an all-ones factor, so the corrected posterior is exactly the
V1 posterior for that game's evidence.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

from gippyrank.modeling import (
    PAIRINGS,
    design_matrix,
    fit_robust_surface,
    game_log_scores,
)
from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    PosteriorResult,
    Team,
)

MARGIN_KNOTS = (-28.0, -14.0, 0.0, 14.0, 28.0)
DF_GRID = (3.0, 5.0, 8.0, 15.0)
SUPPORTED_YPP_PAIRINGS = frozenset(("fbs-fbs", "fbs-fcs"))
YPP_PAIRING_POLICY = "historically_supported_pairings_only"
_MARGIN_LOCATION_CACHE: dict[tuple[object, ...], np.ndarray] = {}
_MARGIN_FACTOR_CACHE: dict[tuple[object, ...], np.ndarray] = {}
_YPP_LOCATION_CACHE: dict[tuple[object, ...], np.ndarray] = {}
_YPP_FACTOR_CACHE: dict[tuple[object, ...], np.ndarray] = {}


def _as_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _finite_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def pairing_for(home_subdivision: str, away_subdivision: str) -> str:
    home = home_subdivision.casefold()
    away = away_subdivision.casefold()
    if home not in {"fbs", "fcs"} or away not in {"fbs", "fcs"}:
        raise ValueError(f"unsupported subdivisions: {home_subdivision}, {away_subdivision}")
    return "fbs-fcs" if home != away else f"{home}-{away}"


def _normalise_pairing_policy(
    pairings: Iterable[str] | None,
) -> frozenset[str]:
    values = PAIRINGS if pairings is None else tuple(str(value) for value in pairings)
    allowed = frozenset(values)
    unknown = allowed - frozenset(PAIRINGS)
    if unknown:
        raise ValueError(f"unsupported YPP pairing policy values: {sorted(unknown)}")
    return allowed


def ypp_pairing_supported(
    home_subdivision: str,
    away_subdivision: str,
) -> bool:
    """Return whether the corrected experiment has historical YPP support."""

    return pairing_for(home_subdivision, away_subdivision) in SUPPORTED_YPP_PAIRINGS


def oriented_margin(
    home_subdivision: str,
    away_subdivision: str,
    home_points: int,
    away_points: int,
) -> float:
    """Return the V1 margin orientation: same-side home-minus-away or FBS-minus-FCS."""

    if home_subdivision.casefold() == away_subdivision.casefold():
        return float(home_points - away_points)
    if home_subdivision.casefold() == "fbs":
        return float(home_points - away_points)
    if away_subdivision.casefold() == "fbs":
        return float(away_points - home_points)
    raise ValueError("cross-subdivision games must contain one FBS team")


def oriented_ypp_difference(
    home_subdivision: str,
    away_subdivision: str,
    home_ypp: object,
    away_ypp: object,
) -> float | None:
    """Orient YPP exactly as V1 or return ``None`` for unusable observations."""

    home = _finite_float(home_ypp)
    away = _finite_float(away_ypp)
    if home is None or away is None:
        return None
    if home_subdivision.casefold() == away_subdivision.casefold():
        return home - away
    if home_subdivision.casefold() == "fbs":
        return home - away
    if away_subdivision.casefold() == "fbs":
        return away - home
    raise ValueError("cross-subdivision games must contain one FBS team")


def oriented_rank_coordinates(
    home_subdivision: str,
    away_subdivision: str,
    home_rank: float,
    away_rank: float,
    home_population: int,
    away_population: int,
) -> tuple[float, float]:
    """Return percentile coordinates in the V1 modeling orientation."""

    home_coordinate = (float(home_rank) - 0.5) / home_population
    away_coordinate = (float(away_rank) - 0.5) / away_population
    if (
        home_subdivision.casefold() != away_subdivision.casefold()
        and home_subdivision.casefold() == "fcs"
    ):
        return away_coordinate, home_coordinate
    return home_coordinate, away_coordinate


def _base_feature_names(include_margin: bool) -> list[str]:
    if not include_margin:
        return ["intercept"]
    return [
        "intercept",
        "margin_scaled",
        "abs_margin_scaled",
        *(f"margin_hinge_{int(k)}" for k in MARGIN_KNOTS),
    ]


def feature_names(*, rank_signal: bool, include_margin: bool) -> list[str]:
    """Return the fixed, predeclared conditional-model feature names."""

    names: list[str] = []
    for pairing in PAIRINGS:
        block = [f"{pairing}:{name}" for name in _base_feature_names(include_margin)]
        if pairing == "fbs-fcs":
            block.extend(
                [f"{pairing}:fbs_home", f"{pairing}:fcs_home"]
            )
        else:
            block.append(f"{pairing}:home_site")
        if rank_signal:
            rank_names = (
                ("rank_diff", "rank_diff_x_mean", "rank_diff_x_abs_diff")
                if pairing != "fbs-fcs"
                else ("rank_diff", "rank_diff_x_mean")
            )
            block.extend(f"{pairing}:{name}" for name in rank_names)
        names.extend(block)
    return names


def conditional_design(
    margin: np.ndarray,
    pairing: np.ndarray,
    neutral: np.ndarray,
    fbs_home: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    *,
    rank_signal: bool,
    include_margin: bool = True,
) -> np.ndarray:
    """Build the small predeclared conditional-YPP design.

    The margin basis is deliberately fixed before evaluation: intercept,
    signed margin, absolute margin, and five fixed hinges.  Same-subdivision
    rank terms use the odd percentile basis's first three terms.  Cross-
    subdivision rank terms use FBS-minus-FCS percentile contrast and its
    interaction with the mean percentile.  Site indicators follow V1.
    """

    margin = np.asarray(margin, dtype=float)
    pairing = np.asarray(pairing)
    neutral = np.asarray(neutral, dtype=float)
    fbs_home = np.asarray(fbs_home, dtype=float)
    n = len(margin)
    if not (len(pairing) == len(neutral) == len(fbs_home) == n):
        raise ValueError("conditional design inputs must have equal lengths")
    if rank_signal and (x is None or y is None):
        raise ValueError("rank coordinates are required for a rank-signal design")
    x = np.zeros(n, dtype=float) if x is None else np.asarray(x, dtype=float)
    y = np.zeros(n, dtype=float) if y is None else np.asarray(y, dtype=float)
    if len(x) != n or len(y) != n:
        raise ValueError("rank coordinate inputs must have equal lengths")

    if include_margin:
        scaled = margin / 20.0
        base = np.column_stack(
            [
                np.ones(n),
                scaled,
                np.abs(scaled),
                *[np.maximum((margin - knot) / 20.0, 0.0) for knot in MARGIN_KNOTS],
            ]
        )
    else:
        base = np.ones((n, 1), dtype=float)

    rows: list[np.ndarray] = []
    for name in PAIRINGS:
        values = np.zeros((n, base.shape[1] + (2 if name == "fbs-fcs" else 1)), dtype=float)
        selected = pairing == name
        values[selected, : base.shape[1]] = base[selected]
        if name == "fbs-fcs":
            values[selected, base.shape[1]] = fbs_home[selected]
            values[selected, base.shape[1] + 1] = (
                (1.0 - neutral[selected]) - fbs_home[selected]
            )
        else:
            values[selected, base.shape[1]] = 1.0 - neutral[selected]
        if rank_signal:
            d = x - y
            s = (x + y) / 2.0
            rank_values = (
                np.column_stack([d, d * s, d * np.abs(d)])
                if name != "fbs-fcs"
                else np.column_stack([d, d * s])
            )
            values = np.column_stack([values, rank_values * selected[:, None]])
        rows.append(values * selected[:, None])
    return np.column_stack(rows)


def _student_t_logpdf(
    value: np.ndarray | float,
    location: np.ndarray,
    scale: float,
    degrees_of_freedom: float,
) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    z = (value - location) / scale
    constant = (
        gammaln((degrees_of_freedom + 1) / 2)
        - gammaln(degrees_of_freedom / 2)
        - 0.5 * np.log(degrees_of_freedom * np.pi)
        - np.log(scale)
    )
    return constant - (degrees_of_freedom + 1) / 2 * np.log1p(
        z * z / degrees_of_freedom
    )


@dataclass(frozen=True)
class YPPData:
    """Rank-pair pseudo-observations with equal total weight per game."""

    x: np.ndarray
    y: np.ndarray
    target: np.ndarray
    margin: np.ndarray
    pairing: np.ndarray
    neutral: np.ndarray
    fbs_home: np.ndarray
    weight: np.ndarray
    game_id: np.ndarray
    season: np.ndarray

    def __len__(self) -> int:
        return len(self.target)


def build_ypp_data(
    rows: Sequence[Mapping[str, object]],
    *,
    allowed_pairings: Iterable[str] | None = SUPPORTED_YPP_PAIRINGS,
) -> YPPData:
    """Expand usable supported-game rank supports into weighted observations.

    The corrected experiment defaults to the prospective support boundary.  A
    caller must explicitly pass ``None`` (all pairings) for the retained
    legacy all-pairings diagnostic.
    """

    allowed = _normalise_pairing_policy(allowed_pairings)
    values: list[list[object]] = [[] for _ in range(10)]
    for row in rows:
        pairing = pairing_for(
            str(row["home_subdivision"]), str(row["away_subdivision"])
        )
        if pairing not in allowed:
            continue
        target = oriented_ypp_difference(
            str(row["home_subdivision"]),
            str(row["away_subdivision"]),
            row.get("home_ypp"),
            row.get("away_ypp"),
        )
        if target is None:
            continue
        rank_pairs_value = row.get("rank_pairs")
        pairs = (
            json.loads(str(rank_pairs_value))
            if isinstance(rank_pairs_value, str)
            else rank_pairs_value
        )
        if not pairs:
            continue
        home_subdivision = str(row["home_subdivision"]).casefold()
        away_subdivision = str(row["away_subdivision"]).casefold()
        neutral = _as_bool(row.get("neutral_site"))
        cross = home_subdivision != away_subdivision
        fbs_home = float(cross and home_subdivision == "fbs" and not neutral)
        n = len(pairs)
        for home_rank, away_rank in pairs:
            x, y = oriented_rank_coordinates(
                home_subdivision,
                away_subdivision,
                float(home_rank),
                float(away_rank),
                int(row["home_team_population"]),
                int(row["away_team_population"]),
            )
            vals = [
                x,
                y,
                target,
                oriented_margin(
                    home_subdivision,
                    away_subdivision,
                    int(row["home_points"]),
                    int(row["away_points"]),
                ),
                pairing,
                float(neutral),
                fbs_home,
                1.0 / n,
                str(row["game_id"]),
                int(row["season"]),
            ]
            for output, value in zip(values, vals):
                output.append(value)
    arrays = tuple(np.asarray(value) for value in values)
    return YPPData(*arrays)


def fit_ypp_model(
    data: YPPData,
    mask: np.ndarray,
    *,
    rank_signal: bool,
    include_margin: bool = True,
    degrees_of_freedom: float,
    allowed_pairings: Iterable[str] | None = SUPPORTED_YPP_PAIRINGS,
) -> dict[str, object]:
    """Fit one frozen candidate specification by weighted robust regression.

    Filtering is repeated here as a semantic guard so unsupported rows cannot
    enter a corrected fit even if the caller supplied an all-pairing data
    object.  The legacy diagnostic opts into all pairings explicitly.
    """

    allowed = _normalise_pairing_policy(allowed_pairings)
    fit_mask = np.asarray(mask, dtype=bool) & np.isin(
        data.pairing, tuple(sorted(allowed))
    )
    if not np.any(fit_mask):
        raise ValueError("YPP fit has no observations in the requested pairing policy")
    matrix = conditional_design(
        data.margin[fit_mask],
        data.pairing[fit_mask],
        data.neutral[fit_mask],
        data.fbs_home[fit_mask],
        data.x[fit_mask],
        data.y[fit_mask],
        rank_signal=rank_signal,
        include_margin=include_margin,
    )
    model = fit_robust_surface(
        matrix,
        data.target[fit_mask],
        data.weight[fit_mask],
        df=degrees_of_freedom,
    )
    return {
        **model,
        "rank_signal": rank_signal,
        "include_margin": include_margin,
        "fit_pairings": tuple(sorted(allowed)),
        "fit_game_count": len(set(data.game_id[fit_mask].tolist())),
        "fit_pseudo_observation_count": int(np.sum(fit_mask)),
        "feature_names": feature_names(
            rank_signal=rank_signal, include_margin=include_margin
        ),
    }


def ypp_model_scores(
    model: Mapping[str, object], data: YPPData, mask: np.ndarray
) -> dict[str, float]:
    """Score a conditional model by equal-game marginalized density."""

    matrix = conditional_design(
        data.margin[mask],
        data.pairing[mask],
        data.neutral[mask],
        data.fbs_home[mask],
        data.x[mask],
        data.y[mask],
        rank_signal=bool(model["rank_signal"]),
        include_margin=bool(model["include_margin"]),
    )
    locations = matrix @ np.asarray(model["beta"], dtype=float)
    return game_log_scores(
        data.target[mask],
        locations,
        data.game_id[mask],
        float(model["scale"]),
        float(model["df"]),
    )


def ypp_model_locations(
    model: Mapping[str, object],
    *,
    margin: np.ndarray,
    pairing: np.ndarray,
    neutral: np.ndarray,
    fbs_home: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    matrix = conditional_design(
        margin,
        pairing,
        neutral,
        fbs_home,
        x,
        y,
        rank_signal=bool(model["rank_signal"]),
        include_margin=bool(model["include_margin"]),
    )
    return matrix @ np.asarray(model["beta"], dtype=float)


def ypp_factor(
    game: Game,
    home: Team,
    away: Team,
    model: Mapping[str, object] | None,
    home_ypp: object,
    away_ypp: object,
    *,
    allowed_pairings: Iterable[str] | None = SUPPORTED_YPP_PAIRINGS,
) -> np.ndarray:
    """Return a normalized conditional YPP factor or exact V1 fallback.

    The default policy disables YPP for FCS--FCS regardless of observation
    availability.  Passing ``None`` is reserved for the explicitly retained
    pre-correction all-pairings diagnostic.
    """

    allowed = _normalise_pairing_policy(allowed_pairings)
    pairing = pairing_for(home.subdivision, away.subdivision)
    target = oriented_ypp_difference(
        home.subdivision, away.subdivision, home_ypp, away_ypp
    )
    if pairing not in allowed or target is None or model is None:
        return np.ones((len(home.prior), len(away.prior)), dtype=float)

    beta = np.asarray(model["beta"], dtype=float)
    factor_key = (
        beta.tobytes(),
        float(model["scale"]),
        float(model["df"]),
        bool(model["rank_signal"]),
        bool(model["include_margin"]),
        home.subdivision,
        away.subdivision,
        len(home.prior),
        len(away.prior),
        game.home_points,
        game.away_points,
        game.neutral_site,
        target,
        tuple(sorted(allowed)),
    )
    cached = _YPP_FACTOR_CACHE.get(factor_key)
    if cached is not None:
        return cached

    locations = _cached_ypp_locations(game, home, away, model)
    logs = _student_t_logpdf(
        target,
        locations,
        float(model["scale"]),
        float(model["df"]),
    )
    factor = np.exp(logs - np.max(logs))
    _YPP_FACTOR_CACHE[factor_key] = factor
    return factor


def _cached_ypp_locations(
    game: Game,
    home: Team,
    away: Team,
    model: Mapping[str, object],
) -> np.ndarray:
    """Cache the conditional mean surface shared by repeated game factors.

    A posterior panel revisits the same score/site/rank geometry at several
    cutoffs and prior families.  The conditional mean depends on that
    geometry and the fitted coefficients, but not on the observed YPP value;
    caching it preserves the factor semantics while avoiding repeated design
    matrix construction.
    """

    beta = np.asarray(model["beta"], dtype=float)
    key = (
        beta.tobytes(),
        bool(model["rank_signal"]),
        bool(model["include_margin"]),
        home.subdivision,
        away.subdivision,
        len(home.prior),
        len(away.prior),
        game.home_points,
        game.away_points,
        game.neutral_site,
    )
    locations = _YPP_LOCATION_CACHE.get(key)
    if locations is not None:
        return locations

    home_rank = np.arange(1, len(home.prior) + 1, dtype=float)
    away_rank = np.arange(1, len(away.prior) + 1, dtype=float)
    home_percentile = (home_rank - 0.5) / len(home_rank)
    away_percentile = (away_rank - 0.5) / len(away_rank)
    cross = home.subdivision != away.subdivision
    if cross and home.subdivision == "fcs":
        x, y = np.meshgrid(away_percentile, home_percentile, indexing="ij")
    else:
        x, y = np.meshgrid(home_percentile, away_percentile, indexing="ij")
    x_flat, y_flat = x.ravel(), y.ravel()
    margin = oriented_margin(
        home.subdivision,
        away.subdivision,
        game.home_points,
        game.away_points,
    )
    pairing = np.full(x_flat.size, pairing_for(home.subdivision, away.subdivision))
    neutral = np.full(x_flat.size, float(game.neutral_site))
    fbs_home = np.full(
        x_flat.size,
        float(cross and home.subdivision == "fbs" and not game.neutral_site),
    )
    locations = ypp_model_locations(
        model,
        margin=np.full(x_flat.size, margin),
        pairing=pairing,
        neutral=neutral,
        fbs_home=fbs_home,
        x=x_flat,
        y=y_flat,
    )
    if cross and home.subdivision == "fcs":
        locations = locations.reshape(len(away.prior), len(home.prior)).T
    else:
        locations = locations.reshape(len(home.prior), len(away.prior))
    _YPP_LOCATION_CACHE[key] = locations
    return locations


def _fast_margin_factor(
    game: Game, home: Team, away: Team, likelihood: LikelihoodV1
) -> np.ndarray:
    """Research-local equivalent of V1's factor with cached rank geometry."""

    key = (
        likelihood.beta.tobytes(),
        likelihood.scale,
        likelihood.degrees_of_freedom,
        home.subdivision,
        away.subdivision,
        len(home.prior),
        len(away.prior),
        game.neutral_site,
    )
    locations = _MARGIN_LOCATION_CACHE.get(key)
    if locations is None:
        home_rank = np.arange(1, len(home.prior) + 1, dtype=float)
        away_rank = np.arange(1, len(away.prior) + 1, dtype=float)
        hp = (home_rank - 0.5) / len(home_rank)
        ap = (away_rank - 0.5) / len(away_rank)
        cross = home.subdivision != away.subdivision
        if cross and home.subdivision == "fcs":
            x, y = np.meshgrid(ap, hp, indexing="ij")
        else:
            x, y = np.meshgrid(hp, ap, indexing="ij")
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
        flat_locations = matrix @ likelihood.beta
        if cross and home.subdivision == "fcs":
            locations = flat_locations.reshape(len(away.prior), len(home.prior)).T
        else:
            locations = flat_locations.reshape(len(home.prior), len(away.prior))
        _MARGIN_LOCATION_CACHE[key] = locations
    margin = oriented_margin(
        home.subdivision,
        away.subdivision,
        game.home_points,
        game.away_points,
    )
    factor_key = (
        likelihood.beta.tobytes(),
        likelihood.scale,
        likelihood.degrees_of_freedom,
        home.subdivision,
        away.subdivision,
        len(home.prior),
        len(away.prior),
        game.neutral_site,
        margin,
    )
    cached_factor = _MARGIN_FACTOR_CACHE.get(factor_key)
    if cached_factor is not None:
        return cached_factor
    logs = _student_t_logpdf(
        margin,
        locations,
        likelihood.scale,
        likelihood.degrees_of_freedom,
    )
    factor = np.exp(logs - np.max(logs))
    _MARGIN_FACTOR_CACHE[factor_key] = factor
    return factor


def infer_posterior_with_ypp(
    teams: list[Team],
    games: Sequence[Game],
    likelihood: LikelihoodV1,
    ypp_by_game: Mapping[str, tuple[object, object]],
    model: Mapping[str, object] | None,
    *,
    allowed_pairings: Iterable[str] | None = SUPPORTED_YPP_PAIRINGS,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    damping: float = 0.35,
) -> PosteriorResult:
    """Run the production BP semantics with an optional research YPP factor."""

    by_id = {team.team_id: team for team in teams}
    if len(by_id) != len(teams):
        raise ValueError("team IDs must be unique")
    grouped: dict[tuple[str, str], tuple[np.ndarray, list[str]]] = {}
    for game in games:
        if game.home_id not in by_id or game.away_id not in by_id:
            raise ValueError(f"game {game.game_id} references a team without a prior")
        home, away = by_id[game.home_id], by_id[game.away_id]
        margin_values = _fast_margin_factor(game, home, away, likelihood)
        home_ypp, away_ypp = ypp_by_game.get(game.game_id, (None, None))
        values = margin_values * ypp_factor(
            game,
            home,
            away,
            model,
            home_ypp,
            away_ypp,
            allowed_pairings=allowed_pairings,
        )
        key = tuple(sorted((game.home_id, game.away_id)))
        if (game.home_id, game.away_id) != key:
            values = values.T
        log_values = np.log(np.maximum(values, np.finfo(float).tiny))
        if key in grouped:
            previous, game_ids = grouped[key]
            grouped[key] = (previous + log_values, [*game_ids, game.game_id])
        else:
            grouped[key] = (log_values, [game.game_id])

    factors = [
        (
            first,
            second,
            np.exp(log_values - np.max(log_values)),
            tuple(game_ids),
        )
        for (first, second), (log_values, game_ids) in sorted(grouped.items())
    ]
    adjacency: dict[str, list[int]] = {team.team_id: [] for team in teams}
    for index, (first, second, _values, _game_ids) in enumerate(factors):
        adjacency[first].append(index)
        adjacency[second].append(index)
    messages: dict[tuple[int, str], np.ndarray] = {}
    for index, (first, second, _values, _game_ids) in enumerate(factors):
        messages[index, first] = np.ones(len(by_id[first].prior))
        messages[index, second] = np.ones(len(by_id[second].prior))
    if not factors:
        return PosteriorResult(
            {team.team_id: team.prior.copy() for team in teams},
            True,
            0,
            0.0,
            0.0,
            0,
            0,
            0,
        )

    def normalise(values: np.ndarray) -> np.ndarray:
        values = np.maximum(values, 0.0)
        total = values.sum()
        if not np.isfinite(total) or total <= 0:
            raise FloatingPointError("zero or non-finite BP message")
        return values / total

    max_delta = np.inf
    for iteration in range(1, max_iterations + 1):
        updates: dict[tuple[int, str], np.ndarray] = {}
        max_delta = 0.0
        for index, (first, second, values, _game_ids) in enumerate(factors):
            for target, other, transpose in (
                (first, second, False),
                (second, first, True),
            ):
                belief = by_id[other].prior.copy()
                for neighbour in adjacency[other]:
                    if neighbour != index:
                        belief *= messages[neighbour, other]
                belief = normalise(belief)
                raw = values.T @ belief if transpose else values @ belief
                proposal = normalise(raw)
                old = messages[index, target]
                update = normalise(damping * old + (1.0 - damping) * proposal)
                max_delta = max(max_delta, float(np.max(np.abs(update - old))))
                updates[index, target] = update
        messages.update(updates)
        if max_delta <= tolerance:
            break
    pmfs: dict[str, np.ndarray] = {}
    for team in teams:
        belief = team.prior.copy()
        for index in adjacency[team.team_id]:
            belief *= messages[index, team.team_id]
        pmfs[team.team_id] = normalise(belief)
    objective = float(sum(np.log(pmf.max()) for pmf in pmfs.values()))
    return PosteriorResult(
        pmfs,
        max_delta <= tolerance,
        iteration,
        max_delta,
        objective,
        len(games),
        len(factors),
        max(map(len, adjacency.values()), default=0),
    )
