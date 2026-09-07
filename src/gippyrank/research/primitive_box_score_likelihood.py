"""Research-only conditional likelihoods for audited primitive box scores.

The production Historical Likelihood V1 is intentionally not imported into or
changed by this module's fitting code.  The research factors are multiplied
onto the frozen V1 margin factor only for the historically supported pairings:
FBS--FBS and FBS--FCS.  FCS--FCS therefore remains an exact V1 fallback.

The primitive representation is lossless at the team-pair level for the four
selected fields.  The oriented differences identify which side was better and
the totals retain game environment:

    yards_diff, yards_total, plays_diff, plays_total

Interceptions thrown and fumbles lost are kept as two separate difference and
total context pairs.  They can alter the conditional location in Candidate C,
but there is no direct turnover likelihood factor.
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
SUPPORTED_PRIMITIVE_PAIRINGS = frozenset(("fbs-fbs", "fbs-fcs"))

# These are fixed response/context scales, not fitted or outcome-dependent
# winsorization thresholds.  The asinh transform is odd and keeps all finite
# audited values, including negative differences.
YARDS_DIFF_SCALE = 100.0
YARDS_TOTAL_SCALE = 200.0
PLAYS_DIFF_SCALE = 10.0
PLAYS_TOTAL_SCALE = 80.0
TURNOVER_SCALE = 1.0

PRIMITIVE_FIELDS = (
    "total_yards",
    "offensive_plays_derived",
    "interceptions_thrown",
    "fumbles_lost",
)

_LOCATION_CACHE: dict[tuple[object, ...], np.ndarray] = {}
_FACTOR_CACHE: dict[tuple[object, ...], np.ndarray] = {}
_V1_LOCATION_CACHE: dict[tuple[object, ...], np.ndarray] = {}
_V1_FACTOR_CACHE: dict[tuple[object, ...], np.ndarray] = {}


def _finite_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _as_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _asinh_scaled(value: np.ndarray | float, scale: float) -> np.ndarray:
    return np.arcsinh(np.asarray(value, dtype=float) / scale)


def pairing_for(home_subdivision: str, away_subdivision: str) -> str:
    """Return the canonical pairing, collapsing cross-subdivision order."""

    home = home_subdivision.casefold()
    away = away_subdivision.casefold()
    if home not in {"fbs", "fcs"} or away not in {"fbs", "fcs"}:
        raise ValueError(f"unsupported subdivisions: {home_subdivision}, {away_subdivision}")
    return "fbs-fcs" if home != away else f"{home}-{away}"


def oriented_margin(
    home_subdivision: str,
    away_subdivision: str,
    home_points: int,
    away_points: int,
) -> float:
    """Orient score margin home-minus-away or FBS-minus-FCS."""

    home = home_subdivision.casefold()
    away = away_subdivision.casefold()
    if home == away or home == "fbs":
        return float(home_points - away_points)
    if away == "fbs":
        return float(away_points - home_points)
    raise ValueError("cross-subdivision games must contain one FBS team")


def oriented_difference(
    home_subdivision: str,
    away_subdivision: str,
    home_value: object,
    away_value: object,
) -> float | None:
    """Orient a finite team value as same-side or FBS-minus-FCS."""

    home = _finite_float(home_value)
    away = _finite_float(away_value)
    if home is None or away is None:
        return None
    if home_subdivision.casefold() == away_subdivision.casefold():
        return home - away
    if home_subdivision.casefold() == "fbs":
        return home - away
    if away_subdivision.casefold() == "fbs":
        return away - home
    raise ValueError("cross-subdivision games must contain one FBS team")


def oriented_total(home_value: object, away_value: object) -> float | None:
    home = _finite_float(home_value)
    away = _finite_float(away_value)
    return None if home is None or away is None else home + away


def oriented_rank_coordinates(
    home_subdivision: str,
    away_subdivision: str,
    home_rank: float,
    away_rank: float,
    home_population: int,
    away_population: int,
) -> tuple[float, float]:
    """Return percentile coordinates in the V1 FBS-first orientation."""

    home_coordinate = (float(home_rank) - 0.5) / home_population
    away_coordinate = (float(away_rank) - 0.5) / away_population
    if (
        home_subdivision.casefold() != away_subdivision.casefold()
        and home_subdivision.casefold() == "fcs"
    ):
        return away_coordinate, home_coordinate
    return home_coordinate, away_coordinate


def _rank_feature_names(pairing: str) -> list[str]:
    if pairing == "fbs-fcs":
        return [
            "intercept",
            "fbs_percentile",
            "fcs_percentile",
            "fbs_percentile_squared",
            "fcs_percentile_squared",
            "fbs_x_fcs_percentile",
            *(f"fbs_hinge_{int(k * 100)}" for k in (0.2, 0.4, 0.6, 0.8)),
            *(f"fcs_hinge_{int(k * 100)}" for k in (0.2, 0.4, 0.6, 0.8)),
            "fbs_home",
            "fcs_home",
        ]
    return [
        "rank_diff",
        "rank_diff_x_mean",
        "rank_diff_x_mean_squared",
        "rank_diff_x_abs_diff",
        *(f"rank_diff_x_hinge_{int(k * 100)}" for k in (0.2, 0.4, 0.6, 0.8)),
        "home_site",
    ]


def feature_names(
    *, response_kind: str, rank_signal: bool, turnover_context: bool
) -> list[str]:
    """Return deterministic names for the complete conditional design."""

    if response_kind not in {"plays", "yards"}:
        raise ValueError(f"unknown primitive response: {response_kind}")
    names: list[str] = []
    for pairing in PAIRINGS:
        names.extend(f"{pairing}:v1:{name}" for name in _rank_feature_names(pairing))
    # design_matrix always reserves the V1 rank columns.  When rank_signal is
    # false, the corresponding inputs are zeros; retaining the columns keeps
    # model shapes comparable and makes the null-control semantics explicit.
    del rank_signal
    for pairing in PAIRINGS:
        names.extend(
            f"{pairing}:margin:{name}"
            for name in (
                "signed_scaled",
                "absolute_scaled",
                *(f"hinge_{int(k)}" for k in MARGIN_KNOTS),
            )
        )
        context = (
            ("plays_total_context",)
            if response_kind == "plays"
            else ("plays_diff_context", "plays_total_context", "yards_total_context")
        )
        if turnover_context:
            context += (
                "interceptions_diff_context",
                "interceptions_total_context",
                "fumbles_lost_diff_context",
                "fumbles_lost_total_context",
            )
        names.extend(f"{pairing}:context:{name}" for name in context)
    return names


@dataclass(frozen=True)
class PrimitiveData:
    """Rank-pair pseudo-observations with equal total weight per game."""

    x: np.ndarray
    y: np.ndarray
    margin: np.ndarray
    pairing: np.ndarray
    neutral: np.ndarray
    fbs_home: np.ndarray
    yards_diff: np.ndarray
    yards_total: np.ndarray
    plays_diff: np.ndarray
    plays_total: np.ndarray
    interceptions_diff: np.ndarray
    interceptions_total: np.ndarray
    fumbles_lost_diff: np.ndarray
    fumbles_lost_total: np.ndarray
    weight: np.ndarray
    game_id: np.ndarray
    season: np.ndarray

    def __len__(self) -> int:
        return len(self.margin)


def _json_pairs(value: object) -> list[tuple[float, float]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not value:
        return []
    return [(float(pair[0]), float(pair[1])) for pair in value]


def build_primitive_data(rows: Sequence[Mapping[str, object]]) -> PrimitiveData:
    """Expand rows into weighted rank-pair primitive observations.

    Missing fields become NaN in the data object and are filtered separately by
    each conditional component.  They are never interpreted as zero.
    """

    values: list[list[object]] = [[] for _ in range(17)]
    for row in rows:
        pairing = pairing_for(str(row["home_subdivision"]), str(row["away_subdivision"]))
        if pairing not in PAIRINGS:
            continue
        pairs = _json_pairs(row.get("rank_pairs"))
        if not pairs:
            continue
        home_subdivision = str(row["home_subdivision"]).casefold()
        away_subdivision = str(row["away_subdivision"]).casefold()
        neutral = float(_as_bool(row.get("neutral_site")))
        cross = home_subdivision != away_subdivision
        fbs_home = float(cross and home_subdivision == "fbs" and not neutral)
        n = len(pairs)
        diffs = {
            field: oriented_difference(
                home_subdivision,
                away_subdivision,
                row.get(f"home_{field}"),
                row.get(f"away_{field}"),
            )
            for field in PRIMITIVE_FIELDS
        }
        totals = {
            field: oriented_total(row.get(f"home_{field}"), row.get(f"away_{field}"))
            for field in PRIMITIVE_FIELDS
        }
        oriented = (
            diffs["total_yards"],
            totals["total_yards"],
            diffs["offensive_plays_derived"],
            totals["offensive_plays_derived"],
            diffs["interceptions_thrown"],
            totals["interceptions_thrown"],
            diffs["fumbles_lost"],
            totals["fumbles_lost"],
        )
        for home_rank, away_rank in pairs:
            x, y = oriented_rank_coordinates(
                home_subdivision,
                away_subdivision,
                home_rank,
                away_rank,
                int(row["home_team_population"]),
                int(row["away_team_population"]),
            )
            vals = [
                x,
                y,
                oriented_margin(
                    home_subdivision,
                    away_subdivision,
                    int(row["home_points"]),
                    int(row["away_points"]),
                ),
                pairing,
                neutral,
                fbs_home,
                *oriented,
                1.0 / n,
                str(row["game_id"]),
                int(row["season"]),
            ]
            for output, value in zip(values, vals):
                output.append(value if value is not None else np.nan)
    arrays = tuple(np.asarray(value) for value in values)
    return PrimitiveData(*arrays)


def _context_arrays(
    data: PrimitiveData,
    response_kind: str,
    turnover_context: bool,
) -> tuple[np.ndarray, ...]:
    if response_kind == "plays":
        context = (_asinh_scaled(data.plays_total, PLAYS_TOTAL_SCALE),)
    elif response_kind == "yards":
        context = (
            _asinh_scaled(data.plays_diff, PLAYS_DIFF_SCALE),
            _asinh_scaled(data.plays_total, PLAYS_TOTAL_SCALE),
            _asinh_scaled(data.yards_total, YARDS_TOTAL_SCALE),
        )
    else:
        raise ValueError(f"unknown primitive response: {response_kind}")
    if turnover_context:
        context += (
            _asinh_scaled(data.interceptions_diff, TURNOVER_SCALE),
            _asinh_scaled(data.interceptions_total, TURNOVER_SCALE),
            _asinh_scaled(data.fumbles_lost_diff, TURNOVER_SCALE),
            _asinh_scaled(data.fumbles_lost_total, TURNOVER_SCALE),
        )
    return context


def primitive_design(
    *,
    margin: np.ndarray,
    pairing: np.ndarray,
    neutral: np.ndarray,
    fbs_home: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    response_kind: str,
    turnover_context: bool,
    plays_diff: np.ndarray,
    plays_total: np.ndarray,
    yards_total: np.ndarray,
    interceptions_diff: np.ndarray,
    interceptions_total: np.ndarray,
    fumbles_lost_diff: np.ndarray,
    fumbles_lost_total: np.ndarray,
    rank_signal: bool,
) -> np.ndarray:
    """Build the fixed conditional design for one primitive component."""

    margin = np.asarray(margin, dtype=float)
    pairing = np.asarray(pairing)
    neutral = np.asarray(neutral, dtype=float)
    fbs_home = np.asarray(fbs_home, dtype=float)
    n = len(margin)
    arrays = (pairing, neutral, fbs_home, np.asarray(x), np.asarray(y))
    if any(len(array) != n for array in arrays):
        raise ValueError("primitive design inputs must have equal lengths")
    rank_x = np.asarray(x, dtype=float) if rank_signal else np.zeros(n)
    rank_y = np.asarray(y, dtype=float) if rank_signal else np.zeros(n)
    base = design_matrix(
        rank_x,
        rank_y,
        pairing,
        1.0 - neutral,
        neutral,
        surface=True,
        fbs_home=fbs_home,
    )
    scaled = margin / 20.0
    margin_values = np.column_stack(
        [
            scaled,
            np.abs(scaled),
            *[np.maximum((margin - knot) / 20.0, 0.0) for knot in MARGIN_KNOTS],
        ]
    )
    if response_kind == "plays":
        context_values = [
            _asinh_scaled(plays_total, PLAYS_TOTAL_SCALE),
        ]
    elif response_kind == "yards":
        context_values = [
            _asinh_scaled(plays_diff, PLAYS_DIFF_SCALE),
            _asinh_scaled(plays_total, PLAYS_TOTAL_SCALE),
            _asinh_scaled(yards_total, YARDS_TOTAL_SCALE),
        ]
    else:
        raise ValueError(f"unknown primitive response: {response_kind}")
    if turnover_context:
        context_values.extend(
            [
                _asinh_scaled(interceptions_diff, TURNOVER_SCALE),
                _asinh_scaled(interceptions_total, TURNOVER_SCALE),
                _asinh_scaled(fumbles_lost_diff, TURNOVER_SCALE),
                _asinh_scaled(fumbles_lost_total, TURNOVER_SCALE),
            ]
        )
    additions: list[np.ndarray] = []
    for index, name in enumerate(PAIRINGS):
        selected = (pairing == name).astype(float)
        additions.append(margin_values * selected[:, None])
        additions.extend(value[:, None] * selected[:, None] for value in context_values)
    return np.column_stack([base, *additions])


def _response(data: PrimitiveData, response_kind: str) -> np.ndarray:
    if response_kind == "plays":
        return _asinh_scaled(data.plays_diff, PLAYS_DIFF_SCALE)
    if response_kind == "yards":
        return _asinh_scaled(data.yards_diff, YARDS_DIFF_SCALE)
    raise ValueError(f"unknown primitive response: {response_kind}")


def _design_from_data(
    data: PrimitiveData,
    mask: np.ndarray,
    *,
    response_kind: str,
    turnover_context: bool,
    rank_signal: bool,
) -> np.ndarray:
    return primitive_design(
        margin=data.margin[mask],
        pairing=data.pairing[mask],
        neutral=data.neutral[mask],
        fbs_home=data.fbs_home[mask],
        x=data.x[mask],
        y=data.y[mask],
        response_kind=response_kind,
        turnover_context=turnover_context,
        plays_diff=data.plays_diff[mask],
        plays_total=data.plays_total[mask],
        yards_total=data.yards_total[mask],
        interceptions_diff=data.interceptions_diff[mask],
        interceptions_total=data.interceptions_total[mask],
        fumbles_lost_diff=data.fumbles_lost_diff[mask],
        fumbles_lost_total=data.fumbles_lost_total[mask],
        rank_signal=rank_signal,
    )


def component_mask(
    data: PrimitiveData,
    seasons: Iterable[int],
    *,
    response_kind: str,
    turnover_context: bool = False,
) -> np.ndarray:
    """Return complete, supported pseudo-observation rows for a component."""

    season_mask = np.isin(data.season, tuple(int(value) for value in seasons))
    supported = np.isin(data.pairing, tuple(sorted(SUPPORTED_PRIMITIVE_PAIRINGS)))
    needed = [data.plays_diff, data.plays_total]
    if response_kind == "yards":
        needed.extend((data.yards_diff, data.yards_total))
    elif response_kind != "plays":
        raise ValueError(f"unknown primitive response: {response_kind}")
    if turnover_context:
        needed.extend(
            (
                data.interceptions_diff,
                data.interceptions_total,
                data.fumbles_lost_diff,
                data.fumbles_lost_total,
            )
        )
    complete = np.ones(len(data), dtype=bool)
    for value in needed:
        complete &= np.isfinite(value)
    return season_mask & supported & complete


def fit_primitive_model(
    data: PrimitiveData,
    mask: np.ndarray,
    *,
    response_kind: str,
    turnover_context: bool,
    rank_signal: bool = True,
    degrees_of_freedom: float,
) -> dict[str, object]:
    """Fit a predeclared robust conditional Student-t component."""

    mask = np.asarray(mask, dtype=bool)
    required = component_mask(
        data,
        data.season[mask].tolist(),
        response_kind=response_kind,
        turnover_context=turnover_context,
    ) & mask
    if not np.any(required):
        raise ValueError("primitive fit has no complete supported observations")
    matrix = _design_from_data(
        data,
        required,
        response_kind=response_kind,
        turnover_context=turnover_context,
        rank_signal=rank_signal,
    )
    model = fit_robust_surface(
        matrix,
        _response(data, response_kind)[required],
        data.weight[required],
        df=degrees_of_freedom,
    )
    return {
        **model,
        "response_kind": response_kind,
        "turnover_context": turnover_context,
        "rank_signal": rank_signal,
        "fit_pairings": tuple(sorted(SUPPORTED_PRIMITIVE_PAIRINGS)),
        "fit_game_count": len(set(data.game_id[required].tolist())),
        "fit_pseudo_observation_count": int(np.sum(required)),
        "feature_names": feature_names(
            response_kind=response_kind,
            rank_signal=rank_signal,
            turnover_context=turnover_context,
        ),
    }


def primitive_model_scores(
    model: Mapping[str, object], data: PrimitiveData, mask: np.ndarray
) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return {"expected_conditional_nll": float("nan"), "marginalized_nll": float("nan"), "n_games": 0}
    matrix = _design_from_data(
        data,
        mask,
        response_kind=str(model["response_kind"]),
        turnover_context=bool(model["turnover_context"]),
        rank_signal=bool(model["rank_signal"]),
    )
    locations = matrix @ np.asarray(model["beta"], dtype=float)
    return game_log_scores(
        _response(data, str(model["response_kind"]))[mask],
        locations,
        data.game_id[mask],
        float(model["scale"]),
        float(model["df"]),
    )


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


def _model_locations(
    model: Mapping[str, object],
    *,
    margin: np.ndarray,
    pairing: np.ndarray,
    neutral: np.ndarray,
    fbs_home: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    plays_diff: np.ndarray,
    plays_total: np.ndarray,
    yards_total: np.ndarray,
    interceptions_diff: np.ndarray,
    interceptions_total: np.ndarray,
    fumbles_lost_diff: np.ndarray,
    fumbles_lost_total: np.ndarray,
) -> np.ndarray:
    matrix = primitive_design(
        margin=margin,
        pairing=pairing,
        neutral=neutral,
        fbs_home=fbs_home,
        x=x,
        y=y,
        response_kind=str(model["response_kind"]),
        turnover_context=bool(model["turnover_context"]),
        plays_diff=plays_diff,
        plays_total=plays_total,
        yards_total=yards_total,
        interceptions_diff=interceptions_diff,
        interceptions_total=interceptions_total,
        fumbles_lost_diff=fumbles_lost_diff,
        fumbles_lost_total=fumbles_lost_total,
        rank_signal=bool(model["rank_signal"]),
    )
    beta = np.asarray(model["beta"], dtype=float)
    if matrix.shape[1] != len(beta):
        raise ValueError(
            f"primitive model has {len(beta)} coefficients but design has {matrix.shape[1]} columns"
        )
    return matrix @ beta


def _evidence_tuple(
    home: Mapping[str, object], away: Mapping[str, object]
) -> tuple[object, ...]:
    return tuple(
        value
        for field in PRIMITIVE_FIELDS
        for value in (home.get(field), away.get(field))
    )


def _oriented_evidence(
    game: Game,
    home: Team,
    away: Team,
    home_evidence: Mapping[str, object],
    away_evidence: Mapping[str, object],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for field in PRIMITIVE_FIELDS:
        result[f"{field}_diff"] = oriented_difference(
            home.subdivision,
            away.subdivision,
            home_evidence.get(field),
            away_evidence.get(field),
        )
        result[f"{field}_total"] = oriented_total(
            home_evidence.get(field), away_evidence.get(field)
        )
    del game
    return result


def _rank_grid(home: Team, away: Team) -> tuple[np.ndarray, np.ndarray, bool]:
    home_rank = np.arange(1, len(home.prior) + 1, dtype=float)
    away_rank = np.arange(1, len(away.prior) + 1, dtype=float)
    home_percentile = (home_rank - 0.5) / len(home_rank)
    away_percentile = (away_rank - 0.5) / len(away_rank)
    cross = home.subdivision != away.subdivision
    if cross and home.subdivision == "fcs":
        x, y = np.meshgrid(away_percentile, home_percentile, indexing="ij")
    else:
        x, y = np.meshgrid(home_percentile, away_percentile, indexing="ij")
    return x, y, cross and home.subdivision == "fcs"


def primitive_factor(
    game: Game,
    home: Team,
    away: Team,
    model: Mapping[str, object] | None,
    *,
    component: str,
    home_evidence: Mapping[str, object],
    away_evidence: Mapping[str, object],
) -> np.ndarray:
    """Return a normalized primitive factor or an all-ones fallback."""

    shape = (len(home.prior), len(away.prior))
    pairing = pairing_for(home.subdivision, away.subdivision)
    if model is None or pairing not in SUPPORTED_PRIMITIVE_PAIRINGS:
        return np.ones(shape, dtype=float)
    if str(model["response_kind"]) != component:
        raise ValueError(f"model response {model['response_kind']} does not match {component}")
    evidence = _oriented_evidence(game, home, away, home_evidence, away_evidence)
    plays_diff = evidence["offensive_plays_derived_diff"]
    plays_total = evidence["offensive_plays_derived_total"]
    yards_diff = evidence["total_yards_diff"]
    yards_total = evidence["total_yards_total"]
    required = (plays_diff is not None and plays_total is not None)
    if component == "yards":
        required = required and yards_diff is not None and yards_total is not None
        target = _asinh_scaled(float(yards_diff), YARDS_DIFF_SCALE) if required else None
    elif component == "plays":
        target = _asinh_scaled(float(plays_diff), PLAYS_DIFF_SCALE) if required else None
    else:
        raise ValueError(f"unknown primitive factor: {component}")
    turnover_values = (
        evidence["interceptions_thrown_diff"],
        evidence["interceptions_thrown_total"],
        evidence["fumbles_lost_diff"],
        evidence["fumbles_lost_total"],
    )
    if bool(model["turnover_context"]):
        required = required and all(value is not None for value in turnover_values)
    if not required or target is None:
        return np.ones(shape, dtype=float)

    key = (
        np.asarray(model["beta"], dtype=float).tobytes(),
        float(model["scale"]),
        float(model["df"]),
        str(model["response_kind"]),
        bool(model["turnover_context"]),
        bool(model["rank_signal"]),
        home.subdivision,
        away.subdivision,
        len(home.prior),
        len(away.prior),
        game.home_points,
        game.away_points,
        game.neutral_site,
        _evidence_tuple(home_evidence, away_evidence),
    )
    cached = _FACTOR_CACHE.get(key)
    if cached is not None:
        return cached

    x, y, fcs_home_listing = _rank_grid(home, away)
    x_flat, y_flat = x.ravel(), y.ravel()
    margin = np.full(x_flat.size, oriented_margin(
        home.subdivision, away.subdivision, game.home_points, game.away_points
    ))
    pairing_values = np.full(x_flat.size, pairing)
    neutral = np.full(x_flat.size, float(game.neutral_site))
    fbs_home = np.full(
        x_flat.size,
        float(home.subdivision != away.subdivision and home.subdivision == "fbs" and not game.neutral_site),
    )
    locations = _model_locations(
        model,
        margin=margin,
        pairing=pairing_values,
        neutral=neutral,
        fbs_home=fbs_home,
        x=x_flat,
        y=y_flat,
        plays_diff=np.full(x_flat.size, float(plays_diff)),
        plays_total=np.full(x_flat.size, float(plays_total)),
        yards_total=np.full(x_flat.size, float(yards_total)),
        interceptions_diff=np.full(x_flat.size, float(turnover_values[0])) if turnover_values[0] is not None else np.full(x_flat.size, np.nan),
        interceptions_total=np.full(x_flat.size, float(turnover_values[1])) if turnover_values[1] is not None else np.full(x_flat.size, np.nan),
        fumbles_lost_diff=np.full(x_flat.size, float(turnover_values[2])) if turnover_values[2] is not None else np.full(x_flat.size, np.nan),
        fumbles_lost_total=np.full(x_flat.size, float(turnover_values[3])) if turnover_values[3] is not None else np.full(x_flat.size, np.nan),
    )
    logs = _student_t_logpdf(
        float(target),
        locations,
        float(model["scale"]),
        float(model["df"]),
    )
    factor = np.exp(logs - np.max(logs))
    if fcs_home_listing:
        factor = factor.reshape(len(away.prior), len(home.prior)).T
    else:
        factor = factor.reshape(shape)
    _FACTOR_CACHE[key] = factor
    return factor


def _turnover_context_complete(
    home: Mapping[str, object], away: Mapping[str, object]
) -> bool:
    return all(
        _finite_float(side.get(field)) is not None
        for side in (home, away)
        for field in ("interceptions_thrown", "fumbles_lost")
    )


def _cached_v1_factor(
    game: Game, home: Team, away: Team, likelihood: LikelihoodV1
) -> np.ndarray:
    """Research-local cached equivalent of the frozen V1 factor."""

    location_key = (
        likelihood.beta.tobytes(),
        likelihood.scale,
        likelihood.degrees_of_freedom,
        home.subdivision,
        away.subdivision,
        len(home.prior),
        len(away.prior),
        game.neutral_site,
    )
    locations = _V1_LOCATION_CACHE.get(location_key)
    cross = home.subdivision != away.subdivision
    fcs_home_listing = cross and home.subdivision == "fcs"
    if locations is None:
        home_rank = np.arange(1, len(home.prior) + 1, dtype=float)
        away_rank = np.arange(1, len(away.prior) + 1, dtype=float)
        home_percentile = (home_rank - 0.5) / len(home_rank)
        away_percentile = (away_rank - 0.5) / len(away_rank)
        if fcs_home_listing:
            x, y = np.meshgrid(away_percentile, home_percentile, indexing="ij")
        else:
            x, y = np.meshgrid(home_percentile, away_percentile, indexing="ij")
        pairing = "fbs-fcs" if cross else f"{home.subdivision}-{away.subdivision}"
        matrix = design_matrix(
            x.ravel(),
            y.ravel(),
            np.full(x.size, pairing),
            np.full(x.size, 1.0 - float(game.neutral_site)),
            np.full(x.size, float(game.neutral_site)),
            surface=True,
            fbs_home=np.full(
                x.size,
                float(cross and home.subdivision == "fbs" and not game.neutral_site),
            ),
        )
        locations = matrix @ likelihood.beta
        if fcs_home_listing:
            locations = locations.reshape(len(away.prior), len(home.prior)).T
        else:
            locations = locations.reshape(len(home.prior), len(away.prior))
        _V1_LOCATION_CACHE[location_key] = locations
    margin = oriented_margin(
        home.subdivision,
        away.subdivision,
        game.home_points,
        game.away_points,
    )
    factor_key = (*location_key, margin)
    factor = _V1_FACTOR_CACHE.get(factor_key)
    if factor is None:
        logs = _student_t_logpdf(
            margin,
            locations,
            likelihood.scale,
            likelihood.degrees_of_freedom,
        )
        factor = np.exp(logs - np.max(logs))
        _V1_FACTOR_CACHE[factor_key] = factor
    return factor


def _candidate_factors(
    variant: str,
    game: Game,
    home: Team,
    away: Team,
    models: Mapping[str, Mapping[str, object] | None],
    home_evidence: Mapping[str, object],
    away_evidence: Mapping[str, object],
) -> np.ndarray:
    values = np.ones((len(home.prior), len(away.prior)), dtype=float)
    if variant == "v1":
        return values
    if variant == "a":
        return values * primitive_factor(
            game,
            home,
            away,
            models.get("a_yards"),
            component="yards",
            home_evidence=home_evidence,
            away_evidence=away_evidence,
        )
    if variant not in {"b", "c"}:
        raise ValueError(f"unknown primitive candidate: {variant}")
    use_c = variant == "c" and _turnover_context_complete(home_evidence, away_evidence)
    prefix = "c" if use_c else "b"
    values *= primitive_factor(
        game,
        home,
        away,
        models.get(f"{prefix}_plays"),
        component="plays",
        home_evidence=home_evidence,
        away_evidence=away_evidence,
    )
    values *= primitive_factor(
        game,
        home,
        away,
        models.get(f"{prefix}_yards"),
        component="yards",
        home_evidence=home_evidence,
        away_evidence=away_evidence,
    )
    return values


def infer_posterior_with_primitives(
    teams: list[Team],
    games: Sequence[Game],
    likelihood: LikelihoodV1,
    evidence_by_game: Mapping[str, tuple[Mapping[str, object], Mapping[str, object]]],
    models: Mapping[str, Mapping[str, object] | None],
    *,
    variant: str,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    damping: float = 0.35,
) -> PosteriorResult:
    """Run research BP with V1 margin and optional conditional primitives."""

    by_id = {team.team_id: team for team in teams}
    if len(by_id) != len(teams):
        raise ValueError("team IDs must be unique")
    grouped: dict[tuple[str, str], tuple[np.ndarray, list[str]]] = {}
    for game in games:
        if game.home_id not in by_id or game.away_id not in by_id:
            raise ValueError(f"game {game.game_id} references a team without a prior")
        home, away = by_id[game.home_id], by_id[game.away_id]
        # Candidate multiplication must not mutate the cached V1 array; the
        # same geometry is revisited by A/B/C and by both prior families.
        values = _cached_v1_factor(game, home, away, likelihood).copy()
        evidence = evidence_by_game.get(game.game_id)
        if evidence is not None:
            values *= _candidate_factors(
                variant,
                game,
                home,
                away,
                models,
                evidence[0],
                evidence[1],
            )
        key = tuple(sorted((game.home_id, game.away_id)))
        if (game.home_id, game.away_id) != key:
            values = values.T
        logs = np.log(np.maximum(values, np.finfo(float).tiny))
        if key in grouped:
            previous, game_ids = grouped[key]
            grouped[key] = (previous + logs, [*game_ids, game.game_id])
        else:
            grouped[key] = (logs, [game.game_id])

    factors = [
        (first, second, np.exp(logs - np.max(logs)), tuple(game_ids))
        for (first, second), (logs, game_ids) in sorted(grouped.items())
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

    max_delta = np.inf
    for iteration in range(1, max_iterations + 1):
        updates: dict[tuple[int, str], np.ndarray] = {}
        max_delta = 0.0
        for index, (first, second, factor, _game_ids) in enumerate(factors):
            for target, other, transpose in (
                (first, second, False),
                (second, first, True),
            ):
                belief = by_id[other].prior.copy()
                for neighbor in adjacency[other]:
                    if neighbor != index:
                        belief *= messages[neighbor, other]
                belief /= belief.sum()
                raw = factor.T @ belief if transpose else factor @ belief
                raw = np.maximum(raw, 0.0)
                raw /= raw.sum()
                old = messages[index, target]
                update = damping * old + (1.0 - damping) * raw
                update /= update.sum()
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
        belief /= belief.sum()
        pmfs[team.team_id] = belief
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
