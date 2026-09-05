"""Build independent History H 1.1 and Context C 1.2 annual forecasts.

Backtest rows have outcomes; annual inference rows deliberately do not.  This
keeps future-season construction from accidentally depending on a final rank.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import build_preseason_prior as v1
import build_preseason_prior_v1_1 as h
import numpy as np

from gippyrank.context_prior import (
    EXPLORATORY,
    PRODUCTION_SAFE_BY_CONSTRUCTION,
    PRODUCTION_SAFE_BY_SEMANTICS,
    PRODUCTION_SAFE_WITH_CAVEAT,
    REJECTED,
    AnnualFittedInstance,
    InferenceRow,
    ModelSpecification,
    coach_at_cutoff,
    only_approved,
)
from gippyrank.preseason import (
    DirectRankModel,
    GenericRankPrior,
    TeamSeason,
    historical_rank_features,
    rank_to_z,
)

ROOT = Path(__file__).resolve().parents[1]
PRESEASON, MODELING = (
    ROOT / "data/processed/preseason",
    ROOT / "data/processed/modeling",
)
HISTORY, CONTEXT = PRESEASON / "history", PRESEASON / "context"
TENURES = ROOT / "data/raw/cfbd/preseason/coach_tenures"
TEST_SEASONS, DEV_TRAIN, DEV_VALIDATION = (
    {2022, 2023, 2024, 2025},
    set(range(2004, 2018)),
    set(range(2018, 2022)),
)
H_FEATURES = ["lag2_z_mean", "lag3_z_mean", "long_run_z_mean"]
H_MODEL_NAME = "V1_1_long_run_baseline"
COACH_FEATURES = ["coach_tenure_seasons"]
RECRUITING_FEATURES = [
    "recruiting_class_rank",
    "recruiting_class_points",
    "recruiting_points_2y_mean",
    "recruiting_points_3y_mean",
    "recruiting_points_4y_mean",
    "recruiting_points_trend",
]
TALENT_FEATURES = ["talent_composite"]
RETURNING_TOTAL_FEATURES = ["returning_pct_ppa"]
RETURNING_PASSING_FEATURES = ["returning_pct_passing_ppa"]
RETURNING_COMPONENT_FEATURES = [
    "returning_pct_receiving_ppa",
    "returning_pct_rushing_ppa",
]
RETURNING_FEATURES = [
    *RETURNING_TOTAL_FEATURES,
    *RETURNING_PASSING_FEATURES,
    *RETURNING_COMPONENT_FEATURES,
]
APPROVED_CONTEXT_FEATURES = frozenset(
    [*COACH_FEATURES, *RECRUITING_FEATURES, *TALENT_FEATURES, *RETURNING_FEATURES]
)

FEATURE_PROVENANCE: dict[str, dict[str, object]] = {
    "rank_history": {
        "production_status": PRODUCTION_SAFE_BY_CONSTRUCTION,
        "source": "processed Massey final constituent-rank distributions",
        "historical_coverage": "2003-2025",
        "evidence": "Each H target uses rank distributions from seasons strictly before the target.",
        "caveats": [],
    },
    "coaching_continuity": {
        "production_status": PRODUCTION_SAFE_BY_CONSTRUCTION,
        "source": "CFBD /coaches/tenures",
        "historical_coverage": "dated cached tenure corpus",
        "evidence": "A continuous tenure must prove the coach active by the August 15 cutoff.",
        "caveats": [
            "Coach change remains unresolved when either cutoff cannot be reconstructed; unknown is never no-change."
        ],
    },
    "coach_change": {
        "production_status": EXPLORATORY,
        "source": "CFBD /coaches/tenures",
        "historical_coverage": "many ends lack effective dates",
        "evidence": "An undated target-year end can be in-season, so a safe historical comparison is not available.",
        "caveats": ["Not used in C 1.2."],
    },
    "recruiting_class": {
        "production_status": PRODUCTION_SAFE_BY_SEMANTICS,
        "source": "CFBD /recruiting/teams",
        "source_fields": ["year", "rank", "points"],
        "historical_coverage": "2003-2026; indicators preserve rows with gaps",
        "evidence": "CFBD defines year as recruiting class year. Rank and points evaluate a signed class, not later college performance; that class is finalized before its football season.",
        "caveats": [
            "No archival as-of field is supplied. The absence is not leakage: this is a preseason-semantic class fact, with ordinary source corrections treated as a retrospective-stability caveat."
        ],
    },
    "team_talent_composite": {
        "production_status": PRODUCTION_SAFE_WITH_CAVEAT,
        "source": "CFBD /talent (247Sports Team Talent Composite)",
        "source_fields": ["year", "team", "talent"],
        "historical_coverage": "2015-2026; indicators preserve gaps",
        "evidence": "CFBD describes a season's 247Sports roster recruiting-talent composite, not a target-season performance statistic.",
        "caveats": [
            "Old roster/rating records may be corrected; no endpoint-semantic evidence says target-season games enter the metric."
        ],
    },
    "returning_production": {
        "production_status": PRODUCTION_SAFE_WITH_CAVEAT,
        "source": "CFBD /player/returning",
        "source_fields": [
            "percentPPA",
            "percentPassingPPA",
            "percentReceivingPPA",
            "percentRushingPPA",
        ],
        "historical_coverage": "2014-2026; indicators preserve gaps",
        "evidence": "It is previous-season player production filtered by return status into the named season. Target-season games cannot change its previous-production component.",
        "caveats": [
            "Roster-return corrections may revise old values, but no evidence says revisions incorporate target-season outcomes."
        ],
    },
    "transfers": {
        "production_status": REJECTED,
        "source": "CFBD /player/portal",
        "historical_coverage": "not used",
        "evidence": "Portal entry, destination, enrollment, eligibility, and late moves are genuinely time-varying; no frozen cutoff roster reconstruction exists here.",
        "caveats": ["Not used in C 1.2."],
    },
}


def cutoff_for(season: int) -> str:
    return f"{season}-08-15"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def maybe_float(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def feature_index() -> dict[tuple[int, str, str], dict[str, str]]:
    return {
        (int(x["season"]), x["subdivision"], x["team_id"]): x
        for x in read_csv(PRESEASON / "team_season_features.csv")
    }


def enrich_features(
    season: int,
    subdivision: str,
    team_id: str,
    base: dict[str, float | None],
    index: dict[tuple[int, str, str], dict[str, str]],
) -> dict[str, float | None]:
    result, current = dict(base), index.get((season, subdivision, team_id), {})
    for name in (
        "recruiting_class_rank",
        "recruiting_class_points",
        "talent_composite",
        "returning_pct_ppa",
        "returning_pct_passing_ppa",
        "returning_pct_receiving_ppa",
        "returning_pct_rushing_ppa",
    ):
        result[name] = maybe_float(current.get(name))
    points = [
        maybe_float(
            index.get((season - lag, subdivision, team_id), {}).get(
                "recruiting_class_points"
            )
        )
        for lag in range(4)
    ]
    for years in (2, 3, 4):
        observed = [point for point in points[:years] if point is not None]
        result[f"recruiting_points_{years}y_mean"] = (
            float(np.mean(observed)) if observed else None
        )
    result["recruiting_points_trend"] = (
        points[0] - points[2]
        if points[0] is not None and points[2] is not None
        else None
    )
    return result


def cached_tenures() -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for path in TENURES.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for item in payload:
            team = (item.get("team") or {}).get("school")
            if team:
                result.setdefault(str(team), []).append(item)
    return result


def attach_context(
    rows: list[TeamSeason],
    index: dict[tuple[int, str, str], dict[str, str]],
    tenures: dict[str, list[dict[str, object]]],
) -> tuple[list[TeamSeason], list[dict[str, object]]]:
    result, coverage = [], []
    for row in rows:
        coach = coach_at_cutoff(
            tenures.get(row.team_name, []),
            row.team_name,
            row.season,
            cutoff_for(row.season),
        )
        features = enrich_features(
            row.season, row.subdivision, row.team_id, row.features, index
        )
        features.update(coach.features(None))
        result.append(replace(row, features=features))
        coverage.append(
            {
                "season": row.season,
                "subdivision": row.subdivision,
                "team_id": row.team_id,
                "team_name": row.team_name,
                "context_as_of": cutoff_for(row.season),
                "coach_tenure_available": coach.known_by_cutoff,
                "coach_reason": coach.unavailable_reason,
                **{
                    f"{name}_available": features.get(name) is not None
                    for name in APPROVED_CONTEXT_FEATURES
                },
            }
        )
    return result, coverage


def build_history_prior(
    rows: list[TeamSeason], *, target_season: int, trained_through_season: int
) -> tuple[DirectRankModel, AnnualFittedInstance]:
    if trained_through_season >= target_season:
        raise ValueError("training must end before target")
    return h.fit(
        [row for row in rows if row.season <= trained_through_season], h.CANDIDATES[3]
    ), AnnualFittedInstance(
        "history_prior", "1.1", trained_through_season, target_season
    )


def build_context_prior(
    rows: list[TeamSeason],
    *,
    target_season: int,
    trained_through_season: int,
    context_features: list[str],
    mode: str,
) -> tuple[DirectRankModel, AnnualFittedInstance]:
    if trained_through_season >= target_season:
        raise ValueError("training must end before target")
    only_approved(
        {name: None for name in context_features}, set(APPROVED_CONTEXT_FEATURES)
    )
    location = (
        [*H_FEATURES, *context_features] if mode in {"location", "both"} else H_FEATURES
    )
    scale = (
        [*H_FEATURES, *context_features] if mode in {"scale", "both"} else H_FEATURES
    )
    training = [row for row in rows if row.season <= trained_through_season]
    try:
        model = DirectRankModel.fit(
            training,
            [*H_FEATURES, *context_features],
            penalty=0.25,
            location_feature_names=location,
            scale_feature_names=scale,
        )
    except RuntimeError as error:
        if "ITERATIONS REACHED LIMIT" not in str(error):
            raise
        # A rich but still predeclared candidate occasionally needs additional
        # L-BFGS steps; retrying is deterministic and recorded in diagnostics.
        model = DirectRankModel.fit(
            training,
            [*H_FEATURES, *context_features],
            penalty=0.25,
            location_feature_names=location,
            scale_feature_names=scale,
            optimizer_options={"maxiter": 2000},
        )
    return model, AnnualFittedInstance(
        "context_prior",
        "1.2",
        trained_through_season,
        target_season,
        cutoff_for(target_season),
    )


def predictions(
    model: DirectRankModel, rows: Iterable[TeamSeason], label: str
) -> list[v1.PriorPrediction]:
    return h.make_predictions(model, list(rows), label)


def evaluate_delta(
    reference: list[v1.PriorPrediction], candidate: list[v1.PriorPrediction]
) -> dict[str, object]:
    if {x.key for x in reference} != {x.key for x in candidate}:
        raise ValueError("H/C comparisons require identical team-season keys")
    ref, cand = v1.score_predictions(reference), v1.score_predictions(candidate)
    per_ref = {
        str(year): v1.score_predictions([x for x in reference if x.season == year])
        for year in sorted({x.season for x in reference})
    }
    per_c = {
        str(year): v1.score_predictions([x for x in candidate if x.season == year])
        for year in sorted({x.season for x in candidate})
    }
    return {
        "same_population_keys": [list(x) for x in sorted(x.key for x in reference)],
        "reference": ref,
        "candidate": cand,
        "delta_c_minus_h": {
            "nll": cand["nll"] - ref["nll"],
            "crps": cand["crps"] - ref["crps"],
            "expected_rank_mae": cand["expected_rank_mae"] - ref["expected_rank_mae"],
            "median_rank_mae": cand["median_rank_mae"] - ref["median_rank_mae"],
            "interval_80_coverage": cand["interval_80_coverage"]
            - ref["interval_80_coverage"],
            "interval_80_average_width": cand["interval_80_average_width"]
            - ref["interval_80_average_width"],
        },
        "per_season": {
            year: {
                "h": per_ref[year],
                "c": per_c[year],
                "delta_nll": per_c[year]["nll"] - per_ref[year]["nll"],
            }
            for year in per_ref
        },
        "paired_season_resampling": h.paired_bootstrap(reference, candidate),
    }


def candidate_definitions() -> list[tuple[str, list[str]]]:
    return [
        ("C1_coach_tenure", COACH_FEATURES),
        ("C2_recruiting", RECRUITING_FEATURES),
        ("C3_team_talent", TALENT_FEATURES),
        ("C4_returning_total", RETURNING_TOTAL_FEATURES),
        ("C4_returning_passing", RETURNING_PASSING_FEATURES),
        ("C4_returning_components", RETURNING_COMPONENT_FEATURES),
        ("C5_recruiting_returning", [*RECRUITING_FEATURES, *RETURNING_FEATURES]),
        ("C6_talent_returning", [*TALENT_FEATURES, *RETURNING_FEATURES]),
        (
            "C6_recruiting_talent_returning",
            [*RECRUITING_FEATURES, *TALENT_FEATURES, *RETURNING_FEATURES],
        ),
        (
            "C6_roster_context_with_coaching",
            [
                *COACH_FEATURES,
                *RECRUITING_FEATURES,
                *TALENT_FEATURES,
                *RETURNING_FEATURES,
            ],
        ),
    ]


def select_candidate(development: dict[str, dict[str, object]]) -> str:
    qualifying = [
        name
        for name, item in development.items()
        if item["comparison"]["delta_c_minus_h"]["nll"] <= -0.005
        and item["comparison"]["delta_c_minus_h"]["crps"] <= 0
        and item["comparison"]["paired_season_resampling"][
            "fraction_candidate_better_nll"
        ]
        >= 0.75
    ]
    return (
        min(
            qualifying,
            key=lambda name: development[name]["comparison"]["delta_c_minus_h"]["nll"],
        )
        if qualifying
        else "C0_history_only"
    )


def raw_history_predictions(
    rows: list[TeamSeason], cold: list[v1.ColdStartSeason]
) -> list[v1.PriorPrediction]:
    return h.join_targets(h.read_predictions(H_MODEL_NAME), rows, cold)


def context_rows(
    history: list[v1.PriorPrediction],
    updated: list[v1.PriorPrediction],
    coverage: dict[tuple[int, str, str], dict[str, object]],
) -> list[dict[str, object]]:
    update = {x.key: x for x in updated}
    result = []
    for baseline in history:
        choice = update.get(baseline.key, baseline)
        row = choice.csv_row()
        row.update(
            {
                "model_family": "context_prior",
                "spec_version": "1.2",
                "context_as_of": coverage.get(baseline.key, {}).get(
                    "context_as_of", cutoff_for(baseline.season)
                ),
                "context_applied": baseline.key in update,
                "fallback_reason": None
                if baseline.key in update
                else "history_cold_start",
            }
        )
        result.append(row)
    return result


def inference_rows(
    target_season: int,
    trained_through_season: int,
    index: dict[tuple[int, str, str], dict[str, str]],
    tenures: dict[str, list[dict[str, object]]],
) -> list[InferenceRow]:
    """Use an FBS universe plus completed outcomes only; target outcomes are never read."""
    outcomes = {
        (int(x["season"]), x["subdivision"], x["team_id"]): x
        for x in read_csv(MODELING / "team_season_rank_distributions.csv")
        if int(x["season"]) <= trained_through_season
    }
    universe = [
        x
        for (season, division, _), x in index.items()
        if season == target_season and division == "fbs"
    ]
    result = []
    for current in universe:
        team_id, team_name = current["team_id"], current["team_name"]
        prior = outcomes.get((trained_through_season, "fbs", team_id))
        prior_ranks = v1.valid_ranks(prior) if prior else np.asarray([], dtype=int)
        base, lag_zs, history = {}, [], []
        for lag in range(2, 4):
            old = outcomes.get((target_season - lag, "fbs", team_id))
            old_ranks = v1.valid_ranks(old) if old else np.asarray([], dtype=int)
            values = (
                rank_to_z(old_ranks, int(old["team_population"]))
                if len(old_ranks)
                else np.asarray([], dtype=float)
            )
            base[f"lag{lag}_z_mean"] = float(np.mean(values)) if len(values) else None
            lag_zs.append(tuple(float(x) for x in values))
        for year in range(2002, target_season):
            old = outcomes.get((year, "fbs", team_id))
            ranks = v1.valid_ranks(old) if old else np.asarray([], dtype=int)
            if len(ranks):
                history.append(rank_to_z(ranks, int(old["team_population"])))
        if len(prior_ranks):
            lag1 = rank_to_z(prior_ranks, int(prior["team_population"]))
            base.update(historical_rank_features(lag1, tuple(history)))
            lag1_tuple, cold = tuple(float(x) for x in lag1), None
        else:
            lag1_tuple = None
            cold = (
                "fcs_to_fbs_transition"
                if outcomes.get((trained_through_season, "fcs", team_id))
                else "no_prior_rank_distribution"
            )
        features = enrich_features(target_season, "fbs", team_id, base, index)
        features.update(
            coach_at_cutoff(
                tenures.get(team_name, []),
                team_name,
                target_season,
                cutoff_for(target_season),
            ).features(None)
        )
        row = InferenceRow(
            target_season,
            "fbs",
            team_id,
            team_name,
            len(universe),
            lag1_tuple,
            tuple(lag_zs),
            features,
            cold,
        )
        row.require_no_target()
        result.append(row)
    return result


def future_predictions(
    cold: list[v1.ColdStartSeason],
    future: list[InferenceRow],
    h_model: DirectRankModel,
    c_model: DirectRankModel | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    promotion_rows = [x for x in v1.cold_start_teams(cold) if x.season <= 2025]
    promotion = DirectRankModel.fit(promotion_rows, [], penalty=0.25)
    generic_source = [
        x
        for x in cold
        if x.subdivision == "fbs"
        and x.reason == "no_prior_rank_distribution"
        and x.season <= 2025
    ]
    generic = GenericRankPrior.fit(
        [
            TeamSeason(
                x.season,
                x.subdivision,
                x.team_id,
                x.team_name,
                x.population,
                np.asarray([0.0]),
                x.target_z,
                x.target_ranks,
                {},
            )
            for x in generic_source
        ]
    )
    outcomes = {
        (int(x["season"]), x["subdivision"], x["team_id"]): x
        for x in read_csv(MODELING / "team_season_rank_distributions.csv")
        if int(x["season"]) <= 2025
    }
    h_out, c_out = [], []
    for row in future:
        if row.lag1_z is not None:
            pmf = h_model.pmf(row.features, np.asarray(row.lag1_z), row.population)
            location, scale, method = (
                h_model.conditional_parameters(row.features, np.asarray(row.lag1_z))[0],
                h_model.conditional_parameters(row.features, np.asarray(row.lag1_z))[1],
                "same_subdivision_lag1",
            )
        elif row.cold_start_reason == "fcs_to_fbs_transition":
            old = outcomes[(2025, "fcs", row.team_id)]
            cross = rank_to_z(v1.valid_ranks(old), int(old["team_population"]))
            pmf, location, scale, method = (
                promotion.pmf({}, cross, row.population),
                promotion.conditional_parameters({}, cross)[0],
                promotion.conditional_parameters({}, cross)[1],
                "learned_fcs_to_fbs_transition",
            )
        else:
            pmf, location, scale, method = (
                generic.pmf(row.population),
                np.asarray([generic.location]),
                generic.scale,
                "generic_fbs_cold_start",
            )
        h_row = {
            "season": row.season,
            "subdivision": row.subdivision,
            "team_id": row.team_id,
            "team_name": row.team_name,
            "model_family": "history_prior",
            "spec_version": "1.1",
            "trained_through_season": 2025,
            "pmf": json.dumps(
                [round(float(x), 12) for x in pmf], separators=(",", ":")
            ),
            "prior_method": method,
            "conditional_location_mean": float(np.mean(location)),
            "predictive_scale": scale,
        }
        h_out.append(h_row)
        if c_model is not None and row.lag1_z is not None:
            c_pmf = c_model.pmf(row.features, np.asarray(row.lag1_z), row.population)
            c_location, c_scale = c_model.conditional_parameters(
                row.features, np.asarray(row.lag1_z)
            )
            c_out.append(
                {
                    **h_row,
                    "model_family": "context_prior",
                    "spec_version": "1.2",
                    "context_as_of": cutoff_for(row.season),
                    "pmf": json.dumps(
                        [round(float(x), 12) for x in c_pmf], separators=(",", ":")
                    ),
                    "conditional_location_mean": float(np.mean(c_location)),
                    "predictive_scale": c_scale,
                    "context_applied": True,
                    "fallback_reason": None,
                }
            )
        else:
            c_out.append(
                {
                    **h_row,
                    "model_family": "context_prior",
                    "spec_version": "1.2",
                    "context_as_of": cutoff_for(row.season),
                    "context_applied": False,
                    "fallback_reason": "history_cold_start",
                }
            )
    return h_out, c_out


def render_report(report: dict[str, object]) -> None:
    lines = [
        "# GippyRank4 Preseason Context Prior V1.2",
        "",
        "H (`history_prior` 1.1) remains the unchanged rank-history-only sibling. C (`context_prior` 1.2) adds only approved preseason context. 2022--2025 is a backtest, not a single annual fit.",
        "",
        "## Provenance",
        "",
        "| Family | Classification | Evidence |",
        "|---|---|---|",
    ]
    for name, item in report["provenance"].items():
        lines.append(f"| {name} | {item['production_status']} | {item['evidence']} |")
    lines += [
        "",
        "## Development ablations (2018--2021, exact same H/C keys)",
        "",
        "| Candidate | Mode | N | H NLL | C NLL | ΔNLL | H CRPS | C CRPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    c0 = report["development"]["C0_history_only"]["comparison"]
    lines.append(
        f"| C0_history_only | exact H | {c0['reference']['n_team_seasons']} | "
        f"{c0['reference']['nll']:.4f} | {c0['candidate']['nll']:.4f} | "
        f"{c0['delta_c_minus_h']['nll']:.4f} | {c0['reference']['crps']:.4f} | "
        f"{c0['candidate']['crps']:.4f} |"
    )
    for name, item in report["development"]["candidates"].items():
        score = item["comparison"]
        lines.append(
            f"| {name} | {item['mode']} | {score['reference']['n_team_seasons']} | {score['reference']['nll']:.4f} | {score['candidate']['nll']:.4f} | {score['delta_c_minus_h']['nll']:.4f} | {score['reference']['crps']:.4f} | {score['candidate']['crps']:.4f} |"
        )
    final = report["final_test"]["all_fbs"]
    lines += [
        "",
        "## Frozen selection",
        "",
        f"Selected C: **{report['selection']['selected']}**. Before test data, the rule required ΔNLL ≤ -0.005, non-worse CRPS, and ≥75% favorable season-bootstrap resamples.",
        "",
        "## Untouched 2022--2025 H vs C",
        "",
        f"N={final['reference']['n_team_seasons']}; H NLL {final['reference']['nll']:.4f}, C NLL {final['candidate']['nll']:.4f}, ΔNLL {final['delta_c_minus_h']['nll']:.4f}; H CRPS {final['reference']['crps']:.4f}, C CRPS {final['candidate']['crps']:.4f}.",
        "",
        "| Season | H NLL | C NLL | ΔNLL |",
        "|---|---:|---:|---:|",
    ]
    for year, item in final["per_season"].items():
        lines.append(
            f"| {year} | {item['h']['nll']:.4f} | {item['c']['nll']:.4f} | {item['delta_nll']:.4f} |"
        )
    lines += [
        "",
        "## Annual inference",
        "",
        "Annual 2026 inputs are outcome-free rows: completed history through 2025, the frozen 2026 FBS universe, and approved preseason context only. Missing context is handled with training-only imputation and indicators; rank-history cold starts retain exact H fallback.",
        "",
    ]
    (CONTEXT / "context_prior_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows, cold, _ = v1.load_rows()
    fbs = [row for row in rows if row.subdivision == "fbs"]
    index, tenures = feature_index(), cached_tenures()
    contextual, coverage = attach_context(fbs, index, tenures)
    train, validation = (
        [row for row in contextual if row.season in DEV_TRAIN],
        [row for row in contextual if row.season in DEV_VALIDATION],
    )
    print("fitting development H", flush=True)
    h_validation = predictions(h.fit(train, h.CANDIDATES[3]), validation, "H")
    development: dict[str, dict[str, object]] = {}
    for family, features in candidate_definitions():
        for mode in ("location", "scale", "both"):
            name = f"{family}_{mode}"
            print(f"fitting {name}", flush=True)
            model, instance = build_context_prior(
                train,
                target_season=2022,
                trained_through_season=2017,
                context_features=features,
                mode=mode,
            )
            development[name] = {
                "family": family,
                "features": features,
                "mode": mode,
                "fit": instance.metadata(),
                "fit_metadata": model.metadata(),
                "comparison": evaluate_delta(
                    h_validation, predictions(model, validation, name)
                ),
            }
    selected = select_candidate(development)
    selected_features = (
        [] if selected == "C0_history_only" else list(development[selected]["features"])
    )
    selected_mode = (
        "both" if selected == "C0_history_only" else str(development[selected]["mode"])
    )
    history = raw_history_predictions(rows, cold)
    test = [row for row in contextual if row.season in TEST_SEASONS]
    if selected == "C0_history_only":
        updated, c_model, final_fit = [], None, {"status": "C0 exact H selected"}
    else:
        print(f"fitting final {selected}", flush=True)
        c_model, _instance = build_context_prior(
            contextual,
            target_season=2022,
            trained_through_season=2021,
            context_features=selected_features,
            mode=selected_mode,
        )
        updated, final_fit = (
            predictions(c_model, test, selected),
            {
                "kind": "evaluation/backtest",
                "trained_through_season": 2021,
                "target_seasons": sorted(TEST_SEASONS),
                "context_as_of": "target-specific August 15 date carried by each PMF",
                "model": c_model.metadata(),
            },
        )
    changed = {x.key: x for x in updated}
    all_context = [changed.get(x.key, x) for x in history]
    h_spec = ModelSpecification(
        "history_prior",
        "1.1",
        tuple(H_FEATURES),
        "normal",
        0.25,
        "full t-1 empirical quadrature; t-2/t-3 summaries",
    ).metadata()
    c_spec = ModelSpecification(
        "context_prior",
        "1.2",
        (*H_FEATURES, *selected_features),
        "normal",
        0.25,
        "full t-1 empirical quadrature; t-2/t-3 summaries",
    ).metadata()
    future = inference_rows(2026, 2025, index, tenures)
    future_h_model, future_h_instance = build_history_prior(
        fbs, target_season=2026, trained_through_season=2025
    )
    future_c_model, future_c_fit = None, {"status": "C0 exact H selected"}
    if selected != "C0_history_only":
        future_c_model, instance = build_context_prior(
            contextual,
            target_season=2026,
            trained_through_season=2025,
            context_features=selected_features,
            mode=selected_mode,
        )
        future_c_fit = {**instance.metadata(), "model": future_c_model.metadata()}
    future_h, future_c = future_predictions(
        cold, future, future_h_model, future_c_model
    )
    if len(future_h) != len(future) or len(future_c) != len(future):
        raise ValueError("annual inference lacks FBS coverage")
    report = {
        "history_specification": h_spec,
        "context_specification": {
            **c_spec,
            "approved_context_features": selected_features,
            "missing_context_behavior": "training-only median imputation plus indicators; H fallback for rank-history cold starts",
        },
        "provenance": FEATURE_PROVENANCE,
        "development": {
            "population": "2018-2021 FBS regular lag-1 rows",
            "C0_history_only": {
                "comparison": evaluate_delta(h_validation, h_validation)
            },
            "candidates": development,
        },
        "selection": {
            "selected": selected,
            "rule": "ΔNLL <= -0.005, ΔCRPS <= 0, bootstrap NLL win rate >= .75; choose lowest qualifying ΔNLL else C0",
            "development_train": [2004, 2017],
            "development_validation": [2018, 2021],
            "untouched_test": sorted(TEST_SEASONS),
        },
        "final_test": {
            "all_fbs": evaluate_delta(history, all_context),
            "n_all_fbs": len(all_context),
            "annual_fit": final_fit,
        },
        "annual_inference": {
            "history_fit": {
                **future_h_instance.metadata(),
                "model": future_h_model.metadata(),
            },
            "context_fit": future_c_fit,
            "target_universe": 2026,
            "n_fbs": len(future),
            "no_2026_game_outcomes_used": True,
        },
        "context_coverage": {
            str(year): {"n": count}
            for year, count in Counter(x["season"] for x in coverage).items()
        },
    }
    raw = [
        x
        for x in read_csv(PRESEASON / "rank_prior_predictions.csv")
        if x["model"] == H_MODEL_NAME
        and x["subdivision"] == "fbs"
        and int(x["season"]) in TEST_SEASONS
    ]
    write_csv(
        HISTORY / "predictions.csv",
        [{**x, "model_family": "history_prior", "spec_version": "1.1"} for x in raw],
    )
    write_json(HISTORY / "model_spec.json", h_spec)
    write_json(
        HISTORY / "fitted_backtest.json",
        {
            "kind": "evaluation/backtest",
            "target_seasons": sorted(TEST_SEASONS),
            "trained_through_season": 2021,
        },
    )
    write_csv(
        CONTEXT / "predictions.csv",
        context_rows(
            history,
            updated,
            {
                tuple(x[k] for k in ("season", "subdivision", "team_id")): x
                for x in coverage
            },
        ),
    )
    write_json(CONTEXT / "model_spec.json", c_spec)
    write_json(CONTEXT / "feature_provenance.json", FEATURE_PROVENANCE)
    write_json(CONTEXT / "model_report.json", report)
    write_json(CONTEXT / "evaluation.json", report["final_test"])
    write_csv(CONTEXT / "context_coverage.csv", coverage)
    write_csv(HISTORY / "annual/2026/predictions.csv", future_h)
    write_json(
        HISTORY / "annual/2026/fitted_instance.json",
        report["annual_inference"]["history_fit"],
    )
    write_csv(CONTEXT / "annual/2026/predictions.csv", future_c)
    write_json(
        CONTEXT / "annual/2026/fitted_instance.json",
        report["annual_inference"]["context_fit"],
    )
    render_report(report)
    print(
        json.dumps(
            {
                "selected": selected,
                "backtest_h": len(history),
                "backtest_c": len(all_context),
                "2026_h": len(future_h),
                "2026_c": len(future_c),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
