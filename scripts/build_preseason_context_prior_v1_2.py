"""Build the permanent History (H) and Context (C) preseason prior artifacts.

H is copied from the frozen V1.1 production PMFs; C is a sibling family.  The
only context family admitted for this V1.2 build is dated coaching continuity.
All other cached offseason values are explicitly audited but fail closed.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import build_preseason_prior as v1
import build_preseason_prior_v1_1 as h

from gippyrank.context_prior import (
    EXPLORATORY,
    PRODUCTION_SAFE,
    RECONSTRUCTABLE_SAFE,
    REJECTED,
    ModelIdentity,
    coach_at_cutoff,
)
from gippyrank.preseason import DirectRankModel, TeamSeason

ROOT = Path(__file__).resolve().parents[1]
PRESEASON = ROOT / "data/processed/preseason"
HISTORY = PRESEASON / "history"
CONTEXT = PRESEASON / "context"
TENURES = ROOT / "data/raw/cfbd/preseason/coach_tenures"
TEST_SEASONS = {2022, 2023, 2024, 2025}
DEV_TRAIN = set(range(2004, 2018))
DEV_VALIDATION = set(range(2018, 2022))
H_FEATURES = ["lag2_z_mean", "lag3_z_mean", "long_run_z_mean"]
# An undated end year cannot prove that a prior coach was still employed at the
# previous preseason cutoff.  Coach tenure itself is dated enough to use; a
# coach-change indicator is deliberately not smuggled in as an assumed zero.
COACH_FEATURES = ["coach_tenure_seasons"]
H_MODEL_NAME = "V1_1_long_run_baseline"
APPROVED_CONTEXT_FEATURES = frozenset(COACH_FEATURES)


FEATURE_PROVENANCE: dict[str, dict[str, object]] = {
    "rank_history": {
        "production_status": PRODUCTION_SAFE,
        "source": "processed Massey final constituent-rank distributions",
        "historical_coverage": "2003-2025 final outcomes",
        "intended_semantics": "completed-season rank information only",
        "effective_preseason_cutoff": "target season kickoff; all inputs are earlier completed seasons",
        "retrospective_change_risk": "none after final-data normalization",
        "evidence": "H V1.1 feature construction excludes the target season",
        "caveats": [],
    },
    "coaching_continuity": {
        "production_status": RECONSTRUCTABLE_SAFE,
        "source": "CFBD /coaches/tenures, one cached raw response per team",
        "source_fields": [
            "coach.id",
            "startYear",
            "endYear",
            "hireDate",
            "effectiveStart",
            "effectiveEnd",
        ],
        "historical_coverage": "depends on cached team responses; CFBD exposes continuous historical tenures",
        "intended_semantics": "head-coach identity and tenure at the target cutoff",
        "effective_preseason_cutoff": "August 15 of each target season",
        "retrospective_change_risk": "source can correct historical tenure records; raw payloads are cached before processing",
        "evidence": "CFBD documentation labels this endpoint continuous head-coaching tenures and exposes dated start/end fields",
        "caveats": [
            "A target-season start without a date by cutoff is missing.",
            "An undated target-season end is missing because it could represent an in-season change.",
        ],
    },
    "coach_change": {
        "production_status": EXPLORATORY,
        "source": "CFBD /coaches/tenures",
        "historical_coverage": "No complete cutoff-safe historical comparison in the cached tenure payloads",
        "intended_semantics": "head-coach identity differs from the prior preseason",
        "effective_preseason_cutoff": "August 15 of target and prior seasons",
        "retrospective_change_risk": "high when a tenure end has no effective date",
        "evidence": "The current tenure corpus supplies many undated effective ends; strict filtering leaves no observed change values.",
        "caveats": ["Not used in C V1.2; unknown is never converted to no change."],
    },
    "recruiting_class": {
        "production_status": EXPLORATORY,
        "source": "CFBD /recruiting/teams cached annual rank and points",
        "source_fields": ["year", "rank", "points"],
        "historical_coverage": "2003-2026, incomplete across teams and years",
        "intended_semantics": "annual class ranking and aggregate points",
        "effective_preseason_cutoff": "not demonstrated by the cached response",
        "retrospective_change_risk": "unknown; endpoint has no finalization date or snapshot version",
        "evidence": "endpoint documents class year and team ranks, but not historical as-of timestamps",
        "caveats": ["Not admitted merely because a class ordinarily signs before kickoff."],
    },
    "team_talent_composite": {
        "production_status": EXPLORATORY,
        "source": "CFBD /talent (247Sports Team Talent Composite)",
        "source_fields": ["year", "team", "talent"],
        "historical_coverage": "2015-2026; coverage varies materially in recent seasons",
        "intended_semantics": "season roster talent composite",
        "effective_preseason_cutoff": "not demonstrated by cached annual values",
        "retrospective_change_risk": "unknown; no snapshot date/version in payload",
        "evidence": "CFBD documents a season rating but not a preseason snapshot or immutability guarantee",
        "caveats": ["Excluded from V1.2 production."],
    },
    "returning_production": {
        "production_status": EXPLORATORY,
        "source": "CFBD /player/returning",
        "source_fields": [
            "percentPPA",
            "percentPassingPPA",
            "percentReceivingPPA",
            "percentRushingPPA",
            "usage",
        ],
        "historical_coverage": "2014-2026 FBS-oriented payloads",
        "intended_semantics": "target-season returning PPA and usage shares",
        "effective_preseason_cutoff": "not demonstrated by cached response",
        "retrospective_change_risk": "unknown; no archived roster/cutoff snapshot",
        "evidence": "endpoint documents team-season metrics but no as-of date",
        "caveats": ["Overall and passing/QB proxies remain exploratory."],
    },
    "transfers": {
        "production_status": REJECTED,
        "source": "CFBD /player/portal (not cached for this corpus)",
        "source_fields": ["origin", "destination", "transferDate", "eligibility"],
        "historical_coverage": "not acquired",
        "intended_semantics": "portal movement into a target roster",
        "effective_preseason_cutoff": "would require a dated roster/eligibility reconstruction",
        "retrospective_change_risk": "high: destination, eligibility, and roster status can change",
        "evidence": "the endpoint has transfer dates but no preserved target-roster snapshot in this repository",
        "caveats": ["Cleanly omitted rather than inferred from current roster state."],
    },
}


def cutoff_for(season: int) -> str:
    """A fixed, conservative date before ordinary FBS kickoff."""
    return f"{season}-08-15"


def build_history_prior(
    rows: list[TeamSeason], *, target_season: int, trained_through_season: int
) -> tuple[DirectRankModel, ModelIdentity]:
    """Fit H without binding its frozen specification to one calendar year."""
    if trained_through_season >= target_season:
        raise ValueError("history training must end before its target season")
    training = [row for row in rows if row.season <= trained_through_season]
    return (
        h.fit(training, h.CANDIDATES[3]),
        ModelIdentity(
            "history_prior", "1.1", trained_through_season, target_season
        ),
    )


def build_context_prior(
    rows: list[TeamSeason],
    *,
    target_season: int,
    trained_through_season: int,
    context_as_of: str,
    context_features: list[str],
) -> tuple[DirectRankModel, ModelIdentity]:
    """Fit C from cached cutoff-safe features with a fail-closed feature gate."""
    if trained_through_season >= target_season:
        raise ValueError("context training must end before its target season")
    if not set(context_features) <= APPROVED_CONTEXT_FEATURES:
        raise ValueError("context fit requested a non-production-safe feature")
    training = [
        row
        for row in rows
        if row.season <= trained_through_season
        and all(row.features.get(name) is not None for name in context_features)
    ]
    return (
        DirectRankModel.fit(training, [*H_FEATURES, *context_features], penalty=0.25),
        ModelIdentity(
            "context_prior",
            "1.2",
            trained_through_season,
            target_season,
            context_as_of,
        ),
    )


def read_prediction_rows() -> list[dict[str, str]]:
    with (PRESEASON / "rank_prior_predictions.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cached_tenures() -> dict[str, list[dict[str, Any]]]:
    """Read only cached raw source data; fitting never makes network calls."""
    by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(TENURES.glob("*.json")):
        if path.name == "manifest.json":
            continue
        for tenure in json.loads(path.read_text(encoding="utf-8")):
            team = (tenure.get("team") or {}).get("school")
            if isinstance(team, str):
                by_team[team].append(tenure)
    return by_team


def with_coach_context(
    rows: list[TeamSeason], tenures: dict[str, list[dict[str, Any]]]
) -> tuple[list[TeamSeason], list[dict[str, object]]]:
    """Attach safe coaching features without changing historical H features."""
    result: list[TeamSeason] = []
    coverage: list[dict[str, object]] = []
    for row in rows:
        current = coach_at_cutoff(
            tenures.get(row.team_name, []), row.team_name, row.season, cutoff_for(row.season)
        )
        prior = coach_at_cutoff(
            tenures.get(row.team_name, []),
            row.team_name,
            row.season - 1,
            cutoff_for(row.season - 1),
        )
        features = dict(row.features)
        features.update(current.features(prior))
        result.append(replace(row, features=features))
        coverage.append(
            {
                "season": row.season,
                "subdivision": row.subdivision,
                "team_id": row.team_id,
                "team_name": row.team_name,
                "context_as_of": cutoff_for(row.season),
                "coach_context_available": current.known_by_cutoff,
                "coach_change_available": current.known_by_cutoff
                and prior.known_by_cutoff,
                "current_coach_unavailable_reason": current.unavailable_reason,
                "prior_coach_unavailable_reason": prior.unavailable_reason,
            }
        )
    return result, coverage


def has_complete_coach_context(row: TeamSeason) -> bool:
    return all(row.features.get(name) is not None for name in COACH_FEATURES)


def prediction_with_model(
    model: DirectRankModel, rows: list[TeamSeason], label: str
) -> list[v1.PriorPrediction]:
    return h.make_predictions(model, rows, label)


def per_season(predictions: list[v1.PriorPrediction]) -> dict[str, object]:
    return {
        str(season): v1.score_predictions(
            [prediction for prediction in predictions if prediction.season == season]
        )
        for season in sorted({prediction.season for prediction in predictions})
    }


def context_prediction_rows(
    history: list[v1.PriorPrediction],
    updated: list[v1.PriorPrediction],
    coverage: dict[tuple[int, str, str], dict[str, object]],
) -> list[dict[str, object]]:
    changed = {prediction.key: prediction for prediction in updated}
    output = []
    for prediction in history:
        replacement = changed.get(prediction.key, prediction)
        item = replacement.csv_row()
        row_coverage = coverage.get(
            prediction.key,
            {
                "context_as_of": cutoff_for(prediction.season),
                "current_coach_unavailable_reason": "established_history_cold_start",
                "prior_coach_unavailable_reason": None,
            },
        )
        item.update(
            {
                "model_family": "context_prior",
                "spec_version": "1.2",
                "context_as_of": row_coverage["context_as_of"],
                "context_applied": prediction.key in changed,
                "fallback_reason": None
                if prediction.key in changed
                else row_coverage["current_coach_unavailable_reason"]
                or row_coverage["prior_coach_unavailable_reason"]
                or "frozen_history_prior",
            }
        )
        output.append(item)
    return output


def history_prediction_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output = []
    for row in rows:
        item: dict[str, object] = dict(row)
        item.update({"model_family": "history_prior", "spec_version": "1.1"})
        output.append(item)
    return output


def evaluate_delta(
    reference: list[v1.PriorPrediction], candidate: list[v1.PriorPrediction]
) -> dict[str, object]:
    reference_metrics = v1.score_predictions(reference)
    candidate_metrics = v1.score_predictions(candidate)
    return {
        "reference": reference_metrics,
        "candidate": candidate_metrics,
        "delta_c_minus_h": {
            "nll": candidate_metrics["nll"] - reference_metrics["nll"],
            "crps": candidate_metrics["crps"] - reference_metrics["crps"],
            "expected_rank_mae": candidate_metrics["expected_rank_mae"]
            - reference_metrics["expected_rank_mae"],
            "median_rank_mae": candidate_metrics["median_rank_mae"]
            - reference_metrics["median_rank_mae"],
            "interval_80_coverage": candidate_metrics["interval_80_coverage"]
            - reference_metrics["interval_80_coverage"],
            "interval_80_average_width": candidate_metrics["interval_80_average_width"]
            - reference_metrics["interval_80_average_width"],
            **{
                f"top{cutoff}_brier": candidate_metrics[f"top{cutoff}"]["reliability"][
                    "brier_score"
                ]
                - reference_metrics[f"top{cutoff}"]["reliability"]["brier_score"]
                for cutoff in (5, 10, 25)
            },
        },
        "per_season": {
            season: {
                "h": per_season(reference)[season],
                "c": per_season(candidate)[season],
                "delta_nll": per_season(candidate)[season]["nll"]
                - per_season(reference)[season]["nll"],
            }
            for season in per_season(reference)
        },
        "paired_season_resampling": h.paired_bootstrap(reference, candidate),
    }


def render_report(report: dict[str, object]) -> None:
    selection = report["selection"]
    test = report["final_test"]["all_fbs"]
    observed = report["final_test"]["coach_observed"]
    development = report["development"]
    candidate = development["C1_coach_tenure_location_and_scale"]
    coverage = report["context_coverage"]
    lines = [
        "# GippyRank4 Preseason Context Prior V1.2",
        "",
        "## Architecture",
        "",
        "H (`history_prior` 1.1) remains the rank-history-only forecast. C (`context_prior` 1.2) is a sibling forecast that starts from H's historical features and may add only context fields admitted by the provenance gate. Neither family overwrites the other's PMFs or metadata.",
        "",
        "H is frozen as full t-1 constituent-rank uncertainty, t-2/t-3 transformed-rank summaries, long-run program history, a heteroscedastic Normal predictive distribution, analytically integrated discrete PMFs, and the established FBS cold-start fallback. C uses the same historical foundation; it never consumes H's expected rank as a synthetic feature.",
        "",
        "## Provenance decision",
        "",
        "Coach tenure is reconstructable-safe when a cached CFBD continuous tenure proves the target coach by the August 15 cutoff. Coach change remains timing-uncertain because many tenure ends are undated. Recruiting, Team Talent Composite, and returning production remain timing-uncertain because their cached annual API payloads lack archival as-of timestamps. Transfers are rejected: no dated target-roster reconstruction is cached.",
        "",
        "| Family | Status | Coverage | Production decision |",
        "|---|---|---|---|",
    ]
    labels = {
        "rank_history": "Rank history",
        "coaching_continuity": "Coach tenure",
        "coach_change": "Coach change",
        "recruiting_class": "Recruiting",
        "team_talent_composite": "Team Talent Composite",
        "returning_production": "Returning production / QB proxy",
        "transfers": "Transfers",
    }
    for name, label in labels.items():
        item = report["provenance"][name]
        lines.append(
            f"| {label} | {item['production_status']} | {item['historical_coverage']} | {item['caveats'][0] if item['caveats'] else 'Retained'} |"
        )
    lines += [
        "",
        "## Context coverage and missingness",
        "",
        "Missing or ambiguous context is never imputed as zero. The selected C PMF falls back exactly to the frozen H PMF. This applies to every FBS cold start as well as to an unavailable tenure row.",
        "",
        "| Season | Dated tenure available | Missing / ambiguous |",
        "|---|---:|---:|",
    ]
    for season in sorted(coverage, key=int):
        if int(season) >= 2018:
            item = coverage[season]
            lines.append(
                f"| {season} | {item['available']} | {item['missing_or_ambiguous']} |"
            )
    lines += [
        "",
        "## Development ablations",
        "",
        "| Candidate | Population | NLL | ΔNLL vs H | CRPS | ΔCRPS vs H |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if "candidate" in candidate:
        c_metrics = candidate["candidate"]
        deltas = candidate["delta_c_minus_h"]
        h_metrics = candidate["reference"]
        lines.extend(
            [
                f"| C0 = H | {h_metrics['n_team_seasons']} | {h_metrics['nll']:.4f} | 0.0000 | {h_metrics['crps']:.4f} | 0.0000 |",
                f"| C1 = H + coach tenure (location and scale) | {c_metrics['n_team_seasons']} | {c_metrics['nll']:.4f} | {deltas['nll']:.4f} | {c_metrics['crps']:.4f} | {deltas['crps']:.4f} |",
            ]
        )
    else:
        lines.append(f"| C1 = H + coach tenure | — | — | — | — | {candidate['status']} |")
    lines += [
        "",
        "## Frozen C selection",
        "",
        f"Selected candidate: **{selection['selected']}**. Development selection used only 2004–2017 training and 2018–2021 validation. The rule required ΔNLL ≤ -0.01 and at least 80% favorable descriptive season resamples; otherwise C falls back to the frozen H PMF.",
        "",
        "## Untouched 2022–2025 comparison",
        "",
        "| Population | H NLL | C NLL | ΔNLL (C − H) | H CRPS | C CRPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, item in (("All FBS", test), ("Coach-observed subset", observed)):
        lines.append(
            f"| {label} | {item['reference']['nll']:.4f} | {item['candidate']['nll']:.4f} | {item['delta_c_minus_h']['nll']:.4f} | {item['reference']['crps']:.4f} | {item['candidate']['crps']:.4f} |"
        )
    lines += [
        "",
        "Because C0 was selected, the final frozen C PMF is exactly H for all 534 untouched FBS team-seasons: ΔNLL, ΔCRPS, rank-error, interval, and Top-5/10/25 Brier differences are all zero. C1 was not evaluated on the untouched test because it did not clear the pre-2022 selection rule.",
        "",
        "The model-report JSON retains full reliability bins, Brier scores, interval coverage and width, the development C1 location/scale optimizer diagnostics, and same-population metrics. The C1 development gain did not meet the pre-specified materiality threshold, so added complexity is not justified in the production specification.",
        "",
        "## Cold starts",
        "",
        "All FBS targets receive one H and one C PMF. C retains H's learned FCS-to-FBS transition PMF or broad generic FBS fallback for the six historical test cold starts; it does not manufacture coach context for them.",
        "",
        "## 2026 readiness",
        "",
        "H is reconstructable from completed 2025 ranks plus a pre-kickoff 2026 FBS universe. C is only partially reconstructable: dated coach tenures can be reconstructed where cached, but no production-safe snapshot exists for recruiting, talent, or returning production. No 2026 game outcome is read.",
        "",
        "## Annual refit",
        "",
        "For a new target season, preserve the specification version and record `trained_through_season`, `target_season`, and—only for C—an explicit `context_as_of` date. Refresh raw coaching tenures separately before fitting; the model build must run exclusively from that cached source.",
        "",
    ]
    (CONTEXT / "context_prior_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("loading historical inputs", flush=True)
    rows, cold_starts, _ = v1.load_rows()
    fbs = [row for row in rows if row.subdivision == "fbs"]
    raw_predictions = [
        row
        for row in read_prediction_rows()
        if row["model"] == H_MODEL_NAME
        and row["subdivision"] == "fbs"
        and int(row["season"]) in TEST_SEASONS
    ]
    if len(raw_predictions) != 534:
        raise ValueError("frozen H artifact is not the expected 534-test-team FBS PMFs")
    history_predictions = h.join_targets(
        h.read_predictions(H_MODEL_NAME), rows, cold_starts
    )
    if len(history_predictions) != len(raw_predictions):
        raise ValueError("H target join changed frozen production coverage")
    tenure_source = cached_tenures()
    contextual_rows, coverage_rows = with_coach_context(fbs, tenure_source)
    coverage_index = {
        tuple(row[name] for name in ("season", "subdivision", "team_id")): row
        for row in coverage_rows
    }
    train = [row for row in contextual_rows if row.season in DEV_TRAIN]
    validation = [row for row in contextual_rows if row.season in DEV_VALIDATION]
    test = [row for row in contextual_rows if row.season in TEST_SEASONS]
    complete_train = [row for row in train if has_complete_coach_context(row)]
    complete_validation = [row for row in validation if has_complete_coach_context(row)]
    complete_test = [row for row in test if has_complete_coach_context(row)]
    expected_coach_teams = {row.team_name for row in fbs}
    missing_tenure_teams = sorted(expected_coach_teams - set(tenure_source))
    coach_corpus_complete = not missing_tenure_teams
    if coach_corpus_complete:
        print("fitting development H", flush=True)
        h_development = h.fit(train, h.CANDIDATES[3])
        h_validation = prediction_with_model(h_development, complete_validation, "H")
        print("fitting development C1", flush=True)
        coach_development = DirectRankModel.fit(
            complete_train, [*H_FEATURES, *COACH_FEATURES], penalty=0.25
        )
        coach_validation = prediction_with_model(
            coach_development, complete_validation, "C1"
        )
        c1_development: dict[str, object] = evaluate_delta(
            h_validation, coach_validation
        )
        qualifies = (
            c1_development["delta_c_minus_h"]["nll"] <= -0.01
            and c1_development["paired_season_resampling"][
                "fraction_candidate_better_nll"
            ]
            >= 0.8
        )
    else:
        c1_development = {
            "status": "not fitted: incomplete dated coach-tenure raw corpus",
            "missing_source_teams": missing_tenure_teams,
        }
        qualifies = False
    selected = "C1_coach_tenure_location_and_scale" if qualifies else "C0_history_only"
    if qualifies:
        print("fitting final C1", flush=True)
        coach_final = DirectRankModel.fit(
            [row for row in contextual_rows if row.season < 2022 and has_complete_coach_context(row)],
            [*H_FEATURES, *COACH_FEATURES],
            penalty=0.25,
        )
        updated_test = prediction_with_model(coach_final, complete_test, "C1")
        final_fit_metadata: dict[str, object] = coach_final.metadata()
    else:
        updated_test = []
        final_fit_metadata = {"status": "C0 selected; no context coefficients admitted"}
    all_context = [
        {prediction.key: prediction for prediction in updated_test}.get(
            prediction.key, prediction
        )
        for prediction in history_predictions
    ]
    observed_keys = {
        (row.season, row.subdivision, row.team_id) for row in complete_test
    }
    observed_h = [prediction for prediction in history_predictions if prediction.key in observed_keys]
    observed_c = [prediction for prediction in all_context if prediction.key in observed_keys]
    history_identity = ModelIdentity("history_prior", "1.1", 2021, 2022)
    context_identity = ModelIdentity(
        "context_prior", "1.2", 2021, 2022, cutoff_for(2022)
    )
    old_report = json.loads((PRESEASON / "preseason_model_report.json").read_text())
    print("writing family artifacts", flush=True)
    h_spec = {
        **history_identity.metadata(),
        "fitted_instance": {
            "target_seasons": sorted(TEST_SEASONS),
            "trained_through_season": 2021,
            "prediction_source_model": H_MODEL_NAME,
        },
        "frozen_specification": old_report["final_test"]["selected_fit_metadata"],
        "cold_start_behavior": "V1 learned FCS-to-FBS transition, then generic FBS fallback",
    }
    c_spec = {
        **context_identity.metadata(),
        "fitted_instance": {"target_seasons": sorted(TEST_SEASONS)},
        "historical_foundation": H_FEATURES,
        "selected_candidate": selected,
        "approved_context_features": COACH_FEATURES if qualifies else [],
        "context_cutoff_semantics": "August 15 of each target season; target-specific date appears per PMF",
        "missing_context_behavior": "use the frozen H PMF exactly",
        "final_fit_metadata": final_fit_metadata,
    }
    coverage_counts = Counter(
        (row["season"], row["coach_context_available"]) for row in coverage_rows
    )
    report = {
        "history_model": h_spec,
        "context_model": c_spec,
        "provenance": FEATURE_PROVENANCE,
        "context_coverage": {
            str(season): {
                "available": coverage_counts[(season, True)],
                "missing_or_ambiguous": coverage_counts[(season, False)],
            }
            for season in sorted({row["season"] for row in coverage_rows})
        },
        "development": {
            "C0_history_only": {
                "same_population": v1.score_predictions(observed_h),
            },
            "C1_coach_tenure_location_and_scale": c1_development,
            "n_training_team_seasons_with_complete_coach_context": len(complete_train),
            "n_validation_team_seasons_with_complete_coach_context": len(complete_validation),
            "dated_coach_tenure_corpus_complete": coach_corpus_complete,
        },
        "selection": {
            "selected": selected,
            "rule": "C1 requires development ΔNLL <= -0.01 and >=0.80 favorable descriptive season resamples; otherwise C0",
            "development_train": [min(DEV_TRAIN), max(DEV_TRAIN)],
            "development_validation": [min(DEV_VALIDATION), max(DEV_VALIDATION)],
            "untouched_test": sorted(TEST_SEASONS),
        },
        "final_test": {
            "all_fbs": evaluate_delta(history_predictions, all_context),
            "coach_observed": evaluate_delta(observed_h, observed_c),
            "n_all_fbs": len(all_context),
            "n_coach_observed": len(observed_c),
            "cold_starts": "H fallback retained unchanged for all V1.1 cold-start PMFs",
        },
        "2026_readiness": {
            "history_prior": "fully reconstructable from completed 2025 outcomes and a frozen 2026 FBS universe",
            "context_prior": "partially reconstructable: dated coaching context only; other context families are not production-safe",
            "no_2026_game_outcomes_used": True,
        },
    }
    write_csv(HISTORY / "predictions.csv", history_prediction_rows(raw_predictions))
    write_json(HISTORY / "model_spec.json", h_spec)
    write_json(HISTORY / "model_report.json", old_report)
    write_json(HISTORY / "evaluation.json", old_report["final_test"])
    write_csv(
        CONTEXT / "predictions.csv",
        context_prediction_rows(history_predictions, updated_test, coverage_index),
    )
    write_json(CONTEXT / "model_spec.json", c_spec)
    write_json(CONTEXT / "feature_provenance.json", FEATURE_PROVENANCE)
    write_json(CONTEXT / "model_report.json", report)
    write_json(CONTEXT / "evaluation.json", report["final_test"])
    write_csv(CONTEXT / "context_coverage.csv", coverage_rows)
    render_report(report)
    print(
        json.dumps(
            {
                "selected": selected,
                "h_predictions": len(history_predictions),
                "c_predictions": len(all_context),
                "coach_observed_test": len(observed_c),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
