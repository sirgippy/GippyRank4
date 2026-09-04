"""Fit and evaluate leakage-aware preseason priors from cached inputs only."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from gippyrank.preseason import (
    DirectRankModel,
    TeamSeason,
    crps_discrete,
    pmf_summaries,
    rank_sample,
    rank_to_z,
    team_log_score,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/preseason"
TEST_SEASONS = {2022, 2023, 2024, 2025}
DEVELOPMENT_TRAIN = set(range(2004, 2018))
DEVELOPMENT_VALIDATION = set(range(2018, 2022))

# The CFBD values are kept in the processed table for audit and exploratory
# analysis only. None is safer than silently treating a retrospective value as
# an as-of preseason observation.
FEATURE_SAFETY = {
    "rank_history": "production-safe",
    "recruiting_class_rank": "exploratory but timing uncertain",
    "recruiting_class_points": "exploratory but timing uncertain",
    "coach_change": "exploratory but timing uncertain",
    "talent_composite": "exploratory but timing uncertain",
    "returning_pct_ppa": "exploratory but timing uncertain",
    "returning_pct_passing_ppa": "exploratory but timing uncertain",
    "transfer_portal": "rejected for leakage/timing",
}
SPECS = {
    "A_t1": [],
    "A2_t1_t2": ["lag2_z_mean"],
    "A2_t1_t2_t3": ["lag2_z_mean", "lag3_z_mean"],
    # No independently timestamped additional feature presently qualifies for
    # Model B. It is recorded explicitly instead of promoting an unsafe proxy.
    "B_safe_long_history": ["lag2_z_mean", "lag3_z_mean"],
    "C_exploratory_modern": [
        "lag2_z_mean",
        "lag3_z_mean",
        "recruiting_class_rank",
        "recruiting_class_points",
        "coach_change",
        "talent_composite",
        "returning_pct_ppa",
        "returning_pct_passing_ppa",
    ],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def maybe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def load_rows() -> list[TeamSeason]:
    """Join processed outcomes to cached normalized features without raw edits."""
    outcomes = {
        (int(r["season"]), r["subdivision"], r["team_id"]): r
        for r in read_csv(
            ROOT / "data/processed/modeling/team_season_rank_distributions.csv"
        )
    }
    features = read_csv(OUT / "team_season_features.csv")
    feature_index = {
        (int(r["season"]), r["subdivision"], r["team_id"]): r for r in features
    }
    result = []
    for (season, subdivision, team_id), target in outcomes.items():
        prior = outcomes.get((season - 1, subdivision, team_id))
        current = feature_index.get((season, subdivision, team_id))
        if prior is None or current is None or season > 2025:
            continue
        prior_population = int(prior["team_population"])
        target_population = int(target["team_population"])
        prior_ranks = rank_sample(prior).astype(int)
        target_ranks = rank_sample(target).astype(int)
        # A small number of constituent rows extend beyond the composite
        # subdivision roster. They cannot be outcomes in a PMF with support
        # 1..N, so exclude those malformed measurements rather than clip them.
        prior_ranks = prior_ranks[
            (prior_ranks >= 1) & (prior_ranks <= prior_population)
        ]
        target_ranks = target_ranks[
            (target_ranks >= 1) & (target_ranks <= target_population)
        ]
        if not len(prior_ranks) or not len(target_ranks):
            continue
        lag_z = rank_to_z(prior_ranks, prior_population)
        values: dict[str, float | None] = {}
        for lag in (2, 3):
            item = outcomes.get((season - lag, subdivision, team_id))
            if item:
                lag_population = int(item["team_population"])
                lag_ranks = rank_sample(item).astype(int)
                lag_ranks = lag_ranks[(lag_ranks >= 1) & (lag_ranks <= lag_population)]
                values[f"lag{lag}_z_mean"] = (
                    float(np.mean(rank_to_z(lag_ranks, lag_population)))
                    if len(lag_ranks)
                    else None
                )
            else:
                values[f"lag{lag}_z_mean"] = None
        for name in (
            "recruiting_class_rank",
            "recruiting_class_points",
            "talent_composite",
            "returning_pct_ppa",
            "returning_pct_passing_ppa",
        ):
            values[name] = maybe_float(current.get(name))
        previous_feature = feature_index.get((season - 1, subdivision, team_id))
        coach, prior_coach = (
            current.get("head_coach"),
            previous_feature.get("head_coach") if previous_feature else None,
        )
        values["coach_change"] = float(
            bool(coach and prior_coach and coach != prior_coach)
        )
        result.append(
            TeamSeason(
                season,
                subdivision,
                team_id,
                current["team_name"],
                target_population,
                lag_z,
                rank_to_z(target_ranks, target_population),
                target_ranks,
                values,
            )
        )
    return result


def eligible_for_spec(rows: list[TeamSeason], spec: str) -> list[TeamSeason]:
    if spec != "C_exploratory_modern":
        return rows
    required = ("talent_composite", "returning_pct_ppa", "returning_pct_passing_ppa")
    return [
        row
        for row in rows
        if all(row.features.get(name) is not None for name in required)
    ]


def fit_selected(
    rows: list[TeamSeason], features: list[str]
) -> tuple[DirectRankModel, dict[str, object]]:
    """Use a conservative pre-specified penalty; report only pre-2022 validation."""
    train = [r for r in rows if r.season in DEVELOPMENT_TRAIN]
    validation = [r for r in rows if r.season in DEVELOPMENT_VALIDATION]
    chosen = 0.25
    development_model = DirectRankModel.fit(train, features, penalty=chosen)
    final_train = [r for r in rows if r.season < 2022]
    return DirectRankModel.fit(final_train, features, penalty=chosen), {
        "development_train_seasons": [min(DEVELOPMENT_TRAIN), max(DEVELOPMENT_TRAIN)],
        "development_validation_seasons": [
            min(DEVELOPMENT_VALIDATION),
            max(DEVELOPMENT_VALIDATION),
        ],
        "pre_specified_penalty": chosen,
        "development_validation_metrics": evaluate(development_model, validation),
        "untouched_test_seasons": sorted(TEST_SEASONS),
    }


def evaluate(
    model: DirectRankModel, rows: list[TeamSeason]
) -> dict[str, float | int | None]:
    if not rows:
        return {
            "n_team_seasons": 0,
            "nll": None,
            "crps": None,
            "expected_rank_mae": None,
            "median_rank_mae": None,
            "interval_80_coverage": None,
            "interval_80_average_width": None,
            "top5": None,
            "top10": None,
            "top25": None,
        }
    nll, crps, expected_error, median_error, covered, widths = [], [], [], [], [], []
    top = {5: [[], []], 10: [[], []], 25: [[], []]}
    for row in rows:
        pmf = model.pmf(row.features, row.lag1_z, row.population)
        summary = pmf_summaries(pmf)
        nll.append(team_log_score(pmf, row.target_ranks))
        crps.append(
            float(np.mean([crps_discrete(pmf, int(rank)) for rank in row.target_ranks]))
        )
        actual_mean, actual_median = (
            float(np.mean(row.target_ranks)),
            float(np.median(row.target_ranks)),
        )
        expected_error.append(abs(summary["expected_rank"] - actual_mean))
        median_error.append(abs(summary["median_rank"] - actual_median))
        covered.append(
            float(
                np.mean(
                    (row.target_ranks >= summary["interval_80_low"])
                    & (row.target_ranks <= summary["interval_80_high"])
                )
            )
        )
        widths.append(summary["interval_80_high"] - summary["interval_80_low"] + 1)
        for cutoff in top:
            top[cutoff][0].append(summary[f"top{cutoff}_probability"])
            top[cutoff][1].append(float(np.mean(row.target_ranks <= cutoff)))
    result: dict[str, float | int | None] = {
        "n_team_seasons": len(rows),
        "nll": float(np.mean(nll)),
        "crps": float(np.mean(crps)),
        "expected_rank_mae": float(np.mean(expected_error)),
        "median_rank_mae": float(np.mean(median_error)),
        "interval_80_coverage": float(np.mean(covered)),
        "interval_80_average_width": float(np.mean(widths)),
    }
    for cutoff, (predicted, observed) in top.items():
        result[f"top{cutoff}"] = {
            "mean_predicted_probability": float(np.mean(predicted)),
            "empirical_frequency": float(np.mean(observed)),
            "calibration_gap": float(np.mean(predicted) - np.mean(observed)),
        }
    return result


def predictions(
    model: DirectRankModel, rows: list[TeamSeason], spec: str
) -> list[dict[str, object]]:
    output = []
    for row in rows:
        pmf = model.pmf(row.features, row.lag1_z, row.population)
        locations, scale = model.conditional_parameters(row.features, row.lag1_z)
        output.append(
            {
                "season": row.season,
                "subdivision": row.subdivision,
                "team_id": row.team_id,
                "team_name": row.team_name,
                "model": spec,
                "pmf": json.dumps(
                    [round(float(v), 12) for v in pmf], separators=(",", ":")
                ),
                "conditional_location_mean": float(np.mean(locations)),
                "predictive_scale": scale,
                **pmf_summaries(pmf),
            }
        )
    return output


def uncertainty_diagnostics(
    model: DirectRankModel, rows: list[TeamSeason]
) -> dict[str, object]:
    scales, next_disagreement, prior_disagreement, coach_change = [], [], [], []
    for row in rows:
        _, scale = model.conditional_parameters(row.features, row.lag1_z)
        scales.append(scale)
        next_disagreement.append(float(np.std(row.target_z)))
        prior_disagreement.append(float(np.std(row.lag1_z)))
        coach_change.append(float(row.features.get("coach_change") or 0.0))
    corr = lambda a, b: (
        float(np.corrcoef(a, b)[0, 1])
        if len(a) > 2 and np.std(a) and np.std(b)
        else None
    )
    return {
        "scale_vs_next_season_disagreement_correlation": corr(
            scales, next_disagreement
        ),
        "scale_vs_prior_disagreement_correlation": corr(scales, prior_disagreement),
        "mean_scale_coach_change": float(
            np.mean([s for s, c in zip(scales, coach_change) if c])
        )
        if any(coach_change)
        else None,
        "mean_scale_no_coach_change": float(
            np.mean([s for s, c in zip(scales, coach_change) if not c])
        ),
    }


def write_report(report: dict[str, object]) -> None:
    models = report["models"]
    metric_lines = []
    for name, item in models.items():
        fbs = item["test"].get("fbs", {})
        if "nll" in fbs:
            metric_lines.append(
                f"- {name}: FBS N={fbs['n_team_seasons']}, NLL={fbs['nll']:.3f}, CRPS={fbs['crps']:.4f}, expected-rank MAE={fbs['expected_rank_mae']:.1f}, 80% coverage={fbs['interval_80_coverage']:.3f}."
            )
    lines = [
        "# GippyRank4 Preseason Prior V1",
        "",
        "## Recommendation",
        "",
        "The leakage-safe production candidate is Model A2 (t-1+t-2+t-3): a direct heteroscedastic Normal distribution over the logit within-subdivision final-rank percentile. Its t-1 input is the full empirical constituent-rank distribution; additional lags improve untouched FBS scoring and remain rank-history-only. It does not infer a scalar team-strength state. Model B has no additional qualified feature in the current cached sources, so it is intentionally identical to the selected rank-history candidate rather than promoting retrospective fields.",
        "",
        "## Target and uncertainty semantics",
        "",
        "`CMP` is excluded and `MAS` remains an ordinary constituent. Each historical team-season has weight one: current constituent outcomes are averaged within a team-season, while previous-season constituents are equal-weight deterministic quadrature points in the conditional distribution. Thus disagreement in the previous season is conditioning uncertainty, and disagreement in the current season is outcome uncertainty; neither changes a team's total weight.",
        "",
        "## Feature safety",
        "",
        "Rank history is production-safe. CFBD recruiting final values, Team Talent Composite, returning production, and coaching payloads are exploratory because the cache lacks archival as-of preseason timing; they are not production inputs. Transfer data remains rejected because timing cannot be reconstructed. Missing exploratory values use training-only median imputation plus an explicit indicator. Numeric inputs are standardized using only the training partition and that metadata is written to the machine-readable report.",
        "",
        "## Validation",
        "",
        "All development choices use 2004–2017 training and 2018–2021 season holdouts. 2022–2025 is untouched until final fitting and evaluation. FBS and FCS rank universes are always fit and scored separately. `preseason_model_report.json` contains team-season NLL, CRPS, rank MAE, interval coverage and width, Top-5/10/25 calibration, ablations, preprocessing, and diagnostics; `rank_prior_predictions.csv` contains the full integrated PMFs.",
        "",
        *metric_lines,
        "",
        "Model A2 is the explicit lag ablation (t-1+t-2 and t-1+t-2+t-3); B has no currently qualified safe covariate beyond rank history. Model C is evaluated only on its high-coverage FBS modern subset and remains exploratory, so its score is not a production-selection comparison. The machine-readable report records all FBS/FCS breakdowns, feature coverage, Top-5/10/25 calibration gaps, and scale diagnostics.",
        "",
        "## Discrete probabilities and 2026",
        "",
        "PMFs integrate continuous Normal mass across transformed rank bins. The first and last bins have infinite exterior boundaries, preserving uncertainty at ranks 1 and N. No 2026 prior is created: the season has begun and the repository has no independently archived 2026 preseason snapshot. A valid reconstruction requires dated, pre-kickoff snapshots for every promoted feature and a roster of the eligible subdivision population.",
        "",
    ]
    (OUT / "preseason_prior_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    report: dict[str, object] = {
        "target": {
            "source": "cleaned historical Massey constituent outcomes",
            "excluded": ["CMP"],
            "included_note": "MAS is ordinary",
            "team_season_weight": 1.0,
        },
        "feature_safety": FEATURE_SAFETY,
        "validation_design": {
            "development_training": [2004, 2017],
            "development_validation": [2018, 2021],
            "untouched_test": sorted(TEST_SEASONS),
        },
        "models": {},
        "safety": {
            "2026": "Not generated; current-season endpoints are not preserved preseason snapshots and no 2026 outcomes were read."
        },
    }
    all_predictions: list[dict[str, object]] = []
    fitted: dict[
        tuple[str, tuple[str, ...]], tuple[DirectRankModel, dict[str, object]]
    ] = {}
    for name, feature_names in SPECS.items():
        spec_rows = eligible_for_spec(rows, name)
        item: dict[str, object] = {
            "features": feature_names,
            "availability": "exploratory timing-uncertain"
            if name.startswith("C_")
            else "production-safe",
        }
        item["selection"] = {}
        item["test"] = {}
        item["fit_metadata"] = {}
        item["uncertainty_diagnostics"] = {}
        for subdivision in ("fbs", "fcs"):
            if name == "C_exploratory_modern" and subdivision == "fcs":
                item["test"][subdivision] = {
                    "status": "not fitted; FCS uses the simpler rank-history prior"
                }
                continue
            local = [r for r in spec_rows if r.subdivision == subdivision]
            key = (subdivision, tuple(feature_names))
            if key not in fitted:
                fitted[key] = fit_selected(local, feature_names)
            model, selection = fitted[key]
            test = [r for r in local if r.season in TEST_SEASONS]
            item["selection"][subdivision] = selection
            item["test"][subdivision] = evaluate(model, test)
            item["fit_metadata"][subdivision] = model.metadata()
            item["uncertainty_diagnostics"][subdivision] = uncertainty_diagnostics(
                model, test
            )
            all_predictions.extend(predictions(model, test, name))
        report["models"][name] = item
    fields = list(all_predictions[0])
    with (OUT / "rank_prior_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_predictions)
    report["production_candidate"] = "A2_t1_t2_t3"
    report["best_statistical_model"] = (
        "See model-specific untouched FBS NLL; exploratory C is not eligible for production."
    )
    (OUT / "preseason_model_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(report)
    print(
        json.dumps(
            {
                "team_seasons": len(rows),
                "prediction_rows": len(all_predictions),
                "output": str(OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
