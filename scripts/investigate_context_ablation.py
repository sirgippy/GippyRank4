"""Observed-coverage, rolling-origin context feature ablation study.

This is deliberately separate from the frozen H 1.1 / C 1.2 builders.  Every
candidate is refit only for completed historical targets and writes solely to
``data/processed/context_ablation``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import build_preseason_context_prior_v1_2 as c12
import build_preseason_prior as v1
import build_preseason_prior_v1_1 as h11
import matplotlib.pyplot as plt
import numpy as np

from gippyrank.context_ablation import (
    aggregate_difference,
    assert_same_population,
    restrict_observed,
    standardized_interaction_rows,
    training_only_impute_rows,
    training_rows,
)
from gippyrank.preseason import DirectRankModel, TeamSeason, pmf_summaries

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/context_ablation"
PLOTS = OUT / "plots"
TARGET_MAX = 2025
MIN_TRAIN_SEASONS = 3
H = c12.H_FEATURES
COACH = c12.COACH_FEATURES
RECRUIT_CURRENT = ["recruiting_class_rank", "recruiting_class_points"]
RECRUIT_HISTORY = [
    "recruiting_points_2y_mean",
    "recruiting_points_3y_mean",
    "recruiting_points_4y_mean",
    "recruiting_points_trend",
]
RECRUIT = [*RECRUIT_CURRENT, *RECRUIT_HISTORY]
TALENT = c12.TALENT_FEATURES
RETURN_TOTAL = c12.RETURNING_TOTAL_FEATURES
RETURN_PASSING = c12.RETURNING_PASSING_FEATURES
RETURN_SKILL = c12.RETURNING_COMPONENT_FEATURES
RETURN_ALL = c12.RETURNING_FEATURES
RTP = [*RECRUIT, *TALENT, *RETURN_ALL]
ALL_CONTEXT = [*COACH, *RTP]


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(name: str, value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fit_h(rows: list[TeamSeason]) -> DirectRankModel:
    return DirectRankModel.fit(
        rows, H, penalty=0.25, optimizer_options={"maxiter": 600, "ftol": 1e-7}
    )


def fit_context(rows: list[TeamSeason], features: list[str]) -> DirectRankModel:
    """The C 1.2 equation layout with an explicit experimental feature subset."""
    return DirectRankModel.fit(
        rows,
        [*H, *features],
        penalty=0.25,
        location_feature_names=[*H, *features],
        scale_feature_names=H,
        optimizer_options={"maxiter": 900, "ftol": 1e-7, "gtol": 1e-5},
    )


def predictions(
    model: DirectRankModel, rows: list[TeamSeason], name: str
) -> list[v1.PriorPrediction]:
    return h11.make_predictions(model, rows, name)


def brier(predictions: list[v1.PriorPrediction], cutoff: int) -> float:
    return float(
        np.mean(
            [
                (
                    pmf_summaries(row.pmf)[f"top{cutoff}_probability"]
                    - float(np.mean(row.target_ranks <= cutoff))
                )
                ** 2
                for row in predictions
            ]
        )
    )


def metrics(predictions_: list[v1.PriorPrediction]) -> dict[str, float]:
    result = {
        key: float(value)
        for key, value in v1.score_predictions(predictions_).items()
        if not isinstance(value, dict) and value is not None
    }
    for cutoff in (5, 10, 25):
        result[f"top{cutoff}_brier"] = brier(predictions_, cutoff)
    return result


def coverage_rows(
    rows: list[TeamSeason],
) -> tuple[list[dict[str, object]], dict[str, list[TeamSeason]]]:
    """Describe raw-value populations before any preprocessor is fit."""
    definitions = {
        "R": RECRUIT,
        "T": TALENT,
        "P": RETURN_ALL,
        "RTP": RTP,
        "ALL_CONTEXT": ALL_CONTEXT,
        "COACH": COACH,
    }
    records: list[dict[str, object]] = []
    populations: dict[str, list[TeamSeason]] = {}
    for name, features in definitions.items():
        subset = restrict_observed(rows, features)
        populations[name] = subset
        by_season = defaultdict(list)
        for row in subset:
            by_season[row.season].append(row)
        records.append(
            {
                "population": name,
                "kind": "summary",
                "season": "all",
                "n_team_seasons": len(subset),
                "n_seasons": len(by_season),
                "teams_per_season": None,
                "features_required": ";".join(features),
                "coverage_definition": "all raw source values observed before imputation",
            }
        )
        for season, items in sorted(by_season.items()):
            records.append(
                {
                    "population": name,
                    "kind": "season",
                    "season": season,
                    "n_team_seasons": len(items),
                    "n_seasons": 1,
                    "teams_per_season": len(items),
                    "features_required": ";".join(features),
                    "coverage_definition": "all raw source values observed before imputation",
                }
            )
    for feature in ALL_CONTEXT:
        observed_count = sum(row.features.get(feature) is not None for row in rows)
        records.append(
            {
                "population": "raw_feature_missingness",
                "kind": "feature",
                "season": "all",
                "n_team_seasons": observed_count,
                "n_seasons": len({row.season for row in rows}),
                "teams_per_season": None,
                "features_required": feature,
                "coverage_definition": f"raw missingness={1 - observed_count / len(rows):.4f}",
            }
        )
    return records, populations


def feature_distribution(
    rows: list[TeamSeason], population: str
) -> list[dict[str, object]]:
    records = []
    for feature in ALL_CONTEXT:
        values = np.asarray(
            [r.features[feature] for r in rows if r.features.get(feature) is not None],
            dtype=float,
        )
        records.append(
            {
                "population": population,
                "feature": feature,
                "n": len(rows),
                "n_observed": len(values),
                "mean": float(values.mean()) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
                "std": float(values.std()) if len(values) else None,
                "p10": float(np.quantile(values, 0.1)) if len(values) else None,
                "p90": float(np.quantile(values, 0.9)) if len(values) else None,
            }
        )
    return records


def usable_target_years(rows: list[TeamSeason]) -> list[int]:
    years = []
    for target in sorted({row.season for row in rows if row.season <= TARGET_MAX}):
        prior = training_rows(rows, target)
        if (
            len({row.season for row in prior}) >= MIN_TRAIN_SEASONS
            and sum(row.season == target for row in rows) >= 20
        ):
            years.append(target)
    return years


def compare(
    name: str,
    population: str,
    source: list[TeamSeason],
    features: list[str],
    *,
    interaction: tuple[str, str] | None = None,
    impute_without_indicators: bool = False,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[int, dict[str, list[v1.PriorPrediction]]],
]:
    """Run a predeclared candidate against H on one fixed observed population."""
    annual, losses = [], []
    stored: dict[int, dict[str, list[v1.PriorPrediction]]] = {}
    for target in usable_target_years(source):
        target_rows = [row for row in source if row.season == target]
        train = training_rows(source, target)
        assert_same_population(target_rows, target_rows)
        candidate_features = list(features)
        if interaction is not None:
            train, target_rows, interaction_name = standardized_interaction_rows(
                train, target_rows, *interaction
            )
            candidate_features.append(interaction_name)
        if impute_without_indicators:
            train, target_rows = training_only_impute_rows(train, target_rows, features)
        h_pred = predictions(fit_h(train), target_rows, "H")
        c_pred = predictions(fit_context(train, candidate_features), target_rows, name)
        if {p.key for p in h_pred} != {p.key for p in c_pred}:
            raise ValueError("same_population_keys invariant failed")
        h_metrics, c_metrics = metrics(h_pred), metrics(c_pred)
        annual.append(
            {
                "population": population,
                "candidate": name,
                "target_season": target,
                "same_population_keys": True,
                "n_team_seasons": len(h_pred),
                "n_training_rows": len(train),
                "n_training_seasons": len({row.season for row in train}),
                **{f"h_{key}": value for key, value in h_metrics.items()},
                **{f"candidate_{key}": value for key, value in c_metrics.items()},
                **{
                    f"delta_{key}": c_metrics[key] - h_metrics[key] for key in h_metrics
                },
            }
        )
        h_loss, c_loss = h11.prediction_losses(h_pred), h11.prediction_losses(c_pred)
        for key in sorted(h_loss):
            losses.append(
                {
                    "population": population,
                    "candidate": name,
                    "season": key[0],
                    "subdivision": key[1],
                    "team_id": key[2],
                    "team_name": next(p.team_name for p in h_pred if p.key == key),
                    "h_nll": h_loss[key][0],
                    "candidate_nll": c_loss[key][0],
                    "candidate_minus_h_nll": c_loss[key][0] - h_loss[key][0],
                    "h_crps": h_loss[key][1],
                    "candidate_crps": c_loss[key][1],
                    "candidate_minus_h_crps": c_loss[key][1] - h_loss[key][1],
                }
            )
        stored[target] = {"H": h_pred, name: c_pred}
        print(f"completed {population}/{name}/{target}", flush=True)
    return annual, losses, stored


def summarize(
    annual: list[dict[str, object]], losses: list[dict[str, object]]
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in annual:
        grouped[str(row["population"]), str(row["candidate"])].append(row)
    result = []
    for (population, candidate), rows in sorted(grouped.items()):
        deltas = np.asarray([float(row["delta_nll"]) for row in rows])
        paired_losses = [
            r
            for r in losses
            if r["population"] == population and r["candidate"] == candidate
        ]
        result.append(
            {
                "population": population,
                "candidate": candidate,
                "same_population_keys": True,
                "target_seasons": ";".join(str(row["target_season"]) for row in rows),
                "n_target_seasons": len(rows),
                "n_team_seasons": len(paired_losses),
                "mean_delta_nll": aggregate_difference(deltas.tolist()),
                "median_delta_nll": float(np.median(deltas)),
                "wins": int(np.sum(deltas < -1e-12)),
                "losses": int(np.sum(deltas > 1e-12)),
                "ties": int(np.sum(np.isclose(deltas, 0))),
                "worst_season_regression": float(deltas.max()),
                "best_season_improvement": float(deltas.min()),
                "mean_delta_crps": float(
                    np.mean([float(row["delta_crps"]) for row in rows])
                ),
                "mean_delta_expected_rank_mae": float(
                    np.mean([float(row["delta_expected_rank_mae"]) for row in rows])
                ),
                "mean_delta_median_rank_mae": float(
                    np.mean([float(row["delta_median_rank_mae"]) for row in rows])
                ),
                "mean_delta_top5_brier": float(
                    np.mean([float(row["delta_top5_brier"]) for row in rows])
                ),
                "mean_delta_top10_brier": float(
                    np.mean([float(row["delta_top10_brier"]) for row in rows])
                ),
                "mean_delta_top25_brier": float(
                    np.mean([float(row["delta_top25_brier"]) for row in rows])
                ),
                "descriptive_bootstrap": h11.paired_bootstrap(
                    [
                        p
                        for year in sorted({int(x["season"]) for x in paired_losses})
                        for p in []
                    ],
                    [],
                )
                if False
                else "season-level bootstrap reported in summary.json",
            }
        )
    return result


def bootstrap_summary(
    stored_all: dict[tuple[str, str], dict[int, dict[str, list[v1.PriorPrediction]]]],
) -> dict[str, object]:
    result = {}
    for (population, candidate), yearly in stored_all.items():
        h_pred = [p for year in yearly.values() for p in year["H"]]
        c_pred = [p for year in yearly.values() for p in year[candidate]]
        result[f"{population}/{candidate}"] = h11.paired_bootstrap(h_pred, c_pred)
    return result


def prediction_index(
    predictions_: list[v1.PriorPrediction],
) -> dict[tuple[int, str, str], v1.PriorPrediction]:
    return {prediction.key: prediction for prediction in predictions_}


def adjustment_rows(
    yearly: dict[int, dict[str, list[v1.PriorPrediction]]],
    candidate: str,
    population: str,
) -> list[dict[str, object]]:
    records = []
    for year, values in yearly.items():
        h_idx, c_idx = (
            prediction_index(values["H"]),
            prediction_index(values[candidate]),
        )
        h_loss, c_loss = (
            h11.prediction_losses(values["H"]),
            h11.prediction_losses(values[candidate]),
        )
        for key, h_prediction in h_idx.items():
            c_prediction = c_idx[key]
            difference = abs(
                c_prediction.conditional_location_mean
                - h_prediction.conditional_location_mean
            )
            # Expected rank is used for the public bin; location is retained as a stable fallback.
            expected_difference = abs(
                pmf_summaries(c_prediction.pmf)["expected_rank"]
                - pmf_summaries(h_prediction.pmf)["expected_rank"]
            )
            label = (
                "0-5"
                if expected_difference < 5
                else "5-10"
                if expected_difference < 10
                else "10-20"
                if expected_difference < 20
                else "20+"
            )
            records.append(
                {
                    "population": population,
                    "candidate": candidate,
                    "season": year,
                    "team_id": key[2],
                    "team_name": h_prediction.team_name,
                    "adjustment_bin": label,
                    "expected_rank_adjustment": expected_difference,
                    "location_adjustment": difference,
                    "candidate_minus_h_nll": c_loss[key][0] - h_loss[key][0],
                }
            )
    return records


def tier_rows(
    yearly: dict[int, dict[str, list[v1.PriorPrediction]]], candidate: str
) -> list[dict[str, object]]:
    records = []
    for year, values in yearly.items():
        h_loss, c_loss = (
            h11.prediction_losses(values["H"]),
            h11.prediction_losses(values[candidate]),
        )
        for h_prediction in values["H"]:
            expected = pmf_summaries(h_prediction.pmf)["expected_rank"]
            tier = (
                "expected_top25"
                if expected <= 25
                else "middle_fbs"
                if expected <= 80
                else "lower_fbs"
            )
            records.append(
                {
                    "season": year,
                    "tier": tier,
                    "candidate_minus_h_nll": c_loss[h_prediction.key][0]
                    - h_loss[h_prediction.key][0],
                }
            )
    return records


def plot_bar(rows: list[dict[str, object]], title: str, filename: str) -> None:
    if not rows:
        return
    plt.figure(figsize=(max(8, len(rows) * 0.55), 4.5))
    labels = [str(row["candidate"]).replace("same_", "") for row in rows]
    values = [float(row["mean_delta_nll"]) for row in rows]
    plt.bar(labels, values, color=["#3b7" if value < 0 else "#b55" for value in values])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("mean ΔNLL (candidate − H)")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS / filename, dpi=160)
    plt.close()


def plots(
    coverage: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    annual: list[dict[str, object]],
    adjustment: list[dict[str, object]],
    decomposition: list[dict[str, object]],
) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    seasonal = [
        row
        for row in coverage
        if row["kind"] == "season"
        and row["population"] in {"R", "T", "P", "RTP", "ALL_CONTEXT"}
    ]
    plt.figure(figsize=(9, 4.5))
    for population in sorted({str(row["population"]) for row in seasonal}):
        part = [row for row in seasonal if row["population"] == population]
        plt.plot(
            [row["season"] for row in part],
            [row["n_team_seasons"] for row in part],
            marker="o",
            label=population,
        )
    plt.legend()
    plt.title("Raw observed feature coverage by season")
    plt.ylabel("team-seasons")
    plt.xlabel("season")
    plt.tight_layout()
    plt.savefig(PLOTS / "feature_coverage_by_season.png", dpi=160)
    plt.close()
    plot_bar(
        [r for r in summary_rows if str(r["population"]).startswith("single_")],
        "H vs single-feature-family candidates",
        "single_family_delta_nll.png",
    )
    plot_bar(
        [r for r in summary_rows if r["population"] == "same_all_context"],
        "Same-population cross-family ablations",
        "same_population_comparison.png",
    )
    plot_bar(
        [
            r
            for r in summary_rows
            if r["candidate"]
            in {"same_recruiting", "same_talent", "same_recruiting_talent"}
        ],
        "Recruiting and Talent incremental value",
        "recruiting_vs_talent.png",
    )
    plot_bar(
        [
            r
            for r in summary_rows
            if r["candidate"]
            in {
                "single_returning_total",
                "single_returning_passing",
                "single_returning_skill",
                "single_returning_all",
            }
        ],
        "Returning-production component ablation",
        "returning_components.png",
    )
    key = [
        row
        for row in annual
        if row["candidate"]
        in {"same_recruiting_talent_returning", "same_full_c", "single_returning_all"}
    ]
    plt.figure(figsize=(8, 4.5))
    for candidate in sorted({str(row["candidate"]) for row in key}):
        part = [row for row in key if row["candidate"] == candidate]
        plt.plot(
            [row["target_season"] for row in part],
            [row["delta_nll"] for row in part],
            marker="o",
            label=candidate,
        )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.legend()
    plt.title("Raw annual ΔNLL for key candidates")
    plt.ylabel("ΔNLL")
    plt.tight_layout()
    plt.savefig(PLOTS / "annual_delta_nll.png", dpi=160)
    plt.close()
    plt.figure(figsize=(8, 4.5))
    for candidate in ("single_recruiting_all", "single_talent", "single_returning_all"):
        part = [row for row in annual if row["candidate"] == candidate]
        if part:
            plt.plot(
                [row["target_season"] for row in part],
                [row["delta_nll"] for row in part],
                marker="o",
                label=candidate,
            )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.legend()
    plt.title("Context value within its coverage era")
    plt.ylabel("ΔNLL")
    plt.tight_layout()
    plt.savefig(PLOTS / "coverage_era_stability.png", dpi=160)
    plt.close()
    bins = (
        sorted(
            {str(row["adjustment_bin"]) for row in adjustment},
            key=lambda x: ["0-5", "5-10", "10-20", "20+"].index(x),
        )
        if adjustment
        else []
    )
    plt.figure(figsize=(7, 4.5))
    plt.bar(
        bins,
        [
            np.mean(
                [
                    float(r["candidate_minus_h_nll"])
                    for r in adjustment
                    if r["adjustment_bin"] == b
                ]
            )
            for b in bins
        ],
    )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("C/H adjustment magnitude versus benefit")
    plt.ylabel("mean ΔNLL")
    plt.tight_layout()
    plt.savefig(PLOTS / "adjustment_magnitude.png", dpi=160)
    plt.close()
    selected = sorted(
        decomposition, key=lambda row: float(row["candidate_minus_h_nll"])
    )
    plt.figure(figsize=(9, 4.5))
    plt.bar(range(len(selected)), [row["candidate_minus_h_nll"] for row in selected])
    plt.title("2025 feature-family decomposition (candidate − H)")
    plt.ylabel("team ΔNLL")
    plt.tight_layout()
    plt.savefig(PLOTS / "decomposition_2025.png", dpi=160)
    plt.close()
    # The final two required views reuse informative same-population summaries.
    plot_bar(
        [r for r in summary_rows if "interaction" in str(r["candidate"])],
        "Interaction candidates",
        "interactions.png",
    )
    plot_bar(
        [r for r in summary_rows if r["population"] == "same_all_context"],
        "Selected team-example model set",
        "team_examples_waterfall.png",
    )


def render_report(summary: dict[str, object]) -> None:
    findings = summary["headline_results"]
    lines = [
        "# Coverage-restricted context-feature ablation",
        "",
        "## Design",
        "",
        "All comparisons use raw observed coverage chosen before any imputation. For each rolling target, both H and its candidate are trained on eligible rows strictly before the target and predicted on identical team-season keys. Location-only context is primary. H 1.1 and C 1.2 remain immutable references; 2026 is absent from this research build.",
        "",
        "## Headline results",
        "",
    ]
    lines.extend(f"- {item}" for item in findings)
    lines += [
        "",
        "## Interpretation limits",
        "",
        "The season-level bootstrap ranges in `summary.json` are descriptive because the common coverage era has few seasons. Raw annual ΔNLL points in `annual_ablation_metrics.csv` are the primary robustness evidence. No candidate is proposed as C 1.3 here.",
        "",
        "## Artifacts",
        "",
        "- `coverage_populations.csv` / `.json` — raw, pre-imputation coverage definitions and annual counts.",
        "- `annual_ablation_metrics.csv`, `candidate_summary.csv`, and `same_population_comparisons.csv` — paired held-out scores with the `same_population_keys` invariant.",
        "- `per_team_losses.csv`, `interaction_results.csv`, `missingness_results.csv`, `adjustment_magnitude.csv`, `decomposition_2025.csv`, and `team_examples_2025.csv` — auditable diagnostics.",
        "- `plots/` — raw coverage, ablations, annual effects, interactions, adjustment-risk, and 2025 diagnostics.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows, _cold, _coverage = v1.load_rows(max_season=TARGET_MAX)
    fbs = [row for row in rows if row.subdivision == "fbs"]
    contextual, _ = c12.attach_context(fbs, c12.feature_index(), c12.cached_tenures())
    coverage, populations = coverage_rows(contextual)
    write_csv("coverage_populations.csv", coverage)
    write_json(
        "coverage_populations.json",
        {
            "definitions": {
                "R": RECRUIT,
                "T": TALENT,
                "P": RETURN_ALL,
                "RTP": RTP,
                "ALL_CONTEXT": ALL_CONTEXT,
            },
            "raw_before_imputation": True,
        },
    )
    write_csv(
        "feature_distributions.csv",
        [
            *feature_distribution(populations["RTP"], "RTP"),
            *feature_distribution(populations["ALL_CONTEXT"], "ALL_CONTEXT"),
        ],
    )

    definitions = [
        ("single_coach", "single_coach", COACH, COACH),
        (
            "single_recruiting_current",
            "single_recruiting_current",
            RECRUIT_CURRENT,
            RECRUIT_CURRENT,
        ),
        (
            "single_recruiting_history",
            "single_recruiting_history",
            RECRUIT_HISTORY,
            RECRUIT_HISTORY,
        ),
        ("single_recruiting_all", "single_recruiting_all", RECRUIT, RECRUIT),
        ("single_talent", "single_talent", TALENT, TALENT),
        (
            "single_returning_total",
            "single_returning_total",
            RETURN_TOTAL,
            RETURN_TOTAL,
        ),
        (
            "single_returning_passing",
            "single_returning_passing",
            RETURN_PASSING,
            RETURN_PASSING,
        ),
        (
            "single_returning_skill",
            "single_returning_skill",
            RETURN_SKILL,
            RETURN_SKILL,
        ),
        ("single_returning_all", "single_returning_all", RETURN_ALL, RETURN_ALL),
    ]
    same_definitions = [
        ("same_h", []),
        ("same_coach", COACH),
        ("same_recruiting", RECRUIT),
        ("same_talent", TALENT),
        ("same_returning", RETURN_ALL),
        ("same_recruiting_talent", [*RECRUIT, *TALENT]),
        ("same_recruiting_returning", [*RECRUIT, *RETURN_ALL]),
        ("same_talent_returning", [*TALENT, *RETURN_ALL]),
        ("same_coach_returning", [*COACH, *RETURN_ALL]),
        ("same_recruiting_talent_returning", RTP),
        ("same_full_c", ALL_CONTEXT),
    ]
    annual: list[dict[str, object]] = []
    losses: list[dict[str, object]] = []
    stored_all: dict[
        tuple[str, str], dict[int, dict[str, list[v1.PriorPrediction]]]
    ] = {}
    for population, name, required, features in definitions:
        source = restrict_observed(contextual, required)
        result = compare(name, population, source, features)
        annual.extend(result[0])
        losses.extend(result[1])
        stored_all[(population, name)] = result[2]
    for name, features in same_definitions:
        # All cross-family candidates use ALL_CONTEXT: one truly identical common population.
        result = compare(name, "same_all_context", populations["ALL_CONTEXT"], features)
        annual.extend(result[0])
        losses.extend(result[1])
        stored_all[("same_all_context", name)] = result[2]
    # A larger coach-free roster common population makes the roster decomposition less sparse.
    for name, features in [
        ("rtp_h", []),
        ("rtp_recruiting", RECRUIT),
        ("rtp_talent", TALENT),
        ("rtp_returning", RETURN_ALL),
        ("rtp_recruiting_talent", [*RECRUIT, *TALENT]),
        ("rtp_recruiting_talent_returning", RTP),
    ]:
        result = compare(name, "same_rtp", populations["RTP"], features)
        annual.extend(result[0])
        losses.extend(result[1])
        stored_all[("same_rtp", name)] = result[2]

    interactions = [
        (
            "interaction_talent_total",
            TALENT + RETURN_TOTAL,
            ("talent_composite", "returning_pct_ppa"),
        ),
        (
            "interaction_talent_passing",
            TALENT + RETURN_PASSING,
            ("talent_composite", "returning_pct_passing_ppa"),
        ),
        (
            "interaction_coach_total",
            [*COACH, *RETURN_TOTAL],
            ("coach_tenure_seasons", "returning_pct_ppa"),
        ),
    ]
    for name, features, interaction in interactions:
        required = [*features]
        result = compare(
            name,
            f"interaction_{name}",
            restrict_observed(contextual, required),
            features,
            interaction=interaction,
        )
        annual.extend(result[0])
        losses.extend(result[1])
        stored_all[(f"interaction_{name}", name)] = result[2]

    # Missingness is a diagnostic on the unrestricted historical universe, not evidence that a feature helps.
    missing_rows: list[dict[str, object]] = []
    for name, features in [
        ("recruiting", RECRUIT),
        ("talent", TALENT),
        ("returning", RETURN_ALL),
    ]:
        raw = compare(
            f"missingness_{name}_with_indicators",
            "full_fbs_imputed",
            contextual,
            features,
        )
        filled = compare(
            f"missingness_{name}_without_indicators",
            "full_fbs_imputed",
            contextual,
            features,
            impute_without_indicators=True,
        )
        for kind, result in (("with_indicators", raw), ("without_indicators", filled)):
            annual.extend(result[0])
            losses.extend(result[1])
            stored_all[("full_fbs_imputed", result[0][0]["candidate"])] = result[2]
            for row in result[0]:
                missing_rows.append(
                    {
                        "family": name,
                        "formulation": kind,
                        "target_season": row["target_season"],
                        "delta_nll": row["delta_nll"],
                        "raw_missingness_present": kind == "with_indicators",
                    }
                )

    summary_rows = summarize(annual, losses)
    write_csv("annual_ablation_metrics.csv", annual)
    write_csv("per_team_losses.csv", losses)
    write_csv("candidate_summary.csv", summary_rows)
    write_csv(
        "same_population_comparisons.csv",
        [row for row in summary_rows if str(row["population"]).startswith("same_")],
    )
    write_csv(
        "interaction_results.csv",
        [row for row in summary_rows if "interaction" in str(row["candidate"])],
    )
    write_csv("missingness_results.csv", missing_rows)

    full = stored_all[("same_all_context", "same_full_c")]
    adjustment = adjustment_rows(full, "same_full_c", "same_all_context")
    write_csv("adjustment_magnitude.csv", adjustment)
    full_losses = [
        row
        for row in losses
        if row["population"] == "same_all_context"
        and row["candidate"] == "same_full_c"
        and row["season"] == 2025
    ]
    write_csv("decomposition_2025.csv", full_losses)
    tiers = tier_rows(
        stored_all[("single_returning_all", "single_returning_all")],
        "single_returning_all",
    )
    write_csv("returning_tier_results.csv", tiers)

    # Team examples include fixed prior losses and several material C gains, avoiding failure-only selection.
    c2025 = full.get(2025, {})
    if c2025:
        h_idx, c_idx = (
            prediction_index(c2025["H"]),
            prediction_index(c2025["same_full_c"]),
        )
        c_losses = h11.prediction_losses(c2025["same_full_c"])
        h_losses = h11.prediction_losses(c2025["H"])
        requested = {"New Mexico", "Utah", "James Madison", "North Texas"}
        improved = sorted(c_idx, key=lambda key: c_losses[key][0] - h_losses[key][0])[
            :4
        ]
        selected = [
            key for key, row in h_idx.items() if row.team_name in requested
        ] + improved
        examples = []
        for key in dict.fromkeys(selected):
            hp, cp = h_idx[key], c_idx[key]
            examples.append(
                {
                    "season": 2025,
                    "team_id": key[2],
                    "team_name": hp.team_name,
                    "h_expected_rank": pmf_summaries(hp.pmf)["expected_rank"],
                    "c_expected_rank": pmf_summaries(cp.pmf)["expected_rank"],
                    "c_minus_h_expected_rank": pmf_summaries(cp.pmf)["expected_rank"]
                    - pmf_summaries(hp.pmf)["expected_rank"],
                    "realized_rank_mean": float(np.mean(hp.target_ranks)),
                    "realized_rank_min": int(np.min(hp.target_ranks)),
                    "realized_rank_max": int(np.max(hp.target_ranks)),
                    "c_minus_h_nll": c_losses[key][0] - h_losses[key][0],
                }
            )
        write_csv("team_examples_2025.csv", examples)
    else:
        examples = []

    bootstrap = bootstrap_summary(stored_all)
    by_name = {(str(r["population"]), str(r["candidate"])): r for r in summary_rows}

    def result(population: str, candidate: str) -> str:
        row = by_name.get((population, candidate))
        return (
            "not evaluable"
            if row is None
            else f"mean ΔNLL {float(row['mean_delta_nll']):+.4f}; {row['wins']} wins/{row['losses']} losses"
        )

    headline = [
        f"Coach tenure on its natural raw-coverage population: {result('single_coach', 'single_coach')}.",
        f"Current recruiting: {result('single_recruiting_current', 'single_recruiting_current')}; recruiting history: {result('single_recruiting_history', 'single_recruiting_history')}; all recruiting: {result('single_recruiting_all', 'single_recruiting_all')}.",
        f"Talent: {result('single_talent', 'single_talent')}; total/passing/skill returning production: {result('single_returning_total', 'single_returning_total')}, {result('single_returning_passing', 'single_returning_passing')}, {result('single_returning_skill', 'single_returning_skill')}.",
        f"On one exact all-context common population, recruiting+Talent+returning: {result('same_all_context', 'same_recruiting_talent_returning')}; full frozen-spec C-equivalent: {result('same_all_context', 'same_full_c')}.",
        f"Talent×total-returning interaction: {result('interaction_interaction_talent_total', 'interaction_talent_total')}; Talent×passing-returning: {result('interaction_interaction_talent_passing', 'interaction_talent_passing')}.",
        "Interpret annual wins/losses and descriptive season-bootstrap ranges conservatively; the coverage-era target count is intentionally limited.",
        "No production H/C specification, frozen 2026 PMF, or 2026 outcome was modified or accessed.",
    ]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            ROOT / "data/processed/preseason/history/annual/2026/predictions.csv",
            ROOT / "data/processed/preseason/context/annual/2026/predictions.csv",
        ]
    }
    summary = {
        "study": "coverage-restricted context feature ablation",
        "target_max": TARGET_MAX,
        "min_train_seasons": MIN_TRAIN_SEASONS,
        "raw_coverage_before_imputation": True,
        "same_population_keys_required": True,
        "no_2026_outcomes_accessed": True,
        "production_models_modified": False,
        "frozen_2026_pmf_hashes": hashes,
        "descriptive_season_bootstrap": bootstrap,
        "headline_results": headline,
        "team_examples_2025_count": len(examples),
    }
    write_json("summary.json", summary)
    render_report(summary)
    plots(coverage, summary_rows, annual, adjustment, full_losses)


if __name__ == "__main__":
    main()
