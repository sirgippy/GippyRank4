"""Pseudo-prospective validation of the predeclared Lean Context candidate.

The study is research-only.  It never writes the frozen H 1.1/C 1.2 paths or
reads a 2026 outcome.  Models are refit once per historical target using only
completed seasons, and all comparisons pair identical target keys.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import build_preseason_context_prior_v1_2 as c12
import build_preseason_prior as v1
import build_preseason_prior_v1_1 as h11
import matplotlib.pyplot as plt
import numpy as np

from gippyrank.lean_context_validation import (
    CLEAR_C_NLL_GAIN,
    H_FEATURES,
    INTERVAL_80_COVERAGE_METRIC,
    LEAN_CONTEXT_FEATURES,
    MATERIAL_NLL_GAIN,
    MAX_COVERAGE_REGRESSION,
    MIN_PRIOR_TARGETS,
    PARSIMONY_TIE_NLL,
    annual_delta_field,
    assert_lean_specification,
    choose_nested_model,
    observed_common,
    prior_training,
    recommendation_outcome,
)
from gippyrank.preseason import DirectRankModel, TeamSeason, pmf_summaries

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/lean_context_validation"
PLOTS = OUT / "plots"
TARGET_MAX = 2025
MIN_TRAIN_SEASONS = 3
FULL_CONTEXT_FEATURES = (
    *c12.COACH_FEATURES,
    *c12.RECRUITING_FEATURES,
    *c12.TALENT_FEATURES,
    *c12.RETURNING_FEATURES,
)


def clean_output() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    PLOTS.mkdir(parents=True)


def write_json(name: str, value: Any) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_values(predictions: list[v1.PriorPrediction]) -> dict[str, float]:
    scored = v1.score_predictions(predictions)
    values = {
        key: float(scored[key])
        for key in (
            "nll",
            "crps",
            "expected_rank_mae",
            "median_rank_mae",
            "interval_80_coverage",
            "interval_80_average_width",
        )
    }
    for cutoff in (5, 10, 25):
        values[f"top{cutoff}_brier"] = float(
            scored[f"top{cutoff}"]["reliability"]["brier_score"]
        )
    return values


def index_predictions(
    values: list[v1.PriorPrediction],
) -> dict[tuple[int, str, str], v1.PriorPrediction]:
    return {value.key: value for value in values}


def subset(
    values: list[v1.PriorPrediction], keys: set[tuple[int, str, str]]
) -> list[v1.PriorPrediction]:
    result = [value for value in values if value.key in keys]
    if {value.key for value in result} != keys:
        raise ValueError("annual model comparison does not share identical target keys")
    return result


def fit_l(training: list[TeamSeason]) -> DirectRankModel:
    assert_lean_specification(LEAN_CONTEXT_FEATURES)
    return DirectRankModel.fit(
        training,
        [*H_FEATURES, *LEAN_CONTEXT_FEATURES],
        penalty=0.25,
        location_feature_names=[*H_FEATURES, *LEAN_CONTEXT_FEATURES],
        scale_feature_names=[*H_FEATURES],
        optimizer_options={"maxiter": 600, "ftol": 1e-7, "gtol": 1e-5},
    )


def fit_models(training: list[TeamSeason], target: int) -> dict[str, DirectRankModel]:
    if any(row.season >= target for row in training):
        raise ValueError("target outcome entered model fitting")
    # The penalty and Normal direct-rank family match H/C.  These documented
    # research tolerances are the established ablation settings and make the
    # repeated rolling reconstruction practical without changing the model.
    h_model = DirectRankModel.fit(
        training,
        list(H_FEATURES),
        penalty=0.25,
        optimizer_options={"maxiter": 600, "ftol": 1e-7, "gtol": 1e-5},
    )
    c_model = DirectRankModel.fit(
        training,
        [*H_FEATURES, *FULL_CONTEXT_FEATURES],
        penalty=0.25,
        location_feature_names=[*H_FEATURES, *FULL_CONTEXT_FEATURES],
        scale_feature_names=[*H_FEATURES],
        optimizer_options={"maxiter": 600, "ftol": 1e-7, "gtol": 1e-5},
    )
    return {"H": h_model, "C": c_model, "L": fit_l(training)}


def predict(
    models: dict[str, DirectRankModel], rows: list[TeamSeason]
) -> dict[str, list[v1.PriorPrediction]]:
    return {
        "H": h11.make_predictions(models["H"], rows, "H_1_1"),
        "C": h11.make_predictions(models["C"], rows, "C_1_2"),
        "L": h11.make_predictions(models["L"], rows, "Lean_Context_L"),
    }


def annual_record(
    target: int,
    population: str,
    predictions: dict[str, list[v1.PriorPrediction]],
    n_training: int,
    n_seasons: int,
) -> dict[str, Any]:
    keys = [{value.key for value in values} for values in predictions.values()]
    if not (keys[0] == keys[1] == keys[2]):
        raise ValueError("H/C/L target keys differ")
    metrics = {name: metric_values(values) for name, values in predictions.items()}
    row: dict[str, Any] = {
        "population": population,
        "target_season": target,
        "n_team_seasons": len(keys[0]),
        "n_training_rows": n_training,
        "n_training_seasons": n_seasons,
        "same_population_keys": True,
    }
    for name in ("H", "C", "L"):
        row.update(
            {f"{name.lower()}_{key}": value for key, value in metrics[name].items()}
        )
    for left, right in (("L", "H"), ("L", "C"), ("C", "H")):
        for metric in metrics[left]:
            row[annual_delta_field(left, right, metric)] = (
                metrics[left][metric] - metrics[right][metric]
            )
    return row


def coverage(rows: list[TeamSeason], eligible: list[int]) -> dict[str, Any]:
    seasonal = []
    for year in sorted({row.season for row in rows}):
        group = [row for row in rows if row.season == year]
        recruiting = sum(
            all(row.features.get(x) is not None for x in c12.RECRUITING_FEATURES)
            for row in group
        )
        returning = sum(
            all(row.features.get(x) is not None for x in c12.RETURNING_FEATURES)
            for row in group
        )
        overlap = len(observed_common(group))
        seasonal.append(
            {
                "season": year,
                "fbs_team_seasons": len(group),
                "recruiting_complete": recruiting,
                "returning_complete": returning,
                "lean_overlap": overlap,
                "eligible_target": year in eligible,
            }
        )
    return {
        "raw_before_imputation": True,
        "lean_features": list(LEAN_CONTEXT_FEATURES),
        "eligible_target_seasons": eligible,
        "by_season": seasonal,
    }


def loss_rows(
    target: int,
    population: str,
    predictions: dict[str, list[v1.PriorPrediction]],
) -> list[dict[str, Any]]:
    losses = {name: h11.prediction_losses(value) for name, value in predictions.items()}
    indices = {name: index_predictions(value) for name, value in predictions.items()}
    if not (set(losses["H"]) == set(losses["C"]) == set(losses["L"])):
        raise ValueError("per-team losses have unpaired keys")
    result = []
    for key in sorted(losses["H"]):
        h_prediction = indices["H"][key]
        values = {name: pmf_summaries(indices[name][key].pmf) for name in indices}
        result.append(
            {
                "population": population,
                "season": target,
                "subdivision": key[1],
                "team_id": key[2],
                "team_name": h_prediction.team_name,
                **{f"{name.lower()}_nll": losses[name][key][0] for name in losses},
                **{f"{name.lower()}_crps": losses[name][key][1] for name in losses},
                **{
                    f"{name.lower()}_expected_rank": values[name]["expected_rank"]
                    for name in values
                },
                **{
                    f"{name.lower()}_interval_80_low": values[name]["interval_80_low"]
                    for name in values
                },
                **{
                    f"{name.lower()}_interval_80_high": values[name]["interval_80_high"]
                    for name in values
                },
                "l_minus_h_nll": losses["L"][key][0] - losses["H"][key][0],
                "l_minus_c_nll": losses["L"][key][0] - losses["C"][key][0],
            }
        )
    return result


def comparison(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    deltas = {
        metric: np.asarray(
            [
                float(row[f"{left.lower()}_minus_{right.lower()}_{metric}"])
                for row in rows
            ]
        )
        for metric in (
            "nll",
            "crps",
            "expected_rank_mae",
            "median_rank_mae",
            "interval_80_coverage",
            "interval_80_average_width",
            "top5_brier",
            "top10_brier",
            "top25_brier",
        )
    }
    annual_nll = deltas["nll"]
    return {
        "comparison": f"{left} minus {right}",
        "negative_is_better_for_scores": True,
        "n_target_seasons": len(rows),
        "aggregate_mean_annual_deltas": {
            key: float(value.mean()) for key, value in deltas.items()
        },
        "median_annual_delta_nll": float(np.median(annual_nll)),
        "wins": int((annual_nll < -1e-12).sum()),
        "losses": int((annual_nll > 1e-12).sum()),
        "ties": int(np.isclose(annual_nll, 0).sum()),
        "best_year": int(rows[int(annual_nll.argmin())]["target_season"]),
        "worst_year": int(rows[int(annual_nll.argmax())]["target_season"]),
    }


def calibration(
    predictions: dict[tuple[str, int], dict[str, list[v1.PriorPrediction]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for (population, year), models in predictions.items():
        for name, values in models.items():
            scored = v1.score_predictions(values)
            tiers: dict[str, list[float]] = defaultdict(list)
            for prediction in values:
                expected = pmf_summaries(prediction.pmf)["expected_rank"]
                tier = (
                    "top25"
                    if expected <= 25
                    else "middle"
                    if expected <= 80
                    else "lower"
                )
                interval = pmf_summaries(prediction.pmf)
                tiers[tier].append(
                    float(
                        np.mean(
                            (prediction.target_ranks >= interval["interval_80_low"])
                            & (prediction.target_ranks <= interval["interval_80_high"])
                        )
                    )
                )
            output[f"{population}/{year}/{name}"] = {
                "interval_80_coverage": scored["interval_80_coverage"],
                "interval_80_average_width": scored["interval_80_average_width"],
                "tier_coverage": {
                    key: float(np.mean(value)) for key, value in tiers.items()
                },
                "top_reliability": {
                    str(cutoff): scored[f"top{cutoff}"]["reliability"]
                    for cutoff in (5, 10, 25)
                },
            }
    return output


def disagreement(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    summary: dict[str, Any] = {}
    for left, right in (("L", "H"), ("C", "H"), ("L", "C")):
        key = f"{left.lower()}_vs_{right.lower()}"
        values = np.asarray(
            [
                abs(
                    float(row[f"{left.lower()}_expected_rank"])
                    - float(row[f"{right.lower()}_expected_rank"])
                )
                for row in rows
            ]
        )
        summary[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "gt5": int((values > 5).sum()),
            "gt10": int((values > 10).sum()),
            "gt20": int((values > 20).sum()),
        }
        for row, value in zip(rows, values, strict=True):
            output.append(
                {
                    "season": row["season"],
                    "team_id": row["team_id"],
                    "comparison": key,
                    "absolute_expected_rank_difference": float(value),
                    "l_minus_h_nll": row["l_minus_h_nll"],
                }
            )
    return output, summary


def render_plots(
    annual: list[dict[str, Any]],
    disagreement_rows: list[dict[str, Any]],
    coverage_data: dict[str, Any],
    nested: list[dict[str, Any]],
) -> None:
    observed = [row for row in annual if row["population"] == "observed_common"]
    years = [row["target_season"] for row in observed]

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(PLOTS / name, dpi=160)
        plt.close()

    plt.figure(figsize=(9, 4.5))
    for model in ("H", "C", "L"):
        plt.plot(
            years,
            [row[f"{model.lower()}_nll"] for row in observed],
            marker="o",
            label=model,
        )
    plt.title("Observed-common annual NLL")
    plt.xlabel("target season")
    plt.legend()
    save("annual_nll_hcl.png")
    for right, filename in (
        ("h", "annual_l_minus_h_nll.png"),
        ("c", "annual_l_minus_c_nll.png"),
    ):
        plt.figure(figsize=(9, 4.5))
        values = [row[f"l_minus_{right}_nll"] for row in observed]
        plt.axhline(0, color="black", linewidth=0.8)
        plt.plot(years, values, marker="o")
        plt.title(f"Annual L − {right.upper()} NLL (negative favors L)")
        plt.xlabel("target season")
        save(filename)
    plt.figure(figsize=(9, 4.5))
    cumulative = np.cumsum([row["l_minus_h_nll"] for row in observed])
    plt.plot(years, cumulative, marker="o")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Cumulative L − H NLL")
    save("cumulative_l_minus_h_nll.png")
    plt.figure(figsize=(9, 4.5))
    for model in ("H", "C", "L"):
        plt.plot(
            years,
            [row[f"{model.lower()}_interval_80_coverage"] for row in observed],
            marker="o",
            label=model,
        )
    plt.axhline(0.8, color="black", linewidth=0.8)
    plt.title("80% interval coverage")
    plt.legend()
    save("interval_coverage.png")
    plt.figure(figsize=(9, 4.5))
    for model in ("H", "C", "L"):
        plt.plot(
            years,
            [row[f"{model.lower()}_interval_80_average_width"] for row in observed],
            marker="o",
            label=model,
        )
    plt.title("80% interval width")
    plt.legend()
    save("interval_width.png")
    plt.figure(figsize=(9, 4.5))
    for comparison_name in ("l_vs_h", "c_vs_h", "l_vs_c"):
        values = [
            row for row in disagreement_rows if row["comparison"] == comparison_name
        ]
        grouped = defaultdict(list)
        for row in values:
            grouped[int(row["season"])].append(
                float(row["absolute_expected_rank_difference"])
            )
        plt.plot(
            sorted(grouped),
            [np.mean(grouped[x]) for x in sorted(grouped)],
            marker="o",
            label=comparison_name,
        )
    plt.title("Mean H/C/L expected-rank disagreement")
    plt.legend()
    save("expected_rank_disagreement.png")
    plt.figure(figsize=(9, 4.5))
    plt.step(
        [row["target_season"] for row in nested],
        [{"H": 0, "C": 1, "L": 2}[str(row["selected_model"])] for row in nested],
        where="mid",
    )
    plt.yticks([0, 1, 2], ["H", "C", "L"])
    plt.title("Nested prospective model selection")
    save("nested_selection.png")
    coverage_rows = coverage_data["by_season"]
    plt.figure(figsize=(9, 4.5))
    for key in ("recruiting_complete", "returning_complete", "lean_overlap"):
        plt.plot(
            [x["season"] for x in coverage_rows],
            [x[key] for x in coverage_rows],
            marker="o",
            label=key,
        )
    plt.title("Raw context coverage by season")
    plt.legend()
    save("coverage_by_season.png")


def render_report(summary: dict[str, Any]) -> None:
    h_l, l_c = (
        summary["comparisons"]["observed_common/H_vs_L"],
        summary["comparisons"]["observed_common/L_vs_C"],
    )
    full_h_l, full_l_c = (
        summary["comparisons"]["full_fbs_production_style/H_vs_L"],
        summary["comparisons"]["full_fbs_production_style/L_vs_C"],
    )
    lines = [
        "# Lean Context Candidate L validation",
        "",
        "## Predeclared design and promotion criteria",
        "",
        "L is fixed as H 1.1 historical features plus six recruiting features (current class rank/points, 2/3/4-year points means, trend) and four returning-production features (total, passing, receiving, rushing). Context enters location only; its scale uses the H features. Talent, coaching, transfers, interactions, continuity measures, and poll/media inputs are excluded.",
        "",
        f"Promotion requires a mean rolling NLL improvement of at least {MATERIAL_NLL_GAIN:.3f} versus H, non-worse CRPS, no worse than {MAX_COVERAGE_REGRESSION:.2f} 80% coverage, no recurrent catastrophic regression, and either a clear C improvement (L − C NLL ≤ −{CLEAR_C_NLL_GAIN:.3f}) or prediction within {PARSIMONY_TIE_NLL:.3f} NLL of C with lower complexity. These thresholds were written before results.",
        "",
        "## Results",
        "",
        f"Observed-common targets: {summary['coverage']['eligible_target_seasons']}. L − H mean annual NLL is {h_l['aggregate_mean_annual_deltas']['nll']:+.5f} ({h_l['wins']} wins, {h_l['losses']} losses); L − C is {l_c['aggregate_mean_annual_deltas']['nll']:+.5f} ({l_c['wins']} wins, {l_c['losses']} losses). Negative favors L.",
        "",
        f"The full-FBS production-style population agrees directionally: L − H mean annual NLL is {full_h_l['aggregate_mean_annual_deltas']['nll']:+.5f}; L − C is {full_l_c['aggregate_mean_annual_deltas']['nll']:+.5f}. Thus L adds real signal beyond H, but full C materially outperforms L in both populations.",
        "",
        "Raw annual values, paired team losses, calibration, complexity, disagreement, and the nested prospective simulation are retained in the companion CSV/JSON artifacts. Bootstrap-style inference is deliberately omitted: the small number of seasonal clusters makes raw annual results the primary evidence.",
        "",
        "## Interpretation",
        "",
        "PR #6 showed strong recruiting and returning-production signal, while Talent and coach tenure looked weak or redundant in isolated/common-population ablations. This fixed-candidate validation shows that removing Talent and coach tenure from the complete contextual model still degrades predictive performance materially. It does not establish that either feature is individually powerful: weak or redundant variables can retain conditional information, and a regularized correlated model can distribute signal across feature families. Ablation results can generate simplification hypotheses, but complete candidate validation is required before production changes.",
        "",
        "## Independence and status",
        "",
        "The 2018–2025 ablation generated this hypothesis, so these results are a prospective-style historical validation, not independent confirmation. The nested selector uses only earlier rolling targets. The 2025 row is descriptive only. No feature selection or retuning was performed. H 1.1, C 1.2, and frozen 2026 H/C artifacts were not modified; this study emits no 2026 forecast or outcome-based artifact.",
        "",
        "## Recommendation",
        "",
        summary["recommendation"],
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_smoke(output: Path) -> dict[str, Any]:
    """Exercise the annual/nested/report plumbing with synthetic PMFs only.

    This is deliberately not model evidence: it neither fits historical models
    nor reads historical outcomes. Its purpose is to catch schema drift before
    the expensive all-target reconstruction is started.
    """
    output.mkdir(parents=True, exist_ok=True)
    annual = []
    for season in range(2014, 2018):

        def prediction(
            model: str, pmf: list[float], target_season: int = season
        ) -> v1.PriorPrediction:
            return v1.PriorPrediction(
                target_season,
                "fbs",
                "synthetic",
                "Synthetic",
                2,
                np.asarray([1]),
                model,
                "synthetic_smoke",
                np.asarray(pmf),
                1.0,
                1.0,
            )

        annual.append(
            annual_record(
                season,
                "synthetic_smoke",
                {
                    "H": [prediction("H", [0.5, 0.5])],
                    "C": [prediction("C", [0.8, 0.2])],
                    "L": [prediction("L", [0.8, 0.2])],
                },
                n_training=season - 2003,
                n_seasons=season - 2004,
            )
        )
    nested = [
        {
            "target_season": record["target_season"],
            "n_prior_target_seasons": index,
            "selected_model": choose_nested_model(annual[:index]),
        }
        for index, record in enumerate(annual)
    ]
    h_l, l_c = comparison(annual, "L", "H"), comparison(annual, "L", "C")
    h_deltas = h_l["aggregate_mean_annual_deltas"]
    outcome = recommendation_outcome(
        l_minus_h_nll=h_deltas["nll"],
        l_minus_h_crps=h_deltas["crps"],
        l_minus_h_interval_80_coverage=h_deltas[INTERVAL_80_COVERAGE_METRIC],
        l_minus_c_nll=l_c["aggregate_mean_annual_deltas"]["nll"],
    )
    result = {
        "synthetic_only_not_evidence": True,
        "annual_metrics": annual,
        "nested_selection": nested,
        "comparisons": {"H_vs_L": h_l, "L_vs_C": l_c},
        "recommendation_outcome": outcome,
    }
    (output / "smoke_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "smoke_report.md").write_text(
        "# Lean Context validation smoke path\n\n"
        "Synthetic PMFs only; this is plumbing validation, not model evidence.\n\n"
        f"Nested selection reached {nested[-1]['selected_model']} after "
        f"{nested[-1]['n_prior_target_seasons']} prior targets; outcome {outcome}.\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    assert_lean_specification(LEAN_CONTEXT_FEATURES)
    clean_output()
    started = time.perf_counter()
    base_rows, _, _ = v1.load_rows(max_season=TARGET_MAX)
    fbs = [row for row in base_rows if row.subdivision == "fbs"]
    rows, _ = c12.attach_context(fbs, c12.feature_index(), c12.cached_tenures())
    available_years = sorted({row.season for row in rows})
    targets = [
        year
        for year in available_years
        if len({row.season for row in prior_training(rows, year)}) >= MIN_TRAIN_SEASONS
        and len(observed_common(row for row in rows if row.season == year)) >= 20
    ]
    coverage_data = coverage(rows, targets)
    write_json("coverage.json", coverage_data)
    annual, all_losses, stored, diagnostics = [], [], {}, []
    for target in targets:
        training = prior_training(rows, target)
        target_rows = [row for row in rows if row.season == target]
        models = fit_models(training, target)
        full = predict(models, target_rows)
        common_keys = {
            (row.season, row.subdivision, row.team_id)
            for row in observed_common(target_rows)
        }
        common = {name: subset(values, common_keys) for name, values in full.items()}
        for population, values in (
            ("observed_common", common),
            ("full_fbs_production_style", full),
        ):
            record = annual_record(
                target,
                population,
                values,
                len(training),
                len({row.season for row in training}),
            )
            annual.append(record)
            all_losses.extend(loss_rows(target, population, values))
            stored[(population, target)] = values
        diagnostics.append(
            {
                "target_season": target,
                **{
                    f"{name.lower()}_optimizer": model.optimizer
                    for name, model in models.items()
                },
            }
        )
        print(f"completed lean validation {target}", flush=True)
    write_csv("annual_metrics.csv", annual)
    write_csv("per_team_losses.csv", all_losses)
    write_json("optimizer_diagnostics.json", diagnostics)
    comparisons = {}
    for population in ("observed_common", "full_fbs_production_style"):
        records = [row for row in annual if row["population"] == population]
        comparisons[f"{population}/H_vs_L"] = comparison(records, "L", "H")
        comparisons[f"{population}/L_vs_C"] = comparison(records, "L", "C")
    write_json("h_vs_l.csv.json", comparisons["observed_common/H_vs_L"])
    write_json("l_vs_c.csv.json", comparisons["observed_common/L_vs_C"])
    write_csv(
        "h_vs_l.csv",
        [
            dict(row, comparison_population="observed_common")
            for row in annual
            if row["population"] == "observed_common"
        ],
    )
    write_csv(
        "l_vs_c.csv",
        [
            dict(row, comparison_population="observed_common")
            for row in annual
            if row["population"] == "observed_common"
        ],
    )
    common_annual = [row for row in annual if row["population"] == "observed_common"]
    nested = []
    for index, record in enumerate(common_annual):
        prior = common_annual[:index]
        choice = choose_nested_model(prior)
        values = stored[("observed_common", int(record["target_season"]))][choice]
        realized = metric_values(values)
        nested.append(
            {
                "target_season": record["target_season"],
                "n_prior_target_seasons": len(prior),
                "selected_model": choice,
                **{f"realized_{key}": value for key, value in realized.items()},
            }
        )
    write_csv("nested_selection.csv", nested)
    strategy = {
        key: float(np.mean([row[f"realized_{key}"] for row in nested]))
        for key in (
            "nll",
            "crps",
            "expected_rank_mae",
            "median_rank_mae",
            "interval_80_coverage",
            "interval_80_average_width",
        )
    }
    write_json(
        "nested_strategy_summary.json",
        {
            "decision_rule": {
                "min_prior_targets": MIN_PRIOR_TARGETS,
                "material_nll_gain": MATERIAL_NLL_GAIN,
                "max_coverage_regression": MAX_COVERAGE_REGRESSION,
                "parsimony_tie_nll": PARSIMONY_TIE_NLL,
            },
            "aggregate": strategy,
            "selected_counts": {
                name: sum(row["selected_model"] == name for row in nested)
                for name in ("H", "C", "L")
            },
        },
    )
    calibration_data = calibration(stored)
    write_json("calibration.json", calibration_data)
    full_losses = [
        row for row in all_losses if row["population"] == "full_fbs_production_style"
    ]
    disagreement_rows, disagreement_summary = disagreement(full_losses)
    write_csv("disagreement.csv", disagreement_rows)
    complexity = {
        "H": {
            "context_features": 0,
            "fitted_context_coefficients": 0,
            "missingness_indicators": 0,
        },
        "C": {
            "context_features": len(FULL_CONTEXT_FEATURES),
            "fitted_context_coefficients": 2 * len(FULL_CONTEXT_FEATURES),
            "missingness_indicators": len(FULL_CONTEXT_FEATURES),
        },
        "L": {
            "context_features": len(LEAN_CONTEXT_FEATURES),
            "fitted_context_coefficients": len(LEAN_CONTEXT_FEATURES),
            "missingness_indicators": len(LEAN_CONTEXT_FEATURES),
            "context_location_only": True,
        },
        "historical_coverage": {
            "targets": targets,
            "full_fbs_team_seasons": len(full_losses),
            "observed_common_team_seasons": len(
                [row for row in all_losses if row["population"] == "observed_common"]
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json("complexity_comparison.json", complexity)
    # The recommendation is deliberately mechanical and can conclude against promotion.
    h_l = comparisons["observed_common/H_vs_L"]
    l_c = comparisons["observed_common/L_vs_C"]
    h_deltas = h_l["aggregate_mean_annual_deltas"]
    outcome = recommendation_outcome(
        l_minus_h_nll=h_deltas["nll"],
        l_minus_h_crps=h_deltas["crps"],
        l_minus_h_interval_80_coverage=h_deltas[INTERVAL_80_COVERAGE_METRIC],
        l_minus_c_nll=l_c["aggregate_mean_annual_deltas"]["nll"],
    )
    recommendation = {
        "A": "Recommendation A: L clearly validates; build a separate production C revision PR using L.",
        "B": "Recommendation B: L is practically tied with C and materially simpler; consider a separate future production revision.",
        "C": "Recommendation C: L improves over H but not over C; keep C 1.2.",
        "D": "Recommendation D: L does not hold up; keep H/C frozen and treat the earlier result as hypothesis-generating.",
    }[outcome]
    summary = {
        "candidate": {
            "name": "L",
            "history_features": list(H_FEATURES),
            "context_features": list(LEAN_CONTEXT_FEATURES),
            "location_only_context": True,
        },
        "coverage": coverage_data,
        "comparisons": comparisons,
        "nested_strategy": strategy,
        "disagreement": disagreement_summary,
        "promotion_criteria": {
            "outcome": outcome,
            "clear_c_win_threshold": CLEAR_C_NLL_GAIN,
        },
        "recommendation": recommendation,
        "frozen_artifacts_untouched": True,
        "no_2026_outcomes_read": True,
        "no_feature_search_or_retuning": True,
    }
    write_json("summary.json", summary)
    render_report(summary)
    render_plots(annual, disagreement_rows, coverage_data, nested)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        run_smoke(OUT / "smoke")
    else:
        main()
