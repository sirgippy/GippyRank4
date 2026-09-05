"""Reproducible rolling-origin regime-stability investigation for H 1.1/C 1.2.

This is research infrastructure.  It reads frozen source and preseason
artifacts but writes only ``data/processed/regime_stability``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import build_preseason_context_prior_v1_2 as c12
import build_preseason_prior as v1
import build_preseason_prior_v1_1 as h11
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, norm

from gippyrank.preseason import (
    DirectRankModel,
    Preprocessor,
    TeamSeason,
    pmf_summaries,
    rank_bin_edges,
)
from gippyrank.regime_stability import (
    assert_same_keys,
    nested_choice,
    training_plan,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/regime_stability"
PLOTS = OUT / "plots"
TARGET_MIN, TARGET_MAX = 2008, 2025
H_FEATURES = c12.H_FEATURES
C_FEATURES = [
    *c12.COACH_FEATURES,
    *c12.RECRUITING_FEATURES,
    *c12.TALENT_FEATURES,
    *c12.RETURNING_FEATURES,
]
HALF_LIVES: list[float | None] = [None, 15.0, 10.0, 7.0, 5.0, 3.0]
WINDOWS: list[int | None] = [None, 15, 10, 7, 5]


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


def make_predictions(
    model: DirectRankModel, rows: list[TeamSeason], name: str
) -> list[v1.PriorPrediction]:
    return h11.make_predictions(model, rows, name)


def fit_h(rows: list[TeamSeason], weights: np.ndarray | None = None) -> DirectRankModel:
    return DirectRankModel.fit(rows, H_FEATURES, penalty=0.25, row_weights=weights)


def fit_c(rows: list[TeamSeason], weights: np.ndarray | None = None) -> DirectRankModel:
    """Exact C 1.2 feature/equation layout; weights are research-only."""
    return DirectRankModel.fit(
        rows,
        [*H_FEATURES, *C_FEATURES],
        penalty=0.25,
        location_feature_names=[*H_FEATURES, *C_FEATURES],
        scale_feature_names=H_FEATURES,
        row_weights=weights,
        optimizer_options={"maxiter": 300, "ftol": 1e-7, "gtol": 1e-5},
    )


def half_life_name(half_life: float | None) -> str:
    return "equal" if half_life is None else f"hl_{half_life:g}"


def window_name(window: int | None) -> str:
    return "all" if window is None else f"window_{window}"


def flatten_metrics(
    metrics: dict[str, object], predictions: list[v1.PriorPrediction]
) -> dict[str, object]:
    result = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    for cutoff in (5, 10, 25):
        result[f"top{cutoff}_brier"] = float(
            np.mean(
                [
                    (
                        pmf_summaries(prediction.pmf)[f"top{cutoff}_probability"]
                        - float(np.mean(prediction.target_ranks <= cutoff))
                    )
                    ** 2
                    for prediction in predictions
                ]
            )
        )
    return result


def weighted_adjustment(
    history: DirectRankModel,
    rows: list[TeamSeason],
    weights: np.ndarray,
) -> tuple[Preprocessor, np.ndarray]:
    """Fast context correction on top of slow frozen-layout H effects.

    The correction is deliberately a small ridge WLS model on H's location
    residual.  It is a two-timescale experiment, not a replacement C model.
    """
    prep = Preprocessor.fit([row.features for row in rows], C_FEATURES)
    x = np.column_stack(
        [np.ones(len(rows)), prep.transform([row.features for row in rows])]
    )
    residual = np.asarray(
        [
            np.mean(row.target_z)
            - np.mean(
                history.conditional_parameters(row.features, row.lag1_z, row.lag_zs)[0]
            )
            for row in rows
        ]
    )
    root_w = np.sqrt(weights)[:, None]
    ridge = np.eye(x.shape[1]) * 0.25
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(
        (x * root_w).T @ (x * root_w) + ridge,
        (x * root_w).T @ (residual * root_w[:, 0]),
    )
    return prep, beta


def hybrid_predictions(
    history: DirectRankModel,
    prep: Preprocessor,
    beta: np.ndarray,
    rows: list[TeamSeason],
    name: str,
) -> list[v1.PriorPrediction]:
    result = []
    for row in rows:
        locations, scale = history.conditional_parameters(
            row.features, row.lag1_z, row.lag_zs
        )
        adjustment = float(np.r_[1.0, prep.transform([row.features])[0]] @ beta)
        edges = rank_bin_edges(row.population)
        masses = np.diff(
            norm.cdf((edges[None, :] - (locations + adjustment)[:, None]) / scale),
            axis=1,
        )
        pmf = np.maximum(masses.mean(axis=0), 0.0)
        pmf /= pmf.sum()
        result.append(
            v1.PriorPrediction(
                row.season,
                row.subdivision,
                row.team_id,
                row.team_name,
                row.population,
                row.target_ranks,
                name,
                "same_subdivision_lag1",
                pmf,
                float(np.mean(locations + adjustment)),
                scale,
            )
        )
    return result


def coefficient_rows(
    model: DirectRankModel, season: int, family: str
) -> list[dict[str, object]]:
    records = []
    for index, feature in enumerate(model.feature_names):
        # beta begins with the lag coefficient; then the intercept and standardised features.
        coefficient = float(model.beta[model.lag_count + 1 + index])
        records.append(
            {
                "target_season": season,
                "family": family,
                "feature": feature,
                "standardized_location_coefficient": coefficient,
                "practical_effect_one_training_sd_z": coefficient,
            }
        )
    return records


def feature_distribution(rows: list[TeamSeason]) -> list[dict[str, object]]:
    records = []
    eras = [
        (2003, 2009, "2003-2009"),
        (2010, 2014, "2010-2014"),
        (2015, 2019, "2015-2019"),
        (2020, 2025, "2020-2025"),
    ]
    for low, high, label in eras:
        subset = [row for row in rows if low <= row.season <= high]
        for feature in C_FEATURES:
            values = np.asarray(
                [
                    row.features.get(feature)
                    for row in subset
                    if row.features.get(feature) is not None
                ],
                dtype=float,
            )
            records.append(
                {
                    "era": label,
                    "start_season": low,
                    "end_season": high,
                    "feature": feature,
                    "n": len(subset),
                    "n_observed": len(values),
                    "missingness": 1 - len(values) / len(subset),
                    "mean": float(values.mean()) if len(values) else None,
                    "median": float(np.median(values)) if len(values) else None,
                    "variance": float(values.var()) if len(values) else None,
                    "p10": float(np.quantile(values, 0.1)) if len(values) else None,
                    "p90": float(np.quantile(values, 0.9)) if len(values) else None,
                }
            )
    return records


def direct_transition_diagnostics(rows: list[TeamSeason]) -> list[dict[str, object]]:
    """Describe observed rank persistence directly, without model coefficients."""
    eras = [
        (2004, 2009, "2004-2009"),
        (2010, 2014, "2010-2014"),
        (2015, 2019, "2015-2019"),
        (2020, 2025, "2020-2025"),
    ]
    result = []
    for low, high, label in eras:
        subset = [row for row in rows if low <= row.season <= high]
        prior = np.asarray(
            [np.mean(1 / (1 + np.exp(-row.lag1_z))) for row in subset], dtype=float
        )
        outcome = np.asarray(
            [np.mean(row.target_ranks / row.population) for row in subset], dtype=float
        )
        slope = float(np.cov(prior, outcome, ddof=0)[0, 1] / np.var(prior))
        result.append(
            {
                "era": label,
                "start_season": low,
                "end_season": high,
                "n_team_seasons": len(subset),
                "lag1_to_target_percentile_slope": slope,
                "lag1_target_correlation": float(np.corrcoef(prior, outcome)[0, 1]),
                "mean_next_percentile_given_top_quartile_lag1": float(
                    outcome[prior <= np.quantile(prior, 0.25)].mean()
                ),
                "mean_next_percentile_given_bottom_quartile_lag1": float(
                    outcome[prior >= np.quantile(prior, 0.75)].mean()
                ),
            }
        )
    return result


def coefficient_time_trends(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Conservative linear-time diagnostic; not a fitted calendar breakpoint."""
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), str(row["feature"]))].append(row)
    result = []
    for (family, feature), group in sorted(grouped.items()):
        x = np.asarray([float(row["target_season"]) for row in group])
        y = np.asarray(
            [float(row["standardized_location_coefficient"]) for row in group]
        )
        fit = linregress(x, y)
        result.append(
            {
                "family": family,
                "feature": feature,
                "n_targets": len(group),
                "coefficient_change_per_year": float(fit.slope),
                "linear_time_p_value": float(fit.pvalue),
                "r_squared": float(fit.rvalue**2),
            }
        )
    return result


def prediction_disagreement(
    h_preds: list[v1.PriorPrediction], c_preds: list[v1.PriorPrediction]
) -> list[dict[str, object]]:
    h_index, c_index = ({p.key: p for p in h_preds}, {p.key: p for p in c_preds})
    assert_same_keys(set(h_index), set(c_index))
    result = []
    for key, hp in h_index.items():
        cp = c_index[key]
        hs, cs = pmf_summaries(hp.pmf), pmf_summaries(cp.pmf)
        midpoint = (hp.pmf + cp.pmf) / 2
        js = 0.5 * np.sum(
            hp.pmf * np.log(np.maximum(hp.pmf, 1e-15) / np.maximum(midpoint, 1e-15))
        ) + 0.5 * np.sum(
            cp.pmf * np.log(np.maximum(cp.pmf, 1e-15) / np.maximum(midpoint, 1e-15))
        )
        result.append(
            {
                "season": key[0],
                "team_id": key[2],
                "expected_rank_absolute_difference": abs(
                    hs["expected_rank"] - cs["expected_rank"]
                ),
                "js_divergence": float(js),
                "moved_over_5": abs(hs["expected_rank"] - cs["expected_rank"]) > 5,
                "moved_over_10": abs(hs["expected_rank"] - cs["expected_rank"]) > 10,
                "moved_over_20": abs(hs["expected_rank"] - cs["expected_rank"]) > 20,
            }
        )
    return result


def plot_series(
    rows: list[dict[str, object]],
    x: str,
    ys: list[str],
    title: str,
    name: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(8, 4.5))
    for y in ys:
        values = [(r[x], r[y]) for r in rows if r.get(y) is not None]
        if values:
            plt.plot(*zip(*values, strict=True), marker="o", label=y.replace("_", " "))
    plt.title(title)
    plt.xlabel("Target season")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS / name, dpi=160)
    plt.close()


def render_report(summary: dict[str, object]) -> None:
    best_h = summary["best_h"]
    best_c = summary["best_c"]
    lines = [
        "# Regime-stability investigation",
        "",
        "## Scope and guardrails",
        "",
        "This research refits historical rolling-origin models only. H 1.1, C 1.2, Historical Likelihood V1, and the frozen 2026 preseason PMFs were not modified. Every target-season fit uses outcomes only from earlier seasons; 2026 is excluded.",
        "",
        "## Rolling-origin evidence",
        "",
        f"Targets span {TARGET_MIN}–{TARGET_MAX}; paired H/C scoring uses identical regular FBS team keys. The best descriptive H recency candidate was **{best_h['candidate']}** (mean ΔNLL {best_h['delta_nll']:.4f} versus H-static). The best descriptive C recency candidate was **{best_c['candidate']}** (mean ΔNLL {best_c['delta_nll']:.4f} versus C-static).",
        "",
        "The machine-readable tables retain the annual NLL, CRPS, expected/median-rank MAE, interval coverage and width, and Top-5/10/25 Brier scores. Plots show raw annual points; no rule-change date was fit as a breakpoint.",
        "",
        "## Interpretation",
        "",
        "Treat the candidate rankings as descriptive, not a production selection: each target-year nested choice is calculated from earlier target forecasts only. The two-timescale experiment holds rank-history effects slow and fits a recent weighted context correction; it is intentionally simple. A future production proposal requires a separate specification and validation PR.",
        "",
        "## Artifacts",
        "",
        "- `annual_metrics.csv` — paired annual scores for static and adaptive candidates.\n- `candidate_results.csv` — aggregate and nested-selection summaries.\n- `coefficient_trajectories.csv` and `feature_distributions.csv` — coefficient/effect and covariate-shift diagnostics.\n- `hc_disagreement.csv` and `decomposition_2025.csv` — forecast disagreement and the 2025 C-vs-H NLL decomposition.\n- `plots/` — requested annual performance, feature/effect, interval, disagreement, and 2025 plots.",
    ]
    findings = summary.get("findings")
    if isinstance(findings, list):
        lines += ["", "## Findings from this run", ""]
        lines.extend(f"- {finding}" for finding in findings)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-min", type=int, default=TARGET_MIN)
    parser.add_argument("--target-max", type=int, default=TARGET_MAX)
    args = parser.parse_args()
    target_min, target_max = args.target_min, args.target_max
    rows, _cold, _coverage = v1.load_rows(max_season=target_max)
    fbs = [row for row in rows if row.subdivision == "fbs"]
    contextual, _ = c12.attach_context(fbs, c12.feature_index(), c12.cached_tenures())
    context_by_key = {
        (row.season, row.subdivision, row.team_id): row for row in contextual
    }
    years = [
        year
        for year in range(target_min, target_max + 1)
        if sum(r.season == year for r in fbs) >= 20
    ]
    annual: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    h_static_all: list[v1.PriorPrediction] = []
    c_static_all: list[v1.PriorPrediction] = []
    config_scores: dict[str, dict[int, float]] = defaultdict(dict)
    decomposed: list[dict[str, object]] = []

    for target in years:
        target_h = [row for row in fbs if row.season == target]
        target_c = [
            context_by_key[row.season, row.subdivision, row.team_id] for row in target_h
        ]
        base_plan = training_plan(fbs, target)
        base_c_plan = training_plan(contextual, target)
        h_model = fit_h(base_plan.rows)
        c_model = fit_c(base_c_plan.rows)
        h_preds, c_preds = (
            make_predictions(h_model, target_h, "H_static"),
            make_predictions(c_model, target_c, "C_static"),
        )
        assert_same_keys({p.key for p in h_preds}, {p.key for p in c_preds})
        h_static_all.extend(h_preds)
        c_static_all.extend(c_preds)
        for family, preds in (("H_static", h_preds), ("C_static", c_preds)):
            annual.append(
                {
                    "target_season": target,
                    "candidate": family,
                    **flatten_metrics(v1.score_predictions(preds), preds),
                    "training_rows": len(base_plan.rows),
                }
            )
            config_scores[family][target] = float(v1.score_predictions(preds)["nll"])
        coefficients.extend(coefficient_rows(h_model, target, "H_static"))
        coefficients.extend(coefficient_rows(c_model, target, "C_static"))
        h_loss, c_loss = h11.prediction_losses(h_preds), h11.prediction_losses(c_preds)
        team_names = {prediction.key: prediction.team_name for prediction in h_preds}
        for key in h_loss:
            decomposed.append(
                {
                    "season": target,
                    "team_id": key[2],
                    "team_name": team_names[key],
                    "h_nll": h_loss[key][0],
                    "c_nll": c_loss[key][0],
                    "c_minus_h_nll": c_loss[key][0] - h_loss[key][0],
                }
            )

        for half_life in HALF_LIVES[1:]:
            h_plan = training_plan(fbs, target, half_life=half_life)
            c_plan = training_plan(contextual, target, half_life=half_life)
            h_name = f"H_{half_life_name(half_life)}"
            preds = make_predictions(
                fit_h(h_plan.rows, h_plan.weights), target_h, h_name
            )
            metrics = v1.score_predictions(preds)
            annual.append(
                {
                    "target_season": target,
                    "candidate": h_name,
                    **flatten_metrics(metrics, preds),
                    "training_rows": len(h_plan.rows),
                }
            )
            config_scores[h_name][target] = float(metrics["nll"])
            prep, beta = weighted_adjustment(h_model, c_plan.rows, c_plan.weights)
            c_name = f"C_context_{half_life_name(half_life)}"
            context_weighted = hybrid_predictions(h_model, prep, beta, target_c, c_name)
            metrics = v1.score_predictions(context_weighted)
            annual.append(
                {
                    "target_season": target,
                    "candidate": c_name,
                    **flatten_metrics(metrics, context_weighted),
                    "training_rows": len(c_plan.rows),
                }
            )
            config_scores[c_name][target] = float(metrics["nll"])

        for window in WINDOWS[1:]:
            plan = training_plan(fbs, target, window=window)
            name = f"H_{window_name(window)}"
            preds = make_predictions(fit_h(plan.rows), target_h, name)
            metrics = v1.score_predictions(preds)
            annual.append(
                {
                    "target_season": target,
                    "candidate": name,
                    **flatten_metrics(metrics, preds),
                    "training_rows": len(plan.rows),
                }
            )
            config_scores[name][target] = float(metrics["nll"])

        print(f"completed target {target}", flush=True)

    static_h = {
        row["target_season"]: row["nll"]
        for row in annual
        if row["candidate"] == "H_static"
    }
    static_c = {
        row["target_season"]: row["nll"]
        for row in annual
        if row["candidate"] == "C_static"
    }
    scores_by_year = {
        year: {
            name: values[year]
            for name, values in config_scores.items()
            if year in values
        }
        for year in years
    }
    for name, scores in sorted(config_scores.items()):
        reference = static_h if name.startswith("H_") else static_c
        if name in {"H_static", "C_static"}:
            continue
        common = sorted(set(scores) & set(reference))
        candidates.append(
            {
                "candidate": name,
                "n_target_seasons": len(common),
                "mean_nll": float(np.mean([scores[y] for y in common])),
                "mean_delta_nll_vs_static": float(
                    np.mean([scores[y] - reference[y] for y in common])
                ),
                "nested_choice_wins": sum(
                    nested_choice(sorted(scores_by_year[y]), scores_by_year, y) == name
                    for y in common
                ),
            }
        )
    candidates.sort(key=lambda row: float(row["mean_delta_nll_vs_static"]))
    best_h = next(row for row in candidates if str(row["candidate"]).startswith("H_"))
    best_c = next(row for row in candidates if str(row["candidate"]).startswith("C_"))
    disagreements = prediction_disagreement(h_static_all, c_static_all)
    write_csv("annual_metrics.csv", annual)
    write_csv("candidate_results.csv", candidates)
    write_csv("coefficient_trajectories.csv", coefficients)
    write_csv("coefficient_time_trends.csv", coefficient_time_trends(coefficients))
    write_csv("feature_distributions.csv", feature_distribution(contextual))
    write_csv("transition_diagnostics.csv", direct_transition_diagnostics(fbs))
    write_csv("hc_disagreement.csv", disagreements)
    write_csv(
        "decomposition_2025.csv", [row for row in decomposed if row["season"] == 2025]
    )
    summary = {
        "target_seasons": years,
        "best_h": {
            "candidate": best_h["candidate"],
            "delta_nll": best_h["mean_delta_nll_vs_static"],
        },
        "best_c": {
            "candidate": best_c["candidate"],
            "delta_nll": best_c["mean_delta_nll_vs_static"],
        },
        "frozen_artifacts_modified": False,
        "no_2026_outcomes_accessed": True,
        "two_timescale_definition": "static H location/scale plus half-life-5 weighted ridge context correction",
    }
    write_json("summary.json", summary)
    render_report(summary)
    pivot = [
        {
            "target_season": y,
            "H_NLL": static_h[y],
            "C_NLL": static_c[y],
            "C_minus_H_NLL": static_c[y] - static_h[y],
        }
        for y in years
    ]
    plot_series(
        pivot, "target_season", ["H_NLL"], "H annual NLL", "h_annual_nll.png", "NLL"
    )
    plot_series(
        pivot, "target_season", ["C_NLL"], "C annual NLL", "c_annual_nll.png", "NLL"
    )
    plot_series(
        pivot,
        "target_season",
        ["C_minus_H_NLL"],
        "C minus H annual NLL",
        "c_minus_h_nll.png",
        "ΔNLL",
    )
    for feature, filename in (
        ("long_run_z_mean", "long_run_effect.png"),
        ("returning_pct_ppa", "returning_effect.png"),
        ("recruiting_class_points", "recruiting_effect.png"),
        ("talent_composite", "talent_effect.png"),
        ("coach_tenure_seasons", "coach_effect.png"),
    ):
        plot_series(
            [r for r in coefficients if r["feature"] == feature],
            "target_season",
            ["standardized_location_coefficient"],
            f"{feature} effect",
            filename,
            "standardized location effect",
        )
    plot_series(
        [r for r in annual if r["candidate"] == "H_static"],
        "target_season",
        ["interval_80_coverage", "interval_80_average_width"],
        "H interval coverage and width",
        "h_intervals.png",
        "value",
    )
    disagreement_year = [
        {
            "target_season": y,
            "mean_disagreement": float(
                np.mean(
                    [
                        r["expected_rank_absolute_difference"]
                        for r in disagreements
                        if r["season"] == y
                    ]
                )
            ),
        }
        for y in years
    ]
    plot_series(
        disagreement_year,
        "target_season",
        ["mean_disagreement"],
        "H/C expected-rank disagreement",
        "hc_disagreement.png",
        "rank positions",
    )
    decomp = [r for r in decomposed if r["season"] == 2025]
    plt.figure(figsize=(9, 4.5))
    plt.bar(
        range(len(decomp)),
        [r["c_minus_h_nll"] for r in sorted(decomp, key=lambda x: x["c_minus_h_nll"])],
        color="#b44",
    )
    plt.title("2025 C minus H team NLL contributions")
    plt.xlabel("Teams sorted by contribution")
    plt.ylabel("ΔNLL")
    plt.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS / "decomposition_2025.png", dpi=160)
    plt.close()
    adaptation = [row for row in candidates if "hl" in str(row["candidate"])]
    plt.figure(figsize=(8, 4.5))
    plt.bar(
        [str(r["candidate"]) for r in adaptation],
        [r["mean_delta_nll_vs_static"] for r in adaptation],
    )
    plt.xticks(rotation=45, ha="right")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Recency candidates versus static")
    plt.ylabel("mean ΔNLL")
    plt.tight_layout()
    plt.savefig(PLOTS / "recency_candidates.png", dpi=160)
    plt.close()
    distribution = [
        r
        for r in feature_distribution(contextual)
        if r["feature"] == "returning_pct_ppa"
    ]
    plt.figure(figsize=(8, 4.5))
    plt.errorbar(
        [r["era"] for r in distribution],
        [r["median"] or np.nan for r in distribution],
        yerr=[
            (r["p90"] - r["p10"]) / 2 if r["p90"] is not None else np.nan
            for r in distribution
        ],
        fmt="o",
    )
    plt.title("Returning-production distribution by era")
    plt.ylabel("returning PPA share")
    plt.tight_layout()
    plt.savefig(PLOTS / "feature_distributions.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
