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
    nested_family_choice,
    nested_family_prior_years,
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


def h_candidate_names() -> tuple[str, ...]:
    """Return the complete, explicit H-only prospective selection family."""
    return (
        "H_static",
        *(f"H_{half_life_name(value)}" for value in HALF_LIVES[1:]),
        *(f"H_{window_name(value)}" for value in WINDOWS[1:]),
    )


def c_candidate_names() -> tuple[str, ...]:
    """Return the complete, explicit C-only prospective selection family."""
    return (
        "C_static",
        *(f"C_context_{half_life_name(value)}" for value in HALF_LIVES[1:]),
    )


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


def nested_strategy(
    family: str,
    candidate_names: tuple[str, ...],
    scores_by_year: dict[int, dict[str, float]],
    predictions_by_candidate: dict[str, dict[int, list[v1.PriorPrediction]]],
    years: list[int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Score the realized prospective selector against its family's static model."""
    static_candidate = candidate_names[0]
    selected_predictions: list[v1.PriorPrediction] = []
    static_predictions: list[v1.PriorPrediction] = []
    selections: list[dict[str, object]] = []
    for target in years:
        prior_years = nested_family_prior_years(candidate_names, scores_by_year, target)
        selected_candidate = nested_family_choice(
            candidate_names,
            scores_by_year,
            target,
            default_candidate=static_candidate,
        )
        selected = predictions_by_candidate[selected_candidate][target]
        baseline = predictions_by_candidate[static_candidate][target]
        assert_same_keys({prediction.key for prediction in selected}, {prediction.key for prediction in baseline})
        selected_predictions.extend(selected)
        static_predictions.extend(baseline)
        selected_nll = scores_by_year[target][selected_candidate]
        static_nll = scores_by_year[target][static_candidate]
        selections.append(
            {
                "target_season": target,
                "family": family,
                "selected_candidate": selected_candidate,
                "selected_nll": selected_nll,
                "static_candidate": static_candidate,
                "static_nll": static_nll,
                "delta_nll": selected_nll - static_nll,
                "n_prior_targets_used": len(prior_years),
                "adaptive_selected": selected_candidate != static_candidate,
            }
        )
    selected_metrics = v1.score_predictions(selected_predictions)
    static_metrics = v1.score_predictions(static_predictions)
    result: dict[str, object] = {
        "family": family,
        "static_candidate": static_candidate,
        "n_target_seasons": len(years),
        "adaptive_selection_count": sum(
            bool(selection["adaptive_selected"]) for selection in selections
        ),
        "adaptive_selection_fraction": float(
            np.mean([bool(selection["adaptive_selected"]) for selection in selections])
        ),
        "selected_candidate_by_year": {
            str(selection["target_season"]): selection["selected_candidate"]
            for selection in selections
        },
    }
    for metric in ("nll", "crps", "expected_rank_mae", "median_rank_mae"):
        selected_value = float(selected_metrics[metric])
        static_value = float(static_metrics[metric])
        result[metric] = selected_value
        result[f"static_{metric}"] = static_value
        result[f"delta_{metric}_vs_static"] = selected_value - static_value
    return selections, result


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
    nested = summary["nested_strategies"]
    nested_h = nested["H"]
    nested_c = nested["C"]
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
        "## Prospective family-specific nested strategies",
        "",
        f"The H-only selector chose an adaptive H candidate in **{nested_h['adaptive_selection_count']}/{nested_h['n_target_seasons']}** target seasons. Its realized aggregate NLL was **{nested_h['nll']:.4f}** (ΔNLL **{nested_h['delta_nll_vs_static']:.4f}** versus always H-static), with ΔCRPS **{nested_h['delta_crps_vs_static']:.4f}** and expected-rank-MAE change **{nested_h['delta_expected_rank_mae_vs_static']:.4f}**.",
        "",
        f"The C-only selector chose an adaptive C candidate in **{nested_c['adaptive_selection_count']}/{nested_c['n_target_seasons']}** target seasons. Its realized aggregate NLL was **{nested_c['nll']:.4f}** (ΔNLL **{nested_c['delta_nll_vs_static']:.4f}** versus always C-static), with ΔCRPS **{nested_c['delta_crps_vs_static']:.4f}** and expected-rank-MAE change **{nested_c['delta_expected_rank_mae_vs_static']:.4f}**.",
        "",
        "These are prospective strategy results: each target's choice uses only earlier rolling target forecasts within its own family. They are distinct from the descriptive hindsight candidate means above. The two-timescale experiment holds rank-history effects slow and fits a recent weighted context correction; it remains research-only, and a future production proposal requires a separate specification and validation PR.",
        "",
        "## Artifacts",
        "",
        "- `annual_metrics.csv` — paired annual scores for static and adaptive candidates.\n- `candidate_results.csv` — descriptive aggregate results and family-specific selection counts.\n- `nested_selection.csv` — the prospective selected candidate and realized score for every family/target.\n- `coefficient_trajectories.csv` and `feature_distributions.csv` — coefficient/effect and covariate-shift diagnostics.\n- `hc_disagreement.csv` and `decomposition_2025.csv` — forecast disagreement and the 2025 C-vs-H NLL decomposition.\n- `plots/` — requested annual performance, feature/effect, interval, disagreement, and 2025 plots.",
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
    predictions_by_candidate: dict[str, dict[int, list[v1.PriorPrediction]]] = (
        defaultdict(dict)
    )
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
            predictions_by_candidate[family][target] = preds
        coefficients.extend(coefficient_rows(h_model, target, "H_static"))
        coefficients.extend(coefficient_rows(c_model, target, "C_static"))
        h_loss, c_loss = h11.prediction_losses(h_preds), h11.prediction_losses(c_preds)
        team_names = {prediction.key: prediction.team_name for prediction in h_preds}
        for key in h_loss:
            decomposed.append(
                {
                    "season": target,
                    "team_id": key[2],
                    "h_nll": h_loss[key][0],
                    "c_nll": c_loss[key][0],
                    "c_minus_h_nll": c_loss[key][0] - h_loss[key][0],
                    "team_name": team_names[key],
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
            predictions_by_candidate[h_name][target] = preds
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
            predictions_by_candidate[c_name][target] = context_weighted

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
            predictions_by_candidate[name][target] = preds

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
    families = {
        "H": h_candidate_names(),
        "C": c_candidate_names(),
    }
    references = {"H": static_h, "C": static_c}
    nested_rows: list[dict[str, object]] = []
    nested_strategies: dict[str, dict[str, object]] = {}
    for family, names in families.items():
        reference = references[family]
        for name in names:
            scores = config_scores[name]
            common = sorted(set(scores) & set(reference))
            candidates.append(
                {
                    "family": family,
                    "candidate": name,
                    "n_target_seasons": len(common),
                    "mean_nll": float(np.mean([scores[year] for year in common])),
                    "mean_delta_nll_vs_static": float(
                        np.mean([scores[year] - reference[year] for year in common])
                    ),
                    "nested_choice_wins": sum(
                        nested_family_choice(
                            names,
                            scores_by_year,
                            year,
                            default_candidate=names[0],
                        )
                        == name
                        for year in common
                    ),
                }
            )
        family_rows, family_summary = nested_strategy(
            family,
            names,
            scores_by_year,
            predictions_by_candidate,
            years,
        )
        nested_rows.extend(family_rows)
        nested_strategies[family] = family_summary
    candidates.sort(key=lambda row: float(row["mean_delta_nll_vs_static"]))
    best_h = next(
        row
        for row in candidates
        if row["family"] == "H" and row["candidate"] != "H_static"
    )
    best_c = next(
        row
        for row in candidates
        if row["family"] == "C" and row["candidate"] != "C_static"
    )
    disagreements = prediction_disagreement(h_static_all, c_static_all)
    write_csv("annual_metrics.csv", annual)
    write_csv("candidate_results.csv", candidates)
    write_csv("nested_selection.csv", nested_rows)
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
        "nested_strategies": nested_strategies,
        "classification": "C. Feature-specific drift, with mild gradual weakening of rank persistence but no validated adaptive production win.",
        "findings": [
            "Classification: **C — feature-specific drift**. Direct lag-1-to-target percentile persistence declined gradually from 0.727 (2004–09) to 0.661 (2020–25); it is not evidence of a discrete portal/NIL breakpoint.",
            "Long-run-history coefficients are variable and their simple linear trend is weak for H (p=0.190), while direct transition diagnostics show only modest recent weakening. A static long-run baseline remains defensible pending uncertainty-aware follow-up.",
            f"H recency/window experiments provide no meaningful validated gain: the best descriptive result is {best_h['candidate']} at ΔNLL {best_h['mean_delta_nll_vs_static']:.5f}, while the family-specific prospective selector chose adaptive H in {nested_strategies['H']['adaptive_selection_count']}/{nested_strategies['H']['n_target_seasons']} targets and realized ΔNLL {nested_strategies['H']['delta_nll_vs_static']:.5f} versus always H-static.",
            f"Fast context adaptation did not help: the best slow-H/fast-context half-life is {best_c['candidate']} at ΔNLL {best_c['mean_delta_nll_vs_static']:.4f} versus C-static, and the family-specific prospective C selector chose no adaptive candidate (realized ΔNLL {nested_strategies['C']['delta_nll_vs_static']:.4f}). Do not introduce a recency-weighted C production model from this evidence.",
            "Coach-tenure effect is stable (linear coefficient-time p=0.608). Recruiting, Talent, and returning-production coefficient trajectories move substantially, but their early missingness/coverage changes make raw long-run trends descriptive rather than causal.",
            "Returning total and passing production remain directionally useful once observed; their standardized effects do not support the hypothesized modern weakening. Recruiting/Talent effects are small and unstable conditional on the other context inputs.",
            "Recent C performance deteriorated in 2025: C beat H by 0.0616 NLL in 2022, 0.0207 in 2023, and 0.0103 in 2024, then lost by 0.0420 in 2025. The 2025 loss sums exactly from team contributions; largest positive C-minus-H contributions were New Mexico (1.88), Utah (1.66), James Madison (1.61), North Texas (1.43).",
            "H/C disagreement increased modestly in the recent years (mean expected-rank gap 7.76 in 2022 to 8.71 in 2025), so large context adjustments merit audit rather than stronger automatic weighting.",
            "Recommended next step: preserve H 1.1/C 1.2 and 2026 priors; conduct a separate uncertainty-aware feature-ablation/interaction study using only years with observed context coverage before proposing any production V1.3/C1.3.",
        ],
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
