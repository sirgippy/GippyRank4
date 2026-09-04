"""Select and score rank-history-only Preseason Prior V1.1 candidates.

This orchestrator deliberately rebuilds the V1 reference first, then uses only
pre-2022 rows to choose a single extension.  The final 2022--25 candidate fit
is performed once, after the selection record is written in memory.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import build_preseason_prior as v1
import numpy as np

from gippyrank.preseason import (
    DirectRankModel,
    TeamSeason,
    crps_discrete,
    pmf_summaries,
    team_log_score,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/preseason"
DEV_TRAIN = set(range(2004, 2018))
DEV_VALIDATION = set(range(2018, 2022))
TEST_SEASONS = {2022, 2023, 2024, 2025}
BASE_FEATURES = ["lag2_z_mean", "lag3_z_mean"]


@dataclass(frozen=True)
class Candidate:
    name: str
    features: list[str]
    lag_count: int = 1
    family: str = "normal"
    degrees_of_freedom: float | None = None
    strict_three_lag_history: bool = False


CANDIDATES = (
    Candidate("V1_A2_reference", BASE_FEATURES),
    Candidate("lag_depth_4", [*BASE_FEATURES, "lag4_z_mean"]),
    Candidate("lag_depth_5", [*BASE_FEATURES, "lag4_z_mean", "lag5_z_mean"]),
    Candidate("long_run_baseline", [*BASE_FEATURES, "long_run_z_mean"]),
    Candidate(
        "trajectory_and_reversion",
        [*BASE_FEATURES, "long_run_z_mean", "trajectory_z", "recent_shock_z"],
    ),
    Candidate(
        "historical_volatility",
        [*BASE_FEATURES, "long_run_z_mean", "lag1_disagreement", "history_z_std"],
    ),
    Candidate(
        "student_t_df_5", BASE_FEATURES, family="student_t", degrees_of_freedom=5
    ),
    Candidate(
        "full_t1_t2_t3_quadrature", [], lag_count=3, strict_three_lag_history=True
    ),
)


def eligible(rows: list[TeamSeason], candidate: Candidate) -> list[TeamSeason]:
    if not candidate.strict_three_lag_history:
        return rows
    return [
        row
        for row in rows
        if len(row.lag_zs) >= 2 and all(len(values) for values in row.lag_zs[:2])
    ]


def fit(rows: list[TeamSeason], candidate: Candidate) -> DirectRankModel:
    return DirectRankModel.fit(
        rows,
        candidate.features,
        penalty=0.25,
        lag_count=candidate.lag_count,
        family=candidate.family,
        degrees_of_freedom=candidate.degrees_of_freedom,
    )


def make_predictions(
    model: DirectRankModel, rows: list[TeamSeason], name: str
) -> list[v1.PriorPrediction]:
    predictions = []
    for row in rows:
        pmf = model.pmf(row.features, row.lag1_z, row.population, row.lag_zs)
        locations, scale = model.conditional_parameters(
            row.features, row.lag1_z, row.lag_zs
        )
        predictions.append(
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
                float(np.mean(locations)),
                scale,
            )
        )
    return predictions


def prediction_losses(
    predictions: list[v1.PriorPrediction],
) -> dict[tuple[int, str, str], tuple[float, float]]:
    return {
        prediction.key: (
            team_log_score(prediction.pmf, prediction.target_ranks),
            float(
                np.mean(
                    [
                        crps_discrete(prediction.pmf, int(rank))
                        for rank in prediction.target_ranks
                    ]
                )
            ),
        )
        for prediction in predictions
    }


def paired_bootstrap(
    reference: list[v1.PriorPrediction],
    candidate: list[v1.PriorPrediction],
    seed: int = 7,
) -> dict[str, object]:
    """Season-resampling uncertainty for candidate minus reference scores."""
    ref, alt = prediction_losses(reference), prediction_losses(candidate)
    keys = sorted(set(ref) & set(alt))
    grouped: dict[int, list[tuple[float, float]]] = {}
    for key in keys:
        grouped.setdefault(key[0], []).append(
            (alt[key][0] - ref[key][0], alt[key][1] - ref[key][1])
        )
    seasons = np.asarray(sorted(grouped), dtype=int)
    season_sums = np.asarray(
        [np.sum(grouped[int(season)], axis=0) for season in seasons]
    )
    season_counts = np.asarray([len(grouped[int(season)]) for season in seasons])
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(seasons), size=(2000, len(seasons)))
    samples = season_sums[drawn].sum(axis=1) / season_counts[drawn].sum(axis=1)[:, None]
    return {
        "n_team_seasons": len(keys),
        "resampling": "season bootstrap; candidate minus V1, 2,000 deterministic resamples",
        "mean_delta_nll": float(np.mean(samples[:, 0])),
        "nll_95_interval": [
            float(value) for value in np.quantile(samples[:, 0], [0.025, 0.975])
        ],
        "fraction_candidate_better_nll": float(np.mean(samples[:, 0] < 0)),
        "mean_delta_crps": float(np.mean(samples[:, 1])),
        "crps_95_interval": [
            float(value) for value in np.quantile(samples[:, 1], [0.025, 0.975])
        ],
        "fraction_candidate_better_crps": float(np.mean(samples[:, 1] < 0)),
    }


def tier_diagnostics(predictions: list[v1.PriorPrediction]) -> dict[str, object]:
    tiers = (
        ("top_10", 0, 10),
        ("11_25", 10, 25),
        ("26_50", 25, 50),
        ("51_100", 50, 100),
        ("101_plus", 100, np.inf),
    )
    result: dict[str, object] = {}
    for name, low, high in tiers:
        subset = [
            prediction
            for prediction in predictions
            if low < pmf_summaries(prediction.pmf)["expected_rank"] <= high
        ]
        if not subset:
            continue
        metrics = v1.score_predictions(subset)
        result[name] = {
            "n_team_seasons": len(subset),
            "nll": metrics["nll"],
            "interval_80_coverage": metrics["interval_80_coverage"],
            "interval_80_average_width": metrics["interval_80_average_width"],
            "average_predictive_scale": float(
                np.mean(
                    [
                        p.predictive_scale
                        for p in subset
                        if p.predictive_scale is not None
                    ]
                )
            ),
        }
    return result


def transition_diagnostics(rows: list[TeamSeason]) -> dict[str, object]:
    """Direct rank-transition summaries, restricted to pre-2022 observations."""
    tiers = (
        ("top_10pct", 0.0, 0.10),
        ("11_25pct", 0.10, 0.25),
        ("26_50pct", 0.25, 0.50),
        ("51_75pct", 0.50, 0.75),
        ("bottom_25pct", 0.75, 1.01),
    )
    summaries = []
    for name, low, high in tiers:
        group = []
        for row in rows:
            prior_percentile = float(np.mean(1 / (1 + np.exp(-row.lag1_z))))
            if low <= prior_percentile < high:
                group.append(row)
        if not group:
            continue
        summaries.append(
            {
                "prior_rank_tier": name,
                "n_team_seasons": len(group),
                "mean_next_rank_percentile": float(
                    np.mean(
                        [np.mean(row.target_ranks / row.population) for row in group]
                    )
                ),
                "next_rank_percentile_std": float(
                    np.std(
                        [np.mean(row.target_ranks / row.population) for row in group]
                    )
                ),
                "mean_prior_constituent_disagreement": float(
                    np.mean([np.std(row.lag1_z) for row in group])
                ),
            }
        )
    prior_disagreement = np.asarray([np.std(row.lag1_z) for row in rows])
    next_disagreement = np.asarray([np.std(row.target_z) for row in rows])
    return {
        "population": "FBS team-seasons with seasons < 2022",
        "conditional_on_lag1_tier": summaries,
        "prior_vs_next_constituent_disagreement_correlation": float(
            np.corrcoef(prior_disagreement, next_disagreement)[0, 1]
        ),
    }


def read_v1_predictions() -> list[v1.PriorPrediction]:
    with (OUT / "rank_prior_predictions.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        if (
            row["model"] != "A2_t1_t2_t3"
            or row["subdivision"] != "fbs"
            or int(row["season"]) not in TEST_SEASONS
        ):
            continue
        result.append(
            v1.PriorPrediction(
                int(row["season"]),
                row["subdivision"],
                row["team_id"],
                row["team_name"],
                len(json.loads(row["pmf"])),
                np.asarray([], dtype=int),
                row["model"],
                row["prior_method"],
                np.asarray(json.loads(row["pmf"]), dtype=float),
                float(row["conditional_location_mean"])
                if row["conditional_location_mean"]
                else None,
                float(row["predictive_scale"]) if row["predictive_scale"] else None,
            )
        )
    return result


def join_targets(
    predictions: list[v1.PriorPrediction],
    rows: list[TeamSeason],
    cold: list[v1.ColdStartSeason],
) -> list[v1.PriorPrediction]:
    targets = {
        (row.season, row.subdivision, row.team_id): row.target_ranks for row in rows
    }
    targets.update(
        {(row.season, row.subdivision, row.team_id): row.target_ranks for row in cold}
    )
    return [
        v1.PriorPrediction(
            p.season,
            p.subdivision,
            p.team_id,
            p.team_name,
            p.population,
            targets[p.key],
            p.model,
            p.prior_method,
            p.pmf,
            p.conditional_location_mean,
            p.predictive_scale,
        )
        for p in predictions
    ]


def render_report(report: dict[str, object]) -> None:
    selected = report["selection"]["selected"]
    lines = [
        "# GippyRank4 Preseason Prior V1.1",
        "",
        "## Recommendation",
        "",
        f"The frozen rank-history-only production specification is **{selected}**. Selection used only 2018–2021 development holdouts after fitting on 2004–2017; 2022–2025 was evaluated only after this choice. Cold starts retain V1 handling.",
        "",
        "## Development experiments",
        "",
        "| Candidate | N | NLL | CRPS | 80% coverage | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, item in report["development"].items():
        metric = item["metrics"]
        lines.append(
            f"| {name} | {metric['n_team_seasons']} | {metric['nll']:.4f} | {metric['crps']:.4f} | {metric['interval_80_coverage']:.3f} | {item['notes']} |"
        )
    final = report["final_test"]
    lines += [
        "",
        "## Untouched 2022–2025 FBS production comparison",
        "",
        "| Model | N | NLL | CRPS | Expected-rank MAE | 80% coverage | 80% width |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metric in final["aggregate"].items():
        lines.append(
            f"| {name} | {metric['n_team_seasons']} | {metric['nll']:.4f} | {metric['crps']:.4f} | {metric['expected_rank_mae']:.2f} | {metric['interval_80_coverage']:.3f} | {metric['interval_80_average_width']:.1f} |"
        )
    paired = final["paired_bootstrap"]
    lines += [
        "",
        (
            "The season-bootstrap candidate-minus-V1 ΔNLL is "
            f"{paired['mean_delta_nll']:.4f} (95% interval {paired['nll_95_interval'][0]:.4f} to {paired['nll_95_interval'][1]:.4f}); V1.1 wins {paired['fraction_candidate_better_nll']:.1%} of resamples."
        ),
        "",
        "The JSON artifact contains tier calibration/sharpness, per-season scores, full transition and history diagnostics, quadrature approximation notes, and the final frozen model metadata.",
        "",
        "## Historical transition behavior",
        "",
        "| Prior-rank tier | N | Mean next-rank percentile | Next-rank percentile SD | Prior constituent disagreement |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["transition_diagnostics"]["conditional_on_lag1_tier"]:
        lines.append(
            f"| {row['prior_rank_tier']} | {row['n_team_seasons']} | {row['mean_next_rank_percentile']:.3f} | {row['next_rank_percentile_std']:.3f} | {row['mean_prior_constituent_disagreement']:.3f} |"
        )
    lines += [
        "",
        "## Scope and safety",
        "",
        "All candidate inputs are functions solely of final-rank distributions from seasons before the target. No offseason, current-season, external ranking, recruiting, betting, or game data is used. Multi-lag uncertainty uses order-statistic product quadrature (two points per lag, eight joint points); exact Cartesian support is used automatically when a tiny input has at most two observations per lag.",
    ]
    (OUT / "preseason_prior_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    start = time.perf_counter()
    # The V1 script is run separately first and supplies the archived reference
    # PMFs, including its six FBS transition priors.  Keeping this pass separate
    # makes the V1.1 experiment runner practical on modest local hardware.
    rows, cold, _ = v1.load_rows()
    fbs = [row for row in rows if row.subdivision == "fbs"]
    train = [row for row in fbs if row.season in DEV_TRAIN]
    validation = [row for row in fbs if row.season in DEV_VALIDATION]
    development: dict[str, object] = {}
    dev_predictions: dict[str, list[v1.PriorPrediction]] = {}
    for candidate in CANDIDATES:
        print(f"fitting development candidate: {candidate.name}", flush=True)
        candidate_train, candidate_validation = (
            eligible(train, candidate),
            eligible(validation, candidate),
        )
        model = fit(candidate_train, candidate)
        scored = make_predictions(model, candidate_validation, candidate.name)
        development[candidate.name] = {
            "metrics": v1.score_predictions(scored),
            "fit_metadata": model.metadata(),
            "natural_population": len(scored),
            "notes": "strict three-lag eligible population"
            if candidate.strict_three_lag_history
            else "same complete lag-1 population; missing deeper summaries are training-only imputed with indicators",
        }
        dev_predictions[candidate.name] = scored
    reference_dev = dev_predictions["V1_A2_reference"]
    for candidate in CANDIDATES[1:]:
        compared = dev_predictions[candidate.name]
        if candidate.strict_three_lag_history:
            reference = [p for p in reference_dev if p.key in {q.key for q in compared}]
            development[candidate.name]["same_population_reference"] = (
                v1.score_predictions(reference)
            )
        else:
            reference = reference_dev
        development[candidate.name]["paired_bootstrap_vs_v1"] = paired_bootstrap(
            reference, compared
        )
    # Complexity is admitted only for a visible, bootstrap-supported development
    # gain.  This selection state is entirely pre-2022 and recorded verbatim.
    qualifying = []
    for candidate in CANDIDATES[1:]:
        paired = development[candidate.name]["paired_bootstrap_vs_v1"]
        if (
            paired["mean_delta_nll"] <= -0.01
            and paired["fraction_candidate_better_nll"] >= 0.8
        ):
            qualifying.append(candidate)
    selected = qualifying[0] if qualifying else CANDIDATES[0]
    selected_rows = eligible(fbs, selected)
    selected_model = fit([row for row in selected_rows if row.season < 2022], selected)
    candidate_standard = make_predictions(
        selected_model,
        [row for row in selected_rows if row.season in TEST_SEASONS],
        "V1_1_" + selected.name,
    )
    v1_predictions = join_targets(read_v1_predictions(), rows, cold)
    v1_standard = [
        p for p in v1_predictions if p.prior_method == "same_subdivision_lag1"
    ]
    # If a strict model was selected, compare its natural population honestly;
    # otherwise the candidate retains all ordinary FBS targets.  Cold starts
    # remain the established V1 PMFs in either case.
    if selected.strict_three_lag_history:
        selected_keys = {p.key for p in candidate_standard}
        comparison_v1 = [p for p in v1_standard if p.key in selected_keys]
        production_candidate = candidate_standard
    else:
        cold_predictions = [
            p for p in v1_predictions if p.prior_method != "same_subdivision_lag1"
        ]
        production_candidate = candidate_standard + [
            v1.PriorPrediction(
                p.season,
                p.subdivision,
                p.team_id,
                p.team_name,
                p.population,
                p.target_ranks,
                "V1_1_" + selected.name,
                p.prior_method,
                p.pmf,
                p.conditional_location_mean,
                p.predictive_scale,
            )
            for p in cold_predictions
        ]
        comparison_v1 = v1_predictions
    aggregate = {
        "V1_A2": v1.score_predictions(comparison_v1),
        "V1_1": v1.score_predictions(production_candidate),
    }
    per_season = {
        str(season): {
            "V1_A2": v1.score_predictions(
                [p for p in comparison_v1 if p.season == season]
            ),
            "V1_1": v1.score_predictions(
                [p for p in production_candidate if p.season == season]
            ),
        }
        for season in sorted(TEST_SEASONS)
    }
    report: dict[str, object] = {
        "version": "Preseason Prior V1.1",
        "selection": {
            "development_train": [2004, 2017],
            "development_validation": [2018, 2021],
            "untouched_final_test": sorted(TEST_SEASONS),
            "rule": "first ordered candidate with development ΔNLL <= -0.01 and season-bootstrap win rate >= 0.80; otherwise preserve V1",
            "selected": selected.name,
        },
        "input_safety": "Only target-season-prior final rank distributions; all features are constructed in load_rows from seasons < target.",
        "development": development,
        "full_lag_quadrature": {
            "method": "permutation-invariant order-statistic product quadrature",
            "points_per_lag": 2,
            "joint_points_for_three_lags": 8,
            "exact_on_tiny_inputs": "all empirical observations are retained when each lag has at most two observations",
        },
        "transition_diagnostics": transition_diagnostics(
            [row for row in fbs if row.season < 2022]
        ),
        "final_test": {
            "aggregate": aggregate,
            "per_season": per_season,
            "paired_bootstrap": paired_bootstrap(comparison_v1, production_candidate),
            "tier_diagnostics": {
                "V1_A2": tier_diagnostics(comparison_v1),
                "V1_1": tier_diagnostics(production_candidate),
            },
            "cold_start": "V1 FCS-to-FBS and generic cold-start PMFs retained unchanged",
            "selected_fit_metadata": selected_model.metadata(),
        },
        "runtime_seconds": time.perf_counter() - start,
    }
    fields = list(production_candidate[0].csv_row())
    with (OUT / "rank_prior_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(p.csv_row() for p in [*v1_predictions, *production_candidate])
    (OUT / "preseason_model_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    render_report(report)
    print(
        json.dumps(
            {"selected": selected.name, "runtime_seconds": report["runtime_seconds"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
