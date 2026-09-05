"""Fit and evaluate leakage-aware preseason priors from cached inputs only."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from gippyrank.preseason import (
    DirectRankModel,
    GenericRankPrior,
    TeamSeason,
    crps_discrete,
    historical_rank_features,
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


@dataclass(frozen=True)
class ColdStartSeason:
    """Historical target missing a same-subdivision lag-1 distribution."""

    season: int
    subdivision: str
    team_id: str
    team_name: str
    population: int
    target_z: np.ndarray
    target_ranks: np.ndarray
    cross_subdivision_lag_z: np.ndarray | None
    reason: str


@dataclass(frozen=True)
class PriorPrediction:
    """One scored preseason PMF and its empirical final-rank target."""

    season: int
    subdivision: str
    team_id: str
    team_name: str
    population: int
    target_ranks: np.ndarray
    model: str
    prior_method: str
    pmf: np.ndarray
    conditional_location_mean: float | None = None
    predictive_scale: float | None = None

    @property
    def key(self) -> tuple[int, str, str]:
        return (self.season, self.subdivision, self.team_id)

    def csv_row(self) -> dict[str, object]:
        return {
            "season": self.season,
            "subdivision": self.subdivision,
            "team_id": self.team_id,
            "team_name": self.team_name,
            "model": self.model,
            "pmf": json.dumps(
                [round(float(value), 12) for value in self.pmf], separators=(",", ":")
            ),
            "conditional_location_mean": self.conditional_location_mean,
            "predictive_scale": self.predictive_scale,
            "prior_method": self.prior_method,
            **pmf_summaries(self.pmf),
        }


def valid_ranks(row: dict[str, str]) -> np.ndarray:
    population = int(row["team_population"])
    ranks = rank_sample(row).astype(int)
    return ranks[(ranks >= 1) & (ranks <= population)]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def maybe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def coach_change_value(
    current_coach: str | None, prior_coach: str | None
) -> float | None:
    """Unknown coaching history remains unknown, not a no-change observation."""
    if not current_coach or not prior_coach:
        return None
    return float(current_coach != prior_coach)


def load_rows(
    *, max_season: int | None = None
) -> tuple[
    list[TeamSeason], list[ColdStartSeason], list[dict[str, object]]
]:
    """Join outcomes through an explicit ceiling, retaining cold-start records.

    ``max_season`` is a caller-owned historical cutoff.  Leaving it unset is
    useful for annual refits after new completed seasons are added; the legacy
    2022--2025 backtest passes its own fixed evaluation ceiling.
    """
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
    result: list[TeamSeason] = []
    cold_starts: list[ColdStartSeason] = []
    coverage: list[dict[str, object]] = []
    for (season, subdivision, team_id), target in outcomes.items():
        prior = outcomes.get((season - 1, subdivision, team_id))
        current = feature_index.get((season, subdivision, team_id))
        if max_season is not None and season > max_season:
            continue
        target_population = int(target["team_population"])
        target_ranks = valid_ranks(target)
        # A small number of constituent rows extend beyond the composite
        # subdivision roster. They cannot be outcomes in a PMF with support
        # 1..N, so exclude those malformed measurements rather than clip them.
        if not len(target_ranks):
            continue
        team_name = current["team_name"] if current else target["team_name"]
        if prior is None or not len(valid_ranks(prior)):
            cross = (
                outcomes.get((season - 1, "fcs", team_id))
                if subdivision == "fbs"
                else None
            )
            cross_ranks = valid_ranks(cross) if cross else np.asarray([], dtype=int)
            reason = (
                "fcs_to_fbs_transition"
                if len(cross_ranks)
                else "no_prior_rank_distribution"
            )
            cold_starts.append(
                ColdStartSeason(
                    season,
                    subdivision,
                    team_id,
                    team_name,
                    target_population,
                    rank_to_z(target_ranks, target_population),
                    target_ranks,
                    rank_to_z(cross_ranks, int(cross["team_population"]))
                    if len(cross_ranks)
                    else None,
                    reason,
                )
            )
            coverage.append(
                {
                    "season": season,
                    "subdivision": subdivision,
                    "team_id": team_id,
                    "team_name": team_name,
                    "generated": False,
                    "reason": reason,
                }
            )
            continue
        prior_population = int(prior["team_population"])
        prior_ranks = valid_ranks(prior)
        lag1_z = rank_to_z(prior_ranks, prior_population)
        values: dict[str, float | None] = {}
        lag_distributions: list[np.ndarray] = []
        for lag in range(2, 6):
            item = outcomes.get((season - lag, subdivision, team_id))
            if item:
                lag_population = int(item["team_population"])
                lag_ranks = valid_ranks(item)
                historical_lag_z = rank_to_z(lag_ranks, lag_population)
                values[f"lag{lag}_z_mean"] = (
                    float(np.mean(historical_lag_z)) if len(historical_lag_z) else None
                )
                if lag <= 3:
                    lag_distributions.append(historical_lag_z)
            else:
                values[f"lag{lag}_z_mean"] = None
                if lag <= 3:
                    lag_distributions.append(np.asarray([], dtype=float))
        history = []
        for prior_season in range(2002, season):
            item = outcomes.get((prior_season, subdivision, team_id))
            if item is None:
                continue
            history_ranks = valid_ranks(item)
            if len(history_ranks):
                history.append(rank_to_z(history_ranks, int(item["team_population"])))
        values.update(historical_rank_features(lag1_z, tuple(history)))
        for name in (
            "recruiting_class_rank",
            "recruiting_class_points",
            "talent_composite",
            "returning_pct_ppa",
            "returning_pct_passing_ppa",
        ):
            values[name] = maybe_float(current.get(name)) if current else None
        previous_feature = feature_index.get((season - 1, subdivision, team_id))
        coach, prior_coach = (
            current.get("head_coach") if current else None,
            previous_feature.get("head_coach") if previous_feature else None,
        )
        values["coach_change"] = coach_change_value(coach, prior_coach)
        result.append(
            TeamSeason(
                season,
                subdivision,
                team_id,
                team_name,
                target_population,
                lag1_z,
                rank_to_z(target_ranks, target_population),
                target_ranks,
                values,
                tuple(lag_distributions),
            )
        )
        coverage.append(
            {
                "season": season,
                "subdivision": subdivision,
                "team_id": team_id,
                "team_name": team_name,
                "generated": True,
                "reason": "same_subdivision_lag1",
            }
        )
    return result, cold_starts, coverage


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
    rows: list[TeamSeason], features: list[str], include_development: bool = True
) -> tuple[DirectRankModel, dict[str, object]]:
    """Use a conservative pre-specified penalty; report only pre-2022 validation."""
    train = [r for r in rows if r.season in DEVELOPMENT_TRAIN]
    validation = [r for r in rows if r.season in DEVELOPMENT_VALIDATION]
    chosen = 0.25
    final_train = [r for r in rows if r.season < 2022]
    selection: dict[str, object] = {
        "development_train_seasons": [min(DEVELOPMENT_TRAIN), max(DEVELOPMENT_TRAIN)],
        "development_validation_seasons": [
            min(DEVELOPMENT_VALIDATION),
            max(DEVELOPMENT_VALIDATION),
        ],
        "pre_specified_penalty": chosen,
        "temporally_held_out_backtest_seasons": sorted(TEST_SEASONS),
    }
    if include_development:
        development_model = DirectRankModel.fit(train, features, penalty=chosen)
        selection["development_validation_metrics"] = evaluate(
            development_model, validation
        )
    else:
        selection["development_validation_metrics"] = {
            "status": "not used for FBS model selection"
        }
    return DirectRankModel.fit(final_train, features, penalty=chosen), selection


def reliability_bins(
    predicted: list[float],
    observed: list[float],
    edges: tuple[float, ...] = (0.0, 0.05, 0.15, 0.3, 0.5, 0.7, 1.0),
) -> dict[str, object]:
    """Coarse conditional-calibration table for fractional team outcomes."""
    probabilities = np.asarray(predicted, dtype=float)
    outcomes = np.asarray(observed, dtype=float)
    bins = []
    for index, (low, high) in enumerate(pairwise(edges)):
        mask = (probabilities >= low) & (
            probabilities < high if index < len(edges) - 2 else probabilities <= high
        )
        if mask.any():
            bins.append(
                {
                    "range": f"[{low:.2f}, {high:.2f}{']' if index == len(edges) - 2 else ')'}",
                    "n_team_seasons": int(mask.sum()),
                    "mean_predicted_probability": float(probabilities[mask].mean()),
                    "empirical_frequency": float(outcomes[mask].mean()),
                }
            )
    return {
        "brier_score": float(np.mean((probabilities - outcomes) ** 2)),
        "bins": bins,
    }


def score_predictions(
    predictions: list[PriorPrediction],
) -> dict[str, float | int | None]:
    """Score any production or exploratory PMF with equal team-season weight."""
    if not predictions:
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
    for prediction in predictions:
        pmf, target_ranks = prediction.pmf, prediction.target_ranks
        summary = pmf_summaries(pmf)
        nll.append(team_log_score(pmf, target_ranks))
        crps.append(
            float(np.mean([crps_discrete(pmf, int(rank)) for rank in target_ranks]))
        )
        actual_mean, actual_median = (
            float(np.mean(target_ranks)),
            float(np.median(target_ranks)),
        )
        expected_error.append(abs(summary["expected_rank"] - actual_mean))
        median_error.append(abs(summary["median_rank"] - actual_median))
        covered.append(
            float(
                np.mean(
                    (target_ranks >= summary["interval_80_low"])
                    & (target_ranks <= summary["interval_80_high"])
                )
            )
        )
        widths.append(summary["interval_80_high"] - summary["interval_80_low"] + 1)
        for cutoff in top:
            top[cutoff][0].append(summary[f"top{cutoff}_probability"])
            top[cutoff][1].append(float(np.mean(target_ranks <= cutoff)))
    result: dict[str, float | int | None] = {
        "n_team_seasons": len(predictions),
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
            "reliability": reliability_bins(predicted, observed),
        }
    return result


def predictions(
    model: DirectRankModel, rows: list[TeamSeason], spec: str
) -> list[PriorPrediction]:
    output = []
    for row in rows:
        pmf = model.pmf(row.features, row.lag1_z, row.population)
        locations, scale = model.conditional_parameters(row.features, row.lag1_z)
        output.append(
            PriorPrediction(
                row.season,
                row.subdivision,
                row.team_id,
                row.team_name,
                row.population,
                row.target_ranks,
                spec,
                "same_subdivision_lag1",
                pmf,
                float(np.mean(locations)),
                scale,
            )
        )
    return output


def evaluate(
    model: DirectRankModel, rows: list[TeamSeason]
) -> dict[str, float | int | None]:
    return score_predictions(predictions(model, rows, "evaluation"))


def cold_start_teams(cold_starts: list[ColdStartSeason]) -> list[TeamSeason]:
    """Adapt FCS-to-FBS observations to the direct rank-distribution model."""
    return [
        TeamSeason(
            row.season,
            row.subdivision,
            row.team_id,
            row.team_name,
            row.population,
            row.cross_subdivision_lag_z,
            row.target_z,
            row.target_ranks,
            {},
        )
        for row in cold_starts
        if row.subdivision == "fbs" and row.cross_subdivision_lag_z is not None
    ]


def coverage_audit(coverage: list[dict[str, object]]) -> dict[str, object]:
    """Make every target team-season's prior availability explicit."""
    per_season = []
    for season in sorted({int(row["season"]) for row in coverage}):
        for subdivision in ("fbs", "fcs"):
            group = [
                row
                for row in coverage
                if row["season"] == season and row["subdivision"] == subdivision
            ]
            if not group:
                continue
            omitted = [row for row in group if not row["generated"]]
            per_season.append(
                {
                    "season": season,
                    "subdivision": subdivision,
                    "total_teams": len(group),
                    "teams_with_generated_prior": len(group) - len(omitted),
                    "teams_omitted": len(omitted),
                    "omitted": [
                        {
                            "team_id": row["team_id"],
                            "team_name": row["team_name"],
                            "reason": row["reason"],
                        }
                        for row in omitted
                    ],
                }
            )
    return {
        "per_season": per_season,
        "method": {
            "fbs_fcs_transition": "learned direct FCS-to-FBS rank-distribution model when cross-subdivision lag exists",
            "fbs_generic": "analytical broad FBS prior fit to historical no-prior FBS target distributions",
            "fcs": "no-prior FCS teams remain explicitly omitted; FCS cold start is outside V1 production scope",
        },
    }


def apply_fbs_cold_start_coverage(coverage: list[dict[str, object]]) -> None:
    """Mark every FBS target as covered by its learned or generic fallback."""
    for item in coverage:
        if item["subdivision"] == "fbs" and not item["generated"]:
            item["generated"] = True
            item["reason"] = (
                "learned_fcs_to_fbs_transition"
                if item["reason"] == "fcs_to_fbs_transition"
                else "generic_fbs_cold_start"
            )


def team_season_keys(rows: list[TeamSeason]) -> set[tuple[int, str, str]]:
    """Canonical keys used for a fair same-population model comparison."""
    return {(row.season, row.subdivision, row.team_id) for row in rows}


def validate_production_coverage(
    predictions: list[PriorPrediction], expected_keys: set[tuple[int, str, str]]
) -> None:
    """Require exactly one production PMF for every target team-season."""
    keys = [prediction.key for prediction in predictions]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate production FBS prediction key")
    if set(keys) != expected_keys:
        missing = sorted(expected_keys - set(keys))
        extra = sorted(set(keys) - expected_keys)
        raise ValueError(
            f"production FBS prediction coverage mismatch: missing={missing}, extra={extra}"
        )


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
    production_fbs = report["production_evaluation"]["combined_all_fbs"]
    reliability_lines = [
        f"- Top {cutoff}: Brier={production_fbs[f'top{cutoff}']['reliability']['brier_score']:.4f}; full coarse reliability bins are in `preseason_model_report.json`."
        for cutoff in (5, 10, 25)
    ]
    same_population = report["same_population_exploratory_comparison"]
    cold = report["coverage_and_cold_start"]["cold_start_fit"]
    production = report["production_evaluation"]
    production_breakdowns = [
        f"- {label}: N={metrics['n_team_seasons']}, NLL={metrics['nll']:.3f}, CRPS={metrics['crps']:.4f}, expected-rank MAE={metrics['expected_rank_mae']:.1f}, 80% coverage={metrics['interval_80_coverage']:.3f}."
        for label, metrics in (
            ("Standard A2", production["standard_a2"]),
            ("FCS-to-FBS transition", production["fcs_to_fbs_transition"]),
            ("Generic FBS cold start", production["generic_fbs_cold_start"]),
            ("Combined production", production["combined_all_fbs"]),
        )
        if metrics["n_team_seasons"]
    ]
    metric_lines = []
    for name, item in models.items():
        fbs = item["test"].get("fbs", {})
        if "nll" in fbs:
            metric_lines.append(
                f"- {name}: standard-lag FBS N={fbs['n_team_seasons']}, NLL={fbs['nll']:.3f}, CRPS={fbs['crps']:.4f}, expected-rank MAE={fbs['expected_rank_mae']:.1f}, 80% coverage={fbs['interval_80_coverage']:.3f}."
            )
    lines = [
        "# GippyRank4 Preseason Prior V1",
        "",
        "## Recommendation",
        "",
        "The leakage-safe production candidate is Model A2 (t-1+t-2+t-3): a direct heteroscedastic Normal distribution over the logit within-subdivision final-rank percentile. Its t-1 input is the full empirical constituent-rank distribution; additional lags improve temporally held-out FBS backtest scoring and remain rank-history-only. It does not infer a scalar team-strength state. Model B has no additional qualified feature in the current cached sources, so it is intentionally identical to the selected rank-history candidate rather than promoting retrospective fields.",
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
        "All development choices use 2004–2017 training and 2018–2021 season holdouts. 2022–2025 is the temporally held-out evaluation period used after selection and final fitting, not a claim of a globally uninspected test set. FBS and FCS rank universes are always fit and scored separately. `preseason_model_report.json` contains team-season NLL, CRPS, rank MAE, interval coverage and width, Top-5/10/25 calibration, ablations, preprocessing, and diagnostics; `rank_prior_predictions.csv` contains the full integrated PMFs.",
        "",
        *metric_lines,
        "",
        f"The production A2 headline combines all {production['total_fbs_test_team_seasons']} FBS target team-seasons: NLL={production_fbs['nll']:.3f}, CRPS={production_fbs['crps']:.4f}, expected-rank MAE={production_fbs['expected_rank_mae']:.1f}, and 80% coverage={production_fbs['interval_80_coverage']:.3f}. The standard-lag A2 row above is a diagnostic subset, not the production headline.",
        "",
        "Model A2 is the explicit lag ablation (t-1+t-2 and t-1+t-2+t-3); B has no currently qualified safe covariate beyond rank history. Model C is evaluated only on its high-coverage FBS modern subset and remains exploratory, so its score is not a production-selection comparison. The machine-readable report records all FBS/FCS breakdowns, feature coverage, Top-5/10/25 calibration gaps, and scale diagnostics.",
        "",
        "## Conditional Top-N calibration",
        "",
        "Top-N diagnostics now include coarse probability reliability bins with fractional empirical constituent outcome frequencies, in addition to aggregate gaps. Brier scores for the production FBS A2 model are:",
        "",
        *reliability_lines,
        "",
        "## Cold starts and fair exploratory comparison",
        "",
        f"Every FBS target team-season now receives a prior. The learned FCS-to-FBS transition fit uses {cold['fbs_training_fcs_to_fbs_transitions']} pre-2022 transitions. The temporally held-out backtest population has {cold['fbs_test_cold_starts']} cold starts: {cold['fbs_test_fcs_to_fbs_transitions']} transition and {cold['fbs_test_generic_cold_starts']} generic. Programs without any prior distribution use a broad analytical FBS no-prior fallback. FCS cold starts remain explicitly reported as omitted ({sum(item['teams_omitted'] for item in report['coverage_and_cold_start']['per_season'] if item['subdivision'] == 'fcs')} historical team-seasons).",
        "",
        *production_breakdowns,
        "",
        f"On the exact {same_population['population']['n_team_seasons']}-team-season modern FBS subset, A2 t-1+t-2+t-3 has NLL={same_population['metrics']['A2_t1_t2_t3']['nll']:.3f}; exploratory Model C has NLL={same_population['metrics']['C_exploratory_modern']['nll']:.3f}. This is incremental signal only: C remains timing-uncertain and is not promoted.",
        "",
        "## Discrete probabilities and 2026",
        "",
        "PMFs integrate continuous Normal mass across transformed rank bins. The first and last bins have infinite exterior boundaries, preserving uncertainty at ranks 1 and N. No 2026 prior is created: the season has begun and the repository has no independently archived 2026 preseason snapshot. A valid reconstruction requires dated, pre-kickoff snapshots for every promoted feature and a roster of the eligible subdivision population.",
        "",
    ]
    (OUT / "preseason_prior_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, cold_starts, coverage = load_rows(max_season=max(TEST_SEASONS))
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
            "temporally_held_out_backtest": sorted(TEST_SEASONS),
        },
        "models": {},
        "safety": {
            "2026": "Not generated; current-season endpoints are not preserved preseason snapshots and no 2026 outcomes were read."
        },
    }
    all_predictions: list[PriorPrediction] = []
    production_standard: list[PriorPrediction] = []
    model_refs: dict[str, dict[str, DirectRankModel]] = {}
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
        model_refs[name] = {}
        for subdivision in ("fbs", "fcs"):
            if name == "C_exploratory_modern" and subdivision == "fcs":
                item["test"][subdivision] = {
                    "status": "not fitted; FCS uses the simpler rank-history prior"
                }
                continue
            if subdivision == "fcs" and name not in {
                "A2_t1_t2_t3",
                "B_safe_long_history",
            }:
                item["test"][subdivision] = {
                    "status": "not fitted; FCS production uses the selected A2 rank-history prior"
                }
                continue
            local = [r for r in spec_rows if r.subdivision == subdivision]
            key = (subdivision, tuple(feature_names))
            if key not in fitted:
                fitted[key] = fit_selected(
                    local,
                    feature_names,
                    include_development=(
                        subdivision == "fbs" and name != "C_exploratory_modern"
                    ),
                )
            model, selection = fitted[key]
            model_refs[name][subdivision] = model
            test = [r for r in local if r.season in TEST_SEASONS]
            item["selection"][subdivision] = selection
            item["test"][subdivision] = evaluate(model, test)
            item["fit_metadata"][subdivision] = model.metadata()
            item["uncertainty_diagnostics"][subdivision] = uncertainty_diagnostics(
                model, test
            )
            generated = predictions(model, test, name)
            all_predictions.extend(generated)
            if name == "A2_t1_t2_t3" and subdivision == "fbs":
                production_standard = generated
        report["models"][name] = item
    modern_test = [
        row
        for row in eligible_for_spec(rows, "C_exploratory_modern")
        if row.subdivision == "fbs" and row.season in TEST_SEASONS
    ]
    report["same_population_exploratory_comparison"] = {
        "population": {
            "n_team_seasons": len(modern_test),
            "keys": [
                f"{season}:{subdivision}:{team_id}"
                for season, subdivision, team_id in sorted(
                    team_season_keys(modern_test)
                )
            ],
        },
        "metrics": {
            name: evaluate(model_refs[name]["fbs"], modern_test)
            for name in ("A_t1", "A2_t1_t2", "A2_t1_t2_t3", "C_exploratory_modern")
        },
    }
    promotion_rows = cold_start_teams(cold_starts)
    promotion_train = [row for row in promotion_rows if row.season < 2022]
    transition_model = DirectRankModel.fit(promotion_train, [], penalty=0.25)
    generic_rows = [
        TeamSeason(
            row.season,
            row.subdivision,
            row.team_id,
            row.team_name,
            row.population,
            np.asarray([0.0]),
            row.target_z,
            row.target_ranks,
            {},
        )
        for row in cold_starts
        if row.subdivision == "fbs"
        and row.cross_subdivision_lag_z is None
        and row.season < 2022
    ]
    generic_model = GenericRankPrior.fit(generic_rows)
    cold_test = [
        row
        for row in cold_starts
        if row.subdivision == "fbs" and row.season in TEST_SEASONS
    ]
    production_cold: list[PriorPrediction] = []
    for row in cold_test:
        if row.cross_subdivision_lag_z is not None:
            pmf = transition_model.pmf({}, row.cross_subdivision_lag_z, row.population)
            method = "learned_fcs_to_fbs_transition"
        else:
            pmf = generic_model.pmf(row.population)
            method = "generic_fbs_cold_start"
        production_cold.append(
            PriorPrediction(
                row.season,
                row.subdivision,
                row.team_id,
                row.team_name,
                row.population,
                row.target_ranks,
                "A2_t1_t2_t3",
                method,
                pmf,
            )
        )
    all_predictions.extend(production_cold)
    production_fbs = production_standard + production_cold
    expected_fbs_keys = team_season_keys(
        [row for row in rows if row.subdivision == "fbs" and row.season in TEST_SEASONS]
    ) | {
        (row.season, row.subdivision, row.team_id)
        for row in cold_starts
        if row.subdivision == "fbs" and row.season in TEST_SEASONS
    }
    validate_production_coverage(production_fbs, expected_fbs_keys)
    report["production_evaluation"] = {
        "total_fbs_test_team_seasons": len(expected_fbs_keys),
        "standard_a2": score_predictions(production_standard),
        "fcs_to_fbs_transition": score_predictions(
            [
                prediction
                for prediction in production_cold
                if prediction.prior_method == "learned_fcs_to_fbs_transition"
            ]
        ),
        "generic_fbs_cold_start": score_predictions(
            [
                prediction
                for prediction in production_cold
                if prediction.prior_method == "generic_fbs_cold_start"
            ]
        ),
        "combined_all_fbs": score_predictions(production_fbs),
    }
    apply_fbs_cold_start_coverage(coverage)
    coverage_result = coverage_audit(coverage)
    coverage_result["cold_start_fit"] = {
        "fbs_training_fcs_to_fbs_transitions": len(promotion_train),
        "fbs_training_generic_cold_starts": len(generic_rows),
        "fbs_test_cold_starts": len(production_cold),
        "fbs_test_fcs_to_fbs_transitions": sum(
            prediction.prior_method == "learned_fcs_to_fbs_transition"
            for prediction in production_cold
        ),
        "fbs_test_generic_cold_starts": sum(
            prediction.prior_method == "generic_fbs_cold_start"
            for prediction in production_cold
        ),
        "transition_model": transition_model.metadata(),
        "generic_model": generic_model.metadata(),
    }
    fields = list(all_predictions[0].csv_row())
    with (OUT / "rank_prior_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(prediction.csv_row() for prediction in all_predictions)
    report["production_candidate"] = "A2_t1_t2_t3"
    report["coverage_and_cold_start"] = coverage_result
    report["best_statistical_model"] = (
        "See model-specific temporally held-out FBS backtest NLL; exploratory C is not eligible for production."
    )
    (OUT / "preseason_model_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUT / "preseason_prior_coverage.json").write_text(
        json.dumps(coverage_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
