"""Test returning production as a predictive-scale signal.

This is research-only infrastructure for issue 90.  It keeps the production
Context C 1.2 feature family and evaluation protocol fixed while changing only
the equation in which returning-production features are allowed to enter.  It
reads cached historical inputs from ``--source-root`` and writes a reproducible
study under ``--output-root``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import build_preseason_context_prior_v1_2 as c12
import build_preseason_prior as v1
import build_preseason_prior_v1_1 as h11
import matplotlib.pyplot as plt
import numpy as np

from gippyrank.preseason import DirectRankModel, TeamSeason, pmf_summaries

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/returning_production_uncertainty"
PLOTS = OUT / "plots"
FROZEN_TRAIN_THROUGH = 2021
TARGET_SEASONS = (2022, 2023, 2024, 2025)
# The stored production artifact and this deterministic refit can differ by a
# few micro-units from L-BFGS termination; this is intentionally much tighter
# than any reported metric precision.
C0_TOLERANCE = 1e-5

H_FEATURES = tuple(c12.H_FEATURES)
NON_RP_FEATURES = (
    *c12.COACH_FEATURES,
    *c12.RECRUITING_FEATURES,
    *c12.TALENT_FEATURES,
)
RP_FEATURES = tuple(c12.RETURNING_FEATURES)
ALL_CONTEXT_FEATURES = (*NON_RP_FEATURES, *RP_FEATURES)
METRICS = (
    "nll",
    "crps",
    "expected_rank_mae",
    "median_rank_mae",
    "interval_80_coverage",
    "interval_80_average_width",
)


@dataclass(frozen=True)
class Variant:
    """A predeclared location/scale feature-equation variant."""

    name: str
    location_context: tuple[str, ...]
    scale_context: tuple[str, ...]

    @property
    def feature_names(self) -> list[str]:
        return [
            *H_FEATURES,
            *[
                feature
                for feature in ALL_CONTEXT_FEATURES
                if feature in (*self.location_context, *self.scale_context)
            ],
        ]

    @property
    def location_features(self) -> list[str]:
        return [*H_FEATURES, *self.location_context]

    @property
    def scale_features(self) -> list[str]:
        return [*H_FEATURES, *self.scale_context]


VARIANTS = (
    Variant("C0_current_location", (*NON_RP_FEATURES, *RP_FEATURES), ()),
    Variant("C1_no_rp", NON_RP_FEATURES, ()),
    Variant("C2_rp_scale_only", NON_RP_FEATURES, RP_FEATURES),
    Variant("C3_rp_both", (*NON_RP_FEATURES, *RP_FEATURES), RP_FEATURES),
)


def configure_source_root(source_root: Path) -> None:
    """Point all legacy cached-input loaders at the explicit source checkout."""

    v1.ROOT = source_root
    v1.OUT = source_root / "data/processed/preseason"
    h11.ROOT = source_root
    h11.OUT = source_root / "data/processed/preseason"
    c12.ROOT = source_root
    c12.PRESEASON = source_root / "data/processed/preseason"
    c12.MODELING = source_root / "data/processed/modeling"
    c12.HISTORY = c12.PRESEASON / "history"
    c12.CONTEXT = c12.PRESEASON / "context"
    c12.TENURES = source_root / "data/raw/cfbd/preseason/coach_tenures"


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty artifact: {name}")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(name: str, value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_variant(rows: list[TeamSeason], variant: Variant) -> DirectRankModel:
    """Fit with the production C 1.2 optimizer, penalty, and retry behavior."""

    c12.only_approved(
        {name: None for name in (*variant.location_context, *variant.scale_context)},
        set(c12.APPROVED_CONTEXT_FEATURES),
    )
    training = [row for row in rows if row.season <= FROZEN_TRAIN_THROUGH]
    if not training:
        raise ValueError("no training rows through the frozen cutoff")
    kwargs = {
        "penalty": 0.25,
        "location_feature_names": variant.location_features,
        "scale_feature_names": variant.scale_features,
    }
    try:
        return DirectRankModel.fit(training, variant.feature_names, **kwargs)
    except RuntimeError as error:
        if "ITERATIONS REACHED LIMIT" not in str(error):
            raise
        return DirectRankModel.fit(
            training,
            variant.feature_names,
            optimizer_options={"maxiter": 2000},
            **kwargs,
        )


def make_predictions(
    model: DirectRankModel, rows: list[TeamSeason], name: str
) -> list[v1.PriorPrediction]:
    return h11.make_predictions(model, rows, name)


def load_frozen_inputs(
    source_root: Path,
) -> tuple[list[TeamSeason], list[TeamSeason], list[v1.PriorPrediction]]:
    """Load non-cold FBS context rows and the exact production H fallback panel."""

    rows, cold, _coverage = v1.load_rows(max_season=max(TARGET_SEASONS))
    fbs = [row for row in rows if row.subdivision == "fbs"]
    contextual, _context_coverage = c12.attach_context(
        fbs, c12.feature_index(), c12.cached_tenures()
    )
    history = h11.join_targets(h11.read_predictions(c12.H_MODEL_NAME), rows, cold)
    history = [
        prediction
        for prediction in history
        if prediction.subdivision == "fbs" and prediction.season in TARGET_SEASONS
    ]
    if not contextual or not history:
        raise ValueError("frozen Context inputs are empty")
    if not Path(
        source_root / "data/processed/modeling/team_season_rank_distributions.csv"
    ).exists():
        raise FileNotFoundError("frozen rank-distribution input is missing")
    return fbs, contextual, history


def merge_h_fallback(
    candidate: list[v1.PriorPrediction], history: list[v1.PriorPrediction]
) -> list[v1.PriorPrediction]:
    """Use C for rank-history rows and stored H for rank-history cold starts."""

    by_key = {prediction.key: prediction for prediction in candidate}
    result = []
    for fallback in history:
        result.append(by_key.get(fallback.key, fallback))
    if {prediction.key for prediction in result} != {
        prediction.key for prediction in history
    }:
        raise ValueError("C/H fallback changed the frozen target population")
    return result


def top_brier(predictions: list[v1.PriorPrediction], cutoff: int) -> float:
    return float(
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


def score(predictions: list[v1.PriorPrediction]) -> dict[str, float]:
    raw = v1.score_predictions(predictions)
    result = {
        metric: float(raw[metric]) for metric in METRICS if raw.get(metric) is not None
    }
    for cutoff in (5, 10, 25):
        result[f"top{cutoff}_brier"] = top_brier(predictions, cutoff)
    return result


def prediction_losses(
    predictions: list[v1.PriorPrediction],
) -> dict[tuple[int, str, str], tuple[float, float]]:
    return {
        prediction.key: (
            v1.team_log_score(prediction.pmf, prediction.target_ranks),
            float(
                np.mean(
                    [
                        v1.crps_discrete(prediction.pmf, int(rank))
                        for rank in prediction.target_ranks
                    ]
                )
            ),
        )
        for prediction in predictions
    }


def annual_rows(
    predictions_by_variant: dict[str, list[v1.PriorPrediction]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    baseline = predictions_by_variant[VARIANTS[0].name]
    baseline_keys = {prediction.key for prediction in baseline}
    annual: list[dict[str, object]] = []
    per_team: list[dict[str, object]] = []
    for name, predictions in predictions_by_variant.items():
        if {prediction.key for prediction in predictions} != baseline_keys:
            raise ValueError("all variants must score identical frozen keys")
        losses = prediction_losses(predictions)
        names = {prediction.key: prediction.team_name for prediction in predictions}
        for season in TARGET_SEASONS:
            yearly = [
                prediction for prediction in predictions if prediction.season == season
            ]
            metrics = score(yearly)
            annual.append(
                {
                    "variant": name,
                    "target_season": season,
                    "n_team_seasons": len(yearly),
                    "n_training_seasons": len(range(2004, FROZEN_TRAIN_THROUGH + 1)),
                    **metrics,
                }
            )
        for key in sorted(losses):
            nll, crps = losses[key]
            prediction = next(item for item in predictions if item.key == key)
            summary = pmf_summaries(prediction.pmf)
            per_team.append(
                {
                    "variant": name,
                    "season": key[0],
                    "subdivision": key[1],
                    "team_id": key[2],
                    "team_name": names[key],
                    "nll": nll,
                    "crps": crps,
                    "predictive_scale": prediction.predictive_scale,
                    "interval_80_width": summary["interval_80_high"]
                    - summary["interval_80_low"]
                    + 1,
                }
            )
    annual_by_key = {
        (str(row["variant"]), int(row["target_season"])): row for row in annual
    }
    for row in annual:
        baseline_row = annual_by_key[(VARIANTS[0].name, int(row["target_season"]))]
        for metric in (*METRICS, "top5_brier", "top10_brier", "top25_brier"):
            row[f"delta_{metric}_vs_c0"] = float(row[metric]) - float(
                baseline_row[metric]
            )
    return annual, per_team


def aggregate_rows(
    predictions_by_variant: dict[str, list[v1.PriorPrediction]],
) -> list[dict[str, object]]:
    scores = {
        name: score(predictions) for name, predictions in predictions_by_variant.items()
    }
    result = []
    baseline = scores[VARIANTS[0].name]
    for variant in VARIANTS:
        metrics = scores[variant.name]
        result.append(
            {
                "variant": variant.name,
                "n_team_seasons": len(predictions_by_variant[variant.name]),
                **metrics,
                **{
                    f"delta_{metric}_vs_c0": metrics[metric] - baseline[metric]
                    for metric in (*METRICS, "top5_brier", "top10_brier", "top25_brier")
                },
            }
        )
    return result


def c2_vs_c1_comparison(
    predictions_by_variant: dict[str, list[v1.PriorPrediction]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Measure the incremental scale-only effect against the no-RP parent."""

    reference = predictions_by_variant[VARIANTS[1].name]
    candidate = predictions_by_variant[VARIANTS[2].name]
    reference_by_key = {prediction.key: prediction for prediction in reference}
    candidate_by_key = {prediction.key: prediction for prediction in candidate}
    if set(reference_by_key) != set(candidate_by_key):
        raise ValueError("C1/C2 comparison requires identical team-season keys")
    reference_losses = prediction_losses(reference)
    candidate_losses = prediction_losses(candidate)
    paired_rows = []
    for key in sorted(reference_losses):
        reference_nll, reference_crps = reference_losses[key]
        candidate_nll, candidate_crps = candidate_losses[key]
        paired_rows.append(
            {
                "season": key[0],
                "subdivision": key[1],
                "team_id": key[2],
                "team_name": reference_by_key[key].team_name,
                "c1_nll": reference_nll,
                "c2_nll": candidate_nll,
                "delta_nll_c2_minus_c1": candidate_nll - reference_nll,
                "c1_crps": reference_crps,
                "c2_crps": candidate_crps,
                "delta_crps_c2_minus_c1": candidate_crps - reference_crps,
            }
        )
    reference_score = score(reference)
    candidate_score = score(candidate)
    all_metrics = (*METRICS, "top5_brier", "top10_brier", "top25_brier")
    aggregate_deltas = {
        metric: candidate_score[metric] - reference_score[metric]
        for metric in all_metrics
    }
    per_season = []
    seasons = sorted({prediction.season for prediction in reference})
    for season in seasons:
        reference_year = [
            prediction for prediction in reference if prediction.season == season
        ]
        candidate_year = [
            prediction for prediction in candidate if prediction.season == season
        ]
        reference_year_score = score(reference_year)
        candidate_year_score = score(candidate_year)
        per_season.append(
            {
                "target_season": season,
                "n_team_seasons": len(reference_year),
                **{
                    f"delta_{metric}_c2_minus_c1": candidate_year_score[metric]
                    - reference_year_score[metric]
                    for metric in all_metrics
                },
            }
        )
    nll_deltas = np.asarray(
        [float(row["delta_nll_c2_minus_c1"]) for row in paired_rows], dtype=float
    )
    crps_deltas = np.asarray(
        [float(row["delta_crps_c2_minus_c1"]) for row in paired_rows], dtype=float
    )
    return paired_rows, {
        "reference": VARIANTS[1].name,
        "candidate": VARIANTS[2].name,
        "aggregate_metric_deltas": aggregate_deltas,
        "per_season": per_season,
        "paired_team_season": {
            "n_team_seasons": len(paired_rows),
            "mean_delta_nll": float(np.mean(nll_deltas)),
            "median_delta_nll": float(np.median(nll_deltas)),
            "fraction_c2_better_nll": float(np.mean(nll_deltas < 0)),
            "mean_delta_crps": float(np.mean(crps_deltas)),
            "median_delta_crps": float(np.median(crps_deltas)),
            "fraction_c2_better_crps": float(np.mean(crps_deltas < 0)),
        },
        "season_bootstrap": h11.paired_bootstrap(reference, candidate),
    }


def scale_diagnostics(
    models: dict[str, DirectRankModel], training: list[TeamSeason]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Report standardized RP scale coefficients and low/median/high predictions."""

    representative = {
        feature: float(
            np.median(
                [
                    row.features[feature]
                    for row in training
                    if row.features.get(feature) is not None
                ]
            )
        )
        for feature in ALL_CONTEXT_FEATURES
        if any(row.features.get(feature) is not None for row in training)
    }
    lag1 = training[len(training) // 2].lag1_z
    coefficients: list[dict[str, object]] = []
    levels: list[dict[str, object]] = []
    for variant in VARIANTS:
        model = models[variant.name]
        feature_index = {name: index for index, name in enumerate(model.feature_names)}
        n_features = len(model.feature_names)
        for feature in RP_FEATURES:
            included = feature in variant.scale_context
            index = feature_index.get(feature)
            coefficients.append(
                {
                    "variant": variant.name,
                    "feature": feature,
                    "included_in_scale": included,
                    "standardized_scale_coefficient": (
                        float(model.gamma[1 + index])
                        if included and index is not None
                        else None
                    ),
                    "missingness_scale_coefficient": (
                        float(model.gamma[1 + n_features + index])
                        if included and index is not None
                        else None
                    ),
                }
            )
        for label, quantile in (("low", 0.10), ("median", 0.50), ("high", 0.90)):
            features = dict(representative)
            for feature in RP_FEATURES:
                observed = np.asarray(
                    [
                        row.features[feature]
                        for row in training
                        if row.features.get(feature) is not None
                    ],
                    dtype=float,
                )
                features[feature] = float(np.quantile(observed, quantile))
            locations, predictive_scale = model.conditional_parameters(features, lag1)
            levels.append(
                {
                    "variant": variant.name,
                    "returning_production_level": label,
                    "quantile": quantile,
                    **{feature: features[feature] for feature in RP_FEATURES},
                    "conditional_location_mean": float(np.mean(locations)),
                    "predictive_scale": predictive_scale,
                }
            )
    median_scales = {
        row["variant"]: float(row["predictive_scale"])
        for row in levels
        if row["returning_production_level"] == "median"
    }
    for row in levels:
        row["scale_delta_vs_median"] = (
            float(row["predictive_scale"]) - median_scales[row["variant"]]
        )
    return coefficients, levels


def load_production_reference(source_root: Path) -> dict[str, object]:
    path = source_root / "data/processed/preseason/context/evaluation.json"
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value["all_fbs"]["candidate"]


def c0_reproduction(
    c0_score: dict[str, float], production: dict[str, object]
) -> dict[str, object]:
    differences = {
        metric: c0_score[metric] - float(production[metric]) for metric in METRICS
    }
    maximum = max(abs(value) for value in differences.values())
    return {
        "production_metrics": {metric: float(production[metric]) for metric in METRICS},
        "recomputed_c0_metrics": {metric: c0_score[metric] for metric in METRICS},
        "differences_recomputed_minus_stored": differences,
        "tolerance": C0_TOLERANCE,
        "max_absolute_difference": maximum,
        "passed": maximum <= C0_TOLERANCE,
    }


def render_report(
    summary: dict[str, object],
    annual: list[dict[str, object]],
    aggregate: list[dict[str, object]],
    coefficients: list[dict[str, object]],
    levels: list[dict[str, object]],
    c2_vs_c1: dict[str, object],
) -> None:
    by_variant = {str(row["variant"]): row for row in aggregate}
    c1 = by_variant[VARIANTS[1].name]
    c2 = by_variant[VARIANTS[2].name]
    c2_levels = {
        str(row["returning_production_level"]): float(row["predictive_scale"])
        for row in levels
        if row["variant"] == VARIANTS[2].name
    }
    direction = (
        "lower RP increases predicted uncertainty"
        if c2_levels["low"] > c2_levels["high"]
        else "lower RP does not increase predicted uncertainty"
    )
    incremental = c2_vs_c1["aggregate_metric_deltas"]
    incremental_per_season = c2_vs_c1["per_season"]
    incremental_paired = c2_vs_c1["paired_team_season"]
    incremental_bootstrap = c2_vs_c1["season_bootstrap"]
    incremental_wins = [
        int(row["target_season"])
        for row in incremental_per_season
        if float(row["delta_nll_c2_minus_c1"]) < 0
    ]
    incremental_losses = [
        int(row["target_season"])
        for row in incremental_per_season
        if float(row["delta_nll_c2_minus_c1"]) > 0
    ]
    lines = [
        "# Returning production as an uncertainty signal (issue 90)",
        "",
        "## Conclusion",
        "",
        f"Removing RP from location (C1) has ΔNLL {float(c1['delta_nll_vs_c0']):+.4f} versus C0 and ΔCRPS {float(c1['delta_crps_vs_c0']):+.4f}. This is the strong result in the experiment.",
        f"Moving RP into scale (C2) adds only ΔNLL {float(incremental['nll']):+.4f} versus C1, with C2 winning in {', '.join(map(str, incremental_wins))} and losing in {', '.join(map(str, incremental_losses))}. Its paired team-season improvement fraction is {float(incremental_paired['fraction_c2_better_nll']):.3f}, and the season-bootstrap 95% ΔNLL range is {float(incremental_bootstrap['nll_central_95_bootstrap_range'][0]):+.4f} to {float(incremental_bootstrap['nll_central_95_bootstrap_range'][1]):+.4f}.",
        f"The representative C2 scale diagnostic says **{direction}**: scale is {c2_levels['low']:.4f} at low RP, {c2_levels['median']:.4f} at median RP, and {c2_levels['high']:.4f} at high RP, but individual RP scale coefficients have mixed signs. Overall, evidence that RP adds meaningful uncertainty-signal value beyond removing it from location is weak and mixed.",
        "",
        "## Frozen protocol",
        "",
        "- Every variant is fit once using rows through 2021 and scored unchanged on 2022–2025.",
        "- The target panel is the exact stored production FBS panel, including H fallback for rank-history cold starts.",
        "- Preprocessing, penalty (0.25), optimizer, retry behavior, quadrature, target weighting, and scoring are unchanged.",
        "- C0 is the current production Context location formulation: all Context features enter location and scale retains H only.",
        "",
        "## Exact variants",
        "",
        "| Variant | Location equation | Scale equation |",
        "|---|---|---|",
    ]
    for variant in VARIANTS:
        lines.append(
            f"| {variant.name} | H + {' + '.join(variant.location_context) or 'none'} | H + {' + '.join(variant.scale_context) or 'none'} |"
        )
    lines += [
        "",
        "C1 is the no-RP comparison. C2 is the primary hypothesis: non-RP Context remains in location while RP is removed from location and added to scale. C3 is the secondary both-equations comparator.",
        "",
        "## C0 reproduction check",
        "",
        f"The recomputed C0 maximum absolute metric difference from the stored production evaluation is **{float(summary['c0_reproduction']['max_absolute_difference']):.3g}**, against tolerance {C0_TOLERANCE:g}; check status: **{'PASS' if summary['c0_reproduction']['passed'] else 'FAIL'}**.",
        "",
        "## Aggregate held-out results",
        "",
        "Lower is better for NLL, CRPS, rank MAE, and interval width. Coverage is closer to the nominal 0.80 target.",
        "",
        "| Variant | N | NLL | ΔNLL | CRPS | ΔCRPS | Exp. rank MAE | Median rank MAE | 80% coverage | 80% width |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['variant']} | {row['n_team_seasons']} | {float(row['nll']):.4f} | {float(row['delta_nll_vs_c0']):+.4f} | {float(row['crps']):.4f} | {float(row['delta_crps_vs_c0']):+.4f} | {float(row['expected_rank_mae']):.2f} | {float(row['median_rank_mae']):.2f} | {float(row['interval_80_coverage']):.3f} | {float(row['interval_80_average_width']):.1f} |"
        )
    lines += ["", "## Year-by-year results", ""]
    for season in TARGET_SEASONS:
        lines += [
            f"### {season}",
            "",
            "| Variant | NLL | ΔNLL | CRPS | ΔCRPS | Exp. MAE | Median MAE | 80% coverage | 80% width |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in [item for item in annual if int(item["target_season"]) == season]:
            lines.append(
                f"| {row['variant']} | {float(row['nll']):.4f} | {float(row['delta_nll_vs_c0']):+.4f} | {float(row['crps']):.4f} | {float(row['delta_crps_vs_c0']):+.4f} | {float(row['expected_rank_mae']):.2f} | {float(row['median_rank_mae']):.2f} | {float(row['interval_80_coverage']):.3f} | {float(row['interval_80_average_width']):.1f} |"
            )
        lines.append("")
    lines += [
        "## Incremental C1→C2 comparison",
        "",
        "These deltas isolate the value of adding RP to scale after RP has already been removed from location. Negative values favor C2. The paired team-season and season-bootstrap summaries are descriptive, not inferential confidence intervals.",
        "",
        "| Metric | Δ C2 − C1 |",
        "|---|---:|",
        f"| NLL | {float(incremental['nll']):+.4f} |",
        f"| CRPS | {float(incremental['crps']):+.5f} |",
        f"| Expected-rank MAE | {float(incremental['expected_rank_mae']):+.2f} |",
        f"| Median-rank MAE | {float(incremental['median_rank_mae']):+.2f} |",
        f"| 80% coverage | {float(incremental['interval_80_coverage']):+.3f} |",
        f"| 80% interval width | {float(incremental['interval_80_average_width']):+.1f} |",
        "",
        "| Season | ΔNLL | ΔCRPS | Δ80% coverage | Δ80% width |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in incremental_per_season:
        lines.append(
            f"| {row['target_season']} | {float(row['delta_nll_c2_minus_c1']):+.4f} | {float(row['delta_crps_c2_minus_c1']):+.4f} | {float(row['delta_interval_80_coverage_c2_minus_c1']):+.3f} | {float(row['delta_interval_80_average_width_c2_minus_c1']):+.1f} |"
        )
    lines += [
        "",
        f"Across {incremental_paired['n_team_seasons']} paired team-seasons, mean ΔNLL was {float(incremental_paired['mean_delta_nll']):+.4f}, and C2 was better on {float(incremental_paired['fraction_c2_better_nll']):.1%} of team-seasons. The exhaustive ordered season bootstrap over {incremental_bootstrap['n_resamples']} resamples had mean ΔNLL {float(incremental_bootstrap['mean_delta_nll']):+.4f}, central 95% range {float(incremental_bootstrap['nll_central_95_bootstrap_range'][0]):+.4f} to {float(incremental_bootstrap['nll_central_95_bootstrap_range'][1]):+.4f}, and C2 favored in {float(incremental_bootstrap['fraction_candidate_better_nll']):.1%} of resamples.",
        "",
        "## Scale diagnostics",
        "",
        "The scale coefficients are standardized feature coefficients. Positive values mean higher RP raises the modeled scale, conditional on the other features; negative values mean higher RP lowers it.",
        "",
        "| Variant | RP feature | In scale? | Standardized coefficient | Missingness coefficient |",
        "|---|---|---:|---:|---:|",
    ]
    for row in coefficients:
        coefficient = (
            f"{float(row['standardized_scale_coefficient']):+.4f}"
            if row["standardized_scale_coefficient"] is not None
            else "—"
        )
        missing = (
            f"{float(row['missingness_scale_coefficient']):+.4f}"
            if row["missingness_scale_coefficient"] is not None
            else "—"
        )
        lines.append(
            f"| {row['variant']} | {row['feature']} | {'yes' if row['included_in_scale'] else 'no'} | {coefficient} | {missing} |"
        )
    lines += [
        "",
        "Representative low/median/high RP predictions are in `scale_diagnostics.csv`; the full fitted coefficient table is in `scale_coefficients.csv`.",
        "",
        "## Interpretation",
        "",
        f"- C0 versus C1: C1 changes RP semantics by removing the four RP features from the location equation; its held-out delta is {float(c1['delta_nll_vs_c0']):+.4f} NLL and {float(c1['delta_crps_vs_c0']):+.4f} CRPS.",
        f"- C2 versus C1: scale-only RP adds a small aggregate NLL improvement of {float(incremental['nll']):+.4f}; it loses in 2022 and wins modestly in 2023–2025. The paired and bootstrap summaries above show why this should be treated as weak evidence.",
        "- C2 versus C0: scale-only RP loses in 2022–2023 but improves on C0 in 2024–2025, so it avoids the recent degradation in this panel without being uniformly better across every season.",
        f"- C2 calibration: 80% coverage is {float(c2['interval_80_coverage']):.3f} with width {float(c2['interval_80_average_width']):.1f}; compare the C0, C1, and C2 rows rather than interpreting coverage alone.",
        f"- Directional check: {direction}, but the individual RP scale coefficients have mixed signs and C2's intervals are wider with coverage farther above nominal. The evidence supports continued investigation of roster continuity as an uncertainty signal, not a production conclusion.",
        "",
        "## Limitations",
        "",
        "- Returning production is correlated with rank history, recruiting, talent, and coaching; the experiment is a conditional semantic ablation, not a causal transfer analysis.",
        "- The data do not model incoming transfers or reconstruct a transfer-adjusted roster, so low RP may proxy for several roster and measurement processes.",
        "- The 2022–2025 panel contains four season clusters and has appeared in earlier research; uncertainty summaries are descriptive and do not establish independent confirmation.",
        "- The scale diagnostic uses representative feature values and should not be mistaken for a population-average effect; the fitted individual RP coefficients also have mixed signs.",
        "",
        "## Reproduction",
        "",
        "```text",
        "uv run python scripts/investigate_returning_production_uncertainty.py --source-root /path/to/cached-input-checkout",
        "```",
        "",
        "Artifacts: `annual_metrics.csv`, `aggregate_metrics.csv`, `per_team_losses.csv`, `c2_vs_c1_annual.csv`, `c2_vs_c1_per_team.csv`, `c2_vs_c1_summary.json`, `scale_coefficients.csv`, `scale_diagnostics.csv`, `summary.json`, and deterministic plots under `plots/`.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(
    annual: list[dict[str, object]], levels: list[dict[str, object]]
) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for variant in VARIANTS:
        rows = [row for row in annual if row["variant"] == variant.name]
        years = [int(row["target_season"]) for row in rows]
        axes[0].plot(
            years, [float(row["nll"]) for row in rows], marker="o", label=variant.name
        )
        axes[1].plot(
            years,
            [float(row["interval_80_average_width"]) for row in rows],
            marker="o",
            label=variant.name,
        )
    axes[0].set_ylabel("NLL")
    axes[1].set_ylabel("80% interval width")
    axes[1].set_xlabel("target season")
    axes[0].legend()
    figure.suptitle("Frozen returning-production uncertainty variants")
    figure.tight_layout()
    figure.savefig(PLOTS / "heldout_metrics.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for variant in VARIANTS:
        rows = [row for row in levels if row["variant"] == variant.name]
        axis.plot(
            [str(row["returning_production_level"]) for row in rows],
            [float(row["predictive_scale"]) for row in rows],
            marker="o",
            label=variant.name,
        )
    axis.set_ylabel("predicted scale")
    axis.set_xlabel("representative returning-production level")
    axis.set_title("Scale response to returning production")
    axis.legend()
    figure.tight_layout()
    figure.savefig(PLOTS / "scale_by_returning_production.png", dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT,
        help="checkout containing cached historical inputs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUT,
        help="directory for this study's generated artifacts",
    )
    return parser.parse_args()


def main() -> None:
    global OUT, PLOTS
    args = parse_args()
    source_root = args.source_root.resolve()
    OUT = args.output_root.resolve()
    PLOTS = OUT / "plots"
    configure_source_root(source_root)
    _fbs, contextual, history = load_frozen_inputs(source_root)
    target_context = [row for row in contextual if row.season in TARGET_SEASONS]
    training = [row for row in contextual if row.season <= FROZEN_TRAIN_THROUGH]
    if {prediction.season for prediction in history} != set(TARGET_SEASONS):
        raise ValueError("stored H fallback does not cover every held-out season")

    models: dict[str, DirectRankModel] = {}
    predictions_by_variant: dict[str, list[v1.PriorPrediction]] = {}
    for variant in VARIANTS:
        print(f"fitting {variant.name}", flush=True)
        model = fit_variant(contextual, variant)
        models[variant.name] = model
        generated = make_predictions(model, target_context, variant.name)
        predictions_by_variant[variant.name] = merge_h_fallback(generated, history)

    annual, per_team = annual_rows(predictions_by_variant)
    aggregate = aggregate_rows(predictions_by_variant)
    c2_vs_c1_rows, c2_vs_c1 = c2_vs_c1_comparison(predictions_by_variant)
    coefficients, levels = scale_diagnostics(models, training)
    production_reference = load_production_reference(source_root)
    reproduction = c0_reproduction(
        score(predictions_by_variant[VARIANTS[0].name]), production_reference
    )
    if not reproduction["passed"]:
        raise RuntimeError(
            "C0 does not reproduce stored production metrics within tolerance: "
            f"{reproduction['max_absolute_difference']}"
        )
    input_paths = {
        "team_season_rank_distributions": source_root
        / "data/processed/modeling/team_season_rank_distributions.csv",
        "team_season_features": source_root
        / "data/processed/preseason/team_season_features.csv",
        "stored_context_evaluation": source_root
        / "data/processed/preseason/context/evaluation.json",
    }
    summary = {
        "study": "issue_90_returning_production_uncertainty",
        "primary_protocol": "fit once through 2021; score unchanged across 2022-2025",
        "target_seasons": list(TARGET_SEASONS),
        "frozen_train_through": FROZEN_TRAIN_THROUGH,
        "population": "exact stored production FBS evaluation panel with H fallback for rank-history cold starts",
        "n_team_seasons": len(predictions_by_variant[VARIANTS[0].name]),
        "metrics": list(METRICS),
        "penalty": 0.25,
        "preprocessing": "production training-only median imputation plus missingness indicators",
        "optimizer": "production L-BFGS-B settings with deterministic 2000-iteration retry",
        "variants": {
            variant.name: {
                "location_features": variant.location_features,
                "scale_features": variant.scale_features,
                "model_features": variant.feature_names,
            }
            for variant in VARIANTS
        },
        "aggregate_metrics": aggregate,
        "c2_vs_c1": c2_vs_c1,
        "c0_reproduction": reproduction,
        "source_hashes": {
            name: sha256_file(path) for name, path in input_paths.items()
        },
        "production_models_modified": False,
        "outcomes_through": 2025,
        "no_2026_outcomes_accessed": True,
    }
    write_csv("annual_metrics.csv", annual)
    write_csv("aggregate_metrics.csv", aggregate)
    write_csv("per_team_losses.csv", per_team)
    write_csv("c2_vs_c1_per_team.csv", c2_vs_c1_rows)
    write_csv("c2_vs_c1_annual.csv", c2_vs_c1["per_season"])
    write_json("c2_vs_c1_summary.json", c2_vs_c1)
    write_csv("scale_coefficients.csv", coefficients)
    write_csv("scale_diagnostics.csv", levels)
    write_json("summary.json", summary)
    render_report(summary, annual, aggregate, coefficients, levels, c2_vs_c1)
    plot_results(annual, levels)


if __name__ == "__main__":
    main()
