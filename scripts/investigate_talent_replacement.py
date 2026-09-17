"""Test whether roster-talent replacement conditions the Context RP signal.

The study is intentionally narrower than the existing context ablation.  It
uses the frozen production C 1.2 predictions as C0, refits C1/C2 variants on
data through 2021, and scores the unchanged 2022--2025 FBS population.  All
outputs are research artifacts under ``data/processed/talent_replacement``;
the production prior and frozen 2026 artifacts are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import build_preseason_context_prior_v1_2 as c12
import build_preseason_prior as v1
import build_preseason_prior_v1_1 as h11
import numpy as np

from gippyrank.context_ablation import standardized_interaction_rows
from gippyrank.preseason import DirectRankModel, Preprocessor, TeamSeason
from gippyrank.talent_replacement import add_seasonal_talent_delta

ROOT = Path(__file__).resolve().parents[1]
TEST_SEASONS = (2022, 2023, 2024, 2025)
TRAIN_THROUGH = 2021
CONTROL_METRIC_TOLERANCE = 1e-3
CONTROL_PMF_TOLERANCE = 2e-4
METRICS = (
    "nll",
    "crps",
    "expected_rank_mae",
    "median_rank_mae",
    "interval_80_coverage",
    "interval_80_average_width",
)
BASE_CONTEXT_FEATURES = [
    *c12.COACH_FEATURES,
    *c12.RECRUITING_FEATURES,
    *c12.TALENT_FEATURES,
    *c12.RETURNING_FEATURES,
]
TALENT_DELTA = "talent_delta"
PRIMARY_INTERACTION = "returning_pct_ppa_x_talent_delta"
SECONDARY_INTERACTIONS = {
    "C2_passing": (
        "returning_pct_passing_ppa",
        "returning_pct_passing_ppa_x_talent_delta",
    ),
    "C2_receiving": (
        "returning_pct_receiving_ppa",
        "returning_pct_receiving_ppa_x_talent_delta",
    ),
    "C2_rushing": (
        "returning_pct_rushing_ppa",
        "returning_pct_rushing_ppa_x_talent_delta",
    ),
}


def configure_data_root(data_root: Path) -> None:
    """Point the legacy builders at a caller-selected cached data root."""

    v1.ROOT = data_root
    v1.OUT = data_root / "data/processed/preseason"
    c12.ROOT = data_root
    c12.PRESEASON = v1.OUT
    c12.MODELING = data_root / "data/processed/modeling"
    c12.HISTORY = c12.PRESEASON / "history"
    c12.CONTEXT = c12.PRESEASON / "context"
    c12.TENURES = data_root / "data/raw/cfbd/preseason/coach_tenures"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty research artifact: {path}")
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


def feature_talent_values(
    index: dict[tuple[int, str, str], dict[str, str]], max_season: int
) -> dict[tuple[int, str, str], float | None]:
    return {
        key: c12.maybe_float(row.get("talent_composite"))
        for key, row in index.items()
        if key[0] <= max_season
    }


def metric_row(metrics: dict[str, Any]) -> dict[str, float | int]:
    return {
        "n_team_seasons": int(metrics["n_team_seasons"]),
        **{name: float(metrics[name]) for name in METRICS},
    }


def score_by_season(
    predictions: list[v1.PriorPrediction],
) -> dict[int, dict[str, float | int]]:
    return {
        season: metric_row(
            v1.score_predictions(
                [
                    prediction
                    for prediction in predictions
                    if prediction.season == season
                ]
            )
        )
        for season in TEST_SEASONS
    }


def load_stored_predictions(
    path: Path,
    rows: list[TeamSeason],
    cold: list[v1.ColdStartSeason],
) -> list[v1.PriorPrediction]:
    targets = {
        (row.season, row.subdivision, row.team_id): row.target_ranks for row in rows
    }
    targets.update(
        {(row.season, row.subdivision, row.team_id): row.target_ranks for row in cold}
    )
    result = []
    with path.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            key = (int(item["season"]), item["subdivision"], item["team_id"])
            if key not in targets:
                continue
            pmf = np.asarray(json.loads(item["pmf"]), dtype=float)
            result.append(
                v1.PriorPrediction(
                    key[0],
                    key[1],
                    key[2],
                    item["team_name"],
                    len(pmf),
                    targets[key],
                    item.get("model", "context_prior"),
                    item.get("prior_method", "same_subdivision_lag1"),
                    pmf,
                    float(item["conditional_location_mean"])
                    if item.get("conditional_location_mean")
                    else None,
                    float(item["predictive_scale"])
                    if item.get("predictive_scale")
                    else None,
                )
            )
    result = [prediction for prediction in result if prediction.season in TEST_SEASONS]
    if len({prediction.key for prediction in result}) != len(result):
        raise ValueError("stored C0 predictions contain duplicate keys")
    return result


def fit_context_variant(rows: list[TeamSeason], features: list[str]) -> DirectRankModel:
    """Fit with the production C layout and retry policy unchanged."""

    all_features = [*c12.H_FEATURES, *features]
    kwargs = {
        "penalty": 0.25,
        "location_feature_names": all_features,
        "scale_feature_names": c12.H_FEATURES,
    }
    try:
        return DirectRankModel.fit(rows, all_features, **kwargs)
    except RuntimeError as error:
        if "ITERATIONS REACHED LIMIT" not in str(error):
            raise
        return DirectRankModel.fit(
            rows, all_features, **kwargs, optimizer_options={"maxiter": 2000}
        )


def candidate_predictions(
    regular: list[v1.PriorPrediction],
    control: list[v1.PriorPrediction],
) -> list[v1.PriorPrediction]:
    regular_by_key = {prediction.key: prediction for prediction in regular}
    regular_keys = set(regular_by_key)
    control_keys = {prediction.key for prediction in control}
    if not regular_keys <= control_keys:
        raise ValueError("candidate population is outside the frozen C0 population")
    return [regular_by_key.get(prediction.key, prediction) for prediction in control]


def metric_deltas(
    candidate: dict[str, float | int], control: dict[str, float | int]
) -> dict[str, float]:
    return {
        f"delta_{name}": float(candidate[name]) - float(control[name])
        for name in METRICS
    }


def refit_control_sanity(
    model: DirectRankModel,
    regular_rows: list[TeamSeason],
    stored_control: list[v1.PriorPrediction],
) -> dict[str, float | int | bool]:
    regular_keys = {row_key(row) for row in regular_rows}
    stored_regular = [
        prediction for prediction in stored_control if prediction.key in regular_keys
    ]
    refit = h11.make_predictions(model, regular_rows, "C0_refit")
    stored_by_key = {prediction.key: prediction for prediction in stored_regular}
    pmf_differences = [
        float(np.max(np.abs(prediction.pmf - stored_by_key[prediction.key].pmf)))
        for prediction in refit
    ]
    refit_metrics = metric_row(v1.score_predictions(refit))
    stored_metrics = metric_row(v1.score_predictions(stored_regular))
    metric_differences = {
        name: abs(float(refit_metrics[name]) - float(stored_metrics[name]))
        for name in METRICS
    }
    return {
        "regular_n_team_seasons": len(refit),
        "max_regular_pmf_abs_difference": max(pmf_differences),
        "max_metric_abs_difference": max(metric_differences.values()),
        "pmf_within_tolerance": max(pmf_differences) <= CONTROL_PMF_TOLERANCE,
        "metrics_within_tolerance": max(metric_differences.values())
        <= CONTROL_METRIC_TOLERANCE,
        "metric_differences": metric_differences,
    }


def row_key(row: TeamSeason) -> tuple[int, str, str]:
    return (row.season, row.subdivision, row.team_id)


def interaction_feature(
    train: list[TeamSeason],
    target: list[TeamSeason],
    left: str,
    name: str,
) -> tuple[list[TeamSeason], list[TeamSeason], Preprocessor]:
    """Create a training-standardized product and retain its audit transform."""

    enriched_train, enriched_target, interaction_name = standardized_interaction_rows(
        train, target, left, TALENT_DELTA, name=name
    )
    if interaction_name != name:
        raise ValueError("interaction helper returned an unexpected feature name")
    prep = Preprocessor.fit([row.features for row in train], [left, TALENT_DELTA])
    return enriched_train, enriched_target, prep


def marginal_effect_diagnostic(
    model: DirectRankModel,
    train: list[TeamSeason],
    interaction_prep: Preprocessor,
    interaction_name: str,
    left_feature: str = "returning_pct_ppa",
) -> list[dict[str, object]]:
    """Evaluate the fitted RP effect at declining/flat/improving talent."""

    deltas = np.asarray(
        [
            row.features[TALENT_DELTA]
            for row in train
            if row.features.get(TALENT_DELTA) is not None
        ],
        dtype=float,
    )
    returning = np.asarray(
        [
            row.features[left_feature]
            for row in train
            if row.features.get(left_feature) is not None
        ],
        dtype=float,
    )
    if not len(deltas) or not len(returning):
        raise ValueError("marginal-effect diagnostic lacks observed training features")
    delta_levels = {
        "declining": float(np.quantile(deltas, 0.10)),
        "flat": float(np.quantile(deltas, 0.50)),
        "improving": float(np.quantile(deltas, 0.90)),
    }
    returning_low, returning_high = (
        float(np.quantile(returning, 0.25)),
        float(np.quantile(returning, 0.75)),
    )
    template = {name: model.preprocessor.medians[name] for name in model.feature_names}
    lag_values = np.concatenate([row.lag1_z for row in train])
    representative_lag = np.asarray([float(np.median(lag_values))])
    result = []
    for label, delta in delta_levels.items():
        low = dict(template)
        high = dict(template)
        low[TALENT_DELTA] = delta
        high[TALENT_DELTA] = delta
        low[left_feature] = returning_low
        high[left_feature] = returning_high
        for features in (low, high):
            transformed = interaction_prep.transform(
                [{left_feature: features[left_feature], TALENT_DELTA: delta}]
            )[0]
            features[interaction_name] = float(transformed[0] * transformed[1])
        low_location = float(
            np.mean(model.conditional_parameters(low, representative_lag)[0])
        )
        high_location = float(
            np.mean(model.conditional_parameters(high, representative_lag)[0])
        )
        result.append(
            {
                "talent_condition": label,
                "talent_delta_training_quantile": delta,
                "returning_pct_ppa_low_q25": returning_low,
                "returning_pct_ppa_high_q75": returning_high,
                "location_effect_high_minus_low_returning": high_location
                - low_location,
                "location_low_returning": low_location,
                "location_high_returning": high_location,
                "interpretation": "negative means the higher-returning case is modeled better on the rank-z coordinate",
            }
        )
    return result


def hashes(paths: list[Path]) -> dict[str, str]:
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }


def render_report(report: dict[str, Any], path: Path) -> None:
    control = report["control_sanity"]
    lines = [
        "# Talent-replacement Context experiment",
        "",
        "## Question and protocol",
        "",
        "This focused study tests whether returning production is better interpreted after conditioning on year-over-year roster-talent movement. C0 is the frozen production Context C 1.2 artifact. C1 adds season-relative `talent_delta` to the Context location equation. Primary C2 adds a training-standardized `returning_pct_ppa × talent_delta` product. Passing, receiving, and rushing interactions are secondary diagnostics.",
        "",
        "All fitted variants use source and outcome rows through 2021 only, production penalty 0.25, H-only scale equation, production preprocessing, optimizer/retry behavior, and the same 534-team-season FBS population in 2022–2025. The exact H fallback PMFs for rank-history cold starts are retained unchanged.",
        "",
        "`talent_delta` is `z(current Team Talent within the target season) − z(previous Team Talent within the previous season)`, using population mean/standard deviation for each FBS season. Missing current or previous talent remains missing and is handled by the existing training-only median plus indicator preprocessing.",
        "",
        "## C0 control sanity check",
        "",
        f"The refit control covers {control['regular_n_team_seasons']} regular-lag rows. Its maximum regular-row PMF difference from the stored C0 artifact is {control['max_regular_pmf_abs_difference']:.6g}; the largest metric difference is {control['max_metric_abs_difference']:.6g}. PMF tolerance pass: **{control['pmf_within_tolerance']}**. Metric tolerance pass: **{control['metrics_within_tolerance']}**.",
        "",
        "## Aggregate held-out results",
        "",
        "Negative deltas favor the candidate. Rank-coordinate effects are in the model's transformed rank-z scale; lower is better.",
        "",
        "| Variant | N | NLL | ΔNLL | CRPS | ΔCRPS | Expected MAE | Median MAE | 80% coverage | 80% width |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["aggregate_results"]:
        lines.append(
            f"| {row['variant']} | {row['n_team_seasons']} | {row['nll']:.4f} | {row['delta_nll']:+.4f} | {row['crps']:.4f} | {row['delta_crps']:+.4f} | {row['expected_rank_mae']:.2f} | {row['median_rank_mae']:.2f} | {row['interval_80_coverage']:.3f} | {row['interval_80_average_width']:.1f} |"
        )
    lines += [
        "",
        "## Year-by-year results",
        "",
        "| Variant | Season | N | NLL | ΔNLL | CRPS | ΔCRPS | Expected MAE | Median MAE | 80% coverage | 80% width |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["annual_results"]:
        lines.append(
            f"| {row['variant']} | {row['season']} | {row['n_team_seasons']} | {row['nll']:.4f} | {row['delta_nll']:+.4f} | {row['crps']:.4f} | {row['delta_crps']:+.4f} | {row['expected_rank_mae']:.2f} | {row['median_rank_mae']:.2f} | {row['interval_80_coverage']:.3f} | {row['interval_80_average_width']:.1f} |"
        )
    lines += [
        "",
        "## Interaction diagnostic",
        "",
        "| Talent condition | Talent-delta level | RP q25→q75 location effect | Low-RP location | High-RP location |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["marginal_effects"]:
        lines.append(
            f"| {row['talent_condition']} | {row['talent_delta_training_quantile']:+.3f} | {row['location_effect_high_minus_low_returning']:+.3f} | {row['location_low_returning']:+.3f} | {row['location_high_returning']:+.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        *[f"- {item}" for item in report["interpretation"]],
        "",
        "## Limitations",
        "",
        "- The 2022–2025 seasons are a leakage-safe historical holdout under this protocol, but they are not independent confirmation because the broader recent degradation motivated the study.",
        "- Team Talent and returning production are retrospective source fields with documented stability caveats; this experiment does not reconstruct an archived preseason payload.",
        "- A normalized talent change is not a direct transfer-portal measure. It can reflect recruiting, roster attrition, rating revisions, team movement, or source coverage changes.",
        "- The interaction diagnostic is descriptive and does not establish that transfers are the causal mechanism. No production C 1.3 promotion is made.",
        "",
        "## Reproduction",
        "",
        "Run `UV_CACHE_DIR=/tmp/gippyrank-uv-cache uv run python scripts/investigate_talent_replacement.py`. Use `--data-root` when the cached processed inputs live in another checkout and `--output-dir` to choose the research artifact directory.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/processed/talent_replacement"
    )
    args = parser.parse_args(argv)
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    configure_data_root(data_root)

    rows, cold, _ = v1.load_rows(max_season=max(TEST_SEASONS))
    fbs = [row for row in rows if row.subdivision == "fbs"]
    index = c12.feature_index()
    contextual, _coverage = c12.attach_context(fbs, index, c12.cached_tenures())
    contextual = add_seasonal_talent_delta(
        contextual, feature_talent_values(index, max(TEST_SEASONS))
    )
    train = [row for row in contextual if row.season <= TRAIN_THROUGH]
    target = [row for row in contextual if row.season in TEST_SEASONS]
    if not train or not target:
        raise ValueError("talent-replacement study lacks training or target rows")

    control = load_stored_predictions(c12.CONTEXT / "predictions.csv", rows, cold)
    if len(control) != 534:
        raise ValueError(f"expected 534 stored C0 rows, found {len(control)}")
    control_metrics = metric_row(v1.score_predictions(control))
    stored_evaluation = json.loads((c12.CONTEXT / "evaluation.json").read_text())[
        "all_fbs"
    ]
    evaluation_differences = {
        name: abs(
            float(control_metrics[name]) - float(stored_evaluation["candidate"][name])
        )
        for name in METRICS
    }
    if max(evaluation_differences.values()) > 1e-8:
        raise ValueError(
            "stored C0 predictions do not reproduce the stored production evaluation"
        )

    c0_model = fit_context_variant(train, BASE_CONTEXT_FEATURES)
    sanity = refit_control_sanity(c0_model, target, control)
    if not sanity["pmf_within_tolerance"] or not sanity["metrics_within_tolerance"]:
        raise ValueError(f"C0 refit exceeded tolerance: {sanity}")

    models: dict[str, DirectRankModel] = {"C0": c0_model}
    interaction_preps: dict[str, Preprocessor] = {}
    regular_predictions: dict[str, list[v1.PriorPrediction]] = {}

    print("fitting C1", flush=True)
    c1_model = fit_context_variant(train, [*BASE_CONTEXT_FEATURES, TALENT_DELTA])
    models["C1"] = c1_model
    regular_predictions["C1"] = h11.make_predictions(c1_model, target, "C1")

    interaction_specs = {
        "C2_total": ("returning_pct_ppa", PRIMARY_INTERACTION),
        **SECONDARY_INTERACTIONS,
    }
    for label, (left_feature, interaction_name) in interaction_specs.items():
        print(f"fitting {label}", flush=True)
        interaction_train, interaction_target, interaction_prep = interaction_feature(
            train, target, left_feature, interaction_name
        )
        model = fit_context_variant(
            interaction_train,
            [*BASE_CONTEXT_FEATURES, TALENT_DELTA, interaction_name],
        )
        models[label] = model
        interaction_preps[label] = interaction_prep
        regular_predictions[label] = h11.make_predictions(
            model, interaction_target, label
        )

    variant_predictions = {"C0": control}
    for label, predictions in regular_predictions.items():
        variant_predictions[label] = candidate_predictions(predictions, control)

    by_variant_season = {
        label: score_by_season(predictions)
        for label, predictions in variant_predictions.items()
    }
    aggregate_results = []
    annual_results = []
    control_by_season = by_variant_season["C0"]
    for label in ("C0", "C1", "C2_total", "C2_passing", "C2_receiving", "C2_rushing"):
        aggregate = metric_row(v1.score_predictions(variant_predictions[label]))
        aggregate_results.append(
            {
                "variant": label,
                **aggregate,
                **metric_deltas(aggregate, control_metrics),
            }
        )
        for season in TEST_SEASONS:
            scored = by_variant_season[label][season]
            annual_results.append(
                {
                    "variant": label,
                    "season": season,
                    **scored,
                    **metric_deltas(scored, control_by_season[season]),
                }
            )

    marginal_effects = marginal_effect_diagnostic(
        models["C2_total"],
        train,
        interaction_preps["C2_total"],
        PRIMARY_INTERACTION,
    )
    c2_by_season = by_variant_season["C2_total"]
    recent = lambda values: float(
        np.mean([values[year]["nll"] for year in (2024, 2025)])
    )
    effects = [
        row["location_effect_high_minus_low_returning"] for row in marginal_effects
    ]
    interpretation = [
        f"C1 talent_delta changes held-out aggregate NLL by {float(aggregate_results[1]['delta_nll']):+.4f} versus frozen C0; its 2024–2025 mean NLL delta is {recent(by_variant_season['C1']) - recent(control_by_season):+.4f}.",
        f"Primary C2_total changes aggregate NLL by {float(next(row['delta_nll'] for row in aggregate_results if row['variant'] == 'C2_total')):+.4f} versus C0 and by {recent(c2_by_season) - recent(control_by_season):+.4f} in 2024–2025; relative to C1, its aggregate NLL change is {float(next(row['delta_nll'] for row in aggregate_results if row['variant'] == 'C2_total')) - float(aggregate_results[1]['delta_nll']):+.4f}.",
        f"The primary interaction's modeled RP q25→q75 location effect is {effects[0]:+.3f} at declining talent, {effects[1]:+.3f} when flat, and {effects[2]:+.3f} when improving; the hypothesized monotone pattern (declining more negative than flat more negative than improving) is {effects[0] < effects[1] < effects[2]}.",
        f"The interaction variants preserve the full {len(control)}-row population and exact H fallback handling; no result is treated as evidence that transfers are the only explanation for recent RP degradation.",
    ]
    report = {
        "study": "talent replacement Context experiment",
        "protocol": {
            "train_through": TRAIN_THROUGH,
            "test_seasons": list(TEST_SEASONS),
            "population": "frozen production C 1.2 FBS evaluation population",
            "n_team_seasons": len(control),
            "scale_equation": "H features only",
            "penalty": 0.25,
            "talent_delta_definition": "season-relative z(current Team Talent) minus season-relative z(previous Team Talent)",
            "interaction_definition": "product of training-standardized returning production and training-standardized talent_delta",
            "cold_start_handling": "exact frozen production H fallback PMFs",
        },
        "feature_coverage": {
            "all_contextual_rows": len(contextual),
            "training_rows": len(train),
            "target_regular_lag_rows": len(target),
            "training_talent_delta_observed": sum(
                row.features.get(TALENT_DELTA) is not None for row in train
            ),
            "target_talent_delta_observed": sum(
                row.features.get(TALENT_DELTA) is not None for row in target
            ),
        },
        "control_sanity": {
            **sanity,
            "stored_control_evaluation_differences": evaluation_differences,
        },
        "aggregate_results": aggregate_results,
        "annual_results": annual_results,
        "marginal_effects": marginal_effects,
        "interpretation": interpretation,
        "input_hashes": hashes(
            [
                c12.PRESEASON / "team_season_features.csv",
                c12.MODELING / "team_season_rank_distributions.csv",
                c12.CONTEXT / "predictions.csv",
                c12.CONTEXT / "evaluation.json",
            ]
        ),
    }
    write_json(output_dir / "summary.json", report)
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_results)
    write_csv(output_dir / "annual_metrics.csv", annual_results)
    write_csv(output_dir / "marginal_effects.csv", marginal_effects)
    write_json(
        output_dir / "model_metadata.json",
        {label: model.metadata() for label, model in models.items()},
    )
    render_report(report, output_dir / "report.md")
    print(
        json.dumps(
            {"output_dir": str(output_dir), "n_team_seasons": len(control)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
