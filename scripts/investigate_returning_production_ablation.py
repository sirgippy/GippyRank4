"""Primary frozen-test and secondary rolling ablations of returning production.

The primary comparison keeps the established C 1.2 evaluation protocol: both
models are fit through 2021 and scored unchanged on 2022--2025.  A secondary
rolling-origin diagnostic retains the full year-by-year history.  C-minus-RP
differs from C-full only by removing the returning-production feature family.
The study writes research artifacts only under
``data/processed/returning_production_ablation``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import build_preseason_context_prior_v1_2 as c12
import build_preseason_prior as v1
import build_preseason_prior_v1_1 as h11
import matplotlib.pyplot as plt
import numpy as np

from gippyrank.context_ablation import (
    assert_same_population,
    remove_features,
    training_rows,
)
from gippyrank.preseason import DirectRankModel, TeamSeason, pmf_summaries

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/returning_production_ablation"
PLOTS = OUT / "plots"
DEFAULT_TARGET_MIN = 2007
DEFAULT_TARGET_MAX = 2025
MIN_TRAIN_SEASONS = 3
RECENT_START = 2022
FROZEN_TRAIN_THROUGH = 2021
RP_OBSERVABLE_START = 2015
PRODUCTION_PARITY_RTOL = 1e-8
PRODUCTION_PARITY_ATOL = 1e-3

H_FEATURES = tuple(c12.H_FEATURES)
CONTEXT_FEATURES = (
    *c12.COACH_FEATURES,
    *c12.RECRUITING_FEATURES,
    *c12.TALENT_FEATURES,
    *c12.RETURNING_FEATURES,
)
RP_FEATURES = tuple(c12.RETURNING_FEATURES)
C_MINUS_RP_FEATURES = tuple(remove_features(CONTEXT_FEATURES, RP_FEATURES))
METRICS = (
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


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    """Write a non-empty tabular artifact under the study output directory."""
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


def configure_source_root(source_root: Path) -> None:
    """Point the existing cached-input loaders at an explicit checkout."""
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


def fit_context(
    rows: list[TeamSeason],
    features: Iterable[str],
    *,
    target_season: int,
    trained_through_season: int,
) -> DirectRankModel:
    """Use the exact production C 1.2 fit, including its retry behavior."""
    model, _instance = c12.build_context_prior(
        rows,
        target_season=target_season,
        trained_through_season=trained_through_season,
        context_features=list(features),
        mode="location",
    )
    return model


def predictions(
    model: DirectRankModel, rows: list[TeamSeason], name: str
) -> list[v1.PriorPrediction]:
    return h11.make_predictions(model, rows, name)


def complete_production_panel(
    history: list[v1.PriorPrediction],
    contextual_predictions: list[v1.PriorPrediction],
    contextual_keys: set[tuple[int, str, str]],
    *,
    target_min: int,
    target_max: int,
) -> tuple[list[v1.PriorPrediction], set[tuple[int, str, str]]]:
    """Restore production H fallbacks around predictions from an ablation model."""
    production_rows = [
        prediction
        for prediction in history
        if target_min <= prediction.season <= target_max
    ]
    if not production_rows:
        raise ValueError("production held-out panel is empty")
    production_keys = {prediction.key for prediction in production_rows}
    updates = {prediction.key: prediction for prediction in contextual_predictions}
    if set(updates) != contextual_keys:
        raise ValueError("context predictions do not cover the expected target keys")
    if not contextual_keys <= production_keys:
        raise ValueError("context predictions extend beyond the production panel")
    fallback_keys = production_keys - contextual_keys
    panel = [updates.get(prediction.key, prediction) for prediction in production_rows]
    if {prediction.key for prediction in panel} != production_keys:
        raise ValueError("completed production panel changed target keys")
    if any(
        prediction.model != c12.H_MODEL_NAME
        for prediction in production_rows
        if prediction.key in fallback_keys
    ):
        raise ValueError("production fallback rows are not the stored H predictions")
    return panel, fallback_keys


def top_brier(predictions_: list[v1.PriorPrediction], cutoff: int) -> float:
    return float(
        np.mean(
            [
                (
                    pmf_summaries(prediction.pmf)[f"top{cutoff}_probability"]
                    - float(np.mean(prediction.target_ranks <= cutoff))
                )
                ** 2
                for prediction in predictions_
            ]
        )
    )


def score(predictions_: list[v1.PriorPrediction]) -> dict[str, float]:
    """Flatten the established Context evaluation metrics for one target year."""
    raw = v1.score_predictions(predictions_)
    result = {
        metric: float(raw[metric])
        for metric in METRICS[:6]
        if raw.get(metric) is not None
    }
    for cutoff in (5, 10, 25):
        result[f"top{cutoff}_brier"] = top_brier(predictions_, cutoff)
    return result


def usable_target_years(
    rows: list[TeamSeason], target_min: int, target_max: int
) -> list[int]:
    years = []
    for target in sorted(
        {row.season for row in rows if target_min <= row.season <= target_max}
    ):
        train = training_rows(rows, target)
        target_count = sum(row.season == target for row in rows)
        if len({row.season for row in train}) >= MIN_TRAIN_SEASONS and target_count >= 20:
            years.append(target)
    return years


def compare_predictions(
    full_predictions: list[v1.PriorPrediction],
    minus_rp_predictions: list[v1.PriorPrediction],
    *,
    protocol: str,
    n_training_rows: int,
    n_training_seasons: int,
    fallback_keys: set[tuple[int, str, str]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Compare paired predictions, retaining annual and team-level metrics."""
    assert_same_population(full_predictions, minus_rp_predictions)
    fallback_keys = fallback_keys or set()
    population_keys = {prediction.key for prediction in full_predictions}
    if not fallback_keys <= population_keys:
        raise ValueError("fallback keys are outside the ablation population")
    full_by_key = {prediction.key: prediction for prediction in full_predictions}
    minus_by_key = {
        prediction.key: prediction for prediction in minus_rp_predictions
    }
    for key in fallback_keys:
        if not np.array_equal(full_by_key[key].pmf, minus_by_key[key].pmf):
            raise ValueError("H fallback predictions differ between ablation variants")
    annual: list[dict[str, object]] = []
    per_team: list[dict[str, object]] = []
    for target in sorted({prediction.season for prediction in full_predictions}):
        full_year = [prediction for prediction in full_predictions if prediction.season == target]
        minus_rp_year = [
            prediction for prediction in minus_rp_predictions if prediction.season == target
        ]
        full_scores = score(full_year)
        minus_scores = score(minus_rp_year)
        annual_row: dict[str, object] = {
            "evaluation_protocol": protocol,
            "target_season": target,
            "n_team_seasons": len(full_year),
            "n_training_rows": n_training_rows,
            "n_training_seasons": n_training_seasons,
            "same_population_keys": True,
            "n_h_fallback_team_seasons": sum(
                prediction.key in fallback_keys for prediction in full_year
            ),
        }
        for metric in METRICS:
            annual_row[f"c_full_{metric}"] = full_scores[metric]
            annual_row[f"c_minus_rp_{metric}"] = minus_scores[metric]
            annual_row[f"rp_contribution_{metric}"] = (
                minus_scores[metric] - full_scores[metric]
            )
        annual.append(annual_row)

        full_losses = h11.prediction_losses(full_year)
        minus_rp_losses = h11.prediction_losses(minus_rp_year)
        names = {prediction.key: prediction.team_name for prediction in full_year}
        for key in sorted(full_losses):
            full_nll, full_crps = full_losses[key]
            minus_nll, minus_crps = minus_rp_losses[key]
            per_team.append(
                {
                    "evaluation_protocol": protocol,
                    "season": key[0],
                    "subdivision": key[1],
                    "team_id": key[2],
                    "team_name": names[key],
                    "c_full_nll": full_nll,
                    "c_minus_rp_nll": minus_nll,
                    "rp_contribution_nll": minus_nll - full_nll,
                    "c_full_crps": full_crps,
                    "c_minus_rp_crps": minus_crps,
                    "rp_contribution_crps": minus_crps - full_crps,
                    "prediction_source": (
                        "h_fallback" if key in fallback_keys else "context_model"
                    ),
                }
            )
    return annual, per_team


def run_frozen_ablation(
    rows: list[TeamSeason],
    history: list[v1.PriorPrediction],
    *,
    target_min: int,
    target_max: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Fit once through 2021 and score the exact production held-out panel."""
    target_rows = [row for row in rows if target_min <= row.season <= target_max]
    if not target_rows:
        raise ValueError("no held-out target rows for returning-production ablation")
    training = [row for row in rows if row.season <= FROZEN_TRAIN_THROUGH]
    if not training:
        raise ValueError("no training rows through the frozen cutoff")
    full_model = fit_context(
        rows,
        CONTEXT_FEATURES,
        target_season=target_min,
        trained_through_season=FROZEN_TRAIN_THROUGH,
    )
    minus_rp_model = fit_context(
        rows,
        C_MINUS_RP_FEATURES,
        target_season=target_min,
        trained_through_season=FROZEN_TRAIN_THROUGH,
    )
    contextual_keys = {
        (row.season, row.subdivision, row.team_id) for row in target_rows
    }
    full_predictions, fallback_keys = complete_production_panel(
        history,
        predictions(full_model, target_rows, "C-full"),
        contextual_keys,
        target_min=target_min,
        target_max=target_max,
    )
    minus_rp_predictions, minus_fallback_keys = complete_production_panel(
        history,
        predictions(minus_rp_model, target_rows, "C-minus-RP"),
        contextual_keys,
        target_min=target_min,
        target_max=target_max,
    )
    if fallback_keys != minus_fallback_keys:
        raise ValueError("ablation variants have different H fallback populations")
    return compare_predictions(
        full_predictions,
        minus_rp_predictions,
        protocol="frozen_through_2021",
        n_training_rows=len(training),
        n_training_seasons=len({row.season for row in training}),
        fallback_keys=fallback_keys,
    )


def run_rolling_ablation(
    rows: list[TeamSeason], *, target_min: int, target_max: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run a secondary rolling-origin diagnostic on a fixed population."""
    full_rows = [row for row in rows if row.season <= target_max]
    years = usable_target_years(full_rows, target_min, target_max)
    if not years:
        raise ValueError("no usable target seasons for returning-production ablation")

    annual: list[dict[str, object]] = []
    per_team: list[dict[str, object]] = []
    for target in years:
        train = training_rows(full_rows, target)
        target_rows = [row for row in full_rows if row.season == target]
        if not target_rows:
            raise ValueError(f"no target rows for {target}")

        full_model = fit_context(
            full_rows,
            CONTEXT_FEATURES,
            target_season=target,
            trained_through_season=target - 1,
        )
        minus_rp_model = fit_context(
            full_rows,
            C_MINUS_RP_FEATURES,
            target_season=target,
            trained_through_season=target - 1,
        )
        full_predictions = predictions(full_model, target_rows, "C-full")
        minus_rp_predictions = predictions(
            minus_rp_model, target_rows, "C-minus-RP"
        )
        year_annual, year_per_team = compare_predictions(
            full_predictions,
            minus_rp_predictions,
            protocol="rolling_origin",
            n_training_rows=len(train),
            n_training_seasons=len({row.season for row in train}),
        )
        annual.extend(year_annual)
        per_team.extend(year_per_team)
        print(f"completed returning-production ablation/{target}", flush=True)
    return annual, per_team


def weighted_mean(rows: list[dict[str, object]], column: str) -> float:
    weights = np.asarray([int(row["n_team_seasons"]) for row in rows], dtype=float)
    values = np.asarray([float(row[column]) for row in rows], dtype=float)
    return float(np.average(values, weights=weights))


def summarize_rows(
    rows: list[dict[str, object]], *, label: str, start: int, end: int
) -> dict[str, object]:
    if not rows:
        raise ValueError(f"cannot summarize empty period: {label}")
    item: dict[str, object] = {
        "period": label,
        "start_season": start,
        "end_season": end,
        "target_seasons": ";".join(str(row["target_season"]) for row in rows),
        "n_target_seasons": len(rows),
        "n_team_seasons": sum(int(row["n_team_seasons"]) for row in rows),
    }
    if "n_h_fallback_team_seasons" in rows[0]:
        item["n_h_fallback_team_seasons"] = sum(
            int(row["n_h_fallback_team_seasons"]) for row in rows
        )
    for metric in METRICS:
        item[f"c_full_{metric}"] = weighted_mean(rows, f"c_full_{metric}")
        item[f"c_minus_rp_{metric}"] = weighted_mean(
            rows, f"c_minus_rp_{metric}"
        )
        item[f"rp_contribution_{metric}"] = weighted_mean(
            rows, f"rp_contribution_{metric}"
        )
    return item


def validate_production_control(
    primary_annual: list[dict[str, object]],
    primary_summary: dict[str, object],
    source_root: Path,
) -> dict[str, object]:
    """Guard that the ablation's C-full reproduces stored production C 1.2."""
    evaluation_path = source_root / "data/processed/preseason/context/evaluation.json"
    report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    all_fbs = report["all_fbs"]
    candidate = all_fbs["candidate"]

    def expected_metrics(item: dict[str, object]) -> dict[str, float]:
        metrics = {metric: float(item[metric]) for metric in METRICS[:6]}
        for cutoff in (5, 10, 25):
            metrics[f"top{cutoff}_brier"] = float(
                item[f"top{cutoff}"]["reliability"]["brier_score"]
            )
        return metrics

    def compare(label: str, actual: dict[str, object], expected: dict[str, object]) -> float:
        differences = {
            metric: abs(float(actual[f"c_full_{metric}"]) - value)
            for metric, value in expected_metrics(expected).items()
        }
        maximum = max(differences.values())
        if not all(
            np.isclose(
                value,
                0.0,
                rtol=PRODUCTION_PARITY_RTOL,
                atol=PRODUCTION_PARITY_ATOL,
            )
            for value in differences.values()
        ):
            raise ValueError(
                f"C-full does not reproduce stored production C metrics for {label}: "
                f"{differences}"
            )
        return maximum

    if int(primary_summary["n_team_seasons"]) != int(candidate["n_team_seasons"]):
        raise ValueError(
            "C-full population does not match stored production C population: "
            f"{primary_summary['n_team_seasons']} != {candidate['n_team_seasons']}"
        )
    aggregate_difference = compare("all_fbs", primary_summary, candidate)
    production_per_season = all_fbs["per_season"]
    annual_by_year = {str(row["target_season"]): row for row in primary_annual}
    annual_differences = [
        compare(year, annual_by_year[year], production_per_season[year]["c"])
        for year in sorted(production_per_season)
    ]
    return {
        "evaluation_artifact": str(evaluation_path.relative_to(source_root)),
        "aggregate_match": True,
        "annual_match": True,
        "expected_n_team_seasons": int(candidate["n_team_seasons"]),
        "n_h_fallback_team_seasons": int(
            primary_summary["n_h_fallback_team_seasons"]
        ),
        "max_abs_metric_difference": max(
            aggregate_difference, *annual_differences
        ),
        "tolerance": (
            f"numpy.isclose(rtol={PRODUCTION_PARITY_RTOL:g}, "
            f"atol={PRODUCTION_PARITY_ATOL:g})"
        ),
    }


def period_rows(
    annual: list[dict[str, object]], *, target_min: int, target_max: int
) -> list[dict[str, object]]:
    """Summarize fixed descriptive periods without fitting a breakpoint."""
    definitions = (
        ("pre_2022", target_min, min(target_max, RECENT_START - 1)),
        (
            "rp_observable_pre_2022",
            max(target_min, RP_OBSERVABLE_START),
            min(target_max, RECENT_START - 1),
        ),
        ("recent_2022_2025", max(target_min, RECENT_START), target_max),
    )
    result = []
    for label, start, end in definitions:
        rows = [
            row
            for row in annual
            if start <= int(row["target_season"]) <= end
        ]
        if not rows:
            continue
        result.append(summarize_rows(rows, label=label, start=start, end=end))
    return result


def slopes(annual: list[dict[str, object]]) -> dict[str, float]:
    years = np.asarray([int(row["target_season"]) for row in annual], dtype=float)
    return {
        metric: float(
            np.polyfit(
                years,
                np.asarray([float(row[f"{metric}"]) for row in annual]),
                1,
            )[0]
        )
        for metric in (
            "c_full_nll",
            "c_minus_rp_nll",
            "rp_contribution_nll",
            "c_full_crps",
            "c_minus_rp_crps",
            "rp_contribution_crps",
        )
    }


def period_difference(
    periods: list[dict[str, object]],
    column: str,
    earlier: str = "pre_2022",
    later: str = "recent_2022_2025",
) -> float | None:
    by_name = {str(row["period"]): row for row in periods}
    if earlier not in by_name or later not in by_name:
        return None
    return float(by_name[later][column] - by_name[earlier][column])


def make_summary(
    primary_annual: list[dict[str, object]],
    primary_summary: dict[str, object],
    rolling_annual: list[dict[str, object]],
    rolling_periods: list[dict[str, object]],
    *,
    target_min: int,
    target_max: int,
    source_root: Path,
    production_control_parity: dict[str, object],
) -> dict[str, object]:
    full_degradation = period_difference(rolling_periods, "c_full_nll")
    minus_degradation = period_difference(rolling_periods, "c_minus_rp_nll")
    contribution_change = period_difference(rolling_periods, "rp_contribution_nll")
    degradation_change = (
        None
        if full_degradation is None or minus_degradation is None
        else minus_degradation - full_degradation
    )
    return {
        "study": "issue_84_returning_production_ablation",
        "primary_evaluation_protocol": "fit both variants through 2021; score unchanged on 2022-2025",
        "primary_target_season_range": [RECENT_START, target_max],
        "primary_target_seasons": [
            int(row["target_season"]) for row in primary_annual
        ],
        "secondary_rolling_target_season_range": [target_min, target_max],
        "secondary_rolling_target_seasons": [
            int(row["target_season"]) for row in rolling_annual
        ],
        "minimum_training_seasons": MIN_TRAIN_SEASONS,
        "recent_period_definition": "2022-2025, the established temporally held-out Context evaluation period",
        "rp_observable_period_definition": "2015-2021, a reporting era after returning-production coverage begins in 2014",
        "same_population_keys_required": True,
        "population": "Primary held-out panel is the exact all-FBS C 1.2 evaluation population, including H fallbacks for rank-history cold starts; secondary rolling diagnostics use contextual rows.",
        "primary_population": "All production C 1.2 FBS team-seasons for 2022-2025, with exact stored H predictions retained for rank-history cold starts",
        "secondary_population": "All contextual FBS team-seasons available to the rolling fit; H fallback rows are not used in the secondary rolling diagnostic",
        "preprocessing": "existing DirectRankModel training-only median imputation and missingness indicators",
        "penalty": 0.25,
        "location_features": {
            "c_full": [*H_FEATURES, *CONTEXT_FEATURES],
            "c_minus_rp": [*H_FEATURES, *C_MINUS_RP_FEATURES],
        },
        "c_full_context_features": list(CONTEXT_FEATURES),
        "c_minus_rp_context_features": list(C_MINUS_RP_FEATURES),
        "removed_returning_production_features": list(RP_FEATURES),
        "production_models_modified": False,
        "no_2026_outcomes_accessed": target_max < 2026,
        "primary_heldout_summary": primary_summary,
        "production_control_parity": production_control_parity,
        "secondary_rolling_periods": rolling_periods,
        "secondary_rolling_linear_slopes_per_year": slopes(rolling_annual),
        "secondary_rolling_nll_period_changes_recent_minus_pre_2022": {
            "c_full": full_degradation,
            "c_minus_rp": minus_degradation,
            "rp_contribution": contribution_change,
            "degradation_change_after_removing_rp": degradation_change,
        },
        "source_hashes": {
            "data/processed/modeling/team_season_rank_distributions.csv": sha256_file(
                source_root
                / "data/processed/modeling/team_season_rank_distributions.csv"
            ),
            "data/processed/preseason/team_season_features.csv": sha256_file(
                source_root / "data/processed/preseason/team_season_features.csv"
            ),
            "data/processed/preseason/context/evaluation.json": sha256_file(
                source_root / "data/processed/preseason/context/evaluation.json"
            ),
        },
    }


def fmt(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def render_report(
    summary: dict[str, object],
    primary_annual: list[dict[str, object]],
    primary_summary: dict[str, object],
    rolling_periods: list[dict[str, object]],
) -> None:
    primary_contribution = float(primary_summary["rp_contribution_nll"])
    primary_full_nll = float(primary_summary["c_full_nll"])
    primary_minus_nll = float(primary_summary["c_minus_rp_nll"])
    primary_fallback_count = int(primary_summary["n_h_fallback_team_seasons"])
    rolling_changes = summary[
        "secondary_rolling_nll_period_changes_recent_minus_pre_2022"
    ]
    rolling_contribution_change = rolling_changes["rp_contribution"]
    period_by_name = {str(row["period"]): row for row in rolling_periods}
    pre_contribution = period_by_name.get("pre_2022", {}).get(
        "rp_contribution_nll"
    )
    observable_contribution = period_by_name.get(
        "rp_observable_pre_2022", {}
    ).get("rp_contribution_nll")
    recent_contribution = period_by_name.get("recent_2022_2025", {}).get(
        "rp_contribution_nll"
    )
    rolling_degradation_change = rolling_changes["degradation_change_after_removing_rp"]
    primary_direction = (
        "helped"
        if primary_contribution > 0
        else "hurt"
        if primary_contribution < 0
        else "made no measurable difference to"
    )
    rolling_direction = (
        "consistent with declining value"
        if rolling_contribution_change is not None and rolling_contribution_change < 0
        else "not evidence of declining value"
    )
    lines = [
        "# Returning-production ablation study (issue 84)",
        "",
        "## Primary result: frozen C evaluation",
        "",
        f"Both variants were fit through 2021 and scored unchanged on 2022–2025. C-full NLL was **{fmt(primary_full_nll)}**; C-minus-RP NLL was **{fmt(primary_minus_nll)}**; RP contribution was **{fmt(primary_contribution)}** NLL, so returning production {primary_direction} on the established held-out panel.",
        f"C-full CRPS was **{fmt(primary_summary['c_full_crps'])}** versus **{fmt(primary_summary['c_minus_rp_crps'])}** without RP. Positive RP contribution means RP helped; negative means it hurt.",
        f"The primary panel contains **{primary_summary['n_team_seasons']}** team-seasons, including **{primary_fallback_count}** rank-history cold starts scored with identical stored H fallback predictions in both variants. The C-full aggregate and yearly metrics match the stored production C 1.2 evaluation within the recorded numerical tolerance.",
        "This frozen comparison is the primary answer to issue 84. It is separate from the rolling-origin diagnostic below.",
        "",
        "## Exact experiment",
        "",
        "- C-full uses the current Context C 1.2 location equation: rank-history features plus coach tenure, recruiting, Talent, and all returning-production features.",
        "- C-minus-RP uses the same equation, penalty (0.25), production fitting path, optimizer defaults, and iteration-limit retry, with only the four returning-production features removed.",
        "- The primary comparison uses the exact all-FBS C 1.2 held-out panel, including stored H fallback predictions for rank-history cold starts; those rows are identical in both variants and therefore contribute zero RP difference. The secondary rolling diagnostic remains on contextual rows only.",
        "- Primary protocol: fit each variant once on rows through 2021, then score the unchanged fits across 2022–2025. 2026 is excluded.",
        "- Secondary protocol: refit each target season using only earlier seasons, preserving the prior report’s rolling-origin diagnostic for temporal shape analysis.",
        "",
        "Reproduce with:",
        "",
        "```text",
        "uv run python scripts/investigate_returning_production_ablation.py --source-root /path/to/cached-input-checkout",
        "```",
        "",
        "## Primary year-by-year results",
        "",
        "RP contribution is `NLL(C-minus-RP) - NLL(C-full)`: positive means RP helped; negative means RP hurt. The complete established metric set is in `annual_metrics.csv`; the table shows the primary metrics.",
        "",
        "| Season | N | C-full NLL | C-minus-RP NLL | RP contribution | C-full CRPS | C-minus-RP CRPS | C-full rank MAE | C-minus-RP rank MAE | C-full 80% cov. | C-minus-RP 80% cov. |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary_annual:
        lines.append(
            "| {target_season} | {n_team_seasons} | {c_full_nll:.4f} | {c_minus_rp_nll:.4f} | {rp_contribution_nll:+.4f} | {c_full_crps:.4f} | {c_minus_rp_crps:.4f} | {c_full_expected_rank_mae:.2f} | {c_minus_rp_expected_rank_mae:.2f} | {c_full_interval_80_coverage:.3f} | {c_minus_rp_interval_80_coverage:.3f} |".format(
                **row
            )
        )
    lines += [
        "",
        "## Secondary rolling-origin diagnostic",
        "",
        f"The rolling-origin diagnostic is secondary and does not replace the frozen held-out comparison. It is {rolling_direction} in the temporal pattern: RP contribution was {fmt(observable_contribution) if observable_contribution is not None else 'n/a'} in the RP-observable pre-2022 era (2015–2021) and {fmt(recent_contribution) if recent_contribution is not None else 'n/a'} in 2022–2025. Removing RP changed recent-vs-pre-2022 deterioration by {fmt(rolling_degradation_change) if rolling_degradation_change is not None else 'n/a'} NLL.",
        "",
        "| Period | Seasons | N | C-full NLL | C-minus-RP NLL | RP contribution | C-full CRPS | C-minus-RP CRPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rolling_periods:
        lines.append(
            "| {period} | {target_seasons} | {n_team_seasons} | {c_full_nll:.4f} | {c_minus_rp_nll:.4f} | {rp_contribution_nll:+.4f} | {c_full_crps:.4f} | {c_minus_rp_crps:.4f} |".format(
                **row
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Returning production is useful when its contribution is positive, but a worse absolute C-minus-RP score is expected and is not by itself evidence against feature drift.",
        f"- The primary frozen result is the relevant held-out answer: RP {primary_direction} by {fmt(primary_contribution)} NLL on 2022–2025.",
        f"- The secondary rolling result compares {fmt(pre_contribution) if pre_contribution is not None else 'n/a'} before 2022, {fmt(observable_contribution) if observable_contribution is not None else 'n/a'} in 2015–2021, and {fmt(recent_contribution) if recent_contribution is not None else 'n/a'} in 2022–2025. This is consistent with the hypothesis that RP became less informative, but it is not causal evidence that omitted transfers caused the drift.",
        "",
        "## Limitations",
        "",
        "- Returning production is correlated with recruiting, Talent, coaching, and rank history; this is a conditional model ablation, not a causal estimate of transfers.",
        "- The returning-production source excludes incoming transfers, but this study does not observe or reconstruct transfer-adjusted rosters. Other roster changes, source revisions, and changing data coverage are alternative explanations.",
        "- The frozen 2022–2025 comparison is short, and the seasons are not independent. The rolling-origin period comparisons and linear slopes in `summary.json` are descriptive.",
        "- The study evaluates historical retrospective source values under the repository's established preprocessing; it does not modify or reforecast the production 2026 artifacts.",
        "",
        "## Artifacts",
        "",
        "- `annual_metrics.csv` — primary frozen 2022–2025 model metrics and RP contribution for every metric.",
        "- `per_team_losses.csv` — primary paired team-season NLL and CRPS losses, with the prediction source recorded for H fallback rows.",
        "- `heldout_summary.csv` — primary aggregate held-out metrics.",
        "- `rolling_annual_metrics.csv` and `rolling_per_team_losses.csv` — secondary rolling-origin diagnostics.",
        "- `period_summary.csv` — secondary pre-2022, RP-observable-era, and recent descriptive aggregates.",
        "- `summary.json` — feature lists, hashes, configuration, period changes, and temporal slopes.",
        "- `plots/heldout_rp_contribution_nll.png` — primary frozen held-out comparison.",
        "- `plots/rp_contribution_nll.png` — secondary rolling-origin comparison.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(
    primary_annual: list[dict[str, object]],
    rolling_annual: list[dict[str, object]],
) -> None:
    def plot(annual: list[dict[str, object]], filename: str, title: str) -> None:
        years = [int(row["target_season"]) for row in annual]
        full_nll = [float(row["c_full_nll"]) for row in annual]
        minus_nll = [float(row["c_minus_rp_nll"]) for row in annual]
        contribution = [float(row["rp_contribution_nll"]) for row in annual]
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(years, full_nll, marker="o", label="C-full")
        axes[0].plot(years, minus_nll, marker="o", label="C-minus-RP")
        axes[0].set_ylabel("annual NLL")
        axes[0].set_title(title)
        axes[0].legend()
        colors = ["#2f855a" if value < 0 else "#c53030" for value in contribution]
        axes[1].bar(years, contribution, color=colors)
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_ylabel("NLL(C-minus-RP) − NLL(C-full)")
        axes[1].set_xlabel("target season")
        axes[1].set_title("Marginal returning-production contribution")
        figure.tight_layout()
        PLOTS.mkdir(parents=True, exist_ok=True)
        figure.savefig(PLOTS / filename, dpi=160)
        plt.close(figure)

    plot(
        primary_annual,
        "heldout_rp_contribution_nll.png",
        "Frozen 2022–2025 Context evaluation",
    )
    plot(
        rolling_annual,
        "rp_contribution_nll.png",
        "Rolling-origin Context diagnostic",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-min", type=int, default=DEFAULT_TARGET_MIN)
    parser.add_argument("--target-max", type=int, default=DEFAULT_TARGET_MAX)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT,
        help="checkout containing the cached historical inputs (defaults to this checkout)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_min > args.target_max:
        raise ValueError("--target-min must not exceed --target-max")
    source_root = args.source_root.resolve()
    configure_source_root(source_root)
    rows, cold, _coverage = v1.load_rows(max_season=args.target_max)
    fbs = [row for row in rows if row.subdivision == "fbs"]
    fbs_cold = [row for row in cold if row.subdivision == "fbs"]
    contextual, _ = c12.attach_context(fbs, c12.feature_index(), c12.cached_tenures())
    history = c12.raw_history_predictions(fbs, fbs_cold)
    primary_annual, primary_per_team = run_frozen_ablation(
        contextual,
        history,
        target_min=max(args.target_min, RECENT_START),
        target_max=args.target_max,
    )
    rolling_annual, rolling_per_team = run_rolling_ablation(
        contextual, target_min=args.target_min, target_max=args.target_max
    )
    rolling_periods = period_rows(
        rolling_annual, target_min=args.target_min, target_max=args.target_max
    )
    primary_summary = summarize_rows(
        primary_annual,
        label="frozen_heldout_2022_2025",
        start=RECENT_START,
        end=args.target_max,
    )
    production_control_parity = validate_production_control(
        primary_annual, primary_summary, source_root
    )
    summary = make_summary(
        primary_annual,
        primary_summary,
        rolling_annual,
        rolling_periods,
        target_min=args.target_min,
        target_max=args.target_max,
        source_root=source_root,
        production_control_parity=production_control_parity,
    )
    write_csv("annual_metrics.csv", primary_annual)
    write_csv("per_team_losses.csv", primary_per_team)
    write_csv("heldout_summary.csv", [primary_summary])
    write_csv("rolling_annual_metrics.csv", rolling_annual)
    write_csv("rolling_per_team_losses.csv", rolling_per_team)
    write_csv("period_summary.csv", rolling_periods)
    write_json("summary.json", summary)
    render_report(summary, primary_annual, primary_summary, rolling_periods)
    plot_results(primary_annual, rolling_annual)


if __name__ == "__main__":
    main()
