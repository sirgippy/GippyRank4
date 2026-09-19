"""Build and validate the frozen Context 1.3 candidate.

This command writes only to the explicit ``context_v1_3_candidate`` namespace.
It never mutates the active Context 1.2 artifacts, including the 2026 annual
publication.  Historical scoring uses the checked-in retrospective research
feature panel; future inference accepts only a validated #114 production
transfer-feature artifact and its immutable on-time snapshot manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_preseason_context_prior_v1_2 as c12
import build_preseason_prior as v1
import build_preseason_prior_v1_1 as h11

from gippyrank.context_prior import InferenceRow
from gippyrank.context_prior_v1_3 import (
    ACTIVE_CONTEXT_PRIOR_VERSION,
    CONTEXT_1_3_FEATURES,
    CONTEXT_PRIOR_CANDIDATE_VERSION,
    D5_CONTEXT_FEATURES,
    LOCATION_FEATURE_NAMES,
    MODEL_FEATURE_NAMES,
    SCALE_FEATURE_NAMES,
    attach_transfer_features,
    attach_transfer_features_to_inference_rows,
    candidate_guard,
    fit_model,
    load_validated_production_transfer_features,
    model_specification_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
PRESEASON = ROOT / "data/processed/preseason"
OUTPUT = PRESEASON / "context_v1_3_candidate"
HISTORICAL_TRANSFER_FEATURES = OUTPUT / "historical_transfer_features.csv"
MODELING = ROOT / "data/processed/modeling"
RESEARCH_P3 = ROOT / "data/processed/defensive_transfer_position_groups"
RESEARCH_D5 = ROOT / "data/processed/transfer_signal_decomposition"
TEST_SEASONS = (2022, 2023, 2024, 2025)
FROZEN_TRAIN_THROUGH = 2021
METRICS = (
    "nll",
    "crps",
    "expected_rank_mae",
    "median_rank_mae",
    "interval_80_coverage",
    "interval_80_average_width",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_reference(path: Path) -> str:
    """Keep committed provenance portable across local checkout paths."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return "external artifact (not committed)"


def _keys(rows: Iterable[object]) -> set[tuple[int, str, str]]:
    return {
        (int(row.season), str(row.subdivision), str(row.team_id))  # type: ignore[union-attr]
        for row in rows
    }


def base_context_rows(
    *, max_season: int = max(TEST_SEASONS)
) -> tuple[list, list, list[dict[str, object]]]:
    """Load outcome-bearing historical rows with only approved base inputs."""
    rows, cold, _ = v1.load_rows(max_season=max_season)
    fbs = [row for row in rows if row.subdivision == "fbs"]
    contextual, coverage = c12.attach_context(
        fbs, c12.feature_index(), c12.cached_tenures()
    )
    stripped = [
        replace(
            row,
            features={name: row.features.get(name) for name in MODEL_FEATURE_NAMES},
        )
        for row in contextual
    ]
    return stripped, cold, coverage


def load_candidate_rows(
    feature_path: Path = HISTORICAL_TRANSFER_FEATURES,
) -> tuple[list, list, list[dict[str, object]]]:
    rows, cold, coverage = base_context_rows()
    attached = attach_transfer_features(rows, read_csv(feature_path), require_all=True)
    return attached, cold, coverage


def prediction_rows(model, rows: list, label: str):
    return h11.make_predictions(model, rows, label)


def merge_modeled_and_fallback(modeled, fallback, label: str):
    modeled_keys = {item.key for item in modeled}
    result = [*modeled]
    result.extend(
        replace(item, model=label)
        for item in fallback
        if item.key not in modeled_keys
    )
    return sorted(result, key=lambda item: item.key)


def score(predictions: list) -> dict[str, float]:
    raw = v1.score_predictions(predictions)
    return {metric: float(raw[metric]) for metric in METRICS}


def score_by_season(predictions: list) -> dict[str, dict[str, float]]:
    return {
        str(season): score([item for item in predictions if item.season == season])
        for season in sorted({item.season for item in predictions})
    }


def parse_prediction_artifact(path: Path) -> list:
    result = []
    for row in read_csv(path):
        pmf = np.asarray(json.loads(row["pmf"]), dtype=float)
        result.append(
            v1.PriorPrediction(
                int(row["season"]),
                row["subdivision"],
                row["team_id"],
                row["team_name"],
                len(pmf),
                np.asarray([], dtype=int),
                row.get("model", row.get("model_family", "artifact")),
                row.get("prior_method", "same_subdivision_lag1"),
                pmf,
                float(row["conditional_location_mean"])
                if row.get("conditional_location_mean")
                else None,
                float(row["predictive_scale"]) if row.get("predictive_scale") else None,
            )
        )
    return result


def fallback_predictions(rows: list, cold: list) -> list:
    fallback = h11.join_targets(h11.read_predictions(c12.H_MODEL_NAME), rows, cold)
    return [item for item in fallback if item.season in TEST_SEASONS]


def fit_panel(
    rows: list,
    cold: list,
    *,
    context_features: Sequence[str],
    label: str,
    target_seasons: Iterable[int] = TEST_SEASONS,
    trained_through: int = FROZEN_TRAIN_THROUGH,
) -> tuple[list, object]:
    target_set = set(target_seasons)
    model, _ = fit_model(
        rows,
        target_season=min(target_set),
        trained_through_season=trained_through,
        context_features=context_features,
    )
    target = [row for row in rows if row.season in target_set]
    modeled = prediction_rows(model, target, label)
    return merge_modeled_and_fallback(modeled, fallback_predictions(rows, cold), label), model


def fit_rolling(
    rows: list,
    cold: list,
    *,
    context_features: Sequence[str],
    label: str,
    target_season: int,
) -> tuple[list, object]:
    model, _ = fit_model(
        rows,
        target_season=target_season,
        trained_through_season=target_season - 1,
        context_features=context_features,
    )
    target = [row for row in rows if row.season == target_season]
    modeled = prediction_rows(model, target, label)
    fallback = [
        item for item in fallback_predictions(rows, cold) if item.season == target_season
    ]
    return merge_modeled_and_fallback(modeled, fallback, label), model


def load_expected_rows(
    path: Path, candidate: str, *, protocol: str | None, season: str
):
    for row in read_csv(path):
        if (
            row.get("candidate") == candidate
            and (protocol is None or row.get("protocol") == protocol)
            and str(row.get("target_season")) == season
        ):
            return row
    raise ValueError(f"missing frozen research row: {path} {protocol}/{candidate}/{season}")


def metric_parity(
    actual: Mapping[str, float],
    expected: Mapping[str, object],
    *,
    tolerance: float = 1e-9,
) -> dict[str, object]:
    comparisons = []
    for metric in METRICS:
        observed = float(actual[metric])
        reference = float(expected[metric])
        comparisons.append(
            {
                "metric": metric,
                "actual": observed,
                "expected": reference,
                "absolute_error": abs(observed - reference),
                "tolerance": tolerance,
                "passed": abs(observed - reference) <= tolerance,
            }
        )
    return {
        "passed": all(row["passed"] for row in comparisons),
        "comparisons": comparisons,
    }


def assert_parity(name: str, result: Mapping[str, object]) -> None:
    if not result["passed"]:
        raise ValueError(f"{name} frozen research parity failed: {result}")


def coefficient_parity(model, expected_path: Path) -> dict[str, object]:
    expected = [
        row
        for row in read_csv(expected_path)
        if row.get("protocol") == "frozen_through_2021"
        and row.get("candidate") == "P3"
        and row.get("target_season") == "aggregate"
    ]
    comparisons = []
    for row in expected:
        feature = row["feature"]
        feature_index = model.feature_names.index(feature)
        numeric_index = model.lag_count + 1 + feature_index
        missingness_index = (
            model.lag_count + 1 + len(model.feature_names) + feature_index
        )
        observed = {
            "standardized_location_coefficient": float(model.beta[numeric_index]),
            "location_missingness_coefficient": float(model.beta[missingness_index]),
            "standardized_scale_coefficient": float(model.gamma[1 + feature_index]),
        }
        for metric, value in observed.items():
            reference = float(row[metric])
            comparisons.append(
                {
                    "feature": feature,
                    "metric": metric,
                    "actual": value,
                    "expected": reference,
                    "absolute_error": abs(value - reference),
                    "tolerance": 1e-9,
                    "passed": abs(value - reference) <= 1e-9,
                }
            )
    return {
        "passed": all(row["passed"] for row in comparisons),
        "comparisons": comparisons,
    }


def population_parity(predictions: list, expected_path: Path) -> dict[str, object]:
    expected = {
        (int(row["season"]), row["subdivision"], row["team_id"])
        for row in read_csv(expected_path)
        if row.get("comparison") == "P3_vs_P0"
        and row.get("protocol") == "frozen_through_2021"
    }
    actual = {item.key for item in predictions}
    return {
        "passed": actual == expected,
        "actual_count": len(actual),
        "expected_count": len(expected),
        "missing": [list(key) for key in sorted(expected - actual)],
        "extra": [list(key) for key in sorted(actual - expected)],
    }


def fallback_parity(rows: list, cold: list, predictions: list) -> dict[str, object]:
    expected = {
        (item.season, item.subdivision, item.team_id)
        for item in fallback_predictions(rows, cold)
        if item.prior_method != "same_subdivision_lag1"
    }
    # The modeled panel and H fallback use the same target universe.  This
    # records the actual cold-start keys explicitly even when the historical
    # 2022--2025 panel has no FBS cold starts.
    actual = {
        (item.season, item.subdivision, item.team_id)
        for item in predictions
        if item.model == "P3" and item.prior_method != "same_subdivision_lag1"
    }
    return {
        "passed": actual == expected,
        "fallback_keys": [list(key) for key in sorted(actual)],
        "expected_fallback_keys": [list(key) for key in sorted(expected)],
    }


def prediction_artifact_rows(predictions: list) -> list[dict[str, object]]:
    return [
        {
            **prediction.csv_row(),
            "model_family": "context_prior",
            "spec_version": CONTEXT_PRIOR_CANDIDATE_VERSION,
            "candidate_status": "implemented_not_active",
            "context_snapshot_mode": "retrospective_reconstruction",
        }
        for prediction in predictions
    ]


def coverage_rows(rows: list, coverage: list[dict[str, object]]) -> list[dict[str, object]]:
    source = {
        (int(item["season"]), str(item["subdivision"]), str(item["team_id"])): item
        for item in coverage
    }
    result = []
    for row in rows:
        key = (row.season, row.subdivision, row.team_id)
        result.append(
            {
                "season": row.season,
                "subdivision": row.subdivision,
                "team_id": row.team_id,
                "team_name": row.team_name,
                "context_effective_cutoff": f"{row.season}-08-15",
                "coach_tenure_available": source.get(key, {}).get(
                    "coach_tenure_available"
                ),
                **{
                    f"{name}_available": row.features.get(name) is not None
                    for name in CONTEXT_1_3_FEATURES
                },
                "transfer_provenance_class": "retrospective_research_reconstruction",
            }
        )
    return result


def feature_provenance() -> dict[str, object]:
    return {
        "rank_history": {
            "production_status": "production-safe-by-construction",
            "source": "processed Massey final constituent-rank distributions",
            "historical_coverage": "2003-2025",
        },
        "coaching": {
            "production_status": "production-safe-by-construction",
            "source": "CFBD /coaches/tenures",
            "feature": "coach_tenure_seasons",
            "cutoff": "target-season August 15",
        },
        "recruiting": {
            "production_status": "production-safe-by-semantics",
            "source": "CFBD /recruiting/teams",
            "features": list(c12.RECRUITING_FEATURES),
            "caveat": "historical values are retrospective endpoint reconstructions, not archived snapshots",
        },
        "team_talent": {
            "production_status": "production-safe-with-retrospective-stability-caveat",
            "source": "CFBD /talent (247Sports Team Talent Composite)",
            "feature": "talent_composite",
        },
        "returning_production": {
            "production_status": "production-safe-with-retrospective-stability-caveat",
            "source": "CFBD /player/returning",
            "feature": "returning_pct_ppa",
            "excluded_features": [
                "returning_pct_passing_ppa",
                "returning_pct_receiving_ppa",
                "returning_pct_rushing_ppa",
            ],
        },
        "incoming_prior_offensive_usage": {
            "production_status": "production-safe-after-immutable-snapshot-validation",
            "source": "production preseason transfer-feature artifact derived from frozen portal + prior usage snapshots",
            "feature": "transfer_in_prior_usage_sum",
            "historical_evaluation_caveat": "retrospective research reconstruction for seasons before the production snapshot process existed",
        },
        "incoming_db_defensive_impact": {
            "production_status": "production-safe-after-immutable-snapshot-validation",
            "source": "production preseason transfer-feature artifact derived from frozen portal, prior roster, and defensive game-player snapshots",
            "feature": "transfer_in_prior_defensive_impact_db_sum",
            "definition": "sum of prior DB defensive impact for incoming DB transfers; each player impact is the frozen equal-weight mean of within-season x DB-group standardized log1p tackles, passes defended, and interceptions",
            "historical_evaluation_caveat": "retrospective research reconstruction for seasons before the production snapshot process existed",
        },
        "db_availability": {
            "production_status": "production-safe-after-immutable-snapshot-validation",
            "feature": "transfer_in_prior_defensive_impact_db_available",
            "semantics": {
                "no_incoming_db_transfers": "impact=0, availability=1",
                "all_incoming_db_transfers_resolved": "summed impact, availability=1",
                "one_or_more_unresolved": "neutral impact=0, availability=0",
            },
        },
    }


def build_annual_inference_rows(
    *,
    target_season: int,
    trained_through_season: int,
    team_feature_path: Path,
    transfer_feature_path: Path,
    transfer_manifest_path: Path,
    provenance: Mapping[str, object],
) -> tuple[list[InferenceRow], dict[str, object]]:
    """Build future rows only after the production transfer chain is valid."""
    candidate_guard(target_season)
    if trained_through_season != target_season - 1:
        raise ValueError("annual Context 1.3 inference requires target - 1 training")
    team_rows = read_csv(team_feature_path)
    index = {
        (int(row["season"]), row["subdivision"], row["team_id"]): row
        for row in team_rows
    }
    expected = {
        key for key in index if key[0] == target_season and key[1] == "fbs"
    }
    feature_rows, metadata = load_validated_production_transfer_features(
        transfer_feature_path,
        transfer_manifest_path,
        target_season=target_season,
        expected_team_keys=expected,
        provenance=provenance,
    )
    base = c12.inference_rows(
        target_season,
        trained_through_season,
        index,
        c12.cached_tenures(),
    )
    base = [
        replace(
            row,
            features={name: row.features.get(name) for name in MODEL_FEATURE_NAMES},
        )
        for row in base
    ]
    attached = attach_transfer_features_to_inference_rows(base, feature_rows)
    return attached, metadata


def build_report(
    *,
    rows: list,
    predictions: list,
    d5_predictions: list,
    p3_model,
    d5_model,
    c12_predictions: list,
    d5_parity: dict[str, object],
    p3_parity: dict[str, object],
    coefficient_result: dict[str, object],
    population_result: dict[str, object],
    fallback_result: dict[str, object],
    rolling_result: dict[str, object],
    coverage: list[dict[str, object]],
) -> dict[str, object]:
    p3_score = score(predictions)
    d5_score = score(d5_predictions)
    c12_score = score(c12_predictions)
    annual = {
        "context_1_2": score_by_season(c12_predictions),
        "D5": score_by_season(d5_predictions),
        "P3": score_by_season(predictions),
    }
    return {
        "candidate": {
            "model_family": "context_prior",
            "spec_version": CONTEXT_PRIOR_CANDIDATE_VERSION,
            "status": "implemented_and_validated_not_active",
            "active_production_context_version": ACTIVE_CONTEXT_PRIOR_VERSION,
        },
        "feature_contract": {
            "features": list(MODEL_FEATURE_NAMES),
            "context_features": list(CONTEXT_1_3_FEATURES),
            "location_features": list(LOCATION_FEATURE_NAMES),
            "scale_features": list(SCALE_FEATURE_NAMES),
            "removed_context_1_2_features": [
                "returning_pct_passing_ppa",
                "returning_pct_receiving_ppa",
                "returning_pct_rushing_ppa",
            ],
        },
        "historical_evaluation": {
            "trained_through_season": FROZEN_TRAIN_THROUGH,
            "target_seasons": list(TEST_SEASONS),
            "n_team_seasons": len(predictions),
            "target_outcomes_used_for_features": False,
            "transfer_provenance": "retrospective_research_reconstruction",
        },
        "aggregate_metrics": {
            "context_1_2": c12_score,
            "D5": d5_score,
            "P3": p3_score,
            "D5_incremental_over_context_1_2": {
                metric: d5_score[metric] - c12_score[metric] for metric in METRICS
            },
            "P3_incremental_over_D5": {
                metric: p3_score[metric] - d5_score[metric] for metric in METRICS
            },
        },
        "annual_metrics": annual,
        "parity": {
            "D5": d5_parity,
            "P3": p3_parity,
            "P3_coefficients": coefficient_result,
            "target_population": population_result,
            "H_fallback": fallback_result,
            "rolling_origin": rolling_result,
        },
        "model_parameters": {
            "distribution": "normal",
            "penalty": 0.25,
            "quadrature": "existing t-1 empirical quadrature; t-2/t-3 history summaries",
            "p3_location_feature_names": list(p3_model.location_feature_names or []),
            "p3_scale_feature_names": list(p3_model.scale_feature_names or []),
            "d5_location_feature_names": list(d5_model.location_feature_names or []),
            "d5_scale_feature_names": list(d5_model.scale_feature_names or []),
        },
        "production_activation": {
            "ready_for_first_future_season_with_valid_on_time_snapshot": bool(
                d5_parity["passed"]
                and p3_parity["passed"]
                and coefficient_result["passed"]
                and population_result["passed"]
                and fallback_result["passed"]
                and rolling_result["passed"]
            ),
            "required_input": "validated immutable production transfer-feature artifact",
            "missing_or_late_behavior": "fail closed",
            "2026_guardrail": "Context 1.2 remains the only valid 2026 publication lineage",
        },
        "coverage": {
            "n_context_rows": len(coverage),
            "n_context_rows_with_missing_transfer_usage": sum(
                row["transfer_in_prior_usage_sum_available"] is False for row in coverage
            ),
        },
    }


def render_report(report: Mapping[str, object], path: Path) -> None:
    aggregate = report["aggregate_metrics"]
    parity = report["parity"]
    lines = [
        "# Context 1.3 candidate validation",
        "",
        "Context 1.3 is implemented and validated as a future-activation candidate. The active production Context identifier remains 1.2.",
        "",
        "## Frozen contract",
        "",
        "- Location: History plus the exact Context 1.3 feature list.",
        "- Scale: History features only.",
        "- Distribution: Normal; penalty 0.25; existing deterministic optimizer retry.",
        "- Incoming transfer values: only the three model-facing columns from the #114 attach-only boundary.",
        "",
        "## Aggregate 2022–2025 metrics",
        "",
        "| Representation | NLL | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("context_1_2", "D5", "P3"):
        metrics = aggregate[label]
        lines.append(
            f"| {label} | {metrics['nll']:.9f} | {metrics['crps']:.9f} | {metrics['expected_rank_mae']:.6f} | {metrics['median_rank_mae']:.6f} | {metrics['interval_80_coverage']:.9f} | {metrics['interval_80_average_width']:.6f} |"
        )
    lines += [
        "",
        "## Frozen parity",
        "",
        f"- D5 parity: {'passed' if parity['D5']['passed'] else 'failed'}.",
        f"- P3 parity: {'passed' if parity['P3']['passed'] else 'failed'}.",
        f"- P3 standardized coefficients: {'passed' if parity['P3_coefficients']['passed'] else 'failed'}.",
        f"- Historical target population: {'unchanged' if parity['target_population']['passed'] else 'changed'}.",
        f"- H fallback population: {'unchanged' if parity['H_fallback']['passed'] else 'changed'}.",
        f"- Rolling-origin P3 parity: {'passed' if parity['rolling_origin']['passed'] else 'failed'}.",
        "",
        "## Provenance and activation guardrails",
        "",
        "Historical 2021–2025 transfer values are retrospective research reconstructions, not archived August 15 snapshots. Future annual inference requires a validated immutable manifest, complete canonical FBS rows, an on-time cutoff, and explicit provenance metadata; absent or late inputs fail closed.",
        "",
        "The 2026 Context 1.2 annual artifact and its weekly publication lineage are not overwritten or reinterpreted by this candidate.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(*, output: Path = OUTPUT, feature_path: Path = HISTORICAL_TRANSFER_FEATURES) -> dict[str, object]:
    output = output.resolve()
    feature_path = feature_path.resolve()
    rows, cold, coverage = load_candidate_rows(feature_path)
    p3_predictions, p3_model = fit_panel(
        rows,
        cold,
        context_features=CONTEXT_1_3_FEATURES,
        label="P3",
    )
    d5_predictions, d5_model = fit_panel(
        rows,
        cold,
        context_features=D5_CONTEXT_FEATURES,
        label="D5",
    )
    c12_predictions = merge_modeled_and_fallback(
        parse_prediction_artifact(PRESEASON / "context/predictions.csv"),
        [],
        "context_1_2",
    )
    c12_predictions = h11.join_targets(c12_predictions, rows, cold)

    p3_expected = load_expected_rows(
        RESEARCH_P3 / "candidate_summary.csv", "P3", protocol="frozen_through_2021", season="aggregate"
    )
    d5_expected = load_expected_rows(
        RESEARCH_D5 / "candidate_summary.csv",
        "D5_total_rp_plus_incoming",
        protocol=None,
        season="aggregate",
    )
    p3_parity = metric_parity(score(p3_predictions), p3_expected)
    d5_parity = metric_parity(score(d5_predictions), d5_expected)
    assert_parity("P3", p3_parity)
    assert_parity("D5", d5_parity)
    coefficient_result = coefficient_parity(p3_model, RESEARCH_P3 / "coefficients.csv")
    assert_parity("P3 coefficients", coefficient_result)
    population_result = population_parity(
        p3_predictions, RESEARCH_P3 / "paired_nll_by_team.csv"
    )
    assert_parity("target population", population_result)
    fallback_result = fallback_parity(rows, cold, p3_predictions)
    assert_parity("H fallback", fallback_result)

    rolling_rows: list[dict[str, object]] = []
    rolling_parity_rows = []
    for season in TEST_SEASONS:
        p3_rolling, _ = fit_rolling(
            rows,
            cold,
            context_features=CONTEXT_1_3_FEATURES,
            label="P3",
            target_season=season,
        )
        d5_rolling, _ = fit_rolling(
            rows,
            cold,
            context_features=D5_CONTEXT_FEATURES,
            label="D5",
            target_season=season,
        )
        rolling_rows.extend(
            [
                {"candidate": "P3", "target_season": season, **score(p3_rolling)},
                {"candidate": "D5", "target_season": season, **score(d5_rolling)},
            ]
        )
        expected = load_expected_rows(
            RESEARCH_P3 / "rolling_metrics.csv",
            "P3",
            protocol="rolling_origin",
            season=str(season),
        )
        parity = metric_parity(score(p3_rolling), expected)
        rolling_parity_rows.append(parity)
    rolling_result = {
        "passed": all(row["passed"] for row in rolling_parity_rows),
        "seasons": {
            str(season): row for season, row in zip(TEST_SEASONS, rolling_parity_rows, strict=True)
        },
    }
    assert_parity("rolling P3", rolling_result)

    report = build_report(
        rows=rows,
        predictions=p3_predictions,
        d5_predictions=d5_predictions,
        p3_model=p3_model,
        d5_model=d5_model,
        c12_predictions=c12_predictions,
        d5_parity=d5_parity,
        p3_parity=p3_parity,
        coefficient_result=coefficient_result,
        population_result=population_result,
        fallback_result=fallback_result,
        rolling_result=rolling_result,
        coverage=coverage_rows(rows, coverage),
    )
    report["source_artifacts"] = {
        "historical_transfer_features": artifact_reference(feature_path),
        "historical_transfer_features_sha256": sha256_file(feature_path),
        "context_1_2_predictions": artifact_reference(
            PRESEASON / "context/predictions.csv"
        ),
        "context_1_2_predictions_sha256": sha256_file(
            PRESEASON / "context/predictions.csv"
        ),
        "p3_research_directory": artifact_reference(RESEARCH_P3),
        "d5_research_directory": artifact_reference(RESEARCH_D5),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "model_spec.json", model_specification_metadata())
    write_json(output / "feature_provenance.json", feature_provenance())
    write_json(output / "evaluation.json", {
        "aggregate": report["aggregate_metrics"],
        "annual": report["annual_metrics"],
    })
    write_json(output / "parity_report.json", report["parity"])
    write_json(output / "model_report.json", report)
    write_json(output / "fitted_model.json", {
        "candidate": "P3_D5_plus_db",
        "spec_version": CONTEXT_PRIOR_CANDIDATE_VERSION,
        "model": p3_model.metadata(),
    })
    write_csv(output / "predictions.csv", prediction_artifact_rows(p3_predictions))
    write_csv(output / "context_coverage.csv", coverage_rows(rows, coverage))
    write_csv(output / "rolling_metrics.csv", rolling_rows)
    coefficient_rows = []
    for feature in CONTEXT_1_3_FEATURES:
        feature_index = p3_model.feature_names.index(feature)
        numeric_index = p3_model.lag_count + 1 + feature_index
        missingness_index = p3_model.lag_count + 1 + len(p3_model.feature_names) + feature_index
        coefficient_rows.append(
            {
                "candidate": "P3",
                "feature": feature,
                "standardized_location_coefficient": p3_model.beta[numeric_index],
                "location_missingness_coefficient": p3_model.beta[missingness_index],
                "standardized_scale_coefficient": p3_model.gamma[1 + feature_index],
            }
        )
    write_csv(output / "coefficients.csv", coefficient_rows)
    render_report(report, output / "candidate_report.md")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--feature-path", type=Path, default=HISTORICAL_TRANSFER_FEATURES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(output=args.output, feature_path=args.feature_path)
    print(json.dumps(report["production_activation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
