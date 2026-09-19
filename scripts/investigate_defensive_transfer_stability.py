"""Rolling-origin stability study for imported defensive experience.

This research-only runner implements issue 109.  It evaluates exactly three
predeclared Context variants:

* R0: production Context C 1.2 (C0);
* R1: D5, the frozen total-returning-production plus incoming-prior-usage
  representation; and
* R2: D5 plus the frozen imported defensive-experience proxy and its explicit
  source-availability indicator.

The runner reuses the existing Context optimizer, penalty, preprocessing,
rank-distribution scoring, cold-start H fallback, and August 15 transfer
cutoff.  It writes research artifacts only; production model artifacts are
never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_preseason_context_prior_v1_2 as c12
import build_preseason_prior_v1_1 as v1_1
import investigate_defensive_transfer_predictive_value as defensive
import investigate_transfer_roster_continuity as prior

from gippyrank.preseason import DirectRankModel, TeamSeason

ROOT = Path(__file__).resolve().parents[1]
TARGET_SEASONS = (2022, 2023, 2024, 2025)
FROZEN_TRAIN_THROUGH = 2021
DEFAULT_CUTOFF = (8, 15)
C0_PARITY_TOLERANCE = 1e-3
FROZEN_PARITY_TOLERANCE = 1e-6
RANDOM_SEED = 7

H_FEATURES = tuple(c12.H_FEATURES)
COACH_FEATURES = tuple(c12.COACH_FEATURES)
RECRUITING_FEATURES = tuple(c12.RECRUITING_FEATURES)
TALENT_FEATURES = tuple(c12.TALENT_FEATURES)
RP_FEATURES = tuple(c12.RETURNING_FEATURES)
BASE_CONTEXT_FEATURES = (
    *COACH_FEATURES,
    *RECRUITING_FEATURES,
    *TALENT_FEATURES,
    *RP_FEATURES,
)
C_MINUS_RP_FEATURES = tuple(
    feature for feature in BASE_CONTEXT_FEATURES if feature not in RP_FEATURES
)
RETURNING_PPA = "returning_pct_ppa"
INCOMING_USAGE = "transfer_in_prior_usage_sum"
EXPERIENCE = defensive.EXPERIENCE
EXPERIENCE_AVAILABLE = defensive.EXPERIENCE_AVAILABLE
D5_CONTINUITY_FEATURES = (RETURNING_PPA, INCOMING_USAGE)
COEFFICIENT_FEATURES = (
    RETURNING_PPA,
    INCOMING_USAGE,
    EXPERIENCE,
    EXPERIENCE_AVAILABLE,
)
METRICS = tuple(prior.METRICS)

RECOMMENDATION_ADVANCE = "advance D5 + defensive experience"
RECOMMENDATION_RETAIN = "retain D5 and stop defensive feature research"


@dataclass(frozen=True)
class Candidate:
    """One of the three predeclared issue-109 candidates."""

    label: str
    name: str
    description: str
    features: tuple[str, ...]


def d5_features() -> tuple[str, ...]:
    """Return the exact D5 representation frozen by issue 96."""
    return (*C_MINUS_RP_FEATURES, *D5_CONTINUITY_FEATURES)


def candidate_definitions() -> tuple[Candidate, ...]:
    """Return exactly R0, R1, and R2; rolling results never add candidates."""
    d5 = d5_features()
    return (
        Candidate(
            "R0",
            "production_C0",
            "Production Context C 1.2 (C0).",
            BASE_CONTEXT_FEATURES,
        ),
        Candidate(
            "R1",
            "D5",
            "C-minus-RP baseline plus total RP and incoming prior offensive usage.",
            d5,
        ),
        Candidate(
            "R2",
            "D5_plus_defensive_experience",
            "D5 plus imported prior defensive experience and its availability indicator.",
            (*d5, EXPERIENCE, EXPERIENCE_AVAILABLE),
        ),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a non-empty tabular research artifact."""
    if not rows:
        raise ValueError(f"refusing to write empty artifact: {path}")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transfer_source_hashes(transfer_root: Path) -> dict[str, str]:
    """Hash immutable transfer payloads without storing machine paths."""
    result: dict[str, str] = {}
    for directory in ("portal", "usage"):
        for path in sorted((transfer_root / directory).glob("*.json")):
            if path.name == "manifest.json" or path.name.endswith(".provenance.json"):
                continue
            result[str(path.relative_to(transfer_root))] = sha256_file(path)
    return result


def configure_source_root(source_root: Path) -> None:
    """Point the existing Context and H loaders at an explicit input root."""
    prior.configure_source_root(source_root)


def fit_context(
    rows: list[TeamSeason],
    features: Iterable[str],
    *,
    target_season: int,
) -> DirectRankModel:
    """Fit Context C 1.2 on seasons strictly before the target season."""
    training = [row for row in rows if row.season < target_season]
    if not training:
        raise ValueError("rolling fit has no training rows")
    feature_list = list(features)
    try:
        return DirectRankModel.fit(
            training,
            [*H_FEATURES, *feature_list],
            penalty=0.25,
            location_feature_names=[*H_FEATURES, *feature_list],
            scale_feature_names=list(H_FEATURES),
        )
    except RuntimeError as error:
        if "ITERATIONS REACHED LIMIT" not in str(error):
            raise
        return DirectRankModel.fit(
            training,
            [*H_FEATURES, *feature_list],
            penalty=0.25,
            location_feature_names=[*H_FEATURES, *feature_list],
            scale_feature_names=list(H_FEATURES),
            optimizer_options={"maxiter": 2000},
        )


def score(predictions: list[v1_1.v1.PriorPrediction]) -> dict[str, float]:
    values = v1_1.v1.score_predictions(predictions)
    return {metric: float(values[metric]) for metric in METRICS}


def coefficient_index(model: DirectRankModel, feature: str) -> tuple[int, int]:
    """Return numeric and explicit-missingness beta indexes for a feature."""
    if feature not in model.feature_names:
        raise ValueError(f"feature is not in model: {feature}")
    feature_index = model.feature_names.index(feature)
    numeric = model.lag_count + 1 + feature_index
    missingness = model.lag_count + 1 + len(model.feature_names) + feature_index
    return numeric, missingness


def sign(value: float, tolerance: float = 1e-10) -> str:
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "zero"


def _observed_source(row: TeamSeason, feature: str) -> bool:
    if feature in {EXPERIENCE, EXPERIENCE_AVAILABLE}:
        return row.features.get(EXPERIENCE_AVAILABLE) == 1.0
    return row.features.get(feature) is not None


def training_coverage(
    rows: list[TeamSeason],
    portal_seasons: set[int],
    *,
    target_season: int,
) -> dict[str, object]:
    """Count raw feature availability before model preprocessing/imputation."""
    training = [row for row in rows if row.season < target_season]
    n_rows = len(training)
    observed_usage = sum(_observed_source(row, INCOMING_USAGE) for row in training)
    observed_defensive = sum(_observed_source(row, EXPERIENCE) for row in training)
    unresolved_defensive = n_rows - observed_defensive
    covered = sorted({row.season for row in training if row.season in portal_seasons})
    return {
        "target_season": target_season,
        "train_through": target_season - 1,
        "n_training_team_seasons": n_rows,
        "n_training_seasons": len({row.season for row in training}),
        "n_transfer_covered_training_seasons": len(covered),
        "transfer_covered_training_seasons": covered,
        "n_training_rows_with_observed_incoming_offensive_usage": observed_usage,
        "fraction_with_observed_incoming_offensive_usage": (
            observed_usage / n_rows if n_rows else None
        ),
        "n_training_rows_with_defensive_experience_source_available": observed_defensive,
        "fraction_with_defensive_experience_source_available": (
            observed_defensive / n_rows if n_rows else None
        ),
        "n_training_rows_unresolved_for_defensive_experience": unresolved_defensive,
        # These aliases make the raw contracts easy to compare with the issue-97
        # stability artifact while retaining the issue-109 wording above.
        "n_team_seasons_with_transfer_in_prior_usage_sum": observed_usage,
        "fraction_with_all_observed_transfer_production": (
            observed_usage / n_rows if n_rows else None
        ),
        "n_team_seasons_with_defensive_experience_source_available": observed_defensive,
    }


def coefficient_rows(
    model: DirectRankModel,
    candidate: Candidate,
    *,
    target_season: int,
    previous: dict[tuple[str, str], float],
    training: list[TeamSeason] | None = None,
) -> list[dict[str, object]]:
    """Extract the four requested R2 standardized location coefficients."""
    if candidate.label != "R2":
        return []
    training_rows = training or []
    result: list[dict[str, object]] = []
    for feature in COEFFICIENT_FEATURES:
        numeric_index, missingness_index = coefficient_index(model, feature)
        feature_index = model.feature_names.index(feature)
        value = float(model.beta[numeric_index])
        key = (candidate.label, feature)
        prior_value = previous.get(key)
        observed = sum(_observed_source(row, feature) for row in training_rows)
        row: dict[str, object] = {
            "target_season": target_season,
            "train_through": target_season - 1,
            "candidate": candidate.label,
            "candidate_name": candidate.name,
            "feature": feature,
            "standardized_location_coefficient": value,
            "sign": sign(value),
            "magnitude": abs(value),
            "change_from_prior_fit": (
                value - prior_value if prior_value is not None else None
            ),
            "location_missingness_indicator_coefficient": float(
                model.beta[missingness_index]
            ),
            "standardized_scale_coefficient": float(model.gamma[1 + feature_index]),
            "training_source_available_n": observed,
            "training_source_unresolved_n": len(training_rows) - observed,
            "training_source_available_fraction": (
                observed / len(training_rows) if training_rows else None
            ),
        }
        result.append(row)
        previous[key] = value
    return result


def coefficient_stability(coefficients: list[dict[str, object]]) -> dict[str, object]:
    """Summarize sign, magnitude, and fit-to-fit coefficient stability."""
    result: dict[str, object] = {}
    for feature in COEFFICIENT_FEATURES:
        rows = [row for row in coefficients if row["feature"] == feature]
        values = [float(row["standardized_location_coefficient"]) for row in rows]
        signs = [str(row["sign"]) for row in rows]
        changes = [
            abs(float(row["change_from_prior_fit"]))
            for row in rows
            if row["change_from_prior_fit"] is not None
        ]
        result[feature] = {
            "signs": signs,
            "sign_flip_count": sum(
                left != right
                for left, right in pairwise(signs)
                if left != "zero" and right != "zero"
            ),
            "coefficient_values": values,
            "min_coefficient": min(values) if values else None,
            "max_coefficient": max(values) if values else None,
            "max_absolute_coefficient": max(map(abs, values)) if values else None,
            "max_absolute_change": max(changes, default=None),
            "first_absolute_coefficient": abs(values[0]) if values else None,
            "last_absolute_coefficient": abs(values[-1]) if values else None,
        }
    return result


def availability_diagnostics(
    coefficients: list[dict[str, object]],
) -> dict[str, object]:
    """Compare the explicit availability path with the defensive value path."""
    value_rows = {
        int(row["target_season"]): row
        for row in coefficients
        if row["feature"] == EXPERIENCE
    }
    availability_rows = {
        int(row["target_season"]): row
        for row in coefficients
        if row["feature"] == EXPERIENCE_AVAILABLE
    }
    rows: list[dict[str, object]] = []
    for target in TARGET_SEASONS:
        value = value_rows[target]
        available = availability_rows[target]
        value_coefficient = float(value["standardized_location_coefficient"])
        availability_coefficient = float(available["standardized_location_coefficient"])
        rows.append(
            {
                "target_season": target,
                "defensive_value_coefficient": value_coefficient,
                "availability_coefficient": availability_coefficient,
                "absolute_value_coefficient": abs(value_coefficient),
                "absolute_availability_coefficient": abs(availability_coefficient),
                "availability_dominates": abs(availability_coefficient)
                > abs(value_coefficient),
            }
        )
    return {
        "rows": rows,
        "availability_dominates_n": sum(
            bool(row["availability_dominates"]) for row in rows
        ),
        "value_feature": EXPERIENCE,
        "availability_feature": EXPERIENCE_AVAILABLE,
    }


def metric_rows(
    predictions_by_candidate: dict[str, list[v1_1.v1.PriorPrediction]],
    *,
    target_season: int,
    train_through: int,
    protocol: str,
) -> list[dict[str, object]]:
    scores = {
        candidate: score(predictions)
        for candidate, predictions in predictions_by_candidate.items()
    }
    counts = {len(predictions) for predictions in predictions_by_candidate.values()}
    if len(counts) != 1:
        raise ValueError("candidate predictions have different target populations")
    n_team_seasons = counts.pop()
    result: list[dict[str, object]] = []
    for candidate in ("R0", "R1", "R2"):
        values = scores[candidate]
        row: dict[str, object] = {
            "protocol": protocol,
            "target_season": target_season,
            "train_through": train_through,
            "candidate": candidate,
            "n_team_seasons": n_team_seasons,
        }
        row.update(values)
        for reference in ("R0", "R1"):
            for metric in METRICS:
                row[f"delta_{metric}_vs_{reference}"] = (
                    values[metric] - scores[reference][metric]
                )
        result.append(row)
    return result


def aggregate_metrics(
    rows: list[dict[str, object]], *, candidate: str, protocol: str
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if row["candidate"] == candidate and row["protocol"] == protocol
    ]
    if not selected:
        raise ValueError(f"no metrics for {candidate}/{protocol}")
    weights = np.asarray([int(row["n_team_seasons"]) for row in selected])
    result: dict[str, object] = {
        "protocol": protocol,
        "candidate": candidate,
        "target_seasons": ";".join(str(row["target_season"]) for row in selected),
        "n_target_seasons": len(selected),
        "n_team_seasons": int(weights.sum()),
    }
    for metric in METRICS:
        result[metric] = float(
            np.average(
                np.asarray([float(row[metric]) for row in selected]), weights=weights
            )
        )
    return result


def comparison_rows(metric_rows_: list[dict[str, object]]) -> list[dict[str, object]]:
    """Materialize R1-R0, R2-R1, and R2-R0 for every target season."""
    by_key = {
        (int(row["target_season"]), str(row["candidate"])): row
        for row in metric_rows_
        if row["protocol"] == "rolling_origin"
    }
    result: list[dict[str, object]] = []
    for target in TARGET_SEASONS:
        for left, right in (("R1", "R0"), ("R2", "R1"), ("R2", "R0")):
            left_row = by_key[(target, left)]
            right_row = by_key[(target, right)]
            row: dict[str, object] = {
                "target_season": target,
                "left_candidate": left,
                "right_candidate": right,
                "n_team_seasons": left_row["n_team_seasons"],
            }
            for metric in METRICS:
                row[f"delta_{metric}"] = float(left_row[metric]) - float(
                    right_row[metric]
                )
            result.append(row)
    return result


def paired_nll_diagnostics(
    candidate_name: str,
    candidate: list[v1_1.v1.PriorPrediction],
    reference_name: str,
    reference: list[v1_1.v1.PriorPrediction],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return paired R2-vs-R1 team-season losses and summaries."""
    candidate_losses = v1_1.prediction_losses(candidate)
    reference_losses = v1_1.prediction_losses(reference)
    if set(candidate_losses) != set(reference_losses):
        raise ValueError("paired candidates have different target keys")
    names = {item.key: item.team_name for item in reference}
    details = [
        {
            "target_season": key[0],
            "subdivision": key[1],
            "team_id": key[2],
            "team_name": names[key],
            "candidate": candidate_name,
            "reference": reference_name,
            "candidate_nll": candidate_losses[key][0],
            "reference_nll": reference_losses[key][0],
            "delta_nll": candidate_losses[key][0] - reference_losses[key][0],
        }
        for key in sorted(candidate_losses)
    ]

    def summarize(scope: str, values: list[float]) -> dict[str, object]:
        numbers = np.asarray(values, dtype=float)
        return {
            "scope": scope,
            "candidate": candidate_name,
            "reference": reference_name,
            "n_team_seasons": len(numbers),
            "mean_delta_nll": float(np.mean(numbers)),
            "median_delta_nll": float(np.median(numbers)),
            "fraction_team_seasons_improved": float(np.mean(numbers < 0)),
        }

    summaries = [
        summarize(
            "aggregate",
            [float(row["delta_nll"]) for row in details],
        )
    ]
    for season in TARGET_SEASONS:
        values = [
            float(row["delta_nll"])
            for row in details
            if int(row["target_season"]) == season
        ]
        if values:
            summaries.append(summarize(str(season), values))
    return details, summaries


def metric_parity(
    computed: dict[str, float],
    expected: dict[str, float],
    *,
    candidate: str,
    reference_path: Path,
    tolerance: float,
) -> dict[str, object]:
    """Compare every required metric and fail outside the declared tolerance."""
    deltas = {metric: abs(computed[metric] - expected[metric]) for metric in METRICS}
    maximum = max(deltas.values(), default=None)
    result: dict[str, object] = {
        "candidate": candidate,
        "reference_path": str(reference_path),
        "reference_metrics": expected,
        "computed_metrics": computed,
        "metric_abs_deltas": deltas,
        "max_abs_metric_delta": maximum,
        "tolerance": tolerance,
        "passed": maximum is not None and maximum <= tolerance,
    }
    if not result["passed"]:
        raise ValueError(f"parity failed for {candidate}: {result}")
    return result


def read_csv_metric_row(
    path: Path, *, field: str, value: str, target_season: str = "aggregate"
) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get(field) == value and row.get("target_season") == target_season:
                return {metric: float(row[metric]) for metric in METRICS}
    raise ValueError(f"missing {field}={value}/{target_season} in {path}")


def read_production_c0_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        row = payload["all_fbs"]["candidate"]
        return {metric: float(row[metric]) for metric in METRICS}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"production C0 metrics are missing from {path}") from error


def plot_outputs(
    output: Path,
    rolling_rows: list[dict[str, object]],
    coefficients: list[dict[str, object]],
    paired_summaries: list[dict[str, object]],
) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 5))
    for candidate in ("R0", "R1", "R2"):
        values = [
            float(row["nll"]) for row in rolling_rows if row["candidate"] == candidate
        ]
        axis.plot(TARGET_SEASONS, values, marker="o", label=candidate)
    axis.set_xlabel("target season")
    axis.set_ylabel("NLL")
    axis.set_title("Defensive-experience rolling-origin NLL")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "rolling_nll.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 6))
    for feature in COEFFICIENT_FEATURES:
        rows = [row for row in coefficients if row["feature"] == feature]
        axis.plot(
            [int(row["target_season"]) for row in rows],
            [float(row["standardized_location_coefficient"]) for row in rows],
            marker="o",
            label=feature,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("target season")
    axis.set_ylabel("standardized location coefficient")
    axis.set_title("R2 defensive-experience coefficient paths")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots / "r2_coefficients.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    rows = [row for row in paired_summaries if row["scope"] != "aggregate"]
    values = [float(row["mean_delta_nll"]) for row in rows]
    labels = [str(row["scope"]) for row in rows]
    axis.bar(
        labels,
        values,
        color=["#2f855a" if value < 0 else "#c53030" for value in values],
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("target season")
    axis.set_ylabel("mean paired ΔNLL (R2 − R1)")
    axis.set_title("Paired defensive-experience losses")
    figure.tight_layout()
    figure.savefig(plots / "paired_delta_nll.png", dpi=160)
    plt.close(figure)


def fmt(value: object, digits: int = 4) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.{digits}f}"


def choose_recommendation(
    comparisons: list[dict[str, object]],
    coefficient_summary: dict[str, object],
    availability: dict[str, object],
    paired_summaries: list[dict[str, object]],
) -> tuple[str, dict[str, object]]:
    """Apply a deliberately strict, predeclared promotion screen.

    The frozen gain is small, so an ambiguous result stays with D5.  The
    screen requires R2 to beat R1 in every rolling target season, requires a
    consistently sensible negative defensive-value coefficient, and rejects
    a path where availability dominates in any fit.  These are deliberately
    stricter than merely winning the weighted aggregate.
    """
    r2_vs_r1 = [
        row
        for row in comparisons
        if row["left_candidate"] == "R2" and row["right_candidate"] == "R1"
    ]
    wins = sum(float(row["delta_nll"]) < 0 for row in r2_vs_r1)
    value_summary = coefficient_summary[EXPERIENCE]
    value_signs = list(value_summary["signs"])
    paired_aggregate = next(
        row for row in paired_summaries if row["scope"] == "aggregate"
    )
    criteria = {
        "r2_wins_all_target_seasons": wins == len(TARGET_SEASONS),
        "r2_wins_n": wins,
        "r2_target_seasons_n": len(TARGET_SEASONS),
        "rolling_aggregate_delta_nll_negative": bool(
            float(paired_aggregate["mean_delta_nll"]) < 0
        ),
        "defensive_value_signs_consistently_negative": value_signs
        and all(value == "negative" for value in value_signs),
        "defensive_value_sign_flip_count": value_summary["sign_flip_count"],
        "availability_never_dominates": availability["availability_dominates_n"] == 0,
        "paired_team_season_fraction_improved_at_least_half": float(
            paired_aggregate["fraction_team_seasons_improved"]
        )
        >= 0.5,
    }
    passed = all(
        value
        for key, value in criteria.items()
        if key
        not in {
            "r2_wins_n",
            "r2_target_seasons_n",
            "defensive_value_sign_flip_count",
        }
    )
    return (RECOMMENDATION_ADVANCE if passed else RECOMMENDATION_RETAIN), criteria


def _reference_metric_map(
    rows: list[dict[str, object]], candidate: str, protocol: str
) -> dict[str, float]:
    aggregate = aggregate_metrics(rows, candidate=candidate, protocol=protocol)
    return {metric: float(aggregate[metric]) for metric in METRICS}


def render_report(
    path: Path,
    *,
    summary: dict[str, object],
    rolling_rows: list[dict[str, object]],
    frozen_rows: list[dict[str, object]],
    rolling_aggregate: list[dict[str, object]],
    comparisons: list[dict[str, object]],
    coverage: list[dict[str, object]],
    coefficients: list[dict[str, object]],
    paired_summaries: list[dict[str, object]],
) -> None:
    parity = summary["frozen_parity"]
    rolling_by_candidate = {
        candidate: next(
            row for row in rolling_aggregate if row["candidate"] == candidate
        )
        for candidate in ("R0", "R1", "R2")
    }
    frozen_aggregate = {
        candidate: _reference_metric_map(frozen_rows, candidate, "frozen_through_2021")
        for candidate in ("R0", "R1", "R2")
    }
    r2_vs_r1 = [
        row
        for row in comparisons
        if row["left_candidate"] == "R2" and row["right_candidate"] == "R1"
    ]
    r2_wins = sum(float(row["delta_nll"]) < 0 for row in r2_vs_r1)
    availability = summary["availability_diagnostics"]
    defensive_stability = summary["coefficient_stability"][EXPERIENCE]

    lines = [
        "# Defensive-transfer experience rolling-origin stability (issue 109)",
        "",
        "## Conclusion",
        "",
        f"Final recommendation: {summary['recommendation']}",
        "",
        (
            f"R2 beats D5 (R1) by rolling-origin NLL in {r2_wins} / "
            f"{len(TARGET_SEASONS)} target seasons. The weighted rolling "
            f"ΔNLL (R2 − R1) is "
            f"{fmt(rolling_by_candidate['R2']['nll'] - rolling_by_candidate['R1']['nll'], 6)}."
        ),
        (
            f"The defensive-value coefficient signs are "
            f"{', '.join(defensive_stability['signs']) or 'unavailable'} with "
            f"{defensive_stability['sign_flip_count']} adjacent non-zero sign flips. "
            f"The availability indicator dominates the value coefficient in "
            f"{availability['availability_dominates_n']} / {len(TARGET_SEASONS)} fits."
        ),
        "",
        "The promotion screen is intentionally stricter than a favorable aggregate: because the frozen incremental gain is small, mixed season-level evidence remains ambiguous and defaults to D5.",
        "",
        "## Exact predeclared candidates",
        "",
        "| Candidate | Definition | Features added beyond C-minus-RP |",
        "|---|---|---|",
    ]
    for candidate in summary["candidates"]:
        added = [
            feature
            for feature in candidate["features"]
            if feature not in C_MINUS_RP_FEATURES
        ]
        lines.append(
            f"| {candidate['label']} | {candidate['description']} | "
            f"{', '.join(f'`{feature}`' for feature in added) or 'none'} |"
        )
    lines += [
        "",
        "R0 is production Context C 1.2. R1 is the frozen D5 representation: C-minus-RP baseline features, `returning_pct_ppa`, and `transfer_in_prior_usage_sum`. R2 adds only `transfer_in_prior_defensive_experience_sum` and `transfer_in_prior_defensive_experience_available`.",
        "",
        "## Frozen parity controls",
        "",
        "Parity uses checked-in study artifacts as the canonical references; rounded issue text is not used as a reference. Every required predictive metric must be within its declared tolerance before rolling results are interpreted.",
        "",
        "| Control | Reference | Maximum absolute metric delta | Tolerance | Status |",
        "|---|---|---:|---:|:---:|",
    ]
    for label in ("R0", "R1", "R2"):
        item = parity[label]
        lines.append(
            f"| {label} | `{item['reference_candidate']}` | "
            f"{fmt(item['max_abs_metric_delta'], 8)} | {fmt(item['tolerance'], 8)} | "
            f"{'passed' if item['passed'] else 'failed'} |"
        )
    lines += [
        "",
        "| Metric | R0 frozen | R1 frozen | R2 frozen |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        lines.append(
            f"| {metric} | {fmt(frozen_aggregate['R0'][metric], 8)} | "
            f"{fmt(frozen_aggregate['R1'][metric], 8)} | "
            f"{fmt(frozen_aggregate['R2'][metric], 8)} |"
        )
    lines += [
        "",
        "## Rolling-origin protocol",
        "",
        "Each target season is fit using only team-seasons strictly before that target: 2022←2021, 2023←2022, 2024←2023, and 2025←2024. All fits use the existing Context C 1.2 family, penalty 0.25, training-only preprocessing and missingness indicators, the deterministic optimizer retry, the production FBS population, the August 15 season-relative portal cutoff, and stored H PMFs for fallback rows. No target-season outcomes enter fitting or feature construction.",
        "",
        "| Target | Train through | R0 N | R1 N | R2 N |",
        "|---:|---:|---:|---:|---:|",
    ]
    for target in TARGET_SEASONS:
        rows = [row for row in rolling_rows if row["target_season"] == target]
        lines.append(
            f"| {target} | {target - 1} | "
            + " | ".join(
                str(
                    next(
                        row["n_team_seasons"]
                        for row in rows
                        if row["candidate"] == candidate
                    )
                )
                for candidate in ("R0", "R1", "R2")
            )
            + " |"
        )
    lines += [
        "",
        "## Training-data coverage",
        "",
        "Counts are calculated from raw feature availability before DirectRankModel imputation. An observed no-incoming defensive-transfer row has natural experience zero but is available; unresolved rows have neutralized zero plus availability zero.",
        "",
        "| Target | Training team-seasons | Transfer-covered seasons | Observed incoming usage | Usage fraction | Defensive source available | Defensive fraction | Defensive unresolved |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in coverage:
        lines.append(
            f"| {row['target_season']} | {row['n_training_team_seasons']} | "
            f"{row['n_transfer_covered_training_seasons']} | "
            f"{row['n_training_rows_with_observed_incoming_offensive_usage']} | "
            f"{fmt(row['fraction_with_observed_incoming_offensive_usage'], 3)} | "
            f"{row['n_training_rows_with_defensive_experience_source_available']} | "
            f"{fmt(row['fraction_with_defensive_experience_source_available'], 3)} | "
            f"{row['n_training_rows_unresolved_for_defensive_experience']} |"
        )
    lines += [
        "",
        "## Rolling predictive metrics",
        "",
        "All deltas are candidate minus reference; negative NLL and CRPS favor the candidate.",
        "",
        "| Target | Candidate | NLL | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width | ΔNLL vs R0 | ΔNLL vs R1 | ΔCRPS vs R0 | ΔCRPS vs R1 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rolling_rows:
        lines.append(
            f"| {row['target_season']} | {row['candidate']} | {fmt(row['nll'])} | "
            f"{fmt(row['crps'])} | {fmt(row['expected_rank_mae'], 2)} | "
            f"{fmt(row['median_rank_mae'], 2)} | {fmt(row['interval_80_coverage'], 3)} | "
            f"{fmt(row['interval_80_average_width'], 2)} | "
            f"{fmt(row['delta_nll_vs_R0'], 6)} | {fmt(row['delta_nll_vs_R1'], 6)} | "
            f"{fmt(row['delta_crps_vs_R0'], 6)} | {fmt(row['delta_crps_vs_R1'], 6)} |"
        )
    lines += [
        "",
        "## Explicit pairwise rolling deltas",
        "",
        "These rows make the three requested comparisons explicit: R1 − R0, R2 − R1, and R2 − R0.",
        "",
        "| Target | Comparison | ΔNLL | ΔCRPS | Δ expected-rank MAE | Δ median-rank MAE | Δ 80% coverage | Δ 80% width |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['target_season']} | {row['left_candidate']} − {row['right_candidate']} | "
            f"{fmt(row['delta_nll'], 6)} | {fmt(row['delta_crps'], 6)} | "
            f"{fmt(row['delta_expected_rank_mae'], 2)} | "
            f"{fmt(row['delta_median_rank_mae'], 2)} | "
            f"{fmt(row['delta_interval_80_coverage'], 3)} | "
            f"{fmt(row['delta_interval_80_average_width'], 2)} |"
        )
    lines += [
        "",
        "## Weighted rolling aggregate",
        "",
        "The aggregate is team-season weighted across the four target seasons.",
        "",
        "| Candidate | NLL | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in ("R0", "R1", "R2"):
        row = rolling_by_candidate[candidate]
        lines.append(
            f"| {candidate} | {fmt(row['nll'])} | {fmt(row['crps'])} | "
            f"{fmt(row['expected_rank_mae'], 2)} | {fmt(row['median_rank_mae'], 2)} | "
            f"{fmt(row['interval_80_coverage'], 3)} | "
            f"{fmt(row['interval_80_average_width'], 2)} |"
        )
    lines += [
        "",
        "## R2 versus R1 paired NLL diagnostics",
        "",
        "Negative paired ΔNLL means R2 improves the same team-season under the same target population.",
        "",
        "| Scope | N | Mean ΔNLL | Median ΔNLL | Fraction improved |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in paired_summaries:
        lines.append(
            f"| {row['scope']} | {row['n_team_seasons']} | {fmt(row['mean_delta_nll'], 6)} | "
            f"{fmt(row['median_delta_nll'], 6)} | "
            f"{fmt(row['fraction_team_seasons_improved'], 3)} |"
        )
    lines += [
        "",
        "## Standardized R2 coefficient paths",
        "",
        "Coefficients are for the location equation after training-only standardization. The availability coefficient is reported separately from the defensive value coefficient; source counts are included in the machine-readable CSV.",
        "",
        "| Target | Feature | Coefficient | Sign | Change from prior fit | Location missingness coefficient | Source available | Source unresolved |",
        "|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in coefficients:
        lines.append(
            f"| {row['target_season']} | `{row['feature']}` | "
            f"{fmt(row['standardized_location_coefficient'], 6)} | {row['sign']} | "
            f"{fmt(row['change_from_prior_fit'], 6)} | "
            f"{fmt(row['location_missingness_indicator_coefficient'], 6)} | "
            f"{row['training_source_available_n']} | {row['training_source_unresolved_n']} |"
        )
    lines += [
        "",
        "## Availability-indicator diagnostic",
        "",
        "The explicit availability path is not treated as equivalent to defensive experience. A larger absolute availability coefficient is flagged as dominance because it can indicate that resolvability, rather than the numeric experience value, explains the gain.",
        "",
        "| Target | Defensive value coefficient | Availability coefficient | Availability dominates |",
        "|---:|---:|---:|:---:|",
    ]
    for row in availability["rows"]:
        lines.append(
            f"| {row['target_season']} | {fmt(row['defensive_value_coefficient'], 6)} | "
            f"{fmt(row['availability_coefficient'], 6)} | "
            f"{'yes' if row['availability_dominates'] else 'no'} |"
        )
    lines += [
        "",
        "## Frozen versus rolling comparison",
        "",
        "The frozen rows fit through 2021 once and score 2022–2025. Rolling rows refit before each target. The frozen study preserves the original PR #107 result; the rolling study tests whether it persists as portal-era seasons enter training.",
        "",
        "| Protocol | Candidate | NLL | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol, values in (
        ("frozen", frozen_aggregate),
        ("rolling", rolling_by_candidate),
    ):
        for candidate in ("R0", "R1", "R2"):
            row = values[candidate]
            lines.append(
                f"| {protocol} | {candidate} | {fmt(row['nll'])} | "
                f"{fmt(row['crps'])} | {fmt(row['expected_rank_mae'], 2)} | "
                f"{fmt(row['median_rank_mae'], 2)} | "
                f"{fmt(row['interval_80_coverage'], 3)} | "
                f"{fmt(row['interval_80_average_width'], 2)} |"
            )
    lines += [
        "",
        "## Decision answers",
        "",
        f"- Does defensive experience remain useful as later portal-era seasons enter training? {'Yes' if float(rolling_by_candidate['R2']['nll']) < float(rolling_by_candidate['R1']['nll']) else 'No'} on the weighted rolling NLL, but the season-level evidence is decisive only under the promotion screen above.",
        f"- Is the coefficient directionally stable? {'Yes' if summary['coefficient_stability'][EXPERIENCE]['sign_flip_count'] == 0 else 'No'}; the defensive-value path is {', '.join(summary['coefficient_stability'][EXPERIENCE]['signs'])}.",
        f"- Is the effect broad or concentrated in one season? The paired team-season improvement fraction is {fmt(next(row['fraction_team_seasons_improved'] for row in paired_summaries if row['scope'] == 'aggregate'), 3)}; see the year-level paired table for concentration.",
        f"- Is the gain actually driven by availability? Availability dominates in {availability['availability_dominates_n']} of {len(TARGET_SEASONS)} fits; this is treated as {'a concern' if availability['availability_dominates_n'] else 'not a concern'} by the promotion screen.",
        f"- Should defensive experience be retained in the candidate architecture? The final recommendation is `{summary['recommendation']}`.",
        "",
        "## Provenance and limitations",
        "",
        "The portal and prior-usage inputs are the same retrospective research oracle used by the frozen transfer studies. Raw payloads are read unchanged from `--transfer-root`; they are not rewritten or copied into the generated artifacts. The defensive audit is likewise read as a frozen input. This is a stability decision, not production snapshot engineering or a redesign of the defensive feature.",
        "",
        "## Artifacts",
        "",
        "- `rolling_metrics.csv`, `frozen_metrics.csv`, `rolling_aggregate.csv` — per-target and weighted metrics.",
        "- `rolling_comparisons.csv` — explicit R1−R0, R2−R1, and R2−R0 deltas.",
        "- `training_coverage.csv` — raw training-panel coverage by rolling fit.",
        "- `paired_nll_summary.csv`, `paired_nll_by_team.csv` — paired R2-vs-R1 diagnostics.",
        "- `coefficients.csv`, `availability_diagnostics.csv` — coefficient paths and availability comparison.",
        "- `summary.json`, `report.md`, `plots/` — parity, protocol, provenance, interpretation, and plots.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.resolve()
    transfer_root = args.transfer_root.resolve()
    defensive_path = args.defensive_features.resolve()
    decomposition_summary = args.decomposition_summary.resolve()
    defensive_summary = args.defensive_summary.resolve()
    production_evaluation = args.production_evaluation.resolve()
    output = args.output.resolve()

    configure_source_root(source_root)
    rows, cold, _ = v1_1.v1.load_rows(max_season=max(TARGET_SEASONS))
    fbs = [row for row in rows if row.subdivision == "fbs"]
    contextual, _ = c12.attach_context(fbs, c12.feature_index(), c12.cached_tenures())

    records, usage, portal_seasons, usage_seasons = prior.load_raw_transfer_data(
        transfer_root
    )
    required_portal_seasons = {FROZEN_TRAIN_THROUGH, *TARGET_SEASONS}
    required_usage_seasons = set(range(FROZEN_TRAIN_THROUGH - 1, max(TARGET_SEASONS)))
    if not required_portal_seasons <= portal_seasons:
        raise FileNotFoundError(
            f"D5 requires portal seasons {sorted(required_portal_seasons)}; "
            f"found {sorted(portal_seasons)} under {transfer_root}"
        )
    if not required_usage_seasons <= usage_seasons:
        raise FileNotFoundError(
            f"D5 requires usage seasons {sorted(required_usage_seasons)}; "
            f"found {sorted(usage_seasons)} under {transfer_root}"
        )
    transfer_features = prior.aggregate_team_features(
        records,
        usage,
        prior.fbs_feature_rows(contextual),
        covered_seasons=portal_seasons,
        cutoff=date(2025, *DEFAULT_CUTOFF),
    )
    contextual = prior.attach_transfer_features(contextual, transfer_features)
    defensive_features = defensive.load_defensive_features(defensive_path)
    defensive_coverage = defensive.load_defensive_coverage(defensive_path)
    contextual = defensive.attach_defensive_features(contextual, defensive_features)
    fallback = prior.fallback_for_panel(fbs, cold)
    fallback = [item for item in fallback if item.season in TARGET_SEASONS]

    candidates = candidate_definitions()
    previous_coefficients: dict[tuple[str, str], float] = {}
    rolling_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    coefficient_rows_: list[dict[str, object]] = []
    paired_details: list[dict[str, object]] = []
    paired_summaries: list[dict[str, object]] = []

    for target in TARGET_SEASONS:
        coverage_rows.append(
            training_coverage(contextual, portal_seasons, target_season=target)
        )
        target_rows = [row for row in contextual if row.season == target]
        target_fallback = [item for item in fallback if item.season == target]
        predictions_by_candidate: dict[str, list[v1_1.v1.PriorPrediction]] = {}
        models: dict[str, DirectRankModel] = {}
        training = [row for row in contextual if row.season < target]
        for candidate in candidates:
            model = fit_context(contextual, candidate.features, target_season=target)
            modeled = v1_1.make_predictions(model, target_rows, candidate.name)
            predictions_by_candidate[candidate.label] = (
                prior.merge_modeled_and_fallback(
                    modeled, target_fallback, candidate.name
                )
            )
            models[candidate.label] = model
        key_sets = {
            frozenset(item.key for item in values)
            for values in predictions_by_candidate.values()
        }
        if len(key_sets) != 1:
            raise ValueError(
                f"rolling candidates changed target population for {target}"
            )
        rolling_rows.extend(
            metric_rows(
                predictions_by_candidate,
                target_season=target,
                train_through=target - 1,
                protocol="rolling_origin",
            )
        )
        coefficient_rows_.extend(
            coefficient_rows(
                models["R2"],
                candidates[2],
                target_season=target,
                previous=previous_coefficients,
                training=training,
            )
        )
        details, summaries = paired_nll_diagnostics(
            "R2",
            predictions_by_candidate["R2"],
            "R1",
            predictions_by_candidate["R1"],
        )
        paired_details.extend(details)
        paired_summaries.extend(summaries[1:])
        print(f"completed defensive-transfer stability/{target}", flush=True)

    frozen_predictions: dict[str, list[v1_1.v1.PriorPrediction]] = {}
    frozen_target_rows = [row for row in contextual if row.season in TARGET_SEASONS]
    frozen_fallback = [item for item in fallback if item.season in TARGET_SEASONS]
    for candidate in candidates:
        model = fit_context(
            contextual,
            candidate.features,
            target_season=FROZEN_TRAIN_THROUGH + 1,
        )
        modeled = v1_1.make_predictions(model, frozen_target_rows, candidate.name)
        frozen_predictions[candidate.label] = prior.merge_modeled_and_fallback(
            modeled, frozen_fallback, candidate.name
        )
    frozen_rows: list[dict[str, object]] = []
    for target in TARGET_SEASONS:
        frozen_rows.extend(
            metric_rows(
                {
                    label: [item for item in predictions if item.season == target]
                    for label, predictions in frozen_predictions.items()
                },
                target_season=target,
                train_through=FROZEN_TRAIN_THROUGH,
                protocol="frozen_through_2021",
            )
        )

    rolling_aggregate = [
        aggregate_metrics(rolling_rows, candidate=candidate, protocol="rolling_origin")
        for candidate in ("R0", "R1", "R2")
    ]
    frozen_aggregate = {
        candidate: _reference_metric_map(frozen_rows, candidate, "frozen_through_2021")
        for candidate in ("R0", "R1", "R2")
    }
    comparisons = comparison_rows(rolling_rows)
    coefficient_summary = coefficient_stability(coefficient_rows_)
    availability = availability_diagnostics(coefficient_rows_)
    recommendation, decision_criteria = choose_recommendation(
        comparisons,
        coefficient_summary,
        availability,
        [
            {
                "scope": "aggregate",
                **{
                    key: value
                    for key, value in {
                        "mean_delta_nll": float(
                            np.mean([float(row["delta_nll"]) for row in paired_details])
                        ),
                        "fraction_team_seasons_improved": float(
                            np.mean(
                                [float(row["delta_nll"]) < 0 for row in paired_details]
                            )
                        ),
                    }.items()
                },
            }
        ]
        + [row for row in paired_summaries if row["scope"] != "aggregate"],
    )

    # The paired summary list is accumulated year by year; add one aggregate
    # row after the recommendation screen has its exact same values.
    paired_aggregate = {
        "scope": "aggregate",
        "candidate": "R2",
        "reference": "R1",
        "n_team_seasons": len(paired_details),
        "mean_delta_nll": float(
            np.mean([float(row["delta_nll"]) for row in paired_details])
        ),
        "median_delta_nll": float(
            np.median([float(row["delta_nll"]) for row in paired_details])
        ),
        "fraction_team_seasons_improved": float(
            np.mean([float(row["delta_nll"]) < 0 for row in paired_details])
        ),
    }
    paired_summaries = [paired_aggregate, *paired_summaries]

    production_expected = read_production_c0_metrics(production_evaluation)
    d5_expected = read_csv_metric_row(
        decomposition_summary,
        field="candidate",
        value="D5_total_rp_plus_incoming",
    )
    defensive_expected = read_csv_metric_row(
        defensive_summary,
        field="variant",
        value="E4_D5_plus_defensive_experience",
    )
    frozen_parity = {
        "R0": {
            "reference_candidate": "production C0",
            **metric_parity(
                frozen_aggregate["R0"],
                production_expected,
                candidate="R0",
                reference_path=production_evaluation,
                tolerance=C0_PARITY_TOLERANCE,
            ),
        },
        "R1": {
            "reference_candidate": "D5_total_rp_plus_incoming",
            **metric_parity(
                frozen_aggregate["R1"],
                d5_expected,
                candidate="R1",
                reference_path=decomposition_summary,
                tolerance=FROZEN_PARITY_TOLERANCE,
            ),
        },
        "R2": {
            "reference_candidate": "E4_D5_plus_defensive_experience",
            **metric_parity(
                frozen_aggregate["R2"],
                defensive_expected,
                candidate="R2",
                reference_path=defensive_summary,
                tolerance=FROZEN_PARITY_TOLERANCE,
            ),
        },
    }

    source_files = {
        "defensive_features": defensive_path,
        "defensive_summary": defensive_summary,
        "decomposition_summary": decomposition_summary,
        "production_evaluation": production_evaluation,
        "preseason_features": source_root
        / "data/processed/preseason/team_season_features.csv",
        "rank_distributions": source_root
        / "data/processed/modeling/team_season_rank_distributions.csv",
    }
    source_hashes = {
        name: sha256_file(path) for name, path in source_files.items() if path.exists()
    }
    summary: dict[str, object] = {
        "study": "issue_109_defensive_transfer_stability",
        "recommendation": recommendation,
        "decision_criteria": decision_criteria,
        "target_seasons": list(TARGET_SEASONS),
        "frozen_train_through": FROZEN_TRAIN_THROUGH,
        "cutoff": f"August {DEFAULT_CUTOFF[1]} season-relative",
        "model_protocol": {
            "family": "Context C 1.2 location fit",
            "penalty": 0.25,
            "location_features": "H plus candidate features",
            "scale_features": "H only",
            "optimizer_retry_maxiter": 2000,
            "preprocessing": "DirectRankModel training-only imputation, standardization, and missingness indicators",
            "fallback": "stored H PMFs for rows without modeled Context predictions",
        },
        "candidates": [
            {
                "label": candidate.label,
                "name": candidate.name,
                "description": candidate.description,
                "features": list(candidate.features),
            }
            for candidate in candidates
        ],
        "frozen_parity": frozen_parity,
        "defensive_missing_data_policy": {
            "observed_statuses": [
                "complete",
                "no_incoming_defensive_transfer",
            ],
            "unresolved_statuses": [
                "partial",
                "no_usable_defensive_transfer",
            ],
            "observed_no_incoming_value": 0.0,
            "unresolved_neutral_value": 0.0,
            "availability_feature": EXPERIENCE_AVAILABLE,
            "population_preserved": True,
        },
        "defensive_coverage": defensive_coverage,
        "training_coverage": coverage_rows,
        "coefficient_stability": coefficient_summary,
        "availability_diagnostics": availability,
        "portal_seasons": sorted(portal_seasons),
        "usage_seasons": sorted(usage_seasons),
        "raw_transfer_input_sha256": transfer_source_hashes(transfer_root),
        "source_hashes": source_hashes,
        "rolling_aggregate": rolling_aggregate,
        "rolling_comparisons": comparisons,
        "frozen_aggregate": frozen_aggregate,
        "paired_nll_summary": paired_summaries,
        "optional_coefficient_resampling": "omitted; coefficient paths and coverage diagnostics are descriptive",
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    write_json(output / "candidate_definitions.json", summary["candidates"])
    write_json(output / "availability_diagnostics.json", availability)
    write_csv(output / "rolling_metrics.csv", rolling_rows)
    write_csv(output / "frozen_metrics.csv", frozen_rows)
    write_csv(output / "rolling_aggregate.csv", rolling_aggregate)
    write_csv(output / "rolling_comparisons.csv", comparisons)
    write_csv(output / "training_coverage.csv", coverage_rows)
    write_csv(output / "coefficients.csv", coefficient_rows_)
    write_csv(output / "availability_diagnostics.csv", availability["rows"])
    write_csv(output / "paired_nll_summary.csv", paired_summaries)
    write_csv(output / "paired_nll_by_team.csv", paired_details)
    plot_outputs(output, rolling_rows, coefficient_rows_, paired_summaries)
    render_report(
        output / "report.md",
        summary=summary,
        rolling_rows=rolling_rows,
        frozen_rows=frozen_rows,
        rolling_aggregate=rolling_aggregate,
        comparisons=comparisons,
        coverage=coverage_rows,
        coefficients=coefficient_rows_,
        paired_summaries=paired_summaries,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument(
        "--transfer-root",
        type=Path,
        default=ROOT / "data/raw/cfbd/preseason/transfers",
        help="Directory containing immutable portal/usage JSON payloads.",
    )
    parser.add_argument(
        "--defensive-features",
        type=Path,
        default=ROOT
        / "data/processed/defensive_transfer_audit/team_season_features.csv",
    )
    parser.add_argument(
        "--decomposition-summary",
        type=Path,
        default=ROOT
        / "data/processed/transfer_signal_decomposition/candidate_summary.csv",
    )
    parser.add_argument(
        "--defensive-summary",
        type=Path,
        default=ROOT
        / "data/processed/defensive_transfer_predictive_value/candidate_summary.csv",
    )
    parser.add_argument(
        "--production-evaluation",
        type=Path,
        default=ROOT / "data/processed/preseason/context/evaluation.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/defensive_transfer_stability",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "study": summary["study"],
                "output": str(args.output),
                "recommendation": summary["recommendation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
