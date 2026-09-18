"""Rolling-origin stability study for the selected transfer-production signal.

This is the issue #97 follow-up to the fixed candidate comparison in issue
#96.  The candidate is deliberately not searched again here: R2 is the D5
representation selected in that decomposition study.  Each target season is
fit using only earlier seasons, with the existing Context C 1.2 optimizer,
penalty, preprocessing, and H fallback path.

The script writes research artifacts only under
``data/processed/transfer_production_stability`` by default.  Transfer data is
passed as a separate raw-data root and is never modified.
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
import build_preseason_prior as v1
import build_preseason_prior_v1_1 as h11
import investigate_transfer_roster_continuity as oracle

from gippyrank.preseason import DirectRankModel, TeamSeason

ROOT = Path(__file__).resolve().parents[1]
TARGET_SEASONS = (2022, 2023, 2024, 2025)
FROZEN_TRAIN_THROUGH = 2021
DEFAULT_CUTOFF = (8, 15)
D5_PARITY_TOLERANCE = 1e-6
DECOMPOSITION_SUMMARY = (
    ROOT / "data/processed/transfer_signal_decomposition/candidate_summary.csv"
)
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
SELECTED_TRANSFER_FEATURES = ("transfer_in_prior_usage_sum",)
C_MINUS_RP_FEATURES = tuple(
    feature for feature in BASE_CONTEXT_FEATURES if feature not in RP_FEATURES
)
ROSTER_CONTINUITY_FEATURES = ("returning_pct_ppa", *SELECTED_TRANSFER_FEATURES)

METRICS = (
    "nll",
    "crps",
    "expected_rank_mae",
    "median_rank_mae",
    "interval_80_coverage",
    "interval_80_average_width",
)


@dataclass(frozen=True)
class Candidate:
    """A predeclared model variant in the stability study."""

    label: str
    name: str
    features: tuple[str, ...]


def candidate_definitions() -> tuple[Candidate, ...]:
    """Return the fixed R0/R1/R2 set; no rolling result selects a feature."""
    return (
        Candidate("R0", "C0_full", BASE_CONTEXT_FEATURES),
        Candidate("R1", "C_minus_RP", C_MINUS_RP_FEATURES),
        Candidate(
            "R2",
            "D5_total_rp_plus_incoming",
            (*C_MINUS_RP_FEATURES, *ROSTER_CONTINUITY_FEATURES),
        ),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def source_hashes(transfer_root: Path) -> dict[str, str]:
    """Hash raw transfer inputs without embedding machine-specific paths."""
    result = {}
    for directory in ("portal", "usage"):
        for path in sorted((transfer_root / directory).glob("*.json")):
            if path.name.endswith(".provenance.json") or path.name == "manifest.json":
                continue
            result[str(path.relative_to(transfer_root))] = sha256_file(path)
    return result


def configure_source_root(source_root: Path) -> None:
    """Point the existing loaders at an explicit cached-input checkout."""
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


def fbs_feature_rows(rows: Iterable[TeamSeason]) -> list[dict[str, object]]:
    return [
        {
            "season": row.season,
            "subdivision": row.subdivision,
            "team_id": row.team_id,
            "team_name": row.team_name,
            "returning_pct_ppa": row.features.get("returning_pct_ppa"),
        }
        for row in rows
    ]


def attach_transfer_features(
    rows: list[TeamSeason],
    transfer_features: dict[tuple[int, str, str], dict[str, float | None]],
) -> list[TeamSeason]:
    from dataclasses import replace

    return [
        replace(
            row,
            features={
                **row.features,
                **transfer_features.get((row.season, row.subdivision, row.team_id), {}),
            },
        )
        for row in rows
    ]


def fit_context(
    rows: list[TeamSeason],
    features: Iterable[str],
    *,
    target_season: int,
) -> DirectRankModel:
    """Fit C 1.2 exactly, with training restricted before the target season."""
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


def score(predictions: list[v1.PriorPrediction]) -> dict[str, float]:
    raw = v1.score_predictions(predictions)
    return {metric: float(raw[metric]) for metric in METRICS}


def read_d5_reference(path: Path = DECOMPOSITION_SUMMARY) -> dict[str, float]:
    """Read the frozen D5 aggregate produced by issue 96."""
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("candidate") == "D5_total_rp_plus_incoming"
                and row.get("target_season") == "aggregate"
            ):
                return {metric: float(row[metric]) for metric in METRICS}
    raise ValueError(f"D5 aggregate reference is missing from {path}")


def d5_frozen_parity(
    computed: dict[str, float],
    *,
    reference_path: Path = DECOMPOSITION_SUMMARY,
    tolerance: float = D5_PARITY_TOLERANCE,
) -> dict[str, object]:
    """Require frozen R2 to reproduce issue 96's selected D5 aggregate."""
    expected = read_d5_reference(reference_path)
    deltas = {
        metric: abs(computed[metric] - expected[metric]) for metric in METRICS
    }
    maximum = max(deltas.values(), default=None)
    result: dict[str, object] = {
        "reference_path": str(reference_path),
        "reference_metrics": expected,
        "computed_metrics": computed,
        "metric_abs_deltas": deltas,
        "max_abs_metric_delta": maximum,
        "tolerance": tolerance,
        "passed": maximum is not None and maximum <= tolerance,
    }
    if not result["passed"]:
        raise ValueError(f"frozen R2 does not reproduce D5: {result}")
    return result


def sign(value: float, tolerance: float = 1e-10) -> str:
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "zero"


def training_coverage(
    rows: list[TeamSeason],
    transfer_features: dict[tuple[int, str, str], dict[str, float | None]],
    portal_seasons: set[int],
    *,
    target_season: int,
) -> dict[str, object]:
    """Count raw transfer coverage before DirectRankModel imputation."""
    training = [row for row in rows if row.season < target_season]
    counts = {
        feature: sum(
            transfer_features.get(
                (row.season, row.subdivision, row.team_id), {}
            ).get(feature)
            is not None
            for row in training
        )
        for feature in SELECTED_TRANSFER_FEATURES
    }
    complete = sum(
        all(
            transfer_features.get(
                (row.season, row.subdivision, row.team_id), {}
            ).get(feature)
            is not None
            for feature in SELECTED_TRANSFER_FEATURES
        )
        for row in training
    )
    any_observed = sum(
        any(
            transfer_features.get(
                (row.season, row.subdivision, row.team_id), {}
            ).get(feature)
            is not None
            for feature in SELECTED_TRANSFER_FEATURES
        )
        for row in training
    )
    n_rows = len(training)
    result: dict[str, object] = {
        "target_season": target_season,
        "train_through": target_season - 1,
        "n_training_team_seasons": n_rows,
        "n_training_seasons": len({row.season for row in training}),
        "n_transfer_covered_training_seasons": len(
            {row.season for row in training if row.season in portal_seasons}
        ),
        "transfer_covered_training_seasons": sorted(
            {row.season for row in training if row.season in portal_seasons}
        ),
        "n_team_seasons_with_any_observed_transfer_production": any_observed,
        "n_team_seasons_with_all_observed_transfer_production": complete,
        "fraction_with_all_observed_transfer_production": (
            complete / n_rows if n_rows else None
        ),
    }
    result.update(
        {
            f"n_team_seasons_with_{feature}": count
            for feature, count in counts.items()
        }
    )
    return result


def coefficient_index(model: DirectRankModel, feature: str) -> tuple[int, int]:
    """Return numeric and missingness-indicator beta indexes for a feature."""
    if feature not in model.feature_names:
        raise ValueError(f"feature is not in model: {feature}")
    feature_index = model.feature_names.index(feature)
    numeric = model.lag_count + 1 + feature_index
    indicator = model.lag_count + 1 + len(model.feature_names) + feature_index
    return numeric, indicator


def coefficient_rows(
    model: DirectRankModel,
    candidate: Candidate,
    *,
    target_season: int,
    previous: dict[tuple[str, str], float],
) -> list[dict[str, object]]:
    """Extract selected D5 coefficients and their fit-to-fit deltas."""
    if candidate.label != "R2":
        return []
    result = []
    for feature in ROSTER_CONTINUITY_FEATURES:
        if feature not in model.feature_names:
            continue
        numeric_index, missing_index = coefficient_index(model, feature)
        feature_index = model.feature_names.index(feature)
        value = float(model.beta[numeric_index])
        key = (candidate.label, feature)
        prior = previous.get(key)
        row: dict[str, object] = {
            "target_season": target_season,
            "train_through": target_season - 1,
            "candidate": candidate.label,
            "candidate_name": candidate.name,
            "feature": feature,
            "standardized_location_coefficient": value,
            "sign": sign(value),
            "magnitude": abs(value),
            "change_from_prior_fit": value - prior if prior is not None else None,
            "missingness_indicator_location_coefficient": float(
                model.beta[missing_index]
            ),
            "standardized_scale_coefficient": float(
                model.gamma[1 + feature_index]
            ),
        }
        result.append(row)
        previous[key] = value
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
        raise ValueError(f"no metric rows for {candidate}/{protocol}")
    weights = np.asarray([int(row["n_team_seasons"]) for row in selected], dtype=float)
    result: dict[str, object] = {
        "candidate": candidate,
        "protocol": protocol,
        "n_target_seasons": len(selected),
        "n_team_seasons": int(weights.sum()),
        "target_seasons": ";".join(str(row["target_season"]) for row in selected),
    }
    for metric in METRICS:
        result[metric] = float(
            np.average(
                np.asarray([float(row[metric]) for row in selected]), weights=weights
            )
        )
    return result


def metric_rows(
    predictions_by_candidate: dict[str, list[v1.PriorPrediction]],
    *,
    target_season: int,
    train_through: int,
    protocol: str,
) -> list[dict[str, object]]:
    scores = {candidate: score(items) for candidate, items in predictions_by_candidate.items()}
    n_team_seasons = len(next(iter(predictions_by_candidate.values())))
    rows = []
    for candidate, values in scores.items():
        row: dict[str, object] = {
            "protocol": protocol,
            "target_season": target_season,
            "train_through": train_through,
            "candidate": candidate,
            "n_team_seasons": n_team_seasons,
        }
        row.update(values)
        for reference in ("R0", "R1"):
            if reference in scores:
                for metric in METRICS:
                    row[f"delta_{metric}_vs_{reference}"] = (
                        values[metric] - scores[reference][metric]
                    )
        rows.append(row)
    return rows


def comparison_rows(metric_rows_: list[dict[str, object]]) -> list[dict[str, object]]:
    """Materialize all three requested pairwise comparisons."""
    by_key = {
        (int(row["target_season"]), str(row["candidate"])): row
        for row in metric_rows_
    }
    result = []
    for target in TARGET_SEASONS:
        for left, right in (("R2", "R0"), ("R2", "R1"), ("R1", "R0")):
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


def source_coverage(
    records: list[oracle.TransferRecord],
    *,
    transfer_root: Path,
    team_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Retain season-level transfer provenance in the stability output."""
    rows = oracle.coverage_rows(
        records,
        target_seasons=range(2021, 2026),
        cutoff=date(2025, *DEFAULT_CUTOFF),
        team_rows=team_rows,
        covered_seasons={
            int(path.stem)
            for path in (transfer_root / "portal").glob("*.json")
            if path.name != "manifest.json" and not path.name.endswith(".provenance.json")
        },
    )
    return rows


def plot_outputs(
    output: Path,
    metric_rows_: list[dict[str, object]],
    coefficients: list[dict[str, object]],
) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    for candidate in ("R0", "R1", "R2"):
        values = [
            float(row["nll"])
            for row in metric_rows_
            if row["protocol"] == "rolling_origin" and row["candidate"] == candidate
        ]
        axis.plot(TARGET_SEASONS, values, marker="o", label=candidate)
    axis.set_xlabel("target season")
    axis.set_ylabel("NLL")
    axis.set_title("Transfer-production rolling-origin NLL")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "rolling_nll.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 6))
    for feature in ROSTER_CONTINUITY_FEATURES:
        values = [
            float(row["standardized_location_coefficient"])
            for row in coefficients
            if row["candidate"] == "R2" and row["feature"] == feature
        ]
        years = [
            int(row["target_season"])
            for row in coefficients
            if row["candidate"] == "R2" and row["feature"] == feature
        ]
        if values:
            axis.plot(years, values, marker="o", label=feature)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("target season")
    axis.set_ylabel("standardized location coefficient")
    axis.set_title("R2 roster-continuity coefficient paths")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots / "r2_coefficients.png", dpi=160)
    plt.close(figure)


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def coefficient_stability(coefficients: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for feature in ROSTER_CONTINUITY_FEATURES:
        rows = [
            row
            for row in coefficients
            if row["candidate"] == "R2" and row["feature"] == feature
        ]
        values = [float(row["standardized_location_coefficient"]) for row in rows]
        signs = [str(row["sign"]) for row in rows]
        result[feature] = {
            "signs": signs,
            "sign_flip_count": sum(
                left != right
                for left, right in pairwise(signs)
                if left != "zero" and right != "zero"
            ),
            "max_absolute_coefficient": max(map(abs, values)) if values else None,
            "max_absolute_change": max(
                (abs(float(row["change_from_prior_fit"])) for row in rows if row["change_from_prior_fit"] is not None),
                default=None,
            ),
        }
    return result


def render_report(
    path: Path,
    *,
    summary: dict[str, object],
    metric_rows_: list[dict[str, object]],
    coverage: list[dict[str, object]],
    coefficients: list[dict[str, object]],
    comparisons: list[dict[str, object]],
) -> None:
    rolling = [row for row in metric_rows_ if row["protocol"] == "rolling_origin"]
    frozen = [row for row in metric_rows_ if row["protocol"] == "frozen_through_2021"]
    rolling_summary = {
        candidate: aggregate_metrics(rolling, candidate=candidate, protocol="rolling_origin")
        for candidate in ("R0", "R1", "R2")
    }
    frozen_summary = {
        candidate: aggregate_metrics(frozen, candidate=candidate, protocol="frozen_through_2021")
        for candidate in ("R0", "R1", "R2")
    }
    stability = summary["coefficient_stability"]
    r2_vs_r0 = [
        row for row in comparisons if row["left_candidate"] == "R2" and row["right_candidate"] == "R0"
    ]
    r2_nll_wins = sum(float(row["delta_nll"]) < 0 for row in r2_vs_r0)
    r2_mean_delta = float(rolling_summary["R2"]["nll"]) - float(rolling_summary["R0"]["nll"])
    r2_sign_flips = sum(
        int(item["sign_flip_count"]) for item in stability.values()
    )
    r2_max_magnitude = max(
        float(item["max_absolute_coefficient"])
        for item in stability.values()
        if item["max_absolute_coefficient"] is not None
    )

    lines = [
        "# Transfer-production coefficient stability (issue 97)",
        "",
        "## Conclusion",
        "",
        f"R2 beats R0 by rolling-origin NLL in **{r2_nll_wins} / {len(r2_vs_r0)}** target seasons, with weighted mean ΔNLL (R2 − R0) of **{r2_mean_delta:+.4f}**. The R2 roster-continuity coefficient paths contain **{r2_sign_flips}** adjacent non-zero sign flips; the largest absolute standardized location coefficient is **{r2_max_magnitude:.3f}**.",
        "These results are a stability diagnostic, not a feature-selection result. The selected representation was fixed before the rolling scores were calculated, and no regularization, interaction, breakpoint, or year-specific term was tuned against 2022–2025.",
        f"The coefficient and performance evidence should be read together: the rolling study {'supports continued investigation' if r2_mean_delta < 0 and r2_sign_flips == 0 else 'does not by itself establish a production-safe relationship'}; it does not replace the original frozen-through-2021 held-out evaluation.",
        "",
        "## Selected representation",
        "",
        "R2 is the exact D5 representation selected by the issue-96 decomposition study:",
        "",
        "- C-minus-RP baseline features: coaching, recruiting, and Team Talent;",
        "- total returning production: `returning_pct_ppa`;",
        "- incoming prior transfer production: `transfer_in_prior_usage_sum`.",
        "",
        "R0 is C0 with the existing Context feature family. R1 removes all returning-production features. R2 is D5 and adds only total RP plus incoming prior transfer production to the C-minus-RP baseline.",
        "",
        "## Frozen D5 parity check",
        "",
        f"The frozen-through-2021 R2 aggregate reproduces the checked-in issue-96 D5 aggregate within the configured tolerance of {summary['d5_frozen_parity']['tolerance']:.1e}; maximum absolute metric difference is {fmt(summary['d5_frozen_parity']['max_abs_metric_delta'], 8)}.",
        "",
        "| Metric | Issue-96 D5 | Frozen R2 | Absolute difference |",
        "|---|---:|---:|---:|",
        *[
            f"| {metric} | {fmt(summary['d5_frozen_parity']['reference_metrics'][metric], 8)} | {fmt(summary['d5_frozen_parity']['computed_metrics'][metric], 8)} | {fmt(summary['d5_frozen_parity']['metric_abs_deltas'][metric], 8)} |"
            for metric in METRICS
        ],
        "",
        "## Rolling-origin protocol",
        "",
        "Each fit uses all FBS team-seasons strictly before its target season and scores the unchanged target-season population. The C 1.2 penalty is 0.25; all model features enter the location equation, H features alone enter scale, and the existing deterministic optimizer retry is retained. Raw transfer fields are filtered to the August 15 season-relative cutoff. Training-only median imputation and missingness indicators remain in `DirectRankModel`. Stored H PMFs remain the cold-start fallback.",
        "",
        "| Target | Train through | R0 | R1 | R2 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for target in TARGET_SEASONS:
        lines.append(
            f"| {target} | {target - 1} | {next(row['n_team_seasons'] for row in rolling if row['target_season'] == target and row['candidate'] == 'R0')} | {next(row['n_team_seasons'] for row in rolling if row['target_season'] == target and row['candidate'] == 'R1')} | {next(row['n_team_seasons'] for row in rolling if row['target_season'] == target and row['candidate'] == 'R2')} |"
        )
    lines += [
        "",
        "## Training-data coverage",
        "",
        "Counts below are computed from raw transfer values before preprocessing. The selected transfer-production input is observed when `transfer_in_prior_usage_sum` is present.",
        "",
        "| Target | Training team-seasons | Transfer-covered seasons | Rows with incoming usage | Fraction observed |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in coverage:
        lines.append(
            f"| {row['target_season']} | {row['n_training_team_seasons']} | {row['n_transfer_covered_training_seasons']} | {row['n_team_seasons_with_transfer_in_prior_usage_sum']} | {fmt(row['fraction_with_all_observed_transfer_production'], 3)} |"
        )
    lines += [
        "",
        "## Rolling-origin predictive metrics",
        "",
        "Δ columns are candidate minus the named reference; negative values favor the candidate.",
        "",
        "| Target | Candidate | NLL | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width | ΔNLL vs R0 | ΔNLL vs R1 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rolling:
        lines.append(
            f"| {row['target_season']} | {row['candidate']} | {fmt(row['nll'])} | {fmt(row['crps'])} | {fmt(row['expected_rank_mae'], 2)} | {fmt(row['median_rank_mae'], 2)} | {fmt(row['interval_80_coverage'], 3)} | {fmt(row['interval_80_average_width'], 2)} | {fmt(row.get('delta_nll_vs_R0'))} | {fmt(row.get('delta_nll_vs_R1'))} |"
        )
    lines += [
        "",
        "## R2 standardized roster-continuity coefficients",
        "",
        "The reported location coefficients are for the two selected D5 continuity inputs after training-only standardization. Missingness-indicator coefficients are shown separately. No outgoing or net transfer-production feature is included.",
        "",
        "| Target | Feature | Coefficient | Sign | Change from prior fit | Missingness indicator |",
        "|---:|---|---:|---|---:|---:|",
    ]
    for row in coefficients:
        if row["candidate"] == "R2":
            lines.append(
                f"| {row['target_season']} | `{row['feature']}` | {fmt(row['standardized_location_coefficient'], 4)} | {row['sign']} | {fmt(row['change_from_prior_fit'], 4)} | {fmt(row['missingness_indicator_location_coefficient'], 4)} |"
            )
    lines += [
        "",
        "## Pairwise comparisons",
        "",
        "| Target | Comparison | ΔNLL | ΔCRPS | Δ expected-rank MAE | Δ median-rank MAE | Δ 80% coverage | Δ 80% width |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['target_season']} | {row['left_candidate']} − {row['right_candidate']} | {fmt(row['delta_nll'])} | {fmt(row['delta_crps'])} | {fmt(row['delta_expected_rank_mae'], 2)} | {fmt(row['delta_median_rank_mae'], 2)} | {fmt(row['delta_interval_80_coverage'], 3)} | {fmt(row['delta_interval_80_average_width'], 2)} |"
        )
    lines += [
        "",
        "## Frozen versus rolling behavior",
        "",
        "The frozen rows refit each candidate through 2021 and score 2022–2025 without retraining between target seasons. The rolling rows refit before each target. The frozen evaluation answers whether the original 2021-trained hypothesis generalized to untouched future seasons; the rolling evaluation answers whether the relationship remains useful as later portal-era seasons enter training.",
        "",
        "| Protocol | Candidate | NLL | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol, items in (("frozen", frozen_summary), ("rolling", rolling_summary)):
        for candidate in ("R0", "R1", "R2"):
            row = items[candidate]
            lines.append(
                f"| {protocol} | {candidate} | {fmt(row['nll'])} | {fmt(row['crps'])} | {fmt(row['expected_rank_mae'], 2)} | {fmt(row['median_rank_mae'], 2)} | {fmt(row['interval_80_coverage'], 3)} | {fmt(row['interval_80_average_width'], 2)} |"
            )
    lines += [
        "",
        "## Optional coefficient uncertainty diagnostic",
        "",
        "No resampling refit was added. The existing optimizer is already the expensive part of this study, and a season bootstrap would be a descriptive sensitivity check rather than a formal uncertainty model. Coefficient path changes and raw coverage counts are reported instead.",
        "",
        "## Provenance and limitations",
        "",
        "The portal and prior-usage inputs are a retrospective research oracle: the endpoint responses are not archived as August 15 snapshots, final destinations can be resolved later, and the portal-to-usage join is name-based. The study therefore tests stability of the selected oracle relationship, not production data readiness. No target-season outcomes enter transfer features or fitting.",
        "",
        "## Artifacts",
        "",
        "- `rolling_metrics.csv`, `rolling_comparisons.csv`, `frozen_metrics.csv` — required scores and pairwise deltas.",
        "- `training_coverage.csv` — training-panel size and raw transfer-feature coverage.",
        "- `coefficients.csv` — standardized roster-continuity coefficients and fit-to-fit changes.",
        "- `summary.json`, `report.md`, `plots/` — protocol, hashes, interpretation, and plots.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.resolve()
    transfer_root = args.transfer_root.resolve()
    output = args.output.resolve()
    configure_source_root(source_root)

    rows, cold, _ = v1.load_rows(max_season=max(TARGET_SEASONS))
    fbs = [row for row in rows if row.subdivision == "fbs"]
    contextual, _ = c12.attach_context(fbs, c12.feature_index(), c12.cached_tenures())
    records, usage, portal_seasons, usage_seasons = oracle.load_raw_transfer_data(
        transfer_root
    )
    transfer_features = oracle.aggregate_team_features(
        records,
        usage,
        fbs_feature_rows(contextual),
        covered_seasons=portal_seasons,
        cutoff=date(2025, *DEFAULT_CUTOFF),
    )
    augmented = attach_transfer_features(contextual, transfer_features)
    fallback = oracle.fallback_for_panel(fbs, cold)
    candidates = candidate_definitions()
    by_name = {candidate.label: candidate for candidate in candidates}
    metric_rows_: list[dict[str, object]] = []
    coefficient_rows_: list[dict[str, object]] = []
    coverage_rows_: list[dict[str, object]] = []
    previous_coefficients: dict[tuple[str, str], float] = {}

    for target in TARGET_SEASONS:
        coverage_rows_.append(
            training_coverage(
                augmented,
                transfer_features,
                portal_seasons,
                target_season=target,
            )
        )
        target_rows = [row for row in augmented if row.season == target]
        target_fallback = [item for item in fallback if item.season == target]
        predictions_by_candidate: dict[str, list[v1.PriorPrediction]] = {}
        for candidate in candidates:
            model = fit_context(
                augmented, candidate.features, target_season=target
            )
            modeled = oracle.predictions(model, target_rows, candidate.name)
            predictions_by_candidate[candidate.label] = oracle.merge_modeled_and_fallback(
                modeled, target_fallback, candidate.name
            )
            coefficient_rows_.extend(
                coefficient_rows(
                    model,
                    candidate,
                    target_season=target,
                    previous=previous_coefficients,
                )
            )
        key_sets = {tuple(item.key for item in values) for values in predictions_by_candidate.values()}
        if len(key_sets) != 1:
            raise ValueError(f"rolling candidates changed target population for {target}")
        metric_rows_.extend(
            metric_rows(
                predictions_by_candidate,
                target_season=target,
                train_through=target - 1,
                protocol="rolling_origin",
            )
        )
        print(f"completed transfer-production stability/{target}", flush=True)

    frozen_predictions: dict[str, list[v1.PriorPrediction]] = {}
    for candidate in candidates:
        model = fit_context(
            augmented, candidate.features, target_season=FROZEN_TRAIN_THROUGH + 1
        )
        modeled = oracle.predictions(
            model,
            [row for row in augmented if row.season in TARGET_SEASONS],
            candidate.name,
        )
        frozen_predictions[candidate.label] = oracle.merge_modeled_and_fallback(
            modeled,
            [item for item in fallback if item.season in TARGET_SEASONS],
            candidate.name,
        )
    for target in TARGET_SEASONS:
        metric_rows_.extend(
            metric_rows(
                {
                    label: [item for item in values if item.season == target]
                    for label, values in frozen_predictions.items()
                },
                target_season=target,
                train_through=FROZEN_TRAIN_THROUGH,
                protocol="frozen_through_2021",
            )
        )

    rolling_rows = [row for row in metric_rows_ if row["protocol"] == "rolling_origin"]
    comparisons = comparison_rows(rolling_rows)
    stability = coefficient_stability(coefficient_rows_)
    rolling_summary = {
        candidate: aggregate_metrics(
            rolling_rows, candidate=candidate, protocol="rolling_origin"
        )
        for candidate in ("R0", "R1", "R2")
    }
    frozen_summary = {
        candidate: aggregate_metrics(
            metric_rows_, candidate=candidate, protocol="frozen_through_2021"
        )
        for candidate in ("R0", "R1", "R2")
    }
    d5_parity = d5_frozen_parity(frozen_summary["R2"])
    coverage = source_coverage(
        records,
        transfer_root=transfer_root,
        team_rows=fbs_feature_rows(contextual),
    )
    summary: dict[str, object] = {
        "study": "issue_97_transfer_production_stability",
        "selected_representation": {
            "source": "issue_96_D5_total_rp_plus_incoming",
            "context_features": list(C_MINUS_RP_FEATURES),
            "continuity_features": list(ROSTER_CONTINUITY_FEATURES),
            "transfer_production_features": list(SELECTED_TRANSFER_FEATURES),
            "full_features": list(by_name["R2"].features),
        },
        "candidates": [
            {"label": item.label, "name": item.name, "features": list(item.features)}
            for item in candidates
        ],
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
        "training_coverage": coverage_rows_,
        "coefficient_stability": stability,
        "d5_frozen_parity": d5_parity,
        "portal_seasons": sorted(portal_seasons),
        "usage_seasons": sorted(usage_seasons),
        "raw_transfer_input_sha256": source_hashes(transfer_root),
        "source_coverage": coverage,
        "optional_coefficient_resampling": "omitted; coefficient paths and coverage diagnostics are reported",
        "rolling_summary": rolling_summary,
        "frozen_summary": frozen_summary,
    }
    write_csv(output / "rolling_metrics.csv", rolling_rows)
    write_csv(output / "frozen_metrics.csv", [row for row in metric_rows_ if row["protocol"] == "frozen_through_2021"])
    write_csv(output / "rolling_comparisons.csv", comparisons)
    write_csv(output / "training_coverage.csv", coverage_rows_)
    write_csv(output / "coefficients.csv", coefficient_rows_)
    write_csv(output / "source_coverage.csv", coverage)
    write_json(output / "summary.json", summary)
    plot_outputs(output, metric_rows_, coefficient_rows_)
    render_report(
        output / "report.md",
        summary=summary,
        metric_rows_=metric_rows_,
        coverage=coverage_rows_,
        coefficients=coefficient_rows_,
        comparisons=comparisons,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument(
        "--transfer-root",
        type=Path,
        default=ROOT / "data/raw/cfbd/preseason/transfers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/transfer_production_stability",
    )
    args = parser.parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "study": summary["study"],
                "output": str(args.output),
                "rolling_summary": summary["rolling_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
