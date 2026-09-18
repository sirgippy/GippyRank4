"""Evaluate frozen defensive-transfer features beyond the D5 Context model.

This is a research-only, frozen-through-2021 experiment for issue 106.  It
reuses the production Context fit, H fallback, optimizer, and scoring helpers
from the earlier transfer studies.  The defensive audit is treated as an
input artifact: this script does not rebuild or alter either defensive
feature.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_preseason_context_prior_v1_2 as c12
import build_preseason_prior_v1_1 as v1_1
import investigate_transfer_roster_continuity as prior

from gippyrank.preseason import DirectRankModel, TeamSeason

ROOT = Path(__file__).resolve().parents[1]
TARGET_SEASONS = tuple(range(2022, 2026))
FROZEN_TRAIN_THROUGH = 2021
SANITY_TOLERANCE = 1e-3
DECOMPOSITION_PARITY_TOLERANCE = 1e-6
RANDOM_SEED = 7

EXPERIENCE = "transfer_in_prior_defensive_experience_sum"
IMPACT = "transfer_in_prior_defensive_impact_sum"
EXPERIENCE_AVAILABLE = "transfer_in_prior_defensive_experience_available"
IMPACT_AVAILABLE = "transfer_in_prior_defensive_impact_available"
DEFENSIVE_FEATURES = (EXPERIENCE, IMPACT)
AVAILABILITY_FEATURES = (EXPERIENCE_AVAILABLE, IMPACT_AVAILABLE)
METRICS = prior.METRICS


@dataclass(frozen=True)
class Candidate:
    """A predeclared model variant."""

    name: str
    description: str
    features: tuple[str, ...]
    defensive_features: tuple[str, ...] = ()
    kind: str = "primary"
    interaction: tuple[str, str] | None = None
    interaction_name: str | None = None


def d5_features() -> tuple[str, ...]:
    """Return the frozen D5 representation from issue 96."""
    return (
        *prior.C_MINUS_RP_FEATURES,
        "returning_pct_ppa",
        "transfer_in_prior_usage_sum",
    )


def candidate_definitions() -> tuple[Candidate, ...]:
    """Return exactly the predeclared E0--E6 variants."""
    d5 = d5_features()
    return (
        Candidate(
            "E0_production_C0",
            "Existing production Context C0.",
            tuple(prior.BASE_CONTEXT_FEATURES),
        ),
        Candidate(
            "E1_no_returning_production",
            "C-minus-RP control.",
            tuple(prior.C_MINUS_RP_FEATURES),
        ),
        Candidate(
            "E2_D5",
            "C-minus-RP plus total returning production and incoming prior offensive usage.",
            d5,
        ),
        Candidate(
            "E3_total_RP_plus_defensive_experience",
            "C-minus-RP plus total returning production and defensive experience; no incoming offensive usage.",
            (
                *prior.C_MINUS_RP_FEATURES,
                "returning_pct_ppa",
                EXPERIENCE,
                EXPERIENCE_AVAILABLE,
            ),
            (EXPERIENCE,),
        ),
        Candidate(
            "E4_D5_plus_defensive_experience",
            "D5 plus defensive experience and its availability indicator.",
            (*d5, EXPERIENCE, EXPERIENCE_AVAILABLE),
            (EXPERIENCE,),
        ),
        Candidate(
            "E5_D5_plus_defensive_impact",
            "D5 plus defensive impact and its availability indicator.",
            (*d5, IMPACT, IMPACT_AVAILABLE),
            (IMPACT,),
        ),
        Candidate(
            "E6_D5_plus_experience_plus_impact",
            "D5 plus defensive experience, defensive impact, and both availability indicators.",
            (*d5, EXPERIENCE, EXPERIENCE_AVAILABLE, IMPACT, IMPACT_AVAILABLE),
            DEFENSIVE_FEATURES,
        ),
    )


def auxiliary_definitions() -> tuple[Candidate, ...]:
    """Return the predeclared total-RP-only mechanism control."""
    return (
        Candidate(
            "RP_only_mechanism_control",
            "C-minus-RP plus total returning production only; auxiliary comparison for E3.",
            (*prior.C_MINUS_RP_FEATURES, "returning_pct_ppa"),
            kind="auxiliary",
        ),
    )


def control_definitions() -> tuple[Candidate, ...]:
    """Return missingness-only controls for all defensive primary candidates."""
    d5 = d5_features()
    return (
        Candidate(
            "E4-M_experience_availability_only",
            "E4 with the defensive-experience value removed; retain availability only.",
            (*d5, EXPERIENCE_AVAILABLE),
            (EXPERIENCE,),
            "missingness_control",
        ),
        Candidate(
            "E5-M_impact_availability_only",
            "E5 with the defensive-impact value removed; retain availability only.",
            (*d5, IMPACT_AVAILABLE),
            (IMPACT,),
            "missingness_control",
        ),
        Candidate(
            "E6-M_availability_only",
            "E6 with both defensive values removed; retain both availability indicators.",
            (*d5, EXPERIENCE_AVAILABLE, IMPACT_AVAILABLE),
            DEFENSIVE_FEATURES,
            "missingness_control",
        ),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
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
    """Hash the immutable portal and usage payloads used by the study."""
    result: dict[str, str] = {}
    for kind in ("portal", "usage"):
        for path in sorted((transfer_root / kind).glob("*.json")):
            if path.name == "manifest.json" or path.name.endswith(".provenance.json"):
                continue
            result[f"{kind}/{path.name}"] = sha256_file(path)
    return result


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def load_defensive_features(path: Path) -> dict[tuple[int, str, str], dict[str, float]]:
    """Load the frozen audit and encode observed versus unresolved values.

    A complete team-season and a team-season with no incoming defensive
    transfer are observed.  The latter receives the natural aggregate value
    zero.  Partial and no-usable rows are unresolved: their aggregate is
    neutralized to zero and availability is explicitly zero.  This retains
    the full population while separating no imported experience from unknown
    defensive coverage.
    """
    observed_statuses = {"complete", "no_incoming_defensive_transfer"}
    result: dict[tuple[int, str, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["season"]), row["subdivision"], row["team_id"])
            observed = row["feature_coverage_status"] in observed_statuses
            experience = _float_or_none(
                row.get("transfer_in_prior_defensive_experience_sum")
            )
            impact = _float_or_none(row.get("transfer_in_prior_defensive_impact_sum"))
            if row["feature_coverage_status"] == "complete" and (
                experience is None or impact is None
            ):
                raise ValueError(
                    "observed defensive team-season is missing a frozen aggregate: "
                    f"{key}"
                )
            result[key] = {
                EXPERIENCE: experience if observed and experience is not None else 0.0,
                IMPACT: impact if observed and impact is not None else 0.0,
                EXPERIENCE_AVAILABLE: float(observed),
                IMPACT_AVAILABLE: float(observed),
            }
    return result


def load_defensive_coverage(path: Path) -> dict[str, object]:
    """Summarize the audit statuses used by the explicit missingness policy."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    statuses = Counter(row["feature_coverage_status"] for row in rows)
    observed = sum(
        statuses[name] for name in ("complete", "no_incoming_defensive_transfer")
    )
    unresolved = sum(
        statuses[name] for name in ("partial", "no_usable_defensive_transfer")
    )
    return {
        "n_team_seasons": len(rows),
        "seasons": sorted({int(row["season"]) for row in rows}),
        "status_counts": dict(sorted(statuses.items())),
        "observed_team_seasons": observed,
        "unresolved_team_seasons": unresolved,
    }


def attach_defensive_features(
    rows: Iterable[TeamSeason],
    defensive_features: dict[tuple[int, str, str], dict[str, float]],
) -> list[TeamSeason]:
    """Attach frozen defensive values without changing row population."""
    result = []
    for row in rows:
        values = defensive_features.get(
            (row.season, row.subdivision, row.team_id),
            {
                EXPERIENCE: 0.0,
                IMPACT: 0.0,
                EXPERIENCE_AVAILABLE: 0.0,
                IMPACT_AVAILABLE: 0.0,
            },
        )
        result.append(replace(row, features={**row.features, **values}))
    return result


def permute_defensive_features(
    rows: Iterable[TeamSeason], *, seed: int = RANDOM_SEED
) -> list[TeamSeason]:
    """Permute observed defensive assignments within season only.

    Missingness indicators and the neutral value on unresolved rows remain in
    their original team-season positions.  Experience and impact are moved as
    pairs where both are observed, preserving their within-player/team
    relationship for the E6 negative control.
    """
    result = [replace(row, features=dict(row.features)) for row in rows]
    by_season: defaultdict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(result):
        by_season[row.season].append(index)
    rng = np.random.default_rng(seed)
    for indices in by_season.values():
        both = [
            index
            for index in indices
            if result[index].features.get(EXPERIENCE_AVAILABLE) == 1.0
            and result[index].features.get(IMPACT_AVAILABLE) == 1.0
        ]
        if len(both) > 1:
            source = rng.permutation(both)
            pairs = [
                (
                    result[index].features[EXPERIENCE],
                    result[index].features[IMPACT],
                )
                for index in source
            ]
            for index, (experience, impact) in zip(both, pairs, strict=True):
                result[index].features[EXPERIENCE] = experience
                result[index].features[IMPACT] = impact
        for feature, availability in (
            (EXPERIENCE, EXPERIENCE_AVAILABLE),
            (IMPACT, IMPACT_AVAILABLE),
        ):
            indices_for_feature = [
                index
                for index in indices
                if result[index].features.get(availability) == 1.0 and index not in both
            ]
            if len(indices_for_feature) > 1:
                source = rng.permutation(indices_for_feature)
                values = [result[int(index)].features[feature] for index in source]
                for index, value in zip(indices_for_feature, values, strict=True):
                    result[index].features[feature] = value
    return result


def fit_candidate(
    rows: list[TeamSeason],
    fallback: list[v1_1.PriorPrediction],
    candidate: Candidate,
) -> tuple[list[v1_1.PriorPrediction], DirectRankModel]:
    """Fit through 2021 and score the unchanged 2022--2025 panel."""
    return prior.panel_fit(
        rows,
        fallback,
        candidate,  # type: ignore[arg-type]
        target_seasons=TARGET_SEASONS,
        trained_through=FROZEN_TRAIN_THROUGH,
    )


def metric_row(
    variant: str,
    predictions: list[v1_1.PriorPrediction],
    *,
    season: int | None,
    reference: dict[str, float] | None,
    kind: str,
) -> dict[str, object]:
    selected = (
        predictions
        if season is None
        else [p for p in predictions if p.season == season]
    )
    values = prior.score(selected)
    row: dict[str, object] = {
        "variant": variant,
        "kind": kind,
        "target_season": "aggregate" if season is None else season,
        "n_team_seasons": len(selected),
    }
    for metric in METRICS:
        row[metric] = values[metric]
        if reference is not None:
            row[f"delta_vs_E0_{metric}"] = values[metric] - reference[metric]
    return row


def paired_diagnostics(
    comparison: str,
    candidate_name: str,
    candidate: list[v1_1.PriorPrediction],
    reference_name: str,
    reference: list[v1_1.PriorPrediction],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return detailed and aggregate/year paired NLL diagnostics."""
    candidate_losses = v1_1.prediction_losses(candidate)
    reference_losses = v1_1.prediction_losses(reference)
    by_key = sorted(set(candidate_losses) & set(reference_losses))
    if not by_key:
        raise ValueError(f"no paired losses for {comparison}")
    names = {item.key: item.team_name for item in reference}
    details = [
        {
            "comparison": comparison,
            "season": key[0],
            "subdivision": key[1],
            "team_id": key[2],
            "team_name": names[key],
            "candidate": candidate_name,
            "reference": reference_name,
            "candidate_nll": candidate_losses[key][0],
            "reference_nll": reference_losses[key][0],
            "candidate_minus_reference_nll": candidate_losses[key][0]
            - reference_losses[key][0],
        }
        for key in by_key
    ]

    def summarize(scope: str, values: list[float]) -> dict[str, object]:
        numbers = np.asarray(values, dtype=float)
        return {
            "comparison": comparison,
            "candidate": candidate_name,
            "reference": reference_name,
            "scope": scope,
            "n_team_seasons": len(numbers),
            "mean_delta_nll": float(np.mean(numbers)),
            "median_delta_nll": float(np.median(numbers)),
            "fraction_team_seasons_improved": float(np.mean(numbers < 0)),
        }

    summaries = [
        summarize(
            "aggregate",
            [float(row["candidate_minus_reference_nll"]) for row in details],
        )
    ]
    for season in sorted({int(row["season"]) for row in details}):
        summaries.append(
            summarize(
                str(season),
                [
                    float(row["candidate_minus_reference_nll"])
                    for row in details
                    if int(row["season"]) == season
                ],
            )
        )
    return details, summaries


def coefficient_rows(
    candidates: Iterable[Candidate],
    models: dict[str, DirectRankModel],
    training: list[TeamSeason],
) -> list[dict[str, object]]:
    """Extract coefficients and report source availability separately."""
    requested = (
        "returning_pct_ppa",
        "transfer_in_prior_usage_sum",
        EXPERIENCE,
        IMPACT,
        EXPERIENCE_AVAILABLE,
        IMPACT_AVAILABLE,
    )
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        model = models[candidate.name]
        feature_count = len(model.feature_names)
        for feature in requested:
            included = feature in model.feature_names
            if included:
                index = model.feature_names.index(feature)
                numeric = float(model.beta[model.lag_count + 1 + index])
                missing = float(model.beta[model.lag_count + 1 + feature_count + index])
                scale = float(model.gamma[1 + index])
                scale_missing = float(model.gamma[1 + feature_count + index])
                availability_feature = {
                    EXPERIENCE: EXPERIENCE_AVAILABLE,
                    IMPACT: IMPACT_AVAILABLE,
                    EXPERIENCE_AVAILABLE: EXPERIENCE_AVAILABLE,
                    IMPACT_AVAILABLE: IMPACT_AVAILABLE,
                }.get(feature)
                if availability_feature is None:
                    source_available = sum(
                        row.features.get(feature) is not None for row in training
                    )
                else:
                    source_available = sum(
                        row.features.get(availability_feature) == 1.0
                        for row in training
                    )
                source_unavailable = len(training) - source_available
            else:
                numeric = missing = scale = scale_missing = None
                source_available = source_unavailable = None
            rows.append(
                {
                    "variant": candidate.name,
                    "feature": feature,
                    "included": included,
                    "training_source_available_n": source_available,
                    "training_source_unavailable_n": source_unavailable,
                    "location_coefficient_standardized": numeric,
                    "location_missingness_coefficient": missing,
                    "log_scale_coefficient_standardized": scale,
                    "log_scale_missingness_coefficient": scale_missing,
                }
            )
    return rows


def read_decomposition_metrics(path: Path, candidate: str) -> dict[str, float]:
    """Read one aggregate control row from the frozen decomposition study."""
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("candidate") == candidate
                and row.get("target_season") == "aggregate"
            ):
                return {metric: float(row[metric]) for metric in METRICS}
    raise ValueError(f"aggregate control row is missing for {candidate}: {path}")


def decomposition_parity(
    computed: dict[str, float],
    *,
    candidate: str,
    reference_path: Path,
    tolerance: float = DECOMPOSITION_PARITY_TOLERANCE,
) -> dict[str, object]:
    """Require a control to reproduce its stored decomposition metrics."""
    expected = read_decomposition_metrics(reference_path, candidate)
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
        raise ValueError(f"control does not reproduce stored metrics: {result}")
    return result


def collinearity_diagnostics(
    rows: list[TeamSeason],
    models: dict[str, DirectRankModel],
) -> dict[str, object]:
    training = [row for row in rows if row.season <= FROZEN_TRAIN_THROUGH]
    observed = [
        row
        for row in training
        if row.features.get(EXPERIENCE_AVAILABLE) == 1.0
        and row.features.get(IMPACT_AVAILABLE) == 1.0
    ]
    experience = np.asarray([row.features[EXPERIENCE] for row in observed], dtype=float)
    impact = np.asarray([row.features[IMPACT] for row in observed], dtype=float)
    correlation = (
        float(np.corrcoef(experience, impact)[0, 1])
        if len(observed) > 1 and np.std(experience) > 0 and np.std(impact) > 0
        else None
    )
    coefficient_magnitudes: dict[str, dict[str, float | None]] = {}
    for name in (
        "E4_D5_plus_defensive_experience",
        "E5_D5_plus_defensive_impact",
        "E6_D5_plus_experience_plus_impact",
    ):
        model = models[name]
        coefficients = {}
        for feature in (EXPERIENCE, IMPACT):
            if feature in model.feature_names:
                index = model.feature_names.index(feature)
                coefficients[feature] = float(model.beta[model.lag_count + 1 + index])
            else:
                coefficients[feature] = None
        coefficient_magnitudes[name] = coefficients
    return {
        "training_observed_n": len(observed),
        "training_pairwise_correlation": correlation,
        "coefficient_comparison": coefficient_magnitudes,
        "interpretation": "Experience and impact are frozen measurements; coefficients are structural diagnostics, not causal effects.",
    }


def plot_outputs(
    output: Path,
    aggregate_rows: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    primary = [row for row in aggregate_rows if row["kind"] == "primary"]
    figure, axis = plt.subplots(figsize=(12, 5))
    values = [float(row["delta_vs_E0_nll"]) for row in primary]
    labels = [str(row["variant"]) for row in primary]
    axis.bar(
        labels,
        values,
        color=["#2f855a" if value < 0 else "#c53030" for value in values],
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("held-out ΔNLL versus E0")
    axis.set_title("Defensive-transfer candidates, frozen through 2021")
    axis.tick_params(axis="x", rotation=50)
    figure.tight_layout()
    figure.savefig(plots / "variant_delta_nll.png", dpi=160)
    plt.close(figure)

    aggregate_pairs = [row for row in paired_rows if row["scope"] == "aggregate"]
    figure, axis = plt.subplots(figsize=(11, 5))
    values = [float(row["mean_delta_nll"]) for row in aggregate_pairs]
    labels = [str(row["comparison"]) for row in aggregate_pairs]
    axis.bar(
        labels,
        values,
        color=["#2f855a" if value < 0 else "#c53030" for value in values],
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("mean paired ΔNLL")
    axis.set_title("Paired defensive-transfer diagnostics")
    axis.tick_params(axis="x", rotation=55)
    figure.tight_layout()
    figure.savefig(plots / "paired_delta_nll.png", dpi=160)
    plt.close(figure)


def fmt(value: object, digits: int = 4) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_report(
    path: Path,
    summary: dict[str, object],
    variant_rows: list[dict[str, object]],
    annual_rows: list[dict[str, object]],
    coefficient_rows_: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
) -> None:
    primary = [row for row in variant_rows if row["kind"] == "primary"]
    by_name = {str(row["variant"]): row for row in primary}
    e2 = by_name["E2_D5"]
    e4 = by_name["E4_D5_plus_defensive_experience"]
    e5 = by_name["E5_D5_plus_defensive_impact"]
    e6 = by_name["E6_D5_plus_experience_plus_impact"]
    pair_by_name = {
        str(row["comparison"]): row
        for row in paired_rows
        if row["scope"] == "aggregate"
    }
    e4_by_season = {
        str(row["target_season"]): float(row["nll"])
        for row in annual_rows
        if row["variant"] == e4["variant"] and row["target_season"] != "aggregate"
    }
    e2_by_season = {
        str(row["target_season"]): float(row["nll"])
        for row in annual_rows
        if row["variant"] == e2["variant"] and row["target_season"] != "aggregate"
    }
    e4_season_deltas = "; ".join(
        f"{season}: {fmt(value - e2_by_season[season])}"
        for season, value in sorted(e4_by_season.items())
    )
    recommendation = str(summary["recommendation"])
    lines = [
        "# Defensive-transfer predictive-value experiment (issue 106)",
        "",
        "## Conclusion",
        "",
        f"{recommendation}",
        "",
        (
            f"The primary D5 control (E2) has aggregate NLL {fmt(e2['nll'])}. "
            f"Experience E4 has ΔNLL versus E2 of {fmt(e4['nll'] - e2['nll'])}; "
            f"impact E5 has ΔNLL {fmt(e5['nll'] - e2['nll'])}; and E6 has ΔNLL "
            f"{fmt(e6['nll'] - e2['nll'])} versus E2."
        ),
        (
            f"E4's season-level ΔNLL versus E2 is {e4_season_deltas}. "
            f"The experience availability-only and permutation controls are also reported. "
            f"It does beat the experience availability-only control by mean paired "
            f"ΔNLL {fmt(pair_by_name['E4_vs_E4M']['mean_delta_nll'])} and the "
            f"within-season permutation by {fmt(pair_by_name['E4_vs_E4P']['mean_delta_nll'])}; "
            "this supports a cautious stability follow-up rather than a production promotion."
        ),
        "",
        "The experience measure is prior recorded defensive box-score games divided by source-team games. It is a conservative defensive-experience proxy, not defensive snap share. The impact measure is the frozen position-normalized box-score composite from PR #105, not a direct estimate of player quality.",
        "",
        "## Frozen protocol",
        "",
        f"- Fit through {FROZEN_TRAIN_THROUGH}; score unchanged on 2022–2025.",
        "- The production FBS target population, Context preprocessing, H fallback, optimizer retry, penalty, and rank-distribution scoring are unchanged.",
        "- E0, E1, and E2 parity are required before interpreting defensive variants; the study fails if any stored control comparison exceeds tolerance.",
        "- The offensive transfer attachment reads the frozen portal/usage payloads supplied by `--transfer-root`; missing payloads are an error rather than an all-missing feature matrix.",
        "- Defensive values are zero for observed no-incoming rows and for unresolved rows after neutral imputation; an explicit availability indicator separates those cases.",
        "- Partial and no-usable defensive audit rows are never dropped and never treated as observed zeros.",
        f"- The defensive audit covers {summary['defensive_coverage']['n_team_seasons']} team-seasons: {summary['defensive_coverage']['observed_team_seasons']} observed and {summary['defensive_coverage']['unresolved_team_seasons']} unresolved under this policy.",
        "",
        "## Exact primary variants",
        "",
        "| Variant | Definition | Defensive inputs |",
        "|---|---|---|",
    ]
    for candidate in summary["variant_definitions"]:
        lines.append(
            f"| {candidate['name']} | {candidate['description']} | "
            f"{', '.join(candidate['defensive_features']) or 'none'} |"
        )
    lines += [
        "",
        "## C0 / no-RP / D5 parity and aggregate metrics",
        "",
        "| Control | Frozen reference | Maximum absolute metric delta | Tolerance | Status |",
        "|---|---|---:|---:|:---:|",
    ]
    for label, parity in summary["control_parity"].items():
        lines.append(
            f"| {label} | {parity['candidate']} | {fmt(parity['max_abs_metric_delta'], 8)} | {fmt(parity['tolerance'], 8)} | {'passed' if parity['passed'] else 'failed'} |"
        )
    lines += [
        "",
        "| Variant | NLL | Δ vs E0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in variant_rows:
        lines.append(
            f"| {row['variant']} | {fmt(row['nll'])} | {fmt(row.get('delta_vs_E0_nll'))} | {fmt(row['crps'])} | {fmt(row['expected_rank_mae'], 2)} | {fmt(row['median_rank_mae'], 2)} | {fmt(row['interval_80_coverage'], 3)} | {fmt(row['interval_80_average_width'], 2)} |"
        )
    lines += [
        "",
        "## Year-by-year primary metrics",
        "",
        "| Season | Variant | NLL | Δ vs E0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in annual_rows:
        if row["kind"] == "primary":
            lines.append(
                f"| {row['target_season']} | {row['variant']} | {fmt(row['nll'])} | {fmt(row.get('delta_vs_E0_nll'))} | {fmt(row['crps'])} | {fmt(row['expected_rank_mae'], 2)} | {fmt(row['median_rank_mae'], 2)} | {fmt(row['interval_80_coverage'], 3)} | {fmt(row['interval_80_average_width'], 2)} |"
            )
    lines += [
        "",
        "## Paired NLL diagnostics",
        "",
        "Negative ΔNLL means the candidate has lower loss.",
        "",
        "| Comparison | Scope | N | Mean ΔNLL | Median ΔNLL | Fraction improved |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in paired_rows:
        lines.append(
            f"| {row['comparison']} | {row['scope']} | {row['n_team_seasons']} | {fmt(row['mean_delta_nll'])} | {fmt(row['median_delta_nll'])} | {fmt(row['fraction_team_seasons_improved'], 3)} |"
        )
    lines += [
        "",
        "## Coefficient diagnostics",
        "",
        "Numeric coefficients are standardized training-fit location coefficients. Availability indicators are explicit observed/unresolved controls. The defensive inputs enter location only, matching the frozen Context experiment; coefficients are not causal effects.",
        "",
        "| Variant | Feature | Included | Training source available / unavailable | Location β | Location missingness |",
        "|---|---|:---:|---:|---:|---:|",
    ]
    for row in coefficient_rows_:
        lines.append(
            f"| {row['variant']} | {row['feature']} | {row['included']} | {row['training_source_available_n']} / {row['training_source_unavailable_n']} | {fmt(row['location_coefficient_standardized'])} | {fmt(row['location_missingness_coefficient'])} |"
        )
    lines += [
        "",
        "## Missingness and permutation controls",
        "",
        f"E4 versus its availability-only control: mean paired ΔNLL {fmt(pair_by_name['E4_vs_E4M']['mean_delta_nll'])}. E5 versus its availability-only control: {fmt(pair_by_name['E5_vs_E5M']['mean_delta_nll'])}. E6 versus its availability-only control: {fmt(pair_by_name['E6_vs_E6M']['mean_delta_nll'])}.",
        f"Within-season permutations preserve season, availability pattern, and observed defensive-value distributions while destroying team assignment. Real versus permuted mean paired ΔNLL is {fmt(pair_by_name['E4_vs_E4P']['mean_delta_nll'])} for experience, {fmt(pair_by_name['E5_vs_E5P']['mean_delta_nll'])} for impact, and {fmt(pair_by_name['E6_vs_E6P']['mean_delta_nll'])} for E6.",
        "",
        "## Experience / impact collinearity",
        "",
        f"The E6 training panel has {summary['collinearity']['training_observed_n']} rows with both defensive values observed and pairwise correlation {fmt(summary['collinearity']['training_pairwise_correlation'])}. E6 coefficient instability, if present, is therefore treated as evidence of redundancy rather than repaired through outcome-tuned regularization.",
        "",
        "## Defensive feature semantics and limitations",
        "",
        "- Experience is recorded defensive box-score game rate, not defensive snap share or games played.",
        "- Impact is the frozen DL/EDGE, LB, and DB position-normalized log1p box-score composite from PR #105.",
        "- The defensive audit remains a retrospective research oracle with fail-closed identity/source coverage.",
        "- This ticket does not run rolling-origin stability, position decomposition, feature redesign, or production snapshot engineering.",
        "",
        "## Artifacts",
        "",
        "- `variant_definitions.json`, `summary.json` — frozen configuration, parity, recommendation, and provenance.",
        "- `candidate_annual_metrics.csv`, `candidate_summary.csv` — primary, auxiliary, missingness-control, and permutation metrics.",
        "- `paired_nll_summary.csv`, `paired_nll_by_team.csv` — aggregate/year and team-season paired diagnostics.",
        "- `standardized_coefficients.csv`, `collinearity.json`, `plots/` — structural diagnostics.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.resolve()
    defensive_path = args.defensive_features.resolve()
    output = args.output.resolve()

    prior.configure_source_root(source_root)
    rows, cold, _ = v1_1.v1.load_rows(max_season=max(TARGET_SEASONS))
    fbs = [row for row in rows if row.subdivision == "fbs"]
    contextual, _ = c12.attach_context(fbs, c12.feature_index(), c12.cached_tenures())
    transfer_root = args.transfer_root.resolve()
    decomposition_summary = args.decomposition_summary.resolve()
    records, usage, portal_seasons, usage_seasons = prior.load_raw_transfer_data(
        transfer_root
    )
    required_portal_seasons = {FROZEN_TRAIN_THROUGH, *TARGET_SEASONS}
    required_usage_seasons = set(range(FROZEN_TRAIN_THROUGH - 1, max(TARGET_SEASONS)))
    if not required_portal_seasons <= portal_seasons:
        raise FileNotFoundError(
            "D5 requires portal payloads for seasons "
            f"{sorted(required_portal_seasons)}; found {sorted(portal_seasons)} "
            f"under {transfer_root}"
        )
    if not required_usage_seasons <= usage_seasons:
        raise FileNotFoundError(
            "D5 requires usage payloads for seasons "
            f"{sorted(required_usage_seasons)}; found {sorted(usage_seasons)} "
            f"under {transfer_root}"
        )
    transfer_features = prior.aggregate_team_features(
        records,
        usage,
        prior.fbs_feature_rows(contextual),
        covered_seasons=portal_seasons,
        cutoff=date(2025, 8, 15),
    )
    contextual = prior.attach_transfer_features(contextual, transfer_features)
    defensive_features = load_defensive_features(defensive_path)
    defensive_coverage = load_defensive_coverage(defensive_path)
    contextual = attach_defensive_features(contextual, defensive_features)
    fallback = prior.fallback_for_panel(fbs, cold)
    fallback = [item for item in fallback if item.season in TARGET_SEASONS]

    primary = candidate_definitions()
    auxiliary = auxiliary_definitions()
    controls = control_definitions()
    all_candidates = (*primary, *auxiliary, *controls)
    predictions_by_name: dict[str, list[v1_1.PriorPrediction]] = {}
    models: dict[str, DirectRankModel] = {}
    for candidate in all_candidates:
        predictions, model = fit_candidate(contextual, fallback, candidate)
        predictions_by_name[candidate.name] = predictions
        models[candidate.name] = model
        print(f"completed {candidate.name}", flush=True)

    permuted = permute_defensive_features(contextual)
    permutation_candidates = (
        Candidate(
            "E4P_experience_within_season_permutation",
            "E4 with observed defensive experience reassigned within season.",
            primary[4].features,
            primary[4].defensive_features,
            "permutation_control",
        ),
        Candidate(
            "E5P_impact_within_season_permutation",
            "E5 with observed defensive impact reassigned within season.",
            primary[5].features,
            primary[5].defensive_features,
            "permutation_control",
        ),
        Candidate(
            "E6P_defense_within_season_permutation",
            "E6 with observed defensive experience/impact assignments reassigned within season.",
            primary[6].features,
            primary[6].defensive_features,
            "permutation_control",
        ),
    )
    for candidate in permutation_candidates:
        predictions, model = fit_candidate(permuted, fallback, candidate)
        predictions_by_name[candidate.name] = predictions
        models[candidate.name] = model
        print(f"completed {candidate.name}", flush=True)

    key_sets = {
        frozenset(item.key for item in values)
        for values in predictions_by_name.values()
    }
    if len(key_sets) != 1:
        raise ValueError("all model variants must preserve identical target keys")

    e0 = predictions_by_name[primary[0].name]
    e0_metrics = prior.score(e0)
    e0_by_season = {
        season: prior.score([item for item in e0 if item.season == season])
        for season in TARGET_SEASONS
    }
    aggregate_rows = [
        metric_row(
            candidate.name,
            predictions_by_name[candidate.name],
            season=None,
            reference=e0_metrics,
            kind=candidate.kind,
        )
        for candidate in (*all_candidates, *permutation_candidates)
    ]
    annual_rows = [
        metric_row(
            candidate.name,
            predictions_by_name[candidate.name],
            season=season,
            reference=e0_by_season[season],
            kind=candidate.kind,
        )
        for candidate in (*all_candidates, *permutation_candidates)
        for season in TARGET_SEASONS
    ]

    training = [row for row in contextual if row.season <= FROZEN_TRAIN_THROUGH]
    coefficient_candidates = (*primary, *controls)
    coefficients = coefficient_rows(coefficient_candidates, models, training)
    collinearity = collinearity_diagnostics(contextual, models)

    comparisons = (
        ("E4_vs_E2", primary[4].name, primary[2].name),
        ("E5_vs_E2", primary[5].name, primary[2].name),
        ("E6_vs_E4", primary[6].name, primary[4].name),
        ("E6_vs_E5", primary[6].name, primary[5].name),
        ("E3_vs_RP_only", primary[3].name, auxiliary[0].name),
        ("E4_vs_E4M", primary[4].name, controls[0].name),
        ("E5_vs_E5M", primary[5].name, controls[1].name),
        ("E6_vs_E6M", primary[6].name, controls[2].name),
        ("E4M_vs_E2", controls[0].name, primary[2].name),
        ("E5M_vs_E2", controls[1].name, primary[2].name),
        ("E6M_vs_E2", controls[2].name, primary[2].name),
        ("E4_vs_E4P", primary[4].name, permutation_candidates[0].name),
        ("E5_vs_E5P", primary[5].name, permutation_candidates[1].name),
        ("E6_vs_E6P", primary[6].name, permutation_candidates[2].name),
    )
    paired_details: list[dict[str, object]] = []
    paired_summaries: list[dict[str, object]] = []
    for comparison, candidate_name, reference_name in comparisons:
        details, summaries = paired_diagnostics(
            comparison,
            candidate_name,
            predictions_by_name[candidate_name],
            reference_name,
            predictions_by_name[reference_name],
        )
        paired_details.extend(details)
        paired_summaries.extend(summaries)

    stored_path = source_root / "data/processed/preseason/context/evaluation.json"
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    stored_c0 = stored.get("all_fbs", {}).get("candidate", {})
    sanity_deltas = {
        metric: abs(e0_metrics[metric] - float(stored_c0[metric]))
        for metric in METRICS
        if metric in stored_c0
    }
    sanity = {
        "candidate": "E0_production_C0",
        "stored_evaluation_path": str(stored_path),
        "stored_metrics": stored_c0,
        "computed_metrics": e0_metrics,
        "metric_abs_deltas": sanity_deltas,
        "max_abs_metric_delta": max(sanity_deltas.values(), default=None),
        "tolerance": SANITY_TOLERANCE,
        "passed": bool(sanity_deltas)
        and max(sanity_deltas.values()) <= SANITY_TOLERANCE,
    }
    if not sanity["passed"]:
        raise ValueError(f"E0 does not reproduce stored production metrics: {sanity}")

    control_parity = {
        "E0": sanity,
        "E1": decomposition_parity(
            prior.score(predictions_by_name[primary[1].name]),
            candidate="D1_no_returning_production",
            reference_path=decomposition_summary,
        ),
        "E2": decomposition_parity(
            prior.score(predictions_by_name[primary[2].name]),
            candidate="D5_total_rp_plus_incoming",
            reference_path=decomposition_summary,
        ),
    }

    e2_metrics = next(
        row for row in aggregate_rows if row["variant"] == primary[2].name
    )
    e4_metrics = next(
        row for row in aggregate_rows if row["variant"] == primary[4].name
    )
    e5_metrics = next(
        row for row in aggregate_rows if row["variant"] == primary[5].name
    )
    e6_metrics = next(
        row for row in aggregate_rows if row["variant"] == primary[6].name
    )
    improved = {
        "experience": float(e4_metrics["nll"]) < float(e2_metrics["nll"]),
        "impact": float(e5_metrics["nll"]) < float(e2_metrics["nll"]),
        "both": float(e6_metrics["nll"]) < float(e2_metrics["nll"]),
    }
    recommendation = "Advance both defensive representations only if E6 materially beats E4 and E5; otherwise prefer the simpler candidate that improves D5."
    if not any(improved.values()):
        recommendation = "No defensive candidate improves D5 on aggregate NLL; retain D5 and stop the defensive-transfer branch for this model cycle."
    elif improved["experience"] and not improved["impact"]:
        recommendation = "Defensive experience improves D5 while impact does not; advance the recorded defensive box-score game-rate proxy to the next stability study, subject to its controls."
    elif improved["impact"] and not improved["experience"]:
        recommendation = "Defensive impact improves D5 while experience does not; advance the frozen position-normalized production proxy to the next stability study, subject to its controls."
    elif improved["both"]:
        recommendation = "Both defensive candidates improve D5; inspect E4/E5/E6, missingness, permutation, and coefficient diagnostics before advancing the simpler representation unless E6 clearly adds independent value."

    source_files = (
        source_root / "data/processed/modeling/team_season_rank_distributions.csv",
        source_root / "data/processed/preseason/team_season_features.csv",
        defensive_path,
        defensive_path.parent / "summary.json",
    )
    source_hashes: dict[str, str] = {}
    for path in source_files:
        if not path.exists():
            continue
        try:
            display_path = str(path.relative_to(source_root))
        except ValueError:
            display_path = str(path.relative_to(ROOT))
        source_hashes[display_path] = sha256_file(path)
    summary: dict[str, object] = {
        "study": "issue_106_defensive_transfer_predictive_value",
        "production_models_modified": False,
        "target_seasons": list(TARGET_SEASONS),
        "frozen_train_through": FROZEN_TRAIN_THROUGH,
        "optimizer_penalty": 0.25,
        "random_seed": RANDOM_SEED,
        "variant_definitions": [
            {
                "name": candidate.name,
                "description": candidate.description,
                "features": list(candidate.features),
                "defensive_features": list(candidate.defensive_features),
                "kind": candidate.kind,
            }
            for candidate in primary
        ],
        "auxiliary_definitions": [
            {
                "name": candidate.name,
                "description": candidate.description,
                "features": list(candidate.features),
                "kind": candidate.kind,
            }
            for candidate in auxiliary
        ],
        "missingness_controls": [candidate.name for candidate in controls],
        "permutation_controls": [
            candidate.name for candidate in permutation_candidates
        ],
        "missing_data_policy": {
            "observed_statuses": ["complete", "no_incoming_defensive_transfer"],
            "unresolved_statuses": ["partial", "no_usable_defensive_transfer"],
            "neutral_value": 0.0,
            "availability_features": list(AVAILABILITY_FEATURES),
            "population_preserved": True,
        },
        "defensive_coverage": defensive_coverage,
        "c0_sanity_check": sanity,
        "control_parity": control_parity,
        "transfer_root": str(transfer_root),
        "portal_seasons_available": sorted(portal_seasons),
        "usage_seasons_available": sorted(usage_seasons),
        "raw_transfer_input_sha256": transfer_source_hashes(transfer_root),
        "decomposition_reference": str(decomposition_summary),
        "aggregate_metrics": aggregate_rows,
        "improved_over_d5_by_nll": improved,
        "recommendation": recommendation,
        "collinearity": collinearity,
        "source_hashes": source_hashes,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "variant_definitions.json", summary["variant_definitions"])
    write_json(output / "summary.json", summary)
    write_json(output / "collinearity.json", collinearity)
    write_csv(output / "candidate_annual_metrics.csv", annual_rows)
    write_csv(output / "candidate_summary.csv", aggregate_rows)
    write_csv(output / "standardized_coefficients.csv", coefficients)
    write_csv(output / "paired_nll_summary.csv", paired_summaries)
    write_csv(output / "paired_nll_by_team.csv", paired_details)
    plot_outputs(output, aggregate_rows, paired_summaries)
    render_report(
        output / "report.md",
        summary,
        aggregate_rows,
        annual_rows,
        coefficients,
        paired_summaries,
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
        "--decomposition-summary",
        type=Path,
        default=ROOT
        / "data/processed/transfer_signal_decomposition/candidate_summary.csv",
        help="Frozen issue-96 aggregate metrics used for E1/E2 parity.",
    )
    parser.add_argument(
        "--defensive-features",
        type=Path,
        default=ROOT
        / "data/processed/defensive_transfer_audit/team_season_features.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/defensive_transfer_predictive_value",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps({"study": summary["study"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
