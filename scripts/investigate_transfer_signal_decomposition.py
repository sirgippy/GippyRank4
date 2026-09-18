"""Research-only decomposition of the C10 transfer-production signal.

This issue-96 study reuses the frozen input construction and evaluation helpers
from the issue-91 transfer oracle study, but evaluates only the predeclared D0
through D8 feature sets.  It does not alter production models or raw inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import investigate_transfer_roster_continuity as prior

from gippyrank.preseason import DirectRankModel, TeamSeason

ROOT = Path(__file__).resolve().parents[1]
TARGET_SEASONS = tuple(range(2022, 2026))
FROZEN_TRAIN_THROUGH = 2021
DEFAULT_CUTOFF = (8, 15)
SANITY_TOLERANCE = 1e-3
METRICS = prior.METRICS

TOTAL_RP = ("returning_pct_ppa",)
INCOMING_PRODUCTION = ("transfer_in_prior_usage_sum",)
OUTGOING_PRODUCTION = ("transfer_out_prior_usage_sum",)
EXISTING_C10_PRODUCTION = (
    "transfer_in_prior_usage_sum",
    "transfer_net_prior_usage",
)


@dataclass(frozen=True)
class Candidate:
    """A predeclared feature set and its continuity-feature description."""

    name: str
    description: str
    features: tuple[str, ...]
    continuity_features: tuple[str, ...]
    transfer_aware: bool = False
    interaction: tuple[str, str] | None = None
    interaction_name: str | None = None


def candidate_definitions() -> tuple[Candidate, ...]:
    """Return the complete D0--D8 set before any held-out result is seen."""
    base = tuple(prior.BASE_CONTEXT_FEATURES)
    minus_rp = tuple(prior.C_MINUS_RP_FEATURES)
    return (
        Candidate(
            "D0_C0_full",
            "Production Context C 1.2; the parity control.",
            base,
            (),
        ),
        Candidate(
            "D1_no_returning_production",
            "C-minus-RP; remove all returning-production features.",
            minus_rp,
            (),
        ),
        Candidate(
            "D2_total_rp_only",
            "C-minus-RP plus total returning production only.",
            (*minus_rp, *TOTAL_RP),
            TOTAL_RP,
        ),
        Candidate(
            "D3_incoming_transfer_production_only",
            "C-minus-RP plus incoming prior transfer production only.",
            (*minus_rp, *INCOMING_PRODUCTION),
            INCOMING_PRODUCTION,
            True,
        ),
        Candidate(
            "D4_outgoing_transfer_production_only",
            "C-minus-RP plus outgoing prior transfer production only.",
            (*minus_rp, *OUTGOING_PRODUCTION),
            OUTGOING_PRODUCTION,
            True,
        ),
        Candidate(
            "D5_total_rp_plus_incoming",
            "C-minus-RP plus total RP and incoming prior transfer production.",
            (*minus_rp, *TOTAL_RP, *INCOMING_PRODUCTION),
            (*TOTAL_RP, *INCOMING_PRODUCTION),
            True,
        ),
        Candidate(
            "D6_total_rp_plus_outgoing",
            "C-minus-RP plus total RP and outgoing prior transfer production.",
            (*minus_rp, *TOTAL_RP, *OUTGOING_PRODUCTION),
            (*TOTAL_RP, *OUTGOING_PRODUCTION),
            True,
        ),
        Candidate(
            "D7_total_rp_plus_incoming_plus_outgoing",
            "C-minus-RP plus total RP, incoming, and outgoing prior production.",
            (*minus_rp, *TOTAL_RP, *INCOMING_PRODUCTION, *OUTGOING_PRODUCTION),
            (*TOTAL_RP, *INCOMING_PRODUCTION, *OUTGOING_PRODUCTION),
            True,
        ),
        Candidate(
            "D8_existing_c10",
            "Existing C10: full RP family plus incoming and net prior production.",
            (*base, *EXISTING_C10_PRODUCTION),
            (*prior.RP_FEATURES, *EXISTING_C10_PRODUCTION),
            True,
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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_candidate(
    rows: list[TeamSeason],
    fallback: list[prior.v1.PriorPrediction],
    candidate: Candidate,
) -> tuple[list[prior.v1.PriorPrediction], DirectRankModel]:
    """Fit a candidate through 2021 and score the unchanged 2022--2025 panel."""
    # The existing panel helper only requires these four Candidate attributes.
    return prior.panel_fit(  # type: ignore[arg-type]
        rows,
        fallback,
        candidate,
        target_seasons=TARGET_SEASONS,
        trained_through=FROZEN_TRAIN_THROUGH,
    )


def metric_row(
    candidate: str,
    predictions: list[prior.v1.PriorPrediction],
    *,
    season: int | None,
    reference: dict[str, float] | None = None,
) -> dict[str, object]:
    selected = (
        predictions
        if season is None
        else [item for item in predictions if item.season == season]
    )
    values = prior.score(selected)
    row: dict[str, object] = {
        "candidate": candidate,
        "target_season": "aggregate" if season is None else season,
        "n_team_seasons": len(selected),
    }
    for metric in METRICS:
        row[metric] = values[metric]
        if reference is not None:
            row[f"delta_vs_D0_{metric}"] = values[metric] - reference[metric]
    return row


def standardized_coefficients(
    candidate: Candidate, model: DirectRankModel
) -> list[dict[str, object]]:
    """Extract location/scale coefficients for already-standardized inputs.

    ``DirectRankModel`` stores numeric coefficients after training-only
    standardization, followed by separate missingness-indicator coefficients.
    The latter are reported explicitly because they are not standardized
    numeric effects.
    """
    feature_count = len(model.feature_names)
    rows: list[dict[str, object]] = []
    for feature in candidate.continuity_features:
        index = model.feature_names.index(feature)
        preprocessor = model.preprocessor
        rows.append(
            {
                "candidate": candidate.name,
                "feature": feature,
                "input_mean": preprocessor.means[feature],
                "input_scale": preprocessor.scales[feature],
                "location_coefficient_standardized": float(
                    model.beta[model.lag_count + 1 + index]
                ),
                "location_missingness_coefficient": float(
                    model.beta[model.lag_count + 1 + feature_count + index]
                ),
                "log_scale_coefficient_standardized": float(model.gamma[1 + index]),
                "log_scale_missingness_coefficient": float(
                    model.gamma[1 + feature_count + index]
                ),
            }
        )
    return rows


def add_training_counts(
    rows: list[dict[str, object]], training: list[TeamSeason]
) -> None:
    for row in rows:
        feature = str(row["feature"])
        values = [item.features.get(feature) for item in training]
        row["training_observed_n"] = sum(value is not None for value in values)
        row["training_missing_n"] = sum(value is None for value in values)


def paired_rows(
    comparison: str,
    candidate_name: str,
    candidate: list[prior.v1.PriorPrediction],
    reference_name: str,
    reference: list[prior.v1.PriorPrediction],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return team-season and aggregate paired NLL diagnostics."""
    candidate_losses = prior.h11.prediction_losses(candidate)
    reference_losses = prior.h11.prediction_losses(reference)
    keys = sorted(set(candidate_losses) & set(reference_losses))
    if not keys:
        raise ValueError(f"no paired losses available for {comparison}")

    def summarize(scope: str, values: list[float]) -> dict[str, object]:
        deltas = np.asarray(values, dtype=float)
        return {
            "comparison": comparison,
            "candidate": candidate_name,
            "reference": reference_name,
            "scope": scope,
            "n_team_seasons": len(deltas),
            "mean_delta_nll": float(np.mean(deltas)),
            "median_delta_nll": float(np.median(deltas)),
            "fraction_team_seasons_improved": float(np.mean(deltas < 0)),
        }

    by_season: dict[int, list[float]] = {}
    for key in keys:
        by_season.setdefault(key[0], []).append(
            candidate_losses[key][0] - reference_losses[key][0]
        )
    rows = [summarize("aggregate", [by_season[year][i] for year in by_season for i in range(len(by_season[year]))])]
    for season in sorted(by_season):
        rows.append(summarize(str(season), by_season[season]))
    bootstrap = prior.paired_season_bootstrap(reference, candidate)
    aggregate = dict(rows[0])
    aggregate["season_bootstrap"] = bootstrap
    return rows, aggregate


def comparison_summary(
    comparison: str,
    candidate_name: str,
    candidate: list[prior.v1.PriorPrediction],
    reference_name: str,
    reference: list[prior.v1.PriorPrediction],
) -> dict[str, object]:
    candidate_score = prior.score(candidate)
    reference_score = prior.score(reference)
    return {
        "comparison": comparison,
        "candidate": candidate_name,
        "reference": reference_name,
        "n_team_seasons": len(candidate),
        **{
            f"delta_{metric}": candidate_score[metric] - reference_score[metric]
            for metric in METRICS
        },
    }


def plot_outputs(
    output: Path,
    aggregate_rows: list[dict[str, object]],
    coefficient_rows: list[dict[str, object]],
    coverage: list[dict[str, object]],
) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 5))
    values = [float(row["delta_vs_D0_nll"]) for row in aggregate_rows]
    labels = [str(row["candidate"]) for row in aggregate_rows]
    axis.bar(labels, values, color=["#2f855a" if value < 0 else "#c53030" for value in values])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("held-out ΔNLL versus D0")
    axis.set_title("C10 transfer-production decomposition, frozen through 2021")
    axis.tick_params(axis="x", rotation=55)
    figure.tight_layout()
    figure.savefig(plots / "variant_delta_nll.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    for candidate in sorted({str(row["candidate"]) for row in coefficient_rows}):
        part = [row for row in coefficient_rows if row["candidate"] == candidate]
        axis.plot(
            [str(row["feature"]) for row in part],
            [float(row["location_coefficient_standardized"]) for row in part],
            marker="o",
            label=candidate,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("standardized location coefficient")
    axis.set_title("Roster-continuity location coefficients")
    axis.tick_params(axis="x", rotation=55)
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(plots / "standardized_location_coefficients.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(
        [int(row["season"]) for row in coverage],
        [float(row["n_on_or_before_cutoff"]) for row in coverage],
        marker="o",
    )
    axis.set_title("Portal records on or before August 15 cutoff")
    axis.set_xlabel("transfer season")
    axis.set_ylabel("records")
    figure.tight_layout()
    figure.savefig(plots / "portal_coverage_by_season.png", dpi=160)
    plt.close(figure)


def fmt(value: object, digits: int = 4) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_report(
    path: Path,
    summary: dict[str, Any],
    annual_rows: list[dict[str, object]],
    aggregate_rows: list[dict[str, object]],
    coefficient_rows: list[dict[str, object]],
    paired_rows_: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    coverage: list[dict[str, object]],
) -> None:
    by_name = {str(row["candidate"]): row for row in aggregate_rows}
    d5_d2 = next(
        row for row in comparison_rows if row["comparison"] == "incoming_conditional_on_rp"
    )
    d7_d5 = next(
        row
        for row in comparison_rows
        if row["comparison"] == "outgoing_beyond_rp_plus_incoming"
    )
    d5_d8 = next(row for row in comparison_rows if row["comparison"] == "simplified_vs_c10")
    d7_d5_paired = next(
        row for row in summary["paired_comparisons"] if row["comparison"] == "D7_vs_D5"
    )
    d3_d1 = next(row for row in comparison_rows if row["comparison"] == "incoming_unconditional")
    d4_d1 = next(row for row in comparison_rows if row["comparison"] == "outgoing_unconditional")
    d6_d2 = next(row for row in comparison_rows if row["comparison"] == "outgoing_conditional_on_rp")
    lines = [
        "# Transfer-production signal decomposition (issue 96)",
        "",
        "## Conclusion",
        "",
        (
            f"D0 reproduced the stored production C 1.2 panel within the configured "
            f"{fmt(summary['c0_sanity_check']['tolerance'], 3)} tolerance. "
            f"The primary candidate D5 (total RP + incoming prior production) has "
            f"held-out ΔNLL {fmt(by_name['D5_total_rp_plus_incoming']['delta_vs_D0_nll'])} "
            f"versus D0; D8 (existing C10) has ΔNLL "
            f"{fmt(by_name['D8_existing_c10']['delta_vs_D0_nll'])}."
        ),
        (
            f"Incoming production conditional on total RP (D5 − D2) has mean paired "
            f"ΔNLL {fmt(d5_d2['delta_nll'])}; outgoing production beyond total RP + "
            f"incoming (D7 − D5) has mean paired ΔNLL {fmt(d7_d5['delta_nll'])} "
            f"and improves {fmt(d7_d5_paired['fraction_team_seasons_improved'], 3)} of team-seasons. "
            f"D5 versus D8 has mean paired ΔNLL {fmt(d5_d8['delta_nll'])}."
        ),
        (
            f"Incoming production also improves the no-RP control (D3 − D1 ΔNLL "
            f"{fmt(d3_d1['delta_nll'])}), while outgoing production alone does not "
            f"(D4 − D1 ΔNLL {fmt(d4_d1['delta_nll'])}). Outgoing production has a "
            f"small conditional comparison against total RP (D6 − D2 ΔNLL "
            f"{fmt(d6_d2['delta_nll'])}), but adding it after both RP and incoming "
            f"production is slightly worse. The recommended representation for the "
            f"subsequent coefficient-stability study is therefore D5: total RP plus "
            "incoming prior transfer production, subject to the oracle caveat."
        ),
        "These are retrospective-oracle mechanism results, not a production-safe feature recommendation.",
        "",
        "## Exact predeclared variants",
        "",
        "| Variant | Definition | Added continuity features |",
        "|---|---|---|",
    ]
    for candidate in summary["variant_definitions"]:
        lines.append(
            f"| {candidate['name']} | {candidate['description']} | "
            f"{', '.join(candidate['continuity_features']) or 'none'} |"
        )
    lines += [
        "",
        "## C0 parity check",
        "",
        f"D0 is the existing C 1.2 control. The check {'passed' if summary['c0_sanity_check']['passed'] else 'failed'} with maximum absolute metric difference {fmt(summary['c0_sanity_check']['max_abs_metric_delta'], 6)}.",
        "",
        "## Aggregate 2022–2025 comparison",
        "",
        "| Variant | NLL | Δ vs D0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['candidate']} | {fmt(row['nll'])} | {fmt(row['delta_vs_D0_nll'])} | {fmt(row['crps'])} | {fmt(row['expected_rank_mae'], 2)} | {fmt(row['median_rank_mae'], 2)} | {fmt(row['interval_80_coverage'], 3)} | {fmt(row['interval_80_average_width'], 2)} |"
        )
    lines += [
        "",
        "## Year-by-year metrics",
        "",
        "| Season | Variant | NLL | Δ vs D0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in annual_rows:
        lines.append(
            f"| {row['target_season']} | {row['candidate']} | {fmt(row['nll'])} | {fmt(row['delta_vs_D0_nll'])} | {fmt(row['crps'])} | {fmt(row['expected_rank_mae'], 2)} | {fmt(row['median_rank_mae'], 2)} | {fmt(row['interval_80_coverage'], 3)} | {fmt(row['interval_80_average_width'], 2)} |"
        )
    lines += [
        "",
        "## Required pairwise comparisons",
        "",
        "Negative ΔNLL means the first variant has lower loss than the second.",
        "",
        "| Comparison | Candidate | Reference | ΔNLL | ΔCRPS | Δ expected-rank MAE | Δ median-rank MAE | Δ 80% coverage |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['comparison']} | {row['candidate']} | {row['reference']} | {fmt(row['delta_nll'])} | {fmt(row['delta_crps'])} | {fmt(row['delta_expected_rank_mae'], 2)} | {fmt(row['delta_median_rank_mae'], 2)} | {fmt(row['delta_interval_80_coverage'], 3)} |"
        )
    lines += [
        "",
        "## Paired team-season NLL diagnostics",
        "",
        "| Comparison | Scope | N | Mean ΔNLL | Median ΔNLL | Fraction improved |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in paired_rows_:
        lines.append(
            f"| {row['comparison']} | {row['scope']} | {row['n_team_seasons']} | {fmt(row['mean_delta_nll'])} | {fmt(row['median_delta_nll'])} | {fmt(row['fraction_team_seasons_improved'], 3)} |"
        )
    lines += [
        "",
        "## Standardized roster-continuity coefficients",
        "",
        "Numeric inputs are standardized using training-only means and scales. Missingness-indicator coefficients are shown separately; coefficient magnitudes are structural diagnostics, not causal effects.",
        "The frozen C 1.2 fit admits roster-continuity features into the location equation only, so their log-scale coefficients are fixed at zero by protocol.",
        "",
        "| Variant | Feature | Observed / missing training rows | Location β | Location missingness | Log-scale γ | Log-scale missingness |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in coefficient_rows:
        lines.append(
            f"| {row['candidate']} | {row['feature']} | {row['training_observed_n']} / {row['training_missing_n']} | {fmt(row['location_coefficient_standardized'])} | {fmt(row['location_missingness_coefficient'])} | {fmt(row['log_scale_coefficient_standardized'])} | {fmt(row['log_scale_missingness_coefficient'])} |"
        )
    lines += [
        "",
        "## Transfer coverage and limitations",
        "",
        "CFBD portal and prior-usage inputs retain the issue-91 retrospective-oracle caveats: endpoint responses are not archived as August 15 snapshots, final destinations may be resolved later, and the portal-to-usage join is name-based. Covered seasons with no matching transfer are zero; uncovered seasons remain missing and use the model's training-only imputation and missingness indicators.",
        "",
        "| Season | Portal payload | Records | Dated/on cutoff | Destination rate | Rating rate |",
        "|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in coverage:
        lines.append(
            f"| {row['season']} | {row['portal_payload_available']} | {row['n_portal_records']} | {row['n_on_or_before_cutoff']} | {fmt(row['destination_rate'], 3)} | {fmt(row['rating_rate'], 3)} |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        "- `variant_definitions.json`, `summary.json` — frozen configuration, parity check, comparisons, and provenance.",
        "- `candidate_annual_metrics.csv`, `candidate_summary.csv` — aggregate and year-by-year metrics for D0–D8.",
        "- `standardized_coefficients.csv` — D2–D8 roster-continuity coefficient diagnostics.",
        "- `paired_nll_diagnostics.csv`, `comparison_summary.csv` — required paired losses and metric comparisons.",
        "- `coverage_by_season.csv`, `provenance_audit.json`, `plots/` — source audit and visual summaries.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.resolve()
    transfer_root = args.transfer_root.resolve()
    output = args.output.resolve()
    prior.configure_source_root(source_root)
    rows, cold, _ = prior.v1.load_rows(max_season=max(TARGET_SEASONS))
    fbs = [row for row in rows if row.subdivision == "fbs"]
    contextual, _ = prior.c12.attach_context(fbs, prior.c12.feature_index(), prior.c12.cached_tenures())
    records, usage, portal_seasons, usage_seasons = prior.load_raw_transfer_data(transfer_root)
    team_rows = prior.fbs_feature_rows(contextual)

    transfer_features = prior.aggregate_team_features(
        records,
        usage,
        team_rows,
        covered_seasons=portal_seasons,
        cutoff=date(2025, *DEFAULT_CUTOFF),
    )
    contextual_augmented = prior.attach_transfer_features(contextual, transfer_features)
    panel_fallback = prior.fallback_for_panel(fbs, cold)
    panel_fallback = [item for item in panel_fallback if item.season in TARGET_SEASONS]
    target_keys = {
        (row.season, row.subdivision, row.team_id)
        for row in contextual_augmented
        if row.season in TARGET_SEASONS
    }
    if not target_keys <= {item.key for item in panel_fallback}:
        raise ValueError("stored H panel is missing regular target keys")

    candidates = candidate_definitions()
    predictions_by_name: dict[str, list[prior.v1.PriorPrediction]] = {}
    models: dict[str, DirectRankModel] = {}
    for candidate in candidates:
        predictions, model = fit_candidate(contextual_augmented, panel_fallback, candidate)
        predictions_by_name[candidate.name] = predictions
        models[candidate.name] = model
        print(f"completed {candidate.name}", flush=True)

    key_sets = {frozenset(item.key for item in values) for values in predictions_by_name.values()}
    if len(key_sets) != 1:
        raise ValueError("D0-D8 candidates do not preserve identical target keys")

    d0_predictions = predictions_by_name[candidates[0].name]
    d0_metrics = prior.score(d0_predictions)
    d0_metrics_by_season = {
        season: prior.score(
            [item for item in d0_predictions if item.season == season]
        )
        for season in TARGET_SEASONS
    }
    annual_rows = [
        metric_row(
            candidate.name,
            predictions_by_name[candidate.name],
            season=season,
            reference=d0_metrics_by_season[season],
        )
        for candidate in candidates
        for season in TARGET_SEASONS
    ]
    aggregate_rows = [
        metric_row(candidate.name, predictions_by_name[candidate.name], season=None, reference=d0_metrics)
        for candidate in candidates
    ]

    training_rows = [row for row in contextual_augmented if row.season <= FROZEN_TRAIN_THROUGH]
    coefficient_rows: list[dict[str, object]] = []
    for candidate in candidates[2:]:
        rows_for_candidate = standardized_coefficients(candidate, models[candidate.name])
        add_training_counts(rows_for_candidate, training_rows)
        coefficient_rows.extend(rows_for_candidate)

    paired_comparison_definitions = (
        ("D5_vs_D2", "D5_total_rp_plus_incoming", "D2_total_rp_only"),
        ("D7_vs_D5", "D7_total_rp_plus_incoming_plus_outgoing", "D5_total_rp_plus_incoming"),
        ("D5_vs_D8", "D5_total_rp_plus_incoming", "D8_existing_c10"),
    )
    paired_rows_all: list[dict[str, object]] = []
    paired_summaries: list[dict[str, object]] = []
    for comparison, candidate_name, reference_name in paired_comparison_definitions:
        rows_for_comparison, aggregate = paired_rows(
            comparison,
            candidate_name,
            predictions_by_name[candidate_name],
            reference_name,
            predictions_by_name[reference_name],
        )
        paired_rows_all.extend(rows_for_comparison)
        paired_summaries.append(aggregate)

    comparison_definitions = (
        ("incoming_unconditional", "D3_incoming_transfer_production_only", "D1_no_returning_production"),
        ("outgoing_unconditional", "D4_outgoing_transfer_production_only", "D1_no_returning_production"),
        ("incoming_conditional_on_rp", "D5_total_rp_plus_incoming", "D2_total_rp_only"),
        ("outgoing_conditional_on_rp", "D6_total_rp_plus_outgoing", "D2_total_rp_only"),
        ("outgoing_beyond_rp_plus_incoming", "D7_total_rp_plus_incoming_plus_outgoing", "D5_total_rp_plus_incoming"),
        ("simplified_vs_c10", "D5_total_rp_plus_incoming", "D8_existing_c10"),
    )
    comparison_rows = [
        comparison_summary(
            comparison,
            candidate_name,
            predictions_by_name[candidate_name],
            reference_name,
            predictions_by_name[reference_name],
        )
        for comparison, candidate_name, reference_name in comparison_definitions
    ]
    for row in paired_rows_all:
        if row["scope"] == "aggregate":
            comparison = str(row["comparison"])
            comparison_row = next(item for item in comparison_rows if item["comparison"] == {
                "D5_vs_D2": "incoming_conditional_on_rp",
                "D7_vs_D5": "outgoing_beyond_rp_plus_incoming",
                "D5_vs_D8": "simplified_vs_c10",
            }[comparison])
            row["delta_nll"] = comparison_row["delta_nll"]

    stored_path = source_root / "data/processed/preseason/context/evaluation.json"
    stored = json.loads(stored_path.read_text(encoding="utf-8")) if stored_path.exists() else None
    stored_c0 = stored.get("all_fbs", {}).get("candidate", {}) if stored else {}
    sanity_deltas = {
        metric: abs(d0_metrics[metric] - float(stored_c0[metric]))
        for metric in METRICS
        if metric in stored_c0
    }
    sanity = {
        "stored_evaluation_path": str(stored_path),
        "stored_metrics": stored_c0,
        "computed_metrics": d0_metrics,
        "metric_abs_deltas": sanity_deltas,
        "max_abs_metric_delta": max(sanity_deltas.values(), default=None),
        "tolerance": SANITY_TOLERANCE,
        "passed": bool(sanity_deltas) and max(sanity_deltas.values()) <= SANITY_TOLERANCE,
    }
    if not sanity["passed"]:
        raise ValueError(f"D0 does not reproduce stored production metrics: {sanity}")

    coverage = prior.coverage_rows(
        records,
        target_seasons=sorted({row.season for row in contextual}),
        cutoff=date(2025, *DEFAULT_CUTOFF),
        team_rows=team_rows,
        covered_seasons=portal_seasons,
    )
    variant_definitions = [
        {
            "name": candidate.name,
            "description": candidate.description,
            "features": list(candidate.features),
            "continuity_features": list(candidate.continuity_features),
            "transfer_aware": candidate.transfer_aware,
        }
        for candidate in candidates
    ]
    summary: dict[str, object] = {
        "study": "issue_96_transfer_signal_decomposition",
        "production_models_modified": False,
        "target_seasons": list(TARGET_SEASONS),
        "frozen_train_through": FROZEN_TRAIN_THROUGH,
        "cutoff": f"season-relative {DEFAULT_CUTOFF[0]:02d}-{DEFAULT_CUTOFF[1]:02d}",
        "portal_seasons_available": sorted(portal_seasons),
        "usage_seasons_available": sorted(usage_seasons),
        "variant_definitions": variant_definitions,
        "c0_sanity_check": sanity,
        "aggregate_metrics": aggregate_rows,
        "paired_comparisons": paired_summaries,
        "comparison_summary": comparison_rows,
        "missing_data_policy": "uncovered portal seasons are None; covered seasons with no matching transfer are zero; DirectRankModel applies training-only median imputation and missingness indicators",
        "provenance": prior.portal_provenance(),
        "source_hashes": {
            str(path.relative_to(source_root)): sha256_file(path)
            for path in (
                source_root / "data/processed/modeling/team_season_rank_distributions.csv",
                source_root / "data/processed/preseason/team_season_features.csv",
            )
            if path.exists()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "variant_definitions.json", variant_definitions)
    write_json(output / "provenance_audit.json", prior.portal_provenance())
    write_csv(output / "coverage_by_season.csv", coverage)
    write_csv(output / "candidate_annual_metrics.csv", annual_rows)
    write_csv(output / "candidate_summary.csv", aggregate_rows)
    write_csv(output / "standardized_coefficients.csv", coefficient_rows)
    write_csv(output / "paired_nll_diagnostics.csv", paired_rows_all)
    write_csv(output / "comparison_summary.csv", comparison_rows)
    write_json(output / "summary.json", summary)
    plot_outputs(output, aggregate_rows, coefficient_rows, coverage)
    render_report(
        output / "report.md",
        summary,
        annual_rows,
        aggregate_rows,
        coefficient_rows,
        paired_rows_all,
        comparison_rows,
        coverage,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument(
        "--transfer-root", type=Path, default=ROOT / "data/raw/cfbd/preseason/transfers"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/transfer_signal_decomposition",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps({"study": summary["study"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
