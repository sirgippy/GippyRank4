"""Reproduce issue 189's fixed, post-hoc OL core comparisons."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import scipy

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_preseason_context_prior_v1_3 as context_builder
import build_preseason_prior as prior_builder
import build_preseason_prior_v1_1 as history_builder
import evaluate_offensive_line_continuity_issue_187 as prior_experiment

from gippyrank.context_prior_v1_3 import (
    CONTEXT_1_3_FEATURES,
    FROZEN_PENALTY,
    H_FEATURES,
    context13_semantic_specification_sha256,
)
from gippyrank.preseason import TeamSeason
from gippyrank.research.offensive_line_continuity_experiment import (
    build_program_memberships,
    paired_season_bootstrap,
)
from gippyrank.research.offensive_line_core_experiment import (
    CORE_INDIVIDUAL,
    CORE_SHARED,
    CORE_TALENT_INTERACTION,
    build_core_panel,
    observed_college_seasons,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/research/offensive_line_core_experiment_issue_189/results"
PRIOR_RESULTS = (
    ROOT / "data/research/offensive_line_continuity_experiment_issue_187/results"
)
ROSTER = ROOT / "data/processed/offensive_line_shared_roster_issue_183"

REFERENCE = "context_1_3_style_restricted_refit"
INDIVIDUAL = "context_plus_core_individual_experience"
SHARED = "context_plus_core_individual_and_shared_experience"
INTERACTION = "context_plus_core_individual_shared_and_talent_interaction"
ARMS = (
    (REFERENCE, ()),
    (INDIVIDUAL, (CORE_INDIVIDUAL,)),
    (SHARED, (CORE_INDIVIDUAL, CORE_SHARED)),
    (INTERACTION, (CORE_INDIVIDUAL, CORE_SHARED, CORE_TALENT_INTERACTION)),
)
PREVIOUS = {INDIVIDUAL: REFERENCE, SHARED: INDIVIDUAL, INTERACTION: SHARED}


def prepare_analysis_rows(
    base_rows: list[TeamSeason],
    prior_panel: list[dict[str, str]],
    core_panel: list[dict[str, Any]],
) -> tuple[list[TeamSeason], dict[str, int]]:
    """Intersect once, before fitting, so every arm has identical target keys."""
    prior_rows, prior_excluded, _ = prior_experiment._analysis_rows(
        base_rows, prior_panel
    )
    core_by_key = {(int(row["season"]), str(row["team_id"])): row for row in core_panel}
    excluded = Counter(prior_excluded)
    result: list[TeamSeason] = []
    for row in prior_rows:
        core = core_by_key[(row.season, row.team_id)]
        if core["core_feature_status"] != "observed":
            excluded[str(core["core_feature_status"])] += 1
            continue
        features = dict(row.features)
        features[CORE_INDIVIDUAL] = float(core[CORE_INDIVIDUAL])
        features[CORE_SHARED] = float(core[CORE_SHARED])
        talent = features.get("talent_composite")
        features[CORE_TALENT_INTERACTION] = (
            None if talent is None else float(core[CORE_SHARED]) * float(talent)
        )
        result.append(replace(row, features=features))
    return result, dict(sorted(excluded.items()))


def _delta_summary(
    left: dict[tuple[int, str, str], tuple[float, float]],
    right: dict[tuple[int, str, str], tuple[float, float]],
    names: dict[tuple[int, str, str], str],
) -> dict[str, Any]:
    paired = prior_experiment._loss_deltas(left, right)
    values = [value[0] for value in paired.values()]
    gains = sorted(
        ((key, -value[0]) for key, value in paired.items() if value[0] < 0),
        key=lambda item: -item[1],
    )
    losses = sorted(
        ((key, value[0]) for key, value in paired.items() if value[0] > 0),
        key=lambda item: -item[1],
    )
    gross_gain = math.fsum(value for _, value in gains)
    gross_loss = math.fsum(value for _, value in losses)
    by_season = prior_experiment._deltas_by_season(paired)
    bootstrap = paired_season_bootstrap(by_season)
    return {
        "mean_delta_nll": math.fsum(values) / len(values),
        "team_seasons_improved": len(gains),
        "fraction_team_seasons_improved": len(gains) / len(values),
        "team_seasons_worsened": len(losses),
        "seasons_improved": sum(math.fsum(items) < 0 for items in by_season.values()),
        "gross_nll_gain": gross_gain,
        "gross_nll_loss": gross_loss,
        "top_10_share_of_gross_gain": math.fsum(value for _, value in gains[:10])
        / gross_gain
        if gross_gain
        else 0.0,
        "top_10_share_of_gross_loss": math.fsum(value for _, value in losses[:10])
        / gross_loss
        if gross_loss
        else 0.0,
        "largest_gain": [
            {"season": key[0], "team": names[key], "delta_nll": -value}
            for key, value in gains[:5]
        ],
        "largest_loss": [
            {"season": key[0], "team": names[key], "delta_nll": value}
            for key, value in losses[:5]
        ],
        "season_sum_delta_nll": {
            str(season): math.fsum(items) for season, items in sorted(by_season.items())
        },
        "season_cluster_bootstrap": bootstrap,
    }


def evaluate(
    rows: list[TeamSeason],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    training = [row for row in rows if 2013 <= row.season <= 2021]
    heldout = [row for row in rows if 2022 <= row.season <= 2025]
    if not training or not heldout:
        raise ValueError("2013-2021 training and 2022-2025 evaluation are required")
    models = {}
    predictions = {}
    losses = {}
    scores = {}
    for name, extras in ARMS:
        model = prior_experiment._fit_arm(training, extras)
        predicted = history_builder.make_predictions(model, heldout, name)
        models[name] = model
        predictions[name] = predicted
        losses[name] = history_builder.prediction_losses(predicted)
        scores[name] = prior_builder.score_predictions(predicted)

    keys = set(losses[REFERENCE])
    if any(set(losses[name]) != keys for name, _ in ARMS):
        raise ValueError("model arms have different evaluation populations")
    names = {
        (row.season, row.subdivision, row.team_id): row.team_name for row in heldout
    }
    comparisons = {
        name: {
            "vs_reference": _delta_summary(losses[REFERENCE], losses[name], names),
            "vs_previous": _delta_summary(losses[PREVIOUS[name]], losses[name], names),
        }
        for name, _ in ARMS
        if name != REFERENCE
    }

    aggregates = []
    annual = []
    for name, extras in ARMS:
        previous = PREVIOUS.get(name, REFERENCE)
        aggregates.append(
            {
                "model": name,
                "additional_features": ";".join(extras),
                "training_team_seasons": len(training),
                "evaluation_team_seasons": len(heldout),
                "nll": float(scores[name]["nll"]),
                "delta_nll_vs_reference": float(
                    scores[name]["nll"] - scores[REFERENCE]["nll"]
                ),
                "incremental_delta_nll": float(
                    scores[name]["nll"] - scores[previous]["nll"]
                ),
                "crps": float(scores[name]["crps"]),
                "expected_rank_mae": float(scores[name]["expected_rank_mae"]),
                "team_seasons_improved_vs_reference": comparisons[name]["vs_reference"][
                    "team_seasons_improved"
                ]
                if name != REFERENCE
                else 0,
                "fraction_improved_vs_reference": comparisons[name]["vs_reference"][
                    "fraction_team_seasons_improved"
                ]
                if name != REFERENCE
                else 0.0,
                "team_seasons_improved_vs_previous": comparisons[name]["vs_previous"][
                    "team_seasons_improved"
                ]
                if name != REFERENCE
                else 0,
                "fraction_improved_vs_previous": comparisons[name]["vs_previous"][
                    "fraction_team_seasons_improved"
                ]
                if name != REFERENCE
                else 0.0,
                "delta_nll_vs_reference_season_cluster_95_low": comparisons[name][
                    "vs_reference"
                ]["season_cluster_bootstrap"]["central_95_interval"][0]
                if name != REFERENCE
                else 0.0,
                "delta_nll_vs_reference_season_cluster_95_high": comparisons[name][
                    "vs_reference"
                ]["season_cluster_bootstrap"]["central_95_interval"][1]
                if name != REFERENCE
                else 0.0,
                "incremental_delta_nll_season_cluster_95_low": comparisons[name][
                    "vs_previous"
                ]["season_cluster_bootstrap"]["central_95_interval"][0]
                if name != REFERENCE
                else 0.0,
                "incremental_delta_nll_season_cluster_95_high": comparisons[name][
                    "vs_previous"
                ]["season_cluster_bootstrap"]["central_95_interval"][1]
                if name != REFERENCE
                else 0.0,
            }
        )
        for season in range(2022, 2026):
            seasonal = {
                arm: prior_builder.score_predictions(
                    [item for item in predictions[arm] if item.season == season]
                )
                for arm, _ in ARMS
            }
            annual.append(
                {
                    "season": season,
                    "model": name,
                    "team_seasons": sum(
                        item.season == season for item in predictions[name]
                    ),
                    "nll": float(seasonal[name]["nll"]),
                    "delta_nll_vs_reference": float(
                        seasonal[name]["nll"] - seasonal[REFERENCE]["nll"]
                    ),
                    "incremental_delta_nll": float(
                        seasonal[name]["nll"] - seasonal[previous]["nll"]
                    ),
                    "crps": float(seasonal[name]["crps"]),
                    "expected_rank_mae": float(seasonal[name]["expected_rank_mae"]),
                }
            )
    team_losses = []
    for key in sorted(keys):
        record = {"season": key[0], "team_id": key[2], "team_name": names[key]}
        for name, _ in ARMS:
            record[f"nll_{name}"] = losses[name][key][0]
            record[f"crps_{name}"] = losses[name][key][1]
        team_losses.append(record)

    coefficients = []
    for name, extras in ARMS:
        if name == REFERENCE:
            continue
        model = models[name]
        beta = model.beta[model.lag_count :]
        for feature in extras:
            index = model.feature_names.index(feature)
            coefficients.append(
                {
                    "model": name,
                    "feature": feature,
                    "standardized_location_coefficient": float(beta[1 + index]),
                    "missingness_location_coefficient": float(
                        beta[1 + len(model.feature_names) + index]
                    ),
                    "training_mean_after_imputation": model.preprocessor.means[feature],
                    "training_scale": model.preprocessor.scales[feature],
                }
            )
    metadata = {
        "training_team_seasons": len(training),
        "evaluation_team_seasons": len(heldout),
        "same_keys_all_arms": True,
        "comparisons": comparisons,
        "optimizer_diagnostics": {
            name: model.optimizer for name, model in models.items()
        },
    }
    return aggregates, annual, team_losses, coefficients, metadata


def feature_diagnostics(
    rows: list[TeamSeason], core_panel: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    behavior = []
    for feature in (CORE_INDIVIDUAL, CORE_SHARED):
        values = np.asarray([float(row.features[feature]) for row in rows])
        ranks = np.asarray(
            [float(np.mean(row.target_ranks) / row.population) for row in rows]
        )
        behavior.append(
            {
                "feature": feature,
                "team_seasons": len(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p10": float(np.quantile(values, 0.10)),
                "p90": float(np.quantile(values, 0.90)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "pearson_r_vs_mean_final_rank_fraction": float(
                    np.corrcoef(values, ranks)[0, 1]
                ),
            }
        )
    eligible = [row for row in core_panel if 2013 <= int(row["season"]) <= 2025]
    observed = [row for row in eligible if row["core_feature_status"] == "observed"]
    analytic_keys = {(row.season, row.team_id) for row in rows}
    analytic_panel = [
        row
        for row in observed
        if (int(row["season"]), str(row["team_id"])) in analytic_keys
    ]
    core_correlation = float(
        np.corrcoef(
            [float(row.features[CORE_INDIVIDUAL]) for row in rows],
            [float(row.features[CORE_SHARED]) for row in rows],
        )[0, 1]
    )
    prior_redundancy = float(
        np.corrcoef(
            [float(row["ol_returning_player_share_4y"]) for row in analytic_panel],
            [float(row["ol_returning_group_share_4y"]) for row in analytic_panel],
        )[0, 1]
    )
    realizations = [int(row["core_tied_realizations"]) for row in observed]
    coverage = {
        "primary_panel_team_seasons": len(eligible),
        "core_feature_status_counts": dict(
            sorted(Counter(str(row["core_feature_status"]) for row in eligible).items())
        ),
        "observed_core_team_seasons": len(observed),
        "boundary_tie_team_seasons": sum(value > 1 for value in realizations),
        "boundary_tie_fraction": sum(value > 1 for value in realizations)
        / len(realizations),
        "realization_count_distribution": dict(sorted(Counter(realizations).items())),
        "max_realizations": max(realizations),
        "median_realizations": float(np.median(realizations)),
        "core_individual_shared_pearson_r": core_correlation,
        "previous_returning_player_group_pearson_r_on_same_cohort": prior_redundancy,
        "target_roster_player_count_p10_p50_p90": [
            float(
                np.quantile([int(row["target_ol_player_count"]) for row in observed], q)
            )
            for q in (0.1, 0.5, 0.9)
        ],
        "target_identity_missing_team_seasons": sum(
            int(row["target_ol_player_ids_missing"]) > 0 for row in eligible
        ),
        "target_unknown_position_rows": sum(
            int(row["target_ol_unknown_position_rows"]) for row in eligible
        ),
        "target_ambiguous_position_rows": sum(
            int(row["target_ol_ambiguous_position_rows"]) for row in eligible
        ),
        "target_roster_temporal_status": "retrospective CFBD target-season OL roster; not archived preseason membership",
    }
    return behavior, coverage


def decision_from_fixed_comparisons(metadata: dict[str, Any]) -> str:
    """Require an incremental gain across seasons and a negative cluster interval."""
    for name in (SHARED, INTERACTION):
        comparison = metadata["comparisons"][name]["vs_previous"]
        if (
            comparison["mean_delta_nll"] < 0
            and comparison["seasons_improved"] >= 3
            and comparison["season_cluster_bootstrap"]["central_95_interval"][1] < 0
        ):
            return "Refine"
    return "Reject"


def render_report(
    aggregates: list[dict[str, Any]],
    annual: list[dict[str, Any]],
    coefficients: list[dict[str, Any]],
    behavior: list[dict[str, Any]],
    coverage: dict[str, Any],
    metadata: dict[str, Any],
    excluded: dict[str, int],
    decision: str,
    reference_audit: dict[str, Any],
) -> str:
    def number(value: float) -> str:
        return f"{value:+.4f}"

    by_name = {row["model"]: row for row in aggregates}
    lines = [
        "# Issue 189: experience-selected OL core continuity",
        "",
        f"**Decision: {decision}.**",
        "",
        (
            "The three additions are post-hoc: the 2022–2025 evaluation years were inspected before this experiment. "
            "A favorable result only supports refinement and future independent evaluation, not production promotion."
        ),
        "",
        "## Construction and population",
        "",
        (
            "Each target-season CFBD OL roster is retrospective. For each player, experience is the count of unique "
            "observed college roster seasons across all programs in T-4 through T-1. The five highest counts define "
            "the proxy core. Every valid five-player choice at a fifth-place tie is evaluated and the two features "
            "are averaged over those choices. Selection never uses shared history. Shared experience averages the "
            "ten pair counts for prior seasons on the current program's roster. The proxy is not an actual starting five."
        ),
        "",
        (
            f"The primary roster panel has {coverage['primary_panel_team_seasons']} team-seasons; "
            f"{coverage['observed_core_team_seasons']} have usable core features. "
            f"The common model population has {metadata['training_team_seasons']} training and "
            f"{metadata['evaluation_team_seasons']} held-out team-seasons. All arms have identical held-out keys. "
            f"Exclusions from the previous common population: {json.dumps(excluded, sort_keys=True)}."
        ),
        "",
        (
            f"The target OL pool has p10/median/p90 sizes of "
            f"{coverage['target_roster_player_count_p10_p50_p90']}. "
            f"Across the primary panel, {coverage['target_identity_missing_team_seasons']} team-seasons have "
            f"missing target OL IDs; unknown and ambiguous position row counts are "
            f"{coverage['target_unknown_position_rows']} and {coverage['target_ambiguous_position_rows']}. "
            f"Core feature statuses: {json.dumps(coverage['core_feature_status_counts'], sort_keys=True)}."
        ),
        "",
        (
            f"Fifth-place ties produced multiple valid cores for {coverage['boundary_tie_team_seasons']} "
            f"observed team-seasons ({coverage['boundary_tie_fraction']:.1%}); median and maximum valid "
            f"realizations were {coverage['median_realizations']:.0f} and {coverage['max_realizations']}. "
            "The full count distribution and retained roster coverage fields are in the manifest and feature panel."
        ),
        "",
        "## Fixed held-out comparisons",
        "",
        (
            "Negative NLL differences favor the added feature. The restricted Context 1.3-style reference "
            "uses the same rank-history, recruiting, talent, returning-production, coaching, and transfer inputs, "
            "training-only preprocessing, rank-distribution likelihood, regularization, and 2013–2021 fit "
            "as issue 187. The interaction uses existing `talent_composite` and retains both main effects. "
            "Its sign was free in fitting."
        ),
        "",
        "| Model | NLL | Δ vs reference | Incremental Δ | CRPS | Expected-rank MAE | Improved vs reference | Improved vs previous | 95% season-cluster interval for incremental Δ |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|:--|",
    ]
    for name, _ in ARMS:
        row = by_name[name]
        improved_previous = (
            f"{row['team_seasons_improved_vs_previous']} ({row['fraction_improved_vs_previous']:.1%})"
            if name != REFERENCE
            else "—"
        )
        improved_reference = (
            f"{row['team_seasons_improved_vs_reference']} ({row['fraction_improved_vs_reference']:.1%})"
            if name != REFERENCE
            else "—"
        )
        interval = (
            f"[{number(row['incremental_delta_nll_season_cluster_95_low'])}, {number(row['incremental_delta_nll_season_cluster_95_high'])}]"
            if name != REFERENCE
            else "—"
        )
        lines.append(
            f"| `{name}` | {row['nll']:.4f} | {number(row['delta_nll_vs_reference'])} | {number(row['incremental_delta_nll'])} | {row['crps']:.4f} | {row['expected_rank_mae']:.3f} | {improved_reference} | {improved_previous} | {interval} |"
        )
    lines += [
        "",
        (
            "The cluster intervals exhaustively resample four held-out seasons (4⁴ ordered draws). "
            "They are descriptive and imprecise with only four clusters. Per-team scores and both paired "
            "reference and incremental uncertainty summaries are in the CSV and manifest."
        ),
        "",
        (
            f"Reproducibility audit: held-out team keys match issue 187 exactly. Its committed reference "
            f"reports {reference_audit['issue187_published_reference_nll']:.4f} NLL; rerunning its "
            f"unchanged adapter on the pinned source hashes in this environment yields "
            f"{reference_audit['current_reference_nll']:.4f}. Its `--check` reproduces the feature panel "
            "but fails on model-score artifacts. The source of that numerical discrepancy is unresolved; "
            "all issue 189 deltas use the same-run refit above."
        ),
        "",
        "### Results by season",
        "",
        "| Season | Model | N | NLL | Δ vs reference | Incremental Δ | CRPS | Expected-rank MAE |",
        "|--:|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for row in annual:
        lines.append(
            f"| {row['season']} | `{row['model']}` | {row['team_seasons']} | {row['nll']:.4f} | {number(row['delta_nll_vs_reference'])} | {number(row['incremental_delta_nll'])} | {row['crps']:.4f} | {row['expected_rank_mae']:.3f} |"
        )
    lines += ["", "## Feature and coefficient behavior", ""]
    lines += [
        "| Feature | Mean | Median | P10 | P90 | r vs final-rank fraction |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for row in behavior:
        lines.append(
            f"| `{row['feature']}` | {row['mean']:.3f} | {row['median']:.3f} | {row['p10']:.3f} | {row['p90']:.3f} | {row['pearson_r_vs_mean_final_rank_fraction']:.3f} |"
        )
    lines += [
        "",
        (
            f"Core individual and shared experience correlate at {coverage['core_individual_shared_pearson_r']:.3f}; "
            f"the earlier returning-player and returning-group features correlate at "
            f"{coverage['previous_returning_player_group_pearson_r_on_same_cohort']:.3f} on this cohort. "
            "The core construction materially reduces the earlier near-redundancy, though that alone does not establish predictive value."
        ),
        "",
        "| Model | Feature | Standardized location coefficient |",
        "|:--|:--|--:|",
    ]
    for row in coefficients:
        lines.append(
            f"| `{row['model']}` | `{row['feature']}` | {number(row['standardized_location_coefficient'])} |"
        )
    lines += [
        "",
        (
            "The individual-experience coefficient changes sign when shared experience enters; its separate interpretation is unstable. "
            "The shared-experience coefficient stays negative in the two nested arms, but held-out likelihood worsens when it enters. "
            "The interaction coefficient is small and its incremental predictive result is essentially zero. "
            "Coefficients use training-standardized inputs, are regularized, and are not causal effects."
        ),
        "",
        "## Concentration and decision",
        "",
    ]
    for name in (INDIVIDUAL, SHARED, INTERACTION):
        c = metadata["comparisons"][name]["vs_previous"]
        season_parts = ", ".join(
            f"{year}: {number(value)}"
            for year, value in c["season_sum_delta_nll"].items()
        )
        biggest = ", ".join(
            f"{item['team']} {item['season']} ({number(item['delta_nll'])})"
            for item in c["largest_loss"][:3]
        )
        lines.append(
            f"- `{name}` vs its simpler arm: {c['team_seasons_improved']} of {metadata['evaluation_team_seasons']} "
            f"team-seasons improved; {c['seasons_improved']}/4 seasons improved. "
            f"The top ten team-seasons account for {c['top_10_share_of_gross_gain']:.1%} of gross gains "
            f"and {c['top_10_share_of_gross_loss']:.1%} of gross losses. "
            f"Season total ΔNLL: {season_parts}. Largest losses: {biggest}."
        )
    lines += [
        "",
        (
            "Core shared experience or its fixed talent interaction met the stated credibility screen: "
            "negative incremental NLL, improvement in at least three seasons, and a negative upper endpoint "
            "of the descriptive season-cluster interval. **Refine** with future independent evaluation before "
            "any production candidate."
            if decision == "Refine"
            else "Neither core shared experience beyond individual core experience nor the fixed talent interaction "
            "met the stated credibility screen. **Reject** CFBD roster-based OL continuity until "
            "materially better evidence, such as point-in-time depth charts, starts, or reliable preseason "
            "OL membership, becomes available. No further feature search is proposed."
        ),
        "",
        "## Limits",
        "",
        (
            "Unobserved JUCO, lower-division, or other college seasons cannot contribute to measured experience. "
            "CFBD target OL rosters are retrospective, and player identities and historical memberships retain "
            "the limitations documented in issues 183, 185, and 187. Prior seasons are strictly before T, "
            "but retrospective target membership can itself reveal later availability. These post-hoc results "
            "do not validate preseason deployment or revise the conclusion of issue 187."
        ),
        "",
    ]
    return "\n".join(lines)


def run(output_dir: Path, *, check: bool = False) -> None:
    paths = {
        "rank_distributions": ROOT
        / "data/processed/modeling/team_season_rank_distributions.csv",
        "context_features": ROOT / "data/processed/preseason/team_season_features.csv",
        "context_transfer_features": context_builder.HISTORICAL_TRANSFER_FEATURES,
        "prior_feature_panel": PRIOR_RESULTS / "feature_panel.csv",
        "prior_manifest": PRIOR_RESULTS / "manifest.json",
        "prior_evaluation_aggregate": PRIOR_RESULTS / "evaluation_aggregate.csv",
        "prior_evaluation_team_losses": PRIOR_RESULTS / "evaluation_team_losses.csv",
        "normalized_roster_seasons": ROSTER / "normalized_roster_player_seasons.csv.gz",
        "normalized_target_ol": ROSTER / "normalized_ol_player_seasons.csv.gz",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "required existing research inputs missing: " + ", ".join(missing)
        )
    previous_manifest = json.loads(paths["prior_manifest"].read_text(encoding="utf-8"))
    if (
        previous_manifest["context13_semantic_specification_sha256"]
        != context13_semantic_specification_sha256()
    ):
        raise ValueError("Context 1.3 reference semantics differ from issue 187")
    for name, previous_name in (
        ("rank_distributions", "rank_distributions"),
        ("context_features", "context_features"),
        ("context_transfer_features", "context13_transfer_features"),
        ("normalized_roster_seasons", "roster_normalized_player_seasons"),
        ("normalized_target_ol", "roster_normalized_ol_player_seasons"),
    ):
        expected = previous_manifest["input_artifacts"][previous_name]["sha256"]
        if prior_experiment.sha256_file(paths[name]) != expected:
            raise ValueError(f"{name} does not match the issue 187 reference input")

    prior_panel = prior_experiment.read_csv(paths["prior_feature_panel"])
    for row in prior_panel:
        for flag in ("target_ol_pool_usable", "primary_four_year_window_complete"):
            row[flag] = row[flag] == "true"
    roster_rows = prior_experiment.read_gzip_csv(paths["normalized_roster_seasons"])
    ol_rows = prior_experiment.read_gzip_csv(paths["normalized_target_ol"])
    memberships, _, _, _ = build_program_memberships(roster_rows)
    core_panel = build_core_panel(
        prior_panel=prior_panel,
        ol_rows=ol_rows,
        college_seasons=observed_college_seasons(roster_rows),
        program_memberships=memberships,
    )
    base_rows, _cold, _context_coverage = context_builder.load_candidate_rows()
    analysis_rows, excluded = prepare_analysis_rows(base_rows, prior_panel, core_panel)
    aggregates, annual, team_losses, coefficients, metadata = evaluate(analysis_rows)
    earlier_losses = prior_experiment.read_csv(paths["prior_evaluation_team_losses"])
    earlier_keys = {(int(row["season"]), str(row["team_id"])) for row in earlier_losses}
    current_keys = {(int(row["season"]), str(row["team_id"])) for row in team_losses}
    if earlier_keys != current_keys:
        raise ValueError("issue 189 evaluation keys differ from issue 187")
    earlier_aggregates = prior_experiment.read_csv(paths["prior_evaluation_aggregate"])
    earlier_reference_nll = float(
        next(row["nll"] for row in earlier_aggregates if row["model"] == REFERENCE)
    )
    current_reference_nll = float(aggregates[0]["nll"])
    reference_audit = {
        "same_heldout_team_keys_as_issue187": True,
        "issue187_published_reference_nll": earlier_reference_nll,
        "current_reference_nll": current_reference_nll,
        "difference": current_reference_nll - earlier_reference_nll,
        "issue187_recheck": "Feature panel reproduces; model-score artifacts differ in current environment.",
    }
    behavior, coverage = feature_diagnostics(analysis_rows, core_panel)
    decision = decision_from_fixed_comparisons(metadata)
    report = render_report(
        aggregates,
        annual,
        coefficients,
        behavior,
        coverage,
        metadata,
        excluded,
        decision,
        reference_audit,
    )
    manifest = {
        "issue": 189,
        "decision": decision,
        "post_hoc": True,
        "training_seasons": [2013, 2021],
        "evaluation_seasons": [2022, 2025],
        "lookback_seasons": 4,
        "core_size": 5,
        "model_arms": [
            {"name": name, "extra_features": list(extras)} for name, extras in ARMS
        ],
        "context13_semantic_specification_sha256": context13_semantic_specification_sha256(),
        "context_features": [*H_FEATURES, *CONTEXT_1_3_FEATURES],
        "regularization_penalty": FROZEN_PENALTY,
        "interaction_definition": f"{CORE_SHARED} × talent_composite, with both main effects",
        "decision_screen": "Refine only for negative incremental held-out NLL, at least three seasons improved, and season-cluster 95% interval entirely below zero; otherwise Reject.",
        "source_hashes": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": prior_experiment.sha256_file(path),
            }
            for name, path in sorted(paths.items())
        },
        "implementation_hashes": {
            "script": prior_experiment.sha256_file(Path(__file__)),
            "core_module": prior_experiment.sha256_file(
                ROOT / "src/gippyrank/research/offensive_line_core_experiment.py"
            ),
        },
        "coverage": coverage,
        "cohort_exclusions": excluded,
        "evaluation": metadata,
        "reference_reproducibility_audit": reference_audit,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "limitations": [
            "Target-season CFBD OL membership is retrospective rather than a preseason snapshot.",
            "JUCO, lower-division, or other college history missing from the existing roster corpus remains unobserved.",
            "Player identities and historical roster records inherit issues 183, 185, and 187 limitations.",
            "Only four already-inspected held-out seasons support cluster uncertainty summaries.",
        ],
    }
    artifacts = {
        "feature_panel.csv": core_panel,
        "evaluation_aggregate.csv": aggregates,
        "evaluation_by_season.csv": annual,
        "evaluation_team_losses.csv": team_losses,
        "feature_behavior.csv": behavior,
        "feature_coefficients.csv": coefficients,
    }
    if check:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="issue189-check-") as directory:
            temporary = Path(directory)
            for filename, rows in artifacts.items():
                prior_experiment.write_csv(temporary / filename, rows)
                if (temporary / filename).read_bytes() != (
                    output_dir / filename
                ).read_bytes():
                    raise ValueError(f"generated artifact differs: {filename}")
            prior_experiment.write_json(temporary / "manifest.json", manifest)
            (temporary / "report.md").write_text(report, encoding="utf-8")
            for filename in ("manifest.json", "report.md"):
                if (temporary / filename).read_bytes() != (
                    output_dir / filename
                ).read_bytes():
                    raise ValueError(f"generated artifact differs: {filename}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, rows in artifacts.items():
            prior_experiment.write_csv(output_dir / filename, rows)
        prior_experiment.write_json(output_dir / "manifest.json", manifest)
        (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Issue 189: {decision}; results at {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, check=args.check)


if __name__ == "__main__":
    main()
