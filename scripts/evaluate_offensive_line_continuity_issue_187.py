"""Run the research-only four-season OL continuity evaluation for issue 187."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_preseason_context_prior_v1_3 as context_builder
import build_preseason_prior as prior_builder
import build_preseason_prior_v1_1 as history_builder

from gippyrank import preseason as fit_semantics
from gippyrank.context_prior_v1_3 import (
    CONTEXT13_DISTRIBUTION_FAMILY,
    CONTEXT13_MINIMUM_SCALE,
    CONTEXT13_TEAM_SEASON_WEIGHT,
    CONTEXT_1_3_FEATURES,
    FROZEN_PENALTY,
    H_FEATURES,
    SCALE_FEATURE_NAMES,
    context13_semantic_specification_sha256,
)
from gippyrank.preseason import DirectRankModel, TeamSeason
from gippyrank.research.offensive_line_continuity_experiment import (
    CONTINUITY_FEATURES,
    DIAGNOSTIC_FEATURES,
    HISTORY_START_SEASON,
    INDIVIDUAL_EXPERIENCE_FEATURES,
    LOOKBACK_SEASONS,
    MODEL_FEATURES,
    POST_HOC_MODEL_ARMS,
    POST_HOC_SHARED_ROSTER_FEATURE,
    PRIMARY_END_SEASON,
    PRIMARY_START_SEASON,
    build_official_pool_sensitivity,
    build_program_memberships,
    build_team_season_feature_panel,
    normalized_person_name,
    paired_season_bootstrap,
    player_identity,
    validate_pairwise_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
ROSTER_PANEL = ROOT / "data/processed/offensive_line_shared_roster_issue_183"
VALIDATION = ROOT / "data/research/offensive_line_target_pool_validation_issue_185"
COSTART_PANEL = ROOT / "data/processed/offensive_line_co_start_pilot_issue_181"
OUTPUT = ROOT / "data/research/offensive_line_continuity_experiment_issue_187/results"
RANK_DISTRIBUTIONS = ROOT / "data/processed/modeling/team_season_rank_distributions.csv"

BASELINE = "context_1_3_style_restricted_refit"
INDIVIDUAL = "context_plus_ol_individual_experience"
CONTINUITY = "context_plus_ol_shared_roster_continuity"
COMBINED = "context_plus_both_ol_experience_and_continuity"
MODEL_ARMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (BASELINE, ()),
    (INDIVIDUAL, INDIVIDUAL_EXPERIENCE_FEATURES),
    (CONTINUITY, CONTINUITY_FEATURES),
    (COMBINED, MODEL_FEATURES),
) + POST_HOC_MODEL_ARMS
POST_HOC_ARM_NAMES = frozenset(name for name, _features in POST_HOC_MODEL_ARMS)


def _analysis_role(model_name: str) -> str:
    if model_name == BASELINE:
        return "reference_model"
    if model_name in POST_HOC_ARM_NAMES:
        return "post_hoc_diagnostic"
    return "original_bundle_comparison"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, mode="rt", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty artifact: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return format(value, ".12g") if math.isfinite(value) else ""
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return "external local input"


def _source_inventory() -> dict[str, Path]:
    paths = {
        "rank_distributions": RANK_DISTRIBUTIONS,
        "context_features": ROOT / "data/processed/preseason/team_season_features.csv",
        "context13_transfer_features": context_builder.HISTORICAL_TRANSFER_FEATURES,
        "context13_transfer_provenance": context_builder.HISTORICAL_TRANSFER_FEATURES.with_name(
            "historical_transfer_features.provenance.json"
        ),
        "roster_team_season_summaries": ROSTER_PANEL / "team_season_summaries.csv",
        "roster_normalized_player_seasons": ROSTER_PANEL
        / "normalized_roster_player_seasons.csv.gz",
        "roster_normalized_ol_player_seasons": ROSTER_PANEL
        / "normalized_ol_player_seasons.csv.gz",
        "roster_pairwise_continuity": ROSTER_PANEL
        / "pairwise_shared_roster_continuity.csv.gz",
        "issue185_player_comparison": VALIDATION / "results/player_comparison.csv",
        "issue185_team_season_summary": VALIDATION / "results/team_season_summary.csv",
        "issue185_identity_crosswalk": VALIDATION / "identity_crosswalk.csv",
        "issue181_eligible_co_start_pairs": COSTART_PANEL
        / "eligible_co_start_pairs.csv",
        "context13_semantic_implementation": ROOT
        / "src/gippyrank/context_prior_v1_3.py",
        "direct_rank_implementation": ROOT / "src/gippyrank/preseason.py",
    }
    for path in sorted(context_builder.c12.TENURES.glob("*.json")):
        paths[f"coach_tenure/{path.name}"] = path
    return paths


def _check_required_sources(paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "issue 187 needs the existing rank/context/roster research artifacts; "
            "missing: " + ", ".join(missing)
        )


def _panel_inputs() -> tuple[
    list[dict[str, Any]],
    dict[tuple[int, str], set[str]],
    dict[tuple[str, str], set[int]],
    dict[tuple[int, str], dict[str, set[str]]],
    dict[str, set[tuple[str, int]]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    summary_rows = read_csv(ROSTER_PANEL / "team_season_summaries.csv")
    roster_rows = read_gzip_csv(
        ROSTER_PANEL / "normalized_roster_player_seasons.csv.gz"
    )
    ol_rows = read_gzip_csv(ROSTER_PANEL / "normalized_ol_player_seasons.csv.gz")
    pair_rows = read_gzip_csv(ROSTER_PANEL / "pairwise_shared_roster_continuity.csv.gz")
    memberships, player_seasons, roster_names, player_program_seasons = (
        build_program_memberships(roster_rows)
    )
    pairwise_audit = validate_pairwise_artifact(summary_rows, pair_rows)
    panel = build_team_season_feature_panel(
        summary_rows=summary_rows,
        ol_rows=ol_rows,
        program_memberships=memberships,
        player_program_seasons=player_program_seasons,
        pairwise_audit=pairwise_audit,
        start_season=HISTORY_START_SEASON,
        end_season=PRIMARY_END_SEASON,
    )
    return (
        panel,
        memberships,
        player_seasons,
        roster_names,
        player_program_seasons,
        pair_rows,
        ol_rows,
    )


def _official_sensitivity(
    *,
    panel: list[dict[str, Any]],
    roster_names: dict[tuple[int, str], dict[str, set[str]]],
    memberships: dict[tuple[str, int], set[str]],
    player_program_seasons: dict[str, set[tuple[str, int]]],
) -> list[dict[str, Any]]:
    panel_by_key = {(int(row["season"]), str(row["team_id"])): row for row in panel}
    team_ids_by_name = {
        (int(row["season"]), normalized_person_name(str(row["team_name"]))): str(
            row["team_id"]
        )
        for row in panel
    }
    validation_teams = read_csv(VALIDATION / "results/team_season_summary.csv")
    for row in validation_teams:
        season = int(row["season"])
        name_key = normalized_person_name(str(row["team_name"]))
        team_id = team_ids_by_name.get((season, name_key))
        if team_id is None and name_key == "appalachianstate":
            team_id = team_ids_by_name.get((season, "appstate"))
        row["team_id"] = team_id or ""
    return build_official_pool_sensitivity(
        player_comparison_rows=read_csv(VALIDATION / "results/player_comparison.csv"),
        team_season_rows=validation_teams,
        normalized_roster_names=roster_names,
        program_memberships=memberships,
        player_program_seasons=player_program_seasons,
        cfbd_feature_rows=panel_by_key,
        crosswalk_rows=read_csv(VALIDATION / "identity_crosswalk.csv"),
    )


def _target_ol_name_index(
    ol_rows: list[dict[str, str]],
) -> tuple[
    dict[tuple[int, str], dict[str, set[str]]],
    dict[tuple[int, str], str],
]:
    by_name: dict[tuple[int, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    team_name_by_key: dict[tuple[int, str], str] = {}
    for row in ol_rows:
        if str(row.get("source_classification") or "").lower() != "fbs":
            continue
        try:
            key = (int(row["season"]), str(row["team_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        identity = player_identity(row)
        name = normalized_person_name(
            str(row.get("normalized_player_name") or row.get("player_name") or "")
        )
        if identity and name:
            by_name[key][name].add(identity)
        team_name_by_key[key] = str(row.get("team_name") or "")
    return {key: dict(value) for key, value in by_name.items()}, team_name_by_key


def _co_start_comparison(
    *,
    ol_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    player_program_seasons: dict[str, set[tuple[str, int]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_target_name, _team_names = _target_ol_name_index(ol_rows)
    team_ids = {
        (int(row["season"]), normalized_person_name(str(row["team_name"]))): str(
            row["team_id"]
        )
        for row in read_csv(ROSTER_PANEL / "team_season_summaries.csv")
    }
    pair_counts: dict[tuple[int, str, str, str], int] = {}
    for row in pair_rows:
        value = row.get("shared_prior_season_count")
        if value in (None, ""):
            continue
        first = str(row.get("player_1_identity") or "")
        second = str(row.get("player_2_identity") or "")
        if first and second:
            pair_counts[
                (
                    int(row["target_season"]),
                    str(row["team_id"]),
                    *sorted((first, second)),
                )
            ] = int(value)

    output: list[dict[str, Any]] = []
    for row in read_csv(COSTART_PANEL / "eligible_co_start_pairs.csv"):
        season = int(row["target_season"])
        team_name = str(row["team"])
        team_id = team_ids.get((season, normalized_person_name(team_name)))
        if team_id is None:
            output.append(
                {
                    "target_season": season,
                    "team_name": team_name,
                    "player_1_name": row["player_1_name"],
                    "player_2_name": row["player_2_name"],
                    "pair_match_status": "team_unmatched",
                    "shared_starts": int(row["shared_starts"]),
                    "shared_start_rate": float(row["shared_start_rate"]),
                }
            )
            continue
        lookup = by_target_name.get((season, team_id), {})
        first = lookup.get(normalized_person_name(row["player_1_name"]), set())
        second = lookup.get(normalized_person_name(row["player_2_name"]), set())
        if len(first) != 1 or len(second) != 1:
            status = "target_ol_player_unmatched_or_ambiguous"
            shared_4y = None
            shared_all_history = None
        else:
            left, right = next(iter(first)), next(iter(second))
            pair_key = (season, team_id, *sorted((left, right)))
            shared_all_history = pair_counts.get(pair_key)
            if shared_all_history is None:
                status = "pair_not_in_continuity_artifact"
                shared_4y = None
            else:
                window_start = max(HISTORY_START_SEASON, season - LOOKBACK_SEASONS)
                left_years = {
                    year
                    for team, year in player_program_seasons.get(left, set())
                    if team == team_id and window_start <= year < season
                }
                right_years = {
                    year
                    for team, year in player_program_seasons.get(right, set())
                    if team == team_id and window_start <= year < season
                }
                shared_4y = len(left_years & right_years)
                status = "matched"
        output.append(
            {
                "target_season": season,
                "team_id": team_id,
                "team_name": team_name,
                "prior_season": int(row["prior_season"]),
                "player_1_name": row["player_1_name"],
                "player_2_name": row["player_2_name"],
                "shared_starts": int(row["shared_starts"]),
                "prior_matrix_games": int(row["prior_matrix_games"]),
                "shared_start_rate": float(row["shared_start_rate"]),
                "shared_roster_seasons_4y": shared_4y,
                "shared_roster_seasons_all_history": shared_all_history,
                "pair_match_status": status,
                "prior_matrix_source_id": row["prior_matrix_source_id"],
            }
        )
    matched = [row for row in output if row.get("pair_match_status") == "matched"]
    if len(matched) >= 3:
        rho = float(
            spearmanr(
                [float(row["shared_start_rate"]) for row in matched],
                [float(row["shared_roster_seasons_4y"]) for row in matched],
            ).statistic
        )
        if not math.isfinite(rho):
            rho = None
    else:
        rho = None
    summary = {
        "co_start_pair_rows": len(output),
        "co_start_pairs_matched_to_cfbd_target_pool": len(matched),
        "matched_team_seasons": len(
            {(row["target_season"], row["team_id"]) for row in matched}
        ),
        "spearman_rho_start_rate_vs_four_year_shared_roster_seasons": rho,
        "description": (
            "Positive co-start pairs only; a small descriptive compatibility check, "
            "not a population-level validation."
        ),
    }
    return output, summary


def _analysis_rows(
    base_rows: list[TeamSeason], panel: list[dict[str, Any]]
) -> tuple[list[TeamSeason], dict[str, int], dict[tuple[int, str], dict[str, Any]]]:
    panel_by_key = {(int(row["season"]), str(row["team_id"])): row for row in panel}
    excluded: Counter[str] = Counter()
    result: list[TeamSeason] = []
    for row in base_rows:
        if not PRIMARY_START_SEASON <= row.season <= PRIMARY_END_SEASON:
            continue
        key = (row.season, str(row.team_id))
        feature_row = panel_by_key.get(key)
        if feature_row is None:
            excluded["missing_feature_panel_row"] += 1
            continue
        if not feature_row["target_ol_pool_usable"]:
            excluded[str(feature_row["feature_status"])] += 1
            continue
        if not feature_row["primary_four_year_window_complete"]:
            excluded[str(feature_row["feature_status"])] += 1
            continue
        features = dict(row.features)
        for name in (*MODEL_FEATURES, *DIAGNOSTIC_FEATURES):
            value = feature_row.get(name)
            features[name] = None if value in (None, "") else float(value)
        result.append(replace(row, features=features))
    result.sort(key=lambda row: (row.season, row.team_id))
    return result, dict(sorted(excluded.items())), panel_by_key


def _fit_arm(
    training_rows: list[TeamSeason], extra_features: tuple[str, ...]
) -> DirectRankModel:
    model_features = [*H_FEATURES, *CONTEXT_1_3_FEATURES, *extra_features]
    location_features = [*H_FEATURES, *CONTEXT_1_3_FEATURES, *extra_features]
    optimizer = {
        "maxiter": fit_semantics.DIRECT_RANK_OPTIMIZER_MAXITER,
        "ftol": fit_semantics.DIRECT_RANK_OPTIMIZER_FTOL,
        "gtol": fit_semantics.DIRECT_RANK_OPTIMIZER_GTOL,
    }

    def fit(options: dict[str, float | int] | None = None) -> DirectRankModel:
        settings = dict(optimizer)
        settings.update(options or {})
        return DirectRankModel.fit(
            training_rows,
            model_features,
            penalty=FROZEN_PENALTY,
            minimum_scale=CONTEXT13_MINIMUM_SCALE,
            optimizer_options=settings,
            lag_count=1,
            family=CONTEXT13_DISTRIBUTION_FAMILY,
            location_feature_names=location_features,
            scale_feature_names=list(SCALE_FEATURE_NAMES),
            row_weights=np.full(
                len(training_rows), CONTEXT13_TEAM_SEASON_WEIGHT, dtype=float
            ),
            preprocessor_scale_floor=fit_semantics.PREPROCESSOR_SCALE_FLOOR,
            preprocessor_std_ddof=fit_semantics.PREPROCESSOR_STD_DDOF,
        )

    try:
        return fit()
    except RuntimeError as error:
        if "ITERATIONS REACHED LIMIT" not in str(error):
            raise
        return fit({"maxiter": 2000})


def _loss_deltas(
    reference: dict[tuple[int, str, str], tuple[float, float]],
    candidate: dict[tuple[int, str, str], tuple[float, float]],
) -> dict[tuple[int, str, str], tuple[float, float]]:
    if reference.keys() != candidate.keys():
        raise ValueError("paired model scoring requires identical team-season keys")
    return {
        key: (
            candidate[key][0] - reference[key][0],
            candidate[key][1] - reference[key][1],
        )
        for key in reference
    }


def _deltas_by_season(
    deltas: dict[tuple[int, str, str], tuple[float, float]], metric_index: int = 0
) -> dict[int, list[float]]:
    result: dict[int, list[float]] = defaultdict(list)
    for (season, _subdivision, _team_id), values in deltas.items():
        result[season].append(values[metric_index])
    return dict(result)


def _improvement_distribution(
    deltas: dict[tuple[int, str, str], tuple[float, float]],
) -> dict[str, Any]:
    values = [item[0] for item in deltas.values()]
    improvements = sorted((-value for value in values if value < 0), reverse=True)
    positive_total = math.fsum(improvements)
    return {
        "team_seasons_better": sum(value < 0 for value in values),
        "team_season_fraction_better": sum(value < 0 for value in values) / len(values),
        "team_seasons_worse": sum(value > 0 for value in values),
        "best_team_season_delta_nll": min(values),
        "worst_team_season_delta_nll": max(values),
        "top_10_share_of_gross_nll_improvements": (
            math.fsum(improvements[:10]) / positive_total if positive_total else 0.0
        ),
    }


def _evaluation_outputs(
    train_eval_rows: list[TeamSeason],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, DirectRankModel],
    dict[str, list[Any]],
    dict[str, dict[tuple[int, str, str], tuple[float, float]]],
]:
    training_rows = [
        row for row in train_eval_rows if PRIMARY_START_SEASON <= row.season <= 2021
    ]
    evaluation_rows = [row for row in train_eval_rows if 2022 <= row.season <= 2025]
    if not training_rows or not evaluation_rows:
        raise ValueError(
            "issue 187 needs both 2013–2021 training and 2022–2025 evaluation rows"
        )
    models: dict[str, DirectRankModel] = {}
    predictions: dict[str, list[Any]] = {}
    losses: dict[str, dict[tuple[int, str, str], tuple[float, float]]] = {}
    score_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []

    for name, extra_features in MODEL_ARMS:
        model = _fit_arm(training_rows, extra_features)
        predicted = history_builder.make_predictions(model, evaluation_rows, name)
        models[name] = model
        predictions[name] = predicted
        losses[name] = history_builder.prediction_losses(predicted)

    base_losses = losses[BASELINE]
    pairwise_deltas: dict[str, dict[tuple[int, str, str], tuple[float, float]]] = {}
    bootstrap_summary: dict[str, Any] = {}
    for name, _features in MODEL_ARMS:
        score = prior_builder.score_predictions(predictions[name])
        if name == BASELINE:
            deltas = {key: (0.0, 0.0) for key in base_losses}
            bootstrap = {
                "mean_delta": 0.0,
                "central_95_interval": [0.0, 0.0],
                "fraction_bootstrap_better": 0.0,
                "n_team_seasons": len(base_losses),
                "n_season_clusters": len({key[0] for key in base_losses}),
                "n_resamples": 0,
                "resampling": "reference arm",
            }
        else:
            deltas = _loss_deltas(base_losses, losses[name])
            bootstrap = paired_season_bootstrap(_deltas_by_season(deltas))
        pairwise_deltas[name] = deltas
        bootstrap_summary[name] = bootstrap
        distribution = _improvement_distribution(deltas)
        seasons_improved = sum(
            math.fsum(values) / len(values) < 0
            for values in _deltas_by_season(deltas).values()
        )
        ci = bootstrap["central_95_interval"]
        score_rows.append(
            {
                "model": name,
                "analysis_role": _analysis_role(name),
                "additional_features": ";".join(_features),
                "training_team_seasons": len(training_rows),
                "evaluation_team_seasons": len(predictions[name]),
                "evaluation_seasons": "2022;2023;2024;2025",
                "nll": float(score["nll"]),
                "delta_nll_vs_context_reference": float(score["nll"])
                - float(prior_builder.score_predictions(predictions[BASELINE])["nll"]),
                "crps": float(score["crps"]),
                "delta_crps_vs_context_reference": float(score["crps"])
                - float(prior_builder.score_predictions(predictions[BASELINE])["crps"]),
                "expected_rank_mae": float(score["expected_rank_mae"]),
                "median_rank_mae": float(score["median_rank_mae"]),
                "delta_nll_bootstrap_95_low": ci[0],
                "delta_nll_bootstrap_95_high": ci[1],
                "bootstrap_fraction_better": bootstrap["fraction_bootstrap_better"],
                "seasons_with_lower_nll": seasons_improved,
                **distribution,
            }
        )

    for season in range(2022, 2026):
        seasonal_predictions = {
            name: [item for item in predictions[name] if item.season == season]
            for name, _features in MODEL_ARMS
        }
        seasonal_scores = {
            name: prior_builder.score_predictions(values)
            for name, values in seasonal_predictions.items()
        }
        for name, _features in MODEL_ARMS:
            annual_rows.append(
                {
                    "season": season,
                    "model": name,
                    "analysis_role": _analysis_role(name),
                    "team_seasons": len(seasonal_predictions[name]),
                    "nll": float(seasonal_scores[name]["nll"]),
                    "delta_nll_vs_context_reference": float(
                        seasonal_scores[name]["nll"] - seasonal_scores[BASELINE]["nll"]
                    ),
                    "crps": float(seasonal_scores[name]["crps"]),
                    "delta_crps_vs_context_reference": float(
                        seasonal_scores[name]["crps"]
                        - seasonal_scores[BASELINE]["crps"]
                    ),
                    "expected_rank_mae": float(
                        seasonal_scores[name]["expected_rank_mae"]
                    ),
                }
            )

    combined_vs_individual = _loss_deltas(losses[INDIVIDUAL], losses[COMBINED])
    combined_ci = paired_season_bootstrap(_deltas_by_season(combined_vs_individual))
    continuity_deltas_by_season = _deltas_by_season(pairwise_deltas[CONTINUITY])
    continuity_total_deterioration = math.fsum(
        math.fsum(values) for values in continuity_deltas_by_season.values()
    )
    continuity_2025_deterioration_share = (
        math.fsum(continuity_deltas_by_season.get(2025, []))
        / continuity_total_deterioration
        if continuity_total_deterioration > 0
        else None
    )
    base_score = prior_builder.score_predictions(predictions[BASELINE])
    combined_score = prior_builder.score_predictions(predictions[COMBINED])
    individual_score = prior_builder.score_predictions(predictions[INDIVIDUAL])
    team_loss_rows: list[dict[str, Any]] = []
    row_by_key = {
        (row.season, row.subdivision, row.team_id): row for row in evaluation_rows
    }
    for key in sorted(base_losses):
        row = row_by_key[key]
        output: dict[str, Any] = {
            "season": row.season,
            "subdivision": row.subdivision,
            "team_id": row.team_id,
            "team_name": row.team_name,
            "team_population": row.population,
            "mean_observed_rank": float(np.mean(row.target_ranks)),
            "mean_observed_rank_fraction": float(
                np.mean(row.target_ranks) / row.population
            ),
        }
        for name, _features in MODEL_ARMS:
            output[f"nll_{name}"] = losses[name][key][0]
            output[f"crps_{name}"] = losses[name][key][1]
        for name, _features in MODEL_ARMS:
            if name == BASELINE:
                continue
            output[f"delta_nll_{name}_vs_context"] = pairwise_deltas[name][key][0]
        output["delta_nll_combined_vs_individual"] = combined_vs_individual[key][0]
        team_loss_rows.append(output)

    incremental = {
        "combined_delta_nll_vs_individual_experience": float(
            combined_score["nll"] - individual_score["nll"]
        ),
        "paired_season_bootstrap": combined_ci,
    }
    metrics = {
        "training_team_seasons": len(training_rows),
        "evaluation_team_seasons": len(evaluation_rows),
        "training_seasons": [2013, 2021],
        "evaluation_seasons": [2022, 2025],
        "same_keys_all_arms": all(
            predictions[name][index].key == predictions[BASELINE][index].key
            for name, _features in MODEL_ARMS
            for index in range(len(predictions[BASELINE]))
        ),
        "incremental_continuity_beyond_individual_experience": incremental,
        "continuity_total_heldout_nll_deterioration": continuity_total_deterioration,
        "continuity_2025_share_of_total_heldout_nll_deterioration": (
            continuity_2025_deterioration_share
        ),
        "post_hoc_single_feature_arm": POST_HOC_MODEL_ARMS[0][0],
        "post_hoc_single_feature_delta_nll_vs_context": next(
            row["delta_nll_vs_context_reference"]
            for row in score_rows
            if row["model"] == POST_HOC_MODEL_ARMS[0][0]
        ),
        "decision_rule_for_currently_tested_approach": (
            "Refine"
            if next(
                row["delta_nll_vs_context_reference"]
                for row in score_rows
                if row["model"] == POST_HOC_MODEL_ARMS[0][0]
            )
            < 0
            else "Reject"
        ),
        "baseline_nll": float(base_score["nll"]),
        "combined_nll": float(combined_score["nll"]),
        "bootstrap_by_model": bootstrap_summary,
    }
    return (
        score_rows,
        annual_rows,
        team_loss_rows,
        [],
        metrics,
        models,
        predictions,
        losses,
    )


def _feature_diagnostics(
    *,
    panel: list[dict[str, Any]],
    model_rows: list[TeamSeason],
    models: dict[str, DirectRankModel],
    evaluation_rows: list[TeamSeason],
    losses: dict[str, dict[tuple[int, str, str], tuple[float, float]]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    panel_by_key = {(int(row["season"]), str(row["team_id"])): row for row in panel}
    team_rows = {(row.season, row.subdivision, row.team_id): row for row in model_rows}
    feature_rows: list[dict[str, Any]] = []
    all_primary = [
        row
        for row in model_rows
        if PRIMARY_START_SEASON <= row.season <= PRIMARY_END_SEASON
    ]
    for feature in MODEL_FEATURES:
        values: list[float] = []
        performance: list[float] = []
        for row in all_primary:
            value = row.features.get(feature)
            if value is None:
                continue
            values.append(float(value))
            performance.append(float(np.mean(row.target_ranks) / row.population))
        correlation = (
            float(np.corrcoef(values, performance)[0, 1])
            if len(values) > 1 and np.std(values) > 0 and np.std(performance) > 0
            else None
        )
        feature_rows.append(
            {
                "feature": feature,
                "team_seasons_with_observed_feature": len(values),
                "feature_mean": float(np.mean(values)) if values else None,
                "feature_median": float(np.median(values)) if values else None,
                "feature_p10": float(np.quantile(values, 0.10)) if values else None,
                "feature_p90": float(np.quantile(values, 0.90)) if values else None,
                "pearson_r_vs_mean_final_rank_fraction": correlation,
                "interpretation": "descriptive; lower final-rank fraction is better",
            }
        )

    collinearity_pair = (
        INDIVIDUAL_EXPERIENCE_FEATURES[0],
        "ol_returning_group_share_4y",
    )
    collinearity_values = [
        (float(row.features[left]), float(row.features[right]))
        for row in all_primary
        for left, right in [collinearity_pair]
        if row.features.get(left) is not None and row.features.get(right) is not None
    ]
    feature_collinearity_rows = [
        {
            "feature_a": collinearity_pair[0],
            "feature_b": collinearity_pair[1],
            "team_seasons": len(collinearity_values),
            "pearson_r": (
                float(np.corrcoef(np.asarray(collinearity_values).T)[0, 1])
                if len(collinearity_values) > 1
                else None
            ),
            "analysis_scope": "primary 2013-2025 common analysis cohort",
            "interpretation": "post-hoc multicollinearity diagnostic; not an effect estimate",
        }
    ]

    eval_panel = [
        panel_by_key[(row.season, row.team_id)]
        for row in evaluation_rows
        if (row.season, row.team_id) in panel_by_key
    ]
    roster_sizes = [int(row["target_ol_player_count"]) for row in eval_panel]
    size_p10 = float(np.quantile(roster_sizes, 0.10)) if roster_sizes else 0.0
    size_p90 = float(np.quantile(roster_sizes, 0.90)) if roster_sizes else 0.0
    segment_predicates = {
        "stable_veteran_room": lambda row: (
            _number(row.get("ol_returning_group_share_4y")) >= 0.50
            and _number(row.get("ol_returning_player_share_4y")) >= 0.75
        ),
        "heavily_rebuilt_room": lambda row: (
            _number(row.get("ol_returning_player_share_4y")) <= 0.25
        ),
        "transfer_heavy_by_prior_other_school_share": lambda row: (
            _number(row.get("ol_prior_other_program_share_4y")) >= 0.50
        ),
        "small_target_ol_pool_p10": lambda row: (
            int(row["target_ol_player_count"]) <= size_p10
        ),
        "large_target_ol_pool_p90": lambda row: (
            int(row["target_ol_player_count"]) >= size_p90
        ),
        "incomplete_pair_or_identity_coverage": lambda row: (
            not bool(row["target_ol_identity_complete"])
            or float(row.get("pair_coverage_all_history") or 0.0) < 1.0
        ),
    }
    rules = {
        "stable_veteran_room": "returning group share >= 0.50 and returning-player share >= 0.75",
        "heavily_rebuilt_room": "returning-player share <= 0.25",
        "transfer_heavy_by_prior_other_school_share": "at least half of target OL IDs appeared at another school during T-4..T-1; other-school history is diagnostic only",
        "small_target_ol_pool_p10": f"target OL pool size <= evaluation p10 ({size_p10:.1f})",
        "large_target_ol_pool_p90": f"target OL pool size >= evaluation p90 ({size_p90:.1f})",
        "incomplete_pair_or_identity_coverage": "target IDs incomplete or all-history pair coverage below 100%",
    }
    eval_losses = {name: losses[name] for name, _features in MODEL_ARMS}
    segment_rows: list[dict[str, Any]] = []
    for segment, predicate in segment_predicates.items():
        keys = [
            (int(row["season"]), "fbs", str(row["team_id"]))
            for row in eval_panel
            if predicate(row)
        ]
        observed = [team_rows[key] for key in keys if key in team_rows]
        output: dict[str, Any] = {
            "segment": segment,
            "classification_rule": rules[segment],
            "team_seasons": len(observed),
            "evaluation_seasons": len({row.season for row in observed}),
            "mean_final_rank_fraction": (
                float(
                    np.mean(
                        [np.mean(row.target_ranks) / row.population for row in observed]
                    )
                )
                if observed
                else None
            ),
            "mean_target_ol_pool_size": (
                float(
                    np.mean(
                        [
                            int(
                                panel_by_key[(row.season, row.team_id)][
                                    "target_ol_player_count"
                                ]
                            )
                            for row in observed
                        ]
                    )
                )
                if observed
                else None
            ),
        }
        base_keys = set(eval_losses[BASELINE])
        selected_keys = [key for key in keys if key in base_keys]
        for name, _features in MODEL_ARMS:
            if name == BASELINE or not selected_keys:
                output[f"mean_delta_nll_{name}_vs_context"] = (
                    0.0 if name == BASELINE else None
                )
                continue
            output[f"mean_delta_nll_{name}_vs_context"] = float(
                np.mean(
                    [
                        eval_losses[name][key][0] - eval_losses[BASELINE][key][0]
                        for key in selected_keys
                    ]
                )
            )
        segment_rows.append(output)

    coefficient_rows: list[dict[str, Any]] = []
    for arm, _features in MODEL_ARMS:
        if arm == BASELINE:
            continue
        model = models[arm]
        extras = dict(MODEL_ARMS)[arm]
        n_features = len(model.feature_names)
        transformed = model.preprocessor.transform(
            [row.features for row in evaluation_rows]
        )
        beta = model.beta[model.lag_count :]
        for feature in extras:
            index = model.feature_names.index(feature)
            numeric_coefficient = float(beta[1 + index])
            missingness_coefficient = float(beta[1 + n_features + index])
            contributions = [
                numeric_coefficient * values[index]
                + missingness_coefficient * values[n_features + index]
                for values in transformed
            ]
            coefficient_rows.append(
                {
                    "model": arm,
                    "feature": feature,
                    "training_median": model.preprocessor.medians[feature],
                    "training_mean_after_imputation": model.preprocessor.means[feature],
                    "training_scale": model.preprocessor.scales[feature],
                    "standardized_location_coefficient": numeric_coefficient,
                    "missingness_location_coefficient": missingness_coefficient,
                    "mean_heldout_location_contribution": float(np.mean(contributions)),
                    "heldout_team_seasons": len(contributions),
                    "coefficient_note": "structural model term, not a causal effect",
                }
            )
    return feature_rows + coefficient_rows, segment_rows, feature_collinearity_rows


def _number(value: Any) -> float:
    return float(value) if value not in (None, "") else 0.0


def _coverage_sensitivity(
    *,
    evaluation_rows: list[TeamSeason],
    losses: dict[str, dict[tuple[int, str, str], tuple[float, float]]],
    official_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validated: dict[tuple[int, str], str] = {}
    for row in official_rows:
        season = int(row["season"])
        team_id = str(row.get("team_id") or "")
        if not team_id:
            continue
        if row.get("official_pool_mapping_status") != "complete":
            group = "official_identity_incomplete"
        elif str(row.get("cfbd_exact_player_set_match", "")).lower() == "true":
            group = "official_pool_exact_set_match"
        else:
            group = "official_pool_target_mismatch"
        validated[(season, team_id)] = group
    groups: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for row in evaluation_rows:
        key = (row.season, row.subdivision, row.team_id)
        state = validated.get(
            (row.season, row.team_id), "not_in_issue185_validation_sample"
        )
        groups[state].append(key)
        groups["all_primary_evaluation_rows"].append(key)

    output: list[dict[str, Any]] = []
    for group, keys in sorted(groups.items()):
        for arm, _features in MODEL_ARMS:
            if arm == BASELINE:
                continue
            deltas = [losses[arm][key][0] - losses[BASELINE][key][0] for key in keys]
            years: dict[int, list[float]] = defaultdict(list)
            for key, value in zip(keys, deltas, strict=True):
                years[key[0]].append(value)
            interval: list[float] | None = None
            bootstrap_fraction: float | None = None
            status = "no_team_seasons"
            if deltas:
                status = "descriptive_only_small_or_uneven_sample"
                if len(deltas) >= 20 and len(years) >= 3:
                    bootstrap = paired_season_bootstrap(years)
                    interval = bootstrap["central_95_interval"]
                    bootstrap_fraction = bootstrap["fraction_bootstrap_better"]
                    status = "sample_supports_cluster_interval"
            output.append(
                {
                    "coverage_group": group,
                    "model": arm,
                    "analysis_role": _analysis_role(arm),
                    "team_seasons": len(deltas),
                    "seasons": len(years),
                    "mean_delta_nll_vs_context": float(np.mean(deltas))
                    if deltas
                    else None,
                    "cluster_bootstrap_95_low": interval[0] if interval else None,
                    "cluster_bootstrap_95_high": interval[1] if interval else None,
                    "bootstrap_fraction_better": bootstrap_fraction,
                    "interpretation_status": status,
                }
            )

    primary_official_rows = [
        row
        for row in official_rows
        if PRIMARY_START_SEASON <= int(row["season"]) <= PRIMARY_END_SEASON
    ]
    feature_shift_rows: list[dict[str, Any]] = []
    for feature in MODEL_FEATURES:
        differences = [
            float(row[f"official_minus_cfbd_{feature}"])
            for row in primary_official_rows
            if row.get("official_pool_mapping_status") == "complete"
            and row.get(f"official_minus_cfbd_{feature}") not in (None, "")
        ]
        values = np.asarray(differences, dtype=float)
        feature_shift_rows.append(
            {
                "feature": feature,
                "validated_primary_team_seasons": len(differences),
                "mean_official_minus_cfbd": float(np.mean(values))
                if len(values)
                else None,
                "mean_absolute_feature_change": float(np.mean(np.abs(values)))
                if len(values)
                else None,
                "median_absolute_feature_change": float(np.median(np.abs(values)))
                if len(values)
                else None,
                "p90_absolute_feature_change": float(np.quantile(np.abs(values), 0.90))
                if len(values)
                else None,
                "max_absolute_feature_change": float(np.max(np.abs(values)))
                if len(values)
                else None,
                "total_validated_primary_team_seasons": len(primary_official_rows),
            }
        )

    return output, feature_shift_rows


def _official_prediction_sensitivity(
    *,
    models: dict[str, DirectRankModel],
    evaluation_rows: list[TeamSeason],
    current_predictions: dict[str, list[Any]],
    official_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    official = {
        (int(row["season"]), str(row.get("team_id") or "")): row
        for row in official_rows
        if row.get("official_pool_mapping_status") == "complete"
    }
    official_eval: list[TeamSeason] = []
    for row in evaluation_rows:
        validation = official.get((row.season, row.team_id))
        if validation is None:
            continue
        if any(
            validation.get(f"official_{feature}") in (None, "")
            for feature in MODEL_FEATURES
        ):
            continue
        features = dict(row.features)
        features.update(
            {
                feature: float(validation[f"official_{feature}"])
                for feature in MODEL_FEATURES
            }
        )
        official_eval.append(replace(row, features=features))

    if not official_eval:
        return [
            {
                "model": arm,
                "analysis_role": _analysis_role(arm),
                "team_seasons": 0,
                "seasons": 0,
                "cfbd_delta_nll_vs_context": None,
                "official_pool_delta_nll_vs_context": None,
                "official_minus_cfbd_delta_nll": None,
                "interpretation_status": "no_complete_official_pool_rows_in_2022_2025",
            }
            for arm, _features in MODEL_ARMS
            if arm != BASELINE
        ]

    base_keys = {(row.season, row.subdivision, row.team_id) for row in official_eval}
    baseline_loss_map = history_builder.prediction_losses(current_predictions[BASELINE])
    base_losses = {
        key: baseline_loss_map[key] for key in base_keys if key in baseline_loss_map
    }
    output: list[dict[str, Any]] = []
    for arm, _features in MODEL_ARMS:
        if arm == BASELINE:
            continue
        current_loss_map = history_builder.prediction_losses(current_predictions[arm])
        current_by_key = {
            key: current_loss_map[key] for key in base_keys if key in current_loss_map
        }
        official_predictions = history_builder.make_predictions(
            models[arm], official_eval, f"{arm}_official_pool_sensitivity"
        )
        official_losses = history_builder.prediction_losses(official_predictions)
        delta_cfbd = [
            current_by_key[key][0] - base_losses[key][0] for key in sorted(base_keys)
        ]
        delta_official = [
            official_losses[key][0] - base_losses[key][0] for key in sorted(base_keys)
        ]
        output.append(
            {
                "model": arm,
                "analysis_role": _analysis_role(arm),
                "team_seasons": len(official_eval),
                "seasons": len({row.season for row in official_eval}),
                "cfbd_delta_nll_vs_context": float(np.mean(delta_cfbd)),
                "official_pool_delta_nll_vs_context": float(np.mean(delta_official)),
                "official_minus_cfbd_delta_nll": float(
                    np.mean(delta_official) - np.mean(delta_cfbd)
                ),
                "interpretation_status": (
                    "descriptive_only_small_or_uneven_sample"
                    if len(official_eval) < 20
                    or len({row.season for row in official_eval}) < 3
                    else "clustered_sensitivity_available"
                ),
            }
        )
    return output


def _format(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.{digits}f}"


def _render_report(
    *,
    panel: list[dict[str, Any]],
    analysis_rows: list[TeamSeason],
    excluded: dict[str, int],
    scores: list[dict[str, Any]],
    annual: list[dict[str, Any]],
    feature_behavior: list[dict[str, Any]],
    feature_collinearity: list[dict[str, Any]],
    feature_coefficients: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    coverage_sensitivity: list[dict[str, Any]],
    feature_shifts: list[dict[str, Any]],
    official_rows: list[dict[str, Any]],
    official_prediction_sensitivity: list[dict[str, Any]],
    costart_summary: dict[str, Any],
    metrics: dict[str, Any],
    decision: str | None,
    recommendation: str | None,
    input_hashes: dict[str, dict[str, str]],
) -> str:
    primary_panel = [
        row
        for row in panel
        if PRIMARY_START_SEASON <= int(row["season"]) <= PRIMARY_END_SEASON
    ]
    statuses = Counter(str(row["feature_status"]) for row in primary_panel)
    exact_sample_rows = [
        row
        for row in read_csv(VALIDATION / "results/team_season_summary.csv")
        if PRIMARY_START_SEASON <= int(row["season"]) <= PRIMARY_END_SEASON
    ]
    complete_official = sum(
        row.get("official_pool_mapping_status") == "complete"
        for row in official_rows
        if PRIMARY_START_SEASON <= int(row["season"]) <= PRIMARY_END_SEASON
    )
    combined_coefficients = {
        row["feature"]: float(row["standardized_location_coefficient"])
        for row in feature_coefficients
        if row["model"] == COMBINED
    }
    individual_coefficients = {
        row["feature"]: float(row["standardized_location_coefficient"])
        for row in feature_coefficients
        if row["model"] == INDIVIDUAL
    }
    returning_player = INDIVIDUAL_EXPERIENCE_FEATURES[0]
    returning_group = "ol_returning_group_share_4y"
    collinearity = feature_collinearity[0]
    post_hoc_scores = {
        row["model"]: row for row in scores if row["model"] in POST_HOC_ARM_NAMES
    }
    single_feature_score = post_hoc_scores[POST_HOC_MODEL_ARMS[0][0]]
    experience_plus_single_score = post_hoc_scores[POST_HOC_MODEL_ARMS[1][0]]
    lines = [
        "# Issue 187: offensive-line shared-roster continuity experiment",
        "",
        f"**Decision:** {decision or 'Pending analyst review'}",
        "",
        f"**Recommendation:** {recommendation or 'Add the evidence-based recommendation after reviewing the generated results.'}",
        "",
        "## Scope and cohort",
        "",
        f"The feature panel covers {HISTORY_START_SEASON}–{PRIMARY_END_SEASON}; 2013–2025 is the primary window. Every primary feature uses the same four prior roster seasons, T-4 through T-1. 2010–2012 remains a partial-history descriptive panel only. The target cohort is the retrospective CFBD offensive-line roster for target season T. Pair evidence comes only from the same team ID and seasons strictly before T; player position in prior seasons is ignored.",
        "",
        f"The panel contains {len(primary_panel):,} primary-window team-seasons. Feature status counts: {', '.join(f'{name}={count}' for name, count in sorted(statuses.items()))}. The rank-prediction comparison retained {len(analysis_rows):,} identical team-seasons across all model arms. Exclusions: {', '.join(f'{name}={count}' for name, count in sorted(excluded.items())) or 'none'}.",
        "",
        "## Controlled evaluation",
        "",
        "The reference is a Context 1.3-style direct rank-distribution model refitted on the restricted 2013–2021 training cohort, with Context 1.3's rank-history, recruiting, talent, returning-production, coaching, and transfer inputs. It is not the currently active production Context 1.4 model. The research adapter holds that feature set, likelihood, training-only imputation and standardization, missingness indicators, location-only Context placement, optimizer, and 0.25 regularization fixed. New coefficients are fitted only on 2013–2021; the identical 2022–2025 FBS team-seasons are scored in every arm. The reported score is the repository's empirical rank-distribution negative log-likelihood (nats per team-season), not a game-level score.",
        "",
        "The two single-feature comparisons are post-hoc diagnostics added after the original holdout was inspected. They use the same train/test cohorts, model settings, target population, and score as the original arms; they are exploratory and do not count as independent confirmation.",
        "",
        "## Held-out predictive results",
        "",
        "Negative ΔNLL favors the added-feature model. The interval resamples held-out seasons as clusters and is descriptive with four evaluation seasons.",
        "",
        "| Model | Analysis role | Team-seasons | NLL | ΔNLL vs restricted reference | 95% season-bootstrap interval | Seasons better | Team-seasons improved | Top 10 share of gross gains |",
        "|:--|:--|--:|--:|--:|:--|--:|--:|--:|",
    ]
    for row in scores:
        ci = f"[{_format(row.get('delta_nll_bootstrap_95_low'))}, {_format(row.get('delta_nll_bootstrap_95_high'))}]"
        lines.append(
            f"| {row['model']} | {row['analysis_role']} | {row['evaluation_team_seasons']} | {_format(row['nll'])} | {_format(row['delta_nll_vs_context_reference'])} | {ci} | {row['seasons_with_lower_nll']} / 4 | {row['team_seasons_better']} ({_format(row['team_season_fraction_better'] * 100, 1)}%) | {_format(row['top_10_share_of_gross_nll_improvements'] * 100, 1)}% |"
        )
    lines.extend(
        [
            "",
            f"Post-hoc single-feature results: Context plus `{POST_HOC_SHARED_ROSTER_FEATURE}` had ΔNLL {_format(single_feature_score['delta_nll_vs_context_reference'])}; adding that feature to the individual-experience arm had ΔNLL {_format(experience_plus_single_score['delta_nll_vs_context_reference'])}. Under the stated decision rule, the first comparison maps to **{metrics['decision_rule_for_currently_tested_approach']}** for the currently tested approach. These are exploratory results on the inspected holdout, not independent confirmation.",
            "",
            f"Continuity beyond individual experience: combined ΔNLL versus the individual-experience arm is {_format(metrics['incremental_continuity_beyond_individual_experience']['combined_delta_nll_vs_individual_experience'])}, with season-cluster interval {_format(metrics['incremental_continuity_beyond_individual_experience']['paired_season_bootstrap']['central_95_interval'][0])} to {_format(metrics['incremental_continuity_beyond_individual_experience']['paired_season_bootstrap']['central_95_interval'][1])}.",
            "",
            "### By season",
            "",
            "| Season | Model | Analysis role | Team-seasons | NLL | ΔNLL vs restricted reference | ΔCRPS vs restricted reference |",
            "|--:|:--|:--|--:|--:|--:|--:|",
        ]
    )
    for row in annual:
        lines.append(
            f"| {row['season']} | {row['model']} | {row['analysis_role']} | {row['team_seasons']} | {_format(row['nll'])} | {_format(row['delta_nll_vs_context_reference'])} | {_format(row['delta_crps_vs_context_reference'])} |"
        )
    lines.extend(
        [
            "",
            f"The three-feature continuity arm deteriorated in all four evaluation years. 2025 accounts for {_format(float(metrics['continuity_2025_share_of_total_heldout_nll_deterioration']) * 100, 1)}% of its total held-out NLL deterioration when summed over team-seasons.",
        ]
    )
    lines.extend(
        [
            "",
            "## Feature behavior",
            "",
            "Correlations use the mean observed final rank divided by team population, where lower is better. They are descriptive associations, not causal effects. Coefficients use training-standardized features; missingness terms are shown separately.",
            "",
            "| Feature | Observed team-seasons | Mean | Median | P10 | P90 | Pearson r vs final-rank fraction |",
            "|:--|--:|--:|--:|--:|--:|--:|",
        ]
    )
    for row in feature_behavior:
        lines.append(
            f"| {row['feature']} | {row['team_seasons_with_observed_feature']} | {_format(row['feature_mean'])} | {_format(row['feature_median'])} | {_format(row['feature_p10'])} | {_format(row['feature_p90'])} | {_format(row['pearson_r_vs_mean_final_rank_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "### Collinearity and coefficient instability",
            "",
            "| Feature A | Feature B | Team-seasons | Pearson r | Scope |",
            "|:--|:--|--:|--:|:--|",
            f"| {collinearity['feature_a']} | {collinearity['feature_b']} | {collinearity['team_seasons']} | {_format(collinearity['pearson_r'], 3)} | {collinearity['analysis_scope']} |",
            "",
            f"Returning-player share and returning-group share correlate at approximately {_format(collinearity['pearson_r'], 3)}. In the combined bundle, their standardized coefficients are {_format(combined_coefficients[returning_player], 3)} and {_format(combined_coefficients[returning_group], 3)}, respectively, with opposing signs. The returning-player coefficient is {_format(individual_coefficients[returning_player], 3)} in the individual-experience-only bundle. This shift is consistent with unstable allocation across nearly redundant predictors; the bundle coefficients do not identify separate effects.",
            "",
            "| Model | Feature | Standardized location coefficient | Missingness coefficient | Mean held-out location contribution |",
            "|:--|:--|--:|--:|--:|",
        ]
    )
    for row in feature_coefficients:
        lines.append(
            f"| {row['model']} | {row['feature']} | {_format(row['standardized_location_coefficient'])} | {_format(row['missingness_location_coefficient'])} | {_format(row['mean_heldout_location_contribution'])} |"
        )
    lines.extend(
        [
            "",
            "### Held-out room profiles",
            "",
            "| Profile | Team-seasons | Seasons | Mean final-rank fraction | Mean OL pool size | ΔNLL vs restricted reference (individual / continuity / both) |",
            "|:--|--:|--:|--:|--:|:--|",
        ]
    )
    for row in segments:
        deltas = ", ".join(
            _format(row.get(f"mean_delta_nll_{arm}_vs_context"))
            for arm in (INDIVIDUAL, CONTINUITY, COMBINED)
        )
        lines.append(
            f"| {row['segment']} | {row['team_seasons']} | {row['evaluation_seasons']} | {_format(row['mean_final_rank_fraction'])} | {_format(row['mean_target_ol_pool_size'])} | {deltas} |"
        )
    lines.extend(
        [
            "",
            "## Roster-quality sensitivity",
            "",
            f"Issue 185's independent frozen sample contains {len(read_csv(VALIDATION / 'results/team_season_summary.csv'))} team-seasons (20 exact CFBD/official player sets and 20 mismatches). Of the sample rows in 2013–2025, {len(exact_sample_rows)} are in this study window; {complete_official} have a complete mapping from every official target-pool player to a unique existing CFBD roster identity. No IDs were inferred for players absent from or ambiguous in the existing CFBD history.",
            "",
            "For complete mappings, feature shifts compare recomputed official-pool values with CFBD-pool values on the same team-season. The frozen sample is stratified and small; these rates are not population estimates.",
            "",
            "| Feature | Mapped sample rows | Mean absolute change | Median absolute change | P90 absolute change | Maximum absolute change |",
            "|:--|--:|--:|--:|--:|--:|",
        ]
    )
    for row in feature_shifts:
        lines.append(
            f"| {row['feature']} | {row['validated_primary_team_seasons']} | {_format(row['mean_absolute_feature_change'])} | {_format(row['median_absolute_feature_change'])} | {_format(row['p90_absolute_feature_change'])} | {_format(row['max_absolute_feature_change'])} |"
        )
    lines.extend(
        [
            "",
            "### Held-out scores by validation coverage",
            "",
            "Rows not included in the issue 185 sample remain unclassified; they are not called high confidence. Any group without at least 20 team-seasons over three seasons is reported as descriptive only.",
            "",
            "| Coverage group | Model | Analysis role | Team-seasons | Seasons | ΔNLL vs restricted reference | 95% interval | Status |",
            "|:--|:--|:--|--:|--:|--:|:--|:--|",
        ]
    )
    for row in coverage_sensitivity:
        ci = (
            f"[{_format(row['cluster_bootstrap_95_low'])}, {_format(row['cluster_bootstrap_95_high'])}]"
            if row.get("cluster_bootstrap_95_low") is not None
            else "—"
        )
        lines.append(
            f"| {row['coverage_group']} | {row['model']} | {row['analysis_role']} | {row['team_seasons']} | {row['seasons']} | {_format(row['mean_delta_nll_vs_context'])} | {ci} | {row['interpretation_status']} |"
        )
    lines.extend(
        [
            "",
            "### Official-pool score substitution",
            "",
            "For target team-seasons in the held-out years with complete official-to-CFBD ID mapping, the already fitted models are rescored after substituting official-pool features. No coefficients are refit on validation data.",
            "",
            "| Model | Analysis role | Team-seasons | Seasons | CFBD-pool ΔNLL vs restricted reference | Official-pool ΔNLL vs restricted reference | Official minus CFBD | Status |",
            "|:--|:--|--:|--:|--:|--:|--:|:--|",
        ]
    )
    for row in official_prediction_sensitivity:
        lines.append(
            f"| {row['model']} | {row['analysis_role']} | {row['team_seasons']} | {row['seasons']} | {_format(row.get('cfbd_delta_nll_vs_context'))} | {_format(row.get('official_pool_delta_nll_vs_context'))} | {_format(row.get('official_minus_cfbd_delta_nll'))} | {row['interpretation_status']} |"
        )
    lines.extend(
        [
            "",
            "## Co-start comparison",
            "",
            f"The issue 181 pilot supplied {costart_summary['co_start_pair_rows']} positive co-start pairs; {costart_summary['co_start_pairs_matched_to_cfbd_target_pool']} matched both players to the CFBD target OL pool across {costart_summary['matched_team_seasons']} team-seasons. Spearman correlation between shared-start rate and four-year shared-roster seasons among matched pairs: {_format(costart_summary['spearman_rho_start_rate_vs_four_year_shared_roster_seasons'])}. {costart_summary['description']}",
            "",
            "## Limits and interpretation",
            "",
            "The target-season CFBD roster responses are retrospective and may include in-season arrivals/departures. Continuity evidence itself uses only earlier seasons, but the target cohort is not a preseason snapshot; this study is not proof of prospective deployability. CFBD's position errors and omissions remain visible through target-pool counts, unknown-position counts, identity/pair coverage flags, and the independent issue 185 sensitivity. The validation sample cannot correct historical rosters at scale. Rank distributions are measurements and the held-out score is an existing Context rank-likelihood score, not a direct game-outcome score.",
            "",
            "No production Context feature, coefficient, or ranking output was modified. The added-feature fitting path is research only.",
            "",
            "## Reproduction",
            "",
            "The script reads the committed #183 normalized roster panel, #185 validation inputs, #181 co-start pairs, the historical Context feature tables, and the locally generated rank-distribution artifact. Large raw/reacquirable CFBD inputs remain outside Git. Local coach-tenure snapshots are also needed by the existing Context row builder.",
            "",
            "```bash",
            "uv run python scripts/evaluate_offensive_line_continuity_issue_187.py --decision "
            + (decision or "Reject")
            + ' --recommendation "'
            + (recommendation or "Evidence-based issue 187 decision.").replace(
                '"', '\\"'
            )
            + '"',
            "uv run python scripts/evaluate_offensive_line_continuity_issue_187.py --decision "
            + (decision or "Reject")
            + ' --recommendation "'
            + (recommendation or "Evidence-based issue 187 decision.").replace(
                '"', '\\"'
            )
            + '" --check',
            "```",
            "",
            "Inputs and hashes are recorded in `manifest.json`; compact team-season features, paired losses, validation changes, and co-start joins are retained beside this report.",
            "",
            "### Inputs",
            "",
        ]
    )
    for key, value in sorted(input_hashes.items()):
        lines.append(f"- `{key}`: `{value['path']}` · SHA-256 `{value['sha256']}`")
    lines.append("")
    return "\n".join(lines)


def _run(
    *,
    output_dir: Path,
    decision: str | None,
    recommendation: str | None,
    check: bool,
) -> None:
    input_paths = _source_inventory()
    _check_required_sources(input_paths)
    input_hashes = {
        name: {"path": source_reference(path), "sha256": sha256_file(path)}
        for name, path in sorted(input_paths.items())
    }
    (
        panel,
        memberships,
        _player_seasons,
        roster_names,
        player_program_seasons,
        pair_rows,
        ol_rows,
    ) = _panel_inputs()
    official_rows = _official_sensitivity(
        panel=panel,
        roster_names=roster_names,
        memberships=memberships,
        player_program_seasons=player_program_seasons,
    )
    costart_rows, costart_summary = _co_start_comparison(
        ol_rows=ol_rows,
        pair_rows=pair_rows,
        player_program_seasons=player_program_seasons,
    )
    base_rows, _cold, _coverage = context_builder.load_candidate_rows()
    analysis_rows, excluded, _panel_by_key = _analysis_rows(base_rows, panel)
    if not analysis_rows:
        raise ValueError(
            "no common FBS Context/OL rows remain for the primary evaluation"
        )

    (
        score_rows,
        annual_rows,
        team_loss_rows,
        _unused,
        metrics,
        models,
        predictions,
        losses,
    ) = _evaluation_outputs(analysis_rows)
    if decision and decision != metrics["decision_rule_for_currently_tested_approach"]:
        raise ValueError(
            "decision must follow the post-hoc Context-plus-single-continuity result: "
            f"expected {metrics['decision_rule_for_currently_tested_approach']}"
        )
    evaluation_rows = [row for row in analysis_rows if 2022 <= row.season <= 2025]
    feature_diagnostic_rows, segment_rows, feature_collinearity_rows = (
        _feature_diagnostics(
            panel=panel,
            model_rows=analysis_rows,
            models=models,
            evaluation_rows=evaluation_rows,
            losses=losses,
        )
    )
    feature_behavior = [
        row
        for row in feature_diagnostic_rows
        if "pearson_r_vs_mean_final_rank_fraction" in row
    ]
    feature_coefficients = [
        row
        for row in feature_diagnostic_rows
        if "standardized_location_coefficient" in row
    ]
    coverage_rows, feature_shifts = _coverage_sensitivity(
        evaluation_rows=evaluation_rows,
        losses=losses,
        official_rows=official_rows,
    )
    official_prediction_rows = _official_prediction_sensitivity(
        models=models,
        evaluation_rows=evaluation_rows,
        current_predictions=predictions,
        official_rows=official_rows,
    )

    artifacts: dict[str, list[dict[str, Any]]] = {
        "feature_panel.csv": panel,
        "evaluation_aggregate.csv": score_rows,
        "evaluation_by_season.csv": annual_rows,
        "evaluation_team_losses.csv": team_loss_rows,
        "feature_behavior.csv": feature_behavior,
        "feature_collinearity.csv": feature_collinearity_rows,
        "feature_coefficients.csv": feature_coefficients,
        "heldout_room_profiles.csv": segment_rows,
        "coverage_sensitivity.csv": coverage_rows,
        "official_pool_sensitivity.csv": official_rows,
        "official_pool_prediction_sensitivity.csv": official_prediction_rows,
        "co_start_pair_comparison.csv": costart_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if check:
        mismatches = []
        for name, rows in artifacts.items():
            expected = output_dir / name
            if not expected.is_file():
                mismatches.append(f"missing {expected}")
                continue
            import tempfile

            with tempfile.TemporaryDirectory(prefix="issue187-check-") as temporary:
                generated = Path(temporary) / name
                write_csv(generated, rows)
                if generated.read_bytes() != expected.read_bytes():
                    mismatches.append(f"content differs: {expected}")
        if mismatches:
            raise ValueError("issue 187 output check failed: " + "; ".join(mismatches))
    else:
        for name, rows in artifacts.items():
            write_csv(output_dir / name, rows)

    sources = {
        key: {"path": item["path"], "sha256": item["sha256"]}
        for key, item in input_hashes.items()
    }
    manifest = {
        "issue": 187,
        "experiment": "offensive_line_shared_roster_continuity",
        "feature_history_start_season": HISTORY_START_SEASON,
        "lookback_seasons": LOOKBACK_SEASONS,
        "primary_target_seasons": [PRIMARY_START_SEASON, PRIMARY_END_SEASON],
        "training_seasons": [2013, 2021],
        "evaluation_seasons": [2022, 2025],
        "decision": decision,
        "recommendation": recommendation,
        "context13_semantic_specification_sha256": context13_semantic_specification_sha256(),
        "model_arms": [
            {
                "name": name,
                "analysis_role": _analysis_role(name),
                "additional_features": list(features),
            }
            for name, features in MODEL_ARMS
        ],
        "post_hoc_diagnostic_disclosure": (
            "The two single-feature diagnostics were added after inspection of the 2022-2025 holdout; they are exploratory and are not independent confirmation."
        ),
        "decision_rule": (
            "Refine only if the post-hoc Context-plus-mean-shared-roster feature has lower held-out NLL than the restricted Context reference; otherwise Reject the currently tested approach."
        ),
        "feature_definitions": {
            CONTINUITY_FEATURES[
                0
            ]: "mean shared same-program roster seasons over all target-OL pairs in T-4..T-1",
            CONTINUITY_FEATURES[
                1
            ]: "fraction of target-OL pairs sharing at least one same-program roster season in T-4..T-1",
            CONTINUITY_FEATURES[
                2
            ]: "largest same-program prior-season OL subgroup divided by target OL pool size; groups smaller than two contribute zero",
            INDIVIDUAL_EXPERIENCE_FEATURES[
                0
            ]: "fraction of target OL with any same-program roster season in T-4..T-1",
            INDIVIDUAL_EXPERIENCE_FEATURES[
                1
            ]: "mean count of same-program roster seasons per target OL in T-4..T-1",
            DIAGNOSTIC_FEATURES[
                0
            ]: "fraction of target OL identities present at another program during T-4..T-1; diagnostic only, not a model input",
        },
        "fitting": {
            "distribution_family": CONTEXT13_DISTRIBUTION_FAMILY,
            "penalty": FROZEN_PENALTY,
            "minimum_scale": CONTEXT13_MINIMUM_SCALE,
            "lag_count": 1,
            "scale_features": list(SCALE_FEATURE_NAMES),
            "location_features": "Context 1.3 features plus the arm's OL features",
            "missingness": "training-median imputation, training mean/scale, missingness indicators",
            "season_cluster_bootstrap_seed": 7,
        },
        "cohort": {
            "training_team_seasons": metrics["training_team_seasons"],
            "evaluation_team_seasons": metrics["evaluation_team_seasons"],
            "same_keys_all_arms": metrics["same_keys_all_arms"],
            "excluded_target_pool_rows": excluded,
        },
        "input_artifacts": sources,
        "source_code_sha256": {
            "analysis_script": sha256_file(Path(__file__)),
            "feature_module": sha256_file(
                ROOT / "src/gippyrank/research/offensive_line_continuity_experiment.py"
            ),
        },
        "results": metrics,
        "co_start_summary": costart_summary,
        "caveats": [
            "The CFBD target-season OL cohort is retrospective, not an archived preseason roster.",
            "The official #185 target-pool sensitivity is limited to the existing stratified manual sample and unique mappings into existing CFBD identities.",
            "The #181 co-start comparison is positive-pair-only and descriptive.",
            "The single-continuity-feature comparisons are post-hoc diagnostics on an already inspected holdout, not independent confirmation.",
            "Held-out rank NLL is not a direct game-outcome score or proof of prospective deployability.",
        ],
    }
    report = _render_report(
        panel=panel,
        analysis_rows=analysis_rows,
        excluded=excluded,
        scores=score_rows,
        annual=annual_rows,
        feature_behavior=feature_behavior,
        feature_collinearity=feature_collinearity_rows,
        feature_coefficients=feature_coefficients,
        segments=segment_rows,
        coverage_sensitivity=coverage_rows,
        feature_shifts=feature_shifts,
        official_rows=official_rows,
        official_prediction_sensitivity=official_prediction_rows,
        costart_summary=costart_summary,
        metrics=metrics,
        decision=decision,
        recommendation=recommendation,
        input_hashes=input_hashes,
    )
    if check:
        manifest_path = output_dir / "manifest.json"
        report_path = output_dir / "report.md"
        if not manifest_path.is_file() or not report_path.is_file():
            raise ValueError("issue 187 output check failed: manifest/report missing")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"content differs: {manifest_path}")
        if report_path.read_text(encoding="utf-8") != report:
            raise ValueError(f"content differs: {report_path}")
    else:
        write_json(output_dir / "manifest.json", manifest)
        (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT,
        help="Output directory for compact research artifacts.",
    )
    parser.add_argument(
        "--decision", choices=("Advance", "Refine", "Reject"), default=None
    )
    parser.add_argument("--recommendation", default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Recompute all outputs and verify byte-for-byte equality without writing.",
    )
    args = parser.parse_args()
    if bool(args.decision) != bool(args.recommendation):
        parser.error("--decision and --recommendation must be supplied together")
    _run(
        output_dir=args.output_dir,
        decision=args.decision,
        recommendation=args.recommendation,
        check=args.check,
    )


if __name__ == "__main__":
    main()
