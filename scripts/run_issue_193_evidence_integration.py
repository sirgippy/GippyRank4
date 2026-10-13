"""Run issue 193's frozen evidence-dependent Context integration comparison."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gippyrank.posterior.engine import Team
from gippyrank.research import issue_201_prebowl_forecast as issue201
from gippyrank.research.context_incremental_information import (
    write_deterministic_gzip_jsonl,
)
from gippyrank.research.issue_193_evidence_integration import (
    EvidenceInput,
    RankTrainingExample,
    evidence_weight,
    fit_outer_fold,
    grouped_examples,
    input_rows_hash,
    predict_rank_distribution,
    team_evidence_game_ids,
)
from gippyrank.research.matched_pipeline_replay import replay_pipeline_bundle

SEASONS = (2022, 2023, 2024, 2025)
EVALUATION_SEASONS = (2023, 2024, 2025)
MODEL_NAMES = (
    "History_existing",
    "Context_existing",
    "History_calibrated",
    "History_calibrated_plus_Context",
)
BASELINE_MODEL_NAMES = ("History_existing", "Context_existing")
CALIBRATED_MODEL_NAMES = (
    "History_calibrated",
    "History_calibrated_plus_Context",
)
CANDIDATE_MODEL = "History_calibrated_plus_Context"
HISTORY_CONTROL_MODEL = "History_calibrated"
CONTEXT_BASELINE_MODEL = "Context_existing"
FORECAST_BASELINE_METRICS = issue201.FORECAST_SCORE_FIELDS
RANK_BASELINE_METRICS = issue201.RANK_SCORE_FIELDS
GAME_SUPPORT_METRICS = (
    "crps",
    "margin_nll",
    "median_absolute_error",
    "home_win_brier",
    "coverage_50",
    "coverage_80",
    "coverage_95",
)
RANK_REPORT_METRICS = (
    "nll",
    "crps",
    "expected_rank_error",
    "interval_80_target_mass",
    "interval_80_width",
)
USEFUL_GAIN_MARGIN = 0.05
EARLY_CRPS_POOLED_TOLERANCE = 0.10
EARLY_CRPS_SEASON_TOLERANCE = 0.25
EARLY_CALIBRATION_POOLED_TOLERANCE = 0.03
EARLY_CALIBRATION_SEASON_TOLERANCE = 0.05
BASELINE_PARITY_TOLERANCE = 1e-8


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(value) for field, value in row.items()})


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if value is None:
        return ""
    return value


def file_record(path: Path, label: str) -> dict[str, object]:
    return {
        "label": label,
        "bytes": path.stat().st_size,
        "sha256": issue201.sha256_file(path),
    }


def verify_issue200_outputs(root: Path, manifest: dict[str, Any]) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise TypeError("issue 200 manifest lacks output hashes")
    for name in (
        "pipeline_bundle.json",
        "fitted_models.json",
        "preseason_feature_manifest.json",
        "target_snapshot_manifest.csv",
    ):
        path = root / name
        record = outputs.get(name)
        if not path.is_file() or not isinstance(record, dict):
            raise FileNotFoundError(f"issue 200 input artifact is missing: {name}")
        if record.get("sha256") != issue201.sha256_file(path):
            raise ValueError(
                f"issue 200 artifact hash differs from its manifest: {name}"
            )


def verify_issue201_outputs(root: Path, manifest: dict[str, Any]) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise TypeError("issue 201 manifest lacks output hashes")
    for name in (
        "schedule_plan.json",
        "schedule_plan.manifest.json",
        "forecast_scores.csv",
        "ranking_scores.csv",
        "replay_evidence.jsonl.gz",
    ):
        path = root / name
        record = outputs.get(name)
        if not path.is_file() or not isinstance(record, dict):
            raise FileNotFoundError(f"issue 201 input artifact is missing: {name}")
        if record.get("sha256") != issue201.sha256_file(path):
            raise ValueError(
                f"issue 201 artifact hash differs from its manifest: {name}"
            )


def frozen_protocol_commit(repo_root: Path, protocol_path: Path) -> str:
    """Require the preregistration to be tracked and unchanged from HEAD."""
    resolved = protocol_path.resolve()
    try:
        relative = resolved.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError(
            "the issue 193 protocol must be inside the repository"
        ) from error
    relative_text = relative.as_posix()
    tracked = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", relative_text],
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise ValueError(
            "the issue 193 protocol must be committed before candidate fitting"
        )
    unchanged = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--quiet", "HEAD", "--", relative_text],
        capture_output=True,
        text=True,
        check=False,
    )
    if unchanged.returncode != 0:
        raise ValueError("the issue 193 protocol has uncommitted changes")
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "log", "-1", "--format=%H", "--", relative_text],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not commit:
        raise ValueError("could not identify the frozen issue 193 protocol commit")
    return commit


def _evidence_replay_index(path: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    indexed = {}
    for row in read_gzip_jsonl(path):
        key = (int(row["season"]), str(row["cutoff_utc"]), str(row["policy"]))
        if key in indexed:
            raise ValueError(f"duplicate #201 replay-evidence row: {key}")
        indexed[key] = row
    return indexed


def verify_replay_evidence(
    observed: dict[str, object],
    retained: dict[str, Any],
    *,
    season: int,
    cutoff: str,
) -> None:
    for field in (
        "included_game_ids",
        "included_game_ids_sha256",
        "included_game_rows_sha256",
        "included_game_count",
        "excluded_game_ids",
        "excluded_game_ids_sha256",
        "excluded_game_count",
    ):
        if observed.get(field) != retained.get(field):
            raise ValueError(f"{season}/{cutoff}: #201 evidence differs for {field}")


def _empty_evidence(season: int) -> dict[str, object]:
    empty_hash = issue201.sha256_json([])
    return {
        "season": season,
        "cutoff_utc": "preseason_no_evidence",
        "policy": "existing",
        "included_game_ids": [],
        "included_game_ids_sha256": empty_hash,
        "included_game_rows_sha256": empty_hash,
        "included_game_count": 0,
        "excluded_game_ids": [],
        "excluded_game_ids_sha256": empty_hash,
        "excluded_game_count": 0,
        "latest_included_kickoff_utc": None,
    }


def _rank_strata(
    season: int,
    team_id: str,
    history_priors: dict[str, dict[str, object]],
    context_priors: dict[str, dict[str, object]],
    transfer_coverage: dict[tuple[int, str], bool],
) -> dict[str, object]:
    history_method = str(history_priors[team_id]["prior_method"])
    context_method = str(context_priors[team_id]["prior_method"])
    fallback = (
        history_method != "same_subdivision_lag1"
        or context_method != "same_subdivision_lag1"
    )
    transfer_available = bool(transfer_coverage.get((season, team_id), False))
    return {
        "component_status": "native_cold_start_fallback"
        if fallback
        else "fitted_prior",
        "history_prior_method": history_method,
        "context_prior_method": context_method,
        "repaired_transfer_feature_available": transfer_available,
        "repaired_transfer_coverage": "available" if transfer_available else "missing",
    }


def _transfer_coverage_map(path: Path) -> dict[tuple[int, str], bool]:
    manifest = read_json(path)
    result = {}
    for season_text, record in manifest.items():
        if not isinstance(record, dict):
            continue
        for row in record.get("predictor_rows", []):
            result[(int(season_text), str(row["team_id"]))] = bool(
                row["repaired_transfer_feature_available"]
            )
    return result


def _make_rank_score_row(
    *,
    season: int,
    cutoff: str,
    checkpoint_index: int,
    checkpoint_phase: str,
    checkpoint_has_games: bool,
    team_id: str,
    team_name: str,
    model_arm: str,
    prior_family: str,
    prediction: np.ndarray,
    target: np.ndarray,
    inputs: EvidenceInput,
    target_row: dict[str, object],
    target_observation_date: str,
    evidence: dict[str, object],
    strata: dict[str, object],
) -> dict[str, object]:
    score = issue201.score_rank_pmf(prediction, target)
    return {
        "season": season,
        "cutoff_utc": cutoff,
        "checkpoint_index": checkpoint_index,
        "checkpoint_phase": checkpoint_phase,
        "checkpoint_label": "preseason" if checkpoint_index == 1 else checkpoint_phase,
        "checkpoint_has_primary_games": checkpoint_has_games,
        "team_id": team_id,
        "team_name": team_name,
        "model_arm": model_arm,
        "policy": "existing",
        "prior_family": prior_family,
        "target_policy": "pre_bowl_204",
        "rank_count": len(target),
        "target_observation_date": target_observation_date,
        "evidence_game_count": evidence["included_game_count"],
        "evidence_ids_sha256": evidence["included_game_ids_sha256"],
        "team_evidence_game_count": inputs.evidence_count,
        "team_evidence_game_ids": list(inputs.team_game_ids),
        "team_history_kl_from_prior": float(inputs.raw_evidence_features[1]),
        "team_evidence_ids_sha256": issue201.sha256_json(list(inputs.team_game_ids)),
        **strata,
        **score,
    }


def _check_score_parity(
    retained_rows: list[dict[str, str]],
    computed_rows: list[dict[str, object]],
    *,
    model_names: tuple[str, ...],
    metrics: tuple[str, ...],
    label: str,
) -> dict[str, object]:
    retained = {
        (
            int(row["season"]),
            str(row.get("forecast_stage", "")),
            str(row.get("cutoff_utc", "")),
            str(row.get("game_id", row.get("team_id", ""))),
            str(row["model_arm"]),
        ): row
        for row in retained_rows
        if row.get("model_arm") in model_names
        and int(row["season"]) in EVALUATION_SEASONS
    }
    computed = {
        (
            int(row["season"]),
            str(row.get("forecast_stage", "")),
            str(row.get("cutoff_utc", "")),
            str(row.get("game_id", row.get("team_id", ""))),
            str(row["model_arm"]),
        ): row
        for row in computed_rows
        if row.get("model_arm") in model_names
        and int(row["season"]) in EVALUATION_SEASONS
    }
    if set(retained) != set(computed):
        missing = sorted(set(retained) - set(computed))[:5]
        extra = sorted(set(computed) - set(retained))[:5]
        raise ValueError(
            f"{label} row population differs: missing={missing}, extra={extra}"
        )
    maximum_error = {metric: 0.0 for metric in metrics}
    checked = 0
    for key, expected in retained.items():
        observed = computed[key]
        for metric in metrics:
            error = abs(float(observed[metric]) - float(expected[metric]))
            maximum_error[metric] = max(maximum_error[metric], error)
            if error > BASELINE_PARITY_TOLERANCE:
                raise ValueError(f"{label} {key} {metric} differs by {error}")
        if "evidence_ids_sha256" in expected and str(
            observed.get("evidence_ids_sha256", "")
        ) != str(expected["evidence_ids_sha256"]):
            raise ValueError(f"{label} {key} evidence ID hash differs")
        checked += 1
    return {
        "rows_checked": checked,
        "metrics": maximum_error,
        "tolerance": BASELINE_PARITY_TOLERANCE,
        "model_arms": list(model_names),
    }


def _forecast_job(
    *,
    metadata: dict[str, object],
    outcome: dict[str, object],
    model_arm: str,
    family: str,
    teams: dict[str, Team],
    likelihood: object,
    evidence: dict[str, object],
    stage: str,
    cutoff: str,
) -> dict[str, object]:
    return {
        "metadata": metadata,
        "outcome": outcome,
        "arm": model_arm,
        "policy": "existing",
        "family": family,
        "posterior_teams": teams,
        "likelihood": likelihood,
        "evidence": evidence,
        "stage": stage,
        "cutoff": cutoff,
    }


def _corrected_team_map(
    base_teams: dict[str, Team],
    corrected_pmfs: dict[str, np.ndarray],
) -> dict[str, Team]:
    result = {}
    for team_id, team in base_teams.items():
        pmf = corrected_pmfs.get(team_id)
        result[team_id] = Team(
            team.team_id,
            team.name,
            team.subdivision,
            team.prior if pmf is None else pmf,
        )
    return result


def _season_game_counts(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("forecast_stage") == "rolling"
            and row.get("forecast_population") == "fbs_fbs"
        ):
            groups[(int(row["season"]), str(row.get("phase") or "unknown"))].append(row)
    return [
        {"season": season, "phase": phase, "games": len(members)}
        for (season, phase), members in sorted(groups.items())
    ]


def _forecast_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["model_arm"]),
                str(row["forecast_stage"]),
                int(row["season"]),
                str(row.get("phase") or "unknown"),
            )
        ].append(row)
    summaries = []
    for (model, stage, season, phase), members in sorted(groups.items()):
        summary: dict[str, object] = {
            "model_arm": model,
            "forecast_stage": stage,
            "forecast_population": str(members[0]["forecast_population"]),
            "season": season,
            "phase": phase,
            "games": len(members),
        }
        for metric in GAME_SUPPORT_METRICS:
            summary[f"mean_{metric}"] = float(
                np.mean([float(row[metric]) for row in members])
            )
        for metric in (
            "expected_home_margin",
            "median_home_margin",
            "home_win_probability",
            "margin_interval_50_low",
            "margin_interval_50_high",
            "margin_interval_80_low",
            "margin_interval_80_high",
            "margin_interval_95_low",
            "margin_interval_95_high",
        ):
            summary[f"mean_{metric}"] = float(
                np.mean([float(row[metric]) for row in members])
            )
        summaries.append(summary)
    return summaries


def _primary_game_groups(
    rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    return {
        model: [
            row
            for row in rows
            if row.get("model_arm") == model
            and row.get("forecast_stage") == "rolling"
            and row.get("forecast_population") == "fbs_fbs"
            and int(row["season"]) in EVALUATION_SEASONS
        ]
        for model in MODEL_NAMES
    }


def _game_comparison_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = _primary_game_groups(rows)
    summaries: list[dict[str, object]] = []
    for model, members in groups.items():
        by_season: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in members:
            by_season[int(row["season"])].append(row)
        for aggregation in ("pooled_equal_game", "pooled_equal_season"):
            summary: dict[str, object] = {
                "model_arm": model,
                "aggregation": aggregation,
                "season": "",
                "games": len(members),
                "evaluation_seasons": len(by_season),
            }
            for metric in GAME_SUPPORT_METRICS:
                if aggregation == "pooled_equal_game":
                    value = float(np.mean([float(row[metric]) for row in members]))
                else:
                    value = float(
                        np.mean(
                            [
                                np.mean([float(row[metric]) for row in season_rows])
                                for season_rows in by_season.values()
                            ]
                        )
                    )
                summary[f"mean_{metric}"] = value
            summaries.append(summary)
        for season, season_rows in sorted(by_season.items()):
            summary = {
                "model_arm": model,
                "aggregation": "season",
                "season": season,
                "games": len(season_rows),
                "evaluation_seasons": 1,
            }
            for metric in GAME_SUPPORT_METRICS:
                summary[f"mean_{metric}"] = float(
                    np.mean([float(row[metric]) for row in season_rows])
                )
            summaries.append(summary)
    return summaries


def _paired_game_differences(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (
            int(row["season"]),
            str(row["forecast_stage"]),
            str(row["cutoff_utc"]),
            str(row["game_id"]),
            str(row["model_arm"]),
        ): row
        for row in rows
    }
    differences = []
    for candidate in CALIBRATED_MODEL_NAMES:
        if candidate != CANDIDATE_MODEL:
            continue
        for comparator in (CONTEXT_BASELINE_MODEL, HISTORY_CONTROL_MODEL):
            for key, row in by_key.items():
                season, stage, cutoff, game_id, model = key
                if model != candidate or season not in EVALUATION_SEASONS:
                    continue
                other = by_key[(season, stage, cutoff, game_id, comparator)]
                record: dict[str, object] = {
                    "season": season,
                    "forecast_stage": stage,
                    "cutoff_utc": cutoff,
                    "game_id": game_id,
                    "phase": row.get("phase"),
                    "forecast_population": row.get("forecast_population"),
                    "candidate": candidate,
                    "comparator": comparator,
                }
                for metric in GAME_SUPPORT_METRICS:
                    record[f"delta_{metric}"] = float(row[metric]) - float(
                        other[metric]
                    )
                differences.append(record)
    return differences


def _rank_weights(rows: list[dict[str, object]]) -> None:
    counts = Counter(
        (int(row["season"]), str(row["team_id"]))
        for row in rows
        if row["model_arm"] == "History_existing"
    )
    for row in rows:
        key = (int(row["season"]), str(row["team_id"]))
        row["team_season_weight"] = 1.0 / counts[key]


def _rank_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    by_checkpoint: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        by_checkpoint[
            (str(row["model_arm"]), int(row["season"]), str(row["cutoff_utc"]))
        ].append(row)
    checkpoint_means = []
    for (model, season, cutoff), members in sorted(by_checkpoint.items()):
        mean = {
            "model_arm": model,
            "season": season,
            "cutoff_utc": cutoff,
            "checkpoint_label": members[0]["checkpoint_label"],
            "checkpoint_has_primary_games": members[0]["checkpoint_has_primary_games"],
            "team_seasons": len(members),
        }
        for metric in RANK_REPORT_METRICS:
            mean[f"mean_{metric}"] = float(
                np.mean([float(row[metric]) for row in members])
            )
        checkpoint_means.append(mean)
        summaries.append({"aggregation": "checkpoint", **mean})

    for model in MODEL_NAMES:
        for season in EVALUATION_SEASONS:
            members = [
                row
                for row in rows
                if row["model_arm"] == model and int(row["season"]) == season
            ]
            if not members:
                continue
            weights = np.asarray([float(row["team_season_weight"]) for row in members])
            record: dict[str, object] = {
                "aggregation": "pooled_team_season_weighted",
                "model_arm": model,
                "season": season,
                "cutoff_utc": "",
                "checkpoint_label": "all_valid_checkpoints",
                "checkpoint_has_primary_games": "",
                "team_seasons": len({(season, str(row["team_id"])) for row in members}),
            }
            for metric in RANK_REPORT_METRICS:
                record[f"mean_{metric}"] = float(
                    np.average([float(row[metric]) for row in members], weights=weights)
                )
            summaries.append(record)

        for season in EVALUATION_SEASONS:
            valid = [
                row
                for row in checkpoint_means
                if row["model_arm"] == model
                and int(row["season"]) == season
                and bool(row["checkpoint_has_primary_games"])
            ]
            if valid:
                record = {
                    "aggregation": "equal_checkpoint_mean_valid_games",
                    "model_arm": model,
                    "season": season,
                    "cutoff_utc": "",
                    "checkpoint_label": "equal_mean_of_valid_checkpoints",
                    "checkpoint_has_primary_games": True,
                    "team_seasons": sum(int(row["team_seasons"]) for row in valid),
                    "valid_checkpoints": len(valid),
                }
                for metric in RANK_REPORT_METRICS:
                    record[f"mean_{metric}"] = float(
                        np.mean([float(row[f"mean_{metric}"]) for row in valid])
                    )
                summaries.append(record)

            preseason = [
                row
                for row in checkpoint_means
                if row["model_arm"] == model
                and int(row["season"]) == season
                and row["checkpoint_label"] == "preseason"
            ]
            if preseason:
                record = dict(preseason[0])
                record["aggregation"] = "preseason"
                summaries.append(record)

    for model in MODEL_NAMES:
        equal_season_rows = [
            row
            for row in summaries
            if row["aggregation"] == "pooled_team_season_weighted"
            and row["model_arm"] == model
        ]
        if equal_season_rows:
            record = {
                "aggregation": "equal_season_team_season_weighted",
                "model_arm": model,
                "season": "",
                "cutoff_utc": "",
                "checkpoint_label": "equal_season_mean",
                "checkpoint_has_primary_games": "",
                "team_seasons": sum(
                    int(row["team_seasons"]) for row in equal_season_rows
                ),
            }
            for metric in RANK_REPORT_METRICS:
                record[f"mean_{metric}"] = float(
                    np.mean([float(row[f"mean_{metric}"]) for row in equal_season_rows])
                )
            summaries.append(record)

        pooled = [
            row
            for row in rows
            if row["model_arm"] == model and int(row["season"]) in EVALUATION_SEASONS
        ]
        if pooled:
            weights = np.asarray([float(row["team_season_weight"]) for row in pooled])
            record = {
                "aggregation": "pooled_team_season_weighted_all_evaluation_seasons",
                "model_arm": model,
                "season": "",
                "cutoff_utc": "",
                "checkpoint_label": "all_valid_checkpoints",
                "checkpoint_has_primary_games": "",
                "team_seasons": len(
                    {(int(row["season"]), str(row["team_id"])) for row in pooled}
                ),
            }
            for metric in RANK_REPORT_METRICS:
                record[f"mean_{metric}"] = float(
                    np.average([float(row[metric]) for row in pooled], weights=weights)
                )
            summaries.append(record)
    return summaries


def _team_rank_scores(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (int(row["season"]), str(row["team_id"]), str(row["model_arm"]))
        ].append(row)
    result = []
    for (season, team_id, model), members in sorted(grouped.items()):
        weights = np.asarray([float(row["team_season_weight"]) for row in members])
        record: dict[str, object] = {
            "season": season,
            "team_id": team_id,
            "team_name": members[0]["team_name"],
            "model_arm": model,
            "team_season_weight_sum": float(weights.sum()),
            **{
                key: members[0][key]
                for key in (
                    "component_status",
                    "history_prior_method",
                    "context_prior_method",
                    "repaired_transfer_coverage",
                )
            },
        }
        for metric in RANK_REPORT_METRICS:
            record[f"mean_{metric}"] = float(
                np.average([float(row[metric]) for row in members], weights=weights)
            )
        result.append(record)
    return result


def _team_rank_differences(
    team_rank_scores: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_team_model = {
        (int(row["season"]), str(row["team_id"]), str(row["model_arm"])): row
        for row in team_rank_scores
    }
    differences = []
    for season in EVALUATION_SEASONS:
        team_ids = {
            team_id
            for row_season, team_id, _model in by_team_model
            if row_season == season
        }
        for team_id in sorted(team_ids, key=issue201._team_id_key):
            candidate = by_team_model[(season, team_id, CANDIDATE_MODEL)]
            for comparator in (CONTEXT_BASELINE_MODEL, HISTORY_CONTROL_MODEL):
                baseline = by_team_model[(season, team_id, comparator)]
                record: dict[str, object] = {
                    "season": season,
                    "team_id": team_id,
                    "team_name": candidate["team_name"],
                    "comparison": f"{CANDIDATE_MODEL}_minus_{comparator}",
                    "component_status": candidate["component_status"],
                    "repaired_transfer_coverage": candidate[
                        "repaired_transfer_coverage"
                    ],
                }
                for metric in RANK_REPORT_METRICS:
                    record[f"delta_{metric}"] = float(
                        candidate[f"mean_{metric}"] - baseline[f"mean_{metric}"]
                    )
                differences.append(record)
    return differences


def _strata_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["model_arm"]),
                int(row["season"]),
                str(row["component_status"]),
                str(row["repaired_transfer_coverage"]),
            )
        ].append(row)
    summaries = []
    for (model, season, status, transfer), members in sorted(groups.items()):
        weights = np.asarray([float(row["team_season_weight"]) for row in members])
        summary: dict[str, object] = {
            "model_arm": model,
            "season": season,
            "component_status": status,
            "repaired_transfer_coverage": transfer,
            "team_checkpoint_rows": len(members),
            "team_seasons": len({str(row["team_id"]) for row in members}),
        }
        for metric in RANK_REPORT_METRICS:
            summary[f"mean_{metric}"] = float(
                np.average([float(row[metric]) for row in members], weights=weights)
            )
        summaries.append(summary)
    return summaries


def _gain_concentration(
    game_differences: list[dict[str, object]],
    team_differences: list[dict[str, object]],
) -> list[dict[str, object]]:
    result = []
    sources = (
        ("game", game_differences, "delta_crps", "comparator"),
        ("team_season", team_differences, "delta_crps", "comparison"),
        ("team_season", team_differences, "delta_nll", "comparison"),
    )
    for unit, rows, metric, label_field in sources:
        labels = sorted({str(row[label_field]) for row in rows})
        for label in labels:
            values = np.asarray(
                [float(row[metric]) for row in rows if str(row[label_field]) == label],
                dtype=float,
            )
            if not len(values):
                continue
            gain = np.maximum(-values, 0.0)
            loss = np.maximum(values, 0.0)
            tail_count = max(1, int(np.ceil(0.1 * len(values))))
            largest_gain = np.sort(gain)[::-1][:tail_count]
            largest_loss = np.sort(loss)[::-1][:tail_count]
            result.append(
                {
                    "unit": unit,
                    "comparison": label,
                    "metric": metric.removeprefix("delta_"),
                    "rows": len(values),
                    "improved_share": float(np.mean(values < 0)),
                    "mean_delta_candidate_minus_comparator": float(np.mean(values)),
                    "median_delta_candidate_minus_comparator": float(np.median(values)),
                    "top_decile_share_of_total_gain": float(
                        largest_gain.sum() / gain.sum() if gain.sum() else 0.0
                    ),
                    "top_decile_share_of_total_loss": float(
                        largest_loss.sum() / loss.sum() if loss.sum() else 0.0
                    ),
                }
            )
    return result


def _candidate_weight_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["model_arm"])].append(row)
    summaries = []
    for model, members in sorted(groups.items()):
        weights = np.asarray([float(row["context_weight"]) for row in members])
        summaries.append(
            {
                "model_arm": model,
                "team_checkpoint_rows": len(members),
                "mean_context_weight": float(np.mean(weights)),
                "median_context_weight": float(np.median(weights)),
                "p10_context_weight": float(np.quantile(weights, 0.10)),
                "p90_context_weight": float(np.quantile(weights, 0.90)),
                "minimum_context_weight": float(np.min(weights)),
                "maximum_context_weight": float(np.max(weights)),
                "mean_team_evidence_games": float(
                    np.mean([int(row["team_evidence_game_count"]) for row in members])
                ),
                "mean_history_kl": float(
                    np.mean([float(row["history_kl_from_prior"]) for row in members])
                ),
            }
        )
    return summaries


def _game_summary_lookup(
    rows: list[dict[str, object]],
    *,
    model: str,
    stage: str,
    phase: str | None = None,
    season: int | None = None,
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row["model_arm"] == model
        and int(row["season"]) in EVALUATION_SEASONS
        and row["forecast_population"] == "fbs_fbs"
        and row["forecast_stage"] == stage
        and (phase is None or row.get("phase") == phase)
        and (season is None or int(row["season"]) == season)
    ]


def _calibration_error(rows: list[dict[str, object]]) -> float:
    if not rows:
        raise ValueError("cannot compute interval calibration without forecasts")
    return float(
        np.mean(
            [
                abs(
                    float(np.mean([float(row[f"coverage_{level}"]) for row in rows]))
                    - nominal
                )
                for level, nominal in ((50, 0.5), (80, 0.8), (95, 0.95))
            ]
        )
    )


def _early_guardrail(
    forecast_rows: list[dict[str, object]],
    *,
    candidate: str,
    comparator: str,
) -> dict[str, object]:
    phase_specs = (
        ("preseason", "preseason", None),
        ("pre_thanksgiving", "rolling", "pre_thanksgiving"),
    )
    results = []
    passed = True
    for label, stage, phase in phase_specs:
        candidate_all = _game_summary_lookup(
            forecast_rows, model=candidate, stage=stage, phase=phase
        )
        comparator_all = _game_summary_lookup(
            forecast_rows, model=comparator, stage=stage, phase=phase
        )
        if not candidate_all or len(candidate_all) != len(comparator_all):
            raise ValueError(f"{label}: early guardrail game population differs")
        crps_delta = float(
            np.mean([float(row["crps"]) for row in candidate_all])
            - np.mean([float(row["crps"]) for row in comparator_all])
        )
        cal_delta = _calibration_error(candidate_all) - _calibration_error(
            comparator_all
        )
        season_details = []
        for season in EVALUATION_SEASONS:
            candidate_rows = _game_summary_lookup(
                forecast_rows, model=candidate, stage=stage, phase=phase, season=season
            )
            comparator_rows = _game_summary_lookup(
                forecast_rows, model=comparator, stage=stage, phase=phase, season=season
            )
            if not candidate_rows or len(candidate_rows) != len(comparator_rows):
                raise ValueError(
                    f"{label}/{season}: early guardrail game population differs"
                )
            season_crps_delta = float(
                np.mean([float(row["crps"]) for row in candidate_rows])
                - np.mean([float(row["crps"]) for row in comparator_rows])
            )
            season_cal_delta = _calibration_error(candidate_rows) - _calibration_error(
                comparator_rows
            )
            season_passed = (
                season_crps_delta <= EARLY_CRPS_SEASON_TOLERANCE
                and season_cal_delta <= EARLY_CALIBRATION_SEASON_TOLERANCE
            )
            passed = passed and season_passed
            season_details.append(
                {
                    "season": season,
                    "games": len(candidate_rows),
                    "crps_delta": season_crps_delta,
                    "calibration_error_delta": season_cal_delta,
                    "passed": season_passed,
                }
            )
        phase_passed = (
            crps_delta <= EARLY_CRPS_POOLED_TOLERANCE
            and cal_delta <= EARLY_CALIBRATION_POOLED_TOLERANCE
        )
        passed = passed and phase_passed
        results.append(
            {
                "phase": label,
                "pooled_games": len(candidate_all),
                "crps_delta_candidate_minus_comparator": crps_delta,
                "crps_tolerance": EARLY_CRPS_POOLED_TOLERANCE,
                "calibration_error_delta": cal_delta,
                "calibration_tolerance": EARLY_CALIBRATION_POOLED_TOLERANCE,
                "passed": phase_passed,
                "by_season": season_details,
            }
        )
    return {"comparator": comparator, "passed": passed, "phases": results}


def _useful_gain(
    game_summary: list[dict[str, object]],
    *,
    candidate: str,
    comparator: str,
) -> dict[str, object]:
    def pooled(model: str, aggregation: str) -> dict[str, object]:
        found = next(
            row
            for row in game_summary
            if row["model_arm"] == model and row["aggregation"] == aggregation
        )
        return found

    equal_game_delta = float(
        pooled(candidate, "pooled_equal_game")["mean_crps"]
        - pooled(comparator, "pooled_equal_game")["mean_crps"]
    )
    equal_season_delta = float(
        pooled(candidate, "pooled_equal_season")["mean_crps"]
        - pooled(comparator, "pooled_equal_season")["mean_crps"]
    )
    season_deltas = {}
    for season in EVALUATION_SEASONS:
        candidate_row = next(
            row
            for row in game_summary
            if row["model_arm"] == candidate
            and row["aggregation"] == "season"
            and int(row["season"]) == season
        )
        comparator_row = next(
            row
            for row in game_summary
            if row["model_arm"] == comparator
            and row["aggregation"] == "season"
            and int(row["season"]) == season
        )
        season_deltas[str(season)] = float(
            candidate_row["mean_crps"] - comparator_row["mean_crps"]
        )
    wins = sum(delta < 0 for delta in season_deltas.values())
    passed = (
        equal_game_delta <= -USEFUL_GAIN_MARGIN
        and equal_season_delta <= -USEFUL_GAIN_MARGIN
        and wins >= 2
    )
    return {
        "candidate": candidate,
        "comparator": comparator,
        "pooled_equal_game_crps_delta": equal_game_delta,
        "pooled_equal_season_crps_delta": equal_season_delta,
        "season_crps_deltas": season_deltas,
        "seasons_with_lower_candidate_crps": wins,
        "minimum_practical_gain_margin": USEFUL_GAIN_MARGIN,
        "passed": passed,
    }


def _make_decision(
    forecast_rows: list[dict[str, object]],
    game_summary: list[dict[str, object]],
    ranking_summary: list[dict[str, object]],
) -> dict[str, object]:
    candidate_vs_context = _useful_gain(
        game_summary, candidate=CANDIDATE_MODEL, comparator=CONTEXT_BASELINE_MODEL
    )
    candidate_vs_calibration = _useful_gain(
        game_summary, candidate=CANDIDATE_MODEL, comparator=HISTORY_CONTROL_MODEL
    )
    calibration_vs_context = _useful_gain(
        game_summary, candidate=HISTORY_CONTROL_MODEL, comparator=CONTEXT_BASELINE_MODEL
    )
    candidate_guardrail = _early_guardrail(
        forecast_rows, candidate=CANDIDATE_MODEL, comparator=CONTEXT_BASELINE_MODEL
    )
    calibration_guardrail = _early_guardrail(
        forecast_rows,
        candidate=HISTORY_CONTROL_MODEL,
        comparator=CONTEXT_BASELINE_MODEL,
    )
    if (
        candidate_vs_context["passed"]
        and candidate_vs_calibration["passed"]
        and candidate_guardrail["passed"]
    ):
        conclusion = "Refine for independent confirmation"
    elif (
        calibration_vs_context["passed"]
        and calibration_guardrail["passed"]
        and not candidate_vs_calibration["passed"]
    ):
        conclusion = "Prefer calibration-only for further confirmation"
    else:
        conclusion = "Retain unmodified Context / inconclusive"

    nll_lookup = {
        (str(row["model_arm"]), str(row["aggregation"]), str(row["season"])): row
        for row in ranking_summary
    }
    rank_nll_deltas = {}
    for aggregation, season in (
        ("equal_season_team_season_weighted", ""),
        ("pooled_team_season_weighted", "2023"),
        ("pooled_team_season_weighted", "2024"),
        ("pooled_team_season_weighted", "2025"),
    ):
        context = nll_lookup.get((CONTEXT_BASELINE_MODEL, aggregation, season))
        candidate = nll_lookup.get((CANDIDATE_MODEL, aggregation, season))
        if context and candidate:
            rank_nll_deltas[f"{aggregation}:{season or 'pooled'}"] = float(
                candidate["mean_nll"] - context["mean_nll"]
            )
    candidate_primary = _game_summary_lookup(
        forecast_rows, model=CANDIDATE_MODEL, stage="rolling"
    )
    context_primary = _game_summary_lookup(
        forecast_rows, model=CONTEXT_BASELINE_MODEL, stage="rolling"
    )
    if not candidate_primary or len(candidate_primary) != len(context_primary):
        raise ValueError("primary game population differs for tradeoff summary")
    candidate_pooled_rank = nll_lookup.get(
        (
            CANDIDATE_MODEL,
            "pooled_team_season_weighted_all_evaluation_seasons",
            "",
        )
    )
    context_pooled_rank = nll_lookup.get(
        (
            CONTEXT_BASELINE_MODEL,
            "pooled_team_season_weighted_all_evaluation_seasons",
            "",
        )
    )
    if candidate_pooled_rank is None or context_pooled_rank is None:
        raise ValueError("pooled team-season ranking NLL summary is missing")
    tradeoff_by_season = {}
    for season in EVALUATION_SEASONS:
        candidate_rows = _game_summary_lookup(
            forecast_rows,
            model=CANDIDATE_MODEL,
            stage="rolling",
            season=season,
        )
        context_rows = _game_summary_lookup(
            forecast_rows,
            model=CONTEXT_BASELINE_MODEL,
            stage="rolling",
            season=season,
        )
        tradeoff_by_season[str(season)] = {
            "games": len(candidate_rows),
            "margin_nll_delta": float(
                np.mean([float(row["margin_nll"]) for row in candidate_rows])
                - np.mean([float(row["margin_nll"]) for row in context_rows])
            ),
            "interval_calibration_error_delta": _calibration_error(candidate_rows)
            - _calibration_error(context_rows),
        }
    candidate_tradeoffs = {
        "comparison": f"{CANDIDATE_MODEL}_minus_{CONTEXT_BASELINE_MODEL}",
        "primary_fbs_fbs_rolling_games": len(candidate_primary),
        "margin_nll_delta": float(
            np.mean([float(row["margin_nll"]) for row in candidate_primary])
            - np.mean([float(row["margin_nll"]) for row in context_primary])
        ),
        "interval_calibration_error_delta": _calibration_error(candidate_primary)
        - _calibration_error(context_primary),
        "pooled_team_season_weighted_rank_nll_delta": float(
            candidate_pooled_rank["mean_nll"] - context_pooled_rank["mean_nll"]
        ),
        "rank_nll_delta_by_season": {
            str(season): rank_nll_deltas.get(f"pooled_team_season_weighted:{season}")
            for season in EVALUATION_SEASONS
        },
        "by_season": tradeoff_by_season,
    }
    return {
        "conclusion": conclusion,
        "primary_measure": "future regular-season FBS–FBS game-margin CRPS; lower is better",
        "comparisons": {
            "candidate_vs_unmodified_context": candidate_vs_context,
            "candidate_vs_calibrated_history": candidate_vs_calibration,
            "calibrated_history_vs_unmodified_context": calibration_vs_context,
        },
        "early_guardrails": {
            "candidate_vs_unmodified_context": candidate_guardrail,
            "calibrated_history_vs_unmodified_context": calibration_guardrail,
        },
        "secondary_ranking_nll_delta_candidate_minus_context": rank_nll_deltas,
        "candidate_tradeoffs_vs_unmodified_context": candidate_tradeoffs,
        "independence_or_significance_claim": False,
        "production_change_authorized": False,
    }


def _render_report(
    *,
    decision: dict[str, object],
    comparison_summary: list[dict[str, object]],
    forecast_summary: list[dict[str, object]],
    ranking_summary: list[dict[str, object]],
    fit_records: list[dict[str, object]],
    baseline_parity: dict[str, object],
    weight_summary: list[dict[str, object]],
    strata_summary: list[dict[str, object]],
) -> str:
    lines = [
        "# Issue 193: evidence-dependent Context integration",
        "",
        "## Decision",
        "",
        f"**{decision['conclusion']}**",
        "",
        (
            "The primary measure is prospective regular-season FBS–FBS margin CRPS. "
            "A lower CRPS is better. 2022–2025 have informed previous research, so these "
            "results are exploratory development rather than untouched confirmation."
        ),
        "",
        "## Future-game CRPS",
        "",
        "| Model path | Pooled equal-game CRPS | Equal-season mean CRPS | 2023 | 2024 | 2025 | Games |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODEL_NAMES:
        equal_game = next(
            row
            for row in comparison_summary
            if row["model_arm"] == model and row["aggregation"] == "pooled_equal_game"
        )
        equal_season = next(
            row
            for row in comparison_summary
            if row["model_arm"] == model and row["aggregation"] == "pooled_equal_season"
        )
        season_values = [
            next(
                row
                for row in comparison_summary
                if row["model_arm"] == model
                and row["aggregation"] == "season"
                and int(row["season"]) == season
            )["mean_crps"]
            for season in EVALUATION_SEASONS
        ]
        lines.append(
            f"| `{model}` | {float(equal_game['mean_crps']):.6f} | "
            f"{float(equal_season['mean_crps']):.6f} | "
            + " | ".join(f"{float(value):.6f}" for value in season_values)
            + f" | {int(equal_game['games'])} |"
        )
    lines.extend(
        [
            "",
            (
                "Path 4 is the only model that uses the Context prior signal. Path 3 measures "
                "generic evidence-dependent History calibration. The full per-game score file "
                "also reports margin NLL, interval coverage/width, median-based margin MAE, and win Brier."
            ),
            "",
            "## Phase-level prospective diagnostics",
            "",
            "| Model path | Stage | Population | Season | Phase | Games | Margin CRPS | Margin NLL | Coverage 50/80/95 |",
            "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in forecast_summary:
        coverage = "/".join(
            f"{float(row[f'mean_coverage_{level}']):.3f}" for level in (50, 80, 95)
        )
        lines.append(
            f"| `{row['model_arm']}` | {row['forecast_stage']} | "
            f"{row['forecast_population']} | {row['season']} | {row['phase']} | "
            f"{row['games']} | {float(row['mean_crps']):.6f} | "
            f"{float(row['mean_margin_nll']):.6f} | {coverage} |"
        )
    lines.extend(
        [
            "",
            (
                "This table keeps preseason forecasts and FBS–FCS rolling forecasts separate from the "
                "primary rolling FBS–FBS comparison. Phase labels come from the frozen #201 schedule plan."
            ),
            "",
            "## Early-season guardrails",
            "",
            "| Candidate vs Context phase | CRPS delta | Calibration-error delta | Passed |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    candidate_guardrail = decision["early_guardrails"][
        "candidate_vs_unmodified_context"
    ]
    for phase in candidate_guardrail["phases"]:
        lines.append(
            f"| {phase['phase']} | {float(phase['crps_delta_candidate_minus_comparator']):.6f} | "
            f"{float(phase['calibration_error_delta']):.6f} | {phase['passed']} |"
        )
    lines.extend(
        [
            "",
            (
                "Guardrails compare pooled preseason and pre-Thanksgiving CRPS and mean absolute "
                "coverage deviation against fixed tolerances. Per-season checks are in `decision.json`."
            ),
            "",
            "## Secondary ranking results",
            "",
            (
                "Ranking scores use the corrected pre-bowl target PMFs. Checkpoint rows are weighted so "
                "each team-season totals one across its valid checkpoints."
            ),
            "",
            "| Model path | Season | Aggregation | NLL | Rank CRPS | Expected-rank error | 80% target mass | 80% width |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in MODEL_NAMES:
        for season, aggregation in (
            ("", "pooled_team_season_weighted_all_evaluation_seasons"),
            ("", "equal_season_team_season_weighted"),
            (2023, "pooled_team_season_weighted"),
            (2024, "pooled_team_season_weighted"),
            (2025, "pooled_team_season_weighted"),
        ):
            row = next(
                record
                for record in ranking_summary
                if record["model_arm"] == model
                and record["aggregation"] == aggregation
                and str(record["season"]) == str(season)
            )
            lines.append(
                f"| `{model}` | {season or '2023–2025'} | {aggregation} | "
                + " | ".join(
                    f"{float(row[f'mean_{metric}']):.6f}"
                    for metric in RANK_REPORT_METRICS
                )
                + " |"
            )
    lines.extend(
        [
            "",
            (
                "Preseason, each checkpoint, and equal-checkpoint means over checkpoints with future games "
                "are retained in `ranking_summary.csv`. Per-team scores and full PMFs are retained in "
                "`ranking_scores.csv` and `ranking_predictions.jsonl.gz`."
            ),
            "",
            "### Per-checkpoint ranking scores",
            "",
            "| Model path | Season | Cutoff UTC | Phase | Team-seasons | NLL | Rank CRPS | Expected-rank error | 80% target mass | 80% width |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in ranking_summary:
        if (
            row["aggregation"] != "checkpoint"
            or int(row["season"]) not in EVALUATION_SEASONS
        ):
            continue
        metrics = " | ".join(
            f"{float(row[f'mean_{metric}']):.6f}" for metric in RANK_REPORT_METRICS
        )
        lines.append(
            f"| `{row['model_arm']}` | {row['season']} | {row['cutoff_utc']} | "
            f"{row['checkpoint_label']} | {row['team_seasons']} | {metrics} |"
        )
    lines.extend(
        [
            "",
            "## Fit and parity audit",
            "",
            (
                f"Baseline game parity checked {baseline_parity['game']['rows_checked']} rows and "
                f"baseline ranking parity checked {baseline_parity['ranking']['rows_checked']} rows. "
                f"The maximum permitted score difference was {BASELINE_PARITY_TOLERANCE:g}."
            ),
            "",
            "| Evaluation season | Training seasons | History lambda | Context lambda |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    fits_by_season: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for record in fit_records:
        fits_by_season[int(record["evaluation_season"])][str(record["model_arm"])] = (
            record
        )
    for season in EVALUATION_SEASONS:
        history_fit = fits_by_season[season][HISTORY_CONTROL_MODEL]
        candidate_fit = fits_by_season[season][CANDIDATE_MODEL]
        lines.append(
            f"| {season} | {','.join(str(x) for x in history_fit['training_seasons'])} | "
            f"{float(history_fit['fit']['regularization']):g} | "
            f"{float(candidate_fit['fit']['regularization']):g} |"
        )
    lines.extend(
        [
            "",
            (
                "The optimizer traces, training-row hashes, standardized evidence parameters, and freely "
                "fitted Context-weight coefficients are in `fit_records.json`. `candidate_weights.csv` "
                "records the fitted weight against each held-out team's evidence values."
            ),
            "",
            "## Fallback and transfer-coverage strata",
            "",
            "| Model | Season | Prior status | Repaired transfer coverage | Team-checkpoint rows | NLL | Rank CRPS |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in strata_summary:
        lines.append(
            f"| `{row['model_arm']}` | {row['season']} | {row['component_status']} | "
            f"{row['repaired_transfer_coverage']} | {row['team_checkpoint_rows']} | "
            f"{float(row['mean_nll']):.6f} | {float(row['mean_crps']):.6f} |"
        )
    lines.extend(
        [
            "",
            (
                "Team-level gain concentration and fitted weight distributions are in "
                "`gain_concentration.csv` and `weight_summary.csv`; paired game deltas are in "
                "`paired_game_differences.csv`."
            ),
            "",
            "## Interpretation",
            "",
            (
                f"Across {decision['candidate_tradeoffs_vs_unmodified_context']['primary_fbs_fbs_rolling_games']} "
                "held-out rolling FBS–FBS games, path 4 minus unmodified Context has margin-NLL delta "
                f"{float(decision['candidate_tradeoffs_vs_unmodified_context']['margin_nll_delta']):.6f} "
                "and interval-calibration-error delta "
                f"{float(decision['candidate_tradeoffs_vs_unmodified_context']['interval_calibration_error_delta']):.6f}. "
                "The pooled team-season-weighted rank-NLL delta is "
                f"{float(decision['candidate_tradeoffs_vs_unmodified_context']['pooled_team_season_weighted_rank_nll_delta']):.6f}."
            ),
            "",
            "| Season | Margin-NLL delta | Interval-calibration-error delta | Rank-NLL delta |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for season, values in decision["candidate_tradeoffs_vs_unmodified_context"][
        "by_season"
    ].items():
        rank_delta = decision["candidate_tradeoffs_vs_unmodified_context"][
            "rank_nll_delta_by_season"
        ][season]
        lines.append(
            f"| {season} | {float(values['margin_nll_delta']):.6f} | "
            f"{float(values['interval_calibration_error_delta']):.6f} | "
            f"{float(rank_delta):.6f} |"
        )
    lines.extend(
        [
            "",
            (
                "The final comparison follows the preregistered practical-gain threshold and early guardrails. "
                "Season/checkpoint rows are dependent; no p-values or independent-row claims are made. "
                "The result does not change production priors, likelihoods, inference settings, or site defaults. "
                "Stop after this candidate; a separate issue is required for independent confirmation or release."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    repo_root = ROOT.resolve()
    protocol_path = args.protocol.resolve()
    protocol_commit = frozen_protocol_commit(repo_root, protocol_path)
    protocol_hash = issue201.sha256_file(protocol_path)
    issue201_protocol = repo_root / "docs/issue_201_prebowl_forecast_comparison.md"
    issue201_protocol_hash = issue201.sha256_file(issue201_protocol)

    issue200_root = args.issue200_root.resolve()
    issue201_root = args.issue201_root.resolve()
    issue204_root = args.issue204_root.resolve()
    game_root = args.game_root.resolve()
    schedule_windows = args.schedule_windows.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory already contains files: {output}")

    issue200_manifest_path = issue200_root / "manifest.json"
    issue200_manifest = read_json(issue200_manifest_path)
    verify_issue200_outputs(issue200_root, issue200_manifest)
    issue201_manifest_path = issue201_root / "manifest.json"
    issue201_manifest = read_json(issue201_manifest_path)
    verify_issue201_outputs(issue201_root, issue201_manifest)
    if issue201_manifest.get("protocol_sha256") != issue201_protocol_hash:
        raise ValueError(
            "issue 201 results were built under a different frozen protocol"
        )

    issue204_manifest_path = issue204_root / "manifest.json"
    pre_bowl_target_path = issue204_root / "pre_bowl_target_distributions.csv"
    pre_bowl_targets, target_provenance = issue201.load_pre_bowl_targets(
        pre_bowl_target_path, issue204_manifest_path
    )
    expected_target_sha = issue201_manifest["pre_bowl_target_provenance"]["sha256"]
    if target_provenance["sha256"] != expected_target_sha:
        raise ValueError("#204 corrected targets differ from the targets used by #201")

    schedule_plan_path = issue201_root / "schedule_plan.json"
    schedule_plan_manifest_path = issue201_root / "schedule_plan.manifest.json"
    schedule_plan = read_json(schedule_plan_path)
    target_snapshot_path = issue200_root / "target_snapshot_manifest.csv"
    outcomes_by_season, raw_game_sources = issue201.verify_score_blind_plan(
        schedule_plan,
        schedule_plan_path,
        schedule_plan_manifest_path,
        game_root,
        protocol_sha256=issue201_protocol_hash,
        target_snapshot_manifest_path=target_snapshot_path,
        schedule_windows_path=schedule_windows,
    )
    bundle_path = issue200_root / "pipeline_bundle.json"
    fitted_models_path = issue200_root / "fitted_models.json"
    bundle, priors_by_season, issue200_provenance = issue201._load_issue200_inputs(
        bundle_path=bundle_path,
        manifest_path=issue200_manifest_path,
        fitted_models_path=fitted_models_path,
    )
    issue201_evidence = _evidence_replay_index(
        issue201_root / "replay_evidence.jsonl.gz"
    )
    retained_forecast_scores = read_csv(issue201_root / "forecast_scores.csv")
    retained_ranking_scores = read_csv(issue201_root / "ranking_scores.csv")
    target_dates = issue201._read_target_snapshot_dates(target_snapshot_path)
    transfer_coverage = _transfer_coverage_map(
        issue200_root / "preseason_feature_manifest.json"
    )
    schedule_by_season = {int(row["season"]): row for row in schedule_plan["seasons"]}
    likelihood = issue201._load_arm_likelihood(bundle, "existing")
    empty_evidence_by_season = {season: _empty_evidence(season) for season in SEASONS}
    target_row_by_season = {
        season: {
            team_id: target for team_id, target in pre_bowl_targets[season].items()
        }
        for season in SEASONS
    }

    examples_by_season: dict[int, list[RankTrainingExample]] = {
        season: [] for season in SEASONS
    }
    states_by_season: dict[int, dict[str, dict[str, object]]] = {}
    ranking_scores: list[dict[str, object]] = []
    rank_prediction_rows: list[dict[str, object]] = []
    primary_game_ids_by_season_cutoff: dict[tuple[int, str], set[str]] = defaultdict(
        set
    )
    baseline_forecast_jobs: list[dict[str, object]] = []
    scoreable_by_season: dict[
        int, dict[str, tuple[dict[str, object], dict[str, object]]]
    ] = {}
    replay_diagnostics: list[dict[str, object]] = []

    # Rebuild the matched existing-policy cutoff posteriors from the frozen #200 bundle.
    for season in SEASONS:
        season_plan = schedule_by_season[season]
        season_outcomes = outcomes_by_season[season]
        scoreable_games, _exclusions = issue201._scoreable_game_rows(
            season_plan, season_outcomes
        )
        scoreable_by_season[season] = scoreable_games
        for game_id, (metadata, _outcome) in scoreable_games.items():
            if metadata.get("forecast_population") == "fbs_fbs":
                primary_game_ids_by_season_cutoff[
                    (season, str(metadata["assigned_cutoff"]))
                ].add(game_id)

        common_teams, _population_exclusions = issue201._common_target_teams(
            season, priors_by_season[str(season)], pre_bowl_targets
        )
        if not common_teams:
            raise ValueError(f"{season}: no common #200/#204 team support")
        fcs_names = issue201._fcs_names(season_outcomes)
        history_priors = priors_by_season[str(season)]["History_existing"]
        context_priors = priors_by_season[str(season)]["Context_existing"]
        safe_target_start = datetime.combine(target_dates[season], time.min, tzinfo=UTC)
        cutoff_phase = {
            str(row["cutoff_utc"]): str(row["phase"])
            for row in season_plan["cutoff_phases"]
        }
        states: dict[str, dict[str, object]] = {}
        for checkpoint_index, cutoff_text in enumerate(
            season_plan["cutoffs_utc"], start=1
        ):
            cutoff = datetime.fromisoformat(str(cutoff_text))
            included, excluded = issue201.filter_evidence_rows(
                season_outcomes, cutoff, "existing"
            )
            evidence_games = [issue201.game_from_outcome(row) for row in included]
            evidence_by_team = team_evidence_game_ids(included)
            arm_results = {}
            arm_teams = {}
            diagnostics = {}
            for family, arm_name in (
                ("History", "History_existing"),
                ("Context", "Context_existing"),
            ):
                replay = replay_pipeline_bundle(
                    bundle,
                    season=season,
                    policy="existing",
                    prior_family=family,
                    games=evidence_games,
                    fcs_team_names=fcs_names,
                )
                diagnostics[family] = issue201._posterior_diagnostics(replay)
                arm_results[arm_name] = replay
                arm_teams[arm_name] = issue201._build_posterior_teams(
                    replay.pmfs,
                    priors_by_season[str(season)][arm_name],
                    fcs_names,
                )
            evidence = issue201._evidence_record(
                season,
                cutoff,
                "existing",
                included,
                excluded,
                diagnostics,
                len(arm_results["History_existing"].pmfs),
                len(fcs_names),
            )
            evidence["excluded_game_details"] = [
                {
                    "game_id": row["game_id"],
                    "kickoff_utc": row.get("kickoff_utc"),
                    "reason": row["evidence_exclusion_reason"],
                    "game_row_sha256": row["game_row_sha256"],
                }
                for row in excluded
            ]
            evidence["included_game_row_hashes"] = [
                {
                    "game_id": row["game_id"],
                    "kickoff_utc": row["kickoff_utc"],
                    "game_row_sha256": row["game_row_sha256"],
                }
                for row in sorted(
                    included,
                    key=lambda item: issue201._game_id_key(str(item["game_id"])),
                )
            ]
            expected_evidence = issue201_evidence.get(
                (season, str(cutoff_text), "existing")
            )
            if expected_evidence is None:
                raise ValueError(
                    f"#201 lacks replay evidence for {season}/{cutoff_text}"
                )
            verify_replay_evidence(
                evidence, expected_evidence, season=season, cutoff=str(cutoff_text)
            )
            post_target_evidence = [
                row
                for row in included
                if datetime.fromisoformat(str(row["kickoff_utc"])) >= safe_target_start
            ]
            cutoff_before_target = cutoff < safe_target_start
            safe_for_ranking = cutoff_before_target and not post_target_evidence
            if checkpoint_index == 1:
                preseason_evidence = issue201_evidence.get(
                    (season, "preseason_no_evidence", "existing")
                )
                if preseason_evidence is None:
                    raise ValueError(f"#201 lacks preseason evidence for {season}")
                verify_replay_evidence(
                    _empty_evidence(season),
                    preseason_evidence,
                    season=season,
                    cutoff="preseason_no_evidence",
                )
                if evidence["included_game_count"] != 0:
                    raise ValueError(
                        f"{season}: first weekly cutoff is not preseason-safe"
                    )

            phase = cutoff_phase[str(cutoff_text)]
            checkpoint_has_games = bool(
                primary_game_ids_by_season_cutoff.get((season, str(cutoff_text)))
            )
            state: dict[str, object] = {
                "season": season,
                "cutoff": str(cutoff_text),
                "checkpoint_index": checkpoint_index,
                "checkpoint_phase": phase,
                "checkpoint_has_primary_games": checkpoint_has_games,
                "safe_for_ranking": safe_for_ranking,
                "target_safety": {
                    "target_observation_date": target_dates[season].isoformat(),
                    "cutoff_before_target_date": cutoff_before_target,
                    "post_target_evidence_count": len(post_target_evidence),
                    "post_target_evidence_ids": [
                        str(row["game_id"]) for row in post_target_evidence
                    ],
                    "safe_for_ranking": safe_for_ranking,
                },
                "included": included,
                "excluded": excluded,
                "evidence": evidence,
                "evidence_by_team": evidence_by_team,
                "results": arm_results,
                "teams": arm_teams,
                "inputs": {},
                "common_teams": common_teams,
            }
            inputs_by_team: dict[str, EvidenceInput] = {}
            for team_id, prior_row in history_priors.items():
                if (
                    team_id not in context_priors
                    or team_id not in arm_results["History_existing"].pmfs
                ):
                    raise ValueError(
                        f"{season}/{team_id}: matched prior/posterior is missing"
                    )
                inputs = EvidenceInput(
                    season=season,
                    team_id=team_id,
                    checkpoint=str(cutoff_text),
                    history_posterior=np.asarray(
                        arm_results["History_existing"].pmfs[team_id], dtype=float
                    ),
                    context_prior=np.asarray(
                        context_priors[team_id]["pmf"], dtype=float
                    ),
                    history_prior=np.asarray(prior_row["pmf"], dtype=float),
                    team_game_ids=evidence_by_team.get(team_id, ()),
                )
                inputs_by_team[team_id] = inputs
                if state["safe_for_ranking"] and team_id in pre_bowl_targets[season]:
                    target = np.asarray(
                        pre_bowl_targets[season][team_id]["pmf"], dtype=float
                    )
                    if team_id in common_teams:
                        examples_by_season[season].append(
                            RankTrainingExample(
                                inputs=inputs,
                                target=target,
                                team_name=str(
                                    pre_bowl_targets[season][team_id]["team_name"]
                                ),
                            )
                        )
            state["inputs"] = inputs_by_team
            states[str(cutoff_text)] = state
            replay_diagnostics.append(
                {
                    "season": season,
                    "cutoff_utc": str(cutoff_text),
                    "included_game_count": evidence["included_game_count"],
                    "included_game_ids": evidence["included_game_ids"],
                    "included_game_ids_sha256": evidence["included_game_ids_sha256"],
                    "included_game_rows_sha256": evidence["included_game_rows_sha256"],
                    "included_game_row_hashes": evidence["included_game_row_hashes"],
                    "excluded_game_ids": evidence["excluded_game_ids"],
                    "excluded_game_details": evidence["excluded_game_details"],
                    "target_safety": state["target_safety"],
                    "history_converged": diagnostics["History"]["converged"],
                    "context_converged": diagnostics["Context"]["converged"],
                    "history_iterations": diagnostics["History"]["iterations"],
                    "context_iterations": diagnostics["Context"]["iterations"],
                }
            )

            if state["safe_for_ranking"]:
                target_rows = target_row_by_season[season]
                for team_id in common_teams:
                    inputs = inputs_by_team[team_id]
                    target_row = target_rows[team_id]
                    target = np.asarray(target_row["pmf"], dtype=float)
                    strata = _rank_strata(
                        season,
                        team_id,
                        history_priors,
                        context_priors,
                        transfer_coverage,
                    )
                    for model, family in (
                        ("History_existing", "History"),
                        ("Context_existing", "Context"),
                    ):
                        prediction = np.asarray(
                            arm_results[model].pmfs[team_id], dtype=float
                        )
                        row = _make_rank_score_row(
                            season=season,
                            cutoff=str(cutoff_text),
                            checkpoint_index=checkpoint_index,
                            checkpoint_phase=phase,
                            checkpoint_has_games=checkpoint_has_games,
                            team_id=team_id,
                            team_name=str(target_row["team_name"]),
                            model_arm=model,
                            prior_family=family,
                            prediction=prediction,
                            target=target,
                            inputs=inputs,
                            target_row=target_row,
                            target_observation_date=target_dates[season].isoformat(),
                            evidence=evidence,
                            strata=strata,
                        )
                        ranking_scores.append(row)
                        rank_prediction_rows.append(
                            {
                                **{
                                    key: row[key]
                                    for key in (
                                        "season",
                                        "cutoff_utc",
                                        "checkpoint_index",
                                        "checkpoint_phase",
                                        "checkpoint_label",
                                        "checkpoint_has_primary_games",
                                        "team_id",
                                        "team_name",
                                        "model_arm",
                                        "policy",
                                        "prior_family",
                                        "target_policy",
                                        "evidence_game_count",
                                        "evidence_ids_sha256",
                                        "team_evidence_game_count",
                                        "team_evidence_game_ids",
                                        "team_history_kl_from_prior",
                                        "team_evidence_ids_sha256",
                                        "component_status",
                                        "repaired_transfer_feature_available",
                                        "repaired_transfer_coverage",
                                    )
                                },
                                "target_pmf": target.tolist(),
                                "prediction_pmf": prediction.tolist(),
                            }
                        )
        states_by_season[season] = states
        print(f"Replayed cutoff posteriors for {season}", flush=True)

        if season not in EVALUATION_SEASONS:
            continue
        first_cutoff = str(season_plan["cutoffs_utc"][0])
        first_state = states[first_cutoff]
        for game_id, (metadata, outcome) in scoreable_games.items():
            for model in BASELINE_MODEL_NAMES:
                family = "History" if model == "History_existing" else "Context"
                baseline_forecast_jobs.append(
                    _forecast_job(
                        metadata=metadata,
                        outcome=outcome,
                        model_arm=model,
                        family=family,
                        teams=first_state["teams"][model],
                        likelihood=likelihood,
                        evidence=empty_evidence_by_season[season],
                        stage="preseason",
                        cutoff="preseason_no_evidence",
                    )
                )
        for cutoff_text, state in states.items():
            future_games = [
                (game_id, scoreable_games[game_id])
                for game_id in sorted(
                    primary_game_ids_by_season_cutoff.get((season, cutoff_text), set()),
                    key=issue201._game_id_key,
                )
            ]
            # Add FBS–FCS game forecasts to the same prospective population.
            future_games = [
                (game_id, pair)
                for game_id, pair in scoreable_games.items()
                if str(pair[0]["assigned_cutoff"]) == cutoff_text
            ]
            for _game_id, (metadata, outcome) in future_games:
                for model in BASELINE_MODEL_NAMES:
                    family = "History" if model == "History_existing" else "Context"
                    baseline_forecast_jobs.append(
                        _forecast_job(
                            metadata=metadata,
                            outcome=outcome,
                            model_arm=model,
                            family=family,
                            teams=state["teams"][model],
                            likelihood=likelihood,
                            evidence=state["evidence"],
                            stage="rolling",
                            cutoff=cutoff_text,
                        )
                    )

    # Rank target parity is checked before any candidate fit or candidate score is produced.
    baseline_ranking = [
        row
        for row in ranking_scores
        if row["model_arm"] in BASELINE_MODEL_NAMES
        and int(row["season"]) in EVALUATION_SEASONS
    ]
    rank_parity = _check_score_parity(
        retained_ranking_scores,
        baseline_ranking,
        model_names=BASELINE_MODEL_NAMES,
        metrics=RANK_BASELINE_METRICS,
        label="#201 baseline ranking",
    )
    baseline_forecast_rows = issue201._parallel_prediction_rows(baseline_forecast_jobs)
    forecast_parity = _check_score_parity(
        retained_forecast_scores,
        baseline_forecast_rows,
        model_names=BASELINE_MODEL_NAMES,
        metrics=FORECAST_BASELINE_METRICS,
        label="#201 baseline game forecast",
    )
    baseline_parity = {"game": forecast_parity, "ranking": rank_parity}
    print("Baseline score parity passed", flush=True)

    # Fit one outer model per evaluation season, using strictly earlier targets only.
    grouped_training = grouped_examples(
        [row for season_rows in examples_by_season.values() for row in season_rows]
    )
    fit_records = []
    fits_by_season: dict[int, dict[str, object]] = {}
    for evaluation_season in EVALUATION_SEASONS:
        training_seasons = tuple(
            season for season in SEASONS if season < evaluation_season
        )
        if not training_seasons:
            raise ValueError(
                f"{evaluation_season}: no earlier correction-training season"
            )
        if any(season >= evaluation_season for season in training_seasons):
            raise AssertionError("outer training includes evaluation-season labels")
        fit_pair = {}
        for model, include_context in (
            (HISTORY_CONTROL_MODEL, False),
            (CANDIDATE_MODEL, True),
        ):
            training_rows = [
                row for season in training_seasons for row in grouped_training[season]
            ]
            fit, selection = fit_outer_fold(
                grouped_training,
                training_seasons,
                include_context_signal=include_context,
            )
            if set(fit.training_seasons) != set(training_seasons):
                raise ValueError(
                    f"{evaluation_season}/{model}: fit season provenance differs"
                )
            training_hash = input_rows_hash(training_rows)
            record = {
                "evaluation_season": evaluation_season,
                "model_arm": model,
                "training_seasons": list(training_seasons),
                "training_team_seasons": len(
                    {(row.inputs.season, row.inputs.team_id) for row in training_rows}
                ),
                "training_checkpoint_rows": len(training_rows),
                "training_rows_sha256": training_hash,
                "selection": selection,
                "fit": fit.as_dict(),
            }
            fit_records.append(record)
            fit_pair[model] = fit
        fits_by_season[evaluation_season] = fit_pair
        print(f"Fitted correction paths for {evaluation_season}", flush=True)

    # Apply each held-out fit once to cutoff-safe History posteriors.
    candidate_weight_rows: list[dict[str, object]] = []
    candidate_rank_scores: list[dict[str, object]] = []
    candidate_rank_predictions: list[dict[str, object]] = []
    corrected_teams_by_cutoff: dict[int, dict[str, dict[str, dict[str, Team]]]] = {}
    corrected_pmfs_by_cutoff: dict[
        int, dict[str, dict[str, dict[str, np.ndarray]]]
    ] = {}
    for season in EVALUATION_SEASONS:
        corrected_teams_by_cutoff[season] = {}
        corrected_pmfs_by_cutoff[season] = {}
        history_priors = priors_by_season[str(season)]["History_existing"]
        context_priors = priors_by_season[str(season)]["Context_existing"]
        for cutoff_text, state in states_by_season[season].items():
            inputs_by_team = state["inputs"]
            history_teams = state["teams"]["History_existing"]
            models_pmfs: dict[str, dict[str, np.ndarray]] = {
                "History_existing": {
                    team_id: np.asarray(pmf, dtype=float)
                    for team_id, pmf in state["results"][
                        "History_existing"
                    ].pmfs.items()
                },
                "Context_existing": {
                    team_id: np.asarray(pmf, dtype=float)
                    for team_id, pmf in state["results"][
                        "Context_existing"
                    ].pmfs.items()
                },
                HISTORY_CONTROL_MODEL: {},
                CANDIDATE_MODEL: {},
            }
            for team_id, inputs in inputs_by_team.items():
                for model in CALIBRATED_MODEL_NAMES:
                    fit = fits_by_season[season][model]
                    prediction = predict_rank_distribution(inputs, fit)
                    models_pmfs[model][team_id] = prediction
                    if model == CANDIDATE_MODEL:
                        diagnostics = inputs.raw_evidence_features
                        candidate_weight_rows.append(
                            {
                                "season": season,
                                "cutoff_utc": cutoff_text,
                                "checkpoint_index": state["checkpoint_index"],
                                "team_id": team_id,
                                "team_name": history_teams[team_id].name,
                                "model_arm": CANDIDATE_MODEL,
                                "evidence_game_count": state["evidence"][
                                    "included_game_count"
                                ],
                                "team_evidence_game_count": inputs.evidence_count,
                                "team_evidence_game_ids": list(inputs.team_game_ids),
                                "log1p_team_games": float(diagnostics[0]),
                                "history_kl_from_prior": float(diagnostics[1]),
                                "standardized_log1p_team_games": fit.standardizer.transform(
                                    inputs
                                )[0],
                                "standardized_history_kl": fit.standardizer.transform(
                                    inputs
                                )[1],
                                "context_weight": evidence_weight(inputs, fit),
                                "context_signal_l1": float(
                                    np.sum(
                                        np.abs(
                                            np.log(
                                                np.asarray(
                                                    context_priors[team_id]["pmf"]
                                                )[
                                                    np.asarray(
                                                        context_priors[team_id]["pmf"]
                                                    )
                                                    > 0
                                                ]
                                            )
                                            - np.log(
                                                np.asarray(
                                                    history_priors[team_id]["pmf"]
                                                )[
                                                    np.asarray(
                                                        history_priors[team_id]["pmf"]
                                                    )
                                                    > 0
                                                ]
                                            )
                                        )
                                    )
                                ),
                            }
                        )
            corrected_pmfs_by_cutoff[season][cutoff_text] = models_pmfs
            corrected_teams_by_cutoff[season][cutoff_text] = {
                HISTORY_CONTROL_MODEL: _corrected_team_map(
                    history_teams, models_pmfs[HISTORY_CONTROL_MODEL]
                ),
                CANDIDATE_MODEL: _corrected_team_map(
                    history_teams, models_pmfs[CANDIDATE_MODEL]
                ),
            }
            if not state["safe_for_ranking"]:
                continue
            for team_id in state["common_teams"]:
                inputs = inputs_by_team[team_id]
                target_row = pre_bowl_targets[season][team_id]
                target = np.asarray(target_row["pmf"], dtype=float)
                strata = _rank_strata(
                    season,
                    team_id,
                    history_priors,
                    context_priors,
                    transfer_coverage,
                )
                for model, family in (
                    (HISTORY_CONTROL_MODEL, "History"),
                    (CANDIDATE_MODEL, "History"),
                ):
                    prediction = models_pmfs[model][team_id]
                    score_row = _make_rank_score_row(
                        season=season,
                        cutoff=cutoff_text,
                        checkpoint_index=int(state["checkpoint_index"]),
                        checkpoint_phase=str(state["checkpoint_phase"]),
                        checkpoint_has_games=bool(
                            state["checkpoint_has_primary_games"]
                        ),
                        team_id=team_id,
                        team_name=str(target_row["team_name"]),
                        model_arm=model,
                        prior_family=family,
                        prediction=prediction,
                        target=target,
                        inputs=inputs,
                        target_row=target_row,
                        target_observation_date=target_dates[season].isoformat(),
                        evidence=state["evidence"],
                        strata=strata,
                    )
                    candidate_rank_scores.append(score_row)
                    candidate_rank_predictions.append(
                        {
                            **{
                                key: score_row[key]
                                for key in (
                                    "season",
                                    "cutoff_utc",
                                    "checkpoint_index",
                                    "checkpoint_phase",
                                    "checkpoint_label",
                                    "checkpoint_has_primary_games",
                                    "team_id",
                                    "team_name",
                                    "model_arm",
                                    "policy",
                                    "prior_family",
                                    "target_policy",
                                    "evidence_game_count",
                                    "evidence_ids_sha256",
                                    "team_evidence_game_count",
                                    "team_evidence_game_ids",
                                    "team_history_kl_from_prior",
                                    "team_evidence_ids_sha256",
                                    "component_status",
                                    "repaired_transfer_feature_available",
                                    "repaired_transfer_coverage",
                                )
                            },
                            "target_pmf": target.tolist(),
                            "prediction_pmf": prediction.tolist(),
                        }
                    )

    ranking_scores.extend(candidate_rank_scores)
    rank_prediction_rows.extend(candidate_rank_predictions)
    _rank_weights(ranking_scores)

    # Score the two fitted paths on exactly the #201 prospective game set.
    candidate_forecast_jobs: list[dict[str, object]] = []
    for season in EVALUATION_SEASONS:
        scoreable_games = scoreable_by_season[season]
        first_cutoff = str(schedule_by_season[season]["cutoffs_utc"][0])
        first_state = states_by_season[season][first_cutoff]
        preseason_teams = corrected_teams_by_cutoff[season][first_cutoff]
        for game_id, (metadata, outcome) in scoreable_games.items():
            for model in CALIBRATED_MODEL_NAMES:
                candidate_forecast_jobs.append(
                    _forecast_job(
                        metadata=metadata,
                        outcome=outcome,
                        model_arm=model,
                        family="History",
                        teams=preseason_teams[model],
                        likelihood=likelihood,
                        evidence=empty_evidence_by_season[season],
                        stage="preseason",
                        cutoff="preseason_no_evidence",
                    )
                )
        for cutoff_text, state in states_by_season[season].items():
            for game_id, (metadata, outcome) in scoreable_games.items():
                if str(metadata["assigned_cutoff"]) != cutoff_text:
                    continue
                for model in CALIBRATED_MODEL_NAMES:
                    candidate_forecast_jobs.append(
                        _forecast_job(
                            metadata=metadata,
                            outcome=outcome,
                            model_arm=model,
                            family="History",
                            teams=corrected_teams_by_cutoff[season][cutoff_text][model],
                            likelihood=likelihood,
                            evidence=state["evidence"],
                            stage="rolling",
                            cutoff=cutoff_text,
                        )
                    )
    candidate_forecast_rows = issue201._parallel_prediction_rows(
        candidate_forecast_jobs
    )
    print("Scored held-out candidate forecasts", flush=True)
    forecast_rows = [*baseline_forecast_rows, *candidate_forecast_rows]
    game_comparisons = _paired_game_differences(forecast_rows)
    game_summary = _game_comparison_summary(forecast_rows)
    forecast_summary = _forecast_summaries(forecast_rows)
    rank_summary = _rank_summaries(ranking_scores)
    team_rank_scores = _team_rank_scores(ranking_scores)
    team_differences = _team_rank_differences(team_rank_scores)
    concentration = _gain_concentration(game_comparisons, team_differences)
    weight_summary = _candidate_weight_summary(candidate_weight_rows)
    strata_summary = _strata_summary(ranking_scores)
    game_counts = _season_game_counts(forecast_rows)
    decision = _make_decision(forecast_rows, game_summary, rank_summary)

    output_records = {
        "forecast_scores.csv": forecast_rows,
        "forecast_summary.csv": forecast_summary,
        "comparison_summary.csv": game_summary,
        "paired_game_differences.csv": game_comparisons,
        "ranking_scores.csv": ranking_scores,
        "ranking_summary.csv": rank_summary,
        "ranking_team_scores.csv": team_rank_scores,
        "ranking_team_differences.csv": team_differences,
        "strata_summary.csv": strata_summary,
        "candidate_weights.csv": candidate_weight_rows,
        "weight_summary.csv": weight_summary,
        "gain_concentration.csv": concentration,
    }
    result_payload = {
        "experiment": "issue-193-evidence-dependent-context-integration",
        "status": "complete",
        "seasons": list(SEASONS),
        "evaluation_seasons": list(EVALUATION_SEASONS),
        "protocol_sha256": protocol_hash,
        "protocol_freeze_commit": protocol_commit,
        "issue200_provenance": issue200_provenance,
        "pre_bowl_target_provenance": target_provenance,
        "baseline_parity": baseline_parity,
        "replay_cutoff_count": len(replay_diagnostics),
        "rank_training_rows_by_season": {
            str(season): len(rows) for season, rows in examples_by_season.items()
        },
        "fit_records": fit_records,
        "game_counts": game_counts,
        "decision": decision,
        "summary": {
            "game_comparison": game_summary,
            "ranking": rank_summary,
            "weight_distribution": weight_summary,
            "fallback_and_transfer_strata": strata_summary,
            "gain_concentration": concentration,
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    for name, rows in output_records.items():
        write_csv(output / name, rows)
    write_deterministic_gzip_jsonl(
        output / "ranking_predictions.jsonl.gz", rank_prediction_rows
    )
    write_deterministic_gzip_jsonl(
        output / "replay_cutoffs.jsonl.gz", replay_diagnostics
    )
    write_json(output / "fit_records.json", fit_records)
    write_json(output / "decision.json", decision)
    write_json(output / "results.json", result_payload)
    report = _render_report(
        decision=decision,
        comparison_summary=game_summary,
        forecast_summary=forecast_summary,
        ranking_summary=rank_summary,
        fit_records=fit_records,
        baseline_parity=baseline_parity,
        weight_summary=weight_summary,
        strata_summary=strata_summary,
    )
    (output / "report.md").write_text(report, encoding="utf-8")

    input_artifacts = [
        file_record(bundle_path, "issue200/pipeline_bundle.json"),
        file_record(issue200_manifest_path, "issue200/manifest.json"),
        file_record(fitted_models_path, "issue200/fitted_models.json"),
        file_record(
            issue200_root / "preseason_feature_manifest.json",
            "issue200/preseason_feature_manifest.json",
        ),
        file_record(target_snapshot_path, "issue200/target_snapshot_manifest.csv"),
        file_record(issue201_manifest_path, "issue201/manifest.json"),
        file_record(schedule_plan_path, "issue201/schedule_plan.json"),
        file_record(
            schedule_plan_manifest_path, "issue201/schedule_plan.manifest.json"
        ),
        file_record(
            issue201_root / "forecast_scores.csv", "issue201/forecast_scores.csv"
        ),
        file_record(
            issue201_root / "ranking_scores.csv", "issue201/ranking_scores.csv"
        ),
        file_record(
            issue201_root / "replay_evidence.jsonl.gz",
            "issue201/replay_evidence.jsonl.gz",
        ),
        file_record(pre_bowl_target_path, "issue204/pre_bowl_target_distributions.csv"),
        file_record(issue204_manifest_path, "issue204/manifest.json"),
        file_record(schedule_windows, "issue198/schedule_windows.csv"),
        file_record(protocol_path, "issue193/protocol.md"),
        file_record(issue201_protocol, "issue201/protocol.md"),
        *[
            {
                "label": f"cfbd/games/{name}",
                "bytes": int(record["bytes"]),
                "sha256": str(record["sha256"]),
                "row_count": int(record["row_count"]),
            }
            for name, record in sorted(raw_game_sources.items())
        ],
    ]
    output_files = sorted(
        path for path in output.iterdir() if path.name != "manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "experiment": "issue-193-evidence-dependent-context-integration",
        "status": "complete",
        "protocol_sha256": protocol_hash,
        "protocol_freeze_commit": protocol_commit,
        "issue201_protocol_sha256": issue201_protocol_hash,
        "seasons": list(SEASONS),
        "evaluation_seasons": list(EVALUATION_SEASONS),
        "prior_and_likelihood_policy": "issue200 existing",
        "correction_target": "corrected issue204 pre-bowl PMFs",
        "baseline_parity": baseline_parity,
        "input_artifacts": input_artifacts,
        "source_schedule_files": raw_game_sources,
        "issue200_provenance": issue200_provenance,
        "pre_bowl_target_provenance": target_provenance,
        "fit_records_sha256": issue201.sha256_file(output / "fit_records.json"),
        "decision": decision,
        "outputs": {path.name: file_record(path, path.name) for path in output_files},
        "manifest_hash_policy": "input and output files are hashed; the manifest does not hash itself",
    }
    write_json(output / "manifest.json", manifest)
    print(f"Issue 193 results written to {output}")
    print(f"Decision: {decision['conclusion']}")
    print(f"Protocol SHA-256: {protocol_hash}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-root", type=Path, required=True)
    parser.add_argument("--issue200-root", type=Path, required=True)
    parser.add_argument("--issue201-root", type=Path, required=True)
    parser.add_argument("--issue204-root", type=Path, required=True)
    parser.add_argument("--schedule-windows", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "docs/issue_193_evidence_integration_protocol.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/research/issue_193_evidence_integration",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
