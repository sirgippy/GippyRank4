"""Historical shared-error diagnostics for issue 194, Part A only."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import numpy as np

from gippyrank.research.context_incremental_information import cross_message_arms
from gippyrank.research.issue_201_prebowl_forecast import (
    SEASONS,
    _fcs_names,
    _load_issue200_inputs,
    _read_target_snapshot_dates,
    _team_id_key,
    filter_evidence_rows,
    game_from_outcome,
    load_pre_bowl_targets,
    score_rank_pmf,
    sha256_file,
    sha256_json,
    verify_score_blind_plan,
)
from gippyrank.research.matched_pipeline_replay import replay_pipeline_bundle

MODEL_ARMS = ("HH", "CH", "HC", "CC")
NATIVE_MODEL_FOR_ARM = {"HH": "History_existing", "CC": "Context_existing"}
METRIC_FIELDS = {
    "rank_nll": "nll",
    "rank_crps": "crps",
    "target_mass_80": "interval_80_target_mass",
    "interval_width_80": "interval_80_width",
    "hhi_concentration": "hhi_concentration",
}
EXPECTED_SAFE_CUTOFFS = {
    2022: (
        "2022-08-27T23:59:59+00:00",
        "2022-09-22T23:59:59+00:00",
        "2022-10-07T23:59:59+00:00",
        "2022-10-27T23:59:59+00:00",
        "2022-11-09T23:59:59+00:00",
        "2022-11-23T23:59:59+00:00",
    ),
    2023: (
        "2023-08-26T23:59:59+00:00",
        "2023-09-17T23:59:59+00:00",
        "2023-10-06T23:59:59+00:00",
        "2023-10-21T23:59:59+00:00",
        "2023-11-04T23:59:59+00:00",
        "2023-11-18T23:59:59+00:00",
    ),
    2024: (
        "2024-08-24T23:59:59+00:00",
        "2024-09-19T23:59:59+00:00",
        "2024-10-06T23:59:59+00:00",
        "2024-10-24T23:59:59+00:00",
        "2024-11-08T23:59:59+00:00",
        "2024-11-23T23:59:59+00:00",
    ),
    2025: (
        "2025-08-23T23:59:59+00:00",
        "2025-09-19T23:59:59+00:00",
        "2025-10-08T23:59:59+00:00",
        "2025-10-24T23:59:59+00:00",
        "2025-11-09T23:59:59+00:00",
        "2025-11-23T23:59:59+00:00",
    ),
}
EXPECTED_GAME_FILES = tuple(
    f"{season}{suffix}.json" for season in SEASONS for suffix in ("", "-fcs")
)
PAIR_FEATURE_NAMES = (
    "pair_expected_rank_level",
    "pair_expected_rank_level_gap",
    "pair_history_context_disagreement",
    "pair_history_context_disagreement_gap",
    "pair_schedule_degree_level",
    "pair_schedule_degree_gap",
)
GAME_PHASE_FOR_CHECKPOINT = {
    "other_checkpoint": "other_regular_season",
    "pre_thanksgiving": "pre_thanksgiving",
    "rivalry_window": "rivalry_window",
    "conference_championship_period": "conference_championship_period",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _file_record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _verify_hash(path: Path, expected: object, label: str) -> dict[str, object]:
    observed = _file_record(path)
    if observed["sha256"] != expected:
        raise ValueError(f"{label} hash mismatch: {path.name}")
    return observed


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    _atomic_write(path, payload)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    else:
        fields = ["empty"]
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    temporary.replace(path)


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def _sort_pair(first: str, second: str) -> tuple[str, str]:
    if first == second:
        raise ValueError(f"self-pair is invalid: {first}")
    return tuple(sorted((str(first), str(second)), key=_team_id_key))  # type: ignore[return-value]


def _pair_id(season: int, pair: tuple[str, str]) -> str:
    return f"{season}:{pair[0]}:{pair[1]}"


def _safe_bool(value: str) -> bool:
    lowered = value.strip().casefold()
    if lowered not in {"true", "false"}:
        raise ValueError(f"invalid boolean value: {value!r}")
    return lowered == "true"


def _validate_issue201_artifacts(
    issue201_root: Path,
    frozen_root: Path,
    protocol_path: Path,
    game_root: Path,
    game_source_files: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    issue201_manifest = _read_json(issue201_root / "manifest.json")
    if issue201_manifest.get("status") != "complete":
        raise ValueError("issue 201 comparison artifacts are not complete")
    output_hashes = issue201_manifest.get("outputs")
    if not isinstance(output_hashes, dict):
        raise TypeError("issue 201 manifest has no output hash map")
    required_outputs = (
        "frozen_checkpoint_safety.csv",
        "frozen_target_scores.csv",
        "forecast_summary.csv",
        "schedule_plan.json",
        "schedule_plan.manifest.json",
    )
    for name in required_outputs:
        expected = output_hashes.get(name)
        if not isinstance(expected, dict):
            raise TypeError(f"issue 201 manifest does not bind {name}")
        _verify_hash(issue201_root / name, expected.get("sha256"), name)

    protocol_sha = sha256_file(protocol_path)
    if protocol_sha != issue201_manifest.get("protocol_sha256"):
        raise ValueError("issue 201 frozen protocol hash differs from its manifest")
    input_artifacts = issue201_manifest.get("input_artifacts")
    if not isinstance(input_artifacts, dict):
        raise TypeError("issue 201 manifest has no input artifact map")
    protocol_record = input_artifacts.get("protocol")
    if (
        not isinstance(protocol_record, dict)
        or protocol_record.get("sha256") != protocol_sha
    ):
        raise ValueError("issue 201 protocol input hash is not retained")

    schedule_record = input_artifacts.get("schedule_plan")
    if not isinstance(schedule_record, dict):
        raise TypeError("issue 201 manifest has no frozen schedule plan record")
    if sha256_file(issue201_root / "schedule_plan.json") != schedule_record.get(
        "sha256"
    ):
        raise ValueError("issue 201 schedule plan input hash differs")
    if sha256_file(
        issue201_root / "schedule_plan.manifest.json"
    ) != schedule_record.get("manifest_sha256"):
        raise ValueError("issue 201 schedule plan manifest hash differs")

    expected_games = input_artifacts.get("raw_game_files")
    if not isinstance(expected_games, dict) or set(expected_games) != set(
        EXPECTED_GAME_FILES
    ):
        raise ValueError("issue 201 raw game manifest does not cover 2022–2025")
    if set(game_source_files) != set(EXPECTED_GAME_FILES):
        raise ValueError("raw source game files do not cover 2022–2025")
    for name in EXPECTED_GAME_FILES:
        expected = expected_games[name]
        observed = game_source_files[name]
        if not isinstance(expected, dict) or any(
            expected.get(field) != observed.get(field)
            for field in ("bytes", "sha256", "row_count")
        ):
            raise ValueError(f"raw game source differs from #201: {name}")

    frozen_record = input_artifacts.get("frozen_192_outputs")
    if not isinstance(frozen_record, dict):
        raise TypeError("issue 201 manifest lacks the frozen #192 input record")
    frozen_manifest = _read_json(frozen_root / "manifest.json")
    frozen_output_hashes = frozen_manifest.get("output_sha256")
    if not isinstance(frozen_output_hashes, dict):
        raise TypeError("frozen #192 manifest has no output hash map")
    for name in (
        "inference_panel.jsonl.gz",
        "scoring_targets.jsonl.gz",
        "evidence.json",
        "team_scores.csv",
    ):
        path = frozen_root / name
        expected = frozen_output_hashes.get(name)
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise ValueError(f"frozen #192 output hash mismatch: {name}")
    if sha256_file(frozen_root / "manifest.json") != frozen_record.get(
        "manifest_sha256"
    ):
        raise ValueError("frozen #192 manifest hash differs from #201")
    if sha256_file(frozen_root / "evidence.json") != frozen_record.get(
        "evidence_sha256"
    ):
        raise ValueError("frozen #192 evidence hash differs from #201")
    if sha256_file(frozen_root / "inference_panel.jsonl.gz") != frozen_record.get(
        "panel_sha256"
    ):
        raise ValueError("frozen #192 panel hash differs from #201")
    if sha256_file(frozen_root / "scoring_targets.jsonl.gz") != frozen_record.get(
        "final_target_sha256"
    ):
        raise ValueError("frozen #192 target hash differs from #201")
    if sha256_file(frozen_root / "team_scores.csv") != frozen_record.get(
        "score_sha256"
    ):
        raise ValueError("frozen #192 score hash differs from #201")

    frozen_source_hashes = frozen_manifest.get("source_sha256", {})
    for name, record in game_source_files.items():
        if frozen_source_hashes.get(f"raw_games/{name}") != record.get("sha256"):
            raise ValueError(f"frozen #192 raw source hash differs: {name}")
    return issue201_manifest, input_artifacts, frozen_manifest


def _load_checkpoint_cohorts(
    issue201_root: Path,
) -> tuple[list[dict[str, object]], dict[tuple[int, str], set[str]]]:
    safety_rows = _read_csv(issue201_root / "frozen_checkpoint_safety.csv")
    safety_by_key: dict[tuple[int, str], dict[str, str]] = {}
    for row in safety_rows:
        season = int(row["season"])
        cutoff = datetime.fromisoformat(row["cutoff_utc"]).astimezone(UTC).isoformat()
        key = season, cutoff
        if key in safety_by_key:
            raise ValueError(f"duplicate issue 201 checkpoint safety row: {key}")
        safety_by_key[key] = row

    selected_keys = {
        (season, cutoff)
        for season, cutoffs in EXPECTED_SAFE_CUTOFFS.items()
        for cutoff in cutoffs
    }
    observed_safe = {
        (season, cutoff)
        for (season, cutoff), row in safety_by_key.items()
        if _safe_bool(row["target_safe"])
    }
    if observed_safe != selected_keys:
        missing = sorted(selected_keys - observed_safe)
        extra = sorted(observed_safe - selected_keys)
        raise ValueError(
            f"issue 201 safe checkpoint cohort changed; missing={missing}, extra={extra}"
        )
    if len(safety_by_key) != 28:
        raise ValueError("issue 201 original checkpoint count changed")

    score_rows = _read_csv(issue201_root / "frozen_target_scores.csv")
    phase_by_key: dict[tuple[int, str], set[str]] = defaultdict(set)
    cohort_by_arm: dict[tuple[int, str, str], set[str]] = defaultdict(set)
    for row in score_rows:
        if row.get("target_policy") != "pre_bowl_204":
            continue
        season = int(row["season"])
        cutoff = datetime.fromisoformat(row["cutoff_utc"]).astimezone(UTC).isoformat()
        key = season, cutoff
        if key not in selected_keys:
            continue
        phase_by_key[key].add(row["checkpoint_phase"])
        if row["model"] in {"history_posterior", "context_posterior"}:
            cohort_by_arm[(season, cutoff, row["model"])].add(row["team_id"])

    cohorts: dict[tuple[int, str], set[str]] = {}
    coverage_rows: list[dict[str, object]] = []
    for key, safety in sorted(safety_by_key.items()):
        season, cutoff = key
        safe = _safe_bool(safety["target_safe"])
        phases = phase_by_key.get(key, set())
        if safe and len(phases) != 1:
            raise ValueError(f"{key}: #201 does not define one checkpoint phase")
        history_ids = cohort_by_arm.get((season, cutoff, "history_posterior"), set())
        context_ids = cohort_by_arm.get((season, cutoff, "context_posterior"), set())
        if safe:
            if not history_ids or history_ids != context_ids:
                raise ValueError(f"{key}: #201 frozen History/Context teams differ")
            cohorts[key] = set(history_ids)
        phase = next(iter(phases)) if phases else ""
        coverage_rows.append(
            {
                "season": season,
                "checkpoint": int(safety["checkpoint"]),
                "cutoff_utc": cutoff,
                "checkpoint_phase": phase,
                "target_safe": safe,
                "included_game_count": int(safety["included_game_count"]),
                "latest_included_kickoff_utc": safety["latest_included_kickoff_utc"],
                "post_target_evidence_count": int(safety["post_target_evidence_count"]),
                "reason": ""
                if safe
                else "post_target_evidence_or_cutoff_not_before_target",
                "issue201_common_team_count": len(history_ids & context_ids),
            }
        )
    return coverage_rows, cohorts


def _evidence_by_checkpoint(frozen_root: Path) -> dict[tuple[int, int], dict[str, Any]]:
    evidence_doc = _read_json(frozen_root / "evidence.json")
    rows = evidence_doc.get("checkpoints")
    if not isinstance(rows, list):
        raise TypeError("frozen #192 evidence document has no checkpoint list")
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = int(row["season"]), int(row["checkpoint"])
        ids = row.get("included_game_ids")
        if not isinstance(ids, list) or len(ids) != int(row["included_game_count"]):
            raise ValueError(f"{key}: evidence IDs/count do not match")
        if sha256_json(ids) != row.get("included_game_ids_sha256"):
            raise ValueError(f"{key}: evidence ID hash does not match")
        if key in result:
            raise ValueError(f"{key}: duplicate frozen evidence checkpoint")
        result[key] = row
    return result


def _validate_checkpoint_evidence(
    *,
    season: int,
    checkpoint: int,
    cutoff_utc: str,
    evidence: Mapping[str, Any],
    outcomes: Sequence[dict[str, object]],
    raw_by_id: Mapping[tuple[int, str], dict[str, object]],
    target_date: date,
) -> tuple[list[dict[str, object]], list[Any]]:
    cutoff = datetime.fromisoformat(cutoff_utc).astimezone(UTC)
    evidence_cutoff = datetime.fromisoformat(str(evidence["cutoff"])).astimezone(UTC)
    if evidence_cutoff != cutoff:
        raise ValueError(f"{season}/{checkpoint}: frozen cutoff differs from #201")
    target_start = datetime.combine(target_date, time.min, tzinfo=UTC)
    if cutoff >= target_start:
        raise ValueError(f"{season}/{checkpoint}: cutoff is not before target date")

    ids = [str(value) for value in evidence["included_game_ids"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{season}/{checkpoint}: duplicate evidence game ID")
    selected_rows = []
    for game_id in ids:
        row = raw_by_id.get((season, game_id))
        if row is None:
            raise ValueError(
                f"{season}/{checkpoint}: evidence game {game_id} is missing"
            )
        selected_rows.append(row)
    reselected, _excluded = filter_evidence_rows(list(outcomes), cutoff, "existing")
    if {str(row["game_id"]) for row in reselected} != set(ids):
        raise ValueError(
            f"{season}/{checkpoint}: retained evidence differs from #201's existing policy"
        )
    for row in selected_rows:
        kickoff_raw = row.get("kickoff_utc")
        if not kickoff_raw:
            raise ValueError(f"{season}/{checkpoint}: evidence kickoff is missing")
        kickoff = datetime.fromisoformat(str(kickoff_raw)).astimezone(UTC)
        if kickoff >= target_start or kickoff > cutoff:
            raise ValueError(
                f"{season}/{checkpoint}: post-target/future evidence found"
            )
    return reselected, [game_from_outcome(row) for row in reselected]


def _rank_score(forecast: np.ndarray, target: np.ndarray) -> dict[str, float]:
    scores = score_rank_pmf(forecast, target)
    prediction = np.asarray(forecast, dtype=np.float64)
    scores["hhi_concentration"] = float(np.dot(prediction, prediction))
    scores["effective_rank_support"] = 1.0 / scores["hhi_concentration"]
    scores["max_rank_probability"] = float(prediction.max())
    return scores


def _crossed_effect_rows(
    score_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_key: dict[tuple[int, str], dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in score_rows:
        by_key[(int(row["season"]), str(row["cutoff_utc"]))][str(row["arm"])].append(
            dict(row)
        )
    results: list[dict[str, object]] = []
    for (season, cutoff), arms in sorted(by_key.items()):
        team_sets = {
            arm: {str(row["team_id"]) for row in rows} for arm, rows in arms.items()
        }
        if any(team_sets.get(arm) != team_sets.get("HH") for arm in MODEL_ARMS):
            raise ValueError(f"{season}/{cutoff}: crossed arms do not share team rows")
        for metric, field in METRIC_FIELDS.items():
            means = {
                arm: float(np.mean([float(row[field]) for row in arms[arm]]))
                for arm in MODEL_ARMS
            }
            direct = means["CH"] - means["HH"]
            transmitted = means["HC"] - means["HH"]
            interaction = means["CC"] - means["CH"] - means["HC"] + means["HH"]
            total = means["CC"] - means["HH"]
            if not math.isclose(
                direct + transmitted + interaction, total, rel_tol=0.0, abs_tol=1e-12
            ):
                raise AssertionError("crossed-message decomposition does not sum")
            results.append(
                {
                    "season": season,
                    "cutoff_utc": cutoff,
                    "checkpoint_phase": str(arms["HH"][0]["checkpoint_phase"]),
                    "metric": metric,
                    "team_count": len(team_sets["HH"]),
                    "hh": means["HH"],
                    "ch": means["CH"],
                    "hc": means["HC"],
                    "cc": means["CC"],
                    "direct_prior": direct,
                    "transmitted": transmitted,
                    "interaction": interaction,
                    "native_context_minus_history": total,
                }
            )
    return results


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float | None:
    if not rows:
        return None
    return float(np.mean([float(row[field]) for row in rows]))


def _mean_by_checkpoint(
    rows: Sequence[Mapping[str, object]], field: str
) -> float | None:
    by_checkpoint: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_checkpoint[str(row["cutoff_utc"])].append(row)
    checkpoint_means = [
        {"value": _mean(checkpoint_rows, field)}
        for checkpoint_rows in by_checkpoint.values()
        if checkpoint_rows
    ]
    return _mean(checkpoint_means, "value")


def _summarize_checkpoint_scores(
    score_rows: Sequence[Mapping[str, object]],
    coverage_rows: Sequence[Mapping[str, object]],
    forecast_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    groups: dict[tuple[int, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in score_rows:
        groups[
            (int(row["season"]), str(row["checkpoint_phase"]), str(row["arm"]))
        ].append(row)
    counts: Counter[tuple[int, str]] = Counter()
    for row in coverage_rows:
        if row["target_safe"]:
            counts[(int(row["season"]), str(row["checkpoint_phase"]))] += 1

    future: dict[tuple[int, str, str], dict[str, str]] = {}
    for row in forecast_rows:
        if (
            row.get("forecast_stage") == "rolling"
            and row.get("forecast_population") == "fbs_fbs"
            and row.get("season")
            and str(row["season"]).isdigit()
            and row.get("model_arm") in {"History_existing", "Context_existing"}
        ):
            future[(int(row["season"]), row["phase"], row["model_arm"])] = row

    summary_rows: list[dict[str, object]] = []
    seasons = sorted({int(row["season"]) for row in coverage_rows})
    phases = (
        "other_checkpoint",
        "pre_thanksgiving",
        "rivalry_window",
        "conference_championship_period",
        "army_navy",
        "all_phases",
    )
    for season in seasons:
        for phase in phases:
            game_phase = GAME_PHASE_FOR_CHECKPOINT.get(phase, phase)
            for arm in MODEL_ARMS:
                if phase == "all_phases":
                    rows = [
                        row
                        for row in score_rows
                        if int(row["season"]) == season and str(row["arm"]) == arm
                    ]
                    checkpoint_count = len({str(row["cutoff_utc"]) for row in rows})
                else:
                    rows = groups.get((season, phase, arm), [])
                    checkpoint_count = counts[(season, phase)]
                future_model = NATIVE_MODEL_FOR_ARM.get(arm)
                game_row = (
                    future.get((season, game_phase, future_model))
                    if future_model is not None
                    else None
                )
                summary_rows.append(
                    {
                        "season": season,
                        "checkpoint_phase": phase,
                        "model_arm": arm,
                        "checkpoint_count": checkpoint_count,
                        "team_checkpoint_rows": len(rows),
                        "mean_nll": _mean_by_checkpoint(rows, "nll"),
                        "mean_rank_crps": _mean_by_checkpoint(rows, "crps"),
                        "mean_target_mass_80": _mean_by_checkpoint(
                            rows, "interval_80_target_mass"
                        ),
                        "mean_interval_width_80": _mean_by_checkpoint(
                            rows, "interval_80_width"
                        ),
                        "mean_hhi_concentration": _mean_by_checkpoint(
                            rows, "hhi_concentration"
                        ),
                        "mean_entropy": _mean_by_checkpoint(rows, "entropy"),
                        "future_game_phase_reference": game_phase,
                        "future_game_crps": (
                            float(game_row["crps"]) if game_row is not None else None
                        ),
                        "future_game_count": (
                            int(game_row["games"]) if game_row is not None else None
                        ),
                        "future_game_metric_model": future_model or "",
                    }
                )
    return summary_rows, _season_score_summary(score_rows)


def _season_score_summary(
    score_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in score_rows:
        groups[(int(row["season"]), str(row["arm"]))].append(row)
    result = []
    for (season, arm), rows in sorted(groups.items()):
        result.append(
            {
                "season": season,
                "model_arm": arm,
                "checkpoint_count": len({str(row["cutoff_utc"]) for row in rows}),
                "team_checkpoint_rows": len(rows),
                "mean_nll": _mean_by_checkpoint(rows, "nll"),
                "mean_rank_crps": _mean_by_checkpoint(rows, "crps"),
                "mean_target_mass_80": _mean_by_checkpoint(
                    rows, "interval_80_target_mass"
                ),
                "mean_interval_width_80": _mean_by_checkpoint(
                    rows, "interval_80_width"
                ),
                "mean_hhi_concentration": _mean_by_checkpoint(
                    rows, "hhi_concentration"
                ),
                "mean_entropy": _mean_by_checkpoint(rows, "entropy"),
            }
        )
    return result


def signed_midpoint_cdf_residual(
    forecast: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
) -> float:
    """Target-weighted midpoint-CDF residual for discrete rank PMFs."""
    prediction = np.asarray(forecast, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if (
        prediction.ndim != 1
        or truth.shape != prediction.shape
        or not len(truth)
        or not np.isfinite(prediction).all()
        or not np.isfinite(truth).all()
        or np.any(prediction < 0)
        or np.any(truth < 0)
        or not np.isclose(float(prediction.sum()), 1.0, rtol=0.0, atol=1e-8)
        or not np.isclose(float(truth.sum()), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise ValueError("forecast and target must be normalized PMFs on equal support")
    before = np.concatenate(([0.0], np.cumsum(prediction[:-1])))
    midpoint_cdf = before + 0.5 * prediction
    return float(np.dot(truth, midpoint_cdf) - 0.5)


def _preseason_pair_universe(
    *,
    season: int,
    history_priors: Mapping[str, Mapping[str, object]],
    context_priors: Mapping[str, Mapping[str, object]],
    season_schedule: Mapping[str, Any],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    history_ids = set(history_priors)
    context_ids = set(context_priors)
    if history_ids != context_ids:
        raise ValueError(f"{season}: existing-policy prior team populations differ")
    team_ids = sorted(history_ids, key=_team_id_key)
    if len(team_ids) < 3:
        raise ValueError(f"{season}: not enough FBS teams to form team pairs")
    population_by_id = {
        team_id: int(history_priors[team_id]["population"]) for team_id in team_ids
    }
    if any(
        population_by_id[team_id] != int(context_priors[team_id]["population"])
        for team_id in team_ids
    ):
        raise ValueError(f"{season}: History and Context prior rank supports differ")
    populations = set(population_by_id.values())
    if len(populations) != 1:
        raise ValueError(f"{season}: FBS teams have inconsistent rank populations")
    population = next(iter(populations))
    ranks = np.arange(1, population + 1, dtype=np.float64)
    expected_by_team: dict[str, dict[str, float]] = {}
    for team_id in team_ids:
        h_pmf = np.asarray(history_priors[team_id]["pmf"], dtype=np.float64)
        c_pmf = np.asarray(context_priors[team_id]["pmf"], dtype=np.float64)
        if h_pmf.shape != (population,) or c_pmf.shape != (population,):
            raise ValueError(f"{season}/{team_id}: invalid preseason prior support")
        h_expected = float(np.dot(ranks, h_pmf))
        c_expected = float(np.dot(ranks, c_pmf))
        expected_by_team[team_id] = {
            "history": h_expected,
            "context": c_expected,
            "level": (h_expected + c_expected) / 2.0,
            "disagreement": abs(h_expected - c_expected),
        }

    scheduled_counts_all: Counter[tuple[str, str]] = Counter()
    neighbors: dict[str, set[str]] = {team_id: set() for team_id in team_ids}
    for game in season_schedule.get("games", []):
        if game.get("forecast_population") != "fbs_fbs":
            continue
        pair = _sort_pair(str(game["home_id"]), str(game["away_id"]))
        scheduled_counts_all[pair] += 1
        if pair[0] in neighbors and pair[1] in neighbors:
            neighbors[pair[0]].add(pair[1])
            neighbors[pair[1]].add(pair[0])
    scheduled_counts = {
        pair: count
        for pair, count in scheduled_counts_all.items()
        if pair[0] in neighbors and pair[1] in neighbors
    }

    pair_rows: list[dict[str, object]] = []
    feature_rows: list[list[float]] = []
    pair_keys: list[tuple[str, str]] = []
    for index, first in enumerate(team_ids):
        for second in team_ids[index + 1 :]:
            first_values = expected_by_team[first]
            second_values = expected_by_team[second]
            pair = first, second
            feature_values = [
                (float(first_values["level"]) + float(second_values["level"]))
                / (2.0 * population),
                abs(float(first_values["level"]) - float(second_values["level"]))
                / population,
                (
                    float(first_values["disagreement"])
                    + float(second_values["disagreement"])
                )
                / (2.0 * population),
                abs(
                    float(first_values["disagreement"])
                    - float(second_values["disagreement"])
                )
                / population,
                (len(neighbors[first]) + len(neighbors[second])) / 2.0,
                float(abs(len(neighbors[first]) - len(neighbors[second]))),
            ]
            pair_keys.append(pair)
            feature_rows.append(feature_values)
            pair_rows.append(
                {
                    "season": season,
                    "pair_id": _pair_id(season, pair),
                    "team_id_1": first,
                    "team_id_2": second,
                    "is_scheduled_opponent_pair": pair in scheduled_counts,
                    "scheduled_game_count": int(scheduled_counts.get(pair, 0)),
                    "team_1_schedule_degree": len(neighbors[first]),
                    "team_2_schedule_degree": len(neighbors[second]),
                    **dict(zip(PAIR_FEATURE_NAMES, feature_values, strict=True)),
                }
            )
    scheduled_pairs_outside = [
        {
            "season": season,
            "team_id_1": first,
            "team_id_2": second,
            "scheduled_game_count": count,
            "reason": "scheduled_team_missing_from_matched_issue200_prior_population",
        }
        for (first, second), count in sorted(
            scheduled_counts_all.items(),
            key=lambda row: (_team_id_key(row[0][0]), _team_id_key(row[0][1])),
        )
        if first not in neighbors or second not in neighbors
    ]
    _match_schedule_pairs(
        pair_rows,
        pair_keys,
        np.asarray(feature_rows, dtype=np.float64),
        season=season,
    )
    return pair_rows, scheduled_pairs_outside


def _match_schedule_pairs(
    pair_rows: list[dict[str, object]],
    pair_keys: list[tuple[str, str]],
    feature_matrix: np.ndarray,
    *,
    season: int,
) -> None:
    if len(pair_rows) != len(pair_keys) or feature_matrix.shape != (
        len(pair_rows),
        len(PAIR_FEATURE_NAMES),
    ):
        raise ValueError("pair feature matrix does not match the pair universe")
    if not pair_rows:
        return
    scale = np.std(feature_matrix, axis=0, ddof=0)
    center = np.mean(feature_matrix, axis=0)
    active = scale > 0
    standardized = np.zeros_like(feature_matrix)
    standardized[:, active] = (feature_matrix[:, active] - center[active]) / scale[
        active
    ]
    row_index = {pair: index for index, pair in enumerate(pair_keys)}
    opponents = [
        pair
        for pair, row in zip(pair_keys, pair_rows, strict=True)
        if bool(row["is_scheduled_opponent_pair"])
    ]
    controls = [
        pair
        for pair, row in zip(pair_keys, pair_rows, strict=True)
        if not bool(row["is_scheduled_opponent_pair"])
    ]
    opponents.sort(key=lambda pair: (_team_id_key(pair[0]), _team_id_key(pair[1])))
    controls.sort(key=lambda pair: (_team_id_key(pair[0]), _team_id_key(pair[1])))
    if not opponents or not controls:
        for pair in opponents:
            pair_rows[row_index[pair]]["match_status"] = "unmatched_no_nonopponent_pair"
        for pair in controls:
            pair_rows[row_index[pair]]["match_status"] = "available_not_selected"
        return

    control_indices = np.asarray([row_index[pair] for pair in controls], dtype=int)
    nearest_distance: dict[tuple[str, str], float] = {}
    for pair in opponents:
        vector = standardized[row_index[pair]]
        distances = np.linalg.norm(standardized[control_indices] - vector, axis=1)
        nearest_distance[pair] = float(distances.min())
    assignment_order = sorted(
        opponents,
        key=lambda pair: (
            -nearest_distance[pair],
            _team_id_key(pair[0]),
            _team_id_key(pair[1]),
        ),
    )
    used_controls: set[tuple[str, str]] = set()
    for match_number, opponent in enumerate(assignment_order, start=1):
        opponent_index = row_index[opponent]
        available = [pair for pair in controls if pair not in used_controls]
        if not available:
            pair_rows[opponent_index]["match_status"] = "unmatched_no_unused_control"
            continue
        available_indices = np.asarray(
            [row_index[pair] for pair in available], dtype=int
        )
        distances = np.linalg.norm(
            standardized[available_indices] - standardized[opponent_index], axis=1
        )
        best_distance = float(distances.min())
        # `available` is canonical-ID sorted, so the first exact tie is stable.
        chosen_offset = int(np.flatnonzero(distances == best_distance)[0])
        control = available[chosen_offset]
        used_controls.add(control)
        match_id = f"match-{season}-{match_number:04d}-{opponent[0]}-{opponent[1]}"
        opponent_row = pair_rows[opponent_index]
        control_row = pair_rows[row_index[control]]
        opponent_row.update(
            {
                "match_status": "matched_opponent",
                "match_id": match_id,
                "matched_control_pair_id": control_row["pair_id"],
                "matched_distance": best_distance,
                "nearest_control_distance_before_assignment": nearest_distance[
                    opponent
                ],
            }
        )
        control_row.update(
            {
                "match_status": "selected_nonopponent_control",
                "match_id": match_id,
                "matched_opponent_pair_id": opponent_row["pair_id"],
                "matched_distance": best_distance,
                "nearest_control_distance_before_assignment": nearest_distance[
                    opponent
                ],
            }
        )
    for pair in controls:
        row = pair_rows[row_index[pair]]
        row.setdefault("match_status", "available_not_selected")


def _preseason_team_scores(
    *,
    season: int,
    history_priors: Mapping[str, Mapping[str, object]],
    context_priors: Mapping[str, Mapping[str, object]],
    targets: Mapping[str, Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    team_rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    common_ids = set(history_priors) & set(context_priors)
    for team_id in sorted(common_ids, key=_team_id_key):
        target_row = targets.get(team_id)
        if target_row is None:
            exclusions.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "reason": "missing_corrected_pre_bowl_target",
                }
            )
            continue
        target = np.asarray(target_row["pmf"], dtype=np.float64)
        supports = {
            len(target),
            int(history_priors[team_id]["population"]),
            int(context_priors[team_id]["population"]),
            len(np.asarray(history_priors[team_id]["pmf"], dtype=np.float64)),
            len(np.asarray(context_priors[team_id]["pmf"], dtype=np.float64)),
        }
        if len(supports) != 1:
            exclusions.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "reason": "preseason_prior_or_target_rank_support_mismatch",
                    "observed_supports": ",".join(map(str, sorted(supports))),
                }
            )
            continue
        target_expected = float(
            np.dot(np.arange(1, len(target) + 1, dtype=np.float64), target)
        )
        for arm, prior_rows in (
            ("History_existing", history_priors),
            ("Context_existing", context_priors),
        ):
            forecast = np.asarray(prior_rows[team_id]["pmf"], dtype=np.float64)
            scores = _rank_score(forecast, target)
            expected_rank_residual = scores["expected_rank"] - target_expected
            team_rows.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "model_arm": arm,
                    "full_pmf_nll": scores["nll"],
                    "signed_midpoint_cdf_residual": signed_midpoint_cdf_residual(
                        forecast, target
                    ),
                    "expected_rank_residual": expected_rank_residual,
                    "absolute_expected_rank_residual": abs(expected_rank_residual),
                }
            )
    return team_rows, exclusions


def _preseason_pair_results(
    *,
    pair_rows: Sequence[Mapping[str, object]],
    team_score_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    team_scores: dict[tuple[int, str, str], Mapping[str, object]] = {
        (int(row["season"]), str(row["model_arm"]), str(row["team_id"])): row
        for row in team_score_rows
    }
    residual_means: dict[tuple[int, str], float] = {}
    teams_by_group: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for row in team_score_rows:
        teams_by_group[(int(row["season"]), str(row["model_arm"]))].append(row)
    for key, rows in teams_by_group.items():
        residual_means[key] = float(
            np.mean([float(row["signed_midpoint_cdf_residual"]) for row in rows])
        )

    pair_by_id = {str(row["pair_id"]): row for row in pair_rows}
    match_edges: dict[str, Mapping[str, object]] = {}
    for row in pair_rows:
        if row.get("match_status") == "matched_opponent":
            match_edges[str(row["match_id"])] = row

    comparison_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    for season in SEASONS:
        for model in ("History_existing", "Context_existing"):
            matched_values: dict[str, list[dict[str, float]]] = defaultdict(list)
            for match_id, edge in sorted(match_edges.items()):
                if int(edge["season"]) != season:
                    continue
                control_id = str(edge.get("matched_control_pair_id", ""))
                control = pair_by_id.get(control_id)
                if control is None:
                    continue
                edge_ids = (str(edge["team_id_1"]), str(edge["team_id_2"]))
                control_ids = (str(control["team_id_1"]), str(control["team_id_2"]))
                edge_scores = [
                    team_scores.get((season, model, team_id)) for team_id in edge_ids
                ]
                control_scores = [
                    team_scores.get((season, model, team_id)) for team_id in control_ids
                ]
                if any(score is None for score in (*edge_scores, *control_scores)):
                    comparison_rows.append(
                        {
                            "season": season,
                            "match_id": match_id,
                            "model_arm": model,
                            "opponent_pair_id": edge["pair_id"],
                            "control_pair_id": control_id,
                            "match_distance": edge["matched_distance"],
                            "exclusion_reason": "missing_common_target_score_for_pair_endpoint",
                        }
                    )
                    continue
                edge_a, edge_b = edge_scores
                control_a, control_b = control_scores
                assert edge_a is not None and edge_b is not None
                assert control_a is not None and control_b is not None
                center = residual_means[(season, model)]

                def pair_values(
                    left: Mapping[str, object],
                    right: Mapping[str, object],
                    center_value: float,
                ) -> dict[str, float]:
                    left_residual = float(left["signed_midpoint_cdf_residual"])
                    right_residual = float(right["signed_midpoint_cdf_residual"])
                    return {
                        "mean_endpoint_nll": (
                            float(left["full_pmf_nll"]) + float(right["full_pmf_nll"])
                        )
                        / 2.0,
                        "mean_endpoint_signed_residual": (
                            left_residual + right_residual
                        )
                        / 2.0,
                        "centered_residual_product": (left_residual - center_value)
                        * (right_residual - center_value),
                        "mean_endpoint_expected_rank_residual": (
                            float(left["expected_rank_residual"])
                            + float(right["expected_rank_residual"])
                        )
                        / 2.0,
                        "mean_endpoint_absolute_expected_rank_residual": (
                            float(left["absolute_expected_rank_residual"])
                            + float(right["absolute_expected_rank_residual"])
                        )
                        / 2.0,
                    }

                opponent_values = pair_values(edge_a, edge_b, center)
                control_values = pair_values(control_a, control_b, center)
                result: dict[str, object] = {
                    "season": season,
                    "match_id": match_id,
                    "model_arm": model,
                    "opponent_pair_id": edge["pair_id"],
                    "control_pair_id": control_id,
                    "match_distance": edge["matched_distance"],
                    "exclusion_reason": "",
                }
                for metric in opponent_values:
                    result[f"opponent_{metric}"] = opponent_values[metric]
                    result[f"control_{metric}"] = control_values[metric]
                    result[f"opponent_minus_control_{metric}"] = (
                        opponent_values[metric] - control_values[metric]
                    )
                comparison_rows.append(result)
                matched_values["opponent_nll"].append(
                    {"value": opponent_values["mean_endpoint_nll"]}
                )
                matched_values["control_nll"].append(
                    {"value": control_values["mean_endpoint_nll"]}
                )
                matched_values["opponent_residual"].append(
                    {"value": opponent_values["mean_endpoint_signed_residual"]}
                )
                matched_values["control_residual"].append(
                    {"value": control_values["mean_endpoint_signed_residual"]}
                )
                matched_values["opponent_comovement"].append(
                    {"value": opponent_values["centered_residual_product"]}
                )
                matched_values["control_comovement"].append(
                    {"value": control_values["centered_residual_product"]}
                )
                matched_values["comovement_delta"].append(
                    {
                        "value": opponent_values["centered_residual_product"]
                        - control_values["centered_residual_product"]
                    }
                )
                matched_values["opponent_rank_error"].append(
                    {"value": opponent_values["mean_endpoint_expected_rank_residual"]}
                )
                matched_values["control_rank_error"].append(
                    {"value": control_values["mean_endpoint_expected_rank_residual"]}
                )
                matched_values["rank_error_delta"].append(
                    {
                        "value": opponent_values["mean_endpoint_expected_rank_residual"]
                        - control_values["mean_endpoint_expected_rank_residual"]
                    }
                )
            matched_count = len(matched_values["comovement_delta"])
            group_rows.append(
                {
                    "season": season,
                    "model_arm": model,
                    "opponent_pairs_in_universe": sum(
                        int(row["season"]) == season
                        and bool(row["is_scheduled_opponent_pair"])
                        for row in pair_rows
                    ),
                    "matched_pair_count": matched_count,
                    "mean_opponent_full_pmf_nll": _mean(
                        matched_values["opponent_nll"], "value"
                    ),
                    "mean_control_full_pmf_nll": _mean(
                        matched_values["control_nll"], "value"
                    ),
                    "mean_opponent_minus_control_full_pmf_nll": (
                        _mean(matched_values["opponent_nll"], "value")
                        - _mean(matched_values["control_nll"], "value")
                        if matched_count
                        else None
                    ),
                    "mean_opponent_signed_residual": _mean(
                        matched_values["opponent_residual"], "value"
                    ),
                    "mean_control_signed_residual": _mean(
                        matched_values["control_residual"], "value"
                    ),
                    "mean_opponent_minus_control_signed_residual": (
                        _mean(matched_values["opponent_residual"], "value")
                        - _mean(matched_values["control_residual"], "value")
                        if matched_count
                        else None
                    ),
                    "mean_opponent_centered_residual_product": _mean(
                        matched_values["opponent_comovement"], "value"
                    ),
                    "mean_control_centered_residual_product": _mean(
                        matched_values["control_comovement"], "value"
                    ),
                    "opponent_minus_control_comovement": _mean(
                        matched_values["comovement_delta"], "value"
                    ),
                    "mean_opponent_expected_rank_residual": _mean(
                        matched_values["opponent_rank_error"], "value"
                    ),
                    "mean_control_expected_rank_residual": _mean(
                        matched_values["control_rank_error"], "value"
                    ),
                    "opponent_minus_control_expected_rank_residual": _mean(
                        matched_values["rank_error_delta"], "value"
                    ),
                }
            )
    for model in ("History_existing", "Context_existing"):
        season_rows = [
            row
            for row in group_rows
            if row["model_arm"] == model and int(row["matched_pair_count"]) > 0
        ]
        if len(season_rows) == len(SEASONS):
            group_rows.append(
                {
                    "season": "equal_season_mean",
                    "model_arm": model,
                    "opponent_pairs_in_universe": None,
                    "matched_pair_count": min(
                        int(row["matched_pair_count"]) for row in season_rows
                    ),
                    **{
                        field: float(
                            np.mean([float(row[field]) for row in season_rows])
                        )
                        for field in (
                            "mean_opponent_full_pmf_nll",
                            "mean_control_full_pmf_nll",
                            "mean_opponent_minus_control_full_pmf_nll",
                            "mean_opponent_signed_residual",
                            "mean_control_signed_residual",
                            "mean_opponent_minus_control_signed_residual",
                            "mean_opponent_centered_residual_product",
                            "mean_control_centered_residual_product",
                            "opponent_minus_control_comovement",
                            "mean_opponent_expected_rank_residual",
                            "mean_control_expected_rank_residual",
                            "opponent_minus_control_expected_rank_residual",
                        )
                    },
                }
            )
    context_by_season = {
        int(row["season"]): float(row["opponent_minus_control_comovement"])
        for row in group_rows
        if row["model_arm"] == "Context_existing"
        and isinstance(row["season"], int)
        and row["opponent_minus_control_comovement"] is not None
    }
    history_by_season = {
        int(row["season"]): float(row["opponent_minus_control_comovement"])
        for row in group_rows
        if row["model_arm"] == "History_existing"
        and isinstance(row["season"], int)
        and row["opponent_minus_control_comovement"] is not None
    }
    if len(context_by_season) != len(SEASONS) or len(history_by_season) != len(SEASONS):
        status = "unresolved"
    else:
        context_positive = sum(context_by_season[s] > 0 for s in SEASONS)
        context_greater = sum(
            context_by_season[s] > history_by_season[s] for s in SEASONS
        )
        context_nonpositive = sum(context_by_season[s] <= 0 for s in SEASONS)
        context_not_greater = sum(
            context_by_season[s] <= history_by_season[s] for s in SEASONS
        )
        if context_positive >= 3 and context_greater >= 3:
            status = "descriptively consistent"
        elif context_nonpositive >= 3 or context_not_greater >= 3:
            status = "descriptively unsupported"
        else:
            status = "unresolved"
    return comparison_rows, group_rows, status


def _render_report(
    *,
    checkpoint_season_rows: Sequence[Mapping[str, object]],
    checkpoint_effect_rows: Sequence[Mapping[str, object]],
    phase_rows: Sequence[Mapping[str, object]],
    pair_summary_rows: Sequence[Mapping[str, object]],
    pair_status: str,
    checkpoint_coverage: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        "# Issue 194 Part A: historical diagnostics",
        "",
        "Part A was selected and run on the existing-policy issue 200 History/Context pair. Part B remains pending. This is an exploratory four-season diagnostic; it does not alter production priors, inference, published snapshots, or website defaults.",
        "",
        "## Checkpoint coverage",
        "",
        "The analysis uses the 24 safe original checkpoints in #201's frozen-target sensitivity: six per season. The final original checkpoint in each season was excluded for target-safety. These are not a matched December replication.",
        "",
        "| Season | Safe checkpoints | Unsafe final checkpoint | Safe phase coverage |",
        "| --- | ---: | --- | --- |",
    ]
    for season in SEASONS:
        safe_rows = [
            row
            for row in checkpoint_coverage
            if int(row["season"]) == season and bool(row["target_safe"])
        ]
        unsafe_rows = [
            row
            for row in checkpoint_coverage
            if int(row["season"]) == season and not bool(row["target_safe"])
        ]
        phases = Counter(str(row["checkpoint_phase"]) for row in safe_rows)
        unsafe_date = str(unsafe_rows[0]["cutoff_utc"]) if unsafe_rows else "none"
        phase_text = ", ".join(
            f"{name}: {count}" for name, count in sorted(phases.items())
        )
        lines.append(f"| {season} | {len(safe_rows)} | {unsafe_date} | {phase_text} |")
    lines.extend(
        [
            "",
            "Across the 24 safe checkpoints, coverage is 16 `other_checkpoint`, four `pre_thanksgiving`, and four `rivalry_window`. There is no safe `conference_championship_period` checkpoint; the postseason/Army–Navy windows have no matching safe original checkpoint either. The final cutoff was not moved to fill that phase.",
            "",
            "## Crossed-message ranking scores",
            "",
            "The table averages all teams and six safe checkpoints within each season. NLL and rank CRPS are lower-is-better; target mass is compared with its nominal 0.80 level; interval width and HHI describe spread/concentration. The crossed arms are frozen-message hybrids.",
            "",
            "| Season | Arm | Rank NLL | Rank CRPS | 80% target mass | 80% width | HHI |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in checkpoint_season_rows:
        lines.append(
            "| {season} | {arm} | {nll:.6f} | {crps:.6f} | {mass:.6f} | {width:.3f} | {hhi:.6f} |".format(
                season=row["season"],
                arm=row["model_arm"],
                nll=float(row["mean_nll"]),
                crps=float(row["mean_rank_crps"]),
                mass=float(row["mean_target_mass_80"]),
                width=float(row["mean_interval_width_80"]),
                hhi=float(row["mean_hhi_concentration"]),
            )
        )
    lines.extend(
        [
            "",
            "### Crossed-message decomposition",
            "",
            "The direct-prior contrast is `CH − HH`, transmitted is `HC − HH`, and interaction is `CC − CH − HC + HH`; all three sum to `CC − HH`. Values below average checkpoint-level team means equally across the six safe checkpoints. These components are descriptive and are not causal shares.",
            "",
            "| Season | Metric | Direct prior | Transmitted | Interaction | Native Context − History |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    metric_labels = {
        "rank_nll": "rank NLL",
        "rank_crps": "rank CRPS",
        "target_mass_80": "80% target mass",
        "interval_width_80": "80% interval width",
        "hhi_concentration": "HHI concentration",
    }
    for season in SEASONS:
        for metric, label in metric_labels.items():
            rows = [
                row
                for row in checkpoint_effect_rows
                if int(row["season"]) == season and row["metric"] == metric
            ]
            if not rows:
                continue
            lines.append(
                "| {season} | {metric} | {direct:+.6f} | {trans:+.6f} | {interaction:+.6f} | {total:+.6f} |".format(
                    season=season,
                    metric=label,
                    direct=float(np.mean([float(row["direct_prior"]) for row in rows])),
                    trans=float(np.mean([float(row["transmitted"]) for row in rows])),
                    interaction=float(
                        np.mean([float(row["interaction"]) for row in rows])
                    ),
                    total=float(
                        np.mean(
                            [float(row["native_context_minus_history"]) for row in rows]
                        )
                    ),
                )
            )
    lines.extend(
        [
            "",
            "`checkpoint_phase_summary.csv` places the ranking measurements alongside #201's retained existing-policy future-game CRPS where a named calendar phase matches. Future-game CRPS measures a different outcome and is contextual, not paired with rank scores. Full season/phase values and game counts are in that file.",
            "",
            "### Retained future-game CRPS context",
            "",
            "| Season | #201 game phase | Games | History CRPS | Context CRPS | Safe ranking checkpoints in mapped phase |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    game_context: dict[tuple[int, str], dict[str, Mapping[str, object]]] = defaultdict(
        dict
    )
    checkpoint_counts: dict[tuple[int, str], int] = {}
    for row in phase_rows:
        season = int(row["season"])
        game_phase = str(row["future_game_phase_reference"])
        checkpoint_counts[(season, game_phase)] = int(row["checkpoint_count"])
        model = str(row["future_game_metric_model"])
        if model and row["future_game_crps"] is not None:
            game_context[(season, game_phase)][model] = row
    for (season, game_phase), models in sorted(game_context.items()):
        history = models.get("History_existing")
        context = models.get("Context_existing")
        games = (
            int(history["future_game_count"])
            if history is not None and history["future_game_count"] is not None
            else int(context["future_game_count"])
            if context is not None and context["future_game_count"] is not None
            else 0
        )
        history_crps = (
            "" if history is None else f"{float(history['future_game_crps']):.6f}"
        )
        context_crps = (
            "" if context is None else f"{float(context['future_game_crps']):.6f}"
        )
        lines.append(
            f"| {season} | {game_phase} | {games} | {history_crps} | {context_crps} | {checkpoint_counts[(season, game_phase)]} |"
        )
    lines.extend(
        [
            "",
            "## Preseason error co-movement",
            "",
            f"Historical co-movement is classified as **{pair_status}** under the frozen directional rule. The table gives opponent and matched-control means with their differences; positive centered residual-product differences mean greater same-direction co-movement among scheduled opponents.",
            "",
            "| Season | Model | Matched pairs | Pair measure | Opponent | Control | Opponent − control |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in pair_summary_rows:
        season = str(row["season"])
        model = str(row["model_arm"])
        count = (
            "equal-season mean"
            if season == "equal_season_mean"
            else str(row["matched_pair_count"])
        )
        for label, opponent_field, control_field, delta_field in (
            (
                "full-PMF NLL",
                "mean_opponent_full_pmf_nll",
                "mean_control_full_pmf_nll",
                "mean_opponent_minus_control_full_pmf_nll",
            ),
            (
                "signed residual",
                "mean_opponent_signed_residual",
                "mean_control_signed_residual",
                "mean_opponent_minus_control_signed_residual",
            ),
            (
                "centered residual product",
                "mean_opponent_centered_residual_product",
                "mean_control_centered_residual_product",
                "opponent_minus_control_comovement",
            ),
            (
                "expected-rank residual",
                "mean_opponent_expected_rank_residual",
                "mean_control_expected_rank_residual",
                "opponent_minus_control_expected_rank_residual",
            ),
        ):
            opponent_value = row[opponent_field]
            control_value = row[control_field]
            delta_value = row[delta_field]
            if opponent_value is None or control_value is None or delta_value is None:
                continue
            lines.append(
                f"| {season} | {model} | {count} | {label} | {float(opponent_value):+.6f} | {float(control_value):+.6f} | {float(delta_value):+.6f} |"
            )
    lines.extend(
        [
            "",
            "Four seasons are only four season clusters. The pair universe overlaps at teams, so no p-values or independent-pair uncertainty claims are reported. The matched null accounts for pair-level schedule-degree level and difference plus preseason rank-level/disagreement features; it does not enforce exact degree balance and is not randomized. Similar levels, schedule structure, shared ranking-measurement error, and common fitted coefficients remain alternative explanations.",
            "",
            "## Retained context and limits",
            "",
            "#201's existing-policy game forecasts remain the primary operational comparison. Context future-game CRPS was lower than History in every season under the four-season rolling summary, while the previous rank-score difference cannot by itself diagnose shared errors. This historical analysis does not establish that Context causes correlated errors or harms game forecasts.",
            "",
            "The ticket's original December final-target pilot values remain a labeled reference only: History NLL 3.83089, Context NLL 3.88793, Context prior with History messages 3.85330, and History prior with Context messages 3.85739. They do not use the corrected pre-bowl target and do not identify a mechanism.",
            "",
            "**Part B is pending.** Synthetic dependence degradation and the small-graph BP approximation question remain unresolved until Part B is explicitly selected and run. No full-season correlated prior, new fit, production change, or follow-on experiment was introduced.",
            "",
            "Inputs are recorded by logical label, byte count, and SHA-256 in `manifest.json`; no machine-specific input paths are stored.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_issue194_part_a(
    *,
    game_root: Path,
    issue200_root: Path,
    issue201_root: Path,
    issue204_root: Path,
    frozen_root: Path,
    target_snapshot_manifest: Path,
    schedule_windows: Path,
    issue201_protocol: Path,
    part_a_protocol: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Run the frozen historical diagnostics without fitting or mutating inputs."""
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty issue 194 output directory: {output_root}"
        )
    if not part_a_protocol.is_file():
        raise FileNotFoundError(part_a_protocol)
    part_a_protocol_sha = sha256_file(part_a_protocol)

    issue200_bundle_path = issue200_root / "pipeline_bundle.json"
    issue200_manifest_path = issue200_root / "manifest.json"
    issue200_fits_path = issue200_root / "fitted_models.json"
    bundle, priors_by_season, issue200_provenance = _load_issue200_inputs(
        bundle_path=issue200_bundle_path,
        manifest_path=issue200_manifest_path,
        fitted_models_path=issue200_fits_path,
    )
    issue200_manifest = _read_json(issue200_manifest_path)

    issue204_target_path = issue204_root / "pre_bowl_target_distributions.csv"
    issue204_manifest_path = issue204_root / "manifest.json"
    pre_bowl_targets, target_provenance = load_pre_bowl_targets(
        issue204_target_path, issue204_manifest_path
    )
    issue204_manifest_sha = sha256_file(issue204_manifest_path)
    issue204_target_record = _file_record(issue204_target_path)

    target_date_record = _file_record(target_snapshot_manifest)
    schedule_window_record = _file_record(schedule_windows)
    issue201_plan_path = issue201_root / "schedule_plan.json"
    issue201_plan_manifest_path = issue201_root / "schedule_plan.manifest.json"
    issue201_manifest_path = issue201_root / "manifest.json"
    issue201_manifest = _read_json(issue201_manifest_path)
    issue201_inputs = issue201_manifest.get("input_artifacts", {})
    if target_date_record["sha256"] != issue201_inputs.get(
        "issue200_target_snapshot_manifest", {}
    ).get("sha256"):
        raise ValueError("target snapshot dates differ from the retained #201 input")
    if schedule_window_record["sha256"] != issue201_inputs.get(
        "issue198_schedule_windows", {}
    ).get("sha256"):
        raise ValueError("schedule window dates differ from the retained #201 input")
    target_record_201 = issue201_manifest.get("pre_bowl_target_provenance", {})
    if (
        issue204_target_record["sha256"] != target_record_201.get("sha256")
        or issue204_manifest_sha != target_record_201.get("manifest_sha256")
        or issue204_target_record["bytes"] != target_record_201.get("bytes")
    ):
        raise ValueError("corrected target files differ from the retained #201 inputs")
    issue200_bundle_record = issue201_inputs.get("issue200_pipeline_bundle", {})
    issue200_manifest_record = issue201_inputs.get("issue200_manifest", {})
    issue200_fits_record = issue201_inputs.get("issue200_fitted_models", {})
    if (
        issue200_provenance["bundle_sha256"] != issue200_bundle_record.get("sha256")
        or issue200_provenance["manifest_sha256"]
        != issue200_manifest_record.get("sha256")
        or issue200_provenance["fitted_models_sha256"]
        != issue200_fits_record.get("sha256")
    ):
        raise ValueError(
            "issue 200 matched bundle differs from the retained #201 inputs"
        )
    target_manifest_inputs = issue200_manifest.get("input_artifacts", [])
    target_manifest_matches = [
        row
        for row in target_manifest_inputs
        if row.get("logical_name") == "corrected_pre_bowl_targets"
    ]
    if (
        len(target_manifest_matches) != 1
        or target_manifest_matches[0].get("sha256") != issue204_target_record["sha256"]
    ):
        raise ValueError("issue 200 did not use these corrected pre-bowl targets")

    protocol_sha = sha256_file(issue201_protocol)
    issue201_schedule_plan = _read_json(issue201_plan_path)
    outcome_rows_by_season, game_source_files = verify_score_blind_plan(
        issue201_schedule_plan,
        issue201_plan_path,
        issue201_plan_manifest_path,
        game_root,
        protocol_sha256=protocol_sha,
        target_snapshot_manifest_path=target_snapshot_manifest,
        schedule_windows_path=schedule_windows,
        seasons=SEASONS,
    )
    _validate_issue201_artifacts(
        issue201_root,
        frozen_root,
        issue201_protocol,
        game_root,
        game_source_files,
    )
    target_dates = _read_target_snapshot_dates(target_snapshot_manifest)
    if set(target_dates) != set(SEASONS):
        raise ValueError("target snapshot manifest does not cover 2022–2025")

    coverage_rows, cohorts = _load_checkpoint_cohorts(issue201_root)
    evidence_by_key = _evidence_by_checkpoint(frozen_root)
    raw_by_id = {
        (season, str(row["game_id"])): row
        for season, rows in outcome_rows_by_season.items()
        for row in rows
    }
    schedule_by_season = {
        int(row["season"]): row for row in issue201_schedule_plan.get("seasons", [])
    }
    if set(schedule_by_season) != set(SEASONS):
        raise ValueError("issue 201 schedule plan does not cover 2022–2025")

    # Freeze the pair universe and matches from preseason PMFs plus schedule
    # structure only. Corrected target PMFs are not passed into these functions.
    pair_universe_rows: list[dict[str, object]] = []
    scheduled_pair_exclusions: list[dict[str, object]] = []
    for season in SEASONS:
        arm_priors = priors_by_season[str(season)]
        pair_rows, outside = _preseason_pair_universe(
            season=season,
            history_priors=arm_priors["History_existing"],
            context_priors=arm_priors["Context_existing"],
            season_schedule=schedule_by_season[season],
        )
        pair_universe_rows.extend(pair_rows)
        scheduled_pair_exclusions.extend(outside)

    preseason_team_rows: list[dict[str, object]] = []
    preseason_team_exclusions: list[dict[str, object]] = []
    for season in SEASONS:
        team_rows, team_exclusions = _preseason_team_scores(
            season=season,
            history_priors=priors_by_season[str(season)]["History_existing"],
            context_priors=priors_by_season[str(season)]["Context_existing"],
            targets=pre_bowl_targets[season],
        )
        preseason_team_rows.extend(team_rows)
        preseason_team_exclusions.extend(team_exclusions)
    pair_comparison_rows, pair_summary_rows, pair_status = _preseason_pair_results(
        pair_rows=pair_universe_rows,
        team_score_rows=preseason_team_rows,
    )

    safe_coverage = [row for row in coverage_rows if bool(row["target_safe"])]
    checkpoint_score_rows: list[dict[str, object]] = []
    checkpoint_exclusions: list[dict[str, object]] = []
    checkpoint_evidence_rows: list[dict[str, object]] = []
    fcs_names_by_season = {
        season: _fcs_names(outcome_rows_by_season[season]) for season in SEASONS
    }
    for season in SEASONS:
        season_coverage = sorted(
            (row for row in safe_coverage if int(row["season"]) == season),
            key=lambda row: str(row["cutoff_utc"]),
        )
        for checkpoint_number, coverage in enumerate(season_coverage, start=1):
            checkpoint = int(coverage["checkpoint"])
            cutoff = str(coverage["cutoff_utc"])
            evidence = evidence_by_key.get((season, checkpoint))
            if evidence is None:
                raise ValueError(f"{season}/{checkpoint}: frozen evidence is missing")
            included_rows, games = _validate_checkpoint_evidence(
                season=season,
                checkpoint=checkpoint,
                cutoff_utc=cutoff,
                evidence=evidence,
                outcomes=outcome_rows_by_season[season],
                raw_by_id=raw_by_id,
                target_date=target_dates[season],
            )
            history_result = replay_pipeline_bundle(
                bundle,
                season=season,
                policy="existing",
                prior_family="History",
                games=games,
                fcs_team_names=fcs_names_by_season[season],
            )
            context_result = replay_pipeline_bundle(
                bundle,
                season=season,
                policy="existing",
                prior_family="Context",
                games=games,
                fcs_team_names=fcs_names_by_season[season],
            )
            if not history_result.converged or not context_result.converged:
                raise ValueError(
                    f"{season}/{checkpoint}: matched replay did not converge"
                )

            arm_priors = priors_by_season[str(season)]
            history_priors = arm_priors["History_existing"]
            context_priors = arm_priors["Context_existing"]
            cohort_ids = cohorts[(season, cutoff)]
            target_ids = set(pre_bowl_targets[season])
            candidate_ids = sorted(cohort_ids, key=_team_id_key)
            scored_ids = []
            support_exclusion_count = 0
            for team_id in candidate_ids:
                if team_id not in history_priors or team_id not in context_priors:
                    checkpoint_exclusions.append(
                        {
                            "season": season,
                            "checkpoint": checkpoint,
                            "cutoff_utc": cutoff,
                            "team_id": team_id,
                            "reason": "missing_from_issue200_existing_prior_population",
                        }
                    )
                    continue
                if team_id not in target_ids:
                    checkpoint_exclusions.append(
                        {
                            "season": season,
                            "checkpoint": checkpoint,
                            "cutoff_utc": cutoff,
                            "team_id": team_id,
                            "reason": "missing_corrected_pre_bowl_target",
                        }
                    )
                    continue
                history_prior = np.asarray(
                    history_priors[team_id]["pmf"], dtype=np.float64
                )
                context_prior = np.asarray(
                    context_priors[team_id]["pmf"], dtype=np.float64
                )
                history_posterior = history_result.pmfs.get(team_id)
                context_posterior = context_result.pmfs.get(team_id)
                if history_posterior is None or context_posterior is None:
                    checkpoint_exclusions.append(
                        {
                            "season": season,
                            "checkpoint": checkpoint,
                            "cutoff_utc": cutoff,
                            "team_id": team_id,
                            "reason": "native_replay_missing_team_posterior",
                        }
                    )
                    continue
                target_row = pre_bowl_targets[season][team_id]
                target = np.asarray(target_row["pmf"], dtype=np.float64)
                support_values = {
                    len(history_prior),
                    len(context_prior),
                    len(history_posterior),
                    len(context_posterior),
                    len(target),
                    int(history_priors[team_id]["population"]),
                    int(context_priors[team_id]["population"]),
                    int(target_row["population"]),
                }
                if len(support_values) != 1:
                    checkpoint_exclusions.append(
                        {
                            "season": season,
                            "checkpoint": checkpoint,
                            "cutoff_utc": cutoff,
                            "team_id": team_id,
                            "reason": "rank_support_mismatch_across_prior_posterior_target",
                            "observed_supports": ",".join(
                                map(str, sorted(support_values))
                            ),
                        }
                    )
                    continue
                try:
                    crossed, identity_errors = cross_message_arms(
                        context_prior,
                        history_prior,
                        np.asarray(context_posterior, dtype=np.float64),
                        np.asarray(history_posterior, dtype=np.float64),
                    )
                except ValueError as error:
                    if str(error) != (
                        "Context and History priors have different positive support"
                    ):
                        raise
                    support_exclusion_count += 1
                    checkpoint_exclusions.append(
                        {
                            "season": season,
                            "checkpoint": checkpoint,
                            "cutoff_utc": cutoff,
                            "team_id": team_id,
                            "reason": "different_positive_support_guard",
                            "detail": str(error),
                        }
                    )
                    continue
                scored_ids.append(team_id)
                for arm in MODEL_ARMS:
                    pmf = crossed[arm]
                    scores = _rank_score(pmf, target)
                    checkpoint_score_rows.append(
                        {
                            "season": season,
                            "checkpoint": checkpoint,
                            "cutoff_utc": cutoff,
                            "checkpoint_phase": coverage["checkpoint_phase"],
                            "team_id": team_id,
                            "team_name": target_row["team_name"],
                            "model_arm": arm,
                            "arm": arm,
                            "target_policy": "issue204_corrected_pre_bowl",
                            "rank_count": len(target),
                            "evidence_game_count": len(included_rows),
                            "evidence_ids_sha256": evidence["included_game_ids_sha256"],
                            "nll": scores["nll"],
                            "crps": scores["crps"],
                            "expected_rank": scores["expected_rank"],
                            "target_expected_rank": scores["target_expected_rank"],
                            "expected_rank_error": scores["expected_rank_error"],
                            "interval_80_low": scores["interval_80_low"],
                            "interval_80_high": scores["interval_80_high"],
                            "interval_80_width": scores["interval_80_width"],
                            "interval_80_target_mass": scores[
                                "interval_80_target_mass"
                            ],
                            "entropy": scores["entropy"],
                            "rank_variance": scores["rank_variance"],
                            "rank_standard_deviation": scores[
                                "rank_standard_deviation"
                            ],
                            "hhi_concentration": scores["hhi_concentration"],
                            "effective_rank_support": scores["effective_rank_support"],
                            "max_rank_probability": scores["max_rank_probability"],
                            "context_message_identity_error": identity_errors[
                                "context_identity_error"
                            ],
                            "history_message_identity_error": identity_errors[
                                "history_identity_error"
                            ],
                        }
                    )
            if not scored_ids:
                raise ValueError(
                    f"{season}/{checkpoint}: no common teams were scoreable"
                )
            # Every member of the frozen #201 team set is either scored on all
            # four arms or listed with an explicit exclusion reason.
            observed_ids = set(scored_ids)
            excluded_ids = {
                str(row["team_id"])
                for row in checkpoint_exclusions
                if row["season"] == season and row["checkpoint"] == checkpoint
            }
            if observed_ids & excluded_ids or observed_ids | excluded_ids != cohort_ids:
                raise ValueError(
                    f"{season}/{checkpoint}: score/exclusion team audit is incomplete"
                )
            checkpoint_evidence_rows.append(
                {
                    "season": season,
                    "checkpoint": checkpoint,
                    "cutoff_utc": cutoff,
                    "checkpoint_phase": coverage["checkpoint_phase"],
                    "included_game_count": len(included_rows),
                    "included_game_ids_sha256": evidence["included_game_ids_sha256"],
                    "latest_included_kickoff_utc": coverage[
                        "latest_included_kickoff_utc"
                    ],
                    "history_iterations": history_result.iterations,
                    "history_max_message_delta": history_result.max_message_delta,
                    "history_converged": history_result.converged,
                    "context_iterations": context_result.iterations,
                    "context_max_message_delta": context_result.max_message_delta,
                    "context_converged": context_result.converged,
                    "issue201_common_team_count": len(cohort_ids),
                    "scoreable_common_team_count": len(scored_ids),
                    "support_guard_exclusion_count": support_exclusion_count,
                }
            )
            coverage["replayed_evidence_game_count"] = len(included_rows)
            coverage["replayed_evidence_ids_sha256"] = evidence[
                "included_game_ids_sha256"
            ]
            coverage["scoreable_common_team_count"] = len(scored_ids)
            print(
                f"Issue 194 Part A: {season} checkpoint {checkpoint_number}/6 "
                f"({len(included_rows)} evidence games, {len(scored_ids)} teams)",
                flush=True,
            )

    checkpoint_effect_rows = _crossed_effect_rows(checkpoint_score_rows)
    phase_summary_rows, season_summary_rows = _summarize_checkpoint_scores(
        checkpoint_score_rows,
        coverage_rows,
        _read_csv(issue201_root / "forecast_summary.csv"),
    )
    future_game_rows = [
        row
        for row in _read_csv(issue201_root / "forecast_summary.csv")
        if row.get("forecast_stage") == "rolling"
        and row.get("forecast_population") == "fbs_fbs"
        and row.get("model_arm") in {"History_existing", "Context_existing"}
    ]
    pair_exclusion_rows = list(scheduled_pair_exclusions)
    pair_exclusion_rows.extend(
        {
            "season": row["season"],
            "pair_id": row["pair_id"],
            "team_id_1": row["team_id_1"],
            "team_id_2": row["team_id_2"],
            "reason": row["match_status"],
        }
        for row in pair_universe_rows
        if row.get("is_scheduled_opponent_pair")
        and row.get("match_status") != "matched_opponent"
    )
    pair_exclusion_rows.extend(
        row for row in pair_comparison_rows if row.get("exclusion_reason")
    )

    outputs: dict[str, Sequence[Mapping[str, object]]] = {
        "checkpoint_scores.csv": checkpoint_score_rows,
        "checkpoint_effects.csv": checkpoint_effect_rows,
        "checkpoint_phase_summary.csv": phase_summary_rows,
        "checkpoint_season_summary.csv": season_summary_rows,
        "checkpoint_coverage.csv": coverage_rows,
        "checkpoint_evidence.csv": checkpoint_evidence_rows,
        "checkpoint_exclusions.csv": checkpoint_exclusions,
        "future_game_crps_context.csv": future_game_rows,
        "preseason_team_scores.csv": preseason_team_rows,
        "preseason_team_exclusions.csv": preseason_team_exclusions,
        "preseason_pair_universe.csv": pair_universe_rows,
        "preseason_pair_exclusions.csv": pair_exclusion_rows,
        "preseason_pair_comparisons.csv": pair_comparison_rows,
        "preseason_pair_summary.csv": pair_summary_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        _write_csv(output_root / name, rows)

    report = _render_report(
        checkpoint_season_rows=season_summary_rows,
        checkpoint_effect_rows=checkpoint_effect_rows,
        phase_rows=phase_summary_rows,
        pair_summary_rows=pair_summary_rows,
        pair_status=pair_status,
        checkpoint_coverage=coverage_rows,
    )
    _atomic_write(output_root / "report.md", report.encode("utf-8"))
    outputs_with_report = [*outputs, "report.md"]
    output_records = {
        name: _file_record(output_root / name) for name in outputs_with_report
    }
    input_records: dict[str, object] = {
        "part_a_protocol": {
            "label": "issue_194_part_a_frozen_protocol",
            **_file_record(part_a_protocol),
        },
        "issue201_protocol": {
            "label": "issue_201_frozen_protocol",
            **_file_record(issue201_protocol),
        },
        "issue200_pipeline_bundle": {
            "label": "issue_200_matched_pipeline_bundle",
            **_file_record(issue200_bundle_path),
        },
        "issue200_manifest": {
            "label": "issue_200_matched_pipeline_manifest",
            **_file_record(issue200_manifest_path),
        },
        "issue200_fitted_models": {
            "label": "issue_200_fitted_model_origin_manifest",
            **_file_record(issue200_fits_path),
        },
        "issue204_target_manifest": {
            "label": "issue_204_corrected_pre_bowl_target_manifest",
            **_file_record(issue204_manifest_path),
        },
        "issue204_pre_bowl_targets": {
            "label": "issue_204_corrected_pre_bowl_target_distributions",
            **_file_record(issue204_target_path),
        },
        "issue201_manifest": {
            "label": "issue_201_replay_and_scoring_manifest",
            **_file_record(issue201_manifest_path),
        },
        "issue201_schedule_plan": {
            "label": "issue_201_score_blind_schedule_plan",
            **_file_record(issue201_plan_path),
        },
        "issue201_frozen_checkpoint_safety": {
            "label": "issue_201_target_safe_original_checkpoint_cohort",
            **_file_record(issue201_root / "frozen_checkpoint_safety.csv"),
        },
        "issue201_frozen_target_scores": {
            "label": "issue_201_common_team_and_phase_labels",
            **_file_record(issue201_root / "frozen_target_scores.csv"),
        },
        "issue201_forecast_summary": {
            "label": "issue_201_existing_policy_future_game_crps",
            **_file_record(issue201_root / "forecast_summary.csv"),
        },
        "frozen192_evidence": {
            "label": "issue_201_verified_original_checkpoint_evidence_ids",
            **_file_record(frozen_root / "evidence.json"),
        },
        "target_snapshot_dates": {
            "label": "issue_200_target_snapshot_dates",
            **_file_record(target_snapshot_manifest),
        },
        "schedule_windows": {
            "label": "issue_198_schedule_phase_windows",
            **_file_record(schedule_windows),
        },
        "raw_game_files": {
            name: {
                "label": f"cfbd_game_schedule_{name}",
                **dict(record),
            }
            for name, record in sorted(game_source_files.items())
        },
    }
    module_path = Path(__file__)
    script_path = (
        output_root.parent.parent.parent / "scripts" / "run_issue_194_part_a.py"
    )
    code_records = {
        "research_module": {
            "label": "issue_194_part_a_research_module",
            **_file_record(module_path),
        }
    }
    if script_path.is_file():
        code_records["runner_script"] = {
            "label": "issue_194_part_a_runner_script",
            **_file_record(script_path),
        }
    manifest = {
        "schema_version": 1,
        "experiment": "issue-194-part-a-historical-shared-context-diagnostics",
        "scope": "part_a_only",
        "part_b_status": "pending",
        "analysis_status": "complete",
        "exploratory_development": True,
        "protocol_sha256": part_a_protocol_sha,
        "issue201_protocol_sha256": protocol_sha,
        "issue200_provenance": issue200_provenance,
        "issue204_target_provenance": target_provenance,
        "input_artifacts": input_records,
        "code_artifacts": code_records,
        "checkpoint_design": {
            "selection": "issue201 frozen-target safety cohort",
            "safe_checkpoint_count": len(safe_coverage),
            "safe_checkpoint_count_by_season": {
                str(season): sum(
                    int(row["season"]) == season and bool(row["target_safe"])
                    for row in coverage_rows
                )
                for season in SEASONS
            },
            "unsafe_checkpoint_count": sum(
                not bool(row["target_safe"]) for row in coverage_rows
            ),
            "phase_counts": dict(
                Counter(
                    str(row["checkpoint_phase"])
                    for row in coverage_rows
                    if bool(row["target_safe"])
                )
            ),
            "excluded_final_cutoffs": [
                {
                    "season": row["season"],
                    "cutoff_utc": row["cutoff_utc"],
                    "reason": row["reason"],
                    "post_target_evidence_count": row["post_target_evidence_count"],
                }
                for row in coverage_rows
                if not bool(row["target_safe"])
            ],
        },
        "crossed_message_design": {
            "arms": list(MODEL_ARMS),
            "score_decomposition": {
                "direct_prior": "CH - HH",
                "transmitted": "HC - HH",
                "interaction": "CC - CH - HC + HH",
                "native_total": "CC - HH",
            },
            "positive_support_guard": "gippyrank.research.context_incremental_information.cross_message_arms",
            "team_score_row_count": len(checkpoint_score_rows),
            "checkpoint_count": len(checkpoint_evidence_rows),
            "checkpoint_evidence": checkpoint_evidence_rows,
            "team_exclusion_count": len(checkpoint_exclusions),
        },
        "preseason_pair_design": {
            "universe_definition": "all unordered pairs in each matched issue200 existing-prior FBS population",
            "matching_method": "deterministic farthest-first greedy one-to-one nearest non-opponent pair in six standardized preseason/schedule features",
            "features": list(PAIR_FEATURE_NAMES),
            "scheduled_opponent_pairs": sum(
                bool(row["is_scheduled_opponent_pair"]) for row in pair_universe_rows
            ),
            "pair_universe_rows": len(pair_universe_rows),
            "matched_pair_count_by_season": {
                str(season): sum(
                    int(row["season"]) == season
                    and row.get("match_status") == "matched_opponent"
                    for row in pair_universe_rows
                )
                for season in SEASONS
            },
            "pairing_exclusion_count": len(pair_exclusion_rows),
            "historical_co_movement_status": pair_status,
            "season_clusters": len(SEASONS),
            "significance_tests": "none",
        },
        "outputs": output_records,
    }
    _write_json(output_root / "manifest.json", manifest)
    manifest_record = _file_record(output_root / "manifest.json")
    manifest["manifest_sha256"] = manifest_record["sha256"]
    manifest["manifest_bytes"] = manifest_record["bytes"]
    return manifest
