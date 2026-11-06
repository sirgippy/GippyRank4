"""Evaluate conditional calibration of retained issue 201 forecasts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from gippyrank.research import issue_201_prebowl_forecast as issue201
from gippyrank.research import issue_247_contradictory_evidence as issue247

SEASONS = (2022, 2023, 2024, 2025)
FAMILIES = ("History", "Context")
MODEL_ARMS = {family: f"{family}_existing" for family in FAMILIES}
POPULATIONS = ("fbs_fbs", "fbs_fcs")
PROGRESSION_GROUPS = ("early_0_2", "middle_3_6", "late_7_plus")
SURPRISE_GROUPS = (
    "sustained_negative",
    "approximately_neutral",
    "sustained_positive",
)
BOOTSTRAP_REPETITIONS = 1_000
BOOTSTRAP_SEED = 256
PROTOCOL_SHA256 = "125ec0c99dd63547b63a842112458ef27fea65575e989d927cbfc91b2125c6a3"
NOMINAL_COVERAGE = {50: 0.50, 80: 0.80, 95: 0.95}

FORECAST_FIELDS = (
    "forecast_stage",
    "season",
    "model_family",
    "model_arm",
    "policy",
    "forecast_id",
    "game_id",
    "forecast_population",
    "phase",
    "cutoff_utc",
    "kickoff_utc",
    "home_id",
    "home_team",
    "home_subdivision",
    "away_id",
    "away_team",
    "away_subdivision",
    "evidence_game_count",
    "evidence_ids_sha256",
    "actual_home_margin",
    "expected_home_margin",
    "median_home_margin",
    "margin_interval_50_low",
    "margin_interval_50_high",
    "margin_interval_50_width",
    "margin_interval_80_low",
    "margin_interval_80_high",
    "margin_interval_80_width",
    "margin_interval_95_low",
    "margin_interval_95_high",
    "margin_interval_95_width",
    "coverage_50",
    "coverage_80",
    "coverage_95",
    "upper_miss_80",
    "lower_miss_80",
    "crps",
    "margin_nll",
)

TEAM_FIELDS = (
    "season",
    "model_family",
    "model_arm",
    "forecast_id",
    "game_id",
    "forecast_population",
    "phase",
    "cutoff_utc",
    "kickoff_utc",
    "evidence_game_count",
    "evidence_ids_sha256",
    "team_id",
    "team",
    "opponent_id",
    "opponent",
    "is_home",
    "team_game_number",
    "prior_game_count",
    "progression_group",
    "eligible_previous_game_ids",
    "primary_eligible",
    "accumulated_surprise",
    "surprise_group",
    "preseason_expected_rank",
    "preseason_rank_sd",
    "preseason_strength",
    "preseason_rank_uncertainty_fraction",
    "preseason_rank_population",
    "preseason_rank_top10_probability",
    "preseason_rank_pmf_sha256",
    "actual_margin",
    "expected_margin",
    "median_margin",
    "margin_interval_50_low",
    "margin_interval_50_high",
    "margin_interval_50_width",
    "margin_interval_80_low",
    "margin_interval_80_high",
    "margin_interval_80_width",
    "margin_interval_95_low",
    "margin_interval_95_high",
    "margin_interval_95_width",
    "coverage_50",
    "coverage_80",
    "coverage_95",
    "upper_miss_80",
    "lower_miss_80",
    "crps",
    "margin_nll",
)

SUMMARY_FIELDS = (
    "analysis",
    "forecast_stage",
    "population",
    "model_family",
    "scope",
    "season",
    "condition",
    "condition_value",
    "observation_unit",
    "n_forecasts",
    "unique_games",
    "unique_teams",
    "seasons_averaged",
    "nominal_coverage_80",
    "observed_coverage_50",
    "observed_coverage_80",
    "coverage_80_ci95_low",
    "coverage_80_ci95_high",
    "coverage_error_80_pp",
    "coverage_error_80_ci95_low_pp",
    "coverage_error_80_ci95_high_pp",
    "upper_miss_80_rate",
    "lower_miss_80_rate",
    "observed_coverage_95",
    "mean_width_50",
    "median_width_50",
    "mean_width_80",
    "median_width_80",
    "mean_width_95",
    "median_width_95",
    "mean_crps",
    "mean_margin_nll",
    "bootstrap_repetitions",
    "bootstrap",
)

PAIRED_FIELDS = (
    "analysis",
    "population",
    "scope",
    "season",
    "condition",
    "condition_value",
    "metric",
    "difference_direction",
    "difference_unit",
    "paired_observations",
    "unique_games",
    "mean_difference",
    "ci95_low",
    "ci95_high",
    "bootstrap_repetitions",
    "bootstrap",
)

MODEL_FIELDS = (
    "analysis",
    "model_family",
    "outcome",
    "term",
    "estimate_pp",
    "ci95_low_pp",
    "ci95_high_pp",
    "observations",
    "team_clusters",
    "game_clusters",
    "strength_center",
    "uncertainty_center",
    "bootstrap_repetitions",
    "bootstrap",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_for(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _number(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if value in (None, ""):
        raise ValueError(f"missing numeric field {field!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric field {field!r}: {value!r}")
    return number


def _integer(row: dict[str, Any], field: str) -> int:
    value = row.get(field)
    if value in (None, ""):
        raise ValueError(f"missing integer field {field!r}")
    return int(value)


def _boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).casefold() in {"1", "true", "yes"}


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timestamp must include a timezone: {value!r}")
    return parsed


def interval_coverage(actual: float, lower: float, upper: float) -> bool:
    """Return inclusive interval coverage, rejecting invalid bounds."""
    if not all(math.isfinite(value) for value in (actual, lower, upper)):
        raise ValueError("interval values must be finite")
    if lower > upper:
        raise ValueError("interval lower bound exceeds its upper bound")
    return lower <= actual <= upper


def orient_interval(
    lower: float, upper: float, *, is_home: bool
) -> tuple[float, float]:
    """Express home-oriented bounds from the focal team's perspective."""
    if lower > upper:
        raise ValueError("interval lower bound exceeds its upper bound")
    return (lower, upper) if is_home else (-upper, -lower)


def progression_group(prior_game_count: int) -> str:
    if prior_game_count < 0:
        raise ValueError("prior game count cannot be negative")
    if prior_game_count <= 2:
        return "early_0_2"
    if prior_game_count <= 6:
        return "middle_3_6"
    return "late_7_plus"


def strength_quartile(
    strength: float, cut_points: tuple[float, float, float]
) -> str:
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("preseason strength must be within [0, 1]")
    if len(cut_points) != 3 or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in cut_points
    ):
        raise ValueError("strength quartile cut points must be three values in [0, 1]")
    if tuple(sorted(cut_points)) != cut_points:
        raise ValueError("strength quartile cut points must be ordered")
    if strength <= cut_points[0]:
        return "q1_low"
    if strength <= cut_points[1]:
        return "q2_lower_middle"
    if strength <= cut_points[2]:
        return "q3_upper_middle"
    return "q4_high"


def strength_quartile_reference(
    team_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Derive family-specific empirical quartiles from distinct team-seasons."""
    reference: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        team_seasons: dict[tuple[int, str], float] = {}
        for row in team_rows:
            if (
                row["model_family"] != family
                or row["forecast_population"] != "fbs_fbs"
            ):
                continue
            key = (int(row["season"]), str(row["team_id"]))
            strength = float(row["preseason_strength"])
            prior = team_seasons.get(key)
            if prior is not None and not math.isclose(
                prior, strength, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"inconsistent preseason strength for {family} {key!r}"
                )
            team_seasons[key] = strength
        if not team_seasons:
            raise ValueError(f"no FBS team-season strengths for {family}")
        values = np.asarray(list(team_seasons.values()), dtype=float)
        cut_points = np.quantile(values, [0.25, 0.50, 0.75], method="linear")
        reference[family] = {
            "team_seasons": len(team_seasons),
            "q25": float(cut_points[0]),
            "q50": float(cut_points[1]),
            "q75": float(cut_points[2]),
        }
    return reference


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _read_evidence(path: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    result: dict[tuple[int, str, str], dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            season = int(row["season"])
            cutoff = str(row["cutoff_utc"])
            policy = str(row["policy"])
            included = row.get("included_game_ids")
            if not isinstance(included, list):
                raise TypeError(f"evidence IDs must be a list at line {line_number}")
            if int(row["included_game_count"]) != len(included):
                raise ValueError(f"evidence count mismatch at line {line_number}")
            if issue201.sha256_json(included) != row.get("included_game_ids_sha256"):
                raise ValueError(f"evidence ID hash mismatch at line {line_number}")
            key = (season, cutoff, policy)
            if key in result:
                raise ValueError(f"duplicate issue 201 evidence key: {key}")
            result[key] = {
                **row,
                "included_game_ids": [str(game_id) for game_id in included],
                "included_game_id_set": {str(game_id) for game_id in included},
            }
    return result


def verify_inputs(repository_root: Path) -> dict[str, Any]:
    data_root = repository_root / "data" / "research"
    issue201_root = data_root / "issue_201_prebowl_forecast"
    issue247_root = data_root / "issue_247_contradictory_evidence"
    paths = {
        "issue201/forecast_scores.csv": issue201_root / "forecast_scores.csv",
        "issue201/forecast_summary.csv": issue201_root / "forecast_summary.csv",
        "issue201/replay_evidence.jsonl.gz": issue201_root / "replay_evidence.jsonl.gz",
        "issue201/manifest.json": issue201_root / "manifest.json",
        "issue247/team_game_observations.csv": issue247_root
        / "team_game_observations.csv",
        "issue247/team_game_errors.csv": issue247_root / "team_game_errors.csv",
        "issue247/manifest.json": issue247_root / "manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing retained issue inputs: {missing}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    issue201_manifest = _read_json(paths["issue201/manifest.json"])
    issue247_manifest = _read_json(paths["issue247/manifest.json"])
    issue201_outputs = issue201_manifest.get("outputs", {})
    issue247_outputs = issue247_manifest.get("outputs", {})
    expected_201_scores = issue201_outputs.get("forecast_scores.csv", {}).get("sha256")
    expected_201_summary = issue201_outputs.get("forecast_summary.csv", {}).get(
        "sha256"
    )
    expected_201_evidence = issue201_outputs.get("replay_evidence.jsonl.gz", {}).get(
        "sha256"
    )
    if hashes["issue201/forecast_scores.csv"] != expected_201_scores:
        raise ValueError("issue 201 forecast-score hash differs from its manifest")
    if hashes["issue201/forecast_summary.csv"] != expected_201_summary:
        raise ValueError("issue 201 forecast-summary hash differs from its manifest")
    if hashes["issue201/replay_evidence.jsonl.gz"] != expected_201_evidence:
        raise ValueError("issue 201 evidence hash differs from its manifest")
    for name in ("team_game_observations.csv", "team_game_errors.csv"):
        expected = issue247_outputs.get(name, {}).get("sha256")
        input_name = f"issue247/{name}"
        if hashes[input_name] != expected:
            raise ValueError(f"issue 247 {name} hash differs from its manifest")
    issue247_inputs = issue247_manifest.get("input_hashes", {})
    for source_name, local_name in (
        ("issue201/forecast_scores.csv", "issue201/forecast_scores.csv"),
        ("issue201/forecast_summary.csv", "issue201/forecast_summary.csv"),
        ("issue201/replay_evidence.jsonl.gz", "issue201/replay_evidence.jsonl.gz"),
        ("issue201/manifest.json", "issue201/manifest.json"),
    ):
        recorded = issue247_inputs.get(source_name, {}).get("sha256")
        if recorded and recorded != hashes[local_name]:
            raise ValueError(f"issue 247 lineage hash mismatch for {source_name}")
    issue247_config = issue247_manifest.get("config", {})
    if issue247_config.get("primary_minimum_prior_games") != 3:
        raise ValueError(
            "issue 247 minimum prior-game rule differs from the frozen rule"
        )
    if issue247_config.get("surprise_group_threshold") != 0.5:
        raise ValueError("issue 247 surprise threshold differs from the frozen rule")
    if issue247_config.get("model_arms") != [
        "History_existing",
        "Context_existing",
    ]:
        raise ValueError("issue 247 model arms differ from the frozen population")
    return {
        "paths": paths,
        "hashes": hashes,
        "issue201_manifest": issue201_manifest,
        "issue247_manifest": issue247_manifest,
    }


def _team_error_index(
    rows: list[dict[str, str]],
) -> dict[tuple[int, str, str], list[dict[str, str]]]:
    indexed: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[int, str, str, str]] = set()
    for row in rows:
        game_key = (
            int(row["season"]),
            str(row["model_arm"]),
            str(row["team_id"]),
            str(row["game_id"]),
        )
        if game_key in seen:
            raise ValueError(f"duplicate issue 247 team-game error row: {game_key}")
        seen.add(game_key)
        indexed[game_key[:3]].append(row)
    return indexed


def _verify_prior_history(
    row: dict[str, str],
    score: dict[str, str],
    evidence_records: dict[tuple[int, str, str], dict[str, Any]],
    errors_by_team: dict[tuple[int, str, str], list[dict[str, str]]],
) -> None:
    season = int(row["season"])
    model_arm = str(row["model_arm"])
    team_id = str(row["team_id"])
    game_id = str(row["game_id"])
    cutoff = str(row["cutoff_utc"])
    evidence_key = (season, cutoff, str(row["policy"]))
    evidence = evidence_records.get(evidence_key)
    if evidence is None:
        raise ValueError(f"missing issue 201 evidence for team forecast {evidence_key}")
    if evidence["included_game_ids_sha256"] != score["evidence_ids_sha256"]:
        raise ValueError(
            f"issue 201 cutoff evidence hash mismatch for {row['forecast_id']}"
        )
    included_ids = evidence["included_game_id_set"]
    if game_id in included_ids:
        raise ValueError(f"target result appears in its own evidence: {game_id}")
    listed_ids = [
        game
        for game in str(row.get("eligible_previous_game_ids", "")).split("|")
        if game
    ]
    if len(listed_ids) != int(row["eligible_prior_game_count"]):
        raise ValueError(
            f"issue 247 prior-game count mismatch for {row['forecast_id']}"
        )
    if len(set(listed_ids)) != len(listed_ids) or game_id in listed_ids:
        raise ValueError(f"invalid prior-game IDs for target game {game_id}")
    cutoff_dt = _parse_utc(cutoff)
    matching_errors = errors_by_team.get((season, model_arm, team_id), [])
    eligible_prior: dict[str, dict[str, str]] = {}
    for prior in matching_errors:
        prior_game_id = str(prior["game_id"])
        if (
            _parse_utc(str(prior["kickoff_utc"])) < cutoff_dt
            and prior_game_id in included_ids
        ):
            eligible_prior[prior_game_id] = prior
    if set(listed_ids) != set(eligible_prior):
        raise ValueError(
            f"issue 247 prior IDs do not match earlier games in issue 201 evidence for {game_id}"
        )
    surprises: list[float] = []
    for prior_game_id in listed_ids:
        prior = eligible_prior[prior_game_id]
        if _parse_utc(str(prior["cutoff_utc"])) >= cutoff_dt:
            raise ValueError(f"prior forecast cutoff is not before {game_id}")
        surprises.append(_number(prior, "standardized_surprise"))
    count = len(listed_ids)
    eligible = count >= issue247.MIN_PRIOR_GAMES
    if _boolean(row["primary_eligible"]) != eligible:
        raise ValueError(f"issue 247 eligibility mismatch for {row['forecast_id']}")
    accumulated_text = row.get("accumulated_surprise", "")
    if count:
        expected = math.fsum(surprises) / count
        if not accumulated_text or not math.isclose(
            float(accumulated_text), expected, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"issue 247 surprise average mismatch for {row['forecast_id']}"
            )
    elif accumulated_text:
        raise ValueError(f"unexpected surprise value without prior games for {game_id}")
    group = str(row.get("surprise_group", ""))
    if eligible and group != issue247.surprise_group(float(accumulated_text)):
        raise ValueError(f"issue 247 surprise group mismatch for {row['forecast_id']}")
    if not eligible and group:
        raise ValueError(
            f"ineligible surprise history has a group for {row['forecast_id']}"
        )


def _coverage_fields(score: dict[str, str]) -> dict[str, Any]:
    actual = _number(score, "actual_home_margin")
    result: dict[str, Any] = {}
    for level in (50, 80, 95):
        lower = _number(score, f"margin_interval_{level}_low")
        upper = _number(score, f"margin_interval_{level}_high")
        covered = interval_coverage(actual, lower, upper)
        retained = int(score[f"coverage_{level}"])
        if retained not in (0, 1) or bool(retained) != covered:
            raise ValueError(
                f"issue 201 {level}% coverage parity failed for {score['forecast_id']}"
            )
        result[f"margin_interval_{level}_width"] = upper - lower
        result[f"coverage_{level}"] = covered
        if level == 80:
            result["upper_miss_80"] = actual > upper
            result["lower_miss_80"] = actual < lower
    return result


def build_calibration_observations(
    score_rows: list[dict[str, str]],
    surprise_rows: list[dict[str, str]],
    team_error_rows: list[dict[str, str]],
    evidence_records: dict[tuple[int, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Join retained issue 201 scores to issue 247 histories and verify lineage."""
    error_index = _team_error_index(team_error_rows)
    errors_by_team = error_index
    forecast_rows: list[dict[str, Any]] = []
    rolling_scores: dict[tuple[str, str], dict[str, str]] = {}
    score_keys: set[tuple[int, str, str, str, str]] = set()
    for score in score_rows:
        arm = str(score["model_arm"])
        stage = str(score["forecast_stage"])
        population = str(score["forecast_population"])
        if arm not in set(MODEL_ARMS.values()) or population not in POPULATIONS:
            continue
        if stage not in {"rolling", "preseason"}:
            continue
        season = int(score["season"])
        if season not in SEASONS or str(score["policy"]) != "existing":
            raise ValueError(
                f"unexpected issue 201 forecast provenance: {score['forecast_id']}"
            )
        family = arm.removesuffix("_existing")
        if score["prior_family"] != family:
            raise ValueError(f"model family mismatch for {score['forecast_id']}")
        row_key = (
            season,
            str(score["game_id"]),
            str(score["cutoff_utc"]),
            population,
            arm,
        )
        if row_key in score_keys:
            raise ValueError(f"duplicate issue 201 game forecast: {row_key}")
        score_keys.add(row_key)
        coverage = _coverage_fields(score)
        forecast: dict[str, Any] = {
            "forecast_stage": stage,
            "season": season,
            "model_family": family,
            "model_arm": arm,
            "policy": "existing",
            "forecast_id": str(score["forecast_id"]),
            "game_id": str(score["game_id"]),
            "forecast_population": population,
            "phase": str(score["phase"]),
            "cutoff_utc": str(score["cutoff_utc"]),
            "kickoff_utc": str(score["kickoff_utc"]),
            "home_id": str(score["home_id"]),
            "home_team": str(score["home_team"]),
            "home_subdivision": str(score["home_subdivision"]),
            "away_id": str(score["away_id"]),
            "away_team": str(score["away_team"]),
            "away_subdivision": str(score["away_subdivision"]),
            "evidence_game_count": int(score["evidence_game_count"]),
            "evidence_ids_sha256": str(score["evidence_ids_sha256"]),
            "actual_home_margin": _number(score, "actual_home_margin"),
            "expected_home_margin": _number(score, "expected_home_margin"),
            "median_home_margin": _number(score, "median_home_margin"),
            "crps": _number(score, "crps"),
            "margin_nll": _number(score, "margin_nll"),
            **coverage,
        }
        for level in (50, 80, 95):
            forecast[f"margin_interval_{level}_low"] = _number(
                score, f"margin_interval_{level}_low"
            )
            forecast[f"margin_interval_{level}_high"] = _number(
                score, f"margin_interval_{level}_high"
            )
        forecast_rows.append(forecast)
        if stage == "rolling":
            key = (str(score["forecast_id"]), arm)
            if key in rolling_scores:
                raise ValueError(f"duplicate rolling forecast ID: {key}")
            rolling_scores[key] = score

    team_rows: list[dict[str, Any]] = []
    seen_team_forecasts: set[tuple[str, str, str]] = set()
    orientations_by_forecast: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in surprise_rows:
        key = (str(row["forecast_id"]), str(row["model_arm"]))
        score = rolling_scores.get(key)
        if score is None:
            raise ValueError(
                f"issue 247 observation has no rolling #201 forecast: {key}"
            )
        for field in ("game_id", "cutoff_utc", "prior_family", "forecast_population"):
            if str(row[field]) != str(score[field]):
                raise ValueError(f"issue 247 {field} join mismatch for {key}")
        if int(row["season"]) != int(score["season"]):
            raise ValueError(f"issue 247 season join mismatch for {key}")
        _verify_prior_history(row, score, evidence_records, errors_by_team)
        is_home = _boolean(row["is_home"])
        team_id = str(row["team_id"])
        home_id = str(score["home_id"])
        away_id = str(score["away_id"])
        expected_team_id = home_id if is_home else away_id
        expected_subdivision = (
            str(score["home_subdivision"])
            if is_home
            else str(score["away_subdivision"])
        )
        if team_id != expected_team_id or expected_subdivision != "fbs":
            raise ValueError(f"issue 247 focal-team orientation mismatch for {key}")
        orientation_key = (key[0], key[1])
        if team_id in orientations_by_forecast[orientation_key]:
            raise ValueError(
                f"duplicate team perspective for forecast {key}: {team_id}"
            )
        orientations_by_forecast[orientation_key].add(team_id)
        unique_key = (key[0], key[1], team_id)
        if unique_key in seen_team_forecasts:
            raise ValueError(f"duplicate issue 247 team forecast: {unique_key}")
        seen_team_forecasts.add(unique_key)
        is_primary = str(score["forecast_population"]) == "fbs_fbs"
        sign = 1.0 if is_home else -1.0
        actual = _number(score, "actual_home_margin") * sign
        expected = _number(score, "expected_home_margin") * sign
        median = _number(score, "median_home_margin") * sign
        if not math.isclose(
            _number(row, "actual_margin"), actual, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"issue 247 actual margin orientation mismatch for {key}")
        if not math.isclose(
            _number(row, "predicted_margin"), expected, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"issue 247 predicted margin orientation mismatch for {key}"
            )
        oriented: dict[str, Any] = {
            "season": int(score["season"]),
            "model_family": str(score["prior_family"]),
            "model_arm": str(score["model_arm"]),
            "forecast_id": str(score["forecast_id"]),
            "game_id": str(score["game_id"]),
            "forecast_population": str(score["forecast_population"]),
            "phase": str(score["phase"]),
            "cutoff_utc": str(score["cutoff_utc"]),
            "kickoff_utc": str(score["kickoff_utc"]),
            "evidence_game_count": int(score["evidence_game_count"]),
            "evidence_ids_sha256": str(score["evidence_ids_sha256"]),
            "team_id": team_id,
            "team": str(row["team"]),
            "opponent_id": str(row["opponent_id"]),
            "opponent": str(row["opponent"]),
            "is_home": is_home,
            "team_game_number": int(row["team_game_number"]),
            "prior_game_count": int(row["eligible_prior_game_count"]),
            "progression_group": progression_group(
                int(row["eligible_prior_game_count"])
            ),
            "eligible_previous_game_ids": str(
                row.get("eligible_previous_game_ids", "")
            ),
            "primary_eligible": _boolean(row["primary_eligible"]),
            "accumulated_surprise": (
                None
                if not row.get("accumulated_surprise")
                else float(row["accumulated_surprise"])
            ),
            "surprise_group": str(row.get("surprise_group", "")),
            "preseason_expected_rank": _number(row, "preseason_expected_rank"),
            "preseason_rank_sd": _number(row, "preseason_rank_sd"),
            "preseason_strength": _number(row, "preseason_rank_strength_percentile"),
            "preseason_rank_uncertainty_fraction": _number(
                row, "preseason_rank_uncertainty_fraction"
            ),
            "preseason_rank_population": int(row["preseason_rank_population"]),
            "preseason_rank_top10_probability": _number(
                row, "preseason_rank_top10_probability"
            ),
            "preseason_rank_pmf_sha256": str(row["preseason_rank_pmf_sha256"]),
            "actual_margin": actual,
            "expected_margin": expected,
            "median_margin": median,
            "crps": _number(score, "crps"),
            "margin_nll": _number(score, "margin_nll"),
        }
        for level in (50, 80, 95):
            lower, upper = orient_interval(
                _number(score, f"margin_interval_{level}_low"),
                _number(score, f"margin_interval_{level}_high"),
                is_home=is_home,
            )
            covered = interval_coverage(actual, lower, upper)
            if covered != bool(int(score[f"coverage_{level}"])):
                raise ValueError(f"oriented interval coverage mismatch for {key}")
            oriented[f"margin_interval_{level}_low"] = lower
            oriented[f"margin_interval_{level}_high"] = upper
            oriented[f"margin_interval_{level}_width"] = upper - lower
            oriented[f"coverage_{level}"] = covered
            if level == 80:
                oriented["upper_miss_80"] = actual > upper
                oriented["lower_miss_80"] = actual < lower
        if not is_primary and not (
            str(score["home_subdivision"]) == "fbs"
            or str(score["away_subdivision"]) == "fbs"
        ):
            raise ValueError(f"unsupported FBS–FCS orientation for {key}")
        team_rows.append(oriented)

    for key, score in rolling_scores.items():
        population = str(score["forecast_population"])
        expected_count = 2 if population == "fbs_fbs" else 1
        actual_teams = orientations_by_forecast.get(key, set())
        expected_teams = (
            {str(score["home_id"]), str(score["away_id"])}
            if population == "fbs_fbs"
            else {
                str(score["home_id"])
                if str(score["home_subdivision"]) == "fbs"
                else str(score["away_id"])
            }
        )
        if len(actual_teams) != expected_count or actual_teams != expected_teams:
            raise ValueError(f"incomplete issue 247 perspectives for forecast {key}")

    forecast_rows.sort(
        key=lambda row: (
            row["forecast_stage"],
            int(row["season"]),
            row["model_arm"],
            row["forecast_population"],
            row["game_id"],
        )
    )
    team_rows.sort(
        key=lambda row: (
            int(row["season"]),
            row["model_arm"],
            row["forecast_population"],
            row["game_id"],
            row["team_id"],
        )
    )
    audit = {
        "forecast_rows": len(forecast_rows),
        "rolling_forecasts_joined": len(rolling_scores),
        "team_oriented_rows": len(team_rows),
        "unique_team_forecasts": len(seen_team_forecasts),
        "coverage_parity_checks": len(forecast_rows) * 3,
        "prior_history_checks": len(surprise_rows),
        "target_game_excluded_from_prior_histories": True,
    }
    return forecast_rows, team_rows, audit


def verify_issue201_summary_parity(
    forecast_rows: list[dict[str, Any]],
    retained_summary_rows: list[dict[str, str]],
) -> dict[str, int]:
    """Check issue 201's season/phase summaries against retained score rows."""
    selected_summary = [
        row
        for row in retained_summary_rows
        if row["model_arm"] in set(MODEL_ARMS.values())
        and row["forecast_population"] in POPULATIONS
        and row["forecast_stage"] in {"rolling", "preseason"}
        and str(row["season"]) in {str(season) for season in SEASONS}
    ]
    phase_fields = (
        "forecast_stage",
        "forecast_population",
        "model_arm",
        "season",
        "phase",
    )
    season_fields = phase_fields[:-1]
    phase_summary_index: dict[tuple[str, ...], dict[str, str]] = {}
    season_summary_index: dict[tuple[str, ...], dict[str, str]] = {}
    for row in selected_summary:
        if row["phase"] == "all_phases":
            key = tuple(str(row[field]) for field in season_fields)
            if key in season_summary_index:
                raise ValueError(f"duplicate issue 201 season summary: {key}")
            season_summary_index[key] = row
        else:
            key = tuple(str(row[field]) for field in phase_fields)
            if key in phase_summary_index:
                raise ValueError(f"duplicate issue 201 phase summary group: {key}")
            phase_summary_index[key] = row
    score_index: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    season_index: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in forecast_rows:
        phase_key = tuple(str(row[field]) for field in phase_fields)
        season_key = tuple(str(row[field]) for field in season_fields)
        score_index[phase_key].append(row)
        season_index[season_key].append(row)
    if set(phase_summary_index) != set(score_index):
        missing = sorted(set(score_index) - set(phase_summary_index))[:3]
        extra = sorted(set(phase_summary_index) - set(score_index))[:3]
        raise ValueError(
            f"issue 201 phase summary mismatch: missing={missing}, extra={extra}"
        )
    if set(season_summary_index) != set(season_index):
        missing = sorted(set(season_index) - set(season_summary_index))[:3]
        extra = sorted(set(season_summary_index) - set(season_index))[:3]
        raise ValueError(
            f"issue 201 season summary mismatch: missing={missing}, extra={extra}"
        )

    def compare(
        summary: dict[str, str], members: list[dict[str, Any]], label: str
    ) -> int:
        if int(summary["games"]) != len(members):
            raise ValueError(f"issue 201 summary game count mismatch for {label}")
        comparisons: list[tuple[str, float]] = []
        for level in (50, 80, 95):
            comparisons.extend(
                (
                    (
                        f"coverage_{level}",
                        float(
                            np.mean([bool(row[f"coverage_{level}"]) for row in members])
                        ),
                    ),
                    (
                        f"margin_interval_{level}_width",
                        float(
                            np.mean(
                                [
                                    float(row[f"margin_interval_{level}_width"])
                                    for row in members
                                ]
                            )
                        ),
                    ),
                )
            )
        comparisons.extend(
            (
                ("crps", float(np.mean([float(row["crps"]) for row in members]))),
                (
                    "margin_nll",
                    float(np.mean([float(row["margin_nll"]) for row in members])),
                ),
            )
        )
        for field, actual in comparisons:
            if not math.isclose(
                float(summary[field]), actual, rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise ValueError(f"issue 201 {field} summary parity failed for {label}")
        return len(comparisons)

    phase_checks = sum(
        compare(summary, score_index[key], f"phase {key}")
        for key, summary in phase_summary_index.items()
    )
    season_checks = sum(
        compare(summary, season_index[key], f"season {key}")
        for key, summary in season_summary_index.items()
    )

    cross_season_checks = 0
    aggregate_keys = {key[:3] for key in season_summary_index}
    for aggregate_key in aggregate_keys:
        summaries = [
            row for key, row in season_summary_index.items() if key[:3] == aggregate_key
        ]
        members = [
            row
            for key, grouped_rows in season_index.items()
            if key[:3] == aggregate_key
            for row in grouped_rows
        ]
        total_games = sum(int(row["games"]) for row in summaries)
        if total_games != len(members):
            raise ValueError(
                f"issue 201 cross-season game count mismatch for {aggregate_key}"
            )
        comparisons: list[tuple[str, float, float]] = []
        for level in (50, 80, 95):
            comparisons.extend(
                (
                    (
                        f"coverage_{level}",
                        float(
                            np.mean([bool(row[f"coverage_{level}"]) for row in members])
                        ),
                        sum(
                            float(row[f"coverage_{level}"]) * int(row["games"])
                            for row in summaries
                        )
                        / total_games,
                    ),
                    (
                        f"margin_interval_{level}_width",
                        float(
                            np.mean(
                                [
                                    float(row[f"margin_interval_{level}_width"])
                                    for row in members
                                ]
                            )
                        ),
                        sum(
                            float(row[f"margin_interval_{level}_width"])
                            * int(row["games"])
                            for row in summaries
                        )
                        / total_games,
                    ),
                )
            )
        for field in ("crps", "margin_nll"):
            comparisons.append(
                (
                    field,
                    float(np.mean([float(row[field]) for row in members])),
                    sum(float(row[field]) * int(row["games"]) for row in summaries)
                    / total_games,
                )
            )
        for field, direct, weighted in comparisons:
            if not math.isclose(direct, weighted, rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(
                    f"issue 201 cross-season {field} parity failed for {aggregate_key}"
                )
            cross_season_checks += 1
    return {
        "phase_summary_checks": phase_checks,
        "season_summary_checks": season_checks,
        "cross_season_aggregate_checks": cross_season_checks,
    }


def _cluster_codes(
    rows: list[dict[str, Any]], mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, int, int]:
    games = sorted({f"{row['season']}:{row['game_id']}" for row in rows})
    game_map = {value: index for index, value in enumerate(games)}
    game_codes = np.asarray(
        [game_map[f"{row['season']}:{row['game_id']}"] for row in rows], dtype=int
    )
    if mode == "team":
        teams = sorted({str(row["team_id"]) for row in rows})
        team_map = {value: index for index, value in enumerate(teams)}
        team_codes = np.asarray(
            [team_map[str(row["team_id"])] for row in rows], dtype=int
        )
        return team_codes, game_codes, None, None, len(teams), len(games)
    if mode == "game":
        teams = sorted(
            {
                str(team_id)
                for row in rows
                for team_id in (row["home_id"], row["away_id"])
            }
        )
        team_map = {value: index for index, value in enumerate(teams)}
        home_codes = np.asarray(
            [team_map[str(row["home_id"])] for row in rows], dtype=int
        )
        away_codes = np.asarray(
            [team_map[str(row["away_id"])] for row in rows], dtype=int
        )
        return (
            np.asarray([], dtype=int),
            game_codes,
            home_codes,
            away_codes,
            len(teams),
            len(games),
        )
    raise ValueError(f"unknown bootstrap mode: {mode}")


def _bootstrap_weights(
    rng: np.random.Generator,
    codes: tuple[
        np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, int, int
    ],
    mode: str,
) -> np.ndarray:
    team_codes, game_codes, home_codes, away_codes, team_count, game_count = codes
    team_counts = np.bincount(
        rng.integers(0, team_count, size=team_count), minlength=team_count
    )
    game_counts = np.bincount(
        rng.integers(0, game_count, size=game_count), minlength=game_count
    )
    if mode == "team":
        return (team_counts[team_codes] * game_counts[game_codes]).astype(float)
    assert home_codes is not None and away_codes is not None
    return (
        game_counts[game_codes]
        * (team_counts[home_codes] + team_counts[away_codes])
        / 2.0
    )


def _bootstrap_draws(
    rows: list[dict[str, Any]],
    values: np.ndarray,
    *,
    mode: str,
    repetitions: int,
    seed: int,
    label: str,
) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=float)
    codes = _cluster_codes(rows, mode)
    rng = np.random.default_rng(_seed_for(seed, label))
    draws: list[float] = []
    for _ in range(repetitions):
        weights = _bootstrap_weights(rng, codes, mode)
        total = float(weights.sum())
        if total > 0.0:
            draws.append(float(np.dot(values, weights) / total))
    return np.asarray(draws, dtype=float)


def _estimate_values(rows: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for level in (50, 80, 95):
        result[f"observed_coverage_{level}"] = float(
            np.mean([bool(row[f"coverage_{level}"]) for row in rows])
        )
        result[f"mean_width_{level}"] = float(
            np.mean([float(row[f"margin_interval_{level}_width"]) for row in rows])
        )
        result[f"median_width_{level}"] = float(
            np.median([float(row[f"margin_interval_{level}_width"]) for row in rows])
        )
    result["upper_miss_80_rate"] = float(
        np.mean([bool(row["upper_miss_80"]) for row in rows])
    )
    result["lower_miss_80_rate"] = float(
        np.mean([bool(row["lower_miss_80"]) for row in rows])
    )
    result["mean_crps"] = float(np.mean([float(row["crps"]) for row in rows]))
    result["mean_margin_nll"] = float(
        np.mean([float(row["margin_nll"]) for row in rows])
    )
    return result


def _base_summary(
    rows: list[dict[str, Any]],
    *,
    analysis: str,
    forecast_stage: str,
    population: str,
    family: str,
    scope: str,
    season: str,
    condition: str,
    condition_value: str,
    observation_unit: str,
    mode: str,
    draws: np.ndarray | None = None,
    seasons_averaged: int = 0,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty calibration group")
    estimates = _estimate_values(rows)
    if draws is None:
        coverage_values = np.asarray(
            [float(bool(row["coverage_80"])) for row in rows], dtype=float
        )
        draws = _bootstrap_draws(
            rows,
            coverage_values,
            mode=mode,
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=BOOTSTRAP_SEED,
            label=f"{analysis}:{forecast_stage}:{population}:{family}:{scope}:{season}:{condition}:{condition_value}",
        )
    low, high = (
        (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))
        if draws.size
        else (None, None)
    )
    if observation_unit == "game":
        teams = {
            str(team_id) for row in rows for team_id in (row["home_id"], row["away_id"])
        }
    else:
        teams = {str(row["team_id"]) for row in rows}
    result: dict[str, Any] = {
        "analysis": analysis,
        "forecast_stage": forecast_stage,
        "population": population,
        "model_family": family,
        "scope": scope,
        "season": season,
        "condition": condition,
        "condition_value": condition_value,
        "observation_unit": observation_unit,
        "n_forecasts": len(rows),
        "unique_games": len({f"{row['season']}:{row['game_id']}" for row in rows}),
        "unique_teams": len(teams),
        "seasons_averaged": seasons_averaged,
        "nominal_coverage_80": NOMINAL_COVERAGE[80],
        "observed_coverage_50": estimates["observed_coverage_50"],
        "observed_coverage_80": estimates["observed_coverage_80"],
        "coverage_80_ci95_low": low,
        "coverage_80_ci95_high": high,
        "coverage_error_80_pp": (estimates["observed_coverage_80"] - 0.80) * 100.0,
        "coverage_error_80_ci95_low_pp": (
            None if low is None else (low - 0.80) * 100.0
        ),
        "coverage_error_80_ci95_high_pp": (
            None if high is None else (high - 0.80) * 100.0
        ),
        "upper_miss_80_rate": estimates["upper_miss_80_rate"],
        "lower_miss_80_rate": estimates["lower_miss_80_rate"],
        "observed_coverage_95": estimates["observed_coverage_95"],
        "mean_width_50": estimates["mean_width_50"],
        "median_width_50": estimates["median_width_50"],
        "mean_width_80": estimates["mean_width_80"],
        "median_width_80": estimates["median_width_80"],
        "mean_width_95": estimates["mean_width_95"],
        "median_width_95": estimates["median_width_95"],
        "mean_crps": estimates["mean_crps"],
        "mean_margin_nll": estimates["mean_margin_nll"],
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap": (
            "two-way pigeonhole: home/away team endpoints and game ID"
            if mode == "game"
            else "two-way pigeonhole: focal team ID and game ID"
        ),
    }
    return result


def _append_summaries(
    output: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    analysis: str,
    stage: str,
    population: str,
    family: str,
    condition: str = "",
    condition_value: str = "",
    observation_unit: str,
    mode: str,
) -> None:
    if not rows:
        return
    pooled = _base_summary(
        rows,
        analysis=analysis,
        forecast_stage=stage,
        population=population,
        family=family,
        scope="pooled",
        season="",
        condition=condition,
        condition_value=condition_value,
        observation_unit=observation_unit,
        mode=mode,
    )
    output.append(pooled)
    season_rows = {
        str(season): [row for row in rows if int(row["season"]) == season]
        for season in SEASONS
    }
    present = [(season, members) for season, members in season_rows.items() if members]
    for season, members in present:
        output.append(
            _base_summary(
                members,
                analysis=analysis,
                forecast_stage=stage,
                population=population,
                family=family,
                scope="season",
                season=season,
                condition=condition,
                condition_value=condition_value,
                observation_unit=observation_unit,
                mode=mode,
            )
        )
    season_draws = [
        _bootstrap_draws(
            members,
            np.asarray([float(bool(row["coverage_80"])) for row in members]),
            mode=mode,
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=BOOTSTRAP_SEED,
            label=f"{analysis}:{stage}:{population}:{family}:equal_season:{season}:{condition}:{condition_value}",
        )
        for season, members in present
    ]
    equal_draws = (
        np.mean(np.vstack(season_draws), axis=0)
        if season_draws
        and all(draws.size == BOOTSTRAP_REPETITIONS for draws in season_draws)
        else np.asarray([], dtype=float)
    )
    season_estimates = [_estimate_values(members) for _, members in present]
    equal_rows = [dict(row) for _, members in present for row in members]
    equal = _base_summary(
        equal_rows,
        analysis=analysis,
        forecast_stage=stage,
        population=population,
        family=family,
        scope="equal_season",
        season="",
        condition=condition,
        condition_value=condition_value,
        observation_unit=observation_unit,
        mode=mode,
        draws=equal_draws,
        seasons_averaged=len(present),
    )
    for key in (
        "observed_coverage_50",
        "observed_coverage_80",
        "upper_miss_80_rate",
        "lower_miss_80_rate",
        "observed_coverage_95",
        "mean_width_50",
        "median_width_50",
        "mean_width_80",
        "median_width_80",
        "mean_width_95",
        "median_width_95",
        "mean_crps",
        "mean_margin_nll",
    ):
        equal[key] = float(np.mean([item[key] for item in season_estimates]))
    equal["coverage_error_80_pp"] = (equal["observed_coverage_80"] - 0.80) * 100.0
    output.append(equal)


def build_calibration_summaries(
    forecast_rows: list[dict[str, Any]],
    team_rows: list[dict[str, Any]],
    strength_reference: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for family in FAMILIES:
        for population in POPULATIONS:
            for stage in ("rolling", "preseason"):
                members = [
                    row
                    for row in forecast_rows
                    if row["model_family"] == family
                    and row["forecast_population"] == population
                    and row["forecast_stage"] == stage
                ]
                if members:
                    _append_summaries(
                        output,
                        members,
                        analysis="overall"
                        if stage == "rolling"
                        else "preseason_reference",
                        stage=stage,
                        population=population,
                        family=family,
                        observation_unit="game",
                        mode="game",
                    )
        members = [
            row
            for row in team_rows
            if row["model_family"] == family and row["forecast_population"] == "fbs_fbs"
        ]
        cut_points = tuple(
            float(strength_reference[family][key]) for key in ("q25", "q50", "q75")
        )
        for analysis, condition, getter, allowed in (
            (
                "season_progression",
                "progression_group",
                lambda row: row["progression_group"],
                PROGRESSION_GROUPS,
            ),
            (
                "preseason_strength",
                "strength_quartile",
                lambda row, cut_points=cut_points: strength_quartile(
                    float(row["preseason_strength"]), cut_points
                ),
                ("q1_low", "q2_lower_middle", "q3_upper_middle", "q4_high"),
            ),
            (
                "accumulated_surprise",
                "surprise_group",
                lambda row: row["surprise_group"],
                SURPRISE_GROUPS,
            ),
        ):
            eligible = (
                [row for row in members if row["primary_eligible"]]
                if analysis == "accumulated_surprise"
                else members
            )
            for value in allowed:
                subset = [row for row in eligible if getter(row) == value]
                if not subset:
                    continue
                _append_summaries(
                    output,
                    subset,
                    analysis=analysis,
                    stage="rolling",
                    population="fbs_fbs",
                    family=family,
                    condition=condition,
                    condition_value=value,
                    observation_unit="team_game",
                    mode="team",
                )
    return output


def _pair_metric_values(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=float)


def _paired_summary_rows(
    rows: list[dict[str, Any]],
    *,
    analysis: str,
    population: str,
    scope: str,
    season: str,
    condition: str,
    condition_value: str,
    observation_unit: str,
    mode: str,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    result: list[dict[str, Any]] = []
    metric_specs = (
        ("coverage_80_error", "coverage_80_difference_pp", "percentage_points", 100.0),
        ("width_difference", "width_difference", "margin_points", 1.0),
        ("crps_difference", "crps_difference", "margin_points", 1.0),
        ("margin_nll_difference", "margin_nll_difference", "log_density", 1.0),
    )
    for field, metric_name, unit, scale in metric_specs:
        values = _pair_metric_values(rows, field) * scale
        draws = _bootstrap_draws(
            rows,
            values,
            mode=mode,
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=BOOTSTRAP_SEED,
            label=f"paired:{analysis}:{population}:{scope}:{season}:{condition}:{condition_value}:{metric_name}",
        )
        low, high = (
            (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))
            if draws.size
            else (None, None)
        )
        result.append(
            {
                "analysis": analysis,
                "population": population,
                "scope": scope,
                "season": season,
                "condition": condition,
                "condition_value": condition_value,
                "metric": metric_name,
                "difference_direction": "History minus Context",
                "difference_unit": unit,
                "paired_observations": len(rows),
                "unique_games": len(
                    {f"{row['season']}:{row['game_id']}" for row in rows}
                ),
                "mean_difference": float(np.mean(values)),
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                "bootstrap": (
                    "two-way pigeonhole: home/away team endpoints and game ID"
                    if mode == "game"
                    else "two-way pigeonhole: focal team ID and game ID"
                ),
            }
        )
    return result


def build_paired_comparisons(
    forecast_rows: list[dict[str, Any]], team_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    game_index: dict[tuple[int, str, str, str], dict[str, dict[str, Any]]] = (
        defaultdict(dict)
    )
    for row in forecast_rows:
        if (
            row["forecast_stage"] != "rolling"
            or row["forecast_population"] != "fbs_fbs"
        ):
            continue
        key = (
            int(row["season"]),
            str(row["game_id"]),
            str(row["cutoff_utc"]),
            str(row["forecast_population"]),
        )
        family = str(row["model_family"])
        if family in game_index[key]:
            raise ValueError(f"duplicate family forecast for paired game {key}")
        game_index[key][family] = row
    paired_games: list[dict[str, Any]] = []
    for key, members in sorted(game_index.items()):
        if set(members) != set(FAMILIES):
            continue
        history = members["History"]
        context = members["Context"]
        if (
            history["evidence_game_count"] != context["evidence_game_count"]
            or history["evidence_ids_sha256"] != context["evidence_ids_sha256"]
        ):
            raise ValueError(f"matched History/Context evidence differs for {key}")
        paired_games.append(
            {
                **history,
                "coverage_80_error": float(history["coverage_80"])
                - float(context["coverage_80"]),
                "width_difference": float(history["margin_interval_80_width"])
                - float(context["margin_interval_80_width"]),
                "crps_difference": float(history["crps"]) - float(context["crps"]),
                "margin_nll_difference": float(history["margin_nll"])
                - float(context["margin_nll"]),
            }
        )
    output: list[dict[str, Any]] = []
    for scope, season, subset in _scopes(paired_games):
        output.extend(
            _paired_summary_rows(
                subset,
                analysis="overall",
                population="fbs_fbs",
                scope=scope,
                season=season,
                condition="",
                condition_value="",
                observation_unit="game",
                mode="game",
            )
        )

    team_index: dict[tuple[int, str, str, str, str], dict[str, dict[str, Any]]] = (
        defaultdict(dict)
    )
    for row in team_rows:
        if row["forecast_population"] != "fbs_fbs":
            continue
        key = (
            int(row["season"]),
            str(row["game_id"]),
            str(row["cutoff_utc"]),
            str(row["team_id"]),
            str(row["forecast_population"]),
        )
        family = str(row["model_family"])
        if family in team_index[key]:
            raise ValueError(f"duplicate team family forecast for paired row {key}")
        team_index[key][family] = row
    paired_teams: list[dict[str, Any]] = []
    for key, members in sorted(team_index.items()):
        if set(members) != set(FAMILIES):
            continue
        history, context = members["History"], members["Context"]
        if history["prior_game_count"] != context["prior_game_count"]:
            raise ValueError(f"matched team progression differs for {key}")
        paired_teams.append(
            {
                **history,
                "coverage_80_error": float(history["coverage_80"])
                - float(context["coverage_80"]),
                "width_difference": float(history["margin_interval_80_width"])
                - float(context["margin_interval_80_width"]),
                "crps_difference": float(history["crps"]) - float(context["crps"]),
                "margin_nll_difference": float(history["margin_nll"])
                - float(context["margin_nll"]),
            }
        )
    for group in PROGRESSION_GROUPS:
        group_rows = [row for row in paired_teams if row["progression_group"] == group]
        for scope, season, subset in _scopes(group_rows):
            output.extend(
                _paired_summary_rows(
                    subset,
                    analysis="season_progression",
                    population="fbs_fbs",
                    scope=scope,
                    season=season,
                    condition="progression_group",
                    condition_value=group,
                    observation_unit="team_game",
                    mode="team",
                )
            )
    return output, {
        "matched_fbs_fbs_forecasts": len(paired_games),
        "paired_team_orientations": len(paired_teams),
        "unmatched_game_forecasts": sum(
            1 for members in game_index.values() if set(members) != set(FAMILIES)
        ),
    }


def _scopes(rows: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    result: list[tuple[str, str, list[dict[str, Any]]]] = [("pooled", "", rows)]
    for season in SEASONS:
        members = [row for row in rows if int(row["season"]) == season]
        if members:
            result.append(("season", str(season), members))
    return result


def _independent_columns(
    columns: list[np.ndarray], names: list[str]
) -> tuple[np.ndarray, list[str]]:
    kept_columns: list[np.ndarray] = []
    kept_names: list[str] = []
    current_rank = 0
    for column, name in zip(columns, names, strict=True):
        candidate = np.column_stack([*kept_columns, column])
        rank = int(np.linalg.matrix_rank(candidate))
        if rank > current_rank:
            kept_columns.append(column)
            kept_names.append(name)
            current_rank = rank
    return np.column_stack(kept_columns), kept_names


def _linear_probability_model(
    rows: list[dict[str, Any]],
    columns: list[np.ndarray],
    names: list[str],
    *,
    analysis: str,
    family: str,
    strength_center: float,
    uncertainty_center: float,
    repetitions: int,
    seed_label: str,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    design, terms = _independent_columns(columns, names)
    outcome = np.asarray([0.0 if bool(row["coverage_80"]) else 1.0 for row in rows])
    observed = np.linalg.lstsq(design, outcome, rcond=None)[0]
    codes = _cluster_codes(rows, "team")
    rng = np.random.default_rng(_seed_for(BOOTSTRAP_SEED, seed_label))
    draws: list[list[float]] = [[] for _ in terms]
    for _ in range(repetitions):
        weights = _bootstrap_weights(rng, codes, "team")
        if float(weights.sum()) <= 0:
            continue
        gram = design.T @ (design * weights[:, None])
        rhs = design.T @ (outcome * weights)
        coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        for index, value in enumerate(coefficients):
            if math.isfinite(float(value)):
                draws[index].append(float(value))
    output: list[dict[str, Any]] = []
    for index, term in enumerate(terms):
        values = np.asarray(draws[index], dtype=float)
        low, high = (
            (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
            if values.size
            else (None, None)
        )
        output.append(
            {
                "analysis": analysis,
                "model_family": family,
                "outcome": "miss_80",
                "term": term,
                "estimate_pp": float(observed[index]) * 100.0,
                "ci95_low_pp": None if low is None else low * 100.0,
                "ci95_high_pp": None if high is None else high * 100.0,
                "observations": len(rows),
                "team_clusters": codes[4],
                "game_clusters": codes[5],
                "strength_center": strength_center,
                "uncertainty_center": uncertainty_center,
                "bootstrap_repetitions": repetitions,
                "bootstrap": "two-way pigeonhole: focal team ID and game ID",
            }
        )
    return output


def fit_conditional_models(
    team_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    fbs_fbs = [row for row in team_rows if row["forecast_population"] == "fbs_fbs"]
    for family in FAMILIES:
        rows = [row for row in fbs_fbs if row["model_family"] == family]
        if not rows:
            continue
        strength = np.asarray([float(row["preseason_strength"]) for row in rows])
        uncertainty = np.asarray(
            [float(row["preseason_rank_uncertainty_fraction"]) for row in rows]
        )
        strength_center = float(np.mean(strength))
        uncertainty_center = float(np.mean(uncertainty))
        strength_scaled = (strength - strength_center) * 10.0
        uncertainty_scaled = (uncertainty - uncertainty_center) * 10.0
        progress = np.asarray([str(row["progression_group"]) for row in rows])
        columns = [np.ones(len(rows))]
        names = ["intercept"]
        for season in SEASONS[1:]:
            columns.append(
                np.asarray([int(row["season"]) == season for row in rows], dtype=float)
            )
            names.append(f"season_{season}")
        for group in ("middle_3_6", "late_7_plus"):
            indicator = np.asarray([value == group for value in progress], dtype=float)
            columns.append(indicator)
            names.append(f"progression_{group}")
        columns.extend((strength_scaled, uncertainty_scaled))
        names.extend(("strength_per_0_1_early", "uncertainty_per_0_1_early"))
        for group in ("middle_3_6", "late_7_plus"):
            indicator = np.asarray([value == group for value in progress], dtype=float)
            columns.extend(
                (strength_scaled * indicator, uncertainty_scaled * indicator)
            )
            names.extend(
                (
                    f"strength_per_0_1_x_{group}",
                    f"uncertainty_per_0_1_x_{group}",
                )
            )
        output.extend(
            _linear_probability_model(
                rows,
                columns,
                names,
                analysis="preseason_expectations",
                family=family,
                strength_center=strength_center,
                uncertainty_center=uncertainty_center,
                repetitions=BOOTSTRAP_REPETITIONS,
                seed_label=f"preseason:{family}",
            )
        )

        surprise_members = [row for row in rows if row["primary_eligible"]]
        if not surprise_members:
            continue
        surprise = np.asarray(
            [float(row["accumulated_surprise"]) for row in surprise_members]
        )
        surprise_abs = np.abs(surprise)
        strength_s = np.asarray(
            [float(row["preseason_strength"]) for row in surprise_members]
        )
        uncertainty_s = np.asarray(
            [
                float(row["preseason_rank_uncertainty_fraction"])
                for row in surprise_members
            ]
        )
        strength_center_s = float(np.mean(strength_s))
        uncertainty_center_s = float(np.mean(uncertainty_s))
        strength_scaled_s = (strength_s - strength_center_s) * 10.0
        uncertainty_scaled_s = (uncertainty_s - uncertainty_center_s) * 10.0
        signed_scaled = (surprise - float(np.mean(surprise))) * 2.0
        absolute_scaled = (surprise_abs - float(np.mean(surprise_abs))) * 2.0
        negative_strength_interaction = (
            strength_scaled_s * np.maximum(-surprise, 0.0) * 2.0
        )
        prior_count = np.log1p(
            np.asarray(
                [int(row["prior_game_count"]) for row in surprise_members], dtype=float
            )
        )
        prior_count -= float(np.mean(prior_count))
        surprise_columns = [np.ones(len(surprise_members))]
        surprise_names = ["intercept"]
        for season in SEASONS[1:]:
            surprise_columns.append(
                np.asarray(
                    [int(row["season"]) == season for row in surprise_members],
                    dtype=float,
                )
            )
            surprise_names.append(f"season_{season}")
        surprise_columns.extend(
            (
                prior_count,
                strength_scaled_s,
                uncertainty_scaled_s,
                signed_scaled,
                absolute_scaled,
                negative_strength_interaction,
            )
        )
        surprise_names.extend(
            (
                "log1p_prior_games",
                "strength_per_0_1",
                "uncertainty_per_0_1",
                "signed_surprise_per_0_5",
                "absolute_surprise_per_0_5",
                "negative_surprise_magnitude_x_strength_per_0_5_by_0_1",
            )
        )
        output.extend(
            _linear_probability_model(
                surprise_members,
                surprise_columns,
                surprise_names,
                analysis="accumulated_surprise_continuous",
                family=family,
                strength_center=strength_center_s,
                uncertainty_center=uncertainty_center_s,
                repetitions=BOOTSTRAP_REPETITIONS,
                seed_label=f"surprise:{family}",
            )
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_gzip_csv(
    path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped,
        io.TextIOWrapper(zipped, encoding="utf-8", newline="") as handle,
    ):
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _format_number(value: object, digits: int = 2) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.{digits}f}"


def _summary_lookup(
    rows: list[dict[str, Any]],
    *,
    analysis: str,
    family: str,
    population: str = "fbs_fbs",
    scope: str = "pooled",
    stage: str = "rolling",
    condition: str = "",
    condition_value: str = "",
    season: str = "",
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["analysis"] == analysis
        and row["model_family"] == family
        and row["population"] == population
        and row["scope"] == scope
        and row["forecast_stage"] == stage
        and row["condition"] == condition
        and row["condition_value"] == condition_value
        and str(row["season"]) == season
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one summary for {analysis}/{family}/{population}/{scope}/"
            f"{condition}={condition_value}/{season}; found {len(matches)}"
        )
    return matches[0]


def classify_findings(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = {
        family: _summary_lookup(
            summary_rows, analysis="overall", family=family, scope="pooled"
        )
        for family in FAMILIES
    }
    global_flags: list[str] = []
    global_band_pass = True
    for family, row in overall.items():
        error = float(row["coverage_error_80_pp"])
        low = float(row["coverage_error_80_ci95_low_pp"])
        high = float(row["coverage_error_80_ci95_high_pp"])
        global_band_pass &= low >= -5.0 and high <= 5.0
        seasons = [
            _summary_lookup(
                summary_rows,
                analysis="overall",
                family=family,
                scope="season",
                season=str(season),
            )
            for season in SEASONS
        ]
        same_direction = sum(
            1
            for season_row in seasons
            if float(season_row["coverage_error_80_pp"]) == 0.0
            or math.copysign(1.0, float(season_row["coverage_error_80_pp"]))
            == math.copysign(1.0, error)
        )
        if abs(error) >= 5.0 and (low > 0.0 or high < 0.0) and same_direction >= 3:
            global_flags.append(family)

    local_analyses = {
        "season_progression",
        "preseason_strength",
        "accumulated_surprise",
    }
    local_rows = [
        row
        for row in summary_rows
        if row["analysis"] in local_analyses
        and row["scope"] == "pooled"
        and int(row["unique_games"]) >= 100
    ]
    local_candidates: list[dict[str, Any]] = []
    local_precision_pass = True
    for row in local_rows:
        error = float(row["coverage_error_80_pp"])
        low = float(row["coverage_error_80_ci95_low_pp"])
        high = float(row["coverage_error_80_ci95_high_pp"])
        if not (-10.0 <= low and high <= 10.0):
            local_precision_pass = False
        season_same_direction = 0
        for season in SEASONS:
            season_matches = [
                candidate
                for candidate in summary_rows
                if candidate["analysis"] == row["analysis"]
                and candidate["model_family"] == row["model_family"]
                and candidate["population"] == row["population"]
                and candidate["scope"] == "season"
                and candidate["condition"] == row["condition"]
                and candidate["condition_value"] == row["condition_value"]
                and str(candidate["season"]) == str(season)
            ]
            if season_matches and math.copysign(
                1.0, float(season_matches[0]["coverage_error_80_pp"])
            ) == math.copysign(1.0, error):
                season_same_direction += 1
        if abs(error) >= 10.0 and (low > 0.0 or high < 0.0):
            local_candidates.append(
                {
                    "analysis": str(row["analysis"]),
                    "condition_value": str(row["condition_value"]),
                    "family": str(row["model_family"]),
                    "direction": 1 if error > 0.0 else -1,
                    "seasons_same_direction": season_same_direction,
                }
            )
    local_flags = [
        candidate
        for candidate in local_candidates
        if candidate["seasons_same_direction"] >= 3
    ]
    for analysis in local_analyses:
        values = {
            candidate["condition_value"]
            for candidate in local_candidates
            if candidate["analysis"] == analysis
        }
        for value in values:
            matches = [
                candidate
                for candidate in local_candidates
                if candidate["analysis"] == analysis
                and candidate["condition_value"] == value
            ]
            families = {candidate["family"] for candidate in matches}
            directions = {candidate["direction"] for candidate in matches}
            if set(FAMILIES).issubset(families) and len(directions) == 1:
                local_flags.extend(
                    candidate for candidate in matches if candidate not in local_flags
                )
    local_flags.sort(
        key=lambda item: (item["analysis"], item["condition_value"], item["family"])
    )
    paired_local = {
        (candidate["analysis"], candidate["condition_value"])
        for candidate in local_flags
        if any(
            other["analysis"] == candidate["analysis"]
            and other["condition_value"] == candidate["condition_value"]
            and other["family"] != candidate["family"]
            for other in local_flags
        )
    }
    if global_flags or local_flags:
        recommendation = "A"
        label = "Potential meaningful predictive miscalibration identified"
    elif global_band_pass and local_precision_pass:
        recommendation = "B"
        label = "No meaningful deficiency detected at the resolution examined"
    else:
        recommendation = "C"
        label = "Inconclusive evidence"
    return {
        "recommendation": recommendation,
        "label": label,
        "global_practical_band_pp": [-5.0, 5.0],
        "global_band_pass_both_families": bool(global_band_pass),
        "global_signal_families": global_flags,
        "local_signal_groups": [
            [item["analysis"], item["condition_value"], item["family"]]
            for item in local_flags
        ],
        "local_signals_replicated_across_families": [
            list(item) for item in sorted(paired_local)
        ],
        "adequately_sized_local_intervals_within_10pp": bool(local_precision_pass),
    }


def render_report(
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    counts: dict[str, Any],
    decision: dict[str, Any],
    strength_reference: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# Issue 256: conditional in-season predictive calibration",
        "",
        "## Finding",
        "",
        f"**Recommendation under #244: {decision['recommendation']} — {decision['label']}.**",
        "",
        "This study evaluates the retained #201 game-margin predictive distributions for the existing-policy History and Context families. It does not directly validate the #253 rank-PMF uncertainty trajectories, which use a different retained configuration. The analysis is historical and exploratory.",
        "",
        "The primary population contains one rolling forecast per FBS–FBS regular-season game, family, and #201 cutoff. Preseason forecasts are shown separately as a reference. FBS–FCS results are separate. The retained #201 interval indicators passed row-level recalculation parity.",
        "",
        "## Primary FBS–FBS rolling calibration",
        "",
        "Coverage error is observed minus nominal coverage in percentage points. Intervals are 95% two-way team/game pigeonhole bootstrap percentile intervals.",
        "",
        "| Family | Games | 80% coverage [95% interval] | Error (pp) | Upper / lower misses | Mean / median 80% width | CRPS | Margin NLL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in FAMILIES:
        row = _summary_lookup(summary_rows, analysis="overall", family=family)
        lines.append(
            "| {family} | {n} | {coverage:.1%} [{low:.1%}, {high:.1%}] | {error:+.2f} | {upper:.1%} / {lower:.1%} | {mean:.2f} / {median:.2f} | {crps:.3f} | {nll:.3f} |".format(
                family=family,
                n=int(row["n_forecasts"]),
                coverage=float(row["observed_coverage_80"]),
                low=float(row["coverage_80_ci95_low"]),
                high=float(row["coverage_80_ci95_high"]),
                error=float(row["coverage_error_80_pp"]),
                upper=float(row["upper_miss_80_rate"]),
                lower=float(row["lower_miss_80_rate"]),
                mean=float(row["mean_width_80"]),
                median=float(row["median_width_80"]),
                crps=float(row["mean_crps"]),
                nll=float(row["mean_margin_nll"]),
            )
        )
    lines.extend(
        [
            "",
            "### Equal-season summary",
            "",
            "Each year contributes equally to these four-season estimates.",
            "",
            "| Family | 80% coverage [95% interval] | Error (pp) | Mean width | CRPS | Margin NLL |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for family in FAMILIES:
        row = _summary_lookup(
            summary_rows, analysis="overall", family=family, scope="equal_season"
        )
        lines.append(
            "| {family} | {coverage:.1%} [{low:.1%}, {high:.1%}] | {error:+.2f} | {width:.2f} | {crps:.3f} | {nll:.3f} |".format(
                family=family,
                coverage=float(row["observed_coverage_80"]),
                low=float(row["coverage_80_ci95_low"]),
                high=float(row["coverage_80_ci95_high"]),
                error=float(row["coverage_error_80_pp"]),
                width=float(row["mean_width_80"]),
                crps=float(row["mean_crps"]),
                nll=float(row["mean_margin_nll"]),
            )
        )
    lines.extend(
        [
            "",
            "### By season",
            "",
            "| Season | Family | Games | 80% coverage [95% interval] | Error (pp) | Mean width |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for season in SEASONS:
        for family in FAMILIES:
            row = _summary_lookup(
                summary_rows,
                analysis="overall",
                family=family,
                scope="season",
                season=str(season),
            )
            lines.append(
                "| {season} | {family} | {n} | {coverage:.1%} [{low:.1%}, {high:.1%}] | {error:+.2f} | {width:.2f} |".format(
                    season=season,
                    family=family,
                    n=int(row["n_forecasts"]),
                    coverage=float(row["observed_coverage_80"]),
                    low=float(row["coverage_80_ci95_low"]),
                    high=float(row["coverage_80_ci95_high"]),
                    error=float(row["coverage_error_80_pp"]),
                    width=float(row["mean_width_80"]),
                )
            )
    lines.extend(
        [
            "",
            "### Season progression and preseason strength",
            "",
            "The following are team-oriented conditional estimates. FBS–FBS games contribute both team perspectives; `unique games` exposes that shared-game dependence. Family-specific strength quartiles use distinct FBS team-seasons pooled across 2022–2025 and linear sample quantiles, before examining coverage; they are descriptive only.",
            "",
            "Strength reference counts and cut points (q25 / q50 / q75): "
            + "; ".join(
                f"{family}: {strength_reference[family]['team_seasons']} team-seasons, "
                f"{float(strength_reference[family]['q25']):.6f} / "
                f"{float(strength_reference[family]['q50']):.6f} / "
                f"{float(strength_reference[family]['q75']):.6f}"
                for family in FAMILIES
            )
            + ".",
            "",
            "| Analysis | Group | Family | Team-game rows | Unique games | 80% coverage [95% interval] | Error (pp) | Mean width |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for analysis, condition, groups in (
        ("season_progression", "progression_group", PROGRESSION_GROUPS),
        (
            "preseason_strength",
            "strength_quartile",
            ("q1_low", "q2_lower_middle", "q3_upper_middle", "q4_high"),
        ),
    ):
        for group in groups:
            for family in FAMILIES:
                row = _summary_lookup(
                    summary_rows,
                    analysis=analysis,
                    family=family,
                    condition=condition,
                    condition_value=group,
                )
                lines.append(
                    "| {analysis} | {group} | {family} | {n} | {games} | {coverage:.1%} [{low:.1%}, {high:.1%}] | {error:+.2f} | {width:.2f} |".format(
                        analysis=analysis,
                        group=group,
                        family=family,
                        n=int(row["n_forecasts"]),
                        games=int(row["unique_games"]),
                        coverage=float(row["observed_coverage_80"]),
                        low=float(row["coverage_80_ci95_low"]),
                        high=float(row["coverage_80_ci95_high"]),
                        error=float(row["coverage_error_80_pp"]),
                        width=float(row["mean_width_80"]),
                    )
                )
    lines.extend(
        [
            "",
            "## Accumulated surprise",
            "",
            "Only issue 247 observations with at least three evidence-eligible prior games enter this section. Surprise groups reuse the frozen ±0.5 thresholds. Counts are team-game perspectives; unique game counts show the shared outcomes.",
            "",
            "| Prior surprise | Family | Team-game rows | Unique games | 80% coverage [95% interval] | Error (pp) | Upper / lower misses | Mean width | CRPS | NLL |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group in SURPRISE_GROUPS:
        for family in FAMILIES:
            row = _summary_lookup(
                summary_rows,
                analysis="accumulated_surprise",
                family=family,
                condition="surprise_group",
                condition_value=group,
            )
            lines.append(
                "| {group} | {family} | {n} | {games} | {coverage:.1%} [{low:.1%}, {high:.1%}] | {error:+.2f} | {upper:.1%} / {lower:.1%} | {width:.2f} | {crps:.3f} | {nll:.3f} |".format(
                    group=group,
                    family=family,
                    n=int(row["n_forecasts"]),
                    games=int(row["unique_games"]),
                    coverage=float(row["observed_coverage_80"]),
                    low=float(row["coverage_80_ci95_low"]),
                    high=float(row["coverage_80_ci95_high"]),
                    error=float(row["coverage_error_80_pp"]),
                    upper=float(row["upper_miss_80_rate"]),
                    lower=float(row["lower_miss_80_rate"]),
                    width=float(row["mean_width_80"]),
                    crps=float(row["mean_crps"]),
                    nll=float(row["mean_margin_nll"]),
                )
            )
    lines.extend(
        [
            "",
            "### Prespecified continuous miss models",
            "",
            "Linear-probability coefficients are percentage-point changes in 80% interval miss probability. Preseason terms are per 0.1 change in normalized strength or uncertainty; surprise terms are per 0.5 surprise units. Interaction terms are adjusted for season and progression as specified in the frozen protocol. They are descriptive associations, not causal effects.",
            "",
            "| Analysis | Family | Term | Estimate (pp) [95% interval] | Team-game rows |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    focus_terms = {
        "strength_per_0_1_early",
        "uncertainty_per_0_1_early",
        "strength_per_0_1_x_middle_3_6",
        "strength_per_0_1_x_late_7_plus",
        "uncertainty_per_0_1_x_middle_3_6",
        "uncertainty_per_0_1_x_late_7_plus",
        "signed_surprise_per_0_5",
        "absolute_surprise_per_0_5",
        "negative_surprise_magnitude_x_strength_per_0_5_by_0_1",
    }
    for row in model_rows:
        if row["term"] not in focus_terms:
            continue
        lines.append(
            "| {analysis} | {family} | {term} | {estimate:+.2f} [{low:+.2f}, {high:+.2f}] | {n} |".format(
                analysis=row["analysis"],
                family=row["model_family"],
                term=row["term"],
                estimate=float(row["estimate_pp"]),
                low=float(row["ci95_low_pp"]),
                high=float(row["ci95_high_pp"]),
                n=int(row["observations"]),
            )
        )
    lines.extend(
        [
            "",
            "## Paired History–Context comparison",
            "",
            "Differences are History minus Context on matched FBS–FBS games and compatible #201 cutoffs. Negative CRPS or NLL differences favor History on that score; coverage differences are descriptive. Intervals use the same two-way dependence-aware bootstrap.",
            "",
            "| Metric | Matched games | Mean difference [95% interval] |",
            "| --- | ---: | ---: |",
        ]
    )
    for metric in (
        "coverage_80_difference_pp",
        "width_difference",
        "crps_difference",
        "margin_nll_difference",
    ):
        matches = [
            row
            for row in paired_rows
            if row["analysis"] == "overall"
            and row["scope"] == "pooled"
            and row["metric"] == metric
        ]
        if len(matches) != 1:
            raise ValueError(f"missing paired summary for {metric}")
        row = matches[0]
        lines.append(
            "| {metric} ({unit}) | {n} | {mean:+.3f} [{low:+.3f}, {high:+.3f}] |".format(
                metric=metric,
                unit=row["difference_unit"],
                n=int(row["paired_observations"]),
                mean=float(row["mean_difference"]),
                low=float(row["ci95_low"]),
                high=float(row["ci95_high"]),
            )
        )
    lines.extend(
        [
            "",
            "## Separate reference results",
            "",
            "| Population / stage | Family | Forecasts | 80% coverage | Mean width | CRPS | Margin NLL |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for population, stage, analysis, label in (
        ("fbs_fcs", "rolling", "overall", "FBS–FCS rolling"),
        ("fbs_fbs", "preseason", "preseason_reference", "FBS–FBS preseason reference"),
        ("fbs_fcs", "preseason", "preseason_reference", "FBS–FCS preseason reference"),
    ):
        for family in FAMILIES:
            row = _summary_lookup(
                summary_rows,
                analysis=analysis,
                family=family,
                population=population,
                scope="pooled",
                stage=stage,
            )
            lines.append(
                "| {label} | {family} | {n} | {coverage:.1%} | {width:.2f} | {crps:.3f} | {nll:.3f} |".format(
                    label=label,
                    family=family,
                    n=int(row["n_forecasts"]),
                    coverage=float(row["observed_coverage_80"]),
                    width=float(row["mean_width_80"]),
                    crps=float(row["mean_crps"]),
                    nll=float(row["mean_margin_nll"]),
                )
            )
    lines.extend(
        [
            "",
            "## Decision and limitations",
            "",
            f"The frozen decision screen returned **{decision['recommendation']}**. Both families' pooled FBS–FBS coverage intervals fall within the ±5 percentage-point global band, and no prespecified subgroup shows a ≥10-point error with the required consistency. However, the sustained-negative and sustained-positive surprise groups have fewer than 250 unique games per family, and their intervals are too wide to rule out a meaningful localized deviation. The continuous miss models also remain imprecise. The four exploratory seasons are not independent replications.",
            "",
            "This supports an inconclusive Phase 1 result rather than a mechanism investigation: do not initiate Phase 2 from these data alone, and do not close #244 as calibrated. If the sustained-surprise question remains important, the specific next evidence would be prospective out-of-time calibration with enough repeated-surprise team-games to narrow those intervals; it does not justify prior, likelihood, or inference changes now.",
            "",
            "CRPS and NLL provide scoring context; wider intervals can improve coverage without improving sharpness.",
            "",
            "No calibration result alone identifies a cause. This study does not attribute deviations to preseason priors, game likelihood, posterior inference, or latent-rank assumptions. Do not change production behavior from these historical results.",
            "",
            "## Reproducibility",
            "",
            "Run `uv run python -m gippyrank.research.issue_256_predictive_calibration` from the repository root. The runner verifies issue 201/247 input and evidence hashes, interval parity, exact cutoff joins, prior-game IDs, prior surprise means, and target-game exclusion. It uses the retained #201 scores and does not rebuild predictive mixtures.",
            "",
            f"Counts: {json.dumps(counts, sort_keys=True)}",
            "",
            "See `manifest.json` for input/output hashes, pairing audits, bootstrap configuration, and row counts.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_analysis(
    repository_root: Path, output_dir: Path | None = None
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    output_dir = output_dir or (
        repository_root / "data" / "research" / "issue_256_predictive_calibration"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.md"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol_hash = sha256_file(protocol_path)
    if protocol_hash != PROTOCOL_SHA256:
        raise ValueError("issue 256 frozen protocol hash mismatch")
    lineage = verify_inputs(repository_root)
    scores = _read_csv(lineage["paths"]["issue201/forecast_scores.csv"])
    retained_score_summary = _read_csv(
        lineage["paths"]["issue201/forecast_summary.csv"]
    )
    surprise_rows = _read_csv(lineage["paths"]["issue247/team_game_observations.csv"])
    team_error_rows = _read_csv(lineage["paths"]["issue247/team_game_errors.csv"])
    evidence_records = _read_evidence(
        lineage["paths"]["issue201/replay_evidence.jsonl.gz"]
    )
    forecast_rows, team_rows, audit = build_calibration_observations(
        scores, surprise_rows, team_error_rows, evidence_records
    )
    aggregate_parity = verify_issue201_summary_parity(
        forecast_rows, retained_score_summary
    )
    audit["issue201_aggregate_parity"] = aggregate_parity
    strength_reference = strength_quartile_reference(team_rows)
    summary_rows = build_calibration_summaries(
        forecast_rows, team_rows, strength_reference
    )
    paired_rows, pairing_audit = build_paired_comparisons(forecast_rows, team_rows)
    model_rows = fit_conditional_models(team_rows)
    counts = {
        **audit,
        "forecast_rows_by_stage_population_family": {
            f"{stage}/{population}/{family}": sum(
                1
                for row in forecast_rows
                if row["forecast_stage"] == stage
                and row["forecast_population"] == population
                and row["model_family"] == family
            )
            for stage in ("rolling", "preseason")
            for population in POPULATIONS
            for family in FAMILIES
        },
        "team_rows_by_population_family": {
            f"{population}/{family}": sum(
                1
                for row in team_rows
                if row["forecast_population"] == population
                and row["model_family"] == family
            )
            for population in POPULATIONS
            for family in FAMILIES
        },
        "surprise_eligible_team_rows_by_family": {
            family: sum(
                1
                for row in team_rows
                if row["forecast_population"] == "fbs_fbs"
                and row["model_family"] == family
                and row["primary_eligible"]
            )
            for family in FAMILIES
        },
        "pairing": pairing_audit,
    }
    decision = classify_findings(summary_rows)
    _write_gzip_csv(
        output_dir / "forecast_observations.csv.gz", forecast_rows, FORECAST_FIELDS
    )
    _write_gzip_csv(
        output_dir / "team_calibration_observations.csv.gz", team_rows, TEAM_FIELDS
    )
    _write_csv(output_dir / "calibration_summary.csv", summary_rows, SUMMARY_FIELDS)
    _write_csv(output_dir / "paired_comparison.csv", paired_rows, PAIRED_FIELDS)
    _write_csv(output_dir / "conditional_models.csv", model_rows, MODEL_FIELDS)
    _write_json(output_dir / "decision.json", decision)
    report = render_report(
        summary_rows, paired_rows, model_rows, counts, decision, strength_reference
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    output_names = (
        "protocol.md",
        "forecast_observations.csv.gz",
        "team_calibration_observations.csv.gz",
        "calibration_summary.csv",
        "paired_comparison.csv",
        "conditional_models.csv",
        "decision.json",
        "report.md",
    )
    manifest = {
        "schema_version": 1,
        "experiment": "issue-256-conditional-in-season-predictive-calibration",
        "status": "complete",
        "protocol_sha256": protocol_hash,
        "seasons": list(SEASONS),
        "model_arms": list(MODEL_ARMS.values()),
        "primary_population": "rolling FBS–FBS regular-season games, counted once per game and arm",
        "secondary_population": "rolling FBS–FCS forecasts, one FBS team perspective per game",
        "forecast_lineage": "issue 201 existing-policy predictions; distinct from issue 253 rank-PMF configurations",
        "input_hashes": {
            name: {"bytes": path.stat().st_size, "sha256": lineage["hashes"][name]}
            for name, path in lineage["paths"].items()
        },
        "counts": counts,
        "strength_quartile_reference": strength_reference,
        "bootstrap": {
            "repetitions": BOOTSTRAP_REPETITIONS,
            "seed": BOOTSTRAP_SEED,
            "interval": "95% percentile",
            "game_level": "two-way pigeonhole, endpoint team IDs and game ID; average team endpoint multiplicity keeps one game contribution",
            "team_level": "two-way pigeonhole, focal team ID and game ID",
        },
        "decision": decision,
        "outputs": {
            name: {
                "bytes": (output_dir / name).stat().st_size,
                "sha256": sha256_file(output_dir / name),
            }
            for name in output_names
        },
        "manifest_self_hash": "omitted to avoid a self-referential digest",
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest = run_analysis(args.root, args.output_dir)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "counts": manifest["counts"],
                "decision": manifest["decision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
