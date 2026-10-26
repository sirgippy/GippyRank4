"""Leakage-safe sequential game-error analysis for issue 246."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

SEASONS = (2022, 2023, 2024, 2025)
ARMS = (
    ("History_existing", "History", "existing"),
    ("Context_existing", "Context", "existing"),
    ("History_pre_bowl", "History", "pre_bowl"),
    ("Context_pre_bowl", "Context", "pre_bowl"),
)
PRIMARY_POPULATION = "fbs_fbs"
ALLOWED_POPULATIONS = {"fbs_fbs", "fbs_fcs"}
STRONG_PRESEASON_TOP10_PROBABILITY = 0.50
BOOTSTRAP_REPETITIONS = 1000
BOOTSTRAP_SEED = 246

PRESEASON_PRIOR_FIELDS = (
    "season",
    "model_arm",
    "prior_family",
    "policy",
    "team_id",
    "team",
    "preseason_rank_top10_probability",
    "preseason_rank_prior_group",
    "preseason_rank_pmf_sha256",
)

TEAM_GAME_FIELDS = (
    "season",
    "model_arm",
    "prior_family",
    "policy",
    "team_id",
    "team",
    "opponent_id",
    "opponent",
    "game_id",
    "forecast_id",
    "kickoff_utc",
    "cutoff_utc",
    "is_home",
    "forecast_population",
    "team_game_number",
    "actual_margin",
    "predicted_margin",
    "signed_error",
    "preseason_rank_top10_probability",
    "preseason_rank_prior_group",
    "evidence_game_count",
)
LAG_PAIR_FIELDS = (
    "pair_id",
    "season",
    "model_arm",
    "prior_family",
    "policy",
    "team_id",
    "team",
    "previous_game_id",
    "next_game_id",
    "previous_kickoff_utc",
    "next_kickoff_utc",
    "days_between_games",
    "previous_team_game_number",
    "next_team_game_number",
    "season_progress_bin",
    "previous_actual_margin",
    "previous_predicted_margin",
    "previous_signed_error",
    "next_actual_margin",
    "next_predicted_margin",
    "next_signed_error",
    "preseason_rank_top10_probability",
    "preseason_rank_prior_group",
    "previous_game_in_next_forecast_evidence",
    "eligible",
    "exclusion_reason",
    "same_error_direction",
)
SUMMARY_FIELDS = (
    "summary_level",
    "season",
    "model_arm",
    "prior_family",
    "policy",
    "preseason_rank_prior_group",
    "season_progress_bin",
    "transitions",
    "team_count",
    "unique_matchups",
    "lag1_correlation",
    "correlation_ci95_low",
    "correlation_ci95_high",
    "directional_persistence",
    "directional_ci95_low",
    "directional_ci95_high",
    "nonzero_directional_pairs",
    "positive_previous_error_share",
    "positive_next_error_share",
    "directional_independence_baseline",
    "directional_baseline_ci95_low",
    "directional_baseline_ci95_high",
    "directional_excess_over_independence",
    "directional_excess_ci95_low",
    "directional_excess_ci95_high",
)
CONTRAST_FIELDS = (
    "model_arm",
    "prior_family",
    "policy",
    "contrast",
    "group_a",
    "group_b",
    "metric",
    "difference_a_minus_b",
    "ci95_low",
    "ci95_high",
    "transitions_a",
    "transitions_b",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _float(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if value in (None, ""):
        raise ValueError(f"missing numeric field {field!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric field {field!r}: {value!r}")
    return number


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timestamp must include a timezone: {value!r}")
    return parsed


def _evidence_key(
    season: int | str, cutoff_utc: str, policy: str
) -> tuple[int, str, str]:
    return int(season), str(cutoff_utc), str(policy)


def load_issue201_inputs(issue201_root: Path) -> dict[str, Any]:
    """Load immutable #201 outputs and verify their recorded output hashes."""
    manifest_path = issue201_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "issue-201-prebowl-forecast-comparison":
        raise ValueError(f"unexpected issue 201 manifest: {manifest_path}")

    required = (
        "forecast_scores.csv",
        "forecast_summary.csv",
        "replay_evidence.jsonl.gz",
    )
    input_hashes: dict[str, dict[str, Any]] = {
        "manifest.json": {
            "sha256": sha256_file(manifest_path),
            "bytes": manifest_path.stat().st_size,
        }
    }
    for name in required:
        path = issue201_root / name
        expected = manifest.get("outputs", {}).get(name)
        if not isinstance(expected, dict):
            raise TypeError(f"issue 201 manifest has no output record for {name}")
        actual_hash = sha256_file(path)
        actual_bytes = path.stat().st_size
        if actual_hash != expected.get("sha256") or actual_bytes != expected.get(
            "bytes"
        ):
            raise ValueError(f"issue 201 artifact hash mismatch: {path}")
        input_hashes[name] = {"sha256": actual_hash, "bytes": actual_bytes}

    with (issue201_root / "forecast_scores.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        all_scores = list(csv.DictReader(handle))
    rolling = [
        row
        for row in all_scores
        if row["forecast_stage"] == "rolling"
        and row["forecast_population"] in ALLOWED_POPULATIONS
    ]
    if not rolling:
        raise ValueError("issue 201 forecast scores lack rolling rows")

    evidence_records: dict[tuple[int, str, str], dict[str, Any]] = {}
    with gzip.open(
        issue201_root / "replay_evidence.jsonl.gz", "rt", encoding="utf-8"
    ) as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            key = _evidence_key(
                record["season"], record["cutoff_utc"], record["policy"]
            )
            if key in evidence_records:
                raise ValueError(f"duplicate issue 201 replay evidence key: {key}")
            included_ids = [str(game_id) for game_id in record["included_game_ids"]]
            if len(included_ids) != int(record["included_game_count"]):
                raise ValueError(f"evidence count mismatch at line {line_number}")
            if len(set(included_ids)) != len(included_ids):
                raise ValueError(f"duplicate included game ID at line {line_number}")
            if _sha256_json(included_ids) != record["included_game_ids_sha256"]:
                raise ValueError(f"evidence ID digest mismatch at line {line_number}")
            record["included_game_id_set"] = set(included_ids)
            evidence_records[key] = record

    parity_checks = _verify_issue201_parity(
        rolling, issue201_root / "forecast_summary.csv"
    )
    return {
        "manifest": manifest,
        "rolling_scores": rolling,
        "evidence_records": evidence_records,
        "input_hashes": input_hashes,
        "parity_checks": parity_checks,
        "all_score_row_count": len(all_scores),
    }


def load_issue200_preseason_rank_priors(
    issue200_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load and verify the retained #200 preseason team-rank PMFs."""
    manifest_path = issue200_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_kind") != "issue_200_matched_pipeline_bundle":
        raise ValueError(f"unexpected issue 200 manifest: {manifest_path}")
    bundle_path = issue200_root / "pipeline_bundle.json"
    expected = manifest.get("outputs", {}).get("pipeline_bundle.json")
    if not isinstance(expected, dict):
        raise TypeError("issue 200 manifest has no pipeline_bundle.json output record")
    bundle_hash = sha256_file(bundle_path)
    bundle_bytes = bundle_path.stat().st_size
    if bundle_hash != expected.get("sha256") or bundle_bytes != expected.get(
        "bytes"
    ):
        raise ValueError(f"issue 200 pipeline bundle hash mismatch: {bundle_path}")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("experiment") != "issue-200-matched-history-context":
        raise ValueError(f"unexpected issue 200 pipeline bundle: {bundle_path}")

    input_hashes = {
        "issue200/manifest.json": {
            "sha256": sha256_file(manifest_path),
            "bytes": manifest_path.stat().st_size,
        },
        "issue200/pipeline_bundle.json": {
            "sha256": bundle_hash,
            "bytes": bundle_bytes,
        },
    }
    arm_by_key = {
        (family.casefold(), policy): (arm, family, policy)
        for arm, family, policy in ARMS
    }
    expected_populations = {
        int(season): int(population)
        for season, population in manifest["fbs_target_population_by_season"].items()
    }
    classifications: dict[tuple[int, str, str], dict[str, Any]] = {}
    for prior in bundle.get("priors", []):
        season = int(prior["season"])
        model_family = str(prior["model_family"])
        policy = str(prior["policy"])
        arm_spec = arm_by_key.get((model_family.casefold(), policy))
        if (
            season not in SEASONS
            or arm_spec is None
            or str(prior.get("subdivision", "")).casefold() != "fbs"
        ):
            continue
        arm, family, policy = arm_spec
        pmf = [float(value) for value in prior["pmf"]]
        if _sha256_json(pmf) != prior.get("pmf_sha256"):
            raise ValueError(
                f"issue 200 preseason PMF hash mismatch for {season}/{arm}/{prior['team_id']}"
            )
        expected_population = expected_populations[season]
        if int(prior["population"]) != expected_population or len(pmf) != expected_population:
            raise ValueError(
                f"issue 200 preseason PMF support mismatch for {season}/{arm}/{prior['team_id']}"
            )
        if any(not math.isfinite(value) or value < 0 for value in pmf):
            raise ValueError(
                f"issue 200 preseason PMF has invalid probabilities for {season}/{arm}/{prior['team_id']}"
            )
        if not math.isclose(math.fsum(pmf), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"issue 200 preseason PMF does not sum to one for {season}/{arm}/{prior['team_id']}"
            )
        p_top10 = top10_rank_probability(pmf)
        rank_summary = rank_pmf_summary(pmf)
        key = (season, arm, str(prior["team_id"]))
        if key in classifications:
            raise ValueError(f"duplicate issue 200 preseason prior: {key}")
        classifications[key] = {
            "season": season,
            "model_arm": arm,
            "prior_family": family,
            "policy": policy,
            "team_id": str(prior["team_id"]),
            "team": str(prior["team_name"]),
            "preseason_rank_top10_probability": p_top10,
            "preseason_rank_prior_group": preseason_rank_prior_group(p_top10),
            **rank_summary,
            "preseason_rank_pmf_sha256": str(prior["pmf_sha256"]),
        }

    for arm, _, _ in ARMS:
        for season, expected_population in expected_populations.items():
            observed = sum(
                1
                for row in classifications.values()
                if row["season"] == season and row["model_arm"] == arm
            )
            if observed != expected_population:
                raise ValueError(
                    f"issue 200 prior coverage mismatch for {season}/{arm}: "
                    f"{observed} != {expected_population}"
                )
    return sorted(
        classifications.values(),
        key=lambda row: (row["season"], row["model_arm"], row["team_id"]),
    ), input_hashes


def _verify_issue201_parity(
    rolling_scores: list[dict[str, str]], summary_path: Path
) -> list[dict[str, Any]]:
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    summary_by_key = {
        (row["model_arm"], int(row["season"])): row
        for row in summary_rows
        if row["forecast_stage"] == "rolling"
        and row["forecast_population"] == PRIMARY_POPULATION
        and row["phase"] == "all_phases"
        and row["season"].isdigit()
    }
    checks: list[dict[str, Any]] = []
    for arm, _, _ in ARMS:
        for season in SEASONS:
            rows = [
                row
                for row in rolling_scores
                if row["model_arm"] == arm
                and int(row["season"]) == season
                and row["forecast_population"] == PRIMARY_POPULATION
            ]
            expected = summary_by_key.get((arm, season))
            if expected is None or int(expected["games"]) != len(rows):
                raise ValueError(
                    f"#201 primary game-count parity failed for {arm}/{season}"
                )
            differences: dict[str, float] = {}
            for metric in ("crps", "margin_nll"):
                observed = math.fsum(_float(row, metric) for row in rows) / len(rows)
                recorded = float(expected[metric])
                difference = abs(observed - recorded)
                if difference > 1.0e-12:
                    raise ValueError(
                        f"#201 {metric} parity failed for {arm}/{season}: {difference}"
                    )
                differences[metric] = difference
            checks.append(
                {
                    "model_arm": arm,
                    "season": season,
                    "games": len(rows),
                    "max_metric_absolute_difference": max(differences.values()),
                }
            )
    return checks


def signed_error_from_home_margin(
    actual_home_margin: float, predicted_home_margin: float, *, is_home: bool
) -> float:
    """Return observed-minus-predicted margin from the focal team's perspective."""
    orientation = 1.0 if is_home else -1.0
    return orientation * (actual_home_margin - predicted_home_margin)


def preseason_rank_prior_group(top10_probability: float) -> str:
    return (
        "strong_preseason_rank_prior"
        if top10_probability >= STRONG_PRESEASON_TOP10_PROBABILITY
        else "other"
    )


def top10_rank_probability(pmf: list[float]) -> float:
    """Return the preseason PMF mass on ranks 1 through 10."""
    if len(pmf) < 10:
        raise ValueError("preseason rank PMF must include at least ten ranks")
    return math.fsum(float(value) for value in pmf[:10])


def rank_pmf_summary(pmf: list[float]) -> dict[str, float | int]:
    """Summarize rank location and uncertainty on a normalized rank scale."""
    if len(pmf) < 2:
        raise ValueError("preseason rank PMF must include at least two ranks")
    probabilities = np.asarray(pmf, dtype=float)
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("preseason rank PMF must contain finite nonnegative values")
    if not math.isclose(
        math.fsum(float(value) for value in pmf), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("preseason rank PMF must sum to one")
    population = len(pmf)
    ranks = np.arange(1, population + 1, dtype=float)
    expected_rank = float(np.dot(ranks, probabilities))
    variance = float(np.dot((ranks - expected_rank) ** 2, probabilities))
    rank_sd = math.sqrt(max(variance, 0.0))
    return {
        "preseason_expected_rank": expected_rank,
        "preseason_rank_sd": rank_sd,
        "preseason_rank_strength_percentile": 1.0
        - (expected_rank - 1.0) / (population - 1.0),
        "preseason_rank_uncertainty_fraction": rank_sd / (population - 1.0),
        "preseason_rank_population": population,
    }


def _team_game_row(
    forecast: dict[str, str],
    preseason_prior: dict[str, Any],
    *,
    is_home: bool,
) -> dict[str, Any]:
    side = "home" if is_home else "away"
    other = "away" if is_home else "home"
    actual_home_margin = _float(forecast, "actual_home_margin")
    predicted_home_margin = _float(forecast, "expected_home_margin")
    actual_margin = actual_home_margin if is_home else -actual_home_margin
    predicted_margin = predicted_home_margin if is_home else -predicted_home_margin
    return {
        "season": int(forecast["season"]),
        "model_arm": forecast["model_arm"],
        "prior_family": forecast["prior_family"],
        "policy": forecast["policy"],
        "team_id": forecast[f"{side}_id"],
        "team": forecast[f"{side}_team"],
        "opponent_id": forecast[f"{other}_id"],
        "opponent": forecast[f"{other}_team"],
        "game_id": forecast["game_id"],
        "forecast_id": forecast["forecast_id"],
        "kickoff_utc": forecast["kickoff_utc"],
        "cutoff_utc": forecast["cutoff_utc"],
        "is_home": is_home,
        "forecast_population": forecast["forecast_population"],
        "actual_margin": actual_margin,
        "predicted_margin": predicted_margin,
        "signed_error": signed_error_from_home_margin(
            actual_home_margin, predicted_home_margin, is_home=is_home
        ),
        "preseason_rank_top10_probability": preseason_prior[
            "preseason_rank_top10_probability"
        ],
        "preseason_rank_prior_group": preseason_prior[
            "preseason_rank_prior_group"
        ],
        "evidence_game_count": int(forecast["evidence_game_count"]),
    }


def build_team_game_errors(
    rolling_scores: list[dict[str, str]],
    preseason_rank_priors: list[dict[str, Any]],
    evidence_records: dict[tuple[int, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Orient each rolling forecast for its FBS team and number games in-season.

    FBS-FCS rows contribute only the FBS team's orientation. This preserves a
    complete FBS regular-season sequence without inventing FCS-vs-FCS history.
    """
    preseason_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in preseason_rank_priors:
        key = (int(row["season"]), row["model_arm"], str(row["team_id"]))
        if key in preseason_by_key:
            raise ValueError(f"duplicate preseason rank prior classification: {key}")
        preseason_by_key[key] = row

    team_rows: list[dict[str, Any]] = []
    outcome_by_game: dict[tuple[int, str], tuple[str, ...]] = {}
    rolling_keys: set[tuple[int, str, str]] = set()
    for row in rolling_scores:
        season = int(row["season"])
        arm = row["model_arm"]
        game_id = row["game_id"]
        policy = row["policy"]
        key = (season, arm, game_id)
        if key in rolling_keys:
            raise ValueError(f"duplicate issue 201 rolling forecast row: {key}")
        rolling_keys.add(key)
        evidence_key = _evidence_key(season, row["cutoff_utc"], policy)
        evidence = evidence_records.get(evidence_key)
        if evidence is None:
            raise ValueError(f"missing #201 replay evidence for {evidence_key}")
        if row["evidence_ids_sha256"] != evidence["included_game_ids_sha256"]:
            raise ValueError(
                f"#201 evidence digest mismatch for forecast {row['forecast_id']}"
            )
        included_ids = evidence["included_game_id_set"]
        if game_id in included_ids:
            raise ValueError(f"forecast game is in its own evidence: {game_id}")
        if _parse_utc(row["cutoff_utc"]) >= _parse_utc(row["kickoff_utc"]):
            raise ValueError(f"forecast cutoff is not before game kickoff: {game_id}")

        outcome_key = (season, game_id)
        outcome = (
            row["actual_home_points"],
            row["actual_away_points"],
            row["home_id"],
            row["away_id"],
        )
        previous_outcome = outcome_by_game.get(outcome_key)
        if previous_outcome is not None and previous_outcome != outcome:
            raise ValueError(
                f"#201 arms disagree on game outcome or teams: {outcome_key}"
            )
        outcome_by_game[outcome_key] = outcome

        home_fbs = row["home_subdivision"].casefold() == "fbs"
        away_fbs = row["away_subdivision"].casefold() == "fbs"
        if row["forecast_population"] == "fbs_fbs" and home_fbs and away_fbs:
            home_prior = preseason_by_key.get((season, arm, row["home_id"]))
            away_prior = preseason_by_key.get((season, arm, row["away_id"]))
            if home_prior is None or away_prior is None:
                raise ValueError(f"missing #200 preseason rank prior for {key}")
            team_rows.extend(
                (
                    _team_game_row(row, home_prior, is_home=True),
                    _team_game_row(row, away_prior, is_home=False),
                )
            )
        elif row["forecast_population"] == "fbs_fcs" and home_fbs != away_fbs:
            fbs_id = row["home_id"] if home_fbs else row["away_id"]
            fbs_prior = preseason_by_key.get((season, arm, fbs_id))
            if fbs_prior is None:
                raise ValueError(f"missing #200 preseason rank prior for {key}")
            team_rows.append(_team_game_row(row, fbs_prior, is_home=home_fbs))
        else:
            raise ValueError(
                f"unexpected population/subdivision combination for {game_id}"
            )

    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in team_rows:
        grouped[(row["season"], row["model_arm"], row["team_id"])].append(row)
    for members in grouped.values():
        members.sort(key=lambda row: (_parse_utc(row["kickoff_utc"]), row["game_id"]))
        if len({row["game_id"] for row in members}) != len(members):
            raise ValueError(
                "a team appears more than once in a team-season game sequence"
            )
        for game_number, row in enumerate(members, start=1):
            row["team_game_number"] = game_number
    return sorted(
        team_rows,
        key=lambda row: (
            row["season"],
            row["model_arm"],
            row["team_id"],
            row["team_game_number"],
        ),
    )


def season_progress_bin(team_game_number: int) -> str:
    """Bin the destination team's game number, not the previous game's number."""
    if team_game_number <= 4:
        return "early_games_2_4"
    if team_game_number <= 8:
        return "middle_games_5_8"
    return "late_game_9_plus"


def build_lagged_pairs(
    team_game_rows: list[dict[str, Any]],
    evidence_records: dict[tuple[int, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build adjacent FBS team-game error pairs and enforce evidence inclusion."""
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in team_game_rows:
        grouped[(row["season"], row["model_arm"], row["team_id"])].append(row)

    pairs: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items()):
        members.sort(key=lambda row: row["team_game_number"])
        for previous, current in pairwise(members):
            evidence_key = _evidence_key(
                current["season"], current["cutoff_utc"], current["policy"]
            )
            evidence = evidence_records.get(evidence_key)
            if evidence is None:
                raise ValueError(f"missing next-game evidence for {evidence_key}")
            prior_included = previous["game_id"] in evidence["included_game_id_set"]
            eligible = prior_included
            exclusion_reason = (
                "" if eligible else "previous_game_absent_from_next_forecast_evidence"
            )
            previous_error = float(previous["signed_error"])
            next_error = float(current["signed_error"])
            if previous_error == 0.0 or next_error == 0.0:
                same_direction: bool | None = None
            else:
                same_direction = (previous_error > 0.0) == (next_error > 0.0)
            pair_identity = "|".join(
                (
                    str(current["season"]),
                    str(current["model_arm"]),
                    str(current["team_id"]),
                    str(previous["game_id"]),
                    str(current["game_id"]),
                )
            )
            days_between = (
                _parse_utc(current["kickoff_utc"]) - _parse_utc(previous["kickoff_utc"])
            ).total_seconds() / 86400.0
            pairs.append(
                {
                    "pair_id": hashlib.sha256(
                        pair_identity.encode("utf-8")
                    ).hexdigest(),
                    "season": current["season"],
                    "model_arm": current["model_arm"],
                    "prior_family": current["prior_family"],
                    "policy": current["policy"],
                    "team_id": current["team_id"],
                    "team": current["team"],
                    "previous_game_id": previous["game_id"],
                    "next_game_id": current["game_id"],
                    "previous_kickoff_utc": previous["kickoff_utc"],
                    "next_kickoff_utc": current["kickoff_utc"],
                    "days_between_games": days_between,
                    "previous_team_game_number": previous["team_game_number"],
                    "next_team_game_number": current["team_game_number"],
                    "season_progress_bin": season_progress_bin(
                        int(current["team_game_number"])
                    ),
                    "previous_actual_margin": previous["actual_margin"],
                    "previous_predicted_margin": previous["predicted_margin"],
                    "previous_signed_error": previous_error,
                    "next_actual_margin": current["actual_margin"],
                    "next_predicted_margin": current["predicted_margin"],
                    "next_signed_error": next_error,
                    "preseason_rank_top10_probability": current[
                        "preseason_rank_top10_probability"
                    ],
                    "preseason_rank_prior_group": current[
                        "preseason_rank_prior_group"
                    ],
                    "previous_game_in_next_forecast_evidence": prior_included,
                    "eligible": eligible,
                    "exclusion_reason": exclusion_reason,
                    "same_error_direction": same_direction,
                }
            )
    return sorted(
        pairs,
        key=lambda row: (
            row["season"],
            row["model_arm"],
            row["team_id"],
            row["next_team_game_number"],
        ),
    )


def lag_statistics(
    rows: list[dict[str, Any]], weights: np.ndarray | None = None
) -> dict[str, float | int | None]:
    """Calculate Pearson lag correlation and same-sign directional persistence."""
    if not rows:
        return {
            "transitions": 0,
            "lag1_correlation": None,
            "directional_persistence": None,
            "nonzero_directional_pairs": 0,
            "positive_previous_error_share": None,
            "positive_next_error_share": None,
            "directional_independence_baseline": None,
            "directional_excess_over_independence": None,
        }
    previous = np.asarray([float(row["previous_signed_error"]) for row in rows])
    following = np.asarray([float(row["next_signed_error"]) for row in rows])
    active_weights = np.ones(len(rows), dtype=float) if weights is None else weights
    valid_weight = active_weights > 0
    if not np.any(valid_weight):
        return {
            "transitions": 0,
            "lag1_correlation": None,
            "directional_persistence": None,
            "nonzero_directional_pairs": 0,
            "positive_previous_error_share": None,
            "positive_next_error_share": None,
            "directional_independence_baseline": None,
            "directional_excess_over_independence": None,
        }
    x = previous[valid_weight]
    y = following[valid_weight]
    w = active_weights[valid_weight]
    total_weight = float(w.sum())
    mean_x = float(np.dot(w, x) / total_weight)
    mean_y = float(np.dot(w, y) / total_weight)
    centered_x = x - mean_x
    centered_y = y - mean_y
    variance_x = float(np.dot(w, centered_x * centered_x) / total_weight)
    variance_y = float(np.dot(w, centered_y * centered_y) / total_weight)
    if variance_x <= 0.0 or variance_y <= 0.0:
        correlation = None
    else:
        covariance = float(np.dot(w, centered_x * centered_y) / total_weight)
        correlation = covariance / math.sqrt(variance_x * variance_y)
    nonzero = (x != 0.0) & (y != 0.0)
    if np.any(nonzero):
        previous_positive = x[nonzero] > 0.0
        next_positive = y[nonzero] > 0.0
        directional_weights = w[nonzero]
        directional_weight = float(directional_weights.sum())
        previous_positive_share = float(
            np.dot(directional_weights, previous_positive.astype(float))
            / directional_weight
        )
        next_positive_share = float(
            np.dot(directional_weights, next_positive.astype(float))
            / directional_weight
        )
        same = previous_positive == next_positive
        directional = float(
            np.dot(directional_weights, same.astype(float)) / directional_weight
        )
        independence_baseline = (
            previous_positive_share * next_positive_share
            + (1.0 - previous_positive_share) * (1.0 - next_positive_share)
        )
        directional_excess = directional - independence_baseline
        nonzero_pair_count = int(np.count_nonzero(nonzero))
    else:
        directional = None
        previous_positive_share = None
        next_positive_share = None
        independence_baseline = None
        directional_excess = None
        nonzero_pair_count = 0
    return {
        "transitions": int(np.count_nonzero(valid_weight)),
        "lag1_correlation": correlation,
        "directional_persistence": directional,
        "nonzero_directional_pairs": nonzero_pair_count,
        "positive_previous_error_share": previous_positive_share,
        "positive_next_error_share": next_positive_share,
        "directional_independence_baseline": independence_baseline,
        "directional_excess_over_independence": directional_excess,
    }


def _cluster_codes(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    teams = sorted({str(row["team_id"]) for row in rows})
    games = sorted(
        {
            f"{row['season']}:{game_id}"
            for row in rows
            for game_id in (row["previous_game_id"], row["next_game_id"])
        }
    )
    team_index = {value: index for index, value in enumerate(teams)}
    game_index = {value: index for index, value in enumerate(games)}
    team_codes = np.asarray(
        [team_index[str(row["team_id"])] for row in rows], dtype=int
    )
    previous_codes = np.asarray(
        [game_index[f"{row['season']}:{row['previous_game_id']}"] for row in rows],
        dtype=int,
    )
    next_codes = np.asarray(
        [game_index[f"{row['season']}:{row['next_game_id']}"] for row in rows],
        dtype=int,
    )
    return team_codes, previous_codes, next_codes, len(teams), len(games)


def _bootstrap_weights(
    rng: np.random.Generator,
    codes: tuple[np.ndarray, np.ndarray, np.ndarray, int, int],
) -> np.ndarray:
    team_codes, previous_codes, next_codes, team_count, game_count = codes
    sampled_teams = np.bincount(
        rng.integers(0, team_count, size=team_count), minlength=team_count
    )
    sampled_games = np.bincount(
        rng.integers(0, game_count, size=game_count), minlength=game_count
    )
    return (
        sampled_teams[team_codes]
        * sampled_games[previous_codes]
        * sampled_games[next_codes]
    ).astype(float)


def _percentile_interval(values: list[float]) -> tuple[float | None, float | None]:
    finite = np.asarray(
        [value for value in values if math.isfinite(value)], dtype=float
    )
    if finite.size == 0:
        return None, None
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def _seed_for(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _cluster_bootstrap_intervals(
    rows: list[dict[str, Any]], repetitions: int, seed: int, label: str
) -> dict[str, float | None]:
    if not rows or repetitions <= 0:
        return {
            "correlation_ci95_low": None,
            "correlation_ci95_high": None,
            "directional_ci95_low": None,
            "directional_ci95_high": None,
            "directional_baseline_ci95_low": None,
            "directional_baseline_ci95_high": None,
            "directional_excess_ci95_low": None,
            "directional_excess_ci95_high": None,
        }
    codes = _cluster_codes(rows)
    rng = np.random.default_rng(_seed_for(seed, label))
    correlations: list[float] = []
    directionals: list[float] = []
    baselines: list[float] = []
    excesses: list[float] = []
    for _ in range(repetitions):
        stats = lag_statistics(rows, _bootstrap_weights(rng, codes))
        if stats["lag1_correlation"] is not None:
            correlations.append(float(stats["lag1_correlation"]))
        if stats["directional_persistence"] is not None:
            directionals.append(float(stats["directional_persistence"]))
            baselines.append(float(stats["directional_independence_baseline"]))
            excesses.append(float(stats["directional_excess_over_independence"]))
    corr_low, corr_high = _percentile_interval(correlations)
    dir_low, dir_high = _percentile_interval(directionals)
    baseline_low, baseline_high = _percentile_interval(baselines)
    excess_low, excess_high = _percentile_interval(excesses)
    return {
        "correlation_ci95_low": corr_low,
        "correlation_ci95_high": corr_high,
        "directional_ci95_low": dir_low,
        "directional_ci95_high": dir_high,
        "directional_baseline_ci95_low": baseline_low,
        "directional_baseline_ci95_high": baseline_high,
        "directional_excess_ci95_low": excess_low,
        "directional_excess_ci95_high": excess_high,
    }


def _summary_row(
    rows: list[dict[str, Any]],
    *,
    summary_level: str,
    season: int | str,
    arm: str,
    family: str,
    policy: str,
    prior_group: str = "",
    progress_bin: str = "",
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    stats = lag_statistics(rows)
    intervals = _cluster_bootstrap_intervals(
        rows,
        repetitions,
        seed,
        f"{summary_level}|{season}|{arm}|{prior_group}|{progress_bin}",
    )
    games = {
        f"{row['season']}:{game_id}"
        for row in rows
        for game_id in (row["previous_game_id"], row["next_game_id"])
    }
    teams = {str(row["team_id"]) for row in rows}
    return {
        "summary_level": summary_level,
        "season": season,
        "model_arm": arm,
        "prior_family": family,
        "policy": policy,
        "preseason_rank_prior_group": prior_group,
        "season_progress_bin": progress_bin,
        "transitions": stats["transitions"],
        "team_count": len(teams),
        "unique_matchups": len(games),
        "lag1_correlation": stats["lag1_correlation"],
        "correlation_ci95_low": intervals["correlation_ci95_low"],
        "correlation_ci95_high": intervals["correlation_ci95_high"],
        "directional_persistence": stats["directional_persistence"],
        "directional_ci95_low": intervals["directional_ci95_low"],
        "directional_ci95_high": intervals["directional_ci95_high"],
        "nonzero_directional_pairs": stats["nonzero_directional_pairs"],
        "positive_previous_error_share": stats["positive_previous_error_share"],
        "positive_next_error_share": stats["positive_next_error_share"],
        "directional_independence_baseline": stats[
            "directional_independence_baseline"
        ],
        "directional_baseline_ci95_low": intervals[
            "directional_baseline_ci95_low"
        ],
        "directional_baseline_ci95_high": intervals[
            "directional_baseline_ci95_high"
        ],
        "directional_excess_over_independence": stats[
            "directional_excess_over_independence"
        ],
        "directional_excess_ci95_low": intervals["directional_excess_ci95_low"],
        "directional_excess_ci95_high": intervals["directional_excess_ci95_high"],
    }


def _contrast_bootstrap(
    rows: list[dict[str, Any]],
    *,
    attribute: str,
    group_a: str,
    group_b: str,
    repetitions: int,
    seed: int,
    label: str,
) -> dict[str, tuple[float | None, float | None]]:
    mask_a = np.asarray([str(row[attribute]) == group_a for row in rows])
    mask_b = np.asarray([str(row[attribute]) == group_b for row in rows])
    codes = _cluster_codes(rows)
    rng = np.random.default_rng(_seed_for(seed, label))
    differences: dict[str, list[float]] = {
        "lag1_correlation": [],
        "directional_excess_over_independence": [],
    }
    for _ in range(repetitions):
        weights = _bootstrap_weights(rng, codes)
        stats_a = lag_statistics(
            [row for row, keep in zip(rows, mask_a) if keep], weights[mask_a]
        )
        stats_b = lag_statistics(
            [row for row, keep in zip(rows, mask_b) if keep], weights[mask_b]
        )
        for metric, metric_values in differences.items():
            value_a = stats_a[metric]
            value_b = stats_b[metric]
            if value_a is not None and value_b is not None:
                metric_values.append(float(value_a) - float(value_b))
    return {
        metric: _percentile_interval(values) for metric, values in differences.items()
    }


def _contrast_rows(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    family: str,
    policy: str,
    contrast: str,
    attribute: str,
    group_a: str,
    group_b: str,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows_a = [row for row in rows if str(row[attribute]) == group_a]
    rows_b = [row for row in rows if str(row[attribute]) == group_b]
    stats_a = lag_statistics(rows_a)
    stats_b = lag_statistics(rows_b)
    intervals = _contrast_bootstrap(
        rows,
        attribute=attribute,
        group_a=group_a,
        group_b=group_b,
        repetitions=repetitions,
        seed=seed,
        label=f"{arm}|{contrast}",
    )
    output: list[dict[str, Any]] = []
    for metric in ("lag1_correlation", "directional_excess_over_independence"):
        value_a = stats_a[metric]
        value_b = stats_b[metric]
        difference = (
            None
            if value_a is None or value_b is None
            else float(value_a) - float(value_b)
        )
        low, high = intervals[metric]
        output.append(
            {
                "model_arm": arm,
                "prior_family": family,
                "policy": policy,
                "contrast": contrast,
                "group_a": group_a,
                "group_b": group_b,
                "metric": metric,
                "difference_a_minus_b": difference,
                "ci95_low": low,
                "ci95_high": high,
                "transitions_a": stats_a["transitions"],
                "transitions_b": stats_b["transitions"],
            }
        )
    return output


def summarize_lagged_pairs(
    all_pairs: list[dict[str, Any]],
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [row for row in all_pairs if row["eligible"]]
    summaries: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    arms = {arm: (family, policy) for arm, family, policy in ARMS}
    for arm, family, policy in ARMS:
        arm_rows = [row for row in eligible if row["model_arm"] == arm]
        summaries.append(
            _summary_row(
                arm_rows,
                summary_level="pooled_model_arm",
                season="2022-2025",
                arm=arm,
                family=family,
                policy=policy,
                repetitions=repetitions,
                seed=seed,
            )
        )
        for season in SEASONS:
            members = [row for row in arm_rows if int(row["season"]) == season]
            summaries.append(
                _summary_row(
                    members,
                    summary_level="season_model_arm",
                    season=season,
                    arm=arm,
                    family=family,
                    policy=policy,
                    repetitions=repetitions,
                    seed=seed,
                )
            )
        for prior_group in ("strong_preseason_rank_prior", "other"):
            members = [
                row for row in arm_rows if row["preseason_rank_prior_group"] == prior_group
            ]
            summaries.append(
                _summary_row(
                    members,
                    summary_level="pooled_prior_stratum",
                    season="2022-2025",
                    arm=arm,
                    family=family,
                    policy=policy,
                    prior_group=prior_group,
                    repetitions=repetitions,
                    seed=seed,
                )
            )
        progress_bins = ("early_games_2_4", "middle_games_5_8", "late_game_9_plus")
        for progress in progress_bins:
            members = [
                row for row in arm_rows if row["season_progress_bin"] == progress
            ]
            summaries.append(
                _summary_row(
                    members,
                    summary_level="pooled_season_progress",
                    season="2022-2025",
                    arm=arm,
                    family=family,
                    policy=policy,
                    progress_bin=progress,
                    repetitions=repetitions,
                    seed=seed,
                )
            )
        contrasts.extend(
            _contrast_rows(
                arm_rows,
                arm=arm,
                family=family,
                policy=policy,
                contrast="strong_preseason_rank_prior_minus_other",
                attribute="preseason_rank_prior_group",
                group_a="strong_preseason_rank_prior",
                group_b="other",
                repetitions=repetitions,
                seed=seed,
            )
        )
        contrasts.extend(
            _contrast_rows(
                arm_rows,
                arm=arm,
                family=family,
                policy=policy,
                contrast="late_minus_early",
                attribute="season_progress_bin",
                group_a="late_game_9_plus",
                group_b="early_games_2_4",
                repetitions=repetitions,
                seed=seed,
            )
        )
    if set(arms) != {row["model_arm"] for row in all_pairs}:
        raise ValueError("lagged pairs do not contain exactly the four #201 model arms")
    return summaries, contrasts


def transition_audit(all_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"candidate_transitions": 0, "eligible_transitions": 0}
    )
    for row in all_pairs:
        key = (str(row["model_arm"]), int(row["season"]))
        counts[key]["candidate_transitions"] += 1
        if row["eligible"]:
            counts[key]["eligible_transitions"] += 1
    output: list[dict[str, Any]] = []
    for arm, family, policy in ARMS:
        for season in SEASONS:
            entry = counts[(arm, season)]
            output.append(
                {
                    "season": season,
                    "model_arm": arm,
                    "prior_family": family,
                    "policy": policy,
                    **entry,
                    "excluded_transitions": entry["candidate_transitions"]
                    - entry["eligible_transitions"],
                }
            )
    return output


def output_manifest(
    output_root: Path,
    *,
    input_hashes: dict[str, dict[str, Any]],
    protocol_path: Path,
    config: dict[str, Any],
    counts: dict[str, Any],
) -> dict[str, Any]:
    outputs: dict[str, dict[str, Any]] = {}
    for path in sorted(output_root.iterdir()):
        if not path.is_file() or path.name == "manifest.json":
            continue
        outputs[path.name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return {
        "experiment": "issue-246-sequential-game-error-persistence",
        "schema_version": 1,
        "status": "complete",
        "seasons": list(SEASONS),
        "input_hashes": input_hashes,
        "protocol_sha256": sha256_file(protocol_path),
        "config": config,
        "counts": counts,
        "outputs": outputs,
    }
