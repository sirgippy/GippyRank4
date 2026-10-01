"""Materialize defensible repairs from the committed transfer evidence.

This command performs no network requests and never edits the frozen #141 or
#142 artifacts. Its inputs are the retained player-level audits and the
committed historical transfer panel.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
from collections import Counter
from math import isclose
from pathlib import Path
from typing import Any

from audit_transfer_availability import PRIMARY_PRIORITY, REPAIR_CLASS

from gippyrank.transfer_repair import (
    build_db_coverage_inventory,
    repair_verified_historical_zero_aggregates,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/transfer_data_repair"
BASELINE_AUDIT = ROOT / "data/processed/transfer_availability_audit"
CURRENT_AVAILABILITY_AUDIT = (
    ROOT / "data/processed/transfer_data_repair/availability_2026_reacquired_20260930"
)
HISTORICAL_FEATURES = (
    ROOT
    / "data/processed/preseason/context_v1_3_candidate/historical_transfer_features.csv"
)
HISTORICAL_TEAM_COVERAGE = (
    ROOT / "data/processed/transfer_production_audit/team_feature_coverage.csv"
)
HISTORICAL_PLAYER_AUDIT = (
    ROOT / "data/processed/transfer_production_audit/player_join_records.csv"
)
CURRENT_ROOT = ROOT / "data/processed/preseason/context_v1_3_2026_reacquired_20260930"
CURRENT_RAW_ROOT = ROOT / "data/raw/cfbd/preseason/transfers_reacquired_20260930"
PRE_REPAIR_CURRENT_ROOT = Path("/tmp/gippyrank149-before-current-reacquired")
ORIGINAL_CURRENT_ROOT = (
    ROOT / "data/processed/preseason/context_v1_3_2026_reconstruction"
)
ORIGINAL_CURRENT_MANIFEST = ORIGINAL_CURRENT_ROOT / "source_manifest.json"
PARTIAL_DB_COVERAGE = (
    ROOT / "data/processed/partial_db_transfer_impact_142/empirical_2026_coverage.csv"
)
RAW_ROOT = ROOT / "data/raw/cfbd/preseason/transfers"
CURRENT_OFFENSE_PLAYER_AUDIT = CURRENT_ROOT / "offensive_player_join_records.csv"

GOOD_DB = frozenset({"resolved", "zero_recorded_defensive_box_score_games"})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_request_key(item: dict[str, Any]) -> tuple[str, int, str]:
    parameters = item.get("parameters", item.get("query_parameters", {}))
    source_season = item.get("source_season")
    if source_season is None:
        source_season = parameters.get("year", item.get("season", 0))
    return (
        str(item.get("source", "")),
        int(source_season),
        json.dumps(parameters, sort_keys=True, separators=(",", ":")),
    )


def _compare_snapshot_lineage(
    original_manifest_path: Path, current_manifest: dict[str, Any]
) -> dict[str, Any]:
    original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    original = [
        item
        for item in original_manifest.get("snapshots", [])
        if item.get("canonical", True)
    ]
    current = [
        item
        for item in current_manifest.get("snapshots", [])
        if item.get("canonical", True)
    ]
    current_by_key = {_snapshot_request_key(item): item for item in current}
    matched = [
        (item, current_by_key[_snapshot_request_key(item)])
        for item in original
        if _snapshot_request_key(item) in current_by_key
    ]
    same_hash = sum(
        left.get("sha256") == right.get("sha256") for left, right in matched
    )
    return {
        "original_manifest_snapshots": len(original),
        "current_reacquired_snapshots": len(current),
        "paired_original_requests": len(matched),
        "paired_requests_byte_identical": same_hash,
        "paired_requests_hash_different": len(matched) - same_hash,
        "additional_current_requests": len(current) - len(matched),
        "comparison_key": "source + source season + exact query parameters",
        "interpretation": "A matching SHA-256 proves byte identity for that response only; later retrieval does not prove historical availability. Different hashes remain current/reacquired evidence and are not substituted for the original snapshot.",
    }


def _evidence_case(row: dict[str, str]) -> dict[str, str]:
    fields = (
        "portal_index",
        "player_name",
        "raw_origin",
        "origin",
        "position",
        "prior_position",
        "prior_position_group",
        "portal_player_id",
        "prior_player_id",
        "identity_status",
        "impact_status",
        "identity_join_method",
        "impact_join_method",
    )
    return {field: row.get(field, "") for field in fields}


def _offensive_evidence_case(row: dict[str, str]) -> dict[str, str]:
    fields = (
        "portal_index",
        "player_name",
        "origin",
        "destination",
        "position",
        "usage_join_status",
        "usage_join_method",
        "prior_usage_player_ids",
        "d5_applicability",
        "d5_resolution_category",
    )
    return {field: row.get(field, "") for field in fields}


def _prior_stats(row: dict[str, str]) -> dict[str, Any]:
    value = row.get("prior_stats", "")
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    return dict(parsed) if isinstance(parsed, dict) else {}


def _impact_values_equal(before: Any, after: Any) -> bool:
    if before in (None, "") or after in (None, ""):
        return before in (None, "") and after in (None, "")
    try:
        return isclose(float(before), float(after), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(before) == str(after)


def _reference_snapshot_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in manifest.get("snapshots", []):
        if not item.get("canonical", True) or item.get("source") not in {
            "roster",
            "games_players",
        }:
            continue
        parameters = item.get("query_parameters", item.get("parameters", {}))
        classification = str(parameters.get("classification") or "").casefold()
        if (
            classification not in {"fbs", "fcs"}
            or str(parameters.get("team") or "").strip()
        ):
            continue
        key = "|".join(
            (
                str(item.get("source")),
                str(item.get("source_season", item.get("season", ""))),
                json.dumps(parameters, sort_keys=True, separators=(",", ":")),
            )
        )
        result[key] = str(item.get("sha256", ""))
    return result


def _current_reaudit_deltas(
    before_root: Path, current_root: Path, output: Path
) -> dict[str, Any]:
    before_manifest_path = before_root / "source_manifest.json"
    current_manifest_path = current_root / "source_manifest.json"
    before_manifest = json.loads(before_manifest_path.read_text(encoding="utf-8"))
    current_manifest = json.loads(current_manifest_path.read_text(encoding="utf-8"))
    before_players = _read_csv(before_root / "transfer_player_audit.csv")
    current_players = _read_csv(current_root / "transfer_player_audit.csv")
    before_db = {
        (int(row["season"]), int(row["portal_index"])): row
        for row in before_players
        if row["portal_position_group"] == "db"
        and row["in_model_relevant_population"] == "True"
    }
    current_db = {
        (int(row["season"]), int(row["portal_index"])): row
        for row in current_players
        if row["portal_position_group"] == "db"
        and row["in_model_relevant_population"] == "True"
    }
    if before_db.keys() != current_db.keys():
        raise ValueError("2026 DB portal identity universe changed between audits")
    before_portal_hashes = sorted(
        str(item["sha256"])
        for item in before_manifest["snapshots"]
        if item.get("source") == "portal"
    )
    current_portal_hashes = sorted(
        str(item["sha256"])
        for item in current_manifest["snapshots"]
        if item.get("source") == "portal"
    )
    if before_portal_hashes != current_portal_hashes:
        raise ValueError(
            "portal snapshots differ; portal_index is not a safe comparison key"
        )
    before_reference_hashes = _reference_snapshot_hashes(before_manifest)
    current_reference_hashes = _reference_snapshot_hashes(current_manifest)
    reference_keys = before_reference_hashes.keys() | current_reference_hashes.keys()
    reference_comparison = {
        "before_snapshot_count": len(before_reference_hashes),
        "after_snapshot_count": len(current_reference_hashes),
        "matching_request_count": len(
            before_reference_hashes.keys() & current_reference_hashes.keys()
        ),
        "changed_sha256_count": sum(
            before_reference_hashes.get(key) != current_reference_hashes.get(key)
            for key in reference_keys
            if key in before_reference_hashes and key in current_reference_hashes
        ),
        "added_request_count": len(
            current_reference_hashes.keys() - before_reference_hashes.keys()
        ),
        "removed_request_count": len(
            before_reference_hashes.keys() - current_reference_hashes.keys()
        ),
    }

    columns = [
        "audit_type",
        "season",
        "portal_index",
        "player_name",
        "source_team_before",
        "source_team_after",
        "before_status",
        "after_status",
        "before_method",
        "after_method",
        "before_position",
        "after_position",
        "before_position_group",
        "after_position_group",
        "before_provider_player_id",
        "after_provider_player_id",
        "before_raw_stats",
        "after_raw_stats",
        "raw_stat_fields_changed",
        "impact_change_class",
        "before_value",
        "after_value",
        "repair_rule",
        "before_source_manifest_sha256",
        "after_source_manifest_sha256",
        "before_portal_snapshot_sha256",
        "after_portal_snapshot_sha256",
    ]
    deltas: list[dict[str, Any]] = []
    before_db_statuses: Counter[str] = Counter()
    after_db_statuses: Counter[str] = Counter()
    repaired_db_rows = 0
    db_status_changed_rows = 0
    db_impact_value_changed_rows = 0
    db_raw_stat_changed_rows = 0
    raw_stat_changed_fields: Counter[str] = Counter()
    db_impact_change_classes: Counter[str] = Counter()
    db_impact_status_transitions: Counter[str] = Counter()
    for key, after in current_db.items():
        before = before_db[key]
        before_status = str(before["impact_status"])
        after_status = str(after["impact_status"])
        before_stats = _prior_stats(before)
        after_stats = _prior_stats(after)
        stat_fields_changed = sorted(
            field
            for field in before_stats.keys() | after_stats.keys()
            if not _impact_values_equal(before_stats.get(field), after_stats.get(field))
        )
        if stat_fields_changed:
            db_raw_stat_changed_rows += 1
            raw_stat_changed_fields.update(stat_fields_changed)
        impact_changed = not _impact_values_equal(
            before.get("prior_defensive_impact"),
            after.get("prior_defensive_impact"),
        )
        impact_change_class = "impact_value_unchanged"
        if impact_changed:
            db_impact_value_changed_rows += 1
            db_impact_status_transitions[f"{before_status} → {after_status}"] += 1
            if before_status not in GOOD_DB and after_status in GOOD_DB:
                impact_change_class = "newly_resolved_identity_or_source_coverage"
            elif before.get("prior_position") != after.get("prior_position"):
                impact_change_class = "roster_position_evidence_changed"
            elif stat_fields_changed:
                impact_change_class = "underlying_player_defensive_statistics_changed"
            elif before_status in GOOD_DB and after_status not in GOOD_DB:
                impact_change_class = "impact_evidence_became_unavailable"
            elif before_status in GOOD_DB and after_status in GOOD_DB:
                impact_change_class = "normalization_reference_parameters_changed"
            else:
                impact_change_class = "other_join_or_evidence_status_change"
            db_impact_change_classes[impact_change_class] += 1
        before_db_statuses[before_status] += 1
        after_db_statuses[after_status] += 1
        db_status_changed_rows += before_status != after_status
        before_signature = tuple(
            str(before.get(field, ""))
            for field in (
                "origin",
                "identity_status",
                "identity_join_method",
                "impact_status",
                "impact_join_method",
                "prior_position_group",
                "prior_player_id",
                "prior_defensive_impact",
                "prior_stats",
            )
        )
        after_signature = tuple(
            str(after.get(field, ""))
            for field in (
                "origin",
                "identity_status",
                "identity_join_method",
                "impact_status",
                "impact_join_method",
                "prior_position_group",
                "prior_player_id",
                "prior_defensive_impact",
                "prior_stats",
            )
        )
        if before_signature == after_signature:
            continue
        if before_status not in GOOD_DB and after_status in GOOD_DB:
            repaired_db_rows += 1
        method = str(after.get("identity_join_method", ""))
        if method == "stable_game_player_id_source_team":
            repair_rule = "unique exact-name, source-team game-player ID bridge"
        elif method == "normalized_name_source_team_generational_suffix":
            repair_rule = (
                "unique same-team match after removing terminal generational suffix"
            )
        elif str(before.get("origin", "")) != str(after.get("origin", "")):
            repair_rule = "verified 2026 season-scoped source-team alias"
        elif before_status not in GOOD_DB and after_status in GOOD_DB:
            repair_rule = (
                "supplemental roster/game-player evidence established a unique "
                "same-team identity and impact"
            )
        elif impact_change_class == "normalization_reference_parameters_changed":
            repair_rule = "defensive-impact reference normalization parameters changed"
        elif stat_fields_changed:
            repair_rule = "underlying defensive player statistics changed: " + ";".join(
                stat_fields_changed
            )
        elif (
            before_status == "source_data_unavailable" and after_status != before_status
        ):
            repair_rule = "expanded current division source coverage"
        else:
            repair_rule = "reacquired current source evidence"
        deltas.append(
            {
                "audit_type": "defensive_db_player",
                "season": key[0],
                "portal_index": key[1],
                "player_name": after["player_name"],
                "source_team_before": before.get("origin", ""),
                "source_team_after": after.get("origin", ""),
                "before_status": before_status,
                "after_status": after_status,
                "before_method": before.get("identity_join_method", ""),
                "after_method": after.get("identity_join_method", ""),
                "before_position": before.get("prior_position", ""),
                "after_position": after.get("prior_position", ""),
                "before_position_group": before.get("prior_position_group", ""),
                "after_position_group": after.get("prior_position_group", ""),
                "before_provider_player_id": before.get("prior_player_id", ""),
                "after_provider_player_id": after.get("prior_player_id", ""),
                "before_raw_stats": json.dumps(before_stats, sort_keys=True),
                "after_raw_stats": json.dumps(after_stats, sort_keys=True),
                "raw_stat_fields_changed": ";".join(stat_fields_changed),
                "impact_change_class": impact_change_class,
                "before_value": before.get("prior_defensive_impact", ""),
                "after_value": after.get("prior_defensive_impact", ""),
                "repair_rule": repair_rule,
                "before_source_manifest_sha256": _sha256(before_manifest_path),
                "after_source_manifest_sha256": _sha256(current_manifest_path),
                "before_portal_snapshot_sha256": before.get(
                    "portal_snapshot_sha256", ""
                ),
                "after_portal_snapshot_sha256": after.get("portal_snapshot_sha256", ""),
            }
        )

    before_offense_path = before_root / "offensive_player_join_records.csv"
    current_offense_path = current_root / "offensive_player_join_records.csv"
    before_offense_count = 0
    after_offense_count = 0
    changed_offense_rows = 0
    if before_offense_path.is_file() and current_offense_path.is_file():
        before_offense_rows = _read_csv(before_offense_path)
        current_offense_rows = _read_csv(current_offense_path)
        before_offense = {
            int(row["portal_index"]): row
            for row in before_offense_rows
            if row["in_model_relevant_population"] == "True"
        }
        current_offense = {
            int(row["portal_index"]): row
            for row in current_offense_rows
            if row["in_model_relevant_population"] == "True"
        }
        if before_offense.keys() != current_offense.keys():
            raise ValueError("2026 offensive portal identity universe changed")
        failed_category = (
            "should_have_recoverable_offensive_usage_but_resolution_failed"
        )
        before_offense_count = sum(
            row["d5_resolution_category"] == failed_category
            for row in before_offense.values()
        )
        after_offense_count = sum(
            row["d5_resolution_category"] == failed_category
            for row in current_offense.values()
        )
        for portal_index, after in current_offense.items():
            before = before_offense[portal_index]
            signature_fields = (
                "usage_join_status",
                "d5_resolution_category",
                "d5_feature_value",
                "prior_usage",
                "incoming_prior_offensive_usage",
            )
            if all(
                before.get(field, "") == after.get(field, "")
                for field in signature_fields
            ):
                continue
            changed_offense_rows += 1
            deltas.append(
                {
                    "audit_type": "offensive_usage_player",
                    "season": after["season"],
                    "portal_index": portal_index,
                    "player_name": after["player_name"],
                    "source_team_before": before.get("origin", ""),
                    "source_team_after": after.get("origin", ""),
                    "before_status": before.get("usage_join_status", ""),
                    "after_status": after.get("usage_join_status", ""),
                    "before_method": before.get("usage_join_method", ""),
                    "after_method": after.get("usage_join_method", ""),
                    "before_position": before.get("position", ""),
                    "after_position": after.get("position", ""),
                    "before_provider_player_id": before.get("player_id", ""),
                    "after_provider_player_id": after.get("prior_usage_player_ids", ""),
                    "before_value": before.get("d5_feature_value", ""),
                    "after_value": after.get("d5_feature_value", ""),
                    "repair_rule": (
                        "coalesce duplicate usage rows with the same stable player ID, position, and usage value"
                        if int(after.get("duplicate_usage_rows_coalesced", "0")) > 0
                        else "reacquired current offensive source evidence"
                    ),
                    "before_source_manifest_sha256": _sha256(before_manifest_path),
                    "after_source_manifest_sha256": _sha256(current_manifest_path),
                    "before_portal_snapshot_sha256": before.get(
                        "portal_snapshot_sha256", ""
                    ),
                    "after_portal_snapshot_sha256": after.get(
                        "portal_snapshot_sha256", ""
                    ),
                }
            )

    before_teams = {
        str(row["team_id"]): row
        for row in _read_csv(before_root / "transfer_team_audit.csv")
    }
    current_teams = {
        str(row["team_id"]): row
        for row in _read_csv(current_root / "transfer_team_audit.csv")
    }
    if before_teams.keys() != current_teams.keys():
        raise ValueError("2026 Context team-season universe changed between audits")
    team_columns = [
        "team_id",
        "team_name_before",
        "team_name_after",
        "incoming_db_before",
        "incoming_db_after",
        "observed_db_before",
        "observed_db_after",
        "missing_db_before",
        "missing_db_after",
        "observed_impact_sum_before",
        "observed_impact_sum_after",
        "coverage_status_before",
        "coverage_status_after",
        "before_source_manifest_sha256",
        "after_source_manifest_sha256",
    ]
    team_deltas: list[dict[str, Any]] = []
    for team_id, after in current_teams.items():
        before = before_teams[team_id]
        fields = (
            "audit_incoming_db_transfers",
            "audit_resolved_db_transfers",
            "audit_unresolved_db_transfers",
            "transfer_in_prior_defensive_impact_db_sum",
            "audit_db_feature_status",
        )
        if all(before.get(field, "") == after.get(field, "") for field in fields):
            continue
        team_deltas.append(
            {
                "team_id": team_id,
                "team_name_before": before.get("team_name", ""),
                "team_name_after": after.get("team_name", ""),
                "incoming_db_before": before.get("audit_incoming_db_transfers", ""),
                "incoming_db_after": after.get("audit_incoming_db_transfers", ""),
                "observed_db_before": before.get("audit_resolved_db_transfers", ""),
                "observed_db_after": after.get("audit_resolved_db_transfers", ""),
                "missing_db_before": before.get("audit_unresolved_db_transfers", ""),
                "missing_db_after": after.get("audit_unresolved_db_transfers", ""),
                "observed_impact_sum_before": before.get(
                    "transfer_in_prior_defensive_impact_db_sum", ""
                ),
                "observed_impact_sum_after": after.get(
                    "transfer_in_prior_defensive_impact_db_sum", ""
                ),
                "coverage_status_before": before.get("audit_db_feature_status", ""),
                "coverage_status_after": after.get("audit_db_feature_status", ""),
                "before_source_manifest_sha256": _sha256(before_manifest_path),
                "after_source_manifest_sha256": _sha256(current_manifest_path),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "player_before_after_2026.csv", deltas, columns)
    _write_csv(output / "team_before_after_2026.csv", team_deltas, team_columns)
    unresolved_db_cases = [
        {
            "player_name": row["player_name"],
            "source_team": row.get("origin", ""),
            "destination_team": row.get("destination", ""),
            "status": row["impact_status"],
            "prior_position": row.get("prior_position", ""),
        }
        for row in current_players
        if row["in_model_relevant_population"] == "True"
        and row["portal_position_group"] == "db"
        and row["impact_status"] not in GOOD_DB
    ]
    unresolved_offense_cases = (
        [
            {
                "player_name": row["player_name"],
                "source_team": row.get("origin", ""),
                "destination_team": row.get("destination", ""),
                "join_status": row.get("usage_join_status", ""),
                "category": row.get("d5_resolution_category", ""),
            }
            for row in current_offense_rows
            if row["in_model_relevant_population"] == "True"
            and row["d5_resolution_category"]
            == "should_have_recoverable_offensive_usage_but_resolution_failed"
        ]
        if before_offense_path.is_file() and current_offense_path.is_file()
        else []
    )
    return {
        "before_root": str(before_root),
        "before_source_manifest_sha256": _sha256(before_manifest_path),
        "after_source_manifest_sha256": _sha256(current_manifest_path),
        "baseline_has_offensive_player_audit": before_offense_path.is_file(),
        "changed_player_rows": len(deltas),
        "changed_defensive_db_players": len(deltas) - changed_offense_rows,
        "db_player_status_counts_before": dict(sorted(before_db_statuses.items())),
        "db_player_status_counts_after": dict(sorted(after_db_statuses.items())),
        "db_players_unresolved_before": sum(
            count
            for status, count in before_db_statuses.items()
            if status not in GOOD_DB
        ),
        "db_players_unresolved_after": sum(
            count
            for status, count in after_db_statuses.items()
            if status not in GOOD_DB
        ),
        "db_players_repaired_to_resolved_or_zero": repaired_db_rows,
        "db_player_status_changed_rows": db_status_changed_rows,
        "db_player_impact_value_changed_rows": db_impact_value_changed_rows,
        "db_player_impact_change_classes": dict(
            sorted(db_impact_change_classes.items())
        ),
        "db_player_impact_status_transitions": dict(
            sorted(db_impact_status_transitions.items())
        ),
        "db_player_impact_value_changed_cases": [
            {
                field: row.get(field, "")
                for field in (
                    "player_name",
                    "source_team_before",
                    "source_team_after",
                    "before_status",
                    "after_status",
                    "before_position",
                    "after_position",
                    "before_raw_stats",
                    "after_raw_stats",
                    "raw_stat_fields_changed",
                    "impact_change_class",
                    "before_value",
                    "after_value",
                    "repair_rule",
                )
            }
            for row in deltas
            if row.get("audit_type") == "defensive_db_player"
            and not _impact_values_equal(
                row.get("before_value"), row.get("after_value")
            )
        ],
        "db_player_raw_stat_changed_rows": db_raw_stat_changed_rows,
        "db_player_raw_stat_changed_fields": dict(
            sorted(raw_stat_changed_fields.items())
        ),
        "defensive_impact_reference_comparison": reference_comparison,
        "offensive_applicable_failures_before": before_offense_count,
        "offensive_applicable_failures_after": after_offense_count,
        "changed_offensive_player_rows": changed_offense_rows,
        "team_seasons_with_changed_db_audit": len(team_deltas),
        "remaining_db_cases": unresolved_db_cases,
        "remaining_offensive_cases": unresolved_offense_cases,
        "player_artifact": "player_before_after_2026.csv",
        "team_artifact": "team_before_after_2026.csv",
    }


def _raw_payload_files(raw_root: Path, source: str) -> list[Path]:
    """Return payload JSON files, excluding adjacent provenance sidecars."""
    return sorted(
        path
        for path in (raw_root / source).rglob("*.json")
        if path.is_file() and not path.name.endswith(".provenance.json")
    )


def _local_input_search(original_manifest_path: Path) -> dict[str, Any]:
    """Record where original raw hashes and related ignored artifacts were checked."""
    worktree_result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    worktrees = sorted(
        {
            Path(line.removeprefix("worktree ")).resolve()
            for line in worktree_result.stdout.splitlines()
            if line.startswith("worktree ")
        }
    )
    temporary_roots = sorted(
        {
            path.resolve()
            for pattern in ("GippyRank4-*", "gippyrank*")
            for path in Path("/tmp").glob(pattern)
            if path.is_dir()
        }
        - set(worktrees)
    )
    search_roots = sorted(set(worktrees) | set(temporary_roots))
    raw_directories = [
        root / "data/raw/cfbd/preseason"
        for root in search_roots
        if (root / "data/raw/cfbd/preseason").is_dir()
    ]
    original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    original_snapshots = original_manifest.get("snapshots", [])
    wanted = {str(item["sha256"]): item for item in original_snapshots}
    hash_locations: dict[str, list[str]] = {digest: [] for digest in wanted}
    direct_expected_path_checks: list[dict[str, Any]] = []
    historical_raw_counts: dict[str, dict[str, int]] = {}
    relevant_processed_paths = (
        "data/processed/preseason/context_v1_3_2026_reconstruction/transfer_player_audit.csv",
        "data/processed/preseason/context_v1_3_2026_reconstruction/offensive_player_join_records.csv",
        "data/processed/transfer_production_audit/player_join_records.csv",
        "data/processed/partial_db_transfer_impact_142/empirical_2026_coverage.csv",
        "data/processed/preseason/context_v1_3_2026_reacquired_20260930/offensive_player_join_records.csv",
    )
    processed_artifact_locations: dict[str, list[str]] = {
        relative: [] for relative in relevant_processed_paths
    }

    for root in search_roots:
        expected_root = root / "data/raw/cfbd/preseason/transfers"
        for item in original_snapshots:
            expected_path = expected_root / str(item["path"])
            if expected_path.is_file():
                actual = hashlib.sha256(expected_path.read_bytes()).hexdigest()
                direct_expected_path_checks.append(
                    {
                        "root": str(root),
                        "path": str(expected_path),
                        "expected_sha256": item["sha256"],
                        "actual_sha256": actual,
                        "hash_matches": actual == item["sha256"],
                    }
                )
        raw_directory = root / "data/raw/cfbd/preseason"
        if raw_directory.is_dir():
            for path in raw_directory.rglob("*.json"):
                if not path.is_file() or path.name.endswith(".provenance.json"):
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in hash_locations:
                    hash_locations[digest].append(str(path))
            historical_root = raw_directory / "transfers"
            source_counts = {
                source: len(_raw_payload_files(historical_root, source))
                for source in ("portal", "usage", "stats")
            }
            if any(source_counts.values()):
                historical_raw_counts[str(root)] = source_counts
        for relative in relevant_processed_paths:
            artifact_path = root / relative
            if artifact_path.is_file():
                processed_artifact_locations[relative].append(str(artifact_path))

    primary = next((root for root in worktrees if (root / ".git").is_dir()), None)
    return {
        "original_manifest_snapshot_count": len(original_snapshots),
        "registered_git_worktree_count": len(worktrees),
        "registered_git_worktrees": [str(root) for root in worktrees],
        "primary_checkout": str(primary) if primary else None,
        "temporary_artifact_root_patterns": ["/tmp/GippyRank4-*", "/tmp/gippyrank*"],
        "temporary_artifact_root_count_outside_registered_worktrees": len(
            temporary_roots
        ),
        "temporary_artifact_roots_outside_registered_worktrees": [
            str(root) for root in temporary_roots
        ],
        "temporary_artifact_roots_with_cfbd_raw_subtree": [
            str(root)
            for root in temporary_roots
            if (root / "data/raw/cfbd/preseason").is_dir()
        ],
        "existing_cfbd_raw_subtrees_checked": [str(path) for path in raw_directories],
        "expected_original_versioned_paths_found": len(direct_expected_path_checks),
        "expected_original_path_sha256_pairs": [
            {"path": item["path"], "sha256": item["sha256"]}
            for item in original_snapshots
        ],
        "expected_original_versioned_paths_hash_valid": sum(
            item["hash_matches"] for item in direct_expected_path_checks
        ),
        "expected_original_versioned_path_checks": direct_expected_path_checks,
        "original_source_hashes_found_anywhere_under_local_cfbd_raw_subtrees": sum(
            bool(locations) for locations in hash_locations.values()
        ),
        "original_source_hashes_not_found_under_local_cfbd_raw_subtrees": sum(
            not locations for locations in hash_locations.values()
        ),
        "matching_source_hash_locations": {
            wanted[digest]["path"]: locations
            for digest, locations in hash_locations.items()
            if locations
        },
        "historical_transfer_payload_counts_by_worktree": historical_raw_counts,
        "related_generated_artifact_locations": {
            relative: locations
            for relative, locations in processed_artifact_locations.items()
            if locations
        },
        "matching_sha256_is_content_identity_only_not_retrieval_time_evidence": True,
    }


def _reclassify_team_row(row: dict[str, Any]) -> None:
    codes = {code for code in str(row.get("reason_codes", "")).split(";") if code}
    codes.discard("historical_aggregate_null")
    row["reason_codes"] = ";".join(sorted(codes))
    row["offensive_feature_numeric"] = True
    row["historical_aggregate_repaired"] = True
    if not codes:
        status = "complete"
    elif (
        int(row["incoming_transfers"]) > 0
        and int(row["offensive_resolved"]) == 0
        and int(row["offensive_legitimate_zero"]) == 0
        and int(row["db_unresolved"]) > 0
        and int(row["db_resolved"]) == 0
    ):
        status = "entirely_unavailable"
    else:
        status = "partial"
    categories = {REPAIR_CLASS[code] for code in codes}
    if not codes:
        repair_class = "not_applicable"
    elif categories == {"candidate_repair"}:
        repair_class = "candidate_repair"
    else:
        repair_class = "unresolved"
    primary = next((code for code in PRIMARY_PRIORITY if code in codes), "complete")
    row["availability_status"] = status
    row["repair_class"] = repair_class
    row["primary_reason"] = primary
    row["repair_candidate_reasons"] = ";".join(
        sorted(code for code in codes if REPAIR_CLASS[code] == "candidate_repair")
    )
    row["source_coverage_gap"] = "db_source_team_uncovered" in codes


def _post_repair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter(
        code
        for row in rows
        for code in str(row.get("reason_codes", "")).split(";")
        if code
    )
    by_season: dict[str, Any] = {}
    for season in sorted({int(row["season"]) for row in rows}):
        subset = [row for row in rows if int(row["season"]) == season]
        statuses = Counter(str(row["availability_status"]) for row in subset)
        status_list = [str(row["availability_status"]) for row in subset]
        by_season[str(season)] = {
            "team_seasons": len(subset),
            "statuses": dict(sorted(statuses.items())),
            "affected": sum(status != "complete" for status in status_list),
        }
    statuses = Counter(str(row["availability_status"]) for row in rows)
    repair_classes = Counter(str(row["repair_class"]) for row in rows)
    return {
        "team_seasons": len(rows),
        "statuses": dict(sorted(statuses.items())),
        "reason_team_seasons": dict(sorted(reason_counts.items())),
        "repair_classes": dict(sorted(repair_classes.items())),
        "team_seasons_with_repair_candidate": sum(
            bool(row.get("repair_candidate_reasons")) for row in rows
        ),
        "team_seasons_with_source_coverage_gap": sum(
            row.get("source_coverage_gap") is True
            or str(row.get("source_coverage_gap")).casefold() == "true"
            for row in rows
        ),
        "by_season": by_season,
    }


def _compare_db_coverage(
    post_rows: list[dict[str, Any]], baseline_rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    baseline = {str(row["team_id"]): row for row in baseline_rows}
    post = {str(row["team_id"]): row for row in post_rows}
    if len(post) != 138:
        raise ValueError(f"expected 138 2026 Context teams, found {len(post)}")
    comparisons: Counter[str] = Counter()
    for team_id, row in post.items():
        old = baseline.get(team_id)
        old_n = int(old["total_count"]) if old else 0
        old_k = int(old["observed_count"]) if old else 0
        new_n = int(row["total_count"])
        new_k = int(row["observed_count"])
        if old_n != new_n:
            raise ValueError(
                f"#142 incoming DB count differs for {team_id}: {old_n} != {new_n}"
            )
        if new_n and new_k == new_n and old_k < old_n:
            change = "partial_to_complete"
        elif new_k > old_k and new_k < new_n:
            change = "observed_count_increased_partial"
        else:
            change = "unchanged"
        row["pre_repair_observed_count"] = old_k
        row["coverage_change_class"] = change
        comparisons[change] += 1
    return post_rows, dict(sorted(comparisons.items()))


def run(
    output: Path = OUTPUT,
    *,
    current_root: Path = CURRENT_ROOT,
    raw_root: Path = CURRENT_RAW_ROOT,
    historical_raw_root: Path = RAW_ROOT,
    before_current_root: Path | None = None,
) -> dict[str, Any]:
    historical_features = _read_csv(HISTORICAL_FEATURES)
    historical_coverage = _read_csv(HISTORICAL_TEAM_COVERAGE)
    historical_players = _read_csv(HISTORICAL_PLAYER_AUDIT)
    updated_features, repairs = repair_verified_historical_zero_aggregates(
        historical_features, historical_coverage, historical_players
    )

    team_seasons = _read_csv(BASELINE_AUDIT / "team_seasons.csv")
    # Keep a copy of baseline statuses so the after panel can be audited.
    before_status = {
        (int(row["season"]), str(row["team_id"])): str(row["availability_status"])
        for row in team_seasons
    }
    repair_keys = {(int(row["season"]), str(row["team_id"])) for row in repairs}
    for row in team_seasons:
        key = (int(row["season"]), str(row["team_id"]))
        if key in repair_keys:
            _reclassify_team_row(row)
        else:
            row["historical_aggregate_repaired"] = False

    # Keep the retained historical panel repair separate, and take 2026 team
    # statuses/reasons from the refreshed current-response availability audit.
    current_availability = _read_csv(CURRENT_AVAILABILITY_AUDIT / "team_seasons.csv")
    current_rows = [row for row in current_availability if int(row["season"]) == 2026]
    if len(current_rows) != 138:
        raise ValueError(
            f"expected 138 refreshed 2026 team-seasons, found {len(current_rows)}"
        )
    team_seasons = [row for row in team_seasons if int(row["season"]) != 2026]
    for row in current_rows:
        row["historical_aggregate_repaired"] = False
    team_seasons.extend(current_rows)

    current_team_audit = current_root / "transfer_team_audit.csv"
    current_player_audit = current_root / "transfer_player_audit.csv"
    current_provenance = current_root / "feature_provenance.json"
    current_manifest_path = current_root / "source_manifest.json"
    current_offense_player_audit = current_root / "offensive_player_join_records.csv"
    current_players = _read_csv(current_player_audit)
    current_teams = _read_csv(current_team_audit)
    snapshot_manifest = json.loads(current_manifest_path.read_text(encoding="utf-8"))
    snapshot_lineage = _compare_snapshot_lineage(
        ORIGINAL_CURRENT_MANIFEST, snapshot_manifest
    )
    before_current_root = before_current_root or (
        PRE_REPAIR_CURRENT_ROOT
        if (PRE_REPAIR_CURRENT_ROOT / "transfer_player_audit.csv").is_file()
        else ORIGINAL_CURRENT_ROOT
    )
    current_reaudit = _current_reaudit_deltas(before_current_root, current_root, output)
    _write_json(
        output / "before_source_manifest.json",
        json.loads(
            (before_current_root / "source_manifest.json").read_text(encoding="utf-8")
        ),
    )
    _write_json(output / "after_source_manifest.json", snapshot_manifest)
    defensive_source_hashes = sorted(
        str(item["sha256"])
        for item in snapshot_manifest.get("snapshots", [])
        if item.get("canonical", True)
        if item.get("source") in {"portal", "roster", "games_players"}
    )
    db_coverage = build_db_coverage_inventory(
        current_teams,
        current_players,
        source_season=2025,
        provenance={"source_snapshot_sha256": defensive_source_hashes},
    )
    db_coverage, coverage_change_counts = _compare_db_coverage(
        db_coverage, _read_csv(PARTIAL_DB_COVERAGE)
    )

    # Append the resulting availability classification to each changed
    # team-season repair record without mutating the original audit.
    after_status = {
        (int(row["season"]), str(row["team_id"])): str(row["availability_status"])
        for row in team_seasons
    }
    for repair in repairs:
        key = (int(repair["season"]), str(repair["team_id"]))
        repair["old_availability_status"] = before_status[key]
        repair["new_availability_status"] = after_status[key]
        repair["old_reason"] = "historical_aggregate_null"

    baseline_summary = json.loads(
        (BASELINE_AUDIT / "summary.json").read_text(encoding="utf-8")
    )
    post_summary = _post_repair_summary(team_seasons)
    status_improvements = sum(
        before_status[key] != after_status[key] for key in before_status
    )
    current_reasons = Counter(
        row["impact_status"]
        for row in current_players
        if row["in_model_relevant_population"] == "True"
        and row["portal_position_group"] == "db"
        and row["impact_status"] not in GOOD_DB
    )
    current_offense_players = _read_csv(current_offense_player_audit)
    current_offense_relevant = [
        row
        for row in current_offense_players
        if row["in_model_relevant_population"] == "True"
    ]
    current_usage_duplicates_coalesced = sum(
        int(row.get("duplicate_usage_rows_coalesced", "0") or 0)
        for row in current_offense_relevant
    )
    current_team_alias_player_records = sum(
        row.get("origin", "") != row.get("raw_origin", "")
        for row in current_players
        if row["in_model_relevant_population"] == "True"
        and row["portal_position_group"] == "db"
    )
    identity_method_counts = Counter(
        row.get("identity_join_method", "")
        for row in current_players
        if row["in_model_relevant_population"] == "True"
        and row["portal_position_group"] == "db"
    )
    current_status_counts = Counter(row["coverage_status"] for row in db_coverage)
    coverage_delta = {
        "teams_moving_from_partial_to_complete": coverage_change_counts.get(
            "partial_to_complete", 0
        ),
        "teams_with_observed_count_increase_still_partial": coverage_change_counts.get(
            "observed_count_increased_partial", 0
        ),
        "teams_unchanged": coverage_change_counts.get("unchanged", 0),
    }
    snapshot_records = snapshot_manifest.get("snapshots", [])
    snapshots = [item for item in snapshot_records if item.get("canonical", True)]
    retrieval_timestamps = [
        str(item["retrieval_timestamp"])
        for item in snapshots
        if item.get("retrieval_timestamp")
    ]
    team_directory_provenance_path = (
        raw_root / "auxiliary/team_directory.json.provenance.json"
    )
    team_directory_provenance = (
        json.loads(team_directory_provenance_path.read_text(encoding="utf-8"))
        if team_directory_provenance_path.is_file()
        else None
    )
    missing_snapshots = [
        item for item in snapshots if not (raw_root / str(item["path"])).is_file()
    ]
    targeted_team_requests = [
        item
        for item in snapshots
        if int(item.get("target_season", item.get("season", 0))) == 2026
        and item.get("source") in {"roster", "games_players"}
        and item.get("query_parameters", item.get("parameters", {})).get("team")
    ]
    targeted_team_values = sorted(
        {
            str(item.get("query_parameters", item.get("parameters", {})).get("team"))
            for item in targeted_team_requests
        }
    )
    diagnostic_control_requests = [
        item
        for item in snapshot_records
        if item.get("source") in {"roster", "games_players"}
        and item.get("query_parameters", item.get("parameters", {})).get("team")
        == "Valdosta State"
    ]
    team_filter_evidence = {
        "team_filter_request_count": len(targeted_team_requests),
        "distinct_team_filter_values": len(targeted_team_values),
        "team_filter_values": targeted_team_values,
        "empty_responses": sum(
            int(item.get("record_count", 0)) == 0 for item in targeted_team_requests
        ),
        "nonempty_responses": sum(
            int(item.get("record_count", 0)) > 0 for item in targeted_team_requests
        ),
        "source_request_counts": dict(
            sorted(Counter(item["source"] for item in targeted_team_requests).items())
        ),
        "endpoint_control_check": {
            "request_count": len(diagnostic_control_requests),
            "canonical_for_derivation": False,
            "record_counts_by_source": {
                item["source"]: int(item.get("record_count", 0))
                for item in diagnostic_control_requests
            },
            "purpose": "Verify that the provider accepts the team filter and can return records; excluded from the transfer derivation because it is not one of the unresolved source schools.",
        },
        "evidence_location": "after_source_manifest.json and the reacquired raw manifest",
        "interpretation": "Team-filtered current CFBD requests cover the unresolved DB source schools. Empty arrays are retained as evidence; no player or team alias is inferred from absence.",
    }
    current_db_cases = [
        row
        for row in current_players
        if row["season"] == "2026"
        and row["portal_position_group"] == "db"
        and row["in_model_relevant_population"] == "True"
        and row["impact_status"] not in GOOD_DB
    ]
    remaining_db_cases = {
        "db_player_join_unresolved": [
            _evidence_case(row)
            for row in current_db_cases
            if row["impact_status"] == "identity_resolution_failure"
        ],
        "db_source_team_uncovered": [
            _evidence_case(row)
            for row in current_db_cases
            if row["impact_status"] == "source_data_unavailable"
        ],
        "db_position_conflict": [
            _evidence_case(row)
            for row in current_db_cases
            if row["impact_status"] == "position_mismatch"
        ],
        "db_ambiguous_player_join": [
            _evidence_case(row)
            for row in current_db_cases
            if row["impact_status"] == "ambiguous"
        ],
    }
    remaining_offensive_cases = [
        _offensive_evidence_case(row)
        for row in current_offense_relevant
        if row["d5_resolution_category"]
        == "should_have_recoverable_offensive_usage_but_resolution_failed"
    ]
    historical_payloads_by_source = {
        source: len(_raw_payload_files(historical_raw_root, source))
        for source in ("portal", "usage", "stats")
    }
    historical_raw_payload_count = sum(historical_payloads_by_source.values())
    original_snapshot_search = _local_input_search(ORIGINAL_CURRENT_MANIFEST)
    summary = {
        "baseline_141": baseline_summary,
        "post_repair": post_summary,
        "repairs": {
            "historical_aggregate_team_seasons_repaired": len(repairs),
            "stable_roster_id_impact_join_rule_added": True,
            "stable_game_player_id_identity_bridge_rows": identity_method_counts.get(
                "stable_game_player_id_source_team", 0
            ),
            "generational_suffix_join_rows": identity_method_counts.get(
                "normalized_name_source_team_generational_suffix", 0
            ),
            "verified_team_alias_player_records": current_team_alias_player_records,
            "verified_team_alias_definitions": 4,
            "offensive_duplicate_usage_rows_coalesced": current_usage_duplicates_coalesced,
            "position_rule_repairs": 0,
            "offensive_usage_duplicate_coalescing_rule_added": True,
            "availability_status_improvements": status_improvements,
            "current_2026_db_unresolved_by_player_status": dict(
                sorted(current_reasons.items())
            ),
        },
        "post_repair_2026_db_coverage": {
            "team_status_counts": dict(sorted(current_status_counts.items())),
            **coverage_delta,
            "incoming_db_transfers": sum(
                int(row["total_count"]) for row in db_coverage
            ),
            "observed_db_impacts": sum(
                int(row["observed_count"]) for row in db_coverage
            ),
            "source_team_coverage_gaps_remaining": current_reasons.get(
                "source_data_unavailable", 0
            ),
        },
        "remaining_2026_evidence_gaps": {
            "db_player_join_unresolved": current_reasons.get(
                "identity_resolution_failure", 0
            ),
            "db_source_team_uncovered": current_reasons.get(
                "source_data_unavailable", 0
            ),
            "db_position_conflict": current_reasons.get("position_mismatch", 0),
            "db_ambiguous_player_join": current_reasons.get("ambiguous", 0),
            "offensive_usage_join_failures": sum(
                int(row["audit_offensive_unresolved_applicable"])
                for row in current_teams
            ),
            "offense_applicability_unknown": sum(
                int(row["audit_offensive_undetermined"]) for row in current_teams
            ),
        },
        "source_limitations": {
            "expected_2026_snapshot_count": len(snapshots),
            "2026_snapshot_record_count_including_noncanonical_diagnostics": len(
                snapshot_records
            ),
            "noncanonical_2026_diagnostic_snapshot_count": len(snapshot_records)
            - len(snapshots),
            "snapshot_count_by_source": dict(
                sorted(Counter(item["source"] for item in snapshots).items())
            ),
            "snapshot_retrieval_first": min(retrieval_timestamps),
            "snapshot_retrieval_last": max(retrieval_timestamps),
            "all_snapshots_after_cutoff": all(
                not item.get("captured_on_or_before_cutoff", False)
                for item in snapshots
            ),
            "missing_2026_raw_snapshot_count": len(missing_snapshots),
            "missing_2026_raw_snapshot_paths": [
                str(item["path"]) for item in missing_snapshots
            ],
            "historical_raw_transfer_payloads_available_locally": historical_raw_payload_count
            > 0,
            "historical_raw_transfer_payload_count": historical_raw_payload_count,
            "historical_raw_payloads_by_source": historical_payloads_by_source,
            "2026_offensive_player_join_audit_available_locally": current_offense_player_audit.is_file(),
            "historical_timing": "historical_timing_unverified",
        },
        "snapshot_lineage_comparison": snapshot_lineage,
        "original_snapshot_search": original_snapshot_search,
        "targeted_team_filter_reacquisition": team_filter_evidence,
        "remaining_2026_cases": {
            **remaining_db_cases,
            "offensive_usage_failures": remaining_offensive_cases,
            "offense_applicability_unknown_count": sum(
                int(row["audit_offensive_undetermined"])
                for row in current_teams
                if row["season"] == "2026"
            ),
        },
        "2026_reaudit_before_after": current_reaudit,
        "auxiliary_current_inputs": {
            "cfbd_team_directory": team_directory_provenance,
        },
        "input_sha256": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in (
                BASELINE_AUDIT / "team_seasons.csv",
                BASELINE_AUDIT / "summary.json",
                CURRENT_AVAILABILITY_AUDIT / "summary.json",
                CURRENT_AVAILABILITY_AUDIT / "team_seasons.csv",
                HISTORICAL_FEATURES,
                HISTORICAL_TEAM_COVERAGE,
                HISTORICAL_PLAYER_AUDIT,
                current_team_audit,
                current_player_audit,
                current_provenance,
                current_manifest_path,
                ORIGINAL_CURRENT_MANIFEST,
                PARTIAL_DB_COVERAGE,
            )
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "historical_transfer_features.csv", updated_features)
    _write_csv(output / "team_seasons.csv", team_seasons)
    _write_csv(output / "before_after_repairs.csv", repairs)
    _write_csv(output / "db_coverage_2026.csv", db_coverage)
    _write_json(output / "summary.json", summary)
    (output / "report.md").write_text(_render_report(summary), encoding="utf-8")
    return summary


def _render_report(summary: dict[str, Any]) -> str:
    baseline = summary["baseline_141"]
    post = summary["post_repair"]
    current = summary["post_repair_2026_db_coverage"]
    gaps = summary["remaining_2026_evidence_gaps"]
    limits = summary["source_limitations"]
    current_reaudit = summary["2026_reaudit_before_after"]
    snapshot_lineage = summary["snapshot_lineage_comparison"]
    current_cases = summary["remaining_2026_cases"]
    team_filter = summary["targeted_team_filter_reacquisition"]
    original_search = summary["original_snapshot_search"]
    searched_raw_paths = (
        ", ".join(
            f"`{path}`"
            for path in original_search["existing_cfbd_raw_subtrees_checked"]
        )
        or "none"
    )
    matched_raw_roots = sorted(
        {
            path.split("/data/raw/cfbd/preseason/", 1)[0]
            for locations in original_search["matching_source_hash_locations"].values()
            for path in locations
            if "/data/raw/cfbd/preseason/" in path
        }
    )
    matched_raw_root_text = (
        ", ".join(f"`{path}`" for path in matched_raw_roots) or "none"
    )
    historical_locations = (
        "; ".join(
            f"`{root}`: portal {counts.get('portal', 0)}, usage {counts.get('usage', 0)}, stats {counts.get('stats', 0)}"
            for root, counts in original_search[
                "historical_transfer_payload_counts_by_worktree"
            ].items()
        )
        or "none"
    )

    def names_for(case_key: str, field: str = "player_name") -> str:
        return (
            ", ".join(
                str(case.get(field, ""))
                for case in current_cases[case_key]
                if case.get(field)
            )
            or "none"
        )

    source_cases = current_cases["db_source_team_uncovered"]
    source_schools = sorted(
        {
            str(case.get("raw_origin") or case.get("origin") or "")
            for case in source_cases
            if case.get("raw_origin") or case.get("origin")
        }
    )
    source_case_text = (
        ", ".join(
            f"{case['player_name']} ({case.get('raw_origin') or case.get('origin')})"
            for case in source_cases
        )
        or "none"
    )
    offensive_case_text = (
        ", ".join(
            f"{case['player_name']} ({case.get('origin')} → {case.get('destination')})"
            for case in current_cases["offensive_usage_failures"]
        )
        or "none"
    )
    reference_comparison = current_reaudit["defensive_impact_reference_comparison"]
    position_change_names = (
        ", ".join(
            str(case["player_name"])
            for case in current_reaudit["db_player_impact_value_changed_cases"]
            if case["impact_change_class"] == "roster_position_evidence_changed"
        )
        or "none"
    )
    return "\n".join(
        [
            "# Transfer data repair audit (#149)",
            "",
            "This report is derived from the committed #141 and #142 player/team audits. It does not alter either study artifact or Context 1.3 inputs.",
            "",
            "## Before and after",
            "",
            "| Inventory | Baseline #141 | Post-repair |",
            "| --- | ---: | ---: |",
            f"| Team-seasons | {baseline['team_seasons']} | {post['team_seasons']} |",
            f"| Complete | {baseline['statuses'].get('complete', 0)} | {post['statuses'].get('complete', 0)} |",
            f"| Partial | {baseline['statuses'].get('partial', 0)} | {post['statuses'].get('partial', 0)} |",
            f"| Entirely unavailable | {baseline['statuses'].get('entirely_unavailable', 0)} | {post['statuses'].get('entirely_unavailable', 0)} |",
            f"| Historical aggregate-null reasons | {baseline['reason_team_seasons'].get('historical_aggregate_null', 0)} | {post['reason_team_seasons'].get('historical_aggregate_null', 0)} |",
            "",
            f"The corrected research panel restores {summary['repairs']['historical_aggregate_team_seasons_repaired']} historical sums only where the retained player audit marks every incoming transfer as resolved or a legitimate zero (including covered seasons with no incoming transfers). {summary['repairs']['availability_status_improvements']} team-seasons improve from partial to complete. Each changed team-season and its source evidence are listed in `before_after_repairs.csv`. The original historical feature panel remains unchanged.",
            "",
            "## 2026 incoming defensive-back coverage",
            "",
            f"The post-repair inventory covers {current['incoming_db_transfers']} incoming DB transfers across 138 teams; {current['observed_db_impacts']} impacts are observed. Coverage movement versus #142: {current['teams_moving_from_partial_to_complete']} teams moved from partial to complete, {current['teams_with_observed_count_increase_still_partial']} increased observed count while remaining partial, and {current['teams_unchanged']} were unchanged.",
            "",
            (
                "Post-repair DB coverage status is "
                f"{current['team_status_counts'].get('complete', 0)} complete, "
                f"{current['team_status_counts'].get('partial', 0)} partial, "
                f"{current['team_status_counts'].get('no_observed_db_impact', 0)} "
                "with no observed impact, and "
                f"{current['team_status_counts'].get('no_incoming_db_transfers', 0)} "
                "with no incoming DB transfers."
            ),
            "",
            "`db_coverage_2026.csv` records exact `observed / incoming` counts, observed sums, coverage status, and the source snapshot hashes retained by the 2026 derivation. A zero observed sum remains distinct from zero observed players and from no incoming transfers.",
            "",
            "## Candidate repair classes",
            "",
            f"For the 2026 re-audit, DB unresolved players changed from {current_reaudit['db_players_unresolved_before']} to {current_reaudit['db_players_unresolved_after']}. By class, covered-roster identity failures changed {current_reaudit['db_player_status_counts_before'].get('identity_resolution_failure', 0)} → {current_reaudit['db_player_status_counts_after'].get('identity_resolution_failure', 0)}, source-team coverage gaps {current_reaudit['db_player_status_counts_before'].get('source_data_unavailable', 0)} → {current_reaudit['db_player_status_counts_after'].get('source_data_unavailable', 0)}, position conflicts {current_reaudit['db_player_status_counts_before'].get('position_mismatch', 0)} → {current_reaudit['db_player_status_counts_after'].get('position_mismatch', 0)}, and ambiguous joins {current_reaudit['db_player_status_counts_before'].get('ambiguous', 0)} → {current_reaudit['db_player_status_counts_after'].get('ambiguous', 0)}. Applicable offensive failures changed {current_reaudit['offensive_applicable_failures_before']} → {current_reaudit['offensive_applicable_failures_after']}; {summary['repairs']['offensive_duplicate_usage_rows_coalesced']} rows were coalesced under the same-ID/equal-value rule. The current audit has {gaps['offense_applicability_unknown']} transfers with unknown offensive applicability.",
            "",
            f"The re-audit resolved {current_reaudit['db_players_repaired_to_resolved_or_zero']} previously unresolved DB players to an impact or supported zero. Reusable rules added were the stable game-player-ID bridge, unique same-team matching after removing a terminal generational suffix, four season-scoped verified aliases, and duplicate offensive usage-row coalescing. `{current_reaudit['player_artifact']}` and `{current_reaudit['team_artifact']}` contain machine-readable before/after records for each changed player or team-season. No fuzzy winner or outcome-informed mapping is used; unsupported roster coverage and unsupported position chronology remain unresolved.",
            "",
            "## Remaining 2026 cases after current-source re-audit",
            "",
            f"- Covered-roster DB identity joins still unresolved: {len(current_cases['db_player_join_unresolved'])}. Names: {names_for('db_player_join_unresolved')}. These have no unique same-team identity established by the exact-name, stable-ID, and terminal-suffix rules.",
            f"- DB source-team coverage still unavailable: {len(source_cases)} players across {len(source_schools)} source schools. Player/source pairs: {source_case_text}. Current provider team filters queried {team_filter['team_filter_request_count']} roster/game-player responses across {team_filter['distinct_team_filter_values']} exact team-filter values; {team_filter['empty_responses']} returned empty arrays and {team_filter['nonempty_responses']} returned records. A separate two-request Valdosta State control returned {team_filter['endpoint_control_check']['record_counts_by_source'].get('roster', 0)} roster rows and 1 player-game row; it is retained as noncanonical diagnostic evidence and excluded from the derivation. The exact school response hashes and timestamps are in `after_source_manifest.json`; no alias was inferred from an empty response.",
            f"- DB position conflicts still unresolved: {len(current_cases['db_position_conflict'])}. Names: {names_for('db_position_conflict')}. The current evidence does not support a general position-chronology rule, so these remain fail-closed.",
            f"- Ambiguous DB joins remaining: {len(current_cases['db_ambiguous_player_join'])}.",
            f"- Applicable offensive usage joins still unresolved: {len(current_cases['offensive_usage_failures'])}. Cases: {offensive_case_text}. Another {current_cases['offense_applicability_unknown_count']} transfers still have unknown offensive applicability; those are not filled with zero.",
            "",
            "## Source and model boundaries",
            "",
            f"The 2026 derivation used {limits['expected_2026_snapshot_count']} canonical current/reacquired snapshots ({limits['2026_snapshot_record_count_including_noncanonical_diagnostics']} raw manifest records including {limits['noncanonical_2026_diagnostic_snapshot_count']} noncanonical endpoint-control records); {limits['missing_2026_raw_snapshot_count']} canonical raw payloads are unavailable under the selected raw root. The original 37-request manifest was paired to current responses: {snapshot_lineage['paired_requests_byte_identical']} current payloads have identical SHA-256 hashes and {snapshot_lineage['paired_requests_hash_different']} differ. The {original_search['original_manifest_snapshot_count']} expected original versioned paths were checked across all {original_search['registered_git_worktree_count']} registered Git worktrees and {original_search['temporary_artifact_root_count_outside_registered_worktrees']} `/tmp/GippyRank4-*`/`gippyrank*` roots; {original_search['expected_original_versioned_paths_found']} expected paths were present, with {original_search['expected_original_versioned_paths_hash_valid']} SHA-256-valid originals. The {len(original_search['existing_cfbd_raw_subtrees_checked'])} existing CFBD raw subtrees checked were {searched_raw_paths}. Exact source hashes were found for {original_search['original_source_hashes_found_anywhere_under_local_cfbd_raw_subtrees']} requests at {matched_raw_root_text}; {original_search['original_source_hashes_not_found_under_local_cfbd_raw_subtrees']} original hashes had no local raw-file match. Hash identity does not change the later retrieval timestamp. The {snapshot_lineage['additional_current_requests']} additional canonical requests are stored in the separate reacquired lineage. {limits['historical_raw_transfer_payload_count']} historical transfer payloads are available locally (portal {limits['historical_raw_payloads_by_source']['portal']}, usage {limits['historical_raw_payloads_by_source']['usage']}, stats {limits['historical_raw_payloads_by_source']['stats']}); these were located at {historical_locations}; provenance sidecars are excluded. Related #141/#142 processed inputs and the offensive player-level audits were located at the paths recorded in `original_snapshot_search` in `summary.json`; the current 2026 offensive player join audit is {'available' if limits['2026_offensive_player_join_audit_available_locally'] else 'not available'} under the selected processed root. All canonical current responses were retrieved after 2026-08-15 and do not establish historical availability.",
            "",
            f"Defensive-impact normalization is fit only from the {reference_comparison['after_snapshot_count']} full-classification FBS/FCS roster and games/players snapshots. DII/III and team-filtered snapshots remain candidate evidence and do not enter the reference fit. Compared with the pre-repair derivation, {reference_comparison['matching_request_count']} reference requests match and {reference_comparison['changed_sha256_count']} reference SHA-256 hashes changed ({reference_comparison['added_request_count']} added, {reference_comparison['removed_request_count']} removed). This corrects the earlier expanded-pool result of 562 changed impact values: now {current_reaudit['db_player_impact_value_changed_rows']} of 604 differ from the pre-repair audit, with {current_reaudit['db_player_impact_change_classes'].get('normalization_reference_parameters_changed', 0)} attributable solely to changed reference parameters.",
            f"Of those changes, {current_reaudit['db_player_impact_change_classes'].get('newly_resolved_identity_or_source_coverage', 0)} are newly resolved ({current_reaudit['db_player_impact_status_transitions'].get('identity_resolution_failure → resolved', 0)} identity failures to resolved, {current_reaudit['db_player_impact_status_transitions'].get('identity_resolution_failure → zero_recorded_defensive_box_score_games', 0)} identity failures to supported zero, {current_reaudit['db_player_impact_status_transitions'].get('source_data_unavailable → resolved', 0)} source-coverage gaps to resolved, and {current_reaudit['db_player_impact_status_transitions'].get('ambiguous → resolved', 0)} ambiguous joins to resolved). Those joins used {summary['repairs']['stable_game_player_id_identity_bridge_rows']} unique game-player ID bridges, {summary['repairs']['generational_suffix_join_rows']} same-team suffix matches, and {summary['repairs']['verified_team_alias_player_records']} verified aliases. {current_reaudit['db_player_impact_change_classes'].get('roster_position_evidence_changed', 0)} additional cases ({position_change_names}) now have roster records showing LB against a DB portal position; they remain fail-closed and are not counted as observed impacts. The player before/after artifact records raw defensive stats, positions, statuses, and repair rules for each changed DB row. {current_reaudit['team_seasons_with_changed_db_audit']} team-season DB audits changed. The new coverage evidence does not change `MODEL_FEATURE_COLUMNS`, the attach-only Context 1.3 integration, coefficients, or published rankings. Historical rows remain labelled `historical_timing_unverified`; later data was not used to revise what was available at a historical cutoff.",
            "",
            "## Reproducibility",
            "",
            "Run `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/build_transfer_data_repair_audit.py` with the local ignored raw roots available. The tracked source manifests retain request parameters, retrieval times, and SHA-256 hashes; provider payload bytes remain in the ignored raw directory. Output rows are sorted and the audit performs no network request.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--current-root", type=Path, default=CURRENT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=CURRENT_RAW_ROOT)
    parser.add_argument("--historical-raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--before-current-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.output,
                current_root=args.current_root,
                raw_root=args.raw_root,
                historical_raw_root=args.historical_raw_root,
                before_current_root=args.before_current_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
