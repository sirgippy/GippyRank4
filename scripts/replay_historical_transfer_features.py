"""Reconcile the historical Context 1.3 transfer panel across oracle versions.

The legacy join semantics are pinned to the merge base of issue 148. Current
semantics are run through the research materializer using the exact raw portal
and usage bytes whose hashes are retained in the historical audit manifest.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
SOURCE_MANIFEST = ROOT / "data/processed/transfer_production_audit/source_manifest.json"
HISTORICAL_PANEL = (
    ROOT
    / "data/processed/preseason/context_v1_3_candidate/historical_transfer_features.csv"
)
HISTORICAL_PLAYER_AUDIT = (
    ROOT / "data/processed/transfer_production_audit/player_join_records.csv"
)
ZERO_EVIDENCE_CATEGORY = "legitimate_zero_or_non_applicable_prior_offensive_usage"
TRANSFER_FEATURES = (
    "transfer_in_prior_usage_sum",
    "transfer_in_prior_defensive_impact_db_sum",
    "transfer_in_prior_defensive_impact_db_available",
)
USAGE_FEATURE = "transfer_in_prior_usage_sum"
LEGACY_ORACLE_COMMIT = "fc8c3924164c19fbb8b28334f613c5937c744673"
LEGACY_ORACLE_SOURCE_SHA256 = (
    "e2d66036102f7ac8bd11fbf95d913b688a0b283224b218db513a43b47d819c49"
)
REQUIRED_SOURCE_SEASONS = {
    "portal": set(range(2021, 2026)),
    "usage": set(range(2020, 2025)),
    "stats": set(range(2020, 2025)),
}

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gippyrank.transfer_oracle import (
    TransferRecord,
    UsageRecord,
    _matched_team,
    _matching_usage_candidates,
    _prior_usage_for_transfer,
    _team_index,
    _usage_index,
    available_by_cutoff,
    normalize_player_name,
    normalize_team_name,
    parse_transfer_payload,
    parse_usage_payload,
    transfer_identity_key,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv_with_columns(
    path: Path, rows: list[dict[str, Any]], columns: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def verify_historical_sources(
    raw_root: Path,
    *,
    manifest_path: Path = SOURCE_MANIFEST,
) -> tuple[dict[str, Any], dict[tuple[str, int], bytes]]:
    """Hash every manifested input before parsing any historical source data."""
    manifest_path = manifest_path.resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    items = manifest.get("files")
    if not isinstance(items, list) or not items:
        raise ValueError(f"historical source manifest has no files: {manifest_path}")

    present: dict[tuple[str, int], bytes] = {}
    metadata: list[dict[str, Any]] = []
    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    seen: set[tuple[str, int]] = set()
    raw_root = raw_root.resolve()

    # First pass hashes all entries. No JSON payload is parsed until every
    # expected input has passed this boundary.
    for item in items:
        kind = str(item.get("kind", ""))
        season = int(item["season"])
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe historical source path in manifest: {relative}")
        key = (kind, season)
        if key in seen:
            raise ValueError(f"duplicate historical source manifest key: {key}")
        seen.add(key)
        path = raw_root / relative
        if not path.is_file():
            missing.append(relative.as_posix())
            continue
        payload = path.read_bytes()
        expected = str(item.get("sha256", ""))
        actual = hashlib.sha256(payload).hexdigest()
        if not expected or actual != expected:
            mismatched.append(
                {"path": relative.as_posix(), "expected": expected, "actual": actual}
            )
            continue
        provenance_hash = (item.get("provenance") or {}).get("content_sha256")
        if provenance_hash and provenance_hash != expected:
            mismatched.append(
                {
                    "path": relative.as_posix(),
                    "expected": expected,
                    "actual": str(provenance_hash),
                }
            )
            continue
        record_count = int(item.get("record_count", -1))
        metadata.append(
            {
                "kind": kind,
                "season": season,
                "path": f"data/raw/{relative.as_posix()}",
                "sha256": actual,
                "record_count": record_count,
            }
        )
        if kind in {"portal", "usage"}:
            present[key] = payload

    missing_keys = {
        (kind, season)
        for kind, seasons in REQUIRED_SOURCE_SEASONS.items()
        for season in seasons
    } - seen
    missing.extend(f"manifest:{kind}:{season}" for kind, season in sorted(missing_keys))
    if missing or mismatched:
        details = {
            "missing": sorted(missing),
            "mismatched": sorted(mismatched, key=lambda row: row["path"]),
        }
        raise ValueError(
            "historical source verification failed closed: "
            + json.dumps(details, sort_keys=True)
        )

    entries_by_kind: dict[str, set[int]] = defaultdict(set)
    for kind, season in seen:
        entries_by_kind[kind].add(season)
    for kind, seasons in REQUIRED_SOURCE_SEASONS.items():
        absent = seasons - entries_by_kind[kind]
        if absent:
            raise ValueError(
                f"historical source manifest omits required {kind} seasons "
                f"{sorted(absent)}"
            )

    metadata.sort(key=lambda row: (str(row["kind"]), int(row["season"])))
    return (
        {
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "manifest_path": "data/processed/transfer_production_audit/source_manifest.json",
            "source_count": len(metadata),
            "sources": metadata,
        },
        present,
    )


def load_verified_transfer_data(
    manifest_summary: dict[str, Any],
    payloads: dict[tuple[str, int], bytes],
) -> tuple[list[TransferRecord], list[UsageRecord], set[int], set[int]]:
    """Parse only the portal and usage bytes returned by hash verification."""
    records: list[TransferRecord] = []
    usage: list[UsageRecord] = []
    portal_seasons: set[int] = set()
    usage_seasons: set[int] = set()
    source_items = {
        (str(item["kind"]), int(item["season"])): item
        for item in manifest_summary["sources"]
    }
    for (kind, season), payload in sorted(payloads.items()):
        decoded = json.loads(payload)
        if not isinstance(decoded, list):
            raise TypeError(f"historical {kind} payload is not a JSON array: {season}")
        source_item = source_items[(kind, season)]
        if len(decoded) != int(source_item["record_count"]):
            raise ValueError(
                f"historical {kind} record count disagrees with source manifest for "
                f"{season}: {len(decoded)} != {source_item['record_count']}"
            )
        if kind == "portal":
            records.extend(parse_transfer_payload(decoded, season=season))
            portal_seasons.add(season)
        elif kind == "usage":
            usage.extend(parse_usage_payload(decoded, season=season))
            usage_seasons.add(season)
        else:
            raise ValueError(f"unexpected parsed historical source kind: {kind}")
    return records, usage, portal_seasons, usage_seasons


def _legacy_normalize_player_name(value: str | None) -> str:
    """The player normalizer at the pinned pre-issue-148 baseline."""
    if not value:
        return ""
    return " ".join(value.casefold().replace("'", "").replace(".", "").split())


def _legacy_usage_index(
    usage: list[UsageRecord],
) -> tuple[
    dict[tuple[int, str, str], UsageRecord],
    dict[tuple[int, str, str], list[UsageRecord]],
]:
    """Reproduce the old first-non-null single-value dictionary behavior."""
    selected: dict[tuple[int, str, str], UsageRecord] = {}
    candidates: defaultdict[tuple[int, str, str], list[UsageRecord]] = defaultdict(list)
    for item in usage:
        if not item.player_name or not item.team:
            continue
        key = (
            item.season,
            normalize_team_name(item.team),
            _legacy_normalize_player_name(item.player_name),
        )
        candidates[key].append(item)
        if key not in selected or (
            selected[key].overall_usage is None and item.overall_usage is not None
        ):
            selected[key] = item
    return selected, candidates


def _value(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _values_equal(left: Any, right: Any) -> bool:
    left_value = _value(left)
    right_value = _value(right)
    if left_value is None or right_value is None:
        return left_value is None and right_value is None
    return math.isclose(left_value, right_value, rel_tol=1e-12, abs_tol=1e-12)


def _usage_candidate_detail(item: UsageRecord) -> dict[str, Any]:
    return {
        "season": item.season,
        "player": item.player_name,
        "team": item.team,
        "player_id": item.player_id,
        "overall_usage": item.overall_usage,
    }


def _verify_historical_portal_audit_bridge(
    records: list[TransferRecord], audit_path: Path
) -> int:
    """Bind every retained portal audit row, including position, to the bytes."""
    audit_rows = _read_csv(audit_path)
    if len(audit_rows) != len(records):
        raise ValueError(
            "historical portal audit row count does not match verified portal "
            f"payloads: {len(audit_rows)} != {len(records)}"
        )
    seen: set[int] = set()
    for row in audit_rows:
        portal_index = int(row["portal_index"])
        if portal_index < 0 or portal_index >= len(records) or portal_index in seen:
            raise ValueError(
                f"invalid or duplicate historical portal audit index {portal_index}"
            )
        seen.add(portal_index)
        record = records[portal_index]
        expected = {
            "season": str(record.season),
            "player_name": record.player_name,
            "origin": record.origin or "",
            "destination": record.destination or "",
            "position": record.position or "",
            "transfer_date": (
                record.transfer_date.isoformat() if record.transfer_date else ""
            ),
        }
        mismatched = {
            field: {"audit": row.get(field, ""), "source": value}
            for field, value in expected.items()
            if row.get(field, "") != value
        }
        if "portal_player_id" in row:
            expected_id = record.player_id or ""
            if row.get("portal_player_id", "") != expected_id:
                mismatched["portal_player_id"] = {
                    "audit": row.get("portal_player_id", ""),
                    "source": expected_id,
                }
        elif record.player_id:
            mismatched["portal_player_id"] = {
                "audit": "<field absent>",
                "source": record.player_id,
            }
        if mismatched:
            raise ValueError(
                "historical player audit does not identify the verified portal "
                f"record at index {portal_index}: {json.dumps(mismatched, sort_keys=True)}"
            )
    if seen != set(range(len(records))):
        raise ValueError("historical player audit does not cover every portal index")
    return len(audit_rows)


def _change_class(
    *,
    record: TransferRecord,
    old_value: float | None,
    new_value: float | None,
    new_candidates: list[UsageRecord],
    name_candidates: list[UsageRecord],
    id_candidates: list[UsageRecord],
    is_explicit_zero: bool,
) -> str:
    if is_explicit_zero:
        return "legitimate_zero_restoration"
    if len(new_candidates) > 1:
        return "ambiguous_usage_join_removed"
    if record.player_id and id_candidates:
        return "stable_player_id_resolution_changed"
    if record.player_id and name_candidates and not new_candidates:
        return "stable_player_id_conflict_removed"
    if old_value is None and new_value is not None:
        return "name_normalization_join_added"
    if old_value is not None and new_value is None:
        return "name_normalization_join_removed"
    if old_value is not None and new_value is not None:
        return "name_normalization_usage_changed"
    return "other_usage_resolution_change"


def _load_materializer():
    module_name = "materialize_preseason_context_prior_v1_3_research_features"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name, SCRIPT_DIR / f"{module_name}.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Context 1.3 research materializer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _panel_index(
    rows: list[dict[str, Any]], *, label: str
) -> dict[tuple[int, str, str], dict[str, Any]]:
    result = {}
    for row in rows:
        key = (int(row["season"]), str(row["subdivision"]), str(row["team_id"]))
        if key in result:
            raise ValueError(f"duplicate {label} panel team-season: {key}")
        result[key] = row
    return result


def replay_historical_materializer(
    *,
    model_source_root: Path,
    raw_root: Path,
    zero_contributors: list[dict[str, Any]],
    zero_evidence_sha256: str,
    zero_evidence_columns: list[str],
    expected_zero_repair_team_seasons: set[tuple[int, str]],
    historical_panel: Path = HISTORICAL_PANEL,
    historical_player_audit: Path = HISTORICAL_PLAYER_AUDIT,
    source_manifest_path: Path = SOURCE_MANIFEST,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare pinned legacy output with a materializer run on verified bytes."""
    source_summary, payloads = verify_historical_sources(
        raw_root, manifest_path=source_manifest_path
    )
    records, usage, portal_seasons, usage_seasons = load_verified_transfer_data(
        source_summary, payloads
    )
    portal_audit_row_count = _verify_historical_portal_audit_bridge(
        records, historical_player_audit
    )
    expected_portal = REQUIRED_SOURCE_SEASONS["portal"]
    expected_usage = REQUIRED_SOURCE_SEASONS["usage"]
    if portal_seasons != expected_portal or usage_seasons != expected_usage:
        raise ValueError(
            "verified historical payload seasons do not match the required panel "
            f"coverage: portal={sorted(portal_seasons)}, usage={sorted(usage_seasons)}"
        )

    if not zero_contributors:
        raise ValueError("historical replay requires retained legitimate-zero evidence")
    zero_indexes = [int(row["portal_index"]) for row in zero_contributors]
    if len(zero_indexes) != len(set(zero_indexes)):
        raise ValueError("duplicate portal index in legitimate-zero evidence")
    zero_evidence_by_index = {
        int(row["portal_index"]): row for row in zero_contributors
    }

    materializer = _load_materializer()
    with tempfile.TemporaryDirectory(prefix="issue148-historical-replay-") as temp:
        temp_root = Path(temp)
        evidence_path = temp_root / "zero_contributors.csv"
        _write_csv_with_columns(evidence_path, zero_contributors, zero_evidence_columns)
        written_evidence_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        if written_evidence_sha256 != zero_evidence_sha256:
            raise ValueError("serialized legitimate-zero evidence hash disagrees")
        new_rows = materializer.run(
            source_root=model_source_root,
            transfer_root=raw_root / "cfbd/preseason/transfers",
            output=temp_root / "after.csv",
            zero_evidence=evidence_path,
            verified_transfer_data=(records, usage, portal_seasons, usage_seasons),
        )

    old_rows = _read_csv(historical_panel)
    old_panel = _panel_index(old_rows, label="frozen")
    new_panel = _panel_index(new_rows, label="new materializer")
    if set(old_panel) != set(new_panel):
        missing = sorted(set(old_panel) - set(new_panel))
        extra = sorted(set(new_panel) - set(old_panel))
        raise ValueError(
            "historical materializer row population changed: "
            f"missing={missing[:10]!r}, extra={extra[:10]!r}"
        )

    # Re-run the previous resolver on these same parsed source records. The
    # frozen panel must match before any before/after inventory is accepted.
    legacy_selected, legacy_candidates = _legacy_usage_index(usage)
    current_usage_index = _usage_index(usage)
    teams = [
        {
            "season": int(row["season"]),
            "subdivision": str(row["subdivision"]),
            "team_id": str(row["team_id"]),
            "team_name": str(row["team_name"]),
            "returning_pct_ppa": None,
        }
        for row in new_rows
    ]
    team_index = _team_index(teams)
    # Materializer.run has already performed full field-level validation.
    zero_portal_indexes = set(zero_indexes)
    if any(index < 0 or index >= len(records) for index in zero_portal_indexes):
        raise ValueError("legitimate-zero portal index is out of range")
    zero_identity_keys = {
        transfer_identity_key(records[index]) for index in zero_portal_indexes
    }

    old_contributions: defaultdict[tuple[int, str, str], list[float | None]] = (
        defaultdict(list)
    )
    new_contributions: defaultdict[tuple[int, str, str], list[float | None]] = (
        defaultdict(list)
    )
    changed_players: defaultdict[tuple[int, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    current_classes: Counter[str] = Counter()
    portal_source = {
        int(item["season"]): item
        for item in source_summary["sources"]
        if item["kind"] == "portal"
    }
    usage_source = {
        int(item["season"]): item
        for item in source_summary["sources"]
        if item["kind"] == "usage"
    }
    stats_source = {
        int(item["season"]): item
        for item in source_summary["sources"]
        if item["kind"] == "stats"
    }

    for portal_index, record in enumerate(records):
        if record.season not in portal_seasons or not available_by_cutoff(
            record, date(2025, 8, 15)
        ):
            continue
        matched = _matched_team(team_index, record.season, record.destination)
        if not matched:
            continue
        team_key = (record.season, "fbs", str(matched[0]))
        if team_key not in new_panel:
            continue

        legacy_key = (
            record.season - 1,
            normalize_team_name(record.origin),
            _legacy_normalize_player_name(record.player_name),
        )
        legacy_item = legacy_selected.get(legacy_key) if record.origin else None
        old_value = legacy_item.overall_usage if legacy_item else None
        old_candidates = legacy_candidates.get(legacy_key, []) if record.origin else []

        name_key = (
            record.season - 1,
            normalize_team_name(record.origin),
            normalize_player_name(record.player_name),
        )
        name_candidates = (
            list(current_usage_index["by_name"].get(name_key, []))
            if record.origin
            else []
        )
        id_candidates = (
            list(
                current_usage_index["by_id"].get(
                    (
                        record.season - 1,
                        normalize_team_name(record.origin),
                        record.player_id,
                    ),
                    [],
                )
            )
            if record.origin and record.player_id
            else []
        )
        matched_candidates = _matching_usage_candidates(record, current_usage_index)
        new_value = _prior_usage_for_transfer(record, current_usage_index)
        is_explicit_zero = transfer_identity_key(record) in zero_identity_keys
        if is_explicit_zero:
            if any(
                candidate.overall_usage is not None and candidate.overall_usage > 0
                for candidate in matched_candidates
            ):
                raise ValueError(
                    "legitimate-zero evidence conflicts with positive usage for "
                    f"portal record {portal_index}"
                )
            new_value = 0.0

        old_contributions[team_key].append(old_value)
        new_contributions[team_key].append(new_value)
        if _values_equal(old_value, new_value):
            continue

        reason = _change_class(
            record=record,
            old_value=old_value,
            new_value=new_value,
            new_candidates=matched_candidates,
            name_candidates=name_candidates,
            id_candidates=id_candidates,
            is_explicit_zero=is_explicit_zero,
        )
        current_classes[reason] += 1
        changed_players[team_key].append(
            {
                "portal_index": portal_index,
                "player": record.player_name,
                "portal_player_id": record.player_id,
                "source_team": record.origin,
                "destination": record.destination,
                "portal_position": record.position,
                "transfer_date": (
                    record.transfer_date.isoformat() if record.transfer_date else ""
                ),
                "old_usage_contribution": old_value,
                "new_usage_contribution": new_value,
                "change_class": reason,
                "legacy_candidate_count": len(old_candidates),
                "legacy_selected_usage": (
                    _usage_candidate_detail(legacy_item) if legacy_item else None
                ),
                "current_name_candidate_count": len(name_candidates),
                "current_resolved_candidate_count": len(matched_candidates),
                "current_id_candidate_count": len(id_candidates),
                "current_usage_candidates": [
                    _usage_candidate_detail(candidate)
                    for candidate in matched_candidates
                ],
                "explicit_zero_evidence": is_explicit_zero,
                "zero_evidence_category": (
                    zero_evidence_by_index[portal_index].get("d5_resolution_category")
                    if is_explicit_zero
                    else None
                ),
                "zero_evidence_value": (
                    zero_evidence_by_index[portal_index].get("d5_feature_value")
                    if is_explicit_zero
                    else None
                ),
                "zero_evidence_provenance": (
                    zero_evidence_by_index[portal_index].get("source_provenance")
                    if is_explicit_zero
                    else None
                ),
            }
        )

    # Assert both implementations reproduce their respective feature panel.
    for key in old_panel:
        if key[0] not in portal_seasons:
            old_value = None
            new_value = None
        else:
            old_items = old_contributions.get(key, [])
            old_numeric = [value for value in old_items if value is not None]
            old_value = (
                float(sum(old_numeric))
                if old_numeric
                else (0.0 if not old_items else None)
            )
            new_items = new_contributions.get(key, [])
            new_numeric = [value for value in new_items if value is not None]
            new_value = (
                float(sum(new_numeric))
                if new_numeric
                else (0.0 if not new_items else None)
            )
        frozen_value = old_panel[key].get(USAGE_FEATURE)
        materialized_value = new_panel[key].get(USAGE_FEATURE)
        if not _values_equal(old_value, frozen_value):
            raise ValueError(
                "legacy replay does not reproduce the frozen historical panel for "
                f"{key}: replay={old_value!r}, frozen={frozen_value!r}"
            )
        if not _values_equal(new_value, materialized_value):
            raise ValueError(
                "player-level current replay does not reproduce the materializer "
                f"for {key}: replay={new_value!r}, materializer={materialized_value!r}"
            )

    audit_ambiguous = {
        (int(row["season"]), int(row["portal_index"]))
        for row in _read_csv(historical_player_audit)
        if row.get("in_model_relevant_population") == "True"
        and row.get("usage_join_status") == "ambiguous_usage_join"
        and int(row["season"]) in portal_seasons
    }
    replay_ambiguous = {
        (int(key[0]), int(player["portal_index"]))
        for key, players in changed_players.items()
        for player in players
        if player["change_class"] == "ambiguous_usage_join_removed"
    }
    if replay_ambiguous != audit_ambiguous:
        raise ValueError(
            "ambiguous historical usage changes do not reconcile to the retained "
            "player audit: "
            f"replay_only={sorted(replay_ambiguous - audit_ambiguous)}, "
            f"audit_only={sorted(audit_ambiguous - replay_ambiguous)}"
        )

    changes: list[dict[str, Any]] = []
    difference_keys: set[tuple[tuple[int, str, str], str]] = set()
    for key in sorted(old_panel):
        before = old_panel[key]
        after = new_panel[key]
        if str(before.get("team_name", "")) != str(after.get("team_name", "")):
            raise ValueError(f"historical materializer team name changed for {key}")
        for feature in TRANSFER_FEATURES:
            old_value = _value(before.get(feature))
            new_value = _value(after.get(feature))
            if _values_equal(old_value, new_value):
                continue
            difference_keys.add((key, feature))
            if feature != USAGE_FEATURE:
                raise ValueError(
                    "unexpected historical non-usage transfer feature change for "
                    f"{key} {feature}: {old_value!r} -> {new_value!r}"
                )
            responsible = sorted(
                changed_players.get(key, []),
                key=lambda player: int(player["portal_index"]),
            )
            classes = sorted({str(player["change_class"]) for player in responsible})
            if not responsible or not classes:
                raise ValueError(
                    f"historical materializer change has no player explanation: {key}"
                )
            change_class = (
                classes[0] if len(classes) == 1 else "mixed:" + ";".join(classes)
            )
            season = key[0]
            portal_item = portal_source.get(season)
            usage_item = usage_source.get(season - 1)
            stats_item = stats_source.get(season - 1)
            evidence_classes = {str(player["change_class"]) for player in responsible}
            provenance = {
                "source_manifest_path": source_summary["manifest_path"],
                "source_manifest_sha256": source_summary["manifest_sha256"],
                "portal_source_path": portal_item["path"] if portal_item else None,
                "portal_source_sha256": portal_item["sha256"] if portal_item else None,
                "usage_source_path": usage_item["path"] if usage_item else None,
                "usage_source_sha256": usage_item["sha256"] if usage_item else None,
                "stats_source_path": (
                    stats_item["path"]
                    if stats_item and "legitimate_zero_restoration" in evidence_classes
                    else None
                ),
                "stats_source_sha256": (
                    stats_item["sha256"]
                    if stats_item and "legitimate_zero_restoration" in evidence_classes
                    else None
                ),
                "player_audit_path": (
                    "data/processed/transfer_production_audit/player_join_records.csv"
                    if "legitimate_zero_restoration" in evidence_classes
                    else None
                ),
                "player_audit_sha256": (
                    hashlib.sha256(historical_player_audit.read_bytes()).hexdigest()
                    if "legitimate_zero_restoration" in evidence_classes
                    else None
                ),
                "zero_evidence_path": (
                    "data/processed/transfer_data_repair/zero_contributors.csv"
                    if "legitimate_zero_restoration" in evidence_classes
                    else None
                ),
                "zero_evidence_sha256": (
                    written_evidence_sha256
                    if "legitimate_zero_restoration" in evidence_classes
                    else None
                ),
                "legacy_oracle_commit": LEGACY_ORACLE_COMMIT,
                "legacy_oracle_source_sha256": LEGACY_ORACLE_SOURCE_SHA256,
            }
            changes.append(
                {
                    "season": season,
                    "team_id": key[2],
                    "team_name": str(after["team_name"]),
                    "feature_name": feature,
                    "old_value": "" if old_value is None else repr(old_value),
                    "new_value": "" if new_value is None else repr(new_value),
                    "change_class": change_class,
                    "players_responsible": json.dumps(
                        responsible, sort_keys=True, separators=(",", ":")
                    ),
                    "source_provenance": json.dumps(
                        provenance, sort_keys=True, separators=(",", ":")
                    ),
                }
            )

    inventory_keys = {
        (
            (int(row["season"]), "fbs", str(row["team_id"])),
            str(row["feature_name"]),
        )
        for row in changes
    }
    if difference_keys != inventory_keys:
        raise ValueError(
            "historical feature change inventory is incomplete: "
            f"unrecorded={sorted(difference_keys - inventory_keys)!r}, "
            f"extraneous={sorted(inventory_keys - difference_keys)!r}"
        )

    classes_by_value: Counter[str] = Counter()
    for row in changes:
        for reason in str(row["change_class"]).removeprefix("mixed:").split(";"):
            classes_by_value[reason] += 1
    changed_team_seasons = {
        (int(row["season"]), str(row["team_id"])) for row in changes
    }
    zero_teams = {
        (int(row["season"]), str(row["team_id"]))
        for row in changes
        if "legitimate_zero_restoration" in str(row["change_class"])
    }
    ambiguous_teams = {
        (int(row["season"]), str(row["team_id"]))
        for row in changes
        if "ambiguous_usage_join_removed" in str(row["change_class"])
    }
    if zero_teams & ambiguous_teams:
        raise ValueError(
            "legitimate-zero and ambiguous-join changes unexpectedly overlap"
        )
    if zero_teams != expected_zero_repair_team_seasons:
        raise ValueError(
            "legitimate-zero feature changes do not match the repair audit: "
            f"inventory_only={sorted(zero_teams - expected_zero_repair_team_seasons)}, "
            f"repair_only={sorted(expected_zero_repair_team_seasons - zero_teams)}"
        )
    if len(ambiguous_teams) != len(audit_ambiguous):
        raise ValueError(
            "ambiguous usage records do not affect distinct changed team-seasons: "
            f"records={len(audit_ambiguous)}, team_seasons={len(ambiguous_teams)}"
        )

    team_seasons_by_reason: defaultdict[str, set[tuple[int, str]]] = defaultdict(set)
    for row in changes:
        key = (int(row["season"]), str(row["team_id"]))
        for reason in str(row["change_class"]).removeprefix("mixed:").split(";"):
            team_seasons_by_reason[reason].add(key)

    feature_rows_compared = len(old_panel) * len(TRANSFER_FEATURES)
    summary = {
        "source_manifest_path": source_summary["manifest_path"],
        "source_manifest_sha256": source_summary["manifest_sha256"],
        "verified_source_count": source_summary["source_count"],
        "verified_sources": source_summary["sources"],
        "legacy_oracle_commit": LEGACY_ORACLE_COMMIT,
        "legacy_oracle_source_sha256": LEGACY_ORACLE_SOURCE_SHA256,
        "before_panel_path": "data/processed/preseason/context_v1_3_candidate/historical_transfer_features.csv",
        "before_panel_sha256": hashlib.sha256(
            historical_panel.read_bytes()
        ).hexdigest(),
        "materializer_panel_row_count": len(new_panel),
        "materializer_feature_values_compared": feature_rows_compared,
        "changed_team_seasons": len(changed_team_seasons),
        "changed_feature_values": len(changes),
        "changed_feature_values_by_class": dict(sorted(classes_by_value.items())),
        "player_contribution_changes_by_class": dict(sorted(current_classes.items())),
        "legitimate_zero_restorations": classes_by_value["legitimate_zero_restoration"],
        "legitimate_zero_contributor_players": len(zero_contributors),
        "zero_evidence_sha256": zero_evidence_sha256,
        "ambiguous_usage_join_removals": classes_by_value[
            "ambiguous_usage_join_removed"
        ],
        "other_change_classes": {
            key: value
            for key, value in sorted(classes_by_value.items())
            if key
            not in {
                "legitimate_zero_restoration",
                "ambiguous_usage_join_removed",
            }
        },
        "changed_team_seasons_by_reason": {
            reason: len(team_seasons)
            for reason, team_seasons in sorted(team_seasons_by_reason.items())
        },
        "ambiguous_player_audit_record_count": len(audit_ambiguous),
        "ambiguous_changed_transfer_count": len(replay_ambiguous),
        "portal_records_with_stable_ids": sum(
            bool(record.player_id) for record in records
        ),
        "portal_audit_rows_bound_to_verified_payloads": portal_audit_row_count,
        "portal_position_checked_against_verified_payloads": True,
        "legacy_replay_matches_frozen_panel": True,
        "new_replay_matches_materializer": True,
        "change_inventory_exactly_matches_materializer_diff": True,
        "unexplained_changed_feature_values": 0,
    }
    return changes, summary
