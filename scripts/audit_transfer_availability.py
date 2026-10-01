"""Reproduce the Context transfer-availability inventory from frozen audits.

This reads committed derived evidence only. It neither fetches source data nor
changes the Context feature panel or any published ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "data/processed/transfer_data_repair/availability_2026_reacquired_20260930"
)
HISTORICAL = (
    ROOT
    / "data/processed/preseason/context_v1_3_candidate/historical_transfer_features.csv"
)
HISTORICAL_OFFENSE = ROOT / "data/processed/transfer_production_audit"
HISTORICAL_DEFENSE = (
    ROOT / "data/processed/defensive_transfer_audit/transfer_player_audit.csv"
)
CURRENT = ROOT / "data/processed/preseason/context_v1_3_2026_reacquired_20260930"

# Each reason describes evidence observed in the committed audit, not an
# assertion that the upstream provider never had the player or transfer.
REASONS = {
    "offense_applicability_unproven": "Prior offensive participation cannot be determined from the audited usage and stats evidence.",
    "offense_usage_join_unresolved": "The applicable offensive transfer aggregate has no resolved usage and the selected player-level join audit has no row for it.",
    "offense_source_team_mismatch": "Current usage evidence has a same-name record under another source team but no supported match under this transfer's source team.",
    "offense_ambiguous_player_join": "More than one eligible current usage identity/value remains after exact duplicate rows with the same stable ID are coalesced.",
    "offense_usage_row_absent": "Current participation evidence establishes applicability, but no prior usage record is available for the source-team player identity.",
    "db_source_team_uncovered": "The current prior-season roster payload does not cover this source team after supported exact aliases are applied.",
    "db_player_join_unresolved": "The current prior-season source-team roster is covered, but this portal player did not join to exactly one roster identity.",
    "db_position_conflict": "The matched prior-season roster position group differs from the portal DB group; position chronology is not inferred.",
    "db_ambiguous_player_join": "More than one prior roster identity remains after stable IDs, exact team/name matches, and unique suffix normalization.",
    "db_impact_source_unavailable": "The player identity joined, but a required prior-season game-player or impact source was unavailable.",
    "historical_aggregate_null": "The retrospective research aggregate has no numeric offensive value despite a covered portal season.",
}
REPAIR_CLASS = {
    "offense_source_team_mismatch": "candidate_repair",
    "offense_ambiguous_player_join": "candidate_repair",
    "offense_usage_row_absent": "unresolved",
    "offense_usage_join_unresolved": "unresolved",
    "offense_applicability_unproven": "unresolved",
    "db_source_team_uncovered": "unresolved",
    "db_player_join_unresolved": "candidate_repair",
    "db_position_conflict": "unresolved",
    "db_ambiguous_player_join": "candidate_repair",
    "db_impact_source_unavailable": "unresolved",
    "historical_aggregate_null": "candidate_repair",
}
DB_STATUS = {
    "source_data_unavailable": "db_source_team_uncovered",
    "identity_resolution_failure": "db_player_join_unresolved",
    "position_mismatch": "db_position_conflict",
    "ambiguous": "db_ambiguous_player_join",
}
GOOD_DB = {"resolved", "zero_recorded_defensive_box_score_games"}
OFFENSE_FAILURE = {
    "source_team_mismatch": "offense_source_team_mismatch",
    "ambiguous_usage_join": "offense_ambiguous_player_join",
    "no_usage_record": "offense_usage_row_absent",
}
PRIMARY_PRIORITY = (
    "db_source_team_uncovered",
    "db_player_join_unresolved",
    "db_ambiguous_player_join",
    "db_position_conflict",
    "db_impact_source_unavailable",
    "offense_source_team_mismatch",
    "offense_ambiguous_player_join",
    "offense_usage_row_absent",
    "offense_usage_join_unresolved",
    "offense_applicability_unproven",
    "historical_aggregate_null",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def index_rows(path: Path) -> dict[tuple[int, str], dict[str, str]]:
    result = {}
    for row in read_csv(path):
        key = (int(row["season"]), row["team_id"])
        if key in result:
            raise ValueError(f"duplicate team-season in {path}: {key}")
        result[key] = row
    return result


def player_index(
    path: Path, *, db_only: bool = False
) -> dict[tuple[int, str], list[dict[str, str]]]:
    result: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        if row["in_model_relevant_population"] != "True":
            continue
        if db_only and row["portal_position_group"] != "db":
            continue
        result[(int(row["season"]), row["destination_team_id"])].append(row)
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def classify_availability(
    codes: set[str],
    *,
    incoming: int,
    offensive_resolved: int,
    offensive_zero: int,
    db_failed: int,
    db_resolved: int,
) -> dict[str, object]:
    """Apply the shared team-season availability classification rules."""
    unknown_codes = codes - REPAIR_CLASS.keys()
    if unknown_codes:
        raise ValueError(f"unknown availability reason codes: {sorted(unknown_codes)}")
    if not codes:
        status = "complete"
    elif (
        incoming
        and offensive_resolved == 0
        and offensive_zero == 0
        and db_failed > 0
        and db_resolved == 0
    ):
        status = "entirely_unavailable"
    else:
        status = "partial"
    categories = {REPAIR_CLASS[code] for code in codes}
    if not codes:
        repair = "not_applicable"
    elif categories == {"candidate_repair"}:
        repair = "candidate_repair"
    else:
        repair = "unresolved"
    ordered = sorted(codes)
    primary = next((code for code in PRIMARY_PRIORITY if code in codes), "complete")
    repair_candidates = [
        code for code in ordered if REPAIR_CLASS[code] == "candidate_repair"
    ]
    return {
        "availability_status": status,
        "primary_reason": primary,
        "reason_codes": ";".join(ordered),
        "repair_class": repair,
        "repair_candidate_reasons": ";".join(repair_candidates),
        "source_coverage_gap": "db_source_team_uncovered" in codes,
    }


def audit(
    current_root: Path = CURRENT,
    *,
    historical_features_path: Path = HISTORICAL,
    historical_offense_root: Path = HISTORICAL_OFFENSE,
    historical_defense_path: Path = HISTORICAL_DEFENSE,
    expected_current_team_count: int | None = 138,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build availability rows from explicitly selected audit inputs.

    The defaults support the local research command. Tests and other callers
    can inject a tracked fixture root and historical source paths without
    relying on ignored generated data.
    """
    historical = {
        key: value
        for key, value in index_rows(historical_features_path).items()
        if 2021 <= key[0] <= 2025
    }
    current = index_rows(current_root / "transfer_team_audit.csv")
    if (
        expected_current_team_count is not None
        and len(current) != expected_current_team_count
    ) or {season for season, _ in current} != {2026}:
        raise ValueError("2026 Context transfer population changed; inspect inputs")
    historical_offense = index_rows(
        historical_offense_root / "team_feature_coverage.csv"
    )
    historical_offense_players = player_index(
        historical_offense_root / "player_join_records.csv"
    )
    historical_db_players = player_index(historical_defense_path, db_only=True)
    current_db_players = player_index(
        current_root / "transfer_player_audit.csv", db_only=True
    )
    current_portal_rows = read_csv(current_root / "transfer_player_audit.csv")
    current_offense_players = player_index(
        current_root / "offensive_player_join_records.csv"
    )
    excluded_portal_rows = [
        row
        for row in current_portal_rows
        if row["in_model_relevant_population"] != "True"
    ]
    snapshot = json.loads(
        (current_root / "source_manifest.json").read_text(encoding="utf-8")
    )
    snapshot_records = snapshot["snapshots"]
    snapshot_rows = [item for item in snapshot_records if item.get("canonical", True)]
    if any(item["captured_on_or_before_cutoff"] for item in snapshot_rows):
        raise ValueError("2026 snapshot timing changed; inspect inputs")
    retrievals = [item["retrieval_timestamp"] for item in snapshot_rows]
    rows: list[dict[str, object]] = []
    for key, source in sorted({**historical, **current}.items()):
        season, team_id = key
        is_current = season == 2026
        offense = source if is_current else historical_offense[key]
        db_players = (current_db_players if is_current else historical_db_players)[key]
        offense_players = (
            current_offense_players[key]
            if is_current
            else historical_offense_players[key]
        )
        incoming = int(
            offense[
                "audit_incoming_fbs_transfers"
                if is_current
                else "incoming_transfer_count"
            ]
        )
        offensive_resolved = int(
            offense[
                "audit_offensive_resolved_applicable"
                if is_current
                else "d5_successfully_resolved_count"
            ]
        )
        offensive_zero = int(
            offense[
                "audit_offensive_legitimate_zero_non_applicable"
                if is_current
                else "d5_legitimate_zero_or_non_applicable_count"
            ]
        )
        offensive_failed = int(
            offense[
                "audit_offensive_unresolved_applicable"
                if is_current
                else "d5_resolution_failure_count"
            ]
        )
        offensive_unknown = int(
            offense[
                "audit_offensive_undetermined"
                if is_current
                else "d5_applicability_unknown_count"
            ]
        )
        if (
            incoming
            != offensive_resolved
            + offensive_zero
            + offensive_failed
            + offensive_unknown
        ):
            raise ValueError(f"offensive classification does not reconcile: {key}")
        db_resolved = sum(row["impact_status"] in GOOD_DB for row in db_players)
        db_failed = len(db_players) - db_resolved
        db_available = source["transfer_in_prior_defensive_impact_db_available"]
        if (float(db_available) == 1) != (db_failed == 0):
            raise ValueError(f"DB availability does not reconcile: {key}")
        codes: set[str] = set()
        if offensive_unknown:
            codes.add("offense_applicability_unproven")
        if is_current and offensive_failed and not offense_players:
            codes.add("offense_usage_join_unresolved")
        for player in offense_players:
            if (
                player["d5_resolution_category"]
                == "should_have_recoverable_offensive_usage_but_resolution_failed"
            ):
                codes.add(
                    OFFENSE_FAILURE.get(
                        player["usage_join_status"], "offense_usage_join_unresolved"
                    )
                )
        for player in db_players:
            if player["impact_status"] in GOOD_DB:
                continue
            codes.add(
                DB_STATUS.get(player["impact_status"], "db_impact_source_unavailable")
            )
        usage_value = source["transfer_in_prior_usage_sum"]
        if not is_current and not usage_value:
            codes.add("historical_aggregate_null")
        classification = classify_availability(
            codes,
            incoming=incoming,
            offensive_resolved=offensive_resolved,
            offensive_zero=offensive_zero,
            db_failed=db_failed,
            db_resolved=db_resolved,
        )
        rows.append(
            {
                "season": season,
                "subdivision": "fbs",
                "team_id": team_id,
                "team_name": source["team_name"],
                **classification,
                "checkpoint_status": "no_archived_on_time_snapshot"
                if is_current
                else "historical_timing_unverified",
                "provenance_class": "reacquired_current_provider_research"
                if is_current
                else "retrospective_research_reconstruction",
                "incoming_transfers": incoming,
                "offensive_resolved": offensive_resolved,
                "offensive_legitimate_zero": offensive_zero,
                "offensive_join_failed": offensive_failed,
                "offensive_applicability_unknown": offensive_unknown,
                "db_incoming": len(db_players),
                "db_resolved": db_resolved,
                "db_unresolved": db_failed,
                "offensive_feature_numeric": bool(usage_value),
                "db_feature_available": db_failed == 0,
            }
        )
    by_season = {}
    for season in sorted({row["season"] for row in rows}):
        subset = [row for row in rows if row["season"] == season]
        by_season[str(season)] = {
            "team_seasons": len(subset),
            "statuses": dict(
                sorted(Counter(row["availability_status"] for row in subset).items())
            ),
            "affected": sum(row["availability_status"] != "complete" for row in subset),
        }
    summary = {
        "population": "Context FBS team-seasons, 2021-2026; historical research panel plus 2026 reacquired beta evidence",
        "team_seasons": len(rows),
        "statuses": dict(
            sorted(Counter(row["availability_status"] for row in rows).items())
        ),
        "reason_team_seasons": dict(
            sorted(
                Counter(
                    code
                    for row in rows
                    for code in str(row["reason_codes"]).split(";")
                    if code
                ).items()
            )
        ),
        "repair_classes": dict(
            sorted(Counter(row["repair_class"] for row in rows).items())
        ),
        "team_seasons_with_repair_candidate": sum(
            bool(row["repair_candidate_reasons"]) for row in rows
        ),
        "team_seasons_with_source_coverage_gap": sum(
            bool(row["source_coverage_gap"]) for row in rows
        ),
        "by_season": by_season,
        "current_2026": {
            "team_seasons": len(current),
            "affected": by_season["2026"]["affected"],
            "incoming_transfers": sum(
                int(row["incoming_transfers"]) for row in rows if row["season"] == 2026
            ),
            "portal_records": len(current_portal_rows),
            "excluded_portal_records": len(excluded_portal_rows),
            "excluded_destination_statuses": dict(
                sorted(
                    Counter(
                        row["destination_match_status"] for row in excluded_portal_rows
                    ).items()
                )
            ),
            "offensive_join_failed": sum(
                int(row["offensive_join_failed"])
                for row in rows
                if row["season"] == 2026
            ),
            "offensive_applicability_unknown": sum(
                int(row["offensive_applicability_unknown"])
                for row in rows
                if row["season"] == 2026
            ),
            "db_unresolved": sum(
                int(row["db_unresolved"]) for row in rows if row["season"] == 2026
            ),
            "teams_with_db_unresolved": sum(
                int(row["db_unresolved"]) > 0 for row in rows if row["season"] == 2026
            ),
            "teams_with_offensive_join_failed": sum(
                int(row["offensive_join_failed"]) > 0
                for row in rows
                if row["season"] == 2026
            ),
            "teams_with_offensive_applicability_unknown": sum(
                int(row["offensive_applicability_unknown"]) > 0
                for row in rows
                if row["season"] == 2026
            ),
            "teams_with_repair_candidate": sum(
                bool(row["repair_candidate_reasons"])
                for row in rows
                if row["season"] == 2026
            ),
            "teams_with_source_coverage_gap": sum(
                bool(row["source_coverage_gap"])
                for row in rows
                if row["season"] == 2026
            ),
            "db_unresolved_player_causes": dict(
                sorted(
                    Counter(
                        DB_STATUS.get(
                            player["impact_status"], "db_impact_source_unavailable"
                        )
                        for players in current_db_players.values()
                        for player in players
                        if player["impact_status"] not in GOOD_DB
                    ).items()
                )
            ),
            "snapshot_count": len(snapshot_rows),
            "snapshot_record_count_including_noncanonical_diagnostics": len(
                snapshot_records
            ),
            "noncanonical_diagnostic_snapshot_count": len(snapshot_records)
            - len(snapshot_rows),
            "snapshot_count_by_source": dict(
                sorted(Counter(item["source"] for item in snapshot_rows).items())
            ),
            "retrieval_first": min(retrievals),
            "retrieval_last": max(retrievals),
            "all_snapshots_after_cutoff": True,
            "evidence_class": snapshot.get("evidence_class", "retained_reconstruction"),
        },
        "definitions": {
            "complete": "All known incoming transfers are resolved or legitimately zero, including DB; no offensive applicability is unknown. A covered portal with no incoming transfers is complete.",
            "partial": "A transfer input or retrospective aggregate has a gap, but a player-level offensive or DB contribution or covered natural-zero DB feature is known.",
            "entirely_unavailable": "Incoming transfers exist, no applicable offensive transfer resolves, no legitimate offensive zero is established, at least one incoming DB transfer is unresolved, and no incoming DB impact resolves.",
            "checkpoint_status": "Independent of present-day completeness; no historical on-time source snapshot is proved by these research artifacts, so individual late arrivals cannot be separated from pre-cutoff data lost in processing.",
        },
    }
    return rows, summary


def write_artifacts(
    output: Path,
    rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    """Write the deterministic availability artifact set."""
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "team_seasons.csv", rows)
    write_csv(
        output / "affected_2026.csv",
        [
            row
            for row in rows
            if row["season"] == 2026 and row["availability_status"] != "complete"
        ],
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "reason_taxonomy.json").write_text(
        json.dumps(
            {
                "reason_codes": {
                    code: {"definition": definition, "repair_class": REPAIR_CLASS[code]}
                    for code, definition in sorted(REASONS.items())
                },
                "repair_class_definitions": {
                    "candidate_repair": "A pipeline or identity-control repair is plausible from the committed evidence, subject to source confirmation.",
                    "unresolved": "The committed derived artifacts cannot establish whether source data or a safe repair exists.",
                    "not_applicable": "No gap observed.",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--current-root", type=Path, default=CURRENT)
    args = parser.parse_args()
    rows, summary = audit(args.current_root)
    write_artifacts(args.output, rows, summary)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "affected_2026": summary["current_2026"]["affected"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
