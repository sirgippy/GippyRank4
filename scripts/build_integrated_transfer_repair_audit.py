"""Build the canonical integrated #150 historical + #151 2026 repair audit.

The historical replay and current reacquisition builders remain independent
workstream audits. This command composes their authoritative outputs without
letting the reacquired player population replace the historical replay or the
FBS/FCS defensive-impact reference scale.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_transfer_data_repair as historical_builder
import build_transfer_data_repair_audit as current_builder

OUTPUT = ROOT / "data/processed/transfer_data_repair"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materializer_source_root(requested: Path | None) -> Path:
    required = Path("data/processed/modeling/team_season_rank_distributions.csv")
    candidates = [requested] if requested is not None else []
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            candidates.append(Path(line.removeprefix("worktree ")))
    candidates.append(ROOT)
    for candidate in candidates:
        if candidate is not None and (candidate / required).is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "the historical materializer requires "
        f"{required}; no registered worktree contains it"
    )


def _status_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["availability_status"] for row in rows)
    return {
        status: counts.get(status, 0)
        for status in ("complete", "partial", "entirely_unavailable")
    }


def _team_season_index(
    rows: list[dict[str, str]], label: str
) -> dict[tuple[int, str], dict[str, str]]:
    indexed: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        key = (int(row["season"]), str(row["team_id"]))
        if key in indexed:
            raise ValueError(f"duplicate {label} team-season row: {key}")
        indexed[key] = row
    return indexed


def _reason_codes(row: dict[str, str]) -> set[str]:
    return {code for code in row.get("reason_codes", "").split(";") if code}


def _player_evidence_for_team_season(
    season: int,
    team_id: str,
    player_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    evidence = []
    for row in player_rows:
        if (
            int(row["season"]) != season
            or str(row.get("destination_team_id", "")) != team_id
            or row.get("in_model_relevant_population") != "True"
            or row.get("d5_resolution_category") != "cannot_determine_applicability"
        ):
            continue
        evidence.append(
            {
                "portal_index": row.get("portal_index", ""),
                "player": row.get("player_name", ""),
                "position": row.get("position", ""),
                "usage_join_status": row.get("usage_join_status", ""),
                "usage_candidate_count": int(
                    row.get("usage_candidate_count", "0") or 0
                ),
                "applicability": row.get("d5_resolution_category", ""),
                "applicability_reason": row.get("d5_applicability_reason", ""),
            }
        )
    return sorted(evidence, key=lambda row: (row["player"], row["portal_index"]))


def _historical_status_reconciliation(
    pre_materializer_rows: list[dict[str, str]],
    final_rows: list[dict[str, str]],
    player_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    pre_index = _team_season_index(pre_materializer_rows, "pre-materializer")
    final_index = _team_season_index(final_rows, "final")
    if pre_index.keys() != final_index.keys():
        missing_final = sorted(pre_index.keys() - final_index.keys())
        missing_pre = sorted(final_index.keys() - pre_index.keys())
        raise ValueError(
            "historical team-season populations differ during reconciliation: "
            f"missing_final={missing_final}, missing_pre={missing_pre}"
        )

    changes = []
    for key in sorted(pre_index):
        before = pre_index[key]
        after = final_index[key]
        old_status = before["availability_status"]
        new_status = after["availability_status"]
        if old_status == new_status:
            continue
        season, team_id = key
        changes.append(
            {
                "season": season,
                "team_id": team_id,
                "team": after.get("team_name") or before.get("team_name", team_id),
                "pre_materializer_status": old_status,
                "final_status": new_status,
                "pre_materializer_primary_reason": before.get("primary_reason", ""),
                "final_primary_reason": after.get("primary_reason", ""),
                "pre_materializer_reason_codes": before.get("reason_codes", ""),
                "final_reason_codes": after.get("reason_codes", ""),
                "player_evidence": _player_evidence_for_team_season(
                    season, team_id, player_rows
                ),
            }
        )
    return changes


def _historical_aggregate_null_reconciliation(
    pre_materializer_rows: list[dict[str, str]],
    final_rows: list[dict[str, str]],
) -> dict[str, Any]:
    pre_index = _team_season_index(pre_materializer_rows, "pre-materializer")
    final_index = _team_season_index(final_rows, "final")
    if pre_index.keys() != final_index.keys():
        missing_final = sorted(pre_index.keys() - final_index.keys())
        missing_pre = sorted(final_index.keys() - pre_index.keys())
        raise ValueError(
            "historical team-season populations differ during aggregate-null "
            f"reconciliation: missing_final={missing_final}, missing_pre={missing_pre}"
        )

    removed: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []
    pre_count = 0
    final_count = 0
    for key in sorted(pre_index):
        before = pre_index[key]
        after = final_index[key]
        had_reason = "historical_aggregate_null" in _reason_codes(before)
        has_reason = "historical_aggregate_null" in _reason_codes(after)
        pre_count += had_reason
        final_count += has_reason
        if had_reason == has_reason:
            continue
        record = {
            "season": key[0],
            "team_id": key[1],
            "team": after.get("team_name") or before.get("team_name", key[1]),
            "pre_materializer_reason_codes": before.get("reason_codes", ""),
            "final_reason_codes": after.get("reason_codes", ""),
            "pre_materializer_usage_value": before.get(
                "post_repair_materialized_usage_value", ""
            ),
            "final_usage_value": after.get("post_repair_materialized_usage_value", ""),
        }
        (removed if had_reason else added).append(record)

    return {
        "pre_materializer_count": pre_count,
        "final_count": final_count,
        "removed_cases": removed,
        "added_cases": added,
        "net_reduction": pre_count - final_count,
    }


def _current_unresolved_db_players(current_root: Path) -> list[dict[str, Any]]:
    rows = _read_csv(current_root / "transfer_player_audit.csv")
    unresolved: list[dict[str, Any]] = []
    status_reasons = {
        "identity_resolution_failure": "db_player_join_unresolved",
        "source_data_unavailable": "db_source_team_uncovered",
        "position_mismatch": "db_position_conflict",
        "ambiguous": "db_ambiguous_player_join",
    }
    good = {"resolved", "zero_recorded_defensive_box_score_games"}
    for row in rows:
        status = row.get("impact_status", "")
        if (
            row.get("season") != "2026"
            or row.get("in_model_relevant_population") != "True"
            or row.get("portal_position_group") != "db"
            or status in good
        ):
            continue
        unresolved.append(
            {
                "season": row["season"],
                "portal_index": row["portal_index"],
                "player": row["player_name"],
                "source_team": row.get("origin", ""),
                "destination_team_id": row.get("destination_team_id", ""),
                "destination_team": row.get("destination_team_name", ""),
                "portal_position": row.get("position", ""),
                "prior_position": row.get("prior_position", ""),
                "portal_player_id": row.get("portal_player_id", ""),
                "prior_player_id": row.get("prior_player_id", ""),
                "old_status": status,
                "old_reason": status_reasons.get(status, "db_impact_unresolved"),
                "new_status": status,
                "repair_outcome": "unresolved_retained",
                "repair_rule": row.get("identity_resolution_detail")
                or row.get("identity_join_method")
                or row.get("impact_join_method")
                or "no_unique_evidence_resolved",
                "source_provenance": (
                    "2026 current/reacquired player audit; see after_source_manifest.json"
                ),
            }
        )
    return unresolved


def _historical_summary(
    source: dict[str, Any],
    preintegrated_source: dict[str, Any],
    preintegrated_rows: list[dict[str, str]],
    current_rows: list[dict[str, str]],
    historical_rows: list[dict[str, str]],
    historical_player_rows: list[dict[str, str]],
    output: Path,
) -> dict[str, Any]:
    years = [int(row["season"]) for row in historical_rows]
    if not years or min(years) != 2021 or max(years) != 2025:
        raise ValueError("integrated historical rows must cover 2021 through 2025")
    replay = source["historical_feature_reconciliation"]
    usage = source["historical_usage_resolution"]
    original_nulls = source["baseline_issue_141"]["reason_team_seasons"].get(
        "historical_aggregate_null", 0
    )
    repaired_nulls = sum(
        "historical_aggregate_null" in row.get("reason_codes", "").split(";")
        for row in historical_rows
    )
    integrated_total = [*historical_rows, *current_rows]
    preintegrated_post = preintegrated_source["post_repair"]
    preintegrated_historical_rows = [
        row for row in preintegrated_rows if int(row["season"]) < 2026
    ]
    preintegrated_status_counts = _status_counts(preintegrated_rows)
    summarized_pre_status_counts = {
        status: preintegrated_post["statuses"].get(status, 0)
        for status in ("complete", "partial", "entirely_unavailable")
    }
    if (
        len(preintegrated_rows) != preintegrated_post["team_seasons"]
        or preintegrated_status_counts != summarized_pre_status_counts
    ):
        raise ValueError(
            "pre-materializer team-season rows disagree with their summary: "
            f"rows={len(preintegrated_rows)}, summary={preintegrated_post['team_seasons']}, "
            f"row_statuses={preintegrated_status_counts}, "
            f"summary_statuses={summarized_pre_status_counts}"
        )
    status_changes = _historical_status_reconciliation(
        preintegrated_historical_rows, historical_rows, historical_player_rows
    )
    aggregate_null_reconciliation = _historical_aggregate_null_reconciliation(
        preintegrated_historical_rows, historical_rows
    )
    summarized_pre_nulls = preintegrated_post["reason_team_seasons"].get(
        "historical_aggregate_null", 0
    )
    if aggregate_null_reconciliation["pre_materializer_count"] != summarized_pre_nulls:
        raise ValueError(
            "pre-materializer aggregate-null rows disagree with their summary: "
            f"rows={aggregate_null_reconciliation['pre_materializer_count']}, "
            f"summary={summarized_pre_nulls}"
        )
    historical_feature_path = output / "historical_transfer_features.csv"
    return {
        "seasons": list(range(min(years), max(years) + 1)),
        "team_seasons": len(historical_rows),
        "availability_status_counts": _status_counts(historical_rows),
        "all_years_combined_checkpoint": {
            "team_seasons": len(integrated_total),
            "availability_status_counts": _status_counts(integrated_total),
        },
        "historical_aggregate_null_reasons": {
            "before": original_nulls,
            "after": repaired_nulls,
        },
        "reconciliation_with_151_pre_materializer_snapshot": {
            "pre_materializer_team_seasons": len(preintegrated_rows),
            "pre_materializer_availability_status_counts": preintegrated_status_counts,
            "pre_materializer_historical_aggregate_null_reasons": aggregate_null_reconciliation[
                "pre_materializer_count"
            ],
            "final_status_delta": {
                status: _status_counts(integrated_total).get(status, 0)
                - preintegrated_status_counts.get(status, 0)
                for status in ("complete", "partial", "entirely_unavailable")
            },
            "team_seasons_reclassified": status_changes,
            "historical_aggregate_null_reconciliation": aggregate_null_reconciliation,
        },
        "materializer_replay": {
            "legacy_replay_matches_frozen_panel": replay[
                "legacy_replay_matches_frozen_panel"
            ],
            "new_replay_matches_materializer": replay[
                "new_replay_matches_materializer"
            ],
            "change_inventory_exactly_matches_materializer_diff": replay[
                "change_inventory_exactly_matches_materializer_diff"
            ],
            "unexplained_changed_feature_values": replay[
                "unexplained_changed_feature_values"
            ],
            "changed_team_seasons": replay["changed_team_seasons"],
            "changed_feature_values": replay["changed_feature_values"],
            "feature_changes_by_class": replay["changed_feature_values_by_class"],
            "duplicate_equivalent_joins_resolved": replay[
                "duplicate_equivalent_ambiguous_joins_resolved"
            ],
            "remaining_ambiguous_usage_joins": usage["remaining_ambiguous_usage_joins"],
        },
        "player_usage_repairs": {
            "resolved_usage_joins": usage["resolved_usage_joins"],
            "normalization_recovered_usage_joins": usage[
                "normalization_recovered_usage_joins"
            ],
            "duplicate_equivalent_usage_joins": usage[
                "duplicate_equivalent_usage_joins"
            ],
            "legitimate_zero_evidence_players": usage[
                "legitimate_zero_evidence_players"
            ],
            "conflicting_usage_value_joins": usage["conflicting_usage_value_joins"],
        },
        "corrected_player_audit": source["corrected_historical_player_audit"],
        "canonical_feature_panel": {
            "path": "data/processed/transfer_data_repair/historical_transfer_features.csv",
            "sha256": _sha256(historical_feature_path),
            "row_count": sum(1 for _ in historical_feature_path.open()) - 1,
            "authority": "#150 replay/materializer parity result",
        },
        "source_manifest_sha256": replay["source_manifest_sha256"],
    }


def _current_summary(source: dict[str, Any]) -> dict[str, Any]:
    coverage = source["post_repair_2026_db_coverage"]
    gaps = source["remaining_2026_evidence_gaps"]
    cases = source["remaining_2026_cases"]
    deltas = source["2026_reaudit_before_after"]
    changes = deltas["db_player_impact_change_classes"]
    return {
        "incoming_db_transfers": coverage["incoming_db_transfers"],
        "observed_db_impacts": coverage["observed_db_impacts"],
        "unresolved_db_impacts": coverage["incoming_db_transfers"]
        - coverage["observed_db_impacts"],
        "unresolved_db_impacts_by_reason": {
            "identity_resolution_failure": gaps["db_player_join_unresolved"],
            "source_data_unavailable": gaps["db_source_team_uncovered"],
            "position_mismatch": gaps["db_position_conflict"],
            "ambiguous": gaps["db_ambiguous_player_join"],
        },
        "team_coverage_status_counts": coverage["team_status_counts"],
        "applicable_offensive_usage_failures_remaining": gaps[
            "offensive_usage_join_failures"
        ],
        "applicable_offensive_usage_failure_cases": cases["offensive_usage_failures"],
        "unknown_offensive_applicability_count": gaps["offense_applicability_unknown"],
        "changed_db_player_impact_values": {
            "total": deltas["db_player_impact_value_changed_rows"],
            "newly_resolved": changes.get(
                "newly_resolved_identity_or_source_coverage", 0
            ),
            "new_position_conflicts": changes.get(
                "roster_position_evidence_changed", 0
            ),
            "normalization_drift_only": changes.get(
                "normalization_reference_parameters_changed", 0
            ),
            "change_classes": changes,
            "changed_teams": deltas["team_seasons_with_changed_db_audit"],
        },
        "identity_repair_counts": {
            key: source["repairs"].get(key, 0)
            for key in (
                "stable_game_player_id_identity_bridge_rows",
                "generational_suffix_join_rows",
                "verified_team_alias_definitions",
                "verified_team_alias_player_records",
                "offensive_duplicate_usage_rows_coalesced",
            )
        },
        "defensive_impact_reference": deltas["defensive_impact_reference_comparison"],
        "source_lineage": {
            "before_manifest_sha256": deltas["before_source_manifest_sha256"],
            "after_manifest_sha256": deltas["after_source_manifest_sha256"],
            "snapshot_comparison": source["snapshot_lineage_comparison"],
            "source_limitations": source["source_limitations"],
        },
        "remaining_player_cases": {
            key: cases[key]
            for key in (
                "db_player_join_unresolved",
                "db_source_team_uncovered",
                "db_position_conflict",
                "db_ambiguous_player_join",
            )
        },
        "context_1_3_invariance": {
            "model_feature_columns_unchanged": True,
            "published_rankings_regenerated": False,
            "model_facing_inputs_changed": False,
        },
    }


def _render_report(summary: dict[str, Any]) -> str:
    historical = summary["historical"]
    current = summary["2026"]
    statuses = historical["availability_status_counts"]
    total_statuses = historical["all_years_combined_checkpoint"][
        "availability_status_counts"
    ]
    repair = historical["materializer_replay"]
    reconciliation = historical["reconciliation_with_151_pre_materializer_snapshot"]
    status_changes = reconciliation["team_seasons_reclassified"]
    aggregate_nulls = reconciliation["historical_aggregate_null_reconciliation"]
    impact = current["changed_db_player_impact_values"]
    coverage = current["team_coverage_status_counts"]
    reference = current["defensive_impact_reference"]
    source = current["source_lineage"]["source_limitations"]

    def row_reason_text(primary: str, codes: str) -> str:
        return (
            f"primary reason {f'`{primary}`' if primary else '`not recorded`'}, "
            f"reason codes {f'`{codes}`' if codes else '`none`'}"
        )

    status_change_text = []
    for change in status_changes:
        evidence = "; ".join(
            f"{item['player']} ({item['position']}): usage join "
            f"`{item['usage_join_status']}`, {item['usage_candidate_count']} usage "
            f"candidate(s), applicability `{item['applicability']}` "
            f"({item['applicability_reason']})"
            for item in change["player_evidence"]
        )
        detail = (
            f"{change['season']} {change['team']} changed from "
            f"`{change['pre_materializer_status']}` to `{change['final_status']}`; "
            f"row evidence changed from "
            f"{row_reason_text(change['pre_materializer_primary_reason'], change['pre_materializer_reason_codes'])} to "
            f"{row_reason_text(change['final_primary_reason'], change['final_reason_codes'])}"
        )
        if evidence:
            detail += f". Player-audit evidence: {evidence}"
        status_change_text.append(detail)
    if status_change_text:
        change_label = "change" if len(status_changes) == 1 else "changes"
        status_reconciliation_text = (
            f"Comparing the pre-materializer and final historical rows found "
            f"{len(status_changes)} status {change_label}: "
            + "; ".join(status_change_text)
            + "."
        )
    else:
        status_reconciliation_text = (
            "Comparing the pre-materializer and final historical rows found no "
            "availability status changes."
        )

    def null_case_text(case: dict[str, Any]) -> str:
        return (
            f"{case['season']} {case['team']} "
            f"(final usage `{case['final_usage_value']}`)"
        )

    removed_null_cases = (
        ", ".join(null_case_text(case) for case in aggregate_nulls["removed_cases"])
        or "none"
    )
    added_null_cases = (
        ", ".join(null_case_text(case) for case in aggregate_nulls["added_cases"])
        or "none"
    )
    aggregate_null_reconciliation_text = (
        "Comparing row-level historical aggregate-null reasons found "
        f"{aggregate_nulls['pre_materializer_count']} before and "
        f"{aggregate_nulls['final_count']} after; removed cases: "
        f"{removed_null_cases}; newly added cases: {added_null_cases}."
    )

    return "\n".join(
        [
            "# Integrated transfer repair audit (#150 + #151)",
            "",
            "This canonical audit combines #150's replayed historical repair with #151's 2026 reacquired evidence. Historical feature values and availability come from the corrected #150 materializer replay; 2026 candidate and coverage evidence comes from the #151 reacquisition. Reacquired sources were captured after the historical cutoff and are not represented as historical availability evidence.",
            "",
            "## Historical transfer audit (2021–2025)",
            "",
            f"The historical panel covers **{historical['team_seasons']}** team-seasons: {statuses['complete']} complete, {statuses['partial']} partial, and {statuses['entirely_unavailable']} entirely unavailable. Across the combined 2021–2026 availability inventory, the checkpoint is {total_statuses['complete']} complete, {total_statuses['partial']} partial, and {total_statuses['entirely_unavailable']} entirely unavailable across {historical['all_years_combined_checkpoint']['team_seasons']} team-seasons.",
            f"Historical aggregate-null reasons fall from {historical['historical_aggregate_null_reasons']['before']} to {historical['historical_aggregate_null_reasons']['after']}.",
            "",
            f"The legacy replay reproduces the frozen panel: **{repair['legacy_replay_matches_frozen_panel']}**. The current replay matches the materializer: **{repair['new_replay_matches_materializer']}**. The change inventory reconciles with no unexplained cells: **{repair['change_inventory_exactly_matches_materializer_diff']}**, {repair['changed_feature_values']} feature values across {repair['changed_team_seasons']} team-seasons.",
            "",
            "| Historical feature change class | Changed team-seasons / values |",
            "| --- | ---: |",
            *[
                f"| `{change_class}` | {count} |"
                for change_class, count in sorted(
                    repair["feature_changes_by_class"].items()
                )
            ],
            "",
            f"The player audit contains {historical['corrected_player_audit']['record_count']} records. It retains {repair['remaining_ambiguous_usage_joins']} genuinely ambiguous usage join; {repair['duplicate_equivalent_joins_resolved']} duplicate-equivalent usage joins collapse under #150's identity/value rules. Changed players and source evidence are listed in the historical repair artifacts.",
            "",
            f"Integration status reconciliation: the #151 pre-materializer snapshot had {reconciliation['pre_materializer_availability_status_counts']['complete']} complete, {reconciliation['pre_materializer_availability_status_counts']['partial']} partial, and {reconciliation['pre_materializer_availability_status_counts']['entirely_unavailable']} unavailable team-seasons; the final combined rows have {total_statuses['complete']} / {total_statuses['partial']} / {total_statuses['entirely_unavailable']}. {status_reconciliation_text}",
            aggregate_null_reconciliation_text,
            "",
            "## 2026 reacquired evidence",
            "",
            f"Incoming DB transfers: **{current['incoming_db_transfers']}**; observed impacts: **{current['observed_db_impacts']}**; unresolved impacts: **{current['unresolved_db_impacts']}**.",
            "",
            "| Unresolved DB impact reason | Players |",
            "| --- | ---: |",
            *[
                f"| `{reason}` | {count} |"
                for reason, count in sorted(
                    current["unresolved_db_impacts_by_reason"].items()
                )
            ],
            "",
            f"Among 138 teams, DB impact coverage is {coverage.get('complete', 0)} complete, {coverage.get('partial', 0)} partial, and {coverage.get('no_incoming_db_transfers', 0)} with no incoming DB transfers. Applicable offensive usage failures remaining: {current['applicable_offensive_usage_failures_remaining']}; offensive applicability remains unknown for {current['unknown_offensive_applicability_count']} transfers.",
            f"Compared with the pre-repair audit, {impact['total']} DB player impact records changed: {impact['newly_resolved']} became resolved and {impact['new_position_conflicts']} exposed position conflicts. {impact['normalization_drift_only']} changes came solely from normalization-reference drift. Changed impact classes: `{json.dumps(impact['change_classes'], sort_keys=True)}`.",
            f"Defensive-impact reference comparison: {reference.get('matching_request_count', 0)} matching requests; {reference.get('changed_sha256_count', 0)} reference hashes changed. Supplemental DII/III and team-filtered records are candidate evidence only. Context 1.3 model-facing inputs and published rankings remain unchanged: **{current['context_1_3_invariance']['model_facing_inputs_changed']}** changed and **{current['context_1_3_invariance']['published_rankings_regenerated']}** regenerated.",
            f"The reacquired lineage uses {source['expected_2026_snapshot_count']} canonical responses; {source['missing_2026_raw_snapshot_count']} are missing from the selected raw root. All were retrieved after the 2026-08-15 cutoff. Manifest hashes and request details are in `before_source_manifest.json` and `after_source_manifest.json`.",
            "",
            "## Canonical and supporting artifacts",
            "",
            "`historical_transfer_features.csv` is the canonical historical feature panel from #150's replay/materializer parity check. `historical_feature_changes.csv`, `historical_player_repair_audit.csv`, `changes.csv`, and `zero_contributors.csv` retain the detailed #150 evidence. `db_coverage_2026.csv`, `player_before_after_2026.csv`, `team_before_after_2026.csv`, `unresolved_db_players_2026.csv`, and the before/after source manifests retain #151's 2026 evidence. `historical_zero_candidate_details.csv` preserves #151's separate zero-candidate view; it is supporting evidence and does not define the canonical post-replay panel. `db_coverage_2026_pre_reacquisition.csv`, when present, is the older coverage snapshot and is not the current 2026 result.",
            "",
            "`team_seasons.csv` combines #150's 2021–2025 corrected availability rows with #151's 2026 reacquired rows. The #151 pre-materializer historical panel is not retained as a competing canonical panel.",
            "",
        ]
    )


def build(
    output: Path = OUTPUT,
    *,
    current_root: Path = current_builder.CURRENT_ROOT,
    raw_root: Path = current_builder.CURRENT_RAW_ROOT,
    historical_raw_root: Path = historical_builder.DEFAULT_HISTORICAL_RAW_ROOT,
    before_current_root: Path | None = None,
    materializer_source_root: Path | None = None,
) -> dict[str, Any]:
    materializer_root = _materializer_source_root(materializer_source_root)
    with tempfile.TemporaryDirectory(prefix="issue149-integrated-") as temporary:
        temp_root = Path(temporary)
        historical_dir = temp_root / "historical"
        current_dir = temp_root / "current"
        historical_summary = historical_builder.build(
            historical_dir,
            historical_raw_root=historical_raw_root,
            materializer_source_root=materializer_root,
        )
        current_summary = current_builder.run(
            current_dir,
            current_root=current_root,
            raw_root=raw_root,
            historical_raw_root=historical_raw_root,
            before_current_root=before_current_root,
        )

        historical_rows = [
            row
            for row in _read_csv(historical_dir / "team_seasons.csv")
            if int(row["season"]) < 2026
        ]
        preintegrated_rows = _read_csv(current_dir / "team_seasons.csv")
        current_rows = [row for row in preintegrated_rows if int(row["season"]) == 2026]
        historical_player_rows = _read_csv(
            historical_dir / "historical_player_repair_audit.csv"
        )
        if len(historical_rows) != 655 or len(current_rows) != 138:
            raise ValueError(
                "integrated team-season population changed: "
                f"historical={len(historical_rows)}, current={len(current_rows)}"
            )

        output.mkdir(parents=True, exist_ok=True)
        # Preserve #150's corrected materialized panel and distinct detailed
        # historical evidence, without copying its pre-reacquisition 2026 view.
        for name in (
            "historical_transfer_features.csv",
            "changes.csv",
            "zero_contributors.csv",
            "historical_feature_changes.csv",
            "historical_player_repair_audit.csv",
        ):
            shutil.copyfile(historical_dir / name, output / name)
        # Keep #151's distinct player/team and source-lineage evidence. Its
        # zero-candidate detail is renamed to make its supporting role explicit.
        for name in (
            "db_coverage_2026.csv",
            "player_before_after_2026.csv",
            "team_before_after_2026.csv",
            "before_source_manifest.json",
            "after_source_manifest.json",
        ):
            shutil.copyfile(current_dir / name, output / name)
        unresolved_players = _current_unresolved_db_players(current_root)
        _write_csv(output / "unresolved_db_players_2026.csv", unresolved_players)
        shutil.copyfile(
            current_dir / "before_after_repairs.csv",
            output / "historical_zero_candidate_details.csv",
        )
        stale_before_after = output / "before_after_repairs.csv"
        if stale_before_after.is_file():
            stale_before_after.unlink()
        pre_reacquisition_coverage = output / "coverage_2026.csv"
        if pre_reacquisition_coverage.is_file():
            os.replace(
                pre_reacquisition_coverage,
                output / "db_coverage_2026_pre_reacquisition.csv",
            )
        _write_csv(output / "team_seasons.csv", [*historical_rows, *current_rows])

        historical_section = _historical_summary(
            historical_summary,
            current_summary,
            preintegrated_rows,
            current_rows,
            historical_rows,
            historical_player_rows,
            output,
        )
        current_section = _current_summary(current_summary)
        integrated_summary = {
            "issue": 149,
            "integration": {
                "historical_authority": "PR #150 corrected replay/materializer parity and availability reconstruction",
                "2026_authority": "PR #151 reacquired source evidence, deterministic joins, coverage, and frozen impact reference",
                "historical_panel_artifact": "historical_transfer_features.csv",
                "team_season_artifact": "team_seasons.csv",
            },
            "historical": historical_section,
            "2026": current_section,
            "generator_input_sha256": {
                "historical": historical_summary["input_sha256"],
                "current_2026": current_summary["input_sha256"],
            },
            "context_1_3_invariance": current_section["context_1_3_invariance"],
        }
        report = _render_report(integrated_summary)
        _write_json(output / "summary.json", integrated_summary)
        (output / "report.md").write_text(report, encoding="utf-8")
        return integrated_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--current-root", type=Path, default=current_builder.CURRENT_ROOT
    )
    parser.add_argument(
        "--raw-root", type=Path, default=current_builder.CURRENT_RAW_ROOT
    )
    parser.add_argument(
        "--historical-raw-root",
        type=Path,
        default=historical_builder.DEFAULT_HISTORICAL_RAW_ROOT,
    )
    parser.add_argument("--before-current-root", type=Path)
    parser.add_argument("--materializer-source-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.output,
                current_root=args.current_root,
                raw_root=args.raw_root,
                historical_raw_root=args.historical_raw_root,
                before_current_root=args.before_current_root,
                materializer_source_root=args.materializer_source_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
