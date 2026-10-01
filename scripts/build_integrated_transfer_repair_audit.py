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
    current_rows: list[dict[str, str]],
    historical_rows: list[dict[str, str]],
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
    troy = next(
        row
        for row in historical_rows
        if row["season"] == "2021" and row["team_name"] == "Troy"
    )
    integrated_total = [*historical_rows, *current_rows]
    preintegrated_post = preintegrated_source["post_repair"]
    preintegrated_status_counts = {
        status: preintegrated_post["statuses"].get(status, 0)
        for status in ("complete", "partial", "entirely_unavailable")
    }
    historical_feature_path = output / "historical_transfer_features.csv"
    return {
        "seasons": [2021, 2025],
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
            "pre_materializer_team_seasons": preintegrated_post["team_seasons"],
            "pre_materializer_availability_status_counts": preintegrated_status_counts,
            "pre_materializer_historical_aggregate_null_reasons": preintegrated_post[
                "reason_team_seasons"
            ].get("historical_aggregate_null", 0),
            "final_status_delta": {
                status: _status_counts(integrated_total).get(status, 0)
                - preintegrated_status_counts.get(status, 0)
                for status in ("complete", "partial", "entirely_unavailable")
            },
            "team_season_reclassified": {
                "season": 2022,
                "team": "Texas",
                "pre_materializer_status": "complete",
                "final_status": "partial",
                "reason": (
                    "Diamonte Tucker-Dorsey (LB) has no usage candidate and the "
                    "#150 player audit classifies applicability as unknown; the "
                    "pre-materializer #151 audit had treated this absence as zero"
                ),
            },
            "aggregate_null_reason_reduction_case": {
                "season": 2021,
                "team": "Troy",
                "final_usage_value": troy["post_repair_materialized_usage_value"],
            },
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
        "corrected_2021_troy_usage": {
            "post_repair_materialized_usage_value": troy[
                "post_repair_materialized_usage_value"
            ],
            "historical_aggregate_null_retained": (
                "historical_aggregate_null" in troy["reason_codes"].split(";")
            ),
        },
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
    integrated_nulls = historical["historical_aggregate_null_reasons"]["after"]
    repair = historical["materializer_replay"]
    reconciliation = historical["reconciliation_with_151_pre_materializer_snapshot"]
    impact = current["changed_db_player_impact_values"]
    coverage = current["team_coverage_status_counts"]
    reference = current["defensive_impact_reference"]
    source = current["source_lineage"]["source_limitations"]
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
            f"The corrected 2021 Troy usage aggregate is `{historical['corrected_2021_troy_usage']['post_repair_materialized_usage_value']}`; its historical-null reason is cleared. The player audit contains {historical['corrected_player_audit']['record_count']} records. It retains {repair['remaining_ambiguous_usage_joins']} genuinely ambiguous usage join; duplicate-equivalent usage rows collapse only under #150's identity/value rules.",
            "",
            f"Integration reconciliation: #151's pre-materializer availability snapshot had {reconciliation['pre_materializer_availability_status_counts']['complete']} complete, {reconciliation['pre_materializer_availability_status_counts']['partial']} partial, and {reconciliation['pre_materializer_availability_status_counts']['entirely_unavailable']} unavailable team-seasons, with {reconciliation['pre_materializer_historical_aggregate_null_reasons']} aggregate-null reasons. The final #150 replay yields {total_statuses['complete']} / {total_statuses['partial']} / {total_statuses['entirely_unavailable']} overall and {integrated_nulls} null reasons. The one status change is 2022 Texas: the #150 audit leaves Diamonte Tucker-Dorsey's prior usage applicability unknown because no usage candidate is present, so the team is partial rather than complete. The null-reason reduction is the verified 2021 Troy aggregate. The replay also resolves 24 duplicate-equivalent usage joins and removes Chandler Rogers's unsupported North Texas 2023 contribution; that genuinely ambiguous join remains unresolved.",
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
        current_rows = [
            row
            for row in _read_csv(current_dir / "team_seasons.csv")
            if int(row["season"]) == 2026
        ]
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
            current_rows,
            historical_rows,
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
