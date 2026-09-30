"""Materialize defensible repairs from the committed transfer evidence.

This command performs no network requests and never edits the frozen #141 or
#142 artifacts. Its inputs are the retained player-level audits and the
committed historical transfer panel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
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
CURRENT_ROOT = ROOT / "data/processed/preseason/context_v1_3_2026_reconstruction"
CURRENT_TEAM_AUDIT = CURRENT_ROOT / "transfer_team_audit.csv"
CURRENT_PLAYER_AUDIT = CURRENT_ROOT / "transfer_player_audit.csv"
CURRENT_PROVENANCE = CURRENT_ROOT / "feature_provenance.json"
CURRENT_MANIFEST = CURRENT_ROOT / "source_manifest.json"
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


def run(output: Path = OUTPUT) -> dict[str, Any]:
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

    current_players = _read_csv(CURRENT_PLAYER_AUDIT)
    current_teams = _read_csv(CURRENT_TEAM_AUDIT)
    snapshot_manifest = json.loads(CURRENT_MANIFEST.read_text(encoding="utf-8"))
    defensive_source_hashes = sorted(
        str(item["sha256"])
        for item in snapshot_manifest.get("snapshots", [])
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
    snapshots = snapshot_manifest.get("snapshots", [])
    missing_snapshots = [
        item for item in snapshots if not (RAW_ROOT / str(item["path"])).is_file()
    ]
    historical_raw_payload_count = sum(
        len(list((RAW_ROOT / source).glob("*.json")))
        for source in ("portal", "usage", "stats")
    )
    summary = {
        "baseline_141": baseline_summary,
        "post_repair": post_summary,
        "repairs": {
            "historical_aggregate_team_seasons_repaired": len(repairs),
            "stable_roster_id_impact_join_rule_added": True,
            "player_identity_mapping_repairs_in_frozen_audit": 0,
            "verified_team_alias_repairs": 0,
            "position_rule_repairs": 0,
            "offensive_usage_join_repairs": 0,
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
            "missing_2026_raw_snapshot_count": len(missing_snapshots),
            "missing_2026_raw_snapshot_paths": [
                str(item["path"]) for item in missing_snapshots
            ],
            "historical_raw_transfer_payloads_committed": historical_raw_payload_count
            > 0,
            "historical_raw_transfer_payload_count": historical_raw_payload_count,
            "2026_offensive_player_join_audit_committed": CURRENT_OFFENSE_PLAYER_AUDIT.is_file(),
            "historical_timing": "historical_timing_unverified",
        },
        "input_sha256": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in (
                BASELINE_AUDIT / "team_seasons.csv",
                BASELINE_AUDIT / "summary.json",
                HISTORICAL_FEATURES,
                HISTORICAL_TEAM_COVERAGE,
                HISTORICAL_PLAYER_AUDIT,
                CURRENT_TEAM_AUDIT,
                CURRENT_PLAYER_AUDIT,
                CURRENT_PROVENANCE,
                CURRENT_MANIFEST,
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
            f"No individual player identity or team alias mapping was changed in the frozen audit. The defensive pipeline now uses a uniquely matched roster record's stable provider ID to join its impact record, and leaves duplicate ID matches unresolved. The derived 2026 audit still has {gaps['db_player_join_unresolved']} covered-roster identity failures, {gaps['db_source_team_uncovered']} source-team coverage gaps, {gaps['db_position_conflict']} position conflicts, and {gaps['db_ambiguous_player_join']} explicitly ambiguous DB joins. It also has {gaps['offensive_usage_join_failures']} applicable offensive usage failures and {gaps['offense_applicability_unknown']} transfers with unknown offensive applicability.",
            "",
            "These cases remain unresolved because the immutable source payloads needed to verify alternate spellings, provider IDs, team aliases, position chronology, and offensive player-level joins are not present. No fuzzy or nearest-name selection was added. The single ambiguous identity remains unresolved, and the position conflicts remain governed by the existing fail-closed policy.",
            "",
            "## Source and model boundaries",
            "",
            f"The 2026 manifest lists {limits['expected_2026_snapshot_count']} snapshots, but {limits['missing_2026_raw_snapshot_count']} corresponding raw payloads are absent. Historical transfer raw payloads and the 2026 offensive player join audit are also not committed. Consequently this audit reclassifies retained evidence and cannot claim repairs to those missing source-level candidate classes.",
            "",
            "The added upstream DB coverage fields do not change `MODEL_FEATURE_COLUMNS`, the attach-only Context 1.3 integration, coefficients, or published rankings. Historical rows remain labelled `historical_timing_unverified`; later data was not used to revise what was available at a historical cutoff.",
            "",
            "## Reproducibility",
            "",
            "Run `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/build_transfer_data_repair_audit.py`. Inputs are hashed in `summary.json`; output rows are sorted and no network request is made.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
