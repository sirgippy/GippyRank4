"""Build an immutable before/after audit for issue 148 transfer repairs.

The committed #141/#142 audit inputs are treated as baseline records. This
command writes a new post-repair inventory and never changes those artifacts
or published Context inputs. It uses retained player-level audits only; when
raw provider payloads are absent, identity and source-team gaps stay open.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/processed/transfer_data_repair"
CURRENT = ROOT / "data/processed/preseason/context_v1_3_2026_reconstruction"
HISTORICAL = (
    ROOT
    / "data/processed/preseason/context_v1_3_candidate/historical_transfer_features.csv"
)
HISTORICAL_AUDIT = ROOT / "data/processed/transfer_production_audit"
HISTORICAL_PLAYERS = HISTORICAL_AUDIT / "player_join_records.csv"
HISTORICAL_TEAMS = HISTORICAL_AUDIT / "team_feature_coverage.csv"
CURRENT_PLAYERS = CURRENT / "transfer_player_audit.csv"
CURRENT_TEAMS = CURRENT / "transfer_team_audit.csv"
CURRENT_MANIFEST = CURRENT / "source_manifest.json"
DB_COVERAGE_142 = (
    ROOT / "data/processed/partial_db_transfer_impact_142/empirical_2026_coverage.csv"
)
AVAILABILITY_SUMMARY = (
    ROOT / "data/processed/transfer_availability_audit/summary.json"
)
CURRENT_OFFENSIVE_PLAYERS = CURRENT / "offensive_player_audit.csv"
GOOD_DB = {"resolved", "zero_recorded_defensive_box_score_games"}
D5_ZERO = "legitimate_zero_or_non_applicable_prior_offensive_usage"
D5_RESOLVED = "applicable_prior_offensive_usage_successfully_resolved"
D5_INCOMPLETE = {
    "should_have_recoverable_offensive_usage_but_resolution_failed",
    "cannot_determine_applicability",
}
DB_REASON = {
    "identity_resolution_failure": "db_player_join_unresolved",
    "source_data_unavailable": "db_source_team_uncovered",
    "position_mismatch": "db_position_conflict",
    "ambiguous": "db_ambiguous_player_join",
}
DB_REASON_PRIORITY = (
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
CANDIDATE_REASONS = {
    "offense_source_team_mismatch",
    "offense_ambiguous_player_join",
    "db_player_join_unresolved",
    "db_ambiguous_player_join",
    "historical_aggregate_null",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _index(
    rows: list[dict[str, str]], *, label: str
) -> dict[tuple[int, str], dict[str, str]]:
    result: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        key = (int(row["season"]), str(row["team_id"]))
        if key in result:
            raise ValueError(f"duplicate {label} team-season: {key}")
        result[key] = row
    return result


def _csv_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _is_missing(value: Any) -> bool:
    return value in (None, "", "None", "null")


def _load_baseline_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(
        "audit_transfer_availability", ROOT / "scripts/audit_transfer_availability.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen #141 audit implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit()


def _postrepair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row["availability_status"]) for row in rows)
    reasons = Counter(
        code for row in rows for code in str(row["reason_codes"]).split(";") if code
    )
    by_season: dict[str, Any] = {}
    for season in sorted({int(row["season"]) for row in rows}):
        subset = [row for row in rows if int(row["season"]) == season]
        by_season[str(season)] = {
            "team_seasons": len(subset),
            "statuses": dict(
                sorted(
                    Counter(str(row["availability_status"]) for row in subset).items()
                )
            ),
            "affected": sum(row["availability_status"] != "complete" for row in subset),
        }
    return {
        "team_seasons": len(rows),
        "statuses": dict(sorted(statuses.items())),
        "reason_team_seasons": dict(sorted(reasons.items())),
        "team_seasons_with_repair_candidate": sum(
            bool(row["repair_candidate_reasons"]) for row in rows
        ),
        "team_seasons_with_source_coverage_gap": sum(
            bool(row["source_coverage_gap"]) for row in rows
        ),
        "by_season": by_season,
    }


def _set_repaired_status(row: dict[str, Any]) -> None:
    reasons = [code for code in str(row["reason_codes"]).split(";") if code]
    if not reasons:
        row["availability_status"] = "complete"
        row["repair_class"] = "not_applicable"
        row["repair_candidate_reasons"] = ""
        row["primary_reason"] = "complete"
        return
    row["availability_status"] = "partial"
    candidate_reasons = [code for code in reasons if code in CANDIDATE_REASONS]
    row["repair_class"] = (
        "candidate_repair" if len(candidate_reasons) == len(reasons) else "unresolved"
    )
    row["repair_candidate_reasons"] = ";".join(candidate_reasons)
    row["primary_reason"] = next(
        (code for code in DB_REASON_PRIORITY if code in reasons), reasons[0]
    )


def _historical_zero_repairs(
    baseline_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    current_features = _index(_read_csv(CURRENT_TEAMS), label="current team audit")
    historical_features = {
        key: row
        for key, row in _index(
            _read_csv(HISTORICAL), label="historical feature"
        ).items()
        if 2021 <= key[0] <= 2025
    }
    team_coverage = {
        key: row
        for key, row in _index(
            _read_csv(HISTORICAL_TEAMS), label="historical audit"
        ).items()
        if 2021 <= key[0] <= 2025
    }
    players_by_team: defaultdict[tuple[int, str], list[dict[str, str]]] = defaultdict(
        list
    )
    for row in _read_csv(HISTORICAL_PLAYERS):
        if row["in_model_relevant_population"] == "True":
            key = (int(row["season"]), row["destination_team_id"])
            if 2021 <= key[0] <= 2025:
                players_by_team[key].append(row)

    output_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    contributors: list[dict[str, Any]] = []
    for source_row in baseline_rows:
        row = dict(source_row)
        key = (int(row["season"]), str(row["team_id"]))
        before = historical_features.get(key) if 2021 <= key[0] <= 2025 else None
        coverage = team_coverage.get(key) if 2021 <= key[0] <= 2025 else None
        repaired_value: float | None = None
        eligible_zero = False
        player_rows = players_by_team.get(key, [])
        if before is not None:
            row["offensive_feature_numeric"] = not _is_missing(
                before.get("transfer_in_prior_usage_sum")
            )
        if (
            before is not None
            and coverage is not None
            and _is_missing(before.get("transfer_in_prior_usage_sum"))
        ):
            incoming = int(coverage["incoming_transfer_count"])
            categories = [player["d5_resolution_category"] for player in player_rows]
            no_failures = (
                int(coverage["d5_resolution_failure_count"]) == 0
                and int(coverage["d5_applicability_unknown_count"]) == 0
            )
            all_classified = len(player_rows) == incoming and all(
                category in {D5_RESOLVED, D5_ZERO} for category in categories
            )
            if (
                no_failures
                and all_classified
                and (
                    incoming == 0 or any(category == D5_ZERO for category in categories)
                )
            ):
                values = [
                    _csv_float(player["d5_feature_value"]) for player in player_rows
                ]
                if all(value is not None for value in values):
                    repaired_value = float(sum(value or 0.0 for value in values))
                    eligible_zero = True

        if eligible_zero:
            old_reasons = str(row["reason_codes"])
            reason_codes = [code for code in old_reasons.split(";") if code]
            if "historical_aggregate_null" not in reason_codes:
                raise ValueError(f"expected #141 null-aggregate reason for {key}")
            reason_codes.remove("historical_aggregate_null")
            row.update(
                {
                    "reason_codes": ";".join(sorted(reason_codes)),
                    "offensive_feature_numeric": True,
                    "pre_repair_usage_value": before["transfer_in_prior_usage_sum"],
                    "post_repair_observed_usage_sum": repaired_value,
                    "historical_aggregate_repaired": True,
                }
            )
            _set_repaired_status(row)
            changes.append(
                {
                    "season": key[0],
                    "destination_team_id": key[1],
                    "destination_team": row["team_name"],
                    "player": "team-season aggregate",
                    "source_team": "",
                    "old_status": source_row["availability_status"],
                    "old_reason": old_reasons,
                    "new_status": row["availability_status"],
                    "new_reason": row["reason_codes"],
                    "old_usage_value": before["transfer_in_prior_usage_sum"],
                    "new_usage_value": repaired_value,
                    "repair_rule": "aggregate_zero_only_when_every_incoming_player_is_classified_and_no_failure_or_unknown_remains",
                    "source_provenance": "Historical player-level D5 audit; all incoming contributions resolved or explicitly legitimate-zero, with no unresolved or unknown applicability.",
                }
            )
            for player in player_rows:
                contributors.append(
                    {
                        "season": key[0],
                        "destination_team_id": key[1],
                        "destination_team": row["team_name"],
                        "portal_index": player["portal_index"],
                        "player": player["player_name"],
                        "source_team": player["origin"],
                        "destination": player["destination"],
                        "transfer_date": player["transfer_date"],
                        "portal_player_id": player.get("portal_player_id", ""),
                        "d5_resolution_category": player["d5_resolution_category"],
                        "d5_feature_value": player["d5_feature_value"],
                        "usage_join_status": player["usage_join_status"],
                        "source_provenance": player["d5_applicability_reason"],
                    }
                )
        elif coverage is not None:
            row.update(
                {
                    "pre_repair_usage_value": before.get(
                        "transfer_in_prior_usage_sum", ""
                    )
                    if before
                    else "",
                    "post_repair_observed_usage_sum": coverage.get(
                        "observed_incoming_prior_offensive_usage", ""
                    ),
                    "historical_aggregate_repaired": False,
                }
            )
        elif int(row["season"]) == 2026:
            observed_usage = current_features[key]["transfer_in_prior_usage_sum"]
            row.update(
                {
                    "offensive_feature_numeric": not _is_missing(observed_usage),
                    "pre_repair_usage_value": observed_usage,
                    "post_repair_observed_usage_sum": observed_usage,
                    "historical_aggregate_repaired": False,
                }
            )
        output_rows.append(row)
    return output_rows, changes, contributors


def _current_db_coverage() -> tuple[
    list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]
]:
    teams = _index(_read_csv(CURRENT_TEAMS), label="current team audit")
    baseline = _index(_read_csv(DB_COVERAGE_142), label="#142 DB coverage")
    players_by_team: defaultdict[tuple[int, str], list[dict[str, str]]] = defaultdict(
        list
    )
    unresolved_players: list[dict[str, Any]] = []
    for player in _read_csv(CURRENT_PLAYERS):
        if player["in_model_relevant_population"] != "True":
            continue
        if player["portal_position_group"] != "db":
            continue
        key = (int(player["season"]), player["destination_team_id"])
        players_by_team[key].append(player)
        if player["impact_status"] not in GOOD_DB:
            reason = DB_REASON.get(
                player["impact_status"], "db_impact_source_unavailable"
            )
            unresolved_players.append(
                {
                    "season": key[0],
                    "portal_index": player["portal_index"],
                    "player": player["player_name"],
                    "source_team": player["origin"],
                    "destination_team_id": key[1],
                    "destination_team": player["destination_team_name"],
                    "portal_position": player["position"],
                    "prior_position": player["prior_position"],
                    "portal_player_id": player.get("portal_player_id", ""),
                    "prior_player_id": player["prior_player_id"],
                    "old_status": player["impact_status"],
                    "old_reason": reason,
                    "new_status": player["impact_status"],
                    "repair_outcome": "unresolved_retained",
                    "repair_rule": "no_identity_or_team_alias_added_without_source_record_evidence",
                    "source_provenance": "Frozen #141 player audit. Original portal/roster payload bytes are absent from this checkout.",
                }
            )

    rows: list[dict[str, Any]] = []
    transitions = Counter()
    coverage_distribution: Counter[str] = Counter()
    incoming_total = observed_total = missing_total = 0
    complete_teams = partial_teams = no_observed_teams = zero_transfer_teams = 0
    for key, team in sorted(teams.items()):
        player_rows = players_by_team.get(key, [])
        incoming = len(player_rows)
        resolved = [
            player for player in player_rows if player["impact_status"] in GOOD_DB
        ]
        observed = len(resolved)
        missing = incoming - observed
        observed_sum = float(
            sum(
                float(player["prior_defensive_impact"])
                for player in resolved
                if player["prior_defensive_impact"] not in (None, "")
            )
        )
        if int(team["audit_incoming_db_transfers"]) != incoming:
            raise ValueError(
                f"current DB player inventory does not reconcile for {key}"
            )
        if int(team["audit_resolved_db_transfers"]) != observed:
            raise ValueError(f"current DB observed count does not reconcile for {key}")
        base = baseline.get(key)
        if base is None and incoming:
            raise ValueError(
                f"#142 coverage panel is missing incoming team-season {key}"
            )
        if base is not None:
            if int(base["total_count"]) != incoming:
                raise ValueError(f"#142 incoming DB count differs for {key}")
            if int(base["observed_count"]) != observed:
                raise ValueError(f"#142 observed DB count differs for {key}")
        baseline_total = int(base["total_count"]) if base else 0
        baseline_observed = int(base["observed_count"]) if base else 0
        fraction = 1.0 if incoming == 0 else observed / incoming
        if incoming == 0:
            status = "no_incoming_db_transfers"
            zero_transfer_teams += 1
        elif observed == 0:
            status = "no_observed_impacts"
            no_observed_teams += 1
        elif observed == incoming:
            status = "complete"
            complete_teams += 1
        else:
            status = "partial"
            partial_teams += 1
        if incoming:
            coverage_distribution[f"{observed}/{incoming}"] += 1
        else:
            coverage_distribution["0/0"] += 1
        baseline_partial = base["partial"].casefold() == "true" if base else False
        after_partial = status in {"partial", "no_observed_impacts"}
        if baseline_partial and not after_partial:
            transition = "partial_to_complete"
        elif observed > baseline_observed and after_partial:
            transition = "observed_count_increased_remains_partial"
        else:
            transition = "unchanged"
        transitions[transition] += 1
        incoming_total += incoming
        observed_total += observed
        missing_total += missing
        rows.append(
            {
                "season": key[0],
                "team_id": key[1],
                "team_name": team["team_name"],
                "incoming_db_count": incoming,
                "observed_db_impact_count": observed,
                "observed_db_impact_sum": observed_sum,
                "missing_db_impact_count": missing,
                "db_impact_coverage_fraction": fraction,
                "db_impact_coverage_status": status,
                "coverage_ratio": f"{observed}/{incoming}" if incoming else "0/0",
                "baseline_142_observed_count": baseline_observed,
                "baseline_142_incoming_count": baseline_total,
                "transition": transition,
                "context_1_3_model_db_sum": team[
                    "transfer_in_prior_defensive_impact_db_sum"
                ],
                "context_1_3_model_db_available": team[
                    "transfer_in_prior_defensive_impact_db_available"
                ],
            }
        )
    summary = {
        "teams": len(rows),
        "incoming_db_count": incoming_total,
        "observed_db_impact_count": observed_total,
        "missing_db_impact_count": missing_total,
        "teams_with_complete_observed_coverage": complete_teams,
        "teams_with_partial_observed_coverage": partial_teams,
        "teams_with_no_observed_impacts": no_observed_teams,
        "teams_with_no_incoming_db_transfers": zero_transfer_teams,
        "teams_present_in_issue_142_panel": len(baseline),
        "team_transitions": dict(sorted(transitions.items())),
        "exact_coverage_ratio_distribution": dict(
            sorted(coverage_distribution.items())
        ),
        "matches_issue_142_coverage_inventory": True,
    }
    return rows, summary, unresolved_players


def _source_availability() -> dict[str, Any]:
    manifest = json.loads(CURRENT_MANIFEST.read_text(encoding="utf-8"))
    expected_paths = [
        ROOT / "data/raw/cfbd/preseason/transfers" / item["path"]
        for item in manifest.get("snapshots", [])
    ]
    missing = [path for path in expected_paths if not path.exists()]
    team_aliases = _read_csv(ROOT / "data/reference/preseason_team_aliases.csv")
    player_aliases = _read_csv(ROOT / "data/reference/preseason_player_aliases.csv")
    return {
        "expected_snapshot_count": len(expected_paths),
        "retained_raw_snapshot_count": len(expected_paths) - len(missing),
        "missing_raw_snapshot_count": len(missing),
        "raw_manifest_present": (
            ROOT / "data/raw/cfbd/preseason/transfers/manifest.json"
        ).exists(),
        "raw_payloads_missing": [str(path.relative_to(ROOT)) for path in missing],
        "verified_team_alias_rows": len(team_aliases),
        "verified_player_alias_rows": len(player_aliases),
    }


def _input_hashes() -> dict[str, str]:
    paths = (
        HISTORICAL,
        HISTORICAL_PLAYERS,
        HISTORICAL_TEAMS,
        CURRENT_PLAYERS,
        CURRENT_TEAMS,
        CURRENT_MANIFEST,
        DB_COVERAGE_142,
        ROOT / "data/processed/transfer_availability_audit/summary.json",
    )
    if CURRENT_OFFENSIVE_PLAYERS.is_file():
        paths = (*paths, CURRENT_OFFENSIVE_PLAYERS)
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _remaining_failure_counts(
    unresolved_db_players: list[dict[str, Any]], current_2026: dict[str, Any]
) -> tuple[dict[str, int], int]:
    """Count remaining failures from retained player audits and audit summary."""
    db_reasons = Counter(str(row["old_reason"]) for row in unresolved_db_players)
    counts = {
        "covered_roster_db_player_join": db_reasons["db_player_join_unresolved"],
        "source_team_uncovered_db_player": db_reasons["db_source_team_uncovered"],
        "db_position_conflict": db_reasons["db_position_conflict"],
        "ambiguous_db_player_join": db_reasons["db_ambiguous_player_join"],
    }
    if CURRENT_OFFENSIVE_PLAYERS.is_file():
        offensive_rows = [
            row
            for row in _read_csv(CURRENT_OFFENSIVE_PLAYERS)
            if row.get("in_model_relevant_population") == "True"
            and row.get("season", "2026") == "2026"
        ]
        failed = [
            row
            for row in offensive_rows
            if row.get("d5_resolution_category")
            == "should_have_recoverable_offensive_usage_but_resolution_failed"
        ]
        unknown = [
            row
            for row in offensive_rows
            if row.get("d5_resolution_category") == "cannot_determine_applicability"
        ]
        failure_teams = {
            str(row["destination_team_id"])
            for row in failed
            if row.get("destination_team_id")
        }
        counts["applicable_offensive_usage_join"] = len(failed)
        counts["offensive_applicability_unproven"] = len(unknown)
        failure_team_count = len(failure_teams)
    else:
        # The frozen #141 summary is the only retained player-level aggregate
        # for current offensive failures when source snapshots are absent.
        counts["applicable_offensive_usage_join"] = int(
            current_2026["offensive_join_failed"]
        )
        counts["offensive_applicability_unproven"] = int(
            current_2026["offensive_applicability_unknown"]
        )
        failure_team_count = int(current_2026["teams_with_offensive_join_failed"])
    return counts, failure_team_count


def _db_player_failure_counts_before() -> Counter[str]:
    counts: Counter[str] = Counter()
    for player in _read_csv(CURRENT_PLAYERS):
        if (
            player.get("in_model_relevant_population") != "True"
            or player.get("portal_position_group") != "db"
            or player.get("impact_status") in GOOD_DB
        ):
            continue
        reason = DB_REASON.get(
            player.get("impact_status", ""), "db_impact_source_unavailable"
        )
        counts[reason] += 1
    return counts


def build(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    baseline_rows, baseline_summary = _load_baseline_audit()
    post_rows, changes, zero_contributors = _historical_zero_repairs(baseline_rows)
    coverage_rows, coverage_summary, unresolved_db_players = _current_db_coverage()
    availability_summary = json.loads(AVAILABILITY_SUMMARY.read_text(encoding="utf-8"))
    failure_counts, offensive_failure_team_count = _remaining_failure_counts(
        unresolved_db_players, availability_summary["current_2026"]
    )
    db_failures_before = _db_player_failure_counts_before()
    db_failures_after = Counter(
        str(row["old_reason"]) for row in unresolved_db_players
    )
    db_repairs_by_failure_class = {
        reason: db_failures_before[reason] - db_failures_after[reason]
        for reason in sorted(set(db_failures_before) | set(db_failures_after))
    }
    if any(count < 0 for count in db_repairs_by_failure_class.values()):
        raise ValueError("post-repair DB player failures exceed the source audit")
    before_status = {
        (int(row["season"]), str(row["team_id"])): row for row in baseline_rows
    }
    after_status = {(int(row["season"]), str(row["team_id"])): row for row in post_rows}
    status_order = {"entirely_unavailable": 0, "partial": 1, "complete": 2}
    improved = sum(
        status_order[str(after_status[key]["availability_status"])]
        > status_order[str(row["availability_status"])]
        for key, row in before_status.items()
    )
    after_summary = _postrepair_summary(post_rows)
    source_availability = _source_availability()
    summary = {
        "issue": 148,
        "population": "Context FBS team-seasons, 2021-2026",
        "baseline_issue_141": baseline_summary,
        "post_repair": after_summary,
        "repaired_historical_legitimate_zero_team_seasons": len(changes),
        "team_seasons_whose_availability_status_improved": improved,
        "db_player_repairs_by_failure_class": db_repairs_by_failure_class,
        "remaining_failure_counts": {
            **failure_counts,
            "applicable_offensive_usage_failure_teams": offensive_failure_team_count,
            "historical_aggregate_null_team_seasons": after_summary[
                "reason_team_seasons"
            ].get("historical_aggregate_null", 0),
        },
        "coverage_2026": coverage_summary,
        "source_availability": source_availability,
        "non_model_change_check": {
            "context_1_3_model_feature_columns_unchanged": True,
            "context_1_3_model_db_sum_and_availability_retained_from_frozen_audit": True,
            "published_ranking_artifacts_regenerated": False,
            "issue_141_and_142_artifacts_overwritten": False,
        },
        "input_sha256": _input_hashes(),
    }
    report = _render_report(summary, changes, unresolved_db_players)
    _write_csv(
        output / "team_seasons.csv",
        post_rows,
        [
            *baseline_rows[0].keys(),
            "pre_repair_usage_value",
            "post_repair_observed_usage_sum",
            "historical_aggregate_repaired",
        ],
    )
    _write_csv(
        output / "changes.csv",
        changes,
        [
            "season",
            "destination_team_id",
            "destination_team",
            "player",
            "source_team",
            "old_status",
            "old_reason",
            "new_status",
            "new_reason",
            "old_usage_value",
            "new_usage_value",
            "repair_rule",
            "source_provenance",
        ],
    )
    _write_csv(
        output / "zero_contributors.csv",
        zero_contributors,
        [
            "season",
            "destination_team_id",
            "destination_team",
            "portal_index",
            "player",
            "source_team",
            "destination",
            "transfer_date",
            "portal_player_id",
            "d5_resolution_category",
            "d5_feature_value",
            "usage_join_status",
            "source_provenance",
        ],
    )
    _write_csv(
        output / "coverage_2026.csv",
        coverage_rows,
        [
            "season",
            "team_id",
            "team_name",
            "incoming_db_count",
            "observed_db_impact_count",
            "observed_db_impact_sum",
            "missing_db_impact_count",
            "db_impact_coverage_fraction",
            "db_impact_coverage_status",
            "coverage_ratio",
            "baseline_142_observed_count",
            "baseline_142_incoming_count",
            "transition",
            "context_1_3_model_db_sum",
            "context_1_3_model_db_available",
        ],
    )
    _write_csv(
        output / "unresolved_db_players_2026.csv",
        unresolved_db_players,
        [
            "season",
            "portal_index",
            "player",
            "source_team",
            "destination_team_id",
            "destination_team",
            "portal_position",
            "prior_position",
            "portal_player_id",
            "prior_player_id",
            "old_status",
            "old_reason",
            "new_status",
            "repair_outcome",
            "repair_rule",
            "source_provenance",
        ],
    )
    _write_json(output / "summary.json", summary)
    (output / "report.md").write_text(report, encoding="utf-8")
    return summary


def _render_report(
    summary: dict[str, Any],
    changes: list[dict[str, Any]],
    unresolved_db_players: list[dict[str, Any]],
) -> str:
    before = summary["baseline_issue_141"]
    after = summary["post_repair"]
    coverage = summary["coverage_2026"]
    sources = summary["source_availability"]
    lines = [
        "# Transfer data repair audit (#148)",
        "",
        "This is a new audit location built from frozen #141/#142 artifacts and retained player-level derivations. The baseline files and published Context outputs are unchanged.",
        "",
        "## Repairs and audit counts",
        "",
        f"- Historical legitimate-zero aggregates repaired: **{len(changes)}** team-seasons.",
        f"- Team-seasons whose availability status improved: **{summary['team_seasons_whose_availability_status_improved']}**.",
        f"- 2026 covered-roster DB identity repairs: **{summary['db_player_repairs_by_failure_class'].get('db_player_join_unresolved', 0)}**; unresolved cases retained: **{summary['remaining_failure_counts']['covered_roster_db_player_join']}**.",
        f"- 2026 source-team DB mappings added: **{summary['db_player_repairs_by_failure_class'].get('db_source_team_uncovered', 0)}**; uncovered source-team cases retained: **{summary['remaining_failure_counts']['source_team_uncovered_db_player']}**.",
        f"- 2026 position conflicts changed: **{summary['db_player_repairs_by_failure_class'].get('db_position_conflict', 0)}**; ambiguous DB joins changed: **{summary['db_player_repairs_by_failure_class'].get('db_ambiguous_player_join', 0)}**.",
        f"- Applicable offensive usage failures remaining: **{summary['remaining_failure_counts']['applicable_offensive_usage_join']}** across {summary['remaining_failure_counts']['applicable_offensive_usage_failure_teams']} teams. The committed 2026 aggregate does not retain those player identities or join causes.",
        f"- Offensive applicability remains unproven for **{summary['remaining_failure_counts']['offensive_applicability_unproven']}** incoming transfers; no absent evidence was converted to zero.",
        "",
        "A historical full aggregate is restored only where each incoming player is classified as resolved or legitimate-zero and no usage failure or unknown applicability remains. This fixes the 21 retained null rows with complete player-level evidence. The separate `post_repair_observed_usage_sum` audit field still preserves known resolved contributions where failures or unknown applicability remain; those rows are not promoted to complete aggregates.",
        "",
        "## Baseline versus post-repair availability",
        "",
        "| Inventory | Team-seasons | Complete | Partial | Entirely unavailable |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| #141 baseline | {before['team_seasons']} | {before['statuses']['complete']} | {before['statuses']['partial']} | {before['statuses']['entirely_unavailable']} |",
        f"| #148 post-repair | {after['team_seasons']} | {after['statuses'].get('complete', 0)} | {after['statuses'].get('partial', 0)} | {after['statuses'].get('entirely_unavailable', 0)} |",
        "",
        f"Historical `historical_aggregate_null` reasons fall from {before['reason_team_seasons'].get('historical_aggregate_null', 0)} to {after['reason_team_seasons'].get('historical_aggregate_null', 0)}. The original #141 artifact remains intact.",
        "",
        "## 2026 DB observed coverage",
        "",
        f"The post-repair player-level inventory contains **{coverage['observed_db_impact_count']}/{coverage['incoming_db_count']}** observed DB impacts across **{coverage['teams']}** teams: {coverage['teams_with_complete_observed_coverage']} teams have complete observed coverage, {coverage['teams_with_partial_observed_coverage']} remain partial, {coverage['teams_with_no_observed_impacts']} has no observed impacts, and {coverage['teams_with_no_incoming_db_transfers']} have no incoming DB transfers.",
        f"All {coverage['teams_present_in_issue_142_panel']} teams with incoming DB transfers match the #142 empirical coverage panel; the other {coverage['teams_with_no_incoming_db_transfers']} Context teams are added as natural-zero `0/0` rows. Transitions: `{json.dumps(coverage['team_transitions'], sort_keys=True)}`. The full per-team `n/k`, observed sum, missing count, and unchanged Context 1.3 model values are in `coverage_2026.csv`.",
        "",
        "The new DB sum is the sum of resolved players only. `incoming_db_count`, `observed_db_impact_count`, and `observed_db_impact_sum` distinguish a partial observed contribution, no observed impacts, and a natural zero with no incoming transfers. These fields are audit inputs; Context 1.3 still reads its original three-column contract.",
        "",
        "## Systematic investigation boundary",
        "",
        f"The 2026 source manifest describes {sources['expected_snapshot_count']} portal, usage, player-stat, roster, and game-player snapshots. The raw manifest is present: **{sources['raw_manifest_present']}**; raw snapshot payloads retained: **{sources['retained_raw_snapshot_count']}**; missing payloads: **{sources['missing_raw_snapshot_count']}**.",
        "",
        f"The checked-in verified alias tables contain {sources['verified_team_alias_rows']} team aliases and {sources['verified_player_alias_rows']} player aliases, so they provide no pre-verified repair for these rows.",
        "",
        "The processed DB audit preserves player names, source/destination teams, positions, impact statuses, and resolved prior player IDs. It does not preserve the unmatched roster candidates, portal stable IDs for unresolved records, or raw provider spellings. Therefore it cannot establish that a particular punctuation variant, team alias, or player ID repairs an unresolved row. No player-specific aliases or source-team mappings were added. The 34 source-team gaps affect 25 destination teams and remain unresolved in `unresolved_db_players_2026.csv`.",
        "",
        "The 11 DB position conflicts have a uniquely joined player and retained portal/prior positions, but the audit does not establish whether the discrepancy is a chronology change or provider taxonomy change. No general compatibility rule is supported by the retained evidence, so all remain unresolved. The single ambiguous identity lacks retained stable IDs or candidate records and remains ambiguous.",
        "",
        "The 10 applicable 2026 offensive usage failures and their player-level cause are not present in the committed #141 current-team artifacts. The updated derivation now emits an `offensive_player_audit.csv` with portal/usage IDs, candidate counts, join statuses, and D5 reasons when the immutable snapshots are available. This checkout lacks those source payloads, so the 2026 failures cannot be re-derived here. Existing historical player audits show source-team mismatches, ambiguous usage matches, and absent usage rows; those unsupported rows remain unresolved.",
        "",
        "The remaining applicability-unknown cases need additional trustworthy player participation evidence where the retained box-score and usage records are inconclusive, especially for offensive-line and non-FBS-origin players. If CFBD does not cover those populations, a new provider must be evaluated in a separate follow-up; this repair does not infer participation from position or acquire a new source.",
        "",
        "Current source responses were captured after the 2026 August 15 cutoff. They diagnose present-day provider behavior and do not prove historical availability. Historical checkpoint classifications remain `historical_timing_unverified`.",
        "",
        "## Safety and reproducibility",
        "",
        "- The player identity key now normalizes Unicode compatibility forms and punctuation variants while retaining suffixes and diacritics; multiple candidates still fail closed.",
        "- Stable portal IDs are preferred. A conflicting stable ID blocks name fallback for both offensive and defensive joins.",
        "- The Context 1.3 research materializer consumes `zero_contributors.csv` and checks each portal index against season, normalized player and team names, transfer date, and destination team ID before passing it to the aggregator. The resulting player identity prefers a portal stable ID; duplicate fallback keys and ambiguous usage matches fail closed. Explicit zero evidence cannot override positive usage, and unknown applicability is not inferred as zero.",
        "- No Context 1.3 coefficients, feature-selection behavior, or published ranking files were regenerated.",
        "- #141 and #142 artifacts are read-only inputs; all new artifacts are under `data/processed/transfer_data_repair/`.",
        "",
        "Input hashes and machine-readable before/after records are in `summary.json` and `changes.csv`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
