from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_transfer_data_repair.py"
spec = importlib.util.spec_from_file_location("transfer_data_repair_audit", SCRIPT)
assert spec is not None and spec.loader is not None
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_postrepair_audit_preserves_baselines_and_reconciles_coverage(
    tmp_path: Path,
) -> None:
    immutable_inputs = (
        ROOT / "data/processed/transfer_availability_audit/team_seasons.csv",
        ROOT / "data/processed/transfer_availability_audit/summary.json",
        ROOT
        / "data/processed/partial_db_transfer_impact_142/empirical_2026_coverage.csv",
    )
    before_hashes = {path: _sha256(path) for path in immutable_inputs}

    summary = audit_module.build(tmp_path)
    changes = _rows(tmp_path / "changes.csv")
    zero_contributors = _rows(tmp_path / "zero_contributors.csv")
    team_seasons = _rows(tmp_path / "team_seasons.csv")
    coverage = _rows(tmp_path / "coverage_2026.csv")
    unresolved = _rows(tmp_path / "unresolved_db_players_2026.csv")

    assert len(changes) == 21
    assert len(zero_contributors) == 46
    assert all(
        row["d5_resolution_category"]
        == "legitimate_zero_or_non_applicable_prior_offensive_usage"
        and row["transfer_date"]
        for row in zero_contributors
    )
    assert all(
        row["repair_rule"].startswith("aggregate_zero_only_when_every_incoming_player")
        for row in changes
    )
    assert len(coverage) == 138
    current_team_seasons = [row for row in team_seasons if row["season"] == "2026"]
    assert len(current_team_seasons) == 138
    assert all(
        row["post_repair_observed_usage_sum"] != "" for row in current_team_seasons
    )
    assert sum(int(row["incoming_db_count"]) for row in coverage) == 604
    assert sum(int(row["observed_db_impact_count"]) for row in coverage) == 518
    assert sum(int(row["missing_db_impact_count"]) for row in coverage) == 86
    assert all(row["transition"] == "unchanged" for row in coverage)
    assert len(unresolved) == 86
    assert summary["post_repair"]["statuses"] == {
        "complete": 88,
        "entirely_unavailable": 2,
        "partial": 703,
    }
    assert summary["team_seasons_whose_availability_status_improved"] == 17
    assert summary["coverage_2026"]["teams_with_no_incoming_db_transfers"] == 6
    assert summary["coverage_2026"]["teams_with_no_observed_impacts"] == 1
    assert summary["source_availability"]["missing_raw_snapshot_count"] == 37
    assert (
        summary["remaining_failure_counts"]["historical_aggregate_null_team_seasons"]
        == 52
    )
    assert before_hashes == {path: _sha256(path) for path in immutable_inputs}

    first_output_hashes = {
        path.name: _sha256(path) for path in tmp_path.iterdir() if path.is_file()
    }
    audit_module.build(tmp_path)
    second_output_hashes = {
        path.name: _sha256(path) for path in tmp_path.iterdir() if path.is_file()
    }
    assert second_output_hashes == first_output_hashes
    written_summary = json.loads((tmp_path / "summary.json").read_text())
    assert written_summary["issue"] == 148


def test_remaining_failure_counts_follow_the_retained_audit_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        audit_module,
        "CURRENT_OFFENSIVE_PLAYERS",
        tmp_path / "offensive_player_audit.csv",
    )
    counts, teams = audit_module._remaining_failure_counts(
        [
            {"old_reason": "db_player_join_unresolved"},
            {"old_reason": "db_source_team_uncovered"},
            {"old_reason": "db_source_team_uncovered"},
        ],
        {
            "offensive_join_failed": 7,
            "offensive_applicability_unknown": 31,
            "teams_with_offensive_join_failed": 4,
        },
    )

    assert counts["covered_roster_db_player_join"] == 1
    assert counts["source_team_uncovered_db_player"] == 2
    assert counts["applicable_offensive_usage_join"] == 7
    assert counts["offensive_applicability_unproven"] == 31
    assert teams == 4

    audit_path = tmp_path / "offensive_player_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "in_model_relevant_population",
                "d5_resolution_category",
                "destination_team_id",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "in_model_relevant_population": "True",
                    "d5_resolution_category": "should_have_recoverable_offensive_usage_but_resolution_failed",
                    "destination_team_id": "10",
                },
                {
                    "in_model_relevant_population": "True",
                    "d5_resolution_category": "should_have_recoverable_offensive_usage_but_resolution_failed",
                    "destination_team_id": "10",
                },
                {
                    "in_model_relevant_population": "True",
                    "d5_resolution_category": "cannot_determine_applicability",
                    "destination_team_id": "11",
                },
                {
                    "in_model_relevant_population": "False",
                    "d5_resolution_category": "should_have_recoverable_offensive_usage_but_resolution_failed",
                    "destination_team_id": "12",
                },
            ]
        )
    monkeypatch.setattr(audit_module, "CURRENT_OFFENSIVE_PLAYERS", audit_path)

    counts, teams = audit_module._remaining_failure_counts(
        [],
        {
            "offensive_join_failed": 99,
            "offensive_applicability_unknown": 101,
            "teams_with_offensive_join_failed": 55,
        },
    )

    assert counts["applicable_offensive_usage_join"] == 2
    assert counts["offensive_applicability_unproven"] == 1
    assert teams == 1
