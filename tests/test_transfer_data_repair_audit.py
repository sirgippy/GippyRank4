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
    team_seasons = _rows(tmp_path / "team_seasons.csv")
    coverage = _rows(tmp_path / "coverage_2026.csv")
    unresolved = _rows(tmp_path / "unresolved_db_players_2026.csv")

    assert len(changes) == 21
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
