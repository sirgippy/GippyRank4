from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gippyrank.transfer_repair import (
    build_db_coverage_inventory,
    repair_verified_historical_zero_aggregates,
)

repair_audit = importlib.import_module("build_transfer_data_repair_audit")


def test_historical_aggregate_repair_requires_complete_known_player_evidence():
    features = [
        {
            "season": "2022",
            "team_id": "a",
            "team_name": "Alpha",
            "transfer_in_prior_usage_sum": "",
        },
        {
            "season": "2022",
            "team_id": "b",
            "team_name": "Beta",
            "transfer_in_prior_usage_sum": "",
        },
        {
            "season": "2022",
            "team_id": "c",
            "team_name": "Gamma",
            "transfer_in_prior_usage_sum": "",
        },
        {
            "season": "2022",
            "team_id": "d",
            "team_name": "Delta",
            "transfer_in_prior_usage_sum": "0.5",
        },
    ]
    coverage = [
        {
            "season": "2022",
            "team_id": "a",
            "team_name": "Alpha",
            "portal_payload_available": "True",
            "incoming_transfer_count": "2",
        },
        {
            "season": "2022",
            "team_id": "b",
            "team_name": "Beta",
            "portal_payload_available": "True",
            "incoming_transfer_count": "1",
        },
        {
            "season": "2022",
            "team_id": "c",
            "team_name": "Gamma",
            "portal_payload_available": "True",
            "incoming_transfer_count": "0",
        },
        {
            "season": "2022",
            "team_id": "d",
            "team_name": "Delta",
            "portal_payload_available": "True",
            "incoming_transfer_count": "1",
        },
    ]
    players = [
        {
            "season": "2022",
            "destination_team_id": "a",
            "portal_index": "1",
            "in_model_relevant_population": "True",
            "player_name": "Known Zero",
            "origin": "Origin One",
            "d5_resolution_category": "legitimate_zero_or_non_applicable_prior_offensive_usage",
            "d5_feature_value": "0.0",
        },
        {
            "season": "2022",
            "destination_team_id": "a",
            "portal_index": "2",
            "in_model_relevant_population": "True",
            "player_name": "Known Usage",
            "origin": "Origin Two",
            "d5_resolution_category": "applicable_prior_offensive_usage_successfully_resolved",
            "d5_feature_value": "0.25",
        },
        {
            "season": "2022",
            "destination_team_id": "b",
            "portal_index": "3",
            "in_model_relevant_population": "True",
            "player_name": "Unknown",
            "origin": "Origin Three",
            "d5_resolution_category": "cannot_determine_applicability",
            "d5_feature_value": "",
        },
    ]
    output, changes = repair_verified_historical_zero_aggregates(
        features, coverage, players
    )
    by_team = {row["team_id"]: row for row in output}
    assert by_team["a"]["transfer_in_prior_usage_sum"] == "0.25"
    assert by_team["b"]["transfer_in_prior_usage_sum"] == ""
    assert by_team["c"]["transfer_in_prior_usage_sum"] == "0"
    assert by_team["d"]["transfer_in_prior_usage_sum"] == "0.5"
    assert [(row["team_id"], row["new_value"]) for row in changes] == [
        ("a", "0.25"),
        ("c", "0"),
    ]


def test_db_inventory_preserves_partial_observations_and_natural_zero():
    teams = [
        {"season": "2026", "team_id": "a", "team_name": "Alpha"},
        {"season": "2026", "team_id": "b", "team_name": "Beta"},
    ]
    players = [
        {
            "season": "2026",
            "destination_team_id": "a",
            "portal_index": "1",
            "in_model_relevant_population": "True",
            "portal_position_group": "db",
            "impact_status": "resolved",
            "prior_defensive_impact": "0.0",
        },
        {
            "season": "2026",
            "destination_team_id": "a",
            "portal_index": "2",
            "in_model_relevant_population": "True",
            "portal_position_group": "db",
            "impact_status": "identity_resolution_failure",
            "prior_defensive_impact": "",
        },
    ]
    provenance = {
        "2026|a|transfer_in_prior_defensive_impact_db_sum": {
            "source_snapshot_sha256": ["portal-hash", "roster-hash"]
        }
    }
    rows = build_db_coverage_inventory(
        teams, players, source_season=2025, provenance=provenance
    )
    alpha, beta = rows
    assert (alpha["observed_count"], alpha["total_count"]) == (1, 2)
    assert alpha["observed_over_incoming"] == "1/2"
    assert alpha["coverage_status"] == "partial"
    assert alpha["observed_db_impact_sum"] == 0.0
    assert alpha["source_snapshot_sha256"] == "portal-hash;roster-hash"
    assert beta["observed_over_incoming"] == "0/0"
    assert beta["coverage_status"] == "no_incoming_db_transfers"
    assert beta["coverage_fraction"] is None


def test_raw_payload_counter_excludes_provenance_sidecars(tmp_path: Path):
    source_dir = tmp_path / "portal" / "2025"
    source_dir.mkdir(parents=True)
    (source_dir / "2025.json").write_text("[]\n", encoding="utf-8")
    (source_dir / "2025.json.provenance.json").write_text("{}\n", encoding="utf-8")
    (source_dir / "2024.json.provenance.json").write_text("{}\n", encoding="utf-8")

    assert repair_audit._raw_payload_files(tmp_path, "portal") == [
        source_dir / "2025.json"
    ]


def test_repair_audit_artifacts_reproduce_in_a_separate_output_dir(tmp_path: Path):
    committed = repair_audit.OUTPUT
    repair_audit.run(tmp_path)
    for name in (
        "historical_transfer_features.csv",
        "team_seasons.csv",
        "before_after_repairs.csv",
        "db_coverage_2026.csv",
        "player_before_after_2026.csv",
        "team_before_after_2026.csv",
        "before_source_manifest.json",
        "after_source_manifest.json",
        "summary.json",
        "report.md",
    ):
        assert (tmp_path / name).read_bytes() == (committed / name).read_bytes()


def test_repair_inventory_is_machine_readable_and_tracks_provenance():
    path = repair_audit.OUTPUT / "before_after_repairs.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 21
    assert all(row["old_reason"] == "historical_aggregate_null" for row in rows)
    assert all(
        row["repair_rule"] == "sum_complete_player_audit_evidence" for row in rows
    )
    assert all(row["source_provenance"] for row in rows)
