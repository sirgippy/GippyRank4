from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_transfer_data_repair.py"
spec = importlib.util.spec_from_file_location("transfer_data_repair_audit", SCRIPT)
assert spec is not None and spec.loader is not None
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)
INTEGRATED_SCRIPT = ROOT / "scripts/build_integrated_transfer_repair_audit.py"
integrated_spec = importlib.util.spec_from_file_location(
    "integrated_transfer_data_repair_audit", INTEGRATED_SCRIPT
)
assert integrated_spec is not None and integrated_spec.loader is not None
integrated_module = importlib.util.module_from_spec(integrated_spec)
integrated_spec.loader.exec_module(integrated_module)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_integrated_audit_reconciles_historical_and_2026_checkpoints() -> None:
    artifact_root = ROOT / "data/processed/transfer_data_repair"
    summary = json.loads((artifact_root / "summary.json").read_text(encoding="utf-8"))
    historical = summary["historical"]
    current = summary["2026"]

    assert historical["availability_status_counts"] == {
        "complete": 83,
        "partial": 570,
        "entirely_unavailable": 2,
    }
    assert historical["all_years_combined_checkpoint"][
        "availability_status_counts"
    ] == {
        "complete": 87,
        "partial": 704,
        "entirely_unavailable": 2,
    }
    assert historical["historical_aggregate_null_reasons"] == {
        "before": 73,
        "after": 51,
    }
    assert (
        historical["materializer_replay"]["legacy_replay_matches_frozen_panel"] is True
    )
    assert historical["materializer_replay"]["new_replay_matches_materializer"] is True
    reconciliation = historical["reconciliation_with_151_pre_materializer_snapshot"]
    assert historical["seasons"] == [2021, 2022, 2023, 2024, 2025]
    assert reconciliation["team_seasons_reclassified"] == [
        {
            "season": 2022,
            "team_id": "251",
            "team": "Texas",
            "pre_materializer_status": "complete",
            "final_status": "partial",
            "pre_materializer_primary_reason": "complete",
            "final_primary_reason": "offense_applicability_unproven",
            "pre_materializer_reason_codes": "",
            "final_reason_codes": "offense_applicability_unproven",
            "player_evidence": [
                {
                    "portal_index": "2572",
                    "player": "Diamonte Tucker-Dorsey",
                    "position": "LB",
                    "usage_join_status": "no_usage_record",
                    "usage_candidate_count": 0,
                    "applicability": "cannot_determine_applicability",
                    "applicability_reason": (
                        "defensive_or_special_portal_position; "
                        "source_team_outside_verified_stats_coverage"
                    ),
                }
            ],
        }
    ]
    null_reconciliation = reconciliation["historical_aggregate_null_reconciliation"]
    assert (
        null_reconciliation["pre_materializer_count"],
        null_reconciliation["final_count"],
        null_reconciliation["net_reduction"],
    ) == (52, 51, 1)
    assert null_reconciliation["added_cases"] == []
    assert null_reconciliation["removed_cases"] == [
        {
            "season": 2021,
            "team_id": "2653",
            "team": "Troy",
            "pre_materializer_reason_codes": (
                "db_player_join_unresolved;db_position_conflict;"
                "historical_aggregate_null;offense_applicability_unproven"
            ),
            "final_reason_codes": (
                "db_player_join_unresolved;db_position_conflict;"
                "offense_applicability_unproven"
            ),
            "pre_materializer_usage_value": "",
            "final_usage_value": "0.163",
        }
    ]

    assert (current["incoming_db_transfers"], current["observed_db_impacts"]) == (
        604,
        550,
    )
    assert current["unresolved_db_impacts"] == 54
    assert current["unresolved_db_impacts_by_reason"] == {
        "identity_resolution_failure": 17,
        "source_data_unavailable": 23,
        "position_mismatch": 14,
        "ambiguous": 0,
    }
    assert current["changed_db_player_impact_values"]["total"] == 35
    assert current["changed_db_player_impact_values"]["normalization_drift_only"] == 0
    assert current["defensive_impact_reference"]["changed_sha256_count"] == 0
    assert current["context_1_3_invariance"]["model_facing_inputs_changed"] is False
    assert len(_rows(artifact_root / "team_seasons.csv")) == 793


def test_historical_reconciliation_is_derived_from_compared_rows() -> None:
    pre_materializer = [
        {
            "season": "2022",
            "team_id": "251",
            "team_name": "Texas",
            "availability_status": "complete",
            "primary_reason": "complete",
            "reason_codes": "",
        }
    ]
    final = [
        {
            "season": "2022",
            "team_id": "251",
            "team_name": "Texas",
            "availability_status": "partial",
            "primary_reason": "offense_applicability_unproven",
            "reason_codes": "offense_applicability_unproven",
        }
    ]
    player_audit = [
        {
            "season": "2022",
            "destination_team_id": "251",
            "in_model_relevant_population": "True",
            "d5_resolution_category": "cannot_determine_applicability",
            "portal_index": "2572",
            "player_name": "Diamonte Tucker-Dorsey",
            "position": "LB",
            "usage_join_status": "no_usage_record",
            "usage_candidate_count": "0",
            "d5_applicability_reason": "source_team_outside_verified_stats_coverage",
        }
    ]

    changes = integrated_module._historical_status_reconciliation(
        pre_materializer, final, player_audit
    )

    assert len(changes) == 1
    assert changes[0]["season"] == 2022
    assert changes[0]["team"] == "Texas"
    assert changes[0]["pre_materializer_status"] == "complete"
    assert changes[0]["final_status"] == "partial"
    assert changes[0]["player_evidence"][0]["player"] == "Diamonte Tucker-Dorsey"
    assert changes[0]["player_evidence"][0]["usage_candidate_count"] == 0


def test_historical_null_reconciliation_is_derived_from_reason_code_changes() -> None:
    pre_materializer = [
        {
            "season": "2021",
            "team_id": "2653",
            "team_name": "Troy",
            "reason_codes": "historical_aggregate_null;other_reason",
        }
    ]
    final = [
        {
            "season": "2021",
            "team_id": "2653",
            "team_name": "Troy",
            "reason_codes": "other_reason",
            "post_repair_materialized_usage_value": "0.163",
        }
    ]

    reconciliation = integrated_module._historical_aggregate_null_reconciliation(
        pre_materializer, final
    )

    assert reconciliation["pre_materializer_count"] == 1
    assert reconciliation["final_count"] == 0
    assert reconciliation["removed_cases"][0]["team"] == "Troy"
    assert reconciliation["removed_cases"][0]["final_usage_value"] == "0.163"
    assert reconciliation["added_cases"] == []


def test_integrated_report_uses_derived_reconciliation_without_stale_narration() -> (
    None
):
    artifact_root = ROOT / "data/processed/transfer_data_repair"
    summary = json.loads((artifact_root / "summary.json").read_text(encoding="utf-8"))
    reconciliation = summary["historical"][
        "reconciliation_with_151_pre_materializer_snapshot"
    ]
    pre_materializer = [
        {
            "season": "2024",
            "team_id": "77",
            "team_name": "Beta",
            "availability_status": "complete",
            "primary_reason": "complete",
            "reason_codes": "historical_aggregate_null",
        }
    ]
    final = [
        {
            "season": "2024",
            "team_id": "77",
            "team_name": "Beta",
            "availability_status": "partial",
            "primary_reason": "audit_evidence_missing",
            "reason_codes": "audit_evidence_missing",
            "post_repair_materialized_usage_value": "0.25",
        }
    ]
    reconciliation["team_seasons_reclassified"] = (
        integrated_module._historical_status_reconciliation(pre_materializer, final, [])
    )
    reconciliation["historical_aggregate_null_reconciliation"] = (
        integrated_module._historical_aggregate_null_reconciliation(
            pre_materializer, final
        )
    )

    report = integrated_module._render_report(summary)

    assert "2024 Beta changed from `complete` to `partial`" in report
    assert "2024 Beta (final usage `0.25`)" in report
    assert "2022 Texas" not in report
    assert "2021 Troy" not in report
    assert "Diamonte Tucker-Dorsey" not in report


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

    artifact_root = tmp_path / "historical_audit"
    summary = audit_module.build(
        artifact_root,
        materializer_source_root=integrated_module._materializer_source_root(None),
    )
    changes = _rows(artifact_root / "changes.csv")
    zero_contributors = _rows(artifact_root / "zero_contributors.csv")
    team_seasons = _rows(artifact_root / "team_seasons.csv")
    coverage = _rows(artifact_root / "coverage_2026.csv")
    unresolved = _rows(artifact_root / "unresolved_db_players_2026.csv")
    feature_changes = _rows(artifact_root / "historical_feature_changes.csv")
    corrected_player_audit = _rows(artifact_root / "historical_player_repair_audit.csv")

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
    assert len(feature_changes) == 24
    assert len(_rows(artifact_root / "historical_transfer_features.csv")) == 2744
    replay = summary["historical_feature_reconciliation"]
    assert replay["verified_source_count"] == 15
    assert replay["verified_source_count"] == len(replay["verified_sources"])
    assert replay["legacy_replay_matches_frozen_panel"] is True
    assert replay["new_replay_matches_materializer"] is True
    assert replay["change_inventory_exactly_matches_materializer_diff"] is True
    assert replay["unexplained_changed_feature_values"] == 0
    assert replay["changed_team_seasons"] == 24
    assert replay["changed_feature_values"] == len(feature_changes)
    assert len({(row["season"], row["team_id"]) for row in feature_changes}) == 24
    assert replay["changed_feature_values_by_class"] == {
        "ambiguous_usage_join_removed": 1,
        "legitimate_zero_restoration": 21,
        "name_normalization_join_added": 2,
    }
    assert replay["legitimate_zero_contributor_players"] == 46
    assert replay["ambiguous_player_audit_record_count"] == 1
    assert replay["ambiguous_changed_transfer_count"] == 1
    assert replay["duplicate_equivalent_ambiguous_joins_resolved"] == 24
    assert replay["remaining_ambiguous_usage_joins"] == 1
    assert all(
        row["feature_name"] == "transfer_in_prior_usage_sum" for row in feature_changes
    )
    assert len(
        {
            (row["season"], row["team_id"], row["feature_name"])
            for row in feature_changes
        }
    ) == len(feature_changes)

    by_team = {(row["season"], row["team_name"]): row for row in feature_changes}
    assert ("2022", "Eastern Michigan") not in by_team
    taylor = next(
        row
        for row in corrected_player_audit
        if row["season"] == "2022" and row["player_name"] == "Taylor Powell"
    )
    assert taylor["usage_join_status"] == "joined"
    assert taylor["usage_candidate_resolution"] == "duplicate_equivalent_rows_collapsed"
    east_michigan = next(
        row
        for row in team_seasons
        if row["season"] == "2022" and row["team_name"] == "Eastern Michigan"
    )
    assert int(east_michigan["duplicate_equivalent_usage_joins"]) >= 1
    assert int(east_michigan["resolved_usage_joins"]) >= 1
    expected_examples = {
        ("2023", "North Texas"): (0.7639999999999999, 0.07400000000000001),
    }
    for key, expected in expected_examples.items():
        row = by_team[key]
        assert (float(row["old_value"]), float(row["new_value"])) == expected
    assert summary["post_repair"]["statuses"] == {
        "complete": 87,
        "entirely_unavailable": 2,
        "partial": 704,
    }
    assert summary["team_seasons_whose_availability_status_improved"] == 17
    assert summary["coverage_2026"]["teams_with_no_incoming_db_transfers"] == 6
    assert summary["coverage_2026"]["teams_with_no_observed_impacts"] == 1
    assert summary["source_availability"]["missing_raw_snapshot_count"] == 37
    assert (
        summary["remaining_failure_counts"]["historical_aggregate_null_team_seasons"]
        == 51
    )
    troy = next(
        row
        for row in team_seasons
        if row["season"] == "2021" and row["team_name"] == "Troy"
    )
    assert troy["offensive_feature_numeric"] == "True"
    assert troy["post_repair_materialized_usage_value"] == "0.163"
    assert troy["post_repair_observed_usage_sum"] == "0.163"
    assert troy["historical_aggregate_repaired"] == "True"
    assert "historical_aggregate_null" not in troy["reason_codes"].split(";")
    assert int(troy["normalization_recovered_usage_joins"]) == 1
    assert all(
        not (
            row["offensive_feature_numeric"] == "True"
            and "historical_aggregate_null" in row["reason_codes"].split(";")
        )
        for row in team_seasons
        if "2021" <= row["season"] <= "2025"
    )
    assert (
        summary["historical_usage_resolution"]["remaining_ambiguous_usage_joins"] == 1
    )
    assert (
        summary["historical_usage_resolution"]["duplicate_equivalent_usage_joins"] == 24
    )
    assert summary["historical_usage_resolution"]["conflicting_usage_value_joins"] == 0
    for field, summary_key in (
        ("resolved_usage_joins", "resolved_usage_joins"),
        ("normalization_recovered_usage_joins", "normalization_recovered_usage_joins"),
        ("duplicate_equivalent_usage_joins", "duplicate_equivalent_usage_joins"),
    ):
        assert (
            sum(int(row[field]) for row in team_seasons if row["season"] != "2026")
            == summary["historical_usage_resolution"][summary_key]
        )
    assert summary["corrected_historical_player_audit"]["record_count"] == len(
        corrected_player_audit
    )
    report = (artifact_root / "report.md").read_text(encoding="utf-8")
    assert "duplicate-equivalent provider observations" in report
    assert "genuinely ambiguous historical usage join remains" in report
    assert "2021 Troy's usage aggregate" in report
    assert before_hashes == {path: _sha256(path) for path in immutable_inputs}


def test_rebuilt_historical_availability_artifact_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = tmp_path / "frozen.csv"
    defensive = tmp_path / "defensive.csv"
    audit_module._write_csv(
        frozen,
        [{"season": 2021, "team_id": "10", "transfer_in_prior_usage_sum": ""}],
        ["season", "team_id", "transfer_in_prior_usage_sum"],
    )
    audit_module._write_csv(
        defensive,
        [],
        [
            "season",
            "destination_team_id",
            "in_model_relevant_population",
            "portal_position_group",
            "impact_status",
        ],
    )
    monkeypatch.setattr(audit_module, "HISTORICAL", frozen)
    monkeypatch.setattr(audit_module, "HISTORICAL_DEFENSIVE_PLAYERS", defensive)

    baseline = [
        {
            "season": 2021,
            "subdivision": "fbs",
            "team_id": "10",
            "team_name": "Alpha",
            "availability_status": "partial",
            "primary_reason": "historical_aggregate_null",
            "reason_codes": "historical_aggregate_null",
            "repair_class": "candidate_repair",
            "repair_candidate_reasons": "historical_aggregate_null",
            "source_coverage_gap": False,
            "checkpoint_status": "historical_timing_unverified",
            "provenance_class": "retrospective_research_reconstruction",
            "incoming_transfers": 1,
            "offensive_resolved": 0,
            "offensive_legitimate_zero": 0,
            "offensive_join_failed": 0,
            "offensive_applicability_unknown": 0,
            "db_incoming": 0,
            "db_resolved": 0,
            "db_unresolved": 0,
            "offensive_feature_numeric": False,
            "db_feature_available": True,
        }
    ]
    materialized = [
        {
            "season": 2021,
            "team_id": "10",
            "team_name": "Alpha",
            "transfer_in_prior_usage_sum": 0.25,
        }
    ]
    player_audit = [
        {
            "season": 2021,
            "portal_index": 0,
            "destination_team_id": "10",
            "in_model_relevant_population": True,
            "d5_resolution_category": audit_module.D5_RESOLVED,
            "d5_feature_value": 0.25,
            "usage_join_status": "joined",
            "usage_candidate_resolution": "single_logical_candidate",
            "normalization_changed_match": False,
        }
    ]

    first = audit_module._rebuild_postrepair_availability(
        baseline, materialized, player_audit
    )
    second = audit_module._rebuild_postrepair_availability(
        baseline, materialized, player_audit
    )
    assert first == second
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    columns = list(first[0])
    audit_module._write_csv(first_path, first, columns)
    audit_module._write_csv(second_path, second, columns)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first[0]["post_repair_materialized_usage_value"] == 0.25
    assert first[0]["reason_codes"] == ""
    assert first[0]["availability_status"] == "complete"


def test_audit_fails_closed_before_writing_when_historical_sources_are_missing(
    tmp_path: Path,
) -> None:
    immutable_inputs = (
        ROOT / "data/processed/transfer_availability_audit/team_seasons.csv",
        ROOT / "data/processed/transfer_availability_audit/summary.json",
        ROOT
        / "data/processed/partial_db_transfer_impact_142/empirical_2026_coverage.csv",
    )
    before_hashes = {path: _sha256(path) for path in immutable_inputs}
    output = tmp_path / "audit"
    with pytest.raises(
        ValueError, match="historical source verification failed closed"
    ):
        audit_module.build(
            output,
            historical_raw_root=tmp_path / "missing_raw",
            materializer_source_root=tmp_path / "missing_model_inputs",
        )
    assert not output.exists()
    assert before_hashes == {path: _sha256(path) for path in immutable_inputs}


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
