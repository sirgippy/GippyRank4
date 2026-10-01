from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter
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


def test_committed_integrated_artifacts_match_their_reconciliation_summary() -> None:
    artifact_root = ROOT / "data/processed/transfer_data_repair"
    summary = json.loads((artifact_root / "summary.json").read_text(encoding="utf-8"))
    team_seasons = _rows(artifact_root / "team_seasons.csv")
    coverage = _rows(artifact_root / "db_coverage_2026.csv")
    historical = [row for row in team_seasons if 2021 <= int(row["season"]) <= 2025]

    assert len(team_seasons) == 793
    assert len(historical) == 655
    assert dict(
        sorted(Counter(row["availability_status"] for row in historical).items())
    ) == {"complete": 83, "entirely_unavailable": 2, "partial": 570}
    assert (
        dict(
            sorted(Counter(row["availability_status"] for row in team_seasons).items())
        )
        == summary["historical"]["all_years_combined_checkpoint"][
            "availability_status_counts"
        ]
    )
    assert (
        sum(int(row["total_count"]) for row in coverage)
        == summary["2026"]["incoming_db_transfers"]
        == 604
    )
    assert (
        sum(int(row["observed_count"]) for row in coverage)
        == summary["2026"]["observed_db_impacts"]
        == 550
    )
    assert sum(int(row["missing_count"]) for row in coverage) == 54

    before_manifest = json.loads(
        (artifact_root / "before_source_manifest.json").read_text(encoding="utf-8")
    )
    after_manifest = json.loads(
        (artifact_root / "after_source_manifest.json").read_text(encoding="utf-8")
    )
    for manifest in (before_manifest, after_manifest):
        canonical = [
            item for item in manifest["snapshots"] if item.get("canonical", True)
        ]
        assert canonical
        assert all(item.get("sha256") for item in canonical)

    assert summary["integration"]["historical_authority"].startswith("PR #150")
    assert summary["integration"]["2026_authority"].startswith("PR #151")
    assert summary["context_1_3_invariance"]["model_facing_inputs_changed"] is False
    assert (
        summary["2026"]["context_1_3_invariance"]["published_rankings_regenerated"]
        is False
    )


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


def test_historical_full_replay_fails_closed_without_local_raw_sources(
    tmp_path: Path,
) -> None:
    replay_module = importlib.import_module("replay_historical_transfer_features")
    manifest_path = (
        ROOT / "data/processed/transfer_production_audit/source_manifest.json"
    )

    with pytest.raises(
        ValueError, match="historical source verification failed closed"
    ):
        replay_module.verify_historical_sources(
            tmp_path / "empty_raw_root", manifest_path=manifest_path
        )


def test_materializer_source_root_uses_explicit_tracked_fixture() -> None:
    fixture_root = ROOT / "tests/fixtures/historical_materializer"
    assert (
        integrated_module._materializer_source_root(fixture_root)
        == fixture_root.resolve()
    )

    with pytest.raises(FileNotFoundError, match="team_season_rank_distributions.csv"):
        integrated_module._materializer_source_root(ROOT / "tests/fixtures")


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
