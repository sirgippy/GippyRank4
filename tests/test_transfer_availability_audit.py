"""Hermetic availability logic and integrity checks for tracked output."""

from __future__ import annotations

import csv
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests/fixtures/transfer_availability"
sys.path.insert(0, str(ROOT / "scripts"))
audit_module = importlib.import_module("audit_transfer_availability")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fixture_audit():
    return audit_module.audit(
        FIXTURE_ROOT / "current",
        historical_features_path=FIXTURE_ROOT / "historical_features.csv",
        historical_offense_root=FIXTURE_ROOT / "historical_offense",
        historical_defense_path=(
            FIXTURE_ROOT / "historical_defense/transfer_player_audit.csv"
        ),
        expected_current_team_count=5,
    )


def test_fixture_teams_are_classified_and_failures_cannot_be_complete():
    rows, summary = _fixture_audit()
    expected = {
        row["team_id"]
        for row in _csv_rows(FIXTURE_ROOT / "current/transfer_team_audit.csv")
    }
    current = [row for row in rows if row["season"] == 2026]
    assert {row["team_id"] for row in current} == expected
    assert len(current) == len(expected) == 5
    assert all(
        row["availability_status"] != "complete"
        for row in current
        if row["offensive_join_failed"]
        or row["offensive_applicability_unknown"]
        or row["db_unresolved"]
    )
    status_by_team = {row["team_id"]: row["availability_status"] for row in current}
    assert status_by_team == {
        "101": "complete",
        "102": "partial",
        "103": "partial",
        "104": "entirely_unavailable",
        "105": "complete",
    }
    assert summary["current_2026"]["db_unresolved"] == 2
    assert summary["current_2026"]["all_snapshots_after_cutoff"] is True
    assert summary["current_2026"]["snapshot_count"] == 1
    assert summary["current_2026"]["noncanonical_diagnostic_snapshot_count"] == 1


def test_natural_zeros_remain_distinct_from_unresolved_evidence():
    rows, _ = _fixture_audit()
    current = {row["team_id"]: row for row in rows if row["season"] == 2026}

    gamma = current["103"]
    assert gamma["offensive_legitimate_zero"] == 1
    assert gamma["db_unresolved"] == 1
    assert gamma["availability_status"] == "partial"

    epsilon = current["105"]
    assert epsilon["db_incoming"] == 1
    assert epsilon["db_resolved"] == 1
    assert epsilon["db_unresolved"] == 0
    assert epsilon["db_feature_available"] is True

    delta = current["104"]
    assert delta["offensive_join_failed"] == 1
    assert delta["offensive_resolved"] == 0
    assert delta["offensive_legitimate_zero"] == 0
    assert delta["db_unresolved"] == 1
    assert delta["availability_status"] == "entirely_unavailable"


def test_fixture_artifact_generation_is_deterministic(tmp_path: Path):
    rows, summary = _fixture_audit()
    first = tmp_path / "first"
    second = tmp_path / "second"
    audit_module.write_artifacts(first, rows, summary)
    audit_module.write_artifacts(second, rows, summary)

    for name in (
        "team_seasons.csv",
        "affected_2026.csv",
        "summary.json",
        "reason_taxonomy.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_committed_availability_artifacts_are_internally_consistent():
    output = audit_module.DEFAULT_OUTPUT
    rows = _csv_rows(output / "team_seasons.csv")
    affected = _csv_rows(output / "affected_2026.csv")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    taxonomy = json.loads((output / "reason_taxonomy.json").read_text(encoding="utf-8"))

    current = [row for row in rows if row["season"] == "2026"]
    assert len(current) == 138
    assert len({row["team_id"] for row in current}) == 138
    assert (
        dict(sorted(Counter(row["availability_status"] for row in rows).items()))
        == (summary["statuses"])
    )
    assert affected == [
        row
        for row in rows
        if row["season"] == "2026" and row["availability_status"] != "complete"
    ]
    assert summary["current_2026"]["db_unresolved"] == 54
    assert summary["current_2026"]["snapshot_count"] == 134
    assert set(taxonomy["reason_codes"]) == set(audit_module.REASONS)
    assert set(taxonomy["reason_codes"]) == set(audit_module.REPAIR_CLASS)
