"""Checks that the committed availability inventory matches its frozen inputs."""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
audit_module = importlib.import_module("audit_transfer_availability")
CURRENT = audit_module.CURRENT
DEFAULT_OUTPUT = audit_module.DEFAULT_OUTPUT
audit = audit_module.audit


def _csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_committed_inventory_matches_frozen_source_audits():
    rows, summary = audit()
    committed = _csv_rows(DEFAULT_OUTPUT / "team_seasons.csv")
    assert len(rows) == len(committed) == 793
    assert [(str(row["season"]), row["team_id"]) for row in rows] == [
        (row["season"], row["team_id"]) for row in committed
    ]
    assert [row["reason_codes"] for row in rows] == [
        row["reason_codes"] for row in committed
    ]
    assert summary == json.loads((DEFAULT_OUTPUT / "summary.json").read_text())


def test_every_2026_context_team_is_classified_and_failures_are_not_complete():
    rows, summary = audit()
    expected = {
        row["team_id"] for row in _csv_rows(CURRENT / "transfer_team_audit.csv")
    }
    current = [row for row in rows if row["season"] == 2026]
    assert {row["team_id"] for row in current} == expected
    assert all(
        row["availability_status"] != "complete"
        for row in current
        if row["offensive_join_failed"]
        or row["offensive_applicability_unknown"]
        or row["db_unresolved"]
    )
    assert summary["current_2026"]["db_unresolved"] == 86
    assert summary["current_2026"]["all_snapshots_after_cutoff"]
