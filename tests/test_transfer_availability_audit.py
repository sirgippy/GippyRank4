"""Checks that the committed availability inventory matches its frozen inputs."""

from __future__ import annotations

import csv
import importlib
import subprocess
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


def test_committed_artifacts_match_fresh_cli_output(tmp_path):
    script = Path(audit_module.__file__)
    subprocess.run(
        [sys.executable, str(script), "--output", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    for name in (
        "team_seasons.csv",
        "affected_2026.csv",
        "summary.json",
        "reason_taxonomy.json",
    ):
        assert (tmp_path / name).read_bytes() == (DEFAULT_OUTPUT / name).read_bytes(), (
            name
        )


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


def test_unavailable_requires_unresolved_db_input():
    rows, _ = audit()
    unavailable = [
        row for row in rows if row["availability_status"] == "entirely_unavailable"
    ]
    assert unavailable
    assert all(
        row["incoming_transfers"] > 0
        and not row["offensive_feature_numeric"]
        and row["db_unresolved"] > 0
        and row["db_resolved"] == 0
        for row in unavailable
    )
    ohio_state_2021 = next(
        row
        for row in rows
        if row["season"] == 2021 and row["team_name"] == "Ohio State"
    )
    assert ohio_state_2021["incoming_transfers"] > 0
    assert not ohio_state_2021["offensive_feature_numeric"]
    assert ohio_state_2021["db_incoming"] == 0
    assert ohio_state_2021["availability_status"] == "partial"
