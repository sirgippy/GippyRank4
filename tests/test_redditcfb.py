from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gippyrank.redditcfb import (
    TeamHandleMappingError,
    audit_team_handle_coverage,
    load_team_handle_mapping,
)

ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "data/reference/redditcfb_team_handles.csv"


def test_checked_in_mapping_has_current_fbs_coverage_and_unique_handles() -> None:
    table = load_team_handle_mapping(MAPPING)

    assert len(table.handles) == 138
    assert len(table.handles) == len(set(table.handles.values()))
    assert all(table.handles.values())
    assert table.handles["58"] == "usf"
    assert table.handles["2026"] == "appalachianstate"
    assert table.handles["41"] == "connecticut"


def test_mapping_audit_reports_unmapped_identities_and_aliases() -> None:
    table = load_team_handle_mapping(MAPPING)
    audit = audit_team_handle_coverage(
        [("58", "South Florida"), ("999", "Unmapped")], table
    )

    assert audit["current_fbs_team_count"] == 2
    assert audit["mapped_count"] == 1
    assert audit["unmapped_teams"] == [{"cfbd_team_id": "999", "team_name": "Unmapped"}]
    assert {
        (item["cfbd_team_id"], item["redditcfb_handle"])
        for item in audit["manually_resolved_aliases"]
    } == {("58", "usf")}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cfbd_team_id", "1", "duplicate CFBD IDs"),
        ("redditcfb_handle", "one", "duplicate r/CFB handles"),
    ],
)
def test_duplicate_mapping_keys_fail_closed(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    rows = [
        {
            "cfbd_team_id": "1",
            "team_name": "One",
            "redditcfb_handle": "one",
        },
        {
            "cfbd_team_id": "2",
            "team_name": "Two",
            "redditcfb_handle": "two",
        },
    ]
    rows[1][field] = value
    path = tmp_path / "handles.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(TeamHandleMappingError, match=message):
        load_team_handle_mapping(path)
