from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from gippyrank.posterior.snapshots import build_snapshot, snapshot_id


def _write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _root(tmp_path: Path) -> Path:
    fields = ["season", "subdivision", "team_id", "team_name", "pmf"]
    rows = [
        {
            "season": "2026",
            "subdivision": "fbs",
            "team_id": "1",
            "team_name": "One",
            "pmf": "[0.7,0.3]",
        },
        {
            "season": "2026",
            "subdivision": "fbs",
            "team_id": "2",
            "team_name": "Two",
            "pmf": "[0.3,0.7]",
        },
        {
            "season": "2026",
            "subdivision": "fcs",
            "team_id": "3",
            "team_name": "Three",
            "pmf": "[0.5,0.5]",
        },
    ]
    for family in ("context", "history"):
        _write(
            tmp_path / f"data/processed/preseason/{family}/annual/2026/predictions.csv",
            fields,
            rows,
        )
    game_fields = [
        "id",
        "season",
        "week",
        "seasonType",
        "startDate",
        "completed",
        "neutralSite",
        "conferenceGame",
        "homeId",
        "homeTeam",
        "homeClassification",
        "homeConference",
        "homePoints",
        "awayId",
        "awayTeam",
        "awayClassification",
        "awayConference",
        "awayPoints",
    ]
    _write(
        tmp_path / "data/processed/cfbd/games.csv",
        game_fields,
        [
            {
                "id": "early",
                "season": "2026",
                "week": "1",
                "seasonType": "regular",
                "startDate": "2026-08-29T00:00:00Z",
                "completed": "True",
                "neutralSite": "False",
                "conferenceGame": "True",
                "homeId": "1",
                "homeTeam": "One",
                "homeClassification": "fbs",
                "homeConference": "",
                "homePoints": "20",
                "awayId": "3",
                "awayTeam": "Three",
                "awayClassification": "fcs",
                "awayConference": "",
                "awayPoints": "10",
            },
            {
                "id": "lower",
                "season": "2026",
                "week": "1",
                "seasonType": "regular",
                "startDate": "2026-08-28T00:00:00Z",
                "completed": "True",
                "neutralSite": "False",
                "conferenceGame": "False",
                "homeId": "1",
                "homeTeam": "One",
                "homeClassification": "fbs",
                "homeConference": "",
                "homePoints": "20",
                "awayId": "4",
                "awayTeam": "Four",
                "awayClassification": "ii",
                "awayConference": "",
                "awayPoints": "10",
            },
            {
                "id": "later",
                "season": "2026",
                "week": "2",
                "seasonType": "regular",
                "startDate": "2026-09-10T00:00:00Z",
                "completed": "True",
                "neutralSite": "False",
                "conferenceGame": "True",
                "homeId": "1",
                "homeTeam": "One",
                "homeClassification": "fbs",
                "homeConference": "",
                "homePoints": "20",
                "awayId": "2",
                "awayTeam": "Two",
                "awayClassification": "fbs",
                "awayConference": "",
                "awayPoints": "10",
            },
        ],
    )
    return tmp_path


def test_preseason_snapshot_is_prior_only_and_complete(tmp_path: Path) -> None:
    root = _root(tmp_path)
    snapshot = build_snapshot(
        season=2026,
        cutoff=None,
        prior_family="context",
        snapshot_type="preseason",
        root=root,
    )
    metadata = json.loads((snapshot.directory / "metadata.json").read_text())
    assert metadata["included_game_count"] == 0
    assert metadata["canonical_public_model"] is True
    assert metadata["prior_artifact_sha256"]
    assert {path.name for path in snapshot.directory.iterdir()} >= {
        "metadata.json",
        "rankings.json",
        "rankings.csv",
        "posterior_pmfs.csv",
        "included_games.csv",
        "diagnostics.json",
    }
    assert len(json.loads((snapshot.directory / "rankings.json").read_text())) == 3


def test_cutoff_and_lower_division_policy_are_explicit(tmp_path: Path) -> None:
    root = _root(tmp_path)
    # No likelihood is needed when cutoff is before the first game; this proves
    # filtering happens before any inference attempt.
    snapshot = build_snapshot(
        season=2026,
        cutoff=date(2026, 8, 28),
        prior_family="history",
        snapshot_type="weekly",
        root=root,
    )
    assert snapshot.metadata["included_game_count"] == 0
    assert snapshot.metadata["excluded_lower_division_games"] == 1
    assert snapshot.metadata["canonical_public_model"] is False


def test_snapshot_ids_are_stable_and_human_readable() -> None:
    assert snapshot_id(2026, "preseason", "context") == "2026-preseason-context"
    assert (
        snapshot_id(2026, "weekly", "history", date(2026, 9, 1))
        == "2026-weekly-2026-09-01-history"
    )
