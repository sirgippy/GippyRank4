"""Focused tests for the offline primitive CFBD box-score audit."""

from __future__ import annotations

import json
from pathlib import Path

from gippyrank.data.cfbd import raw_stat_payload_paths
from gippyrank.research.box_score_audit import (
    build_audit,
    coverage_rows,
    load_stat_audit,
    parse_candidate_fields,
    parse_compound,
    parse_number,
)


def _team(team_id: int, name: str, stats: list[dict[str, object]]) -> dict[str, object]:
    return {
        "teamId": team_id,
        "team": name,
        "homeAway": "home",
        "stats": stats,
    }


def _stat(category: str, value: object) -> dict[str, object]:
    return {"category": category, "stat": value}


def _write_payload(path: Path, game_id: int, team_id: int, total_yards: int) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": game_id,
                    "teams": [
                        _team(
                            team_id,
                            "Alpha",
                            [
                                _stat("totalYards", str(total_yards)),
                                _stat("rushingYards", "100"),
                                _stat("rushingAttempts", "30"),
                                _stat("netPassingYards", "100"),
                                _stat("completionAttempts", "10-20"),
                                _stat("interceptions", "1"),
                                _stat("fumblesLost", "0"),
                                _stat("turnovers", "1"),
                            ],
                        )
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )


def test_compound_and_decimal_parsing_preserves_missing_values() -> None:
    assert parse_compound("17-29") == (17, 29)
    assert parse_compound("10-60") == (10, 60)
    assert parse_compound("not-a-stat") is None
    assert parse_number("2.5") == 2.5
    assert parse_number(None) is None

    fields = parse_candidate_fields(
        [
            _stat("completionAttempts", "8-21"),
            _stat("rushingAttempts", "67"),
            _stat("interceptions", "1"),
            _stat("fumblesLost", "2"),
        ]
    )
    assert fields["completions"] == 8
    assert fields["pass_attempts"] == 21
    assert fields["offensive_plays_derived"] == 88
    assert fields["interceptions_thrown"] == 1
    assert fields["total_fumbles"] is None
    assert fields["fumbles_lost"] == 2


def test_payload_discovery_excludes_provenance_sidecars(tmp_path: Path) -> None:
    directory = tmp_path / "stats"
    directory.mkdir()
    payload = directory / "2024-fbs-week1-regular.json"
    payload.write_text("[]", encoding="utf-8")
    (directory / "2024-fbs-week1-regular.json.provenance.json").write_text(
        "{}", encoding="utf-8"
    )
    (directory / "notes.json").write_text("[]", encoding="utf-8")
    assert raw_stat_payload_paths(directory) == [payload]


def test_duplicate_and_conflict_handling_is_explicit(tmp_path: Path) -> None:
    directory = tmp_path / "stats"
    directory.mkdir()
    first = directory / "2024-fbs-week1-regular.json"
    second = directory / "2024-fcs-week1-regular.json"
    _write_payload(first, 10, 1, 200)
    _write_payload(second, 10, 1, 200)
    audit = load_stat_audit(directory)
    assert audit.exact_duplicates == 1
    assert not audit.conflicts

    _write_payload(second, 10, 1, 201)
    audit = load_stat_audit(directory)
    assert audit.exact_duplicates == 0
    assert len(audit.conflicts) == 1
    assert audit.rows[(10, 1)]["total_yards"] == 200


def test_coverage_is_pairing_specific_and_does_not_zero_fill() -> None:
    games = {
        "1": {
            "id": 1,
            "season": 2020,
            "completed": True,
            "homeId": 10,
            "awayId": 11,
            "homeClassification": "fbs",
            "awayClassification": "fbs",
        },
        "2": {
            "id": 2,
            "season": 2020,
            "completed": True,
            "homeId": 20,
            "awayId": 21,
            "homeClassification": "fbs",
            "awayClassification": "fcs",
        },
        "3": {
            "id": 3,
            "season": 2020,
            "completed": True,
            "homeId": 30,
            "awayId": 31,
            "homeClassification": "fcs",
            "awayClassification": "fcs",
        },
    }
    rows = {
        (1, 10): {"total_yards": 300},
        (1, 11): {"total_yards": 250},
        (2, 20): {"total_yards": 200},
        (2, 21): {"total_yards": None},
    }
    result = coverage_rows(games, rows, ["total_yards"])
    fbs_fbs = next(row for row in result if row["pairing"] == "FBS-FBS")
    fbs_fcs = next(row for row in result if row["pairing"] == "FBS-FCS")
    fcs_fcs = next(row for row in result if row["pairing"] == "FCS-FCS")
    assert fbs_fbs["games_with_both_usable"] == 1
    assert fbs_fcs["games_with_both_usable"] == 0
    assert fbs_fcs["missing_team_games"] == 1
    assert fcs_fcs["team_games_with_value"] == 0


def test_audit_build_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path
    games = root / "data/raw/cfbd/games"
    stats = root / "data/raw/cfbd/game_stats"
    games.mkdir(parents=True)
    stats.mkdir(parents=True)
    schedule = {
        "id": 10,
        "season": 2024,
        "week": 1,
        "seasonType": "regular",
        "completed": True,
        "homeId": 1,
        "homeTeam": "Alpha",
        "homeClassification": "fbs",
        "homePoints": 21,
        "awayId": 2,
        "awayTeam": "Beta",
        "awayClassification": "fcs",
        "awayPoints": 14,
        "homeLineScores": [7, 7, 7, 0],
        "awayLineScores": [7, 0, 7, 0],
    }
    (games / "2024.json").write_text(json.dumps([schedule]), encoding="utf-8")
    (stats / "2024-fbs-week1-regular.json").write_text(
        json.dumps(
            [
                {
                    "id": 10,
                    "teams": [
                        _team(1, "Alpha", [_stat("totalYards", "300")]),
                        _team(2, "Beta", [_stat("totalYards", "200")]),
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    first = root / "out1"
    second = root / "out2"
    build_audit(root, first)
    build_audit(root, second)
    first_files = sorted(path.relative_to(first) for path in first.rglob("*"))
    second_files = sorted(path.relative_to(second) for path in second.rglob("*"))
    assert first_files == second_files
    for relative in first_files:
        if (first / relative).is_file():
            assert (first / relative).read_bytes() == (second / relative).read_bytes()
