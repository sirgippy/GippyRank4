from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from gippyrank.data.cfbd import raw_stat_payload_paths

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "build_cfbd_corpus", ROOT / "scripts/build_cfbd_corpus.py"
)
assert _SPEC is not None and _SPEC.loader is not None
corpus = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(corpus)


def _game(game_id: int) -> dict[str, object]:
    return {
        "id": game_id,
        "season": 2025,
        "week": 1,
        "seasonType": "regular",
        "homeId": 1,
        "awayId": 2,
    }


def _stat_payload(game_id: int) -> list[dict[str, object]]:
    def team(team_id: int, total_yards: int) -> dict[str, object]:
        return {
            "teamId": team_id,
            "stats": [
                {"category": "totalYards", "stat": str(total_yards)},
                {"category": "rushingAttempts", "stat": "20"},
                {"category": "completionAttempts", "stat": "10-20"},
            ],
        }

    return [{"id": game_id, "teams": [team(1, 300), team(2, 200)]}]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_raw_stat_payload_paths_discovers_ordinary_payload(tmp_path: Path) -> None:
    payload = tmp_path / "2025-fbs-week1-regular.json"
    _write_json(payload, [])

    assert raw_stat_payload_paths(tmp_path) == [payload]


def test_raw_stat_payload_paths_excludes_provenance_sidecar(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "2025-fbs-week1-regular.json"
    _write_json(payload, [])
    _write_json(
        tmp_path / "2025-fbs-week1-regular.json.provenance.json",
        {"endpoint": "/games/teams", "content_sha256": "hash"},
    )

    assert raw_stat_payload_paths(tmp_path) == [payload]


def test_build_stats_ignores_provenance_and_cannot_create_fake_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(corpus, "RAW_STATS", tmp_path)
    _write_json(tmp_path / "2025-fbs-week1-regular.json", _stat_payload(101))
    _write_json(
        tmp_path / "2025-fbs-week1-regular.json.provenance.json",
        {
            "content_sha256": "hash",
            "endpoint": "/games/teams",
            "parameters": {
                "year": 2025,
                "week": 1,
                "seasonType": "regular",
                "classification": "fbs",
            },
            "retrieved_at": "2026-09-07T00:00:00+00:00",
            "source_kind": "cfbd_api",
        },
    )

    rows, conflicts, duplicates = corpus.build_stats({(2025, "fbs"): [_game(101)]})

    assert rows == [
        {
            "game_id": 101,
            "team_id": 1,
            "opponent_id": 2,
            "total_yards": 300,
            "plays": 40,
            "yards_per_play": 7.5,
        },
        {
            "game_id": 101,
            "team_id": 2,
            "opponent_id": 1,
            "total_yards": 200,
            "plays": 40,
            "yards_per_play": 5.0,
        },
    ]
    assert conflicts == []
    assert duplicates == 0
    assert {row["game_id"] for row in rows} == {101}


def test_fetch_raw_continues_to_write_provenance_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = [{"id": 101, "teams": []}]
    response = httpx.Response(200, json=payload)
    monkeypatch.setattr(corpus, "request_json", lambda *_args, **_kwargs: response)
    destination = tmp_path / "2025-fbs-week1-regular.json"

    assert (
        corpus.fetch_raw(object(), "/games/teams", {"year": 2025}, destination)
        == payload
    )
    assert destination.is_file()
    assert destination.with_name(f"{destination.name}.provenance.json").is_file()


def test_raw_game_response_count_counts_payloads_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(corpus, "RAW_STATS", tmp_path)
    for name in (
        "2024-fbs-week1-regular.json",
        "2025-fcs-week2-postseason.json",
    ):
        _write_json(tmp_path / name, [])
        _write_json(tmp_path / f"{name}.provenance.json", {"endpoint": "/games/teams"})
    _write_json(tmp_path / "notes.json", {})

    assert corpus.raw_game_response_count({(2025, "fbs"): [], (2025, "fcs"): []}) == 4


def test_raw_stat_payload_paths_are_deterministic_across_seasons_and_queries(
    tmp_path: Path,
) -> None:
    names = [
        "2025-fbs-week10-regular.json",
        "2024-fcs-week2-postseason.json",
        "2025-fcs-week1-regular.json",
        "2024-fbs-week1-regular.json",
    ]
    for name in names:
        _write_json(tmp_path / name, [])
        _write_json(tmp_path / f"{name}.provenance.json", {"endpoint": "/games/teams"})

    assert [path.name for path in raw_stat_payload_paths(tmp_path)] == sorted(names)
