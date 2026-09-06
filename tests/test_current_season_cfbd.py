from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from gippyrank.data.cfbd import (
    CurrentSeasonAcquisition,
    deduplicate_schedule_queries,
    fetch_current_season,
    update_processed_game_corpus,
)


def _game(game_id: int, *, completed: bool = True, away_class: str = "fcs") -> dict:
    return {
        "id": game_id, "season": 2026, "week": 1, "seasonType": "regular",
        "startDate": "2026-08-29T00:00:00Z", "completed": completed,
        "neutralSite": False, "conferenceGame": False, "homeId": 1,
        "homeTeam": "One", "homeClassification": "fbs", "homeConference": "A",
        "homePoints": 20 if completed else None, "awayId": 3, "awayTeam": "Three",
        "awayClassification": away_class, "awayConference": "B",
        "awayPoints": 10 if completed else None,
    }


def test_current_acquisition_requests_only_fbs_and_fcs_games_with_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[_game(100 if request.url.params["classification"] == "fbs" else 101)])

    fbs_time = datetime(2026, 9, 12, 15, tzinfo=UTC)
    fcs_time = datetime(2026, 9, 12, 15, 8, tzinfo=UTC)
    retrieval_times = iter((fbs_time, fcs_time))
    monkeypatch.setattr("gippyrank.data.cfbd._utc", lambda _: next(retrieval_times))
    with httpx.Client(transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer test-secret"}) as client:
        acquisition = fetch_current_season(season=2026, root=tmp_path, client=client)

    assert [request.url.path for request in requests] == ["/games", "/games"]
    assert [request.url.params["classification"] for request in requests] == ["fbs", "fcs"]
    assert all(request.headers["Authorization"] == "Bearer test-secret" for request in requests)
    assert acquisition.retrieved_at == fbs_time
    assert acquisition.source_retrieval_times == {"fbs": fbs_time, "fcs": fcs_time}
    for filename, classification, timestamp in (("2026.json", "fbs", fbs_time), ("2026-fcs.json", "fcs", fcs_time)):
        provenance = json.loads((tmp_path / f"data/raw/cfbd/games/{filename}.provenance.json").read_text())
        assert provenance == {
            "content_sha256": acquisition.response_hashes[filename],
            "endpoint": "/games", "parameters": {"year": 2026, "classification": classification},
            "retrieved_at": timestamp.isoformat(), "source_kind": "cfbd_api_schedule",
        }


def test_overlap_deduplicates_conflicts_fail_and_historical_rows_are_retained(tmp_path: Path) -> None:
    processed = tmp_path / "data/processed/cfbd/games.csv"
    processed.parent.mkdir(parents=True)
    historical = _game(1)
    historical["season"] = 2025
    with processed.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(historical))
        writer.writeheader()
        writer.writerow(historical)
    schedules = {"fbs": [_game(100), _game(101, completed=False)], "fcs": [_game(100)]}
    report = update_processed_game_corpus(root=tmp_path, season=2026, schedules=schedules)
    with processed.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert report == {"overlap_count": 1, "current_game_count": 2}
    assert rows[0] == {key: str(value) if value is not None else "" for key, value in historical.items()}
    assert rows[-1]["completed"] == "False"  # raw corpus retains future games
    conflict = dict(_game(100))
    conflict["homePoints"] = 99
    with pytest.raises(ValueError, match="Conflicting"):
        deduplicate_schedule_queries({"fbs": [_game(100)], "fcs": [conflict]})


def test_acquisition_object_is_simple_to_mock_for_weekly_orchestration() -> None:
    timestamp = datetime(2026, 9, 1, tzinfo=UTC)
    value = CurrentSeasonAcquisition(2026, timestamp, {"fbs": timestamp, "fcs": timestamp}, {"fbs": [], "fcs": []}, {})
    assert value.season == 2026
