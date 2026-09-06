"""Small, schedule-only CFBD acquisition for periodic ranking updates.

This intentionally does not share the historical corpus builder's team-stat
path.  Posterior V1 consumes game results, not YPP/team-stat responses.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

API = "https://api.collegefootballdata.com"
CLASSIFICATIONS = ("fbs", "fcs")
GAME_FIELDS = (
    "id", "season", "week", "seasonType", "startDate", "completed", "neutralSite",
    "conferenceGame", "homeId", "homeTeam", "homeClassification", "homeConference",
    "homePoints", "awayId", "awayTeam", "awayClassification", "awayConference",
    "awayPoints",
)


@dataclass(frozen=True)
class CurrentSeasonAcquisition:
    season: int
    retrieved_at: datetime
    schedules: dict[str, list[dict[str, Any]]]
    response_hashes: dict[str, str]


def _utc(value: datetime | None = None) -> datetime:
    value = datetime.now(UTC) if value is None else value
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _request(client: httpx.Client, params: dict[str, object]) -> httpx.Response:
    for attempt in range(5):
        response = client.get(f"{API}/games", params=params, timeout=60)
        if response.status_code < 500 and response.status_code != 429:
            response.raise_for_status()
            return response
        time.sleep(2**attempt)
    response.raise_for_status()
    return response


def _name(season: int, classification: str) -> str:
    return f"{season}.json" if classification == "fbs" else f"{season}-fcs.json"


def fetch_current_season(
    *,
    season: int,
    root: Path,
    retrieved_at: datetime | None = None,
    client: httpx.Client | None = None,
) -> CurrentSeasonAcquisition:
    """Fetch only the FBS and FCS schedule endpoints and write raw provenance."""
    owns_client = client is None
    if client is None:
        api_key = os.environ.get("CFBD_API_KEY")
        if not api_key:
            raise RuntimeError("CFBD_API_KEY is not configured; add the repository Actions secret")
        client = httpx.Client(headers={"Authorization": f"Bearer {api_key}"})
    raw = root / "data/raw/cfbd/games"
    schedules: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    retrieval_times: list[datetime] = []
    try:
        for classification in CLASSIFICATIONS:
            params = {"year": season, "classification": classification}
            response = _request(client, params)
            payload = response.json()
            if not isinstance(payload, list):
                raise TypeError("CFBD /games response must be a JSON list")
            content = response.content
            response_retrieved_at = _utc(retrieved_at)
            filename = _name(season, classification)
            destination = raw / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            provenance = {
                "content_sha256": digest,
                "endpoint": "/games",
                "parameters": params,
                "retrieved_at": response_retrieved_at.isoformat(),
                "source_kind": "cfbd_api_schedule",
            }
            destination.with_name(f"{filename}.provenance.json").write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            schedules[classification] = payload
            hashes[filename] = digest
            retrieval_times.append(response_retrieved_at)
    finally:
        if owns_client:
            client.close()
    return CurrentSeasonAcquisition(season, max(retrieval_times), schedules, hashes)


def deduplicate_schedule_queries(
    schedules: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], int]:
    """Deduplicate FBS/FCS overlap and reject conflicting payloads."""
    records: dict[str, dict[str, Any]] = {}
    overlaps = 0
    for classification in CLASSIFICATIONS:
        for game in schedules[classification]:
            game_id = str(game["id"])
            previous = records.get(game_id)
            if previous is None:
                records[game_id] = game
            elif previous != game:
                raise ValueError(f"Conflicting CFBD schedule payloads for game {game_id}")
            else:
                overlaps += 1
    return [records[key] for key in sorted(records, key=lambda value: int(value))], overlaps


def _game_row(game: dict[str, Any]) -> dict[str, object]:
    return {field: game.get(field) for field in GAME_FIELDS}


def update_processed_game_corpus(
    *, root: Path, season: int, schedules: dict[str, list[dict[str, Any]]]
) -> dict[str, int]:
    """Replace only one season's rows in the Posterior game corpus.

    Historical raw data and team-game-stat/YPP artifacts are never read or
    rewritten here.  Existing rows for every other season are retained.
    """
    current, overlaps = deduplicate_schedule_queries(schedules)
    path = root / "data/processed/cfbd/games.csv"
    previous: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            previous = list(csv.DictReader(handle))
    historical = [row for row in previous if int(row["season"]) != season]
    rows = historical + [_game_row(game) for game in current]
    rows.sort(key=lambda row: (int(str(row["season"])), int(str(row["id"]))))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GAME_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {"overlap_count": overlaps, "current_game_count": len(current)}
