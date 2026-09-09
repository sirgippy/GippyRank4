"""Materialize the frozen schedule used by historical forecast validation.

The raw CFBD schedule responses are intentionally ignored local acquisition
artifacts. This command turns the four historical FBS/FCS response pairs into
one small, deterministic regular-season schedule snapshot and records the raw
response hashes beside it. Validation consumes the tracked snapshot, so a
clean checkout does not depend on a developer's raw-data cache.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SEASONS = (2022, 2023, 2024, 2025)
RAW_SUFFIXES = ("", "-fcs")
OUTPUT = ROOT / "data/validation/season_forecast_schedule.csv"
PROVENANCE = ROOT / "data/validation/season_forecast_schedule.provenance.json"
FIELDNAMES = (
    "game_id",
    "season",
    "week",
    "season_type",
    "start_date",
    "completed",
    "status",
    "neutral_site",
    "home_team_id",
    "home_team_name",
    "home_subdivision",
    "home_conference",
    "home_points",
    "away_team_id",
    "away_team_name",
    "away_subdivision",
    "away_conference",
    "away_points",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _normalise(game: dict[str, Any]) -> dict[str, str]:
    return {
        "game_id": _text(game.get("id")),
        "season": _text(game.get("season")),
        "week": _text(game.get("week")),
        "season_type": _text(game.get("seasonType") or "regular"),
        "start_date": _text(game.get("startDate")),
        "completed": _text(game.get("completed")),
        "status": _text(game.get("status", game.get("gameStatus"))),
        "neutral_site": _text(game.get("neutralSite")),
        "home_team_id": _text(game.get("homeId")),
        "home_team_name": _text(game.get("homeTeam")),
        "home_subdivision": _text(game.get("homeClassification")),
        "home_conference": _text(game.get("homeConference")),
        "home_points": _text(game.get("homePoints")),
        "away_team_id": _text(game.get("awayId")),
        "away_team_name": _text(game.get("awayTeam")),
        "away_subdivision": _text(game.get("awayClassification")),
        "away_conference": _text(game.get("awayConference")),
        "away_points": _text(game.get("awayPoints")),
    }


def materialize(root: Path = ROOT) -> tuple[Path, Path]:
    raw_directory = root / "data/raw/cfbd/games"
    output = root / OUTPUT.relative_to(ROOT)
    provenance_path = root / PROVENANCE.relative_to(ROOT)
    games: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}

    for season in SEASONS:
        for suffix in RAW_SUFFIXES:
            path = raw_directory / f"{season}{suffix}.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing historical schedule source: {path}")
            source_hashes[path.relative_to(root).as_posix()] = _sha256(path)
            for game in json.loads(path.read_text(encoding="utf-8")):
                if int(game["season"]) != season:
                    raise ValueError(f"{path} contains a game from another season")
                game_id = _text(game.get("id"))
                if not game_id:
                    raise ValueError(f"{path} contains a game without an ID")
                games.setdefault(game_id, game)

    rows = [
        _normalise(game)
        for game in games.values()
        if _text(game.get("seasonType") or "regular").casefold() in {"", "regular"}
    ]
    rows.sort(key=lambda row: (int(row["season"]), row["start_date"], row["game_id"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        "schema_version": "1.0",
        "source_kind": "cfbd_api_schedule",
        "source_endpoint": "/games",
        "materializer": "scripts/materialize_historical_forecast_schedule.py",
        "materialized_path": output.relative_to(root).as_posix(),
        "materialized_schedule_sha256": _sha256(output),
        "materialized_row_count": len(rows),
        "seasons": list(SEASONS),
        "fields": list(FIELDNAMES),
        "source_selection": (
            "For each season, read <season>.json then <season>-fcs.json, keep the "
            "first row for each game ID, and retain regular-season rows."
        ),
        "raw_source_sha256": dict(sorted(source_hashes.items())),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output, provenance_path


def main() -> None:
    output, provenance = materialize()
    print(output)
    print(provenance)


if __name__ == "__main__":
    main()
