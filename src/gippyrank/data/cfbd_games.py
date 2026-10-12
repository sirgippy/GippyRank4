"""Read existing processed or cached CFBD game schedules for rank joins."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from gippyrank.data.massey import historical_subdivision_for_id

GAME_JOIN_FIELDS = (
    "id",
    "season",
    "week",
    "startDate",
    "neutralSite",
    "homePoints",
    "homeTeam",
    "homeId",
    "homeClassification",
    "awayPoints",
    "awayTeam",
    "awayId",
    "awayClassification",
)
_RAW_GAME_FILE = re.compile(r"^(?P<season>\d{4})(?:-fcs)?\.json$")


def apply_historical_game_subdivisions(
    games: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Apply the same identity-based model cohort decisions used for ranks."""
    result = []
    for game in games:
        row = game.copy()
        season = int(row["season"])
        for side in ("home", "away"):
            identity = f"{side}Id"
            classification = f"{side}Classification"
            if row.get(identity) and row.get(classification):
                row[classification] = historical_subdivision_for_id(
                    season, row[identity], row[classification]
                )
        result.append(row)
    return result


def read_cfbd_game_rows(
    source: Path, *, apply_cohort_decisions: bool = True
) -> tuple[list[dict[str, str]], list[Path]]:
    """Read a processed games CSV or deduplicated cached raw schedule directory.

    This helper only reads existing schedule data. It does not fetch games or
    alter the source artifacts.
    """
    if source.is_file():
        with source.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = set(GAME_JOIN_FIELDS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Game CSV is missing join fields: {sorted(missing)}")
            rows = [
                {field: str(row.get(field) or "") for field in GAME_JOIN_FIELDS}
                for row in reader
            ]
            return (
                apply_historical_game_subdivisions(rows)
                if apply_cohort_decisions
                else rows,
                [source],
            )
    if not source.is_dir():
        raise FileNotFoundError(source)

    records: dict[str, dict[str, str]] = {}
    input_paths = []
    for path in sorted(source.glob("*.json")):
        match = _RAW_GAME_FILE.fullmatch(path.name)
        if match is None or not 2003 <= int(match.group("season")) <= 2025:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"cached CFBD schedule is not a list: {path}")
        input_paths.append(path)
        for item in payload:
            row = {field: "" if item.get(field) is None else str(item[field]) for field in GAME_JOIN_FIELDS}
            game_id = row["id"]
            if not game_id:
                raise ValueError(f"cached schedule row has no id: {path}")
            previous = records.get(game_id)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting cached schedule rows for game {game_id}")
            records[game_id] = row
    if not input_paths:
        raise FileNotFoundError(f"No cached CFBD schedule files for 2003–2025 under {source}")
    rows = sorted(records.values(), key=lambda row: int(row["id"]))
    return (
        apply_historical_game_subdivisions(rows) if apply_cohort_decisions else rows,
        input_paths,
    )
