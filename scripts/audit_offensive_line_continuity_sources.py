"""Inventory local CFBD artifacts relevant to the issue 174 OL continuity study.

This is an offline filesystem audit. It does not make API requests, inspect
credentials, modify raw data, or estimate offensive-line participation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/processed/offensive_line_continuity_issue_174"
CFBD_RAW_RELATIVE = "raw/cfbd"

SOURCE_DIRECTORIES = {
    "team_schedules": ("raw/cfbd/games", "season schedules and results"),
    "team_game_box_scores": (
        "raw/cfbd/game_stats",
        "team-level game box scores; not player participation",
    ),
    "preseason_returning_production": (
        "raw/cfbd/preseason/returning",
        "team-season returning-production aggregates",
    ),
    "preseason_team_talent": (
        "raw/cfbd/preseason/talent",
        "team-season roster-talent aggregates",
    ),
    "preseason_team_recruiting": (
        "raw/cfbd/preseason/recruiting_teams",
        "team-season recruiting aggregates",
    ),
    "preseason_coach_tenures": (
        "raw/cfbd/preseason/coach_tenures",
        "team-level coaching-tenure records",
    ),
    "season_team_metadata": (
        "raw/cfbd/preseason/teams",
        "season team metadata, not player rosters",
    ),
}
SNAPSHOT_SOURCES = ("roster", "games_players", "usage", "stats", "portal")
PATH_SOURCE_ALIASES = {
    "roster": {"roster", "rosters"},
    "games_players": {
        "games_players",
        "games-players",
        "game_players",
        "game-players",
        "player_games",
        "player-games",
    },
    "usage": {"usage", "player_usage", "player-usage"},
    "stats": {"stats", "player_stats", "player-stats", "stats_player"},
    "portal": {"portal", "transfer_portal"},
}
YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def _year_tokens(paths: list[Path]) -> list[int]:
    return sorted(
        {
            int(match.group())
            for path in paths
            for match in YEAR_PATTERN.finditer(path.name)
        }
    )


def _payload_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*.json")
        if not path.name.endswith(".provenance.json")
    )


def _directory_summary(data_root: Path, relative: str, meaning: str) -> dict[str, Any]:
    directory = data_root / relative
    payloads = _payload_files(directory)
    sidecars = (
        sorted(directory.rglob("*.provenance.json")) if directory.is_dir() else []
    )
    return {
        "path": relative,
        "meaning": meaning,
        "directory_present": directory.is_dir(),
        "payload_file_count": len(payloads),
        "provenance_sidecar_count": len(sidecars),
        "filename_seasons": _year_tokens(payloads),
        "two_byte_payload_count": sum(path.stat().st_size == 2 for path in payloads),
    }


def _player_snapshot_path_search(data_root: Path) -> dict[str, Any]:
    raw_root = data_root / CFBD_RAW_RELATIVE
    payloads = _payload_files(raw_root)
    matches: dict[str, list[str]] = {source: [] for source in SNAPSHOT_SOURCES}
    for path in payloads:
        parts = {part.casefold() for part in path.relative_to(raw_root).parts[:-1]}
        filename = path.name.casefold()
        for source, aliases in PATH_SOURCE_ALIASES.items():
            if parts & aliases or any(alias in filename for alias in aliases):
                matches[source].append(path.relative_to(data_root).as_posix())
    return {
        source: {
            "payload_file_count": len(paths),
            "paths": paths,
        }
        for source, paths in matches.items()
    }


def _git_tracked_raw_files() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z", "--", "data/raw/cfbd"],
            check=True,
            capture_output=True,
            text=False,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return {"available": False, "error": type(error).__name__}
    paths = [value.decode("utf-8") for value in result.stdout.split(b"\0") if value]
    return {
        "available": True,
        "payload_file_count": sum(not path.endswith(".provenance.json") for path in paths),
        "provenance_sidecar_count": sum(
            path.endswith(".provenance.json") for path in paths
        ),
        "paths": paths,
    }


def _snapshot_manifest(data_root: Path) -> dict[str, Any]:
    relative = "raw/cfbd/preseason/transfers/manifest.json"
    manifest_path = data_root / relative
    grouped: dict[str, dict[str, Any]] = {
        source: {
            "manifest_record_count": 0,
            "raw_payload_present_count": 0,
            "captured_on_or_before_cutoff_count": 0,
            "target_seasons": [],
        }
        for source in SNAPSHOT_SOURCES
    }
    if not manifest_path.is_file():
        return {"path": relative, "present": False, "sources": grouped}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_root_value = manifest.get("raw_root", ".")
    raw_root = Path(raw_root_value)
    if not raw_root.is_absolute():
        raw_root = manifest_path.parent / raw_root
    raw_root = raw_root.resolve()
    for snapshot in manifest.get("snapshots", []):
        source = str(snapshot.get("source", ""))
        if source not in grouped:
            continue
        summary = grouped[source]
        summary["manifest_record_count"] += 1
        target_season = snapshot.get("target_season", snapshot.get("season"))
        if isinstance(target_season, int):
            summary["target_seasons"].append(target_season)
        summary["captured_on_or_before_cutoff_count"] += int(
            snapshot.get("captured_on_or_before_cutoff") is True
        )
        snapshot_path = (raw_root / str(snapshot.get("path", ""))).resolve()
        try:
            snapshot_path.relative_to(raw_root)
        except ValueError:
            continue
        if snapshot_path.is_file():
            summary["raw_payload_present_count"] += 1
    for summary in grouped.values():
        summary["target_seasons"] = sorted(set(summary["target_seasons"]))
    return {"path": relative, "present": True, "sources": grouped}


def build_inventory(data_root: Path) -> dict[str, Any]:
    data_root = data_root.resolve()
    directories = {
        name: _directory_summary(data_root, relative, meaning)
        for name, (relative, meaning) in SOURCE_DIRECTORIES.items()
    }
    manifest = _snapshot_manifest(data_root)
    return {
        "study": "offensive-line-continuity-issue-174",
        "inventory_scope": "offline local artifact inventory; no API calls",
        "data_root_label": data_root.name,
        "repository_tracked_cfbd_raw_files": _git_tracked_raw_files(),
        "local_source_directories": directories,
        "player_snapshot_path_search": _player_snapshot_path_search(data_root),
        "preseason_transfer_snapshot_manifest": manifest,
        "interpretation": {
            "target_roster_membership": (
                "No on-time historical target-season roster panel is established by this inventory."
            ),
            "player_participation": (
                "Team box-score payloads do not establish player appearances, starts, or snaps."
            ),
            "pairwise_co_play": (
                "No pairwise co-play measure is computed without complete player-game or snap evidence."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="data directory to inspect; defaults to this checkout's data/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "source_inventory.json",
        help="destination for deterministic JSON inventory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = build_inventory(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "study": inventory["study"]}))


if __name__ == "__main__":
    main()
