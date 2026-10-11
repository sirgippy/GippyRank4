"""Fetch and archive monthly ESPN schedule sources for issue 198 date windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from gippyrank.data.massey_prebowl import (
    ESPN_SCOREBOARD_URL,
    SEASONS,
    schedule_window_from_espn,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_SCHEDULES = ROOT / "data/raw/massey_prebowl/schedules"
WINDOWS_OUTPUT = ROOT / "data/research/massey_prebowl_issue_198/schedule_windows.csv"
WINDOW_FIELDS = (
    "season",
    "last_fbs_conference_championship_date",
    "first_fbs_postseason_game_date",
    "window_first_full_date",
    "window_last_full_date",
    "eligible_full_days",
    "timezone",
    "source_url",
    "raw_path",
    "raw_sha256",
    "retrieved_at",
)


def fetch_schedule(
    season: int, raw_root: Path
) -> tuple[dict[str, object], dict[str, object]]:
    params = {"dates": f"{season}12", "limit": 1000, "groups": 80}
    response = httpx.get(ESPN_SCOREBOARD_URL, params=params, timeout=45)
    response.raise_for_status()
    raw_bytes = response.content
    raw_root.mkdir(parents=True, exist_ok=True)
    path = raw_root / f"espn_{season}12.json"
    path.write_bytes(raw_bytes)

    source_url = str(response.request.url)
    retrieved_at = datetime.now(UTC).isoformat()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    sidecar = {
        "source": "ESPN College Football scoreboard",
        "source_url": source_url,
        "season": season,
        "retrieved_at": retrieved_at,
        "raw_path": f"data/raw/massey_prebowl/schedules/{path.name}",
        "sha256": digest,
        "bytes": len(raw_bytes),
        "endpoint_parameters": params,
    }
    path.with_suffix(".json.provenance.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return response.json(), sidecar


def build_windows(raw_root: Path, output_path: Path, refresh: bool) -> None:
    rows = []
    for season in SEASONS:
        if refresh:
            payload, sidecar = fetch_schedule(season, raw_root)
        else:
            path = raw_root / f"espn_{season}12.json"
            sidecar_path = path.with_suffix(".json.provenance.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        window = schedule_window_from_espn(payload, season)
        rows.append(
            {
                "season": season,
                "last_fbs_conference_championship_date": window.last_fbs_conference_championship.isoformat(),
                "first_fbs_postseason_game_date": window.first_fbs_postseason_game.isoformat(),
                "window_first_full_date": window.first_eligible_date.isoformat(),
                "window_last_full_date": window.last_eligible_date.isoformat(),
                "eligible_full_days": max(
                    0,
                    (window.last_eligible_date - window.first_eligible_date).days + 1,
                ),
                "timezone": "America/New_York",
                "source_url": sidecar["source_url"],
                "raw_path": sidecar["raw_path"],
                "raw_sha256": sidecar["sha256"],
                "retrieved_at": sidecar["retrieved_at"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=WINDOW_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output_path} ({len(rows)} seasons)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_SCHEDULES)
    parser.add_argument("--output", type=Path, default=WINDOWS_OUTPUT)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch all 23 December scoreboards and preserve each response as raw JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_windows(args.raw_root, args.output, args.refresh)
