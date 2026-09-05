"""Acquire raw, dated offseason context separately from model fitting.

Only the coach-tenure endpoint is fetched here.  Each response is stored
unchanged under ``data/raw`` and is intentionally ignored by Git, consistent
with the repository's raw-data policy.  The Context Prior build consumes the
cached responses and never makes an ad-hoc API call.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/cfbd/preseason/coach_tenures"
FEATURES = ROOT / "data/processed/preseason/team_season_features.csv"
API = "https://api.collegefootballdata.com/coaches/tenures"


def fbs_team_names() -> list[str]:
    with FEATURES.open(newline="", encoding="utf-8") as handle:
        return sorted(
            {
                row["team_name"]
                for row in csv.DictReader(handle)
                if row["subdivision"] == "fbs"
            }
        )


def destination(team: str) -> Path:
    # The filename is an opaque transport key; the source payload retains the
    # canonical team name and is the authority for interpretation.
    return RAW / (team.lower().replace("/", "_").replace(" ", "_") + ".json")


def fetch_team(team: str, api_key: str) -> tuple[str, str]:
    output = destination(team)
    if output.exists():
        return team, "cached"
    with httpx.Client(headers={"Authorization": f"Bearer {api_key}"}) as client:
        for attempt in range(5):
            response = client.get(API, params={"team": team}, timeout=60)
            if response.status_code != 429:
                response.raise_for_status()
                break
            # This endpoint is one-team-at-a-time.  A deliberately serialized
            # refresh is kinder to CFBD and avoids treating transient throttles
            # as missing coaching history.
            time.sleep(float(response.headers.get("Retry-After", "2")) * (attempt + 1))
        else:
            response.raise_for_status()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.content)
    return team, "fetched"


def main() -> None:
    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD_API_KEY is not configured")
    teams = fbs_team_names()
    outcomes: dict[str, str] = {}
    failures: dict[str, str] = {}
    for team in teams:
        try:
            name, status = fetch_team(team, api_key)
            outcomes[name] = status
        except httpx.HTTPError as error:
            failures[team] = str(error)
        # CFBD's tenure route is intentionally serialized.  Raw source
        # acquisition is an occasional research step, not an inference path.
        if outcomes.get(team) == "fetched":
            time.sleep(1.1)
    manifest = {
        "source": "https://api.collegefootballdata.com/coaches/tenures?team=<team>",
        "team_names_requested": teams,
        "outcomes": outcomes,
        "failures": failures,
        "note": "Raw payloads are source evidence; fetch time is intentionally not used as a historical as-of timestamp.",
    }
    (RAW / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"coach tenure acquisition failed for {len(failures)} teams")
    print(json.dumps({"teams": len(teams), "outcomes": outcomes}, indent=2))


if __name__ == "__main__":
    main()
