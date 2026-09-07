"""Fetch and inspect the historical CFBD game and team-stat corpus."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx

from gippyrank.data.cfbd import raw_stat_payload_paths

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/cfbd"
RAW_GAMES = RAW / "games"
RAW_STATS = RAW / "game_stats"
PROCESSED = ROOT / "data/processed/cfbd"
API = "https://api.collegefootballdata.com"
SEASONS = range(2003, 2027)
CLASSIFICATIONS = ("fbs", "fcs")
GAME_FIELDS = (
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
)
STAT_FIELDS = (
    "game_id",
    "team_id",
    "opponent_id",
    "total_yards",
    "plays",
    "yards_per_play",
)


def request_json(
    client: httpx.Client, path: str, params: dict[str, object]
) -> httpx.Response:
    for attempt in range(5):
        response = client.get(API + path, params=params, timeout=60)
        if response.status_code < 500 and response.status_code != 429:
            response.raise_for_status()
            return response
        time.sleep(2**attempt)
    response.raise_for_status()
    return response


def fetch_raw(
    client: httpx.Client,
    path: str,
    params: dict[str, object],
    destination: Path,
    *,
    refresh: bool = False,
) -> list[dict]:
    if destination.exists() and not refresh:
        with destination.open(encoding="utf-8") as handle:
            return json.load(handle)
    response = request_json(client, path, params)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    provenance = {
        "content_sha256": hashlib.sha256(response.content).hexdigest(),
        "endpoint": path,
        "parameters": params,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source_kind": "cfbd_api",
    }
    destination.with_name(f"{destination.name}.provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return response.json()


def fetch_all(
    refresh_seasons: frozenset[int] = frozenset(),
) -> dict[tuple[int, str], list[dict]]:
    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD_API_KEY is not configured")
    schedules: dict[tuple[int, str], list[dict]] = {}
    with httpx.Client(headers={"Authorization": f"Bearer {api_key}"}) as client:
        for season in SEASONS:
            for classification in CLASSIFICATIONS:
                name = (
                    f"{season}.json"
                    if classification == "fbs"
                    else f"{season}-fcs.json"
                )
                schedules[(season, classification)] = fetch_raw(
                    client,
                    "/games",
                    {"year": season, "classification": classification},
                    RAW_GAMES / name,
                    refresh=season in refresh_seasons,
                )
                time.sleep(0.1)

        for season in SEASONS:
            for classification in CLASSIFICATIONS:
                weeks = sorted(
                    {
                        (game["week"], game["seasonType"])
                        for game in schedules[(season, classification)]
                        if game.get("week") is not None and game.get("seasonType")
                    }
                )
                for week, season_type in weeks:
                    name = f"{season}-{classification}-week{week}-{season_type}.json"
                    fetch_raw(
                        client,
                        "/games/teams",
                        {
                            "year": season,
                            "week": week,
                            "seasonType": season_type,
                            "classification": classification,
                        },
                        RAW_STATS / name,
                        refresh=season in refresh_seasons,
                    )
                    time.sleep(0.1)
    return schedules


def load_schedules() -> dict[tuple[int, str], list[dict]]:
    schedules = {}
    for season in SEASONS:
        for classification in CLASSIFICATIONS:
            name = f"{season}.json" if classification == "fbs" else f"{season}-fcs.json"
            with (RAW_GAMES / name).open(encoding="utf-8") as handle:
                schedules[(season, classification)] = json.load(handle)
    return schedules


def game_row(game: dict) -> dict[str, object]:
    return {field: game.get(field) for field in GAME_FIELDS}


def build_games(
    schedules: dict[tuple[int, str], list[dict]],
) -> tuple[list[dict], dict]:
    occurrences: defaultdict[int, list[tuple[str, dict]]] = defaultdict(list)
    for (season, classification), games in schedules.items():
        for game in games:
            occurrences[int(game["id"])].append((classification, game))
    rows = []
    conflicts = []
    overlap = 0
    for game_id, records in sorted(occurrences.items()):
        if len(records) > 1:
            overlap += 1
            if any(record != records[0][1] for _, record in records[1:]):
                conflicts.append(
                    {"id": game_id, "records": [record for _, record in records]}
                )
        rows.append(game_row(records[0][1]))
    return rows, {"overlap_count": overlap, "conflicts": conflicts}


def stat_value(stats: list[dict], category: str) -> str | None:
    for item in stats:
        if item.get("category") == category:
            return item.get("stat")
    return None


def parse_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def raw_game_response_count(
    schedules: dict[tuple[int, str], list[dict]],
) -> int:
    """Count cached schedule and team-stat API response payloads."""
    return len(schedules) + len(raw_stat_payload_paths(RAW_STATS))


def build_stats(
    schedules: dict[tuple[int, str], list[dict]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    games, _ = build_games(schedules)
    game_lookup = {int(game["id"]): game for game in games}
    rows_by_key: dict[tuple[int, int], dict[str, object]] = {}
    conflicts = []
    duplicate_count = 0
    for path in raw_stat_payload_paths(RAW_STATS):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        for game in payload:
            game_id = int(game["id"])
            teams = game.get("teams") or []
            for team in teams:
                team_id = team.get("teamId")
                if team_id is None:
                    continue
                stats = team.get("stats") or []
                total_yards = parse_int(stat_value(stats, "totalYards"))
                rushing = parse_int(stat_value(stats, "rushingAttempts"))
                completion_attempts = stat_value(stats, "completionAttempts")
                pass_attempts = None
                if completion_attempts and "-" in completion_attempts:
                    pass_attempts = parse_int(completion_attempts.rsplit("-", 1)[1])
                plays = (
                    rushing + pass_attempts
                    if rushing is not None and pass_attempts is not None
                    else None
                )
                ypp = (
                    total_yards / plays
                    if total_yards is not None and plays and plays > 0
                    else None
                )
                opponent_id = None
                if game_id in game_lookup:
                    opponent_id = next(
                        (
                            other.get("teamId")
                            for other in teams
                            if other.get("teamId") != team_id
                        ),
                        None,
                    )
                row = {
                    "game_id": game_id,
                    "team_id": team_id,
                    "opponent_id": opponent_id,
                    "total_yards": total_yards,
                    "plays": plays,
                    "yards_per_play": ypp,
                }
                key = (game_id, int(team_id))
                previous = rows_by_key.get(key)
                if previous is not None and previous != row:
                    conflicts.append(
                        {
                            "game_id": game_id,
                            "team_id": team_id,
                            "first": previous,
                            "second": row,
                        }
                    )
                elif previous is not None:
                    duplicate_count += 1
                else:
                    rows_by_key[key] = row
    return list(rows_by_key.values()), conflicts, duplicate_count


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def coverage(
    games: list[dict], schedules: dict[tuple[int, str], list[dict]], stats: list[dict]
) -> dict:
    by_id = defaultdict(list)
    for row in stats:
        by_id[row["game_id"]].append(row)
    report = []
    for season in SEASONS:
        fbs = schedules[(season, "fbs")]
        fcs = schedules[(season, "fcs")]
        fbs_ids = {int(game["id"]) for game in fbs}
        fcs_ids = {int(game["id"]) for game in fcs}
        season_games = [game for game in games if game["season"] == season]
        expected = 2 * len(season_games)
        season_stats = [
            row for game in season_games for row in by_id.get(int(game["id"]), [])
        ]
        pairings = Counter(
            tuple(
                str(game.get(field) or "unknown")
                for field in ("homeClassification", "awayClassification")
            )
            for game in season_games
        )
        report.append(
            {
                "season": season,
                "fbs_query_games": len(fbs),
                "fcs_query_games": len(fcs),
                "overlap_games": len(fbs_ids & fcs_ids),
                "unique_combined_games": len(season_games),
                "games_by_participant_classification": {
                    "-".join(str(v) for v in key): value
                    for key, value in sorted(pairings.items())
                },
                "incomplete_games": sum(
                    not game.get("completed", False) for game in season_games
                ),
                "games_missing_team_ids": sum(
                    game.get("homeId") is None or game.get("awayId") is None
                    for game in season_games
                ),
                "games_missing_scores": sum(
                    game.get("homePoints") is None or game.get("awayPoints") is None
                    for game in season_games
                ),
                "games_missing_dates": sum(
                    not game.get("startDate") for game in season_games
                ),
                "stats_games_with_any_data": len(
                    {row["game_id"] for row in season_stats}
                ),
                "team_game_rows_expected": expected,
                "team_game_rows": len(season_stats),
                "team_game_rows_with_total_yards": sum(
                    row["total_yards"] is not None for row in season_stats
                ),
                "team_game_rows_with_plays": sum(
                    row["plays"] is not None for row in season_stats
                ),
                "team_game_rows_with_usable_ypp": sum(
                    row["yards_per_play"] is not None for row in season_stats
                ),
            }
        )
        current = report[-1]
        for field, output_field in (
            ("team_game_rows_with_total_yards", "total_yards_coverage_pct"),
            ("team_game_rows_with_plays", "plays_coverage_pct"),
            ("team_game_rows_with_usable_ypp", "usable_ypp_coverage_pct"),
        ):
            current[output_field] = (
                round(100 * current[field] / expected, 2) if expected else 0.0
            )
    return {
        "seasons": report,
        "coverage_anomalies": [
            {
                "season": item["season"],
                "reason": (
                    "No team-stat rows were available for any game."
                    if item["stats_games_with_any_data"] == 0
                    else "Usable YPP coverage is below 50% of expected team-game rows."
                ),
            }
            for item in report
            if item["stats_games_with_any_data"] == 0
            or item["usable_ypp_coverage_pct"] < 50
        ],
        "stat_derivation": "plays = rushingAttempts + pass attempts parsed from completionAttempts; YPP omitted for missing/zero plays.",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-season",
        action="append",
        type=int,
        default=[],
        help="Re-acquire a season and write fresh raw-response provenance.",
    )
    args = parser.parse_args()
    fetch_all(frozenset(args.refresh_season))
    schedules = load_schedules()
    games, game_validation = build_games(schedules)
    stats, stat_conflicts, stat_duplicates = build_stats(schedules)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    write_csv(PROCESSED / "games.csv", GAME_FIELDS, games)
    write_csv(PROCESSED / "team_game_stats.csv", STAT_FIELDS, stats)
    report = coverage(games, schedules, stats)
    report["game_id_validation"] = game_validation
    report["team_stat_validation"] = {
        "duplicate_team_game_rows_removed": stat_duplicates,
        "conflicts": stat_conflicts,
    }
    report["raw_game_response_count"] = raw_game_response_count(schedules)
    (PROCESSED / "coverage_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(games)} unique games and {len(stats)} team-game stat rows")


if __name__ == "__main__":
    main()
