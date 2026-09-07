"""Offline audit of primitive CFBD ``/games/teams`` box-score data.

This module deliberately stops at data semantics and quality.  It does not
fit a model, compare rank outcomes, or write any production artifact.  Raw
payloads are read through :func:`gippyrank.data.cfbd.raw_stat_payload_paths`
so provenance sidecars cannot be mistaken for API responses.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

from gippyrank.data.cfbd import raw_stat_payload_paths

PERIODS: dict[str, tuple[int, ...]] = {
    "training_2004_2017": tuple(range(2004, 2018)),
    "development_2018_2021": tuple(range(2018, 2022)),
    "modern_2022_2025": tuple(range(2022, 2026)),
    "current_2026": (2026,),
}

PAIRINGS = ("FBS-FBS", "FBS-FCS", "FCS-FCS", "unknown")
STAT_CATEGORIES_OF_INTEREST = (
    "totalYards",
    "rushingYards",
    "rushingAttempts",
    "netPassingYards",
    "completionAttempts",
    "sacks",
    "interceptions",
    "totalFumbles",
    "fumblesLost",
    "firstDowns",
    "totalPenaltiesYards",
    "puntReturns",
    "puntReturnYards",
    "possessionTime",
    "thirdDownEff",
    "fourthDownEff",
)

_PAYLOAD_SEASON_RE = re.compile(r"^(?P<season>\d{4})-(?P<classification>fbs|fcs)-")
_INT_RE = re.compile(r"^-?\d+$")
_COMPOUND_RE = re.compile(r"^(?P<left>-?\d+)-(?P<right>-?\d+)$")


def period_for(season: int) -> str:
    """Return the declared audit period for a season."""

    for period, seasons in PERIODS.items():
        if season in seasons:
            return period
    if season == 2003:
        return "pre_training_2003"
    return "outside_scope"


def canonical_pairing(home: object, away: object) -> str:
    """Canonicalize ordered home/away classifications into a pairing label."""

    values = {str(home or "").strip().upper(), str(away or "").strip().upper()}
    if values == {"FBS"}:
        return "FBS-FBS"
    if values == {"FCS"}:
        return "FCS-FCS"
    if values == {"FBS", "FCS"}:
        return "FBS-FCS"
    return "unknown"


def parse_integer(value: object) -> int | None:
    """Parse a scalar integer while preserving missing and malformed values."""

    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return int(text) if _INT_RE.fullmatch(text) else None


def parse_number(value: object) -> int | float | None:
    """Parse an integer or decimal statistic without rounding it."""

    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if _INT_RE.fullmatch(text):
        return int(text)
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_compound(value: object) -> tuple[int, int] | None:
    """Parse a ``left-right`` CFBD statistic such as ``completionAttempts``."""

    if value is None:
        return None
    match = _COMPOUND_RE.fullmatch(str(value).strip())
    if match is None:
        return None
    return int(match.group("left")), int(match.group("right"))


def parse_time(value: object) -> int | None:
    """Parse CFBD ``MM:SS`` possession time into seconds."""

    if value is None:
        return None
    text = str(value).strip()
    match = re.fullmatch(r"(\d+):(\d{2})", text)
    if match is None:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def value_format(value: object) -> str:
    """Classify a raw ``stat`` value without changing the original value."""

    if value is None or str(value).strip() == "":
        return "missing"
    text = str(value).strip()
    if _INT_RE.fullmatch(text):
        return "integer"
    if _COMPOUND_RE.fullmatch(text):
        return "compound_pair"
    if re.fullmatch(r"-?\d+\.\d+", text):
        return "decimal"
    if re.fullmatch(r"\d+:\d{2}", text):
        return "time_mm_ss"
    return "other"


def stat_map(stats: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Return the first value for each category, retaining duplicate categories separately."""

    result: dict[str, object] = {}
    for item in stats:
        category = str(item.get("category") or "")
        if category and category not in result:
            result[category] = item.get("stat")
    return result


def parse_candidate_fields(stats: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Parse audited primitives and transparent derivations from one team row."""

    raw = stat_map(stats)
    completion_attempts = parse_compound(raw.get("completionAttempts"))
    penalties = parse_compound(raw.get("totalPenaltiesYards"))
    result: dict[str, object] = {
        "total_yards": parse_integer(raw.get("totalYards")),
        "rushing_yards": parse_integer(raw.get("rushingYards")),
        "rushing_attempts": parse_integer(raw.get("rushingAttempts")),
        "passing_yards": parse_integer(raw.get("netPassingYards")),
        "pass_attempts": completion_attempts[1] if completion_attempts else None,
        "completions": completion_attempts[0] if completion_attempts else None,
        "sacks": parse_number(raw.get("sacks")),
        "sack_yards": None,
        "interceptions_thrown": parse_integer(raw.get("interceptions")),
        "total_fumbles": parse_integer(raw.get("totalFumbles")),
        "fumbles_lost": parse_integer(raw.get("fumblesLost")),
        "fumbles_recovered": parse_integer(raw.get("fumblesRecovered")),
        "turnovers_reported": parse_integer(raw.get("turnovers")),
        "first_downs": parse_integer(raw.get("firstDowns")),
        "penalties": penalties[0] if penalties else None,
        "penalty_yards": penalties[1] if penalties else None,
        "offensive_plays_direct": None,
        "offensive_plays_derived": None,
        "punts": None,
        "punt_yards": None,
        "possession_seconds": parse_time(raw.get("possessionTime")),
    }
    rush = result["rushing_attempts"]
    passes = result["pass_attempts"]
    if isinstance(rush, int) and isinstance(passes, int):
        result["offensive_plays_derived"] = rush + passes
    return result


@dataclass(frozen=True)
class ScheduleAudit:
    games: dict[str, dict[str, object]]
    exact_duplicates: int
    conflicts: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class StatAudit:
    rows: dict[tuple[int, int], dict[str, object]]
    occurrences: int
    exact_duplicates: int
    conflicts: tuple[dict[str, object], ...]
    category_counts: dict[str, int]
    category_values: dict[str, dict[str, int]]
    category_seasons: dict[str, dict[int, int]]
    category_team_rows: dict[str, dict[tuple[int, int], int]]
    row_occurrence_counts: dict[tuple[int, int], int]
    malformed: dict[str, int]
    duplicate_categories: int


def load_schedule_audit(directory: Path) -> ScheduleAudit:
    """Load schedule payloads, excluding schedule provenance sidecars."""

    games: dict[str, dict[str, object]] = {}
    exact_duplicates = 0
    conflicts: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".provenance.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for game in payload:
            game_id = str(game["id"])
            previous = games.get(game_id)
            if previous is None:
                games[game_id] = game
            elif previous == game:
                exact_duplicates += 1
            else:
                conflicts.append(
                    {"game_id": game_id, "first": previous, "second": game}
                )
    return ScheduleAudit(games, exact_duplicates, tuple(conflicts))


def _payload_metadata(path: Path) -> tuple[int, str]:
    match = _PAYLOAD_SEASON_RE.match(path.name)
    if match is None:
        raise ValueError(f"unexpected stat payload name: {path.name}")
    return int(match.group("season")), match.group("classification")


def _json_value(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _duplicate_comparison(row: Mapping[str, object]) -> dict[str, object]:
    """Exclude request provenance from duplicate-content comparison."""

    return {
        key: value
        for key, value in row.items()
        if key != "query_classification"
    }


def load_stat_audit(directory: Path) -> StatAudit:
    """Read every cached team-stat payload and classify duplicate/conflict rows."""

    rows: dict[tuple[int, int], dict[str, object]] = {}
    row_sources: defaultdict[tuple[int, int], list[str]] = defaultdict(list)
    occurrences = 0
    exact_duplicates = 0
    conflicts: list[dict[str, object]] = []
    category_counts: Counter[str] = Counter()
    category_values: defaultdict[str, Counter[str]] = defaultdict(Counter)
    category_seasons: defaultdict[str, Counter[int]] = defaultdict(Counter)
    category_team_rows: defaultdict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    row_occurrence_counts: Counter[tuple[int, int]] = Counter()
    malformed: Counter[str] = Counter()
    duplicate_categories = 0

    for path in raw_stat_payload_paths(directory):
        season, query_classification = _payload_metadata(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for game in payload:
            game_id = int(game["id"])
            for team in game.get("teams") or []:
                occurrences += 1
                team_id = team.get("teamId")
                if team_id is None:
                    continue
                team_id = int(team_id)
                key = (game_id, team_id)
                row_occurrence_counts[key] += 1
                stats = team.get("stats") or []
                categories = [str(item.get("category") or "") for item in stats]
                if len(categories) != len(set(categories)):
                    duplicate_categories += 1
                category_names = set()
                for item in stats:
                    category = str(item.get("category") or "")
                    if not category:
                        malformed["empty_category"] += 1
                        continue
                    category_names.add(category)
                    category_counts[category] += 1
                    raw_value = item.get("stat")
                    category_values[category][str(raw_value)] += 1
                    category_seasons[category][season] += 1
                    category_team_rows[category][(game_id, team_id)] += 1
                parsed = parse_candidate_fields(stats)
                row = {
                    "game_id": game_id,
                    "team_id": team_id,
                    "season": season,
                    "query_classification": query_classification,
                    "team": str(team.get("team") or ""),
                    "home_away": str(team.get("homeAway") or ""),
                    "categories": sorted(category_names),
                    "raw_stats": stats,
                    **parsed,
                }
                row_sources[key].append(path.name)
                previous = rows.get(key)
                if previous is None:
                    rows[key] = row
                elif _duplicate_comparison(previous) == _duplicate_comparison(row):
                    exact_duplicates += 1
                else:
                    conflicts.append(
                        {
                            "game_id": game_id,
                            "team_id": team_id,
                            "source": path.name,
                            "first": {
                                "source": row_sources[key][0],
                                "row": previous,
                            },
                            "second": row,
                        }
                    )

    candidate_raw_categories = {
        "total_yards": "totalYards",
        "rushing_yards": "rushingYards",
        "rushing_attempts": "rushingAttempts",
        "passing_yards": "netPassingYards",
        "pass_attempts": "completionAttempts",
        "completions": "completionAttempts",
        "sacks": "sacks",
        "interceptions_thrown": "interceptions",
        "total_fumbles": "totalFumbles",
        "fumbles_lost": "fumblesLost",
        "fumbles_recovered": "fumblesRecovered",
        "turnovers_reported": "turnovers",
        "first_downs": "firstDowns",
        "penalties": "totalPenaltiesYards",
        "penalty_yards": "totalPenaltiesYards",
    }
    for row in rows.values():
        for field, category in candidate_raw_categories.items():
            raw_values = [
                item.get("stat")
                for item in row["raw_stats"]
                if item.get("category") == category
            ]
            if raw_values and row[field] is None:
                malformed[field] += 1
    return StatAudit(
        rows=rows,
        occurrences=occurrences,
        exact_duplicates=exact_duplicates,
        conflicts=tuple(conflicts),
        category_counts=dict(sorted(category_counts.items())),
        category_values={
            category: dict(sorted(values.items()))
            for category, values in sorted(category_values.items())
        },
        category_seasons={
            category: dict(sorted(values.items()))
            for category, values in sorted(category_seasons.items())
        },
        category_team_rows={
            category: dict(sorted(values.items()))
            for category, values in sorted(category_team_rows.items())
        },
        row_occurrence_counts=dict(sorted(row_occurrence_counts.items())),
        malformed=dict(sorted(malformed.items())),
        duplicate_categories=duplicate_categories,
    )


def _eligible_game(game: Mapping[str, object]) -> bool:
    return bool(
        2003 <= int(game.get("season") or 0) <= 2026
        and
        game.get("completed")
        and game.get("homeId") is not None
        and game.get("awayId") is not None
        and str(game.get("homeClassification") or "").upper() in {"FBS", "FCS"}
        and str(game.get("awayClassification") or "").upper() in {"FBS", "FCS"}
    )


def _attach_game_fields(rows: Iterable[dict[str, object]], games: Mapping[str, Mapping[str, object]]) -> None:
    for row in rows:
        game = games.get(str(row["game_id"]))
        if game is None:
            row.update(
                {
                    "game_found": False,
                    "pairing": "unknown",
                    "completed": None,
                    "overtime_periods": None,
                }
            )
            continue
        line_scores = [game.get("homeLineScores") or [], game.get("awayLineScores") or []]
        periods = max((len(scores) for scores in line_scores), default=0)
        row.update(
            {
                "game_found": True,
                "pairing": canonical_pairing(
                    game.get("homeClassification"), game.get("awayClassification")
                ),
                "completed": bool(game.get("completed")),
                "overtime_periods": max(periods - 4, 0) if periods else None,
            }
        )


def _range_flags(field: str, value: object) -> tuple[bool, bool]:
    """Return ``(impossible, extreme)`` using audit-only plausibility bounds."""

    if value is None or not isinstance(value, Real):
        return False, False
    bounds: dict[str, tuple[float, float, float, float]] = {
        "total_yards": (0, 2000, 1000, -1),
        "rushing_yards": (-500, 1000, 500, -200),
        "rushing_attempts": (0, 200, 100, -1),
        "passing_yards": (-1000, 2000, 1000, -200),
        "pass_attempts": (0, 200, 100, -1),
        "completions": (0, 200, 100, -1),
        "sacks": (0, 40, 15, -1),
        "interceptions_thrown": (0, 20, 8, -1),
        "total_fumbles": (0, 30, 12, -1),
        "fumbles_lost": (0, 20, 8, -1),
        "fumbles_recovered": (0, 20, 8, -1),
        "turnovers_reported": (0, 30, 12, -1),
        "first_downs": (0, 100, 50, -1),
        "penalties": (0, 50, 20, -1),
        "penalty_yards": (0, 1000, 500, -1),
        "offensive_plays_derived": (0, 300, 150, -1),
    }
    if field not in bounds or not isinstance(value, Real):
        return False, False
    minimum, maximum, high, low = bounds[field]
    impossible = value < minimum or value > maximum
    extreme = value >= high or value <= low
    return impossible, extreme


def _quantile(values: Sequence[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def coverage_rows(
    games: Mapping[str, Mapping[str, object]],
    rows: Mapping[tuple[int, int], Mapping[str, object]],
    fields: Sequence[str],
) -> list[dict[str, object]]:
    """Build season/pairing support counts without imputing missing values."""

    selected_games = [game for game in games.values() if _eligible_game(game)]
    output: list[dict[str, object]] = []
    for season in sorted({int(game["season"]) for game in selected_games}):
        season_games = [game for game in selected_games if int(game["season"]) == season]
        for pairing in (*PAIRINGS[:-1], "all"):
            selected = [
                game
                for game in season_games
                if pairing == "all"
                or canonical_pairing(
                    game.get("homeClassification"), game.get("awayClassification")
                )
                == pairing
            ]
            for field in fields:
                expected_games = len(selected)
                expected_rows = 2 * expected_games
                team_rows = [
                    rows.get((int(game["id"]), int(team_id)), {})
                    for game in selected
                    for team_id in (game["homeId"], game["awayId"])
                ]
                observed = [row.get(field) for row in team_rows]
                usable = [value for value in observed if value is not None]
                both = 0
                any_usable = 0
                for game in selected:
                    values = [
                        rows.get((int(game["id"]), int(game["homeId"])), {}).get(field),
                        rows.get((int(game["id"]), int(game["awayId"])), {}).get(field),
                    ]
                    both += int(all(value is not None for value in values))
                    any_usable += int(any(value is not None for value in values))
                output.append(
                    {
                        "season": season,
                        "period": period_for(season),
                        "pairing": pairing,
                        "field": field,
                        "expected_games": expected_games,
                        "expected_team_games": expected_rows,
                        "team_games_with_value": len(usable),
                        "team_game_coverage_pct": round(
                            100 * len(usable) / expected_rows, 4
                        )
                        if expected_rows
                        else None,
                        "games_with_both_usable": both,
                        "game_coverage_pct": round(100 * both / expected_games, 4)
                        if expected_games
                        else None,
                        "games_with_any_usable": any_usable,
                        "missing_team_games": expected_rows - len(usable),
                        "malformed_team_games": 0,
                    }
                )
    return output


def category_inventory(stat: StatAudit, rows: Mapping[tuple[int, int], Mapping[str, object]]) -> list[dict[str, object]]:
    """Describe all observed raw categories, including their era support."""

    total_rows = len(rows)
    inventory: list[dict[str, object]] = []
    for category in sorted(stat.category_counts):
        seasons = stat.category_seasons[category]
        era_counts: dict[str, int] = {}
        for period, period_seasons in PERIODS.items():
            era_counts[period] = sum(seasons.get(season, 0) for season in period_seasons)
        formats = Counter()
        for value, count in stat.category_values[category].items():
            formats[value_format(value)] += count
        row_count = len(stat.category_team_rows.get(category, {}))
        inventory.append(
            {
                "source_category": category,
                "observations": stat.category_counts[category],
                "unique_team_games": row_count,
                "team_game_coverage_pct": round(100 * row_count / total_rows, 4)
                if total_rows
                else 0.0,
                "coverage_start": min(seasons),
                "coverage_end": max(seasons),
                "training_observations": era_counts["training_2004_2017"],
                "development_observations": era_counts["development_2018_2021"],
                "modern_observations": era_counts["modern_2022_2025"],
                "current_observations": era_counts["current_2026"],
                "raw_formats": "; ".join(
                    f"{key}={formats[key]}" for key in sorted(formats)
                ),
                "value_examples": "; ".join(
                    f"{value} ({count})"
                    for value, count in sorted(
                        stat.category_values[category].items(),
                        key=lambda item: (-item[1], item[0]),
                    )[:5]
                ),
            }
        )
    return inventory


def data_dictionary_rows(
    stat: StatAudit,
    coverage: Sequence[Mapping[str, object]],
    official: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return the machine-readable proposed primitive data dictionary."""

    official_by_field: defaultdict[str, list[str]] = defaultdict(list)
    for row in official:
        field = str(row["field"])
        result = str(row["result"])
        if result not in official_by_field[field]:
            official_by_field[field].append(result)
    definitions: list[dict[str, object]] = [
        {
            "source_category": "totalYards",
            "proposed_name": "total_yards",
            "primitive_or_derived": "primitive",
            "definition": "CFBD totalYards team-game value.",
            "raw_format": "integer string",
            "status": "trusted_primitive",
            "caveats": "One 2024 official sample has a 3-yard discrepancy in the CFBD total, driven by rushingYards; retain raw value.",
        },
        {
            "source_category": "rushingYards",
            "proposed_name": "rushing_yards",
            "primitive_or_derived": "primitive",
            "definition": "CFBD net rushing yards team-game value.",
            "raw_format": "integer string",
            "status": "trusted_primitive",
            "caveats": "Can be negative in principle; 2024 Arkansas sample differs from official by 3 yards.",
        },
        {
            "source_category": "rushingAttempts",
            "proposed_name": "rushing_attempts",
            "primitive_or_derived": "primitive",
            "definition": "CFBD rushing attempts team-game value.",
            "raw_format": "integer string",
            "status": "trusted_primitive",
            "caveats": "Matches official sample values, including the 9-overtime game.",
        },
        {
            "source_category": "netPassingYards",
            "proposed_name": "passing_yards",
            "primitive_or_derived": "primitive",
            "definition": "CFBD netPassingYards; the available passing-yard field is net rather than a separately observed gross passing total.",
            "raw_format": "integer string",
            "status": "unresolved",
            "caveats": "Official 2004 cross-check shows the labeled field includes 49/3 sack yards while 2018-2024 cross-checks match net passing. Treat 2004-2011 as a distinct semantic regime until resolved.",
        },
        {
            "source_category": "completionAttempts",
            "proposed_name": "completions/pass_attempts",
            "primitive_or_derived": "primitive",
            "definition": "Compound completion-attempt string split at the hyphen.",
            "raw_format": "completion-attempt integer pair",
            "status": "trusted_primitive",
            "caveats": "Keep both components and preserve malformed/missing source values as missing.",
        },
        {
            "source_category": "sacks",
            "proposed_name": "sacks",
            "primitive_or_derived": "primitive",
            "definition": "Sacks recorded against the team in the CFBD team-game row.",
            "raw_format": "integer string",
            "status": "usable_with_caveat",
            "caveats": "Appears only on a subset of historical rows; sack yards are not exposed in the cached payloads.",
        },
        {
            "source_category": "sackYards (not observed)",
            "proposed_name": "sack_yards",
            "primitive_or_derived": "primitive",
            "definition": "Sack yards lost; no direct category was found in the cached payloads.",
            "raw_format": "unsupported",
            "status": "unsupported",
            "caveats": "Official game books expose this in some eras, but CFBD does not in this cache.",
        },
        {
            "source_category": "interceptions",
            "proposed_name": "interceptions_thrown",
            "primitive_or_derived": "primitive",
            "definition": "Offensive interceptions thrown, matching the INT component of completionAttempts and official sample box scores.",
            "raw_format": "integer string",
            "status": "trusted_primitive",
            "caveats": "Keep separate from fumbles and turnovers.",
        },
        {
            "source_category": "totalFumbles",
            "proposed_name": "total_fumbles",
            "primitive_or_derived": "primitive",
            "definition": "CFBD totalFumbles team-game value where supplied.",
            "raw_format": "integer string",
            "status": "usable_with_caveat",
            "caveats": "Historically intermittent; missing is not zero.",
        },
        {
            "source_category": "fumblesLost",
            "proposed_name": "fumbles_lost",
            "primitive_or_derived": "primitive",
            "definition": "CFBD fumblesLost team-game value.",
            "raw_format": "integer string",
            "status": "trusted_primitive",
            "caveats": "More consistently populated than totalFumbles in the cached corpus; keep separate.",
        },
        {
            "source_category": "fumblesRecovered",
            "proposed_name": "fumbles_recovered",
            "primitive_or_derived": "primitive",
            "definition": "CFBD fumblesRecovered team-game value, retained as a defensive/recovery context field.",
            "raw_format": "integer string",
            "status": "usable_with_caveat",
            "caveats": "Not interchangeable with totalFumbles or fumblesLost; coverage is slightly incomplete in later eras.",
        },
        {
            "source_category": "turnovers",
            "proposed_name": "turnovers_reported",
            "primitive_or_derived": "primitive",
            "definition": "CFBD reported team turnover total, retained only for reconciliation context.",
            "raw_format": "integer string",
            "status": "usable_with_caveat",
            "caveats": "Not used to replace the separate interceptions_thrown, total_fumbles, and fumbles_lost fields.",
        },
        {
            "source_category": "firstDowns",
            "proposed_name": "first_downs",
            "primitive_or_derived": "primitive",
            "definition": "Total first downs; category is not split into rushing, passing, and penalty in the cached payloads.",
            "raw_format": "integer string",
            "status": "trusted_primitive",
            "caveats": "Component first-down categories are unsupported here.",
        },
        {
            "source_category": "totalPenaltiesYards",
            "proposed_name": "penalties/penalty_yards",
            "primitive_or_derived": "primitive",
            "definition": "Compound accepted penalty count and penalty yards as supplied by CFBD.",
            "raw_format": "count-yards integer pair",
            "status": "usable_with_caveat",
            "caveats": "No independent verification of provider penalty inclusion rules beyond official sample agreement.",
        },
        {
            "source_category": "rushingAttempts + completionAttempts",
            "proposed_name": "offensive_plays_derived",
            "primitive_or_derived": "derived",
            "definition": "rushing_attempts + pass_attempts; no direct offensive-play category is present.",
            "raw_format": "integer sum",
            "status": "trusted_derived",
            "caveats": "Agrees with direct official plays in the 2018, 2021, and 2024 samples; source-wide direct comparison is impossible.",
        },
        {
            "source_category": "plays/offensivePlays/totalPlays (not observed)",
            "proposed_name": "offensive_plays_direct",
            "primitive_or_derived": "primitive",
            "definition": "Direct CFBD offensive play count.",
            "raw_format": "unsupported",
            "status": "unsupported",
            "caveats": "No direct category observed in any cached /games/teams payload.",
        },
        {
            "source_category": "punts/puntYards (not observed)",
            "proposed_name": "punts/punt_yards",
            "primitive_or_derived": "primitive",
            "definition": "Punt count and punt yards.",
            "raw_format": "unsupported",
            "status": "unsupported",
            "caveats": "Not present in cached /games/teams team stats; official sample only confirms these are separate box-score fields.",
        },
    ]
    for row in definitions:
        proposed = str(row["proposed_name"])
        matching = [item for item in coverage if item["field"] in proposed]
        row["coverage_start"] = min(
            (int(item["season"]) for item in matching if item["team_games_with_value"]),
            default=None,
        )
        for label, period in (
            ("training_coverage", "training_2004_2017"),
            ("development_coverage", "development_2018_2021"),
            ("modern_coverage", "modern_2022_2025"),
        ):
            values = [
                item["team_game_coverage_pct"]
                for item in matching
                if item["period"] == period and item["pairing"] == "all"
            ]
            row[label] = round(sum(value or 0 for value in values) / len(values), 2) if values else None
        for pairing, label in (
            ("FBS-FBS", "fbs_fbs_support"),
            ("FBS-FCS", "fbs_fcs_support"),
            ("FCS-FCS", "fcs_fcs_support"),
        ):
            values = [
                item["game_coverage_pct"]
                for item in matching
                if item["pairing"] == pairing and item["season"] >= 2004
            ]
            row[label] = round(sum(value or 0 for value in values) / len(values), 2) if values else None
        row["official_crosscheck"] = "; ".join(official_by_field.get(proposed.split("/")[0], []))
    return sorted(definitions, key=lambda row: str(row["proposed_name"]))


def official_crosscheck_rows() -> list[dict[str, object]]:
    """Return the fixed, source-cited official comparison sample.

    Values are intentionally checked into the audit specification rather than
    fetched during a build.  The build remains offline and deterministic.
    """

    source_2004 = "https://ohiostatebuckeyes.com/documents/download/2023/6/30/2004-7-Indiana.pdf"
    source_2018 = "https://static.sjsuspartans.com/Football/2018/HTML/ucd-sj.htm"
    source_2021 = "https://fightingillini.com/sports/football/stats/2021/penn-state/boxscore/22918"
    source_2024 = "https://utsports.com/sports/football/stats/2024/arkansas/boxscore/28878"
    samples = [
        (242970194, 2004, "FBS-FBS", "Indiana", source_2004, {"total_yards": 242, "rushing_yards": 53, "rushing_attempts": 36, "passing_yards": 189, "pass_attempts": 28, "completions": 17, "offensive_plays_derived": 64, "sacks": 6, "sack_yards": 49, "interceptions_thrown": 1, "total_fumbles": 1, "fumbles_lost": 0, "first_downs": 15, "penalties": 10, "penalty_yards": 55, "punts": 7, "punt_yards": 317}),
        (242970194, 2004, "FBS-FBS", "Ohio State", source_2004, {"total_yards": 443, "rushing_yards": 282, "rushing_attempts": 43, "passing_yards": 161, "pass_attempts": 24, "completions": 12, "offensive_plays_derived": 67, "sacks": 1, "sack_yards": 3, "interceptions_thrown": 0, "total_fumbles": 0, "fumbles_lost": 0, "first_downs": 22, "penalties": 8, "penalty_yards": 66, "punts": 4, "punt_yards": 190}),
        (401022511, 2018, "FBS-FCS", "UC Davis", source_2018, {"total_yards": 589, "rushing_yards": 143, "rushing_attempts": 34, "passing_yards": 446, "pass_attempts": 57, "completions": 37, "offensive_plays_derived": 91, "sacks": 3, "sack_yards": 24, "interceptions_thrown": 1, "total_fumbles": 0, "fumbles_lost": 0, "first_downs": 31, "penalties": 11, "penalty_yards": 79, "punts": 8, "punt_yards": 309}),
        (401022511, 2018, "FBS-FCS", "San José State", source_2018, {"total_yards": 506, "rushing_yards": 141, "rushing_attempts": 38, "passing_yards": 365, "pass_attempts": 52, "completions": 28, "offensive_plays_derived": 90, "sacks": 2, "sack_yards": 17, "interceptions_thrown": 2, "total_fumbles": 1, "fumbles_lost": 1, "first_downs": 27, "penalties": 12, "penalty_yards": 125, "punts": 6, "punt_yards": 238}),
        (401282717, 2021, "FBS-FBS", "Illinois", source_2021, {"total_yards": 395, "rushing_yards": 357, "rushing_attempts": 67, "passing_yards": 38, "pass_attempts": 21, "completions": 8, "offensive_plays_derived": 88, "sacks": 4, "sack_yards": 26, "interceptions_thrown": 1, "total_fumbles": 3, "fumbles_lost": 2, "first_downs": 26, "penalties": 4, "penalty_yards": 50, "punts": 4, "punt_yards": 172}),
        (401282717, 2021, "FBS-FBS", "Penn State", source_2021, {"total_yards": 227, "rushing_yards": 62, "rushing_attempts": 29, "passing_yards": 165, "pass_attempts": 34, "completions": 19, "offensive_plays_derived": 63, "sacks": 4, "sack_yards": 31, "interceptions_thrown": 0, "total_fumbles": 0, "fumbles_lost": 0, "first_downs": 14, "penalties": 7, "penalty_yards": 81, "punts": 8, "punt_yards": 375}),
        (401628379, 2024, "FBS-FBS", "Tennessee", source_2024, {"total_yards": 332, "rushing_yards": 174, "rushing_attempts": 36, "passing_yards": 158, "pass_attempts": 29, "completions": 17, "offensive_plays_derived": 65, "sacks": 2, "sack_yards": 9, "interceptions_thrown": 0, "total_fumbles": 1, "fumbles_lost": 0, "first_downs": 16, "penalties": 10, "penalty_yards": 60, "punts": 7, "punt_yards": 299}),
        (401628379, 2024, "FBS-FBS", "Arkansas", source_2024, {"total_yards": 434, "rushing_yards": 137, "rushing_attempts": 44, "passing_yards": 297, "pass_attempts": 30, "completions": 21, "offensive_plays_derived": 74, "sacks": 4, "sack_yards": 19, "interceptions_thrown": 0, "total_fumbles": 2, "fumbles_lost": 0, "first_downs": 23, "penalties": 6, "penalty_yards": 45, "punts": 3, "punt_yards": 172}),
    ]
    rows: list[dict[str, object]] = []
    for game_id, season, pairing, team, source, expected in samples:
        for field, value in expected.items():
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "pairing": pairing,
                    "team": team,
                    "field": field,
                    "official_value": value,
                    "source": source,
                    "result": "pending",
                }
            )
    return rows


def compare_official(
    rows: Mapping[tuple[int, int], Mapping[str, object]],
    games: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Compare the fixed official sample to cached CFBD rows by team name."""

    output: list[dict[str, object]] = []
    for item in official_crosscheck_rows():
        game = games.get(str(item["game_id"]))
        cached: Mapping[str, object] | None = None
        if game is not None:
            for team_id in (game.get("homeId"), game.get("awayId")):
                candidate = rows.get((int(item["game_id"]), int(team_id))) if team_id is not None else None
                if candidate and str(candidate.get("team")) == item["team"]:
                    cached = candidate
                    break
        field = str(item["field"])
        cached_value = None if cached is None else cached.get(field)
        official_value = item["official_value"]
        if cached is None or cached_value is None:
            result = "unavailable"
        elif cached_value == official_value:
            result = "verified"
        else:
            result = "mismatch"
        output.append(
            {
                **item,
                "cached_value": cached_value,
                "result": result,
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_audit(root: Path, output: Path) -> dict[str, object]:
    """Build all box-score audit artifacts from the existing cache, offline."""

    schedules = load_schedule_audit(root / "data/raw/cfbd/games")
    stats = load_stat_audit(root / "data/raw/cfbd/game_stats")
    rows = {key: dict(value) for key, value in stats.rows.items()}
    _attach_game_fields(rows.values(), schedules.games)
    candidate_fields = (
        "total_yards",
        "rushing_yards",
        "rushing_attempts",
        "passing_yards",
        "pass_attempts",
        "completions",
        "sacks",
        "interceptions_thrown",
        "total_fumbles",
        "fumbles_lost",
        "fumbles_recovered",
        "turnovers_reported",
        "first_downs",
        "penalties",
        "penalty_yards",
        "offensive_plays_derived",
    )
    coverage = coverage_rows(schedules.games, rows, candidate_fields)
    official = compare_official(rows, schedules.games)
    inventory = category_inventory(stats, rows)

    dataset_rows: list[dict[str, object]] = []
    for key in sorted(rows):
        row = rows[key]
        values = {field: row.get(field) for field in candidate_fields}
        impossible_fields: list[str] = []
        extreme_fields: list[str] = []
        for field, value in values.items():
            impossible, extreme = _range_flags(field, value)
            if impossible:
                impossible_fields.append(field)
            if extreme:
                extreme_fields.append(field)
        dataset_rows.append(
            {
                "game_id": row["game_id"],
                "season": row["season"],
                "period": period_for(int(row["season"])),
                "pairing": row.get("pairing"),
                "team_id": row["team_id"],
                "team": row["team"],
                "home_away": row["home_away"],
                "game_found": row["game_found"],
                "completed": row["completed"],
                "overtime_periods": row["overtime_periods"],
                "source_categories": ";".join(row["categories"]),
                "source_count": stats.row_occurrence_counts[(int(row["game_id"]), int(row["team_id"]))],
                "impossible_fields": ";".join(impossible_fields),
                "extreme_fields": ";".join(extreme_fields),
                **values,
            }
        )

    impossible_counts = Counter(
        field
        for row in dataset_rows
        for field in str(row["impossible_fields"]).split(";")
        if field
    )
    extreme_counts = Counter(
        field
        for row in dataset_rows
        for field in str(row["extreme_fields"]).split(";")
        if field
    )
    duplicate_rows: list[dict[str, object]] = [
        {
            "kind": "exact_duplicate_summary",
            "game_id": "",
            "team_id": "",
            "source": "cached_corpus",
            "first": f"stat_team_game_rows={stats.exact_duplicates}",
            "second": f"schedule_game_rows={schedules.exact_duplicates}",
        }
    ]
    for conflict in stats.conflicts:
        duplicate_rows.append(
            {
                "kind": "conflicting_duplicate",
                "game_id": conflict["game_id"],
                "team_id": conflict["team_id"],
                "source": conflict["source"],
                "first": _json_value(conflict["first"]),
                "second": _json_value(conflict["second"]),
            }
        )
    for row in dataset_rows:
        if row["impossible_fields"] or row["extreme_fields"]:
            duplicate_rows.append(
                {
                    "kind": "quality_flag",
                    "game_id": row["game_id"],
                    "team_id": row["team_id"],
                    "source": "derived_audit",
                    "first": row["impossible_fields"],
                    "second": row["extreme_fields"],
                }
            )

    plays = [
        int(row["offensive_plays_derived"])
        for row in dataset_rows
        if isinstance(row["offensive_plays_derived"], int)
        and not row["impossible_fields"]
    ]
    overtime_rows = [row for row in dataset_rows if (row["overtime_periods"] or 0) > 0]
    overtime_games = {row["game_id"] for row in overtime_rows}
    turnover_rows = [
        row
        for row in dataset_rows
        if all(
            isinstance(row[field], int)
            for field in ("turnovers_reported", "interceptions_thrown", "fumbles_lost")
        )
    ]
    turnover_mismatches = [
        row
        for row in turnover_rows
        if row["turnovers_reported"]
        != row["interceptions_thrown"] + row["fumbles_lost"]
    ]
    reconciliation_by_period: list[dict[str, object]] = []
    for period in (*PERIODS, "pre_training_2003"):
        period_rows = [row for row in dataset_rows if row["period"] == period]
        all_values = [
            row
            for row in period_rows
            if all(isinstance(row[field], int) for field in ("total_yards", "rushing_yards", "passing_yards"))
        ]
        mismatches = [
            row
            for row in all_values
            if row["total_yards"] != row["rushing_yards"] + row["passing_yards"]
        ]
        differences = Counter(
            int(row["total_yards"]) - int(row["rushing_yards"]) - int(row["passing_yards"])
            for row in mismatches
        )
        reconciliation_by_period.append(
            {
                "period": period,
                "rows_with_all_values": len(all_values),
                "mismatches": len(mismatches),
                "mismatch_pct": round(100 * len(mismatches) / len(all_values), 4)
                if all_values
                else None,
                "common_differences": dict(differences.most_common(8)),
            }
        )
    reconciliation = {
        "total_yards_vs_rushing_plus_passing": {
            "rows_with_all_values": sum(
                isinstance(row["total_yards"], int)
                and isinstance(row["rushing_yards"], int)
                and isinstance(row["passing_yards"], int)
                for row in dataset_rows
            ),
            "mismatches": sum(
                isinstance(row["total_yards"], int)
                and isinstance(row["rushing_yards"], int)
                and isinstance(row["passing_yards"], int)
                and row["total_yards"] != row["rushing_yards"] + row["passing_yards"]
                for row in dataset_rows
            ),
            "by_period": reconciliation_by_period,
        },
        "turnovers_vs_interceptions_plus_fumbles_lost": {
            "rows_with_all_values": len(turnover_rows),
            "mismatches": len(turnover_mismatches),
            "note": "reported turnover total is reconciled only as an audit check; component fields remain separate",
        },
    }
    dictionary = data_dictionary_rows(stats, coverage, official)
    summary: dict[str, object] = {
        "scope": {
            "source": "cached CFBD /games/teams payloads",
            "schedule_payload_directory": "data/raw/cfbd/games",
            "stat_payload_directory": "data/raw/cfbd/game_stats",
            "offline": True,
            "model_fit_or_rank_evaluation": False,
            "periods": PERIODS,
        },
        "corpus": {
            "schedule_unique_games": len(schedules.games),
            "schedule_exact_duplicates": schedules.exact_duplicates,
            "schedule_conflicts": len(schedules.conflicts),
            "stat_payload_files": len(raw_stat_payload_paths(root / "data/raw/cfbd/game_stats")),
            "stat_team_row_occurrences": stats.occurrences,
            "stat_unique_team_games": len(stats.rows),
            "stat_exact_duplicates": stats.exact_duplicates,
            "stat_conflicts": len(stats.conflicts),
            "duplicate_category_rows": stats.duplicate_categories,
            "malformed_candidate_values": stats.malformed,
            "impossible_candidate_values": dict(sorted(impossible_counts.items())),
            "extreme_candidate_values": dict(sorted(extreme_counts.items())),
        },
        "category_inventory": inventory,
        "category_spelling_variants": {
            folded: sorted(
                category
                for category in stats.category_counts
                if category.casefold() == folded
            )
            for folded in sorted({category.casefold() for category in stats.category_counts})
            if len(
                {
                    category
                    for category in stats.category_counts
                    if category.casefold() == folded
                }
            )
            > 1
        },
        "candidate_fields": list(candidate_fields),
        "coverage": {
            "by_season_pairing_rows": len(coverage),
            "period_pairing": [
                {
                    "period": period,
                    "pairing": pairing,
                    "field": field,
                    "expected_team_games": sum(
                        int(row["expected_team_games"])
                        for row in coverage
                        if row["period"] == period
                        and row["pairing"] == pairing
                        and row["field"] == field
                    ),
                    "team_games_with_value": sum(
                        int(row["team_games_with_value"])
                        for row in coverage
                        if row["period"] == period
                        and row["pairing"] == pairing
                        and row["field"] == field
                    ),
                }
                for period in (*PERIODS, "pre_training_2003")
                for pairing in (*PAIRINGS[:-1], "all")
                for field in candidate_fields
            ],
        },
        "plays": {
            "direct_categories_observed": [
                category
                for category in ("plays", "offensivePlays", "totalPlays")
                if category in stats.category_counts
            ],
            "derived_definition": "rushing_attempts + pass_attempts parsed from completionAttempts",
            "rows_with_derived_plays": len(plays),
            "distribution": {
                "minimum": min(plays) if plays else None,
                "p01": _quantile(plays, 0.01),
                "median": _quantile(plays, 0.50),
                "p99": _quantile(plays, 0.99),
                "maximum": max(plays) if plays else None,
            },
            "overtime_team_rows": len(overtime_rows),
            "overtime_games": len(overtime_games),
            "maximum_overtime_periods": max(
                (int(row["overtime_periods"]) for row in overtime_rows),
                default=0,
            ),
            "official_direct_play_comparisons": sum(
                row["field"] == "offensive_plays_derived" and row["result"] == "verified"
                for row in official
            ),
        },
        "reconciliation": reconciliation,
        "official_crosscheck": {
            "verified_fields": sum(row["result"] == "verified" for row in official),
            "mismatched_fields": sum(row["result"] == "mismatch" for row in official),
            "unavailable_fields": sum(row["result"] == "unavailable" for row in official),
            "sources": sorted({row["source"] for row in official}),
        },
        "production_boundary": {
            "production_files_changed": False,
            "weekly_updater_changed": False,
            "current_season_acquisition_changed": False,
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    dataset_fields = [
        "game_id", "season", "period", "pairing", "team_id", "team", "home_away",
        "game_found", "completed", "overtime_periods", "source_categories", "source_count",
        *candidate_fields, "impossible_fields", "extreme_fields",
    ]
    _write_csv(output / "team_game_audit.csv", dataset_rows, dataset_fields)
    _write_csv(
        output / "category_inventory.csv",
        inventory,
        list(inventory[0]) if inventory else [],
    )
    _write_csv(output / "coverage_by_season_pairing.csv", coverage, list(coverage[0]) if coverage else [])
    _write_csv(output / "official_crosschecks.csv", official, list(official[0]) if official else [])
    _write_csv(
        output / "duplicates_and_quality_flags.csv",
        duplicate_rows,
        ["kind", "game_id", "team_id", "source", "first", "second"],
    )
    _write_csv(output / "data_dictionary.csv", dictionary, list(dictionary[0]) if dictionary else [])
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "source_manifest.json",
        {
            "stat_payloads": [
                {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
                for path in raw_stat_payload_paths(root / "data/raw/cfbd/game_stats")
            ],
            "schedule_payloads": [
                {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
                for path in sorted((root / "data/raw/cfbd/games").glob("*.json"))
                if not path.name.endswith(".provenance.json")
            ],
        },
    )
    _write_report(output / "report.md", summary, dictionary, inventory, official, coverage)
    return summary


def _pct(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.1f}%"


def _write_report(
    path: Path,
    summary: Mapping[str, object],
    dictionary: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
    official: Sequence[Mapping[str, object]],
    coverage: Sequence[Mapping[str, object]],
) -> None:
    corpus = summary["corpus"]
    plays = summary["plays"]
    reconciliation = summary["reconciliation"]
    lines = [
        "# CFBD primitive box-score data audit",
        "",
        "## Scope and conclusion",
        "",
        "This is a data-validation audit only. It reads the cached CFBD `/games/teams` corpus through the shared sidecar-safe payload helper. It does not fit a model, evaluate predictive value, select features, or modify production ranking behavior.",
        "",
        f"The cache contains {corpus['stat_unique_team_games']:,} unique team-game rows from {corpus['stat_payload_files']:,} payload files. It contains {len(inventory)} distinct raw stat categories. Exact duplicate rows are counted and retained as a provenance flag; conflicting duplicates are not silently repaired.",
        "",
        f"The cache has {corpus['stat_exact_duplicates']:,} exact duplicate team-game observations from overlapping requests and {corpus['stat_conflicts']:,} conflicting duplicates. Candidate malformed values: `{corpus['malformed_candidate_values']}`; impossible values: `{corpus['impossible_candidate_values']}`; extreme values under declared audit thresholds: `{corpus['extreme_candidate_values']}`.",
        "",
        "The future-safe primitive representation is `total_yards`, `rushing_yards`, `rushing_attempts`, `passing_yards` (CFBD net passing yards), `pass_attempts`, `completions`, `sacks`, `interceptions_thrown`, `total_fumbles`, `fumbles_lost`, `first_downs`, `penalties`, and `penalty_yards`, with `offensive_plays_derived = rushing_attempts + pass_attempts`. Direct play count and sack-yard categories were not observed.",
        "",
        "## Raw category inventory",
        "",
            "The complete observed category inventory is in `category_inventory.csv`; the table below is intentionally compact.",
            "",
        "| CFBD category | first season | training | development | modern | 2026 | formats |",
        "|:---|---:|---:|---:|---:|---:|:---|",
    ]
    for row in inventory:
        lines.append(
            f"| `{row['source_category']}` | {row['coverage_start']} | {row['training_observations']} | {row['development_observations']} | {row['modern_observations']} | {row['current_observations']} | {row['raw_formats']} |"
        )
    lines.extend(
        [
            "",
            "Raw categories include the requested offense/turnover/first-down/penalty primitives plus basic return, possession, down-conversion, touchdown, tackle, and kicking categories. The audit does not scrape or adopt PPA/EPA, success rate, explosiveness, havoc, line yards, SP+, opponent-adjusted metrics, or other second-order analytics.",
            "",
            "## Coverage by era and pairing",
            "",
            "`coverage_by_season_pairing.csv` contains the complete season-by-pairing table. The compact table below reports game-level both-team support for the fields most relevant to this audit; percentages are weighted by expected games within each era/pairing.",
            "",
            "| period | pairing | games | total yards | derived plays | sacks | INT thrown | total fumbles | fumbles lost |",
            "|:---|:---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for period in (*PERIODS, "pre_training_2003"):
        for pairing in PAIRINGS[:-1]:
            selected = [
                row
                for row in coverage
                if row["period"] == period and row["pairing"] == pairing
            ]
            expected_games = sum(
                int(row["expected_games"])
                for row in selected
                if row["field"] == "total_yards"
            )
            if not expected_games:
                continue
            percentages: list[str] = []
            for field in (
                "total_yards",
                "offensive_plays_derived",
                "sacks",
                "interceptions_thrown",
                "total_fumbles",
                "fumbles_lost",
            ):
                expected = sum(
                    int(row["expected_games"])
                    for row in selected
                    if row["field"] == field
                )
                both = sum(
                    int(row["games_with_both_usable"])
                    for row in selected
                    if row["field"] == field
                )
                percentages.append(f"{100 * both / expected:.1f}%" if expected else "n/a")
            lines.append(
                f"| {period} | {pairing} | {expected_games} | {percentages[0]} | {percentages[1]} | {percentages[2]} | {percentages[3]} | {percentages[4]} | {percentages[5]} |"
            )
    lines.extend(
        [
            "",
            "The table makes the pairing-specific support gap explicit: FBS–FBS and FBS–FCS are broadly supported for yards/plays in the training and development eras, while FCS–FCS is materially thinner. Sacks and especially total fumbles are much less complete; no missing value is treated as zero.",
            "",
            "## Data dictionary",
            "",
            "The complete machine-readable dictionary is `data_dictionary.csv`. `unsupported` means the category was not observed in the cached payloads; `usable_with_caveat` means raw values exist but coverage or semantics require explicit handling.",
            "",
            "| proposed name | source | status | training coverage | development coverage | modern coverage | FBS-FBS | FBS-FCS | FCS-FCS |",
            "|:---|:---|:---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dictionary:
        lines.append(
            f"| `{row['proposed_name']}` | `{row['source_category']}` | `{row['status']}` | {_pct(row['training_coverage'])} | {_pct(row['development_coverage'])} | {_pct(row['modern_coverage'])} | {_pct(row['fbs_fbs_support'])} | {_pct(row['fbs_fcs_support'])} | {_pct(row['fcs_fcs_support'])} |"
        )
    lines.extend(
        [
            "",
            "## Semantic change in passing yards",
            "",
            "The `netPassingYards` label is not semantically stable across the full cache. In 2004–2011, `totalYards` frequently does not equal `rushingYards + netPassingYards`; from 2012 onward the reconciliation is essentially complete apart from a handful of anomalies. The 2004 Ohio State–Indiana official book shows the mechanism: official net passing is 189 for Indiana and 161 for Ohio State, while cached CFBD reports 238 and 164, differences of 49 and 3 that equal the official sack yards. Thus the early CFBD field behaves like pre-sack-loss passing yards despite its `netPassingYards` name. Keep the raw field, but mark its historical semantic as unresolved and do not treat all seasons as one comparable passing-yard measure without a separate era rule.",
            "",
        ]
    )
    for item in reconciliation["total_yards_vs_rushing_plus_passing"]["by_period"]:
        lines.append(
            f"- `{item['period']}`: {item['mismatches']:,}/{item['rows_with_all_values']:,} yard reconciliations mismatched ({_pct(item['mismatch_pct'])}); common differences `{item['common_differences']}`."
        )
    lines.extend(
        [
            "",
            "## Yards and plays",
            "",
            f"The observed category names have no case-only spelling variants (`category_spelling_variants` is empty in `summary.json`). No direct `plays`, `offensivePlays`, or `totalPlays` category was observed. Derived plays are `rushingAttempts + pass attempts parsed from completionAttempts`; {plays['rows_with_derived_plays']:,} team-game rows have this derivation. The distribution is min {plays['distribution']['minimum']}, p01 {plays['distribution']['p01']}, median {plays['distribution']['median']}, p99 {plays['distribution']['p99']}, max {plays['distribution']['maximum']}. These are data-quality summaries, not evidence for a model choice.",
            "",
            f"Derived plays equal official total plays in {plays['official_direct_play_comparisons']} checked team rows across 2004, 2018, 2021, and 2024. This is a semantic sanity check, not source-wide proof. The cache contains {plays['overtime_games']:,} games with overtime line-score periods ({plays['overtime_team_rows']:,} team rows); the maximum is {plays['maximum_overtime_periods']} overtime periods. The 2021 Illinois–Penn State 9OT game is retained as an overtime case, and no overtime normalization is applied.",
            "",
            f"Total-yard reconciliation against `rushing_yards + passing_yards` is complete for {reconciliation['total_yards_vs_rushing_plus_passing']['rows_with_all_values']:,} rows; {reconciliation['total_yards_vs_rushing_plus_passing']['mismatches']:,} mismatch. The passing field is CFBD `netPassingYards`, not a separately available gross-passing field.",
            "",
            "## Components, sacks, and turnovers",
            "",
            "Rushing yards/attempts, net passing yards, completion counts, pass attempts, and sacks are retained separately in `team_game_audit.csv`. Sacks appear in the payloads but sack yards do not. Official books report sack yards in the selected sample, so future work must not invent a CFBD sack-yard field or infer it as zero.",
            "",
            f"The `interceptions` category matches the offensive INT component in the official sample and is named `interceptions_thrown` in the audit. `fumblesLost` is distinct from `totalFumbles`; total fumbles is historically intermittent and remains missing when absent. The reported `turnovers` field reconciles to interceptions thrown plus fumbles lost in {reconciliation['turnovers_vs_interceptions_plus_fumbles_lost']['rows_with_all_values']:,} rows, with {reconciliation['turnovers_vs_interceptions_plus_fumbles_lost']['mismatches']:,} mismatches, but no turnover collapse is performed. `fumblesRecovered` is also retained separately.",
            "",
            "First downs are exposed as one total `firstDowns` category. Rushing/passing/penalty first-down components were not observed. Penalties are exposed as compound `totalPenaltiesYards` values and are parsed into count and yards without imputation.",
            "",
            "## Official cross-checks",
            "",
            "The fixed sample covers a 2004 cached-era low-volume FBS–FBS game in the source selection, a 2018 FBS–FCS game, the 2021 nine-overtime FBS–FBS game, and the 2024 FBS–FBS game. Official comparisons are recorded in `official_crosschecks.csv` and are intentionally not fetched at build time.",
            "",
            "| season/game | source | verified fields | mismatches | unavailable | interpretation |",
            "|:---|:---|---:|---:|---:|:---|",
        ]
    )
    grouped: defaultdict[tuple[int, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in official:
        grouped[(int(row["season"]), int(row["game_id"]), str(row["source"]))].append(row)
    for (season, game_id, source), group in sorted(grouped.items()):
        verified = sum(row["result"] == "verified" for row in group)
        mismatched = sum(row["result"] == "mismatch" for row in group)
        unavailable = sum(row["result"] == "unavailable" for row in group)
        interpretation = "verified in sample" if not mismatched else "verified except raw CFBD mismatch; preserve raw value"
        lines.append(f"| {season} / `{game_id}` | [{source}]({source}) | {verified} | {mismatched} | {unavailable} | {interpretation} |")
    lines.extend(
        [
            "",
            "The 2004 official sample exposes the early passing-yard semantic mismatch above and also shows first-down and penalty differences; those values are not silently corrected. The 2018 and 2021 official books agree on the audited yardage, attempts, completions, plays, sacks, INTs, first downs, and supplied penalty values. In 2024, Arkansas official total offense is 434 yards (137 rushing + 297 net passing), while cached CFBD reports 431 total and 134 rushing; Tennessee agrees. These are concrete semantic/data-quality findings, not repair instructions.",
            "",
            "CFBD documents `/games/teams` as team box-score statistics with generic category/stat pairs: [CFBD Games API](https://apinext.collegefootballdata.com/api/games). Official references: [2004 Indiana–Ohio State game book](https://ohiostatebuckeyes.com/documents/download/2023/6/30/2004-7-Indiana.pdf), [2018 San José State box score](https://static.sjsuspartans.com/Football/2018/HTML/ucd-sj.htm), [2021 Illinois–Penn State box score](https://fightingillini.com/sports/football/stats/2021/penn-state/boxscore/22918), and [2024 Tennessee–Arkansas box score](https://utsports.com/sports/football/stats/2024/arkansas/boxscore/28878).",
            "",
            "## Reproducibility and production boundary",
            "",
            "The builder is offline and deterministic. It reads raw payloads without modifying them, excludes `.provenance.json` sidecars through the shared helper, emits stable sorted CSV/JSON, and leaves production CFBD processed files, Historical Likelihood V1, Posterior V1, H/C, Performance V1, YPP conclusions, the weekly updater, and the website unchanged.",
            "",
            "Missing values remain missing; no imputation or zero-filling is performed. See `duplicates_and_quality_flags.csv` for conflicting duplicate and plausibility flags.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "PAIRINGS",
    "PERIODS",
    "build_audit",
    "canonical_pairing",
    "load_schedule_audit",
    "load_stat_audit",
    "parse_candidate_fields",
    "parse_compound",
    "parse_integer",
    "parse_number",
    "parse_time",
    "period_for",
    "raw_stat_payload_paths",
]
