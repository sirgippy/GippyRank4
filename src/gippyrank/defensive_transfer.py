"""Deterministic defensive-transfer feature construction and audit helpers.

This module is research-only.  It turns immutable CFBD roster and
``games/players`` responses into a conservative prior defensive-experience
proxy and a separate position-normalized production composite, then joins
those values to incoming portal records.  It intentionally does not fit or
score a ranking model.

CFBD does not expose defensive snaps or defensive snap share in the selected
endpoints.  The frozen experience candidate is therefore the rate of games
with a recorded defensive box-score row, not snap share or observed
participation.  The production candidate is an equal-weight mean of
log1p-transformed, season-by-position-group z-scores over the components
declared in ``IMPACT_COMPONENTS``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from math import isfinite, log1p, sqrt
from statistics import mean, median, pstdev
from typing import Any

from gippyrank.transfer_oracle import (
    TransferRecord,
    available_by_cutoff,
    normalize_player_name,
    normalize_team_name,
)

Row = dict[str, Any]

# This mapping is deliberately explicit.  Unknown and hybrid labels are not
# assigned to a group by string similarity or by an offensive default.
DEFENSIVE_POSITION_GROUPS: dict[str, str] = {
    "DL": "dl_edge",
    "EDGE": "dl_edge",
    "DE": "dl_edge",
    "DT": "dl_edge",
    "NT": "dl_edge",
    "OLB": "lb",
    "ILB": "lb",
    "MLB": "lb",
    "LB": "lb",
    "CB": "db",
    "DB": "db",
    "S": "db",
    "FS": "db",
    "SS": "db",
    "NB": "db",
}
DEFENSIVE_POSITION_GROUP_NAMES = ("dl_edge", "lb", "db")
OFFENSIVE_POSITION_LABELS = frozenset(
    {"QB", "RB", "FB", "WR", "TE", "OL", "OT", "OG", "OC", "IOL"}
)
SPECIAL_TEAMS_POSITION_LABELS = frozenset({"K", "P", "LS"})

# These are the labels observed in CFBD games/players responses.  ``INT`` and
# ``REC`` live in separate categories from ``defensive`` and are included only
# when their category semantics identify the relevant defensive event.
STAT_TYPE_ALIASES: dict[tuple[str, str], str] = {
    ("defensive", "TOT"): "tackles",
    ("defensive", "SOLO"): "solo_tackles",
    ("defensive", "SACKS"): "sacks",
    ("defensive", "TFL"): "tackles_for_loss",
    ("defensive", "PD"): "passes_defended",
    ("defensive", "QB HUR"): "qb_hurries",
    ("interceptions", "INT"): "interceptions",
    ("fumbles", "REC"): "fumble_recoveries",
}

# Frozen before any downstream model scoring.  Components are intentionally
# simple and football-semantic; no component weights are learned from outcomes.
IMPACT_COMPONENTS: dict[str, tuple[str, ...]] = {
    "dl_edge": ("tackles", "tackles_for_loss", "sacks", "qb_hurries"),
    "lb": ("tackles", "tackles_for_loss", "sacks", "passes_defended"),
    "db": ("tackles", "passes_defended", "interceptions"),
}
ALL_STAT_FIELDS = (
    "tackles",
    "solo_tackles",
    "tackles_for_loss",
    "sacks",
    "qb_hurries",
    "passes_defended",
    "interceptions",
    "fumble_recoveries",
)


def defensive_position_group(position: str | None) -> str | None:
    """Map a source position to the frozen defensive taxonomy."""
    if not position:
        return None
    return DEFENSIVE_POSITION_GROUPS.get(position.strip().upper())


def position_semantics(position: str | None) -> str:
    """Classify a source label without silently assigning unknown positions."""
    if not position or not position.strip():
        return "unknown"
    label = position.strip().upper()
    if label in DEFENSIVE_POSITION_GROUPS:
        return "defensive"
    if label in OFFENSIVE_POSITION_LABELS or label in SPECIAL_TEAMS_POSITION_LABELS:
        return "non_defensive"
    return "unknown"


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _canonical_label(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().upper())


@dataclass(frozen=True)
class RosterPlayer:
    """One player identity and source-team position from a roster response."""

    season: int
    player_id: str
    player_name: str
    team: str
    position: str | None

    @property
    def normalized_team(self) -> str:
        return normalize_team_name(self.team)

    @property
    def normalized_player_name(self) -> str:
        return normalize_player_name(self.player_name)

    @property
    def position_group(self) -> str | None:
        return defensive_position_group(self.position)


@dataclass(frozen=True)
class DefensiveGamePlayer:
    """One player-game row extracted from CFBD defensive categories."""

    season: int
    game_id: str
    team: str
    player_id: str
    player_name: str
    stats: Mapping[str, float]

    @property
    def normalized_team(self) -> str:
        return normalize_team_name(self.team)

    @property
    def normalized_player_name(self) -> str:
        return normalize_player_name(self.player_name)


@dataclass(frozen=True)
class DefensivePlayerSeason:
    """Aggregated prior-season defensive record for one player."""

    season: int
    team: str
    player_id: str
    player_name: str
    position: str | None
    position_group: str | None
    team_games: int | None
    recorded_defensive_box_score_games: int
    stats: Mapping[str, float | None]
    defensive_box_score_game_rate: float | None
    defensive_impact: float | None = None

    @property
    def normalized_team(self) -> str:
        return normalize_team_name(self.team)

    @property
    def normalized_player_name(self) -> str:
        return normalize_player_name(self.player_name)


def parse_roster_payload(
    payload: Iterable[Mapping[str, Any]], *, season: int
) -> list[RosterPlayer]:
    """Parse a raw CFBD roster response without rewriting source records."""
    result: list[RosterPlayer] = []
    for item in payload:
        player_id = str(item.get("id") or "").strip()
        team = str(item.get("team") or "").strip()
        first = str(item.get("firstName") or "").strip()
        last = str(item.get("lastName") or "").strip()
        name = " ".join(part for part in (first, last) if part)
        if not player_id or not team or not name:
            continue
        result.append(
            RosterPlayer(
                season=season,
                player_id=player_id,
                player_name=name,
                team=team,
                position=(
                    str(item.get("position")).strip()
                    if item.get("position") not in (None, "")
                    else None
                ),
            )
        )
    return result


def parse_games_players_payload(
    payload: Iterable[Mapping[str, Any]], *, season: int
) -> list[DefensiveGamePlayer]:
    """Extract defensive player-game statistics from a raw CFBD response.

    CFBD returns a nested game → team → category → stat-type → athlete shape.
    The parser keeps only numeric defensive events and is tolerant of absent
    categories or nonnumeric source values.  A row in an event category is
    itself the recorded box-score observation; zero-valued event statistics are
    retained as explicit zero-stat observations.
    """
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for game in payload:
        game_id = str(game.get("id") or "").strip()
        if not game_id:
            continue
        for team in game.get("teams", []) or []:
            team_name = str(team.get("team") or "").strip()
            if not team_name:
                continue
            for category in team.get("categories", []) or []:
                category_name = _canonical_label(category.get("name"))
                category_key = category_name.casefold()
                for stat_type in category.get("types", []) or []:
                    label = _canonical_label(stat_type.get("name"))
                    canonical = STAT_TYPE_ALIASES.get((category_key, label))
                    if canonical is None:
                        continue
                    for athlete in stat_type.get("athletes", []) or []:
                        player_id = str(athlete.get("id") or "").strip()
                        player_name = str(athlete.get("name") or "").strip()
                        if not player_id or not player_name:
                            continue
                        value = _number(athlete.get("stat"))
                        if value is None:
                            continue
                        key = (game_id, normalize_team_name(team_name), player_id)
                        row = grouped.setdefault(
                            key,
                            {
                                "game_id": game_id,
                                "team": team_name,
                                "player_id": player_id,
                                "player_name": player_name,
                                "stats": {},
                            },
                        )
                        stats = row["stats"]
                        if canonical in stats and stats[canonical] != value:
                            raise ValueError(
                                "conflicting duplicate defensive statistic for "
                                f"{season}/{game_id}/{player_id}/{canonical}"
                            )
                        stats[canonical] = value
    return [
        DefensiveGamePlayer(
            season=season,
            game_id=row["game_id"],
            team=row["team"],
            player_id=row["player_id"],
            player_name=row["player_name"],
            stats=dict(sorted(row["stats"].items())),
        )
        for row in sorted(
            grouped.values(),
            key=lambda value: (
                value["team"].casefold(),
                value["player_name"].casefold(),
                value["game_id"],
                value["player_id"],
            ),
        )
    ]


def team_game_keys(
    payload: Iterable[Mapping[str, Any]], *, season: int
) -> set[tuple[int, str, str]]:
    """Return every team-game key in a raw games/players response."""
    result: set[tuple[int, str, str]] = set()
    for game in payload:
        game_id = str(game.get("id") or "").strip()
        if not game_id:
            continue
        for team in game.get("teams", []) or []:
            name = str(team.get("team") or "").strip()
            if name:
                result.add((season, normalize_team_name(name), game_id))
    return result


def _roster_indexes(
    roster: Sequence[RosterPlayer],
) -> tuple[
    dict[tuple[int, str, str], list[RosterPlayer]],
    dict[tuple[int, str, str], list[RosterPlayer]],
]:
    by_id: defaultdict[tuple[int, str, str], list[RosterPlayer]] = defaultdict(list)
    by_name: defaultdict[tuple[int, str, str], list[RosterPlayer]] = defaultdict(list)
    for item in roster:
        by_id[(item.season, item.normalized_team, item.player_id)].append(item)
        by_name[
            (item.season, item.normalized_team, item.normalized_player_name)
        ].append(item)
    return dict(by_id), dict(by_name)


def aggregate_player_seasons(
    game_players: Iterable[DefensiveGamePlayer],
    roster: Iterable[RosterPlayer],
    team_game_keys_by_season: Iterable[tuple[int, str, str]],
) -> list[DefensivePlayerSeason]:
    """Aggregate player-game rows and attach roster positions.

    Rostered defensive players with no recorded defensive box-score row are
    included as verified zero-stat rows when the source team-game coverage is
    complete.  This keeps the zero-production population in the frozen impact
    normalization rather than treating a standardized impact of zero as the
    raw zero-production value.
    """
    unique_game_rows: dict[tuple[int, str, str, str], DefensiveGamePlayer] = {}
    for item in game_players:
        key = (item.season, item.game_id, item.normalized_team, item.player_id)
        previous = unique_game_rows.get(key)
        if previous is not None:
            if dict(previous.stats) != dict(item.stats):
                raise ValueError(
                    "conflicting duplicate player-game defensive rows for "
                    f"{item.season}/{item.game_id}/{item.player_id}"
                )
            continue
        unique_game_rows[key] = item
    game_rows = list(unique_game_rows.values())
    roster_rows = list(roster)
    by_id, by_name = _roster_indexes(roster_rows)
    team_games: defaultdict[tuple[int, str], set[str]] = defaultdict(set)
    for season, team, game_id in team_game_keys_by_season:
        team_games[(season, team)].add(game_id)
    grouped: dict[tuple[int, str, str], dict[str, Any]] = {}
    for item in game_rows:
        key = (item.season, item.normalized_team, item.player_id)
        row = grouped.setdefault(
            key,
            {
                "season": item.season,
                "team": item.team,
                "player_id": item.player_id,
                "player_name": item.player_name,
                "games": set(),
                "stats": defaultdict(float, {field: 0.0 for field in ALL_STAT_FIELDS}),
            },
        )
        row["games"].add(item.game_id)
        for field, value in item.stats.items():
            row["stats"][field] += value
    # Add verified zero-stat defensive roster rows so raw zero production is in
    # each season × position-group normalization population. Do not fabricate
    # a zero when the source has no team-game coverage to verify it.
    for roster_match in roster_rows:
        if roster_match.position_group is None:
            continue
        key = (
            roster_match.season,
            roster_match.normalized_team,
            roster_match.player_id,
        )
        if key in grouped:
            continue
        team_game_count = len(
            team_games.get((roster_match.season, roster_match.normalized_team), set())
        )
        if not team_game_count:
            continue
        grouped[key] = {
            "season": roster_match.season,
            "team": roster_match.team,
            "player_id": roster_match.player_id,
            "player_name": roster_match.player_name,
            "games": set(),
            "stats": defaultdict(float, {field: 0.0 for field in ALL_STAT_FIELDS}),
        }

    result: list[DefensivePlayerSeason] = []
    for row in grouped.values():
        id_matches = by_id.get(
            (row["season"], normalize_team_name(row["team"]), row["player_id"]), []
        )
        matches = id_matches or by_name.get(
            (
                row["season"],
                normalize_team_name(row["team"]),
                normalize_player_name(row["player_name"]),
            ),
            [],
        )
        roster_match = matches[0] if len(matches) == 1 else None
        position = roster_match.position if roster_match else None
        team_game_count = len(team_games.get((row["season"], row["team"]), set()))
        if not team_game_count:
            team_game_count = len(
                team_games.get((row["season"], normalize_team_name(row["team"])), set())
            )
        team_games_value = team_game_count or None
        recorded_games = len(row["games"])
        result.append(
            DefensivePlayerSeason(
                season=row["season"],
                team=row["team"],
                player_id=row["player_id"],
                player_name=row["player_name"],
                position=position,
                position_group=defensive_position_group(position),
                team_games=team_games_value,
                recorded_defensive_box_score_games=recorded_games,
                stats={field: float(value) for field, value in row["stats"].items()},
                defensive_box_score_game_rate=(
                    recorded_games / team_game_count if team_game_count else None
                ),
            )
        )
    return sorted(
        result,
        key=lambda item: (
            item.season,
            item.normalized_team,
            item.normalized_player_name,
            item.player_id,
        ),
    )


def add_defensive_impact(
    players: Iterable[DefensivePlayerSeason],
) -> list[DefensivePlayerSeason]:
    """Attach the frozen log1p, season×group normalized impact composite."""
    rows = list(players)
    transformed: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        group = row.position_group
        if group not in IMPACT_COMPONENTS:
            continue
        for field in IMPACT_COMPONENTS[group]:
            value = row.stats.get(field)
            if value is not None and value >= 0:
                transformed[(row.season, group, field)].append(log1p(float(value)))
    parameters: dict[tuple[int, str, str], tuple[float, float]] = {
        key: (mean(values), pstdev(values) if len(values) > 1 else 0.0)
        for key, values in transformed.items()
    }
    result: list[DefensivePlayerSeason] = []
    for row in rows:
        group = row.position_group
        components = IMPACT_COMPONENTS.get(group or "", ())
        scores: list[float] = []
        for field in components:
            value = row.stats.get(field)
            if value is None or value < 0:
                continue
            center, spread = parameters[(row.season, group, field)]
            scores.append((log1p(value) - center) / spread if spread > 0 else 0.0)
        result.append(replace(row, defensive_impact=mean(scores) if scores else None))
    return result


@dataclass(frozen=True)
class TeamResolution:
    season: int
    raw_name: str | None
    team_id: str | None
    canonical_name: str | None
    status: str


class TeamResolver:
    """Resolve canonical season-specific teams with exact names and aliases."""

    def __init__(
        self,
        team_rows: Iterable[Mapping[str, Any]],
        aliases: Mapping[str | tuple[int, str], str] | None = None,
    ) -> None:
        self._index: defaultdict[tuple[int, str], list[tuple[str, str]]] = defaultdict(
            list
        )
        for row in team_rows:
            if str(row.get("subdivision", "")).casefold() != "fbs":
                continue
            name = str(row.get("team_name") or "").strip()
            if name:
                self._index[(int(row["season"]), normalize_team_name(name))].append(
                    (str(row["team_id"]), name)
                )
        self.aliases: dict[str | tuple[int, str], str] = {}
        for key, value in (aliases or {}).items():
            normalized_value = normalize_team_name(value)
            if isinstance(key, tuple):
                self.aliases[(int(key[0]), normalize_team_name(key[1]))] = (
                    normalized_value
                )
            else:
                self.aliases[normalize_team_name(key)] = normalized_value

    def resolve(self, season: int, raw_name: str | None) -> TeamResolution:
        normalized = normalize_team_name(raw_name)
        if not normalized:
            return TeamResolution(season, raw_name, None, None, "missing_name")
        candidates = self._index.get((season, normalized), [])
        status = "exact_normalized_name"
        if not candidates:
            alias = self.aliases.get((season, normalized), self.aliases.get(normalized))
            if alias is not None:
                candidates = self._index.get((season, alias), [])
                status = "explicit_alias" if candidates else "alias_target_not_in_fbs"
        if len(candidates) > 1:
            return TeamResolution(season, raw_name, None, None, "ambiguous_team")
        if not candidates:
            return TeamResolution(
                season,
                raw_name,
                None,
                None,
                status if status != "exact_normalized_name" else "not_in_fbs",
            )
        team_id, canonical = candidates[0]
        return TeamResolution(season, raw_name, team_id, canonical, status)


def _player_index(
    roster: Sequence[RosterPlayer],
) -> dict[tuple[int, str, str], list[RosterPlayer]]:
    result: defaultdict[tuple[int, str, str], list[RosterPlayer]] = defaultdict(list)
    for item in roster:
        result[(item.season, item.normalized_team, item.normalized_player_name)].append(
            item
        )
    return dict(result)


def _player_id_index(
    roster: Sequence[RosterPlayer],
) -> dict[tuple[int, str, str], list[RosterPlayer]]:
    result: defaultdict[tuple[int, str, str], list[RosterPlayer]] = defaultdict(list)
    for item in roster:
        result[(item.season, item.normalized_team, item.player_id)].append(item)
    return dict(result)


def _defensive_player_index(
    players: Sequence[DefensivePlayerSeason],
) -> dict[tuple[int, str, str], list[DefensivePlayerSeason]]:
    result: defaultdict[tuple[int, str, str], list[DefensivePlayerSeason]] = (
        defaultdict(list)
    )
    for item in players:
        result[(item.season, item.normalized_team, item.normalized_player_name)].append(
            item
        )
    return dict(result)


def _defensive_player_id_index(
    players: Sequence[DefensivePlayerSeason],
) -> dict[tuple[int, str, str], list[DefensivePlayerSeason]]:
    result: defaultdict[tuple[int, str, str], list[DefensivePlayerSeason]] = (
        defaultdict(list)
    )
    for item in players:
        result[(item.season, item.normalized_team, item.player_id)].append(item)
    return dict(result)


def audit_transfer_records(
    records: Iterable[TransferRecord],
    roster: Iterable[RosterPlayer],
    players: Iterable[DefensivePlayerSeason],
    team_rows: Iterable[Mapping[str, Any]],
    *,
    portal_seasons: set[int],
    defensive_seasons: set[int],
    cutoff: date,
    aliases: Mapping[str | tuple[int, str], str] | None = None,
    team_coverage: set[tuple[int, str]] | None = None,
    roster_teams: set[tuple[int, str]] | None = None,
) -> dict[str, Any]:
    """Join incoming transfers and return player/team coverage artifacts."""
    record_list = list(records)
    roster_list = list(roster)
    player_list = list(players)
    team_input = [dict(row) for row in team_rows]
    resolver = TeamResolver(team_input, aliases)
    roster_idx = _player_index(roster_list)
    roster_id_idx = _player_id_index(roster_list)
    player_idx = _defensive_player_index(player_list)
    player_id_idx = _defensive_player_id_index(player_list)
    zero_impact_by_group = {
        (player.season, player.position_group): player.defensive_impact
        for player in player_list
        if player.recorded_defensive_box_score_games == 0
        and player.position_group in IMPACT_COMPONENTS
        and player.defensive_impact is not None
    }
    team_list = [
        row for row in team_input if str(row.get("subdivision", "")).casefold() == "fbs"
    ]
    audit_rows: list[Row] = []
    for portal_index, record in enumerate(record_list):
        destination = resolver.resolve(record.season, record.destination)
        cutoff_ok = available_by_cutoff(record, cutoff)
        in_scope = (
            record.season in portal_seasons
            and cutoff_ok
            and destination.team_id is not None
        )
        semantics = position_semantics(record.position)
        row: Row = {
            "portal_index": portal_index,
            "season": record.season,
            "player_name": record.player_name,
            "normalized_player_name": normalize_player_name(record.player_name),
            "origin": record.origin,
            "destination": record.destination,
            "destination_team_id": destination.team_id,
            "destination_team_name": destination.canonical_name,
            "destination_match_status": destination.status,
            "position": record.position,
            "portal_position_group": defensive_position_group(record.position),
            "position_semantics": semantics,
            "transfer_date": record.transfer_date.isoformat()
            if record.transfer_date
            else None,
            "cutoff_status": "on_or_before_cutoff"
            if cutoff_ok
            else "after_or_missing_date",
            "in_model_relevant_population": in_scope,
            "defensive_candidate": bool(in_scope and semantics == "defensive"),
            "identity_join_method": "none",
            "identity_status": "not_in_scope",
            "experience_status": "not_in_scope",
            "impact_status": "not_in_scope",
            "prior_position": None,
            "prior_position_group": None,
            "prior_player_id": None,
            "prior_team_games": None,
            "prior_recorded_defensive_box_score_games": None,
            "prior_defensive_box_score_game_rate": None,
            "prior_defensive_impact": None,
            "prior_stats": {},
        }
        if not in_scope:
            audit_rows.append(row)
            continue
        if semantics == "non_defensive":
            row.update(
                {
                    "identity_status": "non_defensive_not_applicable",
                    "experience_status": "non_defensive_not_applicable",
                    "impact_status": "non_defensive_not_applicable",
                }
            )
            audit_rows.append(row)
            continue
        if semantics == "unknown":
            row.update(
                {
                    "identity_status": "unknown_position",
                    "experience_status": "unknown_position",
                    "impact_status": "unknown_position",
                }
            )
            audit_rows.append(row)
            continue
        prior_season = record.season - 1
        if prior_season not in defensive_seasons:
            row.update(
                {
                    "identity_status": "source_data_unavailable",
                    "experience_status": "source_data_unavailable",
                    "impact_status": "source_data_unavailable",
                }
            )
            audit_rows.append(row)
            continue
        source_team = normalize_team_name(record.origin)
        identity_method = "normalized_name_source_team"
        if record.player_id:
            stable_matches = roster_id_idx.get(
                (prior_season, source_team, record.player_id), []
            )
            if stable_matches:
                matches = stable_matches
                identity_method = "stable_player_id_source_team"
            else:
                matches = roster_idx.get(
                    (
                        prior_season,
                        source_team,
                        normalize_player_name(record.player_name),
                    ),
                    [],
                )
        else:
            matches = roster_idx.get(
                (prior_season, source_team, normalize_player_name(record.player_name)),
                [],
            )
        if len(matches) > 1:
            row.update(
                {
                    "identity_status": "ambiguous",
                    "experience_status": "ambiguous",
                    "impact_status": "ambiguous",
                }
            )
            audit_rows.append(row)
            continue
        if not matches:
            source_team_is_covered = (
                (prior_season, source_team) in roster_teams
                if roster_teams is not None
                else True
            )
            failure_status = (
                "identity_resolution_failure"
                if source_team_is_covered
                else "source_data_unavailable"
            )
            row.update(
                {
                    "identity_status": failure_status,
                    "experience_status": failure_status,
                    "impact_status": failure_status,
                }
            )
            audit_rows.append(row)
            continue
        roster_match = matches[0]
        prior_group = roster_match.position_group
        row.update(
            {
                "identity_join_method": identity_method,
                "prior_position": roster_match.position,
                "prior_position_group": prior_group,
                "prior_player_id": roster_match.player_id,
            }
        )
        if identity_method == "stable_player_id_source_team" and record.player_id:
            stats_matches = player_id_idx.get(
                (prior_season, source_team, record.player_id), []
            )
        else:
            stats_matches = player_idx.get(
                (prior_season, source_team, normalize_player_name(record.player_name)),
                [],
            )
        if prior_group != row["portal_position_group"]:
            if len(stats_matches) == 1:
                player = stats_matches[0]
                row.update(
                    {
                        "prior_team_games": player.team_games,
                        "prior_recorded_defensive_box_score_games": player.recorded_defensive_box_score_games,
                        "prior_defensive_box_score_game_rate": player.defensive_box_score_game_rate,
                        "prior_defensive_impact": player.defensive_impact,
                        "prior_stats": dict(player.stats),
                    }
                )
            row.update(
                {
                    "identity_status": "position_mismatch",
                    "experience_status": "position_mismatch",
                    "impact_status": "position_mismatch",
                }
            )
            audit_rows.append(row)
            continue
        if len(stats_matches) > 1:
            row.update(
                {
                    "identity_status": "ambiguous",
                    "experience_status": "ambiguous",
                    "impact_status": "ambiguous",
                }
            )
            audit_rows.append(row)
            continue
        if not stats_matches:
            has_team_games = (
                (prior_season, source_team) in team_coverage
                if team_coverage is not None
                else any(
                    item.season == prior_season and item.normalized_team == source_team
                    for item in player_list
                )
            )
            if has_team_games:
                zero_impact = zero_impact_by_group.get((prior_season, prior_group))
                zero_status = "zero_recorded_defensive_box_score_games"
                row.update(
                    {
                        "identity_status": zero_status,
                        "experience_status": zero_status,
                        "impact_status": zero_status
                        if zero_impact is not None
                        else "source_data_unavailable",
                        "prior_team_games": sum(
                            1
                            for item in team_coverage or set()
                            if item[0] == prior_season and item[1] == source_team
                        )
                        or None,
                        "prior_recorded_defensive_box_score_games": 0,
                        "prior_defensive_box_score_game_rate": 0.0,
                        "prior_defensive_impact": zero_impact,
                    }
                )
            else:
                row.update(
                    {
                        "identity_status": "source_data_unavailable",
                        "experience_status": "source_data_unavailable",
                        "impact_status": "source_data_unavailable",
                    }
                )
            audit_rows.append(row)
            continue
        player = stats_matches[0]
        zero_status = "zero_recorded_defensive_box_score_games"
        is_zero_recorded = player.recorded_defensive_box_score_games == 0
        row.update(
            {
                "identity_status": zero_status if is_zero_recorded else "resolved",
                "experience_status": (
                    zero_status
                    if is_zero_recorded
                    and player.defensive_box_score_game_rate is not None
                    else "resolved"
                    if player.defensive_box_score_game_rate is not None
                    else "source_data_unavailable"
                ),
                "impact_status": (
                    zero_status
                    if is_zero_recorded and player.defensive_impact is not None
                    else "resolved"
                    if player.defensive_impact is not None
                    else "source_data_unavailable"
                ),
                "prior_team_games": player.team_games,
                "prior_recorded_defensive_box_score_games": player.recorded_defensive_box_score_games,
                "prior_defensive_box_score_game_rate": player.defensive_box_score_game_rate,
                "prior_defensive_impact": player.defensive_impact,
                "prior_stats": dict(player.stats),
            }
        )
        audit_rows.append(row)

    features: list[Row] = []
    for team in team_list:
        season = int(team["season"])
        key_team_id = str(team["team_id"])
        incoming = [
            row
            for row in audit_rows
            if row["in_model_relevant_population"]
            and row["destination_team_id"] == key_team_id
            and int(row["season"]) == season
            and row["defensive_candidate"]
        ]
        experience_usable = [
            row
            for row in incoming
            if row["experience_status"]
            in {"resolved", "zero_recorded_defensive_box_score_games"}
        ]
        impact_usable = [
            row
            for row in incoming
            if row["impact_status"]
            in {"resolved", "zero_recorded_defensive_box_score_games"}
        ]
        experience_value = (
            sum(
                float(row["prior_defensive_box_score_game_rate"] or 0.0)
                for row in experience_usable
            )
            if len(experience_usable) == len(incoming)
            else (0.0 if not incoming else None)
        )
        impact_value = (
            sum(float(row["prior_defensive_impact"] or 0.0) for row in impact_usable)
            if len(impact_usable) == len(incoming)
            else (0.0 if not incoming else None)
        )
        status = (
            "portal_payload_missing"
            if season not in portal_seasons
            else "no_incoming_defensive_transfer"
            if not incoming
            else "complete"
            if len(experience_usable) == len(incoming)
            and len(impact_usable) == len(incoming)
            else "partial"
            if experience_usable or impact_usable
            else "no_usable_defensive_transfer"
        )
        features.append(
            {
                "season": season,
                "subdivision": team.get("subdivision"),
                "team_id": key_team_id,
                "team_name": team.get("team_name"),
                "portal_payload_available": season in portal_seasons,
                "defensive_source_available": season - 1 in defensive_seasons,
                "incoming_fbs_transfers": sum(
                    row["in_model_relevant_population"] and int(row["season"]) == season
                    for row in audit_rows
                    if row["destination_team_id"] == key_team_id
                ),
                "incoming_defensive_transfers": len(incoming),
                "resolved_defensive_experience_count": len(experience_usable),
                "resolved_defensive_impact_count": len(impact_usable),
                "unresolved_experience_count": len(incoming) - len(experience_usable),
                "unresolved_impact_count": len(incoming) - len(impact_usable),
                "transfer_in_prior_defensive_experience_sum": experience_value,
                "transfer_in_prior_defensive_impact_sum": impact_value,
                "feature_coverage_status": status,
            }
        )
    return {
        "player_rows": sorted(audit_rows, key=lambda row: int(row["portal_index"])),
        "team_rows": sorted(
            features, key=lambda row: (int(row["season"]), str(row["team_id"]))
        ),
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_ss = sum((x - x_mean) ** 2 for x in xs)
    y_ss = sum((y - y_mean) ** 2 for y in ys)
    return numerator / sqrt(x_ss * y_ss) if x_ss > 0 and y_ss > 0 else None


def descriptive_statistics(values: Iterable[float | None]) -> Row:
    """Return deterministic distribution diagnostics for one candidate field."""
    values_list = list(values)
    observed = [
        float(value)
        for value in values_list
        if value is not None and isfinite(float(value))
    ]
    ordered = sorted(observed)
    count = len(observed)

    def percentile(q: float) -> float | None:
        if not ordered:
            return None
        if len(ordered) == 1:
            return ordered[0]
        index = (len(ordered) - 1) * q
        lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "count": count,
        "mean": mean(observed) if observed else None,
        "median": median(observed) if observed else None,
        "std": pstdev(observed) if count > 1 else 0.0 if count == 1 else None,
        "p05": percentile(0.05),
        "p25": percentile(0.25),
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "max": max(observed) if observed else None,
        "fraction_zero": (
            sum(value == 0 for value in observed) / count if count else None
        ),
        "missing_count": len(values_list) - count,
    }


def correlation_rows(
    player_rows: Iterable[Mapping[str, Any]], team_rows: Iterable[Mapping[str, Any]]
) -> list[Row]:
    """Measure descriptive experience/impact correlation at both levels."""
    result: list[Row] = []
    for level, rows in (
        ("player", list(player_rows)),
        ("team_season", list(team_rows)),
    ):
        pairs = [
            (
                float(
                    row["prior_defensive_box_score_game_rate"]
                    if level == "player"
                    else row["transfer_in_prior_defensive_experience_sum"]
                ),
                float(
                    row["prior_defensive_impact"]
                    if level == "player"
                    else row["transfer_in_prior_defensive_impact_sum"]
                ),
            )
            for row in rows
            if (
                row.get("experience_status")
                in {"resolved", "zero_recorded_defensive_box_score_games"}
                if level == "player"
                else row.get("transfer_in_prior_defensive_experience_sum") is not None
                and row.get("transfer_in_prior_defensive_impact_sum") is not None
            )
            and (
                row.get("prior_defensive_box_score_game_rate")
                if level == "player"
                else row.get("transfer_in_prior_defensive_experience_sum")
            )
            is not None
            and (
                row.get("prior_defensive_impact")
                if level == "player"
                else row.get("transfer_in_prior_defensive_impact_sum")
            )
            is not None
        ]
        result.append(
            {
                "level": level,
                "count": len(pairs),
                "pearson_experience_vs_impact": _pearson(
                    [pair[0] for pair in pairs], [pair[1] for pair in pairs]
                ),
            }
        )
    return result


def position_mapping() -> Row:
    """Return the machine-readable taxonomy and its unknown-label policy."""
    return {
        "version": "issue-103-v1",
        "groups": {
            "dl_edge": {
                "label": "DL / EDGE",
                "source_positions": sorted(
                    position
                    for position, group in DEFENSIVE_POSITION_GROUPS.items()
                    if group == "dl_edge"
                ),
            },
            "lb": {
                "label": "LB",
                "source_positions": sorted(
                    position
                    for position, group in DEFENSIVE_POSITION_GROUPS.items()
                    if group == "lb"
                ),
            },
            "db": {
                "label": "DB",
                "source_positions": sorted(
                    position
                    for position, group in DEFENSIVE_POSITION_GROUPS.items()
                    if group == "db"
                ),
            },
        },
        "known_non_defensive_positions": sorted(
            OFFENSIVE_POSITION_LABELS | SPECIAL_TEAMS_POSITION_LABELS
        ),
        "unknown_policy": "retain unknown labels and fail closed; never infer a defensive group from an unrecognized or hybrid label",
    }


def source_inventory() -> list[Row]:
    """Return the frozen source inventory used by the research report."""
    return [
        {
            "source": "CFBD /player/portal",
            "fields": "season, firstName, lastName, origin, destination, position, transferDate",
            "historical_coverage": "2021 onward in the repository acquisition workflow",
            "player_identifier": "no player ID in the portal response; name + origin fallback",
            "defensive_participation": "none",
            "defensive_production": "none",
            "preseason_semantics": "retrospective response; final destination is not proven as-of cutoff",
            "reproducibility": "raw response bytes and SHA-256 sidecar",
        },
        {
            "source": "CFBD /roster?year=&classification=fbs|fcs",
            "fields": "id, firstName, lastName, team, position",
            "historical_coverage": "season-wide FBS and FCS roster snapshots for requested years",
            "player_identifier": "CFBD athlete ID",
            "defensive_participation": "position identity only",
            "defensive_production": "none",
            "preseason_semantics": "retrospective roster response; used for prior-season identity and position",
            "reproducibility": "raw response bytes and SHA-256 sidecar",
        },
        {
            "source": "CFBD /games/players?year=&week=&classification=fbs|fcs&seasonType=both",
            "fields": "game/team defensive rows: TOT, SOLO, SACKS, TFL, PD, QB HUR; interception INT; fumble REC",
            "historical_coverage": "week-level FBS and FCS game box scores for requested seasons",
            "player_identifier": "CFBD athlete ID",
            "defensive_participation": "recorded defensive box-score games; no snaps or observed participation",
            "defensive_production": "box-score event counts, not play-level opportunity-adjusted impact",
            "preseason_semantics": "completed prior-season outcomes; safe for a prior-season feature after frozen retrieval",
            "reproducibility": "raw weekly response bytes and SHA-256 sidecar",
        },
        {
            "source": "CFBD /player/usage",
            "fields": "usage.overall and offensive usage components",
            "historical_coverage": "existing transfer-oracle acquisition seasons",
            "player_identifier": "CFBD athlete ID, not shared with portal",
            "defensive_participation": "not a defensive snap measure",
            "defensive_production": "not used",
            "preseason_semantics": "offensive participation only; excluded from this feature",
            "reproducibility": "existing raw response bytes and sidecar",
        },
    ]


def field_inventory() -> list[Row]:
    """Return the explicit availability audit for requested defensive fields."""
    return [
        {
            "field": "defensive snaps",
            "availability": "unavailable",
            "measurement": "no snap-count field in selected CFBD endpoints",
            "decision": "do not label the fallback as snap share",
        },
        {
            "field": "defensive snap share",
            "availability": "unavailable",
            "measurement": "no defensive denominator or participation percentage",
            "decision": "use recorded defensive box-score game rate only",
        },
        {
            "field": "games played",
            "availability": "proxy available",
            "measurement": "distinct team games containing the player's recorded defensive box-score row",
            "decision": "frozen experience proxy numerator",
        },
        {
            "field": "starts",
            "availability": "unavailable",
            "measurement": "not present in roster or games/players responses",
            "decision": "not used",
        },
        {
            "field": "defensive play participation",
            "availability": "unavailable",
            "measurement": "no player defensive-play count in selected endpoint",
            "decision": "not used",
        },
        {
            "field": "tackles / solo tackles / tackles for loss / sacks",
            "availability": "available",
            "measurement": "games/players defensive category TOT, SOLO, TFL, SACKS",
            "decision": "position-specific impact components where declared",
        },
        {
            "field": "quarterback hurries",
            "availability": "available",
            "measurement": "games/players defensive category QB HUR",
            "decision": "DL/EDGE impact component",
        },
        {
            "field": "passes defended",
            "availability": "available",
            "measurement": "games/players defensive category PD",
            "decision": "LB/DB impact component",
        },
        {
            "field": "interceptions",
            "availability": "available",
            "measurement": "games/players interceptions category INT",
            "decision": "DB impact component",
        },
        {
            "field": "forced fumbles",
            "availability": "unavailable",
            "measurement": "no player forced-fumble field in selected endpoint",
            "decision": "not used",
        },
        {
            "field": "fumble recoveries",
            "availability": "available",
            "measurement": "games/players fumbles category REC; retained as audit data",
            "decision": "not in the frozen impact composite",
        },
        {
            "field": "position / player ID / source team / season",
            "availability": "available",
            "measurement": "roster identity plus game-player and portal season fields",
            "decision": "join and position taxonomy inputs",
        },
    ]


def cutoff_safety() -> list[Row]:
    """Return field-level leakage and timing classifications."""
    return [
        {
            "field": "prior recorded defensive box-score games",
            "classification": "prior-season outcome; cutoff-safe once frozen",
            "limitation": "recorded box-score activity is not snap share or observed participation and may miss unrecorded activity",
        },
        {
            "field": "prior defensive production",
            "classification": "prior-season outcome; cutoff-safe once frozen",
            "limitation": "opportunity, scheme, and position labeling are not fully observed",
        },
        {
            "field": "portal destination",
            "classification": "retrospective research oracle",
            "limitation": "current endpoint may contain a later final destination or revision",
        },
        {
            "field": "player identity",
            "classification": "deterministic fallback",
            "limitation": "portal lacks a shared athlete ID; ambiguous name + source-team joins fail closed",
        },
    ]


__all__ = [
    "ALL_STAT_FIELDS",
    "DEFENSIVE_POSITION_GROUPS",
    "IMPACT_COMPONENTS",
    "DefensiveGamePlayer",
    "DefensivePlayerSeason",
    "RosterPlayer",
    "TeamResolver",
    "add_defensive_impact",
    "aggregate_player_seasons",
    "audit_transfer_records",
    "correlation_rows",
    "cutoff_safety",
    "defensive_position_group",
    "descriptive_statistics",
    "field_inventory",
    "parse_games_players_payload",
    "parse_roster_payload",
    "position_mapping",
    "position_semantics",
    "source_inventory",
    "team_game_keys",
]
