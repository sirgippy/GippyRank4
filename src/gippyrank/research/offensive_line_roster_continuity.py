"""Normalize CFBD rosters and describe prior same-program OL roster overlap.

This is a research-only data product.  It treats CFBD's season-keyed rosters
as retrospective roster observations, not preseason snapshots, and never uses
target-season roster membership as evidence of prior continuity.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gippyrank.transfer_oracle import normalize_player_name, normalize_team_name

OL_POSITION_LABELS = frozenset(
    {
        "C",
        "G",
        "IOL",
        "LG",
        "LT",
        "OC",
        "OG",
        "OL",
        "OT",
        "RG",
        "RT",
        "T",
        "INTERIOR OL",
        "OFFENSIVE LINE",
        "O LINE",
    }
)

NON_OL_POSITION_LABELS = frozenset(
    {
        "A",
        "ATH",
        "B",
        "CB",
        "DB",
        "DE",
        "DL",
        "DT",
        "EDGE",
        "FB",
        "FS",
        "H",
        "ILB",
        "K",
        "LB",
        "LS",
        "MLB",
        "NB",
        "NT",
        "OLB",
        "P",
        "PK",
        "QB",
        "RB",
        "S",
        "SS",
        "TE",
        "WR",
    }
)


@dataclass(frozen=True)
class ContinuityBuild:
    """Normalized rows and derived audit tables for one roster corpus."""

    player_seasons: list[dict[str, Any]]
    ol_player_seasons: list[dict[str, Any]]
    position_vocabulary: list[dict[str, Any]]
    identity_audit: list[dict[str, Any]]
    identity_events: list[dict[str, Any]]
    team_season_coverage: list[dict[str, Any]]
    coverage_by_season: list[dict[str, Any]]
    pairwise_continuity: list[dict[str, Any]]
    team_season_summaries: list[dict[str, Any]]
    source_audit: dict[str, Any]


def normalize_position_label(value: Any) -> str:
    """Return a stable label while retaining the original label separately."""
    return re.sub(r"\s+", " ", str(value or "").strip().upper())


def classify_offensive_line_position(value: Any) -> str:
    """Classify explicit OL labels and fail closed on unknown or hybrid labels."""
    label = normalize_position_label(value)
    if not label:
        return "unknown"
    if label in OL_POSITION_LABELS:
        return "offensive_line"
    if label in NON_OL_POSITION_LABELS:
        return "non_offensive_line"
    if any(separator in label for separator in ("/", "&", "+", "-", ";", ",")):
        return "ambiguous"
    if re.search(r"\b(?:AND|OR)\b", label):
        return "ambiguous"
    return "unknown"


def _team_alias_maps(
    fbs_teams_by_season: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[dict[int, dict[str, set[str]]], dict[str, set[str]]]:
    by_season: dict[int, dict[str, set[str]]] = {}
    all_fbs_aliases: dict[str, set[str]] = defaultdict(set)
    for season, teams in fbs_teams_by_season.items():
        aliases: dict[str, set[str]] = defaultdict(set)
        for team in teams:
            team_id = str(team.get("id") or "").strip()
            if not team_id:
                continue
            names = [team.get("school"), *(team.get("alternateNames") or [])]
            for name in names:
                normalized = normalize_team_name(str(name or ""))
                if normalized:
                    aliases[normalized].add(team_id)
                    all_fbs_aliases[normalized].add(team_id)
        by_season[int(season)] = dict(aliases)
    return by_season, dict(all_fbs_aliases)


def _match_team(
    source_name: str,
    *,
    season: int,
    source_classification: str,
    aliases_by_season: Mapping[int, Mapping[str, set[str]]],
    all_fbs_aliases: Mapping[str, set[str]],
) -> tuple[str, str]:
    alias = normalize_team_name(source_name)
    if not alias:
        return "", "missing_roster_team_name"
    aliases = (
        aliases_by_season.get(season, {})
        if source_classification == "fbs"
        else all_fbs_aliases
    )
    matches = aliases.get(alias, set())
    if len(matches) == 1:
        return next(iter(matches)), "matched_fbs_program"
    if len(matches) > 1:
        return "", "ambiguous_fbs_team_alias"
    return "", "unmatched_fbs_program_alias"


def normalize_roster_records(
    roster_payloads: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    fbs_teams_by_season: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Normalize FBS/FCS response rows and attach stable FBS team IDs.

    FCS roster rows are retained when the school maps to an FBS program or the
    player ID also appears in an FBS roster. The first case can establish
    same-program exposure before FBS membership; the second can audit an ID
    across an FCS-to-FBS program change. Other FCS records stay in the raw
    corpus but are outside this FBS-focused normalized panel.
    """
    aliases_by_season, all_fbs_aliases = _team_alias_maps(fbs_teams_by_season)
    rows: list[dict[str, Any]] = []
    for (classification, season), payload in sorted(roster_payloads.items()):
        source_classification = classification.casefold()
        for row_number, item in enumerate(payload, start=1):
            source_team = str(item.get("team") or "").strip()
            team_id, team_match_status = _match_team(
                source_team,
                season=int(season),
                source_classification=source_classification,
                aliases_by_season=aliases_by_season,
                all_fbs_aliases=all_fbs_aliases,
            )
            player_id = str(item.get("id") or "").strip()
            first_name = str(item.get("firstName") or "").strip()
            last_name = str(item.get("lastName") or "").strip()
            player_name = " ".join(part for part in (first_name, last_name) if part)
            position_original = str(item.get("position") or "").strip()
            rows.append(
                {
                    "season": int(season),
                    "source_classification": source_classification,
                    "source_row_number": row_number,
                    "team_id": team_id,
                    "team_name": source_team,
                    "team_match_status": team_match_status,
                    "source_player_id": player_id,
                    "normalized_player_identity": f"cfbd:{player_id}"
                    if player_id
                    else "",
                    "first_name": first_name,
                    "last_name": last_name,
                    "player_name": player_name,
                    "normalized_player_name": normalize_player_name(player_name),
                    "position_original": position_original,
                    "position_normalized": normalize_position_label(position_original),
                    "normalized_ol_status": classify_offensive_line_position(
                        position_original
                    ),
                    "jersey_number": item.get("jersey"),
                    "duplicate_id_team_season_count": 1,
                    "duplicate_id_team_season_status": (
                        "unresolved_missing_id" if not player_id else "unique"
                    ),
                }
            )

    duplicate_groups: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["source_player_id"] and row["team_id"]:
            duplicate_groups[
                (row["season"], row["team_id"], row["source_player_id"])
            ].append(index)
    for indices in duplicate_groups.values():
        if len(indices) < 2:
            continue
        group = [rows[index] for index in indices]
        names = {row["normalized_player_name"] for row in group}
        positions = {row["position_normalized"] for row in group}
        status = (
            "repeated_id_team_season_rows_same_name_position"
            if len(names) <= 1 and len(positions) <= 1
            else "conflicting_id_team_season_rows"
        )
        for index in indices:
            rows[index]["duplicate_id_team_season_count"] = len(indices)
            rows[index]["duplicate_id_team_season_status"] = status
    fbs_player_ids = {
        str(row["source_player_id"])
        for row in rows
        if row["source_classification"] == "fbs" and row["source_player_id"]
    }
    return [
        row
        for row in rows
        if row["source_classification"] == "fbs"
        or row["team_id"]
        or (
            row["source_classification"] == "fcs"
            and row["source_player_id"] in fbs_player_ids
        )
    ]


def _identity_keys(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        player_id = str(row.get("source_player_id") or "")
        if player_id:
            grouped[player_id].append(row)
    return dict(grouped)


def audit_player_identities(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize provider-ID continuity and emit unresolved identity events."""
    identities: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    by_id = _identity_keys(rows)
    for player_id, records in sorted(by_id.items()):
        seasons = sorted({int(row["season"]) for row in records})
        names = sorted(
            {str(row["player_name"]) for row in records if row["player_name"]}
        )
        normalized_names = sorted(
            {
                str(row["normalized_player_name"])
                for row in records
                if row["normalized_player_name"]
            }
        )
        positions = sorted(
            {
                str(row["position_normalized"])
                for row in records
                if row["position_normalized"]
            }
        )
        programs_by_season: dict[int, set[str]] = defaultdict(set)
        team_names_by_season: dict[int, set[str]] = defaultdict(set)
        for row in records:
            program_key = str(row["team_id"] or "")
            if not program_key:
                program_key = "source:" + normalize_team_name(
                    str(row["team_name"] or "")
                )
            if program_key != "source:":
                programs_by_season[int(row["season"])].add(program_key)
                team_names_by_season[int(row["season"])].add(str(row["team_name"]))
        adjacent_same_team = sum(
            bool(
                programs_by_season.get(season, set())
                & programs_by_season.get(season + 1, set())
            )
            for season in seasons
            if season + 1 in programs_by_season
        )
        multiteam_seasons = sorted(
            season
            for season, programs in programs_by_season.items()
            if len(programs) > 1
        )
        identity_row = {
            "source_player_id": player_id,
            "normalized_player_identity": f"cfbd:{player_id}",
            "first_observed_season": min(seasons),
            "last_observed_season": max(seasons),
            "observed_seasons": "|".join(map(str, seasons)),
            "observed_fbs_program_ids": "|".join(
                sorted({str(row["team_id"]) for row in records if row["team_id"]})
            ),
            "observed_program_keys": "|".join(
                sorted(
                    {
                        str(row["team_id"])
                        if row["team_id"]
                        else "source:"
                        + normalize_team_name(str(row["team_name"] or ""))
                        for row in records
                        if row["team_id"]
                        or normalize_team_name(str(row["team_name"] or ""))
                    }
                )
            ),
            "observed_source_team_names": " | ".join(
                sorted({str(row["team_name"]) for row in records if row["team_name"]})
            ),
            "source_name_count": len(names),
            "source_names": " | ".join(names),
            "normalized_name_variant_count": len(normalized_names),
            "normalized_names": " | ".join(normalized_names),
            "position_variant_count": len(positions),
            "positions": " | ".join(positions),
            "adjacent_same_program_return_links": adjacent_same_team,
            "multi_team_seasons": "|".join(map(str, multiteam_seasons)),
            "roster_row_count": len(records),
        }
        identities.append(identity_row)
        if len(names) > 1:
            events.append(
                {
                    "event_type": "source_name_variant_for_same_id",
                    "source_player_id": player_id,
                    "season": "|".join(map(str, seasons)),
                    "team_id": identity_row["observed_fbs_program_ids"],
                    "evidence": identity_row["source_names"],
                    "resolution": "provider ID retained; names preserved as variants",
                }
            )
        if len(positions) > 1:
            events.append(
                {
                    "event_type": "position_change_or_label_change_for_same_id",
                    "source_player_id": player_id,
                    "season": "|".join(map(str, seasons)),
                    "team_id": identity_row["observed_fbs_program_ids"],
                    "evidence": identity_row["positions"],
                    "resolution": "provider ID retained; original positions preserved",
                }
            )
        for season in multiteam_seasons:
            events.append(
                {
                    "event_type": "same_id_listed_at_multiple_programs_in_season",
                    "source_player_id": player_id,
                    "season": season,
                    "team_id": "|".join(
                        sorted(
                            {
                                str(row["team_id"])
                                for row in records
                                if int(row["season"]) == season and row["team_id"]
                            }
                        )
                    ),
                    "program_key": "|".join(sorted(programs_by_season[season])),
                    "evidence": "|".join(sorted(team_names_by_season[season])),
                    "resolution": "retrospective multi-team membership; timing unresolved",
                }
            )
        program_keys = sorted(
            {
                program
                for season_programs in programs_by_season.values()
                for program in season_programs
            }
        )
        if len(program_keys) > 1:
            events.append(
                {
                    "event_type": "same_id_observed_at_multiple_programs_across_seasons",
                    "source_player_id": player_id,
                    "season": "|".join(map(str, seasons)),
                    "team_id": identity_row["observed_fbs_program_ids"],
                    "program_key": "|".join(program_keys),
                    "evidence": identity_row["observed_source_team_names"],
                    "resolution": "provider ID links program changes; transfer timing is not inferred",
                }
            )

    by_name_team_season: dict[tuple[int, str, str], set[str]] = defaultdict(set)
    for row in rows:
        if row["team_id"] and row["normalized_player_name"] and row["source_player_id"]:
            by_name_team_season[
                (
                    int(row["season"]),
                    str(row["team_id"]),
                    str(row["normalized_player_name"]),
                )
            ].add(str(row["source_player_id"]))
    for (season, team_id, normalized_name), ids in sorted(by_name_team_season.items()):
        if len(ids) > 1:
            events.append(
                {
                    "event_type": "duplicate_normalized_name_same_program_season",
                    "source_player_id": "|".join(sorted(ids)),
                    "season": season,
                    "team_id": team_id,
                    "evidence": normalized_name,
                    "resolution": "distinct provider IDs retained; no name-based merge",
                }
            )

    name_team_ids: dict[tuple[int, str, str], set[str]] = defaultdict(set)
    for (season, team_id, normalized_name), ids in by_name_team_season.items():
        name_team_ids[(season, team_id, normalized_name)].update(ids)
    for (season, team_id, normalized_name), current_ids in sorted(
        name_team_ids.items()
    ):
        previous_ids = name_team_ids.get((season - 1, team_id, normalized_name), set())
        if previous_ids and current_ids and not previous_ids.intersection(current_ids):
            unique_pair = len(previous_ids) == 1 and len(current_ids) == 1
            events.append(
                {
                    "event_type": (
                        "possible_player_id_change_unresolved"
                        if unique_pair
                        else "ambiguous_same_name_across_adjacent_seasons"
                    ),
                    "source_player_id": "previous="
                    + "|".join(sorted(previous_ids))
                    + ";current="
                    + "|".join(sorted(current_ids)),
                    "season": f"{season - 1}|{season}",
                    "team_id": team_id,
                    "evidence": normalized_name,
                    "resolution": "exact name/team evidence only; IDs are not merged",
                }
            )
    events.sort(
        key=lambda row: (
            str(row["event_type"]),
            str(row["season"]),
            str(row["team_id"]),
            str(row["source_player_id"]),
        )
    )
    return identities, events


def _player_units_for_team_season(
    rows: Sequence[Mapping[str, Any]], *, season: int, team_id: str
) -> tuple[list[dict[str, Any]], int]:
    relevant = [
        row
        for row in rows
        if int(row["season"]) == season and str(row["team_id"]) == team_id
    ]
    by_player_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    no_id_rows: list[Mapping[str, Any]] = []
    for row in relevant:
        player_id = str(row.get("source_player_id") or "")
        if player_id:
            by_player_id[player_id].append(row)
        else:
            no_id_rows.append(row)

    units: list[dict[str, Any]] = []
    ambiguous_ol_rows = sum(
        row["normalized_ol_status"] == "ambiguous" for row in relevant
    )
    for player_id, records in sorted(by_player_id.items()):
        statuses = {str(record["normalized_ol_status"]) for record in records}
        if "offensive_line" not in statuses:
            continue
        if "non_offensive_line" in statuses or "ambiguous" in statuses:
            ambiguous_ol_rows += sum(
                record["normalized_ol_status"] == "offensive_line" for record in records
            )
            continue
        units.append(
            {
                "player_id": player_id,
                "player_identity": f"cfbd:{player_id}",
                "player_key": f"cfbd:{player_id}",
                "player_name": " | ".join(
                    sorted(
                        {
                            str(record["player_name"])
                            for record in records
                            if record["player_name"]
                        }
                    )
                ),
                "normalized_player_name": " | ".join(
                    sorted(
                        {
                            str(record["normalized_player_name"])
                            for record in records
                            if record["normalized_player_name"]
                        }
                    )
                ),
                "position_original": " | ".join(
                    sorted(
                        {
                            str(record["position_original"])
                            for record in records
                            if record["position_original"]
                        }
                    )
                ),
                "identity_status": (
                    "provider_id_with_conflicting_duplicate_rows"
                    if any(
                        record["duplicate_id_team_season_status"]
                        == "conflicting_id_team_season_rows"
                        for record in records
                    )
                    else "provider_id"
                ),
            }
        )
    for record in no_id_rows:
        if record["normalized_ol_status"] != "offensive_line":
            continue
        player_key = f"unresolved:{season}:{team_id}:{record['source_classification']}:{record['source_row_number']}"
        units.append(
            {
                "player_id": "",
                "player_identity": "",
                "player_key": player_key,
                "player_name": str(record["player_name"]),
                "normalized_player_name": str(record["normalized_player_name"]),
                "position_original": str(record["position_original"]),
                "identity_status": "unresolved_missing_provider_id",
            }
        )
    units.sort(
        key=lambda unit: (unit["player_id"], unit["player_name"], unit["player_key"])
    )
    return units, ambiguous_ol_rows


def derive_continuity_tables(
    rows: Sequence[Mapping[str, Any]],
    fbs_teams_by_season: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    roster_response_seasons: set[tuple[str, int]],
    start_season: int,
    end_season: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Build FBS target coverage, player-pair records, and team summaries."""
    memberships: dict[tuple[str, str], set[int]] = defaultdict(set)
    roster_rows: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        team_id = str(row.get("team_id") or "")
        if not team_id:
            continue
        season = int(row["season"])
        if row["source_classification"] == "fbs":
            roster_rows[(season, team_id)].append(row)
        player_id = str(row.get("source_player_id") or "")
        if player_id:
            memberships[(team_id, player_id)].add(season)

    coverage_rows: list[dict[str, Any]] = []
    coverage_by_season: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for season in range(start_season, end_season + 1):
        teams = list(fbs_teams_by_season.get(season, ()))
        season_coverage: list[dict[str, Any]] = []
        for team in sorted(
            teams,
            key=lambda record: (
                int(record.get("id") or 0),
                str(record.get("school") or ""),
            ),
        ):
            team_id = str(team.get("id") or "").strip()
            team_name = str(team.get("school") or "").strip()
            records = roster_rows.get((season, team_id), [])
            is_roster_response_available = ("fbs", season) in roster_response_seasons
            units, ambiguous_ol_rows = _player_units_for_team_season(
                records, season=season, team_id=team_id
            )
            identified_units = [unit for unit in units if unit["player_id"]]
            linked_prior_year_counts = [
                len(
                    {
                        year
                        for year in memberships.get((team_id, unit["player_id"]), set())
                        if year < season
                    }
                )
                for unit in units
                if unit["player_id"]
            ]
            no_prior_linked_count = sum(
                count == 0 for count in linked_prior_year_counts
            )
            ol_record_count = sum(
                row["normalized_ol_status"] == "offensive_line" for row in records
            )
            unique_roster_ids = {
                str(row["source_player_id"])
                for row in records
                if row["source_player_id"]
            }
            missing_id_rows = sum(not row["source_player_id"] for row in records)
            rows_with_name = sum(bool(row["normalized_player_name"]) for row in records)
            rows_with_position = sum(
                bool(row["position_normalized"]) for row in records
            )
            rows_with_jersey = sum(
                row["jersey_number"] not in (None, "") for row in records
            )
            duplicate_rows = sum(
                int(row["duplicate_id_team_season_count"]) > 1 for row in records
            )
            unknown_position_rows = sum(
                row["normalized_ol_status"] == "unknown" for row in records
            )
            if not is_roster_response_available:
                roster_status = "roster_response_unavailable"
            elif not records:
                roster_status = "roster_missing_for_expected_team"
            elif not units:
                roster_status = "roster_present_no_identifiable_ol"
            else:
                roster_status = "roster_present_with_identifiable_ol"
            coverage = {
                "season": season,
                "team_id": team_id,
                "team_name": team_name,
                "roster_response_available": is_roster_response_available,
                "roster_status": roster_status,
                "roster_player_rows": len(records),
                "unique_roster_player_ids": len(unique_roster_ids),
                "roster_rows_with_player_id": len(records) - missing_id_rows,
                "roster_rows_with_player_name": rows_with_name,
                "roster_rows_with_position": rows_with_position,
                "roster_rows_with_jersey_number": rows_with_jersey,
                "duplicate_id_team_season_rows": duplicate_rows,
                "roster_rows_missing_player_id": missing_id_rows,
                "identifiable_ol_player_count": len(units),
                "ol_player_ids_available": len(identified_units),
                "ol_player_ids_missing": len(units) - len(identified_units),
                "ol_with_prior_same_program_roster_link": (
                    ""
                    if season == start_season
                    else sum(count > 0 for count in linked_prior_year_counts)
                ),
                "ol_with_no_prior_same_program_roster_link": (
                    "" if season == start_season else no_prior_linked_count
                ),
                "aggregate_prior_program_roster_seasons_for_linked_ol": (
                    "" if season == start_season else sum(linked_prior_year_counts)
                ),
                "ol_source_rows": ol_record_count,
                "ambiguous_ol_position_rows": ambiguous_ol_rows,
                "unknown_position_rows": unknown_position_rows,
            }
            coverage_rows.append(coverage)
            season_coverage.append(coverage)

            left_censored = season == start_season
            prior_counts: list[int] = []
            for prior_season in range(start_season, season):
                prior_ids = {
                    unit["player_id"]
                    for unit in units
                    if unit["player_id"]
                    and prior_season
                    in memberships.get((team_id, unit["player_id"]), set())
                }
                if prior_ids:
                    prior_counts.append(len(prior_ids))

            evaluable_pair_count = 0
            unresolved_pair_count = 0
            pair_shared_season_counts: list[int] = []
            pairs_with_1 = pairs_with_2 = pairs_with_3 = 0
            if len(units) > 1:
                for first, second in itertools.combinations(units, 2):
                    both_linked = bool(first["player_id"] and second["player_id"])
                    if both_linked:
                        first_years = memberships.get(
                            (team_id, first["player_id"]), set()
                        )
                        second_years = memberships.get(
                            (team_id, second["player_id"]), set()
                        )
                        shared = sorted(
                            year for year in first_years & second_years if year < season
                        )
                        if left_censored:
                            pair_status = "left_censored_no_prior_source_season"
                            shared_value: int | str = ""
                            earliest: int | str = ""
                            most_recent: int | str = ""
                            consecutive: int | str = ""
                        else:
                            pair_status = (
                                "shared_prior_roster_seasons_observed"
                                if shared
                                else "no_shared_prior_roster_season_observed"
                            )
                            shared_value = len(shared)
                            earliest = shared[0] if shared else ""
                            most_recent = shared[-1] if shared else ""
                            consecutive = 0
                            year = season - 1
                            while year in shared:
                                consecutive += 1
                                year -= 1
                            evaluable_pair_count += 1
                            pair_shared_season_counts.append(len(shared))
                            pairs_with_1 += len(shared) >= 1
                            pairs_with_2 += len(shared) >= 2
                            pairs_with_3 += len(shared) >= 3
                    else:
                        shared = []
                        pair_status = "unresolved_target_player_identity"
                        shared_value = earliest = most_recent = consecutive = ""
                        unresolved_pair_count += 1
                    pair_rows.append(
                        {
                            "target_season": season,
                            "team_id": team_id,
                            "team_name": team_name,
                            "player_1_id": first["player_id"],
                            "player_1_identity": first["player_identity"],
                            "player_1_name": first["player_name"],
                            "player_1_position": first["position_original"],
                            "player_1_identity_status": first["identity_status"],
                            "player_2_id": second["player_id"],
                            "player_2_identity": second["player_identity"],
                            "player_2_name": second["player_name"],
                            "player_2_position": second["position_original"],
                            "player_2_identity_status": second["identity_status"],
                            "shared_prior_season_count": shared_value,
                            "earliest_shared_prior_season": earliest,
                            "most_recent_shared_prior_season": most_recent,
                            "consecutive_shared_seasons_immediately_before_target": consecutive,
                            "pair_status": pair_status,
                        }
                    )

            summary_rows.append(
                {
                    **coverage,
                    "history_left_censored": left_censored,
                    "pair_count_evaluable": evaluable_pair_count,
                    "pair_count_identity_unresolved": unresolved_pair_count,
                    "total_pairwise_shared_seasons": (
                        "" if left_censored else sum(pair_shared_season_counts)
                    ),
                    "mean_pairwise_shared_seasons": (
                        ""
                        if left_censored or not pair_shared_season_counts
                        else sum(pair_shared_season_counts)
                        / len(pair_shared_season_counts)
                    ),
                    "max_pairwise_shared_seasons": (
                        ""
                        if left_censored or not pair_shared_season_counts
                        else max(pair_shared_season_counts)
                    ),
                    "pairs_with_at_least_1_shared_season": ""
                    if left_censored
                    else pairs_with_1,
                    "pairs_with_at_least_2_shared_seasons": ""
                    if left_censored
                    else pairs_with_2,
                    "pairs_with_at_least_3_shared_seasons": ""
                    if left_censored
                    else pairs_with_3,
                    "largest_target_ol_group_on_one_prior_roster": (
                        "" if left_censored else max(prior_counts, default=0)
                    ),
                    "ol_with_no_prior_roster_season_at_current_program": (
                        "" if left_censored else no_prior_linked_count
                    ),
                    "ol_share_with_no_prior_roster_season_at_current_program": (
                        ""
                        if left_censored or not linked_prior_year_counts
                        else no_prior_linked_count / len(linked_prior_year_counts)
                    ),
                    "aggregate_prior_program_roster_seasons_for_linked_ol": (
                        "" if left_censored else sum(linked_prior_year_counts)
                    ),
                    "prior_roster_seasons_available_in_panel": max(
                        0, season - start_season
                    ),
                }
            )

        expected = len(teams)
        missing = sum(
            row["roster_status"] == "roster_missing_for_expected_team"
            for row in season_coverage
        )
        unavailable = sum(
            row["roster_status"] == "roster_response_unavailable"
            for row in season_coverage
        )
        no_ol = sum(
            row["roster_status"] == "roster_present_no_identifiable_ol"
            for row in season_coverage
        )
        with_ol = sum(
            row["roster_status"] == "roster_present_with_identifiable_ol"
            for row in season_coverage
        )
        season_ol_counts = [
            int(row["identifiable_ol_player_count"])
            for row in season_coverage
            if row["roster_status"] == "roster_present_with_identifiable_ol"
        ]
        fbs_roster_rows = sum(int(row["roster_player_rows"]) for row in season_coverage)
        fbs_rows_with_id = sum(
            int(row["roster_rows_with_player_id"]) for row in season_coverage
        )
        fbs_rows_with_name = sum(
            int(row["roster_rows_with_player_name"]) for row in season_coverage
        )
        fbs_rows_with_position = sum(
            int(row["roster_rows_with_position"]) for row in season_coverage
        )
        fbs_rows_with_jersey = sum(
            int(row["roster_rows_with_jersey_number"]) for row in season_coverage
        )
        coverage_by_season.append(
            {
                "season": season,
                "expected_fbs_teams": expected,
                "roster_response_available": ("fbs", season) in roster_response_seasons,
                "roster_available_team_count": expected - missing - unavailable,
                "roster_missing_team_count": missing,
                "roster_response_unavailable_team_count": unavailable,
                "roster_present_no_identifiable_ol_team_count": no_ol,
                "roster_present_with_identifiable_ol_team_count": with_ol,
                "identified_ol_player_count": sum(season_ol_counts),
                "ol_players_min": min(season_ol_counts, default=""),
                "ol_players_median": _median(season_ol_counts),
                "ol_players_mean": (
                    sum(season_ol_counts) / len(season_ol_counts)
                    if season_ol_counts
                    else ""
                ),
                "ol_players_max": max(season_ol_counts, default=""),
                "fbs_roster_player_rows": fbs_roster_rows,
                "fbs_roster_rows_with_player_id": fbs_rows_with_id,
                "fbs_roster_player_id_rate": _safe_ratio(
                    fbs_rows_with_id, fbs_roster_rows
                ),
                "fbs_roster_rows_with_player_name": fbs_rows_with_name,
                "fbs_roster_player_name_rate": _safe_ratio(
                    fbs_rows_with_name, fbs_roster_rows
                ),
                "fbs_roster_rows_with_position": fbs_rows_with_position,
                "fbs_roster_position_rate": _safe_ratio(
                    fbs_rows_with_position, fbs_roster_rows
                ),
                "fbs_roster_rows_with_jersey_number": fbs_rows_with_jersey,
                "fbs_roster_jersey_number_rate": _safe_ratio(
                    fbs_rows_with_jersey, fbs_roster_rows
                ),
                "ol_player_id_link_rate": _safe_ratio(
                    sum(int(row["ol_player_ids_available"]) for row in season_coverage),
                    sum(
                        int(row["identifiable_ol_player_count"])
                        for row in season_coverage
                    ),
                ),
                "ol_players_with_prior_same_program_roster_link": sum(
                    int(row["ol_with_prior_same_program_roster_link"] or 0)
                    for row in season_coverage
                ),
                "unmatched_roster_team_rows": sum(
                    row["team_match_status"] != "matched_fbs_program"
                    for row in rows
                    if int(row["season"]) == season
                    and row["source_classification"] == "fbs"
                ),
            }
        )
    return coverage_rows, coverage_by_season, pair_rows, summary_rows


def _safe_ratio(numerator: int, denominator: int) -> float | str:
    return numerator / denominator if denominator else ""


def _median(values: Sequence[int]) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def build_position_vocabulary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        counts[
            (
                str(row["position_original"]),
                str(row["position_normalized"]),
                str(row["normalized_ol_status"]),
            )
        ] += 1
    return [
        {
            "position_original": original,
            "position_normalized": normalized,
            "normalized_ol_status": status,
            "roster_row_count": count,
        }
        for (original, normalized, status), count in sorted(counts.items())
    ]


def build_source_audit(
    rows: Sequence[Mapping[str, Any]],
    roster_payloads: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    fbs_teams_by_season: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Return high-level source, identity, and position-vocabulary counts."""
    identities, events = audit_player_identities(rows)
    ol_rows = [row for row in rows if row["normalized_ol_status"] == "offensive_line"]
    all_ids = {str(row["source_player_id"]) for row in rows if row["source_player_id"]}
    target_fbs_rows = [row for row in rows if row["source_classification"] == "fbs"]
    fbs_ids = {
        str(row["source_player_id"])
        for row in target_fbs_rows
        if row["source_player_id"]
    }
    fbs_rows_with_id = sum(bool(row["source_player_id"]) for row in target_fbs_rows)
    fbs_rows_with_name = sum(
        bool(row["normalized_player_name"]) for row in target_fbs_rows
    )
    fbs_rows_with_position = sum(
        bool(row["position_normalized"]) for row in target_fbs_rows
    )
    fbs_rows_with_jersey = sum(
        row["jersey_number"] not in (None, "") for row in target_fbs_rows
    )
    fbs_rows_by_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in target_fbs_rows:
        if row["source_player_id"]:
            fbs_rows_by_id[str(row["source_player_id"])].append(row)
    fcs_source_rows = sum(
        len(payload)
        for (classification, _season), payload in roster_payloads.items()
        if classification.casefold() == "fcs"
    )
    retained_fcs_rows = len(rows) - len(target_fbs_rows)
    return {
        "source": "College Football Data API /roster and /teams/fbs",
        "target_classification": "FBS",
        "history_classifications": ["FBS", "FCS"],
        "fbs_team_seasons": sum(len(teams) for teams in fbs_teams_by_season.values()),
        "roster_request_count": len(roster_payloads),
        "roster_player_rows_in_normalized_panel": len(rows),
        "roster_player_rows_fbs": len(target_fbs_rows),
        "roster_player_rows_fcs_acquired": fcs_source_rows,
        "roster_player_rows_fcs_retained_for_fbs_history_or_id_audit": retained_fcs_rows,
        "roster_player_rows_fcs_excluded_outside_fbs_program_history": max(
            0, fcs_source_rows - retained_fcs_rows
        ),
        "roster_player_rows_with_source_id": sum(
            bool(row["source_player_id"]) for row in rows
        ),
        "fbs_roster_rows_with_source_id": fbs_rows_with_id,
        "fbs_roster_player_id_rate": _safe_ratio(
            fbs_rows_with_id, len(target_fbs_rows)
        ),
        "fbs_roster_rows_with_player_name": fbs_rows_with_name,
        "fbs_roster_player_name_rate": _safe_ratio(
            fbs_rows_with_name, len(target_fbs_rows)
        ),
        "fbs_roster_rows_with_position": fbs_rows_with_position,
        "fbs_roster_position_rate": _safe_ratio(
            fbs_rows_with_position, len(target_fbs_rows)
        ),
        "fbs_roster_rows_with_jersey_number": fbs_rows_with_jersey,
        "fbs_roster_jersey_number_rate": _safe_ratio(
            fbs_rows_with_jersey, len(target_fbs_rows)
        ),
        "unique_source_player_ids": len(all_ids),
        "unique_fbs_source_player_ids": len(fbs_ids),
        "fbs_source_player_ids_observed_in_multiple_seasons": sum(
            len({int(row["season"]) for row in id_rows}) > 1
            for id_rows in fbs_rows_by_id.values()
        ),
        "fbs_source_player_ids_with_adjacent_same_program_return": sum(
            len({int(row["season"]) for row in id_rows}) > 1
            and any(
                int(first["season"]) + 1 == int(second["season"])
                and first["team_id"]
                and first["team_id"] == second["team_id"]
                for first in id_rows
                for second in id_rows
            )
            for id_rows in fbs_rows_by_id.values()
        ),
        "normalized_ol_rows_all_classifications": len(ol_rows),
        "ambiguous_position_rows_all_classifications": sum(
            row["normalized_ol_status"] == "ambiguous" for row in rows
        ),
        "unknown_position_rows_all_classifications": sum(
            row["normalized_ol_status"] == "unknown" for row in rows
        ),
        "unmatched_fbs_roster_team_rows": sum(
            row["source_classification"] == "fbs"
            and row["team_match_status"] != "matched_fbs_program"
            for row in rows
        ),
        "unmatched_fcs_roster_team_rows_retained_for_id_audit": sum(
            row["source_classification"] == "fcs"
            and row["team_match_status"] != "matched_fbs_program"
            for row in rows
        ),
        "identity_rows": len(identities),
        "identity_events": len(events),
        "identity_event_counts": dict(
            sorted(Counter(event["event_type"] for event in events).items())
        ),
        "position_vocabulary": build_position_vocabulary(rows),
    }


def build_continuity_artifacts(
    roster_payloads: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    fbs_teams_by_season: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    roster_response_seasons: set[tuple[str, int]],
    start_season: int,
    end_season: int,
) -> ContinuityBuild:
    """Normalize source payloads and build all reproducible panel outputs."""
    rows = normalize_roster_records(roster_payloads, fbs_teams_by_season)
    identity_audit, identity_events = audit_player_identities(rows)
    coverage, season_coverage, pairs, summaries = derive_continuity_tables(
        rows,
        fbs_teams_by_season,
        roster_response_seasons=roster_response_seasons,
        start_season=start_season,
        end_season=end_season,
    )
    ol_rows = [row for row in rows if row["normalized_ol_status"] == "offensive_line"]
    source_audit = build_source_audit(rows, roster_payloads, fbs_teams_by_season)
    return ContinuityBuild(
        player_seasons=rows,
        ol_player_seasons=ol_rows,
        position_vocabulary=build_position_vocabulary(rows),
        identity_audit=identity_audit,
        identity_events=identity_events,
        team_season_coverage=coverage,
        coverage_by_season=season_coverage,
        pairwise_continuity=pairs,
        team_season_summaries=summaries,
        source_audit=source_audit,
    )
