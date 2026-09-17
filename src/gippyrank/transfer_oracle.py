"""Leakage-audited transfer-roster research features.

This module is intentionally separate from the production Context model.  The
CFBD transfer endpoint is a retrospective portal record, not an archived
August 15 snapshot.  The resulting features are therefore research-oracle
features unless a caller supplies stronger as-of evidence.

Raw API responses are never modified here.  Parsers return normalized records
and aggregation produces a separate team-season feature table.  A missing
portal season is represented by ``None`` features rather than zero transfer
activity; zero is reserved for a covered season with no matching transfer.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Any

PORTAL_ENDPOINT = "/player/portal"
USAGE_ENDPOINT = "/player/usage"
DEFAULT_CUTOFF = date(2025, 8, 15)

POSITION_GROUPS = {
    "QB": "qb",
    "RB": "rb",
    "WR": "wr",
    "TE": "te",
    "IOL": "ol",
    "OT": "ol",
    "OL": "ol",
    "DL": "dl",
    "EDGE": "dl",
    "LB": "lb",
    "CB": "db",
    "S": "db",
    "DB": "db",
    "K": "st",
    "P": "st",
    "LS": "st",
    "ATH": "ath",
}
POSITION_WEIGHTS = {
    "qb": 1.50,
    "rb": 1.00,
    "wr": 1.00,
    "te": 1.00,
    "ol": 1.00,
    "dl": 1.00,
    "lb": 0.90,
    "db": 0.90,
    "st": 0.50,
    "ath": 0.75,
}
POSITION_GROUP_NAMES = tuple(POSITION_WEIGHTS)


@dataclass(frozen=True)
class TransferRecord:
    """One normalized CFBD portal record."""

    season: int
    player_name: str
    origin: str | None
    destination: str | None
    position: str | None
    transfer_date: date | None
    rating: float | None
    stars: int | None
    eligibility: str | None

    @property
    def position_group(self) -> str | None:
        return position_group(self.position)


@dataclass(frozen=True)
class UsageRecord:
    """The prior-season usage value used by the production oracle."""

    season: int
    player_name: str
    team: str
    position: str | None
    overall_usage: float | None


def normalize_player_name(value: str | None) -> str:
    """Normalize a player name for a deterministic, auditable join."""
    if not value:
        return ""
    value = value.casefold().replace("'", "").replace(".", "")
    return " ".join(value.split())


def normalize_team_name(value: str | None) -> str:
    """Normalize transport team names without fuzzy matching."""
    if not value:
        return ""
    value = value.casefold().replace("&", "and")
    value = re.sub(r"[().,/'-]", " ", value)
    return " ".join(value.split())


def position_group(position: str | None) -> str | None:
    """Map source positions to a small predeclared position vocabulary."""
    if not position:
        return None
    return POSITION_GROUPS.get(position.strip().upper())


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_transfer_date(value: Any) -> date | None:
    """Parse the ISO timestamp emitted by CFBD without changing its meaning."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return date.fromisoformat(text[:10])


def parse_transfer_payload(
    payload: Iterable[Mapping[str, Any]], *, season: int | None = None
) -> list[TransferRecord]:
    """Parse a raw portal response while retaining all usable source fields."""
    result = []
    for item in payload:
        record_season = _optional_int(item.get("season"))
        if record_season is None:
            if season is None:
                raise ValueError("transfer record has no season")
            record_season = season
        if season is not None and record_season != season:
            raise ValueError("transfer record season disagrees with requested season")
        first = str(item.get("firstName") or "").strip()
        last = str(item.get("lastName") or "").strip()
        player_name = " ".join(part for part in (first, last) if part)
        if not player_name:
            raise ValueError("transfer record has no player name")
        result.append(
            TransferRecord(
                season=record_season,
                player_name=player_name,
                origin=str(item["origin"]).strip() if item.get("origin") else None,
                destination=(
                    str(item["destination"]).strip()
                    if item.get("destination")
                    else None
                ),
                position=(
                    str(item["position"]).strip() if item.get("position") else None
                ),
                transfer_date=parse_transfer_date(item.get("transferDate")),
                rating=_optional_float(item.get("rating")),
                stars=_optional_int(item.get("stars")),
                eligibility=(
                    str(item["eligibility"]).strip()
                    if item.get("eligibility")
                    else None
                ),
            )
        )
    return result


def parse_usage_payload(
    payload: Iterable[Mapping[str, Any]], *, season: int | None = None
) -> list[UsageRecord]:
    """Parse raw season-wide ``/player/usage`` records."""
    result = []
    for item in payload:
        record_season = _optional_int(item.get("season"))
        if record_season is None:
            if season is None:
                raise ValueError("usage record has no season")
            record_season = season
        if season is not None and record_season != season:
            raise ValueError("usage record season disagrees with requested season")
        usage = item.get("usage")
        overall = usage.get("overall") if isinstance(usage, Mapping) else None
        result.append(
            UsageRecord(
                season=record_season,
                player_name=str(item.get("name") or "").strip(),
                team=str(item.get("team") or "").strip(),
                position=(
                    str(item["position"]).strip() if item.get("position") else None
                ),
                overall_usage=_optional_float(overall),
            )
        )
    return result


def cutoff_date(season: int, month: int = 8, day: int = 15) -> date:
    return date(season, month, day)


def available_by_cutoff(record: TransferRecord, cutoff: date) -> bool:
    """Whether the record has a dated portal event on or before the cutoff."""
    return record.transfer_date is not None and record.transfer_date <= cutoff


def portal_provenance() -> dict[str, object]:
    """Return the field-level leakage classification used in the report."""
    return {
        "portal_season": {
            "source": "CFBD /player/portal season query",
            "classification": "retrospective-research-oracle",
            "reason": "The requested season is explicit, but the endpoint is not an archived as-of snapshot.",
        },
        "transfer_date": {
            "source": "CFBD /player/portal transferDate",
            "classification": "retrospective-research-oracle",
            "reason": "A date allows deterministic cutoff filtering, but historical publication and revision timing are not archived.",
        },
        "origin": {
            "source": "CFBD /player/portal origin",
            "classification": "retrospective-research-oracle",
            "reason": "The source team is present in the retrospective record; source-team naming is not a historical roster snapshot.",
        },
        "destination": {
            "source": "CFBD /player/portal destination",
            "classification": "retrospective-research-oracle",
            "reason": "Final destination can reflect later portal resolution and is not proven knowable by August 15.",
        },
        "position": {
            "source": "CFBD /player/portal position",
            "classification": "retrospective-research-oracle",
            "reason": "The endpoint supplies a position but no archived as-of roster context.",
        },
        "rating_stars": {
            "source": "CFBD /player/portal rating and stars",
            "classification": "retrospective-research-oracle",
            "reason": "Rating coverage and later corrections are not timestamped as-of the cutoff.",
        },
        "prior_usage": {
            "source": "CFBD /player/usage for season-1",
            "classification": "retrospective-research-oracle",
            "reason": "Prior-season usage is outcome data from the source school and is not itself target-season data, but the name-based portal join is not an archived roster join.",
        },
        "scholarship_player_count": {
            "source": "not available in the selected CFBD fields",
            "classification": "unavailable",
            "reason": "Eligibility does not establish scholarship status.",
        },
    }


def _team_index(
    team_rows: Iterable[Mapping[str, str]],
    aliases: Mapping[str, str] | None = None,
) -> dict[tuple[int, str], tuple[str, str]]:
    aliases = aliases or {}
    result: dict[tuple[int, str], tuple[str, str]] = {}
    for row in team_rows:
        season = int(row["season"])
        name = row["team_name"]
        normalized = aliases.get(normalize_team_name(name), normalize_team_name(name))
        result[(season, normalized)] = (str(row["team_id"]), name)
    return result


def _matched_team(
    index: Mapping[tuple[int, str], tuple[str, str]],
    season: int,
    team: str | None,
) -> tuple[str, str] | None:
    if not team:
        return None
    return index.get((season, normalize_team_name(team)))


def _usage_index(
    usage: Iterable[UsageRecord],
) -> dict[tuple[int, str, str], float | None]:
    result: dict[tuple[int, str, str], float | None] = {}
    for item in usage:
        if not item.player_name or not item.team:
            continue
        key = (
            item.season,
            normalize_team_name(item.team),
            normalize_player_name(item.player_name),
        )
        if key not in result or result[key] is None and item.overall_usage is not None:
            result[key] = item.overall_usage
    return result


def _empty_features() -> dict[str, float | None]:
    names = [
        "transfer_data_available",
        *VOLUME_FEATURES,
        *TALENT_FEATURES,
        *PRODUCTION_FEATURES,
    ]
    return {name: None for name in names}


VOLUME_FEATURES = (
    "transfer_in_count",
    "transfer_out_count",
    "transfer_net_count",
    *(f"transfer_in_count_{group}" for group in POSITION_GROUP_NAMES),
    *(f"transfer_out_count_{group}" for group in POSITION_GROUP_NAMES),
)
TALENT_FEATURES = (
    "transfer_in_rating_sum",
    "transfer_in_rating_mean",
    "transfer_in_rating_max",
    "transfer_out_rating_sum",
    "transfer_out_rating_mean",
    "transfer_out_rating_max",
    "transfer_net_rating_sum",
    "transfer_in_stars_4_count",
    "transfer_in_stars_5_count",
    "transfer_out_stars_4_count",
    "transfer_out_stars_5_count",
    "transfer_in_weighted_rating_sum",
    "transfer_out_weighted_rating_sum",
    "transfer_net_weighted_rating_sum",
)
PRODUCTION_FEATURES = (
    "transfer_in_prior_usage_sum",
    "transfer_in_prior_usage_mean",
    "transfer_in_prior_usage_max",
    "transfer_out_prior_usage_sum",
    "transfer_out_prior_usage_mean",
    "transfer_out_prior_usage_max",
    "transfer_net_prior_usage",
    "transfer_in_prior_usage_qb",
    "transfer_out_prior_usage_qb",
    "effective_returning_production",
)
ALL_FEATURES = (
    "transfer_data_available",
    *VOLUME_FEATURES,
    *TALENT_FEATURES,
    *PRODUCTION_FEATURES,
)


def aggregate_team_features(
    records: Iterable[TransferRecord],
    usage: Iterable[UsageRecord],
    team_rows: Iterable[Mapping[str, str]],
    *,
    covered_seasons: set[int],
    cutoff: date,
    aliases: Mapping[str, str] | None = None,
) -> dict[tuple[int, str, str], dict[str, float | None]]:
    """Aggregate dated portal records into canonical team-season features.

    ``team_rows`` should be the exact model population.  A source team is
    resolved in the transfer season; a destination is also resolved in the
    transfer season.  Unmatched names are intentionally excluded and counted
    by the caller's coverage audit rather than fuzzy-matched.
    """
    rows = list(team_rows)
    index = _team_index(rows, aliases)
    usage_idx = _usage_index(usage)
    output = {
        (int(row["season"]), str(row["subdivision"]), str(row["team_id"])): (
            _empty_features()
            if int(row["season"]) not in covered_seasons
            else {
                "transfer_data_available": 1.0,
                **{
                    name: 0.0
                    for name in ALL_FEATURES
                    if name != "transfer_data_available"
                },
            }
        )
        for row in rows
    }
    incoming: defaultdict[
        tuple[int, str, str], list[tuple[TransferRecord, float | None]]
    ] = defaultdict(list)
    outgoing: defaultdict[
        tuple[int, str, str], list[tuple[TransferRecord, float | None]]
    ] = defaultdict(list)
    for record in records:
        if record.season not in covered_seasons or not available_by_cutoff(
            record, cutoff
        ):
            continue
        source = _matched_team(index, record.season, record.origin)
        destination = _matched_team(index, record.season, record.destination)
        prior_season = record.season - 1
        player_key = normalize_player_name(record.player_name)
        prior_usage = (
            usage_idx.get(
                (prior_season, normalize_team_name(record.origin), player_key)
            )
            if record.origin
            else None
        )
        if destination:
            destination_key = (record.season, "fbs", destination[0])
            if destination_key in output:
                incoming[destination_key].append((record, prior_usage))
        if source:
            source_key = (record.season, "fbs", source[0])
            if source_key in output:
                outgoing[source_key].append((record, prior_usage))

    def apply_side(
        target: dict[str, float | None],
        items: list[tuple[TransferRecord, float | None]],
        prefix: str,
    ) -> None:
        target[f"transfer_{prefix}_count"] = float(len(items))
        ratings = [record.rating for record, _ in items if record.rating is not None]
        target[f"transfer_{prefix}_rating_sum"] = (
            float(sum(ratings)) if ratings else (0.0 if not items else None)
        )
        target[f"transfer_{prefix}_rating_mean"] = (
            float(sum(ratings) / len(ratings))
            if ratings
            else (0.0 if not items else None)
        )
        target[f"transfer_{prefix}_rating_max"] = (
            float(max(ratings)) if ratings else (0.0 if not items else None)
        )
        target[f"transfer_{prefix}_weighted_rating_sum"] = (
            float(
                sum(
                    record.rating
                    * POSITION_WEIGHTS.get(record.position_group or "", 1.0)
                    for record, _ in items
                    if record.rating is not None
                )
            )
            if ratings
            else (0.0 if not items else None)
        )
        for stars in (4, 5):
            target[f"transfer_{prefix}_stars_{stars}_count"] = float(
                sum(
                    record.stars is not None and record.stars >= stars
                    for record, _ in items
                )
            )
        for group in POSITION_GROUP_NAMES:
            target[f"transfer_{prefix}_count_{group}"] = float(
                sum(record.position_group == group for record, _ in items)
            )
        prior = [value for _, value in items if value is not None]
        target[f"transfer_{prefix}_prior_usage_sum"] = (
            float(sum(prior)) if prior else (0.0 if not items else None)
        )
        target[f"transfer_{prefix}_prior_usage_mean"] = (
            float(sum(prior) / len(prior)) if prior else (0.0 if not items else None)
        )
        target[f"transfer_{prefix}_prior_usage_max"] = (
            float(max(prior)) if prior else (0.0 if not items else None)
        )
        target[f"transfer_{prefix}_prior_usage_qb"] = (
            float(
                sum(
                    value
                    for (record, value) in items
                    if record.position_group == "qb" and value is not None
                )
            )
            if any(
                record.position_group == "qb" and value is not None
                for record, value in items
            )
            else (0.0 if not items else None)
        )

    for key, target in output.items():
        if target["transfer_data_available"] is None:
            continue
        apply_side(target, incoming[key], "in")
        apply_side(target, outgoing[key], "out")
        target["transfer_net_count"] = float(
            target["transfer_in_count"] - target["transfer_out_count"]
        )
        if (
            target["transfer_in_rating_sum"] is not None
            and target["transfer_out_rating_sum"] is not None
        ):
            target["transfer_net_rating_sum"] = (
                target["transfer_in_rating_sum"] - target["transfer_out_rating_sum"]
            )
        else:
            target["transfer_net_rating_sum"] = None
        if (
            target["transfer_in_weighted_rating_sum"] is not None
            and target["transfer_out_weighted_rating_sum"] is not None
        ):
            target["transfer_net_weighted_rating_sum"] = (
                target["transfer_in_weighted_rating_sum"]
                - target["transfer_out_weighted_rating_sum"]
            )
        else:
            target["transfer_net_weighted_rating_sum"] = None
        if (
            target["transfer_in_prior_usage_sum"] is not None
            and target["transfer_out_prior_usage_sum"] is not None
        ):
            target["transfer_net_prior_usage"] = (
                target["transfer_in_prior_usage_sum"]
                - target["transfer_out_prior_usage_sum"]
            )
        else:
            target["transfer_net_prior_usage"] = None
        base_returning = next(
            (
                float(row["returning_pct_ppa"])
                for row in rows
                if (int(row["season"]), str(row["subdivision"]), str(row["team_id"]))
                == key
                and row.get("returning_pct_ppa") not in (None, "")
            ),
            None,
        )
        incoming_usage = target["transfer_in_prior_usage_sum"]
        target["effective_returning_production"] = (
            base_returning + incoming_usage
            if base_returning is not None and incoming_usage is not None
            else None
        )
    return output


def coverage_rows(
    records: Iterable[TransferRecord],
    *,
    target_seasons: Iterable[int],
    cutoff: date,
    team_rows: Iterable[Mapping[str, str]],
    covered_seasons: set[int],
) -> list[dict[str, object]]:
    """Build a season-level provenance/coverage table before modeling."""
    rows = list(team_rows)
    index = _team_index(rows)
    by_season: defaultdict[int, list[TransferRecord]] = defaultdict(list)
    for record in records:
        by_season[record.season].append(record)
    result = []
    for season in sorted(set(target_seasons)):
        items = by_season[season]
        dated = [item for item in items if item.transfer_date is not None]
        before = [item for item in items if available_by_cutoff(item, cutoff)]
        result.append(
            {
                "season": season,
                "portal_payload_available": season in covered_seasons,
                "n_portal_records": len(items),
                "n_with_destination": sum(
                    item.destination is not None for item in items
                ),
                "n_with_rating": sum(item.rating is not None for item in items),
                "n_with_stars": sum(item.stars is not None for item in items),
                "n_with_position": sum(item.position is not None for item in items),
                "n_with_transfer_date": len(dated),
                "n_on_or_before_cutoff": len(before),
                "destination_rate": (
                    sum(item.destination is not None for item in items) / len(items)
                    if items
                    else None
                ),
                "rating_rate": (
                    sum(item.rating is not None for item in items) / len(items)
                    if items
                    else None
                ),
                "stars_rate": (
                    sum(item.stars is not None for item in items) / len(items)
                    if items
                    else None
                ),
                "date_rate": (len(dated) / len(items) if items else None),
                "cutoff_rate": (len(before) / len(items) if items else None),
                "n_destination_team_matches": sum(
                    _matched_team(index, season, item.destination) is not None
                    for item in before
                ),
                "n_origin_team_matches": sum(
                    _matched_team(index, season, item.origin) is not None
                    for item in before
                ),
                "cutoff": cutoff.isoformat(),
                "classification": "retrospective-research-oracle",
            }
        )
    return result
