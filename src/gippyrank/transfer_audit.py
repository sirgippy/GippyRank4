"""Auditable data-quality checks for transfer-production features.

The transfer oracle in :mod:`gippyrank.transfer_oracle` is deliberately small
and model-facing.  This module is the corresponding data-audit layer for
issue 98.  It never fuzzy-matches a player or team and it never chooses an
ambiguous usage row.  Every row returned by the audit retains the reason a
transfer was or was not usable.

The usage-weighted metric is intentionally named a *recoverable-denominator
proxy*.  A missing player is missing precisely the incoming prior offensive
usage value needed for a full population-weighted denominator.  The proxy
therefore uses unique usage rows that can be linked to at least one in-scope,
D5-applicable transfer, and reports that limitation alongside the number
rather than treating it as full feature coverage.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from math import isfinite, sqrt
from pathlib import Path
from typing import Any

from gippyrank.transfer_oracle import (
    TransferRecord,
    UsageRecord,
    available_by_cutoff,
    normalize_player_name,
    normalize_team_name,
    position_group,
)

Row = dict[str, Any]

D5_APPLICABLE = "applicable_prior_offensive_usage"
D5_NON_APPLICABLE = "legitimate_zero_or_non_applicable"
D5_UNKNOWN = "applicability_unknown"
D5_CATEGORY_RESOLVED = "applicable_prior_offensive_usage_successfully_resolved"
D5_CATEGORY_ZERO = "legitimate_zero_or_non_applicable_prior_offensive_usage"
D5_CATEGORY_FAILURE = "should_have_recoverable_offensive_usage_but_resolution_failed"
D5_CATEGORY_UNDETERMINED = "cannot_determine_applicability"
PARTICIPATION_POSITIVE = "positive_prior_offensive_participation"
PARTICIPATION_ZERO = "no_prior_offensive_participation"
PARTICIPATION_UNKNOWN = "prior_participation_unknown"
OFFENSIVE_STAT_CATEGORIES = {"passing", "rushing", "receiving"}


@dataclass(frozen=True)
class TeamResolution:
    """Deterministic resolution of one source name into the FBS population."""

    season: int
    raw_name: str | None
    normalized_name: str
    team_id: str | None
    canonical_name: str | None
    match_method: str

    @property
    def matched(self) -> bool:
        return self.team_id is not None


@dataclass(frozen=True)
class ParticipationRecord:
    """One raw player-season-stat row used as independent participation evidence."""

    season: int
    player_name: str
    team: str
    position: str | None
    category: str
    stat_type: str
    stat_value: str | None
    player_id: str | None = None


class TeamResolver:
    """Resolve season-specific FBS names using exact names and explicit aliases."""

    def __init__(
        self,
        team_rows: Iterable[Mapping[str, Any]],
        aliases: Mapping[str | tuple[int, str], str] | None = None,
    ) -> None:
        self.rows = [dict(row) for row in team_rows]
        self.index: dict[tuple[int, str], tuple[str, str]] = {}
        for row in self.rows:
            if str(row.get("subdivision", "")).casefold() != "fbs":
                continue
            season = int(row["season"])
            name = str(row.get("team_name") or "").strip()
            if not name:
                continue
            key = (season, normalize_team_name(name))
            value = (str(row["team_id"]), name)
            if key in self.index and self.index[key] != value:
                raise ValueError(f"duplicate canonical FBS team name: {key}")
            self.index[key] = value
        self.aliases = self._normalize_aliases(aliases or {})

    @staticmethod
    def _normalize_aliases(
        aliases: Mapping[str | tuple[int, str], str],
    ) -> dict[str | tuple[int, str], str]:
        normalized: dict[str | tuple[int, str], str] = {}
        for key, value in aliases.items():
            if isinstance(key, tuple):
                normalized[(int(key[0]), normalize_team_name(key[1]))] = (
                    normalize_team_name(value)
                )
            else:
                normalized[normalize_team_name(key)] = normalize_team_name(value)
        return normalized

    def resolve(self, season: int, raw_name: str | None) -> TeamResolution:
        normalized = normalize_team_name(raw_name)
        if not normalized:
            return TeamResolution(
                season,
                raw_name,
                normalized,
                None,
                None,
                "missing_name",
            )
        exact = self.index.get((season, normalized))
        if exact is not None:
            return TeamResolution(
                season,
                raw_name,
                normalized,
                exact[0],
                exact[1],
                "exact_normalized_name",
            )
        alias = self.aliases.get((season, normalized), self.aliases.get(normalized))
        if alias is not None:
            aliased = self.index.get((season, alias))
            if aliased is not None:
                return TeamResolution(
                    season,
                    raw_name,
                    normalized,
                    aliased[0],
                    aliased[1],
                    "explicit_alias",
                )
            return TeamResolution(
                season,
                raw_name,
                normalized,
                None,
                None,
                "alias_target_not_in_fbs_population",
            )
        return TeamResolution(
            season,
            raw_name,
            normalized,
            None,
            None,
            "not_in_fbs_population_or_unrecognized",
        )


def read_team_aliases(path: Path | None) -> dict[str | tuple[int, str], str]:
    """Read an explicit alias file with ``alias,canonical[,season]`` columns."""
    if path is None:
        return {}
    aliases: dict[str | tuple[int, str], str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            alias = (row.get("alias") or row.get("source") or "").strip()
            canonical = (row.get("canonical") or row.get("team_name") or "").strip()
            if not alias or not canonical:
                raise ValueError(f"alias row needs alias and canonical: {row}")
            season = (row.get("season") or "").strip()
            key: str | tuple[int, str] = (int(season), alias) if season else alias
            if key in aliases and aliases[key] != canonical:
                raise ValueError(f"conflicting team alias: {key}")
            aliases[key] = canonical
    return aliases


def parse_participation_payload(
    payload: Iterable[Mapping[str, Any]], *, season: int | None = None
) -> list[ParticipationRecord]:
    """Parse CFBD player-season stats without collapsing raw stat categories."""
    result: list[ParticipationRecord] = []
    for item in payload:
        raw_season = item.get("season")
        record_season = int(raw_season) if raw_season not in (None, "") else season
        if record_season is None:
            raise ValueError("player-stat record has no season")
        if season is not None and record_season != season:
            raise ValueError("player-stat season disagrees with requested season")
        result.append(
            ParticipationRecord(
                season=record_season,
                player_name=str(item.get("player") or item.get("name") or "").strip(),
                team=str(item.get("team") or "").strip(),
                position=(
                    str(item["position"]).strip() if item.get("position") else None
                ),
                category=str(item.get("category") or "").strip().casefold(),
                stat_type=str(item.get("statType") or "").strip().casefold(),
                stat_value=(
                    str(item["stat"]).strip() if item.get("stat") is not None else None
                ),
                player_id=(
                    str(item.get("playerId") or item.get("id")).strip()
                    if item.get("playerId") or item.get("id")
                    else None
                ),
            )
        )
    return result


def _raw_player_name(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.casefold().split())


def _optional_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _usage_indexes(usage: Sequence[UsageRecord]) -> dict[str, Any]:
    by_key: defaultdict[tuple[int, str, str], list[int]] = defaultdict(list)
    by_raw_key: defaultdict[tuple[int, str, str], list[int]] = defaultdict(list)
    by_name: defaultdict[tuple[int, str], list[int]] = defaultdict(list)
    for index, item in enumerate(usage):
        normalized_team = normalize_team_name(item.team)
        normalized_player = normalize_player_name(item.player_name)
        by_key[(item.season, normalized_team, normalized_player)].append(index)
        by_raw_key[
            (item.season, normalized_team, _raw_player_name(item.player_name))
        ].append(index)
        by_name[(item.season, normalized_player)].append(index)
    return {"by_key": by_key, "by_raw_key": by_raw_key, "by_name": by_name}


def _participation_indexes(
    participation: Sequence[ParticipationRecord],
) -> dict[str, Any]:
    by_key: defaultdict[tuple[int, str, str], list[int]] = defaultdict(list)
    by_name: defaultdict[tuple[int, str], list[int]] = defaultdict(list)
    for index, item in enumerate(participation):
        key = (
            item.season,
            normalize_team_name(item.team),
            normalize_player_name(item.player_name),
        )
        by_key[key].append(index)
        by_name[(item.season, normalize_player_name(item.player_name))].append(index)
    return {"by_key": by_key, "by_name": by_name}


def _stat_has_positive_value(value: str | None) -> bool:
    if value in (None, ""):
        return False
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", value.replace(",", ""))
    return any(float(number) > 0 for number in numbers)


def _participation_evidence(
    record: TransferRecord,
    source_resolution: TeamResolution,
    participation: Sequence[ParticipationRecord],
    indexes: Mapping[str, Any],
    participation_seasons: set[int],
) -> tuple[str, str, str, int]:
    """Find independent prior offensive-participation evidence for one transfer."""
    prior_season = record.season - 1
    if prior_season not in participation_seasons:
        return (
            PARTICIPATION_UNKNOWN,
            "prior_participation_source_not_available",
            "none",
            0,
        )
    if not record.origin or not record.player_name:
        return PARTICIPATION_UNKNOWN, "missing_source_or_player_identity", "none", 0
    normalized_player = normalize_player_name(record.player_name)
    candidates: list[int] = []
    method = "none"
    for team_name, candidate_method in _team_name_candidates(record, source_resolution):
        found = indexes["by_key"].get((prior_season, team_name, normalized_player), [])
        if found:
            candidates = list(found)
            method = f"{candidate_method}_participation_stats"
            break
    if not candidates:
        same_name = indexes["by_name"].get((prior_season, normalized_player), [])
        teams = {normalize_team_name(participation[index].team) for index in same_name}
        player_ids = {
            participation[index].player_id
            for index in same_name
            if participation[index].player_id
        }
        if len(teams) == 1 or len(player_ids) == 1:
            candidates = list(same_name)
            method = "unique_player_name_participation_stats"
        elif same_name:
            return (
                PARTICIPATION_UNKNOWN,
                "ambiguous_player_name_participation_stats",
                "none",
                len(same_name),
            )
    if not candidates:
        return (
            PARTICIPATION_ZERO,
            "no_prior_participation_record",
            "none",
            0,
        )
    has_positive_offense = any(
        participation[index].category in OFFENSIVE_STAT_CATEGORIES
        and _stat_has_positive_value(participation[index].stat_value)
        for index in candidates
    )
    if has_positive_offense:
        return (
            PARTICIPATION_POSITIVE,
            "positive_prior_offensive_stat",
            method,
            len(candidates),
        )
    return (
        PARTICIPATION_ZERO,
        "identity_found_without_positive_offensive_stat",
        method,
        len(candidates),
    )


def _unique_ints(values: Iterable[int]) -> list[int]:
    return sorted(set(values))


def _team_name_candidates(
    record: TransferRecord,
    resolution: TeamResolution,
) -> list[tuple[str, str]]:
    """Return raw and explicit-alias source names usable for usage lookup."""
    candidates: list[tuple[str, str]] = []
    raw = normalize_team_name(record.origin)
    if raw:
        candidates.append((raw, "normalized_source_team"))
    if resolution.matched and resolution.canonical_name:
        canonical = normalize_team_name(resolution.canonical_name)
        if canonical and canonical != raw:
            candidates.append((canonical, "explicit_team_alias"))
    return candidates


def _usage_match(
    record: TransferRecord,
    source_resolution: TeamResolution,
    usage: Sequence[UsageRecord],
    indexes: Mapping[str, Any],
) -> tuple[str, str, list[int], bool]:
    """Return status, method, candidate usage rows, and normalization rescue flag."""
    if not record.origin:
        return "missing_origin", "none", [], False
    normalized_player = normalize_player_name(record.player_name)
    raw_player = _raw_player_name(record.player_name)
    raw_candidates: list[int] = []
    alias_candidates: list[int] = []
    for team_name, method in _team_name_candidates(record, source_resolution):
        found = indexes["by_key"].get(
            (record.season - 1, team_name, normalized_player), []
        )
        if method == "normalized_source_team":
            raw_candidates = list(found)
        else:
            alias_candidates = list(found)
    candidates = raw_candidates or alias_candidates
    method = (
        "normalized_name_and_source_team" if raw_candidates else "explicit_team_alias"
    )
    normalization_rescue = False
    if candidates:
        raw_name_candidates = []
        for team_name, _method in _team_name_candidates(record, source_resolution):
            raw_name_candidates.extend(
                indexes["by_raw_key"].get(
                    (record.season - 1, team_name, raw_player), []
                )
            )
        normalization_rescue = not raw_name_candidates
    if len(candidates) > 1:
        return (
            "ambiguous_usage_join",
            method,
            _unique_ints(candidates),
            normalization_rescue,
        )
    if len(candidates) == 1:
        item = usage[candidates[0]]
        if item.overall_usage is None:
            return (
                "usage_record_without_overall_value",
                method,
                candidates,
                normalization_rescue,
            )
        return "joined", method, candidates, normalization_rescue
    same_name = indexes["by_name"].get((record.season - 1, normalized_player), [])
    if same_name:
        return "source_team_mismatch", "none", [], normalization_rescue
    return "no_usage_record", "none", [], normalization_rescue


def _position_applicability(position: str | None) -> tuple[str, str]:
    """Classify whether a portal position can contribute to D5."""
    raw = str(position or "").strip().upper()
    group = position_group(position)
    if group in {"qb", "rb", "wr", "te", "ol"} or raw in {"FB", "H-BACK"}:
        return D5_APPLICABLE, "offensive_portal_position"
    if group in {"dl", "lb", "db", "st"} or raw in {
        "DE",
        "DT",
        "NT",
        "OLB",
        "ILB",
        "MLB",
        "SS",
        "FS",
        "KR",
        "PR",
    }:
        return D5_NON_APPLICABLE, "defensive_or_special_portal_position"
    return D5_UNKNOWN, "missing_or_ambiguous_portal_position"


def _usage_position_applicability(
    usage: Sequence[UsageRecord], usage_indexes: Sequence[int]
) -> tuple[str, str]:
    kinds = [
        _position_applicability(usage[index].position)[0] for index in usage_indexes
    ]
    if kinds and all(kind == D5_APPLICABLE for kind in kinds):
        return D5_APPLICABLE, "offensive_usage_position"
    if kinds and all(kind == D5_NON_APPLICABLE for kind in kinds):
        return D5_NON_APPLICABLE, "defensive_or_special_usage_position"
    return D5_UNKNOWN, "missing_or_mixed_usage_position"


def _classify_d5(
    record: TransferRecord,
    usage: Sequence[UsageRecord],
    usage_indexes: Sequence[int],
    usage_join_status: str,
    participation_status: str,
    participation_reason: str,
) -> tuple[str, str, str, float | None]:
    """Classify a transfer against D5's offensive-usage semantic requirement."""
    portal_kind, portal_reason = _position_applicability(record.position)
    usage_kind, usage_reason = _usage_position_applicability(usage, usage_indexes)
    prior_usage = (
        usage[usage_indexes[0]].overall_usage
        if usage_join_status == "joined" and len(usage_indexes) == 1
        else None
    )

    if usage_join_status == "joined" and prior_usage is not None:
        if prior_usage > 0:
            return (
                D5_APPLICABLE,
                "positive_numeric_usage_resolved",
                D5_CATEGORY_RESOLVED,
                prior_usage,
            )
        return (
            portal_kind if portal_kind != D5_UNKNOWN else usage_kind,
            "numeric_zero_usage",
            D5_CATEGORY_ZERO,
            0.0,
        )

    if participation_status == PARTICIPATION_POSITIVE:
        return (
            D5_APPLICABLE,
            f"{portal_reason}; {participation_reason}; {usage_join_status}",
            D5_CATEGORY_FAILURE,
            None,
        )
    if participation_status == PARTICIPATION_ZERO:
        return (
            D5_NON_APPLICABLE,
            participation_reason,
            D5_CATEGORY_ZERO,
            0.0,
        )
    if portal_kind == D5_NON_APPLICABLE:
        return D5_NON_APPLICABLE, portal_reason, D5_CATEGORY_ZERO, 0.0
    if portal_kind == D5_UNKNOWN and usage_kind != D5_UNKNOWN:
        portal_kind, portal_reason = usage_kind, usage_reason
    if portal_kind == D5_UNKNOWN:
        return D5_UNKNOWN, portal_reason, D5_CATEGORY_UNDETERMINED, None
    return (
        D5_UNKNOWN,
        f"{portal_reason}; {participation_reason}",
        D5_CATEGORY_UNDETERMINED,
        None,
    )


def _priority_key(row: Row) -> tuple[Any, ...]:
    rating = _optional_number(row.get("rating"))
    stars = _optional_number(row.get("stars"))
    return (
        rating is None,
        -(rating if rating is not None else -1.0),
        stars is None,
        -(stars if stars is not None else -1.0),
        not bool(row.get("is_qb")),
        str(row.get("player_name") or "").casefold(),
    )


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_ss = sum((x - x_mean) ** 2 for x in xs)
    y_ss = sum((y - y_mean) ** 2 for y in ys)
    if x_ss <= 0 or y_ss <= 0:
        return None
    return numerator / sqrt(x_ss * y_ss)


def audit_transfer_records(
    records: Iterable[TransferRecord],
    usage: Iterable[UsageRecord],
    team_rows: Iterable[Mapping[str, Any]],
    *,
    cutoff: date,
    aliases: Mapping[str | tuple[int, str], str] | None = None,
    participation: Iterable[ParticipationRecord] | None = None,
) -> dict[str, Any]:
    """Audit player joins, team mappings, and usage-weighted coverage.

    ``records`` and ``usage`` are already parsed from immutable raw payloads.
    A record is in the model-relevant population when its destination resolves
    to an FBS team and its transfer date is on or before the configured
    season-relative cutoff.  This deliberately includes transfers from
    outside the FBS population; their source-side usage is audited separately.
    """
    record_list = list(records)
    usage_list = list(usage)
    participation_list = list(participation or ())
    team_list = [dict(row) for row in team_rows]
    resolver = TeamResolver(team_list, aliases)
    indexes = _usage_indexes(usage_list)
    participation_indexes = _participation_indexes(participation_list)
    participation_seasons = {item.season for item in participation_list}
    by_portal_key: Counter[tuple[int, str, str]] = Counter(
        (
            item.season,
            normalize_team_name(item.origin),
            normalize_player_name(item.player_name),
        )
        for item in record_list
    )

    join_rows: list[Row] = []
    mapping_inputs: dict[tuple[int, str, str], Row] = {}
    usage_to_portal: defaultdict[int, list[int]] = defaultdict(list)
    successful_usage_to_portal: defaultdict[int, list[int]] = defaultdict(list)
    any_name_usage_to_portal: defaultdict[int, list[int]] = defaultdict(list)
    for portal_index, record in enumerate(record_list):
        source = resolver.resolve(record.season, record.origin)
        destination = resolver.resolve(record.season, record.destination)
        cutoff_state = (
            "on_or_before_cutoff"
            if available_by_cutoff(record, cutoff)
            else "missing_transfer_date_or_after_cutoff"
        )
        in_scope = cutoff_state == "on_or_before_cutoff" and destination.matched
        if in_scope:
            status, join_method, usage_indexes, normalization_rescue = _usage_match(
                record, source, usage_list, indexes
            )
        else:
            status, join_method, usage_indexes, normalization_rescue = (
                "not_in_model_relevant_population",
                "none",
                [],
                False,
            )
        (
            participation_status,
            participation_reason,
            participation_method,
            participation_record_count,
        ) = _participation_evidence(
            record,
            source,
            participation_list,
            participation_indexes,
            participation_seasons,
        )
        prior_usage = (
            usage_list[usage_indexes[0]].overall_usage
            if status == "joined" and len(usage_indexes) == 1
            else None
        )
        (
            d5_applicability,
            d5_applicability_reason,
            d5_category,
            d5_feature_value,
        ) = _classify_d5(
            record,
            usage_list,
            usage_indexes,
            status,
            participation_status,
            participation_reason,
        )
        row: Row = {
            "portal_index": portal_index,
            "season": record.season,
            "player_name": record.player_name,
            "normalized_player_name": normalize_player_name(record.player_name),
            "origin": record.origin,
            "origin_team_id": source.team_id,
            "origin_team_name": source.canonical_name,
            "origin_match_method": source.match_method,
            "destination": record.destination,
            "destination_team_id": destination.team_id,
            "destination_team_name": destination.canonical_name,
            "destination_match_method": destination.match_method,
            "position": record.position,
            "position_group": position_group(record.position),
            "transfer_date": record.transfer_date.isoformat()
            if record.transfer_date
            else None,
            "cutoff_state": cutoff_state,
            "in_model_relevant_population": in_scope,
            "rating": record.rating,
            "stars": record.stars,
            "is_qb": position_group(record.position) == "qb",
            "usage_join_status": status,
            "usage_join_method": join_method,
            "normalization_changed_match": normalization_rescue,
            "prior_participation_status": participation_status,
            "prior_participation_reason": participation_reason,
            "prior_participation_method": participation_method,
            "prior_participation_record_count": participation_record_count,
            "prior_usage": prior_usage,
            "incoming_prior_offensive_usage": prior_usage,
            "d5_applicability": d5_applicability,
            "d5_applicability_reason": d5_applicability_reason,
            "d5_resolution_category": d5_category,
            "d5_feature_value": d5_feature_value,
            "d5_unresolved": d5_category
            in {D5_CATEGORY_FAILURE, D5_CATEGORY_UNDETERMINED},
            "prior_usage_record_count": len(usage_indexes),
            "portal_normalized_key_count": by_portal_key[
                (
                    record.season,
                    normalize_team_name(record.origin),
                    normalize_player_name(record.player_name),
                )
            ],
        }
        join_rows.append(row)
        for field, resolution in (("origin", source), ("destination", destination)):
            if not resolution.raw_name:
                continue
            key = (record.season, field, normalize_team_name(resolution.raw_name))
            item = mapping_inputs.setdefault(
                key,
                {
                    "season": record.season,
                    "field": field,
                    "raw_name": resolution.raw_name,
                    "normalized_name": resolution.normalized_name,
                    "team_id": resolution.team_id,
                    "canonical_name": resolution.canonical_name,
                    "match_method": resolution.match_method,
                    "record_count": 0,
                },
            )
            item["record_count"] += 1
        if in_scope:
            for usage_index in indexes["by_name"].get(
                (record.season - 1, normalize_player_name(record.player_name)), []
            ):
                any_name_usage_to_portal[usage_index].append(portal_index)
        if in_scope and status in {"joined", "ambiguous_usage_join"}:
            for usage_index in usage_indexes:
                usage_to_portal[usage_index].append(portal_index)
        if in_scope and status == "joined":
            for usage_index in usage_indexes:
                successful_usage_to_portal[usage_index].append(portal_index)

    season_rows = _season_join_rows(
        join_rows,
        usage_list,
        usage_to_portal,
        successful_usage_to_portal,
        any_name_usage_to_portal,
    )
    team_rows_out = build_team_feature_coverage(
        join_rows,
        usage_list,
        team_list,
        covered_seasons={item.season for item in record_list},
    )
    missingness = _missingness_summary(team_rows_out)
    unmatched = [
        row
        for row in join_rows
        if row["in_model_relevant_population"] and row["d5_unresolved"]
    ]
    unmatched.sort(key=_priority_key)
    for rank, row in enumerate(unmatched, start=1):
        row["priority_rank"] = rank

    stable_id = _stable_id_assessment(record_list, usage_list)
    return {
        "season_rows": season_rows,
        "join_rows": join_rows,
        "unmatched_rows": unmatched,
        "team_mapping_rows": sorted(
            mapping_inputs.values(),
            key=lambda row: (
                int(row["season"]),
                str(row["field"]),
                str(row["raw_name"]),
            ),
        ),
        "team_feature_rows": team_rows_out,
        "missingness": missingness,
        "stable_player_id": stable_id,
        "portal_normalized_name_collisions": _collision_summary(
            record_list,
            key=lambda item: (
                item.season,
                normalize_team_name(item.origin),
                normalize_player_name(item.player_name),
            ),
        ),
        "usage_normalized_name_collisions": _collision_summary(
            usage_list,
            key=lambda item: (
                item.season,
                normalize_team_name(item.team),
                normalize_player_name(item.player_name),
            ),
        ),
        "participation_normalized_name_collisions": _collision_summary(
            participation_list,
            key=lambda item: (
                item.season,
                normalize_team_name(item.team),
                normalize_player_name(item.player_name),
            ),
        ),
    }


def _collision_summary(items: Sequence[Any], *, key: Any) -> dict[str, Any]:
    groups: defaultdict[Any, list[Any]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    collisions = [values for values in groups.values() if len(values) > 1]
    return {
        "collision_key_count": len(collisions),
        "records_in_collision_keys": sum(len(values) for values in collisions),
        "max_records_per_key": max((len(values) for values in collisions), default=0),
    }


def _season_join_rows(
    join_rows: Sequence[Row],
    usage: Sequence[UsageRecord],
    usage_to_portal: Mapping[int, Sequence[int]],
    successful_usage_to_portal: Mapping[int, Sequence[int]],
    any_name_usage_to_portal: Mapping[int, Sequence[int]],
) -> list[Row]:
    result: list[Row] = []
    seasons = sorted({int(row["season"]) for row in join_rows})
    for season in seasons:
        rows = [row for row in join_rows if int(row["season"]) == season]
        relevant = [row for row in rows if row["in_model_relevant_population"]]
        row_by_portal_index = {int(row["portal_index"]): row for row in relevant}

        def d5_portal_indexes(
            portal_indexes: Sequence[int],
            row_by_portal_index: Mapping[int, Row] = row_by_portal_index,
        ) -> list[int]:
            return [
                portal_index
                for portal_index in portal_indexes
                if row_by_portal_index.get(portal_index, {}).get("d5_applicability")
                == D5_APPLICABLE
            ]

        portal_collision_counts = Counter(
            (
                normalize_team_name(str(row["origin"] or "")),
                str(row["normalized_player_name"]),
            )
            for row in relevant
        )
        usage_collision_counts = Counter(
            (
                normalize_team_name(item.team),
                normalize_player_name(item.player_name),
            )
            for item in usage
            if item.season == season - 1
        )
        source_usage_mass = sum(
            usage[index].overall_usage or 0.0
            for index, portal_indexes in usage_to_portal.items()
            if d5_portal_indexes(portal_indexes) and usage[index].season == season - 1
        )
        unique_usage_mass = sum(
            usage[index].overall_usage or 0.0
            for index, portal_indexes in successful_usage_to_portal.items()
            if d5_portal_indexes(portal_indexes)
            and usage[index].season == season - 1
            and len(set(d5_portal_indexes(portal_indexes))) == 1
        )
        any_name_usage_mass = sum(
            usage[index].overall_usage or 0.0
            for index, portal_indexes in any_name_usage_to_portal.items()
            if d5_portal_indexes(portal_indexes) and usage[index].season == season - 1
        )
        category_counts = Counter(row["d5_resolution_category"] for row in relevant)
        result.append(
            {
                "season": season,
                "total_portal_records": len(rows),
                "records_with_destination": sum(
                    bool(row["destination"]) for row in rows
                ),
                "records_with_transfer_date": sum(
                    row["transfer_date"] is not None for row in rows
                ),
                "records_on_or_before_cutoff": sum(
                    row["cutoff_state"] == "on_or_before_cutoff" for row in rows
                ),
                "incoming_fbs_transfers": len(relevant),
                "incoming_from_fbs_source": sum(
                    bool(row["origin_team_id"]) for row in relevant
                ),
                "incoming_from_non_fbs_or_unrecognized_source": sum(
                    not bool(row["origin_team_id"]) for row in relevant
                ),
                "d5_applicable_transfers": sum(
                    row["d5_applicability"] == D5_APPLICABLE for row in relevant
                ),
                "d5_successfully_resolved": category_counts[D5_CATEGORY_RESOLVED],
                "d5_legitimate_zero_or_non_applicable": category_counts[
                    D5_CATEGORY_ZERO
                ],
                "d5_resolution_failures": category_counts[D5_CATEGORY_FAILURE],
                "d5_applicability_unknown": category_counts[D5_CATEGORY_UNDETERMINED],
                "prior_participation_positive": sum(
                    row["prior_participation_status"] == PARTICIPATION_POSITIVE
                    for row in relevant
                ),
                "prior_participation_zero": sum(
                    row["prior_participation_status"] == PARTICIPATION_ZERO
                    for row in relevant
                ),
                "prior_participation_unknown": sum(
                    row["prior_participation_status"] == PARTICIPATION_UNKNOWN
                    for row in relevant
                ),
                "d5_resolved_rate_among_determined": (
                    (
                        category_counts[D5_CATEGORY_RESOLVED]
                        + category_counts[D5_CATEGORY_ZERO]
                    )
                    / (
                        category_counts[D5_CATEGORY_RESOLVED]
                        + category_counts[D5_CATEGORY_ZERO]
                        + category_counts[D5_CATEGORY_FAILURE]
                    )
                    if category_counts[D5_CATEGORY_RESOLVED]
                    + category_counts[D5_CATEGORY_ZERO]
                    + category_counts[D5_CATEGORY_FAILURE]
                    else None
                ),
                "d5_resolution_rate_among_applicable": (
                    category_counts[D5_CATEGORY_RESOLVED]
                    / (
                        category_counts[D5_CATEGORY_RESOLVED]
                        + category_counts[D5_CATEGORY_FAILURE]
                    )
                    if category_counts[D5_CATEGORY_RESOLVED]
                    + category_counts[D5_CATEGORY_FAILURE]
                    else None
                ),
                "incoming_fbs_with_exact_successful_join": sum(
                    row["usage_join_status"] == "joined"
                    and row["usage_join_method"] == "normalized_name_and_source_team"
                    for row in relevant
                ),
                "incoming_fbs_with_alias_successful_join": sum(
                    row["usage_join_status"] == "joined"
                    and row["usage_join_method"] == "explicit_team_alias"
                    for row in relevant
                ),
                "failed_prior_usage_joins": sum(
                    row["usage_join_status"] != "joined" for row in relevant
                ),
                "ambiguous_joins": sum(
                    row["usage_join_status"] == "ambiguous_usage_join"
                    for row in relevant
                ),
                "usage_records_without_value": sum(
                    row["usage_join_status"] == "usage_record_without_overall_value"
                    for row in relevant
                ),
                "source_team_mismatches": sum(
                    row["usage_join_status"] == "source_team_mismatch"
                    for row in relevant
                ),
                "missing_usage_records": sum(
                    row["usage_join_status"] == "no_usage_record" for row in relevant
                ),
                "normalization_changed_matches": sum(
                    row["normalization_changed_match"] for row in relevant
                ),
                "portal_normalized_name_collision_keys": sum(
                    count > 1 for count in portal_collision_counts.values()
                ),
                "portal_records_in_normalized_name_collisions": sum(
                    count for count in portal_collision_counts.values() if count > 1
                ),
                "usage_normalized_name_collision_keys": sum(
                    count > 1 for count in usage_collision_counts.values()
                ),
                "destination_alias_matches": sum(
                    row["destination_match_method"] == "explicit_alias" for row in rows
                ),
                "origin_alias_matches": sum(
                    row["origin_match_method"] == "explicit_alias" for row in rows
                ),
                "recoverable_source_usage_mass": source_usage_mass,
                "recoverable_unique_usage_mass": unique_usage_mass,
                "any_name_usage_mass": any_name_usage_mass,
                "usage_weighted_join_coverage_proxy": (
                    unique_usage_mass / any_name_usage_mass
                    if any_name_usage_mass > 0
                    else None
                ),
                "usage_weighted_denominator_definition": (
                    "unique numeric usage records whose normalized player name appears "
                    "in at least one D5-applicable transfer; the numerator additionally "
                    "requires a unique source-team/player join. This is a recoverable "
                    "D5 identity-resolution proxy, not overall D5 feature coverage or "
                    "a full-population denominator."
                ),
            }
        )
    return result


def build_team_feature_coverage(
    join_rows: Sequence[Row],
    usage: Sequence[UsageRecord],
    team_rows: Iterable[Mapping[str, Any]],
    *,
    covered_seasons: set[int],
) -> list[Row]:
    """Build team-season completeness and unresolved offensive-usage bounds."""
    rows = [
        dict(row)
        for row in team_rows
        if str(row.get("subdivision", "")).casefold() == "fbs"
    ]
    max_usage: defaultdict[int, float] = defaultdict(float)
    for item in usage:
        if item.overall_usage is not None:
            max_usage[item.season] = max(max_usage[item.season], item.overall_usage)
    result: list[Row] = []
    for team in rows:
        season = int(team["season"])
        team_id = str(team["team_id"])
        incoming = [
            row
            for row in join_rows
            if row["in_model_relevant_population"]
            and int(row["season"]) == season
            and row["destination_team_id"] == team_id
        ]
        category_counts = Counter(row["d5_resolution_category"] for row in incoming)
        resolved = [
            row
            for row in incoming
            if row["d5_resolution_category"] in {D5_CATEGORY_RESOLVED, D5_CATEGORY_ZERO}
        ]
        resolution_failures = category_counts[D5_CATEGORY_FAILURE]
        applicability_unknown = category_counts[D5_CATEGORY_UNDETERMINED]
        if season not in covered_seasons:
            status = "portal_payload_missing"
        elif not incoming:
            status = "no_incoming_transfer"
        elif not resolution_failures and not applicability_unknown:
            status = "complete"
        elif resolved:
            status = "partial"
        elif resolution_failures and not applicability_unknown:
            status = "unresolved_applicable_offensive_usage"
        elif applicability_unknown and not resolution_failures:
            status = "undetermined_applicability"
        else:
            status = "unresolved_and_undetermined"
        observed = sum(
            float(row["d5_feature_value"])
            for row in resolved
            if row["d5_feature_value"] is not None
        )
        missing_upper_bound = resolution_failures * max_usage.get(season - 1, 1.0)
        result.append(
            {
                "season": season,
                "subdivision": team.get("subdivision"),
                "team_id": team_id,
                "team_name": team.get("team_name"),
                "portal_payload_available": season in covered_seasons,
                "incoming_transfer_count": len(incoming),
                "d5_applicable_transfer_count": sum(
                    row["d5_applicability"] == D5_APPLICABLE for row in incoming
                ),
                "d5_successfully_resolved_count": category_counts[D5_CATEGORY_RESOLVED],
                "d5_legitimate_zero_or_non_applicable_count": category_counts[
                    D5_CATEGORY_ZERO
                ],
                "d5_resolution_failure_count": resolution_failures,
                "d5_applicability_unknown_count": applicability_unknown,
                "joined_prior_usage_count": category_counts[D5_CATEGORY_RESOLVED],
                "unmatched_incoming_count": resolution_failures,
                "undetermined_incoming_count": applicability_unknown,
                "observed_incoming_prior_offensive_usage": observed,
                "observed_incoming_prior_usage": observed,
                "unresolved_prior_offensive_usage_upper_bound": (
                    observed + missing_upper_bound
                )
                if season in covered_seasons
                else None,
                "unresolved_offensive_usage_missing_upper_bound": missing_upper_bound
                if season in covered_seasons
                else None,
                "unmatched_prior_usage_upper_bound": missing_upper_bound
                if season in covered_seasons
                else None,
                "plausible_prior_offensive_usage_upper_bound_basis": (
                    "observed incoming prior offensive usage + resolution-failure "
                    f"count × max overall usage in {season - 1} usage payload"
                ),
                "feature_coverage_status": status,
                "missingness_flag": int(
                    resolution_failures > 0 or applicability_unknown > 0
                ),
                "roster_strength_proxy": _optional_number(team.get("talent_composite")),
                "conference": team.get("conference"),
            }
        )
    return sorted(result, key=lambda row: (int(row["season"]), str(row["team_id"])))


def _missingness_summary(team_rows: Sequence[Row]) -> dict[str, Any]:
    by_season: list[Row] = []
    for season in sorted({int(row["season"]) for row in team_rows}):
        rows = [
            row
            for row in team_rows
            if int(row["season"]) == season
            and row["feature_coverage_status"] != "portal_payload_missing"
        ]
        flags = [float(row["missingness_flag"]) for row in rows]
        volumes = [float(row["d5_applicable_transfer_count"]) for row in rows]
        strengths = [
            float(row["roster_strength_proxy"])
            for row in rows
            if row["roster_strength_proxy"] is not None
        ]
        strength_flags = [
            float(row["missingness_flag"])
            for row in rows
            if row["roster_strength_proxy"] is not None
        ]
        by_season.append(
            {
                "season": season,
                "teams_audited": len(rows),
                "complete_teams": sum(
                    row["feature_coverage_status"] == "complete" for row in rows
                ),
                "partial_teams": sum(
                    row["feature_coverage_status"] == "partial" for row in rows
                ),
                "unresolved_applicable_teams": sum(
                    row["feature_coverage_status"]
                    == "unresolved_applicable_offensive_usage"
                    for row in rows
                ),
                "undetermined_applicability_teams": sum(
                    row["feature_coverage_status"] == "undetermined_applicability"
                    for row in rows
                ),
                "unresolved_and_undetermined_teams": sum(
                    row["feature_coverage_status"] == "unresolved_and_undetermined"
                    for row in rows
                ),
                "no_usable_teams": sum(
                    row["feature_coverage_status"]
                    in {
                        "unresolved_applicable_offensive_usage",
                        "undetermined_applicability",
                        "unresolved_and_undetermined",
                    }
                    for row in rows
                ),
                "no_incoming_transfer_teams": sum(
                    row["feature_coverage_status"] == "no_incoming_transfer"
                    for row in rows
                ),
                "mean_incoming_transfers": (
                    sum(volumes) / len(volumes) if volumes else None
                ),
                "mean_applicable_incoming_transfers": (
                    sum(volumes) / len(volumes) if volumes else None
                ),
                "median_incoming_transfers": _median(volumes),
                "mean_joined_prior_usage": _mean(
                    [float(row["observed_incoming_prior_usage"]) for row in rows]
                ),
                "mean_joined_prior_offensive_usage": _mean(
                    [
                        float(row["observed_incoming_prior_offensive_usage"])
                        for row in rows
                    ]
                ),
                "median_joined_prior_usage": _median(
                    [float(row["observed_incoming_prior_usage"]) for row in rows]
                ),
                "missingness_rate": _mean(flags),
                "missingness_vs_transfer_volume_pearson": _pearson(flags, volumes),
                "missingness_vs_roster_strength_pearson": _pearson(
                    strength_flags, strengths
                ),
                "conference_analysis": (
                    "available from input rows"
                    if any(row.get("conference") for row in rows)
                    else "not available in the canonical team-season feature table"
                ),
            }
        )
    return {
        "by_season": by_season,
        "conference_analysis": (
            "not available in the canonical team-season feature table; add an explicit "
            "season-specific conference mapping before using conference-stratified rates"
        ),
        "competition_level_analysis": "FBS destination population only; source level is retained as a join failure dimension",
        "transfer_volume_analysis": "Pearson correlation is descriptive and computed by season",
        "roster_strength_analysis": "talent_composite is used only as a roster-strength proxy when present",
    }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _stable_id_assessment(
    records: Sequence[TransferRecord], usage: Sequence[UsageRecord]
) -> Row:
    usage_ids = [item.player_id for item in usage if item.player_id]
    return {
        "portal_records": len(records),
        "portal_records_with_stable_id": 0,
        "usage_records": len(usage),
        "usage_records_with_stable_id": len(usage_ids),
        "unique_usage_stable_ids": len(set(usage_ids)),
        "cross_endpoint_stable_id_available": False,
        "recommendation": (
            "retain an explicit normalized player-name + source-team join, use a "
            "stable ID only if a future portal snapshot exposes the same identifier, "
            "and fail closed on ambiguity"
        ),
        "limitation": "CFBD /player/portal payloads contain no player ID in the audited seasons; /player/usage IDs cannot be linked cross-endpoint",
    }


def cutoff_safety_assessment() -> list[Row]:
    """Return the fixed field-level cutoff classification used by the report."""
    return [
        {
            "field": "transfer season",
            "required_for_model": True,
            "classification": "likely cutoff-safe but not provable",
            "evidence": "The request selects an explicit season, but the source response is not an archived preseason snapshot.",
        },
        {
            "field": "player identity",
            "required_for_model": True,
            "classification": "retrospective oracle only",
            "evidence": "Portal identity is a name assembled from the current endpoint; no shared stable portal/usage ID is available.",
        },
        {
            "field": "source school",
            "required_for_model": True,
            "classification": "retrospective oracle only",
            "evidence": "Origin is read from the current portal record and is not an archived roster state.",
        },
        {
            "field": "destination school",
            "required_for_model": True,
            "classification": "retrospective oracle only",
            "evidence": "A final destination may have been resolved or revised after the preseason cutoff.",
        },
        {
            "field": "transfer date",
            "required_for_model": True,
            "classification": "retrospective oracle only",
            "evidence": "The date supports deterministic filtering, but publication and revision timing are not archived.",
        },
        {
            "field": "incoming prior offensive usage",
            "required_for_model": True,
            "classification": "retrospective oracle only",
            "evidence": "Usage is a prior-season outcome, but the cross-endpoint name join is not an archived transfer roster join.",
        },
        {
            "field": "prior-season offensive participation",
            "required_for_model": True,
            "classification": "retrospective oracle only",
            "evidence": "Independent player-season stats are used to distinguish positive prior participation, legitimate zero, and undetermined applicability before classifying a missing usage row.",
        },
        {
            "field": "position",
            "required_for_model": False,
            "classification": "retrospective oracle only",
            "evidence": "Position is supplied by the current portal response and is not timestamped as-of the cutoff.",
        },
        {
            "field": "rating / stars",
            "required_for_model": False,
            "classification": "retrospective oracle only",
            "evidence": "The response does not provide an archived rating revision history.",
        },
        {
            "field": "scholarship status",
            "required_for_model": False,
            "classification": "unavailable",
            "evidence": "Eligibility is not a scholarship indicator in the selected source fields.",
        },
    ]


def source_inventory() -> list[Row]:
    """Describe viable sources and the minimum production snapshot process."""
    return [
        {
            "source": "CFBD /player/portal",
            "fields": "season, firstName, lastName, origin, destination, position, transferDate, rating, stars, eligibility",
            "player_identifier": "none in audited portal payload",
            "historical_depth": "2021–2025 fetched for this audit",
            "timestamp_semantics": "current retrospective response; event date is not an as-of publication timestamp",
            "reproducibility": "raw response bytes plus retrieval sidecar hash",
            "production_assessment": "usable for future snapshots only if fetched and frozen before the cutoff",
        },
        {
            "source": "CFBD /player/usage",
            "fields": "season, id, name, team, position, conference, usage.overall and component usage",
            "player_identifier": "stable within usage endpoint, not shared by portal endpoint",
            "historical_depth": "2020–2024 fetched for this audit",
            "timestamp_semantics": "season-wide prior offensive usage; not a transfer-time roster snapshot",
            "reproducibility": "raw response bytes plus retrieval sidecar hash",
            "production_assessment": "supporting incoming prior offensive usage source after deterministic identity resolution",
        },
        {
            "source": "CFBD /stats/player/season",
            "fields": "season, playerId, player, team, position, category, statType, stat",
            "player_identifier": "stable within stats endpoint, not shared by portal endpoint",
            "historical_depth": "2020–2024 fetched for this audit",
            "timestamp_semantics": "season-wide prior player statistics; not a transfer-time roster snapshot",
            "reproducibility": "raw response bytes plus retrieval sidecar hash",
            "production_assessment": "independent prior-participation evidence used before calling a missing offensive usage row a D5 failure",
        },
        {
            "source": "repository-managed preseason snapshot process",
            "fields": "raw portal, usage, and player-stats payloads, endpoint, parameters, retrieval timestamp, SHA-256",
            "player_identifier": "portal ID remains unavailable; exact name + source-team fallback is required",
            "historical_depth": "future seasons from process start; historical reconstruction remains retrospective",
            "timestamp_semantics": "retrieval timestamp proves when GippyRank captured the response, not when CFBD first knew a destination",
            "reproducibility": "immutable ignored raw payloads and committed derived audit artifacts",
            "production_assessment": "recommended minimal path for future production feasibility",
        },
    ]


def snapshot_strategy() -> Row:
    return {
        "steps": [
            "On or before the configured preseason cutoff, fetch each required portal, prior-usage, and prior-player-stats season response.",
            "Store response bytes unchanged under data/raw/cfbd/preseason/transfers/{portal,usage,stats}/.",
            "Write endpoint, query parameters, retrieval timestamp, record count, and SHA-256 in a sidecar.",
            "Never overwrite a prior season snapshot; refresh only into a new explicitly named snapshot when source semantics require it.",
            "Run this audit and derive transfer features only from the frozen snapshot, with explicit aliases and fail-closed ambiguous joins.",
        ],
        "integration": "Add the snapshot acquisition as a prerequisite to the existing preseason pipeline; do not let live portal endpoints enter model fitting directly.",
    }
