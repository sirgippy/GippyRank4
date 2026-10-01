"""Deterministic repairs and coverage inventories for transfer evidence.

The repair helpers consume retained derived player audits. They do not infer
identity from outcomes or fill unresolved player contributions with zero.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from math import fsum
from typing import Any

Row = dict[str, Any]

KNOWN_OFFENSE_CATEGORIES = frozenset(
    {
        "applicable_prior_offensive_usage_successfully_resolved",
        "legitimate_zero_or_non_applicable_prior_offensive_usage",
    }
)
DB_OBSERVED_STATUSES = frozenset(
    {"resolved", "zero_recorded_defensive_box_score_games"}
)


def _unique_index(
    rows: Iterable[Mapping[str, Any]], key_fields: Sequence[str], label: str
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in result:
            raise ValueError(f"duplicate {label} key: {key}")
        result[key] = row
    return result


def repair_verified_historical_zero_aggregates(
    feature_rows: Iterable[Mapping[str, Any]],
    team_coverage_rows: Iterable[Mapping[str, Any]],
    player_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[Row], list[Row]]:
    """Restore null historical sums only when every incoming row is known.

    A covered season with no incoming transfers is a natural zero. For a
    non-empty incoming set, every player must have a numeric value and a
    resolved-usage or legitimate-zero classification. Unknown applicability
    and failed applicable joins leave the aggregate untouched.
    """
    feature_list = [dict(row) for row in feature_rows]
    coverage = _unique_index(
        team_coverage_rows, ("season", "team_id"), "historical transfer coverage"
    )
    grouped_players: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in player_rows:
        if row.get("in_model_relevant_population") == "True":
            grouped_players[
                (str(row["season"]), str(row["destination_team_id"]))
            ].append(row)

    repairs: list[Row] = []
    for feature in feature_list:
        season = str(feature["season"])
        if not 2021 <= int(season) <= 2025 or feature.get(
            "transfer_in_prior_usage_sum"
        ) not in (None, ""):
            continue
        key = (season, str(feature["team_id"]))
        evidence = coverage.get(key)
        if evidence is None or evidence.get("portal_payload_available") != "True":
            continue
        players = sorted(
            grouped_players.get(key, []),
            key=lambda row: int(row.get("portal_index") or 0),
        )
        try:
            incoming_count = int(evidence["incoming_transfer_count"])
        except (KeyError, TypeError, ValueError):
            continue
        if incoming_count != len(players):
            continue
        if any(
            row.get("d5_resolution_category") not in KNOWN_OFFENSE_CATEGORIES
            or row.get("d5_feature_value") in (None, "")
            for row in players
        ):
            continue
        values = [float(row["d5_feature_value"]) for row in players]
        repaired_value = fsum(values)
        feature["transfer_in_prior_usage_sum"] = format(repaired_value, ".15g")
        repairs.append(
            {
                "season": int(season),
                "team_id": str(feature["team_id"]),
                "destination_team": feature.get("team_name")
                or evidence.get("team_name"),
                "player": "; ".join(
                    str(row.get("player_name") or "") for row in players
                ),
                "source_team": "; ".join(
                    str(row.get("origin") or "") for row in players
                ),
                "old_status": "null_historical_aggregate",
                "old_reason": "historical_aggregate_null",
                "new_status": "verified_numeric_aggregate",
                "old_value": "",
                "new_value": format(repaired_value, ".15g"),
                "repair_rule": "sum_complete_player_audit_evidence",
                "source_provenance": (
                    "data/processed/transfer_production_audit/player_join_records.csv; "
                    f"{len(players)} incoming players; "
                    + ",".join(
                        sorted({str(row["d5_resolution_category"]) for row in players})
                    )
                ),
                "incoming_player_count": incoming_count,
            }
        )
    repairs.sort(key=lambda row: (row["season"], row["team_id"]))
    return feature_list, repairs


def build_db_coverage_inventory(
    team_rows: Iterable[Mapping[str, Any]],
    player_rows: Iterable[Mapping[str, Any]],
    *,
    source_season: int,
    provenance: Mapping[str, Any] | None = None,
) -> list[Row]:
    """Build exact incoming-DB ``observed / incoming`` coverage by team."""
    teams = sorted(
        (dict(row) for row in team_rows if str(row.get("season")) == "2026"),
        key=lambda row: str(row["team_id"]),
    )
    team_index = _unique_index(teams, ("season", "team_id"), "2026 team")
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in player_rows:
        if (
            row.get("season") == "2026"
            and row.get("in_model_relevant_population") == "True"
            and row.get("portal_position_group") == "db"
        ):
            grouped[str(row["destination_team_id"])].append(row)

    provenance = provenance or {}
    result: list[Row] = []
    for key in sorted(team_index, key=lambda item: item[1]):
        season, team_id = key
        team = team_index[key]
        incoming = sorted(
            grouped.get(team_id, []),
            key=lambda row: int(row.get("portal_index") or 0),
        )
        observed = [
            row
            for row in incoming
            if row.get("impact_status") in DB_OBSERVED_STATUSES
            and row.get("prior_defensive_impact") not in (None, "")
        ]
        n = len(incoming)
        k = len(observed)
        if n == 0:
            status = "no_incoming_db_transfers"
            fraction = None
        elif k == n:
            status = "complete"
            fraction = 1.0
        elif k == 0:
            status = "no_observed_db_impact"
            fraction = 0.0
        else:
            status = "partial"
            fraction = k / n
        prov_key = f"{season}|{team_id}|transfer_in_prior_defensive_impact_db_sum"
        prov_entry = provenance.get(prov_key, provenance)
        result.append(
            {
                "season": season,
                "source_season": source_season,
                "team_id": team_id,
                "team_name": team.get("team_name"),
                "total_count": n,
                "observed_count": k,
                "missing_count": n - k,
                "coverage_fraction": fraction,
                "observed_db_impact_sum": fsum(
                    float(row["prior_defensive_impact"]) for row in observed
                ),
                "coverage_status": status,
                "observed_over_incoming": f"{k}/{n}",
                "source_snapshot_sha256": ";".join(
                    str(value) for value in prov_entry.get("source_snapshot_sha256", [])
                ),
            }
        )
    return result


__all__ = [
    "build_db_coverage_inventory",
    "repair_verified_historical_zero_aggregates",
]
