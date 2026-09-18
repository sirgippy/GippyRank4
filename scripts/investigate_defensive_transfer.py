"""Construct and audit defensive-transfer candidates for issue 103.

The study stops at feature construction and data quality.  It does not fit a
Context variant, compare NLL/CRPS, or inspect target-season outcomes.  Raw
portal, roster, and game-player payloads are read unchanged; all outputs are
derived audit artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gippyrank.defensive_transfer import (
    ALL_STAT_FIELDS,
    DefensivePlayerSeason,
    RosterPlayer,
    add_defensive_impact,
    aggregate_player_seasons,
    audit_transfer_records,
    correlation_rows,
    cutoff_safety,
    descriptive_statistics,
    field_inventory,
    parse_games_players_payload,
    parse_roster_payload,
    position_mapping,
    source_inventory,
    team_game_keys,
)
from gippyrank.transfer_oracle import TransferRecord, parse_transfer_payload

DEFAULT_PORTAL_ROOT = ROOT / "data/raw/cfbd/preseason/transfers"
DEFAULT_DEFENSIVE_ROOT = ROOT / "data/raw/cfbd/preseason/defensive_transfers_complete"
DEFAULT_TEAM_FILE = ROOT / "data/processed/preseason/team_season_features.csv"
DEFAULT_OUTPUT = ROOT / "data/processed/defensive_transfer_audit"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_team_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_aliases(path: Path | None) -> dict[str | tuple[int, str], str]:
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


def _numeric_json_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*.json")
        if path.stem.isdigit() and not path.name.endswith(".provenance.json")
    )


def _game_json_paths(directory: Path) -> list[tuple[int, Path]]:
    result: list[tuple[int, Path]] = []
    for path in directory.rglob("*.json"):
        if path.name.endswith(".provenance.json"):
            continue
        match = re.fullmatch(r"(\d{4})(?:-[a-z]+)?-week-\d{2}", path.stem)
        if match:
            result.append((int(match.group(1)), path))
    return sorted(result)


def _load_inputs(
    portal_root: Path,
    defensive_root: Path,
) -> tuple[
    list[TransferRecord],
    list[RosterPlayer],
    list[DefensivePlayerSeason],
    set[tuple[int, str]],
    set[tuple[int, str]],
    set[int],
    set[int],
    list[dict[str, Any]],
]:
    portal_paths = _numeric_json_paths(portal_root / "portal")
    if not portal_paths:
        raise FileNotFoundError(
            f"no portal JSON files found under {portal_root / 'portal'}"
        )
    roster_paths = _numeric_json_paths(defensive_root / "roster")
    game_paths = _game_json_paths(defensive_root / "games_players")
    if not roster_paths:
        raise FileNotFoundError(
            f"no roster JSON files found under {defensive_root / 'roster'}"
        )
    if not game_paths:
        raise FileNotFoundError(
            f"no games/players JSON files found under {defensive_root / 'games_players'}"
        )

    records: list[TransferRecord] = []
    rosters: list[RosterPlayer] = []
    game_players = []
    game_keys: set[tuple[int, str, str]] = set()
    source_files: list[dict[str, Any]] = []
    portal_seasons: set[int] = set()
    defensive_seasons: set[int] = set()

    for path in portal_paths:
        season = int(path.stem)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"{path} is not a JSON array")
        records.extend(parse_transfer_payload(payload, season=season))
        portal_seasons.add(season)
        source_files.append(_source_file(path, "portal", season, len(payload)))
    for path in roster_paths:
        season = int(path.stem)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"{path} is not a JSON array")
        rosters.extend(parse_roster_payload(payload, season=season))
        defensive_seasons.add(season)
        source_files.append(_source_file(path, "roster", season, len(payload)))
    for season, path in game_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"{path} is not a JSON array")
        game_players.extend(parse_games_players_payload(payload, season=season))
        game_keys.update(team_game_keys(payload, season=season))
        defensive_seasons.add(season)
        source_files.append(_source_file(path, "games_players", season, len(payload)))

    player_seasons = add_defensive_impact(
        aggregate_player_seasons(game_players, rosters, game_keys)
    )
    team_coverage = {(season, team) for season, team, _game in game_keys}
    roster_teams = {(item.season, item.normalized_team) for item in rosters}
    return (
        records,
        rosters,
        player_seasons,
        team_coverage,
        roster_teams,
        portal_seasons,
        defensive_seasons,
        source_files,
    )


def _source_file(
    path: Path, kind: str, season: int, record_count: int
) -> dict[str, Any]:
    sidecar = path.with_name(f"{path.name}.provenance.json")
    return {
        "kind": kind,
        "season": season,
        "path": _display_path(path),
        "sha256": _sha256(path),
        "record_count": record_count,
        "provenance": (
            json.loads(sidecar.read_text(encoding="utf-8"))
            if sidecar.exists()
            else None
        ),
    }


def _player_feature_rows(
    players: Iterable[DefensivePlayerSeason],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in players:
        row: dict[str, Any] = {
            "season": item.season,
            "team": item.team,
            "player_id": item.player_id,
            "player_name": item.player_name,
            "position": item.position,
            "position_group": item.position_group,
            "team_games": item.team_games,
            "defensive_games": item.defensive_games,
            "defensive_game_appearance_rate": item.defensive_experience,
            "defensive_impact": item.defensive_impact,
        }
        for field in ALL_STAT_FIELDS:
            row[field] = item.stats.get(field)
        rows.append(row)
    return rows


def _coverage_rows(
    player_rows: list[Mapping[str, Any]],
    team_rows: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_season: list[dict[str, Any]] = []
    by_group: list[dict[str, Any]] = []
    for season in sorted(
        {
            int(row["season"])
            for row in player_rows
            if row.get("in_model_relevant_population")
        }
    ):
        scoped = [
            row
            for row in player_rows
            if int(row["season"]) == season and row.get("in_model_relevant_population")
        ]
        by_season.append(
            _coverage_row(
                season,
                scoped,
                [row for row in team_rows if int(row["season"]) == season],
            )
        )
        for group in ("dl_edge", "lb", "db"):
            by_group.append(
                _coverage_row(
                    season,
                    [
                        row
                        for row in scoped
                        if row.get("portal_position_group") == group
                    ],
                    [row for row in team_rows if int(row["season"]) == season],
                    group=group,
                )
            )
    return by_season, by_group


def _coverage_row(
    season: int,
    scoped: list[Mapping[str, Any]],
    team_rows: list[Mapping[str, Any]],
    *,
    group: str | None = None,
) -> dict[str, Any]:
    def count(status: str, field: str = "experience_status") -> int:
        return sum(row.get(field) == status for row in scoped)

    statuses = Counter(str(row.get("identity_status")) for row in scoped)
    recoverable_experience = [
        float(row["prior_defensive_experience"])
        for row in scoped
        if row.get("prior_defensive_experience") is not None
    ]
    resolved_experience = [
        float(row["prior_defensive_experience"])
        for row in scoped
        if row.get("experience_status")
        in {"resolved", "legitimate_zero_defensive_participation"}
        and row.get("prior_defensive_experience") is not None
    ]
    return {
        "season": season,
        "position_group": group or "all",
        "incoming_fbs_transfers": len(scoped)
        if group
        else sum(bool(row.get("in_model_relevant_population")) for row in scoped),
        "incoming_defensive_transfers": len(scoped)
        if group
        else sum(bool(row.get("defensive_candidate")) for row in scoped),
        "resolved_experience": count("resolved"),
        "resolved_experience_mass": sum(resolved_experience),
        "recoverable_experience_mass_proxy": sum(recoverable_experience),
        "experience_mass_coverage_proxy": (
            sum(resolved_experience) / sum(recoverable_experience)
            if recoverable_experience and sum(recoverable_experience) > 0
            else None
        ),
        "resolved_impact": count("resolved", "impact_status"),
        "legitimate_zero_experience": count("legitimate_zero_defensive_participation"),
        "legitimate_zero_impact": count(
            "legitimate_zero_defensive_participation", "impact_status"
        ),
        "identity_resolution_failure": statuses["identity_resolution_failure"],
        "ambiguous": statuses["ambiguous"],
        "source_data_unavailable": statuses["source_data_unavailable"],
        "position_mismatch": statuses["position_mismatch"],
        "unknown_position": statuses["unknown_position"],
        "non_defensive_not_applicable": statuses["non_defensive_not_applicable"],
        "team_seasons": len(team_rows),
        "team_seasons_complete": sum(
            row.get("feature_coverage_status") == "complete" for row in team_rows
        ),
        "team_seasons_partial": sum(
            row.get("feature_coverage_status") == "partial" for row in team_rows
        ),
        "team_seasons_no_usable": sum(
            row.get("feature_coverage_status") == "no_usable_defensive_transfer"
            for row in team_rows
        ),
    }


def _distribution_rows(
    player_features: list[Mapping[str, Any]],
    team_features: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for season in sorted({int(row["season"]) for row in player_features}):
        season_players = [
            row for row in player_features if int(row["season"]) == season
        ]
        for group in ("dl_edge", "lb", "db"):
            scoped = [
                row for row in season_players if row.get("position_group") == group
            ]
            for field in (
                "defensive_game_appearance_rate",
                "defensive_impact",
                *ALL_STAT_FIELDS,
            ):
                result.append(
                    {
                        "level": "player",
                        "season": season,
                        "position_group": group,
                        "feature": field,
                        **descriptive_statistics(row.get(field) for row in scoped),
                    }
                )
        season_teams = [row for row in team_features if int(row["season"]) == season]
        for field in (
            "transfer_in_prior_defensive_experience_sum",
            "transfer_in_prior_defensive_impact_sum",
        ):
            result.append(
                {
                    "level": "team_season",
                    "season": season,
                    "position_group": "all",
                    "feature": field,
                    **descriptive_statistics(row.get(field) for row in season_teams),
                }
            )
    return result


def _spot_checks(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row.get("defensive_candidate")]
    resolved = [
        row
        for row in candidates
        if row.get("experience_status") == "resolved"
        and row.get("prior_defensive_experience") is not None
    ]
    if not resolved:
        return []
    experience_values = sorted(
        float(row["prior_defensive_experience"]) for row in resolved
    )
    experience_median = experience_values[len(experience_values) // 2]
    impact_values = [
        abs(float(row["prior_defensive_impact"]))
        for row in resolved
        if row.get("prior_defensive_impact") is not None
    ]
    impact_median = (
        sorted(impact_values)[len(impact_values) // 2] if impact_values else 0.0
    )
    selections: list[tuple[str, Mapping[str, Any]]] = []
    selections.extend(
        ("high_experience", row)
        for row in sorted(
            resolved,
            key=lambda item: float(item["prior_defensive_experience"]),
            reverse=True,
        )[:3]
    )
    selections.extend(
        ("light_experience", row)
        for row in sorted(
            resolved, key=lambda item: float(item["prior_defensive_experience"])
        )[:3]
    )
    strong_modest = [
        row
        for row in resolved
        if row.get("prior_defensive_impact") is not None
        and float(row["prior_defensive_experience"]) <= experience_median
    ]
    selections.extend(
        ("strong_impact_modest_experience", row)
        for row in sorted(
            strong_modest,
            key=lambda item: float(item["prior_defensive_impact"] or 0),
            reverse=True,
        )[:3]
    )
    high_modest = [
        row
        for row in resolved
        if row.get("prior_defensive_impact") is not None
        and float(row["prior_defensive_experience"]) >= experience_median
        and abs(float(row["prior_defensive_impact"])) <= impact_median
    ]
    selections.extend(("high_experience_modest_impact", row) for row in high_modest[:3])
    unresolved = [
        row
        for row in candidates
        if row.get("experience_status")
        not in {"resolved", "legitimate_zero_defensive_participation"}
    ]
    selections.extend(("unresolved", row) for row in unresolved[:3])

    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for kind, row in selections:
        key = (kind, int(row["portal_index"]))
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "spot_check_type": kind,
                "portal_index": row["portal_index"],
                "season": row["season"],
                "player_name": row["player_name"],
                "origin": row["origin"],
                "destination": row["destination"],
                "position": row["position"],
                "portal_position_group": row["portal_position_group"],
                "experience_status": row["experience_status"],
                "impact_status": row["impact_status"],
                "prior_defensive_games": row["prior_defensive_games"],
                "prior_team_games": row["prior_team_games"],
                "prior_defensive_experience": row["prior_defensive_experience"],
                "prior_defensive_impact": row["prior_defensive_impact"],
                "prior_stats": json.dumps(row.get("prior_stats", {}), sort_keys=True),
                "implausible_value_flag": (
                    row.get("prior_defensive_experience") is not None
                    and not 0 <= float(row["prior_defensive_experience"]) <= 1
                ),
            }
        )
    return result


def _implausible_rows(
    player_features: list[Mapping[str, Any]],
    transfer_rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in player_features:
        experience = row.get("defensive_game_appearance_rate")
        if experience is not None and not 0 <= float(experience) <= 1:
            issues.append({"kind": "experience_outside_0_1", **dict(row)})
        for field in ALL_STAT_FIELDS:
            value = row.get(field)
            if value is not None and float(value) < 0:
                issues.append({"kind": f"negative_{field}", **dict(row)})
    for row in transfer_rows:
        impact = row.get("prior_defensive_impact")
        if impact is not None and not isfinite(float(impact)):
            issues.append({"kind": "nonfinite_impact", **dict(row)})
    return issues


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(
    rows: list[Mapping[str, Any]], columns: list[tuple[str, str]]
) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_fmt(row.get(key)).replace("|", "\\|") for key, _ in columns)
            + " |"
        )
    return lines


def _render_report(
    path: Path,
    *,
    summary: Mapping[str, Any],
    coverage: list[Mapping[str, Any]],
    coverage_by_group: list[Mapping[str, Any]],
    distributions: list[Mapping[str, Any]],
    correlations: list[Mapping[str, Any]],
    spot_checks: list[Mapping[str, Any]],
) -> None:
    lines = [
        "# Defensive transfer experience and production audit (issue 103)",
        "",
        "## Recommendation",
        "",
        str(summary["recommendation"]),
        "",
        "This is a feature-construction and data-quality study only. It does not fit a Context variant, compare NLL/CRPS, select weights from outcomes, or combine the candidates with D5.",
        "",
        "## Source inventory",
        "",
        *_markdown_table(
            summary["source_inventory"],
            [
                ("source", "Source"),
                ("fields", "Fields"),
                ("player_identifier", "Player identity"),
                ("defensive_participation", "Participation"),
                ("defensive_production", "Production"),
                ("historical_coverage", "Coverage"),
                ("preseason_semantics", "Timing / limitation"),
            ],
        ),
        "",
        "CFBD provides no defensive snaps or defensive snap share in these endpoints. `/player/usage.overall` is an offensive participation measure and is not substituted for defense. The fallback therefore measures recorded defensive box-score game appearances.",
        "",
        "## Frozen definitions",
        "",
        "- `prior_defensive_experience`: defensive box-score game appearances divided by all source-team FBS/FCS games present in the frozen `/games/players` inputs. This is a participation proxy, never snap share.",
        "- `transfer_in_prior_defensive_experience_sum`: sum of that rate over incoming defensive transfers. A team feature is numeric only when every known defensive incoming transfer is resolved or a legitimate zero; otherwise it is missing.",
        "- `prior_defensive_impact`: equal-weight mean of `z(log1p(component))`, standardized within prior season × defensive position group. Components are frozen as DL/EDGE = tackles, TFL, sacks, QB hurries; LB = tackles, TFL, sacks, passes defended; DB = tackles, passes defended, interceptions.",
        "- `transfer_in_prior_defensive_impact_sum`: sum of prior-player impact values under the same strict missingness rule. Experience and impact are never collapsed.",
        "- `experience_mass_coverage_proxy`: resolved prior defensive experience mass divided by the recoverable prior defensive experience mass among source-team/player records that could be joined. This is explicitly a recoverable-denominator proxy, not full-population coverage.",
        "",
        "## Field availability audit",
        "",
        *_markdown_table(
            summary["field_inventory"],
            [
                ("field", "Field"),
                ("availability", "Availability"),
                ("measurement", "Measurement"),
                ("decision", "Decision"),
            ],
        ),
        "",
        "## Position taxonomy",
        "",
        "DL / EDGE includes DL, EDGE, DE, DT, and NT; LB includes OLB, ILB, MLB, and LB; DB includes CB, DB, S, FS, SS, and NB. Known offensive/special-teams labels are non-defensive. Unknown or hybrid labels remain visible and fail closed; they are never silently assigned.",
        "",
        "## Identity join and missing-data behavior",
        "",
        "Portal records have no shared athlete ID with the roster/game-player sources. The deterministic fallback is normalized player name + normalized source team. Explicit aliases are supported; no fuzzy matching is used. Ambiguous joins, position mismatches, missing source seasons, and unknown positions remain unresolved. A roster player with complete team-game coverage but no defensive row is a legitimate zero, not a failed identity join.",
        "",
        "## Coverage by season",
        "",
        *_markdown_table(
            coverage,
            [
                ("season", "Season"),
                ("incoming_fbs_transfers", "Incoming FBS"),
                ("incoming_defensive_transfers", "Defensive"),
                ("resolved_experience", "Experience resolved"),
                ("experience_mass_coverage_proxy", "Experience-mass proxy"),
                ("resolved_impact", "Impact resolved"),
                ("legitimate_zero_experience", "Legitimate zero"),
                ("identity_resolution_failure", "Identity fail"),
                ("ambiguous", "Ambiguous"),
                ("source_data_unavailable", "Source unavailable"),
                ("position_mismatch", "Position mismatch"),
                ("unknown_position", "Unknown position"),
                ("team_seasons_complete", "Complete teams"),
                ("team_seasons_partial", "Partial teams"),
            ],
        ),
        "",
        "## Coverage by defensive position group",
        "",
        *_markdown_table(
            coverage_by_group,
            [
                ("season", "Season"),
                ("position_group", "Group"),
                ("incoming_defensive_transfers", "Transfers"),
                ("resolved_experience", "Experience resolved"),
                ("experience_mass_coverage_proxy", "Experience-mass proxy"),
                ("resolved_impact", "Impact resolved"),
                ("legitimate_zero_experience", "Legitimate zero"),
                ("identity_resolution_failure", "Identity fail"),
                ("ambiguous", "Ambiguous"),
                ("source_data_unavailable", "Source unavailable"),
                ("position_mismatch", "Position mismatch"),
            ],
        ),
        "",
        "## Distribution diagnostics",
        "",
        "The machine-readable `distribution_diagnostics.csv` contains count, mean, median, standard deviation, p05/p25/p50/p75/p95, maximum, zero fraction, and missing count by season and position group for player fields, plus team-season aggregates for both candidates.",
        "",
        *_markdown_table(
            [
                row
                for row in distributions
                if row["feature"]
                in {"defensive_game_appearance_rate", "defensive_impact"}
                and row["level"] == "player"
            ][:30],
            [
                ("season", "Season"),
                ("position_group", "Group"),
                ("feature", "Feature"),
                ("count", "N"),
                ("mean", "Mean"),
                ("median", "Median"),
                ("std", "SD"),
                ("p95", "P95"),
                ("max", "Max"),
                ("fraction_zero", "Zero fraction"),
                ("missing_count", "Missing"),
            ],
        ),
        "",
        "## Experience versus impact correlation",
        "",
        *_markdown_table(
            correlations,
            [
                ("level", "Level"),
                ("count", "N"),
                ("pearson_experience_vs_impact", "Pearson r"),
            ],
        ),
        "",
        "These correlations are descriptive redundancy diagnostics, not feature-selection results.",
        "",
        "## Spot checks",
        "",
        *_markdown_table(
            spot_checks,
            [
                ("spot_check_type", "Check"),
                ("season", "Season"),
                ("player_name", "Player"),
                ("origin", "Origin"),
                ("destination", "Destination"),
                ("position", "Position"),
                ("experience_status", "Experience status"),
                ("impact_status", "Impact status"),
                ("prior_defensive_games", "Def games"),
                ("prior_team_games", "Team games"),
                ("prior_defensive_experience", "Experience"),
                ("prior_defensive_impact", "Impact"),
                ("implausible_value_flag", "Implausible"),
            ],
        ),
        "",
        "## Limitations and decision",
        "",
        "The experience proxy is not snap share; a recorded defensive row can undercount special packages and can differ by source box-score completeness. Production is opportunity- and scheme-dependent, normalized only within season and broad position group, and does not include forced fumbles because CFBD does not expose a player forced-fumble field in the selected source. Portal destinations and historical endpoint responses are retrospective rather than archived as-of snapshots.",
        "",
        "The candidate definitions are frozen for a later held-out incremental-value experiment. No predictive claim is made here.",
        "",
        "## Artifacts",
        "",
        "- `prior_player_season_features.csv` — defensive player-season aggregates and derived values.",
        "- `transfer_player_audit.csv` — every portal row with scope, position, join, status, and derived values.",
        "- `team_season_features.csv` — strict incoming experience and impact candidates with coverage flags.",
        "- `coverage_by_season.csv`, `coverage_by_position.csv`, `distribution_diagnostics.csv`, `correlations.json`, `spot_checks.csv` — audit diagnostics.",
        "- `position_mapping.json`, `source_inventory.json`, `cutoff_safety.json`, `source_manifest.json`, `summary.json` — frozen configuration and provenance.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    portal_root: Path = DEFAULT_PORTAL_ROOT,
    defensive_root: Path = DEFAULT_DEFENSIVE_ROOT,
    team_file: Path = DEFAULT_TEAM_FILE,
    aliases_file: Path | None = None,
    output: Path = DEFAULT_OUTPUT,
    cutoff_month: int = 8,
    cutoff_day: int = 15,
) -> dict[str, Any]:
    (
        records,
        roster,
        player_seasons,
        team_coverage,
        roster_teams,
        portal_seasons,
        defensive_seasons,
        source_files,
    ) = _load_inputs(portal_root, defensive_root)
    team_rows = [
        row
        for row in _load_team_rows(team_file)
        if int(row["season"]) in portal_seasons
        and str(row.get("subdivision", "")).casefold() == "fbs"
    ]
    aliases = _load_aliases(aliases_file)
    audit = audit_transfer_records(
        records,
        roster,
        player_seasons,
        team_rows,
        portal_seasons=portal_seasons,
        defensive_seasons=defensive_seasons,
        cutoff=date(2025, cutoff_month, cutoff_day),
        aliases=aliases,
        team_coverage=team_coverage,
        roster_teams=roster_teams,
    )
    transfer_rows = audit["player_rows"]
    team_features = audit["team_rows"]
    player_features = _player_feature_rows(player_seasons)
    coverage, coverage_by_group = _coverage_rows(transfer_rows, team_features)
    distributions = _distribution_rows(player_features, team_features)
    player_correlation_rows = [
        {
            "experience_status": "resolved",
            "prior_defensive_experience": row["defensive_game_appearance_rate"],
            "prior_defensive_impact": row["defensive_impact"],
        }
        for row in player_features
    ]
    correlations = correlation_rows(player_correlation_rows, team_features)
    spot_checks = _spot_checks(transfer_rows)
    implausible = _implausible_rows(player_features, transfer_rows)

    resolved_experience = sum(
        row["experience_status"] == "resolved" for row in transfer_rows
    )
    defensive_candidates = sum(row["defensive_candidate"] for row in transfer_rows)
    if not defensive_candidates:
        recommendation = "Outcome D: no known defensive incoming-transfer population was available; stop without creating a defensive transfer feature."
    elif resolved_experience == 0:
        recommendation = "Outcome D: the available defensive source could not resolve any incoming defensive transfer; stop without creating a defensive transfer feature."
    else:
        recommendation = (
            "Outcome B: carry `transfer_in_prior_defensive_experience_sum` as a conservative "
            "defensive-game-appearance-rate candidate and carry `transfer_in_prior_defensive_impact_sum` "
            "as a separate optional position-normalized production candidate. Direct defensive snaps are "
            "unavailable; both features require the strict fail-closed missing-data rule documented here."
        )
    summary: dict[str, Any] = {
        "study": "issue_103_defensive_transfer_features",
        "production_models_modified": False,
        "target_seasons": sorted(portal_seasons),
        "prior_defensive_seasons": sorted(defensive_seasons),
        "portal_record_count": len(records),
        "roster_record_count": len(roster),
        "prior_defensive_player_season_count": len(player_seasons),
        "cutoff": f"season-relative {cutoff_month:02d}-{cutoff_day:02d}",
        "aliases_file": _display_path(aliases_file) if aliases_file else None,
        "explicit_alias_count": len(aliases),
        "recommendation": recommendation,
        "feature_definitions": {
            "transfer_in_prior_defensive_experience_sum": {
                "player_measure": "defensive box-score game appearances / frozen source-team FBS/FCS games",
                "aggregation": "sum over incoming defensive transfers",
                "missingness": "missing if any known defensive incoming transfer is unresolved",
            },
            "transfer_in_prior_defensive_impact_sum": {
                "player_measure": "equal-weight mean of season×position-group z(log1p(component))",
                "aggregation": "sum over incoming defensive transfers",
                "missingness": "missing if any known defensive incoming transfer lacks a resolved impact",
            },
        },
        "position_mapping": position_mapping(),
        "impact_components": {
            "dl_edge": ["tackles", "tackles_for_loss", "sacks", "qb_hurries"],
            "lb": ["tackles", "tackles_for_loss", "sacks", "passes_defended"],
            "db": ["tackles", "passes_defended", "interceptions"],
        },
        "source_inventory": source_inventory(),
        "field_inventory": field_inventory(),
        "cutoff_safety": cutoff_safety(),
        "coverage_by_season": coverage,
        "coverage_by_position": coverage_by_group,
        "correlations": correlations,
        "spot_check_count": len(spot_checks),
        "implausible_value_count": len(implausible),
        "raw_inputs_unchanged": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "summary.json", summary)
    _write_json(output / "position_mapping.json", position_mapping())
    _write_json(output / "source_inventory.json", source_inventory())
    _write_json(output / "field_inventory.json", field_inventory())
    _write_json(output / "cutoff_safety.json", cutoff_safety())
    _write_json(output / "correlations.json", correlations)
    _write_json(output / "source_manifest.json", {"files": source_files})
    _write_json(output / "implausible_values.json", implausible)
    _write_csv(output / "prior_player_season_features.csv", player_features)
    transfer_csv_rows = [
        {
            **row,
            "prior_stats": json.dumps(row.get("prior_stats", {}), sort_keys=True),
        }
        for row in transfer_rows
    ]
    _write_csv(output / "transfer_player_audit.csv", transfer_csv_rows)
    _write_csv(output / "team_season_features.csv", team_features)
    _write_csv(output / "coverage_by_season.csv", coverage)
    _write_csv(output / "coverage_by_position.csv", coverage_by_group)
    _write_csv(output / "distribution_diagnostics.csv", distributions)
    _write_csv(output / "spot_checks.csv", spot_checks)
    _render_report(
        output / "report.md",
        summary=summary,
        coverage=coverage,
        coverage_by_group=coverage_by_group,
        distributions=distributions,
        correlations=correlations,
        spot_checks=spot_checks,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portal-root", type=Path, default=DEFAULT_PORTAL_ROOT)
    parser.add_argument("--defensive-root", type=Path, default=DEFAULT_DEFENSIVE_ROOT)
    parser.add_argument("--team-file", type=Path, default=DEFAULT_TEAM_FILE)
    parser.add_argument("--aliases", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cutoff-month", type=int, default=8)
    parser.add_argument("--cutoff-day", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        portal_root=args.portal_root,
        defensive_root=args.defensive_root,
        team_file=args.team_file,
        aliases_file=args.aliases,
        output=args.output,
        cutoff_month=args.cutoff_month,
        cutoff_day=args.cutoff_day,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "recommendation": summary["recommendation"],
                "portal_record_count": summary["portal_record_count"],
                "prior_defensive_player_season_count": summary[
                    "prior_defensive_player_season_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
