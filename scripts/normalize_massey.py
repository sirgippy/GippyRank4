"""Normalize the historical Massey observation corpus without aggregation."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from gippyrank.data.massey import (
    MASSEY_TEAM_ALIASES,
    NORMALIZED_FIELDS,
    CFBDTeam,
    build_team_matches,
    iter_kaggle_observations,
    parse_cfbd_teams,
    read_export_observations,
)

ROOT = Path(__file__).resolve().parents[1]
RANKINGS = ROOT / "data/raw/massey_rankings"
EXPORTS = ROOT / "data/raw/massey_composite"
CFBD_TEAMS = ROOT / "data/raw/cfbd/teams"
OUTPUT = ROOT / "data/processed/massey"


def load_cfbd_teams(season: int) -> list[CFBDTeam]:
    with (CFBD_TEAMS / f"{season}.json").open(encoding="utf-8") as handle:
        return parse_cfbd_teams(json.load(handle))


def names_in_kaggle(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row[2].strip() for row in csv.reader(handle)}


def absent_export_cells(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        return sum(
            value.strip() in {"", "--"}
            for row in reader
            if row
            for code, value in zip(header[6:], row[6:])
            if code != "Sort"
        )


def main() -> None:
    mapping_rows, conflicts = [], []
    counts, teams_by_group, systems_by_group = (
        Counter(),
        defaultdict(set),
        defaultdict(set),
    )
    observation_count = 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "observations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=NORMALIZED_FIELDS)
        writer.writeheader()

        def emit(row) -> None:
            nonlocal observation_count
            writer.writerow(asdict(row))
            observation_count += 1
            key = (row.season, row.subdivision, row.source)
            counts[key] += 1
            teams_by_group[key].add(row.team_id)
            systems_by_group[key].add(row.system_code)

        for season in range(2003, 2022):
            source_path = RANKINGS / f"cf{season}.csv"
            names = names_in_kaggle(source_path)
            matched, ambiguous, unmapped = build_team_matches(
                names, load_cfbd_teams(season)
            )
            fallback_seasons = []
            if season == 2020 and (ambiguous or unmapped):
                fallback_teams = load_cfbd_teams(season)
                for fallback_season in (2019, 2021):
                    fallback_seasons.append(fallback_season)
                    fallback_teams.extend(load_cfbd_teams(fallback_season))
                    matched, ambiguous, unmapped = build_team_matches(
                        names, fallback_teams
                    )
                    if not ambiguous and not unmapped:
                        break
            for row in iter_kaggle_observations(source_path, matched):
                emit(row)
            mapping_rows.append(
                {
                    "season": season,
                    "source": "massey_kaggle",
                    "source_team_count": len(names),
                    "mapped_team_count": len(matched),
                    "mapping_success_rate": len(matched) / len(names),
                    "ambiguous": ambiguous,
                    "unmapped": sorted(unmapped),
                    "classification_fallback_seasons": fallback_seasons,
                    "aliases_used": sorted(
                        name for name in names if name in MASSEY_TEAM_ALIASES
                    ),
                }
            )

        for source_path in sorted(EXPORTS.glob("*.csv")):
            season = int(source_path.stem[-4:])
            if not 2022 <= season <= 2025:
                continue
            with source_path.open(newline="", encoding="utf-8-sig") as source_handle:
                reader = csv.reader(source_handle)
                next(reader)
                names = {row[0].strip() for row in reader if row}
            matched, ambiguous, unmapped = build_team_matches(
                names, load_cfbd_teams(season)
            )
            rows, row_unmapped = read_export_observations(source_path, matched)
            for row in rows:
                emit(row)
            subdivision = source_path.stem[:3]
            for name, team in matched.items():
                if team.classification != subdivision:
                    conflicts.append(
                        {
                            "season": season,
                            "team_name": name,
                            "source_subdivision": subdivision,
                            "cfbd_classification": team.classification,
                        }
                    )
            mapping_rows.append(
                {
                    "season": season,
                    "source": "massey_composite_export",
                    "source_team_count": len(names),
                    "mapped_team_count": len(matched),
                    "mapping_success_rate": len(matched) / len(names),
                    "ambiguous": ambiguous,
                    "unmapped": sorted(set(unmapped) | set(row_unmapped)),
                    "classification_fallback_seasons": [],
                    "aliases_used": sorted(
                        name for name in names if name in MASSEY_TEAM_ALIASES
                    ),
                }
            )

    report = {
        "normalized_observation_count": observation_count,
        "counts_by_season_subdivision_source": [
            {
                "season": season,
                "subdivision": subdivision,
                "source": source,
                "observations": counts[(season, subdivision, source)],
                "distinct_teams": len(teams_by_group[(season, subdivision, source)]),
                "distinct_systems": len(
                    systems_by_group[(season, subdivision, source)]
                ),
            }
            for season, subdivision, source in sorted(counts)
        ],
        "mapping": mapping_rows,
        "explicit_aliases": sorted(
            {name for item in mapping_rows for name in item["aliases_used"]}
        ),
        "classification_conflicts": conflicts,
        "classification_anomalies": [
            {
                "season": season,
                "subdivision": subdivision,
                "source": source,
                "observations": counts[(season, subdivision, source)],
                "distinct_teams": len(teams_by_group[(season, subdivision, source)]),
                "reason": "CFBD classification is outside the target fbs/fcs values; preserved as supplied.",
            }
            for season, subdivision, source in sorted(counts)
            if subdivision not in {"fbs", "fcs"}
        ],
        "omitted_observations": {
            "kaggle": 0,
            "exports_absent_or_unranked_cells": sum(
                absent_export_cells(path) for path in EXPORTS.glob("*.csv")
            ),
            "reason": "Blank and -- export cells are absence, not ranking observations.",
        },
        "malformed_observations": [],
    }
    (OUTPUT / "normalization_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {observation_count} observations to {OUTPUT / 'observations.csv'}")
    print(f"mapping report: {OUTPUT / 'normalization_report.json'}")


if __name__ == "__main__":
    main()
