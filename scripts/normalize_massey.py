"""Normalize the historical Massey observation corpus without aggregation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from gippyrank.data.massey import (
    HISTORICAL_SUBDIVISION_OVERRIDE_RATIONALES,
    HISTORICAL_SUBDIVISION_OVERRIDES,
    MASSEY_TEAM_ALIASES,
    NORMALIZED_FIELDS,
    CFBDTeam,
    build_team_matches,
    iter_kaggle_observations,
    parse_cfbd_teams,
    read_export_observations,
)

ROOT = Path(__file__).resolve().parents[1]
CORPUS_VERSION = "massey-rank-population-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cfbd_teams(teams_root: Path, season: int) -> list[CFBDTeam]:
    with (teams_root / f"{season}.json").open(encoding="utf-8") as handle:
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


def main(raw_root: Path, output_root: Path) -> None:
    rankings = raw_root / "massey_rankings"
    exports = raw_root / "massey_composite"
    teams_root = raw_root / "cfbd" / "teams"
    mapping_rows, conflicts = [], []
    counts, teams_by_group, systems_by_group = (
        Counter(),
        defaultdict(set),
        defaultdict(set),
    )
    overridden_rows = Counter()
    overridden_names: dict[tuple[int, str], str] = {}
    overridden_systems: dict[tuple[int, str], set[str]] = defaultdict(set)
    observation_count = 0
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "observations.csv").open(
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
            override_key = (row.season, row.team_id or "")
            if override_key in HISTORICAL_SUBDIVISION_OVERRIDES:
                overridden_rows[override_key] += 1
                overridden_names[override_key] = row.team_name
                overridden_systems[override_key].add(row.system_code)

        for season in range(2003, 2022):
            source_path = rankings / f"cf{season}.csv"
            names = names_in_kaggle(source_path)
            matched, ambiguous, unmapped = build_team_matches(
                names, load_cfbd_teams(teams_root, season)
            )
            fallback_seasons = []
            if season == 2020 and (ambiguous or unmapped):
                fallback_teams = load_cfbd_teams(teams_root, season)
                for fallback_season in (2019, 2021):
                    fallback_seasons.append(fallback_season)
                    fallback_teams.extend(
                        load_cfbd_teams(teams_root, fallback_season)
                    )
                    matched, ambiguous, unmapped = build_team_matches(
                        names, fallback_teams
                    )
                    if not ambiguous and not unmapped:
                        break
            for row in iter_kaggle_observations(
                source_path,
                matched,
                source_file=f"data/raw/massey_rankings/cf{season}.csv",
            ):
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
            print(
                f"normalized season {season}: {observation_count} rows so far",
                flush=True,
            )

        for source_path in sorted(exports.glob("*.csv")):
            season = int(source_path.stem[-4:])
            if not 2022 <= season <= 2025:
                continue
            with source_path.open(newline="", encoding="utf-8-sig") as source_handle:
                reader = csv.reader(source_handle)
                next(reader)
                names = {row[0].strip() for row in reader if row}
            matched, ambiguous, unmapped = build_team_matches(
                names, load_cfbd_teams(teams_root, season)
            )
            rows, row_unmapped = read_export_observations(
                source_path,
                matched,
                source_file=f"data/raw/massey_composite/{source_path.name}",
            )
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
            print(
                f"normalized export {source_path.name}: "
                f"{observation_count} rows so far",
                flush=True,
            )

    source_paths = sorted(
        [*rankings.glob("cf*.csv"), *exports.glob("*.csv"), *teams_root.glob("*.json")]
    )
    override_decisions = []
    for (season, team_id), subdivision in sorted(
        HISTORICAL_SUBDIVISION_OVERRIDES.items()
    ):
        teams = {str(team.id): team for team in load_cfbd_teams(teams_root, season)}
        team = teams.get(team_id)
        override_decisions.append(
            {
                "season": season,
                "team_id": team_id,
                "team_name": overridden_names.get(
                    (season, team_id), team.school if team else None
                ),
                "cfbd_classification": team.classification if team else None,
                "model_subdivision": subdivision,
                "rationale": HISTORICAL_SUBDIVISION_OVERRIDE_RATIONALES[
                    (season, team_id)
                ],
                "normalized_observation_count": overridden_rows[(season, team_id)],
                "constituent_system_count": len(
                    overridden_systems[(season, team_id)]
                ),
            }
        )
    report = {
        "corpus_version": CORPUS_VERSION,
        "rank_population_policy": (
            "Model population is the distinct CMP team identity count after "
            "explicit season-specific cohort decisions. Source ordinal ranks "
            "are retained; no rank renumbering or probability clipping is used."
        ),
        "normalized_observation_count": observation_count,
        "normalized_observations": {
            "path": (output_root / "observations.csv").as_posix(),
            "bytes": (output_root / "observations.csv").stat().st_size,
            "sha256": sha256(output_root / "observations.csv"),
        },
        "input_sources": [
            {
                "path": path.relative_to(raw_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in source_paths
        ],
        "historical_subdivision_decisions": override_decisions,
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
                absent_export_cells(path) for path in exports.glob("*.csv")
            ),
            "reason": "Blank and -- export cells are absence, not ranking observations.",
        },
        "malformed_observations": [],
    }
    report_path = output_root / "normalization_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "corpus_version": CORPUS_VERSION,
        "inputs": report["input_sources"],
        "normalized_observations": report["normalized_observations"],
        "normalization_report": {
            "path": report_path.as_posix(),
            "bytes": report_path.stat().st_size,
            "sha256": sha256(report_path),
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {observation_count} observations to {output_root / 'observations.csv'}")
    print(f"mapping report: {output_root / 'normalization_report.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "data/processed/massey"
    )
    arguments = parser.parse_args()
    main(arguments.raw_root, arguments.output_root)
