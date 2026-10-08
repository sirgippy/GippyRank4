"""Freeze the CFBD-only stratified sample for issue 185.

This script deliberately reads only the issue 183 CFBD season summary and
CFBD FBS team metadata. It never checks whether an official roster is
available, so source discoverability cannot influence sample membership.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "data/processed/offensive_line_shared_roster_issue_183/team_season_summaries.csv"
OUTPUT_PATH = ROOT / "data/research/offensive_line_target_pool_validation_issue_185/frozen_sample.csv"
PROTOCOL_PATH = ROOT / "data/research/offensive_line_target_pool_validation_issue_185/sample_freeze.json"
WINDOWS = ((2009, 2012), (2013, 2016), (2017, 2020), (2021, 2026))
WINDOW_COUNTS = {"low_quartile": 3, "middle_half": 4, "upper_quartile": 3}
KNOWN_CASE_EXCLUSIONS = {("Vanderbilt", 2017), ("Vanderbilt", 2018), ("Vanderbilt", 2019), ("Vanderbilt", 2020)}
SEED = 185


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_conferences(raw_root: Path, years: set[int]) -> dict[tuple[int, str], str]:
    result: dict[tuple[int, str], str] = {}
    for year in sorted(years):
        path = raw_root / "teams_fbs" / f"{year}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for team in payload:
            result[(year, str(team.get("id") or ""))] = str(
                team.get("conference") or "Unknown"
            )
    return result


def _rank_bands(
    rows: list[dict[str, Any]], value_field: str, *, tie_seed: str
) -> dict[tuple[int, str], int]:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row[value_field]),
            hashlib.sha256(
                f"{tie_seed}|{row['season']}|{row['team_id']}".encode()
            ).hexdigest(),
        ),
    )
    return {
        (int(row["season"]), str(row["team_id"])): min(
            3, (4 * index) // len(ordered)
        )
        for index, row in enumerate(ordered)
    }


def _era_rows(
    summary_rows: list[dict[str, str]],
    conferences: dict[tuple[int, str], str],
    low_year: int,
    high_year: int,
) -> list[dict[str, Any]]:
    candidates = [
        dict(row)
        for row in summary_rows
        if low_year <= int(row["season"]) <= high_year
        and row["roster_response_available"] == "true"
        and (row["team_name"], int(row["season"])) not in KNOWN_CASE_EXCLUSIONS
    ]
    if not candidates:
        raise RuntimeError(f"No CFBD roster candidates in {low_year}-{high_year}")
    ol_bands = _rank_bands(candidates, "identifiable_ol_player_count", tie_seed="issue-185-ol")
    size_bands = _rank_bands(candidates, "roster_player_rows", tie_seed="issue-185-size")
    for row in candidates:
        key = (int(row["season"]), str(row["team_id"]))
        row["conference"] = conferences.get(key, "Unknown")
        row["position_count_stratum"] = (
            "low_quartile"
            if ol_bands[key] == 0
            else "upper_quartile"
            if ol_bands[key] == 3
            else "middle_half"
        )
        row["roster_size_quartile"] = size_bands[key]
    return candidates


def _pick_window(
    rows: list[dict[str, Any]],
    rng: random.Random,
    *,
    excluded_teams: set[str],
    require_low_extreme: bool = False,
    require_high_extreme: bool = False,
) -> list[dict[str, Any]]:
    buckets = {
        "low_quartile": [r for r in rows if r["position_count_stratum"] == "low_quartile" and r["team_name"] not in excluded_teams],
        "middle_half": [r for r in rows if r["position_count_stratum"] == "middle_half" and r["team_name"] not in excluded_teams],
        "upper_quartile": [r for r in rows if r["position_count_stratum"] == "upper_quartile" and r["team_name"] not in excluded_teams],
    }
    for bucket in buckets.values():
        bucket.sort(key=lambda row: (int(row["season"]), row["team_name"]))
    for _ in range(100_000):
        chosen: list[dict[str, Any]] = []
        for name, count in WINDOW_COUNTS.items():
            population = buckets[name]
            if len(population) < count:
                raise RuntimeError(f"Insufficient {name} candidates: {len(population)}")
            chosen.extend(rng.sample(population, count))
        teams = [row["team_name"] for row in chosen]
        conferences = {row["conference"] for row in chosen}
        sizes = [int(row["roster_size_quartile"]) for row in chosen]
        if len(set(teams)) != len(teams) or len(conferences) < 8:
            continue
        if any(sizes.count(q) < 2 for q in range(4)):
            continue
        ol_counts = [int(row["identifiable_ol_player_count"]) for row in chosen]
        if require_low_extreme and min(ol_counts) > 1:
            continue
        if require_high_extreme and max(ol_counts) < 30:
            continue
        return chosen
    raise RuntimeError("Could not satisfy conference/team/roster-size balance constraints")


def freeze_sample(raw_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows = _read_csv(SUMMARY_PATH)
    years = {
        int(row["season"])
        for row in summary_rows
        if any(low <= int(row["season"]) <= high for low, high in WINDOWS)
    }
    conferences = _load_conferences(raw_root, years)
    rng = random.Random(SEED)
    sample: list[dict[str, Any]] = []
    selected_teams: set[str] = set()
    for low, high in WINDOWS:
        candidates = _era_rows(summary_rows, conferences, low, high)
        selected = _pick_window(
            candidates,
            rng,
            excluded_teams=selected_teams,
            require_low_extreme=(low == 2009),
            require_high_extreme=(low == 2021),
        )
        selected_teams.update(row["team_name"] for row in selected)
        for row in selected:
            row["era"] = f"{low}-{high}"
            sample.append(row)
    sample.sort(
        key=lambda row: (
            WINDOWS.index(tuple(map(int, row["era"].split("-")))),
            {"low_quartile": 0, "middle_half": 1, "upper_quartile": 2}[
                row["position_count_stratum"]
            ],
            int(row["season"]),
            row["team_name"],
        )
    )
    output = []
    for index, row in enumerate(sample, start=1):
        output.append(
            {
                "sample_id": f"S{index:02d}",
                "era": row["era"],
                "season": row["season"],
                "team_id": row["team_id"],
                "team_name": row["team_name"],
                "conference": row["conference"],
                "cfbd_roster_player_rows": row["roster_player_rows"],
                "cfbd_ol_count": row["identifiable_ol_player_count"],
                "position_count_stratum": row["position_count_stratum"],
                "roster_size_quartile": row["roster_size_quartile"],
            }
        )
    protocol = {
        "issue": 185,
        "frozen_on": "2026-10-07",
        "sample_size": len(output),
        "seed": SEED,
        "windows": [f"{low}-{high}" for low, high in WINDOWS],
        "allocations_per_window": WINDOW_COUNTS,
        "position_count_strata": "Within each era, rank team-seasons by identifiable CFBD OL count; tie-break with SHA-256(issue-185-ol|season|team_id); the bottom quartile is low, the top quartile high, and the middle two quartiles ordinary. Sample 3 low, 4 middle, 3 high.",
        "roster_size_strata": "Within each era, rank by CFBD roster player rows with a separate SHA-256 tie-break; each roster-size quartile must contribute at least two sampled rows.",
        "selection_constraints": [
            "Ten distinct team names per era, with no program repeated across the 40 team-seasons.",
            "At least eight distinct CFBD conference labels per era.",
            "At least two sampled team-seasons in each CFBD roster-size quartile per era.",
            "Include at least one 2009-2012 team-season with 0-1 CFBD OL labels and one 2021-2026 team-season with at least 30 CFBD OL labels.",
            "No official roster availability or content was consulted for sample selection.",
        ],
        "known_case_exclusions": [
            "Vanderbilt 2017-2020 was excluded because it contains the already-known Drew Birchmeier position-history case from issue 183.",
            "Iowa 2008 is outside the study window and is not counted.",
        ],
        "input_summary_path": SUMMARY_PATH.relative_to(ROOT).as_posix(),
        "input_summary_sha256": sha256(SUMMARY_PATH),
        "conference_metadata": "CFBD /teams/fbs payloads under the shared issue 183 raw cache; only team ID and conference were used.",
        "conference_metadata_inputs": [
            {
                "path": f"teams_fbs/{year}.json",
                "sha256": sha256(raw_root / "teams_fbs" / f"{year}.json"),
            }
            for year in sorted(years)
        ],
        "source_root": "<research-data-root>/raw/cfbd/offensive_line_shared_roster_issue_183",
    }
    return output, protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--check", action="store_true", help="Verify frozen files without rewriting them")
    args = parser.parse_args()
    rows, protocol = freeze_sample(args.raw_root.expanduser().resolve())
    if args.check:
        if not args.sample.exists() or not args.protocol.exists():
            raise SystemExit("Frozen sample or protocol is missing")
        expected_csv = _serialize_csv(rows)
        if args.sample.read_text(encoding="utf-8") != expected_csv:
            raise SystemExit("Frozen sample does not match the deterministic draw")
        existing = json.loads(args.protocol.read_text(encoding="utf-8"))
        if existing != protocol:
            raise SystemExit("Sample protocol does not match the frozen inputs")
        print(f"Verified {len(rows)} frozen sample rows")
        return
    if args.sample.exists() or args.protocol.exists():
        raise SystemExit("Refusing to replace a frozen sample; remove files explicitly to re-freeze")
    args.sample.parent.mkdir(parents=True, exist_ok=True)
    args.protocol.parent.mkdir(parents=True, exist_ok=True)
    args.sample.write_text(_serialize_csv(rows), encoding="utf-8")
    args.protocol.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"Froze {len(rows)} team-seasons in {args.sample}")


def _serialize_csv(rows: list[dict[str, Any]]) -> str:
    import io

    if not rows:
        raise ValueError("Sample is empty")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


if __name__ == "__main__":
    main()
