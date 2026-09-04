"""Build durable all-observation and final-snapshot Massey datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = (
    "season",
    "subdivision",
    "team_id",
    "team_source_id",
    "team_name",
    "system_code",
    "system_name",
    "ranking_date",
    "ordinal_rank",
    "is_composite",
    "source",
    "source_file",
)
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/massey/observations.csv"
OUTPUT = ROOT / "data/processed/massey"
ALL = OUTPUT / "all_observations.csv"
FINAL = OUTPUT / "final_observations.csv"
REPORT = OUTPUT / "dataset_validation_report.json"
FINAL_DATE_OVERRIDES = {(2007, "fcs"): 20071217}


def row_key(row: dict[str, str]) -> str:
    return "\x1f".join(row[field] for field in FIELDS)


def valid_rank(value: str) -> bool:
    try:
        return int(value) > 0 and str(int(value)) == value
    except ValueError:
        return False


def read_source() -> tuple[dict[tuple[int, str], int], Counter, int]:
    final_dates: dict[tuple[int, str], int] = {}
    source_counts = Counter()
    seen: set[bytes] = set()
    duplicates = 0
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Normalized source schema does not match expected fields")
        for row in reader:
            season = int(row["season"])
            source_counts[(season, row["subdivision"])] += 1
            digest = hashlib.blake2b(row_key(row).encode(), digest_size=16).digest()
            if digest in seen:
                duplicates += 1
            seen.add(digest)
            if row["source"] == "massey_kaggle" and row["system_code"] == "CMP":
                key = (season, row["subdivision"])
                final_dates[key] = max(
                    final_dates.get(key, 0), int(row["ranking_date"])
                )
    final_dates.update(FINAL_DATE_OVERRIDES)
    return final_dates, source_counts, duplicates


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def validate(
    rows: list[dict[str, str]], final_dates: dict[tuple[int, str], int]
) -> dict[str, object]:
    duplicates = 0
    hashes: set[bytes] = set()
    invalid_ranks = []
    missing_team_ids = []
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        digest = hashlib.blake2b(row_key(row).encode(), digest_size=16).digest()
        if digest in hashes:
            duplicates += 1
        hashes.add(digest)
        key = (int(row["season"]), row["subdivision"])
        groups[key].append(row)
        if not row["team_id"]:
            missing_team_ids.append(row)
        if not valid_rank(row["ordinal_rank"]):
            invalid_ranks.append(row)

    checks = {
        "at_least_one_final_cmp": {},
        "final_constituent_observations": {},
        "team_ids_populated": not missing_team_ids,
        "ordinal_ranks_positive_integers": not invalid_ranks,
        "no_exact_duplicate_observations": duplicates == 0,
        "cmp_and_constituents_distinguishable": {},
        "both_fbs_and_fcs_2003_2025": True,
    }
    summary = []
    for season in range(2003, 2026):
        for subdivision in ("fbs", "fcs"):
            key = (season, subdivision)
            group = groups.get(key, [])
            cmp_rows = [row for row in group if row["system_code"] == "CMP"]
            constituent_rows = [row for row in group if row["system_code"] != "CMP"]
            checks["at_least_one_final_cmp"][f"{season}-{subdivision}"] = bool(cmp_rows)
            checks["final_constituent_observations"][f"{season}-{subdivision}"] = bool(
                constituent_rows
            )
            checks["cmp_and_constituents_distinguishable"][
                f"{season}-{subdivision}"
            ] = bool(cmp_rows) and bool(constituent_rows)
            systems = {row["system_code"] for row in constituent_rows}
            summary.append(
                {
                    "season": season,
                    "subdivision": subdivision,
                    "observations": len(group),
                    "distinct_teams": len({row["team_id"] for row in group}),
                    "constituent_systems": len(systems),
                    "final_cmp_date": final_dates.get(key),
                }
            )
            if not group:
                checks["both_fbs_and_fcs_2003_2025"] = False

    return {
        "validation": checks,
        "invalid_rank_count": len(invalid_ranks),
        "missing_team_id_count": len(missing_team_ids),
        "exact_duplicate_count": duplicates,
        "summary": summary,
    }


def main() -> None:
    final_dates, source_counts, all_duplicates = read_source()
    shutil.copyfile(SOURCE, ALL)

    final_rows = []
    omitted = Counter()
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            season = int(row["season"])
            subdivision = row["subdivision"]
            if row["source"] == "massey_kaggle" and 2003 <= season <= 2021:
                if subdivision not in {"fbs", "fcs"}:
                    omitted[(season, subdivision, "unsupported classification")] += 1
                    continue
                if int(row["ranking_date"]) == final_dates[(season, subdivision)]:
                    final_rows.append(row)
            elif row["source"] == "massey_composite_export" and 2022 <= season <= 2025:
                final_rows.append(row)
            else:
                omitted[(season, subdivision, "outside intended corpus")] += 1

    write_csv(FINAL, final_rows)
    report = validate(final_rows, final_dates)
    report.update(
        {
            "all_observations_count": sum(source_counts.values()),
            "all_observations_exact_duplicate_count": all_duplicates,
            "final_observations_count": len(final_rows),
            "source_counts": {
                f"{season}-{subdivision}": count
                for (season, subdivision), count in sorted(source_counts.items())
            },
            "omitted_or_anomalous_records": [
                {
                    "season": season,
                    "subdivision": subdivision,
                    "reason": reason,
                    "count": count,
                }
                for (season, subdivision, reason), count in sorted(omitted.items())
            ],
            "anomalies": [
                {
                    "season": item["season"],
                    "subdivision": item["subdivision"],
                    "reason": "Final CMP-date snapshot contains fewer than 10 distinct teams; retained exactly as sourced.",
                    "observations": item["observations"],
                    "distinct_teams": item["distinct_teams"],
                    "constituent_systems": item["constituent_systems"],
                }
                for item in report["summary"]
                if item["distinct_teams"] < 10
            ],
            "notes": [
                "Kaggle final observations are restricted to the authoritative final CMP date.",
                "The 2007 FCS final CMP date is explicitly corrected to 2007-12-17; 2008-01-07 rows remain in all_observations only.",
                "Manual 2022-2025 exports are used directly as final snapshots.",
                "The 2020 CFBD 'ii' classification is preserved in all_observations but omitted from the intended FBS/FCS final dataset.",
                "No ranking aggregation or staleness cutoff was applied.",
            ],
        }
    )
    REPORT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {ALL} ({sum(source_counts.values())} rows)")
    print(f"wrote {FINAL} ({len(final_rows)} rows)")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
