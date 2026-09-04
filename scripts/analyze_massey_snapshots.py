"""Describe candidate final-season Massey Composite snapshots."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = ROOT / "data/processed/massey/observations.csv"
EXPORTS = ROOT / "data/raw/massey_composite"
OUTPUT = ROOT / "data/processed/massey"


def iter_kaggle_rows():
    with OBSERVATIONS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["source"] == "massey_kaggle" and 2003 <= int(row["season"]) <= 2021:
                yield row


def date_age(later: int, earlier: int) -> int:
    return (
        date.fromisoformat(
            str(later)[:4] + "-" + str(later)[4:6] + "-" + str(later)[6:]
        )
        - date.fromisoformat(
            str(earlier)[:4] + "-" + str(earlier)[4:6] + "-" + str(earlier)[6:]
        )
    ).days


def bucket(age: int) -> str:
    if age == 0:
        return "same_day"
    if age <= 2:
        return "one_to_two_days"
    if age <= 7:
        return "three_to_seven_days"
    if age <= 14:
        return "eight_to_fourteen_days"
    if age <= 30:
        return "fifteen_to_thirty_days"
    return "more_than_thirty_days"


def snapshot_stats(rows: list[dict[str, str]]) -> dict[str, object]:
    teams = [row["team_id"] for row in rows]
    ranks = [int(row["ordinal_rank"]) for row in rows]
    unique_ranks = set(ranks)
    one_row_per_team = len(teams) == len(set(teams))
    contiguous_ranks = unique_ranks == set(range(min(ranks), max(ranks) + 1))
    return {
        "rows": len(rows),
        "distinct_teams": len(set(teams)),
        "min_rank": min(ranks),
        "max_rank": max(ranks),
        "one_row_per_team": one_row_per_team,
        "rank_values_contiguous": contiguous_ranks,
        "coherent_one_snapshot": one_row_per_team,
    }


def analyze_kaggle() -> list[dict[str, object]]:
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in iter_kaggle_rows():
        groups[(int(row["season"]), row["subdivision"])].append(row)

    final_dates = {
        key: max(
            int(row["ranking_date"]) for row in rows if row["system_code"] == "CMP"
        )
        for key, rows in groups.items()
    }
    latest: dict[tuple[int, str, str], int] = {}
    for (season, subdivision), rows in groups.items():
        final_date = final_dates[(season, subdivision)]
        for row in rows:
            code = row["system_code"]
            ranking_date = int(row["ranking_date"])
            if code != "CMP" and ranking_date <= final_date:
                key = (season, subdivision, code)
                latest[key] = max(latest.get(key, 0), ranking_date)

    selected: dict[tuple[int, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for (season, subdivision), rows in groups.items():
        final_date = final_dates[(season, subdivision)]
        for row in rows:
            code = row["system_code"]
            ranking_date = int(row["ranking_date"])
            if (
                code == "CMP"
                and ranking_date == final_date
                or code != "CMP"
                and latest.get((season, subdivision, code)) == ranking_date
            ):
                selected[(season, subdivision, code, ranking_date)].append(row)

    reports = []
    for (season, subdivision), rows in sorted(groups.items()):
        final_date = final_dates[(season, subdivision)]
        systems = []
        for code in sorted(
            {row["system_code"] for row in rows if row["system_code"] != "CMP"}
        ):
            ranking_date = latest.get((season, subdivision, code))
            if ranking_date is None:
                systems.append({"system_code": code, "available_at_final_date": False})
                continue
            age = date_age(final_date, ranking_date)
            item = {
                "system_code": code,
                "latest_date": ranking_date,
                "age_days": age,
                "staleness_bucket": bucket(age),
            }
            item.update(
                snapshot_stats(selected[(season, subdivision, code, ranking_date)])
            )
            systems.append(item)
        distribution = {
            name: 0
            for name in (
                "same_day",
                "one_to_two_days",
                "three_to_seven_days",
                "eight_to_fourteen_days",
                "fifteen_to_thirty_days",
                "more_than_thirty_days",
            )
        }
        for system in systems:
            if "staleness_bucket" in system:
                distribution[system["staleness_bucket"]] += 1
        cmp_stats = snapshot_stats(selected[(season, subdivision, "CMP", final_date)])
        reports.append(
            {
                "season": season,
                "subdivision": subdivision,
                "final_cmp_date": final_date,
                "constituent_systems_available": len(
                    [s for s in systems if s.get("available_at_final_date", True)]
                ),
                "staleness_distribution": distribution,
                "maximum_staleness_days": max(
                    (s.get("age_days", 0) for s in systems), default=0
                ),
                "unusually_stale_systems": [
                    s for s in systems if s.get("age_days", 0) > 30
                ],
                "cmp_snapshot": cmp_stats,
                "systems": systems,
                "suspicious_final_cmp": (
                    "before December 15; inspect as potentially incomplete postseason snapshot"
                    if date.fromisoformat(
                        str(final_date)[:4]
                        + "-"
                        + str(final_date)[4:6]
                        + "-"
                        + str(final_date)[6:]
                    )
                    < date(season, 12, 15)
                    else None
                ),
            }
        )
    return reports


def analyze_exports() -> list[dict[str, object]]:
    reports = []
    for path in sorted(EXPORTS.glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            active = {
                code
                for code, value in zip(header[4:], [])
                if code not in {"CMP", "Sort"}
            }
            rows = list(reader)
        constituent_codes = [code for code in header[5:] if code != "Sort"]
        active = {
            code
            for code in constituent_codes
            if any(
                len(row) > header.index(code)
                and row[header.index(code)].strip() not in {"", "--"}
                for row in rows
            )
        }
        reports.append(
            {
                "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "season": int(path.stem[-4:]),
                "subdivision": path.stem[:3],
                "constituent_system_columns": len(
                    [code for code in constituent_codes if code != "CMP"]
                ),
                "active_constituent_systems": len(active - {"CMP"}),
                "includes_cmp": "CMP" in header,
            }
        )
    return reports


def main() -> None:
    report = {"kaggle": analyze_kaggle(), "manual_exports": analyze_exports()}
    path = OUTPUT / "snapshot_semantics_report.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
