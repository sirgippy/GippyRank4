"""Build the machine-readable issue 198 snapshot shopping list."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path

from gippyrank.data.massey import build_team_matches, parse_cfbd_teams
from gippyrank.data.massey_prebowl import (
    MASSEY_EXPORT_URL,
    SEASONS,
    PreBowlWindow,
    latest_cmp_date_in_window,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_WINDOWS = ROOT / "data/research/massey_prebowl_issue_198/schedule_windows.csv"
OUTPUT = ROOT / "data/research/massey_prebowl_issue_198/collection_checklist.csv"
KAGGLE_URL = "https://www.kaggle.com/datasets/masseyratings/rankings"
MANUAL_SNAPSHOT_DATES = {
    2019: date(2019, 12, 9),
    2021: date(2021, 12, 6),
    2022: date(2022, 12, 5),
    2023: date(2023, 12, 3),
    2024: date(2024, 12, 8),
    2025: date(2025, 12, 7),
}
APPROVED_PHASE_EXCEPTIONS = {
    2020: {
        "snapshot_date": date(2020, 12, 21),
        "decision": "approved_use_2020-12-21",
        "rationale": (
            "User explicitly approved using the Dec 21, 2020 snapshot despite its "
            "overlap with the first bowl date."
        ),
    }
}
FIELDS = (
    "season",
    "last_fbs_conference_championship_date",
    "first_fbs_postseason_game_date",
    "window_first_full_date",
    "window_last_full_date",
    "snapshot_date",
    "snapshot_source",
    "snapshot_file",
    "phase_exception_decision",
    "phase_exception_rationale",
    "source_url",
    "official_snapshot_url",
    "schedule_source_url",
    "schedule_source_sha256",
    "schedule_raw_path",
    "schedule_retrieved_at",
    "existing_source_file",
    "existing_source_coverage",
    "existing_source_sha256",
    "collection_status",
    "notes",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kaggle_names(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row[2].strip() for row in csv.reader(handle) if len(row) >= 3}


def team_matches(raw_root: Path, season: int, names: set[str]):
    team_path = raw_root / "cfbd" / "teams" / f"{season}.json"
    with team_path.open(encoding="utf-8") as handle:
        teams = parse_cfbd_teams(json.load(handle))
    matches, ambiguous, unmapped = build_team_matches(names, teams)
    if season == 2020 and (ambiguous or unmapped):
        for fallback_season in (2019, 2021):
            fallback_path = raw_root / "cfbd" / "teams" / f"{fallback_season}.json"
            with fallback_path.open(encoding="utf-8") as handle:
                teams.extend(parse_cfbd_teams(json.load(handle)))
            matches, ambiguous, unmapped = build_team_matches(names, teams)
            if not ambiguous and not unmapped:
                break
    return matches, ambiguous, unmapped


def historical_ranking_path(raw_root: Path, season: int) -> Path:
    staged = raw_root / "massey_prebowl" / "historical" / f"cf{season}.csv"
    if staged.exists():
        return staged
    return raw_root / "massey_rankings" / f"cf{season}.csv"


def canonical_raw_path(path: Path, raw_root: Path) -> str:
    return (Path("data/raw") / path.relative_to(raw_root)).as_posix()


def main(raw_root: Path, schedule_windows: Path, output: Path) -> None:
    with schedule_windows.open(newline="", encoding="utf-8-sig") as handle:
        schedules = {int(row["season"]): row for row in csv.DictReader(handle)}
    if set(schedules) != set(SEASONS):
        raise ValueError("Schedule windows must cover seasons 2003 through 2025")

    rows = []
    for season in SEASONS:
        schedule = schedules[season]
        window = PreBowlWindow(
            season=season,
            last_fbs_conference_championship=date.fromisoformat(
                schedule["last_fbs_conference_championship_date"]
            ),
            first_fbs_postseason_game=date.fromisoformat(
                schedule["first_fbs_postseason_game_date"]
            ),
        )
        raw_ranking_path = historical_ranking_path(raw_root, season)
        existing_hash = sha256(raw_ranking_path) if raw_ranking_path.exists() else ""
        snapshot_date: date | None = None
        source = ""
        snapshot_file = ""
        source_url = ""
        official_url = ""
        phase_exception_decision = ""
        phase_exception_rationale = ""
        existing_source_file = (
            canonical_raw_path(raw_ranking_path, raw_root)
            if raw_ranking_path.exists()
            else ""
        )
        coverage = ""
        status = ""
        notes = ""

        if season <= 2018:
            source = "massey_kaggle"
            snapshot_file = existing_source_file or canonical_raw_path(
                raw_ranking_path, raw_root
            )
            source_url = KAGGLE_URL
            if raw_ranking_path.exists():
                matches, ambiguous, unmapped = team_matches(
                    raw_root, season, kaggle_names(raw_ranking_path)
                )
                if ambiguous or unmapped:
                    coverage = (
                        "Team mapping needs review before selecting an FBS snapshot."
                    )
                    status = "mapping_review_required"
                    notes = f"Ambiguous: {sorted(ambiguous)}; unmapped: {unmapped}"
                else:
                    snapshot_date = latest_cmp_date_in_window(
                        raw_ranking_path, window, matches=matches
                    )
                    if snapshot_date is not None:
                        official_url = MASSEY_EXPORT_URL.format(
                            yyyymmdd=snapshot_date.strftime("%Y%m%d")
                        )
                        coverage = "Dated raw rows include the latest FBS CMP date inside the accepted window."
                        status = "available_local"
                    else:
                        coverage = "Dated raw file is present, but no FBS CMP date lies inside the accepted window."
                        status = "exception_decision_required"
                        notes = "Do not substitute a same-day pre-title or post-bowl snapshot."
            else:
                coverage = "No local dated raw ranking file found."
                status = "missing_source"

        elif season in MANUAL_SNAPSHOT_DATES:
            source = "massey_prebowl_export"
            snapshot_date = MANUAL_SNAPSHOT_DATES[season]
            export_name = f"fbs{season}-{snapshot_date:%Y%m%d}.csv"
            snapshot_file = f"data/raw/massey_prebowl/{export_name}"
            source_url = MASSEY_EXPORT_URL.format(
                yyyymmdd=snapshot_date.strftime("%Y%m%d")
            )
            official_url = source_url
            final_export = raw_root / "massey_composite" / f"fbs{season}.csv"
            if final_export.exists():
                existing_source_file = f"data/raw/massey_composite/fbs{season}.csv"
                existing_hash = sha256(final_export)
            export_path = raw_root / "massey_prebowl" / export_name
            if export_path.exists():
                coverage = "Dated pre-bowl export is present locally."
                status = "available_local"
            else:
                coverage = (
                    "Dated pre-bowl export is missing; existing undated/final export is not eligible."
                    if final_export.exists()
                    else "Dated pre-bowl export is missing."
                )
                status = "needs_manual_export"
            notes = (
                "Export the full page table, including CMP and all constituent systems."
            )

        elif season == 2020:
            exception = APPROVED_PHASE_EXCEPTIONS[season]
            source = "massey_prebowl_export"
            snapshot_date = exception["snapshot_date"]
            phase_exception_decision = exception["decision"]
            phase_exception_rationale = exception["rationale"]
            export_name = f"fbs{season}-{snapshot_date:%Y%m%d}.csv"
            snapshot_file = f"data/raw/massey_prebowl/{export_name}"
            source_url = MASSEY_EXPORT_URL.format(
                yyyymmdd=snapshot_date.strftime("%Y%m%d")
            )
            official_url = source_url
            export_path = raw_root / "massey_prebowl" / export_name
            coverage = (
                "The Dec 21 snapshot overlaps the first bowl date and is included under "
                "the explicit user-approved phase exception. The existing dated Dec 19 "
                "source is same-day with the championship and is superseded for this target."
            )
            status = (
                "available_local_with_approved_phase_exception"
                if export_path.exists()
                else "approved_phase_exception_export_missing"
            )
            notes = phase_exception_rationale
            raw_source = historical_ranking_path(raw_root, 2020)
            if raw_source.exists():
                existing_source_file = canonical_raw_path(raw_source, raw_root)
                existing_hash = sha256(raw_source)
                names = kaggle_names(raw_source)
                matches, ambiguous, unmapped = team_matches(raw_root, season, names)
                if ambiguous or unmapped:
                    notes += (
                        f" Team mapping needs review: ambiguous={sorted(ambiguous)}, "
                        f"unmapped={unmapped}."
                    )
                else:
                    latest_in_window = latest_cmp_date_in_window(
                        raw_source, window, matches=matches
                    )
                    if latest_in_window:
                        notes += (
                            f" Raw source latest FBS CMP date in the strict full-day window: "
                            f"{latest_in_window.isoformat()}."
                        )
        first_full = window.first_eligible_date
        last_full = window.last_eligible_date
        rows.append(
            {
                "season": season,
                "last_fbs_conference_championship_date": window.last_fbs_conference_championship.isoformat(),
                "first_fbs_postseason_game_date": window.first_fbs_postseason_game.isoformat(),
                "window_first_full_date": first_full.isoformat(),
                "window_last_full_date": last_full.isoformat(),
                "snapshot_date": snapshot_date.isoformat() if snapshot_date else "",
                "snapshot_source": source,
                "snapshot_file": snapshot_file,
                "phase_exception_decision": phase_exception_decision,
                "phase_exception_rationale": phase_exception_rationale,
                "source_url": source_url,
                "official_snapshot_url": official_url,
                "schedule_source_url": schedule["source_url"],
                "schedule_source_sha256": schedule["raw_sha256"],
                "schedule_raw_path": schedule["raw_path"],
                "schedule_retrieved_at": schedule["retrieved_at"],
                "existing_source_file": existing_source_file,
                "existing_source_coverage": coverage,
                "existing_source_sha256": existing_hash,
                "collection_status": status,
                "notes": notes,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} ({len(rows)} seasons)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--schedule-windows", type=Path, default=SCHEDULE_WINDOWS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments.raw_root, arguments.schedule_windows, arguments.output)
