"""Date selection and validation for pre-bowl Massey snapshots."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from gippyrank.data.massey import (
    NORMALIZED_FIELDS,
    CFBDTeam,
    MasseyObservation,
    historical_subdivision,
    iter_kaggle_observations,
    read_export_observations,
)

SEASONS = range(2003, 2026)
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
)
MASSEY_EXPORT_URL = "https://masseyratings.com/ranks?c=1&d={yyyymmdd}&s=cf"
APPROVED_PHASE_EXCEPTIONS = {
    (2020, date(2020, 12, 21)): "approved_use_2020-12-21",
}
_SNAPSHOT_FILENAME = re.compile(r"^fbs(?P<season>\d{4})-(?P<date>\d{8})\.csv$")
_FCS_CHAMPIONSHIP_MARKERS = (
    "FCS",
    "IAA",
    "I-AA",
    "DIVISION I FOOTBALL CHAMPIONSHIP",
    "DIV 1-AA",
    "PLAYOFF",
)
_FCS_BOWL_MARKERS = ("CELEBRATION BOWL",)


@dataclass(frozen=True)
class PreBowlWindow:
    season: int
    last_fbs_conference_championship: date
    first_fbs_postseason_game: date

    @property
    def first_eligible_date(self) -> date:
        """First full calendar day after all FBS conference title games."""
        return self.last_fbs_conference_championship + timedelta(days=1)

    @property
    def last_eligible_date(self) -> date:
        """Last full calendar day before the first FBS bowl or CFP game."""
        return self.first_fbs_postseason_game - timedelta(days=1)

    def contains(self, snapshot_date: date) -> bool:
        return self.first_eligible_date <= snapshot_date <= self.last_eligible_date

    @property
    def has_full_day_window(self) -> bool:
        return self.first_eligible_date <= self.last_eligible_date


def snapshot_date_is_accepted(
    season: int,
    snapshot_date: date,
    window: PreBowlWindow,
    phase_exception_decision: str = "",
) -> bool:
    """Check the strict window, allowing only recorded phase exceptions."""
    if window.contains(snapshot_date):
        return True
    required_decision = APPROVED_PHASE_EXCEPTIONS.get((season, snapshot_date))
    return required_decision is not None and phase_exception_decision == required_decision


def _local_event_date(event: dict[str, object]) -> date:
    timestamp = str(event["date"])
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(ZoneInfo("America/New_York")).date()


def _event_notes(event: dict[str, object]) -> str:
    competitions = event.get("competitions") or []
    if not competitions:
        return ""
    notes = competitions[0].get("notes") or []
    return " ".join(str(note.get("headline", "")) for note in notes).upper()


def _is_fbs_conference_championship(event: dict[str, object]) -> bool:
    if event.get("season", {}).get("type") != 2:
        return False
    notes = _event_notes(event)
    return "CHAMPIONSHIP" in notes and not any(
        marker in notes for marker in _FCS_CHAMPIONSHIP_MARKERS
    )


def _is_fbs_postseason_game(event: dict[str, object]) -> bool:
    if event.get("season", {}).get("type") != 3:
        return False
    competitions = event.get("competitions") or []
    competition_type = competitions[0].get("type", {}) if competitions else {}
    notes = _event_notes(event)
    if any(marker in notes for marker in _FCS_BOWL_MARKERS):
        return False
    return (
        competition_type.get("id") == "42"
        or "BOWL" in notes
        or "COLLEGE FOOTBALL PLAYOFF" in notes
    )


def schedule_window_from_espn(payload: dict[str, object], season: int) -> PreBowlWindow:
    """Derive the date-only phase window from a December ESPN scoreboard dump."""
    championships = [
        _local_event_date(event)
        for event in payload.get("events", [])
        if _is_fbs_conference_championship(event)
    ]
    postseason = [
        _local_event_date(event)
        for event in payload.get("events", [])
        if _is_fbs_postseason_game(event)
    ]
    if not championships:
        raise ValueError(f"No FBS conference championship found for {season}")
    if not postseason:
        raise ValueError(f"No FBS bowl/CFP game found for {season}")
    return PreBowlWindow(
        season=season,
        last_fbs_conference_championship=max(championships),
        first_fbs_postseason_game=min(postseason),
    )


def date_from_ranking_date(value: str) -> date:
    """Parse the YYYYMMDD date format used in the normalized source schema."""
    normalized = value.strip()
    if len(normalized) != 8 or not normalized.isdigit():
        raise ValueError(f"Invalid ranking date {value!r}; expected YYYYMMDD")
    try:
        return date(int(normalized[:4]), int(normalized[4:6]), int(normalized[6:8]))
    except ValueError as exc:
        raise ValueError(f"Invalid ranking date {value!r}; expected YYYYMMDD") from exc


def latest_cmp_date_in_window(
    path: Path,
    window: PreBowlWindow,
    matches: dict[str, CFBDTeam] | None = None,
) -> date | None:
    """Return the last dated CMP snapshot inside a strict full-day window."""
    candidates: set[date] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if len(row) < 7 or row[3].strip() != "CMP":
                continue
            if row[0].strip().removeprefix("cf") != str(window.season):
                continue
            if matches is not None:
                team = matches.get(row[2].strip())
                if (
                    team is None
                    or historical_subdivision(window.season, team) != "fbs"
                ):
                    continue
            try:
                candidate = date_from_ranking_date(row[5])
            except ValueError:
                continue
            if window.contains(candidate):
                candidates.add(candidate)
    return max(candidates) if candidates else None


def snapshot_file_metadata(path: Path) -> tuple[int, date]:
    """Read season and snapshot date from a dated manual export filename."""
    match = _SNAPSHOT_FILENAME.fullmatch(path.name)
    if match is None:
        raise ValueError(
            f"Dated FBS export must be named fbsYYYY-YYYYMMDD.csv: {path.name}"
        )
    season = int(match.group("season"))
    snapshot_date = date_from_ranking_date(match.group("date"))
    if season not in SEASONS:
        raise ValueError(f"Unexpected Massey season in {path.name}: {season}")
    return season, snapshot_date


def read_dated_fbs_export(
    path: Path,
    matches: dict[str, CFBDTeam],
    *,
    expected_season: int,
    expected_date: date,
    source_file: str,
) -> tuple[list[MasseyObservation], list[str]]:
    season, snapshot_date = snapshot_file_metadata(path)
    if season != expected_season or snapshot_date != expected_date:
        raise ValueError(
            f"{path.name} does not match expected snapshot "
            f"{expected_season}-{expected_date:%Y%m%d}"
        )
    non_fbs = sorted(
        name
        for name, team in matches.items()
        if historical_subdivision(expected_season, team) != "fbs"
    )
    if non_fbs:
        raise ValueError(f"FBS export contains non-FBS teams: {non_fbs}")
    rows, unmapped = read_export_observations(
        path,
        matches,
        ranking_date=snapshot_date.strftime("%Y%m%d"),
        source="massey_prebowl_export",
        source_file=source_file,
    )
    return rows, unmapped


def validate_snapshot(
    rows: list[dict[str, str]], *, season: int, snapshot_date: date
) -> None:
    """Reject malformed or incomplete dated snapshots without imputing ranks."""
    if not rows:
        raise ValueError(f"No observations for FBS snapshot {season}-{snapshot_date}")
    missing_fields = set(NORMALIZED_FIELDS) - set(rows[0])
    if missing_fields:
        raise ValueError(f"Snapshot rows are missing fields: {sorted(missing_fields)}")

    seen: set[tuple[str, str]] = set()
    cmp_count = 0
    constituent_count = 0
    expected_rank_date = snapshot_date.strftime("%Y%m%d")
    for row in rows:
        if int(row["season"]) != season or row["subdivision"] != "fbs":
            raise ValueError(
                "A pre-bowl FBS snapshot contains a wrong season/subdivision"
            )
        if row["ranking_date"] != expected_rank_date:
            raise ValueError("A pre-bowl FBS snapshot contains a mixed ranking date")
        if not row["team_id"] or not row["team_name"]:
            raise ValueError(f"Missing canonical team identity in {row}")
        rank = int(row["ordinal_rank"])
        if rank <= 0:
            raise ValueError(f"Non-positive ordinal rank in {row}")
        system_code = row["system_code"].strip()
        if not system_code:
            raise ValueError("Missing constituent system identifier")
        if (row["team_id"], system_code) in seen:
            raise ValueError(
                f"Duplicate team/system observation: {row['team_id']} {system_code}"
            )
        seen.add((row["team_id"], system_code))
        is_composite = system_code == "CMP"
        if str(row["is_composite"]).lower() != str(is_composite).lower():
            raise ValueError(f"CMP and constituent rows are conflated in {row}")
        cmp_count += int(is_composite)
        constituent_count += int(not is_composite)

    if not cmp_count:
        raise ValueError("Snapshot has no CMP observations")
    if not constituent_count:
        raise ValueError("Snapshot has no constituent-system observations")


def normalized_rows(observations: list[MasseyObservation]) -> list[dict[str, str]]:
    """Convert dataclass observations to the stable CSV representation."""
    from dataclasses import asdict

    rows = []
    for observation in observations:
        row = asdict(observation)
        rows.append(
            {
                field: "" if row[field] is None else str(row[field])
                for field in NORMALIZED_FIELDS
            }
        )
    return rows


def audit_target_rank_support(
    distributions: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Report PMF mass outside each declared integer support without altering it."""
    anomalies = []
    for row in distributions:
        population = int(row["team_population"])
        out_of_support = [
            point
            for point in json.loads(str(row["pmf"]))
            if not 1 <= int(point["rank"]) <= population
        ]
        if not out_of_support:
            continue
        anomalies.append(
            {
                "season": int(row["season"]),
                "subdivision": str(row["subdivision"]),
                "team_id": str(row["team_id"]),
                "team_name": str(row["team_name"]),
                "team_population": population,
                "usable_systems": int(row["usable_systems"]),
                "out_of_support_ranks": json.dumps(
                    sorted({int(point["rank"]) for point in out_of_support})
                ),
                "out_of_support_observations": sum(
                    int(point["count"]) for point in out_of_support
                ),
                "out_of_support_probability_mass": sum(
                    float(point["probability"]) for point in out_of_support
                ),
            }
        )
    return anomalies


def read_kaggle_snapshot(
    path: Path,
    matches: dict[str, CFBDTeam],
    *,
    season: int,
    snapshot_date: date,
    source_file: str,
) -> list[dict[str, str]]:
    """Read only exact-date FBS rows from a historical dated ranking file."""
    expected_date = snapshot_date.strftime("%Y%m%d")
    observations = [
        row
        for row in iter_kaggle_observations(path, matches, source_file=source_file)
        if row.season == season
        and row.ranking_date == expected_date
        and row.subdivision == "fbs"
    ]
    rows = normalized_rows(observations)
    validate_snapshot(rows, season=season, snapshot_date=snapshot_date)
    return rows


def unchanged_fcs_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Select existing FCS observations without changing any field or order."""
    return [row for row in rows if row["subdivision"] == "fcs"]


def fcs_rows_match(
    source_rows: list[dict[str, str]], combined_rows: list[dict[str, str]]
) -> bool:
    """Check exact ordered equality for the carried-forward FCS rows."""
    return unchanged_fcs_rows(source_rows) == unchanged_fcs_rows(combined_rows)


def snapshot_filename(season: int, snapshot_date: date) -> str:
    return f"fbs{season}-{snapshot_date:%Y%m%d}.csv"


def provenance_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.provenance.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_raw_export(
    source: Path,
    *,
    season: int,
    snapshot_date: date,
    output_root: Path,
    source_url: str,
) -> Path:
    """Copy a dated export byte-for-byte and write a verified provenance sidecar."""
    if season not in SEASONS:
        raise ValueError(f"Unexpected Massey season: {season}")
    if snapshot_date.year != season:
        raise ValueError("Snapshot date year must match the requested Massey season")
    if not source_url.strip():
        raise ValueError("A source URL is required for raw provenance")
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = output_root / snapshot_filename(season, snapshot_date)
    sidecar = provenance_path(destination)
    if destination.exists() or sidecar.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing raw export or provenance sidecar: {destination}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    metadata = {
        "schema_version": 1,
        "season": season,
        "snapshot_date": snapshot_date.isoformat(),
        "source_url": source_url,
        "imported_at": datetime.now(UTC).isoformat(),
        "sha256": file_sha256(destination),
        "bytes": destination.stat().st_size,
        "copied_byte_for_byte": True,
    }
    sidecar.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def validate_raw_provenance(
    path: Path,
    *,
    season: int,
    snapshot_date: date,
    source_url: str,
) -> dict[str, object]:
    """Check that a raw dated export has a matching importer sidecar and hash."""
    sidecar = provenance_path(path)
    if not sidecar.exists():
        raise ValueError(f"Missing provenance sidecar: {sidecar}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "season": season,
        "snapshot_date": snapshot_date.isoformat(),
        "source_url": source_url,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "copied_byte_for_byte": True,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    try:
        imported_at = datetime.fromisoformat(str(metadata["imported_at"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid imported_at in {sidecar}") from exc
    if imported_at.tzinfo is None:
        mismatches["imported_at_timezone"] = (
            str(metadata["imported_at"]),
            "UTC offset",
        )
    if mismatches:
        raise ValueError(f"Provenance mismatch in {sidecar}: {mismatches}")
    return metadata
