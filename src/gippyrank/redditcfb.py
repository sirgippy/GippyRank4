"""Checked-in r/CFB team-handle mappings used by ballot export.

The ranking model uses stable CFBD team IDs, while the r/CFB poll importer
uses ``Team.handle`` values.  This module keeps that translation explicit and
rejects ambiguous mapping tables before they can reach the browser.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

REQUIRED_TEAM_HANDLE_FIELDS = {"cfbd_team_id", "team_name", "redditcfb_handle"}
HANDLE_PATTERN = re.compile(r"^[a-z0-9]+$")

# These are namespace aliases rather than name-derived slugs.  Keeping the
# list explicit makes the mapping audit explain the non-obvious handles.
MANUALLY_RESOLVED_ALIASES: tuple[dict[str, str], ...] = (
    {"cfbd_team_id": "58", "team_name": "South Florida", "redditcfb_handle": "usf"},
    {"cfbd_team_id": "98", "team_name": "Western Kentucky", "redditcfb_handle": "wku"},
    {"cfbd_team_id": "151", "team_name": "East Carolina", "redditcfb_handle": "ecu"},
    {
        "cfbd_team_id": "2026",
        "team_name": "App State",
        "redditcfb_handle": "appalachianstate",
    },
    {"cfbd_team_id": "2226", "team_name": "Florida Atlantic", "redditcfb_handle": "fau"},
    {"cfbd_team_id": "41", "team_name": "UConn", "redditcfb_handle": "connecticut"},
    {
        "cfbd_team_id": "2229",
        "team_name": "Florida International",
        "redditcfb_handle": "fiu",
    },
    {"cfbd_team_id": "2433", "team_name": "UL Monroe", "redditcfb_handle": "ulm"},
    {
        "cfbd_team_id": "2534",
        "team_name": "Sam Houston",
        "redditcfb_handle": "samhoustonstate",
    },
    {"cfbd_team_id": "113", "team_name": "Massachusetts", "redditcfb_handle": "umass"},
)

# Historical display-name changes belong here when they are intentionally
# accepted.  An empty table is deliberate for the current publication: a
# changed name must not silently inherit a stable-ID mapping.
ALLOWED_TEAM_NAME_ALIASES: dict[str, frozenset[str]] = {}


class TeamHandleMappingError(ValueError):
    """Raised when the checked-in handle table is not unambiguous."""


class BallotExportError(ValueError):
    """Raised when published ranking rows cannot form an importable ballot."""


@dataclass(frozen=True)
class TeamHandleTable:
    """Validated mapping rows and duplicate diagnostics."""

    handles: dict[str, str]
    team_names: dict[str, str]
    duplicate_cfbd_ids: tuple[str, ...] = ()
    duplicate_redditcfb_handles: tuple[str, ...] = ()

    @classmethod
    def empty(cls) -> TeamHandleTable:
        """Return an explicit no-export mapping for unconfigured fixtures."""
        return cls({}, {})

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, str]]) -> TeamHandleTable:
        handles: dict[str, str] = {}
        team_names: dict[str, str] = {}
        ids: list[str] = []
        handles_seen: list[str] = []
        for row in rows:
            team_id = str(row.get("cfbd_team_id", "")).strip()
            team_name = str(row.get("team_name", "")).strip()
            handle = str(row.get("redditcfb_handle", "")).strip()
            if not team_id or not team_name or not handle:
                raise TeamHandleMappingError(
                    "r/CFB team-handle mapping rows require nonblank IDs, names, and handles"
                )
            if not HANDLE_PATTERN.fullmatch(handle):
                raise TeamHandleMappingError(
                    f"Invalid r/CFB team handle for CFBD team {team_id}: {handle!r}"
                )
            ids.append(team_id)
            handles_seen.append(handle)
            handles.setdefault(team_id, handle)
            team_names.setdefault(team_id, team_name)

        duplicate_ids = tuple(sorted({value for value in ids if ids.count(value) > 1}))
        duplicate_handles = tuple(
            sorted({value for value in handles_seen if handles_seen.count(value) > 1})
        )
        if duplicate_ids or duplicate_handles:
            details = []
            if duplicate_ids:
                details.append(f"duplicate CFBD IDs: {list(duplicate_ids)}")
            if duplicate_handles:
                details.append(f"duplicate r/CFB handles: {list(duplicate_handles)}")
            raise TeamHandleMappingError("; ".join(details))
        return cls(handles, team_names, duplicate_ids, duplicate_handles)


def validate_team_handle_identities(
    identities: Iterable[tuple[str, str]], table: TeamHandleTable
) -> None:
    """Require every mapped published ID to retain its checked-in team name."""
    mismatches = []
    for team_id, team_name in sorted({(str(team_id), str(team_name)) for team_id, team_name in identities}):
        mapped_name = table.team_names.get(team_id)
        if mapped_name is None or team_name == mapped_name:
            continue
        if team_name in ALLOWED_TEAM_NAME_ALIASES.get(team_id, frozenset()):
            continue
        mismatches.append(
            f"{team_id}: published {team_name!r}, mapping {mapped_name!r}"
        )
    if mismatches:
        raise TeamHandleMappingError(
            "r/CFB team-handle mapping identity mismatch: " + "; ".join(mismatches)
        )


def load_team_handle_mapping(path: Path) -> TeamHandleTable:
    """Read and validate a checked-in ``cfbd_team_id -> Team.handle`` CSV."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_TEAM_HANDLE_FIELDS - set(reader.fieldnames or [])
            if missing:
                raise TeamHandleMappingError(
                    f"{path}: mapping missing fields {sorted(missing)}"
                )
            return TeamHandleTable.from_rows(reader)
    except OSError as error:
        raise TeamHandleMappingError(f"Cannot read r/CFB team-handle mapping {path}: {error}") from error


def audit_team_handle_coverage(
    identities: Iterable[tuple[str, str]], table: TeamHandleTable
) -> dict[str, object]:
    """Summarize mapping coverage for the FBS identities selected for publication."""
    identity_values = list(identities)
    validate_team_handle_identities(identity_values, table)
    unique_identities = sorted(
        {(str(team_id), str(team_name)) for team_id, team_name in identity_values}
    )
    unmapped = [
        {"cfbd_team_id": team_id, "team_name": team_name}
        for team_id, team_name in unique_identities
        if team_id not in table.handles
    ]
    identity_ids = {team_id for team_id, _ in unique_identities}
    aliases = [
        alias
        for alias in MANUALLY_RESOLVED_ALIASES
        if alias["cfbd_team_id"] in identity_ids
        and table.handles.get(alias["cfbd_team_id"]) == alias["redditcfb_handle"]
    ]
    return {
        "current_fbs_team_count": len(unique_identities),
        "mapped_count": len(unique_identities) - len(unmapped),
        "unmapped_teams": unmapped,
        "duplicate_cfbd_ids": list(table.duplicate_cfbd_ids),
        "duplicate_redditcfb_handles": list(table.duplicate_redditcfb_handles),
        "manually_resolved_aliases": aliases,
    }


def _ballot_percentage(value: object) -> str:
    """Match the deterministic percentage formatter used by the browser."""
    probability = float(value)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise BallotExportError("Ballot probability must be finite and between 0 and 1")
    percent = probability * 100
    if probability == 1:
        return "100%"
    if percent == 0:
        return "0%"
    if percent < 0.01:
        return "<0.01%"
    if percent < 1:
        return f"{percent:.2f}%" if percent < 0.1 else f"{percent:.1f}%"
    if percent < 10:
        return f"{percent:.1f}%"
    if percent >= 99.95:
        return "<100%"
    if percent >= 95:
        return f"{percent:.1f}%"
    return f"{math.floor(percent + 0.5)}%"


def build_ballot(
    *,
    season: int,
    snapshot_label: str,
    ranking_family: str,
    prior_family: str | None,
    site_url: str,
    rankings: Iterable[Mapping[str, object]],
    handles: Mapping[str, str],
) -> dict[str, object]:
    """Build the r/CFB importer payload from already-published ranking rows.

    The static page has a deliberately equivalent client-side implementation;
    this pure function provides a regression-testable contract for the
    generated JSON without requiring a browser runtime in CI.
    """
    if ranking_family not in {"predictive", "performance"}:
        raise BallotExportError(f"Unsupported ballot ranking family: {ranking_family}")
    if ranking_family == "predictive" and prior_family not in {"context", "history"}:
        raise BallotExportError("Predictive ballot export requires a Context or History prior")
    rated = [row for row in rankings if row.get("rated", True) is not False]
    try:
        ordered = sorted(
            rated,
            key=lambda row: (
                float(row["expected_rank"]),
                int(float(row["display_rank"])),
                str(row["team_id"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BallotExportError("Published ranking rows are missing ballot ordering fields") from error
    if len(ordered) < 25:
        raise BallotExportError(
            f"A ballot requires 25 rated teams; this snapshot has only {len(ordered)}"
        )
    selected = ordered[:25]
    missing = [
        str(row.get("team_name", row.get("team_id", "")))
        for row in selected
        if str(row.get("team_id")) not in handles
    ]
    if missing:
        raise BallotExportError(
            "Canonical r/CFB handles are unavailable for: " + ", ".join(missing)
        )

    family_label = "Predictive" if ranking_family == "predictive" else "Performance"
    prior_label = ""
    if ranking_family == "predictive":
        prior_label = " History" if prior_family == "history" else " Context"
    semantics = (
        "Predictive estimates current underlying team quality using preseason information plus games."
        if ranking_family == "predictive"
        else "Performance asks what quality is implied by games played, using Context estimates to interpret opponent quality. Performance is not standings, strength of record, or postseason deservingness."
    )
    overall = (
        "Generated from GippyRank4, a probabilistic college-football ranking model "
        "that estimates underlying team quality from game performance and expresses "
        "uncertainty rather than treating rank as perfectly known. This ballot uses "
        f"the {season} {snapshot_label} {family_label}{prior_label} rankings. "
        f"{semantics}\n\nExplore the rankings and uncertainty at: {site_url}"
    )
    entries = []
    for rank, row in enumerate(selected, 1):
        try:
            expected = float(row["expected_rank"])
            interval = row["interval_80"]
            low, high = (int(float(interval[0])), int(float(interval[1])))
            probability = _ballot_percentage(row["top25_probability"])
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise BallotExportError("Published ranking rows are missing ballot rationale fields") from error
        prefix = (
            "GippyRank Performance-equivalent expected rank"
            if ranking_family == "performance"
            else "GippyRank expected rank"
        )
        entries.append(
            {
                "rank": rank,
                "team_handle": handles[str(row["team_id"])],
                "rationale": (
                    f"{prefix}: {expected:.1f}. Central 80% interval: {low}–{high}. "
                    f"Top-25 probability: {probability}."
                ),
            }
        )
    return {"poll_type": "computer", "overall_rationale": overall, "entries": entries}
