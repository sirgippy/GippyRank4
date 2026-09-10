"""Checked-in r/CFB team-handle mappings used by ballot export.

The ranking model uses stable CFBD team IDs, while the r/CFB poll importer
uses ``Team.handle`` values.  This module keeps that translation explicit and
rejects ambiguous mapping tables before they can reach the browser.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
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


class TeamHandleMappingError(ValueError):
    """Raised when the checked-in handle table is not unambiguous."""


@dataclass(frozen=True)
class TeamHandleTable:
    """Validated mapping rows and duplicate diagnostics."""

    handles: dict[str, str]
    team_names: dict[str, str]
    duplicate_cfbd_ids: tuple[str, ...] = ()
    duplicate_redditcfb_handles: tuple[str, ...] = ()

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
    unique_identities = sorted({(str(team_id), str(team_name)) for team_id, team_name in identities})
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
