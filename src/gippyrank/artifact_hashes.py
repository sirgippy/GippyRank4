"""Canonical semantic hashes shared by snapshot generation and publication."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

INCLUDED_GAME_FIELDS = (
    "id",
    "season",
    "week",
    "seasonType",
    "startDate",
    "completed",
    "neutralSite",
    "conferenceGame",
    "homeId",
    "homeTeam",
    "homeClassification",
    "homeConference",
    "homePoints",
    "awayId",
    "awayTeam",
    "awayClassification",
    "awayConference",
    "awayPoints",
)


def stable_values_sha256(values: list[str]) -> str:
    """Hash an ordered canonical list without depending on JSON formatting."""
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def posterior_pmfs_sha256(pmfs: Mapping[str, Sequence[float]]) -> str:
    """Hash canonical team/rank probabilities independent of CSV formatting."""
    canonical = [
        [team_id, [float(value).hex() for value in pmfs[team_id]]]
        for team_id in sorted(pmfs)
    ]
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def included_game_rows_sha256(rows: list[dict[str, str]]) -> str:
    """Hash the ordered game evidence consumed by posterior inference.

    The source CSV is intentionally not hashed byte-for-byte: line endings and
    CSV quoting are serialization details, while the fixed field/value surface
    below is the evidence contract shared by source and replay snapshots.
    """
    digest = hashlib.sha256()
    for row in rows:
        canonical = {field: str(row.get(field, "")) for field in INCLUDED_GAME_FIELDS}
        digest.update(
            json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()
