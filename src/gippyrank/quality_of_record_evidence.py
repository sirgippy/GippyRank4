"""One temporary QoR exception for three immutable 2026 source publications.

This is not an ingestion or general tied-game policy. Remove the exception
when the affected historical Context/Quality of Play lineage is repaired.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gippyrank.artifact_hashes import included_game_rows_sha256, stable_values_sha256

CAMPBELL_GAME_ID = "401866625"
CAMPBELL_EXCEPTION_ID = "campbell_western_carolina_2026_frozen_qor_v1"
INFERENCE_GAMES_FILE = "inference_games.csv"
# Snapshot ID -> (prior version, complete source metadata hash, source row hash).
# Pin both artifacts: the older Context 1.2 metadata predates semantic row hashes.
CAMPBELL_EXCEPTION_SOURCES = {
    "2026-weekly-2026-09-06T17-34-56.895103Z-context": (
        "1.2",
        "22bdf4079398e79e89b17041c90e3bae0356b147de75d422587063336e901f03",
        "f3b5f56a4b2e02c7249ba1b14f0ea17244cb37f69efbe01d2b4cc1409e927a35",
    ),
    "2026-weekly-2026-09-07T13-42-50.853587Z-context": (
        "1.2",
        "a0f29b767a784439b319a50230136421481ac467f67c4817f20cf9c3e1793d76",
        "b4d716d5e84dac0ce051389a447c25a674d8a7c7ce978b82900cc7108ab95fbd",
    ),
    "2026-weekly-2026-09-08T11-43-00.275833Z-context-v1.4-db-repair": (
        "1.4",
        "8a8a0f45e3c4a4e0699a9fbd4124c6fdf92157ba39cc25769ae654f450251a94",
        "9fe5aca7aa961dde5177f9eb9266f8fe23c65f8437bcc3e389bdcaff54675082",
    ),
}


def quality_of_record_evidence(
    source: Mapping[str, Any],
    rows: list[dict[str, str]],
    *,
    source_metadata_sha256: str,
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    """Return exact inference rows and an auditable declaration when excluded."""
    identity = CAMPBELL_EXCEPTION_SOURCES.get(str(source["snapshot_id"]))
    used = rows
    declaration = None
    if identity is not None:
        version, metadata_hash, rows_hash = identity
        if (
            source.get("season") != 2026
            or source.get("ranking_family") != "predictive"
            or source.get("prior_family") != "context"
            or source.get("prior_model_version") != version
            or source_metadata_sha256 != metadata_hash
            or included_game_rows_sha256(rows) != rows_hash
        ):
            raise ValueError(
                "Campbell QoR exception no longer matches its frozen source"
            )
        excluded = [row for row in rows if row["id"] == CAMPBELL_GAME_ID]
        expected = {
            "id": CAMPBELL_GAME_ID,
            "season": "2026",
            "completed": "True",
            "homeId": "2097",
            "homeTeam": "Campbell",
            "homeClassification": "fcs",
            "homePoints": "0",
            "awayId": "2717",
            "awayTeam": "Western Carolina",
            "awayClassification": "fcs",
            "awayPoints": "0",
        }
        if len(excluded) != 1 or any(
            excluded[0].get(key) != value for key, value in expected.items()
        ):
            raise ValueError(
                "Campbell QoR exclusion does not match the malformed 2026 fixture"
            )
        used = [row for row in rows if row["id"] != CAMPBELL_GAME_ID]
        declaration = {
            "schema_version": "1.0",
            "exception_id": CAMPBELL_EXCEPTION_ID,
            "source_game_count": len(rows),
            "source_game_ids_sha256": stable_values_sha256([row["id"] for row in rows]),
            "source_game_rows_sha256": rows_hash,
            "exclusions": [
                {
                    "game_id": CAMPBELL_GAME_ID,
                    "reason": "Malformed completed 0-0 original fixture; historical repair deferred",
                    "source_game_row_sha256": included_game_rows_sha256(excluded),
                }
            ],
            "inference_game_count": len(used),
            "inference_game_ids": [row["id"] for row in used],
            "inference_game_ids_sha256": stable_values_sha256(
                [row["id"] for row in used]
            ),
            "inference_game_rows_sha256": included_game_rows_sha256(used),
        }
    for row in used:
        if float(row["homePoints"]) == float(row["awayPoints"]):
            raise ValueError(f"game {row['id']} has no binary winner")
    return used, declaration


def validate_quality_of_record_evidence(
    source: Mapping[str, Any],
    rows: list[dict[str, str]],
    output: Mapping[str, Any],
    *,
    source_metadata_sha256: str,
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    """Require the exact exception, while preserving complete source parity."""
    used, expected = quality_of_record_evidence(
        source, rows, source_metadata_sha256=source_metadata_sha256
    )
    if output.get("quality_of_record_evidence") != expected:
        raise ValueError("Quality of Record evidence exclusion declaration mismatch")
    return used, expected
