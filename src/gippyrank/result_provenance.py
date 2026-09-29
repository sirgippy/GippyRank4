"""Availability rule for a scored game outside posterior evidence."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any


def known_unmodeled_result(
    *,
    game_date: datetime | None,
    cutoff: datetime | None,
    snapshot_type: str | None,
    source_mode: str | None,
    schedule_source: Mapping[str, Any] | None,
    game_corpus_sha256: str | None,
    source_available_at: datetime | None,
) -> bool:
    """Require a coherent schedule source and historical result boundary.

    A frozen schedule hashes the presentation artifact, not the model corpus.
    A current schedule must identify the same corpus used by the snapshot.
    """
    schedule_hash = schedule_source.get("sha256") if schedule_source else None
    valid_hash = (
        isinstance(schedule_hash, str)
        and len(schedule_hash) == 64
        and all(character in "0123456789abcdef" for character in schedule_hash)
    )
    current = (
        source_mode in {"current_cached_cfbd", "historical_frozen"}
        and schedule_source is not None
        and schedule_source.get("kind") == "current_processed_schedule"
        and schedule_source.get("path") == "data/processed/cfbd/games.csv"
        and valid_hash
        and schedule_hash == game_corpus_sha256
        and (
            source_mode == "historical_frozen"
            or source_available_at is not None
            and cutoff is not None
            and source_available_at <= cutoff
        )
    )
    frozen = (
        source_mode in {"current_cached_cfbd", "historical_frozen"}
        and schedule_source is not None
        and schedule_source.get("kind") == "frozen_historical_schedule"
        and isinstance(schedule_source.get("path"), str)
        and bool(schedule_source["path"])
        and valid_hash
    )
    return (
        snapshot_type != "preseason"
        and game_date is not None
        and cutoff is not None
        and game_date <= cutoff
        and isinstance(schedule_source, Mapping)
        and isinstance(game_corpus_sha256, str)
        and (current or frozen)
    )
