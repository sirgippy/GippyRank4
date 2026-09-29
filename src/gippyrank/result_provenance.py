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
    schedule_source: Mapping[str, Any] | None,
    game_corpus_sha256: str | None,
    source_available_at: datetime | None,
    frozen_source: bool,
) -> bool:
    """Require the selected corpus and an established historical result boundary."""
    return (
        snapshot_type != "preseason"
        and game_date is not None
        and cutoff is not None
        and game_date <= cutoff
        and isinstance(schedule_source, Mapping)
        and isinstance(game_corpus_sha256, str)
        and schedule_source.get("sha256") == game_corpus_sha256
        and (
            frozen_source
            or source_available_at is not None
            and source_available_at <= cutoff
        )
    )
