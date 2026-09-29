from __future__ import annotations

from datetime import UTC, datetime, timedelta

from gippyrank.api.store import _known_unmodeled_score
from gippyrank.result_provenance import known_unmodeled_result


def test_unmodeled_score_needs_matching_corpus_and_available_source() -> None:
    cutoff = datetime(2026, 9, 1, 20, tzinfo=UTC)
    game_date = cutoff - timedelta(hours=2)
    values = {
        "game_date": game_date,
        "cutoff": cutoff,
        "snapshot_type": "weekly",
        "source_mode": "current_cached_cfbd",
        "schedule_source": {
            "kind": "current_processed_schedule",
            "path": "data/processed/cfbd/games.csv",
            "sha256": "a" * 64,
        },
        "game_corpus_sha256": "a" * 64,
        "source_available_at": cutoff,
    }
    assert known_unmodeled_result(**values)
    assert not known_unmodeled_result(
        **{**values, "source_available_at": cutoff + timedelta(seconds=1)}
    )
    assert not known_unmodeled_result(
        **{**values, "schedule_source": {**values["schedule_source"], "sha256": "b" * 64}}
    )
    assert not known_unmodeled_result(**{**values, "snapshot_type": "preseason"})
    assert known_unmodeled_result(
        **{
            **values,
            "schedule_source": {
                "kind": "frozen_historical_schedule",
                "path": "frozen-weekly.json",
                "sha256": "b" * 64,
            },
            "source_available_at": None,
        }
    )
    assert not known_unmodeled_result(**{
        **values, "source_mode": "preseason_prior_only",
        "schedule_source": {"kind": "frozen_historical_schedule", "path": "frozen-weekly.json", "sha256": "b" * 64},
    })


def test_api_uses_same_unmodeled_result_boundary() -> None:
    cutoff = datetime(2026, 9, 1, 20, tzinfo=UTC)
    artifact = {
        "snapshot_type": "weekly",
        "source_mode": "current_cached_cfbd",
        "schedule_source": {
            "kind": "current_processed_schedule",
            "path": "data/processed/cfbd/games.csv",
            "sha256": "a" * 64,
        },
        "game_corpus_sha256": "a" * 64,
        "combined_source_available_at": (cutoff + timedelta(seconds=1)).isoformat(),
    }
    assert not _known_unmodeled_score(artifact, cutoff - timedelta(hours=2), cutoff)
    artifact["combined_source_available_at"] = cutoff.isoformat()
    assert _known_unmodeled_score(artifact, cutoff - timedelta(hours=2), cutoff)
