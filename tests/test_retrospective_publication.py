"""Guard the retained production corpus against an unmaterialized artifact."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_retained_predictive_snapshots_publish_completed_game_expectations() -> None:
    config = json.loads((ROOT / "site/publish_config.json").read_text(encoding="utf-8"))
    sources = {
        ROOT / entry["source"]
        for entry in config["snapshots"]
        if "/predictive/" in entry["source"]
    }
    assert sources
    for source in sources:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        team_seasons = json.loads(
            (source / "team_seasons.json").read_text(encoding="utf-8")
        )
        artifact = team_seasons["retrospective_game_expectations"]
        assert metadata["retrospective_game_expectations_version"] == "1.0"
        assert artifact["source_snapshot_id"] == metadata["snapshot_id"]
        assert set(artifact["games"]) == set(metadata["included_game_ids"])
        assert len(artifact["games"]) == metadata["included_game_count"]
