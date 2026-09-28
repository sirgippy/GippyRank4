"""Guard the retained production corpus against an unmaterialized artifact."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from gippyrank import site_data
from gippyrank.site_data import SiteDataValidationError

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


def test_retrospective_records_match_schedule_participants_site_and_score() -> None:
    source = ROOT / "data/processed/snapshots/2026/2026-weekly-2026-09-27T12-27-35.698895Z-context-v1.3/predictive/context"
    original = json.loads((source / "team_seasons.json").read_text(encoding="utf-8"))
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    expected_ids = set(original["included_game_ids"])

    def validate(artifact: dict) -> None:
        site_data._validate_retrospective_game_expectations(
            artifact, metadata, expected_ids, artifact["teams"]
        )

    validate(original)
    game_id = "401856700"

    def record(data: dict) -> dict:
        return data["retrospective_game_expectations"]["games"][game_id]

    def schedule_game(data: dict) -> dict:
        return next(game for game in data["teams"]["61"]["games"] if game["game_id"] == game_id)

    mutations = (
        (lambda data: record(data).update(home_team_id="194", away_team_id="333"), "wrong team"),
        (lambda data: schedule_game(data).update(opponent_id="999"), "opponent mismatch"),
        (lambda data: schedule_game(data).update(game_id="other-game"), "another matchup"),
        (lambda data: record(data).update(neutral_site=True), "site orientation mismatch"),
        (lambda data: record(data).update(actual_home_margin=28.4), "not integral"),
        (lambda data: record(data).update(actual_home_margin=29.0), "disagrees with score"),
    )
    for mutate, message in mutations:
        artifact = deepcopy(original)
        mutate(artifact)
        with pytest.raises(SiteDataValidationError, match=message):
            validate(artifact)
