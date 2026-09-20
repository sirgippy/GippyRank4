from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from gippyrank.context_comparison import ContextComparisonError, validate_week4_pair


def _write_pair_artifact(root: Path, *, prior_version: str, game_id: str) -> None:
    root.mkdir(parents=True)
    metadata = {
        "season": 2026,
        "snapshot_id": f"week-4-context-{prior_version}",
        "snapshot_type": "weekly",
        "requested_cutoff": "2026-09-19T23:59:00+00:00",
        "effective_cutoff": "2026-09-19T23:59:00+00:00",
        "historical_likelihood_version": "V1",
        "included_game_count": 1,
        "included_game_ids": [game_id],
        "game_corpus_sha256": "corpus-hash",
        "season_simulation_configuration": {"simulation_version": "v1"},
        "prior_model_version": prior_version,
        "prior_artifact_sha256": f"prior-{prior_version}",
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with (root / "rankings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["team_id", "subdivision"])
        writer.writeheader()
        writer.writerows([{"team_id": "1", "subdivision": "fbs"}])
    with (root / "included_games.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id"])
        writer.writeheader()
        writer.writerow({"id": game_id})


def test_week4_pair_requires_identical_evidence_boundary(tmp_path: Path) -> None:
    context_1_2 = tmp_path / "context-1.2"
    context_1_3 = tmp_path / "context-1.3"
    _write_pair_artifact(context_1_2, prior_version="1.2", game_id="game-1")
    _write_pair_artifact(context_1_3, prior_version="1.3", game_id="game-1")

    result = validate_week4_pair(context_1_2, context_1_3)

    assert result["prior_model_versions"] == ["1.2", "1.3"]
    assert result["included_game_ids"] == ["game-1"]

    (context_1_3 / "included_games.csv").write_text(
        "id\ngame-2\n", encoding="utf-8"
    )
    with pytest.raises(ContextComparisonError, match="included_games"):
        validate_week4_pair(context_1_2, context_1_3)
