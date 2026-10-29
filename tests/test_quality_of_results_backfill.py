from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_performance_variants import _publish
from test_quality_of_record import _context_snapshot

from gippyrank.quality_of_results_backfill import (
    ResultsBackfillError,
    backfill_quality_of_results,
)
from gippyrank.quality_of_results_snapshot import build_quality_of_results_snapshot
from gippyrank.site_data import PublishedSnapshot, _retained_publications


def test_retention_inventory_includes_pinned_and_new_interims() -> None:
    items = [
        PublishedSnapshot(Path(str(i)), str(i), str(i), status, i, pinned)
        for i, (status, pinned) in enumerate(
            [
                ("temporary", False),
                ("temporary", True),
                ("official", False),
                ("temporary", False),
            ]
        )
    ]
    retained, retired = _retained_publications(
        items, {item.source: {"season": 2026} for item in items}
    )
    assert [item.publication_slot for item in retained] == ["1", "2", "3"]
    assert [item.publication_slot for item in retired] == ["0"]


def test_backfill_generates_and_reuses_without_changes(tmp_path: Path) -> None:
    context = _context_snapshot(tmp_path)
    _publish(tmp_path, [context])
    original = {p: p.read_bytes() for p in context.directory.iterdir() if p.is_file()}
    first = backfill_quality_of_results(root=tmp_path)
    assert first["run_generated_count"] == 1
    assert first["publications"][0]["margin_weight"] == 0.4
    config = tmp_path / "site/publish_config.json"
    manifest = tmp_path / "data/processed/quality_of_results_backfill/manifest.json"
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    second = backfill_quality_of_results(root=tmp_path)
    assert second["run_generated_count"] == 0
    assert second["run_reused_count"] == 1
    assert all(p.read_bytes() == value for p, value in {**before, **original}.items())
    assert len(json.loads(config.read_text())["snapshots"]) == 2
    assert json.loads(manifest.read_text())["status"] == "complete"


def test_unregistered_existing_result_is_reused_and_invalid_fails(
    tmp_path: Path,
) -> None:
    context = _context_snapshot(tmp_path)
    result = build_quality_of_results_snapshot(context, root=tmp_path)
    before = {p: p.read_bytes() for p in result.directory.iterdir()}
    _publish(tmp_path, [context])
    report = backfill_quality_of_results(root=tmp_path)
    assert report["run_reused_count"] == 1
    assert all(p.read_bytes() == contents for p, contents in before.items())
    config = tmp_path / "site/publish_config.json"
    config_before = config.read_bytes()
    (result.directory / "rankings.csv").write_text("corrupt")
    with pytest.raises(ResultsBackfillError, match="2026-one"):
        backfill_quality_of_results(root=tmp_path)
    assert config.read_bytes() == config_before


def test_missing_context_evidence_blocks_registration(tmp_path: Path) -> None:
    context = _context_snapshot(tmp_path)
    _publish(tmp_path, [context])
    config = tmp_path / "site/publish_config.json"
    original = config.read_bytes()
    (context.directory / "included_games.csv").unlink()
    with pytest.raises(ResultsBackfillError, match="2026-one"):
        backfill_quality_of_results(root=tmp_path)
    assert config.read_bytes() == original
    manifest = json.loads(
        (
            tmp_path / "data/processed/quality_of_results_backfill/manifest.json"
        ).read_text()
    )
    assert manifest["blocked_publication_count"] == 1


def test_published_results_match_frozen_issue_269_references() -> None:
    import gzip

    import numpy as np

    from gippyrank.performance_snapshot import _pmfs

    root = Path(__file__).resolve().parents[1]
    references = sorted(
        (root / "data/research/issue_269_lambda_040/networks").glob(
            "baseline-2026*.json.gz"
        )
    )
    assert len(references) == 3
    for path in references:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            reference = json.load(handle)
        source = reference["identity"]["source_snapshot"]["context_snapshot_id"]
        pmfs = _pmfs(
            root
            / "data/processed/snapshots/2026"
            / source
            / "performance/quality_of_results/posterior_pmfs.csv"
        )
        assert pmfs.keys() == reference["pmfs"].keys()
        for team_id, pmf in pmfs.items():
            np.testing.assert_allclose(
                pmf, reference["pmfs"][team_id], atol=1e-12, rtol=0
            )


def test_blocked_slot_does_not_register_other_staged_results(tmp_path: Path) -> None:
    import shutil

    from gippyrank.posterior.snapshots import Snapshot

    context = _context_snapshot(tmp_path)
    blocked_path = tmp_path / "blocked-context"
    shutil.copytree(context.directory, blocked_path)
    metadata = {**context.metadata, "snapshot_id": "2026-blocked-context"}
    (blocked_path / "metadata.json").write_text(json.dumps(metadata))
    (blocked_path / "included_games.csv").unlink()
    blocked = Snapshot(metadata["snapshot_id"], blocked_path, metadata)
    _publish(tmp_path, [context, blocked])
    config_path = tmp_path / "site/publish_config.json"
    config = json.loads(config_path.read_text())
    config["publication_slots"][0]["status"] = "official"
    config["publication_slots"].append({"id": "2026-two", "status": "official"})
    config["snapshots"][1]["publication_slot"] = "2026-two"
    config_path.write_text(json.dumps(config))
    original = config_path.read_bytes()
    with pytest.raises(ResultsBackfillError, match="2026-two"):
        backfill_quality_of_results(root=tmp_path)
    assert config_path.read_bytes() == original
    assert not list(tmp_path.glob("**/quality_of_results/metadata.json"))
    report = backfill_quality_of_results(root=tmp_path, allow_blocked=True)
    assert report["status"] == "blocked"
    assert report["publication_count"] == report["blocked_publication_count"] == 1
    assert len(json.loads(config_path.read_text())["snapshots"]) == 3
