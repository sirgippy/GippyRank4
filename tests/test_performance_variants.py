from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_quality_of_record import _context_snapshot

from gippyrank.performance_backfill import (
    PerformanceBackfillError,
    backfill_performance_variants,
)
from gippyrank.performance_snapshot import build_performance_snapshot
from gippyrank.posterior.snapshots import Snapshot
from gippyrank.quality_of_record_snapshot import build_quality_of_record_snapshot
from gippyrank.site_data import PublicationComparison, resolve_previous_official


def _publish(root: Path, snapshots: list[Snapshot]) -> None:
    path = root / "site/publish_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "default_publication_slot": "2026-one",
                "publication_slots": [{"id": "2026-one", "status": "temporary"}],
                "snapshots": [
                    {
                        "source": snapshot.directory.relative_to(root).as_posix(),
                        "display_label": "One",
                        "publication_slot": "2026-one",
                    }
                    for snapshot in snapshots
                ],
            }
        )
    )


def _legacy(context: Snapshot) -> Snapshot:
    metadata = dict(context.metadata)
    for field in (
        "prior_pmfs_sha256",
        "historical_likelihood_provenance",
        "historical_likelihood_parameters",
        "included_game_rows_sha256",
        "included_game_ids_sha256",
        "posterior_inference_configuration",
    ):
        metadata.pop(field)
    # The retained legacy publications all use the pinned artifact, not supplied
    # research parameters; this fixture preserves the same strict hash contract.
    metadata["historical_likelihood_sha256"] = (
        "89eff21a304939a62beed6fd58dd642e0e4dbba698d7bdaa9aaca9f09f2de153"
    )
    (context.directory / "metadata.json").write_text(json.dumps(metadata))
    return replace(context, metadata=metadata)


def test_backfill_preserves_play_and_is_idempotent(tmp_path: Path) -> None:
    context = _context_snapshot(tmp_path)
    play = build_performance_snapshot(context, root=tmp_path)
    _publish(tmp_path, [context, play])
    before = {
        path: path.read_bytes()
        for snapshot in (context, play)
        for path in snapshot.directory.glob("*")
        if path.is_file()
    }
    first = backfill_performance_variants(root=tmp_path)
    assert first["publication_count"] == 1
    config = (tmp_path / "site/publish_config.json").read_bytes()
    files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    second = backfill_performance_variants(root=tmp_path)
    assert first == second
    assert config == (tmp_path / "site/publish_config.json").read_bytes()
    assert all(path.read_bytes() == value for path, value in files.items())
    assert all(path.read_bytes() == value for path, value in before.items())
    assert first["publications"][0]["publication_status"] == "temporary"


def test_missing_input_reports_publication_and_preserves_config(tmp_path: Path) -> None:
    context = _context_snapshot(tmp_path)
    _publish(tmp_path, [context])
    path = tmp_path / "site/publish_config.json"
    before = path.read_bytes()
    (context.directory / "included_games.csv").unlink()
    with pytest.raises(PerformanceBackfillError, match="2026-one.*included_games"):
        backfill_performance_variants(root=tmp_path)
    assert before == path.read_bytes()


def test_legacy_identity_uses_only_retained_inputs(tmp_path: Path) -> None:
    import shutil

    from gippyrank.methodology import HISTORICAL_LIKELIHOOD_SHA256

    context = _legacy(_context_snapshot(tmp_path))
    likelihood = (
        Path(__file__).resolve().parents[1]
        / "data/processed/posterior/historical_likelihood_v1.json"
    )
    destination = tmp_path / "data/processed/posterior/historical_likelihood_v1.json"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(likelihood, destination)
    frozen = (context.directory / "metadata.json").read_bytes()
    # Neither the mutable corpus nor the Context margin posterior is an input.
    (tmp_path / "data/processed/cfbd/games.csv").unlink()
    (context.directory / "posterior_pmfs.csv").unlink()
    record = build_quality_of_record_snapshot(context, root=tmp_path)
    assert (
        record.metadata["historical_likelihood_sha256"] == HISTORICAL_LIKELIHOOD_SHA256
    )
    assert record.metadata["included_game_ids"] == ["g"]
    assert (
        record.metadata["prior_model_version"]
        == context.metadata["prior_model_version"]
    )
    assert (context.directory / "metadata.json").read_bytes() == frozen


def test_official_comparisons_never_cross_performance_variants() -> None:
    play = PublicationComparison(
        2026, "performance", None, "first", "official", 0, "play", "First"
    )
    record = replace(
        play, snapshot_id="record", performance_variant="quality_of_record"
    )
    results = replace(
        play, snapshot_id="results", performance_variant="quality_of_results"
    )
    current_results = replace(results, publication_slot="second", publication_order=1)
    assert resolve_previous_official(current_results, [play, record]) is None
    assert (
        resolve_previous_official(current_results, [play, record, results]) == results
    )
    current = replace(record, publication_slot="second", publication_order=1)
    assert resolve_previous_official(current, [play]) is None
    assert resolve_previous_official(current, [play, record]) == record
    assert (
        resolve_previous_official(
            replace(current, performance_variant="quality_of_play"), [record, play]
        )
        == play
    )


def test_partial_backfill_reports_each_blocked_slot(tmp_path: Path) -> None:
    import shutil

    context = _context_snapshot(tmp_path)
    blocked_path = tmp_path / "blocked-context"
    shutil.copytree(context.directory, blocked_path)
    metadata = {**context.metadata, "snapshot_id": "2026-blocked-context"}
    (blocked_path / "metadata.json").write_text(json.dumps(metadata))
    (blocked_path / "included_games.csv").unlink()
    blocked = Snapshot(metadata["snapshot_id"], blocked_path, metadata)
    _publish(tmp_path, [context, blocked])
    path = tmp_path / "site/publish_config.json"
    config = json.loads(path.read_text())
    config["publication_slots"].append({"id": "2026-two", "status": "official"})
    config["snapshots"][1]["publication_slot"] = "2026-two"
    path.write_text(json.dumps(config))
    report = backfill_performance_variants(root=tmp_path, allow_blocked=True)
    assert report["status"] == "blocked"
    assert report["eligible_publication_count"] == 2
    assert report["publication_count"] == 1
    assert report["blocked_publication_count"] == 1
    assert "2026-two / 2026-blocked-context" in report["blockers"][0]
    assert len(json.loads(path.read_text())["snapshots"]) == 4
