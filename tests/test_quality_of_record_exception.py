from __future__ import annotations

import copy
import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from gippyrank import quality_of_record_snapshot as qor
from gippyrank.artifact_hashes import INCLUDED_GAME_FIELDS
from gippyrank.performance_snapshot import _as_snapshot, _logical_snapshot_id
from gippyrank.posterior.snapshots import _read_included_game_rows, sha256
from gippyrank.quality_of_record_evidence import (
    CAMPBELL_EXCEPTION_SOURCES,
    CAMPBELL_GAME_ID,
    INFERENCE_GAMES_FILE,
    quality_of_record_evidence,
)
from gippyrank.site_data import SiteDataValidationError, _validate_performance_source

ROOT = Path(__file__).resolve().parents[1]
SOURCES = list(CAMPBELL_EXCEPTION_SOURCES)


@pytest.fixture(scope="module")
def built_sources(tmp_path_factory: pytest.TempPathFactory):
    output = tmp_path_factory.mktemp("campbell-qor")
    results = {}
    before = {}
    for snapshot_id in SOURCES:
        context = _as_snapshot(
            ROOT / "data/processed/snapshots/2026" / snapshot_id / "predictive/context"
        )
        _, base = _logical_snapshot_id(snapshot_id)
        play = ROOT / "data/processed/snapshots/2026" / base / "performance"
        for directory in (context.directory, play):
            before.update(
                {path: sha256(path) for path in directory.glob("*") if path.is_file()}
            )
        factors = []
        infer = qor.infer_quality_of_record

        def capture(
            teams, games, likelihood, *, _factors=factors, _infer=infer, **configuration
        ):
            _factors.extend(
                (game.game_id, game.home_points, game.away_points) for game in games
            )
            return _infer(teams, games, likelihood, **configuration)

        with patch.object(qor, "infer_quality_of_record", side_effect=capture):
            record = qor.build_quality_of_record_snapshot(
                context, root=ROOT, output_root=output
            )
        results[snapshot_id] = context, record, factors
    assert all(sha256(path) == value for path, value in before.items())
    return results


@pytest.mark.parametrize("snapshot_id", SOURCES)
def test_scoped_replay_excludes_only_the_malformed_factor(
    snapshot_id, built_sources
) -> None:
    context, record, factors = built_sources[snapshot_id]
    qor.validate_quality_of_record_against_context(context, record, root=ROOT)
    _validate_performance_source(record.metadata, ROOT, record.directory)
    assert (context.directory / "included_games.csv").read_bytes() == (
        record.directory / "included_games.csv"
    ).read_bytes()
    source_rows = _read_included_game_rows(context.directory / "included_games.csv")
    inference_rows = _read_included_game_rows(record.directory / INFERENCE_GAMES_FILE)
    assert inference_rows == [
        row for row in source_rows if row["id"] != CAMPBELL_GAME_ID
    ]
    assert [item[0] for item in factors] == [row["id"] for row in inference_rows]
    assert not any(item[0] == CAMPBELL_GAME_ID for item in factors)
    legitimate = [item for item in factors if item[0] == "401917058"]
    assert legitimate == ([] if snapshot_id == SOURCES[0] else [("401917058", 28, 19)])
    assert CAMPBELL_GAME_ID in record.metadata["included_game_ids"]
    evidence = record.metadata["quality_of_record_evidence"]
    assert evidence["source_game_count"] == len(source_rows)
    assert evidence["inference_game_count"] == len(source_rows) - 1
    assert evidence["exclusions"][0]["game_id"] == CAMPBELL_GAME_ID
    diagnostics = json.loads((record.directory / "diagnostics.json").read_text())
    assert diagnostics["quality_of_record_evidence"] == evidence
    assert diagnostics["raw_game_factor_count"] == len(inference_rows)


@pytest.mark.parametrize("snapshot_id", SOURCES)
def test_scoped_replay_is_deterministic(
    snapshot_id, built_sources, tmp_path: Path
) -> None:
    context, first, _ = built_sources[snapshot_id]
    second = qor.build_quality_of_record_snapshot(
        context, root=ROOT, output_root=tmp_path
    )
    for name in (
        "metadata.json",
        "rankings.csv",
        "rankings.json",
        "posterior_pmfs.csv",
        "included_games.csv",
        INFERENCE_GAMES_FILE,
    ):
        assert (first.directory / name).read_bytes() == (
            second.directory / name
        ).read_bytes()


@pytest.mark.parametrize(
    "tamper", ["undeclared", "different_game", "extra_exclusion", "inference_ids"]
)
def test_exclusion_declaration_is_exact(tamper, built_sources) -> None:
    context, record, _ = built_sources[SOURCES[2]]
    metadata = copy.deepcopy(record.metadata)
    if tamper == "undeclared":
        metadata.pop("quality_of_record_evidence")
    elif tamper == "different_game":
        metadata["quality_of_record_evidence"]["exclusions"][0]["game_id"] = "401917058"
    elif tamper == "extra_exclusion":
        metadata["quality_of_record_evidence"]["exclusions"].append(
            {"game_id": "401917058"}
        )
    else:
        metadata["quality_of_record_evidence"]["inference_game_ids"].remove("401917058")
    with pytest.raises(ValueError, match="declaration mismatch"):
        qor.validate_quality_of_record_against_context(
            context, replace(record, metadata=metadata), root=ROOT
        )
    with pytest.raises(SiteDataValidationError, match="declaration mismatch"):
        _validate_performance_source(metadata, ROOT, record.directory)


@pytest.mark.parametrize(
    "tamper", ["row", "metadata_hash", "missing_record", "other_game"]
)
def test_changed_frozen_source_is_not_eligible_for_the_exception(
    tamper, built_sources
) -> None:
    context, _, _ = built_sources[SOURCES[2]]
    rows = _read_included_game_rows(context.directory / "included_games.csv")
    metadata_hash = sha256(context.directory / "metadata.json")
    if tamper == "row":
        next(row for row in rows if row["id"] == CAMPBELL_GAME_ID)["homePoints"] = "1"
    elif tamper == "metadata_hash":
        metadata_hash = "0" * 64
    elif tamper == "missing_record":
        rows = [row for row in rows if row["id"] != CAMPBELL_GAME_ID]
    else:
        next(row for row in rows if row["id"] == "401917058")["awayPoints"] = "18"
    with pytest.raises(ValueError, match="no longer matches its frozen source"):
        quality_of_record_evidence(
            context.metadata, rows, source_metadata_sha256=metadata_hash
        )


@pytest.mark.parametrize("game_id", [CAMPBELL_GAME_ID, "unexpected-tie"])
def test_unlisted_winnerless_source_still_fails(game_id, built_sources) -> None:
    context, _, _ = built_sources[SOURCES[0]]
    rows = _read_included_game_rows(context.directory / "included_games.csv")
    row = next(row for row in rows if row["id"] == CAMPBELL_GAME_ID)
    row["id"] = game_id
    source = {**context.metadata, "snapshot_id": "unlisted-historical-context"}
    with pytest.raises(ValueError, match=f"game {game_id} has no binary winner"):
        quality_of_record_evidence(
            source,
            [row],
            source_metadata_sha256=sha256(context.directory / "metadata.json"),
        )


@pytest.mark.parametrize(
    "change", ["additional_exclusion", "reincluded_malformed", "diagnostics"]
)
def test_artifact_discrepancies_fail_validation(
    change, built_sources, tmp_path: Path
) -> None:
    context, original, _ = built_sources[SOURCES[2]]
    directory = tmp_path / "record"
    shutil.copytree(original.directory, directory)
    record = replace(original, directory=directory)
    if change == "diagnostics":
        path = directory / "diagnostics.json"
        diagnostics = json.loads(path.read_text())
        diagnostics["quality_of_record_evidence"]["exclusions"] = []
        path.write_text(json.dumps(diagnostics))
    else:
        rows = _read_included_game_rows(directory / INFERENCE_GAMES_FILE)
        if change == "additional_exclusion":
            rows = [row for row in rows if row["id"] != "401917058"]
        else:
            rows = _read_included_game_rows(directory / "included_games.csv")
        with (directory / INFERENCE_GAMES_FILE).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=INCLUDED_GAME_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    with pytest.raises(
        ValueError, match="inference (game artifact|diagnostics) mismatch"
    ):
        qor.validate_quality_of_record_against_context(context, record, root=ROOT)
    with pytest.raises(
        SiteDataValidationError, match="inference (game artifact|diagnostics) mismatch"
    ):
        _validate_performance_source(record.metadata, ROOT, directory)


def test_scoped_historical_backfill_is_idempotent_and_preserves_play(
    built_sources, tmp_path: Path
) -> None:
    from gippyrank.performance_backfill import backfill_performance_variants
    from gippyrank.posterior.snapshots import _prior_path

    entries = []
    slots = []
    for index, snapshot_id in enumerate(SOURCES):
        context, _, _ = built_sources[snapshot_id]
        _, base = _logical_snapshot_id(snapshot_id)
        play = ROOT / "data/processed/snapshots/2026" / base / "performance"
        slot = f"2026-09-0{index + 6}"
        slots.append({"id": slot, "status": "official" if index == 2 else "temporary"})
        for source in (context.directory, play):
            relative = source.relative_to(ROOT)
            destination = tmp_path / relative
            destination.mkdir(parents=True, exist_ok=True)
            # The binary replay needs no mutable corpus or margin posterior.
            for name in (
                "metadata.json",
                "diagnostics.json",
                "rankings.csv",
                "rankings.json",
                "included_games.csv",
            ):
                shutil.copyfile(source / name, destination / name)
            if source == play:
                shutil.copyfile(
                    source / "posterior_pmfs.csv", destination / "posterior_pmfs.csv"
                )
            entries.append(
                {
                    "source": relative.as_posix(),
                    "publication_slot": slot,
                    "display_label": slot,
                }
            )
    for source in (
        _prior_path(ROOT, "context", 2026, "1.2"),
        _prior_path(ROOT, "context", 2026, "1.4"),
        ROOT / "data/processed/posterior/historical_likelihood_v1.json",
    ):
        destination = tmp_path / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    config_path = tmp_path / "site/publish_config.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "snapshots": entries,
                "publication_slots": slots,
                "default_publication_slot": slots[-1]["id"],
            }
        )
    )
    frozen = {
        path: sha256(path) for path in (tmp_path / "data").rglob("*") if path.is_file()
    }
    first = backfill_performance_variants(root=tmp_path)
    assert first["status"] == "complete"
    assert first["publication_count"] == 3 and first["blockers"] == []
    before = {path: sha256(path) for path in tmp_path.rglob("*") if path.is_file()}
    assert backfill_performance_variants(root=tmp_path) == first
    assert all(sha256(path) == value for path, value in before.items())
    assert all(sha256(path) == value for path, value in frozen.items())
