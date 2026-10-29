from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from test_quality_of_record import _context_snapshot, _likelihood, _teams

from gippyrank import quality_of_results_snapshot
from gippyrank.artifact_hashes import posterior_pmfs_sha256
from gippyrank.performance_snapshot import (
    PerformanceSnapshotValidationError,
    _pmfs,
    build_performance_snapshot,
)
from gippyrank.posterior.engine import Game
from gippyrank.posterior.snapshots import Snapshot, sha256
from gippyrank.quality_of_record_snapshot import build_quality_of_record_snapshot
from gippyrank.quality_of_results import infer_quality_of_results
from gippyrank.quality_of_results_snapshot import (
    build_quality_of_results_snapshot,
    validate_quality_of_results_against_context,
)
from gippyrank.research.quality_of_results import (
    infer_quality_of_results as research_infer,
)


def _metadata(snapshot: Snapshot, **changes: object) -> Snapshot:
    metadata = {**snapshot.metadata, **changes}
    (snapshot.directory / "metadata.json").write_text(json.dumps(metadata))
    return replace(snapshot, metadata=metadata)


def _rehash(snapshot: Snapshot, name: str) -> Snapshot:
    hashes = {
        **snapshot.metadata["artifact_sha256"],
        name: sha256(snapshot.directory / name),
    }
    return _metadata(snapshot, artifact_sha256=hashes)


def test_generation_matches_research_and_keeps_other_variants_unchanged(
    tmp_path: Path,
) -> None:
    context = _context_snapshot(tmp_path)
    before = {
        path: sha256(path) for path in context.directory.iterdir() if path.is_file()
    }
    play = build_performance_snapshot(context, root=tmp_path)
    record = build_quality_of_record_snapshot(context, root=tmp_path)
    existing = {
        path: sha256(path)
        for directory in (play.directory, record.directory)
        for path in directory.iterdir()
        if path.is_file()
    }
    # Current acquisition files are deliberately unavailable; only frozen inputs count.
    (tmp_path / "data/processed/cfbd/games.csv").unlink()
    generation = datetime(2026, 10, 29, tzinfo=UTC)
    result = build_quality_of_results_snapshot(
        context, root=tmp_path, generation_timestamp=generation
    )
    validate_quality_of_results_against_context(
        context.directory, result.directory, root=tmp_path
    )
    assert result.directory == play.directory / "quality_of_results"
    assert result.snapshot_id.endswith("-performance-quality-of-results")
    assert result.metadata["model"] == "quality_of_results_v1"
    assert result.metadata["margin_weight"] == 0.40
    assert result.metadata["generation_timestamp"] == generation.isoformat()
    expected = research_infer(
        _teams(), [Game("g", "a", "b", "fbs", "fbs", 21, 20)], _likelihood(), 0.40
    )
    actual = _pmfs(result.directory / "posterior_pmfs.csv")
    blended = _pmfs(result.directory / "blended_context_posterior_pmfs.csv")
    for team_id in actual:
        np.testing.assert_allclose(
            actual[team_id], expected.pmfs[team_id], atol=1e-12, rtol=0
        )
    for team_id in blended:
        np.testing.assert_allclose(
            blended[team_id],
            expected.blended_context_posterior.pmfs[team_id],
            atol=1e-12,
            rtol=0,
        )
    rows = json.loads((result.directory / "rankings.json").read_text())
    idle = next(row for row in rows if row["team_id"] == "idle")
    assert idle["display_rank"] == "NR" and idle["eligible_evidence_count"] == 0
    np.testing.assert_array_equal(actual["idle"], np.full(3, 1 / 3))
    assert all(
        sha256(path) == digest for path, digest in {**before, **existing}.items()
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        build_quality_of_results_snapshot(context, root=tmp_path)


def test_production_weight_is_not_a_caller_option_and_rematches_keep_parity() -> None:
    games = [
        Game("a-b", "a", "b", "fbs", "fbs", 21, 20),
        Game("b-a", "b", "a", "fbs", "fbs", 20, 14),
        Game("fcs-a", "fcs", "a", "fcs", "fbs", 7, 35, True),
    ]
    result = infer_quality_of_results(_teams(), games, _likelihood())
    reference = research_infer(_teams(), games, _likelihood(), 0.40)
    assert result.blended_context_posterior.raw_game_factor_count == 3
    assert result.blended_context_posterior.unique_pair_factor_count == 2
    for team_id in reference.pmfs:
        np.testing.assert_array_equal(result.pmfs[team_id], reference.pmfs[team_id])
    with pytest.raises(TypeError, match="margin_weight"):
        infer_quality_of_results(_teams(), games, _likelihood(), margin_weight=0.5)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("margin_weight", 0.5),
        ("model", "quality_of_record_v1"),
        ("model_version", "2.0"),
        ("performance_variant", "quality_of_play"),
        ("evidence_mode", "margin"),
        ("source_context_metadata_sha256", "bad"),
        ("prior_pmfs_sha256", "bad"),
        ("snapshot_id", "wrong"),
    ],
)
def test_independent_validator_rejects_corrupt_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    context = _context_snapshot(tmp_path)
    result = build_quality_of_results_snapshot(context, root=tmp_path)
    result = _metadata(result, **{field: value})
    with pytest.raises(PerformanceSnapshotValidationError, match="mismatch"):
        validate_quality_of_results_against_context(
            context.directory, result.directory, root=tmp_path
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "game",
        "prior",
        "population",
        "likelihood",
        "configuration",
        "identity",
        "context_pmf",
        "source_convergence",
    ],
)
def test_builder_rejects_invalid_frozen_inputs(tmp_path: Path, corruption: str) -> None:
    context = _context_snapshot(tmp_path)
    if corruption == "game":
        path = context.directory / "included_games.csv"
        path.write_text(path.read_text().replace("21", "50"))
    elif corruption == "prior":
        path = tmp_path / context.metadata["prior_artifact_path"]
        path.write_text(path.read_text().replace("0.6", "0.5"))
    elif corruption == "population":
        path = context.directory / "rankings.csv"
        path.write_text(
            "\n".join(
                line
                for line in path.read_text().splitlines()
                if not line.startswith("idle,")
            )
            + "\n"
        )
    elif corruption == "likelihood":
        context = _metadata(context, historical_likelihood_sha256="bad")
    elif corruption == "configuration":
        context = _metadata(
            context,
            posterior_inference_configuration={
                **context.metadata["posterior_inference_configuration"],
                "warm_start": True,
            },
        )
    elif corruption == "identity":
        context.metadata.pop("included_game_rows_sha256")
        context = _metadata(context)
    elif corruption == "source_convergence":
        path = context.directory / "diagnostics.json"
        diagnostics = json.loads(path.read_text())
        diagnostics["max_message_delta"] = 0.01
        path.write_text(json.dumps(diagnostics))
    else:
        path = context.directory / "posterior_pmfs.csv"
        lines = path.read_text().splitlines()
        cells = lines[1].split(",")
        cells[-1] = "nan"
        lines[1] = ",".join(cells)
        path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        build_quality_of_results_snapshot(context, root=tmp_path)
    assert not list(tmp_path.glob("**/quality_of_results/metadata.json"))


@pytest.mark.parametrize(
    "corruption", ["count", "summary", "csv", "diagnostics", "support", "strip"]
)
def test_validator_checks_semantics_even_with_updated_file_hash(
    tmp_path: Path, corruption: str
) -> None:
    context = _context_snapshot(tmp_path)
    result = build_quality_of_results_snapshot(context, root=tmp_path)
    if corruption in {"count", "summary"}:
        name = "rankings.json"
        rows = json.loads((result.directory / name).read_text())
        rows[0][
            "eligible_evidence_count" if corruption == "count" else "expected_rank"
        ] += 1
        (result.directory / name).write_text(json.dumps(rows))
    elif corruption == "csv":
        name = "rankings.csv"
        path = result.directory / name
        path.write_text(path.read_text().replace("True", "False"))
    elif corruption == "diagnostics":
        name = "diagnostics.json"
        value = json.loads((result.directory / name).read_text())
        value["converged"] = False
        (result.directory / name).write_text(json.dumps(value))
    else:
        name = "posterior_pmfs.csv"
        path = result.directory / name
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        if corruption == "support":
            rows = [row for row in rows if row["team_id"] != "idle"]
        else:
            for row in rows:
                if row["team_id"] == "a":
                    row["probability"] = str(1 / 3)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        result = _metadata(
            result, posterior_pmfs_sha256=posterior_pmfs_sha256(_pmfs(path))
        )
    result = _rehash(result, name)
    with pytest.raises(PerformanceSnapshotValidationError):
        validate_quality_of_results_against_context(context, result, root=tmp_path)


def test_nonconvergence_does_not_leave_an_output(tmp_path: Path, monkeypatch) -> None:
    context = _context_snapshot(tmp_path)
    monkeypatch.setattr(
        quality_of_results_snapshot,
        "infer_quality_of_results",
        lambda *args, **kwargs: infer_quality_of_results(
            *args, **{**kwargs, "max_iterations": 1}
        ),
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        build_quality_of_results_snapshot(context, root=tmp_path)
    assert not list(tmp_path.glob("**/quality_of_results/metadata.json"))


@pytest.mark.parametrize("legacy", [True, False])
def test_missing_semantic_identities_require_a_declared_legacy_source(
    tmp_path: Path, legacy: bool
) -> None:
    context = _context_snapshot(tmp_path)
    for field in (
        "included_game_rows_sha256",
        "included_game_ids_sha256",
        "prior_pmfs_sha256",
        "posterior_inference_configuration",
    ):
        context.metadata.pop(field)
    context = _metadata(
        context,
        source_retrieved_at_contract="legacy_first_response"
        if legacy
        else "combined_latest",
    )
    before = sha256(context.directory / "metadata.json")
    if legacy:
        result = build_quality_of_results_snapshot(context, root=tmp_path)
        validate_quality_of_results_against_context(context, result, root=tmp_path)
        assert result.metadata["prior_pmfs_sha256"]
        assert sha256(context.directory / "metadata.json") == before
    else:
        with pytest.raises(PerformanceSnapshotValidationError, match="mismatch"):
            build_quality_of_results_snapshot(context, root=tmp_path)


def test_validator_accepts_floating_point_reduction_roundoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context_snapshot(tmp_path)
    result = build_quality_of_results_snapshot(context, root=tmp_path)
    original = quality_of_results_snapshot._rankings

    def different_reduction(*args: object, **kwargs: object) -> list[dict[str, object]]:
        rows = original(*args, **kwargs)
        for row in rows:
            row["expected_rank"] += 1e-13
        return rows

    monkeypatch.setattr(quality_of_results_snapshot, "_rankings", different_reduction)
    validate_quality_of_results_against_context(context, result, root=tmp_path)
