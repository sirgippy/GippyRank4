from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gippyrank.performance_snapshot import (
    PerformanceSnapshotValidationError,
    build_performance_snapshot,
    strip_context_prior,
    validate_performance_against_context,
)
from gippyrank.performance_v1 import remove_focal_prior
from gippyrank.posterior.snapshots import Snapshot, load_teams

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = ROOT / (
    "data/processed/snapshots/2026/"
    "2026-weekly-2026-09-06T17-34-56.895103Z-context/predictive/context"
)
PERFORMANCE_PATH = ROOT / (
    "data/processed/snapshots/2026/"
    "2026-weekly-2026-09-06T17-34-56.895103Z/performance"
)


def _snapshot(path: Path) -> Snapshot:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return Snapshot(metadata["snapshot_id"], path, metadata)


def test_production_ratio_matches_frozen_research_helper(tmp_path: Path) -> None:
    _teams, prior_rows, _ = load_teams(ROOT, 2026, "context")
    prior = {team_id: np.asarray(json.loads(row["pmf"]), dtype=float) for team_id, row in prior_rows.items()}
    with (CONTEXT_PATH / "posterior_pmfs.csv").open(newline="", encoding="utf-8") as handle:
        posterior_rows = list(csv.DictReader(handle))
    posterior: dict[str, list[float]] = {}
    for row in posterior_rows:
        if row["team_id"] in prior:
            posterior.setdefault(row["team_id"], []).append(float(row["probability"]))
    production = build_performance_snapshot(
        CONTEXT_PATH, root=ROOT, output_root=tmp_path
    )
    with (production.directory / "posterior_pmfs.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    with (production.directory / "rankings.csv").open(newline="", encoding="utf-8") as handle:
        rated_ids = {row["team_id"] for row in csv.DictReader(handle) if row["rated"] == "True"}
    actual: dict[str, list[float]] = {}
    for row in rows:
        actual.setdefault(row["team_id"], []).append(float(row["probability"]))
    for team_id, values in posterior.items():
        if team_id not in rated_ids:
            continue
        expected = remove_focal_prior(np.asarray(values), prior[team_id])
        assert np.allclose(actual[team_id], expected, atol=1e-12, rtol=0)


def test_production_zero_game_teams_are_uniform_and_unrated(tmp_path: Path) -> None:
    snapshot = build_performance_snapshot(CONTEXT_PATH, root=ROOT, output_root=tmp_path)
    with (snapshot.directory / "rankings.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    unrated = [row for row in rows if row["rated"] == "False"]
    assert len(unrated) == 7
    assert all(row["display_rank"] == "NR" and row["eligible_games"] == "0" for row in unrated)
    probabilities = [
        float(row["probability"])
        for row in csv.DictReader((snapshot.directory / "posterior_pmfs.csv").open())
        if row["team_id"] == unrated[0]["team_id"]
    ]
    assert np.allclose(probabilities, np.full(138, 1 / 138), atol=1e-15, rtol=0)


def test_history_source_is_rejected(tmp_path: Path) -> None:
    context = _snapshot(CONTEXT_PATH)
    history_metadata = dict(context.metadata)
    history_metadata["prior_family"] = "history"
    history = replace(context, metadata=history_metadata)
    with pytest.raises(PerformanceSnapshotValidationError, match="History-derived"):
        build_performance_snapshot(history, root=ROOT, output_root=tmp_path)


@pytest.mark.parametrize("field", ["effective_cutoff", "included_game_ids", "game_corpus_sha256", "prior_artifact_sha256"])
def test_context_performance_evidence_mismatch_is_rejected(field: str) -> None:
    context = _snapshot(CONTEXT_PATH)
    performance = _snapshot(PERFORMANCE_PATH)
    metadata = dict(performance.metadata)
    value = metadata[field]
    metadata[field] = f"mismatch-{value}" if isinstance(value, str) else ["mismatch"]
    mismatched = replace(performance, metadata=metadata)
    with pytest.raises(PerformanceSnapshotValidationError, match="evidence mismatch"):
        validate_performance_against_context(context, mismatched)


def test_production_path_does_not_use_explicit_neutralization() -> None:
    source = (ROOT / "src/gippyrank/performance_snapshot.py").read_text(encoding="utf-8")
    assert "explicit_neutralized_target" not in source


def test_ratio_rejects_zero_prior_mass_and_malformed_inputs() -> None:
    with pytest.raises(PerformanceSnapshotValidationError, match="zero mass"):
        strip_context_prior(np.array([0.5, 0.5]), np.array([1.0, 0.0]))
    with pytest.raises(PerformanceSnapshotValidationError, match="refusing to renormalize"):
        strip_context_prior(np.array([0.4, 0.4]), np.array([0.5, 0.5]))
