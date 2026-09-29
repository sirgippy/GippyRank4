"""Guard the retained production corpus against an unmaterialized artifact."""

from __future__ import annotations

import csv
import json
import runpy
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from gippyrank import site_data
from gippyrank.methodology import (
    HISTORICAL_LIKELIHOOD_SHA256,
    RETROSPECTIVE_CONDITIONING,
)
from gippyrank.posterior.engine import Team
from gippyrank.posterior.retained import (
    load_retained_posterior,
    preserve_equivalent_retrospective_records,
)
from gippyrank.posterior.snapshots import (
    load_pinned_likelihood,
    posterior_pmfs_sha256,
    sha256,
)
from gippyrank.site_data import SiteDataValidationError

ROOT = Path(__file__).resolve().parents[1]
BACKFILL_SCRIPT = runpy.run_path(str(ROOT / "scripts/backfill_retrospective_game_expectations.py"))
backfill = BACKFILL_SCRIPT["backfill"]
sync_performance_sources = BACKFILL_SCRIPT["sync_performance_sources"]


def test_retained_likelihood_bytes_are_pinned(tmp_path: Path) -> None:
    source = ROOT / "data/processed/posterior/historical_likelihood_v1.json"
    assert sha256(source) == HISTORICAL_LIKELIHOOD_SHA256
    assert load_pinned_likelihood(source).scale > 0
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="likelihood bytes differ"):
        load_pinned_likelihood(changed)


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
        assert metadata["retrospective_game_expectations_version"] == "2.0"
        assert artifact["historical_likelihood_sha256"] == HISTORICAL_LIKELIHOOD_SHA256
        assert artifact["conditioning"] == RETROSPECTIVE_CONDITIONING
        assert artifact["source_snapshot_id"] == metadata["snapshot_id"]
        assert artifact["posterior_pmfs_sha256"] == metadata["posterior_pmfs_sha256"]
        assert "runtime_seconds" not in artifact["inference"]
        assert set(artifact["games"]) == set(metadata["included_game_ids"])
        assert len(artifact["games"]) == metadata["included_game_count"]


def test_retained_source_availability_does_not_relabel_legacy_cutoff() -> None:
    source = ROOT / "data/processed/snapshots/2026/2026-weekly-2026-09-27T12-27-35.698895Z-context-v1.3/predictive/context"
    metadata = json.loads((source / "metadata.json").read_text())
    assert metadata["source_retrieved_at_contract"] == "legacy_first_response"
    assert metadata["combined_source_available_at"] == max(metadata["source_retrieval_times"].values())
    assert metadata["source_retrieved_at"] == metadata["effective_cutoff"]
    assert metadata["combined_source_available_at"] > metadata["effective_cutoff"]


def test_empty_preseason_site_projection_keeps_projection_contract() -> None:
    artifact = json.loads((ROOT / "site/data/team-seasons/2026-preseason-context.json").read_text())
    retrospective = artifact["retrospective_game_expectations"]
    assert retrospective["artifact_kind"] == "retrospective_game_expectations_site_projection"
    assert retrospective["site_projection_version"] == "1.0"
    assert retrospective["coverage"] == "fbs_team_schedules"
    assert retrospective["published_game_ids"] == []
    assert retrospective["games"] == {}


def test_static_publication_rejects_unverified_likelihood() -> None:
    source = ROOT / "data/processed/snapshots/2026/2026-preseason-context/predictive/context"
    artifact = json.loads((source / "team_seasons.json").read_text())
    metadata = json.loads((source / "metadata.json").read_text())
    metadata["historical_likelihood_sha256"] = None
    artifact["retrospective_game_expectations"]["historical_likelihood_sha256"] = None
    with pytest.raises(SiteDataValidationError, match="not pinned V1"):
        site_data._validate_retrospective_game_expectations(
            artifact, metadata, set(), artifact["teams"]
        )


@pytest.mark.parametrize("provenance", ["supplied_parameters", "loaded_from_artifact"])
def test_published_v1_requires_pinned_likelihood_for_every_provenance(
    provenance: str,
) -> None:
    source = ROOT / "data/processed/snapshots/2026/2026-preseason-context/predictive/context"
    artifact = json.loads((source / "team_seasons.json").read_text())
    metadata = json.loads((source / "metadata.json").read_text())
    metadata["historical_likelihood_provenance"] = provenance
    if provenance == "loaded_from_artifact":
        metadata["historical_likelihood_sha256"] = "0" * 64
        artifact["retrospective_game_expectations"]["historical_likelihood_sha256"] = "0" * 64
        message = "not pinned V1"
    else:
        message = "research-only"
    with pytest.raises(SiteDataValidationError, match=message):
        site_data._validate_retrospective_game_expectations(
            artifact, metadata, set(), artifact["teams"]
        )


def test_backfill_rejects_untrusted_posterior_rows(tmp_path: Path) -> None:
    teams = [Team("a", "A", "fbs", np.array([0.5, 0.5]))]
    pmfs = {"a": np.array([0.4, 0.6])}
    metadata = {"posterior_pmfs_sha256": posterior_pmfs_sha256(pmfs)}
    (tmp_path / "diagnostics.json").write_text(json.dumps({
        "converged": True, "iterations": 1, "max_message_delta": 0,
        "objective_surrogate": 0, "raw_game_likelihood_count": 0,
        "unique_pairwise_factor_count": 0, "maximum_team_degree": 0,
    }))
    path = tmp_path / "posterior_pmfs.csv"
    def write(rows: list[tuple[str, int, float]]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("team_id", "rank", "probability"))
            writer.writerows(rows)
    write([("a", 1, 0.4), ("a", 2, 0.6)])
    assert np.array_equal(load_retained_posterior(tmp_path, metadata, teams).pmfs["a"], pmfs["a"])
    for rows, message in (
        ([("a", 1, 0.4), ("a", 1, 0.6)], "duplicate posterior rank"),
        ([("a", 1, 0.4)], "incomplete posterior ranks"),
        ([("a", 1, -0.4), ("a", 2, 1.4)], "invalid posterior"),
        ([("a", 1, 0.5), ("a", 2, 0.5)], "disagree with metadata"),
        ([("b", 1, 0.4), ("a", 2, 0.6)], "unexpected posterior team"),
    ):
        write(rows)
        with pytest.raises(ValueError, match=message):
            load_retained_posterior(tmp_path, metadata, teams)


def test_backfill_preserves_only_equivalent_frozen_records() -> None:
    provenance = {
        "retrospective_game_expectations_version": "2.0",
        "source_snapshot_id": "snapshot",
        "conditioning": RETROSPECTIVE_CONDITIONING,
        "historical_likelihood_sha256": HISTORICAL_LIKELIHOOD_SHA256,
        "posterior_pmfs_sha256": "pmf-hash",
    }
    existing = {**provenance, "games": {
        "stable": {"expected_home_margin": 26.92384615053041, "actual_home_margin": 17},
        "changed": {"expected_home_margin": 12.0, "actual_home_margin": 17},
    }}
    refreshed = {**provenance, "games": {
        "stable": {"expected_home_margin": 26.923846150530405, "actual_home_margin": 17},
        "changed": {"expected_home_margin": 13.0, "actual_home_margin": 17},
    }}
    preserve_equivalent_retrospective_records(existing, refreshed)
    assert refreshed["games"]["stable"] is existing["games"]["stable"]
    assert refreshed["games"]["changed"] is not existing["games"]["changed"]
    different_probability = {**provenance, "games": {
        "stable": {"expected_home_margin": 0.50000000001, "actual_home_margin": 17},
        "changed": refreshed["games"]["changed"],
    }}
    preserve_equivalent_retrospective_records(existing, different_probability)
    assert different_probability["games"]["stable"] is not existing["games"]["stable"]


def test_forced_preseason_backfill_is_byte_identical(tmp_path: Path) -> None:
    source = ROOT / "data/processed/snapshots/2026/2026-preseason-context/predictive/context"
    copied = tmp_path / "preseason"
    shutil.copytree(source, copied)
    backfill(copied)
    first = (copied / "team_seasons.json").read_bytes()
    first_metadata = (copied / "metadata.json").read_bytes()
    backfill(copied)
    assert (copied / "team_seasons.json").read_bytes() == first
    assert (copied / "metadata.json").read_bytes() == first_metadata


def test_targeted_backfill_syncs_only_dependent_performance(tmp_path: Path) -> None:
    base = tmp_path / "data/processed/snapshots/2026"
    changed = base / "changed/predictive/context"
    other = base / "other/predictive/context"
    for source, value in ((changed, "new"), (other, "other")):
        source.mkdir(parents=True)
        (source / "metadata.json").write_text(json.dumps({
            "combined_source_available_at": value,
            "source_retrieved_at_contract": "combined_latest",
        }))
        (source / "team_seasons.json").write_text(value)
    for name in ("changed", "other"):
        performance = base / name / "performance"
        performance.mkdir()
        (performance / "metadata.json").write_text(json.dumps({
            "source_context_path": str((base / name / "predictive/context").relative_to(tmp_path)),
            "source_context_metadata_sha256": "old",
        }))
        (performance / "team_seasons.json").write_text("old")
    sync_performance_sources({changed.resolve()}, root=tmp_path)
    assert (base / "changed/performance/team_seasons.json").read_text() == "new"
    assert (base / "other/performance/team_seasons.json").read_text() == "old"
    metadata = json.loads((base / "changed/performance/metadata.json").read_text())
    assert metadata["source_context_metadata_sha256"] == sha256(changed / "metadata.json")


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

    prose_edit = deepcopy(original)
    prose_edit["retrospective_game_expectations"]["interpretation"] = (
        "A full-posterior hindsight comparison."
    )
    validate(prose_edit)
    for field, value, message in (
        ("conditioning", "leave_one_out", "conditioning"),
        ("historical_likelihood_sha256", "0" * 64, "likelihood hash"),
    ):
        changed = deepcopy(original)
        changed["retrospective_game_expectations"][field] = value
        with pytest.raises(SiteDataValidationError, match=message):
            validate(changed)
