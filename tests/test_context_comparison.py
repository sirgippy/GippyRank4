from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from gippyrank.context_comparison import (
    ContextComparisonError,
    validate_context_backfill,
    validate_week4_pair,
)


def _write_pair_artifact(root: Path, *, prior_version: str, game_id: str) -> None:
    root.mkdir(parents=True)
    metadata = {
        "season": 2026,
        "snapshot_id": f"week-4-context-{prior_version}",
        "snapshot_type": "weekly",
        "source_mode": "current_cached_cfbd",
        "source_kind": "cfbd_api_schedule",
        "source_retrieved_at": "2026-09-19T22:14:03.513809+00:00",
        "source_retrieval_times": {
            "fbs": "2026-09-19T22:14:03.513809+00:00",
            "fcs": "2026-09-19T22:14:04.436222+00:00",
        },
        "source_response_hashes": {"2026.json": "fbs", "2026-fcs.json": "fcs"},
        "requested_cutoff": "2026-09-19T23:59:00+00:00",
        "effective_cutoff": "2026-09-19T23:59:00+00:00",
        "historical_likelihood_version": "V1",
        "included_game_count": 1,
        "included_game_ids": [game_id],
        "included_game_ids_sha256": hashlib.sha256(
            game_id.encode("utf-8")
        ).hexdigest(),
        "game_corpus_sha256": "corpus-hash",
        "fbs_team_count": 1,
        "fbs_team_keys_sha256": hashlib.sha256(b"1").hexdigest(),
        "posterior_inference_configuration": {
            "implementation": "deterministic_damped_loopy_sum_product",
            "max_iterations": 500,
            "tolerance": 1e-9,
            "damping": 0.35,
        },
        "season_simulation_schema_version": "1.0",
        "season_simulation_version": "hierarchical_latent_state_v1",
        "season_simulation_configuration": {"simulation_version": "v1"},
        "excluded_lower_division_games": 0,
        "fcs_fallback_team_ids": [],
        "fcs_fallback_count": 0,
        "fcs_population_size": None,
        "fcs_population_source": None,
        "fcs_fallback_kind": None,
        "fcs_fallback_pmf_semantics": None,
        "lower_division_handling": {
            "excluded_lower_division_games": 0,
            "fcs_fallback_team_ids": [],
            "fcs_fallback_count": 0,
            "fcs_population_size": None,
            "fcs_population_source": None,
            "fcs_fallback_kind": None,
            "fcs_fallback_pmf_semantics": None,
        },
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
    with pytest.raises(ContextComparisonError, match="included_game_ids"):
        validate_week4_pair(context_1_2, context_1_3)


def test_week4_pair_requires_matching_posterior_configuration(tmp_path: Path) -> None:
    context_1_2 = tmp_path / "context-1.2"
    context_1_3 = tmp_path / "context-1.3"
    _write_pair_artifact(context_1_2, prior_version="1.2", game_id="game-1")
    _write_pair_artifact(context_1_3, prior_version="1.3", game_id="game-1")

    metadata_path = context_1_3 / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["posterior_inference_configuration"]["damping"] = 0.25
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ContextComparisonError, match="not matched"):
        validate_week4_pair(context_1_2, context_1_3)


def test_committed_comparison_quartet_is_complete_and_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = (
        root
        / "data/processed/preseason/context_v1_3_2026_reconstruction/comparison_quartet.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "frozen"
    artifacts = manifest["artifacts"]
    assert all(artifacts.values())
    preseason_1_2 = root / artifacts["context_1_2_preseason"]
    preseason_1_3 = root / artifacts["context_1_3_reconstructed_preseason"]
    preseason_1_2_metadata = json.loads(
        (preseason_1_2 / "metadata.json").read_text(encoding="utf-8")
    )
    preseason_1_3_metadata = json.loads(
        (preseason_1_3 / "metadata.json").read_text(encoding="utf-8")
    )
    assert preseason_1_2_metadata["prior_model_version"] == "1.2"
    assert preseason_1_3_metadata["prior_model_version"] == "1.3"
    assert preseason_1_3_metadata["prior_lineage"] == "context_1_3_reconstructed_2026"

    official_1_2 = root / artifacts["context_1_2_official_week_4"]
    official_1_3 = root / artifacts["context_1_3_official_week_4"]
    pair = validate_week4_pair(official_1_2, official_1_3)
    recorded = manifest["official_week_4"]
    assert pair["parity_validation"] == "passed"
    assert recorded["context_1_2_snapshot_id"] == pair["context_1_2_snapshot_id"]
    assert recorded["context_1_3_snapshot_id"] == pair["context_1_3_snapshot_id"]
    assert recorded["requested_cutoff"] == recorded["effective_cutoff"]
    assert recorded["included_game_count"] == 410
    assert recorded["fbs_team_count"] == 138
    assert recorded["historical_likelihood_version"] == "V1"
    assert recorded["prior_model_versions"] == ["1.2", "1.3"]
    assert recorded["prior_artifact_sha256"]["1.2"] != recorded[
        "prior_artifact_sha256"
    ]["1.3"]
    assert manifest["validation"]["non_null_artifact_count"] == 4
    assert manifest["validation"]["boundary_sanity"]["week_3_missing_game_ids"] == []
    assert manifest["validation"]["boundary_sanity"]["week_4_included_game_ids"] == []


@pytest.mark.parametrize(
    ("source_name", "backfill_name", "expected_count", "expected_corpus_hash"),
    [
        (
            "2026-weekly-2026-09-08T11-43-00.275833Z-context",
            "2026-weekly-2026-09-08T11-43-00.275833Z-context-v1.3",
            172,
            "4746d0ef804157c38136bafe67724261a3e113f8de47914c123ce93a263b32c3",
        ),
        (
            "2026-weekly-2026-09-13T12-02-55.255941Z-context",
            "2026-weekly-2026-09-13T12-02-55.255941Z-context-v1.3",
            291,
            "db8aec7f95021871438de8a7a3ddd470eef1fd95165b3c2594b4be2ceef3b016",
        ),
    ],
)
def test_committed_weekly_context_backfills_have_frozen_lineage(
    source_name: str,
    backfill_name: str,
    expected_count: int,
    expected_corpus_hash: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "data/processed/snapshots/2026"
        / source_name
        / "predictive/context"
    )
    backfill = (
        root
        / "data/processed/snapshots/2026"
        / backfill_name
        / "predictive/context"
    )

    result = validate_context_backfill(source, backfill)

    assert result["parity_validation"] == "passed"
    assert result["included_game_count"] == expected_count
    assert result["game_corpus_sha256"] == expected_corpus_hash
    assert result["prior_model_versions"] == ["1.2", "1.3"]


@pytest.mark.parametrize(
    ("backfill_name", "expected_future_predictions"),
    [
        ("2026-weekly-2026-09-08T11-43-00.275833Z-context-v1.3", 787),
        ("2026-weekly-2026-09-13T12-02-55.255941Z-context-v1.3", 703),
    ],
)
def test_committed_context_backfills_use_frozen_historical_schedule_surface(
    backfill_name: str, expected_future_predictions: int
) -> None:
    root = Path(__file__).resolve().parents[1]
    backfill = (
        root
        / "data/processed/snapshots/2026"
        / backfill_name
        / "predictive/context"
    )
    metadata = json.loads((backfill / "metadata.json").read_text(encoding="utf-8"))
    team_seasons = json.loads(
        (backfill / "team_seasons.json").read_text(encoding="utf-8")
    )
    source = metadata["presentation_schedule_source"]

    assert metadata["presentation_schedule_game_count"] == 888
    assert source["kind"] == "frozen_historical_schedule"
    assert source["snapshot_id"] == metadata["source_evidence_snapshot_id"]
    assert (root / source["path"]).is_file()
    assert team_seasons["schedule_source"] == source
    assert len(team_seasons["future_predictions"]) == expected_future_predictions


def test_context_backfill_validator_rejects_row_mutation(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "data/processed/snapshots/2026/2026-weekly-2026-09-08T11-43-00.275833Z-context"
        / "predictive/context"
    )
    backfill = (
        root
        / "data/processed/snapshots/2026/2026-weekly-2026-09-08T11-43-00.275833Z-context-v1.3"
        / "predictive/context"
    )
    copied_source = tmp_path / "source"
    copied_backfill = tmp_path / "backfill"
    shutil.copytree(source, copied_source)
    shutil.copytree(backfill, copied_backfill)

    rows = []
    included_games = copied_backfill / "included_games.csv"
    with included_games.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["homePoints"] = str(int(rows[0]["homePoints"]) + 1)
    with included_games.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ContextComparisonError, match="frozen-evidence replay"):
        validate_context_backfill(copied_source, copied_backfill)


def test_weekly_lineage_manifest_records_both_retrospective_backfills() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (
            root
            / "data/processed/preseason/context_v1_3_2026_reconstruction/weekly_lineage.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["validation"] == {
        "original_context_1_2_artifacts_untouched": True,
        "retrospective_backfills_only": True,
        "source_evidence_replayed": True,
        "status": "passed",
    }
    assert set(manifest["backfills"]) == {
        "context_1_3_week_2_backfill",
        "context_1_3_week_3_backfill",
    }
    for report in manifest["backfills"].values():
        assert report["evidence_parity"] == "passed"
        assert len(report["largest_ranking_differences"]) == 10
        assert report["source_context_1_2_snapshot"] != report[
            "generated_context_1_3_snapshot"
        ]
