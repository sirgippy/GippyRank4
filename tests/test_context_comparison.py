from __future__ import annotations

import csv
import hashlib
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
