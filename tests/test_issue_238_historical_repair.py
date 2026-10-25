from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from gippyrank.api.static import build_static_api
from gippyrank.ranking_variants import performance_variant

ROOT = Path(__file__).resolve().parents[1]
SLOTS = ("2026-09-06", "2026-09-07", "2026-09-08")
PHANTOM_ID = "401866625"
RESULT_ID = "401917058"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sources() -> dict[str, dict[str, tuple[dict[str, Any], Path]]]:
    config = json.loads((ROOT / "site/publish_config.json").read_text())
    result: dict[str, dict[str, tuple[dict[str, Any], Path]]] = {
        slot: {} for slot in SLOTS
    }
    for entry in config["snapshots"]:
        slot = entry["publication_slot"]
        if slot not in result:
            continue
        path = ROOT / entry["source"]
        metadata = json.loads((path / "metadata.json").read_text())
        kind = (
            str(metadata["prior_family"])
            if metadata["ranking_family"] == "predictive"
            else str(performance_variant(metadata))
        )
        result[slot][kind] = (metadata, path)
    return result


def test_repaired_publications_preserve_cutoffs_and_share_corrected_evidence() -> None:
    sources = _sources()
    assert all(
        set(sources[slot])
        == {"context", "history", "quality_of_play", "quality_of_record"}
        for slot in SLOTS
    )
    audit = json.loads(
        (
            ROOT
            / "data/processed/issue_238_historical_repair/lineage_audit.json"
        ).read_text()
    )
    assert audit["issue"] == "238"
    assert [item["publication_slot"] for item in audit["publications"]] == list(
        SLOTS
    )
    assert [item["publication_status"] for item in audit["publications"]] == [
        "temporary",
        "temporary",
        "official",
    ]
    fixture_audit = audit["snapshot_fixture_audit"]
    assert fixture_audit["selected_contaminated_bundle_count"] == 12
    assert all(
        item["currently_published"]
        for item in fixture_audit["selected_contaminated_bundles"]
    )
    assert all(
        not item["currently_published"]
        for item in fixture_audit["unselected_contaminated_bundles"]
    )

    for index, slot in enumerate(SLOTS):
        by_kind = sources[slot]
        rows_by_kind = {
            kind: _rows(path / "included_games.csv")
            for kind, (_metadata, path) in by_kind.items()
        }
        context_rows = rows_by_kind["context"]
        ids = [row["id"] for row in context_rows]
        assert PHANTOM_ID not in ids
        assert all(rows == context_rows for rows in rows_by_kind.values())
        results = [row for row in context_rows if row["id"] == RESULT_ID]
        if index == 0:
            assert results == []
        else:
            assert len(results) == 1
            assert (results[0]["homePoints"], results[0]["awayPoints"]) == (
                "28",
                "19",
            )

        context_metadata = by_kind["context"][0]
        history_metadata = by_kind["history"][0]
        original_history_entry = next(
            item
            for item in audit["superseded_artifacts"]
            if item["publication_slot"] == slot and item["role"] == "history"
        )
        original_history_metadata = json.loads(
            (
                ROOT
                / original_history_entry["superseded"]["path"]
                / "metadata.json"
            ).read_text()
        )
        assert history_metadata["model_versions"] == original_history_metadata[
            "model_versions"
        ]
        assert context_metadata["prior_model_version"] == (
            "1.4" if index == 2 else "1.2"
        )
        assert history_metadata["prior_model_version"] == "1.1"
        assert context_metadata["requested_cutoff"] == history_metadata[
            "requested_cutoff"
        ]
        assert context_metadata["effective_cutoff"] == history_metadata[
            "effective_cutoff"
        ]
        context_schedule = json.loads(
            (
                ROOT
                / "site/data/team-seasons"
                / f"{context_metadata['snapshot_id']}.json"
            ).read_text()
        )
        history_schedule = json.loads(
            (
                ROOT
                / "site/data/team-seasons"
                / f"{history_metadata['snapshot_id']}.json"
            ).read_text()
        )
        assert set(context_schedule["teams"]) == set(history_schedule["teams"])
        for team_id, context_team in context_schedule["teams"].items():
            history_team = history_schedule["teams"][team_id]
            assert [game["game_id"] for game in context_team["games"]] == [
                game["game_id"] for game in history_team["games"]
            ]
        for kind, (metadata, directory) in by_kind.items():
            assert metadata["included_game_count"] == len(context_rows)
            assert metadata["historical_evidence_correction"]["issue"] == "238"
            assert "quality_of_record_evidence" not in metadata
            assert not (directory / "inference_games.csv").exists()
            diagnostics = json.loads((directory / "diagnostics.json").read_text())
            assert "quality_of_record_evidence" not in diagnostics
            if kind in {"quality_of_play", "quality_of_record"}:
                assert metadata["source_context_snapshot_id"] == context_metadata[
                    "snapshot_id"
                ]

    normalized = _rows(ROOT / "data/processed/cfbd/games.csv")
    assert PHANTOM_ID not in {row["id"] for row in normalized}
    result = [row for row in normalized if row["id"] == RESULT_ID]
    assert len(result) == 1
    assert (result[0]["homePoints"], result[0]["awayPoints"]) == ("28", "19")

    backfill = json.loads(
        (
            ROOT
            / "data/processed/quality_of_record_backfill/manifest.json"
        ).read_text()
    )
    assert backfill["status"] == "complete"
    assert backfill["eligible_publication_count"] == audit["performance_backfill"][
        "publication_count"
    ]
    assert backfill["eligible_publication_count"] == len(backfill["publications"])
    assert backfill["blocked_publication_count"] == 0
    assert all(
        "quality_of_record_evidence" not in item
        for item in backfill["publications"]
    )

    superseded = audit["superseded_artifacts"]
    assert len(superseded) == 12
    for item in superseded:
        assert (ROOT / item["superseded"]["path"]).is_dir()
        assert (ROOT / item["repaired"]["path"]).is_dir()


def test_site_and_static_api_no_longer_publish_the_temporary_exception(
    tmp_path: Path,
) -> None:
    manifest = json.loads((ROOT / "site/data/manifest.json").read_text())
    assert all(
        "quality_of_record_evidence" not in item for item in manifest["snapshots"]
    )
    api_root = tmp_path / "api" / "v1"
    build_static_api(data_dir=ROOT / "site/data", output_directory=api_root)
    api = json.loads((api_root / "publications.json").read_text())
    assert all(
        "quality_of_record_evidence" not in item for item in api["publications"]
    )
    for directory in (ROOT / "site/data", api_root):
        for artifact in directory.rglob("*.json"):
            content = artifact.read_text(encoding="utf-8")
            assert "quality_of_record_evidence" not in content
            assert "inference_games.csv" not in content
    audit = json.loads(
        (
            ROOT
            / "data/processed/issue_238_historical_repair/lineage_audit.json"
        ).read_text()
    )
    assert audit["site_export"]["retired_exception_fields"] == 0
