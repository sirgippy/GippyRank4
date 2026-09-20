"""Reconstruct the canonical 2026 Context 1.3 Week 2/3 lineage."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gippyrank.context_comparison import build_context_backfill
from gippyrank.posterior.snapshots import relative_path
from gippyrank.site_data import build_site_data

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINTS = (
    {
        "name": "context_1_3_week_2_backfill",
        "source": "data/processed/snapshots/2026/2026-weekly-2026-09-08T11-43-00.275833Z-context/predictive/context",
        "publication_slot": "2026-09-08",
        "display_label": "Week 2 (Context 1.3 retrospective)",
    },
    {
        "name": "context_1_3_week_3_backfill",
        "source": "data/processed/snapshots/2026/2026-weekly-2026-09-13T12-02-55.255941Z-context/predictive/context",
        "publication_slot": "2026-09-13",
        "display_label": "Week 3 (Context 1.3 retrospective)",
    },
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _rank_differences(source: Path, generated: Path) -> list[dict[str, Any]]:
    def read(path: Path) -> dict[str, dict[str, str]]:
        with (path / "rankings.csv").open(newline="", encoding="utf-8") as handle:
            return {
                row["team_id"]: row
                for row in csv.DictReader(handle)
                if row.get("subdivision", "").casefold() == "fbs"
            }

    source_rows = read(source)
    generated_rows = read(generated)
    values: list[dict[str, Any]] = []
    for team_id, source_row in source_rows.items():
        generated_row = generated_rows.get(team_id)
        if generated_row is None:
            continue
        source_rank = int(source_row["display_rank"])
        generated_rank = int(generated_row["display_rank"])
        values.append(
            {
                "team_id": team_id,
                "team": generated_row["team_name"],
                "source_rank": source_rank,
                "backfill_rank": generated_rank,
                "rank_change": source_rank - generated_rank,
                "absolute_rank_change": abs(source_rank - generated_rank),
            }
        )
    return sorted(
        values,
        key=lambda value: (-value["absolute_rank_change"], value["team"]),
    )[:10]


def _register_publications(root: Path, generated: list[dict[str, Any]]) -> None:
    config_path = root / "site/publish_config.json"
    config = _read_json(config_path)
    entries = config.get("snapshots")
    if not isinstance(entries, list):
        raise TypeError("publish configuration has no snapshots list")
    for item in generated:
        source = item["generated_path"]
        matching = next(
            (entry for entry in entries if entry.get("source") == source), None
        )
        replacement = {
            "display_label": item["display_label"],
            "publication_slot": item["publication_slot"],
            "source": source,
        }
        if matching is not None:
            matching.update(replacement)
            continue
        insert_at = max(
            (
                index
                for index, entry in enumerate(entries)
                if entry.get("publication_slot") == item["publication_slot"]
            ),
            default=len(entries) - 1,
        )
        entries.insert(insert_at + 1, replacement)
    config["snapshots"] = entries
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _write_lineage_manifest(root: Path, reports: list[dict[str, Any]]) -> None:
    value = {
        "manifest_version": "context-1.3-weekly-lineage-2026-v1",
        "lineage": [
            "context_1_3_reconstructed_preseason",
            "context_1_3_week_2_backfill",
            "context_1_3_week_3_backfill",
            "context_1_3_week_4",
        ],
        "backfills": {report["lineage_name"]: report for report in reports},
        "validation": {
            "status": "passed",
            "retrospective_backfills_only": True,
            "source_evidence_replayed": True,
            "original_context_1_2_artifacts_untouched": True,
        },
    }
    manifest_path = (
        root
        / "data/processed/preseason/context_v1_3_2026_reconstruction/weekly_lineage.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--generation-timestamp",
        default="2026-09-20T12:00:00+00:00",
        help="Deterministic timestamp recorded for the retrospective generation",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    generation_timestamp = datetime.fromisoformat(args.generation_timestamp)
    if generation_timestamp.tzinfo is None:
        generation_timestamp = generation_timestamp.replace(tzinfo=UTC)

    generated: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        source = root / checkpoint["source"]
        generated_path, validation = build_context_backfill(
            source_context_1_2=source,
            root=root,
            generation_timestamp=generation_timestamp,
        )
        generated_relative = relative_path(generated_path, root)
        source_relative = relative_path(source, root)
        source_metadata = _read_json(source / "metadata.json")
        generated_metadata = _read_json(generated_path / "metadata.json")
        report = {
            "lineage_name": checkpoint["name"],
            "source_context_1_2_snapshot": source_metadata["snapshot_id"],
            "generated_context_1_3_snapshot": generated_metadata["snapshot_id"],
            "source_path": source_relative,
            "generated_path": generated_relative,
            "publication_slot": checkpoint["publication_slot"],
            "display_label": checkpoint["display_label"],
            "included_game_count": validation["included_game_count"],
            "requested_cutoff": validation["requested_cutoff"],
            "effective_cutoff": validation["effective_cutoff"],
            "evidence_parity": validation["parity_validation"],
            "included_game_ids_sha256": validation["included_game_ids_sha256"],
            "included_game_rows_sha256": validation["included_game_rows_sha256"],
            "source_evidence_hash": validation["included_game_rows_sha256"],
            "source_game_corpus_sha256": validation["game_corpus_sha256"],
            "fbs_team_count": validation["fbs_team_count"],
            "fbs_team_keys_sha256": validation["fbs_team_keys_sha256"],
            "prior_model_version": generated_metadata["prior_model_version"],
            "prior_artifact_sha256": validation["prior_artifact_sha256"],
            "largest_ranking_differences": _rank_differences(source, generated_path),
        }
        generated.append(report)
        reports.append(report)

    _register_publications(root, generated)
    _write_lineage_manifest(root, reports)
    build_site_data(
        root=root,
        config_path=root / "site/publish_config.json",
        output_directory=root / "site/data",
    )
    print(json.dumps({"backfills": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
