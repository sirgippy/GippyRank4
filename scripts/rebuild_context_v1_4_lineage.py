"""Rebuild and publish the six 2026 Context 1.4 states with repaired DB priors."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from gippyrank.artifact_hashes import included_game_rows_sha256
from gippyrank.context_db_repair import sha256
from gippyrank.performance_snapshot import (
    build_performance_snapshot,
    validate_performance_against_context,
)
from gippyrank.posterior.snapshots import Snapshot, build_snapshot, snapshot_id

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("data/processed/context_db_repair_172")
CONFIG = Path("site/publish_config.json")
SLOTS = {
    "2026-preseason": "2026-preseason-context-v1.4",
    "2026-09-08": "2026-weekly-2026-09-08T11-43-00.275833Z-context-v1.4",
    "2026-09-13": "2026-weekly-2026-09-13T12-02-55.255941Z-context-v1.4",
    "2026-09-20": "2026-weekly-2026-09-20T11-00-14.294077Z-context-v1.4",
    "2026-09-27": "2026-weekly-2026-09-27T12-27-35.698895Z-context-v1.4",
    "2026-10-04": "2026-weekly-2026-10-04T14-51-53.251364Z-context-v1.4",
}


def snapshot(root: Path, relative: str) -> Snapshot:
    path = root / relative
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return Snapshot(str(metadata["snapshot_id"]), path, metadata)


def _evidence_parity(context: Snapshot, history: Snapshot) -> None:
    """Compare game rows directly across snapshot schema generations."""
    if sha256(context.directory / "included_games.csv") != sha256(
        history.directory / "included_games.csv"
    ):
        raise ValueError("Context and History included-game files differ")
    for field in (
        "requested_cutoff",
        "effective_cutoff",
        "included_game_count",
        "historical_likelihood_sha256",
    ):
        if context.metadata.get(field) != history.metadata.get(field):
            raise ValueError(f"Context/History evidence differs: {field}")
    if context.metadata["snapshot_type"] == "weekly":
        for field in (
            "game_corpus_sha256",
            "fcs_fallback_team_ids",
            "fcs_population_size",
            "source_retrieval_times",
        ):
            if context.metadata.get(field) != history.metadata.get(field):
                raise ValueError(f"Context/History weekly evidence differs: {field}")


def main(root: Path = ROOT) -> dict[str, object]:
    config = json.loads((root / CONFIG).read_text(encoding="utf-8"))
    entries = list(config["snapshots"])
    by_slot: dict[str, dict[str, str]] = {}
    for slot in SLOTS:
        pairs: dict[str, str] = {}
        for entry in entries:
            if entry["publication_slot"] != slot:
                continue
            path = root / entry["source"]
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            if metadata["ranking_family"] == "performance":
                pairs["performance"] = entry["source"]
            else:
                pairs[str(metadata["prior_family"])] = entry["source"]
        if "history" not in pairs or "context" not in pairs:
            raise ValueError(f"{slot}: publication lacks Context or History")
        by_slot[slot] = pairs
    rows = []
    replacements: dict[tuple[str, str], str] = {}
    prior_path = (
        root / "data/processed/preseason/context_v1_4/annual/2026/predictions.csv"
    )
    for slot, old_id in SLOTS.items():
        print(f"Replaying Context 1.4 {slot}", flush=True)
        source_relative = f"data/processed/snapshots/2026/{old_id}/predictive/context"
        source = snapshot(root, source_relative)
        metadata = source.metadata
        cutoff = (
            None
            if metadata["requested_cutoff"] is None
            else datetime.fromisoformat(str(metadata["requested_cutoff"]))
        )
        schedule = root / "site/data/week-games" / f"{old_id}.json"
        if not schedule.is_file():
            raise FileNotFoundError(schedule)
        corrected_id = snapshot_id(
            2026,
            str(metadata["snapshot_type"]),
            "context",
            cutoff,
            "1.4",
            lineage_suffix="db-repair",
        )
        corrected_relative = (
            f"data/processed/snapshots/2026/{corrected_id}/predictive/context"
        )
        if (root / corrected_relative / "metadata.json").is_file():
            corrected = snapshot(root, corrected_relative)
        else:
            corrected = build_snapshot(
                season=2026,
                cutoff=cutoff,
                prior_family="context",
                snapshot_type=str(metadata["snapshot_type"]),
                prior_model_version="1.4",
                lineage_suffix="db-repair",
                root=root,
                evidence_snapshot=source.directory,
                presentation_schedule_path=schedule,
                canonical_lineage_replay=True,
                generation_timestamp=datetime.fromisoformat(
                    str(metadata["generation_timestamp"])
                ),
            )
        history = snapshot(root, by_slot[slot]["history"])
        _evidence_parity(corrected, history)
        if (
            corrected.metadata["prior_artifact_sha256"] != sha256(prior_path)
            or corrected.metadata["included_game_count"]
            != source.metadata["included_game_count"]
            or corrected.metadata["included_game_rows_sha256"]
            != source.metadata["included_game_rows_sha256"]
        ):
            raise ValueError(
                f"{slot}: corrected prior or cutoff-local game evidence differs"
            )
        with (corrected.directory / "included_games.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            game_hash = included_game_rows_sha256(list(csv.DictReader(handle)))
        if game_hash != corrected.metadata["included_game_rows_sha256"]:
            raise ValueError(f"{slot}: included game rows fail semantic hash")
        replacements[(slot, "context")] = corrected.directory.relative_to(
            root
        ).as_posix()
        performance = None
        if slot != "2026-preseason":
            performance = build_performance_snapshot(
                corrected,
                root=root,
                source_context_path=corrected.directory,
                generation_timestamp=datetime.fromisoformat(
                    str(metadata["generation_timestamp"])
                ),
            )
            validate_performance_against_context(corrected, performance)
            replacements[(slot, "performance")] = performance.directory.relative_to(
                root
            ).as_posix()
        rows.append(
            {
                "publication_slot": slot,
                "source_snapshot_id": old_id,
                "corrected_snapshot_id": corrected.snapshot_id,
                "corrected_snapshot_path": corrected.directory.relative_to(
                    root
                ).as_posix(),
                "performance_snapshot_id": None
                if performance is None
                else performance.snapshot_id,
                "included_games": corrected.metadata["included_game_count"],
                "included_game_rows_sha256": game_hash,
                "context_prior_sha256": corrected.metadata["prior_pmfs_sha256"],
                "context_posterior_sha256": corrected.metadata["posterior_pmfs_sha256"],
                "history_posterior_sha256": history.metadata["posterior_pmfs_sha256"],
                "context_history_evidence_parity": True,
            }
        )
    for entry in entries:
        slot = entry["publication_slot"]
        if slot not in SLOTS:
            continue
        source = entry["source"]
        if source.endswith("/predictive/context"):
            entry["source"] = replacements[(slot, "context")]
        elif source.endswith("/performance"):
            entry["source"] = replacements[(slot, "performance")]
    config["snapshots"] = entries
    (root / CONFIG).write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "correction": "issue_172_repaired_db_transfer_coverage",
        "public_context_version": "1.4",
        "corrected_prior_sha256": sha256(prior_path),
        "publication_config_sha256": sha256(root / CONFIG),
        "canonical_origins": rows,
        "context_history_evidence_parity": all(
            row["context_history_evidence_parity"] for row in rows
        ),
    }
    (root / OUTPUT / "lineage_audit.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
