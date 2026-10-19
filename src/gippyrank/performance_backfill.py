"""One-time, cutoff-local Performance variant backfill for published history."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from gippyrank.performance_snapshot import (
    _as_snapshot,
    build_performance_snapshot,
    validate_performance_against_context,
)
from gippyrank.posterior.snapshots import sha256
from gippyrank.quality_of_record_snapshot import (
    build_quality_of_record_snapshot,
    validate_quality_of_record_against_context,
)
from gippyrank.ranking_variants import performance_variant
from gippyrank.site_data import load_publish_config


class PerformanceBackfillError(ValueError):
    """A retained publication lacks a compatible, frozen input or result."""


def backfill_performance_variants(
    *, root: Path, allow_blocked: bool = False
) -> dict[str, Any]:
    """Register both variants for every configured in-season Context snapshot.

    Validate the whole candidate before registering it. Existing Quality of Play
    bundles are read only. Repeated runs validate and retain existing outputs.
    No acquisition or predictive inference is performed here.
    """
    config_path = root / "site/publish_config.json"
    selected, _ = load_publish_config(config_path, root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sources = [(item, _as_snapshot(item.source)) for item in selected]
    publications = []
    blockers = []
    copies: list[tuple[Path, Path]] = []
    additions = []
    with tempfile.TemporaryDirectory(prefix="gippyrank-performance-backfill-") as temp:
        staging = Path(temp)
        for selected_context, context in sources:
            metadata = context.metadata
            if (
                metadata.get("prior_family") != "context"
                or metadata["snapshot_type"] == "preseason"
            ):
                continue
            slot = selected_context.publication_slot
            companions = [
                snapshot
                for item, snapshot in sources
                if item.publication_slot == slot
                and snapshot.metadata.get("ranking_family") == "performance"
                and snapshot.metadata.get("source_context_snapshot_id")
                == context.snapshot_id
            ]
            variants = {
                performance_variant(snapshot.metadata): snapshot
                for snapshot in companions
            }
            if len(variants) != len(companions):
                blockers.append(
                    f"{slot} / {context.snapshot_id}: duplicate Performance variant"
                )
                continue
            candidate_copies = []
            candidate_additions = []
            try:
                play = variants.get("quality_of_play")
                if play is None:
                    play = build_performance_snapshot(
                        context, root=root, output_root=staging
                    )
                    final = (
                        root
                        / "data/processed/snapshots"
                        / play.directory.relative_to(staging)
                    )
                    candidate_copies.append((play.directory, final))
                    candidate_additions.append((selected_context, final))
                validate_performance_against_context(context, play)
                record = variants.get("quality_of_record")
                if record is None:
                    record = build_quality_of_record_snapshot(
                        context, root=root, output_root=staging
                    )
                    final = (
                        root
                        / "data/processed/snapshots"
                        / record.directory.relative_to(staging)
                    )
                    candidate_copies.append((record.directory, final))
                    candidate_additions.append((selected_context, final))
                validate_quality_of_record_against_context(context, record, root=root)
                publications.append(
                    {
                        "publication_slot": slot,
                        "publication_status": selected_context.publication_status,
                        "source_context_snapshot_id": context.snapshot_id,
                        "source_context_metadata_sha256": sha256(
                            context.directory / "metadata.json"
                        ),
                        "prior_model_version": metadata["prior_model_version"],
                        "prior_artifact_sha256": metadata["prior_artifact_sha256"],
                        "historical_likelihood_sha256": record.metadata[
                            "historical_likelihood_sha256"
                        ],
                        "effective_cutoff": metadata["effective_cutoff"],
                        "included_game_count": metadata["included_game_count"],
                        "included_game_rows_sha256": record.metadata[
                            "included_game_rows_sha256"
                        ],
                        "variants": {
                            name: {
                                "snapshot_id": snapshot.snapshot_id,
                                "rankings_sha256": sha256(
                                    snapshot.directory / "rankings.csv"
                                ),
                                "pmf_file_sha256": sha256(
                                    snapshot.directory / "posterior_pmfs.csv"
                                ),
                            }
                            for name, snapshot in (
                                ("quality_of_play", play),
                                ("quality_of_record", record),
                            )
                        },
                    }
                )
                if "quality_of_record_evidence" in record.metadata:
                    publications[-1]["quality_of_record_evidence"] = record.metadata[
                        "quality_of_record_evidence"
                    ]
                copies.extend(candidate_copies)
                additions.extend(candidate_additions)
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
                blockers.append(f"{slot} / {context.snapshot_id}: {error}")
        if blockers and not allow_blocked:
            raise PerformanceBackfillError(
                "Frozen publication blockers:\n" + "\n".join(blockers)
            )
        # Copy individual files: a newly created Quality of Play parent can
        # also contain a staged Quality of Record child.
        for source, destination in copies:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    target = destination / path.relative_to(source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
        for item, final in additions:
            config["snapshots"].append(
                {
                    "source": final.relative_to(root).as_posix(),
                    "display_label": item.display_label,
                    "publication_slot": item.publication_slot,
                }
            )
        if additions:
            config_path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    report = {
        "status": "blocked" if blockers else "complete",
        "blockers": blockers,
        "eligible_publication_count": len(publications) + len(blockers),
        "blocked_publication_count": len(blockers),
        "schema_version": "1.0",
        "publication_count": len(publications),
        "publications": publications,
    }
    report_path = root / "data/processed/quality_of_record_backfill/manifest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
