"""Retrospective Quality of Results publication from retained frozen evidence."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from gippyrank.performance_snapshot import _as_snapshot, _logical_snapshot_id
from gippyrank.posterior.snapshots import sha256
from gippyrank.quality_of_results_snapshot import (
    build_quality_of_results_snapshot,
    validate_quality_of_results_against_context,
)
from gippyrank.ranking_variants import performance_variant
from gippyrank.site_data import _retained_publications, load_publish_config


class ResultsBackfillError(ValueError):
    """The retained inventory cannot be reconstructed safely."""


def backfill_quality_of_results(
    *, root: Path, allow_blocked: bool = False
) -> dict[str, Any]:
    """Validate all retained candidates before adding any to publication config."""
    config_path = root / "site/publish_config.json"
    selected, _ = load_publish_config(config_path, root)
    metadata = {item.source: _as_snapshot(item.source).metadata for item in selected}
    retained, retired = _retained_publications(selected, metadata)
    slots = sorted(
        {item.publication_slot for item in retained},
        key=lambda slot: next(
            item.publication_order for item in retained if item.publication_slot == slot
        ),
    )
    retired_count = len({item.publication_slot for item in retired})
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report_path = root / "data/processed/quality_of_results_backfill/manifest.json"
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    previous_entries = {
        item["publication_slot"]: item for item in previous.get("publications", [])
    }
    publications: list[dict[str, Any]] = []
    blockers: list[str] = []
    copies: list[tuple[Path, Path]] = []
    additions: list[tuple[Any, Path]] = []
    generated = reused = eligible = 0
    with tempfile.TemporaryDirectory(prefix="gippyrank-results-backfill-") as temp:
        staging = Path(temp)
        for slot in slots:
            entries = [item for item in retained if item.publication_slot == slot]
            # Preseason is a slot-level exclusion, even if it lacks a Context entry.
            if all(
                metadata[item.source].get("snapshot_type") == "preseason"
                for item in entries
            ):
                continue
            eligible += 1
            try:
                contexts = [
                    item
                    for item in entries
                    if metadata[item.source].get("ranking_family") == "predictive"
                    and metadata[item.source].get("prior_family") == "context"
                ]
                if len(contexts) != 1:
                    raise ValueError(
                        f"expected one retained Context snapshot, found {len(contexts)}"
                    )
                item = contexts[0]
                context = _as_snapshot(item.source)
                variants = [
                    entry
                    for entry in entries
                    if metadata[entry.source].get("ranking_family") == "performance"
                    and performance_variant(metadata[entry.source])
                    == "quality_of_results"
                ]
                if len(variants) > 1:
                    raise ValueError("duplicate Quality of Results variant")
                _, base = _logical_snapshot_id(context.snapshot_id)
                final = (
                    root
                    / "data/processed/snapshots"
                    / str(context.metadata["season"])
                    / base
                    / "performance/quality_of_results"
                )
                needs_registration = not variants
                was_generated = False
                if variants:
                    result = _as_snapshot(variants[0].source)
                    validate_quality_of_results_against_context(
                        context, result, root=root
                    )
                elif final.exists():
                    result = _as_snapshot(final)
                    validate_quality_of_results_against_context(
                        context, result, root=root
                    )
                else:
                    result = build_quality_of_results_snapshot(
                        context, root=root, output_root=staging
                    )
                    validate_quality_of_results_against_context(
                        context, result, root=root
                    )
                    was_generated = True
                m = result.metadata
                old = previous_entries.get(slot, {})
                origin = (
                    old.get("disposition")
                    if old.get("snapshot_id") == result.snapshot_id
                    and old.get("rankings_sha256")
                    == sha256(result.directory / "rankings.csv")
                    else None
                )
                publications.append(
                    {
                        "publication_slot": slot,
                        "publication_status": item.publication_status,
                        "source_context_snapshot_id": context.snapshot_id,
                        "source_context_metadata_sha256": sha256(
                            context.directory / "metadata.json"
                        ),
                        "effective_cutoff": m["effective_cutoff"],
                        "prior_model_version": m["prior_model_version"],
                        "prior_artifact_sha256": m["prior_artifact_sha256"],
                        "historical_likelihood_sha256": m[
                            "historical_likelihood_sha256"
                        ],
                        "included_game_count": m["included_game_count"],
                        "included_game_rows_sha256": m["included_game_rows_sha256"],
                        "included_game_ids_sha256": m["included_game_ids_sha256"],
                        "snapshot_id": result.snapshot_id,
                        "margin_weight": m["margin_weight"],
                        "rankings_sha256": sha256(result.directory / "rankings.csv"),
                        "pmf_file_sha256": sha256(
                            result.directory / "posterior_pmfs.csv"
                        ),
                        "validation_status": "valid",
                        "disposition": origin
                        or (
                            "generated"
                            if result.directory.is_relative_to(staging)
                            else "reused"
                        ),
                    }
                )
                if was_generated:
                    copies.append((result.directory, final))
                    generated += 1
                else:
                    reused += 1
                if needs_registration:
                    additions.append((item, final))
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
                blockers.append(f"{slot}: {error}")
        report = {
            "schema_version": "1.0",
            "status": "blocked" if blockers else "complete",
            "eligible_publication_count": eligible,
            "publication_count": len(publications),
            "generated_publication_count": sum(
                p["disposition"] == "generated" for p in publications
            ),
            "reused_publication_count": sum(
                p["disposition"] == "reused" for p in publications
            ),
            "blocked_publication_count": len(blockers),
            "retired_interim_slots_excluded": retired_count,
            "blockers": blockers,
            "publications": publications,
        }
        if not blockers or allow_blocked:
            for source, destination in copies:
                if destination.exists():
                    raise ResultsBackfillError(f"Refusing to overwrite {destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, destination)
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
                    json.dumps(config, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if blockers and not allow_blocked:
            raise ResultsBackfillError(
                "Frozen publication blockers:\n" + "\n".join(blockers)
            )
    return {**report, "run_generated_count": generated, "run_reused_count": reused}
