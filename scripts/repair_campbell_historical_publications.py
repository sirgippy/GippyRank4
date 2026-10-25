"""Rebuild the three 2026 publications contaminated by the Campbell duplicate."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from gippyrank.artifact_hashes import (
    INCLUDED_GAME_FIELDS,
    included_game_rows_sha256,
    stable_values_sha256,
)
from gippyrank.performance_backfill import backfill_performance_variants
from gippyrank.performance_snapshot import (
    _as_snapshot,
    build_performance_snapshot,
    validate_performance_against_context,
)
from gippyrank.posterior.snapshots import (
    Snapshot,
    _game_evidence_exclusions_sha256,
    _read_included_game_rows,
    build_snapshot,
    sha256,
)
from gippyrank.quality_of_record_snapshot import (
    build_quality_of_record_snapshot,
    validate_quality_of_record_against_context,
)
from gippyrank.ranking_variants import performance_variant
from gippyrank.site_data import build_site_data

ROOT = Path(__file__).resolve().parents[1]
ISSUE = "238"
LINEAGE_SUFFIX = "issue-238"
PHANTOM_GAME_ID = "401866625"
REPLACEMENT_GAME_ID = "401917058"
TARGET_SLOTS = ("2026-09-06", "2026-09-07", "2026-09-08")
AUDIT_ROOT = Path("data/processed/issue_238_historical_repair")
CONFIG_PATH = Path("site/publish_config.json")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=INCLUDED_GAME_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _selected_sources(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    config_path = root / CONFIG_PATH
    config = _read_json(config_path)
    publications: dict[str, dict[str, Any]] = {slot: {} for slot in TARGET_SLOTS}
    for entry in config.get("snapshots", []):
        slot = entry.get("publication_slot")
        if slot not in publications:
            continue
        source = str(entry["source"])
        directory = root / source
        metadata = _read_json(directory / "metadata.json")
        if metadata.get("ranking_family") == "predictive":
            kind = str(metadata["prior_family"])
        else:
            kind = str(performance_variant(metadata))
        if kind in publications[slot]:
            raise ValueError(f"{slot}: duplicate configured {kind} artifact")
        publications[slot][kind] = {
            "entry": entry,
            "source": source,
            "directory": directory,
            "snapshot": _as_snapshot(directory),
        }
    required = {"context", "history", "quality_of_play", "quality_of_record"}
    for slot, sources in publications.items():
        if set(sources) != required:
            raise ValueError(
                f"{slot}: expected {sorted(required)}, found {sorted(sources)}"
            )
    return publications, config


def _validate_phantom_row(row: dict[str, str], *, source_id: str) -> None:
    expected = {
        "id": PHANTOM_GAME_ID,
        "season": "2026",
        "completed": "True",
        "homeId": "2097",
        "homeTeam": "Campbell",
        "homeClassification": "fcs",
        "homePoints": "0",
        "awayId": "2717",
        "awayTeam": "Western Carolina",
        "awayClassification": "fcs",
        "awayPoints": "0",
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{source_id}: unexpected row for fixture {PHANTOM_GAME_ID}")


def _prepare_evidence_bundle(
    *,
    root: Path,
    source: dict[str, Any],
    evidence_root: Path,
) -> tuple[Path, dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    snapshot: Snapshot = source["snapshot"]
    source_directory: Path = source["directory"]
    metadata = json.loads(json.dumps(snapshot.metadata))
    rows = _read_included_game_rows(source_directory / "included_games.csv")
    phantom = [row for row in rows if row["id"] == PHANTOM_GAME_ID]
    if len(phantom) != 1:
        raise ValueError(
            f"{snapshot.snapshot_id}: expected one phantom fixture, found {len(phantom)}"
        )
    _validate_phantom_row(phantom[0], source_id=snapshot.snapshot_id)
    original_ids = [row["id"] for row in rows]
    corrected_rows = [row for row in rows if row["id"] != PHANTOM_GAME_ID]
    corrected_ids = [row["id"] for row in corrected_rows]

    correction = {
        "schema_version": "1.0",
        "issue": ISSUE,
        "source_snapshot_id": snapshot.snapshot_id,
        "source_prior_family": metadata["prior_family"],
        "source_prior_model_version": metadata["prior_model_version"],
        "source_requested_cutoff": metadata.get("requested_cutoff"),
        "source_effective_cutoff": metadata.get("effective_cutoff"),
        "source_included_game_count": len(rows),
        "source_included_game_ids_sha256": stable_values_sha256(original_ids),
        "source_included_game_rows_sha256": included_game_rows_sha256(rows),
        "corrected_included_game_count": len(corrected_rows),
        "corrected_included_game_ids_sha256": stable_values_sha256(corrected_ids),
        "corrected_included_game_rows_sha256": included_game_rows_sha256(
            corrected_rows
        ),
        "excluded_games": [
            {
                "game_id": PHANTOM_GAME_ID,
                "replacement_game_ids": [REPLACEMENT_GAME_ID],
                "reason": "superseded_completed_zero_zero_fixture",
                "source_game_row_sha256": included_game_rows_sha256(phantom),
            }
        ],
    }
    metadata.update(
        {
            "included_game_count": len(corrected_rows),
            "included_game_ids": corrected_ids,
            "included_game_ids_sha256": stable_values_sha256(corrected_ids),
            "included_game_rows_sha256": included_game_rows_sha256(corrected_rows),
            "historical_evidence_correction": correction,
        }
    )
    exclusions = list(metadata.get("game_evidence_exclusions", []))
    exclusions = [item for item in exclusions if item.get("game_id") != PHANTOM_GAME_ID]
    exclusions.append(
        {
            "game_id": PHANTOM_GAME_ID,
            "reason": "superseded_completed_zero_zero_fixture",
            "replacement_game_ids": [REPLACEMENT_GAME_ID],
            "investigation_required": False,
        }
    )
    metadata["game_evidence_exclusions"] = exclusions
    metadata["game_evidence_exclusions_sha256"] = _game_evidence_exclusions_sha256(
        exclusions
    )

    destination = evidence_root / snapshot.snapshot_id
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    _write_json(destination / "metadata.json", metadata)
    _write_rows(destination / "included_games.csv", corrected_rows)
    for name in ("rankings.csv", "diagnostics.json"):
        path = source_directory / name
        if path.is_file():
            shutil.copyfile(path, destination / name)

    if metadata["prior_family"] == "context":
        schedule_source = (
            root / "site/data/week-games" / f"{snapshot.snapshot_id}.json"
        )
        schedule = _read_json(schedule_source)
        if schedule.get("snapshot_id") != snapshot.snapshot_id:
            raise ValueError(f"{schedule_source}: schedule snapshot ID mismatch")
        if [str(value) for value in schedule.get("included_game_ids", [])] != original_ids:
            raise ValueError(f"{schedule_source}: schedule game IDs do not match source")
        schedule["included_game_ids"] = corrected_ids
        _write_json(destination / "presentation_week_games.json", schedule)

    return destination, correction, rows, corrected_rows


def _ranking_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["team_id"]: row for row in csv.DictReader(handle)}


def _rank_changes(
    *, slot: str, kind: str, old_path: Path, new_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    old = _ranking_rows(old_path)
    new = _ranking_rows(new_path)
    if old.keys() != new.keys():
        raise ValueError(f"{slot}/{kind}: FBS ranking population changed")
    changes = []
    for team_id in sorted(old):
        before = old[team_id]
        after = new[team_id]
        before_expected = float(before["expected_rank"])
        after_expected = float(after["expected_rank"])
        before_display = str(before["display_rank"])
        after_display = str(after["display_rank"])
        display_delta = (
            int(after_display) - int(before_display)
            if before_display.isdigit() and after_display.isdigit()
            else None
        )
        expected_delta = after_expected - before_expected
        if display_delta or abs(expected_delta) > 1e-8:
            changes.append(
                {
                    "publication_slot": slot,
                    "ranking": kind,
                    "team_id": team_id,
                    "team_name": after["team_name"],
                    "previous_display_rank": before_display,
                    "repaired_display_rank": after_display,
                    "display_rank_change": display_delta,
                    "previous_expected_rank": before_expected,
                    "repaired_expected_rank": after_expected,
                    "expected_rank_change": expected_delta,
                }
            )
    ranked_display_changes = [row for row in changes if row["display_rank_change"]]
    ordered = sorted(
        changes,
        key=lambda row: (
            -(abs(row["display_rank_change"] or 0)),
            -abs(row["expected_rank_change"]),
            row["team_name"],
        ),
    )
    summary = {
        "publication_slot": slot,
        "ranking": kind,
        "team_count": len(old),
        "teams_with_changed_display_rank": len(ranked_display_changes),
        "teams_with_changed_expected_rank": len(changes),
        "largest_display_rank_change": max(
            (abs(row["display_rank_change"] or 0) for row in changes), default=0
        ),
        "largest_expected_rank_change": max(
            (abs(row["expected_rank_change"]) for row in changes), default=0.0
        ),
        "largest_movers": ordered[:5],
    }
    return summary, changes


def _publication_status(root: Path, source: dict[str, Any]) -> str | None:
    status = source["entry"].get("publication_status")
    if isinstance(status, str):
        return status
    snapshot_id = source["snapshot"].snapshot_id
    manifest = _read_json(root / "site/data/manifest.json")
    matches = [
        entry
        for entry in manifest.get("snapshots", [])
        if entry.get("snapshot_id") == snapshot_id
    ]
    if len(matches) > 1:
        raise ValueError(f"{snapshot_id}: duplicate site manifest entries")
    return matches[0].get("publication_status") if matches else None


def _source_audit(
    *,
    root: Path,
    slot: str,
    kind: str,
    old_source: dict[str, Any],
    new_snapshot: Snapshot,
    stage_root: Path,
    corrected_rows: list[dict[str, str]],
) -> dict[str, Any]:
    old_snapshot: Snapshot = old_source["snapshot"]
    old_directory: Path = old_source["directory"]
    old_rows = _read_included_game_rows(old_directory / "included_games.csv")
    metadata = old_snapshot.metadata
    if new_snapshot.metadata.get("prior_model_version") != metadata.get(
        "prior_model_version"
    ):
        raise ValueError(f"{slot}/{kind}: model version changed during repair")
    for field in ("requested_cutoff", "effective_cutoff", "historical_likelihood_sha256"):
        if new_snapshot.metadata.get(field) != metadata.get(field):
            raise ValueError(f"{slot}/{kind}: frozen {field} changed during repair")
    return {
        "publication_slot": slot,
        "publication_status": _publication_status(root, old_source),
        "role": kind,
        "superseded": {
            "snapshot_id": old_snapshot.snapshot_id,
            "path": old_source["source"],
            "metadata_sha256": sha256(old_directory / "metadata.json"),
            "rankings_sha256": sha256(old_directory / "rankings.csv"),
            "included_game_count": len(old_rows),
            "included_game_rows_sha256": included_game_rows_sha256(old_rows),
        },
        "repaired": {
            "snapshot_id": new_snapshot.snapshot_id,
            "path": (
                Path("data/processed/snapshots")
                / new_snapshot.directory.relative_to(stage_root)
            ).as_posix(),
            "prior_family": new_snapshot.metadata.get("prior_family"),
            "prior_model_version": new_snapshot.metadata.get("prior_model_version"),
            "requested_cutoff": new_snapshot.metadata.get("requested_cutoff"),
            "effective_cutoff": new_snapshot.metadata.get("effective_cutoff"),
            "included_game_count": len(corrected_rows),
            "included_game_rows_sha256": included_game_rows_sha256(corrected_rows),
            "metadata_sha256": sha256(new_snapshot.directory / "metadata.json"),
            "rankings_sha256": sha256(new_snapshot.directory / "rankings.csv"),
        },
    }


def _write_change_csv(path: Path, changes: list[dict[str, Any]]) -> None:
    fields = [
        "publication_slot",
        "ranking",
        "team_id",
        "team_name",
        "previous_display_rank",
        "repaired_display_rank",
        "display_rank_change",
        "previous_expected_rank",
        "repaired_expected_rank",
        "expected_rank_change",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(changes)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Issue 238: Campbell historical publication repair",
        "",
        "The September 6, September 7, and official Week 2 publications were",
        "rebuilt from retained cutoff-local inputs after removing the malformed",
        "completed 0–0 fixture `401866625`. The real 28–19 result `401917058`",
        "remains absent from September 6 and remains in September 7 and Week 2.",
        "",
        "Each repaired Context and History snapshot keeps its original prior",
        "version, likelihood, requested/effective cutoff, and inference settings.",
        "Quality of Play and Quality of Record were rebuilt from the corrected",
        "Context evidence. The superseded bundles remain in `data/processed` and",
        "are identified by source ID and hashes in the lineage audit.",
        "",
        (
            "The full repository scan found "
            f"{report['snapshot_fixture_audit']['selected_contaminated_bundle_count']} "
            "contaminated bundles in the active publication config. It also found "
            f"{report['snapshot_fixture_audit']['unselected_contaminated_bundle_count']} "
            "contaminated candidate bundles that are absent from the active publication "
            "manifest; those unselected lineages remain unchanged."
        ),
        "",
        "## Evidence counts",
        "",
        "| Publication | Previous games | Repaired games | `401917058` retained |",
        "| --- | ---: | ---: | --- |",
    ]
    for publication in report["publications"]:
        lines.append(
            "| {slot} | {before} | {after} | {result} |".format(
                slot=publication["publication_slot"],
                before=publication["source_game_count"],
                after=publication["corrected_game_count"],
                result="no" if publication["publication_slot"] == TARGET_SLOTS[0] else "yes",
            )
        )
    lines.extend(
        [
            "",
            "## Ranking changes",
            "",
            "The table reports changed display ranks and the largest moves for each",
            "official Week 2 ranking. Positive display-rank changes mean the team",
            "moved down the table. The full row-level comparison is in",
            "`data/processed/issue_238_historical_repair/ranking_changes.csv`.",
            "",
            "| Ranking | Teams moved | Largest display-rank moves |",
            "| --- | ---: | --- |",
        ]
    )
    official = [
        item
        for item in report["ranking_change_summaries"]
        if item["publication_slot"] == "2026-09-08"
    ]
    for summary in official:
        movers = "; ".join(
            f"{item['team_name']} {item['previous_display_rank']}→"
            f"{item['repaired_display_rank']} ({item['display_rank_change']:+})"
            for item in summary["largest_movers"]
            if item["display_rank_change"]
        ) or "No display-rank changes"
        lines.append(
            f"| {summary['ranking']} | "
            f"{summary['teams_with_changed_display_rank']} | {movers} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            (
                f"- Performance backfill: `{report['performance_backfill']['status']}`; "
                f"{report['performance_backfill']['publication_count']} eligible "
                "publications, "
                f"{report['performance_backfill']['blocked_publication_count']} blocked."
            ),
            "- Context, History, Quality of Play, and Quality of Record use the same",
            "  corrected game rows within each publication slot.",
            "- The active site and static API exports contain no temporary Quality",
            "  of Record exception declaration.",
            "",
            "Machine-readable details and artifact hashes are in",
            "`data/processed/issue_238_historical_repair/lineage_audit.json`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def repair_historical_publications(
    *, root: Path = ROOT, dry_run: bool = False
) -> dict[str, Any]:
    publications, config = _selected_sources(root)
    fixture_audit = _snapshot_fixture_audit(root, publications)
    if fixture_audit["selected_contaminated_bundle_count"] != 12:
        raise ValueError(
            "Expected 12 selected contaminated snapshot bundles across the three "
            "affected publication slots"
        )
    normalized_games = _read_normalized_game_corpus(root)
    if any(row["id"] == PHANTOM_GAME_ID for row in normalized_games):
        raise ValueError("Corrected normalized CFBD corpus still contains the phantom")
    replacement_rows = [
        row for row in normalized_games if row["id"] == REPLACEMENT_GAME_ID
    ]
    if len(replacement_rows) != 1 or (
        replacement_rows[0]["homePoints"], replacement_rows[0]["awayPoints"]
    ) != ("28", "19"):
        raise ValueError("Corrected normalized CFBD corpus lacks the real 28–19 result")

    temporary_parent = root / "data/processed"
    with tempfile.TemporaryDirectory(
        prefix="issue-238-repair-", dir=temporary_parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        stage_root = temporary / "snapshots"
        evidence_root = (
            temporary / "evidence"
            if dry_run
            else root / AUDIT_ROOT / "evidence"
        )
        path_replacements: dict[str, str] = {}
        repaired_publications = []
        audit_sources = []
        ranking_change_summaries = []
        ranking_changes = []
        preserved_hashes = _unaffected_source_hashes(root, publications)

        for slot in TARGET_SLOTS:
            print(f"Rebuilding publication slot {slot}...", flush=True)
            original = publications[slot]
            prepared: dict[
                str,
                tuple[
                    Path,
                    dict[str, Any],
                    list[dict[str, str]],
                    list[dict[str, str]],
                ],
            ] = {}
            for kind in ("context", "history"):
                evidence_path, correction, before_rows, after_rows = (
                    _prepare_evidence_bundle(
                        root=root,
                        source=original[kind],
                        evidence_root=evidence_root,
                    )
                )
                prepared[kind] = (evidence_path, correction, before_rows, after_rows)

            # History inference has no separate site schedule artifact. Reuse the
            # same cutoff-local presentation schedule as Context, keyed to the
            # History source ID so both frozen replays validate and retain the
            # identical completed and future schedule rows.
            history_schedule_path = prepared["history"][0] / "presentation_week_games.json"
            context_schedule = _read_json(
                prepared["context"][0] / "presentation_week_games.json"
            )
            history_snapshot_id = original["history"]["snapshot"].snapshot_id
            history_schedule_ids = [
                str(game_id) for game_id in context_schedule.get("included_game_ids", [])
            ]
            expected_history_ids = [
                row["id"] for row in prepared["history"][3]
            ]
            if history_schedule_ids != expected_history_ids:
                raise ValueError(
                    f"{slot}: Context presentation schedule differs from History evidence"
                )
            context_schedule["snapshot_id"] = history_snapshot_id
            _write_json(history_schedule_path, context_schedule)

            context_source = original["context"]["snapshot"].metadata
            history_source = original["history"]["snapshot"].metadata
            print(f"  Replaying Context from frozen evidence for {slot}...", flush=True)
            context_output = build_snapshot(
                season=int(context_source["season"]),
                cutoff=datetime.fromisoformat(str(context_source["requested_cutoff"])),
                prior_family="context",
                prior_model_version=str(context_source["prior_model_version"]),
                snapshot_type=str(context_source["snapshot_type"]),
                lineage_suffix=LINEAGE_SUFFIX,
                root=root,
                output_root=stage_root,
                generation_timestamp=datetime.fromisoformat(
                    str(context_source["generation_timestamp"])
                ),
                evidence_snapshot=prepared["context"][0],
                presentation_schedule_path=(
                    prepared["context"][0] / "presentation_week_games.json"
                ),
                evidence_correction_replay=True,
            )
            print(f"  Replaying History from frozen evidence for {slot}...", flush=True)
            history_output = build_snapshot(
                season=int(history_source["season"]),
                cutoff=datetime.fromisoformat(str(history_source["requested_cutoff"])),
                prior_family="history",
                prior_model_version=str(history_source["prior_model_version"]),
                snapshot_type=str(history_source["snapshot_type"]),
                lineage_suffix=LINEAGE_SUFFIX,
                root=root,
                output_root=stage_root,
                generation_timestamp=datetime.fromisoformat(
                    str(history_source["generation_timestamp"])
                ),
                evidence_snapshot=prepared["history"][0],
                presentation_schedule_path=history_schedule_path,
                evidence_correction_replay=True,
            )
            context_rows = _read_included_game_rows(
                context_output.directory / "included_games.csv"
            )
            history_rows = _read_included_game_rows(
                history_output.directory / "included_games.csv"
            )
            if context_rows != history_rows:
                raise ValueError(f"{slot}: Context and History evidence differ")
            if any(row["id"] == PHANTOM_GAME_ID for row in context_rows):
                raise ValueError(f"{slot}: phantom fixture survived the repair")
            has_replacement = any(
                row["id"] == REPLACEMENT_GAME_ID
                and row["homePoints"] == "28"
                and row["awayPoints"] == "19"
                for row in context_rows
            )
            if has_replacement != (slot != TARGET_SLOTS[0]):
                raise ValueError(f"{slot}: rescheduled result violates its cutoff")

            relative_context = context_output.directory.relative_to(stage_root)
            final_context_path = root / "data/processed/snapshots" / relative_context
            timestamp = datetime.fromisoformat(
                str(context_source["generation_timestamp"])
            )
            print(f"  Rebuilding Quality of Play for {slot}...", flush=True)
            play_output = build_performance_snapshot(
                context_output,
                root=root,
                output_root=stage_root,
                source_context_path=final_context_path,
                generation_timestamp=timestamp,
            )
            validate_performance_against_context(context_output, play_output)
            print(f"  Rebuilding Quality of Record for {slot}...", flush=True)
            record_output = build_quality_of_record_snapshot(
                context_output,
                root=root,
                output_root=stage_root,
                source_context_path=final_context_path,
                generation_timestamp=timestamp,
            )
            validate_quality_of_record_against_context(
                context_output, record_output, root=root
            )
            role_outputs = {
                "context": context_output,
                "history": history_output,
                "quality_of_play": play_output,
                "quality_of_record": record_output,
            }
            role_rows = {
                "context": context_rows,
                "history": history_rows,
                "quality_of_play": _read_included_game_rows(
                    play_output.directory / "included_games.csv"
                ),
                "quality_of_record": _read_included_game_rows(
                    record_output.directory / "included_games.csv"
                ),
            }
            for kind, output in role_outputs.items():
                rows = role_rows[kind]
                if rows != context_rows:
                    raise ValueError(f"{slot}/{kind}: evidence differs from Context")
                if any(row["id"] == PHANTOM_GAME_ID for row in rows):
                    raise ValueError(f"{slot}/{kind}: phantom fixture remains")
                old_source = original[kind]
                audit_sources.append(
                    _source_audit(
                        root=root,
                        slot=slot,
                        kind=kind,
                        old_source=old_source,
                        new_snapshot=output,
                        stage_root=stage_root,
                        corrected_rows=rows,
                    )
                )
                path_replacements[old_source["source"]] = (
                    Path("data/processed/snapshots")
                    / output.directory.relative_to(stage_root)
                ).as_posix()
                summary, changes = _rank_changes(
                    slot=slot,
                    kind=kind,
                    old_path=old_source["directory"] / "rankings.csv",
                    new_path=output.directory / "rankings.csv",
                )
                ranking_change_summaries.append(summary)
                ranking_changes.extend(changes)

            repaired_publications.append(
                {
                    "publication_slot": slot,
                    "publication_status": _publication_status(
                        root, original["context"]
                    ),
                    "source_game_count": len(prepared["context"][2]),
                    "corrected_game_count": len(context_rows),
                    "source_game_ids_sha256": stable_values_sha256(
                        [row["id"] for row in prepared["context"][2]]
                    ),
                    "corrected_game_ids_sha256": stable_values_sha256(
                        [row["id"] for row in context_rows]
                    ),
                    "corrected_game_rows_sha256": included_game_rows_sha256(
                        context_rows
                    ),
                    "real_result_retained": has_replacement,
                    "sources": {
                        kind: {
                            "snapshot_id": output.snapshot_id,
                            "path": (
                                Path("data/processed/snapshots")
                                / output.directory.relative_to(stage_root)
                            ).as_posix(),
                            "prior_family": output.metadata.get("prior_family"),
                            "prior_model_version": output.metadata.get(
                                "prior_model_version"
                            ),
                            "requested_cutoff": output.metadata.get(
                                "requested_cutoff"
                            ),
                            "effective_cutoff": output.metadata.get(
                                "effective_cutoff"
                            ),
                        }
                        for kind, output in role_outputs.items()
                    },
                }
            )

        _assert_unaffected_sources_unchanged(root, preserved_hashes)
        report: dict[str, Any] = {
            "schema_version": "1.0",
            "issue": ISSUE,
            "lineage_suffix": LINEAGE_SUFFIX,
            "normalized_game_corpus_sha256": sha256(
                root / "data/processed/cfbd/games.csv"
            ),
            "normalized_phantom_present": False,
            "normalized_replacement_result": {
                "game_id": REPLACEMENT_GAME_ID,
                "home_team": replacement_rows[0]["homeTeam"],
                "home_points": replacement_rows[0]["homePoints"],
                "away_team": replacement_rows[0]["awayTeam"],
                "away_points": replacement_rows[0]["awayPoints"],
            },
            "snapshot_fixture_audit": fixture_audit,
            "publications": repaired_publications,
            "superseded_artifacts": audit_sources,
            "ranking_change_summaries": ranking_change_summaries,
            "performance_backfill": {
                "status": "not_run" if dry_run else None,
                "publication_count": None,
                "blocked_publication_count": None,
            },
        }
        if dry_run:
            report["performance_backfill"] = {
                "status": "not_run",
                "publication_count": 0,
                "blocked_publication_count": 0,
            }
            report["dry_run"] = True
            report["ranking_changes"] = ranking_changes
            return report

        print("Installing repaired snapshots and publication config...", flush=True)
        _install_outputs(root, stage_root, path_replacements, config)
        print("Rebuilding dependent performance variants...", flush=True)
        backfill = backfill_performance_variants(root=root)
        if backfill["status"] != "complete" or backfill["blockers"]:
            raise ValueError(f"Historical Performance backfill is blocked: {backfill}")
        print("Rebuilding and validating site and static API exports...", flush=True)
        build_site_data(
            root=root,
            config_path=root / CONFIG_PATH,
            output_directory=root / "site/data",
            static_api_directory=root / "site/api/v1",
        )
        report["performance_backfill"] = {
            "status": backfill["status"],
            "publication_count": backfill["publication_count"],
            "blocked_publication_count": backfill["blocked_publication_count"],
        }
        report["site_export"] = _validate_site_exports(root)
        report["ranking_changes"] = ranking_changes
        report["dry_run"] = False
        audit_directory = root / AUDIT_ROOT
        _write_json(audit_directory / "lineage_audit.json", report)
        _write_change_csv(audit_directory / "ranking_changes.csv", ranking_changes)
        _write_markdown(root / "docs/issue_238_historical_repair.md", report)
        return report


def _read_normalized_game_corpus(root: Path) -> list[dict[str, str]]:
    path = root / "data/processed/cfbd/games.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _snapshot_fixture_audit(
    root: Path, affected: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    selected_paths = {
        (root / item["source"]).resolve()
        for sources in affected.values()
        for item in sources.values()
    }
    manifest = _read_json(root / "site/data/manifest.json")
    published = {
        str(item["snapshot_id"]): item for item in manifest.get("snapshots", [])
    }
    contaminated = []
    scanned = 0
    for games_path in sorted(
        (root / "data/processed/snapshots/2026").rglob("included_games.csv")
    ):
        scanned += 1
        try:
            rows = _read_included_game_rows(games_path)
        except (ValueError, OSError):
            continue
        if not any(row["id"] == PHANTOM_GAME_ID for row in rows):
            continue
        directory = games_path.parent
        metadata = _read_json(directory / "metadata.json")
        snapshot_id = str(metadata["snapshot_id"])
        entry = published.get(snapshot_id)
        selected = directory.resolve() in selected_paths
        contaminated.append(
            {
                "snapshot_id": snapshot_id,
                "path": directory.relative_to(root).as_posix(),
                "prior_family": metadata.get("prior_family"),
                "prior_model_version": metadata.get("prior_model_version"),
                "performance_variant": metadata.get("performance_variant"),
                "publication_slot": None if entry is None else entry.get("publication_slot"),
                "currently_published": entry is not None,
                "selected_for_repair": selected,
                "included_game_count": len(rows),
            }
        )
    selected = [item for item in contaminated if item["selected_for_repair"]]
    unselected = [item for item in contaminated if not item["selected_for_repair"]]
    if any(not item["currently_published"] for item in selected):
        raise ValueError("A selected repair source is absent from the active manifest")
    return {
        "scanned_snapshot_bundle_count": scanned,
        "contaminated_bundle_count": len(contaminated),
        "selected_contaminated_bundle_count": len(selected),
        "selected_contaminated_bundles": selected,
        "unselected_contaminated_bundle_count": len(unselected),
        "unselected_contaminated_bundles": unselected,
    }


def _unaffected_source_hashes(
    root: Path, affected: dict[str, dict[str, Any]]
) -> dict[Path, str]:
    affected_paths = {
        item["source"] for sources in affected.values() for item in sources.values()
    }
    config = _read_json(root / CONFIG_PATH)
    hashes: dict[Path, str] = {}
    for entry in config["snapshots"]:
        source = str(entry["source"])
        if source in affected_paths:
            continue
        directory = root / source
        for path in directory.rglob("*"):
            if path.is_file():
                hashes[path] = sha256(path)
    return hashes


def _assert_unaffected_sources_unchanged(root: Path, hashes: dict[Path, str]) -> None:
    changed = [path for path, digest in hashes.items() if sha256(path) != digest]
    if changed:
        raise ValueError(
            "Unaffected publication source artifacts changed: "
            + ", ".join(str(path) for path in changed[:10])
        )


def _install_outputs(
    root: Path,
    stage_root: Path,
    path_replacements: dict[str, str],
    config: dict[str, Any],
) -> None:
    output_paths = sorted(set(path_replacements.values()))
    roots = {Path(relative) for relative in output_paths}
    performance_roots = {
        path.parent if path.name == "quality_of_record" else path
        for path in roots
        if "performance" in path.parts
    }
    copy_roots = {
        path
        for path in roots
        if "performance" not in path.parts
    } | performance_roots
    for relative in sorted(copy_roots, key=lambda item: item.as_posix()):
        staged = stage_root / relative.relative_to("data/processed/snapshots")
        destination = root / relative
        if not staged.exists():
            raise FileNotFoundError(staged)
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staged, destination)
    for entry in config["snapshots"]:
        current = str(entry["source"])
        if current in path_replacements:
            entry["source"] = path_replacements[current]
    _write_json(root / CONFIG_PATH, config)


def _validate_site_exports(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "site/data/manifest.json")
    published_snapshot_ids = {
        str(entry["snapshot_id"]) for entry in manifest.get("snapshots", [])
    }
    publications = [
        entry
        for entry in manifest.get("snapshots", [])
        if entry.get("publication_slot") in TARGET_SLOTS
    ]
    if any("quality_of_record_evidence" in entry for entry in publications):
        raise ValueError("Site manifest still contains the retired QoR exception")
    if len(publications) != 12:
        raise ValueError(
            f"Expected 12 repaired publication variants, found {len(publications)}"
        )
    api_publications = _read_json(root / "site/api/v1/publications.json")
    if any(
        "quality_of_record_evidence" in entry
        for entry in api_publications.get("publications", [])
    ):
        raise ValueError("Static API still contains the retired QoR exception")
    removed_unreferenced_exports = []
    for path in (root / "site/data", root / "site/api/v1"):
        for artifact in path.rglob("*.json"):
            value = artifact.read_text(encoding="utf-8")
            if "quality_of_record_evidence" in value or "inference_games.csv" in value:
                try:
                    payload = json.loads(value)
                except json.JSONDecodeError:
                    payload = {}
                snapshot_id = (
                    payload.get("snapshot_id") if isinstance(payload, dict) else None
                )
                if (
                    artifact.parent == root / "site/data/snapshots"
                    and isinstance(snapshot_id, str)
                    and snapshot_id not in published_snapshot_ids
                ):
                    artifact.unlink()
                    removed_unreferenced_exports.append(
                        artifact.relative_to(root).as_posix()
                    )
                    continue
                raise ValueError(f"Retired QoR exception remains in {artifact}")
    return {
        "published_variant_count": len(publications),
        "static_api_publication_count": len(api_publications.get("publications", [])),
        "retired_exception_fields": 0,
        "removed_unreferenced_retired_exports": removed_unreferenced_exports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and compare repaired rankings without registering outputs",
    )
    args = parser.parse_args()
    report = repair_historical_publications(root=ROOT, dry_run=args.dry_run)
    print(
        json.dumps(
            {
                "dry_run": report["dry_run"],
                "publications": [
                    {
                        "slot": item["publication_slot"],
                        "games_before": item["source_game_count"],
                        "games_after": item["corrected_game_count"],
                        "real_result_retained": item["real_result_retained"],
                    }
                    for item in report["publications"]
                ],
                "week_2_ranking_changes": [
                    {
                        "ranking": item["ranking"],
                        "display_rank_changes": item[
                            "teams_with_changed_display_rank"
                        ],
                        "largest_movers": item["largest_movers"],
                    }
                    for item in report["ranking_change_summaries"]
                    if item["publication_slot"] == "2026-09-08"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
