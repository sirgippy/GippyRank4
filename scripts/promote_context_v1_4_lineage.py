"""Replay the registered 2026 Context 1.4 states before activation.

``replay`` writes versioned, noncanonical snapshots and a parity audit. Only
``activate`` changes the publication inventory, after rechecking that audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from gippyrank.artifact_hashes import included_game_rows_sha256, stable_values_sha256
from gippyrank.context_v1_4_validation_protocol import load_registered_protocol
from gippyrank.performance_snapshot import (
    build_performance_snapshot,
    validate_performance_against_context,
)
from gippyrank.posterior.snapshots import Snapshot, build_snapshot, snapshot_id
from gippyrank.weekly_update import _same_evidence

ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = Path("data/processed/context_v1_4_production/migration_audit.json")
PROMOTION_PATH = Path(
    "data/processed/preseason/context_v1_4/annual/2026/promotion.json"
)
CONFIG_PATH = Path("site/publish_config.json")
SLOTS = {
    "2026-preseason-context-1.3": "2026-preseason",
    "2026-09-08": "2026-09-08",
    "2026-09-13": "2026-09-13",
    "2026-09-20": "2026-09-20",
    "2026-09-27": "2026-09-27",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(root: Path, origin: dict[str, Any]) -> Path:
    return (root / origin["context_files"]["metadata.json"]["path"]).parent


def _snapshot(root: Path, relative: str) -> Snapshot:
    directory = root / relative
    metadata = _json(directory / "metadata.json")
    return Snapshot(str(metadata["snapshot_id"]), directory, metadata)


def _registered_candidate(provenance: dict[str, Any], slot: str) -> dict[str, Any]:
    matches = [
        row
        for row in provenance["forecast_origins"]
        if row["origin"] == slot and row["model"] == "candidate"
    ]
    if len(matches) != 1:
        raise ValueError(f"{slot}: registered candidate state is ambiguous")
    return matches[0]


def _parity_row(
    *,
    root: Path,
    origin: dict[str, Any],
    snapshot: Snapshot,
    candidate: dict[str, Any],
    promotion: dict[str, Any],
    likelihood_sha256: str,
    inference: dict[str, Any],
) -> dict[str, Any]:
    slot = str(origin["publication_slot"])
    source = _source(root, origin)
    source_metadata = _json(source / "metadata.json")
    actual = snapshot.metadata
    with (source / "included_games.csv").open(newline="", encoding="utf-8") as handle:
        source_evidence_hash = included_game_rows_sha256(list(csv.DictReader(handle)))
    with (source / "rankings.csv").open(newline="", encoding="utf-8") as handle:
        source_fbs_ids = sorted(
            row["team_id"]
            for row in csv.DictReader(handle)
            if row["subdivision"].casefold() == "fbs"
        )
    checks = {
        "included_game_count": (
            actual["included_game_count"]
            == candidate["included_game_count"]
            == origin["included_game_count"]
        ),
        "included_game_evidence": (
            _sha(snapshot.directory / "included_games.csv")
            == candidate["included_games_sha256"]
            == origin["context_files"]["included_games.csv"]["sha256"]
            and actual["included_game_rows_sha256"] == source_evidence_hash
            and actual["included_game_ids"] == source_metadata["included_game_ids"]
        ),
        "cutoff": (
            actual["requested_cutoff"] == source_metadata["requested_cutoff"]
            and actual["effective_cutoff"] == source_metadata["effective_cutoff"]
        ),
        "fcs_fallback": (
            actual["fcs_fallback_team_ids"] == source_metadata["fcs_fallback_team_ids"]
            and actual["fcs_population_size"] == source_metadata["fcs_population_size"]
            and actual["fcs_fallback_count"] == candidate["fcs_fallback_count"]
        ),
        "team_population": (
            actual["fbs_team_count"] == 138
            and actual["fbs_team_keys_sha256"] == stable_values_sha256(source_fbs_ids)
            and actual["fbs_team_count"] + actual["fcs_fallback_count"]
            == candidate["team_count"]
        ),
        "inference_configuration": (
            all(
                actual["posterior_inference_configuration"][key] == value
                for key, value in inference.items()
            )
            and actual["valid"] is True
        ),
        "prior_identity": (
            actual["prior_model_version"] == "1.4"
            and actual["prior_artifact_sha256"] == promotion["predictions_sha256"]
            and actual["prior_pmfs_sha256"] == candidate["prior_pmfs_sha256"]
        ),
        "historical_likelihood": (
            actual["historical_likelihood_version"] == "V1"
            and actual["historical_likelihood_sha256"] == likelihood_sha256
        ),
        "posterior_hash": actual["posterior_pmfs_sha256"]
        == candidate["posterior_sha256"],
    }
    return {
        "origin": slot,
        "publication_slot": SLOTS[slot],
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_path": snapshot.directory.relative_to(root).as_posix(),
        "snapshot_metadata_sha256": _sha(snapshot.directory / "metadata.json"),
        "included_games": actual["included_game_count"],
        "candidate_posterior_sha256": candidate["posterior_sha256"],
        "production_posterior_sha256": actual["posterior_pmfs_sha256"],
        "checks": checks,
        "parity": "PASS" if all(checks.values()) else "FAIL",
    }


def replay(root: Path = ROOT, *, reuse_existing: bool = False) -> dict[str, Any]:
    protocol = load_registered_protocol(root)
    provenance = _json(root / "data/processed/context_v1_4_validation/provenance.json")
    validation = _json(root / "data/processed/context_v1_4_validation/summary.json")
    promotion = _json(root / PROMOTION_PATH)
    if (
        validation["decision"] != "promote"
        or promotion["validation_decision"] != "promote"
    ):
        raise ValueError("Context 1.4 promotion verdict is inconsistent")
    if (
        promotion["candidate_semantics_sha256"]
        != provenance["candidate_semantics_sha256"]
    ):
        raise ValueError("production/candidate semantic identities differ")
    if promotion["predictions_sha256"] != _sha(
        root / "data/processed/preseason/context_v1_4/annual/2026/predictions.csv"
    ):
        raise ValueError("production annual prior changed after its parity gate")
    rows: list[dict[str, Any]] = []
    for origin in protocol.data["forecast_origins"]:
        slot = str(origin["publication_slot"])
        source = _source(root, origin)
        source_metadata = _json(source / "metadata.json")
        cutoff = (
            datetime.fromisoformat(str(source_metadata["requested_cutoff"]))
            if source_metadata["requested_cutoff"] is not None
            else None
        )
        schedule = (
            root / "site/data/week-games" / f"{source_metadata['snapshot_id']}.json"
        )
        sid = snapshot_id(
            2026, source_metadata["snapshot_type"], "context", cutoff, "1.4"
        )
        destination = (
            root / "data/processed/snapshots/2026" / sid / "predictive/context"
        )
        if reuse_existing and (destination / "metadata.json").is_file():
            snapshot = _snapshot(root, destination.relative_to(root).as_posix())
        else:
            snapshot = build_snapshot(
                season=2026,
                cutoff=cutoff,
                prior_family="context",
                snapshot_type=source_metadata["snapshot_type"],
                prior_model_version="1.4",
                root=root,
                evidence_snapshot=source,
                presentation_schedule_path=schedule,
                canonical_lineage_replay=True,
            )
        row = _parity_row(
            root=root,
            origin=origin,
            snapshot=snapshot,
            candidate=_registered_candidate(provenance, slot),
            promotion=promotion,
            likelihood_sha256=protocol.data["evidence"]["likelihood_sha256"],
            inference=protocol.data["evidence"]["inference"],
        )
        rows.append(row)
        print(f"{slot}: {row['parity']} ({row['included_games']} games)", flush=True)
        if row["parity"] != "PASS":
            break
    audit: dict[str, Any] = {
        "production_context_version": "1.4",
        "production_semantics_sha256": promotion["production_semantics_sha256"],
        "candidate_semantics_sha256": provenance["candidate_semantics_sha256"],
        "candidate_validation_verdict": validation["decision"],
        "validation_result_sha256": promotion["validation_result_sha256"],
        "annual_prior_sha256": promotion["predictions_sha256"],
        "registered_protocol_sha256": protocol.sha256,
        "publication_config_before_sha256": _sha(root / CONFIG_PATH),
        "historical_origins": rows,
        "all_historical_parity": len(rows) == 5
        and all(row["parity"] == "PASS" for row in rows),
    }
    path = root / AUDIT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit["all_historical_parity"]:
        raise ValueError(
            "historical parity failed; publication configuration unchanged"
        )
    return audit


def activate(root: Path = ROOT) -> dict[str, Any]:
    audit = _json(root / AUDIT_PATH)
    if not audit["all_historical_parity"] or len(audit["historical_origins"]) != 5:
        raise ValueError("all five historical states must pass before activation")
    if _sha(root / CONFIG_PATH) != audit["publication_config_before_sha256"]:
        raise ValueError("publication config changed after historical replay")
    if (
        _sha(root / "data/processed/preseason/context_v1_4/annual/2026/predictions.csv")
        != audit["annual_prior_sha256"]
    ):
        raise ValueError("annual prior changed after historical replay")
    by_slot: dict[str, tuple[Snapshot, Snapshot | None]] = {}
    for row in audit["historical_origins"]:
        if row["parity"] != "PASS" or not all(row["checks"].values()):
            raise ValueError(f"{row['origin']}: parity gate failed")
        context = _snapshot(root, row["snapshot_path"])
        if (
            _sha(context.directory / "metadata.json") != row["snapshot_metadata_sha256"]
            or context.metadata["posterior_pmfs_sha256"]
            != row["candidate_posterior_sha256"]
        ):
            raise ValueError(
                f"{row['origin']}: generated artifact changed after parity"
            )
        performance = None
        if context.metadata["snapshot_type"] == "weekly":
            performance = build_performance_snapshot(
                context,
                root=root,
                source_context_path=context.directory,
            )
            validate_performance_against_context(context, performance)
        by_slot[row["publication_slot"]] = (context, performance)

    config = _json(root / CONFIG_PATH)
    remove_slots = {"2026-preseason-context-1.3", "2026-09-19-context-1.3"}
    config["publication_slots"] = [
        item for item in config["publication_slots"] if item["id"] not in remove_slots
    ]
    kept = [
        item
        for item in config["snapshots"]
        if item["publication_slot"] not in remove_slots
        and not (
            item["publication_slot"] in by_slot
            and (
                item["source"].endswith("/predictive/context")
                or item["source"].endswith("/performance")
            )
        )
    ]
    for slot, (context, performance) in by_slot.items():
        label = (
            "Preseason"
            if slot == "2026-preseason"
            else {
                "2026-09-08": "Week 2",
                "2026-09-13": "Week 3",
                "2026-09-20": "Week 4",
                "2026-09-27": "Week 5",
            }[slot]
        )
        for snapshot in (context, performance):
            if snapshot is not None:
                kept.append(
                    {
                        "display_label": label,
                        "publication_slot": slot,
                        "source": snapshot.directory.relative_to(root).as_posix(),
                    }
                )
    config["snapshots"] = kept
    (root / CONFIG_PATH).write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit["publication_config_after_sha256"] = _sha(root / CONFIG_PATH)
    audit["historical_publication_activated"] = True
    (root / AUDIT_PATH).write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def finalize_week_6(root: Path = ROOT) -> dict[str, Any]:
    """Add the first ordinary production run to the promotion audit."""
    audit = _json(root / AUDIT_PATH)
    if not audit.get("historical_publication_activated"):
        raise ValueError("historical lineage was not activated")
    config = _json(root / CONFIG_PATH)
    week_6_entries = [
        entry
        for entry in config["snapshots"]
        if entry["publication_slot"] == "2026-10-04"
    ]
    if len(week_6_entries) != 3:
        raise ValueError("Week 6 requires Context, History, and Performance")
    snapshots: dict[str, Snapshot] = {}
    for entry in week_6_entries:
        snapshot = _snapshot(root, entry["source"])
        family = (
            snapshot.metadata.get("prior_family")
            if snapshot.metadata["ranking_family"] == "predictive"
            else "performance"
        )
        if family in snapshots:
            raise ValueError(f"duplicate Week 6 {family} snapshot")
        snapshots[str(family)] = snapshot
    if set(snapshots) != {"context", "history", "performance"}:
        raise ValueError("Week 6 publication families are incomplete")
    context = snapshots["context"]
    history = snapshots["history"]
    performance = snapshots["performance"]
    _same_evidence(context, history)
    validate_performance_against_context(context, performance)
    c = context.metadata
    h = history.metadata
    p = performance.metadata
    prior_week = next(
        row
        for row in audit["historical_origins"]
        if row["publication_slot"] == "2026-09-27"
    )
    prior_metadata = _json(root / prior_week["snapshot_path"] / "metadata.json")
    previous_ids = set(prior_metadata["included_game_ids"])
    current_ids = set(c["included_game_ids"])
    checks = {
        "valid_convergence": c["valid"] is True and h["valid"] is True,
        "active_context_prior": c["prior_model_version"] == "1.4",
        "retained_history_prior": h["prior_model_version"] == "1.1",
        "fbs_population": c["fbs_team_count"] == h["fbs_team_count"] == 138,
        "likelihood": c["historical_likelihood_version"]
        == h["historical_likelihood_version"]
        == "V1",
        "shared_evidence": (
            c["included_game_ids"] == h["included_game_ids"]
            and c["included_game_rows_sha256"] == h["included_game_rows_sha256"]
            and c["effective_cutoff"] == h["effective_cutoff"]
            and c["fcs_fallback_team_ids"] == h["fcs_fallback_team_ids"]
        ),
        "unique_included_games": len(current_ids) == c["included_game_count"],
        "retained_week_5_evidence": previous_ids <= current_ids,
        "performance_context_source": p["source_context_snapshot_id"]
        == context.snapshot_id,
    }
    if not all(checks.values()):
        raise ValueError(f"Week 6 production audit failed: {checks}")
    audit["week_6"] = {
        "publication_slot": "2026-10-04",
        "effective_cutoff": c["effective_cutoff"],
        "included_completed_games": c["included_game_count"],
        "new_games_since_week_5": len(current_ids - previous_ids),
        "new_included_game_ids": sorted(current_ids - previous_ids),
        "context_snapshot_id": context.snapshot_id,
        "context_posterior_sha256": c["posterior_pmfs_sha256"],
        "history_snapshot_id": history.snapshot_id,
        "history_posterior_sha256": h["posterior_pmfs_sha256"],
        "performance_snapshot_id": performance.snapshot_id,
        "checks": checks,
        "shared_evidence_status": "PASS",
    }
    (root / AUDIT_PATH).write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("replay", "activate", "finalize-week-6"))
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    if args.stage == "replay":
        result = replay(reuse_existing=args.reuse_existing)
    elif args.stage == "activate":
        result = activate()
    else:
        result = finalize_week_6()
    print(
        json.dumps(
            {
                "stage": args.stage,
                "all_historical_parity": result["all_historical_parity"],
            }
        )
    )


if __name__ == "__main__":
    main()
