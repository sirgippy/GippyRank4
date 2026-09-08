"""Transactional preparation of a reviewed weekly GippyRank publication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from gippyrank.data.cfbd import fetch_current_season, update_processed_game_corpus
from gippyrank.performance_snapshot import (
    build_performance_snapshot,
    validate_performance_against_context,
)
from gippyrank.posterior.snapshots import Snapshot, build_snapshot
from gippyrank.site_data import build_site_data

SLOT_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class WeeklyUpdate:
    season: int
    publication_slot: str
    display_label: str
    requested_cutoff: datetime
    effective_cutoff: datetime
    context: Snapshot
    history: Snapshot
    performance: Snapshot
    corpus: dict[str, int]
    published: bool
    report: dict[str, Any]
    candidate_paths: PublicationCandidatePaths | None = None


@dataclass(frozen=True)
class PublicationCandidatePaths:
    """Exact durable files and directories produced for a publishable update."""

    fbs_schedule: Path
    fbs_provenance: Path
    fcs_schedule: Path
    fcs_provenance: Path
    processed_games: Path
    context_snapshot: Path
    history_snapshot: Path
    performance_snapshot: Path
    report_md: Path
    report_json: Path
    publish_config: Path
    site_data: Path


def _root() -> Path:
    """Locate the repository from this module's checked-in project markers.

    The CLI is installed/imported from ``src/gippyrank`` in both local and
    GitHub Actions environments, so searching upward is less brittle than a
    fixed ``parents[n]`` index.  A valid root owns both the Python project and
    the explicit site publication configuration.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "site/publish_config.json"
        ).is_file():
            return candidate
    raise RuntimeError("Cannot locate repository root from weekly_update.py")


def _label(value: datetime) -> str:
    return value.strftime("%b. ") + str(value.day)


def _validate_slot(season: int, slot: str) -> str:
    if not SLOT_PATTERN.fullmatch(slot) or not slot.startswith(f"{season}-"):
        raise ValueError("publication_slot must be an ISO date in the selected season")
    return slot


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(directory).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _github_output_lines(update: WeeklyUpdate, *, root: Path) -> list[str]:
    """Return Action outputs, including exact paths only for a real candidate.

    Snapshot names contain the actual successful generation timestamp.  Keeping
    those values here avoids a workflow independently re-creating a path that
    can diverge from the snapshot directory produced by the orchestration.
    """
    lines = [
        f"published={'true' if update.published else 'false'}",
        f"slot={update.publication_slot}",
        f"branch=automation/rankings-{update.publication_slot}",
        # Kept for existing callers that use the Markdown report as the PR body.
        f"report_path=data/processed/weekly_updates/{update.publication_slot}.md",
    ]
    if not update.published:
        return lines
    if update.candidate_paths is None:
        raise ValueError("A publishable weekly update must include candidate paths")
    for name, path in (
        ("fbs_schedule_path", update.candidate_paths.fbs_schedule),
        ("fbs_provenance_path", update.candidate_paths.fbs_provenance),
        ("fcs_schedule_path", update.candidate_paths.fcs_schedule),
        ("fcs_provenance_path", update.candidate_paths.fcs_provenance),
        ("processed_games_path", update.candidate_paths.processed_games),
        ("context_snapshot_path", update.candidate_paths.context_snapshot),
        ("history_snapshot_path", update.candidate_paths.history_snapshot),
        ("performance_snapshot_path", update.candidate_paths.performance_snapshot),
        ("report_md_path", update.candidate_paths.report_md),
        ("report_json_path", update.candidate_paths.report_json),
        ("publish_config_path", update.candidate_paths.publish_config),
        ("site_data_path", update.candidate_paths.site_data),
    ):
        lines.append(f"{name}={path.relative_to(root).as_posix()}")
    return lines


def _same_evidence(context: Snapshot, history: Snapshot) -> None:
    fields = (
        "requested_cutoff", "effective_cutoff", "source_retrieved_at",
        "source_retrieval_times", "source_response_hashes", "game_corpus_sha256",
        "included_game_ids",
    )
    for field in fields:
        if context.metadata[field] != history.metadata[field]:
            raise ValueError(f"Context/History evidence mismatch: {field}")
    if not context.metadata["valid"] or not history.metadata["valid"]:
        raise ValueError("Both Context and History snapshots must converge and be valid")


def _ranking_signature(snapshot: Snapshot) -> tuple[bytes, bytes]:
    return (
        (snapshot.directory / "rankings.csv").read_bytes(),
        (snapshot.directory / "included_games.csv").read_bytes(),
    )


def _configured_snapshot(root: Path, config: dict[str, Any], family: str) -> Path | None:
    slot = config.get("default_publication_slot")
    suffix = {
        "context": "/predictive/context",
        "history": "/predictive/history",
        "performance": "/performance",
    }.get(family)
    if suffix is None:
        raise ValueError(f"Unsupported publication family: {family}")
    for entry in config.get("snapshots", []):
        if entry.get("publication_slot") == slot and entry.get("source", "").endswith(suffix):
            return root / entry["source"]
    return None


def _is_publishable_change(
    root: Path,
    config: dict[str, Any],
    context: Snapshot,
    history: Snapshot,
    performance: Snapshot,
) -> bool:
    for family, snapshot in (
        ("context", context),
        ("history", history),
        ("performance", performance),
    ):
        previous = _configured_snapshot(root, config, family)
        if previous is None or not previous.is_dir():
            return True
        if _ranking_signature(snapshot) != (
            (previous / "rankings.csv").read_bytes(),
            (previous / "included_games.csv").read_bytes(),
        ):
            return True
    return False


def _upsert_publication(
    *, root: Path, config_path: Path, context: Snapshot, history: Snapshot,
    slot: str, label: str, performance: Snapshot | None = None,
) -> None:
    config = _load_json(config_path)
    slot_entries = config.get("publication_slots")
    if not isinstance(slot_entries, list):
        raise TypeError(
            "Publish configuration needs explicit publication_slots metadata before updates"
        )
    slot_ids = {
        entry.get("id")
        for entry in slot_entries
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    if slot not in slot_ids:
        slot_entries.append({"id": slot, "status": "temporary"})
    config["publication_slots"] = slot_entries
    retained = [
        entry for entry in config["snapshots"]
        if entry.get("publication_slot") != slot
    ]
    snapshots = [context, history]
    if performance is not None:
        snapshots.append(performance)
    for snapshot in snapshots:
        retained.append(
            {
                "source": snapshot.directory.relative_to(root).as_posix(),
                "display_label": label,
                "publication_slot": slot,
            }
        )
    config["snapshots"] = retained
    config["default_publication_slot"] = slot
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rows(snapshot: Snapshot) -> list[dict[str, str]]:
    with (snapshot.directory / "rankings.csv").open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["subdivision"] == "fbs"]


def _top25(snapshot: Snapshot) -> str:
    return "\n".join(
        f"{row['display_rank']}. {row['team_name']}"
        for row in sorted(
            (row for row in _rows(snapshot) if row["display_rank"] != "NR"),
            key=lambda row: int(row["display_rank"]),
        )[:25]
    )


def _diagnostics(snapshot: Snapshot) -> dict[str, Any]:
    return _load_json(snapshot.directory / "diagnostics.json")


def _performance_differences(
    context: Snapshot, performance: Snapshot
) -> list[dict[str, object]]:
    context_rows = {row["team_id"]: row for row in _rows(context)}
    values = []
    for row in _rows(performance):
        if row["team_id"] not in context_rows or row.get("rated", "True") != "True":
            continue
        context_row = context_rows[row["team_id"]]
        difference = float(row["expected_rank"]) - float(context_row["expected_rank"])
        values.append(
            {
                "team": row["team_name"],
                "performance_expected_rank": float(row["expected_rank"]),
                "context_expected_rank": float(context_row["expected_rank"]),
                "difference": difference,
                "absolute_difference": abs(difference),
            }
        )
    return sorted(
        values, key=lambda item: (-float(item["absolute_difference"]), str(item["team"]))
    )[:10]


def _broadest_performance(performance: Snapshot) -> list[dict[str, object]]:
    values = [
        {
            "team": row["team_name"],
            "team_id": row["team_id"],
            "interval_80_width": int(float(row["interval_80_width"])),
            "interval_80": [
                int(float(row["interval_80_low"])),
                int(float(row["interval_80_high"])),
            ],
        }
        for row in _rows(performance)
        if row.get("rated", "True") == "True"
    ]
    return sorted(values, key=lambda item: (-int(item["interval_80_width"]), str(item["team"])))[:10]


def _performance_counts(performance: Snapshot) -> dict[str, int]:
    rows = _rows(performance)
    return {
        "rated_count": sum(row.get("rated", "True") == "True" for row in rows),
        "unrated_count": sum(row.get("rated", "True") != "True" for row in rows),
    }


def _newly_rated(performance: Snapshot, previous: list[dict[str, str]]) -> list[str]:
    previous_rated = {row["team_id"] for row in previous if row.get("rated") == "True"}
    return [
        row["team_name"]
        for row in _rows(performance)
        if row.get("rated") == "True" and row["team_id"] not in previous_rated
    ]


def _report(
    context: Snapshot,
    history: Snapshot,
    performance: Snapshot,
    *,
    slot: str,
    label: str,
    corpus: dict[str, int],
) -> dict[str, Any]:
    metadata = context.metadata
    return {
        "season": metadata["season"], "publication_slot": slot, "display_label": label,
        "requested_cutoff": metadata["requested_cutoff"], "effective_cutoff": metadata["effective_cutoff"],
        "source_retrieved_at": metadata["source_retrieved_at"],
        "source_retrieval_times": metadata["source_retrieval_times"],
        "eligible_game_count": metadata["included_game_count"],
        "excluded_lower_division_games": metadata["excluded_lower_division_games"],
        "fcs_population_size": metadata["fcs_population_size"], "fcs_fallback_count": metadata["fcs_fallback_count"],
        "schedule_overlap_count": corpus["overlap_count"],
        "context": {**_diagnostics(context), "top25": _top25(context)},
        "history": {**_diagnostics(history), "top25": _top25(history)},
        "performance": {
            **_diagnostics(performance),
            "top25": _top25(performance),
            **_performance_counts(performance),
            "largest_expected_rank_differences": _performance_differences(context, performance),
            "broadest_distributions": _broadest_performance(performance),
        },
        "validation": "passed",
    }


def _previous_rows(root: Path, config: dict[str, Any], family: str) -> list[dict[str, str]]:
    previous = _configured_snapshot(root, config, family)
    if previous is None or not previous.exists():
        return []
    with (previous / "rankings.csv").open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["subdivision"] == "fbs"]


def _movement(current: Snapshot, previous: list[dict[str, str]]) -> list[dict[str, object]]:
    old = {row["team_id"]: int(row["display_rank"]) for row in previous}
    changes = []
    for row in _rows(current):
        if row["team_id"] in old:
            delta = old[row["team_id"]] - int(row["display_rank"])
            changes.append({"team": row["team_name"], "rank_change": delta})
    return sorted(changes, key=lambda item: (-abs(int(item["rank_change"])), str(item["team"])))[:10]


def _disagreements(context: Snapshot, history: Snapshot) -> list[dict[str, object]]:
    history_ranks = {row["team_id"]: int(row["display_rank"]) for row in _rows(history)}
    values = [
        {"team": row["team_name"], "context_rank": int(row["display_rank"]), "history_rank": history_ranks[row["team_id"]], "difference": abs(int(row["display_rank"]) - history_ranks[row["team_id"]])}
        for row in _rows(context) if row["team_id"] in history_ranks
    ]
    return sorted(values, key=lambda item: (-int(item["difference"]), str(item["team"])))[:10]


def render_review_markdown(report: dict[str, Any]) -> str:
    newly_rated = report["newly_rated_teams"]
    newly_rated_summary = f"{len(newly_rated)} teams"
    if newly_rated:
        preview = ", ".join(newly_rated[:20])
        newly_rated_summary += f": {preview}"
        if len(newly_rated) > 20:
            newly_rated_summary += ", …"

    def section(name: str, values: dict[str, Any]) -> str:
        return (
            f"## {name}\n\n"
            f"Converged: `{values['converged']}` · iterations: `{values['iterations']}` · "
            f"runtime: `{values['runtime_seconds']:.2f}s`\n\n"
            f"### Top 25\n\n{values['top25']}\n"
        )
    return (
        "# GippyRank weekly publication candidate\n\n"
        f"- Publication slot: `{report['publication_slot']}` ({report['display_label']})\n"
        f"- Acquisition/effective cutoff: `{report['source_retrieved_at']}` / `{report['effective_cutoff']}`\n"
        f"- Source retrieval times: FBS `{report['source_retrieval_times'].get('fbs')}`; FCS `{report['source_retrieval_times'].get('fcs')}`\n"
        f"- Eligible games: `{report['eligible_game_count']}`; lower-division excluded: `{report['excluded_lower_division_games']}`\n"
        f"- Newly eligible games since previous publication: `{report['new_eligible_game_count']}`\n"
        f"- FCS population/fallbacks: `{report['fcs_population_size']}` / `{report['fcs_fallback_count']}`\n"
        "- Validation: `passed` (shared cutoff, corpus, and eligible game set)\n\n"
        + section("Context", report["context"]) + "\n" + section("History", report["history"])
        + "\n## Performance\n\n"
        + f"Rated teams: `{report['performance']['rated_count']}` · unrated/NR teams: `{report['performance']['unrated_count']}` · "
        + f"transformation runtime: `{report['performance']['transformation_runtime_seconds']:.4f}s`\n\n"
        + "### Top 25\n\n" + report["performance"]["top25"] + "\n\n"
        + "### Largest Performance / Predictive Context expected-rank differences\n\n"
        + "\n".join(
            f"- {item['team']}: Performance {item['performance_expected_rank']:.1f}, Context {item['context_expected_rank']:.1f} (Δ {item['difference']:+.1f})"
            for item in report["performance"]["largest_expected_rank_differences"]
        )
        + "\n\n### Broadest Performance distributions\n\n"
        + "\n".join(
            f"- {item['team']}: 80% interval {item['interval_80'][0]}–{item['interval_80'][1]} ({item['interval_80_width']} ranks)"
            for item in report["performance"]["broadest_distributions"]
        )
        + "\n\n### Newly rated teams\n\n"
        + newly_rated_summary
        + "\n## Review diagnostics\n\n"
        + "### Context biggest movers\n\n"
        + "\n".join(f"- {item['team']}: {item['rank_change']:+d} display ranks" for item in report["context_movers"])
        + "\n\n### History biggest movers\n\n"
        + "\n".join(f"- {item['team']}: {item['rank_change']:+d} display ranks" for item in report["history_movers"])
        + "\n\n### Largest Context/History disagreements\n\n"
        + "\n".join(f"- {item['team']}: Context {item['context_rank']}, History {item['history_rank']} (Δ {item['difference']})" for item in report["h_c_disagreements"])
        + "\n"
    )


def prepare_weekly_update(
    *, season: int, root: Path | None = None, display_label: str | None = None,
    publication_slot: str | None = None, retrieved_at: datetime | None = None,
) -> WeeklyUpdate:
    """Fetch, infer, validate, and prepare static publication data atomically.

    Git branch and pull-request operations intentionally belong to the Action,
    not this reusable local orchestration function.
    """
    root = _root() if root is None else root
    acquisition = fetch_current_season(season=season, root=root, retrieved_at=retrieved_at)
    corpus = update_processed_game_corpus(root=root, season=season, schedules=acquisition.schedules)
    requested = acquisition.retrieved_at
    slot = _validate_slot(season, publication_slot or requested.date().isoformat())
    label = display_label.strip() if display_label else _label(requested)
    if not label:
        raise ValueError("display_label must not be blank when supplied")
    config_path = root / "site/publish_config.json"
    config = _load_json(config_path)
    with tempfile.TemporaryDirectory(prefix="gippyrank-weekly-", dir=root) as temp:
        temporary = Path(temp)
        common = {
            "season": season, "cutoff": requested, "snapshot_type": "weekly", "root": root,
            "output_root": temporary, "generation_timestamp": requested,
        }
        context = build_snapshot(prior_family="context", **common)
        history = build_snapshot(prior_family="history", **common)
        _same_evidence(context, history)
        final_context_path = (
            root
            / "data/processed/snapshots"
            / str(season)
            / context.snapshot_id
            / "predictive/context"
        )
        performance = build_performance_snapshot(
            context,
            root=root,
            output_root=temporary,
            source_context_path=final_context_path,
            generation_timestamp=requested,
        )
        validate_performance_against_context(context, performance)
        report = _report(context, history, performance, slot=slot, label=label, corpus=corpus)
        previous_context = _previous_rows(root, config, "context")
        previous_history = _previous_rows(root, config, "history")
        previous_performance = _previous_rows(root, config, "performance")
        previous_games = _configured_snapshot(root, config, "context")
        previous_ids: set[str] = set()
        if previous_games is not None and previous_games.exists():
            with (previous_games / "included_games.csv").open(newline="", encoding="utf-8") as handle:
                previous_ids = {row["id"] for row in csv.DictReader(handle)}
        report["new_eligible_game_count"] = len(set(context.metadata["included_game_ids"]) - previous_ids)
        report["context_movers"] = _movement(context, previous_context)
        report["history_movers"] = _movement(history, previous_history)
        report["h_c_disagreements"] = _disagreements(context, history)
        report["newly_rated_teams"] = _newly_rated(performance, previous_performance)
        if not _is_publishable_change(root, config, context, history, performance):
            effective = datetime.fromisoformat(str(context.metadata["effective_cutoff"]))
            return WeeklyUpdate(
                season, slot, label, requested, effective, context, history, performance,
                corpus, False, report
            )
        final_snapshots: list[Snapshot] = []
        for snapshot in (context, history):
            final = root / "data/processed/snapshots" / str(season) / snapshot.snapshot_id / "predictive" / snapshot.metadata["prior_family"]
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                shutil.rmtree(final)
            shutil.copytree(snapshot.directory, final)
            final_snapshots.append(Snapshot(snapshot.snapshot_id, final, snapshot.metadata))
        performance_final = (
            root / "data/processed/snapshots" / performance.directory.relative_to(temporary)
        )
        performance_final.parent.mkdir(parents=True, exist_ok=True)
        if performance_final.exists():
            shutil.rmtree(performance_final)
        shutil.copytree(performance.directory, performance_final)
    context, history = final_snapshots
    performance = Snapshot(performance.snapshot_id, performance_final, performance.metadata)
    validate_performance_against_context(context, performance)
    _upsert_publication(
        root=root,
        config_path=config_path,
        context=context,
        history=history,
        performance=performance,
        slot=slot,
        label=label,
    )
    build_site_data(root=root, config_path=config_path, output_directory=root / "site/data")
    first_export = _tree_hash(root / "site/data")
    build_site_data(root=root, config_path=config_path, output_directory=root / "site/data")
    if _tree_hash(root / "site/data") != first_export:
        raise ValueError("Static site export is not deterministic")
    report = _report(context, history, performance, slot=slot, label=label, corpus=corpus)
    report["new_eligible_game_count"] = len(set(context.metadata["included_game_ids"]) - previous_ids)
    report["context_movers"] = _movement(context, previous_context)
    report["history_movers"] = _movement(history, previous_history)
    report["h_c_disagreements"] = _disagreements(context, history)
    report["newly_rated_teams"] = _newly_rated(performance, previous_performance)
    report_dir = root / "data/processed/weekly_updates"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{slot}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (report_dir / f"{slot}.md").write_text(render_review_markdown(report), encoding="utf-8")
    effective = datetime.fromisoformat(str(context.metadata["effective_cutoff"]))
    candidate_paths = PublicationCandidatePaths(
        fbs_schedule=root / "data/raw/cfbd/games" / f"{season}.json",
        fbs_provenance=root / "data/raw/cfbd/games" / f"{season}.json.provenance.json",
        fcs_schedule=root / "data/raw/cfbd/games" / f"{season}-fcs.json",
        fcs_provenance=root / "data/raw/cfbd/games" / f"{season}-fcs.json.provenance.json",
        processed_games=root / "data/processed/cfbd/games.csv",
        context_snapshot=context.directory,
        history_snapshot=history.directory,
        performance_snapshot=performance.directory,
        report_md=report_dir / f"{slot}.md",
        report_json=report_dir / f"{slot}.json",
        publish_config=config_path,
        site_data=root / "site/data",
    )
    return WeeklyUpdate(
        season, slot, label, requested, effective, context, history, performance,
        corpus, True, report,
        candidate_paths,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a reviewed weekly GippyRank update")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--display-label")
    parser.add_argument("--publication-slot")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--github-summary", type=Path)
    args = parser.parse_args()
    root = _root()
    update = prepare_weekly_update(
        season=args.season,
        root=root,
        display_label=args.display_label,
        publication_slot=args.publication_slot,
    )
    result = {"published": update.published, "publication_slot": update.publication_slot, "report": update.report}
    if args.github_output:
        args.github_output.write_text("\n".join(_github_output_lines(update, root=root)) + "\n", encoding="utf-8")
    if args.github_summary:
        args.github_summary.write_text(render_review_markdown(update.report), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
