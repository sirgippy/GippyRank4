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
    corpus: dict[str, int]
    published: bool
    report: dict[str, Any]


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
    for entry in config.get("snapshots", []):
        if entry.get("publication_slot") == slot and entry.get("source", "").endswith(f"/{family}"):
            return root / entry["source"]
    return None


def _is_publishable_change(root: Path, config: dict[str, Any], context: Snapshot, history: Snapshot) -> bool:
    for family, snapshot in (("context", context), ("history", history)):
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
    slot: str, label: str,
) -> None:
    config = _load_json(config_path)
    retained = [
        entry for entry in config["snapshots"]
        if entry.get("publication_slot") != slot
    ]
    for snapshot in (context, history):
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
        for row in sorted(_rows(snapshot), key=lambda row: int(row["display_rank"]))[:25]
    )


def _diagnostics(snapshot: Snapshot) -> dict[str, Any]:
    return _load_json(snapshot.directory / "diagnostics.json")


def _report(context: Snapshot, history: Snapshot, *, slot: str, label: str, corpus: dict[str, int]) -> dict[str, Any]:
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
        report = _report(context, history, slot=slot, label=label, corpus=corpus)
        previous_context = _previous_rows(root, config, "context")
        previous_history = _previous_rows(root, config, "history")
        previous_games = _configured_snapshot(root, config, "context")
        previous_ids: set[str] = set()
        if previous_games is not None and previous_games.exists():
            with (previous_games / "included_games.csv").open(newline="", encoding="utf-8") as handle:
                previous_ids = {row["id"] for row in csv.DictReader(handle)}
        report["new_eligible_game_count"] = len(set(context.metadata["included_game_ids"]) - previous_ids)
        report["context_movers"] = _movement(context, previous_context)
        report["history_movers"] = _movement(history, previous_history)
        report["h_c_disagreements"] = _disagreements(context, history)
        if not _is_publishable_change(root, config, context, history):
            effective = datetime.fromisoformat(str(context.metadata["effective_cutoff"]))
            return WeeklyUpdate(season, slot, label, requested, effective, context, history, corpus, False, report)
        final_snapshots: list[Snapshot] = []
        for snapshot in (context, history):
            final = root / "data/processed/snapshots" / str(season) / snapshot.snapshot_id / "predictive" / snapshot.metadata["prior_family"]
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                shutil.rmtree(final)
            shutil.copytree(snapshot.directory, final)
            final_snapshots.append(Snapshot(snapshot.snapshot_id, final, snapshot.metadata))
    context, history = final_snapshots
    _upsert_publication(root=root, config_path=config_path, context=context, history=history, slot=slot, label=label)
    build_site_data(root=root, config_path=config_path, output_directory=root / "site/data")
    first_export = _tree_hash(root / "site/data")
    build_site_data(root=root, config_path=config_path, output_directory=root / "site/data")
    if _tree_hash(root / "site/data") != first_export:
        raise ValueError("Static site export is not deterministic")
    report = _report(context, history, slot=slot, label=label, corpus=corpus)
    report["new_eligible_game_count"] = len(set(context.metadata["included_game_ids"]) - previous_ids)
    report["context_movers"] = _movement(context, previous_context)
    report["history_movers"] = _movement(history, previous_history)
    report["h_c_disagreements"] = _disagreements(context, history)
    report_dir = root / "data/processed/weekly_updates"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{slot}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (report_dir / f"{slot}.md").write_text(render_review_markdown(report), encoding="utf-8")
    effective = datetime.fromisoformat(str(context.metadata["effective_cutoff"]))
    return WeeklyUpdate(season, slot, label, requested, effective, context, history, corpus, True, report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a reviewed weekly GippyRank update")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--display-label")
    parser.add_argument("--publication-slot")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--github-summary", type=Path)
    args = parser.parse_args()
    update = prepare_weekly_update(season=args.season, display_label=args.display_label, publication_slot=args.publication_slot)
    result = {"published": update.published, "publication_slot": update.publication_slot, "report": update.report}
    if args.github_output:
        args.github_output.write_text(
            f"published={'true' if update.published else 'false'}\n"
            f"slot={update.publication_slot}\n"
            f"branch=automation/rankings-{update.publication_slot}\n"
            f"report_path=data/processed/weekly_updates/{update.publication_slot}.md\n",
            encoding="utf-8",
        )
    if args.github_summary:
        args.github_summary.write_text(render_review_markdown(update.report), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
