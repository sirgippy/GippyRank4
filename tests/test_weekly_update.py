from __future__ import annotations

import csv
import json
import os
import runpy
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gippyrank import weekly_update
from gippyrank.data.cfbd import GAME_FIELDS, CurrentSeasonAcquisition
from gippyrank.posterior.snapshots import Snapshot
from gippyrank.weekly_update import (
    _github_output_lines,
    _upsert_publication,
    prepare_weekly_update,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, fields: tuple[str, ...] | list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _game(game_id: str, *, completed: bool = True, away_class: str = "fcs") -> dict:
    return {
        "id": game_id, "season": 2026, "week": 1, "seasonType": "regular",
        "startDate": "2026-08-29T00:00:00Z", "completed": completed, "neutralSite": False,
        "conferenceGame": False, "homeId": "1", "homeTeam": "One", "homeClassification": "fbs",
        "homeConference": "A", "homePoints": 21 if completed else None, "awayId": "3",
        "awayTeam": "Three", "awayClassification": away_class, "awayConference": "B",
        "awayPoints": 7 if completed else None,
    }


def _root(tmp_path: Path) -> Path:
    prior_fields = ["season", "subdivision", "team_id", "team_name", "conference", "pmf"]
    prior_rows = [
        {"season": 2026, "subdivision": "fbs", "team_id": "1", "team_name": "One", "conference": "A", "pmf": "[0.7,0.3]"},
        {"season": 2026, "subdivision": "fbs", "team_id": "2", "team_name": "Two", "conference": "A", "pmf": "[0.3,0.7]"},
    ]
    for family in ("context", "history"):
        _write_csv(tmp_path / f"data/processed/preseason/{family}/annual/2026/predictions.csv", prior_fields, prior_rows)
    (tmp_path / "data/processed/posterior").mkdir(parents=True)
    (tmp_path / "data/processed/posterior/historical_likelihood_v1.json").write_text(
        json.dumps({"beta": [0.0] * 34, "scale": 1.0, "degrees_of_freedom": 15.0})
    )
    _write_csv(tmp_path / "data/processed/cfbd/games.csv", GAME_FIELDS, [])
    (tmp_path / "site").mkdir()
    (tmp_path / "site/publish_config.json").write_text(json.dumps({"schema_version": "1.0", "publication_slots": [], "snapshots": []}))
    return tmp_path


def _acquisition(root: Path, timestamp: datetime, fcs_timestamp: datetime | None = None) -> CurrentSeasonAcquisition:
    team_two_future = _game("102", completed=False)
    team_two_future.update({"homeId": "2", "homeTeam": "Two"})
    schedules = {
        "fbs": [_game("100"), _game("101", completed=False), team_two_future],
        "fcs": [_game("100")],
    }
    raw = root / "data/raw/cfbd/games"
    raw.mkdir(parents=True, exist_ok=True)
    fcs_timestamp = timestamp if fcs_timestamp is None else fcs_timestamp
    for name, payload, retrieval_time in (("2026.json", schedules["fbs"], timestamp), ("2026-fcs.json", schedules["fcs"], fcs_timestamp)):
        (raw / name).write_text(json.dumps(payload))
        (raw / f"{name}.provenance.json").write_text(json.dumps({"content_sha256": name, "retrieved_at": retrieval_time.isoformat(), "endpoint": "/games", "parameters": {}, "source_kind": "cfbd_api_schedule"}))
    return CurrentSeasonAcquisition(2026, min(timestamp, fcs_timestamp), {"fbs": timestamp, "fcs": fcs_timestamp}, schedules, {"2026.json": "2026.json", "2026-fcs.json": "2026-fcs.json"})


def test_weekly_update_pairs_h_c_preserves_preseason_and_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    timestamp = datetime(2026, 9, 12, 15, tzinfo=UTC)
    monkeypatch.setattr("gippyrank.weekly_update.fetch_current_season", lambda **_: _acquisition(root, timestamp))
    first = prepare_weekly_update(season=2026, root=root)
    assert first.published and first.publication_slot == "2026-09-12"
    assert first.context.metadata["snapshot_type"] == first.history.metadata["snapshot_type"] == "weekly"
    for field in ("included_game_ids", "effective_cutoff", "requested_cutoff", "source_retrieval_times", "source_response_hashes", "game_corpus_sha256"):
        assert first.context.metadata[field] == first.history.metadata[field]
    assert first.context.metadata["included_game_ids"] == ["100"]  # future game cannot enter inference
    config = json.loads((root / "site/publish_config.json").read_text())
    assert config["default_publication_slot"] == "2026-09-12"
    assert config["publication_slots"][-1] == {
        "id": "2026-09-12",
        "status": "temporary",
    }
    entries = config["snapshots"]
    assert len(entries) == 3 and {entry["publication_slot"] for entry in entries} == {"2026-09-12"}
    assert {entry["source"].rsplit("/", 1)[-1] for entry in entries} == {"context", "history", "performance"}
    manifest = json.loads((root / "site/data/manifest.json").read_text())
    assert manifest["default_publication_slot"] == "2026-09-12"
    assert first.candidate_paths is not None
    assert first.candidate_paths.context_snapshot == first.context.directory
    assert first.candidate_paths.history_snapshot == first.history.directory
    assert first.candidate_paths.performance_snapshot == first.performance.directory
    assert first.candidate_paths.report_md.is_file()
    assert first.candidate_paths.report_json.is_file()
    outputs = dict(line.split("=", 1) for line in _github_output_lines(first, root=root))
    assert outputs["context_snapshot_path"] == first.context.directory.relative_to(root).as_posix()
    assert outputs["history_snapshot_path"] == first.history.directory.relative_to(root).as_posix()
    assert outputs["performance_snapshot_path"] == first.performance.directory.relative_to(root).as_posix()
    assert outputs["report_md_path"] == "data/processed/weekly_updates/2026-09-12.md"
    assert outputs["report_json_path"] == "data/processed/weekly_updates/2026-09-12.json"
    assert outputs["fbs_schedule_path"] == "data/raw/cfbd/games/2026.json"
    assert outputs["fcs_provenance_path"] == "data/raw/cfbd/games/2026-fcs.json.provenance.json"
    second = prepare_weekly_update(season=2026, root=root)
    assert not second.published
    assert set(dict(line.split("=", 1) for line in _github_output_lines(second, root=root))) == {
        "published", "slot", "branch", "report_path"
    }


def test_weekly_h_c_effective_cutoff_uses_earliest_required_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    fbs_time = datetime(2026, 9, 12, 15, tzinfo=UTC)
    fcs_time = datetime(2026, 9, 12, 15, 8, tzinfo=UTC)
    monkeypatch.setattr("gippyrank.weekly_update.fetch_current_season", lambda **_: _acquisition(root, fbs_time, fcs_time))
    update = prepare_weekly_update(season=2026, root=root)
    assert update.requested_cutoff == update.effective_cutoff == fbs_time
    assert update.context.metadata["source_retrieval_times"] == {
        "fbs": fbs_time.isoformat(), "fcs": fcs_time.isoformat()
    }


def test_snapshot_failure_never_reaches_publication_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    timestamp = datetime(2026, 9, 12, 15, tzinfo=UTC)
    monkeypatch.setattr("gippyrank.weekly_update.fetch_current_season", lambda **_: _acquisition(root, timestamp))
    monkeypatch.setattr("gippyrank.weekly_update.build_snapshot", lambda **_: (_ for _ in ()).throw(RuntimeError("inference failed")))
    with pytest.raises(RuntimeError, match="inference failed"):
        prepare_weekly_update(season=2026, root=root)
    assert json.loads((root / "site/publish_config.json").read_text())["snapshots"] == []


def test_upserting_a_slot_preserves_preseason_and_older_slots(tmp_path: Path) -> None:
    config = tmp_path / "publish.json"
    config.write_text(json.dumps({"schema_version": "1.0", "default_publication_slot": "2026-old", "publication_slots": [
        {"id": "2026-preseason", "status": "official"},
        {"id": "2026-old", "status": "temporary"}
    ], "snapshots": [
        {"source": "old/preseason/context", "display_label": "Preseason", "publication_slot": "2026-preseason"},
        {"source": "old/preseason/history", "display_label": "Preseason", "publication_slot": "2026-preseason"},
        {"source": "old/weekly/context", "display_label": "Sep. 5", "publication_slot": "2026-old"},
        {"source": "old/weekly/history", "display_label": "Sep. 5", "publication_slot": "2026-old"},
    ]}))
    context_dir, history_dir = tmp_path / "new/context", tmp_path / "new/history"
    context_dir.mkdir(parents=True)
    history_dir.mkdir(parents=True)
    _upsert_publication(root=tmp_path, config_path=config, slot="2026-09-12", label="Sep. 12",
        context=Snapshot("context", context_dir, {}), history=Snapshot("history", history_dir, {}))
    value = json.loads(config.read_text())
    assert value["default_publication_slot"] == "2026-09-12"
    assert {entry["publication_slot"] for entry in value["snapshots"]} == {"2026-preseason", "2026-old", "2026-09-12"}


def test_update_workflow_is_manual_and_pages_stays_model_and_cfbd_free() -> None:
    workflow = Path(".github/workflows/update-rankings.yml").read_text(encoding="utf-8")
    pages = Path(".github/workflows/deploy-pages.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow and "schedule:" not in workflow
    assert "CFBD_API_KEY" in workflow and "/games/teams" not in workflow
    assert "pull-requests: write" in workflow and "base: main" in workflow
    assert "merge" not in workflow.casefold()
    assert "add-paths:" not in workflow
    assert "weekly_updates/${{ steps.candidate.outputs.slot }}.*" not in workflow
    run_block = workflow.split("        run: |", 1)[1].split("      - name: Report", 1)[0]
    assert "${{ inputs." not in run_block
    assert 'args=(--season "$INPUT_SEASON"' in run_block
    assert "CFBD_API_KEY" not in pages and "build_snapshot" not in pages and "cfbd" not in pages.casefold()


def test_update_workflow_commits_only_a_published_non_dry_run_candidate() -> None:
    workflow = Path(".github/workflows/update-rankings.yml").read_text(encoding="utf-8")
    expected_if = "if: inputs.dry_run != true && steps.candidate.outputs.published == 'true'"
    assert workflow.count(expected_if) == 2
    assert "- name: Commit approved publication candidate" in workflow
    assert "bash scripts/stage_weekly_publication_candidate.sh" in workflow
    assert 'git commit -m "$CANDIDATE_COMMIT_MESSAGE"' in workflow
    assert "git push" not in workflow
    for output_name, environment_name in (
        ("fbs_schedule_path", "FBS_SCHEDULE_PATH"),
        ("fbs_provenance_path", "FBS_PROVENANCE_PATH"),
        ("fcs_schedule_path", "FCS_SCHEDULE_PATH"),
        ("fcs_provenance_path", "FCS_PROVENANCE_PATH"),
        ("processed_games_path", "PROCESSED_GAMES_PATH"),
        ("context_snapshot_path", "CONTEXT_SNAPSHOT_PATH"),
        ("history_snapshot_path", "HISTORY_SNAPSHOT_PATH"),
        ("performance_snapshot_path", "PERFORMANCE_SNAPSHOT_PATH"),
        ("report_md_path", "REPORT_MD_PATH"),
        ("report_json_path", "REPORT_JSON_PATH"),
        ("publish_config_path", "PUBLISH_CONFIG_PATH"),
        ("site_data_path", "SITE_DATA_PATH"),
    ):
        assert f"{environment_name}: ${{{{ steps.candidate.outputs.{output_name} }}}}" in workflow


def _write_candidate_file(root: Path, relative_path: str, content: str = "candidate\n") -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _candidate_environment() -> dict[str, str]:
    return {
        "FBS_SCHEDULE_PATH": "data/raw/cfbd/games/2026.json",
        "FBS_PROVENANCE_PATH": "data/raw/cfbd/games/2026.json.provenance.json",
        "FCS_SCHEDULE_PATH": "data/raw/cfbd/games/2026-fcs.json",
        "FCS_PROVENANCE_PATH": "data/raw/cfbd/games/2026-fcs.json.provenance.json",
        "PROCESSED_GAMES_PATH": "data/processed/cfbd/games.csv",
        "CONTEXT_SNAPSHOT_PATH": "data/processed/snapshots/2026/actual-context/predictive/context",
        "HISTORY_SNAPSHOT_PATH": "data/processed/snapshots/2026/actual-history/predictive/history",
        "PERFORMANCE_SNAPSHOT_PATH": "data/processed/snapshots/2026/actual-performance/performance",
        "REPORT_MD_PATH": "data/processed/weekly_updates/2026-09-12.md",
        "REPORT_JSON_PATH": "data/processed/weekly_updates/2026-09-12.json",
        "PUBLISH_CONFIG_PATH": "site/publish_config.json",
        "SITE_DATA_PATH": "site/data",
    }


def _candidate_repo(tmp_path: Path) -> dict[str, str]:
    (tmp_path / ".gitignore").write_text("data/raw/\ndata/processed/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    environment = _candidate_environment()
    for relative_path in environment.values():
        if relative_path.endswith(("context", "history", "performance", "data")):
            _write_candidate_file(tmp_path, f"{relative_path}/artifact.json")
        else:
            _write_candidate_file(tmp_path, relative_path)
    _write_candidate_file(tmp_path, "data/processed/research/should-stay-ignored.txt")
    _write_candidate_file(tmp_path, "data/raw/private/cache.json")
    _write_candidate_file(tmp_path, ".venv/should-stay-ignored.txt")
    return environment


def test_candidate_staging_force_adds_only_durable_ignored_publication_artifacts(tmp_path: Path) -> None:
    environment = _candidate_repo(tmp_path)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/stage_weekly_publication_candidate.sh")],
        cwd=tmp_path,
        env={**os.environ, **environment},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    assert set(staged) == {
        "data/raw/cfbd/games/2026.json",
        "data/raw/cfbd/games/2026.json.provenance.json",
        "data/raw/cfbd/games/2026-fcs.json",
        "data/raw/cfbd/games/2026-fcs.json.provenance.json",
        "data/processed/cfbd/games.csv",
        "data/processed/snapshots/2026/actual-context/predictive/context/artifact.json",
        "data/processed/snapshots/2026/actual-history/predictive/history/artifact.json",
        "data/processed/snapshots/2026/actual-performance/performance/artifact.json",
        "data/processed/weekly_updates/2026-09-12.md",
        "data/processed/weekly_updates/2026-09-12.json",
        "site/publish_config.json",
        "site/data/artifact.json",
    }


def test_candidate_staging_rejects_any_pre_staged_non_publication_path(tmp_path: Path) -> None:
    environment = _candidate_repo(tmp_path)
    _write_candidate_file(tmp_path, "unrelated-source-edit.txt")
    subprocess.run(["git", "add", "unrelated-source-edit.txt"], cwd=tmp_path, check=True)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/stage_weekly_publication_candidate.sh")],
        cwd=tmp_path,
        env={**os.environ, **environment},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Refusing to commit non-publication path: unrelated-source-edit.txt" in result.stderr


def test_default_weekly_root_is_repository_with_project_and_publication_markers() -> None:
    assert weekly_update._root() == ROOT
    assert (weekly_update._root() / "pyproject.toml").is_file()
    assert (weekly_update._root() / "site/publish_config.json").is_file()


def test_update_rankings_script_reaches_publish_config_with_default_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real CLI entry point without fetching or mutating data."""
    timestamp = datetime(2026, 9, 12, 15, tzinfo=UTC)

    class StopAfterConfigLoad(Exception):
        pass

    monkeypatch.setattr(
        weekly_update,
        "fetch_current_season",
        lambda **_: CurrentSeasonAcquisition(
            2026,
            timestamp,
            {"fbs": timestamp, "fcs": timestamp},
            {"fbs": [], "fcs": []},
            {},
        ),
    )
    monkeypatch.setattr(weekly_update, "update_processed_game_corpus", lambda **_: {})
    original_load = weekly_update._load_json

    def load_then_stop(path: Path) -> dict:
        original_load(path)
        assert path == ROOT / "site/publish_config.json"
        raise StopAfterConfigLoad

    monkeypatch.setattr(weekly_update, "_load_json", load_then_stop)
    monkeypatch.setattr(sys, "argv", ["scripts/update_rankings.py", "--season", "2026"])
    with pytest.raises(StopAfterConfigLoad):
        runpy.run_path(str(ROOT / "scripts/update_rankings.py"), run_name="__main__")
