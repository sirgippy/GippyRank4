from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from gippyrank.performance_snapshot import build_performance_snapshot
from gippyrank.posterior.engine import LikelihoodV1, Team, infer_posterior
from gippyrank.posterior.game_evidence import build_team_season_artifact
from gippyrank.posterior.snapshots import build_snapshot
from gippyrank.site_data import SiteDataValidationError, build_site_data


def _write(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _root(tmp_path: Path) -> Path:
    prior_fields = ["season", "subdivision", "team_id", "team_name", "conference", "pmf"]
    prior_rows = [
        {"season": 2026, "subdivision": "fbs", "team_id": "1", "team_name": "One", "conference": "A", "pmf": "[0.7,0.3]"},
        {"season": 2026, "subdivision": "fbs", "team_id": "2", "team_name": "Two", "conference": "A", "pmf": "[0.3,0.7]"},
    ]
    for family in ("context", "history"):
        _write(tmp_path / f"data/processed/preseason/{family}/annual/2026/predictions.csv", prior_fields, prior_rows)
    fields = [
        "id", "season", "week", "seasonType", "startDate", "completed", "neutralSite",
        "conferenceGame", "homeId", "homeTeam", "homeClassification", "homeConference",
        "homePoints", "awayId", "awayTeam", "awayClassification", "awayConference", "awayPoints",
    ]
    _write(
        tmp_path / "data/processed/cfbd/games.csv",
        fields,
        [
            {"id": "early", "season": 2026, "week": 1, "seasonType": "regular", "startDate": "2026-08-29T00:00:00Z", "completed": "True", "neutralSite": "False", "conferenceGame": "True", "homeId": "1", "homeTeam": "One", "homeClassification": "fbs", "homeConference": "A", "homePoints": 20, "awayId": "2", "awayTeam": "Two", "awayClassification": "fbs", "awayConference": "A", "awayPoints": 10},
            {"id": "later", "season": 2026, "week": 2, "seasonType": "regular", "startDate": "2026-09-10T00:00:00Z", "completed": "True", "neutralSite": "False", "conferenceGame": "True", "homeId": "1", "homeTeam": "One", "homeClassification": "fbs", "homeConference": "A", "homePoints": 7, "awayId": "2", "awayTeam": "Two", "awayClassification": "fbs", "awayConference": "A", "awayPoints": 24},
        ],
    )
    return tmp_path


def test_team_artifact_hides_future_results_and_site_exports_lazy_path(tmp_path: Path) -> None:
    root = _root(tmp_path)
    snapshot = build_snapshot(
        season=2026,
        cutoff=date(2026, 9, 1),
        prior_family="context",
        snapshot_type="weekly",
        root=root,
        likelihood=LikelihoodV1(np.zeros(34), 1.0, 15.0),
    )
    artifact = json.loads((snapshot.directory / "team_seasons.json").read_text())
    games = artifact["teams"]["1"]["games"]
    assert [game["game_id"] for game in games] == ["early", "later"]
    early, later = games
    assert early["modeled"] and early["game_rating"] is not None
    assert len(early["game_rating"]["display_pmf"]) == 40
    assert sum(early["game_rating"]["display_pmf"]) == 1000
    assert 0 <= early["game_rating"]["performance_percentile"] <= 100
    assert early["game_rating"]["performance_grade"] in {"A", "B", "C", "D", "F"}
    assert artifact["performance_axis"] == {
        "bins": 40,
        "direction": "best_to_worst",
        "label": "inferred performance quality",
        "max_rank": 2,
        "min_rank": 1,
        "probability_encoding": {
            "normalization": "divide weights by scale",
            "scale": 1000,
            "total_weight": 1000,
            "type": "fixed_scale_integer",
        },
    }
    assert artifact["performance_percentile"]["reference_count"] == 2
    assert later["result"] is None and later["score"] is None and later["game_rating"] is None
    prediction = artifact["future_predictions"][later["future_prediction_id"]]
    assert len(prediction["display_distribution"]["masses"]) == 40
    assert sum(prediction["display_distribution"]["masses"]) + prediction["display_distribution"]["lower_tail_probability"] + prediction["display_distribution"]["upper_tail_probability"] == 1000
    assert early["future_prediction_id"] is None
    assert later["future_prediction_id"] == "later"
    prediction = artifact["future_predictions"]["later"]
    assert prediction["prediction_source"] == "predictive_context"
    assert prediction["home_team_id"] == "1"
    assert prediction["away_team_id"] == "2"
    assert prediction["margin_interval_50"][0] <= prediction["margin_interval_50"][1]
    assert prediction["home_win_probability"] + prediction["away_win_probability"] == pytest.approx(1.0)

    config = root / "site/publish_config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"schema_version": "1.0", "publication_slots": [{"id": "2026-09-01", "status": "official"}], "snapshots": [{"source": snapshot.directory.relative_to(root).as_posix(), "display_label": "Test", "publication_slot": "2026-09-01"}]}))
    manifest = build_site_data(root=root, config_path=config, output_directory=root / "site/data")
    entry = manifest["snapshots"][0]
    assert entry["team_seasons_path"] == "data/team-seasons/2026-weekly-2026-09-01-context.json"
    assert (root / "site" / entry["team_seasons_path"]).is_file()
    assert entry["completed_visualization_bytes"] > 0
    assert entry["completed_visualization_bytes_per_game"] > 0
    assert entry["future_visualization_bytes"] > 0
    assert entry["future_visualization_bytes_per_game"] > 0
    assert manifest["payload_stats"]["initial_rankings_page_bytes"] == sum(
        (root / "site" / item["data_path"]).stat().st_size
        for item in manifest["snapshots"]
    )


def test_future_predictions_use_posterior_after_completed_evidence(tmp_path: Path) -> None:
    root = _root(tmp_path)
    likelihood = LikelihoodV1(np.r_[10.0, np.zeros(33)], 1.0, 15.0)
    before_evidence = build_snapshot(
        season=2026,
        cutoff=date(2026, 8, 28),
        prior_family="context",
        snapshot_type="weekly",
        root=root,
        likelihood=likelihood,
    )
    after_evidence = build_snapshot(
        season=2026,
        cutoff=date(2026, 9, 1),
        prior_family="context",
        snapshot_type="weekly",
        root=root,
        likelihood=likelihood,
    )

    before = json.loads(
        (before_evidence.directory / "team_seasons.json").read_text(encoding="utf-8")
    )
    after = json.loads(
        (after_evidence.directory / "team_seasons.json").read_text(encoding="utf-8")
    )
    before_margin = before["future_predictions"]["later"]["expected_home_margin"]
    after_margin = after["future_predictions"]["later"]["expected_home_margin"]

    assert before["included_game_ids"] == []
    assert after["included_game_ids"] == ["early"]
    assert after_margin != pytest.approx(before_margin)


def test_team_artifact_provenance_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    snapshot = build_snapshot(
        season=2026,
        cutoff=date(2026, 9, 1),
        prior_family="context",
        snapshot_type="weekly",
        root=root,
        likelihood=LikelihoodV1(np.zeros(34), 1.0, 15.0),
    )
    artifact_path = snapshot.directory / "team_seasons.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["game_corpus_sha256"] = "wrong"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    config = root / "site/publish_config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({"schema_version": "1.0", "publication_slots": [{"id": "2026-09-01", "status": "official"}], "snapshots": [{"source": snapshot.directory.relative_to(root).as_posix(), "display_label": "Test", "publication_slot": "2026-09-01"}]}))
    with pytest.raises(SiteDataValidationError, match="provenance mismatch"):
        build_site_data(root=root, config_path=config, output_directory=root / "site/data")


def test_declared_team_artifact_missing_fails_closed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    snapshot = build_snapshot(
        season=2026,
        cutoff=date(2026, 9, 1),
        prior_family="context",
        snapshot_type="weekly",
        root=root,
        likelihood=LikelihoodV1(np.zeros(34), 1.0, 15.0),
    )
    (snapshot.directory / "team_seasons.json").unlink()
    config = root / "site/publish_config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({"schema_version": "1.0", "publication_slots": [{"id": "2026-09-01", "status": "official"}], "snapshots": [{"source": snapshot.directory.relative_to(root).as_posix(), "display_label": "Test", "publication_slot": "2026-09-01"}]}))
    with pytest.raises(SiteDataValidationError, match="declared team-season artifact is unavailable"):
        build_site_data(root=root, config_path=config, output_directory=root / "site/data")


def test_started_but_unincluded_game_stays_redacted(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _write(
        root / "data/processed/cfbd/games.csv",
        [
            "id", "season", "week", "seasonType", "startDate", "completed", "neutralSite",
            "conferenceGame", "homeId", "homeTeam", "homeClassification", "homeConference",
            "homePoints", "awayId", "awayTeam", "awayClassification", "awayConference", "awayPoints",
        ],
        [{
            "id": "started", "season": 2026, "week": 1, "seasonType": "regular",
            "startDate": "2026-09-01T12:00:00Z", "completed": "True", "neutralSite": "False",
            "conferenceGame": "True", "homeId": "1", "homeTeam": "One", "homeClassification": "fbs",
            "homeConference": "A", "homePoints": 15, "awayId": "2", "awayTeam": "Two",
            "awayClassification": "fbs", "awayConference": "A", "awayPoints": 34,
        }],
    )
    teams = [
        Team("1", "One", "fbs", np.array([0.7, 0.3])),
        Team("2", "Two", "fbs", np.array([0.3, 0.7])),
    ]
    posterior = infer_posterior(teams, [], LikelihoodV1(np.zeros(34), 1.0, 15.0))
    artifact = build_team_season_artifact(
        root=root,
        metadata={
            "snapshot_id": "2026-weekly-test-context",
            "season": 2026,
            "snapshot_type": "weekly",
            "requested_cutoff": "2026-09-01T23:59:59+00:00",
            "effective_cutoff": "2026-09-01T23:59:59+00:00",
            "source_retrieved_at": None,
            "source_retrieval_times": {},
            "source_response_hashes": {},
            "game_corpus_sha256": "historical-evidence-corpus",
            "included_game_ids": [],
        },
        teams=teams,
        team_rows={"1": {"conference": "A"}, "2": {"conference": "A"}},
        games=[],
        included_rows=[],
        posterior=posterior,
        likelihood=None,
    )
    for team in artifact["teams"].values():
        game = team["games"][0]
        assert game["opponent_name"] in {"One", "Two"}
        assert game["result"] is None
        assert game["score"] is None
        assert game["game_rating"] is None
        assert not game["modeled"]


def test_validator_rejects_unincluded_completed_result(tmp_path: Path) -> None:
    root = _root(tmp_path)
    schedule_path = root / "data/processed/cfbd/games.csv"
    with schedule_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["id"] == "later":
            row.update({"startDate": "2026-09-01T12:00:00Z", "completed": "False", "homePoints": "", "awayPoints": ""})
    _write(schedule_path, list(rows[0]), rows)
    snapshot = build_snapshot(
        season=2026,
        cutoff=date(2026, 9, 1),
        prior_family="context",
        snapshot_type="weekly",
        root=root,
        likelihood=LikelihoodV1(np.zeros(34), 1.0, 15.0),
    )

    for row in rows:
        if row["id"] == "later":
            row.update({"completed": "True", "homePoints": "7", "awayPoints": "24"})
    _write(schedule_path, list(rows[0]), rows)
    artifact_path = snapshot.directory / "team_seasons.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["schedule_source"]["sha256"] = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    game = next(game for game in artifact["teams"]["1"]["games"] if game["game_id"] == "later")
    game.update({"result": "L", "score": {"team": 7, "opponent": 24}})
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    config = root / "site/publish_config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({"schema_version": "1.0", "publication_slots": [{"id": "2026-09-01", "status": "official"}], "snapshots": [{"source": snapshot.directory.relative_to(root).as_posix(), "display_label": "Test", "publication_slot": "2026-09-01"}]}))
    with pytest.raises(SiteDataValidationError, match="without snapshot evidence"):
        build_site_data(root=root, config_path=config, output_directory=root / "site/data")


def test_historical_artifact_survives_later_schedule_refresh(tmp_path: Path) -> None:
    root = _root(tmp_path)
    snapshot = build_snapshot(
        season=2026,
        cutoff=date(2026, 9, 1),
        prior_family="context",
        snapshot_type="weekly",
        root=root,
        likelihood=LikelihoodV1(np.zeros(34), 1.0, 15.0),
    )
    artifact_path = snapshot.directory / "team_seasons.json"
    original_artifact = artifact_path.read_bytes()
    original_schedule_source = json.loads(original_artifact)["schedule_source"]
    schedule_path = root / "data/processed/cfbd/games.csv"
    with schedule_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["id"] == "later":
            row["homePoints"] = "8"
    _write(schedule_path, list(rows[0]), rows)
    config = root / "site/publish_config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({"schema_version": "1.0", "publication_slots": [{"id": "2026-09-01", "status": "official"}], "snapshots": [{"source": snapshot.directory.relative_to(root).as_posix(), "display_label": "Test", "publication_slot": "2026-09-01"}]}))

    manifest = build_site_data(root=root, config_path=config, output_directory=root / "site/data")

    assert manifest["snapshots"][0]["team_seasons_bytes"] > 0
    assert artifact_path.read_bytes() == original_artifact
    exported = json.loads((root / "site" / manifest["snapshots"][0]["team_seasons_path"]).read_text())
    assert exported["schedule_source"] == original_schedule_source


def test_context_history_and_performance_use_declared_prediction_sources(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    history_prior = root / "data/processed/preseason/history/annual/2026/predictions.csv"
    with history_prior.open(newline="", encoding="utf-8") as handle:
        history_rows = list(csv.DictReader(handle))
        history_fields = list(history_rows[0])
    history_rows[0]["pmf"] = "[0.1,0.9]"
    history_rows[1]["pmf"] = "[0.9,0.1]"
    _write(history_prior, history_fields, history_rows)

    beta = np.zeros(34)
    beta[0] = 10.0
    likelihood = LikelihoodV1(beta, 1.0, 15.0)
    context = build_snapshot(
        season=2026,
        cutoff=date(2026, 9, 1),
        prior_family="context",
        snapshot_type="weekly",
        root=root,
        likelihood=likelihood,
    )
    history = build_snapshot(
        season=2026,
        cutoff=date(2026, 9, 1),
        prior_family="history",
        snapshot_type="weekly",
        root=root,
        likelihood=likelihood,
    )
    performance = build_performance_snapshot(
        context,
        root=root,
        output_root=root / "data/processed/snapshots",
    )
    config = root / "site/publish_config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "publication_slots": [{"id": "2026-09-01", "status": "official"}],
                "snapshots": [
                    {
                        "source": snapshot.directory.relative_to(root).as_posix(),
                        "display_label": "Test",
                        "publication_slot": "2026-09-01",
                    }
                    for snapshot in (context, history, performance)
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_site_data(root=root, config_path=config, output_directory=root / "site/data")
    entries = {entry["ranking_family"] + ":" + entry.get("prior_family", ""): entry for entry in manifest["snapshots"]}
    artifacts = {
        key: json.loads(
            (root / "site" / entry["team_seasons_path"]).read_text(encoding="utf-8")
        )
        for key, entry in entries.items()
    }

    assert artifacts["predictive:context"]["prediction_source"] == "predictive_context"
    assert artifacts["predictive:history"]["prediction_source"] == "predictive_history"
    assert artifacts["performance:"]["prediction_source"] == "predictive_context"
    assert artifacts["predictive:context"]["future_predictions"]["later"]["expected_home_margin"] != pytest.approx(
        artifacts["predictive:history"]["future_predictions"]["later"]["expected_home_margin"]
    )
    for artifact in artifacts.values():
        prediction = artifact["future_predictions"]["later"]
        assert prediction["home_win_probability"] + prediction["away_win_probability"] == pytest.approx(1.0)
        assert artifact["teams"]["1"]["games"][1]["future_prediction_id"] == "later"
        assert artifact["teams"]["2"]["games"][1]["future_prediction_id"] == "later"
