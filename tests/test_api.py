from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from gippyrank.api.static import build_static_api
from gippyrank.api.store import (
    PublicationDataError,
    PublicationStore,
    _validate_team_season_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
FULL_MANIFEST = json.loads(
    (ROOT / "site/data/manifest.json").read_text(encoding="utf-8")
)


def _tree_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(directory).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _copy_selected_publication(
    tmp_path: Path, selected_entry: dict[str, Any] | None = None
) -> tuple[Path, dict[str, Any]]:
    entry = (
        dict(selected_entry)
        if selected_entry is not None
        else next(
            item
            for item in FULL_MANIFEST["snapshots"]
            if item["ranking_family"] == "predictive"
            and item.get("prior_family") == "context"
            and item["snapshot_type"] == "weekly"
        )
    )
    data_dir = tmp_path / "data"
    methodology_source = (
        ROOT / "site/data" / FULL_MANIFEST["methodology_path"].removeprefix("data/")
    )
    methodology_destination = data_dir / "methodology.json"
    methodology_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(methodology_source, methodology_destination)
    for field in (
        "data_path",
        "distribution_path",
        "team_seasons_path",
        "week_games_path",
    ):
        source = ROOT / "site/data" / entry[field].removeprefix("data/")
        destination = data_dir / entry[field].removeprefix("data/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    manifest = {
        "schema_version": FULL_MANIFEST["schema_version"],
        "site_url": FULL_MANIFEST["site_url"],
        "methodology_path": "data/methodology.json",
        "methodology_schema_version": FULL_MANIFEST["methodology_schema_version"],
        "seasons": [entry["season"]],
        "ranking_families": FULL_MANIFEST["ranking_families"],
        "snapshots": [entry],
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return data_dir, entry


OCTOBER_24_CUTOFF = "2026-10-24T23:49:36.463412+00:00"
OCTOBER_24_GAME_DATE = "2026-10-24T23:30:00+00:00"


@pytest.fixture(scope="module")
def october_24_publication() -> tuple[dict[str, Any], dict[str, Any], Any]:
    entry = next(
        item
        for item in FULL_MANIFEST["snapshots"]
        if item.get("season") == 2026
        and item.get("ranking_family") == "predictive"
        and item.get("prior_family") == "context"
        and item.get("snapshot_type") == "weekly"
        and str(item.get("effective_cutoff", "")).startswith("2026-10-24")
    )
    artifact_path = ROOT / "site" / entry["team_seasons_path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    publication = PublicationStore.load(ROOT / "site/data").get(entry["snapshot_id"])
    return entry, artifact, publication


def _set_prediction_game_date(team_season: dict[str, Any], game_date: str) -> str:
    candidate = next(
        game
        for team in team_season["teams"].values()
        for game in team["games"]
        if game.get("future_prediction_id") is not None
        and game.get("score") is None
        and game.get("result") is None
    )
    game_id = str(candidate["game_id"])
    for team in team_season["teams"].values():
        for game in team["games"]:
            if str(game.get("game_id")) == game_id:
                game["date"] = game_date
    return game_id


def _validate_october_24_prediction(
    october_24_publication: tuple[dict[str, Any], dict[str, Any], Any],
    game_date: str,
) -> None:
    _, source_artifact, publication = october_24_publication
    artifact = deepcopy(source_artifact)
    _set_prediction_game_date(artifact, game_date)
    artifact["effective_cutoff"] = OCTOBER_24_CUTOFF
    metadata = publication.metadata.model_copy(
        update={"effective_cutoff": datetime.fromisoformat(OCTOBER_24_CUTOFF)}
    )
    _validate_team_season_artifact(artifact, metadata, list(publication.rankings))


def test_team_season_predictions_use_the_cutoff_utc_calendar_day(
    october_24_publication: tuple[dict[str, Any], dict[str, Any], Any],
) -> None:
    _validate_october_24_prediction(october_24_publication, OCTOBER_24_GAME_DATE)
    _validate_october_24_prediction(october_24_publication, "2026-10-25T00:00:00+00:00")


def test_team_season_rejects_predictions_from_before_the_cutoff_utc_day(
    october_24_publication: tuple[dict[str, Any], dict[str, Any], Any],
) -> None:
    with pytest.raises(
        PublicationDataError, match="future prediction is not after the cutoff"
    ):
        _validate_october_24_prediction(
            october_24_publication, "2026-10-23T23:59:59+00:00"
        )


def test_team_season_still_rejects_completed_evidence_outside_the_snapshot(
    october_24_publication: tuple[dict[str, Any], dict[str, Any], Any],
) -> None:
    _, source_artifact, publication = october_24_publication
    artifact = deepcopy(source_artifact)
    included_ids = {str(game_id) for game_id in artifact["included_game_ids"]}
    game = next(
        game
        for team in artifact["teams"].values()
        for game in team["games"]
        if game.get("future_prediction_id") is not None
        and str(game["game_id"]) not in included_ids
    )
    game["score"] = {"home": 10, "away": 7}
    game["result"] = "W"
    artifact["effective_cutoff"] = OCTOBER_24_CUTOFF
    metadata = publication.metadata.model_copy(
        update={"effective_cutoff": datetime.fromisoformat(OCTOBER_24_CUTOFF)}
    )

    with pytest.raises(
        PublicationDataError, match="schedule exposes evidence outside the snapshot"
    ):
        _validate_team_season_artifact(artifact, metadata, list(publication.rankings))


def test_static_api_generation_accepts_october_24_same_day_prediction(
    tmp_path: Path,
    october_24_publication: tuple[dict[str, Any], dict[str, Any], Any],
) -> None:
    source_entry, _, _ = october_24_publication
    data_dir, entry = _copy_selected_publication(tmp_path, source_entry)
    team_path = data_dir / entry["team_seasons_path"].removeprefix("data/")
    weekly_path = data_dir / entry["week_games_path"].removeprefix("data/")
    snapshot_path = data_dir / entry["data_path"].removeprefix("data/")
    team_season = json.loads(team_path.read_text(encoding="utf-8"))
    weekly = json.loads(weekly_path.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    game_id = _set_prediction_game_date(team_season, OCTOBER_24_GAME_DATE)
    weekly_game_found = False
    for week in weekly["weeks"]:
        for game in week["games"]:
            if str(game["game_id"]) == game_id:
                game["date"] = OCTOBER_24_GAME_DATE
                weekly_game_found = True
    assert weekly_game_found

    for artifact in (team_season, weekly):
        artifact["effective_cutoff"] = OCTOBER_24_CUTOFF
    for metadata in (entry, snapshot):
        metadata["effective_cutoff"] = OCTOBER_24_CUTOFF
        metadata["requested_cutoff"] = OCTOBER_24_CUTOFF
        metadata["generation_timestamp"] = OCTOBER_24_CUTOFF

    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["snapshots"][0] = entry
    team_path.write_text(json.dumps(team_season), encoding="utf-8")
    weekly_path.write_text(json.dumps(weekly), encoding="utf-8")
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    api_root = tmp_path / "api" / "v1"
    result = build_static_api(data_dir=data_dir, output_directory=api_root)
    assert result["publication_count"] == 1
    assert (api_root / "index.json").is_file()


@pytest.fixture(scope="module")
def static_api(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    data_dir, entry = _copy_selected_publication(tmp_path_factory.mktemp("static-api"))
    api_root = data_dir.parent / "api" / "v1"
    build_static_api(data_dir=data_dir, output_directory=api_root)
    return api_root, entry


def _read(api_root: Path, relative: str) -> dict[str, Any]:
    return json.loads((api_root / relative).read_text(encoding="utf-8"))


def test_static_inventory_and_links_are_complete(
    static_api: tuple[Path, dict[str, Any]],
) -> None:
    api_root, entry = static_api
    index = _read(api_root, "index.json")
    inventory = _read(api_root, "publications.json")
    assert index["api_version"] == "v1"
    assert index["read_only"] is True
    assert inventory["count"] == 1
    publication = inventory["publications"][0]
    assert publication["snapshot_id"] == entry["snapshot_id"]
    assert publication["publication_status"] == entry["publication_status"]
    links = publication["links"]
    assert links["rankings"] == f"rankings/{entry['snapshot_id']}.json"
    assert links["weeks"] == f"rankings/{entry['snapshot_id']}/weeks.json"
    assert links["season_outlook"].endswith("/season-outlook.json")
    assert (api_root / links["rankings"]).is_file()
    assert (api_root / links["weeks"]).is_file()
    assert (api_root / links["season_outlook"]).is_file()


def test_week_games_are_canonical_once_and_match_individual_resources(
    static_api: tuple[Path, dict[str, Any]],
) -> None:
    api_root, entry = static_api
    base = f"rankings/{entry['snapshot_id']}"
    weeks = _read(api_root, f"{base}/weeks.json")
    all_games: list[dict[str, Any]] = []
    for summary in weeks["weeks"]:
        week = _read(api_root, f"{base}/weeks/{summary['key']}.json")
        games = week["games"]
        assert len(games) == summary["scheduled_game_count"]
        all_games.extend(games)
        for game in games:
            individual = _read(api_root, f"{base}/games/{game['game_id']}.json")
            assert individual["game"] == game
            assert game["home_team_id"] == game["home_team"]["team_id"]
            assert game["away_team_id"] == game["away_team"]["team_id"]
            if game["prediction"] is not None:
                assert game["prediction"]["home_team_id"] == game["home_team_id"]
                assert game["prediction"]["away_team_id"] == game["away_team_id"]

    assert len(all_games) == len({game["game_id"] for game in all_games})
    assert all_games
    assert any(
        game["state"] == "future" and game["prediction"] is None for game in all_games
    )


def test_team_resources_are_scoped_and_prediction_orientation_is_stable(
    static_api: tuple[Path, dict[str, Any]],
) -> None:
    api_root, entry = static_api
    base = f"rankings/{entry['snapshot_id']}"
    rankings = _read(api_root, f"{base}.json")
    row = rankings["rankings"][0]
    team = _read(api_root, f"{base}/teams/{row['team_id']}.json")
    season = _read(api_root, f"{base}/teams/{row['team_id']}/season.json")
    assert team["team"] == row
    assert season["ranking"] == row
    assert season["team"]["team_id"] == row["team_id"]
    assert all(
        game["opponent"]["team_id"] != row["team_id"] for game in season["games"]
    )

    weeks = _read(api_root, f"{base}/weeks.json")
    future = next(
        game
        for summary in weeks["weeks"]
        for game in _read(api_root, f"{base}/weeks/{summary['key']}.json")["games"]
        if game["prediction"] is not None
    )
    raw_weekly = json.loads(
        (
            api_root.parent.parent
            / "data"
            / entry["week_games_path"].removeprefix("data/")
        ).read_text(encoding="utf-8")
    )
    raw_prediction = raw_weekly["future_predictions"][future["game_id"]]
    assert future["prediction"]["expected_home_margin"] == pytest.approx(
        raw_prediction["expected_home_margin"]
    )
    assert future["prediction"]["home_team_id"] == future["home_team_id"]
    assert future["prediction"]["away_team_id"] == future["away_team_id"]
    assert weeks["axes"]["future_margin"]["direction"] == "home_minus_away"


def test_static_api_generation_is_deterministic(
    static_api: tuple[Path, dict[str, Any]],
) -> None:
    api_root, _ = static_api
    before = _tree_hash(api_root)
    build_static_api(
        data_dir=api_root.parent.parent / "data", output_directory=api_root
    )
    assert _tree_hash(api_root) == before


def test_canonical_context_is_unambiguous_and_retained_artifacts_are_readable() -> None:
    store = PublicationStore.load(ROOT / "site/data")
    week_2 = store.filter(
        season=2026,
        family="predictive",
        prior="context",
    )
    week_2 = [item for item in week_2 if item.metadata.publication_slot == "2026-09-08"]

    canonical_id = "2026-weekly-2026-09-08T11-43-00.275833Z-context-v1.4-db-repair"
    assert {item.metadata.snapshot_id for item in week_2} == {canonical_id}
    assert {item.metadata.model_versions.context_prior for item in week_2} == {"1.4"}
    assert store.get(canonical_id).metadata.model_versions.context_prior == "1.4"
    assert (
        store.resolve(
            season=2026,
            publication_slot="2026-09-08",
            family="predictive",
            prior="context",
        ).metadata.snapshot_id
        == canonical_id
    )
    for version, suffix in (
        ("1.2", "context"),
        ("1.3", "context-v1.3"),
        ("1.4", "context-v1.4"),
    ):
        retained = (
            ROOT
            / "data/processed/snapshots/2026"
            / f"2026-weekly-2026-09-08T11-43-00.275833Z-{suffix}"
            / "predictive/context/metadata.json"
        )
        assert json.loads(retained.read_text())["prior_model_version"] == version


def test_performance_variants_have_distinct_resources_and_legacy_selection() -> None:
    store = PublicationStore.load(ROOT / "site/data")
    slot = FULL_MANIFEST["default_publication_slot"]
    legacy = store.resolve(season=2026, publication_slot=slot, family="performance")
    record = store.resolve(season=2026, publication_slot=slot, family="performance", variant="quality_of_record")
    assert legacy.metadata.performance_variant == "quality_of_play"
    assert record.metadata.performance_variant == "quality_of_record"
    assert record.metadata.snapshot_id != legacy.metadata.snapshot_id
    assert record.metadata.evidence_mode == "win_loss"
    assert record.metadata.source_context_snapshot_id == legacy.metadata.source_context_snapshot_id
    assert record.predictions == legacy.predictions
    assert record.outlooks == legacy.outlooks
    for game_id, game in record.games.items():
        assert game.home_performance == legacy.games[game_id].home_performance
        assert game.away_performance == legacy.games[game_id].away_performance
    assert record.simulation == legacy.simulation
    assert [row.team_id for row in record.rankings] != [row.team_id for row in legacy.rankings]
    assert len(store.filter(family="performance", variant="quality_of_record")) >= 1


def test_official_week_2_qor_static_api_exposes_the_exact_exception(tmp_path: Path) -> None:
    entries = [entry for entry in FULL_MANIFEST["snapshots"] if entry["publication_slot"] == "2026-09-08" and (entry.get("prior_family") == "context" or entry.get("performance_variant") == "quality_of_record")]
    assert len(entries) == 2
    data_dir = tmp_path / "data"
    for entry in entries:
        for key in ("data_path", "distribution_path", "team_seasons_path", "week_games_path"):
            destination = tmp_path / entry[key]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "site" / entry[key], destination)
    shutil.copyfile(ROOT / "site/data/methodology.json", data_dir / "methodology.json")
    (data_dir / "manifest.json").write_text(json.dumps({**FULL_MANIFEST, "snapshots": entries}))
    api_root = tmp_path / "api"
    build_static_api(data_dir=data_dir, output_directory=api_root)
    inventory = _read(api_root, "publications.json")
    publication = next(item for item in inventory["publications"] if item.get("performance_variant") == "quality_of_record")
    assert publication["publication_status"] == "official"
    evidence = publication["quality_of_record_evidence"]
    assert evidence["source_game_count"] == 172
    assert evidence["inference_game_count"] == 171
    assert evidence["exclusions"][0]["game_id"] == "401866625"
    assert "401866625" not in evidence["inference_game_ids"]
    assert "401917058" in evidence["inference_game_ids"]
    rankings = _read(api_root, publication["links"]["rankings"])
    assert rankings["publication"]["quality_of_record_evidence"] == evidence
    assert len(rankings["rankings"]) == 138
    resolved = PublicationStore.load(data_dir).resolve(season=2026, publication_slot="2026-09-08", family="performance", variant="quality_of_record")
    assert resolved.metadata.snapshot_id == publication["snapshot_id"]
