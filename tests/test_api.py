from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from gippyrank.api.static import build_static_api

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


def _copy_selected_publication(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    entry = next(
        item
        for item in FULL_MANIFEST["snapshots"]
        if item["ranking_family"] == "predictive"
        and item.get("prior_family") == "context"
        and item["snapshot_type"] == "weekly"
    )
    data_dir = tmp_path / "data"
    methodology_source = ROOT / "site/data" / FULL_MANIFEST[
        "methodology_path"
    ].removeprefix("data/")
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
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return data_dir, entry


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
        game["state"] == "future" and game["prediction"] is None
        for game in all_games
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
    assert all(game["opponent"]["team_id"] != row["team_id"] for game in season["games"])

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
    build_static_api(data_dir=api_root.parent.parent / "data", output_directory=api_root)
    assert _tree_hash(api_root) == before
