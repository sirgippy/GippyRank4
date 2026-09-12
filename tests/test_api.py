from __future__ import annotations

import json
from email.utils import format_datetime
from hashlib import sha256
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gippyrank.api.app import app, create_app

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "site/data/manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _entry(*, family: str = "predictive", prior: str | None = "context") -> dict:
    return next(
        item
        for item in MANIFEST["snapshots"]
        if item["ranking_family"] == family and item.get("prior_family") == prior
    )


def _weekly_entry() -> dict:
    return next(
        item
        for item in MANIFEST["snapshots"]
        if item["ranking_family"] == "predictive"
        and item.get("prior_family") == "context"
        and item["snapshot_type"] == "weekly"
    )


def _artifact(entry: dict, field: str) -> dict:
    path = ROOT / "site/data" / entry[field].removeprefix("data/")
    return json.loads(path.read_text(encoding="utf-8"))


def test_root_health_inventory_and_openapi(client: TestClient) -> None:
    root = client.get("/api/v1/")
    assert root.status_code == 200
    assert root.json()["api_version"] == "v1"
    assert root.json()["read_only"] is True
    assert root.json()["data"]["publication_count"] == len(MANIFEST["snapshots"])

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["publication_count"] == len(MANIFEST["snapshots"])
    assert health.headers["cache-control"] == "no-store"

    all_inventory = client.get("/api/v1/publications")
    assert all_inventory.status_code == 200
    assert all_inventory.json()["count"] == len(MANIFEST["snapshots"])

    inventory = client.get(
        "/api/v1/publications?season=2026&family=predictive&prior=context"
    )
    assert inventory.status_code == 200
    assert inventory.json()["count"] > 0
    assert all(
        item["ranking_family"] == "predictive" and item["prior_family"] == "context"
        for item in inventory.json()["publications"]
    )

    store = app.state.publication_store
    assert store is not None
    original_publications = store.publications
    maximum_generation = max(
        item.metadata.generation_timestamp for item in original_publications
    )
    removed = next(
        item
        for item in original_publications
        if item.metadata.generation_timestamp < maximum_generation
    )
    store.publications = tuple(
        item for item in original_publications if item is not removed
    )
    try:
        assert (
            max(item.metadata.generation_timestamp for item in store.publications)
            == maximum_generation
        )
        changed_inventory = client.get(
            "/api/v1/publications",
            headers={
                "If-Modified-Since": format_datetime(maximum_generation, usegmt=True)
            },
        )
        assert changed_inventory.status_code == 200
        assert changed_inventory.json()["count"] == all_inventory.json()["count"] - 1
        assert "last-modified" not in changed_inventory.headers
    finally:
        store.publications = original_publications

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert openapi.json()["info"]["version"] == "v1"
    paths = openapi.json()["paths"]
    assert "/api/v1/rankings/{snapshot_id}/games/{game_id}" in paths
    assert "/api/v1/seasons/{season}/publications/{publication_slot}/rankings" in paths


def test_publication_selectors_are_explicit(client: TestClient) -> None:
    ambiguous = client.get("/api/v1/seasons/2026/publications/2026-09-08/rankings")
    assert ambiguous.status_code == 422
    assert ambiguous.json() == {
        "error": {
            "code": "ambiguous_selector",
            "message": (
                "The publication selector matches more than one family or prior; "
                "specify family and prior explicitly."
            ),
        }
    }

    resolved = client.get(
        "/api/v1/seasons/2026/publications/2026-09-08/rankings"
        "?family=predictive&prior=context"
    )
    assert resolved.status_code == 200
    assert resolved.json()["publication"]["snapshot_id"].endswith("-context")
    assert resolved.json()["publication"]["anchor_family"] == "context"

    invalid = client.get("/api/v1/publications?family=performance&prior=context")
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "invalid_selector"

    no_substitution = client.get(
        "/api/v1/seasons/2026/publications/2026-09-07/rankings"
        "?family=predictive&prior=context&status=official"
    )
    assert no_substitution.status_code == 404
    assert no_substitution.json()["error"]["code"] == "publication_not_found"


def test_predictive_metadata_exposes_the_explicit_preseason_reference(
    client: TestClient,
) -> None:
    entry = _weekly_entry()
    response = client.get(f"/api/v1/rankings/{entry['snapshot_id']}")
    assert response.status_code == 200
    metadata = response.json()["publication"]
    assert metadata["preseason_snapshot_id"] == entry["preseason_snapshot_id"]
    assert metadata["preseason_display_label"] == entry["preseason_display_label"]
    assert metadata["preseason_distribution_path"] == entry[
        "preseason_distribution_path"
    ]

    performance = next(
        item
        for item in MANIFEST["snapshots"]
        if item["ranking_family"] == "performance"
    )
    performance_response = client.get(
        f"/api/v1/rankings/{performance['snapshot_id']}"
    )
    assert performance_response.status_code == 200
    performance_metadata = performance_response.json()["publication"]
    assert performance_metadata["preseason_snapshot_id"] is None
    assert performance_metadata["preseason_distribution_path"] is None


def test_rankings_and_rank_distribution_match_static_publication(
    client: TestClient,
) -> None:
    entry = _entry()
    source = _artifact(entry, "data_path")
    distribution = _artifact(entry, "distribution_path")
    response = client.get(f"/api/v1/rankings/{entry['snapshot_id']}")
    assert response.status_code == 200
    payload = response.json()
    assert [row["team_id"] for row in payload["rankings"]] == [
        row["team_id"] for row in source["rankings"]
    ]
    assert payload["rankings"][0]["expected_rank"] == pytest.approx(
        source["rankings"][0]["expected_rank"]
    )
    assert (
        payload["rankings"][0]["display_rank"] == source["rankings"][0]["display_rank"]
    )

    team_id = source["rankings"][0]["team_id"]
    team_response = client.get(
        f"/api/v1/rankings/{entry['snapshot_id']}/teams/{team_id}"
    )
    assert team_response.status_code == 200
    team_payload = team_response.json()
    assert team_payload["team"]["team_id"] == team_id
    assert team_payload["rank_distribution"]["pmf"] == pytest.approx(
        distribution["teams"][team_id]["pmf"]
    )


def test_team_season_week_and_game_views_share_canonical_published_values(
    client: TestClient,
) -> None:
    entry = _weekly_entry()
    team_artifact = _artifact(entry, "team_seasons_path")
    weekly_artifact = _artifact(entry, "week_games_path")
    team_id = next(iter(team_artifact["teams"]))

    team_response = client.get(
        f"/api/v1/rankings/{entry['snapshot_id']}/teams/{team_id}/season"
    )
    assert team_response.status_code == 200
    team_payload = team_response.json()
    assert team_payload["team"]["team_id"] == team_id
    assert len(team_payload["games"]) == len(team_artifact["teams"][team_id]["games"])

    weeks_response = client.get(f"/api/v1/rankings/{entry['snapshot_id']}/weeks")
    assert weeks_response.status_code == 200
    assert weeks_response.json()["week_count"] == weekly_artifact["week_count"]
    raw_week = next(week for week in weekly_artifact["weeks"] if week["games"])
    week_response = client.get(
        f"/api/v1/rankings/{entry['snapshot_id']}/weeks/{raw_week['key']}"
    )
    assert week_response.status_code == 200
    week_payload = week_response.json()
    assert [game["game_id"] for game in week_payload["games"]] == [
        game["game_id"] for game in raw_week["games"]
    ]

    representative = next(
        game
        for game in week_payload["games"]
        if game["state"] in {"completed", "future"}
    )
    game_response = client.get(
        f"/api/v1/rankings/{entry['snapshot_id']}/games/{representative['game_id']}"
    )
    assert game_response.status_code == 200
    assert game_response.json()["game"] == representative
    if representative["state"] == "future":
        assert representative["score"] is None
        assert representative["home_performance"] is None
        assert representative["away_performance"] is None

    outlook = client.get(f"/api/v1/rankings/{entry['snapshot_id']}/season-outlook")
    assert outlook.status_code == 200
    source_summary = team_artifact["season_simulation"]["teams"][team_id]
    api_summary = next(
        item for item in outlook.json()["outlooks"] if item["team_id"] == team_id
    )
    assert api_summary["expected_final_wins"] == pytest.approx(
        source_summary["expected_final_wins"]
    )


def test_exact_publication_responses_are_cacheable_and_cors_is_read_only(
    client: TestClient,
) -> None:
    entry = _entry()
    path = f"/api/v1/rankings/{entry['snapshot_id']}"
    response = client.get(path, headers={"Origin": "https://example.com"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert "last-modified" in response.headers
    assert response.headers["etag"] == f'"{sha256(response.content).hexdigest()}"'
    assert response.headers["access-control-allow-origin"] == "*"
    conditional = client.get(path, headers={"If-None-Match": response.headers["etag"]})
    assert conditional.status_code == 304
    assert conditional.headers["etag"] == response.headers["etag"]

    write_attempt = client.post(path)
    assert write_attempt.status_code == 405
    assert write_attempt.json()["error"]["code"] == "method_not_allowed"


def test_errors_are_structured_and_exact_resources_do_not_fallback(
    client: TestClient,
) -> None:
    for path, code in (
        ("/api/v1/rankings/unknown", "snapshot_not_found"),
        ("/api/v1/rankings/2026-preseason-context/teams/unknown", "team_not_found"),
        ("/api/v1/rankings/2026-preseason-context/weeks/unknown", "week_not_found"),
        ("/api/v1/rankings/2026-preseason-context/games/unknown", "game_not_found"),
    ):
        response = client.get(path)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == code
    malformed = client.get("/api/v1/publications?season=not-a-season")
    assert malformed.status_code == 400
    assert malformed.json()["error"]["code"] == "malformed_request"


def test_invalid_publication_data_fails_closed(tmp_path: Path) -> None:
    unavailable = TestClient(create_app(tmp_path))
    health = unavailable.get("/health")
    assert health.status_code == 503
    assert health.json()["error"]["code"] == "publication_data_unavailable"
    root = unavailable.get("/api/v1/")
    assert root.status_code == 500
    assert root.json()["error"]["code"] == "publication_data_unavailable"
