"""Build the machine-facing JSON tree published by GitHub Pages.

The static API is generated from the same manifest-approved site artifacts as
the former HTTP API.  :class:`PublicationStore` validates the artifact graph
and converts implementation-shaped JSON into the small public models; this
module only writes those validated models to stable files.  It never acquires
data, runs inference, or starts a server.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .models import (
    API_VERSION,
    APIDataInfo,
    GameResponse,
    PublicationLinks,
    PublicationsResponse,
    RankingsResponse,
    SeasonOutlookResponse,
    StaticAPIIndex,
    StaticRootResources,
    TeamIdentity,
    TeamLinks,
    TeamRankingResponse,
    TeamSeasonResponse,
    WeekResponse,
    WeeksResponse,
)
from .store import LoadedPublication, PublicationStore

STATIC_API_SCHEMA_VERSION = "1.0"
STATIC_API_NAME = "GippyRank Static JSON API"
STATIC_API_DESCRIPTION = (
    "Read-only access to approved GippyRank publication data. "
    "All links are relative to the /api/v1/ directory."
)


class StaticAPIError(ValueError):
    """Raised when a static API resource cannot be generated safely."""


def _segment(value: str, label: str) -> str:
    """Require an identifier to be safe and stable as a path component."""
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise StaticAPIError(f"{label} is not a safe static API path segment")
    return value


def _write_json(path: Path, value: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    return len(encoded)


def _model_data(value: Any) -> dict[str, Any]:
    return value.model_dump(mode="json", by_alias=True)


def _publication_links(snapshot_id: str) -> PublicationLinks:
    snapshot = _segment(snapshot_id, "snapshot ID")
    base = f"rankings/{snapshot}"
    return PublicationLinks(
        rankings=f"{base}.json",
        weeks=f"{base}/weeks.json",
        season_outlook=f"{base}/season-outlook.json",
        teams=f"{base}/teams/{{team_id}}.json",
        team_season=f"{base}/teams/{{team_id}}/season.json",
        games=f"{base}/games/{{game_id}}.json",
    )


def _metadata(publication: LoadedPublication):
    return publication.metadata.model_copy(
        update={"links": _publication_links(publication.metadata.snapshot_id)}
    )


def _base_path(publication: LoadedPublication) -> str:
    return f"rankings/{_segment(publication.metadata.snapshot_id, 'snapshot ID')}"


def _write_rankings(
    api_root: Path, publication: LoadedPublication, metadata: Any
) -> int:
    payload = RankingsResponse(
        publication=metadata,
        ordering="display_rank_ascending_nr_last",
        rankings=list(publication.rankings),
    )
    return _write_json(api_root / f"{_base_path(publication)}.json", _model_data(payload))


def _write_team_resources(
    api_root: Path, publication: LoadedPublication, metadata: Any
) -> int:
    total = 0
    base = _base_path(publication)
    for row in publication.rankings:
        team_id = _segment(row.team_id, "team ID")
        distribution = publication.distributions.get(row.team_id)
        if distribution is None:
            raise StaticAPIError(
                f"{publication.metadata.snapshot_id}: missing rank distribution for {team_id}"
            )
        ranking = TeamRankingResponse(
            publication=metadata,
            team=row,
            rank_distribution=distribution,
            links=TeamLinks(
                rankings=f"{base}/teams/{team_id}.json",
                season=f"{base}/teams/{team_id}/season.json",
            ),
        )
        total += _write_json(
            api_root / base / "teams" / f"{team_id}.json", _model_data(ranking)
        )

        try:
            games = publication.team_games[row.team_id]
        except KeyError as error:
            raise StaticAPIError(
                f"{publication.metadata.snapshot_id}: missing team season for {team_id}"
            ) from error
        season = TeamSeasonResponse(
            publication=metadata,
            team=TeamIdentity(
                team_id=row.team_id,
                team_name=row.team_name,
                subdivision="fbs",
                conference=row.conference,
            ),
            ranking=row,
            games=list(games),
            season_outlook=publication.outlooks.get(row.team_id),
            season_simulation=publication.simulation,
            axes=publication.axes,
        )
        total += _write_json(
            api_root / base / "teams" / team_id / "season.json", _model_data(season)
        )
    return total


def _write_week_resources(
    api_root: Path, publication: LoadedPublication, metadata: Any
) -> int:
    total = 0
    base = _base_path(publication)
    weeks = WeeksResponse(
        publication=metadata,
        default_week=publication.metadata.default_week
        or publication.weekly_games.get("default_week"),
        week_count=len(publication.weeks),
        weeks=list(publication.weeks),
        axes=publication.axes,
    )
    total += _write_json(api_root / base / "weeks.json", _model_data(weeks))

    for summary in publication.weeks:
        week_key = _segment(summary.key, "week")
        index = publication.weeks_by_key[summary.key]
        raw_week = publication.weekly_games["weeks"][index]
        week = WeekResponse(
            publication=metadata,
            week=summary,
            games=[
                publication.games[str(item["game_id"])]
                for item in raw_week["games"]
            ],
            axes=publication.axes,
        )
        total += _write_json(
            api_root / base / "weeks" / f"{week_key}.json", _model_data(week)
        )
    return total


def _write_game_resources(
    api_root: Path, publication: LoadedPublication, metadata: Any
) -> int:
    total = 0
    base = _base_path(publication)
    for game_id in sorted(publication.games):
        safe_game_id = _segment(game_id, "game ID")
        game = GameResponse(
            publication=metadata,
            game=publication.games[game_id],
            axes=publication.axes,
        )
        total += _write_json(
            api_root / base / "games" / f"{safe_game_id}.json", _model_data(game)
        )
    return total


def _write_season_outlook(
    api_root: Path, publication: LoadedPublication, metadata: Any
) -> int:
    payload = SeasonOutlookResponse(
        publication=metadata,
        season_simulation=publication.simulation,
        outlooks=[
            publication.outlooks[row.team_id]
            for row in publication.rankings
            if row.team_id in publication.outlooks
        ],
    )
    return _write_json(
        api_root / _base_path(publication) / "season-outlook.json",
        _model_data(payload),
    )


def _clear_output(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_static_api(
    *, data_dir: Path | str, output_directory: Path | str | None = None
) -> dict[str, Any]:
    """Generate the complete static V1 API tree from published site data.

    ``data_dir`` must contain the generated ``manifest.json`` and its
    manifest-listed artifacts.  By default the API is written next to that
    directory, e.g. ``site/data`` -> ``site/api/v1``.
    """
    source = Path(data_dir).resolve()
    api_root = (
        Path(output_directory).resolve()
        if output_directory is not None
        else source.parent / "api" / "v1"
    )
    if api_root == source:
        raise StaticAPIError("static API output must not overwrite publication data")
    store = PublicationStore.load(source)
    _clear_output(api_root)

    bytes_written = 0
    for publication in store.publications:
        metadata = _metadata(publication)
        bytes_written += _write_rankings(api_root, publication, metadata)
        bytes_written += _write_team_resources(api_root, publication, metadata)
        bytes_written += _write_week_resources(api_root, publication, metadata)
        bytes_written += _write_game_resources(api_root, publication, metadata)
        bytes_written += _write_season_outlook(api_root, publication, metadata)

    inventory = PublicationsResponse(
        count=len(store.publications),
        publications=[_metadata(item) for item in store.publications],
    )
    bytes_written += _write_json(api_root / "publications.json", _model_data(inventory))
    index = StaticAPIIndex(
        schema_version=STATIC_API_SCHEMA_VERSION,
        name=STATIC_API_NAME,
        description=STATIC_API_DESCRIPTION,
        read_only=True,
        data=APIDataInfo(
            schema_version=store.schema_version,
            publication_count=len(store.publications),
            season_count=len(store.seasons),
        ),
        resources=StaticRootResources(
            publications="publications.json",
            rankings="rankings/{snapshot_id}.json",
        ),
    )
    bytes_written += _write_json(api_root / "index.json", _model_data(index))

    return {
        "schema_version": STATIC_API_SCHEMA_VERSION,
        "api_version": API_VERSION,
        "base_path": "api/v1/",
        "index_path": "api/v1/index.json",
        "publications_path": "api/v1/publications.json",
        "publication_count": len(store.publications),
        "resource_bytes": bytes_written,
    }


__all__ = ["STATIC_API_SCHEMA_VERSION", "StaticAPIError", "build_static_api"]
