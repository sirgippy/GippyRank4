"""FastAPI application for the published GippyRank data API."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from email.utils import format_datetime, parsedate_to_datetime
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .models import (
    API_VERSION,
    APIDataInfo,
    APIRootResponse,
    DisplayAxes,
    ErrorResponse,
    GameResponse,
    HealthResponse,
    PublicationsResponse,
    RankingsResponse,
    RankRow,
    RootResources,
    SeasonOutlookResponse,
    TeamIdentity,
    TeamLinks,
    TeamRankingResponse,
    TeamSeasonResponse,
    WeekResponse,
    WeeksResponse,
)
from .store import (
    AmbiguousPublicationError,
    LoadedPublication,
    PublicationDataError,
    PublicationStore,
)

LOGGER = logging.getLogger(__name__)
SERVICE_NAME = "GippyRank Public API"
COMMON_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Malformed request."},
    404: {"model": ErrorResponse, "description": "Published resource not found."},
    422: {"model": ErrorResponse, "description": "Semantically invalid selector."},
    500: {"model": ErrorResponse, "description": "Invalid publication server state."},
}
EXACT_CACHE_CONTROL = "public, max-age=31536000, immutable"
LIST_CACHE_CONTROL = "public, max-age=300"
ALIAS_CACHE_CONTROL = "public, max-age=60"


class APIError(Exception):
    """A stable, user-facing API error."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _error_response(
    error: APIError, *, headers: dict[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"code": error.code, "message": error.message}},
        headers=headers,
    )


def _cors_origins() -> list[str]:
    configured = os.environ.get("GIPPYRANK_API_CORS_ORIGINS", "*")
    origins = [item.strip() for item in configured.split(",") if item.strip()]
    return origins or ["*"]


def _last_modified(value: datetime | None) -> str | None:
    if value is None:
        return None
    normalized = value.astimezone(UTC).replace(microsecond=0)
    return format_datetime(normalized, usegmt=True)


def _response(
    model: object,
    request: Request,
    *,
    cache_control: str,
    last_modified: datetime | None = None,
) -> Response:
    """Serialize a validated public model and apply conditional caching."""
    data = model.model_dump(mode="json", by_alias=True)  # type: ignore[union-attr]
    body = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    etag = f'"{sha256(body).hexdigest()}"'
    headers = {
        "Cache-Control": cache_control,
        "ETag": etag,
    }
    modified = _last_modified(last_modified)
    if modified is not None:
        headers["Last-Modified"] = modified

    if _not_modified(request, etag, last_modified):
        return Response(status_code=304, headers=headers)
    return Response(content=body, media_type="application/json", headers=headers)


def _not_modified(request: Request, etag: str, last_modified: datetime | None) -> bool:
    supplied_etag = request.headers.get("if-none-match")
    if supplied_etag:
        return supplied_etag == "*" or etag in {
            item.strip() for item in supplied_etag.split(",")
        }
    supplied_date = request.headers.get("if-modified-since")
    if not supplied_date or last_modified is None:
        return False
    try:
        requested = parsedate_to_datetime(supplied_date)
    except (TypeError, ValueError, OverflowError):
        return False
    if requested.tzinfo is None:
        requested = requested.replace(tzinfo=UTC)
    return last_modified.astimezone(UTC).replace(microsecond=0) <= requested.astimezone(
        UTC
    )


def _validate_selector(
    *, family: str | None, prior: str | None, status: str | None
) -> None:
    if family is not None and family not in {"predictive", "performance"}:
        raise APIError(
            400, "invalid_family", "family must be predictive or performance."
        )
    if prior is not None and prior not in {"context", "history"}:
        raise APIError(400, "invalid_prior", "prior must be context or history.")
    if status is not None and status not in {"official", "temporary"}:
        raise APIError(400, "invalid_status", "status must be official or temporary.")
    if family == "performance" and prior is not None:
        raise APIError(
            422,
            "invalid_selector",
            "Performance publications do not have a predictive prior family.",
        )


def _store(request: Request) -> PublicationStore:
    store = request.app.state.publication_store
    if store is None:
        raise APIError(
            500,
            "publication_data_unavailable",
            "Published GippyRank data is unavailable.",
        )
    return store


def _publication(request: Request, snapshot_id: str) -> LoadedPublication:
    try:
        return _store(request).get(snapshot_id)
    except KeyError as error:
        raise APIError(
            404,
            "snapshot_not_found",
            "No published snapshot matches the requested identifier.",
        ) from error


def _team_publication(
    request: Request, snapshot_id: str, team_id: str
) -> tuple[LoadedPublication, RankRow]:
    publication = _publication(request, snapshot_id)
    try:
        return publication, publication.rankings_by_team[team_id]
    except KeyError as error:
        raise APIError(
            404,
            "team_not_found",
            "The requested stable team ID is not published in this snapshot.",
        ) from error


def _encoded(value: str) -> str:
    return quote(value, safe="")


def _axes(publication: LoadedPublication) -> DisplayAxes:
    return publication.axes


def _publication_last_modified(publication: LoadedPublication) -> datetime:
    return publication.metadata.generation_timestamp


def create_app(data_dir: Path | str | None = None) -> FastAPI:
    """Create the API application, optionally using a test publication directory."""
    try:
        store = PublicationStore.load(data_dir)
    except PublicationDataError as error:
        LOGGER.exception("Publication data failed validation; API is fail-closed")
        store = None
        load_error = error
    else:
        load_error = None

    app = FastAPI(
        title=SERVICE_NAME,
        version=API_VERSION,
        description=(
            "A stable, read-only publication interface for validated GippyRank "
            "artifacts. Predictive Context, Predictive History, and Performance "
            "are separate publication families. Expected rank is a distribution "
            "mean, not a deterministic ordinal; intervals preserve rank uncertainty. "
            "Completed-game performance describes observed game evidence and is "
            "distinct from team quality. Future predictions use the published "
            "home-minus-away distribution. Historical snapshots are leakage-safe "
            "at their effective cutoff, and official and temporary publications "
            "are never silently substituted. The API performs no inference, CFBD "
            "acquisition, or artifact mutation."
        ),
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.publication_store = store
    app.state.publication_load_error = load_error
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["Accept", "If-None-Match", "If-Modified-Since"],
        expose_headers=["ETag", "Last-Modified", "Cache-Control"],
        max_age=600,
    )

    @app.exception_handler(APIError)
    async def api_error_handler(_: Request, error: APIError) -> JSONResponse:
        if error.status_code >= 500:
            LOGGER.error("API request failed: %s", error.code)
        return _error_response(error)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        _: Request, error: RequestValidationError
    ) -> JSONResponse:
        LOGGER.info("Malformed API request: %s", error.errors())
        return _error_response(
            APIError(400, "malformed_request", "The request parameters are malformed.")
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        _: Request, error: StarletteHTTPException
    ) -> JSONResponse:
        if error.status_code == 404:
            api_error = APIError(
                404, "not_found", "The requested API resource was not found."
            )
        elif error.status_code == 405:
            api_error = APIError(
                405, "method_not_allowed", "Only read-only GET requests are supported."
            )
        else:
            api_error = APIError(
                error.status_code, "http_error", "The request could not be served."
            )
        return _error_response(api_error)

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={
            503: {
                "model": ErrorResponse,
                "description": "Publication data is unavailable.",
            }
        },
        tags=["health"],
        summary="Check service and publication-data readiness",
    )
    def health(request: Request) -> Response:
        store = request.app.state.publication_store
        if store is None:
            raise APIError(
                503,
                "publication_data_unavailable",
                "The API is running but its publication data did not load.",
            )
        result = HealthResponse(
            status="ok",
            service=SERVICE_NAME,
            publication_data_schema_version=store.schema_version,
            publication_count=len(store.publications),
            season_count=len(store.seasons),
        )
        return _response(result, request, cache_control="no-store")

    @app.get(
        "/api/v1/",
        response_model=APIRootResponse,
        responses=COMMON_RESPONSES,
        tags=["root"],
        summary="Identify API V1 and its resources",
    )
    def api_root(request: Request) -> Response:
        store = _store(request)
        result = APIRootResponse(
            name=SERVICE_NAME,
            description="Read-only access to intentionally published GippyRank data.",
            read_only=True,
            data=APIDataInfo(
                schema_version=store.schema_version,
                publication_count=len(store.publications),
                season_count=len(store.seasons),
            ),
            resources=RootResources(
                publications="/api/v1/publications",
                rankings="/api/v1/rankings/{snapshot_id}",
                health="/health",
                openapi="/openapi.json",
            ),
        )
        return _response(result, request, cache_control=LIST_CACHE_CONTROL)

    @app.get(
        "/api/v1/publications",
        response_model=PublicationsResponse,
        responses=COMMON_RESPONSES,
        tags=["publications"],
        summary="List available published snapshots",
    )
    def publications(
        request: Request,
        season: int | None = None,
        family: str | None = None,
        prior: str | None = None,
        status: str | None = None,
    ) -> Response:
        _validate_selector(family=family, prior=prior, status=status)
        store = _store(request)
        values = store.filter(season=season, family=family, prior=prior, status=status)
        result = PublicationsResponse(
            count=len(values),
            publications=[item.metadata for item in values],
        )
        return _response(result, request, cache_control=LIST_CACHE_CONTROL)

    @app.get(
        "/api/v1/rankings/{snapshot_id}",
        response_model=RankingsResponse,
        responses=COMMON_RESPONSES,
        tags=["rankings"],
        summary="Get the ranking table for an exact published snapshot",
    )
    def rankings(request: Request, snapshot_id: str) -> Response:
        publication = _publication(request, snapshot_id)
        result = RankingsResponse(
            publication=publication.metadata,
            ordering="display_rank_ascending_nr_last",
            rankings=list(publication.rankings),
        )
        return _response(
            result,
            request,
            cache_control=EXACT_CACHE_CONTROL,
            last_modified=_publication_last_modified(publication),
        )

    @app.get(
        "/api/v1/rankings/{snapshot_id}/teams/{team_id}",
        response_model=TeamRankingResponse,
        responses=COMMON_RESPONSES,
        tags=["teams"],
        summary="Get one team's ranking and published rank distribution",
    )
    def team_ranking(request: Request, snapshot_id: str, team_id: str) -> Response:
        publication, row = _team_publication(request, snapshot_id, team_id)
        rank_distribution = publication.distributions.get(team_id)
        if rank_distribution is None:
            raise APIError(
                500,
                "publication_data_invalid",
                "The selected publication has no valid rank distribution for this team.",
            )
        base = f"/api/v1/rankings/{_encoded(snapshot_id)}"
        result = TeamRankingResponse(
            publication=publication.metadata,
            team=row,
            rank_distribution=rank_distribution,
            links=TeamLinks(
                rankings=f"{base}/teams/{_encoded(team_id)}",
                season=f"{base}/teams/{_encoded(team_id)}/season",
            ),
        )
        return _response(
            result,
            request,
            cache_control=EXACT_CACHE_CONTROL,
            last_modified=_publication_last_modified(publication),
        )

    @app.get(
        "/api/v1/rankings/{snapshot_id}/teams/{team_id}/season",
        response_model=TeamSeasonResponse,
        responses=COMMON_RESPONSES,
        tags=["teams"],
        summary="Get a team's snapshot-safe schedule and season outlook",
    )
    def team_season(request: Request, snapshot_id: str, team_id: str) -> Response:
        publication, row = _team_publication(request, snapshot_id, team_id)
        try:
            games = publication.team_games[team_id]
        except KeyError as error:
            raise APIError(
                500,
                "publication_data_invalid",
                "The selected publication has no valid team-season data.",
            ) from error
        result = TeamSeasonResponse(
            publication=publication.metadata,
            team=TeamIdentity(
                team_id=row.team_id,
                team_name=row.team_name,
                subdivision="fbs",
                conference=row.conference,
            ),
            ranking=row,
            games=list(games),
            season_outlook=publication.outlooks.get(team_id),
            season_simulation=publication.simulation,
            axes=_axes(publication),
        )
        return _response(
            result,
            request,
            cache_control=EXACT_CACHE_CONTROL,
            last_modified=_publication_last_modified(publication),
        )

    @app.get(
        "/api/v1/rankings/{snapshot_id}/season-outlook",
        response_model=SeasonOutlookResponse,
        responses=COMMON_RESPONSES,
        tags=["season"],
        summary="Get published regular-season simulation summaries",
    )
    def season_outlook(request: Request, snapshot_id: str) -> Response:
        publication = _publication(request, snapshot_id)
        outlooks = [
            publication.outlooks[row.team_id]
            for row in publication.rankings
            if row.team_id in publication.outlooks
        ]
        result = SeasonOutlookResponse(
            publication=publication.metadata,
            season_simulation=publication.simulation,
            outlooks=outlooks,
        )
        return _response(
            result,
            request,
            cache_control=EXACT_CACHE_CONTROL,
            last_modified=_publication_last_modified(publication),
        )

    @app.get(
        "/api/v1/rankings/{snapshot_id}/weeks",
        response_model=WeeksResponse,
        responses=COMMON_RESPONSES,
        tags=["games"],
        summary="List the weeks represented in a published snapshot",
    )
    def weeks(request: Request, snapshot_id: str) -> Response:
        publication = _publication(request, snapshot_id)
        result = WeeksResponse(
            publication=publication.metadata,
            default_week=publication.metadata.default_week
            or publication.weekly_games.get("default_week"),
            week_count=len(publication.weeks),
            weeks=list(publication.weeks),
            axes=_axes(publication),
        )
        return _response(
            result,
            request,
            cache_control=EXACT_CACHE_CONTROL,
            last_modified=_publication_last_modified(publication),
        )

    @app.get(
        "/api/v1/rankings/{snapshot_id}/weeks/{week}",
        response_model=WeekResponse,
        responses=COMMON_RESPONSES,
        tags=["games"],
        summary="Get all canonical games for one exact published week",
    )
    def week(request: Request, snapshot_id: str, week: str) -> Response:
        publication = _publication(request, snapshot_id)
        try:
            week_index = publication.weeks_by_key[week]
        except KeyError as error:
            raise APIError(
                404,
                "week_not_found",
                "The requested week is not represented in this snapshot.",
            ) from error
        raw_week = publication.weekly_games["weeks"][week_index]
        games = [publication.games[str(item["game_id"])] for item in raw_week["games"]]
        result = WeekResponse(
            publication=publication.metadata,
            week=publication.weeks[week_index],
            games=games,
            axes=_axes(publication),
        )
        return _response(
            result,
            request,
            cache_control=EXACT_CACHE_CONTROL,
            last_modified=_publication_last_modified(publication),
        )

    @app.get(
        "/api/v1/rankings/{snapshot_id}/games/{game_id}",
        response_model=GameResponse,
        responses=COMMON_RESPONSES,
        tags=["games"],
        summary="Get one canonical snapshot-scoped game",
    )
    def game(request: Request, snapshot_id: str, game_id: str) -> Response:
        publication = _publication(request, snapshot_id)
        try:
            value = publication.games[game_id]
        except KeyError as error:
            raise APIError(
                404,
                "game_not_found",
                "The requested game is not published in this snapshot.",
            ) from error
        result = GameResponse(
            publication=publication.metadata,
            game=value,
            axes=_axes(publication),
        )
        return _response(
            result,
            request,
            cache_control=EXACT_CACHE_CONTROL,
            last_modified=_publication_last_modified(publication),
        )

    @app.get(
        "/api/v1/seasons/{season}/publications/{publication_slot}/rankings",
        response_model=RankingsResponse,
        responses=COMMON_RESPONSES,
        tags=["publications", "rankings"],
        summary="Resolve a publication slot to one exact ranking view",
    )
    def publication_rankings(
        request: Request,
        season: int,
        publication_slot: str,
        family: str | None = None,
        prior: str | None = None,
        status: str | None = None,
    ) -> Response:
        _validate_selector(family=family, prior=prior, status=status)
        try:
            publication = _store(request).resolve(
                season=season,
                publication_slot=publication_slot,
                family=family,
                prior=prior,
                status=status,
            )
        except AmbiguousPublicationError as error:
            raise APIError(422, "ambiguous_selector", str(error)) from error
        except KeyError as error:
            raise APIError(
                404,
                "publication_not_found",
                "No published view matches the exact season, slot, family, prior, and status selector.",
            ) from error
        result = RankingsResponse(
            publication=publication.metadata,
            ordering="display_rank_ascending_nr_last",
            rankings=list(publication.rankings),
        )
        return _response(
            result,
            request,
            cache_control=ALIAS_CACHE_CONTROL,
        )

    return app


app = create_app()
