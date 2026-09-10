"""Validated, read-only access to the generated publication artifacts.

This module intentionally knows nothing about CFBD acquisition or the model
implementation.  A :class:`PublicationStore` can only serve files selected by
the checked-in publication manifest, and it validates the cross-artifact
relationships before making a publication available to the API.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from pydantic import ValidationError

from .models import (
    API_VERSION,
    CanonicalGame,
    DisplayAxes,
    ExpectedWinsSummary,
    FutureMarginAxis,
    FuturePrediction,
    ModelVersions,
    PerformanceAxis,
    PerformanceSummary,
    PublicationMetadata,
    RankDistribution,
    RankDistributionSummary,
    RankRow,
    SeasonOutlook,
    SeasonScope,
    SeasonSimulationInfo,
    SimulationConfiguration,
    TeamIdentity,
    TeamSeasonGame,
    UnsupportedSeasonGame,
    VarianceDecomposition,
    WeekSummary,
)

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = "1.0"
DISPLAY_BINS = 40
DISPLAY_SCALE = 1000
PMF_TOLERANCE = 1.0e-8


class PublicationDataError(RuntimeError):
    """Raised when the published data cannot be safely served."""


class AmbiguousPublicationError(ValueError):
    """Raised when a convenience selector does not identify one publication."""


@dataclass(frozen=True)
class LoadedPublication:
    metadata: PublicationMetadata
    rankings: tuple[RankRow, ...]
    rankings_by_team: dict[str, RankRow]
    distributions: dict[str, RankDistribution]
    team_seasons: dict[str, dict[str, Any]]
    team_games: dict[str, tuple[TeamSeasonGame, ...]]
    weekly_games: dict[str, Any]
    weeks: tuple[WeekSummary, ...]
    weeks_by_key: dict[str, int]
    games: dict[str, CanonicalGame]
    predictions: dict[str, FuturePrediction]
    simulation: SeasonSimulationInfo | None
    outlooks: dict[str, SeasonOutlook]
    axes: DisplayAxes


class PublicationStore:
    """An immutable in-memory index of validated publication data."""

    def __init__(
        self,
        *,
        data_dir: Path,
        manifest: dict[str, Any],
        methodology: dict[str, Any],
        publications: tuple[LoadedPublication, ...],
    ) -> None:
        self.data_dir = data_dir
        self.manifest = manifest
        self.methodology = methodology
        self.publications = publications
        self._by_snapshot_id = {
            item.metadata.snapshot_id: item for item in publications
        }
        if len(self._by_snapshot_id) != len(publications):
            raise PublicationDataError("publication snapshot IDs are not unique")

    @classmethod
    def load(cls, data_dir: Path | str | None = None) -> PublicationStore:
        """Load and validate all artifacts named by ``manifest.json``."""
        configured = data_dir or os.environ.get("GIPPYRANK_PUBLICATION_DATA")
        if configured is None:
            configured_path = Path(__file__).resolve().parents[3] / "site" / "data"
        else:
            configured_path = Path(configured)
        root = configured_path.resolve()
        try:
            manifest = _read_object(root / "manifest.json", "manifest")
            if manifest.get("schema_version") != SCHEMA_VERSION:
                raise PublicationDataError("manifest schema version is unsupported")
            methodology = _load_methodology(root, manifest)
            entries = manifest.get("snapshots")
            if not isinstance(entries, list) or not entries:
                raise PublicationDataError(
                    "manifest snapshots must be a non-empty list"
                )

            publications: list[LoadedPublication] = []
            seen_slots: set[tuple[int, str, str, str | None]] = set()
            for index, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    raise PublicationDataError(
                        f"manifest snapshot entry {index} is invalid"
                    )
                snapshot_id = entry.get("snapshot_id")
                if not isinstance(snapshot_id, str) or not snapshot_id:
                    raise PublicationDataError(
                        f"manifest snapshot entry {index} has no snapshot ID"
                    )
                if entry.get("valid") is not True:
                    raise PublicationDataError(
                        f"manifest snapshot entry {snapshot_id} is not approved"
                    )
                snapshot_path = _artifact_path(root, entry.get("data_path"), "snapshot")
                snapshot = _read_object(snapshot_path, f"snapshot {snapshot_id}")
                metadata = _publication_metadata(entry, snapshot)
                selector_key = (
                    metadata.season,
                    metadata.publication_slot,
                    metadata.ranking_family,
                    metadata.prior_family,
                )
                if selector_key in seen_slots:
                    raise PublicationDataError(
                        f"duplicate logical publication for {metadata.publication_slot}"
                    )
                seen_slots.add(selector_key)
                rankings = _load_rankings(snapshot, metadata)

                distribution_path = _artifact_path(
                    root, entry.get("distribution_path"), "distribution"
                )
                distribution_artifact = _read_object(
                    distribution_path, f"distribution {snapshot_id}"
                )
                distributions = _load_distributions(
                    distribution_artifact, metadata, rankings
                )

                team_season_path = _artifact_path(
                    root, entry.get("team_seasons_path"), "team-season"
                )
                team_season_artifact = _read_object(
                    team_season_path, f"team-season {snapshot_id}"
                )
                _validate_team_season_artifact(team_season_artifact, metadata, rankings)
                metadata = _merge_team_provenance(metadata, team_season_artifact)

                weekly_path = _artifact_path(
                    root, entry.get("week_games_path"), "weekly-game"
                )
                weekly_artifact = _read_object(
                    weekly_path, f"weekly games {snapshot_id}"
                )
                loaded_weekly = _load_weekly_artifact(
                    weekly_artifact,
                    team_season_artifact,
                    metadata,
                )

                prediction_values = loaded_weekly["predictions"]
                games = loaded_weekly["games"]
                team_games = _load_team_games(
                    team_season_artifact,
                    metadata,
                    rankings,
                    games,
                    prediction_values,
                    loaded_weekly["performance_displays"],
                )
                simulation, outlooks = _load_simulation(
                    team_season_artifact, metadata, rankings
                )
                axes = _load_axes(weekly_artifact, snapshot_id)
                publications.append(
                    LoadedPublication(
                        metadata=metadata,
                        rankings=tuple(rankings),
                        rankings_by_team={row.team_id: row for row in rankings},
                        distributions=distributions,
                        team_seasons=team_season_artifact["teams"],
                        team_games=team_games,
                        weekly_games=weekly_artifact,
                        weeks=tuple(loaded_weekly["weeks"]),
                        weeks_by_key=loaded_weekly["weeks_by_key"],
                        games=games,
                        predictions=prediction_values,
                        simulation=simulation,
                        outlooks=outlooks,
                        axes=axes,
                    )
                )
            LOGGER.info(
                "Loaded %d published snapshots for API %s from publication artifacts",
                len(publications),
                API_VERSION,
            )
            _validate_publication_relationships(publications)
            return cls(
                data_dir=root,
                manifest=manifest,
                methodology=methodology,
                publications=tuple(publications),
            )
        except PublicationDataError:
            raise
        except (KeyError, OSError, TypeError, ValueError, ValidationError) as error:
            raise PublicationDataError(str(error)) from error

    @property
    def schema_version(self) -> str:
        return str(self.manifest["schema_version"])

    @property
    def seasons(self) -> tuple[int, ...]:
        return tuple(
            sorted({item.metadata.season for item in self.publications}, reverse=True)
        )

    def get(self, snapshot_id: str) -> LoadedPublication:
        try:
            return self._by_snapshot_id[snapshot_id]
        except KeyError as error:
            raise KeyError("snapshot_not_found") from error

    def filter(
        self,
        *,
        season: int | None = None,
        family: str | None = None,
        prior: str | None = None,
        status: str | None = None,
    ) -> list[LoadedPublication]:
        return [
            item
            for item in self.publications
            if (season is None or item.metadata.season == season)
            and (family is None or item.metadata.ranking_family == family)
            and (prior is None or item.metadata.prior_family == prior)
            and (status is None or item.metadata.publication_status == status)
        ]

    def resolve(
        self,
        *,
        season: int,
        publication_slot: str,
        family: str | None = None,
        prior: str | None = None,
        status: str | None = None,
    ) -> LoadedPublication:
        matches = self.filter(
            season=season,
            family=family,
            prior=prior,
            status=status,
        )
        matches = [
            item
            for item in matches
            if item.metadata.publication_slot == publication_slot
        ]
        if not matches:
            raise KeyError("publication_not_found")
        if len(matches) != 1:
            raise AmbiguousPublicationError(
                "The publication selector matches more than one family or prior; "
                "specify family and prior explicitly."
            )
        return matches[0]


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicationDataError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise PublicationDataError(f"{label} must contain a JSON object")
    return value


def _artifact_path(root: Path, configured: object, label: str) -> Path:
    if not isinstance(configured, str) or not configured:
        raise PublicationDataError(f"manifest {label} path is missing")
    posix = PurePosixPath(configured)
    windows = PureWindowsPath(configured)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or "\\" in configured
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise PublicationDataError(f"manifest {label} path is not a safe relative path")
    parts = posix.parts[1:] if posix.parts and posix.parts[0] == "data" else posix.parts
    if not parts:
        raise PublicationDataError(f"manifest {label} path is empty")
    candidate = (root / Path(*parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise PublicationDataError(
            f"manifest {label} path escapes publication data"
        ) from error
    if not candidate.is_file():
        raise PublicationDataError(f"manifest {label} artifact is unavailable")
    return candidate


def _load_methodology(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = _artifact_path(root, manifest.get("methodology_path"), "methodology")
    methodology = _read_object(path, "methodology")
    if methodology.get("schema_version") != manifest.get("methodology_schema_version"):
        raise PublicationDataError("methodology schema does not match the manifest")
    if not isinstance(methodology.get("model_versions"), dict) or not isinstance(
        methodology.get("schema_versions"), dict
    ):
        raise PublicationDataError("methodology version metadata is malformed")
    return methodology


def _publication_metadata(
    entry: dict[str, Any], snapshot: dict[str, Any]
) -> PublicationMetadata:
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise PublicationDataError("snapshot schema version is unsupported")
    if snapshot.get("snapshot_id") != entry.get("snapshot_id"):
        raise PublicationDataError("manifest and snapshot IDs do not match")
    for field in (
        "season",
        "snapshot_type",
        "ranking_family",
        "prior_family",
        "publication_slot",
        "publication_status",
        "display_label",
        "requested_cutoff",
        "effective_cutoff",
        "generation_timestamp",
        "included_game_count",
        "rank_count",
    ):
        if field in entry and field in snapshot and entry[field] != snapshot[field]:
            raise PublicationDataError(
                f"manifest and snapshot metadata disagree on {field}"
            )
    family = entry.get("ranking_family", snapshot.get("ranking_family"))
    values = {
        "season": entry.get("season", snapshot.get("season")),
        "snapshot_id": entry.get("snapshot_id"),
        "snapshot_type": entry.get("snapshot_type", snapshot.get("snapshot_type")),
        "publication_slot": entry.get("publication_slot"),
        "publication_status": entry.get("publication_status"),
        "display_label": entry.get("display_label"),
        "ranking_family": family,
        "prior_family": entry.get("prior_family", snapshot.get("prior_family")),
        "requested_cutoff": entry.get(
            "requested_cutoff", snapshot.get("requested_cutoff")
        ),
        "effective_cutoff": entry.get(
            "effective_cutoff", snapshot.get("effective_cutoff")
        ),
        "generation_timestamp": entry.get(
            "generation_timestamp", snapshot.get("generation_timestamp")
        ),
        "source_retrieved_at": entry.get(
            "source_retrieved_at", snapshot.get("source_retrieved_at")
        ),
        "model_versions": _model_versions(
            entry.get("model_versions", snapshot.get("model_versions"))
        ),
        "model_version": entry.get("model_version", snapshot.get("model_version")),
        "method": entry.get("method", snapshot.get("method")),
        "anchor_family": entry.get("anchor_family", snapshot.get("anchor_family")),
        "source_context_snapshot_id": entry.get(
            "source_context_snapshot_id", snapshot.get("source_context_snapshot_id")
        ),
        "included_game_count": entry.get(
            "included_game_count", snapshot.get("included_game_count", 0)
        ),
        "excluded_lower_division_games": entry.get(
            "excluded_lower_division_games",
            snapshot.get("excluded_lower_division_games", 0),
        ),
        "rank_count": entry.get("rank_count", snapshot.get("rank_count")),
        "rated_count": entry.get("rated_count", snapshot.get("rated_count")),
        "unrated_count": entry.get("unrated_count", snapshot.get("unrated_count")),
        "default_week": entry.get("default_week"),
        "comparison_snapshot_id": entry.get(
            "comparison_snapshot_id", snapshot.get("comparison_snapshot_id")
        ),
        "comparison_display_label": entry.get(
            "comparison_display_label", snapshot.get("comparison_display_label")
        ),
    }
    try:
        metadata = PublicationMetadata.model_validate(values)
    except ValidationError as error:
        raise PublicationDataError("publication metadata is malformed") from error
    if metadata.ranking_family == "predictive" and metadata.prior_family is None:
        raise PublicationDataError("predictive publication is missing its prior family")
    if metadata.ranking_family == "performance" and metadata.prior_family is not None:
        raise PublicationDataError(
            "Performance publication cannot declare a prior family"
        )
    return metadata


def _model_versions(value: object) -> ModelVersions:
    if not isinstance(value, dict):
        raise PublicationDataError("publication model versions are malformed")
    known = {
        field: value.get(field)
        for field in ModelVersions.model_fields
        if field in value
    }
    try:
        return ModelVersions.model_validate(known)
    except ValidationError as error:
        raise PublicationDataError(
            "publication model versions are malformed"
        ) from error


def _expected_prediction_source(metadata: PublicationMetadata) -> str:
    if metadata.ranking_family == "performance" or metadata.prior_family == "context":
        return "predictive_context"
    return "predictive_history"


def _expected_prediction_snapshot_id(metadata: PublicationMetadata) -> str:
    if metadata.ranking_family == "performance":
        if metadata.source_context_snapshot_id is None:
            raise PublicationDataError(
                f"{metadata.snapshot_id}: Performance prediction source is missing"
            )
        return metadata.source_context_snapshot_id
    return metadata.snapshot_id


def _validate_publication_relationships(
    publications: list[LoadedPublication],
) -> None:
    by_snapshot_id = {item.metadata.snapshot_id: item for item in publications}
    for publication in publications:
        source_id = publication.metadata.source_context_snapshot_id
        if source_id is None:
            raise PublicationDataError(
                f"{publication.metadata.snapshot_id}: source context snapshot is missing"
            )
        try:
            source = by_snapshot_id[source_id]
        except KeyError as error:
            raise PublicationDataError(
                f"{publication.metadata.snapshot_id}: source context snapshot is unavailable"
            ) from error
        if (
            source.metadata.ranking_family != "predictive"
            or source.metadata.prior_family != "context"
            or source.metadata.season != publication.metadata.season
            or source.metadata.publication_slot != publication.metadata.publication_slot
            or source.metadata.snapshot_type != publication.metadata.snapshot_type
            or source.metadata.effective_cutoff != publication.metadata.effective_cutoff
        ):
            raise PublicationDataError(
                f"{publication.metadata.snapshot_id}: source context snapshot is incompatible"
            )


def _merge_team_provenance(
    metadata: PublicationMetadata, artifact: dict[str, Any]
) -> PublicationMetadata:
    anchor_family = artifact.get("anchor_family")
    source_context_snapshot_id = artifact.get("source_context_snapshot_id")
    if not isinstance(anchor_family, str) or not anchor_family:
        raise PublicationDataError(
            f"{metadata.snapshot_id}: team-season anchor family is missing"
        )
    if (
        not isinstance(source_context_snapshot_id, str)
        or not source_context_snapshot_id
    ):
        raise PublicationDataError(
            f"{metadata.snapshot_id}: team-season source context snapshot is missing"
        )
    if metadata.ranking_family == "performance" and anchor_family != "context":
        raise PublicationDataError(
            f"{metadata.snapshot_id}: Performance is not Context-anchored"
        )
    if metadata.ranking_family == "predictive" and anchor_family != "context":
        raise PublicationDataError(
            f"{metadata.snapshot_id}: predictive provenance is inconsistent"
        )
    return metadata.model_copy(
        update={
            "anchor_family": anchor_family,
            "source_context_snapshot_id": source_context_snapshot_id,
        }
    )


def _load_rankings(
    snapshot: dict[str, Any], metadata: PublicationMetadata
) -> list[RankRow]:
    values = snapshot.get("rankings")
    if not isinstance(values, list) or not values:
        raise PublicationDataError(f"{metadata.snapshot_id}: rankings are missing")
    try:
        rows = [RankRow.model_validate(value) for value in values]
    except ValidationError as error:
        raise PublicationDataError(
            f"{metadata.snapshot_id}: ranking rows are malformed"
        ) from error
    ids = [row.team_id for row in rows]
    if len(ids) != len(set(ids)):
        raise PublicationDataError(
            f"{metadata.snapshot_id}: ranking team IDs are duplicated"
        )
    rated_ranks = [row.display_rank for row in rows if row.rated]
    if rated_ranks != list(range(1, len(rated_ranks) + 1)):
        raise PublicationDataError(
            f"{metadata.snapshot_id}: ranking order is not deterministic"
        )
    if any(row.rated != (row.display_rank != "NR") for row in rows):
        raise PublicationDataError(
            f"{metadata.snapshot_id}: rated/NR state is inconsistent"
        )
    if len(rows) != metadata.rank_count:
        raise PublicationDataError(
            f"{metadata.snapshot_id}: rank count does not match metadata"
        )
    if sum(row.rated for row in rows) != metadata.rated_count:
        raise PublicationDataError(
            f"{metadata.snapshot_id}: rated count does not match metadata"
        )
    if sum(not row.rated for row in rows) != metadata.unrated_count:
        raise PublicationDataError(
            f"{metadata.snapshot_id}: unrated count does not match metadata"
        )
    return rows


def _load_distributions(
    artifact: dict[str, Any],
    metadata: PublicationMetadata,
    rankings: list[RankRow],
) -> dict[str, RankDistribution]:
    if (
        artifact.get("schema_version") != SCHEMA_VERSION
        or artifact.get("snapshot_id") != metadata.snapshot_id
    ):
        raise PublicationDataError(
            f"{metadata.snapshot_id}: distribution metadata is invalid"
        )
    if artifact.get("rank_count") != metadata.rank_count:
        raise PublicationDataError(
            f"{metadata.snapshot_id}: distribution rank count is invalid"
        )
    values = artifact.get("teams")
    ranking_ids = {row.team_id for row in rankings}
    if not isinstance(values, dict) or set(values) != ranking_ids:
        raise PublicationDataError(
            f"{metadata.snapshot_id}: distribution team set is invalid"
        )
    result: dict[str, RankDistribution] = {}
    for team_id, value in values.items():
        if not isinstance(value, dict):
            raise PublicationDataError(
                f"{metadata.snapshot_id}: distribution is malformed"
            )
        pmf = value.get("pmf")
        if not isinstance(pmf, list) or len(pmf) != metadata.rank_count:
            raise PublicationDataError(
                f"{metadata.snapshot_id}: PMF support is invalid"
            )
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not 0 <= float(item) <= 1
            for item in pmf
        ) or not math.isclose(
            math.fsum(float(item) for item in pmf), 1.0, abs_tol=PMF_TOLERANCE
        ):
            raise PublicationDataError(f"{metadata.snapshot_id}: PMF is not normalized")
        try:
            summary = RankDistributionSummary.model_validate(value.get("summary"))
            distribution = RankDistribution.model_validate(
                {"rank_count": metadata.rank_count, "pmf": pmf, "summary": summary}
            )
        except ValidationError as error:
            raise PublicationDataError(
                f"{metadata.snapshot_id}: distribution summary is malformed"
            ) from error
        ranking = next(row for row in rankings if row.team_id == team_id)
        if not math.isclose(
            summary.expected_rank, ranking.expected_rank, abs_tol=PMF_TOLERANCE
        ):
            raise PublicationDataError(
                f"{metadata.snapshot_id}: PMF summary disagrees with ranking"
            )
        if summary.median_rank != int(ranking.median_rank):
            raise PublicationDataError(
                f"{metadata.snapshot_id}: PMF median disagrees with ranking"
            )
        if summary.interval_80 != ranking.interval_80:
            raise PublicationDataError(
                f"{metadata.snapshot_id}: PMF interval disagrees with ranking"
            )
        result[str(team_id)] = distribution
    return result


def _validate_team_season_artifact(
    artifact: dict[str, Any], metadata: PublicationMetadata, rankings: list[RankRow]
) -> None:
    if (
        artifact.get("schema_version") != SCHEMA_VERSION
        or artifact.get("artifact_kind") != "team_season"
    ):
        raise PublicationDataError(
            f"{metadata.snapshot_id}: team-season schema is invalid"
        )
    if (
        artifact.get("snapshot_id") != metadata.snapshot_id
        or artifact.get("season") != metadata.season
    ):
        raise PublicationDataError(
            f"{metadata.snapshot_id}: team-season identity is invalid"
        )
    if _parse_datetime(artifact.get("effective_cutoff")) != metadata.effective_cutoff:
        raise PublicationDataError(
            f"{metadata.snapshot_id}: team-season cutoff provenance is inconsistent"
        )
    if artifact.get("prediction_source") != _expected_prediction_source(metadata):
        raise PublicationDataError(
            f"{metadata.snapshot_id}: team-season prediction source is invalid"
        )
    team_values = artifact.get("teams")
    ranking_ids = {row.team_id for row in rankings}
    if not isinstance(team_values, dict) or set(team_values) != ranking_ids:
        raise PublicationDataError(
            f"{metadata.snapshot_id}: team-season team set is invalid"
        )
    included = artifact.get("included_game_ids", [])
    if not isinstance(included, list) or any(
        not isinstance(value, (str, int)) for value in included
    ):
        raise PublicationDataError(
            f"{metadata.snapshot_id}: included game IDs are invalid"
        )
    included_ids = [str(value) for value in included]
    if len(included_ids) != len(set(included_ids)):
        raise PublicationDataError(
            f"{metadata.snapshot_id}: included game IDs are duplicated"
        )
    predictions = artifact.get("future_predictions", {})
    if not isinstance(predictions, dict):
        raise PublicationDataError(
            f"{metadata.snapshot_id}: future prediction map is invalid"
        )
    cutoff = _parse_datetime(artifact.get("effective_cutoff"))
    for team_id, team in team_values.items():
        if not isinstance(team, dict) or team.get("team_id") != team_id:
            raise PublicationDataError(
                f"{metadata.snapshot_id}: team identity is invalid"
            )
        games = team.get("games")
        if not isinstance(games, list):
            raise PublicationDataError(
                f"{metadata.snapshot_id}: team schedule is invalid"
            )
        game_ids: set[str] = set()
        for game in games:
            if not isinstance(game, dict) or not game.get("game_id"):
                raise PublicationDataError(
                    f"{metadata.snapshot_id}: team game is invalid"
                )
            game_id = str(game["game_id"])
            if game_id in game_ids:
                raise PublicationDataError(
                    f"{metadata.snapshot_id}: team game IDs are duplicated"
                )
            game_ids.add(game_id)
            if not isinstance(game.get("modeled", False), bool):
                raise PublicationDataError(
                    f"{metadata.snapshot_id}: modeled state is invalid"
                )
            game_date = _parse_datetime(game.get("date"))
            prediction_id = game.get("future_prediction_id")
            if prediction_id is not None and str(prediction_id) not in predictions:
                raise PublicationDataError(
                    f"{metadata.snapshot_id}: future prediction reference is invalid"
                )
            if game_id not in included_ids and (
                game.get("result") is not None
                or game.get("score") is not None
                or game.get("game_rating") is not None
                or game.get("modeled") is True
            ):
                raise PublicationDataError(
                    f"{metadata.snapshot_id}: schedule exposes evidence outside the snapshot"
                )
            if (
                prediction_id is not None
                and cutoff is not None
                and (game_date is None or game_date <= cutoff)
            ):
                raise PublicationDataError(
                    f"{metadata.snapshot_id}: future prediction is not after the cutoff"
                )
    expected_ids = {
        str(game.get("game_id"))
        for team in team_values.values()
        for game in team.get("games", [])
    }
    if not expected_ids:
        raise PublicationDataError(f"{metadata.snapshot_id}: team schedules are empty")


def _load_weekly_artifact(
    artifact: dict[str, Any],
    team_season: dict[str, Any],
    metadata: PublicationMetadata,
) -> dict[str, Any]:
    snapshot_id = metadata.snapshot_id
    if (
        artifact.get("schema_version") != SCHEMA_VERSION
        or artifact.get("artifact_kind") != "weekly_games"
    ):
        raise PublicationDataError(f"{snapshot_id}: weekly-game schema is invalid")
    if (
        artifact.get("snapshot_id") != snapshot_id
        or artifact.get("season") != metadata.season
    ):
        raise PublicationDataError(f"{snapshot_id}: weekly-game identity is invalid")
    if artifact.get("effective_cutoff") != team_season.get("effective_cutoff"):
        raise PublicationDataError(
            f"{snapshot_id}: weekly cutoff provenance is inconsistent"
        )
    if _parse_datetime(artifact.get("effective_cutoff")) != metadata.effective_cutoff:
        raise PublicationDataError(
            f"{snapshot_id}: weekly cutoff is inconsistent with publication metadata"
        )
    if artifact.get("prediction_source") != _expected_prediction_source(metadata):
        raise PublicationDataError(
            f"{snapshot_id}: weekly prediction source is invalid"
        )
    if artifact.get("ranking_family") not in {None, metadata.ranking_family}:
        raise PublicationDataError(f"{snapshot_id}: weekly ranking family is invalid")
    if artifact.get("prior_family") not in {None, metadata.prior_family}:
        raise PublicationDataError(f"{snapshot_id}: weekly prior family is invalid")
    weeks_value = artifact.get("weeks")
    if not isinstance(weeks_value, list):
        raise PublicationDataError(f"{snapshot_id}: weekly weeks are invalid")
    predictions_value = artifact.get("future_predictions", {})
    displays = artifact.get("performance_displays", {})
    if not isinstance(predictions_value, dict) or not isinstance(displays, dict):
        raise PublicationDataError(f"{snapshot_id}: weekly maps are invalid")
    predictions: dict[str, FuturePrediction] = {}
    for prediction_id, value in predictions_value.items():
        if not isinstance(value, dict) or str(prediction_id) != str(
            value.get("game_id")
        ):
            raise PublicationDataError(
                f"{snapshot_id}: future prediction identity is invalid"
            )
        try:
            prediction = _public_prediction(value)
        except ValidationError as error:
            raise PublicationDataError(
                f"{snapshot_id}: future prediction is malformed"
            ) from error
        if prediction.prediction_source != _expected_prediction_source(metadata):
            raise PublicationDataError(
                f"{snapshot_id}: future prediction source is invalid"
            )
        if prediction.source_snapshot_id != _expected_prediction_snapshot_id(metadata):
            raise PublicationDataError(
                f"{snapshot_id}: future prediction provenance is invalid"
            )
        predictions[str(prediction_id)] = prediction
    performance_displays: dict[str, list[int]] = {}
    for display_id, value in displays.items():
        if (
            not isinstance(value, list)
            or len(value) != DISPLAY_BINS
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or not 0 <= item <= DISPLAY_SCALE
                for item in value
            )
            or sum(value) != DISPLAY_SCALE
        ):
            raise PublicationDataError(f"{snapshot_id}: performance display is invalid")
        performance_displays[str(display_id)] = list(value)

    team_game_ids = {
        str(game.get("game_id"))
        for team in team_season.get("teams", {}).values()
        for game in team.get("games", [])
    }
    included_game_ids = {
        str(game_id) for game_id in team_season.get("included_game_ids", [])
    }
    games: dict[str, CanonicalGame] = {}
    loaded_weeks: list[WeekSummary] = []
    weeks_by_key: dict[str, int] = {}
    referenced_predictions: set[str] = set()
    for week_index, week in enumerate(weeks_value):
        if not isinstance(week, dict):
            raise PublicationDataError(f"{snapshot_id}: weekly entry is invalid")
        try:
            summary = WeekSummary.model_validate(
                {key: week.get(key) for key in WeekSummary.model_fields}
            )
        except ValidationError as error:
            raise PublicationDataError(
                f"{snapshot_id}: week summary is malformed"
            ) from error
        if summary.key in weeks_by_key:
            raise PublicationDataError(f"{snapshot_id}: week keys are duplicated")
        weeks_by_key[summary.key] = week_index
        loaded_weeks.append(summary)
        raw_games = week.get("games")
        if (
            not isinstance(raw_games, list)
            or len(raw_games) != summary.scheduled_game_count
        ):
            raise PublicationDataError(f"{snapshot_id}: week game count is invalid")
        state_counts = {
            "completed": 0,
            "future": 0,
            "unresolved": 0,
            "cancelled": 0,
            "out_of_scope": 0,
        }
        for raw_game in raw_games:
            if not isinstance(raw_game, dict) or not raw_game.get("game_id"):
                raise PublicationDataError(f"{snapshot_id}: weekly game is invalid")
            game_id = str(raw_game["game_id"])
            if game_id in games:
                raise PublicationDataError(
                    f"{snapshot_id}: game appears more than once"
                )
            if game_id not in team_game_ids:
                raise PublicationDataError(
                    f"{snapshot_id}: weekly game is absent from team schedules"
                )
            prediction_id = raw_game.get("future_prediction_id")
            if prediction_id is not None:
                prediction_id = str(prediction_id)
                if prediction_id not in predictions:
                    raise PublicationDataError(
                        f"{snapshot_id}: weekly prediction reference is invalid"
                    )
                referenced_predictions.add(prediction_id)
            game = _public_canonical_game(raw_game, predictions, performance_displays)
            if (
                metadata.effective_cutoff is not None
                and game.date is not None
                and game.date > metadata.effective_cutoff
                and (
                    game.state == "completed"
                    or game.score is not None
                    or game.home_performance is not None
                    or game.away_performance is not None
                )
            ):
                raise PublicationDataError(
                    f"{snapshot_id}: weekly game exposes evidence after the cutoff"
                )
            if game_id not in included_game_ids and (
                game.state == "completed"
                or game.score is not None
                or game.home_performance is not None
                or game.away_performance is not None
            ):
                raise PublicationDataError(
                    f"{snapshot_id}: weekly game exposes evidence outside the snapshot"
                )
            state_counts[game.state] += 1
            if game.state == "future" and (
                game.score is not None
                or game.home_performance is not None
                or game.away_performance is not None
            ):
                raise PublicationDataError(
                    f"{snapshot_id}: future game exposes completed evidence"
                )
            if game.state != "future" and game.prediction is not None:
                raise PublicationDataError(
                    f"{snapshot_id}: non-future game has a prediction"
                )
            games[game_id] = game
        if state_counts["completed"] != summary.completed_game_count:
            raise PublicationDataError(
                f"{snapshot_id}: completed week count is invalid"
            )
        if state_counts["future"] != summary.future_game_count:
            raise PublicationDataError(f"{snapshot_id}: future week count is invalid")
        if state_counts["cancelled"] != summary.cancelled_game_count:
            raise PublicationDataError(
                f"{snapshot_id}: cancelled week count is invalid"
            )
        if (
            state_counts["unresolved"] + state_counts["out_of_scope"]
            != summary.unresolved_game_count
        ):
            raise PublicationDataError(
                f"{snapshot_id}: unresolved week count is invalid"
            )
    if len(games) != artifact.get("game_count") or len(loaded_weeks) != artifact.get(
        "week_count"
    ):
        raise PublicationDataError(
            f"{snapshot_id}: weekly aggregate counts are invalid"
        )
    if set(games) != team_game_ids:
        raise PublicationDataError(f"{snapshot_id}: weekly game membership is invalid")
    if set(predictions) != referenced_predictions:
        raise PublicationDataError(
            f"{snapshot_id}: future prediction map has unreferenced entries"
        )
    if artifact.get("included_game_ids") != team_season.get("included_game_ids"):
        raise PublicationDataError(
            f"{snapshot_id}: weekly evidence provenance is inconsistent"
        )
    if predictions_value != team_season.get("future_predictions", {}):
        raise PublicationDataError(
            f"{snapshot_id}: weekly prediction map is not canonical"
        )
    return {
        "predictions": predictions,
        "performance_displays": performance_displays,
        "games": games,
        "weeks": loaded_weeks,
        "weeks_by_key": weeks_by_key,
    }


def _public_prediction(value: dict[str, Any]) -> FuturePrediction:
    fields = (
        "game_id",
        "prediction_source",
        "source_snapshot_id",
        "home_team_id",
        "home_team_name",
        "home_subdivision",
        "away_team_id",
        "away_team_name",
        "away_subdivision",
        "neutral_site",
        "expected_home_margin",
        "median_home_margin",
        "home_win_probability",
        "away_win_probability",
        "tie_probability",
        "margin_interval_50",
        "margin_interval_80",
        "margin_interval_95",
        "display_distribution",
    )
    return FuturePrediction.model_validate(
        {field: value.get(field) for field in fields}
    )


def _public_performance(
    value: dict[str, Any] | None, display: list[int] | None = None
) -> PerformanceSummary | None:
    if value is None:
        return None
    fields = (
        "rank_count",
        "expected_rank",
        "median_rank",
        "mode_rank",
        "interval_50",
        "interval_80",
        "interval_95",
        "top5_probability",
        "top10_probability",
        "top25_probability",
        "performance_percentile",
        "performance_grade",
    )
    payload = {field: value.get(field) for field in fields}
    if display is not None:
        payload["display_pmf"] = display
    elif isinstance(value.get("display_pmf"), list):
        payload["display_pmf"] = value["display_pmf"]
    return PerformanceSummary.model_validate(payload)


def _public_team(value: dict[str, Any]) -> TeamIdentity:
    return TeamIdentity.model_validate(
        {
            "team_id": str(value.get("team_id")),
            "team_name": value.get("team_name"),
            "subdivision": str(value.get("subdivision", "")).casefold(),
            "conference": value.get("conference", ""),
        }
    )


def _public_canonical_game(
    value: dict[str, Any],
    predictions: dict[str, FuturePrediction],
    performance_displays: dict[str, list[int]],
) -> CanonicalGame:
    game_id = str(value["game_id"])
    prediction_id = value.get("future_prediction_id")
    prediction_key = str(prediction_id) if prediction_id is not None else None
    prediction = predictions.get(prediction_key) if prediction_key is not None else None
    home_ref = value.get("home_performance_ref")
    away_ref = value.get("away_performance_ref")
    home_performance = _public_performance(
        value.get("home_performance"),
        performance_displays.get(str(home_ref)) if home_ref is not None else None,
    )
    away_performance = _public_performance(
        value.get("away_performance"),
        performance_displays.get(str(away_ref)) if away_ref is not None else None,
    )
    if (home_ref is not None and str(home_ref) not in performance_displays) or (
        away_ref is not None and str(away_ref) not in performance_displays
    ):
        raise PublicationDataError(
            f"{game_id}: performance display reference is invalid"
        )
    payload = {
        "game_id": game_id,
        "week": value.get("week"),
        "date": value.get("date"),
        "season_type": value.get("season_type", ""),
        "conference_game": value.get("conference_game", False),
        "neutral_site": value.get("neutral_site", False),
        "state": value.get("state"),
        "home_team": _public_team(value["home_team"]),
        "away_team": _public_team(value["away_team"]),
        "home_team_id": str(value.get("home_team_id")),
        "away_team_id": str(value.get("away_team_id")),
        "score": value.get("score"),
        "winner_team_id": (
            str(value["winner_team_id"])
            if value.get("winner_team_id") is not None
            else None
        ),
        "home_performance": home_performance,
        "away_performance": away_performance,
        "future_prediction_id": prediction_key,
        "prediction": prediction,
    }
    try:
        game = CanonicalGame.model_validate(payload)
    except ValidationError as error:
        raise PublicationDataError(f"{game_id}: canonical game is malformed") from error
    if (
        game.home_team.team_id != game.home_team_id
        or game.away_team.team_id != game.away_team_id
    ):
        raise PublicationDataError(f"{game_id}: canonical game sides are inconsistent")
    if prediction is not None and prediction.game_id != game_id:
        raise PublicationDataError(f"{game_id}: prediction game ID is inconsistent")
    return game


def _load_team_games(
    artifact: dict[str, Any],
    metadata: PublicationMetadata,
    rankings: list[RankRow],
    games: dict[str, CanonicalGame],
    predictions: dict[str, FuturePrediction],
    performance_displays: dict[str, list[int]],
) -> dict[str, tuple[TeamSeasonGame, ...]]:
    ranking_by_team = {row.team_id: row for row in rankings}
    result: dict[str, tuple[TeamSeasonGame, ...]] = {}
    for team_id, team in artifact["teams"].items():
        ranking = ranking_by_team[team_id]
        converted: list[TeamSeasonGame] = []
        for raw_game in team["games"]:
            game_id = str(raw_game["game_id"])
            canonical = games.get(game_id)
            if canonical is None:
                raise PublicationDataError(
                    f"{metadata.snapshot_id}: team game is not canonical"
                )
            prediction_id = raw_game.get("future_prediction_id")
            prediction_key = str(prediction_id) if prediction_id is not None else None
            if prediction_key is not None and prediction_key not in predictions:
                raise PublicationDataError(
                    f"{metadata.snapshot_id}: team prediction is not canonical"
                )
            opponent_id = str(raw_game.get("opponent_id"))
            if canonical.home_team_id == team_id:
                opponent = canonical.away_team
            elif canonical.away_team_id == team_id:
                opponent = canonical.home_team
            else:
                raise PublicationDataError(
                    f"{metadata.snapshot_id}: team is absent from its game"
                )
            display = (
                raw_game.get("game_rating", {}).get("display_pmf")
                if isinstance(raw_game.get("game_rating"), dict)
                else None
            )
            try:
                team_game = TeamSeasonGame.model_validate(
                    {
                        "game_id": game_id,
                        "week": raw_game.get("week"),
                        "date": raw_game.get("date"),
                        "season_type": raw_game.get("season_type", ""),
                        "conference_game": raw_game.get("conference_game", False),
                        "opponent": opponent,
                        "site": raw_game.get("site"),
                        "state": canonical.state,
                        "result": raw_game.get("result"),
                        "score": raw_game.get("score"),
                        "modeled": raw_game.get("modeled", False),
                        "future_prediction_id": prediction_key,
                        "performance": _public_performance(
                            raw_game.get("game_rating"), display
                        ),
                        "prediction": predictions.get(prediction_key)
                        if prediction_key is not None
                        else None,
                    }
                )
            except ValidationError as error:
                raise PublicationDataError(
                    f"{metadata.snapshot_id}: team-season game is malformed"
                ) from error
            if team_game.opponent.team_id != opponent_id:
                raise PublicationDataError(
                    f"{metadata.snapshot_id}: opponent identity is inconsistent"
                )
            if team_game.state == "future" and team_game.score is not None:
                raise PublicationDataError(
                    f"{metadata.snapshot_id}: future team game exposes a score"
                )
            converted.append(team_game)
        result[team_id] = tuple(converted)
        if ranking.team_id != team_id:
            raise PublicationDataError(
                f"{metadata.snapshot_id}: team ranking identity is inconsistent"
            )
    return result


def _load_axes(artifact: dict[str, Any], snapshot_id: str) -> DisplayAxes:
    performance = artifact.get("performance_axis")
    future_margin = artifact.get("future_margin_axis")
    try:
        return DisplayAxes.model_validate(
            {
                "performance": PerformanceAxis.model_validate(performance)
                if performance is not None
                else None,
                "future_margin": FutureMarginAxis.model_validate(future_margin)
                if future_margin is not None
                else None,
            }
        )
    except ValidationError as error:
        raise PublicationDataError(
            f"{snapshot_id}: display axes are malformed"
        ) from error


def _load_simulation(
    artifact: dict[str, Any], metadata: PublicationMetadata, rankings: list[RankRow]
) -> tuple[SeasonSimulationInfo | None, dict[str, SeasonOutlook]]:
    simulation = artifact.get("season_simulation")
    if simulation is None:
        return None, {}
    if not isinstance(simulation, dict):
        raise PublicationDataError(
            f"{metadata.snapshot_id}: season simulation is malformed"
        )
    config = simulation.get("configuration")
    scope = simulation.get("season_scope")
    try:
        configuration = SimulationConfiguration.model_validate(
            {field: config.get(field) for field in SimulationConfiguration.model_fields}
        )
        scope_payload = {
            field: scope.get(field)
            for field in (
                "season_type",
                "future_game_count",
                "unresolved_game_count",
                "forecast_scope_game_count",
                "supported_future_game_count",
                "future_game_ids",
                "unresolved_game_ids",
                "forecast_scope_game_ids",
                "supported_future_game_ids",
                "unsupported_team_ids",
            )
        }
        unsupported_scope = scope.get("unsupported_future_games", [])
        if not isinstance(unsupported_scope, list) or any(
            not isinstance(item, dict) for item in unsupported_scope
        ):
            raise PublicationDataError(
                f"{metadata.snapshot_id}: unsupported season games are malformed"
            )
        scope_payload["unsupported_future_games"] = [
            UnsupportedSeasonGame.model_validate(
                {
                    "game_id": item.get("game_id"),
                    "reason": item.get("reason"),
                    "detail": item.get("detail"),
                }
            )
            for item in unsupported_scope
        ]
        scope_payload["forecast_status"] = simulation.get("forecast_status")
        season_scope = SeasonScope.model_validate(scope_payload)
        info = SeasonSimulationInfo.model_validate(
            {
                "simulation_version": simulation.get("simulation_version"),
                "prediction_source": simulation.get("prediction_source"),
                "configuration": configuration,
                "season_scope": season_scope,
            }
        )
    except (AttributeError, ValidationError) as error:
        raise PublicationDataError(
            f"{metadata.snapshot_id}: season simulation is malformed"
        ) from error
    raw_outlooks = simulation.get("teams")
    if not isinstance(raw_outlooks, dict):
        raise PublicationDataError(
            f"{metadata.snapshot_id}: season outlooks are missing"
        )
    ranking_ids = {row.team_id for row in rankings}
    if set(raw_outlooks) != ranking_ids:
        raise PublicationDataError(
            f"{metadata.snapshot_id}: season outlook team set is invalid"
        )
    outlooks: dict[str, SeasonOutlook] = {}
    for team_id, value in raw_outlooks.items():
        if not isinstance(value, dict):
            raise PublicationDataError(
                f"{metadata.snapshot_id}: season outlook is malformed"
            )
        unsupported = value.get("unsupported_games", [])
        if not isinstance(unsupported, list):
            raise PublicationDataError(
                f"{metadata.snapshot_id}: unsupported season games are malformed"
            )
        try:
            unsupported_values = [
                UnsupportedSeasonGame.model_validate(
                    {
                        "game_id": item.get("game_id"),
                        "reason": item.get("reason"),
                        "detail": item.get("detail"),
                    }
                )
                for item in unsupported
                if isinstance(item, dict)
            ]
            fields = (
                "team_id",
                "team_name",
                "subdivision",
                "forecast_status",
                "completed_wins",
                "completed_losses",
                "completed_ties",
                "completed_regular_season_games",
                "remaining_games",
                "forecast_scope_games",
                "expected_final_wins",
                "expected_remaining_wins",
                "median_final_wins",
                "median_remaining_wins",
                "final_win_interval_50",
                "final_win_interval_80",
                "final_win_interval_95",
                "remaining_win_interval_50",
                "remaining_win_interval_80",
                "remaining_win_interval_95",
                "final_win_distribution",
                "remaining_win_distribution",
                "record_probabilities",
                "threshold_probabilities",
                "conditional_expected_wins",
            )
            payload = {field: value.get(field) for field in fields}
            payload["variance_decomposition"] = (
                VarianceDecomposition.model_validate(
                    value.get("variance_decomposition")
                )
                if value.get("variance_decomposition") is not None
                else None
            )
            payload["simulated_remaining_games"] = value.get(
                "simulated_remaining_games"
            )
            payload["conditional_expected_wins"] = (
                ExpectedWinsSummary.model_validate(
                    value.get("conditional_expected_wins")
                )
                if value.get("conditional_expected_wins") is not None
                else None
            )
            payload["unsupported_games"] = unsupported_values
            outlook = SeasonOutlook.model_validate(payload)
        except (AttributeError, ValidationError) as error:
            raise PublicationDataError(
                f"{metadata.snapshot_id}: season outlook is malformed"
            ) from error
        if outlook.team_id != team_id:
            raise PublicationDataError(
                f"{metadata.snapshot_id}: season outlook identity is invalid"
            )
        if outlook.forecast_status == "available" and (
            outlook.expected_final_wins is None
            or outlook.final_win_distribution is None
            or outlook.record_probabilities is None
            or outlook.variance_decomposition is None
        ):
            raise PublicationDataError(
                f"{metadata.snapshot_id}: available season outlook is incomplete"
            )
        outlooks[team_id] = outlook
    return info, outlooks


def _parse_datetime(value: object) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
