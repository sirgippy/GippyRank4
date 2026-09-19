"""Production-safe preseason transfer snapshots and feature derivation.

The research transfer modules intentionally operate on retrospective CFBD
responses.  This module is the boundary for production inputs: acquisition
stores response bytes and a manifest, while derivation reads only the
manifest's canonical snapshots.  It never makes a network request.

The model-facing contract is deliberately small and frozen::

    transfer_in_prior_usage_sum
    transfer_in_prior_defensive_impact_db_sum
    transfer_in_prior_defensive_impact_db_available

All other values produced here are audit or provenance data.  In particular,
an unresolved incoming offensive transfer is not silently converted into a
resolved usage value, and an unresolved DB transfer neutralizes the DB sum and
sets its availability indicator to zero.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from math import isfinite
from pathlib import Path
from typing import Any

from gippyrank.defensive_transfer import (
    DefensiveGamePlayer,
    RosterPlayer,
    add_defensive_impact,
    aggregate_player_seasons,
    parse_games_players_payload,
    parse_roster_payload,
    team_game_keys,
)
from gippyrank.defensive_transfer import (
    audit_transfer_records as audit_defensive_transfers,
)
from gippyrank.transfer_audit import (
    D5_APPLICABLE,
    D5_CATEGORY_FAILURE,
    D5_CATEGORY_RESOLVED,
    D5_CATEGORY_UNDETERMINED,
    D5_CATEGORY_ZERO,
    ParticipationRecord,
    parse_participation_payload,
    read_team_aliases,
)
from gippyrank.transfer_audit import (
    audit_transfer_records as audit_offensive_transfers,
)
from gippyrank.transfer_oracle import (
    TransferRecord,
    UsageRecord,
    normalize_player_name,
    normalize_team_name,
    parse_transfer_payload,
    parse_usage_payload,
)

Row = dict[str, Any]

MANIFEST_VERSION = "issue-113-v1"
DEFAULT_CUTOFF_MONTH = 8
DEFAULT_CUTOFF_DAY = 15
CFBD_API = "https://api.collegefootballdata.com"

PORTAL_ENDPOINT = "/player/portal"
USAGE_ENDPOINT = "/player/usage"
PARTICIPATION_ENDPOINT = "/stats/player/season"
ROSTER_ENDPOINT = "/roster"
GAMES_PLAYERS_ENDPOINT = "/games/players"

SNAPSHOT_SOURCES = (
    "portal",
    "usage",
    "stats",
    "roster",
    "games_players",
)
REQUIRED_SNAPSHOT_SOURCES = frozenset(SNAPSHOT_SOURCES)

DB_POSITIONS = frozenset({"CB", "DB", "S", "FS", "SS", "NB"})
DB_COMPONENTS = ("tackles", "passes_defended", "interceptions")
DB_RESOLVED_STATUSES = frozenset(
    {"resolved", "zero_recorded_defensive_box_score_games"}
)

MODEL_FEATURE_COLUMNS = (
    "transfer_in_prior_usage_sum",
    "transfer_in_prior_defensive_impact_db_sum",
    "transfer_in_prior_defensive_impact_db_available",
)
CANONICAL_FEATURE_COLUMNS = (
    "season",
    "subdivision",
    "team_id",
    "team_name",
    *MODEL_FEATURE_COLUMNS,
)


class ManifestValidationError(ValueError):
    """Raised when a snapshot manifest or one of its raw inputs is unsafe."""


@dataclass(frozen=True)
class SnapshotSpec:
    """One request that belongs in a target-season preseason snapshot."""

    target_season: int
    source: str
    source_season: int
    endpoint: str
    query_parameters: Mapping[str, Any]

    @property
    def request_key(self) -> tuple[Any, ...]:
        return (
            self.target_season,
            self.source,
            self.source_season,
            json.dumps(dict(self.query_parameters), sort_keys=True),
        )


@dataclass(frozen=True)
class SnapshotRecord:
    """A manifest entry for one immutable raw response."""

    snapshot_id: str
    target_season: int
    source: str
    source_season: int
    path: str
    source_filename: str
    endpoint: str
    query_parameters: Mapping[str, Any]
    retrieval_timestamp: str
    target_cutoff: str
    captured_on_or_before_cutoff: bool
    sha256: str
    record_count: int
    canonical: bool = True
    version: str = "v1"

    @property
    def request_key(self) -> tuple[Any, ...]:
        return (
            self.target_season,
            self.source,
            self.source_season,
            json.dumps(dict(self.query_parameters), sort_keys=True),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "target_season": self.target_season,
            "season": self.target_season,
            "source": self.source,
            "source_season": self.source_season,
            "path": self.path,
            "source_filename": self.source_filename,
            "endpoint": self.endpoint,
            "query_parameters": dict(self.query_parameters),
            "parameters": dict(self.query_parameters),
            "retrieval_timestamp": self.retrieval_timestamp,
            "retrieved_at": self.retrieval_timestamp,
            "target_cutoff": self.target_cutoff,
            "cutoff": self.target_cutoff,
            "captured_on_or_before_cutoff": self.captured_on_or_before_cutoff,
            "sha256": self.sha256,
            "record_count": self.record_count,
            "canonical": self.canonical,
            "version": self.version,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SnapshotRecord:
        source = str(value.get("source") or "").strip()
        if source not in SNAPSHOT_SOURCES:
            raise ManifestValidationError(f"unknown snapshot source: {source!r}")
        target_season = _required_int(
            value.get("target_season", value.get("season")), "target_season"
        )
        source_season = _optional_int(value.get("source_season"))
        if source_season is None:
            source_season = target_season if source == "portal" else target_season - 1
        path = str(value.get("path") or "").strip()
        if not path:
            raise ManifestValidationError("snapshot record has no path")
        endpoint = str(value.get("endpoint") or "").strip()
        if not endpoint:
            raise ManifestValidationError(f"snapshot has no endpoint: {path}")
        parameters = value.get("query_parameters", value.get("parameters", {}))
        if not isinstance(parameters, Mapping):
            raise ManifestValidationError(
                f"snapshot parameters are not an object: {path}"
            )
        retrieval = str(
            value.get("retrieval_timestamp", value.get("retrieved_at", ""))
        ).strip()
        if not retrieval:
            raise ManifestValidationError(
                f"snapshot has no retrieval timestamp: {path}"
            )
        cutoff = str(value.get("target_cutoff", value.get("cutoff", ""))).strip()
        if not cutoff:
            cutoff = cutoff_for(target_season)
        try:
            cutoff_date = date.fromisoformat(cutoff[:10])
        except ValueError as error:
            raise ManifestValidationError(
                f"snapshot cutoff is not an ISO date: {cutoff!r}"
            ) from error
        captured = value.get("captured_on_or_before_cutoff")
        computed_captured = retrieval_date(retrieval) <= cutoff_date
        if captured is None:
            captured = computed_captured
        if not isinstance(captured, bool):
            raise ManifestValidationError(
                f"snapshot cutoff flag is not boolean: {path}"
            )
        if captured != computed_captured:
            raise ManifestValidationError(
                f"snapshot cutoff flag disagrees with retrieval timestamp: {path}"
            )
        digest = str(value.get("sha256") or value.get("content_sha256") or "").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ManifestValidationError(f"snapshot has invalid SHA-256: {path}")
        record_count = _required_int(value.get("record_count"), "record_count")
        if record_count < 0:
            raise ManifestValidationError(f"snapshot has negative record count: {path}")
        source_filename = str(value.get("source_filename") or Path(path).name).strip()
        snapshot_id = str(value.get("snapshot_id") or "").strip()
        if not snapshot_id:
            snapshot_id = f"{target_season}:{source}:{path}"
        return cls(
            snapshot_id=snapshot_id,
            target_season=target_season,
            source=source,
            source_season=source_season,
            path=path,
            source_filename=source_filename,
            endpoint=endpoint,
            query_parameters=dict(parameters),
            retrieval_timestamp=retrieval,
            target_cutoff=cutoff_date.isoformat(),
            captured_on_or_before_cutoff=captured,
            sha256=digest.lower(),
            record_count=record_count,
            canonical=bool(value.get("canonical", True)),
            version=str(value.get("version") or "v1"),
        )


@dataclass(frozen=True)
class SnapshotManifest:
    """Validated manifest plus the path used to resolve relative snapshots."""

    path: Path
    raw_root: Path
    snapshots: tuple[SnapshotRecord, ...]
    cutoff_month: int = DEFAULT_CUTOFF_MONTH
    cutoff_day: int = DEFAULT_CUTOFF_DAY

    @property
    def target_seasons(self) -> tuple[int, ...]:
        return tuple(sorted({item.target_season for item in self.snapshots}))

    def for_target(self, season: int) -> tuple[SnapshotRecord, ...]:
        return tuple(
            item
            for item in self.snapshots
            if item.target_season == season and item.canonical
        )

    def payload_path(self, record: SnapshotRecord) -> Path:
        path = Path(record.path)
        if path.is_absolute():
            return path
        return self.raw_root / path


@dataclass(frozen=True)
class TeamResolution:
    """Deterministic canonical-team resolution with an explicit status."""

    season: int
    raw_name: str | None
    normalized_name: str
    team_id: str | None
    canonical_name: str | None
    subdivision: str | None
    match_method: str

    @property
    def matched(self) -> bool:
        return self.team_id is not None


class CanonicalTeamResolver:
    """Resolve source and destination names without fuzzy matching."""

    def __init__(
        self,
        team_rows: Iterable[Mapping[str, Any]],
        aliases: Mapping[str | tuple[int, str], str] | None = None,
    ) -> None:
        self.rows = [dict(row) for row in team_rows]
        self._index: defaultdict[tuple[int, str, str], list[tuple[str, str]]] = (
            defaultdict(list)
        )
        for row in self.rows:
            season = _required_int(row.get("season"), "team season")
            team_id = str(row.get("team_id") or "").strip()
            team_name = str(row.get("team_name") or "").strip()
            subdivision = str(row.get("subdivision") or "").strip().casefold()
            if not team_id or not team_name or not subdivision:
                continue
            key = (season, normalize_team_name(team_name), subdivision)
            value = (team_id, team_name)
            if value not in self._index[key]:
                self._index[key].append(value)
        self.aliases: dict[str | tuple[int, str], str] = {}
        for key, value in (aliases or {}).items():
            normalized_value = normalize_team_name(value)
            if not normalized_value:
                raise ValueError(f"team alias has an empty target: {key!r}")
            normalized_key: str | tuple[int, str]
            if isinstance(key, tuple):
                normalized_key = (int(key[0]), normalize_team_name(key[1]))
            else:
                normalized_key = normalize_team_name(key)
            previous = self.aliases.get(normalized_key)
            if previous is not None and previous != normalized_value:
                raise ValueError(f"conflicting team alias: {key!r}")
            self.aliases[normalized_key] = normalized_value

    def _candidate_names(
        self, season: int, raw_name: str | None
    ) -> list[tuple[str, str]]:
        normalized = normalize_team_name(raw_name)
        if not normalized:
            return []
        result = [(normalized, "exact_normalized_name")]
        alias = self.aliases.get((season, normalized), self.aliases.get(normalized))
        if alias and alias != normalized:
            result.append((alias, "explicit_alias"))
        return result

    def resolve(
        self,
        season: int,
        raw_name: str | None,
        *,
        subdivision: str | None = None,
    ) -> TeamResolution:
        normalized = normalize_team_name(raw_name)
        if not normalized:
            return TeamResolution(
                season, raw_name, normalized, None, None, None, "missing_name"
            )
        requested_subdivision = subdivision.casefold() if subdivision else None
        for candidate_name, method in self._candidate_names(season, raw_name):
            matches: list[tuple[str, str, str]] = []
            subdivisions = (
                (requested_subdivision,)
                if requested_subdivision
                else tuple(
                    sorted(
                        {
                            key[2]
                            for key in self._index
                            if key[0] == season and key[1] == candidate_name
                        }
                    )
                )
            )
            for candidate_subdivision in subdivisions:
                for team_id, canonical in self._index.get(
                    (season, candidate_name, candidate_subdivision), []
                ):
                    matches.append((team_id, canonical, candidate_subdivision))
            if len(matches) == 1:
                team_id, canonical, matched_subdivision = matches[0]
                return TeamResolution(
                    season,
                    raw_name,
                    normalized,
                    team_id,
                    canonical,
                    matched_subdivision,
                    method,
                )
            if len(matches) > 1:
                return TeamResolution(
                    season,
                    raw_name,
                    normalized,
                    None,
                    None,
                    None,
                    "ambiguous_team",
                )
        return TeamResolution(
            season,
            raw_name,
            normalized,
            None,
            None,
            None,
            "not_in_canonical_population",
        )

    def alias_target(self, season: int, raw_name: str | None) -> str | None:
        normalized = normalize_team_name(raw_name)
        return self.aliases.get((season, normalized), self.aliases.get(normalized))


class PlayerAliasResolver:
    """Resolve explicit player corrections with optional season/team scope."""

    def __init__(
        self,
        aliases: Mapping[Any, str] | Iterable[Mapping[str, Any]] | None = None,
    ) -> None:
        self._aliases: dict[tuple[int | None, str | None, str], str] = {}
        if isinstance(aliases, Mapping):
            for key, value in aliases.items():
                if isinstance(key, tuple):
                    if len(key) == 3:
                        season, source_team, alias = key
                    elif len(key) == 2:
                        season, alias = key
                        source_team = None
                    else:
                        raise ValueError(f"unsupported player alias key: {key!r}")
                    self.add(alias, value, season=season, source_team=source_team)
                else:
                    self.add(key, value)
        elif aliases is not None:
            for row in aliases:
                self.add(
                    row.get("alias") or row.get("source") or "",
                    row.get("canonical") or row.get("player_name") or "",
                    season=row.get("season"),
                    source_team=row.get("source_team") or row.get("team"),
                )

    def add(
        self,
        alias: Any,
        canonical: Any,
        *,
        season: Any = None,
        source_team: Any = None,
    ) -> None:
        alias_name = normalize_player_name(str(alias or ""))
        canonical_name = " ".join(str(canonical or "").strip().split())
        if not alias_name or not canonical_name:
            raise ValueError("player alias rows need alias and canonical")
        season_value = _optional_int(season)
        team_value = (
            normalize_team_name(str(source_team))
            if source_team not in (None, "")
            else None
        )
        key = (season_value, team_value, alias_name)
        normalized_canonical = normalize_player_name(canonical_name)
        previous = self._aliases.get(key)
        if previous is not None and previous != normalized_canonical:
            raise ValueError(f"conflicting player alias: {key!r}")
        self._aliases[key] = canonical_name

    def resolve(
        self,
        season: int,
        source_team: str | None,
        player_name: str,
    ) -> tuple[str, str]:
        normalized = normalize_player_name(player_name)
        team = normalize_team_name(source_team)
        for key, method in (
            ((season, team, normalized), "explicit_player_alias"),
            ((season, None, normalized), "explicit_player_alias"),
            ((None, team, normalized), "explicit_player_alias"),
            ((None, None, normalized), "explicit_player_alias"),
        ):
            value = self._aliases.get(key)
            if value is not None:
                return value, method
        return player_name, "none"


def read_player_aliases(path: Path | None) -> PlayerAliasResolver:
    """Read ``alias,canonical[,season][,source_team]`` corrections."""
    if path is None:
        return PlayerAliasResolver()
    with path.open(newline="", encoding="utf-8") as handle:
        return PlayerAliasResolver(csv.DictReader(handle))


def cutoff_for(
    season: int,
    month: int = DEFAULT_CUTOFF_MONTH,
    day: int = DEFAULT_CUTOFF_DAY,
) -> str:
    return date(season, month, day).isoformat()


def retrieval_date(value: str) -> date:
    """Parse an ISO timestamp and retain only its UTC calendar date."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return date.fromisoformat(value[:10])
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC)
    return parsed.date()


def _required_int(value: Any, label: str) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        raise ManifestValidationError(f"missing or invalid {label}: {value!r}")
    return parsed


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _resolve_raw_root(manifest_path: Path, value: Any) -> Path:
    candidates: list[Path] = []
    if value not in (None, ""):
        configured = Path(str(value))
        candidates.append(
            configured
            if configured.is_absolute()
            else manifest_path.parent / configured
        )
        candidates.append(configured)
    candidates.append(manifest_path.parent)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_snapshot_path(manifest: SnapshotManifest, record: SnapshotRecord) -> Path:
    path = manifest.payload_path(record)
    if path.exists():
        return path
    # A manifest written relative to the repository root may be consumed from
    # a copied checkout.  Keep this fallback explicit; never discover another
    # JSON file by globbing.
    if not Path(record.path).is_absolute():
        repository_relative = Path(record.path)
        for candidate in (
            manifest.path.parent / repository_relative,
            repository_relative,
        ):
            if candidate.exists():
                return candidate.resolve()
    return path


def load_snapshot_manifest(
    path: Path,
    *,
    required_seasons: Iterable[int] | None = None,
    verify_hashes: bool = True,
) -> SnapshotManifest:
    """Load and validate a manifest without searching for live/raw files."""
    manifest_path = path.resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ManifestValidationError(
            f"snapshot manifest does not exist: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ManifestValidationError(
            f"snapshot manifest is not valid JSON: {path}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ManifestValidationError("snapshot manifest must be a JSON object")
    records_payload = payload.get("snapshots")
    if not isinstance(records_payload, list):
        raise ManifestValidationError("snapshot manifest has no snapshots array")
    if not records_payload:
        raise ManifestValidationError("snapshot manifest contains no snapshots")
    records = tuple(SnapshotRecord.from_mapping(item) for item in records_payload)
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    for record in records:
        if record.path in seen_paths:
            raise ManifestValidationError(f"duplicate snapshot path: {record.path}")
        if record.snapshot_id in seen_ids:
            raise ManifestValidationError(
                f"duplicate snapshot ID: {record.snapshot_id}"
            )
        seen_paths.add(record.path)
        seen_ids.add(record.snapshot_id)
    cutoff_value = payload.get("cutoff", {})
    if isinstance(cutoff_value, Mapping):
        cutoff_month = _required_int(
            cutoff_value.get("month", DEFAULT_CUTOFF_MONTH), "cutoff month"
        )
        cutoff_day = _required_int(
            cutoff_value.get("day", DEFAULT_CUTOFF_DAY), "cutoff day"
        )
    else:
        cutoff_month, cutoff_day = DEFAULT_CUTOFF_MONTH, DEFAULT_CUTOFF_DAY
    if not 1 <= cutoff_month <= 12 or not 1 <= cutoff_day <= 31:
        raise ManifestValidationError("cutoff month/day are outside calendar bounds")
    for record in records:
        expected_cutoff = cutoff_for(record.target_season, cutoff_month, cutoff_day)
        if record.target_cutoff != expected_cutoff:
            raise ManifestValidationError(
                f"snapshot cutoff disagrees with manifest cutoff: {record.path}"
            )
        expected_captured = retrieval_date(
            record.retrieval_timestamp
        ) <= date.fromisoformat(expected_cutoff)
        if record.captured_on_or_before_cutoff != expected_captured:
            raise ManifestValidationError(
                f"snapshot cutoff flag is inconsistent with manifest cutoff: {record.path}"
            )
    manifest = SnapshotManifest(
        path=manifest_path,
        raw_root=_resolve_raw_root(manifest_path, payload.get("raw_root")),
        snapshots=records,
        cutoff_month=cutoff_month,
        cutoff_day=cutoff_day,
    )
    required_requests = payload.get("required_requests", [])
    if required_requests:
        if not isinstance(required_requests, list):
            raise ManifestValidationError("required_requests must be an array")
        canonical_keys = {record.request_key for record in records if record.canonical}
        for required_request in required_requests:
            if not isinstance(required_request, Mapping):
                raise ManifestValidationError(
                    "required snapshot request is not an object"
                )
            try:
                request = SnapshotSpec(
                    target_season=_required_int(
                        required_request.get("target_season"), "required target season"
                    ),
                    source=str(required_request["source"]),
                    source_season=_required_int(
                        required_request.get("source_season"), "required source season"
                    ),
                    endpoint=str(required_request["endpoint"]),
                    query_parameters=dict(required_request.get("query_parameters", {})),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ManifestValidationError(
                    f"invalid required snapshot request: {required_request!r}"
                ) from error
            if request.source not in SNAPSHOT_SOURCES:
                raise ManifestValidationError(
                    f"unknown required snapshot source: {request.source!r}"
                )
            if request.request_key not in canonical_keys:
                raise ManifestValidationError(
                    "manifest is missing required canonical request: "
                    f"{request.request_key}"
                )
    if verify_hashes:
        for record in records:
            snapshot_path = _resolve_snapshot_path(manifest, record)
            if not snapshot_path.exists():
                raise ManifestValidationError(
                    f"manifest snapshot is missing: {record.path}"
                )
            content = snapshot_path.read_bytes()
            actual_hash = _sha256_bytes(content)
            if actual_hash != record.sha256:
                raise ManifestValidationError(
                    f"snapshot hash mismatch for {record.path}: "
                    f"manifest={record.sha256} actual={actual_hash}"
                )
            try:
                decoded = json.loads(content)
            except json.JSONDecodeError as error:
                raise ManifestValidationError(
                    f"snapshot is not valid JSON: {record.path}"
                ) from error
            if not isinstance(decoded, list):
                raise ManifestValidationError(
                    f"CFBD snapshot is not a JSON array: {record.path}"
                )
            if len(decoded) != record.record_count:
                raise ManifestValidationError(
                    f"record-count mismatch for {record.path}: "
                    f"manifest={record.record_count} actual={len(decoded)}"
                )
    required = set(required_seasons or manifest.target_seasons)
    for season in sorted(required):
        available = {item.source for item in manifest.for_target(season)}
        missing = REQUIRED_SNAPSHOT_SOURCES - available
        if missing:
            raise ManifestValidationError(
                f"target season {season} is missing required snapshots: "
                f"{', '.join(sorted(missing))}"
            )
    return manifest


def _snapshot_filename(
    spec: SnapshotSpec,
    *,
    version: str,
) -> str:
    params = dict(spec.query_parameters)
    if spec.source == "games_players":
        classification = str(params.get("classification", "all"))
        week = int(params.get("week", 0))
        stem = f"{spec.source_season}-{classification}-week-{week:02d}"
    elif spec.source == "portal":
        stem = f"{spec.target_season}"
    else:
        stem = f"{spec.source_season}"
        if "classification" in params:
            stem += f"-{params['classification']}"
    return f"{stem}-{version}.json"


def write_immutable_snapshot(
    *,
    raw_root: Path,
    spec: SnapshotSpec,
    content: bytes,
    retrieval_timestamp: str,
    cutoff_month: int = DEFAULT_CUTOFF_MONTH,
    cutoff_day: int = DEFAULT_CUTOFF_DAY,
    version: str = "v1",
    filename: str | None = None,
) -> SnapshotRecord:
    """Write raw response bytes once and return their manifest record.

    Existing paths are never overwritten, even when the new content is
    identical.  A refresh must supply a new explicit version or filename.
    """
    if spec.source not in SNAPSHOT_SOURCES:
        raise ValueError(f"unknown snapshot source: {spec.source}")
    if not isinstance(content, bytes):
        raise TypeError("raw snapshot content must be response bytes")
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("raw snapshot content is not valid JSON") from error
    if not isinstance(decoded, list):
        raise TypeError("raw CFBD snapshot must be a JSON array")
    destination = (
        raw_root
        / spec.source
        / str(spec.target_season)
        / (filename or _snapshot_filename(spec, version=version))
    )
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable snapshot: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    sidecar = destination.with_name(f"{destination.name}.provenance.json")
    if sidecar.exists():
        raise FileExistsError(f"refusing to overwrite snapshot metadata: {sidecar}")
    cutoff = date(spec.target_season, cutoff_month, cutoff_day)
    captured = retrieval_date(retrieval_timestamp) <= cutoff
    digest = _sha256_bytes(content)
    record = SnapshotRecord(
        snapshot_id=f"{spec.target_season}:{spec.source}:{destination.relative_to(raw_root).as_posix()}",
        target_season=spec.target_season,
        source=spec.source,
        source_season=spec.source_season,
        path=destination.relative_to(raw_root).as_posix(),
        source_filename=destination.name,
        endpoint=spec.endpoint,
        query_parameters=dict(spec.query_parameters),
        retrieval_timestamp=retrieval_timestamp,
        target_cutoff=cutoff.isoformat(),
        captured_on_or_before_cutoff=captured,
        sha256=digest,
        record_count=len(decoded),
        canonical=True,
        version=version,
    )
    sidecar.write_text(
        json.dumps(
            {
                "source": "cfbd_api_raw_response",
                "endpoint": spec.endpoint,
                "query_parameters": dict(spec.query_parameters),
                "target_season": spec.target_season,
                "source_season": spec.source_season,
                "retrieval_timestamp": retrieval_timestamp,
                "target_cutoff": cutoff.isoformat(),
                "captured_on_or_before_cutoff": captured,
                "sha256": digest,
                "record_count": len(decoded),
                "source_filename": destination.name,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return record


def write_raw_snapshot(**kwargs: Any) -> SnapshotRecord:
    """Compatibility name for callers that prefer the raw-data terminology."""
    return write_immutable_snapshot(**kwargs)


def _canonical_key(record: SnapshotRecord) -> tuple[Any, ...]:
    return record.request_key


def write_snapshot_manifest(
    path: Path,
    records: Iterable[SnapshotRecord],
    *,
    raw_root: Path | None = None,
    cutoff_month: int = DEFAULT_CUTOFF_MONTH,
    cutoff_day: int = DEFAULT_CUTOFF_DAY,
    preserve_existing: bool = True,
    required_specs: Iterable[SnapshotSpec] | None = None,
    allow_late_canonical: bool = False,
) -> Path:
    """Persist a manifest while retaining superseded raw snapshots.

    Late snapshots are deliberately retained for diagnosis and provenance but
    can never become canonical by default.  The explicit 2026 retrospective
    activation may opt into a separate late-canonical manifest; production
    callers must leave this false.
    """
    existing: list[SnapshotRecord] = []
    existing_required_requests: list[dict[str, Any]] = []
    if preserve_existing and path.exists():
        existing_payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(existing_payload, Mapping):
            if isinstance(existing_payload.get("required_requests"), list):
                existing_required_requests = list(existing_payload["required_requests"])
            existing = [
                SnapshotRecord.from_mapping(item)
                for item in existing_payload.get("snapshots", [])
            ]
    combined: dict[str, SnapshotRecord] = {item.snapshot_id: item for item in existing}
    for record in records:
        combined[record.snapshot_id] = record
    values = list(combined.values())
    newest_by_request: dict[tuple[Any, ...], str] = {}
    for record in sorted(
        values,
        key=lambda item: (
            item.target_season,
            item.source,
            item.retrieval_timestamp,
            item.path,
        ),
    ):
        if record.canonical and (
            record.captured_on_or_before_cutoff or allow_late_canonical
        ):
            newest_by_request[_canonical_key(record)] = record.snapshot_id
    normalized: list[SnapshotRecord] = []
    for record in values:
        selected = newest_by_request.get(_canonical_key(record))
        normalized.append(replace(record, canonical=selected == record.snapshot_id))
    normalized.sort(key=lambda item: (item.target_season, item.source, item.path))
    root = raw_root or path.parent
    try:
        root_value = root.resolve().relative_to(path.parent.resolve()).as_posix()
        root_value = root_value or "."
    except ValueError:
        root_value = str(root.resolve())
    target_seasons = sorted({item.target_season for item in normalized})
    required_requests = existing_required_requests
    if required_specs is not None:
        request_values = {
            json.dumps(request, sort_keys=True): request
            for request in existing_required_requests
        }
        for spec in required_specs:
            request = {
                "target_season": spec.target_season,
                "source": spec.source,
                "source_season": spec.source_season,
                "endpoint": spec.endpoint,
                "query_parameters": dict(spec.query_parameters),
            }
            request_values[json.dumps(request, sort_keys=True)] = request
        required_requests = list(request_values.values())
        required_requests.sort(
            key=lambda request: (
                int(request["target_season"]),
                str(request["source"]),
                json.dumps(request.get("query_parameters", {}), sort_keys=True),
            )
        )
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "raw_root": root_value,
        "cutoff": {"month": cutoff_month, "day": cutoff_day},
        "target_seasons": target_seasons,
        "raw_payloads_unchanged": True,
        "source": "CFBD",
        "required_requests": required_requests,
        "snapshots": [item.as_dict() for item in normalized],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def required_snapshot_specs(
    target_season: int,
    *,
    first_week: int = 1,
    last_week: int = 16,
    classifications: Sequence[str] = ("fbs", "fcs"),
) -> list[SnapshotSpec]:
    """Return the complete CFBD request set for one target season."""
    if first_week > last_week:
        raise ValueError("first_week must not exceed last_week")
    source_season = target_season - 1
    specs = [
        SnapshotSpec(
            target_season,
            "portal",
            target_season,
            PORTAL_ENDPOINT,
            {"year": target_season},
        ),
        SnapshotSpec(
            target_season,
            "usage",
            source_season,
            USAGE_ENDPOINT,
            {"year": source_season},
        ),
        SnapshotSpec(
            target_season,
            "stats",
            source_season,
            PARTICIPATION_ENDPOINT,
            {"year": source_season},
        ),
    ]
    for classification in classifications:
        specs.append(
            SnapshotSpec(
                target_season,
                "roster",
                source_season,
                ROSTER_ENDPOINT,
                {"year": source_season, "classification": classification},
            )
        )
        for week in range(first_week, last_week + 1):
            specs.append(
                SnapshotSpec(
                    target_season,
                    "games_players",
                    source_season,
                    GAMES_PLAYERS_ENDPOINT,
                    {
                        "year": source_season,
                        "week": week,
                        "classification": classification,
                        "seasonType": "both",
                    },
                )
            )
    return specs


def _payload(
    manifest: SnapshotManifest, record: SnapshotRecord
) -> list[Mapping[str, Any]]:
    path = _resolve_snapshot_path(manifest, record)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ManifestValidationError(f"snapshot is missing: {record.path}") from error
    if not isinstance(value, list):
        raise ManifestValidationError(f"snapshot is not a JSON array: {record.path}")
    return value


def _single_snapshot(
    manifest: SnapshotManifest,
    target_season: int,
    source: str,
) -> SnapshotRecord:
    records = [
        item for item in manifest.for_target(target_season) if item.source == source
    ]
    if len(records) != 1:
        raise ManifestValidationError(
            f"target season {target_season} needs exactly one canonical {source} snapshot; "
            f"found {len(records)}"
        )
    return records[0]


def _source_snapshots(
    manifest: SnapshotManifest, target_season: int, source: str
) -> tuple[SnapshotRecord, ...]:
    return tuple(
        sorted(
            (
                item
                for item in manifest.for_target(target_season)
                if item.source == source
            ),
            key=lambda item: (dict(item.query_parameters).get("week", 0), item.path),
        )
    )


def _raw_alias_target(
    aliases: Mapping[str | tuple[int, str], str], season: int, name: str | None
) -> str | None:
    normalized = normalize_team_name(name)
    return aliases.get((season, normalized), aliases.get(normalized))


def _canonical_team_text(
    resolver: CanonicalTeamResolver,
    aliases: Mapping[str | tuple[int, str], str],
    season: int,
    value: str | None,
    *,
    subdivision: str | None = None,
) -> str | None:
    resolution = resolver.resolve(season, value, subdivision=subdivision)
    if resolution.canonical_name:
        return resolution.canonical_name
    alias = _raw_alias_target(aliases, season, value)
    if alias:
        return alias
    return value


def _canonical_transfer_records(
    records: Sequence[TransferRecord],
    *,
    target_season: int,
    resolver: CanonicalTeamResolver,
    team_aliases: Mapping[str | tuple[int, str], str],
    player_aliases: PlayerAliasResolver,
) -> tuple[list[TransferRecord], list[Row]]:
    result: list[TransferRecord] = []
    mapping_rows: list[Row] = []
    for index, record in enumerate(records):
        origin_resolution = resolver.resolve(target_season, record.origin)
        destination_resolution = resolver.resolve(
            target_season, record.destination, subdivision="fbs"
        )
        player_name, player_alias_method = player_aliases.resolve(
            target_season, record.origin, record.player_name
        )
        canonical = replace(
            record,
            player_name=player_name,
            origin=_canonical_team_text(
                resolver, team_aliases, target_season, record.origin
            ),
            destination=_canonical_team_text(
                resolver,
                team_aliases,
                target_season,
                record.destination,
                subdivision="fbs",
            ),
        )
        result.append(canonical)
        mapping_rows.append(
            {
                "portal_index": index,
                "player_name": record.player_name,
                "canonical_player_name": player_name,
                "player_alias_method": player_alias_method,
                "origin": record.origin,
                "origin_team_id": origin_resolution.team_id,
                "origin_team_name": origin_resolution.canonical_name,
                "origin_match_method": origin_resolution.match_method,
                "destination": record.destination,
                "destination_team_id": destination_resolution.team_id,
                "destination_team_name": destination_resolution.canonical_name,
                "destination_match_method": destination_resolution.match_method,
            }
        )
    return result, mapping_rows


def _canonical_usage_records(
    records: Sequence[UsageRecord],
    *,
    resolver: CanonicalTeamResolver,
    aliases: Mapping[str | tuple[int, str], str],
) -> list[UsageRecord]:
    return [
        replace(
            item,
            team=_canonical_team_text(resolver, aliases, item.season, item.team)
            or item.team,
        )
        for item in records
    ]


def _canonical_participation_records(
    records: Sequence[ParticipationRecord],
    *,
    resolver: CanonicalTeamResolver,
    aliases: Mapping[str | tuple[int, str], str],
) -> list[ParticipationRecord]:
    return [
        replace(
            item,
            team=_canonical_team_text(resolver, aliases, item.season, item.team)
            or item.team,
        )
        for item in records
    ]


def _canonical_roster_records(
    records: Sequence[RosterPlayer],
    *,
    target_season: int,
    resolver: CanonicalTeamResolver,
    team_aliases: Mapping[str | tuple[int, str], str],
    player_aliases: PlayerAliasResolver,
) -> list[RosterPlayer]:
    result: list[RosterPlayer] = []
    for item in records:
        team = (
            _canonical_team_text(resolver, team_aliases, item.season, item.team)
            or item.team
        )
        name, _ = player_aliases.resolve(target_season, item.team, item.player_name)
        result.append(replace(item, team=team, player_name=name))
    return result


def _canonical_game_records(
    records: Sequence[DefensiveGamePlayer],
    *,
    target_season: int,
    resolver: CanonicalTeamResolver,
    team_aliases: Mapping[str | tuple[int, str], str],
    player_aliases: PlayerAliasResolver,
) -> list[DefensiveGamePlayer]:
    result: list[DefensiveGamePlayer] = []
    for item in records:
        team = (
            _canonical_team_text(resolver, team_aliases, item.season, item.team)
            or item.team
        )
        name, _ = player_aliases.resolve(target_season, item.team, item.player_name)
        result.append(replace(item, team=team, player_name=name))
    return result


def _canonical_game_keys(
    payloads: Iterable[Sequence[Mapping[str, Any]]],
    *,
    source_season: int,
    resolver: CanonicalTeamResolver,
    team_aliases: Mapping[str | tuple[int, str], str],
) -> set[tuple[int, str, str]]:
    result: set[tuple[int, str, str]] = set()
    for payload in payloads:
        for season, team, game_id in team_game_keys(payload, season=source_season):
            canonical = (
                _canonical_team_text(resolver, team_aliases, season, team) or team
            )
            result.add((season, normalize_team_name(canonical), game_id))
    return result


def classify_db_position(position: str | None) -> str | None:
    """Return ``db`` only for the frozen DB labels; never infer hybrids."""
    normalized = str(position or "").strip().upper()
    return "db" if normalized in DB_POSITIONS else None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _offensive_feature_rows(
    *,
    target_season: int,
    team_rows: Sequence[Mapping[str, Any]],
    offensive_audit: Mapping[str, Any],
    portal_records: Sequence[TransferRecord],
    source_hashes: Mapping[str, Sequence[str]],
    mapping_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Row], dict[str, Any]]:
    join_rows = list(offensive_audit["join_rows"])
    mapping_by_index = {int(row["portal_index"]): row for row in mapping_rows}
    rows: list[Row] = []
    for team in team_rows:
        if (
            int(team["season"]) != target_season
            or str(team.get("subdivision", "")).casefold() != "fbs"
        ):
            continue
        team_id = str(team["team_id"])
        incoming = [
            row
            for row in join_rows
            if row["in_model_relevant_population"]
            and row["destination_team_id"] == team_id
        ]
        resolved_values = [
            float(row["d5_feature_value"])
            for row in incoming
            if row["d5_resolution_category"] in {D5_CATEGORY_RESOLVED, D5_CATEGORY_ZERO}
            and row["d5_feature_value"] is not None
        ]
        rows.append(
            {
                "season": target_season,
                "subdivision": "fbs",
                "team_id": team_id,
                "team_name": team.get("team_name"),
                "transfer_in_prior_usage_sum": float(sum(resolved_values)),
                "audit_incoming_fbs_transfers": len(incoming),
                "audit_offensive_applicable_transfers": sum(
                    row["d5_applicability"] == D5_APPLICABLE for row in incoming
                ),
                "audit_offensive_resolved_applicable": sum(
                    row["d5_resolution_category"] == D5_CATEGORY_RESOLVED
                    for row in incoming
                ),
                "audit_offensive_legitimate_zero_non_applicable": sum(
                    row["d5_resolution_category"] == D5_CATEGORY_ZERO
                    for row in incoming
                ),
                "audit_offensive_unresolved_applicable": sum(
                    row["d5_resolution_category"] == D5_CATEGORY_FAILURE
                    for row in incoming
                ),
                "audit_offensive_undetermined": sum(
                    row["d5_resolution_category"] == D5_CATEGORY_UNDETERMINED
                    for row in incoming
                ),
            }
        )
    season_row = next(
        (
            row
            for row in offensive_audit["season_rows"]
            if int(row["season"]) == target_season
        ),
        {},
    )
    identity_rows = [
        row
        for row in join_rows
        if row["in_model_relevant_population"] and int(row["season"]) == target_season
    ]
    quality = {
        "incoming_fbs_transfers": len(identity_rows),
        "applicable_offensive_transfers": sum(
            row["d5_applicability"] == D5_APPLICABLE for row in identity_rows
        ),
        "resolved_applicable_transfers": sum(
            row["d5_resolution_category"] == D5_CATEGORY_RESOLVED
            for row in identity_rows
        ),
        "legitimate_zero_non_applicable": sum(
            row["d5_resolution_category"] == D5_CATEGORY_ZERO for row in identity_rows
        ),
        "unresolved_applicable": sum(
            row["d5_resolution_category"] == D5_CATEGORY_FAILURE
            for row in identity_rows
        ),
        "undetermined_applicability": sum(
            row["d5_resolution_category"] == D5_CATEGORY_UNDETERMINED
            for row in identity_rows
        ),
        "usage_weighted_identity_resolution_proxy": season_row.get(
            "usage_weighted_join_coverage_proxy"
        ),
        "exact_matches": sum(
            row["usage_join_status"] == "joined"
            and row["usage_join_method"] == "normalized_name_and_source_team"
            for row in identity_rows
        ),
        "alias_matches": sum(
            row["usage_join_status"] == "joined"
            and row["usage_join_method"] == "explicit_team_alias"
            for row in identity_rows
        ),
        "ambiguous_matches": sum(
            row["usage_join_status"] == "ambiguous_usage_join" for row in identity_rows
        ),
        "unresolved_matches": sum(
            row["usage_join_status"] != "joined" for row in identity_rows
        ),
        "source_team_mismatches": sum(
            row["usage_join_status"] == "source_team_mismatch" for row in identity_rows
        ),
        "normalization_changed_matches": sum(
            bool(row["normalization_changed_match"]) for row in identity_rows
        ),
        "team_name_mapping_failures": sum(
            row.get("destination_match_method")
            not in {"exact_normalized_name", "explicit_alias"}
            for row in mapping_rows
            if row.get("destination")
        ),
        "source_team_mapping_failures": sum(
            row.get("origin_match_method")
            not in {"exact_normalized_name", "explicit_alias"}
            for row in mapping_rows
            if row.get("origin")
        ),
    }
    provenance: dict[str, Any] = {}
    for row in rows:
        key_prefix = f"{target_season}|{row['team_id']}"
        incoming = [
            item
            for item in join_rows
            if item["in_model_relevant_population"]
            and item["destination_team_id"] == row["team_id"]
        ]
        contributors = []
        for item in incoming:
            if item["d5_resolution_category"] not in {
                D5_CATEGORY_RESOLVED,
                D5_CATEGORY_ZERO,
            }:
                continue
            portal_index = int(item["portal_index"])
            raw = mapping_by_index.get(portal_index, {})
            contributors.append(
                {
                    "portal_index": portal_index,
                    "player_name": portal_records[portal_index].player_name,
                    "source_team": portal_records[portal_index].origin,
                    "feature_value": item["d5_feature_value"],
                    "resolution_category": item["d5_resolution_category"],
                    "usage_join_method": item["usage_join_method"],
                    "player_alias_method": raw.get("player_alias_method", "none"),
                }
            )
        entry = {
            "season": target_season,
            "destination_team_id": row["team_id"],
            "destination_team_name": row["team_name"],
            "feature": "transfer_in_prior_usage_sum",
            "value": row["transfer_in_prior_usage_sum"],
            "source_snapshot_sha256": sorted(source_hashes.get("offense", ())),
            "contributors": contributors,
            "unresolved_records": [
                {
                    "portal_index": int(item["portal_index"]),
                    "player_name": portal_records[
                        int(item["portal_index"])
                    ].player_name,
                    "reason": item["d5_applicability_reason"],
                    "resolution_category": item["d5_resolution_category"],
                }
                for item in incoming
                if item["d5_resolution_category"]
                in {D5_CATEGORY_FAILURE, D5_CATEGORY_UNDETERMINED}
            ],
        }
        provenance[key_prefix + "|transfer_in_prior_usage_sum"] = entry
    return rows, {"quality": quality, "provenance": provenance}


def _db_feature_rows(
    *,
    target_season: int,
    team_rows: Sequence[Mapping[str, Any]],
    defensive_audit: Mapping[str, Any],
    portal_records: Sequence[TransferRecord],
    source_hashes: Mapping[str, Sequence[str]],
) -> tuple[dict[tuple[int, str], Row], dict[str, Any], dict[str, Any]]:
    player_rows = [
        row
        for row in defensive_audit["player_rows"]
        if int(row["season"]) == target_season
    ]
    by_team: dict[tuple[int, str], Row] = {}
    provenance: dict[str, Any] = {}
    for team in team_rows:
        if (
            int(team["season"]) != target_season
            or str(team.get("subdivision", "")).casefold() != "fbs"
        ):
            continue
        team_id = str(team["team_id"])
        incoming = [
            row
            for row in player_rows
            if row["in_model_relevant_population"]
            and row["destination_team_id"] == team_id
            and classify_db_position(row.get("position")) == "db"
        ]
        resolved = [
            row
            for row in incoming
            if row["impact_status"] in DB_RESOLVED_STATUSES
            and row["prior_defensive_impact"] is not None
        ]
        unresolved = [row for row in incoming if row not in resolved]
        if not incoming:
            impact = 0.0
            available = 1
        elif unresolved:
            # Neutral imputation is explicit and cannot be confused with a
            # natural zero because the availability flag is zero.
            impact = 0.0
            available = 0
        else:
            impact = float(
                sum(float(row["prior_defensive_impact"]) for row in resolved)
            )
            available = 1
        by_team[(target_season, team_id)] = {
            "transfer_in_prior_defensive_impact_db_sum": impact,
            "transfer_in_prior_defensive_impact_db_available": available,
            "audit_incoming_db_transfers": len(incoming),
            "audit_resolved_db_transfers": len(resolved),
            "audit_unresolved_db_transfers": len(unresolved),
            "audit_db_feature_status": (
                "no_incoming_db_transfer"
                if not incoming
                else "complete"
                if available
                else "unavailable_unresolved_db_transfer"
            ),
        }
        contributors = [
            {
                "portal_index": int(row["portal_index"]),
                "player_name": portal_records[int(row["portal_index"])].player_name,
                "source_team": portal_records[int(row["portal_index"])].origin,
                "prior_player_id": row.get("prior_player_id"),
                "prior_position": row.get("prior_position"),
                "prior_stats": dict(row.get("prior_stats") or {}),
                "feature_value": row.get("prior_defensive_impact"),
                "impact_status": row.get("impact_status"),
                "included": row in resolved,
            }
            for row in incoming
        ]
        key_prefix = f"{target_season}|{team_id}"
        provenance[key_prefix + "|transfer_in_prior_defensive_impact_db_sum"] = {
            "season": target_season,
            "destination_team_id": team_id,
            "destination_team_name": team.get("team_name"),
            "feature": "transfer_in_prior_defensive_impact_db_sum",
            "value": impact,
            "source_snapshot_sha256": sorted(
                (*source_hashes.get("portal", ()), *source_hashes.get("defense", ()))
            ),
            "contributors": contributors,
        }
        provenance[key_prefix + "|transfer_in_prior_defensive_impact_db_available"] = {
            "season": target_season,
            "destination_team_id": team_id,
            "destination_team_name": team.get("team_name"),
            "feature": "transfer_in_prior_defensive_impact_db_available",
            "value": available,
            "source_snapshot_sha256": sorted(
                (*source_hashes.get("portal", ()), *source_hashes.get("defense", ()))
            ),
            "contributors": contributors,
        }
    quality = {
        "incoming_db_transfers": sum(
            row["audit_incoming_db_transfers"] for row in by_team.values()
        ),
        "resolved_db_transfers": sum(
            row["audit_resolved_db_transfers"] for row in by_team.values()
        ),
        "unresolved_db_transfers": sum(
            row["audit_unresolved_db_transfers"] for row in by_team.values()
        ),
        "teams_with_observed_db_feature": sum(
            row["audit_incoming_db_transfers"] > 0
            and row["transfer_in_prior_defensive_impact_db_available"] == 1
            for row in by_team.values()
        ),
        "teams_with_no_incoming_db_transfers": sum(
            row["audit_incoming_db_transfers"] == 0 for row in by_team.values()
        ),
        "teams_with_unavailable_db_feature": sum(
            row["transfer_in_prior_defensive_impact_db_available"] == 0
            for row in by_team.values()
        ),
        "unknown_position_records": sum(
            row.get("position_semantics") == "unknown"
            for row in player_rows
            if row.get("in_model_relevant_population")
        ),
    }
    return by_team, quality, provenance


def _load_target_inputs(
    manifest: SnapshotManifest,
    target_season: int,
    *,
    team_rows: Sequence[Mapping[str, Any]],
    team_aliases: Mapping[str | tuple[int, str], str],
    player_aliases: PlayerAliasResolver,
    allow_late_snapshots: bool = False,
) -> dict[str, Any]:
    records = manifest.for_target(target_season)
    if not allow_late_snapshots and any(
        not item.captured_on_or_before_cutoff for item in records
    ):
        late = [item.path for item in records if not item.captured_on_or_before_cutoff]
        raise ManifestValidationError(
            f"target season {target_season} has snapshots captured after its "
            f"preseason cutoff: {', '.join(late)}"
        )
    portal_snapshot = _single_snapshot(manifest, target_season, "portal")
    usage_snapshot = _single_snapshot(manifest, target_season, "usage")
    stats_snapshot = _single_snapshot(manifest, target_season, "stats")
    roster_snapshots = _source_snapshots(manifest, target_season, "roster")
    game_snapshots = _source_snapshots(manifest, target_season, "games_players")
    resolver = CanonicalTeamResolver(team_rows, team_aliases)
    portal_payload = _payload(manifest, portal_snapshot)
    usage_payload = _payload(manifest, usage_snapshot)
    stats_payload = _payload(manifest, stats_snapshot)
    records_raw = parse_transfer_payload(portal_payload, season=target_season)
    portal_keys: defaultdict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, record in enumerate(records_raw):
        portal_keys[
            (
                record.season,
                normalize_team_name(record.origin),
                normalize_player_name(record.player_name),
                normalize_team_name(record.destination),
                record.transfer_date,
            )
        ].append(index)
    duplicate_portal_records = {
        key: indexes for key, indexes in portal_keys.items() if len(indexes) > 1
    }
    if duplicate_portal_records:
        raise ManifestValidationError(
            f"duplicate portal records create ambiguous ownership for target season "
            f"{target_season}: {list(duplicate_portal_records.values())}"
        )
    usage_raw = parse_usage_payload(usage_payload, season=target_season - 1)
    participation_raw = parse_participation_payload(
        stats_payload, season=target_season - 1
    )
    rosters_raw: list[RosterPlayer] = []
    game_players_raw: list[DefensiveGamePlayer] = []
    game_payloads: list[Sequence[Mapping[str, Any]]] = []
    for snapshot in roster_snapshots:
        rosters_raw.extend(
            parse_roster_payload(
                _payload(manifest, snapshot), season=snapshot.source_season
            )
        )
    for snapshot in game_snapshots:
        payload = _payload(manifest, snapshot)
        game_payloads.append(payload)
        game_players_raw.extend(
            parse_games_players_payload(payload, season=snapshot.source_season)
        )
    records, mapping_rows = _canonical_transfer_records(
        records_raw,
        target_season=target_season,
        resolver=resolver,
        team_aliases=team_aliases,
        player_aliases=player_aliases,
    )
    usage = _canonical_usage_records(usage_raw, resolver=resolver, aliases=team_aliases)
    participation = _canonical_participation_records(
        participation_raw, resolver=resolver, aliases=team_aliases
    )
    rosters = _canonical_roster_records(
        rosters_raw,
        target_season=target_season,
        resolver=resolver,
        team_aliases=team_aliases,
        player_aliases=player_aliases,
    )
    game_players = _canonical_game_records(
        game_players_raw,
        target_season=target_season,
        resolver=resolver,
        team_aliases=team_aliases,
        player_aliases=player_aliases,
    )
    game_keys = _canonical_game_keys(
        game_payloads,
        source_season=target_season - 1,
        resolver=resolver,
        team_aliases=team_aliases,
    )
    player_seasons = add_defensive_impact(
        aggregate_player_seasons(game_players, rosters, game_keys)
    )
    return {
        "records_raw": records_raw,
        "records": records,
        "mapping_rows": mapping_rows,
        "usage": usage,
        "participation": participation,
        "rosters": rosters,
        "game_players": game_players,
        "game_keys": game_keys,
        "player_seasons": player_seasons,
        "portal_seasons": {target_season},
        "defensive_seasons": {target_season - 1},
        "roster_teams": {(item.season, item.normalized_team) for item in rosters},
        "source_team_coverage": {(season, team) for season, team, _game in game_keys},
        "source_hashes": {
            "offense": [
                portal_snapshot.sha256,
                usage_snapshot.sha256,
                stats_snapshot.sha256,
            ],
            "defense": [item.sha256 for item in (*roster_snapshots, *game_snapshots)],
            "portal": [portal_snapshot.sha256],
        },
    }


def derive_preseason_transfer_features(
    manifest_path: Path,
    team_rows: Iterable[Mapping[str, Any]],
    *,
    team_aliases: Mapping[str | tuple[int, str], str] | None = None,
    player_aliases: PlayerAliasResolver
    | Mapping[Any, str]
    | Iterable[Mapping[str, Any]]
    | None = None,
    required_seasons: Iterable[int] | None = None,
    allow_late_snapshots: bool = False,
) -> dict[str, Any]:
    """Derive canonical features entirely from validated immutable snapshots.

    ``allow_late_snapshots`` is reserved for the explicitly labelled 2026
    retrospective reconstruction.  The default production path remains
    fail-closed at the preseason cutoff.
    """
    teams = [dict(row) for row in team_rows]
    aliases = dict(team_aliases or {})
    player_alias_resolver = (
        player_aliases
        if isinstance(player_aliases, PlayerAliasResolver)
        else PlayerAliasResolver(player_aliases)
    )
    requested_seasons = (
        tuple(sorted(set(required_seasons))) if required_seasons is not None else None
    )
    manifest = load_snapshot_manifest(
        manifest_path,
        required_seasons=requested_seasons,
        verify_hashes=True,
    )
    target_seasons = requested_seasons or manifest.target_seasons
    if allow_late_snapshots and set(target_seasons) != {2026}:
        raise ManifestValidationError(
            "late transfer snapshots are permitted only for the explicit 2026 reconstruction"
        )
    all_feature_rows: list[Row] = []
    all_audit_rows: list[Row] = []
    all_mapping_rows: list[Row] = []
    all_player_rows: list[Row] = []
    quality_by_season: list[Row] = []
    provenance: dict[str, Any] = {}
    for season in target_seasons:
        target_teams = [
            row
            for row in teams
            if int(row["season"]) == season
            and str(row.get("subdivision", "")).casefold() == "fbs"
        ]
        if not target_teams:
            raise ManifestValidationError(
                f"canonical FBS team population is empty for target season {season}"
            )
        inputs = _load_target_inputs(
            manifest,
            season,
            team_rows=teams,
            team_aliases=aliases,
            player_aliases=player_alias_resolver,
            allow_late_snapshots=allow_late_snapshots,
        )
        offensive = audit_offensive_transfers(
            inputs["records"],
            inputs["usage"],
            teams,
            cutoff=date(season, manifest.cutoff_month, manifest.cutoff_day),
            aliases={},
            participation=inputs["participation"],
        )
        offensive_rows, offensive_meta = _offensive_feature_rows(
            target_season=season,
            team_rows=target_teams,
            offensive_audit=offensive,
            portal_records=inputs["records_raw"],
            source_hashes=inputs["source_hashes"],
            mapping_rows=inputs["mapping_rows"],
        )
        defensive = audit_defensive_transfers(
            inputs["records"],
            inputs["rosters"],
            inputs["player_seasons"],
            teams,
            portal_seasons=inputs["portal_seasons"],
            defensive_seasons=inputs["defensive_seasons"],
            cutoff=date(season, manifest.cutoff_month, manifest.cutoff_day),
            aliases={},
            team_coverage=inputs["source_team_coverage"],
            roster_teams=inputs["roster_teams"],
        )
        db_by_team, db_quality, db_provenance = _db_feature_rows(
            target_season=season,
            team_rows=target_teams,
            defensive_audit=defensive,
            portal_records=inputs["records_raw"],
            source_hashes=inputs["source_hashes"],
        )
        for row in offensive_rows:
            db = db_by_team[(season, str(row["team_id"]))]
            row.update(db)
            all_feature_rows.append(
                {
                    key: row[key]
                    for key in (
                        *CANONICAL_FEATURE_COLUMNS,
                        *[key for key in row if key.startswith("audit_")],
                    )
                }
            )
            all_audit_rows.append(dict(row))
        all_mapping_rows.extend(
            {"target_season": season, **dict(row)} for row in inputs["mapping_rows"]
        )
        for row in defensive["player_rows"]:
            enriched = dict(row)
            portal_index = int(row["portal_index"])
            raw = inputs["records_raw"][portal_index]
            enriched.update(
                {
                    "raw_player_name": raw.player_name,
                    "raw_origin": raw.origin,
                    "raw_destination": raw.destination,
                }
            )
            all_player_rows.append(enriched)
        provenance.update(offensive_meta["provenance"])
        provenance.update(db_provenance)
        season_quality = {
            "season": season,
            **offensive_meta["quality"],
            **db_quality,
            "identity_alias_matches": sum(
                row.get("player_alias_method") != "none"
                for row in inputs["mapping_rows"]
            ),
            "identity_ambiguous_matches": sum(
                row.get("usage_join_status") == "ambiguous_usage_join"
                for row in offensive["join_rows"]
                if row.get("in_model_relevant_population")
            ),
            "identity_exact_matches": sum(
                row.get("identity_join_method")
                in {"normalized_name_source_team", "stable_player_id_source_team"}
                for row in defensive["player_rows"]
                if row.get("in_model_relevant_population")
            ),
            "identity_unresolved_matches": sum(
                row.get("identity_status")
                in {
                    "identity_resolution_failure",
                    "source_data_unavailable",
                    "ambiguous",
                    "position_mismatch",
                }
                for row in defensive["player_rows"]
                if row.get("in_model_relevant_population")
            ),
            "identity_source_team_mismatches": sum(
                row.get("identity_status") == "source_data_unavailable"
                for row in defensive["player_rows"]
                if row.get("in_model_relevant_population")
            ),
        }
        quality_by_season.append(season_quality)
    all_feature_rows.sort(key=lambda row: (int(row["season"]), str(row["team_id"])))
    all_audit_rows.sort(key=lambda row: (int(row["season"]), str(row["team_id"])))
    return {
        "features": all_feature_rows,
        "audit": all_audit_rows,
        "player_audit": all_player_rows,
        "identity_mapping": all_mapping_rows,
        "quality_report": {
            "manifest": str(manifest.path),
            "cutoff": {"month": manifest.cutoff_month, "day": manifest.cutoff_day},
            "seasons": quality_by_season,
            "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
            "fail_closed": True,
        },
        "provenance": provenance,
        "manifest": manifest,
    }


def merge_preseason_transfer_features(
    rows: Iterable[Mapping[str, Any]],
    feature_rows: Iterable[Mapping[str, Any]],
    *,
    require_all: bool = True,
) -> list[Row]:
    """Attach the three transfer fields at the preprocessing/model boundary.

    This is intentionally an attach-only integration point.  It does not fit
    Context 1.3 and it does not fetch or discover any source data.
    """
    features: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in feature_rows:
        key = (int(row["season"]), str(row["subdivision"]), str(row["team_id"]))
        if key in features:
            raise ValueError(f"duplicate transfer feature row: {key}")
        missing = [name for name in MODEL_FEATURE_COLUMNS if name not in row]
        if missing:
            raise ValueError(f"transfer feature row is missing columns: {missing}")
        features[key] = row
    result: list[Row] = []
    for row in rows:
        copied = dict(row)
        key = (int(row["season"]), str(row["subdivision"]), str(row["team_id"]))
        feature = features.get(key)
        if feature is None:
            if require_all:
                raise ValueError(f"no transfer features for team-season: {key}")
            result.append(copied)
            continue
        copied.update({name: feature[name] for name in MODEL_FEATURE_COLUMNS})
        result.append(copied)
    return result


def validate_research_parity(
    actual_rows: Iterable[Mapping[str, Any]],
    expected_rows: Iterable[Mapping[str, Any]],
    *,
    tolerance: float = 1e-9,
) -> list[Row]:
    """Compare production rows to a frozen research fixture exactly enough to audit."""
    actual = {
        (int(row["season"]), str(row["subdivision"]), str(row["team_id"])): row
        for row in actual_rows
    }
    expected = {
        (int(row["season"]), str(row["subdivision"]), str(row["team_id"])): row
        for row in expected_rows
    }
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            f"research parity keys differ: missing={missing} extra={extra}"
        )
    comparisons: list[Row] = []
    for key in sorted(expected):
        for feature in MODEL_FEATURE_COLUMNS:
            observed = actual[key].get(feature)
            reference = expected[key].get(feature)
            if feature.endswith("available"):
                passed = int(observed) == int(reference)
            else:
                if observed is None or reference is None:
                    passed = observed is reference
                else:
                    passed = abs(float(observed) - float(reference)) <= tolerance
            comparisons.append(
                {
                    "season": key[0],
                    "subdivision": key[1],
                    "team_id": key[2],
                    "feature": feature,
                    "actual": observed,
                    "expected": reference,
                    "absolute_error": (
                        abs(float(observed) - float(reference))
                        if observed is not None and reference is not None
                        else None
                    ),
                    "tolerance": tolerance,
                    "parity_passed": passed,
                }
            )
    failures = [row for row in comparisons if not row["parity_passed"]]
    if failures:
        raise ValueError(f"research feature parity failed: {failures[:3]}")
    return comparisons


def source_inventory() -> list[Row]:
    """Return the production source contract for reports and documentation."""
    return [
        {
            "source": "CFBD /player/portal",
            "kind": "portal",
            "required_fields": "season, firstName, lastName, origin, destination, position, transferDate",
            "cutoff_rule": "raw response must be captured on or before target August 15",
        },
        {
            "source": "CFBD /player/usage",
            "kind": "usage",
            "required_fields": "prior season, id/name, source team, usage.overall",
            "cutoff_rule": "captured with the target-season manifest; prior-season outcome data",
        },
        {
            "source": "CFBD /stats/player/season",
            "kind": "stats",
            "required_fields": "prior player participation and position-consistent offensive stats",
            "cutoff_rule": "captured with the target-season manifest",
        },
        {
            "source": "CFBD /roster",
            "kind": "roster",
            "required_fields": "prior player ID, name, source team, defensive position",
            "cutoff_rule": "captured with the target-season manifest",
        },
        {
            "source": "CFBD /games/players",
            "kind": "games_players",
            "required_fields": "prior tackles, passes defended, interceptions and team-game coverage",
            "cutoff_rule": "captured with the target-season manifest",
        },
    ]


__all__ = [
    "CANONICAL_FEATURE_COLUMNS",
    "DB_COMPONENTS",
    "DB_POSITIONS",
    "MANIFEST_VERSION",
    "MODEL_FEATURE_COLUMNS",
    "CanonicalTeamResolver",
    "ManifestValidationError",
    "PlayerAliasResolver",
    "SnapshotManifest",
    "SnapshotRecord",
    "SnapshotSpec",
    "classify_db_position",
    "cutoff_for",
    "derive_preseason_transfer_features",
    "load_snapshot_manifest",
    "merge_preseason_transfer_features",
    "read_player_aliases",
    "read_team_aliases",
    "required_snapshot_specs",
    "source_inventory",
    "validate_research_parity",
    "write_immutable_snapshot",
    "write_raw_snapshot",
    "write_snapshot_manifest",
]
