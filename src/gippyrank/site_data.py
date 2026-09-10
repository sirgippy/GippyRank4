"""Build compact, static website data from explicitly selected snapshot bundles.

This module is intentionally independent of posterior inference.  It reads
already-valid snapshot artifacts and produces the small summaries a browser
needs; it neither imports nor invokes model code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gippyrank.methodology import (
    HISTORICAL_LIKELIHOOD_VERSION,
    METHODOLOGY_SCHEMA_VERSION,
    SITE_SCHEMA_VERSION,
    SUPPORTED_ARTIFACT_MODEL_VERSIONS,
    SUPPORTED_ARTIFACT_SCHEMA_VERSIONS,
    TEAM_SEASON_SCHEMA_VERSION,
    WEEKLY_GAME_SCHEMA_VERSION,
    production_methodology_metadata,
)
from gippyrank.team_logos import TEAM_LOGO_URL_TEMPLATE, logo_url, team_logo_handle

PREDICTION_SOURCE_CONTEXT = "predictive_context"
PREDICTION_SOURCE_HISTORY = "predictive_history"
PERFORMANCE_DISPLAY_BINS = 40
FUTURE_MARGIN_DISPLAY_MIN = -40.0
FUTURE_MARGIN_DISPLAY_MAX = 40.0
FUTURE_MARGIN_DISPLAY_BINS = 40
DISPLAY_PROBABILITY_SCALE = 1000
SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS = SUPPORTED_ARTIFACT_SCHEMA_VERSIONS["snapshot"]
PMF_SUM_TOLERANCE = 1e-9
SUMMARY_TOLERANCE = 1e-8
RANKING_FAMILIES: dict[str, dict[str, str]] = {
    "predictive": {"label": "Predictive"},
    "performance": {"label": "Performance"},
}


class SiteDataValidationError(ValueError):
    """Raised when a snapshot is not safe to publish."""


@dataclass(frozen=True)
class PublishedSnapshot:
    source: Path
    display_label: str
    publication_slot: str
    publication_status: str
    publication_order: int


@dataclass(frozen=True)
class PublicationComparison:
    """The publication metadata needed to resolve an official baseline."""

    season: int
    ranking_family: str
    prior_family: str | None
    publication_slot: str
    publication_status: str
    publication_order: int
    snapshot_id: str
    display_label: str


@dataclass(frozen=True)
class PreparedSnapshot:
    """Validated source data held until publication comparisons are resolved."""

    selected: PublishedSnapshot
    metadata: dict[str, Any]
    snapshot_id: str
    rankings: list[dict[str, Any]]
    distribution: dict[str, Any]
    team_seasons: dict[str, Any]
    records: dict[str, str]
    weekly_games: dict[str, Any] | None = None


def build_fbs_conference_map(
    schedule_path: Path, *, seasons: set[int] | None = None
) -> dict[tuple[int, str], str]:
    """Build season-specific FBS conference metadata from all schedule rows.

    Conference labels are presentation metadata, so this deliberately reads
    the complete processed CFBD corpus rather than a snapshot's included-game
    subset.  Only FBS team sides contribute to the map; an FCS appearance with
    a reused or malformed ID cannot overwrite FBS metadata.  Missing labels
    are not converted to ``Independent`` because CFBD's explicit independent
    convention (``FBS Independents`` in this corpus) is evidence-bearing.
    """
    required = {
        "season",
        "homeId",
        "homeClassification",
        "homeConference",
        "awayId",
        "awayClassification",
        "awayConference",
    }
    try:
        handle = schedule_path.open(newline="", encoding="utf-8")
    except OSError as error:
        raise SiteDataValidationError(
            f"Cannot read CFBD schedule corpus {schedule_path}: {error}"
        ) from error

    observed: defaultdict[tuple[int, str], set[str]] = defaultdict(set)
    with handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SiteDataValidationError(
                f"{schedule_path}: schedule missing {sorted(missing)}"
            )
        for row in reader:
            season_value = (row.get("season") or "").strip()
            try:
                season = int(season_value)
            except ValueError as error:
                raise SiteDataValidationError(
                    f"{schedule_path}: invalid schedule season {season_value!r}"
                ) from error
            if seasons is not None and season not in seasons:
                continue
            for side in ("home", "away"):
                if (row.get(f"{side}Classification") or "").strip().casefold() != "fbs":
                    continue
                team_id = (row.get(f"{side}Id") or "").strip()
                if not team_id:
                    raise SiteDataValidationError(
                        f"{schedule_path}: blank {side} FBS team ID in season {season}"
                    )
                conference = (row.get(f"{side}Conference") or "").strip()
                if conference:
                    observed[(season, team_id)].add(conference)

    conflicts = [
        (season, team_id, sorted(conferences))
        for (season, team_id), conferences in observed.items()
        if len(conferences) > 1
    ]
    if conflicts:
        season, team_id, conferences = min(conflicts)
        raise SiteDataValidationError(
            f"Conflicting nonblank FBS conferences for team {team_id} in season "
            f"{season}: {conferences}"
        )
    return {key: next(iter(conferences)) for key, conferences in observed.items()}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SiteDataValidationError(f"Cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise SiteDataValidationError(f"{path} must contain a JSON object")
    return value


def publication_slot_metadata(config: dict[str, Any]) -> dict[str, tuple[str, int]]:
    """Validate and index the explicitly ordered publication-slot metadata."""
    slot_entries = config.get("publication_slots")
    if not isinstance(slot_entries, list):
        raise SiteDataValidationError(
            "Publish configuration needs a publication_slots list"
        )
    slots: dict[str, tuple[str, int]] = {}
    for order, entry in enumerate(slot_entries):
        if not isinstance(entry, dict):
            raise SiteDataValidationError("Each publication slot must be an object")
        slot = entry.get("id")
        status = entry.get("status")
        if not isinstance(slot, str) or not slot:
            raise SiteDataValidationError(
                "Each publication slot needs a non-empty id string"
            )
        if slot in slots:
            raise SiteDataValidationError(f"Duplicate publication slot metadata: {slot}")
        if status not in {"official", "temporary"}:
            raise SiteDataValidationError(
                f"{slot}: publication status must be official or temporary"
            )
        slots[slot] = (status, order)
    return slots


def load_publish_config(path: Path, root: Path) -> tuple[list[PublishedSnapshot], str | None]:
    """Load explicitly ordered snapshots and their slot-level status metadata.

    ``publication_slots`` is an ordered list.  Its order is the publication
    chronology used for comparison resolution; slot IDs are intentionally not
    parsed or sorted because they are presentation/configuration identifiers.
    """
    config = _read_json(path)
    if config.get("schema_version") != SITE_SCHEMA_VERSION:
        raise SiteDataValidationError("Unsupported publish configuration schema")
    slots = publication_slot_metadata(config)
    if not slots:
        raise SiteDataValidationError(
            "Publish configuration needs a non-empty publication_slots list"
        )
    snapshots = config.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise SiteDataValidationError("Publish configuration needs a non-empty snapshots list")
    selected: list[PublishedSnapshot] = []
    referenced_slots: set[str] = set()
    for entry in snapshots:
        if not isinstance(entry, dict):
            raise SiteDataValidationError("Each publish configuration entry must be an object")
        source = entry.get("source")
        label = entry.get("display_label")
        slot = entry.get("publication_slot")
        if not all(isinstance(value, str) and value for value in (source, label, slot)):
            raise SiteDataValidationError(
                "Each entry needs source, display_label, and publication_slot strings"
            )
        if slot not in slots:
            raise SiteDataValidationError(
                f"Snapshot {source} references a publication slot without status: {slot}"
            )
        source_path = root / source
        if not source_path.is_dir():
            raise SiteDataValidationError(f"Selected snapshot directory does not exist: {source}")
        status, order = slots[slot]
        referenced_slots.add(slot)
        selected.append(PublishedSnapshot(source_path, label, slot, status, order))
    if referenced_slots != set(slots):
        missing = sorted(set(slots) - referenced_slots)
        raise SiteDataValidationError(
            f"Publication slot metadata does not match configured snapshots: {missing}"
        )
    default_slot = config.get("default_publication_slot")
    if default_slot is not None:
        if not isinstance(default_slot, str) or not default_slot:
            raise SiteDataValidationError("default_publication_slot must be a non-empty string")
        if default_slot not in {item.publication_slot for item in selected}:
            raise SiteDataValidationError("default_publication_slot is not a published slot")
    return selected, default_slot


def _logo_url_template(path: Path) -> str:
    """Read and validate the single browser-visible logo URL configuration."""
    config = _read_json(path)
    team_logos = config.get("team_logos", {})
    if team_logos is None:
        team_logos = {}
    if not isinstance(team_logos, dict):
        raise SiteDataValidationError("team_logos configuration must be an object")
    template = team_logos.get("url_template", TEAM_LOGO_URL_TEMPLATE)
    if not isinstance(template, str):
        raise SiteDataValidationError("team_logos.url_template must be a string")
    try:
        logo_url("example", template)
    except ValueError as error:
        raise SiteDataValidationError(f"Invalid team logo URL template: {error}") from error
    return template


def _finite_number(value: str, field: str, snapshot_id: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise SiteDataValidationError(f"{snapshot_id}: {field} is not numeric") from error
    if not math.isfinite(number):
        raise SiteDataValidationError(f"{snapshot_id}: {field} must be finite")
    return number


def _probability(value: str, field: str, snapshot_id: str) -> float:
    number = _finite_number(value, field, snapshot_id)
    if not 0 <= number <= 1:
        raise SiteDataValidationError(f"{snapshot_id}: {field} must be between 0 and 1")
    return number


def _records(included_games: Path) -> dict[str, str]:
    """Derive FBS records from the exact games a snapshot included."""
    records: dict[str, list[int]] = {}
    with included_games.open(newline="", encoding="utf-8") as handle:
        for game in csv.DictReader(handle):
            home_id, away_id = game["homeId"], game["awayId"]
            home_points, away_points = int(game["homePoints"]), int(game["awayPoints"])
            classifications = {
                game["homeClassification"].casefold(),
                game["awayClassification"].casefold(),
            }
            if not classifications <= {"fbs", "fcs"}:
                continue
            for team_id, classification, won in (
                (home_id, game["homeClassification"], home_points > away_points),
                (away_id, game["awayClassification"], away_points > home_points),
            ):
                if classification.casefold() != "fbs":
                    continue
                entry = records.setdefault(team_id, [0, 0, 0])
                if home_points == away_points:
                    entry[2] += 1
                elif won:
                    entry[0] += 1
                else:
                    entry[1] += 1
    return {
        team_id: f"{wins}-{losses}" + (f"-{ties}" if ties else "")
        for team_id, (wins, losses, ties) in records.items()
    }


def _ranking_rows(path: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    snapshot_id = str(metadata["snapshot_id"])
    required = {
        "team_id", "team_name", "subdivision", "conference", "expected_rank",
        "median_rank", "interval_80_low", "interval_80_high", "top5_probability",
        "top10_probability", "top25_probability", "display_rank",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise SiteDataValidationError(f"{snapshot_id}: rankings missing {sorted(missing)}")
        rows = [row for row in reader if row["subdivision"].casefold() == "fbs"]
    if not rows:
        raise SiteDataValidationError(f"{snapshot_id}: no FBS ranking rows")

    team_ids: set[str] = set()
    ranks: set[int] = set()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        team_id = row["team_id"]
        if not team_id or team_id in team_ids:
            raise SiteDataValidationError(f"{snapshot_id}: duplicate or blank FBS team ID")
        team_ids.add(team_id)
        rated_value = row.get("rated", "true").strip().casefold()
        if rated_value not in {"true", "false"}:
            raise SiteDataValidationError(f"{snapshot_id}: rated must be true or false")
        rated = rated_value == "true"
        display_value = row["display_rank"].strip()
        if rated:
            display_rank = int(_finite_number(display_value, "display_rank", snapshot_id))
            if display_rank < 1 or display_rank in ranks:
                raise SiteDataValidationError(
                    f"{snapshot_id}: duplicate or invalid FBS display rank"
                )
            ranks.add(display_rank)
        else:
            if display_value != "NR":
                raise SiteDataValidationError(
                    f"{snapshot_id}: unrated FBS teams must have display rank NR"
                )
            display_rank = "NR"
        low = _finite_number(row["interval_80_low"], "interval_80_low", snapshot_id)
        high = _finite_number(row["interval_80_high"], "interval_80_high", snapshot_id)
        if low > high:
            raise SiteDataValidationError(f"{snapshot_id}: interval_80_low exceeds interval_80_high")
        eligible_games = int(row.get("eligible_games", row.get("games_played", "0")))
        if eligible_games < 0:
            raise SiteDataValidationError(f"{snapshot_id}: eligible game count must be nonnegative")
        normalized.append(
            {
                "display_rank": display_rank,
                "team_id": team_id,
                "team_name": row["team_name"],
                "conference": row["conference"],
                "rated": rated,
                "eligible_games": eligible_games,
                "eligible_evidence_count": int(
                    row.get("eligible_evidence_count", eligible_games)
                ),
                "expected_rank": _finite_number(row["expected_rank"], "expected_rank", snapshot_id),
                "median_rank": _finite_number(row["median_rank"], "median_rank", snapshot_id),
                "interval_80": [low, high],
                "top5_probability": _probability(row["top5_probability"], "top5_probability", snapshot_id),
                "top10_probability": _probability(row["top10_probability"], "top10_probability", snapshot_id),
                "top25_probability": _probability(row["top25_probability"], "top25_probability", snapshot_id),
            }
        )
    rated_count = sum(row["rated"] for row in normalized)
    expected_ranks = set(range(1, rated_count + 1))
    if ranks != expected_ranks:
        raise SiteDataValidationError(
            f"{snapshot_id}: FBS display ranks must be contiguous from 1 through "
            f"{rated_count}"
        )
    return sorted(
        normalized,
        key=lambda row: (
            not row["rated"],
            row["display_rank"] if row["rated"] else math.inf,
            str(row["team_name"]),
        ),
    )


def _comparison_key(snapshot: PublicationComparison) -> tuple[int, str, str | None]:
    return (
        snapshot.season,
        snapshot.ranking_family,
        snapshot.prior_family if snapshot.ranking_family == "predictive" else None,
    )


def _comparison_descriptor(snapshot: PreparedSnapshot) -> PublicationComparison:
    metadata = snapshot.metadata
    return PublicationComparison(
        season=int(metadata["season"]),
        ranking_family=str(metadata["ranking_family"]),
        prior_family=(
            str(metadata["prior_family"])
            if metadata["ranking_family"] == "predictive"
            else None
        ),
        publication_slot=snapshot.selected.publication_slot,
        publication_status=snapshot.selected.publication_status,
        publication_order=snapshot.selected.publication_order,
        snapshot_id=snapshot.snapshot_id,
        display_label=snapshot.selected.display_label,
    )


def resolve_previous_official(
    current: PublicationComparison,
    snapshots: list[PublicationComparison],
) -> PublicationComparison | None:
    """Resolve the latest strictly earlier compatible official publication."""
    compatible = [
        snapshot
        for snapshot in snapshots
        if snapshot.publication_status == "official"
        and snapshot.publication_order < current.publication_order
        and _comparison_key(snapshot) == _comparison_key(current)
    ]
    return max(compatible, key=lambda snapshot: snapshot.publication_order, default=None)


def _previous_official_snapshot(
    current: PreparedSnapshot, snapshots: list[PreparedSnapshot]
) -> PreparedSnapshot | None:
    previous = resolve_previous_official(
        _comparison_descriptor(current),
        [_comparison_descriptor(snapshot) for snapshot in snapshots],
    )
    if previous is None:
        return None
    return next(snapshot for snapshot in snapshots if snapshot.snapshot_id == previous.snapshot_id)


def _rank_change_text(
    *,
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    baseline_label: str | None,
    has_baseline: bool,
) -> tuple[str, str, int | None, int | None, str]:
    """Return machine status, compact display, numeric delta, old rank, and prose."""
    if not has_baseline or previous is None:
        return (
            "no_comparison",
            "—",
            None,
            None,
            "No earlier official ranking baseline",
        )

    current_rated = bool(current["rated"])
    previous_rated = bool(previous["rated"])
    baseline = f" since {baseline_label}" if baseline_label else ""
    previous_rank = previous["display_rank"] if previous_rated else None
    if current_rated and previous_rated:
        change = int(previous["display_rank"]) - int(current["display_rank"])
        if change > 0:
            return "ranked", f"↑{change}", change, previous_rank, f"Up {change}{baseline}"
        if change < 0:
            return "ranked", f"↓{abs(change)}", change, previous_rank, f"Down {abs(change)}{baseline}"
        return "ranked", "—", 0, previous_rank, f"Unchanged{baseline}"
    if current_rated:
        return "newly_rated", "NEW", None, None, f"Newly rated{baseline}"
    if previous_rated:
        return "became_unrated", "NR", None, previous_rank, f"Became unrated{baseline}"
    return "unrated", "—", None, None, "Unrated in both snapshots"


def _apply_rank_changes(
    current: PreparedSnapshot, previous: PreparedSnapshot | None
) -> None:
    """Attach build-time movement metadata to every current ranking row."""
    previous_by_team = (
        {row["team_id"]: row for row in previous.rankings} if previous is not None else {}
    )
    baseline_label = previous.selected.display_label if previous is not None else None
    for row in current.rankings:
        status, display, change, previous_rank, accessible = _rank_change_text(
            current=row,
            previous=previous_by_team.get(row["team_id"]),
            baseline_label=baseline_label,
            has_baseline=previous is not None,
        )
        row.update(
            {
                "previous_official_rank": previous_rank,
                "rank_change": change,
                "rank_change_status": status,
                "rank_change_display": display,
                "rank_change_accessible": accessible,
            }
        )


def _rank(value: str, field: str, snapshot_id: str) -> int:
    number = _finite_number(value, field, snapshot_id)
    if not number.is_integer() or number < 1:
        raise SiteDataValidationError(f"{snapshot_id}: {field} must be a positive integer")
    return int(number)


def _pmf_summary(pmf: list[float]) -> dict[str, Any]:
    """Summarize a discrete PMF with the established left-CDF quantiles."""
    ranks = range(1, len(pmf) + 1)
    cumulative = 0.0

    def quantile(probability: float) -> int:
        nonlocal cumulative
        cumulative = 0.0
        for rank, value in zip(ranks, pmf, strict=True):
            cumulative += value
            if cumulative >= probability:
                return rank
        return len(pmf)  # The validated sum can only leave a rounding-sized tail.

    interval_50 = [quantile(0.25), quantile(0.75)]
    interval_80 = [quantile(0.10), quantile(0.90)]
    interval_95 = [quantile(0.025), quantile(0.975)]
    modal_rank = max(range(len(pmf)), key=lambda index: pmf[index]) + 1
    return {
        "expected_rank": math.fsum(rank * value for rank, value in zip(ranks, pmf, strict=True)),
        "median_rank": quantile(0.50),
        "modal_rank": modal_rank,
        "interval_50": interval_50,
        "interval_80": interval_80,
        "interval_95": interval_95,
        "interval_widths": {
            "50": interval_50[1] - interval_50[0] + 1,
            "80": interval_80[1] - interval_80[0] + 1,
            "95": interval_95[1] - interval_95[0] + 1,
        },
        "rank_1_probability": pmf[0],
        "top5_probability": math.fsum(pmf[:5]),
        "top10_probability": math.fsum(pmf[:10]),
        "top25_probability": math.fsum(pmf[:25]),
    }


def _distribution_artifact(
    path: Path, metadata: dict[str, Any], rankings: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate FBS PMFs and convert them to a lazy browser artifact.

    This is intentionally a presentation/export transform: the posterior CSV is
    authoritative and is never recomputed or normalized here.
    """
    snapshot_id = str(metadata["snapshot_id"])
    ranking_by_team = {str(row["team_id"]): row for row in rankings}
    rank_count = len(rankings)
    pmfs: dict[str, dict[int, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"team_id", "rank", "probability"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SiteDataValidationError(f"{snapshot_id}: PMFs missing {sorted(missing)}")
        for row in reader:
            team_id = row["team_id"]
            if team_id not in ranking_by_team:
                # Source artifacts can also contain FCS PMFs.  Only published FBS
                # rankings belong in this consumer artifact.
                continue
            rank = _rank(row["rank"], "PMF rank", snapshot_id)
            probability = _probability(row["probability"], "PMF probability", snapshot_id)
            team_pmf = pmfs.setdefault(team_id, {})
            if rank in team_pmf:
                raise SiteDataValidationError(f"{snapshot_id}: duplicate PMF rank for team {team_id}")
            team_pmf[rank] = probability

    expected_ranks = set(range(1, rank_count + 1))
    teams: dict[str, Any] = {}
    for team_id, ranking in ranking_by_team.items():
        team_pmf = pmfs.get(team_id)
        if team_pmf is None:
            raise SiteDataValidationError(f"{snapshot_id}: missing PMF for FBS team {team_id}")
        if set(team_pmf) != expected_ranks:
            raise SiteDataValidationError(
                f"{snapshot_id}: PMF ranks for team {team_id} must be contiguous from "
                f"1 through {rank_count} (missing or unexpected ranks)"
            )
        pmf = [team_pmf[rank] for rank in range(1, rank_count + 1)]
        total = math.fsum(pmf)
        if abs(total - 1.0) > PMF_SUM_TOLERANCE:
            raise SiteDataValidationError(
                f"{snapshot_id}: PMF probabilities for team {team_id} sum to {total}, not 1"
            )
        summary = _pmf_summary(pmf)
        if abs(summary["expected_rank"] - ranking["expected_rank"]) > SUMMARY_TOLERANCE:
            raise SiteDataValidationError(f"{snapshot_id}: PMF expected rank disagrees for team {team_id}")
        if summary["median_rank"] != ranking["median_rank"]:
            raise SiteDataValidationError(f"{snapshot_id}: PMF median rank disagrees for team {team_id}")
        if summary["interval_80"] != ranking["interval_80"]:
            raise SiteDataValidationError(f"{snapshot_id}: PMF 80% interval disagrees for team {team_id}")
        teams[team_id] = {"pmf": pmf, "summary": summary}

    declared_rank_count = metadata.get("rank_count")
    if declared_rank_count is not None and int(declared_rank_count) != rank_count:
        raise SiteDataValidationError(
            f"{snapshot_id}: metadata rank_count does not match published FBS support"
        )

    return {
        "schema_version": SITE_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "rank_count": rank_count,
        "teams": teams,
    }


def _validate_metadata(metadata: dict[str, Any], source: Path) -> None:
    required = {
        "schema_version", "season", "snapshot_id", "snapshot_type", "ranking_family",
        "valid", "generation_timestamp",
    }
    missing = required - metadata.keys()
    if missing:
        raise SiteDataValidationError(f"{source}: metadata missing {sorted(missing)}")
    if metadata["schema_version"] not in SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS:
        raise SiteDataValidationError(f"{metadata['snapshot_id']}: unsupported snapshot schema")
    if metadata["valid"] is not True:
        raise SiteDataValidationError(f"{metadata['snapshot_id']}: snapshot metadata is not valid")
    if not isinstance(metadata["season"], int):
        raise SiteDataValidationError(f"{metadata['snapshot_id']}: season must be an integer")
    if metadata["snapshot_type"] not in {"preseason", "weekly", "live"}:
        raise SiteDataValidationError(f"{metadata['snapshot_id']}: unsupported snapshot type")
    if metadata["ranking_family"] not in RANKING_FAMILIES:
        raise SiteDataValidationError(f"{metadata['snapshot_id']}: unsupported ranking family")
    if metadata["ranking_family"] != "performance":
        if metadata.get("prior_family") not in {"context", "history"}:
            raise SiteDataValidationError(f"{metadata['snapshot_id']}: unsupported prior family")
        if "model_versions" not in metadata:
            raise SiteDataValidationError(f"{metadata['snapshot_id']}: model_versions is required")
    else:
        performance_required = {
            "model_version", "method", "anchor_family", "source_context_snapshot_id",
            "source_context_path", "requested_cutoff", "effective_cutoff",
            "source_retrieved_at", "source_retrieval_times", "source_response_hashes",
            "source_evidence_hashes", "game_corpus_sha256", "included_game_ids",
            "included_game_count", "prior_artifact_sha256", "prior_model_version",
        }
        missing_performance = performance_required - metadata.keys()
        if missing_performance:
            raise SiteDataValidationError(
                f"{metadata['snapshot_id']}: metadata missing {sorted(missing_performance)}"
            )
        _require_supported_methodology_version(
            metadata["model_version"],
            SUPPORTED_ARTIFACT_MODEL_VERSIONS["performance"],
            snapshot_id=str(metadata["snapshot_id"]),
            field="Performance model",
        )
        if metadata["method"] != "prior_stripping":
            raise SiteDataValidationError(f"{metadata['snapshot_id']}: unsupported Performance method")
        if metadata["anchor_family"] != "context":
            raise SiteDataValidationError(f"{metadata['snapshot_id']}: Performance must use Context anchoring")
        if metadata["snapshot_type"] == "preseason":
            raise SiteDataValidationError(f"{metadata['snapshot_id']}: Performance cannot be preseason")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_performance_source(metadata: dict[str, Any], root: Path) -> None:
    source_value = metadata["source_context_path"]
    if not isinstance(source_value, str) or not source_value or Path(source_value).is_absolute():
        raise SiteDataValidationError(
            f"{metadata['snapshot_id']}: source_context_path must be repository-relative"
        )
    source = root / source_value
    source_metadata_path = source / "metadata.json"
    if not source_metadata_path.is_file():
        raise SiteDataValidationError(
            f"{metadata['snapshot_id']}: source Context snapshot is unavailable"
        )
    source_metadata = _read_json(source_metadata_path)
    if (
        source_metadata.get("ranking_family") != "predictive"
        or source_metadata.get("prior_family") != "context"
        or source_metadata.get("valid") is not True
    ):
        raise SiteDataValidationError(
            f"{metadata['snapshot_id']}: source snapshot is not Predictive Context"
        )
    if source_metadata.get("snapshot_id") != metadata["source_context_snapshot_id"]:
        raise SiteDataValidationError(
            f"{metadata['snapshot_id']}: source Context snapshot ID mismatch"
        )
    for field in (
        "season", "snapshot_type", "requested_cutoff", "effective_cutoff",
        "source_retrieved_at", "source_retrieval_times", "source_response_hashes",
        "game_corpus_sha256", "included_game_ids", "included_game_count",
        "prior_artifact_sha256", "prior_model_version",
    ):
        if metadata.get(field) != source_metadata.get(field):
            raise SiteDataValidationError(
                f"{metadata['snapshot_id']}: Context/Performance evidence mismatch: {field}"
            )
    if metadata.get("source_evidence_hashes") != source_metadata.get("source_response_hashes"):
        raise SiteDataValidationError(
            f"{metadata['snapshot_id']}: Context/Performance evidence mismatch: source_evidence_hashes"
        )
    source_hash = metadata.get("source_context_metadata_sha256")
    if source_hash is not None and source_hash != _file_sha256(source_metadata_path):
        raise SiteDataValidationError(
            f"{metadata['snapshot_id']}: source Context metadata hash mismatch"
        )


def _write_json(path: Path, value: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, separators=(",", ":"), sort_keys=True)
        if compact
        else json.dumps(value, indent=2, sort_keys=True)
    )
    path.write_text(encoded + "\n", encoding="utf-8")


def _require_supported_methodology_version(
    actual: object,
    supported: frozenset[str],
    *,
    snapshot_id: str,
    field: str,
) -> None:
    """Require an artifact identifier to be in the retained support registry."""
    if not isinstance(actual, str) or actual not in supported:
        supported_values = ", ".join(sorted(supported))
        raise SiteDataValidationError(
            f"{snapshot_id}: {field} {actual!r} is unsupported; supported versions: "
            f"{supported_values}"
        )


def _require_consistent_methodology_version(
    actual: object, expected: object, *, snapshot_id: str, field: str
) -> None:
    """Require duplicated identifiers within one artifact to agree."""
    if actual != expected:
        raise SiteDataValidationError(
            f"{snapshot_id}: {field} {actual!r} does not match {expected!r}"
        )


def _validate_artifact_methodology_versions(
    prepared_snapshots: list[PreparedSnapshot],
) -> dict[str, object]:
    """Validate supported artifact IDs without pinning history to today's IDs."""
    methodology = production_methodology_metadata()
    supported_models = SUPPORTED_ARTIFACT_MODEL_VERSIONS
    supported_schemas = SUPPORTED_ARTIFACT_SCHEMA_VERSIONS
    for prepared in prepared_snapshots:
        metadata = prepared.metadata
        snapshot_id = prepared.snapshot_id
        ranking_family = metadata.get("ranking_family")
        if ranking_family not in {"predictive", "performance"}:
            continue
        _require_supported_methodology_version(
            metadata.get("schema_version"),
            supported_schemas["snapshot"],
            snapshot_id=snapshot_id,
            field="snapshot schema",
        )
        team_season_schema = prepared.team_seasons.get("schema_version")
        _require_supported_methodology_version(
            team_season_schema,
            supported_schemas["team_season"],
            snapshot_id=snapshot_id,
            field="team-season schema",
        )
        _require_supported_methodology_version(
            prepared.weekly_games.get("schema_version")
            if prepared.weekly_games
            else None,
            supported_schemas["weekly_game"],
            snapshot_id=snapshot_id,
            field="weekly-game schema",
        )
        declared_team_season_schema = metadata.get("team_season_schema_version")
        if declared_team_season_schema is not None:
            _require_supported_methodology_version(
                declared_team_season_schema,
                supported_schemas["team_season"],
                snapshot_id=snapshot_id,
                field="declared team-season schema",
            )
            _require_consistent_methodology_version(
                declared_team_season_schema,
                team_season_schema,
                snapshot_id=snapshot_id,
                field="declared team-season schema",
            )
        artifact_likelihood = prepared.team_seasons.get("historical_likelihood_version")
        if artifact_likelihood is not None:
            _require_supported_methodology_version(
                artifact_likelihood,
                supported_models["historical_likelihood"],
                snapshot_id=snapshot_id,
                field="team-season Historical Likelihood",
            )

        if ranking_family == "performance":
            _require_supported_methodology_version(
                metadata.get("model_version"),
                supported_models["performance"],
                snapshot_id=snapshot_id,
                field="Performance model",
            )
            _require_supported_methodology_version(
                metadata.get("prior_model_version"),
                supported_models["context_prior"],
                snapshot_id=snapshot_id,
                field="Performance anchor prior",
            )
            continue

        model_versions = metadata.get("model_versions")
        if not isinstance(model_versions, dict):
            raise SiteDataValidationError(
                f"{snapshot_id}: model_versions must be an object"
            )
        for key in ("context_prior", "history_prior", "historical_likelihood"):
            _require_supported_methodology_version(
                model_versions.get(key),
                supported_models[key],
                snapshot_id=snapshot_id,
                field=f"{key} model",
            )
        if "posterior" in model_versions:
            _require_supported_methodology_version(
                model_versions["posterior"],
                supported_models["posterior"],
                snapshot_id=snapshot_id,
                field="posterior model",
            )
        prior_key = (
            "context_prior"
            if metadata.get("prior_family") == "context"
            else "history_prior"
        )
        prior_model = metadata.get("prior_model_version")
        _require_supported_methodology_version(
            prior_model,
            supported_models[prior_key],
            snapshot_id=snapshot_id,
            field="prior model",
        )
        _require_consistent_methodology_version(
            model_versions.get(prior_key),
            prior_model,
            snapshot_id=snapshot_id,
            field=f"{prior_key} metadata",
        )
        historical_likelihood = metadata.get("historical_likelihood_version")
        _require_supported_methodology_version(
            historical_likelihood,
            supported_models["historical_likelihood"],
            snapshot_id=snapshot_id,
            field="Historical Likelihood",
        )
        _require_consistent_methodology_version(
            model_versions.get("historical_likelihood"),
            historical_likelihood,
            snapshot_id=snapshot_id,
            field="historical likelihood metadata",
        )
        if artifact_likelihood is not None:
            _require_consistent_methodology_version(
                artifact_likelihood,
                historical_likelihood,
                snapshot_id=snapshot_id,
                field="team-season Historical Likelihood",
            )
        prediction_schema = prepared.team_seasons.get("prediction_schema_version")
        if prediction_schema is not None:
            _require_supported_methodology_version(
                prediction_schema,
                supported_schemas["prediction"],
                snapshot_id=snapshot_id,
                field="prediction schema",
            )
        prediction_provenance = prepared.team_seasons.get("prediction_provenance")
        if isinstance(prediction_provenance, dict):
            provenance_likelihood = prediction_provenance.get(
                "historical_likelihood_version"
            )
            if provenance_likelihood is not None:
                _require_supported_methodology_version(
                    provenance_likelihood,
                    supported_models["historical_likelihood"],
                    snapshot_id=snapshot_id,
                    field="prediction Historical Likelihood",
                )
                _require_consistent_methodology_version(
                    provenance_likelihood,
                    historical_likelihood,
                    snapshot_id=snapshot_id,
                    field="prediction Historical Likelihood",
                )
        simulation = prepared.team_seasons.get("season_simulation")
        if isinstance(simulation, dict):
            simulation_schema = simulation.get("schema_version")
            _require_supported_methodology_version(
                simulation_schema,
                supported_schemas["season_simulation"],
                snapshot_id=snapshot_id,
                field="season-simulation schema",
            )
            simulation_version = simulation.get("simulation_version")
            _require_supported_methodology_version(
                simulation_version,
                supported_models["season_simulation"],
                snapshot_id=snapshot_id,
                field="season-simulation model",
            )
            configuration = simulation.get("configuration")
            if isinstance(configuration, dict):
                simulation_likelihood = configuration.get("likelihood_version")
                _require_supported_methodology_version(
                    simulation_likelihood,
                    supported_models["historical_likelihood"],
                    snapshot_id=snapshot_id,
                    field="season-simulation likelihood",
                )
                _require_consistent_methodology_version(
                    simulation_likelihood,
                    historical_likelihood,
                    snapshot_id=snapshot_id,
                    field="season-simulation likelihood",
                )
        declared_simulation_schema = metadata.get("season_simulation_schema_version")
        if declared_simulation_schema is not None:
            _require_supported_methodology_version(
                declared_simulation_schema,
                supported_schemas["season_simulation"],
                snapshot_id=snapshot_id,
                field="declared season-simulation schema",
            )
            if isinstance(simulation, dict):
                _require_consistent_methodology_version(
                    declared_simulation_schema,
                    simulation.get("schema_version"),
                    snapshot_id=snapshot_id,
                    field="declared season-simulation schema",
                )
        declared_simulation_version = metadata.get("season_simulation_version")
        if declared_simulation_version is not None:
            _require_supported_methodology_version(
                declared_simulation_version,
                supported_models["season_simulation"],
                snapshot_id=snapshot_id,
                field="declared season-simulation model",
            )
            if isinstance(simulation, dict):
                _require_consistent_methodology_version(
                    declared_simulation_version,
                    simulation.get("simulation_version"),
                    snapshot_id=snapshot_id,
                    field="declared season-simulation model",
                )
    return methodology


def _empty_team_season_artifact(
    metadata: dict[str, Any], rankings: list[dict[str, Any]]
) -> dict[str, Any]:
    """Keep pre-team-page fixture snapshots readable during schema migration."""
    return {
        "schema_version": TEAM_SEASON_SCHEMA_VERSION,
        "artifact_kind": "team_season",
        "snapshot_id": metadata["snapshot_id"],
        "season": metadata["season"],
        "snapshot_type": metadata["snapshot_type"],
        "anchor_family": "context",
        "source_context_snapshot_id": metadata["snapshot_id"],
        "requested_cutoff": metadata.get("requested_cutoff"),
        "effective_cutoff": metadata.get("effective_cutoff"),
        "source_retrieved_at": metadata.get("source_retrieved_at"),
        "source_retrieval_times": metadata.get("source_retrieval_times", {}),
        "source_response_hashes": metadata.get("source_response_hashes", {}),
        "game_corpus_sha256": metadata.get("game_corpus_sha256"),
        "included_game_ids": sorted(str(value) for value in metadata.get("included_game_ids", [])),
        "historical_likelihood_version": HISTORICAL_LIKELIHOOD_VERSION,
        "rank_count": metadata.get("rank_count"),
        "rating_definition": "normalize(single-game Historical Likelihood × opponent pair-cavity belief)",
        "loopy_bp_caveat": "The focal preseason prior is absent as a direct factor; loopy cycles can leave indirect feedback.",
        "rematch_definition": "All games in a grouped head-to-head pair use the same pair cavity; each game is rated independently.",
        "teams": {
            row["team_id"]: {
                "team_id": row["team_id"],
                "team_name": row["team_name"],
                "conference": row["conference"],
                "games": [],
            }
            for row in rankings
        },
    }


def _prediction_source(metadata: dict[str, Any]) -> str:
    family = str(metadata.get("prior_family", metadata.get("anchor_family", "context")))
    return PREDICTION_SOURCE_HISTORY if family == "history" else PREDICTION_SOURCE_CONTEXT


def _iso_datetime(value: object) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _validate_display_encoding(axis: dict[str, Any], snapshot_id: str) -> None:
    encoding = axis.get("probability_encoding")
    if not isinstance(encoding, dict) or any(
        encoding.get(field) != expected
        for field, expected in (
            ("type", "fixed_scale_integer"),
            ("scale", DISPLAY_PROBABILITY_SCALE),
            ("normalization", "divide weights by scale"),
            ("total_weight", DISPLAY_PROBABILITY_SCALE),
        )
    ):
        raise SiteDataValidationError(
            f"{snapshot_id}: display probability encoding is invalid"
        )


def _validate_display_masses(
    value: object,
    *,
    expected_length: int,
    label: str,
    allow_partial: bool = False,
) -> list[int]:
    """Validate fixed-scale integer display weights without changing values."""
    if not isinstance(value, list) or len(value) != expected_length:
        raise SiteDataValidationError(
            f"{label} must contain exactly {expected_length} display bins"
        )
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise SiteDataValidationError(f"{label} bins must be fixed-scale integers")
    values = list(value)
    if any(item < 0 or item > DISPLAY_PROBABILITY_SCALE for item in values):
        raise SiteDataValidationError(f"{label} bins must be nonnegative")
    total = sum(values)
    if allow_partial:
        if total > DISPLAY_PROBABILITY_SCALE:
            raise SiteDataValidationError(f"{label} bins contain more than the encoded total")
    elif total != DISPLAY_PROBABILITY_SCALE:
        raise SiteDataValidationError(
            f"{label} bins sum to {total}, not {DISPLAY_PROBABILITY_SCALE}"
        )
    return values


def _validate_performance_display(
    artifact: dict[str, Any],
    snapshot_id: str,
    ratings_count: int,
) -> None:
    """Validate the shared completed-game rank display contract when present."""
    axis = artifact.get("performance_axis")
    if axis is None:
        return  # Retain compatibility with pre-issue-46 retained artifacts.
    if not isinstance(axis, dict):
        raise SiteDataValidationError(f"{snapshot_id}: performance_axis must be an object")
    if axis.get("min_rank") != 1 or axis.get("max_rank") != artifact.get("rank_count"):
        raise SiteDataValidationError(f"{snapshot_id}: performance display axis disagrees with rank support")
    if axis.get("bins") != PERFORMANCE_DISPLAY_BINS:
        raise SiteDataValidationError(f"{snapshot_id}: unsupported performance display bin count")
    if axis.get("direction") != "best_to_worst":
        raise SiteDataValidationError(f"{snapshot_id}: performance display direction is invalid")
    _validate_display_encoding(axis, snapshot_id)
    percentile = artifact.get("performance_percentile")
    if not isinstance(percentile, dict):
        raise SiteDataValidationError(f"{snapshot_id}: performance percentile metadata is missing")
    if percentile.get("statistic") != "game_rating.expected_rank":
        raise SiteDataValidationError(f"{snapshot_id}: performance percentile statistic is invalid")
    if percentile.get("direction") != "lower_is_better":
        raise SiteDataValidationError(f"{snapshot_id}: performance percentile direction is invalid")
    reference_count = percentile.get("reference_count")
    if (
        isinstance(reference_count, bool)
        or not isinstance(reference_count, int)
        or reference_count != ratings_count
    ):
        raise SiteDataValidationError(f"{snapshot_id}: performance percentile reference count is invalid")


def _performance_grade_for_percentile(percentile: float) -> str:
    if percentile >= 90:
        return "A"
    if percentile >= 70:
        return "B"
    if percentile >= 30:
        return "C"
    if percentile >= 10:
        return "D"
    return "F"


def _validate_future_display(artifact: dict[str, Any], snapshot_id: str) -> None:
    """Validate fixed-grid predictive display data and explicit tail mass."""
    axis = artifact.get("future_margin_axis")
    if axis is None:
        return  # Retain compatibility with issue-41 artifacts until republished.
    if not isinstance(axis, dict):
        raise SiteDataValidationError(f"{snapshot_id}: future_margin_axis must be an object")
    if (
        axis.get("min_margin") != FUTURE_MARGIN_DISPLAY_MIN
        or axis.get("max_margin") != FUTURE_MARGIN_DISPLAY_MAX
        or axis.get("bins") != FUTURE_MARGIN_DISPLAY_BINS
        or axis.get("direction") != "home_minus_away"
        or axis.get("unit") != "points"
    ):
        raise SiteDataValidationError(f"{snapshot_id}: future margin display axis is invalid")
    _validate_display_encoding(axis, snapshot_id)
    prediction_map = artifact.get("future_predictions") or {}
    if not isinstance(prediction_map, dict):
        raise SiteDataValidationError(f"{snapshot_id}: future_predictions must be an object")
    for prediction_id, prediction in prediction_map.items():
        display = prediction.get("display_distribution") if isinstance(prediction, dict) else None
        if not isinstance(display, dict):
            raise SiteDataValidationError(
                f"{snapshot_id}: prediction {prediction_id} display distribution is missing"
            )
        values = _validate_display_masses(
            display.get("masses"),
            expected_length=FUTURE_MARGIN_DISPLAY_BINS,
            label=f"{snapshot_id}: prediction {prediction_id} display",
            allow_partial=True,
        )
        tail_values = []
        for field in ("lower_tail_probability", "upper_tail_probability"):
            value = display.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= DISPLAY_PROBABILITY_SCALE:
                raise SiteDataValidationError(
                    f"{snapshot_id}: prediction {prediction_id} {field} is invalid"
                )
            tail_values.append(value)
        if sum(values) + sum(tail_values) != DISPLAY_PROBABILITY_SCALE:
            raise SiteDataValidationError(
                f"{snapshot_id}: prediction {prediction_id} display mass is not normalized"
            )


def _validate_future_predictions(
    artifact: dict[str, Any],
    metadata: dict[str, Any],
    source_metadata: dict[str, Any],
    by_team: dict[str, Any],
    expected_ids: set[str],
) -> None:
    """Validate canonical future predictions and their snapshot provenance."""

    prediction_map = artifact.get("future_predictions")
    prediction_schema = artifact.get("prediction_schema_version")
    referenced_ids: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for team_id, team in by_team.items():
        for game in team.get("games", []):
            prediction_id = game.get("future_prediction_id")
            if prediction_id is not None:
                referenced_ids[str(prediction_id)].append((team_id, game))
    if prediction_map is None and not referenced_ids:
        return
    _require_supported_methodology_version(
        prediction_schema,
        SUPPORTED_ARTIFACT_SCHEMA_VERSIONS["prediction"],
        snapshot_id=str(metadata["snapshot_id"]),
        field="future-prediction schema",
    )
    if not isinstance(prediction_map, dict):
        raise SiteDataValidationError(
            f"{metadata['snapshot_id']}: future_predictions must be an object"
        )
    expected_source = _prediction_source(source_metadata)
    if artifact.get("prediction_source") != expected_source:
        raise SiteDataValidationError(
            f"{metadata['snapshot_id']}: future-prediction source mismatch"
        )
    provenance = artifact.get("prediction_provenance")
    if not isinstance(provenance, dict):
        raise SiteDataValidationError(
            f"{metadata['snapshot_id']}: future-prediction provenance is missing"
        )
    provenance_fields = (
        "source_snapshot_id",
        "season",
        "snapshot_type",
        "effective_cutoff",
        "game_corpus_sha256",
        "included_game_ids",
        "historical_likelihood_version",
    )
    for field in provenance_fields:
        expected_value = (
            source_metadata.get("snapshot_id")
            if field == "source_snapshot_id"
            else source_metadata.get(field)
        )
        if provenance.get(field) != expected_value:
            raise SiteDataValidationError(
                f"{metadata['snapshot_id']}: future-prediction provenance mismatch: {field}"
            )
    if provenance.get("prior_family") != source_metadata.get("prior_family"):
        raise SiteDataValidationError(
            f"{metadata['snapshot_id']}: future-prediction provenance mismatch: prior_family"
        )

    required_prediction_fields = {
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
    }
    for key, prediction in prediction_map.items():
        if not isinstance(prediction, dict):
            raise SiteDataValidationError(
                f"{metadata['snapshot_id']}: prediction {key} must be an object"
            )
        missing = required_prediction_fields - prediction.keys()
        if missing:
            raise SiteDataValidationError(
                f"{metadata['snapshot_id']}: prediction {key} missing {sorted(missing)}"
            )
        if str(key) != str(prediction["game_id"]):
            raise SiteDataValidationError(
                f"{metadata['snapshot_id']}: prediction key does not match game ID"
            )
        if prediction["prediction_source"] != expected_source:
            raise SiteDataValidationError(
                f"{metadata['snapshot_id']}: prediction source mismatch for {key}"
            )
        if prediction["source_snapshot_id"] != provenance["source_snapshot_id"]:
            raise SiteDataValidationError(
                f"{metadata['snapshot_id']}: prediction snapshot mismatch for {key}"
            )
        for field in ("expected_home_margin", "median_home_margin"):
            _finite_number(prediction[field], field, str(metadata["snapshot_id"]))
        probabilities = [
            _probability(prediction[field], field, str(metadata["snapshot_id"]))
            for field in ("home_win_probability", "away_win_probability", "tie_probability")
        ]
        if abs(probabilities[0] + probabilities[1] - 1.0) > 1.0e-8:
            raise SiteDataValidationError(
                f"{metadata['snapshot_id']}: prediction probabilities are not complementary"
            )
        intervals: dict[str, tuple[float, float]] = {}
        for field in ("margin_interval_50", "margin_interval_80", "margin_interval_95"):
            value = prediction[field]
            if not isinstance(value, list) or len(value) != 2:
                raise SiteDataValidationError(
                    f"{metadata['snapshot_id']}: {field} must contain two endpoints"
                )
            endpoints = tuple(
                _finite_number(item, f"{field} endpoint", str(metadata["snapshot_id"]))
                for item in value
            )
            if endpoints[0] > endpoints[1]:
                raise SiteDataValidationError(
                    f"{metadata['snapshot_id']}: {field} endpoints are reversed"
                )
            intervals[field] = endpoints
        if not (
            intervals["margin_interval_80"][0] <= intervals["margin_interval_50"][0]
            and intervals["margin_interval_50"][1] <= intervals["margin_interval_80"][1]
            and intervals["margin_interval_95"][0] <= intervals["margin_interval_80"][0]
            and intervals["margin_interval_80"][1] <= intervals["margin_interval_95"][1]
        ):
            raise SiteDataValidationError(
                f"{metadata['snapshot_id']}: predictive intervals are not nested"
            )
        if str(prediction["home_subdivision"]).casefold() not in {"fbs", "fcs"} or str(
            prediction["away_subdivision"]
        ).casefold() not in {"fbs", "fcs"}:
            raise SiteDataValidationError(
                f"{metadata['snapshot_id']}: prediction subdivision is unsupported"
            )

    cutoff = _iso_datetime(source_metadata.get("effective_cutoff"))
    for prediction_id, references in referenced_ids.items():
        if prediction_id not in prediction_map:
            raise SiteDataValidationError(
                f"{metadata['snapshot_id']}: schedule references missing prediction {prediction_id}"
            )
        prediction = prediction_map[prediction_id]
        if prediction_id in expected_ids:
            raise SiteDataValidationError(
                f"{metadata['snapshot_id']}: completed evidence has a future prediction"
            )
        for team_id, game in references:
            if team_id not in {prediction["home_team_id"], prediction["away_team_id"]}:
                raise SiteDataValidationError(
                    f"{metadata['snapshot_id']}: prediction is attached to the wrong team"
                )
            opponent_id = prediction["away_team_id"] if team_id == prediction["home_team_id"] else prediction["home_team_id"]
            if str(game.get("opponent_id")) != str(opponent_id):
                raise SiteDataValidationError(
                    f"{metadata['snapshot_id']}: prediction opponent mismatch"
                )
            if game.get("result") is not None or game.get("score") is not None or game.get("game_rating") is not None or game.get("modeled"):
                raise SiteDataValidationError(
                    f"{metadata['snapshot_id']}: future prediction reveals completed evidence"
                )
            if cutoff is not None:
                game_date = _iso_datetime(game.get("date"))
                if game_date is None or game_date <= cutoff:
                    raise SiteDataValidationError(
                        f"{metadata['snapshot_id']}: future prediction is not strictly after cutoff"
                    )
    for prediction_id in prediction_map:
        if str(prediction_id) not in referenced_ids:
            raise SiteDataValidationError(
                f"{metadata['snapshot_id']}: unreferenced future prediction {prediction_id}"
            )
    _validate_future_display(artifact, str(metadata["snapshot_id"]))


def _validate_season_simulation(
    artifact: dict[str, Any],
    metadata: dict[str, Any],
    rankings: list[dict[str, Any]],
) -> None:
    """Validate the canonical snapshot-level regular-season forecast."""
    simulation = artifact.get("season_simulation")
    if simulation is None:
        if metadata.get("season_simulation_version") is not None or metadata.get(
            "season_simulation_configuration"
        ) is not None:
            raise SiteDataValidationError(
                f"{metadata['snapshot_id']}: Season Simulation V1 is declared but missing"
            )
        return  # Retain compatibility with pre-issue-49 retained artifacts.
    snapshot_id = str(metadata["snapshot_id"])
    if not isinstance(simulation, dict):
        raise SiteDataValidationError(f"{snapshot_id}: season_simulation must be an object")
    _require_supported_methodology_version(
        simulation.get("schema_version"),
        SUPPORTED_ARTIFACT_SCHEMA_VERSIONS["season_simulation"],
        snapshot_id=snapshot_id,
        field="season-simulation schema",
    )
    if simulation.get("artifact_kind") != "season_simulation":
        raise SiteDataValidationError(
            f"{snapshot_id}: unsupported season simulation artifact"
        )
    if simulation.get("simulation_version") != metadata.get(
        "season_simulation_version", simulation.get("simulation_version")
    ):
        raise SiteDataValidationError(f"{snapshot_id}: season simulation version mismatch")
    if simulation.get("prediction_source") != artifact.get("prediction_source"):
        raise SiteDataValidationError(f"{snapshot_id}: season simulation source mismatch")
    configuration = simulation.get("configuration")
    if not isinstance(configuration, dict):
        raise SiteDataValidationError(f"{snapshot_id}: season simulation configuration is missing")
    if configuration.get("simulation_version") != simulation.get("simulation_version"):
        raise SiteDataValidationError(f"{snapshot_id}: season simulation configuration version mismatch")
    declared_configuration = metadata.get("season_simulation_configuration")
    if declared_configuration is not None and configuration != declared_configuration:
        raise SiteDataValidationError(f"{snapshot_id}: season simulation configuration mismatch")
    _require_supported_methodology_version(
        configuration.get("likelihood_version"),
        SUPPORTED_ARTIFACT_MODEL_VERSIONS["historical_likelihood"],
        snapshot_id=snapshot_id,
        field="season-simulation likelihood",
    )
    if configuration.get("likelihood_version") != metadata.get(
        "historical_likelihood_version", HISTORICAL_LIKELIHOOD_VERSION
    ):
        raise SiteDataValidationError(f"{snapshot_id}: season simulation likelihood version mismatch")
    if configuration.get("latent_sampling_method") != "independent_marginal_pmf":
        raise SiteDataValidationError(f"{snapshot_id}: season latent sampling method is invalid")

    provenance = simulation.get("provenance")
    if not isinstance(provenance, dict):
        raise SiteDataValidationError(f"{snapshot_id}: season simulation provenance is missing")
    provenance_fields = (
        "source_snapshot_id",
        "season",
        "snapshot_type",
        "requested_cutoff",
        "effective_cutoff",
        "game_corpus_sha256",
        "historical_likelihood_version",
        "prediction_source",
    )
    for field in provenance_fields:
        if field == "source_snapshot_id":
            expected = metadata.get("snapshot_id")
        elif field == "prediction_source":
            expected = artifact.get("prediction_source")
        else:
            expected = metadata.get(field)
        if field in provenance and provenance[field] != expected:
            raise SiteDataValidationError(
                f"{snapshot_id}: season simulation provenance mismatch: {field}"
            )

    if configuration.get("conditional_distribution_method") != "exact_poisson_binomial":
        raise SiteDataValidationError(f"{snapshot_id}: season simulation conditional method is invalid")
    outer_count = configuration.get("outer_draw_count")
    inner_count = configuration.get("inner_rollout_count")
    seed = configuration.get("seed")
    if (
        isinstance(outer_count, bool)
        or not isinstance(outer_count, int)
        or outer_count < 1
        or isinstance(inner_count, bool)
        or not isinstance(inner_count, int)
        or inner_count < 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise SiteDataValidationError(f"{snapshot_id}: season simulation configuration is invalid")

    scope = simulation.get("season_scope")
    if not isinstance(scope, dict) or scope.get("season_type") != "regular":
        raise SiteDataValidationError(f"{snapshot_id}: season simulation scope is invalid")
    if scope.get("unsupported_behavior") != "fail_closed":
        raise SiteDataValidationError(f"{snapshot_id}: unsupported-game behavior is not fail-closed")
    future_values = scope.get("future_game_ids")
    unresolved_values = scope.get("unresolved_game_ids")
    forecast_scope_values = scope.get("forecast_scope_game_ids")
    supported_values = scope.get("supported_future_game_ids")
    if not all(
        isinstance(value, list)
        for value in (future_values, unresolved_values, forecast_scope_values, supported_values)
    ):
        raise SiteDataValidationError(f"{snapshot_id}: season game scope IDs are invalid")
    future_ids = {str(value) for value in future_values}
    unresolved_ids = {str(value) for value in unresolved_values}
    forecast_scope_ids = {str(value) for value in forecast_scope_values}
    supported_ids = {str(value) for value in supported_values}
    if any(
        len(values) != len({str(value) for value in values})
        for values in (future_values, unresolved_values, forecast_scope_values, supported_values)
    ):
        raise SiteDataValidationError(f"{snapshot_id}: season scope IDs contain duplicates")
    if future_ids & unresolved_ids or future_ids | unresolved_ids != forecast_scope_ids:
        raise SiteDataValidationError(f"{snapshot_id}: season scope state partition is invalid")
    if not supported_ids <= future_ids:
        raise SiteDataValidationError(f"{snapshot_id}: supported season games are not a subset")
    if scope.get("future_game_count") != len(future_ids) or scope.get(
        "unresolved_game_count"
    ) != len(unresolved_ids) or scope.get("forecast_scope_game_count") != len(
        forecast_scope_ids
    ):
        raise SiteDataValidationError(f"{snapshot_id}: season scope counts are inconsistent")
    unsupported = scope.get("unsupported_future_games", [])
    if not isinstance(unsupported, list):
        raise SiteDataValidationError(f"{snapshot_id}: unsupported season games are invalid")
    unsupported_ids = {str(item.get("game_id")) for item in unsupported if isinstance(item, dict)}
    if not unsupported_ids <= forecast_scope_ids:
        raise SiteDataValidationError(f"{snapshot_id}: unsupported season games are outside scope")
    if supported_ids & unsupported_ids:
        raise SiteDataValidationError(f"{snapshot_id}: season game is both supported and unsupported")
    excluded = scope.get("excluded_schedule_games")
    if not isinstance(excluded, list) or any(
        not isinstance(item, dict) or not item.get("game_id") for item in excluded
    ):
        raise SiteDataValidationError(f"{snapshot_id}: excluded schedule games are invalid")
    if {str(item["game_id"]) for item in excluded} & forecast_scope_ids:
        raise SiteDataValidationError(f"{snapshot_id}: excluded schedule game is in forecast scope")

    latent_quality = simulation.get("latent_quality")
    if not isinstance(latent_quality, dict) or latent_quality.get(
        "held_fixed_throughout_universe"
    ) is not True:
        raise SiteDataValidationError(f"{snapshot_id}: latent quality is not held fixed")
    conditional_model = simulation.get("conditional_game_model")
    if not isinstance(conditional_model, dict) or conditional_model.get(
        "games_conditionally_independent"
    ) is not True or conditional_model.get("posterior_updates_from_simulated_games") is not False:
        raise SiteDataValidationError(f"{snapshot_id}: conditional game model is invalid")

    by_team = simulation.get("teams")
    if not isinstance(by_team, dict):
        raise SiteDataValidationError(f"{snapshot_id}: season simulation teams are missing")
    ranking_ids = {str(row["team_id"]) for row in rankings}
    if not ranking_ids <= set(by_team):
        raise SiteDataValidationError(f"{snapshot_id}: season simulation is missing an FBS team")
    probability_fields = (
        "expected_final_wins",
        "expected_remaining_wins",
    )
    unavailable_team_ids: set[str] = set()
    for team_id in ranking_ids:
        summary = by_team[team_id]
        if not isinstance(summary, dict) or summary.get("team_id") != team_id:
            raise SiteDataValidationError(f"{snapshot_id}: invalid season summary for {team_id}")
        status = summary.get("forecast_status")
        if status not in {"available", "unavailable"}:
            raise SiteDataValidationError(f"{snapshot_id}: invalid season status for {team_id}")
        completed_games = summary.get("completed_regular_season_games")
        remaining_games = summary.get("remaining_games")
        forecast_scope_games = summary.get("forecast_scope_games")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (completed_games, remaining_games, forecast_scope_games)
        ) or completed_games + remaining_games != forecast_scope_games:
            raise SiteDataValidationError(
                f"{snapshot_id}: season schedule accounting is invalid for {team_id}"
            )
        if status == "unavailable":
            unavailable_team_ids.add(team_id)
            if not summary.get("unsupported_games"):
                raise SiteDataValidationError(
                    f"{snapshot_id}: unavailable season summary has no unsupported games for {team_id}"
                )
            continue
        for field in probability_fields:
            _finite_number(summary.get(field), field, snapshot_id)
        for field in (
            "final_win_distribution",
            "remaining_win_distribution",
            "record_probabilities",
        ):
            distribution = summary.get(field)
            if not isinstance(distribution, dict) or not distribution:
                raise SiteDataValidationError(
                    f"{snapshot_id}: season summary {field} is missing for {team_id}"
                )
            values = [_finite_number(value, field, snapshot_id) for value in distribution.values()]
            if any(value < 0 for value in values) or not math.isclose(
                sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-8
            ):
                raise SiteDataValidationError(
                    f"{snapshot_id}: season summary {field} is not normalized for {team_id}"
                )
        thresholds = summary.get("threshold_probabilities")
        if not isinstance(thresholds, dict) or any(
            not 0 <= _probability(value, "threshold probability", snapshot_id) <= 1
            for value in thresholds.values()
        ):
            raise SiteDataValidationError(f"{snapshot_id}: invalid season thresholds for {team_id}")
        variance = summary.get("variance_decomposition")
        if not isinstance(variance, dict):
            raise SiteDataValidationError(f"{snapshot_id}: season variance decomposition is missing")
        total = _finite_number(variance.get("total"), "season variance", snapshot_id)
        quality_fraction = _probability(
            variance.get("team_quality_fraction"), "team quality fraction", snapshot_id
        )
        game_fraction = _probability(
            variance.get("game_randomness_fraction"), "game randomness fraction", snapshot_id
        )
        if total > 1.0e-12 and not math.isclose(
            quality_fraction + game_fraction, 1.0, rel_tol=0.0, abs_tol=1.0e-8
        ):
                raise SiteDataValidationError(f"{snapshot_id}: season variance fractions do not sum to one")

    declared_unavailable = {str(value) for value in scope.get("unsupported_team_ids", [])}
    if declared_unavailable != unavailable_team_ids:
        raise SiteDataValidationError(f"{snapshot_id}: season unavailable-team scope is inconsistent")
    expected_status = "partial" if unavailable_team_ids else "available"
    if simulation.get("forecast_status") != expected_status:
        raise SiteDataValidationError(f"{snapshot_id}: season forecast status is inconsistent")
    accounting = simulation.get("schedule_accounting")
    if not isinstance(accounting, dict) or set(accounting) != ranking_ids:
        raise SiteDataValidationError(f"{snapshot_id}: season schedule accounting is missing")
    for team_id, item in accounting.items():
        if not isinstance(item, dict):
            raise SiteDataValidationError(f"{snapshot_id}: invalid season accounting for {team_id}")
        if item.get("invariant") != "completed + remaining = forecast scope":
            raise SiteDataValidationError(f"{snapshot_id}: season accounting invariant is missing")
        if any(
            item.get(key) != by_team[team_id].get(summary_key)
            for key, summary_key in (
                ("completed_regular_season_games", "completed_regular_season_games"),
                ("remaining_regular_season_games", "remaining_games"),
                ("forecast_scope_games", "forecast_scope_games"),
            )
        ):
            raise SiteDataValidationError(f"{snapshot_id}: season accounting disagrees for {team_id}")

    game_marginals = simulation.get("game_marginals")
    if not isinstance(game_marginals, dict) or set(game_marginals) != supported_ids:
        raise SiteDataValidationError(f"{snapshot_id}: season game marginals do not match scope")
    for game_id, marginal in game_marginals.items():
        if not isinstance(marginal, dict) or marginal.get("game_id") != game_id:
            raise SiteDataValidationError(f"{snapshot_id}: invalid season game marginal {game_id}")
        for field in (
            "exact_home_win_probability",
            "outer_home_win_probability",
        ):
            _probability(marginal.get(field), field, snapshot_id)
        for field in (
            "home_win_probability_error",
            "exact_expected_home_margin",
            "outer_expected_home_margin",
            "expected_home_margin_error",
        ):
            _finite_number(marginal.get(field), field, snapshot_id)


def _browser_team_season_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Remove durable forecast diagnostics from the browser-facing copy."""
    exported = dict(artifact)
    simulation = artifact.get("season_simulation")
    if not isinstance(simulation, dict):
        return exported
    browser_simulation = dict(simulation)
    for field in (
        "game_marginals",
        "cross_game_dependence",
        "validation",
        "monte_carlo",
    ):
        browser_simulation.pop(field, None)
    browser_teams = browser_simulation.get("teams")
    if isinstance(browser_teams, dict):
        browser_simulation["teams"] = {
            team_id: {
                key: value
                for key, value in summary.items()
                if key not in {"event_probability_decomposition", "monte_carlo"}
            }
            for team_id, summary in browser_teams.items()
            if isinstance(summary, dict)
        }
    exported["season_simulation"] = browser_simulation
    return exported


def _weekly_week_sort_key(value: object) -> tuple[int, int | str, str]:
    """Sort numeric weeks before named or missing schedule weeks."""
    if isinstance(value, bool):
        return (2, str(value), "")
    try:
        return (0, int(value), "")
    except (TypeError, ValueError):
        if value is None or str(value).strip() == "":
            return (2, "", "")
        return (1, str(value), "")


def _weekly_game_state(game: dict[str, Any], cutoff: datetime | None) -> str:
    """Resolve state for old artifacts that predate the explicit game_state."""
    state = game.get("game_state")
    if state in {"completed", "future", "unresolved", "cancelled", "out_of_scope"}:
        return state
    if game.get("result") is not None or game.get("score") is not None:
        return "completed"
    game_date = _iso_datetime(game.get("date"))
    if game.get("future_prediction_id") is not None or (
        cutoff is not None and game_date is not None and game_date > cutoff
    ):
        return "future"
    return "unresolved"


def default_week_key(
    weekly_games: dict[str, Any], metadata: dict[str, Any] | None = None
) -> str | None:
    """Choose the week relevant to a snapshot when no week is in the URL."""
    weeks = weekly_games.get("weeks", [])
    if not isinstance(weeks, list) or not weeks:
        return None
    metadata = metadata or weekly_games
    if metadata.get("snapshot_type") == "preseason":
        return str(weeks[0]["key"])
    display_label = str(metadata.get("display_label", "")).strip().casefold()
    if display_label:
        labeled_week = next(
            (
                week
                for week in weeks
                if str(week.get("label", "")).strip().casefold() == display_label
            ),
            None,
        )
        if labeled_week is not None:
            return str(labeled_week["key"])
    cutoff = _iso_datetime(metadata.get("effective_cutoff"))
    if cutoff is None:
        return str(weeks[0]["key"])

    relevant = []
    for week in weeks:
        if not isinstance(week, dict):
            continue
        games = week.get("games", [])
        if any(
            isinstance(game, dict)
            and (game_date := _iso_datetime(game.get("date"))) is not None
            and game_date <= cutoff
            for game in games
        ):
            relevant.append(week)
    return str((relevant[-1] if relevant else weeks[0])["key"])


def _weekly_team_descriptor(
    team_id: str,
    team_name: str,
    subdivision: str,
    conference: str,
) -> dict[str, str]:
    return {
        "team_id": str(team_id),
        "team_name": str(team_name),
        "subdivision": str(subdivision or "").casefold(),
        "conference": str(conference or ""),
    }


def _weekly_performance_summary(
    rating: dict[str, Any] | None,
    *,
    display_ref: str | None,
) -> dict[str, Any] | None:
    """Keep exact rating summaries while referencing, rather than copying, bins."""
    if not isinstance(rating, dict):
        return None
    summary = {key: value for key, value in rating.items() if key != "display_pmf"}
    if display_ref is not None:
        summary["display_pmf_ref"] = display_ref
    return summary


def build_weekly_game_artifact(
    team_season_artifact: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one canonical, game-centric view from a team-season artifact.

    Team-season data intentionally stores a schedule entry from each FBS
    team's perspective.  This function folds those entries by stable game ID,
    so a weekly page renders each scheduled game once while retaining the
    existing rating and prediction sources.  Display PMFs live in keyed maps;
    cards carry references to them rather than serializing another copy in
    each performance summary.
    """
    metadata = metadata or team_season_artifact
    cutoff = _iso_datetime(team_season_artifact.get("effective_cutoff"))
    if cutoff is None:
        cutoff = _iso_datetime(metadata.get("effective_cutoff"))
    teams = team_season_artifact.get("teams", {})
    if not isinstance(teams, dict):
        raise SiteDataValidationError("weekly game source teams must be an object")
    future_predictions = team_season_artifact.get("future_predictions", {})
    if not isinstance(future_predictions, dict):
        raise SiteDataValidationError("weekly game source future_predictions must be an object")

    games_by_id: dict[str, dict[str, Any]] = {}
    performance_displays: dict[str, list[int]] = {}
    for team_id in sorted(teams, key=str):
        team = teams[team_id]
        if not isinstance(team, dict):
            continue
        team_id = str(team_id)
        team_name = str(team.get("team_name", team_id))
        team_conference = str(team.get("conference", ""))
        entries = team.get("games", [])
        if not isinstance(entries, list):
            raise SiteDataValidationError(f"weekly game source games for {team_id} must be a list")
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("game_id"):
                raise SiteDataValidationError(f"weekly game source has an invalid game for {team_id}")
            game_id = str(entry["game_id"])
            opponent_id = str(entry.get("opponent_id", ""))
            if not opponent_id:
                raise SiteDataValidationError(f"weekly game {game_id} is missing its opponent")
            site = str(entry.get("site", "")).casefold()
            prediction_id_value = entry.get("future_prediction_id")
            prediction = (
                future_predictions.get(str(prediction_id_value))
                if prediction_id_value is not None
                else None
            )
            if isinstance(prediction, dict) and prediction.get("home_team_id") is not None:
                # Future predictions own the canonical home-minus-away
                # orientation, including neutral-site games whose schedule
                # entries intentionally do not privilege either team.
                home_id = str(prediction["home_team_id"])
                away_id = str(prediction["away_team_id"])
            elif site == "away":
                home_id, away_id = opponent_id, team_id
            elif site == "neutral":
                home_id, away_id = sorted((team_id, opponent_id))
            else:
                home_id, away_id = team_id, opponent_id
            home_is_focal = home_id == team_id
            prediction_home = prediction.get("home_team_name") if isinstance(prediction, dict) else None
            prediction_away = prediction.get("away_team_name") if isinstance(prediction, dict) else None
            home_name = prediction_home or (team_name if home_is_focal else str(entry.get("opponent_name", opponent_id)))
            away_name = prediction_away or (str(entry.get("opponent_name", opponent_id)) if home_is_focal else team_name)
            home_conference = team_conference if home_is_focal else str(entry.get("opponent_conference", ""))
            away_conference = str(entry.get("opponent_conference", "")) if home_is_focal else team_conference
            home_subdivision = (
                str(prediction.get("home_subdivision", ""))
                if isinstance(prediction, dict) and prediction.get("home_subdivision")
                else "fbs" if home_is_focal else str(entry.get("opponent_classification", ""))
            )
            away_subdivision = (
                str(prediction.get("away_subdivision", ""))
                if isinstance(prediction, dict) and prediction.get("away_subdivision")
                else str(entry.get("opponent_classification", "")) if home_is_focal else "fbs"
            )
            state = _weekly_game_state(entry, cutoff)
            record = games_by_id.get(game_id)
            if record is None:
                home = _weekly_team_descriptor(home_id, home_name, home_subdivision, home_conference)
                away = _weekly_team_descriptor(away_id, away_name, away_subdivision, away_conference)
                record = {
                    "game_id": game_id,
                    "week": entry.get("week"),
                    "date": entry.get("date"),
                    "season_type": entry.get("season_type", ""),
                    "conference_game": bool(entry.get("conference_game", False)),
                    "neutral_site": site == "neutral",
                    "state": state,
                    "home_team": home,
                    "away_team": away,
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "score": None,
                    "winner_team_id": None,
                    "home_performance": None,
                    "away_performance": None,
                    "home_performance_ref": None,
                    "away_performance_ref": None,
                    "future_prediction_id": None,
                }
                games_by_id[game_id] = record
            else:
                if (record["home_team_id"], record["away_team_id"]) != (home_id, away_id):
                    raise SiteDataValidationError(f"weekly game {game_id} has inconsistent home/away sides")
                if record["week"] != entry.get("week") or record["date"] != entry.get("date"):
                    raise SiteDataValidationError(f"weekly game {game_id} has inconsistent schedule metadata")
                if record["state"] != state:
                    raise SiteDataValidationError(f"weekly game {game_id} has inconsistent snapshot state")

            score = entry.get("score")
            if isinstance(score, dict) and score.get("team") is not None and score.get("opponent") is not None:
                focal_score = int(score["team"])
                opponent_score = int(score["opponent"])
                home_score, away_score = (
                    (focal_score, opponent_score) if home_is_focal else (opponent_score, focal_score)
                )
                canonical_score = {"home": home_score, "away": away_score}
                if record["score"] is not None and record["score"] != canonical_score:
                    raise SiteDataValidationError(f"weekly game {game_id} has inconsistent scores")
                record["score"] = canonical_score
                record["winner_team_id"] = (
                    record["home_team_id"] if home_score > away_score
                    else record["away_team_id"] if away_score > home_score
                    else None
                )

            rating = entry.get("game_rating")
            if rating is not None:
                side = "home" if home_is_focal else "away"
                display_ref = f"{game_id}:{team_id}"
                display = rating.get("display_pmf") if isinstance(rating, dict) else None
                if isinstance(display, list):
                    performance_displays[display_ref] = list(display)
                else:
                    display_ref = None
                record[f"{side}_performance"] = _weekly_performance_summary(
                    rating, display_ref=display_ref
                )
                record[f"{side}_performance_ref"] = display_ref

            prediction_id = entry.get("future_prediction_id")
            if prediction_id is not None:
                prediction_id = str(prediction_id)
                if record["future_prediction_id"] not in {None, prediction_id}:
                    raise SiteDataValidationError(f"weekly game {game_id} has inconsistent prediction references")
                record["future_prediction_id"] = prediction_id

    games = sorted(
        games_by_id.values(),
        key=lambda game: (
            _weekly_week_sort_key(game.get("week")),
            str(game.get("date") or ""),
            str(game.get("game_id")),
            str(game["home_team_id"]),
            str(game["away_team_id"]),
        ),
    )
    weeks_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    week_values: dict[str, object] = {}
    for game in games:
        week = game.get("week")
        key = "unknown" if week is None or str(week).strip() == "" else str(week)
        weeks_by_key[key].append(game)
        week_values[key] = week
    weeks: list[dict[str, Any]] = []
    for key in sorted(weeks_by_key, key=lambda value: _weekly_week_sort_key(week_values[value])):
        grouped = weeks_by_key[key]
        counts = defaultdict(int)
        for game in grouped:
            counts[game["state"]] += 1
        week = week_values[key]
        label = f"Week {week}" if week is not None and str(week).strip() else "Unscheduled"
        weeks.append(
            {
                "week": week,
                "key": key,
                "label": label,
                "scheduled_game_count": len(grouped),
                "completed_game_count": counts["completed"],
                "future_game_count": counts["future"],
                "unresolved_game_count": counts["unresolved"] + counts["out_of_scope"],
                "cancelled_game_count": counts["cancelled"],
                "games": grouped,
            }
        )
    weekly_artifact = {
        "schema_version": WEEKLY_GAME_SCHEMA_VERSION,
        "artifact_kind": "weekly_games",
        "snapshot_id": team_season_artifact.get("snapshot_id"),
        "season": team_season_artifact.get("season", metadata.get("season")),
        "snapshot_type": team_season_artifact.get("snapshot_type", metadata.get("snapshot_type")),
        "ranking_family": metadata.get("ranking_family"),
        "prior_family": metadata.get("prior_family"),
        "requested_cutoff": team_season_artifact.get("requested_cutoff"),
        "effective_cutoff": team_season_artifact.get("effective_cutoff"),
        "included_game_ids": list(team_season_artifact.get("included_game_ids", [])),
        "prediction_schema_version": team_season_artifact.get("prediction_schema_version"),
        "prediction_source": team_season_artifact.get("prediction_source"),
        "prediction_provenance": team_season_artifact.get("prediction_provenance"),
        "performance_axis": team_season_artifact.get("performance_axis"),
        "performance_percentile": team_season_artifact.get("performance_percentile"),
        "future_margin_axis": team_season_artifact.get("future_margin_axis"),
        "performance_displays": performance_displays,
        "future_predictions": future_predictions,
        "game_count": len(games),
        "week_count": len(weeks),
        "weeks": weeks,
    }
    weekly_artifact["default_week"] = default_week_key(weekly_artifact, metadata)
    return weekly_artifact


def _validate_weekly_game_artifact(
    artifact: dict[str, Any],
    team_season_artifact: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Fail closed if the canonical weekly fold loses source consistency."""
    snapshot_id = str(metadata.get("snapshot_id", artifact.get("snapshot_id", "unknown")))
    _require_supported_methodology_version(
        artifact.get("schema_version"),
        SUPPORTED_ARTIFACT_SCHEMA_VERSIONS["weekly_game"],
        snapshot_id=snapshot_id,
        field="weekly-game schema",
    )
    if artifact.get("artifact_kind") != "weekly_games":
        raise SiteDataValidationError(f"{snapshot_id}: invalid weekly game artifact kind")
    if artifact.get("snapshot_id") != team_season_artifact.get("snapshot_id"):
        raise SiteDataValidationError(f"{snapshot_id}: weekly artifact snapshot mismatch")
    weeks = artifact.get("weeks")
    if not isinstance(weeks, list):
        raise SiteDataValidationError(f"{snapshot_id}: weekly artifact weeks must be a list")
    games = [game for week in weeks for game in week.get("games", [])]
    ids = [str(game.get("game_id")) for game in games]
    if len(ids) != len(set(ids)):
        raise SiteDataValidationError(f"{snapshot_id}: a game appears more than once in weekly artifact")
    source_ids = {
        str(game.get("game_id"))
        for team in team_season_artifact.get("teams", {}).values()
        for game in team.get("games", [])
    }
    if set(ids) != source_ids:
        raise SiteDataValidationError(f"{snapshot_id}: weekly artifact game membership mismatch")
    predictions = artifact.get("future_predictions", {})
    if predictions != team_season_artifact.get("future_predictions", {}):
        raise SiteDataValidationError(f"{snapshot_id}: weekly prediction map mismatch")
    performance_displays = artifact.get("performance_displays", {})
    if not isinstance(performance_displays, dict):
        raise SiteDataValidationError(f"{snapshot_id}: weekly performance displays must be an object")
    for game in games:
        if game.get("state") not in {"completed", "future", "unresolved", "cancelled", "out_of_scope"}:
            raise SiteDataValidationError(f"{snapshot_id}: weekly game state is invalid")
        prediction_id = game.get("future_prediction_id")
        if prediction_id is not None and str(prediction_id) not in predictions:
            raise SiteDataValidationError(f"{snapshot_id}: weekly game references missing prediction")
        for side in ("home", "away"):
            performance = game.get(f"{side}_performance")
            reference = game.get(f"{side}_performance_ref")
            if performance is None:
                if reference is not None:
                    raise SiteDataValidationError(f"{snapshot_id}: weekly performance reference has no summary")
            elif reference not in performance_displays:
                raise SiteDataValidationError(f"{snapshot_id}: weekly performance display is missing")
        if game.get("state") in {"future", "cancelled", "unresolved"} and game.get("score") is not None:
            raise SiteDataValidationError(f"{snapshot_id}: non-completed weekly game reveals a score")


def _validate_team_season_artifact(
    artifact: dict[str, Any],
    metadata: dict[str, Any],
    rankings: list[dict[str, Any]],
    *,
    anchor_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate provenance, cutoff redaction, and compact game-rating fields."""
    snapshot_id = str(metadata["snapshot_id"])
    _require_supported_methodology_version(
        artifact.get("schema_version"),
        SUPPORTED_ARTIFACT_SCHEMA_VERSIONS["team_season"],
        snapshot_id=snapshot_id,
        field="team-season schema",
    )
    if artifact.get("artifact_kind") != "team_season":
        raise SiteDataValidationError(f"{snapshot_id}: invalid team-season artifact kind")
    source_metadata = anchor_metadata or metadata
    schedule_source = artifact.get("schedule_source")
    if not isinstance(schedule_source, dict) or (
        schedule_source.get("kind") != "current_processed_schedule"
        or schedule_source.get("path") != "data/processed/cfbd/games.csv"
        or not isinstance(schedule_source.get("sha256"), str)
        or len(schedule_source.get("sha256", "")) != 64
        or not all(
            character in "0123456789abcdefABCDEF"
            for character in schedule_source.get("sha256", "")
        )
    ):
        raise SiteDataValidationError(f"{snapshot_id}: schedule provenance is missing or invalid")
    for field in (
        "season", "snapshot_type", "requested_cutoff", "effective_cutoff",
        "game_corpus_sha256", "source_retrieved_at", "source_retrieval_times",
        "source_response_hashes",
    ):
        if artifact.get(field) != source_metadata.get(field):
            raise SiteDataValidationError(
                f"{snapshot_id}: team-season provenance mismatch: {field}"
            )
    if anchor_metadata is not None:
        for field in (
            "season", "snapshot_type", "requested_cutoff", "effective_cutoff",
            "game_corpus_sha256", "source_retrieved_at", "source_retrieval_times",
            "source_response_hashes", "included_game_ids",
        ):
            if metadata.get(field) != anchor_metadata.get(field):
                raise SiteDataValidationError(
                    f"{snapshot_id}: Context/team-season evidence mismatch: {field}"
                )
    expected_ids = {str(value) for value in source_metadata.get("included_game_ids", [])}
    actual_ids = [str(value) for value in artifact.get("included_game_ids", [])]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise SiteDataValidationError(f"{snapshot_id}: team-season included game IDs mismatch")
    if artifact.get("anchor_family") != "context":
        raise SiteDataValidationError(f"{snapshot_id}: game ratings must be Context anchored")
    source_context_id = artifact.get("source_context_snapshot_id")
    if not isinstance(source_context_id, str) or not source_context_id:
        raise SiteDataValidationError(f"{snapshot_id}: team-season Context source is missing")
    if source_context_id != str(source_metadata["snapshot_id"]):
        raise SiteDataValidationError(f"{snapshot_id}: team-season Context source ID mismatch")
    by_team = artifact.get("teams")
    if not isinstance(by_team, dict):
        raise SiteDataValidationError(f"{snapshot_id}: team-season teams must be an object")
    ranking_ids = {str(row["team_id"]) for row in rankings}
    if not ranking_ids <= set(by_team):
        raise SiteDataValidationError(f"{snapshot_id}: team-season is missing an FBS team")
    rating_fields = (
        "rank_count", "expected_rank", "median_rank", "mode_rank", "interval_50", "interval_80",
        "interval_95", "top5_probability", "top10_probability", "top25_probability",
    )
    performance_display = artifact.get("performance_axis") is not None
    ratings_count = 0
    for team_id in ranking_ids:
        team = by_team[team_id]
        games = team.get("games")
        if not isinstance(games, list):
            raise SiteDataValidationError(f"{snapshot_id}: games for {team_id} must be a list")
        for game in games:
            if not isinstance(game, dict) or not game.get("game_id"):
                raise SiteDataValidationError(f"{snapshot_id}: invalid team-season game for {team_id}")
            try:
                datetime.fromisoformat(str(game.get("date")))
            except (TypeError, ValueError) as error:
                raise SiteDataValidationError(
                    f"{snapshot_id}: invalid team-season date for {team_id}"
                ) from error
            game_state = game.get("game_state")
            if game_state is not None and game_state not in {
                "completed",
                "future",
                "unresolved",
                "cancelled",
                "out_of_scope",
            }:
                raise SiteDataValidationError(
                    f"{snapshot_id}: unsupported game state for {team_id}"
                )
            known_by_snapshot = str(game["game_id"]) in expected_ids
            if not known_by_snapshot and (
                game.get("result") is not None
                or game.get("score") is not None
                or game.get("game_rating") is not None
                or game.get("modeled")
            ):
                raise SiteDataValidationError(
                    f"{snapshot_id}: game {game['game_id']} has result or rating without snapshot evidence"
                )
            if game_state == "future" and (
                game.get("result") is not None or game.get("score") is not None
            ):
                raise SiteDataValidationError(
                    f"{snapshot_id}: future game {game['game_id']} reveals completed evidence"
                )
            rating = game.get("game_rating")
            if rating is None:
                continue
            ratings_count += 1
            if not game.get("modeled"):
                raise SiteDataValidationError(
                    f"{snapshot_id}: ineligible game {game['game_id']} has a rating"
                )
            missing_rating = [field for field in rating_fields if field not in rating]
            if missing_rating:
                raise SiteDataValidationError(
                    f"{snapshot_id}: game rating missing {missing_rating}"
                )
            if rating["rank_count"] != artifact.get("rank_count"):
                raise SiteDataValidationError(
                    f"{snapshot_id}: game rating rank support disagrees with artifact"
                )
            for field in rating_fields:
                value = rating[field]
                values = value if isinstance(value, list) else [value]
                if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in values):
                    raise SiteDataValidationError(
                        f"{snapshot_id}: game rating {field} must be finite"
                    )
            for field in ("top5_probability", "top10_probability", "top25_probability"):
                if not 0 <= rating[field] <= 1:
                    raise SiteDataValidationError(
                        f"{snapshot_id}: game rating {field} must be between 0 and 1"
                    )
            if performance_display:
                _validate_display_masses(
                    rating.get("display_pmf"),
                    expected_length=PERFORMANCE_DISPLAY_BINS,
                    label=f"{snapshot_id}: game rating display",
                )
                percentile = rating.get("performance_percentile")
                if not isinstance(percentile, (int, float)) or not math.isfinite(percentile) or not 0 <= percentile <= 100:
                    raise SiteDataValidationError(
                        f"{snapshot_id}: game rating performance percentile is invalid"
                    )
                if rating.get("performance_grade") not in {"A", "B", "C", "D", "F"}:
                    raise SiteDataValidationError(
                        f"{snapshot_id}: game rating performance grade is invalid"
                    )
                if rating["performance_grade"] != _performance_grade_for_percentile(percentile):
                    raise SiteDataValidationError(
                        f"{snapshot_id}: game rating performance grade disagrees with percentile"
                    )
    _validate_performance_display(artifact, snapshot_id, ratings_count)
    _validate_future_predictions(
        artifact,
        metadata,
        source_metadata,
        by_team,
        expected_ids,
    )
    _validate_season_simulation(artifact, source_metadata, rankings)
    adapted = dict(artifact)
    adapted["snapshot_id"] = snapshot_id
    adapted["season"] = metadata["season"]
    adapted["snapshot_type"] = metadata["snapshot_type"]
    return adapted


def _team_season_artifact(
    *,
    source: Path,
    metadata: dict[str, Any],
    rankings: list[dict[str, Any]],
    context_source: tuple[Path, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Load ratings and the family-specific future-prediction artifact.

    Completed-game ratings remain Context-anchored for compatibility with the
    existing team-page contract.  A History artifact, when available, is
    therefore merged onto the Context schedule only for its canonical future
    prediction map and references.  Performance continues to consume the
    same-slot Context artifact for both purposes.
    """

    def candidate_for(path: Path, source_metadata: dict[str, Any]) -> Path:
        return path / str(source_metadata.get("team_season_path") or "team_seasons.json")

    def load_candidate(
        candidate: Path,
        validation_metadata: dict[str, Any],
        *,
        anchor_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifact = _read_json(candidate)
        return _validate_team_season_artifact(
            artifact,
            validation_metadata,
            rankings,
            anchor_metadata=anchor_metadata,
        )

    is_performance = metadata.get("ranking_family") == "performance"
    is_history = metadata.get("ranking_family") == "predictive" and metadata.get(
        "prior_family"
    ) == "history"
    selected_candidate = candidate_for(source, metadata)

    if is_performance:
        if context_source is None:
            raise SiteDataValidationError(
                f"{metadata['snapshot_id']}: no paired Context snapshot for team-season ratings"
            )
        context_path, context_metadata = context_source
        selected_candidate = candidate_for(context_path, context_metadata)
        if not selected_candidate.is_file():
            if metadata.get("team_season_path"):
                raise SiteDataValidationError(
                    f"{metadata['snapshot_id']}: declared team-season artifact is unavailable"
                )
            return _empty_team_season_artifact(metadata, rankings)
        return load_candidate(
            selected_candidate,
            metadata,
            anchor_metadata=context_metadata,
        )

    if is_history and selected_candidate.is_file() and context_source is not None:
        # Validate the History prediction artifact against its own posterior
        # provenance before merging it with the Context-anchored ratings.
        history_artifact = load_candidate(selected_candidate, metadata)
        context_path, context_metadata = context_source
        context_candidate = candidate_for(context_path, context_metadata)
        if not context_candidate.is_file():
            raise SiteDataValidationError(
                f"{metadata['snapshot_id']}: paired Context team-season artifact is unavailable"
            )
        context_artifact = load_candidate(
            context_candidate,
            metadata,
            anchor_metadata=context_metadata,
        )
        merged = dict(context_artifact)
        merged["prediction_schema_version"] = history_artifact.get(
            "prediction_schema_version"
        )
        merged["prediction_source"] = history_artifact.get("prediction_source")
        merged["prediction_provenance"] = history_artifact.get("prediction_provenance")
        merged["future_predictions"] = history_artifact.get("future_predictions", {})
        merged["season_simulation"] = history_artifact.get("season_simulation")
        history_teams = history_artifact.get("teams", {})
        merged_teams: dict[str, Any] = {}
        for team_id, context_team in context_artifact.get("teams", {}).items():
            history_games = {
                str(game.get("game_id")): game
                for game in history_teams.get(team_id, {}).get("games", [])
            }
            team = dict(context_team)
            team["games"] = []
            for context_game in context_team.get("games", []):
                game = dict(context_game)
                history_game = history_games.get(str(game.get("game_id")))
                game["future_prediction_id"] = (
                    history_game.get("future_prediction_id") if history_game else None
                )
                team["games"].append(game)
            merged_teams[team_id] = team
        merged["teams"] = merged_teams
        return merged

    if selected_candidate.is_file():
        return load_candidate(selected_candidate, metadata)
    if context_source is not None:
        context_path, context_metadata = context_source
        context_candidate = candidate_for(context_path, context_metadata)
        if context_candidate.is_file():
            return load_candidate(
                context_candidate,
                metadata,
                anchor_metadata=context_metadata,
            )
    declared_path = metadata.get("team_season_path")
    if declared_path:
        raise SiteDataValidationError(
            f"{metadata['snapshot_id']}: declared team-season artifact is unavailable"
        )
    return _empty_team_season_artifact(metadata, rankings)


def _rendered_team_identities(artifact: dict[str, Any]) -> set[tuple[str, str]]:
    """Collect every team identity that the team-season pages can render."""
    identities: set[tuple[str, str]] = set()
    for team in artifact.get("teams", {}).values():
        team_id = str(team.get("team_id", ""))
        team_name = str(team.get("team_name", ""))
        if team_id and team_name:
            identities.add((team_id, team_name))
        for game in team.get("games", []):
            opponent_id = str(game.get("opponent_id", ""))
            opponent_name = str(game.get("opponent_name", ""))
            if opponent_id and opponent_name:
                identities.add((opponent_id, opponent_name))
    return identities


def _logo_audit(identities: set[tuple[str, str]]) -> dict[str, Any]:
    """Summarize exact logo coverage for a set of rendered identities."""
    missing = [
        {"team_id": team_id, "team_name": team_name}
        for team_id, team_name in sorted(identities)
        if team_logo_handle(team_id, team_name) is None
    ]
    return {
        "mapped": len(identities) - len(missing),
        "total": len(identities),
        "missing": missing,
    }


def _logo_handles(identities: set[tuple[str, str]]) -> dict[str, str]:
    """Build the manifest's stable-ID-to-handle map without guessing names."""
    names_by_id: defaultdict[str, set[str]] = defaultdict(set)
    for team_id, team_name in identities:
        names_by_id[team_id].add(team_name)
    handles: dict[str, str] = {}
    for team_id, names in names_by_id.items():
        if len(names) != 1:
            continue
        handle = team_logo_handle(team_id, next(iter(names)))
        if handle is not None:
            handles[team_id] = handle
    return handles


def build_site_data(*, root: Path, config_path: Path, output_directory: Path) -> dict[str, Any]:
    """Validate configured artifacts and write deterministic consumer JSON."""
    selected, default_slot = load_publish_config(config_path, root)
    logo_url_template = _logo_url_template(config_path)
    manifest_entries: list[dict[str, Any]] = []
    prepared_snapshots: list[PreparedSnapshot] = []
    seen_ids: set[str] = set()
    seen_publications: set[tuple[int, str, str, str]] = set()
    published_fbs_identities: set[tuple[str, str]] = set()
    rendered_team_identities: set[tuple[str, str]] = set()
    schedule_path = root / "data/processed/cfbd/games.csv"
    conference_maps: dict[int, dict[tuple[int, str], str]] = {}
    metadata_by_source = {
        selected_snapshot.source: _read_json(selected_snapshot.source / "metadata.json")
        for selected_snapshot in selected
    }
    context_sources: dict[tuple[int, str], tuple[Path, dict[str, Any]]] = {}
    for selected_snapshot in selected:
        metadata = metadata_by_source[selected_snapshot.source]
        if (
            metadata.get("ranking_family") == "predictive"
            and metadata.get("prior_family") == "context"
        ):
            context_sources[(metadata["season"], selected_snapshot.publication_slot)] = (
                selected_snapshot.source,
                metadata,
            )
    for selected_snapshot in selected:
        source = selected_snapshot.source
        metadata = metadata_by_source[source]
        _validate_metadata(metadata, source)
        if metadata["ranking_family"] == "performance":
            _validate_performance_source(metadata, root)
        snapshot_id = str(metadata["snapshot_id"])
        if snapshot_id in seen_ids:
            raise SiteDataValidationError(f"Duplicate published snapshot ID: {snapshot_id}")
        seen_ids.add(snapshot_id)
        publication = (
            metadata["season"],
            metadata["ranking_family"],
            metadata.get("prior_family", metadata.get("anchor_family")),
            selected_snapshot.publication_slot,
        )
        if publication in seen_publications:
            raise SiteDataValidationError(
                f"Duplicate published logical snapshot: {selected_snapshot.publication_slot}"
            )
        seen_publications.add(publication)
        rankings = _ranking_rows(source / "rankings.csv", metadata)
        published_fbs_identities.update(
            (str(row["team_id"]), str(row["team_name"])) for row in rankings
        )
        season = metadata["season"]
        conference_map = conference_maps.get(season)
        if conference_map is None:
            conference_map = build_fbs_conference_map(schedule_path, seasons={season})
            conference_maps[season] = conference_map
        for row in rankings:
            conference = conference_map.get((season, row["team_id"]))
            if conference is None:
                raise SiteDataValidationError(
                    f"{snapshot_id}: no nonblank FBS conference evidence for team "
                    f"{row['team_id']} in season {season}"
                )
            row["conference"] = conference
        distribution = _distribution_artifact(source / "posterior_pmfs.csv", metadata, rankings)
        team_seasons = _team_season_artifact(
            source=source,
            metadata=metadata,
            rankings=rankings,
            context_source=context_sources.get((season, selected_snapshot.publication_slot)),
        )
        weekly_games = build_weekly_game_artifact(
            team_seasons,
            {**metadata, "display_label": selected_snapshot.display_label},
        )
        _validate_weekly_game_artifact(weekly_games, team_seasons, metadata)
        rendered_team_identities.update(_rendered_team_identities(team_seasons))
        records = _records(source / "included_games.csv")
        for row in rankings:
            row["record"] = records.get(row["team_id"], "0-0")
            summary = distribution["teams"][row["team_id"]]["summary"]
            row.update(
                {
                    "mode_rank": summary["modal_rank"],
                    "interval_50": summary["interval_50"],
                    "interval_95": summary["interval_95"],
                    "interval_widths": summary["interval_widths"],
                    "rank_1_probability": summary["rank_1_probability"],
                    "top5_probability": summary["top5_probability"],
                    "top10_probability": summary["top10_probability"],
                    "top25_probability": summary["top25_probability"],
                }
            )
        prepared_snapshots.append(
            PreparedSnapshot(
                selected_snapshot,
                metadata,
                snapshot_id,
                rankings,
                distribution,
                team_seasons,
                records,
                weekly_games,
            )
        )
    methodology = _validate_artifact_methodology_versions(prepared_snapshots)
    for prepared in prepared_snapshots:
        _apply_rank_changes(
            prepared,
            _previous_official_snapshot(prepared, prepared_snapshots),
        )
    for prepared in prepared_snapshots:
        selected_snapshot = prepared.selected
        metadata = prepared.metadata
        snapshot_id = prepared.snapshot_id
        rankings = prepared.rankings
        distribution = prepared.distribution
        team_seasons = prepared.team_seasons
        weekly_games = prepared.weekly_games or build_weekly_game_artifact(
            team_seasons, metadata
        )
        _validate_weekly_game_artifact(weekly_games, team_seasons, metadata)
        relative_data_path = f"data/snapshots/{snapshot_id}.json"
        relative_distribution_path = f"data/distributions/{snapshot_id}.json"
        relative_team_seasons_path = f"data/team-seasons/{snapshot_id}.json"
        relative_weekly_games_path = f"data/week-games/{snapshot_id}.json"
        previous = _previous_official_snapshot(prepared, prepared_snapshots)
        consumer_snapshot = {
            "schema_version": SITE_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "season": metadata["season"],
            "snapshot_type": metadata["snapshot_type"],
            "ranking_family": metadata["ranking_family"],
            "publication_slot": selected_snapshot.publication_slot,
            "publication_status": selected_snapshot.publication_status,
            "requested_cutoff": metadata.get("requested_cutoff"),
            "effective_cutoff": metadata.get("effective_cutoff"),
            "generation_timestamp": metadata["generation_timestamp"],
            "source_retrieved_at": metadata.get("source_retrieved_at"),
            "included_game_count": metadata.get("included_game_count", 0),
            "excluded_lower_division_games": metadata.get("excluded_lower_division_games", 0),
            "model_versions": metadata.get(
                "model_versions", {"performance": metadata.get("model_version", "unknown")}
            ),
            "team_seasons_path": relative_team_seasons_path,
            "rank_count": distribution["rank_count"],
            "rated_count": metadata.get("rated_count", sum(row["rated"] for row in rankings)),
            "unrated_count": metadata.get(
                "unrated_count", sum(not row["rated"] for row in rankings)
            ),
            "comparison_snapshot_id": previous.snapshot_id if previous is not None else None,
            "comparison_display_label": (
                previous.selected.display_label if previous is not None else None
            ),
            "rankings": rankings,
        }
        if metadata["ranking_family"] != "performance":
            consumer_snapshot["prior_family"] = metadata["prior_family"]
        else:
            consumer_snapshot.update(
                {
                    "model_version": metadata["model_version"],
                    "method": metadata["method"],
                    "anchor_family": metadata["anchor_family"],
                    "source_context_snapshot_id": metadata["source_context_snapshot_id"],
                }
            )
        _write_json(output_directory / "snapshots" / f"{snapshot_id}.json", consumer_snapshot)
        _write_json(
            output_directory / "distributions" / f"{snapshot_id}.json",
            distribution,
            compact=True,
        )
        browser_team_seasons = _browser_team_season_artifact(team_seasons)
        simulation = team_seasons.get("season_simulation")
        if isinstance(simulation, dict):
            simulation_bytes = len(
                json.dumps(simulation, separators=(",", ":"), sort_keys=True).encode(
                    "utf-8"
                )
            )
            browser_simulation = browser_team_seasons["season_simulation"]
            browser_simulation_bytes = len(
                json.dumps(
                    browser_simulation, separators=(",", ":"), sort_keys=True
                ).encode("utf-8")
            )
            team_summary_sizes = [
                len(json.dumps(summary, separators=(",", ":"), sort_keys=True).encode("utf-8"))
                for summary in simulation.get("teams", {}).values()
                if isinstance(summary, dict)
            ]
            game_marginal_bytes = len(
                json.dumps(
                    simulation.get("game_marginals", {}),
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            )
            event_decomposition_bytes = sum(
                len(
                    json.dumps(
                        summary.get("event_probability_decomposition", {}),
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                )
                for summary in simulation.get("teams", {}).values()
                if isinstance(summary, dict)
            )
        else:
            simulation_bytes = 0
            browser_simulation_bytes = 0
            team_summary_sizes = []
            game_marginal_bytes = 0
            event_decomposition_bytes = 0
        _write_json(
            output_directory / "team-seasons" / f"{snapshot_id}.json",
            browser_team_seasons,
            compact=True,
        )
        _write_json(
            output_directory / "week-games" / f"{snapshot_id}.json",
            weekly_games,
            compact=True,
        )
        future_predictions = team_seasons.get("future_predictions", {})
        future_prediction_bytes = len(
            json.dumps(
                future_predictions,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        completed_displays = {
            f"{team_id}:{game['game_id']}": game["game_rating"]["display_pmf"]
            for team_id, team in team_seasons.get("teams", {}).items()
            for game in team.get("games", [])
            if (game.get("game_rating") or {}).get("display_pmf") is not None
        }
        completed_visualization_bytes = len(
            json.dumps(
                completed_displays,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        future_displays = {
            str(game_id): prediction.get("display_distribution")
            for game_id, prediction in future_predictions.items()
            if prediction.get("display_distribution") is not None
        }
        future_visualization_bytes = len(
            json.dumps(
                future_displays,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        manifest_entry = {
            "season": metadata["season"],
            "snapshot_id": snapshot_id,
            "snapshot_type": metadata["snapshot_type"],
            "ranking_family": metadata["ranking_family"],
            "publication_slot": selected_snapshot.publication_slot,
            "publication_status": selected_snapshot.publication_status,
            "requested_cutoff": metadata.get("requested_cutoff"),
            "effective_cutoff": metadata.get("effective_cutoff"),
            "generation_timestamp": metadata["generation_timestamp"],
            "source_retrieved_at": metadata.get("source_retrieved_at"),
            "included_game_count": metadata.get("included_game_count", 0),
            "excluded_lower_division_games": metadata.get(
                "excluded_lower_division_games", 0
            ),
            "display_label": selected_snapshot.display_label,
            "comparison_snapshot_id": previous.snapshot_id if previous is not None else None,
            "comparison_display_label": (
                previous.selected.display_label if previous is not None else None
            ),
            "data_path": relative_data_path,
            "distribution_path": relative_distribution_path,
            "team_seasons_path": relative_team_seasons_path,
            "week_games_path": relative_weekly_games_path,
            "default_week": weekly_games.get("default_week"),
            "team_seasons_bytes": (
                output_directory / "team-seasons" / f"{snapshot_id}.json"
            ).stat().st_size,
            "week_games_bytes": (
                output_directory / "week-games" / f"{snapshot_id}.json"
            ).stat().st_size,
            "week_game_count": weekly_games["game_count"],
            "week_count": weekly_games["week_count"],
            "season_simulation_bytes": simulation_bytes,
            "season_simulation_browser_bytes": browser_simulation_bytes,
            "season_simulation_incremental_lazy_bytes": browser_simulation_bytes,
            "season_simulation_average_team_summary_bytes": (
                sum(team_summary_sizes) / len(team_summary_sizes)
                if team_summary_sizes
                else 0
            ),
            "season_simulation_game_marginals_bytes": game_marginal_bytes,
            "season_simulation_event_decomposition_bytes": event_decomposition_bytes,
            "future_prediction_count": len(future_predictions),
            "future_prediction_bytes": future_prediction_bytes,
            "completed_game_count": len(completed_displays),
            "completed_visualization_bytes": completed_visualization_bytes,
            "completed_visualization_bytes_per_game": (
                completed_visualization_bytes / len(completed_displays)
                if completed_displays
                else 0
            ),
            "future_visualization_bytes": future_visualization_bytes,
            "future_visualization_bytes_per_game": (
                future_visualization_bytes / len(future_displays)
                if future_displays
                else 0
            ),
            "rank_count": distribution["rank_count"],
            "model_versions": metadata.get(
                "model_versions", {"performance": metadata.get("model_version", "unknown")}
            ),
            "rated_count": metadata.get("rated_count", sum(row["rated"] for row in rankings)),
            "unrated_count": metadata.get(
                "unrated_count", sum(not row["rated"] for row in rankings)
            ),
            "valid": True,
        }
        if metadata["ranking_family"] != "performance":
            manifest_entry["prior_family"] = metadata["prior_family"]
        else:
            manifest_entry.update(
                {
                    "model_version": metadata["model_version"],
                    "method": metadata["method"],
                    "anchor_family": metadata["anchor_family"],
                }
            )
        manifest_entries.append(manifest_entry)
    published_families = {entry["ranking_family"] for entry in manifest_entries}
    published_fbs_logo_audit = _logo_audit(published_fbs_identities)
    rendered_logo_audit = _logo_audit(rendered_team_identities)
    ranking_snapshot_bytes = sum(
        (output_directory / entry["data_path"].removeprefix("data/")).stat().st_size
        for entry in manifest_entries
    )
    distribution_bytes = sum(
        (output_directory / entry["distribution_path"].removeprefix("data/")).stat().st_size
        for entry in manifest_entries
    )
    team_season_bytes = sum(int(entry["team_seasons_bytes"]) for entry in manifest_entries)
    weekly_game_bytes = sum(int(entry["week_games_bytes"]) for entry in manifest_entries)
    future_prediction_bytes = sum(
        int(entry["future_prediction_bytes"]) for entry in manifest_entries
    )
    completed_visualization_bytes = sum(
        int(entry["completed_visualization_bytes"]) for entry in manifest_entries
    )
    future_visualization_bytes = sum(
        int(entry["future_visualization_bytes"]) for entry in manifest_entries
    )
    season_simulation_bytes = sum(
        int(entry["season_simulation_bytes"]) for entry in manifest_entries
    )
    season_simulation_browser_bytes = sum(
        int(entry["season_simulation_browser_bytes"]) for entry in manifest_entries
    )
    manifest = {
        "schema_version": SITE_SCHEMA_VERSION,
        "methodology_path": "data/methodology.json",
        "methodology_schema_version": METHODOLOGY_SCHEMA_VERSION,
        "seasons": sorted({entry["season"] for entry in manifest_entries}, reverse=True),
        "ranking_families": [
            {"id": family, "label": definition["label"]}
            for family, definition in RANKING_FAMILIES.items()
            if family in published_families
        ],
        "snapshots": manifest_entries,
        "default_publication_slot": default_slot,
        "team_season_schema_version": TEAM_SEASON_SCHEMA_VERSION,
        "weekly_game_schema_version": WEEKLY_GAME_SCHEMA_VERSION,
        "team_logos": {
            "source": "RedditCFB",
            "url_template": logo_url_template,
            "handles": _logo_handles(rendered_team_identities),
            "fallback": "text",
        },
        "team_logo_audit": {
            "published_fbs": {
                key: value
                for key, value in published_fbs_logo_audit.items()
                if key != "missing"
            },
            "all_rendered_team_identities": {
                key: value
                for key, value in rendered_logo_audit.items()
                if key != "missing"
            },
            "missing": rendered_logo_audit["missing"],
        },
        "payload_stats": {
            "published_snapshot_count": len(manifest_entries),
            "published_ranking_snapshot_bytes": ranking_snapshot_bytes,
            "lazy_rank_distribution_bytes": distribution_bytes,
            "lazy_team_season_bytes": team_season_bytes,
            "lazy_week_games_bytes": weekly_game_bytes,
            "lazy_future_prediction_bytes": future_prediction_bytes,
            "lazy_completed_visualization_bytes": completed_visualization_bytes,
            "lazy_future_visualization_bytes": future_visualization_bytes,
            "season_simulation_bytes": season_simulation_bytes,
            "season_simulation_browser_bytes": season_simulation_browser_bytes,
            "initial_rankings_page_bytes": ranking_snapshot_bytes,
        },
    }
    _write_json(output_directory / "methodology.json", methodology)
    _write_json(output_directory / "manifest.json", manifest)
    return manifest
