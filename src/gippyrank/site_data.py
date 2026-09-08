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
from datetime import datetime
from pathlib import Path
from typing import Any

from gippyrank.team_logos import TEAM_LOGO_URL_TEMPLATE, logo_url, team_logo_handle

SITE_SCHEMA_VERSION = "1.0"
TEAM_SEASON_SCHEMA_VERSION = "1.0"
SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS = {"1.0"}
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
        if metadata["model_version"] != "1.0":
            raise SiteDataValidationError(f"{metadata['snapshot_id']}: unsupported model version")
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
        "historical_likelihood_version": "V1",
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


def _validate_team_season_artifact(
    artifact: dict[str, Any],
    metadata: dict[str, Any],
    rankings: list[dict[str, Any]],
    *,
    anchor_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate provenance, cutoff redaction, and compact game-rating fields."""
    snapshot_id = str(metadata["snapshot_id"])
    if artifact.get("schema_version") != TEAM_SEASON_SCHEMA_VERSION:
        raise SiteDataValidationError(f"{snapshot_id}: unsupported team-season schema")
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
            rating = game.get("game_rating")
            if rating is None:
                continue
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
    """Load the Context artifact for any published family in a slot."""
    candidate = source / str(metadata.get("team_season_path") or "team_seasons.json")
    anchor_metadata = metadata
    requires_context = metadata.get("ranking_family") == "performance" or (
        metadata.get("ranking_family") == "predictive"
        and metadata.get("prior_family") == "history"
    )
    if requires_context and context_source is None:
        raise SiteDataValidationError(
            f"{metadata['snapshot_id']}: no paired Context snapshot for team-season ratings"
        )
    if requires_context:
        # Performance snapshots intentionally copy the Context artifact, so
        # validate the underlying source ID against the paired Context bundle.
        # History also consumes Context-anchored game ratings by contract.
        if context_source is not None:
            candidate, anchor_metadata = context_source
            candidate = candidate / str(anchor_metadata.get("team_season_path") or "team_seasons.json")
    elif not candidate.is_file() and context_source is not None:
        candidate, anchor_metadata = context_source
        candidate = candidate / str(anchor_metadata.get("team_season_path") or "team_seasons.json")
    if not candidate.is_file():
        declared_path = metadata.get("team_season_path") or (
            anchor_metadata.get("team_season_path") if anchor_metadata else None
        )
        if declared_path:
            raise SiteDataValidationError(
                f"{metadata['snapshot_id']}: declared team-season artifact is unavailable"
            )
        return _empty_team_season_artifact(metadata, rankings)
    artifact = _read_json(candidate)
    artifact = _validate_team_season_artifact(
        artifact,
        metadata,
        rankings,
        anchor_metadata=anchor_metadata,
    )
    return artifact


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
            )
        )
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
        relative_data_path = f"data/snapshots/{snapshot_id}.json"
        relative_distribution_path = f"data/distributions/{snapshot_id}.json"
        relative_team_seasons_path = f"data/team-seasons/{snapshot_id}.json"
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
        _write_json(
            output_directory / "team-seasons" / f"{snapshot_id}.json",
            team_seasons,
            compact=True,
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
            "team_seasons_bytes": (
                output_directory / "team-seasons" / f"{snapshot_id}.json"
            ).stat().st_size,
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
    manifest = {
        "schema_version": SITE_SCHEMA_VERSION,
        "seasons": sorted({entry["season"] for entry in manifest_entries}, reverse=True),
        "ranking_families": [
            {"id": family, "label": definition["label"]}
            for family, definition in RANKING_FAMILIES.items()
            if family in published_families
        ],
        "snapshots": manifest_entries,
        "default_publication_slot": default_slot,
        "team_season_schema_version": TEAM_SEASON_SCHEMA_VERSION,
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
        },
    }
    _write_json(output_directory / "manifest.json", manifest)
    return manifest
