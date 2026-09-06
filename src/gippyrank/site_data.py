"""Build compact, static website data from explicitly selected snapshot bundles.

This module is intentionally independent of posterior inference.  It reads
already-valid snapshot artifacts and produces the small summaries a browser
needs; it neither imports nor invokes model code.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SITE_SCHEMA_VERSION = "1.0"
SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS = {"1.0"}
PMF_SUM_TOLERANCE = 1e-9
SUMMARY_TOLERANCE = 1e-8
RANKING_FAMILIES: dict[str, dict[str, str]] = {
    "predictive": {"label": "Predictive"},
}


class SiteDataValidationError(ValueError):
    """Raised when a snapshot is not safe to publish."""


@dataclass(frozen=True)
class PublishedSnapshot:
    source: Path
    display_label: str
    publication_slot: str


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SiteDataValidationError(f"Cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise SiteDataValidationError(f"{path} must contain a JSON object")
    return value


def load_publish_config(path: Path, root: Path) -> tuple[list[PublishedSnapshot], str | None]:
    """Load an explicit, ordered list of snapshot directories to publish."""
    config = _read_json(path)
    if config.get("schema_version") != SITE_SCHEMA_VERSION:
        raise SiteDataValidationError("Unsupported publish configuration schema")
    snapshots = config.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise SiteDataValidationError("Publish configuration needs a non-empty snapshots list")
    selected: list[PublishedSnapshot] = []
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
        source_path = root / source
        if not source_path.is_dir():
            raise SiteDataValidationError(f"Selected snapshot directory does not exist: {source}")
        selected.append(PublishedSnapshot(source_path, label, slot))
    default_slot = config.get("default_publication_slot")
    if default_slot is not None:
        if not isinstance(default_slot, str) or not default_slot:
            raise SiteDataValidationError("default_publication_slot must be a non-empty string")
        if default_slot not in {item.publication_slot for item in selected}:
            raise SiteDataValidationError("default_publication_slot is not a published slot")
    return selected, default_slot


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
        display_rank = int(_finite_number(row["display_rank"], "display_rank", snapshot_id))
        if display_rank < 1 or display_rank in ranks:
            raise SiteDataValidationError(f"{snapshot_id}: duplicate or invalid FBS display rank")
        ranks.add(display_rank)
        low = _finite_number(row["interval_80_low"], "interval_80_low", snapshot_id)
        high = _finite_number(row["interval_80_high"], "interval_80_high", snapshot_id)
        if low > high:
            raise SiteDataValidationError(f"{snapshot_id}: interval_80_low exceeds interval_80_high")
        normalized.append(
            {
                "display_rank": display_rank,
                "team_id": team_id,
                "team_name": row["team_name"],
                "conference": row["conference"],
                "expected_rank": _finite_number(row["expected_rank"], "expected_rank", snapshot_id),
                "median_rank": _finite_number(row["median_rank"], "median_rank", snapshot_id),
                "interval_80": [low, high],
                "top5_probability": _probability(row["top5_probability"], "top5_probability", snapshot_id),
                "top10_probability": _probability(row["top10_probability"], "top10_probability", snapshot_id),
                "top25_probability": _probability(row["top25_probability"], "top25_probability", snapshot_id),
            }
        )
    expected_ranks = set(range(1, len(normalized) + 1))
    if ranks != expected_ranks:
        raise SiteDataValidationError(
            f"{snapshot_id}: FBS display ranks must be contiguous from 1 through "
            f"{len(normalized)}"
        )
    return sorted(normalized, key=lambda row: row["display_rank"])


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

    return {
        "schema_version": SITE_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "rank_count": rank_count,
        "teams": teams,
    }


def _validate_metadata(metadata: dict[str, Any], source: Path) -> None:
    required = {
        "schema_version", "season", "snapshot_id", "snapshot_type", "ranking_family",
        "prior_family", "valid", "generation_timestamp", "model_versions",
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
    if metadata["prior_family"] not in {"context", "history"}:
        raise SiteDataValidationError(f"{metadata['snapshot_id']}: unsupported prior family")


def _write_json(path: Path, value: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, separators=(",", ":"), sort_keys=True)
        if compact
        else json.dumps(value, indent=2, sort_keys=True)
    )
    path.write_text(encoded + "\n", encoding="utf-8")


def build_site_data(*, root: Path, config_path: Path, output_directory: Path) -> dict[str, Any]:
    """Validate configured artifacts and write deterministic consumer JSON."""
    selected, default_slot = load_publish_config(config_path, root)
    manifest_entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_publications: set[tuple[int, str, str, str]] = set()
    for selected_snapshot in selected:
        source = selected_snapshot.source
        metadata = _read_json(source / "metadata.json")
        _validate_metadata(metadata, source)
        snapshot_id = str(metadata["snapshot_id"])
        if snapshot_id in seen_ids:
            raise SiteDataValidationError(f"Duplicate published snapshot ID: {snapshot_id}")
        seen_ids.add(snapshot_id)
        publication = (
            metadata["season"],
            metadata["ranking_family"],
            metadata["prior_family"],
            selected_snapshot.publication_slot,
        )
        if publication in seen_publications:
            raise SiteDataValidationError(
                f"Duplicate published logical snapshot: {selected_snapshot.publication_slot}"
            )
        seen_publications.add(publication)
        rankings = _ranking_rows(source / "rankings.csv", metadata)
        distribution = _distribution_artifact(source / "posterior_pmfs.csv", metadata, rankings)
        records = _records(source / "included_games.csv")
        for row in rankings:
            row["record"] = records.get(row["team_id"], "0-0")
        relative_data_path = f"data/snapshots/{snapshot_id}.json"
        relative_distribution_path = f"data/distributions/{snapshot_id}.json"
        consumer_snapshot = {
            "schema_version": SITE_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "season": metadata["season"],
            "snapshot_type": metadata["snapshot_type"],
            "ranking_family": metadata["ranking_family"],
            "prior_family": metadata["prior_family"],
            "publication_slot": selected_snapshot.publication_slot,
            "requested_cutoff": metadata.get("requested_cutoff"),
            "effective_cutoff": metadata.get("effective_cutoff"),
            "generation_timestamp": metadata["generation_timestamp"],
            "source_retrieved_at": metadata.get("source_retrieved_at"),
            "included_game_count": metadata.get("included_game_count", 0),
            "excluded_lower_division_games": metadata.get("excluded_lower_division_games", 0),
            "model_versions": metadata["model_versions"],
            "rank_count": distribution["rank_count"],
            "rankings": rankings,
        }
        _write_json(output_directory / "snapshots" / f"{snapshot_id}.json", consumer_snapshot)
        _write_json(
            output_directory / "distributions" / f"{snapshot_id}.json",
            distribution,
            compact=True,
        )
        manifest_entries.append(
            {
                "season": metadata["season"],
                "snapshot_id": snapshot_id,
                "snapshot_type": metadata["snapshot_type"],
                "ranking_family": metadata["ranking_family"],
                "prior_family": metadata["prior_family"],
                "publication_slot": selected_snapshot.publication_slot,
                "requested_cutoff": metadata.get("requested_cutoff"),
                "effective_cutoff": metadata.get("effective_cutoff"),
                "generation_timestamp": metadata["generation_timestamp"],
                "source_retrieved_at": metadata.get("source_retrieved_at"),
                "included_game_count": metadata.get("included_game_count", 0),
                "excluded_lower_division_games": metadata.get(
                    "excluded_lower_division_games", 0
                ),
                "display_label": selected_snapshot.display_label,
                "data_path": relative_data_path,
                "distribution_path": relative_distribution_path,
                "rank_count": distribution["rank_count"],
                "model_versions": metadata["model_versions"],
                "valid": True,
            }
        )
    published_families = {entry["ranking_family"] for entry in manifest_entries}
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
    }
    _write_json(output_directory / "manifest.json", manifest)
    return manifest
