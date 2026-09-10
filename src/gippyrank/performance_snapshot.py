"""Production Performance V1 snapshots derived from Context snapshots.

This module deliberately performs a cheap, auditable transformation.  It does
not rerun posterior inference and it never calls the research-only explicit
focal-prior neutralization baseline.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from gippyrank.methodology import (
    PERFORMANCE_SCHEMA_VERSION,
    PERFORMANCE_VERSION,
    TEAM_SEASON_SCHEMA_VERSION,
)
from gippyrank.posterior.snapshots import Snapshot, load_teams, sha256

PERFORMANCE_MODEL_VERSION = PERFORMANCE_VERSION
PERFORMANCE_RANKING_FAMILY = "performance"
PERFORMANCE_METHOD = "prior_stripping"
PMF_TOLERANCE = 1e-9


class PerformanceSnapshotValidationError(ValueError):
    """Raised when a production Performance source or output is unsafe."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PerformanceSnapshotValidationError(f"{path} must contain a JSON object")
    return value


def _as_snapshot(value: Snapshot | Path) -> Snapshot:
    if isinstance(value, Snapshot):
        return value
    metadata = _read_json(value / "metadata.json")
    return Snapshot(str(metadata["snapshot_id"]), value, metadata)


def _require_metadata(metadata: dict[str, Any], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if field not in metadata]
    if missing:
        raise PerformanceSnapshotValidationError(
            f"Context snapshot metadata is missing {sorted(missing)}"
        )


def validate_context_source(source_context: Snapshot, root: Path) -> Path:
    """Validate that ``source_context`` is a valid canonical Context snapshot."""
    metadata = source_context.metadata
    _require_metadata(
        metadata,
        (
            "season",
            "snapshot_id",
            "snapshot_type",
            "ranking_family",
            "prior_family",
            "valid",
            "requested_cutoff",
            "effective_cutoff",
            "source_retrieved_at",
            "source_retrieval_times",
            "source_response_hashes",
            "game_corpus_sha256",
            "included_game_ids",
            "included_game_count",
            "prior_artifact_sha256",
            "prior_model_version",
        ),
    )
    if metadata["ranking_family"] != "predictive":
        raise PerformanceSnapshotValidationError(
            "Performance production requires a Predictive Context source"
        )
    if metadata["prior_family"] != "context":
        raise PerformanceSnapshotValidationError(
            "Performance production rejects History-derived sources"
        )
    if metadata["snapshot_type"] == "preseason":
        raise PerformanceSnapshotValidationError(
            "Performance has no preseason ranking because it requires observed games"
        )
    if metadata["valid"] is not True:
        raise PerformanceSnapshotValidationError("Context source snapshot is not valid")
    if metadata["snapshot_id"] != source_context.snapshot_id:
        raise PerformanceSnapshotValidationError("Context snapshot ID does not match metadata")

    _teams, _rows, prior_path = load_teams(root, int(metadata["season"]), "context")
    actual_hash = sha256(prior_path)
    if actual_hash != metadata["prior_artifact_sha256"]:
        raise PerformanceSnapshotValidationError(
            "Context source prior artifact hash does not match the frozen Context prior"
        )
    return prior_path


def validate_performance_against_context(
    context: Snapshot, performance: Snapshot
) -> None:
    """Fail closed when Performance and Context do not share evidence."""
    context_metadata = context.metadata
    performance_metadata = performance.metadata
    if performance_metadata.get("ranking_family") != PERFORMANCE_RANKING_FAMILY:
        raise PerformanceSnapshotValidationError("Snapshot is not a Performance snapshot")
    if performance_metadata.get("anchor_family") != "context":
        raise PerformanceSnapshotValidationError("Performance anchor family must be Context")
    if performance_metadata.get("method") != PERFORMANCE_METHOD:
        raise PerformanceSnapshotValidationError("Unsupported Performance production method")
    if performance_metadata.get("source_context_snapshot_id") != context_metadata.get(
        "snapshot_id"
    ):
        raise PerformanceSnapshotValidationError("Performance source Context snapshot ID mismatch")

    for field in (
        "season",
        "snapshot_type",
        "requested_cutoff",
        "effective_cutoff",
        "source_retrieved_at",
        "source_retrieval_times",
        "source_response_hashes",
        "game_corpus_sha256",
        "included_game_ids",
        "included_game_count",
        "prior_artifact_sha256",
        "prior_model_version",
    ):
        if performance_metadata.get(field) != context_metadata.get(field):
            raise PerformanceSnapshotValidationError(
                f"Context/Performance evidence mismatch: {field}"
            )

    context_evidence = context_metadata.get(
        "source_evidence_hashes", context_metadata.get("source_response_hashes")
    )
    performance_evidence = performance_metadata.get(
        "source_evidence_hashes", performance_metadata.get("source_response_hashes")
    )
    if performance_evidence != context_evidence:
        raise PerformanceSnapshotValidationError(
            "Context/Performance evidence mismatch: source_evidence_hashes"
        )
    if not context_metadata.get("valid") or not performance_metadata.get("valid"):
        raise PerformanceSnapshotValidationError(
            "Context and Performance snapshots must both be valid"
        )


def _validated_pmf(values: list[float] | np.ndarray, label: str) -> np.ndarray:
    pmf = np.asarray(values, dtype=float)
    if pmf.ndim != 1 or not len(pmf) or not np.isfinite(pmf).all():
        raise PerformanceSnapshotValidationError(f"{label} must be finite and nonempty")
    if np.any(pmf < 0):
        raise PerformanceSnapshotValidationError(f"{label} contains negative probability")
    total = float(np.sum(pmf, dtype=float))
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=PMF_TOLERANCE):
        raise PerformanceSnapshotValidationError(
            f"{label} sums to {total}, not 1; refusing to renormalize"
        )
    return pmf


def strip_context_prior(posterior: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Compute the frozen ratio in log space without clipping or smoothing."""
    posterior = _validated_pmf(posterior, "Context posterior PMF")
    prior = _validated_pmf(prior, "Context prior PMF")
    if posterior.shape != prior.shape:
        raise PerformanceSnapshotValidationError(
            "Context posterior and prior PMFs have different rank support"
        )
    zero_prior_with_mass = (prior == 0) & (posterior > 0)
    if np.any(zero_prior_with_mass):
        raise PerformanceSnapshotValidationError(
            "Context prior has zero mass where the Context posterior has positive mass"
        )

    positive = (prior > 0) & (posterior > 0)
    if not np.any(positive):
        raise PerformanceSnapshotValidationError(
            "Context posterior/prior ratio has no finite positive support"
        )
    log_raw = np.full(posterior.shape, -np.inf, dtype=float)
    log_raw[positive] = np.log(posterior[positive]) - np.log(prior[positive])
    maximum = float(np.max(log_raw[positive]))
    scaled = np.zeros_like(log_raw)
    scaled[positive] = np.exp(log_raw[positive] - maximum)
    total = float(np.sum(scaled, dtype=float))
    if not math.isfinite(total) or total <= 0:
        raise PerformanceSnapshotValidationError(
            "Context posterior/prior ratio could not be normalized stably"
        )
    result = scaled / total
    if not np.isfinite(result).all() or not math.isclose(
        float(np.sum(result, dtype=float)), 1.0, rel_tol=0.0, abs_tol=PMF_TOLERANCE
    ):
        raise PerformanceSnapshotValidationError("Performance PMF normalization failed")
    return result


def _quantile(pmf: np.ndarray, probability: float) -> int:
    cumulative = np.cumsum(pmf)
    index = int(np.searchsorted(cumulative, probability, side="left"))
    return min(index + 1, len(pmf))


def _summary(pmf: np.ndarray) -> dict[str, float | int]:
    pmf = _validated_pmf(pmf, "Performance PMF")
    result: dict[str, float | int] = {
        "expected_rank": float(np.dot(np.arange(1, len(pmf) + 1), pmf)),
        "median_rank": _quantile(pmf, 0.50),
        "mode_rank": int(np.argmax(pmf) + 1),
        "rank_1_probability": float(pmf[0]),
    }
    for level, tail in ((50, 0.25), (80, 0.10), (95, 0.025)):
        low = _quantile(pmf, tail)
        high = _quantile(pmf, 1 - tail)
        result[f"interval_{level}_low"] = low
        result[f"interval_{level}_high"] = high
        result[f"interval_{level}_width"] = high - low + 1
    for threshold in (5, 10, 25):
        result[f"top{threshold}_probability"] = float(np.sum(pmf[:threshold]))
    return result


def _pmfs(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"team_id", "rank", "probability"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise PerformanceSnapshotValidationError(
                f"Context posterior PMF artifact missing {sorted(missing)}"
            )
        rows: dict[str, dict[int, float]] = {}
        for row in reader:
            team_id = row["team_id"]
            rank = int(row["rank"])
            if rank < 1 or rank in rows.setdefault(team_id, {}):
                raise PerformanceSnapshotValidationError(
                    f"Context posterior PMF has duplicate or invalid rank for {team_id}"
                )
            rows[team_id][rank] = float(row["probability"])
    result: dict[str, np.ndarray] = {}
    for team_id, values in rows.items():
        expected = set(range(1, max(values) + 1))
        if set(values) != expected:
            raise PerformanceSnapshotValidationError(
                f"Context posterior PMF ranks for {team_id} are not contiguous"
            )
        result[team_id] = _validated_pmf(
            [values[rank] for rank in range(1, max(values) + 1)],
            f"Context posterior PMF for {team_id}",
        )
    return result


def _eligible_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for side in ("home", "away"):
                if row[f"{side}Classification"].casefold() == "fbs":
                    team_id = row[f"{side}Id"]
                    counts[team_id] = counts.get(team_id, 0) + 1
    return counts


def _ranking_rows(source: Snapshot) -> dict[str, dict[str, str]]:
    with (source.directory / "rankings.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = {
            row["team_id"]: row
            for row in csv.DictReader(handle)
            if row["subdivision"].casefold() == "fbs"
        }
    if not rows:
        raise PerformanceSnapshotValidationError("Context source has no FBS ranking rows")
    return rows


def _logical_snapshot_id(source_snapshot_id: str) -> tuple[str, str]:
    if source_snapshot_id.endswith("-context"):
        base = source_snapshot_id[: -len("-context")]
    else:
        base = source_snapshot_id
    return f"{base}-performance", base


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def build_performance_snapshot(
    source_context: Snapshot | Path,
    *,
    root: Path,
    output_root: Path | None = None,
    source_context_path: Path | None = None,
    generation_timestamp: datetime | None = None,
) -> Snapshot:
    """Build a production Performance snapshot from one valid Context snapshot."""
    started = time.perf_counter()
    context = _as_snapshot(source_context)
    prior_path = validate_context_source(context, root)
    metadata = context.metadata
    source_dir = context.directory
    included_games = source_dir / "included_games.csv"
    if not included_games.is_file():
        raise PerformanceSnapshotValidationError("Context source is missing included_games.csv")
    if int(metadata["included_game_count"]) == 0:
        raise PerformanceSnapshotValidationError(
            "Performance cannot be published without at least one eligible game"
        )

    context_pmfs = _pmfs(source_dir / "posterior_pmfs.csv")
    context_rows = _ranking_rows(context)
    _teams, prior_rows, _ = load_teams(root, int(metadata["season"]), "context")
    prior_pmfs = {
        team_id: _validated_pmf(json.loads(row["pmf"]), f"Context prior for {team_id}")
        for team_id, row in prior_rows.items()
        if row.get("subdivision", "").casefold() == "fbs"
    }
    counts = _eligible_counts(included_games)
    ranked: list[dict[str, Any]] = []
    pmf_rows: list[dict[str, Any]] = []
    for team_id, source_row in context_rows.items():
        if team_id not in context_pmfs or team_id not in prior_pmfs:
            raise PerformanceSnapshotValidationError(
                f"Context source is missing a posterior or prior PMF for {team_id}"
            )
        games_played = counts.get(team_id, 0)
        pmf = (
            strip_context_prior(context_pmfs[team_id], prior_pmfs[team_id])
            if games_played
            else np.full(len(prior_pmfs[team_id]), 1 / len(prior_pmfs[team_id]))
        )
        summary = _summary(pmf)
        ranked.append(
            {
                "team_id": team_id,
                "team_name": source_row["team_name"],
                "subdivision": "fbs",
                "conference": source_row.get("conference", ""),
                "record_through_cutoff": "",
                "movement_from_previous": None,
                "rated": games_played > 0,
                "eligible_games": games_played,
                "eligible_evidence_count": games_played,
                "games_played": games_played,
                **summary,
            }
        )
        pmf_rows.extend(
            {"team_id": team_id, "rank": rank, "probability": float(probability)}
            for rank, probability in enumerate(pmf, 1)
        )

    ranked.sort(
        key=lambda row: (
            not row["rated"],
            float(row["expected_rank"]) if row["rated"] else math.inf,
            str(row["team_name"]),
            str(row["team_id"]),
        )
    )
    display_rank = 0
    for row in ranked:
        if row["rated"]:
            display_rank += 1
            row["display_rank"] = display_rank
        else:
            row["display_rank"] = "NR"

    performance_snapshot_id, logical_base = _logical_snapshot_id(
        str(metadata["snapshot_id"])
    )
    output_root = root / "data/processed/snapshots" if output_root is None else output_root
    directory = output_root / str(metadata["season"]) / logical_base / "performance"
    directory.mkdir(parents=True, exist_ok=True)
    source_path = source_context_path or source_dir
    generation = generation_timestamp
    if generation is None:
        generation = datetime.fromisoformat(str(metadata["generation_timestamp"]))
    generation = generation if generation.tzinfo is not None else generation.replace(tzinfo=UTC)

    performance_metadata: dict[str, Any] = {
        "schema_version": PERFORMANCE_SCHEMA_VERSION,
        "season": metadata["season"],
        "snapshot_id": performance_snapshot_id,
        "snapshot_type": metadata["snapshot_type"],
        "ranking_family": PERFORMANCE_RANKING_FAMILY,
        "model_version": PERFORMANCE_MODEL_VERSION,
        "method": PERFORMANCE_METHOD,
        "anchor_family": "context",
        "source_context_snapshot_id": metadata["snapshot_id"],
        "source_context_path": _relative_or_absolute(source_path, root),
        "source_context_metadata_sha256": sha256(source_dir / "metadata.json"),
        "requested_cutoff": metadata["requested_cutoff"],
        "effective_cutoff": metadata["effective_cutoff"],
        "source_retrieved_at": metadata["source_retrieved_at"],
        "source_retrieval_times": metadata["source_retrieval_times"],
        "source_response_hashes": metadata["source_response_hashes"],
        "source_evidence_hashes": metadata["source_response_hashes"],
        "game_corpus_sha256": metadata["game_corpus_sha256"],
        "included_game_count": metadata["included_game_count"],
        "included_game_ids": metadata["included_game_ids"],
        "prior_artifact_path": _relative_or_absolute(prior_path, root),
        "prior_artifact_sha256": metadata["prior_artifact_sha256"],
        "prior_model_version": metadata["prior_model_version"],
        "generation_timestamp": generation.isoformat(),
        "display_statistic": "expected_rank_rated_only",
        "rank_count": len(next(iter(prior_pmfs.values()))),
        "rated_count": sum(row["rated"] for row in ranked),
        "unrated_count": sum(not row["rated"] for row in ranked),
        "valid": True,
    }
    diagnostics = {
        "method": PERFORMANCE_METHOD,
        "source_context_snapshot_id": metadata["snapshot_id"],
        "source_context_runtime_seconds": _read_json(source_dir / "diagnostics.json").get(
            "runtime_seconds"
        ),
        "transformation_runtime_seconds": time.perf_counter() - started,
        "runtime_seconds": time.perf_counter() - started,
        "rated_count": performance_metadata["rated_count"],
        "unrated_count": performance_metadata["unrated_count"],
        "inference": "none_prior_stripping_only",
    }
    (directory / "metadata.json").write_text(
        json.dumps(performance_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "rankings.json").write_text(
        json.dumps(ranked, indent=2) + "\n", encoding="utf-8"
    )
    ranking_fields = list(ranked[0])
    with (directory / "rankings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ranking_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ranked)
    with (directory / "posterior_pmfs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["team_id", "rank", "probability"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(pmf_rows)
    shutil.copyfile(included_games, directory / "included_games.csv")
    team_seasons = source_dir / "team_seasons.json"
    if team_seasons.is_file():
        shutil.copyfile(team_seasons, directory / "team_seasons.json")
        performance_metadata["team_season_path"] = "team_seasons.json"
        performance_metadata["team_season_schema_version"] = TEAM_SEASON_SCHEMA_VERSION
        (directory / "metadata.json").write_text(
            json.dumps(performance_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return Snapshot(performance_snapshot_id, directory, performance_metadata)
