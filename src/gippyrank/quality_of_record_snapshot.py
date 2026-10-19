"""Standalone Quality of Record snapshots from frozen Predictive Context inputs."""

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

from gippyrank.artifact_hashes import (
    included_game_rows_sha256,
    posterior_pmfs_sha256,
    stable_values_sha256,
)
from gippyrank.likelihood_provenance import supplied_likelihood_sha256
from gippyrank.methodology import (
    HISTORICAL_LIKELIHOOD_SHA256,
    HISTORICAL_LIKELIHOOD_VERSION,
    PERFORMANCE_SCHEMA_VERSION,
    PERFORMANCE_VERSION,
)
from gippyrank.performance_snapshot import (
    PERFORMANCE_METHOD,
    PERFORMANCE_RANKING_FAMILY,
    PerformanceSnapshotValidationError,
    _as_snapshot,
    _logical_snapshot_id,
    _pmfs,
    _ranking_rows,
    _read_json,
    _relative_or_absolute,
    _summary,
    validate_context_source,
    validate_performance_against_context,
)
from gippyrank.posterior.engine import LikelihoodV1
from gippyrank.posterior.snapshots import (
    Snapshot,
    _load_frozen_evidence,
    _read_included_game_rows,
    _source_inference_configuration,
    load_likelihood,
    load_pinned_likelihood,
    load_teams,
    sha256,
)
from gippyrank.quality_of_record import infer_quality_of_record

QUALITY_OF_RECORD_VARIANT = "quality_of_record"
QUALITY_OF_RECORD_EVIDENCE_MODE = "win_loss"


def _load_source_likelihood(metadata: dict[str, Any], root: Path) -> LikelihoodV1:
    if metadata.get("historical_likelihood_version") != HISTORICAL_LIKELIHOOD_VERSION:
        raise PerformanceSnapshotValidationError("Source likelihood is not frozen V1")
    expected = metadata.get("historical_likelihood_sha256")
    if not isinstance(expected, str) or not expected:
        raise PerformanceSnapshotValidationError(
            "Source likelihood identity is missing"
        )
    provenance = metadata.get("historical_likelihood_provenance")
    if provenance is None and expected == HISTORICAL_LIKELIHOOD_SHA256:
        provenance = "loaded_from_artifact"
    if provenance == "supplied_parameters":
        parameters = metadata.get("historical_likelihood_parameters")
        if (
            not isinstance(parameters, dict)
            or supplied_likelihood_sha256(parameters) != expected
        ):
            raise PerformanceSnapshotValidationError(
                "Supplied likelihood identity mismatch"
            )
        return LikelihoodV1(
            np.asarray([float.fromhex(value) for value in parameters["beta_hex"]]),
            float.fromhex(parameters["scale_hex"]),
            float.fromhex(parameters["degrees_of_freedom_hex"]),
            str(parameters["fit_kind"]),
        )
    if provenance != "loaded_from_artifact":
        raise PerformanceSnapshotValidationError("Unknown source likelihood provenance")
    path = root / "data/processed/posterior/historical_likelihood_v1.json"
    if sha256(path) != expected:
        raise PerformanceSnapshotValidationError(
            "Frozen likelihood artifact hash mismatch"
        )
    production = (
        Path(__file__).resolve().parents[2]
        / "data/processed/posterior/historical_likelihood_v1.json"
    )
    return (
        load_pinned_likelihood(path)
        if path.resolve() == production.resolve()
        else load_likelihood(path)
    )


def _retained_configuration(context: Snapshot) -> dict[str, Any]:
    if not isinstance(context.metadata.get("posterior_inference_configuration"), dict):
        diagnostics = _read_json(context.directory / "diagnostics.json")
        if (
            diagnostics.get("inference") != "deterministic_damped_loopy_sum_product"
            or diagnostics.get("converged") is not True
        ):
            raise PerformanceSnapshotValidationError(
                "Legacy Context requires retained, converged BP diagnostics"
            )
    return _source_inference_configuration(context.directory, context.metadata)


def validate_quality_of_record_against_context(
    context: Snapshot, quality_of_record: Snapshot, *, root: Path | None = None
) -> None:
    """Check the frozen source identity and binary evidence declaration."""
    validate_performance_against_context(context, quality_of_record)
    if context.metadata != _read_json(context.directory / "metadata.json"):
        raise PerformanceSnapshotValidationError(
            "Context metadata differs from its frozen artifact"
        )
    if root is not None:
        validate_context_source(context, root)
        _load_source_likelihood(context.metadata, root)
    source = context.metadata
    output = quality_of_record.metadata
    required = (
        "historical_likelihood_version",
        "historical_likelihood_sha256",
        "historical_likelihood_provenance",
        "historical_likelihood_parameters",
        "prior_pmfs_sha256",
        "included_game_rows_sha256",
        "included_game_ids_sha256",
        "posterior_inference_configuration",
    )
    identity = dict(source)
    rows = _read_included_game_rows(context.directory / "included_games.csv")
    identity.setdefault("included_game_rows_sha256", included_game_rows_sha256(rows))
    identity.setdefault(
        "included_game_ids_sha256", stable_values_sha256([row["id"] for row in rows])
    )
    identity.setdefault(
        "posterior_inference_configuration",
        _retained_configuration(context),
    )
    if source.get("historical_likelihood_sha256") == HISTORICAL_LIKELIHOOD_SHA256:
        identity.setdefault("historical_likelihood_provenance", "loaded_from_artifact")
        identity.setdefault("historical_likelihood_parameters", None)
    if "prior_pmfs_sha256" not in identity and root is not None:
        validate_context_source(context, root)
        teams, team_rows, _ = load_teams(
            root, int(source["season"]), "context", str(source["prior_model_version"])
        )
        _load_frozen_evidence(
            source=context.directory,
            season=int(source["season"]),
            snapshot_type=str(source["snapshot_type"]),
            teams=teams,
            team_rows=team_rows,
        )
        identity["prior_pmfs_sha256"] = posterior_pmfs_sha256(
            {team.team_id: team.prior for team in teams}
        )
    for field in required:
        if field not in identity or output.get(field) != identity[field]:
            raise PerformanceSnapshotValidationError(
                f"Context/Quality of Record evidence mismatch: {field}"
            )
    if output.get("performance_variant") != QUALITY_OF_RECORD_VARIANT:
        raise PerformanceSnapshotValidationError("Snapshot is not Quality of Record")
    if output.get("model") != "quality_of_record_v1":
        raise PerformanceSnapshotValidationError(
            "Quality of Record model identity mismatch"
        )
    if output.get("evidence_mode") != QUALITY_OF_RECORD_EVIDENCE_MODE:
        raise PerformanceSnapshotValidationError(
            "Quality of Record evidence must be binary"
        )
    if output.get("source_context_metadata_sha256") != sha256(
        context.directory / "metadata.json"
    ):
        raise PerformanceSnapshotValidationError("Context metadata hash mismatch")
    source_rows = _read_included_game_rows(context.directory / "included_games.csv")
    output_rows = _read_included_game_rows(
        quality_of_record.directory / "included_games.csv"
    )
    if (
        source_rows != output_rows
        or included_game_rows_sha256(output_rows) != output["included_game_rows_sha256"]
    ):
        raise PerformanceSnapshotValidationError(
            "Quality of Record game artifact mismatch"
        )
    pmfs = _pmfs(quality_of_record.directory / "posterior_pmfs.csv")
    if posterior_pmfs_sha256(pmfs) != output.get("posterior_pmfs_sha256"):
        raise PerformanceSnapshotValidationError(
            "Quality of Record PMF artifact mismatch"
        )


def build_quality_of_record_snapshot(
    source_context: Snapshot | Path,
    *,
    root: Path,
    output_root: Path | None = None,
    generation_timestamp: datetime | None = None,
    source_context_path: Path | None = None,
) -> Snapshot:
    """Rerun the frozen Context network with win/loss factors only."""
    started = time.perf_counter()
    context = _as_snapshot(source_context)
    if context.metadata != _read_json(context.directory / "metadata.json"):
        raise PerformanceSnapshotValidationError(
            "Context metadata object differs from its frozen artifact"
        )
    prior_path = validate_context_source(context, root)
    source = context.metadata
    source_dir = context.directory
    if int(source["included_game_count"]) < 1:
        raise PerformanceSnapshotValidationError(
            "Quality of Record requires at least one eligible game"
        )
    _retained_configuration(context)
    teams, team_rows, _ = load_teams(
        root, int(source["season"]), "context", str(source["prior_model_version"])
    )
    (
        _,
        games,
        included_rows,
        _,
        _,
        _,
        _,
        _,
        _,
        configuration,
    ) = _load_frozen_evidence(
        source=source_dir,
        season=int(source["season"]),
        snapshot_type=str(source["snapshot_type"]),
        teams=teams,
        team_rows=team_rows,
    )
    # Legacy publications retain the actual rows, prior hash, V1 likelihood
    # hash, and original BP diagnostics. Derive newer identity fields from
    # those frozen artifacts in memory; never rewrite the source snapshot.
    source = dict(source)
    source.setdefault("source_retrieval_times", {})
    source.setdefault(
        "included_game_rows_sha256", included_game_rows_sha256(included_rows)
    )
    source.setdefault(
        "included_game_ids_sha256",
        stable_values_sha256([row["id"] for row in included_rows]),
    )
    source.setdefault(
        "prior_pmfs_sha256",
        posterior_pmfs_sha256({team.team_id: team.prior for team in teams}),
    )
    source.setdefault("posterior_inference_configuration", configuration)
    if source.get("historical_likelihood_sha256") == HISTORICAL_LIKELIHOOD_SHA256:
        source.setdefault("historical_likelihood_provenance", "loaded_from_artifact")
        source.setdefault("historical_likelihood_parameters", None)
    ids = [row["id"] for row in included_rows]
    if (
        ids != source["included_game_ids"]
        or len(games) != int(source["included_game_count"])
        or stable_values_sha256(ids) != source["included_game_ids_sha256"]
        or included_game_rows_sha256(included_rows)
        != source["included_game_rows_sha256"]
    ):
        raise PerformanceSnapshotValidationError(
            "Frozen included games mismatch Context"
        )
    priors = {team.team_id: team.prior for team in teams}
    if posterior_pmfs_sha256(priors) != source["prior_pmfs_sha256"]:
        raise PerformanceSnapshotValidationError("Frozen team priors mismatch Context")
    if configuration != source["posterior_inference_configuration"]:
        raise PerformanceSnapshotValidationError(
            "Frozen inference configuration mismatch"
        )
    if configuration["implementation"] != "deterministic_damped_loopy_sum_product":
        raise PerformanceSnapshotValidationError("Unsupported Context inference engine")
    likelihood = _load_source_likelihood(source, root)
    result = infer_quality_of_record(
        teams,
        games,
        likelihood,
        max_iterations=int(configuration["max_iterations"]),
        tolerance=float(configuration["tolerance"]),
        damping=float(configuration["damping"]),
    )
    source_rows = _ranking_rows(context)
    if set(source_rows) != set(result.pmfs):
        raise PerformanceSnapshotValidationError(
            "Context FBS ranking population mismatch"
        )
    ranked: list[dict[str, Any]] = []
    pmf_rows: list[dict[str, Any]] = []
    for team_id, source_row in source_rows.items():
        count = result.eligible_game_counts[team_id]
        pmf = result.pmfs[team_id]
        ranked.append(
            {
                "team_id": team_id,
                "team_name": source_row["team_name"],
                "subdivision": "fbs",
                "conference": source_row.get("conference", ""),
                "record_through_cutoff": "",
                "movement_from_previous": None,
                "rated": count > 0,
                "eligible_games": count,
                "eligible_evidence_count": count,
                "games_played": count,
                **_summary(pmf),
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

    _, logical_base = _logical_snapshot_id(str(source["snapshot_id"]))
    snapshot_id = f"{logical_base}-performance-quality-of-record"
    output_root = (
        root / "data/processed/snapshots" if output_root is None else output_root
    )
    directory = (
        output_root
        / str(source["season"])
        / logical_base
        / "performance"
        / QUALITY_OF_RECORD_VARIANT
    )
    generation = generation_timestamp
    if generation is None:
        generation = datetime.fromisoformat(str(source["generation_timestamp"]))
    if generation.tzinfo is None:
        generation = generation.replace(tzinfo=UTC)
    metadata: dict[str, Any] = {
        "schema_version": PERFORMANCE_SCHEMA_VERSION,
        "season": source["season"],
        "snapshot_id": snapshot_id,
        "snapshot_type": source["snapshot_type"],
        "ranking_family": PERFORMANCE_RANKING_FAMILY,
        "performance_variant": QUALITY_OF_RECORD_VARIANT,
        "model": "quality_of_record_v1",
        "model_version": PERFORMANCE_VERSION,
        "evidence_mode": QUALITY_OF_RECORD_EVIDENCE_MODE,
        "method": PERFORMANCE_METHOD,
        "anchor_family": "context",
        "source_context_snapshot_id": source["snapshot_id"],
        "source_context_path": _relative_or_absolute(
            source_context_path or source_dir, root
        ),
        "source_context_metadata_sha256": sha256(source_dir / "metadata.json"),
        "requested_cutoff": source["requested_cutoff"],
        "effective_cutoff": source["effective_cutoff"],
        "source_retrieved_at": source["source_retrieved_at"],
        "combined_source_available_at": source.get("combined_source_available_at"),
        "source_retrieved_at_contract": source.get("source_retrieved_at_contract"),
        "source_retrieval_times": source["source_retrieval_times"],
        "source_response_hashes": source["source_response_hashes"],
        "source_evidence_hashes": source.get(
            "source_evidence_hashes", source["source_response_hashes"]
        ),
        "game_corpus_sha256": source["game_corpus_sha256"],
        "included_game_count": source["included_game_count"],
        "included_game_ids": source["included_game_ids"],
        "included_game_ids_sha256": source["included_game_ids_sha256"],
        "included_game_rows_sha256": source["included_game_rows_sha256"],
        "prior_artifact_path": _relative_or_absolute(prior_path, root),
        "prior_artifact_sha256": source["prior_artifact_sha256"],
        "prior_model_version": source["prior_model_version"],
        "prior_pmfs_sha256": source["prior_pmfs_sha256"],
        "historical_likelihood_version": source["historical_likelihood_version"],
        "historical_likelihood_sha256": source["historical_likelihood_sha256"],
        "historical_likelihood_provenance": source["historical_likelihood_provenance"],
        "historical_likelihood_parameters": source["historical_likelihood_parameters"],
        "posterior_inference_configuration": source[
            "posterior_inference_configuration"
        ],
        "binary_context_posterior_pmfs_sha256": posterior_pmfs_sha256(
            result.binary_context_posterior.pmfs
        ),
        "posterior_pmfs_sha256": posterior_pmfs_sha256(result.pmfs),
        "generation_timestamp": generation.isoformat(),
        "display_statistic": "expected_rank_rated_only",
        "rank_count": len(next(iter(result.pmfs.values()))),
        "rated_count": sum(row["rated"] for row in ranked),
        "unrated_count": sum(not row["rated"] for row in ranked),
        "valid": True,
    }
    snapshot = Snapshot(snapshot_id, directory, metadata)
    diagnostics = {
        "inference": "deterministic_damped_loopy_sum_product_binary",
        "converged": result.binary_context_posterior.converged,
        "iterations": result.binary_context_posterior.iterations,
        "max_message_delta": result.binary_context_posterior.max_message_delta,
        "raw_game_factor_count": result.binary_context_posterior.raw_game_factor_count,
        "unique_pair_factor_count": result.binary_context_posterior.unique_pair_factor_count,
        "runtime_seconds": time.perf_counter() - started,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "rankings.json").write_text(
        json.dumps(ranked, indent=2) + "\n", encoding="utf-8"
    )
    with (directory / "rankings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0]), lineterminator="\n")
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
    shutil.copyfile(source_dir / "included_games.csv", directory / "included_games.csv")
    validate_quality_of_record_against_context(context, snapshot, root=root)
    return snapshot
