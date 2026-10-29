"""Standalone, validated Quality of Results snapshots from frozen Context inputs."""

from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from gippyrank.artifact_hashes import (
    included_game_rows_sha256,
    posterior_pmfs_sha256,
    stable_values_sha256,
)
from gippyrank.methodology import (
    HISTORICAL_LIKELIHOOD_SHA256,
    PERFORMANCE_SCHEMA_VERSION,
    SUPPORTED_ARTIFACT_MODEL_VERSIONS,
    SUPPORTED_ARTIFACT_SCHEMA_VERSIONS,
)
from gippyrank.performance_snapshot import (
    PERFORMANCE_METHOD,
    PMF_TOLERANCE,
    PerformanceSnapshotValidationError,
    _as_snapshot,
    _logical_snapshot_id,
    _pmfs,
    _read_json,
    _relative_or_absolute,
    _summary,
    _validated_pmf,
    strip_context_prior,
    validate_context_source,
    validate_performance_against_context,
)
from gippyrank.performance_v1 import evidence_counts
from gippyrank.posterior.engine import Game, LikelihoodV1, Team
from gippyrank.posterior.snapshots import (
    Snapshot,
    _load_frozen_evidence,
    _read_included_game_rows,
    load_teams,
    sha256,
)
from gippyrank.quality_of_record_snapshot import (
    _load_source_likelihood,
    _retained_configuration,
    _validate_binary_evidence_rows,
)
from gippyrank.quality_of_results import (
    QUALITY_OF_RESULTS_EVIDENCE_MODE,
    QUALITY_OF_RESULTS_MARGIN_WEIGHT,
    QUALITY_OF_RESULTS_MODEL,
    QUALITY_OF_RESULTS_VARIANT,
    QUALITY_OF_RESULTS_VERSION,
    infer_quality_of_results,
)

_ENGINE = "deterministic_damped_loopy_sum_product"
_FILES = (
    "included_games.csv",
    "posterior_pmfs.csv",
    "blended_context_posterior_pmfs.csv",
    "rankings.csv",
    "rankings.json",
    "diagnostics.json",
)
_INHERITED = (
    "season",
    "snapshot_type",
    "requested_cutoff",
    "effective_cutoff",
    "source_retrieved_at",
    "combined_source_available_at",
    "source_retrieved_at_contract",
    "source_retrieval_times",
    "source_response_hashes",
    "source_evidence_hashes",
    "game_corpus_sha256",
    "included_game_count",
    "included_game_ids",
    "included_game_ids_sha256",
    "included_game_rows_sha256",
    "prior_artifact_sha256",
    "prior_model_version",
    "prior_pmfs_sha256",
    "historical_likelihood_version",
    "historical_likelihood_sha256",
    "historical_likelihood_provenance",
    "historical_likelihood_parameters",
    "posterior_inference_configuration",
    "game_evidence_exclusions",
    "game_evidence_exclusions_sha256",
    "historical_evidence_correction",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PerformanceSnapshotValidationError(message)


def _configuration(value: dict[str, Any]) -> None:
    _require(
        set(value) == {"implementation", "max_iterations", "tolerance", "damping"}
        and value["implementation"] == _ENGINE,
        "Unsupported Context inference configuration",
    )
    _require(
        type(value["max_iterations"]) is int and value["max_iterations"] > 0,
        "Invalid inference max_iterations",
    )
    for field in ("tolerance", "damping"):
        number = value[field]
        _require(
            type(number) in (int, float)
            and math.isfinite(number)
            and (0 < number if field == "tolerance" else 0 <= number)
            and number < 1,
            f"Invalid inference {field}",
        )


def _validate_pmfs(pmfs: dict[str, np.ndarray], teams: list[Team], label: str) -> None:
    _require(
        set(pmfs) == {team.team_id for team in teams}, f"{label} population mismatch"
    )
    for team in teams:
        pmf = _validated_pmf(pmfs[team.team_id], f"{label} PMF {team.team_id}")
        _require(pmf.shape == team.prior.shape, f"{label} rank support mismatch")


def _validate_convergence(diagnostics: dict[str, Any], config: dict[str, Any]) -> None:
    _require(diagnostics.get("converged") is True, "Inference did not converge")
    iterations, delta = (
        diagnostics.get("iterations"),
        diagnostics.get("max_message_delta"),
    )
    _require(
        type(iterations) is int and 0 < iterations <= config["max_iterations"],
        "Invalid inference iteration count",
    )
    _require(
        type(delta) in (int, float)
        and math.isfinite(delta)
        and 0 <= delta <= config["tolerance"],
        "Inference convergence tolerance mismatch",
    )


@dataclass(frozen=True)
class _Source:
    snapshot: Snapshot
    identity: dict[str, Any]
    teams: list[Team]
    games: list[Game]
    rows: list[dict[str, str]]
    rankings: dict[str, dict[str, str]]
    prior_path: Path
    likelihood: LikelihoodV1
    counts: dict[str, int]


def _load_source(source_context: Snapshot | Path, root: Path) -> _Source:
    context = _as_snapshot(source_context)
    _require(
        context.metadata == _read_json(context.directory / "metadata.json"),
        "Context metadata differs from its frozen artifact",
    )
    prior_path = validate_context_source(context, root)
    source = dict(context.metadata)
    _require(
        isinstance(source["snapshot_id"], str)
        and bool(source["snapshot_id"])
        and source["snapshot_id"] not in {".", ".."}
        and "/" not in source["snapshot_id"]
        and "\\" not in source["snapshot_id"],
        "Invalid Context snapshot identity",
    )
    _require(
        source.get("schema_version") in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS["snapshot"]
        and source["prior_model_version"]
        in SUPPORTED_ARTIFACT_MODEL_VERSIONS["context_prior"],
        "Unsupported Context schema or prior version",
    )
    _require(
        int(source["included_game_count"]) > 0,
        "Quality of Results requires eligible games",
    )
    config = _retained_configuration(context)
    _configuration(config)
    source_diagnostics = _read_json(context.directory / "diagnostics.json")
    _validate_convergence(source_diagnostics, config)
    _require(
        source_diagnostics.get("converged") is True
        and source_diagnostics.get("inference") == _ENGINE,
        "Source Context inference did not converge",
    )
    teams, team_rows, _ = load_teams(
        root, int(source["season"]), "context", str(source["prior_model_version"])
    )
    _require(
        len(teams) == len({team.team_id for team in teams}), "Duplicate prior team IDs"
    )
    # Team construction has historical normalization semantics. Validate the
    # serialized prior first as well, without relaxing the snapshot PMF tolerance.
    for row in team_rows.values():
        _validated_pmf(json.loads(row["pmf"]), f"Frozen prior {row['team_id']}")
    frozen = _load_frozen_evidence(
        source=context.directory,
        season=int(source["season"]),
        snapshot_type=source["snapshot_type"],
        teams=teams,
        team_rows=team_rows,
    )
    games, rows = frozen[1], frozen[2]
    _require(frozen[-1] == config, "Frozen inference configuration mismatch")
    priors = {team.team_id: team.prior for team in teams}
    identities = {
        "included_game_rows_sha256": included_game_rows_sha256(rows),
        "included_game_ids_sha256": stable_values_sha256([row["id"] for row in rows]),
        "prior_pmfs_sha256": posterior_pmfs_sha256(priors),
        "posterior_inference_configuration": config,
    }
    # Legacy Context snapshots predate these four fields. Derive all four only
    # for that older contract; partially missing modern identities fail closed.
    legacy = not (identities.keys() & source.keys()) and source.get(
        "source_retrieved_at_contract"
    ) in {"legacy_first_response", "legacy_unverified_availability"}
    for field, expected in identities.items():
        _require(
            legacy or source.get(field) == expected, f"Frozen Context {field} mismatch"
        )
        source[field] = expected
    by_id = {team.team_id: team for team in teams}
    cutoff = datetime.fromisoformat(source["effective_cutoff"])
    for game, row in zip(games, rows, strict=True):
        _require(
            int(row["season"]) == int(source["season"])
            and row["completed"].casefold() in {"true", "1", "yes"}
            and row["neutralSite"].casefold()
            in {"true", "false", "1", "0", "yes", "no"}
            and datetime.fromisoformat(row["startDate"]) <= cutoff
            and game.home_id != game.away_id
            and game.home_points >= 0
            and game.away_points >= 0,
            f"Invalid frozen game evidence: {game.game_id}",
        )
        for team_id, subdivision in (
            (game.home_id, game.home_subdivision),
            (game.away_id, game.away_subdivision),
        ):
            _require(
                team_id in by_id and by_id[team_id].subdivision == subdivision,
                f"Game/prior identity mismatch: {game.game_id}/{team_id}",
            )
    _validate_binary_evidence_rows(rows)
    context_pmfs = _pmfs(context.directory / "posterior_pmfs.csv")
    _validate_pmfs(context_pmfs, teams, "Context posterior")
    _require(
        posterior_pmfs_sha256(context_pmfs) == source.get("posterior_pmfs_sha256"),
        "Context posterior PMF hash mismatch",
    )
    with (context.directory / "rankings.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        ranked = [
            row
            for row in csv.DictReader(handle)
            if row["subdivision"].casefold() == "fbs"
        ]
    rankings = {row["team_id"]: row for row in ranked}
    fbs = [team for team in teams if team.subdivision == "fbs"]
    _require(
        len(ranked) == len(rankings)
        and set(rankings) == {team.team_id for team in fbs},
        "Context ranking population mismatch",
    )
    _require(
        len({len(team.prior) for team in fbs}) == 1, "Inconsistent FBS rank support"
    )
    for team in fbs:
        _require(
            rankings[team.team_id]["team_name"] == team.name,
            "Context team identity mismatch",
        )
    likelihood = _load_source_likelihood(source, root)
    _require(
        math.isfinite(likelihood.scale)
        and math.isfinite(likelihood.degrees_of_freedom),
        "Invalid likelihood parameters",
    )
    if source["historical_likelihood_sha256"] == HISTORICAL_LIKELIHOOD_SHA256:
        source.setdefault("historical_likelihood_provenance", "loaded_from_artifact")
        source.setdefault("historical_likelihood_parameters", None)
    source.setdefault("source_retrieval_times", {})
    source.setdefault("source_evidence_hashes", source["source_response_hashes"])
    return _Source(
        context,
        source,
        teams,
        games,
        rows,
        rankings,
        prior_path,
        likelihood,
        evidence_counts(teams, games),
    )


def _rankings(source: _Source, pmfs: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    ranked = []
    for team_id, row in source.rankings.items():
        count = source.counts[team_id]
        ranked.append(
            {
                "team_id": team_id,
                "team_name": row["team_name"],
                "subdivision": "fbs",
                "conference": row.get("conference", ""),
                "record_through_cutoff": row.get("record_through_cutoff", ""),
                "movement_from_previous": None,
                "rated": count > 0,
                "eligible_games": count,
                "eligible_evidence_count": count,
                "games_played": count,
                **_summary(pmfs[team_id]),
            }
        )
    ranked.sort(
        key=lambda row: (
            not row["rated"],
            row["expected_rank"] if row["rated"] else math.inf,
            row["team_name"],
            row["team_id"],
        )
    )
    for index, row in enumerate(ranked, 1):
        row["display_rank"] = index if row["rated"] else "NR"
    return ranked


def _declaration(source: _Source, root: Path) -> dict[str, Any]:
    snapshot = source.snapshot
    _, base = _logical_snapshot_id(snapshot.snapshot_id)
    return {
        **{key: source.identity[key] for key in _INHERITED if key in source.identity},
        "schema_version": PERFORMANCE_SCHEMA_VERSION,
        "snapshot_id": f"{base}-performance-quality-of-results",
        "ranking_family": "performance",
        "performance_variant": QUALITY_OF_RESULTS_VARIANT,
        "model": QUALITY_OF_RESULTS_MODEL,
        "model_version": QUALITY_OF_RESULTS_VERSION,
        "margin_weight": QUALITY_OF_RESULTS_MARGIN_WEIGHT,
        "evidence_mode": QUALITY_OF_RESULTS_EVIDENCE_MODE,
        "method": PERFORMANCE_METHOD,
        "anchor_family": "context",
        "source_context_snapshot_id": snapshot.snapshot_id,
        "source_context_path": _relative_or_absolute(snapshot.directory, root),
        "source_context_metadata_sha256": sha256(snapshot.directory / "metadata.json"),
        "source_context_diagnostics_sha256": sha256(
            snapshot.directory / "diagnostics.json"
        ),
        "prior_artifact_path": _relative_or_absolute(source.prior_path, root),
        "display_statistic": "expected_rank_rated_only",
        "rank_count": len(
            next(team.prior for team in source.teams if team.subdivision == "fbs")
        ),
        "rated_count": sum(source.counts[team_id] > 0 for team_id in source.rankings),
        "unrated_count": sum(
            source.counts[team_id] == 0 for team_id in source.rankings
        ),
        "valid": True,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_pmfs(path: Path, pmfs: dict[str, np.ndarray]) -> None:
    _write_csv(
        path,
        [
            {"team_id": team_id, "rank": rank, "probability": float(probability)}
            for team_id in sorted(pmfs)
            for rank, probability in enumerate(pmfs[team_id], 1)
        ],
    )


def validate_quality_of_results_against_context(
    context: Snapshot | Path, quality_of_results: Snapshot | Path, *, root: Path
) -> None:
    """Validate retained output and its frozen source without rerunning inference.

    Both the blended network PMFs and stripped FBS PMFs are retained, allowing the
    ratio, evidence counts, and complete ranking representations to be checked.
    """
    source = _load_source(context, root)
    output = _as_snapshot(quality_of_results)
    metadata, directory = output.metadata, output.directory
    _require(metadata.get("valid") is True, "Output snapshot is not valid")
    _require(
        metadata == _read_json(directory / "metadata.json"),
        "Output metadata artifact mismatch",
    )
    validate_performance_against_context(source.snapshot, output)
    for field, expected in _declaration(source, root).items():
        _require(
            metadata.get(field) == expected, f"Quality of Results {field} mismatch"
        )
    _require(
        output.snapshot_id == metadata["snapshot_id"], "Snapshot identity mismatch"
    )
    generation = datetime.fromisoformat(metadata["generation_timestamp"])
    _require(generation.tzinfo is not None, "Generation timestamp must have a timezone")
    hashes = metadata.get("artifact_sha256")
    _require(
        isinstance(hashes, dict) and set(hashes) == set(_FILES),
        "Artifact inventory mismatch",
    )
    for name in _FILES:
        _require(
            sha256(directory / name) == hashes[name], f"Artifact hash mismatch: {name}"
        )
    _require(
        _read_included_game_rows(directory / "included_games.csv") == source.rows,
        "Quality of Results frozen game artifact mismatch",
    )
    pmfs = _pmfs(directory / "posterior_pmfs.csv")
    blended = _pmfs(directory / "blended_context_posterior_pmfs.csv")
    _validate_pmfs(
        pmfs,
        [team for team in source.teams if team.subdivision == "fbs"],
        "Performance",
    )
    _validate_pmfs(blended, source.teams, "Blended Context")
    for field, values in (
        ("posterior_pmfs_sha256", pmfs),
        ("blended_context_posterior_pmfs_sha256", blended),
    ):
        _require(
            posterior_pmfs_sha256(values) == metadata.get(field), f"{field} mismatch"
        )
    for team in source.teams:
        if team.subdivision != "fbs":
            continue
        expected = (
            strip_context_prior(blended[team.team_id], team.prior)
            if source.counts[team.team_id]
            else np.full(len(team.prior), 1 / len(team.prior))
        )
        _require(
            np.allclose(pmfs[team.team_id], expected, atol=PMF_TOLERANCE, rtol=0),
            f"Prior stripping mismatch: {team.team_id}",
        )
    rankings = _rankings(source, pmfs)
    _require(
        json.loads((directory / "rankings.json").read_text()) == rankings,
        "Ranking summaries, order or evidence counts mismatch",
    )
    with (directory / "rankings.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    expected_csv = [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in rankings
    ]
    _require(csv_rows == expected_csv, "Ranking CSV mismatch")
    diagnostics = _read_json(directory / "diagnostics.json")
    config = source.identity["posterior_inference_configuration"]
    _require(
        diagnostics.get("inference") == _ENGINE
        and diagnostics.get("model") == QUALITY_OF_RESULTS_MODEL
        and diagnostics.get("margin_weight") == QUALITY_OF_RESULTS_MARGIN_WEIGHT
        and diagnostics.get("converged") is True,
        "Quality of Results inference identity or convergence mismatch",
    )
    _validate_convergence(diagnostics, config)
    _require(
        diagnostics.get("raw_game_factor_count") == len(source.games)
        and diagnostics.get("unique_pair_factor_count")
        == len({tuple(sorted((game.home_id, game.away_id))) for game in source.games})
        and diagnostics.get("eligible_game_counts") == source.counts,
        "Inference game counts mismatch",
    )
    runtime = diagnostics.get("runtime_seconds")
    _require(
        type(runtime) in (int, float) and math.isfinite(runtime) and runtime >= 0,
        "Invalid inference runtime",
    )


def build_quality_of_results_snapshot(
    source_context: Snapshot | Path,
    *,
    root: Path,
    output_root: Path | None = None,
    generation_timestamp: datetime | None = None,
) -> Snapshot:
    """Generate and validate a new standalone λ=.40 snapshot; never acquire data."""
    started = time.perf_counter()
    source = _load_source(source_context, root)
    metadata = _declaration(source, root)
    _, base = _logical_snapshot_id(source.snapshot.snapshot_id)
    output_root = (
        output_root if output_root is not None else root / "data/processed/snapshots"
    )
    directory = (
        output_root
        / str(metadata["season"])
        / base
        / "performance"
        / QUALITY_OF_RESULTS_VARIANT
    )
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite retained snapshot: {directory}")
    config = source.identity["posterior_inference_configuration"]
    result = infer_quality_of_results(
        source.teams,
        source.games,
        source.likelihood,
        max_iterations=config["max_iterations"],
        tolerance=config["tolerance"],
        damping=config["damping"],
    )
    _require(
        result.margin_weight == QUALITY_OF_RESULTS_MARGIN_WEIGHT,
        "Inference weight mismatch",
    )
    _require(
        result.eligible_game_counts == source.counts,
        "Inference evidence counts mismatch",
    )
    posterior = result.blended_context_posterior
    _require(posterior.converged, "Quality of Results inference did not converge")
    _validate_pmfs(
        result.pmfs,
        [team for team in source.teams if team.subdivision == "fbs"],
        "Performance",
    )
    _validate_pmfs(posterior.pmfs, source.teams, "Blended Context")
    generation = generation_timestamp or datetime.now(UTC)
    _require(generation.tzinfo is not None, "Generation timestamp must have a timezone")
    metadata.update(
        {
            "generation_timestamp": generation.isoformat(),
            "posterior_pmfs_sha256": posterior_pmfs_sha256(result.pmfs),
            "blended_context_posterior_pmfs_sha256": posterior_pmfs_sha256(
                posterior.pmfs
            ),
        }
    )
    diagnostics = {
        "inference": _ENGINE,
        "model": QUALITY_OF_RESULTS_MODEL,
        "margin_weight": result.margin_weight,
        "converged": posterior.converged,
        "iterations": posterior.iterations,
        "max_message_delta": posterior.max_message_delta,
        "raw_game_factor_count": posterior.raw_game_factor_count,
        "unique_pair_factor_count": posterior.unique_pair_factor_count,
        "eligible_game_counts": result.eligible_game_counts,
        "runtime_seconds": time.perf_counter() - started,
    }
    ranked = _rankings(source, result.pmfs)
    directory.parent.mkdir(parents=True, exist_ok=True)
    # A failed validation cannot leave a seemingly valid snapshot at its final ID.
    with tempfile.TemporaryDirectory(
        prefix=".quality-of-results-", dir=directory.parent
    ) as temp:
        staging = Path(temp)
        _write_pmfs(staging / "posterior_pmfs.csv", result.pmfs)
        _write_pmfs(staging / "blended_context_posterior_pmfs.csv", posterior.pmfs)
        _write_csv(staging / "rankings.csv", ranked)
        for name, value in (
            ("diagnostics.json", diagnostics),
            ("rankings.json", ranked),
        ):
            (staging / name).write_text(
                json.dumps(value, indent=2) + "\n", encoding="utf-8"
            )
        shutil.copyfile(
            source.snapshot.directory / "included_games.csv",
            staging / "included_games.csv",
        )
        metadata["artifact_sha256"] = {name: sha256(staging / name) for name in _FILES}
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_quality_of_results_against_context(source.snapshot, staging, root=root)
        staging.rename(directory)
    return Snapshot(metadata["snapshot_id"], directory, metadata)
