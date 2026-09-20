"""Matched Context 1.2/1.3 snapshot construction and parity checks."""

from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from gippyrank.posterior.snapshots import (
    INCLUDED_GAME_FIELDS,
    build_snapshot,
    filter_games,
    included_game_rows_sha256,
    relative_path,
    stable_values_sha256,
)

EXPECTED_HISTORICAL_LIKELIHOOD_VERSION = "V1"
EXPECTED_POSTERIOR_IMPLEMENTATION = "deterministic_damped_loopy_sum_product"
EXPECTED_BACKFILL_KIND = "retrospective_prior_replay"

# These fields define the evidence and inference surface that must be identical
# for a controlled Context-prior comparison. Prior version and prior hash are
# intentionally checked separately below as the only allowed differences.
WEEK4_COMPARISON_FIELDS = (
    "season",
    "snapshot_type",
    "requested_cutoff",
    "effective_cutoff",
    "source_mode",
    "source_kind",
    "source_retrieved_at",
    "source_retrieval_times",
    "source_response_hashes",
    "historical_likelihood_version",
    "included_game_count",
    "included_game_ids",
    "included_game_ids_sha256",
    "game_corpus_sha256",
    "fbs_team_count",
    "fbs_team_keys_sha256",
    "posterior_inference_configuration",
    "season_simulation_schema_version",
    "season_simulation_version",
    "season_simulation_configuration",
    "excluded_lower_division_games",
    "fcs_fallback_team_ids",
    "fcs_fallback_count",
    "fcs_population_size",
    "fcs_population_source",
    "fcs_fallback_kind",
    "fcs_fallback_pmf_semantics",
    "lower_division_handling",
)


class ContextComparisonError(ValueError):
    """Raised when the controlled 1.2/1.3 evidence boundary is not matched."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContextComparisonError(f"{path} must contain an object")
    return value


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _team_keys(path: Path) -> set[str]:
    return {
        row["team_id"]
        for row in _read_rows(path / "rankings.csv")
        if row["subdivision"].casefold() == "fbs"
    }


def _included_game_ids(path: Path) -> list[str]:
    return [row["id"] for row in _read_rows(path / "included_games.csv")]


def _canonical_included_rows(path: Path) -> list[dict[str, str]]:
    rows = _read_rows(path / "included_games.csv")
    return [
        {field: str(row.get(field, "")) for field in INCLUDED_GAME_FIELDS}
        for row in rows
    ]


def _missing_fields(metadata: dict[str, Any]) -> list[str]:
    return [field for field in WEEK4_COMPARISON_FIELDS if field not in metadata]


def _derived_parity_metadata(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Return parity fields, deriving them for retained pre-parity snapshots."""
    rows = _canonical_included_rows(path)
    game_ids = [row["id"] for row in rows]
    team_keys = _team_keys(path)
    configuration = metadata.get("posterior_inference_configuration")
    if not isinstance(configuration, dict):
        diagnostics_path = path / "diagnostics.json"
        diagnostics = (
            _read_json(diagnostics_path) if diagnostics_path.is_file() else {}
        )
        configuration = {
            "implementation": diagnostics.get(
                "inference", EXPECTED_POSTERIOR_IMPLEMENTATION
            ),
            "max_iterations": 500,
            "tolerance": 1e-9,
            "damping": 0.35,
        }
    lower = metadata.get("lower_division_handling")
    if not isinstance(lower, dict):
        lower = {
            "excluded_lower_division_games": metadata.get(
                "excluded_lower_division_games", 0
            ),
            "fcs_fallback_team_ids": metadata.get("fcs_fallback_team_ids", []),
            "fcs_fallback_count": metadata.get("fcs_fallback_count", 0),
            "fcs_population_size": metadata.get("fcs_population_size"),
            "fcs_population_source": metadata.get("fcs_population_source"),
            "fcs_fallback_kind": metadata.get("fcs_fallback_kind"),
            "fcs_fallback_pmf_semantics": metadata.get(
                "fcs_fallback_pmf_semantics"
            ),
        }
    return {
        **metadata,
        "included_game_count": len(rows),
        "included_game_ids": game_ids,
        "included_game_ids_sha256": stable_values_sha256(game_ids),
        "included_game_rows_sha256": included_game_rows_sha256(rows),
        "fbs_team_count": len(team_keys),
        "fbs_team_keys_sha256": stable_values_sha256(sorted(team_keys)),
        "posterior_inference_configuration": configuration,
        "lower_division_handling": lower,
    }


def _declared_metadata_mismatches(
    path: Path, metadata: dict[str, Any], label: str
) -> dict[str, Any]:
    """Reject a snapshot whose declared parity fields disagree with artifacts."""
    rows = _canonical_included_rows(path)
    game_ids = [row["id"] for row in rows]
    team_keys = _team_keys(path)
    expected = {
        "included_game_count": len(rows),
        "included_game_ids": game_ids,
        "included_game_ids_sha256": stable_values_sha256(game_ids),
        "included_game_rows_sha256": included_game_rows_sha256(rows),
        "fbs_team_count": len(team_keys),
        "fbs_team_keys_sha256": stable_values_sha256(sorted(team_keys)),
    }
    mismatches: dict[str, Any] = {}
    for field, expected_value in expected.items():
        if field not in metadata:
            continue
        actual = metadata[field]
        if field == "included_game_ids":
            actual = [str(value) for value in actual] if isinstance(actual, list) else actual
        if actual != expected_value:
            mismatches[f"{label}.{field}"] = (actual, expected_value)
    lower = metadata.get("lower_division_handling")
    if isinstance(lower, dict):
        for field in (
            "excluded_lower_division_games",
            "fcs_fallback_team_ids",
            "fcs_fallback_count",
            "fcs_population_size",
            "fcs_population_source",
            "fcs_fallback_kind",
            "fcs_fallback_pmf_semantics",
        ):
            if field in metadata and lower.get(field) != metadata[field]:
                mismatches[f"{label}.lower_division_handling.{field}"] = (
                    lower.get(field),
                    metadata[field],
                )
    return mismatches


def _validate_snapshot_internal_consistency(
    path: Path, metadata: dict[str, Any], label: str
) -> tuple[list[str], set[str]]:
    game_ids = _included_game_ids(path)
    team_keys = _team_keys(path)
    mismatches: dict[str, Any] = {}
    if metadata.get("included_game_count") != len(game_ids):
        mismatches["included_game_count"] = (
            metadata.get("included_game_count"),
            len(game_ids),
        )
    if metadata.get("included_game_ids") != game_ids:
        mismatches["included_game_ids_metadata"] = True
    if metadata.get("included_game_ids_sha256") != stable_values_sha256(game_ids):
        mismatches["included_game_ids_sha256"] = True
    if metadata.get("fbs_team_count") != len(team_keys):
        mismatches["fbs_team_count"] = (metadata.get("fbs_team_count"), len(team_keys))
    if metadata.get("fbs_team_keys_sha256") != stable_values_sha256(sorted(team_keys)):
        mismatches["fbs_team_keys_sha256"] = True
    if mismatches:
        raise ContextComparisonError(
            f"{label} metadata does not match its canonical artifacts: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return game_ids, team_keys


def validate_week4_pair(
    context_1_2: Path, context_1_3: Path
) -> dict[str, Any]:
    """Assert that a Week 4 pair differs only in its Context prior."""
    first = _read_json(context_1_2 / "metadata.json")
    second = _read_json(context_1_3 / "metadata.json")
    missing = {
        label: fields
        for label, metadata in (("context_1_2", first), ("context_1_3", second))
        if (fields := _missing_fields(metadata))
    }
    if missing:
        raise ContextComparisonError(
            "official Week 4 Context snapshots are missing parity metadata: "
            + json.dumps(missing, sort_keys=True)
        )

    first_games, first_teams = _validate_snapshot_internal_consistency(
        context_1_2, first, "Context 1.2"
    )
    second_games, second_teams = _validate_snapshot_internal_consistency(
        context_1_3, second, "Context 1.3"
    )
    mismatches = {
        field: (first.get(field), second.get(field))
        for field in WEEK4_COMPARISON_FIELDS
        if first.get(field) != second.get(field)
    }
    if first_games != second_games:
        mismatches["included_games.csv"] = (first_games, second_games)
    if first_teams != second_teams:
        mismatches["fbs_team_keys"] = (sorted(first_teams), sorted(second_teams))
    if first.get("historical_likelihood_version") != EXPECTED_HISTORICAL_LIKELIHOOD_VERSION:
        mismatches["historical_likelihood_version_expected"] = (
            first.get("historical_likelihood_version"),
        )
    for label, metadata in (("context_1_2", first), ("context_1_3", second)):
        configuration = metadata.get("posterior_inference_configuration", {})
        if configuration.get("implementation") != EXPECTED_POSTERIOR_IMPLEMENTATION:
            mismatches[f"{label}_posterior_implementation"] = configuration.get(
                "implementation"
            )
    if first.get("prior_model_version") != "1.2":
        mismatches["context_1_2_prior_model_version"] = first.get("prior_model_version")
    if second.get("prior_model_version") != "1.3":
        mismatches["context_1_3_prior_model_version"] = second.get("prior_model_version")
    if first.get("prior_artifact_sha256") == second.get("prior_artifact_sha256"):
        mismatches["prior_artifact_sha256"] = "the two priors must be distinct"
    if mismatches:
        raise ContextComparisonError(
            "official Week 4 Context snapshots are not matched: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "comparison": "context_1_2_vs_context_1_3",
        "season": first["season"],
        "boundary": "official_week_4",
        "parity_validation": "passed",
        "context_1_2_snapshot_id": first["snapshot_id"],
        "context_1_3_snapshot_id": second["snapshot_id"],
        "fbs_team_count": first["fbs_team_count"],
        "fbs_team_keys_sha256": first["fbs_team_keys_sha256"],
        "team_keys_sha256": first["fbs_team_keys_sha256"],
        "included_game_ids": first_games,
        "included_game_count": len(first_games),
        "included_game_ids_sha256": first["included_game_ids_sha256"],
        "requested_cutoff": first["requested_cutoff"],
        "effective_cutoff": first["effective_cutoff"],
        "game_corpus_sha256": first["game_corpus_sha256"],
        "historical_likelihood_version": first["historical_likelihood_version"],
        "posterior_inference_configuration": first[
            "posterior_inference_configuration"
        ],
        "season_simulation_schema_version": first["season_simulation_schema_version"],
        "season_simulation_version": first["season_simulation_version"],
        "season_simulation_configuration": first[
            "season_simulation_configuration"
        ],
        "lower_division_handling": first["lower_division_handling"],
        "prior_model_versions": ["1.2", "1.3"],
        "prior_artifact_sha256": {
            "1.2": first["prior_artifact_sha256"],
            "1.3": second["prior_artifact_sha256"],
        },
    }


BACKFILL_PARITY_FIELDS = (
    "season",
    "snapshot_type",
    "requested_cutoff",
    "effective_cutoff",
    "source_mode",
    "source_kind",
    "source_retrieved_at",
    "source_retrieval_times",
    "source_response_hashes",
    "historical_likelihood_version",
    "included_game_count",
    "included_game_ids",
    "included_game_ids_sha256",
    "included_game_rows_sha256",
    "game_corpus_sha256",
    "fbs_team_count",
    "fbs_team_keys_sha256",
    "posterior_inference_configuration",
    "season_simulation_schema_version",
    "season_simulation_version",
    "season_simulation_configuration",
    "excluded_lower_division_games",
    "fcs_fallback_team_ids",
    "fcs_fallback_count",
    "fcs_population_size",
    "fcs_population_source",
    "fcs_fallback_kind",
    "fcs_fallback_pmf_semantics",
    "lower_division_handling",
)


def validate_context_backfill(
    source_context_1_2: Path, context_1_3_backfill: Path
) -> dict[str, Any]:
    """Fail closed unless a 1.3 replay changes only the prior.

    The source snapshot may predate the parity metadata added for the Week 4
    comparison.  Its actual CSV/JSON artifacts are therefore canonicalized
    first; no parity fact is accepted merely because it was copied into a
    generated metadata file.
    """
    source_raw = _read_json(source_context_1_2 / "metadata.json")
    replay_raw = _read_json(context_1_3_backfill / "metadata.json")
    mismatches = {}
    mismatches.update(
        _declared_metadata_mismatches(
            source_context_1_2, source_raw, "source"
        )
    )
    mismatches.update(
        _declared_metadata_mismatches(
            context_1_3_backfill, replay_raw, "backfill"
        )
    )
    source = _derived_parity_metadata(source_context_1_2, source_raw)
    replay = _derived_parity_metadata(context_1_3_backfill, replay_raw)
    source_rows = _canonical_included_rows(source_context_1_2)
    replay_rows = _canonical_included_rows(context_1_3_backfill)
    mismatches.update({
        field: (source.get(field), replay.get(field))
        for field in BACKFILL_PARITY_FIELDS
        if source.get(field) != replay.get(field)
    })
    if source_rows != replay_rows:
        mismatches["included_games.csv"] = "frozen source rows differ"
    if source.get("historical_likelihood_version") != EXPECTED_HISTORICAL_LIKELIHOOD_VERSION:
        mismatches["historical_likelihood_version_expected"] = source.get(
            "historical_likelihood_version"
        )
    for label, metadata in (("source", source), ("backfill", replay)):
        implementation = metadata["posterior_inference_configuration"].get(
            "implementation"
        )
        if implementation != EXPECTED_POSTERIOR_IMPLEMENTATION:
            mismatches[f"{label}_posterior_implementation"] = implementation
    if source.get("prior_model_version") != "1.2":
        mismatches["source_prior_model_version"] = source.get("prior_model_version")
    if replay.get("prior_model_version") != "1.3":
        mismatches["backfill_prior_model_version"] = replay.get("prior_model_version")
    if replay.get("snapshot_id") != f"{source['snapshot_id']}-v1.3":
        mismatches["backfill_snapshot_id"] = replay.get("snapshot_id")
    if replay.get("prior_lineage") != "context_1_3_reconstructed_2026":
        mismatches["backfill_prior_lineage"] = replay.get("prior_lineage")
    if source.get("prior_artifact_sha256") == replay.get("prior_artifact_sha256"):
        mismatches["prior_artifact_sha256"] = "the source and replay priors must differ"
    if replay.get("backfill") is not True:
        mismatches["backfill"] = replay.get("backfill")
    if replay.get("backfill_kind") != EXPECTED_BACKFILL_KIND:
        mismatches["backfill_kind"] = replay.get("backfill_kind")
    if replay.get("source_evidence_snapshot_id") != source.get("snapshot_id"):
        mismatches["source_evidence_snapshot_id"] = replay.get(
            "source_evidence_snapshot_id"
        )
    if replay.get("source_evidence_game_corpus_sha256") != source.get(
        "game_corpus_sha256"
    ):
        mismatches["source_evidence_game_corpus_sha256"] = replay.get(
            "source_evidence_game_corpus_sha256"
        )
    if replay.get("source_evidence_included_game_count") != len(source_rows):
        mismatches["source_evidence_included_game_count"] = replay.get(
            "source_evidence_included_game_count"
        )
    if replay.get("source_evidence_included_game_ids_sha256") != source.get(
        "included_game_ids_sha256"
    ):
        mismatches["source_evidence_included_game_ids_sha256"] = replay.get(
            "source_evidence_included_game_ids_sha256"
        )
    expected_hash = included_game_rows_sha256(source_rows)
    if replay.get("source_evidence_hash") != expected_hash:
        mismatches["source_evidence_hash"] = replay.get("source_evidence_hash")
    if replay.get("source_evidence_included_game_rows_sha256") != expected_hash:
        mismatches["source_evidence_included_game_rows_sha256"] = replay.get(
            "source_evidence_included_game_rows_sha256"
        )
    if mismatches:
        raise ContextComparisonError(
            "Context 1.3 backfill is not an exact frozen-evidence replay: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "comparison": "context_1_2_source_vs_context_1_3_backfill",
        "parity_validation": "passed",
        "source_context_1_2_snapshot_id": source["snapshot_id"],
        "context_1_3_backfill_snapshot_id": replay["snapshot_id"],
        "source_context_1_2_path": source_context_1_2.as_posix(),
        "context_1_3_backfill_path": context_1_3_backfill.as_posix(),
        "season": source["season"],
        "requested_cutoff": source["requested_cutoff"],
        "effective_cutoff": source["effective_cutoff"],
        "included_game_count": len(source_rows),
        "included_game_ids_sha256": source["included_game_ids_sha256"],
        "included_game_rows_sha256": expected_hash,
        "game_corpus_sha256": source["game_corpus_sha256"],
        "fbs_team_count": source["fbs_team_count"],
        "fbs_team_keys_sha256": source["fbs_team_keys_sha256"],
        "historical_likelihood_version": source[
            "historical_likelihood_version"
        ],
        "posterior_inference_configuration": source[
            "posterior_inference_configuration"
        ],
        "lower_division_handling": source["lower_division_handling"],
        "prior_model_versions": ["1.2", "1.3"],
        "prior_artifact_sha256": {
            "1.2": source["prior_artifact_sha256"],
            "1.3": replay["prior_artifact_sha256"],
        },
    }


def build_context_backfill(
    *,
    source_context_1_2: Path,
    root: Path,
    output_root: Path | None = None,
    generation_timestamp: datetime | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Build and validate one retrospective Context 1.3 weekly replay."""
    source_metadata = _read_json(source_context_1_2 / "metadata.json")
    if (
        source_metadata.get("prior_family") != "context"
        or source_metadata.get("prior_model_version") != "1.2"
        or source_metadata.get("snapshot_type") != "weekly"
    ):
        raise ContextComparisonError(
            f"{source_context_1_2}: expected a weekly Context 1.2 source snapshot"
        )
    cutoff_value = source_metadata.get("requested_cutoff")
    if cutoff_value is None:
        raise ContextComparisonError(f"{source_context_1_2}: source cutoff is missing")
    cutoff = datetime.fromisoformat(str(cutoff_value))
    generated = build_snapshot(
        root=root,
        output_root=output_root or root / "data/processed/snapshots",
        season=int(source_metadata["season"]),
        cutoff=cutoff,
        prior_family="context",
        prior_model_version="1.3",
        snapshot_type="weekly",
        generation_timestamp=generation_timestamp,
        evidence_snapshot=source_context_1_2,
    )
    validation = validate_context_backfill(
        source_context_1_2, generated.directory
    )
    return generated.directory, validation


def _boundary_sanity_check(
    *, root: Path, snapshot_path: Path, result: dict[str, Any]
) -> dict[str, Any]:
    metadata = _read_json(snapshot_path / "metadata.json")
    effective_cutoff = datetime.fromisoformat(str(metadata["effective_cutoff"]))
    expected_games, expected_rows, expected_lower, _ = filter_games(
        root, 2026, effective_cutoff, "weekly"
    )
    expected_ids = [row["id"] for row in expected_rows]
    actual_ids = list(result["included_game_ids"])
    if actual_ids != expected_ids:
        raise ContextComparisonError(
            "official Week 4 boundary does not reproduce the eligible game corpus"
        )
    included_rows = _read_rows(snapshot_path / "included_games.csv")
    week4_ids = [row["id"] for row in included_rows if int(row["week"]) >= 4]
    if week4_ids:
        raise ContextComparisonError(
            "official Week 4 boundary unexpectedly includes Week 4 games: "
            + json.dumps(week4_ids)
        )
    week3_expected = {
        row["id"] for row in expected_rows if int(row["week"]) == 3
    }
    week3_present = week3_expected.intersection(actual_ids)
    missing_week3 = sorted(week3_expected - week3_present)
    if missing_week3:
        raise ContextComparisonError(
            "official Week 4 boundary is missing eligible Week 3 games: "
            + json.dumps(missing_week3)
        )
    if expected_lower != metadata["excluded_lower_division_games"]:
        raise ContextComparisonError(
            "official Week 4 lower-division exclusion count is not reproducible"
        )
    return {
        "requested_cutoff": result["requested_cutoff"],
        "effective_cutoff": result["effective_cutoff"],
        "week_3_eligible_game_count": len(week3_expected),
        "week_3_included_game_count": len(week3_present),
        "week_3_missing_game_ids": missing_week3,
        "week_4_included_game_ids": week4_ids,
        "week_4_eligible_game_count": sum(
            int(row["week"]) == 4 for row in expected_rows
        ),
        "included_game_count": len(expected_games),
        "excluded_lower_division_games": expected_lower,
        "lower_division_handling_match": True,
        "exact_included_game_set_match": True,
    }


def build_official_week4_pair(
    *,
    root: Path,
    cutoff: datetime | date,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Build both posteriors independently from the same official cutoff."""
    output_root = output_root or root / "data/processed/snapshots"
    first = build_snapshot(
        root=root,
        output_root=output_root,
        season=2026,
        cutoff=cutoff,
        prior_family="context",
        prior_model_version="1.2",
        snapshot_type="weekly",
    )
    second = build_snapshot(
        root=root,
        output_root=output_root,
        season=2026,
        cutoff=cutoff,
        prior_family="context",
        prior_model_version="1.3",
        snapshot_type="weekly",
    )
    result = validate_week4_pair(first.directory, second.directory)
    result["context_1_2_path"] = relative_path(first.directory, root)
    result["context_1_3_path"] = relative_path(second.directory, root)
    result["boundary_sanity"] = _boundary_sanity_check(
        root=root,
        snapshot_path=first.directory,
        result=result,
    )
    return result


__all__ = [
    "ContextComparisonError",
    "build_context_backfill",
    "build_official_week4_pair",
    "validate_context_backfill",
    "validate_week4_pair",
]
