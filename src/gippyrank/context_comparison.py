"""Matched Context 1.2/1.3 snapshot construction and parity checks."""

from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from gippyrank.posterior.snapshots import build_snapshot

WEEK4_COMPARISON_FIELDS = (
    "season",
    "snapshot_type",
    "requested_cutoff",
    "effective_cutoff",
    "historical_likelihood_version",
    "included_game_count",
    "included_game_ids",
    "game_corpus_sha256",
    "season_simulation_configuration",
)


class ContextComparisonError(ValueError):
    """Raised when the controlled 1.2/1.3 evidence boundary is not matched."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContextComparisonError(f"{path} must contain an object")
    return value


def _team_keys(path: Path) -> set[str]:
    with (path / "rankings.csv").open(newline="", encoding="utf-8") as handle:
        return {row["team_id"] for row in csv.DictReader(handle) if row["subdivision"] == "fbs"}


def _included_game_ids(path: Path) -> list[str]:
    with (path / "included_games.csv").open(newline="", encoding="utf-8") as handle:
        return [row["id"] for row in csv.DictReader(handle)]


def validate_week4_pair(
    context_1_2: Path, context_1_3: Path
) -> dict[str, Any]:
    """Assert that a Week 4 pair differs only in its Context prior."""
    first = _read_json(context_1_2 / "metadata.json")
    second = _read_json(context_1_3 / "metadata.json")
    mismatches = {
        field: (first.get(field), second.get(field))
        for field in WEEK4_COMPARISON_FIELDS
        if first.get(field) != second.get(field)
    }
    first_games = _included_game_ids(context_1_2)
    second_games = _included_game_ids(context_1_3)
    if first_games != second_games:
        mismatches["included_games.csv"] = (first_games, second_games)
    first_teams = _team_keys(context_1_2)
    second_teams = _team_keys(context_1_3)
    if first_teams != second_teams:
        mismatches["fbs_team_keys"] = (sorted(first_teams), sorted(second_teams))
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
        "context_1_2_snapshot_id": first["snapshot_id"],
        "context_1_3_snapshot_id": second["snapshot_id"],
        "team_keys_sha256": _stable_digest(sorted(first_teams)),
        "included_game_ids": first_games,
        "included_game_count": len(first_games),
        "requested_cutoff": first.get("requested_cutoff"),
        "effective_cutoff": first.get("effective_cutoff"),
        "historical_likelihood_version": first.get("historical_likelihood_version"),
        "posterior_implementation": "deterministic_damped_loopy_sum_product",
        "prior_model_versions": ["1.2", "1.3"],
        "prior_artifact_sha256": {
            "1.2": first["prior_artifact_sha256"],
            "1.3": second["prior_artifact_sha256"],
        },
    }


def _stable_digest(values: list[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


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
    result["context_1_2_path"] = str(first.directory)
    result["context_1_3_path"] = str(second.directory)
    return result


__all__ = [
    "ContextComparisonError",
    "build_official_week4_pair",
    "validate_week4_pair",
]
