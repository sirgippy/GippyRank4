"""Strict loading of retained posterior PMFs for hindsight publication."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gippyrank.posterior.engine import PosteriorResult, Team
from gippyrank.posterior.snapshots import posterior_pmfs_sha256


def _equivalent_values(left: Any, right: Any) -> bool:
    """Compare published records allowing only last-bit floating-point drift."""
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(
            left, right, rel_tol=0.0, abs_tol=1e-11
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _equivalent_values(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _equivalent_values(a, b) for a, b in zip(left, right)
        )
    return left == right


def preserve_equivalent_retrospective_records(
    existing: Any, refreshed: dict[str, Any]
) -> None:
    """Keep frozen numeric spellings when recomputation is semantically equal."""
    if not isinstance(existing, dict):
        return
    for field in (
        "retrospective_game_expectations_version",
        "source_snapshot_id",
        "conditioning",
        "historical_likelihood_sha256",
        "posterior_pmfs_sha256",
    ):
        if existing.get(field) != refreshed.get(field):
            return
    old_games = existing.get("games")
    if not isinstance(old_games, dict) or old_games.keys() != refreshed["games"].keys():
        return
    for game_id, record in refreshed["games"].items():
        if _equivalent_values(old_games[game_id], record):
            refreshed["games"][game_id] = old_games[game_id]


def load_retained_posterior(
    source: Path, metadata: Mapping[str, Any], teams: Sequence[Team]
) -> PosteriorResult:
    diagnostics = json.loads((source / "diagnostics.json").read_text(encoding="utf-8"))
    if not diagnostics["converged"]:
        raise ValueError(f"{source}: production posterior did not converge")
    expected_lengths = {team.team_id: len(team.prior) for team in teams}
    pmfs: dict[str, dict[int, float]] = {}
    with (source / "posterior_pmfs.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"team_id", "rank", "probability"} <= set(reader.fieldnames or []):
            raise ValueError(f"{source}: posterior PMF columns are incomplete")
        for row in reader:
            team_id = row["team_id"]
            if team_id not in expected_lengths:
                raise ValueError(f"{source}: unexpected posterior team {team_id}")
            rank = int(row["rank"])
            value = float(row["probability"])
            if rank < 1 or rank > expected_lengths[team_id] or not np.isfinite(value) or value < 0:
                raise ValueError(f"{source}: invalid posterior rank or probability for {team_id}")
            ranks = pmfs.setdefault(team_id, {})
            if rank in ranks:
                raise ValueError(f"{source}: duplicate posterior rank {team_id}/{rank}")
            ranks[rank] = value
    if set(pmfs) != set(expected_lengths):
        raise ValueError(f"{source}: posterior team coverage differs from the prior")
    ordered = {}
    for team_id, ranks in pmfs.items():
        if set(ranks) != set(range(1, expected_lengths[team_id] + 1)):
            raise ValueError(f"{source}: incomplete posterior ranks for {team_id}")
        pmf = np.asarray([ranks[rank] for rank in sorted(ranks)])
        if not np.isclose(pmf.sum(), 1.0, rtol=0.0, atol=1e-8):
            raise ValueError(f"{source}: posterior PMF is not normalized for {team_id}")
        ordered[team_id] = pmf
    if posterior_pmfs_sha256(ordered) != metadata.get("posterior_pmfs_sha256"):
        raise ValueError(f"{source}: frozen posterior PMFs disagree with metadata")
    return PosteriorResult(
        pmfs=ordered,
        converged=True,
        iterations=int(diagnostics["iterations"]),
        max_message_delta=float(diagnostics["max_message_delta"]),
        objective=float(diagnostics["objective_surrogate"]),
        raw_game_factor_count=int(diagnostics["raw_game_likelihood_count"]),
        unique_pair_factor_count=int(diagnostics["unique_pairwise_factor_count"]),
        max_team_degree=int(diagnostics["maximum_team_degree"]),
    )
