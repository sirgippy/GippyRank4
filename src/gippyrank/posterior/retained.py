"""Strict loading of retained posterior PMFs for hindsight publication."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gippyrank.posterior.engine import PosteriorResult, Team
from gippyrank.posterior.snapshots import posterior_pmfs_sha256


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
