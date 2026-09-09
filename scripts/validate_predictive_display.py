"""Measure the fixed-grid future-display approximation on real predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from gippyrank.posterior.engine import Team
from gippyrank.posterior.predictive import (
    ScheduledGame,
    margin_display_approximation_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


def _latest_context_source(root: Path) -> Path:
    candidates = []
    for metadata_path in root.glob("data/processed/snapshots/*/*/predictive/context/metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("ranking_family") == "predictive":
            candidates.append((str(metadata["generation_timestamp"]), metadata_path.parent))
    if not candidates:
        raise FileNotFoundError("No Predictive Context snapshot artifacts were found")
    return max(candidates)[1]


def _load_teams(source: Path) -> dict[str, Team]:
    rankings = {}
    with (source / "rankings.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rankings[row["team_id"]] = row
    pmfs: dict[str, dict[int, float]] = defaultdict(dict)
    with (source / "posterior_pmfs.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pmfs[row["team_id"]][int(row["rank"])] = float(row["probability"])
    return {
        team_id: Team(
            team_id,
            row["team_name"],
            row["subdivision"].casefold(),
            np.asarray(
                [pmfs[team_id][rank] for rank in range(1, max(pmfs[team_id]) + 1)],
                dtype=float,
            ),
        )
        for team_id, row in rankings.items()
        if team_id in pmfs
    }


def _representative_predictions(
    source: Path, limit_per_group: int
) -> list[tuple[str, dict[str, Any]]]:
    artifact = json.loads((source / "team_seasons.json").read_text(encoding="utf-8"))
    grouped: defaultdict[tuple[str, str, bool], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for prediction_id, prediction in artifact.get("future_predictions", {}).items():
        group = (
            str(prediction["home_subdivision"]),
            str(prediction["away_subdivision"]),
            bool(prediction["neutral_site"]),
        )
        grouped[group].append((str(prediction_id), prediction))
    selected = []
    for group in sorted(grouped):
        selected.extend(sorted(grouped[group])[:limit_per_group])
    return selected


def build_report(
    *, root: Path = ROOT, source: Path | None = None, limit_per_group: int = 8
) -> dict[str, Any]:
    """Return deterministic approximation metrics for representative games."""
    source = _latest_context_source(root) if source is None else source
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    likelihood_data = json.loads(
        (root / "data/processed/posterior/historical_likelihood_v1.json").read_text(
            encoding="utf-8"
        )
    )
    from gippyrank.posterior.engine import LikelihoodV1

    likelihood = LikelihoodV1(
        np.asarray(likelihood_data["beta"], dtype=float),
        float(likelihood_data["scale"]),
        float(likelihood_data["degrees_of_freedom"]),
        likelihood_data.get("fit_kind", "weighted_pseudo"),
    )
    teams = _load_teams(source)
    metrics = []
    for prediction_id, prediction in _representative_predictions(source, limit_per_group):
        game = ScheduledGame(
            game_id=prediction_id,
            home_id=str(prediction["home_team_id"]),
            away_id=str(prediction["away_team_id"]),
            home_subdivision=str(prediction["home_subdivision"]),
            away_subdivision=str(prediction["away_subdivision"]),
            neutral_site=bool(prediction["neutral_site"]),
        )
        metrics.append(
            margin_display_approximation_metrics(
                game,
                teams[game.home_id],
                teams[game.away_id],
                likelihood,
            )
        )
    if not metrics:
        raise ValueError(f"No future predictions were found in {source}")
    fields = (
        "max_absolute_cdf_error",
        "mean_absolute_cdf_error",
        "max_absolute_bin_mass_error",
        "mean_absolute_bin_mass_error",
    )
    return {
        "snapshot_id": metadata["snapshot_id"],
        "source": source.relative_to(root).as_posix(),
        "sample_count": len(metrics),
        "limit_per_matchup_group": limit_per_group,
        "metrics": {
            field: {
                "maximum": max(float(item[field]) for item in metrics),
                "mean": float(np.mean([item[field] for item in metrics])),
            }
            for field in fields
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--limit-per-group", type=int, default=8)
    arguments = parser.parse_args()
    selected_source = (
        (ROOT / arguments.source).resolve() if arguments.source else None
    )
    print(json.dumps(build_report(source=selected_source, limit_per_group=arguments.limit_per_group), indent=2, sort_keys=True))
