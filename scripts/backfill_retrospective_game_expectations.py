"""Refresh full-posterior game expectations in retained predictive publications.

Use each snapshot's frozen game CSV, prior, and inference configuration. The
existing posterior, rankings, schedule, and game ratings are not regenerated.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from gippyrank.posterior.engine import PosteriorResult
from gippyrank.posterior.retrospective import build_retrospective_game_expectations
from gippyrank.posterior.snapshots import (
    _frozen_fcs_fallbacks,
    _game_from_included_row,
    included_game_rows_sha256,
    load_likelihood,
    load_teams,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _retained_sources() -> list[Path]:
    config = _read_json(ROOT / "site/publish_config.json")
    return list(
        dict.fromkeys(
            ROOT / entry["source"]
            for entry in config["snapshots"]
            if "/predictive/" in entry["source"]
        )
    )


def _posterior(source: Path) -> PosteriorResult:
    diagnostics = _read_json(source / "diagnostics.json")
    if not diagnostics["converged"]:
        raise ValueError(f"{source}: production posterior did not converge")
    pmfs: dict[str, list[float]] = {}
    with (source / "posterior_pmfs.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pmfs.setdefault(row["team_id"], []).append(float(row["probability"]))
    return PosteriorResult(
        pmfs={team_id: np.asarray(pmf) for team_id, pmf in pmfs.items()},
        converged=True,
        iterations=int(diagnostics["iterations"]),
        max_message_delta=float(diagnostics["max_message_delta"]),
        objective=float(diagnostics["objective_surrogate"]),
        raw_game_factor_count=int(diagnostics["raw_game_likelihood_count"]),
        unique_pair_factor_count=int(diagnostics["unique_pairwise_factor_count"]),
        max_team_degree=int(diagnostics["maximum_team_degree"]),
    )


def backfill(source: Path) -> None:
    metadata_path = source / "metadata.json"
    metadata = _read_json(metadata_path)
    artifact_path = source / "team_seasons.json"
    artifact = _read_json(artifact_path)
    with (source / "included_games.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    game_ids = [row["id"] for row in rows]
    if game_ids != [str(value) for value in metadata["included_game_ids"]]:
        raise ValueError(f"{source}: frozen game IDs disagree with metadata")
    expected_hash = metadata.get("included_game_rows_sha256")
    if expected_hash is not None and included_game_rows_sha256(rows) != expected_hash:
        raise ValueError(f"{source}: frozen game rows disagree with metadata")

    teams, team_rows, prior_path = load_teams(
        ROOT,
        int(metadata["season"]),
        metadata["prior_family"],
        str(metadata["prior_model_version"]),
    )
    if sha256(prior_path) != metadata["prior_artifact_sha256"]:
        raise ValueError(f"{source}: frozen prior hash differs from selected prior")
    teams, _, _, _ = _frozen_fcs_fallbacks(
        source=source,
        source_metadata=metadata,
        included_rows=rows,
        teams=teams,
        team_rows=team_rows,
    )
    games = [_game_from_included_row(row) for row in rows]
    expectation = build_retrospective_game_expectations(
        metadata=metadata,
        teams=teams,
        games=games,
        posterior=_posterior(source),
        likelihood=load_likelihood(
            ROOT / "data/processed/posterior/historical_likelihood_v1.json"
        ),
    )
    artifact["retrospective_game_expectations"] = expectation
    for team in artifact["teams"].values():
        for game in team["games"]:
            game["retrospective_expectation_id"] = (
                str(game["game_id"])
                if str(game["game_id"]) in expectation["games"]
                else None
            )
    metadata["retrospective_game_expectations_path"] = "team_seasons.json"
    metadata["retrospective_game_expectations_version"] = expectation[
        "retrospective_game_expectations_version"
    ]
    _write_json(artifact_path, artifact)
    _write_json(metadata_path, metadata)
    print(
        f"{metadata['snapshot_id']}: {len(games)} games, "
        f"{expectation['inference']['runtime_seconds']:.1f}s",
        flush=True,
    )


def sync_performance_sources() -> None:
    for metadata_path in sorted(
        ROOT.glob("data/processed/snapshots/2026/*/performance/metadata.json")
    ):
        performance = metadata_path.parent
        metadata = _read_json(metadata_path)
        source = ROOT / metadata["source_context_path"]
        metadata["source_context_metadata_sha256"] = sha256(source / "metadata.json")
        _write_json(metadata_path, metadata)
        shutil.copyfile(source / "team_seasons.json", performance / "team_seasons.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, help="One retained predictive source")
    parser.add_argument(
        "--force", action="store_true", help="Recompute existing records"
    )
    args = parser.parse_args()
    sources = [ROOT / args.source] if args.source else _retained_sources()
    for source in sources:
        existing = _read_json(source / "team_seasons.json").get(
            "retrospective_game_expectations", {}
        )
        if (
            not args.force
            and existing.get("retrospective_game_expectations_version") == "2.0"
        ):
            continue
        backfill(source)
    if args.source is None:
        sync_performance_sources()


if __name__ == "__main__":
    main()
