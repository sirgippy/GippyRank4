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
from time import perf_counter

from gippyrank.methodology import (
    HISTORICAL_LIKELIHOOD_SHA256,
    RETROSPECTIVE_CONDITIONING,
    RETROSPECTIVE_GAME_EXPECTATIONS_VERSION,
)
from gippyrank.posterior.retained import load_retained_posterior
from gippyrank.posterior.retrospective import build_retrospective_game_expectations
from gippyrank.posterior.snapshots import (
    _frozen_fcs_fallbacks,
    _game_from_included_row,
    included_game_rows_sha256,
    load_pinned_likelihood,
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


def backfill(source: Path) -> None:
    started = perf_counter()
    metadata_path = source / "metadata.json"
    metadata = _read_json(metadata_path)
    if metadata.get("historical_likelihood_sha256") != HISTORICAL_LIKELIHOOD_SHA256:
        raise ValueError(f"{source}: frozen likelihood hash differs from pinned V1")
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
        posterior=load_retained_posterior(source, metadata, teams),
        likelihood=load_pinned_likelihood(
            ROOT / "data/processed/posterior/historical_likelihood_v1.json"
        ),
        likelihood_sha256=HISTORICAL_LIKELIHOOD_SHA256,
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
        f"{perf_counter() - started:.1f}s",
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
        source_metadata = _read_json(source / "metadata.json")
        for field in ("combined_source_available_at", "source_retrieved_at_contract"):
            metadata[field] = source_metadata.get(field)
        _write_json(metadata_path, metadata)
        shutil.copyfile(source / "team_seasons.json", performance / "team_seasons.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, help="One retained predictive source")
    parser.add_argument(
        "--force", action="store_true", help="Recompute existing records"
    )
    args = parser.parse_args()
    load_pinned_likelihood(
        ROOT / "data/processed/posterior/historical_likelihood_v1.json"
    )
    sources = [ROOT / args.source] if args.source else _retained_sources()
    for source in sources:
        metadata = _read_json(source / "metadata.json")
        existing = _read_json(source / "team_seasons.json").get(
            "retrospective_game_expectations", {}
        )
        if (
            not args.force
            and existing.get("retrospective_game_expectations_version")
            == RETROSPECTIVE_GAME_EXPECTATIONS_VERSION
            and existing.get("historical_likelihood_sha256")
            == HISTORICAL_LIKELIHOOD_SHA256
            and existing.get("conditioning") == RETROSPECTIVE_CONDITIONING
            and existing.get("posterior_pmfs_sha256") == metadata.get("posterior_pmfs_sha256")
            and "runtime_seconds" not in existing.get("inference", {})
        ):
            continue
        backfill(source)
    if args.source is None:
        sync_performance_sources()


if __name__ == "__main__":
    main()
