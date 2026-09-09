"""Backfill Season Simulation V1 without rebuilding retained posteriors.

This command is intentionally different from the normal snapshot builder. It
reads each committed snapshot's posterior PMFs, metadata, and included-game
IDs, then adds only the new season-simulation field. That preserves the
historical information boundary of a retained snapshot when the mutable
current schedule cache has since advanced.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gippyrank.posterior.game_evidence import _completed_regular_records
from gippyrank.posterior.predictive import ScheduledGame
from gippyrank.posterior.season_simulation import (
    SeasonSimulationConfig,
    simulate_season,
)
from gippyrank.posterior.snapshots import (
    add_fcs_fallbacks,
    load_likelihood,
    load_teams,
    subdivision_population_size,
    subdivision_population_source,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _datetime(value: object) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _regular(row: dict[str, str]) -> bool:
    return str(row.get("seasonType", "regular") or "regular").casefold() in {
        "",
        "regular",
    }


def _cancelled(row: dict[str, str]) -> bool:
    status = str(row.get("status", row.get("gameStatus", "")) or "").casefold()
    return "cancel" in status or "postpon" in status


def _involves_fbs(row: dict[str, str], fbs_ids: set[str]) -> bool:
    return any(
        row.get(f"{side}Id", "") in fbs_ids
        or row.get(f"{side}Classification", "").casefold() == "fbs"
        for side in ("home", "away")
    )


def _state(
    row: dict[str, str],
    metadata: dict[str, Any],
    included_ids: set[str],
) -> str:
    """Classify using the retained evidence boundary, not later score data."""
    if _cancelled(row):
        return "cancelled"
    if str(row.get("id", "")) in included_ids:
        return "completed"
    if metadata.get("snapshot_type") == "preseason":
        return "future"
    kickoff = _datetime(row.get("startDate"))
    cutoff = _datetime(metadata.get("effective_cutoff"))
    if kickoff is None or cutoff is None:
        return "unresolved"
    return "future" if kickoff > cutoff else "unresolved"


def _schedule(root: Path, season: int) -> list[dict[str, str]]:
    path = root / "data/processed/cfbd/games.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if int(row["season"]) == season and _regular(row)
        ]


def _posterior_pmfs(path: Path) -> dict[str, list[float]]:
    pmfs: dict[str, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pmfs.setdefault(row["team_id"], []).append(float(row["probability"]))
    return pmfs


def _game(row: dict[str, str], state: str) -> ScheduledGame:
    return ScheduledGame(
        game_id=row.get("id", ""),
        home_id=row.get("homeId", ""),
        away_id=row.get("awayId", ""),
        home_subdivision=row.get("homeClassification", ""),
        away_subdivision=row.get("awayClassification", ""),
        neutral_site=str(row.get("neutralSite", "")).casefold() in {"true", "1", "yes"},
        season_type=row.get("seasonType", "regular") or "regular",
        date=row.get("startDate"),
        schedule_status=state,
    )


def _simulation_provenance(
    metadata: dict[str, Any],
    prediction_source: str,
    fallback_team_ids: list[str],
    fcs_population: int,
    fcs_population_source: str,
) -> dict[str, Any]:
    """Describe both the retained posterior and simulation-only additions."""
    return {
        "source_snapshot_id": metadata["snapshot_id"],
        "season": metadata["season"],
        "snapshot_type": metadata["snapshot_type"],
        "requested_cutoff": metadata.get("requested_cutoff"),
        "effective_cutoff": metadata.get("effective_cutoff"),
        "game_corpus_sha256": metadata.get("game_corpus_sha256"),
        "historical_likelihood_version": metadata.get(
            "historical_likelihood_version", "V1"
        ),
        "prediction_source": prediction_source,
        "season_simulation_fcs_fallback_count": len(fallback_team_ids),
        "season_simulation_fcs_fallback_team_ids": fallback_team_ids,
        "season_simulation_fcs_fallback_kind": (
            "uniform_full_subdivision_rank" if fallback_team_ids else None
        ),
        "season_simulation_fcs_fallback_pmf_semantics": (
            "uniform ranks 1..N_FCS" if fallback_team_ids else None
        ),
        "season_simulation_fcs_population_size": fcs_population,
        "season_simulation_fcs_population_source": fcs_population_source,
    }


def build_simulation_inputs(source: Path) -> dict[str, Any]:
    """Load retained posterior inputs without mutating the snapshot."""
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    season = int(metadata["season"])
    family = str(metadata.get("prior_family", "context"))
    teams, team_rows, _prior_path = load_teams(ROOT, season, family)
    with (source / "included_games.csv").open(newline="", encoding="utf-8") as handle:
        included_rows = list(csv.DictReader(handle))
    included_ids = {str(value) for value in metadata.get("included_game_ids", [])}
    schedule = _schedule(ROOT, season)
    prior_fbs_ids = {team.team_id for team in teams if team.subdivision == "fbs"}
    scheduled_fcs = [
        row
        for row in schedule
        if _involves_fbs(row, prior_fbs_ids)
        and any(
            row.get(f"{side}Classification", "").casefold() == "fcs"
            for side in ("home", "away")
        )
    ]
    cutoff = _datetime(metadata.get("effective_cutoff"))
    if metadata.get("snapshot_type") != "preseason":
        scheduled_fcs = [
            row
            for row in scheduled_fcs
            if not _cancelled(row)
            and (
                _datetime(row.get("startDate")) is None
                or cutoff is None
                or _datetime(row.get("startDate")) > cutoff
            )
        ]
    fcs_population = subdivision_population_size(ROOT, season, "fcs")
    teams, fallbacks = add_fcs_fallbacks(
        teams,
        team_rows,
        included_rows,
        fcs_population,
        scheduled_fcs,
    )
    fallback_team_ids = sorted(set(fallbacks))
    fbs_ids = {team.team_id for team in teams if team.subdivision == "fbs"}
    future_games: list[ScheduledGame] = []
    excluded: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in schedule:
        if not _involves_fbs(row, fbs_ids):
            continue
        state = _state(row, metadata, included_ids)
        game_id = str(row.get("id", ""))
        if game_id in seen:
            raise ValueError(f"duplicate schedule game ID {game_id}")
        seen.add(game_id)
        if state == "cancelled":
            excluded.append(
                {
                    "game_id": game_id,
                    "home_team_id": row.get("homeId", ""),
                    "away_team_id": row.get("awayId", ""),
                    "reason": "cancelled_or_postponed",
                }
            )
        elif state != "completed":
            future_games.append(_game(row, state))
    posterior = _posterior_pmfs(source / "posterior_pmfs.csv")
    posterior.update(
        {
            team.team_id: team.prior.tolist()
            for team in teams
            if team.team_id not in posterior
        }
    )
    prediction_source = (
        "predictive_history"
        if metadata.get("prior_family") == "history"
        else "predictive_context"
    )
    fcs_population_source = subdivision_population_source(ROOT, season, "fcs")
    return {
        "metadata": metadata,
        "teams": teams,
        "posterior": posterior,
        "future_games": future_games,
        "completed_records": _completed_regular_records(included_rows, fbs_ids),
        "likelihood": load_likelihood(
            ROOT / "data/processed/posterior/historical_likelihood_v1.json"
        ),
        "prediction_source": prediction_source,
        "provenance": _simulation_provenance(
            metadata,
            prediction_source,
            fallback_team_ids,
            fcs_population,
            fcs_population_source,
        ),
        "excluded_schedule_games": excluded,
        "fallback_team_ids": fallback_team_ids,
    }


def backfill(source: Path, config: SeasonSimulationConfig) -> None:
    inputs = build_simulation_inputs(source)
    simulation = simulate_season(
        inputs["teams"],
        inputs["posterior"],
        inputs["future_games"],
        inputs["completed_records"],
        inputs["likelihood"],
        config=config,
        prediction_source=inputs["prediction_source"],
        provenance=inputs["provenance"],
        excluded_schedule_games=inputs["excluded_schedule_games"],
    )
    artifact_path = source / "team_seasons.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["season_simulation"] = simulation
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata_path = source / "metadata.json"
    metadata = inputs["metadata"]
    metadata.update(
        {
            "season_simulation_schema_version": "1.0",
            "season_simulation_version": config.simulation_version,
            "season_simulation_configuration": config.as_dict(),
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sync_performance_team_seasons() -> None:
    """Copy refreshed Context simulations and update their pairing hashes."""
    for metadata_path in sorted(
        ROOT.glob("data/processed/snapshots/2026/*/performance/metadata.json")
    ):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        context = ROOT / metadata["source_context_path"]
        metadata["source_context_metadata_sha256"] = _sha256(context / "metadata.json")
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.copyfile(
            context / "team_seasons.json", metadata_path.parent / "team_seasons.json"
        )


def main() -> None:
    config = SeasonSimulationConfig()
    sources = sorted(
        path.parent
        for path in ROOT.glob(
            "data/processed/snapshots/2026/*/predictive/*/metadata.json"
        )
    )
    for source in sources:
        backfill(source, config)
        print(source, flush=True)
    sync_performance_team_seasons()


if __name__ == "__main__":
    main()
