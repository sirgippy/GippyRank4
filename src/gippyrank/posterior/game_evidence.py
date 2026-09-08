"""Snapshot-scoped single-game evidence derived from converged BP cavities."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    PosteriorResult,
    Team,
    game_evidence_pmf,
)

TEAM_SEASON_SCHEMA_VERSION = "1.0"


def _normalise(pmf: np.ndarray) -> np.ndarray:
    values = np.asarray(pmf, dtype=float)
    total = float(values.sum())
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or total <= 0:
        raise ValueError("game evidence must be a finite, positive PMF")
    return values / total


def _quantile(pmf: np.ndarray, probability: float) -> int:
    return int(np.searchsorted(np.cumsum(pmf), probability, side="left") + 1)


def game_evidence_summary(pmf: np.ndarray) -> dict[str, Any]:
    """Summarize a game-evidence PMF without exporting the full distribution."""
    pmf = _normalise(pmf)
    ranks = np.arange(1, len(pmf) + 1)
    intervals = {
        str(level): [
            _quantile(pmf, (1 - level / 100) / 2),
            _quantile(pmf, 1 - (1 - level / 100) / 2),
        ]
        for level in (50, 80, 95)
    }
    return {
        "rank_count": len(pmf),
        "expected_rank": float(np.dot(ranks, pmf)),
        "median_rank": _quantile(pmf, 0.5),
        "mode_rank": int(np.argmax(pmf) + 1),
        "interval_50": intervals["50"],
        "interval_80": intervals["80"],
        "interval_95": intervals["95"],
        "top5_probability": float(pmf[:5].sum()),
        "top10_probability": float(pmf[:10].sum()),
        "top25_probability": float(pmf[:25].sum()),
    }


def _bool(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _int_or_none(value: object) -> int | None:
    if value is None or str(value).strip() in {"", "None", "null"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _week(value: object) -> int | str | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _result_and_score(
    row: dict[str, str], focal_id: str, *, reveal: bool
) -> tuple[str | None, dict[str, int] | None]:
    home_points = _int_or_none(row.get("homePoints"))
    away_points = _int_or_none(row.get("awayPoints"))
    if not reveal or home_points is None or away_points is None:
        return None, None
    focal_home = row["homeId"] == focal_id
    focal_points, opponent_points = (
        (home_points, away_points) if focal_home else (away_points, home_points)
    )
    result = "W" if focal_points > opponent_points else "L" if focal_points < opponent_points else "T"
    return result, {"team": focal_points, "opponent": opponent_points}


def build_team_season_artifact(
    *,
    root: Path,
    metadata: dict[str, Any],
    teams: list[Team] | tuple[Team, ...],
    team_rows: dict[str, dict[str, str]],
    games: list[Game] | tuple[Game, ...],
    included_rows: list[dict[str, str]],
    posterior: PosteriorResult,
    likelihood: LikelihoodV1 | None,
) -> dict[str, Any]:
    """Build the compact, snapshot-aware team-season source artifact.

    The schedule is read from the processed corpus so future rows can be
    represented without revealing scores. Ratings are generated only for
    completed games in ``included_rows`` and use the Context BP state passed by
    the snapshot builder. No full per-game PMFs are serialized.
    """
    fbs_teams = {team.team_id: team for team in teams if team.subdivision == "fbs"}
    included_ids = {str(row["id"]) for row in included_rows}
    game_by_id = {game.game_id: game for game in games}
    ratings: dict[tuple[str, str], dict[str, Any]] = {}
    if likelihood is not None:
        for game in games:
            for focal_id in (game.home_id, game.away_id):
                if focal_id in fbs_teams:
                    ratings[game.game_id, focal_id] = game_evidence_summary(
                        game_evidence_pmf(game, focal_id, list(teams), likelihood, posterior)
                    )

    schedule_path = root / "data/processed/cfbd/games.csv"
    schedule_corpus_sha256 = (
        hashlib.sha256(schedule_path.read_bytes()).hexdigest()
        if schedule_path.is_file()
        else None
    )
    schedule: list[dict[str, str]] = []
    if schedule_path.is_file():
        with schedule_path.open(newline="", encoding="utf-8") as handle:
            schedule = list(csv.DictReader(handle))

    team_games: dict[str, list[dict[str, Any]]] = {team_id: [] for team_id in fbs_teams}
    for row in schedule:
        if str(row.get("season", "")) != str(metadata["season"]):
            continue
        home_id, away_id = row.get("homeId", ""), row.get("awayId", "")
        focal_sides = [
            (home_id, away_id, "homeTeam", "awayTeam", "awayClassification", "awayConference"),
            (away_id, home_id, "awayTeam", "homeTeam", "homeClassification", "homeConference"),
        ]
        game_id = str(row.get("id", ""))
        eligible_matchup = (
            row.get("homeClassification", "").casefold() in {"fbs", "fcs"}
            and row.get("awayClassification", "").casefold() in {"fbs", "fcs"}
        )
        modeled = (
            game_id in included_ids
            and eligible_matchup
            and game_id in game_by_id
        )
        for focal_id, opponent_id, focal_name_field, opponent_name_field, opponent_class_field, opponent_conf_field in focal_sides:
            if focal_id not in fbs_teams:
                continue
            result, score = _result_and_score(row, focal_id, reveal=modeled)
            entry: dict[str, Any] = {
                "game_id": game_id,
                "week": _week(row.get("week")),
                "date": row.get("startDate"),
                "opponent_id": opponent_id,
                "opponent_name": row.get(opponent_name_field, ""),
                "opponent_classification": row.get(opponent_class_field, "").lower(),
                "opponent_conference": row.get(opponent_conf_field, ""),
                "site": "neutral" if _bool(row.get("neutralSite")) else "home" if focal_id == home_id else "away",
                "result": result,
                "score": score,
                "modeled": modeled,
                "game_rating": ratings.get((game_id, focal_id)) if modeled else None,
                "season_type": row.get("seasonType", ""),
                "conference_game": _bool(row.get("conferenceGame")),
            }
            team_games[focal_id].append(entry)

    for entries in team_games.values():
        entries.sort(key=lambda entry: (str(entry["date"]), str(entry["game_id"])))

    rank_count = len(next(iter(fbs_teams.values())).prior) if fbs_teams else 0
    return {
        "schema_version": TEAM_SEASON_SCHEMA_VERSION,
        "artifact_kind": "team_season",
        "snapshot_id": metadata["snapshot_id"],
        "season": metadata["season"],
        "snapshot_type": metadata["snapshot_type"],
        "anchor_family": "context",
        "source_context_snapshot_id": metadata["snapshot_id"],
        "requested_cutoff": metadata.get("requested_cutoff"),
        "effective_cutoff": metadata.get("effective_cutoff"),
        "source_retrieved_at": metadata.get("source_retrieved_at"),
        "source_retrieval_times": metadata.get("source_retrieval_times"),
        "source_response_hashes": metadata.get("source_response_hashes", {}),
        "game_corpus_sha256": metadata.get("game_corpus_sha256"),
        "schedule_source": {
            "kind": "current_processed_schedule",
            "path": "data/processed/cfbd/games.csv",
            "sha256": schedule_corpus_sha256,
        },
        "included_game_ids": sorted(included_ids),
        "historical_likelihood_version": metadata.get("historical_likelihood_version", "V1"),
        "rank_count": rank_count,
        "rating_definition": "normalize(single-game Historical Likelihood × opponent pair-cavity belief)",
        "loopy_bp_caveat": "The focal preseason prior is absent as a direct factor; loopy cycles can leave indirect feedback.",
        "rematch_definition": "All games in a grouped head-to-head pair use the same pair cavity; each game is rated independently.",
        "teams": {
            team_id: {
                "team_id": team_id,
                "team_name": team.name,
                "conference": team_rows.get(team_id, {}).get("conference", ""),
                "games": entries,
            }
            for team_id, team in sorted(fbs_teams.items())
            for entries in [team_games[team_id]]
        },
    }
