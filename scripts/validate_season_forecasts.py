"""Validate Season Simulation V1 against frozen historical seasons.

The validation is Context-primary and leakage-safe: each run starts with the
frozen preseason Context PMFs for its target season, adds only completed games
at the selected cutoff to the posterior, and scores the resulting final-win
forecast against that season's completed regular-season schedule.  It is a
research report, not a production snapshot builder.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from backfill_season_simulation import build_simulation_inputs

from gippyrank.posterior.engine import Game, LikelihoodV1, Team, infer_posterior
from gippyrank.posterior.predictive import ScheduledGame
from gippyrank.posterior.season_simulation import (
    CompletedRecord,
    SeasonSimulationConfig,
    simulate_season,
)

ROOT = Path(__file__).resolve().parents[1]
SEASONS = (2022, 2023, 2024, 2025)
PHASES = ("preseason", "early", "late")
THRESHOLDS = (6, 8, 10)
HISTORICAL_SCHEDULE = ROOT / "data/validation/season_forecast_schedule.csv"
HISTORICAL_SCHEDULE_PROVENANCE = (
    ROOT / "data/validation/season_forecast_schedule.provenance.json"
)
CONVERGENCE_OUTER_COUNTS = (250, 500, 1_000, 2_000)
CONVERGENCE_TEAM_ID = "194"
CONVERGENCE_THRESHOLDS = (8, 10, 11)


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _int_or_none(value: object) -> int | None:
    if value is None or str(value).strip() in {"", "None", "null"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _regular(game: dict[str, Any]) -> bool:
    return str(game.get("seasonType", "regular") or "regular").casefold() in {
        "",
        "regular",
    }


def _cancelled(game: dict[str, Any]) -> bool:
    status = str(game.get("status", game.get("gameStatus", "")) or "").casefold()
    return "cancel" in status or "postpon" in status


def _historical_source() -> dict[str, Any]:
    if (
        not HISTORICAL_SCHEDULE.is_file()
        or not HISTORICAL_SCHEDULE_PROVENANCE.is_file()
    ):
        raise FileNotFoundError(
            "historical validation requires the tracked frozen schedule and provenance"
        )
    provenance = json.loads(HISTORICAL_SCHEDULE_PROVENANCE.read_text(encoding="utf-8"))
    actual_hash = _sha256(HISTORICAL_SCHEDULE)
    expected_hash = provenance.get("materialized_schedule_sha256")
    if actual_hash != expected_hash:
        raise ValueError(
            "historical validation schedule hash does not match its provenance"
        )
    return {
        "path": HISTORICAL_SCHEDULE.relative_to(ROOT).as_posix(),
        "provenance_path": HISTORICAL_SCHEDULE_PROVENANCE.relative_to(ROOT).as_posix(),
        "sha256": actual_hash,
        "raw_source_sha256": provenance.get("raw_source_sha256", {}),
        "materialized_row_count": provenance.get("materialized_row_count"),
    }


def _load_likelihood() -> LikelihoodV1:
    value = json.loads(
        (ROOT / "data/processed/posterior/historical_likelihood_v1.json").read_text(
            encoding="utf-8"
        )
    )
    return LikelihoodV1(
        np.asarray(value["beta"], dtype=float),
        float(value["scale"]),
        float(value["degrees_of_freedom"]),
        value.get("fit_kind", "weighted_pseudo"),
    )


def _load_context_prior(season: int) -> dict[str, Team]:
    path = ROOT / "data/processed/preseason/context/predictions.csv"
    teams: dict[str, Team] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["season"]) != season or row["subdivision"] != "fbs":
                continue
            teams[row["team_id"]] = Team(
                row["team_id"],
                row["team_name"],
                "fbs",
                np.asarray(json.loads(row["pmf"]), dtype=float),
            )
    if not teams:
        raise ValueError(f"no frozen Context prior rows found for {season}")
    return teams


def _fcs_population(season: int) -> int:
    population = {
        str(game[f"{side}Id"])
        for game in _load_schedule(season)
        for side in ("home", "away")
        if str(game.get(f"{side}Classification", "") or "").casefold() == "fcs"
    }
    if not population:
        raise ValueError(f"no FCS population found in frozen schedule for {season}")
    return len(population)


def _load_schedule(season: int) -> list[dict[str, Any]]:
    _historical_source()
    games: list[dict[str, Any]] = []
    with HISTORICAL_SCHEDULE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["season"]) != season:
                continue
            game = {
                "id": row["game_id"],
                "season": int(row["season"]),
                "week": row["week"],
                "seasonType": row["season_type"],
                "startDate": row["start_date"],
                "completed": _bool(row["completed"]),
                "status": row["status"],
                "neutralSite": _bool(row["neutral_site"]),
                "homeId": row["home_team_id"],
                "homeTeam": row["home_team_name"],
                "homeClassification": row["home_subdivision"],
                "homeConference": row["home_conference"],
                "homePoints": _int_or_none(row["home_points"]),
                "awayId": row["away_team_id"],
                "awayTeam": row["away_team_name"],
                "awayClassification": row["away_subdivision"],
                "awayConference": row["away_conference"],
                "awayPoints": _int_or_none(row["away_points"]),
            }
            if _regular(game):
                games.append(game)
    return sorted(
        games, key=lambda game: (_utc(str(game["startDate"])), str(game["id"]))
    )


def _cutoffs(schedule: list[dict[str, Any]]) -> dict[str, datetime | None]:
    dates = sorted(
        {
            _utc(str(game["startDate"])).date()
            for game in schedule
            if _bool(game.get("completed"))
            and _int_or_none(game.get("homePoints")) is not None
            and _int_or_none(game.get("awayPoints")) is not None
        }
    )
    if len(dates) < 3:
        raise ValueError("historical schedule does not have enough dates")
    return {
        "preseason": None,
        "early": datetime.combine(
            dates[round(0.20 * (len(dates) - 1))], datetime.max.time(), tzinfo=UTC
        ),
        "late": datetime.combine(
            dates[round(0.72 * (len(dates) - 1))], datetime.max.time(), tzinfo=UTC
        ),
    }


def _teams_and_fcs(
    season: int, prior: dict[str, Team], schedule: list[dict[str, Any]]
) -> list[Team]:
    fbs_ids = set(prior)
    fcs: dict[str, tuple[str, np.ndarray]] = {}
    for game in schedule:
        sides = (
            ("home", str(game.get("homeClassification", "") or "").casefold()),
            ("away", str(game.get("awayClassification", "") or "").casefold()),
        )
        if not any(
            team_id in fbs_ids for team_id in (str(game["homeId"]), str(game["awayId"]))
        ):
            continue
        for side, subdivision in sides:
            if subdivision == "fcs":
                fcs[str(game[f"{side}Id"])] = (
                    str(game[f"{side}Team"]),
                    np.asarray([], dtype=float),
                )
    population = _fcs_population(season)
    return [
        *prior.values(),
        *(
            Team(team_id, name, "fcs", np.full(population, 1 / population))
            for team_id, (name, _unused) in sorted(fcs.items())
        ),
    ]


def _model_game(game: dict[str, Any], teams: dict[str, Team]) -> bool:
    home_id, away_id = str(game["homeId"]), str(game["awayId"])
    home_subdivision = str(game.get("homeClassification", "") or "").casefold()
    away_subdivision = str(game.get("awayClassification", "") or "").casefold()
    return (
        home_id in teams
        and away_id in teams
        and home_subdivision in {"fbs", "fcs"}
        and away_subdivision in {"fbs", "fcs"}
        and (home_subdivision == "fbs" or away_subdivision == "fbs")
    )


def _involves_fbs(game: dict[str, Any], fbs_ids: set[str]) -> bool:
    return any(
        str(game.get(f"{side}Id", "")) in fbs_ids
        or str(game.get(f"{side}Classification", "") or "").casefold() == "fbs"
        for side in ("home", "away")
    )


def _schedule_state(game: dict[str, Any], cutoff: datetime | None) -> str:
    """Classify every regular schedule row using production semantics."""
    if _cancelled(game):
        return "cancelled"
    if cutoff is None:
        return "future"
    start_date = str(game.get("startDate", "") or "")
    if not start_date:
        return "unresolved"
    if _utc(start_date) > cutoff:
        return "future"
    if (
        _bool(game.get("completed"))
        and _int_or_none(game.get("homePoints")) is not None
        and _int_or_none(game.get("awayPoints")) is not None
    ):
        return "completed"
    return "unresolved"


def _record_for_cutoff(
    schedule: list[dict[str, Any]], fbs_ids: set[str], cutoff: datetime | None
) -> dict[str, CompletedRecord]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for game in schedule:
        if _schedule_state(game, cutoff) != "completed":
            continue
        home_id, away_id = str(game["homeId"]), str(game["awayId"])
        home_points, away_points = (
            _int_or_none(game.get("homePoints")),
            _int_or_none(game.get("awayPoints")),
        )
        if home_points is None or away_points is None:
            continue
        if home_points == away_points:
            result = 2
        elif home_points > away_points:
            result = 0
        else:
            result = 1
        if home_id in fbs_ids:
            counts[home_id][result] += 1
        if away_id in fbs_ids:
            counts[away_id][1 - result if result < 2 else 2] += 1
    return {team_id: CompletedRecord(*counts[team_id]) for team_id in fbs_ids}


def _actual_final_wins(
    schedule: list[dict[str, Any]], fbs_ids: set[str]
) -> dict[str, int]:
    wins = defaultdict(int)
    for game in schedule:
        if _schedule_state(game, datetime.max.replace(tzinfo=UTC)) != "completed":
            continue
        home_id, away_id = str(game["homeId"]), str(game["awayId"])
        home_points = _int_or_none(game.get("homePoints"))
        away_points = _int_or_none(game.get("awayPoints"))
        if home_points is None or away_points is None:
            continue
        if home_points > away_points and home_id in fbs_ids:
            wins[home_id] += 1
        if away_points > home_points and away_id in fbs_ids:
            wins[away_id] += 1
    return dict(wins)


def _crps(distribution: dict[str, float], observed: int) -> float:
    values = np.asarray([int(key) for key in distribution], dtype=float)
    probabilities = np.asarray(list(distribution.values()), dtype=float)
    first = float(np.sum(probabilities * np.abs(values - observed)))
    pairwise = np.abs(values[:, None] - values[None, :])
    second = 0.5 * float(probabilities @ pairwise @ probabilities)
    return first - second


def _run(
    season: int,
    phase: str,
    cutoff: datetime | None,
    likelihood: LikelihoodV1,
    config: SeasonSimulationConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior = _load_context_prior(season)
    schedule = _load_schedule(season)
    teams = _teams_and_fcs(season, prior, schedule)
    teams_by_id = {team.team_id: team for team in teams}
    fbs_ids = set(prior)
    states = {str(game["id"]): _schedule_state(game, cutoff) for game in schedule}
    completed = [
        Game(
            str(game["id"]),
            str(game["homeId"]),
            str(game["awayId"]),
            str(game.get("homeClassification", "") or "").casefold(),
            str(game.get("awayClassification", "") or "").casefold(),
            int(game["homePoints"]),
            int(game["awayPoints"]),
            bool(game.get("neutralSite")),
        )
        for game in schedule
        if states[str(game["id"])] == "completed" and _model_game(game, teams_by_id)
    ]
    future = [
        ScheduledGame(
            str(game["id"]),
            str(game["homeId"]),
            str(game["awayId"]),
            str(game.get("homeClassification", "") or "").casefold(),
            str(game.get("awayClassification", "") or "").casefold(),
            bool(game.get("neutralSite")),
            str(game.get("seasonType", "regular") or "regular"),
            str(game.get("startDate", "") or ""),
            states[str(game["id"])],
        )
        for game in schedule
        if _involves_fbs(game, fbs_ids)
        and states[str(game["id"])] in {"future", "unresolved"}
    ]
    excluded = [
        {
            "game_id": str(game["id"]),
            "home_team_id": str(game["homeId"]),
            "away_team_id": str(game["awayId"]),
            "reason": "cancelled_or_postponed",
        }
        for game in schedule
        if _involves_fbs(game, fbs_ids) and states[str(game["id"])] == "cancelled"
    ]
    posterior = (
        infer_posterior(teams, completed, likelihood)
        if completed
        else infer_posterior(teams, [], likelihood)
    )
    if not posterior.converged:
        raise RuntimeError(f"Context posterior did not converge for {season} {phase}")
    simulation = simulate_season(
        teams,
        posterior.pmfs,
        future,
        _record_for_cutoff(schedule, fbs_ids, cutoff),
        likelihood,
        config=config,
        prediction_source="predictive_context",
        provenance={
            "source_snapshot_id": f"historical-{season}-{phase}-context",
            "season": season,
            "snapshot_type": "preseason" if phase == "preseason" else "weekly",
            "requested_cutoff": cutoff.isoformat() if cutoff else None,
            "effective_cutoff": cutoff.isoformat() if cutoff else None,
            "historical_likelihood_version": "V1",
            "historical_schedule_source": _historical_source()["path"],
            "historical_schedule_sha256": _historical_source()["sha256"],
        },
        excluded_schedule_games=excluded,
    )
    actual = _actual_final_wins(schedule, fbs_ids)
    rows: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for team_id in sorted(fbs_ids):
        summary = simulation["teams"][team_id]
        if summary["forecast_status"] != "available":
            unavailable.append(
                {
                    "team_id": team_id,
                    "team_name": summary["team_name"],
                    "actual_final_wins": actual.get(team_id, 0),
                    "unsupported_games": summary.get("unsupported_games", []),
                }
            )
            continue
        observed = actual.get(team_id, 0)
        row = {
            "season": season,
            "phase": phase,
            "team_id": team_id,
            "actual_final_wins": observed,
            "expected_final_wins": summary["expected_final_wins"],
            "expected_final_win_error": abs(summary["expected_final_wins"] - observed),
            "crps": _crps(summary["final_win_distribution"], observed),
            "future_game_count": summary["remaining_games"],
            "forecast_scope_game_count": summary["forecast_scope_games"],
            "completed_game_count": summary["completed_regular_season_games"],
        }
        for level in (50, 80, 95):
            low, high = summary[f"final_win_interval_{level}"]
            row[f"coverage_{level}"] = int(low <= observed <= high)
        for threshold in THRESHOLDS:
            key = f"wins_{threshold}_plus"
            row[f"threshold_{threshold}_prediction"] = summary[
                "threshold_probabilities"
            ].get(key, 0.0)
            row[f"threshold_{threshold}_actual"] = int(observed >= threshold)
        rows.append(row)
    return rows, {
        "season": season,
        "phase": phase,
        "cutoff": cutoff.isoformat() if cutoff else None,
        "fbs_team_count": len(fbs_ids),
        "scored_team_count": len(rows),
        "unavailable_team_count": len(unavailable),
        "unavailable_teams": unavailable,
        "schedule_game_count": len(schedule),
        "fbs_involving_schedule_game_count": sum(
            _involves_fbs(game, fbs_ids) for game in schedule
        ),
        "fixed_record_game_count": sum(
            states[str(game["id"])] == "completed" and _involves_fbs(game, fbs_ids)
            for game in schedule
        ),
        "excluded_schedule_game_count": len(excluded),
        "completed_model_game_count": len(completed),
        "future_candidate_game_count": len(future),
        "future_supported_game_count": simulation["season_scope"][
            "supported_future_game_count"
        ],
        "unresolved_candidate_game_count": sum(
            game.schedule_status == "unresolved" for game in future
        ),
        "unsupported_candidate_game_count": len(
            simulation["season_scope"]["unsupported_future_games"]
        ),
        "posterior_iterations": posterior.iterations,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"team_forecasts": len(rows)}
    for key in ("expected_final_win_error", "crps"):
        result[key] = float(np.mean([row[key] for row in rows]))
    for level in (50, 80, 95):
        result[f"coverage_{level}"] = float(
            np.mean([row[f"coverage_{level}"] for row in rows])
        )
    for threshold in THRESHOLDS:
        predictions = np.asarray(
            [row[f"threshold_{threshold}_prediction"] for row in rows]
        )
        actual = np.asarray([row[f"threshold_{threshold}_actual"] for row in rows])
        result[f"threshold_{threshold}_predicted_rate"] = float(predictions.mean())
        result[f"threshold_{threshold}_observed_rate"] = float(actual.mean())
        result[f"threshold_{threshold}_brier"] = float(
            np.mean((predictions - actual) ** 2)
        )
    return result


def _bytes(value: float) -> str:
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def _payload_summary() -> dict[str, Any]:
    manifest = json.loads(
        (ROOT / "site/data/manifest.json").read_text(encoding="utf-8")
    )
    entries = manifest["snapshots"]
    return {
        "published_snapshot_count": len(entries),
        "season_simulation_bytes": sum(
            int(item["season_simulation_bytes"]) for item in entries
        ),
        "season_simulation_browser_bytes": sum(
            int(item["season_simulation_browser_bytes"]) for item in entries
        ),
        "average_browser_increment_bytes": float(
            np.mean(
                [
                    int(item["season_simulation_incremental_lazy_bytes"])
                    for item in entries
                ]
            )
        ),
        "average_team_summary_bytes": float(
            np.mean(
                [
                    float(item["season_simulation_average_team_summary_bytes"])
                    for item in entries
                ]
            )
        ),
        "average_game_marginals_bytes": float(
            np.mean(
                [
                    int(item["season_simulation_game_marginals_bytes"])
                    for item in entries
                ]
            )
        ),
        "average_event_decomposition_bytes": float(
            np.mean(
                [
                    int(item["season_simulation_event_decomposition_bytes"])
                    for item in entries
                ]
            )
        ),
        "lazy_team_season_bytes": int(
            manifest["payload_stats"]["lazy_team_season_bytes"]
        ),
        "browser_fields_removed": [
            "game_marginals",
            "cross_game_dependence",
            "validation",
            "monte_carlo",
            "event_probability_decomposition",
        ],
    }


def _current_equivalence() -> dict[str, Any]:
    candidates = []
    for path in (ROOT / "data/processed/snapshots/2026").glob(
        "*-context/predictive/context/team_seasons.json"
    ):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        effective = artifact["season_simulation"]["provenance"].get("effective_cutoff")
        if effective:
            candidates.append((effective, path, artifact))
    _effective, path, artifact = max(candidates, key=lambda item: item[0])
    equivalence = artifact["season_simulation"]["validation"][
        "single_game_marginal_equivalence"
    ]
    return {
        "snapshot_id": artifact["season_simulation"]["provenance"][
            "source_snapshot_id"
        ],
        "game_count": equivalence["game_count"],
        "max_abs_home_win_probability_error": equivalence[
            "max_abs_home_win_probability_error"
        ],
        "max_abs_expected_home_margin_error": equivalence[
            "max_abs_expected_home_margin_error"
        ],
        "path": path.relative_to(ROOT).as_posix(),
    }


def _latest_context_source() -> Path:
    candidates = []
    for path in (ROOT / "data/processed/snapshots/2026").glob(
        "*-context/predictive/context/metadata.json"
    ):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        effective = metadata.get("effective_cutoff")
        if effective:
            candidates.append((effective, path.parent))
    if not candidates:
        raise FileNotFoundError(
            "no current Context snapshot is available for convergence"
        )
    return max(candidates, key=lambda item: item[0])[1]


def _convergence_check() -> dict[str, Any]:
    source = _latest_context_source()
    inputs = build_simulation_inputs(source)
    results: list[dict[str, Any]] = []
    record_probabilities: list[dict[str, float]] = []
    for outer_count in CONVERGENCE_OUTER_COUNTS:
        config = SeasonSimulationConfig(
            outer_draw_count=outer_count,
            inner_rollout_count=0,
            seed=49_049,
        )
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
        team = simulation["teams"].get(CONVERGENCE_TEAM_ID)
        if team is None or team.get("forecast_status") != "available":
            raise RuntimeError(
                f"convergence team {CONVERGENCE_TEAM_ID} is unavailable in {source}"
            )
        record_probabilities.append(team["record_probabilities"])
        results.append(
            {
                "outer_draw_count": outer_count,
                "expected_final_wins": team["expected_final_wins"],
                "selected_final_record_probabilities": {},
                "selected_threshold_probabilities": {
                    f"wins_{threshold}_plus": team["threshold_probabilities"].get(
                        f"wins_{threshold}_plus", 0.0
                    )
                    for threshold in CONVERGENCE_THRESHOLDS
                },
                "team_quality_fraction": team["variance_decomposition"][
                    "team_quality_fraction"
                ],
                "game_randomness_fraction": team["variance_decomposition"][
                    "game_randomness_fraction"
                ],
            }
        )

    final_records = sorted(
        record_probabilities[-1].items(),
        key=lambda item: (-item[1], item[0]),
    )[:3]
    selected_records = [record for record, _probability in final_records]
    for result, probabilities in zip(results, record_probabilities, strict=True):
        result["selected_final_record_probabilities"] = {
            record: probabilities.get(record, 0.0) for record in selected_records
        }

    metadata = inputs["metadata"]
    team = next(team for team in inputs["teams"] if team.team_id == CONVERGENCE_TEAM_ID)
    return {
        "snapshot_id": metadata["snapshot_id"],
        "snapshot_path": source.relative_to(ROOT).as_posix(),
        "team_id": CONVERGENCE_TEAM_ID,
        "team_name": team.name,
        "seed": 49_049,
        "outer_draw_counts": list(CONVERGENCE_OUTER_COUNTS),
        "selected_final_records": selected_records,
        "selected_thresholds": [
            f"wins_{threshold}_plus" for threshold in CONVERGENCE_THRESHOLDS
        ],
        "results": results,
    }


def _write_report(
    rows: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    payload: dict[str, Any],
    equivalence: dict[str, Any],
    historical_source: dict[str, Any],
    convergence: dict[str, Any],
) -> None:
    output_dir = ROOT / "data/processed/season_forecast_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "seasons": list(SEASONS),
        "phases": list(PHASES),
        "configuration": SeasonSimulationConfig().as_dict(),
        "runs": runs,
        "aggregate": {
            phase: _aggregate(
                [row for row in rows if phase == "all" or row["phase"] == phase]
            )
            for phase in [*PHASES, "all"]
        },
        "historical_schedule_source": historical_source,
        "current_snapshot_single_game_equivalence": equivalence,
        "convergence": convergence,
        "payload": payload,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Season Forecast Validation",
        "",
        (
            "Season Simulation V1 was evaluated with frozen preseason Context PMFs for "
            "2022–2025 at three actual-date phases. Preseason uses no game outcomes; "
            "early and late use only completed regular-season games at or before that "
            "season's cutoff. The final target is the completed regular-season win total."
        ),
        "",
        (
            "The production configuration was used unchanged: 2,000 outer draws, "
            "exact conditional Poisson-binomial win totals, seed 49,049, and "
            "Historical Likelihood V1. No retuning was performed. Context is the "
            "primary reported family; History was not substituted for it."
        ),
        "",
        (
            f"The validation schedule is the tracked frozen artifact "
            f"`{historical_source['path']}` (SHA-256 `{historical_source['sha256']}`). "
            "Its provenance records the eight raw CFBD response hashes and the "
            "deterministic first-seen game-ID merge. Validation does not read the "
            "ignored raw-data cache."
        ),
        "",
        "## Schedule accounting",
        "",
        (
            "Every regular-season row involving a known FBS team is accounted for. "
            "Completed scored rows contribute to the fixed record even when the "
            "game is outside Historical Likelihood support. Future and unresolved "
            "rows enter the candidate forecast scope; unsupported candidates make "
            "the affected team-cutoff unavailable and are excluded from scoring."
        ),
        "",
        "| Season | Phase | Schedule rows | FBS-involving | Fixed record | Future candidates | Supported | Unresolved | Unsupported | Scored teams | Unavailable teams |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        lines.append(
            f"| {run['season']} | {run['phase']} | {run['schedule_game_count']} | "
            f"{run['fbs_involving_schedule_game_count']} | {run['fixed_record_game_count']} | "
            f"{run['future_candidate_game_count']} | {run['future_supported_game_count']} | "
            f"{run['unresolved_candidate_game_count']} | {run['unsupported_candidate_game_count']} | "
            f"{run['scored_team_count']} | {run['unavailable_team_count']} |"
        )
    unavailable_lines = []
    for run in runs:
        for team in run["unavailable_teams"]:
            game_ids = sorted(
                {str(game["game_id"]) for game in team["unsupported_games"]}
            )
            unavailable_lines.append(
                f"{run['season']} {run['phase']}: {team['team_name']} "
                f"({team['team_id']}) — games {', '.join(game_ids)}"
            )
    if unavailable_lines:
        lines.extend(
            [
                "",
                (
                    "The unavailable team-cutoffs are retained in the machine-readable "
                    "summary with their actual target and unsupported game descriptors: "
                    + "; ".join(unavailable_lines)
                    + "."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Forecast metrics",
            "",
            (
                "Expected final-win MAE and CRPS are in wins. Coverage is the fraction of "
                "realized final totals inside the central predictive interval."
            ),
            "",
            "| Phase | Team forecasts | Expected-win MAE | CRPS | 50% coverage | 80% coverage | 95% coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for phase in [*PHASES, "all"]:
        item = summary["aggregate"][phase]
        lines.append(
            f"| {phase} | {item['team_forecasts']} | {item['expected_final_win_error']:.3f} | "
            f"{item['crps']:.3f} | {item['coverage_50']:.1%} | "
            f"{item['coverage_80']:.1%} | {item['coverage_95']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Threshold calibration",
            "",
            (
                "The table reports the mean predicted probability, observed frequency, "
                "and Brier score for selected final-win thresholds."
            ),
            "",
            "| Phase | Threshold | Predicted | Observed | Brier |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for phase in [*PHASES, "all"]:
        item = summary["aggregate"][phase]
        for threshold in THRESHOLDS:
            lines.append(
                f"| {phase} | {threshold}+ wins | {item[f'threshold_{threshold}_predicted_rate']:.1%} | "
                f"{item[f'threshold_{threshold}_observed_rate']:.1%} | "
                f"{item[f'threshold_{threshold}_brier']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Convergence check",
            "",
            (
                f"For the latest current Context snapshot `{convergence['snapshot_id']}`, "
                f"the report reran {convergence['team_name']} ({convergence['team_id']}) "
                f"with seed {convergence['seed']} at outer-draw counts "
                f"{', '.join(str(value) for value in convergence['outer_draw_counts'])}. "
                "The selected final-record rows are the three highest-probability "
                "records at the largest run; threshold and quality/game variance "
                "fractions are recomputed at each budget."
            ),
            "",
            "| Outer draws | Expected final wins | Selected final records | 8+ wins | 10+ wins | 11+ wins | Quality fraction | Game fraction |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in convergence["results"]:
        records = "; ".join(
            f"{record} {probability:.3f}"
            for record, probability in result[
                "selected_final_record_probabilities"
            ].items()
        )
        threshold_values = result["selected_threshold_probabilities"]
        lines.append(
            f"| {result['outer_draw_count']} | {result['expected_final_wins']:.3f} | {records} | "
            f"{threshold_values['wins_8_plus']:.3f} | {threshold_values['wins_10_plus']:.3f} | "
            f"{threshold_values['wins_11_plus']:.3f} | {result['team_quality_fraction']:.3f} | "
            f"{result['game_randomness_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Current single-game equivalence",
            "",
            (
                f"The latest current Context snapshot `{equivalence['snapshot_id']}` "
                f"contains {equivalence['game_count']} supported future games. Its "
                "outer latent-rank integration has maximum absolute error of "
                f"{equivalence['max_abs_home_win_probability_error']:.6f} in home-win "
                "probability and "
                f"{equivalence['max_abs_expected_home_margin_error']:.6f} points in "
                "expected home margin versus the exact single-game V1 mixture."
            ),
            "",
            "## Published payload impact",
            "",
            (
                "The manifest measures the durable diagnostics and the browser-facing "
                "lazy artifact separately. The browser copy keeps team summaries and "
                "distributions, while stripping game marginals, cross-game dependence, "
                "validation, Monte Carlo, and event-decomposition diagnostics."
            ),
            "",
            "| Measure | Value |",
            "| --- | ---: |",
            f"| Published snapshots | {payload['published_snapshot_count']} |",
            f"| Durable season-simulation bytes | {_bytes(payload['season_simulation_bytes'])} |",
            f"| Browser season-simulation bytes | {_bytes(payload['season_simulation_browser_bytes'])} |",
            f"| Mean browser lazy increment per snapshot | {_bytes(payload['average_browser_increment_bytes'])} |",
            f"| Mean team-summary bytes per snapshot | {_bytes(payload['average_team_summary_bytes'])} |",
            f"| Mean game-marginals diagnostic bytes per snapshot | {_bytes(payload['average_game_marginals_bytes'])} |",
            f"| Mean event-decomposition diagnostic bytes per snapshot | {_bytes(payload['average_event_decomposition_bytes'])} |",
            f"| Existing lazy team-season payload total | {_bytes(payload['lazy_team_season_bytes'])} |",
            "",
            (
                "Validation is a historical diagnostic over already-frozen seasons, "
                "not independent confirmation: those seasons have been used by prior "
                "GippyRank research. The check is still leakage-safe with respect to "
                "each selected cutoff and does not alter production artifacts."
            ),
            "",
        ]
    )
    (ROOT / "docs/season_forecast_validation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    config = SeasonSimulationConfig()
    likelihood = _load_likelihood()
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for season in SEASONS:
        schedule = _load_schedule(season)
        for phase, cutoff in _cutoffs(schedule).items():
            phase_rows, run = _run(season, phase, cutoff, likelihood, config)
            rows.extend(phase_rows)
            runs.append(run)
            print(f"validated {season} {phase} ({len(phase_rows)} teams)", flush=True)
    _write_report(
        rows,
        runs,
        _payload_summary(),
        _current_equivalence(),
        _historical_source(),
        _convergence_check(),
    )
    print(ROOT / "docs/season_forecast_validation.md")


if __name__ == "__main__":
    main()
