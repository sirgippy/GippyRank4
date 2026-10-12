"""Issue 201 matched replay, scoring, and frozen-target sensitivity helpers."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from scipy.special import gammaln, logsumexp, stdtr
from scipy.stats import t as student_t

from gippyrank.posterior.engine import Game, LikelihoodV1, Team
from gippyrank.posterior.predictive import (
    PredictiveMarginSummary,
    ScheduledGame,
    predictive_components,
    win_probabilities,
)
from gippyrank.research.matched_pipeline_replay import (
    Policy,
    PriorFamily,
    read_pipeline_bundle,
    replay_pipeline_bundle,
)

SEASONS = (2022, 2023, 2024, 2025)
MODEL_ARMS = (
    ("History_existing", "History", "existing"),
    ("Context_existing", "Context", "existing"),
    ("History_pre_bowl", "History", "pre_bowl"),
    ("Context_pre_bowl", "Context", "pre_bowl"),
)
PRIMARY_POPULATION = "fbs_fbs"
SECONDARY_POPULATION = "fbs_fcs"
RESULT_AVAILABILITY_LAG = timedelta(hours=48)
WEEKLY_CUTOFF_WEEKDAY = 1  # Tuesday, where Monday is 0.
WEEKLY_CUTOFF_UTC_TIME = time(0, 0, 0)
CRPS_TAIL_MULTIPLE = 40.0
CRPS_ABSOLUTE_TOLERANCE = 2.0e-6
CRPS_QUADRATURE_ORDERS = (128, 256, 512)
CRPS_DF_MINIMUM = 2.0
FROZEN_PANEL_NAME = "inference_panel.jsonl.gz"
FROZEN_TARGET_NAME = "scoring_targets.jsonl.gz"
FROZEN_EVIDENCE_NAME = "evidence.json"
FROZEN_SCORE_NAME = "team_scores.csv"
EASTERN = ZoneInfo("America/New_York")

SCHEDULE_FIELDS = (
    "id",
    "season",
    "week",
    "seasonType",
    "startDate",
    "neutralSite",
    "conferenceGame",
    "homeId",
    "homeTeam",
    "homeClassification",
    "homeConference",
    "awayId",
    "awayTeam",
    "awayClassification",
    "awayConference",
)
OUTCOME_FIELDS = ("completed", "homePoints", "awayPoints")
RANK_SCORE_FIELDS = (
    "nll",
    "crps",
    "expected_rank",
    "target_expected_rank",
    "expected_rank_error",
    "interval_80_low",
    "interval_80_high",
    "interval_80_width",
    "interval_80_target_mass",
    "entropy",
    "rank_variance",
    "rank_standard_deviation",
)
FORECAST_SCORE_FIELDS = (
    "expected_home_margin",
    "median_home_margin",
    "home_win_probability",
    "away_win_probability",
    "tie_probability",
    "margin_interval_50_low",
    "margin_interval_50_high",
    "margin_interval_80_low",
    "margin_interval_80_high",
    "margin_interval_95_low",
    "margin_interval_95_high",
    "crps",
    "crps_numerical_error_estimate",
    "crps_quadrature_order",
    "margin_nll",
    "median_absolute_error",
    "home_win_brier",
    "coverage_50",
    "coverage_80",
    "coverage_95",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).casefold() in {"true", "1", "yes"}


def _team_id_key(team_id: str) -> tuple[int, int | str]:
    try:
        return 0, int(team_id)
    except ValueError:
        return 1, team_id


def _game_id_key(game_id: str) -> tuple[int, int | str]:
    return _team_id_key(game_id)


def _parse_kickoff(value: str) -> tuple[datetime | None, str]:
    if not value:
        return None, "missing"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None, "invalid"
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None, "date_or_timestamp_without_timezone"
    return parsed.astimezone(UTC), "timestamp_with_timezone"


def _schedule_metadata(raw: dict[str, object]) -> dict[str, object]:
    """Normalize only outcome-independent schedule fields from a source row."""
    kickoff_raw = _text(raw.get("startDate"))
    kickoff, precision = _parse_kickoff(kickoff_raw)
    season_value = raw.get("season")
    try:
        season = int(season_value) if season_value is not None else None
    except (TypeError, ValueError):
        season = None
    fields: dict[str, object] = {
        "game_id": _text(raw.get("id")),
        "season": season,
        "week": None if raw.get("week") is None else _text(raw.get("week")),
        "season_type": _text(raw.get("seasonType")).casefold(),
        "kickoff_source": kickoff_raw,
        "kickoff_utc": None if kickoff is None else kickoff.isoformat(),
        "kickoff_precision": precision,
        "schedule_availability": "not_archived_retrospective_schedule",
        "home_id": _text(raw.get("homeId")),
        "home_team": _text(raw.get("homeTeam")),
        "home_subdivision": _text(raw.get("homeClassification")).casefold(),
        "home_conference": _text(raw.get("homeConference")),
        "away_id": _text(raw.get("awayId")),
        "away_team": _text(raw.get("awayTeam")),
        "away_subdivision": _text(raw.get("awayClassification")).casefold(),
        "away_conference": _text(raw.get("awayConference")),
        "neutral_site": _truthy(raw.get("neutralSite")),
        "conference_game": _truthy(raw.get("conferenceGame")),
    }
    fields["schedule_row_sha256"] = sha256_json(fields)
    return fields


def _source_paths(game_root: Path, season: int) -> tuple[Path, Path]:
    return game_root / f"{season}.json", game_root / f"{season}-fcs.json"


def load_schedule_metadata(
    game_root: Path, seasons: tuple[int, ...] = SEASONS
) -> tuple[dict[int, list[dict[str, object]]], dict[str, dict[str, object]]]:
    """Read schedule metadata without accessing completion or score fields."""
    by_season: dict[int, list[dict[str, object]]] = {}
    source_metadata: dict[str, dict[str, object]] = {}
    for season in seasons:
        merged: dict[str, dict[str, object]] = {}
        for path in _source_paths(game_root, season):
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise TypeError(f"CFBD schedule payload must be a list: {path}")
            file_rows: dict[str, dict[str, object]] = {}
            for item in payload:
                if not isinstance(item, dict):
                    raise TypeError(f"CFBD schedule row must be an object: {path}")
                metadata = _schedule_metadata(item)
                game_id = str(metadata["game_id"])
                if not game_id:
                    continue
                previous = merged.get(game_id)
                if previous is not None and previous != metadata:
                    raise ValueError(
                        f"conflicting schedule metadata for game {game_id}"
                    )
                merged[game_id] = metadata
                file_rows[game_id] = metadata
            ordered_rows = [
                file_rows[key] for key in sorted(file_rows, key=_game_id_key)
            ]
            source_metadata[path.name] = {
                "row_count": len(ordered_rows),
                "schedule_metadata_sha256": sha256_json(ordered_rows),
            }
        rows = [merged[key] for key in sorted(merged, key=_game_id_key)]
        if not rows:
            raise ValueError(f"no schedule rows found for season {season}")
        by_season[season] = rows
    return by_season, source_metadata


def _candidate_population(row: dict[str, object]) -> str | None:
    home = str(row["home_subdivision"])
    away = str(row["away_subdivision"])
    if home == "fbs" and away == "fbs":
        return PRIMARY_POPULATION
    if {home, away} == {"fbs", "fcs"}:
        return SECONDARY_POPULATION
    return None


def _valid_schedule_candidate(row: dict[str, object]) -> tuple[bool, str | None]:
    if row["season_type"] != "regular":
        return False, "not_regular_season"
    if _candidate_population(row) is None:
        return False, "not_fbs_fbs_or_fbs_fcs"
    if not row["game_id"] or not row["home_id"] or not row["away_id"]:
        return False, "missing_canonical_team_or_game_id"
    if row["kickoff_utc"] is None:
        return False, f"kickoff_{row['kickoff_precision']}"
    return True, None


def _previous_weekly_cutoff(kickoff: datetime) -> datetime:
    days_since_tuesday = (kickoff.weekday() - WEEKLY_CUTOFF_WEEKDAY) % 7
    candidate_date = kickoff.date() - timedelta(days=days_since_tuesday)
    candidate = datetime.combine(candidate_date, WEEKLY_CUTOFF_UTC_TIME, tzinfo=UTC)
    if candidate >= kickoff:
        candidate -= timedelta(days=7)
    return candidate


def _cutoff_for_game(cutoffs: list[datetime], kickoff: datetime) -> datetime:
    index = bisect_left(cutoffs, kickoff) - 1
    if index < 0:
        raise ValueError("weekly replay schedule has no strictly prior cutoff")
    return cutoffs[index]


def thanksgiving_date(season: int) -> date:
    """Return the fourth Thursday in November for a season."""
    first = date(season, 11, 1)
    offset = (3 - first.weekday()) % 7
    return first + timedelta(days=offset + 21)


def _is_army_navy(row: dict[str, object]) -> bool:
    def is_army(value: object) -> bool:
        normalized = re.sub(r"[^a-z]+", " ", _text(value).casefold()).strip()
        return normalized == "army" or normalized.startswith("army west point")

    def is_navy(value: object) -> bool:
        normalized = re.sub(r"[^a-z]+", " ", _text(value).casefold()).strip()
        return normalized == "navy" or normalized.startswith("navy midshipmen")

    return (is_army(row["home_team"]) and is_navy(row["away_team"])) or (
        is_navy(row["home_team"]) and is_army(row["away_team"])
    )


def game_phase(row: dict[str, object], last_championship_date: date) -> str:
    """Assign one non-overlapping game phase using final kickoff local date."""
    if _is_army_navy(row):
        return "army_navy"
    kickoff_raw = row.get("kickoff_utc")
    if not kickoff_raw:
        return "unknown_kickoff"
    kickoff = datetime.fromisoformat(str(kickoff_raw)).astimezone(EASTERN)
    local_day = kickoff.date()
    thanksgiving = thanksgiving_date(int(row["season"]))
    if (
        last_championship_date - timedelta(days=1)
        <= local_day
        <= last_championship_date
    ):
        return "conference_championship_period"
    if thanksgiving <= local_day <= thanksgiving + timedelta(days=3):
        return "rivalry_window"
    if local_day < thanksgiving:
        return "pre_thanksgiving"
    return "other_regular_season"


def checkpoint_phase(
    cutoff: datetime,
    season: int,
    last_championship_date: date,
    first_postseason_date: date,
) -> str:
    """Map a weekly/checkpoint anchor into the preregistered event windows."""
    local_day = cutoff.astimezone(EASTERN).date()
    thanksgiving = thanksgiving_date(season)
    if (
        thanksgiving - timedelta(days=21)
        <= local_day
        <= thanksgiving - timedelta(days=8)
    ):
        return "pre_thanksgiving"
    if (
        thanksgiving - timedelta(days=7)
        <= local_day
        <= thanksgiving + timedelta(days=3)
    ):
        return "rivalry_window"
    if last_championship_date - timedelta(days=7) <= local_day <= first_postseason_date:
        return "conference_championship_period"
    return "other_checkpoint"


def build_schedule_plan(
    game_root: Path, *, protocol_sha256: str, seasons: tuple[int, ...] = SEASONS
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Create a score-blind, deterministic weekly game assignment plan."""
    metadata_by_season, source_metadata = load_schedule_metadata(game_root, seasons)
    season_plans: list[dict[str, object]] = []
    for season in seasons:
        all_rows = metadata_by_season[season]
        candidates: list[dict[str, object]] = []
        excluded: list[dict[str, object]] = []
        for row in all_rows:
            valid, reason = _valid_schedule_candidate(row)
            if not valid:
                if row["home_subdivision"] in {"fbs", "fcs"} and row[
                    "away_subdivision"
                ] in {"fbs", "fcs"}:
                    excluded.append({**row, "plan_exclusion_reason": reason})
                continue
            candidates.append(row)
        if not candidates:
            raise ValueError(f"{season}: no eligible regular-season games in schedule")
        kickoff_values = [
            datetime.fromisoformat(str(row["kickoff_utc"])) for row in candidates
        ]
        first_cutoff = _previous_weekly_cutoff(min(kickoff_values))
        last_cutoff = _previous_weekly_cutoff(max(kickoff_values))
        cutoffs: list[datetime] = []
        cutoff = first_cutoff
        while cutoff <= last_cutoff:
            cutoffs.append(cutoff)
            cutoff += timedelta(days=7)
        planned_games: list[dict[str, object]] = []
        for row in candidates:
            kickoff = datetime.fromisoformat(str(row["kickoff_utc"]))
            assigned = _cutoff_for_game(cutoffs, kickoff)
            if assigned.isoformat() >= kickoff.isoformat():
                raise AssertionError(f"{row['game_id']}: cutoff is not pregame")
            planned_games.append(
                {
                    **row,
                    "assigned_cutoff": assigned.isoformat(),
                    "forecast_population": _candidate_population(row),
                    "phase": "pending_calendar_mapping",
                }
            )
        planned_games.sort(
            key=lambda row: (
                str(row["assigned_cutoff"]),
                str(row["kickoff_utc"]),
                _game_id_key(str(row["game_id"])),
            )
        )
        excluded.sort(key=lambda row: _game_id_key(str(row["game_id"])))
        season_plans.append(
            {
                "season": season,
                "cutoffs_utc": [value.isoformat() for value in cutoffs],
                "scheduled_game_count": len(planned_games),
                "scheduled_fbs_fbs_count": sum(
                    row["forecast_population"] == PRIMARY_POPULATION
                    for row in planned_games
                ),
                "scheduled_fbs_fcs_count": sum(
                    row["forecast_population"] == SECONDARY_POPULATION
                    for row in planned_games
                ),
                "games": planned_games,
                "excluded_schedule_rows": excluded,
            }
        )
    plan: dict[str, object] = {
        "schema_version": 1,
        "experiment": "issue-201-prebowl-forecast-comparison",
        "protocol_sha256": protocol_sha256,
        "weekly_cutoff_rule": {
            "weekday_utc": "Tuesday",
            "time_utc": "00:00:00",
            "assignment": "latest cutoff strictly before final scheduled kickoff",
            "outcome_fields_used_during_plan": False,
        },
        "schedule_source_files": sorted(source_metadata),
        "seasons": season_plans,
    }
    return plan, source_metadata


def attach_plan_phases(
    plan: dict[str, object], target_manifest_path: Path, schedule_windows_path: Path
) -> dict[str, object]:
    """Add event phase labels from fixed, dated source manifests."""
    target_rows = _read_csv(target_manifest_path)
    target_dates = {
        int(row["season"]): row
        for row in target_rows
        if row.get("subdivision") == "fbs"
    }
    window_rows = _read_csv(schedule_windows_path)
    window_dates = {int(row["season"]): row for row in window_rows}
    result = json.loads(canonical_json(plan))
    for season_plan in result["seasons"]:
        season = int(season_plan["season"])
        target_row = target_dates.get(season)
        window_row = window_dates.get(season)
        if target_row is None or window_row is None:
            raise ValueError(f"{season}: missing target date or schedule window")
        championship = date.fromisoformat(
            window_row["last_fbs_conference_championship_date"]
        )
        first_postseason = date.fromisoformat(
            window_row["first_fbs_postseason_game_date"]
        )
        for game in season_plan["games"]:
            game["phase"] = game_phase(game, championship)
            game["after_pre_bowl_target_date"] = datetime.fromisoformat(
                str(game["kickoff_utc"])
            ) >= datetime.combine(
                date.fromisoformat(target_row["snapshot_date"]), time.min, tzinfo=UTC
            )
        season_plan["cutoff_phases"] = [
            {
                "cutoff_utc": cutoff,
                "phase": checkpoint_phase(
                    datetime.fromisoformat(cutoff),
                    season,
                    championship,
                    first_postseason,
                ),
            }
            for cutoff in season_plan["cutoffs_utc"]
        ]
    return result


def _outcome_record(raw: dict[str, object]) -> dict[str, object]:
    metadata = _schedule_metadata(raw)
    completed = _truthy(raw.get("completed"))
    try:
        home_points = None if raw.get("homePoints") is None else int(raw["homePoints"])
        away_points = None if raw.get("awayPoints") is None else int(raw["awayPoints"])
    except (TypeError, ValueError):
        home_points = away_points = None
    outcome = {
        "completed": completed,
        "home_points": home_points,
        "away_points": away_points,
    }
    return {
        **metadata,
        **outcome,
        "game_row_sha256": sha256_json({**metadata, **outcome}),
    }


def load_outcome_games(
    game_root: Path, seasons: tuple[int, ...] = SEASONS
) -> tuple[dict[int, list[dict[str, object]]], dict[str, dict[str, object]]]:
    """Read full result fields only after a score-blind plan is available."""
    by_season: dict[int, list[dict[str, object]]] = {}
    source_files: dict[str, dict[str, object]] = {}
    for season in seasons:
        merged: dict[str, dict[str, object]] = {}
        for path in _source_paths(game_root, season):
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise TypeError(f"CFBD schedule payload must be a list: {path}")
            file_ids: list[str] = []
            for item in payload:
                if not isinstance(item, dict):
                    raise TypeError(f"CFBD schedule row must be an object: {path}")
                record = _outcome_record(item)
                game_id = str(record["game_id"])
                if not game_id:
                    continue
                previous = merged.get(game_id)
                if previous is not None and previous != record:
                    raise ValueError(f"conflicting schedule/outcome rows for {game_id}")
                merged[game_id] = record
                file_ids.append(game_id)
            source_files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "row_count": len(file_ids),
            }
        by_season[season] = [merged[key] for key in sorted(merged, key=_game_id_key)]
    return by_season, source_files


def verify_score_blind_plan(
    plan: dict[str, object],
    plan_path: Path,
    plan_manifest_path: Path,
    game_root: Path,
    *,
    protocol_sha256: str,
    target_snapshot_manifest_path: Path,
    schedule_windows_path: Path,
    seasons: tuple[int, ...] = SEASONS,
) -> tuple[dict[int, list[dict[str, object]]], dict[str, dict[str, object]]]:
    """Validate frozen metadata assignments before loading score-bearing rows."""
    plan_manifest = json.loads(plan_manifest_path.read_text(encoding="utf-8"))
    actual_plan_sha = sha256_file(plan_path)
    if plan_manifest.get("plan_sha256") != actual_plan_sha:
        raise ValueError("schedule plan changed after the pre-scoring freeze")
    if plan.get("protocol_sha256") != protocol_sha256:
        raise ValueError("schedule plan was frozen under a different protocol")
    expected, metadata_sources = build_schedule_plan(
        game_root, protocol_sha256=protocol_sha256, seasons=seasons
    )
    expected = attach_plan_phases(
        expected, target_snapshot_manifest_path, schedule_windows_path
    )
    if canonical_json(plan) != canonical_json(expected):
        raise ValueError(
            "score-blind schedule metadata no longer matches the frozen plan"
        )
    if plan_manifest.get("source_schedule_metadata") != metadata_sources:
        raise ValueError("schedule metadata hashes changed after the plan was frozen")
    return load_outcome_games(game_root, seasons)


def filter_evidence_rows(
    rows: list[dict[str, object]], cutoff: datetime, policy: Policy
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Apply the common 48-hour lag and the matched postseason policy."""
    cutoff_utc = cutoff.astimezone(UTC)
    lag_boundary = cutoff_utc - RESULT_AVAILABILITY_LAG
    included: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for row in rows:
        kickoff_raw = row.get("kickoff_utc")
        if not kickoff_raw:
            continue
        kickoff = datetime.fromisoformat(str(kickoff_raw)).astimezone(UTC)
        if kickoff > cutoff_utc:
            continue
        if kickoff > lag_boundary:
            if row.get("completed"):
                excluded.append(
                    {**row, "evidence_exclusion_reason": "within_48_hour_safety_lag"}
                )
            continue
        if (
            not row.get("completed")
            or row.get("home_points") is None
            or row.get("away_points") is None
        ):
            continue
        if row.get("home_subdivision") not in {"fbs", "fcs"} or row.get(
            "away_subdivision"
        ) not in {"fbs", "fcs"}:
            continue
        if row.get("season_type") not in {"regular", "postseason"}:
            continue
        fbs_involved = "fbs" in {
            str(row.get("home_subdivision")),
            str(row.get("away_subdivision")),
        }
        if (
            policy == "pre_bowl"
            and row.get("season_type") == "postseason"
            and fbs_involved
        ):
            excluded.append(
                {**row, "evidence_exclusion_reason": "pre_bowl_fbs_postseason_policy"}
            )
            continue
        included.append(row)
    included.sort(
        key=lambda row: (str(row["kickoff_utc"]), _game_id_key(str(row["game_id"])))
    )
    excluded.sort(key=lambda row: _game_id_key(str(row["game_id"])))
    return included, excluded


def game_from_outcome(row: dict[str, object]) -> Game:
    if (
        not row.get("completed")
        or row.get("home_points") is None
        or row.get("away_points") is None
    ):
        raise ValueError(f"{row.get('game_id')}: game outcome is unavailable")
    return Game(
        str(row["game_id"]),
        str(row["home_id"]),
        str(row["away_id"]),
        str(row["home_subdivision"]),  # type: ignore[arg-type]
        str(row["away_subdivision"]),  # type: ignore[arg-type]
        int(row["home_points"]),
        int(row["away_points"]),
        bool(row["neutral_site"]),
    )


def _tail_first_moment(a: float, degrees_of_freedom: float) -> float:
    """Compute E[(T-a)+] for a positive Student-t threshold."""
    if a < 0 or degrees_of_freedom <= 1:
        raise ValueError("Student-t positive-tail moment needs a >= 0 and df > 1")
    log_constant = (
        gammaln((degrees_of_freedom + 1.0) / 2.0)
        - gammaln(degrees_of_freedom / 2.0)
        - 0.5 * math.log(degrees_of_freedom * math.pi)
    )
    first_moment = (
        math.exp(log_constant) * degrees_of_freedom / (degrees_of_freedom - 1.0)
    )
    first_moment *= (1.0 + (a * a) / degrees_of_freedom) ** (
        -(degrees_of_freedom - 1.0) / 2.0
    )
    survival = float(stdtr(degrees_of_freedom, -a))
    return max(0.0, first_moment - a * survival)


def _component_absolute_error_expectation(
    locations: np.ndarray,
    observation: float,
    scale: float,
    degrees_of_freedom: float,
) -> np.ndarray:
    standardized = np.abs((locations - observation) / scale)
    cdf = stdtr(degrees_of_freedom, standardized)
    log_constant = (
        gammaln((degrees_of_freedom + 1.0) / 2.0)
        - gammaln(degrees_of_freedom / 2.0)
        - 0.5 * math.log(degrees_of_freedom * math.pi)
    )
    first_abs_tail = (
        math.exp(log_constant) * degrees_of_freedom / (degrees_of_freedom - 1.0)
    )
    first_abs_tail = first_abs_tail * np.power(
        1.0 + (standardized * standardized) / degrees_of_freedom,
        -(degrees_of_freedom - 1.0) / 2.0,
    )
    standardized_expectation = standardized * (2.0 * cdf - 1.0) + 2.0 * first_abs_tail
    return scale * standardized_expectation


@lru_cache(maxsize=10)
def _legendre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


@lru_cache(maxsize=32)
def _cdf_quadrature_surface(
    location_bytes: bytes,
    scale: float,
    degrees_of_freedom: float,
    order: int,
    tail_multiple: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cache deterministic CDF values at one fixed Gauss–Legendre rule."""
    locations = np.frombuffer(location_bytes, dtype=np.float64)
    lower = float(locations.min() - tail_multiple * scale)
    upper = float(locations.max() + tail_multiple * scale)
    nodes, weights = _legendre_rule(order)
    values = lower + (nodes + 1.0) * (upper - lower) / 2.0
    integral_weights = weights * (upper - lower) / 2.0
    cdf_values = stdtr(
        degrees_of_freedom,
        (values[:, None] - locations[None, :]) / scale,
    )
    tail_bound = 2.0 * scale * _tail_first_moment(tail_multiple, degrees_of_freedom)
    values.setflags(write=False)
    integral_weights.setflags(write=False)
    cdf_values.setflags(write=False)
    return cdf_values, integral_weights, tail_bound


def _cdf_pair_integral(
    locations: np.ndarray,
    weights: np.ndarray,
    scale: float,
    degrees_of_freedom: float,
    order: int,
    tail_multiple: float,
) -> tuple[float, float]:
    cdf_values, integral_weights, tail_bound = _cdf_quadrature_surface(
        locations.tobytes(), scale, degrees_of_freedom, order, tail_multiple
    )
    cdf = cdf_values @ weights
    value = float(np.dot(integral_weights, cdf * (1.0 - cdf)))
    return value, tail_bound


def mixture_margin_crps(
    locations: np.ndarray,
    weights: np.ndarray,
    observation: float,
    scale: float,
    degrees_of_freedom: float,
    *,
    tolerance: float = CRPS_ABSOLUTE_TOLERANCE,
) -> tuple[float, float, int, float]:
    """Score a full equal-scale Student-t location mixture deterministically."""
    locations = np.asarray(locations, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        locations.ndim != 1
        or weights.shape != locations.shape
        or not len(locations)
        or not np.isfinite(locations).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isclose(float(weights.sum()), 1.0, atol=1e-10, rtol=0.0)
        or not np.isfinite(observation)
        or scale <= 0
        or degrees_of_freedom <= CRPS_DF_MINIMUM
    ):
        raise ValueError("invalid mixture or Student-t parameters for margin CRPS")
    absolute = _component_absolute_error_expectation(
        locations, observation, scale, degrees_of_freedom
    )
    first_term = float(np.dot(weights, absolute))
    tail_multiple = CRPS_TAIL_MULTIPLE
    tail_bound = 2.0 * scale * _tail_first_moment(tail_multiple, degrees_of_freedom)
    while tail_bound > tolerance / 2.0:
        tail_multiple *= 2.0
        if tail_multiple > 1_000_000.0:
            raise FloatingPointError(
                "Student-t CRPS tail bound could not meet tolerance"
            )
        tail_bound = 2.0 * scale * _tail_first_moment(tail_multiple, degrees_of_freedom)
    previous: float | None = None
    best_value = math.nan
    error_estimate = math.inf
    selected_order = 0
    for order in CRPS_QUADRATURE_ORDERS:
        pair_integral, tail_bound = _cdf_pair_integral(
            locations, weights, scale, degrees_of_freedom, order, tail_multiple
        )
        if previous is not None:
            error_estimate = abs(pair_integral - previous) + tail_bound
            if error_estimate <= tolerance:
                best_value = first_term - pair_integral
                selected_order = order
                break
        previous = pair_integral
    if selected_order == 0:
        raise FloatingPointError(
            f"margin CRPS quadrature error {error_estimate:.6g} exceeds {tolerance}"
        )
    if best_value < -tolerance:
        raise FloatingPointError(f"mixture CRPS became negative: {best_value}")
    return max(0.0, best_value), error_estimate, selected_order, tail_multiple


def mixture_margin_nll(
    locations: np.ndarray,
    weights: np.ndarray,
    observation: float,
    scale: float,
    degrees_of_freedom: float,
) -> float:
    """Return continuous-mixture NLL using stable log arithmetic."""
    locations = np.asarray(locations, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if locations.shape != weights.shape or not len(locations):
        raise ValueError("locations and mixture weights must be matching vectors")
    positive = weights > 0
    if not positive.any():
        raise ValueError("margin mixture has no positive weight")
    standardized = (observation - locations[positive]) / scale
    log_constant = (
        gammaln((degrees_of_freedom + 1.0) / 2.0)
        - gammaln(degrees_of_freedom / 2.0)
        - 0.5 * math.log(degrees_of_freedom * math.pi)
        - math.log(scale)
    )
    log_density = log_constant - (degrees_of_freedom + 1.0) / 2.0 * np.log1p(
        standardized * standardized / degrees_of_freedom
    )
    log_mixture_density = float(logsumexp(np.log(weights[positive]) + log_density))
    return -log_mixture_density


def score_rank_pmf(forecast: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Mirror the retained #192 rank PMF score contract on common support."""
    prediction = np.asarray(forecast, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if prediction.ndim != 1 or truth.shape != prediction.shape or not len(truth):
        raise ValueError(
            "forecast and target must have identical one-dimensional support"
        )
    if (
        not np.isfinite(prediction).all()
        or not np.isfinite(truth).all()
        or np.any(prediction < 0)
        or np.any(truth < 0)
        or not np.isclose(float(prediction.sum()), 1.0, atol=1e-8, rtol=0.0)
        or not np.isclose(float(truth.sum()), 1.0, atol=1e-8, rtol=0.0)
    ):
        raise ValueError("rank forecast and target must be normalized nonnegative PMFs")
    prediction = prediction / prediction.sum()
    truth = truth / truth.sum()
    ranks = np.arange(1, len(prediction) + 1, dtype=np.float64)
    cumulative_prediction = np.cumsum(prediction)
    cumulative_target = np.cumsum(truth)

    def quantile(probability: float) -> int:
        return int(np.searchsorted(cumulative_prediction, probability, side="left") + 1)

    expected = float(np.dot(ranks, prediction))
    target_expected = float(np.dot(ranks, truth))
    variance = float(np.dot((ranks - expected) ** 2, prediction))
    low, high = quantile(0.1), quantile(0.9)
    inside = (ranks >= low) & (ranks <= high)
    positive = prediction > 0
    return {
        "nll": float(-np.sum(truth * np.log(np.maximum(prediction, 1e-15)))),
        "crps": float(np.mean((cumulative_prediction - cumulative_target) ** 2)),
        "expected_rank": expected,
        "target_expected_rank": target_expected,
        "expected_rank_error": abs(expected - target_expected),
        "interval_80_low": float(low),
        "interval_80_high": float(high),
        "interval_80_width": float(high - low),
        "interval_80_target_mass": float(truth[inside].sum()),
        "entropy": float(-np.sum(prediction[positive] * np.log(prediction[positive]))),
        "rank_variance": variance,
        "rank_standard_deviation": float(np.sqrt(variance)),
    }


def target_pmf_from_row(row: dict[str, str]) -> np.ndarray:
    """Expand the #204 rank/count/probability PMF to full integer support."""
    population = int(row["team_population"])
    if population < 1:
        raise ValueError("target population must be positive")
    pmf = np.zeros(population, dtype=np.float64)
    points = json.loads(row["pmf"])
    if not isinstance(points, list):
        raise TypeError("target PMF field must be a list")
    for point in points:
        rank = int(point["rank"])
        probability = float(point["probability"])
        if not 1 <= rank <= population:
            raise ValueError(
                f"{row['season']}/{row['team_id']}: target rank outside support"
            )
        if pmf[rank - 1] != 0:
            raise ValueError(f"{row['season']}/{row['team_id']}: duplicate PMF rank")
        pmf[rank - 1] = probability
    if (
        not np.isfinite(pmf).all()
        or np.any(pmf < 0)
        or not np.isclose(float(pmf.sum()), 1.0, atol=1e-8, rtol=0.0)
    ):
        raise ValueError(f"{row['season']}/{row['team_id']}: invalid target PMF")
    return pmf / pmf.sum()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_pre_bowl_targets(
    target_path: Path, issue204_manifest_path: Path
) -> tuple[dict[int, dict[str, dict[str, object]]], dict[str, object]]:
    """Check #204 readiness/hash before loading corrected target PMFs."""
    manifest = json.loads(issue204_manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("downstream_ready") or manifest.get("status") != "complete":
        raise ValueError("#204 corrected target manifest is not ready")
    support = manifest.get("target_support_validation")
    if (
        not isinstance(support, dict)
        or support.get("status") != "passed"
        or int(support.get("anomaly_count", -1)) != 0
    ):
        raise ValueError("#204 target support gate did not pass")
    artifact = target_path.name
    expected = manifest.get("outputs", {}).get(artifact)
    if not isinstance(expected, dict):
        raise TypeError(f"#204 manifest does not bind {artifact}")
    actual_sha = sha256_file(target_path)
    if actual_sha != expected.get("sha256"):
        raise ValueError("#204 pre-bowl target file hash differs from its manifest")
    targets: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in _read_csv(target_path):
        if row.get("subdivision") != "fbs" or int(row["season"]) not in SEASONS:
            continue
        season = int(row["season"])
        team_id = str(row["team_id"])
        if team_id in targets[season]:
            raise ValueError(f"{season}/{team_id}: duplicate pre-bowl target")
        pmf = target_pmf_from_row(row)
        targets[season][team_id] = {
            "team_name": row["team_name"],
            "population": int(row["team_population"]),
            "pmf": pmf,
            "row_sha256": sha256_json(row),
        }
    if set(targets) != set(SEASONS):
        raise ValueError("#204 pre-bowl target file does not cover 2022–2025")
    return dict(targets), {
        "path_label": "issue_204_corrected_pre_bowl_target_distributions.csv",
        "bytes": target_path.stat().st_size,
        "sha256": actual_sha,
        "manifest_sha256": sha256_file(issue204_manifest_path),
        "row_count_by_season": {
            str(season): len(targets[season]) for season in SEASONS
        },
    }


def _prior_rows_by_arm(
    bundle: dict[str, object], season: int
) -> dict[str, dict[str, dict[str, object]]]:
    raw_priors = bundle.get("priors")
    if not isinstance(raw_priors, list):
        raise TypeError("issue 200 bundle has no prior list")
    by_arm: dict[str, dict[str, dict[str, object]]] = {}
    for arm, family, policy in MODEL_ARMS:
        selected: dict[str, dict[str, object]] = {}
        for raw in raw_priors:
            if not isinstance(raw, dict):
                raise TypeError("issue 200 prior row must be an object")
            if (
                raw.get("season") != season
                or raw.get("policy") != policy
                or raw.get("model_family") != family
            ):
                continue
            team_id = str(raw.get("team_id", ""))
            pmf = np.asarray(raw.get("pmf"), dtype=np.float64)
            population = int(raw.get("population", 0))
            if (
                not team_id
                or team_id in selected
                or pmf.ndim != 1
                or len(pmf) != population
                or not np.isfinite(pmf).all()
                or np.any(pmf < 0)
                or not np.isclose(float(pmf.sum()), 1.0, atol=1e-8, rtol=0.0)
            ):
                raise ValueError(f"{season}/{arm}: invalid or duplicate prior row")
            selected[team_id] = raw
        if not selected:
            raise ValueError(f"{season}/{arm}: no issue 200 prior rows")
        by_arm[arm] = selected
    populations = {name: set(rows) for name, rows in by_arm.items()}
    first_name = next(iter(populations))
    if any(ids != populations[first_name] for ids in populations.values()):
        raise ValueError(f"{season}: four issue 200 arms have different FBS teams")
    return by_arm


def _common_target_teams(
    season: int,
    prior_rows: dict[str, dict[str, dict[str, object]]],
    targets: dict[int, dict[str, dict[str, object]]],
) -> tuple[list[str], list[dict[str, object]]]:
    all_ids = set().union(*(set(rows) for rows in prior_rows.values())) | set(
        targets.get(season, {})
    )
    common_sets = [set(rows) for rows in prior_rows.values()]
    common_sets.append(set(targets.get(season, {})))
    common = set.intersection(*common_sets)
    excluded: list[dict[str, object]] = []
    for team_id in sorted(all_ids - common, key=_team_id_key):
        excluded.append(
            {
                "season": season,
                "team_id": team_id,
                "reason": "missing_from_one_or_more_four_arm_or_pre_bowl_target_populations",
            }
        )
    for team_id in sorted(common, key=_team_id_key):
        population_values = {
            int(prior_rows[arm][team_id]["population"]) for arm, _, _ in MODEL_ARMS
        }
        population_values.add(int(targets[season][team_id]["population"]))
        if len(population_values) != 1:
            common.remove(team_id)
            excluded.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "reason": "rank_support_differs_across_arms_or_pre_bowl_target",
                    "observed_supports": ",".join(map(str, sorted(population_values))),
                }
            )
    return sorted(common, key=_team_id_key), excluded


def validate_forward_only_fit(
    manifest_path: Path, fitted_models_path: Path
) -> dict[str, object]:
    """Confirm issue 200's four fit origins stop before every target season."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fitted = json.loads(fitted_models_path.read_text(encoding="utf-8"))
    if manifest.get("fitting_status") != "complete":
        raise ValueError("issue 200 model bundle is not marked complete")
    expected_training = list(range(2003, 2022))
    if manifest.get("training_seasons") != expected_training:
        raise ValueError("issue 200 training seasons differ from 2003–2021")
    result: dict[str, object] = {}
    rolling = fitted.get("rolling_models")
    if not isinstance(rolling, dict):
        raise TypeError("issue 200 fit records do not contain rolling models")
    for season in SEASONS:
        policy_rows = rolling.get(str(season))
        if not isinstance(policy_rows, dict) or set(policy_rows) != {
            "existing",
            "pre_bowl",
        }:
            raise ValueError(f"{season}: issue 200 lacks both policy fit records")
        result[str(season)] = {}
        for policy in ("existing", "pre_bowl"):
            entry = policy_rows[policy]
            trained_through = int(entry["trained_through_season"])
            if trained_through != season - 1:
                raise ValueError(f"{season}/{policy}: fit saw season {trained_through}")
            result[str(season)][policy] = {
                "trained_through_season": trained_through,
                "training_seasons": expected_training,
                "forward_only": True,
            }
    return result


def _read_target_snapshot_dates(path: Path) -> dict[int, date]:
    result: dict[int, date] = {}
    for row in _read_csv(path):
        if row.get("subdivision") != "fbs":
            continue
        season = int(row["season"])
        if season not in SEASONS:
            continue
        if season in result:
            raise ValueError(f"{season}: duplicate FBS target snapshot date")
        result[season] = date.fromisoformat(row["snapshot_date"])
    if set(result) != set(SEASONS):
        raise ValueError("issue 200 target snapshot manifest misses a target season")
    return result


def _fcs_names(rows: list[dict[str, object]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for row in rows:
        for prefix in ("home", "away"):
            if row.get(f"{prefix}_subdivision") == "fcs":
                team_id = str(row[f"{prefix}_id"])
                name = str(row.get(f"{prefix}_team") or team_id)
                prior = names.get(team_id)
                if prior is not None and prior != name:
                    raise ValueError(f"FCS team {team_id} has conflicting names")
                names[team_id] = name
    return names


def _build_posterior_teams(
    result_pmfs: dict[str, np.ndarray],
    arm_priors: dict[str, dict[str, object]],
    fcs_names: dict[str, str],
) -> dict[str, Team]:
    teams: dict[str, Team] = {}
    for team_id, row in arm_priors.items():
        if team_id not in result_pmfs:
            raise ValueError(f"posterior is missing FBS team {team_id}")
        pmf = np.asarray(result_pmfs[team_id], dtype=np.float64)
        if (
            pmf.ndim != 1
            or len(pmf) != int(row["population"])
            or not np.isfinite(pmf).all()
            or np.any(pmf < 0)
            or not np.isclose(float(pmf.sum()), 1.0, atol=1e-8, rtol=0.0)
        ):
            raise ValueError(f"{team_id}: posterior FBS PMF is not normalized")
        teams[team_id] = Team(team_id, str(row["team_name"]), "fbs", pmf)
    for team_id, name in fcs_names.items():
        if team_id not in result_pmfs:
            raise ValueError(f"posterior is missing FCS fallback team {team_id}")
        pmf = np.asarray(result_pmfs[team_id], dtype=np.float64)
        if (
            pmf.ndim != 1
            or not len(pmf)
            or not np.isfinite(pmf).all()
            or np.any(pmf < 0)
            or not np.isclose(float(pmf.sum()), 1.0, atol=1e-8, rtol=0.0)
        ):
            raise ValueError(f"{team_id}: posterior FCS PMF is not normalized")
        teams[team_id] = Team(team_id, name, "fcs", pmf)
    return teams


def _evidence_hashes(
    included: list[dict[str, object]], excluded: list[dict[str, object]]
) -> dict[str, object]:
    included_ids = sorted((str(row["game_id"]) for row in included), key=_game_id_key)
    included_rows = [
        [str(row["game_id"]), str(row["game_row_sha256"])]
        for row in sorted(included, key=lambda row: _game_id_key(str(row["game_id"])))
    ]
    excluded_ids = sorted((str(row["game_id"]) for row in excluded), key=_game_id_key)
    return {
        "included_game_ids": included_ids,
        "included_game_ids_sha256": sha256_json(included_ids),
        "included_game_rows_sha256": sha256_json(included_rows),
        "included_game_count": len(included_ids),
        "excluded_game_ids": excluded_ids,
        "excluded_game_ids_sha256": sha256_json(excluded_ids),
        "excluded_game_count": len(excluded_ids),
    }


def _evidence_record(
    season: int,
    cutoff: datetime,
    policy: Policy,
    included: list[dict[str, object]],
    excluded: list[dict[str, object]],
    replay_diagnostics: dict[str, dict[str, object]],
    total_team_count: int,
    fallback_team_count: int,
) -> dict[str, object]:
    hashes = _evidence_hashes(included, excluded)
    latest_kickoff = max((str(row["kickoff_utc"]) for row in included), default=None)
    return {
        "season": season,
        "cutoff_utc": cutoff.isoformat(),
        "policy": policy,
        "safety_lag_hours": 48,
        **hashes,
        "latest_included_kickoff_utc": latest_kickoff,
        "replay_diagnostics": replay_diagnostics,
        "posterior_team_count": total_team_count,
        "fcs_fallback_team_count": fallback_team_count,
    }


def _run_prediction_job(job: dict[str, object]) -> dict[str, object]:
    return _prediction_row(**job)  # type: ignore[arg-type]


def _parallel_prediction_rows(
    jobs: list[dict[str, object]], *, max_workers: int = 8
) -> list[dict[str, object]]:
    if len(jobs) < 4:
        return [_run_prediction_job(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as executor:
        return list(executor.map(_run_prediction_job, jobs))


def _predictive_summary_from_components(
    locations: np.ndarray,
    weights: np.ndarray,
    likelihood: LikelihoodV1,
) -> PredictiveMarginSummary:
    """Batch exact mixture quantiles using the canonical safeguarded Newton solve."""
    probabilities = np.asarray(
        [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975], dtype=np.float64
    )
    tail = max(
        float(student_t.ppf(1.0 - 1.0e-12, likelihood.degrees_of_freedom)),
        32.0,
    )
    span = max(likelihood.scale * tail, 1.0)
    lower = np.full(len(probabilities), float(locations.min() - span))
    upper = np.full(len(probabilities), float(locations.max() + span))
    log_constant = (
        gammaln((likelihood.degrees_of_freedom + 1.0) / 2.0)
        - gammaln(likelihood.degrees_of_freedom / 2.0)
        - 0.5 * math.log(likelihood.degrees_of_freedom * math.pi)
        - math.log(likelihood.scale)
    )
    location_mean = float(np.dot(weights, locations))
    location_variance = float(np.dot(weights, (locations - location_mean) ** 2))
    effective_scale = likelihood.scale
    if likelihood.degrees_of_freedom > 2.0:
        effective_scale = math.sqrt(
            likelihood.scale**2
            + location_variance
            * (likelihood.degrees_of_freedom - 2.0)
            / likelihood.degrees_of_freedom
        )
    values = location_mean + effective_scale * student_t.ppf(
        probabilities, likelihood.degrees_of_freedom
    )
    active = np.ones(len(values), dtype=bool)
    for _ in range(64):
        standardized = (values[:, None] - locations[None, :]) / likelihood.scale
        cdf = stdtr(likelihood.degrees_of_freedom, standardized) @ weights
        density = (
            np.exp(
                log_constant
                - (likelihood.degrees_of_freedom + 1.0)
                / 2.0
                * np.log1p(standardized * standardized / likelihood.degrees_of_freedom)
            )
            @ weights
        )
        converged = np.abs(cdf - probabilities) <= 1.0e-12
        active &= ~converged
        if not active.any():
            break
        below = cdf < probabilities
        lower = np.where(active & below, values, lower)
        upper = np.where(active & ~below, values, upper)
        proposal = values - (cdf - probabilities) / density
        safe = np.isfinite(proposal) & (proposal > lower) & (proposal < upper)
        proposed_values = np.where(safe, proposal, (lower + upper) / 2.0)
        narrow = (upper - lower) <= 1.0e-10
        values = np.where(active & ~narrow, proposed_values, values)
        values = np.where(active & narrow, (lower + upper) / 2.0, values)
        active &= ~narrow
    if active.any():
        raise FloatingPointError("predictive mixture quantiles did not converge")
    cdf_at_zero = float(
        np.dot(
            weights,
            stdtr(
                likelihood.degrees_of_freedom,
                (0.0 - locations) / likelihood.scale,
            ),
        )
    )
    home_win, away_win = win_probabilities(cdf_at_zero)
    return PredictiveMarginSummary(
        expected_home_margin=location_mean,
        median_home_margin=float(values[3]),
        home_win_probability=home_win,
        away_win_probability=away_win,
        tie_probability=0.0,
        margin_interval_50=(float(values[2]), float(values[4])),
        margin_interval_80=(float(values[1]), float(values[5])),
        margin_interval_95=(float(values[0]), float(values[6])),
    )


def _prediction_row(
    *,
    metadata: dict[str, object],
    outcome: dict[str, object],
    arm: str,
    policy: Policy,
    family: PriorFamily,
    posterior_teams: dict[str, Team],
    likelihood: LikelihoodV1,
    evidence: dict[str, object],
    stage: str,
    cutoff: str,
) -> dict[str, object]:
    home_id, away_id = str(metadata["home_id"]), str(metadata["away_id"])
    if home_id not in posterior_teams or away_id not in posterior_teams:
        raise KeyError(f"{metadata['game_id']}: a team has no posterior PMF")
    home, away = posterior_teams[home_id], posterior_teams[away_id]
    scheduled = ScheduledGame(
        game_id=str(metadata["game_id"]),
        home_id=home_id,
        away_id=away_id,
        home_subdivision=str(metadata["home_subdivision"]),  # type: ignore[arg-type]
        away_subdivision=str(metadata["away_subdivision"]),  # type: ignore[arg-type]
        neutral_site=bool(metadata["neutral_site"]),
        season_type=str(metadata["season_type"]),
        date=str(metadata["kickoff_utc"]),
        schedule_status="completed_source_schedule",
    )
    locations, weights = predictive_components(scheduled, home, away, likelihood)
    summary = _predictive_summary_from_components(locations, weights, likelihood)
    actual_margin = int(outcome["home_points"]) - int(outcome["away_points"])
    crps, crps_error, crps_order, crps_tail_multiple = mixture_margin_crps(
        locations,
        weights,
        float(actual_margin),
        likelihood.scale,
        likelihood.degrees_of_freedom,
    )
    margin_nll = mixture_margin_nll(
        locations,
        weights,
        float(actual_margin),
        likelihood.scale,
        likelihood.degrees_of_freedom,
    )
    actual_home_win = 1.0 if actual_margin > 0 else 0.0 if actual_margin < 0 else 0.5
    return {
        "forecast_id": sha256_json(
            {
                "stage": stage,
                "season": metadata["season"],
                "game_id": metadata["game_id"],
                "cutoff": cutoff,
                "model_arm": arm,
            }
        ),
        "forecast_stage": stage,
        "season": metadata["season"],
        "cutoff_utc": cutoff,
        "game_id": metadata["game_id"],
        "phase": metadata.get("phase"),
        "forecast_population": metadata.get("forecast_population"),
        "home_id": home_id,
        "home_team": metadata["home_team"],
        "home_subdivision": metadata["home_subdivision"],
        "away_id": away_id,
        "away_team": metadata["away_team"],
        "away_subdivision": metadata["away_subdivision"],
        "kickoff_source": metadata["kickoff_source"],
        "kickoff_utc": metadata["kickoff_utc"],
        "kickoff_precision": metadata["kickoff_precision"],
        "schedule_availability": metadata["schedule_availability"],
        "neutral_site": metadata["neutral_site"],
        "conference_game": metadata["conference_game"],
        "actual_home_points": outcome["home_points"],
        "actual_away_points": outcome["away_points"],
        "actual_home_margin": actual_margin,
        "model_arm": arm,
        "policy": policy,
        "prior_family": family,
        "evidence_game_count": evidence["included_game_count"],
        "evidence_ids_sha256": evidence["included_game_ids_sha256"],
        "evidence_rows_sha256": evidence["included_game_rows_sha256"],
        "expected_home_margin": summary.expected_home_margin,
        "median_home_margin": summary.median_home_margin,
        "home_win_probability": summary.home_win_probability,
        "away_win_probability": summary.away_win_probability,
        "tie_probability": summary.tie_probability,
        "margin_interval_50_low": summary.margin_interval_50[0],
        "margin_interval_50_high": summary.margin_interval_50[1],
        "margin_interval_80_low": summary.margin_interval_80[0],
        "margin_interval_80_high": summary.margin_interval_80[1],
        "margin_interval_95_low": summary.margin_interval_95[0],
        "margin_interval_95_high": summary.margin_interval_95[1],
        "crps": crps,
        "crps_numerical_error_estimate": crps_error,
        "crps_quadrature_order": crps_order,
        "crps_tail_multiple": crps_tail_multiple,
        "margin_nll": margin_nll,
        "median_absolute_error": abs(summary.median_home_margin - actual_margin),
        "home_win_brier": (summary.home_win_probability - actual_home_win) ** 2,
        "coverage_50": int(
            summary.margin_interval_50[0]
            <= actual_margin
            <= summary.margin_interval_50[1]
        ),
        "coverage_80": int(
            summary.margin_interval_80[0]
            <= actual_margin
            <= summary.margin_interval_80[1]
        ),
        "coverage_95": int(
            summary.margin_interval_95[0]
            <= actual_margin
            <= summary.margin_interval_95[1]
        ),
    }


def frozen_panel_identity(panel_rows: list[dict[str, object]]) -> str:
    """Hash every retained PMF arm without normalizing or altering values."""
    digest = hashlib.sha256()
    ordered = sorted(
        panel_rows,
        key=lambda row: (
            int(row["season"]),
            int(row["checkpoint"]),
            str(row["subdivision"]),
            _team_id_key(str(row["team_id"])),
        ),
    )
    for row in ordered:
        payload = {
            "season": row["season"],
            "checkpoint": row["checkpoint"],
            "subdivision": row["subdivision"],
            "team_id": row["team_id"],
            "context_prior_pmf": row["context_prior_pmf"],
            "context_posterior_pmf": row["context_posterior_pmf"],
            "history_prior_pmf": row["history_prior_pmf"],
            "history_posterior_pmf": row["history_posterior_pmf"],
            "crossed_message_pmfs": row["crossed_message_pmfs"],
        }
        digest.update(canonical_json(payload).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_gzip_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _load_frozen_panel(
    frozen_root: Path,
    outcome_rows_by_season: dict[int, list[dict[str, object]]],
    game_source_files: dict[str, dict[str, object]],
) -> tuple[
    list[dict[str, object]],
    dict[int, dict[str, np.ndarray]],
    dict[tuple[int, int], dict[str, object]],
    dict[tuple[int, int, str, str], dict[str, str]],
    str,
    dict[str, object],
]:
    manifest_path = frozen_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_outputs = manifest.get("output_sha256", {})
    for name in (
        FROZEN_PANEL_NAME,
        FROZEN_TARGET_NAME,
        FROZEN_EVIDENCE_NAME,
        FROZEN_SCORE_NAME,
    ):
        path = frozen_root / name
        expected = expected_outputs.get(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen #192 artifact hash mismatch: {name}")
    expected_sources = manifest.get("source_sha256", {})
    for file_name, record in game_source_files.items():
        key = f"raw_games/{file_name}"
        expected = expected_sources.get(key)
        if expected is None or record["sha256"] != expected:
            raise ValueError(f"frozen #192 source schedule hash mismatch: {file_name}")

    panel_rows = _read_gzip_jsonl(frozen_root / FROZEN_PANEL_NAME)
    identity_before = frozen_panel_identity(panel_rows)
    if any(
        "target" in row or "target_pmf" in row or "rank_observations" in row
        for row in panel_rows
    ):
        raise ValueError("frozen inference panel crossed its target-free boundary")
    target_rows = _read_gzip_jsonl(frozen_root / FROZEN_TARGET_NAME)
    targets: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    for row in target_rows:
        season = int(row["season"])
        team_id = str(row["team_id"])
        pmf = np.asarray(row["target_pmf"], dtype=np.float64)
        count = int(row["rank_count"])
        if (
            season not in SEASONS
            or team_id in targets[season]
            or pmf.ndim != 1
            or len(pmf) != count
            or not np.isfinite(pmf).all()
            or np.any(pmf < 0)
            or not np.isclose(float(pmf.sum()), 1.0, atol=1e-8, rtol=0.0)
        ):
            raise ValueError(f"{season}/{team_id}: invalid frozen final target PMF")
        targets[season][team_id] = pmf / pmf.sum()
    if set(targets) != set(SEASONS):
        raise ValueError("frozen #192 final targets do not cover 2022–2025")

    evidence_doc = json.loads((frozen_root / FROZEN_EVIDENCE_NAME).read_text())
    checkpoint_rows = evidence_doc.get("checkpoints")
    if not isinstance(checkpoint_rows, list):
        raise TypeError("frozen #192 evidence file has no checkpoint list")
    evidence_by_key: dict[tuple[int, int], dict[str, object]] = {}
    for row in checkpoint_rows:
        key = (int(row["season"]), int(row["checkpoint"]))
        ids = row.get("included_game_ids")
        if not isinstance(ids, list) or len(ids) != int(row["included_game_count"]):
            raise ValueError(f"{key}: frozen evidence ID list/count mismatch")
        if sha256_json(ids) != row.get("included_game_ids_sha256"):
            raise ValueError(f"{key}: frozen evidence ID hash mismatch")
        if key in evidence_by_key:
            raise ValueError(f"{key}: duplicate frozen checkpoint evidence")
        evidence_by_key[key] = row

    score_rows: dict[tuple[int, int, str, str], dict[str, str]] = {}
    with (frozen_root / FROZEN_SCORE_NAME).open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        for row in csv.DictReader(handle):
            if row["score_type"] != "baseline" or row["model"] not in {
                "history_posterior",
                "context_posterior",
            }:
                continue
            key = (
                int(row["season"]),
                int(row["checkpoint"]),
                row["team_id"],
                row["model"],
            )
            score_rows[key] = row
    raw_by_id = {
        (season, str(row["game_id"])): row
        for season, rows in outcome_rows_by_season.items()
        for row in rows
    }
    for key, evidence in evidence_by_key.items():
        season, _ = key
        for game_id in evidence["included_game_ids"]:
            if (season, str(game_id)) not in raw_by_id:
                raise ValueError(f"{key}: frozen evidence game {game_id} is missing")
    return (
        panel_rows,
        dict(targets),
        evidence_by_key,
        score_rows,
        identity_before,
        {
            "manifest_sha256": sha256_file(manifest_path),
            "panel_sha256": expected_outputs[FROZEN_PANEL_NAME],
            "final_target_sha256": expected_outputs[FROZEN_TARGET_NAME],
            "evidence_sha256": expected_outputs[FROZEN_EVIDENCE_NAME],
            "score_sha256": expected_outputs[FROZEN_SCORE_NAME],
            "evidence_doc": evidence_doc,
            "raw_game_rows_by_id": raw_by_id,
        },
    )


def score_frozen_target_sensitivity(
    *,
    frozen_root: Path,
    panel_rows: list[dict[str, object]],
    original_targets: dict[int, dict[str, np.ndarray]],
    evidence_by_key: dict[tuple[int, int], dict[str, object]],
    stored_scores: dict[tuple[int, int, str, str], dict[str, str]],
    identity_before: str,
    pre_bowl_targets: dict[int, dict[str, dict[str, object]]],
    target_dates: dict[int, date],
    schedule_windows: dict[int, dict[str, str]],
    raw_game_rows_by_id: dict[tuple[int, str], dict[str, object]],
) -> dict[str, object]:
    """Rescore the frozen #192 panel on one common, target-safe cohort."""
    del frozen_root  # The verified rows and hashes are passed explicitly.
    panel_by_key: dict[tuple[int, int, str], dict[str, object]] = {}
    for row in panel_rows:
        if row.get("subdivision") != "fbs":
            continue
        key = (int(row["season"]), int(row["checkpoint"]), str(row["team_id"]))
        if key in panel_by_key:
            raise ValueError(f"duplicate frozen panel row {key}")
        panel_by_key[key] = row
    by_checkpoint: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for key, row in panel_by_key.items():
        by_checkpoint[key[:2]].append(row)
    score_rows: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    parity_errors: list[float] = []
    checkpoint_safety: dict[tuple[int, int], dict[str, object]] = {}

    for checkpoint_key, rows in sorted(by_checkpoint.items()):
        season, checkpoint = checkpoint_key
        evidence = evidence_by_key.get(checkpoint_key)
        if evidence is None:
            raise ValueError(f"{checkpoint_key}: retained panel lacks evidence record")
        target_start = datetime.combine(target_dates[season], time.min, tzinfo=UTC)
        cutoff = datetime.fromisoformat(str(evidence["cutoff"]))
        ids = [str(game_id) for game_id in evidence["included_game_ids"]]
        evidence_rows = [raw_game_rows_by_id[(season, game_id)] for game_id in ids]
        latest_kickoff = max(
            (str(row["kickoff_utc"]) for row in evidence_rows), default=None
        )
        post_target_ids = [
            str(row["game_id"])
            for row in evidence_rows
            if datetime.fromisoformat(str(row["kickoff_utc"])) >= target_start
        ]
        target_safe = not post_target_ids and cutoff < target_start
        checkpoint_safety[checkpoint_key] = {
            "season": season,
            "checkpoint": checkpoint,
            "cutoff_utc": cutoff.isoformat(),
            "target_observation_date": target_dates[season].isoformat(),
            "latest_included_kickoff_utc": latest_kickoff,
            "included_game_count": len(ids),
            "post_target_evidence_count": len(post_target_ids),
            "post_target_evidence_ids_sha256": sha256_json(
                sorted(post_target_ids, key=_game_id_key)
            ),
            "target_safe": target_safe,
        }
        if not target_safe:
            excluded.append(
                {
                    "season": season,
                    "checkpoint": checkpoint,
                    "team_id": "",
                    "reason": (
                        "checkpoint_evidence_contains_post_target_game"
                        if post_target_ids
                        else "cutoff_not_strictly_before_target_observation_date"
                    ),
                    "cutoff_utc": cutoff.isoformat(),
                    "target_observation_date": target_dates[season].isoformat(),
                    "latest_included_kickoff_utc": latest_kickoff or "",
                    "excluded_team_count": len(rows),
                }
            )
            continue

        for row in rows:
            team_id = str(row["team_id"])
            population = int(row["rank_count"])
            final_target = original_targets.get(season, {}).get(team_id)
            pre_target_row = pre_bowl_targets.get(season, {}).get(team_id)
            pre_target = None if pre_target_row is None else pre_target_row["pmf"]
            if final_target is None or pre_target is None:
                excluded.append(
                    {
                        "season": season,
                        "checkpoint": checkpoint,
                        "team_id": team_id,
                        "reason": "missing_original_or_pre_bowl_target",
                        "cutoff_utc": cutoff.isoformat(),
                        "target_observation_date": target_dates[season].isoformat(),
                        "latest_included_kickoff_utc": latest_kickoff or "",
                        "excluded_team_count": 1,
                    }
                )
                continue
            supports = {
                population,
                len(final_target),
                len(pre_target),
                int(pre_target_row["population"]),
            }
            if len(supports) != 1:
                excluded.append(
                    {
                        "season": season,
                        "checkpoint": checkpoint,
                        "team_id": team_id,
                        "reason": "frozen_pmf_or_target_rank_support_mismatch",
                        "observed_supports": ",".join(map(str, sorted(supports))),
                        "cutoff_utc": cutoff.isoformat(),
                        "target_observation_date": target_dates[season].isoformat(),
                        "latest_included_kickoff_utc": latest_kickoff or "",
                        "excluded_team_count": 1,
                    }
                )
                continue
            forecasts = {
                "history_posterior": np.asarray(
                    row["history_posterior_pmf"], dtype=np.float64
                ),
                "context_posterior": np.asarray(
                    row["context_posterior_pmf"], dtype=np.float64
                ),
            }
            targets_for_policy = {
                "original_final_192": final_target,
                "pre_bowl_204": pre_target,
            }
            for model, forecast in forecasts.items():
                for target_policy, target in targets_for_policy.items():
                    metrics = score_rank_pmf(forecast, target)
                    score_rows.append(
                        {
                            "season": season,
                            "checkpoint": checkpoint,
                            "checkpoint_phase": checkpoint_phase(
                                cutoff,
                                season,
                                date.fromisoformat(
                                    schedule_windows[season][
                                        "last_fbs_conference_championship_date"
                                    ]
                                ),
                                date.fromisoformat(
                                    schedule_windows[season][
                                        "first_fbs_postseason_game_date"
                                    ]
                                ),
                            ),
                            "team_id": team_id,
                            "team_name": row["team_name"],
                            "component_status": row["component_status"],
                            "games_played": row["games_played"],
                            "model": model,
                            "target_policy": target_policy,
                            "rank_count": population,
                            "cutoff_utc": cutoff.isoformat(),
                            "target_observation_date": target_dates[season].isoformat(),
                            "latest_included_kickoff_utc": latest_kickoff or "",
                            "evidence_ids_sha256": evidence["included_game_ids_sha256"],
                            **metrics,
                        }
                    )
                    if target_policy == "original_final_192":
                        stored = stored_scores.get((season, checkpoint, team_id, model))
                        if stored is None:
                            raise ValueError(
                                f"missing retained original score for "
                                f"{season}/{checkpoint}/{team_id}/{model}"
                            )
                        for metric in RANK_SCORE_FIELDS:
                            delta = abs(metrics[metric] - float(stored[metric]))
                            parity_errors.append(delta)
                            if delta > 1e-8:
                                raise ValueError(
                                    f"original score parity failed for "
                                    f"{season}/{checkpoint}/{team_id}/{model}/"
                                    f"{metric}: {delta}"
                                )
    identity_after = frozen_panel_identity(panel_rows)
    if identity_after != identity_before:
        raise ValueError("frozen #192 PMFs changed while computing target scores")

    grouped: dict[tuple[int, int, str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in score_rows:
        grouped[
            (
                int(row["season"]),
                int(row["checkpoint"]),
                str(row["target_policy"]),
                str(row["model"]),
            )
        ].append(row)
    summary: list[dict[str, object]] = []
    for (season, checkpoint, target_policy, model), members in sorted(grouped.items()):
        summary.append(
            {
                "season": season,
                "checkpoint": checkpoint,
                "target_policy": target_policy,
                "model": model,
                "team_seasons": len(members),
                **{
                    f"mean_{metric}": float(
                        np.mean([float(row[metric]) for row in members])
                    )
                    for metric in RANK_SCORE_FIELDS
                },
            }
        )
    paired_summary: list[dict[str, object]] = []
    summary_by_key = {
        (row["season"], row["checkpoint"], row["target_policy"], row["model"]): row
        for row in summary
    }
    pair_keys = sorted({key[:3] for key in summary_by_key})
    for season, checkpoint, target_policy in pair_keys:
        context = summary_by_key.get(
            (season, checkpoint, target_policy, "context_posterior")
        )
        history = summary_by_key.get(
            (season, checkpoint, target_policy, "history_posterior")
        )
        if context is None or history is None:
            continue
        paired_summary.append(
            {
                "season": season,
                "checkpoint": checkpoint,
                "target_policy": target_policy,
                "team_seasons": min(context["team_seasons"], history["team_seasons"]),
                "context_minus_history_mean_nll": float(
                    context["mean_nll"] - history["mean_nll"]
                ),
                "context_minus_history_mean_crps": float(
                    context["mean_crps"] - history["mean_crps"]
                ),
                "context_minus_history_mean_expected_rank_error": float(
                    context["mean_expected_rank_error"]
                    - history["mean_expected_rank_error"]
                ),
                "context_minus_history_mean_interval_80_target_mass": float(
                    context["mean_interval_80_target_mass"]
                    - history["mean_interval_80_target_mass"]
                ),
            }
        )
    return {
        "scores": score_rows,
        "summary": summary,
        "paired_summary": paired_summary,
        "exclusions": excluded,
        "checkpoint_safety": [
            checkpoint_safety[key] for key in sorted(checkpoint_safety)
        ],
        "parity": {
            "passed": True,
            "score_rows_compared": len(parity_errors),
            "max_absolute_metric_difference": max(parity_errors, default=0.0),
            "tolerance": 1e-8,
            "frozen_pmf_identity_before": identity_before,
            "frozen_pmf_identity_after": identity_after,
            "frozen_pmf_identity_unchanged": identity_before == identity_after,
        },
    }


def _load_arm_likelihood(bundle: dict[str, object], policy: str) -> LikelihoodV1:
    likelihoods = bundle.get("likelihoods")
    if not isinstance(likelihoods, dict) or not isinstance(
        likelihoods.get(policy), dict
    ):
        raise TypeError(f"issue 200 bundle is missing the {policy} likelihood")
    row = likelihoods[policy]
    return LikelihoodV1(
        beta=np.asarray(row["beta"], dtype=np.float64),
        scale=float(row["scale"]),
        degrees_of_freedom=float(row["degrees_of_freedom"]),
        fit_kind=str(row.get("fit_kind", "weighted_pseudo")),
    )


def _scoreable_game_rows(
    season_plan: dict[str, object], outcome_rows: list[dict[str, object]]
) -> tuple[
    dict[str, tuple[dict[str, object], dict[str, object]]],
    list[dict[str, object]],
]:
    outcomes = {str(row["game_id"]): row for row in outcome_rows}
    if len(outcomes) != len(outcome_rows):
        raise ValueError(f"{season_plan['season']}: duplicate outcome game IDs")
    result: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    exclusions: list[dict[str, object]] = []
    for metadata in season_plan["games"]:
        game_id = str(metadata["game_id"])
        outcome = outcomes.get(game_id)
        if outcome is None:
            exclusions.append(
                {
                    **metadata,
                    "exclusion_reason": "scheduled_game_missing_from_outcome_source",
                }
            )
            continue
        if outcome.get("schedule_row_sha256") != metadata.get("schedule_row_sha256"):
            raise ValueError(f"{game_id}: scheduled metadata changed after plan freeze")
        if (
            not outcome.get("completed")
            or outcome.get("home_points") is None
            or outcome.get("away_points") is None
        ):
            exclusions.append(
                {**metadata, "exclusion_reason": "no_completed_score_in_source"}
            )
            continue
        result[game_id] = (metadata, outcome)
    return result, exclusions


def _posterior_diagnostics(result: Any) -> dict[str, object]:
    if not result.converged:
        raise RuntimeError(
            f"posterior replay did not converge after {result.iterations} iterations"
        )
    if any(
        pmf.ndim != 1
        or not np.isfinite(pmf).all()
        or np.any(pmf < 0)
        or not np.isclose(float(pmf.sum()), 1.0, atol=1e-8, rtol=0.0)
        for pmf in result.pmfs.values()
    ):
        raise ValueError("posterior replay returned a non-normalized PMF")
    return {
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "max_message_delta": float(result.max_message_delta),
        "objective": float(result.objective),
        "raw_game_factor_count": int(result.raw_game_factor_count),
        "unique_pair_factor_count": int(result.unique_pair_factor_count),
        "max_team_degree": int(result.max_team_degree),
        "warnings": list(result.warnings),
    }


def summarize_forecast_scores(
    rows: list[dict[str, object]], *, stage: str, population: str
) -> list[dict[str, object]]:
    metric_names = (
        "crps",
        "margin_nll",
        "median_absolute_error",
        "home_win_brier",
        "coverage_50",
        "coverage_80",
        "coverage_95",
        "margin_interval_50_width",
        "margin_interval_80_width",
        "margin_interval_95_width",
    )
    interval_endpoints = {
        "margin_interval_50_width": (
            "margin_interval_50_low",
            "margin_interval_50_high",
        ),
        "margin_interval_80_width": (
            "margin_interval_80_low",
            "margin_interval_80_high",
        ),
        "margin_interval_95_width": (
            "margin_interval_95_low",
            "margin_interval_95_high",
        ),
    }

    def metric_value(row: dict[str, object], metric: str) -> float:
        if metric in row:
            return float(row[metric])
        endpoints = interval_endpoints.get(metric)
        if endpoints is not None:
            low, high = endpoints
            return float(row[high]) - float(row[low])
        return float(row[metric])

    members_by_key: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        if row["forecast_stage"] != stage or row["forecast_population"] != population:
            continue
        members_by_key[
            (str(row["model_arm"]), str(row["season"]), str(row["phase"]))
        ].append(row)
    output: list[dict[str, object]] = []
    for (arm, season, phase), members in sorted(members_by_key.items()):
        values = {
            metric: float(np.mean([metric_value(row, metric) for row in members]))
            for metric in metric_names
        }
        output.append(
            {
                "forecast_stage": stage,
                "forecast_population": population,
                "model_arm": arm,
                "season": season,
                "phase": phase,
                "games": len(members),
                **values,
            }
        )
    by_arm_season: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in output:
        by_arm_season[(str(row["model_arm"]), str(row["season"]))].append(row)
    for (arm, season), members in sorted(by_arm_season.items()):
        count = sum(int(row["games"]) for row in members)
        output.append(
            {
                "forecast_stage": stage,
                "forecast_population": population,
                "model_arm": arm,
                "season": season,
                "phase": "all_phases",
                "games": count,
                **{
                    metric: float(
                        sum(
                            metric_value(row, metric) * int(row["games"])
                            for row in members
                        )
                        / count
                    )
                    for metric in metric_names
                },
            }
        )
    all_seasons = [row for row in output if row["phase"] == "all_phases"]
    if all_seasons:
        total_games = sum(int(row["games"]) for row in all_seasons)
        pooled_by_arm: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in all_seasons:
            pooled_by_arm[str(row["model_arm"])].append(row)
        for arm, members in sorted(pooled_by_arm.items()):
            games = sum(int(row["games"]) for row in members)
            output.append(
                {
                    "forecast_stage": stage,
                    "forecast_population": population,
                    "model_arm": arm,
                    "season": "pooled_equal_game",
                    "phase": "all_phases",
                    "games": games,
                    **{
                        metric: float(
                            sum(
                                metric_value(row, metric) * int(row["games"])
                                for row in members
                            )
                            / games
                        )
                        for metric in metric_names
                    },
                }
            )
            output.append(
                {
                    "forecast_stage": stage,
                    "forecast_population": population,
                    "model_arm": arm,
                    "season": "equal_season_sensitivity",
                    "phase": "all_phases",
                    "games": games,
                    **{
                        metric: float(
                            np.mean([metric_value(row, metric) for row in members])
                        )
                        for metric in metric_names
                    },
                }
            )
        if total_games <= 0:
            raise AssertionError("score summaries contain no games")
    return output


def summarize_rank_scores(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    metrics = RANK_SCORE_FIELDS
    grouped: dict[tuple[int, str, str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[
            (
                int(row["season"]),
                str(row["cutoff_utc"]),
                str(row["checkpoint_phase"]),
                str(row["model_arm"]),
            )
        ].append(row)
    result: list[dict[str, object]] = []
    for (season, cutoff, phase, arm), members in sorted(grouped.items()):
        result.append(
            {
                "season": season,
                "cutoff_utc": cutoff,
                "checkpoint_phase": phase,
                "model_arm": arm,
                "team_seasons": len(members),
                **{
                    f"mean_{metric}": float(
                        np.mean([float(row[metric]) for row in members])
                    )
                    for metric in metrics
                },
            }
        )
    return result


def paired_game_differences(
    season_summaries: list[dict[str, object]], population: str
) -> list[dict[str, object]]:
    selected = [
        row
        for row in season_summaries
        if row["forecast_stage"] == "rolling"
        and row["forecast_population"] == population
        and row["phase"] == "all_phases"
    ]
    metrics = ("crps", "margin_nll", "median_absolute_error", "home_win_brier")
    arms = [row[0] for row in MODEL_ARMS]
    summary_by = {(str(row["season"]), str(row["model_arm"])): row for row in selected}
    output: list[dict[str, object]] = []

    def append_pair_summary(label: str, left: str, right: str) -> None:
        a = summary_by.get((label, left))
        b = summary_by.get((label, right))
        if a is None or b is None:
            return
        if int(a["games"]) != int(b["games"]):
            raise ValueError(f"{label}: arm game populations are not common")
        output.append(
            {
                "aggregation": label,
                "season": label,
                "arm_a": left,
                "arm_b": right,
                "games": a["games"],
                **{
                    f"{metric}_a_minus_b": float(a[metric]) - float(b[metric])
                    for metric in metrics
                },
            }
        )

    for season in map(str, SEASONS):
        for left_index, left in enumerate(arms):
            for right in arms[left_index + 1 :]:
                append_pair_summary(season, left, right)

    for aggregate in ("pooled_equal_game", "equal_season_sensitivity"):
        for left_index, left in enumerate(arms):
            for right in arms[left_index + 1 :]:
                append_pair_summary(aggregate, left, right)
    return output


def _validate_unique_forecasts(rows: list[dict[str, object]]) -> dict[str, object]:
    ids = [str(row["forecast_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate forecast IDs were generated")
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (
            str(row["forecast_stage"]),
            str(row["forecast_population"]),
            str(row["model_arm"]),
            str(row["game_id"]),
        )
        if key in seen:
            raise ValueError(f"duplicate game forecast across anchors: {key}")
        seen.add(key)
    populations: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in rows:
        if row["forecast_population"] not in {PRIMARY_POPULATION, SECONDARY_POPULATION}:
            continue
        populations[(str(row["forecast_stage"]), str(row["forecast_population"]))][
            str(row["model_arm"])
        ].add(str(row["game_id"]))
    for key, arms in populations.items():
        if set(arms) != {arm for arm, _, _ in MODEL_ARMS}:
            raise ValueError(f"{key}: not every four-arm forecast is present")
        sets = list(arms.values())
        if any(game_ids != sets[0] for game_ids in sets[1:]):
            raise ValueError(f"{key}: model arms scored different game populations")
    return {
        "forecast_row_count": len(rows),
        "forecast_id_count": len(set(ids)),
        "unique_forecast_ids": True,
        "common_game_population_by_stage": {
            f"{stage}/{population}": {
                "unique_games": len(next(iter(arms.values()))),
                "arm_counts": {
                    arm: len(game_ids) for arm, game_ids in sorted(arms.items())
                },
            }
            for (stage, population), arms in sorted(populations.items())
        },
    }


def _target_window_rows(path: Path) -> dict[int, dict[str, str]]:
    rows = _read_csv(path)
    result = {int(row["season"]): row for row in rows if int(row["season"]) in SEASONS}
    if set(result) != set(SEASONS):
        raise ValueError("issue 198 schedule windows do not cover 2022–2025")
    return result


def _input_record(path: Path, *, label: str) -> dict[str, object]:
    return {"label": label, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _load_issue200_inputs(
    *, bundle_path: Path, manifest_path: Path, fitted_models_path: Path
) -> tuple[
    dict[str, object], dict[str, dict[str, dict[str, object]]], dict[str, object]
]:
    bundle = read_pipeline_bundle(bundle_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs", {})
    expected = outputs.get(bundle_path.name) if isinstance(outputs, dict) else None
    if expected is not None and (
        not isinstance(expected, dict)
        or expected.get("sha256") != sha256_file(bundle_path)
    ):
        raise ValueError("issue 200 pipeline bundle hash differs from its manifest")
    if isinstance(outputs, dict) and bundle_path.name not in outputs:
        raise ValueError("issue 200 manifest does not list its pipeline bundle output")
    if manifest.get("fitting_status") != "complete":
        raise ValueError("issue 200 model pipeline is not complete")
    fit_origin = validate_forward_only_fit(manifest_path, fitted_models_path)
    arm_priors = {str(season): _prior_rows_by_arm(bundle, season) for season in SEASONS}
    return (
        bundle,
        arm_priors,
        {
            "bundle_sha256": sha256_file(bundle_path),
            "manifest_sha256": sha256_file(manifest_path),
            "fitted_models_sha256": sha256_file(fitted_models_path),
            "bundle_manifest_expected_sha256": (
                expected.get("sha256") if isinstance(expected, dict) else None
            ),
            "fit_origin_by_season_policy": fit_origin,
            "fitted_prior_arm_count": len(manifest.get("fitted_prior_arms", [])),
            "training_seasons": manifest["training_seasons"],
            "target_seasons": manifest["target_seasons"],
        },
    )


def run_issue_201_comparison(
    *,
    repository_root: Path,
    game_root: Path,
    schedule_plan: dict[str, object],
    schedule_plan_path: Path,
    schedule_plan_manifest_path: Path,
    issue200_bundle_path: Path,
    issue200_manifest_path: Path,
    issue200_fitted_models_path: Path,
    pre_bowl_target_path: Path,
    issue204_manifest_path: Path,
    target_snapshot_manifest_path: Path,
    schedule_windows_path: Path,
    frozen_root: Path,
    protocol_path: Path,
) -> dict[str, object]:
    """Run the frozen issue 201 replay, game scoring, and rank sensitivities."""
    protocol_sha = sha256_file(protocol_path)
    outcome_rows_by_season, game_source_files = verify_score_blind_plan(
        schedule_plan,
        schedule_plan_path,
        schedule_plan_manifest_path,
        game_root,
        protocol_sha256=protocol_sha,
        target_snapshot_manifest_path=target_snapshot_manifest_path,
        schedule_windows_path=schedule_windows_path,
    )
    bundle, priors_by_season, issue200_provenance = _load_issue200_inputs(
        bundle_path=issue200_bundle_path,
        manifest_path=issue200_manifest_path,
        fitted_models_path=issue200_fitted_models_path,
    )
    pre_bowl_targets, pre_bowl_provenance = load_pre_bowl_targets(
        pre_bowl_target_path, issue204_manifest_path
    )
    issue204_manifest = json.loads(issue204_manifest_path.read_text(encoding="utf-8"))
    issue200_manifest = json.loads(issue200_manifest_path.read_text(encoding="utf-8"))
    issue200_inputs = issue200_manifest.get("input_artifacts", [])
    for logical_name, output_name in (
        ("corrected_final_targets", "final_target_distributions.csv"),
        ("corrected_pre_bowl_targets", "pre_bowl_target_distributions.csv"),
    ):
        expected_sha = issue204_manifest["outputs"][output_name]["sha256"]
        matches = [
            row for row in issue200_inputs if row.get("logical_name") == logical_name
        ]
        if len(matches) != 1 or matches[0].get("sha256") != expected_sha:
            raise ValueError(f"issue 200 did not use #204's {logical_name} artifact")
    target_dates = _read_target_snapshot_dates(target_snapshot_manifest_path)
    windows = _target_window_rows(schedule_windows_path)
    for season, row in windows.items():
        issue200_date = target_dates[season]
        expected_date = row.get("snapshot_date")
        if expected_date and date.fromisoformat(expected_date) != issue200_date:
            raise ValueError(f"{season}: issue 200 and #198 target dates disagree")

    forecast_rows: list[dict[str, object]] = []
    forecast_exclusions: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    ranking_exclusions: list[dict[str, object]] = []
    replay_evidence_rows: list[dict[str, object]] = []
    rank_population_audit: dict[str, object] = {}
    schedule_seasons = {int(row["season"]): row for row in schedule_plan["seasons"]}
    calendar_mismatches = [
        {
            "season": row["season"],
            "game_id": row["game_id"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "kickoff_utc": row["kickoff_utc"],
            "target_snapshot_date": target_dates[int(row["season"])].isoformat(),
            "phase": row["phase"],
            "reason": "army_navy_kickoff_after_pre_bowl_target_snapshot",
        }
        for season_plan in schedule_plan["seasons"]
        for row in season_plan["games"]
        if row["phase"] == "army_navy" and row["after_pre_bowl_target_date"]
    ]

    for season in SEASONS:
        _cdf_quadrature_surface.cache_clear()
        print(f"Issue 201: replaying season {season}", flush=True)
        season_plan = schedule_seasons[season]
        season_outcomes = outcome_rows_by_season[season]
        scoreable_games, outcome_exclusions = _scoreable_game_rows(
            season_plan, season_outcomes
        )
        forecast_exclusions.extend(outcome_exclusions)
        for row in season_plan["excluded_schedule_rows"]:
            forecast_exclusions.append(
                {**row, "exclusion_reason": row["plan_exclusion_reason"]}
            )

        common_teams, population_exclusions = _common_target_teams(
            season, priors_by_season[str(season)], pre_bowl_targets
        )
        rank_population_audit[str(season)] = {
            "common_team_count": len(common_teams),
            "excluded_team_seasons": len(population_exclusions),
            "fbs_prior_team_count": len(
                next(iter(priors_by_season[str(season)].values()))
            ),
            "pre_bowl_target_team_count": len(pre_bowl_targets[season]),
        }
        ranking_exclusions.extend(population_exclusions)
        if not common_teams:
            raise ValueError(f"{season}: no common teams for four-arm ranking scores")

        fcs_names = _fcs_names(season_outcomes)
        games_by_cutoff: dict[str, list[str]] = defaultdict(list)
        plan_game_by_id = {str(row["game_id"]): row for row in season_plan["games"]}
        for game_id, (metadata, _outcome) in scoreable_games.items():
            games_by_cutoff[str(metadata["assigned_cutoff"])].append(game_id)
        for game_ids in games_by_cutoff.values():
            game_ids.sort(
                key=lambda game_id: (
                    str(plan_game_by_id[game_id]["kickoff_utc"]),
                    _game_id_key(game_id),
                )
            )

        preseason_posteriors: dict[str, tuple[dict[str, Team], dict[str, object]]] = {}
        empty_hashes = {
            "included_game_ids": [],
            "included_game_ids_sha256": sha256_json([]),
            "included_game_rows_sha256": sha256_json([]),
            "included_game_count": 0,
            "excluded_game_ids": [],
            "excluded_game_ids_sha256": sha256_json([]),
            "excluded_game_count": 0,
        }
        for arm, family, policy in MODEL_ARMS:
            replay = replay_pipeline_bundle(
                bundle,
                season=season,
                policy=policy,  # type: ignore[arg-type]
                prior_family=family,  # type: ignore[arg-type]
                games=[],
                fcs_team_names=fcs_names,
            )
            diagnostics = _posterior_diagnostics(replay)
            arm_prior_rows = priors_by_season[str(season)][arm]
            teams = _build_posterior_teams(replay.pmfs, arm_prior_rows, fcs_names)
            preseason_posteriors[arm] = (teams, diagnostics)
        empty_evidence_by_policy = {
            policy: {
                "season": season,
                "cutoff_utc": "preseason_no_evidence",
                "policy": policy,
                "safety_lag_hours": 48,
                **empty_hashes,
                "latest_included_kickoff_utc": None,
                "replay_diagnostics": {
                    family: preseason_posteriors[arm][1]
                    for arm, family, arm_policy in MODEL_ARMS
                    if arm_policy == policy
                },
                "posterior_team_count": len(
                    preseason_posteriors[
                        next(
                            arm
                            for arm, _, arm_policy in MODEL_ARMS
                            if arm_policy == policy
                        )
                    ][0]
                ),
                "fcs_fallback_team_count": len(fcs_names),
            }
            for policy in ("existing", "pre_bowl")
        }
        replay_evidence_rows.extend(empty_evidence_by_policy.values())
        likelihoods = {
            policy: _load_arm_likelihood(bundle, policy)
            for policy in ("existing", "pre_bowl")
        }
        preseason_jobs: list[dict[str, object]] = []
        for arm, family, policy in MODEL_ARMS:
            teams, _diagnostics = preseason_posteriors[arm]
            evidence = empty_evidence_by_policy[policy]
            for metadata, outcome in scoreable_games.values():
                preseason_jobs.append(
                    {
                        "metadata": metadata,
                        "outcome": outcome,
                        "arm": arm,
                        "policy": policy,
                        "family": family,
                        "posterior_teams": teams,
                        "likelihood": likelihoods[policy],
                        "evidence": evidence,
                        "stage": "preseason",
                        "cutoff": "preseason_no_evidence",
                    }
                )
        forecast_rows.extend(_parallel_prediction_rows(preseason_jobs))

        cutoff_phase = {
            str(row["cutoff_utc"]): str(row["phase"])
            for row in season_plan["cutoff_phases"]
        }
        target_start = datetime.combine(target_dates[season], time.min, tzinfo=UTC)
        for cutoff_index, cutoff_text in enumerate(season_plan["cutoffs_utc"], start=1):
            if cutoff_index == 1 or cutoff_index % 4 == 0:
                print(
                    f"Issue 201: {season} cutoff {cutoff_index}/"
                    f"{len(season_plan['cutoffs_utc'])}",
                    flush=True,
                )
            cutoff = datetime.fromisoformat(str(cutoff_text))
            future_ids = set(games_by_cutoff.get(str(cutoff_text), []))
            policy_results: dict[str, dict[str, Any]] = {}
            for policy in ("existing", "pre_bowl"):
                included, excluded = filter_evidence_rows(
                    season_outcomes,
                    cutoff,
                    policy,  # type: ignore[arg-type]
                )
                included_ids = {str(row["game_id"]) for row in included}
                overlap = included_ids & future_ids
                if overlap:
                    raise ValueError(
                        f"{season}/{cutoff_text}: future forecast games entered evidence "
                        f"{sorted(overlap, key=_game_id_key)[:5]}"
                    )
                evidence_games = [game_from_outcome(row) for row in included]
                diagnostics_by_family: dict[str, dict[str, object]] = {}
                arm_states: dict[str, dict[str, object]] = {}
                for arm, family, arm_policy in MODEL_ARMS:
                    if arm_policy != policy:
                        continue
                    replay = replay_pipeline_bundle(
                        bundle,
                        season=season,
                        policy=policy,  # type: ignore[arg-type]
                        prior_family=family,  # type: ignore[arg-type]
                        games=evidence_games,
                        fcs_team_names=fcs_names,
                    )
                    diagnostics = _posterior_diagnostics(replay)
                    diagnostics_by_family[family] = diagnostics
                    team_posteriors = _build_posterior_teams(
                        replay.pmfs,
                        priors_by_season[str(season)][arm],
                        fcs_names,
                    )
                    arm_states[arm] = {
                        "result": replay,
                        "teams": team_posteriors,
                        "diagnostics": diagnostics,
                    }
                evidence = _evidence_record(
                    season,
                    cutoff,
                    policy,  # type: ignore[arg-type]
                    included,
                    excluded,
                    diagnostics_by_family,
                    len(next(iter(arm_states.values()))["result"].pmfs),
                    len(fcs_names),
                )
                evidence["excluded_game_details"] = [
                    {
                        "game_id": row["game_id"],
                        "kickoff_utc": row.get("kickoff_utc"),
                        "reason": row["evidence_exclusion_reason"],
                        "game_row_sha256": row["game_row_sha256"],
                    }
                    for row in excluded
                ]
                evidence["included_game_row_hashes"] = [
                    {
                        "game_id": row["game_id"],
                        "kickoff_utc": row["kickoff_utc"],
                        "game_row_sha256": row["game_row_sha256"],
                    }
                    for row in sorted(
                        included, key=lambda item: _game_id_key(str(item["game_id"]))
                    )
                ]
                replay_evidence_rows.append(evidence)
                policy_results[policy] = {
                    "evidence": evidence,
                    "included": included,
                    "arms": arm_states,
                }

            if cutoff < target_start:
                for arm, family, policy in MODEL_ARMS:
                    selected = policy_results[policy]
                    included = selected["included"]
                    late_ids = [
                        str(row["game_id"])
                        for row in included
                        if datetime.fromisoformat(str(row["kickoff_utc"]))
                        >= target_start
                    ]
                    if late_ids:
                        raise ValueError(
                            f"{season}/{cutoff_text}/{policy}: ranking inference used "
                            "evidence after the pre-bowl target"
                        )
                    posterior = selected["arms"][arm]["result"].pmfs
                    evidence = selected["evidence"]
                    phase = cutoff_phase[str(cutoff_text)]
                    for team_id in common_teams:
                        forecast = np.asarray(posterior[team_id], dtype=np.float64)
                        target = np.asarray(
                            pre_bowl_targets[season][team_id]["pmf"], dtype=np.float64
                        )
                        if forecast.shape != target.shape:
                            raise ValueError(
                                f"{season}/{team_id}/{arm}: ranking support changed"
                            )
                        ranking_rows.append(
                            {
                                "season": season,
                                "cutoff_utc": cutoff_text,
                                "checkpoint_phase": phase,
                                "team_id": team_id,
                                "team_name": pre_bowl_targets[season][team_id][
                                    "team_name"
                                ],
                                "model_arm": arm,
                                "policy": policy,
                                "prior_family": family,
                                "target_policy": "pre_bowl_204",
                                "rank_count": len(target),
                                "target_observation_date": target_dates[
                                    season
                                ].isoformat(),
                                "latest_included_kickoff_utc": evidence[
                                    "latest_included_kickoff_utc"
                                ]
                                or "",
                                "evidence_game_count": evidence["included_game_count"],
                                "evidence_ids_sha256": evidence[
                                    "included_game_ids_sha256"
                                ],
                                **score_rank_pmf(forecast, target),
                            }
                        )
            else:
                ranking_exclusions.append(
                    {
                        "season": season,
                        "checkpoint": "weekly",
                        "cutoff_utc": cutoff_text,
                        "team_id": "",
                        "reason": "cutoff_not_strictly_before_target_observation_date",
                        "target_observation_date": target_dates[season].isoformat(),
                        "latest_included_kickoff_utc": max(
                            (
                                str(row["kickoff_utc"])
                                for state in policy_results.values()
                                for row in state["included"]
                            ),
                            default="",
                        ),
                    }
                )

            rolling_jobs: list[dict[str, object]] = []
            for game_id in sorted(future_ids, key=_game_id_key):
                metadata, outcome = scoreable_games[game_id]
                missing_arm = [
                    arm
                    for arm, _, policy in MODEL_ARMS
                    if str(metadata["home_id"])
                    not in policy_results[policy]["arms"][arm]["teams"]
                    or str(metadata["away_id"])
                    not in policy_results[policy]["arms"][arm]["teams"]
                ]
                if missing_arm:
                    forecast_exclusions.append(
                        {
                            **metadata,
                            "exclusion_reason": "team_missing_from_one_or_more_four_arm_posteriors",
                            "missing_model_arms": ",".join(missing_arm),
                        }
                    )
                    continue
                for arm, family, policy in MODEL_ARMS:
                    state = policy_results[policy]
                    rolling_jobs.append(
                        {
                            "metadata": metadata,
                            "outcome": outcome,
                            "arm": arm,
                            "policy": policy,
                            "family": family,
                            "posterior_teams": state["arms"][arm]["teams"],
                            "likelihood": likelihoods[policy],
                            "evidence": state["evidence"],
                            "stage": "rolling",
                            "cutoff": str(cutoff_text),
                        }
                    )
            forecast_rows.extend(_parallel_prediction_rows(rolling_jobs))

    replay_consistency = _validate_unique_forecasts(forecast_rows)
    rolling_fbs_summary = summarize_forecast_scores(
        forecast_rows, stage="rolling", population=PRIMARY_POPULATION
    )
    rolling_cross_summary = summarize_forecast_scores(
        forecast_rows, stage="rolling", population=SECONDARY_POPULATION
    )
    preseason_summary = summarize_forecast_scores(
        forecast_rows, stage="preseason", population=PRIMARY_POPULATION
    )
    paired_differences = paired_game_differences(
        rolling_fbs_summary, PRIMARY_POPULATION
    )
    ranking_summary = summarize_rank_scores(ranking_rows)

    # Frozen #192 panels and final target PMFs are independently hash-gated.
    (
        frozen_panel_rows,
        original_targets,
        frozen_evidence,
        frozen_stored_scores,
        frozen_identity_before,
        frozen_provenance,
    ) = _load_frozen_panel(frozen_root, outcome_rows_by_season, game_source_files)
    frozen_diagnostics = score_frozen_target_sensitivity(
        frozen_root=frozen_root,
        panel_rows=frozen_panel_rows,
        original_targets=original_targets,
        evidence_by_key=frozen_evidence,
        stored_scores=frozen_stored_scores,
        identity_before=frozen_identity_before,
        pre_bowl_targets=pre_bowl_targets,
        target_dates=target_dates,
        schedule_windows=windows,
        raw_game_rows_by_id=frozen_provenance["raw_game_rows_by_id"],
    )
    frozen_provenance = {
        key: value
        for key, value in frozen_provenance.items()
        if key not in {"raw_game_rows_by_id", "evidence_doc"}
    }

    issue204_inputs = json.loads(issue204_manifest_path.read_text(encoding="utf-8"))
    input_artifacts = {
        "issue200_pipeline_bundle": _input_record(
            issue200_bundle_path, label="matched_four_arm_pipeline_bundle"
        ),
        "issue200_manifest": _input_record(
            issue200_manifest_path, label="matched_pipeline_manifest"
        ),
        "issue200_fitted_models": _input_record(
            issue200_fitted_models_path, label="rolling_fit_origins"
        ),
        "issue204_pre_bowl_targets": pre_bowl_provenance,
        "issue204_manifest": _input_record(
            issue204_manifest_path, label="corrected_population_and_support_manifest"
        ),
        "issue200_target_snapshot_manifest": _input_record(
            target_snapshot_manifest_path, label="target_snapshot_dates"
        ),
        "issue198_schedule_windows": _input_record(
            schedule_windows_path, label="event_phase_dates"
        ),
        "frozen_192_outputs": frozen_provenance,
        "raw_game_files": game_source_files,
        "protocol": _input_record(protocol_path, label="frozen_issue201_protocol"),
        "schedule_plan": {
            "path_label": "issue_201_schedule_plan.json",
            "bytes": schedule_plan_path.stat().st_size,
            "sha256": sha256_file(schedule_plan_path),
            "manifest_sha256": sha256_file(schedule_plan_manifest_path),
        },
        "issue204_target_support_status": {
            "corpus_version": issue204_inputs.get("corpus_version"),
            "status": issue204_inputs.get("status"),
            "downstream_ready": issue204_inputs.get("downstream_ready"),
            "support_status": issue204_inputs.get("target_support_validation", {}).get(
                "status"
            ),
            "support_anomaly_count": issue204_inputs.get(
                "target_support_validation", {}
            ).get("anomaly_count"),
        },
    }
    return {
        "forecast_scores": forecast_rows,
        "forecast_exclusions": forecast_exclusions,
        "replay_evidence": replay_evidence_rows,
        "ranking_scores": ranking_rows,
        "ranking_summary": ranking_summary,
        "ranking_exclusions": ranking_exclusions,
        "frozen_scores": frozen_diagnostics["scores"],
        "frozen_summary": frozen_diagnostics["summary"],
        "frozen_paired_summary": frozen_diagnostics["paired_summary"],
        "frozen_exclusions": frozen_diagnostics["exclusions"],
        "frozen_checkpoint_safety": frozen_diagnostics["checkpoint_safety"],
        "frozen_parity": frozen_diagnostics["parity"],
        "forecast_summaries": rolling_fbs_summary + rolling_cross_summary,
        "preseason_summary": preseason_summary,
        "calendar_mismatches": calendar_mismatches,
        "paired_game_differences": paired_differences,
        "replay_consistency": replay_consistency,
        "rank_population_audit": rank_population_audit,
        "issue200_provenance": issue200_provenance,
        "pre_bowl_target_provenance": pre_bowl_provenance,
        "input_artifacts": input_artifacts,
        "protocol_sha256": protocol_sha,
        "model_arms": [
            {
                "model_arm": arm,
                "prior_family": family,
                "policy": policy,
                "likelihood_sha256": sha256_json(bundle["likelihoods"][policy]),
                "prior_row_sha256_by_season": {
                    str(season): sha256_json(
                        [
                            {
                                "team_id": team_id,
                                "population": int(row["population"]),
                                "pmf": row["pmf"],
                            }
                            for team_id, row in sorted(
                                priors_by_season[str(season)][arm].items(),
                                key=lambda item: _team_id_key(item[0]),
                            )
                        ]
                    )
                    for season in SEASONS
                },
            }
            for arm, family, policy in MODEL_ARMS
        ],
        "fit_origin_by_season_policy": issue200_provenance[
            "fit_origin_by_season_policy"
        ],
        "target_snapshot_dates": {
            str(season): target_dates[season].isoformat() for season in SEASONS
        },
        "schedule_window_dates": {
            str(season): {
                field: windows[season][field]
                for field in (
                    "last_fbs_conference_championship_date",
                    "first_fbs_postseason_game_date",
                )
            }
            for season in SEASONS
        },
        "ranking_common_population": rank_population_audit,
    }
