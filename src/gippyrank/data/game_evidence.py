"""Validation and reconciliation for completed CFBD game evidence."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta

STATUS_FIELDS = ("status", "gameStatus")
_TRUE_VALUES = {"true", "1", "yes"}
_FALSE_VALUES = {"false", "0", "no", ""}
_REPLACEMENT_WINDOW = timedelta(days=7)
_AMBIGUOUS_COMPLETED_WINDOW = timedelta(days=30)


def completed_flag(row: Mapping[str, object]) -> bool:
    """Parse CFBD's completion flag without treating malformed values as false."""
    value = str(row.get("completed", "") or "").strip().casefold()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(
        f"CFBD game {row.get('id', '<unknown>')} has invalid completed flag {value!r}"
    )


def _status_category(row: Mapping[str, object]) -> str:
    categories = set()
    has_unrecognized_status = False
    for field in STATUS_FIELDS:
        value = str(row.get(field, "") or "").strip().casefold()
        if not value:
            continue
        normalized = re.sub(r"[^a-z0-9]+", " ", value).strip()
        if re.search(r"\b(cancel\w*|postpon\w*|ppd)\b", normalized):
            categories.add("cancelled")
        elif re.search(r"\b(final|complete|completed)\b", normalized):
            categories.add("completed")
        elif re.search(
            r"\b(scheduled|pregame|pre game|live|in ?progress|delayed|suspended)\b",
            normalized,
        ):
            categories.add("pending")
        else:
            has_unrecognized_status = True
    if len(categories) > 1:
        raise ValueError(
            f"CFBD game {row.get('id', '<unknown>')} has conflicting status fields"
        )
    if has_unrecognized_status:
        return "unknown"
    return next(iter(categories), "")


def _score(row: Mapping[str, object], side: str) -> int:
    game_id = row.get("id", "<unknown>")
    value = str(row.get(f"{side}Points", "") or "").strip()
    if not re.fullmatch(r"\d+", value):
        raise ValueError(
            f"completed CFBD game {game_id} needs a non-negative integer {side} score"
        )
    return int(value)


def _start_time(row: Mapping[str, object]) -> datetime | None:
    value = str(row.get("startDate", "") or "").strip()
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return (
        parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None
    )


def _matchup_key(row: Mapping[str, object]) -> tuple[str, str, tuple[str, str]] | None:
    home_id = str(row.get("homeId", "") or "").strip()
    away_id = str(row.get("awayId", "") or "").strip()
    if not home_id or not away_id or home_id == away_id:
        return None
    season = str(row.get("season", "") or "").strip()
    season_type = str(row.get("seasonType", "regular") or "regular").strip().casefold()
    return season, season_type, tuple(sorted((home_id, away_id)))


def _game_id_order(game_id: str) -> tuple[int, int | str]:
    return (0, int(game_id)) if game_id.isdigit() else (1, game_id)


def _clear_home_and_away_rematch(
    first: Mapping[str, object], second: Mapping[str, object]
) -> bool:
    false_values = {"false", "0", "no"}
    first_neutral = str(first.get("neutralSite", "") or "").strip().casefold()
    second_neutral = str(second.get("neutralSite", "") or "").strip().casefold()
    if first_neutral not in false_values or second_neutral not in false_values:
        return False
    return str(first.get("homeId", "")) == str(second.get("awayId", "")) and str(
        first.get("awayId", "")
    ) == str(second.get("homeId", ""))


def reconcile_game_evidence_rows(
    rows: Iterable[Mapping[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    """Return eligible completed rows and explicit exclusions for one cutoff.

    The caller must first remove records whose source was unavailable at the
    publication cutoff and completed records whose scheduled kickoff is later
    than that cutoff. Distinct IDs are never merged merely because they share
    teams: suspicious scores are excluded, while plausible completed records
    for the same matchup in the same CFBD week or within 30 days fail as an
    unresolved event unless the non-neutral home and away teams are reversed.
    A longer gap in a different week remains eligible as a possible rematch.
    """
    unique_rows: list[dict[str, str]] = []
    by_id: dict[str, dict[str, str]] = {}
    for source in rows:
        row = {str(key): str(value or "") for key, value in source.items()}
        game_id = row.get("id", "").strip()
        if not game_id:
            raise ValueError("CFBD game row has no game ID")
        previous = by_id.get(game_id)
        if previous is not None:
            if previous != row:
                raise ValueError(f"Conflicting CFBD game rows for game {game_id}")
            continue
        by_id[game_id] = row
        unique_rows.append(row)

    eligible: list[dict[str, str]] = []
    excluded: list[dict[str, object]] = []
    cancelled_rows: list[dict[str, str]] = []
    for row in unique_rows:
        game_id = row["id"]
        completed = completed_flag(row)
        status = _status_category(row)
        if completed and status == "unknown":
            raise ValueError(f"CFBD game {game_id} has an unrecognized nonempty status")
        scores_present = bool(
            str(row.get("homePoints", "") or "").strip()
            or str(row.get("awayPoints", "") or "").strip()
        )

        if status == "cancelled":
            cancelled_rows.append(row)
            if completed or scores_present:
                home_score = _score(row, "home") if row.get("homePoints", "") else None
                away_score = _score(row, "away") if row.get("awayPoints", "") else None
                if (home_score not in (None, 0)) or (away_score not in (None, 0)):
                    raise ValueError(
                        f"CFBD game {game_id} has a cancelled status and a non-zero result"
                    )
                excluded.append(
                    {
                        "game_id": game_id,
                        "reason": "cancelled_status_conflicts_with_completion_or_scores",
                        "replacement_game_ids": [],
                        "investigation_required": True,
                    }
                )
            continue

        if status == "pending" and completed:
            raise ValueError(
                f"CFBD game {game_id} is marked completed while its status is pending"
            )
        if status == "completed" and not completed:
            raise ValueError(
                f"CFBD game {game_id} has a completed status but completed is false"
            )
        if not completed:
            continue

        if _matchup_key(row) is None:
            raise ValueError(
                f"completed CFBD game {game_id} needs two distinct team IDs"
            )
        if _start_time(row) is None:
            raise ValueError(
                f"completed CFBD game {game_id} needs a timezone-aware startDate"
            )
        home_score = _score(row, "home")
        away_score = _score(row, "away")
        if home_score == away_score:
            reason = (
                "suspicious_completed_zero_zero_score"
                if home_score == 0
                else "suspicious_completed_tied_score"
            )
            excluded.append(
                {
                    "game_id": game_id,
                    "reason": reason,
                    "replacement_game_ids": [],
                    "investigation_required": True,
                }
            )
            continue
        eligible.append(row)

    # Same-opponent records close in the season schedule can be replacement IDs,
    # even when their scores are individually plausible. The 30-day window is
    # intentionally conservative; it does not silently choose or merge rows.
    matchups: defaultdict[tuple[str, str, tuple[str, str]], list[dict[str, str]]] = (
        defaultdict(list)
    )
    for row in eligible:
        key = _matchup_key(row)
        assert key is not None
        matchups[key].append(row)
    for matchup_rows in matchups.values():
        matchup_rows.sort(key=lambda row: _start_time(row))
        for index, first in enumerate(matchup_rows):
            first_time = _start_time(first)
            assert first_time is not None
            for second in matchup_rows[index + 1 :]:
                second_time = _start_time(second)
                assert second_time is not None
                same_cfbd_week = bool(first.get("week")) and (
                    first.get("week") == second.get("week")
                )
                clear_home_and_away_rematch = _clear_home_and_away_rematch(
                    first, second
                )
                if not same_cfbd_week and (
                    second_time - first_time > _AMBIGUOUS_COMPLETED_WINDOW
                    or clear_home_and_away_rematch
                ):
                    continue
                raise ValueError(
                    "ambiguous completed CFBD records for the same matchup in the "
                    f"same season schedule: {first['id']} and {second['id']}"
                )

    # Link an explicit cancellation or a rejected tied result to the nearby
    # valid replacement ID when the source makes that relationship unambiguous.
    review_by_id = {str(item["game_id"]): item for item in excluded}
    for row in cancelled_rows:
        game_id = row["id"]
        if game_id in review_by_id:
            continue
        category = "cancelled"
        source_time = _start_time(row)
        source_key = _matchup_key(row)
        if source_time is None or source_key is None:
            continue
        candidates = [
            item
            for item in eligible
            if _matchup_key(item) == source_key
            and (candidate_time := _start_time(item)) is not None
            and timedelta(0) <= candidate_time - source_time <= _REPLACEMENT_WINDOW
        ]
        if candidates:
            excluded.append(
                {
                    "game_id": game_id,
                    "reason": f"{category}_fixture_replaced_by_completed_record",
                    "replacement_game_ids": sorted(
                        (item["id"] for item in candidates), key=_game_id_order
                    ),
                    "investigation_required": False,
                }
            )

    for item in excluded:
        if item["replacement_game_ids"]:
            continue
        row = by_id[str(item["game_id"])]
        source_time = _start_time(row)
        source_key = _matchup_key(row)
        if source_time is None or source_key is None:
            continue
        candidates = [
            candidate
            for candidate in eligible
            if _matchup_key(candidate) == source_key
            and (candidate_time := _start_time(candidate)) is not None
            and timedelta(0) <= candidate_time - source_time <= _REPLACEMENT_WINDOW
        ]
        item["replacement_game_ids"] = sorted(
            (candidate["id"] for candidate in candidates), key=_game_id_order
        )

    excluded.sort(key=lambda item: _game_id_order(str(item["game_id"])))
    return eligible, excluded
