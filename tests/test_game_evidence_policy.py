from __future__ import annotations

import pytest

from gippyrank.data.game_evidence import reconcile_game_evidence_rows
from gippyrank.posterior.game_evidence import _schedule_state


def _game(
    game_id: str,
    *,
    start: str,
    completed: bool = True,
    status: str = "",
    home_score: int | None = 28,
    away_score: int | None = 19,
) -> dict[str, str]:
    return {
        "id": game_id,
        "season": "2026",
        "week": "1",
        "seasonType": "regular",
        "startDate": start,
        "completed": str(completed),
        "status": status,
        "neutralSite": "False",
        "homeId": "2097",
        "homeTeam": "Campbell",
        "homeClassification": "fcs",
        "homePoints": "" if home_score is None else str(home_score),
        "awayId": "2717",
        "awayTeam": "Western Carolina",
        "awayClassification": "fcs",
        "awayPoints": "" if away_score is None else str(away_score),
    }


def test_campbell_zero_zero_is_excluded_and_linked_to_replacement_id() -> None:
    phantom = _game(
        "401866625",
        start="2026-09-05T19:30:00+00:00",
        home_score=0,
        away_score=0,
    )
    result = _game("401917058", start="2026-09-06T15:00:00+00:00")

    eligible, exclusions = reconcile_game_evidence_rows([phantom, result])

    assert [row["id"] for row in eligible] == ["401917058"]
    assert exclusions == [
        {
            "game_id": "401866625",
            "reason": "suspicious_completed_zero_zero_score",
            "replacement_game_ids": ["401917058"],
            "investigation_required": True,
        }
    ]


def test_completed_zero_zero_without_a_replacement_is_reviewed_not_included() -> None:
    phantom = _game(
        "401866625",
        start="2026-09-05T19:30:00+00:00",
        home_score=0,
        away_score=0,
    )

    eligible, exclusions = reconcile_game_evidence_rows([phantom])

    assert eligible == []
    assert exclusions[0]["reason"] == "suspicious_completed_zero_zero_score"
    assert exclusions[0]["investigation_required"] is True
    assert exclusions[0]["replacement_game_ids"] == []


def test_postponed_fixture_is_reconciled_to_a_new_id() -> None:
    postponed = _game(
        "old-id",
        start="2026-09-05T19:30:00+00:00",
        completed=False,
        status="Postponed",
        home_score=None,
        away_score=None,
    )
    replacement = _game("new-id", start="2026-09-10T19:30:00+00:00")

    eligible, exclusions = reconcile_game_evidence_rows([postponed, replacement])

    assert [row["id"] for row in eligible] == ["new-id"]
    assert exclusions == [
        {
            "game_id": "old-id",
            "reason": "cancelled_fixture_replaced_by_completed_record",
            "replacement_game_ids": ["new-id"],
            "investigation_required": False,
        }
    ]


def test_cancelled_game_never_becomes_evidence() -> None:
    cancelled = _game(
        "cancelled-id",
        start="2026-09-05T19:30:00+00:00",
        completed=False,
        status="Cancelled",
        home_score=None,
        away_score=None,
    )

    eligible, exclusions = reconcile_game_evidence_rows([cancelled])

    assert eligible == []
    assert exclusions == []


def test_legitimate_same_team_rematch_is_not_collapsed() -> None:
    first = _game("first", start="2026-09-05T19:30:00+00:00")
    rematch = _game(
        "rematch",
        start="2026-11-21T19:30:00+00:00",
        home_score=21,
        away_score=17,
    )

    eligible, exclusions = reconcile_game_evidence_rows([first, rematch])

    assert [row["id"] for row in eligible] == ["first", "rematch"]
    assert exclusions == []


@pytest.mark.parametrize(
    ("second_start", "second_home_score", "second_away_score"),
    [
        ("2026-09-06T15:00:00+00:00", 28, 19),
        ("2026-09-06T15:00:00+00:00", 21, 17),
    ],
)
def test_two_plausible_close_results_fail_instead_of_being_collapsed(
    second_start: str, second_home_score: int, second_away_score: int
) -> None:
    first = _game("first-id", start="2026-09-05T19:30:00+00:00")
    second = _game(
        "second-id",
        start=second_start,
        home_score=second_home_score,
        away_score=second_away_score,
    )

    with pytest.raises(ValueError, match="ambiguous completed CFBD records"):
        reconcile_game_evidence_rows([first, second])


def test_conflicting_duplicate_game_id_fails_visibly() -> None:
    original = _game("same-id", start="2026-09-05T19:30:00+00:00")
    changed = {**original, "homePoints": "29"}

    with pytest.raises(ValueError, match="Conflicting CFBD game rows"):
        reconcile_game_evidence_rows([original, changed])


@pytest.mark.parametrize(
    ("completed", "status", "message"),
    [
        (True, "Scheduled", "marked completed while its status is pending"),
        (False, "Final", "completed status but completed is false"),
    ],
)
def test_conflicting_game_state_fields_fail_visibly(
    completed: bool, status: str, message: str
) -> None:
    row = _game(
        "state-conflict",
        start="2026-09-05T19:30:00+00:00",
        completed=completed,
        status=status,
        home_score=20 if completed else None,
        away_score=10 if completed else None,
    )

    with pytest.raises(ValueError, match=message):
        reconcile_game_evidence_rows([row])


def test_completed_scores_must_be_nonnegative_integers() -> None:
    row = _game("invalid-score", start="2026-09-05T19:30:00+00:00")
    row["homePoints"] = "-1"

    with pytest.raises(ValueError, match="non-negative integer home score"):
        reconcile_game_evidence_rows([row])


def test_excluded_suspicious_score_is_not_published_as_completed_schedule_state() -> None:
    row = _game(
        "401866625",
        start="2026-09-05T19:30:00+00:00",
        home_score=0,
        away_score=0,
    )
    metadata = {
        "snapshot_type": "weekly",
        "effective_cutoff": "2026-09-07T14:00:00+00:00",
        "source_retrieved_at": "2026-09-07T13:30:00+00:00",
        "included_game_ids": ["401917058"],
        "game_evidence_exclusions": [
            {
                "game_id": "401866625",
                "reason": "suspicious_completed_zero_zero_score",
                "replacement_game_ids": ["401917058"],
                "investigation_required": True,
            }
        ],
    }

    assert _schedule_state(row, metadata) == "unresolved"
