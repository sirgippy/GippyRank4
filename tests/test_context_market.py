from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from gippyrank.context_market import (
    COMPARISON_VARIANTS,
    ContextMarketError,
    PublicationChoice,
    actual_home_margin,
    build_per_game_rows,
    common_game_ids,
    context_version,
    favorite_side,
    parse_market_lines,
    prediction_rows,
    provider_coverage,
    select_model_publication,
    select_pregame_publication,
    select_same_provider_lines,
    sportsbook_spread_to_home_margin,
)


def _timestamp(day: int, hour: int = 0) -> str:
    return datetime(2026, 9, day, hour, tzinfo=UTC).isoformat().replace("+00:00", "Z")


def _publication(
    snapshot_id: str,
    *,
    version: str = "1.2",
    generation_timestamp: str,
    effective_cutoff: str | None,
    snapshot_type: str = "weekly",
    display_label: str = "Week publication",
) -> dict[str, object]:
    return {
        "season": 2026,
        "ranking_family": "predictive",
        "prior_family": "context",
        "model_versions": {"context_prior": version},
        "snapshot_id": snapshot_id,
        "snapshot_type": snapshot_type,
        "publication_slot": snapshot_id,
        "publication_status": "official",
        "generation_timestamp": generation_timestamp,
        "effective_cutoff": effective_cutoff,
        "requested_cutoff": effective_cutoff,
        "display_label": display_label,
    }


def test_spread_conversion_and_favorites_use_home_minus_away_orientation() -> None:
    # CFBD stores a home favorite as a negative spread; the study uses home
    # score minus away score, so its margin must be positive instead.
    assert sportsbook_spread_to_home_margin(-7.5) == 7.5
    assert favorite_side(sportsbook_spread_to_home_margin(-7.5)) == "home"

    # An away favorite is a positive CFBD spread and therefore a negative
    # expected home margin in the analysis orientation.
    assert sportsbook_spread_to_home_margin(3.0) == -3.0
    assert favorite_side(sportsbook_spread_to_home_margin(3.0)) == "away"

    assert sportsbook_spread_to_home_margin(0) == 0.0
    assert favorite_side(sportsbook_spread_to_home_margin(0)) == "pickem"


def test_actual_home_margin_keeps_home_minus_away_orientation() -> None:
    assert actual_home_margin(31, 17) == 14.0
    assert actual_home_margin(17, 31) == -14.0
    assert actual_home_margin(21, None) is None


def test_same_provider_matching_never_fills_one_books_missing_open_with_another() -> (
    None
):
    market_rows = parse_market_lines(
        [
            {
                "id": 100,
                "lines": [
                    {
                        "provider": "DraftKings",
                        "formattedSpread": "Home -6",
                        "spread": -6,
                        "spreadOpen": -4,
                    },
                    {
                        "provider": "Bovada",
                        "formattedSpread": "Home -7",
                        "spread": -7,
                        "spreadOpen": -5,
                    },
                ],
            },
            {
                "id": 101,
                "lines": [
                    {
                        "provider": "DraftKings",
                        "formattedSpread": "Away -3",
                        "spread": 3,
                        "spreadOpen": None,
                    },
                    {
                        "provider": "Bovada",
                        "formattedSpread": "Away -2",
                        "spread": 2,
                        "spreadOpen": 1,
                    },
                ],
            },
        ]
    )

    draftkings = select_same_provider_lines(market_rows, "DraftKings")
    assert draftkings["100"]["opening_home_margin"] == 4.0
    assert draftkings["101"]["later_home_margin"] == -3.0
    assert draftkings["101"]["opening_home_margin"] is None

    coverage = {
        row["provider"]: row
        for row in provider_coverage(market_rows, game_ids=["100", "101"])
    }
    assert coverage["DraftKings"]["same_provider_open_later_games"] == 1
    assert coverage["Bovada"]["same_provider_open_later_games"] == 2


def test_context_version_and_cutoff_safe_selection_reject_late_retrospective() -> None:
    assert context_version({"model_versions": {"context_prior": 1.3}}) == "1.3"
    assert context_version({"prior_model_version": "1.2"}) == "1.2"
    assert context_version({"model_versions": {"history_prior": "1.1"}}) is None

    week_start = datetime(2026, 9, 17, tzinfo=UTC)
    publications = [
        _publication(
            "valid-earlier",
            generation_timestamp=_timestamp(10),
            effective_cutoff=_timestamp(10),
        ),
        _publication(
            "valid-latest",
            generation_timestamp=_timestamp(12),
            effective_cutoff=_timestamp(12),
        ),
        _publication(
            "cutoff-too-late",
            generation_timestamp=_timestamp(12),
            effective_cutoff=_timestamp(17),
        ),
        _publication(
            "generated-too-late",
            generation_timestamp=_timestamp(17),
            effective_cutoff=_timestamp(12),
        ),
        _publication(
            "retrospective",
            version="1.3",
            generation_timestamp=_timestamp(20),
            effective_cutoff=_timestamp(13),
            display_label="Week 3 (Context 1.3 retrospective)",
        ),
    ]

    selected = select_pregame_publication(
        publications,
        season=2026,
        week=3,
        week_start=week_start,
        version="1.2",
    )
    assert selected.snapshot_id == "valid-latest"
    assert selected.effective_cutoff == _timestamp(12)

    rejected = select_pregame_publication(
        publications,
        season=2026,
        week=3,
        week_start=week_start,
        version="1.3",
    )
    assert rejected.snapshot_id is None
    assert rejected.reason is not None
    assert "explicitly retrospective" in rejected.reason

    # Even if a caller elects to include a retrospective reconstruction, it
    # remains ineligible when the artifact was generated after kickoff.
    late = select_pregame_publication(
        publications,
        season=2026,
        week=3,
        week_start=week_start,
        version="1.3",
        allow_retrospective=True,
    )
    assert late.snapshot_id is None
    assert late.reason is not None
    assert "generated after target-week start" in late.reason


def test_reconstruction_selects_history_baseline_and_postgame_context_replay() -> None:
    week_start = datetime(2026, 9, 17, tzinfo=UTC)
    history = _publication(
        "history-1.1",
        generation_timestamp=_timestamp(20),
        effective_cutoff=_timestamp(13),
    )
    history["prior_family"] = "history"
    history["model_versions"] = {"history_prior": "1.1", "context_prior": "1.2"}
    history["source_retrieved_at"] = _timestamp(13)
    history["source_context_snapshot_id"] = "canonical-context-1.2"
    history["comparison_snapshot_id"] = "comparison-context-1.2"
    context_replay = _publication(
        "context-1.3-replay",
        version="1.3",
        generation_timestamp=_timestamp(20),
        effective_cutoff=_timestamp(13),
        display_label="Week 3 (Context 1.3 retrospective)",
    )
    variants = {variant.key: variant for variant in COMPARISON_VARIANTS}

    history_choice = select_model_publication(
        [history],
        season=2026,
        week=3,
        week_start=week_start,
        variant=variants["history-1.1"],
        selection_policy="reconstruction",
    )
    assert history_choice.snapshot_id == "history-1.1"
    assert history_choice.variant_label == "History 1.1"
    assert history_choice.model_family == "history"
    assert history_choice.source_context_snapshot_id == "canonical-context-1.2"
    assert history_choice.comparison_snapshot_id == "comparison-context-1.2"
    assert history_choice.timing_classification == (
        "postgame_generated_reconstruction_with_pregame_cutoff"
    )

    replay_choice = select_model_publication(
        [context_replay],
        season=2026,
        week=3,
        week_start=week_start,
        variant=variants["context-1.3"],
        selection_policy="reconstruction",
    )
    assert replay_choice.snapshot_id == "context-1.3-replay"
    assert replay_choice.is_retrospective_artifact is True
    assert replay_choice.generated_after_target_week_start is True

    strict_choice = select_model_publication(
        [context_replay],
        season=2026,
        week=3,
        week_start=week_start,
        variant=variants["context-1.3"],
        selection_policy="strict_pregame",
    )
    assert strict_choice.snapshot_id is None
    assert strict_choice.reason is not None
    assert "explicitly retrospective" in strict_choice.reason


def test_reconstruction_allows_historical_source_state_only_when_requested() -> None:
    choice = PublicationChoice(
        variant_key="context-1.3",
        week=3,
        snapshot_id="reconstructed-context",
        publication_slot="2026-09-13",
        publication_status="official",
        snapshot_type="weekly",
        generation_timestamp=_timestamp(20),
        effective_cutoff=_timestamp(13),
        requested_cutoff=_timestamp(13),
        reason=None,
        variant_label="Context 1.3",
        model_family="context",
        model_version="1.3",
        model_version_field="context_prior",
        selection_policy="reconstruction",
        is_reconstruction=True,
    )
    payload = {
        "week": {"week": 3},
        "games": [
            {
                "game_id": "501",
                "state": "completed",
                "home_team_id": "home",
                "away_team_id": "away",
                "home_team": {"team_id": "home", "subdivision": "fbs"},
                "away_team": {"team_id": "away", "subdivision": "fbs"},
                "prediction": {
                    "game_id": "501",
                    "source_snapshot_id": "reconstructed-context",
                    "expected_home_margin": 7,
                },
            }
        ],
    }
    with pytest.raises(ContextMarketError, match="non-future"):
        prediction_rows(payload, choice=choice)

    rows = prediction_rows(payload, choice=choice, allow_nonfuture_state=True)
    assert rows[0]["source_prediction_state"] == "completed"


def test_common_game_ids_uses_primary_complete_paired_intersection() -> None:
    frame = pd.DataFrame(
        [
            {
                "variant_key": version,
                "game_id": "shared",
                "is_primary_fbs_vs_fbs": True,
                "gippy_expected_home_margin": 1.0,
                "opening_home_margin": 2.0,
                "actual_home_margin": 3.0,
            }
            for version in ("1.1", "1.2", "1.3")
        ]
        + [
            {
                "variant_key": "1.1",
                "game_id": "missing-for-1.3",
                "is_primary_fbs_vs_fbs": True,
                "gippy_expected_home_margin": 1.0,
                "opening_home_margin": 2.0,
                "actual_home_margin": 3.0,
            },
            {
                "variant_key": "1.2",
                "game_id": "missing-for-1.3",
                "is_primary_fbs_vs_fbs": True,
                "gippy_expected_home_margin": 1.0,
                "opening_home_margin": 2.0,
                "actual_home_margin": 3.0,
            },
            {
                "variant_key": "1.3",
                "game_id": "not-primary",
                "is_primary_fbs_vs_fbs": False,
                "gippy_expected_home_margin": 1.0,
                "opening_home_margin": 2.0,
                "actual_home_margin": 3.0,
            },
        ]
    )

    assert common_game_ids(
        frame,
        versions=("1.1", "1.2", "1.3"),
        required=(
            "gippy_expected_home_margin",
            "opening_home_margin",
            "actual_home_margin",
        ),
    ) == {"shared"}


def test_build_per_game_rows_preserves_gippy_orientation_and_movement_toward() -> None:
    choice = PublicationChoice(
        variant_key="context-1.2",
        week=3,
        snapshot_id="pregame-context",
        publication_slot="2026-09-13",
        publication_status="official",
        snapshot_type="weekly",
        generation_timestamp=_timestamp(13),
        effective_cutoff=_timestamp(13),
        requested_cutoff=_timestamp(13),
        reason=None,
    )
    predictions = [
        {
            "game_id": "501",
            # Positive remains a home-team expected win; it must not be
            # inverted while joining to schedule or market data.
            "gippy_expected_home_margin": 10.0,
            "gippy_median_home_margin": 10.5,
            "prediction_home_team_id": "home",
            "prediction_away_team_id": "away",
            "prediction_home_subdivision": "fbs",
            "prediction_away_subdivision": "fbs",
        }
    ]
    schedule = {
        "501": {
            "id": "501",
            "week": 3,
            "homeId": "home",
            "awayId": "away",
            "homeTeam": "Home",
            "awayTeam": "Away",
            "homeClassification": "fbs",
            "awayClassification": "fbs",
            "homePoints": 27,
            "awayPoints": 20,
            "startDate": "2026-09-19T16:00:00Z",
            "neutralSite": False,
            "completed": True,
        }
    }
    market = {
        "501": {
            "provider": "Bovada",
            "formatted_spread": "Home -6",
            "opening_home_margin": 6.0,
            "later_home_margin": 8.0,
        }
    }

    row = build_per_game_rows(
        predictions,
        schedule=schedule,
        market=market,
        choice=choice,
        provider="Bovada",
    )[0]

    assert row["gippy_expected_home_margin"] == 10.0
    assert row["actual_home_margin"] == 7.0
    assert row["gippy_minus_open"] == 4.0
    assert row["gippy_minus_close"] == 2.0
    assert row["market_move"] == 2.0
    assert row["market_movement_toward_gippy"] == 2.0
    assert row["is_primary_fbs_vs_fbs"] is True
    assert row["variant_key"] == "context-1.2"

    schedule["501"]["week"] = 2
    with pytest.raises(ContextMarketError, match="target week"):
        build_per_game_rows(
            predictions,
            schedule=schedule,
            market=market,
            choice=choice,
            provider="Bovada",
        )
