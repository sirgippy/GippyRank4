from __future__ import annotations

import pytest

from gippyrank.research.offensive_line_continuity_experiment import (
    CONTINUITY_FEATURES,
    INDIVIDUAL_EXPERIENCE_FEATURES,
    build_program_memberships,
    build_team_season_feature_panel,
    feature_values_for_pool,
    paired_season_bootstrap,
    validate_pairwise_artifact,
)


def _indexes(rows: list[dict[str, str]]):
    members, _seasons, _names, player_program_seasons = build_program_memberships(rows)
    return members, player_program_seasons


def test_features_use_only_the_prior_four_same_program_seasons() -> None:
    members, player_program_seasons = _indexes(
        [
            {"season": "2012", "team_id": "1", "normalized_player_identity": "A"},
            {"season": "2013", "team_id": "1", "normalized_player_identity": "A"},
            {"season": "2012", "team_id": "1", "normalized_player_identity": "B"},
            {"season": "2013", "team_id": "2", "normalized_player_identity": "B"},
            {"season": "2010", "team_id": "1", "normalized_player_identity": "C"},
            {"season": "2014", "team_id": "1", "normalized_player_identity": "A"},
            {"season": "2014", "team_id": "1", "normalized_player_identity": "B"},
            {"season": "2014", "team_id": "1", "normalized_player_identity": "C"},
            {"season": "2014", "team_id": "1", "normalized_player_identity": "D"},
        ]
    )

    values = feature_values_for_pool(
        season=2014,
        team_id="1",
        target_player_ids={"A", "B", "C", "D"},
        program_memberships=members,
        player_program_seasons=player_program_seasons,
    )

    assert values[CONTINUITY_FEATURES[0]] == pytest.approx(1 / 6)
    assert values[CONTINUITY_FEATURES[1]] == pytest.approx(1 / 6)
    assert values[CONTINUITY_FEATURES[2]] == pytest.approx(0.5)
    assert values[INDIVIDUAL_EXPERIENCE_FEATURES[0]] == pytest.approx(0.75)
    assert values[INDIVIDUAL_EXPERIENCE_FEATURES[1]] == pytest.approx(1.0)
    assert values["ol_prior_other_program_share_4y"] == pytest.approx(0.25)


def test_other_school_history_does_not_create_same_program_continuity() -> None:
    members, player_program_seasons = _indexes(
        [
            {"season": "2012", "team_id": "1", "normalized_player_identity": "B"},
            {"season": "2013", "team_id": "2", "normalized_player_identity": "A"},
        ]
    )

    values = feature_values_for_pool(
        season=2014,
        team_id="1",
        target_player_ids={"A", "B"},
        program_memberships=members,
        player_program_seasons=player_program_seasons,
    )

    assert values[CONTINUITY_FEATURES[0]] == 0.0
    assert values[CONTINUITY_FEATURES[1]] == 0.0
    assert values[CONTINUITY_FEATURES[2]] == 0.0
    assert values[INDIVIDUAL_EXPERIENCE_FEATURES[0]] == 0.5
    assert values[INDIVIDUAL_EXPERIENCE_FEATURES[1]] == 0.5
    assert values["ol_prior_other_program_share_4y"] == 0.5


def test_observed_zero_is_distinct_from_unavailable_feature_evidence() -> None:
    observed = feature_values_for_pool(
        season=2014,
        team_id="1",
        target_player_ids={"A", "B"},
        program_memberships={},
        player_program_seasons={},
    )
    censored = feature_values_for_pool(
        season=2014,
        team_id="1",
        target_player_ids={"A", "B"},
        program_memberships={},
        player_program_seasons={},
        history_window_observed=False,
    )
    identity_incomplete = feature_values_for_pool(
        season=2014,
        team_id="1",
        target_player_ids={"A", "B"},
        program_memberships={},
        player_program_seasons={},
        identity_complete=False,
    )

    assert all(
        observed[name] == 0.0
        for name in (*CONTINUITY_FEATURES, *INDIVIDUAL_EXPERIENCE_FEATURES)
    )
    assert all(
        censored[name] is None
        for name in (*CONTINUITY_FEATURES, *INDIVIDUAL_EXPERIENCE_FEATURES)
    )
    assert all(
        identity_incomplete[name] is None
        for name in (*CONTINUITY_FEATURES, *INDIVIDUAL_EXPERIENCE_FEATURES)
    )


def test_pairwise_artifact_is_checked_against_summary_counts() -> None:
    summaries = [
        {
            "season": "2013",
            "team_id": "1",
            "pair_count_evaluable": "1",
            "total_pairwise_shared_seasons": "0",
            "pairs_with_at_least_1_shared_season": "0",
        }
    ]
    pairs = [
        {
            "target_season": "2013",
            "team_id": "1",
            "shared_prior_season_count": "0",
        }
    ]

    audit = validate_pairwise_artifact(summaries, pairs)
    assert audit[(2013, "1")]["pair_mean_all_history"] == 0.0

    with pytest.raises(ValueError, match="count mismatch"):
        validate_pairwise_artifact(summaries, [])


def test_feature_panel_keeps_zero_and_coverage_flags_for_observed_target_pool() -> None:
    summary = {
        "season": "2013",
        "team_id": "1",
        "team_name": "Test State",
        "roster_response_available": "true",
        "roster_status": "roster_present_with_identifiable_ol",
        "continuity_target_evaluable": "true",
        "continuity_history_status": "evaluable_complete_history_window",
        "continuity_history_seasons_available": "4",
        "identifiable_ol_player_count": "2",
        "ol_source_rows": "2",
        "ol_player_ids_available": "2",
        "ol_player_ids_missing": "0",
        "unknown_position_rows": "0",
        "ambiguous_ol_position_rows": "0",
        "pair_count_evaluable": "1",
        "pair_count_history_censored": "0",
        "pair_count_identity_unresolved": "0",
        "total_pairwise_shared_seasons": "0",
        "pairs_with_at_least_1_shared_season": "0",
    }
    ol_rows = [
        {
            "season": "2013",
            "source_classification": "fbs",
            "team_id": "1",
            "normalized_player_identity": "A",
        },
        {
            "season": "2013",
            "source_classification": "fbs",
            "team_id": "1",
            "normalized_player_identity": "B",
        },
    ]
    audit = validate_pairwise_artifact(
        [summary],
        [
            {
                "target_season": "2013",
                "team_id": "1",
                "shared_prior_season_count": "0",
            }
        ],
    )

    panel = build_team_season_feature_panel(
        summary_rows=[summary],
        ol_rows=ol_rows,
        program_memberships={},
        player_program_seasons={},
        pairwise_audit=audit,
    )

    row = panel[0]
    assert row["target_ol_pool_usable"] is True
    assert row["primary_four_year_window_complete"] is True
    assert row["feature_status"] == "observed"
    assert row[CONTINUITY_FEATURES[0]] == 0.0
    assert row[CONTINUITY_FEATURES[1]] == 0.0


def test_season_cluster_bootstrap_is_deterministic_and_paired() -> None:
    deltas = {2022: [-0.2, 0.1], 2023: [-0.1], 2024: [0.3, -0.2]}

    first = paired_season_bootstrap(deltas)
    second = paired_season_bootstrap(deltas)

    assert first == second
    assert first["mean_delta"] == pytest.approx(-0.02)
    assert first["n_team_seasons"] == 5
    assert first["n_season_clusters"] == 3
    assert first["n_resamples"] == 27
