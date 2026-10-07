from __future__ import annotations

from gippyrank.research.offensive_line_roster_continuity import (
    build_continuity_artifacts,
    classify_offensive_line_position,
    normalize_position_label,
)


def _roster_row(
    player_id: str,
    first_name: str,
    last_name: str,
    team: str,
    position: str,
) -> dict[str, str]:
    return {
        "id": player_id,
        "firstName": first_name,
        "lastName": last_name,
        "team": team,
        "position": position,
        "jersey": "72",
    }


def test_offensive_line_position_mapping_is_explicit_and_fails_closed() -> None:
    for label in ("OL", "OT", "T", "OG", "G", "C", "OC", "LT", "RT", "LG", "RG", "IOL"):
        assert classify_offensive_line_position(label) == "offensive_line"
    assert classify_offensive_line_position("QB") == "non_offensive_line"
    assert classify_offensive_line_position("OL/DE") == "ambiguous"
    assert classify_offensive_line_position("Offensive Tackle?") == "unknown"
    assert classify_offensive_line_position("") == "unknown"
    assert normalize_position_label("  ol  ") == "OL"


def test_shared_roster_pairs_use_prior_same_program_seasons_only() -> None:
    teams = {
        2020: [
            {"id": 1, "school": "Alpha", "alternateNames": ["Alpha University"]},
            {"id": 2, "school": "Beta", "alternateNames": []},
        ],
        2021: [{"id": 1, "school": "Alpha", "alternateNames": ["Alpha University"]}],
    }
    roster_payloads = {
        ("fcs", 2019): [
            _roster_row("p1", "Alex", "One", "Alpha University", "QB"),
            _roster_row("p2", "Blair", "Two", "Alpha University", "DB"),
        ],
        ("fbs", 2020): [
            _roster_row("p1", "Alex", "One", "Alpha", "OL"),
            _roster_row("p2", "Blair", "Two", "Alpha", "OG"),
            _roster_row("p4", "Casey", "Four", "Beta", "OL"),
        ],
        ("fbs", 2021): [
            _roster_row("p1", "Alex", "One", "Alpha", "OT"),
            _roster_row("p2", "Blair", "Two", "Alpha", "G"),
            _roster_row("p3", "Dee", "Three", "Alpha", "C"),
            _roster_row("p4", "Casey", "Four", "Alpha", "OG"),
        ],
    }
    built = build_continuity_artifacts(
        roster_payloads,
        teams,
        roster_response_seasons={("fbs", 2020), ("fbs", 2021), ("fcs", 2019)},
        start_season=2019,
        end_season=2021,
    )
    pair = next(
        row
        for row in built.pairwise_continuity
        if row["target_season"] == 2021
        and {row["player_1_id"], row["player_2_id"]} == {"p1", "p2"}
    )
    assert pair["shared_prior_season_count"] == 2
    assert pair["earliest_shared_prior_season"] == 2019
    assert pair["most_recent_shared_prior_season"] == 2020
    assert pair["consecutive_shared_seasons_immediately_before_target"] == 2

    transfer_pair = next(
        row
        for row in built.pairwise_continuity
        if row["target_season"] == 2021
        and {row["player_1_id"], row["player_2_id"]} == {"p1", "p4"}
    )
    assert transfer_pair["shared_prior_season_count"] == 0
    assert (
        transfer_pair["pair_status"]
        == "observed_zero_shared_prior_roster_seasons_in_history_window"
    )

    target_coverage = next(
        row
        for row in built.team_season_coverage
        if row["season"] == 2021 and row["team_id"] == "1"
    )
    assert target_coverage["roster_player_rows"] == 4
    assert target_coverage["roster_rows_with_player_id"] == 4
    assert target_coverage["roster_rows_with_player_name"] == 4
    assert target_coverage["roster_rows_with_position"] == 4
    assert target_coverage["roster_rows_with_jersey_number"] == 4
    season_coverage = next(
        row for row in built.coverage_by_season if row["season"] == 2021
    )
    assert season_coverage["fbs_roster_player_rows"] == 4
    assert season_coverage["fbs_roster_player_id_rate"] == 1
    assert season_coverage["fbs_roster_player_name_rate"] == 1
    assert season_coverage["fbs_roster_position_rate"] == 1
    assert season_coverage["fbs_roster_jersey_number_rate"] == 1
    assert any(row["source_classification"] == "fcs" for row in built.player_seasons)


def test_unresolved_player_ids_do_not_become_zero_continuity() -> None:
    teams = {2021: [{"id": 1, "school": "Alpha", "alternateNames": []}]}
    roster_payloads = {
        ("fbs", 2020): [
            _roster_row("p1", "Alex", "One", "Alpha", "OL"),
        ],
        ("fbs", 2021): [
            _roster_row("p1", "Alex", "One", "Alpha", "OL"),
            _roster_row("", "No", "Id", "Alpha", "OT"),
        ],
    }
    built = build_continuity_artifacts(
        roster_payloads,
        teams,
        roster_response_seasons={("fbs", 2020), ("fbs", 2021)},
        start_season=2020,
        end_season=2021,
    )
    pair = built.pairwise_continuity[0]
    assert pair["pair_status"] == "unresolved_target_player_identity"
    assert pair["shared_prior_season_count"] == ""


def test_left_censored_first_target_season_has_blank_pair_measure() -> None:
    teams = {2004: [{"id": 1, "school": "Alpha", "alternateNames": []}]}
    roster_payloads = {
        ("fbs", 2004): [
            _roster_row("p1", "Alex", "One", "Alpha", "OL"),
            _roster_row("p2", "Blair", "Two", "Alpha", "OG"),
        ]
    }
    built = build_continuity_artifacts(
        roster_payloads,
        teams,
        roster_response_seasons={("fbs", 2004)},
        start_season=2004,
        end_season=2004,
    )
    assert (
        built.pairwise_continuity[0]["pair_status"]
        == "censored_incomplete_pre_2009_history"
    )
    assert built.pairwise_continuity[0]["shared_prior_season_count"] == ""
    assert built.team_season_summaries[0]["history_left_censored"] is True


def test_pre_2009_rosters_are_censored_and_2010_is_first_evaluable_target() -> None:
    teams = {
        season: [{"id": 1, "school": "Alpha", "alternateNames": []}]
        for season in (2008, 2009, 2010)
    }
    roster_payloads = {
        ("fbs", 2008): [
            _roster_row("p1", "Alex", "One", "Alpha", "OL"),
            _roster_row("p2", "Blair", "Two", "Alpha", "OG"),
            _roster_row("p3", "Casey", "Three", "Alpha", "OT"),
            _roster_row("p4", "Drew", "Four", "Alpha", "C"),
        ],
        ("fbs", 2009): [
            _roster_row("p1", "Alex", "One", "Alpha", "OL"),
            _roster_row("p2", "Blair", "Two", "Alpha", "OG"),
        ],
        ("fbs", 2010): [
            _roster_row("p1", "Alex", "One", "Alpha", "OL"),
            _roster_row("p2", "Blair", "Two", "Alpha", "OG"),
            _roster_row("p3", "Casey", "Three", "Alpha", "OT"),
            _roster_row("p4", "Drew", "Four", "Alpha", "C"),
        ],
    }
    built = build_continuity_artifacts(
        roster_payloads,
        teams,
        roster_response_seasons={("fbs", season) for season in (2008, 2009, 2010)},
        start_season=2008,
        end_season=2010,
    )

    censored_2009_pair = next(
        row
        for row in built.pairwise_continuity
        if row["target_season"] == 2009
        and {row["player_1_id"], row["player_2_id"]} == {"p1", "p2"}
    )
    assert censored_2009_pair["continuity_target_evaluable"] is False
    assert censored_2009_pair["continuity_history_status"] == (
        "censored_incomplete_pre_2009_history"
    )
    assert censored_2009_pair["shared_prior_season_count"] == ""

    evaluable_shared_pair = next(
        row
        for row in built.pairwise_continuity
        if row["target_season"] == 2010
        and {row["player_1_id"], row["player_2_id"]} == {"p1", "p2"}
    )
    assert evaluable_shared_pair["shared_prior_season_count"] == 1
    assert evaluable_shared_pair["shared_prior_season_count"] != ""

    observed_zero_pair = next(
        row
        for row in built.pairwise_continuity
        if row["target_season"] == 2010
        and {row["player_1_id"], row["player_2_id"]} == {"p3", "p4"}
    )
    assert observed_zero_pair["shared_prior_season_count"] == 0
    assert observed_zero_pair["pair_status"] == (
        "observed_zero_shared_prior_roster_seasons_in_history_window"
    )

    coverage_2009 = next(
        row for row in built.team_season_coverage if row["season"] == 2009
    )
    assert coverage_2009["continuity_target_evaluable"] is False
    assert coverage_2009["ol_with_no_prior_same_program_roster_link"] == ""
    summary_2009 = next(
        row for row in built.team_season_summaries if row["season"] == 2009
    )
    assert summary_2009["pair_count_history_censored"] == 1
    assert summary_2009["pairs_with_at_least_1_shared_season"] == ""
    annual_2009 = next(row for row in built.coverage_by_season if row["season"] == 2009)
    assert annual_2009["continuity_censored_team_season_count"] == 1
    assert annual_2009["pair_count_history_censored"] == 1
    annual_2010 = next(row for row in built.coverage_by_season if row["season"] == 2010)
    assert annual_2010["continuity_evaluable_team_season_count"] == 1
    assert built.source_audit["first_evaluable_target_season"] == 2010
