from dataclasses import replace
from datetime import date

import pytest

from gippyrank.defensive_transfer import (
    add_defensive_impact,
    aggregate_player_seasons,
    audit_transfer_records,
    descriptive_statistics,
    parse_games_players_payload,
    parse_roster_payload,
    position_mapping,
    position_semantics,
    team_game_keys,
)
from gippyrank.transfer_oracle import parse_transfer_payload


def _roster() -> list:
    return parse_roster_payload(
        [
            {
                "id": "1",
                "firstName": "A",
                "lastName": "Defender",
                "team": "Alpha",
                "position": "DE",
            },
            {
                "id": "2",
                "firstName": "B",
                "lastName": "Linebacker",
                "team": "Alpha",
                "position": "LB",
            },
            {
                "id": "3",
                "firstName": "C",
                "lastName": "Back",
                "team": "Alpha",
                "position": "CB",
            },
            {
                "id": "4",
                "firstName": "D",
                "lastName": "Receiver",
                "team": "Alpha",
                "position": "WR",
            },
            {
                "id": "5",
                "firstName": "E",
                "lastName": "Linebacker",
                "team": "Alpha",
                "position": "LB",
            },
        ],
        season=2021,
    )


def _game_payload() -> list[dict]:
    return [
        {
            "id": 100,
            "teams": [
                {
                    "team": "Alpha",
                    "categories": [
                        {
                            "name": "defensive",
                            "types": [
                                {
                                    "name": "TOT",
                                    "athletes": [
                                        {"id": "1", "name": "A Defender", "stat": "8"},
                                        {"id": "3", "name": "C Back", "stat": "0"},
                                        {
                                            "id": "5",
                                            "name": "E Linebacker",
                                            "stat": "10",
                                        },
                                    ],
                                },
                                {
                                    "name": "TFL",
                                    "athletes": [
                                        {"id": "1", "name": "A Defender", "stat": "2"},
                                        {"id": "3", "name": "C Back", "stat": "0"},
                                        {
                                            "id": "5",
                                            "name": "E Linebacker",
                                            "stat": "3",
                                        },
                                    ],
                                },
                                {
                                    "name": "SACKS",
                                    "athletes": [
                                        {"id": "1", "name": "A Defender", "stat": "1"},
                                        {"id": "3", "name": "C Back", "stat": "0"},
                                    ],
                                },
                                {
                                    "name": "QB HUR",
                                    "athletes": [
                                        {"id": "1", "name": "A Defender", "stat": "3"},
                                        {"id": "3", "name": "C Back", "stat": "0"},
                                    ],
                                },
                                {
                                    "name": "PD",
                                    "athletes": [
                                        {"id": "1", "name": "A Defender", "stat": "0"},
                                        {"id": "3", "name": "C Back", "stat": "1"},
                                    ],
                                },
                            ],
                        },
                        {
                            "name": "interceptions",
                            "types": [
                                {
                                    "name": "INT",
                                    "athletes": [
                                        {"id": "3", "name": "C Back", "stat": "1"}
                                    ],
                                }
                            ],
                        },
                    ],
                },
                {"team": "Beta", "categories": []},
            ],
        },
        {
            "id": 101,
            "teams": [
                {
                    "team": "Alpha",
                    "categories": [
                        {
                            "name": "defensive",
                            "types": [
                                {
                                    "name": "TOT",
                                    "athletes": [
                                        {"id": "1", "name": "A Defender", "stat": "4"}
                                    ],
                                }
                            ],
                        }
                    ],
                },
                {"team": "Beta", "categories": []},
            ],
        },
    ]


def test_position_taxonomy_is_explicit_and_unknowns_fail_closed() -> None:
    mapping = position_mapping()
    assert mapping["groups"]["dl_edge"]["source_positions"] == [
        "DE",
        "DL",
        "DT",
        "EDGE",
        "NT",
    ]
    assert position_semantics("CB") == "defensive"
    assert position_semantics("WR") == "non_defensive"
    assert position_semantics("ATH") == "unknown"


def test_parser_and_aggregation_preserve_zero_events_and_game_rate() -> None:
    game_players = parse_games_players_payload(_game_payload(), season=2021)
    assert len(game_players) == 4
    assert {item.player_id for item in game_players} == {"1", "3", "5"}
    assert (
        next(item for item in game_players if item.player_id == "3").stats[
            "interceptions"
        ]
        == 1
    )
    players = add_defensive_impact(
        aggregate_player_seasons(
            game_players, _roster(), team_game_keys(_game_payload(), season=2021)
        )
    )
    defender = next(item for item in players if item.player_id == "1")
    corner = next(item for item in players if item.player_id == "3")
    assert defender.recorded_defensive_box_score_games == 2
    assert defender.team_games == 2
    assert defender.defensive_box_score_game_rate == 1.0
    assert corner.recorded_defensive_box_score_games == 1
    assert corner.defensive_box_score_game_rate == 0.5
    assert defender.defensive_impact == pytest.approx(0.0)
    assert corner.defensive_impact == pytest.approx(0.0)
    zero_linebacker = next(item for item in players if item.player_id == "2")
    assert zero_linebacker.recorded_defensive_box_score_games == 0
    assert zero_linebacker.defensive_box_score_game_rate == 0.0
    assert zero_linebacker.defensive_impact < 0.0


def test_aggregation_deduplicates_same_game_from_overlapping_division_queries() -> None:
    game_players = parse_games_players_payload(_game_payload(), season=2021)
    players = aggregate_player_seasons(
        [*game_players, *game_players],
        _roster(),
        team_game_keys(_game_payload(), season=2021),
    )
    defender = next(item for item in players if item.player_id == "1")
    assert defender.recorded_defensive_box_score_games == 2
    assert defender.stats["tackles"] == 12


def test_audit_distinguishes_zero_identity_failure_and_non_defensive() -> None:
    game_players = parse_games_players_payload(_game_payload(), season=2021)
    players = add_defensive_impact(
        aggregate_player_seasons(
            game_players, _roster(), team_game_keys(_game_payload(), season=2021)
        )
    )
    records = parse_transfer_payload(
        [
            {
                "season": 2022,
                "firstName": "A",
                "lastName": "Defender",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "DE",
                "transferDate": "2022-08-01",
            },
            {
                "season": 2022,
                "firstName": "B",
                "lastName": "Linebacker",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "LB",
                "transferDate": "2022-08-01",
            },
            {
                "season": 2022,
                "firstName": "Missing",
                "lastName": "Player",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "CB",
                "transferDate": "2022-08-01",
            },
            {
                "season": 2022,
                "firstName": "D",
                "lastName": "Receiver",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "WR",
                "transferDate": "2022-08-01",
            },
        ],
        season=2022,
    )
    result = audit_transfer_records(
        records,
        _roster(),
        players,
        [{"season": "2022", "subdivision": "fbs", "team_id": "2", "team_name": "Beta"}],
        portal_seasons={2022},
        defensive_seasons={2021},
        team_coverage={(2021, "alpha"), (2021, "beta")},
        cutoff=date(2025, 8, 15),
    )
    by_player = {row["player_name"]: row for row in result["player_rows"]}
    assert by_player["A Defender"]["experience_status"] == "resolved"
    assert (
        by_player["B Linebacker"]["experience_status"]
        == "zero_recorded_defensive_box_score_games"
    )
    assert by_player["B Linebacker"]["prior_defensive_impact"] < 0.0
    assert (
        by_player["Missing Player"]["experience_status"]
        == "identity_resolution_failure"
    )
    assert (
        by_player["D Receiver"]["experience_status"] == "non_defensive_not_applicable"
    )
    beta = result["team_rows"][0]
    assert beta["incoming_defensive_transfers"] == 3
    assert beta["feature_coverage_status"] == "partial"
    assert beta["transfer_in_prior_defensive_experience_sum"] is None


def test_roster_player_id_resolves_stats_when_player_name_differs() -> None:
    game_players = parse_games_players_payload(_game_payload(), season=2021)
    players = add_defensive_impact(
        aggregate_player_seasons(
            game_players, _roster(), team_game_keys(_game_payload(), season=2021)
        )
    )
    players = [
        replace(row, player_name="Provider Stats Alias")
        if row.player_id == "1"
        else row
        for row in players
    ]
    record = parse_transfer_payload(
        [
            {
                "season": 2022,
                "firstName": "A",
                "lastName": "Defender",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "DE",
                "transferDate": "2022-08-01",
            }
        ],
        season=2022,
    )[0]
    result = audit_transfer_records(
        [record],
        _roster(),
        players,
        [{"season": "2022", "subdivision": "fbs", "team_id": "2", "team_name": "Beta"}],
        portal_seasons={2022},
        defensive_seasons={2021},
        team_coverage={(2021, "alpha"), (2021, "beta")},
        cutoff=date(2025, 8, 15),
    )
    row = result["player_rows"][0]
    assert row["identity_status"] == "resolved"
    assert row["impact_status"] == "resolved"
    assert row["prior_player_id"] == "1"
    assert row["impact_join_method"] == "stable_player_id_source_team"


def test_unique_game_player_id_bridges_roster_name_mismatch() -> None:
    game_players = parse_games_players_payload(_game_payload(), season=2021)
    roster = [
        replace(item, player_name="Provider Roster Spelling")
        if item.player_id == "1"
        else item
        for item in _roster()
    ]
    players = add_defensive_impact(
        aggregate_player_seasons(
            game_players, roster, team_game_keys(_game_payload(), season=2021)
        )
    )
    record = parse_transfer_payload(
        [
            {
                "season": 2022,
                "firstName": "A",
                "lastName": "Defender",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "DE",
                "transferDate": "2022-08-01",
            }
        ],
        season=2022,
    )[0]
    teams = [
        {"season": "2022", "subdivision": "fbs", "team_id": "2", "team_name": "Beta"}
    ]

    result = audit_transfer_records(
        [record],
        roster,
        players,
        teams,
        portal_seasons={2022},
        defensive_seasons={2021},
        team_coverage={(2021, "alpha"), (2021, "beta")},
        cutoff=date(2025, 8, 15),
        identity_bridge_players=game_players,
    )

    row = result["player_rows"][0]
    assert row["identity_status"] == "resolved"
    assert row["impact_status"] == "resolved"
    assert row["identity_join_method"] == "stable_game_player_id_source_team"
    assert row["prior_player_id"] == "1"


def test_generational_suffix_match_uses_unique_same_team_roster_row() -> None:
    roster = _roster()
    game_players = parse_games_players_payload(_game_payload(), season=2021)
    players = add_defensive_impact(
        aggregate_player_seasons(
            game_players, roster, team_game_keys(_game_payload(), season=2021)
        )
    )
    record = parse_transfer_payload(
        [
            {
                "season": 2022,
                "firstName": "A",
                "lastName": "Defender Jr.",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "DE",
                "transferDate": "2022-08-01",
            }
        ],
        season=2022,
    )[0]
    result = audit_transfer_records(
        [record],
        roster,
        players,
        [{"season": "2022", "subdivision": "fbs", "team_id": "2", "team_name": "Beta"}],
        portal_seasons={2022},
        defensive_seasons={2021},
        team_coverage={(2021, "alpha"), (2021, "beta")},
        roster_teams={(2021, "alpha")},
        cutoff=date(2025, 8, 15),
    )

    row = result["player_rows"][0]
    assert row["identity_status"] == "resolved"
    assert (
        row["identity_join_method"] == "normalized_name_source_team_generational_suffix"
    )
    assert row["prior_player_id"] == "1"


def test_generational_suffix_match_fails_closed_for_multiple_roster_candidates() -> (
    None
):
    roster = [
        *_roster(),
        replace(_roster()[0], player_name="A Defender II", player_id="6"),
    ]
    record = parse_transfer_payload(
        [
            {
                "season": 2022,
                "firstName": "A",
                "lastName": "Defender Jr.",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "DE",
                "transferDate": "2022-08-01",
            }
        ],
        season=2022,
    )[0]
    result = audit_transfer_records(
        [record],
        roster,
        [],
        [{"season": "2022", "subdivision": "fbs", "team_id": "2", "team_name": "Beta"}],
        portal_seasons={2022},
        defensive_seasons={2021},
        roster_teams={(2021, "alpha")},
        cutoff=date(2025, 8, 15),
    )

    row = result["player_rows"][0]
    assert row["identity_status"] == "ambiguous"
    assert row["impact_status"] == "ambiguous"


def test_game_player_identity_bridge_fails_closed_when_provider_ids_conflict() -> None:
    game_players = parse_games_players_payload(_game_payload(), season=2021)
    roster = [
        replace(item, player_name="Provider Roster Spelling")
        if item.player_id == "1"
        else item
        for item in _roster()
    ]
    conflicting_game_player = replace(
        next(item for item in game_players if item.player_id == "1"), player_id="6"
    )
    record = parse_transfer_payload(
        [
            {
                "season": 2022,
                "firstName": "A",
                "lastName": "Defender",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "DE",
                "transferDate": "2022-08-01",
            }
        ],
        season=2022,
    )[0]

    result = audit_transfer_records(
        [record],
        roster,
        [],
        [{"season": "2022", "subdivision": "fbs", "team_id": "2", "team_name": "Beta"}],
        portal_seasons={2022},
        defensive_seasons={2021},
        team_coverage={(2021, "alpha"), (2021, "beta")},
        cutoff=date(2025, 8, 15),
        identity_bridge_players=[*game_players, conflicting_game_player],
    )

    assert result["player_rows"][0]["identity_status"] == "ambiguous"


def test_ambiguous_stats_under_roster_id_fail_closed() -> None:
    game_players = parse_games_players_payload(_game_payload(), season=2021)
    players = add_defensive_impact(
        aggregate_player_seasons(
            game_players, _roster(), team_game_keys(_game_payload(), season=2021)
        )
    )
    duplicate = next(row for row in players if row.player_id == "1")
    players.append(duplicate)
    record = parse_transfer_payload(
        [
            {
                "season": 2022,
                "firstName": "A",
                "lastName": "Defender",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "DE",
                "transferDate": "2022-08-01",
            }
        ],
        season=2022,
    )[0]
    result = audit_transfer_records(
        [record],
        _roster(),
        players,
        [{"season": "2022", "subdivision": "fbs", "team_id": "2", "team_name": "Beta"}],
        portal_seasons={2022},
        defensive_seasons={2021},
        team_coverage={(2021, "alpha"), (2021, "beta")},
        cutoff=date(2025, 8, 15),
    )
    row = result["player_rows"][0]
    assert row["identity_status"] == "ambiguous"
    assert row["impact_status"] == "ambiguous"


def test_distribution_statistics_report_missing_values() -> None:
    stats = descriptive_statistics([0.0, 1.0, None, 2.0])
    assert stats["count"] == 3
    assert stats["missing_count"] == 1
    assert stats["fraction_zero"] == pytest.approx(1 / 3)


def test_conflicting_portal_player_id_does_not_join_by_name() -> None:
    records = parse_transfer_payload(
        [
            {
                "season": 2022,
                "id": "different-id",
                "firstName": "A",
                "lastName": "Defender",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "CB",
                "transferDate": "2022-08-01",
            }
        ],
        season=2022,
    )
    result = audit_transfer_records(
        records,
        _roster(),
        [],
        [{"season": "2022", "subdivision": "fbs", "team_id": "2", "team_name": "Beta"}],
        portal_seasons={2022},
        defensive_seasons={2021},
        cutoff=date(2025, 8, 15),
        roster_teams={(2021, "alpha")},
    )
    row = result["player_rows"][0]
    assert row["identity_status"] == "identity_resolution_failure"
    assert row["identity_join_method"] == "stable_player_id_conflict"
    assert (
        row["identity_resolution_detail"]
        == "portal_player_id_conflicts_with_name_match"
    )
    assert row["roster_candidate_count"] == 1
