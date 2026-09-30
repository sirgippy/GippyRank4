from datetime import date

import pytest

from gippyrank.transfer_oracle import (
    aggregate_team_features,
    available_by_cutoff,
    normalize_player_name,
    parse_transfer_date,
    parse_transfer_payload,
    parse_usage_payload,
    position_group,
    transfer_identity_key,
)


def team_rows() -> list[dict[str, str]]:
    return [
        {
            "season": "2022",
            "subdivision": "fbs",
            "team_id": "1",
            "team_name": "Alpha",
            "returning_pct_ppa": "0.4",
        },
        {
            "season": "2022",
            "subdivision": "fbs",
            "team_id": "2",
            "team_name": "Beta",
            "returning_pct_ppa": "0.6",
        },
    ]


def test_portal_parser_keeps_fields_and_cutoff_semantics() -> None:
    records = parse_transfer_payload(
        [
            {
                "season": 2022,
                "firstName": "A.",
                "lastName": "Player",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "QB",
                "transferDate": "2022-08-01T05:00:00.000Z",
                "rating": 0.98,
                "stars": 5,
                "eligibility": "Immediate",
            }
        ],
        season=2022,
    )
    assert records[0].player_name == "A. Player"
    assert records[0].transfer_date == date(2022, 8, 1)
    assert available_by_cutoff(records[0], date(2022, 8, 15))
    assert not available_by_cutoff(records[0], date(2022, 7, 1))
    assert position_group("EDGE") == "dl"


def test_player_name_normalization_handles_unicode_punctuation_safely() -> None:
    assert normalize_player_name("Ja’Bari Odoemenem") == normalize_player_name(
        "Ja'Bari Odoemenem"
    )
    assert normalize_player_name("Wells–Ross") == normalize_player_name("Wells Ross")
    assert normalize_player_name("Ａ．Ｂ. Smith") == "ab smith"
    assert normalize_player_name("José García") != normalize_player_name("Jose Garcia")
    assert normalize_player_name("Sam Player Jr.") != normalize_player_name(
        "Sam Player"
    )


def test_cutoff_is_relative_to_each_transfer_season() -> None:
    records = [
        *parse_transfer_payload(
            [
                {
                    "season": 2022,
                    "firstName": "Late",
                    "lastName": "Transfer",
                    "transferDate": "2022-08-16",
                }
            ],
            season=2022,
        ),
        *parse_transfer_payload(
            [
                {
                    "season": 2023,
                    "firstName": "Cutoff",
                    "lastName": "Transfer",
                    "transferDate": "2023-08-15",
                }
            ],
            season=2023,
        ),
    ]
    configured_cutoff = date(2025, 8, 15)
    assert not available_by_cutoff(records[0], configured_cutoff)
    assert available_by_cutoff(records[1], configured_cutoff)


def test_aggregate_distinguishes_uncovered_from_zero_activity() -> None:
    records = parse_transfer_payload(
        [
            {
                "season": 2022,
                "firstName": "A",
                "lastName": "Player",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "QB",
                "transferDate": "2022-08-01",
                "rating": 0.98,
                "stars": 5,
            }
        ],
        season=2022,
    )
    usage = parse_usage_payload(
        [
            {
                "season": 2021,
                "name": "A Player",
                "team": "Alpha",
                "position": "QB",
                "usage": {"overall": 0.5},
            }
        ],
        season=2021,
    )
    rows = [*team_rows(), {**team_rows()[0], "season": "2021", "team_id": "1"}]
    result = aggregate_team_features(
        records,
        usage,
        rows,
        covered_seasons={2022},
        cutoff=date(2022, 8, 15),
    )
    beta = result[(2022, "fbs", "2")]
    alpha = result[(2022, "fbs", "1")]
    old = result[(2021, "fbs", "1")]
    assert beta["transfer_data_available"] == 1.0
    assert beta["transfer_in_count"] == 1.0
    assert beta["transfer_in_count_qb"] == 1.0
    assert beta["transfer_in_prior_usage_sum"] == pytest.approx(0.5)
    assert alpha["transfer_out_count"] == 1.0
    assert old["transfer_data_available"] is None
    assert old["transfer_in_count"] is None


def test_usage_join_is_name_normalized_and_missing_quality_is_not_zero() -> None:
    records = parse_transfer_payload(
        [
            {
                "season": 2022,
                "firstName": "A",
                "lastName": "Player",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "WR",
                "transferDate": "2022-07-01",
            }
        ],
        season=2022,
    )
    result = aggregate_team_features(
        records,
        [],
        team_rows(),
        covered_seasons={2022},
        cutoff=date(2022, 8, 15),
    )
    beta = result[(2022, "fbs", "2")]
    assert beta["transfer_in_count"] == 1.0
    assert beta["transfer_in_rating_sum"] is None
    assert beta["transfer_in_prior_usage_sum"] is None


def test_aggregation_uses_only_explicit_audit_evidence_for_legitimate_zero() -> None:
    records = parse_transfer_payload(
        [
            {
                "season": 2022,
                "firstName": "A",
                "lastName": "Player",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "QB",
                "transferDate": "2022-07-01",
            }
        ],
        season=2022,
    )
    missing = aggregate_team_features(
        records,
        [],
        team_rows(),
        covered_seasons={2022},
        cutoff=date(2022, 8, 15),
    )
    repaired = aggregate_team_features(
        records,
        [],
        team_rows(),
        covered_seasons={2022},
        cutoff=date(2022, 8, 15),
        verified_zero_usage_keys={transfer_identity_key(records[0])},
    )
    assert missing[(2022, "fbs", "2")]["transfer_in_prior_usage_sum"] is None
    assert repaired[(2022, "fbs", "2")]["transfer_in_prior_usage_sum"] == 0.0


def test_verified_zero_evidence_cannot_override_positive_usage() -> None:
    records = parse_transfer_payload(
        [
            {
                "season": 2022,
                "firstName": "A",
                "lastName": "Player",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "QB",
                "transferDate": "2022-07-01",
            }
        ],
        season=2022,
    )
    usage = parse_usage_payload(
        [
            {
                "season": 2021,
                "name": "A Player",
                "team": "Alpha",
                "position": "QB",
                "usage": {"overall": 0.25},
            }
        ],
        season=2021,
    )
    with pytest.raises(ValueError, match="conflicts with a positive usage row"):
        aggregate_team_features(
            records,
            usage,
            team_rows(),
            covered_seasons={2022},
            cutoff=date(2022, 8, 15),
            verified_zero_usage_keys={transfer_identity_key(records[0])},
        )


def test_ambiguous_normalized_usage_names_fail_closed() -> None:
    records = parse_transfer_payload(
        [
            {
                "season": 2022,
                "firstName": "A-B",
                "lastName": "Player",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "WR",
                "transferDate": "2022-07-01",
            }
        ],
        season=2022,
    )
    usage = parse_usage_payload(
        [
            {
                "season": 2021,
                "name": "A-B Player",
                "team": "Alpha",
                "usage": {"overall": 0.2},
            },
            {
                "season": 2021,
                "name": "A B Player",
                "team": "Alpha",
                "usage": {"overall": 0.8},
            },
        ],
        season=2021,
    )
    result = aggregate_team_features(
        records,
        usage,
        team_rows(),
        covered_seasons={2022},
        cutoff=date(2022, 8, 15),
    )
    assert result[(2022, "fbs", "2")]["transfer_in_prior_usage_sum"] is None


def test_explicit_zero_cannot_hide_positive_value_in_ambiguous_usage_rows() -> None:
    records = parse_transfer_payload(
        [
            {
                "season": 2022,
                "firstName": "A-B",
                "lastName": "Player",
                "origin": "Alpha",
                "destination": "Beta",
                "transferDate": "2022-07-01",
            }
        ],
        season=2022,
    )
    usage = parse_usage_payload(
        [
            {
                "season": 2021,
                "name": "A-B Player",
                "team": "Alpha",
                "usage": {"overall": 0.2},
            },
            {
                "season": 2021,
                "name": "A B Player",
                "team": "Alpha",
                "usage": {"overall": 0.0},
            },
        ],
        season=2021,
    )

    with pytest.raises(ValueError, match="conflicts with a positive usage row"):
        aggregate_team_features(
            records,
            usage,
            team_rows(),
            covered_seasons={2022},
            cutoff=date(2022, 8, 15),
            verified_zero_usage_keys={transfer_identity_key(records[0])},
        )


def test_portal_player_id_is_preferred_by_zero_evidence_identity() -> None:
    records = parse_transfer_payload(
        [
            {
                "season": 2022,
                "id": "player-1",
                "firstName": "A",
                "lastName": "Player",
                "origin": "Alpha",
                "destination": "Beta",
                "transferDate": "2022-07-01",
            },
            {
                "season": 2022,
                "id": "player-2",
                "firstName": "A",
                "lastName": "Player",
                "origin": "Alpha",
                "destination": "Gamma",
                "transferDate": "2022-07-01",
            },
        ],
        season=2022,
    )
    assert transfer_identity_key(records[0]) != transfer_identity_key(records[1])
    rows = [
        *team_rows(),
        {
            "season": "2022",
            "subdivision": "fbs",
            "team_id": "3",
            "team_name": "Gamma",
        },
    ]
    result = aggregate_team_features(
        records,
        [],
        rows,
        covered_seasons={2022},
        cutoff=date(2022, 8, 15),
        verified_zero_usage_keys={transfer_identity_key(records[0])},
    )
    assert result[(2022, "fbs", "2")]["transfer_in_prior_usage_sum"] == 0.0
    assert result[(2022, "fbs", "3")]["transfer_in_prior_usage_sum"] is None


def test_duplicate_fallback_identity_fails_closed() -> None:
    records = parse_transfer_payload(
        [
            {
                "season": 2022,
                "firstName": "A",
                "lastName": "Player",
                "origin": "Alpha",
                "destination": "Beta",
                "transferDate": "2022-07-01",
            },
            {
                "season": 2022,
                "firstName": "A",
                "lastName": "Player",
                "origin": "Alpha",
                "destination": "Beta",
                "transferDate": "2022-07-01",
            },
        ],
        season=2022,
    )
    with pytest.raises(ValueError, match="duplicate transfer identity key"):
        aggregate_team_features(
            records,
            [],
            team_rows(),
            covered_seasons={2022},
            cutoff=date(2022, 8, 15),
            verified_zero_usage_keys={transfer_identity_key(records[0])},
        )


def test_transfer_date_parser_rejects_invalid_text() -> None:
    with pytest.raises(ValueError):
        parse_transfer_date("not-a-date")


def test_portal_parser_retains_a_shared_identifier_when_the_source_provides_one() -> (
    None
):
    records = parse_transfer_payload(
        [
            {
                "season": 2022,
                "id": "shared-1",
                "firstName": "Player",
                "lastName": "One",
                "origin": "Alpha",
                "destination": "Beta",
                "transferDate": "2022-08-01",
            }
        ],
        season=2022,
    )
    assert records[0].player_id == "shared-1"
