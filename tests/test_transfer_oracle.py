from datetime import date

import pytest

from gippyrank.transfer_oracle import (
    aggregate_team_features,
    available_by_cutoff,
    parse_transfer_date,
    parse_transfer_payload,
    parse_usage_payload,
    position_group,
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


def test_transfer_date_parser_rejects_invalid_text() -> None:
    with pytest.raises(ValueError):
        parse_transfer_date("not-a-date")
