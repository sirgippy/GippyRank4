from datetime import date

import pytest

from gippyrank.transfer_audit import (
    D5_CATEGORY_FAILURE,
    D5_CATEGORY_RESOLVED,
    D5_CATEGORY_UNDETERMINED,
    D5_CATEGORY_ZERO,
    audit_transfer_records,
)
from gippyrank.transfer_oracle import parse_transfer_payload, parse_usage_payload


def _team_rows() -> list[dict[str, str]]:
    return [
        {
            "season": "2022",
            "subdivision": "fbs",
            "team_id": "1",
            "team_name": "Alpha",
            "talent_composite": "80",
        },
        {
            "season": "2022",
            "subdivision": "fbs",
            "team_id": "2",
            "team_name": "Beta",
            "talent_composite": "70",
        },
    ]


def test_audit_is_explicit_about_aliases_ambiguity_and_normalization() -> None:
    records = parse_transfer_payload(
        [
            {
                "season": 2022,
                "firstName": "A.",
                "lastName": "Player",
                "origin": "Old State",
                "destination": "Beta",
                "position": "QB",
                "transferDate": "2022-08-01",
                "rating": 0.9,
                "stars": 4,
            },
            {
                "season": 2022,
                "firstName": "B",
                "lastName": "Player",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "WR",
                "transferDate": "2022-08-01",
            },
            {
                "season": 2022,
                "firstName": "C",
                "lastName": "Player",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "WR",
                "transferDate": "2022-08-01",
            },
            {
                "season": 2022,
                "firstName": "E",
                "lastName": "Defender",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "CB",
                "transferDate": "2022-08-01",
            },
            {
                "season": 2022,
                "firstName": "F",
                "lastName": "Unknown",
                "origin": "Alpha",
                "destination": "Beta",
                "position": "ATH",
                "transferDate": "2022-08-01",
            },
            {
                "season": 2022,
                "firstName": "D",
                "lastName": "Player",
                "origin": "Alpha",
                "destination": None,
                "transferDate": "2022-08-01",
            },
        ],
        season=2022,
    )
    usage = parse_usage_payload(
        [
            {
                "season": 2021,
                "id": "1",
                "name": "A Player",
                "team": "Alpha",
                "usage": {"overall": 0.5},
            },
            {
                "season": 2021,
                "id": "2",
                "name": "B Player",
                "team": "Alpha",
                "usage": {"overall": 0.3},
            },
            {
                "season": 2021,
                "id": "3",
                "name": "B Player",
                "team": "Alpha",
                "usage": {"overall": 0.2},
            },
            {
                "season": 2021,
                "id": "4",
                "name": "C Player",
                "team": "Other State",
                "usage": {"overall": 0.8},
            },
        ],
        season=2021,
    )
    audit = audit_transfer_records(
        records,
        usage,
        _team_rows(),
        cutoff=date(2025, 8, 15),
        aliases={"Old State": "Alpha"},
    )

    by_player = {row["player_name"]: row for row in audit["join_rows"]}
    assert by_player["A. Player"]["usage_join_status"] == "joined"
    assert by_player["A. Player"]["usage_join_method"] == "explicit_team_alias"
    assert by_player["A. Player"]["normalization_changed_match"] is True
    assert by_player["A. Player"]["d5_resolution_category"] == D5_CATEGORY_RESOLVED
    assert by_player["B Player"]["usage_join_status"] == "ambiguous_usage_join"
    assert by_player["B Player"]["d5_resolution_category"] == D5_CATEGORY_FAILURE
    assert by_player["C Player"]["usage_join_status"] == "source_team_mismatch"
    assert by_player["C Player"]["d5_resolution_category"] == D5_CATEGORY_FAILURE
    assert by_player["E Defender"]["d5_resolution_category"] == D5_CATEGORY_ZERO
    assert by_player["F Unknown"]["d5_resolution_category"] == D5_CATEGORY_UNDETERMINED
    assert by_player["D Player"]["in_model_relevant_population"] is False

    season = audit["season_rows"][0]
    assert season["incoming_fbs_transfers"] == 5
    assert season["incoming_fbs_with_alias_successful_join"] == 1
    assert season["ambiguous_joins"] == 1
    assert season["d5_applicable_transfers"] == 3
    assert season["d5_successfully_resolved"] == 1
    assert season["d5_legitimate_zero_or_non_applicable"] == 1
    assert season["d5_resolution_failures"] == 2
    assert season["d5_applicability_unknown"] == 1
    assert season["d5_resolved_rate_among_determined"] == pytest.approx(0.5)
    assert season["d5_resolution_rate_among_applicable"] == pytest.approx(1 / 3)
    assert season["usage_weighted_join_coverage_proxy"] == pytest.approx(0.5 / 1.8)
    assert len(audit["unmatched_rows"]) == 3

    beta = next(row for row in audit["team_feature_rows"] if row["team_id"] == "2")
    assert beta["feature_coverage_status"] == "partial"
    assert beta["d5_legitimate_zero_or_non_applicable_count"] == 1
    assert beta["unmatched_incoming_count"] == 2
    assert beta["undetermined_incoming_count"] == 1
    assert beta["observed_incoming_prior_offensive_usage"] == pytest.approx(0.5)
    assert beta["unresolved_prior_offensive_usage_upper_bound"] == pytest.approx(2.1)
    assert beta["unmatched_prior_usage_upper_bound"] == pytest.approx(1.6)


def test_usage_parser_retains_identifier_for_stable_id_audit() -> None:
    usage = parse_usage_payload(
        [
            {
                "season": 2021,
                "id": "123",
                "name": "Player",
                "team": "Alpha",
                "conference": "TEST",
                "usage": {"overall": 0.1},
            }
        ],
        season=2021,
    )
    assert usage[0].player_id == "123"
    assert usage[0].conference == "TEST"
