import csv
import sys
from pathlib import Path

import numpy as np

from gippyrank.preseason import TeamSeason

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import investigate_defensive_transfer_position_groups as study


def test_candidate_set_is_exactly_p0_through_p4() -> None:
    candidates = study.candidate_definitions()
    assert [candidate.label for candidate in candidates] == [
        "P0",
        "P1",
        "P2",
        "P3",
        "P4",
    ]
    assert candidates[0].features == study.d5_features()
    assert candidates[1].position_groups == ("dl_edge",)
    assert candidates[2].position_groups == ("lb",)
    assert candidates[3].position_groups == ("db",)
    assert candidates[4].position_groups == study.POSITION_GROUPS
    for candidate in candidates[1:]:
        for group in candidate.position_groups:
            assert study.POSITION_IMPACT_FEATURES[group] in candidate.features
            assert study.POSITION_AVAILABILITY_FEATURES[group] in candidate.features


def test_loader_distinguishes_no_incoming_from_unresolved(tmp_path: Path) -> None:
    audit_path = tmp_path / "transfer_player_audit.csv"
    audit_fields = [
        "season",
        "in_model_relevant_population",
        "defensive_candidate",
        "portal_position_group",
        "destination_team_id",
        "impact_status",
        "prior_defensive_impact",
    ]
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerow(
            {
                "season": 2022,
                "in_model_relevant_population": "True",
                "defensive_candidate": "True",
                "portal_position_group": "dl_edge",
                "destination_team_id": "1",
                "impact_status": "resolved",
                "prior_defensive_impact": "1.5",
            }
        )
        writer.writerow(
            {
                "season": 2022,
                "in_model_relevant_population": "True",
                "defensive_candidate": "True",
                "portal_position_group": "lb",
                "destination_team_id": "2",
                "impact_status": "source_data_unavailable",
                "prior_defensive_impact": "",
            }
        )
    team_path = tmp_path / "team_season_features.csv"
    team_fields = [
        "season",
        "subdivision",
        "team_id",
        "team_name",
        "feature_coverage_status",
        "transfer_in_prior_defensive_impact_sum",
    ]
    with team_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=team_fields)
        writer.writeheader()
        writer.writerow(
            {
                "season": 2022,
                "subdivision": "fbs",
                "team_id": "1",
                "team_name": "Team One",
                "feature_coverage_status": "complete",
                "transfer_in_prior_defensive_impact_sum": "1.5",
            }
        )
        writer.writerow(
            {
                "season": 2022,
                "subdivision": "fbs",
                "team_id": "2",
                "team_name": "Team Two",
                "feature_coverage_status": "partial",
                "transfer_in_prior_defensive_impact_sum": "",
            }
        )
    coverage_path = tmp_path / "coverage_by_position.csv"
    with coverage_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["season", "position_group", "experience_mass_coverage_proxy"],
        )
        writer.writeheader()
        for group in study.POSITION_GROUPS:
            writer.writerow(
                {
                    "season": 2022,
                    "position_group": group,
                    "experience_mass_coverage_proxy": "0.9",
                }
            )

    bundle = study.load_position_features(audit_path, coverage_path)
    team_one = bundle.values[(2022, "fbs", "1")]
    team_two = bundle.values[(2022, "fbs", "2")]
    assert team_one[study.DL_EDGE_IMPACT] == 1.5
    assert team_one[study.DL_EDGE_AVAILABLE] == 1.0
    assert team_one[study.LB_AVAILABLE] == 1.0
    assert team_two[study.LB_IMPACT] == 0.0
    assert team_two[study.LB_AVAILABLE] == 0.0
    assert team_two[study.DL_EDGE_AVAILABLE] == 1.0
    assert bundle.reconstruction_rows[0]["parity_passed"] is True
    assert bundle.reconstruction_rows[1]["parity_passed"] is None


def _row(season: int, team_id: str, value: float, available: float) -> TeamSeason:
    return TeamSeason(
        season,
        "fbs",
        team_id,
        team_id,
        2,
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([1]),
        {
            study.DL_EDGE_IMPACT: value,
            study.DL_EDGE_AVAILABLE: available,
            study.LB_IMPACT: value + 10.0,
            study.LB_AVAILABLE: available,
            study.DB_IMPACT: value + 20.0,
            study.DB_AVAILABLE: available,
        },
    )


def test_permutation_preserves_group_values_and_missingness() -> None:
    rows = [
        _row(2022, "a", 1.0, 1.0),
        _row(2022, "b", 2.0, 1.0),
        _row(2022, "c", 0.0, 0.0),
        _row(2023, "d", 3.0, 1.0),
    ]
    permuted = study.permute_position_features(rows, ("dl_edge",), seed=7)
    for season in (2022, 2023):
        original = [row for row in rows if row.season == season]
        actual = [row for row in permuted if row.season == season]
        assert [row.features[study.DL_EDGE_AVAILABLE] for row in actual] == [
            row.features[study.DL_EDGE_AVAILABLE] for row in original
        ]
        assert sorted(
            row.features[study.DL_EDGE_IMPACT]
            for row in actual
            if row.features[study.DL_EDGE_AVAILABLE] == 1.0
        ) == sorted(
            row.features[study.DL_EDGE_IMPACT]
            for row in original
            if row.features[study.DL_EDGE_AVAILABLE] == 1.0
        )
        assert [row.features[study.LB_IMPACT] for row in actual] == [
            row.features[study.LB_IMPACT] for row in original
        ]


def test_missingness_control_removes_only_numeric_group_value() -> None:
    candidate = study.candidate_definitions()[1]
    control = study.missingness_control(candidate)
    assert control.kind == "missingness_control"
    assert study.DL_EDGE_IMPACT not in control.features
    assert study.DL_EDGE_AVAILABLE in control.features
    assert control.features[: len(study.d5_features())] == study.d5_features()


def test_paired_diagnostics_labels_rolling_protocol_and_scopes() -> None:
    def prediction(season: int, model: str, probability: float):
        return study.v1_1.v1.PriorPrediction(
            season=season,
            subdivision="fbs",
            team_id=f"team-{season}",
            team_name=f"Team {season}",
            population=2,
            target_ranks=np.asarray([1]),
            model=model,
            prior_method="same_subdivision_lag1",
            pmf=np.asarray([probability, 1.0 - probability]),
        )

    details, summaries = study.paired_diagnostics(
        "P3_vs_P3-M",
        "P3",
        [prediction(2022, "P3", 0.8), prediction(2023, "P3", 0.8)],
        "P3-M",
        [prediction(2022, "P3-M", 0.6), prediction(2023, "P3-M", 0.6)],
        protocol="rolling_origin",
    )

    assert {row["protocol"] for row in details} == {"rolling_origin"}
    assert {row["scope"] for row in summaries} == {"aggregate", "2022", "2023"}
    assert all(row["mean_delta_nll"] < 0 for row in summaries)
