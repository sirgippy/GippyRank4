import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from gippyrank.modeling import (
    audit_population_coverage,
    build_constituent_rank_pairs,
    build_distributions,
    coverage_table,
    design_matrix,
    fit_marginalized,
    fit_robust_surface,
    game_log_scores,
    mixture_central_interval,
    validate_distribution_row,
)

_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "historical_build",
    Path(__file__).parents[1] / "scripts/build_historical_modeling.py",
)
_HISTORICAL_BUILD = importlib.util.module_from_spec(_SCRIPT_SPEC)
assert _SCRIPT_SPEC.loader is not None
sys.modules[_SCRIPT_SPEC.name] = _HISTORICAL_BUILD
_SCRIPT_SPEC.loader.exec_module(_HISTORICAL_BUILD)
pseudo_data = _HISTORICAL_BUILD.pseudo_data
build_joined = _HISTORICAL_BUILD.build_joined
evaluate = _HISTORICAL_BUILD.evaluate


def test_depth_excludes_composite_and_reports_coverage() -> None:
    rows = [
        {
            "season": "2020",
            "subdivision": "fbs",
            "team_id": "1",
            "is_composite": "True",
            "system_code": "CMP",
            "ordinal_rank": "1",
        },
        {
            "season": "2020",
            "subdivision": "fbs",
            "team_id": "2",
            "is_composite": "True",
            "system_code": "CMP",
            "ordinal_rank": "2",
        },
        {
            "season": "2020",
            "subdivision": "fbs",
            "team_id": "1",
            "is_composite": "False",
            "system_code": "MAS",
            "ordinal_rank": "1",
        },
        {
            "season": "2020",
            "subdivision": "fbs",
            "team_id": "2",
            "is_composite": "False",
            "system_code": "MAS",
            "ordinal_rank": "2",
        },
        {
            "season": "2020",
            "subdivision": "fbs",
            "team_id": "1",
            "is_composite": "False",
            "system_code": "AP",
            "ordinal_rank": "1",
        },
    ]
    table = coverage_table(rows)
    assert {row["system_code"] for row in table} == {"MAS", "AP"}
    assert (
        next(row for row in table if row["system_code"] == "MAS")["coverage_ratio"]
        == 1.0
    )
    assert (
        next(row for row in table if row["system_code"] == "AP")["coverage_ratio"]
        == 0.5
    )


@pytest.mark.parametrize(
    ("season", "subdivision", "old_population", "new_population", "outlier_ranks"),
    [
        (2004, "fcs", 119, 120, (120,)),
        (2007, "fbs", 119, 120, (120,)),
        (2020, "fcs", 125, 127, (126, 127)),
    ],
)
def test_issue_204_cohort_membership_repairs_rank_support_and_coverage(
    season: int,
    subdivision: str,
    old_population: int,
    new_population: int,
    outlier_ranks: tuple[int, ...],
) -> None:
    def make_rows(population: int) -> list[dict[str, str]]:
        rows = []
        for team_number in range(1, population + 1):
            team_id = str(team_number)
            rank = team_number
            if team_number == old_population and len(outlier_ranks) == 1:
                rank = outlier_ranks[0]
            if team_number == old_population - 1 and len(outlier_ranks) == 2:
                rank = outlier_ranks[0]
            if team_number == old_population and len(outlier_ranks) == 2:
                rank = outlier_ranks[1]
            if (
                len(outlier_ranks) == 1
                and team_number == new_population
                and population > old_population
            ):
                rank = old_population
            rows.append(
                {
                    "season": str(season),
                    "subdivision": subdivision,
                    "team_id": team_id,
                    "team_name": f"Team {team_id}",
                    "system_code": "CMP",
                    "ordinal_rank": str(rank),
                    "is_composite": "True",
                }
            )
            rows.append(
                {
                    "season": str(season),
                    "subdivision": subdivision,
                    "team_id": team_id,
                    "team_name": f"Team {team_id}",
                    "system_code": "SYS",
                    "ordinal_rank": str(rank),
                    "is_composite": "False",
                }
            )
        return rows

    old_rows = make_rows(old_population)
    old_depth = coverage_table(old_rows)
    with pytest.raises(ValueError, match="outside model population"):
        build_distributions(old_rows, old_depth, threshold=1.0)

    corrected_rows = make_rows(new_population)
    corrected_depth = coverage_table(corrected_rows)
    distributions, _ = build_distributions(corrected_rows, corrected_depth, 1.0)
    coverage = audit_population_coverage(
        corrected_rows, corrected_depth, distributions, 1.0
    )
    assert len(distributions) == new_population
    assert coverage[0]["teams_missing_selected_constituents"] == 0
    assert all(int(row["team_population"]) == new_population for row in distributions)


def test_selected_rank_observation_requires_cmp_membership() -> None:
    rows = [
        {
            "season": "2020",
            "subdivision": "fbs",
            "team_id": "1",
            "team_name": "CMP team",
            "system_code": "CMP",
            "ordinal_rank": "1",
            "is_composite": "True",
        },
        {
            "season": "2020",
            "subdivision": "fbs",
            "team_id": "2",
            "team_name": "Constituent-only team",
            "system_code": "SYS",
            "ordinal_rank": "1",
            "is_composite": "False",
        },
    ]
    with pytest.raises(ValueError, match="no CMP population member"):
        build_distributions(rows, coverage_table(rows), threshold=0.0)


def test_population_coverage_gate_flags_cmp_team_without_selected_rank() -> None:
    rows = [
        {
            "season": "2020",
            "subdivision": "fbs",
            "team_id": "1",
            "team_name": "First",
            "system_code": "CMP",
            "ordinal_rank": "1",
            "is_composite": "True",
        },
        {
            "season": "2020",
            "subdivision": "fbs",
            "team_id": "2",
            "team_name": "Second",
            "system_code": "CMP",
            "ordinal_rank": "2",
            "is_composite": "True",
        },
        {
            "season": "2020",
            "subdivision": "fbs",
            "team_id": "1",
            "team_name": "First",
            "system_code": "SYS",
            "ordinal_rank": "1",
            "is_composite": "False",
        },
    ]
    depth = coverage_table(rows)
    distributions, _ = build_distributions(rows, depth, threshold=0.5)

    coverage = audit_population_coverage(rows, depth, distributions, threshold=0.5)

    assert coverage[0]["teams_missing_selected_constituents"] == 1
    assert coverage[0]["missing_team_ids"] == '["2"]'


def test_constituent_rank_pairs_retain_system_identity_and_populations() -> None:
    by_team = {
        (2020, "fbs", "1"): [("SYS", 2)],
        (2020, "fcs", "2"): [("SYS", 3)],
    }
    depth = [
        {"season": 2020, "subdivision": "fbs", "team_population": 10},
        {"season": 2020, "subdivision": "fcs", "team_population": 20},
    ]
    games = [
        {
            "id": "9",
            "season": "2020",
            "homeId": "1",
            "homeClassification": "fbs",
            "awayId": "2",
            "awayClassification": "fcs",
        }
    ]

    pairs = build_constituent_rank_pairs(games, by_team, depth)

    assert len(pairs) == 1
    assert pairs[0]["system_code"] == "SYS"
    assert pairs[0]["home_team_population"] == 10
    assert pairs[0]["away_team_population"] == 20
    assert pairs[0]["home_rank_percentile"] == pytest.approx(0.15)
    assert pairs[0]["away_rank_percentile"] == pytest.approx(0.125)


def test_distribution_validation_requires_pmf_to_match_retained_ranks() -> None:
    row: dict[str, object] = {
        "season": 2020,
        "subdivision": "fbs",
        "team_id": "1",
        "team_name": "Example",
        "team_population": 2,
        "usable_systems": 2,
        "rank_observations": "[1,2]",
        "pmf": '[{"rank":1,"count":1,"probability":0.5},'
        '{"rank":2,"count":1,"probability":0.5}]',
    }
    validate_distribution_row(row)
    row["pmf"] = '[{"rank":1,"count":2,"probability":1.0}]'
    with pytest.raises(ValueError, match="support does not match"):
        validate_distribution_row(row)


def test_rank_population_change_leaves_unaffected_cohort_distributions_identical() -> None:
    unaffected = [
        {
            "season": "2021",
            "subdivision": "fbs",
            "team_id": team_id,
            "team_name": f"Team {team_id}",
            "system_code": system,
            "ordinal_rank": rank,
            "is_composite": is_composite,
        }
        for team_id, system, rank, is_composite in (
            ("1", "CMP", "1", "True"),
            ("2", "CMP", "2", "True"),
            ("1", "SYS", "1", "False"),
            ("2", "SYS", "2", "False"),
        )
    ]
    added_cohort = [
        {
            "season": "2020",
            "subdivision": "fcs",
            "team_id": str(team_number),
            "team_name": f"Added FCS team {team_number}",
            "system_code": system,
            "ordinal_rank": str(team_number),
            "is_composite": is_composite,
        }
        for team_number in range(1, 128)
        for system, is_composite in (("CMP", "True"), ("SYS", "False"))
    ]
    before_depth = coverage_table(unaffected)
    after_depth = coverage_table(unaffected + added_cohort)
    before, _ = build_distributions(unaffected, before_depth, threshold=1.0)
    after, _ = build_distributions(unaffected + added_cohort, after_depth, threshold=1.0)

    before_2021 = [row for row in before if row["season"] == 2021]
    after_2021 = [row for row in after if row["season"] == 2021]
    assert before_2021 == after_2021


def test_surface_has_pairing_specific_smooth_features() -> None:
    x = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0.2, 0.3, 0.7, 0.8])
    pairing = np.array(["fbs-fbs", "fbs-fcs", "fcs-fcs", "fbs-fcs"])
    matrix = design_matrix(x, y, pairing, np.ones(4), np.zeros(4))
    assert matrix.shape[0] == 4
    assert matrix.shape[1] > 20
    assert not np.allclose(matrix[0], matrix[1])


def test_robust_fit_returns_heavy_tailed_scale() -> None:
    x = np.linspace(0, 1, 20)
    X = np.column_stack([np.ones(20), x])
    fit = fit_robust_surface(X, 10 + 5 * x, np.ones(20))
    assert fit["df"] == 5.0
    assert fit["scale"] >= 1.0


def test_same_subdivision_neutral_swap_is_antisymmetric() -> None:
    pairing = np.array(["fbs-fbs", "fbs-fbs"])
    X = design_matrix(
        np.array([0.2, 0.8]), np.array([0.8, 0.2]), pairing, np.zeros(2), np.ones(2)
    )
    beta = np.arange(X.shape[1], dtype=float)
    assert np.allclose(X[0] @ beta, -(X[1] @ beta))


def test_equal_neutral_ranks_have_zero_margin_by_construction() -> None:
    X = design_matrix(
        np.array([0.5]), np.array([0.5]), np.array(["fcs-fcs"]), np.zeros(1), np.ones(1)
    )
    assert np.all(X[0] == 0)


def test_cross_subdivision_coordinates_are_fbs_then_fcs() -> None:
    # The fbs and fcs rows have identical stable coordinates despite opposite home teams.
    pairing = np.array(["fbs-fcs", "fbs-fcs"])
    first = design_matrix(
        np.array([0.2]),
        np.array([0.8]),
        pairing[:1],
        np.ones(1),
        np.zeros(1),
        fbs_home=np.ones(1),
    )[0]
    second = design_matrix(
        np.array([0.2]),
        np.array([0.8]),
        pairing[1:],
        np.ones(1),
        np.zeros(1),
        fbs_home=np.zeros(1),
    )[0]
    # Surface coordinates occupy the same cross block; only site-role columns differ.
    assert np.allclose(first[:11], second[:11])
    assert not np.allclose(first[11:], second[11:])


def test_game_scores_are_equal_weighted_and_marginalized_differs() -> None:
    target = np.array([0.0, 0.0, 0.0])
    locations = np.array([-4.0, 2.0, 0.0])
    groups = np.array([1, 1, 2])
    scores = game_log_scores(target, locations, groups, 1.0, 5.0)
    assert scores["n_games"] == 2
    assert scores["expected_conditional_nll"] != scores["marginalized_nll"]


def test_marginalized_fit_uses_game_groups() -> None:
    X = np.column_stack([np.ones(4), np.array([0.0, 1.0, 0.0, 1.0])])
    fit = fit_marginalized(
        X, np.array([0.0, 10.0, 0.0, 10.0]), np.array([1, 1, 2, 2]), maxiter=20
    )
    assert fit["objective"] == "marginalized"


def test_mixture_interval_reflects_rank_pair_uncertainty() -> None:
    lower, upper = mixture_central_interval(np.array([-20.0, 20.0]), 1.0, 5.0)
    single_lower = -1.0 * 2.015
    assert lower < single_lower
    assert upper > -single_lower


def test_pseudo_data_named_fields_and_site_roles() -> None:
    rows = [
        {
            "rank_pairs": "[[1,2],[2,3]]",
            "home_team_population": "10",
            "away_team_population": "20",
            "margin": "7",
            "pairing": "fbs-fcs",
            "home_subdivision": "fbs",
            "neutral_site": "False",
            "game_id": "11",
            "season": "2023",
        },
        {
            "rank_pairs": "[[1,2]]",
            "home_team_population": "10",
            "away_team_population": "20",
            "margin": "-4",
            "pairing": "fbs-fcs",
            "home_subdivision": "fcs",
            "neutral_site": "False",
            "game_id": "12",
            "season": "2024",
        },
        {
            "rank_pairs": "[[1,2]]",
            "home_team_population": "10",
            "away_team_population": "10",
            "margin": "3",
            "pairing": "fbs-fbs",
            "home_subdivision": "fbs",
            "neutral_site": "True",
            "game_id": "13",
            "season": "2022",
        },
    ]
    data = pseudo_data(rows)
    assert np.all(np.isin(data.fbs_home, [0.0, 1.0]))
    assert set(data.season) == {2022.0, 2023.0, 2024.0}
    assert data.fbs_home.tolist()[:2] == [1.0, 1.0]
    assert data.fbs_home.tolist()[2] == 0.0
    assert np.allclose(data.x[:2], [(1 - 0.5) / 10, (2 - 0.5) / 10])


def test_pseudo_data_rejects_out_of_population_rank_pairs() -> None:
    row = {
        "rank_pairs": "[[1,11]]",
        "home_team_population": "10",
        "away_team_population": "10",
        "margin": "7",
        "pairing": "fbs-fcs",
        "home_subdivision": "fbs",
        "neutral_site": "False",
        "game_id": "11",
        "season": "2023",
    }
    with pytest.raises(ValueError, match="away rank 11 is outside 1..10"):
        pseudo_data([row])


def test_evaluate_all_pairings_has_finite_subgroup_metrics() -> None:
    pairing = np.array(["fbs-fbs", "fbs-fcs", "fcs-fcs"])
    X = design_matrix(
        np.array([0.2, 0.2, 0.2]),
        np.array([0.4, 0.4, 0.4]),
        pairing,
        np.zeros(3),
        np.ones(3),
    )
    model = {"beta": np.zeros(X.shape[1]), "scale": 10.0, "df": 5.0}
    result = evaluate(model, X, np.array([1.0, 2.0, 3.0]), pairing, np.array([1, 2, 3]))
    assert result["n_games"] == 3
    assert all(np.isfinite(result[key]["marginalized_nll"]) for key in pairing)


def test_joined_cross_subdivision_response_is_fbs_minus_fcs() -> None:
    distributions = [
        {
            "season": "2023",
            "subdivision": "fbs",
            "team_id": "1",
            "team_population": "10",
            "rank_observations": "[1]",
        },
        {
            "season": "2023",
            "subdivision": "fcs",
            "team_id": "2",
            "team_population": "10",
            "rank_observations": "[2]",
        },
    ]
    base = {
        "season": "2023",
        "week": "1",
        "startDate": "",
        "homeId": "2",
        "awayId": "1",
        "homeClassification": "fcs",
        "awayClassification": "fbs",
        "homePoints": "10",
        "awayPoints": "24",
        "neutralSite": "False",
        "id": "99",
    }
    rows, _ = build_joined(
        [base],
        distributions,
        {(99, (2023, "fcs", "2"), (2023, "fbs", "1")): [(2, 1)]},
        [],
    )
    assert rows[0]["margin"] == 14
    assert rows[0]["pairing"] == "fbs-fcs"
