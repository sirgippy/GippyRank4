import numpy as np

from gippyrank.modeling import (
    coverage_table,
    design_matrix,
    fit_marginalized,
    fit_robust_surface,
    game_log_scores,
)


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
