import numpy as np

from gippyrank.modeling import coverage_table, design_matrix, fit_robust_surface


def test_depth_excludes_composite_and_reports_coverage() -> None:
    rows = [
        {"season": "2020", "subdivision": "fbs", "team_id": "1", "is_composite": "True", "system_code": "CMP", "ordinal_rank": "1"},
        {"season": "2020", "subdivision": "fbs", "team_id": "2", "is_composite": "True", "system_code": "CMP", "ordinal_rank": "2"},
        {"season": "2020", "subdivision": "fbs", "team_id": "1", "is_composite": "False", "system_code": "MAS", "ordinal_rank": "1"},
        {"season": "2020", "subdivision": "fbs", "team_id": "2", "is_composite": "False", "system_code": "MAS", "ordinal_rank": "2"},
        {"season": "2020", "subdivision": "fbs", "team_id": "1", "is_composite": "False", "system_code": "AP", "ordinal_rank": "1"},
    ]
    table = coverage_table(rows)
    assert {row["system_code"] for row in table} == {"MAS", "AP"}
    assert next(row for row in table if row["system_code"] == "MAS")["coverage_ratio"] == 1.0
    assert next(row for row in table if row["system_code"] == "AP")["coverage_ratio"] == 0.5


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
