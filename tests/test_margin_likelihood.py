"""Semantic tests for the Issue #26 margin-likelihood investigation."""

from __future__ import annotations

import numpy as np
import pytest

from gippyrank.research.margin_likelihood import (
    BLOWOUT_KS,
    blowout_inverse,
    blowout_log_jacobian,
    blowout_transform,
    build_margin_data,
    fit_variable_scale,
    model_location,
    scale_design,
    scale_values,
    select_development_candidate,
    standardize_log_total_points,
    strict_comparison_key_audit,
    student_t_logpdf,
)

pytestmark = pytest.mark.research


def _row(
    *,
    home_subdivision: str = "fbs",
    away_subdivision: str = "fbs",
    home_points: int = 31,
    away_points: int = 17,
) -> dict[str, object]:
    return {
        "game_id": "g",
        "season": 2020,
        "home_subdivision": home_subdivision,
        "away_subdivision": away_subdivision,
        "home_team_population": 10,
        "away_team_population": 20,
        "home_points": home_points,
        "away_points": away_points,
        "neutral_site": "False",
        "rank_pairs": "[[2, 1], [3, 2]]",
    }


def test_fbs_fcs_orientation_is_used_for_margin_and_rank_coordinates() -> None:
    data = build_margin_data(
        [_row(home_subdivision="fcs", away_subdivision="fbs", home_points=10, away_points=24)]
    )
    assert np.all(data.margin == 14)
    # x is always FBS and y is always FCS for cross-subdivision games.
    assert data.x.tolist() == pytest.approx([(1 - 0.5) / 20, (2 - 0.5) / 20])
    assert data.y.tolist() == pytest.approx([(2 - 0.5) / 10, (3 - 0.5) / 10])


def test_scale_is_positive_and_mismatch_is_orientation_invariant() -> None:
    pairing = np.asarray(["fbs-fcs", "fbs-fcs"])
    first, names, _ = scale_design(pairing, np.asarray([14.0, -14.0]))
    values = scale_values(np.ones(first.shape[1]), first)
    assert names[-1] == "log1p_abs_v1_expected_margin"
    assert np.all(values > 0)
    assert first[0, -1] == pytest.approx(first[1, -1])


def test_total_points_context_changes_scale_design_but_not_location_surface() -> None:
    pairing = np.asarray(["fbs-fbs", "fbs-fbs"])
    expected = np.asarray([4.0, 4.0])
    a_features, _, _ = scale_design(pairing, expected)
    b_features, names, normalization = scale_design(
        pairing,
        expected,
        total_points=np.asarray([17.0, 123.0]),
        include_total_points=True,
    )
    assert names[-1] == "standardized_log1p_total_points"
    assert normalization is not None
    assert not np.allclose(b_features[:, -1], b_features[0, -1])
    beta = np.arange(3.0)
    X = np.column_stack([np.ones(2), np.zeros(2), np.ones(2)])
    assert np.array_equal(model_location({"beta": beta}, X), X @ beta)
    assert a_features.shape[1] + 1 == b_features.shape[1]


def test_total_points_normalization_is_fixed_from_training_values() -> None:
    _, fitted = standardize_log_total_points(np.asarray([17.0, 31.0, 45.0]))
    development, reused = standardize_log_total_points(
        np.asarray([123.0, 123.0]), mean=fitted["mean"], scale=fitted["scale"]
    )
    assert reused == fitted
    assert not np.isclose(float(np.mean(development)), 0.0)


def test_total_points_alone_cannot_create_rank_evidence() -> None:
    pairing = np.asarray(["fbs-fbs", "fbs-fbs"])
    features, _, _ = scale_design(
        pairing,
        np.asarray([0.0, 0.0]),
        total_points=np.asarray([17.0, 123.0]),
        include_total_points=True,
    )
    # No rank coordinate or mean-surface column is present in scale features.
    assert features.shape[1] == 5
    assert np.all(features[:, :4] == features[0, :4])


def test_variable_scale_can_hold_the_leakage_safe_v1_beta_fixed() -> None:
    design = np.eye(3)
    target = np.asarray([10.0, 2.0, -4.0])
    beta = np.asarray([9.0, 3.0, -5.0])
    fit = fit_variable_scale(
        design,
        target,
        np.ones(3),
        np.ones((3, 4)),
        fixed_beta=beta,
        maxiter=10,
    )
    assert np.array_equal(fit["beta"], beta)
    assert fit["mean_fit"] == "leakage_safe_v1_beta_fixed"


@pytest.mark.parametrize("k", BLOWOUT_KS)
def test_blowout_transform_is_symmetric_monotonic_and_invertible(k: float) -> None:
    values = np.asarray([-100.0, -14.0, 0.0, 14.0, 100.0])
    transformed = blowout_transform(values, k)
    assert np.all(np.diff(transformed) > 0)
    assert transformed[0] == pytest.approx(-transformed[-1])
    assert np.all(blowout_inverse(transformed, k) == pytest.approx(values))
    assert blowout_transform(np.asarray([1.0e-6]), k)[0] == pytest.approx(1.0e-6)


def test_blowout_jacobian_is_included_in_original_margin_density() -> None:
    k = 21.0
    margin = np.asarray([42.0])
    location = blowout_transform(margin, k)
    without_jacobian = student_t_logpdf(location, location, np.asarray([7.0]))
    with_jacobian = without_jacobian + blowout_log_jacobian(margin, k)
    assert with_jacobian[0] < without_jacobian[0]
    assert blowout_log_jacobian(np.asarray([0.0]), k)[0] == pytest.approx(0.0)


def test_strict_comparison_keys_require_identical_game_sets() -> None:
    same = strict_comparison_key_audit({"v1": ["a", "b"], "a": ["b", "a"]})
    different = strict_comparison_key_audit({"v1": ["a", "b"], "a": ["a"]})
    assert same["identical"] is True
    assert different["identical"] is False


def test_candidate_selection_is_development_only() -> None:
    def metrics(nll: float, coverage: float = 0.8) -> dict[str, float]:
        return {
            "marginalized_nll": nll,
            "expected_margin_mae": 10.0,
            "coverage_80": coverage,
        }

    aggregate = {
        "v1": metrics(4.0),
        "a": metrics(3.9),
        "b": metrics(3.8),
        "c14": metrics(3.7),
        "c21": metrics(3.7),
        "c28": metrics(3.7),
        "c42": metrics(3.7),
    }
    seasons = {
        name: [
            {"season": season, "marginalized_nll": nll}
            for season, nll in zip((2018, 2019, 2020, 2021), (nll, nll, nll, nll))
        ]
        for name, nll in (
            ("v1", 4.0),
            ("a", 3.9),
            ("b", 3.8),
            ("c14", 3.7),
            ("c21", 3.7),
            ("c28", 3.7),
            ("c42", 3.7),
        )
    }
    selected, trace = select_development_candidate(aggregate, seasons)
    assert selected == "c14"
    assert trace["selection_metric_period"] == "development_2018_2021_only"
    assert trace["final_2022_2025_used"] is False


def test_alternative_qualification_does_not_require_primary_nll_gain() -> None:
    def metrics(nll: float, coverage: float) -> dict[str, float]:
        return {
            "marginalized_nll": nll,
            "expected_margin_mae": 10.0,
            "coverage_80": coverage,
        }

    aggregate = {
        "v1": metrics(4.0, 0.75),
        "a": metrics(4.001, 0.8),
        "b": metrics(4.1, 0.8),
        "c14": metrics(4.1, 0.8),
        "c21": metrics(4.1, 0.8),
        "c28": metrics(4.1, 0.8),
        "c42": metrics(4.1, 0.8),
    }
    seasons = {
        name: [
            {"season": season, "marginalized_nll": nll}
            for season, nll in zip((2018, 2019, 2020, 2021), (nll, nll, nll, nll))
        ]
        for name, nll in ((name, values["marginalized_nll"]) for name, values in aggregate.items())
    }
    selected, trace = select_development_candidate(aggregate, seasons)
    assert selected == "a"
    assert trace["decisions"]["a"]["qualifies_vs_v1"] is True
    assert trace["decisions"]["a"]["complexity_gain_over_current"] == pytest.approx(-0.001)


def test_candidate_is_qualified_against_v1_before_complexity_comparison() -> None:
    def metrics(nll: float) -> dict[str, float]:
        return {
            "marginalized_nll": nll,
            "expected_margin_mae": 10.0,
            "coverage_80": 0.8,
        }

    aggregate = {
        "v1": metrics(4.0),
        "a": metrics(4.08),
        "b": metrics(3.994),
        "c14": metrics(4.2),
        "c21": metrics(4.2),
        "c28": metrics(4.2),
        "c42": metrics(4.2),
    }
    seasons = {
        "v1": [{"season": season, "marginalized_nll": 4.0} for season in (2018, 2019, 2020, 2021)],
        "a": [
            {"season": season, "marginalized_nll": value}
            for season, value in zip((2018, 2019, 2020, 2021), (3.98, 3.98, 4.18, 4.18))
        ],
        "b": [{"season": season, "marginalized_nll": 3.994} for season in (2018, 2019, 2020, 2021)],
        "c14": [{"season": season, "marginalized_nll": 4.2} for season in (2018, 2019, 2020, 2021)],
        "c21": [{"season": season, "marginalized_nll": 4.2} for season in (2018, 2019, 2020, 2021)],
        "c28": [{"season": season, "marginalized_nll": 4.2} for season in (2018, 2019, 2020, 2021)],
        "c42": [{"season": season, "marginalized_nll": 4.2} for season in (2018, 2019, 2020, 2021)],
    }
    selected, trace = select_development_candidate(aggregate, seasons)
    assert selected == "b"
    assert trace["decisions"]["a"]["qualifies_vs_v1"] is False
    assert trace["decisions"]["b"]["qualifies_vs_v1"] is True
