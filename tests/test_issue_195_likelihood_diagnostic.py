from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import t as student_t

from gippyrank.context_db_repair import MODEL_FEATURES
from gippyrank.modeling import design_matrix
from gippyrank.research.issue_195_likelihood_diagnostic import (
    CONTEXT_FEATURES,
    ContextPreprocessor,
    FitResult,
    LikelihoodData,
    LikelihoodSlice,
    _group_log_mixture,
    _locations_from_fit,
    _read_context_features,
    _read_issue201_games,
    _read_rank_pair_lookup,
    fit_context_preprocessor,
    identity_parameters,
    marginalized_nll_gradient,
    run_synthetic_controls,
    score_likelihood_slice,
    sha256_json,
    training_indices,
)


def _likelihood_slice() -> LikelihoodSlice:
    return LikelihoodSlice(
        margins=np.asarray([3.0, -4.0, 6.0]),
        home_field=np.asarray([1.0, 0.0, 1.0]),
        pair_game=np.asarray([0, 1, 1, 1, 2, 2]),
        base_locations=np.asarray([1.5, -3.0, -1.0, 1.0, 2.0, 4.0]),
        context=np.asarray([[0.2], [-0.5], [0.8]]),
    )


def _data() -> LikelihoodData:
    return LikelihoodData(
        game_ids=("1", "2", "3"),
        seasons=np.asarray([2022, 2023, 2024]),
        margins=np.asarray([3.0, -4.0, 6.0]),
        home_field=np.asarray([1.0, 0.0, 1.0]),
        pair_game=np.asarray([0, 1, 1, 1, 2, 2]),
        base_locations=np.asarray([1.5, -3.0, -1.0, 1.0, 2.0, 4.0]),
        context_home=np.asarray([[2.0], [1.0], [3.0]]),
        context_away=np.asarray([[0.0], [3.0], [1.0]]),
        metadata=({}, {}, {}),
    )


def test_marginalized_objective_weights_games_not_rank_pair_rows() -> None:
    view = _likelihood_slice()
    parameters = identity_parameters(1)
    objective, _ = marginalized_nll_gradient(
        parameters,
        view,
        baseline_scale=9.0,
        degrees_of_freedom=15.0,
        regularization=0.0,
    )
    expected_game_nll = []
    for game in range(len(view.margins)):
        pair_locations = view.base_locations[view.pair_game == game]
        log_density = student_t.logpdf(
            (view.margins[game] - pair_locations) / 9.0, 15.0
        ) - np.log(9.0)
        expected_game_nll.append(
            -(logsumexp(log_density) - np.log(len(pair_locations)))
        )
    assert objective == pytest.approx(sum(expected_game_nll))

    evaluation = score_likelihood_slice(
        view, fit=None, baseline_scale=9.0, degrees_of_freedom=15.0
    )
    assert evaluation.mean_nll == pytest.approx(np.mean(expected_game_nll))
    assert [row["rank_pair_count"] for row in evaluation.rows] == [1, 3, 2]


def test_mixture_integration_is_stable_for_extreme_log_densities() -> None:
    log_mix, responsibilities, counts = _group_log_mixture(
        np.asarray([-10001.0, -10003.0, -50000.0]),
        np.asarray([0, 0, 1]),
        2,
    )
    assert np.isfinite(log_mix).all()
    assert log_mix[0] == pytest.approx(-10001.56621917)
    assert log_mix[1] == pytest.approx(-50000.0)
    assert counts.tolist() == [2, 1]
    assert np.bincount(
        np.asarray([0, 0, 1]), weights=responsibilities, minlength=2
    ) == pytest.approx([1.0, 1.0])


def test_optimizer_analytic_gradient_matches_central_difference() -> None:
    view = _likelihood_slice()
    parameters = np.asarray([0.13, 1.25, -0.08, 0.4])
    value, gradient = marginalized_nll_gradient(
        parameters,
        view,
        baseline_scale=8.5,
        degrees_of_freedom=15.0,
        regularization=0.7,
    )
    assert np.isfinite(value)
    finite_difference = np.empty_like(parameters)
    step = 1e-6
    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step
        plus_value = marginalized_nll_gradient(
            plus,
            view,
            baseline_scale=8.5,
            degrees_of_freedom=15.0,
            regularization=0.7,
        )[0]
        minus_value = marginalized_nll_gradient(
            minus,
            view,
            baseline_scale=8.5,
            degrees_of_freedom=15.0,
            regularization=0.7,
        )[0]
        finite_difference[index] = (plus_value - minus_value) / (2 * step)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-5, atol=2e-5)


def test_identity_parameters_recover_frozen_baseline_exactly() -> None:
    view = _likelihood_slice()
    assert identity_parameters(3).tolist() == [0.0] * 6
    identity = FitResult(
        log_multiplier=0.0,
        home_field_adjustment=0.0,
        log_scale_correction=0.0,
        context_coefficients=np.zeros(1),
        scale=9.0,
        objective=0.0,
        optimizer_success=True,
        optimizer_status=0,
        optimizer_message="identity",
        optimizer_iterations=0,
        optimizer_evaluations=0,
        gradient_norm=0.0,
        regularization=0.0,
    )
    np.testing.assert_array_equal(
        _locations_from_fit(view, identity), view.base_locations
    )
    baseline = score_likelihood_slice(
        view, fit=None, baseline_scale=9.0, degrees_of_freedom=15.0
    )
    adjusted = score_likelihood_slice(
        view, fit=identity, baseline_scale=9.0, degrees_of_freedom=15.0
    )
    assert adjusted.mean_nll == pytest.approx(baseline.mean_nll, abs=1e-12)


def test_orientation_reversal_flips_rank_and_context_differences_but_keeps_home_term() -> (
    None
):
    preprocessor = ContextPreprocessor(
        feature_names=("talent",),
        means=np.asarray([0.0]),
        scales=np.asarray([1.0]),
        missing_means=np.asarray([0.0]),
        missing_scales=np.asarray([1.0]),
    )
    forward = preprocessor.transform(np.asarray([[2.0]]), np.asarray([[0.5]]))
    reverse = preprocessor.transform(np.asarray([[0.5]]), np.asarray([[2.0]]))
    np.testing.assert_array_equal(reverse, -forward)

    beta = np.zeros(34)
    beta[0] = 5.0
    beta[8] = 3.0
    forward_design = design_matrix(
        np.asarray([0.2]),
        np.asarray([0.8]),
        np.asarray(["fbs-fbs"]),
        np.asarray([1.0]),
        np.asarray([0.0]),
    )
    reverse_design = design_matrix(
        np.asarray([0.8]),
        np.asarray([0.2]),
        np.asarray(["fbs-fbs"]),
        np.asarray([1.0]),
        np.asarray([0.0]),
    )
    neutral_design = design_matrix(
        np.asarray([0.2]),
        np.asarray([0.8]),
        np.asarray(["fbs-fbs"]),
        np.asarray([0.0]),
        np.asarray([1.0]),
    )
    assert (forward_design @ beta).item() == pytest.approx(0.0)
    assert (reverse_design @ beta).item() == pytest.approx(6.0)
    assert (neutral_design @ beta).item() == pytest.approx(-3.0)


def test_neutral_site_suppresses_additional_home_term() -> None:
    view = LikelihoodSlice(
        margins=np.asarray([0.0, 0.0]),
        home_field=np.asarray([0.0, 1.0]),
        pair_game=np.asarray([0, 1]),
        base_locations=np.asarray([2.0, 2.0]),
        context=np.zeros((2, 0)),
    )
    fit = FitResult(
        log_multiplier=0.0,
        home_field_adjustment=4.0,
        log_scale_correction=0.0,
        context_coefficients=np.zeros(0),
        scale=10.0,
        objective=0.0,
        optimizer_success=True,
        optimizer_status=0,
        optimizer_message="fixture",
        optimizer_iterations=0,
        optimizer_evaluations=0,
        gradient_norm=0.0,
        regularization=0.0,
    )
    np.testing.assert_array_equal(_locations_from_fit(view, fit), [2.0, 6.0])


def test_preprocessing_uses_only_training_sides_and_emits_missingness_contrasts() -> (
    None
):
    home = np.asarray([[1.0], [3.0], [1000.0]])
    away = np.asarray([[0.0], [1.0], [0.0]])
    preprocessor = fit_context_preprocessor(home, away, np.asarray([0, 1]), ("talent",))
    assert preprocessor.means.tolist() == pytest.approx([1.25])
    assert preprocessor.scales[0] < 2.0

    with_missing = home.copy()
    with_missing[1, 0] = np.nan
    missing_preprocessor = fit_context_preprocessor(
        with_missing, away, np.asarray([0, 1]), ("talent",)
    )
    transformed = missing_preprocessor.transform(with_missing[:2], away[:2])
    assert transformed.shape == (2, 2)
    assert transformed[1, 1] != 0.0


def test_corrected_rank_pairs_must_match_target_population(tmp_path: Path) -> None:
    pair_file = tmp_path / "pairs.csv"
    fields = (
        "game_id",
        "season",
        "system_code",
        "home_team_id",
        "home_subdivision",
        "home_rank",
        "home_team_population",
        "home_rank_percentile",
        "away_team_id",
        "away_subdivision",
        "away_rank",
        "away_team_population",
        "away_rank_percentile",
    )
    with pair_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "game_id": "100",
                "season": "2023",
                "system_code": "SYS",
                "home_team_id": "1",
                "home_subdivision": "fbs",
                "home_rank": "1",
                "home_team_population": "4",
                "home_rank_percentile": "0.125",
                "away_team_id": "2",
                "away_subdivision": "fbs",
                "away_rank": "2",
                "away_team_population": "5",
                "away_rank_percentile": "0.3",
            }
        )
    targets = {
        (2023, "fbs", "1"): {"team_population": "4"},
        (2023, "fbs", "2"): {"team_population": "4"},
    }
    with pytest.raises(ValueError, match="target population mismatch"):
        _read_rank_pair_lookup(pair_file, {"100"}, targets)


def test_target_season_is_excluded_from_residual_training() -> None:
    data = _data()
    indices = training_indices(data, target_season=2024)
    assert indices.tolist() == [0, 1]
    assert np.all(data.seasons[indices] < 2024)


def test_context_feature_export_hashes_and_missing_values_are_preserved(
    tmp_path: Path,
) -> None:
    feature_path = tmp_path / "context_features.csv"
    manifest_path = tmp_path / "feature_manifest.json"
    values = {name: float(index) for index, name in enumerate(MODEL_FEATURES)}
    values[CONTEXT_FEATURES[0]] = None
    feature_hash = sha256_json(values)
    predictor_hash = "frozen-predictor-input-hash"
    row = {
        "season": 2023,
        "subdivision": "fbs",
        "team_id": "1",
        "team_name": "Team One",
        "predictor_input_sha256": predictor_hash,
        "context_feature_sha256": feature_hash,
        "repaired_transfer_feature_available": "True",
        **values,
        "features_json": json.dumps(values, sort_keys=True, separators=(",", ":")),
    }
    with feature_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    manifest_path.write_text(
        json.dumps(
            {
                "2023": {
                    "predictor_rows": [
                        {
                            "team_id": "1",
                            "predictor_input_sha256": predictor_hash,
                            "context_feature_sha256": feature_hash,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    indexed = _read_context_features(
        feature_path, manifest_path, target_seasons=(2023,)
    )

    assert np.isnan(indexed[(2023, "1")]["features"][CONTEXT_FEATURES[0]])
    assert indexed[(2023, "1")]["predictor_input_sha256"] == predictor_hash


def test_issue201_game_reader_uses_id_as_game_record_key(tmp_path: Path) -> None:
    score_path = tmp_path / "forecast_scores.csv"
    fields = (
        "forecast_stage",
        "forecast_population",
        "season",
        "game_id",
        "model_arm",
        "phase",
        "home_id",
        "away_id",
        "home_team",
        "away_team",
        "actual_home_points",
        "actual_away_points",
        "actual_home_margin",
        "neutral_site",
        "kickoff_utc",
        "conference_game",
    )
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm in (
            "History_existing",
            "Context_existing",
            "History_pre_bowl",
            "Context_pre_bowl",
        ):
            writer.writerow(
                {
                    "forecast_stage": "preseason",
                    "forecast_population": "fbs_fbs",
                    "season": "2022",
                    "game_id": "123",
                    "model_arm": arm,
                    "phase": "army_navy",
                    "home_id": "1",
                    "away_id": "2",
                    "home_team": "Home",
                    "away_team": "Away",
                    "actual_home_points": "21",
                    "actual_away_points": "17",
                    "actual_home_margin": "4",
                    "neutral_site": "True",
                    "kickoff_utc": "2022-12-10T17:00:00Z",
                    "conference_game": "False",
                }
            )

    games = _read_issue201_games(score_path)

    assert len(games) == 1
    assert games[0]["id"] == "123"
    assert games[0]["phase"] == "army_navy"
    assert games[0]["army_navy"] is True


def test_frozen_likelihood_synthetic_null_and_positive_controls_pass() -> None:
    root = Path(__file__).resolve().parents[1]
    baseline_path = (
        root
        / "data/research/matched_history_context_issue_200/historical_likelihood_existing.json"
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    controls = run_synthetic_controls(
        np.asarray(baseline["beta"], dtype=float),
        float(baseline["scale"]),
        float(baseline["degrees_of_freedom"]),
    )
    assert controls["status"] == "passed"
    assert controls["cases"]["null"]["status"] == "passed"
    assert controls["cases"]["positive_control"]["status"] == "passed"
    assert controls["cases"]["positive_control"]["fitted_context_effect_points"] > 0
    assert controls["cases"]["positive_control"]["context_minus_generic_nll"] < 0
