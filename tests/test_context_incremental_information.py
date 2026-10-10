from __future__ import annotations

import gzip

import numpy as np
import pytest

from gippyrank.research.context_incremental_information import (
    FeaturePreprocessor,
    ProbeExample,
    ProbeFit,
    ProbeInputs,
    ProbeObservation,
    apply_log_message,
    cross_message_arms,
    extract_log_message,
    fit_probe,
    normalized_pmf,
    predict_probe,
    probe_objective_and_gradient,
    select_regularization,
    validate_evidence_parity,
    write_deterministic_gzip_jsonl,
)


def example(
    season: int,
    team_id: str,
    *,
    feature: float | None,
    target: tuple[float, float, float] = (0.2, 0.5, 0.3),
) -> ProbeExample:
    return ProbeExample(
        season=season,
        team_id=team_id,
        team_name=f"Team {team_id}",
        history_posterior=np.asarray([0.3, 0.4, 0.3]),
        context_prior=np.asarray([0.25, 0.5, 0.25]),
        history_prior=np.asarray([0.3, 0.4, 0.3]),
        target=np.asarray(target),
        features={"x": feature},
    )


def test_pmf_normalization_support_and_message_identity() -> None:
    prior = normalized_pmf([0.4, 0.0, 0.6], name="prior")
    message = np.asarray([-0.5, -np.inf, 0.0])
    posterior = apply_log_message(prior, message)
    recovered, error = extract_log_message(prior, posterior)

    assert posterior[1] == 0.0
    assert posterior.sum() == pytest.approx(1.0)
    assert apply_log_message(prior, recovered) == pytest.approx(posterior)
    assert error < 1e-15
    with pytest.raises(ValueError, match="sum to one"):
        normalized_pmf([0.2, 0.2], name="invalid")
    with pytest.raises(ValueError, match="outside prior support"):
        extract_log_message([1.0, 0.0], [0.9, 0.1])


def test_crossed_arms_and_retained_evidence_parity() -> None:
    context_prior = np.asarray([0.2, 0.5, 0.3])
    history_prior = np.asarray([0.4, 0.4, 0.2])
    context_posterior = np.asarray([0.1, 0.6, 0.3])
    history_posterior = np.asarray([0.5, 0.3, 0.2])
    arms, errors = cross_message_arms(
        context_prior, history_prior, context_posterior, history_posterior
    )
    assert arms["CC"] == pytest.approx(context_posterior)
    assert arms["HH"] == pytest.approx(history_posterior)
    assert set(arms) == {"HH", "CH", "HC", "CC"}
    assert max(errors.values()) < 1e-15

    validate_evidence_parity(
        {"cutoff": "date", "ids_hash": "abc"},
        {"cutoff": "date", "ids_hash": "abc"},
        {"cutoff": "date", "ids_hash": "abc"},
        ("cutoff", "ids_hash"),
    )
    with pytest.raises(ValueError, match="replayed evidence differs"):
        validate_evidence_parity(
            {"cutoff": "date"}, {"cutoff": "date"}, {"cutoff": "other"}, ("cutoff",)
        )


def test_crossed_arms_reject_unequal_prior_support_without_games() -> None:
    context_prior = np.asarray([1.0, 0.0])
    history_prior = np.asarray([0.5, 0.5])

    with pytest.raises(ValueError, match="different positive support"):
        cross_message_arms(
            context_prior,
            history_prior,
            context_prior,
            history_prior,
        )


def test_zero_probe_is_history_identity() -> None:
    heldout = example(2024, "1", feature=7.0)
    preprocessor = FeaturePreprocessor.fit([{"x": 0.0}, {"x": 2.0}], ["x"])
    parameter_count = 2 + len(preprocessor.design_names)
    fitted = ProbeFit(
        include_signal=False,
        regularization=0.1,
        coefficients=np.zeros(parameter_count),
        coefficient_names=tuple(str(index) for index in range(parameter_count)),
        preprocessor=preprocessor,
        optimizer={},
        training_seasons=(2022, 2023),
    )

    assert predict_probe(heldout, fitted) == pytest.approx(
        normalized_pmf(heldout.history_posterior, name="History posterior")
    )


def test_preprocessing_and_regularization_use_training_seasons_only() -> None:
    training_rows = [{"x": 0.0}, {"x": 2.0}]
    preprocessor = FeaturePreprocessor.fit(training_rows, ["x"])
    heldout_extreme = {"x": 1_000_000.0}
    assert preprocessor.means["x"] == pytest.approx(1.0)
    assert preprocessor.scales["x"] == pytest.approx(1.0)
    assert preprocessor.transform([heldout_extreme])[0, 0] == pytest.approx(999_999.0)

    examples_by_season = {
        2022: [example(2022, "a", feature=0.0), example(2022, "b", feature=2.0)],
        2023: [
            example(2023, "a", feature=1.0, target=(0.1, 0.2, 0.7)),
            example(2023, "b", feature=3.0, target=(0.7, 0.2, 0.1)),
        ],
        2024: [example(2024, "a", feature=heldout_extreme["x"])],
    }
    selected_a, trace_a = select_regularization(
        examples_by_season, (2022, 2023), ("x",), include_signal=False
    )
    changed_heldout = dict(examples_by_season)
    changed_heldout[2024] = [
        example(2024, "a", feature=-1_000_000.0, target=(1.0, 0.0, 0.0))
    ]
    selected_b, trace_b = select_regularization(
        changed_heldout, (2022, 2023), ("x",), include_signal=False
    )
    assert selected_a == selected_b
    assert trace_a == trace_b
    assert trace_a["training_seasons"] == [2022, 2023]
    assert all(
        fold["validation_season"] == 2023 and fold["training_seasons"] == [2022]
        for candidate in trace_a["candidates"]
        for fold in candidate["folds"]
    )


def test_probe_gradient_matches_finite_differences() -> None:
    observation = ProbeObservation(
        inputs=ProbeInputs(
            history_posterior=np.asarray([0.2, 0.5, 0.3]),
            covariates=np.asarray([-0.4, 0.8]),
            context_signal=np.asarray([0.3, -0.5, 0.2]),
        ),
        target=np.asarray([0.4, 0.1, 0.5]),
    )
    coefficients = np.asarray([0.25, -0.1, 0.3, -0.2, 0.15])
    objective, analytic = probe_objective_and_gradient(
        coefficients,
        [observation],
        regularization=0.1,
        include_signal=True,
    )
    step = 1e-6
    numeric = np.empty_like(coefficients)
    for index in range(len(coefficients)):
        plus = coefficients.copy()
        minus = coefficients.copy()
        plus[index] += step
        minus[index] -= step
        plus_value = probe_objective_and_gradient(
            plus, [observation], regularization=0.1, include_signal=True
        )[0]
        minus_value = probe_objective_and_gradient(
            minus, [observation], regularization=0.1, include_signal=True
        )[0]
        numeric[index] = (plus_value - minus_value) / (2 * step)
    assert np.isfinite(objective)
    assert analytic == pytest.approx(numeric, abs=2e-7, rel=2e-6)


def test_cold_start_rows_keep_missing_features_and_still_predict() -> None:
    cold = example(2022, "cold", feature=None)
    fitted_row = example(2022, "fitted", feature=2.0)
    preprocessor = FeaturePreprocessor.fit([cold.features, fitted_row.features], ["x"])
    transformed = preprocessor.transform([cold.features])[0]
    assert transformed[-1] == 1.0
    assert preprocessor.medians["x"] == pytest.approx(2.0)

    fitted = fit_probe(
        [cold, fitted_row],
        ("x",),
        regularization=0.1,
        include_signal=False,
        training_seasons=(2022,),
    )
    prediction = predict_probe(cold, fitted)
    assert prediction.sum() == pytest.approx(1.0)
    assert np.all(prediction > 0)


def test_gzip_jsonl_is_deterministic(tmp_path) -> None:
    rows = [{"b": 2, "a": 1}, {"pmf": [0.25, 0.75]}]
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "nested/second.jsonl.gz"
    write_deterministic_gzip_jsonl(first, rows)
    write_deterministic_gzip_jsonl(second, rows)

    assert first.read_bytes() == second.read_bytes()
    assert gzip.decompress(first.read_bytes()) == (
        b'{"a":1,"b":2}\n{"pmf":[0.25,0.75]}\n'
    )
