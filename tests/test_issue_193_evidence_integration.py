from __future__ import annotations

import sys
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from gippyrank.research.issue_193_evidence_integration import (
    EvidenceInput,
    EvidenceStandardizer,
    RankTrainingExample,
    WeightedObservation,
    count_distinct_game_ids,
    fit_correction,
    fit_outer_fold,
    input_signal_diagnostics,
    objective_and_gradient,
    predict_rank_distribution,
    select_regularization,
    team_evidence_game_ids,
    team_season_sample_weights,
)

_runner_spec = spec_from_file_location(
    "issue193_runner",
    Path(__file__).resolve().parents[1]
    / "scripts/run_issue_193_evidence_integration.py",
)
if _runner_spec is None or _runner_spec.loader is None:
    raise ImportError("could not load the issue 193 runner for tests")
issue193_runner = module_from_spec(_runner_spec)
sys.modules[_runner_spec.name] = issue193_runner
_runner_spec.loader.exec_module(issue193_runner)


def evidence_input(
    season: int,
    team_id: str,
    checkpoint: str,
    *,
    evidence: tuple[str, ...] = (),
    history_posterior: tuple[float, ...] = (0.25, 0.5, 0.25),
    history_prior: tuple[float, ...] = (0.3, 0.4, 0.3),
    context_prior: tuple[float, ...] = (0.2, 0.55, 0.25),
) -> EvidenceInput:
    return EvidenceInput(
        season=season,
        team_id=team_id,
        checkpoint=checkpoint,
        history_posterior=np.asarray(history_posterior, dtype=float),
        context_prior=np.asarray(context_prior, dtype=float),
        history_prior=np.asarray(history_prior, dtype=float),
        team_game_ids=evidence,
    )


def training_row(
    season: int,
    team_id: str,
    checkpoint: str,
    *,
    target: tuple[float, ...] = (0.2, 0.5, 0.3),
    context_prior: tuple[float, ...] = (0.2, 0.55, 0.25),
    evidence: tuple[str, ...] = (),
) -> RankTrainingExample:
    return RankTrainingExample(
        inputs=evidence_input(
            season,
            team_id,
            checkpoint,
            evidence=evidence,
            context_prior=context_prior,
        ),
        target=np.asarray(target, dtype=float),
        team_name=f"Team {team_id}",
    )


def test_evidence_counts_are_distinct_and_game_order_invariant() -> None:
    assert count_distinct_game_ids(("g2", "g1", "g2")) == 2
    rows = [
        {"game_id": "g2", "home_id": "a", "away_id": "b"},
        {"game_id": "g1", "home_id": "b", "away_id": "c"},
    ]
    reversed_rows = list(reversed(rows))
    assert team_evidence_game_ids(rows) == team_evidence_game_ids(reversed_rows)
    assert team_evidence_game_ids(rows)["b"] == ("g1", "g2")
    with pytest.raises(ValueError, match="duplicate included evidence game ID"):
        team_evidence_game_ids([rows[0], rows[0]])
    with pytest.raises(ValueError, match="empty game ID"):
        count_distinct_game_ids(("",))


def test_each_training_team_season_has_total_weight_one() -> None:
    rows = [
        training_row(2022, "a", "c1"),
        training_row(2022, "a", "c2"),
        training_row(2022, "b", "c1"),
        training_row(2023, "a", "c1"),
        training_row(2023, "a", "c2"),
        training_row(2023, "a", "c3"),
    ]
    weights = team_season_sample_weights(rows)
    assert weights == pytest.approx([0.5, 0.5, 1.0, 1 / 3, 1 / 3, 1 / 3])
    totals: dict[tuple[int, str], float] = {}
    for row, weight in zip(rows, weights, strict=True):
        key = (row.inputs.season, row.inputs.team_id)
        totals[key] = totals.get(key, 0.0) + float(weight)
    assert all(value == pytest.approx(1.0) for value in totals.values())


def test_zero_context_signal_recovers_the_history_calibration_control() -> None:
    rows = [
        training_row(2022, "a", "c1", context_prior=(0.3, 0.4, 0.3)),
        training_row(
            2022,
            "a",
            "c2",
            target=(0.4, 0.2, 0.4),
            context_prior=(0.3, 0.4, 0.3),
            evidence=("g1", "g2"),
        ),
        training_row(
            2022,
            "b",
            "c1",
            target=(0.1, 0.7, 0.2),
            context_prior=(0.3, 0.4, 0.3),
            evidence=("g3",),
        ),
    ]
    history_fit = fit_correction(
        rows,
        regularization=0.1,
        include_context_signal=False,
        training_seasons=(2022,),
    )
    candidate_fit = fit_correction(
        rows,
        regularization=0.1,
        include_context_signal=True,
        training_seasons=(2022,),
    )
    heldout = evidence_input(
        2023,
        "x",
        "cutoff",
        evidence=("g9",),
        history_posterior=(0.15, 0.65, 0.2),
        history_prior=(0.3, 0.4, 0.3),
        context_prior=(0.3, 0.4, 0.3),
    )
    assert predict_rank_distribution(heldout, candidate_fit) == pytest.approx(
        predict_rank_distribution(heldout, history_fit), abs=1e-11
    )
    assert candidate_fit.coefficients[4:] == pytest.approx(np.zeros(3), abs=1e-9)


def test_equal_inputs_have_equal_predictions_and_permuted_game_ids() -> None:
    rows = [
        training_row(2022, "a", "c1", evidence=("g2", "g1")),
        training_row(2022, "b", "c1", target=(0.5, 0.2, 0.3), evidence=("g3",)),
    ]
    fitted = fit_correction(
        rows,
        regularization=0.1,
        include_context_signal=True,
        training_seasons=(2022,),
    )
    first = evidence_input(2023, "x", "c1", evidence=("g2", "g1"))
    reordered = replace(first, team_game_ids=("g1", "g2"))
    assert predict_rank_distribution(first, fitted) == pytest.approx(
        predict_rank_distribution(reordered, fitted), abs=1e-13
    )
    assert first.raw_evidence_features == pytest.approx(reordered.raw_evidence_features)


def test_heldout_targets_do_not_change_outer_fit_or_selection() -> None:
    examples_by_season = {
        2022: [training_row(2022, "a", "c1"), training_row(2022, "b", "c1")],
        2023: [
            training_row(2023, "a", "c1", target=(0.7, 0.2, 0.1)),
            training_row(2023, "b", "c1", target=(0.1, 0.2, 0.7)),
        ],
        2024: [training_row(2024, "a", "c1", target=(0.3, 0.3, 0.4))],
    }
    selected_before = select_regularization(
        examples_by_season, (2022, 2023), include_context_signal=True
    )
    original = examples_by_season[2024][0]
    examples_by_season[2024] = [replace(original, target=np.asarray((0.0, 1.0, 0.0)))]
    selected_after = select_regularization(
        examples_by_season, (2022, 2023), include_context_signal=True
    )
    assert selected_after == selected_before

    fit_before = fit_outer_fold(
        examples_by_season,
        (2022,),
        include_context_signal=True,
    )[0]
    examples_by_season[2024] = [replace(original, target=np.asarray((1.0, 0.0, 0.0)))]
    fit_after = fit_outer_fold(
        examples_by_season,
        (2022,),
        include_context_signal=True,
    )[0]
    assert fit_after.coefficients == pytest.approx(fit_before.coefficients, abs=1e-12)
    assert fit_after.standardizer == fit_before.standardizer


def test_optimizer_gradient_matches_finite_differences() -> None:
    rows = [
        training_row(2022, "a", "c1", evidence=("g1", "g2")),
        training_row(2022, "b", "c1", target=(0.5, 0.2, 0.3), evidence=("g3",)),
    ]
    weights = team_season_sample_weights(rows)
    standardizer = EvidenceStandardizer.fit([row.inputs for row in rows], weights)
    observations = [
        WeightedObservation(
            inputs=row.inputs,
            target=row.target,
            standardized_features=standardizer.transform(row.inputs),
            weight=float(weight),
        )
        for row, weight in zip(rows, weights, strict=True)
    ]
    theta = np.asarray([0.2, -0.1, 0.3, -0.2, 0.1, -0.25, 0.15])
    _objective, analytic = objective_and_gradient(
        theta,
        observations,
        regularization=0.1,
        include_context_signal=True,
    )
    numeric = np.empty_like(theta)
    step = 1e-6
    for index in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += step
        minus[index] -= step
        plus_value = objective_and_gradient(
            plus,
            observations,
            regularization=0.1,
            include_context_signal=True,
        )[0]
        minus_value = objective_and_gradient(
            minus,
            observations,
            regularization=0.1,
            include_context_signal=True,
        )[0]
        numeric[index] = (plus_value - minus_value) / (2 * step)
    assert analytic == pytest.approx(numeric, abs=2e-7, rel=2e-6)


def test_full_rank_normalization_and_explicit_support_policy() -> None:
    supported = training_row(
        2022,
        "a",
        "c1",
        target=(0.4, 0.6, 0.0),
    )
    supported = replace(
        supported,
        inputs=evidence_input(
            2022,
            "a",
            "c1",
            history_posterior=(0.4, 0.6, 0.0),
            history_prior=(0.5, 0.5, 0.0),
            context_prior=(0.3, 0.7, 0.0),
        ),
    )
    fitted = fit_correction(
        [supported, training_row(2022, "b", "c1")],
        regularization=0.1,
        include_context_signal=True,
        training_seasons=(2022,),
    )
    prediction = predict_rank_distribution(supported.inputs, fitted)
    assert prediction.shape == (3,)
    assert prediction.sum() == pytest.approx(1.0)
    assert prediction[2] == 0.0

    unequal_support = evidence_input(
        2022,
        "x",
        "c1",
        history_prior=(1.0, 0.0, 0.0),
        context_prior=(0.5, 0.5, 0.0),
    )
    with pytest.raises(ValueError, match="different positive support"):
        input_signal_diagnostics(unequal_support)

    target_outside = replace(
        supported,
        target=np.asarray((0.4, 0.5, 0.1)),
    )
    with pytest.raises(ValueError, match="outside History posterior support"):
        fit_correction(
            [target_outside],
            regularization=0.1,
            include_context_signal=True,
            training_seasons=(2022,),
        )


def test_evidence_standardization_uses_only_training_rows() -> None:
    rows = [
        training_row(2022, "a", "c1", evidence=("g1",)),
        training_row(2022, "b", "c1", evidence=("g1", "g2", "g3")),
    ]
    weights = team_season_sample_weights(rows)
    standardizer = EvidenceStandardizer.fit([row.inputs for row in rows], weights)
    assert np.all(np.isfinite(standardizer.transform(rows[0].inputs)))
    assert standardizer.scales[0] > 0
    assert standardizer.means[0] == pytest.approx(np.mean([np.log1p(1), np.log1p(3)]))


def test_runner_pairs_team_scores_by_season_team_and_model() -> None:
    rows = []
    for season in issue193_runner.EVALUATION_SEASONS:
        for model in issue193_runner.MODEL_NAMES:
            rows.append(
                {
                    "season": season,
                    "team_id": "17",
                    "team_name": "Example Team",
                    "model_arm": model,
                    "component_status": "fitted_prior",
                    "repaired_transfer_coverage": "available",
                    "mean_nll": (
                        1.0 if model == issue193_runner.CANDIDATE_MODEL else 1.5
                    ),
                    "mean_crps": 0.2,
                    "mean_expected_rank_error": 4.0,
                    "mean_interval_80_target_mass": 0.8,
                    "mean_interval_80_width": 20.0,
                }
            )

    differences = issue193_runner._team_rank_differences(rows)

    assert len(differences) == 2 * len(issue193_runner.EVALUATION_SEASONS)
    assert all(row["delta_nll"] == pytest.approx(-0.5) for row in differences)


def test_runner_summarizes_candidate_context_weights() -> None:
    rows = [
        {
            "model_arm": issue193_runner.CANDIDATE_MODEL,
            "context_weight": -0.25,
            "team_evidence_game_count": 3,
            "history_kl_from_prior": 0.4,
        }
    ]

    summary = issue193_runner._candidate_weight_summary(rows)

    assert summary[0]["mean_context_weight"] == pytest.approx(-0.25)
    assert summary[0]["mean_team_evidence_games"] == pytest.approx(3)


def _forecast_summary_row(
    population: str,
    game_id: str,
    *,
    crps: float,
    stage: str = "rolling",
) -> dict[str, object]:
    row: dict[str, object] = {
        "model_arm": "History_existing",
        "forecast_stage": stage,
        "forecast_population": population,
        "season": 2023,
        "phase": "pre_thanksgiving",
        "game_id": game_id,
    }
    summary_metrics = set(issue193_runner.GAME_SUPPORT_METRICS) | {
        "expected_home_margin",
        "median_home_margin",
        "home_win_probability",
        "margin_interval_50_low",
        "margin_interval_50_high",
        "margin_interval_80_low",
        "margin_interval_80_high",
        "margin_interval_95_low",
        "margin_interval_95_high",
    }
    row.update({metric: crps for metric in summary_metrics})
    return row


def test_runner_summarizes_forecast_populations_separately() -> None:
    rows = [
        _forecast_summary_row("fbs_fbs", "game-1", crps=1.0),
        _forecast_summary_row("fbs_fcs", "game-2", crps=3.0),
        _forecast_summary_row("fbs_fcs", "game-3", crps=5.0),
    ]

    summaries = issue193_runner._forecast_summaries(rows)
    by_population = {row["forecast_population"]: row for row in summaries}

    assert by_population["fbs_fbs"]["games"] == 1
    assert by_population["fbs_fbs"]["mean_crps"] == pytest.approx(1.0)
    assert by_population["fbs_fcs"]["games"] == 2
    assert by_population["fbs_fcs"]["mean_crps"] == pytest.approx(4.0)


def test_runner_counts_unique_games_once_across_model_arms() -> None:
    rows = [
        {
            "season": 2024,
            "phase": "regular",
            "forecast_stage": "rolling",
            "forecast_population": "fbs_fbs",
            "model_arm": model,
            "game_id": game_id,
        }
        for model in issue193_runner.MODEL_NAMES
        for game_id in ("army-navy-2024", "game-2")
    ]

    assert issue193_runner._season_game_counts(rows) == [
        {"season": 2024, "phase": "regular", "games": 2}
    ]


def test_runner_keeps_game_gain_concentration_cohorts_separate() -> None:
    rows = [
        {
            "comparator": "History_calibrated",
            "forecast_stage": "rolling",
            "forecast_population": "fbs_fbs",
            "delta_crps": -0.05,
        },
        {
            "comparator": "History_calibrated",
            "forecast_stage": "rolling",
            "forecast_population": "fbs_fbs",
            "delta_crps": -0.07,
        },
        {
            "comparator": "History_calibrated",
            "forecast_stage": "preseason",
            "forecast_population": "fbs_fbs",
            "delta_crps": 0.2,
        },
        {
            "comparator": "History_calibrated",
            "forecast_stage": "rolling",
            "forecast_population": "fbs_fcs",
            "delta_crps": -0.4,
        },
    ]

    concentration = issue193_runner._gain_concentration(rows, [])
    by_cohort = {
        (row["forecast_stage"], row["forecast_population"]): row
        for row in concentration
    }

    primary = by_cohort[("rolling", "fbs_fbs")]
    assert primary["rows"] == 2
    assert primary["mean_delta_candidate_minus_comparator"] == pytest.approx(-0.06)
    assert by_cohort[("preseason", "fbs_fbs")]["rows"] == 1
    assert by_cohort[("rolling", "fbs_fcs")]["rows"] == 1


def test_runner_rejects_heldout_targets_outside_prediction_support() -> None:
    with pytest.raises(ValueError, match="outside prediction support"):
        issue193_runner._make_rank_score_row(
            season=2023,
            cutoff="2023-10-01T00:00:00Z",
            checkpoint_index=1,
            checkpoint_phase="regular",
            checkpoint_has_games=True,
            team_id="team-1",
            team_name="Team 1",
            model_arm="History_calibrated_plus_Context",
            prior_family="History",
            prediction=np.asarray([0.4, 0.6, 0.0]),
            target=np.asarray([0.2, 0.3, 0.5]),
            inputs=evidence_input(2023, "team-1", "checkpoint"),
            target_row={},
            target_observation_date="2023-12-01",
            evidence={},
            strata={},
        )
