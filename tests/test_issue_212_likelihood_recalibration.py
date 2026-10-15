from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from scipy.stats import t as student_t

from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    Team,
    game_factor,
    game_margin_parameters,
    infer_posterior,
)
from gippyrank.posterior.predictive import (
    ScheduledGame,
    predict_game,
    predictive_components,
)
from gippyrank.research.issue_195_likelihood_diagnostic import (
    FitResult,
    LikelihoodSlice,
    _locations_from_fit,
)
from gippyrank.research.issue_201_prebowl_forecast import (
    PRIMARY_POPULATION,
    summarize_forecast_scores,
)
from gippyrank.research.issue_212_likelihood_recalibration import (
    BASELINE_ARM,
    CANDIDATE_ARM,
    CalibrationParameters,
    RecalibratedLikelihood,
    _add_phase_pooled_summaries,
    _decision,
    _deduplicate_games,
    _rank_summary,
    _validate_target_exclusion,
    _verify_baseline_parity,
)


def _likelihood() -> LikelihoodV1:
    beta = np.zeros(34)
    beta[0] = 18.0
    beta[1] = -4.0
    beta[8] = 2.5
    beta[17] = -1.5
    beta[25] = 3.0
    return LikelihoodV1(beta, scale=10.0, degrees_of_freedom=15.0)


def _teams() -> list[Team]:
    return [
        Team("1", "Home", "fbs", np.asarray([0.15, 0.35, 0.5])),
        Team("2", "Away", "fbs", np.asarray([0.5, 0.3, 0.2])),
    ]


def _parameters(
    *, multiplier: float = 1.0, home: float = 0.0, log_scale: float = 0.0
) -> CalibrationParameters:
    return CalibrationParameters(2024, multiplier, home, log_scale, {})


def test_identity_adapter_recovers_factors_posteriors_and_predictive_components() -> (
    None
):
    likelihood = _likelihood()
    game = Game("g", "1", "2", "fbs", "fbs", 31, 17, False)
    teams = _teams()
    identity = RecalibratedLikelihood(likelihood, _parameters())

    baseline = infer_posterior(teams, [game], likelihood)
    adapted = infer_posterior(
        teams,
        [game],
        likelihood,
        factor_builder=identity.factor,
        factor_cache_key=identity.factor_cache_key,
    )

    np.testing.assert_array_equal(
        baseline.bp_state.factors[0].values, adapted.bp_state.factors[0].values
    )
    for team_id in baseline.pmfs:
        np.testing.assert_array_equal(baseline.pmfs[team_id], adapted.pmfs[team_id])

    scheduled = ScheduledGame("g", "1", "2", "fbs", "fbs")
    base_home = Team("1", "Home", "fbs", baseline.pmfs["1"])
    base_away = Team("2", "Away", "fbs", baseline.pmfs["2"])
    locations, weights = predictive_components(
        scheduled, base_home, base_away, likelihood
    )
    candidate_locations = identity.locations_for(locations, game)
    assert identity.scale_for(base_home, base_away) == likelihood.scale
    np.testing.assert_array_equal(candidate_locations, locations)
    np.testing.assert_array_equal(
        weights, predictive_components(scheduled, base_home, base_away, likelihood)[1]
    )
    assert predict_game(
        scheduled, base_home, base_away, likelihood
    ).expected_home_margin == pytest.approx(float(np.dot(weights, locations)))


def test_candidate_reconstructs_issue195_generic_conditional_distribution() -> None:
    likelihood = _likelihood()
    parameters = _parameters(multiplier=0.82, home=-1.7, log_scale=np.log(1.13))
    adapter = RecalibratedLikelihood(likelihood, parameters)
    neutral = Game("neutral", "1", "2", "fbs", "fbs", 28, 21, True)
    home = Game("home", "1", "2", "fbs", "fbs", 28, 21, False)
    locations = np.asarray([-4.0, 1.0, 6.0, 11.0])
    view = LikelihoodSlice(
        margins=np.asarray([7.0]),
        home_field=np.asarray([1.0]),
        pair_game=np.asarray([0, 0, 0, 0]),
        base_locations=locations,
        context=np.empty((1, 0)),
    )
    fit = FitResult(
        log_multiplier=np.log(parameters.multiplier),
        home_field_adjustment=parameters.home_adjustment,
        log_scale_correction=parameters.log_scale_correction,
        context_coefficients=np.empty(0),
        scale=likelihood.scale * parameters.scale_multiplier,
        objective=0.0,
        optimizer_success=True,
        optimizer_status=0,
        optimizer_message="test",
        optimizer_iterations=0,
        optimizer_evaluations=0,
        gradient_norm=0.0,
        regularization=0.0,
    )

    np.testing.assert_allclose(
        adapter.locations_for(locations, home), _locations_from_fit(view, fit)
    )
    np.testing.assert_allclose(
        adapter.locations_for(locations, neutral),
        parameters.multiplier * locations,
    )
    assert adapter.scale_for(_teams()[0], _teams()[1]) == pytest.approx(fit.scale)

    team_home, team_away = _teams()
    actual = adapter.factor(home, team_home, team_away)
    base_locations, margin = game_margin_parameters(
        home, team_home, team_away, likelihood
    )
    adjusted = parameters.multiplier * base_locations + parameters.home_adjustment
    raw = (
        student_t.pdf((margin - adjusted) / fit.scale, df=likelihood.degrees_of_freedom)
        / fit.scale
    )
    expected = raw / raw.max()
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0)


def test_cross_subdivision_factor_stays_exactly_unchanged() -> None:
    likelihood = _likelihood()
    fbs = Team("1", "FBS", "fbs", np.asarray([0.2, 0.3, 0.5]))
    fcs = Team("2", "FCS", "fcs", np.asarray([0.2, 0.3, 0.5]))
    game = Game("cross", "1", "2", "fbs", "fcs", 42, 14, False)
    adapter = RecalibratedLikelihood(
        likelihood, _parameters(multiplier=0.6, home=5.0, log_scale=-0.2)
    )
    np.testing.assert_array_equal(
        adapter.factor(game, fbs, fcs), game_factor(game, fbs, fcs, likelihood)
    )
    assert adapter.scale_for(fbs, fcs) == likelihood.scale


def test_candidate_cache_key_is_parameter_specific_and_inference_is_order_invariant() -> (
    None
):
    likelihood = _likelihood()
    teams = _teams()
    games = [
        Game("g1", "1", "2", "fbs", "fbs", 31, 17, False),
        Game("g2", "1", "2", "fbs", "fbs", 17, 20, True),
    ]
    first = RecalibratedLikelihood(likelihood, _parameters(multiplier=0.75))
    second = RecalibratedLikelihood(likelihood, _parameters(multiplier=1.2))
    assert first.factor_cache_key != second.factor_cache_key
    forward = infer_posterior(
        teams,
        games,
        likelihood,
        factor_builder=first.factor,
        factor_cache_key=first.factor_cache_key,
    )
    reversed_games = infer_posterior(
        teams,
        list(reversed(games)),
        likelihood,
        factor_builder=first.factor,
        factor_cache_key=first.factor_cache_key,
    )
    changed = infer_posterior(
        teams,
        games,
        likelihood,
        factor_builder=second.factor,
        factor_cache_key=second.factor_cache_key,
    )
    assert forward.converged and reversed_games.converged and changed.converged
    for team_id in forward.pmfs:
        np.testing.assert_allclose(
            forward.pmfs[team_id], reversed_games.pmfs[team_id], atol=1e-12
        )
    assert any(
        not np.allclose(forward.pmfs[team_id], changed.pmfs[team_id], atol=1e-8)
        for team_id in forward.pmfs
    )


def test_exact_duplicate_evidence_is_deduplicated_and_conflicts_fail() -> None:
    game = Game("g", "1", "2", "fbs", "fbs", 24, 17, False)
    assert _deduplicate_games([game, game]) == [game]
    with pytest.raises(ValueError, match="conflicting duplicate"):
        _deduplicate_games([game, Game("g", "1", "2", "fbs", "fbs", 30, 17, False)])


def test_target_game_must_be_absent_and_cutoff_precedes_kickoff() -> None:
    cutoff = datetime(2024, 9, 1, tzinfo=UTC)
    kickoff = cutoff + timedelta(hours=30)
    _validate_target_exclusion(
        game_id="target", cutoff=cutoff, kickoff=kickoff, evidence_ids={"older"}
    )
    with pytest.raises(ValueError, match="own evidence"):
        _validate_target_exclusion(
            game_id="target", cutoff=cutoff, kickoff=kickoff, evidence_ids={"target"}
        )
    with pytest.raises(ValueError, match="strictly before kickoff"):
        _validate_target_exclusion(
            game_id="target", cutoff=cutoff, kickoff=cutoff, evidence_ids=set()
        )


def test_baseline_parity_ignores_candidate_arm_for_same_forecast_key(
    tmp_path,
) -> None:
    metrics = (
        "crps",
        "margin_nll",
        "median_absolute_error",
        "home_win_brier",
        "expected_home_margin",
        "median_home_margin",
        "home_win_probability",
        "away_win_probability",
        "margin_interval_50_low",
        "margin_interval_50_high",
        "margin_interval_80_low",
        "margin_interval_80_high",
        "margin_interval_95_low",
        "margin_interval_95_high",
    )
    baseline = {
        "season": 2023,
        "game_id": "401520145",
        "cutoff_utc": "2023-08-22T00:00:00+00:00",
        "model_arm": BASELINE_ARM,
        "forecast_stage": "rolling",
        "forecast_population": PRIMARY_POPULATION,
        "home_id": "55",
        "away_id": "2638",
        "evidence_ids_sha256": "ids",
        "evidence_rows_sha256": "rows",
        **{metric: 0.0 for metric in metrics},
    }
    candidate = {**baseline, "model_arm": CANDIDATE_ARM, "crps": 1.0}
    reference_path = tmp_path / "issue201.csv"
    from gippyrank.research.issue_212_likelihood_recalibration import write_csv

    write_csv(reference_path, [baseline])

    parity = _verify_baseline_parity([baseline, candidate], reference_path)

    assert parity["passed"] is True
    assert parity["forecast_rows_compared"] == 1


def test_rank_aggregate_weights_each_team_season_equally_across_checkpoints() -> None:
    rows = []
    for cutoff, team_id, nll in (
        ("1", "a", 1.0),
        ("2", "a", 3.0),
        ("1", "b", 4.0),
    ):
        rows.append(
            {
                "season": 2024,
                "cutoff_utc": cutoff,
                "model_arm": BASELINE_ARM,
                "team_id": team_id,
                "nll": nll,
                "crps": nll,
                "expected_rank_error": nll,
                "interval_80_width": nll,
                "interval_80_target_mass": nll,
            }
        )
    summary = _rank_summary(rows)
    season = next(row for row in summary if row["aggregation"].startswith("season_"))
    assert season["team_seasons"] == 2
    assert season["nll"] == pytest.approx(3.0)


def _forecast(
    *, arm: str, season: int, stage: str, game_id: str, crps: float
) -> dict[str, object]:
    return {
        "forecast_stage": stage,
        "forecast_population": PRIMARY_POPULATION,
        "model_arm": arm,
        "season": season,
        "game_id": game_id,
        "phase": "pre_thanksgiving" if stage == "rolling" else "regular_season",
        "crps": crps,
        "margin_nll": 4.0,
        "median_absolute_error": 10.0,
        "home_win_brier": 0.2,
        "coverage_50": 0.5,
        "coverage_80": 0.8,
        "coverage_95": 0.95,
        "margin_interval_50_low": -10.0,
        "margin_interval_50_high": 10.0,
        "margin_interval_80_low": -20.0,
        "margin_interval_80_high": 20.0,
        "margin_interval_95_low": -30.0,
        "margin_interval_95_high": 30.0,
    }


def test_decision_uses_fixed_gain_and_guardrail_thresholds() -> None:
    rows = []
    for season in (2023, 2024, 2025):
        for game in range(3):
            for arm, crps in ((BASELINE_ARM, 10.0), (CANDIDATE_ARM, 9.9)):
                rows.append(
                    _forecast(
                        arm=arm,
                        season=season,
                        stage="rolling",
                        game_id=f"r{game}",
                        crps=crps,
                    )
                )
                rows.append(
                    _forecast(
                        arm=arm,
                        season=season,
                        stage="preseason",
                        game_id=f"p{game}",
                        crps=10.05,
                    )
                )
    summaries = summarize_forecast_scores(
        rows, stage="rolling", population=PRIMARY_POPULATION
    )
    summaries += summarize_forecast_scores(
        rows, stage="preseason", population=PRIMARY_POPULATION
    )
    decision = _decision(_add_phase_pooled_summaries(rows, summaries))
    assert decision["primary_rule"]["pass"] is True
    assert decision["early_guardrails_pass"] is True
    assert decision["recommend_independent_confirmation"] is True
