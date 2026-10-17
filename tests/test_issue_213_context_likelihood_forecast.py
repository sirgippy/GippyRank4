from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import log

import numpy as np
import pytest
from scipy.stats import t as student_t

from gippyrank.posterior import engine as posterior_engine
from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    PosteriorResult,
    Team,
    game_factor,
    game_margin_parameters,
    infer_posterior,
)
from gippyrank.posterior.predictive import ScheduledGame, predictive_components
from gippyrank.research.issue_195_likelihood_diagnostic import (
    CONTEXT_FEATURES,
    ContextPreprocessor,
    FitResult,
    LikelihoodSlice,
    _locations_from_fit,
    sha256_json,
)
from gippyrank.research.issue_201_prebowl_forecast import (
    mixture_margin_crps,
    mixture_margin_nll,
)
from gippyrank.research.issue_213_context_likelihood_forecast import (
    BASELINE_ARM,
    CONTEXT_ARM,
    GENERIC_ARM,
    ContextLikelihoodAdapter,
    FrozenFit,
    _add_phase_aggregates,
    _deduplicate_games,
    _gain_concentration,
    _guardrail,
    _paired_forecasts,
    _posterior_rows,
    _predictive_mixture_rows,
    _predictive_model_parameters,
    _predictive_summary_with_shift,
    _rank_pmf_concentration,
    _validate_target_exclusion,
    _verify_baseline_checkpoint,
    _verify_baseline_parity,
    _write_csv,
)


def _base_likelihood() -> LikelihoodV1:
    beta = np.linspace(-1.5, 1.5, 34)
    return LikelihoodV1(beta=beta, scale=7.5, degrees_of_freedom=15.0)


def _fit(*, context_coefficients: np.ndarray | None = None) -> FrozenFit:
    feature_count = len(CONTEXT_FEATURES)
    coefficients = np.zeros(feature_count * 2)
    if context_coefficients is not None:
        coefficients = np.asarray(context_coefficients, dtype=float)
    preprocessor = ContextPreprocessor(
        feature_names=tuple(CONTEXT_FEATURES),
        means=np.zeros(feature_count),
        scales=np.full(feature_count, 2.0),
        missing_means=np.full(feature_count, 0.25),
        missing_scales=np.full(feature_count, 0.5),
    )
    return FrozenFit(
        season=2023,
        multiplier=1.4,
        home_adjustment=2.25,
        log_scale_correction=log(1.2),
        context_coefficients=coefficients,
        preprocessor=preprocessor,
        training_seasons=(2022,),
        context_penalty=0.1,
        source_record={
            "rank_site_calibration_fit": {
                "overall_mean_multiplier": 1.4,
                "additional_non_neutral_home_field_points": 2.25,
                "common_log_scale_correction": log(1.2),
            }
        },
    )


def _feature_row(team_id: str, values: list[float | None], digest: str) -> dict[str, object]:
    return {
        "features": dict(zip(CONTEXT_FEATURES, values, strict=True)),
        "context_feature_sha256": digest,
    }


def _feature_rows() -> dict[tuple[int, str], dict[str, object]]:
    home = [0.0] * len(CONTEXT_FEATURES)
    away = [0.0] * len(CONTEXT_FEATURES)
    home[0] = 2.0
    away[0] = -2.0
    home[1] = None
    away[1] = 1.0
    return {
        (2023, "home"): _feature_row("home", home, "home-features"),
        (2023, "away"): _feature_row("away", away, "away-features"),
    }


def _teams() -> tuple[Team, Team]:
    return (
        Team("home", "Home", "fbs", np.array([0.2, 0.3, 0.5])),
        Team("away", "Away", "fbs", np.array([0.5, 0.3, 0.2])),
    )


def _game(*, neutral: bool = False) -> Game:
    return Game("g-1", "home", "away", "fbs", "fbs", 31, 20, neutral)


def test_phase_aggregates_include_pooled_and_equal_season_summaries() -> None:
    summaries = [
        {
            "forecast_stage": "rolling",
            "forecast_population": "fbs_fbs",
            "model_arm": BASELINE_ARM,
            "season": season,
            "phase": "pre_thanksgiving",
            "games": games,
            "crps": crps,
            "coverage_50": coverage,
        }
        for season, games, crps, coverage in (
            ("2023", 10, 1.0, 0.4),
            ("2024", 30, 3.0, 0.6),
            ("2025", 20, 2.0, 0.5),
        )
    ]

    result = _add_phase_aggregates(summaries)
    pooled = next(
        row
        for row in result
        if row["season"] == "pooled_equal_game"
        and row["phase"] == "pre_thanksgiving"
    )
    equal_season = next(
        row
        for row in result
        if row["season"] == "equal_season_sensitivity"
        and row["phase"] == "pre_thanksgiving"
    )
    assert pooled["games"] == 60
    assert pooled["crps"] == pytest.approx((10 * 1 + 30 * 3 + 20 * 2) / 60)
    assert pooled["coverage_50"] == pytest.approx((10 * 0.4 + 30 * 0.6 + 20 * 0.5) / 60)
    assert equal_season["crps"] == pytest.approx(2.0)


def test_early_guardrail_uses_pooled_pre_thanksgiving_rows() -> None:
    summaries = []
    for arm in (BASELINE_ARM, GENERIC_ARM):
        for stage, phase in (
            ("preseason", "all_phases"),
            ("rolling", "pre_thanksgiving"),
        ):
            for season in ("2023", "2024", "2025"):
                summaries.append(
                    {
                        "forecast_stage": stage,
                        "forecast_population": "fbs_fbs",
                        "model_arm": arm,
                        "season": season,
                        "phase": phase,
                        "games": 10,
                        "crps": 1.0,
                        "coverage_50": 0.5,
                        "coverage_80": 0.8,
                        "coverage_95": 0.95,
                    }
                )
            summaries.append(
                {
                    "forecast_stage": stage,
                    "forecast_population": "fbs_fbs",
                    "model_arm": arm,
                    "season": "pooled_equal_game",
                    "phase": "all_phases",
                    "games": 30,
                    "crps": 1.0,
                    "coverage_50": 0.5,
                    "coverage_80": 0.8,
                    "coverage_95": 0.95,
                }
            )

    guardrail = _guardrail(_add_phase_aggregates(summaries), GENERIC_ARM)
    assert guardrail["passed"] is True
    assert guardrail["by_phase"]["pre_thanksgiving"]["pooled_crps_deterioration"] == 0.0


def test_joint_context_adapter_reconstructs_saved_location_and_scale_equation() -> None:
    base = _base_likelihood()
    fit = _fit()
    coefficients = np.zeros(len(CONTEXT_FEATURES) * 2)
    coefficients[0] = 0.7
    coefficients[len(CONTEXT_FEATURES) + 1] = -0.2
    fit = FrozenFit(**{**fit.__dict__, "context_coefficients": coefficients})
    adapter = ContextLikelihoodAdapter(base, fit, CONTEXT_ARM, _feature_rows())
    home, away = _teams()
    game = _game()

    delta, home_hash, away_hash = adapter.context_delta(game)
    assert (home_hash, away_hash) == ("home-features", "away-features")
    # First raw feature difference is 2.0 after scaling; the second Context
    # feature uses exact saved missingness indicators and training statistics.
    design = fit.preprocessor.transform(
        np.array([[2.0, np.nan, *([0.0] * (len(CONTEXT_FEATURES) - 2))]]),
        np.array([[-2.0, 1.0, *([0.0] * (len(CONTEXT_FEATURES) - 2))]]),
    )
    assert delta == pytest.approx(float(design[0] @ coefficients))

    corrected = adapter.likelihood_for(game, home, away)
    assert corrected.scale == pytest.approx(base.scale * 1.2)
    expected_locations, margin = game_margin_parameters(game, home, away, base)
    locations, corrected_margin = game_margin_parameters(game, home, away, corrected)
    assert corrected_margin == margin
    expected = fit.multiplier * expected_locations + fit.home_adjustment + delta
    assert np.allclose(locations + delta, expected, atol=1e-12)

    view = LikelihoodSlice(
        margins=np.array([margin]),
        home_field=np.array([1.0]),
        pair_game=np.zeros(expected_locations.size, dtype=int),
        base_locations=expected_locations.ravel(),
        context=design,
    )
    frozen_fit = FitResult(
        log_multiplier=log(fit.multiplier),
        home_field_adjustment=fit.home_adjustment,
        log_scale_correction=fit.log_scale_correction,
        context_coefficients=coefficients,
        scale=base.scale * np.exp(fit.log_scale_correction),
        objective=0.0,
        optimizer_success=True,
        optimizer_status=0,
        optimizer_message="synthetic saved fit",
        optimizer_iterations=0,
        optimizer_evaluations=0,
        gradient_norm=0.0,
        regularization=fit.context_penalty,
    )
    assert np.allclose(
        _locations_from_fit(view, frozen_fit).reshape(expected.shape), expected, atol=1e-12
    )

    factor = adapter.factor(game, home, away)
    expected_density = student_t.pdf(
        (margin - expected) / corrected.scale,
        corrected.degrees_of_freedom,
    )
    expected_density /= expected_density.max()
    assert np.allclose(factor, expected_density, atol=1e-14)


def test_home_adjustment_and_context_shift_respect_home_neutral_and_side_orientation() -> None:
    adapter = ContextLikelihoodAdapter(
        _base_likelihood(),
        _fit(context_coefficients=np.ones(len(CONTEXT_FEATURES) * 2)),
        CONTEXT_ARM,
        _feature_rows(),
    )
    home, away = _teams()
    neutral = _game(neutral=True)
    non_neutral = _game()
    delta, _, _ = adapter.context_delta(non_neutral)
    reversed_game = Game("g-2", "away", "home", "fbs", "fbs", 20, 31)
    reversed_delta, _, _ = adapter.context_delta(reversed_game)
    assert reversed_delta == pytest.approx(-delta)
    assert delta != 0.0

    neutral_likelihood = adapter.likelihood_for(neutral, home, away)
    home_index = 8
    assert neutral_likelihood.beta[home_index] == pytest.approx(
        _base_likelihood().beta[home_index] * adapter.fit.multiplier
        + adapter.fit.home_adjustment
    )
    # The non-neutral indicator is encoded by the game design, so the same
    # coefficient contributes nothing to a neutral site's location surface.
    neutral_locations, _ = game_margin_parameters(
        neutral, home, away, neutral_likelihood
    )
    base_neutral, _ = game_margin_parameters(neutral, home, away, _base_likelihood())
    expected_neutral, _ = game_margin_parameters(
        neutral,
        home,
        away,
        LikelihoodV1(
            beta=_base_likelihood().beta * np.r_[np.full(9, adapter.fit.multiplier), np.ones(25)],
            scale=_base_likelihood().scale * adapter.scale_multiplier,
            degrees_of_freedom=_base_likelihood().degrees_of_freedom,
        ),
    )
    assert np.allclose(neutral_locations, expected_neutral, atol=1e-12)
    assert not np.allclose(neutral_locations, base_neutral)


def test_context_feature_alignment_requires_exact_evaluation_team_rows() -> None:
    features = _feature_rows()
    adapter = ContextLikelihoodAdapter(
        _base_likelihood(),
        _fit(),
        CONTEXT_ARM,
        {(2023, "away"): features[(2023, "away")]},
    )
    with pytest.raises(ValueError, match="exact #200 Context feature row is missing"):
        adapter.context_delta(_game())


def test_generic_and_zero_context_joint_fit_are_identical_for_factors_and_predictions() -> None:
    base = _base_likelihood()
    fit = _fit()
    rows = _feature_rows()
    generic = ContextLikelihoodAdapter(base, fit, GENERIC_ARM, rows)
    context = ContextLikelihoodAdapter(base, fit, CONTEXT_ARM, rows)
    home, away = _teams()
    game = _game(neutral=True)
    assert np.array_equal(generic.factor(game, home, away), context.factor(game, home, away))

    schedule = ScheduledGame(
        game_id=game.game_id,
        home_id=game.home_id,
        away_id=game.away_id,
        home_subdivision="fbs",
        away_subdivision="fbs",
        neutral_site=True,
        season_type="regular",
        date="2023-09-01T00:00:00Z",
        schedule_status="scheduled",
    )
    generic_locations, generic_weights, generic_likelihood = generic.predictive_components(
        schedule, game, home, away
    )
    context_locations, context_weights, context_likelihood = context.predictive_components(
        schedule, game, home, away
    )
    assert np.array_equal(generic_locations, context_locations)
    assert np.array_equal(generic_weights, context_weights)
    assert np.array_equal(generic_likelihood.beta, context_likelihood.beta)
    assert generic_likelihood.scale == context_likelihood.scale


def test_identity_fit_recovers_the_base_factor_and_predictive_mixture() -> None:
    base = _base_likelihood()
    identity_record = _fit()
    generic = {
        "overall_mean_multiplier": 1.0,
        "additional_non_neutral_home_field_points": 0.0,
        "common_log_scale_correction": 0.0,
    }
    identity = FrozenFit(
        **{
            **identity_record.__dict__,
            "multiplier": 1.0,
            "home_adjustment": 0.0,
            "log_scale_correction": 0.0,
            "source_record": {"rank_site_calibration_fit": generic},
        }
    )
    adapter = ContextLikelihoodAdapter(base, identity, GENERIC_ARM, _feature_rows())
    home, away = _teams()
    game = _game()
    assert np.allclose(
        adapter.factor(game, home, away), game_factor(game, home, away, base), atol=1e-15
    )
    schedule = ScheduledGame(
        game_id=game.game_id,
        home_id=game.home_id,
        away_id=game.away_id,
        home_subdivision="fbs",
        away_subdivision="fbs",
        neutral_site=False,
        season_type="regular",
        date="2023-09-01T00:00:00Z",
        schedule_status="scheduled",
    )
    locations, weights, likelihood = adapter.predictive_components(
        schedule, game, home, away
    )
    base_locations, base_weights = predictive_components(schedule, home, away, base)
    assert np.array_equal(locations, base_locations)
    assert np.array_equal(weights, base_weights)
    assert np.array_equal(likelihood.beta, base.beta)
    assert likelihood.scale == base.scale


def test_predictive_components_apply_saved_context_shift_to_every_rank_pair() -> None:
    coefficients = np.zeros(len(CONTEXT_FEATURES) * 2)
    coefficients[0] = 1.0
    adapter = ContextLikelihoodAdapter(
        _base_likelihood(), _fit(context_coefficients=coefficients), CONTEXT_ARM, _feature_rows()
    )
    home, away = _teams()
    game = _game()
    schedule = ScheduledGame(
        game_id=game.game_id,
        home_id=game.home_id,
        away_id=game.away_id,
        home_subdivision="fbs",
        away_subdivision="fbs",
        neutral_site=False,
        season_type="regular",
        date="2023-09-01T00:00:00Z",
        schedule_status="scheduled",
    )
    locations, weights, likelihood = adapter.predictive_components(
        schedule, game, home, away
    )
    score_blind_locations, score_blind_weights, _ = adapter.predictive_components(
        schedule,
        Game(game.game_id, game.home_id, game.away_id, "fbs", "fbs", 0, 0),
        home,
        away,
    )
    base_likelihood = adapter.likelihood_for(game, home, away)
    base_locations, base_weights = predictive_components(
        schedule, home, away, base_likelihood
    )
    delta, _, _ = adapter.context_delta(game)
    assert np.allclose(locations, base_locations + delta, atol=1e-12)
    assert np.array_equal(weights, base_weights)
    assert np.array_equal(locations, score_blind_locations)
    assert np.array_equal(weights, score_blind_weights)
    assert likelihood.scale == pytest.approx(_base_likelihood().scale * 1.2)


def test_margin_scores_are_invariant_to_centering_both_mixture_and_outcome() -> None:
    locations = np.array([-12.0, -3.0, 4.0, 17.0])
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    observation = 5.5
    shift = -8.25
    original_crps = mixture_margin_crps(locations, weights, observation, 7.0, 15.0)
    centered_crps = mixture_margin_crps(
        locations + shift, weights, observation + shift, 7.0, 15.0
    )
    assert centered_crps[0] == pytest.approx(original_crps[0], abs=2e-10)
    assert centered_crps[1] == pytest.approx(original_crps[1], abs=2e-10)
    assert centered_crps[2:] == original_crps[2:]
    assert mixture_margin_nll(locations, weights, observation, 7.0, 15.0) == pytest.approx(
        mixture_margin_nll(
            locations + shift, weights, observation + shift, 7.0, 15.0
        ),
        abs=2e-12,
    )


def test_shifted_predictive_summary_matches_direct_mixture_summary() -> None:
    from gippyrank.research import issue_201_prebowl_forecast as issue201

    locations = np.array([-12.0, -3.0, 4.0, 17.0])
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    shift = 8.25
    likelihood = _base_likelihood()
    direct = issue201._predictive_summary_from_components(
        locations + shift, weights, likelihood
    )
    translated = _predictive_summary_with_shift(
        locations, weights, likelihood, shift
    )
    assert translated.expected_home_margin == pytest.approx(
        direct.expected_home_margin, abs=1e-10
    )
    assert translated.median_home_margin == pytest.approx(
        direct.median_home_margin, abs=1e-10
    )
    assert translated.home_win_probability == pytest.approx(
        direct.home_win_probability, abs=1e-12
    )
    assert translated.away_win_probability == pytest.approx(
        direct.away_win_probability, abs=1e-12
    )
    for level in (50, 80, 95):
        assert getattr(translated, f"margin_interval_{level}") == pytest.approx(
            getattr(direct, f"margin_interval_{level}"), abs=1e-10
        )


def test_cross_subdivision_factors_and_predictions_are_unchanged() -> None:
    base = _base_likelihood()
    adapter = ContextLikelihoodAdapter(base, _fit(), CONTEXT_ARM, {})
    fbs = Team("fbs", "FBS", "fbs", np.array([0.3, 0.7]))
    fcs = Team("fcs", "FCS", "fcs", np.array([0.6, 0.4]))
    game = Game("cross", "fbs", "fcs", "fbs", "fcs", 28, 17)
    assert adapter.likelihood_for(game, fbs, fcs) is base
    assert np.array_equal(adapter.factor(game, fbs, fcs), game_factor(game, fbs, fcs, base))
    schedule = ScheduledGame(
        game_id="cross",
        home_id="fbs",
        away_id="fcs",
        home_subdivision="fbs",
        away_subdivision="fcs",
        neutral_site=False,
        season_type="regular",
        date="2023-09-01T00:00:00Z",
        schedule_status="scheduled",
    )
    expected = predictive_components(schedule, fbs, fcs, base)
    actual = adapter.predictive_components(schedule, game, fbs, fcs)
    assert actual[2] is base
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])


def test_factor_cache_key_changes_with_arm_and_context_rows() -> None:
    base = _base_likelihood()
    fit = _fit()
    home, away = _teams()
    game = _game()
    rows = _feature_rows()
    generic = ContextLikelihoodAdapter(base, fit, GENERIC_ARM, rows)
    context = ContextLikelihoodAdapter(base, fit, CONTEXT_ARM, rows)
    assert generic.factor_cache_key(game, home, away) != context.factor_cache_key(
        game, home, away
    )
    changed = {key: dict(row) for key, row in rows.items()}
    changed_home = dict(changed[(2023, "home")])
    changed_home["context_feature_sha256"] = "changed"
    changed[(2023, "home")] = changed_home
    changed_context = ContextLikelihoodAdapter(base, fit, CONTEXT_ARM, changed)
    assert context.factor_cache_key(game, home, away) != changed_context.factor_cache_key(
        game, home, away
    )


def test_custom_factor_cache_contract_is_required_and_uses_adapter_identity() -> None:
    base = _base_likelihood()
    fit = _fit()
    adapter = ContextLikelihoodAdapter(base, fit, GENERIC_ARM, {})
    home, away = _teams()
    game = _game()
    with pytest.raises(ValueError, match="distinct factor_cache_key"):
        infer_posterior([home, away], [game], base, factor_builder=adapter.factor)
    with pytest.raises(ValueError, match="requires a custom factor_builder"):
        infer_posterior([home, away], [game], base, factor_cache_key="missing-builder")
    result = infer_posterior(
        [home, away],
        [game],
        base,
        factor_builder=adapter.factor,
        factor_cache_key=adapter.factor_cache_key,
    )
    assert result.converged
    assert all(np.isclose(pmf.sum(), 1.0) for pmf in result.pmfs.values())


def test_candidate_inference_is_invariant_to_game_order() -> None:
    base = _base_likelihood()
    fit = _fit()
    rows = _feature_rows()
    adapter = ContextLikelihoodAdapter(base, fit, CONTEXT_ARM, rows)
    home, away = _teams()
    extra = Team("third", "Third", "fbs", np.array([0.4, 0.4, 0.2]))
    games = [
        _game(),
        Game("g-2", "away", "third", "fbs", "fbs", 28, 17, True),
        Game("g-3", "third", "home", "fbs", "fbs", 14, 21),
    ]
    with pytest.raises(ValueError, match="exact #200 Context feature row is missing"):
        adapter.factor(games[1], away, extra)
    adapter.feature_rows[(2023, "third")] = _feature_row(
        "third", [0.0] * len(CONTEXT_FEATURES), "third-features"
    )
    forward = infer_posterior(
        [home, away, extra],
        games,
        base,
        factor_builder=adapter.factor,
        factor_cache_key=adapter.factor_cache_key,
    )
    reverse = infer_posterior(
        [home, away, extra],
        list(reversed(games)),
        base,
        factor_builder=adapter.factor,
        factor_cache_key=adapter.factor_cache_key,
    )
    assert forward.converged and reverse.converged
    for team_id in forward.pmfs:
        assert np.array_equal(forward.pmfs[team_id], reverse.pmfs[team_id])


def test_context_factor_cache_uses_each_games_context_correction() -> None:
    coefficients = np.zeros(len(CONTEXT_FEATURES) * 2)
    coefficients[0] = 1.0
    adapter = ContextLikelihoodAdapter(
        _base_likelihood(),
        _fit(context_coefficients=coefficients),
        CONTEXT_ARM,
        {
            (2023, team_id): _feature_row(
                team_id,
                [feature_value, *([0.0] * (len(CONTEXT_FEATURES) - 1))],
                f"{team_id}-features",
            )
            for team_id, feature_value in (
                ("a", 4.0),
                ("b", 0.0),
                ("c", 0.0),
                ("d", 0.0),
            )
        },
    )
    teams = [
        Team(team_id, team_id, "fbs", np.full(5, 0.2))
        for team_id in ("a", "b", "c", "d")
    ]
    games = [
        Game("g-ab", "a", "b", "fbs", "fbs", 28, 21),
        Game("g-cd", "c", "d", "fbs", "fbs", 28, 21),
    ]
    assert adapter.context_delta(games[0])[0] != adapter.context_delta(games[1])[0]
    assert not np.array_equal(
        adapter.factor(games[0], teams[0], teams[1]),
        adapter.factor(games[1], teams[2], teams[3]),
    )

    def infer_in_order(ordered_games: list[Game]) -> PosteriorResult:
        posterior_engine._GAME_FACTOR_CACHE.clear()
        return infer_posterior(
            teams,
            ordered_games,
            _base_likelihood(),
            factor_builder=adapter.factor,
            factor_cache_key=adapter.factor_cache_key,
        )

    forward = infer_in_order(games)
    reverse = infer_in_order(list(reversed(games)))
    assert forward.converged and reverse.converged
    for team_id in forward.pmfs:
        assert np.array_equal(forward.pmfs[team_id], reverse.pmfs[team_id])


def test_duplicate_evidence_is_deduplicated_but_conflicts_fail() -> None:
    game = _game()
    assert _deduplicate_games([game, game]) == [game]
    conflicting = Game(game.game_id, game.home_id, game.away_id, "fbs", "fbs", 30, 20)
    with pytest.raises(ValueError, match="conflicting duplicate evidence"):
        _deduplicate_games([game, conflicting])


def test_target_game_must_be_future_and_absent_from_evidence() -> None:
    cutoff = datetime(2023, 9, 1, tzinfo=UTC)
    kickoff = cutoff + timedelta(hours=1)
    _validate_target_exclusion(
        game_id="target", cutoff=cutoff, kickoff=kickoff, evidence_ids={"other"}
    )
    with pytest.raises(ValueError, match="appears in its own evidence"):
        _validate_target_exclusion(
            game_id="target", cutoff=cutoff, kickoff=kickoff, evidence_ids={"target"}
        )
    with pytest.raises(ValueError, match="not strictly before kickoff"):
        _validate_target_exclusion(
            game_id="target", cutoff=cutoff, kickoff=cutoff, evidence_ids=set()
        )


def test_posterior_artifact_preserves_normalized_full_pmf_and_shared_prior() -> None:
    prior = np.array([0.2, 0.3, 0.5])
    posterior = np.array([0.4, 0.4, 0.2000000000005])
    team = Team("a", "A", "fbs", prior)
    posterior_team = Team("a", "A", "fbs", posterior)
    result = PosteriorResult(
        pmfs={"a": posterior},
        converged=True,
        iterations=1,
        max_message_delta=0.0,
        objective=0.0,
        raw_game_factor_count=0,
        unique_pair_factor_count=0,
        max_team_degree=0,
    )
    evidence = {
        "included_game_ids_sha256": "ids",
        "included_game_rows_sha256": "rows",
    }
    arms = {
        arm: ({"a": team}, result, {"converged": True})
        for arm in (BASELINE_ARM, GENERIC_ARM, CONTEXT_ARM)
    }
    rows = _posterior_rows(
        season=2023,
        cutoff="preseason_no_evidence",
        stage="preseason",
        arms=arms,
        evidence=evidence,
        fbs_prior_rows={"a": {"pmf": prior}},
        fcs_population=2,
    )
    assert len(rows) == 3
    assert all(row["rank_population"] == 3 for row in rows)
    assert all(np.isclose(sum(row["prior_pmf"]), 1.0) for row in rows)
    assert all(np.isclose(sum(row["posterior_pmf"]), 1.0) for row in rows)
    assert all(row["prior_pmf"] == prior.tolist() for row in rows)
    normalized_posterior = posterior / posterior.sum()
    assert rows[0]["posterior_pmf"] == normalized_posterior.tolist()
    assert rows[0]["posterior_pmf_sha256"] == sha256_json(posterior_team.prior.tolist())


def test_predictive_mixtures_reconstruct_from_frozen_parameters_and_pmf_references() -> None:
    base = _base_likelihood()
    seed_fit = _fit()
    fits = {
        season: FrozenFit(
            **{
                **seed_fit.__dict__,
                "season": season,
                "training_seasons": (season - 1,),
            }
        )
        for season in (2023, 2024, 2025)
    }
    parameters = _predictive_model_parameters(base, fits)
    models = {
        row["model_arm"]: row for row in parameters["models"] if row["season"] == 2023
    }
    expected_generic_beta = base.beta.copy()
    expected_generic_beta[:9] *= 1.4
    expected_generic_beta[8] += 2.25
    assert models[GENERIC_ARM]["rank_location_beta"] == expected_generic_beta.tolist()
    assert models[GENERIC_ARM]["scale"] == pytest.approx(base.scale * 1.2)
    assert models[CONTEXT_ARM]["context_preprocessor"] == fits[2023].preprocessor.metadata()

    team_pmfs = {"home": [0.2, 0.3, 0.5], "away": [0.5, 0.3, 0.2]}
    posterior_rows = [
        {
            "season": 2023,
            "forecast_stage": "rolling",
            "cutoff_utc": "2023-09-01T00:00:00+00:00",
            "model_arm": arm,
            "team_id": team_id,
            "rank_population": len(pmf),
            "posterior_pmf_sha256": sha256_json(pmf),
        }
        for arm in (BASELINE_ARM, GENERIC_ARM, CONTEXT_ARM)
        for team_id, pmf in team_pmfs.items()
    ]
    forecasts = []
    for arm in (BASELINE_ARM, GENERIC_ARM, CONTEXT_ARM):
        forecast = {
            "forecast_id": f"forecast-{arm}",
            "forecast_stage": "rolling",
            "season": 2023,
            "cutoff_utc": "2023-09-01T00:00:00+00:00",
            "game_id": "game-1",
            "model_arm": arm,
            "home_id": "home",
            "away_id": "away",
            "neutral_site": "False" if arm == BASELINE_ARM else False,
            "context_feature_mean_adjustment_points": 1.25 if arm == CONTEXT_ARM else 0.0,
        }
        if arm != BASELINE_ARM:
            forecast["predictive_scale"] = float(models[arm]["scale"])
        forecasts.append(forecast)

    mixtures = _predictive_mixture_rows(forecasts, posterior_rows, parameters)
    assert len(mixtures) == 3
    context_row = next(row for row in mixtures if row["model_arm"] == CONTEXT_ARM)
    assert context_row["home_posterior_pmf_sha256"] == sha256_json(team_pmfs["home"])
    assert context_row["away_posterior_pmf_sha256"] == sha256_json(team_pmfs["away"])
    assert (context_row["home_rank_population"], context_row["away_rank_population"]) == (3, 3)
    assert context_row["context_feature_mean_adjustment_points"] == 1.25
    assert context_row["model_parameters_sha256"] == models[CONTEXT_ARM]["model_parameters_sha256"]
    baseline_row = next(row for row in mixtures if row["model_arm"] == BASELINE_ARM)
    assert baseline_row["neutral_site"] is False
    assert baseline_row["student_t_scale"] == base.scale


def test_rank_pmf_concentration_is_reported_per_checkpoint_and_arm() -> None:
    posterior_rows = [
        {
            "season": 2023,
            "forecast_stage": "rolling",
            "cutoff_utc": "2023-10-01T00:00:00+00:00",
            "model_arm": arm,
            "subdivision": "fbs",
            "posterior_pmf": [0.5, 0.5],
        }
        for arm in (BASELINE_ARM, GENERIC_ARM, CONTEXT_ARM)
    ]
    result = _rank_pmf_concentration(
        posterior_rows,
        {(2023, "2023-10-01T00:00:00+00:00"): "regular_season"},
    )
    assert len(result) == 3
    assert all(row["checkpoint_phase"] == "regular_season" for row in result)
    assert all(row["mean_entropy_nats"] == pytest.approx(log(2)) for row in result)
    assert all(row["mean_effective_rank_count"] == pytest.approx(2) for row in result)


def test_gain_concentration_keeps_reference_arms_separate() -> None:
    metric_values = {
        "crps": 1.0,
        "margin_nll": 1.0,
        "home_win_brier": 1.0,
        "median_absolute_error": 1.0,
        "coverage_50": 1.0,
        "coverage_80": 1.0,
        "coverage_95": 1.0,
    }
    rows = []
    for arm, crps in (
        (BASELINE_ARM, 1.0),
        (GENERIC_ARM, 0.8),
        (CONTEXT_ARM, 0.6),
    ):
        rows.append(
            {
                "forecast_stage": "rolling",
                "season": 2023,
                "game_id": "army-navy-2023",
                "cutoff_utc": "2023-12-09T00:00:00+00:00",
                "phase": "army_navy",
                "home_id": "army",
                "away_id": "navy",
                "evidence_ids_sha256": "ids",
                "evidence_rows_sha256": "rows",
                "model_arm": arm,
                **metric_values,
                "crps": crps,
            }
        )

    paired = _paired_forecasts(rows)
    concentration = _gain_concentration(paired)
    army_context = {
        row["reference_arm"]: row
        for row in concentration
        if row["candidate_arm"] == CONTEXT_ARM and row["team_id"] == "army"
    }

    assert set(army_context) == {BASELINE_ARM, GENERIC_ARM}
    assert army_context[BASELINE_ARM]["game_appearances"] == 1
    assert army_context[BASELINE_ARM]["mean_crps_delta"] == pytest.approx(-0.4)
    assert army_context[GENERIC_ARM]["game_appearances"] == 1
    assert army_context[GENERIC_ARM]["mean_crps_delta"] == pytest.approx(-0.2)


def test_baseline_checkpoint_binds_targets_to_cutoff_evidence_and_valid_pmfs() -> None:
    teams = {
        team_id: Team(team_id, team_id, "fbs", np.array([0.4, 0.6]))
        for team_id in ("a", "b")
    }
    metadata = {"home_id": "a", "away_id": "b"}
    outcome = {"home_points": 24, "away_points": 17}
    evidence = {
        "included_game_ids_sha256": "ids",
        "included_game_rows_sha256": "rows",
    }
    reference = {
        ("rolling", 2023, "game"): {
            "cutoff_utc": "2023-09-01T00:00:00+00:00",
            "home_id": "a",
            "away_id": "b",
            "evidence_ids_sha256": "ids",
            "evidence_rows_sha256": "rows",
            "actual_home_points": "24",
            "actual_away_points": "17",
        }
    }
    checks = _verify_baseline_checkpoint(
        season=2023,
        stage="rolling",
        cutoff="2023-09-01T00:00:00+00:00",
        teams=teams,
        target_games={"game": (metadata, outcome)},
        evidence=evidence,
        reference_by_key=reference,
    )
    assert checks == 2
    reference[("rolling", 2023, "game")]["evidence_rows_sha256"] = "wrong"
    with pytest.raises(ValueError, match="evidence_rows_sha256 mismatch"):
        _verify_baseline_checkpoint(
            season=2023,
            stage="rolling",
            cutoff="2023-09-01T00:00:00+00:00",
            teams=teams,
            target_games={"game": (metadata, outcome)},
            evidence=evidence,
            reference_by_key=reference,
        )


def test_baseline_parity_fails_when_fresh_score_is_perturbed(tmp_path) -> None:
    numeric_fields = (
        "expected_home_margin",
        "median_home_margin",
        "home_win_probability",
        "away_win_probability",
        "tie_probability",
        "margin_interval_50_low",
        "margin_interval_50_high",
        "margin_interval_80_low",
        "margin_interval_80_high",
        "margin_interval_95_low",
        "margin_interval_95_high",
        "crps",
        "crps_numerical_error_estimate",
        "crps_quadrature_order",
        "crps_tail_multiple",
        "margin_nll",
        "median_absolute_error",
        "home_win_brier",
        "coverage_50",
        "coverage_80",
        "coverage_95",
    )
    reference = {
        "forecast_stage": "rolling",
        "season": 2023,
        "game_id": "game-1",
        "model_arm": BASELINE_ARM,
        "forecast_population": "fbs_fbs",
        "cutoff_utc": "2023-09-01T00:00:00+00:00",
        "phase": "regular_season",
        "home_id": "a",
        "away_id": "b",
        "actual_home_points": 28,
        "actual_away_points": 21,
        "evidence_ids_sha256": "ids",
        "evidence_rows_sha256": "rows",
        **{field: 1.0 for field in numeric_fields},
    }
    _write_csv(tmp_path / "forecast_scores.csv", [reference])

    parity = _verify_baseline_parity([reference], tmp_path, baseline_pmf_checks=2)
    assert parity["passed"] is True

    perturbed = {**reference, "crps": 1.1}
    with pytest.raises(ValueError, match="baseline parity failed.*crps"):
        _verify_baseline_parity([perturbed], tmp_path, baseline_pmf_checks=2)
