from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    Team,
    game_factor,
    infer_posterior,
)
from gippyrank.research.issue_194_part_b import (
    RANK_VALUES,
    SCENARIO_WEIGHTS,
    TEAM_IDS,
    TEAM_MARGINALS,
    enumerate_exact_mixture_posterior,
    enumerate_exact_posterior,
    infer_trial,
    mixture_joint_prior,
    product_joint_prior,
    scenario_components,
    simulate_trial,
)


def _likelihood() -> LikelihoodV1:
    beta = np.zeros(34)
    beta[0] = -50.0
    return LikelihoodV1(beta, scale=7.0, degrees_of_freedom=15.0)


def test_shared_scenario_marginals_match_independent_priors_and_normalize() -> None:
    components = scenario_components()
    component_pmfs = np.stack([pmfs for _, pmfs in components])
    assert np.array_equal(
        np.asarray([weight for weight, _ in components]), SCENARIO_WEIGHTS
    )
    assert np.array_equal(
        np.average(component_pmfs, axis=0, weights=SCENARIO_WEIGHTS), TEAM_MARGINALS
    )
    assert np.allclose(component_pmfs.sum(axis=2), 1.0)

    independent = product_joint_prior(TEAM_MARGINALS)
    shared = mixture_joint_prior(components)
    assert independent.shape == (5, 5, 5, 5)
    assert np.isclose(independent.sum(), 1.0)
    assert np.isclose(shared.sum(), 1.0)
    assert not np.allclose(independent, shared)
    assert np.allclose(
        np.stack(
            [
                shared.sum(axis=tuple(i for i in range(4) if i != team))
                for team in range(4)
            ]
        ),
        TEAM_MARGINALS,
    )


def test_exact_mixture_uses_evidence_weighted_component_posterior() -> None:
    components = (
        (0.5, np.asarray([[0.9, 0.1], [0.5, 0.5]])),
        (0.5, np.asarray([[0.1, 0.9], [0.5, 0.5]])),
    )
    factor = np.asarray([[1.0, 1.0], [9.0, 9.0]])
    result = enumerate_exact_mixture_posterior(components, [(0, 1, factor)])
    assert np.allclose(result.component_posterior_weights, [0.18, 0.82])
    assert not np.allclose(result.component_posterior_weights, [0.5, 0.5])
    assert np.allclose(result.pmfs.sum(axis=1), 1.0)
    assert np.isclose(result.evidence, 5.0)


def test_exact_enumerator_normalizes_full_five_team_five_state_space() -> None:
    team_pmfs = np.full((5, 5), 0.2)
    result = enumerate_exact_posterior(product_joint_prior(team_pmfs), [])
    assert result.pmfs.shape == (5, 5)
    assert np.allclose(result.pmfs, team_pmfs)
    assert np.isclose(result.evidence, 1.0)
    with pytest.raises(ValueError, match="five rank states"):
        enumerate_exact_posterior(np.full((6, 6), 1.0 / 36), [])


def test_exact_and_existing_bp_agree_on_product_prior_tree() -> None:
    likelihood = _likelihood()
    teams = [
        Team(team_id, team_id, "fbs", TEAM_MARGINALS[index].copy())
        for index, team_id in enumerate(TEAM_IDS)
    ]
    games = [
        Game("ab", "A", "B", "fbs", "fbs", 24, 10),
        Game("bc", "B", "C", "fbs", "fbs", 7, 20, True),
        Game("cd", "C", "D", "fbs", "fbs", 31, 17),
    ]
    index = {team.team_id: position for position, team in enumerate(teams)}
    factors = [
        (
            index[game.home_id],
            index[game.away_id],
            game_factor(
                game, teams[index[game.home_id]], teams[index[game.away_id]], likelihood
            ),
        )
        for game in games
    ]
    exact = enumerate_exact_posterior(product_joint_prior(TEAM_MARGINALS), factors)
    bp = infer_posterior(teams, games, likelihood, tolerance=1e-12)
    bp_pmfs = np.stack([bp.pmfs[team_id] for team_id in TEAM_IDS])
    assert bp.converged
    assert np.allclose(bp_pmfs, exact.pmfs, atol=1e-10)
    assert np.array_equal(RANK_VALUES, np.arange(1, 6, dtype=float))


def test_paired_draws_are_deterministic_and_inference_ignores_known_ranks() -> None:
    likelihood = _likelihood()
    first = simulate_trial("shared_context", "cycle_4", 1, 194300, likelihood)
    repeated = simulate_trial("shared_context", "cycle_4", 1, 194300, likelihood)
    assert first.true_ranks == repeated.true_ranks
    assert first.selected_scenario == repeated.selected_scenario
    assert first.games == repeated.games

    changed_labels = replace(
        first, true_ranks=tuple(6 - rank for rank in first.true_ranks)
    )
    original_fit = infer_trial(first, likelihood)
    relabeled_fit = infer_trial(changed_labels, likelihood)
    for arm in original_fit.pmfs:
        assert np.array_equal(original_fit.pmfs[arm], relabeled_fit.pmfs[arm])


def test_reordering_cyclic_games_does_not_change_inference() -> None:
    likelihood = _likelihood()
    trial = simulate_trial("independent_null", "cycle_4", 1, 194100, likelihood)
    reversed_trial = replace(trial, games=tuple(reversed(trial.games)))
    first = infer_trial(trial, likelihood)
    second = infer_trial(reversed_trial, likelihood)
    for arm in first.pmfs:
        assert np.allclose(first.pmfs[arm], second.pmfs[arm], atol=1e-12)
