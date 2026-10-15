"""Bounded exact synthetic mechanism check for issue 194 Part B.

This research module keeps simulated-rank generation, exact enumeration, and
the existing product-prior BP call separate. Nothing here participates in
production inference or publication.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np

from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    Team,
    game_factor,
    game_margin_parameters,
    infer_posterior,
)

TEAM_IDS = ("A", "B", "C", "D")
RANK_VALUES = np.arange(1, 6, dtype=float)
TEAM_MARGINALS = np.asarray(
    [
        [0.35, 0.25, 0.18, 0.13, 0.09],
        [0.10, 0.18, 0.24, 0.26, 0.22],
        [0.17, 0.21, 0.24, 0.22, 0.16],
        [0.26, 0.24, 0.22, 0.16, 0.12],
    ],
    dtype=float,
)
SCENARIO_DELTA = np.asarray([0.04, 0.02, 0.0, -0.02, -0.04], dtype=float)
SCENARIO_NAMES = ("rank_profile_a", "rank_profile_b")
SCENARIO_WEIGHTS = np.asarray([0.5, 0.5], dtype=float)
SCENARIO_PMFS = np.stack(
    (TEAM_MARGINALS + SCENARIO_DELTA, TEAM_MARGINALS - SCENARIO_DELTA)
)
GRAPH_EDGES: dict[str, tuple[tuple[int, int], ...]] = {
    "path_tree": ((0, 1), (1, 2), (2, 3)),
    "cycle_4": ((0, 1), (1, 2), (2, 3), (3, 0)),
}
CASE_NAMES = ("independent_null", "shared_context")
DRAWS_PER_PANEL = 100
SEED_RANGES: dict[tuple[str, str], range] = {
    ("independent_null", "path_tree"): range(194000, 194100),
    ("independent_null", "cycle_4"): range(194100, 194200),
    ("shared_context", "path_tree"): range(194200, 194300),
    ("shared_context", "cycle_4"): range(194300, 194400),
}
MAX_ASSIGNMENTS = 3125


@dataclass(frozen=True)
class ExactPosterior:
    pmfs: np.ndarray
    evidence: float
    component_posterior_weights: tuple[float, ...] = ()


@dataclass(frozen=True)
class SyntheticTrial:
    case: str
    graph: str
    draw_id: int
    seed: int
    true_ranks: tuple[int, ...]
    selected_scenario: str | None
    games: tuple[Game, ...]


@dataclass(frozen=True)
class TrialInference:
    trial: SyntheticTrial
    pmfs: dict[str, np.ndarray]
    bp_converged: bool
    bp_iterations: int
    bp_max_message_delta: float
    scenario_posterior_weights: tuple[float, ...]


def scenario_components() -> tuple[tuple[float, np.ndarray], ...]:
    """Return equally weighted conditional team PMFs for the shared case."""
    return tuple(
        (float(weight), pmf.copy())
        for weight, pmf in zip(SCENARIO_WEIGHTS, SCENARIO_PMFS, strict=True)
    )


def product_joint_prior(team_pmfs: np.ndarray) -> np.ndarray:
    """Return the independent joint prior with one axis per team."""
    pmfs = np.asarray(team_pmfs, dtype=float)
    if pmfs.ndim != 2 or pmfs.shape[0] == 0 or pmfs.shape[1] == 0:
        raise ValueError("team PMFs must be a nonempty team-by-rank matrix")
    if pmfs.shape[0] > 5 or pmfs.shape[1] > 5:
        raise ValueError("exact mechanism check supports at most five teams/states")
    if not np.isfinite(pmfs).all() or np.any(pmfs < 0):
        raise ValueError("team PMFs must be finite and non-negative")
    if not np.allclose(pmfs.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("each team PMF must sum to one")
    joint = pmfs[0]
    for pmf in pmfs[1:]:
        joint = np.multiply.outer(joint, pmf)
    return np.asarray(joint, dtype=float)


def mixture_joint_prior(
    components: Sequence[tuple[float, np.ndarray]],
) -> np.ndarray:
    """Form a prior joint PMF by mixing products of component team PMFs."""
    if not components:
        raise ValueError("at least one mixture component is required")
    weights = np.asarray([weight for weight, _ in components], dtype=float)
    if (
        np.any(weights <= 0)
        or not np.isfinite(weights).all()
        or not np.isclose(weights.sum(), 1.0, atol=1e-12)
    ):
        raise ValueError("mixture weights must be positive and sum to one")
    joints = [product_joint_prior(pmfs) for _, pmfs in components]
    if any(joint.shape != joints[0].shape for joint in joints[1:]):
        raise ValueError("mixture components must have the same team/rank shape")
    return np.average(np.stack(joints), axis=0, weights=weights)


@lru_cache(maxsize=8)
def _assignment_table(team_count: int, rank_state_count: int) -> np.ndarray:
    count = rank_state_count**team_count
    if team_count < 1 or rank_state_count < 1 or count > MAX_ASSIGNMENTS:
        raise ValueError("exact enumeration exceeds the 5-team, 5-state ceiling")
    table = np.asarray(
        list(product(range(rank_state_count), repeat=team_count)), dtype=np.int16
    )
    table.setflags(write=False)
    return table


def enumerate_exact_posterior(
    joint_prior: np.ndarray,
    factors: Sequence[tuple[int, int, np.ndarray]],
) -> ExactPosterior:
    """Enumerate a rank-valued joint posterior under pairwise likelihoods."""
    prior = np.asarray(joint_prior, dtype=float)
    if prior.ndim < 1 or prior.ndim > 5 or len(set(prior.shape)) != 1:
        raise ValueError("joint prior must have 1–5 equally sized rank axes")
    rank_state_count = prior.shape[0]
    if rank_state_count > 5:
        raise ValueError("exact mechanism check supports at most five rank states")
    assignment_count = rank_state_count**prior.ndim
    if assignment_count > MAX_ASSIGNMENTS:
        raise ValueError("exact enumeration exceeds the 5-team, 5-state ceiling")
    if not np.isfinite(prior).all() or np.any(prior < 0):
        raise ValueError("joint prior must be finite and non-negative")
    prior_sum = float(prior.sum())
    if prior_sum <= 0 or not np.isclose(prior_sum, 1.0, atol=1e-12):
        raise ValueError("joint prior must be normalized")

    assignments = _assignment_table(prior.ndim, rank_state_count)
    weights = prior[tuple(assignments.T)].astype(float, copy=True)
    for first, second, raw_factor in factors:
        factor = np.asarray(raw_factor, dtype=float)
        if first == second or not (
            0 <= first < prior.ndim and 0 <= second < prior.ndim
        ):
            raise ValueError("factor team indices must name two distinct teams")
        if factor.shape != (rank_state_count, rank_state_count):
            raise ValueError("pairwise factor shape must match the rank-state count")
        if not np.isfinite(factor).all() or np.any(factor < 0):
            raise ValueError("pairwise factors must be finite and non-negative")
        weights *= factor[assignments[:, first], assignments[:, second]]

    evidence = float(weights.sum())
    if not np.isfinite(evidence) or evidence <= 0:
        raise FloatingPointError("joint evidence is zero or non-finite")
    assignment_posterior = weights / evidence
    marginals = np.stack(
        [
            np.bincount(
                assignments[:, team_index],
                weights=assignment_posterior,
                minlength=rank_state_count,
            )
            for team_index in range(prior.ndim)
        ]
    )
    return ExactPosterior(marginals, evidence)


def enumerate_exact_mixture_posterior(
    components: Sequence[tuple[float, np.ndarray]],
    factors: Sequence[tuple[int, int, np.ndarray]],
) -> ExactPosterior:
    """Integrate component posteriors using prior weight times evidence."""
    if not components:
        raise ValueError("at least one mixture component is required")
    prior_weights = np.asarray([weight for weight, _ in components], dtype=float)
    if (
        np.any(prior_weights <= 0)
        or not np.isfinite(prior_weights).all()
        or not np.isclose(prior_weights.sum(), 1.0, atol=1e-12)
    ):
        raise ValueError("mixture weights must be positive and sum to one")
    component_results = [
        enumerate_exact_posterior(product_joint_prior(pmfs), factors)
        for _, pmfs in components
    ]
    if any(
        result.pmfs.shape != component_results[0].pmfs.shape
        for result in component_results
    ):
        raise ValueError("mixture components must have the same team/rank shape")
    weighted_evidence = prior_weights * np.asarray(
        [result.evidence for result in component_results], dtype=float
    )
    evidence = float(weighted_evidence.sum())
    if not np.isfinite(evidence) or evidence <= 0:
        raise FloatingPointError("mixture evidence is zero or non-finite")
    posterior_weights = weighted_evidence / evidence
    marginals = np.average(
        np.stack([result.pmfs for result in component_results]),
        axis=0,
        weights=posterior_weights,
    )
    return ExactPosterior(marginals, evidence, tuple(posterior_weights.tolist()))


def simulate_trial(
    case: str,
    graph: str,
    draw_id: int,
    seed: int,
    likelihood: LikelihoodV1,
) -> SyntheticTrial:
    """Generate known rank states and V1-consistent continuous game margins."""
    if case not in CASE_NAMES or graph not in GRAPH_EDGES:
        raise ValueError("unknown case or graph")
    rng = np.random.default_rng(seed)
    selected_scenario: str | None = None
    if case == "shared_context":
        scenario_index = int(rng.choice(len(SCENARIO_WEIGHTS), p=SCENARIO_WEIGHTS))
        selected_scenario = SCENARIO_NAMES[scenario_index]
        generating_pmfs = SCENARIO_PMFS[scenario_index]
    else:
        generating_pmfs = TEAM_MARGINALS

    true_ranks = tuple(
        int(rng.choice(RANK_VALUES.astype(int), p=pmf)) for pmf in generating_pmfs
    )
    teams = [
        Team(team_id, team_id, "fbs", TEAM_MARGINALS[index].copy())
        for index, team_id in enumerate(TEAM_IDS)
    ]
    games: list[Game] = []
    for edge_index, (first_index, second_index) in enumerate(GRAPH_EDGES[graph]):
        home_index, away_index = first_index, second_index
        if int(rng.integers(0, 2)):
            home_index, away_index = away_index, home_index
        neutral_site = bool(rng.random() < 0.25)
        game_id = f"{graph}-g{edge_index}"
        template = Game(
            game_id,
            TEAM_IDS[home_index],
            TEAM_IDS[away_index],
            "fbs",
            "fbs",
            0,
            0,
            neutral_site,
        )
        locations, _ = game_margin_parameters(
            template, teams[home_index], teams[away_index], likelihood
        )
        location = locations[true_ranks[home_index] - 1, true_ranks[away_index] - 1]
        margin = float(
            location + likelihood.scale * rng.standard_t(likelihood.degrees_of_freedom)
        )
        # Game's factor builder consumes only the point difference. V1 models
        # a continuous margin, so these research-only ephemeral values remain
        # continuous and are never written as actual football scores.
        games.append(
            Game(
                game_id,
                TEAM_IDS[home_index],
                TEAM_IDS[away_index],
                "fbs",
                "fbs",
                margin,
                0.0,
                neutral_site,
            )
        )
    return SyntheticTrial(
        case,
        graph,
        draw_id,
        seed,
        true_ranks,
        selected_scenario,
        tuple(games),
    )


def infer_trial(
    trial: SyntheticTrial,
    likelihood: LikelihoodV1,
) -> TrialInference:
    """Run all three inference arms without consulting simulated true ranks."""
    teams = [
        Team(team_id, team_id, "fbs", TEAM_MARGINALS[index].copy())
        for index, team_id in enumerate(TEAM_IDS)
    ]
    team_indices = {team.team_id: index for index, team in enumerate(teams)}
    factors = [
        (
            team_indices[game.home_id],
            team_indices[game.away_id],
            game_factor(
                game,
                teams[team_indices[game.home_id]],
                teams[team_indices[game.away_id]],
                likelihood,
            ),
        )
        for game in trial.games
    ]
    product_prior = product_joint_prior(TEAM_MARGINALS)
    exact_product = enumerate_exact_posterior(product_prior, factors)
    if trial.case == "shared_context":
        exact_true = enumerate_exact_mixture_posterior(scenario_components(), factors)
    elif trial.case == "independent_null":
        exact_true = enumerate_exact_posterior(product_prior, factors)
    else:
        raise ValueError(f"unknown generative case: {trial.case}")

    bp = infer_posterior(
        teams,
        list(trial.games),
        likelihood,
        tolerance=1e-12,
        max_iterations=500,
        damping=0.35,
    )
    bp_pmfs = np.stack([bp.pmfs[team_id] for team_id in TEAM_IDS])
    predictions = {
        "exact_true_joint": exact_true.pmfs,
        "exact_product_marginals": exact_product.pmfs,
        "bp_product_marginals": bp_pmfs,
    }
    return TrialInference(
        trial,
        predictions,
        bp.converged,
        bp.iterations,
        bp.max_message_delta,
        exact_true.component_posterior_weights,
    )


def _score_arm(pmfs: np.ndarray, true_ranks: tuple[int, ...]) -> dict[str, float]:
    probabilities = np.asarray(pmfs, dtype=float)
    ranks = np.asarray(true_ranks, dtype=int)
    observed = np.eye(probabilities.shape[1], dtype=float)[ranks - 1]
    expected_ranks = probabilities @ RANK_VALUES[: probabilities.shape[1]]
    rank_errors = expected_ranks - ranks
    return {
        "mean_rank_nll": float(
            np.mean(-np.log(probabilities[np.arange(len(ranks)), ranks - 1]))
        ),
        "mean_multicategory_brier": float(
            np.mean(np.sum((probabilities - observed) ** 2, axis=1))
        ),
        "mean_signed_expected_rank_error": float(rank_errors.mean()),
        "mean_absolute_expected_rank_error": float(np.abs(rank_errors).mean()),
    }


def _posterior_difference(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    tv = 0.5 * np.abs(first - second).sum(axis=1)
    expected_rank_delta = np.abs((first - second) @ RANK_VALUES[: first.shape[1]])
    return float(tv.mean()), float(expected_rank_delta.mean())


def run_experiment(likelihood: LikelihoodV1) -> dict[str, list[dict[str, object]]]:
    """Run the protocol's fixed 400 simulations and return compact row sets."""
    arm_names = (
        "exact_true_joint",
        "exact_product_marginals",
        "bp_product_marginals",
    )
    arm_draw_rows: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    predictions_by_arm: dict[tuple[str, str, str], list[np.ndarray]] = {}
    ranks_by_panel: dict[tuple[str, str], list[tuple[int, ...]]] = {}
    diagnostic_draws: dict[tuple[str, str], list[dict[str, float]]] = {}
    scenario_rows: list[dict[str, object]] = []

    for (case, graph), seeds in SEED_RANGES.items():
        if len(seeds) != DRAWS_PER_PANEL:
            raise RuntimeError("frozen seed panel no longer contains 100 draws")
        panel = (case, graph)
        ranks_by_panel[panel] = []
        diagnostic_draws[panel] = []
        for draw_id, seed in enumerate(seeds, start=1):
            trial = simulate_trial(case, graph, draw_id, seed, likelihood)
            result = infer_trial(trial, likelihood)
            ranks_by_panel[panel].append(trial.true_ranks)
            for arm in arm_names:
                key = (case, graph, arm)
                arm_draw_rows.setdefault(key, [])
                predictions_by_arm.setdefault(key, [])
                prediction = result.pmfs[arm]
                prediction.setflags(write=False)
                predictions_by_arm[key].append(prediction)
                scores = _score_arm(prediction, trial.true_ranks)
                arm_draw_rows[key].append(
                    {
                        "case": case,
                        "graph": graph,
                        "draw_id": draw_id,
                        "seed": seed,
                        "arm": arm,
                        "true_ranks": ";".join(map(str, trial.true_ranks)),
                        **scores,
                        "bp_converged": (
                            result.bp_converged if arm == "bp_product_marginals" else ""
                        ),
                        "bp_iterations": (
                            result.bp_iterations
                            if arm == "bp_product_marginals"
                            else ""
                        ),
                    }
                )
            independent = result.pmfs["exact_product_marginals"]
            true_joint = result.pmfs["exact_true_joint"]
            bp = result.pmfs["bp_product_marginals"]
            dep_tv, dep_rank = _posterior_difference(independent, true_joint)
            bp_tv, bp_rank = _posterior_difference(bp, independent)
            combined_tv, combined_rank = _posterior_difference(bp, true_joint)
            diagnostic_draws[panel].append(
                {
                    "product_vs_true_joint_tv": dep_tv,
                    "product_vs_true_joint_abs_expected_rank_error": dep_rank,
                    "bp_vs_product_tv": bp_tv,
                    "bp_vs_product_abs_expected_rank_error": bp_rank,
                    "bp_vs_true_joint_tv": combined_tv,
                    "bp_vs_true_joint_abs_expected_rank_error": combined_rank,
                }
            )
            if case == "shared_context":
                weights = result.scenario_posterior_weights
                if len(weights) != 2:
                    raise RuntimeError("shared-prior inference lost mixture weights")
                scenario_rows.append(
                    {
                        "case": case,
                        "graph": graph,
                        "draw_id": draw_id,
                        "seed": seed,
                        "generated_scenario": trial.selected_scenario,
                        "prior_weight_a": float(SCENARIO_WEIGHTS[0]),
                        "prior_weight_b": float(SCENARIO_WEIGHTS[1]),
                        "posterior_weight_a": weights[0],
                        "posterior_weight_b": weights[1],
                    }
                )

    calibration_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for (case, graph, arm), draw_rows in sorted(arm_draw_rows.items()):
        panel_ranks = ranks_by_panel[(case, graph)]
        panel_pmfs = np.stack(predictions_by_arm[(case, graph, arm)])
        panel_rank_array = np.asarray(panel_ranks, dtype=int)
        calibration_gaps: list[float] = []
        for team_index, team_id in enumerate(TEAM_IDS):
            for rank_index, rank in enumerate(range(1, 6)):
                mean_probability = float(panel_pmfs[:, team_index, rank_index].mean())
                observed_frequency = float(
                    np.mean(panel_rank_array[:, team_index] == rank)
                )
                gap = mean_probability - observed_frequency
                calibration_gaps.append(abs(gap))
                calibration_rows.append(
                    {
                        "case": case,
                        "graph": graph,
                        "arm": arm,
                        "team_id": team_id,
                        "rank_state": rank,
                        "draws": len(panel_ranks),
                        "mean_forecast_probability": mean_probability,
                        "observed_frequency": observed_frequency,
                        "forecast_minus_frequency": gap,
                    }
                )
        scores = {
            key: float(np.mean([float(row[key]) for row in draw_rows]))
            for key in (
                "mean_rank_nll",
                "mean_multicategory_brier",
                "mean_signed_expected_rank_error",
                "mean_absolute_expected_rank_error",
            )
        }
        bp_rows = [row for row in draw_rows if row["bp_iterations"] != ""]
        summary_rows.append(
            {
                "case": case,
                "graph": graph,
                "arm": arm,
                "draws": len(draw_rows),
                "teams_per_draw": len(TEAM_IDS),
                **scores,
                "rank_calibration_macro_mae": float(np.mean(calibration_gaps)),
                "bp_converged_fraction": (
                    float(np.mean([bool(row["bp_converged"]) for row in bp_rows]))
                    if bp_rows
                    else ""
                ),
                "bp_mean_iterations": (
                    float(np.mean([int(row["bp_iterations"]) for row in bp_rows]))
                    if bp_rows
                    else ""
                ),
                "bp_max_iterations": (
                    max(int(row["bp_iterations"]) for row in bp_rows) if bp_rows else ""
                ),
            }
        )

    diagnostics_rows = []
    for (case, graph), rows in sorted(diagnostic_draws.items()):
        diagnostics_rows.append(
            {
                "case": case,
                "graph": graph,
                "draws": len(rows),
                **{
                    metric: float(np.mean([row[metric] for row in rows]))
                    for metric in rows[0]
                },
                **{
                    f"max_{metric}": float(max(row[metric] for row in rows))
                    for metric in rows[0]
                },
            }
        )

    return {
        "summary": summary_rows,
        "calibration": calibration_rows,
        "draw_scores": [
            row for key in sorted(arm_draw_rows) for row in arm_draw_rows[key]
        ],
        "diagnostics": diagnostics_rows,
        "scenario_weights": scenario_rows,
    }
