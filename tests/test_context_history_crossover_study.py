"""Numerical and target-isolation checks for issue-146 diagnostics."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from study_context_history_crossover import (
    Components,
    component_from_model,
    hybrid_arms,
    normal_mixture_pmf,
    uncertainty_trigger,
)

from gippyrank.posterior.engine import Game, LikelihoodV1, Team, infer_posterior


def _component(
    team_id: str, locations: list[float], scale: float, population: int = 12
) -> Components:
    points = np.asarray(locations, dtype=float)
    pmf = normal_mixture_pmf(points, scale, population)
    return Components(
        team_id=team_id,
        team_name=team_id,
        population=population,
        features={"history": 0.0},
        lag1_z=np.asarray([0.0]),
        lag_zs=(),
        locations=points,
        scale=scale,
        pmf=pmf,
    )


def test_primary_hybrids_reconstruct_fitted_arms_and_keep_valid_support() -> None:
    context = {"team": _component("team", [-0.3, 0.15, 0.45], 0.55)}
    history = {"team": _component("team", [-0.1, 0.05, 0.22], 0.72)}
    c_priors = {"team": (12, "Team", context["team"].pmf)}
    h_priors = {"team": (12, "Team", history["team"].pmf)}

    arms, matched = hybrid_arms(context, history, c_priors, h_priors)

    assert matched == {"team"}
    np.testing.assert_allclose(arms["CC"]["team"], context["team"].pmf, atol=1e-12)
    np.testing.assert_allclose(arms["HH"]["team"], history["team"].pmf, atol=1e-12)
    for arm in arms.values():
        pmf = arm["team"]
        assert pmf.shape == (12,)
        assert np.isfinite(pmf).all()
        assert np.all(pmf >= 0)
        assert np.isclose(pmf.sum(), 1.0, atol=1e-12)

    repeated, _ = hybrid_arms(context, history, c_priors, h_priors)
    for arm in arms:
        np.testing.assert_array_equal(arms[arm]["team"], repeated[arm]["team"])


def test_component_extraction_never_reads_target_outcomes() -> None:
    class OutcomeFreeRow:
        def __init__(self) -> None:
            self.team_id = "team"
            self.team_name = "Team"
            self.population = 12
            self.features = {"history": 0.2}
            self.lag1_z = np.asarray([0.1, 0.2])
            self.lag_zs = ()

        @property
        def target_z(self):
            raise AssertionError("target outcomes must not be read")

        @property
        def target_ranks(self):
            raise AssertionError("target outcomes must not be read")

    class Model:
        def conditional_parameters(self, features, lag1_z, lag_zs):
            assert features == {"history": 0.2}
            np.testing.assert_array_equal(lag1_z, [0.1, 0.2])
            return np.asarray([-0.1, 0.3]), 0.6

        def pmf(self, features, lag1_z, population, lag_zs):
            locations, scale = self.conditional_parameters(features, lag1_z, lag_zs)
            return normal_mixture_pmf(locations, scale, population)

    components = component_from_model(Model(), OutcomeFreeRow())
    assert components.team_id == "team"
    np.testing.assert_allclose(components.locations, [-0.1, 0.3])


def test_uncertainty_decomposition_trigger_uses_predeclared_effect_threshold() -> None:
    rows = []
    for season in (2022, 2023, 2024):
        for checkpoint in (2, 3):
            values = {
                "CC": 1.1,
                "HH": 1.0,
                "C_center_H_uncertainty": 1.075,
                "H_center_C_uncertainty": 1.0,
            }
            for arm, value in values.items():
                rows.append(
                    {
                        "summary_type": "season_checkpoint",
                        "season": season,
                        "checkpoint": checkpoint,
                        "arm": arm,
                        "population": "matched_primary",
                        "period": "september",
                        "mean_posterior_nll": value,
                    }
                )

    result = uncertainty_trigger(rows)

    assert result["triggered"] is True
    assert result["qualifying_seasons"] == [2023, 2024]


def test_identical_inputs_produce_identical_posterior_results() -> None:
    beta = np.zeros(34)
    beta[0] = -50.0
    likelihood = LikelihoodV1(beta, scale=7.0, degrees_of_freedom=15.0)
    teams = [
        Team("a", "A", "fbs", np.asarray([0.7, 0.3])),
        Team("b", "B", "fbs", np.asarray([0.4, 0.6])),
    ]
    games = [Game("game", "a", "b", "fbs", "fbs", 28, 10)]

    first = infer_posterior(teams, games, likelihood, tolerance=1e-12)
    second = infer_posterior(teams, games, likelihood, tolerance=1e-12)

    assert first.converged and second.converged
    assert first.iterations == second.iterations
    assert first.max_message_delta == second.max_message_delta
    for team_id in first.pmfs:
        np.testing.assert_array_equal(first.pmfs[team_id], second.pmfs[team_id])


def test_committed_study_audits_record_baseline_and_arm_contracts() -> None:
    results = ROOT / "data/processed/context_history_crossover"

    def read_csv(name: str) -> list[dict[str, str]]:
        with (results / name).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    parity = read_csv("baseline_reproduction_checks.csv")
    assert len(parity) == 700
    assert all(row["passed"] == "True" for row in parity)
    assert max(float(row["max_absolute_error"]) for row in parity) <= 1e-8

    evidence = read_csv("arm_evidence_and_convergence.csv")
    arms = {"CC", "HH", "C_center_H_uncertainty", "H_center_C_uncertainty"}
    assert len(evidence) == 112
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in evidence:
        grouped[row["season"], row["checkpoint"]].append(row)
    assert len(grouped) == 28
    for rows in grouped.values():
        assert {row["arm"] for row in rows} == arms
        assert all(row["converged"] == "True" for row in rows)
        for field in (
            "included_game_ids_sha256",
            "included_game_rows_sha256",
            "fcs_fallback_count",
            "fcs_fallback_ids_sha256",
            "total_inference_team_count",
            "likelihood_sha256",
            "inference_max_iterations",
            "inference_tolerance",
            "inference_damping",
        ):
            assert len({row[field] for row in rows}) == 1
        assert {row["inference_max_iterations"] for row in rows} == {"500"}
        assert {row["inference_tolerance"] for row in rows} == {"1e-09"}
        assert {row["inference_damping"] for row in rows} == {"0.35"}

    teams = read_csv("hybrid_posterior_team_results.csv")
    populations: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in teams:
        populations[row["season"], row["checkpoint"]][row["arm"]].add(row["team_id"])
    assert len(populations) == 28
    for by_arm in populations.values():
        assert set(by_arm) == arms
        assert all(team_ids == by_arm["CC"] for team_ids in by_arm.values())

    provenance = json.loads((results / "provenance.json").read_text("utf-8"))
    assert provenance["secondary_uncertainty_decomposition"]["triggered"] is False
    for season_text, models in provenance["models"].items():
        season = int(season_text)
        assert models["context_instance"]["trained_through_season"] == season - 1
        assert models["history_instance"]["trained_through_season"] == season - 1
