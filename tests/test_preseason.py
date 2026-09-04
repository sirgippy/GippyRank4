import runpy
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from gippyrank.preseason import (
    DirectRankModel,
    GenericRankPrior,
    Preprocessor,
    TeamSeason,
    clean_name,
    deterministic_quadrature,
    normal_pmf,
    pmf_summaries,
    rank_bin_edges,
    rank_to_z,
    team_log_score,
)


def test_team_name_cleaning_is_conservative_and_joinable() -> None:
    assert clean_name("  Texas  A&M ") == "texas aandm"


def test_rank_pmf_is_full_support_normalized_and_integrates_boundary_bins() -> None:
    location, scale, population = -0.4, 0.8, 10
    pmf = normal_pmf(location, scale, population)
    assert pmf.shape == (population,)
    assert np.isclose(pmf.sum(), 1.0)
    assert np.all(pmf >= 0)
    edges = rank_bin_edges(population)
    assert np.isclose(pmf[0], norm.cdf((edges[1] - location) / scale))
    assert np.isclose(pmf[-1], norm.sf((edges[-2] - location) / scale))
    assert set(pmf_summaries(pmf)) >= {
        "expected_rank",
        "median_rank",
        "interval_80_low",
        "interval_80_high",
    }


def test_larger_scale_widens_rank_interval() -> None:
    narrow = pmf_summaries(normal_pmf(0.0, 0.25, 130))
    wide = pmf_summaries(normal_pmf(0.0, 1.25, 130))
    assert (
        wide["interval_80_high"] - wide["interval_80_low"]
        > narrow["interval_80_high"] - narrow["interval_80_low"]
    )


def test_training_preprocessor_does_not_use_holdout_values() -> None:
    processor = Preprocessor.fit([{"x": 1.0}, {"x": 3.0}, {"x": None}], ["x"])
    assert processor.medians["x"] == 2.0
    assert processor.means["x"] == 2.0
    holdout = processor.transform([{"x": 1000.0}])
    assert holdout[0, 0] > 100
    assert holdout[0, 1] == 0.0


def test_team_season_scoring_is_invariant_to_duplicate_constituents() -> None:
    pmf = normal_pmf(0.0, 1.0, 20)
    ranks = np.asarray([2, 5, 7])
    assert np.isclose(
        team_log_score(pmf, ranks), team_log_score(pmf, np.repeat(ranks, 3))
    )


def test_prior_rank_quadrature_preserves_equal_weight_of_each_observation() -> None:
    processor = Preprocessor.fit([{}], [])
    model = DirectRankModel(
        [], processor, np.asarray([1.0, 0.0]), np.asarray([np.log(0.4)])
    )
    prior = rank_to_z(np.asarray([2, 9]), 10)
    actual = model.pmf({}, prior, 10)
    expected = (normal_pmf(prior[0], 0.5, 10) + normal_pmf(prior[1], 0.5, 10)) / 2
    assert np.allclose(actual, expected)


def test_quadrature_and_pmf_are_invariant_to_constituent_order() -> None:
    ranks = np.asarray([9, 1, 6, 3, 8, 2, 10, 4, 7, 5, 5, 2, 9])
    shuffled = ranks[[4, 2, 10, 1, 7, 12, 0, 9, 5, 11, 3, 8, 6]]
    assert np.array_equal(
        deterministic_quadrature(ranks, 6), deterministic_quadrature(shuffled, 6)
    )
    processor = Preprocessor.fit([{}], [])
    model = DirectRankModel(
        [], processor, np.asarray([0.8, 0.0]), np.asarray([np.log(0.4)])
    )
    assert np.allclose(
        model.pmf({}, rank_to_z(ranks, 10), 10),
        model.pmf({}, rank_to_z(shuffled, 10), 10),
    )


def test_optimizer_must_converge_and_retains_diagnostics() -> None:
    rows = [
        TeamSeason(
            2010,
            "fbs",
            str(index),
            str(index),
            20,
            np.asarray([float(index)]),
            np.asarray([float(index) / 2]),
            np.asarray([5]),
            {},
        )
        for index in range(1, 5)
    ]
    with pytest.raises(RuntimeError, match="optimizer failed"):
        DirectRankModel.fit(rows, [], optimizer_options={"maxiter": 0})
    model = DirectRankModel.fit(rows, [], optimizer_options={"maxiter": 100})
    assert model.optimizer and model.optimizer["success"] is True
    assert {"iterations", "function_evaluations", "objective"} <= set(model.optimizer)


def test_fitting_permuted_empirical_distributions_is_equivalent() -> None:
    def make_rows(reverse: bool) -> list[TeamSeason]:
        result = []
        for index, location in enumerate((-1.0, -0.3, 0.4, 1.0)):
            lag = np.asarray([location - 0.2, location, location + 0.3])
            target = np.asarray([location - 0.1, location + 0.2, location + 0.4])
            if reverse:
                lag, target = lag[::-1], target[::-1]
            result.append(
                TeamSeason(
                    2010,
                    "fbs",
                    str(index),
                    str(index),
                    20,
                    lag,
                    target,
                    np.asarray([3, 4, 5]),
                    {},
                )
            )
        return result

    first = DirectRankModel.fit(
        make_rows(False), [], optimizer_options={"maxiter": 100}
    )
    second = DirectRankModel.fit(
        make_rows(True), [], optimizer_options={"maxiter": 100}
    )
    assert np.allclose(first.beta, second.beta, atol=1e-7)
    assert np.allclose(first.gamma, second.gamma, atol=1e-7)


def test_generic_cold_start_pmf_is_valid() -> None:
    rows = [
        TeamSeason(
            2010,
            "fbs",
            "x",
            "X",
            10,
            np.asarray([0.0]),
            np.asarray([-1.0, 0.0]),
            np.asarray([2, 5]),
            {},
        )
    ]
    pmf = GenericRankPrior.fit(rows).pmf(10)
    assert pmf.shape == (10,)
    assert np.isclose(pmf.sum(), 1.0)


def test_true_t1_baseline_has_no_hidden_rank_history_features() -> None:
    values = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/build_preseason_prior.py")
    )
    assert values["SPECS"]["A_t1"] == []
    assert values["SPECS"]["A2_t1_t2"] == ["lag2_z_mean"]
    assert values["SPECS"]["A2_t1_t2_t3"] == ["lag2_z_mean", "lag3_z_mean"]


def test_season_holdout_is_disjoint_from_training() -> None:
    values = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/build_preseason_prior.py")
    )
    assert values["TEST_SEASONS"].isdisjoint(values["DEVELOPMENT_TRAIN"])
    assert values["TEST_SEASONS"].isdisjoint(values["DEVELOPMENT_VALIDATION"])


def test_coach_missingness_and_reliability_bins_are_semantic() -> None:
    values = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/build_preseason_prior.py")
    )
    coach_change = values["coach_change_value"]
    reliability = values["reliability_bins"]
    assert coach_change(None, "Coach") is None
    assert coach_change("Coach", None) is None
    assert coach_change("A", "A") == 0.0
    assert coach_change("A", "B") == 1.0
    result = reliability([0.02, 0.04, 0.31, 0.33], [0.0, 1.0, 0.0, 1.0])
    assert result["brier_score"] == pytest.approx(0.36675)
    assert result["bins"][0]["n_team_seasons"] == 2
    assert result["bins"][1]["empirical_frequency"] == 0.5


def test_fbs_cold_start_coverage_cannot_silently_drop_a_target() -> None:
    values = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/build_preseason_prior.py")
    )
    coverage = [
        {
            "season": 2025,
            "subdivision": "fbs",
            "team_id": "1",
            "team_name": "Move",
            "generated": False,
            "reason": "fcs_to_fbs_transition",
        },
        {
            "season": 2025,
            "subdivision": "fcs",
            "team_id": "2",
            "team_name": "New",
            "generated": False,
            "reason": "no_prior_rank_distribution",
        },
    ]
    values["apply_fbs_cold_start_coverage"](coverage)
    assert coverage[0]["generated"] is True
    assert coverage[0]["reason"] == "learned_fcs_to_fbs_transition"
    assert coverage[1]["generated"] is False


def test_same_population_comparison_uses_identical_team_season_keys() -> None:
    values = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/build_preseason_prior.py")
    )
    rows = [
        TeamSeason(
            2022,
            "fbs",
            "a",
            "A",
            10,
            np.asarray([0.0]),
            np.asarray([0.0]),
            np.asarray([5]),
            {},
        ),
        TeamSeason(
            2023,
            "fbs",
            "b",
            "B",
            10,
            np.asarray([0.0]),
            np.asarray([0.0]),
            np.asarray([5]),
            {},
        ),
    ]
    assert values["team_season_keys"](rows) == values["team_season_keys"](
        list(reversed(rows))
    )


def test_model_fitting_weights_each_team_season_not_constituent_count() -> None:
    def row(team_id: str, lag: float, target: float) -> TeamSeason:
        ranks = np.asarray([target])
        return TeamSeason(
            2010,
            "fbs",
            team_id,
            team_id,
            20,
            np.asarray([lag]),
            np.asarray([target]),
            ranks,
            {},
        )

    rows = [row("a", -1.0, -0.7), row("b", 0.0, -0.1), row("c", 0.8, 0.4)]
    duplicated = [
        TeamSeason(
            r.season,
            r.subdivision,
            r.team_id,
            r.team_name,
            r.population,
            r.lag1_z,
            np.repeat(r.target_z, 4),
            np.repeat(r.target_ranks, 4),
            r.features,
        )
        for r in rows
    ]
    first = DirectRankModel.fit(rows, [], penalty=0.1)
    second = DirectRankModel.fit(duplicated, [], penalty=0.1)
    assert np.allclose(first.beta, second.beta, atol=1e-5)
