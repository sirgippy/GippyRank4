import runpy
from pathlib import Path

import numpy as np
from scipy.stats import norm

from gippyrank.preseason import (
    DirectRankModel,
    Preprocessor,
    TeamSeason,
    clean_name,
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
