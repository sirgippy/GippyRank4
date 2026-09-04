import numpy as np

from gippyrank.preseason import clean_name, normal_pmf, pmf_summaries


def test_team_name_cleaning_is_conservative_and_joinable() -> None:
    assert clean_name("  Texas  A&M ") == "texas aandm"


def test_rank_pmf_is_full_support_and_normalized() -> None:
    pmf = normal_pmf(0.25, 0.08, 10, draws=20_000, seed=7)
    assert pmf.shape == (10,)
    assert np.isclose(pmf.sum(), 1.0)
    assert np.all(pmf >= 0)
    assert set(pmf_summaries(pmf)) >= {
        "expected_rank",
        "median_rank",
        "interval_80_low",
        "interval_80_high",
        "top5_probability",
        "top10_probability",
        "top25_probability",
    }
