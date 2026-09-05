import hashlib
from pathlib import Path

import numpy as np
import pytest

from gippyrank.preseason import TeamSeason
from gippyrank.regime_stability import (
    assert_same_keys,
    decomposition_total,
    exponential_weights,
    nested_choice,
    training_plan,
)


def row(season: int) -> TeamSeason:
    return TeamSeason(
        season,
        "fbs",
        str(season),
        str(season),
        2,
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([1]),
        {},
    )


def test_rolling_target_never_sees_its_outcome() -> None:
    plan = training_plan([row(2019), row(2020), row(2021)], 2021)
    assert [item.season for item in plan.rows] == [2019, 2020]
    with pytest.raises(ValueError, match="strictly before target"):
        exponential_weights(np.asarray([2021]), 2021, 5)


def test_recency_weights_are_monotonic_and_equal_limit_is_static() -> None:
    assert np.array_equal(
        exponential_weights(np.asarray([2018, 2019]), 2020, None), np.ones(2)
    )
    weights = exponential_weights(np.asarray([2017, 2018, 2019]), 2020, 5)
    assert weights[0] < weights[1] < weights[2]


def test_windows_exclude_old_rows_and_nested_choice_uses_only_prior_targets() -> None:
    plan = training_plan([row(2015), row(2016), row(2017), row(2018)], 2019, window=3)
    assert [item.season for item in plan.rows] == [2016, 2017, 2018]
    scores = {2018: {"equal": 2.0, "fast": 1.0}, 2019: {"equal": 0.0, "fast": 5.0}}
    assert nested_choice(["equal", "fast"], scores, 2019) == "fast"


def test_pairing_rejects_different_team_keys() -> None:
    with pytest.raises(ValueError, match="identical"):
        assert_same_keys({(2020, "fbs", "a")}, {(2020, "fbs", "b")})


def test_score_decomposition_sums_to_paired_aggregate_difference() -> None:
    h_loss = np.asarray([2.0, 4.0, 1.0])
    c_loss = np.asarray([3.5, 2.0, 1.75])
    assert np.isclose(decomposition_total(c_loss - h_loss), c_loss.sum() - h_loss.sum())


def test_frozen_2026_preseason_pmfs_are_byte_identical() -> None:
    root = Path(__file__).parents[1]
    expected = {
        "history/annual/2026/predictions.csv": "0b3454a09288019e17739869c42aed3123fdda2163694f52f63bca65baf37f90",
        "context/annual/2026/predictions.csv": "641182890ec88ea8bc6150cc97047ddc688d0c680486d9fcc63a58d7bfae9132",
    }
    for relative, digest in expected.items():
        actual = hashlib.sha256(
            (root / "data/processed/preseason" / relative).read_bytes()
        ).hexdigest()
        assert actual == digest
