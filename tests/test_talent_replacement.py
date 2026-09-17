import numpy as np
import pytest

from gippyrank.preseason import TeamSeason
from gippyrank.talent_replacement import (
    add_seasonal_talent_delta,
    season_talent_statistics,
)


def team_season(season: int, team_id: str) -> TeamSeason:
    return TeamSeason(
        season,
        "fbs",
        team_id,
        team_id,
        2,
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([1]),
        {},
    )


def test_season_talent_statistics_use_population_scale() -> None:
    values = {
        (2021, "fbs", "a"): 10.0,
        (2021, "fbs", "b"): 14.0,
    }
    assert season_talent_statistics(values)[(2021, "fbs")] == pytest.approx((12.0, 2.0))


def test_talent_delta_is_current_and_previous_season_relative_z_difference() -> None:
    values = {
        (2021, "fbs", "a"): 10.0,
        (2021, "fbs", "b"): 14.0,
        (2022, "fbs", "a"): 24.0,
        (2022, "fbs", "b"): 20.0,
    }
    rows = add_seasonal_talent_delta([team_season(2022, "a")], values)
    # 2021 z(a)=-1, 2022 z(a)=+1, so the change is +2.
    assert rows[0].features["talent_delta"] == pytest.approx(2.0)


def test_talent_delta_is_missing_when_previous_source_value_is_unavailable() -> None:
    values = {
        (2022, "fbs", "a"): 24.0,
        (2022, "fbs", "b"): 20.0,
    }
    rows = add_seasonal_talent_delta([team_season(2022, "a")], values)
    assert rows[0].features["talent_delta"] is None
