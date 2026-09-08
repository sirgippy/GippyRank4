import numpy as np
import pytest

from gippyrank.context_ablation import (
    aggregate_difference,
    assert_same_population,
    interaction_is_useful,
    paired_loss_differences,
    restrict_observed,
    standardized_interaction_rows,
    training_only_impute_rows,
    training_rows,
)
from gippyrank.preseason import TeamSeason


def row(season: int, team: str, left: float | None, right: float | None) -> TeamSeason:
    return TeamSeason(
        season,
        "fbs",
        team,
        team,
        2,
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([1]),
        {"left": left, "right": right},
    )


def test_raw_observed_coverage_precedes_imputation() -> None:
    rows = [row(2020, "a", 1.0, 2.0), row(2020, "b", None, 3.0)]
    observed = restrict_observed(rows, ["left", "right"])
    assert [item.team_id for item in observed] == ["a"]


def test_rolling_training_excludes_target_and_future_outcomes() -> None:
    plan = training_rows([row(2020, "a", 1, 1), row(2021, "b", 1, 1)], 2021)
    assert [item.season for item in plan] == [2020]


def test_pairing_rejects_target_population_drift() -> None:
    with pytest.raises(ValueError, match="identical keys"):
        assert_same_population([row(2020, "a", 1, 1)], [row(2020, "b", 1, 1)])


def test_interaction_is_standardized_with_training_rows_only() -> None:
    train = [row(2020, "a", 1.0, 10.0), row(2021, "b", 3.0, 30.0)]
    target = [row(2022, "c", 5.0, 50.0)]
    enriched_train, enriched_target, name = standardized_interaction_rows(
        train, target, "left", "right"
    )
    assert name == "interaction__left__right"
    # Both target standard scores are +3 under training mean/std, so product is 9.
    assert enriched_target[0].features[name] == pytest.approx(9.0)
    assert enriched_train[0].features[name] == pytest.approx(1.0)


def test_training_only_imputation_does_not_use_target_median() -> None:
    train = [row(2020, "a", 1.0, 2.0), row(2021, "b", None, 4.0)]
    target = [row(2022, "c", 1000.0, 6.0)]
    filled_train, filled_target = training_only_impute_rows(train, target, ["left"])
    assert filled_train[1].features["left"] == 1.0
    assert filled_target[0].features["left"] == 1000.0


def test_stored_per_team_difference_aggregation_is_exact() -> None:
    values = [-0.2, 0.1, 0.4]
    assert aggregate_difference(values) == pytest.approx(sum(values) / len(values))


def test_interaction_parent_pairing_and_difference_semantics() -> None:
    parent = {(2025, "fbs", "a"): (1.0, 0.1)}
    interaction = {(2025, "fbs", "a"): (1.2, 0.2)}
    difference = paired_loss_differences(parent, interaction)
    assert difference[(2025, "fbs", "a")] == pytest.approx((0.2, 0.1))
    with pytest.raises(ValueError, match="identical target keys"):
        paired_loss_differences(parent, {(2025, "fbs", "b"): (1.2, 0.2)})


def test_interaction_beating_h_but_losing_parent_is_not_useful() -> None:
    h_nll, parent_nll, interaction_nll = 3.0, 1.0, 1.2
    assert interaction_nll - h_nll < 0
    assert interaction_nll - parent_nll > 0
    assert not interaction_is_useful([interaction_nll - parent_nll])
