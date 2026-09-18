import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from investigate_transfer_production_stability import (
    C_MINUS_RP_FEATURES,
    METRICS,
    ROSTER_CONTINUITY_FEATURES,
    SELECTED_TRANSFER_FEATURES,
    candidate_definitions,
    coefficient_index,
    coefficient_rows,
    d5_frozen_parity,
    training_coverage,
)

from gippyrank.preseason import DirectRankModel, TeamSeason


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


def test_candidate_set_keeps_d5_fixed_and_does_not_search_rolling_results() -> None:
    candidates = candidate_definitions()
    assert [candidate.label for candidate in candidates] == ["R0", "R1", "R2"]
    assert candidates[1].features == C_MINUS_RP_FEATURES
    assert SELECTED_TRANSFER_FEATURES == ("transfer_in_prior_usage_sum",)
    assert candidates[2].name == "D5_total_rp_plus_incoming"
    assert candidates[2].features == (
        *C_MINUS_RP_FEATURES,
        *ROSTER_CONTINUITY_FEATURES,
    )
    assert set(ROSTER_CONTINUITY_FEATURES) <= set(candidates[2].features)


def test_training_coverage_excludes_target_and_counts_raw_observations() -> None:
    rows = [team_season(2021, "a"), team_season(2021, "b"), team_season(2022, "a")]
    features = {
        (2021, "fbs", "a"): {
            "transfer_in_prior_usage_sum": 1.0,
            "transfer_net_prior_usage": 0.5,
        },
        (2021, "fbs", "b"): {
            "transfer_in_prior_usage_sum": 1.0,
            "transfer_net_prior_usage": None,
        },
        (2022, "fbs", "a"): {
            "transfer_in_prior_usage_sum": 99.0,
            "transfer_net_prior_usage": 99.0,
        },
    }
    result = training_coverage(
        rows, features, {2021, 2022}, target_season=2022
    )
    assert result["n_training_team_seasons"] == 2
    assert result["n_transfer_covered_training_seasons"] == 1
    assert result["n_team_seasons_with_transfer_in_prior_usage_sum"] == 2
    assert result["n_team_seasons_with_all_observed_transfer_production"] == 2
    assert result["fraction_with_all_observed_transfer_production"] == 1.0


def test_coefficient_index_accounts_for_lag_and_missingness_columns() -> None:
    model = DirectRankModel(
        feature_names=["history", "returning_pct_ppa"],
        preprocessor=None,  # type: ignore[arg-type]
        beta=np.zeros(6),
        gamma=np.zeros(5),
        lag_count=1,
    )
    assert coefficient_index(model, "returning_pct_ppa") == (3, 5)


def test_coefficient_diagnostics_only_include_selected_d5_candidate() -> None:
    model = DirectRankModel(
        feature_names=list(ROSTER_CONTINUITY_FEATURES),
        preprocessor=None,  # type: ignore[arg-type]
        beta=np.zeros(5),
        gamma=np.zeros(3),
        lag_count=0,
    )
    r0, r1, r2 = candidate_definitions()
    assert coefficient_rows(model, r0, target_season=2022, previous={}) == []
    assert coefficient_rows(model, r1, target_season=2022, previous={}) == []
    assert [row["feature"] for row in coefficient_rows(
        model, r2, target_season=2022, previous={}
    )] == list(ROSTER_CONTINUITY_FEATURES)


def test_d5_frozen_parity_check_compares_all_required_metrics(tmp_path: Path) -> None:
    values = {metric: float(index + 1) for index, metric in enumerate(METRICS)}
    reference = tmp_path / "candidate_summary.csv"
    fields = ["candidate", "target_season", *METRICS]
    reference.write_text(
        ",".join(fields)
        + "\n"
        + ",".join(
            ["D5_total_rp_plus_incoming", "aggregate"]
            + [str(values[metric]) for metric in METRICS]
        )
        + "\n",
        encoding="utf-8",
    )
    result = d5_frozen_parity(values, reference_path=reference)
    assert result["passed"] is True
    assert result["max_abs_metric_delta"] == 0.0
