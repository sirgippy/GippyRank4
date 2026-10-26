from __future__ import annotations

import copy

import pytest

from gippyrank.research.context_incremental_information import kl_posterior_from_prior
from gippyrank.research.issue_253_posterior_movement import (
    bootstrap_surprise_group_contrasts,
    build_team_checkpoint_observations,
    summarize_rank_pmf,
)


def _panel(checkpoint: int, *, games_played: int, posterior: list[float]) -> dict[str, object]:
    prior = [0.1, 0.2, 0.4, 0.3]
    return {
        "season": 2022,
        "checkpoint": checkpoint,
        "cutoff": (
            "2022-08-27T23:59:59+00:00"
            if checkpoint == 1
            else "2022-09-18T23:59:59+00:00"
        ),
        "subdivision": "fbs",
        "team_id": "1",
        "team_name": "Test Team",
        "component_status": "fitted",
        "rank_count": 4,
        "games_played": games_played,
        "history_kl_posterior_from_prior": kl_posterior_from_prior(
            prior if checkpoint == 1 else posterior, prior
        ),
        "history_prior_pmf": prior,
        "context_prior_pmf": prior,
        "history_posterior_pmf": prior if checkpoint == 1 else posterior,
        "context_posterior_pmf": prior if checkpoint == 1 else posterior,
    }


def _game(family: str, number: int, surprise: float) -> dict[str, object]:
    game_id = f"g{number}"
    kickoff_day = 3 + number * 2
    return {
        "season": 2022,
        "model_arm": f"{family}_existing",
        "prior_family": family,
        "policy": "existing",
        "team_id": "1",
        "team": "Test Team",
        "game_id": game_id,
        "forecast_id": f"{family}-{game_id}",
        "kickoff_utc": f"2022-09-{kickoff_day:02}T20:00:00+00:00",
        "cutoff_utc": f"2022-09-{kickoff_day - 2:02}T00:00:00+00:00",
        "margin_80_interval_half_width": 10.0,
        "signed_error": surprise * 10.0,
        "standardized_surprise": surprise,
    }


def _inputs() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    posterior = [0.1, 0.4, 0.4, 0.1]
    panel = [
        _panel(1, games_played=0, posterior=posterior),
        _panel(2, games_played=3, posterior=posterior),
    ]
    evidence = [
        {
            "season": 2022,
            "checkpoint": 1,
            "cutoff": panel[0]["cutoff"],
            "included_game_count": 0,
            "included_game_ids": [],
        },
        {
            "season": 2022,
            "checkpoint": 2,
            "cutoff": panel[1]["cutoff"],
            "included_game_count": 3,
            "included_game_ids": ["g1", "g2", "g3"],
        },
    ]
    games = [
        _game(family, number, surprise)
        for family in ("History", "Context")
        for number, surprise in ((1, 0.7), (2, -0.2), (3, 0.4), (4, 2.0))
    ]
    return panel, evidence, games


def test_rank_pmf_summary_uses_full_support_and_equal_tail_interval() -> None:
    summary = summarize_rank_pmf([0.0, 0.1, 0.8, 0.1], name="test")

    assert summary["rank_population"] == 4
    assert summary["rank_support_min"] == 1
    assert summary["rank_support_max"] == 4
    assert summary["expected_rank"] == pytest.approx(3.0)
    assert summary["rank_sd"] == pytest.approx(0.2**0.5)
    assert summary["central80_lower_rank"] == 2
    assert summary["central80_upper_rank"] == 3
    assert summary["central80_interval_width"] == 1
    assert summary["rank_strength"] == pytest.approx(1 / 3)


def test_checkpoint_join_uses_only_game_ids_in_actual_evidence() -> None:
    panel, evidence, games = _inputs()

    rows = build_team_checkpoint_observations(panel, evidence, games)
    history = next(
        row
        for row in rows
        if row["model_arm"] == "History_existing" and row["checkpoint"] == 2
    )

    assert history["focal_games_included_in_evidence"] == 3
    assert history["total_included_game_evidence"] == 3
    assert history["prior_surprise_game_count"] == 3
    assert history["eligible_prior_game_ids"] == "g1|g2|g3"
    assert history["accumulated_surprise"] == pytest.approx(0.3)
    assert history["surprise_group"] == "approximately_neutral"
    assert history["surprise_group_eligible"] is True
    assert history["signed_location_change_from_preseason"] == pytest.approx(0.4)
    assert history["signed_location_change_from_preseason_normalized"] == pytest.approx(0.4 / 3)
    assert history["signed_location_change_since_previous"] == pytest.approx(0.4)
    assert history["total_variation_since_previous"] == pytest.approx(0.2)
    assert len(history["prior_pmf"]) == len(history["posterior_pmf"]) == 4


def test_checkpoint_join_rejects_game_after_checkpoint_cutoff() -> None:
    panel, evidence, games = _inputs()
    future_evidence = copy.deepcopy(evidence)
    next(game for game in games if game["game_id"] == "g4")[
        "kickoff_utc"
    ] = "2022-09-20T20:00:00+00:00"
    future_evidence[1]["included_game_ids"].append("g4")
    future_evidence[1]["included_game_count"] = 4
    for row in panel:
        if row["checkpoint"] == 2:
            row["games_played"] = 4

    with pytest.raises(ValueError, match="occurs after #192 checkpoint"):
        build_team_checkpoint_observations(panel, future_evidence, games)


def test_history_kl_is_recomputed_from_the_full_pmf() -> None:
    panel, evidence, games = _inputs()
    panel[1]["history_kl_posterior_from_prior"] = 0.25

    with pytest.raises(ValueError, match="recomputed History KL differs"):
        build_team_checkpoint_observations(panel, evidence, games)


def test_cluster_bootstrap_results_are_reproducible() -> None:
    rows = []
    for family in ("History", "Context"):
        for group_index, group in enumerate(
            ("sustained_negative", "approximately_neutral", "sustained_positive")
        ):
            for replicate in range(3):
                rows.append(
                    {
                        "season": 2022,
                        "checkpoint": 4,
                        "model_arm": f"{family}_existing",
                        "team_id": str(group_index * 3 + replicate + 1),
                        "surprise_group_eligible": True,
                        "surprise_group": group,
                        "accumulated_surprise": (group_index - 1) * 0.7,
                        "signed_location_change_since_previous_normalized": (
                            group_index - 1
                        )
                        * 0.02,
                        "rank_sd_change_since_previous_normalized": -0.01,
                        "central80_width_change_since_previous_normalized": -0.02,
                    }
                )

    first = bootstrap_surprise_group_contrasts(rows, repetitions=25, seed=99)
    second = bootstrap_surprise_group_contrasts(rows, repetitions=25, seed=99)

    assert first == second
