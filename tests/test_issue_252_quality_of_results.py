from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from gippyrank.posterior.engine import LikelihoodV1, Team
from gippyrank.research.issue_252_quality_of_results import (
    SITE_SLOTS,
    OpponentSlot,
    _pmf_distances,
    expected_focal_win_probability,
    generate_trial,
    refresh_report_from_artifacts,
    summarize_pairs,
)


def _likelihood() -> LikelihoodV1:
    return LikelihoodV1(np.zeros(34), scale=7.0, degrees_of_freedom=15.0)


def _schedules() -> dict[str, tuple[OpponentSlot, ...]]:
    opponents = [
        Team(f"opponent-{index}", f"Opponent {index}", "fbs", np.array([0.3, 0.4, 0.3]))
        for index in range(10)
    ]
    schedule = tuple(
        OpponentSlot(team, venue, 0.5, 0.5)
        for team, venue in zip(opponents, SITE_SLOTS, strict=True)
    )
    return {
        "strong_opponents": schedule,
        "approximately_even": schedule,
        "weaker_opponents": schedule,
    }


def test_expected_win_probability_integrates_uncertain_rank_pmfs() -> None:
    likelihood = _likelihood()
    focal = np.array([0.1, 0.6, 0.3])
    opponent = np.array([0.5, 0.2, 0.3])

    for venue in ("home", "away", "neutral"):
        assert np.isclose(
            expected_focal_win_probability(focal, opponent, venue, likelihood),
            0.5,
            atol=1e-12,
        )


def test_matched_season_generation_is_reproducible_and_site_aligned() -> None:
    focal_prior = np.array([0.2, 0.5, 0.3])
    schedules = _schedules()
    first = generate_trial(12345, focal_prior, schedules, _likelihood())
    second = generate_trial(12345, focal_prior, schedules, _likelihood())

    for group in first.groups:
        first_margins = [game.home_points for game in first.groups[group].games]
        second_margins = [game.home_points for game in second.groups[group].games]
        assert first_margins == second_margins
    even = first.groups["approximately_even"]
    weak = first.groups["weaker_opponents"]
    assert [game.home_points for game in even.games] == [
        game.home_points for game in weak.games
    ]
    assert [game.neutral_site for game in even.games] == [
        site == "neutral" for site in SITE_SLOTS
    ]
    assert first.groups["same_schedule_a"].wins != first.groups["same_schedule_b"].wins


def test_sensitivity_trial_includes_the_strong_opponent_schedule() -> None:
    trial = generate_trial(
        252,
        np.array([0.2, 0.5, 0.3]),
        _schedules(),
        _likelihood(),
        include_control=False,
        include_strong=True,
    )
    assert len(trial.groups["strong_opponents"].games) == 10


def test_pair_summary_names_each_uncertainty_difference() -> None:
    row = {
        "comparison": "schedule_pair",
        "margin_weight": 0.25,
        "rank_difference": -1.0,
        "rank_sd_difference": 2.0,
        "entropy_difference_bits": 0.1,
        "interval_80_width_difference": 3,
        "pmf_total_variation": 0.2,
        "pmf_wasserstein_rank_distance": 4.0,
        "right_outranks_left": True,
    }
    summary = summarize_pairs([row], "unconditional")[0]
    assert summary["entropy_bits_mean"] == 0.1
    assert summary["pmf_wasserstein_mean"] == 4.0


def test_posterior_distance_metrics_match_discrete_rank_definitions() -> None:
    left = np.array([0.0, 1.0, 0.0])
    right = np.array([1.0, 0.0, 0.0])
    total_variation, wasserstein = _pmf_distances(left, right)
    assert total_variation == 1.0
    assert wasserstein == 1.0


def test_report_only_refresh_preserves_simulation_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source_artifacts = root / "data/research/issue_252_quality_of_results"
    output_dir = tmp_path / "artifacts"
    shutil.copytree(source_artifacts, output_dir)
    original_manifest = json.loads((output_dir / "manifest.json").read_text())
    original_summary_hash = hashlib.sha256(
        (output_dir / "simulation_summary.csv").read_bytes()
    ).hexdigest()

    refreshed = refresh_report_from_artifacts(root, output_dir)

    assert refreshed["report_generation"]["simulation_rerun"] is False
    assert refreshed["simulation_provenance"] == original_manifest.get(
        "simulation_provenance",
        {
            "source_revision": original_manifest["source_revision"],
            "source_code_sha256": original_manifest["source_code_sha256"],
        },
    )
    assert all(
        refreshed["artifact_sha256"][name] == expected
        for name, expected in original_manifest["artifact_sha256"].items()
        if name != "report.md"
    )
    assert (
        refreshed["artifact_sha256"]["report.md"]
        == hashlib.sha256((output_dir / "report.md").read_bytes()).hexdigest()
    )
    assert (
        hashlib.sha256((output_dir / "simulation_summary.csv").read_bytes()).hexdigest()
        == original_summary_hash
    )
    report = (output_dir / "report.md").read_text()
    assert "not evidence of equivalence" in report
    assert "not selected on their own records" in report
    assert "cannot be attributed exclusively to the strength" in report
