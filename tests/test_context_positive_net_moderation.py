"""Invariants for the issue 158 frozen rolling-origin location experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from gippyrank.context_positive_net_moderation import (
    ALPHAS,
    LocationParts,
    moderate_positive_net,
    moderated_location_points,
    parts_from_fitted_contributions,
    require_rolling_origin,
)
from gippyrank.context_prior_v1_3 import H_FEATURES, LOCATION_FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_context_positive_net_moderation as experiment


def test_positive_only_semantics_and_center_reconstruction() -> None:
    assert ALPHAS == (1.0, 0.75, 0.5, 0.25, 0.0)
    for value in (-2.0, 0.0, 2.0):
        parts = LocationParts(0.5, -0.25, value, np.array([value - 0.75, value + 1.25]))
        for alpha in ALPHAS:
            expected = value if value <= 0 else alpha * value
            assert moderate_positive_net(value, alpha) == expected
            points = moderated_location_points(parts, alpha)
            assert np.mean(points) == pytest.approx(0.5 - 0.25 + expected, abs=1e-12)
            np.testing.assert_allclose(
                points - points.mean(),
                parts.conditional_location_points
                - parts.conditional_location_points.mean(),
                rtol=0,
                atol=1e-12,
            )
            if alpha == 1 or value <= 0:
                np.testing.assert_array_equal(points, parts.conditional_location_points)
    with pytest.raises(ValueError, match="alpha"):
        moderate_positive_net(2.0, 1.1)
    with pytest.raises(ValueError, match="reconstruct"):
        LocationParts(0.0, 0.0, 1.0, np.array([0.0, 0.0]))


def test_contribution_projection_cannot_read_outcome_columns() -> None:
    row = {"context_location_intercept": "0.1", "lag1_contribution": "0.2"}
    for name in LOCATION_FEATURE_NAMES:
        row[f"feature_{name}_contribution"] = "0.1" if name in H_FEATURES else "0.2"
        row[f"feature_{name}_missing_contribution"] = "0"
    history = 0.2 + 0.1 * len(H_FEATURES)
    context = 0.2 * (len(LOCATION_FEATURE_NAMES) - len(H_FEATURES))
    points = np.array([0.1 + history + context - 0.3, 0.1 + history + context + 0.3])
    first = parts_from_fitted_contributions(row, points)
    row["context_final_nll"] = "999999"
    row["context_minus_history_final_nll"] = "-999999"
    second = parts_from_fitted_contributions(row, points)
    assert first.history_derived_subtotal == second.history_derived_subtotal
    assert first.context_only_subtotal == second.context_only_subtotal
    np.testing.assert_array_equal(
        moderated_location_points(first, 0.5), moderated_location_points(second, 0.5)
    )


def test_rolling_origin_is_fail_closed() -> None:
    context = {
        "model_family": "context_prior",
        "spec_version": "1.3",
        "target_season": 2024,
        "trained_through_season": 2023,
    }
    history = {
        "model_family": "history_prior",
        "spec_version": "1.1",
        "target_season": 2024,
        "trained_through_season": 2023,
    }
    require_rolling_origin(2024, 2023, 2023, context, history)
    with pytest.raises(ValueError, match="not rolling-origin"):
        require_rolling_origin(2024, 2024, 2023, context, history)
    with pytest.raises(ValueError, match="origin or version"):
        require_rolling_origin(
            2024, 2023, 2023, {**context, "trained_through_season": 2024}, history
        )


def test_frozen_alpha_one_prior_parity_and_fallback_immutability() -> None:
    priors, parts, _, _, _, parity = experiment.load_frozen_priors()
    assert len(parts) == 528
    assert parity["max_prior_pmf_error"] <= experiment.PMF_TOLERANCE
    for season in experiment.SEASONS:
        for team_id, baseline in priors[season]["context_1_3"].items():
            np.testing.assert_array_equal(
                priors[season]["alpha_1.00"][team_id], baseline
            )
            if (season, team_id) not in parts or parts[
                season, team_id
            ].context_only_subtotal <= 0:
                for alpha in ALPHAS:
                    np.testing.assert_array_equal(
                        priors[season][f"alpha_{alpha:.2f}"][team_id], baseline
                    )


def test_pooling_uses_team_season_weights_after_season_separation() -> None:
    rows = []
    for season, values in (
        (2022, [9.0]),
        (2023, [1.0]),
        (2024, [2.0]),
        (2025, [3.0, 6.0]),
    ):
        for index, value in enumerate(values):
            row = {
                "season": season,
                "checkpoint": 7,
                "period": "december",
                "cutoff": f"{season}-12-01",
                "team_id": str(index),
                "model": "alpha_0.50",
                "alpha": "0.50",
                "posterior_nll_delta_vs_context": value,
                "posterior_nll_delta_vs_history": value + 0.5,
                "prior_nll_delta_vs_context": value,
                "prior_expected_rank_error_delta_vs_context": value,
            }
            for stage in ("prior", "posterior"):
                for metric in (
                    "nll",
                    "crps",
                    "expected_rank_error",
                    "interval_80_width",
                    "interval_80_target_mass",
                ):
                    row[f"{stage}_{metric}"] = value
            rows.append(row)
    summary = {
        (row["cohort"], row["checkpoint"]): row
        for row in experiment.summarize_checkpoint(rows)
    }
    assert summary["2022", 7]["team_seasons"] == 1
    assert summary["2023-2025", 7]["team_seasons"] == 4
    assert summary["2023-2025", 7]["mean_posterior_nll"] == pytest.approx(3.0)
    assert summary["2023-2025", 7]["mean_posterior_nll"] != pytest.approx(
        (1 + 2 + 4.5) / 3
    )


def test_committed_parity_and_artifact_hashes() -> None:
    output = ROOT / "data/processed/context_positive_net_moderation"
    summary = json.loads((output / "summary.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    assert summary["parity"]["max_posterior_pmf_error"] == 0
    assert summary["parity"]["max_retained_team_metric_error"] == 0
    assert summary["parity"]["max_aggregate_metric_error"] == 0
    assert len(experiment.read_csv(output / "checkpoint_results.csv")) == 245
    assert len(experiment.read_csv(output / "team_season_results.csv")) == 3738
    assert provenance["script_sha256"] == experiment.crossover.sha256(
        ROOT / "scripts/experiment_context_positive_net_moderation.py"
    )
    for name, expected in provenance["output_hashes"].items():
        assert experiment.crossover.sha256(output / name) == expected


def test_prior_construction_leaves_frozen_baselines_unchanged() -> None:
    paths = (
        ROOT / "data/processed/context_history_crossover/hybrid_prior_results.csv",
        ROOT
        / "data/processed/context_history_crossover/hybrid_posterior_team_results.csv",
        ROOT / "data/processed/preseason/context_v1_3_candidate/predictions.csv",
        ROOT / "data/processed/preseason/history/predictions.csv",
    )
    before = {path: experiment.crossover.sha256(path) for path in paths}
    experiment.load_frozen_priors()
    assert {path: experiment.crossover.sha256(path) for path in paths} == before
