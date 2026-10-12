from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_matched_history_context_issue_200 as builder
from build_preseason_prior import ColdStartSeason

from gippyrank.context_db_repair import MODEL_FEATURES
from gippyrank.posterior.engine import Game
from gippyrank.preseason import TeamSeason
from gippyrank.research.matched_history_context_issue_200 import (
    apply_fbs_cold_response_targets,
    apply_fbs_response_targets,
    attach_repaired_context_rows,
    common_fbs_keys,
    predictor_payload,
    read_response_targets,
    validate_fcs_targets_unchanged,
)
from gippyrank.research.matched_pipeline_replay import (
    read_pipeline_bundle,
    replay_pipeline_bundle,
)


def _target(
    season: int,
    subdivision: str,
    team_id: str,
    ranks: list[int],
    population: int,
) -> dict[str, str]:
    counts = Counter(ranks)
    return {
        "season": str(season),
        "subdivision": subdivision,
        "team_id": team_id,
        "team_name": f"Team {team_id}",
        "team_population": str(population),
        "usable_systems": str(len(ranks)),
        "rank_observations": json.dumps(ranks),
        "pmf": json.dumps(
            [
                {
                    "rank": rank,
                    "count": count,
                    "probability": count / len(ranks),
                }
                for rank, count in sorted(counts.items())
            ]
        ),
    }


def test_response_substitution_changes_labels_but_preserves_predictors() -> None:
    final = read_response_targets([_target(2024, "fbs", "1", [2, 2], 4)], label="final")
    pre_bowl = read_response_targets(
        [_target(2024, "fbs", "1", [1, 3], 3)], label="pre-bowl"
    )
    matched = common_fbs_keys(final, pre_bowl)
    row = TeamSeason(
        season=2024,
        subdivision="fbs",
        team_id="1",
        team_name="Team 1",
        population=4,
        lag1_z=np.asarray([-0.5, 0.5]),
        target_z=np.asarray([0.0, 0.0]),
        target_ranks=np.asarray([2, 2]),
        features={"lag2_z_mean": 0.1},
    )

    changed = apply_fbs_response_targets([row], pre_bowl, matched_keys=matched)[0]

    assert predictor_payload(changed) == predictor_payload(row)
    assert changed.population == 3
    assert changed.target_ranks.tolist() == [1, 3]


def test_cold_start_response_substitution_preserves_fcs_transition_lag() -> None:
    target = read_response_targets(
        [_target(2024, "fbs", "1", [1, 3], 4)], label="pre-bowl"
    )
    row = ColdStartSeason(
        season=2024,
        subdivision="fbs",
        team_id="1",
        team_name="Team 1",
        population=4,
        target_z=np.asarray([0.0]),
        target_ranks=np.asarray([2]),
        cross_subdivision_lag_z=np.asarray([-1.0, 1.0]),
        reason="fcs_to_fbs_transition",
    )

    changed = apply_fbs_cold_response_targets(
        [row], target, matched_keys={(2024, "fbs", "1")}
    )[0]

    np.testing.assert_array_equal(changed.cross_subdivision_lag_z, [-1.0, 1.0])
    assert changed.target_ranks.tolist() == [1, 3]


def test_fcs_targets_must_remain_identical_between_policies() -> None:
    final = read_response_targets(
        [
            _target(2024, "fcs", "10", [1, 3], 4),
            _target(2024, "fbs", "1", [2, 2], 4),
        ],
        label="final",
    )
    pre_bowl = read_response_targets(
        [
            _target(2024, "fcs", "10", [1, 3], 4),
            _target(2024, "fbs", "1", [1, 3], 4),
        ],
        label="pre-bowl",
    )

    validate_fcs_targets_unchanged(final, pre_bowl)
    pre_bowl[(2024, "fcs", "10")] = read_response_targets(
        [_target(2024, "fcs", "10", [2, 4], 4)], label="changed FCS"
    )[(2024, "fcs", "10")]
    with pytest.raises(ValueError, match="FCS response policy changed"):
        validate_fcs_targets_unchanged(final, pre_bowl)


def test_missing_repaired_context_source_stays_missing() -> None:
    values = {name: 0.5 for name in MODEL_FEATURES}
    row = TeamSeason(
        season=2008,
        subdivision="fbs",
        team_id="98",
        team_name="Western Kentucky",
        population=120,
        lag1_z=np.asarray([0.0]),
        target_z=np.asarray([0.0]),
        target_ranks=np.asarray([60]),
        features=values,
    )

    attached, missing = attach_repaired_context_rows([row], {})

    assert missing == [(2008, "98")]
    assert attached[0].features["transfer_in_prior_usage_sum"] is None
    assert attached[0].features["observed_db_impact_sum"] is None
    assert attached[0].features["db_impact_coverage_fraction"] is None
    assert attached[0].features["long_run_z_mean"] == 0.5


@pytest.mark.parametrize(
    ("policy", "game", "phase", "expected"),
    [
        (
            "pre_bowl",
            {
                "homeClassification": "fbs",
                "awayClassification": "fbs",
                "homeTeam": "A",
                "awayTeam": "B",
            },
            "postseason",
            "fbs_postseason_excluded",
        ),
        (
            "pre_bowl",
            {
                "homeClassification": "fbs",
                "awayClassification": "fbs",
                "homeTeam": "Army",
                "awayTeam": "Navy",
            },
            "postseason",
            "",
        ),
        (
            "pre_bowl",
            {
                "homeClassification": "fcs",
                "awayClassification": "fcs",
                "homeTeam": "A",
                "awayTeam": "B",
            },
            "postseason",
            "",
        ),
        (
            "existing",
            {
                "homeClassification": "fbs",
                "awayClassification": "fbs",
                "homeTeam": "A",
                "awayTeam": "B",
            },
            "postseason",
            "",
        ),
    ],
)
def test_pre_bowl_game_policy(
    policy: str, game: dict[str, str], phase: str, expected: str
) -> None:
    assert (
        builder._game_exclusion_reason(policy, game, phase, has_joined_rank_row=True)
        == expected
    )


def test_game_without_joined_rank_row_is_recorded_as_missing() -> None:
    game = {
        "homeClassification": "fbs",
        "awayClassification": "fbs",
        "homeTeam": "A",
        "awayTeam": "B",
    }
    assert (
        builder._game_exclusion_reason(
            "pre_bowl", game, "regular", has_joined_rank_row=False
        )
        == "missing_rank_target_or_score"
    )


def test_replay_uses_selected_prior_likelihood_and_uniform_fcs_fallback(
    tmp_path: Path,
) -> None:
    priors = []
    for policy in ("existing", "pre_bowl"):
        for family, pmf in (("History", [0.1, 0.6, 0.3]), ("Context", [0.7, 0.2, 0.1])):
            priors.append(
                {
                    "season": 2025,
                    "policy": policy,
                    "model_family": family,
                    "team_id": "1",
                    "team_name": "Team 1",
                    "population": 3,
                    "pmf": pmf,
                }
            )
    bundle = {
        "schema_version": 1,
        "experiment": "issue-200-matched-history-context",
        "target_seasons": [2025],
        "fcs_population_by_season": {"2025": 4},
        "inference_configuration": {
            "max_iterations": 10,
            "tolerance": 1e-9,
            "damping": 0.35,
        },
        "priors": priors,
        "likelihoods": {
            policy: {
                "beta": [0.0] * 34,
                "scale": 1.0,
                "degrees_of_freedom": 15.0,
            }
            for policy in ("existing", "pre_bowl")
        },
    }
    path = tmp_path / "pipeline_bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    loaded = read_pipeline_bundle(path)
    result = replay_pipeline_bundle(
        loaded,
        season=2025,
        policy="pre_bowl",
        prior_family="Context",
        games=[Game("g1", "1", "10", "fbs", "fcs", 21, 14)],
    )

    assert result.converged
    np.testing.assert_allclose(result.pmfs["1"], [0.7, 0.2, 0.1])
    np.testing.assert_allclose(result.pmfs["10"], np.full(4, 0.25))
