from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_matched_history_context_issue_200 as builder
from build_preseason_prior import ColdStartSeason

from gippyrank.context_db_repair import MODEL_FEATURES
from gippyrank.posterior.engine import Game, LikelihoodV1, Team, game_factor
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


def _analytic_two_team_posterior(
    home_prior: list[float],
    away_prior: list[float],
    beta: np.ndarray,
    *,
    margin: float,
    scale: float,
    df: float,
    neutral: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    ranks = (np.arange(1, 3, dtype=float) - 0.5) / 2
    joint = np.zeros((2, 2), dtype=float)
    for home_index, home_percentile in enumerate(ranks):
        for away_index, away_percentile in enumerate(ranks):
            location = beta[0] * (home_percentile - away_percentile)
            if not neutral:
                location += beta[8]
            density = student_t.pdf(margin, df, loc=location, scale=scale)
            joint[home_index, away_index] = (
                home_prior[home_index] * away_prior[away_index] * density
            )
    joint /= joint.sum()
    return joint.sum(axis=1), joint.sum(axis=0)


def test_replay_selects_each_policy_with_distinct_nonzero_likelihoods() -> None:
    existing_beta = np.zeros(34)
    existing_beta[0], existing_beta[8] = 18.0, 7.0
    pre_bowl_beta = np.zeros(34)
    pre_bowl_beta[0], pre_bowl_beta[8] = -13.0, -5.0
    priors = []
    policy_priors = {
        "existing": {"1": [0.75, 0.25], "2": [0.30, 0.70]},
        "pre_bowl": {"1": [0.20, 0.80], "2": [0.65, 0.35]},
    }
    for policy, teams in policy_priors.items():
        for team_id, pmf in teams.items():
            priors.append(
                {
                    "season": 2025,
                    "policy": policy,
                    "model_family": "History",
                    "team_id": team_id,
                    "team_name": f"Team {team_id}",
                    "population": 2,
                    "pmf": pmf,
                }
            )
    bundle = {
        "schema_version": 1,
        "experiment": "issue-200-matched-history-context",
        "target_seasons": [2025],
        "fcs_population_by_season": {"2025": 4},
        "inference_configuration": {
            "max_iterations": 500,
            "tolerance": 1e-9,
            "damping": 0.35,
        },
        "priors": priors,
        "likelihoods": {
            "existing": {
                "beta": existing_beta.tolist(),
                "scale": 5.0,
                "degrees_of_freedom": 15.0,
            },
            "pre_bowl": {
                "beta": pre_bowl_beta.tolist(),
                "scale": 8.0,
                "degrees_of_freedom": 15.0,
            },
        },
    }
    game = Game("g1", "1", "2", "fbs", "fbs", 24, 10)
    expected = {
        policy: _analytic_two_team_posterior(
            teams["1"],
            teams["2"],
            existing_beta if policy == "existing" else pre_bowl_beta,
            margin=14,
            scale=5.0 if policy == "existing" else 8.0,
            df=15.0,
        )
        for policy, teams in policy_priors.items()
    }

    actual = {
        policy: replay_pipeline_bundle(
            bundle,
            season=2025,
            policy=policy,
            prior_family="History",
            games=[game],
        )
        for policy in ("existing", "pre_bowl")
    }

    for policy in ("existing", "pre_bowl"):
        assert actual[policy].converged
        np.testing.assert_allclose(actual[policy].pmfs["1"], expected[policy][0])
        np.testing.assert_allclose(actual[policy].pmfs["2"], expected[policy][1])
    assert not np.allclose(actual["existing"].pmfs["1"], actual["pre_bowl"].pmfs["1"])


def _analytic_cross_factor(
    fbs_population: int,
    fcs_population: int,
    beta: np.ndarray,
    *,
    margin: float,
    scale: float,
    df: float,
    home_indicator: str,
) -> np.ndarray:
    fbs_rank = (np.arange(1, fbs_population + 1, dtype=float) - 0.5) / fbs_population
    fcs_rank = (np.arange(1, fcs_population + 1, dtype=float) - 0.5) / fcs_population
    fbs_percentile, fcs_percentile = np.meshgrid(fbs_rank, fcs_rank, indexing="xy")
    location = (
        beta[9]
        + beta[10] * fbs_percentile
        + beta[11] * fcs_percentile
        + beta[23] * (home_indicator == "fbs")
        + beta[24] * (home_indicator == "fcs")
    )
    factor = student_t.pdf(margin, df, loc=location, scale=scale)
    return factor / factor.max()


def test_likelihood_game_orientation_and_neutral_site_match_frozen_semantics() -> None:
    beta = np.zeros(34)
    beta[9], beta[10], beta[11] = 1.5, 3.0, -2.0
    beta[23], beta[24] = 5.0, -4.0
    likelihood = LikelihoodV1(beta, scale=6.0, degrees_of_freedom=15.0)
    fbs = Team("f", "FBS", "fbs", np.full(3, 1 / 3))
    fcs = Team("c", "FCS", "fcs", np.full(2, 0.5))

    fbs_home = game_factor(
        Game("fbs-home", "f", "c", "fbs", "fcs", 24, 16),
        fbs,
        fcs,
        likelihood,
    )
    fcs_home = game_factor(
        Game("fcs-home", "c", "f", "fcs", "fbs", 16, 24),
        fcs,
        fbs,
        likelihood,
    )
    neutral = game_factor(
        Game("neutral", "f", "c", "fbs", "fcs", 24, 16, neutral_site=True),
        fbs,
        fcs,
        likelihood,
    )

    np.testing.assert_allclose(
        fbs_home,
        _analytic_cross_factor(
            3, 2, beta, margin=8, scale=6, df=15, home_indicator="fbs"
        ).T,
    )
    np.testing.assert_allclose(
        fcs_home,
        _analytic_cross_factor(
            3, 2, beta, margin=8, scale=6, df=15, home_indicator="fcs"
        ),
    )
    np.testing.assert_allclose(
        neutral,
        _analytic_cross_factor(
            3, 2, beta, margin=8, scale=6, df=15, home_indicator="neutral"
        ).T,
    )
    assert not np.allclose(fbs_home, neutral)
    assert not np.allclose(fcs_home, neutral.T)


def _likelihood_game(
    game_id: str,
    home_id: str,
    away_id: str,
    home_name: str,
    away_name: str,
    home_division: str,
    away_division: str,
    home_points: int,
    away_points: int,
) -> dict[str, str]:
    return {
        "id": game_id,
        "season": "2020",
        "week": "14",
        "startDate": "2020-12-01T00:00:00Z",
        "homeId": home_id,
        "awayId": away_id,
        "homeTeam": home_name,
        "awayTeam": away_name,
        "homeClassification": home_division,
        "awayClassification": away_division,
        "homePoints": str(home_points),
        "awayPoints": str(away_points),
        "neutralSite": "false",
    }


def test_actual_likelihood_builder_filters_only_fbs_postseason_games(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    games = [
        _likelihood_game("10", "1", "2", "A", "B", "fbs", "fbs", 24, 20),
        # CFBD identifies conference championships as regular-season games.
        _likelihood_game("11", "1", "3", "A", "C", "fbs", "fbs", 27, 21),
        _likelihood_game("12", "5", "6", "Army", "Navy", "fbs", "fbs", 17, 14),
        # Bowl and CFP games share the raw postseason seasonType.
        _likelihood_game("13", "1", "2", "A", "B", "fbs", "fbs", 31, 28),
        _likelihood_game("14", "1", "3", "A", "C", "fbs", "fbs", 35, 31),
        _likelihood_game("15", "4", "7", "D", "E", "fcs", "fcs", 21, 17),
    ]
    phases = {
        "10": "regular",
        "11": "regular",
        "12": "postseason",
        "13": "postseason",
        "14": "postseason",
        "15": "postseason",
    }
    team_rows = [
        _target(2020, division, team_id, [1, 2], 4)
        for division, team_id in (
            ("fbs", "1"),
            ("fbs", "2"),
            ("fbs", "3"),
            ("fbs", "5"),
            ("fbs", "6"),
            ("fcs", "4"),
            ("fcs", "7"),
        )
    ]
    distributions = [
        {
            "season": row["season"],
            "subdivision": row["subdivision"],
            "team_id": row["team_id"],
            "team_population": row["team_population"],
            "rank_observations": row["rank_observations"],
        }
        for row in team_rows
    ]
    rank_pairs = {}
    for game in games:
        home_key = (2020, game["homeClassification"], game["homeId"])
        away_key = (2020, game["awayClassification"], game["awayId"])
        rank_pairs[(int(game["id"]), home_key, away_key)] = [(1, 2)]
    fitted_matrices = []

    def fit(matrix, _margin, _weight, *, df):
        fitted_matrices.append(matrix.copy())
        return {"beta": np.zeros(matrix.shape[1]), "scale": 10.0, "df": df}

    monkeypatch.setattr(builder.historical_builder, "fit_robust_surface", fit)
    results = {
        policy: builder.build_likelihood_policy(
            policy=policy,
            game_rows=games,
            game_phases=phases,
            distributions=distributions,
            rank_pairs=rank_pairs,
        )
        for policy in ("existing", "pre_bowl")
    }
    existing_likelihood, existing_manifest, existing_included = results["existing"]
    pre_likelihood, pre_manifest, pre_included = results["pre_bowl"]

    assert [row["game_id"] for row in existing_included] == [
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
    ]
    assert [row["game_id"] for row in pre_included] == ["10", "11", "12", "15"]
    assert all(row["included_for_fit"] for row in existing_manifest)
    assert len(fitted_matrices) == 2
    assert fitted_matrices[0].shape[0] == 6
    assert fitted_matrices[1].shape[0] == 4
    pre_decisions = {row["game_id"]: row for row in pre_manifest}
    assert pre_decisions["11"]["included_for_fit"] is True
    assert pre_decisions["12"]["included_for_fit"] is True
    assert pre_decisions["13"]["exclusion_reason"] == "fbs_postseason_excluded"
    assert pre_decisions["14"]["exclusion_reason"] == "fbs_postseason_excluded"
    assert pre_decisions["15"]["included_for_fit"] is True
    assert existing_likelihood["optimizer"]["iterations"] == 8
    assert existing_likelihood["optimizer"]["convergence_assessed"] is False
    assert "converged" not in existing_likelihood["optimizer"]
    assert pre_likelihood["included_game_count"] == 4


@pytest.mark.parametrize(
    ("directory_state", "error"),
    [("missing", FileNotFoundError), ("empty", ValueError)],
)
def test_prior_fit_fails_without_required_coaching_sources(
    tmp_path: Path, directory_state: str, error: type[Exception]
) -> None:
    source_root = tmp_path / "source"
    tenure_directory = source_root / "data/raw/cfbd/preseason/coach_tenures"
    if directory_state == "empty":
        tenure_directory.mkdir(parents=True)

    with pytest.raises(error, match="coaching-tenure"):
        builder.build_prior_pipelines(
            final_targets=[],
            pre_bowl_targets=[],
            final_target_path=tmp_path / "final.csv",
            feature_path=tmp_path / "features.csv",
            source_root=source_root,
            output_root=tmp_path / "output",
        )


def test_coaching_tenures_are_loaded_from_source_root_and_manifested_per_file(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    tenure_directory = source_root / "data/raw/cfbd/preseason/coach_tenures"
    tenure_directory.mkdir(parents=True)
    files = {
        "alpha.json": [{"team": {"school": "Alpha"}, "coach": "A"}],
        "beta.json": [{"team": {"school": "Beta"}, "coach": "B"}],
    }
    for name, payload in files.items():
        (tenure_directory / name).write_text(json.dumps(payload), encoding="utf-8")
    (tenure_directory / "manifest.json").write_text(
        json.dumps({"outcomes": {"Alpha": "cached"}}), encoding="utf-8"
    )

    tenures = builder.load_required_coaching_tenures(source_root)
    inputs = builder.input_paths(tmp_path, source_root, tmp_path, tmp_path)
    tenure_inputs = {
        logical_name: path
        for logical_name, path in inputs.items()
        if logical_name.startswith("coaching_tenure_file_")
    }
    records = [
        builder.file_record(logical_name, path, source_root)
        for logical_name, path in tenure_inputs.items()
    ]

    assert set(tenures) == {"Alpha", "Beta"}
    assert "Uncovered Team" not in tenures
    assert {Path(str(record["path"])).name for record in records} == {
        *files,
        "manifest.json",
    }
    for record in records:
        source = source_root / str(record["path"])
        assert record["bytes"] == source.stat().st_size
        assert record["sha256"] == builder.sha256_file(source)


def test_rank_pair_composition_uses_each_policy_observation_dates(
    tmp_path: Path,
) -> None:
    observation_rows = {
        "existing": [
            (2003, "fbs", "20040105"),
            (2003, "fcs", "20031219"),
            (2011, "fbs", "20120109"),
            (2011, "fcs", "20120109"),
        ],
        "pre_bowl": [
            (2003, "fbs", "20031208"),
            (2003, "fcs", "20031219"),
            (2011, "fbs", "20111205"),
            (2011, "fcs", "20120109"),
        ],
    }
    dates_by_policy = {}
    for policy, rows in observation_rows.items():
        source = tmp_path / f"{policy}_observations.csv"
        source.write_text(
            "season,subdivision,ranking_date\n"
            + "".join(
                f"{season},{subdivision},{ranking_date}\n"
                for season, subdivision, ranking_date in rows
            ),
            encoding="utf-8",
        )
        dates_by_policy[policy] = builder.load_observation_snapshot_dates(
            source, required_seasons=(2003, 2011)
        )

    included_games = {
        "existing": [
            {
                "season": "2003",
                "pairing": "fbs-fcs",
                "pairing_method": "same_system",
                "rank_pairs": "[1]",
            },
            {
                "season": "2011",
                "pairing": "fbs-fcs",
                "pairing_method": "same_system",
                "rank_pairs": "[1]",
            },
            {
                "season": "2011",
                "pairing": "fbs-fbs",
                "pairing_method": "same_system",
                "rank_pairs": "[1]",
            },
        ],
        "pre_bowl": [
            {
                "season": "2003",
                "pairing": "fbs-fcs",
                "pairing_method": "same_system",
                "rank_pairs": "[1]",
            },
            {
                "season": "2011",
                "pairing": "fbs-fcs",
                "pairing_method": "same_system",
                "rank_pairs": "[1]",
            },
        ],
    }

    result = builder.build_rank_pair_composition(included_games, dates_by_policy)
    by_key = {(row["policy"], row["season"], row["pairing"]): row for row in result}

    existing_2003 = by_key[("existing", 2003, "fbs-fcs")]
    assert existing_2003["fbs_target_snapshot_date"] == "2004-01-05"
    assert existing_2003["fcs_target_snapshot_date"] == "2003-12-19"
    assert existing_2003["cross_subdivision_dates_differ"] is True
    existing_2011 = by_key[("existing", 2011, "fbs-fcs")]
    assert existing_2011["fbs_target_snapshot_date"] == "2012-01-09"
    assert existing_2011["fcs_target_snapshot_date"] == "2012-01-09"
    assert existing_2011["cross_subdivision_dates_differ"] is False
    assert (
        by_key[("existing", 2011, "fbs-fbs")]["cross_subdivision_dates_differ"] is False
    )
    pre_bowl_2011 = by_key[("pre_bowl", 2011, "fbs-fcs")]
    assert pre_bowl_2011["fbs_target_snapshot_date"] == "2011-12-05"
    assert pre_bowl_2011["fcs_target_snapshot_date"] == "2012-01-09"
    assert pre_bowl_2011["cross_subdivision_dates_differ"] is True


def test_policy_intervention_diagnostics_compare_fitted_arms_separately() -> None:
    final_rows = [
        _target(season, "fbs", "1", [1, 3], 4) for season in range(2022, 2026)
    ]
    pre_bowl_rows = [
        _target(season, "fbs", "1", [2, 3], 4) for season in range(2022, 2026)
    ]
    final_targets = read_response_targets(final_rows, label="final")
    pre_bowl_targets = read_response_targets(pre_bowl_rows, label="pre-bowl")
    priors = [
        {
            "season": season,
            "policy": policy,
            "model_family": family,
            "team_id": "1",
            "population": 2,
            "pmf": [0.5, 0.5] if policy == "existing" else [0.25, 0.75],
        }
        for season in range(2022, 2026)
        for family in ("History", "Context")
        for policy in ("existing", "pre_bowl")
    ]
    model = {
        "location_coefficients": [1.0, 2.0],
        "log_scale_coefficients": [0.1],
    }
    changed_model = {
        "location_coefficients": [3.0, 2.0],
        "log_scale_coefficients": [0.3],
    }
    fitted_models = {
        "rolling_models": {
            str(season): {
                policy: {
                    "history": {"model": model, "transition_model": None},
                    "context": {"model": model},
                }
                for policy in ("existing", "pre_bowl")
            }
            for season in range(2022, 2026)
        }
    }
    for season in range(2022, 2026):
        fitted_models["rolling_models"][str(season)]["pre_bowl"]["history"]["model"] = (
            changed_model
        )
        fitted_models["rolling_models"][str(season)]["pre_bowl"]["context"]["model"] = (
            changed_model
        )
    likelihoods = {
        "existing": {
            "beta": [1.0, -2.0],
            "scale": 10.0,
            "included_game_count": 3,
            "rank_pair_pseudo_observations": 6,
        },
        "pre_bowl": {
            "beta": [2.0, -2.0],
            "scale": 8.0,
            "included_game_count": 2,
            "rank_pair_pseudo_observations": 5,
        },
    }
    composition = [
        {
            "policy": "existing",
            "pairing": "fbs-fbs",
            "pairing_method": "same_system",
            "games": 3,
            "rank_pair_rows": 6,
        },
        {
            "policy": "pre_bowl",
            "pairing": "fbs-fbs",
            "pairing_method": "same_system",
            "games": 2,
            "rank_pair_rows": 5,
        },
    ]

    result = builder.policy_intervention_diagnostics(
        priors,
        final_targets,
        pre_bowl_targets,
        fitted_models,
        likelihoods,
        composition,
    )

    assert len(result["prior_pmf_differences"]) == 8
    assert all(
        row["mean_total_variation_distance"] == pytest.approx(0.25)
        and row["materially_changed_team_count"] == 1
        for row in result["prior_pmf_differences"]
    )
    assert all(
        row["mean_total_variation_distance"] == pytest.approx(0.5)
        and row["changed_distribution_team_count"] == 1
        for row in result["fbs_target_distribution_differences"]
    )
    assert all(
        row["mean_absolute_difference"] == pytest.approx(2.2 / 3)
        and row["maximum_absolute_difference"] == pytest.approx(2.0)
        for row in result["fitted_coefficient_differences"]
    )
    likelihood_diff = result["likelihood_differences"]
    assert likelihood_diff["mean_absolute_coefficient_difference"] == pytest.approx(0.5)
    assert likelihood_diff["absolute_scale_difference"] == pytest.approx(2.0)
    assert likelihood_diff["pre_bowl_minus_existing_included_game_count"] == -1
    assert (
        likelihood_diff["rank_pair_composition"][0][
            "pre_bowl_minus_existing_rank_pair_rows"
        ]
        == -1
    )
