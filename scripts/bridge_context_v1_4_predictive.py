"""Development-only Context 1.4 / 1.3 future-game predictive bridge.

The four seasons and cutoff are fixed in issue 164. This script constructs only
2022–2025 candidate priors; it does not read any 2026 candidate score.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_performance_v1 as study

from gippyrank.context_prior import AnnualFittedInstance, InferenceRow
from gippyrank.context_prior_v1_3 import (
    Context13FittedModelSource,
    ContextTransferInputProvenance,
)
from gippyrank.context_prior_v1_4_candidate import (
    COLD_START_STATUS,
    FITTED_STATUS,
    Context13FallbackSource,
    Context13PriorInput,
    candidate_spec_sha256,
    construct_candidate_prior,
)
from gippyrank.posterior.engine import Game, LikelihoodV1, Team, infer_posterior
from gippyrank.posterior.predictive import (
    ScheduledGame,
    mixture_cdf,
    posterior_prediction_teams,
    predictive_components,
    win_probabilities,
)
from gippyrank.posterior.snapshots import add_fcs_fallbacks
from gippyrank.preseason import DirectRankModel, Preprocessor

ROOT = Path(__file__).resolve().parents[1]
SEASONS = (2022, 2023, 2024, 2025)
CUTOFF_INDEX = 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_from_metadata(metadata: dict[str, object]) -> DirectRankModel:
    preprocessing = metadata["preprocessing"]
    assert isinstance(preprocessing, dict)
    return DirectRankModel(
        feature_names=list(metadata["feature_names"]),
        preprocessor=Preprocessor(
            tuple(preprocessing["feature_names"]),
            dict(preprocessing["medians"]),
            dict(preprocessing["means"]),
            dict(preprocessing["scales"]),
        ),
        beta=np.asarray(metadata["location_coefficients"], dtype=float),
        gamma=np.asarray(metadata["log_scale_coefficients"], dtype=float),
        minimum_scale=float(metadata["minimum_scale"]),
        penalty=float(metadata["penalty"]),
        optimizer=metadata["optimizer"],
        lag_count=int(metadata["lag_count"]),
        family=str(metadata["family"]),
        degrees_of_freedom=metadata["degrees_of_freedom"],
        location_feature_names=list(metadata["location_feature_names"]),
        scale_feature_names=list(metadata["scale_feature_names"]),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _development_prior_teams() -> tuple[
    dict[int, list[Team]], dict[int, list[Team]], dict[str, str]
]:
    audit = json.loads((ROOT / "config/context_v1_4_candidate.json").read_text())
    if (
        candidate_spec_sha256(audit)
        != "db84045d2f7d80b1648360da52ef1e71f76093696cad4e4e7bbdbd57a086573f"
    ):
        raise ValueError("frozen candidate identity changed")
    panel = audit["development_panel"]
    fixture_path = ROOT / panel["canonical_input_fixture"]
    if _sha256(fixture_path) != panel["canonical_input_fixture_sha256"]:
        raise ValueError("development candidate input fixture changed")
    for relative, expected in panel["source_sha256"].items():
        if _sha256(ROOT / relative) != expected:
            raise ValueError(f"development source changed: {relative}")
    fixture = json.loads(fixture_path.read_text())
    source_path = (
        ROOT
        / "data/processed/context_history_crossover/team_season_prior_decomposition.csv"
    )
    retained_path = (
        ROOT / "data/processed/context_history_crossover/hybrid_prior_results.csv"
    )
    source = {(int(r["season"]), r["team_id"]): r for r in _read_csv(source_path)}
    retained = {
        (int(r["season"]), r["team_id"]): r
        for r in _read_csv(retained_path)
        if r["arm"] == "CC"
    }
    if source.keys() != retained.keys():
        raise ValueError("development Context 1.3 source population differs")
    baseline_by_season: dict[int, list[Team]] = {}
    candidate_by_season: dict[int, list[Team]] = {}
    for season in SEASONS:
        season_fixture = fixture["seasons"][str(season)]
        model = _model_from_metadata(season_fixture["model_metadata"])
        instance = AnnualFittedInstance(**season_fixture["fitted_instance"])
        fit_source = Context13FittedModelSource.research_only(
            model, instance, fixture_id="context-v1-4-unit-test-fixture"
        )
        target_rows = {
            row["team_id"]: InferenceRow(
                season=int(row["season"]),
                subdivision=row["subdivision"],
                team_id=row["team_id"],
                team_name=row["team_name"],
                population=int(row["population"]),
                lag1_z=tuple(float(value) for value in row["lag1_z"]),
                lag_zs=tuple(tuple(values) for values in row["lag_zs"]),
                features=row["features"],
            )
            for row in season_fixture["fitted_rows"]
        }
        transfer_values = {
            row["team_id"]: {
                name: row["features"][name]
                for name in (
                    "transfer_in_prior_usage_sum",
                    "transfer_in_prior_defensive_impact_db_sum",
                    "transfer_in_prior_defensive_impact_db_available",
                )
            }
            for row in season_fixture["fitted_rows"]
        }
        ids = tuple(
            team_id for source_season, team_id in source if source_season == season
        )
        transfer = ContextTransferInputProvenance.research_only(
            season,
            feature_artifact_id=f"development_model_inputs.json/season-{season}/transfer-features",
            transfer_feature_values_by_team=transfer_values,
            source_artifact_payload=transfer_values,
            target_team_ids=ids,
        )
        baseline_teams: list[Team] = []
        candidate_teams: list[Team] = []
        for key, row in sorted(source.items()):
            if key[0] != season:
                continue
            original = np.asarray(json.loads(retained[key]["prior_pmf"]), dtype=float)
            if row["component_status"] == FITTED_STATUS:
                prior = Context13PriorInput.fitted(
                    model=model,
                    fitted_instance=instance,
                    fitted_model_source=fit_source,
                    inference_row=target_rows[key[1]],
                    transfer_provenance=transfer,
                    prior_pmf=original,
                    expected_team_id=key[1],
                    expected_target_season=season,
                )
            elif row["component_status"] == COLD_START_STATUS:
                fallback = Context13FallbackSource.from_research_fixture_csv(
                    retained_path,
                    target_season=season,
                    team_id=key[1],
                    population=int(row["target_population"]),
                )
                prior = Context13PriorInput.cold_start(
                    fitted_instance=instance,
                    fitted_model_source=fit_source,
                    transfer_provenance=transfer,
                    fallback_source=fallback,
                )
            else:
                raise ValueError(f"unknown development component status: {key}")
            candidate = construct_candidate_prior(prior)
            baseline_teams.append(Team(key[1], row["team_name"], "fbs", original))
            candidate_teams.append(Team(key[1], row["team_name"], "fbs", candidate.pmf))
        baseline_by_season[season] = baseline_teams
        candidate_by_season[season] = candidate_teams
    return (
        baseline_by_season,
        candidate_by_season,
        {
            "candidate_fixture_sha256": _sha256(fixture_path),
            "prior_decomposition_sha256": _sha256(source_path),
            "hybrid_prior_results_sha256": _sha256(retained_path),
        },
    )


def _score_bridge(
    future_games: tuple[Game, ...],
    teams: list[Team],
    posterior_pmfs: dict[str, np.ndarray],
    likelihood: LikelihoodV1,
) -> dict[str, float | int]:
    """Use the established exact mixture and three registered bridge metrics.

    These are the existing validator's NLL/MAE/Brier formulas, without seven
    interval inversions per game because this bridge does not report coverage.
    """
    prediction_teams = posterior_prediction_teams(teams, posterior_pmfs)
    total = {"n": 0, "nll": 0.0, "mae": 0.0, "brier": 0.0}
    for game in future_games:
        if game.home_id not in prediction_teams or game.away_id not in prediction_teams:
            continue
        scheduled = ScheduledGame(
            game.game_id,
            game.home_id,
            game.away_id,
            game.home_subdivision,
            game.away_subdivision,
            game.neutral_site,
        )
        locations, weights = predictive_components(
            scheduled,
            prediction_teams[game.home_id],
            prediction_teams[game.away_id],
            likelihood,
        )
        actual_margin = game.home_points - game.away_points
        density = float(
            np.dot(
                weights,
                student_t.pdf(
                    (actual_margin - locations) / likelihood.scale,
                    likelihood.degrees_of_freedom,
                )
                / likelihood.scale,
            )
        )
        home_win, _ = win_probabilities(
            mixture_cdf(
                0.0,
                locations,
                weights,
                likelihood.scale,
                likelihood.degrees_of_freedom,
            )
        )
        actual_win = 1.0 if actual_margin > 0 else 0.0 if actual_margin < 0 else 0.5
        total["n"] += 1
        total["nll"] += -float(np.log(max(density, 1e-300)))
        total["mae"] += abs(float(np.dot(weights, locations)) - actual_margin)
        total["brier"] += (home_win - actual_win) ** 2
    if total["n"] == 0:
        raise ValueError("development bridge has no eligible future games")
    return {
        name: value if name == "n" else value / total["n"]
        for name, value in total.items()
    }


def run_bridge() -> dict[str, object]:
    baseline, candidate, sources = _development_prior_teams()
    corpus = study.load_corpus(ROOT, SEASONS)
    likelihood_path = ROOT / "data/processed/posterior/historical_likelihood_v1.json"
    likelihood = study.load_likelihood(likelihood_path)
    seasons: dict[str, object] = {}
    totals = {
        arm: {name: 0.0 for name in ("n", "nll", "mae", "brier")}
        for arm in ("context_1_3", "context_1_4")
    }
    for season in SEASONS:
        dates = study.standard_cutoffs(corpus.rows_by_season[season], season)
        cutoff = dates[CUTOFF_INDEX]
        prepared = study._prepare_cutoff(ROOT, corpus, season, cutoff)
        result: dict[str, object] = {"cutoff": cutoff.isoformat()}
        expected_game_ids: set[str] | None = None
        for arm, initial in (
            ("context_1_3", baseline[season]),
            ("context_1_4", candidate[season]),
        ):
            teams = list(initial)
            team_rows: dict[str, dict[str, str]] = {}
            teams, _ = add_fcs_fallbacks(
                teams,
                team_rows,
                list(prepared.included_rows),
                prepared.fcs_population_size,
            )
            posterior = infer_posterior(
                teams,
                list(prepared.games),
                likelihood,
                max_iterations=100,
                tolerance=1e-6,
                damping=0.35,
            )
            if not posterior.converged:
                raise ValueError(
                    f"{season} {arm}: posterior inference did not converge"
                )
            ids = {
                game.game_id
                for game in prepared.future_games
                if game.home_id in posterior.pmfs and game.away_id in posterior.pmfs
            }
            if expected_game_ids is not None and ids != expected_game_ids:
                raise ValueError(f"{season}: development game rows differ between arms")
            expected_game_ids = ids
            scored = _score_bridge(
                prepared.future_games, teams, posterior.pmfs, likelihood
            )
            result[arm] = scored
            for name, value in scored.items():
                totals[arm][name] += (
                    float(value) if name == "n" else float(value) * int(scored["n"])
                )
        seasons[str(season)] = result
    for values in totals.values():
        count = int(values["n"])
        for name in values:
            if name != "n":
                values[name] /= count
        values["n"] = count
    return {
        "purpose": "development_only_predictive_bridge_no_candidate_retuning",
        "candidate_semantics_sha256": candidate_spec_sha256(),
        "cutoff_index": CUTOFF_INDEX,
        "seasons": seasons,
        "pooled": totals,
        "sources": {
            **sources,
            "historical_likelihood_sha256": _sha256(likelihood_path),
            "game_corpus_by_season": {
                str(season): [
                    {
                        "filename": path.name,
                        "sha256": _sha256(path),
                    }
                    for path in corpus.source_paths_by_season[season]
                ]
                for season in SEASONS
            },
            "uv_lock_sha256": _sha256(ROOT / "uv.lock"),
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_bridge(), indent=2, sort_keys=True))
