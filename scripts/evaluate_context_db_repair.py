"""Paired cutoff-safe predictive comparison for the repaired Context 1.4 fit."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_context_v1_4_predictive as bridge
import build_performance_v1 as study

from gippyrank.context_db_repair import read_csv, sha256
from gippyrank.posterior.engine import Game, Team, infer_posterior
from gippyrank.posterior.predictive import (
    ScheduledGame,
    mixture_cdf,
    predictive_components,
    win_probabilities,
)
from gippyrank.posterior.snapshots import add_fcs_fallbacks, load_teams

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("data/processed/context_db_repair_172")
SEASONS = (2022, 2023, 2024, 2025)
METRICS = (
    "margin_nll",
    "margin_mae",
    "win_brier",
    "coverage_50",
    "coverage_80",
    "coverage_95",
)


def summarize(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("predictive comparison has no eligible games")
    return {
        "games": len(rows),
        **{
            name: float(np.mean([float(row[name]) for row in rows])) for name in METRICS
        },
    }


def score_game(game: Game, teams: dict[str, Team], likelihood) -> dict[str, object]:
    scheduled = ScheduledGame(
        game.game_id,
        game.home_id,
        game.away_id,
        game.home_subdivision,
        game.away_subdivision,
        game.neutral_site,
    )
    locations, weights = predictive_components(
        scheduled, teams[game.home_id], teams[game.away_id], likelihood
    )
    margin = game.home_points - game.away_points
    density = float(
        np.dot(
            weights,
            student_t.pdf(
                (margin - locations) / likelihood.scale, likelihood.degrees_of_freedom
            )
            / likelihood.scale,
        )
    )
    actual_win = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
    cdf_zero = mixture_cdf(
        0.0, locations, weights, likelihood.scale, likelihood.degrees_of_freedom
    )
    cdf_actual = mixture_cdf(
        margin, locations, weights, likelihood.scale, likelihood.degrees_of_freedom
    )
    home_win, _ = win_probabilities(cdf_zero)
    result: dict[str, object] = {
        "game_id": game.game_id,
        "margin_nll": -float(np.log(max(density, 1e-300))),
        "margin_mae": abs(float(np.dot(weights, locations)) - margin),
        "win_brier": (home_win - actual_win) ** 2,
    }
    for size in (50, 80, 95):
        lower = (1 - size / 100) / 2
        result[f"coverage_{size}"] = int(lower <= cdf_actual <= 1 - lower)
    return result


def historical(root: Path = ROOT) -> dict[str, object]:
    _, old, _ = bridge._development_prior_teams()
    corpus = study.load_corpus(root, SEASONS)
    likelihood_path = root / "data/processed/posterior/historical_likelihood_v1.json"
    likelihood = study.load_likelihood(likelihood_path)
    season_results: dict[str, object] = {}
    pooled: dict[str, list[dict[str, object]]] = defaultdict(list)
    source_hashes = {}
    for season in SEASONS:
        print(f"Historical predictive replay: {season}", flush=True)
        prior_path = root / OUTPUT / "historical_priors" / f"{season}.csv"
        corrected = [
            Team(
                row["team_id"],
                row["team_name"],
                "fbs",
                np.asarray(json.loads(row["pmf"])),
            )
            for row in read_csv(prior_path)
        ]
        history, _, history_path = load_teams(root, season, "history")
        arms = {
            "current_context_1_4": old[season],
            "corrected_context_1_4": corrected,
            "history_1_1": history,
        }
        if (
            len(
                {
                    tuple(sorted(team.team_id for team in teams))
                    for teams in arms.values()
                }
            )
            != 1
        ):
            raise ValueError(f"{season}: historical prior populations differ")
        cutoff = study.standard_cutoffs(corpus.rows_by_season[season], season)[3]
        prepared = study._prepare_cutoff(root, corpus, season, cutoff)
        eligible: set[str] | None = None
        result = {
            "requested_cutoff": cutoff.isoformat(),
            "effective_cutoff": prepared.effective_cutoff.isoformat(),
            "included_game_count": len(prepared.games),
            "latest_included_start_date": max(
                (row["startDate"] for row in prepared.included_rows), default=None
            ),
        }
        for arm, initial in arms.items():
            teams = list(initial)
            teams, _ = add_fcs_fallbacks(
                teams, {}, list(prepared.included_rows), prepared.fcs_population_size
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
            prediction_teams = {
                team.team_id: Team(
                    team.team_id,
                    team.name,
                    team.subdivision,
                    posterior.pmfs[team.team_id],
                )
                for team in teams
            }
            games = [
                game
                for game in prepared.future_games
                if game.home_id in prediction_teams and game.away_id in prediction_teams
            ]
            ids = {game.game_id for game in games}
            if eligible is not None and ids != eligible:
                raise ValueError(f"{season}: game populations differ between arms")
            eligible = ids
            scored = [score_game(game, prediction_teams, likelihood) for game in games]
            result[arm] = summarize(scored)
            pooled[arm].extend(scored)
        season_results[str(season)] = result
        source_hashes[str(season)] = {
            "corrected_prior_sha256": sha256(prior_path),
            "history_prior_sha256": sha256(history_path),
            "game_source_sha256": {
                path.relative_to(root).as_posix(): sha256(path)
                for path in corpus.source_paths_by_season[season]
            },
        }
    report = {
        "cutoff_policy": "same season-local fixed midseason cutoff for all arms; only earlier final games in posterior evidence",
        "future_game_policy": "all eligible games after the cutoff; same IDs for every arm",
        "season_results": season_results,
        "pooled": {arm: summarize(rows) for arm, rows in pooled.items()},
        "source_sha256": source_hashes,
        "likelihood_sha256": sha256(likelihood_path),
    }
    path = root / OUTPUT / "historical_predictive.json"
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", action="store_true")
    args = parser.parse_args()
    if not args.historical:
        parser.error("supply --historical")
    print(json.dumps(historical()["pooled"], indent=2))


if __name__ == "__main__":
    main()
