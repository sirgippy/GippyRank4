"""Run a compact leakage-safe validation of exact predictive game summaries."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_performance_v1 as study

from gippyrank.posterior.engine import LikelihoodV1, Team
from gippyrank.posterior.predictive import (
    ScheduledGame,
    predict_game,
    predictive_components,
)

ROOT = Path(__file__).resolve().parents[1]


def _score_model(
    run: study.CutoffRun,
    model: str,
    inference: study.PerformanceInference,
    likelihood: LikelihoodV1,
) -> dict[str, float | int]:
    posterior = inference.anchor_result.pmfs
    teams = {team.team_id: team for team in inference.teams}
    totals = {
        "n": 0,
        "mae": 0.0,
        "nll": 0.0,
        "brier": 0.0,
        "coverage_50": 0,
        "coverage_80": 0,
        "coverage_95": 0,
    }
    for game in run.prepared.future_games:
        if game.home_id not in posterior or game.away_id not in posterior:
            continue
        home_base, away_base = teams[game.home_id], teams[game.away_id]
        home = Team(
            home_base.team_id,
            home_base.name,
            home_base.subdivision,
            posterior[game.home_id],
        )
        away = Team(
            away_base.team_id,
            away_base.name,
            away_base.subdivision,
            posterior[game.away_id],
        )
        scheduled = ScheduledGame(
            game.game_id,
            game.home_id,
            game.away_id,
            game.home_subdivision,
            game.away_subdivision,
            game.neutral_site,
        )
        summary = predict_game(scheduled, home, away, likelihood)
        locations, weights = predictive_components(scheduled, home, away, likelihood)
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
        actual_win = 1.0 if actual_margin > 0 else 0.0 if actual_margin < 0 else 0.5
        totals["n"] += 1
        totals["mae"] += abs(summary.expected_home_margin - actual_margin)
        totals["nll"] += -np.log(max(density, 1e-300))
        totals["brier"] += (summary.home_win_probability - actual_win) ** 2
        totals["coverage_50"] += int(
            summary.margin_interval_50[0]
            <= actual_margin
            <= summary.margin_interval_50[1]
        )
        totals["coverage_80"] += int(
            summary.margin_interval_80[0]
            <= actual_margin
            <= summary.margin_interval_80[1]
        )
        totals["coverage_95"] += int(
            summary.margin_interval_95[0]
            <= actual_margin
            <= summary.margin_interval_95[1]
        )

    count = int(totals["n"])
    if count == 0:
        raise ValueError(f"{model}: no eligible future game instances were scored")
    for field in ("mae", "nll", "brier", "coverage_50", "coverage_80", "coverage_95"):
        totals[field] = float(totals[field]) / count
    return totals


def validate(*, cutoff_index: int) -> dict[str, object]:
    likelihood = study.load_likelihood(
        ROOT / "data/processed/posterior/historical_likelihood_v1.json"
    )
    corpus = study.load_corpus(ROOT, study.EVALUATION_SEASONS)
    aggregate = {
        "Predictive_C": {
            "n": 0,
            "mae": 0.0,
            "nll": 0.0,
            "brier": 0.0,
            "coverage_50": 0.0,
            "coverage_80": 0.0,
            "coverage_95": 0.0,
        },
        "Predictive_H": {
            "n": 0,
            "mae": 0.0,
            "nll": 0.0,
            "brier": 0.0,
            "coverage_50": 0.0,
            "coverage_80": 0.0,
            "coverage_95": 0.0,
        },
    }
    cutoffs: dict[str, str] = {}
    for season in study.EVALUATION_SEASONS:
        dates = study.standard_cutoffs(corpus.rows_by_season[season], season)
        if cutoff_index < 0 or cutoff_index >= len(dates):
            raise ValueError(f"cutoff index {cutoff_index} is unavailable for {season}")
        cutoff = dates[cutoff_index]
        cutoffs[str(season)] = cutoff.isoformat()
        run = study.build_cutoff_run(
            ROOT,
            corpus,
            season,
            cutoff,
            likelihood,
            "prior_stripping",
            cutoff_index=cutoff_index,
            cutoff_count=len(dates),
            max_iterations=100,
            tolerance=1e-6,
            damping=0.35,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            scored = executor.map(
                _score_model,
                (run, run),
                ("Predictive_C", "Predictive_H"),
                (run.context, run.history),
                (likelihood, likelihood),
            )
            for model, values in zip(
                ("Predictive_C", "Predictive_H"), scored, strict=True
            ):
                aggregate[model]["n"] += values["n"]
                for field in (
                    "mae",
                    "nll",
                    "brier",
                    "coverage_50",
                    "coverage_80",
                    "coverage_95",
                ):
                    aggregate[model][field] += values[field] * values["n"]

    for values in aggregate.values():
        count = values["n"]
        for field in (
            "mae",
            "nll",
            "brier",
            "coverage_50",
            "coverage_80",
            "coverage_95",
        ):
            values[field] = float(values[field]) / count
    return {
        "cutoff_index": cutoff_index,
        "cutoff_semantics": "standard mid-season cutoff per season",
        "cutoffs": cutoffs,
        "metrics": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff-index", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(cutoff_index=args.cutoff_index)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
