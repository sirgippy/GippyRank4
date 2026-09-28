"""Compare retained LOO expectations with same-snapshot full-posterior views.

This is a research diagnostic. It reads frozen snapshot inputs and writes
comparison tables under ``data/processed/retrospective_expectation_holdout``;
it does not modify a snapshot or any published artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from gippyrank.posterior.predictive import (
    ScheduledGame,
    mixture_cdf,
    mixture_quantile,
    posterior_prediction_team,
    predictive_components,
)
from gippyrank.posterior.snapshots import (
    _frozen_fcs_fallbacks,
    _game_from_included_row,
    included_game_rows_sha256,
    load_likelihood,
    load_teams,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/processed/retrospective_expectation_holdout"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _posterior_pmfs(path: Path) -> dict[str, np.ndarray]:
    rows = _read_rows(path)
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        values[row["team_id"]].append(float(row["probability"]))
    return {team_id: np.asarray(pmf, dtype=float) for team_id, pmf in values.items()}


def _mean_and_sd(pmf: np.ndarray) -> tuple[float, float]:
    ranks = np.arange(1, len(pmf) + 1, dtype=float)
    mean = float(np.dot(ranks, pmf))
    variance = float(np.dot((ranks - mean) ** 2, pmf))
    return mean, float(np.sqrt(max(variance, 0.0)))


def _retained_sources() -> list[Path]:
    config = _read_json(ROOT / "site/publish_config.json")
    selected_slot = str(config["default_publication_slot"])
    return [
        ROOT / entry["source"]
        for entry in config["snapshots"]
        if entry.get("publication_slot") == selected_slot
        and "/predictive/" in entry["source"]
    ]


def _analyze_source(
    source: Path, likelihood_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metadata = _read_json(source / "metadata.json")
    diagnostics = _read_json(source / "diagnostics.json")
    if not diagnostics.get("converged"):
        raise ValueError(f"{source}: full posterior did not converge")

    included_rows = _read_rows(source / "included_games.csv")
    included_ids = [str(row["id"]) for row in included_rows]
    if included_ids != [str(value) for value in metadata["included_game_ids"]]:
        raise ValueError(f"{source}: frozen game IDs disagree with metadata")
    expected_rows_hash = metadata.get("included_game_rows_sha256")
    if (
        expected_rows_hash
        and included_game_rows_sha256(included_rows) != expected_rows_hash
    ):
        raise ValueError(f"{source}: frozen game rows disagree with metadata hash")

    team_seasons = _read_json(source / "team_seasons.json")
    loo_artifact = team_seasons.get("retrospective_game_expectations")
    if not isinstance(loo_artifact, dict):
        raise TypeError(f"{source}: retained LOO expectation artifact is missing")
    if loo_artifact.get("source_snapshot_id") != metadata.get("snapshot_id"):
        raise ValueError(f"{source}: LOO artifact refers to a different snapshot")
    if loo_artifact.get("historical_likelihood_version") != metadata.get(
        "historical_likelihood_version"
    ):
        raise ValueError(f"{source}: LOO likelihood version differs from snapshot")
    if [
        str(value) for value in loo_artifact.get("included_game_ids", [])
    ] != included_ids:
        raise ValueError(f"{source}: LOO evidence boundary differs from snapshot")
    loo_records = loo_artifact.get("games")
    if not isinstance(loo_records, dict) or set(loo_records) != set(included_ids):
        raise ValueError(f"{source}: LOO game records do not match included games")

    teams, team_rows, prior_path = load_teams(
        ROOT,
        int(metadata["season"]),
        metadata["prior_family"],
        str(metadata["prior_model_version"]),
    )
    if sha256(prior_path) != metadata["prior_artifact_sha256"]:
        raise ValueError(f"{source}: frozen prior hash differs from snapshot metadata")
    teams, _, _, _ = _frozen_fcs_fallbacks(
        source=source,
        source_metadata=metadata,
        included_rows=included_rows,
        teams=teams,
        team_rows=team_rows,
    )
    teams_by_id = {team.team_id: team for team in teams}
    posterior = _posterior_pmfs(source / "posterior_pmfs.csv")
    if set(posterior) != set(teams_by_id):
        missing = sorted(set(teams_by_id) - set(posterior))
        extra = sorted(set(posterior) - set(teams_by_id))
        raise ValueError(
            f"{source}: PMF team coverage differs (missing={missing}, extra={extra})"
        )
    for team_id, pmf in posterior.items():
        if len(pmf) != len(teams_by_id[team_id].prior) or not np.isclose(
            pmf.sum(), 1.0, atol=1e-8
        ):
            raise ValueError(f"{source}: invalid full-posterior PMF for team {team_id}")

    likelihood = load_likelihood(likelihood_path)
    game_rows: list[dict[str, Any]] = []
    team_games: dict[str, list[dict[str, Any]]] = defaultdict(list)
    snapshot_id = str(metadata["snapshot_id"])
    prior_family = str(metadata["prior_family"])

    for row in included_rows:
        game = _game_from_included_row(row)
        scheduled = ScheduledGame(
            game_id=game.game_id,
            home_id=game.home_id,
            away_id=game.away_id,
            home_subdivision=game.home_subdivision,
            away_subdivision=game.away_subdivision,
            neutral_site=game.neutral_site,
            season_type=str(row.get("seasonType", "regular")),
            date=str(row.get("startDate", "")) or None,
            schedule_status="completed",
        )
        home = posterior_prediction_team(teams_by_id[game.home_id], posterior)
        away = posterior_prediction_team(teams_by_id[game.away_id], posterior)
        locations, weights = predictive_components(scheduled, home, away, likelihood)
        full_expected = float(np.dot(weights, locations))
        actual = float(game.home_points - game.away_points)
        full_percentile = mixture_cdf(
            actual,
            locations,
            weights,
            likelihood.scale,
            likelihood.degrees_of_freedom,
        )
        loo = loo_records[game.game_id]
        loo_expected = float(loo["expected_home_margin"])
        loo_percentile = float(loo["observed_margin_percentile"])
        loo_residual = actual - loo_expected
        full_residual = actual - full_expected
        absolute_residual_reduction = abs(loo_residual) - abs(full_residual)
        result: dict[str, Any] = {
            "snapshot_id": snapshot_id,
            "prior_family": prior_family,
            "effective_cutoff": metadata.get("effective_cutoff"),
            "game_id": game.game_id,
            "date": row.get("startDate", ""),
            "week": int(row["week"]) if row.get("week", "").isdigit() else None,
            "home_team_id": game.home_id,
            "home_team": row.get("homeTeam", game.home_id),
            "home_subdivision": game.home_subdivision,
            "away_team_id": game.away_id,
            "away_team": row.get("awayTeam", game.away_id),
            "away_subdivision": game.away_subdivision,
            "neutral_site": game.neutral_site,
            "actual_home_margin": actual,
            "loo_expected_home_margin": loo_expected,
            "full_expected_home_margin": full_expected,
            "full_minus_loo_expected_margin": full_expected - loo_expected,
            "loo_observed_percentile": loo_percentile,
            "full_observed_percentile": full_percentile,
            "percentile_change": full_percentile - loo_percentile,
            "loo_observed_direction_tail_probability": min(
                loo_percentile, 1.0 - loo_percentile
            ),
            "full_observed_direction_tail_probability": min(
                full_percentile, 1.0 - full_percentile
            ),
            "loo_focal_residual_home": loo_residual,
            "full_focal_residual_home": full_residual,
            "absolute_residual_reduction": absolute_residual_reduction,
            "relative_absolute_residual_reduction": (
                absolute_residual_reduction / abs(loo_residual)
                if abs(loo_residual) >= 5.0
                else None
            ),
            "residual_sign_changed": (loo_residual > 0) != (full_residual > 0),
        }
        game_rows.append(result)

        for side, team_id in (("home", game.home_id), ("away", game.away_id)):
            sign = 1.0 if side == "home" else -1.0
            team_games[team_id].append(
                {
                    "game_id": game.game_id,
                    "week": result["week"],
                    "actual_margin": sign * actual,
                    "loo_expected_margin": sign * loo_expected,
                    "full_expected_margin": sign * full_expected,
                    "loo_residual": sign * (actual - loo_expected),
                    "full_residual": sign * (actual - full_expected),
                }
            )

    team_rows_out: list[dict[str, Any]] = []
    prior_means: dict[str, float] = {}
    posterior_means: dict[str, float] = {}
    posterior_sds: dict[str, float] = {}
    for team in teams:
        if team.subdivision != "fbs" or team.team_id not in team_games:
            continue
        post_mean, post_sd = _mean_and_sd(posterior[team.team_id])
        prior_mean, _ = _mean_and_sd(team.prior)
        prior_means[team.team_id] = prior_mean
        posterior_means[team.team_id] = post_mean
        posterior_sds[team.team_id] = post_sd
        games = team_games[team.team_id]
        loo_residuals = np.asarray([game["loo_residual"] for game in games])
        full_residuals = np.asarray([game["full_residual"] for game in games])
        loo_positive = int(np.count_nonzero(loo_residuals > 0))
        full_positive = int(np.count_nonzero(full_residuals > 0))
        flips = int(np.count_nonzero((loo_residuals > 0) != (full_residuals > 0)))
        team_rows_out.append(
            {
                "snapshot_id": snapshot_id,
                "prior_family": prior_family,
                "team_id": team.team_id,
                "team": team.name,
                "games": len(games),
                "prior_mean_rank": prior_mean,
                "posterior_mean_rank": post_mean,
                "posterior_minus_prior_rank": post_mean - prior_mean,
                "posterior_rank_sd": post_sd,
                "loo_positive_residual_games": loo_positive,
                "loo_positive_residual_share": loo_positive / len(games),
                "full_positive_residual_games": full_positive,
                "full_positive_residual_share": full_positive / len(games),
                "mean_loo_residual": float(loo_residuals.mean()),
                "mean_full_residual": float(full_residuals.mean()),
                "median_loo_residual": float(np.median(loo_residuals)),
                "median_full_residual": float(np.median(full_residuals)),
                "residual_sign_flips": flips,
            }
        )

    team_rows_out.sort(key=lambda item: str(item["team"]).casefold())
    summary = _summarize(snapshot_id, prior_family, metadata, game_rows, team_rows_out)
    try:
        summary["source_path"] = source.relative_to(ROOT).as_posix()
    except ValueError:
        summary["source_path"] = source.as_posix()
    summary["prior_path"] = prior_path.relative_to(ROOT).as_posix()
    summary["selected_team_examples"] = _select_teams(team_rows_out)
    summary["representative_game_interval_summaries"] = _game_interval_summaries(
        summary["interpretation_examples"]["representative_game_ids"],
        {str(row["id"]): row for row in included_rows},
        teams_by_id,
        posterior,
        likelihood,
        loo_records,
    )
    return game_rows, team_rows_out, summary


def _game_interval_summaries(
    game_ids: dict[str, str | None],
    included_rows: dict[str, dict[str, str]],
    teams_by_id: dict[str, Any],
    posterior: dict[str, np.ndarray],
    likelihood: Any,
    loo_records: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute exact 80% intervals for a handful of illustrative games."""

    examples = []
    for label, selected_id in game_ids.items():
        if selected_id is None:
            continue
        game_id = str(selected_id)
        row = included_rows[game_id]
        game = _game_from_included_row(row)
        scheduled = ScheduledGame(
            game_id=game.game_id,
            home_id=game.home_id,
            away_id=game.away_id,
            home_subdivision=game.home_subdivision,
            away_subdivision=game.away_subdivision,
            neutral_site=game.neutral_site,
            schedule_status="completed",
        )
        home = posterior_prediction_team(teams_by_id[game.home_id], posterior)
        away = posterior_prediction_team(teams_by_id[game.away_id], posterior)
        locations, weights = predictive_components(scheduled, home, away, likelihood)
        examples.append(
            {
                "label": label,
                "game_id": game_id,
                "home_team": row.get("homeTeam", game.home_id),
                "away_team": row.get("awayTeam", game.away_id),
                "actual_home_margin": float(game.home_points - game.away_points),
                "loo_expected_home_margin": float(
                    loo_records[game_id]["expected_home_margin"]
                ),
                "full_expected_home_margin": float(np.dot(weights, locations)),
                "loo_interval_80": loo_records[game_id]["margin_interval_80"],
                "full_interval_80": [
                    mixture_quantile(
                        probability,
                        locations,
                        weights,
                        likelihood.scale,
                        likelihood.degrees_of_freedom,
                    )
                    for probability in (0.10, 0.90)
                ],
                "loo_observed_percentile": float(
                    loo_records[game_id]["observed_margin_percentile"]
                ),
            }
        )
    return examples


def _percentile(values: list[float], probability: float) -> float | None:
    return float(np.quantile(values, probability)) if values else None


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "p10": _percentile(values, 0.10),
        "p25": _percentile(values, 0.25),
        "median": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
    }


def _summarize_population(games: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize margin, tail, and residual changes for one game population."""

    deltas = [float(game["full_minus_loo_expected_margin"]) for game in games]
    absolute_deltas = [abs(value) for value in deltas]
    percentile_changes = [abs(float(game["percentile_change"])) for game in games]
    loo_surprise = [abs(float(game["loo_observed_percentile"]) - 0.5) for game in games]
    full_surprise = [
        abs(float(game["full_observed_percentile"]) - 0.5) for game in games
    ]
    loo_tails = [
        float(game["loo_observed_direction_tail_probability"]) for game in games
    ]
    full_tails = [
        float(game["full_observed_direction_tail_probability"]) for game in games
    ]
    residual_reductions = [float(game["absolute_residual_reduction"]) for game in games]
    relative_reductions = [
        float(game["relative_absolute_residual_reduction"])
        for game in games
        if game["relative_absolute_residual_reduction"] is not None
    ]
    same_sign_reductions = [
        game
        for game in games
        if not bool(game["residual_sign_changed"])
        and float(game["absolute_residual_reduction"]) >= 5.0
    ]
    loo_surprise_full_ordinary = [
        game
        for game in games
        if float(game["loo_observed_direction_tail_probability"]) < 0.05
        and float(game["full_observed_direction_tail_probability"]) >= 0.10
    ]
    surprising_both = [
        game
        for game in games
        if float(game["loo_observed_direction_tail_probability"]) < 0.05
        and float(game["full_observed_direction_tail_probability"]) < 0.05
    ]
    examples = [
        {
            "game_id": str(game["game_id"]),
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "actual_home_margin": float(game["actual_home_margin"]),
            "loo_expected_home_margin": float(game["loo_expected_home_margin"]),
            "full_expected_home_margin": float(game["full_expected_home_margin"]),
            "loo_residual_home": float(game["loo_focal_residual_home"]),
            "full_residual_home": float(game["full_focal_residual_home"]),
            "absolute_residual_reduction": float(game["absolute_residual_reduction"]),
            "loo_observed_direction_tail_probability": float(
                game["loo_observed_direction_tail_probability"]
            ),
            "full_observed_direction_tail_probability": float(
                game["full_observed_direction_tail_probability"]
            ),
        }
        for game in sorted(
            same_sign_reductions,
            key=lambda item: float(item["absolute_residual_reduction"]),
            reverse=True,
        )[:5]
    ]
    return {
        "game_count": len(games),
        "expected_margin_shift_full_minus_loo": {
            "signed_mean": float(np.mean(deltas)),
            "signed_median": float(np.median(deltas)),
            "absolute_shift_distribution": _distribution(absolute_deltas),
            "absolute_shift_over_threshold_share": {
                str(threshold): sum(value > threshold for value in absolute_deltas)
                / len(games)
                for threshold in (1, 3, 5, 10)
            },
        },
        "observed_percentile_movement": {
            "absolute_change_distribution": _distribution(percentile_changes),
            "absolute_change_at_least_10_points_share": sum(
                value >= 0.10 for value in percentile_changes
            )
            / len(games),
            "absolute_change_at_least_20_points_share": sum(
                value >= 0.20 for value in percentile_changes
            )
            / len(games),
            "closer_to_50th_percentile_share": sum(
                full < loo
                for loo, full in zip(loo_surprise, full_surprise, strict=True)
            )
            / len(games),
        },
        "observed_direction_tail_probability": {
            "loo_below_5pct_count": sum(value < 0.05 for value in loo_tails),
            "full_below_5pct_count": sum(value < 0.05 for value in full_tails),
            "loo_below_5pct_full_at_least_10pct_count": len(loo_surprise_full_ordinary),
            "both_below_5pct_count": len(surprising_both),
        },
        "absolute_residual_normalization": {
            "reduction_distribution_points": _distribution(residual_reductions),
            "reduced_residual_share": sum(value > 0 for value in residual_reductions)
            / len(games),
            "reduced_by_at_least_3_points_share": sum(
                value >= 3 for value in residual_reductions
            )
            / len(games),
            "reduced_by_at_least_5_points_share": sum(
                value >= 5 for value in residual_reductions
            )
            / len(games),
            "residual_grew_share": sum(value < 0 for value in residual_reductions)
            / len(games),
            "relative_reduction_for_loo_absolute_residual_at_least_5": {
                **_distribution(relative_reductions),
                "denominator_threshold_points": 5,
                "share_reduced_by_at_least_25pct": sum(
                    value >= 0.25 for value in relative_reductions
                )
                / len(relative_reductions)
                if relative_reductions
                else None,
            },
            "residual_sign_changed_count": sum(
                bool(game["residual_sign_changed"]) for game in games
            ),
            "same_sign_reduced_by_at_least_5_points_count": len(same_sign_reductions),
            "largest_same_sign_reductions": examples,
        },
    }


def _summarize(
    snapshot_id: str,
    prior_family: str,
    metadata: dict[str, Any],
    games: list[dict[str, Any]],
    teams: list[dict[str, Any]],
) -> dict[str, Any]:
    deltas = [float(game["full_minus_loo_expected_margin"]) for game in games]
    absolute = [abs(value) for value in deltas]
    percentile_changes = [abs(float(game["percentile_change"])) for game in games]
    loo_surprise = [abs(float(game["loo_observed_percentile"]) - 0.5) for game in games]
    full_surprise = [
        abs(float(game["full_observed_percentile"]) - 0.5) for game in games
    ]
    moved_toward_center = [
        full < loo for full, loo in zip(full_surprise, loo_surprise, strict=True)
    ]
    surprise_cases = [
        game
        for game in games
        if float(game["loo_observed_direction_tail_probability"]) < 0.05
        and float(game["full_observed_direction_tail_probability"]) >= 0.10
    ]
    surprising_both = [
        game
        for game in games
        if float(game["loo_observed_direction_tail_probability"]) < 0.05
        and float(game["full_observed_direction_tail_probability"]) < 0.05
    ]
    stable_cases = [
        game
        for game in games
        if abs(float(game["full_minus_loo_expected_margin"])) < 1.0
    ]
    by_week: dict[str, list[float]] = defaultdict(list)
    for game in games:
        by_week[str(game["week"])].append(
            abs(float(game["full_minus_loo_expected_margin"]))
        )
    large_margin = [
        game for game in games if abs(float(game["actual_home_margin"])) >= 35
    ]
    mean_abs_by_uncertainty: dict[str, list[float]] = defaultdict(list)
    team_by_id = {str(team["team_id"]): team for team in teams}
    for game in games:
        home = team_by_id.get(str(game["home_team_id"]))
        away = team_by_id.get(str(game["away_team_id"]))
        if home is None or away is None:
            continue
        uncertainty = (
            float(home["posterior_rank_sd"]) + float(away["posterior_rank_sd"])
        ) / 2.0
        band = "higher" if uncertainty >= 20 else "lower"
        mean_abs_by_uncertainty[band].append(
            abs(float(game["full_minus_loo_expected_margin"]))
        )
    return {
        "snapshot_id": snapshot_id,
        "prior_family": prior_family,
        "season": metadata["season"],
        "effective_cutoff": metadata.get("effective_cutoff"),
        "game_count": len(games),
        "team_count_fbs": len(teams),
        "expected_margin_shift_full_minus_loo": {
            "mean": float(np.mean(deltas)),
            "median": float(np.median(deltas)),
            "absolute_median": float(np.median(absolute)),
            "absolute_p75": _percentile(absolute, 0.75),
            "absolute_p90": _percentile(absolute, 0.90),
            "absolute_p95": _percentile(absolute, 0.95),
            "absolute_max": max(absolute),
            "absolute_over_threshold_share": {
                str(threshold): sum(value > threshold for value in absolute)
                / len(absolute)
                for threshold in (1, 3, 5, 10)
            },
        },
        "observed_percentile_change": {
            "absolute_median": float(np.median(percentile_changes)),
            "absolute_p90": _percentile(percentile_changes, 0.90),
            "absolute_over_10_points_share": sum(
                value >= 0.10 for value in percentile_changes
            )
            / len(games),
            "absolute_over_20_points_share": sum(
                value >= 0.20 for value in percentile_changes
            )
            / len(games),
        },
        "full_posterior_observed_direction_tail_behavior": {
            "percent_games_moved_closer_to_50th_percentile": sum(moved_toward_center)
            / len(games),
            "median_reduction_in_absolute_distance_from_50th_percentile": float(
                np.median(np.asarray(loo_surprise) - np.asarray(full_surprise))
            ),
            "loo_tail_below_5pct_and_full_tail_at_least_10pct": len(surprise_cases),
            "loo_and_full_both_tail_below_5pct": len(surprising_both),
        },
        "population_summaries": {
            "all_modeled_games": _summarize_population(games),
            "at_least_one_fbs_team": _summarize_population(
                [
                    game
                    for game in games
                    if game["home_subdivision"] == "fbs"
                    or game["away_subdivision"] == "fbs"
                ]
            ),
            "fbs_vs_fbs": _summarize_population(
                [
                    game
                    for game in games
                    if game["home_subdivision"] == "fbs"
                    and game["away_subdivision"] == "fbs"
                ]
            ),
        },
        "context_slices": {
            "fbs_fbs_game_count": sum(
                game["home_subdivision"] == "fbs" and game["away_subdivision"] == "fbs"
                for game in games
            ),
            "mean_absolute_margin_shift_by_game_week": {
                week: float(np.mean(values)) for week, values in sorted(by_week.items())
            },
            "mean_absolute_shift_abs_actual_margin_at_least_35": (
                float(
                    np.mean(
                        [
                            abs(float(game["full_minus_loo_expected_margin"]))
                            for game in large_margin
                        ]
                    )
                )
                if large_margin
                else None
            ),
            "mean_absolute_shift_posterior_rank_sd_average_below_20": (
                float(np.mean(mean_abs_by_uncertainty["lower"]))
                if mean_abs_by_uncertainty["lower"]
                else None
            ),
            "fbs_fbs_game_count_posterior_rank_sd_average_below_20": len(
                mean_abs_by_uncertainty["lower"]
            ),
            "mean_absolute_shift_posterior_rank_sd_average_at_least_20": (
                float(np.mean(mean_abs_by_uncertainty["higher"]))
                if mean_abs_by_uncertainty["higher"]
                else None
            ),
            "fbs_fbs_game_count_posterior_rank_sd_average_at_least_20": len(
                mean_abs_by_uncertainty["higher"]
            ),
        },
        "interpretation_examples": {
            "near_identical_margin_expectation_games": len(stable_cases),
            "loo_surprising_but_full_ordinary_game_ids": [
                str(game["game_id"]) for game in surprise_cases
            ],
            "largest_absolute_margin_shift_game_ids": [
                str(game["game_id"])
                for game in sorted(
                    games,
                    key=lambda item: abs(float(item["full_minus_loo_expected_margin"])),
                    reverse=True,
                )[:5]
            ],
            "representative_game_ids": {
                "near_identical_expectations": str(
                    min(
                        games,
                        key=lambda item: abs(
                            float(item["full_minus_loo_expected_margin"])
                        ),
                    )["game_id"]
                ),
                "largest_expected_margin_shift": str(
                    max(
                        games,
                        key=lambda item: abs(
                            float(item["full_minus_loo_expected_margin"])
                        ),
                    )["game_id"]
                ),
                "loo_surprising_full_ordinary": str(
                    max(
                        surprise_cases,
                        key=lambda item: abs(
                            float(item["full_minus_loo_expected_margin"])
                        ),
                    )["game_id"]
                )
                if surprise_cases
                else None,
                "remains_surprising_under_both": str(
                    max(
                        surprising_both,
                        key=lambda item: abs(
                            float(item["full_minus_loo_expected_margin"])
                        ),
                    )["game_id"]
                )
                if surprising_both
                else None,
            },
        },
        "team_residual_sign_checks": {
            "teams_with_3plus_games": sum(int(team["games"]) >= 3 for team in teams),
            "teams_loo_positive_share_at_least_75pct": sum(
                int(team["games"]) >= 3
                and float(team["loo_positive_residual_share"]) >= 0.75
                for team in teams
            ),
            "teams_full_positive_share_at_least_75pct": sum(
                int(team["games"]) >= 3
                and float(team["full_positive_residual_share"]) >= 0.75
                for team in teams
            ),
            "teams_loo_negative_share_at_least_75pct": sum(
                int(team["games"]) >= 3
                and float(team["loo_positive_residual_share"]) <= 0.25
                for team in teams
            ),
            "teams_full_negative_share_at_least_75pct": sum(
                int(team["games"]) >= 3
                and float(team["full_positive_residual_share"]) <= 0.25
                for team in teams
            ),
            "total_residual_sign_flips": sum(
                int(team["residual_sign_flips"]) for team in teams
            ),
        },
    }


def _select_teams(teams: list[dict[str, Any]]) -> dict[str, str]:
    by_name = {str(team["team"]).casefold(): team for team in teams}
    eligible = [team for team in teams if int(team["games"]) >= 3]
    picks = {
        "georgia": by_name["georgia"]["team_id"] if "georgia" in by_name else "missing",
        "closest_posterior_to_prior_with_3plus_games": min(
            eligible, key=lambda team: abs(float(team["posterior_minus_prior_rank"]))
        )["team_id"],
        "largest_posterior_rank_improvement_with_3plus_games": min(
            eligible, key=lambda team: float(team["posterior_minus_prior_rank"])
        )["team_id"],
        "largest_posterior_rank_decline_with_3plus_games": max(
            eligible, key=lambda team: float(team["posterior_minus_prior_rank"])
        )["team_id"],
        "highest_posterior_rank_uncertainty_with_3plus_games": max(
            eligible, key=lambda team: float(team["posterior_rank_sd"])
        )["team_id"],
    }
    return {key: str(value) for key, value in picks.items()}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty comparison table: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        help="Snapshot predictive directory; repeat to compare multiple models",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--likelihood",
        type=Path,
        default=ROOT / "data/processed/posterior/historical_likelihood_v1.json",
    )
    args = parser.parse_args()

    sources = args.source or _retained_sources()
    if not sources:
        raise ValueError(
            "no predictive sources found for the selected publication slot"
        )
    all_games: list[dict[str, Any]] = []
    all_teams: list[dict[str, Any]] = []
    summaries = []
    for source in sources:
        resolved = source if source.is_absolute() else ROOT / source
        games, teams, summary = _analyze_source(resolved, args.likelihood)
        all_games.extend(games)
        all_teams.extend(teams)
        summaries.append(summary)
        print(f"{summary['snapshot_id']}: {summary['game_count']} games", flush=True)

    if args.source is None and len(summaries) > 1:
        evidence_sets = [
            tuple(
                str(row["game_id"])
                for row in all_games
                if row["snapshot_id"] == summary["snapshot_id"]
            )
            for summary in summaries
        ]
        if any(evidence != evidence_sets[0] for evidence in evidence_sets[1:]):
            raise ValueError(
                "selected prior-family snapshots have different game evidence"
            )

    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "game_comparisons.csv", all_games)
    _write_csv(output_dir / "team_comparisons.csv", all_teams)
    (output_dir / "summary.json").write_text(
        json.dumps({"snapshots": summaries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote comparison tables to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
