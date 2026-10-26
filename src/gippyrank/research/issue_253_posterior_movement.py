"""Summarize retained posterior movement and join checkpoint-safe surprises."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from gippyrank.research.context_incremental_information import (
    kl_posterior_from_prior,
    normalized_pmf,
)
from gippyrank.research.issue_247_contradictory_evidence import surprise_group

FAMILIES = ("History", "Context")
MODEL_ARMS = {family: f"{family}_existing" for family in FAMILIES}
MIN_SURPRISE_GAMES = 3
BOOTSTRAP_SEED = 253

MOVEMENT_METRICS = (
    "signed_location_change_from_preseason_normalized",
    "signed_location_change_since_previous_normalized",
    "posterior_rank_sd_normalized",
    "rank_sd_change_from_preseason_normalized",
    "rank_sd_change_since_previous_normalized",
    "posterior_central80_width_normalized",
    "central80_width_change_from_preseason_normalized",
    "central80_width_change_since_previous_normalized",
    "kl_posterior_from_prior",
    "total_variation_since_previous",
)


def summarize_rank_pmf(values: Sequence[float], *, name: str) -> dict[str, float | int]:
    """Return rank moments and an equal-tail central 80% interval for ranks 1..N."""
    pmf = normalized_pmf(values, name=name)
    ranks = np.arange(1, len(pmf) + 1, dtype=float)
    expected_rank = float(np.dot(ranks, pmf))
    variance = float(np.dot((ranks - expected_rank) ** 2, pmf))
    cumulative = np.cumsum(pmf)
    lower_rank = int(np.searchsorted(cumulative, 0.10, side="left") + 1)
    upper_rank = int(np.searchsorted(cumulative, 0.90, side="left") + 1)
    population = len(pmf)
    scale = max(population - 1, 1)
    sd = math.sqrt(max(variance, 0.0))
    return {
        "rank_population": population,
        "rank_support_min": 1,
        "rank_support_max": population,
        "expected_rank": expected_rank,
        "rank_sd": sd,
        "central80_lower_rank": lower_rank,
        "central80_upper_rank": upper_rank,
        "central80_interval_width": upper_rank - lower_rank,
        "rank_strength": 1.0 - (expected_rank - 1.0) / scale,
        "rank_sd_normalized": sd / scale,
        "central80_width_normalized": (upper_rank - lower_rank) / scale,
    }


def _utc(value: str) -> Any:
    from datetime import datetime

    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timestamp must include timezone: {value!r}")
    return parsed


def _surprise_index(
    game_rows: list[dict[str, Any]],
) -> dict[tuple[int, str, str], list[dict[str, Any]]]:
    result: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_games: set[tuple[int, str, str, str]] = set()
    forecast_targets: dict[tuple[int, str, str], str] = {}
    for row in game_rows:
        season = int(row["season"])
        family = str(row["prior_family"])
        policy = str(row["policy"])
        if policy != "existing" or family not in FAMILIES:
            continue
        arm = str(row["model_arm"])
        if arm != MODEL_ARMS[family]:
            raise ValueError(f"unexpected #247 model arm {arm!r} for {family}")
        forecast_id = str(row["forecast_id"])
        forecast_key = (season, family, forecast_id)
        game_id = str(row["game_id"])
        if forecast_key in forecast_targets and forecast_targets[forecast_key] != game_id:
            raise ValueError(f"#247 forecast ID points to multiple games: {forecast_id}")
        forecast_targets[forecast_key] = game_id
        team_key = (season, family, str(row["team_id"]))
        game_key = (*team_key, game_id)
        if game_key in seen_games:
            raise ValueError(f"duplicate #247 team-game surprise {game_key}")
        seen_games.add(game_key)
        kickoff = _utc(str(row["kickoff_utc"]))
        forecast_cutoff = _utc(str(row["cutoff_utc"]))
        if forecast_cutoff >= kickoff:
            raise ValueError(f"#247 forecast cutoff is not pregame for {game_key}")
        half_width = float(row["margin_80_interval_half_width"])
        signed_error = float(row["signed_error"])
        surprise = float(row["standardized_surprise"])
        if not all(math.isfinite(value) for value in (half_width, signed_error, surprise)):
            raise ValueError(f"non-finite #247 surprise for {game_key}")
        if half_width <= 0 or not math.isclose(
            signed_error / half_width, surprise, rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ValueError(f"#247 surprise definition mismatch for {game_key}")
        result[team_key].append({
            **row,
            "_kickoff": kickoff,
            "_forecast_cutoff": forecast_cutoff,
            "_surprise": surprise,
        })
    for members in result.values():
        members.sort(key=lambda game: (game["_kickoff"], str(game["game_id"])))
    return result


def build_team_checkpoint_observations(
    panel_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    game_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize full PMFs and join only evidence-available prior game surprises."""
    evidence_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for evidence in evidence_rows:
        key = (int(evidence["season"]), int(evidence["checkpoint"]))
        if key in evidence_by_key:
            raise ValueError(f"duplicate #192 evidence checkpoint {key}")
        evidence_by_key[key] = evidence

    surprise_by_game = _surprise_index(game_rows)
    rows: list[dict[str, Any]] = []
    previous_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    baseline_prior_by_key: dict[tuple[int, str, str], np.ndarray] = {}
    seen_panel_keys: set[tuple[int, int, str]] = set()

    for panel in sorted(
        (row for row in panel_rows if str(row["subdivision"]).lower() == "fbs"),
        key=lambda row: (
            int(row["season"]),
            int(row["checkpoint"]),
            int(row["team_id"]),
        ),
    ):
        season = int(panel["season"])
        checkpoint = int(panel["checkpoint"])
        team_id = str(panel["team_id"])
        panel_key = (season, checkpoint, team_id)
        if panel_key in seen_panel_keys:
            raise ValueError(f"duplicate #192 FBS panel row {panel_key}")
        seen_panel_keys.add(panel_key)
        evidence = evidence_by_key.get((season, checkpoint))
        if evidence is None:
            raise ValueError(f"missing #192 evidence for {season}/{checkpoint}")
        cutoff_text = str(panel["cutoff"])
        if _utc(cutoff_text) != _utc(str(evidence["cutoff"])):
            raise ValueError(f"panel/evidence cutoff mismatch for {panel_key}")
        included_ids = [str(game_id) for game_id in evidence["included_game_ids"]]
        if len(included_ids) != int(evidence["included_game_count"]):
            raise ValueError(f"#192 evidence count mismatch for {season}/{checkpoint}")
        if len(set(included_ids)) != len(included_ids):
            raise ValueError(f"duplicate #192 evidence game ID for {season}/{checkpoint}")
        included_set = set(included_ids)
        cutoff = _utc(cutoff_text)
        rank_count = int(panel["rank_count"])
        if rank_count < 2:
            raise ValueError(f"invalid FBS rank population {rank_count} for {panel_key}")

        for family in FAMILIES:
            arm = MODEL_ARMS[family]
            prior_values = panel[f"{family.lower()}_prior_pmf"]
            posterior_values = panel[f"{family.lower()}_posterior_pmf"]
            prior_pmf = normalized_pmf(prior_values, name=f"{family} prior {panel_key}")
            posterior_pmf = normalized_pmf(
                posterior_values, name=f"{family} posterior {panel_key}"
            )
            if len(prior_pmf) != rank_count or len(posterior_pmf) != rank_count:
                raise ValueError(f"PMF support mismatch for {panel_key}/{family}")
            prior_summary = summarize_rank_pmf(prior_pmf, name=f"{family} prior")
            posterior_summary = summarize_rank_pmf(
                posterior_pmf, name=f"{family} posterior"
            )
            team_key = (season, family, team_id)
            if checkpoint == 1:
                baseline_prior_by_key[team_key] = prior_pmf.copy()
            else:
                baseline = baseline_prior_by_key.get(team_key)
                if baseline is None or baseline.shape != prior_pmf.shape:
                    raise ValueError(f"missing preseason prior for {team_key}")
                if not np.allclose(baseline, prior_pmf, rtol=0.0, atol=1e-12):
                    raise ValueError(f"preseason prior changed across checkpoints for {team_key}")

            focal_games: list[dict[str, Any]] = []
            for game in surprise_by_game.get((season, family, team_id), []):
                if str(game["game_id"]) not in included_set:
                    continue
                if game["_kickoff"] >= cutoff:
                    raise ValueError(
                            f"#247 game {game['game_id']} occurs after #192 checkpoint {panel_key}"
                    )
                focal_games.append(game)
            focal_games.sort(key=lambda game: (game["_kickoff"], str(game["game_id"])))
            expected_focal_games = int(panel["games_played"])
            if len(focal_games) != expected_focal_games:
                raise ValueError(
                    f"#247/#192 focal-game coverage mismatch for {panel_key}/{family}: "
                    f"{len(focal_games)} != {expected_focal_games}"
                )
            surprise_count = len(focal_games)
            accumulated = (
                math.fsum(float(game["_surprise"]) for game in focal_games)
                / surprise_count
                if surprise_count
                else None
            )
            primary_eligible = surprise_count >= MIN_SURPRISE_GAMES
            surprise_category = (
                surprise_group(float(accumulated)) if primary_eligible else None
            )
            location_from_prior = (
                prior_summary["expected_rank"] - posterior_summary["expected_rank"]
            )
            normalized_location_from_prior = location_from_prior / (rank_count - 1)
            kl = kl_posterior_from_prior(posterior_pmf, prior_pmf)
            if family == "History":
                retained_kl = float(panel["history_kl_posterior_from_prior"])
                if not math.isclose(kl, retained_kl, rel_tol=1e-9, abs_tol=1e-10):
                    raise ValueError(f"recomputed History KL differs for {panel_key}")

            previous = previous_by_key.get(team_key)
            if checkpoint == 1:
                location_since_previous = None
                sd_change_since_previous = None
                width_change_since_previous = None
                total_variation = None
            else:
                if previous is None or int(previous["checkpoint"]) != checkpoint - 1:
                    raise ValueError(f"missing previous checkpoint for {team_key}/{checkpoint}")
                location_since_previous = (
                    float(previous["posterior_expected_rank"])
                    - float(posterior_summary["expected_rank"])
                )
                sd_change_since_previous = (
                    float(posterior_summary["rank_sd"])
                    - float(previous["posterior_rank_sd"])
                )
                width_change_since_previous = (
                    float(posterior_summary["central80_interval_width"])
                    - float(previous["posterior_central80_width"])
                )
                previous_pmf = np.asarray(previous["posterior_pmf"], dtype=float)
                total_variation = float(0.5 * np.abs(posterior_pmf - previous_pmf).sum())

            row: dict[str, Any] = {
                "season": season,
                "checkpoint": checkpoint,
                "cutoff": cutoff_text,
                "model_arm": arm,
                "prior_family": family,
                "team_id": team_id,
                "team_name": str(panel["team_name"]),
                "component_status": str(panel["component_status"]),
                "rank_population": rank_count,
                "rank_support_min": 1,
                "rank_support_max": rank_count,
                "focal_games_included_in_evidence": expected_focal_games,
                "total_included_game_evidence": int(evidence["included_game_count"]),
                "prior_surprise_game_count": surprise_count,
                "eligible_prior_game_ids": "|".join(
                    str(game["game_id"]) for game in focal_games
                ),
                "accumulated_surprise": accumulated,
                "surprise_group": surprise_category,
                "surprise_group_eligible": primary_eligible,
                "preseason_expected_rank": prior_summary["expected_rank"],
                "posterior_expected_rank": posterior_summary["expected_rank"],
                "signed_location_change_from_preseason": location_from_prior,
                "signed_location_change_from_preseason_normalized": normalized_location_from_prior,
                "signed_location_change_since_previous": location_since_previous,
                "signed_location_change_since_previous_normalized": (
                    None
                    if location_since_previous is None
                    else location_since_previous / (rank_count - 1)
                ),
                "preseason_strength": prior_summary["rank_strength"],
                "preseason_rank_sd": prior_summary["rank_sd"],
                "preseason_rank_sd_normalized": prior_summary["rank_sd_normalized"],
                "posterior_rank_sd": posterior_summary["rank_sd"],
                "posterior_rank_sd_normalized": posterior_summary["rank_sd_normalized"],
                "rank_sd_change_from_preseason": posterior_summary["rank_sd"]
                - prior_summary["rank_sd"],
                "rank_sd_change_from_preseason_normalized": posterior_summary[
                    "rank_sd_normalized"
                ]
                - prior_summary["rank_sd_normalized"],
                "rank_sd_change_since_previous": sd_change_since_previous,
                "rank_sd_change_since_previous_normalized": (
                    None
                    if sd_change_since_previous is None
                    else sd_change_since_previous / (rank_count - 1)
                ),
                "preseason_central80_lower_rank": prior_summary[
                    "central80_lower_rank"
                ],
                "preseason_central80_upper_rank": prior_summary[
                    "central80_upper_rank"
                ],
                "preseason_central80_width": prior_summary[
                    "central80_interval_width"
                ],
                "preseason_central80_width_normalized": prior_summary[
                    "central80_width_normalized"
                ],
                "posterior_central80_lower_rank": posterior_summary[
                    "central80_lower_rank"
                ],
                "posterior_central80_upper_rank": posterior_summary[
                    "central80_upper_rank"
                ],
                "posterior_central80_width": posterior_summary[
                    "central80_interval_width"
                ],
                "posterior_central80_width_normalized": posterior_summary[
                    "central80_width_normalized"
                ],
                "central80_width_change_from_preseason": posterior_summary[
                    "central80_interval_width"
                ]
                - prior_summary["central80_interval_width"],
                "central80_width_change_from_preseason_normalized": posterior_summary[
                    "central80_width_normalized"
                ]
                - prior_summary["central80_width_normalized"],
                "central80_width_change_since_previous": width_change_since_previous,
                "central80_width_change_since_previous_normalized": (
                    None
                    if width_change_since_previous is None
                    else width_change_since_previous / (rank_count - 1)
                ),
                "kl_posterior_from_prior": kl,
                "total_variation_since_previous": total_variation,
                "prior_pmf": prior_pmf.tolist(),
                "posterior_pmf": posterior_pmf.tolist(),
            }
            rows.append(row)
            previous_by_key[team_key] = row

    return rows


def _quantile_summary(values: Sequence[Any]) -> dict[str, float | None]:
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=float,
    )
    if finite.size == 0:
        return {name: None for name in ("mean", "median", "p10", "p25", "p75", "p90")}
    p10, p25, median, p75, p90 = np.quantile(
        finite, [0.10, 0.25, 0.50, 0.75, 0.90]
    )
    return {
        "mean": float(np.mean(finite)),
        "median": float(median),
        "p10": float(p10),
        "p25": float(p25),
        "p75": float(p75),
        "p90": float(p90),
    }


def aggregate_movement(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize metric distributions by season/checkpoint/family/status."""
    grouped: dict[tuple[int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["season"]),
            int(row["checkpoint"]),
            str(row["model_arm"]),
            str(row["component_status"]),
        )
        grouped[key].append(row)
    output: list[dict[str, Any]] = []
    for (season, checkpoint, arm, status), members in sorted(grouped.items()):
        result: dict[str, Any] = {
            "season": season,
            "checkpoint": checkpoint,
            "model_arm": arm,
            "prior_family": arm.removesuffix("_existing"),
            "component_status": status,
            "team_seasons": len(members),
            "mean_focal_games_included": float(
                np.mean([row["focal_games_included_in_evidence"] for row in members])
            ),
            "mean_total_included_game_evidence": float(
                np.mean([row["total_included_game_evidence"] for row in members])
            ),
        }
        for metric in MOVEMENT_METRICS:
            for stat, value in _quantile_summary([row[metric] for row in members]).items():
                result[f"{metric}_{stat}"] = value
        output.append(result)
    return output


def aggregate_by_surprise_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row["surprise_group_eligible"] and int(row["checkpoint"]) > 1
    ]
    grouped: dict[tuple[int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        key = (
            int(row["season"]),
            int(row["checkpoint"]),
            str(row["model_arm"]),
            str(row["surprise_group"]),
        )
        grouped[key].append(row)
    metrics = (
        "accumulated_surprise",
        "signed_location_change_since_previous_normalized",
        "signed_location_change_from_preseason_normalized",
        "rank_sd_change_since_previous_normalized",
        "rank_sd_change_from_preseason_normalized",
        "central80_width_change_since_previous_normalized",
        "central80_width_change_from_preseason_normalized",
        "kl_posterior_from_prior",
        "total_variation_since_previous",
    )
    output: list[dict[str, Any]] = []
    for (season, checkpoint, arm, category), members in sorted(grouped.items()):
        result: dict[str, Any] = {
            "season": season,
            "checkpoint": checkpoint,
            "model_arm": arm,
            "prior_family": arm.removesuffix("_existing"),
            "surprise_group": category,
            "team_checkpoints": len(members),
            "team_seasons": len({(row["season"], row["team_id"]) for row in members}),
        }
        for metric in metrics:
            for stat, value in _quantile_summary([row[metric] for row in members]).items():
                result[f"{metric}_{stat}"] = value
        output.append(result)
    return output


def _seed_for(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _two_way_weights(
    rng: np.random.Generator,
    team_codes: np.ndarray,
    snapshot_codes: np.ndarray,
    team_count: int,
    snapshot_count: int,
) -> np.ndarray:
    team_counts = np.bincount(
        rng.integers(0, team_count, size=team_count), minlength=team_count
    )
    snapshot_counts = np.bincount(
        rng.integers(0, snapshot_count, size=snapshot_count),
        minlength=snapshot_count,
    )
    return (team_counts[team_codes] * snapshot_counts[snapshot_codes]).astype(float)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    active = (weights > 0) & np.isfinite(values)
    if not active.any():
        return None
    return float(np.average(values[active], weights=weights[active]))


def bootstrap_surprise_group_contrasts(
    rows: list[dict[str, Any]],
    *,
    repetitions: int = 1000,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Estimate pooled group differences with team and snapshot resampling."""
    eligible = [
        row
        for row in rows
        if row["surprise_group_eligible"] and int(row["checkpoint"]) > 1
    ]
    measures = (
        "signed_location_change_since_previous_normalized",
        "absolute_location_change_since_previous_normalized",
        "rank_sd_change_since_previous_normalized",
        "central80_width_change_since_previous_normalized",
    )
    contrasts = (
        ("sustained_negative", "approximately_neutral"),
        ("sustained_positive", "approximately_neutral"),
        ("sustained_positive", "sustained_negative"),
    )
    output: list[dict[str, Any]] = []
    for arm in (MODEL_ARMS[family] for family in FAMILIES):
        members = [row for row in eligible if row["model_arm"] == arm]
        if not members:
            continue
        for row in members:
            row["absolute_location_change_since_previous_normalized"] = abs(
                float(row["signed_location_change_since_previous_normalized"])
            )
        teams = sorted({str(row["team_id"]) for row in members})
        snapshots = sorted(
            {f"{row['season']}:{row['checkpoint']}" for row in members}
        )
        team_index = {value: index for index, value in enumerate(teams)}
        snapshot_index = {value: index for index, value in enumerate(snapshots)}
        team_codes = np.asarray([team_index[str(row["team_id"])] for row in members])
        snapshot_codes = np.asarray(
            [snapshot_index[f"{row['season']}:{row['checkpoint']}"] for row in members]
        )
        groups = np.asarray([str(row["surprise_group"]) for row in members])
        rng = np.random.default_rng(_seed_for(seed, f"group:{arm}"))
        bootstrap_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for _ in range(repetitions):
            weights = _two_way_weights(
                rng, team_codes, snapshot_codes, len(teams), len(snapshots)
            )
            for measure in measures:
                values = np.asarray([float(row[measure]) for row in members])
                for group_a, group_b in contrasts:
                    mean_a = _weighted_mean(values[groups == group_a], weights[groups == group_a])
                    mean_b = _weighted_mean(values[groups == group_b], weights[groups == group_b])
                    if mean_a is not None and mean_b is not None:
                        bootstrap_values[(measure, group_a, group_b)].append(mean_a - mean_b)

        for measure in measures:
            values = np.asarray([float(row[measure]) for row in members])
            for group_a, group_b in contrasts:
                mask_a = groups == group_a
                mask_b = groups == group_b
                mean_a = float(np.mean(values[mask_a])) if mask_a.any() else None
                mean_b = float(np.mean(values[mask_b])) if mask_b.any() else None
                draws = bootstrap_values[(measure, group_a, group_b)]
                low, high = (
                    (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))
                    if draws
                    else (None, None)
                )
                output.append(
                    {
                        "model_arm": arm,
                        "prior_family": arm.removesuffix("_existing"),
                        "outcome": measure,
                        "contrast": f"{group_a}_minus_{group_b}",
                        "mean_a": mean_a,
                        "mean_b": mean_b,
                        "difference": None if mean_a is None or mean_b is None else mean_a - mean_b,
                        "ci95_low": low,
                        "ci95_high": high,
                        "team_checkpoints_a": int(mask_a.sum()),
                        "team_checkpoints_b": int(mask_b.sum()),
                        "valid_bootstrap_repetitions": len(draws),
                        "bootstrap_repetitions": repetitions,
                        "bootstrap": "two-way pigeonhole: team_id x season-checkpoint",
                    }
                )
    return output


def _design_matrix(
    rows: list[dict[str, Any]], *, use_absolute_surprise: bool
) -> tuple[np.ndarray, list[str]]:
    strength = np.asarray([float(row["preseason_strength"]) for row in rows])
    uncertainty = np.asarray([float(row["preseason_rank_sd_normalized"]) for row in rows])
    game_count = np.log1p(
        np.asarray([int(row["prior_surprise_game_count"]) for row in rows], dtype=float)
    )
    if use_absolute_surprise:
        surprise = np.abs(np.asarray([float(row["accumulated_surprise"]) for row in rows]))
        surprise_name = "absolute_accumulated_surprise"
    else:
        surprise = np.asarray([float(row["accumulated_surprise"]) for row in rows])
        surprise_name = "accumulated_surprise"
    strength = strength - float(np.mean(strength))
    uncertainty = uncertainty - float(np.mean(uncertainty))
    game_count = game_count - float(np.mean(game_count))
    surprise = surprise - float(np.mean(surprise))
    columns = [np.ones(len(rows)), surprise, strength]
    names = ["intercept", surprise_name, "preseason_strength"]
    if not use_absolute_surprise:
        columns.append(surprise * strength)
        names.append("accumulated_surprise_x_preseason_strength")
    columns.extend((uncertainty, game_count))
    names.extend(("preseason_uncertainty", "log1p_prior_game_count"))

    seasons = sorted({int(row["season"]) for row in rows})
    checkpoints = sorted({int(row["checkpoint"]) for row in rows})
    for season in seasons[1:]:
        columns.append(np.asarray([int(row["season"]) == season for row in rows], dtype=float))
        names.append(f"season_{season}")
    for checkpoint in checkpoints[1:]:
        columns.append(
            np.asarray([int(row["checkpoint"]) == checkpoint for row in rows], dtype=float)
        )
        names.append(f"checkpoint_{checkpoint}")
    design = np.column_stack(columns)
    active = np.linalg.norm(design, axis=0) > 0
    return design[:, active], [name for name, keep in zip(names, active, strict=True) if keep]


def _fit_wls(design: np.ndarray, outcome: np.ndarray, weights: np.ndarray) -> np.ndarray:
    active = weights > 0
    weighted_design = design[active] * np.sqrt(weights[active, None])
    weighted_outcome = outcome[active] * np.sqrt(weights[active])
    return np.linalg.lstsq(weighted_design, weighted_outcome, rcond=None)[0]


def analyze_surprise_relationships(
    rows: list[dict[str, Any]],
    *,
    repetitions: int = 1000,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Fit descriptive family-specific OLS models with two-way cluster bootstrap."""
    eligible = [
        row
        for row in rows
        if row["surprise_group_eligible"]
        and int(row["checkpoint"]) > 1
        and row["signed_location_change_since_previous_normalized"] is not None
    ]
    output: list[dict[str, Any]] = []
    model_specs: tuple[tuple[str, str, str, bool], ...] = (
        (
            "directional_location_movement",
            "signed_location_change_since_previous_normalized",
            "accumulated_surprise",
            False,
        ),
        (
            "absolute_location_movement",
            "absolute_location_change_since_previous_normalized",
            "absolute_accumulated_surprise",
            True,
        ),
        (
            "rank_sd_change",
            "rank_sd_change_since_previous_normalized",
            "accumulated_surprise",
            False,
        ),
        (
            "central80_width_change",
            "central80_width_change_since_previous_normalized",
            "accumulated_surprise",
            False,
        ),
    )
    for family in FAMILIES:
        arm = MODEL_ARMS[family]
        members = [row.copy() for row in eligible if row["model_arm"] == arm]
        for row in members:
            row["absolute_location_change_since_previous_normalized"] = abs(
                float(row["signed_location_change_since_previous_normalized"])
            )
        if not members:
            continue
        teams = sorted({str(row["team_id"]) for row in members})
        snapshots = sorted({f"{row['season']}:{row['checkpoint']}" for row in members})
        team_index = {value: index for index, value in enumerate(teams)}
        snapshot_index = {value: index for index, value in enumerate(snapshots)}
        team_codes = np.asarray([team_index[str(row["team_id"])] for row in members])
        snapshot_codes = np.asarray(
            [snapshot_index[f"{row['season']}:{row['checkpoint']}"] for row in members]
        )
        for model_name, outcome_name, focus_term, use_abs in model_specs:
            model_rows = [row for row in members if row[outcome_name] is not None]
            if not model_rows:
                continue
            index = np.arange(len(model_rows), dtype=int)
            design, terms = _design_matrix(model_rows, use_absolute_surprise=use_abs)
            outcome = np.asarray([float(row[outcome_name]) for row in model_rows])
            observed = _fit_wls(design, outcome, np.ones(len(model_rows)))
            model_team_codes = team_codes[index]
            model_snapshot_codes = snapshot_codes[index]
            rng = np.random.default_rng(_seed_for(seed, f"regression:{arm}:{model_name}"))
            draws: list[list[float]] = [[] for _ in terms]
            for _ in range(repetitions):
                weights = _two_way_weights(
                    rng,
                    model_team_codes,
                    model_snapshot_codes,
                    len(teams),
                    len(snapshots),
                )
                coefficients = _fit_wls(design, outcome, weights)
                for term_index, coefficient in enumerate(coefficients):
                    if math.isfinite(float(coefficient)):
                        draws[term_index].append(float(coefficient))
            for term_index, term in enumerate(terms):
                if term == "intercept" or (
                    focus_term not in {"accumulated_surprise", "absolute_accumulated_surprise"}
                    and term != "accumulated_surprise_x_preseason_strength"
                ):
                    continue
                samples = draws[term_index]
                low, high = (
                    (float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))
                    if samples
                    else (None, None)
                )
                output.append(
                    {
                        "model_arm": arm,
                        "prior_family": family,
                        "analysis": model_name,
                        "outcome": outcome_name,
                        "term": term,
                        "estimate": float(observed[term_index]),
                        "ci95_low": low,
                        "ci95_high": high,
                        "observations": len(model_rows),
                        "team_ids": len(teams),
                        "season_checkpoint_clusters": len(snapshots),
                        "valid_bootstrap_repetitions": len(samples),
                        "bootstrap_repetitions": repetitions,
                        "bootstrap": "two-way pigeonhole: team_id x season-checkpoint",
                    }
                )
    return output


def paired_family_movement_differences(
    rows: list[dict[str, Any]],
    *,
    repetitions: int = 1000,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Compare History minus Context movement on matched team-checkpoints."""
    by_key = {
        (
            int(row["season"]),
            int(row["checkpoint"]),
            str(row["team_id"]),
            str(row["prior_family"]),
        ): row
        for row in rows
    }
    if len(by_key) != len(rows):
        raise ValueError("duplicate family/team-checkpoint row in paired comparison")
    metrics = (
        "signed_location_change_from_preseason_normalized",
        "signed_location_change_since_previous_normalized",
        "rank_sd_change_from_preseason_normalized",
        "rank_sd_change_since_previous_normalized",
        "central80_width_change_from_preseason_normalized",
        "central80_width_change_since_previous_normalized",
        "kl_posterior_from_prior",
        "total_variation_since_previous",
    )
    output: list[dict[str, Any]] = []
    for metric in metrics:
        for scope in ("all_checkpoints_after_preseason", "checkpoint_7"):
            pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            base_keys = sorted({key[:3] for key in by_key})
            for base_key in base_keys:
                history = by_key[(*base_key, "History")]
                if scope == "all_checkpoints_after_preseason" and int(history["checkpoint"]) <= 1:
                    continue
                if scope == "checkpoint_7" and int(history["checkpoint"]) != 7:
                    continue
                context = by_key.get((*base_key, "Context"))
                if context is None:
                    raise ValueError(f"missing paired Context row {base_key}")
                if history[metric] is None or context[metric] is None:
                    continue
                pairs.append((history, context))
            if not pairs:
                continue
            differences = np.asarray(
                [float(history[metric]) - float(context[metric]) for history, context in pairs]
            )
            teams = sorted({str(history["team_id"]) for history, _ in pairs})
            snapshots = sorted(
                {f"{history['season']}:{history['checkpoint']}" for history, _ in pairs}
            )
            team_index = {value: index for index, value in enumerate(teams)}
            snapshot_index = {value: index for index, value in enumerate(snapshots)}
            team_codes = np.asarray(
                [team_index[str(history["team_id"])] for history, _ in pairs]
            )
            snapshot_codes = np.asarray(
                [snapshot_index[f"{history['season']}:{history['checkpoint']}"] for history, _ in pairs]
            )
            rng = np.random.default_rng(_seed_for(seed, f"family-diff:{metric}:{scope}"))
            draws: list[float] = []
            for _ in range(repetitions):
                weights = _two_way_weights(
                    rng, team_codes, snapshot_codes, len(teams), len(snapshots)
                )
                estimate = _weighted_mean(differences, weights)
                if estimate is not None:
                    draws.append(estimate)
            low, high = (
                (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))
                if draws
                else (None, None)
            )
            output.append(
                {
                    "comparison": "History_minus_Context",
                    "scope": scope,
                    "outcome": metric,
                    "difference": float(np.mean(differences)),
                    "median_paired_difference": float(np.median(differences)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "team_checkpoints": len(pairs),
                    "team_ids": len(teams),
                    "season_checkpoint_clusters": len(snapshots),
                    "valid_bootstrap_repetitions": len(draws),
                    "bootstrap_repetitions": repetitions,
                    "bootstrap": "two-way pigeonhole: team_id x season-checkpoint",
                }
            )
    return output


def paired_family_surprise_slope_differences(
    rows: list[dict[str, Any]],
    *,
    repetitions: int = 1000,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Bootstrap Context-minus-History surprise slopes on paired checkpoints."""
    eligible = [
        row
        for row in rows
        if row["surprise_group_eligible"]
        and int(row["checkpoint"]) > 1
        and row["signed_location_change_since_previous_normalized"] is not None
    ]
    by_family = {
        family: sorted(
            [row.copy() for row in eligible if row["prior_family"] == family],
            key=lambda row: (int(row["season"]), int(row["checkpoint"]), int(row["team_id"])),
        )
        for family in FAMILIES
    }
    keys = [
        (int(row["season"]), int(row["checkpoint"]), str(row["team_id"]))
        for row in by_family["History"]
    ]
    context_keys = [
        (int(row["season"]), int(row["checkpoint"]), str(row["team_id"]))
        for row in by_family["Context"]
    ]
    if keys != context_keys:
        raise ValueError("surprise-slope family comparison lacks paired team-checkpoints")
    for family_rows in by_family.values():
        for row in family_rows:
            row["absolute_location_change_since_previous_normalized"] = abs(
                float(row["signed_location_change_since_previous_normalized"])
            )

    specs = (
        (
            "directional_location_movement",
            "signed_location_change_since_previous_normalized",
            False,
            "accumulated_surprise",
        ),
        (
            "absolute_location_movement",
            "absolute_location_change_since_previous_normalized",
            True,
            "absolute_accumulated_surprise",
        ),
        ("rank_sd_change", "rank_sd_change_since_previous_normalized", False, "accumulated_surprise"),
        (
            "central80_width_change",
            "central80_width_change_since_previous_normalized",
            False,
            "accumulated_surprise",
        ),
    )
    teams = sorted({key[2] for key in keys})
    snapshots = sorted({f"{key[0]}:{key[1]}" for key in keys})
    team_index = {value: index for index, value in enumerate(teams)}
    snapshot_index = {value: index for index, value in enumerate(snapshots)}
    team_codes = np.asarray([team_index[key[2]] for key in keys])
    snapshot_codes = np.asarray([snapshot_index[f"{key[0]}:{key[1]}"] for key in keys])
    output: list[dict[str, Any]] = []

    for model_name, outcome_name, use_abs, slope_term in specs:
        h_rows = by_family["History"]
        c_rows = by_family["Context"]
        h_design, h_terms = _design_matrix(h_rows, use_absolute_surprise=use_abs)
        c_design, c_terms = _design_matrix(c_rows, use_absolute_surprise=use_abs)
        if h_terms != c_terms or len(h_rows) != len(c_rows):
            raise ValueError(f"family-specific design mismatch for {model_name}")
        term_index = h_terms.index(slope_term)
        h_outcome = np.asarray([float(row[outcome_name]) for row in h_rows])
        c_outcome = np.asarray([float(row[outcome_name]) for row in c_rows])
        h_observed = _fit_wls(h_design, h_outcome, np.ones(len(h_rows)))[term_index]
        c_observed = _fit_wls(c_design, c_outcome, np.ones(len(c_rows)))[term_index]
        rng = np.random.default_rng(_seed_for(seed, f"family-slope:{model_name}"))
        draws: list[float] = []
        for _ in range(repetitions):
            weights = _two_way_weights(
                rng, team_codes, snapshot_codes, len(teams), len(snapshots)
            )
            h_slope = _fit_wls(h_design, h_outcome, weights)[term_index]
            c_slope = _fit_wls(c_design, c_outcome, weights)[term_index]
            if math.isfinite(float(h_slope)) and math.isfinite(float(c_slope)):
                draws.append(float(c_slope - h_slope))
        low, high = (
            (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))
            if draws
            else (None, None)
        )
        output.append(
            {
                "comparison": "Context_minus_History_slope",
                "analysis": model_name,
                "outcome": outcome_name,
                "term": slope_term,
                "context_estimate": float(c_observed),
                "history_estimate": float(h_observed),
                "difference": float(c_observed - h_observed),
                "ci95_low": low,
                "ci95_high": high,
                "team_checkpoints": len(h_rows),
                "team_ids": len(teams),
                "season_checkpoint_clusters": len(snapshots),
                "valid_bootstrap_repetitions": len(draws),
                "bootstrap_repetitions": repetitions,
                "bootstrap": "paired two-way pigeonhole: team_id x season-checkpoint",
            }
        )
    return output


def select_trajectories(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the most negative and positive checkpoint-7 surprises, not outcomes."""
    grouped: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if (
            int(row["checkpoint"]) == 7
            and row["surprise_group_eligible"]
            and int(row["prior_surprise_game_count"]) >= MIN_SURPRISE_GAMES
        ):
            grouped[(int(row["season"]), str(row["team_id"]))][
                str(row["prior_family"])
            ] = row
    candidates: list[dict[str, Any]] = []
    for (season, team_id), families in sorted(grouped.items()):
        if not all(family in families for family in FAMILIES):
            continue
        score = float(
            np.mean([float(families[family]["accumulated_surprise"]) for family in FAMILIES])
        )
        candidates.append(
            {
                "season": season,
                "team_id": team_id,
                "team_name": str(families["History"]["team_name"]),
                "selection_score_mean_family_surprise": score,
                "history_accumulated_surprise": float(
                    families["History"]["accumulated_surprise"]
                ),
                "context_accumulated_surprise": float(
                    families["Context"]["accumulated_surprise"]
                ),
                "history_prior_surprise_game_count": int(
                    families["History"]["prior_surprise_game_count"]
                ),
                "context_prior_surprise_game_count": int(
                    families["Context"]["prior_surprise_game_count"]
                ),
            }
        )
    if len(candidates) < 2:
        raise ValueError("need at least two checkpoint-7 team-seasons for trajectories")
    lowest = min(
        candidates,
        key=lambda row: (
            row["selection_score_mean_family_surprise"],
            row["season"],
            int(row["team_id"]),
        ),
    )
    remaining = [
        row
        for row in candidates
        if (row["season"], row["team_id"]) != (lowest["season"], lowest["team_id"])
    ]
    highest = max(
        remaining,
        key=lambda row: (
            row["selection_score_mean_family_surprise"],
            -row["season"],
            -int(row["team_id"]),
        ),
    )
    selected = []
    for case_id, rule, candidate in (
        ("negative-surprise", "minimum mean History/Context checkpoint-7 surprise", lowest),
        ("positive-surprise", "maximum mean History/Context checkpoint-7 surprise", highest),
    ):
        selected.append(
            {
                **candidate,
                "trajectory_id": case_id,
                "selection_rule": rule,
            }
        )
    return selected


def build_selected_trajectory_rows(
    rows: list[dict[str, Any]], selected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    selection_by_key = {
        (int(item["season"]), str(item["team_id"])): item for item in selected
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        selection = selection_by_key.get((int(row["season"]), str(row["team_id"])))
        if selection is None:
            continue
        output.append(
            {
                **row,
                "trajectory_id": selection["trajectory_id"],
                "selection_rule": selection["selection_rule"],
                "selection_score_mean_family_surprise": selection[
                    "selection_score_mean_family_surprise"
                ],
            }
        )
    return output
