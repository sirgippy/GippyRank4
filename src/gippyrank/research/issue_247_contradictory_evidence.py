"""Analyze accumulated in-season surprise against subsequent game errors."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from gippyrank.research import issue_246_error_persistence as issue246

EXISTING_ARMS = (
    ("History_existing", "History", "existing"),
    ("Context_existing", "Context", "existing"),
)
MIN_PRIOR_GAMES = 3
EXAMPLE_MIN_PRIOR_GAMES = 5
SURPRISE_GROUP_THRESHOLD = 0.5

TEAM_GAME_FIELDS = (
    "season",
    "model_arm",
    "prior_family",
    "policy",
    "team_id",
    "team",
    "opponent_id",
    "opponent",
    "game_id",
    "forecast_id",
    "kickoff_utc",
    "cutoff_utc",
    "is_home",
    "forecast_population",
    "team_game_number",
    "actual_margin",
    "predicted_margin",
    "signed_error",
    "margin_80_interval_half_width",
    "standardized_surprise",
    "preseason_expected_rank",
    "preseason_rank_sd",
    "preseason_rank_strength_percentile",
    "preseason_rank_uncertainty_fraction",
    "preseason_rank_population",
    "preseason_rank_top10_probability",
    "preseason_rank_pmf_sha256",
)

OBSERVATION_FIELDS = (
    "season",
    "model_arm",
    "prior_family",
    "policy",
    "team_id",
    "team",
    "opponent_id",
    "opponent",
    "game_id",
    "forecast_id",
    "kickoff_utc",
    "cutoff_utc",
    "is_home",
    "forecast_population",
    "team_game_number",
    "actual_margin",
    "predicted_margin",
    "next_signed_error",
    "margin_80_interval_half_width",
    "next_standardized_error",
    "candidate_previous_game_count",
    "eligible_prior_game_count",
    "excluded_previous_game_count",
    "eligible_previous_game_ids",
    "accumulated_surprise",
    "surprise_group",
    "primary_eligible",
    "exclusion_reason",
    "preseason_expected_rank",
    "preseason_rank_sd",
    "preseason_rank_strength_percentile",
    "preseason_rank_uncertainty_fraction",
    "preseason_rank_population",
    "preseason_rank_top10_probability",
    "preseason_rank_pmf_sha256",
)

GROUP_SUMMARY_FIELDS = (
    "summary_level",
    "season",
    "model_arm",
    "prior_family",
    "surprise_group",
    "observations",
    "team_count",
    "mean_next_signed_error",
    "mean_next_signed_error_ci95_low",
    "mean_next_signed_error_ci95_high",
    "mean_next_standardized_error",
    "mean_next_standardized_error_ci95_low",
    "mean_next_standardized_error_ci95_high",
)

CONTRAST_FIELDS = (
    "model_arm",
    "prior_family",
    "contrast",
    "metric",
    "difference",
    "ci95_low",
    "ci95_high",
    "observations_a",
    "observations_b",
)

MODEL_FIELDS = (
    "summary_level",
    "season",
    "model_arm",
    "prior_family",
    "outcome",
    "term",
    "estimate",
    "ci95_low",
    "ci95_high",
    "observations",
    "team_count",
    "preseason_strength_p25",
    "preseason_strength_p75",
)

EXAMPLE_FIELDS = (
    "model_arm",
    "prior_family",
    "surprise_group",
    "selection_rule",
    "season",
    "team_id",
    "team",
    "opponent",
    "game_id",
    "team_game_number",
    "eligible_prior_game_count",
    "accumulated_surprise",
    "preseason_rank_strength_percentile",
    "preseason_rank_uncertainty_fraction",
    "predicted_margin",
    "actual_margin",
    "next_signed_error",
    "next_standardized_error",
)


def _finite(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if value in (None, ""):
        raise ValueError(f"missing numeric field {field!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric field {field!r}: {value!r}")
    return number


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timestamp must include a timezone: {value!r}")
    return parsed


def surprise_group(value: float) -> str:
    """Classify a mean interval-scaled error using frozen symmetric cutoffs."""
    if value <= -SURPRISE_GROUP_THRESHOLD:
        return "sustained_negative"
    if value >= SURPRISE_GROUP_THRESHOLD:
        return "sustained_positive"
    return "approximately_neutral"


def standardized_margin_error(signed_error: float, interval_half_width: float) -> float:
    """Scale an oriented margin error by its retained predictive uncertainty."""
    if not math.isfinite(signed_error):
        raise ValueError("signed margin error must be finite")
    if not math.isfinite(interval_half_width) or interval_half_width <= 0.0:
        raise ValueError("predictive interval half-width must be finite and positive")
    return signed_error / interval_half_width


def build_team_game_errors(
    rolling_scores: list[dict[str, str]],
    preseason_rank_priors: list[dict[str, Any]],
    evidence_records: dict[tuple[int, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add predictive-width scaling and continuous prior summaries to #246 rows."""
    score_by_id: dict[str, dict[str, str]] = {}
    for score in rolling_scores:
        forecast_id = score["forecast_id"]
        if forecast_id in score_by_id:
            raise ValueError(f"duplicate #201 forecast ID: {forecast_id}")
        score_by_id[forecast_id] = score

    prior_by_key = {
        (int(row["season"]), str(row["model_arm"]), str(row["team_id"])): row
        for row in preseason_rank_priors
    }
    oriented = issue246.build_team_game_errors(
        rolling_scores, preseason_rank_priors, evidence_records
    )
    output: list[dict[str, Any]] = []
    for row in oriented:
        if row["model_arm"] not in {arm for arm, _, _ in EXISTING_ARMS}:
            continue
        score = score_by_id[str(row["forecast_id"])]
        lower = _finite(score, "margin_interval_80_low")
        upper = _finite(score, "margin_interval_80_high")
        half_width = (upper - lower) / 2.0
        if half_width <= 0.0:
            raise ValueError(
                f"non-positive #201 80% interval half-width for {row['forecast_id']}"
            )
        prior = prior_by_key[
            (int(row["season"]), str(row["model_arm"]), str(row["team_id"]))
        ]
        output.append(
            {
                **row,
                "margin_80_interval_half_width": half_width,
                "standardized_surprise": standardized_margin_error(
                    float(row["signed_error"]), half_width
                ),
                "preseason_expected_rank": prior["preseason_expected_rank"],
                "preseason_rank_sd": prior["preseason_rank_sd"],
                "preseason_rank_strength_percentile": prior[
                    "preseason_rank_strength_percentile"
                ],
                "preseason_rank_uncertainty_fraction": prior[
                    "preseason_rank_uncertainty_fraction"
                ],
                "preseason_rank_population": prior["preseason_rank_population"],
                "preseason_rank_top10_probability": prior[
                    "preseason_rank_top10_probability"
                ],
                "preseason_rank_pmf_sha256": prior["preseason_rank_pmf_sha256"],
            }
        )
    return sorted(
        output,
        key=lambda row: (
            int(row["season"]),
            str(row["model_arm"]),
            str(row["team_id"]),
            int(row["team_game_number"]),
        ),
    )


def build_observations(
    team_games: list[dict[str, Any]],
    evidence_records: dict[tuple[int, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build each future FBS team-game with its exact evidence-available history."""
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in team_games:
        grouped[
            (int(row["season"]), str(row["model_arm"]), str(row["team_id"]))
        ].append(row)

    observations: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items()):
        members.sort(key=lambda row: int(row["team_game_number"]))
        for index, target in enumerate(members):
            evidence_key = (
                int(target["season"]),
                str(target["cutoff_utc"]),
                str(target["policy"]),
            )
            evidence = evidence_records.get(evidence_key)
            if evidence is None:
                raise ValueError(f"missing #201 evidence for target {evidence_key}")
            included_ids = evidence["included_game_id_set"]
            target_game_id = str(target["game_id"])
            if target_game_id in included_ids:
                raise ValueError(
                    f"target game appears in its own evidence: {target_game_id}"
                )
            cutoff = _parse_utc(str(target["cutoff_utc"]))
            prior_candidates = members[:index]
            eligible_prior: list[dict[str, Any]] = []
            for prior in prior_candidates:
                prior_game_id = str(prior["game_id"])
                if prior_game_id not in included_ids:
                    continue
                if _parse_utc(str(prior["kickoff_utc"])) >= cutoff:
                    raise ValueError(
                        f"prior game {prior_game_id} was not complete before target cutoff"
                    )
                eligible_prior.append(prior)

            eligible_count = len(eligible_prior)
            mean_surprise = (
                math.fsum(float(row["standardized_surprise"]) for row in eligible_prior)
                / eligible_count
                if eligible_count
                else None
            )
            primary_eligible = eligible_count >= MIN_PRIOR_GAMES
            if primary_eligible:
                reason = ""
                group = surprise_group(float(mean_surprise))
            else:
                reason = "fewer_than_three_evidence_eligible_prior_games"
                group = ""
            observations.append(
                {
                    **target,
                    "season_progress_bin": issue246.season_progress_bin(
                        int(target["team_game_number"])
                    ),
                    "next_signed_error": target["signed_error"],
                    "next_standardized_error": target["standardized_surprise"],
                    "candidate_previous_game_count": index,
                    "eligible_prior_game_count": eligible_count,
                    "excluded_previous_game_count": index - eligible_count,
                    "eligible_previous_game_ids": "|".join(
                        str(row["game_id"]) for row in eligible_prior
                    ),
                    "accumulated_surprise": mean_surprise,
                    "surprise_group": group,
                    "primary_eligible": primary_eligible,
                    "exclusion_reason": reason,
                }
            )
    return sorted(
        observations,
        key=lambda row: (
            int(row["season"]),
            str(row["model_arm"]),
            str(row["team_id"]),
            int(row["team_game_number"]),
        ),
    )


def _scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.mean(values))
    deviation = float(np.std(values))
    return center, deviation if deviation > 0.0 else 1.0


def _design_matrix(
    rows: list[dict[str, Any]], *, include_season_effects: bool
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, float]]:
    surprise = np.asarray([float(row["accumulated_surprise"]) for row in rows])
    strength = np.asarray(
        [float(row["preseason_rank_strength_percentile"]) for row in rows]
    )
    uncertainty = np.asarray(
        [float(row["preseason_rank_uncertainty_fraction"]) for row in rows]
    )
    history_count = np.asarray(
        [float(row["eligible_prior_game_count"]) for row in rows]
    )
    game_number = np.asarray([float(row["team_game_number"]) for row in rows])
    strength_mean, strength_sd = _scale(strength)
    uncertainty_mean, uncertainty_sd = _scale(uncertainty)
    count_mean, count_sd = _scale(history_count)
    game_mean, game_sd = _scale(game_number)
    strength_z = (strength - strength_mean) / strength_sd
    uncertainty_z = (uncertainty - uncertainty_mean) / uncertainty_sd
    count_z = (history_count - count_mean) / count_sd
    game_z = (game_number - game_mean) / game_sd

    columns = [
        np.ones(len(rows)),
        surprise,
        strength_z,
        uncertainty_z,
        surprise * strength_z,
        count_z,
        game_z,
    ]
    terms = [
        "intercept",
        "accumulated_surprise",
        "preseason_strength_z",
        "preseason_uncertainty_z",
        "surprise_x_preseason_strength_z",
        "eligible_prior_game_count_z",
        "target_team_game_number_z",
    ]
    if include_season_effects:
        seasons = sorted({int(row["season"]) for row in rows})
        for season in seasons[1:]:
            columns.append(
                np.asarray([float(int(row["season"]) == season) for row in rows])
            )
            terms.append(f"season_{season}")
    design = np.column_stack(columns)
    outcomes = np.column_stack(
        (
            np.asarray([float(row["next_signed_error"]) for row in rows]),
            np.asarray([float(row["next_standardized_error"]) for row in rows]),
        )
    )
    scales = {
        "preseason_strength_mean": strength_mean,
        "preseason_strength_sd": strength_sd,
        "preseason_uncertainty_mean": uncertainty_mean,
        "preseason_uncertainty_sd": uncertainty_sd,
        "eligible_prior_game_count_mean": count_mean,
        "eligible_prior_game_count_sd": count_sd,
        "target_team_game_number_mean": game_mean,
        "target_team_game_number_sd": game_sd,
    }
    return design, outcomes, tuple(terms), scales


def _cluster_codes(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    teams = sorted({str(row["team_id"]) for row in rows})
    games = sorted({f"{row['season']}:{row['game_id']}" for row in rows})
    team_map = {value: index for index, value in enumerate(teams)}
    game_map = {value: index for index, value in enumerate(games)}
    team_codes = np.asarray([team_map[str(row["team_id"])] for row in rows])
    game_codes = np.asarray(
        [game_map[f"{row['season']}:{row['game_id']}"] for row in rows]
    )
    return team_codes, game_codes, len(teams), len(games)


def _bootstrap_weights(
    rng: np.random.Generator,
    codes: tuple[np.ndarray, np.ndarray, int, int],
) -> np.ndarray:
    team_codes, game_codes, team_count, game_count = codes
    team_counts = np.bincount(
        rng.integers(0, team_count, size=team_count), minlength=team_count
    )
    game_counts = np.bincount(
        rng.integers(0, game_count, size=game_count), minlength=game_count
    )
    return (team_counts[team_codes] * game_counts[game_codes]).astype(float)


def _fit_wls(
    design: np.ndarray, outcomes: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    active = weights > 0.0
    root_weights = np.sqrt(weights[active])
    weighted_design = design[active] * root_weights[:, None]
    weighted_outcomes = outcomes[active] * root_weights[:, None]
    return np.linalg.lstsq(weighted_design, weighted_outcomes, rcond=None)[0]


def _percentile_interval(values: list[float]) -> tuple[float | None, float | None]:
    finite = np.asarray(
        [value for value in values if math.isfinite(value)], dtype=float
    )
    if finite.size == 0:
        return None, None
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def _seed_for(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    active = weights > 0.0
    if not np.any(active):
        return None
    return float(np.dot(values[active], weights[active]) / np.sum(weights[active]))


def _group_means(
    rows: list[dict[str, Any]], weights: np.ndarray, group_name: str
) -> tuple[float | None, float | None]:
    active = np.asarray([row["surprise_group"] == group_name for row in rows])
    group_weights = weights * active
    raw = np.asarray([float(row["next_signed_error"]) for row in rows])
    standardized = np.asarray([float(row["next_standardized_error"]) for row in rows])
    return (
        _weighted_mean(raw, group_weights),
        _weighted_mean(standardized, group_weights),
    )


def analyze_arm(
    rows: list[dict[str, Any]], *, repetitions: int, seed: int, label: str
) -> dict[str, Any]:
    """Fit pooled results and dependence-aware group summaries for one model arm."""
    if not rows:
        raise ValueError(f"no eligible observations for {label}")
    design, outcomes, terms, scales = _design_matrix(rows, include_season_effects=True)
    baseline_weights = np.ones(len(rows), dtype=float)
    coefficients = _fit_wls(design, outcomes, baseline_weights)
    term_index = {term: index for index, term in enumerate(terms)}
    strength_values = np.asarray(
        [float(row["preseason_rank_strength_percentile"]) for row in rows]
    )
    strength_p25, strength_p75 = np.quantile(strength_values, [0.25, 0.75])
    strength_z_p25 = (float(strength_p25) - scales["preseason_strength_mean"]) / scales[
        "preseason_strength_sd"
    ]
    strength_z_p75 = (float(strength_p75) - scales["preseason_strength_mean"]) / scales[
        "preseason_strength_sd"
    ]
    codes = _cluster_codes(rows)
    rng = np.random.default_rng(_seed_for(seed, label))

    coefficient_draws: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    shift_draws: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    contrast_draws: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    group_draws: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    group_names = (
        "sustained_negative",
        "approximately_neutral",
        "sustained_positive",
    )
    raw_values = outcomes[:, 0]
    standardized_values = outcomes[:, 1]
    for _ in range(repetitions):
        weights = _bootstrap_weights(rng, codes)
        if not np.any(weights > 0.0):
            continue
        fitted = _fit_wls(design, outcomes, weights)
        for outcome_index in (0, 1):
            outcome_name = (
                "margin_points" if outcome_index == 0 else "standardized_80_widths"
            )
            for term in (
                "accumulated_surprise",
                "preseason_strength_z",
                "preseason_uncertainty_z",
                "surprise_x_preseason_strength_z",
            ):
                coefficient_draws[(outcome_name, outcome_index, term)].append(
                    float(fitted[term_index[term], outcome_index])
                )
            beta_surprise = float(
                fitted[term_index["accumulated_surprise"], outcome_index]
            )
            beta_interaction = float(
                fitted[term_index["surprise_x_preseason_strength_z"], outcome_index]
            )
            for percentile, strength_z in (
                ("p25", strength_z_p25),
                ("p75", strength_z_p75),
            ):
                shift_draws[(outcome_name, percentile, outcome_index)].append(
                    -beta_surprise - beta_interaction * strength_z
                )
            shift_draws[(outcome_name, "p75_minus_p25", outcome_index)].append(
                -beta_interaction * (strength_z_p75 - strength_z_p25)
            )

        means = {group: _group_means(rows, weights, group) for group in group_names}
        for group, (raw_mean, standardized_mean) in means.items():
            if raw_mean is not None:
                group_draws[(group, "margin_points", 0)].append(raw_mean)
            if standardized_mean is not None:
                group_draws[(group, "standardized_80_widths", 1)].append(
                    standardized_mean
                )
        for contrast, group_a, group_b in (
            (
                "sustained_negative_minus_neutral",
                "sustained_negative",
                "approximately_neutral",
            ),
            (
                "sustained_positive_minus_neutral",
                "sustained_positive",
                "approximately_neutral",
            ),
            (
                "sustained_positive_minus_negative",
                "sustained_positive",
                "sustained_negative",
            ),
        ):
            for outcome_name, index in (
                ("margin_points", 0),
                ("standardized_80_widths", 1),
            ):
                a = means[group_a][index]
                b = means[group_b][index]
                if a is not None and b is not None:
                    contrast_draws[(contrast, index)].append(float(a - b))

    model_rows: list[dict[str, Any]] = []
    outcome_names = ("margin_points", "standardized_80_widths")
    for outcome_index, outcome_name in enumerate(outcome_names):
        for term in (
            "accumulated_surprise",
            "preseason_strength_z",
            "preseason_uncertainty_z",
            "surprise_x_preseason_strength_z",
        ):
            estimate = float(coefficients[term_index[term], outcome_index])
            low, high = _percentile_interval(
                coefficient_draws[(outcome_name, outcome_index, term)]
            )
            model_rows.append(
                {
                    "summary_level": "pooled_model_arm",
                    "season": "2022-2025",
                    "model_arm": label,
                    "outcome": outcome_name,
                    "term": term,
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "observations": len(rows),
                    "team_count": len({str(row["team_id"]) for row in rows}),
                    "preseason_strength_p25": float(strength_p25),
                    "preseason_strength_p75": float(strength_p75),
                }
            )
        for percentile in ("p25", "p75", "p75_minus_p25"):
            values = shift_draws[(outcome_name, percentile, outcome_index)]
            low, high = _percentile_interval(values)
            if percentile == "p75_minus_p25":
                estimate = -float(
                    coefficients[
                        term_index["surprise_x_preseason_strength_z"], outcome_index
                    ]
                ) * (strength_z_p75 - strength_z_p25)
                term = "surprise_minus_one_vs_zero_shift_" + percentile
            else:
                strength_z = strength_z_p25 if percentile == "p25" else strength_z_p75
                estimate = -float(
                    coefficients[term_index["accumulated_surprise"], outcome_index]
                    + coefficients[
                        term_index["surprise_x_preseason_strength_z"], outcome_index
                    ]
                    * strength_z
                )
                term = f"surprise_minus_one_vs_zero_shift_at_{percentile}"
            model_rows.append(
                {
                    "summary_level": "pooled_model_arm",
                    "season": "2022-2025",
                    "model_arm": label,
                    "outcome": outcome_name,
                    "term": term,
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "observations": len(rows),
                    "team_count": len({str(row["team_id"]) for row in rows}),
                    "preseason_strength_p25": float(strength_p25),
                    "preseason_strength_p75": float(strength_p75),
                }
            )

    group_rows: list[dict[str, Any]] = []
    for group in group_names:
        members = [row for row in rows if row["surprise_group"] == group]
        if not members:
            continue
        group_mask = np.asarray([row["surprise_group"] == group for row in rows])
        raw_estimate = _weighted_mean(raw_values, group_mask.astype(float))
        standardized_estimate = _weighted_mean(
            standardized_values, group_mask.astype(float)
        )
        raw_low, raw_high = _percentile_interval(
            group_draws[(group, "margin_points", 0)]
        )
        std_low, std_high = _percentile_interval(
            group_draws[(group, "standardized_80_widths", 1)]
        )
        group_rows.append(
            {
                "summary_level": "pooled_model_arm",
                "season": "2022-2025",
                "model_arm": label,
                "surprise_group": group,
                "observations": len(members),
                "team_count": len({str(row["team_id"]) for row in members}),
                "mean_next_signed_error": raw_estimate,
                "mean_next_signed_error_ci95_low": raw_low,
                "mean_next_signed_error_ci95_high": raw_high,
                "mean_next_standardized_error": standardized_estimate,
                "mean_next_standardized_error_ci95_low": std_low,
                "mean_next_standardized_error_ci95_high": std_high,
            }
        )

    contrast_rows: list[dict[str, Any]] = []
    for contrast, group_a, group_b in (
        (
            "sustained_negative_minus_neutral",
            "sustained_negative",
            "approximately_neutral",
        ),
        (
            "sustained_positive_minus_neutral",
            "sustained_positive",
            "approximately_neutral",
        ),
        (
            "sustained_positive_minus_negative",
            "sustained_positive",
            "sustained_negative",
        ),
    ):
        count_a = sum(row["surprise_group"] == group_a for row in rows)
        count_b = sum(row["surprise_group"] == group_b for row in rows)
        for outcome_name, outcome_index in (
            ("margin_points", 0),
            ("standardized_80_widths", 1),
        ):
            means = {
                group: _group_means(rows, np.ones(len(rows)), group)[outcome_index]
                for group in (group_a, group_b)
            }
            if means[group_a] is None or means[group_b] is None:
                continue
            low, high = _percentile_interval(contrast_draws[(contrast, outcome_index)])
            contrast_rows.append(
                {
                    "model_arm": label,
                    "contrast": contrast,
                    "metric": outcome_name,
                    "difference": means[group_a] - means[group_b],
                    "ci95_low": low,
                    "ci95_high": high,
                    "observations_a": count_a,
                    "observations_b": count_b,
                }
            )

    return {
        "model_rows": model_rows,
        "group_rows": group_rows,
        "contrast_rows": contrast_rows,
        "scales": scales,
        "bootstrap_repetitions": repetitions,
    }


def analyze_season(
    rows: list[dict[str, Any]], *, repetitions: int, seed: int, label: str
) -> list[dict[str, Any]]:
    """Estimate adjusted surprise slopes separately by season and outcome."""
    if not rows:
        return []
    design, outcomes, terms, _ = _design_matrix(rows, include_season_effects=False)
    surprise_index = terms.index("accumulated_surprise")
    baseline = _fit_wls(design, outcomes, np.ones(len(rows), dtype=float))
    codes = _cluster_codes(rows)
    rng = np.random.default_rng(_seed_for(seed, label))
    draws: list[list[float]] = [[], []]
    for _ in range(repetitions):
        weights = _bootstrap_weights(rng, codes)
        if not np.any(weights > 0.0):
            continue
        fitted = _fit_wls(design, outcomes, weights)
        for outcome_index in (0, 1):
            draws[outcome_index].append(float(fitted[surprise_index, outcome_index]))
    output: list[dict[str, Any]] = []
    for outcome_index, outcome_name in enumerate(
        ("margin_points", "standardized_80_widths")
    ):
        low, high = _percentile_interval(draws[outcome_index])
        output.append(
            {
                "summary_level": "season_model_arm",
                "season": int(rows[0]["season"]),
                "model_arm": label,
                "outcome": outcome_name,
                "term": "accumulated_surprise",
                "estimate": float(baseline[surprise_index, outcome_index]),
                "ci95_low": low,
                "ci95_high": high,
                "observations": len(rows),
                "team_count": len({str(row["team_id"]) for row in rows}),
                "preseason_strength_p25": None,
                "preseason_strength_p75": None,
            }
        )
    return output


def select_examples(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select predictor extremes per family without inspecting next errors."""
    eligible = [
        row
        for row in observations
        if row["primary_eligible"]
        and int(row["eligible_prior_game_count"]) >= EXAMPLE_MIN_PRIOR_GAMES
    ]
    examples: list[dict[str, Any]] = []
    for arm, family, _ in EXISTING_ARMS:
        members = [row for row in eligible if row["model_arm"] == arm]
        if not members:
            continue
        for group, selected in (
            (
                "minimum_accumulated_surprise",
                min(
                    members,
                    key=lambda row: (
                        float(row["accumulated_surprise"]),
                        int(row["season"]),
                        str(row["team_id"]),
                        int(row["team_game_number"]),
                    ),
                ),
            ),
            (
                "maximum_accumulated_surprise",
                max(
                    members,
                    key=lambda row: (
                        float(row["accumulated_surprise"]),
                        -int(row["season"]),
                        str(row["team_id"]),
                        int(row["team_game_number"]),
                    ),
                ),
            ),
        ):
            examples.append(
                {
                    "model_arm": arm,
                    "prior_family": family,
                    "surprise_group": selected["surprise_group"],
                    "selection_rule": (
                        f"{group.removesuffix('_accumulated_surprise')} accumulated surprise "
                        f"among rows with at least {EXAMPLE_MIN_PRIOR_GAMES} eligible prior games"
                    ),
                    "season": selected["season"],
                    "team_id": selected["team_id"],
                    "team": selected["team"],
                    "opponent": selected["opponent"],
                    "game_id": selected["game_id"],
                    "team_game_number": selected["team_game_number"],
                    "eligible_prior_game_count": selected["eligible_prior_game_count"],
                    "accumulated_surprise": selected["accumulated_surprise"],
                    "preseason_rank_strength_percentile": selected[
                        "preseason_rank_strength_percentile"
                    ],
                    "preseason_rank_uncertainty_fraction": selected[
                        "preseason_rank_uncertainty_fraction"
                    ],
                    "predicted_margin": selected["predicted_margin"],
                    "actual_margin": selected["actual_margin"],
                    "next_signed_error": selected["next_signed_error"],
                    "next_standardized_error": selected["next_standardized_error"],
                }
            )
    return examples
