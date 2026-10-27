"""Feasibility design for the multi-strength Quality of Results follow-up.

This module selects focal priors and controlled schedules from the frozen 2026
Context 1.4 FBS population. Its small pilot counts generated records only; it
does not run Quality of Results inference or inspect ranking outcomes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.stats import binom
from scipy.stats import t as student_t

from gippyrank.methodology import HISTORICAL_LIKELIHOOD_SHA256
from gippyrank.posterior.engine import Team
from gippyrank.posterior.snapshots import (
    load_pinned_likelihood,
    load_teams,
)
from gippyrank.research.issue_252_quality_of_results import (
    SITE_SLOTS,
    _site_location_matrices,
    expected_focal_win_probability,
    sha256_file,
)

SEASON = 2026
PRIOR_VERSION = "1.4"
# Frozen when the issue 258 design artifacts were first generated.
PINNED_SOURCE_REVISION = "1c61356957ed71a94d878cfa253cf4b6f5ae0452"
STRENGTH_TARGETS = (("elite", 5.0), ("strong", 20.0), ("middle", 65.0), ("weak", 120.0))
REFERENCE_TEAM_ID = "38"  # Colorado, the issue 252 focal prior.
SCHEDULE_TARGET_MEANS = {
    "elite": (0.60, 0.80),
    "strong": (0.50, 0.80),
    "middle": (0.50, 0.80),
    "weak": (0.20, 0.40),
    "issue_252_reference": (0.50, 0.80),
}
SCHEDULE_PROFILE_HALF_WIDTH = 0.08
PILOT_SEED_START = 2_580_000
PILOT_DRAWS = 10_000
UNCONDITIONAL_SEED_START = 2_700_000
UNCONDITIONAL_DRAWS = 500
PRIMARY_CONDITIONED_SEED_START = 2_710_000
CONTROL_CONDITIONED_SEED_START = 2_810_000
CONDITIONED_DRAWS = 200
CONDITIONED_SEARCH_LIMIT = 20_000
INFERENCE_SECONDS_PER_FOUR_COMPONENT_TRIAL = 6.089 / 3.0
VENUE_ORDER = (
    "home",
    "away",
    "home",
    "away",
    "neutral",
    "home",
    "away",
    "home",
    "away",
    "neutral",
)


def _expected_rank(pmf: np.ndarray) -> float:
    return float(np.dot(np.arange(1, len(pmf) + 1, dtype=float), pmf))


def _rank_quantile(pmf: np.ndarray, probability: float) -> int:
    return int(np.searchsorted(np.cumsum(pmf), probability, side="left") + 1)


def _rank_summary(pmf: np.ndarray) -> dict[str, float | int]:
    ranks = np.arange(1, len(pmf) + 1, dtype=float)
    mean = float(np.dot(ranks, pmf))
    sd = float(np.sqrt(np.dot((ranks - mean) ** 2, pmf)))
    return {
        "expected_rank": mean,
        "rank_sd": sd,
        "median_rank": _rank_quantile(pmf, 0.5),
        "interval_80_low": _rank_quantile(pmf, 0.1),
        "interval_80_high": _rank_quantile(pmf, 0.9),
        "interval_80_width": _rank_quantile(pmf, 0.9) - _rank_quantile(pmf, 0.1) + 1,
    }


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    half = (
        z
        * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials**2))
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_gzip_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty gzip CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    with (
        path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed,
    ):
        compressed.write(buffer.getvalue().encode("utf-8"))


def _select_profile(teams: list[Team], target: float) -> Team:
    fbs = [team for team in teams if team.subdivision == "fbs"]
    return min(
        fbs,
        key=lambda team: (
            abs(_expected_rank(team.prior) - target),
            team.team_id,
        ),
    )


def _target_schedule(
    focal: Team,
    candidates: list[Team],
    probabilities: np.ndarray,
    desired_mean: float,
) -> tuple[float, tuple[tuple[Team, str, float, float], ...]]:
    """Match a common-width game-probability profile to unique FBS opponents."""
    offsets = np.linspace(
        -SCHEDULE_PROFILE_HALF_WIDTH, SCHEDULE_PROFILE_HALF_WIDTH, len(SITE_SLOTS)
    )
    candidate_indexes = np.arange(len(candidates), dtype=float)
    options: list[
        tuple[
            tuple[float, float, float, tuple[str, ...]],
            float,
            tuple[tuple[Team, str, float, float], ...],
        ]
    ] = []
    for center_hundredth in range(1, 100):
        profile_center = center_hundredth / 100.0
        targets = np.clip(profile_center + offsets, 0.01, 0.99)
        costs = (
            np.abs(probabilities - targets[:, None])
            + candidate_indexes[None, :] * 1e-12
        )
        rows, columns = linear_sum_assignment(costs)
        ordered = sorted(zip(rows.tolist(), columns.tolist(), strict=True))
        chosen = tuple(
            (
                candidates[column],
                SITE_SLOTS[row],
                float(probabilities[row, column]),
                float(targets[row]),
            )
            for row, column in ordered
        )
        actual_mean = float(np.mean([slot[2] for slot in chosen]))
        identifiers = tuple(slot[0].team_id for slot in chosen)
        score = (
            abs(actual_mean - desired_mean),
            abs(profile_center - desired_mean),
            profile_center,
            identifiers,
        )
        options.append((score, profile_center, chosen))
    _, center, schedule = min(options, key=lambda option: option[0])
    return center, schedule


def _feasible_extreme_schedule_means(probabilities: np.ndarray) -> tuple[float, float]:
    minimum_rows, minimum_columns = linear_sum_assignment(probabilities)
    maximum_rows, maximum_columns = linear_sum_assignment(-probabilities)
    return (
        float(np.mean(probabilities[minimum_rows, minimum_columns])),
        float(np.mean(probabilities[maximum_rows, maximum_columns])),
    )


def _sample_rank(pmf: np.ndarray, uniform: float) -> int:
    return min(
        int(np.searchsorted(np.cumsum(pmf), uniform, side="right")) + 1, len(pmf)
    )


def _draw_wins(
    schedule: tuple[tuple[Team, str, float, float], ...],
    focal_rank: int,
    opponent_uniforms: np.ndarray,
    margin_uniforms: np.ndarray,
    cumulative_priors: dict[str, np.ndarray],
    conditional_win_probabilities: dict[str, np.ndarray],
) -> int:
    count = 0
    for index, (opponent, venue, _, _) in enumerate(schedule):
        opponent_rank = min(
            int(
                np.searchsorted(
                    cumulative_priors[opponent.team_id],
                    float(opponent_uniforms[index]),
                    side="right",
                )
            )
            + 1,
            len(cumulative_priors[opponent.team_id]),
        )
        if venue == "away":
            probability = conditional_win_probabilities[venue][
                opponent_rank - 1, focal_rank - 1
            ]
        else:
            probability = conditional_win_probabilities[venue][
                focal_rank - 1, opponent_rank - 1
            ]
        count += int(margin_uniforms[index] < probability)
    return count


def _pilot_acceptance(
    *,
    focal: Team,
    lower_schedule: tuple[tuple[Team, str, float, float], ...],
    higher_schedule: tuple[tuple[Team, str, float, float], ...],
    lower_record: int,
    higher_record: int,
    cumulative_priors: dict[str, np.ndarray],
    conditional_win_probabilities: dict[str, np.ndarray],
    seed_start: int,
    draws: int,
) -> list[dict[str, object]]:
    matches = {"primary_pair": 0, "lower_schedule_same_record_control": 0}
    win_totals = {
        "lower_schedule": 0,
        "higher_schedule": 0,
        "control_a": 0,
        "control_b": 0,
    }
    for seed in range(seed_start, seed_start + draws):
        rng = np.random.default_rng(seed)
        focal_rank = min(
            int(
                np.searchsorted(
                    cumulative_priors[focal.team_id], float(rng.random()), side="right"
                )
            )
            + 1,
            len(cumulative_priors[focal.team_id]),
        )
        opponent_uniforms = rng.random(len(SITE_SLOTS))
        primary_margin_uniforms = rng.random(len(SITE_SLOTS))
        lower_wins = _draw_wins(
            lower_schedule,
            focal_rank,
            opponent_uniforms,
            primary_margin_uniforms,
            cumulative_priors,
            conditional_win_probabilities,
        )
        higher_wins = _draw_wins(
            higher_schedule,
            focal_rank,
            opponent_uniforms,
            primary_margin_uniforms,
            cumulative_priors,
            conditional_win_probabilities,
        )
        control_margin_a = rng.random(len(SITE_SLOTS))
        control_margin_b = rng.random(len(SITE_SLOTS))
        control_a_wins = _draw_wins(
            lower_schedule,
            focal_rank,
            opponent_uniforms,
            control_margin_a,
            cumulative_priors,
            conditional_win_probabilities,
        )
        control_b_wins = _draw_wins(
            lower_schedule,
            focal_rank,
            opponent_uniforms,
            control_margin_b,
            cumulative_priors,
            conditional_win_probabilities,
        )
        win_totals["lower_schedule"] += lower_wins
        win_totals["higher_schedule"] += higher_wins
        win_totals["control_a"] += control_a_wins
        win_totals["control_b"] += control_b_wins
        if lower_wins == lower_record and higher_wins == higher_record:
            matches["primary_pair"] += 1
        if control_a_wins == lower_record and control_b_wins == lower_record:
            matches["lower_schedule_same_record_control"] += 1

    rows: list[dict[str, object]] = []
    for panel, successes in matches.items():
        lower, upper = _wilson(successes, draws)
        conservative_rate = lower
        required_draws = CONDITIONED_SEARCH_LIMIT
        if conservative_rate > 0:
            low, high = CONDITIONED_DRAWS, CONDITIONED_SEARCH_LIMIT
            while low < high:
                middle = (low + high) // 2
                if (
                    float(binom.sf(CONDITIONED_DRAWS - 1, middle, conservative_rate))
                    >= 0.95
                ):
                    high = middle
                else:
                    low = middle + 1
            required_draws = low
        rows.append(
            {
                "panel": panel,
                "seed_start": seed_start,
                "pilot_draws": draws,
                "accepted_draws": successes,
                "acceptance_rate": successes / draws,
                "acceptance_rate_wilson_95_low": lower,
                "acceptance_rate_wilson_95_high": upper,
                "target_accepted_draws": CONDITIONED_DRAWS,
                "planned_search_draws_for_95pct_completion_at_lower_rate": required_draws,
                "search_ceiling": CONDITIONED_SEARCH_LIMIT,
                "expected_draw_count_exceeds_ceiling": (
                    conservative_rate == 0
                    or float(
                        binom.sf(
                            CONDITIONED_DRAWS - 1,
                            CONDITIONED_SEARCH_LIMIT,
                            conservative_rate,
                        )
                    )
                    < 0.95
                ),
                "expected_lower_schedule_wins": sum(slot[2] for slot in lower_schedule),
                "mean_lower_schedule_wins_in_pilot": win_totals["lower_schedule"]
                / draws,
                "expected_higher_schedule_wins": sum(
                    slot[2] for slot in higher_schedule
                ),
                "mean_higher_schedule_wins_in_pilot": win_totals["higher_schedule"]
                / draws,
                "expected_same_schedule_control_wins": sum(
                    slot[2] for slot in lower_schedule
                ),
                "mean_control_a_wins_in_pilot": win_totals["control_a"] / draws,
                "mean_control_b_wins_in_pilot": win_totals["control_b"] / draws,
            }
        )
    return rows


def _table(
    rows: list[dict[str, object]],
    columns: tuple[tuple[str, str], ...],
    precision: int = 3,
) -> str:
    def formatted(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        return str(value)

    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    lines.extend(
        "| " + " | ".join(formatted(row[key]) for _, key in columns) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _render_report(
    *,
    profile_rows: list[dict[str, object]],
    feasibility_rows: list[dict[str, object]],
    schedule_rows: list[dict[str, object]],
    schedule_summary_rows: list[dict[str, object]],
    pilot_rows: list[dict[str, object]],
) -> str:
    primary_pilot = [row for row in pilot_rows if row["panel"] == "primary_pair"]
    control_pilot = [
        row
        for row in pilot_rows
        if row["panel"] == "lower_schedule_same_record_control"
    ]
    game_slates: list[str] = []
    for profile_row in profile_rows:
        profile = str(profile_row["profile"])
        lower_games = sorted(
            [
                row
                for row in schedule_rows
                if row["profile"] == profile
                and row["schedule_arm"] == "lower_win_probability"
            ],
            key=lambda row: int(row["game_number"]),
        )
        higher_games = sorted(
            [
                row
                for row in schedule_rows
                if row["profile"] == profile
                and row["schedule_arm"] == "higher_win_probability"
            ],
            key=lambda row: int(row["game_number"]),
        )
        game_rows = [
            {
                "game": lower["game_number"],
                "venue": lower["venue"],
                "lower_opponent": lower["opponent_name"],
                "lower_probability": float(lower["v1_expected_focal_win_probability"]),
                "higher_opponent": higher["opponent_name"],
                "higher_probability": float(
                    higher["v1_expected_focal_win_probability"]
                ),
            }
            for lower, higher in zip(lower_games, higher_games, strict=True)
        ]
        game_slates.extend(
            [
                f"### {profile}: {profile_row['team_name']} (ID {profile_row['team_id']})",
                "",
                _table(
                    game_rows,
                    (
                        ("Game", "game"),
                        ("Venue", "venue"),
                        ("Lower schedule opponent", "lower_opponent"),
                        ("p(win)", "lower_probability"),
                        ("Higher schedule opponent", "higher_opponent"),
                        ("p(win)", "higher_probability"),
                    ),
                ),
                "",
            ]
        )
    total_seconds = (
        len(STRENGTH_TARGETS)
        * (UNCONDITIONAL_DRAWS + CONDITIONED_DRAWS)
        * INFERENCE_SECONDS_PER_FOUR_COMPONENT_TRIAL
        + len(STRENGTH_TARGETS)
        * CONDITIONED_DRAWS
        * INFERENCE_SECONDS_PER_FOUR_COMPONENT_TRIAL
        / 2
    )
    minutes = total_seconds / 60.0

    return "\n".join(
        [
            "# Issue 258: Multi-strength Quality of Results design",
            "",
            "## Decision and scope",
            "",
            "This is a research-only feasibility and design artifact. It selects four deterministic focal priors, estimates feasible FBS opponent and schedule win-probability ranges, fixes schedule pairs, and runs a generation-only record-acceptance pilot. The published issue 252 report is used only for sample-size and compute planning. This ticket runs no candidate posterior inference, generates no issue 258 ranking results, selects no margin weight, and changes no ranking behavior.",
            "",
            "The four main profiles are selected by nearest 2026 Context 1.4 preseason expected rank to 5, 20, 65, and 120, with canonical team ID breaking ties. The complete prior PMFs are retained. Colorado is included as a reference to the issue 252 focal prior; it is not an additional stratum in the main follow-up budget.",
            "",
            "## Frozen profiles and feasibility",
            "",
            _table(
                profile_rows,
                (
                    ("Profile", "profile"),
                    ("Selected team", "team_name"),
                    ("ID", "team_id"),
                    ("Target rank", "selection_target_expected_rank"),
                    ("Expected rank", "expected_rank"),
                    ("Rank SD", "rank_sd"),
                    ("80% interval", "interval_80"),
                ),
            ),
            "",
            "The game-level range is the minimum and maximum integrated focal win probability over every other FBS opponent and the fixed home, away, or neutral venue. Schedule-average limits use a one-to-one assignment of ten distinct FBS opponents to the ten venue slots. The 50% and 80% feasibility flags below refer to those achievable average limits. A feasible average does not guarantee a schedule with a balanced game-probability distribution; both the selected slates and their full ten-game distributions are therefore retained.",
            "",
            _table(
                feasibility_rows,
                (
                    ("Profile", "profile"),
                    ("Game p range", "all_game_probability_range"),
                    ("Home", "home_probability_range"),
                    ("Away", "away_probability_range"),
                    ("Neutral", "neutral_probability_range"),
                    ("Feasible schedule means", "schedule_mean_range"),
                    ("50%", "target_50_mean_feasible"),
                    ("80%", "target_80_mean_feasible"),
                ),
            ),
            "",
            "## Frozen schedule pairs",
            "",
            "Every schedule has ten distinct FBS opponents and the issue 252 venue sequence (four home, four away, two neutral). Win probabilities integrate the focal and opponent rank PMFs through Historical Likelihood V1. The schedule builder matches a common ±0.08 target-probability profile to the closest unique opponents by venue; target-profile centers are calibrated using only these frozen model inputs and the requested schedule mean. Full game-level probabilities and opponent prior PMFs are in `schedule_games.csv` and `opponent_prior_pmfs.csv.gz`.",
            "",
            _table(
                schedule_summary_rows,
                (
                    ("Profile", "profile"),
                    ("Arm", "schedule_arm"),
                    ("Requested mean", "requested_schedule_mean"),
                    ("Achieved mean", "mean_win_probability"),
                    ("SD", "sd_win_probability"),
                    ("Game p range", "game_probability_range"),
                    ("Median", "median_win_probability"),
                ),
            ),
            "",
            "Achieved schedule probability contrasts are the difference between the two arm means. The follow-up must report each stratum's actual contrast alongside its ranking effects and consider the contrast size when comparing effects across strata.",
            "",
            _table(
                [
                    {
                        "profile": row["profile"],
                        "lower_schedule_mean": row["achieved_lower_mean"],
                        "higher_schedule_mean": row["achieved_higher_mean"],
                        "achieved_probability_separation": row[
                            "achieved_mean_separation"
                        ],
                    }
                    for row in feasibility_rows
                ],
                (
                    ("Profile", "profile"),
                    ("Lower schedule mean", "lower_schedule_mean"),
                    ("Higher schedule mean", "higher_schedule_mean"),
                    ("Achieved separation", "achieved_probability_separation"),
                ),
            ),
            "",
            "### Game-level schedule pairs",
            "",
            *game_slates,
            "The original 50% versus 80% average contrast is achievable for the strong profile, and the mean limits also bracket both targets for the middle profile. The balanced target-profile builder reaches approximately 50% and 80% for the strong profile. For the middle profile it reaches the 80% mean only with a target-profile center near the upper end of the FBS opponent range, so its final achieved mean is used and documented rather than substituting expected ranks or forcing a high-variance schedule. Elite teams cannot average 50% across ten FBS games; their proposed comparison is approximately 60% versus 80%. Weak teams cannot reach either 50% or 80%; their proposed comparison is approximately 20% versus 40%. The narrower alternatives preserve a meaningful separation while keeping each slate's game probabilities clustered, rather than assembling near-certain wins and near-certain losses to hit an average.",
            "",
            "## Record-conditioned panels and generation-only pilot",
            "",
            "Record targets follow each pair's achieved schedule means: elite 6–4 versus 8–2; strong 5–5 versus 8–2; middle 5–5 versus 8–2; weak 2–8 versus 4–6. The pilot draws one focal rank per seed, samples opponent ranks from their full independent priors, and samples win/loss using the exact conditional V1 Student-t probability at the sampled rank pair and venue. The follow-up draws continuous synthetic margins from that same Student-t distribution; margin signs determine wins and losses. Shared focal ranks, opponent rank quantiles, and margin quantiles match the two schedule arms. Two copies of the lower-win-probability schedule use independent margin draws for the same-schedule control. No margins are altered to obtain a record.",
            "",
            _table(
                primary_pilot,
                (
                    ("Profile", "profile"),
                    ("Accepted / pilot", "accepted_draws"),
                    ("Rate", "acceptance_rate"),
                    ("Wilson 95%", "acceptance_rate_interval"),
                    ("Mean wins L/H", "pilot_mean_wins_pair"),
                    (
                        "Planned search",
                        "planned_search_draws_for_95pct_completion_at_lower_rate",
                    ),
                ),
            ),
            "",
            "The separate same-schedule record-conditioned control panel accepts seeds only when both duplicated control teams attain the lower schedule's target record. It is not selected from the primary schedule-pair seeds.",
            "",
            _table(
                control_pilot,
                (
                    ("Profile", "profile"),
                    ("Accepted / pilot", "accepted_draws"),
                    ("Rate", "acceptance_rate"),
                    ("Wilson 95%", "acceptance_rate_interval"),
                    ("Mean wins A/B", "pilot_mean_wins_control"),
                    (
                        "Planned search",
                        "planned_search_draws_for_95pct_completion_at_lower_rate",
                    ),
                ),
            ),
            "",
            "The next ticket should use 500 unconditional paired seasons, 200 accepted primary record-conditioned pairs, and 200 accepted lower-schedule same-schedule control pairs per strength stratum. Each candidate λ receives identical game evidence within a trial. The primary conditional and control conditional panels use separate, disjoint seed ranges. The unconditional control is observed at every unconditional seed; controls copied onto primary-conditioned seeds are labeled as primary-seed controls and are not used as own-record-conditioned controls. The generation pilot also checks that simulated mean win counts track the schedule probabilities before posterior inference.",
            "",
            "## Follow-up analysis protocol",
            "",
            "- Reuse issue 252's disconnected inference forest, issue 242 Quality of Results implementation, complete 138-rank support, default deterministic BP settings, focal prior stripping, and five frozen candidates (λ = 0, 0.25, 0.50, 0.75, 1). Keep each opponent prior uncertain in inference; do not plug in expected ranks or treat opponents as known.",
            "- Generate identical trial evidence for all five candidates. The two primary teams in a pair share the same underlying focal PMF, focal latent-rank draw, starting-prior treatment, schedule game count, and venue distribution. Opponents are independently sampled from their own Context 1.4 PMFs; no opponent repeats within one slate.",
            "- Report results separately by strength stratum. Alongside each stratum's ranking effects, report its actual achieved schedule win-probability separation (the higher-arm mean minus the lower-arm mean). The primary schedule effect is the paired difference in expected rank (higher-win-probability arm minus lower-win-probability arm), with a 95% normal Monte Carlo interval for the mean, the difference distribution, outrank probability with a 95% Wilson interval, rank SD, entropy, central 80% width, PMF total variation, and Wasserstein distance.",
            "- When comparing schedule effects across strata, explicitly consider differences in their achieved schedule separations. Do not conclude that the methodology is more or less schedule-sensitive at a strength level solely because its absolute ranking differences are larger or smaller. Do not compare absolute expected ranks across strata as if those were schedule effects, and do not introduce a new normalization formula.",
            "- Use same-schedule controls to estimate outcome-driven variation at the lower schedule distribution. Report them beside, not subtracted from, the primary schedule contrast. Keep unconditional controls unconditional. Build a separate control-conditioned panel accepted on the two controls' own records; never label controls selected only by a primary pair's seed as record-conditioned controls.",
            "- Separate schedule-driven differences (paired schedule contrast), realized-outcome variation (within-panel difference distributions and same-schedule controls), posterior uncertainty (paired rank SD, entropy, interval width), and between-stratum context (within-stratum effects and their contrasts). Do not select or revise schedules after seeing candidate posterior ranks.",
            "",
            "## Sample and compute budget",
            "",
            f"The unconditional and primary conditioned panels require {UNCONDITIONAL_DRAWS + CONDITIONED_DRAWS} four-component inference trials per stratum; the separately conditioned control panel adds {CONDITIONED_DRAWS} two-component trials per stratum. Scaling issue 252's measured 6.089 seconds for three five-weight, four-component trials gives an estimated **{minutes:.0f} minutes** total for the four strata, including the separate control panel. This is a planning estimate, not a new timing run. Generation and seed search are expected to be small relative to posterior inference. The pilot estimates a per-panel seed search that targets a 95% chance of finding 200 accepted draws using the lower Wilson acceptance bound, capped at {CONDITIONED_SEARCH_LIMIT:,} attempts. If a panel exceeds that cap, pause before inference and report the conditioning limitation; do not change records in response to candidate results.",
            "",
            "## Reproducibility and frozen decisions",
            "",
            "The manifest pins the source prior CSV, Historical Likelihood V1 bytes, issue 252 schedule protocol, this design code, selected focal PMFs, opponent PMFs, schedule definitions, pilot seed ranges, NumPy/SciPy versions, and SHA-256 hashes for every output. Gzip output uses a fixed timestamp. All profile selection, schedule construction, pilot criteria, outcome targets, and follow-up budgets are fixed without issue 258 candidate posterior inference. If a comparison cannot be run within the stated FBS population and budgets, the follow-up should report that limitation rather than introduce FCS opponents, duplicate opponents, schedule bonuses, likelihood changes, or invented profiles.",
            "",
            f"Pinned source revision: `{PINNED_SOURCE_REVISION}`.",
            "",
        ]
    )


def run_design(root: Path, output_dir: Path | None = None) -> dict[str, object]:
    root = root.resolve()
    output_dir = output_dir or root / "data/research/issue_258_quality_of_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    teams, _team_rows, prior_path = load_teams(root, SEASON, "context", PRIOR_VERSION)
    fbs = [team for team in teams if team.subdivision == "fbs"]
    if not fbs:
        raise ValueError("frozen 2026 Context 1.4 priors contain no FBS teams")
    rank_supports = {len(team.prior) for team in fbs}
    if len(rank_supports) != 1:
        raise ValueError("frozen FBS priors do not share one rank support")
    rank_support = rank_supports.pop()
    likelihood_path = root / "data/processed/posterior/historical_likelihood_v1.json"
    likelihood = load_pinned_likelihood(likelihood_path)
    if sha256_file(likelihood_path) != HISTORICAL_LIKELIHOOD_SHA256:
        raise ValueError(
            "Historical Likelihood V1 bytes differ from the pinned artifact"
        )
    surfaces = _site_location_matrices(rank_support, likelihood)
    conditional_win_probabilities = {
        venue: (
            student_t.cdf(-surface / likelihood.scale, likelihood.degrees_of_freedom)
            if venue == "away"
            else student_t.sf(
                -surface / likelihood.scale, likelihood.degrees_of_freedom
            )
        )
        for venue, surface in surfaces.items()
    }
    targets = list(STRENGTH_TARGETS)
    profile_targets: list[tuple[str, float, Team]] = [
        (name, target, _select_profile(teams, target)) for name, target in targets
    ]
    reference = next((team for team in fbs if team.team_id == REFERENCE_TEAM_ID), None)
    if reference is None:
        raise ValueError(f"issue 252 reference team ID {REFERENCE_TEAM_ID} is absent")
    profile_targets.append(
        ("issue_252_reference", _expected_rank(reference.prior), reference)
    )

    profile_rows: list[dict[str, object]] = []
    focal_pmf_rows: list[dict[str, object]] = []
    feasibility_rows: list[dict[str, object]] = []
    schedule_rows: list[dict[str, object]] = []
    schedule_summary_rows: list[dict[str, object]] = []
    opponent_prior_rows: dict[tuple[str, str], dict[str, object]] = {}
    pilot_rows: list[dict[str, object]] = []
    profile_schedules: dict[
        str,
        tuple[
            Team,
            tuple[tuple[Team, str, float, float], ...],
            tuple[tuple[Team, str, float, float], ...],
        ],
    ] = {}
    teams_by_id = {team.team_id: team for team in fbs}
    cumulative_priors = {team.team_id: np.cumsum(team.prior) for team in fbs}

    for profile_index, (profile, selection_target, focal) in enumerate(profile_targets):
        summary = _rank_summary(focal.prior)
        profile_rows.append(
            {
                "profile": profile,
                "selection_target_expected_rank": selection_target,
                "team_id": focal.team_id,
                "team_name": focal.name,
                "expected_rank": summary["expected_rank"],
                "rank_sd": summary["rank_sd"],
                "median_rank": summary["median_rank"],
                "interval_80_low": summary["interval_80_low"],
                "interval_80_high": summary["interval_80_high"],
                "interval_80": f"{summary['interval_80_low']}–{summary['interval_80_high']}",
                "selection_method": (
                    "nearest preseason expected rank to target; ties by canonical team ID"
                    if profile != "issue_252_reference"
                    else "issue 252 manifest focal team; retained as a design reference"
                ),
                "source_artifact": str(prior_path.relative_to(root)),
                "prior_model_version": PRIOR_VERSION,
            }
        )
        for rank, probability in enumerate(focal.prior, start=1):
            focal_pmf_rows.append(
                {
                    "profile": profile,
                    "team_id": focal.team_id,
                    "team_name": focal.name,
                    "rank": rank,
                    "probability": float(probability),
                }
            )

        candidates = sorted(
            (team for team in fbs if team.team_id != focal.team_id),
            key=lambda team: team.team_id,
        )
        probabilities = np.array(
            [
                [
                    expected_focal_win_probability(
                        focal.prior, opponent.prior, venue, likelihood, surfaces
                    )
                    for opponent in candidates
                ]
                for venue in VENUE_ORDER
            ],
            dtype=float,
        )
        minimum_schedule_mean, maximum_schedule_mean = _feasible_extreme_schedule_means(
            probabilities
        )
        venue_ranges: dict[str, tuple[float, float]] = {}
        for venue_index, venue in enumerate(("home", "away", "neutral")):
            values = probabilities[
                [i for i, item in enumerate(VENUE_ORDER) if item == venue], :
            ].ravel()
            venue_ranges[venue] = (float(values.min()), float(values.max()))
        all_probs = probabilities.ravel()
        desired_lower, desired_higher = SCHEDULE_TARGET_MEANS[profile]
        lower_center, lower_schedule = _target_schedule(
            focal, candidates, probabilities, desired_lower
        )
        higher_center, higher_schedule = _target_schedule(
            focal, candidates, probabilities, desired_higher
        )
        profile_schedules[profile] = (focal, lower_schedule, higher_schedule)

        feasibility_rows.append(
            {
                "profile": profile,
                "team_id": focal.team_id,
                "all_game_probability_min": float(all_probs.min()),
                "all_game_probability_max": float(all_probs.max()),
                "all_game_probability_range": f"{all_probs.min():.3f}–{all_probs.max():.3f}",
                "home_probability_min": venue_ranges["home"][0],
                "home_probability_max": venue_ranges["home"][1],
                "home_probability_range": f"{venue_ranges['home'][0]:.3f}–{venue_ranges['home'][1]:.3f}",
                "away_probability_min": venue_ranges["away"][0],
                "away_probability_max": venue_ranges["away"][1],
                "away_probability_range": f"{venue_ranges['away'][0]:.3f}–{venue_ranges['away'][1]:.3f}",
                "neutral_probability_min": venue_ranges["neutral"][0],
                "neutral_probability_max": venue_ranges["neutral"][1],
                "neutral_probability_range": f"{venue_ranges['neutral'][0]:.3f}–{venue_ranges['neutral'][1]:.3f}",
                "minimum_schedule_mean_probability": minimum_schedule_mean,
                "maximum_schedule_mean_probability": maximum_schedule_mean,
                "schedule_mean_range": f"{minimum_schedule_mean:.3f}–{maximum_schedule_mean:.3f}",
                "target_50_mean_feasible": minimum_schedule_mean
                <= 0.50
                <= maximum_schedule_mean,
                "target_80_mean_feasible": minimum_schedule_mean
                <= 0.80
                <= maximum_schedule_mean,
                "desired_lower_mean": desired_lower,
                "lower_profile_center": lower_center,
                "achieved_lower_mean": float(
                    np.mean([slot[2] for slot in lower_schedule])
                ),
                "desired_higher_mean": desired_higher,
                "higher_profile_center": higher_center,
                "achieved_higher_mean": float(
                    np.mean([slot[2] for slot in higher_schedule])
                ),
                "achieved_mean_separation": float(
                    np.mean([slot[2] for slot in higher_schedule])
                    - np.mean([slot[2] for slot in lower_schedule])
                ),
            }
        )

        for arm, requested_mean, center, schedule in (
            ("lower_win_probability", desired_lower, lower_center, lower_schedule),
            ("higher_win_probability", desired_higher, higher_center, higher_schedule),
        ):
            game_probabilities = [slot[2] for slot in schedule]
            schedule_summary_rows.append(
                {
                    "profile": profile,
                    "team_id": focal.team_id,
                    "team_name": focal.name,
                    "schedule_arm": arm,
                    "requested_schedule_mean": requested_mean,
                    "target_profile_center": center,
                    "target_profile_half_width": SCHEDULE_PROFILE_HALF_WIDTH,
                    "mean_win_probability": float(np.mean(game_probabilities)),
                    "sd_win_probability": float(np.std(game_probabilities)),
                    "game_probability_min": min(game_probabilities),
                    "game_probability_max": max(game_probabilities),
                    "game_probability_range": f"{min(game_probabilities):.3f}–{max(game_probabilities):.3f}",
                    "median_win_probability": float(np.median(game_probabilities)),
                    "probability_p10": float(np.quantile(game_probabilities, 0.1)),
                    "probability_p90": float(np.quantile(game_probabilities, 0.9)),
                    "probability_vector": ";".join(
                        f"{value:.6f}" for value in game_probabilities
                    ),
                }
            )
            for game_number, (
                opponent,
                venue,
                probability,
                target_probability,
            ) in enumerate(schedule, start=1):
                schedule_rows.append(
                    {
                        "profile": profile,
                        "focal_team_id": focal.team_id,
                        "focal_team_name": focal.name,
                        "schedule_arm": arm,
                        "game_number": game_number,
                        "venue": venue,
                        "opponent_team_id": opponent.team_id,
                        "opponent_name": opponent.name,
                        "opponent_subdivision": opponent.subdivision,
                        "target_profile_center": center,
                        "target_probability": target_probability,
                        "v1_expected_focal_win_probability": probability,
                        "opponent_prior_source": str(prior_path.relative_to(root)),
                        "opponent_prior_model_version": PRIOR_VERSION,
                    }
                )
                opponent_prior_rows[(profile, opponent.team_id)] = {
                    "profile": profile,
                    "team_id": opponent.team_id,
                    "team_name": opponent.name,
                    "rank": 0,
                    "probability": 0.0,
                }

        lower_record = round(
            np.mean([slot[2] for slot in lower_schedule]) * len(SITE_SLOTS)
        )
        higher_record = round(
            np.mean([slot[2] for slot in higher_schedule]) * len(SITE_SLOTS)
        )
        pilot_seed = PILOT_SEED_START + profile_index * PILOT_DRAWS
        primary_seed = (
            PRIMARY_CONDITIONED_SEED_START + profile_index * CONDITIONED_SEARCH_LIMIT
        )
        control_seed = (
            CONTROL_CONDITIONED_SEED_START + profile_index * CONDITIONED_SEARCH_LIMIT
        )
        pilot_rows.extend(
            _pilot_acceptance(
                focal=focal,
                lower_schedule=lower_schedule,
                higher_schedule=higher_schedule,
                lower_record=lower_record,
                higher_record=higher_record,
                cumulative_priors=cumulative_priors,
                conditional_win_probabilities=conditional_win_probabilities,
                seed_start=pilot_seed,
                draws=PILOT_DRAWS,
            )
        )
        for pilot_row in pilot_rows[-2:]:
            pilot_row.update(
                {
                    "profile": profile,
                    "focal_team_id": focal.team_id,
                    "focal_team_name": focal.name,
                    "lower_schedule_target_wins": lower_record,
                    "higher_schedule_target_wins": higher_record,
                    "games_per_schedule": len(SITE_SLOTS),
                    "planned_primary_conditioned_seed_start": primary_seed,
                    "planned_control_conditioned_seed_start": control_seed,
                    "pilot_mean_wins_pair": (
                        f"{float(pilot_row['mean_lower_schedule_wins_in_pilot']):.2f}/"
                        f"{float(pilot_row['mean_higher_schedule_wins_in_pilot']):.2f}"
                    ),
                    "pilot_mean_wins_control": (
                        f"{float(pilot_row['mean_control_a_wins_in_pilot']):.2f}/"
                        f"{float(pilot_row['mean_control_b_wins_in_pilot']):.2f}"
                    ),
                }
            )

    # Save complete opponent rank PMFs once per profile/team pair.
    full_opponent_rows: list[dict[str, object]] = []
    for profile, team_id in sorted(opponent_prior_rows):
        opponent = teams_by_id[team_id]
        for rank, probability in enumerate(opponent.prior, start=1):
            full_opponent_rows.append(
                {
                    "profile": profile,
                    "team_id": team_id,
                    "team_name": opponent.name,
                    "rank": rank,
                    "probability": float(probability),
                }
            )

    prior_copy = output_dir / "historical_likelihood_v1.json"
    prior_copy.write_bytes(likelihood_path.read_bytes())
    focal_pmf_path = output_dir / "focal_prior_pmfs.csv.gz"
    opponent_pmf_path = output_dir / "opponent_prior_pmfs.csv.gz"
    _write_gzip_csv(focal_pmf_path, focal_pmf_rows)
    _write_gzip_csv(opponent_pmf_path, full_opponent_rows)
    _write_csv(output_dir / "profile_summary.csv", profile_rows)
    _write_csv(output_dir / "feasibility.csv", feasibility_rows)
    _write_csv(output_dir / "schedule_games.csv", schedule_rows)
    _write_csv(output_dir / "schedule_summary.csv", schedule_summary_rows)
    _write_csv(output_dir / "generation_pilot.csv", pilot_rows)

    # Add human-readable confidence intervals for report rendering.
    for row in pilot_rows:
        row["acceptance_rate_interval"] = (
            f"{float(row['acceptance_rate_wilson_95_low']):.3%}–"
            f"{float(row['acceptance_rate_wilson_95_high']):.3%}"
        )
    report = _render_report(
        profile_rows=profile_rows,
        feasibility_rows=feasibility_rows,
        schedule_rows=schedule_rows,
        schedule_summary_rows=schedule_summary_rows,
        pilot_rows=pilot_rows,
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    (root / "docs/issue_258_quality_of_results_design.md").write_text(
        report, encoding="utf-8"
    )

    artifact_names = (
        "focal_prior_pmfs.csv.gz",
        "opponent_prior_pmfs.csv.gz",
        "profile_summary.csv",
        "feasibility.csv",
        "schedule_games.csv",
        "schedule_summary.csv",
        "generation_pilot.csv",
        "historical_likelihood_v1.json",
        "report.md",
    )
    manifest = {
        "schema_version": "1.0",
        "issue": 258,
        "study": "research_only_multistrength_quality_of_results_schedule_design",
        "base_source_revision": PINNED_SOURCE_REVISION,
        "design_code_sha256": sha256_file(Path(__file__)),
        "issue_252_protocol_sha256": sha256_file(
            root / "docs/issue_252_quality_of_results_protocol.md"
        ),
        "inputs": {
            "season": SEASON,
            "prior_family": "context",
            "prior_model_version": PRIOR_VERSION,
            "prior_path": str(prior_path.relative_to(root)),
            "prior_sha256": sha256_file(prior_path),
            "fbs_team_count": len(fbs),
            "rank_support": rank_support,
            "historical_likelihood_path": str(likelihood_path.relative_to(root)),
            "historical_likelihood_version": "V1",
            "historical_likelihood_sha256": sha256_file(likelihood_path),
        },
        "profile_selection": {
            "method": "nearest frozen preseason expected rank to target; ties by canonical team ID",
            "targets": {name: target for name, target in STRENGTH_TARGETS},
            "issue_252_reference_team_id": REFERENCE_TEAM_ID,
            "complete_focal_pmfs_retained": True,
        },
        "schedule_design": {
            "games_per_schedule": len(SITE_SLOTS),
            "venue_slots": list(SITE_SLOTS),
            "venue_counts": {
                venue: SITE_SLOTS.count(venue) for venue in ("home", "away", "neutral")
            },
            "opponent_population": "2026 FBS only",
            "no_repeated_opponent_within_schedule": True,
            "opponent_beliefs": "independent full Context 1.4 rank PMFs",
            "win_probability": "Historical Likelihood V1 Student-t CDF integrated over focal and opponent rank PMFs with venue effects",
            "target_profile_half_width": SCHEDULE_PROFILE_HALF_WIDTH,
            "selection": "minimum absolute V1 probability distance to target profile by venue with unique opponents; target profile center calibrated to requested mean using frozen priors only",
            "requested_means": {
                profile: list(means) for profile, means in SCHEDULE_TARGET_MEANS.items()
            },
            "candidate_quality_of_results_inference_run": False,
            "issue_258_candidate_ranking_outcomes_inspected": False,
        },
        "generation_pilot": {
            "seed_start": PILOT_SEED_START,
            "draws_per_profile": PILOT_DRAWS,
            "candidate_quality_of_results_inference_run": False,
            "matched_draws": "shared focal latent rank and opponent/margin quantiles across schedule pair; independent margin draws across same-schedule controls",
            "record_conditioned_target_acceptances": "primary schedule pair and lower schedule same-schedule control pair are counted separately",
        },
        "follow_up_plan": {
            "margin_weights": [0.0, 0.25, 0.5, 0.75, 1.0],
            "unconditional_pairs_per_profile": UNCONDITIONAL_DRAWS,
            "primary_record_conditioned_pairs_per_profile": CONDITIONED_DRAWS,
            "self_record_conditioned_control_pairs_per_profile": CONDITIONED_DRAWS,
            "conditioning_search_ceiling_per_panel": CONDITIONED_SEARCH_LIMIT,
            "primary_seed_start": PRIMARY_CONDITIONED_SEED_START,
            "control_seed_start": CONTROL_CONDITIONED_SEED_START,
            "unconditional_shared_seed_range": [
                UNCONDITIONAL_SEED_START,
                UNCONDITIONAL_SEED_START + UNCONDITIONAL_DRAWS - 1,
            ],
            "primary_conditioned_seed_ranges": {
                profile: [
                    PRIMARY_CONDITIONED_SEED_START + index * CONDITIONED_SEARCH_LIMIT,
                    PRIMARY_CONDITIONED_SEED_START
                    + (index + 1) * CONDITIONED_SEARCH_LIMIT
                    - 1,
                ]
                for index, (profile, _) in enumerate(STRENGTH_TARGETS)
            },
            "control_conditioned_seed_ranges": {
                profile: [
                    CONTROL_CONDITIONED_SEED_START + index * CONDITIONED_SEARCH_LIMIT,
                    CONTROL_CONDITIONED_SEED_START
                    + (index + 1) * CONDITIONED_SEARCH_LIMIT
                    - 1,
                ]
                for index, (profile, _) in enumerate(STRENGTH_TARGETS)
            },
            "seeds_disjoint_from_pilot": True,
            "all_candidates_share_trial_evidence": True,
            "estimate_uses_issue_252_three_trial_five_weight_timing": INFERENCE_SECONDS_PER_FOUR_COMPONENT_TRIAL,
        },
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "artifact_sha256": {
            name: sha256_file(output_dir / name) for name in artifact_names
        },
        "documentation_path": "docs/issue_258_quality_of_results_design.md",
        "documentation_sha256": sha256_file(
            root / "docs/issue_258_quality_of_results_design.md"
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = run_design(args.root, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
