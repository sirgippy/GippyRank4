"""Run the frozen issue 247 accumulated-surprise analysis."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gippyrank.research.issue_246_error_persistence import (
    BOOTSTRAP_REPETITIONS,
    SEASONS,
    load_issue200_preseason_rank_priors,
    load_issue201_inputs,
    sha256_file,
)
from gippyrank.research.issue_247_contradictory_evidence import (
    CONTRAST_FIELDS,
    EXAMPLE_FIELDS,
    EXAMPLE_MIN_PRIOR_GAMES,
    EXISTING_ARMS,
    GROUP_SUMMARY_FIELDS,
    MIN_PRIOR_GAMES,
    MODEL_FIELDS,
    OBSERVATION_FIELDS,
    SURPRISE_GROUP_THRESHOLD,
    TEAM_GAME_FIELDS,
    analyze_arm,
    analyze_season,
    build_observations,
    build_team_game_errors,
    select_examples,
)

DEFAULT_ISSUE201_ROOT = ROOT / "data/research/issue_201_prebowl_forecast"
DEFAULT_ISSUE200_ROOT = ROOT / "data/research/matched_history_context_issue_200"
DEFAULT_PROTOCOL = ROOT / "docs/issue_247_contradictory_evidence.md"
DEFAULT_OUTPUT = ROOT / "data/research/issue_247_contradictory_evidence"
BOOTSTRAP_SEED = 247


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return ""
    return value


def _write_csv(
    path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _fmt(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.{digits}f}"


def _interval(value: Any, low: Any, high: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "—"
    return f"{_fmt(value, digits)} [{_fmt(low, digits)}, {_fmt(high, digits)}]"


def render_report(
    *,
    parity_checks: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    bootstrap_repetitions: int,
) -> str:
    eligible = [row for row in observations if row["primary_eligible"]]
    grouped_counts: dict[tuple[int, str], dict[str, int]] = {}
    for arm, _, _ in EXISTING_ARMS:
        for season in SEASONS:
            grouped_counts[(season, arm)] = {
                "all": sum(
                    int(row["season"]) == season and row["model_arm"] == arm
                    for row in observations
                ),
                "eligible": sum(
                    int(row["season"]) == season
                    and row["model_arm"] == arm
                    and bool(row["primary_eligible"])
                    for row in observations
                ),
            }
    parity_max = max(
        (float(row["max_metric_absolute_difference"]) for row in parity_checks),
        default=0.0,
    )

    def model(arm: str, outcome: str, term: str) -> dict[str, Any]:
        return next(
            row
            for row in model_rows
            if row["summary_level"] == "pooled_model_arm"
            and row["model_arm"] == arm
            and row["outcome"] == outcome
            and row["term"] == term
        )

    def seasonal(arm: str, season: int, outcome: str) -> dict[str, Any]:
        return next(
            row
            for row in model_rows
            if row["summary_level"] == "season_model_arm"
            and int(row["season"]) == season
            and row["model_arm"] == arm
            and row["outcome"] == outcome
            and row["term"] == "accumulated_surprise"
        )

    finding_lines = []
    for arm, family, _ in EXISTING_ARMS:
        slope = model(arm, "margin_points", "accumulated_surprise")
        if float(slope["ci95_low"]) > 0.0 or float(slope["ci95_high"]) < 0.0:
            interval_result = "its 95% interval excludes zero"
        else:
            interval_result = "its 95% interval includes zero"
        finding_lines.append(
            f"{family}'s adjusted pooled margin-error slope was "
            f"{_interval(slope['estimate'], slope['ci95_low'], slope['ci95_high'])} "
            f"points per surprise unit; {interval_result}."
        )
    pooled_slope_rows = [
        model(arm, "margin_points", "accumulated_surprise")
        for arm, _, _ in EXISTING_ARMS
    ]
    if all(
        float(row["ci95_low"]) <= 0.0 <= float(row["ci95_high"])
        for row in pooled_slope_rows
    ):
        finding_lines.append(
            "Neither prior family shows a detectable adjusted pooled association between accumulated surprise and next-game margin error at this resolution. The point estimates are negative, but their intervals also allow a positive association."
        )
    interaction_rows = [
        model(arm, "margin_points", "surprise_x_preseason_strength_z")
        for arm, _, _ in EXISTING_ARMS
    ]
    if all(
        float(row["ci95_low"]) <= 0.0 <= float(row["ci95_high"])
        for row in interaction_rows
    ):
        finding_lines.append(
            "The continuous preseason-strength interaction intervals include zero in both families, so this panel does not show a clear stronger or weaker surprise/error relationship for stronger preseason expectations."
        )
    all_group_contrasts_include_zero = all(
        float(row["ci95_low"]) <= 0.0 <= float(row["ci95_high"])
        for row in contrast_rows
    )
    if all_group_contrasts_include_zero:
        finding_lines.append(
            "The sustained-negative versus neutral, sustained-positive versus neutral, and sustained-positive versus sustained-negative contrasts all have intervals including zero in both families."
        )

    lines = [
        "# Issue 247: accumulated contradictory evidence and subsequent forecast errors",
        "",
        "## Finding",
        "",
        "This analysis asks whether the mean of a team's evidence-available prior-game surprises predicts the signed error in its next forecast, after those results were incorporated. The 80% interval half-width is the uncertainty scale; signed error divided by that half-width is the standardized surprise. The primary outcome remains signed margin error in points.",
        "",
        f"The eligible panel contains **{len(eligible):,} team-game observations** across **{len({(row['season'], row['team_id']) for row in eligible}):,} team-seasons** in the existing-policy History and Context arms. At least {MIN_PRIOR_GAMES} eligible prior games are required. All {len(observations):,} oriented target team-games, including those below the threshold, are retained in `team_game_observations.csv`.",
        "",
        f"The retained #201 score summary passed parity for **{len(parity_checks)} arm-season checks**; the maximum absolute CRPS or margin-NLL difference was {parity_max:.3g}. Input artifacts are hash-verified by the reused #246 readers. The four seasons remain exploratory historical data, not independent confirmation.",
        "",
        *finding_lines,
        "",
        "## Pooled adjusted association",
        "",
        "OLS adjusts for continuous preseason rank strength and uncertainty, their interaction with accumulated surprise, eligible prior-game count, target team-game number, and season indicators. The surprise slope is the expected change in the next signed margin error (points) per one-unit increase in mean prior surprise, evaluated at mean preseason strength. The secondary outcome divides the next error by its own 80% interval half-width. Intervals are 95% multiway team/target-game bootstrap percentile intervals.",
        "",
        "| Prior family | Observations | Surprise slope, points [95% interval] | Surprise slope, interval-width units [95% interval] | Surprise × preseason strength, points [95% interval] |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm, family, _ in EXISTING_ARMS:
        raw = model(arm, "margin_points", "accumulated_surprise")
        standardized = model(arm, "standardized_80_widths", "accumulated_surprise")
        interaction = model(arm, "margin_points", "surprise_x_preseason_strength_z")
        lines.append(
            f"| {family} | {raw['observations']} | "
            f"{_interval(raw['estimate'], raw['ci95_low'], raw['ci95_high'])} | "
            f"{_interval(standardized['estimate'], standardized['ci95_low'], standardized['ci95_high'])} | "
            f"{_interval(interaction['estimate'], interaction['ci95_low'], interaction['ci95_high'])} |"
        )

    lines.extend(
        [
            "",
            "## Preseason expectation and underperformance",
            "",
            "Preseason strength is the continuous reverse expected-rank percentile from the retained #200 distribution (higher means stronger expectation); uncertainty is rank SD divided by population minus one. The interaction term above tests whether the surprise/error relationship changes with preseason strength.",
            "",
            "The model-implied next-error shift compares mean prior surprise -1 with 0, holding the included covariates fixed at their reference values. Strength is set to the observed 25th and 75th percentiles. Negative shifts mean worse-than-forecast future performance after a one-unit sustained negative history.",
            "",
            "| Prior family | Strength percentile | Shift in next error, points [95% interval] | Shift in standardized next error [95% interval] |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for arm, family, _ in EXISTING_ARMS:
        for percentile in ("p25", "p75"):
            raw = model(
                arm,
                "margin_points",
                f"surprise_minus_one_vs_zero_shift_at_{percentile}",
            )
            standardized = model(
                arm,
                "standardized_80_widths",
                f"surprise_minus_one_vs_zero_shift_at_{percentile}",
            )
            lines.append(
                f"| {family} | {raw[f'preseason_strength_{percentile}']:.3f} | "
                f"{_interval(raw['estimate'], raw['ci95_low'], raw['ci95_high'])} | "
                f"{_interval(standardized['estimate'], standardized['ci95_low'], standardized['ci95_high'])} |"
            )
        interaction = model(
            arm,
            "margin_points",
            "surprise_minus_one_vs_zero_shift_p75_minus_p25",
        )
        lines.append(
            f"| {family} | P75 minus P25 | "
            f"{_interval(interaction['estimate'], interaction['ci95_low'], interaction['ci95_high'])} | — |"
        )

    lines.extend(
        [
            "",
            "## Frozen surprise groups",
            "",
            "Sustained negative means mean prior standardized surprise ≤ -0.5; sustained positive means ≥ +0.5; values between are approximately neutral. Group boundaries were fixed before comparing them with future errors.",
            "",
            "| Prior family | Prior history | Observations | Mean next error, points [95% interval] | Mean standardized next error [95% interval] |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    group_label = {
        "sustained_negative": "Sustained negative",
        "approximately_neutral": "Approximately neutral",
        "sustained_positive": "Sustained positive",
    }
    for arm, family, _ in EXISTING_ARMS:
        for group, label in group_label.items():
            row = next(
                item
                for item in group_rows
                if item["model_arm"] == arm and item["surprise_group"] == group
            )
            lines.append(
                f"| {family} | {label} | {row['observations']} | "
                f"{_interval(row['mean_next_signed_error'], row['mean_next_signed_error_ci95_low'], row['mean_next_signed_error_ci95_high'])} | "
                f"{_interval(row['mean_next_standardized_error'], row['mean_next_standardized_error_ci95_low'], row['mean_next_standardized_error_ci95_high'])} |"
            )
    lines.extend(
        [
            "",
            "| Prior family | Contrast | Difference in next error, points [95% interval] | Difference in standardized error [95% interval] |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for arm, family, _ in EXISTING_ARMS:
        for contrast, label in (
            ("sustained_negative_minus_neutral", "Negative minus neutral"),
            ("sustained_positive_minus_neutral", "Positive minus neutral"),
            ("sustained_positive_minus_negative", "Positive minus negative"),
        ):
            raw = next(
                row
                for row in contrast_rows
                if row["model_arm"] == arm
                and row["contrast"] == contrast
                and row["metric"] == "margin_points"
            )
            standardized = next(
                row
                for row in contrast_rows
                if row["model_arm"] == arm
                and row["contrast"] == contrast
                and row["metric"] == "standardized_80_widths"
            )
            lines.append(
                f"| {family} | {label} | "
                f"{_interval(raw['difference'], raw['ci95_low'], raw['ci95_high'])} | "
                f"{_interval(standardized['difference'], standardized['ci95_low'], standardized['ci95_high'])} |"
            )

    lines.extend(
        [
            "",
            "## Season stability",
            "",
            "Season-specific surprise slopes use the same adjustment for prior-game count and season progress, without season indicators. These checks describe stability across the four exploratory years.",
            "",
            "| Season | Prior family | Eligible observations | Surprise slope, points [95% interval] | Surprise slope, interval-width units [95% interval] |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for season in SEASONS:
        for arm, family, _ in EXISTING_ARMS:
            raw = seasonal(arm, season, "margin_points")
            standardized = seasonal(arm, season, "standardized_80_widths")
            lines.append(
                f"| {season} | {family} | {raw['observations']} | "
                f"{_interval(raw['estimate'], raw['ci95_low'], raw['ci95_high'])} | "
                f"{_interval(standardized['estimate'], standardized['ci95_low'], standardized['ci95_high'])} |"
            )
    lines.extend(
        [
            "",
            "| Season | Prior family | All target games | At least three eligible prior games |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    for season in SEASONS:
        for arm, family, _ in EXISTING_ARMS:
            count = grouped_counts[(season, arm)]
            lines.append(
                f"| {season} | {family} | {count['all']} | {count['eligible']} |"
            )

    lines.extend(
        [
            "",
            "## Selected examples",
            "",
            f"Examples are the most negative and positive accumulated-surprise observations within each prior family, requiring at least {EXAMPLE_MIN_PRIOR_GAMES} eligible prior games. Selection used accumulated surprise only and did not use the target game's prediction error.",
            "",
            "| Family | Team / season | History | Prior surprise | Preseason strength | Future game | Predicted margin | Actual margin | Signed error |",
            "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in examples:
        lines.append(
            f"| {row['prior_family']} | {row['team']} / {row['season']} | {group_label[row['surprise_group']]} ({row['eligible_prior_game_count']} games) | "
            f"{_fmt(row['accumulated_surprise'])} | {_fmt(row['preseason_rank_strength_percentile'])} | "
            f"{row['opponent']} ({row['game_id']}) | {_fmt(row['predicted_margin'], 1)} | "
            f"{_fmt(row['actual_margin'], 1)} | {_fmt(row['next_signed_error'], 1)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation and recommendation",
            "",
            "Because signed error is observed margin minus predicted margin, positive future errors mean the team outperformed its forecast and negative errors mean it performed worse than forecast. A positive surprise slope is directional persistence; a negative slope is a reversal. The preseason interaction assesses whether that relationship changes continuously with preseason expectations. None of these associations identifies prior overconfidence, likelihood misspecification, posterior inference error, or a latent-rank limitation.",
            "",
            "These analyses are observational and exploratory; bootstrap intervals describe dependence in the retained panel, not independent confirmation. A null interval is failure to detect at this resolution, not proof of no predictive bias. Group contrasts are descriptive complements to the adjusted continuous model.",
            "",
            "**Recommendation under #244:** Keep model behavior unchanged and do not start a cause-specific mechanism search from this inconclusive panel. When the 2026 season is complete, run this frozen analysis across all eligible teams without tuning it to Oregon; report it as motivated exploratory evidence, not independent confirmation. Use 2027 as the first untouched confirmation season. Only open a bounded prior-versus-likelihood mechanism study if the predictive-bias pattern replicates there.",
            "",
            "## Reproducibility and limits",
            "",
            f"Reproduce with `uv run python scripts/run_issue_247_contradictory_evidence.py` ({bootstrap_repetitions:,} bootstrap repetitions). The output manifest records the protocol, retained #200/#201 inputs, configuration, row counts, and SHA-256 hashes. No 2026 outcome, model refit, parameter search, or production change is used.",
            "",
            f"Source parity covered {len(parity_checks)} #201 arm-season rows with maximum absolute metric difference {parity_max:.3g}.",
            "",
        ]
    )
    return "\n".join(lines)


def _output_manifest(
    output_root: Path,
    *,
    input_hashes: dict[str, dict[str, Any]],
    protocol_path: Path,
    config: dict[str, Any],
    counts: dict[str, Any],
) -> dict[str, Any]:
    outputs: dict[str, dict[str, Any]] = {}
    for path in sorted(output_root.iterdir()):
        if not path.is_file() or path.name == "manifest.json":
            continue
        outputs[path.name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return {
        "experiment": "issue-247-accumulated-contradictory-evidence",
        "schema_version": 1,
        "status": "complete",
        "seasons": list(SEASONS),
        "input_hashes": input_hashes,
        "protocol_sha256": sha256_file(protocol_path),
        "config": config,
        "counts": counts,
        "outputs": outputs,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    issue201 = load_issue201_inputs(args.issue201_root)
    preseason_priors, issue200_hashes = load_issue200_preseason_rank_priors(
        args.issue200_root
    )
    team_games = build_team_game_errors(
        issue201["rolling_scores"],
        preseason_priors,
        issue201["evidence_records"],
    )
    observations = build_observations(team_games, issue201["evidence_records"])
    primary_by_arm = {
        arm: [
            row
            for row in observations
            if row["model_arm"] == arm and row["primary_eligible"]
        ]
        for arm, _, _ in EXISTING_ARMS
    }

    model_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    arm_summaries: dict[str, Any] = {}
    for arm, family, _ in EXISTING_ARMS:
        pooled = analyze_arm(
            primary_by_arm[arm],
            repetitions=args.bootstrap_repetitions,
            seed=args.bootstrap_seed,
            label=arm,
        )
        for row in pooled["model_rows"]:
            row["prior_family"] = family
        for row in pooled["group_rows"]:
            row["prior_family"] = family
        for row in pooled["contrast_rows"]:
            row["prior_family"] = family
        model_rows.extend(pooled["model_rows"])
        group_rows.extend(pooled["group_rows"])
        contrast_rows.extend(pooled["contrast_rows"])
        seasonal_counts: dict[str, int] = {}
        for season in SEASONS:
            season_rows = [
                row for row in primary_by_arm[arm] if int(row["season"]) == season
            ]
            seasonal_counts[str(season)] = len(season_rows)
            seasonal_model = analyze_season(
                season_rows,
                repetitions=args.bootstrap_repetitions,
                seed=args.bootstrap_seed,
                label=f"{arm}:{season}",
            )
            for row in seasonal_model:
                row["model_arm"] = arm
                row["prior_family"] = family
            model_rows.extend(seasonal_model)
        arm_summaries[arm] = {
            "pooled_model_rows": pooled["model_rows"],
            "group_rows": pooled["group_rows"],
            "contrast_rows": pooled["contrast_rows"],
            "season_counts": seasonal_counts,
            "preseason_strength_mean": pooled["scales"]["preseason_strength_mean"],
            "preseason_strength_sd": pooled["scales"]["preseason_strength_sd"],
            "preseason_uncertainty_mean": pooled["scales"][
                "preseason_uncertainty_mean"
            ],
            "preseason_uncertainty_sd": pooled["scales"]["preseason_uncertainty_sd"],
        }

    examples = select_examples(observations)
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    _write_csv(output_root / "team_game_errors.csv", TEAM_GAME_FIELDS, team_games)
    _write_csv(
        output_root / "team_game_observations.csv",
        OBSERVATION_FIELDS,
        observations,
    )
    _write_csv(
        output_root / "summary_by_surprise.csv", GROUP_SUMMARY_FIELDS, group_rows
    )
    _write_csv(output_root / "group_contrasts.csv", CONTRAST_FIELDS, contrast_rows)
    _write_csv(output_root / "model_estimates.csv", MODEL_FIELDS, model_rows)
    _write_csv(output_root / "selected_examples.csv", EXAMPLE_FIELDS, examples)

    counts = {
        "issue201_score_rows_read": issue201["all_score_row_count"],
        "rolling_forecast_rows_used": len(issue201["rolling_scores"]),
        "oriented_fbs_team_games_existing_arms": len(team_games),
        "target_team_game_observations": len(observations),
        "primary_eligible_observations": sum(
            bool(row["primary_eligible"]) for row in observations
        ),
        "excluded_below_minimum_history": sum(
            not bool(row["primary_eligible"]) for row in observations
        ),
        "unique_arm_team_seasons": len(
            {(row["season"], row["model_arm"], row["team_id"]) for row in team_games}
        ),
        "unique_schedule_matchups": len(
            {(row["season"], row["game_id"]) for row in team_games}
        ),
        "selected_examples": len(examples),
        "issue201_parity_rows": len(issue201["parity_checks"]),
        "eligible_observations_by_arm_and_season": {
            arm: summary["season_counts"] for arm, summary in arm_summaries.items()
        },
        "surprise_groups_by_arm": {
            arm: {
                row["surprise_group"]: row["observations"]
                for row in group_rows
                if row["model_arm"] == arm
            }
            for arm, _, _ in EXISTING_ARMS
        },
    }
    config = {
        "issue201_root_label": "data/research/issue_201_prebowl_forecast",
        "issue200_root_label": "data/research/matched_history_context_issue_200",
        "model_arms": [arm for arm, _, _ in EXISTING_ARMS],
        "primary_minimum_prior_games": MIN_PRIOR_GAMES,
        "example_minimum_prior_games": EXAMPLE_MIN_PRIOR_GAMES,
        "uncertainty_scale": "half-width of each retained central 80% predictive margin interval",
        "standardized_surprise": "oriented signed margin error divided by the forecast's 80% interval half-width",
        "accumulated_surprise": "unweighted arithmetic mean of every earlier same-season team-game surprise in the exact future forecast evidence set",
        "surprise_group_threshold": SURPRISE_GROUP_THRESHOLD,
        "preseason_strength": "1 - (expected rank - 1) / (FBS population - 1)",
        "preseason_uncertainty": "rank PMF standard deviation / (FBS population - 1)",
        "primary_outcome": "future signed margin error in points",
        "secondary_outcome": "future signed margin error / future forecast 80% interval half-width",
        "ols_adjustments": [
            "continuous preseason strength",
            "continuous preseason uncertainty",
            "accumulated surprise by preseason strength interaction",
            "eligible previous-game count",
            "target team-game number",
            "season fixed effects for pooled models",
        ],
        "cluster_bootstrap": "pigeonhole resampling focal team IDs and target game IDs",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.bootstrap_seed,
        "example_selection": "minimum and maximum accumulated surprise per model family among observations with at least five eligible prior games; target errors are not used for selection",
    }
    payload = {
        "experiment": "issue-247-accumulated-contradictory-evidence",
        "schema_version": 1,
        "input_hashes": {
            **{
                f"issue201/{key}": value
                for key, value in issue201["input_hashes"].items()
            },
            **issue200_hashes,
        },
        "config": config,
        "counts": counts,
        "issue201_parity_checks": issue201["parity_checks"],
        "arm_summaries": arm_summaries,
        "season_model_rows": [
            row for row in model_rows if row["summary_level"] == "season_model_arm"
        ],
        "examples": examples,
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = render_report(
        parity_checks=issue201["parity_checks"],
        observations=observations,
        model_rows=model_rows,
        group_rows=group_rows,
        contrast_rows=contrast_rows,
        examples=examples,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    (output_root / "report.md").write_text(report, encoding="utf-8")

    manifest = _output_manifest(
        output_root,
        input_hashes=payload["input_hashes"],
        protocol_path=args.protocol,
        config=config,
        counts=counts,
    )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue201-root", type=Path, default=DEFAULT_ISSUE201_ROOT)
    parser.add_argument("--issue200-root", type=Path, default=DEFAULT_ISSUE200_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--bootstrap-repetitions", type=int, default=BOOTSTRAP_REPETITIONS
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.bootstrap_repetitions <= 0:
        parser.error("--bootstrap-repetitions must be positive")
    if not args.protocol.is_file():
        parser.error(f"protocol not found: {args.protocol}")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result["counts"], indent=2, sort_keys=True))
    print(f"Report: {args.output_root / 'report.md'}")


if __name__ == "__main__":
    main()
