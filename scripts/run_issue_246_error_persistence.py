"""Run the frozen issue 246 sequential forecast-error analysis."""

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
    ARMS,
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    CONTRAST_FIELDS,
    LAG_PAIR_FIELDS,
    PRESEASON_PRIOR_FIELDS,
    SUMMARY_FIELDS,
    TEAM_GAME_FIELDS,
    build_lagged_pairs,
    build_team_game_errors,
    load_issue200_preseason_rank_priors,
    load_issue201_inputs,
    output_manifest,
    summarize_lagged_pairs,
    transition_audit,
)

DEFAULT_ISSUE201_ROOT = ROOT / "data/research/issue_201_prebowl_forecast"
DEFAULT_ISSUE200_ROOT = ROOT / "data/research/matched_history_context_issue_200"
DEFAULT_PROTOCOL = ROOT / "docs/issue_246_error_persistence.md"
DEFAULT_OUTPUT = ROOT / "data/research/issue_246_error_persistence"
AUDIT_FIELDS = (
    "season",
    "model_arm",
    "prior_family",
    "policy",
    "candidate_transitions",
    "eligible_transitions",
    "excluded_transitions",
)


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
            handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n"
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


def _summary_lookup(
    summaries: list[dict[str, Any]],
    level: str,
    arm: str,
    *,
    season: int | str | None = None,
    prior_group: str = "",
    progress: str = "",
) -> dict[str, Any]:
    for row in summaries:
        if (
            row["summary_level"] == level
            and row["model_arm"] == arm
            and (season is None or str(row["season"]) == str(season))
            and row["preseason_rank_prior_group"] == prior_group
            and row["season_progress_bin"] == progress
        ):
            return row
    raise ValueError(
        f"missing issue 246 summary row: {level}/{arm}/{season}/{prior_group}/{progress}"
    )


def _contrast_lookup(
    contrasts: list[dict[str, Any]], arm: str, contrast: str, metric: str
) -> dict[str, Any]:
    for row in contrasts:
        if (
            row["model_arm"] == arm
            and row["contrast"] == contrast
            and row["metric"] == metric
        ):
            return row
    raise ValueError(f"missing issue 246 contrast row: {arm}/{contrast}/{metric}")


def render_report(
    *,
    parity_checks: list[dict[str, Any]],
    team_games: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    audit: list[dict[str, Any]],
    preseason_rank_classifications: list[dict[str, Any]],
    bootstrap_repetitions: int,
) -> str:
    unique_games = {(row["season"], row["game_id"]) for row in team_games}
    unique_fbs_team_seasons = {
        (row["season"], row["model_arm"], row["team_id"]) for row in team_games
    }
    eligible_pairs = [row for row in pairs if row["eligible"]]
    excluded_pairs = len(pairs) - len(eligible_pairs)
    parity_max = max(
        (float(row["max_metric_absolute_difference"]) for row in parity_checks),
        default=0.0,
    )
    pooled_rows = [
        _summary_lookup(summaries, "pooled_model_arm", arm, season="2022-2025")
        for arm, _, _ in ARMS
    ]
    season_rows = [
        row for row in summaries if row["summary_level"] == "season_model_arm"
    ]
    progress_rows = [
        row for row in summaries if row["summary_level"] == "pooled_season_progress"
    ]
    prior_contrasts = [
        row
        for row in contrasts
        if row["contrast"] == "strong_preseason_rank_prior_minus_other"
    ]
    progress_contrasts = [
        row for row in contrasts if row["contrast"] == "late_minus_early"
    ]

    def covers_zero(row: dict[str, Any], low_key: str, high_key: str) -> bool:
        low, high = row[low_key], row[high_key]
        return low is not None and high is not None and low <= 0 <= high

    pooled_intervals_cover_null = all(
        covers_zero(row, "correlation_ci95_low", "correlation_ci95_high")
        and covers_zero(
            row, "directional_excess_ci95_low", "directional_excess_ci95_high"
        )
        for row in pooled_rows
    )
    season_intervals_cover_null = all(
        covers_zero(row, "correlation_ci95_low", "correlation_ci95_high")
        and covers_zero(
            row, "directional_excess_ci95_low", "directional_excess_ci95_high"
        )
        for row in season_rows
    )
    progress_intervals_cover_null = all(
        covers_zero(row, "correlation_ci95_low", "correlation_ci95_high")
        and covers_zero(
            row, "directional_excess_ci95_low", "directional_excess_ci95_high"
        )
        for row in progress_rows
    )
    prior_differences_cover_zero = all(
        covers_zero(row, "ci95_low", "ci95_high") for row in prior_contrasts
    )
    progress_differences_cover_zero = all(
        covers_zero(row, "ci95_low", "ci95_high") for row in progress_contrasts
    )
    no_consistent_signal = (
        pooled_intervals_cover_null
        and season_intervals_cover_null
        and progress_intervals_cover_null
        and prior_differences_cover_zero
        and progress_differences_cover_zero
    )
    pooled_r_values = [float(row["lag1_correlation"]) for row in pooled_rows]
    pooled_excess_values = [
        float(row["directional_excess_over_independence"]) for row in pooled_rows
    ]
    if no_consistent_signal:
        conclusion = (
            "No consistent sequential error-persistence signal was detected. Across the four pooled arms, "
            f"lag-1 correlations range from {min(pooled_r_values):.3f} to {max(pooled_r_values):.3f}, "
            "and directional excess over the sign-marginal independence baseline ranges from "
            f"{min(pooled_excess_values):.3f} to {max(pooled_excess_values):.3f}; their 95% intervals "
            "include zero. Season, progress, and preseason-rank-prior contrasts also include their "
            "no-effect values. These results fail to detect a consistent effect; they do not demonstrate "
            "that meaningful effects are absent."
        )
    else:
        conclusion = (
            "The sequential-error results are mixed or leave a possible effect. The pooled estimates do "
            "not settle the question; interpret the arm, season, preseason-rank-prior, and progress "
            "intervals together."
        )
    recommendation = (
        "Defer model changes whose rationale is sequential error persistence based on this exploratory "
        "retrospective analysis. This does not resolve the broader calibration epic or establish that the "
        "prior-likelihood balance is correct; assess other calibration questions using their own evidence."
    )

    def directional_rate(row: dict[str, Any]) -> str:
        return (
            f"{_fmt(row['directional_persistence'])} / "
            f"{_fmt(row['directional_independence_baseline'])}"
        )

    def directional_excess(row: dict[str, Any]) -> str:
        return _interval(
            row["directional_excess_over_independence"],
            row["directional_excess_ci95_low"],
            row["directional_excess_ci95_high"],
        )

    lines = [
        "# Issue 246: sequential in-season forecast-error persistence",
        "",
        "## Result",
        "",
        "This leakage-safe descriptive analysis asks whether a team's signed pregame margin error in one regular-season game predicts the error in that team's next game, after the earlier result enters the next forecast's evidence.",
        "",
        "The four previously studied seasons (2022–2025) are exploratory historical data, not independent confirmation. No models are refit and no production behavior is changed.",
        "",
        f"The retained #201 primary FBS–FBS score summaries passed parity for **{len(parity_checks)} arm-season rows**; the largest absolute difference from its stored mean CRPS or margin NLL was {parity_max:.3g}. The #201 inputs and retained #200 rank-PMF bundle are recorded by hash in manifest.json.",
        "",
        f"**Finding:** {conclusion}",
        "",
        f"**Recommendation:** {recommendation}",
        "",
        "## Sample and eligibility",
        "",
        f"The panel contains **{len(team_games):,} oriented FBS team-games** across **{len(unique_games):,} matchups** and **{len(unique_fbs_team_seasons):,} model-arm/team-seasons**. It includes both FBS perspectives for FBS–FBS games and the FBS perspective for FBS–FCS games, preserving each FBS team's regular-season sequence.",
        "",
        f"The retained #200 preseason-rank PMFs classify **{len(preseason_rank_classifications):,} team-seasons**, once per model arm. There are **{len(pairs):,} adjacent candidate transitions**; **{len(eligible_pairs):,}** pass the evidence check and **{excluded_pairs:,}** are excluded because the previous game is absent from the next forecast's evidence IDs. Excluded pairs and reasons remain in lagged_error_pairs.csv.",
        "",
        "| Season | Prior family | Policy | Candidate pairs | Eligible pairs | Excluded |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in audit:
        lines.append(
            f"| {row['season']} | {row['prior_family']} | {row['policy']} | "
            f"{row['candidate_transitions']} | {row['eligible_transitions']} | {row['excluded_transitions']} |"
        )

    lines.extend(
        [
            "",
            "## Lag-1 results by season and prior family",
            "",
            "Signed error is observed team margin minus expected team margin; positive means the team performed better than expected. Lag-1 correlation is Pearson correlation between consecutive eligible errors. For nonzero error pairs, the table shows the observed same-direction share and the independence baseline computed from the observed positive-sign shares for the previous and next errors. The directional excess is observed persistence minus that baseline; zero is the no-association reference.",
            "",
            f"Intervals are 95% percentile intervals from {bootstrap_repetitions:,} deterministic multiway pigeonhole bootstrap repetitions, resampling focal team IDs and game IDs jointly across both lag positions. The directional-excess interval recomputes the sign marginals and independence baseline on every resample.",
            "",
            "| Season | Prior family | Policy | Pairs | Teams | Lag-1 r [95% interval] | Same direction / baseline | Excess over independence [95% interval] |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    seasons = sorted({int(row["season"]) for row in audit})
    for season in seasons:
        for arm, family, policy in ARMS:
            row = _summary_lookup(summaries, "season_model_arm", arm, season=season)
            lines.append(
                f"| {season} | {family} | {policy} | {row['transitions']} | {row['team_count']} | "
                f"{_interval(row['lag1_correlation'], row['correlation_ci95_low'], row['correlation_ci95_high'])} | "
                f"{directional_rate(row)} | {directional_excess(row)} |"
            )

    lines.extend(
        [
            "",
            "## Pooled results by model arm",
            "",
            "The previous/next positive-sign shares show why 0.5 is not assumed as the null. With marginal positive shares p1 and p2, the independence baseline is p1*p2 + (1-p1)*(1-p2).",
            "",
            "| Prior family | Policy | Pairs | Positive previous / next | Same direction / independence baseline | Excess over independence [95% interval] | Lag-1 r [95% interval] |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm, family, policy in ARMS:
        row = _summary_lookup(summaries, "pooled_model_arm", arm, season="2022-2025")
        sign_shares = (
            f"{_fmt(row['positive_previous_error_share'])} / "
            f"{_fmt(row['positive_next_error_share'])}"
        )
        lines.append(
            f"| {family} | {policy} | {row['transitions']} | {sign_shares} | "
            f"{directional_rate(row)} | {directional_excess(row)} | "
            f"{_interval(row['lag1_correlation'], row['correlation_ci95_low'], row['correlation_ci95_high'])} |"
        )

    lines.extend(
        [
            "",
            "## Strong preseason rank priors",
            "",
            "A team-season is classified independently within each arm from its retained #200 preseason rank PMF. Strong prior means P(rank <= 10) >= 0.50. The group label is fixed for every game and transition in that team-season; it does not depend on opponents or game-level win probabilities.",
            "",
            "| Prior family | Policy | Stratum | Pairs | Same direction / independence baseline | Excess over independence [95% interval] | Lag-1 r [95% interval] |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm, family, policy in ARMS:
        for prior_group in ("strong_preseason_rank_prior", "other"):
            row = _summary_lookup(
                summaries,
                "pooled_prior_stratum",
                arm,
                season="2022-2025",
                prior_group=prior_group,
            )
            lines.append(
                f"| {family} | {policy} | {prior_group} | {row['transitions']} | "
                f"{directional_rate(row)} | {directional_excess(row)} | "
                f"{_interval(row['lag1_correlation'], row['correlation_ci95_low'], row['correlation_ci95_high'])} |"
            )
    lines.extend(
        [
            "",
            "Paired strong-prior-minus-other differences:",
            "",
            "| Prior family | Policy | Metric | Difference [95% interval] |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for arm, family, policy in ARMS:
        for metric in ("lag1_correlation", "directional_excess_over_independence"):
            row = _contrast_lookup(
                contrasts, arm, "strong_preseason_rank_prior_minus_other", metric
            )
            lines.append(
                f"| {family} | {policy} | {metric} | "
                f"{_interval(row['difference_a_minus_b'], row['ci95_low'], row['ci95_high'])} |"
            )

    lines.extend(
        [
            "",
            "## Change over the season",
            "",
            "Transitions are assigned by the destination team's game number: early games 2–4, middle games 5–8, and late games 9+.",
            "",
            "| Prior family | Policy | Progress | Pairs | Same direction / independence baseline | Excess over independence [95% interval] | Lag-1 r [95% interval] |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    progress_labels = {
        "early_games_2_4": "Early (games 2–4)",
        "middle_games_5_8": "Middle (games 5–8)",
        "late_game_9_plus": "Late (game 9+)",
    }
    for arm, family, policy in ARMS:
        for progress, label in progress_labels.items():
            row = _summary_lookup(
                summaries,
                "pooled_season_progress",
                arm,
                season="2022-2025",
                progress=progress,
            )
            lines.append(
                f"| {family} | {policy} | {label} | {row['transitions']} | "
                f"{directional_rate(row)} | {directional_excess(row)} | "
                f"{_interval(row['lag1_correlation'], row['correlation_ci95_low'], row['correlation_ci95_high'])} |"
            )
    lines.extend(
        [
            "",
            "Paired late-minus-early differences:",
            "",
            "| Prior family | Policy | Metric | Difference [95% interval] |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for arm, family, policy in ARMS:
        for metric in ("lag1_correlation", "directional_excess_over_independence"):
            row = _contrast_lookup(contrasts, arm, "late_minus_early", metric)
            lines.append(
                f"| {family} | {policy} | {metric} | "
                f"{_interval(row['difference_a_minus_b'], row['ci95_low'], row['ci95_high'])} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation and limits",
            "",
            "Intervals that include zero mean the analysis did not detect a clear effect at this resolution; they do not demonstrate that meaningful effects are absent. These intervals are not equivalence tests. Likewise, lack of lag-1 persistence does not establish that the prior-likelihood balance is correct.",
            "",
            "The retained #201 forecasts account for opponent strength and game site. Error persistence alone does not identify a cause or justify changing a likelihood, prior, or production behavior. The broader calibration epic remains a separate question.",
            "",
            "This differs from issue #34, which examined historical performance/rank residuals. Issue 246 uses leakage-safe pregame margin forecasts and requires the earlier game to appear in the next forecast's evidence. The four arms are shown separately; repeated teams and shared matchups are handled in the bootstrap, while the four seasons remain exploratory historical data.",
            "",
            "## Reproducibility",
            "",
            "Reproduce from the repository root with: uv run python scripts/run_issue_246_error_persistence.py",
            "",
            "The runner verifies the retained #201 forecast artifacts and the #200 pipeline bundle and PMF hashes. It does not refit models or rebuild the forecasting pipeline. preseason_rank_classification.csv, team_game_errors.csv, and lagged_error_pairs.csv retain the team-season labels, game errors, and transition eligibility decisions; all outputs and inputs are hashed in manifest.json.",
            "",
            "## Source parity",
            "",
            "| Arm-season rows checked | Maximum absolute difference | Status |",
            "| ---: | ---: | --- |",
            f"| {len(parity_checks)} | {parity_max:.3g} | passed |",
            "",
        ]
    )
    return "\n".join(lines)

def run(args: argparse.Namespace) -> dict[str, Any]:
    issue201 = load_issue201_inputs(args.issue201_root)
    preseason_rank_classifications, issue200_hashes = (
        load_issue200_preseason_rank_priors(args.issue200_root)
    )
    input_hashes = {
        f"issue201/{name}": record
        for name, record in issue201["input_hashes"].items()
    }
    input_hashes.update(issue200_hashes)
    team_games = build_team_game_errors(
        issue201["rolling_scores"],
        preseason_rank_classifications,
        issue201["evidence_records"],
    )
    pairs = build_lagged_pairs(team_games, issue201["evidence_records"])
    summaries, contrasts = summarize_lagged_pairs(
        pairs,
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed,
    )
    audit = transition_audit(pairs)
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    _write_csv(
        output_root / "preseason_rank_classification.csv",
        PRESEASON_PRIOR_FIELDS,
        preseason_rank_classifications,
    )
    _write_csv(output_root / "team_game_errors.csv", TEAM_GAME_FIELDS, team_games)
    _write_csv(output_root / "lagged_error_pairs.csv", LAG_PAIR_FIELDS, pairs)
    _write_csv(output_root / "aggregate_summary.csv", SUMMARY_FIELDS, summaries)
    _write_csv(output_root / "contrasts.csv", CONTRAST_FIELDS, contrasts)
    _write_csv(output_root / "transition_audit.csv", AUDIT_FIELDS, audit)
    report = render_report(
        parity_checks=issue201["parity_checks"],
        team_games=team_games,
        pairs=pairs,
        summaries=summaries,
        contrasts=contrasts,
        audit=audit,
        preseason_rank_classifications=preseason_rank_classifications,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    (output_root / "report.md").write_text(report, encoding="utf-8")

    counts = {
        "issue201_score_rows_read": issue201["all_score_row_count"],
        "rolling_forecast_rows_used": len(issue201["rolling_scores"]),
        "oriented_fbs_team_games": len(team_games),
        "unique_schedule_matchups": len(
            {(row["season"], row["game_id"]) for row in team_games}
        ),
        "team_seasons_by_arm": len(
            {(row["season"], row["model_arm"], row["team_id"]) for row in team_games}
        ),
        "candidate_transitions": len(pairs),
        "eligible_transitions": sum(bool(row["eligible"]) for row in pairs),
        "excluded_transitions": sum(not bool(row["eligible"]) for row in pairs),
        "issue201_parity_rows": len(issue201["parity_checks"]),
        "preseason_rank_classifications": len(preseason_rank_classifications),
        "strong_preseason_rank_prior_team_seasons": sum(
            row["preseason_rank_prior_group"] == "strong_preseason_rank_prior"
            for row in preseason_rank_classifications
        ),
    }
    payload = {
        "experiment": "issue-246-sequential-game-error-persistence",
        "schema_version": 1,
        "input_hashes": input_hashes,
        "config": {
            "issue201_root_label": "data/research/issue_201_prebowl_forecast",
            "issue200_root_label": "data/research/matched_history_context_issue_200",
            "score_population": ["fbs_fbs", "fbs_fcs"],
            "focal_team_population": "FBS only",
            "signed_error": "observed team margin minus expected team margin",
            "preseason_rank_prior_classification": "P(rank <= 10) >= 0.50 from the #200 preseason rank PMF; once per team-season per model arm",
            "directional_independence_baseline": "p(previous error > 0) * p(next error > 0) + p(previous error < 0) * p(next error < 0), among nonzero error pairs",
            "season_progress_bins": {
                "early_games_2_4": [2, 4],
                "middle_games_5_8": [5, 8],
                "late_game_9_plus": [9, None],
            },
            "cluster_bootstrap": "pigeonhole resampling of focal team IDs and both game IDs per transition",
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "counts": counts,
        "issue201_parity_checks": issue201["parity_checks"],
        "aggregate_summary": summaries,
        "contrasts": contrasts,
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = output_manifest(
        output_root,
        input_hashes=input_hashes,
        protocol_path=args.protocol,
        config=payload["config"],
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
