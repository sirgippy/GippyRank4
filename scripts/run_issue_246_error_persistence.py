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
    SUMMARY_FIELDS,
    TEAM_GAME_FIELDS,
    build_lagged_pairs,
    build_team_game_errors,
    load_issue201_inputs,
    output_manifest,
    summarize_lagged_pairs,
    transition_audit,
)

DEFAULT_ISSUE201_ROOT = ROOT / "data/research/issue_201_prebowl_forecast"
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
    favorite: str = "",
    progress: str = "",
) -> dict[str, Any]:
    for row in summaries:
        if (
            row["summary_level"] == level
            and row["model_arm"] == arm
            and (season is None or str(row["season"]) == str(season))
            and row["preseason_favorite_group"] == favorite
            and row["season_progress_bin"] == progress
        ):
            return row
    raise ValueError(
        f"missing issue 246 summary row: {level}/{arm}/{season}/{favorite}/{progress}"
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
    favorite_contrasts = [
        row
        for row in contrasts
        if row["contrast"] == "strong_preseason_favorite_minus_other"
    ]
    progress_contrasts = [
        row for row in contrasts if row["contrast"] == "late_minus_early"
    ]
    all_pooled_intervals_cover_null = all(
        row["correlation_ci95_low"] <= 0 <= row["correlation_ci95_high"]
        and row["directional_ci95_low"] <= 0.5 <= row["directional_ci95_high"]
        for row in pooled_rows
    )
    all_season_intervals_cover_null = all(
        row["correlation_ci95_low"] <= 0 <= row["correlation_ci95_high"]
        and row["directional_ci95_low"] <= 0.5 <= row["directional_ci95_high"]
        for row in season_rows
    )
    all_progress_intervals_cover_null = all(
        row["correlation_ci95_low"] <= 0 <= row["correlation_ci95_high"]
        and row["directional_ci95_low"] <= 0.5 <= row["directional_ci95_high"]
        for row in progress_rows
    )
    favorite_differences_cover_zero = all(
        row["ci95_low"] <= 0 <= row["ci95_high"] for row in favorite_contrasts
    )
    progress_differences_cover_zero = all(
        row["ci95_low"] <= 0 <= row["ci95_high"] for row in progress_contrasts
    )
    no_consistent_signal = (
        all_pooled_intervals_cover_null
        and all_season_intervals_cover_null
        and all_progress_intervals_cover_null
        and favorite_differences_cover_zero
        and progress_differences_cover_zero
    )
    pooled_r_values = [float(row["lag1_correlation"]) for row in pooled_rows]
    pooled_direction_values = [
        float(row["directional_persistence"]) for row in pooled_rows
    ]
    if no_consistent_signal:
        conclusion = (
            "No consistent sequential directional-error signal was detected. Across the four pooled arms, "
            f"lag-1 correlations range from {min(pooled_r_values):.3f} to {max(pooled_r_values):.3f}, "
            f"and same-direction rates range from {min(pooled_direction_values):.3f} to "
            f"{max(pooled_direction_values):.3f}. Their intervals cover zero and 0.5 respectively; "
            "the season, progress, and preseason-favorite contrasts also include their no-effect values. "
            "Weak effects remain possible because these intervals are not equivalence tests."
        )
        recommendation = (
            "Defer model changes and further retrospective responsiveness probes. If this question remains "
            "decision-relevant, run a pre-registered prospective replication on a genuinely later holdout, "
            "keeping the same evidence-membership rule and team/game dependence handling. Do not tune the "
            "model from this historical panel."
        )
    else:
        conclusion = (
            "The sequential-error results are mixed or leave a possible effect; interpret the arm, season, "
            "favorite, and progress tables together rather than treating the pooled estimate as confirmation."
        )
        recommendation = (
            "Do not change model behavior from this historical panel. Scope any next responsiveness study "
            "around the strata and uncertainty shown here, and use a pre-registered holdout if one is available."
        )
    lines = [
        "# Issue 246: sequential in-season forecast-error persistence",
        "",
        "## Result",
        "",
        "This is a leakage-safe descriptive analysis of retained issue 201 forecasts. It asks whether a team's signed pregame margin error in one regular-season game predicts the error in that same team's next game, after the earlier result is present in the next forecast's evidence.",
        "",
        "The sample contains **four previously studied seasons (2022–2025)**, so the estimates are exploratory and are not independent confirmation. The analysis makes no model or production changes.",
        "",
        f"The retained issue 201 primary FBS–FBS score summaries passed parity for **{len(parity_checks)} arm-season rows**; the largest absolute difference from its stored mean CRPS or margin NLL was `{parity_max:.3g}`. The source score, summary, evidence, and manifest hashes are recorded in `manifest.json`.",
        "",
        f"**Finding:** {conclusion}",
        "",
        f"**Recommendation:** {recommendation}",
        "",
        "## Sample and eligibility",
        "",
        f"The team-game panel has **{len(team_games):,} oriented FBS team-games** across **{len(unique_games):,} scheduled matchups** and **{len(unique_fbs_team_seasons):,} model-arm/team-seasons**. It includes each FBS–FBS game from both FBS teams' perspectives, and each FBS–FCS game once from the FBS team's perspective. FBS–FCS rows are needed to keep the FBS team's full regular-season sequence; FCS-only team histories are not inferred from this corpus.",
        "",
        f"There are **{len(pairs):,} adjacent candidate transitions**; **{len(eligible_pairs):,}** pass the evidence check and **{excluded_pairs:,}** are excluded because the previous game is absent from the next forecast's included evidence IDs. The excluded pairs remain in `lagged_error_pairs.csv` with reasons. The next game's own game ID is checked to be absent from its evidence, and its cutoff must precede kickoff.",
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
            "Signed error is `observed team margin − expected team margin`. Positive means the team scored better than the model expected from its perspective; negative means worse. The retained #201 predictive mean is used. Lag-1 correlation is Pearson correlation between consecutive eligible signed errors. Directional persistence is the share of nonzero pairs with the same sign; 0.5 is the no-directional-persistence reference.",
            "",
            f"Intervals are 95% percentile intervals from {bootstrap_repetitions:,} deterministic multiway pigeonhole bootstrap repetitions, resampling focal team IDs and game IDs jointly across both lag positions. This preserves repeated-team and shared-matchup dependence, including the two orientations of an FBS–FBS matchup. The intervals describe this historical panel and do not create four independent seasons.",
            "",
            "| Season | Prior family | Policy | Eligible pairs | Teams | Lag-1 r [95% interval] | Same-direction rate [95% interval] |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    seasons = sorted({int(row["season"]) for row in audit})
    for season in seasons:
        for arm, family, policy in ARMS:
            row = _summary_lookup(summaries, "season_model_arm", arm, season=season)
            lines.append(
                f"| {season} | {family} | {policy} | {row['transitions']} | {row['team_count']} | "
                f"{_interval(row['lag1_correlation'], row['correlation_ci95_low'], row['correlation_ci95_high'])} | "
                f"{_interval(row['directional_persistence'], row['directional_ci95_low'], row['directional_ci95_high'])} |"
            )

    lines.extend(
        [
            "",
            "## Pooled results by model arm",
            "",
            "| Prior family | Policy | Eligible pairs | Lag-1 r [95% interval] | Same-direction rate [95% interval] |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for arm, family, policy in ARMS:
        row = _summary_lookup(summaries, "pooled_model_arm", arm, season="2022-2025")
        lines.append(
            f"| {family} | {policy} | {row['transitions']} | "
            f"{_interval(row['lag1_correlation'], row['correlation_ci95_low'], row['correlation_ci95_high'])} | "
            f"{_interval(row['directional_persistence'], row['directional_ci95_low'], row['directional_ci95_high'])} |"
        )

    lines.extend(
        [
            "",
            "## Strong preseason favorites",
            "",
            "A destination game is a strong preseason favorite case when the focal team's issue 201 preseason-only win probability is at least 0.80. This fixed threshold uses no in-season outcome. Each arm's rows are classified independently from its own preseason forecast.",
            "",
            "| Prior family | Policy | Stratum | Eligible pairs | Lag-1 r [95% interval] | Same-direction rate [95% interval] |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for arm, family, policy in ARMS:
        for favorite in ("strong_preseason_favorite", "other"):
            row = _summary_lookup(
                summaries,
                "pooled_favorite_stratum",
                arm,
                season="2022-2025",
                favorite=favorite,
            )
            lines.append(
                f"| {family} | {policy} | {favorite} | {row['transitions']} | "
                f"{_interval(row['lag1_correlation'], row['correlation_ci95_low'], row['correlation_ci95_high'])} | "
                f"{_interval(row['directional_persistence'], row['directional_ci95_low'], row['directional_ci95_high'])} |"
            )
    lines.extend(
        [
            "",
            "The next table reports the paired stratum difference (strong favorite minus other). The difference intervals use the same team and matchup resamples for both strata.",
            "",
            "| Prior family | Policy | Metric | Difference [95% interval] |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for arm, family, policy in ARMS:
        for metric in ("lag1_correlation", "directional_persistence"):
            row = _contrast_lookup(
                contrasts, arm, "strong_preseason_favorite_minus_other", metric
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
            "Destination team-game ordinals 2–4 are early, 5–8 are middle, and 9 or later are late. Each transition is assigned according to the next game's ordinal.",
            "",
            "| Prior family | Policy | Season progress | Eligible pairs | Lag-1 r [95% interval] | Same-direction rate [95% interval] |",
            "| --- | --- | --- | ---: | ---: | ---: |",
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
                f"{_interval(row['lag1_correlation'], row['correlation_ci95_low'], row['correlation_ci95_high'])} | "
                f"{_interval(row['directional_persistence'], row['directional_ci95_low'], row['directional_ci95_high'])} |"
            )
    lines.extend(
        [
            "",
            "Paired late-minus-early differences (late minus early):",
            "",
            "| Prior family | Policy | Metric | Difference [95% interval] |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for arm, family, policy in ARMS:
        for metric in ("lag1_correlation", "directional_persistence"):
            row = _contrast_lookup(contrasts, arm, "late_minus_early", metric)
            lines.append(
                f"| {family} | {policy} | {metric} | "
                f"{_interval(row['difference_a_minus_b'], row['ci95_low'], row['ci95_high'])} |"
            )
    lines.extend(
        [
            "",
            "Read sparse late-season rows with their interval widths.",
            "",
            "## Interpretation and limits",
            "",
            "The existing #201 forecasts already account for opponent strength and game site. A positive sequential association therefore describes residual errors that remain after the earlier game was available to the later posterior. It does not identify whether a specific information source, likelihood feature, or changing team state caused the association.",
            "",
            "This differs from issue #34: that study examined persistence in historical performance/rank residuals and recency-style probes. Issue 246 measures errors in leakage-safe pregame game-margin forecasts, then requires the preceding game to appear in the next forecast's evidence. No rankings or in-sample fitted-game residuals are substituted here.",
            "",
            "The four arms are shown separately: History and Context, each under the existing and pre-bowl policies. Policies are not pooled as independent replications. Repeated team-game pairs and shared matchup orientations are handled in the bootstrap, but the underlying four seasons remain previously studied exploratory data. This report supports a bounded decision about posterior responsiveness; it does not justify a model change or parameter tuning.",
            "",
            "## Reproducibility",
            "",
            "Run from the repository root with:",
            "",
            "```bash",
            "uv run python scripts/run_issue_246_error_persistence.py",
            "```",
            "",
            "The runner reads only the retained issue 201 score, summary, and replay-evidence artifacts; it verifies their stored hashes and reproduces issue 201's primary per-season CRPS and margin NLL means before calculating sequential errors. `team_game_errors.csv` and `lagged_error_pairs.csv` retain the auditable observations and eligibility decisions. The complete input and output hashes are in `manifest.json`.",
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
    team_games = build_team_game_errors(
        issue201["rolling_scores"],
        issue201["preseason_scores"],
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
    }
    payload = {
        "experiment": "issue-246-sequential-game-error-persistence",
        "schema_version": 1,
        "input_hashes": issue201["input_hashes"],
        "config": {
            "issue201_root_label": "data/research/issue_201_prebowl_forecast",
            "score_population": ["fbs_fbs", "fbs_fcs"],
            "focal_team_population": "FBS only",
            "signed_error": "observed team margin minus expected team margin",
            "strong_preseason_favorite_probability": 0.80,
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
        input_hashes=issue201["input_hashes"],
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
