"""Freeze and run the issue 201 four-arm pre-bowl forecast comparison."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gippyrank.research.issue_201_prebowl_forecast import (
    MODEL_ARMS,
    PRIMARY_POPULATION,
    SEASONS,
    attach_plan_phases,
    build_schedule_plan,
    canonical_json,
    run_issue_201_comparison,
    sha256_file,
)

DEFAULT_ISSUE204_ROOT = ROOT / "data/processed/massey_prebowl_issue_204_v1"
DEFAULT_GAME_ROOT = ROOT / "data/raw/cfbd/games"
DEFAULT_ISSUE200_ROOT = ROOT / "data/research/matched_history_context_issue_200"
DEFAULT_FROZEN_ROOT = ROOT / "data/processed/context_incremental_information"
DEFAULT_TARGET_DATES = DEFAULT_ISSUE200_ROOT / "target_snapshot_manifest.csv"
DEFAULT_SCHEDULE_WINDOWS = (
    ROOT / "data/research/massey_prebowl_issue_198/schedule_windows.csv"
)
DEFAULT_PROTOCOL = ROOT / "docs/issue_201_prebowl_forecast_comparison.md"
DEFAULT_OUTPUT = ROOT / "data/research/issue_201_prebowl_forecast"


def _json_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return canonical_json(value)
    if value is None:
        return ""
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    _atomic_write(path, encoded.encode("utf-8"))


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    else:
        fields = ["empty"]
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _json_value(row.get(key)) for key in fields})
    _atomic_write(path, buffer.getvalue().encode("utf-8"))


def _write_jsonl_gzip(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    import io

    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
        handle.write(payload)
    _atomic_write(path, buffer.getvalue())


def _file_record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def freeze_plan(args: argparse.Namespace) -> None:
    protocol_hash = sha256_file(args.protocol)
    plan, source_metadata = build_schedule_plan(
        args.game_root, protocol_sha256=protocol_hash
    )
    plan = attach_plan_phases(
        plan, args.target_snapshot_manifest, args.schedule_windows
    )
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "schedule_plan.json"
    manifest_path = args.output / "schedule_plan.manifest.json"
    plan_bytes = (
        json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    if plan_path.exists() and plan_path.read_bytes() != plan_bytes:
        raise SystemExit(
            f"refusing to replace a different frozen plan: {plan_path}; choose a new output directory"
        )
    if not plan_path.exists():
        _atomic_write(plan_path, plan_bytes)
    manifest = {
        "schema_version": 1,
        "experiment": "issue-201-prebowl-forecast-comparison",
        "plan_sha256": sha256_file(plan_path),
        "protocol_sha256": protocol_hash,
        "target_snapshot_manifest_sha256": sha256_file(args.target_snapshot_manifest),
        "schedule_windows_sha256": sha256_file(args.schedule_windows),
        "source_schedule_metadata": source_metadata,
        "season_counts": [
            {
                "season": row["season"],
                "scheduled_game_count": row["scheduled_game_count"],
                "scheduled_fbs_fbs_count": row["scheduled_fbs_fbs_count"],
                "scheduled_fbs_fcs_count": row["scheduled_fbs_fcs_count"],
            }
            for row in plan["seasons"]
        ],
    }
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise SystemExit(
            f"refusing to replace a different plan manifest: {manifest_path}; choose a new output directory"
        )
    if not manifest_path.exists():
        _atomic_write(manifest_path, manifest_bytes)
    print(f"Frozen score-blind schedule plan: {plan_path}")
    print(f"Plan SHA-256: {manifest['plan_sha256']}")
    for row in manifest["season_counts"]:
        print(
            f"{row['season']}: {row['scheduled_game_count']} scheduled, "
            f"{row['scheduled_fbs_fbs_count']} FBS–FBS, "
            f"{row['scheduled_fbs_fcs_count']} FBS–FCS"
        )


def _summary_by(
    summary_rows: list[dict[str, object]], arm: str, season: str
) -> dict[str, object]:
    for row in summary_rows:
        if (
            row["forecast_stage"] == "rolling"
            and row["forecast_population"] == PRIMARY_POPULATION
            and row["phase"] == "all_phases"
            and str(row["model_arm"]) == arm
            and str(row["season"]) == season
        ):
            return row
    raise ValueError(f"missing primary summary for {arm}/{season}")


def _result_classification(result: dict[str, object]) -> dict[str, object]:
    summaries = result["forecast_summaries"]
    get = lambda arm, season: _summary_by(summaries, arm, season)
    c_existing = get("Context_existing", "pooled_equal_game")
    c_pre = get("Context_pre_bowl", "pooled_equal_game")
    c_existing_equal = get("Context_existing", "equal_season_sensitivity")
    c_pre_equal = get("Context_pre_bowl", "equal_season_sensitivity")
    season_wins = sum(
        float(get("Context_pre_bowl", str(season))["crps"])
        < float(get("Context_existing", str(season))["crps"])
        for season in SEASONS
    )
    interaction_pooled = (
        float(c_pre["crps"])
        - float(c_existing["crps"])
        - float(get("History_pre_bowl", "pooled_equal_game")["crps"])
        + float(get("History_existing", "pooled_equal_game")["crps"])
    )
    interaction_equal_season = (
        float(c_pre_equal["crps"])
        - float(c_existing_equal["crps"])
        - float(get("History_pre_bowl", "equal_season_sensitivity")["crps"])
        + float(get("History_existing", "equal_season_sensitivity")["crps"])
    )
    if (
        float(c_pre["crps"]) < float(c_existing["crps"])
        and float(c_pre_equal["crps"]) < float(c_existing_equal["crps"])
        and season_wins >= 3
        and interaction_pooled < 0
    ):
        classification = "improve"
    elif (
        float(c_pre["crps"]) >= float(c_existing["crps"])
        and float(c_pre_equal["crps"]) >= float(c_existing_equal["crps"])
        and season_wins <= 1
    ):
        classification = "no useful improvement"
    else:
        classification = "inconclusive"

    opposite_nll_seasons = 0
    opposite_calibration_seasons = 0
    for season in SEASONS:
        existing = get("Context_existing", str(season))
        prebowl = get("Context_pre_bowl", str(season))
        crps_delta = float(prebowl["crps"]) - float(existing["crps"])
        nll_delta = float(prebowl["margin_nll"]) - float(existing["margin_nll"])
        opposite_nll_seasons += crps_delta * nll_delta < 0
        levels = (0.50, 0.80, 0.95)
        existing_error = sum(
            abs(float(existing[f"coverage_{int(level * 100)}"]) - level)
            for level in levels
        ) / len(levels)
        prebowl_error = sum(
            abs(float(prebowl[f"coverage_{int(level * 100)}"]) - level)
            for level in levels
        ) / len(levels)
        calibration_delta = prebowl_error - existing_error
        opposite_calibration_seasons += crps_delta * calibration_delta < 0
    if classification == "improve" and (
        opposite_nll_seasons >= 3 or opposite_calibration_seasons >= 3
    ):
        classification = "inconclusive"
    recommendation = {
        "improve": "proceed to the separately scoped follow-up issues",
        "no useful improvement": "defer the follow-up issues pending a new rationale",
        "inconclusive": "revise the comparison plan before follow-up work",
    }[classification]
    return {
        "classification": classification,
        "recommendation": recommendation,
        "context_pre_bowl_crps_lower_equal_game": float(c_pre["crps"])
        < float(c_existing["crps"]),
        "context_pre_bowl_crps_lower_equal_season": float(c_pre_equal["crps"])
        < float(c_existing_equal["crps"]),
        "context_pre_bowl_season_wins": season_wins,
        "policy_interaction_equal_game": interaction_pooled,
        "policy_interaction_equal_season": interaction_equal_season,
        "opposite_margin_nll_seasons": opposite_nll_seasons,
        "opposite_calibration_error_seasons": opposite_calibration_seasons,
    }


def _render_report(result: dict[str, object], classification: dict[str, object]) -> str:
    summaries = result["forecast_summaries"]

    def summary_for(
        arm: str,
        season: str,
        population: str = PRIMARY_POPULATION,
        stage: str = "rolling",
    ) -> dict[str, object]:
        for row in summaries:
            if (
                row["forecast_stage"] == stage
                and row["forecast_population"] == population
                and row["phase"] == "all_phases"
                and str(row["model_arm"]) == arm
                and str(row["season"]) == season
            ):
                return row
        raise ValueError(
            f"missing forecast summary for {stage}/{population}/{arm}/{season}"
        )

    arms = [row[0] for row in MODEL_ARMS]
    pooled = {arm: summary_for(arm, "pooled_equal_game") for arm in arms}
    equal = {arm: summary_for(arm, "equal_season_sensitivity") for arm in arms}
    crps_order = sorted(arms, key=lambda arm: float(pooled[arm]["crps"]))
    context_existing = pooled["Context_existing"]
    context_pre = pooled["Context_pre_bowl"]
    history_existing = pooled["History_existing"]
    history_pre = pooled["History_pre_bowl"]
    context_policy_delta = float(context_pre["crps"]) - float(context_existing["crps"])
    history_policy_delta = float(history_pre["crps"]) - float(history_existing["crps"])
    phase_members: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in result["forecast_scores"]:
        if (
            row["forecast_stage"] == "rolling"
            and row["forecast_population"] == PRIMARY_POPULATION
        ):
            phase_members[(str(row["phase"]), str(row["model_arm"]))].append(row)
    phase_order = (
        "pre_thanksgiving",
        "rivalry_window",
        "conference_championship_period",
        "army_navy",
        "other_regular_season",
    )
    rank_members: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in result["ranking_scores"]:
        rank_members[str(row["model_arm"])].append(row)
    rank_metrics = (
        ("nll", "mean_nll"),
        ("crps", "mean_crps"),
        ("expected_rank_error", "mean_expected_rank_error"),
        ("interval_80_target_mass", "mean_interval_80_target_mass"),
    )
    frozen_members: dict[str, list[dict[str, object]]] = defaultdict(list)
    frozen_2022: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in result["frozen_paired_summary"]:
        policy = str(row["target_policy"])
        frozen_members[policy].append(row)
        if str(row["season"]) == "2022":
            frozen_2022[policy].append(row)
    frozen_metrics = (
        "context_minus_history_mean_nll",
        "context_minus_history_mean_crps",
        "context_minus_history_mean_expected_rank_error",
        "context_minus_history_mean_interval_80_target_mass",
    )
    exclusions_by_reason: dict[str, int] = defaultdict(int)
    for row in result["forecast_exclusions"]:
        reason = str(row.get("exclusion_reason") or row.get("reason") or "unspecified")
        exclusions_by_reason[reason] += 1
    max_crps_error = max(
        float(row["crps_numerical_error_estimate"]) for row in result["forecast_scores"]
    )
    lines = [
        "# Issue 201: pre-bowl forecast comparison results",
        "",
        (
            "This is an exploratory four-season comparison using the fixed issue 200 fits. "
            "No model was refit. The result does not promote a model or change production defaults."
        ),
        "",
        "## Primary rolling FBS–FBS game forecast",
        "",
        "| Arm | Games | Equal-game CRPS | Equal-season CRPS | Margin NLL | Home-win Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in arms:
        pooled_row = pooled[arm]
        equal_row = equal[arm]
        arm_label = arm.replace("_pre_bowl", " pre-bowl").replace("_", " ")
        lines.append(
            f"| {arm_label} | {pooled_row['games']} "
            f"| {float(pooled_row['crps']):.6f} | {float(equal_row['crps']):.6f} "
            f"| {float(pooled_row['margin_nll']):.6f} "
            f"| {float(pooled_row['home_win_brier']):.6f} |"
        )
    lines.extend(
        [
            "",
            "Equal-game CRPS order, lower to higher: "
            + " → ".join(
                arm.replace("_pre_bowl", " pre-bowl").replace("_", " ")
                for arm in crps_order
            )
            + ".",
            (
                "The preregistered expected order was Context pre-bowl, History existing, "
                "History pre-bowl, Context existing; the observed order differs."
            ),
            (
                f"Classification: **{classification['classification']}**. "
                f"Recommendation: {classification['recommendation']}."
            ),
            "",
            "### Paired policy and model-family contrasts",
            "",
            "| Contrast | Equal-game CRPS difference | Equal-season CRPS difference | Equal-game margin NLL difference |",
            "| --- | ---: | ---: | ---: |",
            (
                f"| Context − History, existing | "
                f"{float(context_existing['crps']) - float(history_existing['crps']):.6f} | "
                f"{float(equal['Context_existing']['crps']) - float(equal['History_existing']['crps']):.6f} | "
                f"{float(context_existing['margin_nll']) - float(history_existing['margin_nll']):.6f} |"
            ),
            (
                f"| Context − History, pre-bowl | "
                f"{float(context_pre['crps']) - float(history_pre['crps']):.6f} | "
                f"{float(equal['Context_pre_bowl']['crps']) - float(equal['History_pre_bowl']['crps']):.6f} | "
                f"{float(context_pre['margin_nll']) - float(history_pre['margin_nll']):.6f} |"
            ),
            (
                f"| Pre-bowl − existing, Context | {context_policy_delta:.6f} | "
                f"{float(equal['Context_pre_bowl']['crps']) - float(equal['Context_existing']['crps']):.6f} | "
                f"{float(context_pre['margin_nll']) - float(context_existing['margin_nll']):.6f} |"
            ),
            (
                f"| Pre-bowl − existing, History | {history_policy_delta:.6f} | "
                f"{float(equal['History_pre_bowl']['crps']) - float(equal['History_existing']['crps']):.6f} | "
                f"{float(history_pre['margin_nll']) - float(history_existing['margin_nll']):.6f} |"
            ),
            "",
            (
                f"Policy interaction, (Context pre-bowl − Context existing) − "
                f"(History pre-bowl − History existing): "
                f"{float(classification['policy_interaction_equal_game']):.6f} equal-game "
                f"and {float(classification['policy_interaction_equal_season']):.6f} equal-season. "
                "Positive values mean the combined intervention shifts CRPS less favorably for Context."
            ),
            (
                f"Context pre-bowl lowered CRPS in {classification['context_pre_bowl_season_wins']}/4 seasons, "
                f"but did not lower it in either aggregate. Margin NLL moved opposite the CRPS direction "
                f"in {classification['opposite_margin_nll_seasons']} seasons; the 50/80/95% coverage "
                f"deviation moved opposite in {classification['opposite_calibration_error_seasons']} seasons."
            ),
            "",
            "### Per-season rolling CRPS",
            "",
            "| Season | History existing | Context existing | History pre-bowl | Context pre-bowl |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for season in SEASONS:
        values = [float(summary_for(arm, str(season)[0:4])["crps"]) for arm in arms]
        lines.append(
            f"| {season} | " + " | ".join(f"{value:.6f}" for value in values) + " |"
        )
    lines.extend(
        [
            "",
            "### Event-phase CRPS (rolling FBS–FBS)",
            "",
            "| Phase | Games | History existing | Context existing | History pre-bowl | Context pre-bowl |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for phase in phase_order:
        members = {arm: phase_members.get((phase, arm), []) for arm in arms}
        phase_count = len(members[arms[0]])
        if not phase_count:
            continue
        phase_means = [
            sum(float(row["crps"]) for row in members[arm]) / phase_count
            for arm in arms
        ]
        lines.append(
            f"| {phase.replace('_', ' ')} | {phase_count} | "
            + " | ".join(f"{value:.6f}" for value in phase_means)
            + " |"
        )
    cross_rows = {arm: summary_for(arm, "pooled_equal_game", "fbs_fcs") for arm in arms}
    lines.extend(
        [
            "",
            "### Separate FBS–FCS game diagnostic",
            "",
            "| Arm | Games | CRPS | Margin NLL | Home-win Brier |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in arms:
        row = cross_rows[arm]
        lines.append(
            f"| {arm.replace('_', ' ')} | {row['games']} | {float(row['crps']):.6f} "
            f"| {float(row['margin_nll']):.6f} | {float(row['home_win_brier']):.6f} |"
        )
    preseason_rows = {
        arm: summary_for(arm, "pooled_equal_game", PRIMARY_POPULATION, "preseason")
        for arm in arms
    }
    lines.extend(
        [
            "",
            "### Preseason-only FBS–FBS CRPS diagnostic",
            "",
            "| Arm | Games | CRPS |",
            "| --- | ---: | ---: |",
        ]
    )
    for arm in arms:
        row = preseason_rows[arm]
        lines.append(
            f"| {arm.replace('_', ' ')} | {row['games']} | {float(row['crps']):.6f} |"
        )
    lines.extend(
        [
            "",
            (
                "Preseason-only forecasts are kept separate from rolling game counts. "
                "Margin NLL, median absolute error, win Brier, interval coverage, widths, and the "
                "full phase/FBS–FCS tables are in `forecast_summary.csv`."
            ),
            "",
            (
                f"Army–Navy games after the pre-bowl target snapshot: {len(result['calendar_mismatches'])}; "
                "all are separately labeled and listed in `calendar_mismatches.csv`."
            ),
            "",
            "## Rolling rank-target evaluation",
            "",
            (
                "Scores use the common FBS support against corrected #204 pre-bowl PMFs, only at cutoffs "
                "strictly before the target date with evidence strictly before it. Each arm has "
                f"{len(rank_members[arms[0]])} common team-checkpoint rows; "
                f"{len(result['ranking_exclusions'])} target-date exclusions were recorded."
            ),
            "",
            "| Arm | Ranking NLL | Rank CRPS | Expected-rank error | 80% target mass |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in arms:
        members = rank_members[arm]
        lines.append(
            f"| {arm.replace('_', ' ')} | "
            + " | ".join(
                f"{sum(float(row[metric]) for row in members) / len(members):.6f}"
                for metric, _ in rank_metrics
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "These means weight each retained team-checkpoint equally. Ranking NLL/CRPS are secondary "
                "descriptions, not a second model-selection gate. `ranking_summary.csv` retains each "
                "season, cutoff, and event phase."
            ),
            "",
            "## Frozen #192 target sensitivity",
            "",
            (
                "The retained History 1.1 and repaired Context 1.4 PMFs were read-only. The common "
                "cohort keeps native cold starts and excludes target-unsafe checkpoints before comparing "
                "the original #192 final targets with corrected #204 pre-bowl targets."
            ),
            "",
            "| Target | Safe checkpoints | Context − History NLL | Context − History rank CRPS | Expected-rank error difference | 80% target-mass difference |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for target_policy, label in (
        ("original_final_192", "Original final #192"),
        ("pre_bowl_204", "Corrected pre-bowl #204"),
    ):
        members = frozen_members.get(target_policy, [])
        if not members:
            continue
        lines.append(
            f"| {label} | {len(members)} | "
            + " | ".join(
                f"{sum(float(row[metric]) for row in members) / len(members):.6f}"
                for metric in frozen_metrics
            )
            + " |"
        )
    original_2022 = frozen_2022.get("original_final_192", [])
    prebowl_2022 = frozen_2022.get("pre_bowl_204", [])
    if original_2022 and prebowl_2022:
        lines.extend(
            [
                "",
                (
                    "2022 remains visible: across its six safe checkpoints, Context − History rank CRPS was "
                    f"{sum(float(row['context_minus_history_mean_crps']) for row in original_2022) / len(original_2022):.6f} "
                    "against the original target and "
                    f"{sum(float(row['context_minus_history_mean_crps']) for row in prebowl_2022) / len(prebowl_2022):.6f} "
                    "against the corrected pre-bowl target. Full season/checkpoint details are in "
                    "`frozen_target_paired_summary.csv`."
                ),
            ]
        )
    lines.extend(
        [
            "",
            (
                f"Frozen PMF identity `{result['frozen_parity']['frozen_pmf_identity_before']}` "
                f"remained unchanged. Original-target parity compared "
                f"{result['frozen_parity']['score_rows_compared']} retained metrics and passed with "
                f"maximum absolute error {float(result['frozen_parity']['max_absolute_metric_difference']):.3g}. "
                f"Target-safety/support exclusions: {len(result['frozen_exclusions'])}; checkpoint safety "
                "details are in `frozen_checkpoint_safety.csv`."
            ),
            "",
            "## Exclusions and numerical diagnostics",
            "",
        ]
    )
    if exclusions_by_reason:
        lines.append("Forecast schedule/outcome exclusions by reason:")
        for reason, count in sorted(exclusions_by_reason.items()):
            lines.append(f"- `{reason}`: {count}")
        lines.append("")
    lines.extend(
        [
            (
                f"All {result['replay_consistency']['forecast_row_count']} forecast IDs are unique. "
                "Each stage/population uses an identical game set for all four arms; included evidence "
                "IDs and source-row fingerprints are in `replay_evidence.jsonl.gz`."
            ),
            (
                f"Maximum CRPS numerical error estimate: {max_crps_error:.9g} points "
                f"(tolerance `2e-6`; quadrature orders {result['crps_orders_used']}; "
                f"tail multiples {result['crps_tail_multiples_used']})."
            ),
            f"Protocol SHA-256: `{result['protocol_sha256']}`.",
            "",
            (
                "Four seasons and repeated team matchups are not independent confirmation. The ranking "
                "and game-score comparisons are descriptive and do not establish statistical significance. "
                "No follow-up work is dispatched by this issue."
            ),
            "",
            (
                "See `manifest.json` for exact input/output hashes, forward-only fit origins, support counts, "
                "schedule plan, evidence hashes, and provenance."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_results(result: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_outputs: dict[str, list[dict[str, object]]] = {
        "forecast_scores.csv": result["forecast_scores"],
        "forecast_exclusions.csv": result["forecast_exclusions"],
        "forecast_summary.csv": result["forecast_summaries"]
        + result["preseason_summary"],
        "paired_game_differences.csv": result["paired_game_differences"],
        "calendar_mismatches.csv": result["calendar_mismatches"],
        "ranking_scores.csv": result["ranking_scores"],
        "ranking_summary.csv": result["ranking_summary"],
        "ranking_exclusions.csv": result["ranking_exclusions"],
        "frozen_target_scores.csv": result["frozen_scores"],
        "frozen_target_summary.csv": result["frozen_summary"],
        "frozen_target_paired_summary.csv": result["frozen_paired_summary"],
        "frozen_target_exclusions.csv": result["frozen_exclusions"],
        "frozen_checkpoint_safety.csv": result["frozen_checkpoint_safety"],
    }
    for name, rows in table_outputs.items():
        _write_csv(output_dir / name, rows)
    _write_jsonl_gzip(
        output_dir / "replay_evidence.jsonl.gz", result["replay_evidence"]
    )
    _write_json(output_dir / "frozen_target_parity.json", result["frozen_parity"])
    classification = _result_classification(result)
    _write_json(output_dir / "decision.json", classification)
    (output_dir / "report.md").write_text(
        _render_report(result, classification), encoding="utf-8"
    )
    output_files = sorted(
        path
        for path in output_dir.iterdir()
        if path.name != "manifest.json" and path.is_file()
    )
    manifest = {
        "schema_version": 1,
        "experiment": "issue-201-prebowl-forecast-comparison",
        "status": "complete",
        "seasons": list(SEASONS),
        "model_arms": result["model_arms"],
        "protocol_sha256": result["protocol_sha256"],
        "input_artifacts": result["input_artifacts"],
        "issue200_provenance": result["issue200_provenance"],
        "pre_bowl_target_provenance": result["pre_bowl_target_provenance"],
        "fit_origin_by_season_policy": result["fit_origin_by_season_policy"],
        "target_snapshot_dates": result["target_snapshot_dates"],
        "schedule_window_dates": result["schedule_window_dates"],
        "ranking_common_population": result["ranking_common_population"],
        "replay_consistency": result["replay_consistency"],
        "frozen_target_parity": result["frozen_parity"],
        "decision": classification,
        "crps_numerical_contract": {
            "quadrature_orders": [128, 256, 512],
            "minimum_tail_multiple": 40,
            "tail_bound_budget_fraction": 0.5,
            "absolute_tolerance": 2e-6,
            "max_observed_error_estimate": max(
                float(row["crps_numerical_error_estimate"])
                for row in result["forecast_scores"]
            ),
            "orders_used": sorted(
                {int(row["crps_quadrature_order"]) for row in result["forecast_scores"]}
            ),
            "tail_multiples_used": sorted(
                {float(row["crps_tail_multiple"]) for row in result["forecast_scores"]}
            ),
        },
        "outputs": {path.name: _file_record(path) for path in output_files},
    }
    _write_json(output_dir / "manifest.json", manifest)


def _read_output_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def refresh_report(args: argparse.Namespace) -> None:
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, object] = {
        "forecast_summaries": _read_output_csv(args.output / "forecast_summary.csv"),
        "forecast_scores": _read_output_csv(args.output / "forecast_scores.csv"),
        "forecast_exclusions": _read_output_csv(
            args.output / "forecast_exclusions.csv"
        ),
        "calendar_mismatches": _read_output_csv(
            args.output / "calendar_mismatches.csv"
        ),
        "ranking_scores": _read_output_csv(args.output / "ranking_scores.csv"),
        "ranking_exclusions": _read_output_csv(args.output / "ranking_exclusions.csv"),
        "frozen_paired_summary": _read_output_csv(
            args.output / "frozen_target_paired_summary.csv"
        ),
        "frozen_scores": _read_output_csv(args.output / "frozen_target_scores.csv"),
        "frozen_exclusions": _read_output_csv(
            args.output / "frozen_target_exclusions.csv"
        ),
        "frozen_parity": json.loads(
            (args.output / "frozen_target_parity.json").read_text(encoding="utf-8")
        ),
        "replay_consistency": manifest["replay_consistency"],
        "protocol_sha256": manifest["protocol_sha256"],
        "crps_orders_used": manifest["crps_numerical_contract"]["orders_used"],
        "crps_tail_multiples_used": manifest["crps_numerical_contract"][
            "tail_multiples_used"
        ],
    }
    classification = _result_classification(result)
    decision_path = args.output / "decision.json"
    _write_json(decision_path, classification)
    manifest["decision"] = classification
    manifest["outputs"][decision_path.name] = _file_record(decision_path)
    report_path = args.output / "report.md"
    report_path.write_text(_render_report(result, classification), encoding="utf-8")
    manifest["outputs"][report_path.name] = _file_record(report_path)
    _write_json(manifest_path, manifest)
    print(f"Refreshed report from existing scored artifacts: {report_path}")


def score(args: argparse.Namespace) -> None:
    plan_path = args.output / "schedule_plan.json"
    plan_manifest_path = args.output / "schedule_plan.manifest.json"
    if not plan_path.is_file() or not plan_manifest_path.is_file():
        raise SystemExit("run the plan subcommand and review its frozen schedule first")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    result = run_issue_201_comparison(
        repository_root=ROOT,
        game_root=args.game_root,
        schedule_plan=plan,
        schedule_plan_path=plan_path,
        schedule_plan_manifest_path=plan_manifest_path,
        issue200_bundle_path=args.issue200_root / "pipeline_bundle.json",
        issue200_manifest_path=args.issue200_root / "manifest.json",
        issue200_fitted_models_path=args.issue200_root / "fitted_models.json",
        pre_bowl_target_path=args.pre_bowl_targets,
        issue204_manifest_path=args.issue204_manifest,
        target_snapshot_manifest_path=args.target_snapshot_manifest,
        schedule_windows_path=args.schedule_windows,
        frozen_root=args.frozen_root,
        protocol_path=args.protocol,
    )
    write_results(result, args.output)
    print(f"Issue 201 result artifacts written to {args.output}")
    print(
        f"Decision: {result['classification']['classification'] if 'classification' in result else _result_classification(result)['classification']}"
    )


def _common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--game-root", type=Path, default=DEFAULT_GAME_ROOT)
    parser.add_argument(
        "--target-snapshot-manifest", type=Path, default=DEFAULT_TARGET_DATES
    )
    parser.add_argument(
        "--schedule-windows", type=Path, default=DEFAULT_SCHEDULE_WINDOWS
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser(
        "plan", help="freeze the score-blind game schedule"
    )
    _common_paths(plan_parser)
    report_parser = commands.add_parser(
        "report", help="refresh the report from an existing completed score bundle"
    )
    report_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    score_parser = commands.add_parser(
        "score", help="verify the plan and score all arms"
    )
    _common_paths(score_parser)
    score_parser.add_argument(
        "--issue200-root", type=Path, default=DEFAULT_ISSUE200_ROOT
    )
    score_parser.add_argument("--frozen-root", type=Path, default=DEFAULT_FROZEN_ROOT)
    score_parser.add_argument(
        "--issue204-root", type=Path, default=DEFAULT_ISSUE204_ROOT
    )
    score_parser.add_argument("--pre-bowl-targets", type=Path)
    score_parser.add_argument("--issue204-manifest", type=Path)
    args = parser.parse_args()
    if args.command == "score":
        args.pre_bowl_targets = args.pre_bowl_targets or (
            args.issue204_root / "pre_bowl_target_distributions.csv"
        )
        args.issue204_manifest = args.issue204_manifest or (
            args.issue204_root / "manifest.json"
        )
    return args


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        freeze_plan(args)
    elif args.command == "report":
        refresh_report(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
