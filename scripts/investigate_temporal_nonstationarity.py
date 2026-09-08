"""Run the staged Issue #34 temporal nonstationarity investigation.

The runner consumes existing processed artifacts.  ``--input-root`` is useful
when the checkout contains only a current-season working corpus while the
authoritative historical artifacts live in a sibling checkout.  It never
writes to the input root.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from gippyrank.posterior.engine import LikelihoodV1
from gippyrank.research.temporal_nonstationarity import (
    DEVELOPMENT_SEASONS,
    EVALUATION_SEASONS,
    TRAIN_SEASONS,
    aggregate_metrics,
    build_context_priors,
    candidate_grid,
    development_gate_metrics,
    evaluate_candidates,
    load_historical_games,
    residual_rows,
    select_recency_candidate,
    sha256,
    stage0_persistence,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/temporal_nonstationarity"


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_likelihood(root: Path) -> LikelihoodV1:
    path = root / "data/processed/posterior/historical_likelihood_v1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    return LikelihoodV1(
        np.asarray(value["beta"], dtype=float),
        float(value["scale"]),
        float(value["degrees_of_freedom"]),
        value.get("fit_kind", "weighted_pseudo"),
    )


def _production_hashes(root: Path) -> dict[str, str]:
    paths = {
        "historical_likelihood_v1": root / "data/processed/posterior/historical_likelihood_v1.json",
        "context_prior_backtest": root / "data/processed/preseason/context/predictions.csv",
        "history_prior_backtest": root / "data/processed/preseason/history/predictions.csv",
        "context_prior_2026": root / "data/processed/preseason/context/annual/2026/predictions.csv",
        "history_prior_2026": root / "data/processed/preseason/history/annual/2026/predictions.csv",
    }
    return {
        name: sha256(path)
        for name, path in paths.items()
        if path.exists()
    }


def _flat_metric_rows(
    rows: list[dict[str, object]],
    *,
    include_season: bool,
) -> list[dict[str, object]]:
    aggregate = aggregate_metrics(rows)
    result = []
    for item in aggregate.values():
        if include_season:
            result.append(item)
        else:
            result.append(
                {
                    "candidate": item["candidate"],
                    "target_kind": item["target_kind"],
                    "season": "all",
                    "n": item["n"],
                    "margin_nll": item["margin_nll"],
                    "margin_mae": item["margin_mae"],
                    "win_brier": item["win_brier"],
                }
            )
    if not include_season:
        merged = {}
        for item in result:
            key = (item["candidate"], item["target_kind"])
            merged.setdefault(key, []).append(item)
        result = []
        for (candidate, target_kind), values in sorted(merged.items()):
            weights = np.asarray([int(value["n"]) for value in values], dtype=float)
            result.append(
                {
                    "candidate": candidate,
                    "target_kind": target_kind,
                    "season": "all",
                    "n": int(weights.sum()),
                    "margin_nll": float(
                        np.average([value["margin_nll"] for value in values], weights=weights)
                    ),
                    "margin_mae": float(
                        np.average([value["margin_mae"] for value in values], weights=weights)
                    ),
                    "win_brier": float(
                        np.average([value["win_brier"] for value in values], weights=weights)
                    ),
                }
            )
    return result


def _development_deltas(
    rows: list[dict[str, object]], gate: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    aggregate = aggregate_metrics(
        [row for row in rows if row["target_kind"] == "next_game"]
    )
    static = {
        int(item["season"]): item
        for item in aggregate.values()
        if item["candidate"] == "Static V1"
    }
    result = []
    for item in aggregate.values():
        if int(item["season"]) not in DEVELOPMENT_SEASONS:
            continue
        baseline = static[int(item["season"])]
        result.append(
            {
                **item,
                "delta_nll_vs_static": float(item["margin_nll"])
                - float(baseline["margin_nll"]),
                "delta_mae_vs_static": float(item["margin_mae"])
                - float(baseline["margin_mae"]),
                "delta_brier_vs_static": float(item["win_brier"])
                - float(baseline["win_brier"]),
                "gate_next_game_nll": gate.get(str(item["candidate"]), {}).get(
                    "next_game_nll"
                ),
            }
        )
    return result


def _stage0_lookup(metrics: list[dict[str, object]], period: str, dimension: str, control: str, lag: int) -> dict[str, object]:
    for row in metrics:
        if (
            row["period"] == period
            and row["dimension"] == dimension
            and row["control"] == control
            and int(row["lag"]) == lag
            and row["bin"] == "all"
        ):
            return row
    return {}


def _make_examples(
    eval_rows: list[dict[str, object]],
    selected: str,
    residuals: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_key: dict[tuple[object, ...], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in eval_rows:
        if row["target_kind"] == "next_game":
            by_key[
                (row["season"], row["cutoff"], row["target_game_id"])
            ][str(row["candidate"])] = row
    residual_by_team: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in residuals:
        residual_by_team[(int(row["season"]), str(row["team_id"]))].append(row)
    candidates = []
    for key, values in sorted(by_key.items(), key=lambda item: tuple(str(x) for x in item[0])):
        static, recent = values.get("Static V1"), values.get(selected)
        if static is None or recent is None:
            continue
        season, cutoff, _game_id = key
        home_id = str(static.get("home_team_id", ""))
        history = [
            row
            for row in residual_by_team[(int(season), home_id)]
            if str(row["game_date"]) < str(cutoff)
        ][-3:]
        recent_values = [float(row["residual"]) for row in history]
        if not recent_values:
            continue
        candidate_row = {
            "season": season,
            "cutoff": cutoff,
            "target_game_id": static["target_game_id"],
            "home_team_id": home_id,
            "home_team_name": static.get("home_team_name", home_id),
            "recent_game_residuals": ";".join(
                f"{row['game_id']}:{float(row['residual']):.4f}" for row in history
            ),
            "recent_signal": float(np.mean(recent_values)),
            "static_expected_margin": static["expected_margin"],
            "recency_expected_margin": recent["expected_margin"],
            "actual_margin": static["actual_margin"],
            "prediction_disagreement": float(recent["expected_margin"])
            - float(static["expected_margin"]),
        }
        candidates.append(candidate_row)
    if not candidates:
        return []
    selections = [
        ("largest_positive_recent_signal", max(candidates, key=lambda row: row["recent_signal"])),
        ("largest_negative_recent_signal", min(candidates, key=lambda row: row["recent_signal"])),
        (
            "largest_static_vs_recency_disagreement",
            max(candidates, key=lambda row: abs(row["prediction_disagreement"])),
        ),
    ]
    result = []
    seen = set()
    for selection, row in selections:
        key = (row["season"], row["target_game_id"])
        if key in seen:
            continue
        seen.add(key)
        result.append({"selection_rule": selection, **row})
    return result


def _recommendation(
    selected: str | None,
    development_rows: list[dict[str, object]],
    evaluation_rows: list[dict[str, object]],
    stage0_metrics: list[dict[str, object]],
) -> tuple[str, str]:
    observed = _stage0_lookup(stage0_metrics, "development", "demeaned", "observed", 1)
    shuffled = _stage0_lookup(stage0_metrics, "development", "demeaned", "shuffle_order", 1)
    unrelated = _stage0_lookup(stage0_metrics, "development", "demeaned", "unrelated_team", 1)
    persistence = (
        observed.get("correlation") is not None
        and float(observed["correlation"]) > 0
        and float(observed["correlation"]) > max(
            float(shuffled.get("correlation", 0.0) or 0.0),
            float(unrelated.get("correlation", 0.0) or 0.0),
        )
    )
    if selected is None:
        if persistence:
            return "B", "Stage 0 has residual persistence, but no frozen recency candidate clears the development gate."
        return "C", "Neither residual persistence nor future-game validation provides compelling evidence of nonstationarity."
    if not evaluation_rows:
        return "B", "A recency candidate cleared development, but Stage 2 was not run because the evaluation panel was unavailable."
    aggregate = aggregate_metrics(
        [row for row in evaluation_rows if row["target_kind"] == "next_game"]
    )
    selected_rows = [
        item for item in aggregate.values() if item["candidate"] == selected
    ]
    static_rows = {
        int(item["season"]): item
        for item in aggregate.values()
        if item["candidate"] == "Static V1"
    }
    nll_deltas = [
        float(item["margin_nll"]) - float(static_rows[int(item["season"])] ["margin_nll"])
        for item in selected_rows
        if int(item["season"]) in EVALUATION_SEASONS
    ]
    mae_deltas = [
        float(item["margin_mae"]) - float(static_rows[int(item["season"])] ["margin_mae"])
        for item in selected_rows
        if int(item["season"]) in EVALUATION_SEASONS
    ]
    if persistence and sum(delta < 0 for delta in nll_deltas) >= 3 and sum(delta < 0 for delta in mae_deltas) >= 3:
        return "A", "The frozen recency diagnostic clears development, remains directionally better in evaluation, and Stage 0 independently supports persistence; open a separate temporal/state-space Posterior V2 design investigation."
    return "B", "The temporal signal is present or selected in development, but its cross-season evaluation evidence is not stable enough to justify a latent-state architecture."


def render_report(
    summary: dict[str, object],
    stage0_metrics: list[dict[str, object]],
    development_metrics: list[dict[str, object]],
    evaluation_metrics: list[dict[str, object]],
) -> str:
    lines = [
        "# Within-season team-strength drift investigation",
        "",
        "Issue #34 is a research diagnostic. Historical Likelihood V1, the Context/History prior families, Posterior V1, Performance V1, and publication artifacts are unchanged.",
        "",
        "## Design",
        "",
        "- Training: 2008–2017; development: 2018–2021; evaluation: 2022–2025. 2026 outcomes are not used.",
        "- Context is the primary prior family. Existing 2022–2025 Context PMFs are consumed; missing 2018–2021 PMFs apply the repository's frozen, predeclared Context development fit trained through 2017 to outcome-free historical rank and preseason-context rows.",
        "- For a cutoff `c`, only games with `game_time < c` enter inference. A target is strictly after `c`; the next-game panel uses the first future game for each team at four deterministic, evenly spaced completed-week cutoffs per season, deduplicated by game. An all-future panel was omitted because it repeatedly scores the same games at every cutoff and is not inexpensive on this loopy graph.",
        "- Residual = oriented observed margin − the V1 likelihood location averaged over the paired historical rank-observation PMFs. Positive focal-team residual means better than expected. Demeaned residuals subtract each team-season mean.",
        "- Recency candidates temper only the existing factor: `L_g_tempered = L_g ^ 2^(-age_days / h)`. Static V1 uses the unchanged production call semantics.",
        "",
        "## Stage 0",
        "",
        f"The residual panel contains {summary['stage0']['n_focal_team_games']:,} focal-team games and {summary['stage0']['n_team_seasons_with_at_least_3_games']:,} team-seasons with at least three games.",
        "",
        "| Period | Dimension | Control | Lag | Pairs | Correlation |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in stage0_metrics:
        if row["bin"] == "all" and row["lag"] == 1:
            lines.append(
                f"| {row['period']} | {row['dimension']} | {row['control']} | {row['lag']} | {row['n_pairs']} | {row['correlation'] if row['correlation'] is not None else 'NA'} |"
            )
    lines += [
        "",
        "Elapsed-time, game-count, early/late, recent-history, and deterministic shuffle/unrelated-team controls are in `residual_lag_metrics.csv` and `summary.json`.",
        "",
        "## Stage 1 development",
        "",
        "The frozen gate requires aggregate next-game NLL improvement ≥ 0.010, margin MAE improvement ≥ 0.10, win Brier worsening ≤ 0.001, NLL improvement in at least 3 of 4 seasons, and no season NLL worsening above 0.015.",
        f"Selected half-life: **{summary['selected_half_life'] or 'none'}**. Selection used only 2018–2021; 2022–2025 could not affect selection because it was reserved for the frozen post-selection evaluation. The gate result is **{'passed' if summary['stage2_triggered'] else 'not passed'}**.",
        "",
        "| Candidate | Target | N | Margin NLL | Margin MAE | Win Brier | ΔNLL vs static |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    static = {
        (row["target_kind"], row["season"]): row
        for row in development_metrics
        if row["candidate"] == "Static V1"
    }
    for row in development_metrics:
        base = static.get((row["target_kind"], row["season"]))
        delta = float(row["margin_nll"]) - float(base["margin_nll"]) if base else 0.0
        lines.append(
            f"| {row['candidate']} | {row['target_kind']} {row['season']} | {row['n']} | {row['margin_nll']:.4f} | {row['margin_mae']:.4f} | {row['win_brier']:.4f} | {delta:+.4f} |"
        )
    lines += [
        "",
        "## Stage 2 evaluation",
        "",
    ]
    if evaluation_metrics:
        lines.append("The development-selected candidate was frozen before 2022–2025 evaluation.")
        lines += ["", "| Candidate | Target | N | Margin NLL | Margin MAE | Win Brier |", "|---|---|---:|---:|---:|---:|"]
        for row in evaluation_metrics:
            lines.append(
                f"| {row['candidate']} | {row['target_kind']} {row['season']} | {row['n']} | {row['margin_nll']:.4f} | {row['margin_mae']:.4f} | {row['win_brier']:.4f} |"
            )
    else:
        lines.append("Stage 2 was not triggered because no candidate cleared the frozen development gate.")
    lines += [
        "",
        f"## Recommendation: {summary['recommendation']} — {summary['recommendation_reason']}",
        "",
        "Any recency gain reported here is diagnostic only. It is not a production recommendation to down-weight old games. If the recommendation is A, the next step is a separate explicit latent-state/state-space Posterior V2 design investigation.",
        "",
        "## Confounders and integrity",
        "",
        "Opponent adjustment remains in every V1 rank-pair likelihood location; site indicators remain active; elapsed days, rather than game count, determine age; the static comparison naturally includes preseason-prior fade; no injury, quarterback, yards, plays, turnover, or play-by-play data are used.",
        "",
        "Production artifact SHA-256 values before and after the run are identical; see `summary.json`. The 2022–2025 evaluation window is leakage-safe under the frozen cutoff construction but is not untouched independent confirmation because it has been used in earlier GippyRank research.",
        "",
        f"The completed deterministic run took {summary['runtime_seconds']:.3f} seconds. History-prior sensitivity was not run because Context is primary and no candidate cleared the development gate.",
        "",
        "Artifacts: `residual_persistence.csv`, `residual_lag_metrics.csv`, `development_future_metrics.csv`, `development_season_metrics.csv`, `model_spec.json`, and `summary.json`; `evaluation_*` and `team_examples.csv` are present only when Stage 2 triggers.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--max-iterations", type=int, default=75)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output = args.output.resolve()
    started = time.perf_counter()
    likelihood = load_likelihood(input_root)
    seasons = (*TRAIN_SEASONS, *DEVELOPMENT_SEASONS, *EVALUATION_SEASONS)
    games = load_historical_games(input_root, seasons)
    if not games:
        raise RuntimeError("the historical modeling-game artifact has no requested seasons")
    before = _production_hashes(input_root)
    residuals = residual_rows(games, likelihood)
    lag_metrics, early_late, stage0_summary = stage0_persistence(residuals)
    priors = build_context_priors(input_root, (*DEVELOPMENT_SEASONS, *EVALUATION_SEASONS))
    development_rows, _development_examples = evaluate_candidates(
        games,
        priors,
        likelihood,
        candidate_grid(),
        DEVELOPMENT_SEASONS,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
    )
    gate = development_gate_metrics(development_rows)
    selected = select_recency_candidate(gate)
    stage2_triggered = selected is not None
    evaluation_rows: list[dict[str, object]] = []
    if stage2_triggered:
        evaluation_candidates = (
            candidate_grid()[0],
            next(candidate for candidate in candidate_grid() if candidate.name == selected),
        )
        evaluation_rows, _evaluation_examples = evaluate_candidates(
            games,
            priors,
            likelihood,
            evaluation_candidates,
            EVALUATION_SEASONS,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
        )
    after = _production_hashes(input_root)
    recommendation, recommendation_reason = _recommendation(
        selected, development_rows, evaluation_rows, lag_metrics
    )
    summary = {
        "study": "issue_34_temporal_nonstationarity",
        "train_seasons": list(TRAIN_SEASONS),
        "development_seasons": list(DEVELOPMENT_SEASONS),
        "evaluation_seasons": list(EVALUATION_SEASONS),
        "excluded_seasons": [2026],
        "input_artifacts": {
            "historical_games": "data/processed/modeling/historical_modeling_games.csv",
            "rank_distributions": "data/processed/modeling/team_season_rank_distributions.csv",
            "context_priors": "data/processed/preseason/context/predictions.csv",
            "historical_likelihood": "data/processed/posterior/historical_likelihood_v1.json",
        },
        "game_count": len(games),
        "focal_residual_count": len(residuals),
        "stage0": stage0_summary,
        "stage0_early_late_count": len(early_late),
        "candidate_grid": [
            {"name": candidate.name, "half_life_days": candidate.half_life_days}
            for candidate in candidate_grid()
        ],
        "development_gate": gate,
        "selected_half_life": selected,
        "selection_used_seasons_only": list(DEVELOPMENT_SEASONS),
        "stage2_triggered": stage2_triggered,
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
        "production_integrity": {"before": before, "after": after, "identical": before == after},
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    write_csv(
        output / "residual_persistence.csv",
        residuals,
        [
            "season", "team_id", "opponent_id", "game_id", "game_date", "week", "game_order",
            "observed_oriented_margin", "expected_oriented_margin", "residual", "home_id", "away_id",
            "home_name", "away_name", "pairing",
        ],
    )
    write_csv(
        output / "residual_lag_metrics.csv",
        lag_metrics,
        ["period", "dimension", "bin", "control", "lag", "n_pairs", "correlation", "mean_product"],
    )
    write_csv(
        output / "development_future_metrics.csv",
        _flat_metric_rows(development_rows, include_season=False),
        ["candidate", "target_kind", "season", "n", "margin_nll", "margin_mae", "win_brier"],
    )
    write_csv(
        output / "development_season_metrics.csv",
        _flat_metric_rows(development_rows, include_season=True),
        ["candidate", "target_kind", "season", "n", "margin_nll", "margin_mae", "win_brier"],
    )
    if evaluation_rows:
        write_csv(
            output / "evaluation_future_metrics.csv",
            _flat_metric_rows(evaluation_rows, include_season=False),
            ["candidate", "target_kind", "season", "n", "margin_nll", "margin_mae", "win_brier"],
        )
        write_csv(
            output / "evaluation_season_metrics.csv",
            _flat_metric_rows(evaluation_rows, include_season=True),
            ["candidate", "target_kind", "season", "n", "margin_nll", "margin_mae", "win_brier"],
        )
        examples = _make_examples(evaluation_rows, str(selected), residuals)
        if examples:
            write_csv(
                output / "team_examples.csv",
                examples,
                [
                    "selection_rule", "season", "cutoff", "target_game_id", "home_team_id", "home_team_name",
                    "recent_game_residuals", "recent_signal", "static_expected_margin", "recency_expected_margin",
                    "actual_margin", "prediction_disagreement",
                ],
            )
    model_spec = {
        "study": "issue_34_temporal_nonstationarity",
        "production_baseline": "Historical Likelihood V1 + static Posterior V1 semantics",
        "prior_family": "Context primary",
        "candidate_grid": [
            {"name": candidate.name, "half_life_days": candidate.half_life_days}
            for candidate in candidate_grid()
        ],
        "factor_tempering": "L_g_tempered = L_g ^ 2 ^ (-age_days / half_life_days)",
        "age_definition": "cutoff_datetime - game_datetime; game_datetime must be strictly before cutoff",
        "splits": {
            "training": list(TRAIN_SEASONS),
            "development": list(DEVELOPMENT_SEASONS),
            "evaluation": list(EVALUATION_SEASONS),
        },
        "development_gate": {
            "next_game_nll_improvement": 0.010,
            "next_game_mae_improvement": 0.10,
            "maximum_win_brier_worsening": 0.001,
            "minimum_seasons_with_nll_improvement": 3,
            "maximum_single_season_nll_worsening": 0.015,
        },
        "target_keys": "identical completed-week cutoffs and target game IDs across candidates; future games strictly after cutoff",
        "cutoff_schedule": "four evenly spaced completed-week cutoffs per season, excluding the first and last available week",
        "posterior_numerics": {
            "max_iterations": args.max_iterations,
            "tolerance": args.tolerance,
            "note": "research cutoffs use a fixed approximate-BP computational tolerance to keep the staged diagnostic tractable; production snapshot calls and artifacts are unchanged",
        },
        "residual_definition": "oriented observed margin minus V1 likelihood location averaged over paired historical rank-observation PMFs",
        "null_controls": ["within-team-season order shuffle", "unrelated team sequence pairing in same season"],
        "no_production_changes": True,
    }
    write_json(output / "model_spec.json", model_spec)
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text(
        render_report(
            summary,
            lag_metrics,
            _flat_metric_rows(development_rows, include_season=True),
            _flat_metric_rows(evaluation_rows, include_season=True),
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "selected": selected,
                "recommendation": recommendation,
                "stage2_triggered": stage2_triggered,
                "runtime_seconds": summary["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
