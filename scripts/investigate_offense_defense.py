"""Run the staged Issue #40 offense/defense latent-state investigation.

The historical modeling corpus is often kept in a sibling checkout because it
is a large research input.  Use ``--input-root`` for that checkout and keep
``--output-root`` pointed at the current checkout.  The runner never writes to
the input root and never calls production snapshot/publication builders.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from gippyrank.research.offense_defense import (
    DEVELOPMENT_SEASONS,
    EVALUATION_SEASONS,
    TRAIN_SEASONS,
    aggregate_future_metrics,
    aggregate_phase_metrics,
    build_score_environment,
    candidate_grid,
    development_gate,
    evaluate_candidates,
    leakage_safe_residual_rows,
    load_likelihood,
    production_hashes,
    select_candidate,
    stage0_component_persistence,
    write_csv,
    write_json,
)
from gippyrank.research.temporal_nonstationarity import (
    build_context_priors,
    load_historical_games,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_RELATIVE = Path("data/processed/offense_defense_latent")
STAGE1_ARTIFACTS = (
    "development_future_metrics.csv",
    "development_season_metrics.csv",
    "development_phase_metrics.csv",
    "development_component_diagnostics.csv",
    "evaluation_future_metrics.csv",
    "evaluation_season_metrics.csv",
    "evaluation_phase_metrics.csv",
    "evaluation_component_diagnostics.csv",
    "team_profiles.csv",
    "representative_examples.csv",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--null-permutations", type=int, default=10)
    return parser.parse_args()


def _write_metric_tables(
    output: Path, rows: list[dict[str, object]], diagnostics: list[dict[str, object]]
) -> None:
    write_csv(output / "development_future_metrics.csv", rows)
    write_csv(output / "development_season_metrics.csv", aggregate_future_metrics(rows))
    write_csv(output / "development_phase_metrics.csv", aggregate_phase_metrics(rows))
    write_csv(output / "development_component_diagnostics.csv", diagnostics)


def _clear_stage1_artifacts(output: Path) -> None:
    for name in STAGE1_ARTIFACTS:
        (output / name).unlink(missing_ok=True)


def _stage0_signal(
    metrics: list[dict[str, object]], *, period: str
) -> dict[str, object]:
    """Summarize one period relative to shuffled and unrelated nulls."""

    relations = (
        "offense->offense",
        "defense->defense",
        "offense->defense",
        "defense->offense",
    )
    lookup: dict[tuple[str, str], float | None] = {}
    for row in metrics:
        if (
            row["period"] == period
            and row["scale"] == "demeaned"
            and row["lag"] == 1
            and row["season"] == "all"
            and row["bin"] == "all"
        ):
            lookup[(str(row["relation"]), str(row["control"]))] = (
                None if row["correlation"] is None else float(row["correlation"])
            )

    relation_effects: dict[str, dict[str, float | None]] = {}
    for relation in relations:
        observed = lookup.get((relation, "observed"))
        shuffled = lookup.get((relation, "shuffle_order"))
        unrelated = lookup.get((relation, "unrelated_team"))
        relation_effects[relation] = {
            "observed": observed,
            "shuffle_order_null": shuffled,
            "unrelated_team_null": unrelated,
            "observed_minus_shuffle_order": (
                None if observed is None or shuffled is None else observed - shuffled
            ),
            "observed_minus_unrelated_team": (
                None if observed is None or unrelated is None else observed - unrelated
            ),
        }

    same_relations = relations[:2]
    cross_relations = relations[2:]

    def _mean_effect(key: str, selected: tuple[str, ...]) -> float | None:
        values = [
            float(relation_effects[relation][key])
            for relation in selected
            if relation_effects[relation][key] is not None
        ]
        return float(np.mean(values)) if values else None

    same_mean = _mean_effect("observed", same_relations)
    cross_mean = _mean_effect("observed", cross_relations)
    same_shuffle_excess = _mean_effect("observed_minus_shuffle_order", same_relations)
    cross_shuffle_excess = _mean_effect("observed_minus_shuffle_order", cross_relations)
    same_unrelated_excess = _mean_effect(
        "observed_minus_unrelated_team", same_relations
    )
    same_shuffle_effects = [
        relation_effects[relation]["observed_minus_shuffle_order"]
        for relation in same_relations
    ]
    return {
        "period": period,
        "same_component_mean_lag1": same_mean,
        "cross_component_mean_lag1": cross_mean,
        "same_minus_cross": None
        if same_mean is None or cross_mean is None
        else same_mean - cross_mean,
        "same_minus_shuffle_order_null": same_shuffle_excess,
        "cross_minus_shuffle_order_null": cross_shuffle_excess,
        "same_minus_cross_excess": (
            None
            if same_shuffle_excess is None or cross_shuffle_excess is None
            else same_shuffle_excess - cross_shuffle_excess
        ),
        "same_minus_unrelated_team_null": same_unrelated_excess,
        "relation_effects": relation_effects,
        "supports_component_persistence": bool(
            same_shuffle_excess is not None
            and cross_shuffle_excess is not None
            and all(
                value is not None and float(value) > 0.01
                for value in same_shuffle_effects
            )
            and same_shuffle_excess > cross_shuffle_excess + 0.01
        ),
    }


def _representative_examples(
    evaluation_rows: list[dict[str, object]],
    selected: str,
) -> list[dict[str, object]]:
    by_key: dict[tuple[object, ...], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in evaluation_rows:
        by_key[(row["season"], row["cutoff"], row["target_game_id"])][
            str(row["candidate"])
        ] = row
    candidates = []
    for key, values in by_key.items():
        baseline = values.get("Posterior V1")
        od = values.get(selected)
        if baseline is None or od is None:
            continue
        candidates.append(
            {
                "season": key[0],
                "cutoff": key[1],
                "target_game_id": key[2],
                "home_team_id": baseline["home_team_id"],
                "away_team_id": baseline["away_team_id"],
                "actual_margin": baseline["actual_margin"],
                "v1_expected_margin": baseline["expected_margin"],
                "od_expected_margin": od["expected_margin"],
                "prediction_disagreement": float(od["expected_margin"])
                - float(baseline["expected_margin"]),
            }
        )
    if not candidates:
        return []
    rules = (
        (
            "largest_positive_od_disagreement",
            max,
            lambda row: row["prediction_disagreement"],
        ),
        (
            "largest_negative_od_disagreement",
            min,
            lambda row: row["prediction_disagreement"],
        ),
        (
            "largest_absolute_od_disagreement",
            max,
            lambda row: abs(row["prediction_disagreement"]),
        ),
    )
    result = []
    seen: set[tuple[object, ...]] = set()
    for rule, chooser, key_function in rules:
        row = chooser(candidates, key=key_function)
        identity = (row["season"], row["target_game_id"])
        if identity in seen:
            continue
        seen.add(identity)
        result.append({"selection_rule": rule, **row})
    return result


def _recommendation(
    selected: str | None,
    stage0_signal: dict[str, object],
    evaluation_rows: list[dict[str, object]],
) -> tuple[str, str]:
    if selected is None:
        if stage0_signal["supports_component_persistence"]:
            return (
                "B",
                "Same-component residual ordering exceeds cross-component ordering, but no frozen OD candidate clears the development gate.",
            )
        return (
            "C",
            "Stage 0 does not show convincing distinct component persistence; Stage 1 was not run.",
        )
    if not evaluation_rows:
        return (
            "B",
            "An OD candidate clears development, but historical evaluation was unavailable.",
        )
    aggregate = aggregate_future_metrics(evaluation_rows)
    selected_rows = [row for row in aggregate if row["candidate"] == selected]
    baseline = {
        int(row["season"]): row
        for row in aggregate
        if row["candidate"] == "Posterior V1"
    }
    deltas = [
        float(row["margin_nll"]) - float(baseline[int(row["season"])]["margin_nll"])
        for row in selected_rows
        if int(row["season"]) in baseline
    ]
    if (
        stage0_signal["supports_component_persistence"]
        and sum(delta < 0 for delta in deltas) >= 3
    ):
        return (
            "A",
            "The selected OD candidate clears development and improves future-game NLL in at least three evaluation seasons.",
        )
    return (
        "B",
        "The selected OD candidate clears development, but evaluation evidence is not stable enough for a production-design recommendation.",
    )


def _render_report(
    summary: dict[str, object],
    stage0: list[dict[str, object]],
    dev_metrics: list[dict[str, object]],
    eval_metrics: list[dict[str, object]],
) -> str:
    def _display(value: object) -> str:
        return "NA" if value is None else f"{float(value):+.6f}"

    lines = [
        "# Offense/defense latent-state investigation",
        "",
        "Issue #40 is research-only. Historical Likelihood V1, Context/History priors, Posterior V1, Performance V1, weekly publication, and published artifacts were not modified.",
        "",
        "## Frozen design",
        "",
        "- Training: 2008–2017; development: 2018–2021; evaluation: 2022–2025. 2026 outcomes are excluded.",
        "- Context is primary. Existing Context priors are consumed; missing development priors use the repository's frozen Context fit trained through 2017.",
        "- Stage 0 expected margins use only outcome-free preseason Context PMFs (or uniform cold-start PMFs); historical `rank_pairs` are deliberately excluded because they are end-of-season observations. Stage 1 uses the existing pregame Posterior V1 construction.",
        "- A training-only pairing/site total environment gives expected scores `(T + margin) / 2` and `(T - margin) / 2`, with FBS-first orientation for cross-subdivision games.",
        "- Offensive residual is points scored minus expected points scored. Defensive residual is expected opponent points minus opponent points allowed. Both are reported for both teams in every game.",
        "- OD score means are `environment + O_team - D_opponent`; offense and defense each obey a population sum-to-zero constraint. The candidate score distribution is independent Normal with a training-frozen scale.",
        "- The OD margin is `environment margin + (O + D)_home - (O + D)_away`, so a margin gate mostly tests the scalar sum `O + D`; component separation is additionally diagnosed through score and total predictions.",
        "- Component diagnostics report offense/defense-to-scalar-quality alignment and offense-minus-defense variance. Pace/scoring-environment variation can confound score residuals, so this is not interpreted as a pure possession-level offense/defense decomposition.",
        "",
        "## Stage 0",
        "",
        f"The residual panel contains {summary['stage0']['n_focal_team_games']:,} focal-team games and {summary['stage0']['n_team_seasons_with_at_least_3_games']:,} team-seasons with at least three games.",
        "",
        "| Period | Scale | Relation | Control | Lag | Bin | Pairs | Correlation |",
        "|---|---|---|---|---:|---|---:|---:|",
    ]
    for row in stage0:
        if row["season"] == "all" and row["lag"] == 1 and row["bin"] == "all":
            correlation = (
                "NA"
                if row["correlation"] is None
                else f"{float(row['correlation']):.6f}"
            )
            lines.append(
                f"| {row['period']} | {row['scale']} | {row['relation']} | {row['control']} | {row['lag']} | {row['bin']} | {row['n_pairs']} | {correlation} |"
            )
    signal = summary["stage0_signal"]
    lines += [
        "",
        f"Development-only demeaned lag-1 same-component mean correlation: `{signal['same_component_mean_lag1']}`; cross-component mean: `{signal['cross_component_mean_lag1']}`; raw difference: `{signal['same_minus_cross']}`.",
        f"The Stage 0 decision uses the per-relation observed-minus-shuffle effects, summarized as same-component `{signal['same_minus_shuffle_order_null']}`, cross-component `{signal['cross_minus_shuffle_order_null']}`, and same-minus-cross excess `{signal['same_minus_cross_excess']}`. Unrelated-team effects are reported as a descriptive secondary null because team-season demeaning can make their comparison negative.",
        "The evaluation-period signal is descriptive only. Early-vs-late correlations, lag-2 results, elapsed-time bins, season rows, shuffle-order nulls, and unrelated-team nulls are retained in the CSV artifacts.",
        "",
        "Per-relation observed and null-relative effects:",
        "",
        "| Relation | Observed | Shuffle null | Unrelated null | Observed - shuffle | Observed - unrelated |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for relation, values in signal["relation_effects"].items():
        lines.append(
            f"| {relation} | {_display(values['observed'])} | {_display(values['shuffle_order_null'])} | {_display(values['unrelated_team_null'])} | {_display(values['observed_minus_shuffle_order'])} | {_display(values['observed_minus_unrelated_team'])} |"
        )
    lines += [
        "",
        "## Stage 1",
        "",
        "The predeclared family is OD0 (independent component prior) and OD1 (rho=0.25 offense/defense prior correlation), compared with a matched scalar scoreboard control using the same environment, Normal likelihood, prior regularization, and cutoff construction. The development gate requires margin NLL improvement >= 0.010, margin MAE improvement >= 0.10 points, win Brier worsening <= 0.001, improvement in at least 3 of 4 seasons, no season NLL worsening > 0.015, and no worse NLL or MAE than the matched scalar control.",
        f"Stage 0 gate passed: **{summary['stage0_gate_passed']}**. Stage 1 ran: **{summary['stage1_ran']}**. Selected candidate: **{summary['selected_candidate'] or 'none'}**. Evaluation triggered: **{summary['stage2_triggered']}**.",
        "",
        "| Candidate | NLL improvement | MAE improvement | vs scalar NLL | vs scalar MAE | Brier delta | Improved seasons | Worst season NLL delta | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["development_gate"]:
        scalar_nll = row["od_vs_scalar_nll_improvement"]
        scalar_mae = row["od_vs_scalar_mae_improvement"]
        lines.append(
            f"| {row['candidate']} | {float(row['nll_improvement']):+.6f} | {float(row['mae_improvement']):+.6f} | {_display(scalar_nll)} | {_display(scalar_mae)} | {float(row['brier_delta']):+.6f} | {row['seasons_nll_improved']} | {float(row['worst_season_nll_delta']):+.6f} | {row['passes_gate']} |"
        )
    if eval_metrics:
        lines += [
            "",
            "## Historical evaluation",
            "",
            "2022–2025 is leakage-safe under the frozen cutoff construction but is not untouched independent confirmation because it has been used in earlier GippyRank research.",
            "",
            "| Candidate | Season | N | Margin NLL | Margin MAE | Win Brier |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in eval_metrics:
            lines.append(
                f"| {row['candidate']} | {row['season']} | {row['n']} | {float(row['margin_nll']):.6f} | {float(row['margin_mae']):.6f} | {float(row['win_brier']):.6f} |"
            )
    lines += [
        "",
        f"## Recommendation: {summary['recommendation_class']}",
        "",
        summary["recommendation"],
        "",
        "Production integrity hashes before and after are identical. The research runner writes only `data/processed/offense_defense_latent/` and does not change production artifacts.",
        "",
        f"Runtime seconds: {summary['runtime_seconds']:.3f}.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    input_root = args.input_root.resolve()
    output = args.output_root.resolve() / OUT_RELATIVE
    all_seasons = (*TRAIN_SEASONS, *DEVELOPMENT_SEASONS, *EVALUATION_SEASONS)
    print("loading historical games", flush=True)
    games = load_historical_games(input_root, all_seasons)
    likelihood = load_likelihood(input_root)
    environment = build_score_environment(games, TRAIN_SEASONS)
    before_hashes = production_hashes(input_root)

    print("loading Context priors", flush=True)
    priors = build_context_priors(input_root, all_seasons)
    print(f"building Stage 0 residuals for {len(games)} games", flush=True)
    residuals = leakage_safe_residual_rows(games, priors, likelihood, environment)
    stage0_metrics, early_late, stage0_summary = stage0_component_persistence(
        residuals, permutations=args.null_permutations
    )
    print("Stage 0 complete", flush=True)
    stage0_signal = _stage0_signal(stage0_metrics, period="development")
    evaluation_stage0_signal = _stage0_signal(stage0_metrics, period="evaluation")
    write_csv(output / "stage0_component_persistence.csv", stage0_metrics)
    write_csv(
        output / "stage0_null_metrics.csv",
        [row for row in stage0_metrics if row["control"] != "observed"]
        + [{"kind": "early_late", **row} for row in early_late],
    )

    _clear_stage1_artifacts(output)
    stage0_gate_passed = bool(stage0_signal["supports_component_persistence"])
    development_rows: list[dict[str, object]] = []
    development_diagnostics: list[dict[str, object]] = []
    gate: list[dict[str, object]] = []
    selected: str | None = None
    if stage0_gate_passed:
        print(
            f"evaluating development candidates with {len(priors)} priors", flush=True
        )
        development_rows, development_diagnostics, _development_profiles = (
            evaluate_candidates(
                games,
                priors,
                likelihood,
                environment,
                candidate_grid(),
                DEVELOPMENT_SEASONS,
            )
        )
        gate = development_gate(development_rows)
        selected = select_candidate(gate)
        _write_metric_tables(output, development_rows, development_diagnostics)
    else:
        print("Stage 0 gate failed; skipping Stage 1 and evaluation", flush=True)

    evaluation_rows: list[dict[str, object]] = []
    evaluation_diagnostics: list[dict[str, object]] = []
    evaluation_profiles: list[dict[str, object]] = []
    if stage0_gate_passed and selected is not None:
        selected_candidate = next(
            candidate for candidate in candidate_grid() if candidate.name == selected
        )
        evaluation_rows, evaluation_diagnostics, evaluation_profiles = (
            evaluate_candidates(
                games,
                priors,
                likelihood,
                environment,
                (selected_candidate,),
                EVALUATION_SEASONS,
            )
        )
        write_csv(output / "evaluation_future_metrics.csv", evaluation_rows)
        write_csv(
            output / "evaluation_season_metrics.csv",
            aggregate_future_metrics(evaluation_rows),
        )
        write_csv(
            output / "evaluation_phase_metrics.csv",
            aggregate_phase_metrics(evaluation_rows),
        )
        write_csv(output / "team_profiles.csv", evaluation_profiles)
        write_csv(
            output / "evaluation_component_diagnostics.csv", evaluation_diagnostics
        )
        write_csv(
            output / "representative_examples.csv",
            _representative_examples(evaluation_rows, selected),
        )

    after_hashes = production_hashes(input_root)
    if before_hashes != after_hashes:
        raise RuntimeError("production artifact hashes changed during research run")
    recommendation_class, recommendation = _recommendation(
        selected, stage0_signal, evaluation_rows
    )
    summary = {
        "issue": 40,
        "training_seasons": list(TRAIN_SEASONS),
        "development_seasons": list(DEVELOPMENT_SEASONS),
        "evaluation_seasons": list(EVALUATION_SEASONS),
        "input_artifacts": {
            "historical_games": "data/processed/modeling/historical_modeling_games.csv",
            "rank_distributions": "data/processed/modeling/team_season_rank_distributions.csv",
            "context_priors": "data/processed/preseason/context/predictions.csv",
            "historical_likelihood": "data/processed/posterior/historical_likelihood_v1.json",
        },
        "games_loaded": len(games),
        "environment": {
            "training_seasons": list(environment.training_seasons),
            "score_means": {
                str(key): value for key, value in environment.score_means.items()
            },
            "fallback_mean": environment.fallback_mean,
            "score_scale": environment.scale,
        },
        "stage0": stage0_summary,
        "stage0_signal": stage0_signal,
        "stage0_evaluation_signal": evaluation_stage0_signal,
        "stage0_gate_passed": stage0_gate_passed,
        "stage1_ran": stage0_gate_passed,
        "development_gate": gate,
        "selected_candidate": selected,
        "stage2_triggered": selected is not None,
        "recommendation_class": recommendation_class,
        "recommendation": recommendation,
        "production_hashes_before": before_hashes,
        "production_hashes_after": after_hashes,
        "production_hashes_unchanged": before_hashes == after_hashes,
        "runtime_seconds": time.perf_counter() - started,
    }
    model_spec = {
        "artifact_kind": "offense_defense_latent_research",
        "issue": 40,
        "candidate_grid": [candidate.__dict__ for candidate in candidate_grid()],
        "identifiability": "offense and defense each have population sum zero; fixed pairing/site score environment is the intercept",
        "expected_margin_decomposition": "Stage 0 uses an outcome-free preseason Context scalar PMF; Stage 1 V1 uses pregame posterior PMFs. Training-only pairing/site total gives expected first/second scores=(total +/- margin)/2; FBS-first cross-subdivision orientation",
        "residual_definitions": {
            "offense": "points scored - expected points scored",
            "defense": "expected opponent points - opponent points scored",
            "stage0_source": "Context preseason scalar expectation; historical rank_pairs are excluded to prevent end-of-season leakage",
        },
        "score_model": "independent Normal scores with training-frozen scale; OD mean=environment+O_team-D_opponent",
        "selection": "Stage 0 development-only null-relative gate; if passed, development 2018-2021 compares Posterior V1, matched scalar scoreboard, OD0, and OD1; evaluation 2022-2025 cannot influence candidate choice",
        "matched_scalar_control": "one centered Q scoreboard model using the same environment, Normal likelihood, prior SD, regularization, score scale, and cutoffs as OD",
        "additive_margin_limitation": "OD margin equals environment margin plus (O+D)_home-(O+D)_away, so margin metrics primarily test scalar O+D rather than separate composition",
        "confounds": "pace and scoring-environment variation can affect residuals and component diagnostics",
        "stage1_gate": "Stage 1 is skipped when Stage 0 fails; evaluation is skipped unless an OD candidate clears development and matched-scalar gates",
        "context_primary": True,
        "production_path_unchanged": True,
    }
    write_json(output / "model_spec.json", model_spec)
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text(
        _render_report(
            summary,
            stage0_metrics,
            aggregate_future_metrics(development_rows),
            aggregate_future_metrics(evaluation_rows),
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
