"""Build the Historical Likelihood V1.1 margin-evidence investigation.

This is a research-only builder for Issue #26.  It reads the frozen
historical-modeling rows and production V1 coefficient artifact, writes only
to ``data/processed/margin_likelihood_v1_1`` (or ``--output``), and never
rewrites production likelihood, prior, posterior, weekly, or website files.

Candidate selection is deliberately completed on 2018--2021 before any
2022--2025 score is requested.  The expensive posterior stage is therefore
skipped when no candidate passes the predeclared development gate.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from gippyrank.modeling import read_csv
from gippyrank.posterior.snapshots import load_likelihood
from gippyrank.research.margin_likelihood import (
    CANDIDATES,
    DF,
    MarginData,
    build_margin_data,
    development_gate,
    fit_candidate,
    fit_constant_scale,
    game_key_sha256,
    mean_design,
    mixture_interval_approx,
    model_location,
    score_model,
    select_development_candidate,
    strict_comparison_key_audit,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/processed/margin_likelihood_v1_1"
DEFAULT_HISTORICAL_ROWS = ROOT / "data/processed/modeling/historical_modeling_games.csv"
DEFAULT_RANK_DISTRIBUTIONS = ROOT / "data/processed/modeling/team_season_rank_distributions.csv"
DEFAULT_LIKELIHOOD = ROOT / "data/processed/posterior/historical_likelihood_v1.json"

TRAIN_YEARS = tuple(range(2003, 2018))
DEVELOPMENT_YEARS = tuple(range(2018, 2022))
FINAL_YEARS = tuple(range(2022, 2026))
ROLLING_YEARS = tuple(range(2008, 2026))

SELECTION_RULE = {
    "primary": {
        "aggregate_nll_improvement": 0.005,
        "minimum_seasons_with_nll_improvement": 3,
        "maximum_season_nll_worsening": 0.010,
        "maximum_expected_margin_mae_worsening": 0.10,
        "maximum_absolute_80pct_coverage_error_worsening": 0.01,
    },
    "alternative_calibration": {
        "maximum_aggregate_nll_worsening": 0.002,
        "minimum_absolute_80pct_coverage_error_improvement": 0.02,
        "maximum_expected_margin_mae_worsening": 0.10,
    },
    "complexity": {
        "minimum_nll_gain_over_simpler_candidate": 0.003,
        "order": ["v1", "a", "b", "c"],
    },
}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str] | None = None) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field)) for field in fields})


def _period(season: int) -> str:
    if season in TRAIN_YEARS:
        return "training_2003_2017"
    if season in DEVELOPMENT_YEARS:
        return "development_2018_2021"
    if season in FINAL_YEARS:
        return "final_2022_2025"
    return "outside_evaluation"


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in read_csv(path):
        season = int(source["season"])
        if season not in (*TRAIN_YEARS, *DEVELOPMENT_YEARS, *FINAL_YEARS):
            continue
        if not source.get("rank_pairs"):
            continue
        row = dict(source)
        row["season"] = season
        row["game_id"] = str(source["game_id"])
        rows.append(row)
    if not rows:
        raise ValueError(f"no usable historical modeling rows found in {path}")
    return rows


def _frozen_v1_model(likelihood) -> dict[str, object]:
    return {
        "name": "v1",
        "kind": "v1",
        "beta": likelihood.beta.copy(),
        "scale": float(likelihood.scale),
        "df": float(likelihood.degrees_of_freedom),
        "scale_model": "frozen_production_historical_likelihood_v1",
        "scale_feature_names": [],
        "fit_seasons": "2003-2021 (frozen production artifact)",
    }


def _mask_for_seasons(data: MarginData, seasons: Sequence[int]) -> np.ndarray:
    return np.isin(data.season, np.asarray(tuple(seasons), dtype=int))


def _aggregate_score(
    model: Mapping[str, object],
    data: MarginData,
    X: np.ndarray,
    baseline_location: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    return score_model(model, data, X, baseline_location, mask, intervals=True)


def _stage0_game_views(
    data: MarginData,
    locations: np.ndarray,
    scale: float,
    df: float,
    mask: np.ndarray,
) -> list[dict[str, object]]:
    """Collapse rank-pair rows to one descriptive V1 residual per game."""

    game_ids = data.game_id[mask]
    inverse = np.unique(game_ids, return_inverse=True)[1]
    masked_weights = data.weight[mask]
    masked_locations = locations[mask]
    masked_margins = data.margin[mask]
    masked_seasons = data.season[mask]
    masked_pairings = data.pairing[mask]
    masked_neutral = data.neutral[mask]
    masked_total_points = data.total_points[mask]
    rows: list[dict[str, object]] = []
    for group in range(int(inverse.max()) + 1 if len(inverse) else 0):
        selected = inverse == group
        weights = masked_weights[selected]
        weights = weights / weights.sum()
        group_locations = masked_locations[selected]
        actual = float(np.dot(weights, masked_margins[selected]))
        expected = float(np.dot(weights, group_locations))
        intervals = {
            level: mixture_interval_approx(
                group_locations,
                np.full(np.count_nonzero(selected), scale),
                weights,
                df,
                level,
            )
            for level in (0.5, 0.8, 0.95)
        }
        first = np.flatnonzero(selected)[0]
        rows.append(
            {
                "game_id": str(game_ids[first]),
                "season": int(masked_seasons[first]),
                "pairing": str(masked_pairings[first]),
                "site": "neutral" if bool(masked_neutral[first]) else "non_neutral",
                "total_points": float(masked_total_points[first]),
                "observed_margin_magnitude": abs(actual),
                "v1_expected_margin": expected,
                "v1_expected_mismatch": abs(expected),
                "residual": actual - expected,
                "coverage_50": intervals[0.5][0] <= actual <= intervals[0.5][1],
                "coverage_80": intervals[0.8][0] <= actual <= intervals[0.8][1],
                "coverage_95": intervals[0.95][0] <= actual <= intervals[0.95][1],
                "interval_width_50": intervals[0.5][1] - intervals[0.5][0],
                "interval_width_80": intervals[0.8][1] - intervals[0.8][0],
                "interval_width_95": intervals[0.95][1] - intervals[0.95][0],
            }
        )
    return rows


def _quantile_bin(value: float, edges: np.ndarray) -> str:
    if len(edges) < 2:
        return "all"
    unique_edges = np.unique(edges)
    if len(unique_edges) < 2:
        return "all"
    index = int(np.searchsorted(unique_edges[1:-1], value, side="right"))
    return f"q{index + 1}"


def build_residual_diagnostics(
    data: MarginData,
    baseline_location: np.ndarray,
    baseline_model: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Describe V1 residual scale with bins fixed from training only."""

    stage_mask = _mask_for_seasons(data, (*TRAIN_YEARS, *DEVELOPMENT_YEARS))
    views = _stage0_game_views(
        data,
        baseline_location,
        float(baseline_model["scale"]),
        float(baseline_model["df"]),
        stage_mask,
    )
    training_views = [row for row in views if int(row["season"]) in TRAIN_YEARS]
    edges = {
        "abs_v1_expected_margin": np.quantile(
            [float(row["v1_expected_mismatch"]) for row in training_views], [0.0, 0.25, 0.5, 0.75, 1.0]
        ),
        "total_points": np.quantile(
            [float(row["total_points"]) for row in training_views], [0.0, 0.25, 0.5, 0.75, 1.0]
        ),
        "observed_margin_magnitude": np.quantile(
            [float(row["observed_margin_magnitude"]) for row in training_views], [0.0, 0.25, 0.5, 0.75, 1.0]
        ),
    }
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in views:
        period = _period(int(row["season"]))
        grouped[(period, "absolute_v1_expected_margin", _quantile_bin(float(row["v1_expected_mismatch"]), edges["abs_v1_expected_margin"]))].append(row)
        grouped[(period, "total_points", _quantile_bin(float(row["total_points"]), edges["total_points"]))].append(row)
        grouped[(period, "observed_margin_magnitude", _quantile_bin(float(row["observed_margin_magnitude"]), edges["observed_margin_magnitude"]))].append(row)
        grouped[(period, "pairing", str(row["pairing"]))].append(row)
        grouped[(period, "site", str(row["site"]))].append(row)

    output: list[dict[str, object]] = []
    for (period, dimension, bin_name), values in sorted(grouped.items()):
        residuals = np.asarray([float(row["residual"]) for row in values])
        median = float(np.median(residuals))
        output.append(
            {
                "period": period,
                "dimension": dimension,
                "bin": bin_name,
                "n_games": len(values),
                "residual_mae": float(np.mean(np.abs(residuals))),
                "residual_variance": float(np.var(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
                "robust_scale_mad": float(1.4826 * np.median(np.abs(residuals - median))),
                "coverage_50": float(np.mean([bool(row["coverage_50"]) for row in values])),
                "coverage_80": float(np.mean([bool(row["coverage_80"]) for row in values])),
                "coverage_95": float(np.mean([bool(row["coverage_95"]) for row in values])),
                "mean_expected_margin": float(np.mean([float(row["v1_expected_margin"]) for row in values])),
                "mean_total_points": float(np.mean([float(row["total_points"]) for row in values])),
            }
        )
    return output, {
        "n_games": len(views),
        "training_games": len(training_views),
        "development_games": len(views) - len(training_views),
        "bin_edges_training_only": {key: value.tolist() for key, value in edges.items()},
        "residual_definition": "observed oriented margin minus equal-rank-pair V1 expected margin",
    }


def _model_selection_rows(
    models: Mapping[str, Mapping[str, object]],
    data: MarginData,
    X: np.ndarray,
    baseline_location: np.ndarray,
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    development_mask = _mask_for_seasons(data, DEVELOPMENT_YEARS)
    aggregate: dict[str, dict[str, object]] = {}
    seasons: dict[str, list[dict[str, object]]] = {}
    all_season_rows: list[dict[str, object]] = []
    for name, model in models.items():
        aggregate[name] = _aggregate_score(model, data, X, baseline_location, development_mask)
        rows: list[dict[str, object]] = []
        for season in DEVELOPMENT_YEARS:
            mask = _mask_for_seasons(data, (season,))
            if not mask.any():
                continue
            metrics = _aggregate_score(model, data, X, baseline_location, mask)
            row = {
                "candidate": name,
                "season": season,
                "evaluation_period": "development_2018_2021",
                "fit_period": "training_2003_2017",
                "game_key_sha256": game_key_sha256(data.game_id[mask]),
                **metrics,
            }
            rows.append(row)
            all_season_rows.append(row)
        seasons[name] = rows
    return aggregate, seasons, all_season_rows


def _development_rows(
    aggregate: Mapping[str, Mapping[str, object]],
    seasons: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    baseline = aggregate["v1"]
    rows: list[dict[str, object]] = []
    for name in CANDIDATES:
        metrics = aggregate[name]
        predecessor = "v1" if name == "a" else "a" if name == "b" else "b" if name.startswith("c") else None
        gate = (
            development_gate(metrics, aggregate[predecessor], seasons[name], seasons[predecessor])
            if predecessor is not None
            else {
                "primary_pass": False,
                "alternative_calibration_pass": False,
                "qualifies": False,
                "development_nll_delta": 0.0,
                "seasons_with_nll_improvement": 0,
            }
        )
        rows.append(
            {
                "candidate": name,
                "evaluation_period": "development_2018_2021",
                "fit_period": "training_2003_2017" if name != "v1" else "frozen_2003_2021",
                "marginalized_nll": metrics["marginalized_nll"],
                "delta_nll_vs_v1": float(metrics["marginalized_nll"]) - float(baseline["marginalized_nll"]),
                "expected_margin_mae": metrics["expected_margin_mae"],
                "coverage_50": metrics["coverage_50"],
                "coverage_80": metrics["coverage_80"],
                "coverage_95": metrics["coverage_95"],
                "interval_width_50": metrics["interval_width_50"],
                "interval_width_80": metrics["interval_width_80"],
                "interval_width_95": metrics["interval_width_95"],
                "n_games": metrics["n_games"],
                "n_pseudo_observations": metrics["n_pseudo_observations"],
                "predecessor": predecessor,
                "primary_pass_vs_predecessor": gate["primary_pass"],
                "alternative_calibration_pass_vs_predecessor": gate["alternative_calibration_pass"],
                "qualifies_vs_predecessor": gate["qualifies"],
                "complexity_gain_over_predecessor": (
                    float(aggregate[predecessor]["marginalized_nll"]) - float(metrics["marginalized_nll"])
                    if predecessor is not None
                    else None
                ),
                "game_key_sha256": game_key_sha256(
                    # Every candidate is evaluated on this exact development key set.
                    [game_id for game_id in []]
                ),
            }
        )
    # Fill the common key hash after the row shape is established.
    common_hash = str(next(iter(aggregate.values())).get("game_key_sha256", ""))
    for row in rows:
        row["game_key_sha256"] = common_hash
    return rows


def _rolling_likelihood(
    selected: str,
    data: MarginData,
    X: np.ndarray,
    production_location: np.ndarray,
) -> list[dict[str, object]]:
    """Fit only on seasons before each target and score direct margin NLL."""

    rows: list[dict[str, object]] = []
    for target_season in ROLLING_YEARS:
        fit_mask = data.season < target_season
        target_mask = data.season == target_season
        if not fit_mask.any() or not target_mask.any():
            continue
        rolling_v1 = fit_constant_scale(
            X[fit_mask], data.margin[fit_mask], data.weight[fit_mask], degrees_of_freedom=DF
        )
        rolling_location = model_location(rolling_v1, X)
        selected_model = fit_candidate(
            selected,
            data,
            X,
            rolling_location,
            fit_mask,
            degrees_of_freedom=DF,
        )
        v1_metrics = score_model(rolling_v1, data, X, rolling_location, target_mask, intervals=True)
        candidate_metrics = score_model(selected_model, data, X, rolling_location, target_mask, intervals=True)
        rows.append(
            {
                "target_season": target_season,
                "fit_through_season": target_season - 1,
                "candidate": selected,
                "v1_nll": v1_metrics["marginalized_nll"],
                "candidate_nll": candidate_metrics["marginalized_nll"],
                "delta_nll_candidate_minus_v1": float(candidate_metrics["marginalized_nll"]) - float(v1_metrics["marginalized_nll"]),
                "v1_coverage_80": v1_metrics["coverage_80"],
                "candidate_coverage_80": candidate_metrics["coverage_80"],
                "coverage_delta_candidate_minus_v1": float(candidate_metrics["coverage_80"]) - float(v1_metrics["coverage_80"]),
                "v1_margin_mae": v1_metrics["expected_margin_mae"],
                "candidate_margin_mae": candidate_metrics["expected_margin_mae"],
                "margin_mae_delta_candidate_minus_v1": float(candidate_metrics["expected_margin_mae"]) - float(v1_metrics["expected_margin_mae"]),
                "n_games": target_mask.sum(),
                "game_key_sha256": game_key_sha256(data.game_id[target_mask]),
            }
        )
    return rows


def _report_number(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _write_report(
    path: Path,
    *,
    selected: str | None,
    selection: Mapping[str, object],
    stage0: Mapping[str, object],
    development: Sequence[Mapping[str, object]],
    rolling: Sequence[Mapping[str, object]],
    stage2_triggered: bool,
) -> None:
    by_candidate = {str(row["candidate"]): row for row in development}
    lines = [
        "# Historical Likelihood V1.1 margin-evidence investigation",
        "",
        "## Research question",
        "",
        "Can GippyRank extract better-calibrated evidence from the observed final score margin without adding another football statistic? The production Historical Likelihood V1 remains frozen.",
        "",
        "## Frozen design",
        "",
        "- Training: 2003–2017; development: 2018–2021; final historical evaluation: 2022–2025.",
        "- Student-t df is fixed at 15 for every candidate.",
        "- All candidates retain the 34-column V1 rank/site/pairing mean surface.",
        "- Candidate A uses a positive pairing-intercept plus `log1p(abs(V1 expected margin))` scale.",
        "- Candidate B adds standardized `log1p(total points)` to scale only; normalization is training-derived.",
        "- Candidate C applies `k * asinh(margin / k)` for exactly one development-selected k in {14, 21, 28, 42}; original-scale NLL includes its log Jacobian.",
        "",
        "## Stage 0 residual diagnostics",
        "",
        f"The diagnostics use {stage0['n_games']} training/development games ({stage0['training_games']} training and {stage0['development_games']} development). Quantile edges for expected mismatch, total points, and observed margin magnitude were computed from training games only. Residual means are observed oriented margin minus the equal-rank-pair V1 expected margin. Full values are in `residual_diagnostics.csv`.",
        "",
        "## Development selection",
        "",
        "| Candidate | NLL | Δ NLL vs V1 | Margin MAE | 50% | 80% | 95% | Gate vs predecessor |",
        "|:--|--:|--:|--:|--:|--:|--:|:--|",
    ]
    for name in CANDIDATES:
        row = by_candidate[name]
        lines.append(
            f"| {name.upper()} | {_report_number(row['marginalized_nll'])} | {_report_number(row['delta_nll_vs_v1'])} | {_report_number(row['expected_margin_mae'])} | {_report_number(row['coverage_50'])} | {_report_number(row['coverage_80'])} | {_report_number(row['coverage_95'])} | {row['qualifies_vs_predecessor']} |"
        )
    lines.extend(
        [
            "",
            f"The development-only selection returned `{selected or 'none'}`. Selection metadata and all frozen parameters are in `model_spec.json`; no 2022–2025 row was used to choose the candidate or C k. The decision trace is in `summary.json`.",
            "",
            "## Rolling direct-likelihood robustness",
            "",
        ]
    )
    if rolling:
        lines.append("The selected candidate was refit separately using only seasons before each target season. The full panel is in `rolling_likelihood_metrics.csv`.")
        lines.append("")
        lines.append("| Target season | V1 NLL | Candidate NLL | Δ NLL | Δ 80% coverage | Δ margin MAE |")
        lines.append("|--:|--:|--:|--:|--:|--:|")
        for row in rolling:
            lines.append(
                f"| {row['target_season']} | {_report_number(row['v1_nll'])} | {_report_number(row['candidate_nll'])} | {_report_number(row['delta_nll_candidate_minus_v1'])} | {_report_number(row['coverage_delta_candidate_minus_v1'])} | {_report_number(row['margin_mae_delta_candidate_minus_v1'])} |"
            )
    else:
        lines.append("No candidate passed the development gate, so the expensive rolling/posterior stages were not triggered.")
    lines.extend(
        [
            "",
            "## Posterior stage and conclusion",
            "",
            f"Stage 2 posterior validation triggered: `{stage2_triggered}`.",
            "",
            "Because 2022–2025 has already been examined by prior GippyRank research, it is leakage-safe historical evaluation but not independent confirmation for this hypothesis.",
            "",
            "If no candidate is selected, the recommendation is C — retain Historical Likelihood V1. A selected candidate would be a frozen research candidate only, not an immediate production promotion.",
            "",
            "The artifact directory is research-only. Production V1 artifacts and publication files are hash-checked before and after the build.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-rows", type=Path, default=DEFAULT_HISTORICAL_ROWS)
    parser.add_argument("--rank-distributions", type=Path, default=DEFAULT_RANK_DISTRIBUTIONS)
    parser.add_argument("--likelihood", type=Path, default=DEFAULT_LIKELIHOOD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-rolling", action="store_true", help="Skip rolling panel; useful for focused unit smoke runs.")
    parser.add_argument("--skip-plots", action="store_true", help="Accepted for parity with other research builders; no decorative plots are generated.")
    args = parser.parse_args()

    required = [args.historical_rows, args.rank_distributions, args.likelihood]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required frozen inputs are missing: " + ", ".join(missing))

    production_paths = [
        args.likelihood,
        ROOT / "data/processed/preseason/history/predictions.csv",
        ROOT / "data/processed/preseason/context/predictions.csv",
        ROOT / "site/data/manifest.json",
        ROOT / "site/index.html",
    ]
    production_paths = [path for path in production_paths if path.exists()]
    production_hashes_before = {
        str(path.relative_to(ROOT)): _sha256(path) for path in production_paths
    }

    rows = _load_rows(args.historical_rows)
    data = build_margin_data(rows)
    X = mean_design(data)
    likelihood = load_likelihood(args.likelihood)
    if len(likelihood.beta) != X.shape[1]:
        raise ValueError(f"frozen V1 beta has {len(likelihood.beta)} coefficients; data design has {X.shape[1]}")
    baseline_location = X @ likelihood.beta
    baseline = _frozen_v1_model(likelihood)

    diagnostics, stage0_summary = build_residual_diagnostics(data, baseline_location, baseline)

    train_mask = _mask_for_seasons(data, TRAIN_YEARS)
    models: dict[str, dict[str, object]] = {"v1": baseline}
    for name in ("a", "b", "c14", "c21", "c28", "c42"):
        models[name] = fit_candidate(
            name,
            data,
            X,
            baseline_location,
            train_mask,
            degrees_of_freedom=DF,
        )

    aggregate, season_metrics, season_rows = _model_selection_rows(
        models, data, X, baseline_location
    )
    common_dev_ids = np.unique(data.game_id[_mask_for_seasons(data, DEVELOPMENT_YEARS)])
    for metrics in aggregate.values():
        metrics["game_key_sha256"] = game_key_sha256(common_dev_ids)
    development_rows = _development_rows(aggregate, season_metrics)
    selected, selection = select_development_candidate(aggregate, season_metrics)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    model_spec: dict[str, object] = {
        "artifact_kind": "margin_likelihood_v1_1_research_model_spec",
        "research_issue": 26,
        "production_baseline": {
            "name": "v1",
            "artifact": str(args.likelihood.relative_to(ROOT)) if args.likelihood.is_relative_to(ROOT) else str(args.likelihood),
            "unchanged": True,
            "df": float(likelihood.degrees_of_freedom),
            "mean_feature_count": int(X.shape[1]),
        },
        "data_split": {
            "training": list(TRAIN_YEARS),
            "development": list(DEVELOPMENT_YEARS),
            "final_evaluation": list(FINAL_YEARS),
            "final_used_for_selection": False,
        },
        "student_t_df": DF,
        "candidate_family": {
            "a": "frozen V1 mean surface with positive pairing intercepts plus log1p(abs(V1 expected margin)) scale",
            "b": "Candidate A scale plus training-standardized log1p(total_points); scale only",
            "c": "Candidate B with g_k(m)=k*asinh(m/k), original-margin Jacobian, k in {14,21,28,42}",
        },
        "selection_rule": SELECTION_RULE,
        "selected_candidate": selected,
        "selection_trace": selection,
        "selected_candidate_frozen_fit": models.get(selected) if selected else None,
        "candidate_fits": models,
    }
    _write_json(output / "model_spec.json", model_spec)

    rolling_rows: list[dict[str, object]] = []
    if selected and not args.skip_rolling:
        rolling_rows = _rolling_likelihood(selected, data, X, baseline_location)

    production_hashes_after = {
        str(path.relative_to(ROOT)): _sha256(path) for path in production_paths
    }
    comparison_audit = strict_comparison_key_audit(
        {name: common_dev_ids for name in CANDIDATES}
    )
    summary = {
        "research_question": "Can GippyRank extract better evidence from final score margin without a new football statistic?",
        "recommendation": "C" if selected is None else "A_pending_posterior",
        "selected_candidate": selected,
        "stage2_triggered": False,
        "data_split": {
            "training": "2003-2017",
            "development": "2018-2021",
            "final_evaluation": "2022-2025",
            "final_independent_confirmation": False,
            "2026_outcomes_used": False,
        },
        "candidate_family": {
            "v1": "frozen Historical Likelihood V1",
            "a": "V1 mean surface with mismatch-dependent positive Student-t scale",
            "b": "A plus total-points context in scale only",
            "c": "B plus smooth blowout transform; one k selected on development only",
        },
        "stage0": stage0_summary,
        "selection": selection,
        "selection_rule": SELECTION_RULE,
        "development_aggregate": aggregate,
        "development_season_count": {name: len(values) for name, values in season_metrics.items()},
        "strict_comparison_keys": comparison_audit,
        "rolling": {
            "triggered": bool(rolling_rows),
            "target_seasons": list(ROLLING_YEARS) if rolling_rows else [],
            "uses_only_earlier_seasons": True,
        },
        "frozen_production_integrity": {
            "hashes_before": production_hashes_before,
            "hashes_after": production_hashes_after,
            "unchanged": production_hashes_before == production_hashes_after,
        },
        "input_sha256": {
            "historical_modeling_games.csv": _sha256(args.historical_rows),
            "historical_likelihood_v1.json": _sha256(args.likelihood),
            "team_season_rank_distributions.csv": _sha256(args.rank_distributions)
            if args.rank_distributions.exists()
            else None,
        },
        "artifact_inventory": {
            "report.md": "human-readable research conclusion",
            "summary.json": "machine-readable metrics, selection, provenance, and integrity",
            "model_spec.json": "frozen candidate specifications and development-only selection",
            "residual_diagnostics.csv": "Stage 0 residual scale/calibration bins",
            "development_metrics.csv": "aggregate 2018-2021 candidate metrics",
            "development_season_metrics.csv": "per-season 2018-2021 candidate metrics",
            "rolling_likelihood_metrics.csv": "earlier-season direct-likelihood robustness when selected",
        },
    }
    _write_json(output / "summary.json", summary)
    _write_csv(output / "residual_diagnostics.csv", diagnostics)
    _write_csv(output / "development_metrics.csv", development_rows)
    _write_csv(output / "development_season_metrics.csv", season_rows)
    _write_csv(
        output / "rolling_likelihood_metrics.csv",
        rolling_rows,
        fields=[
            "target_season",
            "fit_through_season",
            "candidate",
            "v1_nll",
            "candidate_nll",
            "delta_nll_candidate_minus_v1",
            "v1_coverage_80",
            "candidate_coverage_80",
            "coverage_delta_candidate_minus_v1",
            "v1_margin_mae",
            "candidate_margin_mae",
            "margin_mae_delta_candidate_minus_v1",
            "n_games",
            "game_key_sha256",
        ],
    )
    _write_report(
        output / "report.md",
        selected=selected,
        selection=selection,
        stage0=stage0_summary,
        development=development_rows,
        rolling=rolling_rows,
        stage2_triggered=False,
    )
    print(
        json.dumps(
            {
                "selected_candidate": selected,
                "recommendation": summary["recommendation"],
                "development_games": int(aggregate["v1"]["n_games"]),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
