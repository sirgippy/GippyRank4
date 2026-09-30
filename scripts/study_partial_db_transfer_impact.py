"""Reproduce issue #142's research-only DB impact masking study.

Uses committed retrospective player audits. Nothing in the production Context
feature construction or published rankings is changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from gippyrank.preseason import pmf_summaries, rank_bin_edges

ROOT = Path(__file__).resolve().parents[1]
PLAYER_PATH = ROOT / "data/processed/defensive_transfer_audit/transfer_player_audit.csv"
FEATURE_PATH = (
    ROOT
    / "data/processed/preseason/context_v1_3_candidate/historical_transfer_features.csv"
)
AVAILABILITY_PATH = ROOT / "data/processed/transfer_availability_audit/team_seasons.csv"
CURRENT_PLAYER_PATH = (
    ROOT
    / "data/processed/preseason/context_v1_3_2026_reconstruction/transfer_player_audit.csv"
)
CURRENT_FEATURE_PATH = (
    ROOT
    / "data/processed/preseason/context_v1_3_2026_reconstruction/transfer_features.csv"
)
CURRENT_MODEL_PATH = (
    ROOT / "data/processed/preseason/context_v1_3/annual/2026/fitted_model.json"
)
CURRENT_PREDICTIONS_PATH = (
    ROOT / "data/processed/preseason/context_v1_3/annual/2026/predictions.csv"
)
DEFAULT_OUTPUT = ROOT / "data/processed/partial_db_transfer_impact_142"
SEED = 142
GOOD = {"resolved", "zero_recorded_defensive_box_score_games"}
POLICIES = ("all_or_nothing", "observed", "coverage_scaled", "missing_mean")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_values(
    observed: list[float], missing_count: int, prior_mean: float
) -> dict[str, float]:
    """All policies preserve the known roster; only its impacts are masked."""
    known = float(sum(observed))
    total = len(observed) + missing_count
    return {
        "all_or_nothing": known if not missing_count else 0.0,
        "observed": known,
        "coverage_scaled": known * total / len(observed) if observed else 0.0,
        "missing_mean": known + missing_count * prior_mean,
    }


def mask_sets(n: int, k: int) -> list[tuple[int, ...]]:
    """Enumerate small masks; cap large strata reproducibly and keep each tail."""
    masks = list(itertools.combinations(range(n), n - k))
    if len(masks) > 80:
        rng = random.Random(SEED + 100 * n + k)
        selected = rng.sample(masks, 78)
        masks = sorted({masks[0], masks[-1], *selected})
    return masks


def summarize(rows: list[dict[str, object]], group: str) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row[group]), str(row["policy"]))].append(row)
    result = []
    for (stratum, policy), items in sorted(groups.items()):
        truth = np.array([float(r["full_impact"]) for r in items])
        estimate = np.array([float(r["estimate"]) for r in items])
        error = estimate - truth
        corr = (
            float(np.corrcoef(truth, estimate)[0, 1])
            if np.std(truth) > 0 and np.std(estimate) > 0
            else None
        )
        result.append(
            {
                "stratum_type": group,
                "stratum": stratum,
                "policy": policy,
                "masks": len(items),
                "team_seasons": len({(r["season"], r["team_id"]) for r in items}),
                "bias": float(np.mean(error)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "correlation": corr,
                "large_error_gt_1": float(np.mean(np.abs(error) > 1.0)),
                "wins_vs_zero_fraction": float(np.mean(np.abs(error) < np.abs(truth))),
            }
        )
    return result


def summarize_team_weighted(
    rows: list[dict[str, object]], group: str
) -> list[dict[str, object]]:
    """Give every historical (team-season, exact coverage) cell one vote."""
    cells: dict[tuple[str, str, int, str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        key = (
            str(row[group]),
            str(row["policy"]),
            int(row["season"]),
            str(row["team_id"]),
            str(row["coverage"]),
        )
        cells[key].append(row)
    grouped: dict[tuple[str, str], list[tuple[tuple, list[dict[str, object]]]]] = (
        defaultdict(list)
    )
    for key, items in cells.items():
        grouped[key[:2]].append((key, items))
    result = []
    for (stratum, policy), members in sorted(grouped.items()):
        errors = [
            np.array([float(r["estimate"]) - float(r["full_impact"]) for r in items])
            for _, items in members
        ]
        truth = [float(items[0]["full_impact"]) for _, items in members]
        all_truth = np.array(
            [float(r["full_impact"]) for _, items in members for r in items]
        )
        all_estimate = np.array(
            [float(r["estimate"]) for _, items in members for r in items]
        )
        weights = np.array([1.0 / len(items) for _, items in members for _ in items])
        weights /= weights.sum()
        truth_mean = float(np.dot(weights, all_truth))
        estimate_mean = float(np.dot(weights, all_estimate))
        truth_variance = float(np.dot(weights, (all_truth - truth_mean) ** 2))
        estimate_variance = float(np.dot(weights, (all_estimate - estimate_mean) ** 2))
        correlation = (
            float(
                np.dot(
                    weights,
                    (all_truth - truth_mean) * (all_estimate - estimate_mean),
                )
                / np.sqrt(truth_variance * estimate_variance)
            )
            if truth_variance > 0 and estimate_variance > 0
            else None
        )
        result.append(
            {
                "stratum_type": group,
                "stratum": stratum,
                "policy": policy,
                "masks": sum(len(items) for _, items in members),
                "team_seasons": len({(key[2], key[3]) for key, _ in members}),
                "team_coverage_cells": len(members),
                "bias": float(np.mean([np.mean(error) for error in errors])),
                "mae": float(np.mean([np.mean(np.abs(error)) for error in errors])),
                "rmse": float(
                    np.sqrt(np.mean([np.mean(error**2) for error in errors]))
                ),
                "correlation": correlation,
                "large_error_gt_1": float(
                    np.mean([np.mean(np.abs(error) > 1.0) for error in errors])
                ),
                "wins_vs_zero_fraction": float(
                    np.mean(
                        [
                            np.mean(np.abs(error) < abs(full))
                            for error, full in zip(errors, truth, strict=True)
                        ]
                    )
                ),
            }
        )
    return result


def shift_pmf(pmf: np.ndarray, delta: float) -> np.ndarray:
    """Interpolate a stored rank CDF after a frozen location-coordinate shift."""
    edges = rank_bin_edges(len(pmf))
    cdf = np.r_[0.0, np.cumsum(pmf)]
    shifted = np.interp(edges[1:-1] - delta, edges, cdf)
    masses = np.maximum(np.diff(np.r_[0.0, shifted, 1.0]), 0.0)
    return masses / masses.sum()


def downstream(
    prior_mean: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """2026 one-missing sensitivity from frozen PMFs and location coefficient."""
    model = json.loads(CURRENT_MODEL_PATH.read_text(encoding="utf-8"))["model"]
    names = model["feature_names"]
    feature = "transfer_in_prior_defensive_impact_db_sum"
    idx = names.index(feature)
    coefficient = model["location_coefficients"][model["lag_count"] + 1 + idx]
    scale = model["preprocessing"]["scales"][feature]
    availability = "transfer_in_prior_defensive_impact_db_available"
    availability_idx = names.index(availability)
    availability_coefficient = model["location_coefficients"][
        model["lag_count"] + 1 + availability_idx
    ]
    availability_scale = model["preprocessing"]["scales"][availability]
    if feature in model["scale_feature_names"]:
        raise ValueError("PMF shift diagnostic requires location-only DB impact")
    features = {r["team_id"]: r for r in read_csv(CURRENT_FEATURE_PATH)}
    predictions = {r["team_id"]: r for r in read_csv(CURRENT_PREDICTIONS_PATH)}
    players: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(CURRENT_PLAYER_PATH):
        if (
            row["in_model_relevant_population"] == "True"
            and row["portal_position_group"] == "db"
        ):
            players[row["destination_team_id"]].append(row)
    full_rank = {key: float(row["expected_rank"]) for key, row in predictions.items()}
    cases = []
    result = []
    for team_id, group in sorted(players.items()):
        if len(group) < 2 or any(
            r["impact_status"] not in GOOD or not r["prior_defensive_impact"]
            for r in group
        ):
            continue
        values = [float(r["prior_defensive_impact"]) for r in group]
        truth = sum(values)
        if (
            features[team_id]["transfer_in_prior_defensive_impact_db_available"] != "1"
            or abs(float(features[team_id][feature]) - truth) > 1e-8
        ):
            raise ValueError(f"2026 full impact does not reconstruct: {team_id}")
        original = np.asarray(json.loads(predictions[team_id]["pmf"]), dtype=float)
        peers = [rank for key, rank in full_rank.items() if key != team_id]
        original_ordinal = 1 + sum(rank < full_rank[team_id] for rank in peers)
        for hidden, missing in enumerate(values):
            observed = values[:hidden] + values[hidden + 1 :]
            candidates = candidate_values(observed, 1, prior_mean)
            cases.append(
                {
                    "season": 2026,
                    "team_id": team_id,
                    "total_count": len(values),
                    "observed_count": len(observed),
                    "coverage": f"{len(observed)}/{len(values)}",
                    "hidden_index": hidden,
                    "full_impact": truth,
                    "missing_impact": missing,
                    **candidates,
                }
            )
            for policy, estimate in candidates.items():
                # The frozen binary availability flag changes from 1 to 0.
                # A distinct partial-coverage indicator would require refit.
                delta = (
                    coefficient * (estimate - truth) / scale
                    - availability_coefficient / availability_scale
                )
                pmf = shift_pmf(original, delta)
                expected = pmf_summaries(pmf)["expected_rank"]
                new_ordinal = 1 + sum(rank < expected for rank in peers)
                result.append(
                    {
                        "season": 2026,
                        "team_id": team_id,
                        "total_count": len(values),
                        "coverage": f"{len(observed)}/{len(values)}",
                        "hidden_index": hidden,
                        "policy": policy,
                        "expected_rank_delta": expected - full_rank[team_id],
                        "pmf_total_variation": float(
                            0.5 * np.abs(pmf - original).sum()
                        ),
                        "ordinal_displacement": new_ordinal - original_ordinal,
                    }
                )
    return cases, result


def run(output: Path, *, with_downstream: bool = True) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    panel = {(r["season"], r["team_id"]): r for r in read_csv(FEATURE_PATH)}
    players: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(PLAYER_PATH):
        if (
            row["in_model_relevant_population"] == "True"
            and row["portal_position_group"] == "db"
            and 2021 <= int(row["season"]) <= 2025
        ):
            players[(row["season"], row["destination_team_id"])].append(row)
    complete = {}
    excluded = Counter()
    for key, group in sorted(players.items()):
        if key not in panel:
            excluded["outside_context_panel"] += 1
        elif any(
            r["impact_status"] not in GOOD or not r["prior_defensive_impact"]
            for r in group
        ):
            excluded["incomplete_impact"] += 1
        else:
            values = [float(r["prior_defensive_impact"]) for r in group]
            full = float(panel[key]["transfer_in_prior_defensive_impact_db_sum"])
            if (
                panel[key]["transfer_in_prior_defensive_impact_db_available"] != "1.0"
                or abs(sum(values) - full) > 1e-8
            ):
                raise ValueError(
                    f"full impact does not reconstruct frozen feature: {key}"
                )
            complete[key] = values
    year_values = defaultdict(list)
    for (season, _), values in complete.items():
        year_values[season].extend(values)
    heldout_mean = {}
    heldout_sd = {}
    for season in year_values:
        training = [
            v for other, vals in year_values.items() if other != season for v in vals
        ]
        heldout_mean[season] = float(np.mean(training))
        heldout_sd[season] = float(np.std(training))
    cases = []
    metrics = []
    for (season, team_id), values in sorted(complete.items()):
        n = len(values)
        if n < 2:
            continue  # one missing leaves no observed impact
        for k in range(1, n):
            for hidden in mask_sets(n, k):
                hidden_set = set(hidden)
                observed = [v for i, v in enumerate(values) if i not in hidden_set]
                missing = [values[i] for i in hidden]
                candidates = candidate_values(
                    observed, len(missing), heldout_mean[season]
                )
                missing_sum = float(sum(missing))
                case = {
                    "season": int(season),
                    "team_id": team_id,
                    "total_count": n,
                    "observed_count": k,
                    "missing_count": n - k,
                    "coverage_fraction": k / n,
                    "hidden_indices": ";".join(map(str, hidden)),
                    "observed_sum": float(sum(observed)),
                    "missing_sum": missing_sum,
                    "full_impact": float(sum(values)),
                    "missing_abs": abs(missing_sum),
                    "missing_sd_proxy": heldout_sd[season] * np.sqrt(n - k),
                    "missing_90_interval_covers": abs(
                        missing_sum - (n - k) * heldout_mean[season]
                    )
                    <= 1.645 * heldout_sd[season] * np.sqrt(n - k),
                    "missing_direction": "positive"
                    if missing_sum > 0
                    else "negative"
                    if missing_sum < 0
                    else "zero",
                    "coverage_band": "high_80_plus"
                    if k / n >= 0.8
                    else "moderate_50_to_80"
                    if k / n >= 0.5
                    else "low_under_50",
                    **candidates,
                }
                cases.append(case)
                for policy, estimate in candidates.items():
                    metrics.append(
                        {
                            "season": int(season),
                            "team_id": team_id,
                            "total_count": n,
                            "missing_count": n - k,
                            "hidden_contribution": missing_sum,
                            "coverage": f"{k}/{n}",
                            "overall": "all",
                            "coverage_band": case["coverage_band"],
                            "missing_direction": case["missing_direction"],
                            "missing_magnitude": "large_gt_1"
                            if abs(missing_sum) > 1
                            else "small_le_1",
                            "policy": policy,
                            "full_impact": case["full_impact"],
                            "estimate": estimate,
                        }
                    )
    write_csv(output / "mask_cases.csv", cases)
    uncertainty = []
    for band in ("high_80_plus", "moderate_50_to_80", "low_under_50"):
        selected = [r for r in cases if r["coverage_band"] == band]
        uncertainty.append(
            {
                "coverage_band": band,
                "masks": len(selected),
                "nominal_coverage": 0.90,
                "actual_coverage": float(
                    np.mean([r["missing_90_interval_covers"] for r in selected])
                ),
                "mean_missing_90_half_width": float(
                    np.mean([1.645 * r["missing_sd_proxy"] for r in selected])
                ),
            }
        )
    write_csv(output / "uncertainty_summary.csv", uncertainty)
    write_csv(output / "policy_errors.csv", metrics)
    summary_rows = (
        summarize(metrics, "overall")
        + summarize(metrics, "coverage")
        + summarize(metrics, "coverage_band")
        + summarize(metrics, "missing_magnitude")
        + summarize(metrics, "missing_direction")
    )
    write_csv(output / "policy_summary.csv", summary_rows)
    team_weighted = (
        summarize_team_weighted(metrics, "overall")
        + summarize_team_weighted(metrics, "coverage")
        + summarize_team_weighted(metrics, "coverage_band")
    )
    write_csv(output / "team_weighted_summary.csv", team_weighted)
    empirical = []
    for row in read_csv(AVAILABILITY_PATH):
        if row["season"] != "2026" or int(row["db_incoming"]) == 0:
            continue
        n, k = int(row["db_incoming"]), int(row["db_resolved"])
        empirical.append(
            {
                "season": 2026,
                "team_id": row["team_id"],
                "team_name": row["team_name"],
                "total_count": n,
                "observed_count": k,
                "missing_count": n - k,
                "coverage_fraction": k / n,
                "maskable_exact_n": n
                in {len(v) for v in complete.values() if len(v) >= 2},
                "partial": n > k,
            }
        )
    write_csv(output / "empirical_2026_coverage.csv", empirical)
    by_coverage: dict[tuple[int, int, str], list[dict[str, object]]] = defaultdict(list)
    for item in metrics:
        n = int(item["total_count"])
        k = int(str(item["coverage"]).split("/")[0])
        by_coverage[(n, k, str(item["policy"]))].append(item)
    production_coverage = Counter(
        (int(item["total_count"]), int(item["observed_count"]))
        for item in empirical
        if item["partial"] and item["observed_count"]
    )
    empirical_support = []
    for (n, k), production_teams in sorted(production_coverage.items()):
        comparable = by_coverage[(n, k, "observed")]
        historical_teams = len({(r["season"], r["team_id"]) for r in comparable})
        empirical_support.append(
            {
                "coverage": f"{k}/{n}",
                "production_teams": production_teams,
                "historical_team_seasons": historical_teams,
                "historical_masks": len(comparable),
                "support_status": "unmatched"
                if historical_teams == 0
                else "sparse_one_team"
                if historical_teams == 1
                else "sparse_two_teams"
                if historical_teams == 2
                else "multi_team",
            }
        )
    write_csv(output / "empirical_replay_support.csv", empirical_support)
    empirical_replay = []
    for item in empirical:
        if not item["partial"] or not item["observed_count"]:
            continue
        for policy in POLICIES:
            comparable = by_coverage[
                (item["total_count"], item["observed_count"], policy)
            ]
            if not comparable:
                continue
            team_errors: dict[tuple[int, str], list[float]] = defaultdict(list)
            for row in comparable:
                team_errors[(int(row["season"]), str(row["team_id"]))].append(
                    float(row["estimate"]) - float(row["full_impact"])
                )
            empirical_replay.append(
                {
                    "team_id": item["team_id"],
                    "coverage": f"{item['observed_count']}/{item['total_count']}",
                    "policy": policy,
                    "historical_masks": len(comparable),
                    "historical_team_seasons": len(team_errors),
                    "sparse_single_team": len(team_errors) == 1,
                    "sparse_at_most_two_teams": len(team_errors) <= 2,
                    "historical_mae": float(
                        np.mean(
                            [np.mean(np.abs(errors)) for errors in team_errors.values()]
                        )
                    ),
                    "historical_bias": float(
                        np.mean([np.mean(errors) for errors in team_errors.values()])
                    ),
                }
            )
    write_csv(output / "empirical_replay.csv", empirical_replay)
    current_cases, down = (
        downstream(float(np.mean([v for vals in year_values.values() for v in vals])))
        if with_downstream
        else ([], [])
    )
    if current_cases:
        write_csv(output / "current_complete_one_missing.csv", current_cases)
    if down:
        write_csv(output / "downstream_one_missing.csv", down)
    else:
        for stale in ("current_complete_one_missing.csv", "downstream_one_missing.csv"):
            (output / stale).unlink(missing_ok=True)
    report = make_report(
        complete,
        excluded,
        cases,
        metrics,
        team_weighted,
        empirical,
        empirical_replay,
        empirical_support,
        current_cases,
        down,
        heldout_mean,
        heldout_sd,
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    metadata = {
        "issue": 142,
        "seed": SEED,
        "historical_seasons": [2021, 2022, 2023, 2024, 2025],
        "input_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                PLAYER_PATH,
                FEATURE_PATH,
                AVAILABILITY_PATH,
                CURRENT_PLAYER_PATH,
                CURRENT_FEATURE_PATH,
                CURRENT_MODEL_PATH,
                CURRENT_PREDICTIONS_PATH,
            )
        },
        "complete_team_seasons": len(complete),
        "masked_team_seasons": sum(len(v) >= 2 for v in complete.values()),
        "mask_cases": len(cases),
        "excluded": dict(excluded),
        "heldout_player_mean_by_season": heldout_mean,
        "heldout_player_sd_by_season": heldout_sd,
        "downstream_model": "frozen 2026 Context 1.3 model and published PMFs; rank-CDF interpolation of location shifts; no refit"
        if with_downstream
        else "disabled",
    }
    (output / "summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def make_report(
    complete,
    excluded,
    cases,
    metrics,
    team_weighted,
    empirical,
    empirical_replay,
    empirical_support,
    current_cases,
    down,
    means,
    sds,
) -> str:
    def row(policy: str, subset: list[dict[str, object]]) -> tuple[float, float, float]:
        items = [r for r in subset if r["policy"] == policy]
        e = np.array([float(r["estimate"]) - float(r["full_impact"]) for r in items])
        return (
            float(np.mean(np.abs(e))),
            float(np.mean(e)),
            float(np.mean(np.abs(e) > 1)),
        )

    lines = [
        "# Partial DB transfer-impact coverage study (issue #142)",
        "",
        "## Design",
        "",
        f"Ground truth is {len(complete)} complete incoming-DB team-seasons in the 2021–2025 frozen retrospective Context panel; {sum(len(v) >= 2 for v in complete.values())} have at least two players and can retain a nonempty observed subset after masking. {len(cases)} team-level masks were evaluated. The full player sum is checked against the committed Context feature before inclusion. Exclusions: {dict(excluded)}. Only DB impacts are in scope; offensive usage is a separate feature.",
        "",
        "For each complete roster, masks keep the roster fixed and hide one or more player impacts. Small combinations are exhaustive; strata with more than 80 masks retain the first and last plus a deterministic seeded sample. Policies: current all-or-nothing zero with availability 0; observed sum; observed sum multiplied by total/observed count; observed sum plus missing count times the other seasons' complete-player mean. The last policy never replaces known contributions. The primary table weights masks equally; a sensitivity analysis first averages within each historical (team-season, exact n/k coverage) cell and then gives those cells equal weight. These are feature-recovery diagnostics under controlled masking, not real missingness bias or predictive accuracy.",
        "",
        "## Feature recovery",
        "",
        "The machine-readable `policy_summary.csv` includes signed bias, MAE, RMSE, correlation with the full feature, frequency of errors above one impact unit, and fraction of masks beating the all-or-nothing zero. It breaks results out by exact coverage, coverage band, and missing-contribution magnitude and direction.",
        "",
        "| Coverage | Masks | Policy | MAE | Bias | P(|error| > 1) |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for coverage in ("high_80_plus", "moderate_50_to_80", "low_under_50"):
        subset = [r for r in metrics if r["coverage_band"] == coverage]
        for policy in POLICIES:
            mae, bias, large = row(policy, subset)
            lines.append(
                f"| {coverage} | {len(subset) // 4} | {policy} | {mae:.3f} | {bias:+.3f} | {large:.1%} |"
            )
    weighted_lookup = {
        (r["stratum_type"], r["stratum"], r["policy"]): r for r in team_weighted
    }
    lines += [
        "",
        "### Equal historical team-season/coverage weight",
        "",
        "Each exact n/k mask set for one historical team-season contributes one average error before broad coverage aggregation. This limits the influence of rosters with many mask combinations. A team with several distinct coverage levels can contribute once to each level.",
        "",
        "| Coverage | Team-season/coverage cells | Policy | MAE | P(|error| > 1) | Beats zero |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for band in ("high_80_plus", "moderate_50_to_80", "low_under_50"):
        for policy in POLICIES:
            item = weighted_lookup[("coverage_band", band, policy)]
            lines.append(
                f"| {band} | {item['team_coverage_cells']} | {policy} | {item['mae']:.3f} | {item['large_error_gt_1']:.1%} | {item['wins_vs_zero_fraction']:.1%} |"
            )
    lines += [
        "",
        "Exact high-coverage strata:",
        "",
        "| Coverage | Masks | Baseline MAE | Observed MAE | Scaled MAE | Missing-mean MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fraction in ("4/5", "7/8", "9/10", "11/12"):
        subset = [r for r in metrics if r["coverage"] == fraction]
        if subset:
            lines.append(
                f"| {fraction} | {len(subset) // 4} | "
                + " | ".join(f"{row(p, subset)[0]:.3f}" for p in POLICIES)
                + " |"
            )
        else:
            lines.append(
                f"| {fraction} | 0 | unavailable | unavailable | unavailable | unavailable |"
            )
    lines += [
        "",
        "There is no complete ten-player DB team-season in the 2021–2025 historical panel. The 7/8 result comes from two complete eight-player rosters; interpret it as directional, not a precise population estimate.",
        "",
        "## Consequential missing contributions",
        "",
        "| Hidden contribution | Masks | Baseline MAE | Observed MAE | Scaled MAE | Missing-mean MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for magnitude in ("small_le_1", "large_gt_1"):
        for direction in ("positive", "negative"):
            subset = [
                r
                for r in metrics
                if r["missing_magnitude"] == magnitude
                and r["missing_direction"] == direction
            ]
            if subset:
                lines.append(
                    f"| {magnitude}, {direction} | {len(subset) // 4} | "
                    + " | ".join(f"{row(p, subset)[0]:.3f}" for p in POLICIES)
                    + " |"
                )
    player_values = np.array([v for vals in complete.values() for v in vals])
    lower_tail, upper_tail = np.quantile(player_values, [0.10, 0.90])
    lines += [
        "",
        f"For one hidden player in the bottom or top decile of complete-player impact (cutoffs {lower_tail:.3f} and {upper_tail:.3f}):",
        "",
        "| Hidden player impact | Masks | Baseline MAE | Observed MAE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for direction in ("top_decile", "bottom_decile"):
        selected = [
            r
            for r in metrics
            if r["missing_count"] == 1
            and (
                r["hidden_contribution"] >= upper_tail
                if direction == "top_decile"
                else r["hidden_contribution"] <= lower_tail
            )
        ]
        if selected:
            lines.append(
                f"| {direction} | {len(selected) // 4} | {row('all_or_nothing', selected)[0]:.3f} | {row('observed', selected)[0]:.3f} |"
            )
        else:
            lines.append(f"| {direction} | 0 | unavailable | unavailable |")
    lines += [
        "",
        "## Missing-part uncertainty check",
        "",
        "A secondary uncertainty proxy assigns the missing contribution a leave-season-out player mean and standard deviation times the square root of missing count. The known sum stays fixed. This assumes independent missing players, so empirical interval coverage is checked below rather than presumed. The interval is not a fitted Context posterior and is not recommended for production without calibration.",
        "",
        "| Coverage | Masks | Nominal 90% interval coverage | Mean half-width |",
        "| --- | ---: | ---: | ---: |",
    ]
    for band in ("high_80_plus", "moderate_50_to_80", "low_under_50"):
        selected = [r for r in cases if r["coverage_band"] == band]
        lines.append(
            f"| {band} | {len(selected)} | {np.mean([r['missing_90_interval_covers'] for r in selected]):.1%} | {np.mean([1.645 * r['missing_sd_proxy'] for r in selected]):.3f} |"
        )
    partial = [r for r in empirical if r["partial"]]
    exact = sum(r["maskable_exact_n"] for r in partial)
    lines += [
        "",
        "## Empirical coverage and limitations",
        "",
        f"The separate issue #141 audit identifies {len(partial)} partially observed 2026 DB teams among {len(empirical)} teams with incoming DB players; {exact} partial teams have a roster size represented by at least one complete historical maskable team. The committed `empirical_2026_coverage.csv` records each actual n/k. This is a coverage comparison only: masking complete teams uniformly cannot establish how unresolved identities, origin coverage, or position conflicts select high-impact players. The retrospective source snapshots also do not establish August 15 availability.",
        "",
        "## Downstream Context response",
        "",
    ]
    support_lines = [
        "Exact historical support for each 2026 n/k cell with at least one observed player is shown below. Cells backed by only one or two historical team-seasons are flagged even if they contain many masks or are reused for several current teams.",
        "",
        "| Coverage | 2026 teams | Historical team-seasons | Historical masks | Support |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for cell in empirical_support:
        support_lines.append(
            f"| {cell['coverage']} | {cell['production_teams']} | {cell['historical_team_seasons']} | {cell['historical_masks']} | {cell['support_status']} |"
        )
    support_lines.append("")
    lines[-2:-2] = support_lines
    matched = len({r["team_id"] for r in empirical_replay})
    if empirical_replay:
        well_supported = [
            r for r in empirical_replay if r["historical_team_seasons"] >= 2
        ]
        well_supported_teams = len({r["team_id"] for r in well_supported})
        lines.insert(
            -2,
            f"The exact n/k empirical replay matches {matched} of {len(partial)} partially covered 2026 teams with observed values. Equal weighting by those teams gives "
            + ", ".join(
                f"{policy} MAE {np.mean([r['historical_mae'] for r in empirical_replay if r['policy'] == policy]):.3f}"
                for policy in POLICIES
            )
            + ". Each historical cell MAE is averaged across its distinct team-seasons before current teams are weighted. This transports historical masking errors to today's coverage frequencies; it cannot correct selection bias in which players are missing.",
        )
        lines.insert(-2, "")
        if well_supported:
            lines.insert(
                -2,
                f"Excluding single-team historical cells leaves {well_supported_teams} current teams; their equal-team replay gives "
                + ", ".join(
                    f"{policy} MAE {np.mean([r['historical_mae'] for r in well_supported if r['policy'] == policy]):.3f}"
                    for policy in POLICIES
                )
                + ". The excluded cells are diagnostic only and should not set a coverage threshold.",
            )
            lines.insert(-2, "")
    ten = [r for r in current_cases if r["coverage"] == "9/10"]
    if ten:
        lines.insert(
            -2,
            f"One complete 2026 ten-player DB roster permits a separate 9/10 one-missing sensitivity ({len(ten)} masks): "
            + ", ".join(
                f"{policy} MAE {np.mean([abs(r[policy] - r['full_impact']) for r in ten]):.3f}"
                for policy in POLICIES
            )
            + ". This is a current-season feature reconstruction, not a historical outcome test.",
        )
        lines.insert(-2, "")
    if down:
        lines += [
            "The frozen 2026 Context 1.3 model and its committed preseason PMFs were used for one-missing masks on complete 2026 DB teams. Because the historical rank-distribution input panel is absent from this checkout, the exact posterior replay is unavailable: the model's fitted DB impact and binary availability location coefficients translate each saved rank PMF by interpolating its CDF on the rank-logit axis. The DB scale coefficients are zero in this fit; a new partial-aware indicator has no fitted coefficient. These are approximate local sensitivity diagnostics, not out-of-sample ranking scores or newly published predictions. Ordinal displacement compares the changed team with all other 2026 teams held at their published expected ranks.",
            "",
            "| Policy | Cases | Mean absolute expected-rank shift | Mean PMF total variation | Mean absolute ordinal shift |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for policy in POLICIES:
            items = [r for r in down if r["policy"] == policy]
            lines.append(
                f"| {policy} | {len(items)} | {np.mean([abs(r['expected_rank_delta']) for r in items]):.3f} | {np.mean([r['pmf_total_variation'] for r in items]):.4f} | {np.mean([abs(r['ordinal_displacement']) for r in items]):.3f} |"
            )
    else:
        lines.append("Downstream diagnostic disabled by runner option.")
    lines += [
        "",
        "## Recommendation",
        "",
        f"Preserve the observed incoming-DB impact sum whenever at least one relevant player's impact resolves, including at low coverage; zero remains a valid observed contribution. This is a representation recommendation, not a claim that Context should trust a 1/n partial value as strongly as a complete sum. In the low-coverage mask-weighted band, observed-sum MAE falls from 2.342 to 1.985, but P(|error| > 1) rises from 55.6% to 65.0% and observed beats zero on only 54.8% of masks. Under equal historical team-season/coverage weight, the same tail frequency falls from {weighted_lookup[('coverage_band', 'low_under_50', 'all_or_nothing')]['large_error_gt_1']:.1%} to {weighted_lookup[('coverage_band', 'low_under_50', 'observed')]['large_error_gt_1']:.1%}; the tail comparison depends on weighting. The missing-part 90% proxy covers only about 80% there. The model should account for coverage and learn or calibrate reduced confidence before a production change; the study does not establish that simply swapping numeric values into frozen Context improves low-coverage posteriors. Do not scale by count or fill the entire team feature. Keep the unobserved portion unknown pending a separately validated uncertainty model. For no observed players, keep impact unavailable and numeric neutral zero. For no incoming DB transfers, retain the current natural zero with complete coverage. The results do not justify a hard fractional cutoff: average recovery and tail risk move differently, and sparse exact n/k cells cannot locate a stable break point.",
        "",
        "Persist `incoming_db_count`, `observed_db_impact_count`, and `observed_db_impact_sum` with source/season provenance. Derive missing count and fraction from the first two counts. These fields distinguish complete, partial, and absent coverage; zero incoming is a complete natural zero. Keep player-level contributor/status audit links for traceability. The existing binary availability flag cannot express partial coverage, and changing the numeric feature while retaining frozen coefficients changes the model's input semantics. Treat adoption as a new Context model version with a trained and validated coverage-aware contract, not a silent Context 1.3 clarification. No production behavior or published rankings changed in this issue.",
        "",
        "## Reproduce",
        "",
        "`UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/study_partial_db_transfer_impact.py` writes this report and the machine-readable masks, errors, coverage, downstream diagnostics, and input hashes. `summary.json` records the seed and leave-season-out player means. The downstream response is an interpolation approximation, not a causal estimate of future-season performance.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-downstream", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(run(args.output, with_downstream=not args.skip_downstream), indent=2)
    )


if __name__ == "__main__":
    main()
