"""Reproduce the issue 154 descriptive Context 1.3 location-error study.

All inputs come from the committed issue 153 diagnostic artifact. No model is fit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/context_location_error_diagnostics"
OUT = ROOT / "data/processed/context_location_error_study"
SEASONS = (2022, 2023, 2024, 2025)
PRIMARY = (2023, 2024, 2025)
OUTCOMES = (
    "context_preseason_abs_rank_error",
    "context_minus_history_abs_rank_error",
    "context_minus_history_final_nll",
)
HISTORY_FEATURES = frozenset(("lag2_z_mean", "lag3_z_mean", "long_run_z_mean"))
TRANSFER_FEATURES = (
    "transfer_in_prior_usage_sum",
    "transfer_in_prior_defensive_impact_db_sum",
    "transfer_in_prior_defensive_impact_db_available",
)
TRANSFER_METADATA = (
    "returning_production",
    "incoming_transfer_count",
    "incoming_offensive_applicable_count",
    "incoming_offensive_observed_usage_count",
    "incoming_offensive_observed_usage_sum",
    "incoming_offensive_applicability_unknown_count",
    "incoming_db_count",
    "observed_db_impact_count",
    "observed_db_impact_sum",
    "db_coverage_fraction",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quartiles(frame: pd.DataFrame, column: str) -> pd.Series:
    """Four balanced within-season bins; team ID breaks exact value ties."""
    result = pd.Series(index=frame.index, dtype="int64")
    for _, part in frame.groupby("season", sort=True):
        ordered = part.sort_values(
            [column, "team_id"], kind="stable", na_position="last"
        )
        if ordered[column].isna().any():
            raise ValueError(f"{column} is missing in a quantile population")
        result.loc[ordered.index] = (
            np.floor(4 * np.arange(len(ordered)) / len(ordered)) + 1
        )
    return result.astype(int)


def validate_input(frame: pd.DataFrame) -> None:
    if len(frame) != 534 or frame.groupby("season").size().to_dict() != {
        2022: 131,
        2023: 133,
        2024: 134,
        2025: 136,
    }:
        raise ValueError("issue 153 population changed")
    if frame.duplicated(["season", "team_id"]).any():
        raise ValueError("duplicate team-season")
    if frame.component_status.value_counts().to_dict() != {
        "fitted": 528,
        "cold_start_fallback": 6,
    }:
        raise ValueError("fitted/fallback population changed")
    checks = (
        (
            "context_preseason_abs_rank_error",
            (frame.context_preseason_expected_rank - frame.target_expected_rank).abs(),
        ),
        (
            "history_preseason_abs_rank_error",
            (frame.history_preseason_expected_rank - frame.target_expected_rank).abs(),
        ),
        (
            "context_minus_history_abs_rank_error",
            frame.context_preseason_abs_rank_error
            - frame.history_preseason_abs_rank_error,
        ),
        (
            "context_minus_history_final_nll",
            frame.context_final_nll - frame.history_final_nll,
        ),
        (
            "context_minus_history_preseason_expected_rank",
            frame.context_preseason_expected_rank
            - frame.history_preseason_expected_rank,
        ),
    )
    for column, expected in checks:
        if not np.allclose(frame[column], expected, atol=1e-8, rtol=0):
            raise ValueError(f"orientation/parity failed: {column}")


def assign_groups(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["error_quartile"] = quartiles(frame, "context_preseason_abs_rank_error")
    frame["nll_gap_quartile"] = quartiles(frame, "context_minus_history_final_nll")
    frame["disagreement_quartile"] = quartiles(
        frame, "abs_context_minus_history_preseason_expected_rank"
    )
    frame["expensive_miss"] = (frame.error_quartile == 4) & (
        frame.nll_gap_quartile == 4
    )
    frame["large_miss_recovered"] = (frame.error_quartile == 4) & (
        frame.context_minus_history_final_nll <= 0
    )
    frame["helpful_disagreement"] = (frame.disagreement_quartile == 4) & (
        frame.context_minus_history_abs_rank_error < 0
    )
    frame["harmful_disagreement"] = (frame.disagreement_quartile == 4) & (
        frame.context_minus_history_abs_rank_error > 0
    )
    frame["group"] = np.select(
        [
            frame.expensive_miss,
            frame.large_miss_recovered,
            frame.helpful_disagreement,
            frame.harmful_disagreement,
        ],
        [
            "expensive miss",
            "large miss recovered",
            "helpful disagreement",
            "harmful disagreement",
        ],
        default="ordinary",
    )
    frame["season_mean_gap_contribution"] = (
        frame.context_minus_history_final_nll
        / frame.groupby("season").season.transform("size")
    )
    frame["primary_mean_gap_contribution"] = np.where(
        frame.season.isin(PRIMARY), frame.context_minus_history_final_nll / 403, np.nan
    )
    difference_columns = [
        f"{feature}_difference_reason" for feature in TRANSFER_FEATURES
    ]
    frame["historical_repair_state"] = (
        frame[difference_columns].bfill(axis=1).iloc[:, 0].fillna("none")
    )
    return frame


def concentration(frame: pd.DataFrame) -> list[dict]:
    result = []
    for period, part in periods(frame):
        ordered = part.sort_values(
            ["context_minus_history_final_nll", "season", "team_id"],
            ascending=[False, True, True],
        )
        total = float(part.context_minus_history_final_nll.sum())
        direct = float(part.context_final_nll.mean() - part.history_final_nll.mean())
        if not np.isclose(total / len(part), direct, atol=1e-12):
            raise ValueError("aggregate NLL contribution does not reconcile")
        for fraction in (0.05, 0.1, 0.2, 0.25, 1.0):
            n = int(np.ceil(len(part) * fraction))
            top = float(ordered.head(n).context_minus_history_final_nll.sum())
            result.append(
                {
                    "period": period,
                    "fraction": fraction,
                    "n": n,
                    "top_gap_sum": top,
                    "total_gap_sum": total,
                    "share_of_net_gap": top / total if total else None,
                }
            )
    return result


def periods(frame: pd.DataFrame):
    for season in SEASONS:
        yield str(season), frame.loc[frame.season == season]
    yield "2023-2025", frame.loc[frame.season.isin(PRIMARY)]


def contribution_diagnostics(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    fitted = frame.loc[frame.component_status == "fitted"].copy()
    for feature in features:
        fitted[f"total_{feature}"] = (
            fitted[f"feature_{feature}_contribution"]
            + fitted[f"feature_{feature}_missing_contribution"]
        )
    values = fitted[[f"total_{f}" for f in features]].to_numpy(dtype=float)
    largest = np.argmax(np.abs(values), axis=1)
    positive = np.maximum(values, 0)
    negative = np.minimum(values, 0)
    abs_sum = np.abs(values).sum(axis=1)
    max_abs = np.max(np.abs(values), axis=1)
    history_indices = [
        i for i, feature in enumerate(features) if feature in HISTORY_FEATURES
    ]
    context_indices = [
        i for i, feature in enumerate(features) if feature not in HISTORY_FEATURES
    ]
    fitted["largest_positive_feature"] = [
        features[i] if values[j, i] > 0 else ""
        for j, i in enumerate(np.argmax(values, axis=1))
    ]
    fitted["largest_positive_contribution"] = positive.max(axis=1)
    fitted["largest_negative_feature"] = [
        features[i] if values[j, i] < 0 else ""
        for j, i in enumerate(np.argmin(values, axis=1))
    ]
    fitted["largest_negative_contribution"] = negative.min(axis=1)
    fitted["largest_absolute_feature"] = [features[i] for i in largest]
    fitted["largest_absolute_contribution"] = values[np.arange(len(values)), largest]
    fitted["absolute_contribution_sum"] = abs_sum
    fitted["largest_absolute_share"] = np.divide(
        max_abs, abs_sum, out=np.zeros_like(max_abs), where=abs_sum > 0
    )
    fitted["substantial_positive_count"] = (values >= (max_abs * 0.25)[:, None]).sum(
        axis=1
    )
    fitted["substantial_negative_count"] = (values <= (-max_abs * 0.25)[:, None]).sum(
        axis=1
    )
    dominant = values[np.arange(len(values)), largest]
    fitted["opposing_substantial_count"] = (
        (values * dominant[:, None] < 0) & (np.abs(values) >= (max_abs * 0.25)[:, None])
    ).sum(axis=1)
    fitted["positive_contribution_sum"] = positive.sum(axis=1)
    fitted["negative_contribution_sum"] = negative.sum(axis=1)
    fitted["cancellation_fraction"] = 1 - np.abs(values.sum(axis=1)) / abs_sum
    fitted["history_derived_subtotal"] = (
        values[:, history_indices].sum(axis=1) + fitted.lag1_contribution
    )
    fitted["context_only_subtotal"] = values[:, context_indices].sum(axis=1)
    fitted["context_only_positive_sum"] = positive[:, context_indices].sum(axis=1)
    fitted["context_only_negative_sum"] = negative[:, context_indices].sum(axis=1)
    fitted["context_only_cancellation_fraction"] = 1 - np.abs(
        fitted.context_only_subtotal
    ) / (fitted.context_only_positive_sum - fitted.context_only_negative_sum)
    fitted["absolute_location_displacement"] = (
        fitted.context_minus_history_location_center.abs()
    )
    returning = fitted.total_returning_pct_ppa
    imported = fitted.total_transfer_in_prior_usage_sum
    fitted["continuity_import_opposition"] = np.where(
        returning * imported < 0, np.minimum(returning.abs(), imported.abs()), 0
    )
    fitted["history_context_opposition"] = np.where(
        fitted.history_derived_subtotal * fitted.context_only_subtotal < 0,
        np.minimum(
            fitted.history_derived_subtotal.abs(), fitted.context_only_subtotal.abs()
        ),
        0,
    )
    reconstructed = (
        fitted.context_location_intercept
        + fitted.history_derived_subtotal
        + fitted.context_only_subtotal
    )
    if not np.allclose(
        reconstructed, fitted.context_location_center, atol=1e-8, rtol=0
    ):
        raise ValueError("contribution subtotals do not reconstruct the center")
    columns = [
        "season",
        "team",
        "team_id",
        "group",
        "expensive_miss",
        "helpful_disagreement",
        "harmful_disagreement",
        *OUTCOMES,
        "season_mean_gap_contribution",
        "primary_mean_gap_contribution",
        "context_minus_history_location_center",
        "absolute_location_displacement",
        "largest_positive_feature",
        "largest_positive_contribution",
        "largest_negative_feature",
        "largest_negative_contribution",
        "largest_absolute_feature",
        "largest_absolute_contribution",
        "absolute_contribution_sum",
        "largest_absolute_share",
        "substantial_positive_count",
        "substantial_negative_count",
        "opposing_substantial_count",
        "positive_contribution_sum",
        "negative_contribution_sum",
        "cancellation_fraction",
        "history_derived_subtotal",
        "context_only_subtotal",
        "context_only_positive_sum",
        "context_only_negative_sum",
        "context_only_cancellation_fraction",
        "continuity_import_opposition",
        "history_context_opposition",
        "nearest_neighbor_distance",
        "feature_range_violation_count",
    ]
    return fitted[columns].sort_values(["season", "team_id"])


def correlation(x: pd.Series, y: pd.Series, method: str) -> float | None:
    joined = pd.concat([x, y], axis=1).dropna()
    if (
        len(joined) < 3
        or joined.iloc[:, 0].nunique() < 2
        or joined.iloc[:, 1].nunique() < 2
    ):
        return None
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1], method=method))


def feature_associations(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    fitted = frame.loc[frame.component_status == "fitted"].copy()
    for feature in features:
        fitted[f"combined_{feature}"] = (
            fitted[f"feature_{feature}_contribution"]
            + fitted[f"feature_{feature}_missing_contribution"]
        )
    measures = [
        (feature, representation, column)
        for feature in features
        for representation, column in (
            ("frozen_raw", f"feature_{feature}_raw"),
            ("fitted_contribution", f"combined_{feature}"),
        )
    ]
    measures += [
        (feature, "training_percentile", f"feature_{feature}_training_percentile")
        for feature in features
    ]
    measures += [
        ("training_support", "support_metric", column)
        for column in (
            "nearest_neighbor_distance",
            "mean_k_nearest_neighbor_distance",
            "feature_range_violation_count",
        )
    ]
    measures += [
        ("transfer_conditions", "descriptive_metadata", column)
        for column in TRANSFER_METADATA
    ]
    for feature, representation, column in measures:
        for outcome in OUTCOMES:
            for period, part in periods(fitted):
                x, y = part[column], part[outcome]
                joined = (
                    part[[column, outcome, "team_id", "season"]]
                    .dropna(subset=[column, outcome])
                    .sort_values([column, "team_id"], kind="stable")
                )
                if len(joined):
                    joined = joined.assign(bin=quartiles(joined, column))
                    bottom, top = (
                        joined.loc[joined.bin == 1, outcome],
                        joined.loc[joined.bin == 4, outcome],
                    )
                    decile = max(1, int(np.ceil(len(joined) * 0.1)))
                else:
                    bottom = top = pd.Series(dtype=float)
                    decile = 0
                standardized = []
                for _, season_part in part.groupby("season"):
                    valid = season_part[[column, outcome]].dropna()
                    sx, sy = valid[column].std(ddof=0), valid[outcome].std(ddof=0)
                    if sx > 0 and sy > 0:
                        standardized.append(
                            pd.DataFrame(
                                {
                                    "x": (valid[column] - valid[column].mean()) / sx,
                                    "y": (valid[outcome] - valid[outcome].mean()) / sy,
                                }
                            )
                        )
                z = (
                    pd.concat(standardized, ignore_index=True)
                    if standardized
                    else pd.DataFrame(columns=["x", "y"])
                )
                rows.append(
                    {
                        "feature": feature,
                        "representation": representation,
                        "measure": column,
                        "outcome": outcome,
                        "period": period,
                        "population_n": len(part),
                        "observed_n": len(joined),
                        "missing_n": len(part) - len(joined),
                        "pearson": correlation(x, y, "pearson"),
                        "spearman": correlation(x, y, "spearman"),
                        "within_season_standardized_pearson": correlation(
                            z.x, z.y, "pearson"
                        ),
                        "bottom_quartile_n": len(bottom),
                        "bottom_quartile_mean": bottom.mean(),
                        "bottom_quartile_median": bottom.median(),
                        "top_quartile_n": len(top),
                        "top_quartile_mean": top.mean(),
                        "top_quartile_median": top.median(),
                        "bottom_decile_n": decile,
                        "bottom_decile_mean": joined.head(decile)[outcome].mean(),
                        "top_decile_n": decile,
                        "top_decile_mean": joined.tail(decile)[outcome].mean(),
                    }
                )
    return pd.DataFrame(rows)


def summarize_subset(
    part: pd.DataFrame, period: str, family: str, bin_name: str
) -> dict:
    return {
        "period": period,
        "family": family,
        "bin": str(bin_name),
        "n": len(part),
        "context_abs_error_mean": part.context_preseason_abs_rank_error.mean(),
        "history_abs_error_mean": part.history_preseason_abs_rank_error.mean(),
        "error_difference_mean": part.context_minus_history_abs_rank_error.mean(),
        "final_nll_gap_mean": part.context_minus_history_final_nll.mean(),
        "final_nll_gap_sum": part.context_minus_history_final_nll.sum(),
        "context_better_fraction": (
            part.context_minus_history_abs_rank_error < 0
        ).mean(),
        "expensive_miss_n": part.expensive_miss.sum(),
        "expensive_miss_rate": part.expensive_miss.mean(),
    }


def grouped_diagnostics(
    frame: pd.DataFrame, contributions: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    fitted = frame.loc[frame.component_status == "fitted"].copy()
    contribution_columns = (
        "absolute_location_displacement",
        "largest_absolute_share",
        "cancellation_fraction",
        "context_only_cancellation_fraction",
        "context_only_subtotal",
        "context_only_positive_sum",
        "continuity_import_opposition",
        "history_context_opposition",
    )
    fitted = fitted.merge(
        contributions[["season", "team_id", *contribution_columns]],
        on=["season", "team_id"],
        validate="one_to_one",
    )
    fitted["support_quartile"] = quartiles(fitted, "nearest_neighbor_distance")
    fitted["mean_k_support_quartile"] = quartiles(
        fitted, "mean_k_nearest_neighbor_distance"
    )
    for period, part in periods(frame):
        for label, sub in part.groupby("disagreement_quartile"):
            rows.append(summarize_subset(sub, period, "disagreement_quartile", label))
        for label, sub in part.groupby("group"):
            rows.append(summarize_subset(sub, period, "exclusive_group", label))
    for period, part in periods(fitted):
        for family, column in (
            ("nearest_support_quartile", "support_quartile"),
            ("mean_k_support_quartile", "mean_k_support_quartile"),
            ("feature_range_violations", "feature_range_violation_count"),
            ("db_coverage", "db_coverage_status"),
            (
                "unknown_offensive_applicability",
                "incoming_offensive_applicability_unknown_count",
            ),
            ("historical_transfer_input_difference", "historical_repair_state"),
        ):
            for label, sub in part.groupby(column, dropna=False):
                rows.append(summarize_subset(sub, period, family, label))
        for column in (
            "incoming_transfer_count",
            "returning_production",
            "incoming_offensive_observed_usage_sum",
            "observed_db_impact_sum",
        ):
            observed = part.loc[part[column].notna()].copy()
            if len(observed):
                observed["bin"] = quartiles(observed, column)
                for label, sub in observed.groupby("bin"):
                    rows.append(
                        summarize_subset(sub, period, f"{column}_quartile", label)
                    )
        for column in contribution_columns:
            observed = part.copy()
            observed["bin"] = quartiles(observed, column)
            for label, sub in observed.groupby("bin"):
                rows.append(summarize_subset(sub, period, f"{column}_quartile", label))
        for feature in pd.read_csv(SOURCE / "feature_inventory.csv").feature_name:
            column = f"feature_{feature}_training_range"
            for label, sub in part.groupby(column):
                rows.append(
                    summarize_subset(sub, period, f"{feature}_training_range", label)
                )
    return pd.DataFrame(rows).sort_values(["family", "period", "bin"])


def case_studies(
    frame: pd.DataFrame, contributions: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    fitted = frame.loc[frame.component_status == "fitted"].copy()
    subsets = [
        (
            "largest_context_error",
            frame.nlargest(4, "context_preseason_abs_rank_error"),
        ),
        ("largest_final_gap", frame.nlargest(4, "context_minus_history_final_nll")),
        (
            "helpful_disagreement",
            frame.loc[frame.helpful_disagreement].nlargest(
                4, "abs_context_minus_history_preseason_expected_rank"
            ),
        ),
        (
            "harmful_disagreement",
            frame.loc[frame.harmful_disagreement].nlargest(
                4, "abs_context_minus_history_preseason_expected_rank"
            ),
        ),
        ("high_extrapolation", fitted.nlargest(4, "nearest_neighbor_distance")),
        (
            "transfer_repair",
            fitted.loc[
                fitted.transfer_repair_class.isin(
                    ("ambiguous_usage_join_removed", "name_normalization_join_added")
                )
            ],
        ),
        (
            "transfer_repair",
            fitted.loc[(fitted.team == "North Texas") & (fitted.season == 2023)],
        ),
        (
            "transfer_repair",
            fitted.loc[(fitted.team == "Coastal Carolina") & (fitted.season == 2023)],
        ),
    ]
    selected = {}
    for reason, sub in subsets:
        for row in sub.itertuples():
            key = (row.season, row.team_id)
            selected.setdefault(key, []).append(reason)
    indexed = frame.set_index(["season", "team_id"])
    contribution_index = contributions.set_index(["season", "team_id"])
    result = []
    for key, reasons in sorted(selected.items()):
        row = indexed.loc[key]
        info = {
            "season": key[0],
            "team": row.team,
            "team_id": key[1],
            "selection_reason": ";".join(dict.fromkeys(reasons)),
            "component_status": row.component_status,
            "group": row.group,
        }
        for column in (
            "context_preseason_expected_rank",
            "history_preseason_expected_rank",
            "target_expected_rank",
            "context_preseason_abs_rank_error",
            "history_preseason_abs_rank_error",
            "context_minus_history_abs_rank_error",
            "context_final_nll",
            "history_final_nll",
            "context_minus_history_final_nll",
            "context_location_center",
            "history_location_center",
            "nearest_neighbor_distance",
            "mean_k_nearest_neighbor_distance",
            "feature_range_violation_count",
            "db_coverage_status",
            "db_coverage_fraction",
            "historical_repair_state",
            "transfer_repair_class",
            "transfer_checkpoint_status",
            "transfer_provenance_class",
            "transfer_in_prior_usage_sum_model_input_value",
            "transfer_in_prior_usage_sum_corrected_diagnostic_value",
            "transfer_in_prior_usage_sum_difference_timing_status",
        ):
            info[column] = row[column]
        if key in contribution_index.index:
            values = [
                (
                    f,
                    float(
                        row[f"feature_{f}_contribution"]
                        + row[f"feature_{f}_missing_contribution"]
                    ),
                )
                for f in features
            ]
            ordered = sorted(values, key=lambda pair: -abs(pair[1]))
            dominant = ordered[0][1]
            info["largest_fitted_contributions"] = "; ".join(
                f"{name}:{value:+.3f}" for name, value in ordered[:5]
            )
            info["major_opposing_contributions"] = "; ".join(
                f"{name}:{value:+.3f}"
                for name, value in ordered
                if value * dominant < 0 and abs(value) >= 0.25 * abs(dominant)
            )
        result.append(info)
    return pd.DataFrame(result)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(
        path, index=False, float_format="%.12g", lineterminator="\n", na_rep=""
    )


def main() -> None:
    source = SOURCE / "team_seasons.csv"
    inventory = SOURCE / "feature_inventory.csv"
    frame = pd.read_csv(source, keep_default_na=False, na_values=[""])
    features = pd.read_csv(inventory).feature_name.tolist()
    validate_input(frame)
    frame = assign_groups(frame)
    contributions = contribution_diagnostics(frame, features)
    associations = feature_associations(frame, features)
    grouped = grouped_diagnostics(frame, contributions)
    cases = case_studies(frame, contributions, features)
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "feature_associations.csv": associations,
        "contribution_diagnostics.csv": contributions,
        "grouped_diagnostics.csv": grouped,
        "case_studies.csv": cases,
    }
    for name, output in outputs.items():
        write_csv(OUT / name, output)
    summary = {
        "population": {
            period: {
                "all": len(part),
                "fitted": int((part.component_status == "fitted").sum()),
                "fallback": int((part.component_status == "cold_start_fallback").sum()),
                "expensive_misses": int(part.expensive_miss.sum()),
                "mean_context_abs_error": float(
                    part.context_preseason_abs_rank_error.mean()
                ),
                "mean_error_difference": float(
                    part.context_minus_history_abs_rank_error.mean()
                ),
                "median_error_difference": float(
                    part.context_minus_history_abs_rank_error.median()
                ),
                "mean_final_nll_gap": float(
                    part.context_minus_history_final_nll.mean()
                ),
                "median_final_nll_gap": float(
                    part.context_minus_history_final_nll.median()
                ),
                "sum_final_nll_gap": float(part.context_minus_history_final_nll.sum()),
            }
            for period, part in periods(frame)
        },
        "concentration": concentration(frame),
        "group_definitions": {
            "expensive_miss": "within-season top quartile Context absolute error AND top quartile final NLL disadvantage",
            "large_miss_recovered": "within-season top quartile Context absolute error AND final NLL gap <= 0",
            "material_disagreement": "within-season top quartile absolute Context-History expected-rank disagreement",
            "helpful_harmful": "material disagreement with negative/positive preseason absolute-error difference",
            "exclusive_precedence": "expensive miss, large miss recovered, helpful disagreement, harmful disagreement, ordinary",
            "quartiles": "balanced within season; ascending value with team ID breaking exact ties",
        },
        "historical_repair_states": frame.loc[
            frame.component_status == "fitted", "historical_repair_state"
        ]
        .value_counts()
        .to_dict(),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    provenance = {
        "study": "issue-154 descriptive Context location-error study",
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                source,
                inventory,
                SOURCE / "summary.json",
                SOURCE / "provenance.json",
            )
        },
        "source": "committed issue-153 artifact only",
        "weighting": "equal team-season",
        "transfer_timing": "repaired historical transfer evidence is descriptive; August 15 historical availability unverified",
        "analysis_code_sha256": sha256(Path(__file__)),
        "output_hashes": {
            name: sha256(OUT / name) for name in (*outputs, "summary.json")
        },
    }
    (OUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
