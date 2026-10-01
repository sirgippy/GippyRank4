"""Join the frozen #147 comparison with exact rolling Context 1.3 inputs.

Run with the same historical modeling corpus used for #147, for example::

    uv run python scripts/build_context_location_error_diagnostics.py \
        --targets /path/to/data/processed/modeling/team_season_rank_distributions.csv

The repaired transfer panel is descriptive evidence only. It never enters a fit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study_context_history_crossover as crossover

from gippyrank.context_prior_v1_3 import H_FEATURES, LOCATION_FEATURE_NAMES
from gippyrank.preseason import QUADRATURE_POINTS, product_quadrature

BASE = ROOT / "data/processed/context_history_crossover"
REPAIR = ROOT / "data/processed/transfer_data_repair"
HISTORICAL_CHANGES = REPAIR / "historical_feature_changes.csv"
DB_AUDIT = ROOT / "data/processed/defensive_transfer_audit/transfer_player_audit.csv"
OUT = ROOT / "data/processed/context_location_error_diagnostics"
REPORT = ROOT / "docs/context_location_error_diagnostic_dataset.md"
K = 5
TOL = 1e-8
TRANSFER_FEATURES = (
    "transfer_in_prior_usage_sum",
    "transfer_in_prior_defensive_impact_db_sum",
    "transfer_in_prior_defensive_impact_db_available",
)
HISTORICAL_CHANGE_CLASSES = frozenset(
    {
        "legitimate_zero_restoration",
        "ambiguous_usage_join_removed",
        "name_normalization_join_added",
    }
)

# These are descriptions, never the source of feature membership or ordering.
DESCRIPTIONS = {
    "lag2_z_mean": (
        "Mean latent rank in season t-2",
        "historical rank distributions",
        "prior-season distribution mean",
    ),
    "lag3_z_mean": (
        "Mean latent rank in season t-3",
        "historical rank distributions",
        "prior-season distribution mean",
    ),
    "long_run_z_mean": (
        "Long-run mean latent rank",
        "historical rank distributions",
        "historical aggregation",
    ),
    "coach_tenure_seasons": (
        "Head-coach tenure in seasons",
        "CFBD coach tenure",
        "season count",
    ),
    "recruiting_class_rank": (
        "Current recruiting class rank",
        "CFBD recruiting",
        "rank",
    ),
    "recruiting_class_points": (
        "Current recruiting class points",
        "CFBD recruiting",
        "points",
    ),
    "recruiting_points_2y_mean": (
        "Two-year mean recruiting points",
        "CFBD recruiting",
        "two-year mean",
    ),
    "recruiting_points_3y_mean": (
        "Three-year mean recruiting points",
        "CFBD recruiting",
        "three-year mean",
    ),
    "recruiting_points_4y_mean": (
        "Four-year mean recruiting points",
        "CFBD recruiting",
        "four-year mean",
    ),
    "recruiting_points_trend": ("Recruiting points trend", "CFBD recruiting", "trend"),
    "talent_composite": ("Roster talent composite", "CFBD talent", "composite"),
    "returning_pct_ppa": (
        "Returning production PPA fraction",
        "CFBD returning production",
        "fraction",
    ),
    "transfer_in_prior_usage_sum": (
        "Incoming applicable prior offensive usage",
        "frozen retrospective transfer reconstruction",
        "sum",
    ),
    "transfer_in_prior_defensive_impact_db_sum": (
        "Incoming DB prior defensive impact",
        "frozen retrospective transfer reconstruction",
        "sum or neutral zero",
    ),
    "transfer_in_prior_defensive_impact_db_available": (
        "DB impact sum availability",
        "frozen retrospective transfer reconstruction",
        "binary flag",
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row: dict[str, str]) -> tuple[int, str]:
    return int(row["season"]), row["team_id"]


def optional_float(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def same_optional_float(left: float | None, right: float | None) -> bool:
    return (left is None and right is None) or (
        left is not None
        and right is not None
        and bool(np.isclose(left, right, rtol=0, atol=1e-12))
    )


def index_historical_feature_changes(
    records: list[dict[str, str]],
) -> dict[tuple[int, str, str], dict[str, str]]:
    """Index the #151 inventory, rejecting ambiguous or invalid source rows."""
    indexed: dict[tuple[int, str, str], dict[str, str]] = {}
    for record in records:
        change_key = (
            int(record["season"]),
            record["team_id"],
            record["feature_name"],
        )
        if change_key in indexed:
            raise ValueError(
                f"duplicate historical feature-change record: {change_key}"
            )
        if record["feature_name"] not in TRANSFER_FEATURES:
            raise ValueError(f"unknown historical transfer feature: {change_key}")
        if record["change_class"] not in HISTORICAL_CHANGE_CLASSES:
            raise ValueError(f"unknown historical change class: {change_key}")
        old_value = optional_float(record["old_value"])
        new_value = optional_float(record["new_value"])
        if any(
            value is not None and not np.isfinite(value)
            for value in (old_value, new_value)
        ):
            raise ValueError(f"nonfinite historical feature-change value: {change_key}")
        if same_optional_float(old_value, new_value):
            raise ValueError(f"contradictory historical no-change record: {change_key}")
        indexed[change_key] = record
    return indexed


def reconciled_change_class(
    change_key: tuple[int, str, str],
    frozen: float | None,
    corrected: float | None,
    changes: dict[tuple[int, str, str], dict[str, str]],
) -> str:
    """Every fitted value difference must have exactly one matching #151 record."""
    record = changes.get(change_key)
    changed = not same_optional_float(frozen, corrected)
    if record is None:
        if changed:
            raise ValueError(f"missing historical feature-change record: {change_key}")
        return ""
    if not changed:
        raise ValueError(
            f"historical change record contradicts equal values: {change_key}"
        )
    if not same_optional_float(
        frozen, optional_float(record["old_value"])
    ) or not same_optional_float(corrected, optional_float(record["new_value"])):
        raise ValueError(
            f"historical feature-change values contradict inputs: {change_key}"
        )
    return record["change_class"]


def inventory() -> list[dict[str, object]]:
    names = tuple(
        crossover.hcp.c13.model_specification_metadata()["location_feature_names"]
    )
    if names != LOCATION_FEATURE_NAMES or set(names) != set(DESCRIPTIONS):
        raise ValueError("Context 1.3 inventory differs from fitting implementation")
    result = []
    for name in names:
        description, source, transformation = DESCRIPTIONS[name]
        result.append(
            {
                "feature_name": name,
                "semantic_description": description,
                "source": source,
                "transformation": transformation,
                "normalization": "training-only median imputation, then (value - training mean) / training SD",
                "location_or_scale": "location_and_scale"
                if name in H_FEATURES
                else "location",
                "transfer_derived": name.startswith("transfer_"),
                "also_used_by_history": name in H_FEATURES,
                "missingness_semantics": "separate fitted indicator; missing numeric value receives training median",
            }
        )
    return result


def percentile_and_range(value: float, training: np.ndarray) -> tuple[float, str]:
    """Midrank empirical percentile and range from training observations only."""
    if not len(training):
        raise ValueError("no training support")
    percentile = (
        100
        * (
            np.count_nonzero(training < value)
            + 0.5 * np.count_nonzero(training == value)
        )
        / len(training)
    )
    status = (
        "below training minimum"
        if value < np.min(training)
        else (
            "above training maximum"
            if value > np.max(training)
            else "within training range"
        )
    )
    return float(percentile), status


def nearest_support(
    target: np.ndarray, training: np.ndarray, k: int = K
) -> tuple[float, float]:
    """Euclidean distance in fitted standardized numeric + missing-indicator space."""
    if len(training) < k:
        raise ValueError("fewer training rows than fixed support k")
    distances = np.linalg.norm(training - target, axis=1)
    nearest = np.partition(distances, k - 1)[:k]
    return float(np.min(nearest)), float(np.mean(nearest))


def db_coverage_status(incoming: int, observed: int, source_available: bool) -> str:
    if not source_available:
        return "unavailable"
    if incoming == 0:
        return "no incoming DB players"
    if observed == incoming:
        return "complete"
    return "partial" if observed > 0 else "unavailable"


def observed_db_impacts() -> dict[tuple[int, str], tuple[int, float]]:
    """Sum resolved historical DB impacts in the retained player audit."""
    counts: Counter[tuple[int, str]] = Counter()
    sums: defaultdict[tuple[int, str], float] = defaultdict(float)
    for row in read_csv(DB_AUDIT):
        if (
            row["portal_position_group"] != "db"
            or row["cutoff_status"] != "on_or_before_cutoff"
            or row["in_model_relevant_population"] != "True"
            or row["impact_status"]
            not in {"resolved", "zero_recorded_defensive_box_score_games"}
        ):
            continue
        if row["prior_defensive_impact"] == "":
            raise ValueError("resolved DB player lacks an impact value")
        item = int(row["season"]), row["destination_team_id"]
        counts[item] += 1
        sums[item] += float(row["prior_defensive_impact"])
    return {item: (count, sums[item]) for item, count in counts.items()}


def feature_diagnostics(model, target_row, training_rows: list) -> dict[str, object]:
    if any(row.season >= target_row.season for row in training_rows):
        raise ValueError("training includes target or future season")
    names = model.feature_names
    if tuple(names) != LOCATION_FEATURE_NAMES:
        raise ValueError("fitted location inventory drift")
    prep = model.preprocessor
    target = prep.transform([target_row.features])[0]
    training = prep.transform([row.features for row in training_rows])
    n = len(names)
    beta = model.beta
    if model.lag_count != 1 or len(beta) != 2 + 2 * n:
        raise ValueError("unexpected Context location coefficient layout")
    result: dict[str, object] = {"context_location_intercept": float(beta[1])}
    range_violations = 0
    for i, name in enumerate(names):
        raw = target_row.features.get(name)
        numeric = float(target[i])
        missing = float(target[n + i])
        percentile, status = percentile_and_range(numeric, training[:, i])
        range_violations += status != "within training range"
        result.update(
            {
                f"feature_{name}_raw": "" if raw is None else float(raw),
                f"feature_{name}_transformed": float(
                    prep.medians[name] if raw is None else raw
                ),
                f"feature_{name}_standardized": numeric,
                f"feature_{name}_missing": int(missing),
                f"feature_{name}_coefficient": float(beta[2 + i]),
                f"feature_{name}_contribution": float(numeric * beta[2 + i]),
                f"feature_{name}_missing_coefficient": float(beta[2 + n + i]),
                f"feature_{name}_missing_contribution": float(
                    missing * beta[2 + n + i]
                ),
                f"feature_{name}_training_percentile": percentile,
                f"feature_{name}_training_range": status,
            }
        )
    lag_points = product_quadrature((target_row.lag1_z,), QUADRATURE_POINTS)
    lag_contribution = float(np.mean(lag_points[:, 0]) * beta[0])
    result["lag1_quadrature_mean"] = float(np.mean(lag_points[:, 0]))
    result["lag1_coefficient"] = float(beta[0])
    result["lag1_contribution"] = lag_contribution
    reconstructed = float(
        beta[1]
        + lag_contribution
        + sum(
            float(result[f"feature_{name}_contribution"])
            + float(result[f"feature_{name}_missing_contribution"])
            for name in names
        )
    )
    fitted = float(
        np.mean(
            model.conditional_parameters(
                target_row.features, target_row.lag1_z, target_row.lag_zs
            )[0]
        )
    )
    residual = reconstructed - fitted
    if abs(residual) > 1e-10:
        raise ValueError(f"location contribution reconstruction residual {residual}")
    near, mean_k = nearest_support(target, training)
    result.update(
        {
            "context_reconstructed_location_center": reconstructed,
            "context_fitted_location_center": fitted,
            "context_location_reconstruction_residual": residual,
            "feature_range_violation_count": range_violations,
            "training_support_k": K,
            "nearest_neighbor_distance": near,
            "mean_k_nearest_neighbor_distance": mean_k,
            "context_training_row_count": len(training_rows),
        }
    )
    return result


def baseline_rows() -> tuple[
    list[dict[str, str]],
    dict[tuple[int, str], dict[str, str]],
    dict[int, str],
    dict[str, object],
]:
    prior = read_csv(BASE / "team_season_prior_decomposition.csv")
    posterior = read_csv(BASE / "hybrid_posterior_team_results.csv")
    reproduction = read_csv(BASE / "baseline_reproduction_checks.csv")
    provenance = json.loads((BASE / "provenance.json").read_text(encoding="utf-8"))
    if len(prior) != 534 or len({key(row) for row in prior}) != 534:
        raise ValueError("#147 population differs from 534 unique team-seasons")
    if len(reproduction) != 700 or any(row["passed"] != "True" for row in reproduction):
        raise ValueError("#147 baseline reproduction checks did not all pass")
    if max(float(row["max_absolute_error"]) for row in reproduction) > TOL:
        raise ValueError("#147 baseline reproduction exceeds tolerance")
    final_checkpoint = {
        season: max(
            int(row["checkpoint"]) for row in posterior if int(row["season"]) == season
        )
        for season in crossover.SEASONS
    }
    final = {
        key(row): row
        for row in posterior
        if row["arm"] in {"CC", "HH"}
        and int(row["checkpoint"]) == final_checkpoint[int(row["season"])]
        and row["arm"] == "CC"
    }
    history = {
        key(row): row
        for row in posterior
        if row["arm"] == "HH"
        and int(row["checkpoint"]) == final_checkpoint[int(row["season"])]
    }
    if (
        len(final) != 534
        or set(final) != set(history)
        or set(final) != {key(row) for row in prior}
    ):
        raise ValueError("#147 final checkpoint evaluation population differs")
    for item in prior:
        k = key(item)
        for family, source in (("context", final[k]), ("history", history[k])):
            for metric in ("nll", "crps", "expected_rank"):
                if (
                    abs(
                        float(item[f"{family}_prior_{metric}"])
                        - float(source[f"prior_{metric}"])
                    )
                    > TOL
                ):
                    raise ValueError(
                        f"#147 prior metric parity failed: {k} {family} {metric}"
                    )
    return (
        prior,
        {k: {"context": final[k], "history": history[k]} for k in final},
        final_checkpoint,
        provenance,
    )


def transfer_diagnostics(
    season: int,
    team_id: str,
    model_features: dict[str, float | None],
    repaired: dict[tuple[int, str], dict[str, str]],
    corrected: dict[tuple[int, str], dict[str, str]],
    coverage: dict[tuple[int, str], dict[str, str]],
    db_impacts: dict[tuple[int, str], tuple[int, float]],
    changes: dict[tuple[int, str, str], dict[str, str]],
) -> dict[str, object]:
    k = season, team_id
    evidence = repaired.get(k)
    fixed = corrected.get(k)
    prior = coverage.get(k)
    result: dict[str, object] = {
        "transfer_provenance_class": ""
        if evidence is None
        else evidence["provenance_class"],
        "transfer_checkpoint_status": ""
        if evidence is None
        else evidence["checkpoint_status"],
        "transfer_repair_class": "" if evidence is None else evidence["repair_class"],
        "transfer_availability_status": ""
        if evidence is None
        else evidence["availability_status"],
        "returning_production": model_features.get("returning_pct_ppa", ""),
    }
    if evidence is None:
        if model_features:
            raise ValueError(f"{k}: missing repaired transfer coverage for fitted row")
        result["db_coverage_status"] = "unavailable"
        return result
    if model_features and fixed is None:
        raise ValueError(f"{k}: missing corrected transfer features for fitted row")
    incoming = int(evidence["incoming_transfers"])
    db_incoming = int(evidence["db_incoming"])
    db_observed = int(evidence["db_resolved"])
    observed_count, observed_sum = db_impacts.get(k, (0, 0.0))
    if observed_count != db_observed:
        raise ValueError(f"{k}: repaired DB count differs from player audit")
    result.update(
        {
            "incoming_transfer_count": incoming,
            "incoming_offensive_applicable_count": int(evidence["offensive_resolved"])
            + int(evidence["offensive_join_failed"]),
            "incoming_offensive_observed_usage_count": int(
                evidence["offensive_resolved"]
            ),
            "incoming_offensive_applicability_unknown_count": int(
                evidence["offensive_applicability_unknown"]
            ),
            "incoming_offensive_observed_usage_sum": (
                ""
                if optional_float(evidence["post_repair_observed_usage_sum"]) is None
                else optional_float(evidence["post_repair_observed_usage_sum"])
            ),
            "incoming_db_count": db_incoming,
            "observed_db_impact_count": db_observed,
            "db_coverage_fraction": 1.0
            if db_incoming == 0
            else db_observed / db_incoming,
            "db_coverage_status": db_coverage_status(db_incoming, db_observed, True),
            "transfer_source_coverage_gap": evidence["source_coverage_gap"],
            "transfer_primary_reason": evidence["primary_reason"],
        }
    )
    result["observed_db_impact_sum"] = (
        observed_sum if db_observed or db_incoming == 0 else ""
    )
    if db_observed == db_incoming and fixed is not None:
        corrected_sum = optional_float(
            fixed["transfer_in_prior_defensive_impact_db_sum"]
        )
        if corrected_sum is None or not np.isclose(
            observed_sum, corrected_sum, atol=1e-10, rtol=0
        ):
            raise ValueError(
                f"{k}: DB player sum differs from corrected complete feature"
            )
    for name in TRANSFER_FEATURES:
        frozen = model_features.get(name)
        repair_value = optional_float(fixed[name]) if fixed is not None else None
        result[f"{name}_model_input_value"] = "" if frozen is None else frozen
        result[f"{name}_corrected_diagnostic_value"] = (
            "" if repair_value is None else repair_value
        )
        if not model_features:
            reason = "cold-start fallback; no fitted Context feature input"
            timing_status = ""
        else:
            reason = reconciled_change_class(
                (season, team_id, name), frozen, repair_value, changes
            )
            timing_status = evidence["checkpoint_status"] if reason else ""
        result[f"{name}_difference_reason"] = reason
        result[f"{name}_difference_timing_status"] = timing_status
    result["frozen_transfer_usage_available"] = (
        "" if prior is None else prior["transfer_in_prior_usage_sum_available"]
    )
    return result


def build(
    target_path: Path, output: Path = OUT, report: Path = REPORT
) -> dict[str, object]:
    prior, final, checkpoints, frozen_provenance = baseline_rows()
    target_path = target_path.resolve()
    source_hashes = frozen_provenance["sources_sha256"]
    if (
        sha256(target_path)
        != source_hashes["external_input/team_season_rank_distributions.csv"]
    ):
        raise ValueError("target distribution corpus differs from #147")
    data_root = target_path.parents[3]
    crossover.hcp.c13.v1.ROOT = data_root
    crossover.hcp.c13.c12.MODELING = target_path.parent
    crossover.hcp.c13.c12.TENURES = data_root / "data/raw/cfbd/preseason/coach_tenures"
    rows, _cold, _coverage = crossover.hcp.c13.load_candidate_rows()
    by_season = defaultdict(list)
    for row in rows:
        by_season[row.season].append(row)
    repaired = {
        key(r): r
        for r in read_csv(REPAIR / "team_seasons.csv")
        if r["subdivision"] == "fbs"
    }
    corrected = {
        key(r): r
        for r in read_csv(REPAIR / "historical_transfer_features.csv")
        if r["subdivision"] == "fbs"
    }
    coverage = {
        key(r): r for r in read_csv(crossover.hcp.COVERAGE) if r["subdivision"] == "fbs"
    }
    db_impacts = observed_db_impacts()
    changes = index_historical_feature_changes(read_csv(HISTORICAL_CHANGES))
    models = {}
    model_hashes = {}
    training = {}
    for season in crossover.SEASONS:
        train = [row for row in rows if row.season < season]
        model, instance = crossover.hcp.c13.fit_model(
            rows, target_season=season, trained_through_season=season - 1
        )
        if instance.trained_through_season != season - 1:
            raise ValueError("rolling origin drift")
        digest = crossover.sha256_json(model.metadata())
        if digest != frozen_provenance["models"][str(season)]["context_model_sha256"]:
            raise ValueError(f"{season}: Context model differs from #147")
        models[season], training[season], model_hashes[str(season)] = (
            model,
            train,
            digest,
        )
    result = []
    for old in prior:
        season, team_id = key(old)
        checkpoint = final[(season, team_id)]
        c, h = checkpoint["context"], checkpoint["history"]
        c_rank, h_rank = (
            float(old["context_prior_expected_rank"]),
            float(old["history_prior_expected_rank"]),
        )
        target_rank = float(old["context_prior_target_expected_rank"])
        if abs(target_rank - float(old["history_prior_target_expected_rank"])) > TOL:
            raise ValueError("Context and History target differ")
        row: dict[str, object] = {
            "season": season,
            "team": old["team_name"],
            "team_id": team_id,
            "component_status": old["component_status"],
            "context_model_version": "1.3",
            "history_model_version": "1.1",
            "context_training_cutoff": season - 1,
            "history_training_cutoff": season - 1,
            "preseason_reconstruction_identifier": "issue-147-rolling-origin-retrospective",
            "preseason_boundary": f"{season}-08-15",
            "final_shared_checkpoint": c["cutoff"],
            "final_shared_checkpoint_index": checkpoints[season],
            "target_source": "team_season_rank_distributions.csv",
            "target_pmf_sha256": old["target_pmf_sha256"],
            "context_model_sha256": model_hashes[str(season)],
            "history_model_sha256": frozen_provenance["models"][str(season)][
                "history_model_sha256"
            ],
            "context_prior_pmf_sha256": old["context_prior_pmf_sha256"],
            "history_prior_pmf_sha256": old["history_prior_pmf_sha256"],
            "context_preseason_expected_rank": c_rank,
            "history_preseason_expected_rank": h_rank,
            "target_expected_rank": target_rank,
            "context_preseason_signed_rank_error": c_rank - target_rank,
            "history_preseason_signed_rank_error": h_rank - target_rank,
            "context_preseason_abs_rank_error": abs(c_rank - target_rank),
            "history_preseason_abs_rank_error": abs(h_rank - target_rank),
            "context_minus_history_abs_rank_error": abs(c_rank - target_rank)
            - abs(h_rank - target_rank),
            "context_minus_history_preseason_expected_rank": c_rank - h_rank,
            "abs_context_minus_history_preseason_expected_rank": abs(c_rank - h_rank),
            "context_final_expected_rank": float(c["posterior_expected_rank"]),
            "history_final_expected_rank": float(h["posterior_expected_rank"]),
            "context_final_nll": float(c["posterior_nll"]),
            "history_final_nll": float(h["posterior_nll"]),
            "context_final_crps": float(c["posterior_crps"]),
            "history_final_crps": float(h["posterior_crps"]),
            "context_minus_history_final_nll": float(c["posterior_nll"])
            - float(h["posterior_nll"]),
            "context_location_center": old["context_location_center"],
            "history_location_center": old["history_location_center"],
            "context_minus_history_location_center": old[
                "context_minus_history_location_center"
            ],
            "context_location_center_feature_input_sha256": old[
                "context_location_center_feature_input_sha256"
            ],
            "history_location_center_feature_input_sha256": old[
                "history_location_center_feature_input_sha256"
            ],
            "target_population": old["target_population"],
        }
        for family in ("context", "history"):
            for source_name, dest_name in (
                ("nll", "nll"),
                ("crps", "crps"),
                ("interval_80_low", "interval_low"),
                ("interval_80_high", "interval_high"),
                ("interval_80_width", "interval_width"),
                ("interval_80_target_mass", "interval_target_mass"),
                ("entropy", "entropy"),
                ("rank_variance", "rank_variance"),
                ("rank_standard_deviation", "rank_standard_deviation"),
            ):
                row[f"{family}_preseason_{dest_name}"] = float(
                    old[f"{family}_prior_{source_name}"]
                )
        if (
            c["cutoff"] != h["cutoff"]
            or c["component_status"] != old["component_status"]
        ):
            raise ValueError("#147 checkpoint or fallback mismatch")
        source_row = next((r for r in by_season[season] if r.team_id == team_id), None)
        if source_row is not None and old["component_status"] == "fitted":
            model_features = source_row.features
            if (
                crossover.sha256_json(crossover.to_float_features(model_features))
                != old["context_location_center_feature_input_sha256"]
            ):
                raise ValueError("#147 Context feature input hash differs")
            row.update(
                feature_diagnostics(models[season], source_row, training[season])
            )
            if (
                abs(
                    float(row["context_fitted_location_center"])
                    - float(old["context_location_center"])
                )
                > TOL
            ):
                raise ValueError("#147 Context center differs")
        else:
            if old["component_status"] != "cold_start_fallback":
                raise ValueError("missing fitted Context team row")
            model_features = {}
        row.update(
            transfer_diagnostics(
                season,
                team_id,
                model_features,
                repaired,
                corrected,
                coverage,
                db_impacts,
                changes,
            )
        )
        result.append(row)
    if len(result) != 534:
        raise ValueError("population drift")
    fitted_keys = {
        (int(row["season"]), str(row["team_id"]))
        for row in result
        if row["component_status"] == "fitted"
    }
    reconciled_keys = {
        (int(row["season"]), str(row["team_id"]), name)
        for row in result
        if row["component_status"] == "fitted"
        for name in TRANSFER_FEATURES
        if row[f"{name}_difference_reason"]
    }
    expected_change_keys = {
        change_key for change_key in changes if change_key[:2] in fitted_keys
    }
    if reconciled_keys != expected_change_keys:
        raise ValueError("eligible historical feature changes were not reconciled")
    result.sort(key=lambda r: (int(r["season"]), str(r["team_id"])))
    inventory_rows = inventory()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "team_seasons.csv", result)
    write_csv(output / "feature_inventory.csv", inventory_rows)
    paths = [
        BASE / name
        for name in (
            "team_season_prior_decomposition.csv",
            "hybrid_posterior_team_results.csv",
            "baseline_reproduction_checks.csv",
            "provenance.json",
        )
    ]
    paths += [
        REPAIR / name
        for name in (
            "team_seasons.csv",
            "historical_transfer_features.csv",
            "historical_feature_changes.csv",
        )
    ]
    paths.append(DB_AUDIT)
    paths += [
        crossover.hcp.c13.HISTORICAL_TRANSFER_FEATURES,
        crossover.hcp.COVERAGE,
        target_path,
    ]
    hashes = {
        str(path.relative_to(ROOT))
        if path.is_relative_to(ROOT)
        else "external_input/" + path.name: sha256(path)
        for path in paths
    }
    provenance = {
        "study": "issue-152 Context location-error diagnostic dataset",
        "baseline": "issue-147 committed rolling-origin comparison",
        "target_seasons": list(crossover.SEASONS),
        "source_hashes": hashes,
        "context_model_hashes": model_hashes,
        "preseason_input_timing": "historical retrospective reconstruction; archived August 15 availability unverified",
        "corrected_transfer_evidence": "diagnostic only; historical timing unverified and never used to refit",
        "support_metric": {
            "k": K,
            "distance": "Euclidean",
            "columns": "fitted standardized numeric features and missingness indicators",
            "training": "all Context fit rows with season before target",
        },
    }
    write_json(output / "provenance.json", provenance)
    fitted = [row for row in result if row["component_status"] == "fitted"]
    residuals = [
        abs(float(row["context_location_reconstruction_residual"])) for row in fitted
    ]
    summary = {
        "team_seasons_expected": 534,
        "team_seasons_included": len(result),
        "team_seasons_excluded": 0,
        "fitted_decompositions": len(fitted),
        "fallback_team_seasons": len(result) - len(fitted),
        "context_location_features": len(inventory_rows),
        "historical_repair_reconciled_changes": len(reconciled_keys),
        "historical_repair_difference_classes": dict(
            sorted(
                Counter(
                    str(row[f"{name}_difference_reason"])
                    for row in fitted
                    for name in TRANSFER_FEATURES
                    if row[f"{name}_difference_reason"]
                ).items()
            )
        ),
        "missing_diagnostic_fields": {
            "fitted_context_location_center": len(result) - len(fitted),
            "observed_db_impact_sum": sum(
                row.get("observed_db_impact_sum", "") == "" for row in result
            ),
            "incoming_offensive_observed_usage_sum": sum(
                row.get("incoming_offensive_observed_usage_sum", "") == ""
                for row in result
            ),
        },
        "baseline_reproduction_checks": 700,
        "baseline_max_absolute_error": max(
            float(r["max_absolute_error"])
            for r in read_csv(BASE / "baseline_reproduction_checks.csv")
        ),
        "max_abs_reconstruction_residual": max(residuals),
        "mean_abs_reconstruction_residual": float(np.mean(residuals)),
        "db_coverage_states": dict(
            sorted(Counter(str(row["db_coverage_status"]) for row in result).items())
        ),
        "nearest_neighbor_distance": {
            name: float(value)
            for name, value in zip(
                ("min", "median", "max"),
                np.percentile(
                    [float(row["nearest_neighbor_distance"]) for row in fitted],
                    [0, 50, 100],
                ),
                strict=True,
            )
        },
        "feature_range_violation_counts": dict(
            sorted(
                Counter(
                    int(row["feature_range_violation_count"]) for row in fitted
                ).items()
            )
        ),
        "final_checkpoint_by_season": {
            str(season): checkpoints[season] for season in crossover.SEASONS
        },
    }
    write_json(output / "summary.json", summary)
    report.write_text(
        "# Context location-error diagnostic dataset\n\n"
        "This artifact joins the frozen #147 rolling-origin Context 1.3 / History 1.1 comparison with exact Context location inputs and training support. It is an infrastructure artifact; no error pattern is interpreted here.\n\n"
        f"## Population\n\nExpected: 534 team-seasons. Included: {len(result)}. Excluded: 0. Of these, {len(fitted)} have fitted location decompositions and {len(result) - len(fitted)} retain native cold-start fallback priors with blank feature/contribution fields.\n\n"
        f"## #147 reproduction\n\nAll 700 retained #147 baseline checks passed (maximum absolute error {summary['baseline_max_absolute_error']:.3g}); this builder also checks all 534 Context and History prior NLL, CRPS, and expected-rank values against the final checkpoint rows, checks the team population, and requires SHA-256 parity for each rolling Context fit's metadata. Final posterior values are copied from #147's CC and HH rows at the final shared checkpoint.\n\n"
        f"## Feature coverage and contributions\n\n{len(inventory_rows)} location features come from the production fitting specification; all {len(inventory_rows)} are decomposed for each fitted team. The six fallback cases have no fitted Context location center. Missing diagnostic field counts: {summary['missing_diagnostic_fields']}. The fitted center includes the intercept, 15 standardized feature contributions, 15 missing-indicator contributions, and the mean t-1 rank-distribution quadrature contribution. Maximum / mean absolute reconstruction residual: {summary['max_abs_reconstruction_residual']:.3g} / {summary['mean_abs_reconstruction_residual']:.3g}.\n\n"
        f"## Transfer coverage\n\nDB states: {summary['db_coverage_states']}. The {summary['historical_repair_reconciled_changes']} fitted-row frozen/corrected differences reconcile one-to-one with #151's historical feature-change inventory: {summary['historical_repair_difference_classes']}. Each `*_difference_reason` is the authoritative `change_class`; historical timing remains in the separate `*_difference_timing_status` and `transfer_checkpoint_status` fields. Repaired #151 historical evidence is descriptive and has unverified August 15 availability. The frozen #147 model-facing values remain in separate columns. Observed DB sums are reconstructed from retained player audit rows marked on or before the cutoff, with count parity against #151 and sum parity against complete corrected aggregates. Unavailable DB sums remain blank; neutral model zeros are never presented as observed partial sums. Unknown offensive applicability remains a separate count.\n\n"
        f"## Training support\n\nNearest-neighbor Euclidean distance uses the 15 training-standardized numeric features plus their 15 missing indicators and fixed k={K}. Minimum / median / maximum nearest distance: {summary['nearest_neighbor_distance']}. Feature range-violation counts: {summary['feature_range_violation_counts']}. Percentiles use midranks against the target season's rolling training rows only.\n\n"
        "## Conventions and provenance\n\nSigned expected-rank error is forecast minus target; positive means a worse (numerically larger) predicted rank. Positive Context-minus-History absolute-rank error means Context was farther from the target. Positive Context-minus-History final NLL means History assigned the target more probability. Preseason NLL, CRPS, absolute error, interval target mass, and final posterior scores are outcome-derived evaluation columns; they are never used as preseason inputs. Source paths and SHA-256 hashes are in `data/processed/context_location_error_diagnostics/provenance.json`; the #147 provenance records likelihood, evidence, and inference settings. Historical transfer inputs are retrospective reconstructions, not certified archived preseason snapshots.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=crossover.hcp.TARGETS)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.targets, args.output, args.report), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
