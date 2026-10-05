"""Rebuild corrected Context 1.4 prior from repaired DB evidence (#172)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_context_v1_4_predictive as bridge
import build_preseason_context_prior_v1_3 as c13

from gippyrank.artifact_hashes import posterior_pmfs_sha256
from gippyrank.context_db_repair import (
    ALPHA,
    CURRENT_COVERAGE,
    CURRENT_OFFENSE,
    HISTORICAL_COVERAGE,
    HISTORICAL_FEATURES,
    HISTORICAL_PLAYERS,
    MODEL_FEATURES,
    attach_historical,
    current_repaired_features,
    fit_model,
    historical_repaired_features,
    production_pmf,
    read_csv,
    sha256,
)
from gippyrank.context_prior import InferenceRow
from gippyrank.context_prior_v1_4 import (
    construct_production_prior,
    production_semantics_sha256,
    production_spec,
)
from gippyrank.posterior.engine import Team
from gippyrank.posterior.snapshots import load_teams
from gippyrank.preseason import pmf_summaries, team_log_score

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("data/processed/context_db_repair_172")
ANNUAL = Path("data/processed/preseason/context_v1_4/annual/2026")
OLD_AUDIT = Path(
    "data/processed/preseason/context_v1_3_2026_reconstruction/transfer_team_audit.csv"
)
VALIDATOR_ROWS = Path("data/processed/context_v1_4_validation/validator_inputs.jsonl")
SEASONS = (2022, 2023, 2024, 2025)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def build_inventory(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    _, repaired = current_repaired_features(root)
    old = {row["team_id"]: row for row in read_csv(root / OLD_AUDIT)}
    if set(old) != set(repaired):
        raise ValueError("old and repaired 2026 team inventories differ")
    rows = []
    for team_id, db in sorted(repaired.items(), key=lambda item: int(item[0])):
        source = old[team_id]
        old_incoming = int(source["audit_incoming_db_transfers"])
        old_observed = int(source["audit_resolved_db_transfers"])
        old_impact = float(source["transfer_in_prior_defensive_impact_db_sum"])
        old_status = source["audit_db_feature_status"]
        if int(source["transfer_in_prior_defensive_impact_db_available"]) != int(
            old_incoming == old_observed
        ):
            raise ValueError(f"old DB availability/count mismatch: {team_id}")
        if db.incoming_db_count == 0:
            change_class = "natural_zero"
        elif old_observed < old_incoming and db.status == "complete":
            change_class = "became_complete"
        elif db.status == "partial" and db.observed_db_impact_count > old_observed:
            change_class = "partial_gained_observed"
        elif (
            db.status == "partial"
            and old_observed == db.observed_db_impact_count
            and not np.isclose(
                old_impact, db.observed_db_impact_sum, rtol=0, atol=1e-12
            )
        ):
            change_class = "partial_observed_sum_exposed"
        elif (
            db.status == "complete"
            and old_observed == old_incoming
            and not np.isclose(
                old_impact, db.observed_db_impact_sum, rtol=0, atol=1e-12
            )
        ):
            change_class = "complete_impact_corrected"
        elif (
            old_incoming != db.incoming_db_count
            or old_observed != db.observed_db_impact_count
            or not np.isclose(old_impact, db.observed_db_impact_sum, rtol=0, atol=1e-12)
        ):
            change_class = "other_changed"
        else:
            change_class = "unchanged"
        rows.append(
            {
                "team_id": team_id,
                "team": source["team_name"],
                "old_incoming_count": old_incoming,
                "old_observed_count": old_observed,
                "new_incoming_count": db.incoming_db_count,
                "new_observed_count": db.observed_db_impact_count,
                "old_db_impact": old_impact,
                "new_observed_db_impact": db.observed_db_impact_sum,
                "old_availability": int(
                    source["transfer_in_prior_defensive_impact_db_available"]
                ),
                "old_status": old_status,
                "new_missing_count": db.missing_db_impact_count,
                "new_coverage_fraction": db.db_impact_coverage_fraction,
                "new_status": db.status,
                "change_class": change_class,
            }
        )
    auburn = next(row for row in rows if row["team_id"] == "2")
    if not (
        auburn["old_observed_count"] == 5
        and auburn["old_incoming_count"] == 6
        and auburn["old_db_impact"] == 0
        and auburn["new_observed_count"] == 6
        and auburn["new_incoming_count"] == 6
        and np.isclose(auburn["new_observed_db_impact"], 3.3319573489815544)
        and auburn["new_status"] == "complete"
    ):
        raise ValueError("Auburn repaired DB regression")
    summary = {
        "team_count": len(rows),
        "class_counts": dict(Counter(str(row["change_class"]) for row in rows)),
        "old_complete_including_natural_zero": sum(
            row["old_observed_count"] == row["old_incoming_count"] for row in rows
        ),
        "new_complete_including_natural_zero": sum(
            row["new_status"] in {"complete", "natural_zero"} for row in rows
        ),
        "old_partial": sum(
            0 < row["old_observed_count"] < row["old_incoming_count"] for row in rows
        ),
        "new_partial": sum(row["new_status"] == "partial" for row in rows),
        "natural_zero": sum(row["new_status"] == "natural_zero" for row in rows),
        "auburn": auburn,
    }
    write_csv(root / OUTPUT / "2026_before_after.csv", rows)
    write_json(root / OUTPUT / "2026_inventory_summary.json", summary)
    return rows, summary


def forecast_rows(root: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for line in (root / VALIDATOR_ROWS).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["kind"] == "forecast_row":
            value = record["value"]
            rows[str(value["team_id"])] = value
    if len(rows) != 138:
        raise ValueError("retained 2026 inference rows must include 138 FBS teams")
    return rows


def run(root: Path = ROOT, *, audit_only: bool = False) -> dict[str, object]:
    inventory, summary = build_inventory(root)
    if audit_only:
        return summary
    historic_features, historic_db = historical_repaired_features(root)
    if len(historic_db) != 655:
        raise ValueError("repaired 2021-2025 DB panel must have 655 team-seasons")
    base, _, _ = c13.base_context_rows()
    rows = attach_historical(base, historic_features)
    current_features, current_db = current_repaired_features(root)
    saved = forecast_rows(root)
    old_path = root / ANNUAL / "predictions.csv"
    baseline_path = root / OUTPUT / "baseline_context14_predictions.csv"
    if not baseline_path.is_file():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_bytes(old_path.read_bytes())
    if (
        sha256(baseline_path)
        != "b03850ff79f319d35b582cd733d035c1c7e4e48368c5354fb696958ae965ef8a"
    ):
        raise ValueError("retained production Context 1.4 baseline changed")
    old_rows = read_csv(baseline_path)
    old = {row["team_id"]: row for row in old_rows}
    if set(saved) != set(old) or set(old) != set(current_db):
        raise ValueError("2026 fitted, retained, and repaired populations differ")

    metrics: list[dict[str, object]] = []
    _, old_historical, _ = bridge._development_prior_teams()
    for season in SEASONS:
        print(f"Fitting repaired Context through {season - 1}", flush=True)
        model = fit_model(rows, season - 1)
        write_json(
            root / OUTPUT / "historical_models" / f"{season}.json",
            {
                "target_season": season,
                "trained_through_season": season - 1,
                "model": model.metadata(),
            },
        )
        target = [row for row in rows if row.season == season]
        current_by_id = {team.team_id: team for team in old_historical[season]}
        history_teams, _, _ = load_teams(root, season, "history")
        history_by_id = {team.team_id: team for team in history_teams}
        scores = {
            arm: {"nll": [], "expected_rank_mae": []}
            for arm in ("corrected_context_1_4", "current_context_1_4", "history_1_1")
        }
        corrected_prior_rows: list[dict[str, object]] = []
        for row in target:
            corrected_pmf, _, _ = production_pmf(
                model, row.features, row.lag1_z, row.population
            )
            for arm, pmf in (
                ("corrected_context_1_4", corrected_pmf),
                ("current_context_1_4", current_by_id[row.team_id].prior),
                ("history_1_1", history_by_id[row.team_id].prior),
            ):
                scores[arm]["nll"].append(team_log_score(pmf, row.target_ranks))
                scores[arm]["expected_rank_mae"].append(
                    float(
                        np.mean(
                            np.abs(
                                pmf_summaries(pmf)["expected_rank"] - row.target_ranks
                            )
                        )
                    )
                )
            corrected_prior_rows.append(
                {
                    "season": season,
                    "team_id": row.team_id,
                    "team_name": row.team_name,
                    "pmf": json.dumps(corrected_pmf.tolist(), separators=(",", ":")),
                }
            )
        fitted_ids = {str(item["team_id"]) for item in corrected_prior_rows}
        for team_id, team in current_by_id.items():
            if team_id not in fitted_ids:
                corrected_prior_rows.append(
                    {
                        "season": season,
                        "team_id": team_id,
                        "team_name": team.name,
                        "pmf": json.dumps(team.prior.tolist(), separators=(",", ":")),
                    }
                )
        if len(corrected_prior_rows) != len(current_by_id):
            raise ValueError(
                f"{season}: corrected prior population differs from baseline"
            )
        write_csv(
            root / OUTPUT / "historical_priors" / f"{season}.csv",
            sorted(corrected_prior_rows, key=lambda item: int(str(item["team_id"]))),
        )
        metrics.append(
            {
                "season": season,
                "trained_through": season - 1,
                "fitted_team_count": len(target),
                **{
                    f"{arm}_{metric}": float(np.mean(values))
                    for arm, arm_scores in scores.items()
                    for metric, values in arm_scores.items()
                },
                "model_sha256": __import__("hashlib")
                .sha256(json.dumps(model.metadata(), sort_keys=True).encode())
                .hexdigest(),
            }
        )
    print("Fitting repaired Context through 2025", flush=True)
    model = fit_model(rows, 2025)
    output = root / ANNUAL
    output.mkdir(parents=True, exist_ok=True)
    new_rows: list[dict[str, object]] = []
    prior_changes: list[dict[str, object]] = []
    pmfs: dict[str, np.ndarray] = {}
    cold_ids = set(
        json.loads((root / "config/context_v1_4_validation.json").read_text())[
            "population"
        ]["cold_start_team_ids"]
    )
    for old_row in old_rows:
        team_id = old_row["team_id"]
        source = saved[team_id]
        db = current_db[team_id]
        values = {name: source["features"].get(name) for name in MODEL_FEATURES}
        values.update(current_features[team_id])
        if team_id in cold_ids:
            pmf = np.asarray(json.loads(old_row["pmf"]), dtype=float)
            center = old_row["conditional_location_mean"]
            scale = old_row["predictive_scale"]
        else:
            inference_row = InferenceRow(
                season=2026,
                subdivision="fbs",
                team_id=team_id,
                team_name=old_row["team_name"],
                population=138,
                lag1_z=tuple(float(value) for value in source["lag1_z"]),
                lag_zs=(),
                features=values,
                cold_start_reason=None,
            )
            production = construct_production_prior(model, inference_row)
            pmf = production.pmf
            center = production.conditional_location_mean
            scale = production.predictive_scale
        old_pmf = np.asarray(json.loads(old_row["pmf"]), dtype=float)
        old_rank = pmf_summaries(old_pmf)["expected_rank"]
        new_rank = pmf_summaries(pmf)["expected_rank"]
        prior_changes.append(
            {
                "team_id": team_id,
                "team": old_row["team_name"],
                "old_expected_rank": old_rank,
                "new_expected_rank": new_rank,
                "rank_change": new_rank - old_rank,
                "coverage_status": db.status,
                "observed_db_impact_sum": db.observed_db_impact_sum,
                "old_db_impact": next(
                    item["old_db_impact"]
                    for item in inventory
                    if item["team_id"] == team_id
                ),
            }
        )
        new_rows.append(
            {
                **old_row,
                "pmf": json.dumps(pmf.tolist(), separators=(",", ":")),
                "conditional_location_mean": center,
                "predictive_scale": scale,
                **pmf_summaries(pmf),
                **db.audit_fields(),
            }
        )
        pmfs[team_id] = Team(team_id, old_row["team_name"], "fbs", pmf).prior
    prior_changes.sort(key=lambda row: abs(float(row["rank_change"])), reverse=True)
    write_csv(root / OUTPUT / "2026_prior_changes.csv", prior_changes)
    write_csv(old_path, new_rows)
    model_path = output / "fitted_model.json"
    spec = {
        **production_spec(),
        "positive_context_only_moderation_alpha": ALPHA,
        "history_1_1": "unchanged",
        "historical_db_before_2021": "source unavailable; both DB features missing",
        "2026_transfer_timing": "Reacquired after the August 15 cutoff; retrospective reconstruction",
    }
    write_json(output / "model_spec.json", spec)
    write_json(model_path, {"model": model.metadata(), "spec_version": "1.4"})
    write_json(
        output / "fitted_instance.json",
        {
            "model_family": "context_prior",
            "spec_version": "1.4",
            "trained_through_season": 2025,
            "target_season": 2026,
        },
    )
    inputs = {
        path.as_posix(): sha256(root / path)
        for path in (
            HISTORICAL_FEATURES,
            HISTORICAL_COVERAGE,
            HISTORICAL_PLAYERS,
            CURRENT_COVERAGE,
            CURRENT_OFFENSE,
            OLD_AUDIT,
            VALIDATOR_ROWS,
            Path("data/processed/modeling/team_season_rank_distributions.csv"),
            Path("data/processed/preseason/team_season_features.csv"),
        )
    }
    report = {
        "spec": spec,
        "production_semantics_sha256": production_semantics_sha256(),
        "source_sha256": inputs,
        "historical_preseason": metrics,
        "historical_db_status": dict(Counter(db.status for db in historic_db.values())),
        "2026_inventory": summary,
        "2026_prior_pmfs_sha256": posterior_pmfs_sha256(pmfs),
        "2026_predictions_sha256": sha256(old_path),
        "2026_fitted_model_sha256": sha256(model_path),
        "top_prior_rank_changes": prior_changes[:20],
        "auburn_prior_change": next(
            row for row in prior_changes if row["team_id"] == "2"
        ),
    }
    write_json(root / OUTPUT / "model_comparison.json", report)
    write_json(
        output / "promotion.json",
        {
            "model_family": "context_prior",
            "spec_version": "1.4",
            "status": "active_production",
            "correction": spec["correction"],
            "moderation_alpha": ALPHA,
            "predictions_sha256": sha256(old_path),
            "prior_pmfs_sha256": report["2026_prior_pmfs_sha256"],
            "fitted_model_sha256": sha256(model_path),
            "source_sha256": inputs,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    result = run(audit_only=args.audit_only)
    print(
        json.dumps(
            result
            if args.audit_only
            else {
                "2026_prior_pmfs_sha256": result["2026_prior_pmfs_sha256"],
                "historical_preseason": result["historical_preseason"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
