"""Diagnose the Context 1.3 / History 1.1 posterior crossover.

This issue-146 research runner reuses the rolling-origin model builders and
cutoff-safe historical evidence contract from issue #140. It does not fit a
replacement model or alter production artifacts.

Run with ``uv run python scripts/study_context_history_crossover.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study_history_context_posterior as hcp

from gippyrank.posterior.engine import Team, infer_posterior
from gippyrank.posterior.snapshots import (
    _scheduled_future_fcs_rows,
    add_fcs_fallbacks,
    filter_games,
    load_pinned_likelihood,
)
from gippyrank.preseason import conditional_rank_mixture_pmf

SEASONS = hcp.SEASONS
OUT = ROOT / "data/processed/context_history_crossover"
BASELINE_TEAMS = hcp.OUT / "historical_teams.csv"
BASELINE_SUMMARIES = hcp.OUT / "historical_checkpoints.csv"
BASELINE_EVIDENCE = hcp.OUT / "historical_evidence.csv"
BASELINE_AUDIT = hcp.OUT / "historical_cutoff_audit.csv"
INFERENCE = {"max_iterations": 500, "tolerance": 1e-9, "damping": 0.35}
PRIMARY_ARMS = (
    "CC",
    "HH",
    "C_center_H_uncertainty",
    "H_center_C_uncertainty",
)
SECONDARY_ARMS = (
    "C_center_H_offsets_C_scale",
    "C_center_C_offsets_H_scale",
    "H_center_H_offsets_C_scale",
    "H_center_C_offsets_H_scale",
)
SECONDARY_TRIGGER_SHARE = 0.25
BASELINE_TOLERANCE = 1e-8
RECONSTRUCTION_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Components:
    """Outcome-free predictive inputs and the fitted model's decomposition."""

    team_id: str
    team_name: str
    population: int
    features: dict[str, float | None]
    lag1_z: np.ndarray
    lag_zs: tuple[np.ndarray, ...]
    locations: np.ndarray
    scale: float
    pmf: np.ndarray


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def json_values(values: Iterable[float]) -> str:
    return json.dumps([float(value) for value in values], separators=(",", ":"))


def normal_mixture_pmf(
    locations: np.ndarray, scale: float, population: int
) -> np.ndarray:
    """Use the shared fitted-model Normal rank-bin mixture implementation."""
    return conditional_rank_mixture_pmf(locations, scale, population)


def pmf_summary(pmf: np.ndarray) -> dict[str, float]:
    values = np.asarray(pmf, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("PMF must be a finite non-empty vector")
    if np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-8):
        raise ValueError("PMF must be nonnegative and normalized")
    ranks = np.arange(1, len(values) + 1, dtype=float)
    cdf = np.cumsum(values)

    def quantile(probability: float) -> int:
        return int(np.searchsorted(cdf, probability, side="left") + 1)

    expected = float(np.dot(ranks, values))
    variance = float(np.dot((ranks - expected) ** 2, values))
    low, high = quantile(0.1), quantile(0.9)
    return {
        "expected_rank": expected,
        "interval_80_low": float(low),
        "interval_80_high": float(high),
        "interval_80_width": float(high - low),
        "entropy": float(-np.sum(values[values > 0] * np.log(values[values > 0]))),
        "rank_variance": variance,
        "rank_standard_deviation": float(np.sqrt(variance)),
    }


def score_pmf(pmf: np.ndarray, target: np.ndarray) -> dict[str, float]:
    forecast = np.asarray(pmf, dtype=float)
    truth = np.asarray(target, dtype=float)
    if forecast.shape != truth.shape:
        raise ValueError("forecast and target PMFs must have identical rank support")
    if not np.isclose(truth.sum(), 1.0, atol=1e-8) or np.any(truth < 0):
        raise ValueError("target PMF must be normalized and nonnegative")
    summary = pmf_summary(forecast)
    target_summary = pmf_summary(truth)
    ranks = np.arange(1, len(forecast) + 1)
    low, high = int(summary["interval_80_low"]), int(summary["interval_80_high"])
    return {
        "nll": float(-np.sum(truth * np.log(np.maximum(forecast, 1e-15)))),
        "crps": float(np.mean((np.cumsum(forecast) - np.cumsum(truth)) ** 2)),
        "expected_rank": summary["expected_rank"],
        "target_expected_rank": target_summary["expected_rank"],
        "expected_rank_error": abs(
            summary["expected_rank"] - target_summary["expected_rank"]
        ),
        "interval_80_low": summary["interval_80_low"],
        "interval_80_high": summary["interval_80_high"],
        "interval_80_width": summary["interval_80_width"],
        "interval_80_target_mass": float(truth[(ranks >= low) & (ranks <= high)].sum()),
        "entropy": summary["entropy"],
        "rank_variance": summary["rank_variance"],
        "rank_standard_deviation": summary["rank_standard_deviation"],
    }


def to_float_features(features: dict[str, object]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name, value in features.items():
        result[name] = None if value is None else float(value)
    return result


def component_from_model(model, row) -> Components:
    """Extract fitted conditional parameters without carrying target outcomes."""
    if row.lag1_z is None or not len(row.lag1_z):
        raise ValueError(f"{row.team_id}: modeled input has no lag-1 rank distribution")
    features = to_float_features(row.features)
    lag1 = np.asarray(row.lag1_z, dtype=float)
    lag_zs = tuple(np.asarray(values, dtype=float) for values in row.lag_zs)
    locations, scale = model.conditional_parameters(features, lag1, lag_zs)
    pmf = model.pmf(features, lag1, row.population, lag_zs)
    return Components(
        row.team_id,
        row.team_name,
        int(row.population),
        features,
        lag1,
        lag_zs,
        np.asarray(locations, dtype=float),
        float(scale),
        np.asarray(pmf, dtype=float),
    )


def modeled_fallback_prior(prediction) -> tuple[int, str, np.ndarray]:
    return (
        int(prediction.population),
        str(prediction.team_name),
        np.asarray(prediction.pmf, dtype=float),
    )


def load_rolling_models(rows: list, cold: list) -> tuple[dict, dict, dict, dict, dict]:
    """Fit validated rolling Context/History models and collect components."""
    context_components: dict[int, dict[str, Components]] = {}
    history_components: dict[int, dict[str, Components]] = {}
    context_priors: dict[int, dict[str, tuple[int, str, np.ndarray]]] = {}
    history_priors: dict[int, dict[str, tuple[int, str, np.ndarray]]] = {}
    model_provenance: dict[str, object] = {}
    retained_context = {
        int(row["target_season"]): row
        for row in read_csv(hcp.c13.OUTPUT / "rolling_metrics.csv")
        if row["candidate"] == "P3"
    }
    feature_index = hcp.c13.c12.feature_index()
    tenures = hcp.c13.c12.cached_tenures()

    for season in SEASONS:
        through = season - 1
        context_model, context_instance = hcp.c13.fit_model(
            rows,
            target_season=season,
            trained_through_season=through,
            context_features=hcp.c13.CONTEXT_1_3_FEATURES,
        )
        if (
            context_instance.trained_through_season != through
            or context_instance.target_season != season
        ):
            raise ValueError(f"{season}: Context rolling origin is invalid")
        context_target_rows = [
            row for row in rows if row.season == season and row.subdivision == "fbs"
        ]
        context_prediction_rows = hcp.c13.prediction_rows(
            context_model, context_target_rows, "P3"
        )
        context_prediction_rows = hcp.c13.merge_modeled_and_fallback(
            context_prediction_rows,
            [
                prediction
                for prediction in hcp.c13.fallback_predictions(rows, cold)
                if prediction.season == season
            ],
            "P3",
        )
        context_actual = hcp.c13.score(context_prediction_rows)
        retained = retained_context.get(season)
        if retained is None or any(
            abs(context_actual[metric] - float(retained[metric])) > 1e-8
            for metric in hcp.c13.METRICS
        ):
            raise ValueError(
                f"{season}: rolling Context prior fails retained parity; "
                f"actual={context_actual}, retained={retained}"
            )

        history_model, history_instance = hcp.c13.c12.build_history_prior(
            rows, target_season=season, trained_through_season=through
        )
        if (
            history_instance.trained_through_season != through
            or history_instance.target_season != season
        ):
            raise ValueError(f"{season}: History rolling origin is invalid")
        future_rows = hcp.c13.c12.inference_rows(
            season, through, feature_index, tenures
        )
        promotion, generic = hcp.c13.c12.annual_cold_start_models(
            cold, trained_through_season=through
        )
        history_prediction_rows, _ = hcp.c13.c12.future_predictions(
            future_rows,
            history_model,
            None,
            trained_through_season=through,
            promotion_model=promotion,
            generic_prior=generic,
        )

        c_priors = {
            prediction.team_id: modeled_fallback_prior(prediction)
            for prediction in context_prediction_rows
            if prediction.subdivision == "fbs"
        }
        h_priors = {
            row["team_id"]: (
                len(json.loads(row["pmf"])),
                row["team_name"],
                np.asarray(json.loads(row["pmf"]), dtype=float),
            )
            for row in history_prediction_rows
            if row["subdivision"] == "fbs"
        }
        if set(c_priors) != set(h_priors):
            raise ValueError(f"{season}: rolling C/H FBS populations differ")

        c_components = {
            row.team_id: component_from_model(context_model, row)
            for row in context_target_rows
            if row.lag1_z is not None and len(row.lag1_z)
        }
        h_future = {row.team_id: row for row in future_rows if row.subdivision == "fbs"}
        h_components = {
            team_id: component_from_model(history_model, row)
            for team_id, row in h_future.items()
            if row.lag1_z is not None and len(row.lag1_z)
        }
        if set(c_components) != set(h_components):
            raise ValueError(
                f"{season}: comparable fitted C/H populations differ: "
                f"Context-only={sorted(set(c_components) - set(h_components))}, "
                f"History-only={sorted(set(h_components) - set(c_components))}"
            )
        if not set(c_components) <= set(c_priors):
            raise ValueError(
                f"{season}: fitted Context components exceed prior population"
            )
        for team_id, component in c_components.items():
            if component.population != c_priors[team_id][0] or not np.allclose(
                component.pmf,
                c_priors[team_id][2],
                atol=RECONSTRUCTION_TOLERANCE,
                rtol=0,
            ):
                raise ValueError(
                    f"{season} {team_id}: Context PMF reconstruction failed"
                )
        for team_id, component in h_components.items():
            if component.population != h_priors[team_id][0] or not np.allclose(
                component.pmf, h_priors[team_id][2], atol=2e-12, rtol=0
            ):
                # History's retained prediction CSV rounds PMFs to 12 decimal
                # places. The component-level model output remains unrounded.
                raise ValueError(
                    f"{season} {team_id}: History PMF reconstruction failed"
                )

        context_components[season] = c_components
        history_components[season] = h_components
        context_priors[season] = c_priors
        history_priors[season] = h_priors

        context_metadata = context_model.metadata()
        history_metadata = history_model.metadata()
        context_location = context_metadata.get("location_feature_names")
        context_scale = context_metadata.get("scale_feature_names")
        history_location = (
            history_metadata.get("location_feature_names")
            or history_metadata["feature_names"]
        )
        history_scale = (
            history_metadata.get("scale_feature_names")
            or history_metadata["feature_names"]
        )
        if context_location != list(hcp.c13.LOCATION_FEATURE_NAMES):
            raise ValueError(
                "fitted Context metadata does not match frozen location features"
            )
        if context_scale != list(hcp.c13.SCALE_FEATURE_NAMES):
            raise ValueError(
                "fitted Context metadata does not match frozen scale features"
            )
        if (
            history_location != history_metadata["feature_names"]
            or history_scale != history_metadata["feature_names"]
        ):
            raise ValueError(
                "fitted History metadata does not show its features in both models"
            )
        model_provenance[str(season)] = {
            "trained_through_season": through,
            "context_instance": context_instance.__dict__,
            "history_instance": history_instance.__dict__,
            "context_model_sha256": sha256_json(context_metadata),
            "history_model_sha256": sha256_json(history_metadata),
            "context_location_features": context_location,
            "context_scale_features": context_scale,
            "history_location_features": history_location,
            "history_scale_features": history_scale,
            "context_optimizer": context_metadata.get("optimizer"),
            "history_optimizer": history_metadata.get("optimizer"),
        }

    return (
        context_components,
        history_components,
        context_priors,
        history_priors,
        model_provenance,
    )


def target_team_pmfs(
    seasons: tuple[int, ...], target_path: Path
) -> dict[int, dict[str, np.ndarray]]:
    return {season: hcp.load_targets(season, target_path) for season in seasons}


def hybrid_arms(
    context: dict[str, Components],
    history: dict[str, Components],
    c_priors: dict[str, tuple[int, str, np.ndarray]],
    h_priors: dict[str, tuple[int, str, np.ndarray]],
    secondary: bool = False,
) -> tuple[dict[str, dict[str, np.ndarray]], set[str]]:
    """Construct primary (and optionally decomposed) priors without refitting."""
    keys = set(c_priors)
    if keys != set(h_priors):
        raise ValueError("C/H priors must have identical FBS IDs")
    common = set(context) & set(history)
    if set(context) != set(history):
        raise ValueError("C/H fitted component populations must match")
    arms = {name: {} for name in PRIMARY_ARMS}
    for team_id in sorted(keys):
        if team_id not in common:
            c_fallback = c_priors[team_id][2]
            h_fallback = h_priors[team_id][2]
            # Keep each ordinary arm's real fallback. Hybrids have no fitted
            # components to swap here, so preserve the fallback associated
            # with their location-center lineage and exclude these teams from
            # matched-primary scoring.
            arms["CC"][team_id] = c_fallback.copy()
            arms["HH"][team_id] = h_fallback.copy()
            arms["C_center_H_uncertainty"][team_id] = c_fallback.copy()
            arms["H_center_C_uncertainty"][team_id] = h_fallback.copy()
            continue
        c, h = context[team_id], history[team_id]
        mu_c, mu_h = float(np.mean(c.locations)), float(np.mean(h.locations))
        d_c, d_h = c.locations - mu_c, h.locations - mu_h
        arms["CC"][team_id] = normal_mixture_pmf(mu_c + d_c, c.scale, c.population)
        arms["HH"][team_id] = normal_mixture_pmf(mu_h + d_h, h.scale, h.population)
        if not np.allclose(
            arms["CC"][team_id], c.pmf, atol=RECONSTRUCTION_TOLERANCE, rtol=0
        ):
            raise ValueError(f"{team_id}: synthetic CC does not reconstruct Context")
        if not np.allclose(
            arms["HH"][team_id], h.pmf, atol=RECONSTRUCTION_TOLERANCE, rtol=0
        ):
            raise ValueError(f"{team_id}: synthetic HH does not reconstruct History")
        arms["C_center_H_uncertainty"][team_id] = normal_mixture_pmf(
            mu_c + d_h, h.scale, c.population
        )
        arms["H_center_C_uncertainty"][team_id] = normal_mixture_pmf(
            mu_h + d_c, c.scale, h.population
        )

    if secondary:
        for arm in SECONDARY_ARMS:
            arms[arm] = {}
        for team_id in sorted(keys):
            if team_id not in common:
                c_fallback, h_fallback = c_priors[team_id][2], h_priors[team_id][2]
                for arm in SECONDARY_ARMS:
                    fallback = c_fallback if arm.startswith("C_center") else h_fallback
                    arms[arm][team_id] = fallback.copy()
                continue
            c, h = context[team_id], history[team_id]
            mu_c, mu_h = float(np.mean(c.locations)), float(np.mean(h.locations))
            d_c, d_h = c.locations - mu_c, h.locations - mu_h
            arms["C_center_H_offsets_C_scale"][team_id] = normal_mixture_pmf(
                mu_c + d_h, c.scale, c.population
            )
            arms["C_center_C_offsets_H_scale"][team_id] = normal_mixture_pmf(
                mu_c + d_c, h.scale, c.population
            )
            arms["H_center_H_offsets_C_scale"][team_id] = normal_mixture_pmf(
                mu_h + d_h, c.scale, h.population
            )
            arms["H_center_C_offsets_H_scale"][team_id] = normal_mixture_pmf(
                mu_h + d_c, h.scale, h.population
            )
    return arms, common


def prior_decomposition_rows(
    season: int,
    context: dict[str, Components],
    history: dict[str, Components],
    c_priors: dict[str, tuple[int, str, np.ndarray]],
    h_priors: dict[str, tuple[int, str, np.ndarray]],
    targets: dict[str, np.ndarray],
    transfer: dict[str, str],
) -> list[dict[str, object]]:
    result = []
    for team_id in sorted(targets):
        c_population, team_name, c_pmf = c_priors[team_id]
        h_population, _h_name, h_pmf = h_priors[team_id]
        target = targets[team_id]
        if c_population != len(target) or h_population != len(target):
            raise ValueError(f"{season} {team_id}: rank support does not match target")
        c_score, h_score = score_pmf(c_pmf, target), score_pmf(h_pmf, target)
        c, h = context.get(team_id), history.get(team_id)
        row: dict[str, object] = {
            "season": season,
            "team_id": team_id,
            "team_name": team_name,
            "target_population": len(target),
            "transfer_completeness_group": transfer[team_id],
            "component_status": "fitted"
            if c is not None and h is not None
            else "cold_start_fallback",
            "target_pmf_sha256": sha256_json(target.tolist()),
        }
        for family, pmf, score in (
            ("context", c_pmf, c_score),
            ("history", h_pmf, h_score),
        ):
            row.update(
                {f"{family}_prior_{name}": value for name, value in score.items()}
            )
            row[f"{family}_prior_pmf_sha256"] = sha256_json(pmf.tolist())
        if c is not None and h is not None:
            mu_c, mu_h = float(np.mean(c.locations)), float(np.mean(h.locations))
            d_c, d_h = c.locations - mu_c, h.locations - mu_h
            row.update(
                {
                    "context_conditional_location_points": json_values(c.locations),
                    "context_location_center": mu_c,
                    "context_centered_location_offsets": json_values(d_c),
                    "context_location_mixture_variance": float(np.var(c.locations)),
                    "context_location_mixture_sd": float(np.std(c.locations)),
                    "context_conditional_residual_scale": c.scale,
                    "context_total_latent_variance": float(
                        np.var(c.locations) + c.scale**2
                    ),
                    "history_conditional_location_points": json_values(h.locations),
                    "history_location_center": mu_h,
                    "history_centered_location_offsets": json_values(d_h),
                    "history_location_mixture_variance": float(np.var(h.locations)),
                    "history_location_mixture_sd": float(np.std(h.locations)),
                    "history_conditional_residual_scale": h.scale,
                    "history_total_latent_variance": float(
                        np.var(h.locations) + h.scale**2
                    ),
                    "context_minus_history_location_center": mu_c - mu_h,
                    "context_minus_history_location_mixture_variance": float(
                        np.var(c.locations) - np.var(h.locations)
                    ),
                    "context_minus_history_residual_scale": c.scale - h.scale,
                    "context_minus_history_total_latent_variance": float(
                        np.var(c.locations)
                        + c.scale**2
                        - np.var(h.locations)
                        - h.scale**2
                    ),
                    "context_location_center_feature_input_sha256": sha256_json(
                        c.features
                    ),
                    "history_location_center_feature_input_sha256": sha256_json(
                        h.features
                    ),
                }
            )
        else:
            for family in ("context", "history"):
                for name in (
                    "conditional_location_points",
                    "location_center",
                    "centered_location_offsets",
                    "location_mixture_variance",
                    "location_mixture_sd",
                    "conditional_residual_scale",
                    "total_latent_variance",
                ):
                    row[f"{family}_{name}"] = ""
        row.update(
            {
                "prior_nll_gap_context_minus_history": c_score["nll"] - h_score["nll"],
                "prior_crps_gap_context_minus_history": c_score["crps"]
                - h_score["crps"],
                "prior_expected_rank_error_gap_context_minus_history": c_score[
                    "expected_rank_error"
                ]
                - h_score["expected_rank_error"],
                "prior_width_gap_context_minus_history": c_score["interval_80_width"]
                - h_score["interval_80_width"],
                "prior_entropy_gap_context_minus_history": c_score["entropy"]
                - h_score["entropy"],
                "context_relative_sharpness": h_score["interval_80_width"]
                - c_score["interval_80_width"],
                "context_location_error": c_score["expected_rank_error"],
                "history_location_error": h_score["expected_rank_error"],
            }
        )
        result.append(row)
    return result


def hybrid_prior_rows(
    season: int,
    arms: dict[str, dict[str, np.ndarray]],
    component_keys: set[str],
    c_components: dict[str, Components],
    h_components: dict[str, Components],
    targets: dict[str, np.ndarray],
    names: dict[str, str],
) -> list[dict[str, object]]:
    rows = []
    for arm, pmfs in arms.items():
        for team_id, pmf in sorted(pmfs.items()):
            metrics = score_pmf(pmf, targets[team_id])
            if team_id not in component_keys:
                center_source = offsets_source = scale_source = "native_fallback"
            elif arm in {"CC"}:
                center_source, offsets_source, scale_source = (
                    "Context",
                    "Context",
                    "Context",
                )
            elif arm == "HH":
                center_source, offsets_source, scale_source = (
                    "History",
                    "History",
                    "History",
                )
            elif arm == "C_center_H_uncertainty":
                center_source, offsets_source, scale_source = (
                    "Context",
                    "History",
                    "History",
                )
            elif arm == "H_center_C_uncertainty":
                center_source, offsets_source, scale_source = (
                    "History",
                    "Context",
                    "Context",
                )
            elif arm.endswith("H_offsets_C_scale"):
                center_source = "Context" if arm.startswith("C_center") else "History"
                offsets_source, scale_source = "History", "Context"
            elif arm.endswith("C_offsets_H_scale"):
                center_source = "Context" if arm.startswith("C_center") else "History"
                offsets_source, scale_source = "Context", "History"
            else:
                raise ValueError(f"unknown arm {arm}")
            rows.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "team_name": names[team_id],
                    "arm": arm,
                    "component_status": "fitted"
                    if team_id in component_keys
                    else "cold_start_fallback",
                    "center_source": center_source,
                    "location_offsets_source": offsets_source,
                    "conditional_scale_source": scale_source,
                    "prior_pmf": json_values(pmf),
                    "prior_pmf_sha256": sha256_json(pmf.tolist()),
                    **{f"prior_{name}": value for name, value in metrics.items()},
                }
            )
    return rows


def quantile_labels(
    rows: list[dict[str, object]], field: str
) -> tuple[dict[str, int], list[float]]:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    cuts = [float(value) for value in np.quantile(values, [0.25, 0.5, 0.75])]
    labels = {
        str(row["team_id"]): int(
            np.searchsorted(cuts, float(row[field]), side="right") + 1
        )
        for row in rows
    }
    return labels, cuts


def checkpoint_inputs(root: Path, season: int) -> list[dict[str, object]]:
    panel = json.loads((hcp.BACKTEST / f"{season}_rolling.json").read_text())
    return [
        {"checkpoint": index, "cutoff": row["cutoff"]}
        for index, row in enumerate(panel["cutoffs"], 1)
    ]


def stable_evidence_hash(rows: list[dict[str, str]]) -> str:
    evidence_fields = (
        "id",
        "season",
        "startDate",
        "homeId",
        "homeTeam",
        "homeClassification",
        "homePoints",
        "awayId",
        "awayTeam",
        "awayClassification",
        "awayPoints",
        "neutralSite",
    )
    canonical = [
        {field: str(row.get(field, "")) for field in evidence_fields}
        for row in sorted(rows, key=lambda item: int(item["id"]))
    ]
    return sha256_json(canonical)


def run_posteriors(
    root: Path,
    season: int,
    prior_arms: dict[str, dict[str, np.ndarray]],
    component_keys: set[str],
    names: dict[str, str],
    targets: dict[str, np.ndarray],
    target_path: Path,
    likelihood,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    team_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    fcs_population = hcp.fcs_population(season, target_path)
    arm_ids: dict[str, tuple[str, ...]] = {}
    for point in checkpoint_inputs(root, season):
        checkpoint = int(point["checkpoint"])
        cutoff = datetime.fromisoformat(str(point["cutoff"]))
        games, included, excluded_lower, _ = filter_games(
            root, season, cutoff, "weekly"
        )
        games, included, audit = hcp.cutoff_safe_games(games, included, cutoff)
        audit_rows.extend(
            {
                "season": season,
                "checkpoint": checkpoint,
                "cutoff": cutoff.isoformat(),
                **row,
            }
            for row in audit
        )
        future_fcs = _scheduled_future_fcs_rows(root, season, cutoff, "weekly")
        game_hash = stable_evidence_hash(included)
        included_ids = tuple(sorted((str(row["id"]) for row in included), key=int))
        for arm, prior_map in prior_arms.items():
            teams = [
                Team(team_id, names[team_id], "fbs", pmf.copy())
                for team_id, pmf in sorted(prior_map.items())
            ]
            meta = {team_id: {"team_name": names[team_id]} for team_id in prior_map}
            teams, fallback_ids = add_fcs_fallbacks(
                teams, meta, included, fcs_population, future_fcs
            )
            ids = tuple(sorted(team.team_id for team in teams))
            if arm_ids and ids != next(iter(arm_ids.values())):
                raise ValueError(
                    f"{season} checkpoint {checkpoint}: arm team populations differ"
                )
            arm_ids[arm] = ids
            result = infer_posterior(teams, games, likelihood, **INFERENCE)
            if not result.converged:
                raise RuntimeError(
                    f"nonconverged {season} checkpoint {checkpoint} arm {arm} "
                    f"after {result.iterations} iterations"
                )
            if set(result.pmfs) != set(ids):
                raise ValueError(
                    f"{season} checkpoint {checkpoint} {arm}: posterior keys differ"
                )
            for team_id, prior_pmf in sorted(prior_map.items()):
                target = targets[team_id]
                prior_metrics = score_pmf(prior_pmf, target)
                posterior_metrics = score_pmf(result.pmfs[team_id], target)
                row: dict[str, object] = {
                    "season": season,
                    "checkpoint": checkpoint,
                    "cutoff": cutoff.isoformat(),
                    "period": cutoff.strftime("%B").lower(),
                    "team_id": team_id,
                    "team_name": names[team_id],
                    "arm": arm,
                    "prior_to_posterior_nll_improvement": prior_metrics["nll"]
                    - posterior_metrics["nll"],
                    "prior_to_posterior_crps_improvement": prior_metrics["crps"]
                    - posterior_metrics["crps"],
                    "location_progress": prior_metrics["expected_rank_error"]
                    - posterior_metrics["expected_rank_error"],
                    "expected_rank_shift": posterior_metrics["expected_rank"]
                    - prior_metrics["expected_rank"],
                    "absolute_expected_rank_shift": abs(
                        posterior_metrics["expected_rank"]
                        - prior_metrics["expected_rank"]
                    ),
                    "posterior_to_prior_width_ratio": posterior_metrics[
                        "interval_80_width"
                    ]
                    / max(prior_metrics["interval_80_width"], 1e-15),
                    "entropy_reduction": prior_metrics["entropy"]
                    - posterior_metrics["entropy"],
                    "component_status": "fitted"
                    if team_id in component_keys
                    else "cold_start_fallback",
                    **{f"prior_{name}": value for name, value in prior_metrics.items()},
                    **{
                        f"posterior_{name}": value
                        for name, value in posterior_metrics.items()
                    },
                }
                team_rows.append(row)
            evidence_rows.append(
                {
                    "season": season,
                    "checkpoint": checkpoint,
                    "cutoff": cutoff.isoformat(),
                    "arm": arm,
                    "included_games": len(included),
                    "included_game_ids_sha256": sha256_json(included_ids),
                    "included_game_rows_sha256": game_hash,
                    "excluded_near_cutoff_games": len(audit),
                    "excluded_lower_division_games": excluded_lower,
                    "fcs_fallback_count": len(fallback_ids),
                    "fcs_fallback_ids_sha256": sha256_json(sorted(fallback_ids)),
                    "total_inference_team_count": len(teams),
                    "likelihood_sha256": sha256(hcp.LIKELIHOOD),
                    "inference_max_iterations": INFERENCE["max_iterations"],
                    "inference_tolerance": INFERENCE["tolerance"],
                    "inference_damping": INFERENCE["damping"],
                    "iterations": result.iterations,
                    "converged": result.converged,
                    "max_message_delta": result.max_message_delta,
                }
            )
        print(f"completed {season} checkpoint {checkpoint}", flush=True)
        # This catches future edits that accidentally make arm evidence differ.
        cohort = [
            row
            for row in evidence_rows
            if row["season"] == season and row["checkpoint"] == checkpoint
        ]
        if len({row["included_game_rows_sha256"] for row in cohort}) != 1:
            raise ValueError(
                f"{season} checkpoint {checkpoint}: evidence rows differ by arm"
            )
        if len({row["fcs_fallback_ids_sha256"] for row in cohort}) != 1:
            raise ValueError(
                f"{season} checkpoint {checkpoint}: FCS fallback sets differ by arm"
            )
    return team_rows, evidence_rows, audit_rows


def baseline_parity_rows(team_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline = {
        (int(row["season"]), int(row["checkpoint"]), row["team_id"]): row
        for row in read_csv(BASELINE_TEAMS)
        if row["panel"] == "rolling_origin"
    }
    fields = (
        "nll",
        "crps",
        "expected_rank_error",
        "interval_80_target_mass",
        "interval_80_width",
        "expected_rank",
    )
    comparison = []
    for arm, prefix in (("CC", "context"), ("HH", "history")):
        grouped: dict[tuple[int, int, str], list[float]] = defaultdict(list)
        for row in team_rows:
            if row["arm"] != arm:
                continue
            key = (int(row["season"]), int(row["checkpoint"]), str(row["team_id"]))
            old = baseline.get(key)
            if old is None:
                raise ValueError(f"missing #140 paired baseline row {key}")
            for stage, source in (("prior", "prior"), ("posterior", "posterior")):
                for field in fields:
                    actual_name = f"{source}_{field}"
                    old_name = f"{prefix}_{stage}_{'interval_80_coverage' if field == 'interval_80_target_mass' else 'interval_80_width' if field == 'interval_80_width' else 'expected_rank_mae' if field == 'expected_rank_error' else field}"
                    observed = float(row[actual_name])
                    expected = float(old[old_name])
                    grouped[
                        (int(row["season"]), int(row["checkpoint"]), f"{stage}_{field}")
                    ].append(abs(observed - expected))
        for (season, checkpoint, metric), diffs in sorted(grouped.items()):
            maximum = max(diffs)
            comparison.append(
                {
                    "season": season,
                    "checkpoint": checkpoint,
                    "arm": arm,
                    "metric": metric,
                    "max_absolute_error": maximum,
                    "tolerance": BASELINE_TOLERANCE,
                    "passed": maximum <= BASELINE_TOLERANCE,
                }
            )
            if maximum > BASELINE_TOLERANCE:
                raise ValueError(
                    f"#140 parity failed for {season} checkpoint {checkpoint} "
                    f"{arm} {metric}: {maximum}"
                )
    baseline_summaries = {
        (int(row["season"]), int(row["checkpoint"])): row
        for row in read_csv(BASELINE_SUMMARIES)
        if row["panel"] == "rolling_origin" and row["transfer_group"] == "all"
    }
    by_checkpoint: dict[tuple[int, int], dict[str, list[dict[str, object]]]] = (
        defaultdict(lambda: {"CC": [], "HH": []})
    )
    for row in team_rows:
        if row["arm"] in {"CC", "HH"}:
            by_checkpoint[(int(row["season"]), int(row["checkpoint"]))][
                str(row["arm"])
            ].append(row)
    for (season, checkpoint), arms in sorted(by_checkpoint.items()):
        context = sorted(arms["CC"], key=lambda row: str(row["team_id"]))
        history = sorted(arms["HH"], key=lambda row: str(row["team_id"]))
        correlation = float(
            spearmanr(
                [float(row["posterior_expected_rank"]) for row in context],
                [float(row["posterior_expected_rank"]) for row in history],
            ).statistic
        )
        expected = float(
            baseline_summaries[season, checkpoint]["posterior_expected_rank_spearman"]
        )
        error = abs(correlation - expected)
        comparison.append(
            {
                "season": season,
                "checkpoint": checkpoint,
                "arm": "CC_vs_HH",
                "metric": "posterior_expected_rank_spearman",
                "max_absolute_error": error,
                "tolerance": BASELINE_TOLERANCE,
                "passed": error <= BASELINE_TOLERANCE,
            }
        )
        if error > BASELINE_TOLERANCE:
            raise ValueError(
                f"#140 rank-correlation parity failed for {season} "
                f"checkpoint {checkpoint}: {error}"
            )
    return comparison


def make_summaries(team_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    all_arms = sorted({str(row["arm"]) for row in team_rows})
    groups: list[tuple[str, dict[str, object], list[dict[str, object]]]] = []
    seasons = sorted({int(row["season"]) for row in team_rows})
    for season in seasons:
        selected = [row for row in team_rows if int(row["season"]) == season]
        for checkpoint in sorted({int(row["checkpoint"]) for row in selected}):
            groups.append(
                (
                    "season_checkpoint",
                    {"season": season, "checkpoint": checkpoint},
                    [row for row in selected if int(row["checkpoint"]) == checkpoint],
                )
            )
        for period in sorted({str(row["period"]) for row in selected}):
            groups.append(
                (
                    "season_period",
                    {"season": season, "period": period},
                    [row for row in selected if row["period"] == period],
                )
            )
    for checkpoint in sorted({int(row["checkpoint"]) for row in team_rows}):
        groups.append(
            (
                "pooled_checkpoint",
                {"checkpoint": checkpoint},
                [row for row in team_rows if int(row["checkpoint"]) == checkpoint],
            )
        )
    for period in sorted({str(row["period"]) for row in team_rows}):
        groups.append(
            (
                "pooled_period",
                {"period": period},
                [row for row in team_rows if row["period"] == period],
            )
        )

    metric_fields = (
        "prior_nll",
        "posterior_nll",
        "prior_crps",
        "posterior_crps",
        "prior_expected_rank_error",
        "posterior_expected_rank_error",
        "prior_interval_80_target_mass",
        "posterior_interval_80_target_mass",
        "prior_interval_80_width",
        "posterior_interval_80_width",
        "prior_entropy",
        "posterior_entropy",
        "prior_rank_standard_deviation",
        "posterior_rank_standard_deviation",
        "prior_to_posterior_nll_improvement",
        "prior_to_posterior_crps_improvement",
        "location_progress",
        "expected_rank_shift",
        "absolute_expected_rank_shift",
        "posterior_to_prior_width_ratio",
        "entropy_reduction",
    )
    output = []
    for summary_type, dimensions, selected in groups:
        for arm in all_arms:
            rows = [row for row in selected if row["arm"] == arm]
            for population in ("all_fbs", "matched_primary", "cold_start_fallback"):
                subset = (
                    rows
                    if population == "all_fbs"
                    else [
                        row
                        for row in rows
                        if row["component_status"]
                        == (
                            "fitted"
                            if population == "matched_primary"
                            else "cold_start_fallback"
                        )
                    ]
                )
                if not subset:
                    continue
                result: dict[str, object] = {
                    "summary_type": summary_type,
                    **dimensions,
                    "arm": arm,
                    "population": population,
                    "team_seasons": len(subset),
                    "weighting": "equal team-season; each team-season contributes one row",
                }
                if subset:
                    result["period"] = (
                        subset[0]["period"]
                        if len({row["period"] for row in subset}) == 1
                        else "pooled"
                    )
                    result["cutoff"] = (
                        subset[0]["cutoff"]
                        if len({row["cutoff"] for row in subset}) == 1
                        else ""
                    )
                for metric in metric_fields:
                    result[f"mean_{metric}"] = float(
                        np.mean([float(row[metric]) for row in subset])
                    )
                predicted_ranks = [
                    float(row["posterior_expected_rank"]) for row in subset
                ]
                target_ranks = [
                    float(row["posterior_target_expected_rank"]) for row in subset
                ]
                result["rank_correlation_team_seasons"] = len(subset)
                result["posterior_expected_rank_target_spearman"] = (
                    float(spearmanr(predicted_ranks, target_ranks).statistic)
                    if len(subset) > 1
                    and np.std(predicted_ranks) > 0
                    and np.std(target_ranks) > 0
                    else ""
                )
                cc_by_key = {
                    (
                        int(row["season"]),
                        int(row["checkpoint"]),
                        str(row["team_id"]),
                    ): float(row["posterior_expected_rank"])
                    for row in selected
                    if row["arm"] == "CC"
                }
                paired = [
                    (
                        float(row["posterior_expected_rank"]),
                        cc_by_key[
                            (
                                int(row["season"]),
                                int(row["checkpoint"]),
                                str(row["team_id"]),
                            )
                        ],
                    )
                    for row in subset
                    if (
                        int(row["season"]),
                        int(row["checkpoint"]),
                        str(row["team_id"]),
                    )
                    in cc_by_key
                ]
                result["posterior_expected_rank_CC_spearman"] = (
                    float(
                        spearmanr(
                            [item[0] for item in paired],
                            [item[1] for item in paired],
                        ).statistic
                    )
                    if len(paired) > 1
                    and np.std([item[0] for item in paired]) > 0
                    and np.std([item[1] for item in paired]) > 0
                    else ""
                )
                output.append(result)
    return output


def reversal_rows(team_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    indexed = {
        (
            int(row["season"]),
            int(row["checkpoint"]),
            str(row["team_id"]),
            str(row["arm"]),
        ): row
        for row in team_rows
    }
    keys = sorted({(int(row["season"]), int(row["checkpoint"])) for row in team_rows})
    output = []
    for season, checkpoint in keys:
        cutoff = next(
            str(row["cutoff"])
            for row in team_rows
            if int(row["season"]) == season and int(row["checkpoint"]) == checkpoint
        )
        date = datetime.fromisoformat(cutoff)
        teams = sorted(
            team_id
            for s, cp, team_id, arm in indexed
            if s == season and cp == checkpoint and arm == "CC"
        )
        classified: dict[str, list[float]] = defaultdict(list)
        total_gaps = []
        for team_id in teams:
            c = indexed[season, checkpoint, team_id, "CC"]
            h = indexed[season, checkpoint, team_id, "HH"]
            prior_gap = float(c["prior_nll"]) - float(h["prior_nll"])
            posterior_gap = float(c["posterior_nll"]) - float(h["posterior_nll"])
            prior_status = (
                "context_better"
                if prior_gap < -1e-12
                else "history_better"
                if prior_gap > 1e-12
                else "tied"
            )
            posterior_status = (
                "context_better"
                if posterior_gap < -1e-12
                else "history_better"
                if posterior_gap > 1e-12
                else "tied"
            )
            if prior_status == "tied":
                category = "tied_prior"
            elif posterior_status == "tied":
                category = "tied_posterior"
            elif (
                prior_status == "context_better"
                and posterior_status == "context_better"
            ):
                category = "Context better prior to Context better posterior"
            elif (
                prior_status == "context_better"
                and posterior_status == "history_better"
            ):
                category = "Context better prior to History better posterior"
            elif (
                prior_status == "history_better"
                and posterior_status == "history_better"
            ):
                category = "History better prior to History better posterior"
            else:
                category = "History better prior to Context better posterior"
            classified[category].append(posterior_gap)
            total_gaps.append(posterior_gap)
        total_sum = float(np.sum(total_gaps))
        meaningful = date.month == 9 or checkpoint == max(
            cp for s, cp in keys if s == season
        )
        for category in (
            "Context better prior to Context better posterior",
            "Context better prior to History better posterior",
            "History better prior to History better posterior",
            "History better prior to Context better posterior",
            "tied_prior",
            "tied_posterior",
        ):
            gaps = classified.get(category, [])
            output.append(
                {
                    "season": season,
                    "checkpoint": checkpoint,
                    "cutoff": cutoff,
                    "period": date.strftime("%B").lower(),
                    "first_observed_september_or_final": meaningful,
                    "category": category,
                    "team_seasons": len(gaps),
                    "share_of_population": len(gaps) / max(len(teams), 1),
                    "mean_c_minus_h_posterior_nll_gap": float(np.mean(gaps))
                    if gaps
                    else "",
                    "median_c_minus_h_posterior_nll_gap": float(np.median(gaps))
                    if gaps
                    else "",
                    "sum_c_minus_h_posterior_nll_gap": float(np.sum(gaps))
                    if gaps
                    else 0.0,
                    "contribution_to_pooled_gap": (
                        float(np.sum(gaps) / total_sum)
                        if gaps and abs(total_sum) > 1e-12
                        else ""
                    ),
                    "population_pooled_mean_gap": total_sum / max(len(teams), 1),
                    "population": len(teams),
                }
            )
    return output


def explanatory_rows(
    prior_rows: list[dict[str, object]], team_rows: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    strata: list[dict[str, object]] = []
    correlations: list[dict[str, object]] = []
    priors = {int(row["season"]): [] for row in prior_rows}
    for row in prior_rows:
        priors[int(row["season"])].append(row)
    labels: dict[tuple[int, str], dict[str, int]] = {}
    cut_report: dict[int, dict[str, list[float]]] = {}
    for season, rows in priors.items():
        sharp, sharp_cuts = quantile_labels(rows, "context_relative_sharpness")
        error, error_cuts = quantile_labels(
            rows, "prior_expected_rank_error_gap_context_minus_history"
        )
        labels[(season, "sharpness")] = sharp
        labels[(season, "error")] = error
        cut_report[season] = {
            "sharpness_cuts": sharp_cuts,
            "error_gap_cuts": error_cuts,
        }

    prior_index = {(int(row["season"]), str(row["team_id"])): row for row in prior_rows}
    for season in sorted(priors):
        season_checkpoints = sorted(
            {
                (int(row["checkpoint"]), str(row["arm"]))
                for row in team_rows
                if int(row["season"]) == season
            }
        )
        for checkpoint, arm in season_checkpoints:
            selected = [
                row
                for row in team_rows
                if int(row["season"]) == season
                and int(row["checkpoint"]) == checkpoint
                and row["arm"] == arm
            ]
            group_values: dict[str, list[dict[str, object]]] = defaultdict(list)
            paired_rows = {
                str(row["team_id"]): row
                for row in team_rows
                if int(row["season"]) == season
                and int(row["checkpoint"]) == checkpoint
                and row["arm"] == "CC"
            }
            history_rows = {
                str(row["team_id"]): row
                for row in team_rows
                if int(row["season"]) == season
                and int(row["checkpoint"]) == checkpoint
                and row["arm"] == "HH"
            }
            for row in selected:
                team_id = str(row["team_id"])
                sharp_q = labels[(season, "sharpness")][team_id]
                error_q = labels[(season, "error")][team_id]
                group_values[f"sharpness_q{sharp_q}"].append(row)
                group_values[f"preseason_error_gap_q{error_q}"].append(row)
                group_values[f"sharpness_q{sharp_q}_x_error_gap_q{error_q}"].append(row)
                prior = prior_index[season, team_id]
                group_values[
                    f"transfer_completeness_{prior['transfer_completeness_group']}"
                ].append(row)
            for stratum, members in sorted(group_values.items()):
                member_gaps = [
                    float(paired_rows[str(row["team_id"])]["posterior_nll"])
                    - float(history_rows[str(row["team_id"])]["posterior_nll"])
                    for row in members
                    if str(row["team_id"]) in paired_rows
                    and str(row["team_id"]) in history_rows
                ]
                strata.append(
                    {
                        "season": season,
                        "checkpoint": checkpoint,
                        "arm": arm,
                        "stratum": stratum,
                        "team_seasons": len(members),
                        "mean_prior_nll": float(
                            np.mean([float(row["prior_nll"]) for row in members])
                        ),
                        "mean_posterior_nll": float(
                            np.mean([float(row["posterior_nll"]) for row in members])
                        ),
                        "mean_prior_to_posterior_nll_improvement": float(
                            np.mean(
                                [
                                    float(row["prior_to_posterior_nll_improvement"])
                                    for row in members
                                ]
                            )
                        ),
                        "mean_posterior_c_minus_h_gap": float(np.mean(member_gaps))
                        if member_gaps
                        else "",
                        "mean_location_progress": float(
                            np.mean(
                                [float(row["location_progress"]) for row in members]
                            )
                        ),
                        "mean_absolute_expected_rank_shift": float(
                            np.mean(
                                [
                                    float(row["absolute_expected_rank_shift"])
                                    for row in members
                                ]
                            )
                        ),
                        "mean_posterior_to_prior_width_ratio": float(
                            np.mean(
                                [
                                    float(row["posterior_to_prior_width_ratio"])
                                    for row in members
                                ]
                            )
                        ),
                    }
                )
    for season in sorted(priors):
        checkpoints = sorted(
            {
                int(row["checkpoint"])
                for row in team_rows
                if int(row["season"]) == season and row["arm"] == "CC"
            }
        )
        for checkpoint in checkpoints:
            c_rows = {
                str(row["team_id"]): row
                for row in team_rows
                if int(row["season"]) == season
                and int(row["checkpoint"]) == checkpoint
                and row["arm"] == "CC"
            }
            h_rows = {
                str(row["team_id"]): row
                for row in team_rows
                if int(row["season"]) == season
                and int(row["checkpoint"]) == checkpoint
                and row["arm"] == "HH"
            }
            predictor_values: dict[str, list[float]] = defaultdict(list)
            predictor_gaps: dict[str, list[float]] = defaultdict(list)
            gaps = []
            for team_id in sorted(c_rows.keys() & h_rows.keys()):
                prior = prior_index[season, team_id]
                posterior_gap = float(c_rows[team_id]["posterior_nll"]) - float(
                    h_rows[team_id]["posterior_nll"]
                )
                gaps.append(posterior_gap)
                predictor_fields = {
                    "prior_nll_gap": "prior_nll_gap_context_minus_history",
                    "context_relative_sharpness": "context_relative_sharpness",
                    "context_error_gap": "prior_expected_rank_error_gap_context_minus_history",
                    "location_center_gap": "context_minus_history_location_center",
                    "context_scale": "context_conditional_residual_scale",
                    "context_mixture_sd": "context_location_mixture_sd",
                    "context_total_variance": "context_total_latent_variance",
                }
                for label, field in predictor_fields.items():
                    value = prior.get(field)
                    if value not in (None, ""):
                        predictor_values[label].append(float(value))
                        predictor_gaps[label].append(posterior_gap)
            correlations.append(
                {
                    "season": season,
                    "checkpoint": checkpoint,
                    "correlation_target": "C_minus_H_posterior_NLL_gap",
                    "team_seasons": len(gaps),
                    **{
                        f"spearman_{name}": (
                            float(spearmanr(values, predictor_gaps[name]).statistic)
                            if len(values) > 1
                            and np.std(values) > 0
                            and np.std(predictor_gaps[name]) > 0
                            else ""
                        )
                        for name, values in predictor_values.items()
                    },
                    **{
                        f"n_{name}": len(values)
                        for name, values in predictor_values.items()
                    },
                }
            )
    for row in strata:
        if "_x_" in str(row["stratum"]):
            parts = str(row["stratum"]).split("_x_")
            if len(parts) == 2:
                sharp_q = int(parts[0].split("q")[-1])
                error_q = int(parts[1].split("q")[-1])
                row["sharpness_quartile"] = sharp_q
                row["preseason_error_gap_quartile"] = error_q
    return strata, correlations, cut_report


def uncertainty_trigger(summary_rows: list[dict[str, object]]) -> dict[str, object]:
    """Predeclared trigger: >=25% gap gain in at least two post-crossover years."""
    indexed = {
        (
            int(row["season"]),
            int(row["checkpoint"]),
            str(row["arm"]),
            str(row["population"]),
        ): row
        for row in summary_rows
        if row["summary_type"] == "season_checkpoint"
        and row["population"] == "matched_primary"
    }
    by_season: dict[int, list[dict[str, float]]] = defaultdict(list)
    candidates = []
    for season in (2023, 2024, 2025):
        checkpoints = sorted(
            cp
            for s, cp, arm, pop in indexed
            if s == season and arm == "CC" and pop == "matched_primary"
        )
        for checkpoint in checkpoints:
            c = indexed[season, checkpoint, "CC", "matched_primary"]
            h = indexed[season, checkpoint, "HH", "matched_primary"]
            ch = indexed[
                season, checkpoint, "C_center_H_uncertainty", "matched_primary"
            ]
            if c.get("period") == "august":
                continue
            gap = float(c["mean_posterior_nll"]) - float(h["mean_posterior_nll"])
            gain = float(c["mean_posterior_nll"]) - float(ch["mean_posterior_nll"])
            share = gain / gap if gap > 1e-12 else 0.0
            candidates.append(
                {
                    "season": season,
                    "checkpoint": checkpoint,
                    "ordinary_gap": gap,
                    "uncertainty_gain": gain,
                    "gain_share": share,
                }
            )
            if gap > 0:
                by_season[season].append({"gap": gap, "gain": gain, "share": share})
    qualifying = []
    season_results = {}
    for season, rows in by_season.items():
        gap = float(np.mean([row["gap"] for row in rows]))
        gain = float(np.mean([row["gain"] for row in rows]))
        share = gain / gap if gap > 1e-12 else 0.0
        season_results[str(season)] = {
            "qualifying_post_crossover_checkpoints": len(rows),
            "mean_ordinary_gap": gap,
            "mean_uncertainty_gain": gain,
            "gain_share": share,
            "qualifies": share >= SECONDARY_TRIGGER_SHARE,
        }
        if share >= SECONDARY_TRIGGER_SHARE:
            qualifying.append(season)
    return {
        "triggered": len(qualifying) >= 2,
        "rule": (
            "Run component decomposition only if C-center/H-uncertainty gains at least "
            "25% of the ordinary C/H posterior NLL gap on average across eligible "
            "post-September checkpoints in at least two of 2023-2025."
        ),
        "qualifying_seasons": sorted(qualifying),
        "by_season": season_results,
        "checkpoint_effects": candidates,
    }


def provenance(
    target_path: Path,
    raw_hashes: dict[str, str],
    model_info: dict[str, object],
    secondary: dict[str, object],
    component_counts: dict[str, object],
) -> dict[str, object]:
    source_paths = [
        hcp.c13.HISTORICAL_TRANSFER_FEATURES,
        hcp.COVERAGE,
        hcp.c13.OUTPUT / "rolling_metrics.csv",
        hcp.c13.OUTPUT / "model_spec.json",
        hcp.c13.OUTPUT / "fitted_model.json",
        ROOT / "data/processed/preseason/history/model_spec.json",
        hcp.BACKTEST / "2022_rolling.json",
        hcp.BACKTEST / "2023_rolling.json",
        hcp.BACKTEST / "2024_rolling.json",
        hcp.BACKTEST / "2025_rolling.json",
        BASELINE_TEAMS,
        BASELINE_SUMMARIES,
        BASELINE_EVIDENCE,
        BASELINE_AUDIT,
    ]
    source_hashes = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in source_paths
        if path.is_file()
    }
    resolved_target = target_path.resolve()
    try:
        target_key = resolved_target.relative_to(ROOT).as_posix()
    except ValueError:
        target_key = f"external_input/{resolved_target.name}"
    source_hashes[target_key] = sha256(resolved_target)
    for path, digest in raw_hashes.items():
        source_hashes[f"raw_games/{Path(path).name}"] = digest
    for path in sorted(hcp.c13.c12.TENURES.glob("*.json")):
        source_hashes[f"raw_coach_tenures/{path.name}"] = sha256(path)
    return {
        "study": "issue-146 Context 1.3 / History 1.1 crossover diagnostic",
        "study_version": "1",
        "target_seasons": list(SEASONS),
        "interpretation": "diagnostic analysis of the already-observed 2022-2025 behavior; not an untouched validation set",
        "models": model_info,
        "context_scale_model_check": {
            "finding": "Fitted Context 1.3 metadata places all Context 1.3 features in location and keeps only History features in scale.",
            "interpretation": "Any Context/History scale difference comes from jointly fitted coefficients; rich Context features do not directly enter Context scale.",
        },
        "inference": INFERENCE,
        "evidence_contract": {
            "source": "Issue #140 retained rolling cutoff panel and cutoff-safe replay helper",
            "result_availability_lag_hours": int(
                hcp.RESULT_AVAILABILITY_LAG.total_seconds() / 3600
            ),
            "fcs_fallback": "same uniform full-FCS rank prior and scheduled-future opponent rule in every arm",
            "arms_fail_closed_on_evidence_or_population_mismatch": True,
        },
        "likelihood": {
            "version": "Historical Likelihood V1",
            "path": hcp.LIKELIHOOD.relative_to(ROOT).as_posix(),
            "sha256": sha256(hcp.LIKELIHOOD),
        },
        "component_population": component_counts,
        "secondary_uncertainty_decomposition": secondary,
        "sources_sha256": source_hashes,
        "script_sha256": sha256(Path(__file__)),
        "software": {
            "python_version_pin": (ROOT / ".python-version").read_text().strip(),
            "study_runner": Path(__file__).relative_to(ROOT).as_posix(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=hcp.TARGETS)
    parser.add_argument("--raw-games", type=Path, default=ROOT / "data/raw/cfbd/games")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()

    required = (BASELINE_TEAMS, BASELINE_SUMMARIES, BASELINE_EVIDENCE, BASELINE_AUDIT)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"issue-140 baseline artifacts are required: {missing}")
    if INFERENCE != hcp.INFERENCE:
        raise ValueError(
            "Issue-146 inference settings drifted from the #140 production defaults"
        )

    target_path = args.targets.resolve()
    if target_path.name != "team_season_rank_distributions.csv":
        raise ValueError(
            "--targets must name the final team-season rank distribution CSV"
        )
    if len(target_path.parents) < 4:
        raise ValueError("--targets path must be inside a data/processed/modeling tree")
    # The historical corpus is gitignored and may live in a neighboring local
    # checkout. Point existing builders at that read-only source without
    # copying it into this worktree.
    data_root = target_path.parents[3]
    hcp.c13.v1.ROOT = data_root
    hcp.c13.c12.MODELING = target_path.parent
    hcp.c13.c12.TENURES = data_root / "data/raw/cfbd/preseason/coach_tenures"

    context_rows, cold, _ = hcp.c13.load_candidate_rows()
    c_components, h_components, c_priors, h_priors, model_info = load_rolling_models(
        context_rows, cold
    )
    targets = target_team_pmfs(SEASONS, args.targets)
    transfer = {
        season: hcp.transfer_groups(season, set(targets[season])) for season in SEASONS
    }
    prior_rows: list[dict[str, object]] = []
    hybrid_prior_output: list[dict[str, object]] = []
    priors_by_season: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    component_counts = {}
    for season in SEASONS:
        if set(c_priors[season]) != set(targets[season]) or set(
            h_priors[season]
        ) != set(targets[season]):
            raise ValueError(f"{season}: prior and final target populations differ")
        prior_rows.extend(
            prior_decomposition_rows(
                season,
                c_components[season],
                h_components[season],
                c_priors[season],
                h_priors[season],
                targets[season],
                transfer[season],
            )
        )
        arms, common = hybrid_arms(
            c_components[season],
            h_components[season],
            c_priors[season],
            h_priors[season],
        )
        if set(arms) != set(PRIMARY_ARMS) or any(
            set(arm_rows) != set(targets[season]) for arm_rows in arms.values()
        ):
            raise ValueError(f"{season}: primary hybrid populations do not match")
        names = {team_id: c_priors[season][team_id][1] for team_id in targets[season]}
        for team_id in common:
            for arm in PRIMARY_ARMS:
                pmf = arms[arm][team_id]
                if (
                    len(pmf) != len(targets[season][team_id])
                    or not np.isfinite(pmf).all()
                    or np.any(pmf < 0)
                ):
                    raise ValueError(f"{season} {team_id} {arm}: hybrid PMF invalid")
                if not np.isclose(pmf.sum(), 1.0, atol=1e-12):
                    raise ValueError(
                        f"{season} {team_id} {arm}: hybrid PMF is not normalized"
                    )
        hybrid_prior_output.extend(
            hybrid_prior_rows(
                season,
                arms,
                common,
                c_components[season],
                h_components[season],
                targets[season],
                names,
            )
        )
        priors_by_season[season] = arms
        component_counts[str(season)] = {
            "all_fbs_target_teams": len(targets[season]),
            "matched_fitted_components": len(common),
            "cold_start_or_unmodeled_fallbacks": len(targets[season]) - len(common),
            "fitted_team_ids_sha256": sha256_json(sorted(common)),
        }

    posterior_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    cutoff_audit_rows: list[dict[str, object]] = []
    with hcp.historical_game_root(args.raw_games) as (game_root, _, _, raw_hashes):
        likelihood = load_pinned_likelihood(hcp.LIKELIHOOD)
        for season in SEASONS:
            names = {
                team_id: c_priors[season][team_id][1] for team_id in targets[season]
            }
            common = set(c_components[season]) & set(h_components[season])
            rows, evidence, audit = run_posteriors(
                game_root,
                season,
                priors_by_season[season],
                common,
                names,
                targets[season],
                args.targets,
                likelihood,
            )
            posterior_rows.extend(rows)
            evidence_rows.extend(evidence)
            cutoff_audit_rows.extend(audit)

    parity = baseline_parity_rows(posterior_rows)
    summary_rows = make_summaries(posterior_rows)
    trigger = uncertainty_trigger(summary_rows)
    if trigger["triggered"]:
        for season in SEASONS:
            secondary_arms, common = hybrid_arms(
                c_components[season],
                h_components[season],
                c_priors[season],
                h_priors[season],
                secondary=True,
            )
            selected = {arm: secondary_arms[arm] for arm in SECONDARY_ARMS}
            names = {
                team_id: c_priors[season][team_id][1] for team_id in targets[season]
            }
            with hcp.historical_game_root(args.raw_games) as (game_root, _, _, _):
                secondary_rows, secondary_evidence, secondary_audit = run_posteriors(
                    game_root,
                    season,
                    selected,
                    common,
                    names,
                    targets[season],
                    args.targets,
                    likelihood,
                )
                posterior_rows.extend(secondary_rows)
                evidence_rows.extend(secondary_evidence)
                cutoff_audit_rows.extend(secondary_audit)
            hybrid_prior_output.extend(
                hybrid_prior_rows(
                    season,
                    selected,
                    common,
                    c_components[season],
                    h_components[season],
                    targets[season],
                    names,
                )
            )
        summary_rows = make_summaries(posterior_rows)

    prior_index = {(int(row["season"]), str(row["team_id"])): row for row in prior_rows}
    for row in posterior_rows:
        prior = prior_index[int(row["season"]), str(row["team_id"])]
        row["context_prior_nll"] = prior["context_prior_nll"]
        row["history_prior_nll"] = prior["history_prior_nll"]
        row["context_prior_error"] = prior["context_prior_expected_rank_error"]
        row["history_prior_error"] = prior["history_prior_expected_rank_error"]
        row["prior_nll_gap_context_minus_history"] = prior[
            "prior_nll_gap_context_minus_history"
        ]
        row["posterior_nll_gap_context_minus_history"] = ""

    # Add one paired C-H posterior gap to every team row; hybrids remain intact.
    post_index = {
        (
            int(row["season"]),
            int(row["checkpoint"]),
            str(row["team_id"]),
            str(row["arm"]),
        ): row
        for row in posterior_rows
    }
    for key, row in post_index.items():
        s, cp, team_id, arm = key
        if (s, cp, team_id, "CC") in post_index and (
            s,
            cp,
            team_id,
            "HH",
        ) in post_index:
            row["posterior_nll_gap_context_minus_history"] = float(
                post_index[s, cp, team_id, "CC"]["posterior_nll"]
                - post_index[s, cp, team_id, "HH"]["posterior_nll"]
            )

    strata, correlations, quartile_cutpoints = explanatory_rows(
        prior_rows, posterior_rows
    )
    reversals = reversal_rows(posterior_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "team_season_prior_decomposition.csv", prior_rows)
    write_csv(args.output / "hybrid_prior_results.csv", hybrid_prior_output)
    write_csv(args.output / "hybrid_posterior_team_results.csv", posterior_rows)
    write_csv(args.output / "hybrid_posterior_summary.csv", summary_rows)
    write_csv(args.output / "confidence_error_strata.csv", strata)
    write_csv(args.output / "confidence_error_correlations.csv", correlations)
    (args.output / "confidence_error_quartile_cutpoints.json").write_text(
        json.dumps(quartile_cutpoints, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output / "reversal_categories.csv", reversals)
    write_csv(args.output / "arm_evidence_and_convergence.csv", evidence_rows)
    write_csv(args.output / "baseline_reproduction_checks.csv", parity)
    write_csv(args.output / "cutoff_audit.csv", cutoff_audit_rows)
    metadata = provenance(
        args.targets,
        raw_hashes,
        model_info,
        trigger,
        component_counts,
    )
    (args.output / "provenance.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "prior_team_seasons": len(prior_rows),
                "posterior_team_rows": len(posterior_rows),
                "primary_baseline_checks": len(parity),
                "secondary_uncertainty_decomposition_triggered": trigger["triggered"],
                "secondary_qualifying_seasons": trigger["qualifying_seasons"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
