"""Research-only transfer-aware Context study for issue #91.

The study keeps the production Context equation and evaluation population
fixed, then adds a small predeclared set of transfer-volume, transfer-talent,
and prior-usage oracle features.  The transfer candidates are predeclared and
compared descriptively.  No candidate is selected from the 2018--2021 panel
because the transfer features have no observed values in candidate training
rows through 2017.

Raw portal and usage payloads are read from a separate transfer-data root and
are never modified.  Outputs are written only to the requested research
directory.  No production prior or current-season artifact is changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_preseason_context_prior_v1_2 as c12
import build_preseason_prior as v1
import build_preseason_prior_v1_1 as h11

from gippyrank.context_ablation import standardized_interaction_rows
from gippyrank.preseason import DirectRankModel, TeamSeason
from gippyrank.transfer_oracle import (
    ALL_FEATURES,
    TALENT_FEATURES,
    VOLUME_FEATURES,
    TransferRecord,
    UsageRecord,
    aggregate_team_features,
    coverage_rows,
    parse_transfer_payload,
    parse_usage_payload,
    portal_provenance,
)

ROOT = Path(__file__).resolve().parents[1]
TARGET_SEASONS = tuple(range(2022, 2026))
DEVELOPMENT_SEASONS = tuple(range(2018, 2022))
FROZEN_TRAIN_THROUGH = 2021
DEVELOPMENT_TRAIN_THROUGH = 2017
DEFAULT_CUTOFF = (8, 15)
SANITY_TOLERANCE = 1e-3
RANDOM_SEED = 7

H_FEATURES = tuple(c12.H_FEATURES)
COACH_FEATURES = tuple(c12.COACH_FEATURES)
RECRUITING_FEATURES = tuple(c12.RECRUITING_FEATURES)
TALENT_CONTEXT_FEATURES = tuple(c12.TALENT_FEATURES)
RP_FEATURES = tuple(c12.RETURNING_FEATURES)
BASE_CONTEXT_FEATURES = (
    *COACH_FEATURES,
    *RECRUITING_FEATURES,
    *TALENT_CONTEXT_FEATURES,
    *RP_FEATURES,
)
C_MINUS_RP_FEATURES = tuple(
    feature for feature in BASE_CONTEXT_FEATURES if feature not in RP_FEATURES
)
QB_FEATURES = tuple(feature for feature in ALL_FEATURES if feature.endswith("_qb"))

METRICS = (
    "nll",
    "crps",
    "expected_rank_mae",
    "median_rank_mae",
    "interval_80_coverage",
    "interval_80_average_width",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    features: tuple[str, ...]
    interaction: tuple[str, str] | None = None
    interaction_name: str | None = None
    transfer_aware: bool = False


def candidate_definitions() -> tuple[Candidate, ...]:
    """The complete candidate set, frozen before held-out scoring."""
    return (
        Candidate("C0_full", BASE_CONTEXT_FEATURES),
        Candidate("C1_minus_rp", C_MINUS_RP_FEATURES),
        Candidate(
            "C2_transfer_volume",
            (*BASE_CONTEXT_FEATURES, *VOLUME_FEATURES),
            transfer_aware=True,
        ),
        Candidate(
            "C3_minus_rp_transfer_volume",
            (*C_MINUS_RP_FEATURES, *VOLUME_FEATURES),
            transfer_aware=True,
        ),
        Candidate(
            "C4_rp_x_transfer_volume",
            (*BASE_CONTEXT_FEATURES,),
            interaction=("returning_pct_ppa", "transfer_in_count"),
            interaction_name="rp_x_transfer_volume",
            transfer_aware=True,
        ),
        Candidate(
            "C5_transfer_talent",
            (*BASE_CONTEXT_FEATURES, *TALENT_FEATURES),
            transfer_aware=True,
        ),
        Candidate(
            "C6_minus_rp_transfer_talent",
            (*C_MINUS_RP_FEATURES, *TALENT_FEATURES),
            transfer_aware=True,
        ),
        Candidate(
            "C7_rp_x_incoming_value",
            (*BASE_CONTEXT_FEATURES,),
            interaction=("returning_pct_ppa", "transfer_in_weighted_rating_sum"),
            interaction_name="rp_x_incoming_transfer_value",
            transfer_aware=True,
        ),
        Candidate(
            "C8_rp_x_net_value",
            (*BASE_CONTEXT_FEATURES,),
            interaction=("returning_pct_ppa", "transfer_net_weighted_rating_sum"),
            interaction_name="rp_x_net_transfer_value",
            transfer_aware=True,
        ),
        Candidate(
            "C9_ad_hoc_rp_usage_hybrid",
            (*C_MINUS_RP_FEATURES, "ad_hoc_rp_usage_hybrid"),
            transfer_aware=True,
        ),
        Candidate(
            "C10_transfer_production",
            (
                *BASE_CONTEXT_FEATURES,
                "transfer_in_prior_usage_sum",
                "transfer_net_prior_usage",
            ),
            transfer_aware=True,
        ),
    )


def added_transfer_feature_names(candidate: Candidate) -> tuple[str, ...]:
    """Return transfer-derived inputs added by a candidate."""
    names = set(candidate.features) & set(ALL_FEATURES)
    if candidate.interaction is not None:
        names.add(candidate.interaction[1])
    return tuple(sorted(names))


def development_transfer_observation_counts(
    rows: Iterable[TeamSeason],
    candidates: Iterable[Candidate],
    *,
    trained_through: int,
) -> dict[str, dict[str, int]]:
    """Count non-missing transfer values available to each training fit."""
    training = [row for row in rows if row.season <= trained_through]
    return {
        candidate.name: {
            feature: sum(row.features.get(feature) is not None for row in training)
            for feature in added_transfer_feature_names(candidate)
        }
        for candidate in candidates
        if candidate.transfer_aware
    }


def select_development_candidate(
    development_summary: list[dict[str, object]],
    observation_counts: dict[str, dict[str, int]],
) -> str:
    """Select only when the development fit has observed transfer values.

    The current issue-91 study intentionally does not call this function:
    its historical transfer training panel has no observations. Keeping the
    guard here prevents a future caller from treating imputed all-missing
    transfer columns as evidence for candidate selection.
    """
    if not any(
        count > 0 for counts in observation_counts.values() for count in counts.values()
    ):
        raise ValueError(
            "cannot select a transfer candidate: development training rows "
            "contain zero observed transfer-feature values"
        )
    eligible = [
        row
        for row in development_summary
        if row["candidate"] not in {"C0_full", "C1_minus_rp"}
        and float(row["delta_vs_c0_nll"]) < 0
    ]
    if not eligible:
        raise ValueError("no transfer candidate improves C0 on development rows")
    return str(
        min(
            eligible,
            key=lambda row: (float(row["delta_vs_c0_nll"]), str(row["candidate"])),
        )["candidate"]
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_source_root(source_root: Path) -> None:
    """Point the existing frozen Context loaders at a cached input checkout."""
    v1.ROOT = source_root
    v1.OUT = source_root / "data/processed/preseason"
    c12.ROOT = source_root
    c12.PRESEASON = source_root / "data/processed/preseason"
    c12.MODELING = source_root / "data/processed/modeling"
    c12.HISTORY = c12.PRESEASON / "history"
    c12.CONTEXT = c12.PRESEASON / "context"
    c12.TENURES = source_root / "data/raw/cfbd/preseason/coach_tenures"
    h11.OUT = source_root / "data/processed/preseason"


def load_raw_transfer_data(
    transfer_root: Path,
) -> tuple[list[TransferRecord], list[UsageRecord], set[int], set[int]]:
    records: list[TransferRecord] = []
    usage: list[UsageRecord] = []
    portal_seasons, usage_seasons = set(), set()
    for path in sorted((transfer_root / "portal").glob("*.json")):
        if path.name == "manifest.json" or path.name.endswith(".provenance.json"):
            continue
        season = int(path.stem)
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(parse_transfer_payload(payload, season=season))
        portal_seasons.add(season)
    for path in sorted((transfer_root / "usage").glob("*.json")):
        if path.name == "manifest.json" or path.name.endswith(".provenance.json"):
            continue
        season = int(path.stem)
        payload = json.loads(path.read_text(encoding="utf-8"))
        usage.extend(parse_usage_payload(payload, season=season))
        usage_seasons.add(season)
    return records, usage, portal_seasons, usage_seasons


def fbs_feature_rows(rows: Iterable[TeamSeason]) -> list[dict[str, object]]:
    return [
        {
            "season": row.season,
            "subdivision": row.subdivision,
            "team_id": row.team_id,
            "team_name": row.team_name,
            "returning_pct_ppa": row.features.get("returning_pct_ppa"),
        }
        for row in rows
    ]


def attach_transfer_features(
    rows: list[TeamSeason],
    transfer_features: dict[tuple[int, str, str], dict[str, float | None]],
) -> list[TeamSeason]:
    return [
        replace(
            row,
            features={
                **row.features,
                **transfer_features.get((row.season, row.subdivision, row.team_id), {}),
            },
        )
        for row in rows
    ]


def fit_context(
    rows: list[TeamSeason],
    features: Iterable[str],
    *,
    target_season: int,
    trained_through: int,
) -> DirectRankModel:
    """Match the C 1.2 location fit, including its deterministic retry."""
    if trained_through >= target_season:
        raise ValueError("training must end before target")
    feature_list = list(features)
    training = [row for row in rows if row.season <= trained_through]
    try:
        return DirectRankModel.fit(
            training,
            [*H_FEATURES, *feature_list],
            penalty=0.25,
            location_feature_names=[*H_FEATURES, *feature_list],
            scale_feature_names=list(H_FEATURES),
        )
    except RuntimeError as error:
        if "ITERATIONS REACHED LIMIT" not in str(error):
            raise
        return DirectRankModel.fit(
            training,
            [*H_FEATURES, *feature_list],
            penalty=0.25,
            location_feature_names=[*H_FEATURES, *feature_list],
            scale_feature_names=list(H_FEATURES),
            optimizer_options={"maxiter": 2000},
        )


def predictions(
    model: DirectRankModel, rows: list[TeamSeason], name: str
) -> list[v1.PriorPrediction]:
    return h11.make_predictions(model, rows, name)


def replace_fallback_labels(
    fallback: list[v1.PriorPrediction], name: str
) -> list[v1.PriorPrediction]:
    return [replace(item, model=name) for item in fallback]


def merge_modeled_and_fallback(
    modeled: list[v1.PriorPrediction],
    fallback: list[v1.PriorPrediction],
    name: str,
) -> list[v1.PriorPrediction]:
    modeled_keys = {item.key for item in modeled}
    result = [
        *modeled,
        *replace_fallback_labels(
            [item for item in fallback if item.key not in modeled_keys], name
        ),
    ]
    return sorted(result, key=lambda item: item.key)


def score(predictions_: list[v1.PriorPrediction]) -> dict[str, float]:
    raw = v1.score_predictions(predictions_)
    return {metric: float(raw[metric]) for metric in METRICS}


def score_by_season(
    predictions_: list[v1.PriorPrediction],
) -> dict[int, dict[str, float]]:
    return {
        season: score([item for item in predictions_ if item.season == season])
        for season in sorted({item.season for item in predictions_})
    }


def paired_season_bootstrap(
    reference: list[v1.PriorPrediction], candidate: list[v1.PriorPrediction]
) -> dict[str, float | int]:
    ref = h11.prediction_losses(reference)
    alt = h11.prediction_losses(candidate)
    keys = sorted(set(ref) & set(alt))
    by_season: defaultdict[int, list[float]] = defaultdict(list)
    for key in keys:
        by_season[key[0]].append(alt[key][0] - ref[key][0])
    seasons = np.asarray(sorted(by_season), dtype=int)
    if not len(seasons):
        return {
            "n_seasons": 0,
            "n_team_seasons": 0,
            "mean_delta_nll": float("nan"),
            "fraction_better": float("nan"),
        }
    season_means = np.asarray([np.mean(by_season[int(year)]) for year in seasons])
    rng = np.random.default_rng(RANDOM_SEED)
    draws = rng.choice(season_means, size=(2000, len(season_means)), replace=True)
    samples = draws.mean(axis=1)
    return {
        "n_seasons": len(seasons),
        "n_team_seasons": len(keys),
        "mean_delta_nll": float(np.mean(season_means)),
        "central_95_low": float(np.quantile(samples, 0.025)),
        "central_95_high": float(np.quantile(samples, 0.975)),
        "fraction_better": float(np.mean(samples < 0)),
    }


def enrich_for_candidate(
    train: list[TeamSeason],
    target: list[TeamSeason],
    candidate: Candidate,
) -> tuple[list[TeamSeason], list[TeamSeason], list[str]]:
    features = list(candidate.features)
    if candidate.interaction is None:
        return train, target, features
    enriched_train, enriched_target, interaction_name = standardized_interaction_rows(
        train,
        target,
        *candidate.interaction,
        name=candidate.interaction_name,
    )
    return enriched_train, enriched_target, [*features, interaction_name]


def fallback_for_panel(
    rows: list[TeamSeason], cold: list[v1.ColdStartSeason]
) -> list[v1.PriorPrediction]:
    """Load the frozen H PMFs, which provide the production cold-start path."""
    return h11.join_targets(h11.read_predictions(c12.H_MODEL_NAME), rows, cold)


def panel_fit(
    rows: list[TeamSeason],
    fallback: list[v1.PriorPrediction],
    candidate: Candidate,
    *,
    target_seasons: tuple[int, ...],
    trained_through: int,
) -> tuple[list[v1.PriorPrediction], DirectRankModel]:
    target = [row for row in rows if row.season in target_seasons]
    train = [row for row in rows if row.season <= trained_through]
    enriched_train, enriched_target, features = enrich_for_candidate(
        train, target, candidate
    )
    model = fit_context(
        enriched_train,
        features,
        target_season=min(target_seasons),
        trained_through=trained_through,
    )
    modeled = predictions(model, enriched_target, candidate.name)
    return merge_modeled_and_fallback(modeled, fallback, candidate.name), model


def compare_candidate_rows(
    candidate: Candidate,
    candidate_predictions: list[v1.PriorPrediction],
    reference: list[v1.PriorPrediction],
    minus_rp: list[v1.PriorPrediction],
    *,
    protocol: str,
    training_rows: int,
    training_seasons: int,
    transfer_features: dict[tuple[int, str, str], dict[str, float | None]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    keys = {item.key for item in candidate_predictions}
    if keys != {item.key for item in reference} or keys != {
        item.key for item in minus_rp
    }:
        raise ValueError("all Context candidates must preserve identical target keys")
    annual, losses = [], []
    candidate_losses = h11.prediction_losses(candidate_predictions)
    reference_losses = h11.prediction_losses(reference)
    minus_losses = h11.prediction_losses(minus_rp)
    for season in sorted({item.season for item in candidate_predictions}):
        cand_score = score(
            [item for item in candidate_predictions if item.season == season]
        )
        c0_score = score([item for item in reference if item.season == season])
        minus_score = score([item for item in minus_rp if item.season == season])
        row: dict[str, object] = {
            "candidate": candidate.name,
            "protocol": protocol,
            "target_season": season,
            "n_team_seasons": sum(
                item.season == season for item in candidate_predictions
            ),
            "n_training_rows": training_rows,
            "n_training_seasons": training_seasons,
            "same_population_keys": True,
        }
        for metric in METRICS:
            row[f"candidate_{metric}"] = cand_score[metric]
            row[f"c0_{metric}"] = c0_score[metric]
            row[f"minus_rp_{metric}"] = minus_score[metric]
            row[f"delta_vs_c0_{metric}"] = cand_score[metric] - c0_score[metric]
            row[f"delta_vs_minus_rp_{metric}"] = (
                cand_score[metric] - minus_score[metric]
            )
        annual.append(row)
    names = {item.key: item.team_name for item in reference}
    for key in sorted(candidate_losses):
        feature_row = transfer_features.get(key, {})
        losses.append(
            {
                "candidate": candidate.name,
                "protocol": protocol,
                "season": key[0],
                "subdivision": key[1],
                "team_id": key[2],
                "team_name": names[key],
                "candidate_nll": candidate_losses[key][0],
                "c0_nll": reference_losses[key][0],
                "minus_rp_nll": minus_losses[key][0],
                "delta_nll_vs_c0": candidate_losses[key][0] - reference_losses[key][0],
                "delta_nll_vs_minus_rp": candidate_losses[key][0]
                - minus_losses[key][0],
                "candidate_crps": candidate_losses[key][1],
                "c0_crps": reference_losses[key][1],
                "minus_rp_crps": minus_losses[key][1],
                "transfer_in_count": feature_row.get("transfer_in_count"),
                "transfer_in_weighted_rating_sum": feature_row.get(
                    "transfer_in_weighted_rating_sum"
                ),
                "transfer_in_prior_usage_sum": feature_row.get(
                    "transfer_in_prior_usage_sum"
                ),
                "transfer_data_available": feature_row.get("transfer_data_available"),
            }
        )
    return annual, losses


def aggregate_annual(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    groups: defaultdict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["candidate"]), str(row["protocol"]))].append(row)
    for candidate, protocol in sorted(groups):
        part = groups[(candidate, protocol)]
        weights = np.asarray([int(row["n_team_seasons"]) for row in part], dtype=float)
        output: dict[str, object] = {
            "candidate": candidate,
            "protocol": protocol,
            "target_seasons": ";".join(str(row["target_season"]) for row in part),
            "n_target_seasons": len(part),
            "n_team_seasons": int(weights.sum()),
        }
        for metric in METRICS:
            for prefix in (
                "candidate",
                "c0",
                "minus_rp",
                "delta_vs_c0",
                "delta_vs_minus_rp",
            ):
                values = np.asarray([float(row[f"{prefix}_{metric}"]) for row in part])
                output[f"{prefix}_{metric}"] = float(
                    np.average(values, weights=weights)
                )
        output["development_bootstrap"] = None
        result.append(output)
    return result


def permute_transfer_features(
    rows: list[TeamSeason], feature_names: Iterable[str], seed: int = RANDOM_SEED
) -> list[TeamSeason]:
    """Permute transfer values within season, preserving each season's distribution."""
    names = tuple(feature_names)
    rng = np.random.default_rng(seed)
    by_season: defaultdict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_season[row.season].append(index)
    output = [replace(row, features=dict(row.features)) for row in rows]
    for indices in by_season.values():
        for name in names:
            values = [rows[index].features.get(name) for index in indices]
            shuffled = list(rng.permutation(values))
            for index, value in zip(indices, shuffled, strict=True):
                output[index].features[name] = value
    return output


def transfer_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value <= 0:
        return "low_0"
    if value <= 3:
        return "medium_1_3"
    return "high_4_plus"


def diagnostics(
    c0: list[v1.PriorPrediction],
    minus_rp: list[v1.PriorPrediction],
    diagnostic: list[v1.PriorPrediction],
    transfer_features: dict[tuple[int, str, str], dict[str, float | None]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    c0_losses, minus_losses, diagnostic_losses = (
        h11.prediction_losses(c0),
        h11.prediction_losses(minus_rp),
        h11.prediction_losses(diagnostic),
    )
    team_rows = []
    for prediction in c0:
        key = prediction.key
        feature = transfer_features.get(key, {})
        team_rows.append(
            {
                "season": key[0],
                "team_id": key[2],
                "team_name": prediction.team_name,
                "transfer_bucket": transfer_bucket(feature.get("transfer_in_count")),
                "transfer_in_count": feature.get("transfer_in_count"),
                "transfer_in_weighted_rating_sum": feature.get(
                    "transfer_in_weighted_rating_sum"
                ),
                "transfer_in_prior_usage_sum": feature.get(
                    "transfer_in_prior_usage_sum"
                ),
                "transfer_in_count_qb": feature.get("transfer_in_count_qb"),
                "transfer_data_available": feature.get("transfer_data_available"),
                "c0_nll": c0_losses[key][0],
                "minus_rp_nll": minus_losses[key][0],
                "diagnostic_transfer_nll": diagnostic_losses[key][0],
                "rp_contribution_nll": minus_losses[key][0] - c0_losses[key][0],
                "diagnostic_minus_c0_nll": diagnostic_losses[key][0]
                - c0_losses[key][0],
            }
        )
    bucket_rows = []
    for (season, bucket), part in sorted(
        (
            (key, list(group))
            for key, group in __import__("itertools").groupby(
                sorted(
                    team_rows,
                    key=lambda item: (item["season"], item["transfer_bucket"]),
                ),
                key=lambda item: (item["season"], item["transfer_bucket"]),
            )
        )
    ):
        bucket_rows.append(
            {
                "season": season,
                "transfer_bucket": bucket,
                "n_team_seasons": len(part),
                "mean_rp_contribution_nll": float(
                    np.mean([row["rp_contribution_nll"] for row in part])
                ),
                "mean_diagnostic_minus_c0_nll": float(
                    np.mean([row["diagnostic_minus_c0_nll"] for row in part])
                ),
                "mean_c0_nll": float(np.mean([row["c0_nll"] for row in part])),
                "mean_minus_rp_nll": float(
                    np.mean([row["minus_rp_nll"] for row in part])
                ),
            }
        )
    return bucket_rows, team_rows, team_rows


def plot_outputs(
    output: Path,
    annual: list[dict[str, object]],
    coverage: list[dict[str, object]],
    buckets: list[dict[str, object]],
) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    heldout = [row for row in annual if row["protocol"] == "frozen_through_2021"]
    summary = aggregate_annual(heldout)
    if summary:
        figure, axis = plt.subplots(figsize=(12, 5))
        values = [float(row["delta_vs_c0_nll"]) for row in summary]
        labels = [str(row["candidate"]) for row in summary]
        axis.bar(
            labels,
            values,
            color=["#2f855a" if value < 0 else "#c53030" for value in values],
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("held-out ΔNLL versus C0")
        axis.set_title("Transfer-aware Context candidates, frozen through 2021")
        axis.tick_params(axis="x", rotation=50)
        figure.tight_layout()
        figure.savefig(plots / "candidate_delta_nll.png", dpi=160)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(
        [int(row["season"]) for row in coverage],
        [float(row["n_on_or_before_cutoff"]) for row in coverage],
        marker="o",
    )
    axis.set_title("Portal records on or before August 15 cutoff")
    axis.set_xlabel("transfer season")
    axis.set_ylabel("records")
    figure.tight_layout()
    figure.savefig(plots / "portal_coverage_by_season.png", dpi=160)
    plt.close(figure)
    bucket_labels = sorted({str(row["transfer_bucket"]) for row in buckets})
    if bucket_labels:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        values = [
            float(
                np.mean(
                    [
                        row["mean_rp_contribution_nll"]
                        for row in buckets
                        if row["transfer_bucket"] == label
                    ]
                )
            )
            for label in bucket_labels
        ]
        axis.bar(bucket_labels, values)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title("RP contribution by incoming-transfer bucket")
        axis.set_ylabel("mean NLL(C-minus-RP) − NLL(C0)")
        figure.tight_layout()
        figure.savefig(plots / "rp_contribution_by_transfer_bucket.png", dpi=160)
        plt.close(figure)


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_report(
    path: Path,
    summary: dict[str, object],
    annual: list[dict[str, object]],
    candidate_summary: list[dict[str, object]],
    coverage: list[dict[str, object]],
) -> None:
    sanity = summary["c0_sanity_check"]
    comparison = summary["comparison"]
    diagnostic = comparison["diagnostic_candidate"]
    descriptive_best = comparison["heldout_descriptive_best_transfer_candidate"]
    clean_primary = comparison["clean_primary_candidate"]
    heldout = [
        row for row in candidate_summary if row["protocol"] == "frozen_through_2021"
    ]
    clean_primary_delta = next(
        row["delta_vs_c0_nll"] for row in heldout if row["candidate"] == clean_primary
    )
    descriptive_best_delta = next(
        row["delta_vs_c0_nll"]
        for row in heldout
        if row["candidate"] == descriptive_best
    )
    lines = [
        "# Transfer-aware roster continuity oracle (issue 91)",
        "",
        "## Conclusion",
        "",
        f"No development candidate was selected: the rows available to train through 2017 contain zero observed values for every added transfer feature. The predeclared candidates are exploratory retrospective-oracle comparisons. The clean C10 candidate **{clean_primary}** has held-out ΔNLL **{fmt(clean_primary_delta)}** versus production C0; the strongest descriptive result is **{descriptive_best}**, with ΔNLL **{fmt(descriptive_best_delta)}**. {diagnostic} is used only for mechanism diagnostics. This is not a production-safe Context change.",
        "C9 is retained as an explicitly ad hoc hybrid index: returning-production percentage plus incoming prior-usage sum. Those quantities have incompatible denominators, so C9 is not interpreted as reconstructed effective returning production; C10 keeps returning production and incoming prior usage as separate features.",
        f"The C0 control reproduction check {'passed' if sanity['passed'] else 'failed'}: maximum absolute metric difference from the stored production evaluation was {fmt(sanity['max_abs_metric_delta'], 6)}.",
        "",
        "## Provenance and leakage audit",
        "",
        "CFBD `/player/portal` records supply season, origin, destination, position, transfer date, rating, and stars. CFBD `/player/usage` supplies prior-season overall usage. These fields are retained as a retrospective research oracle because endpoint responses are not archived as August 15 snapshots, final destinations may be resolved later, and the portal-to-usage join is name-based. Scholarship-player counts are unavailable because eligibility does not establish scholarship status.",
        "",
        "All cutoff features include only records with a transfer date on or before the cutoff. Missing portal seasons remain missing; a covered season with no matching transfer is zero. No target-season outcomes are used as transfer features.",
        "",
        "## Evaluation protocol",
        "",
        "- C0 is the existing C 1.2 location equation; C1 removes returning production. Transfer candidates were predeclared in the script.",
        "- Fits use rows through 2021 and score unchanged on 2022--2025. The 2018--2021 development comparison is reported for transparency only; it cannot select a transfer candidate because its training rows end before portal coverage begins. The production FBS population and stored H cold-start fallback are retained.",
        "- Training-only imputation and missingness indicators remain in `DirectRankModel`; transfer missingness is never silently converted to zero.",
        "- The negative-control permutation shuffles transfer values within season with seed 7 and never changes the target keys.",
        "",
        "## Held-out candidate comparison",
        "",
        "| Candidate | NLL | Δ vs C0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in heldout:
        lines.append(
            f"| {row['candidate']} | {fmt(row['candidate_nll'])} | {fmt(row['delta_vs_c0_nll'], 4)} | {fmt(row['candidate_crps'])} | {fmt(row['candidate_expected_rank_mae'], 2)} | {fmt(row['candidate_median_rank_mae'], 2)} | {fmt(row['candidate_interval_80_coverage'], 3)} | {fmt(row['candidate_interval_80_average_width'], 2)} |"
        )
    lines += [
        "",
        "## Year-by-year primary metrics",
        "",
        "| Season | Candidate | NLL | Δ vs C0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in annual:
        if row["protocol"] == "frozen_through_2021":
            lines.append(
                f"| {row['target_season']} | {row['candidate']} | {fmt(row['candidate_nll'])} | {fmt(row['delta_vs_c0_nll'], 4)} | {fmt(row['candidate_crps'])} | {fmt(row['candidate_expected_rank_mae'], 2)} | {fmt(row['candidate_median_rank_mae'], 2)} | {fmt(row['candidate_interval_80_coverage'], 3)} | {fmt(row['candidate_interval_80_average_width'], 2)} |"
            )
    lines += [
        "",
        "## Transfer coverage",
        "",
        "| Season | Portal payload | Records | Dated/on cutoff | Destination rate | Rating rate | Usage joins are audited in team-level artifacts |",
        "|---:|:---:|---:|---:|---:|---:|---|",
    ]
    for row in coverage:
        lines.append(
            f"| {row['season']} | {row['portal_payload_available']} | {row['n_portal_records']} | {row['n_on_or_before_cutoff']} | {fmt(row['destination_rate'], 3)} | {fmt(row['rating_rate'], 3)} | yes |"
        )
    lines += [
        "",
        "## Diagnostics and limitations",
        "",
        f"The bucket, replacement-quality, negative-control, cutoff, QB, Team Talent overlap, and returning-production component tables are the auditable diagnostics for the mechanism question. They use the predeclared {diagnostic} candidate as a descriptive reference, not a selected model. A positive RP contribution means C-minus-RP has higher NLL than C0, so RP helped; a negative value means RP hurt.",
        "",
        "Because historical portal coverage begins late and the endpoint is retrospective, these results cannot establish that a feature was knowable on August 15 in the historical years. A positive result identifies a promising oracle representation; productionization requires an archived cutoff-safe source and a stable player/team identity join.",
        "",
        "## Artifacts",
        "",
        "- `provenance_audit.json`, `coverage_by_season.csv` — source classification and season coverage.",
        "- `candidate_annual_metrics.csv`, `candidate_summary.csv`, `candidate_per_team_losses.csv` — paired primary scores and losses.",
        "- `transfer_bucket_diagnostics.csv`, `replacement_quality_diagnostics.csv` — RP mechanism diagnostics.",
        "- `negative_control_metrics.csv`, `cutoff_sensitivity.csv`, `qb_sensitivity.csv`, `talent_overlap_sensitivity.csv`, `rp_component_sensitivity.csv` — predeclared robustness checks.",
        "- `summary.json`, `plots/` — configuration, hashes, comparison record, and visual summaries.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.resolve()
    transfer_root = args.transfer_root.resolve()
    output = args.output.resolve()
    configure_source_root(source_root)
    rows, cold, _ = v1.load_rows(max_season=max(TARGET_SEASONS))
    fbs = [row for row in rows if row.subdivision == "fbs"]
    contextual, _ = c12.attach_context(fbs, c12.feature_index(), c12.cached_tenures())
    records, usage, portal_seasons, usage_seasons = load_raw_transfer_data(
        transfer_root
    )
    team_rows = fbs_feature_rows(contextual)

    def rows_at_cutoff(
        cutoff: date,
    ) -> tuple[list[TeamSeason], dict[tuple[int, str, str], dict[str, float | None]]]:
        transfer_features = aggregate_team_features(
            records,
            usage,
            team_rows,
            covered_seasons=portal_seasons,
            cutoff=cutoff,
        )
        return attach_transfer_features(
            contextual, transfer_features
        ), transfer_features

    contextual_augmented, transfer_features = rows_at_cutoff(
        date(2025, *DEFAULT_CUTOFF)
    )
    panel_fallback = fallback_for_panel(fbs, cold)
    panel_keys = {item.key for item in panel_fallback if item.season in TARGET_SEASONS}
    target_regular_keys = {
        row.key if hasattr(row, "key") else (row.season, row.subdivision, row.team_id)
        for row in contextual_augmented
        if row.season in TARGET_SEASONS
    }
    if not target_regular_keys <= panel_keys:
        raise ValueError("stored H panel is missing regular target keys")
    panel_fallback = [item for item in panel_fallback if item.season in TARGET_SEASONS]

    candidates = candidate_definitions()
    c0 = next(candidate for candidate in candidates if candidate.name == "C0_full")
    c1 = next(candidate for candidate in candidates if candidate.name == "C1_minus_rp")
    c0_predictions, _c0_model = panel_fit(
        contextual_augmented,
        panel_fallback,
        c0,
        target_seasons=TARGET_SEASONS,
        trained_through=FROZEN_TRAIN_THROUGH,
    )
    c1_predictions, _ = panel_fit(
        contextual_augmented,
        panel_fallback,
        c1,
        target_seasons=TARGET_SEASONS,
        trained_through=FROZEN_TRAIN_THROUGH,
    )
    candidate_predictions: dict[str, list[v1.PriorPrediction]] = {
        "C0_full": c0_predictions,
        "C1_minus_rp": c1_predictions,
    }
    development_predictions: dict[str, list[v1.PriorPrediction]] = {}
    dev_fallback = [
        item for item in panel_fallback if item.season in DEVELOPMENT_SEASONS
    ]
    # The stored PMF corpus has only the primary test panel. Development has no
    # known FBS cold start in the common Context population; ordinary rows are
    # still paired exactly and any future cold start is reported as missing.
    dev_augmented = contextual_augmented
    development_observation_counts = development_transfer_observation_counts(
        dev_augmented,
        candidates,
        trained_through=DEVELOPMENT_TRAIN_THROUGH,
    )
    dev_c0, _ = panel_fit(
        dev_augmented,
        dev_fallback,
        c0,
        target_seasons=DEVELOPMENT_SEASONS,
        trained_through=DEVELOPMENT_TRAIN_THROUGH,
    )
    dev_c1, _ = panel_fit(
        dev_augmented,
        dev_fallback,
        c1,
        target_seasons=DEVELOPMENT_SEASONS,
        trained_through=DEVELOPMENT_TRAIN_THROUGH,
    )
    development_predictions["C0_full"] = dev_c0
    development_predictions["C1_minus_rp"] = dev_c1

    annual_rows, loss_rows = [], []
    for candidate in candidates[2:]:
        heldout, _ = panel_fit(
            contextual_augmented,
            panel_fallback,
            candidate,
            target_seasons=TARGET_SEASONS,
            trained_through=FROZEN_TRAIN_THROUGH,
        )
        development, _ = panel_fit(
            dev_augmented,
            dev_fallback,
            candidate,
            target_seasons=DEVELOPMENT_SEASONS,
            trained_through=DEVELOPMENT_TRAIN_THROUGH,
        )
        candidate_predictions[candidate.name] = heldout
        development_predictions[candidate.name] = development
        heldout_annual, heldout_losses = compare_candidate_rows(
            candidate,
            heldout,
            c0_predictions,
            c1_predictions,
            protocol="frozen_through_2021",
            training_rows=sum(
                row.season <= FROZEN_TRAIN_THROUGH for row in contextual_augmented
            ),
            training_seasons=len(
                {
                    row.season
                    for row in contextual_augmented
                    if row.season <= FROZEN_TRAIN_THROUGH
                }
            ),
            transfer_features=transfer_features,
        )
        dev_annual, _ = compare_candidate_rows(
            candidate,
            development,
            dev_c0,
            dev_c1,
            protocol="development_through_2017",
            training_rows=sum(
                row.season <= DEVELOPMENT_TRAIN_THROUGH for row in dev_augmented
            ),
            training_seasons=len(
                {
                    row.season
                    for row in dev_augmented
                    if row.season <= DEVELOPMENT_TRAIN_THROUGH
                }
            ),
            transfer_features=transfer_features,
        )
        annual_rows.extend(heldout_annual)
        annual_rows.extend(dev_annual)
        loss_rows.extend(heldout_losses)
        print(f"completed {candidate.name}", flush=True)

    # Include controls in the tabular output.
    for candidate, prediction_set, reference, minus, protocol in (
        (c0, c0_predictions, c0_predictions, c1_predictions, "frozen_through_2021"),
        (c1, c1_predictions, c0_predictions, c1_predictions, "frozen_through_2021"),
    ):
        rows_for_candidate, losses_for_candidate = compare_candidate_rows(
            candidate,
            prediction_set,
            reference,
            minus,
            protocol=protocol,
            training_rows=sum(
                row.season <= FROZEN_TRAIN_THROUGH for row in contextual_augmented
            ),
            training_seasons=len(
                {
                    row.season
                    for row in contextual_augmented
                    if row.season <= FROZEN_TRAIN_THROUGH
                }
            ),
            transfer_features=transfer_features,
        )
        annual_rows.extend(rows_for_candidate)
        loss_rows.extend(losses_for_candidate)
    candidate_summary = aggregate_annual(annual_rows)

    stored_path = source_root / "data/processed/preseason/context/evaluation.json"
    stored = (
        json.loads(stored_path.read_text(encoding="utf-8"))
        if stored_path.exists()
        else None
    )
    computed_c0 = score(c0_predictions)
    stored_c0 = stored.get("all_fbs", {}).get("candidate", {}) if stored else {}
    sanity_deltas = {
        metric: abs(computed_c0[metric] - float(stored_c0[metric]))
        for metric in METRICS
        if metric in stored_c0
    }
    sanity = {
        "stored_evaluation_path": str(stored_path),
        "stored_metrics": stored_c0,
        "computed_metrics": computed_c0,
        "metric_abs_deltas": sanity_deltas,
        "max_abs_metric_delta": max(sanity_deltas.values(), default=None),
        "tolerance": SANITY_TOLERANCE,
        "passed": bool(sanity_deltas)
        and max(sanity_deltas.values()) <= SANITY_TOLERANCE,
    }
    if not sanity["passed"]:
        raise ValueError(f"C0 does not reproduce stored production metrics: {sanity}")

    development_summary = aggregate_annual(
        [row for row in annual_rows if row["protocol"] == "development_through_2017"]
    )
    diagnostic_name = "C10_transfer_production"
    diagnostic_candidate = next(
        candidate for candidate in candidates if candidate.name == diagnostic_name
    )
    heldout_transfer_summary = [
        row
        for row in candidate_summary
        if row["protocol"] == "frozen_through_2021"
        and row["candidate"] not in {"C0_full", "C1_minus_rp"}
    ]
    heldout_best = min(
        heldout_transfer_summary,
        key=lambda row: (float(row["delta_vs_c0_nll"]), str(row["candidate"])),
    )

    diagnostic_predictions = candidate_predictions[diagnostic_name]

    bucket_rows, replacement_rows, _ = diagnostics(
        c0_predictions, c1_predictions, diagnostic_predictions, transfer_features
    )
    negative_rows = []
    permuted = permute_transfer_features(contextual_augmented, VOLUME_FEATURES)
    permuted_target = [row for row in permuted if row.season in TARGET_SEASONS]
    permuted_train = [row for row in permuted if row.season <= FROZEN_TRAIN_THROUGH]
    perm_candidate = next(
        candidate for candidate in candidates if candidate.name == "C2_transfer_volume"
    )
    enriched_train, enriched_target, perm_features = enrich_for_candidate(
        permuted_train, permuted_target, perm_candidate
    )
    perm_model = fit_context(
        enriched_train,
        perm_features,
        target_season=2022,
        trained_through=FROZEN_TRAIN_THROUGH,
    )
    perm_predictions = merge_modeled_and_fallback(
        predictions(perm_model, enriched_target, "negative_control_permuted"),
        panel_fallback,
        "negative_control_permuted",
    )
    for season in TARGET_SEASONS:
        original_score = score(
            [
                item
                for item in candidate_predictions["C2_transfer_volume"]
                if item.season == season
            ]
        )
        perm_score = score([item for item in perm_predictions if item.season == season])
        c0_score = score([item for item in c0_predictions if item.season == season])
        negative_rows.append(
            {
                "season": season,
                "control": "within_season_transfer_permutation",
                "seed": RANDOM_SEED,
                "original_delta_nll_vs_c0": original_score["nll"] - c0_score["nll"],
                "permuted_delta_nll_vs_c0": perm_score["nll"] - c0_score["nll"],
                "original_delta_crps_vs_c0": original_score["crps"] - c0_score["crps"],
                "permuted_delta_crps_vs_c0": perm_score["crps"] - c0_score["crps"],
            }
        )

    sensitivity_rows = []
    for month, day in ((7, 1), (8, 1), (8, 15)):
        augmented, _ = rows_at_cutoff(date(2025, month, day))
        candidate_panel, _ = panel_fit(
            augmented,
            panel_fallback,
            diagnostic_candidate,
            target_seasons=TARGET_SEASONS,
            trained_through=FROZEN_TRAIN_THROUGH,
        )
        candidate_score = score(candidate_panel)
        c0_score = score(c0_predictions)
        sensitivity_rows.append(
            {
                "candidate": diagnostic_candidate.name,
                "cutoff": f"month={month},day={day}",
                "nll": candidate_score["nll"],
                "delta_nll_vs_c0": candidate_score["nll"] - c0_score["nll"],
                "crps": candidate_score["crps"],
                "delta_crps_vs_c0": candidate_score["crps"] - c0_score["crps"],
                "coverage": candidate_score["interval_80_coverage"],
            }
        )

    qb_rows = []
    qb_features = tuple(
        feature
        for feature in diagnostic_candidate.features
        if feature not in QB_FEATURES
    )
    qb_candidate = Candidate(
        "diagnostic_without_qb_features",
        qb_features,
        diagnostic_candidate.interaction,
        diagnostic_candidate.interaction_name,
        True,
    )
    qb_panel, _ = panel_fit(
        contextual_augmented,
        panel_fallback,
        qb_candidate,
        target_seasons=TARGET_SEASONS,
        trained_through=FROZEN_TRAIN_THROUGH,
    )
    qb_rows.append(
        {
            "candidate": diagnostic_candidate.name,
            "variant": "with_qb_features",
            "nll": score(diagnostic_predictions)["nll"],
            "delta_nll_vs_c0": score(diagnostic_predictions)["nll"]
            - score(c0_predictions)["nll"],
        }
    )
    qb_rows.append(
        {
            "candidate": diagnostic_candidate.name,
            "variant": "without_qb_features",
            "nll": score(qb_panel)["nll"],
            "delta_nll_vs_c0": score(qb_panel)["nll"] - score(c0_predictions)["nll"],
        }
    )

    overlap_rows = []
    without_talent = tuple(
        feature for feature in BASE_CONTEXT_FEATURES if feature != "talent_composite"
    )
    overlap_candidate = Candidate(
        "diagnostic_without_team_talent",
        (
            *without_talent,
            *[
                feature
                for feature in diagnostic_candidate.features
                if feature not in BASE_CONTEXT_FEATURES
            ],
        ),
        diagnostic_candidate.interaction,
        diagnostic_candidate.interaction_name,
        True,
    )
    overlap_panel, _ = panel_fit(
        contextual_augmented,
        panel_fallback,
        overlap_candidate,
        target_seasons=TARGET_SEASONS,
        trained_through=FROZEN_TRAIN_THROUGH,
    )
    overlap_rows.extend(
        [
            {
                "variant": "with_team_talent",
                "nll": score(diagnostic_predictions)["nll"],
                "delta_nll_vs_c0": score(diagnostic_predictions)["nll"]
                - score(c0_predictions)["nll"],
            },
            {
                "variant": "without_team_talent",
                "nll": score(overlap_panel)["nll"],
                "delta_nll_vs_c0": score(overlap_panel)["nll"]
                - score(c0_predictions)["nll"],
            },
        ]
    )

    component_rows = []
    for label, component in (
        ("total", "returning_pct_ppa"),
        ("passing", "returning_pct_passing_ppa"),
        ("receiving", "returning_pct_receiving_ppa"),
        ("rushing", "returning_pct_rushing_ppa"),
    ):
        component_candidate = Candidate(
            f"rp_{label}_plus_transfer_volume",
            (*C_MINUS_RP_FEATURES, component, *VOLUME_FEATURES),
            transfer_aware=True,
        )
        component_panel, _ = panel_fit(
            contextual_augmented,
            panel_fallback,
            component_candidate,
            target_seasons=TARGET_SEASONS,
            trained_through=FROZEN_TRAIN_THROUGH,
        )
        component_score = score(component_panel)
        minus_score = score(c1_predictions)
        component_rows.append(
            {
                "rp_component": label,
                "candidate": component_candidate.name,
                "nll": component_score["nll"],
                "delta_nll_vs_c_minus_rp": component_score["nll"] - minus_score["nll"],
                "crps": component_score["crps"],
                "delta_crps_vs_c_minus_rp": component_score["crps"]
                - minus_score["crps"],
            }
        )

    coverage = coverage_rows(
        records,
        target_seasons=sorted({row.season for row in contextual}),
        cutoff=date(2025, *DEFAULT_CUTOFF),
        team_rows=team_rows,
        covered_seasons=portal_seasons,
    )
    write_json(output / "provenance_audit.json", portal_provenance())
    write_csv(output / "coverage_by_season.csv", coverage)
    write_csv(output / "candidate_annual_metrics.csv", annual_rows)
    write_csv(output / "candidate_summary.csv", candidate_summary)
    write_csv(output / "candidate_per_team_losses.csv", loss_rows)
    write_csv(output / "transfer_bucket_diagnostics.csv", bucket_rows)
    write_csv(output / "replacement_quality_diagnostics.csv", replacement_rows)
    write_csv(output / "negative_control_metrics.csv", negative_rows)
    write_csv(output / "cutoff_sensitivity.csv", sensitivity_rows)
    write_csv(output / "qb_sensitivity.csv", qb_rows)
    write_csv(output / "talent_overlap_sensitivity.csv", overlap_rows)
    write_csv(output / "rp_component_sensitivity.csv", component_rows)
    summary = {
        "study": "issue_91_transfer_aware_roster_continuity_oracle",
        "production_models_modified": False,
        "target_seasons": list(TARGET_SEASONS),
        "development_seasons": list(DEVELOPMENT_SEASONS),
        "frozen_train_through": FROZEN_TRAIN_THROUGH,
        "development_train_through": DEVELOPMENT_TRAIN_THROUGH,
        "cutoff": f"season-relative {DEFAULT_CUTOFF[0]:02d}-{DEFAULT_CUTOFF[1]:02d}",
        "portal_seasons_available": sorted(portal_seasons),
        "usage_seasons_available": sorted(usage_seasons),
        "candidate_definitions": [
            {
                "name": candidate.name,
                "features": list(candidate.features),
                "interaction": candidate.interaction,
                "transfer_aware": candidate.transfer_aware,
            }
            for candidate in candidates
        ],
        "c0_sanity_check": sanity,
        "comparison": {
            "development_selection_performed": False,
            "development_selection_reason": "transfer portal coverage begins after the development training period; selection is refused when all added transfer features are unobserved",
            "development_summary": development_summary,
            "development_transfer_observation_counts": development_observation_counts,
            "diagnostic_candidate": diagnostic_name,
            "clean_primary_candidate": "C10_transfer_production",
            "heldout_descriptive_best_transfer_candidate": heldout_best["candidate"],
            "heldout_best_delta_nll_vs_c0": heldout_best["delta_vs_c0_nll"],
        },
        "heldout_summary": [
            row for row in candidate_summary if row["protocol"] == "frozen_through_2021"
        ],
        "negative_control": {
            "method": "permute transfer volume features within season",
            "seed": RANDOM_SEED,
        },
        "missing_data_policy": "uncovered portal seasons are None; covered seasons with no matching transfer are zero; DirectRankModel applies training-only median imputation and missingness indicators",
        "provenance": portal_provenance(),
        "source_hashes": {
            str(path.relative_to(source_root)): sha256_file(path)
            for path in (
                source_root
                / "data/processed/modeling/team_season_rank_distributions.csv",
                source_root / "data/processed/preseason/team_season_features.csv",
            )
            if path.exists()
        },
    }
    write_json(output / "summary.json", summary)
    plot_outputs(output, annual_rows, coverage, bucket_rows)
    render_report(
        output / "report.md", summary, annual_rows, candidate_summary, coverage
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument(
        "--transfer-root", type=Path, default=ROOT / "data/raw/cfbd/preseason/transfers"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/transfer_roster_continuity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "diagnostic_candidate": summary["comparison"]["diagnostic_candidate"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
