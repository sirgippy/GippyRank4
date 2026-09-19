"""Test frozen defensive-transfer impact by position group (issue 111).

This research-only runner is the final defensive-transfer decomposition.  It
reuses the frozen PR #105 player impact values and the existing Context C 1.2
fit, H fallback, August 15 transfer construction, and scoring helpers.  The
only new feature construction is the deterministic sum of those values by the
three already-frozen groups: ``dl_edge``, ``lb``, and ``db``.

No production model or raw source is changed.  The runner writes a report and
machine-readable artifacts under the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import date
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_preseason_context_prior_v1_2 as c12
import build_preseason_prior_v1_1 as v1_1
import investigate_transfer_roster_continuity as prior

from gippyrank.preseason import DirectRankModel, TeamSeason

ROOT = Path(__file__).resolve().parents[1]
TARGET_SEASONS = (2022, 2023, 2024, 2025)
FROZEN_TRAIN_THROUGH = 2021
DEFAULT_CUTOFF = date(2025, 8, 15)
RANDOM_SEED = 7
D5_PARITY_TOLERANCE = 1e-6
RECONSTRUCTION_TOLERANCE = 1e-6

POSITION_GROUPS = ("dl_edge", "lb", "db")
POSITION_GROUP_LABELS = {
    "dl_edge": "DL / EDGE",
    "lb": "LB",
    "db": "DB",
}
POSITION_IMPACT_FEATURES = {
    group: f"transfer_in_prior_defensive_impact_{group}_sum"
    for group in POSITION_GROUPS
}
POSITION_AVAILABILITY_FEATURES = {
    group: f"transfer_in_prior_defensive_impact_{group}_available"
    for group in POSITION_GROUPS
}
DL_EDGE_IMPACT = POSITION_IMPACT_FEATURES["dl_edge"]
LB_IMPACT = POSITION_IMPACT_FEATURES["lb"]
DB_IMPACT = POSITION_IMPACT_FEATURES["db"]
DL_EDGE_AVAILABLE = POSITION_AVAILABILITY_FEATURES["dl_edge"]
LB_AVAILABLE = POSITION_AVAILABILITY_FEATURES["lb"]
DB_AVAILABLE = POSITION_AVAILABILITY_FEATURES["db"]

RETURNING_PPA = "returning_pct_ppa"
INCOMING_USAGE = "transfer_in_prior_usage_sum"
METRICS = tuple(prior.METRICS)
USABLE_IMPACT_STATUSES = frozenset(
    {"resolved", "zero_recorded_defensive_box_score_games"}
)


@dataclass(frozen=True)
class Candidate:
    """A predeclared primary candidate or an explicitly permitted control."""

    label: str
    name: str
    description: str
    features: tuple[str, ...]
    position_groups: tuple[str, ...] = ()
    kind: str = "primary"
    interaction: tuple[str, str] | None = None
    interaction_name: str | None = None


@dataclass(frozen=True)
class PositionFeatureBundle:
    """Position-specific values plus construction and coverage diagnostics."""

    values: dict[tuple[int, str, str], dict[str, float]]
    statuses: dict[tuple[int, str, str], dict[str, str]]
    coverage_rows: list[dict[str, object]]
    reconstruction_rows: list[dict[str, object]]


def d5_features() -> tuple[str, ...]:
    """Return the frozen D5 representation from the transfer decomposition."""
    return (*prior.C_MINUS_RP_FEATURES, RETURNING_PPA, INCOMING_USAGE)


def _position_features(groups: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(groups)
    return (
        *d5_features(),
        *(POSITION_IMPACT_FEATURES[group] for group in selected),
        *(POSITION_AVAILABILITY_FEATURES[group] for group in selected),
    )


def candidate_definitions() -> tuple[Candidate, ...]:
    """Return exactly the predeclared P0--P4 candidates."""
    d5 = d5_features()
    return (
        Candidate(
            "P0",
            "P0_D5",
            "D5: C-minus-RP plus returning production and incoming prior usage.",
            d5,
        ),
        Candidate(
            "P1",
            "P1_D5_plus_dl_edge",
            "D5 plus imported prior defensive impact from DL / EDGE and its availability indicator.",
            _position_features(("dl_edge",)),
            ("dl_edge",),
        ),
        Candidate(
            "P2",
            "P2_D5_plus_lb",
            "D5 plus imported prior defensive impact from LB and its availability indicator.",
            _position_features(("lb",)),
            ("lb",),
        ),
        Candidate(
            "P3",
            "P3_D5_plus_db",
            "D5 plus imported prior defensive impact from DB and its availability indicator.",
            _position_features(("db",)),
            ("db",),
        ),
        Candidate(
            "P4",
            "P4_D5_plus_all_position_groups",
            "D5 plus all three position-specific defensive impact values and availability indicators.",
            _position_features(POSITION_GROUPS),
            POSITION_GROUPS,
        ),
    )


def missingness_control(candidate: Candidate) -> Candidate:
    """Return the availability-only control for a successful candidate."""
    groups = candidate.position_groups
    return Candidate(
        f"{candidate.label}-M",
        f"{candidate.name}_availability_only",
        f"{candidate.label} with numeric defensive impact removed; retain availability for {', '.join(groups)}.",
        (*d5_features(), *(POSITION_AVAILABILITY_FEATURES[group] for group in groups)),
        groups,
        "missingness_control",
    )


def permutation_control(candidate: Candidate) -> Candidate:
    """Return the within-season permutation control for a candidate."""
    return Candidate(
        f"{candidate.label}-P",
        f"{candidate.name}_within_season_permutation",
        f"{candidate.label} with observed position-group impact reassigned within season.",
        candidate.features,
        candidate.position_groups,
        "permutation_control",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty artifact: {path}")
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


def transfer_source_hashes(transfer_root: Path) -> dict[str, str]:
    """Hash immutable portal and usage payloads without storing machine paths."""
    result: dict[str, str] = {}
    for directory in ("portal", "usage"):
        for path in sorted((transfer_root / directory).glob("*.json")):
            if path.name == "manifest.json" or path.name.endswith(".provenance.json"):
                continue
            result[str(path.relative_to(transfer_root))] = sha256_file(path)
    return result


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _is_true(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _coverage_lookup(path: Path) -> dict[tuple[int, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (int(row["season"]), row["position_group"]): row
            for row in csv.DictReader(handle)
        }


def load_position_features(
    transfer_audit_path: Path,
    coverage_path: Path,
) -> PositionFeatureBundle:
    """Construct group sums from the frozen transfer-player audit.

    Group membership is the existing portal position group.  Every in-scope
    defensive portal row is therefore counted in exactly one frozen group,
    including rows whose prior identity cannot be resolved.  A group is
    observed when it has no incoming rows or every incoming row has a usable
    frozen impact.  An unresolved group receives neutral zero plus an explicit
    availability value of zero.
    """
    incoming: defaultdict[tuple[int, str], dict[str, list[dict[str, str]]]] = (
        defaultdict(lambda: {group: [] for group in POSITION_GROUPS})
    )
    with transfer_audit_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not _is_true(row["in_model_relevant_population"]):
                continue
            if not _is_true(row["defensive_candidate"]):
                continue
            group = row.get("portal_position_group", "")
            if group not in POSITION_GROUPS:
                raise ValueError(f"unknown frozen defensive position group: {group!r}")
            destination = row.get("destination_team_id", "")
            if not destination:
                raise ValueError("in-scope defensive transfer has no destination team")
            incoming[(int(row["season"]), destination)][group].append(row)

    coverage = _coverage_lookup(coverage_path)
    values: dict[tuple[int, str, str], dict[str, float]] = {}
    statuses: dict[tuple[int, str, str], dict[str, str]] = {}
    coverage_rows: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    team_rows: list[dict[str, str]] = []
    with transfer_audit_path.parent.joinpath("team_season_features.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        team_rows = list(csv.DictReader(handle))

    by_season: defaultdict[int, list[tuple[int, str, str]]] = defaultdict(list)
    for team in team_rows:
        season = int(team["season"])
        subdivision = team["subdivision"]
        team_id = team["team_id"]
        key = (season, subdivision, team_id)
        group_rows = incoming[(season, team_id)]
        group_values: dict[str, float] = {}
        group_statuses: dict[str, str] = {}
        for group in POSITION_GROUPS:
            rows = group_rows[group]
            usable = [
                row
                for row in rows
                if row.get("impact_status") in USABLE_IMPACT_STATUSES
                and row.get("prior_defensive_impact") not in (None, "")
            ]
            if not rows:
                status = "no_incoming_transfer_group"
                value = 0.0
            elif len(usable) == len(rows):
                status = "complete"
                value = sum(
                    float(row["prior_defensive_impact"] or 0.0) for row in usable
                )
            else:
                status = "unresolved"
                value = 0.0
            group_values[group] = value
            group_statuses[group] = status
        values[key] = {
            POSITION_IMPACT_FEATURES[group]: group_values[group]
            for group in POSITION_GROUPS
        }
        values[key].update(
            {
                POSITION_AVAILABILITY_FEATURES[group]: float(
                    group_statuses[group] != "unresolved"
                )
                for group in POSITION_GROUPS
            }
        )
        statuses[key] = group_statuses
        by_season[season].append(key)

        all_observed = all(
            group_statuses[group] != "unresolved" for group in POSITION_GROUPS
        )
        aggregate_value = _float_or_none(
            team.get("transfer_in_prior_defensive_impact_sum")
        )
        if all_observed and aggregate_value is None:
            raise ValueError(
                "all position groups are observed but the frozen aggregate is missing: "
                f"{key}"
            )
        position_sum = sum(group_values.values()) if all_observed else None
        reconstruction_delta = (
            position_sum - aggregate_value
            if position_sum is not None and aggregate_value is not None
            else None
        )
        reconstruction_rows.append(
            {
                "season": season,
                "subdivision": subdivision,
                "team_id": team_id,
                "team_name": team["team_name"],
                "existing_aggregate_status": team["feature_coverage_status"],
                "all_position_groups_observed": all_observed,
                **{
                    f"{group}_status": group_statuses[group]
                    for group in POSITION_GROUPS
                },
                "existing_aggregate_impact": aggregate_value,
                "position_group_impact_sum": position_sum,
                "delta_position_sum_minus_aggregate": reconstruction_delta,
                "parity_passed": (
                    None
                    if reconstruction_delta is None
                    else abs(reconstruction_delta) <= RECONSTRUCTION_TOLERANCE
                ),
                "explanation": (
                    "compared: all three groups observed"
                    if all_observed
                    else "not compared: at least one group unresolved"
                ),
            }
        )

    for season in sorted(by_season):
        for group in POSITION_GROUPS:
            keys = by_season[season]
            group_rows = [(key, incoming[(key[0], key[2])][group]) for key in keys]
            incoming_count = sum(len(rows) for _, rows in group_rows)
            resolved_count = sum(
                sum(
                    row.get("impact_status") in USABLE_IMPACT_STATUSES
                    and row.get("prior_defensive_impact") not in (None, "")
                    for row in rows
                )
                for _, rows in group_rows
            )
            no_incoming = sum(not rows for _, rows in group_rows)
            complete = sum(
                bool(rows)
                and all(
                    row.get("impact_status") in USABLE_IMPACT_STATUSES
                    and row.get("prior_defensive_impact") not in (None, "")
                    for row in rows
                )
                for _, rows in group_rows
            )
            unresolved = len(keys) - no_incoming - complete
            observed = no_incoming + complete
            audit_coverage = coverage.get((season, group), {})
            coverage_rows.append(
                {
                    "season": season,
                    "position_group": group,
                    "position_group_label": POSITION_GROUP_LABELS[group],
                    "incoming_transfers": incoming_count,
                    "resolved_player_impacts": resolved_count,
                    "unresolved_player_impacts": incoming_count - resolved_count,
                    "resolved_impact_fraction": (
                        resolved_count / incoming_count if incoming_count else None
                    ),
                    "experience_mass_coverage_proxy": _float_or_none(
                        audit_coverage.get("experience_mass_coverage_proxy")
                    ),
                    "resolved_experience_mass": _float_or_none(
                        audit_coverage.get("resolved_experience_mass")
                    ),
                    "recoverable_experience_mass_proxy": _float_or_none(
                        audit_coverage.get("recoverable_experience_mass_proxy")
                    ),
                    "impact_mass_coverage_proxy": None,
                    "impact_mass_coverage_proxy_available": False,
                    "team_seasons": len(keys),
                    "team_seasons_complete": complete,
                    "team_seasons_no_incoming": no_incoming,
                    "team_seasons_unresolved": unresolved,
                    "team_seasons_observed": observed,
                    "fraction_team_seasons_observed": observed / len(keys),
                }
            )

    return PositionFeatureBundle(values, statuses, coverage_rows, reconstruction_rows)


def attach_position_features(
    rows: Iterable[TeamSeason],
    values: Mapping[tuple[int, str, str], Mapping[str, float]],
) -> list[TeamSeason]:
    """Attach group features while preserving the complete Context population."""
    neutral = {
        feature: 0.0
        for group in POSITION_GROUPS
        for feature in (
            POSITION_IMPACT_FEATURES[group],
            POSITION_AVAILABILITY_FEATURES[group],
        )
    }
    result = []
    for row in rows:
        additions = values.get((row.season, row.subdivision, row.team_id), neutral)
        result.append(replace(row, features={**row.features, **additions}))
    return result


def permute_position_features(
    rows: Iterable[TeamSeason],
    position_groups: Iterable[str],
    *,
    seed: int = RANDOM_SEED,
) -> list[TeamSeason]:
    """Permute observed numeric group values within season only.

    Availability indicators and all unresolved neutral values stay attached to
    their original team-seasons.  Each selected group's observed values are
    shuffled independently, preserving its season-level value distribution and
    missingness pattern.
    """
    selected = tuple(position_groups)
    if any(group not in POSITION_GROUPS for group in selected):
        raise ValueError(f"unknown position group in permutation: {selected}")
    result = [replace(row, features=dict(row.features)) for row in rows]
    by_season: defaultdict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(result):
        by_season[row.season].append(index)
    rng = np.random.default_rng(seed)
    for indices in by_season.values():
        for group in selected:
            impact = POSITION_IMPACT_FEATURES[group]
            available = POSITION_AVAILABILITY_FEATURES[group]
            observed = [
                index
                for index in indices
                if result[index].features.get(available) == 1.0
            ]
            if len(observed) < 2:
                continue
            shuffled = rng.permutation(observed)
            values = [result[int(index)].features[impact] for index in shuffled]
            for index, value in zip(observed, values, strict=True):
                result[index].features[impact] = value
    return result


def configure_source_root(source_root: Path) -> None:
    """Point the existing Context loaders at the requested input root."""
    prior.configure_source_root(source_root)


def fit_frozen(
    rows: list[TeamSeason],
    fallback: list[v1_1.PriorPrediction],
    candidate: Candidate,
) -> tuple[list[v1_1.PriorPrediction], DirectRankModel]:
    """Fit through 2021 and score the unchanged 2022--2025 panel."""
    return prior.panel_fit(
        rows,
        fallback,
        candidate,  # type: ignore[arg-type]
        target_seasons=TARGET_SEASONS,
        trained_through=FROZEN_TRAIN_THROUGH,
    )


def fit_rolling(
    rows: list[TeamSeason],
    fallback: list[v1_1.PriorPrediction],
    candidate: Candidate,
    target_season: int,
) -> tuple[list[v1_1.PriorPrediction], DirectRankModel]:
    """Fit strictly before one target season and score that season."""
    training = [row for row in rows if row.season < target_season]
    target = [row for row in rows if row.season == target_season]
    model = prior.fit_context(
        training,
        candidate.features,
        target_season=target_season,
        trained_through=target_season - 1,
    )
    modeled = prior.predictions(model, target, candidate.name)
    target_fallback = [item for item in fallback if item.season == target_season]
    return prior.merge_modeled_and_fallback(
        modeled, target_fallback, candidate.name
    ), model


def _metric_row(
    candidate: Candidate,
    predictions: list[v1_1.PriorPrediction],
    *,
    protocol: str,
    target_season: int | str,
    train_through: int,
    reference: Mapping[str, float],
) -> dict[str, object]:
    values = prior.score(predictions)
    row: dict[str, object] = {
        "protocol": protocol,
        "target_season": target_season,
        "train_through": train_through,
        "candidate": candidate.label,
        "candidate_name": candidate.name,
        "kind": candidate.kind,
        "n_team_seasons": len(predictions),
    }
    for metric in METRICS:
        row[metric] = values[metric]
        row[f"delta_{metric}_vs_P0"] = values[metric] - reference[metric]
    return row


def metric_summary(
    rows: list[dict[str, object]], *, protocol: str, candidate: str
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if row["protocol"] == protocol and row["candidate"] == candidate
    ]
    if not selected:
        raise ValueError(f"no metric rows for {protocol}/{candidate}")
    weights = np.asarray([int(row["n_team_seasons"]) for row in selected])
    result: dict[str, object] = {
        "protocol": protocol,
        "candidate": candidate,
        "candidate_name": selected[0]["candidate_name"],
        "n_target_seasons": len(selected),
        "target_seasons": ";".join(str(row["target_season"]) for row in selected),
        "n_team_seasons": int(weights.sum()),
    }
    for metric in METRICS:
        result[metric] = float(
            np.average([float(row[metric]) for row in selected], weights=weights)
        )
        result[f"delta_{metric}_vs_P0"] = float(
            np.average(
                [float(row[f"delta_{metric}_vs_P0"]) for row in selected],
                weights=weights,
            )
        )
    return result


def paired_diagnostics(
    comparison: str,
    candidate_name: str,
    candidate: list[v1_1.PriorPrediction],
    reference_name: str,
    reference: list[v1_1.PriorPrediction],
    *,
    protocol: str = "frozen_through_2021",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return paired team-season losses and aggregate/year summaries."""
    candidate_losses = v1_1.prediction_losses(candidate)
    reference_losses = v1_1.prediction_losses(reference)
    keys = sorted(set(candidate_losses) & set(reference_losses))
    if not keys:
        raise ValueError(f"no paired losses for {comparison}")
    names = {item.key: item.team_name for item in reference}
    details = [
        {
            "comparison": comparison,
            "protocol": protocol,
            "season": key[0],
            "subdivision": key[1],
            "team_id": key[2],
            "team_name": names[key],
            "candidate": candidate_name,
            "reference": reference_name,
            "candidate_nll": candidate_losses[key][0],
            "reference_nll": reference_losses[key][0],
            "delta_nll": candidate_losses[key][0] - reference_losses[key][0],
        }
        for key in keys
    ]

    def summarize(scope: str, values: list[float]) -> dict[str, object]:
        numbers = np.asarray(values, dtype=float)
        return {
            "comparison": comparison,
            "protocol": protocol,
            "candidate": candidate_name,
            "reference": reference_name,
            "scope": scope,
            "n_team_seasons": len(numbers),
            "mean_delta_nll": float(np.mean(numbers)),
            "median_delta_nll": float(np.median(numbers)),
            "fraction_team_seasons_improved": float(np.mean(numbers < 0)),
        }

    summaries = [summarize("aggregate", [float(row["delta_nll"]) for row in details])]
    for season in TARGET_SEASONS:
        values = [
            float(row["delta_nll"]) for row in details if int(row["season"]) == season
        ]
        if values:
            summaries.append(summarize(str(season), values))
    return details, summaries


def _coefficient_indexes(model: DirectRankModel, feature: str) -> tuple[int, int, int]:
    if feature not in model.feature_names:
        raise ValueError(f"feature is not in model: {feature}")
    feature_index = model.feature_names.index(feature)
    numeric = model.lag_count + 1 + feature_index
    missingness = model.lag_count + 1 + len(model.feature_names) + feature_index
    return feature_index, numeric, missingness


def _sign(value: float, tolerance: float = 1e-10) -> str:
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "zero"


def _source_available(row: TeamSeason, feature: str) -> bool:
    if (
        feature in POSITION_IMPACT_FEATURES.values()
        or feature in POSITION_AVAILABILITY_FEATURES.values()
    ):
        group = (
            next(
                group
                for group, name in POSITION_IMPACT_FEATURES.items()
                if name == feature
            )
            if feature in POSITION_IMPACT_FEATURES.values()
            else next(
                group
                for group, name in POSITION_AVAILABILITY_FEATURES.items()
                if name == feature
            )
        )
        return row.features.get(POSITION_AVAILABILITY_FEATURES[group]) == 1.0
    return row.features.get(feature) is not None


def coefficient_rows(
    candidates: Iterable[Candidate],
    models: Mapping[str, DirectRankModel],
    training_by_candidate: Mapping[str, list[TeamSeason]],
    *,
    protocol: str,
    target_season: int | str,
    train_through: int,
    previous: dict[tuple[str, str], float] | None = None,
) -> list[dict[str, object]]:
    """Extract required standardized location paths and source coverage."""
    previous = previous if previous is not None else {}
    result: list[dict[str, object]] = []
    for candidate in candidates:
        if not candidate.position_groups:
            continue
        model = models[candidate.name]
        requested = [RETURNING_PPA, INCOMING_USAGE]
        for group in candidate.position_groups:
            requested.extend(
                (POSITION_IMPACT_FEATURES[group], POSITION_AVAILABILITY_FEATURES[group])
            )
        training = training_by_candidate[candidate.name]
        for feature in requested:
            feature_index, numeric_index, missingness_index = _coefficient_indexes(
                model, feature
            )
            value = float(model.beta[numeric_index])
            key = (candidate.label, feature)
            old = previous.get(key)
            observed = sum(_source_available(row, feature) for row in training)
            result.append(
                {
                    "protocol": protocol,
                    "target_season": target_season,
                    "train_through": train_through,
                    "candidate": candidate.label,
                    "candidate_name": candidate.name,
                    "feature": feature,
                    "feature_role": (
                        "defensive_value"
                        if feature in POSITION_IMPACT_FEATURES.values()
                        else "availability"
                        if feature in POSITION_AVAILABILITY_FEATURES.values()
                        else "baseline"
                    ),
                    "standardized_location_coefficient": value,
                    "sign": _sign(value),
                    "magnitude": abs(value),
                    "change_from_previous_fit": value - old
                    if old is not None
                    else None,
                    "location_missingness_coefficient": float(
                        model.beta[missingness_index]
                    ),
                    "standardized_scale_coefficient": float(
                        model.gamma[1 + feature_index]
                    ),
                    "training_source_available_n": observed,
                    "training_source_unresolved_n": len(training) - observed,
                    "training_source_available_fraction": (
                        observed / len(training) if training else None
                    ),
                }
            )
            previous[key] = value
    return result


def training_coverage(
    rows: list[TeamSeason],
    statuses: Mapping[tuple[int, str, str], Mapping[str, str]],
) -> list[dict[str, object]]:
    """Report effective observed group rows for every rolling fit."""
    result: list[dict[str, object]] = []
    for target_season in TARGET_SEASONS:
        training = [row for row in rows if row.season < target_season]
        for group in POSITION_GROUPS:
            available_feature = POSITION_AVAILABILITY_FEATURES[group]
            observed = [
                row for row in training if row.features.get(available_feature) == 1.0
            ]
            complete = 0
            no_incoming = 0
            unresolved = 0
            unavailable = 0
            for row in training:
                state = statuses.get(
                    (row.season, row.subdivision, row.team_id), {}
                ).get(group, "source_unavailable")
                if state == "complete":
                    complete += 1
                elif state == "no_incoming_transfer_group":
                    no_incoming += 1
                elif state == "unresolved":
                    unresolved += 1
                else:
                    unavailable += 1
            result.append(
                {
                    "target_season": target_season,
                    "train_through": target_season - 1,
                    "position_group": group,
                    "position_group_label": POSITION_GROUP_LABELS[group],
                    "n_training_team_seasons": len(training),
                    "n_observed_training_team_seasons": len(observed),
                    "n_complete_training_team_seasons": complete,
                    "n_no_incoming_training_team_seasons": no_incoming,
                    "n_unresolved_training_team_seasons": unresolved,
                    "n_source_unavailable_training_team_seasons": unavailable,
                    "fraction_observed_training_team_seasons": (
                        len(observed) / len(training) if training else None
                    ),
                }
            )
    return result


def _stability_summary(
    coefficient_rows_: list[dict[str, object]],
    candidate: Candidate,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for group in candidate.position_groups:
        value_feature = POSITION_IMPACT_FEATURES[group]
        availability_feature = POSITION_AVAILABILITY_FEATURES[group]
        value_rows = [
            row
            for row in coefficient_rows_
            if row["candidate"] == candidate.label
            and row["protocol"] == "rolling_origin"
            and row["feature"] == value_feature
        ]
        availability_rows = [
            row
            for row in coefficient_rows_
            if row["candidate"] == candidate.label
            and row["protocol"] == "rolling_origin"
            and row["feature"] == availability_feature
        ]
        values = [float(row["standardized_location_coefficient"]) for row in value_rows]
        availability = [
            float(row["standardized_location_coefficient"]) for row in availability_rows
        ]
        signs = [_sign(value) for value in values]
        changes = [
            abs(float(row["change_from_previous_fit"]))
            for row in value_rows
            if row["change_from_previous_fit"] is not None
        ]
        result[group] = {
            "value_feature": value_feature,
            "availability_feature": availability_feature,
            "value_coefficients": values,
            "availability_coefficients": availability,
            "value_signs": signs,
            "sign_flip_count": sum(
                left != right
                for left, right in pairwise(signs)
                if left != "zero" and right != "zero"
            ),
            "max_absolute_change": max(changes, default=None),
            "first_absolute_value": abs(values[0]) if values else None,
            "last_absolute_value": abs(values[-1]) if values else None,
            "magnitude_ratio_last_to_first": (
                abs(values[-1]) / abs(values[0])
                if values and abs(values[0]) > 1e-12
                else None
            ),
            "availability_dominates_n": sum(
                abs(available) > abs(value)
                for value, available in zip(values, availability, strict=True)
            ),
        }
    return result


def _summary_metric(
    rows: list[dict[str, object]], protocol: str, candidate: str, metric: str
) -> float:
    row = next(
        row
        for row in rows
        if row["protocol"] == protocol and row["candidate"] == candidate
    )
    return float(row[metric])


def _paired_aggregate(
    paired_summaries: list[dict[str, object]],
    comparison: str,
    *,
    protocol: str = "frozen_through_2021",
) -> dict[str, object] | None:
    return next(
        (
            row
            for row in paired_summaries
            if row["comparison"] == comparison
            and row["protocol"] == protocol
            and row["scope"] == "aggregate"
        ),
        None,
    )


def build_group_screens(
    primary: tuple[Candidate, ...],
    frozen_rows: list[dict[str, object]],
    rolling_rows: list[dict[str, object]],
    paired_summaries: list[dict[str, object]],
    coefficients: list[dict[str, object]],
    controls: Mapping[str, Candidate],
    permutations: Mapping[str, Candidate],
) -> dict[str, dict[str, object]]:
    """Apply the predeclared frozen/control/rolling stability screen."""
    screens: dict[str, dict[str, object]] = {}
    p0_nll = _summary_metric(frozen_rows, "frozen_through_2021", "P0", "nll")
    for candidate in primary[1:4]:
        control = controls.get(candidate.label)
        permutation = permutations.get(candidate.label)
        frozen_nll = _summary_metric(
            frozen_rows, "frozen_through_2021", candidate.label, "nll"
        )
        rolling = [
            float(row["nll"])
            for row in rolling_rows
            if row["candidate"] == candidate.label
        ]
        rolling_p0 = {
            int(row["target_season"]): float(row["nll"])
            for row in rolling_rows
            if row["candidate"] == "P0"
        }
        value_stability = _stability_summary(coefficients, candidate)
        group_stability = value_stability[candidate.position_groups[0]]
        signs = group_stability["value_signs"]
        same_nonzero_sign = bool(signs) and all(
            item != "zero" and item == signs[0] for item in signs
        )
        availability_ok = group_stability["availability_dominates_n"] == 0
        magnitude_ratio = group_stability["magnitude_ratio_last_to_first"]
        magnitude_ok = magnitude_ratio is not None and float(magnitude_ratio) >= 0.5
        rolling_deltas = [
            rolling[index] - rolling_p0[season]
            for index, season in enumerate(TARGET_SEASONS)
        ]
        missingness_pair = _paired_aggregate(
            paired_summaries,
            f"{candidate.label}_vs_{candidate.label}-M",
        )
        permutation_pair = _paired_aggregate(
            paired_summaries,
            f"{candidate.label}_vs_{candidate.label}-P",
        )
        rolling_missingness_pair = _paired_aggregate(
            paired_summaries,
            f"{candidate.label}_vs_{candidate.label}-M",
            protocol="rolling_origin",
        )
        rolling_missingness_by_season = {
            str(row["scope"]): float(row["mean_delta_nll"])
            for row in paired_summaries
            if row["comparison"] == f"{candidate.label}_vs_{candidate.label}-M"
            and row["protocol"] == "rolling_origin"
            and row["scope"] != "aggregate"
        }
        rolling_value_beats_availability = bool(
            rolling_missingness_pair
            and float(rolling_missingness_pair["mean_delta_nll"]) < 0
        )
        criteria = {
            "frozen_nll_improves_d5": frozen_nll < p0_nll,
            "beats_availability_only_control": bool(
                missingness_pair and float(missingness_pair["mean_delta_nll"]) < 0
            ),
            "beats_within_season_permutation": bool(
                permutation_pair and float(permutation_pair["mean_delta_nll"]) < 0
            ),
            "rolling_nll_improves_every_target_season": all(
                delta < 0 for delta in rolling_deltas
            ),
            "rolling_value_beats_availability_control_aggregate": rolling_value_beats_availability,
            "value_coefficient_same_nonzero_sign": same_nonzero_sign,
            "value_magnitude_not_collapsed": magnitude_ok,
        }
        screens[candidate.label] = {
            "candidate": candidate.label,
            "candidate_name": candidate.name,
            "position_group": candidate.position_groups[0],
            "position_group_label": POSITION_GROUP_LABELS[candidate.position_groups[0]],
            "frozen_nll": frozen_nll,
            "frozen_delta_nll_vs_P0": frozen_nll - p0_nll,
            "rolling_delta_nll_vs_P0": {
                str(season): delta
                for season, delta in zip(TARGET_SEASONS, rolling_deltas, strict=True)
            },
            "missingness_control": control.name if control else None,
            "permutation_control": permutation.name if permutation else None,
            "criteria": criteria,
            "rolling_delta_nll_vs_availability_control": rolling_missingness_by_season,
            "rolling_aggregate_delta_nll_vs_availability_control": (
                float(rolling_missingness_pair["mean_delta_nll"])
                if rolling_missingness_pair
                else None
            ),
            "rolling_availability_control_target_seasons_improved": sum(
                value < 0 for value in rolling_missingness_by_season.values()
            ),
            "diagnostics": {
                "availability_never_dominates_value": availability_ok,
                "availability_dominates_n": group_stability["availability_dominates_n"],
            },
            "stability": group_stability,
            "passed": all(criteria.values()),
        }

    p4 = primary[4]
    p4_nll = _summary_metric(frozen_rows, "frozen_through_2021", p4.label, "nll")
    p4_rolling = [
        float(row["nll"]) for row in rolling_rows if row["candidate"] == p4.label
    ]
    p4_p0 = {
        int(row["target_season"]): float(row["nll"])
        for row in rolling_rows
        if row["candidate"] == "P0"
    }
    p4_screen = {
        "candidate": p4.label,
        "candidate_name": p4.name,
        "position_groups": list(p4.position_groups),
        "frozen_nll": p4_nll,
        "frozen_delta_nll_vs_P0": p4_nll - p0_nll,
        "rolling_delta_nll_vs_P0": {
            str(season): value - p4_p0[season]
            for season, value in zip(TARGET_SEASONS, p4_rolling, strict=True)
        },
        "controls_run": p4.label in controls,
        "permutation_run": p4.label in permutations,
        "stability": _stability_summary(coefficients, p4),
        "interpretation": (
            "P4 improves over D5; inspect whether the gain is distributed across multiple stable value coefficients or is primarily availability structure."
            if p4_nll < p0_nll
            else "P4 does not improve D5 on frozen aggregate NLL."
        ),
    }
    screens[p4.label] = p4_screen
    return screens


def _fmt(value: object, digits: int = 4) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _markdown_metrics(
    rows: list[dict[str, object]], candidate_order: Iterable[str]
) -> list[str]:
    by_candidate = {str(row["candidate"]): row for row in rows}
    lines = [
        "| Candidate | NLL | ΔNLL vs P0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidate_order:
        row = by_candidate[candidate]
        lines.append(
            f"| {candidate} | {_fmt(row['nll'])} | {_fmt(row['delta_nll_vs_P0'], 6)} | {_fmt(row['crps'])} | {_fmt(row['expected_rank_mae'], 2)} | {_fmt(row['median_rank_mae'], 2)} | {_fmt(row['interval_80_coverage'], 3)} | {_fmt(row['interval_80_average_width'], 2)} |"
        )
    return lines


def render_report(
    path: Path,
    *,
    summary: dict[str, object],
    primary: tuple[Candidate, ...],
    frozen_summary: list[dict[str, object]],
    frozen_annual: list[dict[str, object]],
    rolling_rows: list[dict[str, object]],
    rolling_aggregate: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
    training_rows: list[dict[str, object]],
    coefficients: list[dict[str, object]],
    paired_summaries: list[dict[str, object]],
    reconstruction_rows: list[dict[str, object]],
) -> None:
    candidate_labels = [candidate.label for candidate in primary]
    p0 = next(row for row in frozen_summary if row["candidate"] == "P0")
    lines = [
        "# Defensive-transfer production by position group (issue 111)",
        "",
        "## Conclusion",
        "",
        f"Final recommendation: **{summary['recommendation']}**",
        "",
        summary["conclusion"],
        "",
        "## Frozen player construction",
        "",
        "The study uses the PR #105 impact values unchanged. It does not refit the player metric, change component weights, add a position taxonomy, or split any group further.",
        "",
        "| Group | Frozen source positions | Frozen impact components | Normalization |",
        "|---|---|---|---|",
        "| DL / EDGE | DL, EDGE, DE, DT, NT | tackles; tackles for loss; sacks; quarterback hurries | prior season × group |",
        "| LB | OLB, ILB, MLB, LB | tackles; tackles for loss; sacks; passes defended | prior season × group |",
        "| DB | CB, DB, S, FS, SS, NB | tackles; passes defended; interceptions | prior season × group |",
        "",
        "Impact is the equal-weight mean of standardized `log1p` components. The incoming team features are sums of those frozen player-level impacts.",
        "",
        "## Candidate definitions",
        "",
        "| Candidate | Definition | Added features beyond D5 |",
        "|---|---|---|",
    ]
    for candidate in primary:
        added = [
            feature for feature in candidate.features if feature not in d5_features()
        ]
        lines.append(
            f"| {candidate.label} | {candidate.description} | {', '.join(f'`{feature}`' for feature in added) or 'none'} |"
        )
    lines += [
        "",
        "## Missing-data treatment",
        "",
        "For each group, no incoming transfers are observed zeros with availability 1. Any incoming transfer whose frozen impact cannot be resolved is an unresolved group: its numeric value is neutralized to zero and availability is 0. Team-seasons remain in the model population. Rows before the defensive audit coverage window use the same neutral value with availability 0.",
        "",
        "## Protocol and parity",
        "",
        f"- D5 is the P0 control. Fits use data through {FROZEN_TRAIN_THROUGH} for frozen 2022–2025 scoring; rolling fits train through 2021, 2022, 2023, and 2024 for targets 2022, 2023, 2024, and 2025.",
        "- The production FBS population, Context preprocessing, model family, penalty, optimizer retry, H fallback, August 15 cutoff, response, and scoring implementation are unchanged.",
        "- No target-season outcomes enter feature construction or fitting.",
        f"- D5 parity: {'passed' if summary['d5_parity']['passed'] else 'failed'}; maximum absolute metric delta {_fmt(summary['d5_parity']['max_abs_metric_delta'], 9)} against the stored decomposition artifact.",
        f"- Aggregate reconstruction: {summary['reconstruction']['n_passed']} / {summary['reconstruction']['n_compared']} compared team-seasons passed at tolerance {RECONSTRUCTION_TOLERANCE:g}; unresolved groups are not silently treated as reconstructed.",
        "",
        "## Aggregate-versus-position reconstruction",
        "",
        "The position sum is compared with the existing aggregate only when all three position groups are observed.",
        "",
        "| Season | Compared | Passed | Maximum absolute delta |",
        "|---:|---:|---:|---:|",
    ]
    for season in sorted({int(row["season"]) for row in reconstruction_rows}):
        part = [row for row in reconstruction_rows if int(row["season"]) == season]
        compared = [row for row in part if row["parity_passed"] not in (None, "")]
        passed = [row for row in compared if _is_true(row["parity_passed"])]
        maximum = max(
            (abs(float(row["delta_position_sum_minus_aggregate"])) for row in compared),
            default=None,
        )
        lines.append(
            f"| {season} | {len(compared)} | {len(passed)} | {_fmt(maximum, 9)} |"
        )

    lines += [
        "",
        "## Coverage by season and position group",
        "",
        "The existing audit supplies an experience-mass coverage proxy. No impact-mass proxy is present in that artifact, so the report leaves that field unavailable rather than inventing one.",
        "Resolved impact counts include the audit's verified zero-recorded-box-score impacts; those are observed zeros, not unresolved players.",
        "",
        "| Season | Group | Incoming | Resolved impacts | Unresolved impacts | Experience-mass proxy | Complete teams | No incoming | Unresolved teams | Observed fraction |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in coverage_rows:
        lines.append(
            f"| {row['season']} | {row['position_group_label']} | {row['incoming_transfers']} | {row['resolved_player_impacts']} | {row['unresolved_player_impacts']} | {_fmt(row['experience_mass_coverage_proxy'], 3)} | {row['team_seasons_complete']} | {row['team_seasons_no_incoming']} | {row['team_seasons_unresolved']} | {_fmt(row['fraction_team_seasons_observed'], 3)} |"
        )

    lines += [
        "",
        "### Observed training coverage by rolling fit",
        "",
        "| Target | Train through | Group | Observed team-seasons | Total training team-seasons | Observed fraction |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for row in training_rows:
        lines.append(
            f"| {row['target_season']} | {row['train_through']} | {row['position_group_label']} | {row['n_observed_training_team_seasons']} | {row['n_training_team_seasons']} | {_fmt(row['fraction_observed_training_team_seasons'], 3)} |"
        )

    lines += [
        "",
        "## Frozen 2022–2025 metrics",
        "",
        *_markdown_metrics(frozen_summary, candidate_labels),
        "",
        "The D5 control P0 has aggregate NLL "
        + _fmt(p0["nll"])
        + ". Negative ΔNLL favors the position-group candidate.",
        "",
        "### Frozen year-by-year metrics",
        "",
        "| Season | Candidate | NLL | ΔNLL vs P0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frozen_annual:
        lines.append(
            f"| {row['target_season']} | {row['candidate']} | {_fmt(row['nll'])} | {_fmt(row['delta_nll_vs_P0'], 6)} | {_fmt(row['crps'])} | {_fmt(row['expected_rank_mae'], 2)} | {_fmt(row['median_rank_mae'], 2)} | {_fmt(row['interval_80_coverage'], 3)} | {_fmt(row['interval_80_average_width'], 2)} |"
        )

    lines += [
        "",
        "## Rolling-origin metrics",
        "",
        "### Year-by-year rolling metrics",
        "",
        "| Target | Candidate | NLL | ΔNLL vs P0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rolling_rows:
        lines.append(
            f"| {row['target_season']} | {row['candidate']} | {_fmt(row['nll'])} | {_fmt(row['delta_nll_vs_P0'], 6)} | {_fmt(row['crps'])} | {_fmt(row['expected_rank_mae'], 2)} | {_fmt(row['median_rank_mae'], 2)} | {_fmt(row['interval_80_coverage'], 3)} | {_fmt(row['interval_80_average_width'], 2)} |"
        )
    lines += [
        "",
        "### Weighted rolling aggregate",
        "",
        *_markdown_metrics(rolling_aggregate, candidate_labels),
        "",
        "### Rolling DB availability-only control",
        "",
        "P3-M is D5 plus the DB availability indicator, with the numeric DB impact removed. Both P3 and P3-M are refit at each rolling origin. Each row below is P3 minus P3-M, so a negative value means the resolved DB impact adds predictive information beyond resolvability. The aggregate pools the paired team-seasons across the four target seasons.",
        "",
        "| Comparison | Scope | Mean ΔNLL | Median ΔNLL | Fraction improved | Team-seasons |",
        "|---|---|---:|---:|---:|---:|",
    ]
    rolling_control_summaries = [
        row
        for row in paired_summaries
        if row["protocol"] == "rolling_origin" and str(row["reference"]).endswith("-M")
    ]
    for row in sorted(
        rolling_control_summaries,
        key=lambda row: (
            str(row["comparison"]),
            str(row["scope"]) != "aggregate",
            int(row["scope"]) if str(row["scope"]).isdigit() else -1,
        ),
    ):
        lines.append(
            f"| {row['comparison']} | {row['scope']} | {_fmt(row['mean_delta_nll'], 6)} | {_fmt(row['median_delta_nll'], 6)} | {_fmt(row['fraction_team_seasons_improved'], 3)} | {row['n_team_seasons']} |"
        )

    lines += [
        "",
        "## Paired team-season NLL diagnostics",
        "",
        "| Protocol | Comparison | Scope | Mean ΔNLL | Median ΔNLL | Fraction improved | Team-seasons |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in paired_summaries:
        if row["scope"] == "aggregate" or str(row["comparison"]).endswith("_vs_P0"):
            lines.append(
                f"| {row['protocol']} | {row['comparison']} | {row['scope']} | {_fmt(row['mean_delta_nll'], 6)} | {_fmt(row['median_delta_nll'], 6)} | {_fmt(row['fraction_team_seasons_improved'], 3)} | {row['n_team_seasons']} |"
            )

    control_metrics = [
        row
        for row in summary.get("control_metrics", [])
        if row.get("target_season") == "aggregate"
    ]
    if control_metrics:
        lines += [
            "",
            "### Frozen control metrics",
            "",
            "| Control | NLL | ΔNLL vs P0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in control_metrics:
            lines.append(
                f"| {row['candidate']} | {_fmt(row['nll'])} | {_fmt(row['delta_nll_vs_P0'], 6)} | {_fmt(row['crps'])} | {_fmt(row['expected_rank_mae'], 2)} | {_fmt(row['median_rank_mae'], 2)} | {_fmt(row['interval_80_coverage'], 3)} | {_fmt(row['interval_80_average_width'], 2)} |"
            )

    lines += [
        "",
        "## Coefficient paths and stability",
        "",
        "The coefficient artifact reports standardized location coefficients for returning production, incoming usage, each candidate's position-specific value, and its availability indicator. Rolling rows include sign, magnitude, and change from the previous fit.",
        "",
        "| Candidate | Group | Value signs | Availability dominates | First | Last | Last/first magnitude |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for candidate in primary[1:]:
        stability = summary["screens"][candidate.label].get("stability", {})
        if candidate.label == "P4":
            for group in candidate.position_groups:
                row = stability[group]
                lines.append(
                    f"| {candidate.label} | {POSITION_GROUP_LABELS[group]} | {', '.join(row['value_signs'])} | {row['availability_dominates_n']} | {_fmt(row['first_absolute_value'])} | {_fmt(row['last_absolute_value'])} | {_fmt(row['magnitude_ratio_last_to_first'])} |"
                )
        else:
            group = candidate.position_groups[0]
            row = stability
            lines.append(
                f"| {candidate.label} | {POSITION_GROUP_LABELS[group]} | {', '.join(row['value_signs'])} | {row['availability_dominates_n']} | {_fmt(row['first_absolute_value'])} | {_fmt(row['last_absolute_value'])} | {_fmt(row['magnitude_ratio_last_to_first'])} |"
            )

    lines += [
        "",
        "## Controls and position-group decisions",
        "",
        "A group is eligible only if it improves frozen D5 NLL, beats its frozen availability-only control, beats its within-season permutation, improves in every rolling target season, beats its rolling availability-only control in the weighted aggregate, keeps one non-zero coefficient sign, and does not collapse to less than half its first rolling magnitude. The rolling control table reports the target-season pattern as a diagnostic. Availability-coefficient dominance is reported as a stability diagnostic, not used as an automatic promotion veto; the direct availability-only control is the predictive test for a missingness artifact.",
        "",
        "| Group candidate | Frozen ΔNLL | Frozen availability control | Rolling availability control (aggregate) | Permutation control | Rolling persistence | Stable value path | Availability dominates (diagnostic) | Passed |",
        "|---|---:|:---:|:---:|:---:|:---:|:---:|---:|:---:|",
    ]
    for label in ("P1", "P2", "P3"):
        screen = summary["screens"][label]
        criteria = screen["criteria"]
        lines.append(
            f"| {label} ({screen['position_group_label']}) | {_fmt(screen['frozen_delta_nll_vs_P0'], 6)} | {'yes' if criteria['beats_availability_only_control'] else 'no'} | {'yes' if criteria['rolling_value_beats_availability_control_aggregate'] else 'no'} | {'yes' if criteria['beats_within_season_permutation'] else 'no'} | {'yes' if criteria['rolling_nll_improves_every_target_season'] else 'no'} | {'yes' if criteria['value_coefficient_same_nonzero_sign'] and criteria['value_magnitude_not_collapsed'] else 'no'} | {screen['diagnostics']['availability_dominates_n']} / {len(screen['stability']['value_coefficients'])} | {'yes' if screen['passed'] else 'no'} |"
        )
    p4 = summary["screens"]["P4"]
    lines += [
        "",
        "### Answers",
        "",
        f"- DL / EDGE: {'passes the full screen' if summary['screens']['P1']['passed'] else 'does not pass the full screen'}.",
        f"- LB: {'passes the full screen' if summary['screens']['P2']['passed'] else 'does not pass the full screen'}.",
        f"- DB: {'passes the full screen' if summary['screens']['P3']['passed'] else 'does not pass the full screen'}.",
        f"- Combining position groups: {p4['interpretation']}",
        f"- Aggregate defensive null: {summary['aggregate_null_interpretation']}",
        "",
        "## Artifacts",
        "",
        "- `candidate_definitions.json`, `summary.json` — frozen candidates, parity, controls, screens, and recommendation.",
        "- `control_metrics.csv` — frozen availability-only and permutation-control metrics when a primary candidate improves D5.",
        "- `position_coverage.csv`, `training_coverage.csv` — group coverage and rolling training counts.",
        "- `aggregate_reconstruction.csv` — position sum versus existing aggregate sanity check.",
        "- `candidate_summary.csv`, `candidate_annual_metrics.csv` — frozen aggregate and annual metrics.",
        "- `rolling_metrics.csv`, `rolling_aggregate.csv` — rolling annual and weighted aggregate metrics.",
        "- `rolling_control_metrics.csv`, `rolling_control_aggregate.csv` — rolling availability-only control metrics.",
        "- `coefficients.csv` — standardized coefficient paths and source coverage.",
        "- `paired_nll_summary.csv`, `paired_nll_by_team.csv` — paired diagnostics.",
        "- `plots/` — frozen deltas, rolling NLL, coefficients, and paired losses.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(
    output: Path,
    frozen_summary: list[dict[str, object]],
    rolling_rows: list[dict[str, object]],
    coefficients: list[dict[str, object]],
    paired_summaries: list[dict[str, object]],
) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 5))
    values = [float(row["delta_nll_vs_P0"]) for row in frozen_summary]
    labels = [str(row["candidate"]) for row in frozen_summary]
    axis.bar(
        labels,
        values,
        color=["#2f855a" if value < 0 else "#c53030" for value in values],
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("frozen ΔNLL versus P0")
    axis.set_title("Defensive-transfer position groups, frozen through 2021")
    figure.tight_layout()
    figure.savefig(plots / "frozen_delta_nll.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    for candidate in ("P0", "P1", "P2", "P3", "P4"):
        rows = [row for row in rolling_rows if row["candidate"] == candidate]
        axis.plot(
            [int(row["target_season"]) for row in rows],
            [float(row["nll"]) for row in rows],
            marker="o",
            label=candidate,
        )
    axis.set_xlabel("target season")
    axis.set_ylabel("NLL")
    axis.set_title("Position-group rolling-origin NLL")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "rolling_nll.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 6))
    for candidate in ("P1", "P2", "P3", "P4"):
        for role, style in (("defensive_value", "-"), ("availability", "--")):
            rows = [
                row
                for row in coefficients
                if row["protocol"] == "rolling_origin"
                and row["candidate"] == candidate
                and row["feature_role"] == role
            ]
            if not rows:
                continue
            label = f"{candidate} {role.replace('_', ' ')}"
            axis.plot(
                [int(row["target_season"]) for row in rows],
                [float(row["standardized_location_coefficient"]) for row in rows],
                marker="o",
                linestyle=style,
                label=label,
            )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("target season")
    axis.set_ylabel("standardized location coefficient")
    axis.set_title("Position-group value and availability coefficient paths")
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(plots / "coefficient_paths.png", dpi=160)
    plt.close(figure)

    aggregate = [row for row in paired_summaries if row["scope"] == "aggregate"]
    figure, axis = plt.subplots(figsize=(11, 5))
    labels = [f"{row['protocol']}: {row['comparison']}" for row in aggregate]
    values = [float(row["mean_delta_nll"]) for row in aggregate]
    axis.bar(
        labels,
        values,
        color=["#2f855a" if value < 0 else "#c53030" for value in values],
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("mean paired ΔNLL")
    axis.set_title("Paired position-group losses")
    axis.tick_params(axis="x", rotation=55)
    figure.tight_layout()
    figure.savefig(plots / "paired_delta_nll.png", dpi=160)
    plt.close(figure)


def _read_decomposition_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("candidate") == "D5_total_rp_plus_incoming"
                and row.get("target_season") == "aggregate"
            ):
                return {metric: float(row[metric]) for metric in METRICS}
    raise ValueError(f"stored D5 aggregate metrics are missing from {path}")


def _metric_parity(
    computed: Mapping[str, float], expected: Mapping[str, float], path: Path
) -> dict[str, object]:
    deltas = {metric: abs(computed[metric] - expected[metric]) for metric in METRICS}
    maximum = max(deltas.values(), default=None)
    result = {
        "candidate": "P0_D5",
        "reference_path": str(path),
        "reference_metrics": dict(expected),
        "computed_metrics": dict(computed),
        "metric_abs_deltas": deltas,
        "max_abs_metric_delta": maximum,
        "tolerance": D5_PARITY_TOLERANCE,
        "passed": maximum is not None and maximum <= D5_PARITY_TOLERANCE,
    }
    if not result["passed"]:
        raise ValueError(f"D5 parity failed: {result}")
    return result


def _source_hashes(
    source_root: Path,
    transfer_root: Path,
    paths: Iterable[Path],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            label = str(path.relative_to(source_root))
        except ValueError:
            label = str(path)
        result[label] = sha256_file(path)
    result.update(
        {
            f"transfer/{key}": value
            for key, value in transfer_source_hashes(transfer_root).items()
        }
    )
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.resolve()
    transfer_root = args.transfer_root.resolve()
    defensive_features_path = args.defensive_features.resolve()
    coverage_path = args.defensive_coverage.resolve()
    decomposition_path = args.decomposition_summary.resolve()
    output = args.output.resolve()
    configure_source_root(source_root)

    rows, cold, _ = prior.v1.load_rows(max_season=max(TARGET_SEASONS))
    fbs = [row for row in rows if row.subdivision == "fbs"]
    contextual, _ = c12.attach_context(fbs, c12.feature_index(), c12.cached_tenures())
    records, usage, portal_seasons, usage_seasons = prior.load_raw_transfer_data(
        transfer_root
    )
    required_portal = {FROZEN_TRAIN_THROUGH, *TARGET_SEASONS}
    required_usage = set(range(FROZEN_TRAIN_THROUGH - 1, max(TARGET_SEASONS)))
    if not required_portal <= portal_seasons:
        raise FileNotFoundError(
            f"D5 requires portal seasons {sorted(required_portal)}; found {sorted(portal_seasons)}"
        )
    if not required_usage <= usage_seasons:
        raise FileNotFoundError(
            f"D5 requires usage seasons {sorted(required_usage)}; found {sorted(usage_seasons)}"
        )
    transfer_features = prior.aggregate_team_features(
        records,
        usage,
        prior.fbs_feature_rows(contextual),
        covered_seasons=portal_seasons,
        cutoff=DEFAULT_CUTOFF,
    )
    contextual = prior.attach_transfer_features(contextual, transfer_features)
    position_bundle = load_position_features(defensive_features_path, coverage_path)
    contextual = attach_position_features(contextual, position_bundle.values)
    fallback = prior.fallback_for_panel(fbs, cold)
    fallback = [item for item in fallback if item.season in TARGET_SEASONS]

    reconstruction_compared = [
        row
        for row in position_bundle.reconstruction_rows
        if row["parity_passed"] is not None
    ]
    reconstruction_failed = [
        row for row in reconstruction_compared if not bool(row["parity_passed"])
    ]
    if reconstruction_failed:
        raise ValueError(
            "position-specific impact sum does not reproduce aggregate: "
            f"{reconstruction_failed[:3]}"
        )

    primary = candidate_definitions()
    frozen_predictions: dict[str, list[v1_1.PriorPrediction]] = {}
    frozen_models: dict[str, DirectRankModel] = {}
    for candidate in primary:
        predictions, model = fit_frozen(contextual, fallback, candidate)
        frozen_predictions[candidate.label] = predictions
        frozen_models[candidate.name] = model
        print(f"completed frozen {candidate.label}", flush=True)

    frozen_keys = {
        frozenset(item.key for item in values) for values in frozen_predictions.values()
    }
    if len(frozen_keys) != 1:
        raise ValueError("frozen candidates do not preserve identical target keys")
    frozen_rows: list[dict[str, object]] = []
    frozen_annual: list[dict[str, object]] = []
    p0_frozen = frozen_predictions["P0"]
    p0_metrics = prior.score(p0_frozen)
    p0_by_season = {
        season: prior.score([item for item in p0_frozen if item.season == season])
        for season in TARGET_SEASONS
    }
    for candidate in primary:
        frozen_rows.append(
            _metric_row(
                candidate,
                frozen_predictions[candidate.label],
                protocol="frozen_through_2021",
                target_season="aggregate",
                train_through=FROZEN_TRAIN_THROUGH,
                reference=p0_metrics,
            )
        )
        for season in TARGET_SEASONS:
            frozen_annual.append(
                _metric_row(
                    candidate,
                    [
                        item
                        for item in frozen_predictions[candidate.label]
                        if item.season == season
                    ],
                    protocol="frozen_through_2021",
                    target_season=season,
                    train_through=FROZEN_TRAIN_THROUGH,
                    reference=p0_by_season[season],
                )
            )

    d5_parity = _metric_parity(
        p0_metrics, _read_decomposition_metrics(decomposition_path), decomposition_path
    )
    rolling_predictions: dict[tuple[int, str], list[v1_1.PriorPrediction]] = {}
    rolling_models: dict[tuple[int, str], DirectRankModel] = {}
    rolling_rows: list[dict[str, object]] = []
    for target_season in TARGET_SEASONS:
        predictions_by_candidate: dict[str, list[v1_1.PriorPrediction]] = {}
        target_models: dict[str, DirectRankModel] = {}
        for candidate in primary:
            predictions, model = fit_rolling(
                contextual, fallback, candidate, target_season
            )
            predictions_by_candidate[candidate.label] = predictions
            target_models[candidate.name] = model
            rolling_predictions[(target_season, candidate.label)] = predictions
            rolling_models[(target_season, candidate.name)] = model
            print(f"completed rolling {target_season} {candidate.label}", flush=True)
        reference = prior.score(predictions_by_candidate["P0"])
        for candidate in primary:
            rolling_rows.append(
                _metric_row(
                    candidate,
                    predictions_by_candidate[candidate.label],
                    protocol="rolling_origin",
                    target_season=target_season,
                    train_through=target_season - 1,
                    reference=reference,
                )
            )

    rolling_aggregate = [
        metric_summary(
            rolling_rows, protocol="rolling_origin", candidate=candidate.label
        )
        for candidate in primary
    ]
    for row in rolling_aggregate:
        row["protocol"] = "rolling_origin_weighted_aggregate"

    # Controls are selected only after the primary frozen comparison, exactly
    # as permitted by the issue.  The P4 controls are also run whenever P4
    # improves, so a multi-group gain cannot hide behind missingness.
    p0_nll = float(frozen_rows[0]["nll"])
    successful = {
        candidate.label: candidate
        for candidate in primary[1:]
        if float(
            next(row for row in frozen_rows if row["candidate"] == candidate.label)[
                "nll"
            ]
        )
        < p0_nll
    }
    control_candidates: list[Candidate] = []
    permutation_candidates: list[Candidate] = []
    for candidate in primary[1:]:
        if candidate.label in successful:
            control_candidates.append(missingness_control(candidate))
            permutation_candidates.append(permutation_control(candidate))
    controls = {
        candidate.label.removesuffix("-M"): candidate
        for candidate in control_candidates
    }
    permutations = {
        candidate.label.removesuffix("-P"): candidate
        for candidate in permutation_candidates
    }
    for candidate in (*control_candidates, *permutation_candidates):
        source_rows = contextual
        if candidate.kind == "permutation_control":
            source_rows = permute_position_features(
                contextual,
                candidate.position_groups,
                seed=RANDOM_SEED,
            )
        predictions, model = fit_frozen(source_rows, fallback, candidate)
        frozen_predictions[candidate.label] = predictions
        frozen_models[candidate.name] = model
        print(f"completed frozen control {candidate.label}", flush=True)

    # The availability-only control must also be refit at each rolling origin.
    # This is the direct test of whether a numeric DB value adds information
    # beyond knowing that the corresponding transfer data is resolvable.
    primary_by_label = {candidate.label: candidate for candidate in primary}
    rolling_control_predictions: dict[tuple[int, str], list[v1_1.PriorPrediction]] = {}
    rolling_control_rows: list[dict[str, object]] = []
    rolling_control_aggregate: list[dict[str, object]] = []
    for candidate in control_candidates:
        base = primary_by_label[candidate.label.removesuffix("-M")]
        for target_season in TARGET_SEASONS:
            predictions, _ = fit_rolling(contextual, fallback, candidate, target_season)
            rolling_control_predictions[(target_season, candidate.label)] = predictions
            rolling_control_rows.append(
                _metric_row(
                    candidate,
                    predictions,
                    protocol="rolling_origin",
                    target_season=target_season,
                    train_through=target_season - 1,
                    reference=prior.score(rolling_predictions[(target_season, "P0")]),
                )
            )
            print(
                f"completed rolling {target_season} control {candidate.label}",
                flush=True,
            )
        aggregate = metric_summary(
            rolling_control_rows,
            protocol="rolling_origin",
            candidate=candidate.label,
        )
        aggregate["protocol"] = "rolling_origin_weighted_aggregate"
        aggregate["reference_candidate"] = base.label
        rolling_control_aggregate.append(aggregate)

    paired_details: list[dict[str, object]] = []
    paired_summaries: list[dict[str, object]] = []
    comparisons = [
        (f"{candidate.label}_vs_P0", candidate.label, "P0") for candidate in primary[1:]
    ]
    comparisons.extend(
        (
            f"{candidate.label.removesuffix('-M')}_vs_{candidate.label}",
            candidate.label.removesuffix("-M"),
            candidate.label,
        )
        for candidate in control_candidates
    )
    comparisons.extend(
        (
            f"{candidate.label.removesuffix('-P')}_vs_{candidate.label}",
            candidate.label.removesuffix("-P"),
            candidate.label,
        )
        for candidate in permutation_candidates
    )
    for comparison, candidate_label, reference_label in comparisons:
        details, summaries = paired_diagnostics(
            comparison,
            candidate_label,
            frozen_predictions[candidate_label],
            reference_label,
            frozen_predictions[reference_label],
        )
        paired_details.extend(details)
        paired_summaries.extend(summaries)
    for candidate in control_candidates:
        base_label = candidate.label.removesuffix("-M")
        rolling_base = [
            item
            for target_season in TARGET_SEASONS
            for item in rolling_predictions[(target_season, base_label)]
        ]
        rolling_control = [
            item
            for target_season in TARGET_SEASONS
            for item in rolling_control_predictions[(target_season, candidate.label)]
        ]
        details, summaries = paired_diagnostics(
            f"{base_label}_vs_{candidate.label}",
            base_label,
            rolling_base,
            candidate.label,
            rolling_control,
            protocol="rolling_origin",
        )
        paired_details.extend(details)
        paired_summaries.extend(summaries)

    control_rows: list[dict[str, object]] = []
    for candidate in (*control_candidates, *permutation_candidates):
        control_rows.append(
            _metric_row(
                candidate,
                frozen_predictions[candidate.label],
                protocol="frozen_through_2021",
                target_season="aggregate",
                train_through=FROZEN_TRAIN_THROUGH,
                reference=p0_metrics,
            )
        )
        for season in TARGET_SEASONS:
            control_rows.append(
                _metric_row(
                    candidate,
                    [
                        item
                        for item in frozen_predictions[candidate.label]
                        if item.season == season
                    ],
                    protocol="frozen_through_2021",
                    target_season=season,
                    train_through=FROZEN_TRAIN_THROUGH,
                    reference=p0_by_season[season],
                )
            )
    training_by_candidate = {
        candidate.name: [
            row for row in contextual if row.season <= FROZEN_TRAIN_THROUGH
        ]
        for candidate in primary
    }
    coefficients: list[dict[str, object]] = []
    coefficients.extend(
        coefficient_rows(
            primary[1:],
            frozen_models,
            training_by_candidate,
            protocol="frozen_through_2021",
            target_season="aggregate",
            train_through=FROZEN_TRAIN_THROUGH,
        )
    )
    previous: dict[tuple[str, str], float] = {}
    for target_season in TARGET_SEASONS:
        rolling_training = {
            candidate.name: [row for row in contextual if row.season < target_season]
            for candidate in primary
        }
        models = {
            candidate.name: rolling_models[(target_season, candidate.name)]
            for candidate in primary
        }
        coefficients.extend(
            coefficient_rows(
                primary[1:],
                models,
                rolling_training,
                protocol="rolling_origin",
                target_season=target_season,
                train_through=target_season - 1,
                previous=previous,
            )
        )

    coverage_training = training_coverage(contextual, position_bundle.statuses)
    screens = build_group_screens(
        primary,
        frozen_rows,
        rolling_rows,
        paired_summaries,
        coefficients,
        controls,
        permutations,
    )
    promoted = [
        group
        for label, group in (("P1", "dl_edge"), ("P2", "lb"), ("P3", "db"))
        if bool(screens[label]["passed"])
    ]
    if not promoted:
        recommendation = "retain D5 and stop defensive research"
        aggregate_null_interpretation = "The aggregate defensive null is not plausibly explained by cancellation across DL / EDGE, LB, and DB under this frozen decomposition; no group survives the controls and rolling-origin stability screen."
    elif len(promoted) == 1:
        recommendation = f"advance D5 + {POSITION_GROUP_LABELS[promoted[0]]} defensive position-group feature"
        aggregate_null_interpretation = "The aggregate null may have obscured a stable position-specific effect, but only the named surviving group is carried forward."
    else:
        names = ", ".join(POSITION_GROUP_LABELS[group] for group in promoted)
        recommendation = f"advance D5 + multiple named position-group features: {names}"
        aggregate_null_interpretation = "The aggregate null is plausibly related to heterogeneous position effects; multiple groups survive only if each independently passes the frozen, control, and rolling screen."

    reconstruction_summary = {
        "n_team_seasons": len(position_bundle.reconstruction_rows),
        "n_compared": len(reconstruction_compared),
        "n_passed": len(reconstruction_compared) - len(reconstruction_failed),
        "n_not_compared_unresolved": len(position_bundle.reconstruction_rows)
        - len(reconstruction_compared),
        "max_abs_delta": max(
            (
                abs(float(row["delta_position_sum_minus_aggregate"]))
                for row in reconstruction_compared
            ),
            default=None,
        ),
        "tolerance": RECONSTRUCTION_TOLERANCE,
    }
    source_paths = [
        source_root / "data/processed/modeling/team_season_rank_distributions.csv",
        source_root / "data/processed/preseason/team_season_features.csv",
        defensive_features_path,
        coverage_path,
        decomposition_path,
    ]
    summary: dict[str, object] = {
        "study": "issue_111_defensive_transfer_position_groups",
        "production_models_modified": False,
        "target_seasons": list(TARGET_SEASONS),
        "frozen_train_through": FROZEN_TRAIN_THROUGH,
        "rolling_train_through": {str(season): season - 1 for season in TARGET_SEASONS},
        "cutoff": DEFAULT_CUTOFF.isoformat(),
        "optimizer_penalty": 0.25,
        "random_seed": RANDOM_SEED,
        "position_taxonomy": {
            group: {
                "label": POSITION_GROUP_LABELS[group],
                "impact_feature": POSITION_IMPACT_FEATURES[group],
                "availability_feature": POSITION_AVAILABILITY_FEATURES[group],
            }
            for group in POSITION_GROUPS
        },
        "candidate_definitions": [
            {
                "label": candidate.label,
                "name": candidate.name,
                "description": candidate.description,
                "features": list(candidate.features),
                "position_groups": list(candidate.position_groups),
                "kind": candidate.kind,
            }
            for candidate in primary
        ],
        "missingness_controls": [candidate.name for candidate in control_candidates],
        "permutation_controls": [
            candidate.name for candidate in permutation_candidates
        ],
        "missing_data_policy": {
            "no_incoming_status": "no_incoming_transfer_group",
            "unresolved_status": "unresolved",
            "neutral_value": 0.0,
            "observed_availability": 1.0,
            "unresolved_availability": 0.0,
            "population_preserved": True,
        },
        "portal_seasons_available": sorted(portal_seasons),
        "usage_seasons_available": sorted(usage_seasons),
        "d5_parity": d5_parity,
        "reconstruction": reconstruction_summary,
        "frozen_metrics": frozen_rows,
        "control_metrics": control_rows,
        "rolling_metrics": rolling_rows,
        "rolling_aggregate": rolling_aggregate,
        "rolling_control_metrics": rolling_control_rows,
        "rolling_control_aggregate": rolling_control_aggregate,
        "coverage_by_season_group": position_bundle.coverage_rows,
        "training_coverage": coverage_training,
        "paired_nll_summary": paired_summaries,
        "screens": screens,
        "promoted_position_groups": promoted,
        "recommendation": recommendation,
        "aggregate_null_interpretation": aggregate_null_interpretation,
        "conclusion": (
            f"P0 D5 aggregate NLL is {_fmt(p0_nll)}. "
            f"The individual position-group decisions are DL / EDGE: {'pass' if screens['P1']['passed'] else 'fail'}, "
            f"LB: {'pass' if screens['P2']['passed'] else 'fail'}, and DB: {'pass' if screens['P3']['passed'] else 'fail'}. "
            f"P4 has frozen ΔNLL {_fmt(screens['P4']['frozen_delta_nll_vs_P0'], 6)} versus D5."
        ),
        "source_hashes": _source_hashes(source_root, transfer_root, source_paths),
    }

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "candidate_definitions.json", summary["candidate_definitions"])
    write_json(output / "summary.json", summary)
    write_csv(output / "position_coverage.csv", position_bundle.coverage_rows)
    write_csv(output / "training_coverage.csv", coverage_training)
    write_csv(
        output / "aggregate_reconstruction.csv", position_bundle.reconstruction_rows
    )
    write_csv(output / "candidate_summary.csv", frozen_rows)
    if control_rows:
        write_csv(output / "control_metrics.csv", control_rows)
    write_csv(output / "candidate_annual_metrics.csv", frozen_annual)
    write_csv(output / "rolling_metrics.csv", rolling_rows)
    write_csv(output / "rolling_aggregate.csv", rolling_aggregate)
    if rolling_control_rows:
        write_csv(output / "rolling_control_metrics.csv", rolling_control_rows)
        write_csv(output / "rolling_control_aggregate.csv", rolling_control_aggregate)
    write_csv(output / "coefficients.csv", coefficients)
    write_csv(output / "paired_nll_summary.csv", paired_summaries)
    write_csv(output / "paired_nll_by_team.csv", paired_details)
    plot_outputs(output, frozen_rows, rolling_rows, coefficients, paired_summaries)
    render_report(
        output / "report.md",
        summary=summary,
        primary=primary,
        frozen_summary=frozen_rows,
        frozen_annual=frozen_annual,
        rolling_rows=rolling_rows,
        rolling_aggregate=rolling_aggregate,
        coverage_rows=position_bundle.coverage_rows,
        training_rows=coverage_training,
        coefficients=coefficients,
        paired_summaries=paired_summaries,
        reconstruction_rows=position_bundle.reconstruction_rows,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument(
        "--transfer-root",
        type=Path,
        default=ROOT / "data/raw/cfbd/preseason/transfers",
        help="Directory containing immutable portal/*.json and usage/*.json payloads.",
    )
    parser.add_argument(
        "--defensive-features",
        type=Path,
        default=ROOT
        / "data/processed/defensive_transfer_audit/transfer_player_audit.csv",
        help="Frozen PR #105 transfer-player audit CSV.",
    )
    parser.add_argument(
        "--defensive-coverage",
        type=Path,
        default=ROOT
        / "data/processed/defensive_transfer_audit/coverage_by_position.csv",
        help="Frozen PR #105 position coverage CSV.",
    )
    parser.add_argument(
        "--decomposition-summary",
        type=Path,
        default=ROOT
        / "data/processed/transfer_signal_decomposition/candidate_summary.csv",
        help="Stored issue-96 candidate summary used for D5 parity.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/defensive_transfer_position_groups",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps({"study": summary["study"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
