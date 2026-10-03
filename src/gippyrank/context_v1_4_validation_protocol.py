"""Outcome-free checks and decision rule for registered Context 1.4 validation.

This module never constructs a 2026 candidate prior or reads a 2026 target.
The later validator consumes the checked-in registration through this loader.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from gippyrank.context_prior_v1_3 import load_validated_committed_2026_reconstruction
from gippyrank.context_prior_v1_4_candidate import candidate_spec_sha256
from gippyrank.history_annual_v1_1 import (
    history11_semantic_specification_sha256,
    load_validated_history_annual_artifact,
)


class ProtocolError(ValueError):
    """The registered protocol or one of its pinned sources is inconsistent."""


@dataclass(frozen=True)
class RegisteredProtocol:
    data: dict[str, Any]
    sha256: str


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_file(root: Path, path: str, expected_sha256: str) -> None:
    if _file_sha256(root / path) != expected_sha256:
        raise ProtocolError(f"registered source hash changed: {path}")


def _read_prior_rows(path: Path, model: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["team_id"]: row for row in rows}
    if (
        len(result) != len(rows)
        or any(row["season"] != "2026" or row["subdivision"] != "fbs" for row in rows)
        or any(row["spec_version"] != model for row in rows)
    ):
        raise ProtocolError(f"{path}: duplicate or wrong-season baseline rows")
    return result


def _validate_sources(root: Path, data: dict[str, Any]) -> None:
    models = data["models"]
    candidate = models["candidate"]
    context = models["context_1_3"]
    history = models["history_1_1"]
    pinned = (
        (candidate, "audit_spec"),
        (candidate, "builder"),
        (context, "fit_source"),
        (context, "predictions"),
        (context, "transfer_features"),
        (context, "transfer_manifest"),
        (context, "transfer_provenance"),
        (history, "predictions"),
        (history, "fitted_instance"),
        (data["descriptive_strata"]["positive_context_only_magnitude"], "source"),
    )
    for owner, prefix in pinned:
        _check_file(
            root,
            owner[f"{prefix}_path"],
            owner[f"{prefix}_file_sha256"]
            if prefix != "source"
            else owner["source_sha256"],
        )

    audit_spec = json.loads((root / candidate["audit_spec_path"]).read_text())
    if candidate_spec_sha256(audit_spec) != candidate["candidate_semantics_sha256"]:
        raise ProtocolError("frozen Context 1.4 candidate identity changed")
    fit_source = json.loads((root / context["fit_source_path"]).read_text())
    if (
        any(
            fit_source[key] != context[expected]
            for key, expected in (
                ("source_identity_sha256", "fit_source_identity_sha256"),
                ("model_metadata_sha256", "model_metadata_sha256"),
                ("frozen_model_spec_identity_sha256", "semantic_model_spec_sha256"),
                ("provenance_class", "fit_provenance_class"),
            )
        )
        or fit_source["target_season"] != 2026
        or fit_source["trained_through_season"] != 2025
    ):
        raise ProtocolError("Context 1.3 fit identity changed")
    transfer_rows, transfer_source = load_validated_committed_2026_reconstruction(
        (root / context["transfer_features_path"]).parent
    )
    if (
        len(transfer_rows) != data["population"]["expected_count"]
        or transfer_source.source_identity_sha256
        != context["transfer_source_identity_sha256"]
        or transfer_source.provenance_class != context["transfer_provenance_class"]
    ):
        raise ProtocolError("Context 1.3 transfer identity changed")
    if history11_semantic_specification_sha256() != history["semantic_spec_sha256"]:
        raise ProtocolError("History 1.1 semantic identity changed")
    history_source = load_validated_history_annual_artifact(
        root / history["predictions_path"],
        root / history["fitted_instance_path"],
        target_season=2026,
        trained_through_season=2025,
    )
    if (
        history_source.model_metadata_sha256 != history["model_metadata_sha256"]
        or history_source.prediction_semantic_sha256
        != history["predictions_semantic_sha256"]
        or history_source.provenance_class != history["provenance_class"]
    ):
        raise ProtocolError("History 1.1 retained source identity changed")
    history_instance = json.loads((root / history["fitted_instance_path"]).read_text())
    if (
        history_instance["model_family"] != "history_prior"
        or history_instance["spec_version"] != "1.1"
        or history_instance["target_season"] != 2026
        or history_instance["trained_through_season"] != 2025
    ):
        raise ProtocolError("History 1.1 retained fit identity changed")

    c_rows = _read_prior_rows(root / context["predictions_path"], "1.3")
    h_rows = _read_prior_rows(root / history["predictions_path"], "1.1")
    population = data["population"]
    ids = population["expected_team_ids"]
    if (
        ids != sorted(set(ids))
        or len(ids) != population["expected_count"]
        or set(c_rows) != set(ids)
        or set(h_rows) != set(ids)
        or hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        != population["team_ids_sha256"]
    ):
        raise ProtocolError("locked 2026 FBS population changed")
    cold = sorted(
        team_id
        for team_id, row in c_rows.items()
        if row["prior_method"] != "same_subdivision_lag1"
    )
    if cold != population["cold_start_team_ids"]:
        raise ProtocolError("locked cold-start population changed")

    evidence = data["evidence"]
    _check_file(root, evidence["likelihood_path"], evidence["likelihood_sha256"])
    _check_file(
        root,
        evidence["retained_2025_cutoffs_path"],
        evidence["retained_2025_cutoffs_file_sha256"],
    )
    old = json.loads((root / evidence["retained_2025_cutoffs_path"]).read_text())
    shifted = [
        datetime.fromisoformat(item["cutoff"]) + timedelta(days=364)
        for item in old["cutoffs"]
    ]
    current = [
        datetime.fromisoformat(item["cutoff_utc"]) for item in data["checkpoints"]
    ]
    if current != shifted:
        raise ProtocolError("2026 checkpoints differ from registered calendar shift")
    fcs = evidence["fcs_fallback"]
    fcs_ids: set[str] = set()
    for source in fcs["raw_schedule_files"]:
        _check_file(root, source["path"], source["sha256"])
        for game in json.loads((root / source["path"]).read_text()):
            for side in ("home", "away"):
                if (
                    str(game.get(f"{side}Classification", "")).lower() == "fcs"
                    and game.get(f"{side}Id") is not None
                ):
                    fcs_ids.add(str(game[f"{side}Id"]))
    if (
        len(fcs_ids) != fcs["population_size"]
        or hashlib.sha256(
            json.dumps(sorted(fcs_ids), separators=(",", ":")).encode()
        ).hexdigest()
        != fcs["population_team_ids_sha256"]
        or not fcs["target_corpus_cannot_supply_fcs_population"]
    ):
        raise ProtocolError("outcome-free FCS fallback population changed")


def _validate_choices(data: dict[str, Any]) -> None:
    if data["schema_version"] != 1 or data["registration"]["issue"] != 164:
        raise ProtocolError("unknown registration version or issue")
    population = data["population"]
    if (
        not 1
        <= population["minimum_scored_teams_for_decision"]
        <= population["expected_count"]
    ):
        raise ProtocolError("invalid population coverage rule")
    checkpoints = data["checkpoints"]
    cutoffs = [datetime.fromisoformat(item["cutoff_utc"]) for item in checkpoints]
    if (
        len(checkpoints) != 7
        or [item["index"] for item in checkpoints] != list(range(1, 8))
        or checkpoints[0]["id"] != "preseason"
        or checkpoints[-1]["id"] != "final"
        or any(
            cutoff.year != 2026
            or cutoff.utcoffset() is None
            or cutoff.utcoffset().total_seconds() != 0
            for cutoff in cutoffs
        )
        or cutoffs != sorted(set(cutoffs))
        or data["evidence"]["preseason_game_count"] != 0
        or data["evidence"]["result_availability_lag_hours"] != 48
    ):
        raise ProtocolError("invalid checkpoint or evidence rule")
    analysis = data["analysis"]
    primary = analysis["primary"]
    bootstrap = analysis["bootstrap"]
    decision = analysis["decision"]
    threshold = decision["minimum_practical_effect_nats_per_team"]
    if (
        primary["checkpoint_id"] != "final"
        or primary["score"] != "posterior_target_weighted_nll"
        or primary["difference"] != "candidate_minus_context_1_3"
        or primary["weighting"] != "one_equal_weight_per_scored_team"
        or bootstrap["method"] != "paired_percentile"
        or bootstrap["unit"] != "team_id"
        or bootstrap["resamples"] != 10000
        or bootstrap["seed"] != 1642026
        or bootstrap["confidence_level"] != 0.95
        or bootstrap["quantile_method"] != "linear"
        or threshold <= 0
        or decision["promote"]
        != {"mean_difference_lte": -threshold, "bootstrap_upper_lt": 0.0}
        or decision["retain_context_1_3"]
        != {"mean_difference_gte": threshold, "bootstrap_lower_gt": 0.0}
        or not decision["secondary_metrics_cannot_override"]
    ):
        raise ProtocolError("invalid primary analysis or decision rule")
    strata = data["descriptive_strata"]["positive_context_only_magnitude"]
    if not 0 < strata["positive_only_q50"] < strata["positive_only_q75"]:
        raise ProtocolError("invalid development-derived stratum cutpoints")


def load_registered_protocol(
    root: Path, config_path: Path | None = None
) -> RegisteredProtocol:
    """Read and check the registration without opening 2026 candidate results."""
    config = config_path or root / "config/context_v1_4_validation.json"
    raw = config.read_bytes()
    data = json.loads(raw)
    _validate_choices(data)
    _validate_sources(root, data)
    sha256 = hashlib.sha256(raw).hexdigest()
    provenance_path = (
        root / "data/processed/context_v1_4_validation/preregistration.json"
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance["validation_config_sha256"] != sha256
        or provenance["candidate_semantics_sha256"]
        != data["models"]["candidate"]["candidate_semantics_sha256"]
        or provenance["issue"] != data["registration"]["issue"]
    ):
        raise ProtocolError("registration provenance does not match the config")
    return RegisteredProtocol(data=data, sha256=sha256)


def paired_mean_bootstrap(
    differences_by_team_id: dict[str, float], protocol: RegisteredProtocol
) -> tuple[float, float, float]:
    """Return mean and registered percentile interval for paired differences."""
    if not differences_by_team_id:
        raise ValueError("a paired population is required")
    values = np.asarray(
        [differences_by_team_id[key] for key in sorted(differences_by_team_id)],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("paired differences must be finite")
    bootstrap = protocol.data["analysis"]["bootstrap"]
    rng = np.random.Generator(np.random.PCG64(bootstrap["seed"]))
    draws = rng.integers(0, len(values), size=(bootstrap["resamples"], len(values)))
    means = values[draws].mean(axis=1)
    tail = (1 - bootstrap["confidence_level"]) / 2
    low, high = np.quantile(
        means, [tail, 1 - tail], method=bootstrap["quantile_method"]
    )
    return float(values.mean()), float(low), float(high)


def classify_primary(
    mean_difference: float,
    interval_low: float,
    interval_high: float,
    scored_teams: int,
    protocol: RegisteredProtocol,
) -> str:
    """Apply the locked decision boundary after all identity guardrails pass."""
    if (
        not all(
            math.isfinite(value)
            for value in (mean_difference, interval_low, interval_high)
        )
        or interval_low > interval_high
        or scored_teams < 0
        or scored_teams > protocol.data["population"]["expected_count"]
    ):
        raise ValueError("invalid paired endpoint")
    if scored_teams < protocol.data["population"]["minimum_scored_teams_for_decision"]:
        return "inconclusive"
    decision = protocol.data["analysis"]["decision"]
    if (
        mean_difference <= decision["promote"]["mean_difference_lte"]
        and interval_high < decision["promote"]["bootstrap_upper_lt"]
    ):
        return "promote"
    if (
        mean_difference >= decision["retain_context_1_3"]["mean_difference_gte"]
        and interval_low > decision["retain_context_1_3"]["bootstrap_lower_gt"]
    ):
        return "retain_context_1_3"
    return "inconclusive"
