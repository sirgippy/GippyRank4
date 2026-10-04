"""Outcome-free checks for the registered in-season Context 1.4 confirmation.

This module does not construct a 2026 candidate or open its scores. The later
validator consumes the checked-in registration through this loader.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
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


def _read_snapshot_games(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_forecast_origins(root: Path, data: dict[str, Any]) -> None:
    """Bind each named official origin to its retained local evidence state."""
    origins = data["forecast_origins"]
    expected_slots = (
        "2026-preseason-context-1.3",
        "2026-09-08",
        "2026-09-13",
        "2026-09-20",
        "2026-09-27",
    )
    if tuple(origin["publication_slot"] for origin in origins) != expected_slots:
        raise ProtocolError("official 2026 forecast origin list changed")
    cutoffs = [
        datetime.fromisoformat(origin["logical_cutoff_utc"]) for origin in origins
    ]
    if (
        cutoffs != sorted(set(cutoffs))
        or any(
            cutoff.year != 2026
            or cutoff.utcoffset() is None
            or cutoff.utcoffset().total_seconds() != 0
            for cutoff in cutoffs
        )
        or cutoffs[0].isoformat() != "2026-08-22T23:59:59+00:00"
    ):
        raise ProtocolError("official forecast cutoffs are invalid")
    for index, origin in enumerate(origins):
        states = []
        for arm, family, version in (
            ("context", "context", "1.3"),
            ("history", "history", "1.1"),
        ):
            files = origin[f"{arm}_files"]
            expected_base = (
                f"data/processed/snapshots/2026/{origin[f'{arm}_snapshot_id']}"
                f"/predictive/{family}/"
            )
            for name in ("metadata.json", "included_games.csv", "posterior_pmfs.csv"):
                entry = files[name]
                if entry["path"] != expected_base + name:
                    raise ProtocolError(
                        f"{origin['publication_slot']}: snapshot path changed"
                    )
                _check_file(root, entry["path"], entry["sha256"])
            metadata = json.loads((root / files["metadata.json"]["path"]).read_text())
            games = _read_snapshot_games(root / files["included_games.csv"]["path"])
            if (
                metadata["snapshot_id"] != origin[f"{arm}_snapshot_id"]
                or metadata["prior_family"] != family
                or metadata["prior_model_version"] != version
                or metadata["historical_likelihood_sha256"]
                != data["evidence"]["likelihood_sha256"]
                or metadata["included_game_count"] != origin["included_game_count"]
                or len(games) != origin["included_game_count"]
                or [row["id"] for row in games]
                != [str(game_id) for game_id in metadata["included_game_ids"]]
                or (metadata["effective_cutoff"] or cutoffs[0].isoformat())
                != origin["logical_cutoff_utc"]
            ):
                raise ProtocolError(
                    f"{origin['publication_slot']}: official snapshot mismatch"
                )
            if (
                index
                and (
                    arm == "context" or "posterior_inference_configuration" in metadata
                )
                and metadata.get("posterior_inference_configuration")
                != {
                    **data["evidence"]["inference"],
                    "implementation": "deterministic_damped_loopy_sum_product",
                }
            ):
                raise ProtocolError(
                    f"{origin['publication_slot']}: inference configuration changed"
                )
            states.append((files, metadata, games))
        if (
            states[0][0]["included_games.csv"]["sha256"]
            != states[1][0]["included_games.csv"]["sha256"]
        ):
            raise ProtocolError(
                f"{origin['publication_slot']}: Context/History evidence differs"
            )
        if index == 0 and origin["included_game_count"] != 0:
            raise ProtocolError("preseason origin must have no games")
        if (
            index > 0
            and origin["included_game_count"]
            <= origins[index - 1]["included_game_count"]
        ):
            raise ProtocolError("official evidence counts must increase")


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
    _validate_forecast_origins(root, data)
    fcs = evidence["fcs_fallback"]
    fcs_ids = fcs["population_team_ids"]
    if (
        fcs_ids != sorted(set(fcs_ids))
        or len(fcs_ids) != fcs["population_size"]
        or hashlib.sha256(
            json.dumps(fcs_ids, separators=(",", ":")).encode()
        ).hexdigest()
        != fcs["population_team_ids_sha256"]
        or fcs["raw_schedule_hash_role"]
        != "registration_provenance_only_not_mutable_week_6_score_source"
        or fcs["unseen_forecast_fcs_team_rule"]
        != "locked_uniform_128_rank_fallback_for_forecast_only_without_future_outcome_evidence"
    ):
        raise ProtocolError("outcome-free FCS fallback population changed")
    bridge = data["development_predictive_bridge"]
    _check_file(root, bridge["script_path"], bridge["script_sha256"])
    _check_file(root, bridge["result_path"], bridge["result_sha256"])
    bridge_result = json.loads((root / bridge["result_path"]).read_text())
    if (
        bridge_result["candidate_semantics_sha256"]
        != candidate["candidate_semantics_sha256"]
        or bridge_result["cutoff_index"] != 3
        or set(bridge_result["seasons"]) != {"2022", "2023", "2024", "2025"}
        or bridge_result["pooled"]["context_1_3"]["n"] != 2700
        or bridge_result["pooled"]["context_1_4"]["n"] != 2700
    ):
        raise ProtocolError("development predictive bridge identity changed")


def _validate_choices(data: dict[str, Any]) -> None:
    if data["schema_version"] != 2 or data["registration"]["issue"] != 164:
        raise ProtocolError("unknown registration version or issue")
    analysis = data["analysis"]
    primary = analysis["primary"]
    bootstrap = analysis["bootstrap"]
    decision = analysis["decision"]
    if (
        primary["score"] != "posterior_predictive_margin_nll"
        or primary["difference"] != "candidate_minus_context_1_3"
        or primary["weighting"] != "one_equal_weight_per_unique_completed_game"
        or bootstrap["method"] != "paired_percentile"
        or bootstrap["unit"] != "game_id"
        or bootstrap["resamples"] != 10000
        or bootstrap["seed"] != 1642026
        or bootstrap["confidence_level"] != 0.95
        or bootstrap["quantile_method"] != "linear"
        or decision["retain_context_1_3"]
        != {"paired_mean_delta_gt": 0.0, "paired_interval_low_gt": 0.0}
        or decision["otherwise_if_valid"] != "promote"
        or not decision["guardrails_required"]
        or not decision["week_6_alone_cannot_veto_except_structural_failure"]
        or not decision["secondary_metrics_cannot_override_primary"]
    ):
        raise ProtocolError("invalid primary analysis or decision rule")
    evidence = data["evidence"]
    if (
        evidence["forecast_assignment"]
        != "latest_registered_origin_with_logical_cutoff_strictly_before_kickoff"
        or evidence["week_6_origin"] != "2026-09-27"
        or not evidence["each_game_scored_once"]
        or not evidence["snapshot_local_included_game_evidence_preferred"]
        or evidence["minimum_full_sample_games"] != 500
        or evidence["minimum_week_6_games"] != 40
        or evidence["inference"]
        != {"max_iterations": 500, "tolerance": 1e-9, "damping": 0.35}
    ):
        raise ProtocolError("invalid 2026 evidence rule")
    bridge = data["development_predictive_bridge"]
    if (
        bridge["seasons"] != [2022, 2023, 2024, 2025]
        or bridge["historical_panel"] != "standard_midseason_cutoff_index_3_per_season"
        or not bridge["run_before_2026_candidate_score_access"]
        or not bridge["no_retuning"]
    ):
        raise ProtocolError("invalid development bridge rule")
    amendment = data["registration"]["protocol_amendment"]
    if (
        amendment["initial_pr_165_draft_commit"]
        != "a118ef21364232090ab084d6ea58fdb81f0c18e7"
        or amendment["candidate_score_influence"] != "none"
        or amendment["authoritative_preregistration"] != "final_merged_commit"
    ):
        raise ProtocolError("protocol amendment provenance changed")
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


def assign_completed_games_to_origins(
    game_rows: list[dict[str, str]], protocol: RegisteredProtocol, root: Path
) -> dict[str, str]:
    """Assign each eligible game once, using a pre-kickoff official state.

    The caller separately audits source completeness and terminal schedule
    status. No candidate posterior or score is loaded here.
    """
    origins = protocol.data["forecast_origins"]
    included_ids = {
        origin["publication_slot"]: {
            row["id"]
            for row in _read_snapshot_games(
                root / origin["context_files"]["included_games.csv"]["path"]
            )
        }
        for origin in origins
    }
    assigned: dict[str, str] = {}
    for row in game_rows:
        if row.get("completed", "").casefold() not in {"true", "1", "yes"}:
            continue
        try:
            season, week = int(row["season"]), int(row["week"])
        except (KeyError, TypeError, ValueError) as error:
            raise ProtocolError("completed game has invalid season or week") from error
        if season != 2026 or not 1 <= week <= 6 or row.get("seasonType") != "regular":
            continue
        if any(
            row.get(key, "").casefold() not in {"fbs", "fcs"}
            for key in ("homeClassification", "awayClassification")
        ):
            continue
        game_id = row.get("id", "")
        if (
            not game_id
            or not row.get("homeId")
            or not row.get("awayId")
            or row.get("homePoints", "") == ""
            or row.get("awayPoints", "") == ""
        ):
            raise ProtocolError(f"completed game {game_id}: missing scoring fields")
        try:
            kickoff = datetime.fromisoformat(row["startDate"])
        except (KeyError, TypeError, ValueError) as error:
            raise ProtocolError(f"completed game {game_id}: invalid kickoff") from error
        if kickoff.utcoffset() is None:
            raise ProtocolError(f"completed game {game_id}: kickoff lacks timezone")
        eligible_origins = [
            origin
            for origin in origins
            if datetime.fromisoformat(origin["logical_cutoff_utc"]) < kickoff
        ]
        if not eligible_origins:
            raise ProtocolError(f"completed game {game_id}: no earlier official origin")
        selected = eligible_origins[-1]["publication_slot"]
        if week == 6 and selected != protocol.data["evidence"]["week_6_origin"]:
            raise ProtocolError(f"Week 6 game {game_id}: wrong forecast origin")
        if game_id in included_ids[selected]:
            raise ProtocolError(
                f"game {game_id}: forecast origin already included its result"
            )
        if game_id in assigned:
            raise ProtocolError(f"game {game_id}: duplicate canonical game row")
        assigned[game_id] = selected
    return dict(sorted(assigned.items()))


def paired_mean_bootstrap(
    differences_by_game_id: dict[str, float], protocol: RegisteredProtocol
) -> tuple[float, float, float]:
    """Return mean and registered percentile interval for paired differences."""
    if not differences_by_game_id:
        raise ValueError("a paired population is required")
    values = np.asarray(
        [differences_by_game_id[key] for key in sorted(differences_by_game_id)],
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
    scored_games: int,
    week_6_games: int,
    protocol: RegisteredProtocol,
) -> str:
    """Apply the locked decision boundary after all identity guardrails pass."""
    if (
        not all(
            math.isfinite(value)
            for value in (mean_difference, interval_low, interval_high)
        )
        or interval_low > interval_high
        or scored_games < 0
        or week_6_games < 0
        or week_6_games > scored_games
    ):
        raise ValueError("invalid paired endpoint")
    evidence = protocol.data["evidence"]
    if (
        scored_games < evidence["minimum_full_sample_games"]
        or week_6_games < evidence["minimum_week_6_games"]
    ):
        return "inconclusive"
    decision = protocol.data["analysis"]["decision"]
    if (
        mean_difference > decision["retain_context_1_3"]["paired_mean_delta_gt"]
        and interval_low > decision["retain_context_1_3"]["paired_interval_low_gt"]
    ):
        return "retain_context_1_3"
    return "promote"
