"""Materialize the canonical 2026 Context 1.4 prior from retained inputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from gippyrank.artifact_hashes import posterior_pmfs_sha256
from gippyrank.context_prior import InferenceRow
from gippyrank.context_prior_v1_3 import (
    load_validated_committed_2026_reconstruction,
    load_validated_context13_fitted_model,
    moderated_location_points,
)
from gippyrank.context_prior_v1_4 import (
    VALIDATION_RESULT_PATH,
    construct_production_prior,
    production_semantics_sha256,
    production_spec,
)
from gippyrank.context_prior_v1_4_candidate import (
    Context13FallbackSource,
    Context13PriorInput,
)
from gippyrank.context_v1_4_validation_protocol import load_registered_protocol
from gippyrank.context_v1_4_validator import (
    VALIDATOR_INPUTS_PATH,
    construct_frozen_priors,
    load_validator_inputs,
)
from gippyrank.history_annual_v1_1 import load_validated_history_annual_artifact
from gippyrank.posterior.engine import Team
from gippyrank.preseason import pmf_summaries

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("data/processed/preseason/context_v1_4/annual/2026")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path = ROOT) -> dict[str, object]:
    """Fail closed if source, candidate, and production PMFs differ."""
    protocol = load_registered_protocol(root)
    inputs = load_validator_inputs(root, protocol)
    validation_path = root / VALIDATION_RESULT_PATH
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("decision") != "promote":
        raise ValueError("registered Context 1.4 validation did not promote")
    candidate_teams = construct_frozen_priors(root, protocol, inputs)["candidate"]
    context = protocol.data["models"]["context_1_3"]
    annual_path = root / context["predictions_path"]
    model, instance, fit_source = load_validated_context13_fitted_model(
        annual_path.parent / "fitted_model.json",
        annual_path.parent / "fitted_instance.json",
        fitted_source_path=root / context["fit_source_path"],
        prediction_artifact_path=annual_path,
    )
    _, transfer = load_validated_committed_2026_reconstruction(
        (root / context["transfer_features_path"]).parent
    )
    history = protocol.data["models"]["history_1_1"]
    history_source = load_validated_history_annual_artifact(
        root / history["predictions_path"],
        root / history["fitted_instance_path"],
        target_season=2026,
        trained_through_season=2025,
    )
    with annual_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        source_rows = list(reader)
    forecast_rows = {row["team_id"]: row for row in inputs["forecast_rows"]}
    cold_ids = set(protocol.data["population"]["cold_start_team_ids"])
    expected_ids = set(protocol.data["population"]["expected_team_ids"])
    if {row["team_id"] for row in source_rows} != expected_ids:
        raise ValueError("Context 1.3 source population changed")

    rows: list[dict[str, str]] = []
    pmfs: dict[str, np.ndarray] = {}
    counts = {"positive": 0, "nonpositive": 0, "cold_start": 0}
    for source_row in source_rows:
        team_id = source_row["team_id"]
        saved = forecast_rows[team_id]
        inference_row = InferenceRow(
            season=int(saved["season"]),
            subdivision=saved["subdivision"],
            team_id=team_id,
            team_name=saved["team_name"],
            population=int(saved["population"]),
            lag1_z=(
                None
                if saved["lag1_z"] is None
                else tuple(float(value) for value in saved["lag1_z"])
            ),
            lag_zs=tuple(
                tuple(float(value) for value in values) for values in saved["lag_zs"]
            ),
            features=saved["features"],
            cold_start_reason=saved["cold_start_reason"],
        )
        inference_row.require_no_target()
        if team_id in cold_ids:
            fallback = Context13FallbackSource.from_history_annual_source(
                source=history_source,
                target_season=2026,
                trained_through_season=2025,
                team_id=team_id,
                team_name=inference_row.team_name,
                population=inference_row.population,
                cold_start_reason=inference_row.cold_start_reason
                or "fcs_to_fbs_transition",
            )
            prior_input = Context13PriorInput.cold_start(
                fitted_instance=instance,
                fitted_model_source=fit_source,
                transfer_provenance=transfer,
                fallback_source=fallback,
            )
            if not np.array_equal(
                fallback.pmf, np.asarray(json.loads(source_row["pmf"]))
            ):
                raise ValueError(f"cold-start source PMF changed: {team_id}")
            counts["cold_start"] += 1
        else:
            exact = model.pmf(
                {name: inference_row.features[name] for name in model.feature_names},
                np.asarray(inference_row.lag1_z),
                inference_row.population,
            )
            if not np.allclose(
                exact, json.loads(source_row["pmf"]), rtol=0, atol=5.01e-13
            ):
                raise ValueError(f"Context 1.3 source PMF changed: {team_id}")
            prior_input = Context13PriorInput.fitted(
                model=model,
                fitted_instance=instance,
                fitted_model_source=fit_source,
                inference_row=inference_row,
                transfer_provenance=transfer,
                prior_pmf=exact,
                expected_team_id=team_id,
                expected_target_season=2026,
            )
            decomposition = prior_input.location_decomposition()
            assert decomposition is not None
            if decomposition.context_only_subtotal > 0:
                points = moderated_location_points(decomposition, 0.75)
                counts["positive"] += 1
            else:
                points = decomposition.conditional_location_points
                counts["nonpositive"] += 1
            source_row["conditional_location_mean"] = str(float(np.mean(points)))
        production = construct_production_prior(prior_input)
        candidate = candidate_teams[team_id]
        # Team performs the same final normalization in the registered
        # validator and normal snapshot loader. Compare after that boundary.
        production_team = Team(team_id, inference_row.team_name, "fbs", production.pmf)
        if not np.array_equal(production_team.prior, candidate.prior):
            raise ValueError(f"validated candidate/production PMF mismatch: {team_id}")
        source_row.update(
            model="context_prior_v1_4",
            pmf=json.dumps(production.pmf.tolist(), separators=(",", ":")),
            spec_version="1.4",
            **{key: str(value) for key, value in pmf_summaries(production.pmf).items()},
        )
        rows.append(source_row)
        pmfs[team_id] = production_team.prior
    if len(rows) != 138 or set(pmfs) != expected_ids:
        raise ValueError("Context 1.4 annual prior must contain 138 FBS teams")
    prior_hash = posterior_pmfs_sha256(pmfs)
    preseason_origin = next(
        origin
        for origin in protocol.data["forecast_origins"]
        if origin["publication_slot"] == "2026-preseason-context-1.3"
    )
    preseason_metadata = json.loads(
        (root / preseason_origin["context_files"]["metadata.json"]["path"]).read_text()
    )
    fcs_population = int(preseason_metadata["fcs_population_size"])
    validation_network = dict(pmfs)
    for team_id in preseason_metadata["fcs_fallback_team_ids"]:
        validation_network[str(team_id)] = Team(
            str(team_id),
            str(team_id),
            "fcs",
            np.full(fcs_population, 1.0 / fcs_population),
        ).prior
    validation_network_hash = posterior_pmfs_sha256(validation_network)
    registered_preseason = next(
        row
        for row in json.loads(
            (
                root / "data/processed/context_v1_4_validation/provenance.json"
            ).read_text()
        )["forecast_origins"]
        if row["origin"] == "2026-preseason-context-1.3" and row["model"] == "candidate"
    )
    if validation_network_hash != registered_preseason["posterior_sha256"]:
        raise ValueError("2026 annual prior differs from validated candidate preseason")

    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "predictions.csv"
    with predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    provenance: dict[str, object] = {
        **production_spec(),
        "production_semantics_sha256": production_semantics_sha256(),
        "target_season": 2026,
        "team_count": len(rows),
        "component_counts": counts,
        "prior_pmfs_sha256": prior_hash,
        "validated_preseason_network_sha256": validation_network_hash,
        "predictions_sha256": _sha256(predictions),
        "context_1_3_source_path": context["predictions_path"],
        "context_1_3_source_sha256": _sha256(annual_path),
        "context_1_3_fitted_model_sha256": _sha256(
            annual_path.parent / "fitted_model.json"
        ),
        "transfer_features_path": context["transfer_features_path"],
        "transfer_features_sha256": _sha256(root / context["transfer_features_path"]),
        "transfer_provenance": transfer.to_metadata(),
        "frozen_validator_inputs_path": VALIDATOR_INPUTS_PATH,
        "frozen_validator_inputs_sha256": _sha256(root / VALIDATOR_INPUTS_PATH),
        "validation_result_path": VALIDATION_RESULT_PATH,
        "validation_result_sha256": _sha256(validation_path),
        "validation_provenance_path": "data/processed/context_v1_4_validation/provenance.json",
        "validation_provenance_sha256": _sha256(
            root / "data/processed/context_v1_4_validation/provenance.json"
        ),
    }
    (output / "promotion.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return provenance


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
