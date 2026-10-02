"""Freeze, lineage, and parity checks for the Context 1.4 research candidate."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

import gippyrank.context_prior_v1_4_candidate as candidate_module
from gippyrank.context_positive_net_moderation import parts_from_fitted_contributions
from gippyrank.context_prior import AnnualFittedInstance, InferenceRow
from gippyrank.context_prior_v1_3 import (
    H_FEATURES,
    MODEL_FEATURE_NAMES,
    PRODUCTION_TRANSFER_PROVENANCE,
    RETROSPECTIVE_2026_PROVENANCE,
    RETROSPECTIVE_RESEARCH_PROVENANCE,
    ContextTransferInputProvenance,
    decompose_context13_location,
)
from gippyrank.context_prior_v1_3 import (
    sha256_json as context_sha256_json,
)
from gippyrank.context_prior_v1_4_candidate import (
    COLD_START_STATUS,
    FITTED_STATUS,
    Context13FallbackSource,
    Context13PriorInput,
    Context14CandidatePrior,
    candidate_spec_sha256,
    construct_candidate_prior,
    load_candidate_spec,
    sha256_json,
)
from gippyrank.preseason import (
    DirectRankModel,
    GenericRankPrior,
    Preprocessor,
    conditional_rank_mixture_pmf,
    rank_bin_edges,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/context_v1_4_candidate.json"
PMF_PARITY_ABSOLUTE_TOLERANCE = 5e-17


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _instance(
    *, target_season: int = 2026, trained_through_season: int = 2025, version: str = "1.3"
) -> AnnualFittedInstance:
    return AnnualFittedInstance(
        "context_prior", version, trained_through_season, target_season
    )


def _model() -> DirectRankModel:
    names = list(MODEL_FEATURE_NAMES)
    preprocessor = Preprocessor(
        tuple(names),
        {name: 0.0 for name in names},
        {name: 0.0 for name in names},
        {name: 1.0 for name in names},
    )
    beta = np.zeros(2 + 2 * len(names))
    beta[0] = 0.2  # Lag-1 coefficient.
    beta[1] = -1.0  # Intercept.
    base = beta[1:]
    base[1 + names.index(H_FEATURES[0])] = 0.3
    base[1 + names.index("coach_tenure_seasons")] = 0.25
    return DirectRankModel(
        feature_names=names,
        preprocessor=preprocessor,
        beta=beta,
        gamma=np.zeros(1 + 2 * len(names)),
        minimum_scale=0.1,
        penalty=0.25,
        optimizer={"success": True},
        lag_count=1,
        family="normal",
        degrees_of_freedom=None,
        location_feature_names=list(MODEL_FEATURE_NAMES),
        scale_feature_names=list(H_FEATURES),
    )


def _row(
    *, season: int = 2026, team_id: str = "team-1", coach_tenure: float = 4.0
) -> InferenceRow:
    features: dict[str, float | None] = {name: 0.0 for name in MODEL_FEATURE_NAMES}
    features[H_FEATURES[0]] = 0.5
    features[H_FEATURES[1]] = None
    features[H_FEATURES[2]] = 0.25
    features["coach_tenure_seasons"] = coach_tenure
    return InferenceRow(
        season=season,
        subdivision="fbs",
        team_id=team_id,
        team_name="Example University",
        population=8,
        lag1_z=(-1.0, 0.5, 2.0),
        lag_zs=(),
        features=features,
    )


def _provenance(
    season: int = 2026,
    *,
    team_id: str = "team-1",
    population: int = 8,
    target_team_ids: tuple[str, ...] | None = None,
    feature_values: dict[str, float | None] | None = None,
) -> ContextTransferInputProvenance:
    values = {
        name: 0.0
        for name in (
            "transfer_in_prior_usage_sum",
            "transfer_in_prior_defensive_impact_db_sum",
            "transfer_in_prior_defensive_impact_db_available",
        )
    }
    if feature_values:
        values.update({name: feature_values[name] for name in values})
    return ContextTransferInputProvenance.research_only(
        season,
        feature_artifact_id=f"synthetic-context-transfer-features/season-{season}",
        transfer_feature_values_by_team={team_id: values},
        source_artifact_payload={
            "purpose": "synthetic research-only candidate test input",
            "season": season,
            "team_id": team_id,
            "population": population,
            "transfer_features": values,
        },
        target_team_ids=target_team_ids,
    )


def _fitted_input(
    *, coach_tenure: float = 4.0, provenance: ContextTransferInputProvenance | None = None
) -> Context13PriorInput:
    model = _model()
    row = _row(coach_tenure=coach_tenure)
    return Context13PriorInput.fitted(
        model=model,
        fitted_instance=_instance(),
        inference_row=row,
        transfer_provenance=provenance or _provenance(),
        prior_pmf=model.pmf(row.features, np.asarray(row.lag1_z), row.population),
        expected_team_id=row.team_id,
        expected_target_season=row.season,
    )


def _model_from_metadata(metadata: dict[str, object]) -> DirectRankModel:
    preprocessing = metadata["preprocessing"]
    assert isinstance(preprocessing, dict)
    return DirectRankModel(
        feature_names=list(metadata["feature_names"]),
        preprocessor=Preprocessor(
            tuple(preprocessing["feature_names"]),
            dict(preprocessing["medians"]),
            dict(preprocessing["means"]),
            dict(preprocessing["scales"]),
        ),
        beta=np.asarray(metadata["location_coefficients"], dtype=float),
        gamma=np.asarray(metadata["log_scale_coefficients"], dtype=float),
        minimum_scale=float(metadata["minimum_scale"]),
        penalty=float(metadata["penalty"]),
        optimizer=metadata["optimizer"],
        lag_count=int(metadata["lag_count"]),
        family=str(metadata["family"]),
        degrees_of_freedom=metadata["degrees_of_freedom"],
        location_feature_names=list(metadata["location_feature_names"]),
        scale_feature_names=list(metadata["scale_feature_names"]),
    )


def _legacy_normal_mixture_pmf(
    locations: np.ndarray, scale: float, population: int
) -> np.ndarray:
    """Independent copy of the PR #159 research reference for parity only."""
    edges = rank_bin_edges(population)
    cdf = norm.cdf((edges[None, :] - locations[:, None]) / scale)
    masses = np.maximum(np.diff(cdf, axis=1), 0.0)
    pmf = np.mean(masses, axis=0)
    return pmf / pmf.sum()


def test_candidate_semantics_are_exact_and_lifecycle_is_not_hashed() -> None:
    audit_spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    loaded = load_candidate_spec(audit_spec)
    assert loaded["alpha"] == 0.75
    assert loaded["moderation_metric"] == "context_only_subtotal"
    assert loaded["base_model"]["spec_version"] == "1.3"
    assert loaded["moderation_rule"] == {
        "x_lte_0": "x",
        "x_gt_0": "0.75 * x",
        "formula": "min(x, 0) + 0.75 * max(x, 0)",
    }
    assert candidate_spec_sha256(audit_spec) == candidate_spec_sha256()

    changed_lifecycle = dict(audit_spec, validation_status="holdout validated")
    assert candidate_spec_sha256(changed_lifecycle) == candidate_spec_sha256()

    for key, value in (
        ("alpha", 0.70),
        ("moderation_metric", "history_derived_subtotal"),
        ("base_model", {"model_family": "context_prior", "spec_version": "1.2"}),
    ):
        tampered = dict(audit_spec)
        tampered[key] = value
        with pytest.raises(ValueError, match="semantics differ"):
            load_candidate_spec(tampered)


def test_embedded_semantic_edit_fails_candidate_construction(monkeypatch) -> None:
    tampered = json.loads(candidate_module._SEMANTIC_CONTRACT_JSON)
    tampered["alpha"] = 0.70
    monkeypatch.setattr(candidate_module, "_SEMANTIC_CONTRACT_JSON", json.dumps(tampered))
    with pytest.raises(ValueError, match="embedded Context 1.4 candidate semantics changed"):
        construct_candidate_prior(_fitted_input())


def test_candidate_semantics_load_without_repository_layout(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert candidate_spec_sha256() == candidate_spec_sha256(load_candidate_spec())
    assert load_candidate_spec()["alpha"] == 0.75


def test_context_model_hash_is_derived_and_fake_hash_is_not_accepted() -> None:
    prior = _fitted_input()
    candidate = construct_candidate_prior(prior)
    assert prior.fitted_model is not None
    assert candidate.context_model_sha256 == context_sha256_json(prior.fitted_model.metadata())
    assert candidate.context_model_sha256 != "a" * 64
    with pytest.raises(TypeError, match="created by construct_candidate_prior"):
        Context14CandidatePrior(**candidate.__dict__)
    with pytest.raises(TypeError):
        Context13PriorInput.fitted(
            model=prior.fitted_model,
            fitted_instance=prior.fitted_instance,
            inference_row=prior.inference_row,
            transfer_provenance=prior.transfer_provenance,
            prior_pmf=prior.source_prior_pmf(),
            expected_team_id=prior.team_id,
            expected_target_season=prior.fitted_instance.target_season,
            context_model_sha256="a" * 64,
        )


def test_wrong_context_model_spec_or_annual_identity_fails_closed() -> None:
    model = _model()
    row = _row()
    pmf = model.pmf(row.features, np.asarray(row.lag1_z), row.population)
    with pytest.raises(ValueError, match="rolling-origin Context 1.3 instance"):
        Context13PriorInput.fitted(
            model=model,
            fitted_instance=_instance(version="1.2"),
            inference_row=row,
            transfer_provenance=_provenance(),
            prior_pmf=pmf,
            expected_team_id=row.team_id,
            expected_target_season=2026,
        )
    bad_model = _model()
    bad_model.location_feature_names = [H_FEATURES[0]]
    with pytest.raises(ValueError, match="Context 1.3 model specification"):
        Context13PriorInput.fitted(
            model=bad_model,
            fitted_instance=_instance(),
            inference_row=row,
            transfer_provenance=_provenance(),
            prior_pmf=pmf,
            expected_team_id=row.team_id,
            expected_target_season=2026,
        )


@pytest.mark.parametrize("forbidden_coefficient", ["numeric", "missingness"])
def test_context_only_scale_coefficients_must_remain_zero(
    forbidden_coefficient: str,
) -> None:
    model = _model()
    feature_index = model.feature_names.index("coach_tenure_seasons")
    if forbidden_coefficient == "numeric":
        index = 1 + feature_index
    else:
        index = 1 + len(model.feature_names) + feature_index
    model.gamma[index] = 0.01
    row = _row()
    pmf = model.pmf(row.features, np.asarray(row.lag1_z), row.population)
    with pytest.raises(ValueError, match="Context-only scale coefficients must be zero"):
        Context13PriorInput.fitted(
            model=model,
            fitted_instance=_instance(),
            inference_row=row,
            transfer_provenance=_provenance(),
            prior_pmf=pmf,
            expected_team_id=row.team_id,
            expected_target_season=row.season,
        )


def test_canonical_decomposition_binds_team_season_and_source() -> None:
    model = _model()
    instance = _instance()
    provenance = _provenance()
    valid = decompose_context13_location(model, instance, _row(), provenance)
    assert valid.team_id == "team-1"
    assert valid.target_season == 2026
    assert valid.context_model_sha256 == context_sha256_json(model.metadata())
    assert valid.transfer_provenance.provenance_class == RETROSPECTIVE_RESEARCH_PROVENANCE

    with pytest.raises(ValueError, match="team input and model seasons differ"):
        decompose_context13_location(model, instance, _row(season=2025), provenance)
    with pytest.raises(ValueError, match="transfer provenance does not match"):
        Context13PriorInput.fitted(
            model=model,
            fitted_instance=instance,
            inference_row=_row(),
            transfer_provenance=_provenance(season=2025),
            prior_pmf=model.pmf(_row().features, np.asarray(_row().lag1_z), 8),
            expected_team_id="team-1",
            expected_target_season=2026,
        )
    with pytest.raises(ValueError, match="source team or season"):
        Context13PriorInput.fitted(
            model=model,
            fitted_instance=instance,
            inference_row=_row(),
            transfer_provenance=provenance,
            prior_pmf=model.pmf(_row().features, np.asarray(_row().lag1_z), 8),
            expected_team_id="different-team",
            expected_target_season=2026,
        )
    with pytest.raises(ValueError, match="team identity must be non-empty"):
        decompose_context13_location(
            model, instance, _row(team_id=""), provenance
        )
    with pytest.raises(TypeError, match="must come from a transfer loader"):
        ContextTransferInputProvenance(
            target_season=2026,
            provenance_class=RETROSPECTIVE_2026_PROVENANCE,
            transfer_feature_artifact_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            canonical_snapshot_ids=("snapshot",),
            canonical_snapshot_sha256=("c" * 64,),
            cutoff_state="retrospective_reconstruction",
            feature_artifact_id="f",
            manifest_artifact_id="m",
            target_fbs_team_ids=("team-1",),
            target_fbs_population_sha256="d" * 64,
            target_team_feature_sha256=(("team-1", "e" * 64),),
        )
    research = _provenance()
    assert research.provenance_class == RETROSPECTIVE_RESEARCH_PROVENANCE
    with pytest.raises(TypeError):
        replace(research, provenance_class=PRODUCTION_TRANSFER_PROVENANCE)
    with pytest.raises(TypeError):
        ContextTransferInputProvenance.research_only(
            2026,
            feature_artifact_id="fake-production",
            transfer_feature_values_by_team={"team-1": {
                "transfer_in_prior_usage_sum": 0.0,
                "transfer_in_prior_defensive_impact_db_sum": 0.0,
                "transfer_in_prior_defensive_impact_db_available": 0.0,
            }},
            source_artifact_payload={"claims": "production"},
            provenance_class=PRODUCTION_TRANSFER_PROVENANCE,
        )


def test_changed_transfer_input_cannot_reuse_the_same_authoritative_identity() -> None:
    model = _model()
    row = _row()
    provenance = _provenance()
    changed_features = dict(row.features)
    changed_features["transfer_in_prior_usage_sum"] = 1.0
    changed_row = InferenceRow(
        row.season,
        row.subdivision,
        row.team_id,
        row.team_name,
        row.population,
        row.lag1_z,
        row.lag_zs,
        changed_features,
    )
    changed_pmf = model.pmf(
        changed_features, np.asarray(changed_row.lag1_z), changed_row.population
    )
    with pytest.raises(ValueError, match="transfer values differ from their source artifact"):
        Context13PriorInput.fitted(
            model=model,
            fitted_instance=_instance(),
            inference_row=changed_row,
            transfer_provenance=provenance,
            prior_pmf=changed_pmf,
            expected_team_id=changed_row.team_id,
            expected_target_season=changed_row.season,
        )
def test_nonpositive_fitted_and_cold_start_pmfs_remain_bit_identical() -> None:
    negative = construct_candidate_prior(_fitted_input(coach_tenure=-4.0))
    negative_input = _fitted_input(coach_tenure=-4.0)
    np.testing.assert_array_equal(negative.pmf, negative_input.source_prior_pmf())

    generic = GenericRankPrior(location=0.2, scale=1.1, n_team_seasons=10)
    fallback = Context13FallbackSource.from_generic_rank_prior(
        prior=generic,
        target_season=2026,
        team_id="cold-team",
        team_name="Cold Start University",
        population=3,
        cold_start_reason="no_prior_rank_distribution",
    )
    with pytest.raises(ValueError, match="only valid for no-prior cold starts"):
        Context13FallbackSource.from_generic_rank_prior(
            prior=generic,
            target_season=2026,
            team_id="cold-team",
            team_name="Cold Start University",
            population=3,
            cold_start_reason="fcs_to_fbs_transition",
        )
    cold_source = fallback.pmf.copy()
    cold = Context13PriorInput.cold_start(
        fitted_instance=_instance(),
        transfer_provenance=_provenance(
            team_id="cold-team", population=3
        ),
        fallback_source=fallback,
    )
    cold_result = construct_candidate_prior(cold)
    assert cold_result.pmf.tobytes() == cold_source.tobytes()
    assert cold_result.component_status == COLD_START_STATUS
    assert cold_result.context_model_sha256 is None
    assert cold_result.fallback_source is not None
    assert cold_result.fallback_source.pmf_sha256 == sha256_json(cold_source.tolist())
    assert cold_result.fallback_source.source_parameters_sha256 == sha256_json(
        generic.metadata()
    )
    assert cold_result.fallback_source.metadata()["source_parameters"] == generic.metadata()
    with pytest.raises(TypeError, match="fallback sources must be loaded"):
        Context13FallbackSource(
            target_season=2026,
            team_id="cold-team",
            team_name="Cold Start University",
            population=3,
            cold_start_reason="no_prior_rank_distribution",
            model_identity="fake",
            method_identity="copy supplied PMF",
            source_artifact_id="fake.csv",
            source_artifact_sha256="a" * 64,
            source_parameters_json=None,
            source_parameters_sha256=None,
            pmf=np.array([0.2, 0.3, 0.5]),
            research_fixture=False,
        )
    with pytest.raises(TypeError):
        Context13PriorInput.cold_start(
            fitted_instance=_instance(),
            transfer_provenance=_provenance(team_id="cold-team", population=3),
            prior_pmf=np.array([0.2, 0.3, 0.5]),
        )
    with pytest.raises(ValueError, match="read-only"):
        cold_result.pmf[0] = 0.0


def test_canonical_history_cold_start_is_loaded_and_bound_without_context_model(
    tmp_path: Path,
) -> None:
    artifact_path = ROOT / "data/processed/preseason/history/annual/2026/predictions.csv"
    fallback = Context13FallbackSource.from_history_prediction_artifact(
        artifact_path,
        target_season=2026,
        trained_through_season=2025,
        team_id="16",
        team_name="Sacramento State",
        population=138,
        cold_start_reason="fcs_to_fbs_transition",
    )
    prior = Context13PriorInput.cold_start(
        fitted_instance=_instance(),
        transfer_provenance=_provenance(team_id="16", population=138),
        fallback_source=fallback,
    )
    candidate = construct_candidate_prior(prior)
    assert candidate.context_model_sha256 is None
    assert candidate.pmf.tobytes() == fallback.pmf.tobytes()
    assert candidate.fallback_source is not None
    assert candidate.fallback_source.source_artifact_sha256 == _sha256(artifact_path)
    assert candidate.fallback_source.model_identity == "history_prior/1.1"
    fake_path = tmp_path / "history_predictions.csv"
    fake_path.write_bytes(artifact_path.read_bytes())
    with pytest.raises(ValueError, match="canonical annual prediction artifact"):
        Context13FallbackSource.from_history_prediction_artifact(
            fake_path,
            target_season=2026,
            trained_through_season=2025,
            team_id="16",
            team_name="Sacramento State",
            population=138,
            cold_start_reason="fcs_to_fbs_transition",
        )


def test_shared_pmf_helper_preserves_context13_model_parity() -> None:
    model = _model()
    row = _row()
    locations, scale = model.conditional_parameters(row.features, np.asarray(row.lag1_z))
    expected = conditional_rank_mixture_pmf(locations, scale, row.population)
    actual = model.pmf(row.features, np.asarray(row.lag1_z), row.population)
    np.testing.assert_array_equal(actual, expected)

    edges = rank_bin_edges(row.population)
    legacy_cdf = norm.cdf((edges[None, :] - locations[:, None]) / scale)
    legacy_masses = np.maximum(np.diff(legacy_cdf, axis=1), 0.0)
    legacy_pmf = np.mean(legacy_masses, axis=0)
    legacy_pmf = legacy_pmf / legacy_pmf.sum()
    np.testing.assert_array_equal(actual, legacy_pmf)


def test_artifact_records_computed_source_semantics_and_retrospective_inputs() -> None:
    prior = _fitted_input()
    candidate = construct_candidate_prior(prior)
    artifact = json.loads(candidate.artifact_bytes())
    assert artifact["source_prior_pmf_sha256"] == sha256_json(
        prior.source_prior_pmf().tolist()
    )
    assert artifact["candidate_semantics_sha256"] == candidate_spec_sha256()
    assert prior.fitted_model is not None
    assert artifact["context_model_sha256"] == context_sha256_json(prior.fitted_model.metadata())
    assert artifact["target_season"] == 2026
    assert artifact["trained_through_season"] == 2025
    assert artifact["team_id"] == "team-1"
    assert artifact["transfer_input_provenance"]["provenance_class"] == (
        RETROSPECTIVE_RESEARCH_PROVENANCE
    )
    assert artifact["transfer_input_provenance"]["transfer_feature_artifact_sha256"]
    assert artifact["transfer_input_provenance"]["source_identity_sha256"] == (
        prior.transfer_provenance.source_identity_sha256
    )
    assert artifact["transfer_input_provenance"] == prior.transfer_provenance.to_metadata()
    assert artifact["transfer_input_provenance"]["transfer_feature_artifact_sha256"] == (
        prior.transfer_provenance.transfer_feature_artifact_sha256
    )
    assert artifact["transfer_input_provenance"]["source_manifest_sha256"] is None
    assert artifact["transfer_input_provenance"]["target_fbs_team_ids"] == ["team-1"]
    assert "diagnostic_paths" not in artifact["transfer_input_provenance"]
    assert artifact["decomposition_sha256"]


def test_candidate_construction_does_not_mutate_production_artifacts() -> None:
    protected = (
        ROOT / "data/processed/preseason/context_v1_3_candidate/predictions.csv",
        ROOT / "data/processed/preseason/history/predictions.csv",
        ROOT / "data/processed/context_history_crossover/hybrid_prior_results.csv",
        ROOT / "data/processed/context_positive_net_moderation/team_season_results.csv",
    )
    before = {path: _sha256(path) for path in protected}
    construct_candidate_prior(_fitted_input())
    assert {path: _sha256(path) for path in protected} == before


def test_canonical_candidate_path_reproduces_every_pr159_development_pmf() -> None:
    audit_spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    parity_path = ROOT / "data/processed/context_v1_4_candidate/development_panel_parity.json"
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    assert parity["candidate_spec_sha256"] == candidate_spec_sha256(audit_spec)
    fixture_path = ROOT / audit_spec["development_panel"]["canonical_input_fixture"]
    assert _sha256(fixture_path) == audit_spec["development_panel"][
        "canonical_input_fixture_sha256"
    ]
    candidate_provenance_path = ROOT / "data/processed/context_v1_4_candidate/provenance.json"
    candidate_provenance = json.loads(candidate_provenance_path.read_text(encoding="utf-8"))
    assert candidate_provenance["candidate_spec_sha256"] == candidate_spec_sha256()
    assert candidate_provenance["development_lineage"]["pull_request_head_commit"] == (
        "ec0ef4c08b5aa10e534488bb7292cb2901626aee"
    )
    assert candidate_provenance["source_hashes"][
        "data/processed/context_v1_4_candidate/development_model_inputs.json"
    ] == _sha256(fixture_path)
    implementation_sources = {
        "candidate_builder": ROOT / "src/gippyrank/context_prior_v1_4_candidate.py",
        "positive_net_moderation": ROOT
        / "src/gippyrank/context_positive_net_moderation.py",
        "context_prior_v1_3": ROOT / "src/gippyrank/context_prior_v1_3.py",
        "transfer_manifest_validator": ROOT / "src/gippyrank/preseason_transfer.py",
        "context_prior_core": ROOT / "src/gippyrank/context_prior.py",
        "rank_distribution": ROOT / "src/gippyrank/preseason.py",
        "research_crossover": ROOT / "scripts/study_context_history_crossover.py",
        "dependency_lock": ROOT / "uv.lock",
    }
    assert candidate_provenance["implementation_sources_sha256"] == {
        name: _sha256(path) for name, path in implementation_sources.items()
    }
    assert candidate_provenance["development_panel_parity_sha256"] == _sha256(
        parity_path
    )
    assert candidate_provenance["candidate_reference_numerical_parity"] == parity[
        "candidate_reference_numerical_parity"
    ]
    assert "retained PR #159 reference outputs" in candidate_provenance[
        "development_panel_hash_scope"
    ]

    for relative, expected in audit_spec["development_panel"]["source_sha256"].items():
        assert _sha256(ROOT / relative) == expected
    for relative, expected in audit_spec["development_panel"]["artifact_sha256"].items():
        assert _sha256(ROOT / relative) == expected
    experiment_provenance = json.loads(
        (ROOT / "data/processed/context_positive_net_moderation/provenance.json")
        .read_text(encoding="utf-8")
    )
    assert experiment_provenance["script_sha256"] == audit_spec[
        "development_panel"
    ]["experiment_script_sha256"]

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    source_path = ROOT / "data/processed/context_history_crossover/team_season_prior_decomposition.csv"
    with source_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    retained_path = ROOT / "data/processed/context_history_crossover/hybrid_prior_results.csv"
    with retained_path.open(newline="", encoding="utf-8") as handle:
        retained_rows = [
            row for row in csv.DictReader(handle) if row["arm"] == "CC"
        ]
    source_by_key = {(int(row["season"]), row["team_id"]): row for row in source_rows}
    retained_by_key = {(int(row["season"]), row["team_id"]): row for row in retained_rows}
    diagnostics_path = ROOT / "data/processed/context_location_error_diagnostics/team_seasons.csv"
    with diagnostics_path.open(newline="", encoding="utf-8") as handle:
        diagnostic_rows = list(csv.DictReader(handle))
    diagnostic_by_key = {
        (int(row["season"]), row["team_id"]): row for row in diagnostic_rows
    }
    provenance_path = ROOT / "data/processed/context_history_crossover/provenance.json"
    base_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_panel: dict[str, int] = {}
    assert "retained PR #159 reference" in parity["reference_output_hash_scope"]

    max_panel_abs_difference = 0.0
    max_panel_rel_difference = 0.0
    for season in audit_spec["development_panel"]["seasons"]:
        season_fixture = fixture["seasons"][str(season)]
        model = _model_from_metadata(season_fixture["model_metadata"])
        instance = AnnualFittedInstance(**season_fixture["fitted_instance"])
        expected_model = base_provenance["models"][str(season)]
        model_sha256 = context_sha256_json(model.metadata())
        assert model_sha256 == expected_model["context_model_sha256"]
        assert instance.metadata() == expected_model["context_instance"]
        target_rows = {
            row["team_id"]: InferenceRow(
                season=int(row["season"]),
                subdivision=row["subdivision"],
                team_id=row["team_id"],
                team_name=row["team_name"],
                population=int(row["population"]),
                lag1_z=tuple(float(value) for value in row["lag1_z"]),
                lag_zs=tuple(tuple(values) for values in row["lag_zs"]),
                features=row["features"],
            )
            for row in season_fixture["fitted_rows"]
        }
        transfer_values_by_team = {
            row["team_id"]: {
                name: row["features"][name]
                for name in (
                    "transfer_in_prior_usage_sum",
                    "transfer_in_prior_defensive_impact_db_sum",
                    "transfer_in_prior_defensive_impact_db_available",
                )
            }
            for row in season_fixture["fitted_rows"]
        }
        target_team_ids = tuple(
            team_id for (source_season, team_id) in source_by_key if source_season == season
        )
        transfer_provenance = ContextTransferInputProvenance.research_only(
            season,
            feature_artifact_id=(
                f"development_model_inputs.json/season-{season}/transfer-features"
            ),
            transfer_feature_values_by_team=transfer_values_by_team,
            source_artifact_payload=transfer_values_by_team,
            target_team_ids=target_team_ids,
        )
        team_count = fitted_count = fallback_count = 0
        for (source_season, team_id), source in sorted(
            source_by_key.items(), key=lambda item: (item[0][0], item[0][1])
        ):
            if source_season != season:
                continue
            team_count += 1
            retained = retained_by_key[source_season, team_id]
            source_pmf = np.asarray(json.loads(retained["prior_pmf"]), dtype=float)
            if source["component_status"] == FITTED_STATUS:
                fitted_count += 1
                prior = Context13PriorInput.fitted(
                    model=model,
                    fitted_instance=instance,
                    inference_row=target_rows[team_id],
                    transfer_provenance=transfer_provenance,
                    prior_pmf=source_pmf,
                    expected_team_id=team_id,
                    expected_target_season=season,
                )
                diagnostic = diagnostic_by_key[source_season, team_id]
                source_locations = np.asarray(
                    json.loads(source["context_conditional_location_points"]),
                    dtype=float,
                )
                research_parts = parts_from_fitted_contributions(
                    diagnostic, source_locations
                )
                if research_parts.context_only_subtotal <= 0:
                    historical_reference = source_pmf.copy()
                else:
                    moderated_context = min(
                        research_parts.context_only_subtotal, 0.0
                    ) + 0.75 * max(research_parts.context_only_subtotal, 0.0)
                    reference_locations = source_locations + (
                        moderated_context - research_parts.context_only_subtotal
                    )
                    historical_reference = _legacy_normal_mixture_pmf(
                        reference_locations,
                        float(source["context_conditional_residual_scale"]),
                        int(source["target_population"]),
                    )
            else:
                assert source["component_status"] == COLD_START_STATUS
                fallback_count += 1
                fallback_source = Context13FallbackSource.from_research_fixture_csv(
                    retained_path,
                    target_season=season,
                    team_id=team_id,
                    population=int(source["target_population"]),
                )
                prior = Context13PriorInput.cold_start(
                    fitted_instance=instance,
                    transfer_provenance=transfer_provenance,
                    fallback_source=fallback_source,
                )
                historical_reference = source_pmf.copy()
            candidate = construct_candidate_prior(prior)
            if source["component_status"] == FITTED_STATUS:
                assert candidate.context_model_sha256 == model_sha256
            else:
                assert candidate.context_model_sha256 is None
                assert candidate.fallback_source is not None
                assert candidate.fallback_source.source_artifact_sha256 == _sha256(retained_path)
                assert candidate.fallback_source.cold_start_reason == (
                    "unspecified_in_pr159_reference"
                )
            assert candidate.source_prior_pmf_sha256 == sha256_json(source_pmf.tolist())
            if source["component_status"] == COLD_START_STATUS:
                assert candidate.pmf.tobytes() == source_pmf.tobytes()
            else:
                np.testing.assert_array_equal(prior.source_prior_pmf(), source_pmf)
            np.testing.assert_allclose(
                candidate.pmf,
                historical_reference,
                rtol=0,
                atol=PMF_PARITY_ABSOLUTE_TOLERANCE,
            )
            differences = np.abs(candidate.pmf - historical_reference)
            max_panel_abs_difference = max(
                max_panel_abs_difference, float(np.max(differences))
            )
            nonzero_reference = np.abs(historical_reference) > 0
            if np.any(nonzero_reference):
                max_panel_rel_difference = max(
                    max_panel_rel_difference,
                    float(
                        np.max(
                            differences[nonzero_reference]
                            / np.abs(historical_reference[nonzero_reference])
                        )
                    ),
                )

        expected_season = parity["seasons"][str(season)]
        assert len(expected_season["pr159_reference_team_pmf_hashes_sha256"]) == 64
        assert team_count == expected_season["team_seasons"]
        assert fitted_count == expected_season["fitted"]
        assert fallback_count == expected_season["cold_start_fallback"]
        expected_panel[str(season)] = team_count

    assert expected_panel == {"2022": 131, "2023": 133, "2024": 134, "2025": 136}
    numerical = parity["candidate_reference_numerical_parity"]
    assert max_panel_abs_difference <= numerical["max_absolute_difference_allowed"]
    assert max_panel_rel_difference <= numerical["max_relative_difference_allowed"]
    assert abs(
        max_panel_abs_difference - numerical["max_absolute_difference_observed"]
    ) <= numerical["max_absolute_difference_allowed"]
    assert abs(
        max_panel_rel_difference - numerical["max_relative_difference_observed"]
    ) <= numerical["max_relative_difference_allowed"]
    assert numerical["max_absolute_difference_observed"] <= numerical[
        "max_absolute_difference_allowed"
    ]
    assert numerical["max_relative_difference_observed"] <= numerical[
        "max_relative_difference_allowed"
    ]
