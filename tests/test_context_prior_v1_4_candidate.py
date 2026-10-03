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
    Context13FittedModelSource,
    ContextTransferInputProvenance,
    decompose_context13_location,
    fit_model_with_source,
    load_validated_committed_2026_reconstruction,
    load_validated_context13_fitted_model,
    validate_context13_fit_transfer_compatibility,
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
    load_validated_history_annual_artifact,
    sha256_json,
)
from gippyrank.preseason import (
    DirectRankModel,
    Preprocessor,
    TeamSeason,
    conditional_rank_mixture_pmf,
    rank_bin_edges,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/context_v1_4_candidate.json"
PMF_PARITY_ABSOLUTE_TOLERANCE = 1e-16


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def committed_context13_fit():
    annual = ROOT / "data/processed/preseason/context_v1_3/annual/2026"
    model, instance, source = load_validated_context13_fitted_model(
        annual / "fitted_model.json",
        annual / "fitted_instance.json",
    )
    return model, instance, source


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


def _research_fit_source(
    model: DirectRankModel, instance: AnnualFittedInstance
) -> Context13FittedModelSource:
    return Context13FittedModelSource.research_only(
        model,
        instance,
        fixture_id="context-v1-4-unit-test-fixture",
    )


def _fitted_input(
    *, coach_tenure: float = 4.0, provenance: ContextTransferInputProvenance | None = None
) -> Context13PriorInput:
    model = _model()
    instance = _instance()
    row = _row(coach_tenure=coach_tenure)
    return Context13PriorInput.fitted(
        model=model,
        fitted_instance=instance,
        fitted_model_source=_research_fit_source(model, instance),
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
            fitted_model_source=prior.fitted_model_source,
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
            fitted_model_source=_research_fit_source(_model(), _instance()),
            inference_row=row,
            transfer_provenance=_provenance(),
            prior_pmf=pmf,
            expected_team_id=row.team_id,
            expected_target_season=2026,
        )


def test_hand_built_model_without_authoritative_fit_source_fails() -> None:
    model = _model()
    instance = _instance()
    row = _row()
    with pytest.raises(ValueError, match="require authoritative fit provenance"):
        Context13PriorInput.fitted(
            model=model,
            fitted_instance=instance,
            inference_row=row,
            transfer_provenance=_provenance(),
            prior_pmf=model.pmf(row.features, np.asarray(row.lag1_z), row.population),
            expected_team_id=row.team_id,
            expected_target_season=row.season,
        )


def test_context13_fit_source_cannot_be_forged_from_plausible_fields() -> None:
    model = _model()
    instance = _instance()
    with pytest.raises(TypeError, match="canonical fit or loader"):
        Context13FittedModelSource(
            model_family="context_prior",
            spec_version="1.3",
            target_season=2026,
            trained_through_season=2025,
            model_metadata_sha256=context_sha256_json(model.metadata()),
            fitted_instance_identity_sha256=context_sha256_json(instance.metadata()),
            frozen_model_spec_identity_sha256="a" * 64,
            training_corpus_input_sha256="b" * 64,
            training_row_count=2744,
            training_corpus_source_identity_sha256=None,
            training_corpus_provenance_class="canonical_reproducible",
            provenance_class="canonical_context13_fit",
        )


def test_context13_fit_source_binds_model_instance_and_training_corpus(
    committed_context13_fit,
) -> None:
    model, instance, source = committed_context13_fit
    assert source.model_metadata_sha256 == context_sha256_json(model.metadata())
    with pytest.raises(ValueError, match="training-corpus identity"):
        source.validate_model(model, instance, training_rows=[])
    research_source = _research_fit_source(model, instance)
    assert research_source.model_metadata_sha256 == context_sha256_json(model.metadata())
    with pytest.raises(ValueError, match="training-corpus identity"):
        research_source.validate_model(
            model,
            instance,
            allow_research_only=True,
            training_rows=[],
        )
    wrong_instance = replace(instance, trained_through_season=2024)
    with pytest.raises(ValueError, match="annual instance"):
        source.validate_instance(wrong_instance)
    with pytest.raises(ValueError, match="not promotable"):
        _research_fit_source(model, instance).validate_model(model, instance)
    target_row = TeamSeason(
        season=instance.target_season,
        subdivision="fbs",
        team_id="target-season-row",
        team_name="Target Season Row",
        population=1,
        lag1_z=np.asarray([0.0]),
        target_z=np.asarray([0.0]),
        target_ranks=np.asarray([1]),
        features={name: 0.0 for name in MODEL_FEATURE_NAMES},
    )
    with pytest.raises(ValueError, match="includes rows beyond its cutoff"):
        fit_model_with_source(
            [target_row],
            target_season=instance.target_season,
            trained_through_season=instance.trained_through_season,
        )


@pytest.mark.parametrize(
    ("fit_class", "transfer_class", "allowed"),
    [
        ("canonical_context13_fit", PRODUCTION_TRANSFER_PROVENANCE, True),
        ("canonical_context13_fit", RETROSPECTIVE_2026_PROVENANCE, True),
        ("legacy_attested_context13_fit", RETROSPECTIVE_2026_PROVENANCE, True),
        ("research_only", RETROSPECTIVE_RESEARCH_PROVENANCE, True),
        ("research_only", PRODUCTION_TRANSFER_PROVENANCE, False),
        ("research_only", RETROSPECTIVE_2026_PROVENANCE, False),
        ("canonical_context13_fit", RETROSPECTIVE_RESEARCH_PROVENANCE, False),
        ("legacy_attested_context13_fit", PRODUCTION_TRANSFER_PROVENANCE, False),
    ],
)
def test_fit_transfer_provenance_compatibility_matrix(
    fit_class: str, transfer_class: str, allowed: bool
) -> None:
    if allowed:
        validate_context13_fit_transfer_compatibility(fit_class, transfer_class)
    else:
        with pytest.raises(ValueError, match="provenance classes are incompatible"):
            validate_context13_fit_transfer_compatibility(fit_class, transfer_class)


def test_research_fit_cannot_pair_with_authoritative_2026_transfer_for_fitted_or_cold() -> None:
    reconstruction = ROOT / "data/processed/preseason/context_v1_3_2026_reconstruction"
    _, transfer_source = load_validated_committed_2026_reconstruction(reconstruction)
    model = _model()
    instance = _instance()
    research_source = _research_fit_source(model, instance)
    row = _row(team_id="16")
    with pytest.raises(ValueError, match="provenance classes are incompatible"):
        Context13PriorInput.fitted(
            model=model,
            fitted_instance=instance,
            fitted_model_source=research_source,
            inference_row=row,
            transfer_provenance=transfer_source,
            prior_pmf=model.pmf(row.features, np.asarray(row.lag1_z), row.population),
            expected_team_id=row.team_id,
            expected_target_season=row.season,
        )

    history_dir = ROOT / "data/processed/preseason/history/annual/2026"
    history_source = load_validated_history_annual_artifact(
        history_dir / "predictions.csv",
        history_dir / "fitted_instance.json",
        target_season=2026,
        trained_through_season=2025,
    )
    fallback = Context13FallbackSource.from_history_annual_source(
        source=history_source,
        target_season=2026,
        trained_through_season=2025,
        team_id="16",
        team_name="Sacramento State",
        population=138,
        cold_start_reason="fcs_to_fbs_transition",
    )
    with pytest.raises(ValueError, match="provenance classes are incompatible"):
        Context13PriorInput.cold_start(
            fitted_instance=instance,
            fitted_model_source=research_source,
            transfer_provenance=transfer_source,
            fallback_source=fallback,
        )
@pytest.mark.parametrize("forbidden_coefficient", ["numeric", "missingness"])
def test_context_only_scale_coefficients_must_remain_zero(
    forbidden_coefficient: str,
) -> None:
    model = _model()
    instance = _instance()
    fit_source = _research_fit_source(model, instance)
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
            fitted_instance=instance,
            fitted_model_source=fit_source,
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
            fitted_model_source=_research_fit_source(model, instance),
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
            fitted_model_source=_research_fit_source(model, instance),
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
            fitted_model_source=_research_fit_source(model, _instance()),
            inference_row=changed_row,
            transfer_provenance=provenance,
            prior_pmf=changed_pmf,
            expected_team_id=changed_row.team_id,
            expected_target_season=changed_row.season,
        )
def test_nonpositive_fitted_and_canonical_cold_start_pmfs_remain_bit_identical() -> None:
    negative = construct_candidate_prior(_fitted_input(coach_tenure=-4.0))
    negative_input = _fitted_input(coach_tenure=-4.0)
    np.testing.assert_array_equal(negative.pmf, negative_input.source_prior_pmf())
    assert not hasattr(Context13FallbackSource, "from_generic_rank_prior")
    history_dir = ROOT / "data/processed/preseason/history/annual/2026"
    history_source = load_validated_history_annual_artifact(
        history_dir / "predictions.csv",
        history_dir / "fitted_instance.json",
        target_season=2026,
        trained_through_season=2025,
    )
    fallback = Context13FallbackSource.from_history_annual_source(
        source=history_source,
        target_season=2026,
        trained_through_season=2025,
        team_id="16",
        team_name="Sacramento State",
        population=138,
        cold_start_reason="fcs_to_fbs_transition",
    )
    cold_source = fallback.pmf.copy()
    model = _model()
    instance = _instance()
    cold = Context13PriorInput.cold_start(
        fitted_instance=instance,
        fitted_model_source=_research_fit_source(model, instance),
        transfer_provenance=_provenance(team_id="16", population=138),
        fallback_source=fallback,
    )
    cold_result = construct_candidate_prior(cold)
    assert cold_result.pmf.tobytes() == cold_source.tobytes()
    assert cold_result.component_status == COLD_START_STATUS
    assert cold_result.context_model_sha256 is None
    assert cold_result.fallback_source is not None
    assert cold_result.fallback_source.pmf_sha256 == sha256_json(cold_source.tolist())
    assert cold_result.fallback_source.trained_through_season == 2025
    assert cold_result.fallback_source.fitted_instance_identity_sha256 == (
        history_source.fitted_instance_identity_sha256
    )
    assert cold_result.fitted_model_source.model_metadata_sha256 == (
        context_sha256_json(model.metadata())
    )
    assert cold_result.fitted_model_source.to_metadata()["provenance_class"] == "research_only"
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
            fitted_model_source=_research_fit_source(_model(), _instance()),
            transfer_provenance=_provenance(team_id="cold-team", population=3),
            prior_pmf=np.array([0.2, 0.3, 0.5]),
        )
    with pytest.raises(ValueError, match="read-only"):
        cold_result.pmf[0] = 0.0


def test_canonical_history_cold_start_is_loaded_and_bound_without_context_model(
    tmp_path: Path,
) -> None:
    history_dir = ROOT / "data/processed/preseason/history/annual/2026"
    artifact_path = history_dir / "predictions.csv"
    source = load_validated_history_annual_artifact(
        artifact_path,
        history_dir / "fitted_instance.json",
        target_season=2026,
        trained_through_season=2025,
    )
    fallback = Context13FallbackSource.from_history_annual_source(
        source=source,
        target_season=2026,
        trained_through_season=2025,
        team_id="16",
        team_name="Sacramento State",
        population=138,
        cold_start_reason="fcs_to_fbs_transition",
    )
    context_model = _model()
    context_instance = _instance()
    prior = Context13PriorInput.cold_start(
        fitted_instance=context_instance,
        fitted_model_source=_research_fit_source(context_model, context_instance),
        transfer_provenance=_provenance(team_id="16", population=138),
        fallback_source=fallback,
    )
    candidate = construct_candidate_prior(prior)
    assert candidate.context_model_sha256 is None
    assert candidate.pmf.tobytes() == fallback.pmf.tobytes()
    assert candidate.fallback_source is not None
    assert candidate.fallback_source.source_artifact_sha256 == _sha256(artifact_path)
    assert candidate.fallback_source.model_identity == "history_prior/1.1"
    assert candidate.fallback_source.trained_through_season == 2025
    assert candidate.fallback_source.source_model_metadata_sha256 == (
        source.model_metadata_sha256
    )
    with pytest.raises(ValueError, match="no row for the requested team"):
        Context13FallbackSource.from_history_annual_source(
            source=source,
            target_season=2026,
            trained_through_season=2025,
            team_id="unknown-team",
            team_name="Unknown",
            population=138,
            cold_start_reason="fcs_to_fbs_transition",
        )
    with pytest.raises(ValueError, match="team identity"):
        Context13FallbackSource.from_history_annual_source(
            source=source,
            target_season=2026,
            trained_through_season=2025,
            team_id="16",
            team_name="Wrong Name",
            population=138,
            cold_start_reason="fcs_to_fbs_transition",
        )
    with pytest.raises(ValueError, match="method does not match"):
        Context13FallbackSource.from_history_annual_source(
            source=source,
            target_season=2026,
            trained_through_season=2025,
            team_id="16",
            team_name="Sacramento State",
            population=138,
            cold_start_reason="no_prior_rank_distribution",
        )

    canonical_looking = (
        tmp_path
        / "data/processed/preseason/history/annual/2026/predictions.csv"
    )
    canonical_looking.parent.mkdir(parents=True)
    canonical_looking.write_text(
        "season,subdivision,team_id,team_name,model_family,spec_version,"
        "trained_through_season,pmf,prior_method\n"
        '2026,fbs,16,Sacramento State,history_prior,1.1,2025,"[1.0]",'
        "learned_fcs_to_fbs_transition\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="retained canonical annual artifact"):
        load_validated_history_annual_artifact(
            canonical_looking,
            history_dir / "fitted_instance.json",
            target_season=2026,
            trained_through_season=2025,
        )

    changed_bytes = tmp_path / "history_predictions_crlf.csv"
    changed_bytes.write_bytes(artifact_path.read_bytes().replace(b"\n", b"\r\n"))
    changed_source = load_validated_history_annual_artifact(
        changed_bytes,
        history_dir / "fitted_instance.json",
        target_season=2026,
        trained_through_season=2025,
    )
    assert changed_source.prediction_artifact_sha256 != source.prediction_artifact_sha256
    assert changed_source.source_identity_sha256 != source.source_identity_sha256
    with pytest.raises(ValueError, match="no authoritative History annual source"):
        load_validated_history_annual_artifact(
            artifact_path,
            history_dir / "fitted_instance.json",
            target_season=2026,
            trained_through_season=2024,
        )
    mismatched_instance = tmp_path / "fitted_instance_wrong_cutoff.json"
    fitted_payload = json.loads(
        (history_dir / "fitted_instance.json").read_text(encoding="utf-8")
    )
    fitted_payload["trained_through_season"] = 2024
    mismatched_instance.write_text(json.dumps(fitted_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fitted instance does not match"):
        load_validated_history_annual_artifact(
            artifact_path,
            mismatched_instance,
            target_season=2026,
            trained_through_season=2025,
        )


def test_committed_2026_context_lineage_loads_without_relabeling_or_side_effects(
    committed_context13_fit,
) -> None:
    model, instance, fit_source = committed_context13_fit
    annual = ROOT / "data/processed/preseason/context_v1_3/annual/2026"
    reconstruction = ROOT / "data/processed/preseason/context_v1_3_2026_reconstruction"
    transfer_provenance = json.loads(
        (reconstruction / "feature_provenance.json").read_text(encoding="utf-8")
    )
    assert transfer_provenance["provenance_class"] == "retrospective_2026_reconstruction"
    model_artifact = json.loads((annual / "fitted_model.json").read_text(encoding="utf-8"))
    assert model_artifact["transfer_provenance_class"] == "retrospective_2026_reconstruction"
    with (annual / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    assert {row["transfer_provenance_class"] for row in prediction_rows} == {
        "retrospective_2026_reconstruction"
    }
    assert fit_source.provenance_class == "legacy_attested_context13_fit"
    assert fit_source.training_corpus_provenance_class == "legacy_attested"
    assert fit_source.training_corpus_source_identity_sha256 is None
    assert fit_source.provenance_schema_version == 2
    assert fit_source.reproducibility_level == "retained_legacy_attestation"
    assert model_artifact.get("artifact_schema_version") is None
    fit_sidecar = json.loads((annual / "fitted_model_source.json").read_text(encoding="utf-8"))
    assert fit_sidecar["provenance_schema_version"] == 2
    assert fit_sidecar["provenance_class"] == "legacy_attested_context13_fit"
    assert fit_source.model_metadata_sha256 == context_sha256_json(model.metadata())
    assert fit_source.training_corpus_input_sha256
    assert fit_source.frozen_model_spec_identity_sha256
    assert instance.trained_through_season == 2025

    historical_rank_distribution = ROOT / "data/processed/modeling/team_season_rank_distributions.csv"
    assert not historical_rank_distribution.exists()
    import sys

    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import build_preseason_context_prior_v1_3 as context_builder

    with pytest.raises(FileNotFoundError, match="team_season_rank_distributions.csv"):
        context_builder.load_context13_training_corpus(
            target_season=2026,
            trained_through_season=2025,
        )
    with pytest.raises(ValueError, match="authoritative training-corpus source"):
        load_validated_context13_fitted_model(
            annual / "fitted_model.json",
            annual / "fitted_instance.json",
            training_rows=[],
        )

    feature_rows, transfer_source = load_validated_committed_2026_reconstruction(
        reconstruction
    )
    assert len(feature_rows) == 138
    assert transfer_source.provenance_class == "retrospective_2026_reconstruction"

    history_dir = ROOT / "data/processed/preseason/history/annual/2026"
    history_source = load_validated_history_annual_artifact(
        history_dir / "predictions.csv",
        history_dir / "fitted_instance.json",
        target_season=2026,
        trained_through_season=2025,
    )
    fallback = Context13FallbackSource.from_history_annual_source(
        source=history_source,
        target_season=2026,
        trained_through_season=2025,
        team_id="16",
        team_name="Sacramento State",
        population=138,
        cold_start_reason="fcs_to_fbs_transition",
    )
    before = {
        "context_predictions": _sha256(annual / "predictions.csv"),
        "history_predictions": _sha256(history_dir / "predictions.csv"),
        "published_context_predictions": _sha256(
            ROOT / "data/processed/preseason/context_v1_3/annual/2026/predictions.csv"
        ),
    }
    candidate = construct_candidate_prior(
        Context13PriorInput.cold_start(
            fitted_instance=instance,
            fitted_model_source=fit_source,
            transfer_provenance=transfer_source,
            fallback_source=fallback,
        )
    )
    artifact = json.loads(candidate.artifact_bytes())
    assert artifact["artifact_schema_version"] == 2
    assert artifact["transfer_input_provenance"]["provenance_class"] == (
        "retrospective_2026_reconstruction"
    )
    assert artifact["transfer_input_provenance"]["source_identity_sha256"] == (
        transfer_source.source_identity_sha256
    )
    assert artifact["context_fit_provenance"] == fit_source.to_metadata()
    assert artifact["context_fit_provenance"]["provenance_class"] == (
        "legacy_attested_context13_fit"
    )
    assert artifact["context_fit_provenance"]["training_row_count"] == 2744
    assert artifact["fallback_source"]["source_model_metadata_sha256"] == (
        history_source.model_metadata_sha256
    )
    assert candidate.context_model_sha256 is None
    after = {
        "context_predictions": _sha256(annual / "predictions.csv"),
        "history_predictions": _sha256(history_dir / "predictions.csv"),
        "published_context_predictions": _sha256(
            ROOT / "data/processed/preseason/context_v1_3/annual/2026/predictions.csv"
        ),
    }
    assert after == before
    assert before["context_predictions"] == (
        "2ef3cc2e5249271c5de2fca471b862b8762eee66b40aefc9a679abf2a45ef1f5"
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
    assert artifact["context_fit_provenance_class"] == "research_only"
    assert artifact["context_fit_reproducibility_level"] == "research_only"
    assert artifact["training_corpus_source_identity_sha256"] is None
    assert artifact["training_corpus_row_count"] == 0
    assert artifact["semantic_model_spec_sha256"] == (
        prior.fitted_model_source.frozen_model_spec_identity_sha256
    )
    assert "diagnostic_paths" not in artifact["transfer_input_provenance"]
    assert artifact["decomposition_sha256"]


def test_candidate_construction_does_not_mutate_production_artifacts() -> None:
    protected_directories = (
        ROOT / "data/processed/preseason/context_v1_3/annual/2026",
        ROOT / "data/processed/preseason/context_v1_3_2026_reconstruction",
        ROOT / "data/processed/preseason/context_v1_3_candidate",
        ROOT / "data/processed/preseason/history/annual/2026",
    )
    protected_files = (
        ROOT / "data/processed/preseason/history/predictions.csv",
        ROOT / "data/processed/context_history_crossover/hybrid_prior_results.csv",
        ROOT / "data/processed/context_positive_net_moderation/team_season_results.csv",
        ROOT / "site/data/snapshots/2026-preseason-context-v1.3.json",
        ROOT / "site/data/distributions/2026-preseason-context-v1.3.json",
    )
    protected = tuple(
        path
        for directory in protected_directories
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ) + protected_files
    assert all(path.is_file() for path in protected)
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
        fit_source = _research_fit_source(model, instance)
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
                    fitted_model_source=fit_source,
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
                    fitted_model_source=fit_source,
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
                assert candidate.fallback_source.research_fixture is True
                assert candidate.fallback_source.metadata()["source_identity_sha256"]
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
