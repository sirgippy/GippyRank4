"""Freeze, lineage, and parity checks for the Context 1.4 research candidate."""

from __future__ import annotations

import csv
import hashlib
import json
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
    RETROSPECTIVE_2026_PROVENANCE,
    RETROSPECTIVE_RESEARCH_PROVENANCE,
    ContextInputProvenance,
    decompose_context13_location,
)
from gippyrank.context_prior_v1_3 import (
    sha256_json as context_sha256_json,
)
from gippyrank.context_prior_v1_4_candidate import (
    COLD_START_STATUS,
    FITTED_STATUS,
    Context13PriorInput,
    Context14CandidatePrior,
    candidate_spec_sha256,
    construct_candidate_prior,
    load_candidate_spec,
    sha256_json,
)
from gippyrank.preseason import (
    DirectRankModel,
    Preprocessor,
    conditional_rank_mixture_pmf,
    rank_bin_edges,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/context_v1_4_candidate.json"


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
    season: int = 2026, provenance_class: str = RETROSPECTIVE_2026_PROVENANCE
) -> ContextInputProvenance:
    metadata = {
        "target_season": season,
        "provenance_class": provenance_class,
        "source_snapshot_sha256": ["1" * 64, "2" * 64],
        "known_absence_of_archived_august_15_transfer_snapshot": (
            provenance_class == RETROSPECTIVE_2026_PROVENANCE
        ),
    }
    if provenance_class == RETROSPECTIVE_RESEARCH_PROVENANCE:
        return ContextInputProvenance.from_research_metadata(season, metadata)
    return ContextInputProvenance.from_validated_metadata(season, metadata)


def _fitted_input(
    *, coach_tenure: float = 4.0, provenance: ContextInputProvenance | None = None
) -> Context13PriorInput:
    model = _model()
    row = _row(coach_tenure=coach_tenure)
    return Context13PriorInput.fitted(
        model=model,
        fitted_instance=_instance(),
        inference_row=row,
        input_provenance=provenance or _provenance(),
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
    assert candidate.context_model_sha256 == context_sha256_json(prior.fitted_model.metadata())
    assert candidate.context_model_sha256 != "a" * 64
    with pytest.raises(TypeError, match="created by construct_candidate_prior"):
        Context14CandidatePrior(**candidate.__dict__)
    with pytest.raises(TypeError):
        Context13PriorInput.fitted(
            model=prior.fitted_model,
            fitted_instance=prior.fitted_instance,
            inference_row=prior.inference_row,
            input_provenance=prior.input_provenance,
            prior_pmf=prior.source_prior_pmf(),
            expected_team_id=prior.team_id,
            expected_target_season=prior.fitted_instance.target_season,
            context_model_sha256="a" * 64,
        )


def test_wrong_context_model_spec_or_annual_identity_fails_closed() -> None:
    model = _model()
    row = _row()
    pmf = model.pmf(row.features, np.asarray(row.lag1_z), row.population)
    with pytest.raises(ValueError, match="rolling-origin Context 1.3 fit"):
        Context13PriorInput.fitted(
            model=model,
            fitted_instance=_instance(version="1.2"),
            inference_row=row,
            input_provenance=_provenance(),
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
            input_provenance=_provenance(),
            prior_pmf=pmf,
            expected_team_id=row.team_id,
            expected_target_season=2026,
        )


def test_canonical_decomposition_binds_team_season_and_source() -> None:
    model = _model()
    instance = _instance()
    provenance = _provenance()
    valid = decompose_context13_location(model, instance, _row(), provenance)
    assert valid.team_id == "team-1"
    assert valid.target_season == 2026
    assert valid.context_model_sha256 == context_sha256_json(model.metadata())
    assert valid.input_provenance.provenance_class == RETROSPECTIVE_2026_PROVENANCE

    with pytest.raises(ValueError, match="team input and model seasons differ"):
        decompose_context13_location(model, instance, _row(season=2025), provenance)
    with pytest.raises(ValueError, match="input provenance does not match"):
        Context13PriorInput.fitted(
            model=model,
            fitted_instance=instance,
            inference_row=_row(),
            input_provenance=_provenance(season=2025),
            prior_pmf=model.pmf(_row().features, np.asarray(_row().lag1_z), 8),
            expected_team_id="team-1",
            expected_target_season=2026,
        )
    with pytest.raises(ValueError, match="source team or season"):
        Context13PriorInput.fitted(
            model=model,
            fitted_instance=instance,
            inference_row=_row(),
            input_provenance=provenance,
            prior_pmf=model.pmf(_row().features, np.asarray(_row().lag1_z), 8),
            expected_team_id="different-team",
            expected_target_season=2026,
        )
    with pytest.raises(ValueError, match="team identity must be non-empty"):
        decompose_context13_location(
            model, instance, _row(team_id=""), provenance
        )
    with pytest.raises(ValueError, match="unsupported Context input provenance class"):
        ContextInputProvenance(2026, "archived_august_15_snapshot", "a" * 64)


def test_nonpositive_fitted_and_cold_start_pmfs_remain_bit_identical() -> None:
    negative = construct_candidate_prior(_fitted_input(coach_tenure=-4.0))
    negative_input = _fitted_input(coach_tenure=-4.0)
    np.testing.assert_array_equal(negative.pmf, negative_input.source_prior_pmf())

    cold_source = np.array([0.2, 0.3, 0.5])
    cold = Context13PriorInput.cold_start(
        model=_model(),
        fitted_instance=_instance(),
        input_provenance=_provenance(),
        team_id="cold-team",
        team_name="Cold Start University",
        population=3,
        prior_pmf=cold_source,
        reason="no_lag1_rank_history",
    )
    cold_result = construct_candidate_prior(cold)
    assert cold_result.pmf.tobytes() == cold_source.tobytes()
    assert cold_result.component_status == COLD_START_STATUS


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
    assert artifact["source_context13_pmf_sha256"] == sha256_json(
        prior.source_prior_pmf().tolist()
    )
    assert artifact["candidate_semantics_sha256"] == candidate_spec_sha256()
    assert artifact["context_model_sha256"] == context_sha256_json(
        prior.fitted_model.metadata()
    )
    assert artifact["target_season"] == 2026
    assert artifact["trained_through_season"] == 2025
    assert artifact["team_id"] == "team-1"
    assert artifact["input_provenance"]["provenance_class"] == (
        RETROSPECTIVE_2026_PROVENANCE
    )
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

    for season in audit_spec["development_panel"]["seasons"]:
        season_fixture = fixture["seasons"][str(season)]
        model = _model_from_metadata(season_fixture["model_metadata"])
        instance = AnnualFittedInstance(**season_fixture["fitted_instance"])
        expected_model = base_provenance["models"][str(season)]
        model_sha256 = context_sha256_json(model.metadata())
        assert model_sha256 == expected_model["context_model_sha256"]
        assert instance.metadata() == expected_model["context_instance"]
        input_provenance = ContextInputProvenance.from_research_metadata(
            season, season_fixture["input_provenance_metadata"]
        )
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
                    input_provenance=input_provenance,
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
                prior = Context13PriorInput.cold_start(
                    model=model,
                    fitted_instance=instance,
                    input_provenance=input_provenance,
                    team_id=team_id,
                    team_name=source["team_name"],
                    population=int(source["target_population"]),
                    prior_pmf=source_pmf,
                    reason=source.get("cold_start_reason") or "no_fitted_lag1_input",
                )
                historical_reference = source_pmf.copy()
            candidate = construct_candidate_prior(prior)
            assert candidate.context_model_sha256 == model_sha256
            assert candidate.source_context13_pmf_sha256 == sha256_json(source_pmf.tolist())
            if source["component_status"] == COLD_START_STATUS:
                assert candidate.pmf.tobytes() == source_pmf.tobytes()
            else:
                np.testing.assert_array_equal(prior.source_prior_pmf(), source_pmf)
            np.testing.assert_array_equal(candidate.pmf, historical_reference)

        expected_season = parity["seasons"][str(season)]
        assert team_count == expected_season["team_seasons"]
        assert fitted_count == expected_season["fitted"]
        assert fallback_count == expected_season["cold_start_fallback"]
        expected_panel[str(season)] = team_count

    assert expected_panel == {"2022": 131, "2023": 133, "2024": 134, "2025": 136}
