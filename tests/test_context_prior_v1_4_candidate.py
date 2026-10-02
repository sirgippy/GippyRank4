"""Freeze and parity checks for the Context 1.4 research candidate."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from gippyrank.context_positive_net_moderation import (
    LocationParts,
    moderate_positive_net,
    moderated_location_points,
)
from gippyrank.context_prior import AnnualFittedInstance
from gippyrank.context_prior_v1_4_candidate import (
    COLD_START_STATUS,
    FITTED_STATUS,
    Context13PriorInput,
    candidate_spec_sha256,
    construct_candidate_prior,
    load_candidate_spec,
    normal_mixture_pmf,
    sha256_json,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_context_positive_net_moderation as issue_158


def _instance(
    *,
    target_season: int = 2026,
    trained_through_season: int = 2025,
    spec_version: str = "1.3",
) -> AnnualFittedInstance:
    return AnnualFittedInstance(
        "context_prior", spec_version, trained_through_season, target_season
    )


def _fitted_input(context_only: float = 4.0) -> Context13PriorInput:
    parts = LocationParts(
        intercept=1.0,
        history_derived_subtotal=2.0,
        context_only_subtotal=context_only,
        conditional_location_points=np.array([2.0 + context_only, 4.0 + context_only]),
    )
    base_pmf = normal_mixture_pmf(parts.conditional_location_points, 0.8, 8)
    return Context13PriorInput(
        fitted_instance=_instance(),
        context_model_sha256="a" * 64,
        team_id="team-1",
        component_status=FITTED_STATUS,
        population=8,
        context_prior_pmf=base_pmf,
        location_parts=parts,
        residual_scale=0.8,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_spec_is_frozen_research_only_and_hash_sensitive() -> None:
    spec = load_candidate_spec()
    assert spec["alpha"] == 0.75
    assert spec["model_family"] == "context_prior"
    assert spec["spec_version"] == "1.4-candidate"
    assert spec["base_model"]["spec_version"] == "1.3"
    assert spec["status"] == "frozen research candidate"
    assert spec["production"] is False
    assert spec["validation_status"] == "not yet holdout validated"
    assert (
        "alpha"
        not in __import__("inspect").signature(construct_candidate_prior).parameters
    )

    changed = copy.deepcopy(spec)
    changed["alpha"] = 0.70
    assert candidate_spec_sha256(changed) != candidate_spec_sha256(spec)


@pytest.mark.parametrize(
    ("context_only", "expected"),
    [(-2.0, -2.0), (0.0, 0.0), (4.0, 3.0)],
)
def test_positive_only_semantics_and_location_reconstruction(
    context_only: float, expected: float
) -> None:
    assert moderate_positive_net(context_only, 0.75) == expected
    parts = LocationParts(
        intercept=1.0,
        history_derived_subtotal=2.0,
        context_only_subtotal=context_only,
        conditional_location_points=np.array(
            [1.0 + 2.0 + context_only - 1.0, 1.0 + 2.0 + context_only + 1.0]
        ),
    )
    points = moderated_location_points(parts, 0.75)
    assert points.mean() == pytest.approx(1.0 + 2.0 + expected, abs=1e-12)
    np.testing.assert_array_equal(
        points - points.mean(),
        parts.conditional_location_points - parts.conditional_location_points.mean(),
    )


@pytest.mark.parametrize("context_only", [-3.0, 0.0])
def test_nonpositive_fitted_term_keeps_context_prior_pmf_bit_identical(
    context_only: float,
) -> None:
    prior = _fitted_input(context_only)
    original_bytes = prior.context_prior_pmf.tobytes()
    candidate = construct_candidate_prior(prior)
    assert candidate.pmf.tobytes() == original_bytes
    assert candidate.pmf is not prior.context_prior_pmf


def test_positive_term_rebuilds_candidate_and_serialization_is_deterministic() -> None:
    prior = _fitted_input(4.0)
    candidate = construct_candidate_prior(prior)
    expected_locations = prior.location_parts.conditional_location_points - 1.0
    expected = normal_mixture_pmf(expected_locations, 0.8, 8)
    np.testing.assert_array_equal(candidate.pmf, expected)
    assert candidate.candidate_spec_sha256 == candidate_spec_sha256()
    assert (
        candidate.artifact_bytes() == construct_candidate_prior(prior).artifact_bytes()
    )


def test_cold_start_fallback_is_copied_without_synthesized_terms() -> None:
    fallback_pmf = np.array([0.2, 0.3, 0.5])
    prior = Context13PriorInput(
        fitted_instance=_instance(),
        context_model_sha256="b" * 64,
        team_id="cold-start",
        component_status=COLD_START_STATUS,
        population=3,
        context_prior_pmf=fallback_pmf,
        location_parts=None,
        residual_scale=None,
    )
    candidate = construct_candidate_prior(prior)
    assert candidate.pmf.tobytes() == fallback_pmf.tobytes()
    with pytest.raises(ValueError, match="cannot synthesize fitted"):
        Context13PriorInput(
            fitted_instance=_instance(),
            context_model_sha256="b" * 64,
            team_id="cold-start",
            component_status=COLD_START_STATUS,
            population=3,
            context_prior_pmf=fallback_pmf,
            location_parts=LocationParts(0.0, 0.0, 0.0, np.array([0.0])),
            residual_scale=None,
        )


@pytest.mark.parametrize(
    ("instance", "message"),
    [
        (_instance(trained_through_season=2026), "not valid rolling-origin"),
        (_instance(spec_version="1.2"), "Context 1.3 fit"),
    ],
)
def test_candidate_fails_closed_for_non_rolling_or_wrong_version_fits(
    instance: AnnualFittedInstance, message: str
) -> None:
    prior = _fitted_input()
    invalid = Context13PriorInput(
        fitted_instance=instance,
        context_model_sha256=prior.context_model_sha256,
        team_id=prior.team_id,
        component_status=prior.component_status,
        population=prior.population,
        context_prior_pmf=prior.context_prior_pmf,
        location_parts=prior.location_parts,
        residual_scale=prior.residual_scale,
    )
    with pytest.raises(ValueError, match=message):
        construct_candidate_prior(invalid)


def test_committed_development_panel_reproduces_pr_159_alpha_075_pmfs() -> None:
    spec = load_candidate_spec()
    parity_path = (
        ROOT / "data/processed/context_v1_4_candidate/development_panel_parity.json"
    )
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    assert parity["candidate_spec_sha256"] == candidate_spec_sha256(spec)

    experiment = Path("data/processed/context_positive_net_moderation")
    source_hashes = spec["development_panel"]["source_sha256"]
    for relative, expected in source_hashes.items():
        assert _sha256(ROOT / relative) == expected
    for relative, expected in spec["development_panel"]["artifact_sha256"].items():
        assert _sha256(ROOT / relative) == expected
    experiment_provenance = json.loads(
        (ROOT / experiment / "provenance.json").read_text(encoding="utf-8")
    )
    assert (
        experiment_provenance["script_sha256"]
        == spec["development_panel"]["experiment_script_sha256"]
    )

    base = ROOT / "data/processed/context_history_crossover"
    diagnostic_root = ROOT / "data/processed/context_location_error_diagnostics"
    decomposition = issue_158.checked_index(
        issue_158.read_csv(base / "team_season_prior_decomposition.csv"),
        ("season", "team_id"),
    )
    diagnostics = issue_158.checked_index(
        issue_158.read_csv(diagnostic_root / "team_seasons.csv"),
        ("season", "team_id"),
    )
    retained_priors = issue_158.checked_index(
        [
            row
            for row in issue_158.read_csv(base / "hybrid_prior_results.csv")
            if row["arm"] == "CC"
        ],
        ("season", "team_id"),
    )
    base_provenance = json.loads((base / "provenance.json").read_text())

    panel = {}
    for season in spec["development_panel"]["seasons"]:
        season_hashes = []
        fitted_count = fallback_count = 0
        for key in sorted(
            (key for key in decomposition if int(key[0]) == season),
            key=lambda item: item[1],
        ):
            season_text, team_id = key
            source_row = decomposition[key]
            diagnostic_row = diagnostics[key]
            retained = retained_priors[key]
            status = source_row["component_status"]
            assert status == diagnostic_row["component_status"]
            retained_pmf = np.asarray(json.loads(retained["prior_pmf"]), dtype=float)
            assert (
                issue_158.crossover.sha256_json(retained_pmf.tolist())
                == retained["prior_pmf_sha256"]
            )
            fitted_model = base_provenance["models"][season_text]
            instance = AnnualFittedInstance(**fitted_model["context_instance"])
            if status == FITTED_STATUS:
                fitted_count += 1
                locations = np.asarray(
                    json.loads(source_row["context_conditional_location_points"]),
                    dtype=float,
                )
                parts = issue_158.parts_from_fitted_contributions(
                    diagnostic_row, locations
                )
                issue_158.require_rolling_origin(
                    season,
                    int(diagnostic_row["context_training_cutoff"]),
                    int(diagnostic_row["history_training_cutoff"]),
                    fitted_model["context_instance"],
                    fitted_model["history_instance"],
                )
                prior = Context13PriorInput(
                    fitted_instance=instance,
                    context_model_sha256=fitted_model["context_model_sha256"],
                    team_id=team_id,
                    component_status=FITTED_STATUS,
                    population=int(source_row["target_population"]),
                    context_prior_pmf=retained_pmf,
                    location_parts=parts,
                    residual_scale=float(
                        source_row["context_conditional_residual_scale"]
                    ),
                )
                # Independently reconstruct the PR #159 alpha=.75 row with its
                # retained helper and PMF routine, then compare to the API output.
                if parts.context_only_subtotal <= 0:
                    reference = retained_pmf.copy()
                else:
                    reference_points = issue_158.moderated_location_points(parts, 0.75)
                    reference = issue_158.crossover.normal_mixture_pmf(
                        reference_points,
                        float(source_row["context_conditional_residual_scale"]),
                        int(source_row["target_population"]),
                    )
            else:
                assert status == COLD_START_STATUS
                fallback_count += 1
                prior = Context13PriorInput(
                    fitted_instance=instance,
                    context_model_sha256=fitted_model["context_model_sha256"],
                    team_id=team_id,
                    component_status=COLD_START_STATUS,
                    population=int(source_row["target_population"]),
                    context_prior_pmf=retained_pmf,
                    location_parts=None,
                    residual_scale=None,
                )
                reference = retained_pmf.copy()

            actual = construct_candidate_prior(prior).pmf
            np.testing.assert_array_equal(actual, reference)
            season_hashes.append(
                {
                    "team_id": team_id,
                    "prior_pmf_sha256": issue_158.crossover.sha256_json(
                        reference.tolist()
                    ),
                }
            )
        expected_season = parity["seasons"][str(season)]
        assert expected_season["team_seasons"] == len(season_hashes)
        assert expected_season["fitted"] == fitted_count
        assert expected_season["cold_start_fallback"] == fallback_count
        assert expected_season["team_pmf_hashes_sha256"] == sha256_json(season_hashes)
        panel[str(season)] = len(season_hashes)

    assert panel == {"2022": 131, "2023": 133, "2024": 134, "2025": 136}


def test_candidate_provenance_pins_implementation_and_keeps_production_artifacts() -> (
    None
):
    provenance_path = ROOT / "data/processed/context_v1_4_candidate/provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    parity_path = (
        ROOT / "data/processed/context_v1_4_candidate/development_panel_parity.json"
    )
    assert provenance["candidate_spec_sha256"] == candidate_spec_sha256()
    implementation_sources = {
        "candidate_builder": ROOT / "src/gippyrank/context_prior_v1_4_candidate.py",
        "positive_net_moderation": ROOT
        / "src/gippyrank/context_positive_net_moderation.py",
        "context_prior_v1_3": ROOT / "src/gippyrank/context_prior_v1_3.py",
        "context_prior_core": ROOT / "src/gippyrank/context_prior.py",
        "rank_distribution": ROOT / "src/gippyrank/preseason.py",
        "dependency_lock": ROOT / "uv.lock",
    }
    assert provenance["implementation_sources_sha256"] == {
        name: _sha256(path) for name, path in implementation_sources.items()
    }
    assert provenance["development_panel_parity_sha256"] == _sha256(parity_path)

    protected = (
        ROOT / "data/processed/preseason/context_v1_3_candidate/predictions.csv",
        ROOT / "data/processed/preseason/history/predictions.csv",
        ROOT / "data/processed/context_history_crossover/hybrid_prior_results.csv",
        ROOT / "data/processed/context_positive_net_moderation/team_season_results.csv",
    )
    before = {path: _sha256(path) for path in protected}
    construct_candidate_prior(_fitted_input())
    assert {path: _sha256(path) for path in protected} == before
