"""Frozen, research-only Context 1.4 candidate prior construction.

The semantic contract lives in this module as immutable code-level data. The
checked-in JSON is an audit mirror; candidate construction has no repository
layout or configuration-file dependency.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from gippyrank.context_prior import AnnualFittedInstance, InferenceRow
from gippyrank.context_prior_v1_3 import (
    Context13LocationDecomposition,
    ContextInputProvenance,
    decompose_context13_location,
    moderated_location_points,
    validate_context13_fitted_model,
)
from gippyrank.preseason import DirectRankModel, conditional_rank_mixture_pmf

_SEMANTIC_CONTRACT_JSON = (
    '{"alpha":0.75,"base_model":{"display_name":"Context 1.3",'
    '"model_family":"context_prior","spec_version":"1.3"},'
    '"candidate_version":"Context 1.4 candidate",'
    '"model_family":"context_prior",'
    '"moderation_metric":"context_only_subtotal",'
    '"moderation_rule":{"formula":"min(x, 0) + 0.75 * max(x, 0)",'
    '"x_gt_0":"0.75 * x","x_lte_0":"x"},'
    '"spec_version":"1.4-candidate"}'
)
_DEFAULT_LIFECYCLE_JSON = (
    '{"production":false,"status":"frozen research candidate",'
    '"validation_status":"not yet holdout validated"}'
)
FROZEN_CANDIDATE_SEMANTICS_SHA256 = (
    "db84045d2f7d80b1648360da52ef1e71f76093696cad4e4e7bbdbd57a086573f"
)
CONTEXT_1_3_VERSION = "1.3"
FITTED_STATUS = "fitted"
COLD_START_STATUS = "cold_start_fallback"
PMF_TOLERANCE = 1e-12
_CANDIDATE_PRIOR_TOKEN = object()


def _semantic_contract() -> dict[str, object]:
    semantic = json.loads(_SEMANTIC_CONTRACT_JSON)
    if sha256_json(semantic) != FROZEN_CANDIDATE_SEMANTICS_SHA256:
        raise ValueError("embedded Context 1.4 candidate semantics changed without a new freeze")
    return semantic


def sha256_json(value: object) -> str:
    """Hash canonical JSON values for deterministic artifact identities."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_candidate_spec(spec: Mapping[str, object] | None = None) -> dict[str, object]:
    """Return and fail-closed validate the sole frozen candidate contract.

    An optional mapping is useful for validating the audit mirror. Lifecycle
    metadata is checked for research-only status, but it is excluded from the
    semantic identity so a later validation-status update cannot change the
    model's identity.
    """
    payload: dict[str, object] = (
        {
            **_semantic_contract(),
            **json.loads(_DEFAULT_LIFECYCLE_JSON),
        }
        if spec is None
        else dict(spec)
    )
    expected = _semantic_contract()
    semantic = {key: payload.get(key) for key in expected}
    if semantic != expected:
        raise ValueError("Context 1.4 candidate semantics differ from the frozen contract")
    if payload.get("production") is not False:
        raise ValueError("Context 1.4 candidate must remain research-only")
    if payload.get("status") != "frozen research candidate":
        raise ValueError("Context 1.4 candidate lifecycle status is invalid")
    if not isinstance(payload.get("validation_status"), str) or not payload[
        "validation_status"
    ]:
        raise ValueError("Context 1.4 candidate validation status must be explicit")
    return payload


def candidate_spec_sha256(spec: Mapping[str, object] | None = None) -> str:
    """Hash only immutable semantics, never lifecycle or validation metadata."""
    payload = load_candidate_spec(spec)
    semantic = {key: payload[key] for key in _semantic_contract()}
    return sha256_json(semantic)


def _validate_pmf(pmf: np.ndarray, population: int, label: str) -> np.ndarray:
    values = np.asarray(pmf, dtype=float)
    if (
        isinstance(population, bool)
        or not isinstance(population, int)
        or population < 1
        or values.ndim != 1
        or len(values) != population
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.isclose(values.sum(), 1.0, rtol=0, atol=1e-8)
    ):
        raise ValueError(f"{label} PMF is invalid")
    return values.copy()


@dataclass(frozen=True)
class Context13PriorInput:
    """Actual Context 1.3 model and outcome-free inputs for one target team.

    Fitted priors are computed from ``fitted_model`` and ``inference_row``;
    callers cannot provide a model hash, a diagnostic-row decomposition, or a
    fitted PMF. Cold-start priors carry the PMF emitted by the fallback path.
    """

    fitted_model: DirectRankModel
    fitted_instance: AnnualFittedInstance
    input_provenance: ContextInputProvenance
    component_status: str
    fitted_prior_pmf: np.ndarray | None = None
    inference_row: InferenceRow | None = None
    expected_team_id: str | None = None
    expected_target_season: int | None = None
    cold_start_team_id: str | None = None
    cold_start_team_name: str | None = None
    cold_start_population: int | None = None
    cold_start_prior_pmf: np.ndarray | None = None
    cold_start_reason: str | None = None

    def __post_init__(self) -> None:
        validate_context13_fitted_model(self.fitted_model, self.fitted_instance)
        if self.input_provenance.target_season != self.fitted_instance.target_season:
            raise ValueError("Context input provenance does not match the fitted target season")
        if self.component_status == FITTED_STATUS:
            if self.inference_row is None:
                raise ValueError("fitted Context priors require an inference row")
            if (
                self.expected_team_id != self.inference_row.team_id
                or self.expected_target_season != self.fitted_instance.target_season
            ):
                raise ValueError("Context source team or season does not match inference inputs")
            if self.fitted_prior_pmf is None:
                raise ValueError("fitted Context priors require their source C1.3 PMF")
            if any(
                value is not None
                for value in (
                    self.cold_start_team_id,
                    self.cold_start_team_name,
                    self.cold_start_population,
                    self.cold_start_prior_pmf,
                    self.cold_start_reason,
                )
            ):
                raise ValueError("fitted Context priors cannot carry fallback inputs")
            # Validate the model-row-season-team and decomposition at construction.
            decompose_context13_location(
                self.fitted_model,
                self.fitted_instance,
                self.inference_row,
                self.input_provenance,
            )
            source_pmf = _validate_pmf(
                self.fitted_prior_pmf,
                self.inference_row.population,
                "Context 1.3 source",
            )
            model_row = self.inference_row
            model_pmf = self.fitted_model.pmf(
                {name: model_row.features[name] for name in self.fitted_model.feature_names},
                np.asarray(model_row.lag1_z, dtype=float),
                model_row.population,
            )
            if not np.allclose(model_pmf, source_pmf, rtol=0, atol=PMF_TOLERANCE):
                raise ValueError("Context 1.3 source PMF does not match its fitted model inputs")
            object.__setattr__(self, "fitted_prior_pmf", source_pmf)
        elif self.component_status == COLD_START_STATUS:
            if self.inference_row is not None:
                raise ValueError("cold-start fallbacks cannot carry fitted inference rows")
            if self.fitted_prior_pmf is not None:
                raise ValueError("cold-start priors cannot carry a fitted model PMF")
            if (
                not self.cold_start_team_id
                or not self.cold_start_team_name
                or isinstance(self.cold_start_population, bool)
                or not isinstance(self.cold_start_population, int)
                or self.cold_start_population < 1
                or self.cold_start_prior_pmf is None
            ):
                raise ValueError("cold-start priors require team, population, and source PMF")
            fallback = _validate_pmf(
                self.cold_start_prior_pmf,
                self.cold_start_population,
                "cold-start Context 1.3 source",
            )
            object.__setattr__(self, "cold_start_prior_pmf", fallback)
            if not self.cold_start_reason:
                raise ValueError("cold-start priors require their fallback reason")
        else:
            raise ValueError(f"unsupported Context component status: {self.component_status}")

    @classmethod
    def fitted(
        cls,
        *,
        model: DirectRankModel,
        fitted_instance: AnnualFittedInstance,
        inference_row: InferenceRow,
        input_provenance: ContextInputProvenance,
        prior_pmf: np.ndarray,
        expected_team_id: str,
        expected_target_season: int,
    ) -> Context13PriorInput:
        """Create a fitted input whose PMF and decomposition come from the model."""
        return cls(
            model,
            fitted_instance,
            input_provenance,
            FITTED_STATUS,
            fitted_prior_pmf=prior_pmf,
            inference_row=inference_row,
            expected_team_id=expected_team_id,
            expected_target_season=expected_target_season,
        )

    @classmethod
    def cold_start(
        cls,
        *,
        model: DirectRankModel,
        fitted_instance: AnnualFittedInstance,
        input_provenance: ContextInputProvenance,
        team_id: str,
        team_name: str,
        population: int,
        prior_pmf: np.ndarray,
        reason: str,
    ) -> Context13PriorInput:
        """Create an unchanged Context 1.3 fallback prior with full provenance."""
        return cls(
            model,
            fitted_instance,
            input_provenance,
            COLD_START_STATUS,
            cold_start_team_id=team_id,
            cold_start_team_name=team_name,
            cold_start_population=population,
            cold_start_prior_pmf=prior_pmf,
            cold_start_reason=reason,
        )

    @property
    def team_id(self) -> str:
        if self.inference_row is not None:
            return self.inference_row.team_id
        assert self.cold_start_team_id is not None
        return self.cold_start_team_id

    @property
    def team_name(self) -> str:
        if self.inference_row is not None:
            return self.inference_row.team_name
        assert self.cold_start_team_name is not None
        return self.cold_start_team_name

    @property
    def population(self) -> int:
        if self.inference_row is not None:
            return self.inference_row.population
        assert self.cold_start_population is not None
        return self.cold_start_population

    @property
    def context_model_sha256(self) -> str:
        return validate_context13_fitted_model(self.fitted_model, self.fitted_instance)

    def source_prior_pmf(self) -> np.ndarray:
        if self.component_status == COLD_START_STATUS:
            assert self.cold_start_prior_pmf is not None
            return self.cold_start_prior_pmf.copy()
        assert self.fitted_prior_pmf is not None
        return self.fitted_prior_pmf.copy()

    def location_decomposition(self) -> Context13LocationDecomposition | None:
        if self.component_status == COLD_START_STATUS:
            return None
        assert self.inference_row is not None
        return decompose_context13_location(
            self.fitted_model,
            self.fitted_instance,
            self.inference_row,
            self.input_provenance,
        )

    def inference_inputs_sha256(self) -> str:
        if self.inference_row is not None:
            return self.location_decomposition().inference_inputs_sha256  # type: ignore[union-attr]
        return sha256_json(
            {
                "target_season": self.fitted_instance.target_season,
                "team_id": self.team_id,
                "team_name": self.team_name,
                "population": self.population,
                "cold_start_reason": self.cold_start_reason,
            }
        )


@dataclass(frozen=True, init=False)
class Context14CandidatePrior:
    """Candidate PMF plus independently computed source and semantic identities."""

    team_id: str
    target_season: int
    trained_through_season: int
    context_model_sha256: str
    source_context13_pmf_sha256: str
    candidate_semantics_sha256: str
    inference_inputs_sha256: str
    decomposition_sha256: str | None
    input_provenance: ContextInputProvenance
    component_status: str
    population: int
    pmf: np.ndarray

    def __init__(
        self,
        *,
        team_id: str,
        target_season: int,
        trained_through_season: int,
        context_model_sha256: str,
        source_context13_pmf_sha256: str,
        candidate_semantics_sha256: str,
        inference_inputs_sha256: str,
        decomposition_sha256: str | None,
        input_provenance: ContextInputProvenance,
        component_status: str,
        population: int,
        pmf: np.ndarray,
        _construction_token: object = None,
    ) -> None:
        if _construction_token is not _CANDIDATE_PRIOR_TOKEN:
            raise TypeError("candidate priors must be created by construct_candidate_prior")
        for name, value in (
            ("team_id", team_id),
            ("target_season", target_season),
            ("trained_through_season", trained_through_season),
            ("context_model_sha256", context_model_sha256),
            ("source_context13_pmf_sha256", source_context13_pmf_sha256),
            ("candidate_semantics_sha256", candidate_semantics_sha256),
            ("inference_inputs_sha256", inference_inputs_sha256),
            ("decomposition_sha256", decomposition_sha256),
            ("input_provenance", input_provenance),
            ("component_status", component_status),
            ("population", population),
            ("pmf", pmf),
        ):
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.team_id or self.target_season <= 1:
            raise ValueError("candidate identity requires a team and target season")
        if self.trained_through_season != self.target_season - 1:
            raise ValueError("candidate identity must be rolling-origin")
        if self.input_provenance.target_season != self.target_season:
            raise ValueError("candidate source provenance has the wrong target season")
        if self.component_status not in {FITTED_STATUS, COLD_START_STATUS}:
            raise ValueError("candidate component status is invalid")
        if self.candidate_semantics_sha256 != candidate_spec_sha256():
            raise ValueError("candidate semantic identity does not match the frozen contract")
        for name, digest in (
            ("Context model", self.context_model_sha256),
            ("source PMF", self.source_context13_pmf_sha256),
            ("candidate semantics", self.candidate_semantics_sha256),
            ("inference inputs", self.inference_inputs_sha256),
        ):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"candidate {name} identity must be SHA-256")
        if self.decomposition_sha256 is not None and (
            len(self.decomposition_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.decomposition_sha256)
        ):
            raise ValueError("candidate decomposition identity must be SHA-256")
        object.__setattr__(self, "pmf", _validate_pmf(self.pmf, self.population, "candidate"))

    def artifact_bytes(self) -> bytes:
        """Serialize all source-state identities deterministically."""
        values = np.asarray(self.pmf, dtype=float).tolist()
        return json.dumps(
            {
                "candidate_semantics_sha256": self.candidate_semantics_sha256,
                "component_status": self.component_status,
                "context_model_sha256": self.context_model_sha256,
                "decomposition_sha256": self.decomposition_sha256,
                "inference_inputs_sha256": self.inference_inputs_sha256,
                "input_provenance": {
                    "provenance_class": self.input_provenance.provenance_class,
                    "source_metadata_sha256": self.input_provenance.source_metadata_sha256,
                },
                "model_family": "context_prior",
                "population": self.population,
                "pmf": values,
                "pmf_sha256": sha256_json(values),
                "source_context13_pmf_sha256": self.source_context13_pmf_sha256,
                "spec_version": "1.4-candidate",
                "target_season": self.target_season,
                "team_id": self.team_id,
                "trained_through_season": self.trained_through_season,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


def _decomposition_sha256(
    decomposition: Context13LocationDecomposition,
) -> str:
    return sha256_json(
        {
            "context_model_sha256": decomposition.context_model_sha256,
            "context_only_subtotal": decomposition.context_only_subtotal,
            "conditional_location_points": decomposition.conditional_location_points.tolist(),
            "history_derived_subtotal": decomposition.history_derived_subtotal,
            "inference_inputs_sha256": decomposition.inference_inputs_sha256,
            "input_provenance_class": decomposition.input_provenance.provenance_class,
            "input_provenance_sha256": decomposition.input_provenance.source_metadata_sha256,
            "intercept": decomposition.intercept,
            "residual_scale": decomposition.residual_scale,
            "target_season": decomposition.target_season,
            "team_id": decomposition.team_id,
            "trained_through_season": decomposition.trained_through_season,
        }
    )


def construct_candidate_prior(prior: Context13PriorInput) -> Context14CandidatePrior:
    """Construct the frozen candidate from actual Context 1.3 model/input state."""
    spec = load_candidate_spec()
    semantic_sha256 = candidate_spec_sha256(spec)
    instance = prior.fitted_instance
    model_sha256 = validate_context13_fitted_model(prior.fitted_model, instance)
    if prior.input_provenance.target_season != instance.target_season:
        raise ValueError("Context input provenance does not match the fitted target season")
    source_pmf = prior.source_prior_pmf()
    decomposition = prior.location_decomposition()
    decomposition_hash: str | None = None
    if decomposition is None:
        candidate_pmf = source_pmf.copy()
    else:
        if decomposition.context_model_sha256 != model_sha256:
            raise ValueError("Context decomposition model identity does not match its fit")
        if decomposition.team_id != prior.team_id or decomposition.target_season != instance.target_season:
            raise ValueError("Context decomposition team or season identity does not match")
        canonical_base = conditional_rank_mixture_pmf(
            decomposition.conditional_location_points,
            decomposition.residual_scale,
            decomposition.population,
        )
        if not np.array_equal(canonical_base, source_pmf) and not np.allclose(
            canonical_base, source_pmf, rtol=0, atol=PMF_TOLERANCE
        ):
            raise ValueError("Context 1.3 source PMF does not match its fitted model inputs")
        decomposition_hash = _decomposition_sha256(decomposition)
        if decomposition.context_only_subtotal <= 0:
            # Keep both nonpositive fitted terms and their source PMF bit-identical.
            candidate_pmf = source_pmf.copy()
        else:
            points = moderated_location_points(decomposition, float(spec["alpha"]))
            candidate_pmf = conditional_rank_mixture_pmf(
                points, decomposition.residual_scale, decomposition.population
            )

    return Context14CandidatePrior(
        team_id=prior.team_id,
        target_season=instance.target_season,
        trained_through_season=instance.trained_through_season,
        context_model_sha256=model_sha256,
        source_context13_pmf_sha256=sha256_json(source_pmf.tolist()),
        candidate_semantics_sha256=semantic_sha256,
        inference_inputs_sha256=prior.inference_inputs_sha256(),
        decomposition_sha256=decomposition_hash,
        input_provenance=prior.input_provenance,
        component_status=prior.component_status,
        population=prior.population,
        pmf=candidate_pmf,
        _construction_token=_CANDIDATE_PRIOR_TOKEN,
    )


__all__ = [
    "COLD_START_STATUS",
    "FITTED_STATUS",
    "Context13PriorInput",
    "Context14CandidatePrior",
    "candidate_spec_sha256",
    "construct_candidate_prior",
    "load_candidate_spec",
    "sha256_json",
]
