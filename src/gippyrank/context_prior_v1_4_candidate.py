"""Frozen, research-only Context 1.4 candidate prior construction.

The semantic contract lives in this module as immutable code-level data. The
checked-in JSON is an audit mirror; candidate construction has no repository
layout or configuration-file dependency.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gippyrank.context_prior import AnnualFittedInstance, InferenceRow
from gippyrank.context_prior_v1_3 import (
    Context13FittedModelSource,
    Context13LocationDecomposition,
    ContextTransferInputProvenance,
    decompose_context13_location,
    moderated_location_points,
    validate_context13_fit_transfer_compatibility,
    validate_context13_fitted_model,
)
from gippyrank.history_annual_v1_1 import HistoryAnnualArtifactSource
from gippyrank.preseason import (
    DirectRankModel,
    conditional_rank_mixture_pmf,
)

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
CANDIDATE_ARTIFACT_SCHEMA_VERSION = 5
CONTEXT_1_3_VERSION = "1.3"
FITTED_STATUS = "fitted"
COLD_START_STATUS = "cold_start_fallback"
PMF_TOLERANCE = 5e-16
_CANDIDATE_PRIOR_TOKEN = object()
_FALLBACK_SOURCE_TOKEN = object()


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
    result = values.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, init=False)
class Context13FallbackSource:
    """A cold-start PMF derived from a named model or read from a source artifact."""

    target_season: int
    team_id: str
    team_name: str
    population: int
    cold_start_reason: str
    model_identity: str
    method_identity: str
    source_artifact_id: str
    source_artifact_sha256: str | None
    source_parameters_json: str | None
    source_parameters_sha256: str | None
    trained_through_season: int | None
    fitted_instance_identity_sha256: str | None
    source_model_metadata_sha256: str | None
    source_producer_kind: str | None
    source_producer_identity_status: str | None
    source_producer_identity_sha256: str | None
    upstream_source_identity_sha256: str | None
    upstream_provenance_class: str | None
    source_history_semantic_spec_identity_sha256: str | None
    source_training_input_source_identity_sha256: str | None
    source_prediction_semantic_sha256: str | None
    source_population_identity_sha256: str | None
    source_team_rows_sha256: str | None
    source_prior_methods: tuple[str, ...]
    pmf_sha256: str
    pmf: np.ndarray
    research_fixture: bool
    source_identity_sha256: str

    def __init__(
        self,
        *,
        target_season: int,
        team_id: str,
        team_name: str,
        population: int,
        cold_start_reason: str,
        model_identity: str,
        method_identity: str,
        source_artifact_id: str,
        source_artifact_sha256: str | None,
        source_parameters_json: str | None,
        source_parameters_sha256: str | None,
        trained_through_season: int | None = None,
        fitted_instance_identity_sha256: str | None = None,
        source_model_metadata_sha256: str | None = None,
        source_producer_kind: str | None = None,
        source_producer_identity_status: str | None = None,
        source_producer_identity_sha256: str | None = None,
        upstream_source_identity_sha256: str | None = None,
        upstream_provenance_class: str | None = None,
        source_history_semantic_spec_identity_sha256: str | None = None,
        source_training_input_source_identity_sha256: str | None = None,
        source_prediction_semantic_sha256: str | None = None,
        source_population_identity_sha256: str | None = None,
        source_team_rows_sha256: str | None = None,
        source_prior_methods: tuple[str, ...] = (),
        pmf: np.ndarray,
        research_fixture: bool,
        _construction_token: object = None,
    ) -> None:
        if _construction_token is not _FALLBACK_SOURCE_TOKEN:
            raise TypeError("fallback sources must be loaded from a canonical path or research fixture")
        values = _validate_pmf(pmf, population, "Context 1.3 cold-start fallback")
        for name, value in (
            ("target_season", target_season),
            ("team_id", team_id),
            ("team_name", team_name),
            ("population", population),
            ("cold_start_reason", cold_start_reason),
            ("model_identity", model_identity),
            ("method_identity", method_identity),
            ("source_artifact_id", source_artifact_id),
            ("source_artifact_sha256", source_artifact_sha256),
            ("source_parameters_json", source_parameters_json),
            ("source_parameters_sha256", source_parameters_sha256),
            ("trained_through_season", trained_through_season),
            ("fitted_instance_identity_sha256", fitted_instance_identity_sha256),
            ("source_model_metadata_sha256", source_model_metadata_sha256),
            ("source_producer_kind", source_producer_kind),
            ("source_producer_identity_status", source_producer_identity_status),
            ("source_producer_identity_sha256", source_producer_identity_sha256),
            ("upstream_source_identity_sha256", upstream_source_identity_sha256),
            ("upstream_provenance_class", upstream_provenance_class),
            ("source_history_semantic_spec_identity_sha256", source_history_semantic_spec_identity_sha256),
            ("source_training_input_source_identity_sha256", source_training_input_source_identity_sha256),
            ("source_prediction_semantic_sha256", source_prediction_semantic_sha256),
            ("source_population_identity_sha256", source_population_identity_sha256),
            ("source_team_rows_sha256", source_team_rows_sha256),
            ("source_prior_methods", tuple(source_prior_methods)),
            ("pmf_sha256", sha256_json(values.tolist())),
            ("pmf", values),
            ("research_fixture", research_fixture),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source_identity_sha256", sha256_json(self.identity_payload()))
        self.__post_init__()

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_season, bool)
            or not isinstance(self.target_season, int)
            or self.target_season <= 1
            or not self.team_id
            or not self.team_name
            or not self.cold_start_reason
            or not self.model_identity
            or not self.method_identity
            or not self.source_artifact_id
        ):
            raise ValueError("cold-start fallback identity is incomplete")
        if self.research_fixture:
            if (
                self.model_identity != "pr159-retained-research-reference"
                or self.cold_start_reason != "unspecified_in_pr159_reference"
            ):
                raise ValueError(
                    "retained PR #159 fallback reason must remain explicitly unspecified"
                )
        elif self.model_identity == "history_prior/1.1":
            # HistoryAnnualArtifactSource.cold_start_prediction has already
            # validated its method/reason mapping against the source row.
            if self.upstream_provenance_class not in {
                "retained_legacy_history_artifact", "canonical_history_1_1_annual_output"
            }:
                raise ValueError("History fallback requires its exact annual provenance class")
            if self.upstream_provenance_class == "canonical_history_1_1_annual_output":
                if not _is_sha256(self.source_history_semantic_spec_identity_sha256) or not _is_sha256(
                    self.source_training_input_source_identity_sha256
                ):
                    raise ValueError("canonical History fallback requires its build lineage")
            elif (
                self.source_history_semantic_spec_identity_sha256 is not None
                or self.source_training_input_source_identity_sha256 is not None
            ):
                raise ValueError("retained History fallback cannot claim canonical build lineage")
            if (
                self.trained_through_season != self.target_season - 1
                or not _is_sha256(self.fitted_instance_identity_sha256)
                or not _is_sha256(self.upstream_source_identity_sha256)
                or not _is_sha256(self.source_prediction_semantic_sha256)
                or not _is_sha256(self.source_population_identity_sha256)
                or not _is_sha256(self.source_team_rows_sha256)
                or not self.source_prior_methods
                or tuple(sorted(set(self.source_prior_methods)))
                != self.source_prior_methods
                or self.method_identity not in self.source_prior_methods
            ):
                raise ValueError("History fallback must bind its rolling-origin fitted instance")
            if self.upstream_provenance_class == "canonical_history_1_1_annual_output":
                if self.source_producer_identity_status != "verified_canonical_build" or not _is_sha256(
                    self.source_producer_identity_sha256
                ):
                    raise ValueError("canonical History fallback requires its actual producer identity")
                if self.method_identity == "learned_fcs_to_fbs_transition":
                    if (
                        self.source_producer_kind != "direct_rank_model"
                        or not _is_sha256(self.source_model_metadata_sha256)
                        or self.source_parameters_json is not None
                    ):
                        raise ValueError("History transition fallback requires its transition model")
                elif self.method_identity == "generic_fbs_cold_start":
                    if (
                        self.source_producer_kind != "generic_rank_prior"
                        or self.source_model_metadata_sha256 is not None
                        or self.source_parameters_json is None
                    ):
                        raise ValueError("History generic fallback requires its GenericRankPrior parameters")
                else:
                    raise ValueError("History fallback method is not a cold-start method")
            elif (
                self.source_producer_identity_status != "unavailable_in_retained_legacy_artifact"
                or self.source_producer_kind is not None
                or self.source_producer_identity_sha256 is not None
                or self.source_model_metadata_sha256 is not None
                or self.source_parameters_json is not None
                or self.method_identity != "learned_fcs_to_fbs_transition"
            ):
                raise ValueError("retained History transition producer identity must be unavailable")
        else:
            raise ValueError("cold-start fallback model identity is unsupported")
        if self.source_artifact_sha256 is None and self.source_parameters_sha256 is None:
            raise ValueError("cold-start fallback must bind its source artifact or parameters")
        for digest in (self.source_artifact_sha256, self.source_parameters_sha256):
            if digest is not None and not _is_sha256(digest):
                raise ValueError("cold-start source identity must be SHA-256")
        if self.source_parameters_json is not None:
            try:
                parameters = json.loads(self.source_parameters_json)
            except json.JSONDecodeError as error:
                raise ValueError("cold-start source parameters must be valid JSON") from error
            if (
                self.source_parameters_sha256 is None
                or sha256_json(parameters) != self.source_parameters_sha256
            ):
                raise ValueError("cold-start source parameters do not match their SHA-256")
        elif self.source_parameters_sha256 is not None:
            raise ValueError("cold-start source parameter hash requires its parameters")
        if self.upstream_provenance_class == "canonical_history_1_1_annual_output":
            producer_payload = {
                "producer_kind": self.source_producer_kind,
                "producer_role": self.method_identity,
                "producer_identity_status": self.source_producer_identity_status,
                "producer_model_metadata_sha256": self.source_model_metadata_sha256,
                "producer_parameters": (
                    json.loads(self.source_parameters_json)
                    if self.source_parameters_json is not None
                    else None
                ),
                "producer_parameters_sha256": self.source_parameters_sha256,
            }
            if sha256_json(producer_payload) != self.source_producer_identity_sha256:
                raise ValueError("History fallback producer identity is inconsistent")
        if self.research_fixture and (
            any(
                value is not None
                for value in (
                    self.trained_through_season,
                    self.fitted_instance_identity_sha256,
                    self.source_model_metadata_sha256,
                    self.source_producer_kind,
                    self.source_producer_identity_status,
                    self.source_producer_identity_sha256,
                    self.upstream_source_identity_sha256,
                    self.upstream_provenance_class,
                    self.source_history_semantic_spec_identity_sha256,
                    self.source_training_input_source_identity_sha256,
                    self.source_prediction_semantic_sha256,
                    self.source_population_identity_sha256,
                    self.source_team_rows_sha256,
                )
            )
            or self.source_prior_methods
        ):
            raise ValueError("research fallback cannot claim a canonical History fitted instance")
        if self.source_identity_sha256 != sha256_json(self.identity_payload()):
            raise ValueError("cold-start fallback source identity is inconsistent")

    def identity_payload(self) -> dict[str, object]:
        return {
            "target_season": self.target_season,
            "trained_through_season": self.trained_through_season,
            "team_id": self.team_id,
            "team_name": self.team_name,
            "population": self.population,
            "cold_start_reason": self.cold_start_reason,
            "model_identity": self.model_identity,
            "method_identity": self.method_identity,
            "source_artifact_id": self.source_artifact_id,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_parameters": (
                json.loads(self.source_parameters_json)
                if self.source_parameters_json is not None
                else None
            ),
            "source_parameters_sha256": self.source_parameters_sha256,
            "fitted_instance_identity_sha256": self.fitted_instance_identity_sha256,
            "source_model_metadata_sha256": self.source_model_metadata_sha256,
            "source_producer_kind": self.source_producer_kind,
            "source_producer_identity_status": self.source_producer_identity_status,
            "source_producer_identity_sha256": self.source_producer_identity_sha256,
            "upstream_source_identity_sha256": self.upstream_source_identity_sha256,
            **(
                {
                    "upstream_provenance_class": self.upstream_provenance_class,
                    "source_history_semantic_spec_identity_sha256": self.source_history_semantic_spec_identity_sha256,
                    "source_training_input_source_identity_sha256": self.source_training_input_source_identity_sha256,
                }
                if self.upstream_provenance_class == "canonical_history_1_1_annual_output"
                else {}
            ),
            "source_prediction_semantic_sha256": self.source_prediction_semantic_sha256,
            "source_population_identity_sha256": self.source_population_identity_sha256,
            "source_team_rows_sha256": self.source_team_rows_sha256,
            "source_prior_methods": list(self.source_prior_methods),
            "pmf_sha256": self.pmf_sha256,
            "research_fixture": self.research_fixture,
        }

    @classmethod
    def from_history_annual_source(
        cls,
        *,
        source: HistoryAnnualArtifactSource,
        target_season: int,
        trained_through_season: int,
        team_id: str,
        team_name: str,
        population: int,
        cold_start_reason: str,
    ) -> Context13FallbackSource:
        """Select a canonical History row from an already validated typed source."""
        if type(source) is not HistoryAnnualArtifactSource:
            raise TypeError("History fallback requires a validated typed annual source")
        row = source.cold_start_prediction(
            target_season=target_season,
            trained_through_season=trained_through_season,
            team_id=team_id,
            team_name=team_name,
            population=population,
            cold_start_reason=cold_start_reason,
        )
        pmf = np.asarray(row["pmf"], dtype=float)
        producer = row["producer"]
        parameters = producer["producer_parameters"]
        return cls(
            target_season=target_season,
            team_id=team_id,
            team_name=team_name,
            population=population,
            cold_start_reason=cold_start_reason,
            model_identity=f"{source.model_family}/{source.spec_version}",
            method_identity=str(row["prior_method"]),
            source_artifact_id=source.artifact_id,
            source_artifact_sha256=source.prediction_artifact_sha256,
            source_parameters_json=(
                json.dumps(parameters, sort_keys=True, separators=(",", ":"))
                if parameters is not None
                else None
            ),
            source_parameters_sha256=producer["producer_parameters_sha256"],
            trained_through_season=source.trained_through_season,
            fitted_instance_identity_sha256=source.fitted_instance_identity_sha256,
            source_model_metadata_sha256=producer["producer_model_metadata_sha256"],
            source_producer_kind=producer["producer_kind"],
            source_producer_identity_status=producer["producer_identity_status"],
            source_producer_identity_sha256=producer["producer_identity_sha256"],
            upstream_source_identity_sha256=source.source_identity_sha256,
            upstream_provenance_class=source.provenance_class,
            source_history_semantic_spec_identity_sha256=source.history_semantic_spec_identity_sha256,
            source_training_input_source_identity_sha256=source.training_input_source_identity_sha256,
            source_prediction_semantic_sha256=source.prediction_semantic_sha256,
            source_population_identity_sha256=source.team_ids_sha256,
            source_team_rows_sha256=source.team_rows_sha256,
            source_prior_methods=source.prior_methods,
            pmf=pmf,
            research_fixture=False,
            _construction_token=_FALLBACK_SOURCE_TOKEN,
        )

    @classmethod
    def from_research_fixture_csv(
        cls,
        artifact_path: str | Path,
        *,
        target_season: int,
        team_id: str,
        population: int,
    ) -> Context13FallbackSource:
        """Read a retained PR #159 PMF with no cold-start reason in the source."""
        content = Path(artifact_path).read_bytes()
        try:
            reader = csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""))
            rows = [
                row
                for row in reader
                if row.get("season") == str(target_season)
                and row.get("team_id") == team_id
                and row.get("arm") == "CC"
            ]
        except (UnicodeDecodeError, csv.Error) as error:
            raise ValueError("research fallback fixture is not a valid UTF-8 CSV") from error
        if len(rows) != 1 or rows[0].get("component_status") != COLD_START_STATUS:
            raise ValueError("research fallback fixture row is not a retained cold-start PMF")
        row = rows[0]
        try:
            pmf = np.asarray(json.loads(row["prior_pmf"]), dtype=float)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("research fallback fixture has an invalid PMF") from error
        if len(pmf) != population:
            raise ValueError("research fallback fixture population does not match the team")
        return cls(
            target_season=target_season,
            team_id=team_id,
            team_name=row["team_name"],
            population=population,
            cold_start_reason="unspecified_in_pr159_reference",
            model_identity="pr159-retained-research-reference",
            method_identity="copy-retained-CC-source-PMF",
            source_artifact_id="context-history-crossover/hybrid-prior-results.csv",
            source_artifact_sha256=hashlib.sha256(content).hexdigest(),
            source_parameters_json=None,
            source_parameters_sha256=None,
            trained_through_season=None,
            fitted_instance_identity_sha256=None,
            source_model_metadata_sha256=None,
            upstream_source_identity_sha256=None,
            source_prediction_semantic_sha256=None,
            source_population_identity_sha256=None,
            source_team_rows_sha256=None,
            source_prior_methods=(),
            pmf=pmf,
            research_fixture=True,
            _construction_token=_FALLBACK_SOURCE_TOKEN,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "source_identity_sha256": self.source_identity_sha256,
            "cold_start_reason": self.cold_start_reason,
            "model_identity": self.model_identity,
            "method_identity": self.method_identity,
            "source_artifact_id": self.source_artifact_id,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_parameters": (
                json.loads(self.source_parameters_json)
                if self.source_parameters_json is not None
                else None
            ),
            "source_parameters_sha256": self.source_parameters_sha256,
            "pmf_sha256": self.pmf_sha256,
            "research_fixture": self.research_fixture,
            "target_season": self.target_season,
            "trained_through_season": self.trained_through_season,
            "team_id": self.team_id,
            "team_name": self.team_name,
            "population": self.population,
            "fitted_instance_identity_sha256": self.fitted_instance_identity_sha256,
            "source_model_metadata_sha256": self.source_model_metadata_sha256,
            "source_producer_kind": self.source_producer_kind,
            "source_producer_identity_status": self.source_producer_identity_status,
            "source_producer_identity_sha256": self.source_producer_identity_sha256,
            "upstream_source_identity_sha256": self.upstream_source_identity_sha256,
            "upstream_provenance_class": self.upstream_provenance_class,
            "source_history_semantic_spec_identity_sha256": self.source_history_semantic_spec_identity_sha256,
            "source_training_input_source_identity_sha256": self.source_training_input_source_identity_sha256,
            "source_prediction_semantic_sha256": self.source_prediction_semantic_sha256,
            "source_population_identity_sha256": self.source_population_identity_sha256,
            "source_team_rows_sha256": self.source_team_rows_sha256,
            "source_prior_methods": list(self.source_prior_methods),
        }


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class Context13PriorInput:
    """Actual Context 1.3 model and outcome-free inputs for one target team.

    Fitted priors are computed from ``fitted_model`` and ``inference_row``;
    callers cannot provide a model hash, a diagnostic-row decomposition, or a
    fitted PMF. Cold-start priors require a typed, bound fallback source.
    """

    fitted_model: DirectRankModel | None
    fitted_instance: AnnualFittedInstance
    fitted_model_source: Context13FittedModelSource | None
    transfer_provenance: ContextTransferInputProvenance
    component_status: str
    fitted_prior_pmf: np.ndarray | None = None
    inference_row: InferenceRow | None = None
    expected_team_id: str | None = None
    expected_target_season: int | None = None
    fallback_source: Context13FallbackSource | None = None

    def __post_init__(self) -> None:
        if (
            self.fitted_instance.model_family != "context_prior"
            or self.fitted_instance.spec_version != CONTEXT_1_3_VERSION
            or self.fitted_instance.target_season <= 1
            or self.fitted_instance.trained_through_season
            != self.fitted_instance.target_season - 1
            or self.fitted_instance.context_effective_cutoff is not None
        ):
            raise ValueError("candidate input must name a rolling-origin Context 1.3 instance")
        if self.transfer_provenance.target_season != self.fitted_instance.target_season:
            raise ValueError("transfer provenance does not match the Context target season")
        model_source = self.fitted_model_source
        if model_source is None:
            raise ValueError("Context 1.3 candidate inputs require authoritative fit provenance")
        model_source.validate_instance(self.fitted_instance)
        validate_context13_fit_transfer_compatibility(
            model_source.provenance_class,
            self.transfer_provenance.provenance_class,
        )
        if self.component_status == FITTED_STATUS:
            if self.fitted_model is None or self.inference_row is None:
                raise ValueError("fitted Context priors require an inference row")
            model_source.validate_model(
                self.fitted_model,
                self.fitted_instance,
                allow_research_only=(model_source.provenance_class == "research_only"),
            )
            if (
                self.expected_team_id != self.inference_row.team_id
                or self.expected_target_season != self.fitted_instance.target_season
            ):
                raise ValueError("Context source team or season does not match inference inputs")
            if self.fitted_prior_pmf is None:
                raise ValueError("fitted Context priors require their source C1.3 PMF")
            if self.fallback_source is not None:
                raise ValueError("fitted Context priors cannot carry fallback inputs")
            # Validate the model-row-season-team and decomposition at construction.
            decompose_context13_location(
                self.fitted_model,
                self.fitted_instance,
                self.inference_row,
                self.transfer_provenance,
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
            if self.fitted_model is not None or self.inference_row is not None:
                raise ValueError("cold-start fallbacks cannot carry fitted model inputs")
            if self.fitted_prior_pmf is not None:
                raise ValueError("cold-start priors cannot carry a fitted model PMF")
            fallback = self.fallback_source
            if fallback is None:
                raise ValueError("cold-start priors require a bound fallback source")
            if (
                fallback.target_season != self.fitted_instance.target_season
                or fallback.target_season != self.transfer_provenance.target_season
            ):
                raise ValueError("cold-start fallback target season does not match its input")
            if fallback.team_id not in self.transfer_provenance.target_fbs_team_ids:
                raise ValueError("cold-start team is absent from the validated transfer population")
            if fallback.research_fixture and self.transfer_provenance.provenance_class != (
                "retrospective_research_transfer_reconstruction"
            ):
                raise ValueError("research and non-research fallback provenance cannot be mixed")
        else:
            raise ValueError(f"unsupported Context component status: {self.component_status}")

    @classmethod
    def fitted(
        cls,
        *,
        model: DirectRankModel,
        fitted_instance: AnnualFittedInstance,
        fitted_model_source: Context13FittedModelSource | None = None,
        inference_row: InferenceRow,
        transfer_provenance: ContextTransferInputProvenance,
        prior_pmf: np.ndarray,
        expected_team_id: str,
        expected_target_season: int,
    ) -> Context13PriorInput:
        """Create a fitted input whose PMF and decomposition come from the model."""
        return cls(
            fitted_model=model,
            fitted_instance=fitted_instance,
            fitted_model_source=fitted_model_source,
            transfer_provenance=transfer_provenance,
            component_status=FITTED_STATUS,
            fitted_prior_pmf=prior_pmf,
            inference_row=inference_row,
            expected_team_id=expected_team_id,
            expected_target_season=expected_target_season,
        )

    @classmethod
    def cold_start(
        cls,
        *,
        fitted_instance: AnnualFittedInstance,
        fitted_model_source: Context13FittedModelSource | None = None,
        transfer_provenance: ContextTransferInputProvenance,
        fallback_source: Context13FallbackSource,
    ) -> Context13PriorInput:
        """Create an unchanged, source-bound fallback with no C1.3 model claim."""
        return cls(
            fitted_model=None,
            fitted_instance=fitted_instance,
            fitted_model_source=fitted_model_source,
            transfer_provenance=transfer_provenance,
            component_status=COLD_START_STATUS,
            fallback_source=fallback_source,
        )

    @property
    def team_id(self) -> str:
        if self.inference_row is not None:
            return self.inference_row.team_id
        assert self.fallback_source is not None
        return self.fallback_source.team_id

    @property
    def team_name(self) -> str:
        if self.inference_row is not None:
            return self.inference_row.team_name
        assert self.fallback_source is not None
        return self.fallback_source.team_name

    @property
    def population(self) -> int:
        if self.inference_row is not None:
            return self.inference_row.population
        assert self.fallback_source is not None
        return self.fallback_source.population

    @property
    def context_model_sha256(self) -> str | None:
        if self.fitted_model is None:
            return None
        return validate_context13_fitted_model(self.fitted_model, self.fitted_instance)

    def source_prior_pmf(self) -> np.ndarray:
        if self.component_status == COLD_START_STATUS:
            assert self.fallback_source is not None
            return self.fallback_source.pmf.copy()
        assert self.fitted_prior_pmf is not None
        return self.fitted_prior_pmf.copy()

    def location_decomposition(self) -> Context13LocationDecomposition | None:
        if self.component_status == COLD_START_STATUS:
            return None
        assert self.inference_row is not None and self.fitted_model is not None
        return decompose_context13_location(
            self.fitted_model,
            self.fitted_instance,
            self.inference_row,
            self.transfer_provenance,
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
                "cold_start_reason": (
                    self.fallback_source.cold_start_reason
                    if self.fallback_source is not None
                    else None
                ),
                "fallback_source": (
                    self.fallback_source.metadata()
                    if self.fallback_source is not None
                    else None
                ),
                "transfer_provenance_sha256": self.transfer_provenance.source_identity_sha256,
            }
        )


@dataclass(frozen=True, init=False)
class Context14CandidatePrior:
    """Candidate PMF plus independently computed source and semantic identities."""

    team_id: str
    target_season: int
    trained_through_season: int
    context_model_sha256: str | None
    fitted_model_source: Context13FittedModelSource
    source_prior_pmf_sha256: str
    candidate_semantics_sha256: str
    inference_inputs_sha256: str
    decomposition_sha256: str | None
    transfer_provenance: ContextTransferInputProvenance
    fallback_source: Context13FallbackSource | None
    component_status: str
    population: int
    pmf: np.ndarray

    def __init__(
        self,
        *,
        team_id: str,
        target_season: int,
        trained_through_season: int,
        context_model_sha256: str | None,
        fitted_model_source: Context13FittedModelSource,
        source_prior_pmf_sha256: str,
        candidate_semantics_sha256: str,
        inference_inputs_sha256: str,
        decomposition_sha256: str | None,
        transfer_provenance: ContextTransferInputProvenance,
        fallback_source: Context13FallbackSource | None,
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
            ("fitted_model_source", fitted_model_source),
            ("source_prior_pmf_sha256", source_prior_pmf_sha256),
            ("candidate_semantics_sha256", candidate_semantics_sha256),
            ("inference_inputs_sha256", inference_inputs_sha256),
            ("decomposition_sha256", decomposition_sha256),
            ("transfer_provenance", transfer_provenance),
            ("fallback_source", fallback_source),
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
        if self.transfer_provenance.target_season != self.target_season:
            raise ValueError("candidate source provenance has the wrong target season")
        self.fitted_model_source.validate_instance(
            AnnualFittedInstance(
                "context_prior",
                CONTEXT_1_3_VERSION,
                self.trained_through_season,
                self.target_season,
                None,
            )
        )
        validate_context13_fit_transfer_compatibility(
            self.fitted_model_source.provenance_class,
            self.transfer_provenance.provenance_class,
        )
        if self.component_status not in {FITTED_STATUS, COLD_START_STATUS}:
            raise ValueError("candidate component status is invalid")
        if self.candidate_semantics_sha256 != candidate_spec_sha256():
            raise ValueError("candidate semantic identity does not match the frozen contract")
        if self.component_status == FITTED_STATUS and (
            self.context_model_sha256 is None
            or self.context_model_sha256
            != self.fitted_model_source.model_metadata_sha256
            or self.fallback_source is not None
            or self.decomposition_sha256 is None
        ):
            raise ValueError("fitted candidate requires its model identity and no fallback source")
        if self.component_status == COLD_START_STATUS and (
            self.context_model_sha256 is not None
            or self.fallback_source is None
            or self.decomposition_sha256 is not None
        ):
            raise ValueError("cold-start candidate must bind its fallback without claiming a C1.3 model")
        for name, digest in (
            ("source prior PMF", self.source_prior_pmf_sha256),
            ("candidate semantics", self.candidate_semantics_sha256),
            ("inference inputs", self.inference_inputs_sha256),
        ):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"candidate {name} identity must be SHA-256")
        if self.context_model_sha256 is not None and not _is_sha256(self.context_model_sha256):
            raise ValueError("candidate Context model identity must be SHA-256")
        if self.decomposition_sha256 is not None and (
            len(self.decomposition_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.decomposition_sha256)
        ):
            raise ValueError("candidate decomposition identity must be SHA-256")
        candidate_pmf = _validate_pmf(self.pmf, self.population, "candidate")
        if self.component_status == COLD_START_STATUS:
            assert self.fallback_source is not None
            if (
                self.fallback_source.target_season != self.target_season
                or self.fallback_source.team_id != self.team_id
                or self.fallback_source.population != self.population
                or self.source_prior_pmf_sha256 != self.fallback_source.pmf_sha256
                or not np.array_equal(candidate_pmf, self.fallback_source.pmf)
            ):
                raise ValueError("cold-start candidate identity differs from its fallback source")
            if self.fallback_source.research_fixture and self.transfer_provenance.provenance_class != (
                "retrospective_research_transfer_reconstruction"
            ):
                raise ValueError("research fixture fallback cannot claim production provenance")
        object.__setattr__(self, "pmf", candidate_pmf)

    def artifact_bytes(self) -> bytes:
        """Serialize all source-state identities deterministically."""
        values = np.asarray(self.pmf, dtype=float).tolist()
        return json.dumps(
            {
                "artifact_schema_version": CANDIDATE_ARTIFACT_SCHEMA_VERSION,
                "candidate_semantics_sha256": self.candidate_semantics_sha256,
                "component_status": self.component_status,
                "context_model_sha256": self.context_model_sha256,
                "context_fit_provenance": self.fitted_model_source.to_metadata(),
                "context_fit_provenance_class": self.fitted_model_source.provenance_class,
                "context_fit_reproducibility_level": self.fitted_model_source.reproducibility_level,
                "context_fit_training_lineage": self.fitted_model_source.training_lineage_metadata(),
                "semantic_model_spec_sha256": self.fitted_model_source.frozen_model_spec_identity_sha256,
                "fitted_model_metadata_sha256": self.fitted_model_source.model_metadata_sha256,
                "fitted_instance_identity_sha256": self.fitted_model_source.fitted_instance_identity_sha256,
                "decomposition_sha256": self.decomposition_sha256,
                "inference_inputs_sha256": self.inference_inputs_sha256,
                "transfer_input_provenance": self.transfer_provenance.to_metadata(),
                "fallback_source": (
                    self.fallback_source.metadata()
                    if self.fallback_source is not None
                    else None
                ),
                "model_family": "context_prior",
                "population": self.population,
                "pmf": values,
                "pmf_sha256": sha256_json(values),
                "source_prior_pmf_sha256": self.source_prior_pmf_sha256,
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
            "transfer_provenance_class": decomposition.transfer_provenance.provenance_class,
            "transfer_provenance_sha256": decomposition.transfer_provenance.source_identity_sha256,
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
    model_sha256 = prior.context_model_sha256
    if prior.transfer_provenance.target_season != instance.target_season:
        raise ValueError("transfer provenance does not match the target season")
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
        fitted_model_source=prior.fitted_model_source,
        source_prior_pmf_sha256=sha256_json(source_pmf.tolist()),
        candidate_semantics_sha256=semantic_sha256,
        inference_inputs_sha256=prior.inference_inputs_sha256(),
        decomposition_sha256=decomposition_hash,
        transfer_provenance=prior.transfer_provenance,
        fallback_source=prior.fallback_source,
        component_status=prior.component_status,
        population=prior.population,
        pmf=candidate_pmf,
        _construction_token=_CANDIDATE_PRIOR_TOKEN,
    )


__all__ = [
    "COLD_START_STATUS",
    "FITTED_STATUS",
    "Context13FallbackSource",
    "Context13PriorInput",
    "Context14CandidatePrior",
    "HistoryAnnualArtifactSource",
    "candidate_spec_sha256",
    "construct_candidate_prior",
    "load_candidate_spec",
    "sha256_json",
]
