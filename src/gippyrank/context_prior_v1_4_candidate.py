"""Frozen, research-only Context 1.4 candidate prior construction.

The candidate applies the checked-in specification to a valid rolling-origin
Context 1.3 fitted prior. It does not fit coefficients, use target-season
outcomes, or alter the active Context 1.3 model.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm

from gippyrank.context_positive_net_moderation import (
    LocationParts,
    moderated_location_points,
    parts_from_fitted_contributions,
)
from gippyrank.context_prior import AnnualFittedInstance
from gippyrank.preseason import rank_bin_edges

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SPEC_PATH = ROOT / "config/context_v1_4_candidate.json"
CONTEXT_1_3_VERSION = "1.3"
FITTED_STATUS = "fitted"
COLD_START_STATUS = "cold_start_fallback"
PMF_TOLERANCE = 1e-12


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def load_candidate_spec() -> dict[str, object]:
    """Read and validate the one committed, non-tunable candidate spec."""
    spec = json.loads(CANDIDATE_SPEC_PATH.read_text(encoding="utf-8"))
    required = {
        "alpha",
        "base_model",
        "candidate_version",
        "model_family",
        "spec_version",
        "moderation_metric",
        "moderation_rule",
        "status",
        "production",
        "validation_status",
    }
    if not isinstance(spec, dict) or not required <= set(spec):
        raise ValueError("Context 1.4 candidate specification is incomplete")
    alpha = spec["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("candidate alpha must be numeric")
    if not math.isfinite(float(alpha)) or not 0 <= float(alpha) <= 1:
        raise ValueError("candidate alpha must be finite and within [0, 1]")
    rule = spec["moderation_rule"]
    if not isinstance(rule, dict) or rule != {
        "x_lte_0": "x",
        "x_gt_0": f"{float(alpha):g} * x",
        "formula": f"min(x, 0) + {float(alpha):g} * max(x, 0)",
    }:
        raise ValueError("candidate moderation rule does not match frozen alpha")
    if (
        spec["model_family"] != "context_prior"
        or spec["spec_version"] != "1.4-candidate"
        or spec["candidate_version"] != "Context 1.4 candidate"
        or spec["base_model"]
        != {
            "model_family": "context_prior",
            "spec_version": "1.3",
            "display_name": "Context 1.3",
        }
        or spec["moderation_metric"] != "context_only_subtotal"
        or spec["status"] != "frozen research candidate"
        or spec["production"] is not False
        or spec["validation_status"] != "not yet holdout validated"
    ):
        raise ValueError("Context 1.4 candidate identity or status is invalid")
    return spec


def candidate_spec_sha256(spec: Mapping[str, object] | None = None) -> str:
    """Hash the canonical specification so any semantic edit changes identity."""
    return sha256_json(dict(spec) if spec is not None else load_candidate_spec())


def normal_mixture_pmf(
    locations: np.ndarray, scale: float, population: int
) -> np.ndarray:
    """Match Context 1.3 equal-weight Normal rank-bin mixture construction."""
    points = np.asarray(locations, dtype=float)
    if points.ndim != 1 or not len(points) or not np.isfinite(points).all():
        raise ValueError("conditional location points must be a finite vector")
    if isinstance(population, bool) or population < 1:
        raise ValueError("population must be a positive integer")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("conditional residual scale must be finite and positive")
    edges = rank_bin_edges(population)
    cdf = norm.cdf((edges[None, :] - points[:, None]) / scale)
    masses = np.maximum(np.diff(cdf, axis=1), 0.0)
    pmf = np.mean(masses, axis=0)
    total = float(pmf.sum())
    if not np.isfinite(pmf).all() or np.any(pmf < 0) or total <= 0:
        raise ValueError("candidate construction produced an invalid PMF")
    pmf /= total
    if len(pmf) != population or not np.isclose(pmf.sum(), 1.0, atol=1e-12):
        raise ValueError("candidate PMF has invalid support or normalization")
    return pmf


@dataclass(frozen=True)
class Context13PriorInput:
    """Outcome-free inputs required to transform one Context 1.3 prior."""

    fitted_instance: AnnualFittedInstance
    context_model_sha256: str
    team_id: str
    component_status: str
    population: int
    context_prior_pmf: np.ndarray
    location_parts: LocationParts | None
    residual_scale: float | None

    def __post_init__(self) -> None:
        if not self.team_id:
            raise ValueError("team_id must be non-empty")
        if (
            isinstance(self.population, bool)
            or not isinstance(self.population, int)
            or self.population < 1
        ):
            raise ValueError("population must be a positive integer")
        digest = self.context_model_sha256
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("Context 1.3 model hash must be lowercase SHA-256")
        pmf = np.asarray(self.context_prior_pmf, dtype=float)
        if (
            pmf.ndim != 1
            or len(pmf) != self.population
            or not np.isfinite(pmf).all()
            or np.any(pmf < 0)
            or not np.isclose(pmf.sum(), 1.0, rtol=0, atol=1e-8)
        ):
            raise ValueError("Context 1.3 prior PMF is invalid")
        object.__setattr__(self, "context_prior_pmf", pmf.copy())
        if self.component_status == FITTED_STATUS:
            if self.location_parts is None:
                raise ValueError("fitted Context priors require fitted location parts")
            if (
                self.residual_scale is None
                or not np.isfinite(self.residual_scale)
                or self.residual_scale <= 0
            ):
                raise ValueError(
                    "fitted Context priors require a positive residual scale"
                )
        elif self.component_status == COLD_START_STATUS:
            if self.location_parts is not None or self.residual_scale is not None:
                raise ValueError("cold-start fallbacks cannot synthesize fitted terms")
        else:
            raise ValueError(
                f"unsupported Context component status: {self.component_status}"
            )


@dataclass(frozen=True)
class Context14CandidatePrior:
    """A candidate PMF plus the machine-readable identity used to construct it."""

    team_id: str
    target_season: int
    trained_through_season: int
    context_model_sha256: str
    candidate_spec_sha256: str
    pmf: np.ndarray

    def artifact_bytes(self) -> bytes:
        """Serialize deterministically, including the identity hash for provenance."""
        return _canonical_json(
            {
                "candidate_spec_sha256": self.candidate_spec_sha256,
                "context_model_sha256": self.context_model_sha256,
                "model_family": "context_prior",
                "spec_version": "1.4-candidate",
                "target_season": self.target_season,
                "trained_through_season": self.trained_through_season,
                "team_id": self.team_id,
                "pmf": np.asarray(self.pmf, dtype=float).tolist(),
                "pmf_sha256": sha256_json(np.asarray(self.pmf, dtype=float).tolist()),
            }
        )


def context13_prior_from_fitted_contributions(
    *,
    fitted_instance: AnnualFittedInstance,
    context_model_sha256: str,
    team_id: str,
    component_status: str,
    population: int,
    context_prior_pmf: np.ndarray,
    residual_scale: float | None,
    contribution_row: Mapping[str, str] | None = None,
    conditional_location_points: np.ndarray | None = None,
) -> Context13PriorInput:
    """Build typed inputs from the same fitted contribution fields used in #159."""
    parts: LocationParts | None = None
    if component_status == FITTED_STATUS:
        if contribution_row is None or conditional_location_points is None:
            raise ValueError(
                "fitted Context priors require contribution and location data"
            )
        parts = parts_from_fitted_contributions(
            contribution_row, conditional_location_points
        )
    elif contribution_row is not None or conditional_location_points is not None:
        raise ValueError("cold-start fallbacks must not carry fitted contribution data")
    return Context13PriorInput(
        fitted_instance=fitted_instance,
        context_model_sha256=context_model_sha256,
        team_id=team_id,
        component_status=component_status,
        population=population,
        context_prior_pmf=context_prior_pmf,
        location_parts=parts,
        residual_scale=residual_scale,
    )


def construct_candidate_prior(prior: Context13PriorInput) -> Context14CandidatePrior:
    """Construct the sole frozen Context 1.4 candidate, without an alpha argument."""
    instance = prior.fitted_instance
    if (
        instance.model_family != "context_prior"
        or instance.spec_version != CONTEXT_1_3_VERSION
    ):
        raise ValueError("candidate input must be a Context 1.3 fit")
    if (
        isinstance(instance.target_season, bool)
        or isinstance(instance.trained_through_season, bool)
        or instance.target_season <= 1
        or instance.trained_through_season != instance.target_season - 1
    ):
        raise ValueError("Context 1.3 fit is not valid rolling-origin for its target")

    spec = load_candidate_spec()
    spec_hash = candidate_spec_sha256(spec)
    base_pmf = np.asarray(prior.context_prior_pmf, dtype=float)
    if prior.component_status == COLD_START_STATUS:
        candidate_pmf = base_pmf.copy()
    else:
        assert prior.location_parts is not None
        assert prior.residual_scale is not None
        reconstructed = normal_mixture_pmf(
            prior.location_parts.conditional_location_points,
            prior.residual_scale,
            prior.population,
        )
        if not np.allclose(reconstructed, base_pmf, rtol=0, atol=PMF_TOLERANCE):
            raise ValueError("Context 1.3 PMF does not match its fitted prior inputs")
        subtotal = prior.location_parts.context_only_subtotal
        if subtotal <= 0:
            # Preserve the original artifact exactly for zero and negative terms.
            candidate_pmf = base_pmf.copy()
        else:
            alpha = float(spec["alpha"])
            locations = moderated_location_points(prior.location_parts, alpha)
            candidate_pmf = normal_mixture_pmf(
                locations, prior.residual_scale, prior.population
            )

    return Context14CandidatePrior(
        team_id=prior.team_id,
        target_season=instance.target_season,
        trained_through_season=instance.trained_through_season,
        context_model_sha256=prior.context_model_sha256,
        candidate_spec_sha256=spec_hash,
        pmf=candidate_pmf.copy(),
    )


__all__ = [
    "CANDIDATE_SPEC_PATH",
    "COLD_START_STATUS",
    "FITTED_STATUS",
    "Context13PriorInput",
    "Context14CandidatePrior",
    "candidate_spec_sha256",
    "construct_candidate_prior",
    "context13_prior_from_fitted_contributions",
    "load_candidate_spec",
    "normal_mixture_pmf",
    "sha256_json",
]
