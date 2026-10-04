"""Production identity for the validated Context 1.4 candidate.

The frozen candidate remains the sole mathematical implementation. Promotion
changes its lifecycle identity and publication routing, not its computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gippyrank.context_prior_v1_4_candidate import (
    FROZEN_CANDIDATE_SEMANTICS_SHA256,
    Context13PriorInput,
    Context14CandidatePrior,
    candidate_spec_sha256,
    construct_candidate_prior,
    load_candidate_spec,
    sha256_json,
)

CONTEXT_1_4_VERSION = "1.4"
MODERATION_ALPHA = 0.75
VALIDATION_RESULT_PATH = "data/processed/context_v1_4_validation/summary.json"
VALIDATION_PROVENANCE_PATH = "data/processed/context_v1_4_validation/provenance.json"


def production_spec() -> dict[str, object]:
    """Return the promoted identity, bound to the exact frozen candidate."""
    candidate = load_candidate_spec()
    if (
        candidate["alpha"] != MODERATION_ALPHA
        or candidate_spec_sha256() != FROZEN_CANDIDATE_SEMANTICS_SHA256
    ):
        raise ValueError("production Context 1.4 differs from the frozen candidate")
    return {
        "model_family": "context_prior",
        "spec_version": CONTEXT_1_4_VERSION,
        "status": "active_production",
        "base_model": "Context 1.3",
        "moderation_alpha": MODERATION_ALPHA,
        "moderation_target": "positive context_only_subtotal",
        "candidate_semantics_sha256": FROZEN_CANDIDATE_SEMANTICS_SHA256,
        "validation": "Issue #168 / PR #169",
        "validation_decision": "promote",
    }


def production_semantics_sha256() -> str:
    """Identify production semantics independently of artifact serialization."""
    return sha256_json(
        {
            "model_family": "context_prior",
            "spec_version": CONTEXT_1_4_VERSION,
            "base_model": "Context 1.3",
            "moderation_alpha": MODERATION_ALPHA,
            "moderation_target": "positive context_only_subtotal",
            "candidate_semantics_sha256": candidate_spec_sha256(),
        }
    )


@dataclass(frozen=True)
class Context14ProductionPrior:
    """A production lifecycle view of the exact candidate PMF."""

    candidate: Context14CandidatePrior

    @property
    def pmf(self) -> np.ndarray:
        return self.candidate.pmf.copy()

    @property
    def team_id(self) -> str:
        return self.candidate.team_id

    @property
    def candidate_semantics_sha256(self) -> str:
        return self.candidate.candidate_semantics_sha256

    @property
    def production_semantics_sha256(self) -> str:
        return production_semantics_sha256()


def construct_production_prior(prior: Context13PriorInput) -> Context14ProductionPrior:
    """Compute a production prior with the frozen validated candidate builder."""
    production_spec()
    candidate = construct_candidate_prior(prior)
    if candidate.candidate_semantics_sha256 != FROZEN_CANDIDATE_SEMANTICS_SHA256:
        raise ValueError("candidate semantic identity changed during promotion")
    return Context14ProductionPrior(candidate)
