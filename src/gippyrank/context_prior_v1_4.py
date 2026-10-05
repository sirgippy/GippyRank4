"""Active Context 1.4 prior with repaired incoming DB evidence (#172)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from gippyrank.context_db_repair import (
    ALPHA,
    DB_COVERAGE,
    DB_SUM,
    MODEL_FEATURES,
    production_pmf,
)
from gippyrank.context_prior import InferenceRow
from gippyrank.preseason import DirectRankModel

CONTEXT_1_4_VERSION = "1.4"
MODERATION_ALPHA = ALPHA


def production_spec() -> dict[str, object]:
    return {
        "model_family": "context_prior",
        "spec_version": CONTEXT_1_4_VERSION,
        "status": "active_production",
        "correction": "issue_172_repaired_db_transfer_coverage",
        "feature_names": list(MODEL_FEATURES),
        "db_features": [DB_SUM, DB_COVERAGE],
        "moderation_alpha": MODERATION_ALPHA,
        "moderation_target": "positive context_only_subtotal",
    }


def production_semantics_sha256() -> str:
    return hashlib.sha256(
        json.dumps(production_spec(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Context14ProductionPrior:
    team_id: str
    pmf: np.ndarray
    conditional_location_mean: float
    predictive_scale: float

    @property
    def production_semantics_sha256(self) -> str:
        return production_semantics_sha256()


def construct_production_prior(
    model: DirectRankModel, row: InferenceRow
) -> Context14ProductionPrior:
    """Consume the coverage-aware refit and its outcome-free 2026 input row."""
    row.require_no_target()
    if row.subdivision != "fbs" or row.lag1_z is None:
        raise ValueError("fitted production Context requires an FBS lag-one row")
    if row.lag_zs:
        raise ValueError("Context 1.4 uses only the lag-one rank distribution")
    pmf, center, scale = production_pmf(
        model, row.features, np.asarray(row.lag1_z, dtype=float), row.population
    )
    return Context14ProductionPrior(row.team_id, pmf, center, scale)
