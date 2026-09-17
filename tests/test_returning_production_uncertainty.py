import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_preseason_prior as v1
from investigate_returning_production_uncertainty import (
    ALL_CONTEXT_FEATURES,
    H_FEATURES,
    NON_RP_FEATURES,
    RP_FEATURES,
    VARIANTS,
    merge_h_fallback,
)


def prediction(team_id: str, model: str) -> v1.PriorPrediction:
    return v1.PriorPrediction(
        season=2022,
        subdivision="fbs",
        team_id=team_id,
        team_name=team_id,
        population=2,
        target_ranks=np.asarray([1]),
        model=model,
        prior_method="same_subdivision_lag1",
        pmf=np.asarray([0.5, 0.5]),
    )


def test_variants_preserve_context_equation_semantics() -> None:
    c0, c1, c2, c3 = VARIANTS
    assert c0.location_context == (*NON_RP_FEATURES, *RP_FEATURES)
    assert c0.scale_context == ()
    assert c1.location_context == NON_RP_FEATURES
    assert c1.scale_context == ()
    assert c2.location_context == NON_RP_FEATURES
    assert c2.scale_context == RP_FEATURES
    assert c3.location_context == (*NON_RP_FEATURES, *RP_FEATURES)
    assert c3.scale_context == RP_FEATURES
    assert c2.feature_names == [*H_FEATURES, *ALL_CONTEXT_FEATURES]


def test_merge_h_fallback_keeps_the_exact_target_keys() -> None:
    candidate = [prediction("candidate", "C2")]
    history = [prediction("candidate", "H"), prediction("cold", "H")]
    merged = merge_h_fallback(candidate, history)
    assert [item.team_id for item in merged] == ["candidate", "cold"]
    assert merged[0].model == "C2"
    assert merged[1].model == "H"
