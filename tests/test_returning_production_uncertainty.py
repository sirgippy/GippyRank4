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
    c2_vs_c1_comparison,
    merge_h_fallback,
)


def prediction(
    team_id: str, model: str, pmf: np.ndarray | None = None
) -> v1.PriorPrediction:
    return v1.PriorPrediction(
        season=2022,
        subdivision="fbs",
        team_id=team_id,
        team_name=team_id,
        population=2,
        target_ranks=np.asarray([1]),
        model=model,
        prior_method="same_subdivision_lag1",
        pmf=np.asarray([0.5, 0.5]) if pmf is None else pmf,
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


def test_c2_vs_c1_comparison_reports_paired_and_bootstrap_deltas() -> None:
    rows, summary = c2_vs_c1_comparison(
        {
            "C1_no_rp": [prediction("a", "C1", np.asarray([0.4, 0.6]))],
            "C2_rp_scale_only": [prediction("a", "C2", np.asarray([0.5, 0.5]))],
        }
    )
    assert len(rows) == 1
    assert rows[0]["delta_nll_c2_minus_c1"] < 0
    assert summary["paired_team_season"]["n_team_seasons"] == 1
    assert summary["season_bootstrap"]["n_resamples"] == 1
