from __future__ import annotations

import json
from pathlib import Path


def test_serialized_likelihood_provenance_is_frozen_v1() -> None:
    artifact = json.loads(
        Path("data/processed/posterior/historical_likelihood_v1.json").read_text()
    )
    assert artifact["artifact_kind"] == "historical_likelihood_v1_coefficients"
    assert artifact["fit_kind"] == "weighted_pseudo"
    assert artifact["training_seasons"] == "2003-2021"
    assert artifact["excluded_evaluation_seasons"] == "2022-2025"
    assert artifact["student_t_df"] == 15.0
    assert artifact["degrees_of_freedom"] == 15.0
    assert len(artifact["beta"]) == 34
