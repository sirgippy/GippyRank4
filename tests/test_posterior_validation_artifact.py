from __future__ import annotations

import json
from pathlib import Path


def test_exact_bp_validation_artifact_meets_predeclared_threshold() -> None:
    artifact = json.loads(
        Path("data/processed/posterior_validation/bp_exact_validation.json").read_text()
    )
    summary = artifact["summary"]
    assert summary["case_count"] == 72
    assert summary["convergence_rate"] == 1.0
    assert (
        summary["p95_marginal_tv"]
        <= summary["predeclared_acceptance"]["p95_marginal_tv_lte"]
    )
    assert (
        summary["worst_marginal_tv"]
        <= summary["predeclared_acceptance"]["worst_marginal_tv_lte"]
    )
    assert summary["accepted"] is True
