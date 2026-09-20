from __future__ import annotations

import json
from pathlib import Path

from gippyrank.site_preseason_evidence import build_preseason_input_projection

ROOT = Path(__file__).resolve().parents[1]


def _metadata(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative / "metadata.json").read_text(encoding="utf-8"))


def _fields(group: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(field["id"]): field for field in group["fields"]}  # type: ignore[index]


def test_context_1_3_projection_exposes_actual_team_preseason_evidence() -> None:
    projection = build_preseason_input_projection(
        root=ROOT,
        metadata=_metadata(
            "data/processed/snapshots/2026/2026-preseason-context-v1.3/predictive/context"
        ),
        team_ids=["194"],
    )

    assert projection is not None
    assert projection["schema_version"] == "1.0"
    ohio_state = projection["teams"]["194"]  # type: ignore[index]
    assert set(ohio_state) == {
        "program_history",
        "coach",
        "recruiting",
        "talent",
        "transfers",
    }
    recruiting = _fields(ohio_state["recruiting"])
    assert recruiting["recruiting_class_rank"]["raw_value"] == 4.0
    assert recruiting["recruiting_class_rank"]["comparison"]["fbs_rank"] == 4
    assert recruiting["recruiting_points_4y_mean"]["model_feature"] == (
        "recruiting_points_4y_mean"
    )
    transfers = _fields(ohio_state["transfers"])
    assert transfers["transfer_in_prior_usage_sum"]["raw_value"] == 0.371
    assert (
        "4/4"
        in transfers["transfer_in_prior_defensive_impact_db_available"]["display_value"]
    )
    caveat = projection["provenance"]["transfer_caveat"]  # type: ignore[index]
    assert caveat["status"] == "retrospective_reconstruction"
    assert "after the Aug. 15 cutoff" in caveat["visible_label"]


def test_history_projection_exposes_program_history_only() -> None:
    projection = build_preseason_input_projection(
        root=ROOT,
        metadata=_metadata(
            "data/processed/snapshots/2026/2026-preseason-history-context-1.3/predictive/history"
        ),
        team_ids=["194"],
    )

    assert projection is not None
    ohio_state = projection["teams"]["194"]  # type: ignore[index]
    assert set(ohio_state) == {"program_history"}
    fields = _fields(ohio_state["program_history"])
    assert fields["season_2025_consensus_rank"]["model_feature"] == (
        "lag1_rank_distribution"
    )
    assert fields["long_run_program_level"]["model_feature"] == "long_run_z_mean"
    assert "transfer_caveat" not in projection["provenance"]
