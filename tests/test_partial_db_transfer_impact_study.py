"""Invariants for the research-only partial DB transfer-impact policies."""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from study_partial_db_transfer_impact import (
    candidate_values,
    mask_sets,
    run,
    shift_pmf,
    summarize_team_weighted,
)


def test_partial_policies_preserve_known_contribution() -> None:
    values = candidate_values([2.0, -0.5, 1.0], 1, 0.25)
    assert values == {
        "all_or_nothing": 0.0,
        "observed": 2.5,
        "coverage_scaled": 10.0 / 3.0,
        "missing_mean": 2.75,
    }
    assert candidate_values([], 2, 0.25)["observed"] == 0.0


def test_high_coverage_masks_hide_each_player_once() -> None:
    assert mask_sets(8, 7) == [(i,) for i in range(8)]
    capped = mask_sets(12, 6)
    assert len(capped) == 80
    assert len(set(capped)) == 80
    assert capped[0] == next(combinations(range(12), 6))
    assert capped[-1] == tuple(range(6, 12))
    assert capped == mask_sets(12, 6)


def test_team_weighted_summary_does_not_count_masks_as_teams() -> None:
    rows = [
        {
            "overall": "all",
            "policy": "observed",
            "season": 2022,
            "team_id": "a",
            "coverage": "1/2",
            "full_impact": 3.0,
            "estimate": 0.0,
        }
        for _ in range(3)
    ]
    rows.append(
        {
            "overall": "all",
            "policy": "observed",
            "season": 2022,
            "team_id": "b",
            "coverage": "1/2",
            "full_impact": 1.0,
            "estimate": 1.0,
        }
    )
    result = summarize_team_weighted(rows, "overall")[0]
    assert result["masks"] == 4
    assert result["team_coverage_cells"] == 2
    assert result["mae"] == 1.5


def test_skip_downstream_removes_stale_artifacts(tmp_path: Path) -> None:
    for name in ("current_complete_one_missing.csv", "downstream_one_missing.csv"):
        (tmp_path / name).write_text("stale", encoding="utf-8")
    metadata = run(tmp_path, with_downstream=False)
    assert metadata["downstream_model"] == "disabled"
    assert not (tmp_path / "current_complete_one_missing.csv").exists()
    assert not (tmp_path / "downstream_one_missing.csv").exists()
    assert (tmp_path / "report.md").exists()
    assert "matches 50 of 57 partially covered 2026 teams with observed values" in (
        tmp_path / "report.md"
    ).read_text(encoding="utf-8")


def test_downstream_translation_preserves_probability() -> None:
    original = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    np.testing.assert_allclose(shift_pmf(original, 0.0), original)
    shifted = shift_pmf(original, 0.25)
    np.testing.assert_allclose(shifted.sum(), 1.0)
    assert np.dot(np.arange(1, 6), shifted) > np.dot(np.arange(1, 6), original)
