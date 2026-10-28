from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from gippyrank.artifact_hashes import included_game_rows_sha256
from gippyrank.research import issue_263_historical_evidence as historical
from gippyrank.research import issue_263_real_season_design as design

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / design.ARTIFACT_DIR


def test_archives_recovered_from_other_worktrees_are_portable_and_deduplicated():
    rows, audit = historical.normalized_archive(FROZEN / "raw")
    assert audit["raw_response_counts"] == {"fbs": 934, "fcs": 879}
    assert audit["overlap_count"] == 126
    assert len(rows) == audit["unique_game_count"] == 1687
    assert audit["full_season_fcs_population_size"] == 129
    assert len(audit["reviewed_rematches"]) == 16
    assert len({row["id"] for row in rows}) == len(rows)


def test_separate_historical_populations_preserve_cutoffs_and_real_rematches():
    before_teams, before_games, before = historical.load_historical_network(
        FROZEN / "inputs/2025-prebowl"
    )
    final_teams, final_games, final = historical.load_historical_network(
        FROZEN / "inputs/2025-final"
    )
    assert len(before_games) == 1570
    assert len(final_games) == 1625
    assert len(before_teams) == len(final_teams) == 265
    assert len(before["fbs_team_ids"]) == len(final["fbs_team_ids"]) == 136
    assert all(
        len(team.prior) == (136 if team.subdivision == "fbs" else 129)
        for team in final_teams
    )
    assert before["publication_id"] is None
    assert final["publication_id"] is None
    assert before["context_prior_version"] == final["context_prior_version"] == "1.2"
    assert not before["source_available_at_cutoff_verified"]
    assert before["source_acquisition_retrieved_at"] is None
    before_ids = {g.game_id for g in before_games}
    final_ids = {g.game_id for g in final_games}
    assert before_ids < final_ids
    assert len(final_ids - before_ids) == 55
    assert "401762521" not in before_ids  # Army–Navy occurs after pre-bowl cutoff.
    assert "401762521" in final_ids
    assert "401769076" not in before_ids  # National championship.
    assert "401769076" in final_ids
    assert {
        "401754593",
        "401777328",
    } <= before_ids  # Duke–Virginia plus ACC title rematch.
    for snapshot in (before, final):
        rows = design._read_rows(
            FROZEN / snapshot["input_directory"] / "included_games.csv"
        )
        assert included_game_rows_sha256(rows) == snapshot["included_game_rows_sha256"]
        assert all(
            datetime.fromisoformat(r["startDate"])
            <= datetime.fromisoformat(snapshot["latest_eligible_kickoff"])
            for r in rows
        )
        assert snapshot["quality_of_play_reference"] is None
        assert snapshot["quality_of_record_reference"] is None


def test_changed_archive_cannot_be_used(tmp_path):
    shutil.copytree(FROZEN / "raw", tmp_path / "raw")
    target = tmp_path / "raw/2025.json"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="historical archive hash mismatch"):
        historical.normalized_archive(tmp_path / "raw")


def test_changed_historical_prior_cannot_silently_change_inference(tmp_path):
    shutil.copytree(FROZEN / "inputs/2025-final", tmp_path / "final")
    target = tmp_path / "final/prior_pmfs.json"
    values = json.loads(target.read_text())
    values[0]["pmf"][0], values[0]["pmf"][1] = values[0]["pmf"][1], values[0]["pmf"][0]
    target.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="historical frozen prior mismatch"):
        historical.load_historical_network(tmp_path / "final")


def test_historical_cases_are_selected_before_postseason_or_candidate_outputs():
    rows = design._read_rows(FROZEN / "historical_reference_descriptors.csv")
    before = [r for r in rows if r["snapshot_id"] == "2025-prebowl"]
    for row in before:
        for key in ("games", "wins", "losses", "reference_ordinal"):
            row[key] = int(row[key])
        for key in ("win_fraction", "schedule_reference_mean_win_probability"):
            row[key] = float(row[key])
    selected = historical.select_historical_cases(before)
    assert selected == historical.select_historical_cases(before[::-1])
    assert len(selected) == 11
    cases = json.loads((FROZEN / "historical_cases.json").read_text())
    assert len(cases) == 22
    assert {r["selection_snapshot_id"] for r in cases} == {"2025-prebowl"}
    assert {r["reference_kind"] for r in cases} == {"preseason_context_v1.2"}
    assert {r["snapshot_id"] for r in cases} == {"2025-prebowl", "2025-final"}
    assert all(r["questions"] for r in cases)
    assert all("qop_ordinal" not in r and "qor_ordinal" not in r for r in cases)
