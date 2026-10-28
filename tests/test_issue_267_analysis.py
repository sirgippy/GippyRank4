"""Scientific coverage and numerical consistency of the retained frozen study."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from gippyrank.posterior.snapshots import sha256
from gippyrank.research import issue_267_analysis as analysis
from gippyrank.research.issue_242_quality_of_results import MARGIN_WEIGHTS
from gippyrank.research.issue_263_real_season_design import ARTIFACT_DIR
from gippyrank.research.issue_265_real_seasons import (
    mean_rank,
    verify_completed_execution,
)
from gippyrank.research.issue_267_analysis import (
    adjacent_gaps,
    numeric_rank,
    read_csv,
    rich_compare,
    verify_completed_diagnostics,
)
from gippyrank.research.issue_267_diagnostics import (
    BASELINE_OUTPUT,
    OUTPUT,
    perturb,
)

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / OUTPUT


@pytest.fixture(scope="module")
def study():
    baseline = verify_completed_execution(ROOT, ROOT / BASELINE_OUTPUT)
    origin, jobs, cells = verify_completed_diagnostics(ROOT, SAVED, baseline)
    return baseline, origin, jobs, cells


def test_all_original_cases_and_games_and_rankings_are_retained(study):
    baseline, *_ = study
    cases = [
        *json.loads((ROOT / ARTIFACT_DIR / "cases.json").read_text()),
        *json.loads((ROOT / ARTIFACT_DIR / "historical_cases.json").read_text()),
    ]
    frozen = {
        (r.get("snapshot_id", r.get("publication_slot")), r["team_id"]): r
        for r in cases
    }
    original = {
        (r["snapshot_id"], r["team_id"], r["margin_weight"]): r
        for r in read_csv(ROOT / BASELINE_OUTPUT / "rankings.csv")
    }
    results = read_csv(SAVED / "case_results.csv")
    assert len(results) == 350
    assert len({(r["season"], r["team_id"]) for r in results}) == 27
    assert {(r["snapshot_id"], r["team_id"]) for r in results} == set(frozen)
    for r in results:
        key = r["snapshot_id"], r["team_id"]
        c = frozen[key]
        assert json.loads(r["frozen_selection"]) == c
        assert json.loads(r["categories"]) == c["categories"]
        assert json.loads(r["questions"]) == c["questions"]
        assert [a["question"] for a in json.loads(r["answers_after_observation"])] == c[
            "questions"
        ]
        games = json.loads(r["actual_games"])
        n = baseline.networks[r["snapshot_id"]]
        actual_ids = {
            g.game_id for g in n.games if r["team_id"] in (g.home_id, g.away_id)
        }
        assert {g["game_id"] for g in games} == actual_ids
        assert len(games) == c["games"]
        assert sum(g["margin"] > 0 for g in games) == c["wins"]
        assert sum(g["frozen_primary_win_probability"] for g in games) == pytest.approx(
            float(r["expected_wins"]), abs=1e-10
        )
        b = original[r["snapshot_id"], r["team_id"], r["margin_weight"]]
        assert all(r[k] == v for k, v in b.items())
    evolution = read_csv(SAVED / "case_evolution.csv")
    assert len(evolution) == 215
    assert all(
        int(r["added_network_game_count"]) == 55
        for r in evolution
        if r["season"] == "2025"
    )


def test_complete_populations_counts_and_opponents_include_fcs(study):
    baseline, _, jobs, cells = study
    effects = read_csv(SAVED / "network_changes.csv.gz")
    focused = read_csv(SAVED / "game_diagnostics.csv")
    assert len(effects) == 8970 and len(focused) == 130
    assert set(
        Counter((r["perturbation_key"], r["margin_weight"]) for r in effects).values()
    ) == {138}
    for job in jobs:
        before = baseline.networks[job.snapshot_id]
        after = perturb(before, job)
        for w in MARGIN_WEIGHTS:
            rows = [
                r
                for r in effects
                if r["perturbation_key"] == job.key and float(r["margin_weight"]) == w
            ]
            assert {r["team_id"] for r in rows} == set(before.fbs)
            for r in rows:
                assert int(r["eligible_games"]) == after.counts[r["team_id"]]
                expected = after.counts[r["team_id"]] - before.counts[r["team_id"]]
                assert int(r["delta_eligible_games"]) == expected
            roles = [
                r
                for r in focused
                if r["perturbation_key"] == job.key and float(r["margin_weight"]) == w
            ]
            assert {r["role"] for r in roles} == {"focal", "opponent"}
            if job.game_id == "401858455":
                opponent = next(r for r in roles if r["role"] == "opponent")
                assert opponent["subdivision"] == "fcs"
                assert opponent["posterior_family"] == "context_on_fcs_support"
                assert opponent["ordinal_rank"] == opponent["delta_ordinal_rank"] == ""
                assert float(opponent["expected_rank"]) == pytest.approx(
                    mean_rank(np.array(cells[job.key, w]["context"][job.opponent_id])),
                    abs=1e-10,
                )


def test_controlled_benefits_are_computed_from_full_pmfs_and_endpoint_agrees(study):
    _, _, jobs, cells = study
    rows = read_csv(SAVED / "outcome_sensitivity.csv")
    assert len(rows) == 10
    for row in rows:
        gid, w, t = row["game_id"], float(row["margin_weight"]), row["focal_team_id"]
        pmfs = {
            m: np.array(cells[f"replacement-{gid}-{m:+d}", w]["pmfs"][t])
            for m in (-3, 3, 10, 35)
        }
        benefit = mean_rank(pmfs[-3]) - mean_rank(pmfs[3])
        margin = mean_rank(pmfs[10]) - mean_rank(pmfs[35])
        assert float(row["outcome_boundary_benefit"]) == pytest.approx(
            benefit, abs=1e-10
        )
        assert float(row["within_win_margin_benefit"]) == pytest.approx(
            margin, abs=1e-10
        )
        assert float(row["outcome_minus_margin_benefit"]) == pytest.approx(
            benefit - margin, abs=1e-10
        )
        if w == 0:
            for m in (10, 35):
                assert np.max(abs(pmfs[3] - pmfs[m])) <= 1e-8
    assert len(jobs) == 13


def test_portable_reader_preserves_runtime_identity_and_rejects_source_drift(
    study, monkeypatch
):
    baseline, origin, _, _ = study
    current = analysis.provenance

    def other_runtime(*args):
        p = current(*args)
        p["environment"] = {
            "platform": "a different CI reader",
            "python": "another reader",
        }
        return p

    monkeypatch.setattr(analysis, "provenance", other_runtime)
    consumed, _, _ = analysis.verify_completed_diagnostics(ROOT, SAVED, baseline)
    assert consumed == origin

    def other_source(*args):
        p = other_runtime(*args)
        p["source_sha256"] = {**p["source_sha256"], "changed-source": "changed"}
        return p

    monkeypatch.setattr(analysis, "provenance", other_source)
    with pytest.raises(ValueError, match="input/configuration/source"):
        analysis.verify_completed_diagnostics(ROOT, SAVED, baseline)


def test_manifest_covers_every_artifact_and_current_analysis_sources():
    manifest = json.loads((SAVED / "manifest.json").read_text())
    actual = {
        str(p.relative_to(SAVED))
        for p in SAVED.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    assert set(manifest["output_sha256"]) == actual
    assert all(
        sha256(SAVED / p) == digest for p, digest in manifest["output_sha256"].items()
    )
    assert all(
        sha256(ROOT / p) == digest
        for p, digest in manifest["analysis_source_sha256"].items()
    )


def test_uncertainty_distances_and_boundary_conventions():
    p, q = np.array([0.25, 0.5, 0.25]), np.array([0.5, 0.5, 0])
    d = rich_compare(p, q)
    assert d["delta_expected_rank"] == -0.5
    assert d["total_variation"] == 0.25
    assert d["wasserstein_rank"] == 0.5
    assert d["delta_high_95"] == -1
    assert d["delta_width_95"] == -1
    assert adjacent_gaps("b", {"a": 1, "b": 2}, {"a": 10.0, "b": 10.001})[
        "gap_to_better"
    ] == pytest.approx(0.001)
    assert numeric_rank({"ordinal_rank": "", "rank_sd": "1.2"})["ordinal_rank"] is None
