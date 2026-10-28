"""Scientific coverage and numerical consistency of the retained frozen study."""

from __future__ import annotations

import json
import re
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


@pytest.fixture(scope="module")
def interpreted_cases():
    rows = [numeric_rank(r) for r in read_csv(SAVED / "case_results.csv")]
    for r in rows:
        for key in (
            "categories",
            "questions",
            "actual_games",
            "answers_after_observation",
            "evolution_across_snapshots",
        ):
            r[key] = json.loads(r[key])
        for key in (
            "average_margin",
            "expected_wins",
            "wins_minus_expectation",
            "schedule_reference_mean_win_probability",
        ):
            r[key] = float(r[key])
    return rows


def answer_for(row, category):
    return next(
        a["answer"]
        for a in row["answers_after_observation"]
        if a["category"] == category
    )


def assert_reported_numbers(answer, values, precision):
    """Check evidence values, without pinning narrative phrasing or punctuation."""
    tokens = re.findall(r"[-+]?\d+\.\d+", answer)
    actual = {float(t) for t in tokens}
    assert {float(f"{float(v):.{precision}f}") for v in values} <= actual


def test_each_frozen_question_has_its_own_category_without_keyword_routing(
    interpreted_cases,
):
    for r in interpreted_cases:
        expected = [*r["categories"], *(["postseason"] if r["season"] == 2025 else [])]
        assert [a["category"] for a in r["answers_after_observation"]] == expected
        assert [a["question"] for a in r["answers_after_observation"]] == r["questions"]
        assert all(
            a["answer"] == a["answer"].strip() for a in r["answers_after_observation"]
        )
        misleading = {
            **r,
            "questions": ["uncertainty boundary postseason"] * len(expected),
        }
        assert analysis.case_question_categories(misleading) == expected
    r = interpreted_cases[0]
    candidates = [
        c
        for c in interpreted_cases
        if (c["snapshot_id"], c["team_id"]) == (r["snapshot_id"], r["team_id"])
    ]
    with pytest.raises(ValueError, match="unhandled frozen"):
        analysis.case_answer(
            "unregistered_category", r, candidates=candidates, panels={}
        )
    with pytest.raises(ValueError, match="question/category"):
        analysis.case_question_categories({**r, "questions": []})


@pytest.mark.parametrize("team_id,category", [("2117", "weak"), ("333", "elite")])
def test_weak_and_elite_questions_use_their_completed_controlled_responses(
    interpreted_cases, team_id, category
):
    probes = read_csv(SAVED / "outcome_sensitivity.csv")
    for r in interpreted_cases:
        if r["team_id"] != team_id or r["snapshot_id"] != "2026-10-25T12-59-22.246522Z":
            continue
        p = next(
            p
            for p in probes
            if p["focal_team_id"] == team_id
            and float(p["margin_weight"]) == r["margin_weight"]
        )
        answer = answer_for(r, category)
        assert p["game_id"] in answer
        assert_reported_numbers(
            answer,
            [
                p[k]
                for k in (
                    "outcome_boundary_benefit",
                    "within_win_margin_benefit",
                    "outcome_minus_margin_benefit",
                )
            ],
            3,
        )
        assert_reported_numbers(
            answer, [p[k] for k in ("margin_-3_rank_sd", "margin_+3_rank_sd")], 2
        )
        assert_reported_numbers(
            answer,
            [p[k] for k in ("margin_-3_entropy_bits", "margin_+3_entropy_bits")],
            3,
        )
        # Counterfactual figures must be data-driven, not embedded conclusions.
        changed = {**p, "snapshot_id": r["snapshot_id"]}
        for k in (
            "outcome_boundary_benefit",
            "within_win_margin_benefit",
            "margin_-3_rank_sd",
            "margin_+3_rank_sd",
            "margin_-3_entropy_bits",
            "margin_+3_entropy_bits",
        ):
            changed[k] = float(p[k])
        changed["outcome_boundary_benefit"] += 123.456
        rendered = analysis.controlled_response_text(changed, None)
        assert_reported_numbers(rendered, [changed["outcome_boundary_benefit"]], 3)


def test_stanford_disagreement_addresses_games_and_endpoint_uncertainty_separately(
    interpreted_cases,
):
    for r in interpreted_cases:
        if r["team_id"] != "24" or r["season"] != 2026:
            continue
        answer = answer_for(r, "endpoint_disagreement")
        assert "Massachusetts" not in answer
        for g in r["actual_games"]:
            assert g["opponent_name"] in answer
            assert_reported_numbers(answer, [g["frozen_primary_win_probability"]], 3)
        endpoints = [
            c
            for c in interpreted_cases
            if c["snapshot_id"] == r["snapshot_id"]
            and c["team_id"] == r["team_id"]
            and c["margin_weight"] in (0, 1)
        ]
        assert_reported_numbers(
            answer, [c[k] for c in endpoints for k in ("expected_rank", "rank_sd")], 2
        )
        assert "Massachusetts" in answer_for(r, "weak_schedule_pair")


def test_later_counterfactuals_are_not_borrowed_for_early_or_historical_cases(
    interpreted_cases,
):
    ids = {p["game_id"] for p in read_csv(SAVED / "outcome_sensitivity.csv")}
    for r in interpreted_cases:
        if r["snapshot_id"] == "2026-10-25T12-59-22.246522Z":
            continue
        assert all(
            gid not in a["answer"]
            for a in r["answers_after_observation"]
            for gid in ids
        )


def test_schedule_questions_use_the_retained_population_or_frozen_peer(
    interpreted_cases,
):
    panels = {
        (r["snapshot_id"], float(r["margin_weight"]), r["panel"]): r
        for r in read_csv(ROOT / BASELINE_OUTPUT / "uncertainty_panels.csv")
    }
    for r in interpreted_cases:
        for category in set(r["categories"]) & {"weak", "middle"}:
            hard, easy = (
                panels[
                    r["snapshot_id"],
                    r["margin_weight"],
                    f"band/{category}/schedule/{s}",
                ]
                for s in ("hard", "easy")
            )
            answer = answer_for(r, category)
            assert_reported_numbers(
                answer,
                [
                    float(easy[k]) - float(hard[k])
                    for k in ("expected_rank_mean", "rank_sd_mean", "width_80_mean")
                ],
                2,
            )
        for category in set(r["categories"]) & {
            "weak_schedule_pair",
            "middle_schedule_pair",
        }:
            peer = next(
                c
                for c in interpreted_cases
                if c["snapshot_id"] == r["snapshot_id"]
                and c["margin_weight"] == r["margin_weight"]
                and c["team_id"] != r["team_id"]
                and category in c["categories"]
            )
            answer = answer_for(r, category)
            assert peer["team_name"] in answer
            assert_reported_numbers(
                answer, [peer[k] for k in ("expected_rank", "rank_sd")], 2
            )


def test_game_boundary_and_postseason_questions_use_their_specific_evidence(
    interpreted_cases,
):
    removals = read_csv(SAVED / "game_diagnostics.csv")
    categories = {
        "competitive_elite_loss",
        "narrow_win_weaker_opponent",
        "dominant_win_overmatched_opponent",
    }
    for r in interpreted_cases:
        for category in set(r["categories"]) & categories:
            if r["snapshot_id"] == "2026-10-25T12-59-22.246522Z":
                d = next(
                    d
                    for d in removals
                    if d["team_id"] == r["team_id"]
                    and float(d["margin_weight"]) == r["margin_weight"]
                    and d["kind"] == "removal"
                    and d["role"] == "focal"
                )
                g = next(g for g in r["actual_games"] if g["game_id"] == d["game_id"])
                answer = answer_for(r, category)
                assert d["game_id"] in answer and g["opponent_name"] in answer
                assert_reported_numbers(
                    answer, [d["delta_expected_rank"], d["total_variation"]], 3
                )
        for category in r["categories"]:
            if category.startswith(("boundary_", "reference_boundary_")):
                k = category.rsplit("_", 1)[1]
                assert_reported_numbers(answer_for(r, category), [r[f"mass_le_{k}"]], 3)
        if r["season"] == 2025:
            answer = answer_for(r, "postseason")
            e = r["evolution_across_snapshots"][0]
            assert_reported_numbers(
                answer, [e["delta_expected_rank"], e["delta_rank_sd"]], 2
            )
            assert all(g["opponent_name"] in answer for g in e["added_own_games"])
