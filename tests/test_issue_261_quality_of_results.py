from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gippyrank.posterior.engine import LikelihoodV1, Team
from gippyrank.research import issue_252_quality_of_results as original
from gippyrank.research import issue_261_quality_of_results as study

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def design() -> study.Design:
    return study.load_design(ROOT)


def test_frozen_inputs_match_complete_design_contract(design: study.Design) -> None:
    assert set(design.focal) == {*study.PROFILES, "issue_252_reference"}
    assert design.records == {
        "elite": (6, 8),
        "strong": (5, 8),
        "middle": (5, 8),
        "weak": (2, 4),
    }
    for profile in study.PROFILES:
        assert len(design.focal[profile].prior) == 138
        for slots in design.schedules[profile].values():
            assert tuple(slot.venue for slot in slots) == original.SITE_SLOTS
            assert len({slot.team.team_id for slot in slots}) == 10
            assert all(len(slot.team.prior) == 138 for slot in slots)


def test_modified_frozen_file_is_rejected(tmp_path: Path) -> None:
    source = ROOT / "data/research/issue_258_quality_of_results"
    target = tmp_path / "data/research/issue_258_quality_of_results"
    shutil.copytree(source, target)
    with (target / "schedule_games.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="frozen artifact hash mismatch"):
        study.load_design(tmp_path)


def test_conditioning_stops_at_frozen_ceiling_and_never_changes_records(
    design: study.Design, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = json.loads(json.dumps(design.manifest))
    manifest["follow_up_plan"]["conditioning_search_ceiling_per_panel"] = 3
    shortened = replace(design, manifest=manifest)
    seen = []

    def trial(seed, *args):
        seen.append(seed)
        return SimpleNamespace(
            groups={
                "approximately_even": SimpleNamespace(wins=6),
                "weaker_opponents": SimpleNamespace(wins=8),
                "same_schedule_a": SimpleNamespace(wins=5),
                "same_schedule_b": SimpleNamespace(wins=5),
            }
        )

    monkeypatch.setattr(study, "generate_trial", trial)
    primary = study.select_seeds(shortened, "elite", "primary_conditioned")
    assert primary["seeds"] == [2710000, 2710001, 2710002]
    assert not primary["complete"] and primary["attempts"] == 3
    control = study.select_seeds(shortened, "elite", "control_conditioned")
    assert not control["seeds"] and control["attempts"] == 3
    assert seen == [2710000, 2710001, 2710002, 2810000, 2810001, 2810002]


def test_all_candidates_receive_identical_evidence_and_uncertain_opponents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = np.array([0.2, 0.5, 0.3])
    slots = tuple(
        original.OpponentSlot(
            Team(str(i), f"Opponent {i}", "fbs", prior), venue, 0.5, 0.5
        )
        for i, venue in enumerate(original.SITE_SLOTS)
    )
    likelihood = LikelihoodV1(np.zeros(34), 7.0, 15.0)
    trial = original.generate_trial(
        261,
        prior,
        {
            "approximately_even": slots,
            "weaker_opponents": slots,
        },
        likelihood,
    )
    snapshots = []
    infer = original.infer_quality_of_results

    def capture(teams, games, *args, **kwargs):
        snapshots.append(
            (tuple((team.team_id, tuple(team.prior)) for team in teams), tuple(games))
        )
        return infer(teams, games, *args, **kwargs)

    monkeypatch.setattr(original, "infer_quality_of_results", capture)
    before = study.evidence_hash(trial)
    teams, pairs, pmfs = original._infer_trial(trial, likelihood)
    assert len(snapshots) == 5 and all(item == snapshots[0] for item in snapshots)
    assert before == study.evidence_hash(trial)
    assert len(teams) == 20 and len(pairs) == 10
    for groups in pmfs.values():
        np.testing.assert_allclose(
            groups["approximately_even"], groups["weaker_opponents"]
        )


def test_checkpoint_rejects_changed_source_or_seed(tmp_path: Path) -> None:
    path = tmp_path / "trial.json"
    study.write_json(path, {"fingerprint": "original", "seed": 7})
    with pytest.raises(ValueError, match="provenance differs"):
        study.load_checkpoint(path, "changed", 7)
    with pytest.raises(ValueError, match="provenance differs"):
        study.load_checkpoint(path, "original", 8)


def test_completed_artifact_contract() -> None:
    """Audit retained results without executing expensive inference in CI."""
    output = ROOT / "data/research/issue_261_quality_of_results"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_revision"] == study.PINNED_EXECUTION_REVISION
    for name, expected in manifest["artifact_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert manifest["counts"] == {
        "team_candidate_rows": 64000,
        "pair_candidate_rows": 32000,
        "simulated_game_rows": 128000,
    }
    for panels in manifest["completed_panels"].values():
        assert len(panels["unconditional"]["seeds"]) == 500
        assert len(panels["primary_conditioned"]["seeds"]) == 200
        assert len(panels["control_conditioned"]["seeds"]) == 200
        assert all(row["complete"] for row in panels.values())
        assert all(row["attempts"] <= 20000 for row in panels.values())
    rows = study.read_csv(output / "simulation_summary.csv")
    assert len(rows) == 100
    assert {row["profile"] for row in rows} == set(study.PROFILES)
    assert {float(row["margin_weight"]) for row in rows} == set(original.MARGIN_WEIGHTS)
    assert (
        sum(row["comparison_population"] == "primary_seed_control" for row in rows)
        == 20
    )


def test_retained_evidence_records_and_candidate_pairing() -> None:
    """Audit the scientific populations from portable game and trial tables."""
    from collections import Counter, defaultdict

    output = ROOT / "data/research/issue_261_quality_of_results"
    records = {"elite": (6, 8), "strong": (5, 8), "middle": (5, 8), "weak": (2, 4)}
    evidence = defaultdict(list)
    for row in study.read_csv(output / "simulated_games.csv.gz"):
        evidence[row["profile"], row["panel"], row["seed"], row["focal_group"]].append(
            row
        )
        margin = float(row["synthetic_focal_margin"])
        assert np.isfinite(margin)
        assert (row["focal_won"] == "True") == (margin > 0)
        home_margin = float(row["synthetic_home_margin"])
        assert margin == (-home_margin if row["venue"] == "away" else home_margin)
    candidates = defaultdict(list)
    trial_hashes = defaultdict(set)
    for row in study.read_csv(output / "team_trial_summaries.csv.gz"):
        key = row["profile"], row["panel"], row["seed"], row["focal_group"]
        candidates[key].append(float(row["margin_weight"]))
        trial_hashes[key[:3]].add(row["evidence_sha256"])
        games = evidence[key]
        assert len(games) == 10
        assert int(row["wins"]) == sum(game["focal_won"] == "True" for game in games)
        assert {int(game["focal_true_rank"]) for game in games} == {
            int(row["focal_true_rank"])
        }
        assert row["converged"] == "True" and int(row["iterations"]) <= 500
        assert tuple(game["venue"] for game in games) == original.SITE_SLOTS
        if row["panel"] == "primary_conditioned" and row["focal_group"] in {
            "approximately_even",
            "weaker_opponents",
        }:
            assert (
                int(row["wins"])
                == records[row["profile"]][row["focal_group"] == "weaker_opponents"]
            )
        if row["panel"] == "control_conditioned":
            assert row["focal_group"].startswith("same_schedule_")
            assert int(row["wins"]) == records[row["profile"]][0]
    assert len(candidates) == 12800 and len(trial_hashes) == 3600
    assert all(
        set(weights) == set(original.MARGIN_WEIGHTS) and len(weights) == 5
        for weights in candidates.values()
    )
    assert all(len(hashes) == 1 for hashes in trial_hashes.values())
    group_counts = Counter(key[:3] for key in candidates)
    assert all(
        count == (2 if key[1] == "control_conditioned" else 4)
        for key, count in group_counts.items()
    )
    # Matching is across observations, not just the metadata assertion.
    for profile, panel, seed in group_counts:
        names = ("same_schedule_a", "same_schedule_b")
        if panel != "control_conditioned":
            names = ("approximately_even", "weaker_opponents", *names)
        groups = {name: evidence[profile, panel, seed, name] for name in names}
        assert len({games[0]["focal_true_rank"] for games in groups.values()}) == 1
        if panel == "control_conditioned":
            continue
        lower, higher = groups["approximately_even"], groups["weaker_opponents"]
        assert [row["standardized_margin_residual"] for row in lower] == [
            row["standardized_margin_residual"] for row in higher
        ]


def test_retained_aggregate_means_match_paired_trials() -> None:
    from collections import defaultdict

    output = ROOT / "data/research/issue_261_quality_of_results"
    paired = defaultdict(list)
    for row in study.read_csv(output / "pair_trial_summaries.csv.gz"):
        paired[
            row["profile"], row["panel"], row["comparison"], row["margin_weight"]
        ].append(row)
    for summary in study.read_csv(output / "simulation_summary.csv"):
        rows = paired[
            summary["profile"],
            summary["panel"],
            summary["comparison"],
            summary["margin_weight"],
        ]
        differences = np.array([float(row["rank_difference"]) for row in rows])
        count = int(summary["draw_count"])
        assert (
            len(rows) == count == (500 if summary["panel"] == "unconditional" else 200)
        )
        mean = float(differences.mean())
        half_width = 1.959963984540054 * float(differences.std(ddof=1)) / count**0.5
        assert float(summary["rank_mean"]) == pytest.approx(mean)
        assert float(summary["rank_mean_ci95_low"]) == pytest.approx(mean - half_width)
        assert float(summary["rank_mean_ci95_high"]) == pytest.approx(mean + half_width)
        assert float(summary["rank_difference_median"]) == pytest.approx(
            float(np.median(differences))
        )
        assert float(summary["right_outranks_left_probability"]) == pytest.approx(
            float(np.mean(differences < 0))
        )
        expected_label = (
            "primary_seed_control"
            if summary["panel"] == "primary_conditioned"
            and summary["comparison"] == "same_schedule_control"
            else summary["panel"]
        )
        assert summary["comparison_population"] == expected_label


def test_retained_pair_direction_and_distance_invariants() -> None:
    output = ROOT / "data/research/issue_261_quality_of_results"
    teams = {
        (
            row["profile"],
            row["panel"],
            row["seed"],
            row["margin_weight"],
            row["focal_group"],
        ): row
        for row in study.read_csv(output / "team_trial_summaries.csv.gz")
    }
    for row in study.read_csv(output / "pair_trial_summaries.csv.gz"):
        key = row["profile"], row["panel"], row["seed"], row["margin_weight"]
        left, right = teams[*key, row["left_group"]], teams[*key, row["right_group"]]
        delta = float(right["expected_rank"]) - float(left["expected_rank"])
        assert float(row["rank_difference"]) == pytest.approx(delta)
        assert (row["right_outranks_left"] == "True") == (delta < 0)
        if row["comparison"] == "schedule_pair":
            assert row["left_group"] == "approximately_even"
            assert row["right_group"] == "weaker_opponents"
        assert 0 <= float(row["pmf_total_variation"]) <= 1
        assert float(row["pmf_wasserstein_rank_distance"]) + 1e-9 >= abs(delta)


def test_report_refresh_preserves_simulation_and_is_deterministic(
    tmp_path: Path,
) -> None:
    from gippyrank.research.issue_261_report import render

    source = ROOT / "data/research/issue_261_quality_of_results"
    target = tmp_path / "results"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(".checkpoints"))
    original_manifest = json.loads((target / "manifest.json").read_text())
    render(ROOT, target)
    refreshed = json.loads((target / "manifest.json").read_text())
    # PNG compression/font rendering can vary between execution environments;
    # numerical inputs and report text must survive an inference-free refresh.
    assert {
        name: digest
        for name, digest in refreshed["artifact_sha256"].items()
        if name != "schedule_effects.png"
    } == {
        name: digest
        for name, digest in original_manifest["artifact_sha256"].items()
        if name != "schedule_effects.png"
    }
    render(ROOT, target)
    assert (
        json.loads((target / "manifest.json").read_text())["artifact_sha256"]
        == refreshed["artifact_sha256"]
    )
    assert refreshed["source_revision"] == original_manifest["source_revision"]
    assert len(study.read_csv(target / "candidate_contrasts.csv")) == 200
    report = (target / "report.md").read_text()
    assert "primary_seed_control" in report
    assert "Separation (pp)" in report
    assert "not bias estimates subtracted" in report
    assert "CPU time nor study wall time" in report
