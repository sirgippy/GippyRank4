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
    if not (output / "manifest.json").exists():
        pytest.skip("full study artifacts not yet materialized")
    manifest = json.loads((output / "manifest.json").read_text())
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
