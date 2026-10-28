"""Frozen evidence changes, cold-start endpoints and fail-closed checkpoint reuse."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gippyrank.posterior.engine import Game, Team
from gippyrank.posterior.snapshots import load_pinned_likelihood
from gippyrank.research import issue_267_diagnostics as runner
from gippyrank.research.issue_263_real_season_design import ARTIFACT_DIR
from gippyrank.research.issue_265_real_seasons import Network

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def verified():
    return runner.verify_completed_execution(ROOT, ROOT / runner.BASELINE_OUTPUT)


@pytest.fixture
def tiny():
    teams = [
        Team("a", "A", "fbs", np.array([0.2, 0.3, 0.5])),
        Team("b", "B", "fbs", np.array([0.5, 0.3, 0.2])),
        Team("c", "C", "fcs", np.array([0.3, 0.7])),
    ]
    games = [
        Game("g", "a", "b", "fbs", "fbs", 7, 0),
        Game("rematch", "b", "a", "fbs", "fbs", 14, 21),
        Game("fcs", "c", "a", "fcs", "fbs", 3, 14),
    ]
    rows = [
        {
            "id": g.game_id,
            "homeId": g.home_id,
            "awayId": g.away_id,
            "homePoints": str(g.home_points),
            "awayPoints": str(g.away_points),
            "startDate": "frozen-date",
            "neutralSite": "False",
        }
        for g in games
    ]
    return Network(
        "tiny",
        {"season": 2026},
        teams,
        games,
        rows,
        load_pinned_likelihood(
            ROOT / "data/processed/posterior/historical_likelihood_v1.json"
        ),
        {},
        {},
    )


def removal(n, gid="g"):
    row = next(r for r in n.rows if r["id"] == gid)
    return runner.Perturbation(
        n.snapshot_id, gid, "removal", "a", "b", "test", None, row, None
    )


def test_complete_frozen_inventory_and_immutable_handoff(verified, monkeypatch):
    def forbidden(*_a, **_k):
        pytest.fail("baseline inference must not run")

    monkeypatch.setattr(runner, "solve", forbidden)
    jobs = runner.inventory(verified)
    assert len(jobs) == 13
    assert sum(j.kind == "replacement" for j in jobs) == 8
    assert {j.game_id for j in jobs if j.kind == "removal"} == set(
        verified.specification["game_diagnostics"]["removal_game_ids"]
    )
    categories = {
        c
        for r in json.loads((ROOT / ARTIFACT_DIR / "cases.json").read_text())
        for c in r["categories"]
    }
    assert all(j.profile in categories for j in jobs if j.kind == "removal")
    original = verified.networks[jobs[0].snapshot_id]
    before = copy.deepcopy(original.rows)
    for job in jobs:
        changed = runner.perturb(original, job)
        assert changed.teams is original.teams
        assert changed.likelihood is original.likelihood
        assert [g for g in changed.games if g.game_id != job.game_id] == [
            g for g in original.games if g.game_id != job.game_id
        ]
        assert [r for r in changed.rows if r["id"] != job.game_id] == [
            r for r in original.rows if r["id"] != job.game_id
        ]
        if job.kind == "replacement":
            g = next(g for g in changed.games if g.game_id == job.game_id)
            margin = g.home_points - g.away_points
            assert (margin if g.home_id == job.focal_id else -margin) == job.margin
        else:
            assert len(changed.games) == len(original.games) - 1
    assert original.rows == before
    assert (
        verified.origin["execution_revision"]
        == runner._read_json(ROOT / runner.BASELINE_OUTPUT / "provenance.json")[
            "execution_revision"
        ]
    )


def test_replacement_zero_floor_and_both_orientations(tiny):
    for home in (True, False):
        focal, opponent = ("a", "b") if home else ("b", "a")
        row = tiny.rows[0]
        op = max(int(row["awayPoints" if home else "homePoints"]), 3)
        for margin in (-3, 3, 10, 35):
            new = {
                **row,
                "homePoints": str(op + margin if home else op),
                "awayPoints": str(op if home else op + margin),
            }
            job = runner.Perturbation(
                "tiny", "g", "replacement", focal, opponent, "test", margin, row, new
            )
            changed = runner.perturb(tiny, job)
            assert min(changed.games[0].home_points, changed.games[0].away_points) >= 0
            with pytest.raises(ValueError, match="non-score"):
                runner.perturb(
                    tiny,
                    replace(
                        job, replacement_observation={**new, "startDate": "changed"}
                    ),
                )
            with pytest.raises(ValueError, match="score rule"):
                runner.perturb(
                    tiny,
                    replace(job, replacement_observation={**new, "homePoints": "999"}),
                )


def test_removal_keeps_rematch_fcs_and_zero_game_behavior(tiny):
    changed = runner.perturb(tiny, removal(tiny))
    assert [g.game_id for g in changed.games] == ["rematch", "fcs"]
    assert changed.counts == {"a": 2, "b": 1, "c": 1}
    changed = runner.perturb(changed, removal(changed, "rematch"))
    cell = runner.solve(
        changed, "removal", 0.25, "fixture", {}, time.monotonic() + 30, "revision"
    )
    assert cell["counts"]["b"] == 0
    assert np.allclose(cell["pmfs"]["b"], np.ones(3) / 3)
    assert cell["diagnostics"]["raw_game_factor_count"] == 1


def test_fingerprint_and_checkpoint_reject_every_identity_change(tmp_path, tiny):
    job = removal(tiny)
    n = runner.perturb(tiny, job)
    origin = {"fingerprint": "execution", "execution_revision": "revision"}
    identity = runner.cell_identity(n, job, 0.25, origin)
    cell = runner.solve(
        n,
        "removal",
        0.25,
        runner._sha256_json(identity),
        {},
        time.monotonic() + 30,
        "revision",
    )
    cell["identity"] = identity
    path = tmp_path / "cell.json.gz"
    runner.write_checkpoint(path, cell)
    first = path.read_bytes()
    assert runner.validate_cell(path, n, identity, origin) == cell
    runner.write_checkpoint(path, cell)
    assert path.read_bytes() == first
    for key, value in (
        ("configuration", {"tolerance": 0.01}),
        ("margin_weight", 0.5),
        ("cold_start", False),
        ("execution_fingerprint", "other"),
        ("input_rows_sha256", "other"),
        ("prior_pmfs_sha256", "other"),
    ):
        with pytest.raises(ValueError, match="input/configuration/source"):
            runner.validate_cell(path, n, {**identity, key: value}, origin)
    for key, value in (
        ("counts", {"a": 99}),
        ("identity", {}),
        ("execution_revision", "other"),
    ):
        runner.write_checkpoint(path, {**cell, key: value})
        with pytest.raises(ValueError):
            runner.validate_cell(path, n, identity, origin)
    altered = copy.deepcopy(cell)
    altered["diagnostics"]["unique_pair_factor_count"] = 999
    runner.write_checkpoint(path, altered)
    with pytest.raises(ValueError, match="factor evidence"):
        runner.validate_cell(path, n, identity, origin)


def test_failure_preserves_success_and_stops_network(tmp_path, tiny, monkeypatch):
    jobs = [removal(tiny)]
    verified = runner.VerifiedExecution({}, {}, {}, {"tiny": tiny}, {}, False)
    monkeypatch.setattr(runner, "inventory", lambda *_: jobs)
    monkeypatch.setattr(
        runner,
        "provenance",
        lambda *_: {"fingerprint": "execution", "execution_revision": "revision"},
    )
    actual = runner.solve
    calls = []

    def fail_second(n, kind, weight, *args):
        calls.append(weight)
        if weight == 0.25:
            raise TimeoutError("frozen timeout")
        return actual(n, kind, weight, *args)

    monkeypatch.setattr(runner, "solve", fail_second)
    result = runner.execute(ROOT, tmp_path, verified)
    assert not result["complete"] and result["completed_cells"] == 1
    assert calls == [0, 0.25]
    path = runner.checkpoint_path(tmp_path, jobs[0], 0)
    assert path.exists()
    original = path.read_bytes()
    assert list((tmp_path / "failures").glob("*.json"))
    result = runner.execute(ROOT, tmp_path, verified)
    assert not result["complete"] and path.read_bytes() == original
    assert len(list((tmp_path / "failures").glob("*.json"))) == 2
