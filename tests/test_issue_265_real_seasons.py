"""Integrity, numerical conventions and population analysis regression tests."""

from __future__ import annotations

import copy
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pytest

from gippyrank.posterior.engine import Game, Team
from gippyrank.posterior.snapshots import load_pinned_likelihood
from gippyrank.research import issue_265_population as population
from gippyrank.research import issue_265_real_seasons as execution
from gippyrank.research.issue_263_real_season_design import ARTIFACT_DIR, check_design

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def network():
    teams = [
        Team("a", "A", "fbs", np.array([0.2, 0.3, 0.5])),
        Team("b", "B", "fbs", np.array([0.6, 0.3, 0.1])),
        Team("c", "C", "fcs", np.array([0.4, 0.6])),
    ]
    games = [
        Game("g", "a", "b", "fbs", "fbs", 21, 14),
        Game("h", "c", "a", "fcs", "fbs", 10, 31),
    ]
    likelihood = load_pinned_likelihood(
        ROOT / "data/processed/posterior/historical_likelihood_v1.json"
    )
    return execution.Network(
        "fixture", {"season": 2025}, teams, games, [], likelihood, {}, {}
    )


def test_frozen_loaders_preserve_every_snapshot_population():
    spec = check_design(ROOT, ROOT / ARTIFACT_DIR)
    for sid, count, fbs in zip(
        spec["selected_snapshot_ids"],
        (1570, 1625, 171, 530, 975),
        (136, 136, 138, 138, 138),
        strict=True,
    ):
        n = execution.load_network(ROOT, spec, sid)
        assert len(n.games) == count
        assert len(n.fbs) == fbs
        assert any(t.subdivision == "fcs" for t in n.teams)
        assert n.specification["prior_pmfs_sha256"]


def test_pmf_validation_rejects_missing_shape_negative_nan_and_unnormalized():
    for p in (
        np.array([1.0]),
        np.array([-1.0, 2.0]),
        np.array([np.nan, 1.0]),
        np.array([0.2, 0.2]),
    ):
        with pytest.raises(ValueError):
            execution.validate_pmfs({"a": p}, {"a": 2})
    with pytest.raises(ValueError):
        execution.validate_pmfs({}, {"a": 2})


def test_endpoint_gate_checks_full_pmfs_and_order_and_reports_ties():
    p = {"a": np.array([0.5, 0.0, 0.5]), "b": np.array([0.0, 1.0, 0.0])}
    gate = execution.parity(p, p, {"a": 1, "b": 1})
    assert gate["passed"]
    assert len(gate["numerical_ties"]) == 1
    changed = {"a": p["b"], "b": p["a"]}  # Same means; very different distributions.
    assert not execution.parity(p, changed, {"a": 1, "b": 1})["passed"]
    p = {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.0, 0.0, 1.0])}
    assert execution.parity(p, {"a": p["b"], "b": p["a"]}, {"a": 1, "b": 1})[
        "non_tied_order_inversions"
    ] == [["a", "b"]]


def test_checkpoint_requires_fingerprint_and_integrity_and_valid_network(
    tmp_path, network
):
    value = {
        "fingerprint": "frozen",
        "diagnostics": {"converged": True, "iterations": 1, "max_message_delta": 0.0},
        "pmfs": {
            t.team_id: t.prior.tolist() for t in network.teams if t.subdivision == "fbs"
        },
        "context": {t.team_id: t.prior.tolist() for t in network.teams},
        "counts": network.counts,
    }
    path = tmp_path / "cell.json.gz"
    execution.write_checkpoint(path, value)
    original = path.read_bytes()
    assert execution.read_checkpoint(path, "frozen", network) == value
    execution.write_checkpoint(path, value)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="input/configuration/source"):
        execution.read_checkpoint(path, "new-configuration", network)
    path.write_bytes(original + b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        execution.read_checkpoint(path, "frozen", network)
    value["pmfs"].pop("b")
    execution.write_checkpoint(path, value)
    with pytest.raises(ValueError, match="population"):
        execution.read_checkpoint(path, "frozen", network)


def test_cell_fingerprint_distinguishes_source_weight_order_and_configuration(network):
    fingerprint = execution.cell_fingerprint(
        network, {"fingerprint": "source"}, "baseline", 0.25
    )
    assert fingerprint != execution.cell_fingerprint(
        network, {"fingerprint": "new-source"}, "baseline", 0.25
    )
    assert fingerprint != execution.cell_fingerprint(
        network, {"fingerprint": "source"}, "baseline", 0.5
    )
    assert fingerprint != execution.cell_fingerprint(
        network, {"fingerprint": "source"}, "reversed", 0.25
    )
    changed = copy.deepcopy(network)
    changed.specification["configuration"] = "changed"
    assert fingerprint != execution.cell_fingerprint(
        changed, {"fingerprint": "source"}, "baseline", 0.25
    )


def test_rank_summaries_and_distribution_distance_conventions():
    p = np.array([0.5, 0.0, 0.5])
    s = population.summary(p)
    assert s["expected_rank"] == 2
    assert s["median_rank"] == s["modal_rank"] == 1
    assert s["rank_sd"] == 1
    assert s["entropy_bits"] == 1
    assert s["low_50"] == 1 and s["high_50"] == 3 and s["width_50"] == 3
    change = population.compare(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    assert change["delta_expected_rank"] == 2
    assert change["total_variation"] == 1
    assert change["wasserstein_rank"] == 2
    assert population.distribution([])["n"] == 0
    assert population.distribution([0.0, 10.0])["p05"] == 0.5


def test_actual_game_descriptions_keep_venue_orientation_and_full_opponent_pmfs(
    network,
):
    ds = {
        "a": {"team_id": "a", "reference_team_id": "a", "wins": 2},
        "b": {"team_id": "b", "reference_team_id": "a", "wins": 0},
    }
    context = {t.team_id: t.prior for t in network.teams}
    rows, descriptions = population.game_descriptions(
        network, ds, context, {"a": 1, "b": 2}
    )
    a = next(r for r in rows if r["team_id"] == "a" and r["game_id"] == "g")
    b = next(r for r in rows if r["team_id"] == "b")
    assert a["context_win_probability"] + b["context_win_probability"] == pytest.approx(
        1
    )
    assert a["expected_margin"] == -b["expected_margin"]
    assert a["margin"] == -b["margin"] == 7
    assert next(r for r in rows if r["game_id"] == "h")["venue"] == "away"
    assert next(r for r in descriptions if r["team_id"] == "a")["fcs_games"] == 1


def test_frozen_strata_remain_independent_of_candidate_rankings():
    ds = population.frozen_descriptors(ROOT, "2025-prebowl", 2025)
    assert {r["reference_kind"] for r in ds.values()} == {"preseason_context_v1.2"}
    assert sum(r["strength_band"] == "middle" for r in ds.values()) == 55
    groups = dict(population.panels(ds))
    assert len(groups["band/elite"]) == 10
    assert set(groups["band/middle/schedule/hard"]).isdisjoint(
        groups["band/middle/schedule/easy"]
    )
    assert any(not ids for ids in groups.values())


def test_resource_guard_preserves_frozen_limits():
    with pytest.raises(TimeoutError), execution.resource_guard(time.monotonic() - 1):
        pass
    with (
        pytest.raises(MemoryError),
        execution.resource_guard(time.monotonic() + 1, rss_limit=0),
    ):
        pass


def test_gzip_population_outputs_are_byte_deterministic(tmp_path):
    path = tmp_path / "rows.csv.gz"
    rows = [{"team_id": "a", "profile": [0.2, 0.8]}]
    population.write_csv(path, rows)
    first = path.read_bytes()
    population.write_csv(path, rows)
    assert path.read_bytes() == first
    assert json.loads(
        gzip.decompress(first)
        .decode()
        .splitlines()[1]
        .split(",", 1)[1]
        .strip('"')
        .replace('""', '"')
    ) == [0.2, 0.8]


def test_failed_cell_preserves_successes_and_does_not_claim_complete(
    tmp_path, monkeypatch, network
):
    spec = {
        "selected_snapshot_ids": ["first", "second"],
        "execution": {"wall_budget_hours": 3},
        "game_diagnostics": {"publication_slot": "neither"},
    }
    monkeypatch.setattr(execution, "check_design", lambda *_: spec)
    monkeypatch.setattr(
        execution,
        "provenance",
        lambda *_: {"fingerprint": "inputs", "execution_revision": "revision"},
    )

    def load(_root, _spec, sid):
        n = copy.deepcopy(network)
        n.snapshot_id = sid
        return n

    called = []

    def solve(n, kind, weight, fingerprint, *_args):
        called.append((n.snapshot_id, kind, weight))
        if n.snapshot_id == "first" and weight == 0.25:
            raise RuntimeError("synthetic convergence failure")
        return {
            "fingerprint": fingerprint,
            "pmfs": {
                t.team_id: t.prior.tolist() for t in n.teams if t.subdivision == "fbs"
            },
            "context": {t.team_id: t.prior.tolist() for t in n.teams},
            "counts": n.counts,
            "diagnostics": {
                "converged": True,
                "iterations": 1,
                "max_message_delta": 0.0,
            },
        }

    monkeypatch.setattr(execution, "load_network", load)
    monkeypatch.setattr(execution, "solve", solve)
    result = execution.execute(ROOT, tmp_path)
    assert not result["complete"]
    assert len(result["cells"]) == 8
    assert len(result["failures"]) == 1
    assert ("first", "baseline", 0.5) not in called
    assert (tmp_path / "checkpoints/first/baseline-0.json.gz").is_file()
    assert (tmp_path / "failures/first-baseline-0.25.json").is_file()
    assert (
        len(json.loads((tmp_path / "endpoint_parity.json").read_text())["second"]) == 2
    )
