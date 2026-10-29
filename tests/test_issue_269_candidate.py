"""Extension coverage without executing the expensive research inventory."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from gippyrank.posterior.engine import LikelihoodV1, Team
from gippyrank.research import issue_252_quality_of_results as synthetic
from gippyrank.research import issue_269_candidate as candidate
from gippyrank.research.issue_269_analysis import matched_rows, numeric


def test_only_requested_extension_weight_is_solved(monkeypatch):
    prior = np.array([0.2, 0.5, 0.3])
    schedule = tuple(
        synthetic.OpponentSlot(Team(str(i), str(i), "fbs", prior), venue, 0.5, 0.5)
        for i, venue in enumerate(synthetic.SITE_SLOTS)
    )
    likelihood = LikelihoodV1(np.zeros(34), scale=7.0, degrees_of_freedom=15.0)
    trial = synthetic.generate_trial(
        1234,
        prior,
        {"approximately_even": schedule, "weaker_opponents": schedule},
        likelihood,
    )
    weights = []
    infer = synthetic.infer_quality_of_results

    def record(*args, **kwargs):
        weights.append(args[3])
        return infer(*args, **kwargs)

    monkeypatch.setattr(synthetic, "infer_quality_of_results", record)
    teams, pairs, pmfs = synthetic._infer_trial(
        trial, likelihood, margin_weights=(0.4,)
    )
    assert weights == [0.4]
    assert set(pmfs) == {0.4}
    assert all(r["max_message_delta"] <= 1e-9 for r in teams)
    assert {
        r["margin_weight"]
        for r in synthetic.summarize_pairs(
            pairs * 2, "unconditional", margin_weights=(0.4,)
        )
    } == {0.4}
    assert synthetic.MARGIN_WEIGHTS == (0, 0.25, 0.5, 0.75, 1)


def synthetic_cell():
    origin = {"fingerprint": "execution", "execution_revision": "revision"}
    identity = candidate.synthetic_identity(
        origin, "elite", "control_conditioned", 42, "evidence"
    )
    groups = ("same_schedule_a", "same_schedule_b")
    value = {
        "identity": identity,
        "fingerprint": candidate._sha256_json(identity),
        "execution_revision": "revision",
        "pmfs": {k: (np.ones(138) / 138).tolist() for k in groups},
        "teams": [
            {
                "focal_group": k,
                "margin_weight": 0.4,
                "converged": True,
                "iterations": 30,
                "max_message_delta": 1e-10,
            }
            for k in groups
        ],
        "pairs": [{}],
    }
    return value, identity, origin


@pytest.mark.parametrize(
    "change", ["seed", "evidence_sha256", "profile", "panel", "margin_weight"]
)
def test_checkpoint_rejects_wrong_frozen_membership(change):
    value, identity, origin = synthetic_cell()
    wrong = deepcopy(identity)
    wrong[change] = "different"
    with pytest.raises(ValueError, match="identity"):
        candidate.validate_synthetic(value, wrong, origin)


@pytest.mark.parametrize(
    "failure", ["support", "normalization", "convergence", "population"]
)
def test_checkpoint_rejects_invalid_or_incomplete_posteriors(failure):
    value, identity, origin = synthetic_cell()
    candidate.validate_synthetic(value, identity, origin)
    if failure == "support":
        value["pmfs"]["same_schedule_a"].pop()
    elif failure == "normalization":
        value["pmfs"]["same_schedule_a"][0] = 0.5
    elif failure == "convergence":
        value["teams"][0]["max_message_delta"] = 1e-8
    else:
        value["pmfs"].pop("same_schedule_a")
    with pytest.raises(ValueError):
        candidate.validate_synthetic(value, identity, origin)


def test_archive_hash_is_checked_before_reuse(tmp_path: Path):
    path = tmp_path / "cells.json.gz"
    value, _, _ = synthetic_cell()
    candidate.write_checkpoint(path, value)
    assert candidate.read_hashed(path) == value
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="integrity"):
        candidate.read_hashed(path)


def test_retained_boolean_flags_do_not_become_truthy_strings():
    assert numeric({"right_outranks_left": "False", "seed": "42", "team_id": "01"}) == {
        "right_outranks_left": False,
        "seed": 42.0,
        "team_id": "01",
    }


def test_matched_comparisons_reject_different_samples_and_duplicates():
    rows = [
        {"seed": s, "margin_weight": w} for w in candidate.NEIGHBORS for s in (1, 2)
    ]
    assert all(set(v) == {1, 2} for v in matched_rows(list(reversed(rows))).values())
    with pytest.raises(ValueError, match="unmatched"):
        matched_rows(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        matched_rows([*rows, rows[0]])


def test_provenance_requires_same_sources_but_portable_reads_preserve_runtime(
    monkeypatch, tmp_path
):
    payload = {"source_sha256": {"runner": "first"}, "environment": {"python": "old"}}
    origin = {
        **payload,
        "fingerprint": candidate._sha256_json(payload),
        "execution_revision": "original",
    }
    candidate.write_json(tmp_path / "provenance.json", origin)

    def current(*args):
        p = {**payload, "environment": {"python": "new"}}
        return {
            **p,
            "fingerprint": candidate._sha256_json(p),
            "execution_revision": "new",
        }

    monkeypatch.setattr(candidate, "provenance", current)
    with pytest.raises(ValueError, match="source/input/runtime"):
        candidate.origin_for(tmp_path, tmp_path, None)
    assert candidate.origin_for(tmp_path, tmp_path, None, portable=True) == origin
    payload["source_sha256"]["runner"] = "changed"
    with pytest.raises(ValueError, match="source/input/runtime"):
        candidate.origin_for(tmp_path, tmp_path, None, portable=True)
