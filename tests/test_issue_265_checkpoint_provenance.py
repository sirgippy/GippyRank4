"""Narrow regression coverage for new fingerprints and the original 32 cells."""

from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

import pytest

from gippyrank.posterior.snapshots import sha256
from gippyrank.research import issue_265_population as population
from gippyrank.research import issue_265_real_seasons as baseline
from gippyrank.research.issue_263_real_season_design import ARTIFACT_DIR, check_design

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / baseline.OUTPUT


@pytest.fixture
def source_tree(tmp_path):
    """Editable sources/design; immutable large production inputs are shared."""
    root = tmp_path / "repo"
    shutil.copytree(
        ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(ROOT / ARTIFACT_DIR, root / ARTIFACT_DIR)
    (root / "data/processed").symlink_to(
        ROOT / "data/processed", target_is_directory=True
    )
    (root / "docs").symlink_to(ROOT / "docs", target_is_directory=True)
    for name in (
        ".python-version",
        "uv.lock",
        "pyproject.toml",
        "scripts/design_issue_263_real_seasons.py",
        "scripts/run_issue_265_real_seasons.py",
    ):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, root / name)
    return root


def forbid_inference_and_writes(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("consuming the historical run must not infer or write")

    for name in (
        "solve",
        "infer_posterior",
        "infer_quality_of_record",
        "infer_quality_of_results",
        "write_json",
        "write_checkpoint",
    ):
        monkeypatch.setattr(baseline, name, forbidden)
    monkeypatch.setattr(population, "write_csv", forbidden)
    return forbidden


def test_unrelated_research_changes_preserve_new_fingerprint_and_original_results(
    source_tree, monkeypatch
):
    spec = check_design(source_tree, source_tree / ARTIFACT_DIR)
    monkeypatch.setattr(
        baseline.subprocess, "check_output", lambda *_a, **_kw: "new-revision\n"
    )
    original = baseline.provenance(source_tree, spec)
    (source_tree / "src/gippyrank/research/next_diagnostic.py").write_text(
        "raise RuntimeError('must never be imported')\n"
    )
    unrelated = source_tree / "src/gippyrank/research/issue_261_quality_of_results.py"
    unrelated.write_text(unrelated.read_text() + "\n# An unrelated research change.\n")
    assert baseline.provenance(source_tree, spec) == original
    assert set(original["source_sha256"]) == set(baseline.BASELINE_SOURCES)
    forbid_inference_and_writes(monkeypatch)
    verified = baseline.verify_completed_execution(source_tree, SAVED)
    assert verified.historical
    assert sum(len(cells) for cells in verified.cells.values()) == 32
    assert verified.origin == json.loads((SAVED / "provenance.json").read_text())
    assert verified.origin["execution_revision"] == baseline.HISTORICAL_REVISION
    assert verified.origin["fingerprint"] == baseline.HISTORICAL_FINGERPRINT
    assert all(
        cell["execution_revision"] == baseline.HISTORICAL_REVISION
        for cells in verified.cells.values()
        for cell in cells.values()
    )
    assert baseline.execute(source_tree, SAVED) == verified.execution
    for name, digest in json.loads((SAVED / "manifest.json").read_text())[
        "output_sha256"
    ].items():
        assert sha256(SAVED / name) == digest


@pytest.mark.parametrize(
    "change", ["dependency", "runner", "evidence", "configuration"]
)
def test_historical_consumer_detects_material_changes(source_tree, monkeypatch, change):
    if change == "dependency":
        path = source_tree / "src/gippyrank/modeling.py"
        path.write_text(path.read_text() + "\n# Changed inference dependency.\n")
    elif change == "runner":
        path = source_tree / "src/gippyrank/research/issue_265_real_seasons.py"
        source = path.read_text()
        changed = source.replace(
            'if kind == "reference" and weight == 0:',
            'if kind == "reference" and weight == .5:',
        )
        assert source != changed
        path.write_text(changed)
    elif change == "evidence":
        path = source_tree / ARTIFACT_DIR / "inputs/2025-final/included_games.csv"
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        monkeypatch.setattr(baseline, "CONFIG", {**baseline.CONFIG, "damping": 0.4})
    forbid_inference_and_writes(monkeypatch)
    with pytest.raises(
        ValueError,
        match="dependency changed|inference path changed|hash mismatch|configuration mismatch",
    ):
        baseline.verify_completed_execution(source_tree, SAVED)


def test_new_checkpoints_require_exact_material_source_and_configuration(
    source_tree, monkeypatch
):
    spec = check_design(source_tree, source_tree / ARTIFACT_DIR)
    monkeypatch.setattr(
        baseline.subprocess, "check_output", lambda *_a, **_kw: "revision\n"
    )
    original = baseline.provenance(source_tree, spec)
    path = source_tree / "src/gippyrank/modeling.py"
    path.write_text(path.read_text() + "\n# Changed dependency.\n")
    assert (
        baseline.provenance(source_tree, spec)["fingerprint"] != original["fingerprint"]
    )
    path.write_bytes((ROOT / "src/gippyrank/modeling.py").read_bytes())
    monkeypatch.setattr(baseline, "CONFIG", {**baseline.CONFIG, "damping": 0.4})
    changed = baseline.provenance(source_tree, spec)
    assert changed["fingerprint"] != original["fingerprint"]
    # Historical cells are never relabeled into the new fingerprint scheme.
    monkeypatch.setattr(baseline, "CONFIG", original["configuration"])
    network = baseline.load_network(ROOT, spec, spec["selected_snapshot_ids"][0])
    expected = baseline.cell_fingerprint(network, original, "baseline", 0.0)
    with pytest.raises(ValueError, match="input/configuration/source"):
        baseline.read_checkpoint(
            SAVED / "checkpoints/2025-prebowl/baseline-0.json.gz", expected, network
        )


@pytest.mark.parametrize("change", ["corruption", "fingerprint", "revision", "gate"])
def test_historical_artifact_integrity_cannot_be_repaired_by_relabeling(
    tmp_path, monkeypatch, change
):
    output = tmp_path / "saved"
    shutil.copytree(SAVED, output)
    if change in {"corruption", "fingerprint"}:
        path = output / "checkpoints/2025-prebowl/baseline-0.json.gz"
        if change == "corruption":
            path.write_bytes(path.read_bytes() + b"corruption")
        else:
            cell = json.loads(gzip.decompress(path.read_bytes()))
            cell["fingerprint"] = "a replacement identity"
            baseline.write_checkpoint(path, cell)  # Even with a matching sidecar.
    elif change == "revision":
        path = output / "provenance.json"
        origin = json.loads(path.read_text())
        origin["execution_revision"] = "a different revision"
        path.write_text(json.dumps(origin))
    else:
        path = output / "endpoint_parity.json"
        gates = json.loads(path.read_text())
        gates["2025-final"]["qop"]["passed"] = False
        path.write_text(json.dumps(gates))
    forbid_inference_and_writes(monkeypatch)
    with pytest.raises(ValueError, match="integrity mismatch|identity mismatch"):
        baseline.verify_completed_execution(ROOT, output)


def test_verify_cli_and_default_consume_original_run_without_git_inference_or_writes(
    monkeypatch, capsys
):
    forbidden = forbid_inference_and_writes(monkeypatch)
    monkeypatch.setattr(baseline.subprocess, "check_output", forbidden)
    for flag in ("--verify-only", None):
        argv = ["run_issue_265_real_seasons.py"] + ([flag] if flag else [])
        monkeypatch.setattr(sys, "argv", argv)
        baseline.main()
        status = json.loads(capsys.readouterr().out)
        assert status == {
            "status": "verified_historical",
            "cells": 32,
            "fingerprint": baseline.HISTORICAL_FINGERPRINT,
            "execution_revision": baseline.HISTORICAL_REVISION,
        }
    with pytest.raises(ValueError, match="historical results are immutable"):
        population.analyze(ROOT, SAVED)


def test_incompatible_resume_does_not_overwrite_existing_identity(
    tmp_path, monkeypatch
):
    path = tmp_path / "provenance.json"
    original = '{"fingerprint":"old","execution_revision":"original"}\n'
    path.write_text(original)
    monkeypatch.setattr(
        baseline,
        "provenance",
        lambda *_a: {"fingerprint": "new", "execution_revision": "current"},
    )
    forbid_inference_and_writes(monkeypatch)
    with pytest.raises(
        ValueError, match="existing execution input/configuration/source"
    ):
        baseline.execute(ROOT, tmp_path)
    assert path.read_text() == original
