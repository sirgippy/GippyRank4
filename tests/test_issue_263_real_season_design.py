from __future__ import annotations

import copy
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from gippyrank.posterior import engine
from gippyrank.posterior.engine import Game, Team
from gippyrank.posterior.snapshots import load_teams, sha256
from gippyrank.quality_of_record_snapshot import _load_source_likelihood
from gippyrank.research import issue_242_quality_of_results as replay
from gippyrank.research import issue_263_real_season_design as design
from gippyrank.research import quality_of_results

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / design.ARTIFACT_DIR


def _compare_audit_only_inputs(actual, frozen, root, frozen_dir):
    assert actual.keys() == frozen.keys()
    for path, digest in actual.items():
        assert digest == sha256(root / path)
    assert frozen["site/publish_config.json"] == sha256(
        frozen_dir / "publication_config.json"
    )


@pytest.fixture(scope="module")
def specification():
    return json.loads((FROZEN / "protocol.json").read_text())


def test_frozen_population_and_provenance(specification):
    spec = design.check_design(ROOT, FROZEN)
    assert spec["selected_publication_slots"] == list(design.SELECTED_SLOTS)
    assert len(spec["snapshot_audit"]) == 8
    assert spec["historical_audit"]["context_2025_prior_rows"] == 136
    assert spec["unavailable_snapshots"] == []
    assert len(spec["selected_snapshot_ids"]) == 5
    assert {row["information_phase"] for row in spec["historical_snapshots"]} == {
        "pre_bowl",
        "final",
    }
    assert spec["candidate_configuration"]["margin_weights"] == [0, 0.25, 0.5, 0.75, 1]
    for snapshot in spec["snapshot_audit"]:
        assert snapshot["campbell_corrected"]
        assert len(snapshot["fbs_team_ids"]) == 138
        assert snapshot["fbs_team_ids"] == snapshot["rated_fbs_team_ids"]
        assert snapshot["historical_likelihood_version"] == "V1"
    assert len(spec["case_selection"]["case_team_ids"]) == 16
    assert spec["execution"]["maximum_network_solves"] == 97
    assert spec["execution"]["baseline_solves"] == 25
    assert spec["execution"]["historical_endpoint_reference_solves"] == 4
    probes = spec["game_diagnostics"]["counterfactuals"]
    assert {p["profile"] for p in probes} == {"elite", "weak"}
    assert all(p["margins"] == [-3, 3, 10, 35] for p in probes)


def test_reference_ordinals_preserve_published_order_even_for_near_ties(specification):
    descriptors = design._read_rows(FROZEN / "reference_descriptors.csv")
    for snapshot in specification["snapshot_audit"]:
        if not snapshot["selected"]:
            continue
        rows = [
            row
            for row in descriptors
            if row["publication_slot"] == snapshot["publication_slot"]
        ]
        for family in ("qop", "qor"):
            published = {
                r["team_id"]: r["display_rank"]
                for r in design._read_rows(
                    ROOT / snapshot[family + "_source"] / "rankings.csv"
                )
            }
            assert {r["team_id"]: r[family + "_ordinal"] for r in rows} == published


def test_regeneration_is_deterministic_without_any_candidate_inference(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("design must not run candidate or reference inference")

    monkeypatch.setattr(engine, "infer_posterior", forbidden)
    monkeypatch.setattr(quality_of_results, "infer_quality_of_results", forbidden)
    monkeypatch.setattr(replay, "infer_quality_of_results", forbidden)
    # Give regeneration an independently changed live inventory while all
    # frozen inputs and retained evidence still point to the original files.
    live_root = tmp_path / "live-root"
    live_root.mkdir()
    inventory = Path("data/processed/cfbd/games.csv")
    for parent in (Path(), *inventory.parents[-2::-1]):
        (live_root / parent).mkdir(exist_ok=True)
        next_part = inventory.parts[len(parent.parts)]
        for path in (ROOT / parent).iterdir():
            if path.name != next_part:
                (live_root / parent / path.name).symlink_to(path)
    (live_root / inventory).write_bytes((ROOT / inventory).read_bytes() + b"\n")
    first = tmp_path / "first"
    repeated = tmp_path / "repeated"
    generated = design.build_design(live_root, first)
    design.build_design(live_root, repeated)
    assert generated["candidate_inference_performed"] is False
    assert generated["audit_only_input_sha256"][str(inventory)] != sha256(
        ROOT / inventory
    )
    design.check_design(live_root, first)

    def equivalent(actual, expected):
        if isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                equivalent(actual[key], expected[key])
        elif isinstance(expected, list):
            assert len(actual) == len(expected)
            for left, right in zip(actual, expected, strict=True):
                equivalent(left, right)
        elif isinstance(expected, float):
            assert actual == pytest.approx(expected, rel=0, abs=1e-10)
        else:
            assert actual == expected

    # Exact repeated output is required within one runtime. BLAS/libm and
    # Python patch versions may differ on CI, so compare recomputed numeric
    # descriptors to the retained values with a tight absolute tolerance.
    # IDs, categories, frozen inputs, and snapshot hashes still compare exactly.
    for artifact in FROZEN.rglob("*"):
        if not artifact.is_file():
            continue
        relative = artifact.relative_to(FROZEN)
        actual = first / relative
        assert actual.read_bytes() == (repeated / relative).read_bytes()
        if artifact.name == "manifest.json":
            continue  # Output hashes are verified against the generated files.
        if artifact.name == "publication_config.json":
            # Regeneration records the current configuration, which can grow
            # without changing the frozen study population.
            assert (
                actual.read_bytes() == (ROOT / "site/publish_config.json").read_bytes()
            )
            continue
        if artifact.suffix == ".csv":

            def numeric_rows(path):
                with path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    for key, value in row.items():
                        if key in {
                            "season",
                            "team_id",
                            "game_id",
                            "opponent_id",
                            "reference_team_id",
                        }:
                            continue
                        try:
                            row[key] = float(value)
                        except ValueError:
                            pass
                return rows

            equivalent(numeric_rows(actual), numeric_rows(artifact))
        else:
            left = json.loads(actual.read_text())
            right = json.loads(artifact.read_text())
            if artifact.name == "protocol.json":
                left.pop("artifact_sha256")
                right.pop("artifact_sha256")
                _compare_audit_only_inputs(
                    left.pop("audit_only_input_sha256"),
                    right.pop("audit_only_input_sha256"),
                    live_root,
                    FROZEN,
                )
            equivalent(left, right)


def test_audit_only_inventory_can_change_but_regeneration_records_its_hash(tmp_path):
    config = tmp_path / "site/publish_config.json"
    inventory = tmp_path / "data/processed/cfbd/games.csv"
    config.parent.mkdir(parents=True)
    inventory.parent.mkdir(parents=True)
    config.write_text("frozen config")
    inventory.write_text("old inventory")
    frozen_dir = tmp_path / "frozen"
    frozen_dir.mkdir()
    (frozen_dir / "publication_config.json").write_text(config.read_text())
    paths = ["site/publish_config.json", "data/processed/cfbd/games.csv"]
    frozen = {path: sha256(tmp_path / path) for path in paths}
    inventory.write_text("updated inventory")
    current = {path: sha256(tmp_path / path) for path in paths}
    _compare_audit_only_inputs(current, frozen, tmp_path, frozen_dir)
    current[paths[1]] = frozen[paths[1]]
    with pytest.raises(AssertionError):
        _compare_audit_only_inputs(current, frozen, tmp_path, frozen_dir)


@pytest.mark.parametrize(
    "artifact", ["cases.json", "protocol.json", "reference_games.csv"]
)
def test_modified_design_is_rejected(tmp_path, artifact):
    shutil.copytree(FROZEN, tmp_path / "design")
    target = tmp_path / "design" / artifact
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen design artifact hash mismatch"):
        design.check_design(ROOT, tmp_path / "design")


def test_new_publication_does_not_change_or_block_frozen_population(
    tmp_path, specification
):
    paths = set(specification["input_sha256"])
    for snapshot in specification["snapshot_audit"]:
        paths.update(snapshot["artifact_sha256"])
    for path in paths:
        linked = tmp_path / path
        linked.parent.mkdir(parents=True, exist_ok=True)
        linked.symlink_to(ROOT / path)
    target = tmp_path / "site/publish_config.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"new_official_week": true}')
    checked = design.check_design(tmp_path, FROZEN)
    assert checked["selected_publication_slots"] == list(design.SELECTED_SLOTS)
    # Actual retained evidence drift must still fail closed.
    changed = (
        tmp_path
        / specification["snapshot_audit"][0]["context_source"]
        / "included_games.csv"
    )
    changed.unlink()
    changed.write_text("modified retained evidence")
    with pytest.raises(ValueError, match="frozen snapshot hash mismatch"):
        design.check_design(tmp_path, FROZEN)


def test_modified_frozen_input_is_rejected(tmp_path, specification):
    paths = set(specification["input_sha256"])
    for snapshot in specification["snapshot_audit"]:
        paths.update(snapshot["artifact_sha256"])
    for path in paths:
        linked = tmp_path / path
        linked.parent.mkdir(parents=True, exist_ok=True)
        linked.symlink_to(ROOT / path)
    changed = tmp_path / next(iter(specification["input_sha256"]))
    changed.unlink()
    changed.write_text("modified frozen input")
    with pytest.raises(ValueError, match="frozen input hash mismatch"):
        design.check_design(tmp_path, FROZEN)


def test_selected_cases_ignore_row_order_and_cover_every_category(specification):
    rows = [
        r
        for r in design._read_rows(FROZEN / "reference_descriptors.csv")
        if r["publication_slot"] == design.SELECTED_SLOTS[-1]
    ]
    # Use the frozen JSON's typed records, not CSV string values.
    inventory = json.loads((FROZEN / "cases.json").read_text())
    typed_by_id = {
        r["team_id"]: r
        for r in inventory
        if r["publication_slot"] == design.SELECTED_SLOTS[-1]
    }
    for row in rows:
        for key in ("games", "wins", "losses", "qop_ordinal", "qor_ordinal"):
            row[key] = int(row[key])
        for key in (
            "win_fraction",
            "schedule_reference_mean_win_probability",
            "qor_minus_qop_expected_rank",
        ):
            row[key] = float(row[key])
    games = design._read_rows(FROZEN / "reference_games.csv")
    games = [g for g in games if g["publication_slot"] == design.SELECTED_SLOTS[-1]]
    for game in games:
        game["margin"] = int(game["margin"])
        game["context_win_probability"] = float(game["context_win_probability"])
        game["opponent_qop_ordinal"] = (
            int(game["opponent_qop_ordinal"]) if game["opponent_qop_ordinal"] else None
        )
    cases, anchors = design.select_cases(rows, games)
    reversed_cases, reversed_anchors = design.select_cases(rows[::-1], games[::-1])
    assert cases == reversed_cases
    assert anchors == reversed_anchors
    assert (
        sorted(r["team_id"] for r in cases)
        == specification["case_selection"]["case_team_ids"]
    )
    assert {c for r in cases for c in r["categories"]} == {
        *design.CASE_QUESTIONS,
        "boundary_10",
        "boundary_12",
        "boundary_25",
    }
    for case in cases:
        assert case["categories"] == typed_by_id[case["team_id"]]["categories"]
        assert typed_by_id[case["team_id"]]["questions"]
    for category in ("middle_schedule_pair", "weak_schedule_pair"):
        pair = [c for c in cases if category in c["categories"]]
        assert len(pair) == 2
        assert (pair[0]["wins"], pair[0]["losses"]) == (
            pair[1]["wins"],
            pair[1]["losses"],
        )


@pytest.mark.parametrize(
    "mutation", ["future", "duplicate", "phantom", "score", "cutoff", "replacement"]
)
def test_evidence_gate_rejects_invalid_or_repaired_input_drift(specification, mutation):
    metadata = design._read_json(
        ROOT / specification["snapshot_audit"][0]["context_source"] / "metadata.json"
    )
    rows = design._read_rows(
        ROOT
        / specification["snapshot_audit"][0]["context_source"]
        / "included_games.csv"
    )
    metadata = copy.deepcopy(metadata)
    if mutation == "future":
        rows[0]["startDate"] = "2026-12-01T00:00:00Z"
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "phantom":
        rows[0]["id"] = design.BAD_CAMPBELL_FIXTURE_ID
    elif mutation == "score":
        rows[0]["homePoints"] = rows[0]["awayPoints"]
    elif mutation == "cutoff":
        metadata["historical_evidence_correction"]["source_requested_cutoff"] = (
            "changed"
        )
    else:
        next(r for r in rows if r["id"] == design.REPAIRED_CAMPBELL_GAME_ID)[
            "homePoints"
        ] = "27"
        metadata.pop("included_game_rows_sha256", None)
    with pytest.raises(ValueError):
        design.validate_rows(rows, metadata)


def test_integrated_reference_uses_full_pmfs_and_orients_cross_subdivision(
    specification,
):
    teams, _, _ = load_teams(ROOT, 2026, "context", "1.4")
    home = teams[0]
    away = Team("fcs", "FCS", "fcs", np.full(128, 1 / 128))
    snapshot = specification["snapshot_audit"][0]
    metadata = design._read_json(ROOT / snapshot["context_source"] / "metadata.json")
    likelihood = _load_source_likelihood(metadata, ROOT)
    neutral = Game("probe", home.team_id, away.team_id, "fbs", "fcs", 35, 7, True)
    reverse = Game("probe", away.team_id, home.team_id, "fcs", "fbs", 7, 35, True)
    forward_p = design.integrated_win_probability(neutral, home, away, likelihood)
    reverse_p = design.integrated_win_probability(reverse, away, home, likelihood)
    assert 0 < forward_p < 1
    assert forward_p + reverse_p == pytest.approx(1, abs=1e-12)
    surface = design.conditional_margin_location_surface(
        design.ScheduledGame("probe", home.team_id, away.team_id, "fbs", "fcs", True),
        home,
        away,
        likelihood,
    )
    probabilities = design.student_t.sf(
        -surface / likelihood.scale, likelihood.degrees_of_freedom
    )
    assert forward_p == pytest.approx(
        float(np.sum(np.outer(home.prior, away.prior) * probabilities)), abs=1e-12
    )
