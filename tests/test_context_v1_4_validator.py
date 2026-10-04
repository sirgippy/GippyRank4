"""Outcome-blind validator tests using registered state and synthetic results."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t as student_t

import gippyrank.context_v1_4_validator as validator
from gippyrank.context_v1_4_validation_protocol import (
    ProtocolError,
    load_registered_protocol,
)
from gippyrank.posterior.engine import Game, Team
from gippyrank.posterior.predictive import (
    ScheduledGame,
    predict_game,
    predictive_components,
)
from gippyrank.posterior.snapshots import load_likelihood

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_context_v1_4 as cli

run = cli.run


@pytest.fixture(scope="module")
def protocol():
    return load_registered_protocol(ROOT)


@pytest.fixture(scope="module")
def inputs(protocol):
    return validator.load_validator_inputs(ROOT, protocol)


def _write_games(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=validator.GAME_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def synthetic_games(tmp_path: Path, inputs) -> tuple[Path, list[dict[str, str]]]:
    rows = [
        {
            **item,
            "completed": "True",
            "homePoints": "21",
            "awayPoints": "14",
        }
        for item in inputs["registered_schedule"]
    ]
    path = tmp_path / "synthetic_games.csv"
    _write_games(path, rows)
    return path, rows


def _audit(path: Path, protocol, inputs):
    return validator.audit_game_source(path, protocol, inputs, ROOT)


def _write_synthetic_manifest(
    directory: Path,
    games_path: Path,
    rows: list[dict[str, str]],
    exceptions: list[dict[str, str]],
) -> tuple[Path, dict[str, object]]:
    raw_by_coverage: dict[str, list[dict[str, object]]] = {"fbs": [], "fcs": []}
    for row in rows:
        raw: dict[str, object] = dict(row)
        for field in ("id", "season", "week", "homeId", "awayId"):
            raw[field] = int(row[field])
        raw["completed"] = row["completed"] == "True"
        for field in ("homePoints", "awayPoints"):
            raw[field] = int(row[field]) if raw["completed"] else None
        raw["neutralSite"] = row["neutralSite"] == "True"
        raw["startTimeTBD"] = False
        if "fbs" in (row["homeClassification"], row["awayClassification"]):
            raw_by_coverage["fbs"].append(raw)
        if "fcs" in (row["homeClassification"], row["awayClassification"]):
            raw_by_coverage["fcs"].append(raw)

    files = []
    for coverage in ("fbs", "fcs"):
        raw_path = directory / f"raw_{coverage}.json"
        raw_path.write_text(
            json.dumps(raw_by_coverage[coverage], sort_keys=True), encoding="utf-8"
        )
        files.append(
            {
                "coverage": coverage,
                "path": raw_path.name,
                "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "source_kind": "cfbd_api_schedule",
        "retrieved_at_utc": "2026-10-05T12:00:00+00:00",
        "games_sha256": hashlib.sha256(games_path.read_bytes()).hexdigest(),
        "source_files": files,
    }
    manifest_path = directory / "source-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    (directory / "exceptions.json").write_text(
        json.dumps(exceptions, sort_keys=True), encoding="utf-8"
    )
    return manifest_path, manifest


@pytest.fixture(scope="module")
def complete_cli_runs(tmp_path_factory, inputs):
    directory = tmp_path_factory.mktemp("context-v1-4-complete-cli")
    week_5 = [row for row in inputs["registered_schedule"] if row["week"] == "5"]
    prior_weeks = [row for row in inputs["registered_schedule"] if row["week"] != "5"]
    completed_ids = {row["id"] for row in prior_weeks[:10]} | {
        row["id"] for row in week_5[:40]
    }
    rows = [
        {
            **item,
            "completed": "True" if item["id"] in completed_ids else "False",
            "homePoints": "21" if item["id"] in completed_ids else "",
            "awayPoints": "14" if item["id"] in completed_ids else "",
        }
        for item in inputs["registered_schedule"]
    ]
    exceptions = [
        {
            "game_id": row["id"],
            "disposition": "terminal_exception",
            "reason": "synthetic terminal cancellation",
            "evidence": "synthetic complete-path fixture",
        }
        for row in rows
        if row["id"] not in completed_ids
    ]
    games_path = directory / "synthetic_games.csv"
    _write_games(games_path, rows)
    manifest_path, manifest = _write_synthetic_manifest(
        directory, games_path, rows, exceptions
    )
    exceptions_path = directory / "exceptions.json"
    outputs = (directory / "successful_run_a", directory / "successful_run_b")
    captured: dict[str, object] = {}
    original_priors = validator.construct_frozen_priors
    original_construct = cli.construct_forecast_states
    original_score = cli.score_complete_source

    def capture_priors(*args, **kwargs):
        value = original_priors(*args, **kwargs)
        captured["priors"] = value
        return value

    def capture_states(*args, **kwargs):
        value = original_construct(*args, **kwargs)
        captured["states"] = value
        return value

    def capture_scores(*args, **kwargs):
        value = original_score(*args, **kwargs)
        captured["scores"] = value
        return value

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(validator, "construct_frozen_priors", capture_priors)
        patch.setattr(cli, "construct_forecast_states", capture_states)
        patch.setattr(cli, "score_complete_source", capture_scores)
        assert (
            run(
                games=games_path,
                output=outputs[0],
                source_manifest=manifest_path,
                exceptions=exceptions_path,
                root=ROOT,
            )
            == 0
        )
    assert (
        run(
            games=games_path,
            output=outputs[1],
            source_manifest=manifest_path,
            exceptions=exceptions_path,
            root=ROOT,
        )
        == 0
    )
    machine_names = (
        "game_results.csv",
        "by_origin.json",
        "evidence_audit.csv",
        "exclusions.csv",
        "summary.json",
        "provenance.json",
    )
    machine_artifacts = tuple(
        {name: (output / name).read_bytes() for name in machine_names}
        for output in outputs
    )
    artifact_names = tuple(
        tuple(sorted(path.name for path in output.iterdir())) for output in outputs
    )
    reports = tuple(
        (output / "context_v1_4_validation_result.md").read_bytes()
        for output in outputs
    )
    return {
        "directory": directory,
        "games_path": games_path,
        "rows": rows,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "exceptions_path": exceptions_path,
        "completed_ids": completed_ids,
        "exceptions": exceptions,
        "priors": captured["priors"],
        "states": captured["states"],
        "scores": captured["scores"],
        "outputs": outputs,
        "machine_names": machine_names,
        "machine_artifacts": machine_artifacts,
        "artifact_names": artifact_names,
        "reports": reports,
    }


def test_complete_synthetic_cli_path_is_deterministic(
    complete_cli_runs, protocol, inputs
):
    artifacts = complete_cli_runs
    expected_origins = {
        origin["publication_slot"] for origin in protocol.data["forecast_origins"]
    }
    expected_models = set(validator.MODELS)
    expected_artifact_names = set(artifacts["machine_names"]) | {
        "context_v1_4_validation_result.md"
    }
    for names in artifacts["artifact_names"]:
        assert set(names) == expected_artifact_names
        assert "failure.json" not in names
    assert artifacts["machine_artifacts"][0] == artifacts["machine_artifacts"][1]
    assert artifacts["reports"][0] == artifacts["reports"][1]

    evidence = list(
        csv.DictReader(
            io.StringIO(
                artifacts["machine_artifacts"][0]["evidence_audit.csv"].decode()
            )
        )
    )
    assert {(row["origin"], row["model"]) for row in evidence} == {
        (origin, model) for origin in expected_origins for model in expected_models
    }
    assert len(evidence) == 5 * 3
    assert all(row["converged"] == "True" for row in evidence)

    scores = list(
        csv.DictReader(
            io.StringIO(artifacts["machine_artifacts"][0]["game_results.csv"].decode())
        )
    )
    expected_game_ids = artifacts["completed_ids"]
    score_counts = Counter((row["game_id"], row["model"]) for row in scores)
    assert set(score_counts) == {
        (game_id, model) for game_id in expected_game_ids for model in expected_models
    }
    assert all(count == 1 for count in score_counts.values())
    assert len(scores) == len(expected_game_ids) * len(expected_models)

    week_5_ids = artifacts["completed_ids"] & {
        row["id"] for row in inputs["registered_schedule"] if row["week"] == "5"
    }
    assert len(week_5_ids) == 40
    prospective = [row for row in scores if row["game_id"] in week_5_ids]
    assert len(prospective) == len(week_5_ids) * len(expected_models)
    assert {(row["origin"], row["stratum"]) for row in prospective} == {
        ("2026-09-27", "prospective_week_5")
    }
    summary = json.loads(artifacts["machine_artifacts"][0]["summary.json"])
    assert summary["prospective_week_5"]["eligible_game_count"] == len(week_5_ids)
    assert set(summary["by_origin"]) == expected_origins
    assert summary["decision"] == "inconclusive"


def test_manifest_abort_removes_previous_success_artifacts(complete_cli_runs):
    artifacts = complete_cli_runs
    output = artifacts["outputs"][0]
    assert (output / "summary.json").is_file()
    assert (output / "game_results.csv").is_file()
    assert (output / "context_v1_4_validation_result.md").is_file()

    bad_manifest = dict(artifacts["manifest"])
    bad_manifest["games_sha256"] = "0" * 64
    bad_manifest_path = artifacts["directory"] / "bad-source-manifest.json"
    bad_manifest_path.write_text(
        json.dumps(bad_manifest, sort_keys=True), encoding="utf-8"
    )
    result = run(
        games=artifacts["games_path"],
        output=output,
        source_manifest=bad_manifest_path,
        exceptions=artifacts["exceptions_path"],
        root=ROOT,
    )
    assert result == 2
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "VALIDATION_ABORTED"
    assert failure["abort_stage"] == "source_manifest_validation"
    assert failure["candidate_scores_opened"] is False
    assert failure["decision"] == "unavailable"
    assert failure["observed_completed_count"] == len(artifacts["completed_ids"])
    assert {path.name for path in output.iterdir()} == {"failure.json"}


@pytest.mark.parametrize(
    ("failure_point", "expected_stage", "candidate_scores_opened"),
    [
        ("forecast_state", "forecast_state_construction", False),
        ("posterior", "forecast_state_construction", False),
        ("scoring", "scoring", True),
        ("aggregation", "aggregation_and_decision", True),
        ("artifact_generation", "artifact_generation", True),
    ],
)
def test_post_gate_abort_stages_clear_previous_success(
    tmp_path: Path,
    monkeypatch,
    complete_cli_runs,
    failure_point: str,
    expected_stage: str,
    candidate_scores_opened: bool,
):
    artifacts = complete_cli_runs
    output = tmp_path / "old-success"
    shutil.copytree(artifacts["outputs"][1], output)

    def abort(*args, **kwargs):
        raise ProtocolError(f"synthetic {failure_point} failure")

    if failure_point == "forecast_state":
        monkeypatch.setattr(cli, "construct_forecast_states", abort)
    elif failure_point == "posterior":
        monkeypatch.setattr(
            validator,
            "construct_frozen_priors",
            lambda *args, **kwargs: artifacts["priors"],
        )
        monkeypatch.setattr(validator, "infer_registered_posterior", abort)
    elif failure_point == "scoring":
        monkeypatch.setattr(
            cli,
            "construct_forecast_states",
            lambda *args, **kwargs: artifacts["states"],
        )
        monkeypatch.setattr(cli, "score_complete_source", abort)
    else:
        monkeypatch.setattr(
            cli,
            "construct_forecast_states",
            lambda *args, **kwargs: artifacts["states"],
        )
        monkeypatch.setattr(
            cli,
            "score_complete_source",
            lambda *args, **kwargs: artifacts["scores"],
        )
        if failure_point == "aggregation":
            monkeypatch.setattr(validator, "decide_validation", abort)
        else:

            def write_partial_then_abort(output_path, *args, **kwargs):
                (output_path / "summary.json").write_text(
                    '{"decision":"promote"}', encoding="utf-8"
                )
                abort()

            monkeypatch.setattr(cli, "write_success", write_partial_then_abort)

    result = run(
        games=artifacts["games_path"],
        output=output,
        source_manifest=artifacts["manifest_path"],
        exceptions=artifacts["exceptions_path"],
        root=ROOT,
    )
    assert result == 2
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "VALIDATION_ABORTED"
    assert failure["abort_stage"] == expected_stage
    assert failure["candidate_scores_opened"] is candidate_scores_opened
    assert failure["decision"] == "unavailable"
    assert {path.name for path in output.iterdir()} == {"failure.json"}


def test_frozen_protocol_and_validator_inputs_fail_closed(
    monkeypatch, protocol, inputs
):
    assert len(inputs["forecast_rows"]) == protocol.data["population"]["expected_count"]
    assert [item["publication_slot"] for item in protocol.data["forecast_origins"]] == [
        "2026-preseason-context-1.3",
        "2026-09-08",
        "2026-09-13",
        "2026-09-20",
        "2026-09-27",
    ]
    monkeypatch.setattr(validator, "VALIDATOR_INPUTS_SHA256", "0" * 64)
    with pytest.raises(ProtocolError, match="input hash changed"):
        validator.load_validator_inputs(ROOT, protocol)


def test_early_abort_reports_unavailable_source_counts_truthfully():
    artifact = validator.validation_abort_object(
        reason="ProtocolError: preregistration mismatch",
        abort_stage="protocol_validation",
        candidate_scores_opened=False,
        protocol=None,
        inputs=None,
        audit=None,
    )
    assert artifact["status"] == "VALIDATION_ABORTED"
    assert artifact["candidate_scores_opened"] is False
    assert artifact["registered_expected_scope"]["expected_game_count"] is None
    assert artifact["registered_expected_scope"]["expected_week_5_game_count"] is None
    assert artifact["missing_or_unresolved_game_ids"] is None


def test_incomplete_week_5_source_refuses_before_scoring(
    tmp_path: Path, protocol, synthetic_games
):
    protected = [
        ROOT / "site/data/manifest.json",
        ROOT / protocol.data["models"]["context_1_3"]["predictions_path"],
        ROOT / protocol.data["models"]["history_1_1"]["predictions_path"],
    ]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in protected]
    games_path, rows = synthetic_games
    for row in rows:
        if row["week"] == "5":
            row["completed"] = "False"
            row["homePoints"] = ""
            row["awayPoints"] = ""
    _write_games(games_path, rows)
    output = tmp_path / "negative"
    result = run(
        games=games_path,
        output=output,
        root=ROOT,
    )
    assert result == 2
    failure = json.loads((output / "failure.json").read_text())
    assert failure["status"] == "SOURCE_INCOMPLETE"
    assert failure["candidate_scores_opened"] is False
    assert failure["decision"] == "unavailable"
    assert len(failure["missing_or_unresolved_game_ids"]) == 114
    assert failure["registered_expected_scope"]["expected_week_5_game_count"] == 114
    assert not (output / "summary.json").exists()
    assert before == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in protected
    ]


def test_complete_synthetic_source_and_prospective_week_5_assignment(
    synthetic_games, protocol, inputs
):
    path, _ = synthetic_games
    audit = _audit(path, protocol, inputs)
    assert audit.status == "SOURCE_COMPLETE"
    assert audit.completed_count == len(inputs["registered_schedule"])
    week_5 = [item for item in inputs["registered_schedule"] if item["week"] == "5"]
    assert len(week_5) == 114
    assert {audit.assignments[item["id"]] for item in week_5} == {"2026-09-27"}
    assert not audit.unresolved_ids


@pytest.mark.parametrize(
    "change,expected",
    [
        ("remove", "SOURCE_INCOMPLETE"),
        ("unresolved", "SOURCE_INCOMPLETE"),
        ("no_kickoff", "unknown kickoff"),
        ("bad_score", "invalid final score"),
        ("before_first", "no earlier registered origin"),
        ("at_boundary", "wrong origin"),
        ("leakage", "forecast evidence already contains result"),
    ],
)
def test_assignment_and_source_failures(
    synthetic_games, protocol, inputs, change: str, expected: str
):
    path, rows = synthetic_games
    week5 = next(row for row in rows if row["week"] == "5")
    if change == "remove":
        rows.remove(week5)
    elif change == "unresolved":
        week5.update(completed="False", homePoints="", awayPoints="")
    elif change == "no_kickoff":
        week5["startDate"] = ""
    elif change == "bad_score":
        week5["homePoints"] = "-1"
    elif change == "before_first":
        first = next(row for row in rows if row["week"] == "1")
        first["startDate"] = protocol.data["forecast_origins"][0]["logical_cutoff_utc"]
    elif change == "at_boundary":
        week5["startDate"] = protocol.data["forecast_origins"][-1]["logical_cutoff_utc"]
    elif change == "leakage":
        included = {
            row["id"]
            for row in validator._read_csv(
                ROOT
                / protocol.data["forecast_origins"][-1]["context_files"][
                    "included_games.csv"
                ]["path"]
            )
        }
        prior_game = next(
            row for row in rows if row["id"] in included and row["week"] == "1"
        )
        prior_game["startDate"] = "2026-10-01T12:00:00+00:00"
    _write_games(path, rows)
    if expected == "SOURCE_INCOMPLETE":
        assert _audit(path, protocol, inputs).status == expected
    else:
        with pytest.raises(ProtocolError, match=expected):
            _audit(path, protocol, inputs)


def test_reschedule_uses_actual_kickoff_and_duplicates_fail_closed(
    synthetic_games, protocol, inputs
):
    path, rows = synthetic_games
    extra = {
        **rows[0],
        "id": "synthetic-reschedule",
        "week": "3",
        "startDate": "2026-09-21T18:00:00+00:00",
    }
    rows.append(extra)
    rows.append(dict(extra))
    _write_games(path, rows)
    audit = _audit(path, protocol, inputs)
    assert audit.assignments[extra["id"]] == "2026-09-20"
    assert audit.exact_duplicate_count == 1
    rows[-1]["homePoints"] = "22"
    _write_games(path, rows)
    with pytest.raises(ProtocolError, match="conflicting duplicate"):
        _audit(path, protocol, inputs)


def test_explicit_terminal_exception_is_audited(
    synthetic_games, protocol, inputs, tmp_path: Path
):
    path, rows = synthetic_games
    week5 = next(row for row in rows if row["week"] == "5")
    week5.update(completed="False", homePoints="", awayPoints="")
    _write_games(path, rows)
    exceptions = tmp_path / "exceptions.json"
    exceptions.write_text(
        json.dumps(
            [
                {
                    "game_id": week5["id"],
                    "disposition": "terminal_exception",
                    "reason": "synthetic cancellation",
                    "evidence": "fixture audit",
                }
            ]
        )
    )
    audit = validator.audit_game_source(
        path, protocol, inputs, ROOT, exceptions_path=exceptions
    )
    assert audit.status == "SOURCE_COMPLETE"
    assert audit.completed_count == len(rows) - 1
    assert audit.exclusions[0]["game_id"] == week5["id"]


def test_source_manifest_binds_canonical_and_raw_and_rejects_tbd(
    synthetic_games, tmp_path: Path
):
    path, rows = synthetic_games
    fbs: list[dict[str, object]] = []
    fcs: list[dict[str, object]] = []
    for row in rows:
        raw: dict[str, object] = dict(row)
        for field in (
            "id",
            "season",
            "week",
            "homeId",
            "awayId",
            "homePoints",
            "awayPoints",
        ):
            raw[field] = int(row[field])
        raw["completed"] = True
        raw["neutralSite"] = row["neutralSite"] == "True"
        raw["startTimeTBD"] = False
        if "fbs" in (row["homeClassification"], row["awayClassification"]):
            fbs.append(raw)
        if "fcs" in (row["homeClassification"], row["awayClassification"]):
            fcs.append(raw)
    files = []
    for coverage, items in (("fbs", fbs), ("fcs", fcs)):
        raw_path = tmp_path / f"{coverage}.json"
        raw_path.write_text(json.dumps(items), encoding="utf-8")
        files.append(
            {
                "coverage": coverage,
                "path": raw_path.name,
                "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "source_kind": "cfbd_api_schedule",
        "retrieved_at_utc": "2026-10-05T12:00:00+00:00",
        "games_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_files": files,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validator.load_source_manifest(manifest_path, path) == manifest
    fbs[0]["startTimeTBD"] = True
    (tmp_path / "fbs.json").write_text(json.dumps(fbs), encoding="utf-8")
    manifest["source_files"][0]["sha256"] = hashlib.sha256(
        (tmp_path / "fbs.json").read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ProtocolError, match="ambiguous actual kickoff"):
        validator.load_source_manifest(manifest_path, path)


def test_candidate_prior_construction_is_outcome_free(protocol, inputs):
    priors = validator.construct_frozen_priors(ROOT, protocol, inputs)
    assert all(len(teams) == 138 for teams in priors.values())
    for team_id in protocol.data["population"]["cold_start_team_ids"]:
        assert np.array_equal(
            priors["candidate"][team_id].prior,
            priors["context_1_3"][team_id].prior,
        )
    assert any(
        not np.array_equal(
            priors["candidate"][team_id].prior,
            priors["context_1_3"][team_id].prior,
        )
        for team_id in priors["candidate"]
    )


def test_registered_inference_repeats_and_nonconvergence_aborts(monkeypatch, protocol):
    likelihood = load_likelihood(ROOT / protocol.data["evidence"]["likelihood_path"])
    teams = [
        Team("A", "A", "fbs", np.array([0.7, 0.3])),
        Team("B", "B", "fbs", np.array([0.4, 0.6])),
    ]
    games = [Game("synthetic", "A", "B", "fbs", "fbs", 21, 14)]
    first = validator.infer_registered_posterior(
        teams, games, likelihood, protocol, origin="fixture", model="candidate"
    )
    second = validator.infer_registered_posterior(
        teams, games, likelihood, protocol, origin="fixture", model="candidate"
    )
    assert first.converged and second.converged
    assert first.iterations == second.iterations
    assert validator.posterior_pmfs_sha256(
        first.pmfs
    ) == validator.posterior_pmfs_sha256(second.pmfs)
    monkeypatch.setattr(
        validator,
        "infer_posterior",
        lambda *_args, **_kwargs: replace(first, converged=False),
    )
    with pytest.raises(ProtocolError, match="did not converge"):
        validator.infer_registered_posterior(
            teams, games, likelihood, protocol, origin="fixture", model="candidate"
        )


def test_exact_predictive_score_parity_and_orientation(protocol):
    likelihood = load_likelihood(ROOT / protocol.data["evidence"]["likelihood_path"])
    teams = {
        "A": Team("A", "A", "fbs", np.array([0.7, 0.3])),
        "B": Team("B", "B", "fbs", np.array([0.4, 0.6])),
    }
    state = validator.ForecastState(
        "fixture",
        "candidate",
        teams,
        {key: team.prior for key, team in teams.items()},
        {},
    )
    game = Game("synthetic", "A", "B", "fbs", "fbs", 21, 14)
    scored = validator.score_predictive_game(
        game,
        week=5,
        state=state,
        likelihood=likelihood,
        locked_fcs_ids=set(),
        fcs_population=128,
    )
    scheduled = ScheduledGame("synthetic", "A", "B", "fbs", "fbs")
    prediction = predict_game(scheduled, teams["A"], teams["B"], likelihood)
    locations, weights = predictive_components(
        scheduled, teams["A"], teams["B"], likelihood
    )
    expected_nll = -np.log(
        max(
            float(
                np.dot(
                    weights,
                    student_t.pdf(
                        (7 - locations) / likelihood.scale,
                        likelihood.degrees_of_freedom,
                    )
                    / likelihood.scale,
                )
            ),
            1e-300,
        )
    )
    assert scored["margin_nll"] == pytest.approx(expected_nll)
    assert scored["expected_home_margin"] == prediction.expected_home_margin
    assert scored["win_brier"] == pytest.approx(
        (prediction.home_win_probability - 1.0) ** 2
    )
    assert scored["coverage_80"] == int(
        prediction.margin_interval_80[0] <= 7 <= prediction.margin_interval_80[1]
    )
    assert scored["stratum"] == "prospective_week_5"


def _score_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for game_id, origin, week, delta in (
        ("a", "2026-09-20", 4, -0.2),
        ("b", "2026-09-27", 5, 0.1),
        ("c", "2026-09-27", 5, -0.3),
    ):
        for model, nll in (
            ("candidate", 4.0 + delta),
            ("context_1_3", 4.0),
            ("history_1_1", 4.2),
        ):
            rows.append(
                {
                    "game_id": game_id,
                    "origin": origin,
                    "stratum": "prospective_week_5"
                    if week == 5
                    else "candidate_score_unseen_prior_weeks",
                    "week": week,
                    "home_id": "A",
                    "away_id": "B",
                    "model": model,
                    "actual_home_margin": 7,
                    "margin_nll": nll,
                    "expected_home_margin": 5.0,
                    "margin_mae": 2.0,
                    "home_win_probability": 0.75,
                    "win_brier": 0.0625,
                    "interval_50_low": 0.0,
                    "interval_50_high": 10.0,
                    "interval_80_low": -5.0,
                    "interval_80_high": 15.0,
                    "interval_95_low": -10.0,
                    "interval_95_high": 20.0,
                    "coverage_50": 1,
                    "coverage_80": 1,
                    "coverage_95": 1,
                }
            )
    return rows


def test_aggregation_bootstrap_and_decision_are_frozen(protocol):
    rows = _score_rows()
    first = validator.aggregate_scores(rows, protocol)
    second = validator.aggregate_scores(list(reversed(rows)), protocol)
    assert validator._json_bytes(first) == validator._json_bytes(second)
    assert first["full_sample"]["eligible_game_count"] == 3
    assert first["full_sample"]["primary"]["paired_mean_delta"] == pytest.approx(
        (-0.2 + 0.1 - 0.3) / 3
    )
    assert first["prospective_week_5"]["eligible_game_count"] == 2
    assert first["decision"] == "inconclusive"
    full = {
        "eligible_game_count": protocol.data["evidence"]["minimum_full_sample_games"],
        "primary": {
            "paired_mean_delta": -0.1,
            "paired_mean_95_percent_interval": [-0.2, -0.01],
        },
    }
    prospective = {
        "eligible_game_count": protocol.data["evidence"]["minimum_week_6_games"]
    }
    decide = validator.decide_validation
    assert (
        decide(
            full, prospective, protocol, source_complete=True, guardrails_passed=True
        )
        == "promote"
    )
    full["primary"] = {
        "paired_mean_delta": 0.02,
        "paired_mean_95_percent_interval": [-0.01, 0.04],
    }
    assert (
        decide(
            full, prospective, protocol, source_complete=True, guardrails_passed=True
        )
        == "promote"
    )
    full["primary"] = {
        "paired_mean_delta": 0.05,
        "paired_mean_95_percent_interval": [0.01, 0.08],
    }
    assert (
        decide(
            full, prospective, protocol, source_complete=True, guardrails_passed=True
        )
        == "retain_context_1_3"
    )
    full["eligible_game_count"] -= 1
    assert (
        decide(
            full, prospective, protocol, source_complete=True, guardrails_passed=True
        )
        == "inconclusive"
    )
    full["eligible_game_count"] += 1
    prospective["eligible_game_count"] -= 1
    assert (
        decide(
            full, prospective, protocol, source_complete=True, guardrails_passed=True
        )
        == "inconclusive"
    )
    with pytest.raises(ProtocolError):
        decide(
            full, prospective, protocol, source_complete=False, guardrails_passed=True
        )
    with pytest.raises(ProtocolError):
        decide(
            full, prospective, protocol, source_complete=True, guardrails_passed=False
        )


def test_fixture_artifact_writers_are_byte_deterministic(tmp_path: Path, protocol):
    rows = _score_rows()
    summary = validator.aggregate_scores(rows, protocol)
    states = {
        ("2026-09-20", "candidate"): validator.ForecastState(
            "2026-09-20",
            "candidate",
            {},
            {},
            {
                "origin": "2026-09-20",
                "model": "candidate",
                "posterior_sha256": "fixture",
            },
        )
    }
    audit = validator.SourceAudit(
        "SOURCE_COMPLETE", (), {}, (), (), (), 0, "fixture-source-hash"
    )
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        validator.write_success(
            output,
            rows,
            states,
            audit,
            summary,
            protocol,
            {"games_sha256": "fixture-source-hash"},
        )
    names = sorted(path.name for path in outputs[0].iterdir())
    assert names == sorted(path.name for path in outputs[1].iterdir())
    assert all(
        (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
        for name in names
    )
    assert "failure.json" not in names

    validator.write_failure(outputs[0], {"status": "SOURCE_INCOMPLETE"})
    assert sorted(path.name for path in outputs[0].iterdir()) == ["failure.json"]
    validator.write_source_audit(outputs[0], {"status": "SOURCE_COMPLETE"})
    assert sorted(path.name for path in outputs[0].iterdir()) == ["source_audit.json"]
