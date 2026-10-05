"""Regression checks for the repaired coverage-aware Context 1.4 transfer path."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from gippyrank.context_db_repair import (
    DB_COVERAGE,
    DB_SUM,
    MODEL_FEATURES,
    DBEvidence,
    current_repaired_features,
    historical_repaired_features,
)
from gippyrank.context_prior import InferenceRow
from gippyrank.context_prior_v1_3 import _direct_rank_model_from_metadata
from gippyrank.context_prior_v1_4 import (
    construct_production_prior,
    production_spec,
)
from gippyrank.site_data import (
    SiteDataValidationError,
    _project_retained_kickoff_certainty,
)
from gippyrank.site_preseason_evidence import build_preseason_input_projection

ROOT = Path(__file__).resolve().parents[1]
ANNUAL = ROOT / "data/processed/preseason/context_v1_4/annual/2026"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _forecast(team_id: str) -> dict[str, object]:
    for line in (
        (ROOT / "data/processed/context_v1_4_validation/validator_inputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        record = json.loads(line)
        if record["kind"] == "forecast_row" and record["value"]["team_id"] == team_id:
            return record["value"]
    raise AssertionError(f"missing forecast team {team_id}")


def _published(team_id: str) -> dict[str, str]:
    return next(
        row for row in _rows(ANNUAL / "predictions.csv") if row["team_id"] == team_id
    )


def _production(team_id: str):
    source = _forecast(team_id)
    transfer, _ = current_repaired_features(ROOT)
    values = {name: source["features"].get(name) for name in MODEL_FEATURES}
    values.update(transfer[team_id])
    row = InferenceRow(
        season=2026,
        subdivision="fbs",
        team_id=team_id,
        team_name=source["team_name"],
        population=138,
        lag1_z=tuple(source["lag1_z"]),
        lag_zs=(),
        features=values,
    )
    model = _direct_rank_model_from_metadata(
        json.loads((ANNUAL / "fitted_model.json").read_text())["model"]
    )
    return construct_production_prior(model, row)


def test_repaired_auburn_reaches_production_prior() -> None:
    values, evidence = current_repaired_features(ROOT)
    auburn = evidence["2"]
    assert auburn.incoming_db_count == 6
    assert auburn.observed_db_impact_count == 6
    assert auburn.observed_db_impact_sum == pytest.approx(3.3319573489815544)
    assert auburn.status == "complete"
    assert auburn.db_impact_coverage_fraction == 1.0
    assert values["2"][DB_SUM] == pytest.approx(auburn.observed_db_impact_sum)
    assert values["2"][DB_COVERAGE] == 1.0
    assert production_spec()["db_features"] == [DB_SUM, DB_COVERAGE]
    assert np.allclose(
        _production("2").pmf,
        np.asarray(json.loads(_published("2")["pmf"])),
        rtol=0,
        atol=1e-12,
    )


def test_partial_observed_sum_is_consumed_with_coverage() -> None:
    values, evidence = current_repaired_features(ROOT)
    memphis = evidence["235"]
    assert memphis.status == "partial"
    assert memphis.observed_db_impact_count > 0
    assert memphis.missing_db_impact_count > 0
    assert memphis.observed_db_impact_sum > 0
    assert values["235"][DB_SUM] == memphis.observed_db_impact_sum
    assert 0 < values["235"][DB_COVERAGE] < 1
    assert float(_published("235")[DB_SUM]) == pytest.approx(
        memphis.observed_db_impact_sum
    )
    assert np.allclose(
        _production("235").pmf,
        np.asarray(json.loads(_published("235")["pmf"])),
        rtol=0,
        atol=1e-12,
    )


def test_fitted_model_distinguishes_same_sum_at_different_coverage() -> None:
    source = _forecast("2")
    model = _direct_rank_model_from_metadata(
        json.loads((ANNUAL / "fitted_model.json").read_text())["model"]
    )
    values = {name: source["features"].get(name) for name in MODEL_FEATURES}
    values[DB_SUM] = 3.0
    values[DB_COVERAGE] = 1.0
    complete = InferenceRow(
        season=2026,
        subdivision="fbs",
        team_id="2",
        team_name="Auburn",
        population=138,
        lag1_z=tuple(source["lag1_z"]),
        lag_zs=(),
        features=values,
    )
    partial = InferenceRow(
        season=2026,
        subdivision="fbs",
        team_id="2",
        team_name="Auburn",
        population=138,
        lag1_z=tuple(source["lag1_z"]),
        lag_zs=(),
        features={**values, DB_COVERAGE: 0.5},
    )
    assert not np.allclose(
        construct_production_prior(model, complete).pmf,
        construct_production_prior(model, partial).pmf,
        rtol=0,
        atol=1e-12,
    )


def test_natural_zero_and_no_observed_are_distinct() -> None:
    natural = DBEvidence(0, 0, 0.0)
    none_observed = DBEvidence(3, 0, 0.0)
    assert natural.status == "natural_zero"
    assert natural.model_features() == {DB_SUM: 0.0, DB_COVERAGE: 1.0}
    assert none_observed.status == "no_observed"
    assert none_observed.missing_db_impact_count == 3
    assert none_observed.model_features() == {DB_SUM: 0.0, DB_COVERAGE: 0.0}


def test_historical_player_counts_match_corrected_team_panel() -> None:
    values, evidence = historical_repaired_features(ROOT)
    assert len(evidence) == 655
    assert len(values) == 2744
    assert any(
        db.status == "partial" and db.observed_db_impact_sum != 0
        for db in evidence.values()
    )
    assert all(
        values[key][DB_SUM] == db.observed_db_impact_sum
        and values[key][DB_COVERAGE] == db.db_impact_coverage_fraction
        for key, db in evidence.items()
    )
    assert all(
        values[key][DB_SUM] is None and values[key][DB_COVERAGE] is None
        for key in values
        if key[0] < 2021
    )


def test_published_site_preseason_evidence_matches_model_input() -> None:
    metadata_path = (
        ROOT / "data/processed/snapshots/2026/"
        "2026-preseason-context-v1.4-db-repair/predictive/context/metadata.json"
    )
    projection = build_preseason_input_projection(
        root=ROOT,
        metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
        team_ids=["2", "235", "2005"],
    )
    assert projection is not None
    teams = projection["teams"]
    auburn = {field["id"]: field for field in teams["2"]["transfers"]["fields"]}
    assert auburn[DB_SUM]["raw_value"] == pytest.approx(3.3319573489815544)
    assert auburn[DB_COVERAGE]["display_value"] == "6/6 complete"
    partial = {field["id"]: field for field in teams["235"]["transfers"]["fields"]}
    assert partial[DB_SUM]["raw_value"] > 0
    assert "partial" in partial[DB_COVERAGE]["display_value"]
    assert "Observed incoming DB contributions only" in partial[DB_SUM]["detail"]


def test_historical_replay_audit_has_cutoff_safe_game_evidence() -> None:
    path = ROOT / "data/processed/context_db_repair_172/historical_predictive.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    for row in report["season_results"].values():
        assert row["included_game_count"] > 0
        assert row["latest_included_start_date"] <= row["effective_cutoff"]
        assert (
            row["current_context_1_4"]["games"]
            == row["corrected_context_1_4"]["games"]
            == row["history_1_1"]["games"]
        )


def test_week6_retained_kickoff_certainty_has_checked_source_identity() -> None:
    path = (
        ROOT / "data/processed/snapshots/2026/"
        "2026-weekly-2026-10-04T14-51-53.251364Z-context-v1.4-db-repair/"
        "predictive/context"
    )
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    source = json.loads(
        (
            ROOT
            / "site/data/team-seasons"
            / f"{metadata['source_evidence_snapshot_id']}.json"
        ).read_text(encoding="utf-8")
    )
    assert source["kickoff_time_certainty_version"] == "1"
    assert source.get("kickoff_time_certainty_provenance") is None
    candidate = json.loads((path / "team_seasons.json").read_text(encoding="utf-8"))
    projected = _project_retained_kickoff_certainty(candidate, metadata, ROOT)
    assert (
        projected["kickoff_time_certainty_provenance"][
            "retained_source_identity_location"
        ]
        == "team_season_root"
    )
    assert all(
        isinstance(game["kickoff_time_known"], bool)
        for team in projected["teams"].values()
        for game in team["games"]
    )
    with pytest.raises(SiteDataValidationError, match="retained kickoff provenance"):
        _project_retained_kickoff_certainty(
            candidate,
            {**metadata, "effective_cutoff": "2026-10-03T00:00:00+00:00"},
            ROOT,
        )
