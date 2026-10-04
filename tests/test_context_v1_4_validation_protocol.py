"""Preregistration checks use pinned sources and synthetic game rows only."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gippyrank.context_v1_4_validation_protocol import (
    ProtocolError,
    assign_completed_games_to_origins,
    classify_primary,
    load_registered_protocol,
    paired_mean_bootstrap,
)

ROOT = Path(__file__).resolve().parents[1]


def _changed_config(tmp_path: Path, change: Callable[[dict[str, Any]], None]) -> Path:
    data = json.loads((ROOT / "config/context_v1_4_validation.json").read_text())
    change(data)
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _game(game_id: str, week: int, kickoff: str) -> dict[str, str]:
    return {
        "id": game_id,
        "season": "2026",
        "week": str(week),
        "seasonType": "regular",
        "startDate": kickoff,
        "completed": "True",
        "homeId": "333",
        "awayId": "151",
        "homeClassification": "fbs",
        "awayClassification": "fbs",
        "homePoints": "21",
        "awayPoints": "14",
    }


def test_registration_preserves_frozen_candidate_and_amendment() -> None:
    protocol = load_registered_protocol(ROOT)
    data = protocol.data
    provenance = json.loads(
        (
            ROOT / "data/processed/context_v1_4_validation/preregistration.json"
        ).read_text()
    )
    assert provenance["validation_config_sha256"] == protocol.sha256
    assert (
        provenance["candidate_semantics_sha256"]
        == (data["models"]["candidate"]["candidate_semantics_sha256"])
    )
    assert (
        provenance["context_1_4_2026_outcome_evaluation_in_this_registration"] is False
    )
    assert provenance["protocol_amended_before_2026_context_1_4_score_access"] is True
    assert (
        data["registration"]["candidate_2026_predictive_scores_at_amendment"]
        == "unopened"
    )
    assert (
        data["registration"]["protocol_amendment"]["candidate_score_influence"]
        == "none"
    )
    assert data["population"]["expected_count"] == 138
    assert data["population"]["cold_start_team_ids"] == ["16", "2449"]
    assert data["analysis"]["primary"]["score"] == "posterior_predictive_margin_nll"
    assert data["analysis"]["primary"]["negative_difference_favors"] == "candidate"
    assert (
        data["later_postseason_corroboration"]["blocks_issue_164_completion"] is False
    )
    assert '"alpha"' not in json.dumps(data)
    bridge = data["development_predictive_bridge"]
    assert provenance["development_predictive_bridge_sha256"] == bridge["result_sha256"]
    bridge_result = json.loads((ROOT / bridge["result_path"]).read_text())
    assert bridge_result["pooled"]["context_1_3"]["n"] == 2700
    assert bridge_result["pooled"]["context_1_4"]["n"] == 2700


def test_only_five_official_origins_and_week_6_state() -> None:
    data = load_registered_protocol(ROOT).data
    assert [origin["publication_slot"] for origin in data["forecast_origins"]] == [
        "2026-preseason-context-1.3",
        "2026-09-08",
        "2026-09-13",
        "2026-09-20",
        "2026-09-27",
    ]
    assert [origin["included_game_count"] for origin in data["forecast_origins"]] == [
        0,
        172,
        291,
        410,
        530,
    ]
    assert data["evidence"]["week_6_origin"] == "2026-09-27"
    assert (
        data["forecast_origins"][0]["evidence_status"]
        == "retrospective_official_backfill"
    )


def test_unique_latest_pre_kickoff_assignment_and_leakage_guard() -> None:
    protocol = load_registered_protocol(ROOT)
    games = [
        _game("synthetic-1", 1, "2026-09-05T16:00:00Z"),
        _game("synthetic-2", 2, "2026-09-11T00:00:00Z"),
        _game("synthetic-3", 3, "2026-09-17T23:30:00Z"),
        _game("synthetic-4", 4, "2026-09-24T23:30:00Z"),
        _game("synthetic-5", 5, "2026-10-02T00:00:00Z"),
        _game("synthetic-6", 6, "2026-10-07T00:00:00Z"),
    ]
    assert assign_completed_games_to_origins(games, protocol, ROOT) == {
        "synthetic-1": "2026-preseason-context-1.3",
        "synthetic-2": "2026-09-08",
        "synthetic-3": "2026-09-13",
        "synthetic-4": "2026-09-20",
        "synthetic-5": "2026-09-27",
        "synthetic-6": "2026-09-27",
    }
    with pytest.raises(ProtocolError, match="duplicate canonical"):
        assign_completed_games_to_origins([games[0], games[0]], protocol, ROOT)
    with pytest.raises(ProtocolError, match="no earlier official origin"):
        assign_completed_games_to_origins(
            [_game("too-early", 1, "2026-08-22T23:59:59Z")], protocol, ROOT
        )
    latest = protocol.data["forecast_origins"][-1]
    with (ROOT / latest["context_files"]["included_games.csv"]["path"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        already_included = next(csv.DictReader(handle))["id"]
    with pytest.raises(ProtocolError, match="already included"):
        assign_completed_games_to_origins(
            [_game(already_included, 6, "2026-10-07T00:00:00Z")], protocol, ROOT
        )


def test_development_strata_cutpoints_unchanged() -> None:
    strata = load_registered_protocol(ROOT).data["descriptive_strata"][
        "positive_context_only_magnitude"
    ]
    with (ROOT / strata["source_path"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {int(row["season"]) for row in rows} == set(strata["development_seasons"])
    positive = [
        float(row["context_only_subtotal"])
        for row in rows
        if float(row["context_only_subtotal"]) > 0
    ]
    assert np.quantile(positive, 0.5, method="linear") == pytest.approx(
        strata["positive_only_q50"], abs=1e-12
    )
    assert np.quantile(positive, 0.75, method="linear") == pytest.approx(
        strata["positive_only_q75"], abs=1e-12
    )


def test_candidate_decision_or_origin_mutation_fails_closed(tmp_path: Path) -> None:
    bad_candidate = _changed_config(
        tmp_path,
        lambda data: data["models"]["candidate"].update(
            candidate_semantics_sha256="0" * 64
        ),
    )
    with pytest.raises(ProtocolError, match="candidate identity"):
        load_registered_protocol(ROOT, bad_candidate)
    bad_decision = _changed_config(
        tmp_path,
        lambda data: data["analysis"]["decision"]["retain_context_1_3"].update(
            paired_interval_low_gt=-0.01
        ),
    )
    with pytest.raises(ProtocolError, match="decision rule"):
        load_registered_protocol(ROOT, bad_decision)
    bad_origin = _changed_config(
        tmp_path,
        lambda data: data["forecast_origins"][0]["context_files"][
            "metadata.json"
        ].update(sha256="0" * 64),
    )
    with pytest.raises(ProtocolError, match="source hash changed"):
        load_registered_protocol(ROOT, bad_origin)


def test_valid_unregistered_change_and_population_mutation_fail(tmp_path: Path) -> None:
    changed = _changed_config(
        tmp_path,
        lambda data: data["population"].update(cold_start_policy="changed"),
    )
    with pytest.raises(ProtocolError, match="registration provenance"):
        load_registered_protocol(ROOT, changed)
    bad = _changed_config(
        tmp_path,
        lambda data: data["population"]["expected_team_ids"].pop(),
    )
    with pytest.raises(ProtocolError, match="population"):
        load_registered_protocol(ROOT, bad)


def test_synthetic_paired_bootstrap_is_deterministic_and_game_order_independent() -> (
    None
):
    protocol = load_registered_protocol(ROOT)
    differences = {"game-2": -0.02, "game-1": -0.04, "game-3": 0.01, "game-4": -0.03}
    result = paired_mean_bootstrap(differences, protocol)
    assert result == paired_mean_bootstrap(
        dict(reversed(list(differences.items()))), protocol
    )
    assert result[0] == pytest.approx(-0.02)
    assert result[1] <= result[0] <= result[2]
    with pytest.raises(ValueError, match="finite"):
        paired_mean_bootstrap({"game-1": float("nan")}, protocol)


@pytest.mark.parametrize(
    ("mean", "low", "high", "n", "week_6", "expected"),
    [
        (-0.02, -0.04, -0.001, 600, 80, "promote"),
        (0.02, 0.001, 0.04, 600, 80, "retain_context_1_3"),
        (0.02, -0.01, 0.04, 600, 80, "promote"),
        (-0.02, -0.04, 0.001, 600, 80, "promote"),
        (-0.02, -0.04, -0.001, 499, 80, "inconclusive"),
        (-0.02, -0.04, -0.001, 600, 39, "inconclusive"),
    ],
)
def test_synthetic_asymmetric_decision_rule(
    mean: float, low: float, high: float, n: int, week_6: int, expected: str
) -> None:
    protocol = load_registered_protocol(ROOT)
    assert classify_primary(mean, low, high, n, week_6, protocol) == expected
