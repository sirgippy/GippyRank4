"""Preregistration checks use source identities and synthetic differences only."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gippyrank.context_v1_4_validation_protocol import (
    ProtocolError,
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


def test_registered_sources_and_locked_2026_population() -> None:
    protocol = load_registered_protocol(ROOT)
    data = protocol.data
    assert len(protocol.sha256) == 64
    provenance = json.loads(
        (
            ROOT / "data/processed/context_v1_4_validation/preregistration.json"
        ).read_text()
    )
    assert provenance["validation_config_sha256"] == protocol.sha256
    assert (
        provenance["candidate_semantics_sha256"]
        == data["models"]["candidate"]["candidate_semantics_sha256"]
    )
    assert (
        provenance["context_1_4_2026_outcome_evaluation_in_this_registration"] is False
    )
    assert data["population"]["expected_count"] == 138
    assert data["population"]["cold_start_team_ids"] == ["16", "2449"]
    assert (
        data["registration"]["candidate_2026_outcome_scoring_at_registration"]
        == "unopened"
    )
    assert '"alpha"' not in json.dumps(data)
    assert data["analysis"]["primary"]["negative_difference_favors"] == "candidate"
    assert (
        data["analysis"]["decision"]["minimum_practical_effect_nats_per_team"] == 0.01
    )


def test_checkpoints_are_the_retained_2025_panel_shifted_364_days() -> None:
    protocol = load_registered_protocol(ROOT)
    old = json.loads(
        (ROOT / "data/processed/posterior_backtest/2025_rolling.json").read_text()
    )
    old_cutoffs = [datetime.fromisoformat(item["cutoff"]) for item in old["cutoffs"]]
    new_cutoffs = [
        datetime.fromisoformat(item["cutoff_utc"])
        for item in protocol.data["checkpoints"]
    ]
    assert new_cutoffs == [cutoff + timedelta(days=364) for cutoff in old_cutoffs]
    assert protocol.data["evidence"]["preseason_game_count"] == 0
    assert protocol.data["evidence"]["result_availability_lag_hours"] == 48


def test_descriptive_cutpoints_are_development_only() -> None:
    protocol = load_registered_protocol(ROOT)
    strata = protocol.data["descriptive_strata"]["positive_context_only_magnitude"]
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


def test_candidate_or_decision_mutation_fails_closed(tmp_path: Path) -> None:
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
        lambda data: data["analysis"]["decision"]["promote"].update(
            mean_difference_lte=-0.005
        ),
    )
    with pytest.raises(ProtocolError, match="decision rule"):
        load_registered_protocol(ROOT, bad_decision)


def test_valid_unregistered_config_change_is_rejected(tmp_path: Path) -> None:
    changed = _changed_config(
        tmp_path,
        lambda data: data["population"].update(minimum_scored_teams_for_decision=133),
    )
    with pytest.raises(ProtocolError, match="registration provenance"):
        load_registered_protocol(ROOT, changed)


def test_population_mutation_fails_closed(tmp_path: Path) -> None:
    bad = _changed_config(
        tmp_path,
        lambda data: data["population"]["expected_team_ids"].pop(),
    )
    with pytest.raises(ProtocolError, match="population"):
        load_registered_protocol(ROOT, bad)


def test_synthetic_paired_bootstrap_is_deterministic_and_order_independent() -> None:
    protocol = load_registered_protocol(ROOT)
    differences = {"2": -0.02, "1": -0.04, "3": 0.01, "4": -0.03}
    result = paired_mean_bootstrap(differences, protocol)
    assert result == paired_mean_bootstrap(
        dict(reversed(list(differences.items()))), protocol
    )
    assert result[0] == pytest.approx(-0.02)
    assert result[1] <= result[0] <= result[2]
    with pytest.raises(ValueError, match="finite"):
        paired_mean_bootstrap({"1": float("nan")}, protocol)


@pytest.mark.parametrize(
    ("mean", "low", "high", "n", "expected"),
    [
        (-0.02, -0.04, -0.001, 138, "promote"),
        (0.02, 0.001, 0.04, 138, "retain_context_1_3"),
        (-0.02, -0.04, 0.001, 138, "inconclusive"),
        (-0.005, -0.009, -0.001, 138, "inconclusive"),
        (-0.02, -0.04, -0.001, 131, "inconclusive"),
    ],
)
def test_synthetic_decision_rule(
    mean: float, low: float, high: float, n: int, expected: str
) -> None:
    protocol = load_registered_protocol(ROOT)
    assert classify_primary(mean, low, high, n, protocol) == expected
