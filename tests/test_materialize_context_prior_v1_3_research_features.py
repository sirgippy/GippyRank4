from __future__ import annotations

import csv
import importlib.util
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from gippyrank.transfer_oracle import TransferRecord, aggregate_team_features

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/materialize_preseason_context_prior_v1_3_research_features.py"
spec = importlib.util.spec_from_file_location("materialize_context_1_3", SCRIPT)
assert spec is not None and spec.loader is not None
materializer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(materializer)


def test_materializer_applies_retained_zero_evidence_to_historical_features(
    tmp_path: Path, monkeypatch
) -> None:
    evidence_path = ROOT / "data/processed/transfer_data_repair/zero_contributors.csv"
    with evidence_path.open(newline="", encoding="utf-8") as handle:
        evidence_rows = list(csv.DictReader(handle))
    by_index = {int(row["portal_index"]): row for row in evidence_rows}
    records = [
        TransferRecord(
            season=1999,
            player_name="unused placeholder",
            origin=None,
            destination=None,
            position=None,
            transfer_date=None,
            rating=None,
            stars=None,
            eligibility=None,
        )
        for _ in range(max(by_index) + 1)
    ]
    for portal_index, row in by_index.items():
        records[portal_index] = TransferRecord(
            season=int(row["season"]),
            player_name=row["player"],
            origin=row["source_team"],
            destination=row["destination"],
            position="WR",
            transfer_date=date.fromisoformat(row["transfer_date"]),
            rating=None,
            stars=None,
            eligibility=None,
            player_id=row["portal_player_id"] or None,
        )
    teams = {
        (int(row["season"]), row["destination_team_id"]): row["destination"]
        for row in evidence_rows
    }
    context_rows = [
        SimpleNamespace(
            season=season,
            subdivision="fbs",
            team_id=team_id,
            team_name=team_name,
            features={},
        )
        for (season, team_id), team_name in sorted(teams.items())
    ]
    count = len(context_rows)
    expected_changes_path = ROOT / "data/processed/transfer_data_repair/changes.csv"
    with expected_changes_path.open(newline="", encoding="utf-8") as handle:
        expected_changes = list(csv.DictReader(handle))

    monkeypatch.setattr(materializer.transfer_research, "configure_source_root", lambda _: None)
    monkeypatch.setattr(materializer.v1, "load_rows", lambda **_: (context_rows, None, None))
    monkeypatch.setattr(materializer.c12, "feature_index", dict)
    monkeypatch.setattr(materializer.c12, "cached_tenures", dict)
    monkeypatch.setattr(
        materializer.c12, "attach_context", lambda rows, *_: (rows, None)
    )
    monkeypatch.setattr(
        materializer.transfer_research,
        "load_raw_transfer_data",
        lambda _: (
            records,
            [],
            {2021, 2022, 2023, 2024, 2025},
            set(range(2020, 2025)),
        ),
    )
    monkeypatch.setattr(
        materializer.position_groups,
        "load_position_features",
        lambda *_: SimpleNamespace(values={}),
    )

    real_aggregate = aggregate_team_features
    observed_aggregates: list[dict[tuple[int, str, str], dict[str, float | None]]] = []

    def checked_aggregate(*args, **kwargs):
        assert len(kwargs["verified_zero_usage_keys"]) == len(evidence_rows)
        result = real_aggregate(*args, **kwargs)
        observed_aggregates.append(result)
        return result

    monkeypatch.setattr(
        materializer.transfer_research, "aggregate_team_features", checked_aggregate
    )

    rows = materializer.run(
        source_root=tmp_path,
        transfer_root=tmp_path,
        output=tmp_path / "panel.csv",
        zero_evidence=evidence_path,
    )

    assert len(observed_aggregates) == 1
    assert count == len(expected_changes) == 21
    assert len(rows) == count
    assert len(evidence_rows) == 46
    assert all(float(row["new_usage_value"]) == 0.0 for row in expected_changes)
    assert all(row["transfer_in_prior_usage_sum"] == 0.0 for row in rows)
    assert all(
        features["transfer_in_prior_usage_sum"] == 0.0
        for features in observed_aggregates[0].values()
    )
