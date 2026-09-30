from __future__ import annotations

import csv
import importlib.util
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from gippyrank.transfer_oracle import (
    TransferRecord,
    UsageRecord,
    aggregate_team_features,
    transfer_identity_key,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/materialize_preseason_context_prior_v1_3_research_features.py"
spec = importlib.util.spec_from_file_location("materialize_context_1_3", SCRIPT)
assert spec is not None and spec.loader is not None
materializer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(materializer)

AUDIT_SCRIPT = ROOT / "scripts/audit_transfer_data_repair.py"
audit_spec = importlib.util.spec_from_file_location("transfer_data_repair", AUDIT_SCRIPT)
assert audit_spec is not None and audit_spec.loader is not None
audit_module = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(audit_module)


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


def test_mixed_resolved_and_zero_players_keep_positive_usage_in_materializer(
    tmp_path: Path, monkeypatch
) -> None:
    historical = tmp_path / "historical.csv"
    historical_teams = tmp_path / "historical_teams.csv"
    historical_players = tmp_path / "historical_players.csv"
    current_teams = tmp_path / "current_teams.csv"

    def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)

    write_csv(
        current_teams,
        [],
        ["season", "team_id"],
    )
    write_csv(
        historical,
        [{"season": "2022", "team_id": "20", "transfer_in_prior_usage_sum": ""}],
        ["season", "team_id", "transfer_in_prior_usage_sum"],
    )
    write_csv(
        historical_teams,
        [
            {
                "season": "2022",
                "team_id": "20",
                "incoming_transfer_count": "2",
                "d5_resolution_failure_count": "0",
                "d5_applicability_unknown_count": "0",
            }
        ],
        [
            "season",
            "team_id",
            "incoming_transfer_count",
            "d5_resolution_failure_count",
            "d5_applicability_unknown_count",
        ],
    )
    portal_audit_rows = [
        {
            "portal_index": "0",
            "season": "2022",
            "player_name": "Resolved Player",
            "origin": "Alpha",
            "destination": "Beta",
            "destination_team_id": "20",
            "transfer_date": "2022-07-01",
            "portal_player_id": "resolved-id",
            "in_model_relevant_population": "True",
            "d5_resolution_category": audit_module.D5_RESOLVED,
            "d5_feature_value": "0.25",
            "usage_join_status": "joined",
            "d5_applicability_reason": "positive numeric usage",
        },
        {
            "portal_index": "1",
            "season": "2022",
            "player_name": "Zero Player",
            "origin": "Alpha",
            "destination": "Beta",
            "destination_team_id": "20",
            "transfer_date": "2022-07-02",
            "portal_player_id": "zero-id",
            "in_model_relevant_population": "True",
            "d5_resolution_category": audit_module.D5_ZERO,
            "d5_feature_value": "0.0",
            "usage_join_status": "no_usage_record",
            "d5_applicability_reason": "verified non-participant",
        },
    ]
    write_csv(
        historical_players,
        portal_audit_rows,
        list(portal_audit_rows[0]),
    )
    monkeypatch.setattr(audit_module, "CURRENT_TEAMS", current_teams)
    monkeypatch.setattr(audit_module, "HISTORICAL", historical)
    monkeypatch.setattr(audit_module, "HISTORICAL_TEAMS", historical_teams)
    monkeypatch.setattr(audit_module, "HISTORICAL_PLAYERS", historical_players)

    baseline = [
        {
            "season": 2022,
            "team_id": "20",
            "team_name": "Beta",
            "availability_status": "partial",
            "reason_codes": "historical_aggregate_null",
        }
    ]
    repaired_rows, changes, zero_contributors = audit_module._historical_zero_repairs(
        baseline
    )
    assert repaired_rows[0]["post_repair_observed_usage_sum"] == 0.25
    assert len(changes) == 1
    assert len(zero_contributors) == 1
    assert zero_contributors[0]["player"] == "Zero Player"
    assert zero_contributors[0]["d5_resolution_category"] == audit_module.D5_ZERO

    evidence_path = tmp_path / "zero_contributors.csv"
    write_csv(evidence_path, zero_contributors, list(zero_contributors[0]))
    records = [
        TransferRecord(
            season=2022,
            player_name="Resolved Player",
            origin="Alpha",
            destination="Beta",
            position="WR",
            transfer_date=date(2022, 7, 1),
            rating=None,
            stars=None,
            eligibility=None,
            player_id="resolved-id",
        ),
        TransferRecord(
            season=2022,
            player_name="Zero Player",
            origin="Alpha",
            destination="Beta",
            position="WR",
            transfer_date=date(2022, 7, 2),
            rating=None,
            stars=None,
            eligibility=None,
            player_id="zero-id",
        ),
    ]
    usage = [
        UsageRecord(
            season=2021,
            player_name="Resolved Player",
            team="Alpha",
            position="WR",
            overall_usage=0.25,
            player_id="resolved-id",
        )
    ]
    context_rows = [
        SimpleNamespace(
            season=2022,
            subdivision="fbs",
            team_id="20",
            team_name="Beta",
            features={},
        )
    ]
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
            usage,
            {2021, 2022, 2023, 2024, 2025},
            set(range(2020, 2025)),
        ),
    )
    monkeypatch.setattr(
        materializer.position_groups,
        "load_position_features",
        lambda *_: SimpleNamespace(values={}),
    )
    observed_zero_keys: list[set[tuple[int, str, str, str, str, str | None]]] = []
    real_aggregate = aggregate_team_features

    def checked_aggregate(*args, **kwargs):
        observed_zero_keys.append(kwargs["verified_zero_usage_keys"])
        return real_aggregate(*args, **kwargs)

    monkeypatch.setattr(
        materializer.transfer_research, "aggregate_team_features", checked_aggregate
    )

    result_rows = materializer.run(
        source_root=tmp_path,
        transfer_root=tmp_path,
        output=tmp_path / "mixed-panel.csv",
        zero_evidence=evidence_path,
    )

    assert observed_zero_keys == [{transfer_identity_key(records[1])}]
    assert result_rows[0]["transfer_in_prior_usage_sum"] == 0.25
