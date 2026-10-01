from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pytest

from gippyrank.transfer_oracle import aggregate_team_features

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/replay_historical_transfer_features.py"
MODEL_SOURCE_ROOT = ROOT / "tests/fixtures/historical_materializer"
spec = importlib.util.spec_from_file_location("historical_transfer_replay", SCRIPT)
assert spec is not None and spec.loader is not None
replay = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = replay
spec.loader.exec_module(replay)


def _source_fixture(root: Path) -> tuple[Path, Path, dict[tuple[str, int], bytes]]:
    raw_root = root / "raw"
    manifest_path = root / "source_manifest.json"
    payloads: dict[tuple[str, int], bytes] = {}
    manifest_items = []
    season_sets = {
        "portal": range(2021, 2026),
        "usage": range(2020, 2025),
        "stats": range(2020, 2025),
    }
    for kind, seasons in season_sets.items():
        for season in seasons:
            rows: list[dict[str, object]] = []
            if kind == "portal" and season == 2022:
                rows = [
                    {
                        "season": 2022,
                        "firstName": "Taylor",
                        "lastName": "Powell",
                        "origin": "Troy",
                        "destination": "Eastern Michigan",
                        "position": "WR",
                        "transferDate": "2022-07-01",
                    },
                    {
                        "season": 2022,
                        "firstName": "Zero",
                        "lastName": "Player",
                        "origin": "Alpha",
                        "destination": "Beta",
                        "position": "TE",
                        "transferDate": "2022-07-02",
                        "id": "zero-id",
                    },
                ]
            if kind == "usage" and season == 2021:
                rows = [
                    {
                        "season": 2021,
                        "name": "Taylor Powell",
                        "team": "Troy",
                        "position": "WR",
                        "usage": {"overall": 0.519},
                    },
                    {
                        "season": 2021,
                        "name": "Taylor Powell",
                        "team": "Troy",
                        "position": "WR",
                        "usage": {"overall": 0.25},
                    },
                ]
            payload = json.dumps(rows, separators=(",", ":")).encode("utf-8")
            relative = Path(f"cfbd/preseason/transfers/{kind}/{season}.json")
            path = raw_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            payloads[(kind, season)] = payload
            digest = hashlib.sha256(payload).hexdigest()
            manifest_items.append(
                {
                    "kind": kind,
                    "path": relative.as_posix(),
                    "season": season,
                    "sha256": digest,
                    "record_count": len(rows),
                    "provenance": {"content_sha256": digest},
                }
            )
    manifest_path.write_text(
        json.dumps({"files": manifest_items}, indent=2) + "\n", encoding="utf-8"
    )
    return raw_root, manifest_path, payloads


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_source_verification_fails_closed_for_missing_or_modified_payloads(
    tmp_path: Path,
) -> None:
    raw_root, manifest_path, _ = _source_fixture(tmp_path / "missing")
    missing_path = raw_root / "cfbd/preseason/transfers/portal/2022.json"
    missing_path.unlink()
    with pytest.raises(
        ValueError, match="historical source verification failed closed"
    ):
        replay.verify_historical_sources(raw_root, manifest_path=manifest_path)

    raw_root, manifest_path, _ = _source_fixture(tmp_path / "modified")
    changed_path = raw_root / "cfbd/preseason/transfers/usage/2021.json"
    changed_path.write_text("[] ", encoding="utf-8")
    with pytest.raises(
        ValueError, match="historical source verification failed closed"
    ):
        replay.verify_historical_sources(raw_root, manifest_path=manifest_path)


def test_historical_replay_is_deterministic_and_inventories_ambiguous_and_zero_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root, manifest_path, _ = _source_fixture(tmp_path / "inputs")
    source_summary, payloads = replay.verify_historical_sources(
        raw_root, manifest_path=manifest_path
    )
    records, _, _, _, _, _ = replay.load_verified_transfer_data(
        source_summary, payloads
    )

    player_rows = [
        {
            "portal_index": "0",
            "season": "2022",
            "player_name": "Taylor Powell",
            "origin": "Troy",
            "destination": "Eastern Michigan",
            "position": "WR",
            "transfer_date": "2022-07-01",
            "portal_player_id": "",
            "in_model_relevant_population": "True",
            "usage_join_status": "ambiguous_usage_join",
        },
        {
            "portal_index": "1",
            "season": "2022",
            "player_name": "Zero Player",
            "origin": "Alpha",
            "destination": "Beta",
            "position": "TE",
            "transfer_date": "2022-07-02",
            "portal_player_id": "zero-id",
            "in_model_relevant_population": "True",
            "usage_join_status": "no_usage_record",
        },
    ]
    audit_path = tmp_path / "player_join_records.csv"
    _write_csv(audit_path, player_rows, list(player_rows[0]))
    assert replay._verify_historical_portal_audit_bridge(records, audit_path) == 2
    position_mismatch = [dict(row) for row in player_rows]
    position_mismatch[0]["position"] = "DB"
    mismatch_path = tmp_path / "position_mismatch.csv"
    _write_csv(mismatch_path, position_mismatch, list(position_mismatch[0]))
    with pytest.raises(
        ValueError, match="does not identify the verified portal record"
    ):
        replay._verify_historical_portal_audit_bridge(records, mismatch_path)
    baseline_rows = [
        {
            "season": "2022",
            "subdivision": "fbs",
            "team_id": "10",
            "team_name": "Eastern Michigan",
            "transfer_in_prior_usage_sum": "0.519",
            "transfer_in_prior_defensive_impact_db_sum": "0.0",
            "transfer_in_prior_defensive_impact_db_available": "0.0",
        },
        {
            "season": "2022",
            "subdivision": "fbs",
            "team_id": "20",
            "team_name": "Beta",
            "transfer_in_prior_usage_sum": "",
            "transfer_in_prior_defensive_impact_db_sum": "0.0",
            "transfer_in_prior_defensive_impact_db_available": "0.0",
        },
    ]
    panel_path = tmp_path / "historical_panel.csv"
    panel_fields = list(baseline_rows[0])
    _write_csv(panel_path, baseline_rows, panel_fields)
    team_rows = [
        {
            "season": 2022,
            "subdivision": "fbs",
            "team_id": "10",
            "team_name": "Eastern Michigan",
            "returning_pct_ppa": None,
        },
        {
            "season": 2022,
            "subdivision": "fbs",
            "team_id": "20",
            "team_name": "Beta",
            "returning_pct_ppa": None,
        },
    ]
    zero_fields = [
        "season",
        "destination_team_id",
        "destination_team",
        "portal_index",
        "player",
        "source_team",
        "destination",
        "transfer_date",
        "portal_player_id",
        "d5_resolution_category",
        "d5_feature_value",
        "usage_join_status",
        "source_provenance",
    ]
    zero_rows = [
        {
            "season": "2022",
            "destination_team_id": "20",
            "destination_team": "Beta",
            "portal_index": "1",
            "player": "Zero Player",
            "source_team": "Alpha",
            "destination": "Beta",
            "transfer_date": "2022-07-02",
            "portal_player_id": "zero-id",
            "d5_resolution_category": replay.ZERO_EVIDENCE_CATEGORY,
            "d5_feature_value": "0.0",
            "usage_join_status": "no_usage_record",
            "source_provenance": "fixture zero evidence",
        }
    ]
    evidence_bytes = tmp_path / "expected_zero.csv"
    _write_csv(evidence_bytes, zero_rows, zero_fields)
    evidence_hash = hashlib.sha256(evidence_bytes.read_bytes()).hexdigest()

    materializer_spec = importlib.util.spec_from_file_location(
        "materializer_for_replay_test",
        ROOT / "scripts/materialize_preseason_context_prior_v1_3_research_features.py",
    )
    assert materializer_spec is not None and materializer_spec.loader is not None
    materializer = importlib.util.module_from_spec(materializer_spec)
    materializer_spec.loader.exec_module(materializer)

    def fake_run(**kwargs):
        assert kwargs["source_root"] == MODEL_SOURCE_ROOT.resolve()
        assert (
            MODEL_SOURCE_ROOT
            / "data/processed/modeling/team_season_rank_distributions.csv"
        ).is_file()
        parsed_records, usage, portal_seasons, _ = kwargs["verified_transfer_data"]
        verified_zero_keys = materializer._verified_zero_usage_keys(
            parsed_records, kwargs["zero_evidence"], team_rows
        )
        features = aggregate_team_features(
            parsed_records,
            usage,
            team_rows,
            covered_seasons=portal_seasons,
            cutoff=date(2025, 8, 15),
            verified_zero_usage_keys=verified_zero_keys,
        )
        output = []
        for row in team_rows:
            key = (row["season"], row["subdivision"], row["team_id"])
            output.append(
                {
                    **row,
                    "transfer_in_prior_usage_sum": features[key][
                        "transfer_in_prior_usage_sum"
                    ],
                    "transfer_in_prior_defensive_impact_db_sum": 0.0,
                    "transfer_in_prior_defensive_impact_db_available": 0.0,
                }
            )
        return output

    materializer.run = fake_run
    monkeypatch.setattr(replay, "_load_materializer", lambda: materializer)

    kwargs = {
        "model_source_root": MODEL_SOURCE_ROOT,
        "raw_root": raw_root,
        "zero_contributors": zero_rows,
        "zero_evidence_sha256": evidence_hash,
        "zero_evidence_columns": zero_fields,
        "expected_zero_repair_team_seasons": {(2022, "20")},
        "historical_panel": panel_path,
        "historical_player_audit": audit_path,
        "source_manifest_path": manifest_path,
    }
    first_changes, first_summary, _, first_player_audit = (
        replay.replay_historical_materializer(**kwargs)
    )
    second_changes, second_summary, _, second_player_audit = (
        replay.replay_historical_materializer(**kwargs)
    )

    assert first_changes == second_changes
    assert first_summary == second_summary
    assert first_player_audit == second_player_audit
    assert first_summary["changed_team_seasons"] == 2
    assert first_summary["changed_feature_values"] == 2
    assert first_summary["changed_feature_values_by_class"] == {
        "ambiguous_usage_join_removed": 1,
        "legitimate_zero_restoration": 1,
    }
    assert first_summary["change_inventory_exactly_matches_materializer_diff"]
    east_michigan = next(
        row for row in first_changes if row["team_name"] == "Eastern Michigan"
    )
    assert east_michigan["old_value"] == "0.519"
    assert east_michigan["new_value"] == ""
    assert east_michigan["change_class"] == "ambiguous_usage_join_removed"
    responsible = json.loads(east_michigan["players_responsible"])
    assert responsible[0]["player"] == "Taylor Powell"
    assert responsible[0]["legacy_candidate_count"] == 2
    assert responsible[0]["current_resolved_candidate_count"] == 2
    assert responsible[0]["portal_position"] == "WR"
    assert first_player_audit[0]["usage_join_status"] == "ambiguous_usage_join"
    beta = next(row for row in first_changes if row["team_name"] == "Beta")
    beta_responsible = json.loads(beta["players_responsible"])
    assert (
        beta_responsible[0]["zero_evidence_category"] == replay.ZERO_EVIDENCE_CATEGORY
    )
    assert beta_responsible[0]["zero_evidence_value"] == "0.0"
    beta_provenance = json.loads(beta["source_provenance"])
    assert beta_provenance["stats_source_path"].endswith("stats/2021.json")
    assert beta_provenance["stats_source_sha256"]
