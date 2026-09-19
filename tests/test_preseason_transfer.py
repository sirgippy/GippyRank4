import csv
import hashlib
import json
from pathlib import Path

import pytest

from gippyrank.preseason_transfer import (
    MODEL_FEATURE_COLUMNS,
    ManifestValidationError,
    PlayerAliasResolver,
    SnapshotSpec,
    classify_db_position,
    derive_preseason_transfer_features,
    load_snapshot_manifest,
    merge_preseason_transfer_features,
    validate_research_parity,
    write_immutable_snapshot,
    write_snapshot_manifest,
)


def _payload_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _team_rows() -> list[dict[str, str]]:
    return [
        {"season": "2021", "subdivision": "fbs", "team_id": "a", "team_name": "Alpha"},
        {"season": "2021", "subdivision": "fbs", "team_id": "b", "team_name": "Beta"},
        {"season": "2022", "subdivision": "fbs", "team_id": "a", "team_name": "Alpha"},
        {"season": "2022", "subdivision": "fbs", "team_id": "b", "team_name": "Beta"},
    ]


def _fixture_manifest(tmp_path: Path) -> Path:
    raw_root = tmp_path / "raw"
    manifest_path = raw_root / "manifest.json"
    specs = [
        SnapshotSpec(2022, "portal", 2022, "/player/portal", {"year": 2022}),
        SnapshotSpec(2022, "usage", 2021, "/player/usage", {"year": 2021}),
        SnapshotSpec(2022, "stats", 2021, "/stats/player/season", {"year": 2021}),
        SnapshotSpec(
            2022, "roster", 2021, "/roster", {"year": 2021, "classification": "fbs"}
        ),
        SnapshotSpec(
            2022,
            "games_players",
            2021,
            "/games/players",
            {"year": 2021, "week": 1, "classification": "fbs", "seasonType": "both"},
        ),
    ]
    payloads = [
        [
            {
                "season": 2022,
                "firstName": "Alex",
                "lastName": "Player",
                "origin": "A State",
                "destination": "Beta",
                "position": "QB",
                "transferDate": "2022-08-01",
            },
            {
                "season": 2022,
                "firstName": "Bob",
                "lastName": "Defender",
                "origin": "A State",
                "destination": "Beta",
                "position": "CB",
                "transferDate": "2022-08-01",
            },
            {
                "season": 2022,
                "firstName": "Cara",
                "lastName": "Defender",
                "origin": "A State",
                "destination": "Beta",
                "position": "CB",
                "transferDate": "2022-08-01",
            },
            {
                "season": 2022,
                "firstName": "Chris",
                "lastName": "Receiver",
                "origin": "A State",
                "destination": "Beta",
                "position": "WR",
                "transferDate": "2022-08-01",
            },
        ],
        [
            {
                "season": 2021,
                "id": "off-1",
                "name": "A Player",
                "team": "Alpha",
                "position": "QB",
                "usage": {"overall": 0.5},
            }
        ],
        [
            {
                "season": 2021,
                "playerId": "off-1",
                "player": "A Player",
                "team": "Alpha",
                "position": "QB",
                "category": "passing",
                "statType": "YDS",
                "stat": "100",
            },
            {
                "season": 2021,
                "playerId": "off-2",
                "player": "Chris Receiver",
                "team": "Alpha",
                "position": "WR",
                "category": "receiving",
                "statType": "REC",
                "stat": "3",
            },
        ],
        [
            {
                "id": "db-1",
                "firstName": "Bob",
                "lastName": "Defender",
                "team": "Alpha",
                "position": "CB",
            },
            {
                "id": "db-2",
                "firstName": "Other",
                "lastName": "Defender",
                "team": "Alpha",
                "position": "CB",
            },
        ],
        [
            {
                "id": 1,
                "teams": [
                    {
                        "team": "Alpha",
                        "categories": [
                            {
                                "name": "defensive",
                                "types": [
                                    {
                                        "name": "TOT",
                                        "athletes": [
                                            {
                                                "id": "db-1",
                                                "name": "Bob Defender",
                                                "stat": "10",
                                            },
                                            {
                                                "id": "db-2",
                                                "name": "Other Defender",
                                                "stat": "5",
                                            },
                                        ],
                                    },
                                    {
                                        "name": "PD",
                                        "athletes": [
                                            {
                                                "id": "db-1",
                                                "name": "Bob Defender",
                                                "stat": "2",
                                            },
                                            {
                                                "id": "db-2",
                                                "name": "Other Defender",
                                                "stat": "1",
                                            },
                                        ],
                                    },
                                ],
                            },
                            {
                                "name": "interceptions",
                                "types": [
                                    {
                                        "name": "INT",
                                        "athletes": [
                                            {
                                                "id": "db-1",
                                                "name": "Bob Defender",
                                                "stat": "1",
                                            },
                                            {
                                                "id": "db-2",
                                                "name": "Other Defender",
                                                "stat": "0",
                                            },
                                        ],
                                    }
                                ],
                            },
                        ],
                    }
                ],
            }
        ],
    ]
    records = [
        write_immutable_snapshot(
            raw_root=raw_root,
            spec=spec,
            content=_payload_bytes(payload),
            retrieval_timestamp="2022-08-01T12:00:00+00:00",
        )
        for spec, payload in zip(specs, payloads)
    ]
    write_snapshot_manifest(
        manifest_path,
        records,
        raw_root=raw_root,
        required_specs=specs,
    )
    return manifest_path


def test_snapshot_hash_validation_and_overwrite_protection(tmp_path: Path) -> None:
    spec = SnapshotSpec(2022, "portal", 2022, "/player/portal", {"year": 2022})
    content = _payload_bytes([{"season": 2022}])
    record = write_immutable_snapshot(
        raw_root=tmp_path,
        spec=spec,
        content=content,
        retrieval_timestamp="2022-08-14T12:00:00+00:00",
    )
    assert record.sha256 == hashlib.sha256(content).hexdigest()
    assert record.captured_on_or_before_cutoff is True
    with pytest.raises(FileExistsError):
        write_immutable_snapshot(
            raw_root=tmp_path,
            spec=spec,
            content=content,
            retrieval_timestamp="2022-08-14T12:00:00+00:00",
        )
    manifest_path = tmp_path / "manifest.json"
    write_snapshot_manifest(manifest_path, [record], raw_root=tmp_path)
    raw_path = tmp_path / record.path
    raw_path.write_bytes(_payload_bytes([{"season": 2022, "changed": True}]))
    with pytest.raises(ManifestValidationError, match="hash mismatch"):
        load_snapshot_manifest(manifest_path, verify_hashes=True)


def test_manifest_rejects_a_missing_required_source(tmp_path: Path) -> None:
    manifest = _fixture_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["snapshots"] = [
        snapshot
        for snapshot in payload["snapshots"]
        if snapshot["source"] != "games_players"
    ]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="games_players"):
        load_snapshot_manifest(manifest)


def test_derive_features_is_cutoff_safe_and_fail_closed(tmp_path: Path) -> None:
    manifest = _fixture_manifest(tmp_path)
    result = derive_preseason_transfer_features(
        manifest,
        _team_rows(),
        team_aliases={"A State": "Alpha"},
        player_aliases=PlayerAliasResolver({"Alex Player": "A Player"}),
    )
    by_team = {row["team_id"]: row for row in result["features"]}
    beta = by_team["b"]
    alpha = by_team["a"]
    assert beta["transfer_in_prior_usage_sum"] == pytest.approx(0.5)
    assert beta["audit_offensive_unresolved_applicable"] == 1
    assert beta["audit_offensive_undetermined"] == 0
    assert beta["audit_incoming_db_transfers"] == 2
    assert beta["audit_resolved_db_transfers"] == 1
    assert beta["audit_unresolved_db_transfers"] == 1
    assert beta["transfer_in_prior_defensive_impact_db_sum"] == 0.0
    assert beta["transfer_in_prior_defensive_impact_db_available"] == 0
    assert alpha["transfer_in_prior_defensive_impact_db_sum"] == 0.0
    assert alpha["transfer_in_prior_defensive_impact_db_available"] == 1
    provenance_key = "2022|b|transfer_in_prior_defensive_impact_db_sum"
    assert (
        result["provenance"][provenance_key]["contributors"][0]["player_name"]
        == "Bob Defender"
    )
    assert result["quality_report"]["seasons"][0]["identity_alias_matches"] == 1


def test_frozen_db_position_taxonomy_does_not_infer_unknowns() -> None:
    assert {
        label: classify_db_position(label)
        for label in ("CB", "DB", "S", "FS", "SS", "NB")
    } == {
        "CB": "db",
        "DB": "db",
        "S": "db",
        "FS": "db",
        "SS": "db",
        "NB": "db",
    }
    assert classify_db_position("ATH") is None
    assert classify_db_position("HYBRID") is None


def test_research_parity_and_attach_hook() -> None:
    rows = [
        {
            "season": 2022,
            "subdivision": "fbs",
            "team_id": "1",
            "transfer_in_prior_usage_sum": 0.5,
            "transfer_in_prior_defensive_impact_db_sum": -0.25,
            "transfer_in_prior_defensive_impact_db_available": 1,
        }
    ]
    expected = [
        {
            "season": 2022,
            "subdivision": "fbs",
            "team_id": "1",
            "transfer_in_prior_usage_sum": 0.5,
            "transfer_in_prior_defensive_impact_db_sum": -0.25,
            "transfer_in_prior_defensive_impact_db_available": 1,
        }
    ]
    comparisons = validate_research_parity(rows, expected)
    assert len(comparisons) == len(MODEL_FEATURE_COLUMNS)
    merged = merge_preseason_transfer_features(
        [{"season": 2022, "subdivision": "fbs", "team_id": "1", "team_name": "One"}],
        rows,
    )
    assert merged[0]["transfer_in_prior_usage_sum"] == 0.5


def test_production_derivation_matches_checked_in_research_fixture(
    tmp_path: Path,
) -> None:
    fixture_root = (
        Path(__file__).parent / "fixtures" / "preseason_transfer_research_parity"
    )
    case = json.loads((fixture_root / "case.json").read_text(encoding="utf-8"))
    expected = list(
        csv.DictReader(
            (fixture_root / "expected_features.csv").open(newline="", encoding="utf-8")
        )
    )
    raw_root = tmp_path / "raw"
    manifest = raw_root / "manifest.json"
    specs = []
    records = []
    for snapshot in case["snapshots"]:
        spec = SnapshotSpec(
            target_season=case["target_season"],
            source=snapshot["source"],
            source_season=snapshot["source_season"],
            endpoint=snapshot["endpoint"],
            query_parameters=snapshot["query_parameters"],
        )
        specs.append(spec)
        records.append(
            write_immutable_snapshot(
                raw_root=raw_root,
                spec=spec,
                content=_payload_bytes(snapshot["payload"]),
                retrieval_timestamp=case["retrieval_timestamp"],
                filename=snapshot["filename"],
            )
        )
    write_snapshot_manifest(
        manifest,
        records,
        raw_root=raw_root,
        required_specs=specs,
    )

    result = derive_preseason_transfer_features(
        manifest,
        case["team_rows"],
        required_seasons=[case["target_season"]],
    )
    comparisons = validate_research_parity(result["features"], expected)

    assert len(comparisons) == len(expected) * len(MODEL_FEATURE_COLUMNS)
    by_team = {row["team_id"]: row for row in result["features"]}
    assert by_team["ucf"]["transfer_in_prior_usage_sum"] == pytest.approx(0.053)
    assert by_team["ksu"]["transfer_in_prior_defensive_impact_db_available"] == 1
    assert by_team["syr"]["transfer_in_prior_defensive_impact_db_available"] == 0
    assert by_team["gamma"]["transfer_in_prior_defensive_impact_db_available"] == 1
    assert any(
        row["origin"] == "Prairie View A&M" and row["identity_status"] == "resolved"
        for row in result["player_audit"]
    )


def test_late_refresh_preserves_on_time_canonical(tmp_path: Path) -> None:
    manifest = _fixture_manifest(tmp_path)
    raw_root = manifest.parent
    spec = SnapshotSpec(2022, "portal", 2022, "/player/portal", {"year": 2022})
    late = write_immutable_snapshot(
        raw_root=raw_root,
        spec=spec,
        content=_payload_bytes([{"season": 2022, "late": True}]),
        retrieval_timestamp="2022-08-16T00:00:00+00:00",
        version="v2",
    )
    write_snapshot_manifest(manifest, [late], raw_root=raw_root)

    loaded = load_snapshot_manifest(manifest, verify_hashes=True)
    portal_records = [
        record
        for record in loaded.snapshots
        if record.target_season == 2022 and record.source == "portal"
    ]
    assert len(portal_records) == 2
    assert sum(record.canonical for record in portal_records) == 1
    assert next(
        record for record in portal_records if record.canonical
    ).captured_on_or_before_cutoff
    assert (
        next(record for record in portal_records if not record.canonical).path
        == late.path
    )
    assert (
        len([record for record in loaded.for_target(2022) if record.source == "portal"])
        == 1
    )


def test_late_only_snapshot_does_not_satisfy_production_request(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    manifest = raw_root / "manifest.json"
    spec = SnapshotSpec(2022, "portal", 2022, "/player/portal", {"year": 2022})
    late = write_immutable_snapshot(
        raw_root=raw_root,
        spec=spec,
        content=_payload_bytes([{"season": 2022, "late": True}]),
        retrieval_timestamp="2022-08-16T00:00:00+00:00",
    )
    write_snapshot_manifest(
        manifest,
        [late],
        raw_root=raw_root,
    )

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["snapshots"][0]["canonical"] is False
    assert (raw_root / late.path).exists()
    with pytest.raises(ManifestValidationError, match="missing required snapshots"):
        load_snapshot_manifest(manifest, verify_hashes=True)


def test_late_manifest_snapshot_is_rejected(tmp_path: Path) -> None:
    manifest = _fixture_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for snapshot in payload["snapshots"]:
        snapshot["retrieval_timestamp"] = "2022-08-16T00:00:00+00:00"
        snapshot["retrieved_at"] = snapshot["retrieval_timestamp"]
        snapshot["captured_on_or_before_cutoff"] = False
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="captured after"):
        derive_preseason_transfer_features(manifest, _team_rows())
