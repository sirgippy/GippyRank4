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
    _raw_alias_target,
    _snapshot_filename,
    classify_db_position,
    derive_preseason_transfer_features,
    load_snapshot_manifest,
    merge_preseason_transfer_features,
    summarize_db_impact_coverage,
    validate_research_parity,
    write_immutable_snapshot,
    write_snapshot_manifest,
)
from gippyrank.transfer_audit import read_team_aliases


def _payload_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _team_rows() -> list[dict[str, str]]:
    return [
        {"season": "2021", "subdivision": "fbs", "team_id": "a", "team_name": "Alpha"},
        {"season": "2021", "subdivision": "fbs", "team_id": "b", "team_name": "Beta"},
        {"season": "2022", "subdivision": "fbs", "team_id": "a", "team_name": "Alpha"},
        {"season": "2022", "subdivision": "fbs", "team_id": "b", "team_name": "Beta"},
    ]


def _fixture_manifest(tmp_path: Path, *, bob_roster_first_name: str = "Bob") -> Path:
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
        SnapshotSpec(
            2022, "roster", 2021, "/roster", {"year": 2021, "classification": "fcs"}
        ),
        SnapshotSpec(
            2022,
            "games_players",
            2021,
            "/games/players",
            {"year": 2021, "week": 1, "classification": "fcs", "seasonType": "both"},
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
                "firstName": bob_roster_first_name,
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
        [],
        [],
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


def _add_supplemental_dii_player(manifest_path: Path) -> None:
    raw_root = manifest_path.parent
    specs = [
        SnapshotSpec(
            2022, "roster", 2021, "/roster", {"year": 2021, "classification": "ii"}
        ),
        SnapshotSpec(
            2022,
            "games_players",
            2021,
            "/games/players",
            {"year": 2021, "week": 2, "classification": "ii", "seasonType": "both"},
        ),
    ]
    payloads = [
        [
            {
                "id": "dii-db-1",
                "firstName": "Supplemental",
                "lastName": "Defender",
                "team": "Delta",
                "position": "CB",
            }
        ],
        [
            {
                "id": 2,
                "teams": [
                    {
                        "team": "Delta",
                        "categories": [
                            {
                                "name": "defensive",
                                "types": [
                                    {
                                        "name": "TOT",
                                        "athletes": [
                                            {
                                                "id": "dii-db-1",
                                                "name": "Supplemental Defender",
                                                "stat": "1000",
                                            }
                                        ],
                                    },
                                    {
                                        "name": "PD",
                                        "athletes": [
                                            {
                                                "id": "dii-db-1",
                                                "name": "Supplemental Defender",
                                                "stat": "100",
                                            }
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
                                                "id": "dii-db-1",
                                                "name": "Supplemental Defender",
                                                "stat": "20",
                                            }
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
    specs.extend(
        (
            SnapshotSpec(
                2022, "roster", 2021, "/roster", {"year": 2021, "team": "Delta"}
            ),
            SnapshotSpec(
                2022,
                "games_players",
                2021,
                "/games/players",
                {"year": 2021, "team": "Delta", "seasonType": "both"},
            ),
        )
    )
    payloads.extend((payloads[0], payloads[1]))
    added = [
        write_immutable_snapshot(
            raw_root=raw_root,
            spec=spec,
            content=_payload_bytes(payload),
            retrieval_timestamp="2022-08-02T12:00:00+00:00",
        )
        for spec, payload in zip(specs, payloads, strict=True)
    ]
    existing = load_snapshot_manifest(manifest_path, verify_hashes=True)
    write_snapshot_manifest(
        manifest_path,
        [*existing.snapshots, *added],
        raw_root=raw_root,
    )


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


def test_supplemental_acquisition_preserves_repaired_fbs_fcs_transfer_join(
    tmp_path: Path,
) -> None:
    baseline_manifest = _fixture_manifest(
        tmp_path / "baseline", bob_roster_first_name="Robert"
    )
    supplemental_manifest = _fixture_manifest(
        tmp_path / "supplemental", bob_roster_first_name="Robert"
    )
    _add_supplemental_dii_player(supplemental_manifest)

    baseline = derive_preseason_transfer_features(
        baseline_manifest,
        _team_rows(),
        team_aliases={(2022, "A State"): "Alpha"},
    )
    expanded = derive_preseason_transfer_features(
        supplemental_manifest,
        _team_rows(),
        team_aliases={(2022, "A State"): "Alpha"},
    )
    baseline_bob = next(
        row for row in baseline["player_audit"] if row["player_name"] == "Bob Defender"
    )
    expanded_bob = next(
        row for row in expanded["player_audit"] if row["player_name"] == "Bob Defender"
    )
    reference = expanded["quality_report"]["seasons"][0]["defensive_impact_reference"]

    assert expanded_bob["identity_status"] == "resolved"
    assert expanded_bob["identity_join_method"] == "stable_game_player_id_source_team"
    assert expanded_bob["impact_status"] == "resolved"
    assert expanded_bob["impact_join_method"] == "stable_player_id_source_team"
    assert expanded_bob["prior_defensive_impact"] == pytest.approx(
        baseline_bob["prior_defensive_impact"]
    )
    assert reference["classifications"] == ["fbs", "fcs"]
    assert reference["roster_snapshot_count"] == 2
    assert reference["games_players_snapshot_count"] == 2


def test_derivation_fails_if_the_fbs_fcs_reference_corpus_is_incomplete(
    tmp_path: Path,
) -> None:
    manifest = _fixture_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["snapshots"] = [
        snapshot
        for snapshot in payload["snapshots"]
        if snapshot.get("query_parameters", {}).get("classification") != "fcs"
    ]
    payload["required_requests"] = [
        request
        for request in payload["required_requests"]
        if request.get("query_parameters", {}).get("classification") != "fcs"
    ]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ManifestValidationError, match="missing full-division roster snapshots for fcs"
    ):
        derive_preseason_transfer_features(manifest, _team_rows())


def test_team_filtered_snapshot_filenames_are_unique_without_week_numbers() -> None:
    hyphenated = SnapshotSpec(
        2026,
        "games_players",
        2025,
        "/games/players",
        {"year": 2025, "team": "Nebraska-Kearney", "seasonType": "both"},
    )
    spaced = SnapshotSpec(
        2026,
        "games_players",
        2025,
        "/games/players",
        {"year": 2025, "team": "Nebraska Kearney", "seasonType": "both"},
    )

    hyphenated_name = _snapshot_filename(hyphenated, version="teams")
    spaced_name = _snapshot_filename(spaced, version="teams")

    assert hyphenated_name != spaced_name
    assert "team-nebraska-kearney-" in hyphenated_name
    assert "week-00" not in hyphenated_name


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
        team_aliases={(2022, "A State"): "Alpha"},
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
    assert beta["incoming_db_count"] == 2
    assert beta["observed_db_impact_count"] == 1
    assert beta["missing_db_impact_count"] == 1
    assert beta["db_impact_coverage_fraction"] == pytest.approx(0.5)
    assert beta["db_impact_coverage_status"] == "partial"
    assert beta["db_impact_source_season"] == 2021
    bob = next(
        row for row in result["player_audit"] if row["player_name"] == "Bob Defender"
    )
    assert beta["observed_db_impact_sum"] == pytest.approx(
        float(bob["prior_defensive_impact"])
    )
    assert bob["prior_player_id"] == "db-1"
    assert bob["impact_join_method"] == "stable_player_id_source_team"
    assert bob["source_season"] == 2021
    assert bob["portal_snapshot_sha256"]
    assert bob["roster_snapshot_sha256"]
    assert bob["games_players_snapshot_sha256"]
    assert beta["observed_db_impact_sum"] == pytest.approx(
        result["provenance"]["2022|b|observed_db_impact_sum"]["value"]
    )
    assert alpha["transfer_in_prior_defensive_impact_db_sum"] == 0.0
    assert alpha["transfer_in_prior_defensive_impact_db_available"] == 1
    assert alpha["incoming_db_count"] == 0
    assert alpha["observed_db_impact_count"] == 0
    assert alpha["observed_db_impact_sum"] == 0.0
    assert alpha["db_impact_coverage_status"] == "no_incoming_db_transfers"
    assert alpha["db_impact_coverage_fraction"] is None
    provenance_key = "2022|b|transfer_in_prior_defensive_impact_db_sum"
    assert (
        result["provenance"][provenance_key]["contributors"][0]["player_name"]
        == "Bob Defender"
    )
    assert result["quality_report"]["seasons"][0]["identity_alias_matches"] == 1
    assert result["offensive_player_audit"]
    assert "usage_candidate_player_ids" in result["offensive_player_audit"][0]

    wrong_season = derive_preseason_transfer_features(
        manifest,
        _team_rows(),
        team_aliases={(2021, "A State"): "Alpha"},
        player_aliases=PlayerAliasResolver({"Alex Player": "A Player"}),
    )
    assert wrong_season["identity_mapping"][0]["origin_team_name"] != "Alpha"


def test_2026_team_aliases_are_evidence_scoped_and_do_not_merge_similar_schools() -> (
    None
):
    aliases = read_team_aliases(
        Path(__file__).resolve().parents[1]
        / "data/reference/preseason_team_aliases.csv"
    )

    assert aliases[(2026, "Albany")] == "UAlbany"
    assert aliases[(2026, "LIU Post")] == "Long Island University"
    assert aliases[(2026, "Southeastern Louisiana")] == "SE Louisiana"
    assert aliases[(2026, "Saint Francis (PA)")] == "Saint Francis"
    assert (2026, "Albany State") not in aliases
    assert (2026, "Saint Francis (IN)") not in aliases
    assert (2025, "Albany") not in aliases
    assert _raw_alias_target(aliases, 2026, "Saint Francis (PA)") == "Saint Francis"
    assert _raw_alias_target(aliases, 2025, "Saint Francis (PA)") is None


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
            "incoming_db_count": 4,
            "observed_db_impact_count": 3,
            "observed_db_impact_sum": 2.0,
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
    assert "observed_db_impact_sum" not in merged[0]
    assert "incoming_db_count" not in merged[0]


def test_db_coverage_separates_partial_zero_unavailable_and_natural_zero() -> None:
    partial = summarize_db_impact_coverage(
        [
            {
                "impact_status": "resolved",
                "prior_defensive_impact": 0.0,
            },
            {
                "impact_status": "identity_resolution_failure",
                "prior_defensive_impact": None,
            },
        ],
        source_season=2025,
    )
    assert partial["incoming_db_count"] == 2
    assert partial["observed_db_impact_count"] == 1
    assert partial["observed_db_impact_sum"] == 0.0
    assert partial["missing_db_impact_count"] == 1
    assert partial["db_impact_coverage_fraction"] == 0.5
    assert partial["db_impact_coverage_status"] == "partial"

    none_observed = summarize_db_impact_coverage(
        [{"impact_status": "ambiguous", "prior_defensive_impact": None}],
        source_season=2025,
    )
    assert none_observed["observed_db_impact_sum"] == 0.0
    assert none_observed["db_impact_coverage_status"] == "no_observed_db_impact"

    natural_zero = summarize_db_impact_coverage([], source_season=2025)
    assert natural_zero["incoming_db_count"] == 0
    assert natural_zero["observed_db_impact_count"] == 0
    assert natural_zero["observed_db_impact_sum"] == 0.0
    assert natural_zero["db_impact_coverage_status"] == "no_incoming_db_transfers"
    assert natural_zero["db_impact_coverage_fraction"] is None


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
