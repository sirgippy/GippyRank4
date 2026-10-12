import csv
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pytest

from gippyrank.data.massey import (
    NORMALIZED_FIELDS,
    CFBDTeam,
    build_team_matches,
)
from gippyrank.data.massey_prebowl import (
    APPROVED_PHASE_EXCEPTIONS,
    PreBowlWindow,
    audit_target_rank_support,
    fcs_rows_match,
    import_raw_export,
    latest_cmp_date_in_window,
    normalized_rows,
    read_dated_fbs_export,
    schedule_window_from_espn,
    snapshot_date_is_accepted,
    validate_raw_provenance,
    validate_snapshot,
)
from gippyrank.modeling import build_distributions, coverage_table


def _event(
    *,
    date_text: str,
    season_type: int,
    name: str,
    note: str,
    competition_type: str = "1",
) -> dict:
    return {
        "date": date_text,
        "name": name,
        "season": {"type": season_type},
        "competitions": [
            {
                "type": {"id": competition_type},
                "notes": [{"headline": note}],
            }
        ],
    }


def test_schedule_window_excludes_fcs_playoffs_and_fcs_bowl() -> None:
    payload = {
        "events": [
            _event(
                date_text="2025-12-07T01:00:00Z",
                season_type=2,
                name="Conference championship",
                note="BIG TEN CHAMPIONSHIP",
            ),
            _event(
                date_text="2025-12-14T01:00:00Z",
                season_type=3,
                name="Bowl game",
                note="BUCKED UP LA BOWL",
            ),
            _event(
                date_text="2025-12-13T18:00:00Z",
                season_type=3,
                name="FCS playoff",
                note="FCS CHAMPIONSHIP QUARTERFINAL",
            ),
            _event(
                date_text="2025-12-13T22:00:00Z",
                season_type=3,
                name="Celebration Bowl",
                note="CELEBRATION BOWL",
            ),
        ]
    }

    window = schedule_window_from_espn(payload, 2025)

    assert window.last_fbs_conference_championship == date(2025, 12, 6)
    assert window.first_fbs_postseason_game == date(2025, 12, 13)
    assert window.first_eligible_date == date(2025, 12, 7)
    assert window.last_eligible_date == date(2025, 12, 12)


def test_schedule_window_requires_both_championship_and_bowl() -> None:
    with pytest.raises(ValueError, match="No FBS conference championship"):
        schedule_window_from_espn({"events": []}, 2025)


def test_latest_cmp_date_uses_strict_window_and_ignores_constituents(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cf2003.csv"
    source.write_text(
        "cf2003,1,Example,CMP,Composite,20031206,1\n"
        "cf2003,1,Example,SYS,System,20031214,1\n"
        "cf2003,1,Example,CMP,Composite,20031208,1\n"
        "cf2003,1,Example,CMP,Composite,20031210,1\n"
        "cf2003,1,Example,CMP,Composite,20031216,1\n",
        encoding="utf-8",
    )
    window = PreBowlWindow(2003, date(2003, 12, 6), date(2003, 12, 16))

    assert latest_cmp_date_in_window(source, window) == date(2003, 12, 10)


def test_latest_cmp_date_can_filter_fcs_rows(tmp_path: Path) -> None:
    source = tmp_path / "cf2003.csv"
    source.write_text(
        "cf2003,1,FBS Team,CMP,Composite,20031208,1\n"
        "cf2003,2,FCS Team,CMP,Composite,20031215,1\n",
        encoding="utf-8",
    )
    window = PreBowlWindow(2003, date(2003, 12, 6), date(2003, 12, 16))
    matches = {
        "FBS Team": CFBDTeam(1, "FBS Team", "fbs"),
        "FCS Team": CFBDTeam(2, "FCS Team", "fcs"),
    }

    assert latest_cmp_date_in_window(source, window, matches) == date(2003, 12, 8)


def test_latest_cmp_date_applies_historical_wku_fbs_decision(tmp_path: Path) -> None:
    source = tmp_path / "cf2007.csv"
    source.write_text(
        "cf2007,98,WKU,CMP,Composite,20071210,120\n",
        encoding="utf-8",
    )
    window = PreBowlWindow(2007, date(2007, 12, 6), date(2007, 12, 12))
    matches = {"WKU": CFBDTeam(98, "Western Kentucky", "fcs")}

    assert latest_cmp_date_in_window(source, window, matches) == date(2007, 12, 10)


def test_2020_phase_exception_requires_exact_approval_and_snapshot_date() -> None:
    window = PreBowlWindow(2020, date(2020, 12, 19), date(2020, 12, 21))
    snapshot_date = date(2020, 12, 21)
    approval = APPROVED_PHASE_EXCEPTIONS[(2020, snapshot_date)]

    assert snapshot_date_is_accepted(2020, date(2020, 12, 20), window)
    assert snapshot_date_is_accepted(2020, snapshot_date, window, approval)
    assert not snapshot_date_is_accepted(2020, snapshot_date, window)
    assert not snapshot_date_is_accepted(2020, snapshot_date, window, "approved")
    assert not snapshot_date_is_accepted(2021, snapshot_date, window, approval)


def test_target_support_audit_reports_mass_without_changing_the_pmf() -> None:
    row: dict[str, object] = {
        "season": 2007,
        "subdivision": "fbs",
        "team_id": "70",
        "team_name": "Idaho",
        "team_population": 119,
        "usable_systems": 114,
        "pmf": '[{"rank":119,"count":69,"probability":0.6052631579},'
        '{"rank":120,"count":45,"probability":0.3947368421}]',
    }
    original_pmf = row["pmf"]

    anomalies = audit_target_rank_support([row])

    assert anomalies == [
        {
            "season": 2007,
            "subdivision": "fbs",
            "team_id": "70",
            "team_name": "Idaho",
            "team_population": 119,
            "usable_systems": 114,
            "out_of_support_ranks": "[120]",
            "out_of_support_observations": 45,
            "out_of_support_probability_mass": pytest.approx(0.3947368421),
        }
    ]
    assert row["pmf"] == original_pmf


def test_history_target_loader_accepts_valid_pmf_and_rejects_bad_support(
    tmp_path: Path,
) -> None:
    script = Path(__file__).parents[1] / "scripts/study_history_context_posterior.py"
    spec = importlib.util.spec_from_file_location("history_target_loader", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    target = tmp_path / "targets.csv"
    fields = [
        "season",
        "subdivision",
        "team_id",
        "team_name",
        "team_population",
        "usable_systems",
        "rank_observations",
        "pmf",
    ]
    row = {
        "season": "2020",
        "subdivision": "fbs",
        "team_id": "1",
        "team_name": "Example",
        "team_population": "2",
        "usable_systems": "2",
        "rank_observations": "[1,2]",
        "pmf": '[{"rank":1,"count":1,"probability":0.5},'
        '{"rank":2,"count":1,"probability":0.5}]',
    }
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    loaded = module.load_targets(2020, target)
    assert loaded["1"].tolist() == [0.5, 0.5]

    row["rank_observations"] = "[1,3]"
    row["pmf"] = (
        '[{"rank":1,"count":1,"probability":0.5},'
        '{"rank":3,"count":1,"probability":0.5}]'
    )
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="outside 1..2"):
        module.load_targets(2020, target)


def test_raw_export_import_preserves_bytes_and_records_verified_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "browser-download.csv"
    raw = b"Team,Conf,W-L,&Delta;,CMP,Sort,SYS\r\nExample, ,1-0, ,1,>>,1\r\n"
    source.write_bytes(raw)

    destination = import_raw_export(
        source,
        season=2025,
        snapshot_date=date(2025, 12, 7),
        output_root=tmp_path / "raw",
        source_url="https://masseyratings.com/ranks?c=1&d=20251207&s=cf",
    )

    assert destination.read_bytes() == raw
    metadata = validate_raw_provenance(
        destination,
        season=2025,
        snapshot_date=date(2025, 12, 7),
        source_url="https://masseyratings.com/ranks?c=1&d=20251207&s=cf",
    )
    assert metadata["bytes"] == len(raw)
    assert metadata["sha256"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        import_raw_export(
            source,
            season=2025,
            snapshot_date=date(2025, 12, 7),
            output_root=tmp_path / "raw",
            source_url="https://masseyratings.com/ranks?c=1&d=20251207&s=cf",
        )


def test_raw_export_provenance_detects_changed_file(tmp_path: Path) -> None:
    source = tmp_path / "browser-download.csv"
    source.write_text("raw bytes", encoding="utf-8")
    destination = import_raw_export(
        source,
        season=2025,
        snapshot_date=date(2025, 12, 7),
        output_root=tmp_path / "raw",
        source_url="https://masseyratings.com/ranks?c=1&d=20251207&s=cf",
    )
    destination.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="Provenance mismatch"):
        validate_raw_provenance(
            destination,
            season=2025,
            snapshot_date=date(2025, 12, 7),
            source_url="https://masseyratings.com/ranks?c=1&d=20251207&s=cf",
        )


def test_dated_export_maps_ids_and_omits_missing_cells(tmp_path: Path) -> None:
    export = tmp_path / "fbs2025-20251207.csv"
    export.write_text(
        "Team,Conf,W-L,&Delta;,CMP,Sort,SYS,EMPTY\n"
        "Example FBS, ,10-1, ,1,>>,2,--\n"
        "Other FBS, ,9-2, ,2,>>,1,\n",
        encoding="utf-8",
    )
    teams = {
        "Example FBS": CFBDTeam(101, "Example FBS", "fbs"),
        "Other FBS": CFBDTeam(102, "Other FBS", "fbs"),
    }

    observations, unmapped = read_dated_fbs_export(
        export,
        teams,
        expected_season=2025,
        expected_date=date(2025, 12, 7),
        source_file="data/raw/massey_prebowl/fbs2025-20251207.csv",
    )
    rows = normalized_rows(observations)

    assert unmapped == []
    assert {row["team_id"] for row in rows} == {"101", "102"}
    assert {row["ranking_date"] for row in rows} == {"20251207"}
    assert {row["system_code"] for row in rows} == {"CMP", "SYS"}
    assert all(row["source"] == "massey_prebowl_export" for row in rows)
    assert len(rows) == 4
    assert all(row["system_code"] != "EMPTY" for row in rows)
    validate_snapshot(rows, season=2025, snapshot_date=date(2025, 12, 7))


def test_dated_export_preserves_omitted_trailing_cells_as_missing(
    tmp_path: Path,
) -> None:
    export = tmp_path / "fbs2025-20251207.csv"
    export.write_text(
        "Team,Conf,W-L,&Delta;,CMP,Sort,SYS,LATE\nExample FBS, ,10-1, ,1,>>,2\n",
        encoding="utf-8",
    )
    teams = {"Example FBS": CFBDTeam(101, "Example FBS", "fbs")}

    observations, unmapped = read_dated_fbs_export(
        export,
        teams,
        expected_season=2025,
        expected_date=date(2025, 12, 7),
        source_file="data/raw/massey_prebowl/fbs2025-20251207.csv",
    )

    assert unmapped == []
    assert {row.system_code for row in observations} == {"CMP", "SYS"}


def test_massey_alias_maps_to_stable_canonical_team_id() -> None:
    matches, ambiguous, unmapped = build_team_matches(
        ["Appalachian St"], [CFBDTeam(333, "App State", "fbs")]
    )

    assert matches["Appalachian St"].id == 333
    assert not ambiguous
    assert not unmapped


def test_dated_export_rejects_filename_date_mismatch(tmp_path: Path) -> None:
    export = tmp_path / "fbs2025-20251208.csv"
    export.write_text(
        "Team,Conf,W-L,&Delta;,CMP,Sort,SYS\nExample, ,1-0, ,1,>>,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match expected snapshot"):
        read_dated_fbs_export(
            export,
            {"Example": CFBDTeam(1, "Example", "fbs")},
            expected_season=2025,
            expected_date=date(2025, 12, 7),
            source_file="data/raw/massey_prebowl/fbs2025-20251208.csv",
        )


def test_dated_fbs_export_rejects_mapped_fcs_team(tmp_path: Path) -> None:
    export = tmp_path / "fbs2025-20251207.csv"
    export.write_text(
        "Team,Conf,W-L,&Delta;,CMP,Sort,SYS\nExample FCS, ,1-0, ,1,>>,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-FBS teams"):
        read_dated_fbs_export(
            export,
            {"Example FCS": CFBDTeam(101, "Example FCS", "fcs")},
            expected_season=2025,
            expected_date=date(2025, 12, 7),
            source_file="data/raw/massey_prebowl/fbs2025-20251207.csv",
        )


def _snapshot_rows() -> list[dict[str, str]]:
    rows = []
    for team_id, team_name, cmp_rank, sys_rank in (
        ("1", "Example A", 1, 2),
        ("2", "Example B", 2, 1),
    ):
        for system_code, rank in (("CMP", cmp_rank), ("SYS", sys_rank)):
            rows.append(
                {
                    "season": "2025",
                    "subdivision": "fbs",
                    "team_id": team_id,
                    "team_source_id": "",
                    "team_name": team_name,
                    "system_code": system_code,
                    "system_name": "",
                    "ranking_date": "20251207",
                    "ordinal_rank": str(rank),
                    "is_composite": str(system_code == "CMP"),
                    "source": "massey_prebowl_export",
                    "source_file": "data/raw/massey_prebowl/fbs2025-20251207.csv",
                }
            )
    return rows


def test_snapshot_validation_detects_duplicates_and_cmp_conflation() -> None:
    rows = _snapshot_rows()
    with pytest.raises(ValueError, match="Duplicate team/system"):
        validate_snapshot(
            rows + [rows[0]], season=2025, snapshot_date=date(2025, 12, 7)
        )

    conflated = _snapshot_rows()
    conflated[0]["is_composite"] = "False"
    with pytest.raises(ValueError, match="conflated"):
        validate_snapshot(conflated, season=2025, snapshot_date=date(2025, 12, 7))


def test_snapshot_pmf_keeps_constituent_support_and_normalizes() -> None:
    rows = _snapshot_rows()
    depth = coverage_table(rows)
    distributions, by_team = build_distributions(rows, depth, threshold=1.0)

    assert len(distributions) == 2
    assert by_team[(2025, "fbs", "1")] == [("SYS", 2)]
    for item in distributions:
        pmf = json.loads(item["pmf"])
        assert item["rank_observations"] in {"[1]", "[2]"}
        assert sum(point["probability"] for point in pmf) == pytest.approx(1.0)
        assert sum(point["count"] for point in pmf) == 1


def test_fcs_rows_remain_ordered_and_field_identical() -> None:
    source = [
        {field: "" for field in NORMALIZED_FIELDS},
        {field: "" for field in NORMALIZED_FIELDS},
    ]
    source[0].update({"season": "2003", "subdivision": "fcs", "team_id": "1"})
    source[1].update({"season": "2004", "subdivision": "fcs", "team_id": "2"})
    combined = _snapshot_rows() + [row.copy() for row in source]

    assert fcs_rows_match(source, combined)
    combined[-1]["ordinal_rank"] = "999"
    assert not fcs_rows_match(source, combined)
