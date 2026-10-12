from pathlib import Path

from gippyrank.data.massey import (
    HISTORICAL_SUBDIVISION_OVERRIDES,
    CFBDTeam,
    build_team_matches,
    historical_subdivision,
    iter_kaggle_observations,
    read_export_observations,
)


def test_alias_matching_does_not_use_rank_heuristics() -> None:
    teams = [
        CFBDTeam(1, "App State", "fbs", ("Appalachian State",)),
        CFBDTeam(2, "Central Connecticut", "fcs", ()),
    ]
    matched, ambiguous, unmapped = build_team_matches(
        ["Appalachian St", "Central Conn", "Unknown"], teams
    )
    assert matched["Appalachian St"].classification == "fbs"
    assert matched["Central Conn"].classification == "fcs"
    assert ambiguous == {}
    assert unmapped == ["Unknown"]


def test_historical_subdivision_uses_explicit_identity_and_season_overrides() -> None:
    famu = CFBDTeam(50, "Florida A&M", "fbs")
    wku = CFBDTeam(98, "Western Kentucky", "fcs")
    tarleton = CFBDTeam(2627, "Tarleton State", "ii")
    dixie = CFBDTeam(3101, "Utah Tech", "ii")

    assert historical_subdivision(2004, famu) == "fcs"
    assert historical_subdivision(2007, wku) == "fbs"
    assert historical_subdivision(2020, tarleton) == "fcs"
    assert historical_subdivision(2020, dixie) == "fcs"
    assert historical_subdivision(2008, wku) == "fcs"
    assert set(HISTORICAL_SUBDIVISION_OVERRIDES) == {
        (2004, "50"),
        (2007, "98"),
        (2020, "2627"),
        (2020, "3101"),
    }


def test_historical_override_preserves_original_source_rank(tmp_path: Path) -> None:
    source = tmp_path / "cf2007.csv"
    source.write_text(
        "cf2007,98,WKU,CMP,Composite,20071203,120\n",
        encoding="utf-8",
    )
    observations = list(
        iter_kaggle_observations(
            source,
            {"WKU": CFBDTeam(98, "Western Kentucky", "fcs")},
            source_file="data/raw/massey_rankings/cf2007.csv",
        )
    )

    assert len(observations) == 1
    assert observations[0].subdivision == "fbs"
    assert observations[0].ordinal_rank == 120
    assert observations[0].source_file == "data/raw/massey_rankings/cf2007.csv"


def test_export_is_long_form_and_omits_missing_values(tmp_path: Path) -> None:
    export = tmp_path / "fcs2022.csv"
    export.write_text(
        "Team,Conf,W-L,&Delta;,CMP,Sort,MAS\n"
        "Example FCS, ,0-0, ,1,--,2\n",
        encoding="utf-8",
    )
    observations, _ = read_export_observations(export)
    assert observations
    assert all(item.subdivision == "fcs" for item in observations)
    assert all(item.team_source_id is None for item in observations)
    assert all(item.system_name is None for item in observations)
    assert all(item.ranking_date is None for item in observations)
    assert any(item.is_composite for item in observations)
    assert any(
        item.system_code == "MAS" and not item.is_composite for item in observations
    )
