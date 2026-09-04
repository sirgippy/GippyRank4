from pathlib import Path

from gippyrank.data.massey import CFBDTeam, build_team_matches, read_export_observations


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


def test_export_is_long_form_and_omits_missing_values() -> None:
    observations, _ = read_export_observations(
        Path("data/raw/massey_composite/fcs2022.csv")
    )
    assert observations
    assert all(item.subdivision == "fcs" for item in observations)
    assert all(item.team_source_id is None for item in observations)
    assert all(item.system_name is None for item in observations)
    assert all(item.ranking_date is None for item in observations)
    assert any(item.is_composite for item in observations)
    assert any(
        item.system_code == "MAS" and not item.is_composite for item in observations
    )
