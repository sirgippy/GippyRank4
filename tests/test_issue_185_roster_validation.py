import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "data/research/offensive_line_target_pool_validation_issue_185"
RESULTS = RESEARCH / "results"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_south_carolina_2019_official_pool_includes_omitted_ol_players() -> None:
    expected = {"Will Rogers", "M.J. Webb"}
    official_names = {
        row["official_player_name"]
        for row in read_rows(RESEARCH / "official_ol_rosters.csv")
        if row["sample_id"] == "S25"
    }
    comparison = {
        row["official_player_name"]: row
        for row in read_rows(RESULTS / "player_comparison.csv")
        if row["sample_id"] == "S25"
    }

    assert expected <= official_names
    for name in expected:
        row = comparison[name]
        assert row["comparison_status"] == "official_only"
        assert row["cfbd_roster_presence"] == "present_at_another_position"
        assert row["cfbd_roster_position_original"] == "DL"


def test_iowa_state_2009_official_pool_does_not_inherit_cfbd_candidates() -> None:
    official_names = {
        row["official_player_name"]
        for row in read_rows(RESEARCH / "official_ol_rosters.csv")
        if row["sample_id"] == "S01"
    }
    comparison = {
        (row["cfbd_player_name"], row["official_player_name"]): row
        for row in read_rows(RESULTS / "player_comparison.csv")
        if row["sample_id"] == "S01"
    }

    assert "Mike Knapp" in official_names
    assert "Carter Bykowski" not in official_names
    assert comparison[("", "Mike Knapp")]["cfbd_roster_presence"] == "absent_from_roster"
    assert comparison[("Carter Bykowski", "")]["comparison_status"] == "cfbd_only"
