import json
from pathlib import Path

from gippyrank.data.cfbd_games import read_cfbd_game_rows


def test_raw_game_reader_applies_rank_corpus_cohort_and_preserves_source(
    tmp_path: Path,
) -> None:
    game_root = tmp_path / "games"
    game_root.mkdir()
    source = game_root / "2007.json"
    payload = [
        {
            "id": 1,
            "season": 2007,
            "homeId": 98,
            "homeClassification": "fcs",
            "awayId": 2,
            "awayClassification": "fbs",
        }
    ]
    original_bytes = json.dumps(payload).encode()
    source.write_bytes(original_bytes)

    games, inputs = read_cfbd_game_rows(game_root)

    assert games[0]["homeClassification"] == "fbs"
    assert games[0]["awayClassification"] == "fbs"
    assert inputs == [source]
    assert source.read_bytes() == original_bytes
