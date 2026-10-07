from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from gippyrank.research.data_paths import cfbd_roster_raw_dir, research_data_dir

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY_ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_dataset = _load_script_module(
    "build_offensive_line_shared_roster_continuity",
    "scripts/build_offensive_line_shared_roster_continuity.py",
).build_dataset
acquire_roster_history = _load_script_module(
    "fetch_offensive_line_roster_history",
    "scripts/fetch_offensive_line_roster_history.py",
).acquire_roster_history


class _Response:
    status_code = 200

    def __init__(self, payload: list[dict[str, Any]]) -> None:
        self.headers: dict[str, str] = {}
        self._payload = payload
        self.content = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def json(self) -> list[dict[str, Any]]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _Client:
    def get(self, url: str, *, params: dict[str, Any], timeout: int) -> _Response:
        del timeout
        if url.endswith("/teams/fbs"):
            return _Response([{"id": 1, "school": "Alpha", "alternateNames": []}])
        if params["classification"] == "fbs":
            return _Response(
                [
                    {
                        "id": "p1",
                        "firstName": "Alex",
                        "lastName": "One",
                        "team": "Alpha",
                        "position": "OL",
                        "jersey": "72",
                    },
                    {
                        "id": "p2",
                        "firstName": "Blair",
                        "lastName": "Two",
                        "team": "Alpha",
                        "position": "OG",
                        "jersey": "73",
                    },
                ]
            )
        return _Response([])


def test_research_data_root_configuration(monkeypatch, tmp_path: Path) -> None:
    configured_root = tmp_path / "configured"
    monkeypatch.setenv("GIPPYRANK_DATA_DIR", str(configured_root))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert research_data_dir() == configured_root.resolve()
    assert (
        cfbd_roster_raw_dir()
        == (
            configured_root / "raw/cfbd/offensive_line_shared_roster_issue_183"
        ).resolve()
    )

    monkeypatch.delenv("GIPPYRANK_DATA_DIR")
    assert research_data_dir() == (tmp_path / "xdg/gippyrank/research-data").resolve()
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    assert (
        research_data_dir()
        == (Path.home() / ".cache/gippyrank/research-data").resolve()
    )


def test_acquire_and_build_with_external_raw_root(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    raw_root = tmp_path / "shared-raw-cache"
    output_dir = tmp_path / "processed"
    sample_path = tmp_path / "issue181.csv"
    sample_path.write_text(
        "sample_id,window_start,target_season,team,program_stratum,season_pair_id\n"
        "01,2024,2024,Alpha,test,2024-2025\n",
        encoding="utf-8",
    )

    manifest = acquire_roster_history(
        raw_root=raw_root,
        start_season=2024,
        end_season=2024,
        client=_Client(),  # type: ignore[arg-type]
    )
    assert Path(manifest["raw_root"]) == raw_root.resolve()
    assert all(request["path_base"] == "raw_root" for request in manifest["requests"])
    assert manifest["requests"][0]["path"] == "teams_fbs/2024.json"

    build_dataset(
        raw_root=raw_root,
        output_dir=output_dir,
        start_season=2024,
        end_season=2024,
        issue181_sample_path=sample_path,
    )
    with (output_dir / "source_inventory.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        inventory = list(csv.DictReader(handle))
    assert inventory[0]["path"] == "teams_fbs/2024.json"
    assert inventory[0]["path_base"] == "raw_root"
    assert all(str(repository_root) not in row["path"] for row in inventory)

    with (output_dir / "issue_181_overlap_coverage.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        overlap = next(csv.DictReader(handle))
    assert overlap["continuity_history_status"] == "censored_requested_panel_start"
    assert overlap["continuity_target_evaluable"] == "false"
    assert overlap["pairs_with_at_least_1_shared_season"] == ""
