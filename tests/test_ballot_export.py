from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_publishes_canonical_ballot_mapping_audit() -> None:
    manifest = json.loads((ROOT / "site/data/manifest.json").read_text(encoding="utf-8"))
    redditcfb = manifest["redditcfb"]

    assert manifest["site_url"] == "https://sirgippy.github.io/GippyRank4/"
    assert redditcfb["mapping_path"] == "data/reference/redditcfb_team_handles.csv"
    assert redditcfb["mapping_audit"]["current_fbs_team_count"] == 138
    assert redditcfb["mapping_audit"]["mapped_count"] == 138
    assert redditcfb["mapping_audit"]["unmapped_teams"] == []
    assert redditcfb["mapping_audit"]["duplicate_cfbd_ids"] == []
    assert redditcfb["mapping_audit"]["duplicate_redditcfb_handles"] == []


def test_rankings_page_has_deterministic_r_cfb_export_without_submission_code() -> None:
    index = (ROOT / "site/index.html").read_text(encoding="utf-8")
    app = (ROOT / "site/assets/app.js").read_text(encoding="utf-8")

    assert 'id="download-ballot"' in index
    assert "poll_type: \"computer\"" in app
    assert "overallBallotRationale(entry)" in app
    assert "teamBallotRationale(row)" in app
    assert "Central 80% interval" in app
    assert "GippyRank Performance-equivalent expected rank" in app
    assert "state.manifest.redditcfb.team_handles" in app
    assert "JSON.stringify(ballot, null, 2)" in app
    assert "createObjectURL" in app
    assert "oauth" not in app.casefold()
    assert "submit" not in app.casefold()


def test_ballot_export_uses_published_rows_and_fail_closed_guard() -> None:
    app = (ROOT / "site/assets/app.js").read_text(encoding="utf-8")

    assert "displayedRatedRows(snapshot).slice(0, 25)" in app
    assert "A ballot requires 25 rated teams" in app
    assert "canonical r/CFB handles are unavailable" in app
    assert "row.expected_rank.toFixed(1)" in app
    assert "row.interval_80[0]" in app
    assert "percentage(row.top25_probability)" in app
    assert "ballotRows(state.snapshot).map((row, index)" in app
