from __future__ import annotations

import json
from pathlib import Path

import pytest

from gippyrank.redditcfb import BallotExportError, build_ballot

ROOT = Path(__file__).resolve().parents[1]


def _published_entry(manifest: dict[str, object], family: str) -> dict[str, object]:
    return next(
        entry
        for entry in manifest["snapshots"]
        if entry["display_label"] == "Week 2"
        and entry["ranking_family"] == family
        and (family == "performance" or entry["prior_family"] == "context")
    )


@pytest.mark.parametrize("family", ["predictive", "performance"])
def test_functional_ballot_export_matches_published_ranking(family: str) -> None:
    manifest = json.loads((ROOT / "site/data/manifest.json").read_text(encoding="utf-8"))
    entry = _published_entry(manifest, family)
    snapshot = json.loads((ROOT / "site" / entry["data_path"]).read_text(encoding="utf-8"))
    handles = manifest["redditcfb"]["team_handles"]
    ballot = build_ballot(
        season=entry["season"],
        snapshot_label=entry["display_label"],
        ranking_family=family,
        prior_family=entry.get("prior_family"),
        site_url=manifest["site_url"],
        rankings=snapshot["rankings"],
        handles=handles,
    )

    rated = sorted(
        (row for row in snapshot["rankings"] if row["rated"] is not False),
        key=lambda row: (row["expected_rank"], row["display_rank"], row["team_id"]),
    )[:25]
    entries = ballot["entries"]
    assert ballot["poll_type"] == "computer"
    assert len(entries) == 25
    assert [entry["rank"] for entry in entries] == list(range(1, 26))
    assert len({entry["team_handle"] for entry in entries}) == 25
    assert [entry["team_handle"] for entry in entries] == [
        handles[row["team_id"]] for row in rated
    ]
    assert "GippyRank 4.0" in ballot["overall_rationale"]
    assert "2026 Week 2" in ballot["overall_rationale"]
    assert manifest["site_url"] in ballot["overall_rationale"]
    assert all("Central 80% interval" in entry["rationale"] for entry in entries)
    if family == "predictive":
        assert "2026 Week 2 Predictive Context" in ballot["overall_rationale"]
        assert all("GippyRank expected rank:" in entry["rationale"] for entry in entries)
    else:
        assert "2026 Week 2 Performance" in ballot["overall_rationale"]
        assert all(
            "GippyRank Performance-equivalent expected rank:" in entry["rationale"]
            for entry in entries
        )


def test_functional_ballot_export_rejects_missing_handles_and_short_rankings() -> None:
    manifest = json.loads((ROOT / "site/data/manifest.json").read_text(encoding="utf-8"))
    entry = _published_entry(manifest, "predictive")
    snapshot = json.loads((ROOT / "site" / entry["data_path"]).read_text(encoding="utf-8"))
    kwargs = {
        "season": entry["season"],
        "snapshot_label": entry["display_label"],
        "ranking_family": "predictive",
        "prior_family": "context",
        "site_url": manifest["site_url"],
        "rankings": snapshot["rankings"],
        "handles": manifest["redditcfb"]["team_handles"],
    }
    with pytest.raises(BallotExportError, match="handles are unavailable"):
        build_ballot(**{**kwargs, "handles": {}})
    with pytest.raises(BallotExportError, match="requires 25 rated teams"):
        build_ballot(**{**kwargs, "rankings": snapshot["rankings"][:24]})


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
