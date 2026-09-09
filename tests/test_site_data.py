from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from gippyrank import site_data
from gippyrank.site_data import (
    SiteDataValidationError,
    build_fbs_conference_map,
    build_site_data,
)
from gippyrank.team_logos import logo_url, missing_team_logos, team_logo_handle

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "site/publish_config.json"
SCHEDULE_FIELDS = [
    "season",
    "homeId",
    "homeClassification",
    "homeConference",
    "awayId",
    "awayClassification",
    "awayConference",
]


def _hash_tree(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(directory).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _config_for(source: Path, tmp_path: Path, *, slot: str = "test") -> Path:
    config = tmp_path / "publish.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "publication_slots": [{"id": slot, "status": "official"}],
                "snapshots": [
                    {
                        "source": str(source.relative_to(tmp_path)),
                        "display_label": "Test",
                        "publication_slot": slot,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return config


def _copied_snapshot(tmp_path: Path, relative_source: str | None = None) -> Path:
    source = ROOT / (
        relative_source
        or "data/processed/snapshots/2026/2026-preseason-context/predictive/context"
    )
    destination = tmp_path / "snapshot"
    shutil.copytree(source, destination)
    schedule = tmp_path / "data/processed/cfbd/games.csv"
    schedule.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "data/processed/cfbd/games.csv", schedule)
    return destination


def _schedule_row(
    season: int,
    home_id: str,
    home_classification: str,
    home_conference: str,
    away_id: str,
    away_classification: str,
    away_conference: str,
) -> dict[str, str]:
    return {
        "season": str(season),
        "homeId": home_id,
        "homeClassification": home_classification,
        "homeConference": home_conference,
        "awayId": away_id,
        "awayClassification": away_classification,
        "awayConference": away_conference,
    }


def _write_schedule(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEDULE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _pmf_rows(source: Path) -> tuple[Path, list[str], list[dict[str, str]]]:
    path = source / "posterior_pmfs.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    return path, fields, rows


def _write_pmf_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_publication_slot_status_is_required_and_closed(tmp_path: Path) -> None:
    source = _copied_snapshot(tmp_path)
    config = _config_for(source, tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    del payload["publication_slots"]
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SiteDataValidationError, match="publication_slots"):
        build_site_data(root=tmp_path, config_path=config, output_directory=tmp_path / "data")

    payload["publication_slots"] = [{"id": "test", "status": "unknown"}]
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SiteDataValidationError, match="official or temporary"):
        build_site_data(root=tmp_path, config_path=config, output_directory=tmp_path / "data")


def test_logo_url_uses_canonical_handle_and_supports_template_override() -> None:
    assert team_logo_handle("194", "Ohio State") == "ohiostate"
    assert logo_url("ohiostate") == "https://cdn.redditcfb.com/60x40/cfb/ohiostate.png"
    assert logo_url("ohiostate", "https://assets.example/{handle}/mark.svg") == (
        "https://assets.example/ohiostate/mark.svg"
    )


def test_logo_mapping_fails_closed_on_unknown_or_renamed_identity() -> None:
    assert team_logo_handle("194", "The Ohio State University") is None
    assert team_logo_handle("not-a-team", "Ohio State") is None
    assert missing_team_logos([("194", "Ohio State"), ("194", "Renamed")]) == [
        ("194", "Renamed")
    ]


def test_exported_team_logos_are_canonical_and_audited(tmp_path: Path) -> None:
    manifest = build_site_data(
        root=ROOT, config_path=CONFIG, output_directory=tmp_path / "data"
    )
    assert manifest["team_logos"]["source"] == "RedditCFB"
    assert manifest["team_logos"]["url_template"] == (
        "https://cdn.redditcfb.com/60x40/cfb/{handle}.png"
    )
    assert manifest["team_logos"]["handles"]["194"] == "ohiostate"
    assert manifest["team_logos"]["handles"]["311"] == "maine"
    assert "2277" not in manifest["team_logos"]["handles"]
    audit = manifest["team_logo_audit"]
    assert audit["published_fbs"] == {"mapped": 138, "total": 138}
    assert audit["all_rendered_team_identities"] == {"mapped": 235, "total": 238}
    assert audit["missing"] == [
        {"team_id": "2277", "team_name": "Houston Christian"},
        {"team_id": "2546", "team_name": "Southeast Missouri State"},
        {"team_id": "2837", "team_name": "East Texas A&M"},
    ]
    entry = manifest["snapshots"][0]
    snapshot = json.loads(
        (tmp_path / "data" / entry["data_path"].removeprefix("data/")).read_text()
    )
    assert "logo_handle" not in snapshot["rankings"][0]


def test_publish_config_logo_template_override_is_manifest_visible(
    tmp_path: Path,
) -> None:
    source = _copied_snapshot(tmp_path)
    config = _config_for(source, tmp_path)
    payload = json.loads(config.read_text())
    payload["team_logos"] = {
        "url_template": "https://static.example/logos/{handle}.svg"
    }
    config.write_text(json.dumps(payload), encoding="utf-8")
    manifest = build_site_data(
        root=tmp_path, config_path=config, output_directory=tmp_path / "output"
    )
    assert manifest["team_logos"]["url_template"] == (
        "https://static.example/logos/{handle}.svg"
    )


def test_team_season_export_uses_manifest_logo_handles(tmp_path: Path) -> None:
    manifest = build_site_data(
        root=ROOT, config_path=CONFIG, output_directory=tmp_path / "data"
    )
    entry = manifest["snapshots"][0]
    artifact = json.loads(
        (tmp_path / "data" / entry["team_seasons_path"].removeprefix("data/")).read_text()
    )
    ohio_state = artifact["teams"]["194"]
    assert "logo_handle" not in ohio_state
    assert all("opponent_logo_handle" not in game for game in ohio_state["games"])
    assert manifest["team_logos"]["handles"]["194"] == "ohiostate"
    assert manifest["team_logos"]["handles"]["311"] == "maine"


def test_static_site_uses_manifest_logo_config_and_decorative_fallback() -> None:
    app = (ROOT / "site/assets/app.js").read_text(encoding="utf-8")
    team = (ROOT / "site/assets/team.js").read_text(encoding="utf-8")
    css = (ROOT / "site/assets/style.css").read_text(encoding="utf-8")
    assert 'state.manifest?.team_logos?.url_template' in app
    assert 'state.manifest?.team_logos?.handles?.[teamId]' in app
    assert 'teamLogo(row.team_id' in app
    assert 'image.alt = ""' in app
    assert 'image.addEventListener("error", () => frame.remove()' in app
    assert 'logoUrlTemplate = manifest.team_logos?.url_template || null' in team
    assert 'logoHandles = manifest.team_logos?.handles || {}' in team
    assert 'teamLogo(game.opponent_id)' in team
    assert 'row.logo_handle' not in app + team
    assert 'opponent_logo_handle' not in app + team
    assert ".team-logo-frame" in css
    assert "loading = \"lazy\"" in app


def test_team_schedule_uses_distinct_accessible_performance_and_margin_plots() -> None:
    team = (ROOT / "site/assets/team.js").read_text(encoding="utf-8")
    css = (ROOT / "site/assets/style.css").read_text(encoding="utf-8")
    html = (ROOT / "site/team.html").read_text(encoding="utf-8")

    assert "performanceChart" in team
    assert "futureChart" in team
    assert "performance_axis" in team
    assert "future_margin_axis" in team
    assert "probability_encoding" in team
    assert "role: \"img\"" in team
    assert "aria-label" in team
    assert "Central 80% range" in team
    assert ".game-distribution-chart" in css
    assert ".game-distribution-future .distribution-bar" in css
    assert "More performance detail" in team
    assert "More predictive detail" in team
    assert "Expected performance rank" in team
    assert "expectedPrimary" in team
    assert "Played like" not in team
    assert "game-prediction-interval" not in team
    assert "best FBS performance is on the left" in html
    assert "Even in the middle" in html


def test_future_prediction_range_labels_follow_focal_margin_sign() -> None:
    team = (ROOT / "site/assets/team.js").read_text(encoding="utf-8")

    assert (
        'if (high < 0) return `${marginSide(oriented.opponentName, -high)} '
        'to ${marginSide(oriented.opponentName, -low)}`;' in team
    )
    assert (
        'if (low > 0) return `${marginSide(oriented.focalName, low)} '
        'to ${marginSide(oriented.focalName, high)}`;' in team
    )
    assert (
        'const lower = low < 0 ? marginSide(oriented.opponentName, -low) : "Even";'
        in team
    )
    assert (
        'const upper = high > 0 ? marginSide(oriented.focalName, high) : "Even";'
        in team
    )


def _counterpart(
    entries: list[dict[str, object]], current: dict[str, object], prior: str
) -> dict[str, object]:
    """Mirror the UI's slot-based counterpart lookup contract."""
    return next(
        (
            entry
            for entry in entries
            if entry["season"] == current["season"]
            and entry["ranking_family"] == current["ranking_family"]
            and entry["publication_slot"] == current["publication_slot"]
            and entry.get("prior_family") == prior
        ),
        current,
    )


def test_conference_map_uses_both_home_and_away_appearances(tmp_path: Path) -> None:
    schedule = tmp_path / "games.csv"
    _write_schedule(
        schedule,
        [_schedule_row(2026, "1", "fbs", "Home League", "2", "fbs", "Away League")],
    )

    assert build_fbs_conference_map(schedule) == {
        (2026, "1"): "Home League",
        (2026, "2"): "Away League",
    }


def test_conference_map_is_season_specific(tmp_path: Path) -> None:
    schedule = tmp_path / "games.csv"
    _write_schedule(
        schedule,
        [
            _schedule_row(2025, "1", "fbs", "Old League", "2", "fbs", "Other"),
            _schedule_row(2026, "1", "fbs", "New League", "3", "fbs", "Other"),
        ],
    )

    conference_map = build_fbs_conference_map(schedule)

    assert conference_map[(2025, "1")] == "Old League"
    assert conference_map[(2026, "1")] == "New League"


def test_fcs_conference_values_cannot_overwrite_fbs_metadata(tmp_path: Path) -> None:
    schedule = tmp_path / "games.csv"
    _write_schedule(
        schedule,
        [
            _schedule_row(2026, "1", "fbs", "FBS League", "1", "fcs", "FCS League"),
        ],
    )

    assert build_fbs_conference_map(schedule) == {(2026, "1"): "FBS League"}


def test_conflicting_fbs_conferences_fail_closed(tmp_path: Path) -> None:
    schedule = tmp_path / "games.csv"
    _write_schedule(
        schedule,
        [
            _schedule_row(2026, "1", "fbs", "First League", "2", "fbs", "Other"),
            _schedule_row(2026, "3", "fbs", "Other", "1", "fbs", "Second League"),
        ],
    )

    with pytest.raises(SiteDataValidationError, match="Conflicting nonblank FBS conferences"):
        build_fbs_conference_map(schedule)


def test_preseason_zero_game_team_uses_schedule_conference(tmp_path: Path) -> None:
    schedule = tmp_path / "games.csv"
    _write_schedule(
        schedule,
        [_schedule_row(2026, "1", "fbs", "Future League", "2", "fcs", "FCS League")],
    )

    conference_map = build_fbs_conference_map(schedule, seasons={2026})

    assert conference_map[(2026, "1")] == "Future League"


def test_blank_ranking_conferences_are_enriched_from_schedule(tmp_path: Path) -> None:
    source = _copied_snapshot(tmp_path)
    with (source / "rankings.csv").open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    assert all(row["conference"] == "" for row in source_rows)

    manifest = build_site_data(
        root=tmp_path,
        config_path=_config_for(source, tmp_path),
        output_directory=tmp_path / "data",
    )
    entry = manifest["snapshots"][0]
    snapshot = json.loads(
        (tmp_path / entry["data_path"]).read_text(encoding="utf-8")
    )
    conference_map = build_fbs_conference_map(
        tmp_path / "data/processed/cfbd/games.csv", seasons={2026}
    )

    assert all(
        row["conference"] == conference_map[(2026, row["team_id"])]
        for row in snapshot["rankings"]
    )


def test_all_ranking_families_receive_consistent_season_conferences(tmp_path: Path) -> None:
    manifest = build_site_data(
        root=ROOT, config_path=CONFIG, output_directory=tmp_path / "data"
    )
    conferences_by_team: dict[tuple[int, str], set[str]] = {}
    for entry in manifest["snapshots"]:
        snapshot = json.loads(
            (tmp_path / "data" / entry["data_path"].removeprefix("data/")).read_text(
                encoding="utf-8"
            )
        )
        for row in snapshot["rankings"]:
            conferences_by_team.setdefault((entry["season"], row["team_id"]), set()).add(
                row["conference"]
            )

    assert conferences_by_team
    assert all(len(conferences) == 1 for conferences in conferences_by_team.values())


def test_conference_enrichment_preserves_ranking_pmf_and_record_values(
    tmp_path: Path,
) -> None:
    source = _copied_snapshot(tmp_path)
    with (source / "rankings.csv").open(newline="", encoding="utf-8") as handle:
        source_rows = {row["team_id"]: row for row in csv.DictReader(handle)}
    source_records = site_data._records(source / "included_games.csv")
    source_pmfs: dict[str, dict[int, float]] = {}
    with (source / "posterior_pmfs.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["team_id"] in source_rows:
                source_pmfs.setdefault(row["team_id"], {})[int(row["rank"])] = float(
                    row["probability"]
                )

    manifest = build_site_data(
        root=tmp_path,
        config_path=_config_for(source, tmp_path),
        output_directory=tmp_path / "data",
    )
    entry = manifest["snapshots"][0]
    snapshot = json.loads(
        (tmp_path / entry["data_path"]).read_text(encoding="utf-8")
    )
    distribution = json.loads(
        (tmp_path / entry["distribution_path"]).read_text(encoding="utf-8")
    )

    numeric_fields = (
        "expected_rank",
        "median_rank",
        "top5_probability",
        "top10_probability",
        "top25_probability",
    )
    for row in snapshot["rankings"]:
        original = source_rows[row["team_id"]]
        for field in numeric_fields:
            assert row[field] == pytest.approx(float(original[field]))
        assert str(row["display_rank"]) == original["display_rank"]
        assert row["record"] == source_records.get(row["team_id"], "0-0")
        assert distribution["teams"][row["team_id"]]["pmf"] == pytest.approx(
            [
                source_pmfs[row["team_id"]][rank]
                for rank in range(1, len(source_pmfs[row["team_id"]]) + 1)
            ]
        )


def test_site_data_publishes_all_initial_h_c_preseason_and_current_snapshots(
    tmp_path: Path,
) -> None:
    manifest = build_site_data(
        root=ROOT, config_path=CONFIG, output_directory=tmp_path / "data"
    )
    assert manifest["seasons"] == [2026]
    configured_default_slot = json.loads(CONFIG.read_text(encoding="utf-8"))[
        "default_publication_slot"
    ]
    assert manifest["default_publication_slot"] == configured_default_slot
    predictive = [entry for entry in manifest["snapshots"] if entry["ranking_family"] == "predictive"]
    assert {(entry["snapshot_type"], entry["prior_family"]) for entry in predictive} == {
        ("preseason", "context"),
        ("preseason", "history"),
        ("live", "context"),
        ("live", "history"),
        ("weekly", "context"),
        ("weekly", "history"),
    }
    performance = [
        entry
        for entry in manifest["snapshots"]
        if entry["ranking_family"] == "performance"
        and entry["publication_slot"] == configured_default_slot
    ]
    assert len(performance) == 1
    assert performance[0]["publication_slot"] == configured_default_slot
    assert "prior_family" not in performance[0]
    assert (
        performance[0]["rated_count"] + performance[0]["unrated_count"]
        == performance[0]["rank_count"]
    )
    current = [entry for entry in manifest["snapshots"] if entry["snapshot_type"] == "live"]
    assert {entry["effective_cutoff"] for entry in current} == {
        "2026-09-05T20:11:12.864212+00:00"
    }
    weekly = [
        entry
        for entry in manifest["snapshots"]
        if entry["snapshot_type"] == "weekly"
        and entry["publication_slot"] == configured_default_slot
    ]
    assert {entry["effective_cutoff"] for entry in weekly} == {
        performance[0]["effective_cutoff"]
    }


def test_publication_status_and_official_comparison_chain(tmp_path: Path) -> None:
    manifest = build_site_data(
        root=ROOT, config_path=CONFIG, output_directory=tmp_path / "data"
    )
    status_by_slot = {
        entry["publication_slot"]: entry["publication_status"]
        for entry in manifest["snapshots"]
        if entry["ranking_family"] == "predictive" and entry["prior_family"] == "context"
    }
    assert status_by_slot == {
        "2026-preseason": "official",
        "2026-sep-05": "temporary",
        "2026-09-06": "temporary",
        "2026-09-07": "temporary",
        "2026-09-08": "official",
    }

    def entry(slot: str, family: str = "context", ranking_family: str = "predictive"):
        return next(
            item
            for item in manifest["snapshots"]
            if item["publication_slot"] == slot
            and item["ranking_family"] == ranking_family
            and item.get("prior_family") == (family if ranking_family == "predictive" else None)
        )

    preseason = entry("2026-preseason")
    sep_5 = entry("2026-sep-05")
    sep_7 = entry("2026-09-07")
    week_2 = entry("2026-09-08")
    performance = entry("2026-09-08", ranking_family="performance")
    assert preseason["comparison_snapshot_id"] is None
    assert sep_5["comparison_snapshot_id"] == preseason["snapshot_id"]
    assert sep_7["comparison_snapshot_id"] == preseason["snapshot_id"]
    assert week_2["comparison_snapshot_id"] == preseason["snapshot_id"]
    assert performance["comparison_snapshot_id"] is None


def test_comparison_resolution_uses_explicit_order_and_compatible_family() -> None:
    def prepared(
        slot: str,
        order: int,
        status: str,
        *,
        family: str = "context",
        ranking_family: str = "predictive",
    ) -> site_data.PreparedSnapshot:
        selected = site_data.PublishedSnapshot(
            Path(slot), slot, slot, status, order
        )
        metadata = {
            "season": 2026,
            "ranking_family": ranking_family,
            "prior_family": family,
        }
        return site_data.PreparedSnapshot(selected, metadata, slot, [], {}, {}, {})

    preseason = prepared("week-z", 0, "official")
    temporary = prepared("week-a", 1, "temporary")
    history = prepared("week-b", 2, "official", family="history")
    current = prepared("week-c", 3, "official")
    performance = prepared(
        "week-d", 4, "official", ranking_family="performance", family="context"
    )
    week_2_performance = prepared(
        "week-e", 5, "official", ranking_family="performance", family="context"
    )
    later_performance = prepared(
        "week-f", 6, "temporary", ranking_family="performance", family="context"
    )
    assert site_data._previous_official_snapshot(
        temporary,
        [preseason, temporary, history, current, performance, week_2_performance, later_performance],
    ) is preseason
    assert site_data._previous_official_snapshot(
        current,
        [preseason, temporary, history, current, performance, week_2_performance, later_performance],
    ) is preseason
    assert site_data._previous_official_snapshot(
        performance,
        [preseason, temporary, history, current, performance, week_2_performance, later_performance],
    ) is None
    assert site_data._previous_official_snapshot(
        later_performance,
        [preseason, temporary, history, current, performance, week_2_performance, later_performance],
    ) is week_2_performance


def test_future_predictive_temporary_and_week3_compare_to_week2() -> None:
    def prepared(slot: str, order: int, status: str) -> site_data.PublicationComparison:
        return site_data.PublicationComparison(
            season=2026,
            ranking_family="predictive",
            prior_family="context",
            publication_slot=slot,
            publication_status=status,
            publication_order=order,
            snapshot_id=slot,
            display_label=slot,
        )

    preseason = prepared("2026-preseason", 0, "official")
    week_2 = prepared("2026-09-08", 1, "official")
    later_temporary = prepared("2026-09-09", 2, "temporary")
    week_3 = prepared("2026-09-15", 3, "official")
    snapshots = [preseason, week_2, later_temporary, week_3]

    assert site_data.resolve_previous_official(later_temporary, snapshots) is week_2
    assert site_data.resolve_previous_official(week_3, snapshots) is week_2


def test_rank_change_uses_display_rank_and_handles_nr_transitions() -> None:
    previous = site_data.PreparedSnapshot(
        site_data.PublishedSnapshot(Path("old"), "Preseason", "old", "official", 0),
        {"season": 2026, "ranking_family": "performance"},
        "old",
        [
            {"team_id": "up", "rated": True, "display_rank": 17},
            {"team_id": "same", "rated": True, "display_rank": 4},
            {"team_id": "new", "rated": False, "display_rank": "NR"},
            {"team_id": "gone", "rated": True, "display_rank": 35},
            {"team_id": "still-nr", "rated": False, "display_rank": "NR"},
        ],
        {},
        {},
        {},
    )
    current = site_data.PreparedSnapshot(
        site_data.PublishedSnapshot(Path("new"), "Week 2", "new", "official", 1),
        {"season": 2026, "ranking_family": "performance"},
        "new",
        [
            {"team_id": "up", "rated": True, "display_rank": 12},
            {"team_id": "same", "rated": True, "display_rank": 4},
            {"team_id": "new", "rated": True, "display_rank": 22},
            {"team_id": "gone", "rated": False, "display_rank": "NR"},
            {"team_id": "still-nr", "rated": False, "display_rank": "NR"},
        ],
        {},
        {},
        {},
    )
    site_data._apply_rank_changes(current, previous)
    rows = {row["team_id"]: row for row in current.rankings}
    assert (rows["up"]["rank_change"], rows["up"]["rank_change_status"]) == (5, "ranked")
    assert rows["same"]["rank_change_display"] == "—"
    assert rows["new"]["rank_change_status"] == "newly_rated"
    assert rows["new"]["rank_change"] is None
    assert rows["gone"]["rank_change_status"] == "became_unrated"
    assert rows["gone"]["rank_change"] is None
    assert rows["still-nr"]["rank_change_status"] == "unrated"


@pytest.mark.parametrize(
    ("slot", "from_prior", "to_prior"),
    [
        ("2026-preseason", "context", "history"),
        ("2026-preseason", "history", "context"),
        ("2026-sep-05", "context", "history"),
        ("2026-sep-05", "history", "context"),
        ("2026-09-06", "context", "history"),
        ("2026-09-06", "history", "context"),
    ],
)
def test_logical_publication_slots_pair_context_and_history(
    tmp_path: Path, slot: str, from_prior: str, to_prior: str
) -> None:
    manifest = build_site_data(
        root=ROOT, config_path=CONFIG, output_directory=tmp_path / "data"
    )
    current = next(
        entry
        for entry in manifest["snapshots"]
        if entry["publication_slot"] == slot and entry["prior_family"] == from_prior
    )
    counterpart = _counterpart(manifest["snapshots"], current, to_prior)
    assert counterpart["prior_family"] == to_prior
    assert counterpart["publication_slot"] == slot
    assert counterpart["snapshot_id"] != current["snapshot_id"]
    assert counterpart["distribution_path"] != current["distribution_path"]


def test_missing_logical_counterpart_keeps_current_selection(tmp_path: Path) -> None:
    manifest = build_site_data(
        root=ROOT, config_path=CONFIG, output_directory=tmp_path / "data"
    )
    current = next(
        entry
        for entry in manifest["snapshots"]
        if entry["publication_slot"] == "2026-sep-05"
        and entry["prior_family"] == "context"
    )
    entries = [
        entry
        for entry in manifest["snapshots"]
        if not (
            entry["publication_slot"] == "2026-sep-05"
            and entry.get("prior_family") == "history"
        )
    ]
    assert _counterpart(entries, current, "history") == current


def test_site_data_is_byte_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "data"
    build_site_data(root=ROOT, config_path=CONFIG, output_directory=output)
    first = _hash_tree(output)
    build_site_data(root=ROOT, config_path=CONFIG, output_directory=output)
    assert _hash_tree(output) == first


def test_distribution_artifact_contains_complete_fbs_pmfs_and_summaries(tmp_path: Path) -> None:
    manifest = build_site_data(
        root=ROOT, config_path=CONFIG, output_directory=tmp_path / "data"
    )
    entry = next(item for item in manifest["snapshots"] if item["prior_family"] == "context")
    distribution = json.loads(
        (tmp_path / "data" / entry["distribution_path"].removeprefix("data/")).read_text()
    )
    snapshot = json.loads(
        (tmp_path / "data" / entry["data_path"].removeprefix("data/")).read_text()
    )
    assert distribution["schema_version"] == "1.0"
    assert distribution["snapshot_id"] == entry["snapshot_id"]
    assert distribution["rank_count"] == 138
    assert set(distribution["teams"]) == {row["team_id"] for row in snapshot["rankings"]}
    team = distribution["teams"][snapshot["rankings"][0]["team_id"]]
    assert len(team["pmf"]) == distribution["rank_count"]
    assert sum(team["pmf"]) == pytest.approx(1.0, abs=1e-9)
    assert set(team["summary"]) == {
        "expected_rank", "median_rank", "modal_rank", "interval_50", "interval_80",
        "interval_95", "interval_widths", "rank_1_probability", "top5_probability",
        "top10_probability", "top25_probability",
    }


def test_performance_export_keeps_nr_after_rated_teams_and_out_of_top25(tmp_path: Path) -> None:
    manifest = build_site_data(
        root=ROOT, config_path=CONFIG, output_directory=tmp_path / "data"
    )
    entry = next(item for item in manifest["snapshots"] if item["ranking_family"] == "performance")
    snapshot = json.loads(
        (tmp_path / "data" / entry["data_path"].removeprefix("data/")).read_text()
    )
    assert all(row["rated"] for row in snapshot["rankings"][:25])
    assert all(row["display_rank"] == "NR" for row in snapshot["rankings"][131:])


def test_pmf_summary_uses_established_discrete_quantiles() -> None:
    summary = site_data._pmf_summary([0.1, 0.2, 0.3, 0.2, 0.2])
    assert summary["expected_rank"] == pytest.approx(3.2)
    assert summary["median_rank"] == 3
    assert summary["modal_rank"] == 3
    assert summary["interval_50"] == [2, 4]
    assert summary["interval_80"] == [1, 5]
    assert summary["interval_95"] == [1, 5]
    assert summary["interval_widths"] == {"50": 3, "80": 5, "95": 5}


def test_exported_80_percent_interval_remains_the_ranking_interval(tmp_path: Path) -> None:
    manifest = build_site_data(
        root=ROOT, config_path=CONFIG, output_directory=tmp_path / "data"
    )
    for entry in manifest["snapshots"]:
        snapshot = json.loads(
            (tmp_path / "data" / entry["data_path"].removeprefix("data/")).read_text()
        )
        distribution = json.loads(
            (tmp_path / "data" / entry["distribution_path"].removeprefix("data/")).read_text()
        )
        for row in snapshot["rankings"]:
            assert distribution["teams"][row["team_id"]]["summary"]["interval_80"] == row["interval_80"]


@pytest.mark.parametrize(
    "relative_source",
    [
        "data/processed/snapshots/2026/2026-preseason-context/predictive/context",
        "data/processed/snapshots/2026/2026-live-2026-09-05T23-59-59Z-context/predictive/context",
    ],
)
def test_preseason_and_in_season_distribution_exports_work(
    tmp_path: Path, relative_source: str
) -> None:
    source = _copied_snapshot(tmp_path, relative_source)
    manifest = build_site_data(
        root=tmp_path, config_path=_config_for(source, tmp_path), output_directory=tmp_path / "data"
    )
    entry = manifest["snapshots"][0]
    distribution = json.loads(
        (tmp_path / "data" / entry["distribution_path"].removeprefix("data/")).read_text()
    )
    assert distribution["rank_count"] == 138
    assert len(distribution["teams"]) == 138


def test_context_and_history_export_distinct_distribution_artifacts(tmp_path: Path) -> None:
    manifest = build_site_data(
        root=ROOT, config_path=CONFIG, output_directory=tmp_path / "data"
    )
    for slot in {entry["publication_slot"] for entry in manifest["snapshots"]}:
        context, history = (
            next(
                entry
                for entry in manifest["snapshots"]
                if entry["publication_slot"] == slot and entry["prior_family"] == prior
            )
            for prior in ("context", "history")
        )
        context_path = tmp_path / "data" / context["distribution_path"].removeprefix("data/")
        history_path = tmp_path / "data" / history["distribution_path"].removeprefix("data/")
        assert context_path != history_path
        assert context_path.read_bytes() != history_path.read_bytes()


def test_missing_pmf_rank_is_refused(tmp_path: Path) -> None:
    source = _copied_snapshot(tmp_path)
    path, fields, rows = _pmf_rows(source)
    team_id = rows[0]["team_id"]
    _write_pmf_rows(path, fields, [row for row in rows if not (row["team_id"] == team_id and row["rank"] == "1")])
    with pytest.raises(SiteDataValidationError, match="missing or unexpected ranks"):
        build_site_data(root=tmp_path, config_path=_config_for(source, tmp_path), output_directory=tmp_path / "data")


def test_duplicate_pmf_rank_is_refused(tmp_path: Path) -> None:
    source = _copied_snapshot(tmp_path)
    path, fields, rows = _pmf_rows(source)
    team_id = rows[0]["team_id"]
    next(row for row in rows if row["team_id"] == team_id and row["rank"] == "2")["rank"] = "1"
    _write_pmf_rows(path, fields, rows)
    with pytest.raises(SiteDataValidationError, match="duplicate PMF rank"):
        build_site_data(root=tmp_path, config_path=_config_for(source, tmp_path), output_directory=tmp_path / "data")


def test_pmf_team_mismatch_is_refused(tmp_path: Path) -> None:
    source = _copied_snapshot(tmp_path)
    path, fields, rows = _pmf_rows(source)
    team_id = rows[0]["team_id"]
    for row in rows:
        if row["team_id"] == team_id:
            row["team_id"] = "not-a-ranking-team"
    _write_pmf_rows(path, fields, rows)
    with pytest.raises(SiteDataValidationError, match=f"missing PMF for FBS team {team_id}"):
        build_site_data(root=tmp_path, config_path=_config_for(source, tmp_path), output_directory=tmp_path / "data")


@pytest.mark.parametrize(
    ("value", "message"),
    [("nan", "must be finite"), ("-0.01", "must be between 0 and 1"), ("1.01", "must be between 0 and 1")],
)
def test_malformed_pmf_probability_is_refused(
    tmp_path: Path, value: str, message: str
) -> None:
    source = _copied_snapshot(tmp_path)
    path, fields, rows = _pmf_rows(source)
    rows[0]["probability"] = value
    _write_pmf_rows(path, fields, rows)
    with pytest.raises(SiteDataValidationError, match=message):
        build_site_data(root=tmp_path, config_path=_config_for(source, tmp_path), output_directory=tmp_path / "data")


def test_unnormalized_pmf_is_refused_without_renormalizing(tmp_path: Path) -> None:
    source = _copied_snapshot(tmp_path)
    path, fields, rows = _pmf_rows(source)
    rows[0]["probability"] = "0.1"
    _write_pmf_rows(path, fields, rows)
    with pytest.raises(SiteDataValidationError, match="sum to"):
        build_site_data(root=tmp_path, config_path=_config_for(source, tmp_path), output_directory=tmp_path / "data")


def test_invalid_snapshot_is_refused(tmp_path: Path) -> None:
    source = _copied_snapshot(tmp_path)
    metadata = json.loads((source / "metadata.json").read_text())
    metadata["valid"] = False
    (source / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(SiteDataValidationError, match="not valid"):
        build_site_data(
            root=tmp_path,
            config_path=_config_for(source, tmp_path),
            output_directory=tmp_path / "data",
        )


def test_lower_division_game_does_not_change_modeled_record(tmp_path: Path) -> None:
    source = _copied_snapshot(
        tmp_path,
        "data/processed/snapshots/2026/2026-live-2026-09-05T23-59-59Z-context/"
        "predictive/context",
    )
    config = _config_for(source, tmp_path)
    build_site_data(
        root=tmp_path, config_path=config, output_directory=tmp_path / "first"
    )
    snapshot_id = json.loads((source / "metadata.json").read_text())["snapshot_id"]
    first_data = json.loads((tmp_path / f"first/snapshots/{snapshot_id}.json").read_text())
    team = first_data["rankings"][0]
    games = source / "included_games.csv"
    with games.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    lower_division = dict(rows[0])
    lower_division.update(
        {
            "id": "not-eligible",
            "homeId": team["team_id"],
            "homeClassification": "fbs",
            "homePoints": "99",
            "awayId": "division-ii",
            "awayClassification": "ii",
            "awayPoints": "0",
        }
    )
    rows.append(lower_division)
    with games.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    build_site_data(root=tmp_path, config_path=config, output_directory=tmp_path / "second")
    second_data = json.loads(
        (tmp_path / f"second/snapshots/{snapshot_id}.json").read_text()
    )
    assert second_data["rankings"][0]["record"] == team["record"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("display_rank", "1", "duplicate or invalid FBS display rank"),
        ("top25_probability", "1.1", "must be between 0 and 1"),
    ],
)
def test_malformed_ranking_values_are_refused(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    source = _copied_snapshot(tmp_path)
    rankings = source / "rankings.csv"
    with rankings.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[1][field] = value
    with rankings.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SiteDataValidationError, match=message):
        build_site_data(
            root=tmp_path,
            config_path=_config_for(source, tmp_path),
            output_directory=tmp_path / "data",
        )


def test_unsupported_snapshot_schema_is_refused(tmp_path: Path) -> None:
    source = _copied_snapshot(tmp_path)
    metadata_path = source / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["schema_version"] = "999.0"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(SiteDataValidationError, match="unsupported snapshot schema"):
        build_site_data(
            root=tmp_path,
            config_path=_config_for(source, tmp_path),
            output_directory=tmp_path / "data",
        )


def test_unknown_ranking_family_is_refused(tmp_path: Path) -> None:
    source = _copied_snapshot(tmp_path)
    metadata_path = source / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["ranking_family"] = "unknown"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(SiteDataValidationError, match="unsupported ranking family"):
        build_site_data(
            root=tmp_path,
            config_path=_config_for(source, tmp_path),
            output_directory=tmp_path / "data",
        )


def test_registered_published_families_drive_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _copied_snapshot(tmp_path)
    metadata_path = source / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["ranking_family"] = "resume"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setitem(site_data.RANKING_FAMILIES, "resume", {"label": "Résumé"})
    manifest = build_site_data(
        root=tmp_path,
        config_path=_config_for(source, tmp_path),
        output_directory=tmp_path / "data",
    )
    assert manifest["ranking_families"] == [{"id": "resume", "label": "Résumé"}]


def test_site_builder_has_no_model_import_path() -> None:
    source = (ROOT / "src/gippyrank/site_data.py").read_text(encoding="utf-8")
    assert "gippyrank.posterior" not in source
    assert "gippyrank.preseason" not in source


def test_site_uses_base_safe_relative_paths() -> None:
    index = (ROOT / "site/index.html").read_text(encoding="utf-8")
    app = (ROOT / "site/assets/app.js").read_text(encoding="utf-8")
    assert 'href="./assets/style.css"' in index
    assert 'src="./assets/app.js"' in index
    assert 'fetch("./data/manifest.json")' in app
    assert 'fetch(`./${entry.distribution_path}`)' in app
    assert "distributionCache" in app
    assert "selectedEntry()?.snapshot_id !== entry.snapshot_id" in app
    assert "publication_slot" in app
    assert "publication_status" in app
    assert "rank_change_accessible" in app
    assert "Change vs" in app
    assert "rank-change-inline" in app
    assert "staying on" in app
    assert 'state.family === "performance"' in app
    assert "No eligible games played" in app


def test_rankings_header_is_compact_and_links_to_existing_about_disclosure() -> None:
    index = (ROOT / "site/index.html").read_text(encoding="utf-8")
    app = (ROOT / "site/assets/app.js").read_text(encoding="utf-8")

    assert "<title>GippyRank4</title>" in index
    assert 'aria-label="GippyRank4 home">GippyRank4</a>' in index
    assert '<a class="about-link" href="#about-rankings">About these rankings</a>' in index
    assert '<details id="about-rankings" class="about-rankings">' in index
    assert '<section class="intro"' not in index
    assert "GippyRank 4.0" not in index
    assert "College football rankings, built from the games." not in index
    assert "Expected rank and an 80% interval for every team." not in index
    assert 'aboutRankings.open = true' in app


def test_bare_rankings_url_defaults_to_a_published_season() -> None:
    """Keep the dependency-free site bootstrap safe when the URL has no query."""
    app = (ROOT / "site/assets/app.js").read_text(encoding="utf-8")
    manifest = json.loads((ROOT / "site/data/manifest.json").read_text(encoding="utf-8"))
    assert 'pageParams.has("season") ? Number(pageParams.get("season")) : null' in app
    assert "Number.isSafeInteger(requestedSeason) && requestedSeason > 0" in app
    assert "if (!seasons.includes(state.season)) state.season = seasons[0];" in app
    assert manifest["seasons"]
    assert manifest["default_publication_slot"] in {
        entry["publication_slot"] for entry in manifest["snapshots"]
    }


def test_uncertainty_copy_uses_central_interval_language() -> None:
    app = (ROOT / "site/assets/app.js").read_text(encoding="utf-8")
    assert "The central 50% interval spans" in app
    assert "The central 80% interval spans" in app
    assert "The central 95% interval spans" in app
    assert "Half of the posterior lies between" not in app
    assert "80% lies between" not in app
    assert "95% lies between" not in app


def test_percentage_formatter_preserves_nonzero_and_noncertainty_distinctions() -> None:
    """Keep lightweight static coverage because this dependency-free site has no JS runner."""
    app = (ROOT / "site/assets/app.js").read_text(encoding="utf-8")
    assert 'if (value === 1) return "100%";' in app
    assert 'if (valueAsPercent < 0.01) return "<0.01%";' in app
    assert 'if (valueAsPercent >= 99.95) return "<100%";' in app
    assert 'if (valueAsPercent >= 95) return `${valueAsPercent.toFixed(1)}%`;' in app
    assert 'return `${Math.round(valueAsPercent)}%`;' in app
