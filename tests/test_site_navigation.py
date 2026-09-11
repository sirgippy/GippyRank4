from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PAGES = ("index.html", "team.html", "schedule.html", "methodology.html")


def _page(filename: str) -> str:
    return (ROOT / "site" / filename).read_text(encoding="utf-8")


def test_public_headers_share_canonical_brand_and_global_navigation() -> None:
    for filename in PUBLIC_PAGES:
        page = _page(filename)
        assert '<header class="site-header">' in page
        assert '<div class="shell masthead">' in page
        assert 'aria-label="GippyRank4 home">GippyRank4</a>' in page
        assert 'aria-label="Site navigation"' in page
        assert 'id="rankings-view-link" href="./"' in page
        assert 'id="schedule-view-link" class="schedule-link" href="./schedule.html"' in page
        assert "GippyRank 4.0" not in page
        assert "College football rankings, built from the games." not in page

    assert 'id="rankings-view-link" href="./" aria-current="page">Rankings</a>' in _page("index.html")
    assert 'id="schedule-view-link" class="schedule-link" href="./schedule.html" aria-current="page">Schedule</a>' in _page("schedule.html")
    assert '<a href="./methodology.html" aria-current="page">Methodology</a>' in _page("methodology.html")
    assert 'id="rankings-view-link" href="./" aria-current="page">Rankings</a>' not in _page("team.html")


def test_schedule_has_canonical_name_and_route() -> None:
    schedule = _page("schedule.html")

    assert '<link rel="canonical" href="./schedule.html">' in schedule
    assert "<title>Schedule · GippyRank4</title>" in schedule
    assert '<main class="shell schedule-page" aria-labelledby="schedule-page-title">' in schedule
    assert '<h1 id="schedule-page-title">Schedule</h1>' in schedule
    assert 'id="week-select" aria-label="Selected week"' in schedule
    assert 'src="./assets/week.js"' in schedule
    assert "Week games" not in schedule


def test_schedule_presentation_keeps_the_primary_flow_scannable() -> None:
    schedule = _page("schedule.html")
    week = (ROOT / "site/assets/week.js").read_text(encoding="utf-8")

    assert 'id="schedule-context"' not in schedule
    assert 'id="schedule-kind"' not in schedule
    assert 'class="schedule-summary"' not in schedule
    assert 'id="schedule-summary-cards"' not in schedule
    assert 'id="weekly-schedule-context"' not in schedule
    assert 'class="team-page-status sr-only"' in schedule
    assert "How to read this week" in schedule
    assert "Completed games show how well each team played" in schedule
    assert "Upcoming games show GippyRank's win probabilities" in schedule

    for phrase in (
        "Completed at snapshot",
        "Future at snapshot",
        "One card per scheduled game",
        "canonical home-minus-away",
        "selected snapshot through",
        "Home minus away",
        "Prediction unavailable —",
        "Not modeled —",
        "Inferred performance",
    ):
        assert phrase not in week

    assert "completed: \"Final\"" in week
    assert "future: \"Upcoming\"" in week
    assert "cancelled: \"Canceled / postponed\"" in week
    assert "unresolved: \"Result unavailable\"" in week
    assert "No prediction available for this matchup." in week
    assert "Performance rating unavailable for this game." in week
    assert "Result unavailable in this snapshot." in week
    assert "weekly-winner-label\", \"Winner\"" in week
    assert "weekly-result" not in week
    assert "scheduled_game_count" not in week
    assert "Neutral site" in week
    assert "team_name} home" not in week
    assert 'node("span", "axis-label", "Even")' in week
    assert 'node("span", "axis-label", "Performance")' in week


def test_week_route_is_a_query_preserving_compatibility_shell() -> None:
    legacy = _page("week.html")
    redirect = (ROOT / "site/assets/week-redirect.js").read_text(encoding="utf-8")

    assert '<link rel="canonical" href="./schedule.html">' in legacy
    assert "<title>Schedule · GippyRank4</title>" in legacy
    assert 'src="./assets/week.js"' not in legacy
    assert 'src="./assets/week-redirect.js"' in legacy
    assert 'scheduleUrl.search = window.location.search;' in redirect
    assert 'scheduleUrl.hash = window.location.hash;' in redirect
    assert 'window.location.replace(scheduleUrl.href);' in redirect
    assert "Week games" not in legacy + redirect


def test_state_preserving_schedule_links_keep_publication_context() -> None:
    app = (ROOT / "site/assets/app.js").read_text(encoding="utf-8")
    team = (ROOT / "site/assets/team.js").read_text(encoding="utf-8")
    schedule = (ROOT / "site/assets/week.js").read_text(encoding="utf-8")

    assert "function schedulePageUrl(entry, week = null)" in app
    assert 'new URL("./schedule.html", document.baseURI)' in app
    assert 'url.searchParams.set("season", String(entry.season));' in app
    assert 'url.searchParams.set("snapshot", entry.snapshot_id);' in app
    assert 'url.searchParams.set("family", state.family);' in app
    assert 'url.searchParams.set("prior", entry.prior_family);' in app
    assert 'url.searchParams.set("week", String(week));' in app

    assert 'const scheduleUrl = new URL("./schedule.html", document.baseURI);' in team
    assert "scheduleUrl.search = url.search;" in team
    assert 'scheduleUrl.searchParams.set("week", params.get("week") ?? entry.default_week);' in team
    assert 'const context = contextParams(entry, week).toString();' in schedule
    assert "scheduleUrl.search = context;" in schedule
    assert "next.set(\"week\", String(week));" in schedule


def test_schedule_naming_is_updated_in_documentation_and_ballot_copy() -> None:
    weekly_schema = (ROOT / "docs/weekly_game_schema.md").read_text(encoding="utf-8")
    ballot_docs = (ROOT / "docs/redditcfb_ballot_export.md").read_text(encoding="utf-8")
    ballot_source = (ROOT / "src/gippyrank/redditcfb.py").read_text(encoding="utf-8")
    app = (ROOT / "site/assets/app.js").read_text(encoding="utf-8")

    assert "The canonical public page is `schedule.html`." in weekly_schema
    assert "query string" in weekly_schema
    assert "hash to `schedule.html`" in weekly_schema
    assert "Generated from GippyRank4" in ballot_docs
    assert "Generated from GippyRank4" in ballot_source
    assert "Generated from GippyRank4" in app
    assert "GippyRank 4.0" not in ballot_docs + ballot_source + app
