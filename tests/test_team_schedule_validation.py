"""Adversarial checks at the published team-schedule boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gippyrank.site_data import (
    SiteDataValidationError,
    _validate_matching_team_schedules,
    _validate_team_season_artifact,
    build_weekly_game_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
SLOT = "2026-weekly-2026-09-27T12-27-35.698895Z"


def _source(prior: str) -> tuple[dict, dict]:
    folder = ROOT / "data/processed/snapshots/2026" / f"{SLOT}-{prior}"
    folder = folder / "predictive" / ("context" if prior == "context-v1.3" else prior)
    return (
        json.loads((folder / "team_seasons.json").read_text()),
        json.loads((folder / "metadata.json").read_text()),
    )


def _validate(artifact: dict, metadata: dict) -> None:
    snapshot = ROOT / "site/data/snapshots" / f"{metadata['snapshot_id']}.json"
    rankings = json.loads(snapshot.read_text())["rankings"]
    _validate_team_season_artifact(artifact, metadata, rankings)


def test_each_fbs_participant_must_reference_its_retrospective() -> None:
    artifact, metadata = _source("context-v1.3")
    game = next(
        game
        for game in artifact["teams"]["201"]["games"]
        if game["game_id"] == "401856700"
    )
    game["retrospective_expectation_id"] = None
    with pytest.raises(
        SiteDataValidationError, match="missing retrospective references"
    ):
        _validate(artifact, metadata)


def test_duplicate_game_for_team_is_rejected() -> None:
    artifact, metadata = _source("context-v1.3")
    artifact["teams"]["61"]["games"].append(dict(artifact["teams"]["61"]["games"][0]))
    with pytest.raises(SiteDataValidationError, match="duplicate game"):
        _validate(artifact, metadata)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"result": "L"}, "result contradicts score"),
        ({"score": {"team": -3, "opponent": 17}}, "invalid score or result"),
        ({"score": {"team": 41.0, "opponent": 13}}, "invalid score or result"),
        ({"score": None}, "incomplete result"),
        ({"result": None, "score": None}, "completed game .* lacks a result"),
        ({"game_state": "unresolved"}, "unresolved game .* has a result"),
        ({"game_state": "cancelled"}, "cancelled game .* has a result"),
    ],
)
def test_completed_evidence_cannot_contradict_itself(
    change: dict, message: str
) -> None:
    artifact, metadata = _source("context-v1.3")
    game = next(
        game
        for game in artifact["teams"]["61"]["games"]
        if game["game_id"] == "401856700"
    )
    game.update(change)
    with pytest.raises(SiteDataValidationError, match=message):
        _validate(artifact, metadata)


def test_history_merge_refuses_changed_matchup() -> None:
    context, _ = _source("context-v1.3")
    history, metadata = _source("history")
    history["teams"]["61"]["games"][0]["opponent_id"] = "different-team"
    with pytest.raises(
        SiteDataValidationError, match="Context/History schedule differs"
    ):
        _validate_matching_team_schedules(context, history, metadata["snapshot_id"])


@pytest.mark.parametrize("field,value", [
    ("opponent_name", "Different FCS name"),
    ("opponent_conference", "Different league"),
    ("conference_game", True),
    ("season_type", "postseason"),
])
def test_history_merge_checks_browser_schedule_metadata(field: str, value: object) -> None:
    context, _ = _source("context-v1.3")
    history, metadata = _source("history")
    game = next(game for game in history["teams"]["61"]["games"] if game["game_id"] == "401856658")
    game[field] = value
    with pytest.raises(SiteDataValidationError, match="Context/History schedule differs"):
        _validate_matching_team_schedules(context, history, metadata["snapshot_id"])


def test_future_game_cannot_publish_completed_evidence() -> None:
    artifact, metadata = _source("context-v1.3")
    game = next(
        game
        for game in artifact["teams"]["61"]["games"]
        if game["game_state"] == "future"
    )
    game.update({"result": "W", "score": {"team": 21, "opponent": 7}})
    with pytest.raises(SiteDataValidationError, match="future game .* has a result"):
        _validate(artifact, metadata)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("swapped_sides", "prediction site orientation mismatch"),
        ("neutral_site", "prediction site orientation mismatch"),
        ("wrong_game_id", "prediction reference points at another matchup"),
        ("missing_side", "missing FBS participant references"),
    ],
)
def test_future_prediction_requires_exact_matchup_and_both_fbs_references(
    change: str, message: str
) -> None:
    artifact, metadata = _source("context-v1.3")
    prediction = artifact["future_predictions"]["401856712"]
    if change == "swapped_sides":
        prediction["home_team_id"], prediction["away_team_id"] = (
            prediction["away_team_id"],
            prediction["home_team_id"],
        )
    elif change == "neutral_site":
        prediction["neutral_site"] = True
    elif change == "wrong_game_id":
        artifact["future_predictions"]["wrong-matchup"] = artifact[
            "future_predictions"
        ].pop("401856712")
        artifact["future_predictions"]["wrong-matchup"]["game_id"] = "wrong-matchup"
        for team_id in ("61", "333"):
            game = next(
                game
                for game in artifact["teams"][team_id]["games"]
                if game["game_id"] == "401856712"
            )
            game["future_prediction_id"] = "wrong-matchup"
    else:
        game = next(
            game
            for game in artifact["teams"]["333"]["games"]
            if game["game_id"] == "401856712"
        )
        game["future_prediction_id"] = None
    with pytest.raises(SiteDataValidationError, match=message):
        _validate(artifact, metadata)


@pytest.mark.parametrize("field", ["date", "week", "game_state"])
def test_fbs_schedule_projections_share_one_canonical_game(field: str) -> None:
    artifact, metadata = _source("context-v1.3")
    game = next(
        game
        for game in artifact["teams"]["61"]["games"]
        if game["game_id"] == "401856712"
    )
    game[field] = {
        "date": "2026-10-11T04:00:00.000Z",
        "week": 7,
        "game_state": "unresolved",
    }[field]
    with pytest.raises(SiteDataValidationError, match="cross-team schedule mismatch"):
        _validate(artifact, metadata)


def test_completed_fbs_game_requires_both_schedule_projections() -> None:
    artifact, metadata = _source("context-v1.3")
    artifact["teams"]["201"]["games"] = [
        game
        for game in artifact["teams"]["201"]["games"]
        if game["game_id"] != "401856700"
    ]
    with pytest.raises(
        SiteDataValidationError, match="missing reciprocal FBS schedule projection"
    ):
        _validate(artifact, metadata)


def test_retained_retrospective_requires_a_schedule_projection() -> None:
    artifact, metadata = _source("context-v1.3")
    for team_id in ("61", "201"):
        artifact["teams"][team_id]["games"] = [
            game for game in artifact["teams"][team_id]["games"]
            if game["game_id"] != "401856700"
        ]
    with pytest.raises(SiteDataValidationError, match="missing schedule projections"):
        _validate(artifact, metadata)


def test_fbs_fcs_retrospective_requires_its_fbs_schedule_projection() -> None:
    artifact, metadata = _source("context-v1.3")
    artifact["teams"]["61"]["games"] = [
        game for game in artifact["teams"]["61"]["games"]
        if game["game_id"] != "401856658"
    ]
    with pytest.raises(SiteDataValidationError, match="missing schedule projections"):
        _validate(artifact, metadata)


def test_browser_artifact_excludes_fcs_only_retrospectives() -> None:
    artifact, metadata = _source("context-v1.3")
    source_games = artifact["retrospective_game_expectations"]["games"]
    assert any(game["home_subdivision"] == game["away_subdivision"] == "fcs" for game in source_games.values())
    snapshot = ROOT / "site/data/snapshots" / f"{metadata['snapshot_id']}.json"
    rankings = json.loads(snapshot.read_text())["rankings"]
    adapted = _validate_team_season_artifact(artifact, metadata, rankings)
    projection = adapted["retrospective_game_expectations"]
    browser_games = projection["games"]
    assert projection["artifact_kind"] == "retrospective_game_expectations_site_projection"
    assert projection["coverage"] == "fbs_team_schedules"
    assert set(projection["published_game_ids"]) == set(browser_games)
    assert len(projection["included_game_ids"]) == len(source_games)
    assert projection["inference"]["games_evaluated"] == len(source_games)
    assert all(game["home_subdivision"] == "fbs" or game["away_subdivision"] == "fbs" for game in browser_games.values())
    assert len(browser_games) < len(source_games)


def test_team_schedule_must_follow_date_order() -> None:
    artifact, metadata = _source("context-v1.3")
    games = artifact["teams"]["61"]["games"]
    games[4], games[5] = games[5], games[4]
    with pytest.raises(SiteDataValidationError, match="schedule .*order"):
        _validate(artifact, metadata)


def test_schedule_order_uses_instants_across_different_offsets() -> None:
    artifact, metadata = _source("context-v1.3")
    games = artifact["teams"]["61"]["games"]
    first, second = games[:2]
    dates = {
        first["game_id"]: "2026-09-20T00:00:00-10:00",
        second["game_id"]: "2026-09-20T09:00:00+00:00",
    }
    for team in artifact["teams"].values():
        for game in team["games"]:
            if game["game_id"] in dates:
                game["date"] = dates[game["game_id"]]
    with pytest.raises(SiteDataValidationError, match="schedule order"):
        _validate(artifact, metadata)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("site", "unknown", "invalid site"),
        ("opponent_classification", "fcs", "invalid opponent classification"),
        ("opponent_id", "missing-fbs", "invalid opponent classification"),
        ("opponent_classification", "banana", "invalid opponent classification"),
        ("conference_game", "false", "invalid conference flag"),
    ],
)
def test_schedule_row_base_contract(field: str, value: object, message: str) -> None:
    artifact, metadata = _source("context-v1.3")
    game = next(game for game in artifact["teams"]["61"]["games"] if game["game_id"] == "401856712")
    game[field] = value
    with pytest.raises(SiteDataValidationError, match=message):
        _validate(artifact, metadata)


@pytest.mark.parametrize("field,value", [("team_id", "201"), ("team_name", "Oklahoma")])
def test_team_identity_is_rooted_in_rankings(field: str, value: str) -> None:
    artifact, metadata = _source("context-v1.3")
    artifact["teams"]["61"][field] = value
    with pytest.raises(SiteDataValidationError, match="team identity differs from rankings"):
        _validate(artifact, metadata)


def test_consistently_rewritten_display_identity_still_fails_ranking_anchor() -> None:
    artifact, metadata = _source("context-v1.3")
    artifact["teams"]["61"]["team_name"] = "Fake Georgia"
    for team in artifact["teams"].values():
        for game in team["games"]:
            if game["opponent_id"] == "61":
                game["opponent_name"] = "Fake Georgia"
    for prediction in artifact["future_predictions"].values():
        for side in ("home", "away"):
            if prediction[f"{side}_team_id"] == "61":
                prediction[f"{side}_team_name"] = "Fake Georgia"
    with pytest.raises(SiteDataValidationError, match="team identity differs from rankings"):
        _validate(artifact, metadata)


def test_scored_game_requires_timezone_aware_date() -> None:
    artifact, metadata = _source("context-v1.3")
    game = next(game for game in artifact["teams"]["61"]["games"] if game["game_id"] == "401856700")
    game["date"] = game["date"].replace("Z", "").split("+")[0]
    with pytest.raises(SiteDataValidationError, match="timezone-aware date"):
        _validate(artifact, metadata)


@pytest.mark.parametrize("date", ["2026-10-10T00:00:00", "2026-10-10T00:00:00+02:00"])
def test_unscored_calendar_anchor_must_be_timezone_aware_utc(date: str) -> None:
    artifact, metadata = _source("context-v1.3")
    game = next(game for game in artifact["teams"]["61"]["games"] if game["game_id"] == "401856712")
    game["date"] = date
    with pytest.raises(SiteDataValidationError, match="timezone-aware|UTC calendar anchor"):
        _validate(artifact, metadata)


def test_future_prediction_accepts_cutoff_calendar_day() -> None:
    artifact, metadata = _source("context-v1.3")
    date = metadata["effective_cutoff"][:10] + "T00:00:00Z"
    for team in artifact["teams"].values():
        for game in team["games"]:
            if game["game_id"] == "401856705":
                game["date"] = date
    _validate(artifact, metadata)


@pytest.mark.parametrize("target", ["team", "opponent"])
def test_fbs_conference_is_rooted_in_rankings(target: str) -> None:
    artifact, metadata = _source("context-v1.3")
    if target == "team":
        artifact["teams"]["61"]["conference"] = "MAC"
        message = "team conference differs"
    else:
        game = next(game for game in artifact["teams"]["61"]["games"] if game["opponent_classification"] == "fbs")
        game["opponent_conference"] = "MAC"
        message = "opponent conference differs"
    with pytest.raises(SiteDataValidationError, match=message):
        _validate(artifact, metadata)


def test_weekly_fold_rejects_conflicting_reciprocal_descriptors() -> None:
    artifact, metadata = _source("context-v1.3")
    rankings = json.loads((ROOT / "site/data/snapshots" / f"{metadata['snapshot_id']}.json").read_text())["rankings"]
    adapted = _validate_team_season_artifact(artifact, metadata, rankings)
    assert adapted["teams"]["61"]["conference"] == "SEC"
    game = next(game for game in adapted["teams"]["61"]["games"] if game["game_id"] == "401856700")
    assert game["opponent_conference"] == "SEC"
    game["opponent_conference"] = "MAC"
    with pytest.raises(SiteDataValidationError, match="conflicting team descriptors"):
        build_weekly_game_artifact(adapted, rankings=rankings)


def test_retrospective_schedule_subdivision_matches_record() -> None:
    artifact, metadata = _source("context-v1.3")
    game = next(game for game in artifact["teams"]["61"]["games"] if game["game_id"] == "401856658")
    game["opponent_classification"] = "fbs"
    with pytest.raises(SiteDataValidationError, match="invalid opponent classification"):
        _validate(artifact, metadata)


@pytest.mark.parametrize(
    "field,value",
    [
        ("opponent_name", "Auburn"),
        ("opponent_classification", "fcs"),
    ],
)
def test_fbs_schedule_display_identity_matches_canonical_team(
    field: str, value: str
) -> None:
    artifact, metadata = _source("context-v1.3")
    game = next(
        game
        for game in artifact["teams"]["61"]["games"]
        if game["game_id"] == "401856700"
    )
    game[field] = value
    message = "invalid opponent classification" if field == "opponent_classification" else "cross-team display identity mismatch"
    with pytest.raises(SiteDataValidationError, match=message):
        _validate(artifact, metadata)


@pytest.mark.parametrize(
    "field", ["home_team_name", "away_team_name", "away_subdivision"]
)
def test_prediction_display_identity_matches_schedule(field: str) -> None:
    artifact, metadata = _source("context-v1.3")
    prediction = artifact["future_predictions"]["401856712"]
    prediction[field] = "Auburn" if field.endswith("name") else "fcs"
    with pytest.raises(
        SiteDataValidationError, match="prediction display identity mismatch"
    ):
        _validate(artifact, metadata)


def test_prediction_tie_mass_must_be_zero() -> None:
    artifact, metadata = _source("context-v1.3")
    artifact["future_predictions"]["401856712"]["tie_probability"] = 0.75
    with pytest.raises(SiteDataValidationError, match="tie probability must be zero"):
        _validate(artifact, metadata)


@pytest.mark.parametrize("kind", ["predictive", "retrospective"])
def test_margin_median_must_lie_inside_central_intervals(kind: str) -> None:
    artifact, metadata = _source("context-v1.3")
    if kind == "predictive":
        record = artifact["future_predictions"]["401856712"]
    else:
        record = artifact["retrospective_game_expectations"]["games"]["401856700"]
    record["median_home_margin"] = record["margin_interval_95"][1] + 1
    with pytest.raises(SiteDataValidationError, match=f"{kind} median lies outside"):
        _validate(artifact, metadata)


def test_scored_out_of_scope_date_uses_local_display_policy() -> None:
    artifact, metadata = _source("context-v1.3")
    game = next(
        game
        for game in artifact["teams"]["61"]["games"]
        if game["game_id"] == "401856658"
    )
    game["game_state"] = "out_of_scope"
    snapshot = ROOT / "site/data/snapshots" / f"{metadata['snapshot_id']}.json"
    rankings = json.loads(snapshot.read_text())["rankings"]
    adapted = _validate_team_season_artifact(artifact, metadata, rankings)
    selected = next(
        game
        for game in adapted["teams"]["61"]["games"]
        if game["game_id"] == "401856658"
    )
    assert selected["date_display_mode"] == "local_time"
