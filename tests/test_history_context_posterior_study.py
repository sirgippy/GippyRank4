"""Guard the matched study against model-origin and evidence drift."""

import csv
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from gippyrank.posterior.engine import Game

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from study_history_context_posterior import cutoff_safe_games

STUDY = ROOT / "data/processed/history_context_posterior_study"


def rows(name: str) -> list[dict[str, str]]:
    with (STUDY / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_cutoff_excludes_final_cached_scores_without_safe_availability() -> None:
    cutoff = datetime(2024, 9, 19, 23, 59, 59, tzinfo=UTC)
    games = [
        Game(key, "home", "away", "fbs", "fbs", 7, 3)
        for key in ("recent", "boundary", "old")
    ]
    included = [
        {
            "id": key,
            "startDate": (cutoff - timedelta(hours=hours)).isoformat(),
            "homeTeam": "Home",
            "awayTeam": "Away",
        }
        for key, hours in (("recent", 0.4), ("boundary", 48), ("old", 72))
    ]
    safe_games, safe_rows, audit = cutoff_safe_games(games, included, cutoff)
    assert [game.game_id for game in safe_games] == ["boundary", "old"]
    assert [row["id"] for row in safe_rows] == ["boundary", "old"]
    assert [row["game_id"] for row in audit] == ["recent"]


def test_historical_arms_share_history_and_evidence() -> None:
    summaries = rows("historical_checkpoints.csv")
    evidence = rows("historical_evidence.csv")
    all_teams = [row for row in summaries if row["transfer_group"] == "all"]
    by_checkpoint: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in all_teams:
        key = row["season"], row["checkpoint"]
        by_checkpoint.setdefault(key, {})[row["panel"]] = row
    assert len(by_checkpoint) == 28
    for pair in by_checkpoint.values():
        assert set(pair) == {
            "frozen_2021",
            "rolling_origin",
            "rolling_context_frozen_history",
        }
        frozen = pair["frozen_2021"]
        rolling = pair["rolling_origin"]
        sensitivity = pair["rolling_context_frozen_history"]
        assert len({row["teams"] for row in pair.values()}) == 1
        for field in frozen:
            if field.startswith("history_"):
                assert frozen[field] == sensitivity[field]
            if field.startswith("context_"):
                assert rolling[field] == sensitivity[field]
    for season in (2023, 2024, 2025):
        pair = by_checkpoint[str(season), "1"]
        assert (
            pair["frozen_2021"]["history_prior_nll"]
            != pair["rolling_origin"]["history_prior_nll"]
        )

    by_evidence: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in evidence:
        key = row["season"], row["checkpoint"]
        by_evidence.setdefault(key, {})[row["family"]] = row
        assert row["converged"] == "True"
        assert int(row["iterations"]) <= 500
    assert set(by_evidence) == set(by_checkpoint)
    audit = rows("historical_cutoff_audit.csv")
    excluded = {}
    for row in audit:
        key = row["season"], row["checkpoint"]
        excluded[key] = excluded.get(key, 0) + 1
        assert 0 <= float(row["hours_since_kickoff"]) < 48
        assert row["game_id"]
    assert {(row["season"], row["checkpoint"], row["game_id"]) for row in audit} >= {
        ("2024", "2", "401643790"),  # South Alabama at Appalachian State
        ("2025", "2", "401756900"),  # Tulsa at Oklahoma State
    }
    for key, quartet in by_evidence.items():
        assert set(quartet) == {
            "frozen_context",
            "rolling_context",
            "frozen_history",
            "rolling_history",
        }
        assert len({row["included_games"] for row in quartet.values()}) == 1
        assert (
            len({row["excluded_lower_division_games"] for row in quartet.values()}) == 1
        )
        assert {
            int(row["indeterminate_games_excluded"]) for row in quartet.values()
        } == {excluded.get(key, 0)}


def test_study_uses_production_inference_configuration() -> None:
    provenance = json.loads((STUDY / "provenance.json").read_text())
    assert provenance["inference"] == {
        "max_iterations": 500,
        "tolerance": 1e-9,
        "damping": 0.35,
    }
    assert provenance["models"]["context_rolling_origin"] == {
        str(season): f"1.3 research P3 fitted through {season - 1}"
        for season in (2022, 2023, 2024, 2025)
    }
    assert provenance["models"]["history_rolling_origin"] == {
        str(season): f"1.1 annual production builder fitted through {season - 1}"
        for season in (2022, 2023, 2024, 2025)
    }
    assert "48 hours" in provenance["result_availability"]


def test_all_panels_cover_same_annual_checkpoints() -> None:
    summaries = [
        row
        for row in rows("historical_checkpoints.csv")
        if row["transfer_group"] == "all"
    ]
    for panel in ("frozen_2021", "rolling_origin", "rolling_context_frozen_history"):
        for season in (2022, 2023, 2024, 2025):
            annual = sorted(
                (
                    row
                    for row in summaries
                    if row["panel"] == panel and int(row["season"]) == season
                ),
                key=lambda row: int(row["checkpoint"]),
            )
            assert len(annual) == 7
            assert [int(row["checkpoint"]) for row in annual] == list(range(1, 8))
            assert len({row["cutoff"] for row in annual}) == 7
