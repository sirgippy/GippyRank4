from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gippyrank.posterior.engine import Game, LikelihoodV1, Team, game_factor
from gippyrank.posterior.snapshots import (
    add_fcs_fallbacks,
    subdivision_population_size,
    subdivision_population_source,
)


def _row(team_id: str, team_name: str) -> dict[str, str]:
    return {
        "homeId": "fbs",
        "homeTeam": "FBS",
        "homeClassification": "fbs",
        "homeConference": "FBS conference",
        "awayId": team_id,
        "awayTeam": team_name,
        "awayClassification": "fcs",
        "awayConference": "FCS conference",
    }


def test_single_encountered_fcs_team_uses_full_universe_support() -> None:
    teams, fallbacks = add_fcs_fallbacks(
        [Team("fbs", "FBS", "fbs", np.array([0.5, 0.5]))],
        {},
        [_row("fcs-1", "FCS 1")],
        129,
    )
    fallback = next(team for team in teams if team.team_id == "fcs-1")
    assert fallbacks == ("fcs-1",)
    assert len(fallback.prior) == 129
    assert fallback.prior.sum() == pytest.approx(1.0)


def test_fallback_support_is_cutoff_and_encounter_count_invariant() -> None:
    base = [Team("fbs", "FBS", "fbs", np.array([0.5, 0.5]))]
    early, _ = add_fcs_fallbacks(base.copy(), {}, [_row("fcs-1", "FCS 1")], 127)
    later, _ = add_fcs_fallbacks(
        base.copy(), {}, [_row("fcs-1", "FCS 1"), _row("fcs-2", "FCS 2")], 127
    )
    assert len(next(team for team in early if team.team_id == "fcs-1").prior) == 127
    assert len(next(team for team in later if team.team_id == "fcs-1").prior) == 127
    assert len(next(team for team in later if team.team_id == "fcs-2").prior) == 127


def test_frozen_fcs_prior_precedes_fallback() -> None:
    fcs_prior = Team("fcs-1", "FCS 1", "fcs", np.array([0.2, 0.3, 0.5]))
    teams, fallbacks = add_fcs_fallbacks(
        [Team("fbs", "FBS", "fbs", np.array([0.5, 0.5])), fcs_prior],
        {},
        [_row("fcs-1", "FCS 1")],
        129,
    )
    assert fallbacks == ()
    assert next(team for team in teams if team.team_id == "fcs-1") is fcs_prior


def test_missing_fcs_population_fails_clearly() -> None:
    with pytest.raises(ValueError, match="authoritative full-season FCS population"):
        add_fcs_fallbacks([], {}, [_row("fcs-1", "FCS 1")], None)


def test_cross_factor_uses_full_fcs_rank_coordinate_support() -> None:
    beta = np.zeros(34)
    beta[10], beta[11] = -50.0, 50.0
    factor = game_factor(
        Game("fbs-fcs", "fbs", "fcs", "fbs", "fcs", 35, 7),
        Team("fbs", "FBS", "fbs", np.array([0.5, 0.5])),
        Team("fcs", "FCS", "fcs", np.full(129, 1 / 129)),
        LikelihoodV1(beta, 7.0, 15.0),
    )
    assert factor.shape == (2, 129)


def test_historical_fcs_population_comes_from_durable_full_universe() -> None:
    assert subdivision_population_size(Path("."), 2025, "fcs") == 129


def test_current_fcs_population_uses_full_cached_schedule_not_cutoff(
    tmp_path: Path,
) -> None:
    games = tmp_path / "data/raw/cfbd/games"
    games.mkdir(parents=True)
    (games / "2026-fcs.json").write_text(
        """[
          {"homeId": 1, "homeClassification": "fcs", "awayId": 2, "awayClassification": "fcs"},
          {"homeId": 2, "homeClassification": "fcs", "awayId": 3, "awayClassification": "fcs"}
        ]""",
        encoding="utf-8",
    )
    assert subdivision_population_size(tmp_path, 2026, "fcs") == 3
    assert subdivision_population_source(tmp_path, 2026, "fcs") == "cfbd_full_season_schedule"
