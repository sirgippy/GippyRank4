"""Check that the development bridge's compact scores match the canonical scorer."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gippyrank.posterior.engine import Game, Team
from gippyrank.posterior.snapshots import load_likelihood

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bridge_context_v1_4_predictive import _score_bridge
from validate_predictive_game_v1 import _score_model


def test_three_bridge_metrics_match_established_exact_mixture_scorer() -> None:
    likelihood = load_likelihood(
        ROOT / "data/processed/posterior/historical_likelihood_v1.json"
    )
    home = Team("home", "Home", "fbs", np.array([0.2, 0.3, 0.5]))
    away = Team("away", "Away", "fbs", np.array([0.4, 0.4, 0.2]))
    game = Game("synthetic", "home", "away", "fbs", "fbs", 24, 17)
    pmfs = {team.team_id: team.prior for team in (home, away)}
    compact = _score_bridge((game,), [home, away], pmfs, likelihood)
    canonical = _score_model(
        SimpleNamespace(prepared=SimpleNamespace(future_games=(game,))),
        "synthetic",
        SimpleNamespace(anchor_result=SimpleNamespace(pmfs=pmfs), teams=(home, away)),
        likelihood,
    )
    assert compact["n"] == canonical["n"] == 1
    for metric in ("nll", "mae", "brier"):
        assert compact[metric] == pytest.approx(canonical[metric], abs=1e-12)
