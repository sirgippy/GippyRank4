from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from scipy.special import stdtr

from gippyrank.artifact_hashes import INCLUDED_GAME_FIELDS
from gippyrank.performance_snapshot import PerformanceSnapshotValidationError
from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    Team,
    game_margin_parameters,
)
from gippyrank.posterior.snapshots import Snapshot, build_snapshot, load_likelihood
from gippyrank.quality_of_record import binary_game_factor, infer_quality_of_record
from gippyrank.quality_of_record_snapshot import (
    build_quality_of_record_snapshot,
    validate_quality_of_record_against_context,
)


def _likelihood() -> LikelihoodV1:
    beta = np.zeros(34)
    beta[0] = -30.0
    beta[10], beta[11] = -30.0, 30.0
    return LikelihoodV1(beta, 7.0, 15.0)


def _teams() -> list[Team]:
    return [
        Team("a", "A", "fbs", np.array([0.6, 0.3, 0.1])),
        Team("b", "B", "fbs", np.array([0.2, 0.4, 0.4])),
        Team("c", "C", "fbs", np.array([0.3, 0.4, 0.3])),
        Team("idle", "Idle", "fbs", np.array([0.7, 0.2, 0.1])),
        Team("fcs", "FCS", "fcs", np.array([0.5, 0.5])),
    ]


def _game(
    game_id: str,
    home: str,
    away: str,
    home_points: int,
    away_points: int,
    *,
    neutral: bool = False,
) -> Game:
    subdivisions = {team.team_id: team.subdivision for team in _teams()}
    return Game(
        game_id,
        home,
        away,
        subdivisions[home],
        subdivisions[away],
        home_points,
        away_points,
        neutral,
    )


@pytest.mark.parametrize(("first", "second"), [(1, 50), (-1, -50)])
def test_binary_factor_ignores_victory_and_defeat_size(first: int, second: int) -> None:
    teams = _teams()
    likelihood = _likelihood()
    left = _game("g", "a", "b", 60 + first, 60)
    right = _game("g", "a", "b", 60 + second, 60)
    assert np.array_equal(
        binary_game_factor(left, teams[0], teams[1], likelihood),
        binary_game_factor(right, teams[0], teams[1], likelihood),
    )
    opposite = _game("g", "a", "b", 60 - first, 60)
    assert np.allclose(
        binary_game_factor(left, teams[0], teams[1], likelihood)
        + binary_game_factor(opposite, teams[0], teams[1], likelihood),
        1,
    )


@pytest.mark.parametrize(
    ("home", "away", "neutral"),
    [("a", "b", False), ("a", "b", True), ("a", "fcs", False), ("fcs", "a", False)],
)
def test_binary_factor_uses_frozen_site_and_subdivision_orientation(
    home: str, away: str, neutral: bool
) -> None:
    by_id = {team.team_id: team for team in _teams()}
    likelihood = _likelihood()
    game = _game("g", home, away, 21, 20, neutral=neutral)
    location, margin = game_margin_parameters(
        game, by_id[home], by_id[away], likelihood
    )
    factor = binary_game_factor(game, by_id[home], by_id[away], likelihood)
    assert factor.shape == (len(by_id[home].prior), len(by_id[away].prior))
    assert np.allclose(
        factor,
        stdtr(
            likelihood.degrees_of_freedom, np.sign(margin) * location / likelihood.scale
        ),
    )
    reverse = replace(game, home_points=20, away_points=21)
    assert np.allclose(
        factor + binary_game_factor(reverse, by_id[home], by_id[away], likelihood), 1
    )


def test_frozen_site_effect_and_neutral_cross_subdivision_transpose() -> None:
    by_id = {team.team_id: team for team in _teams()}
    likelihood = load_likelihood(
        Path(__file__).resolve().parents[1]
        / "data/processed/posterior/historical_likelihood_v1.json"
    )
    home = _game("home", "a", "b", 21, 20)
    neutral = replace(home, neutral_site=True)
    assert not np.allclose(
        binary_game_factor(home, by_id["a"], by_id["b"], likelihood),
        binary_game_factor(neutral, by_id["a"], by_id["b"], likelihood),
    )
    fbs_home = _game("fbs", "a", "fcs", 21, 20, neutral=True)
    fcs_home = _game("fcs", "fcs", "a", 20, 21, neutral=True)
    assert np.allclose(
        binary_game_factor(fbs_home, by_id["a"], by_id["fcs"], likelihood),
        binary_game_factor(fcs_home, by_id["fcs"], by_id["a"], likelihood).T,
    )


def test_binary_network_is_margin_invariant_for_every_team() -> None:
    teams = _teams()
    games = [
        _game("ab1", "a", "b", 21, 20),
        _game("ab2", "b", "a", 10, 9),
        _game("bc", "b", "c", 10, 9),
        _game("ca", "c", "a", 10, 9, neutral=True),
        _game("afcs", "fcs", "a", 10, 9),
    ]
    widened = [replace(game, home_points=50, away_points=0) for game in games]
    first = infer_quality_of_record(teams, games, _likelihood())
    second = infer_quality_of_record(teams, widened, _likelihood())
    repeated = infer_quality_of_record(teams, games, _likelihood())
    for team in teams:
        assert np.array_equal(
            first.binary_context_posterior.pmfs[team.team_id],
            second.binary_context_posterior.pmfs[team.team_id],
        )
        if team.subdivision == "fbs":
            assert np.array_equal(first.pmfs[team.team_id], second.pmfs[team.team_id])
            assert np.array_equal(first.pmfs[team.team_id], repeated.pmfs[team.team_id])
    assert first.eligible_game_counts["a"] == 4
    assert first.eligible_game_counts["idle"] == 0
    assert np.array_equal(first.pmfs["idle"], np.full(3, 1 / 3))


def test_undefeated_and_winless_teams_remain_valid() -> None:
    games = [
        _game("ab", "a", "b", 21, 20),
        _game("ac", "a", "c", 21, 20),
        _game("bc", "b", "c", 21, 20),
    ]
    result = infer_quality_of_record(_teams(), games, _likelihood())
    assert result.eligible_game_counts["a"] == 2
    assert result.eligible_game_counts["c"] == 2
    assert all(
        np.isfinite(pmf).all() and np.isclose(pmf.sum(), 1)
        for pmf in result.pmfs.values()
    )
    assert not np.array_equal(result.pmfs["a"], result.pmfs["c"])


def test_ties_and_nonconvergence_fail_closed() -> None:
    teams = _teams()
    tied = _game("tie", "a", "b", 20, 20)
    with pytest.raises(ValueError, match="no binary winner"):
        infer_quality_of_record(teams, [tied], _likelihood())
    with pytest.raises(RuntimeError, match="did not converge"):
        infer_quality_of_record(
            teams, [_game("g", "a", "b", 21, 20)], _likelihood(), max_iterations=1
        )


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _context_snapshot(root: Path) -> Snapshot:
    _write_csv(
        root / "data/processed/preseason/context_v1_4/annual/2026/predictions.csv",
        ["season", "subdivision", "team_id", "team_name", "conference", "pmf"],
        [
            {
                "season": 2026,
                "subdivision": team.subdivision,
                "team_id": team.team_id,
                "team_name": team.name,
                "conference": "",
                "pmf": json.dumps(team.prior.tolist()),
            }
            for team in _teams()
        ],
    )
    row = {field: "" for field in INCLUDED_GAME_FIELDS}
    row.update(
        id="g",
        season="2026",
        week="1",
        seasonType="regular",
        startDate="2026-08-29T00:00:00Z",
        completed="True",
        neutralSite="False",
        homeId="a",
        homeTeam="A",
        homeClassification="fbs",
        homePoints="21",
        awayId="b",
        awayTeam="B",
        awayClassification="fbs",
        awayPoints="20",
    )
    _write_csv(
        root / "data/processed/cfbd/games.csv",
        [*INCLUDED_GAME_FIELDS, "startTimeTBD"],
        [{**row, "startTimeTBD": "False"}],
    )
    return build_snapshot(
        season=2026,
        cutoff=datetime(2026, 9, 1, tzinfo=UTC),
        prior_family="context",
        snapshot_type="weekly",
        root=root,
        likelihood=_likelihood(),
        generation_timestamp=datetime(2026, 9, 1, tzinfo=UTC),
    )


def test_snapshot_uses_frozen_binary_inputs_and_validates_lineage(
    tmp_path: Path,
) -> None:
    context = _context_snapshot(tmp_path)
    first = build_quality_of_record_snapshot(
        context, root=tmp_path, output_root=tmp_path / "first"
    )
    second = build_quality_of_record_snapshot(
        context, root=tmp_path, output_root=tmp_path / "second"
    )
    validate_quality_of_record_against_context(context, first)
    assert first.metadata["performance_variant"] == "quality_of_record"
    assert first.metadata["model"] == "quality_of_record_v1"
    assert first.metadata["evidence_mode"] == "win_loss"
    assert (
        first.metadata["historical_likelihood_sha256"]
        == context.metadata["historical_likelihood_sha256"]
    )
    assert (first.directory / "posterior_pmfs.csv").read_bytes() == (
        second.directory / "posterior_pmfs.csv"
    ).read_bytes()
    with (first.directory / "rankings.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        ranked = {row["team_id"]: row for row in csv.DictReader(handle)}
    assert ranked["idle"]["display_rank"] == "NR"
    assert ranked["idle"]["rated"] == "False"
    assert ranked["a"]["eligible_games"] == "1"
    for key in (
        "expected_rank",
        "median_rank",
        "mode_rank",
        "interval_50_low",
        "interval_80_high",
        "interval_95_low",
        "top5_probability",
        "top10_probability",
        "top25_probability",
    ):
        assert key in ranked["a"]
    for field in (
        "effective_cutoff",
        "included_game_ids",
        "historical_likelihood_sha256",
        "prior_pmfs_sha256",
    ):
        metadata = dict(first.metadata)
        metadata[field] = "wrong"
        with pytest.raises(PerformanceSnapshotValidationError, match="mismatch"):
            validate_quality_of_record_against_context(
                context, replace(first, metadata=metadata)
            )


def test_snapshot_rejects_tampered_included_game_scores(tmp_path: Path) -> None:
    context = _context_snapshot(tmp_path)
    path = context.directory / "included_games.csv"
    path.write_text(
        path.read_text(encoding="utf-8").replace("21", "50"), encoding="utf-8"
    )
    with pytest.raises(
        PerformanceSnapshotValidationError, match="included games mismatch"
    ):
        build_quality_of_record_snapshot(context, root=tmp_path)


def test_snapshot_validator_rejects_tampered_output(tmp_path: Path) -> None:
    context = _context_snapshot(tmp_path)
    snapshot = build_quality_of_record_snapshot(context, root=tmp_path)
    path = snapshot.directory / "posterior_pmfs.csv"
    path.write_text(
        path.read_text(encoding="utf-8").replace("0.3333333333333333", "0.5"),
        encoding="utf-8",
    )
    with pytest.raises(PerformanceSnapshotValidationError, match="PMF"):
        validate_quality_of_record_against_context(context, snapshot)
