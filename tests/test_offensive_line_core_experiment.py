from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_offensive_line_core_issue_189 as experiment

from gippyrank.preseason import TeamSeason
from gippyrank.research.offensive_line_continuity_experiment import (
    build_program_memberships,
)
from gippyrank.research.offensive_line_core_experiment import (
    CORE_INDIVIDUAL,
    CORE_SHARED,
    CORE_TALENT_INTERACTION,
    build_core_panel,
    core_values_for_pool,
    observed_college_seasons,
)


def _history_rows() -> list[dict[str, str]]:
    rows = [
        {"season": str(year), "team_id": "home", "normalized_player_identity": player}
        for player in "ABCD"
        for year in (2017, 2018, 2019)
    ]
    rows += [
        {"season": "2019", "team_id": "home", "normalized_player_identity": "E"},
        {"season": "2018", "team_id": "away", "normalized_player_identity": "F"},
        {"season": "2018", "team_id": "other", "normalized_player_identity": "F"},
        {"season": "2015", "team_id": "home", "normalized_player_identity": "F"},
        {"season": "2020", "team_id": "home", "normalized_player_identity": "F"},
        {"season": "2021", "team_id": "home", "normalized_player_identity": "F"},
    ]
    return rows


def _values(rows: list[dict[str, str]], ids: list[str]):
    memberships, _, _, _ = build_program_memberships(rows)
    return core_values_for_pool(
        season=2020,
        team_id="home",
        target_player_ids=ids,
        college_seasons=observed_college_seasons(rows),
        program_memberships=memberships,
    )


def test_all_program_experience_deduplicates_multi_program_year_and_obeys_chronology() -> (
    None
):
    rows = _history_rows()
    college = observed_college_seasons(rows)
    assert college["F"] == {2015, 2018, 2020, 2021}
    values = _values(rows, list("ABCDEF"))
    assert values[CORE_INDIVIDUAL] == pytest.approx(2.6)
    assert values["core_tied_realizations"] == 2
    assert values["core_boundary_tied_players"] == 2
    # Future and T-5 records do not promote F above E.
    assert values[CORE_SHARED] == pytest.approx(2.0)


def test_shared_continuity_uses_current_program_only_and_averages_fifth_place_tie() -> (
    None
):
    rows = _history_rows()
    assert _values(rows, list("ABCDE"))[CORE_SHARED] == pytest.approx(2.2)
    assert _values(rows, list("ABCDF"))[CORE_SHARED] == pytest.approx(1.8)
    expected = _values(rows, list("ABCDEF"))
    assert expected[CORE_SHARED] == pytest.approx(2.0)
    assert _values(list(reversed(rows)), list(reversed("ABCDEF"))) == expected


def test_fewer_than_five_usable_identities_is_unavailable() -> None:
    values = _values(_history_rows(), list("ABCD"))
    assert values[CORE_INDIVIDUAL] is None
    assert values[CORE_SHARED] is None
    assert values["core_tied_realizations"] == 0


def test_incomplete_four_year_history_is_unavailable() -> None:
    values = core_values_for_pool(
        season=2012,
        team_id="home",
        target_player_ids=list("ABCDE"),
        college_seasons={},
        program_memberships={},
    )
    assert values[CORE_INDIVIDUAL] is None
    assert values[CORE_SHARED] is None


def test_core_panel_preserves_coverage_and_marks_missingness() -> None:
    prior = [
        {
            "season": "2020",
            "team_id": "home",
            "feature_status": "observed",
            "target_ol_player_count": "4",
        }
    ]
    target = [
        {
            "season": "2020",
            "team_id": "home",
            "source_classification": "fbs",
            "normalized_player_identity": identity,
        }
        for identity in "ABCD"
    ]
    panel = build_core_panel(
        prior_panel=prior,
        ol_rows=target,
        college_seasons=observed_college_seasons(_history_rows()),
        program_memberships={},
    )
    assert panel[0]["core_feature_status"] == "fewer_than_five_usable_identities"
    assert panel[0]["target_ol_player_count"] == "4"
    assert panel[0][CORE_SHARED] is None


def _team(team_id: str) -> TeamSeason:
    return TeamSeason(
        season=2022,
        subdivision="fbs",
        team_id=team_id,
        team_name=team_id,
        population=2,
        lag1_z=np.asarray([0.1]),
        target_z=np.asarray([0.2]),
        target_ranks=np.asarray([1.0]),
        features={"talent_composite": 5.0},
    )


def test_common_population_and_single_existing_talent_interaction() -> None:
    previous = [
        {
            "season": "2022",
            "team_id": identity,
            "target_ol_pool_usable": True,
            "primary_four_year_window_complete": True,
            "feature_status": "observed",
        }
        for identity in ("yes", "no")
    ]
    core = [
        {
            "season": "2022",
            "team_id": identity,
            "core_feature_status": status,
            CORE_INDIVIDUAL: 2.0,
            CORE_SHARED: 1.5,
        }
        for identity, status in (
            ("yes", "observed"),
            ("no", "fewer_than_five_usable_identities"),
        )
    ]
    rows, excluded = experiment.prepare_analysis_rows(
        [_team("yes"), _team("no")], previous, core
    )
    assert len(rows) == 1
    assert excluded == {"fewer_than_five_usable_identities": 1}
    assert rows[0].features[CORE_TALENT_INTERACTION] == 7.5
    assert experiment.ARMS == (
        (experiment.REFERENCE, ()),
        (experiment.INDIVIDUAL, (CORE_INDIVIDUAL,)),
        (experiment.SHARED, (CORE_INDIVIDUAL, CORE_SHARED)),
        (
            experiment.INTERACTION,
            (CORE_INDIVIDUAL, CORE_SHARED, CORE_TALENT_INTERACTION),
        ),
    )
    keys = {(2022, "fbs", "yes"): (1.0, 0.1)}
    with pytest.raises(ValueError, match="identical team-season keys"):
        experiment.prior_experiment._loss_deltas(keys, {})


def test_generated_arms_share_the_prior_evaluation_population() -> None:
    root = Path(__file__).resolve().parents[1]
    current = root / "data/research/offensive_line_core_experiment_issue_189/results"
    previous = (
        root / "data/research/offensive_line_continuity_experiment_issue_187/results"
    )
    with (current / "evaluation_aggregate.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        aggregates = list(csv.DictReader(file))
    assert [row["model"] for row in aggregates] == [name for name, _ in experiment.ARMS]
    assert {row["evaluation_team_seasons"] for row in aggregates} == {"528"}
    assert {row["training_team_seasons"] for row in aggregates} == {"1140"}
    with (current / "evaluation_team_losses.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        current_rows = list(csv.DictReader(file))
    with (previous / "evaluation_team_losses.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        previous_rows = list(csv.DictReader(file))
    assert {(row["season"], row["team_id"]) for row in current_rows} == {
        (row["season"], row["team_id"]) for row in previous_rows
    }
    assert len(current_rows) == 528
    manifest = json.loads((current / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evaluation"]["same_keys_all_arms"] is True
