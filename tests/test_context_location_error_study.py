"""Issue 154 study invariants against the frozen issue 153 diagnostic artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study_context_location_errors as study


@pytest.fixture(scope="module")
def source() -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(
        study.SOURCE / "team_seasons.csv", keep_default_na=False, na_values=[""]
    )
    features = pd.read_csv(study.SOURCE / "feature_inventory.csv").feature_name.tolist()
    return frame, features


def test_exact_input_population_and_orientation(
    source: tuple[pd.DataFrame, list[str]],
) -> None:
    frame, _ = source
    study.validate_input(frame)
    assert frame.groupby("season").size().to_dict() == {
        2022: 131,
        2023: 133,
        2024: 134,
        2025: 136,
    }
    assert frame.component_status.value_counts().to_dict() == {
        "fitted": 528,
        "cold_start_fallback": 6,
    }
    changed = frame.copy()
    changed.loc[0, "context_minus_history_final_nll"] *= -1
    with pytest.raises(ValueError, match="orientation/parity"):
        study.validate_input(changed)


def test_within_season_quantiles_and_disagreement_groups(
    source: tuple[pd.DataFrame, list[str]],
) -> None:
    tiny = pd.DataFrame(
        {
            "season": [2022] * 4 + [2023] * 4,
            "team_id": [4, 2, 3, 1] * 2,
            "value": [10, 10, 10, 10, 100, 200, 300, 400],
        }
    )
    assert study.quartiles(tiny, "value").tolist() == [4, 2, 3, 1, 1, 2, 3, 4]
    frame = study.assign_groups(source[0])
    for _, part in frame.groupby("season"):
        assert set(part.disagreement_quartile) == {1, 2, 3, 4}
        assert (
            part.disagreement_quartile.value_counts().max()
            - part.disagreement_quartile.value_counts().min()
            <= 1
        )
        assert (part.loc[part.expensive_miss, "error_quartile"] == 4).all()
        assert (part.loc[part.expensive_miss, "nll_gap_quartile"] == 4).all()
        assert (
            part.loc[part.helpful_disagreement, "context_minus_history_abs_rank_error"]
            < 0
        ).all()
        assert (
            part.loc[part.harmful_disagreement, "context_minus_history_abs_rank_error"]
            > 0
        ).all()


def test_contributions_and_fallback_exclusion(
    source: tuple[pd.DataFrame, list[str]],
) -> None:
    frame, features = source
    grouped = study.assign_groups(frame)
    contributions = study.contribution_diagnostics(grouped, features)
    assert len(contributions) == 528
    assert set(zip(contributions.season, contributions.team_id, strict=True)) == set(
        zip(
            frame.loc[frame.component_status == "fitted", "season"],
            frame.loc[frame.component_status == "fitted", "team_id"],
            strict=True,
        )
    )
    source_fitted = frame.loc[frame.component_status == "fitted"].set_index(
        ["season", "team_id"]
    )
    for row in contributions.itertuples():
        original = source_fitted.loc[(row.season, row.team_id)]
        expected = sum(
            original[f"feature_{feature}_contribution"]
            + original[f"feature_{feature}_missing_contribution"]
            for feature in features
        )
        assert (
            row.history_derived_subtotal + row.context_only_subtotal
            == pytest.approx(expected + original.lag1_contribution)
        )
        assert (
            row.context_only_positive_sum + row.context_only_negative_sum
            == pytest.approx(row.context_only_subtotal)
        )
        assert row.largest_absolute_share <= 1
        assert row.positive_contribution_sum >= 0
        assert row.negative_contribution_sum <= 0
        values = [
            original[f"feature_{feature}_contribution"]
            + original[f"feature_{feature}_missing_contribution"]
            for feature in features
        ]
        dominant = max(values, key=abs)
        assert row.opposing_substantial_count == sum(
            value * dominant < 0 and abs(value) >= 0.25 * abs(dominant)
            for value in values
        )


def test_gap_reconciliation_and_population_separation(
    source: tuple[pd.DataFrame, list[str]],
) -> None:
    frame = study.assign_groups(source[0])
    result = study.concentration(frame)
    for period, part in study.periods(frame):
        reported = next(
            row for row in result if row["period"] == period and row["fraction"] == 1
        )
        assert reported["total_gap_sum"] / len(part) == pytest.approx(
            part.context_final_nll.mean() - part.history_final_nll.mean()
        )
        assert (
            part.season.eq(2022).all()
            if period == "2022"
            else part.season.ne(2022).all()
        )
    assert (
        next(
            row
            for row in result
            if row["period"] == "2023-2025" and row["fraction"] == 0.1
        )["share_of_net_gap"]
        > 0.7
    )
    assert frame.loc[
        frame.season.isin(study.PRIMARY), "primary_mean_gap_contribution"
    ].sum() == pytest.approx(
        frame.loc[frame.season.isin(study.PRIMARY), "context_final_nll"].mean()
        - frame.loc[frame.season.isin(study.PRIMARY), "history_final_nll"].mean()
    )


def test_support_and_transfer_grouping(source: tuple[pd.DataFrame, list[str]]) -> None:
    frame, features = source
    grouped = study.assign_groups(frame)
    assert grouped.loc[
        grouped.component_status == "fitted", "historical_repair_state"
    ].value_counts().to_dict() == {
        "none": 513,
        "legitimate_zero_restoration": 13,
        "ambiguous_usage_join_removed": 1,
        "name_normalization_join_added": 1,
    }
    contribution = study.contribution_diagnostics(grouped, features)
    diagnostics = study.grouped_diagnostics(grouped, contribution)
    for period, expected in (
        ("2022", 130),
        ("2023", 131),
        ("2024", 133),
        ("2025", 134),
        ("2023-2025", 398),
    ):
        part = diagnostics.loc[
            (diagnostics.period == period)
            & (diagnostics.family == "nearest_support_quartile")
        ]
        assert int(part.n.sum()) == expected
        assert set(part.bin) == {"1", "2", "3", "4"}
        transfer = diagnostics.loc[
            (diagnostics.period == period) & (diagnostics.family == "db_coverage")
        ]
        assert int(transfer.n.sum()) == expected
    assert (
        diagnostics.loc[
            (diagnostics.period == "2022")
            & (diagnostics.family == "historical_transfer_input_difference")
            & (diagnostics.bin == "legitimate_zero_restoration"),
            "n",
        ].iloc[0]
        == 8
    )


def test_net_context_only_subtotal_finding_is_pinned(
    source: tuple[pd.DataFrame, list[str]],
) -> None:
    frame, features = source
    frame = study.assign_groups(frame)
    contributions = study.contribution_diagnostics(frame, features)
    committed = pd.read_csv(study.OUT / "grouped_diagnostics.csv", dtype={"bin": str})
    reproduced = study.grouped_diagnostics(frame, contributions)

    def quartile(table: pd.DataFrame, period: str, bin_name: str) -> pd.Series:
        match = table.loc[
            (table.family == "context_only_subtotal_quartile")
            & (table.period == period)
            & (table.bin == bin_name)
        ]
        assert len(match) == 1
        return match.iloc[0]

    top = quartile(committed, "2023-2025", "4")
    bottom = quartile(committed, "2023-2025", "1")
    control_top = quartile(committed, "2022", "4")
    assert (top.n, top.expensive_miss_n) == (98, 19)
    assert (bottom.n, bottom.expensive_miss_n) == (101, 5)
    assert top.final_nll_gap_mean == pytest.approx(0.168340168249, abs=1e-10)
    assert bottom.final_nll_gap_mean == pytest.approx(0.0503487045285, abs=1e-10)
    assert top.final_nll_gap_mean > bottom.final_nll_gap_mean
    assert control_top.final_nll_gap_mean == pytest.approx(-0.0406806208577, abs=1e-10)
    by_season = {
        "2023": ((33, 1, 0.0132115786879), (32, 6, 0.231670312939)),
        "2024": ((34, 3, 0.115845049941), (33, 8, 0.106267873964)),
        "2025": ((34, 1, 0.0208972165496), (33, 5, 0.169001413138)),
    }
    for period, (expected_q1, expected_q4) in by_season.items():
        q1, q4 = quartile(committed, period, "1"), quartile(committed, period, "4")
        for observed, (n, misses, gap) in ((q1, expected_q1), (q4, expected_q4)):
            assert (observed.n, observed.expensive_miss_n) == (n, misses)
            assert observed.expensive_miss_rate == pytest.approx(misses / n, abs=1e-10)
            assert observed.final_nll_gap_mean == pytest.approx(gap, abs=1e-10)
        assert q4.expensive_miss_rate > q1.expensive_miss_rate
        assert (q4.final_nll_gap_mean > q1.final_nll_gap_mean) == (period != "2024")

    period_bins = [("2023-2025", "1"), ("2023-2025", "4"), ("2022", "4")]
    period_bins.extend(
        (period, bin_name) for period in by_season for bin_name in ("1", "4")
    )
    for period, bin_name in period_bins:
        actual = quartile(reproduced, period, bin_name)
        saved = quartile(committed, period, bin_name)
        assert (actual.n, actual.expensive_miss_n) == (saved.n, saved.expensive_miss_n)
        assert actual.final_nll_gap_mean == pytest.approx(
            saved.final_nll_gap_mean, abs=1e-10
        )

    committed_contributions = pd.read_csv(study.OUT / "contribution_diagnostics.csv")
    expected_correlations = {
        2023: 0.323814530719,
        2024: -0.014333952588,
        2025: 0.278771619449,
    }
    for season, expected_correlation in expected_correlations.items():
        period = committed_contributions.loc[committed_contributions.season == season]
        assert period.context_only_subtotal.corr(
            period.context_minus_history_final_nll
        ) == pytest.approx(expected_correlation, abs=1e-6)

    summary = json.loads((study.OUT / "summary.json").read_text())
    assert (
        summary["contribution_diagnostic_definitions"][
            "recommended_intervention_metric"
        ]
        == "context_only_subtotal"
    )
    primary = contributions.loc[contributions.season.isin(study.PRIMARY)].copy()
    primary["net_q"] = study.quartiles(primary, "context_only_subtotal")
    primary["positive_sum_q"] = study.quartiles(primary, "context_only_positive_sum")
    assert ((primary.net_q == 4) & (primary.positive_sum_q == 4)).sum() == 74


def test_non_zero_restoration_cases_use_generic_repair_state(
    source: tuple[pd.DataFrame, list[str]],
) -> None:
    frame, features = source
    frame = study.assign_groups(frame)
    expected = frame.loc[
        (frame.component_status == "fitted")
        & frame.historical_repair_state.isin(study.NON_ZERO_RESTORATION_REPAIR_STATES)
    ]
    assert {(row.season, row.team) for row in expected.itertuples()} == {
        (2023, "North Texas"),
        (2023, "Coastal Carolina"),
    }
    renamed = frame.copy()
    renamed.loc[expected.index, "team"] = ["Renamed repair A", "Renamed repair B"]
    contributions = study.contribution_diagnostics(frame, features)
    cases = study.case_studies(renamed, contributions, features)
    selected = cases.loc[
        cases.selection_reason.str.split(";").apply(
            lambda reasons: "transfer_repair" in reasons
        )
    ]
    assert set(zip(selected.season, selected.team_id, strict=True)) == set(
        zip(expected.season, expected.team_id, strict=True)
    )
    assert set(selected.team) == {"Renamed repair A", "Renamed repair B"}
    assert (
        set(selected.historical_repair_state)
        == study.NON_ZERO_RESTORATION_REPAIR_STATES
    )


def test_machine_outputs_reproduce_committed_artifact_bytes(
    source: tuple[pd.DataFrame, list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del source
    committed_output = study.OUT
    destination = tmp_path / "generated"
    monkeypatch.setattr(study, "OUT", destination)
    study.main()

    generated = {path.name: path.read_bytes() for path in destination.iterdir()}
    committed = {
        path.name: path.read_bytes() for path in committed_output.iterdir()
    }
    # The committed provenance pins the analysis code, four source files, and
    # five generated data artifacts; parity checks this run against that
    # independent canonical materialization, including provenance itself.
    assert generated == committed
    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["source_hashes"][
        str((study.SOURCE / "team_seasons.csv").relative_to(ROOT))
    ] == study.sha256(study.SOURCE / "team_seasons.csv")
