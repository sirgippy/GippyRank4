import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

from gippyrank.context_ablation import (
    aggregate_difference,
    assert_same_population,
    interaction_is_useful,
    paired_loss_differences,
    restrict_observed,
    standardized_interaction_rows,
    training_only_impute_rows,
    training_rows,
)
from gippyrank.preseason import TeamSeason

ROOT = Path(__file__).parents[1]

pytestmark = pytest.mark.research


def row(season: int, team: str, left: float | None, right: float | None) -> TeamSeason:
    return TeamSeason(
        season,
        "fbs",
        team,
        team,
        2,
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([1]),
        {"left": left, "right": right},
    )


def test_raw_observed_coverage_precedes_imputation() -> None:
    rows = [row(2020, "a", 1.0, 2.0), row(2020, "b", None, 3.0)]
    observed = restrict_observed(rows, ["left", "right"])
    assert [item.team_id for item in observed] == ["a"]


def test_rolling_training_excludes_target_and_future_outcomes() -> None:
    plan = training_rows([row(2020, "a", 1, 1), row(2021, "b", 1, 1)], 2021)
    assert [item.season for item in plan] == [2020]


def test_pairing_rejects_target_population_drift() -> None:
    with pytest.raises(ValueError, match="identical keys"):
        assert_same_population([row(2020, "a", 1, 1)], [row(2020, "b", 1, 1)])


def test_interaction_is_standardized_with_training_rows_only() -> None:
    train = [row(2020, "a", 1.0, 10.0), row(2021, "b", 3.0, 30.0)]
    target = [row(2022, "c", 5.0, 50.0)]
    enriched_train, enriched_target, name = standardized_interaction_rows(
        train, target, "left", "right"
    )
    assert name == "interaction__left__right"
    # Both target standard scores are +3 under training mean/std, so product is 9.
    assert enriched_target[0].features[name] == pytest.approx(9.0)
    assert enriched_train[0].features[name] == pytest.approx(1.0)


def test_training_only_imputation_does_not_use_target_median() -> None:
    train = [row(2020, "a", 1.0, 2.0), row(2021, "b", None, 4.0)]
    target = [row(2022, "c", 1000.0, 6.0)]
    filled_train, filled_target = training_only_impute_rows(train, target, ["left"])
    assert filled_train[1].features["left"] == 1.0
    assert filled_target[0].features["left"] == 1000.0


def test_stored_per_team_difference_aggregation_is_exact() -> None:
    values = [-0.2, 0.1, 0.4]
    assert aggregate_difference(values) == pytest.approx(sum(values) / len(values))


def test_interaction_parent_pairing_and_difference_semantics() -> None:
    parent = {(2025, "fbs", "a"): (1.0, 0.1)}
    interaction = {(2025, "fbs", "a"): (1.2, 0.2)}
    difference = paired_loss_differences(parent, interaction)
    assert difference[(2025, "fbs", "a")] == pytest.approx((0.2, 0.1))
    with pytest.raises(ValueError, match="identical target keys"):
        paired_loss_differences(parent, {(2025, "fbs", "b"): (1.2, 0.2)})


def test_interaction_beating_h_but_losing_parent_is_not_useful() -> None:
    h_nll, parent_nll, interaction_nll = 3.0, 1.0, 1.2
    assert interaction_nll - h_nll < 0
    assert interaction_nll - parent_nll > 0
    assert not interaction_is_useful([interaction_nll - parent_nll])


def test_report_interaction_claims_use_parent_summary_fields() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from investigate_context_ablation import report_interaction_lines

        line = report_interaction_lines(
            [
                {
                    "interaction": "Talent × total",
                    "parent": "H + Talent + total",
                    "mean_interaction_minus_parent_nll": 0.0014,
                    "wins": 2,
                    "losses": 6,
                }
            ]
        )[0]
    finally:
        sys.path.pop(0)
    assert "versus H + Talent + total +0.0014" in line


def test_production_specs_and_frozen_2026_pmfs_remain_guarded() -> None:
    root = Path(__file__).parents[1]
    source = (root / "scripts/investigate_context_ablation.py").read_text()
    assert (
        "data/processed/preseason/history/annual/2026/predictions.csv"
        not in source.split("write_csv")[0]
    )
    expected = {
        "history/annual/2026/predictions.csv": "0b3454a09288019e17739869c42aed3123fdda2163694f52f63bca65baf37f90",
        "context/annual/2026/predictions.csv": "641182890ec88ea8bc6150cc97047ddc688d0c680486d9fcc63a58d7bfae9132",
    }
    for relative, digest in expected.items():
        assert (
            hashlib.sha256(
                (root / "data/processed/preseason" / relative).read_bytes()
            ).hexdigest()
            == digest
        )


def test_stored_ablation_metrics_recompute_from_paired_team_losses() -> None:
    root = Path(__file__).parents[1]
    artifact = root / "data/processed/context_ablation"
    annual = list(csv.DictReader((artifact / "annual_ablation_metrics.csv").open()))
    losses = list(csv.DictReader((artifact / "per_team_losses.csv").open()))
    assert annual and losses
    assert all(row["same_population_keys"] == "True" for row in annual)
    for row in annual:
        paired = [
            float(item["candidate_minus_h_nll"])
            for item in losses
            if item["population"] == row["population"]
            and item["candidate"] == row["candidate"]
            and item["season"] == row["target_season"]
        ]
        assert paired
        assert float(row["delta_nll"]) == pytest.approx(np.mean(paired))


def test_2025_decomposition_is_the_full_c_paired_aggregate() -> None:
    root = Path(__file__).parents[1]
    artifact = root / "data/processed/context_ablation"
    decomposition = list(csv.DictReader((artifact / "decomposition_2025.csv").open()))
    annual = list(csv.DictReader((artifact / "annual_ablation_metrics.csv").open()))
    target = next(
        row
        for row in annual
        if row["population"] == "same_all_context"
        and row["candidate"] == "same_full_c"
        and row["target_season"] == "2025"
    )
    assert aggregate_difference(
        [float(row["candidate_minus_h_nll"]) for row in decomposition]
    ) == pytest.approx(float(target["delta_nll"]))


def test_missingness_control_records_raw_absence_separately() -> None:
    root = Path(__file__).parents[1]
    rows = list(
        csv.DictReader(
            (root / "data/processed/context_ablation/missingness_results.csv").open()
        )
    )
    for family in ("recruiting", "talent", "returning"):
        forms = {row["formulation"] for row in rows if row["family"] == family}
        assert forms == {"with_indicators", "without_indicators"}


def test_generated_interaction_artifacts_remain_parent_relative() -> None:
    artifact = ROOT / "data/processed/context_ablation"
    annual = list(csv.DictReader((artifact / "annual_ablation_metrics.csv").open()))
    interactions = list(csv.DictReader((artifact / "interaction_results.csv").open()))
    summaries = list(csv.DictReader((artifact / "interaction_summary.csv").open()))
    assert interactions and summaries
    assert all(row["same_population_keys"] == "True" for row in interactions)
    assert all(row["same_training_keys"] == "True" for row in interactions)
    assert all("interaction_minus_parent_nll" in row for row in interactions)
    # This guards against restoring the generic candidate-vs-H CSV writer.
    assert "candidate" not in interactions[0]
    for summary in summaries:
        rows = [
            float(row["interaction_minus_parent_nll"])
            for row in interactions
            if row["interaction"] == summary["interaction"]
        ]
        assert float(summary["mean_interaction_minus_parent_nll"]) == pytest.approx(
            np.mean(rows)
        )
    report = (artifact / "report.md").read_text()
    assert "## Interaction parent comparisons" in report
    assert "mean ΔNLL versus H + talent_composite" in report
    assert any("interaction_talent_total" in row["candidate"] for row in annual)
