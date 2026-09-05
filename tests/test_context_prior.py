import csv
import json
import runpy
import sys
from pathlib import Path

import numpy as np

from gippyrank.context_prior import (
    PRODUCTION_SAFE_BY_SEMANTICS,
    AnnualFittedInstance,
    InferenceRow,
    ModelSpecification,
    coach_at_cutoff,
    only_approved,
)
from gippyrank.preseason import DirectRankModel, TeamSeason

ROOT = Path(__file__).parents[1]
PRESEASON = ROOT / "data/processed/preseason"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def script_values() -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        return runpy.run_path(
            str(ROOT / "scripts/build_preseason_context_prior_v1_2.py")
        )
    finally:
        sys.path.pop(0)


def tenure(
    *,
    coach_id: int = 1,
    start: int = 2010,
    end: int | None = None,
    hire: str = "2010-01-01",
) -> dict[str, object]:
    return {
        "coach": {"id": coach_id, "firstName": "A", "lastName": "Coach"},
        "team": {"school": "Example"},
        "startYear": start,
        "endYear": end,
        "hireDate": hire,
        "effectiveStart": None,
        "effectiveEnd": None,
    }


def test_specification_and_annual_fit_are_separate() -> None:
    spec = ModelSpecification(
        "context_prior", "1.2", ("history", "context"), "normal", 0.25, "empirical"
    )
    fit = AnnualFittedInstance("context_prior", "1.2", 2025, 2026, "2026-08-15")
    assert "trained_through_season" not in spec.metadata()
    assert fit.metadata()["target_season"] == 2026
    assert (
        json.loads((PRESEASON / "history/fitted_backtest.json").read_text())["kind"]
        == "evaluation/backtest"
    )
    assert (
        json.loads(
            (PRESEASON / "context/annual/2026/fitted_instance.json").read_text()
        )["target_season"]
        == 2026
    )
    report = json.loads((PRESEASON / "context/model_report.json").read_text())
    assert report["final_test"]["annual_fit"]["kind"] == "evaluation/backtest"


def test_coach_reconstruction_does_not_convert_unknown_to_no_change() -> None:
    current = coach_at_cutoff([tenure()], "Example", 2020, "2020-08-15")
    assert current.features(None)["coach_change"] is None
    undated_end = coach_at_cutoff([tenure(end=2020)], "Example", 2020, "2020-08-15")
    assert not undated_end.known_by_cutoff


def test_gate_admits_explicit_semantic_safe_feature_without_as_of() -> None:
    values = script_values()
    provenance = values["FEATURE_PROVENANCE"]
    assert (
        provenance["recruiting_class"]["production_status"]
        == PRODUCTION_SAFE_BY_SEMANTICS
    )
    assert "as_of" not in provenance["recruiting_class"]
    assert only_approved({"recruiting_class_points": 1.0}, {"recruiting_class_points"})
    try:
        only_approved({"transfer_count": 1.0}, {"recruiting_class_points"})
    except ValueError:
        pass
    else:
        raise AssertionError("unapproved transfer feature reached the production gate")


def test_builder_is_annual_and_context_attachment_preserves_h_features() -> None:
    values = script_values()
    assert values["cutoff_for"](2027) == "2027-08-15"
    original = {"lag2_z_mean": 0.2, "lag3_z_mean": 0.1, "long_run_z_mean": 0.3}
    row = TeamSeason(
        2020,
        "fbs",
        "example",
        "Example",
        10,
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([5]),
        original,
    )
    attached, _ = values["attach_context"]([row], {}, {"Example": [tenure()]})
    for name, value in original.items():
        assert attached[0].features[name] == value
    assert "coach_tenure_seasons" in attached[0].features


def test_development_report_is_same_population_and_algebraically_consistent() -> None:
    report = json.loads((PRESEASON / "context/model_report.json").read_text())
    c0 = report["development"]["C0_history_only"]["comparison"]
    assert c0["reference"]["n_team_seasons"] == c0["candidate"]["n_team_seasons"]
    assert c0["delta_c_minus_h"]["nll"] == 0.0
    for candidate in report["development"]["candidates"].values():
        result = candidate["comparison"]
        assert (
            result["reference"]["n_team_seasons"]
            == result["candidate"]["n_team_seasons"]
        )
        assert np.isclose(
            result["delta_c_minus_h"]["nll"],
            result["candidate"]["nll"] - result["reference"]["nll"],
        )
        assert (
            len(result["same_population_keys"]) == result["reference"]["n_team_seasons"]
        )
    markdown = (PRESEASON / "context/context_prior_report.md").read_text()
    assert "| C0_history_only | exact H |" in markdown


def test_model_selection_is_frozen_before_the_untouched_test() -> None:
    report = json.loads((PRESEASON / "context/model_report.json").read_text())
    selection = report["selection"]
    assert selection["development_train"] == [2004, 2017]
    assert selection["development_validation"] == [2018, 2021]
    assert selection["untouched_test"] == [2022, 2023, 2024, 2025]
    assert "2022" not in selection["rule"]


def test_historical_h_artifact_is_byte_stable() -> None:
    baseline = {
        (x["season"], x["subdivision"], x["team_id"]): x["pmf"]
        for x in read_csv(PRESEASON / "rank_prior_predictions.csv")
        if x["model"] == "V1_1_long_run_baseline"
        and x["subdivision"] == "fbs"
        and int(x["season"]) in {2022, 2023, 2024, 2025}
    }
    history = {
        (x["season"], x["subdivision"], x["team_id"]): x["pmf"]
        for x in read_csv(PRESEASON / "history/predictions.csv")
    }
    assert history == baseline
    assert {
        row["model_family"] for row in read_csv(PRESEASON / "history/predictions.csv")
    } == {"history_prior"}


def test_future_season_rows_have_no_target_and_both_families_emit_normalized_pmfs() -> (
    None
):
    report = json.loads((PRESEASON / "context/model_report.json").read_text())
    assert report["annual_inference"]["no_2026_game_outcomes_used"] is True
    assert report["annual_inference"]["n_fbs"] == 138
    for family in ("history", "context"):
        rows = read_csv(PRESEASON / family / "annual/2026/predictions.csv")
        assert len(rows) == 138
        assert len({row["team_id"] for row in rows}) == 138
        for row in rows:
            pmf = np.asarray(json.loads(row["pmf"]), dtype=float)
            assert np.isclose(pmf.sum(), 1.0)
            assert np.all(pmf >= 0)


def test_synthetic_future_inference_needs_no_target_outcome() -> None:
    train = [
        TeamSeason(
            2020,
            "fbs",
            "a",
            "A",
            2,
            np.asarray([0.0]),
            np.asarray([0.1]),
            np.asarray([1]),
            {"history": 0.0, "context": 0.2},
        )
    ]
    h_model = DirectRankModel.fit(train, ["history"], optimizer_options={"maxiter": 30})
    c_model = DirectRankModel.fit(
        train,
        ["history", "context"],
        location_feature_names=["history", "context"],
        scale_feature_names=["history"],
        optimizer_options={"maxiter": 30},
    )
    future = InferenceRow(
        2021, "fbs", "a", "A", 2, (0.0,), (), {"history": 0.0, "context": 0.3}
    )
    future.require_no_target()
    assert not hasattr(future, "target_ranks")
    for model in (h_model, c_model):
        pmf = model.pmf(future.features, np.asarray(future.lag1_z), future.population)
        assert np.isclose(pmf.sum(), 1.0)


def test_missing_context_is_not_population_exclusion() -> None:
    rows = read_csv(PRESEASON / "context/predictions.csv")
    assert len(rows) == 534
    assert len({(row["season"], row["team_id"]) for row in rows}) == 534
    # Regular rows receive explicit preprocessor missingness indicators; cold
    # starts are the only exact-H fallback and remain included.
    assert any(row["context_applied"] == "False" for row in rows)


def test_context_artifact_is_a_distinct_sibling_family() -> None:
    rows = read_csv(PRESEASON / "context/predictions.csv")
    assert {row["model_family"] for row in rows} == {"context_prior"}
    assert {row["spec_version"] for row in rows} == {"1.2"}
