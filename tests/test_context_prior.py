import csv
import hashlib
import inspect
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
    history_fit = json.loads(
        (PRESEASON / "history/annual/2026/fitted_instance.json").read_text()
    )
    assert history_fit["artifact_kind"] == "frozen_preseason_forecast"
    assert history_fit["target_season"] == 2026
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


def test_historical_h_artifact_is_byte_stable() -> None:
    expected = {
        "history/annual/2026/predictions.csv": "0b3454a09288019e17739869c42aed3123fdda2163694f52f63bca65baf37f90",
        "context/annual/2026/predictions.csv": "641182890ec88ea8bc6150cc97047ddc688d0c680486d9fcc63a58d7bfae9132",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256(
            (PRESEASON / relative).read_bytes()
        ).hexdigest() == digest
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
    assert report["annual_inference"]["target_season_outcomes_forbidden"] is True
    assert report["annual_inference"]["n_fbs"] == 138
    for family in ("history", "context"):
        rows = read_csv(PRESEASON / family / "annual/2026/predictions.csv")
        assert len(rows) == 138
        assert len({row["team_id"] for row in rows}) == 138
        for row in rows:
            pmf = np.asarray(json.loads(row["pmf"]), dtype=float)
            assert np.isclose(pmf.sum(), 1.0)
            assert np.all(pmf >= 0)
    context_fit = report["annual_inference"]["context_fit"]
    assert context_fit["context_effective_cutoff"] == "2026-08-15"
    assert context_fit["context_snapshot_mode"] == "retrospective_reconstruction"
    assert context_fit["pmf_count"] == 138
    history_fit = report["annual_inference"]["history_fit"]
    assert history_fit["artifact_kind"] == "frozen_preseason_forecast"
    assert history_fit["pmf_count"] == 138


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


def _rank_row(season: int, subdivision: str, team_id: str, rank: int) -> dict[str, str]:
    return {
        "season": str(season),
        "subdivision": subdivision,
        "team_id": team_id,
        "team_name": team_id,
        "team_population": "2",
        "rank_observations": json.dumps([rank]),
    }


def test_row_builder_ceiling_is_explicit_and_allows_completed_2026(monkeypatch) -> None:
    values = script_values()
    legacy = values["v1"]
    outcomes = [
        _rank_row(2025, "fbs", "a", 1),
        _rank_row(2026, "fbs", "a", 2),
        _rank_row(2027, "fbs", "a", 1),
    ]
    features = [
        {"season": str(year), "subdivision": "fbs", "team_id": "a", "team_name": "a"}
        for year in (2025, 2026, 2027)
    ]

    def fake_read_csv(path: Path) -> list[dict[str, str]]:
        return features if path.name == "team_season_features.csv" else outcomes

    monkeypatch.setattr(legacy, "read_csv", fake_read_csv)
    rows, _, _ = legacy.load_rows(max_season=2026)
    assert {row.season for row in rows} == {2026}
    assert all(row.season != 2027 for row in rows)


def test_synthetic_2027_rows_use_2026_history_and_exclude_2027_outcome(monkeypatch) -> None:
    values = script_values()
    outcomes = [
        _rank_row(2024, "fbs", "a", 1),
        _rank_row(2025, "fbs", "a", 2),
        _rank_row(2026, "fbs", "a", 1),
        _rank_row(2027, "fbs", "a", 2),
    ]
    index = {(2027, "fbs", "a"): {"season": "2027", "subdivision": "fbs", "team_id": "a", "team_name": "a"}}
    monkeypatch.setitem(values["inference_rows"].__globals__, "read_csv", lambda _: outcomes)
    future = values["inference_rows"](2027, 2026, index, {})
    assert len(future) == 1
    assert future[0].season == 2027
    assert future[0].lag1_z is not None
    assert not hasattr(future[0], "target_ranks")
    assert np.isclose(future[0].lag1_z[0], -np.log(3))


def test_synthetic_2027_fcs_promotion_uses_2026_fcs_history(monkeypatch) -> None:
    values = script_values()
    legacy = values["v1"]
    cold = [
        legacy.ColdStartSeason(2026, "fbs", "old", "old", 2, np.asarray([0.0]), np.asarray([1]), np.asarray([0.0]), "fcs_to_fbs_transition"),
        legacy.ColdStartSeason(2026, "fbs", "generic", "generic", 2, np.asarray([0.0]), np.asarray([1]), None, "no_prior_rank_distribution"),
        legacy.ColdStartSeason(2027, "fbs", "later", "later", 2, np.asarray([0.0]), np.asarray([1]), None, "no_prior_rank_distribution"),
    ]
    promotion, generic = values["annual_cold_start_models"](cold, trained_through_season=2026)
    assert generic.n_team_seasons == 1
    monkeypatch.setitem(
        values["future_predictions"].__globals__,
        "read_csv",
        lambda _: [_rank_row(2026, "fcs", "promo", 1)],
    )
    future = InferenceRow(2027, "fbs", "promo", "promo", 2, None, (), {}, "fcs_to_fbs_transition")
    history, context = values["future_predictions"](future=[future], h_model=None, c_model=None, trained_through_season=2026, promotion_model=promotion, generic_prior=generic)
    assert history[0]["prior_method"] == "learned_fcs_to_fbs_transition"
    assert history[0]["trained_through_season"] == 2026
    assert context[0]["fallback_reason"] == "history_cold_start"


def test_annual_helpers_have_no_fixed_2025_or_2026_dependency() -> None:
    values = script_values()
    for name in ("inference_rows", "annual_cold_start_models", "future_predictions"):
        source = inspect.getsource(values[name])
        assert "2025" not in source
        assert "2026" not in source
