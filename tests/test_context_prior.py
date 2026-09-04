import csv
import json
import runpy
from pathlib import Path

import numpy as np

from gippyrank.context_prior import ModelIdentity, coach_at_cutoff, only_approved

ROOT = Path(__file__).parents[1]
PRESEASON = ROOT / "data/processed/preseason"


def tenure(
    *,
    coach_id: int = 1,
    start: int = 2010,
    end: int | None = None,
    hire_date: str | None = "2010-01-01",
) -> dict[str, object]:
    return {
        "coach": {"id": coach_id, "firstName": "A", "lastName": "Coach"},
        "team": {"school": "Example"},
        "startYear": start,
        "endYear": end,
        "hireDate": hire_date,
        "effectiveStart": None,
        "effectiveEnd": None,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_model_identity_distinguishes_family_specification_and_fit() -> None:
    history = ModelIdentity("history_prior", "1.1", 2025, 2026)
    context = ModelIdentity("context_prior", "1.2", 2025, 2026, "2026-08-15")
    assert history.metadata() == {
        "model_family": "history_prior",
        "spec_version": "1.1",
        "trained_through_season": 2025,
        "target_season": 2026,
        "context_as_of": None,
    }
    assert context.metadata()["context_as_of"] == "2026-08-15"


def test_coach_context_requires_dated_target_start_and_nonambiguous_end() -> None:
    known = coach_at_cutoff([tenure(start=2010, end=2022)], "Example", 2020, "2020-08-15")
    assert known.known_by_cutoff is True
    assert known.tenure_seasons == 11

    late = coach_at_cutoff(
        [tenure(start=2020, end=None, hire_date="2020-09-01")],
        "Example",
        2020,
        "2020-08-15",
    )
    assert late.known_by_cutoff is False
    assert late.unavailable_reason == "target_season_start_not_dated_by_cutoff"

    undated_end = coach_at_cutoff(
        [tenure(start=2010, end=2020)], "Example", 2020, "2020-08-15"
    )
    assert undated_end.known_by_cutoff is False
    assert undated_end.unavailable_reason == "target_season_end_is_undated"


def test_unknown_coach_history_never_becomes_no_change() -> None:
    current = coach_at_cutoff([tenure(coach_id=2)], "Example", 2020, "2020-08-15")
    assert current.features(None) == {
        "coach_change": None,
        "coach_tenure_seasons": 11.0,
    }


def test_production_gate_rejects_unapproved_context() -> None:
    assert only_approved({"coach_tenure_seasons": 3.0}, {"coach_tenure_seasons"}) == {
        "coach_tenure_seasons": 3.0
    }
    try:
        only_approved({"returning_pct_ppa": 0.5}, {"coach_tenure_seasons"})
    except ValueError as error:
        assert "unapproved" in str(error)
    else:
        raise AssertionError("unapproved feature reached the production gate")


def test_context_builder_is_not_hard_coded_to_2026() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        values = runpy.run_path(
            str(ROOT / "scripts/build_preseason_context_prior_v1_2.py")
        )
    finally:
        sys.path.pop(0)
    assert values["cutoff_for"](2027) == "2027-08-15"
    assert callable(values["build_history_prior"])
    assert callable(values["build_context_prior"])


def test_context_attachment_does_not_mutate_historical_features() -> None:
    import sys

    from gippyrank.preseason import TeamSeason

    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        values = runpy.run_path(
            str(ROOT / "scripts/build_preseason_context_prior_v1_2.py")
        )
    finally:
        sys.path.pop(0)
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
    attached, _ = values["with_coach_context"]([row], {"Example": [tenure()]})
    for name, value in original.items():
        assert attached[0].features[name] == value
    assert "coach_tenure_seasons" in attached[0].features


def test_family_artifacts_are_distinct_and_h_is_byte_stable() -> None:
    frozen = {
        (row["season"], row["subdivision"], row["team_id"]): row["pmf"]
        for row in read_csv(PRESEASON / "rank_prior_predictions.csv")
        if row["model"] == "V1_1_long_run_baseline"
        and row["subdivision"] == "fbs"
        and int(row["season"]) in {2022, 2023, 2024, 2025}
    }
    history = read_csv(PRESEASON / "history/predictions.csv")
    context = read_csv(PRESEASON / "context/predictions.csv")
    history_pmfs = {
        (row["season"], row["subdivision"], row["team_id"]): row["pmf"]
        for row in history
    }
    context_keys = {(row["season"], row["subdivision"], row["team_id"]) for row in context}
    assert history_pmfs == frozen
    assert context_keys == set(frozen)
    assert {row["model_family"] for row in history} == {"history_prior"}
    assert {row["model_family"] for row in context} == {"context_prior"}


def test_context_has_one_normalized_pmf_per_fbs_target_and_safe_fallbacks() -> None:
    history = {
        (row["season"], row["subdivision"], row["team_id"]): row
        for row in read_csv(PRESEASON / "history/predictions.csv")
    }
    context = read_csv(PRESEASON / "context/predictions.csv")
    assert len(context) == 534
    for row in context:
        pmf = np.asarray(json.loads(row["pmf"]), dtype=float)
        assert np.isclose(pmf.sum(), 1.0)
        assert np.all(pmf >= 0)
        key = (row["season"], row["subdivision"], row["team_id"])
        if row["context_applied"] != "True":
            assert row["fallback_reason"]
            assert row["pmf"] == history[key]["pmf"]


def test_selected_c_spec_admits_no_timing_uncertain_feature() -> None:
    report = json.loads((PRESEASON / "context/model_report.json").read_text())
    spec = json.loads((PRESEASON / "context/model_spec.json").read_text())
    provenance = report["provenance"]
    assert report["selection"]["untouched_test"] == [2022, 2023, 2024, 2025]
    assert spec["model_family"] == "context_prior"
    assert spec["trained_through_season"] == 2021
    assert spec["context_as_of"] == "2022-08-15"
    assert all(
        provenance[name]["production_status"] in {"production-safe", "reconstructable-safe"}
        for name in spec["approved_context_features"]
    )
    assert provenance["recruiting_class"]["production_status"] == "exploratory-timing-uncertain"
    assert provenance["transfers"]["production_status"] == "rejected"
