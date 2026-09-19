import sys
from pathlib import Path

import numpy as np

from gippyrank.preseason import DirectRankModel, TeamSeason

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import investigate_defensive_transfer_stability as study


def _row(
    season: int, team_id: str, *, usage: float | None, available: float
) -> TeamSeason:
    return TeamSeason(
        season,
        "fbs",
        team_id,
        team_id,
        2,
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([1]),
        {
            study.RETURNING_PPA: 0.25,
            study.INCOMING_USAGE: usage,
            study.EXPERIENCE: 0.5 if available else 0.0,
            study.EXPERIENCE_AVAILABLE: available,
        },
    )


def test_candidate_set_is_exactly_r0_r1_r2() -> None:
    candidates = study.candidate_definitions()
    assert [candidate.label for candidate in candidates] == ["R0", "R1", "R2"]
    assert candidates[0].features == study.BASE_CONTEXT_FEATURES
    assert candidates[1].features == study.d5_features()
    assert candidates[2].features == (
        *study.d5_features(),
        study.EXPERIENCE,
        study.EXPERIENCE_AVAILABLE,
    )


def test_training_coverage_excludes_target_and_separates_sources() -> None:
    rows = [
        _row(2021, "a", usage=1.0, available=1.0),
        _row(2021, "b", usage=None, available=0.0),
        _row(2022, "c", usage=99.0, available=1.0),
    ]
    result = study.training_coverage(rows, {2021, 2022}, target_season=2022)
    assert result["n_training_team_seasons"] == 2
    assert result["n_transfer_covered_training_seasons"] == 1
    assert result["n_training_rows_with_observed_incoming_offensive_usage"] == 1
    assert result["n_training_rows_with_defensive_experience_source_available"] == 1
    assert result["n_training_rows_unresolved_for_defensive_experience"] == 1


def test_coefficient_index_includes_explicit_missingness_columns() -> None:
    model = DirectRankModel(
        feature_names=["history", study.EXPERIENCE],
        preprocessor=None,  # type: ignore[arg-type]
        beta=np.zeros(6),
        gamma=np.zeros(5),
        lag_count=1,
    )
    assert study.coefficient_index(model, study.EXPERIENCE) == (3, 5)


def test_coefficient_rows_include_value_and_availability_paths() -> None:
    candidate = study.candidate_definitions()[2]
    model = DirectRankModel(
        feature_names=list(study.COEFFICIENT_FEATURES),
        preprocessor=None,  # type: ignore[arg-type]
        beta=np.zeros(9),
        gamma=np.zeros(5),
        lag_count=0,
    )
    rows = [
        _row(2021, "a", usage=1.0, available=1.0),
        _row(2021, "b", usage=None, available=0.0),
    ]
    coefficients = study.coefficient_rows(
        model,
        candidate,
        target_season=2022,
        previous={},
        training=rows,
    )
    by_feature = {row["feature"]: row for row in coefficients}
    assert set(by_feature) == set(study.COEFFICIENT_FEATURES)
    assert by_feature[study.EXPERIENCE]["training_source_available_n"] == 1
    assert by_feature[study.EXPERIENCE_AVAILABLE]["training_source_unresolved_n"] == 1


def test_metric_parity_compares_all_metrics(tmp_path: Path) -> None:
    values = {metric: float(index + 1) for index, metric in enumerate(study.METRICS)}
    result = study.metric_parity(
        values,
        values,
        candidate="R2",
        reference_path=tmp_path / "reference.csv",
        tolerance=1e-6,
    )
    assert result["passed"] is True
    assert result["max_abs_metric_delta"] == 0.0
