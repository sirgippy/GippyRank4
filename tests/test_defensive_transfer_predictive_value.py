import csv
import sys
from pathlib import Path

import numpy as np

from gippyrank.preseason import TeamSeason

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import investigate_defensive_transfer_predictive_value as study


def test_primary_variants_match_issue_106_declarations() -> None:
    candidates = study.candidate_definitions()
    assert [candidate.name.split("_", 1)[0] for candidate in candidates] == [
        "E0",
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
    ]
    assert candidates[2].features == study.d5_features()
    assert candidates[3].defensive_features == (study.EXPERIENCE,)
    assert candidates[4].defensive_features == (study.EXPERIENCE,)
    assert candidates[5].defensive_features == (study.IMPACT,)
    assert candidates[6].defensive_features == study.DEFENSIVE_FEATURES
    assert study.EXPERIENCE_AVAILABLE in candidates[4].features
    assert study.IMPACT_AVAILABLE in candidates[5].features


def test_defensive_loader_separates_zero_from_unresolved(tmp_path: Path) -> None:
    path = tmp_path / "team_season_features.csv"
    fields = [
        "season",
        "subdivision",
        "team_id",
        "feature_coverage_status",
        study.EXPERIENCE,
        study.IMPACT,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "season": 2022,
                "subdivision": "fbs",
                "team_id": "1",
                "feature_coverage_status": "no_incoming_defensive_transfer",
                study.EXPERIENCE: "",
                study.IMPACT: "",
            }
        )
        writer.writerow(
            {
                "season": 2022,
                "subdivision": "fbs",
                "team_id": "2",
                "feature_coverage_status": "partial",
                study.EXPERIENCE: "",
                study.IMPACT: "",
            }
        )
    values = study.load_defensive_features(path)
    assert values[(2022, "fbs", "1")][study.EXPERIENCE_AVAILABLE] == 1.0
    assert values[(2022, "fbs", "1")][study.EXPERIENCE] == 0.0
    assert values[(2022, "fbs", "2")][study.EXPERIENCE_AVAILABLE] == 0.0
    assert values[(2022, "fbs", "2")][study.EXPERIENCE] == 0.0


def _row(
    season: int, team_id: str, experience: float, impact: float, available: float
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
            study.EXPERIENCE: experience,
            study.IMPACT: impact,
            study.EXPERIENCE_AVAILABLE: available,
            study.IMPACT_AVAILABLE: available,
        },
    )


def test_permutation_preserves_season_values_and_missingness() -> None:
    rows = [
        _row(2022, "a", 1.0, 10.0, 1.0),
        _row(2022, "b", 2.0, 20.0, 1.0),
        _row(2022, "c", 0.0, 0.0, 0.0),
        _row(2023, "d", 3.0, 30.0, 1.0),
    ]
    permuted = study.permute_defensive_features(rows, seed=7)
    for season in (2022, 2023):
        original = [row for row in rows if row.season == season]
        actual = [row for row in permuted if row.season == season]
        assert [row.features[study.EXPERIENCE_AVAILABLE] for row in actual] == [
            row.features[study.EXPERIENCE_AVAILABLE] for row in original
        ]
        assert sorted(
            row.features[study.EXPERIENCE]
            for row in actual
            if row.features[study.EXPERIENCE_AVAILABLE] == 1.0
        ) == sorted(
            row.features[study.EXPERIENCE]
            for row in original
            if row.features[study.EXPERIENCE_AVAILABLE] == 1.0
        )
        assert sorted(
            row.features[study.IMPACT]
            for row in actual
            if row.features[study.IMPACT_AVAILABLE] == 1.0
        ) == sorted(
            row.features[study.IMPACT]
            for row in original
            if row.features[study.IMPACT_AVAILABLE] == 1.0
        )


def test_controls_are_availability_only() -> None:
    controls = study.control_definitions()
    assert study.EXPERIENCE not in controls[0].features
    assert study.EXPERIENCE_AVAILABLE in controls[0].features
    assert study.IMPACT not in controls[1].features
    assert study.IMPACT_AVAILABLE in controls[1].features
    assert study.EXPERIENCE not in controls[2].features
    assert study.IMPACT not in controls[2].features


def test_coefficient_coverage_uses_defensive_availability_indicators() -> None:
    candidate = study.candidate_definitions()[4]
    model = study.DirectRankModel(
        feature_names=[study.EXPERIENCE, study.EXPERIENCE_AVAILABLE],
        preprocessor=None,  # type: ignore[arg-type]
        beta=np.zeros(5),
        gamma=np.zeros(5),
        lag_count=0,
    )
    rows = [
        _row(2021, "a", 1.0, 1.0, 1.0),
        _row(2021, "b", 0.0, 0.0, 0.0),
    ]
    coefficients = study.coefficient_rows([candidate], {candidate.name: model}, rows)
    by_feature = {row["feature"]: row for row in coefficients}
    assert by_feature[study.EXPERIENCE]["training_source_available_n"] == 1
    assert by_feature[study.EXPERIENCE]["training_source_unavailable_n"] == 1
    assert by_feature[study.EXPERIENCE_AVAILABLE]["training_source_available_n"] == 1
