"""Guard the committed matched posterior study against arm or evidence drift."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "data/processed/history_context_posterior_study"


def rows(name: str) -> list[dict[str, str]]:
    with (STUDY / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_historical_arms_share_history_and_evidence() -> None:
    summaries = rows("historical_checkpoints.csv")
    evidence = rows("historical_evidence.csv")
    all_teams = [row for row in summaries if row["transfer_group"] == "all"]
    by_checkpoint: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in all_teams:
        key = row["season"], row["checkpoint"]
        by_checkpoint.setdefault(key, {})[row["context_arm"]] = row
    assert len(by_checkpoint) == 28
    for pair in by_checkpoint.values():
        assert set(pair) == {"frozen_2021", "rolling_origin"}
        frozen, rolling = pair["frozen_2021"], pair["rolling_origin"]
        assert frozen["teams"] == rolling["teams"]
        for field in frozen:
            if field.startswith("history_"):
                assert frozen[field] == rolling[field]

    by_evidence: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in evidence:
        key = row["season"], row["checkpoint"]
        by_evidence.setdefault(key, {})[row["family"]] = row
        assert row["converged"] == "True"
        assert int(row["iterations"]) <= 500
    assert set(by_evidence) == set(by_checkpoint)
    for triple in by_evidence.values():
        assert set(triple) == {"frozen_2021", "rolling_origin", "history"}
        assert len({row["included_games"] for row in triple.values()}) == 1
        assert (
            len({row["excluded_lower_division_games"] for row in triple.values()}) == 1
        )


def test_study_uses_production_inference_configuration() -> None:
    provenance = json.loads((STUDY / "provenance.json").read_text())
    assert provenance["inference"] == {
        "max_iterations": 500,
        "tolerance": 1e-9,
        "damping": 0.35,
    }
    assert provenance["models"]["context_rolling_origin"] == {
        str(season): f"1.3 research P3 fitted through {season - 1}"
        for season in (2022, 2023, 2024, 2025)
    }


def test_reported_september_crossover_is_present_in_both_arms() -> None:
    summaries = [
        row
        for row in rows("historical_checkpoints.csv")
        if row["transfer_group"] == "all"
    ]
    for arm in ("frozen_2021", "rolling_origin"):
        for season in (2023, 2024, 2025):
            annual = sorted(
                (
                    row
                    for row in summaries
                    if row["context_arm"] == arm and int(row["season"]) == season
                ),
                key=lambda row: int(row["checkpoint"]),
            )
            assert len(annual) == 7
            assert float(annual[0]["posterior_c_minus_h_nll"]) < 0
            assert all(float(row["posterior_c_minus_h_nll"]) > 0 for row in annual[1:])
