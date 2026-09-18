import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import investigate_transfer_signal_decomposition as study


def test_decomposition_declares_exactly_d0_through_d8() -> None:
    candidates = study.candidate_definitions()

    assert [candidate.name.split("_", 1)[0] for candidate in candidates] == [
        "D0",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
    ]
    assert candidates[0].features == tuple(study.prior.BASE_CONTEXT_FEATURES)
    assert candidates[1].features == tuple(study.prior.C_MINUS_RP_FEATURES)
    assert candidates[2].continuity_features == ("returning_pct_ppa",)
    assert candidates[3].continuity_features == ("transfer_in_prior_usage_sum",)
    assert candidates[4].continuity_features == ("transfer_out_prior_usage_sum",)
    assert candidates[5].continuity_features == (
        "returning_pct_ppa",
        "transfer_in_prior_usage_sum",
    )
    assert candidates[6].continuity_features == (
        "returning_pct_ppa",
        "transfer_out_prior_usage_sum",
    )
    assert candidates[7].continuity_features == (
        "returning_pct_ppa",
        "transfer_in_prior_usage_sum",
        "transfer_out_prior_usage_sum",
    )
    assert candidates[8].continuity_features == (
        *study.prior.RP_FEATURES,
        "transfer_in_prior_usage_sum",
        "transfer_net_prior_usage",
    )


def test_decomposition_has_no_interaction_variants() -> None:
    assert all(candidate.interaction is None for candidate in study.candidate_definitions())


def test_fmt_handles_missing_csv_values() -> None:
    assert study.fmt("") == "n/a"
    assert study.fmt(None) == "n/a"
