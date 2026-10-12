from __future__ import annotations

import importlib
import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import pytest

ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"


class _LoaderReached(Exception):
    """Stop an entrypoint as soon as it reaches its input loader."""


def _run_script(name: str) -> dict[str, object]:
    sys.path.insert(0, str(SCRIPTS))
    try:
        return runpy.run_path(str(SCRIPTS / name))
    finally:
        sys.path.pop(0)


def _record_original_load(
    monkeypatch: pytest.MonkeyPatch,
    loader_module: object,
    entrypoint: Callable[[], object],
    attribute: str,
    expected: dict[str, object],
) -> None:
    calls: list[dict[str, object]] = []

    def stop_at_load(**kwargs: object) -> NoReturn:
        calls.append(kwargs)
        raise _LoaderReached

    monkeypatch.setattr(loader_module, attribute, stop_at_load)
    with pytest.raises(_LoaderReached):
        entrypoint()
    assert calls == [expected]


def test_context_ablation_entrypoint_selects_original_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _run_script("investigate_context_ablation.py")
    _record_original_load(
        monkeypatch,
        values["v1"],
        values["main"],
        "load_rows",
        {"max_season": 2025, "corpus_policy": "original"},
    )


def test_lean_context_entrypoint_selects_original_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _run_script("validate_lean_context.py")
    main = values["main"]
    monkeypatch.setitem(main.__globals__, "clean_output", lambda: None)
    _record_original_load(
        monkeypatch,
        values["v1"],
        main,
        "load_rows",
        {"max_season": 2025, "corpus_policy": "original"},
    )


def test_primitive_prior_reconstruction_selects_original_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _run_script("build_primitive_box_score_likelihood.py")
    sys.path.insert(0, str(SCRIPTS))
    try:
        history_prior = importlib.import_module("build_preseason_prior")
    finally:
        sys.path.pop(0)

    calls: list[dict[str, object]] = []

    def stop_at_load(**kwargs: object) -> NoReturn:
        calls.append(kwargs)
        raise _LoaderReached

    monkeypatch.setattr(history_prior, "load_rows", stop_at_load)
    with pytest.raises(_LoaderReached):
        values["reconstruct_historical_priors"]({}, [], [2022])
    assert calls == [{"max_season": 2021, "corpus_policy": "original"}]


def test_standalone_preseason_prior_selects_original_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = _run_script("build_preseason_prior.py")
    main = values["main"]
    globals_ = main.__globals__
    monkeypatch.setitem(globals_, "OUT", tmp_path / "preseason")
    calls: list[dict[str, object]] = []

    def stop_at_load(**kwargs: object) -> NoReturn:
        calls.append(kwargs)
        raise _LoaderReached

    monkeypatch.setitem(globals_, "load_rows", stop_at_load)
    with pytest.raises(_LoaderReached):
        main()
    assert calls == [{"max_season": 2025, "corpus_policy": "original"}]


def test_original_loader_filters_out_of_population_rank_but_default_stays_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _run_script("build_preseason_prior.py")
    prior = {
        "season": "2006",
        "subdivision": "fcs",
        "team_id": "2046",
        "team_name": "Austin Peay",
        "team_population": "119",
        "usable_systems": "1",
        "rank_observations": "[1]",
        "pmf": '[{"rank":1,"count":1,"probability":1.0}]',
    }
    original_target = {
        **prior,
        "season": "2007",
        "rank_observations": "[119,120]",
        "pmf": (
            '[{"rank":119,"count":1,"probability":0.5},'
            '{"rank":120,"count":1,"probability":0.5}]'
        ),
    }
    features = [
        {
            "season": season,
            "subdivision": "fcs",
            "team_id": "2046",
            "team_name": "Austin Peay",
        }
        for season in ("2006", "2007")
    ]

    def fake_read_csv(path: Path) -> list[dict[str, str]]:
        return (
            features
            if path.name == "team_season_features.csv"
            else [prior, original_target]
        )

    monkeypatch.setitem(values["load_rows"].__globals__, "read_csv", fake_read_csv)
    with pytest.raises(ValueError, match=r"constituent rank 120 is outside 1\.\.119"):
        values["load_rows"](max_season=2007)

    rows, _, _ = values["load_rows"](max_season=2007, corpus_policy="original")
    target = next(row for row in rows if row.season == 2007)
    assert target.target_ranks.tolist() == [119]
