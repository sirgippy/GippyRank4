from __future__ import annotations

import csv
import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest

from gippyrank.posterior.engine import LikelihoodV1
from gippyrank.posterior.snapshots import build_snapshot, corpus_provenance, snapshot_id


def _write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _root(tmp_path: Path) -> Path:
    fields = ["season", "subdivision", "team_id", "team_name", "pmf"]
    rows = [
        {
            "season": "2026",
            "subdivision": "fbs",
            "team_id": "1",
            "team_name": "One",
            "pmf": "[0.7,0.3]",
        },
        {
            "season": "2026",
            "subdivision": "fbs",
            "team_id": "2",
            "team_name": "Two",
            "pmf": "[0.3,0.7]",
        },
        {
            "season": "2026",
            "subdivision": "fcs",
            "team_id": "3",
            "team_name": "Three",
            "pmf": "[0.5,0.5]",
        },
    ]
    for family in ("context", "history"):
        roots = [family]
        if family == "context":
            roots.append("context_v1_3")
        for family_root in roots:
            _write(
                tmp_path
                / f"data/processed/preseason/{family_root}/annual/2026/predictions.csv",
                fields,
                rows,
            )
    game_fields = [
        "id",
        "season",
        "week",
        "seasonType",
        "startDate",
        "completed",
        "neutralSite",
        "conferenceGame",
        "homeId",
        "homeTeam",
        "homeClassification",
        "homeConference",
        "homePoints",
        "awayId",
        "awayTeam",
        "awayClassification",
        "awayConference",
        "awayPoints",
    ]
    _write(
        tmp_path / "data/processed/cfbd/games.csv",
        game_fields,
        [
            {
                "id": "early",
                "season": "2026",
                "week": "1",
                "seasonType": "regular",
                "startDate": "2026-08-29T00:00:00Z",
                "completed": "True",
                "neutralSite": "False",
                "conferenceGame": "True",
                "homeId": "1",
                "homeTeam": "One",
                "homeClassification": "fbs",
                "homeConference": "",
                "homePoints": "20",
                "awayId": "3",
                "awayTeam": "Three",
                "awayClassification": "fcs",
                "awayConference": "",
                "awayPoints": "10",
            },
            {
                "id": "lower",
                "season": "2026",
                "week": "1",
                "seasonType": "regular",
                "startDate": "2026-08-28T00:00:00Z",
                "completed": "True",
                "neutralSite": "False",
                "conferenceGame": "False",
                "homeId": "1",
                "homeTeam": "One",
                "homeClassification": "fbs",
                "homeConference": "",
                "homePoints": "20",
                "awayId": "4",
                "awayTeam": "Four",
                "awayClassification": "ii",
                "awayConference": "",
                "awayPoints": "10",
            },
            {
                "id": "later",
                "season": "2026",
                "week": "2",
                "seasonType": "regular",
                "startDate": "2026-09-10T00:00:00Z",
                "completed": "True",
                "neutralSite": "False",
                "conferenceGame": "True",
                "homeId": "1",
                "homeTeam": "One",
                "homeClassification": "fbs",
                "homeConference": "",
                "homePoints": "20",
                "awayId": "2",
                "awayTeam": "Two",
                "awayClassification": "fbs",
                "awayConference": "",
                "awayPoints": "10",
            },
        ],
    )
    return tmp_path


def _write_current_provenance(
    root: Path, fbs_retrieved_at: datetime, fcs_retrieved_at: datetime | None = None
) -> None:
    directory = root / "data/raw/cfbd/games"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "2026.json").write_text("[]", encoding="utf-8")
    (directory / "2026-fcs.json").write_text(
        '[{"homeId": "3", "homeClassification": "fcs"}]', encoding="utf-8"
    )
    fcs_retrieved_at = fbs_retrieved_at if fcs_retrieved_at is None else fcs_retrieved_at
    for name, retrieved_at in (
        ("2026.json", fbs_retrieved_at),
        ("2026-fcs.json", fcs_retrieved_at),
    ):
        (directory / f"{name}.provenance.json").write_text(
            json.dumps(
                {
                    "content_sha256": f"hash-{name}",
                    "retrieved_at": retrieved_at.isoformat(),
                    "source_kind": "cfbd_api",
                }
            ),
            encoding="utf-8",
        )


def test_preseason_snapshot_is_prior_only_and_complete(tmp_path: Path) -> None:
    root = _root(tmp_path)
    snapshot = build_snapshot(
        season=2026,
        cutoff=None,
        prior_family="context",
        snapshot_type="preseason",
        root=root,
    )
    metadata = json.loads((snapshot.directory / "metadata.json").read_text())
    assert metadata["included_game_count"] == 0
    assert "canonical_public_model" not in metadata
    assert metadata["prior_family"] == "context"
    assert metadata["source_mode"] == "preseason_prior_only"
    assert metadata["source_retrieved_at"] is None
    assert metadata["source_retrieval_times"] == {}
    assert metadata["prior_artifact_sha256"]
    assert {path.name for path in snapshot.directory.iterdir()} >= {
        "metadata.json",
        "rankings.json",
        "rankings.csv",
        "posterior_pmfs.csv",
        "included_games.csv",
        "diagnostics.json",
    }
    assert len(json.loads((snapshot.directory / "rankings.json").read_text())) == 3


def test_missing_active_context_does_not_fall_back_to_legacy_prior(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    active_prior = (
        root / "data/processed/preseason/context_v1_3/annual/2026/predictions.csv"
    )
    legacy_prior = (
        root / "data/processed/preseason/context/annual/2026/predictions.csv"
    )
    active_prior.unlink()
    assert legacy_prior.exists()

    with pytest.raises(FileNotFoundError, match="context_v1_3"):
        build_snapshot(
            season=2026,
            cutoff=None,
            prior_family="context",
            snapshot_type="preseason",
            root=root,
        )


def test_cutoff_and_lower_division_policy_are_explicit(tmp_path: Path) -> None:
    root = _root(tmp_path)
    # No likelihood is needed when cutoff is before the first game; this proves
    # filtering happens before any inference attempt.
    snapshot = build_snapshot(
        season=2026,
        cutoff=date(2026, 8, 28),
        prior_family="history",
        snapshot_type="weekly",
        root=root,
    )
    assert snapshot.metadata["included_game_count"] == 0
    assert snapshot.metadata["excluded_lower_division_games"] == 1
    assert snapshot.metadata["source_mode"] == "historical_frozen"
    assert snapshot.metadata["requested_cutoff"] == snapshot.metadata["effective_cutoff"]
    assert snapshot.metadata["source_retrieval_times"] == {}


def test_prior_families_have_equivalent_neutral_snapshot_schema(tmp_path: Path) -> None:
    root = _root(tmp_path)
    context = build_snapshot(
        season=2026,
        cutoff=None,
        prior_family="context",
        snapshot_type="preseason",
        root=root,
    )
    history = build_snapshot(
        season=2026,
        cutoff=None,
        prior_family="history",
        snapshot_type="preseason",
        root=root,
    )
    assert context.metadata.keys() == history.metadata.keys()
    assert context.metadata["prior_family"] == "context"
    assert history.metadata["prior_family"] == "history"
    assert "canonical_public_model" not in context.metadata


def test_live_snapshot_clamps_stale_cache_to_explicit_effective_cutoff(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    retrieved_at = datetime(2026, 8, 30, 12, tzinfo=UTC)
    _write_current_provenance(root, retrieved_at)
    snapshot = build_snapshot(
        season=2026,
        cutoff=datetime(2026, 9, 11, 23, 59, tzinfo=UTC),
        prior_family="context",
        snapshot_type="live",
        root=root,
        likelihood=LikelihoodV1(np.zeros(34), 1.0, 15.0),
    )
    metadata = snapshot.metadata
    assert metadata["source_mode"] == "current_cached_cfbd"
    assert metadata["requested_cutoff"] == "2026-09-11T23:59:00+00:00"
    assert metadata["source_retrieved_at"] == "2026-08-30T12:00:00+00:00"
    assert metadata["source_retrieval_times"] == {
        "fbs": "2026-08-30T12:00:00+00:00",
        "fcs": "2026-08-30T12:00:00+00:00",
    }
    assert metadata["effective_cutoff"] == metadata["source_retrieved_at"]
    assert metadata["effective_cutoff"] < metadata["requested_cutoff"]
    assert metadata["source_response_hashes"] == {
        "2026-fcs.json.provenance.json": "hash-2026-fcs.json",
        "2026.json.provenance.json": "hash-2026.json",
    }
    assert metadata["included_game_ids"] == ["early"]


@pytest.mark.parametrize(
    ("fbs_time", "fcs_time", "expected"),
    [
        (datetime(2026, 8, 30, 12, tzinfo=UTC), datetime(2026, 8, 30, 12, 8, tzinfo=UTC), datetime(2026, 8, 30, 12, tzinfo=UTC)),
        (datetime(2026, 8, 30, 12, 8, tzinfo=UTC), datetime(2026, 8, 30, 12, tzinfo=UTC), datetime(2026, 8, 30, 12, tzinfo=UTC)),
    ],
)
def test_combined_current_source_boundary_is_earliest_required_response(
    tmp_path: Path, fbs_time: datetime, fcs_time: datetime, expected: datetime
) -> None:
    root = _root(tmp_path)
    _write_current_provenance(root, fbs_time, fcs_time)
    provenance = corpus_provenance(root, 2026)
    assert provenance.source_retrieved_at == expected
    assert provenance.source_retrieval_times == {"fbs": fbs_time, "fcs": fcs_time}
    snapshot = build_snapshot(
        season=2026,
        cutoff=datetime(2026, 9, 11, 23, 59, tzinfo=UTC),
        prior_family="context",
        snapshot_type="weekly",
        root=root,
        likelihood=LikelihoodV1(np.zeros(34), 1.0, 15.0),
    )
    assert snapshot.metadata["effective_cutoff"] == expected.isoformat()
    assert snapshot.metadata["effective_cutoff"] <= fbs_time.isoformat()
    assert snapshot.metadata["effective_cutoff"] <= fcs_time.isoformat()
    assert snapshot.metadata["effective_cutoff"] < snapshot.metadata["requested_cutoff"]


def test_identical_eligible_inputs_produce_identical_ranking_rows(tmp_path: Path) -> None:
    root = _root(tmp_path)
    retrieved_at = datetime(2026, 8, 30, 12, tzinfo=UTC)
    _write_current_provenance(root, retrieved_at)
    common = {
        "season": 2026,
        "cutoff": datetime(2026, 9, 11, 23, 59, tzinfo=UTC),
        "prior_family": "history",
        "snapshot_type": "live",
        "root": root,
        "likelihood": LikelihoodV1(np.zeros(34), 1.0, 15.0),
    }
    first = build_snapshot(**common, output_root=tmp_path / "one")
    second = build_snapshot(**common, output_root=tmp_path / "two")
    assert (first.directory / "rankings.json").read_bytes() == (
        second.directory / "rankings.json"
    ).read_bytes()


def test_context_replay_uses_frozen_included_games_after_schedule_mutation(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    likelihood = LikelihoodV1(np.zeros(34), 1.0, 15.0)
    cutoff = datetime(2026, 9, 11, 23, 59, tzinfo=UTC)
    source = build_snapshot(
        season=2026,
        cutoff=cutoff,
        prior_family="context",
        prior_model_version="1.2",
        snapshot_type="weekly",
        root=root,
        output_root=tmp_path / "source-output",
        likelihood=likelihood,
    )
    source_rows = (source.directory / "included_games.csv").read_bytes()
    games_path = root / "data/processed/cfbd/games.csv"
    games = []
    with games_path.open(newline="", encoding="utf-8") as handle:
        games = list(csv.DictReader(handle))
    games[-1]["homePoints"] = "99"
    _write(games_path, list(games[0]), games)

    replay = build_snapshot(
        season=2026,
        cutoff=cutoff,
        prior_family="context",
        prior_model_version="1.3",
        snapshot_type="weekly",
        root=root,
        output_root=tmp_path / "replay-output",
        likelihood=likelihood,
        evidence_snapshot=source.directory,
    )

    assert (replay.directory / "included_games.csv").read_bytes() == source_rows
    assert replay.metadata["backfill"] is True
    assert replay.metadata["source_evidence_snapshot_id"] == source.snapshot_id
    assert replay.metadata["game_corpus_sha256"] == source.metadata["game_corpus_sha256"]
    team_seasons = json.loads(
        (replay.directory / "team_seasons.json").read_text(encoding="utf-8")
    )
    assert team_seasons["schedule_source"]["kind"] == "frozen_included_games"


def test_posterior_docs_name_the_80_percent_interval() -> None:
    text = Path("docs/posterior_v1.md").read_text(encoding="utf-8")
    assert "80% interval coverage" in text
    assert "90%" not in text


def test_snapshot_ids_are_stable_and_human_readable() -> None:
    assert snapshot_id(2026, "preseason", "context") == "2026-preseason-context"
    assert (
        snapshot_id(2026, "weekly", "history", date(2026, 9, 1))
        == "2026-weekly-2026-09-01-history"
    )
