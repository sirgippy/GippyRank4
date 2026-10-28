"""Prepare the recovered 2025 archive for the issue 263 design, without inference."""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from gippyrank.artifact_hashes import (
    INCLUDED_GAME_FIELDS,
    included_game_rows_sha256,
    posterior_pmfs_sha256,
)
from gippyrank.data.cfbd import deduplicate_schedule_queries
from gippyrank.data.cfbd_games import apply_historical_game_subdivisions
from gippyrank.data.game_evidence import reconcile_game_evidence_rows
from gippyrank.posterior.engine import Game, LikelihoodV1, Team
from gippyrank.posterior.predictive import (
    ScheduledGame,
    conditional_margin_location_surface,
)
from gippyrank.posterior.snapshots import (
    _game_from_included_row,
    add_fcs_fallbacks,
    load_teams,
    sha256,
)
from gippyrank.quality_of_record_snapshot import _load_source_likelihood
from gippyrank.research.issue_242_quality_of_results import (
    _read_json,
    _read_rows,
    _write_csv,
)

ARTIFACT_DIR = "data/research/issue_263_real_season_design"
ARCHIVES = {
    "2025.json": "6312d8406be09dee0064069668fc40879dd0736ed3e49946c6c72192e6a0e4fc",
    "2025-fcs.json": "38329f854c0df38d17b34d5f7841753d6638d7377592dcf743cb4e27cbd9d32f",
}
HISTORICAL_CUTOFFS = {
    "2025-prebowl": "2025-12-10T00:00:00+00:00",
    "2025-final": "2026-01-23T00:00:00+00:00",
}
REVIEWED_REMATCHES = {
    frozenset(pair)
    for pair in (
        ("401767238", "401836216"),
        ("401767541", "401836195"),
        ("401767438", "401836198"),
        ("401767104", "401833989"),
        ("401752711", "401779842"),
        ("401752718", "401777351"),
        ("401752870", "401769074"),
        ("401752742", "401769073"),
        ("401760393", "401777326"),
        ("401762874", "401777329"),
        ("401752943", "401752946"),
        ("401761644", "401778324"),
        ("401756945", "401777327"),
        ("401752765", "401779840"),
        ("401754593", "401777328"),
        ("401757302", "401777325"),
    )
}


def normalized_archive(raw: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    payloads = {}
    for classification, name in (("fbs", "2025.json"), ("fcs", "2025-fcs.json")):
        if sha256(raw / name) != ARCHIVES[name]:
            raise ValueError(f"historical archive hash mismatch: {name}")
        payloads[classification] = json.loads((raw / name).read_text())
    records, overlap = deduplicate_schedule_queries(payloads)
    rows = apply_historical_game_subdivisions(
        [
            {
                key: "" if record.get(key) is None else str(record[key])
                for key in INCLUDED_GAME_FIELDS
            }
            for record in records
        ]
    )
    fcs_ids = {
        r[side + "Id"]
        for r in rows
        for side in ("home", "away")
        if r[side + "Classification"] == "fcs"
    }
    repeated = []
    groups = {}
    for record in records:
        if record["homeClassification"] not in {"fbs", "fcs"} or record[
            "awayClassification"
        ] not in {"fbs", "fcs"}:
            continue
        key = tuple(sorted((str(record["homeId"]), str(record["awayId"]))))
        groups.setdefault(key, []).append(record)
    for pair in groups.values():
        if len(pair) < 2:
            continue
        if frozenset(str(r["id"]) for r in pair) not in REVIEWED_REMATCHES:
            raise ValueError("unreviewed historical repeated matchup")
        repeated.append(
            [
                {
                    "game_id": str(r["id"]),
                    "date": r["startDate"],
                    "notes": r["notes"],
                    "home_id": str(r["homeId"]),
                    "away_id": str(r["awayId"]),
                }
                for r in sorted(pair, key=lambda r: r["startDate"])
            ]
        )
    return rows, {
        "reviewed_rematches": repeated,
        "raw_response_counts": {k: len(v) for k, v in payloads.items()},
        "overlap_count": overlap,
        "unique_game_count": len(rows),
        "full_season_fcs_population_size": len(fcs_ids),
    }


def cutoff_rows(
    rows: list[dict[str, str]], cutoff: str
) -> tuple[list[dict[str, str]], int]:
    before = datetime.fromisoformat(cutoff) - timedelta(hours=48)
    candidates = [
        r
        for r in rows
        if int(r["season"]) == 2025 and datetime.fromisoformat(r["startDate"]) <= before
    ]
    # The generic 30-day ambiguity guard rejects genuine conference-title
    # rematches here. Their exact pairs are audited above; retain both games
    # and still apply the established completion/score validation individually.
    reconciled = []
    for row in candidates:
        valid, exclusions = reconcile_game_evidence_rows([row])
        if exclusions:
            raise ValueError(f"historical evidence requires review: {row['id']}")
        reconciled.extend(valid)
    eligible = []
    excluded = 0
    for row in reconciled:
        if row["homeClassification"] not in {"fbs", "fcs"} or row[
            "awayClassification"
        ] not in {"fbs", "fcs"}:
            excluded += 1
            continue
        if (
            row["completed"] != "True"
            or min(int(row["homePoints"]), int(row["awayPoints"])) < 0
            or row["homePoints"] == row["awayPoints"]
        ):
            raise ValueError(f"invalid historical completed evidence: {row['id']}")
        eligible.append(row)
    return eligible, excluded


def prior_descriptors(
    teams: list[Team],
    rows: list[dict[str, str]],
    prior_rows: dict[str, dict[str, str]],
    likelihood: LikelihoodV1,
    slot: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {t.team_id: t for t in teams}
    fbs = sorted(t.team_id for t in teams if t.subdivision == "fbs")
    order = sorted(fbs, key=lambda t: (float(prior_rows[t]["expected_rank"]), t))
    ordinal = {t: i for i, t in enumerate(order, 1)}
    reference = min(
        fbs, key=lambda t: (abs(float(prior_rows[t]["expected_rank"]) - 65), t)
    )
    evidence = []
    surfaces = {}

    def chance(game: Game, home: Team, away: Team) -> float:
        key = (
            home.subdivision,
            away.subdivision,
            len(home.prior),
            len(away.prior),
            game.neutral_site,
        )
        if key not in surfaces:
            matchup = ScheduledGame(
                game.game_id,
                home.team_id,
                away.team_id,
                home.subdivision,
                away.subdivision,
                game.neutral_site,
            )
            location = conditional_margin_location_surface(
                matchup, home, away, likelihood
            )
            surfaces[key] = student_t.sf(
                -location / likelihood.scale, likelihood.degrees_of_freedom
            )
        return float(home.prior @ surfaces[key] @ away.prior)

    for row in rows:
        game = _game_from_included_row(row)
        home, away = by_id[game.home_id], by_id[game.away_id]
        if home.subdivision == away.subdivision == "fcs":
            continue
        actual = chance(game, home, away)
        for focal, opponent, margin, is_home in (
            (home, away, game.home_points - game.away_points, True),
            (away, home, game.away_points - game.home_points, False),
        ):
            if focal.subdivision != "fbs":
                continue
            standardized = Team(
                focal.team_id, focal.name, "fbs", by_id[reference].prior
            )
            p = chance(
                game,
                standardized if is_home else opponent,
                opponent if is_home else standardized,
            )
            evidence.append(
                {
                    "season": 2025,
                    "snapshot_id": slot,
                    "team_id": focal.team_id,
                    "game_id": game.game_id,
                    "opponent_id": opponent.team_id,
                    "opponent_subdivision": opponent.subdivision,
                    "margin": margin,
                    "prior_win_probability": actual if is_home else 1 - actual,
                    "schedule_reference_win_probability": p if is_home else 1 - p,
                }
            )
    descriptors = []
    for team_id in fbs:
        own = [r for r in evidence if r["team_id"] == team_id]
        wins = sum(r["margin"] > 0 for r in own)
        descriptors.append(
            {
                "season": 2025,
                "snapshot_id": slot,
                "team_id": team_id,
                "team_name": by_id[team_id].name,
                "reference_kind": "preseason_context_v1.2",
                "reference_ordinal": ordinal[team_id],
                "reference_expected_rank": float(prior_rows[team_id]["expected_rank"]),
                "games": len(own),
                "wins": wins,
                "losses": len(own) - wins,
                "win_fraction": wins / len(own),
                "reference_team_id": reference,
                "schedule_reference_mean_win_probability": float(
                    np.mean([r["schedule_reference_win_probability"] for r in own])
                ),
                "prior_expected_wins": sum(r["prior_win_probability"] for r in own),
                "average_margin": float(np.mean([r["margin"] for r in own])),
            }
        )
    return descriptors, evidence


def select_historical_cases(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    rows = sorted(rows, key=lambda r: r["team_id"])
    selected = {}

    def add(label: str, row: dict[str, Any]) -> None:
        selected.setdefault(row["team_id"], []).append(label)

    for label, target in (
        ("elite", 5),
        ("strong", 20),
        ("middle", 65),
        ("weak", 120),
        ("reference_boundary_12", 12),
        ("reference_boundary_25", 25),
    ):
        add(
            label,
            min(
                rows, key=lambda r: (abs(r["reference_ordinal"] - target), r["team_id"])
            ),
        )
    excellent = [r for r in rows if r["win_fraction"] >= 0.8 and r["games"] >= 5]
    middling = [r for r in rows if 0.3 <= r["win_fraction"] <= 0.6 and r["games"] >= 5]
    add(
        "excellent_record_easy_schedule",
        min(
            excellent,
            key=lambda r: (-r["schedule_reference_mean_win_probability"], r["team_id"]),
        ),
    )
    add(
        "middling_record_difficult_schedule",
        min(
            middling,
            key=lambda r: (r["schedule_reference_mean_win_probability"], r["team_id"]),
        ),
    )
    for label, low, high in (
        ("middle_schedule_pair", 41, 95),
        ("weak_schedule_pair", 96, 136),
    ):
        group = [r for r in rows if low <= r["reference_ordinal"] <= high]
        pairs = [
            (a, b)
            for a, b in itertools.combinations(group, 2)
            if (a["wins"], a["losses"]) == (b["wins"], b["losses"])
        ]
        if not pairs:
            raise ValueError(f"no historical matched-record pair: {label}")
        for row in min(
            pairs,
            key=lambda p: (
                -abs(
                    p[0]["schedule_reference_mean_win_probability"]
                    - p[1]["schedule_reference_mean_win_probability"]
                ),
                p[0]["team_id"],
                p[1]["team_id"],
            ),
        ):
            add(label, row)
    return selected


def build_historical_design(root: Path, output: Path) -> dict[str, Any]:
    source = root / ARTIFACT_DIR / "raw"
    target = output / "raw"
    target.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVES:
        if source.resolve() != target.resolve():
            (target / name).write_bytes((source / name).read_bytes())
    normalized, audit = normalized_archive(target)
    _write_csv(
        output / "historical_normalized_games.csv",
        normalized,
        list(INCLUDED_GAME_FIELDS),
    )
    source_audit = {
        "source_kind": "retained_cfbd_api_archives",
        "acquisition_retrieved_at": None,
        "acquisition_time_limitation": "No retained retrieval sidecars; local file timestamps are not acquisition proof",
        "source_worktrees": [
            {
                "path": "/home/gippy/src/GippyRank4",
                "head_at_discovery": "c0de2e52c658c945bd575e061dc35954359aef19",
            },
            {
                "path": "/tmp/GippyRank4-issue-164",
                "head_at_discovery": "0fca0ede27ddaa153b1b22516979700f1f950646",
            },
        ],
        "original_relative_directory": "data/raw/cfbd/games",
        "byte_identical_copies_verified": True,
        "archive_sha256": ARCHIVES,
        **audit,
    }
    (output / "historical_source_audit.json").write_text(
        json.dumps(source_audit, indent=2, sort_keys=True) + "\n"
    )
    likelihood = _load_source_likelihood(
        {
            "historical_likelihood_sha256": sha256(
                root / "data/processed/posterior/historical_likelihood_v1.json"
            ),
            "historical_likelihood_provenance": "loaded_from_artifact",
            "historical_likelihood_version": "V1",
        },
        root,
    )
    snapshots = []
    all_descriptors, all_evidence = [], []
    for slot, cutoff in HISTORICAL_CUTOFFS.items():
        included, lower = cutoff_rows(normalized, cutoff)
        teams, team_rows, prior = load_teams(root, 2025, "context", "1.2")
        raw_fbs_ids = {
            r[side + "Id"]
            for r in normalized
            for side in ("home", "away")
            if r[side + "Classification"] == "fbs"
        }
        if {t.team_id for t in teams} != raw_fbs_ids:
            raise ValueError("historical prior and game FBS populations differ")
        prior_rows = dict(team_rows)
        teams, fallback_ids = add_fcs_fallbacks(
            teams, team_rows, included, audit["full_season_fcs_population_size"]
        )
        directory = output / "inputs" / slot
        directory.mkdir(parents=True, exist_ok=True)
        _write_csv(
            directory / "included_games.csv", included, list(INCLUDED_GAME_FIELDS)
        )
        prior_payload = [
            {
                "team_id": t.team_id,
                "name": t.name,
                "subdivision": t.subdivision,
                # Retain constructor inputs so the execution loader applies
                # Team's normalization exactly once, just as load_teams does.
                "pmf": (
                    json.loads(prior_rows[t.team_id]["pmf"])
                    if t.subdivision == "fbs"
                    else np.full(
                        audit["full_season_fcs_population_size"],
                        1 / audit["full_season_fcs_population_size"],
                    ).tolist()
                ),
            }
            for t in sorted(teams, key=lambda t: t.team_id)
        ]
        (directory / "prior_pmfs.json").write_text(
            json.dumps(prior_payload, separators=(",", ":")) + "\n"
        )
        metadata = {
            "snapshot_id": slot,
            "season": 2025,
            "publication_id": None,
            "publication_status": "unpublished_research_reconstruction",
            "input_kind": "frozen_reconstructed_network",
            "selected": True,
            "information_phase": "pre_bowl" if slot.endswith("prebowl") else "final",
            "requested_cutoff": cutoff,
            "effective_cutoff": cutoff,
            "kickoff_safety_lag_hours": 48,
            "latest_eligible_kickoff": (
                datetime.fromisoformat(cutoff) - timedelta(hours=48)
            ).isoformat(),
            "source_available_at_cutoff_verified": False,
            "source_acquisition_retrieved_at": None,
            "context_prior_version": "1.2",
            "context_prior_effective_cutoff": "2025-08-15",
            "prior_artifact": str(prior.relative_to(root)),
            "prior_artifact_sha256": sha256(prior),
            "prior_pmfs_sha256": posterior_pmfs_sha256(
                {t.team_id: t.prior for t in teams}
            ),
            "historical_likelihood_version": "V1",
            "historical_likelihood_sha256": sha256(
                root / "data/processed/posterior/historical_likelihood_v1.json"
            ),
            "inference_configuration": {
                "max_iterations": 500,
                "tolerance": 1e-9,
                "damping": 0.35,
            },
            "included_game_count": len(included),
            "included_game_ids": [r["id"] for r in included],
            "included_game_rows_sha256": included_game_rows_sha256(included),
            "excluded_lower_or_unknown_division_count": lower,
            "fbs_team_ids": sorted(t.team_id for t in teams if t.subdivision == "fbs"),
            "network_team_ids": sorted(t.team_id for t in teams),
            "fcs_population_size": audit["full_season_fcs_population_size"],
            "fcs_fallback_team_ids": list(fallback_ids),
            "quality_of_play_reference": None,
            "quality_of_record_reference": None,
            "endpoint_reference_plan": "Compute independent existing margin/binary BP plus existing prior stripping in execution; compare with lambda=1/0, no stored publication parity claim",
            "input_directory": f"inputs/{slot}",
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        snapshots.append(metadata)
        desc, evidence = prior_descriptors(
            teams, included, prior_rows, likelihood, slot
        )
        all_descriptors.extend(desc)
        all_evidence.extend(evidence)
    selection = select_historical_cases(
        [r for r in all_descriptors if r["snapshot_id"] == "2025-prebowl"]
    )
    cases = [
        dict(
            row,
            categories=selection[row["team_id"]],
            selection_snapshot_id="2025-prebowl",
            questions=[
                (
                    {
                        "elite": "How do winning and margins affect elite-season credit?",
                        "strong": "How do winning and margins affect this strong prior-reference team?",
                        "middle": "Does this middle prior-reference case show a schedule advantage amid uncertainty?",
                        "weak": "Does the weak-team schedule/uncertainty reversal appear in this completed season?",
                        "middle_schedule_pair": "How do identical middle-team records with different schedules compare across candidates?",
                        "weak_schedule_pair": "Does the easier weak-team schedule reduce posterior SD and interval widths?",
                        "excellent_record_easy_schedule": "Does this excellent record receive disproportionate credit for an easier schedule?",
                        "middling_record_difficult_schedule": "Does opponent evidence explain this team's ordinary record?",
                    }.get(
                        category,
                        "Does movement near this prior-reference boundary reflect substantial evidence or closely spaced teams?",
                    )
                )
                for category in selection[row["team_id"]]
            ]
            + [
                "How do uncertainty and postseason evidence change the pre-bowl interpretation?"
            ],
        )
        for row in all_descriptors
        if row["team_id"] in selection
    ]
    for name, rows in (
        ("historical_reference_descriptors.csv", all_descriptors),
        ("historical_reference_games.csv", all_evidence),
    ):
        _write_csv(output / name, rows, list(rows[0]))
    (output / "historical_cases.json").write_text(
        json.dumps(cases, indent=2, sort_keys=True) + "\n"
    )
    paths = [
        output / "historical_normalized_games.csv",
        output / "historical_source_audit.json",
        output / "historical_reference_descriptors.csv",
        output / "historical_reference_games.csv",
        output / "historical_cases.json",
        *target.glob("*.json"),
        *(output / "inputs").glob("**/*.*"),
    ]
    return {
        "snapshots": snapshots,
        "source_audit": source_audit,
        "case_selection": {
            "selection_snapshot_id": "2025-prebowl",
            "reference_kind": "preseason_context_v1.2",
            "team_ids": sorted(selection),
            "categories_by_team": selection,
            "inventory_path": "historical_cases.json",
            "intermediate_candidate_outputs_read": False,
        },
        "artifact_sha256": {str(p.relative_to(output)): sha256(p) for p in paths},
    }


def load_historical_network(
    directory: Path,
) -> tuple[list[Team], list[Game], dict[str, Any]]:
    """Read the exact frozen 2025 team priors and eligible game rows for execution."""
    metadata = _read_json(directory / "metadata.json")
    rows = _read_rows(directory / "included_games.csv")
    if (
        included_game_rows_sha256(rows) != metadata["included_game_rows_sha256"]
        or [r["id"] for r in rows] != metadata["included_game_ids"]
    ):
        raise ValueError("historical frozen evidence mismatch")
    teams = [
        Team(r["team_id"], r["name"], r["subdivision"], np.asarray(r["pmf"]))
        for r in json.loads((directory / "prior_pmfs.json").read_text())
    ]
    if (
        posterior_pmfs_sha256({t.team_id: t.prior for t in teams})
        != metadata["prior_pmfs_sha256"]
    ):
        raise ValueError("historical frozen prior mismatch")
    return teams, [_game_from_included_row(r) for r in rows], metadata
