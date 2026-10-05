"""Score corrected Context 1.4 against retained 2026 Context and History games."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_context_db_repair import score_game, summarize

from gippyrank.context_db_repair import (
    current_repaired_features,
    read_csv,
    sha256,
)
from gippyrank.posterior.engine import Game, Team
from gippyrank.posterior.snapshots import load_likelihood

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("data/processed/context_db_repair_172")
GAME_SOURCE = Path(
    "data/processed/context_v1_4_validation/"
    "confirmation_source_2026_10_04/completed_week_5_games.csv"
)
RETAINED = Path("data/processed/context_v1_4_validation/game_results.csv")
LIKELIHOOD = Path("data/processed/posterior/historical_likelihood_v1.json")


def _bool(value: str) -> bool:
    return value.casefold() in {"true", "1", "yes"}


def _game(row: dict[str, str]) -> Game:
    return Game(
        row["id"],
        row["homeId"],
        row["awayId"],
        row["homeClassification"].casefold(),
        row["awayClassification"].casefold(),
        int(row["homePoints"]),
        int(row["awayPoints"]),
        _bool(row["neutralSite"]),
    )


def _teams(path: Path) -> tuple[dict[str, Team], dict[str, object]]:
    metadata = json.loads((path / "metadata.json").read_text())
    names = {row["team_id"]: row for row in read_csv(path / "rankings.csv")}
    masses: dict[str, list[float]] = defaultdict(list)
    for row in read_csv(path / "posterior_pmfs.csv"):
        masses[row["team_id"]].append(float(row["probability"]))
    teams = {
        team_id: Team(
            team_id,
            names[team_id]["team_name"],
            names[team_id]["subdivision"].casefold(),
            np.asarray(values),
        )
        for team_id, values in masses.items()
    }
    return teams, metadata


def run(root: Path = ROOT) -> dict[str, object]:
    lineage = json.loads((root / OUTPUT / "lineage_audit.json").read_text())
    by_slot = {row["publication_slot"]: row for row in lineage["canonical_origins"]}
    retained = read_csv(root / RETAINED)
    sources = {row["id"]: row for row in read_csv(root / GAME_SOURCE)}
    old = [row for row in retained if row["model"] == "candidate"]
    history = [row for row in retained if row["model"] == "history_1_1"]
    if len(old) != 644 or len(history) != 644:
        raise ValueError("retained 2026 comparison must have 644 games per baseline")
    if {row["game_id"] for row in old} != {row["game_id"] for row in history}:
        raise ValueError("Context and History baseline game populations differ")
    likelihood = load_likelihood(root / LIKELIHOOD)
    corrected_rows: list[dict[str, object]] = []
    indexed_snapshots: dict[str, tuple[dict[str, Team], dict[str, object]]] = {}
    for old_row in old:
        game_id = old_row["game_id"]
        source = sources.get(game_id)
        if source is None:
            raise ValueError(f"missing retained 2026 game result: {game_id}")
        old_origin = old_row["origin"]
        slot = (
            "2026-preseason"
            if old_origin == "2026-preseason-context-1.3"
            else old_origin
        )
        if slot not in indexed_snapshots:
            path = root / by_slot[slot]["corrected_snapshot_path"]
            indexed_snapshots[slot] = _teams(path)
        teams, metadata = indexed_snapshots[slot]
        if game_id in set(metadata["included_game_ids"]):
            raise ValueError(f"{game_id}: score was included in its forecast origin")
        if metadata["requested_cutoff"] is not None:
            kickoff = datetime.fromisoformat(source["startDate"])
            cutoff = datetime.fromisoformat(metadata["requested_cutoff"])
            if kickoff <= cutoff:
                raise ValueError(f"{game_id}: forecast origin is after kickoff")
        game = _game(source)
        for team_id, subdivision in (
            (game.home_id, game.home_subdivision),
            (game.away_id, game.away_subdivision),
        ):
            if team_id not in teams:
                if subdivision != "fcs":
                    raise ValueError(f"{game_id}: FBS team absent from posterior")
                population = int(metadata["fcs_population_size"])
                teams[team_id] = Team(
                    team_id,
                    f"FCS {team_id}",
                    "fcs",
                    np.full(population, 1.0 / population),
                )
        row = score_game(game, teams, likelihood)
        row["origin"] = old_origin
        corrected_rows.append(row)
    if {row["game_id"] for row in corrected_rows} != {row["game_id"] for row in old}:
        raise ValueError("corrected Context scored a different 2026 game population")
    for source_row in retained:
        for name in (
            "margin_nll",
            "margin_mae",
            "win_brier",
            "coverage_50",
            "coverage_80",
            "coverage_95",
        ):
            source_row[name] = float(source_row[name])
    full = {
        "current_context_1_4": summarize(old),
        "corrected_context_1_4": summarize(corrected_rows),
        "history_1_1": summarize(history),
    }
    by_origin: dict[str, dict[str, object]] = {}
    for origin in sorted({row["origin"] for row in old}):
        by_origin[origin] = {
            "current_context_1_4": summarize(
                [row for row in old if row["origin"] == origin]
            ),
            "corrected_context_1_4": summarize(
                [row for row in corrected_rows if row["origin"] == origin]
            ),
            "history_1_1": summarize(
                [row for row in history if row["origin"] == origin]
            ),
        }
    _, db_evidence = current_repaired_features(root)
    old_by_id = {row["game_id"]: row for row in old}
    coverage_deltas: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in corrected_rows:
        source = sources[str(row["game_id"])]
        statuses = [
            db_evidence[source[f"{side}Id"]].status
            for side in ("home", "away")
            if source[f"{side}Classification"].casefold() == "fbs"
        ]
        stratum = "any_partial" if "partial" in statuses else "complete_or_natural_zero"
        baseline = old_by_id[row["game_id"]]
        coverage_deltas[stratum].append(
            {
                "margin_nll": float(row["margin_nll"]) - float(baseline["margin_nll"]),
                "margin_mae": float(row["margin_mae"]) - float(baseline["margin_mae"]),
                "win_brier": float(row["win_brier"]) - float(baseline["win_brier"]),
            }
        )
    coverage_strata = {
        name: {
            "games": len(items),
            **{
                f"mean_{metric}_delta_vs_current": float(
                    np.mean([item[metric] for item in items])
                )
                for metric in ("margin_nll", "margin_mae", "win_brier")
            },
        }
        for name, items in coverage_deltas.items()
    }
    prior_changes = read_csv(root / OUTPUT / "2026_prior_changes.csv")
    prior_changes_by_coverage = {
        name: {
            "teams": len(items),
            "mean_absolute_expected_rank_change": float(
                np.mean([abs(float(item["rank_change"])) for item in items])
            ),
        }
        for name in sorted({item["coverage_status"] for item in prior_changes})
        if (
            items := [item for item in prior_changes if item["coverage_status"] == name]
        )
    }
    report = {
        "game_count": 644,
        "game_population_parity": True,
        "cutoff_local_evidence_checked": True,
        "full": full,
        "by_origin": by_origin,
        "coverage_strata": coverage_strata,
        "prior_changes_by_coverage": prior_changes_by_coverage,
        "source_sha256": {
            GAME_SOURCE.as_posix(): sha256(root / GAME_SOURCE),
            RETAINED.as_posix(): sha256(root / RETAINED),
            LIKELIHOOD.as_posix(): sha256(root / LIKELIHOOD),
        },
        "corrected_snapshot_ids": {
            slot: row["corrected_snapshot_id"] for slot, row in by_slot.items()
        },
    }
    (root / OUTPUT / "current_predictive.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run()["full"], indent=2))
