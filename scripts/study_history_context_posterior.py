"""Matched History 1.1 versus Context 1.3 historical posterior study.

Run with ``uv run python scripts/study_history_context_posterior.py``. Outputs
are research artifacts; no production prior or snapshot is rewritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from inspect import signature
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from gippyrank.posterior.engine import Team, infer_posterior
from gippyrank.posterior.snapshots import (
    _scheduled_future_fcs_rows,
    add_fcs_fallbacks,
    build_snapshot,
    filter_games,
    load_pinned_likelihood,
    load_teams,
)
from gippyrank.preseason import pmf_summaries

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_preseason_context_prior_v1_3 as c13

OUT = ROOT / "data/processed/history_context_posterior_study"
SEASONS = (2022, 2023, 2024, 2025)
CONTEXT = ROOT / "data/processed/preseason/context_v1_3_candidate/predictions.csv"
COVERAGE = ROOT / "data/processed/preseason/context_v1_3_candidate/context_coverage.csv"
HISTORY = ROOT / "data/processed/preseason/history/predictions.csv"
LIKELIHOOD = ROOT / "data/processed/posterior/historical_likelihood_v1.json"
TARGETS = ROOT / "data/processed/modeling/team_season_rank_distributions.csv"
GAMES = ROOT / "data/processed/cfbd/games.csv"
GAME_FIELDS = (
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
)
BACKTEST = ROOT / "data/processed/posterior_backtest"
CONTEXT_ARMS = ("frozen_2021", "rolling_origin")
SNAPSHOT_PARAMETERS = signature(build_snapshot).parameters
INFERENCE = {
    key: SNAPSHOT_PARAMETERS[f"inference_{key}"].default
    for key in ("max_iterations", "tolerance", "damping")
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_context(season: int) -> tuple[list[Team], dict[str, dict[str, str]]]:
    rows = {
        row["team_id"]: row for row in read_csv(CONTEXT) if int(row["season"]) == season
    }
    teams = [
        Team(key, row["team_name"], "fbs", np.asarray(json.loads(row["pmf"])))
        for key, row in rows.items()
    ]
    return teams, rows


def load_rolling_context() -> dict[int, tuple[list[Team], dict[str, dict[str, str]]]]:
    """Recreate the validated P3 rolling fits, checking their retained prior scores."""
    rows, cold, _ = c13.load_candidate_rows()
    retained = {
        int(row["target_season"]): row
        for row in read_csv(c13.OUTPUT / "rolling_metrics.csv")
        if row["candidate"] == "P3"
    }
    result = {}
    for season in SEASONS:
        predictions, _ = c13.fit_rolling(
            rows,
            cold,
            context_features=c13.CONTEXT_1_3_FEATURES,
            label="P3",
            target_season=season,
        )
        actual = c13.score(predictions)
        if season not in retained or any(
            abs(actual[metric] - float(retained[season][metric])) > 1e-8
            for metric in c13.METRICS
        ):
            raise ValueError(f"{season}: rolling P3 prior fails retained parity")
        current = [item for item in predictions if item.season == season]
        teams = [
            Team(item.team_id, item.team_name, "fbs", item.pmf)
            for item in current
            if item.subdivision == "fbs"
        ]
        meta = {item.team_id: {"team_name": item.team_name} for item in current}
        result[season] = teams, meta
    return result


def load_targets(season: int, target_path: Path) -> dict[str, np.ndarray]:
    targets = {}
    for row in read_csv(target_path):
        if int(row["season"]) != season or row["subdivision"] != "fbs":
            continue
        pmf = np.zeros(int(row["team_population"]))
        for point in json.loads(row["pmf"]):
            pmf[int(point["rank"]) - 1] = float(point["probability"])
        targets[row["team_id"]] = pmf
    return targets


def fcs_population(season: int, target_path: Path) -> int:
    sizes = {
        int(row["team_population"])
        for row in read_csv(target_path)
        if int(row["season"]) == season and row["subdivision"] == "fcs"
    }
    if len(sizes) != 1:
        raise ValueError(f"{season}: final-rank corpus has no unique FCS population")
    return sizes.pop()


def transfer_groups(season: int, team_ids: set[str]) -> dict[str, str]:
    observed = {}
    for row in read_csv(COVERAGE):
        if int(row["season"]) != season or row["subdivision"] != "fbs":
            continue
        usage = row["transfer_in_prior_usage_sum_available"] == "True"
        db = row["transfer_in_prior_defensive_impact_db_sum_available"] == "True"
        observed[row["team_id"]] = "complete" if usage and db else "incomplete"
    return {key: observed.get(key, "history_fallback") for key in team_ids}


def score(pmf: np.ndarray, target: np.ndarray) -> dict[str, float]:
    summary = pmf_summaries(pmf)
    truth = pmf_summaries(target)
    ranks = np.arange(1, len(pmf) + 1)
    low, high = summary["interval_80_low"], summary["interval_80_high"]
    return {
        "nll": float(-np.sum(target * np.log(np.maximum(pmf, 1e-15)))),
        "crps": float(np.mean((np.cumsum(pmf) - np.cumsum(target)) ** 2)),
        "expected_rank_mae": abs(summary["expected_rank"] - truth["expected_rank"]),
        "interval_80_coverage": float(target[(ranks >= low) & (ranks <= high)].sum()),
        "interval_80_width": float(high - low),
        "expected_rank": summary["expected_rank"],
        "top25_probability": summary["top25_probability"],
    }


def mean(rows: list[dict[str, object]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def summarize(
    rows: list[dict[str, object]],
    season: int,
    checkpoint: int,
    group: str,
    context_arm: str,
) -> dict[str, object]:
    selected = (
        [row for row in rows if row["transfer_group"] == group]
        if group != "all"
        else rows
    )
    result: dict[str, object] = {
        "season": season,
        "checkpoint": checkpoint,
        "context_arm": context_arm,
        "transfer_group": group,
        "teams": len(selected),
    }
    for family in ("context", "history"):
        for stage in ("prior", "posterior"):
            for metric in (
                "nll",
                "crps",
                "expected_rank_mae",
                "interval_80_coverage",
                "interval_80_width",
            ):
                result[f"{family}_{stage}_{metric}"] = mean(
                    selected, f"{family}_{stage}_{metric}"
                )
        result[f"{family}_nll_improvement"] = float(
            result[f"{family}_posterior_nll"]
        ) - float(result[f"{family}_prior_nll"])
        result[f"{family}_width_contraction"] = float(
            result[f"{family}_posterior_interval_80_width"]
        ) - float(result[f"{family}_prior_interval_80_width"])
    for metric in (
        "nll",
        "crps",
        "expected_rank_mae",
        "interval_80_coverage",
        "interval_80_width",
    ):
        result[f"posterior_c_minus_h_{metric}"] = float(
            result[f"context_posterior_{metric}"]
        ) - float(result[f"history_posterior_{metric}"])
    if len(selected) > 1:
        result["posterior_expected_rank_spearman"] = float(
            spearmanr(
                [row["context_posterior_expected_rank"] for row in selected],
                [row["history_posterior_expected_rank"] for row in selected],
            ).statistic
        )
    return result


@contextmanager
def historical_game_root(raw_games: Path):
    """Reconstruct the processed game rows from cached immutable CFBD responses."""
    by_id = {}
    sources = []
    for season in SEASONS:
        for suffix in ("", "-fcs"):
            path = raw_games / f"{season}{suffix}.json"
            sources.append(path)
            for game in json.loads(path.read_text()):
                key = str(game["id"])
                if key in by_id and by_id[key] != game:
                    raise ValueError(f"conflicting cached CFBD game {key}")
                by_id[key] = game
    with tempfile.TemporaryDirectory(prefix="history-context-games-") as temporary:
        root = Path(temporary)
        path = root / "data/processed/cfbd/games.csv"
        path.parent.mkdir(parents=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=GAME_FIELDS, lineterminator="\n")
            writer.writeheader()
            for key in sorted(by_id, key=int):
                writer.writerow({field: by_id[key].get(field) for field in GAME_FIELDS})
        yield (
            root,
            path,
            {str(path.name): sha256(path)},
            {str(source): sha256(source) for source in sources},
        )


def historical(
    target_path: Path, game_root: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    likelihood = load_pinned_likelihood(LIKELIHOOD)
    team_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    rolling_context = load_rolling_context()
    for season in SEASONS:
        frozen_context, frozen_meta = load_context(season)
        contexts = {
            "frozen_2021": (frozen_context, frozen_meta),
            "rolling_origin": rolling_context[season],
        }
        history, history_meta, _ = load_teams(ROOT, season, "history")
        c_keys = {team.team_id for team in frozen_context}
        h_keys = {team.team_id for team in history if team.subdivision == "fbs"}
        targets = load_targets(season, target_path)
        season_fcs_population = fcs_population(season, target_path)
        if (
            c_keys != h_keys
            or c_keys != set(targets)
            or any(
                {team.team_id for team in arm[0]} != c_keys for arm in contexts.values()
            )
        ):
            raise ValueError(f"{season}: unmatched Context/History/target FBS keys")
        groups = transfer_groups(season, c_keys)
        old_panel = json.loads((BACKTEST / f"{season}_rolling.json").read_text())
        for checkpoint, old in enumerate(old_panel["cutoffs"], 1):
            cutoff = datetime.fromisoformat(old["cutoff"])
            games, included, excluded, _ = filter_games(
                game_root, season, cutoff, "weekly"
            )
            future_fcs = _scheduled_future_fcs_rows(game_root, season, cutoff, "weekly")
            outputs = {}
            history_replay_nll_delta = 0.0
            for family, original, meta in (
                ("frozen_2021", *contexts["frozen_2021"]),
                ("rolling_origin", *contexts["rolling_origin"]),
                ("history", history, history_meta),
            ):
                with_fcs, fallback = add_fcs_fallbacks(
                    list(original),
                    dict(meta),
                    included,
                    season_fcs_population,
                    future_fcs,
                )
                result = infer_posterior(
                    with_fcs,
                    games,
                    likelihood,
                    **INFERENCE,
                )
                if not result.converged:
                    raise RuntimeError(
                        f"nonconverged {season} checkpoint {checkpoint} {family}"
                    )
                outputs[family] = result.pmfs
                if family == "history":
                    old_nll = old["history"]["posterior"]["nll"]
                    actual_nll = float(
                        np.mean(
                            [score(result.pmfs[k], targets[k])["nll"] for k in c_keys]
                        )
                    )
                    history_replay_nll_delta = actual_nll - old_nll
                    if abs(history_replay_nll_delta) > 0.01:
                        raise ValueError(
                            f"{season}/{checkpoint}: History replay materially differs from retained panel"
                        )
                evidence_rows.append(
                    {
                        "season": season,
                        "checkpoint": checkpoint,
                        "cutoff": old["cutoff"],
                        "family": family,
                        "included_games": len(included),
                        "excluded_lower_division_games": excluded,
                        "fcs_fallbacks": len(fallback),
                        "iterations": result.iterations,
                        "converged": result.converged,
                        "history_nll_delta_vs_retained_panel": (
                            history_replay_nll_delta if family == "history" else ""
                        ),
                    }
                )
            for context_arm in CONTEXT_ARMS:
                context, context_meta = contexts[context_arm]
                prior = {
                    "context": {team.team_id: team.prior for team in context},
                    "history": {
                        team.team_id: team.prior
                        for team in history
                        if team.subdivision == "fbs"
                    },
                }
                at_checkpoint = []
                for key in sorted(c_keys):
                    row: dict[str, object] = {
                        "season": season,
                        "checkpoint": checkpoint,
                        "context_arm": context_arm,
                        "cutoff": old["cutoff"],
                        "team_id": key,
                        "team_name": context_meta[key]["team_name"],
                        "transfer_group": groups[key],
                    }
                    for family in ("context", "history"):
                        for stage, pmf in (
                            ("prior", prior[family][key]),
                            (
                                "posterior",
                                outputs[
                                    context_arm if family == "context" else "history"
                                ][key],
                            ),
                        ):
                            row.update(
                                {
                                    f"{family}_{stage}_{metric}": value
                                    for metric, value in score(
                                        pmf, targets[key]
                                    ).items()
                                }
                            )
                    row["posterior_c_minus_h_nll"] = float(
                        row["context_posterior_nll"]
                    ) - float(row["history_posterior_nll"])
                    row["posterior_c_minus_h_expected_rank"] = float(
                        row["context_posterior_expected_rank"]
                    ) - float(row["history_posterior_expected_rank"])
                    at_checkpoint.append(row)
                team_rows.extend(at_checkpoint)
                for group in ("all", "complete", "incomplete", "history_fallback"):
                    if group == "all" or any(
                        row["transfer_group"] == group for row in at_checkpoint
                    ):
                        summary_rows.append(
                            summarize(
                                at_checkpoint, season, checkpoint, group, context_arm
                            )
                        )
            print(f"completed {season} checkpoint {checkpoint}", flush=True)
    return team_rows, summary_rows, evidence_rows


def snapshot_pmfs(directory: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in read_csv(directory / "posterior_pmfs.csv"):
        values[row["team_id"]].append(float(row["probability"]))
    return {key: np.asarray(pmf) for key, pmf in values.items()}


def current_season() -> list[dict[str, object]]:
    root = ROOT / "data/processed/snapshots/2026"
    candidates: dict[tuple[str, str], tuple[Path, dict[str, object]]] = {}
    for path in root.glob("*/predictive/*/metadata.json"):
        metadata = json.loads(path.read_text())
        if metadata.get("snapshot_type") != "weekly" or not metadata.get("valid"):
            continue
        family = metadata["prior_family"]
        version = metadata["prior_model_version"]
        if (family, version) not in {("context", "1.3"), ("history", "1.1")}:
            continue
        key = (metadata["requested_cutoff"], family)
        if key not in candidates or ("history-context-1.3" in str(path)):
            candidates[key] = (path.parent, metadata)
    rows = []
    for cutoff in sorted({key[0] for key in candidates}):
        pair = [candidates.get((cutoff, family)) for family in ("context", "history")]
        if any(item is None for item in pair):
            continue
        (c_path, c_meta), (h_path, h_meta) = pair
        if read_csv(c_path / "included_games.csv") != read_csv(
            h_path / "included_games.csv"
        ):
            raise ValueError(f"2026 {cutoff}: included game evidence differs")
        for field in (
            "effective_cutoff",
            "historical_likelihood_version",
            "historical_likelihood_sha256",
            "posterior_inference_configuration",
            "fbs_team_keys_sha256",
        ):
            if field in c_meta and field in h_meta and c_meta[field] != h_meta[field]:
                raise ValueError(f"2026 {cutoff}: evidence/config mismatch in {field}")
        for metadata in (c_meta, h_meta):
            configuration = metadata.get("posterior_inference_configuration")
            if configuration is not None and any(
                configuration.get(key) != value for key, value in INFERENCE.items()
            ):
                raise ValueError(
                    f"2026 {cutoff}: production inference configuration differs"
                )
        c_pmfs, h_pmfs = snapshot_pmfs(c_path), snapshot_pmfs(h_path)
        c_prior, _, _ = load_teams(ROOT, 2026, "context")
        h_prior, _, _ = load_teams(ROOT, 2026, "history")
        fbs = {team.team_id for team in c_prior if team.subdivision == "fbs"}
        if fbs != {team.team_id for team in h_prior if team.subdivision == "fbs"}:
            raise ValueError("2026 unmatched FBS keys")
        p = {
            "context": {team.team_id: team.prior for team in c_prior},
            "history": {team.team_id: team.prior for team in h_prior},
        }
        for key in sorted(fbs):
            c, h = c_pmfs[key], h_pmfs[key]
            cs, hs = pmf_summaries(c), pmf_summaries(h)
            rows.append(
                {
                    "cutoff": cutoff,
                    "effective_cutoff": c_meta["effective_cutoff"],
                    "included_games": c_meta["included_game_count"],
                    "team_id": key,
                    "context_expected_rank": cs["expected_rank"],
                    "history_expected_rank": hs["expected_rank"],
                    "context_interval_80_width": cs["interval_80_high"]
                    - cs["interval_80_low"],
                    "history_interval_80_width": hs["interval_80_high"]
                    - hs["interval_80_low"],
                    "context_prior_interval_80_width": pmf_summaries(p["context"][key])[
                        "interval_80_high"
                    ]
                    - pmf_summaries(p["context"][key])["interval_80_low"],
                    "history_prior_interval_80_width": pmf_summaries(p["history"][key])[
                        "interval_80_high"
                    ]
                    - pmf_summaries(p["history"][key])["interval_80_low"],
                    "context_top25_probability": cs["top25_probability"],
                    "history_top25_probability": hs["top25_probability"],
                }
            )
    return rows


def write_provenance(
    target_path: Path, game_hash: str, source_hashes: dict[str, str]
) -> None:
    provenance = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (
            CONTEXT,
            COVERAGE,
            HISTORY,
            LIKELIHOOD,
            c13.HISTORICAL_TRANSFER_FEATURES,
            c13.OUTPUT / "rolling_metrics.csv",
            c13.OUTPUT / "model_spec.json",
            c13.OUTPUT / "parity_report.json",
            ROOT / "data/processed/preseason/team_season_features.csv",
        )
    }
    provenance["data/processed/modeling/team_season_rank_distributions.csv"] = sha256(
        target_path
    )
    provenance["reconstructed historical games.csv"] = game_hash
    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                "models": {
                    "context_frozen_2021": "1.3 research P3 fitted through 2021",
                    "context_rolling_origin": {
                        str(season): f"1.3 research P3 fitted through {season - 1}"
                        for season in SEASONS
                    },
                    "history": "1.1",
                    "likelihood": "Historical Likelihood V1",
                },
                "inference": INFERENCE,
                "source_sha256": provenance,
                "raw_game_source_sha256": {
                    Path(key).name: value for key, value in source_hashes.items()
                },
                "raw_coach_tenure_source_sha256": {
                    path.name: sha256(path)
                    for path in sorted(c13.c12.TENURES.glob("*.json"))
                },
                "historical_seasons": SEASONS,
                "historical_cutoffs": "exact dates from retained rolling posterior backtest",
                "current_2026": "retained matched published snapshots; descriptive only, no final-rank target",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=TARGETS)
    parser.add_argument("--raw-games", type=Path, default=ROOT / "data/raw/cfbd/games")
    args = parser.parse_args()
    target_path = args.targets.resolve()
    if not target_path.is_file():
        parser.error(
            f"final-rank target corpus is missing: {target_path}; supply --targets"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    with historical_game_root(args.raw_games.resolve()) as (
        game_root,
        game_path,
        _,
        source_hashes,
    ):
        team, summary, evidence = historical(target_path, game_root)
        game_hash = sha256(game_path)
    write_csv(OUT / "historical_teams.csv", team)
    write_csv(OUT / "historical_checkpoints.csv", summary)
    write_csv(OUT / "historical_evidence.csv", evidence)
    current = current_season()
    write_csv(OUT / "current_2026_descriptive.csv", current)
    write_provenance(target_path, game_hash, source_hashes)


if __name__ == "__main__":
    main()
