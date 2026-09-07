"""Build the research-only Performance V1 study.

This script deliberately does not register a public ranking family or call
CFBD.  It reads the already cached processed/raw game corpus, frozen H/C
preseason PMFs, and the serialized Historical Likelihood V1 artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from gippyrank.performance_v1 import (
    AnchorFamily,
    PerformanceInference,
    PerformanceMethod,
    explicit_neutralized_target,
    infer_performance,
    performance_pmf_summaries,
    pmf_tv_distance,
    remove_focal_prior,
    uniform_pmf,
)
from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    Team,
    game_margin_surface,
)
from gippyrank.posterior.snapshots import (
    add_fcs_fallbacks,
    corpus_provenance,
    load_likelihood,
    load_teams,
    sha256,
    subdivision_population_size,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/performance_v1"
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
EVALUATION_SEASONS = (2022, 2023, 2024, 2025)
CURRENT_SEASON = 2026
STANDARD_FRACTIONS = (0.0, 0.2, 0.35, 0.55, 0.72, 0.87, 1.0)
FUTURE_MODELS = ("Predictive_C", "Predictive_H", "Performance_C", "Performance_H")

# The stripping audit is a prospective design.  The target is selected from
# ordinary posterior, anchor, and graph descriptors only; no stripping error
# or explicit-neutralization result is used in selection.
PRIOR_REMOVAL_VALIDATION_RULE_VERSION = "fixed-season-phase-depth-diagnostics-v2"
PRIOR_REMOVAL_VALIDATION_CUTOFF_INDICES = (0, 1, 3, 6)
PRIOR_REMOVAL_VALIDATION_PLAN = (
    {
        "label": "zero_games",
        "cutoff_index": 0,
        "games_bucket": "0",
        "selector": "lowest_expected_rank",
    },
    {
        "label": "elite_early",
        "cutoff_index": 1,
        "games_bucket": "1-3",
        "selector": "lowest_expected_rank",
    },
    {
        "label": "middle_early",
        "cutoff_index": 1,
        "games_bucket": "1-3",
        "selector": "middle_expected_rank",
    },
    {
        "label": "weak_early",
        "cutoff_index": 1,
        "games_bucket": "1-3",
        "selector": "highest_expected_rank",
    },
    {
        "label": "narrow_mid",
        "cutoff_index": 3,
        "games_bucket": "4-6",
        "selector": "narrowest_posterior",
    },
    {
        "label": "broad_mid",
        "cutoff_index": 3,
        "games_bucket": "4-6",
        "selector": "broadest_posterior",
    },
    {
        "label": "irregular_mid",
        "cutoff_index": 3,
        "games_bucket": "4-6",
        "selector": "most_irregular_posterior",
    },
    {
        "label": "anchor_disagreement_late",
        "cutoff_index": 6,
        "games_bucket": "7+",
        "selector": "largest_context_history_pmf_tv",
    },
    {
        "label": "dense_graph_late",
        "cutoff_index": 6,
        "games_bucket": "7+",
        "selector": "highest_graph_degree",
    },
    {
        "label": "cycle_exposure_late",
        "cutoff_index": 6,
        "games_bucket": "7+",
        "selector": "highest_cycle_edge_exposure",
    },
)

# Predeclared before reading the full comparison table.  These are intentionally
# aligned with the existing small-graph BP audit rather than tuned to this run.
PRIOR_REMOVAL_P95_TV_LIMIT = 0.05
PRIOR_REMOVAL_WORST_TV_LIMIT = 0.15
_MARGIN_SURFACE_CACHE: dict[
    tuple[object, ...], tuple[np.ndarray, np.ndarray, float]
] = {}


@dataclass(frozen=True)
class Corpus:
    rows_by_season: dict[int, tuple[dict[str, str], ...]]
    source_paths_by_season: dict[int, tuple[Path, ...]]


@dataclass(frozen=True)
class PreparedCutoff:
    season: int
    requested_cutoff: datetime
    effective_cutoff: datetime
    all_rows: tuple[dict[str, str], ...]
    included_rows: tuple[dict[str, str], ...]
    games: tuple[Game, ...]
    future_games: tuple[Game, ...]
    future_rows: tuple[dict[str, str], ...]
    excluded_lower_division: int
    fcs_population_size: int | None


@dataclass(frozen=True)
class CutoffRun:
    prepared: PreparedCutoff
    context: PerformanceInference
    history: PerformanceInference
    context_rows: dict[str, dict[str, str]]
    history_rows: dict[str, dict[str, str]]
    cutoff_index: int | None
    cutoff_count: int | None


def _parse_datetime(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    return result if result.tzinfo is not None else result.replace(tzinfo=UTC)


def _normalise_raw_row(game: dict[str, object]) -> dict[str, str]:
    return {
        field: "" if game.get(field) is None else str(game.get(field))
        for field in GAME_FIELDS
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_raw_season(
    root: Path, season: int
) -> tuple[list[dict[str, str]], tuple[Path, ...]]:
    directory = root / "data/raw/cfbd/games"
    records: dict[str, dict[str, str]] = {}
    paths: list[Path] = []
    for suffix in ("", "-fcs"):
        path = directory / f"{season}{suffix}.json"
        if not path.exists():
            continue
        paths.append(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"cached CFBD schedule is not a list: {path}")
        for item in payload:
            row = _normalise_raw_row(item)
            previous = records.get(row["id"])
            if previous is not None and previous != row:
                raise ValueError(
                    f"conflicting cached schedule rows for game {row['id']}"
                )
            records[row["id"]] = row
    if not records:
        raise FileNotFoundError(
            f"No cached processed or raw game corpus is available for season {season}"
        )
    return [
        records[key] for key in sorted(records, key=lambda value: int(value))
    ], tuple(paths)


def load_corpus(root: Path, seasons: Iterable[int]) -> Corpus:
    """Load processed rows when present, otherwise use cached raw schedules.

    Raw files are only read.  The current processed corpus is never rewritten,
    so the research build cannot alter acquisition artifacts.
    """
    processed_path = root / "data/processed/cfbd/games.csv"
    processed = _read_csv_rows(processed_path) if processed_path.exists() else []
    processed_by_season: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in processed:
        processed_by_season[int(row["season"])].append(row)
    rows_by_season: dict[int, tuple[dict[str, str], ...]] = {}
    source_paths: dict[int, tuple[Path, ...]] = {}
    for season in sorted(set(seasons)):
        if processed_by_season.get(season):
            rows_by_season[season] = tuple(processed_by_season[season])
            source_paths[season] = (processed_path,)
        else:
            rows, paths = _read_raw_season(root, season)
            rows_by_season[season] = tuple(rows)
            source_paths[season] = paths
    return Corpus(rows_by_season, source_paths)


def _is_true(value: str) -> bool:
    return value.strip().casefold() in {"true", "1", "yes"}


def _valid_game_row(row: dict[str, str]) -> bool:
    return (
        row["homeClassification"].casefold() in {"fbs", "fcs"}
        and row["awayClassification"].casefold() in {"fbs", "fcs"}
        and row["homePoints"] != ""
        and row["awayPoints"] != ""
        and row["homeId"] != ""
        and row["awayId"] != ""
        and row["startDate"] != ""
    )


def _input_coverage(corpus: Corpus, root: Path) -> dict[str, object]:
    required_fields = (
        "id",
        "season",
        "startDate",
        "completed",
        "homeId",
        "awayId",
        "homeClassification",
        "awayClassification",
        "homePoints",
        "awayPoints",
    )
    result: dict[str, object] = {}
    for season in sorted(corpus.rows_by_season):
        rows = corpus.rows_by_season[season]
        completed = [row for row in rows if _is_true(row["completed"])]
        eligible = [row for row in completed if _valid_game_row(row)]
        result[str(season)] = {
            "source_mode": (
                "processed_csv"
                if any(
                    path.name == "games.csv"
                    for path in corpus.source_paths_by_season[season]
                )
                else "cached_raw_json"
            ),
            "source_paths": [
                path.relative_to(root).as_posix()
                for path in corpus.source_paths_by_season[season]
            ],
            "source_sha256": [
                sha256(path) for path in corpus.source_paths_by_season[season]
            ],
            "row_count": len(rows),
            "completed_row_count": len(completed),
            "eligible_completed_fbs_fcs_row_count": len(eligible),
            "completed_rows_not_eligible": len(completed) - len(eligible),
            "missing_required_field_counts": {
                field: sum(not row[field] for row in rows) for field in required_fields
            },
        }
    return result


def _game_from_row(row: dict[str, str]) -> Game:
    return Game(
        row["id"],
        row["homeId"],
        row["awayId"],
        row["homeClassification"].casefold(),
        row["awayClassification"].casefold(),
        int(row["homePoints"]),
        int(row["awayPoints"]),
        _is_true(row["neutralSite"]),
    )


def _prepare_cutoff(
    root: Path, corpus: Corpus, season: int, requested_cutoff: datetime
) -> PreparedCutoff:
    requested_cutoff = (
        requested_cutoff
        if requested_cutoff.tzinfo is not None
        else requested_cutoff.replace(tzinfo=UTC)
    )
    provenance = corpus_provenance(root, season)
    effective_cutoff = requested_cutoff
    if provenance.source_retrieved_at is not None:
        effective_cutoff = min(effective_cutoff, provenance.source_retrieved_at)
    all_rows = corpus.rows_by_season[season]
    included_rows: list[dict[str, str]] = []
    future_rows: list[dict[str, str]] = []
    excluded_lower = 0
    for row in all_rows:
        if not _is_true(row["completed"]) or not row["startDate"]:
            continue
        when = _parse_datetime(row["startDate"])
        if when <= effective_cutoff:
            if not _valid_game_row(row):
                excluded_lower += 1
                continue
            included_rows.append(row)
        elif _valid_game_row(row):
            future_rows.append(row)
    included_rows.sort(key=lambda row: (row["startDate"], int(row["id"])))
    future_rows.sort(key=lambda row: (row["startDate"], int(row["id"])))
    games = tuple(_game_from_row(row) for row in included_rows)
    future_games = tuple(_game_from_row(row) for row in future_rows)
    has_fcs = any(
        row[f"{side}Classification"].casefold() == "fcs"
        for row in included_rows
        for side in ("home", "away")
    )
    return PreparedCutoff(
        season,
        requested_cutoff,
        effective_cutoff,
        tuple(all_rows),
        tuple(included_rows),
        games,
        future_games,
        tuple(future_rows),
        excluded_lower,
        subdivision_population_size(root, season, "fcs") if has_fcs else None,
    )


def standard_cutoffs(rows: Iterable[dict[str, str]], season: int) -> list[datetime]:
    dates = sorted(
        {
            row["startDate"][:10]
            for row in rows
            if int(row["season"]) == season
            and row["seasonType"].casefold() == "regular"
            and row["startDate"]
        }
    )
    if len(dates) < 7:
        raise ValueError(f"not enough regular-season dates for {season}")
    indexes = [round(fraction * (len(dates) - 1)) for fraction in STANDARD_FRACTIONS]
    return [
        datetime.fromisoformat(f"{dates[index]}T23:59:59+00:00") for index in indexes
    ]


def _build_inference(
    root: Path,
    prepared: PreparedCutoff,
    family: AnchorFamily,
    likelihood: LikelihoodV1,
    method: PerformanceMethod,
    *,
    max_iterations: int,
    tolerance: float,
    damping: float,
) -> tuple[PerformanceInference, dict[str, dict[str, str]]]:
    teams, team_rows, _ = load_teams(root, prepared.season, family)
    teams, _ = add_fcs_fallbacks(
        teams,
        team_rows,
        list(prepared.included_rows),
        prepared.fcs_population_size,
    )
    inference = infer_performance(
        teams,
        list(prepared.games),
        likelihood,
        anchor_family=family,
        method=method,
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
    )
    return inference, team_rows


def build_cutoff_run(
    root: Path,
    corpus: Corpus,
    season: int,
    cutoff: datetime,
    likelihood: LikelihoodV1,
    method: PerformanceMethod,
    *,
    cutoff_index: int | None,
    cutoff_count: int | None,
    max_iterations: int,
    tolerance: float,
    damping: float,
) -> CutoffRun:
    prepared = _prepare_cutoff(root, corpus, season, cutoff)
    context, context_rows = _build_inference(
        root,
        prepared,
        "context",
        likelihood,
        method,
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
    )
    history, history_rows = _build_inference(
        root,
        prepared,
        "history",
        likelihood,
        method,
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
    )
    if (
        context.anchor_result.raw_game_factor_count
        != history.anchor_result.raw_game_factor_count
    ):
        raise AssertionError(
            "Context and History did not use the same eligible game set"
        )
    return CutoffRun(
        prepared,
        context,
        history,
        context_rows,
        history_rows,
        cutoff_index,
        cutoff_count,
    )


def _team_lookup(inference: PerformanceInference) -> dict[str, Team]:
    return {team.team_id: team for team in inference.teams}


def _week_label(rows: Iterable[dict[str, str]]) -> int | None:
    weeks = [int(row["week"]) for row in rows if row["week"].isdigit()]
    return max(weeks) if weeks else None


def _phase(run: CutoffRun) -> str:
    if run.cutoff_index is None or run.cutoff_count is None:
        return "current"
    if run.cutoff_index <= 1:
        return "early"
    if run.cutoff_index >= run.cutoff_count - 2:
        return "late"
    return "mid"


def _games_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 3:
        return "1-3"
    if count <= 6:
        return "4-6"
    return "7+"


def _local_irregularity(pmf: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(pmf, n=2))))


def _network_metrics(
    teams: Iterable[Team], games: Iterable[Game]
) -> dict[str, dict[str, int | bool]]:
    """Describe undirected schedule topology without adding a graph dependency.

    Rematches are represented by one edge, matching the posterior engine's
    pair-factor grouping.  ``cycle_edge_count`` counts incident edges that are
    not bridges; it is an exposure measure, not a count of distinct cycles.
    ``component_cycle_rank`` is E - V + 1 for the connected component.
    """
    adjacency: dict[str, set[str]] = {team.team_id: set() for team in teams}
    edges: set[tuple[str, str]] = set()
    for game in games:
        first, second = sorted((game.home_id, game.away_id))
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
        edges.add((first, second))

    component_by_team: dict[str, int] = {}
    components: list[set[str]] = []
    for start in sorted(adjacency):
        if start in component_by_team:
            continue
        component_index = len(components)
        component: set[str] = set()
        pending = [start]
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            component_by_team[current] = component_index
            pending.extend(sorted(adjacency[current] - component))
        components.append(component)

    def connected_without_edge(
        start: str, target: str, blocked: tuple[str, str]
    ) -> bool:
        seen = {start}
        pending = [start]
        while pending:
            current = pending.pop()
            for neighbor in adjacency[current]:
                if tuple(sorted((current, neighbor))) == blocked:
                    continue
                if neighbor == target:
                    return True
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        return False

    bridge_edges = {
        edge
        for edge in edges
        if not connected_without_edge(edge[0], edge[1], edge)
    }
    cycle_edges = edges - bridge_edges
    component_edges = [
        sum(first in component and second in component for first, second in edges)
        for component in components
    ]
    component_cycle_rank = [
        max(0, edge_count - len(component) + 1)
        for component, edge_count in zip(components, component_edges, strict=True)
    ]

    result: dict[str, dict[str, int | bool]] = {}
    for team_id, opponents in adjacency.items():
        component_index = component_by_team[team_id]
        result[team_id] = {
            "unique_opponent_count": len(opponents),
            "graph_degree": len(opponents),
            "component_node_count": len(components[component_index]),
            "component_edge_count": component_edges[component_index],
            "component_cycle_rank": component_cycle_rank[component_index],
            "cycle_edge_count": sum(
                tuple(sorted((team_id, opponent))) in cycle_edges
                for opponent in opponents
            ),
            "cycle_participation": any(
                tuple(sorted((team_id, opponent))) in cycle_edges
                for opponent in opponents
            ),
        }
    return result


def _anchor_decomposition(
    run: CutoffRun, team_id: str, network: dict[str, dict[str, int | bool]]
) -> dict[str, object]:
    """Return row-level descriptors for explaining C/H anchor sensitivity."""
    context_teams = _team_lookup(run.context)
    history_teams = _team_lookup(run.history)
    opponents = {
        game.away_id if game.home_id == team_id else game.home_id
        for game in run.prepared.games
        if team_id in {game.home_id, game.away_id}
    }
    posterior_tvs: list[float] = []
    posterior_expected_differences: list[float] = []
    prior_tvs: list[float] = []
    for opponent_id in sorted(opponents):
        context_pmf = run.context.anchor_result.pmfs.get(opponent_id)
        history_pmf = run.history.anchor_result.pmfs.get(opponent_id)
        context_team = context_teams.get(opponent_id)
        history_team = history_teams.get(opponent_id)
        if context_pmf is not None and history_pmf is not None:
            posterior_tvs.append(pmf_tv_distance(context_pmf, history_pmf))
            posterior_expected_differences.append(
                abs(
                    float(
                        np.dot(
                            np.arange(1, len(context_pmf) + 1), context_pmf
                        )
                    )
                    - float(
                        np.dot(
                            np.arange(1, len(history_pmf) + 1), history_pmf
                        )
                    )
                )
            )
        if context_team is not None and history_team is not None:
            prior_tvs.append(pmf_tv_distance(context_team.prior, history_team.prior))

    def total(values: list[float]) -> float:
        return float(np.sum(values)) if values else 0.0

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def maximum(values: list[float]) -> float:
        return float(np.max(values)) if values else 0.0

    posterior_tv_sum = total(posterior_tvs)
    posterior_tv_max = maximum(posterior_tvs)
    return {
        **network.get(
            team_id,
            {
                "unique_opponent_count": 0,
                "graph_degree": 0,
                "component_node_count": 1,
                "component_edge_count": 0,
                "component_cycle_rank": 0,
                "cycle_edge_count": 0,
                "cycle_participation": False,
            },
        ),
        "opponent_anchor_count": len(posterior_tvs),
        "opponent_anchor_posterior_tv_mean": mean(posterior_tvs),
        "opponent_anchor_posterior_tv_sum": posterior_tv_sum,
        "opponent_anchor_posterior_tv_max": posterior_tv_max,
        "opponent_anchor_posterior_tv_max_share": (
            posterior_tv_max / posterior_tv_sum if posterior_tv_sum else 0.0
        ),
        "opponent_anchor_expected_rank_difference_mean": mean(
            posterior_expected_differences
        ),
        "opponent_anchor_expected_rank_difference_sum": total(
            posterior_expected_differences
        ),
        "opponent_anchor_expected_rank_difference_max": maximum(
            posterior_expected_differences
        ),
        "opponent_prior_pmf_tv_mean": mean(prior_tvs),
        "opponent_prior_pmf_tv_sum": total(prior_tvs),
        "opponent_prior_pmf_tv_max": maximum(prior_tvs),
    }


def _validation_target_metrics(
    run: CutoffRun, team_id: str
) -> dict[str, object]:
    summary = performance_pmf_summaries(run.context.anchor_result.pmfs[team_id])
    network = _network_metrics(run.context.teams, run.prepared.games)
    metrics = _anchor_decomposition(run, team_id, network)
    metrics.update(
        {
            "expected_rank": summary["expected_rank"],
            "interval_80_width": int(summary["interval_80_high"])
            - int(summary["interval_80_low"]),
            "local_irregularity": _local_irregularity(
                run.context.anchor_result.pmfs[team_id]
            ),
            "context_history_pmf_tv": pmf_tv_distance(
                run.context.pmfs[team_id], run.history.pmfs[team_id]
            ),
            "context_history_expected_rank_difference": float(
                performance_pmf_summaries(run.context.pmfs[team_id])["expected_rank"]
                - performance_pmf_summaries(run.history.pmfs[team_id])["expected_rank"]
            ),
            "games_played": run.context.eligible_game_counts[team_id],
        }
    )
    return metrics


def _rank_row(
    inference: PerformanceInference,
    team: Team,
    team_rows: dict[str, dict[str, str]],
    run: CutoffRun,
) -> dict[str, object]:
    summary = performance_pmf_summaries(inference.pmfs[team.team_id])
    games = inference.eligible_game_counts[team.team_id]
    return {
        "season": run.prepared.season,
        "cutoff": run.prepared.effective_cutoff.isoformat(),
        "anchor_family": inference.anchor_family,
        "method": inference.method,
        "team_id": team.team_id,
        "team_name": team.name,
        "subdivision": team.subdivision,
        "conference": team_rows.get(team.team_id, {}).get("conference", ""),
        "rated": games > 0,
        "games_played": games,
        "eligible_evidence_count": games,
        **summary,
        "interval_50_width": int(summary["interval_50_high"])
        - int(summary["interval_50_low"]),
        "interval_80_width": int(summary["interval_80_high"])
        - int(summary["interval_80_low"]),
        "interval_95_width": int(summary["interval_95_high"])
        - int(summary["interval_95_low"]),
        "week_through_cutoff": _week_label(run.prepared.included_rows),
    }


def ranking_rows(
    inference: PerformanceInference,
    team_rows: dict[str, dict[str, str]],
    run: CutoffRun,
) -> list[dict[str, object]]:
    rows = [
        _rank_row(inference, team, team_rows, run)
        for team in inference.teams
        if team.subdivision == "fbs"
    ]
    rows.sort(
        key=lambda row: (
            not bool(row["rated"]),
            float(row["expected_rank"]),
            str(row["team_name"]),
            str(row["team_id"]),
        )
    )
    display_rank = 0
    for row in rows:
        if row["rated"]:
            display_rank += 1
            row["display_rank"] = display_rank
        else:
            row["display_rank"] = None
    return rows


def pmf_rows(
    inference: PerformanceInference, run: CutoffRun
) -> list[dict[str, object]]:
    rows = []
    for team in inference.teams:
        if team.subdivision != "fbs":
            continue
        for rank, probability in enumerate(inference.pmfs[team.team_id], 1):
            rows.append(
                {
                    "season": run.prepared.season,
                    "cutoff": run.prepared.effective_cutoff.isoformat(),
                    "anchor_family": inference.anchor_family,
                    "method": inference.method,
                    "team_id": team.team_id,
                    "team_name": team.name,
                    "rated": inference.eligible_game_counts[team.team_id] > 0,
                    "games_played": inference.eligible_game_counts[team.team_id],
                    "rank": rank,
                    "probability": float(probability),
                }
            )
    return rows


def evidence_rows(
    inference: PerformanceInference,
    team_rows: dict[str, dict[str, str]],
    run: CutoffRun,
) -> list[dict[str, object]]:
    by_id = _team_lookup(inference)
    source_rows = {row["id"]: row for row in run.prepared.included_rows}
    result = []
    for game in run.prepared.games:
        row = source_rows[game.game_id]
        for focal_id in (game.home_id, game.away_id):
            focal = by_id.get(focal_id)
            if focal is None or focal.subdivision != "fbs":
                continue
            opponent_id = game.away_id if focal_id == game.home_id else game.home_id
            opponent = by_id[opponent_id]
            focal_home = focal_id == game.home_id
            focal_points = game.home_points if focal_home else game.away_points
            opponent_points = game.away_points if focal_home else game.home_points
            cross = game.home_subdivision != game.away_subdivision
            if cross:
                model_margin = (
                    game.home_points - game.away_points
                    if game.home_subdivision == "fbs"
                    else game.away_points - game.home_points
                )
            else:
                model_margin = game.home_points - game.away_points
            opponent_anchor = performance_pmf_summaries(opponent.prior)
            opponent_joint = inference.anchor_result.pmfs[opponent_id]
            result.append(
                {
                    "season": run.prepared.season,
                    "cutoff": run.prepared.effective_cutoff.isoformat(),
                    "game_id": game.game_id,
                    "team_id": focal.team_id,
                    "team_name": focal.name,
                    "opponent_id": opponent.team_id,
                    "opponent_name": opponent.name,
                    "site": "neutral"
                    if game.neutral_site
                    else ("home" if focal_home else "away"),
                    "focal_subdivision": focal.subdivision,
                    "opponent_subdivision": opponent.subdivision,
                    "home_points": game.home_points,
                    "away_points": game.away_points,
                    "focal_points": focal_points,
                    "opponent_points": opponent_points,
                    "focal_margin": focal_points - opponent_points,
                    "model_oriented_margin": model_margin,
                    "opponent_anchor_family": inference.anchor_family,
                    "opponent_anchor_expected_rank": opponent_anchor["expected_rank"],
                    "opponent_anchor_median_rank": opponent_anchor["median_rank"],
                    "opponent_joint_expected_rank": float(
                        np.dot(np.arange(1, len(opponent_joint) + 1), opponent_joint)
                    ),
                    "focal_performance_expected_rank": performance_pmf_summaries(
                        inference.pmfs[focal.team_id]
                    )["expected_rank"],
                    "eligible_evidence_count": inference.eligible_game_counts[
                        focal.team_id
                    ],
                    "week": row["week"],
                    "season_type": row["seasonType"],
                }
            )
    return result


def _prediction_for_game(
    game: Game,
    focal_id: str,
    home: Team,
    away: Team,
    home_pmf: np.ndarray,
    away_pmf: np.ndarray,
    likelihood: LikelihoodV1,
) -> dict[str, float | int | None]:
    cache_key = (
        game.game_id,
        game.home_points,
        game.away_points,
        game.neutral_site,
        game.home_subdivision,
        game.away_subdivision,
        len(home.prior),
        len(away.prior),
        likelihood.beta.tobytes(),
        likelihood.scale,
        likelihood.degrees_of_freedom,
    )
    surface = _MARGIN_SURFACE_CACHE.get(cache_key)
    if surface is None:
        surface = game_margin_surface(game, home, away, likelihood)
        _MARGIN_SURFACE_CACHE[cache_key] = surface
    locations, density, actual_model_margin = surface
    joint = home_pmf[:, None] * away_pmf[None, :]
    marginal_density = float(np.sum(joint * density))
    expected_model_margin = float(np.sum(joint * locations))
    win_probability_model_orientation = float(
        np.sum(
            joint
            * student_t.sf(
                (0.0 - locations) / likelihood.scale,
                likelihood.degrees_of_freedom,
            )
        )
    )
    if game.home_subdivision != game.away_subdivision:
        focal_is_fbs = (
            game.home_id == focal_id and game.home_subdivision == "fbs"
        ) or (game.away_id == focal_id and game.away_subdivision == "fbs")
        if not focal_is_fbs:
            raise ValueError(
                "future validation focal must be FBS in a cross-subdivision game"
            )
        focal_expected_margin = expected_model_margin
        focal_win_probability = win_probability_model_orientation
        focal_actual_margin = actual_model_margin
    elif game.home_id == focal_id:
        focal_expected_margin = expected_model_margin
        focal_win_probability = win_probability_model_orientation
        focal_actual_margin = float(game.home_points - game.away_points)
    else:
        focal_expected_margin = -expected_model_margin
        focal_win_probability = float(
            np.sum(
                joint
                * student_t.cdf(
                    (0.0 - locations) / likelihood.scale,
                    likelihood.degrees_of_freedom,
                )
            )
        )
        focal_actual_margin = float(game.away_points - game.home_points)
    actual_win = (
        1 if focal_actual_margin > 0 else 0 if focal_actual_margin < 0 else None
    )
    return {
        "marginalized_nll": float(-math.log(max(marginal_density, 1e-300))),
        "predicted_margin": focal_expected_margin,
        "actual_margin": focal_actual_margin,
        "margin_absolute_error": abs(focal_expected_margin - focal_actual_margin),
        "win_probability": focal_win_probability,
        "actual_win": actual_win,
        "brier": (
            (focal_win_probability - actual_win) ** 2
            if actual_win is not None
            else None
        ),
        "actual_model_margin": actual_model_margin,
    }


def _pmf_for_team(
    model: str,
    team_id: str,
    inference: PerformanceInference,
    teams: dict[str, Team],
    fcs_population_size: int | None,
) -> tuple[Team, np.ndarray] | None:
    team = teams.get(team_id)
    pmfs = (
        inference.anchor_result.pmfs
        if model.startswith("Predictive")
        else inference.pmfs
    )
    if team is None:
        if fcs_population_size is None:
            return None
        team = Team(team_id, team_id, "fcs", uniform_pmf(fcs_population_size))
    pmf = pmfs.get(team_id)
    if pmf is None:
        if team.subdivision != "fcs":
            return None
        pmf = uniform_pmf(len(team.prior))
    return team, pmf


def future_prediction_rows(
    run: CutoffRun, likelihood: LikelihoodV1
) -> list[dict[str, object]]:
    models = (
        (FUTURE_MODELS[0], run.context),
        (FUTURE_MODELS[1], run.history),
        (FUTURE_MODELS[2], run.context),
        (FUTURE_MODELS[3], run.history),
    )
    context_teams = _team_lookup(run.context)
    history_teams = _team_lookup(run.history)
    next_game_ids: set[tuple[str, str]] = set()
    for focal_id in sorted(
        {
            team_id
            for game in run.prepared.future_games
            for team_id, subdivision in (
                (game.home_id, game.home_subdivision),
                (game.away_id, game.away_subdivision),
            )
            if subdivision == "fbs"
        }
    ):
        candidates = [
            game
            for game in run.prepared.future_games
            if focal_id in {game.home_id, game.away_id}
        ]
        if candidates:
            next_game_ids.add((focal_id, candidates[0].game_id))
    source_rows = {row["id"]: row for row in run.prepared.future_rows}
    rows = []
    for model, inference in models:
        teams = context_teams if inference.anchor_family == "context" else history_teams
        for game in run.prepared.future_games:
            fbs_ids = [
                team_id
                for team_id, subdivision in (
                    (game.home_id, game.home_subdivision),
                    (game.away_id, game.away_subdivision),
                )
                if subdivision == "fbs"
            ]
            for focal_id in fbs_ids:
                home_value = _pmf_for_team(
                    model,
                    game.home_id,
                    inference,
                    teams,
                    run.prepared.fcs_population_size,
                )
                away_value = _pmf_for_team(
                    model,
                    game.away_id,
                    inference,
                    teams,
                    run.prepared.fcs_population_size,
                )
                if home_value is None or away_value is None:
                    continue
                home, home_pmf = home_value
                away, away_pmf = away_value
                scores = _prediction_for_game(
                    game, focal_id, home, away, home_pmf, away_pmf, likelihood
                )
                focal_team = teams.get(focal_id)
                if focal_team is None:
                    continue
                games_played = inference.eligible_game_counts.get(focal_id, 0)
                row = source_rows[game.game_id]
                rows.append(
                    {
                        "season": run.prepared.season,
                        "cutoff": run.prepared.effective_cutoff.isoformat(),
                        "season_phase": _phase(run),
                        "model": model,
                        "game_id": game.game_id,
                        "focal_team_id": focal_id,
                        "focal_team_name": focal_team.name,
                        "opponent_id": game.away_id
                        if focal_id == game.home_id
                        else game.home_id,
                        "games_played": games_played,
                        "games_played_bucket": _games_bucket(games_played),
                        "is_next_game": (focal_id, game.game_id) in next_game_ids,
                        "game_start_date": row["startDate"],
                        "site": (
                            "neutral"
                            if game.neutral_site
                            else "home"
                            if focal_id == game.home_id
                            else "away"
                        ),
                        "week": row["week"],
                        **scores,
                    }
                )
    return rows


def _future_scoring_key(row: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row["season"]),
        str(row["cutoff"]),
        str(row["game_id"]),
        str(row["focal_team_id"]),
    )


def validate_common_future_keys(rows: list[dict[str, object]]) -> dict[str, object]:
    """Require all four model variants to score identical team-game keys."""
    by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(row)
    missing_models = [model for model in FUTURE_MODELS if model not in by_model]
    unexpected_models = sorted(set(by_model) - set(FUTURE_MODELS))
    if missing_models or unexpected_models:
        raise ValueError(
            "future validation models do not match the declared model panel: "
            f"missing={missing_models}, unexpected={unexpected_models}"
        )
    keys_by_model: dict[str, set[tuple[str, str, str, str]]] = {}
    duplicate_counts: dict[str, int] = {}
    for model in FUTURE_MODELS:
        keys = [_future_scoring_key(row) for row in by_model[model]]
        duplicate_counts[model] = len(keys) - len(set(keys))
        if duplicate_counts[model]:
            raise ValueError(f"future validation has duplicate keys for {model}")
        keys_by_model[model] = set(keys)
    reference = keys_by_model[FUTURE_MODELS[0]]
    differing_models = {
        model: {
            "missing_from_reference": len(reference - keys),
            "extra_against_reference": len(keys - reference),
        }
        for model, keys in keys_by_model.items()
        if keys != reference
    }
    if differing_models:
        raise ValueError(
            "future validation models do not share strict common scoring keys: "
            f"{differing_models}"
        )
    return {
        "models": list(FUTURE_MODELS),
        "key_count_per_model": {
            model: len(keys_by_model[model]) for model in FUTURE_MODELS
        },
        "common_key_count": len(reference),
        "all_models_share_identical_keys": True,
        "duplicate_key_counts": duplicate_counts,
    }


def future_validation_population(
    runs: list[CutoffRun],
    predictions: list[dict[str, object]],
    key_audit: dict[str, object],
) -> dict[str, object]:
    """Summarize candidate, scored, excluded, and next-game populations."""
    candidate_game_instances: set[tuple[str, str, str]] = set()
    candidate_keys: set[tuple[str, str, str, str]] = set()
    by_season: dict[int, dict[str, int]] = defaultdict(
        lambda: {
            "candidate_future_game_instances": 0,
            "candidate_fbs_team_game_keys": 0,
            "scored_fbs_team_game_keys": 0,
            "next_game_keys": 0,
        }
    )
    for run in runs:
        cutoff = run.prepared.effective_cutoff.isoformat()
        season_counts = by_season[run.prepared.season]
        for game in run.prepared.future_games:
            candidate_game_instances.add(
                (str(run.prepared.season), cutoff, game.game_id)
            )
            season_counts["candidate_future_game_instances"] += 1
            for team_id, subdivision in (
                (game.home_id, game.home_subdivision),
                (game.away_id, game.away_subdivision),
            ):
                if subdivision != "fbs":
                    continue
                key = (str(run.prepared.season), cutoff, game.game_id, team_id)
                candidate_keys.add(key)
                season_counts["candidate_fbs_team_game_keys"] += 1

    reference_model = FUTURE_MODELS[0]
    scored_rows = [
        row for row in predictions if row["model"] == reference_model
    ]
    scored_keys = {_future_scoring_key(row) for row in scored_rows}
    for row in scored_rows:
        counts = by_season[int(row["season"])]
        counts["scored_fbs_team_game_keys"] += 1
        if row["is_next_game"]:
            counts["next_game_keys"] += 1
    return {
        "seasons": sorted(by_season),
        "cutoff_count": len(
            {(str(run.prepared.season), run.prepared.effective_cutoff.isoformat()) for run in runs}
        ),
        "cutoffs": sorted(
            {
                f"{run.prepared.season}:{run.prepared.effective_cutoff.isoformat()}"
                for run in runs
            }
        ),
        "candidate_future_game_instance_count": len(candidate_game_instances),
        "candidate_fbs_team_game_key_count": len(candidate_keys),
        "scored_fbs_team_game_key_count": len(scored_keys),
        "excluded_fbs_team_game_key_count": len(candidate_keys - scored_keys),
        "next_game_key_count": sum(
            int(row["is_next_game"]) for row in scored_rows
        ),
        "later_future_key_count": sum(
            not bool(row["is_next_game"]) for row in scored_rows
        ),
        "model_row_counts": {
            str(model): sum(1 for row in predictions if row["model"] == model)
            for model in FUTURE_MODELS
        },
        "strict_common_key_audit": key_audit,
        "by_season": [
            {"season": season, **counts}
            for season, counts in sorted(by_season.items())
        ],
        "missingness_policy": (
            "Candidate keys are completed, valid FBS/FCS rows after the effective "
            "cutoff. A key is excluded only when a model cannot construct the "
            "required team PMF; the strict common-key check fails if model panels "
            "then differ."
        ),
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def aggregate_future_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        horizons = ("all_future", "next_game") if row["is_next_game"] else ("all_future",)
        for horizon in horizons:
            key = (
                row["season"],
                row["cutoff"],
                row["season_phase"],
                row["model"],
                horizon,
                row["games_played_bucket"],
            )
            groups[key].append(row)
    result = []
    for key, group in sorted(
        groups.items(), key=lambda item: tuple(str(v) for v in item[0])
    ):
        predictions = [float(row["win_probability"]) for row in group]
        outcomes = [row["actual_win"] for row in group if row["actual_win"] is not None]
        brier = [float(row["brier"]) for row in group if row["brier"] is not None]
        bins: list[tuple[list[float], list[int]]] = [([], []) for _ in range(10)]
        for row in group:
            if row["actual_win"] is None:
                continue
            index = min(9, int(float(row["win_probability"]) * 10))
            bins[index][0].append(float(row["win_probability"]))
            bins[index][1].append(int(row["actual_win"]))
        calibration_error = sum(
            len(predicted) * abs(float(np.mean(predicted)) - float(np.mean(observed)))
            for predicted, observed in bins
            if predicted
        ) / max(len(outcomes), 1)
        season, cutoff, phase, model, horizon, bucket = key
        result.append(
            {
                "season": season,
                "cutoff": cutoff,
                "season_phase": phase,
                "model": model,
                "horizon": horizon,
                "games_played_bucket": bucket,
                "prediction_count": len(group),
                "brier_count": len(brier),
                "marginalized_nll": _mean(
                    [float(row["marginalized_nll"]) for row in group]
                ),
                "margin_absolute_error": _mean(
                    [float(row["margin_absolute_error"]) for row in group]
                ),
                "win_brier": _mean(brier),
                "calibration_absolute_error": calibration_error,
                "mean_win_probability": _mean(predictions),
                "empirical_win_rate": _mean([float(value) for value in outcomes]),
            }
        )
    return result


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


ARTIFACT_PURPOSES = {
    "anchor_sensitivity.csv": (
        "historical Context/History sensitivity and row-level decomposition",
        True,
        True,
    ),
    "anchor_sensitivity_decomposition.csv": (
        "aggregated sensitivity decomposition by evidence and network descriptors",
        True,
        True,
    ),
    "bp_prior_removal_validation.csv": (
        "prospective prior-stripping versus explicit-neutralization audit cases",
        True,
        True,
    ),
    "current_anchor_sensitivity.csv": (
        "latest current-season Context/History sensitivity",
        False,
        True,
    ),
    "current_performance_vs_predictive.csv": (
        "latest current-season Performance versus Predictive comparison",
        False,
        True,
    ),
    "current_pmfs.csv": (
        "latest current-season Context Performance PMFs",
        True,
        True,
    ),
    "current_pmfs_history.csv": (
        "latest current-season History Performance PMFs",
        True,
        True,
    ),
    "current_rankings.csv": (
        "latest current-season Context Performance summaries",
        True,
        True,
    ),
    "current_rankings_history.csv": (
        "latest current-season History Performance summaries",
        True,
        True,
    ),
    "future_game_predictions.csv": (
        "row-level future predictions for leakage and scoring-key audit",
        True,
        True,
    ),
    "future_game_validation.csv": (
        "aggregated future-game validation metrics",
        True,
        True,
    ),
    "game_evidence.csv": (
        "game-level evidence and opponent-anchor provenance",
        True,
        True,
    ),
    "idle_team_examples.csv": (
        "examples of network updates without new focal games",
        False,
        True,
    ),
    "illustrative_cases.csv": (
        "deterministic corpus examples for interpretation",
        False,
        True,
    ),
    "performance_vs_predictive.csv": (
        "historical Performance versus Predictive comparison",
        True,
        True,
    ),
    "report.md": (
        "human-readable generated research report",
        True,
        True,
    ),
    "summary.json": (
        "machine-readable study summary, provenance, and diagnostics",
        True,
        True,
    ),
    "plots/anchor_sensitivity_by_games.png": (
        "anchor sensitivity visualization",
        False,
        True,
    ),
    "plots/current_uncertainty_by_games.png": (
        "current Performance uncertainty visualization",
        False,
        True,
    ),
    "plots/future_validation_by_games.png": (
        "future validation visualization",
        False,
        True,
    ),
    "plots/performance_vs_predictive_by_games.png": (
        "Performance versus Predictive visualization",
        False,
        True,
    ),
}


def _artifact_inventory(
    output: Path, *, exclude: frozenset[str] = frozenset()
) -> list[dict[str, object]]:
    """Measure generated bundle members without including the inventory itself."""
    inventory = []
    for path in sorted(path for path in output.rglob("*") if path.is_file()):
        relative = path.relative_to(output).as_posix()
        if relative == "artifact_inventory.csv" or relative in exclude:
            continue
        purpose, reproducible, audit = ARTIFACT_PURPOSES.get(
            relative,
            ("generated research artifact", False, True),
        )
        row_count: int | None = None
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                row_count = max(sum(1 for _ in handle) - 1, 0)
        inventory.append(
            {
                "artifact": relative,
                "bytes": path.stat().st_size,
                "rows": row_count,
                "purpose": purpose,
                "generated": True,
                "needed_for_reproducibility": reproducible,
                "needed_for_audit": audit,
            }
        )
    return inventory


def _summary_for_rows(
    rows: list[dict[str, object]], value_key: str
) -> dict[str, float | int | None]:
    values = [float(row[value_key]) for row in rows if row[value_key] is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _validation_case_targets(
    runs: list[CutoffRun],
) -> list[tuple[str, CutoffRun, str]]:
    """Select the fixed audit panel without looking at stripping errors."""
    by_season_and_index = {
        (run.prepared.season, run.cutoff_index): run
        for run in runs
        if run.cutoff_index is not None
    }
    selected: dict[tuple[int, str, str], list[str]] = defaultdict(list)
    selected_runs: dict[tuple[int, str, str], CutoffRun] = {}
    for season in EVALUATION_SEASONS:
        for plan in PRIOR_REMOVAL_VALIDATION_PLAN:
            run = by_season_and_index.get((season, plan["cutoff_index"]))
            if run is None:
                continue
            candidates = [
                team
                for team in run.context.teams
                if team.subdivision == "fbs"
                and _games_bucket(
                    run.context.eligible_game_counts[team.team_id]
                )
                == plan["games_bucket"]
            ]
            if not candidates:
                continue
            metrics = {
                team.team_id: _validation_target_metrics(run, team.team_id)
                for team in candidates
            }
            selector = plan["selector"]
            if selector == "middle_expected_rank":
                ordered = sorted(
                    candidates,
                    key=lambda team: (
                        float(metrics[team.team_id]["expected_rank"]),
                        team.name,
                        team.team_id,
                    ),
                )
                target = ordered[(len(ordered) - 1) // 2]
            else:
                descending = selector not in {
                    "lowest_expected_rank",
                    "narrowest_posterior",
                }
                metric_key = {
                    "lowest_expected_rank": "expected_rank",
                    "highest_expected_rank": "expected_rank",
                    "narrowest_posterior": "interval_80_width",
                    "broadest_posterior": "interval_80_width",
                    "most_irregular_posterior": "local_irregularity",
                    "largest_context_history_pmf_tv": "context_history_pmf_tv",
                    "highest_graph_degree": "graph_degree",
                    "highest_cycle_edge_exposure": "cycle_edge_count",
                }[selector]
                target = min(
                    candidates,
                    key=lambda team: (
                        (-1 if descending else 1)
                        * float(metrics[team.team_id][metric_key]),
                        team.name,
                        team.team_id,
                    ),
                )
            key = (season, run.prepared.effective_cutoff.isoformat(), target.team_id)
            selected[key].append(str(plan["label"]))
            selected_runs[key] = run
    return [
        (";".join(selected[key]), selected_runs[key], key[2])
        for key in sorted(selected, key=lambda value: (value[0], value[1], value[2]))
    ]


def _validation_group_summary(
    rows: list[dict[str, object]], group_key: str
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    return [
        {
            "group": group,
            "count": len(items),
            "median_tv": float(np.median([float(item["tv_distance"]) for item in items])),
            "p95_tv": float(
                np.quantile([float(item["tv_distance"]) for item in items], 0.95)
            ),
            "max_tv": float(np.max([float(item["tv_distance"]) for item in items])),
            "median_abs_expected_rank_difference": float(
                np.median(
                    [
                        abs(float(item["expected_rank_difference"]))
                        for item in items
                    ]
                )
            ),
        }
        for group, items in sorted(grouped.items())
    ]


def run_prior_removal_validation(
    runs: list[CutoffRun],
    likelihood: LikelihoodV1,
    *,
    max_iterations: int,
    tolerance: float,
    damping: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cases = _validation_case_targets(runs)
    rows: list[dict[str, object]] = []
    for label, run, target_id in cases:
        team = next(team for team in run.context.teams if team.team_id == target_id)
        target_metrics = _validation_target_metrics(run, target_id)
        stripped = remove_focal_prior(
            run.context.anchor_result.pmfs[target_id], team.prior
        )
        explicit, explicit_result = explicit_neutralized_target(
            run.context.teams,
            run.prepared.games,
            likelihood,
            target_id,
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
        )
        stripped_summary = performance_pmf_summaries(stripped)
        explicit_summary = performance_pmf_summaries(explicit)
        rows.append(
            {
                "season": run.prepared.season,
                "cutoff": run.prepared.effective_cutoff.isoformat(),
                "case_label": label,
                "team_id": target_id,
                "team_name": team.name,
                "games_played": run.context.eligible_game_counts[target_id],
                "selection_rule": label,
                "unique_opponent_count": target_metrics["unique_opponent_count"],
                "graph_degree": target_metrics["graph_degree"],
                "component_cycle_rank": target_metrics["component_cycle_rank"],
                "cycle_edge_count": target_metrics["cycle_edge_count"],
                "context_history_pmf_tv": target_metrics["context_history_pmf_tv"],
                "context_history_expected_rank_difference": target_metrics[
                    "context_history_expected_rank_difference"
                ],
                "expected_rank_stripped": stripped_summary["expected_rank"],
                "expected_rank_explicit_neutralized": explicit_summary["expected_rank"],
                "expected_rank_difference": float(
                    stripped_summary["expected_rank"]
                    - explicit_summary["expected_rank"]
                ),
                "tv_distance": pmf_tv_distance(stripped, explicit),
                "median_difference": int(stripped_summary["median_rank"])
                - int(explicit_summary["median_rank"]),
                "interval_80_low_difference": int(stripped_summary["interval_80_low"])
                - int(explicit_summary["interval_80_low"]),
                "interval_80_high_difference": int(stripped_summary["interval_80_high"])
                - int(explicit_summary["interval_80_high"]),
                "interval_80_width_difference": (
                    int(stripped_summary["interval_80_high"])
                    - int(stripped_summary["interval_80_low"])
                    - int(explicit_summary["interval_80_high"])
                    + int(explicit_summary["interval_80_low"])
                ),
                "explicit_iterations": explicit_result.iterations,
                "explicit_converged": explicit_result.converged,
            }
        )
    tv = [float(row["tv_distance"]) for row in rows]
    coverage = {
        "rule_version": PRIOR_REMOVAL_VALIDATION_RULE_VERSION,
        "cutoff_indices": list(PRIOR_REMOVAL_VALIDATION_CUTOFF_INDICES),
        "requested_case_roles": [
            str(plan["label"]) for plan in PRIOR_REMOVAL_VALIDATION_PLAN
        ],
        "case_count": len(rows),
        "unique_team_season_case_count": len(
            {(row["season"], row["team_id"]) for row in rows}
        ),
        "seasons": sorted({int(row["season"]) for row in rows}),
        "season_phases": sorted(
            {
                _phase(run)
                for _label, run, _target_id in cases
            }
        ),
        "games_played_buckets": sorted(
            {_games_bucket(int(row["games_played"])) for row in rows}
        ),
        "represented_case_roles": sorted(
            {
                role
                for row in rows
                for role in str(row["selection_rule"]).split(";")
            }
        ),
        "cutoffs": sorted({str(row["cutoff"]) for row in rows}),
    }
    summary = {
        "case_count": len(rows),
        "coverage": coverage,
        "predeclared_p95_tv_limit": PRIOR_REMOVAL_P95_TV_LIMIT,
        "predeclared_worst_tv_limit": PRIOR_REMOVAL_WORST_TV_LIMIT,
        "median_tv": float(np.median(tv)),
        "p95_tv": float(np.quantile(tv, 0.95)),
        "worst_tv": float(np.max(tv)),
        "median_abs_expected_rank_difference": float(
            np.median(np.abs([float(row["expected_rank_difference"]) for row in rows]))
        ),
        "worst_abs_expected_rank_difference": float(
            np.max(np.abs([float(row["expected_rank_difference"]) for row in rows]))
        ),
        "by_season": _validation_group_summary(rows, "season"),
        "by_games_played_bucket": [
            {
                **group,
                "group": group["group"],
            }
            for group in _validation_group_summary(
                [
                    {**row, "games_played_bucket": _games_bucket(int(row["games_played"]))}
                    for row in rows
                ],
                "games_played_bucket",
            )
        ],
        "by_season_phase": _validation_group_summary(
            [
                {
                    **row,
                    "season_phase": _phase(
                        next(
                            run
                            for _label, run, target_id in cases
                            if target_id == row["team_id"]
                            and run.prepared.season == row["season"]
                            and run.prepared.effective_cutoff.isoformat()
                            == row["cutoff"]
                        )
                    ),
                }
                for row in rows
            ],
            "season_phase",
        ),
        "worst_cases": sorted(
            rows,
            key=lambda row: (-float(row["tv_distance"]), str(row["team_name"])),
        )[:10],
        "accepted": bool(
            np.quantile(tv, 0.95) <= PRIOR_REMOVAL_P95_TV_LIMIT
            and np.max(tv) <= PRIOR_REMOVAL_WORST_TV_LIMIT
        ),
        "recommendation_if_auto": (
            "prior_stripping"
            if np.quantile(tv, 0.95) <= PRIOR_REMOVAL_P95_TV_LIMIT
            and np.max(tv) <= PRIOR_REMOVAL_WORST_TV_LIMIT
            else "explicit_neutralized"
        ),
    }
    return rows, summary


def anchor_sensitivity_rows(runs: list[CutoffRun]) -> list[dict[str, object]]:
    rows = []
    for run in runs:
        context_teams = _team_lookup(run.context)
        network = _network_metrics(run.context.teams, run.prepared.games)
        for team_id, context_pmf in run.context.pmfs.items():
            if team_id not in run.history.pmfs or team_id not in context_teams:
                continue
            team = context_teams[team_id]
            if team.subdivision != "fbs":
                continue
            history_pmf = run.history.pmfs[team_id]
            c_summary = performance_pmf_summaries(context_pmf)
            h_summary = performance_pmf_summaries(history_pmf)
            games = run.context.eligible_game_counts[team_id]
            decomposition = _anchor_decomposition(run, team_id, network)
            rows.append(
                {
                    "season": run.prepared.season,
                    "cutoff": run.prepared.effective_cutoff.isoformat(),
                    "season_phase": _phase(run),
                    "team_id": team_id,
                    "team_name": team.name,
                    "games_played": games,
                    "games_played_bucket": _games_bucket(games),
                    "week_through_cutoff": _week_label(run.prepared.included_rows),
                    "context_expected_rank": c_summary["expected_rank"],
                    "history_expected_rank": h_summary["expected_rank"],
                    "context_minus_history_expected_rank": float(
                        c_summary["expected_rank"] - h_summary["expected_rank"]
                    ),
                    "context_median_rank": c_summary["median_rank"],
                    "history_median_rank": h_summary["median_rank"],
                    "context_minus_history_median_rank": int(c_summary["median_rank"])
                    - int(h_summary["median_rank"]),
                    "pmf_tv": pmf_tv_distance(context_pmf, history_pmf),
                    "context_interval_80_width": int(c_summary["interval_80_high"])
                    - int(c_summary["interval_80_low"]),
                    "history_interval_80_width": int(h_summary["interval_80_high"])
                    - int(h_summary["interval_80_low"]),
                    **decomposition,
                }
            )
    return rows


def performance_predictive_rows(runs: list[CutoffRun]) -> list[dict[str, object]]:
    rows = []
    for run in runs:
        teams = _team_lookup(run.context)
        for team_id, performance_pmf in run.context.pmfs.items():
            team = teams[team_id]
            if team.subdivision != "fbs":
                continue
            predictive_pmf = run.context.anchor_result.pmfs[team_id]
            performance = performance_pmf_summaries(performance_pmf)
            predictive = performance_pmf_summaries(predictive_pmf)
            games = run.context.eligible_game_counts[team_id]
            rows.append(
                {
                    "season": run.prepared.season,
                    "cutoff": run.prepared.effective_cutoff.isoformat(),
                    "season_phase": _phase(run),
                    "team_id": team_id,
                    "team_name": team.name,
                    "games_played": games,
                    "games_played_bucket": _games_bucket(games),
                    "performance_expected_rank": performance["expected_rank"],
                    "predictive_expected_rank": predictive["expected_rank"],
                    "performance_minus_predictive_expected_rank": float(
                        performance["expected_rank"] - predictive["expected_rank"]
                    ),
                    "performance_median_rank": performance["median_rank"],
                    "predictive_median_rank": predictive["median_rank"],
                    "performance_minus_predictive_median_rank": int(
                        performance["median_rank"]
                    )
                    - int(predictive["median_rank"]),
                    "performance_interval_80_width": int(
                        performance["interval_80_high"]
                    )
                    - int(performance["interval_80_low"]),
                    "predictive_interval_80_width": int(predictive["interval_80_high"])
                    - int(predictive["interval_80_low"]),
                    "pmf_tv": pmf_tv_distance(performance_pmf, predictive_pmf),
                    "rated": games > 0,
                }
            )
    return rows


def idle_rows(runs: list[CutoffRun]) -> list[dict[str, object]]:
    by_season: dict[int, list[CutoffRun]] = defaultdict(list)
    for run in runs:
        by_season[run.prepared.season].append(run)
    rows = []
    for season_runs in by_season.values():
        season_runs.sort(key=lambda run: run.prepared.effective_cutoff)
        for before, after in pairwise(season_runs):
            for team_id, before_pmf in before.context.pmfs.items():
                if team_id not in after.context.pmfs:
                    continue
                before_count = before.context.eligible_game_counts[team_id]
                after_count = after.context.eligible_game_counts[team_id]
                if before_count != after_count:
                    continue
                team = _team_lookup(before.context)[team_id]
                if team.subdivision != "fbs":
                    continue
                after_pmf = after.context.pmfs[team_id]
                before_summary = performance_pmf_summaries(before_pmf)
                after_summary = performance_pmf_summaries(after_pmf)
                rows.append(
                    {
                        "season": season_runs[0].prepared.season,
                        "from_cutoff": before.prepared.effective_cutoff.isoformat(),
                        "to_cutoff": after.prepared.effective_cutoff.isoformat(),
                        "team_id": team_id,
                        "team_name": team.name,
                        "games_played": before_count,
                        "expected_rank_before": before_summary["expected_rank"],
                        "expected_rank_after": after_summary["expected_rank"],
                        "expected_rank_change": float(
                            after_summary["expected_rank"]
                            - before_summary["expected_rank"]
                        ),
                        "pmf_tv": pmf_tv_distance(before_pmf, after_pmf),
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            -abs(float(row["expected_rank_change"])),
            str(row["team_name"]),
        ),
    )


def _records_by_team(run: CutoffRun) -> dict[str, tuple[int, int]]:
    records: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for game in run.prepared.games:
        if (
            game.home_subdivision == game.away_subdivision
            or game.home_subdivision == "fbs"
        ):
            home_margin = game.home_points - game.away_points
            away_margin = -home_margin
        else:
            home_margin = game.away_points - game.home_points
            away_margin = -home_margin
        for team_id, margin in (
            (game.home_id, home_margin),
            (game.away_id, away_margin),
        ):
            if team_id in run.context.eligible_game_counts:
                records[team_id][0] += int(margin > 0)
                records[team_id][1] += int(margin < 0)
    return {team_id: (values[0], values[1]) for team_id, values in records.items()}


def illustrative_cases(run: CutoffRun) -> list[dict[str, object]]:
    """Select deterministic, rule-based examples from the real 2025 corpus."""
    by_id = _team_lookup(run.context)
    rows = evidence_rows(run.context, run.context_rows, run)
    records = _records_by_team(run)
    for row in rows:
        wins, losses = records.get(str(row["team_id"]), (0, 0))
        row["wins_through_cutoff"] = wins
        row["losses_through_cutoff"] = losses
        row["win_rate_through_cutoff"] = wins / max(wins + losses, 1)
        row["selection_rule"] = ""

    def choose(
        candidates: list[dict[str, object]], rule: str, key
    ) -> dict[str, object] | None:
        if not candidates:
            return None
        selected = min(candidates, key=key)
        selected = dict(selected)
        selected["case_label"] = rule
        selected["selection_rule"] = rule
        return selected

    close_road_losses = [
        row
        for row in rows
        if row["site"] == "away"
        and int(row["focal_margin"]) < 0
        and abs(int(row["focal_margin"])) <= 7
        and float(row["opponent_anchor_expected_rank"]) <= 25
    ]
    ugly_wins = [
        row
        for row in rows
        if int(row["focal_margin"]) > 0
        and int(row["focal_margin"]) <= 7
        and float(row["opponent_anchor_expected_rank"]) >= 90
    ]
    dominant_wins = [
        row
        for row in rows
        if int(row["focal_margin"]) >= 21
        and int(row["focal_margin"]) > 0
        and float(row["opponent_anchor_expected_rank"]) <= 45
    ]
    selected: list[dict[str, object]] = []
    for value in (
        choose(
            close_road_losses,
            "close_road_loss_to_excellent_opponent",
            lambda row: (
                abs(int(row["focal_margin"])),
                float(row["opponent_anchor_expected_rank"]),
                str(row["game_id"]),
            ),
        ),
        choose(
            ugly_wins,
            "ugly_win_over_weak_opponent",
            lambda row: (
                int(row["focal_margin"]),
                -float(row["opponent_anchor_expected_rank"]),
                str(row["game_id"]),
            ),
        ),
        choose(
            dominant_wins,
            "dominant_win_over_strong_opponent",
            lambda row: (
                -int(row["focal_margin"]),
                float(row["opponent_anchor_expected_rank"]),
                str(row["game_id"]),
            ),
        ),
    ):
        if value is not None:
            selected.append(value)

    team_rows = []
    for team_id, (wins, losses) in records.items():
        team = by_id.get(team_id)
        if team is None or team.subdivision != "fbs" or wins + losses < 8:
            continue
        summary = performance_pmf_summaries(run.context.pmfs[team_id])
        team_rows.append(
            {
                "team_id": team_id,
                "team_name": team.name,
                "wins": wins,
                "losses": losses,
                "win_rate": wins / (wins + losses),
                "performance_expected_rank": summary["expected_rank"],
            }
        )
    good_record_weak_performance = choose(
        [row for row in team_rows if row["win_rate"] >= 0.75],
        "good_record_but_weaker_played_performance",
        lambda row: (
            -float(row["performance_expected_rank"]),
            -float(row["win_rate"]),
            str(row["team_id"]),
        ),
    )
    if good_record_weak_performance is not None:
        selected.append(good_record_weak_performance)
    loss_strong_performance = choose(
        [
            row
            for row in team_rows
            if row["losses"] > 0 and float(row["performance_expected_rank"]) <= 25
        ],
        "loss_but_strong_played_performance",
        lambda row: (
            float(row["performance_expected_rank"]),
            -int(row["losses"]),
            str(row["team_id"]),
        ),
    )
    if loss_strong_performance is not None:
        selected.append(loss_strong_performance)
    largest_difference = choose(
        team_rows,
        "largest_uncomfortable_record_performance_gap",
        lambda row: (
            -abs(float(row["performance_expected_rank"]) - (float(row["losses"]) + 1)),
            str(row["team_id"]),
        ),
    )
    if largest_difference is not None:
        selected.append(largest_difference)
    return selected


def _synthetic_cycle_games(
    topology: str, cycle_length: int = 3, returning_paths: int = 1
) -> tuple[list[Team], list[Game]]:
    if topology not in {"tree", "cycle"}:
        raise ValueError(f"unsupported synthetic topology: {topology}")
    if cycle_length < 3 or returning_paths < 1:
        raise ValueError("synthetic cycles need length >= 3 and at least one path")
    support = 30
    uniform = uniform_pmf(support)
    teams = [Team("focal", "Focal", "fbs", uniform)]
    games: list[Game] = []
    edge_index = 0
    original_cycle_scores = ((31, 24), (21, 17), (28, 27))
    for path_index in range(returning_paths):
        nodes = [f"path{path_index}_node{index}" for index in range(cycle_length - 1)]
        teams.extend(Team(node, node, "fbs", uniform) for node in nodes)
        edges = list(zip(("focal", *nodes), nodes, strict=False))
        if topology == "cycle":
            edges.append((nodes[-1], "focal"))
        for first, second in edges:
            if edge_index < len(original_cycle_scores):
                home_points, away_points = original_cycle_scores[edge_index]
            else:
                home_points, away_points = 32 + (edge_index % 3) * 3, 30
            games.append(
                Game(
                    f"synthetic-{edge_index}",
                    first,
                    second,
                    "fbs",
                    "fbs",
                    home_points,
                    away_points,
                )
            )
            edge_index += 1
    return teams, games


def _synthetic_focal_prior_pair(
    support: int, strength: float, *, linear: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    if linear:
        first = np.linspace(1, support, support) ** -1
        second = np.linspace(1, support, support)
    else:
        coordinate = np.linspace(-1.0, 1.0, support)
        first = np.exp(-strength * coordinate)
        second = first[::-1]
    first /= first.sum()
    second /= second.sum()
    return first, second


def _synthetic_prior_comparison(
    likelihood: LikelihoodV1,
    topology: str,
    cycle_length: int,
    returning_paths: int,
    focal_prior_strength: float,
    *,
    linear_priors: bool = False,
    max_iterations: int,
    tolerance: float,
    damping: float,
    label: str,
) -> dict[str, object]:
    teams, games = _synthetic_cycle_games(
        topology, cycle_length, returning_paths
    )
    first_prior, second_prior = _synthetic_focal_prior_pair(
        len(teams[0].prior), focal_prior_strength, linear=linear_priors
    )
    common = teams[1:]
    first_teams = [Team("focal", "Focal", "fbs", first_prior), *common]
    second_teams = [Team("focal", "Focal", "fbs", second_prior), *common]
    stripped_first = infer_performance(
        first_teams,
        games,
        likelihood,
        anchor_family="context",
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
    )
    stripped_second = infer_performance(
        second_teams,
        games,
        likelihood,
        anchor_family="context",
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
    )
    explicit_first, explicit_first_result = explicit_neutralized_target(
        first_teams,
        games,
        likelihood,
        "focal",
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
    )
    explicit_second, explicit_second_result = explicit_neutralized_target(
        second_teams,
        games,
        likelihood,
        "focal",
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
    )
    ranks = np.arange(1, len(first_prior) + 1)
    return {
        "label": label,
        "topology": topology,
        "cycle_length": cycle_length if topology == "cycle" else None,
        "returning_path_count": returning_paths if topology == "cycle" else 0,
        "focal_prior_strength": focal_prior_strength,
        "linear_reversal_priors": linear_priors,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "damping": damping,
        "stripping_tv_distance": pmf_tv_distance(
            stripped_first.pmfs["focal"], stripped_second.pmfs["focal"]
        ),
        "stripping_expected_rank_difference": float(
            np.dot(ranks, stripped_first.pmfs["focal"])
            - np.dot(ranks, stripped_second.pmfs["focal"])
        ),
        "explicit_neutralized_tv_distance": pmf_tv_distance(
            explicit_first, explicit_second
        ),
        "stripping_converged": bool(
            stripped_first.anchor_result.converged
            and stripped_second.anchor_result.converged
        ),
        "explicit_converged": bool(
            explicit_first_result.converged and explicit_second_result.converged
        ),
        "stripping_iterations": max(
            stripped_first.anchor_result.iterations,
            stripped_second.anchor_result.iterations,
        ),
        "explicit_iterations": max(
            explicit_first_result.iterations, explicit_second_result.iterations
        ),
    }


def loopy_cycle_diagnostics(likelihood: LikelihoodV1) -> dict[str, object]:
    """Separate returning-path feedback from initialization and tolerance."""
    support = 30
    baseline = _synthetic_prior_comparison(
        likelihood,
        "cycle",
        3,
        1,
        0.0,
        linear_priors=True,
        max_iterations=500,
        tolerance=1e-9,
        damping=0.35,
        label="baseline_original_cycle",
    )
    topology_rows = [
        _synthetic_prior_comparison(
            likelihood,
            "tree",
            cycle_length,
            1,
            strength,
            max_iterations=500,
            tolerance=1e-9,
            damping=0.35,
            label=f"tree_length_{cycle_length}_strength_{strength}",
        )
        for cycle_length in (3, 4, 5)
        for strength in (0.5, 1.5, 3.0)
    ]
    topology_rows.extend(
        _synthetic_prior_comparison(
            likelihood,
            "cycle",
            cycle_length,
            returning_paths,
            strength,
            max_iterations=500,
            tolerance=1e-9,
            damping=0.35,
            label=(
                f"cycle_length_{cycle_length}_paths_{returning_paths}_"
                f"strength_{strength}"
            ),
        )
        for cycle_length in (3, 4, 5)
        for returning_paths in (1, 2)
        for strength in (0.5, 1.5, 3.0)
    )
    settings_rows = [
        _synthetic_prior_comparison(
            likelihood,
            "cycle",
            3,
            1,
            0.0,
            linear_priors=True,
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=0.35,
            label=label,
        )
        for label, max_iterations, tolerance in (
            ("one_message_sweep", 1, 1e-12),
            ("production_like", 100, 1e-6),
            ("tight_converged", 500, 1e-10),
        )
    ]
    return {
        "support": support,
        "baseline": baseline,
        "topology_rows": topology_rows,
        "settings_rows": settings_rows,
        "conclusion": (
            "A tree has no path that returns focal information, so stripping is "
            "invariant. A cycle lets the focal prior affect an outgoing message, "
            "which can return through an opponent and survive algebraic stripping. "
            "The residual remains after convergence and explicit target neutralization "
            "removes it; one message sweep suppresses the feedback because the first "
            "outgoing focal message is initialized uniformly."
        ),
    }


def structural_diagnostics(likelihood: LikelihoodV1) -> dict[str, object]:
    """Run small semantic checks against the frozen likelihood surface."""
    support = 30
    uniform = uniform_pmf(support)
    opponent_strong = np.exp(-np.arange(support) / 7.0)
    opponent_strong /= opponent_strong.sum()
    opponent_weak = opponent_strong[::-1]

    def one_game(
        margin: int, focal_home: bool, opponent_prior: np.ndarray
    ) -> np.ndarray:
        focal = Team("focal", "Focal", "fbs", uniform)
        opponent = Team("opponent", "Opponent", "fbs", opponent_prior)
        game = (
            Game("g", "focal", "opponent", "fbs", "fbs", 30 + margin, 30)
            if focal_home
            else Game("g", "opponent", "focal", "fbs", "fbs", 30, 30 + margin)
        )
        inference = infer_performance(
            [focal, opponent], [game], likelihood, anchor_family="context"
        )
        return inference.pmfs["focal"]

    margin_pmfs = [one_game(margin, True, uniform) for margin in (-1, 0, 1)]
    margin_expected = [
        float(np.dot(np.arange(1, support + 1), pmf)) for pmf in margin_pmfs
    ]
    opponent_pmfs = [
        one_game(7, True, prior) for prior in (opponent_weak, opponent_strong)
    ]
    opponent_expected = [
        float(np.dot(np.arange(1, support + 1), pmf)) for pmf in opponent_pmfs
    ]
    venue_pmfs = [one_game(7, focal_home, uniform) for focal_home in (True, False)]
    venue_expected = [
        float(np.dot(np.arange(1, support + 1), pmf)) for pmf in venue_pmfs
    ]
    focal_a_prior = np.linspace(1, support, support) ** -1
    focal_a_prior /= focal_a_prior.sum()
    focal_b_prior = np.linspace(1, support, support)
    focal_b_prior /= focal_b_prior.sum()
    focal_a = Team("focal", "Focal", "fbs", focal_a_prior)
    focal_b = Team("focal", "Focal", "fbs", focal_b_prior)
    opponent = Team("opponent", "Opponent", "fbs", uniform)
    game = Game("g", "focal", "opponent", "fbs", "fbs", 37, 30)
    first = infer_performance(
        [focal_a, opponent], [game], likelihood, anchor_family="context"
    ).pmfs["focal"]
    second = infer_performance(
        [focal_b, opponent], [game], likelihood, anchor_family="context"
    ).pmfs["focal"]
    cycle_diagnostics = loopy_cycle_diagnostics(likelihood)
    cycle_baseline = cycle_diagnostics["baseline"]
    no_game = infer_performance(
        [focal_a], [], likelihood, anchor_family="context"
    ).pmfs["focal"]
    continuity_jumps = [
        abs(margin_expected[index + 1] - margin_expected[index]) for index in range(2)
    ]
    return {
        "margin_monotonicity": {
            "expected_ranks_for_minus1_zero_plus1": margin_expected,
            "passes": bool(
                margin_expected[0] >= margin_expected[1] >= margin_expected[2]
            ),
        },
        "opponent_strength_monotonicity": {
            "expected_ranks_weak_opponent_then_strong_opponent": opponent_expected,
            "passes": bool(opponent_expected[1] <= opponent_expected[0]),
        },
        "venue_semantics": {
            "expected_ranks_home_then_road": venue_expected,
            "passes": bool(venue_expected[1] <= venue_expected[0]),
        },
        "win_loss_continuity": {
            "adjacent_expected_rank_jumps": continuity_jumps,
            "max_adjacent_jump": max(continuity_jumps),
            "passes": bool(max(continuity_jumps) < support / 2),
        },
        "focal_prior_independence": {
            "tv_distance": pmf_tv_distance(first, second),
            "passes": bool(np.allclose(first, second, atol=1e-10)),
            "loopy_cycle_residual_tv_distance": cycle_baseline[
                "stripping_tv_distance"
            ],
            "loopy_cycle_residual_expected_rank_difference": cycle_baseline[
                "stripping_expected_rank_difference"
            ],
            "loopy_cycle_residual_within_predeclared_tolerance": bool(
                float(cycle_baseline["stripping_tv_distance"])
                <= PRIOR_REMOVAL_WORST_TV_LIMIT
            ),
            "explicit_neutralized_cycle_tv_distance": cycle_baseline[
                "explicit_neutralized_tv_distance"
            ],
            "explicit_neutralized_cycle_passes": bool(
                float(cycle_baseline["explicit_neutralized_tv_distance"]) <= 1e-10
            ),
        },
        "loopy_cycle_diagnostics": cycle_diagnostics,
        "zero_game_neutrality": {
            "expected_rank": float(np.dot(np.arange(1, support + 1), no_game)),
            "expected_uniform_rank": (support + 1) / 2,
            "passes": bool(np.allclose(no_game, uniform, atol=1e-10)),
        },
    }


def _plot_anchor(rows: list[dict[str, object]], path: Path) -> None:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["games_played"])].append(row)
    x = sorted(grouped)
    mean_abs = [
        float(
            np.mean(
                np.abs(
                    [
                        float(row["context_minus_history_expected_rank"])
                        for row in grouped[value]
                    ]
                )
            )
        )
        for value in x
    ]
    mean_tv = [
        float(np.mean([float(row["pmf_tv"]) for row in grouped[value]])) for value in x
    ]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(x, mean_abs, marker="o", label="mean |C − H| expected rank")
    axis.set_xlabel("Eligible games played")
    axis.set_ylabel("Expected-rank disagreement")
    second = axis.twinx()
    second.plot(x, mean_tv, color="tab:orange", marker="s", label="mean PMF TV")
    second.set_ylabel("PMF total variation")
    axis.set_title("Performance anchor sensitivity by evidence depth")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_performance_predictive(rows: list[dict[str, object]], path: Path) -> None:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["games_played"])].append(row)
    x = sorted(grouped)
    values = [
        float(
            np.mean(
                np.abs(
                    [
                        float(row["performance_minus_predictive_expected_rank"])
                        for row in grouped[value]
                    ]
                )
            )
        )
        for value in x
    ]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(x, values, marker="o")
    axis.set_xlabel("Eligible games played")
    axis.set_ylabel("Mean absolute expected-rank difference")
    axis.set_title("Performance C versus Predictive C")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_uncertainty(rows: list[dict[str, object]], path: Path) -> None:
    rated = [row for row in rows if row["rated"]]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(
        [int(row["games_played"]) for row in rated],
        [int(row["interval_80_width"]) for row in rated],
        alpha=0.75,
    )
    axis.set_xlabel("Eligible games played")
    axis.set_ylabel("80% interval width")
    axis.set_title("Current Performance C uncertainty")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_future(rows: list[dict[str, object]], path: Path) -> None:
    selected = [row for row in rows if row["horizon"] == "next_game"]
    models = sorted({str(row["model"]) for row in selected})
    buckets = ["0", "1-3", "4-6", "7+"]
    figure, axis = plt.subplots(figsize=(8, 4))
    for model in models:
        values = []
        for bucket in buckets:
            matches = [
                row
                for row in selected
                if row["model"] == model and row["games_played_bucket"] == bucket
            ]
            values.append(
                float(
                    np.average(
                        [float(row["marginalized_nll"]) for row in matches],
                        weights=[int(row["prediction_count"]) for row in matches],
                    )
                )
                if matches
                else np.nan
            )
        axis.plot(buckets, values, marker="o", label=model)
    axis.set_xlabel("Games played at cutoff")
    axis.set_ylabel("Next-game marginalized NLL")
    axis.set_title("Future-game validation by evidence depth")
    axis.legend(fontsize="small")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _latest_current_cutoff(root: Path, season: int) -> datetime:
    candidates = []
    for path in (root / f"data/processed/snapshots/{season}").glob(
        "*/predictive/context/metadata.json"
    ):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("snapshot_type") == "preseason" or not value.get(
            "effective_cutoff"
        ):
            continue
        candidates.append(_parse_datetime(str(value["effective_cutoff"])))
    if not candidates:
        raise FileNotFoundError(f"no checked-in current-season cutoff for {season}")
    return max(candidates)


def _current_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    rated = [row for row in rows if row["rated"]]
    ranked = sorted(
        rows,
        key=lambda row: (
            not bool(row["rated"]),
            float(row["expected_rank"]),
            str(row["team_name"]),
        ),
    )
    broad = sorted(
        rated, key=lambda row: (-int(row["interval_95_width"]), str(row["team_name"]))
    )[:10]
    concentrated = sorted(
        rated, key=lambda row: (int(row["interval_95_width"]), str(row["team_name"]))
    )[:10]
    return {
        "team_count": len(rows),
        "rated_team_count": len(rated),
        "unrated_team_count": len(rows) - len(rated),
        "top25": ranked[:25],
        "broadest_distributions": broad,
        "most_concentrated_distributions": concentrated,
    }


def _group_metric_summary(
    rows: list[dict[str, object]], value: str
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["season_phase"]), str(row["games_played_bucket"]))].append(row)
    result = []
    for (phase, bucket), group in sorted(grouped.items()):
        result.append(
            {
                "season_phase": phase,
                "games_played_bucket": bucket,
                **_summary_for_rows(group, value),
            }
        )
    return result


def _group_metric_summary_by_week(
    rows: list[dict[str, object]], value: str
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        week = row.get("week_through_cutoff")
        if week is None:
            continue
        grouped[(int(row["season"]), int(week))].append(row)
    return [
        {
            "season": season,
            "week_through_cutoff": week,
            **_summary_for_rows(group, value),
        }
        for (season, week), group in sorted(grouped.items())
    ]


def _correlation_with_pmf_tv(
    rows: list[dict[str, object]], value_key: str
) -> float | None:
    left = np.asarray([float(row["pmf_tv"]) for row in rows], dtype=float)
    right = np.asarray([float(row[value_key]) for row in rows], dtype=float)
    if len(left) < 2 or np.isclose(left.std(), 0.0) or np.isclose(right.std(), 0.0):
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _rank_quartile_labels(
    rows: list[dict[str, object]], value_key: str
) -> dict[int, str]:
    ordered = sorted(
        range(len(rows)),
        key=lambda index: (
            float(rows[index][value_key]),
            int(rows[index]["season"]),
            str(rows[index]["cutoff"]),
            str(rows[index]["team_id"]),
        ),
    )
    return {
        index: f"Q{min(4, position * 4 // max(len(ordered), 1) + 1)}"
        for position, index in enumerate(ordered)
    }


def _anchor_decomposition_group_rows(
    rows: list[dict[str, object]], dimension: str, labels: list[str]
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row, label in zip(rows, labels, strict=True):
        grouped[label].append(row)
    result = []
    for bucket, group in sorted(grouped.items()):
        result.append(
            {
                "dimension": dimension,
                "bucket": bucket,
                "count": len(group),
                "median_pmf_tv": float(
                    np.median([float(row["pmf_tv"]) for row in group])
                ),
                "p95_pmf_tv": float(
                    np.quantile([float(row["pmf_tv"]) for row in group], 0.95)
                ),
                "median_abs_expected_rank_difference": float(
                    np.median(
                        [
                            abs(float(row["context_minus_history_expected_rank"]))
                            for row in group
                        ]
                    )
                ),
                "median_games_played": float(
                    np.median([int(row["games_played"]) for row in group])
                ),
                "median_unique_opponents": float(
                    np.median([int(row["unique_opponent_count"]) for row in group])
                ),
                "median_opponent_anchor_posterior_tv_sum": float(
                    np.median(
                        [
                            float(row["opponent_anchor_posterior_tv_sum"])
                            for row in group
                        ]
                    )
                ),
                "median_opponent_anchor_posterior_tv_max": float(
                    np.median(
                        [
                            float(row["opponent_anchor_posterior_tv_max"])
                            for row in group
                        ]
                    )
                ),
                "median_context_interval_80_width": float(
                    np.median(
                        [int(row["context_interval_80_width"]) for row in group]
                    )
                ),
            }
        )
    return result


def anchor_decomposition_summary(
    rows: list[dict[str, object]]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Summarize whether C/H disagreement tracks breadth or divergent anchors."""
    dimensions: list[tuple[str, list[str]]] = [
        (
            "season_phase",
            [str(row["season_phase"]) for row in rows],
        ),
        (
            "games_played_bucket",
            [str(row["games_played_bucket"]) for row in rows],
        ),
        (
            "unique_opponents_bucket",
            [_games_bucket(int(row["unique_opponent_count"])) for row in rows],
        ),
    ]
    for value_key, dimension in (
        ("opponent_anchor_posterior_tv_sum", "opponent_anchor_tv_sum_quartile"),
        ("opponent_anchor_posterior_tv_max", "opponent_anchor_tv_max_quartile"),
        ("context_interval_80_width", "performance_width_quartile"),
        ("graph_degree", "graph_degree_quartile"),
        ("cycle_edge_count", "cycle_exposure_quartile"),
    ):
        quartiles = _rank_quartile_labels(rows, value_key)
        dimensions.append(
            (dimension, [quartiles[index] for index in range(len(rows))])
        )
    group_rows = [
        group
        for dimension, labels in dimensions
        for group in _anchor_decomposition_group_rows(rows, dimension, labels)
    ]
    metric_keys = (
        "games_played",
        "unique_opponent_count",
        "graph_degree",
        "component_cycle_rank",
        "cycle_edge_count",
        "opponent_anchor_posterior_tv_mean",
        "opponent_anchor_posterior_tv_sum",
        "opponent_anchor_posterior_tv_max",
        "opponent_anchor_expected_rank_difference_sum",
        "context_interval_80_width",
    )
    correlations = {
        key: _correlation_with_pmf_tv(rows, key) for key in metric_keys
    }
    nonzero_sums = [
        float(row["opponent_anchor_posterior_tv_sum"])
        for row in rows
        if float(row["opponent_anchor_posterior_tv_sum"]) > 0
    ]
    max_shares = [
        float(row["opponent_anchor_posterior_tv_max_share"])
        for row in rows
        if float(row["opponent_anchor_posterior_tv_sum"]) > 0
    ]
    dominance = {
        "nonzero_opponent_anchor_sum_count": len(nonzero_sums),
        "median_opponent_anchor_tv_sum": float(np.median(nonzero_sums))
        if nonzero_sums
        else 0.0,
        "median_largest_opponent_share": float(np.median(max_shares))
        if max_shares
        else 0.0,
        "fraction_largest_opponent_share_at_least_half": (
            float(np.mean(np.asarray(max_shares) >= 0.5)) if max_shares else 0.0
        ),
        "correlation_pmf_tv_with_opponent_sum": correlations[
            "opponent_anchor_posterior_tv_sum"
        ],
        "correlation_pmf_tv_with_largest_opponent": correlations[
            "opponent_anchor_posterior_tv_max"
        ],
    }
    network = {
        "unique_opponent_count": _summary_for_rows(
            rows, "unique_opponent_count"
        ),
        "graph_degree": _summary_for_rows(rows, "graph_degree"),
        "component_cycle_rank": _summary_for_rows(rows, "component_cycle_rank"),
        "cycle_edge_count": _summary_for_rows(rows, "cycle_edge_count"),
        "cycle_participation_count": sum(
            bool(row["cycle_participation"]) for row in rows
        ),
    }
    return group_rows, {
        "metric_correlations_with_pmf_tv": correlations,
        "dominance": dominance,
        "network": network,
        "group_rows": group_rows,
    }


def _weighted_future_metric(
    group: list[dict[str, object]], value_key: str, weight_key: str, denominator: int
) -> float:
    return float(
        sum(
            float(item[value_key]) * int(item[weight_key])
            for item in group
            if item[value_key] is not None
        )
        / max(denominator, 1)
    )


def _report(
    path: Path,
    summary: dict[str, object],
    current_rows: list[dict[str, object]],
    cases: list[dict[str, object]],
    artifact_inventory: list[dict[str, object]],
) -> None:
    top25 = sorted(
        [row for row in current_rows if row["rated"]],
        key=lambda row: (float(row["expected_rank"]), str(row["team_name"])),
    )[:25]
    bp_validation = summary["bp_prior_removal_validation"]
    bp_coverage = bp_validation["coverage"]
    anchor_summary = summary["anchor_sensitivity"]
    decomposition = anchor_summary["decomposition"]
    future_validation = summary["future_validation"]
    future_population = future_validation["population"]
    loopy_diagnostics = summary["structural_diagnostics"]["loopy_cycle_diagnostics"]
    loopy_baseline = loopy_diagnostics["baseline"]
    tree_diagnostics = [
        row
        for row in loopy_diagnostics["topology_rows"]
        if row["topology"] == "tree"
    ]
    cycle_diagnostics = [
        row
        for row in loopy_diagnostics["topology_rows"]
        if row["topology"] == "cycle"
    ]
    lines = [
        "# Performance V1 research report",
        "",
        "This is a research-only descriptive quality analysis. It is not standings, strength of record, postseason selection, a poll emulator, or a claim about deservingness. No Performance family is registered with or published by the website.",
        "",
        "## Definition",
        "",
        "For target team `i`, let `C_i(r)` be its Context preseason PMF and let `Post_C_i(r | G)` be the Context-started BP marginal using every eligible game factor in the cutoff network. The primary Performance C PMF is:",
        "",
        "`Performance_C_i(r | G) = normalize(Post_C_i(r | G) / C_i(r))`.",
        "",
        "Performance H uses the identical construction with History priors for the network and the focal denominator. Equivalently, under exact factorized inference this is the game-derived likelihood profile under a uniform focal prior. The focal preseason prior is removed as a direct factor. Under loopy BP, a small indirect feedback residue can remain because prior information may propagate through opponents and return through schedule cycles. Explicit focal-prior neutralization is the correctness baseline; empirical stripping error on historical cases is quantified separately. Context remains the primary opponent anchor. Opponent quality is allowed to update after the focal game and before the cutoff, so an idle team's Performance can move when its opponent plays.",
        "",
        "## Prior-removal validation",
        "",
        f"The predeclared acceptance rule was p95 PMF TV ≤ {bp_validation['predeclared_p95_tv_limit']:.2f} and worst PMF TV ≤ {bp_validation['predeclared_worst_tv_limit']:.2f}, measured against ordinary BP with only the target prior replaced by a uniform PMF. The selected method was **{summary['performance_method']}**.",
        "",
        f"The prospective panel contains {bp_coverage['case_count']} unique comparison cases across seasons {', '.join(str(value) for value in bp_coverage['seasons'])}, phases {', '.join(bp_coverage['season_phases'])}, games-played buckets {', '.join(bp_coverage['games_played_buckets'])}, and {len(bp_coverage['cutoffs'])} cutoff instances. Target selection uses fixed cutoff indices {bp_coverage['cutoff_indices']} and ordinary posterior shape/location, Context/History disagreement, and graph descriptors; it never uses stripping error, explicit-neutralization output, or future outcomes.",
        "",
        f"Represented roles: {', '.join(bp_coverage['represented_case_roles'])}. Observed median/p95/worst TV: {bp_validation['median_tv']:.6f} / {bp_validation['p95_tv']:.6f} / {bp_validation['worst_tv']:.6f}. The validation was {'accepted' if bp_validation['accepted'] else 'not accepted'} under that predeclared rule.",
        "",
        "Validation error by selected phase:",
    ]
    for group in bp_validation["by_season_phase"]:
        lines.append(
            f"- {group['group']}: n={group['count']}, median TV {float(group['median_tv']):.6f}, p95 {float(group['p95_tv']):.6f}, max {float(group['max_tv']):.6f}."
        )
    lines.append("Worst selected cases:")
    for row in bp_validation["worst_cases"][:5]:
        lines.append(
            f"- {row['case_label']}: {row['season']} {row['team_name']} ({row['games_played']} games, {row['unique_opponent_count']} unique opponents, cycle-edge exposure {row['cycle_edge_count']}), TV {float(row['tv_distance']):.6f}, expected-rank difference {float(row['expected_rank_difference']):+.6f}."
        )
    lines.extend(
        [
            "",
            "The original seven-case conclusion is therefore reassessed on a deliberately broader panel; the acceptance rule itself is unchanged.",
        ]
    )
    lines.extend(
        [
            "",
            "## Current 2026 Performance C",
            "",
            f"Latest checked-in cutoff: `{summary['current_cutoff']}`. Rated teams: {summary['current']['rated_team_count']} of {summary['current']['team_count']}; zero-game teams are marked unrated and kept out of meaningful display ordering.",
            f"Observed local build runtime: {float(summary['runtime_seconds_observed']):.3f} seconds (wall-clock and hardware dependent).",
            "",
            "| Display | Team | Expected quality-equivalent rank | Median | 50% | 80% | 95% | Top 5 | Top 10 | Top 25 | Games |",
            "|---:|---|---:|---:|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in top25:
        lines.append(
            f"| {row['display_rank']} | {row['team_name']} | {float(row['expected_rank']):.2f} | {row['median_rank']} | {row['interval_50_low']}–{row['interval_50_high']} | {row['interval_80_low']}–{row['interval_80_high']} | {row['interval_95_low']}–{row['interval_95_high']} | {float(row['top5_probability']):.3f} | {float(row['top10_probability']):.3f} | {float(row['top25_probability']):.3f} | {row['games_played']} |"
        )
    lines.extend(
        [
            "",
            "Display order is an ordering of expected latent quality; it is not a joint probability permutation over ranks 1…N. The PMF is over quality-equivalent rank, not poll rank, standings position, or postseason probability.",
            "",
            "## Anchor sensitivity and predictive comparison",
            "",
            f"Across the 2022–2025 cutoff panel, Context-minus-History expected-rank disagreement had median/p95/max absolute values {summary['anchor_sensitivity']['absolute_expected_rank_difference']['median']:.3f} / {summary['anchor_sensitivity']['absolute_expected_rank_difference']['p95']:.3f} / {summary['anchor_sensitivity']['absolute_expected_rank_difference']['max']:.3f}; median/p95/max PMF TV was {summary['anchor_sensitivity']['pmf_tv']['median']:.4f} / {summary['anchor_sensitivity']['pmf_tv']['p95']:.4f} / {summary['anchor_sensitivity']['pmf_tv']['max']:.4f}. This difference is opponent-quality uncertainty, not focal-team preseason contamination.",
            "",
            "Anchor sensitivity is also emitted by games-played bucket and season/week so the early-to-late hypothesis can be checked rather than assumed.",
            "",
            f"Performance C versus Predictive C expected-rank difference (Performance minus Predictive) had median {summary['performance_vs_predictive']['expected_rank_difference']['median']:.3f} and absolute p95/max {summary['performance_vs_predictive']['absolute_expected_rank_difference']['p95']:.3f} / {summary['performance_vs_predictive']['absolute_expected_rank_difference']['max']:.3f}. Performance uncertainty remains broader when evidence is sparse; it is not artificially narrowed.",
            "",
        ]
    )
    lines.append("Historical anchor sensitivity by phase and games-played bucket:")
    expected_rank_by_group = {
        (str(row["season_phase"]), str(row["games_played_bucket"])): row
        for row in anchor_summary["by_games_bucket"]
    }
    for row in anchor_summary["by_games_bucket_pmf_tv"]:
        expected_rank = expected_rank_by_group[
            (str(row["season_phase"]), str(row["games_played_bucket"]))
        ]
        lines.append(
            f"- {row['season_phase']} / {row['games_played_bucket']}: median PMF TV {float(row['median']):.4f}, p95 {float(row['p95']):.4f}; median absolute expected-rank difference {float(expected_rank['median']):.3f}, n={row['count']}."
        )
    lines.extend(
        [
            "",
            "Targeted decomposition of the increasing C/H sensitivity:",
            f"- Pearson correlation of focal PMF TV with summed opponent posterior C/H TV: {decomposition['metric_correlations_with_pmf_tv']['opponent_anchor_posterior_tv_sum']!s}; with the largest opponent TV: {decomposition['metric_correlations_with_pmf_tv']['opponent_anchor_posterior_tv_max']!s}.",
            f"- Among rows with nonzero opponent disagreement, the median largest-opponent share of summed TV was {float(decomposition['dominance']['median_largest_opponent_share']):.3f}; the largest opponent supplied at least half the sum in {float(decomposition['dominance']['fraction_largest_opponent_share_at_least_half']):.1%} of such rows.",
            "- The row-level CSV records games played, unique opponents, graph degree, cycle-edge exposure, opponent posterior disagreement sum/max, and focal posterior width; the grouped companion artifact reports each fixed quartile. These are descriptive associations, not a causal decomposition.",
            "",
            "Largest current Context-versus-History anchor disagreements (C − H expected rank):",
        ]
    )
    for row in summary["current_anchor_disagreements"][:10]:
        lines.append(
            f"- {row['team_name']}: {float(row['context_minus_history_expected_rank']):+.2f} ranks, PMF TV {float(row['pmf_tv']):.4f}, {row['games_played']} games."
        )
    lines.extend(
        [
            "",
            "Largest current Performance C versus Predictive C differences:",
        ]
    )
    for row in summary["current_performance_vs_predictive"]["largest_differences"][:10]:
        lines.append(
            f"- {row['team_name']}: Performance {float(row['performance_expected_rank']):.2f}, Predictive {float(row['predictive_expected_rank']):.2f}, difference {float(row['performance_minus_predictive_expected_rank']):+.2f}, {row['games_played']} games."
        )
    lines.extend(
        [
            "",
            "Performance C versus Predictive C by games-played bucket (absolute expected-rank difference):",
        ]
    )
    for row in summary["performance_vs_predictive"]["by_games_bucket"]:
        lines.append(
            f"- {row['season_phase']} / {row['games_played_bucket']}: median {float(row['median']):.3f}, p95 {float(row['p95']):.3f}, n={row['count']}."
        )
    lines.extend(
        [
            "",
            "Current uncertainty extremes (Performance C):",
        ]
    )
    for label, values in (
        ("broadest", summary["current"]["broadest_distributions"][:3]),
        (
            "most concentrated",
            summary["current"]["most_concentrated_distributions"][:3],
        ),
    ):
        lines.append(
            f"- {label}: "
            + "; ".join(
                f"{row['team_name']} ({row['interval_95_width']}-rank 95% width, {row['games_played']} games)"
                for row in values
            )
        )
    lines.extend(
        [
            "",
            "## Future-game validation",
            "",
            "The leakage-safe 2022–2025 panel rates each team only with games at or before its cutoff, then scores later completed FBS/FCS games. The table compares Predictive C/H and Performance C/H descriptively; it does not promote Performance over Predictive based on NLL.",
            "",
        ]
    )
    lines.extend(
        [
            f"Population audit: {future_population['cutoff_count']} cutoff instances across seasons {', '.join(str(value) for value in future_population['seasons'])}; {future_population['candidate_future_game_instance_count']} candidate future game instances produced {future_population['candidate_fbs_team_game_key_count']} FBS team-game keys. All four models share {future_population['scored_fbs_team_game_key_count']} strict common scoring keys; {future_population['excluded_fbs_team_game_key_count']} candidate keys were not scored. `all_future` is the inclusive population ({future_population['scored_fbs_team_game_key_count']} keys), while `next_game` is its {future_population['next_game_key_count']}-key subset and the remaining {future_population['later_future_key_count']} keys are later future games.",
            "The cutoff invariant is enforced from source start dates: completed games at or before the effective cutoff form inference, completed games after it form the future pool, and incomplete/invalid/lower-division rows are excluded explicitly. A model-panel mismatch or duplicate scoring key fails the build rather than changing the comparison population.",
            "",
        ]
    )
    for row in summary["future_validation"]["overall"]:
        lines.append(
            f"- {row['model']} / {row['horizon']}: n={row['prediction_count']}, marginalized NLL={row['marginalized_nll']:.4f}, margin MAE={row['margin_absolute_error']:.4f}, win Brier={row['win_brier']:.4f}, calibration absolute error={row['calibration_absolute_error']:.4f}."
        )
    lines.extend(["", "## Illustrative corpus cases", ""])
    for case in cases:
        if "game_id" in case:
            lines.append(
                f"- **{case['case_label']}**: {case['team_name']} {case['focal_points']}–{case['opponent_points']} vs {case['opponent_name']} ({case['site']}); opponent Context anchor expected rank {float(case['opponent_anchor_expected_rank']):.2f}; focal Performance expected rank at cutoff {float(case['focal_performance_expected_rank']):.2f}; record through cutoff {case['wins_through_cutoff']}–{case['losses_through_cutoff']}."
            )
        else:
            lines.append(
                f"- **{case['case_label']}**: {case['team_name']}, record {case['wins']}–{case['losses']}, Performance expected rank {float(case['performance_expected_rank']):.2f}."
            )
    lines.extend(
        [
            "",
            "## Loopy-BP residual diagnostic",
            "",
            f"The synthetic baseline (one 3-team cycle, converged damped BP, reversed linear focal priors) produced stripping TV {float(loopy_baseline['stripping_tv_distance']):.6f} and explicit-neutralized TV {float(loopy_baseline['explicit_neutralized_tv_distance']):.6g}. Tree controls had maximum stripping TV {max(float(row['stripping_tv_distance']) for row in tree_diagnostics):.6g}; cycle variants had maximum {max(float(row['stripping_tv_distance']) for row in cycle_diagnostics):.6f} under the same converged settings.",
            f"The historical panel worst TV was {float(bp_validation['worst_tv']):.6f}; the synthetic baseline is {float(loopy_baseline['stripping_tv_distance']) / max(float(bp_validation['worst_tv']), 1e-300):.1f} times larger (and its median historical comparison is {float(loopy_baseline['stripping_tv_distance']) / max(float(bp_validation['median_tv']), 1e-300):.1f} times larger).",
            f"{loopy_diagnostics['conclusion']}",
            "",
            "Tolerance/iteration controls:",
        ]
    )
    for row in loopy_diagnostics["settings_rows"]:
        lines.append(
            f"- {row['label']}: iterations={row['max_iterations']}, tolerance={row['tolerance']}, converged={row['stripping_converged']}, stripping TV={float(row['stripping_tv_distance']):.6f}, explicit-neutralized TV={float(row['explicit_neutralized_tv_distance']):.6g}."
        )
    lines.extend(
        [
            "",
            "This is a limitation of approximate loopy inference, not a production retuning target. Historical validation is the realistic check: its worst stripping error is compared with this synthetic stress panel separately, and explicit neutralization remains the correctness baseline.",
        ]
    )
    lines.extend(
        [
            "",
            "The selector is deterministic and rule-based; it includes an uncomfortable record/performance gap rather than only favorable examples. These comparisons describe played football and opponent interpretation, not reward or punishment for winning.",
            "",
            "## Structural semantics and limitations",
            "",
            f"The frozen Historical Likelihood V1 surface is used unchanged: Student-t df 15, rank-percentile surface, site semantics, FBS/FCS orientation, and margin. There is no win indicator, record feature, capped margin, YPP, recency weight, poll input, or selection logic. Structural checks passed: direct focal-prior TV {float(summary['structural_diagnostics']['focal_prior_independence']['tv_distance']):.3g}; the loopy-cycle stripping residual is {float(summary['structural_diagnostics']['focal_prior_independence']['loopy_cycle_residual_tv_distance']):.3g} and explicit-neutralized cycle TV is {float(summary['structural_diagnostics']['focal_prior_independence']['explicit_neutralized_cycle_tv_distance']):.3g}; maximum adjacent −1/0/+1 expected-rank jump {float(summary['structural_diagnostics']['win_loss_continuity']['max_adjacent_jump']):.3f}; zero-game output is uniform. The focal preseason prior is removed as a direct factor, not claimed to be perfectly independent under approximate loopy BP.",
            "",
            f"Idle-team updates were observed in {summary['idle_update']['row_count']} no-new-game cutoff transitions; the largest expected-rank movement was {summary['idle_update']['largest_abs_expected_rank_change']:.3f}. This is expected network updating, not a bug.",
            "",
            "FCS opponents use the established full-season support/fallback policy. Historical 2018–2021 priors are not available in the frozen H/C prediction artifacts, so they were not fabricated; the temporal evaluation uses 2022–2025. Current 2026 uses the latest checked-in cached cutoff and does not fetch new data.",
            "",
            "## Artifact footprint and reproducibility",
            "",
            f"The generated bundle retains evidence needed for the audit: {len(artifact_inventory)} measured pre-report members totaling {sum(int(item['bytes']) for item in artifact_inventory):,} bytes before this report and its manifest are written. No artifact was pruned or compacted in this audit, and no runtime cache is committed. The final detailed byte/row/purpose inventory, including this report, is written to `artifact_inventory.csv`.",
            "",
            "Row-level PMFs, future predictions, game evidence, validation cases, aggregates, and plots are intentionally retained because each supports reproducibility, leakage review, or interpretation; repeated representations are not treated as interchangeable evidence.",
            "",
            "Artifacts are research-only and reproducibly generated by `uv run python scripts/build_performance_v1.py`. The website selectors and Predictive H/C/Likelihood V1 production behavior are unchanged.",
        ]
    )
    lines.extend(["", "Measured artifact inventory before this report and its manifest are written:"])
    for item in sorted(
        artifact_inventory,
        key=lambda value: (-int(value["bytes"]), str(value["artifact"])),
    ):
        rows = "" if item["rows"] is None else str(item["rows"])
        lines.append(
            f"- `{item['artifact']}`: {int(item['bytes']):,} bytes, {rows or 'n/a'} rows; {item['purpose']}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--current-cutoff", help="Override the latest checked-in current cutoff"
    )
    parser.add_argument(
        "--method",
        choices=("auto", "prior_stripping", "explicit_neutralized"),
        default="auto",
    )
    parser.add_argument("--inference-max-iterations", type=int, default=100)
    parser.add_argument("--inference-tolerance", type=float, default=1e-6)
    parser.add_argument("--inference-damping", type=float, default=0.35)
    args = parser.parse_args()
    started = time.perf_counter()
    likelihood = load_likelihood(
        ROOT / "data/processed/posterior/historical_likelihood_v1.json"
    )
    corpus = load_corpus(ROOT, (*EVALUATION_SEASONS, CURRENT_SEASON))
    historical_cutoffs = {
        season: standard_cutoffs(corpus.rows_by_season[season], season)
        for season in EVALUATION_SEASONS
    }
    runs: list[CutoffRun] = []
    for season in EVALUATION_SEASONS:
        cutoffs = historical_cutoffs[season]
        for index, cutoff in enumerate(cutoffs):
            print(f"building {season} cutoff {index + 1}/{len(cutoffs)}", flush=True)
            runs.append(
                build_cutoff_run(
                    ROOT,
                    corpus,
                    season,
                    cutoff,
                    likelihood,
                    "prior_stripping",
                    cutoff_index=index,
                    cutoff_count=len(cutoffs),
                    max_iterations=args.inference_max_iterations,
                    tolerance=args.inference_tolerance,
                    damping=args.inference_damping,
                )
            )
    # Validation uses a fixed prospective case panel after all ordinary
    # Context/History runs are available.  It is never selected by stripping
    # error, and the ordinary runs are reused when auto selects stripping.
    bp_rows, bp_summary = run_prior_removal_validation(
        runs,
        likelihood,
        max_iterations=args.inference_max_iterations,
        tolerance=args.inference_tolerance,
        damping=args.inference_damping,
    )
    method: PerformanceMethod = (
        bp_summary["recommendation_if_auto"] if args.method == "auto" else args.method
    )
    if method == "explicit_neutralized":
        runs = []
        for season in EVALUATION_SEASONS:
            cutoffs = historical_cutoffs[season]
            for index, cutoff in enumerate(cutoffs):
                print(
                    f"building {season} explicit cutoff {index + 1}/{len(cutoffs)}",
                    flush=True,
                )
                runs.append(
                    build_cutoff_run(
                        ROOT,
                        corpus,
                        season,
                        cutoff,
                        likelihood,
                        method,
                        cutoff_index=index,
                        cutoff_count=len(cutoffs),
                        max_iterations=args.inference_max_iterations,
                        tolerance=args.inference_tolerance,
                        damping=args.inference_damping,
                    )
                )
    current_requested = (
        _parse_datetime(args.current_cutoff)
        if args.current_cutoff
        else _latest_current_cutoff(ROOT, CURRENT_SEASON)
    )
    print(
        f"building current {CURRENT_SEASON} cutoff {current_requested.isoformat()}",
        flush=True,
    )
    current_run = build_cutoff_run(
        ROOT,
        corpus,
        CURRENT_SEASON,
        current_requested,
        likelihood,
        method,
        cutoff_index=None,
        cutoff_count=None,
        max_iterations=args.inference_max_iterations,
        tolerance=args.inference_tolerance,
        damping=args.inference_damping,
    )
    anchor_rows = anchor_sensitivity_rows(runs)
    predictive_rows = performance_predictive_rows(runs)
    future_predictions = [
        prediction
        for run in runs
        for prediction in future_prediction_rows(run, likelihood)
    ]
    future_key_audit = validate_common_future_keys(future_predictions)
    future_population = future_validation_population(
        runs, future_predictions, future_key_audit
    )
    future_summary_rows = aggregate_future_rows(future_predictions)
    idle = idle_rows(runs)
    cases = illustrative_cases(
        next(
            run for run in runs if run.prepared.season == 2025 and run.cutoff_index == 6
        )
    )
    current_context_rows = ranking_rows(
        current_run.context, current_run.context_rows, current_run
    )
    current_history_rows = ranking_rows(
        current_run.history, current_run.history_rows, current_run
    )
    evidence = [
        evidence_row
        for run in [*runs, current_run]
        for inference, team_rows in (
            (run.context, run.context_rows),
            (run.history, run.history_rows),
        )
        for evidence_row in evidence_rows(inference, team_rows, run)
    ]
    current_cutoff = current_run.prepared.effective_cutoff.isoformat()
    current_context_summary = _current_summary(current_context_rows)
    current_history_summary = _current_summary(current_history_rows)
    current_predictive_rows = performance_predictive_rows([current_run])
    current_predictive_summary = {
        "absolute_expected_rank_difference": _summary_for_rows(
            [
                {"value": abs(float(row["performance_minus_predictive_expected_rank"]))}
                for row in current_predictive_rows
            ],
            "value",
        ),
        "pmf_tv": _summary_for_rows(current_predictive_rows, "pmf_tv"),
        "largest_differences": sorted(
            current_predictive_rows,
            key=lambda row: (
                -abs(float(row["performance_minus_predictive_expected_rank"])),
                str(row["team_name"]),
            ),
        )[:20],
    }
    structural = structural_diagnostics(likelihood)
    anchor_decomposition_rows, anchor_decomposition = anchor_decomposition_summary(
        anchor_rows
    )
    anchor_expected_rank_rows = [
        {
            **row,
            "absolute_expected_rank_difference": abs(
                float(row["context_minus_history_expected_rank"])
            ),
        }
        for row in anchor_rows
    ]
    anchor_summary = {
        "absolute_expected_rank_difference": _summary_for_rows(
            [
                {"value": abs(float(row["context_minus_history_expected_rank"]))}
                for row in anchor_rows
            ],
            "value",
        ),
        "pmf_tv": _summary_for_rows(anchor_rows, "pmf_tv"),
        "by_games_bucket": _group_metric_summary(
            anchor_expected_rank_rows,
            "absolute_expected_rank_difference",
        ),
        "by_games_bucket_pmf_tv": _group_metric_summary(anchor_rows, "pmf_tv"),
        "by_phase": _group_metric_summary(
            anchor_expected_rank_rows,
            "absolute_expected_rank_difference",
        ),
        "by_phase_pmf_tv": _group_metric_summary(anchor_rows, "pmf_tv"),
        "by_week": _group_metric_summary_by_week(
            anchor_expected_rank_rows,
            "absolute_expected_rank_difference",
        ),
        "by_week_pmf_tv": _group_metric_summary_by_week(anchor_rows, "pmf_tv"),
        "largest_disagreements": sorted(
            anchor_rows,
            key=lambda row: (
                -abs(float(row["context_minus_history_expected_rank"])),
                str(row["team_name"]),
            ),
        )[:20],
        "decomposition": anchor_decomposition,
    }
    predictive_summary = {
        "expected_rank_difference": _summary_for_rows(
            predictive_rows, "performance_minus_predictive_expected_rank"
        ),
        "absolute_expected_rank_difference": _summary_for_rows(
            [
                {"value": abs(float(row["performance_minus_predictive_expected_rank"]))}
                for row in predictive_rows
            ],
            "value",
        ),
        "pmf_tv": _summary_for_rows(predictive_rows, "pmf_tv"),
        "by_games_bucket": _group_metric_summary(
            [
                {
                    **row,
                    "absolute_expected_rank_difference": abs(
                        float(row["performance_minus_predictive_expected_rank"])
                    ),
                }
                for row in predictive_rows
            ],
            "absolute_expected_rank_difference",
        ),
        "largest_differences": sorted(
            predictive_rows,
            key=lambda row: (
                -abs(float(row["performance_minus_predictive_expected_rank"])),
                str(row["team_name"]),
            ),
        )[:20],
    }
    overall_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in future_summary_rows:
        overall_groups[(str(row["model"]), str(row["horizon"]))].append(row)
    overall_future: list[dict[str, object]] = []
    for (model, horizon), group in sorted(overall_groups.items()):
        if not group:
            continue
        total = sum(int(item["prediction_count"]) for item in group)
        brier_total = sum(int(item["brier_count"]) for item in group)
        overall_future.append(
            {
                "model": model,
                "horizon": horizon,
                "prediction_count": total,
                "marginalized_nll": _weighted_future_metric(
                    group, "marginalized_nll", "prediction_count", total
                ),
                "margin_absolute_error": _weighted_future_metric(
                    group, "margin_absolute_error", "prediction_count", total
                ),
                "win_brier": _weighted_future_metric(
                    group, "win_brier", "brier_count", brier_total
                ),
                "calibration_absolute_error": _weighted_future_metric(
                    group, "calibration_absolute_error", "prediction_count", total
                ),
            }
        )
    # The durable anchor table is historical by design; current disagreement is
    # emitted separately below from the two current ranking bundles.
    current_anchor_rows = []
    context_by_id = {row["team_id"]: row for row in current_context_rows}
    history_by_id = {row["team_id"]: row for row in current_history_rows}
    for team_id in sorted(set(context_by_id) & set(history_by_id)):
        c, h = context_by_id[team_id], history_by_id[team_id]
        current_anchor_rows.append(
            {
                "team_id": team_id,
                "team_name": c["team_name"],
                "cutoff": current_cutoff,
                "games_played": c["games_played"],
                "context_expected_rank": c["expected_rank"],
                "history_expected_rank": h["expected_rank"],
                "context_minus_history_expected_rank": float(c["expected_rank"])
                - float(h["expected_rank"]),
                "pmf_tv": pmf_tv_distance(
                    current_run.context.pmfs[team_id], current_run.history.pmfs[team_id]
                ),
            }
        )
    summary = {
        "artifact_kind": "performance_v1_research",
        "specification_version": "Performance V1",
        "generated_from": "cached processed/raw CFBD game corpus; no network acquisition",
        "evaluation_seasons": list(EVALUATION_SEASONS),
        "development_seasons": [2018, 2019, 2020, 2021],
        "development_note": "Frozen H/C preseason prediction artifacts begin in 2022; no 2018-2021 priors were fabricated.",
        "primary_anchor_family": "context",
        "performance_method": method,
        "focal_prior": (
            "uniform across applicable FBS rank support; the focal preseason prior "
            "is removed as a direct factor, but approximate loopy BP can retain "
            "indirect cycle feedback"
        ),
        "prior_removal_validation_design": {
            "rule_version": PRIOR_REMOVAL_VALIDATION_RULE_VERSION,
            "selection_uses": [
                "season",
                "fixed cutoff index",
                "games-played bucket",
                "ordinary posterior location and width",
                "ordinary posterior shape",
                "Context/History posterior disagreement",
                "schedule graph degree and cycle-edge exposure",
            ],
            "selection_does_not_use": [
                "prior-stripping error",
                "explicit-neutralization output",
                "future-game outcomes",
            ],
        },
        "zero_game_policy": "uniform PMF and rated=false; zero-game rows have no meaningful display rank",
        "historical_likelihood": "Historical Likelihood V1 unchanged; Student-t df 15; margin/site/subdivision semantics retained",
        "eligible_game_policy": "completed FBS/FCS games at or before cutoff; lower divisions excluded; FCS fallback/support unchanged",
        "current_requested_cutoff": current_run.prepared.requested_cutoff.isoformat(),
        "current_cutoff": current_cutoff,
        "current": current_context_summary,
        "current_history": current_history_summary,
        "current_anchor_disagreements": sorted(
            current_anchor_rows,
            key=lambda row: (
                -abs(float(row["context_minus_history_expected_rank"])),
                str(row["team_name"]),
            ),
        )[:20],
        "current_performance_vs_predictive": current_predictive_summary,
        "bp_prior_removal_validation": bp_summary,
        "structural_diagnostics": structural,
        "anchor_sensitivity": anchor_summary,
        "performance_vs_predictive": predictive_summary,
        "future_validation": {
            "overall": overall_future,
            "row_count": len(future_summary_rows),
            "population": future_population,
        },
        "idle_update": {
            "row_count": len(idle),
            "largest_abs_expected_rank_change": max(
                (abs(float(row["expected_rank_change"])) for row in idle), default=0.0
            ),
            "examples": idle[:20],
        },
        "illustrative_cases": cases,
        "input_hashes": {
            str(season): [
                sha256(path) for path in corpus.source_paths_by_season[season]
            ]
            for season in sorted(corpus.source_paths_by_season)
        },
        "input_coverage": _input_coverage(corpus, ROOT),
        "artifact_status": {
            "inventory_path": "artifact_inventory.csv",
            "runtime_caches_committed": False,
            "pruned_or_compacted": [],
            "retention_note": (
                "Row-level PMFs, predictions, evidence, validation cases, and "
                "aggregates are retained because they support reproducibility "
                "or a specific audit claim; no runtime cache is committed."
            ),
        },
        "runtime_seconds_observed": round(time.perf_counter() - started, 3),
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    ranking_fields = list(current_context_rows[0]) if current_context_rows else []
    pmf_fields = [
        "season",
        "cutoff",
        "anchor_family",
        "method",
        "team_id",
        "team_name",
        "rated",
        "games_played",
        "rank",
        "probability",
    ]
    _write_csv(output / "current_rankings.csv", current_context_rows, ranking_fields)
    _write_csv(
        output / "current_rankings_history.csv",
        current_history_rows,
        list(current_history_rows[0]) if current_history_rows else ranking_fields,
    )
    _write_csv(
        output / "current_pmfs.csv",
        pmf_rows(current_run.context, current_run),
        pmf_fields,
    )
    _write_csv(
        output / "current_pmfs_history.csv",
        pmf_rows(current_run.history, current_run),
        pmf_fields,
    )
    anchor_fields = list(anchor_rows[0]) if anchor_rows else []
    _write_csv(output / "anchor_sensitivity.csv", anchor_rows, anchor_fields)
    decomposition_fields = (
        list(anchor_decomposition_rows[0]) if anchor_decomposition_rows else []
    )
    _write_csv(
        output / "anchor_sensitivity_decomposition.csv",
        anchor_decomposition_rows,
        decomposition_fields,
    )
    _write_csv(
        output / "current_anchor_sensitivity.csv",
        current_anchor_rows,
        list(current_anchor_rows[0]) if current_anchor_rows else [],
    )
    predictive_fields = list(predictive_rows[0]) if predictive_rows else []
    _write_csv(
        output / "performance_vs_predictive.csv", predictive_rows, predictive_fields
    )
    _write_csv(
        output / "current_performance_vs_predictive.csv",
        current_predictive_rows,
        list(current_predictive_rows[0]) if current_predictive_rows else [],
    )
    future_fields = list(future_summary_rows[0]) if future_summary_rows else []
    _write_csv(
        output / "future_game_validation.csv", future_summary_rows, future_fields
    )
    prediction_fields = list(future_predictions[0]) if future_predictions else []
    _write_csv(
        output / "future_game_predictions.csv", future_predictions, prediction_fields
    )
    evidence_fields = list(evidence[0]) if evidence else []
    _write_csv(output / "game_evidence.csv", evidence, evidence_fields)
    _write_csv(
        output / "bp_prior_removal_validation.csv",
        bp_rows,
        list(bp_rows[0]) if bp_rows else [],
    )
    _write_csv(output / "idle_team_examples.csv", idle, list(idle[0]) if idle else [])
    case_fields = sorted({key for row in cases for key in row})
    _write_csv(output / "illustrative_cases.csv", cases, case_fields)
    _write_json(output / "summary.json", summary)
    plots = output / "plots"
    _plot_anchor(anchor_rows, plots / "anchor_sensitivity_by_games.png")
    _plot_performance_predictive(
        predictive_rows, plots / "performance_vs_predictive_by_games.png"
    )
    _plot_uncertainty(current_context_rows, plots / "current_uncertainty_by_games.png")
    _plot_future(future_summary_rows, plots / "future_validation_by_games.png")
    inventory_preview = _artifact_inventory(
        output, exclude=frozenset({"report.md", "artifact_inventory.csv"})
    )
    _report(output / "report.md", summary, current_context_rows, cases, inventory_preview)
    artifact_inventory = _artifact_inventory(output)
    _write_csv(
        output / "artifact_inventory.csv",
        artifact_inventory,
        [
            "artifact",
            "bytes",
            "rows",
            "purpose",
            "generated",
            "needed_for_reproducibility",
            "needed_for_audit",
        ],
    )
    print(f"wrote Performance V1 research artifacts to {output}")
    print(f"selected method: {method}")
    print(f"observed runtime seconds: {time.perf_counter() - started:.3f}")


if __name__ == "__main__":
    main()
