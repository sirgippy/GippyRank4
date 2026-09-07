"""Build the reproducible YPP-vs-margin likelihood investigation.

This is a research-only build.  It reads the already cached historical CFBD
corpus and frozen prior/rank artifacts, writes under
``data/processed/ypp_investigation/``, and never rewrites production snapshots
or model artifacts.

The candidate family is deliberately fixed in this file before the
2022--2025 evaluation:

* Y0: frozen Historical Likelihood V1;
* Y1: rank-independent conditional YPP factor on supported pairings;
* Y2-supported: the corrected conditional YPP factor on FBS-FBS and FBS-FCS;
* Y2-all-pairings-original: the retained invalid PR #17 diagnostic; and
* a naïve independent YPP diagnostic, never a promotion candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import t as student_t

from gippyrank.modeling import read_csv
from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    PosteriorResult,
    Team,
    game_margin_parameters,
)
from gippyrank.posterior.snapshots import load_likelihood, load_teams
from gippyrank.preseason import pmf_summaries
from gippyrank.research.ypp_likelihood import (
    DF_GRID,
    SUPPORTED_YPP_PAIRINGS,
    YPP_PAIRING_POLICY,
    YPPData,
    build_ypp_data,
    fit_ypp_model,
    infer_posterior_with_ypp,
    oriented_margin,
    oriented_rank_coordinates,
    oriented_ypp_difference,
    pairing_for,
    ypp_model_locations,
    ypp_model_scores,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/ypp_investigation"
HISTORICAL_ROWS_PATH = ROOT / "data/processed/modeling/historical_modeling_games.csv"
RANK_DISTRIBUTIONS_PATH = (
    ROOT / "data/processed/modeling/team_season_rank_distributions.csv"
)

TRAIN_YEARS = tuple(range(2004, 2018))
DEVELOPMENT_YEARS = tuple(range(2018, 2022))
FINAL_YEARS = (2022, 2023, 2024, 2025)
ALL_HISTORICAL_YEARS = tuple(range(2003, 2027))
CUTOFF_FRACTIONS = (0.0, 0.20, 0.35, 0.55, 0.72, 0.87, 1.0)
NAIVE_VARIANT = "naive_independent_diagnostic"
Y2_ORIGINAL_VARIANT = "y2_all_pairings_original"
Y2_SUPPORTED_VARIANT = "y2_supported"
SUPPORTED_YPP_OBSERVED_VIEW = "supported_ypp_observed"
POSTERIOR_VARIANTS = ("v1", "y1", Y2_ORIGINAL_VARIANT, Y2_SUPPORTED_VARIANT)
_FUTURE_SURFACE_CACHE: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray, float]] = {}

# These thresholds are declared in source before the final test is read.  They
# are intentionally about downstream rank quality and calibration, not joint
# density NLL for an extra observed variable.
PROMOTION_CRITERIA: dict[str, object] = {
    "aggregate_final_rank_nll_improvement_nats_per_team": 0.02,
    "maximum_allowed_aggregate_crps_degradation": 0.002,
    "maximum_allowed_aggregate_interval_coverage_drop": 0.03,
    "minimum_held_out_seasons_with_nll_improvement": 3,
    "maximum_allowed_single_season_nll_degradation": 0.10,
    "maximum_allowed_single_season_crps_degradation": 0.02,
    "maximum_allowed_future_margin_mae_degradation_points": 0.50,
    "maximum_allowed_future_margin_nll_degradation_nats": 0.02,
    "minimum_common_subset_seasons_with_positive_quality_slope": 3,
}


def _bool(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _period(season: int) -> str:
    if season in TRAIN_YEARS:
        return "training_2004_2017"
    if season in DEVELOPMENT_YEARS:
        return "development_2018_2021"
    if season in FINAL_YEARS:
        return "final_2022_2025"
    return "outside_evaluation"


def _parse_date(value: object) -> datetime:
    text = str(value).replace("Z", "+00:00")
    result = datetime.fromisoformat(text)
    return result if result.tzinfo is not None else result.replace(tzinfo=UTC)


def _site_label(row: Mapping[str, object]) -> str:
    return "neutral" if _bool(row.get("neutral_site")) else "home_site"


def _row_pairing(row: Mapping[str, object]) -> str:
    return pairing_for(str(row["home_subdivision"]), str(row["away_subdivision"]))


def _usable_ypp_difference(row: Mapping[str, object]) -> float | None:
    existing = _float(row.get("ypp_diff"))
    if existing is not None:
        return existing
    return oriented_ypp_difference(
        str(row["home_subdivision"]),
        str(row["away_subdivision"]),
        row.get("home_ypp"),
        row.get("away_ypp"),
    )


def is_supported_ypp_observed(row: Mapping[str, object]) -> bool:
    """Return whether a game belongs to the matched supported-YPP view."""

    return (
        _row_pairing(row) in SUPPORTED_YPP_PAIRINGS
        and _usable_ypp_difference(row) is not None
    )


def select_population_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    season: int,
    cutoff: datetime,
    view: str,
) -> list[Mapping[str, object]]:
    """Select one cutoff population without changing the primary full view."""

    if view not in {"full", SUPPORTED_YPP_OBSERVED_VIEW}:
        raise ValueError(f"unknown population view: {view}")
    return [
        row
        for row in rows
        if int(row["season"]) == season
        and _parse_date(row["start_date"]) <= cutoff
        and (view == "full" or is_supported_ypp_observed(row))
    ]


def _normalised_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def _slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.var(x) <= 1e-14:
        return None
    return float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))


def _safe(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _stat_value(stats: Sequence[Mapping[str, object]], category: str) -> str | None:
    for item in stats:
        if item.get("category") == category:
            value = item.get("stat")
            return None if value is None else str(value)
    return None


def _parse_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _derived_stat_row(game: Mapping[str, object], team: Mapping[str, object]) -> dict[str, object]:
    stats = team.get("stats") or []
    total_yards = _parse_int(_stat_value(stats, "totalYards"))
    rushing_attempts = _parse_int(_stat_value(stats, "rushingAttempts"))
    completion_attempts = _stat_value(stats, "completionAttempts")
    pass_attempts = None
    if completion_attempts and "-" in completion_attempts:
        pass_attempts = _parse_int(completion_attempts.rsplit("-", 1)[1])
    plays = (
        rushing_attempts + pass_attempts
        if rushing_attempts is not None and pass_attempts is not None
        else None
    )
    ypp = (
        total_yards / plays
        if total_yards is not None and plays is not None and plays > 0
        else None
    )
    return {
        "game_id": int(game["id"]),
        "team_id": int(team["teamId"]),
        "team": str(team.get("team") or ""),
        "total_yards": total_yards,
        "rushing_attempts": rushing_attempts,
        "completion_attempts": completion_attempts,
        "pass_attempts": pass_attempts,
        "sacks": _parse_int(_stat_value(stats, "sacks")),
        "plays": plays,
        "yards_per_play": ypp,
    }


def load_raw_games() -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Load and validate the immutable raw schedule payloads."""

    records: dict[str, dict[str, object]] = {}
    overlap_count = 0
    conflicts: list[dict[str, object]] = []
    for path in sorted((ROOT / "data/raw/cfbd/games").glob("*.json")):
        if path.name.endswith(".provenance.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for game in payload:
            key = str(game["id"])
            previous = records.get(key)
            if previous is None:
                records[key] = game
            elif previous == game:
                overlap_count += 1
            else:
                conflicts.append({"game_id": key, "first": previous, "second": game})
    return records, {
        "raw_schedule_unique_games": len(records),
        "raw_schedule_exact_overlaps": overlap_count,
        "raw_schedule_conflicts": len(conflicts),
        "schedule_conflict_examples": conflicts[:3],
    }


def load_raw_stats() -> tuple[dict[tuple[int, int], dict[str, object]], dict[str, object]]:
    """Derive the current YPP rows while auditing every raw response.

    Provenance sidecars are intentionally excluded here.  The historical
    corpus builder's broad ``*.json`` glob would otherwise try to parse a
    provenance object as a game response; that operational issue is reported,
    but this audit does not rewrite the corpus or change the derivation.
    """

    rows: dict[tuple[int, int], dict[str, object]] = {}
    duplicate_count = 0
    conflicts: list[dict[str, object]] = []
    sources: dict[tuple[int, int], list[str]] = defaultdict(list)
    category_counts: Counter[str] = Counter()
    category_values: dict[str, Counter[str]] = defaultdict(Counter)
    payload_game_count = 0
    team_row_count = 0
    file_count = 0
    current_season_stat_files_by_classification: Counter[str] = Counter()
    for path in sorted((ROOT / "data/raw/cfbd/game_stats").glob("*.json")):
        if path.name.endswith(".provenance.json"):
            continue
        file_count += 1
        path_parts = path.stem.split("-")
        if path_parts and path_parts[0] == "2026" and len(path_parts) > 1:
            current_season_stat_files_by_classification[path_parts[1]] += 1
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload_game_count += len(payload)
        for game in payload:
            for team in game.get("teams") or []:
                team_row_count += 1
                stats = team.get("stats") or []
                for item in stats:
                    category = str(item.get("category") or "")
                    category_counts[category] += 1
                    if item.get("stat") is not None:
                        category_values[category][str(item["stat"])] += 1
                if team.get("teamId") is None:
                    continue
                row = _derived_stat_row(game, team)
                key = (int(row["game_id"]), int(row["team_id"]))
                sources[key].append(path.name)
                previous = rows.get(key)
                if previous is None:
                    rows[key] = row
                elif previous == row:
                    duplicate_count += 1
                else:
                    conflicts.append(
                        {
                            "game_id": row["game_id"],
                            "team_id": row["team_id"],
                            "first": previous,
                            "second": row,
                            "source": path.name,
                        }
                    )
    duplicate_keys = sum(len(set(value)) > 1 for value in sources.values())
    return rows, {
        "raw_stat_files": file_count,
        "raw_stat_payload_games": payload_game_count,
        "raw_stat_team_rows": team_row_count,
        "raw_stat_unique_team_game_rows": len(rows),
        "raw_stat_exact_duplicate_rows": duplicate_count,
        "raw_stat_duplicate_keys_from_multiple_files": duplicate_keys,
        "current_season_stat_files": sum(current_season_stat_files_by_classification.values()),
        "current_season_stat_files_by_classification": dict(
            sorted(current_season_stat_files_by_classification.items())
        ),
        "raw_stat_conflicts": len(conflicts),
        "stat_conflict_examples": conflicts[:3],
        "category_counts": dict(sorted(category_counts.items())),
        "category_value_examples": {
            key: value.most_common(8)
            for key, value in sorted(category_values.items())
            if key
            in {
                "yardsPerPlay",
                "plays",
                "offensivePlays",
                "totalPlays",
                "totalYards",
                "rushingAttempts",
                "completionAttempts",
                "sacks",
            }
        },
        "direct_yards_per_play_category_present": bool(category_counts["yardsPerPlay"]),
        "direct_plays_category_present": bool(
            category_counts["plays"]
            or category_counts["offensivePlays"]
            or category_counts["totalPlays"]
        ),
    }


def _pairing_from_game(game: Mapping[str, object]) -> str:
    return pairing_for(str(game["homeClassification"]), str(game["awayClassification"]))


def _eligible_game(game: Mapping[str, object]) -> bool:
    return bool(
        game.get("completed")
        and game.get("homePoints") is not None
        and game.get("awayPoints") is not None
        and str(game.get("homeClassification", "")).casefold() in {"fbs", "fcs"}
        and str(game.get("awayClassification", "")).casefold() in {"fbs", "fcs"}
    )


def build_coverage_audit(
    games: Mapping[str, Mapping[str, object]],
    stats: Mapping[tuple[int, int], Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    coverage_rows: list[dict[str, object]] = []
    seasons: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for game in games.values():
        if _eligible_game(game):
            seasons[int(game["season"])].append(game)
    all_ypp: list[float] = []
    season_summary: list[dict[str, object]] = []
    for season in sorted(seasons):
        season_games = seasons[season]
        pairings = sorted({_pairing_from_game(game) for game in season_games})
        for pairing in [*pairings, "all"]:
            selected = [
                game
                for game in season_games
                if pairing == "all" or _pairing_from_game(game) == pairing
            ]
            rows_expected = 2 * len(selected)
            usable_rows = 0
            both = 0
            any_usable = 0
            missing_home = missing_away = missing_both = 0
            for game in selected:
                home_key = (int(game["id"]), int(game["homeId"]))
                away_key = (int(game["id"]), int(game["awayId"]))
                home_value = stats.get(home_key, {}).get("yards_per_play")
                away_value = stats.get(away_key, {}).get("yards_per_play")
                usable = int(home_value is not None) + int(away_value is not None)
                usable_rows += usable
                any_usable += int(usable > 0)
                both += int(usable == 2)
                if usable == 0:
                    missing_both += 1
                elif home_value is None:
                    missing_home += 1
                elif away_value is None:
                    missing_away += 1
                if pairing == "all":
                    if home_value is not None:
                        all_ypp.append(float(home_value))
                    if away_value is not None:
                        all_ypp.append(float(away_value))
            coverage_rows.append(
                {
                    "season": season,
                    "pairing": pairing,
                    "games": len(selected),
                    "games_with_any_usable_ypp": any_usable,
                    "games_with_both_usable_ypp": both,
                    "team_game_rows_expected": rows_expected,
                    "team_game_rows_with_usable_ypp": usable_rows,
                    "team_game_coverage_pct": round(
                        100 * usable_rows / rows_expected, 4
                    )
                    if rows_expected
                    else 0.0,
                    "game_coverage_pct": round(100 * both / len(selected), 4)
                    if selected
                    else 0.0,
                    "missing_both": missing_both,
                    "missing_home_only": missing_home,
                    "missing_away_only": missing_away,
                }
            )
        summary = next(
            row
            for row in coverage_rows
            if row["season"] == season and row["pairing"] == "all"
        )
        season_summary.append(summary)
    values = np.asarray(all_ypp, dtype=float)
    extremes = {
        "count": len(values),
        "minimum": float(np.min(values)) if len(values) else None,
        "maximum": float(np.max(values)) if len(values) else None,
        "negative_count": int(np.sum(values < 0)) if len(values) else 0,
        "zero_count": int(np.sum(values == 0)) if len(values) else 0,
        "below_one_count": int(np.sum(values < 1)) if len(values) else 0,
        "above_twelve_count": int(np.sum(values > 12)) if len(values) else 0,
        "quantiles": {
            str(q): float(np.quantile(values, q))
            for q in (0.001, 0.01, 0.5, 0.99, 0.999)
        }
        if len(values)
        else {},
    }
    return coverage_rows, {
        "season_summary": season_summary,
        "extreme_ypp_values": extremes,
        "coverage_materially_changes_over_time": bool(
            season_summary
            and max(float(row["team_game_coverage_pct"]) for row in season_summary)
            - min(float(row["team_game_coverage_pct"]) for row in season_summary)
            > 20
        ),
    }


def summarise_pairing_coverage(
    coverage_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate raw pairing coverage over the declared temporal periods."""

    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in coverage_rows:
        pairing = str(row["pairing"])
        if pairing not in {"fbs-fbs", "fbs-fcs", "fcs-fcs"}:
            continue
        grouped[(_period(int(row["season"])), pairing)].append(row)
    output: list[dict[str, object]] = []
    for (period, pairing), values in sorted(grouped.items()):
        games = sum(int(row["games"]) for row in values)
        both = sum(int(row["games_with_both_usable_ypp"]) for row in values)
        expected = sum(int(row["team_game_rows_expected"]) for row in values)
        usable = sum(int(row["team_game_rows_with_usable_ypp"]) for row in values)
        output.append(
            {
                "period": period,
                "pairing": pairing,
                "seasons": ",".join(str(row["season"]) for row in values),
                "games": games,
                "games_with_both_usable_ypp": both,
                "game_coverage_pct": round(100 * both / games, 4) if games else 0.0,
                "team_game_rows_expected": expected,
                "team_game_rows_with_usable_ypp": usable,
                "team_game_coverage_pct": round(100 * usable / expected, 4)
                if expected
                else 0.0,
            }
        )
    return output


def load_historical_rows() -> list[dict[str, object]]:
    rows = [dict(row) for row in read_csv(HISTORICAL_ROWS_PATH)]
    for row in rows:
        row["season"] = int(row["season"])
        row["game_id"] = str(row["game_id"])
        row["home_points"] = int(row["home_points"])
        row["away_points"] = int(row["away_points"])
        row["neutral_site"] = _bool(row["neutral_site"])
        row["start_date"] = str(row["start_date"])
    return rows


def load_rank_targets() -> dict[tuple[int, str, str], dict[str, object]]:
    targets: dict[tuple[int, str, str], dict[str, object]] = {}
    for row in read_csv(RANK_DISTRIBUTIONS_PATH):
        population = int(row["team_population"])
        items = json.loads(row["pmf"])
        # A small pre-existing lower-division/source-population inconsistency
        # has a few observed ranks above the stored population.  Preserve the
        # source population for all percentile semantics, but allocate enough
        # target support to retain the observed mass rather than dropping it.
        support = max([population, *(int(item["rank"]) for item in items)])
        pmf = np.zeros(support, dtype=float)
        for item in items:
            pmf[int(item["rank"]) - 1] = float(item["probability"])
        targets[(int(row["season"]), row["subdivision"], row["team_id"])] = {
            "population": population,
            "mean_rank": float(row["rank_mean"]),
            "pmf": pmf,
            "team_name": row["team_name"],
        }
    return targets


def load_processed_ypp() -> dict[tuple[int, int], dict[str, object]]:
    result: dict[tuple[int, int], dict[str, object]] = {}
    path = ROOT / "data/processed/cfbd/team_game_stats.csv"
    if not path.exists():
        return result
    for row in read_csv(path):
        key = (int(row["game_id"]), int(row["team_id"]))
        result[key] = {
            "total_yards": _parse_int(row["total_yards"]),
            "plays": _parse_int(row["plays"]),
            "yards_per_play": _float(row["yards_per_play"]),
        }
    return result


def audit_processed_derivation(
    raw_stats: Mapping[tuple[int, int], Mapping[str, object]],
    processed_stats: Mapping[tuple[int, int], Mapping[str, object]],
) -> dict[str, object]:
    comparable = set(raw_stats) & set(processed_stats)
    mismatches: list[dict[str, object]] = []
    for key in sorted(comparable):
        raw = raw_stats[key]
        processed = processed_stats[key]
        if (
            raw.get("total_yards") != processed.get("total_yards")
            or raw.get("plays") != processed.get("plays")
            or (
                raw.get("yards_per_play") is None
                and processed.get("yards_per_play") is not None
            )
            or (
                raw.get("yards_per_play") is not None
                and processed.get("yards_per_play") is None
            )
            or (
                raw.get("yards_per_play") is not None
                and processed.get("yards_per_play") is not None
                and not math.isclose(
                    float(raw["yards_per_play"]),
                    float(processed["yards_per_play"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        ):
            mismatches.append({"key": key, "raw": raw, "processed": processed})
    return {
        "processed_team_game_rows": len(processed_stats),
        "raw_processed_common_rows": len(comparable),
        "raw_processed_derivation_mismatches": len(mismatches),
        "raw_processed_mismatch_examples": mismatches[:3],
        "raw_rows_not_in_processed": len(set(raw_stats) - set(processed_stats)),
        "processed_rows_not_in_raw": len(set(processed_stats) - set(raw_stats)),
    }


def build_enriched_rows(
    rows: Sequence[Mapping[str, object]],
    games: Mapping[str, Mapping[str, object]],
    targets: Mapping[tuple[int, str, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    """Attach names, season type, target quality, and oriented coordinates."""

    enriched: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        game = games.get(str(row["game_id"]), {})
        row["home_team"] = game.get("homeTeam", row["home_team_id"])
        row["away_team"] = game.get("awayTeam", row["away_team_id"])
        row["season_type"] = game.get("seasonType", "unknown")
        row["conference_game"] = _bool(game.get("conferenceGame"))
        row["pairing"] = pairing_for(
            str(row["home_subdivision"]), str(row["away_subdivision"])
        )
        row["site"] = _site_label(row)
        ypp = oriented_ypp_difference(
            str(row["home_subdivision"]),
            str(row["away_subdivision"]),
            row.get("home_ypp"),
            row.get("away_ypp"),
        )
        row["ypp_diff"] = ypp
        home_target = targets.get(
            (int(row["season"]), str(row["home_subdivision"]), str(row["home_team_id"]))
        )
        away_target = targets.get(
            (int(row["season"]), str(row["away_subdivision"]), str(row["away_team_id"]))
        )
        row["home_final_rank_mean"] = (
            home_target["mean_rank"] if home_target is not None else None
        )
        row["away_final_rank_mean"] = (
            away_target["mean_rank"] if away_target is not None else None
        )
        if home_target is not None and away_target is not None:
            x, y = oriented_rank_coordinates(
                str(row["home_subdivision"]),
                str(row["away_subdivision"]),
                float(home_target["mean_rank"]),
                float(away_target["mean_rank"]),
                int(row["home_team_population"]),
                int(row["away_team_population"]),
            )
            row["final_rank_x"] = x
            row["final_rank_y"] = y
            row["final_quality_advantage"] = y - x
        else:
            row["final_rank_x"] = None
            row["final_rank_y"] = None
            row["final_quality_advantage"] = None
        enriched.append(row)
    return enriched


def _rank_quality_value(
    row: Mapping[str, object], targets: Mapping[tuple[int, str, str], Mapping[str, object]]
) -> tuple[float, float] | None:
    home = targets.get(
        (int(row["season"]), str(row["home_subdivision"]), str(row["home_team_id"]))
    )
    away = targets.get(
        (int(row["season"]), str(row["away_subdivision"]), str(row["away_team_id"]))
    )
    if home is None or away is None:
        return None
    x, y = oriented_rank_coordinates(
        str(row["home_subdivision"]),
        str(row["away_subdivision"]),
        float(home["mean_rank"]),
        float(away["mean_rank"]),
        int(row["home_team_population"]),
        int(row["away_team_population"]),
    )
    return x, y


def _future_margin_for_team(
    row: Mapping[str, object], team_id: str
) -> float:
    home = str(row["home_team_id"])
    margin = int(row["home_points"]) - int(row["away_points"])
    return float(margin if team_id == home else -margin)


def attach_next_game_performance(rows: Sequence[Mapping[str, object]]) -> None:
    """Mutate enriched rows with strict-after-date next-game margins."""

    by_team: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        item = row if isinstance(row, dict) else dict(row)
        by_team[str(item["home_team_id"])].append(item)
        by_team[str(item["away_team_id"])].append(item)
    for entries in by_team.values():
        entries.sort(key=lambda item: (_parse_date(item["start_date"]), str(item["game_id"])))
    for row in rows:
        if not isinstance(row, dict):
            continue
        current_date = _parse_date(row["start_date"])
        home_id = str(row["home_team_id"])
        away_id = str(row["away_team_id"])
        first_id, second_id = (
            (home_id, away_id)
            if str(row["home_subdivision"]) == str(row["away_subdivision"])
            or str(row["home_subdivision"]) == "fbs"
            else (away_id, home_id)
        )

        def next_game(team_id: str, current: datetime) -> dict[str, object] | None:
            for candidate in by_team[team_id]:
                candidate_date = _parse_date(candidate["start_date"])
                if candidate_date > current:
                    return candidate
            return None

        first_game = next_game(first_id, current_date)
        second_game = next_game(second_id, current_date)
        if first_game is None or second_game is None:
            row["next_margin_diff"] = None
            row["next_first_margin"] = None
            row["next_second_margin"] = None
        else:
            first_margin = _future_margin_for_team(first_game, first_id)
            second_margin = _future_margin_for_team(second_game, second_id)
            row["next_first_margin"] = first_margin
            row["next_second_margin"] = second_margin
            row["next_margin_diff"] = first_margin - second_margin


def _model_location_for_row(
    model: Mapping[str, object], row: Mapping[str, object], x: float = 0.0, y: float = 0.0
) -> float:
    pairing = np.asarray([str(row["pairing"])])
    neutral = np.asarray([float(_bool(row["neutral_site"]))])
    cross = str(row["home_subdivision"]) != str(row["away_subdivision"])
    fbs_home = np.asarray(
        [float(cross and str(row["home_subdivision"]) == "fbs" and not neutral[0])]
    )
    locations = ypp_model_locations(
        model,
        margin=np.asarray(
            [
                oriented_margin(
                    str(row["home_subdivision"]),
                    str(row["away_subdivision"]),
                    int(row["home_points"]),
                    int(row["away_points"]),
                )
            ]
        ),
        pairing=pairing,
        neutral=neutral,
        fbs_home=fbs_home,
        x=np.asarray([x]),
        y=np.asarray([y]),
    )
    return float(locations[0])


def build_signal_rows(
    rows: Sequence[Mapping[str, object]],
    y1_model: Mapping[str, object],
    y2_model: Mapping[str, object],
    targets: Mapping[tuple[int, str, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    """Create row-level residual and future-quality diagnostics."""

    result: list[dict[str, object]] = []
    for source in rows:
        if source.get("ypp_diff") is None:
            continue
        row = dict(source)
        y1_location = _model_location_for_row(y1_model, row)
        quality = _rank_quality_value(row, targets)
        if quality is not None:
            x, y = quality
            y2_location = _model_location_for_row(y2_model, row, x, y)
            row["rank_effect_at_final_quality"] = y2_location - y1_location
            row["final_quality_advantage"] = y - x
        else:
            y2_location = None
            row["rank_effect_at_final_quality"] = None
            row["final_quality_advantage"] = None
        row["conditional_ypp_location"] = y1_location
        row["conditional_ypp_residual"] = float(row["ypp_diff"]) - y1_location
        row["rank_signal_location"] = y2_location
        row["period"] = _period(int(row["season"]))
        result.append(row)
    return result


def _margin_bin(margin: float) -> str:
    if margin <= -28:
        return "<=-28"
    if margin <= -14:
        return "(-28,-14]"
    if margin <= 0:
        return "(-14,0]"
    if margin <= 14:
        return "(0,14]"
    if margin <= 28:
        return "(14,28]"
    return ">28"


def _residual_bin(value: float, edges: np.ndarray) -> str:
    index = int(np.searchsorted(edges, value, side="right"))
    labels = ("q1", "q2", "q3", "q4", "q5")
    return labels[min(max(index, 0), len(labels) - 1)]


def build_conditional_signal_table(signal_rows: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    residuals = np.asarray(
        [
            float(row["conditional_ypp_residual"])
            for row in signal_rows
            if int(row["season"]) in (*TRAIN_YEARS, *DEVELOPMENT_YEARS)
        ],
        dtype=float,
    )
    edges = np.quantile(residuals, [0.2, 0.4, 0.6, 0.8]) if len(residuals) else np.zeros(4)
    group_values: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in signal_rows:
        margin = oriented_margin(
            str(row["home_subdivision"]),
            str(row["away_subdivision"]),
            int(row["home_points"]),
            int(row["away_points"]),
        )
        key = (
            row["period"],
            int(row["season"]),
            row["pairing"],
            row["site"],
            _margin_bin(margin),
            _residual_bin(float(row["conditional_ypp_residual"]), edges),
        )
        group_values[key].append(row)
    output: list[dict[str, object]] = []
    for key, values in sorted(group_values.items(), key=lambda item: tuple(str(v) for v in item[0])):
        quality_pairs = [
            (float(row["conditional_ypp_residual"]), float(row["final_quality_advantage"]))
            for row in values
            if row.get("final_quality_advantage") is not None
        ]
        residual = np.asarray([float(row["conditional_ypp_residual"]) for row in values])
        quality_residual = np.asarray([pair[0] for pair in quality_pairs])
        quality = np.asarray([pair[1] for pair in quality_pairs])
        next_diff = np.asarray(
            [float(row["next_margin_diff"]) for row in values if row.get("next_margin_diff") is not None]
        )
        output.append(
            {
                "summary_type": "margin_residual_bin",
                "period": key[0],
                "season": key[1],
                "pairing": key[2],
                "site": key[3],
                "margin_bin": key[4],
                "residual_bin": key[5],
                "n_games": len(values),
                "mean_residual": float(np.mean(residual)),
                "mean_final_quality_advantage": float(np.mean(quality)) if len(quality) else None,
                "better_quality_rate": float(np.mean(quality > 0)) if len(quality) else None,
                "mean_next_margin_diff": float(np.mean(next_diff)) if len(next_diff) else None,
                "quality_spearman": _normalised_correlation(quality_residual, quality)
                if len(quality)
                else None,
            }
        )
    return output, {
        "residual_quintile_edges": edges.tolist(),
        "period_pairing": _signal_summaries(signal_rows),
    }


def _signal_summaries(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(row["period"], int(row["season"]), row["pairing"], row["site"])].append(row)
    for key, values in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        quality_pairs = [
            (float(row["conditional_ypp_residual"]), float(row["final_quality_advantage"]))
            for row in values
            if row.get("final_quality_advantage") is not None
        ]
        next_pairs = [
            (float(row["conditional_ypp_residual"]), float(row["next_margin_diff"]))
            for row in values
            if row.get("next_margin_diff") is not None
        ]
        quality_residual = np.asarray([pair[0] for pair in quality_pairs])
        quality = np.asarray([pair[1] for pair in quality_pairs])
        next_residual = np.asarray([pair[0] for pair in next_pairs])
        next_diff = np.asarray([pair[1] for pair in next_pairs])
        output.append(
            {
                "period": key[0],
                "season": key[1],
                "pairing": key[2],
                "site": key[3],
                "n_games": len(values),
                "quality_spearman": _normalised_correlation(quality_residual, quality)
                if len(quality)
                else None,
                "quality_slope": _slope(quality_residual, quality)
                if len(quality)
                else None,
                "next_margin_spearman": _normalised_correlation(next_residual, next_diff)
                if len(next_diff)
                else None,
                "next_margin_slope": _slope(next_residual, next_diff)
                if len(next_diff)
                else None,
                "mean_rank_effect_at_final_quality": float(
                    np.mean(
                        [
                            float(row["rank_effect_at_final_quality"])
                            for row in values
                            if row.get("rank_effect_at_final_quality") is not None
                        ]
                    )
                )
                if any(row.get("rank_effect_at_final_quality") is not None for row in values)
                else None,
            }
        )
    return output


def select_ypp_models(
    data: YPPData,
    *,
    allowed_pairings: Iterable[str] | None = SUPPORTED_YPP_PAIRINGS,
    model_panel: str = "supported_pairings",
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]], dict[str, object]]:
    allowed = (
        frozenset(data.pairing.tolist())
        if allowed_pairings is None
        else frozenset(str(value) for value in allowed_pairings)
    )
    pairing_mask = np.isin(data.pairing, tuple(sorted(allowed)))
    train = np.isin(data.season, TRAIN_YEARS) & pairing_mask
    development = np.isin(data.season, DEVELOPMENT_YEARS) & pairing_mask
    pretest = (data.season < 2022) & pairing_mask
    if not np.any(train):
        raise ValueError(f"{model_panel} has no training YPP observations")
    models: dict[str, dict[str, object]] = {}
    selection_rows: list[dict[str, object]] = []
    selected: dict[str, object] = {}
    for name, rank_signal, include_margin in (
        ("y1", False, True),
        ("y2", True, True),
        ("naive", True, False),
    ):
        scored: list[tuple[float, float, dict[str, object]]] = []
        for df in DF_GRID:
            fit = fit_ypp_model(
                data,
                train,
                rank_signal=rank_signal,
                include_margin=include_margin,
                degrees_of_freedom=df,
                allowed_pairings=allowed,
            )
            train_scores = ypp_model_scores(fit, data, train)
            development_scores = ypp_model_scores(fit, data, development)
            scored.append((float(development_scores["marginalized_nll"]), df, fit))
            selection_rows.append(
                {
                    "candidate": name,
                    "model_panel": model_panel,
                    "allowed_pairings": ",".join(sorted(allowed)),
                    "student_t_df": df,
                    "train_games": train_scores["n_games"],
                    "development_games": development_scores["n_games"],
                    "train_conditional_nll": train_scores["expected_conditional_nll"],
                    "train_marginalized_nll": train_scores["marginalized_nll"],
                    "development_conditional_nll": development_scores["expected_conditional_nll"],
                    "development_marginalized_nll": development_scores["marginalized_nll"],
                    "selected": False,
                }
            )
        selected_dev_nll, selected_df, _ = min(scored, key=lambda item: (item[0], item[1]))
        final_fit = fit_ypp_model(
            data,
            pretest,
            rank_signal=rank_signal,
            include_margin=include_margin,
            degrees_of_freedom=selected_df,
            allowed_pairings=allowed,
        )
        final_fit["model_panel"] = model_panel
        models[name] = final_fit
        selected[name] = {
            "model_panel": model_panel,
            "student_t_df": selected_df,
            "development_marginalized_nll": selected_dev_nll,
            "rank_signal": rank_signal,
            "include_margin": include_margin,
            "selection_rule": "minimum development equal-game marginalized YPP NLL; ties choose lower df",
            "fit_pairings": sorted(allowed),
            "training_seasons": "2004-2017",
            "development_seasons": "2018-2021",
            "final_fit_seasons": "2004-2021",
            "final_fit_game_count": final_fit["fit_game_count"],
            "final_fit_pseudo_observation_count": final_fit[
                "fit_pseudo_observation_count"
            ],
        }
        for row in selection_rows:
            if row["candidate"] == name and row["student_t_df"] == selected_df:
                row["selected"] = True
    return models, selection_rows, selected


def standard_cutoffs(
    rows: Sequence[Mapping[str, object]], season: int
) -> list[tuple[str, datetime]]:
    dates = sorted(
        {
            str(row["start_date"])[:10]
            for row in rows
            if int(row["season"]) == season and str(row.get("season_type")) == "regular"
        }
    )
    if not dates:
        dates = sorted(
            {str(row["start_date"])[:10] for row in rows if int(row["season"]) == season}
        )
    if not dates:
        raise ValueError(f"no dates available for {season}")
    indexes = [round(fraction * (len(dates) - 1)) for fraction in CUTOFF_FRACTIONS]
    # Distinct labels are retained if unusual source dates collapse adjacent
    # quantiles; game populations and comparison keys remain explicit.
    result: list[tuple[str, datetime]] = []
    seen: set[str] = set()
    for ordinal, index in enumerate(indexes, 1):
        date = dates[index]
        if date in seen:
            continue
        seen.add(date)
        result.append((f"q{ordinal}_{date}", datetime.fromisoformat(f"{date}T23:59:59+00:00")))
    return result


def _team_name_lookup(
    games: Mapping[str, Mapping[str, object]], rows: Sequence[Mapping[str, object]]
) -> dict[str, str]:
    names: dict[str, str] = {}
    for game in games.values():
        for side in ("home", "away"):
            if game.get(f"{side}Id") is not None:
                names[str(game[f"{side}Id"])] = str(game.get(f"{side}Team") or game[f"{side}Id"])
    for row in rows:
        names.setdefault(str(row["home_team_id"]), str(row.get("home_team") or row["home_team_id"]))
        names.setdefault(str(row["away_team_id"]), str(row.get("away_team") or row["away_team_id"]))
    return names


def _fcs_population(
    targets: Mapping[tuple[int, str, str], Mapping[str, object]], season: int
) -> int:
    values = {
        int(value["population"])
        for (target_season, subdivision, _team_id), value in targets.items()
        if target_season == season and subdivision == "fcs"
    }
    if len(values) != 1:
        raise ValueError(f"expected one FCS population for {season}, found {values}")
    return values.pop()


def make_teams(
    season: int,
    family: str,
    rows: Sequence[Mapping[str, object]],
    targets: Mapping[tuple[int, str, str], Mapping[str, object]],
    root: Path = ROOT,
) -> list[Team]:
    """Load the frozen FBS prior and add uniform, explicit FCS supports."""

    teams, _metadata, _path = load_teams(root, season, family)  # type: ignore[arg-type]
    known = {team.team_id for team in teams}
    name_lookup = _team_name_lookup({}, rows)
    fcs_ids = {
        str(row[f"{side}_team_id"])
        for row in rows
        for side in ("home", "away")
        if str(row[f"{side}_subdivision"]) == "fcs"
    }
    population = _fcs_population(targets, season)
    for team_id in sorted(fcs_ids - known):
        teams.append(
            Team(
                team_id,
                name_lookup.get(team_id, team_id),
                "fcs",
                np.full(population, 1.0 / population),
            )
        )
    return teams


def row_to_game(row: Mapping[str, object]) -> Game:
    return Game(
        str(row["game_id"]),
        str(row["home_team_id"]),
        str(row["away_team_id"]),
        str(row["home_subdivision"]),  # type: ignore[arg-type]
        str(row["away_subdivision"]),  # type: ignore[arg-type]
        int(row["home_points"]),
        int(row["away_points"]),
        bool(row["neutral_site"]),
    )


def _game_key_hash(team_ids: Iterable[str]) -> str:
    value = "\n".join(sorted(set(team_ids))).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _game_population_hash(rows: Sequence[Mapping[str, object]]) -> str:
    value = "\n".join(sorted(str(row["game_id"]) for row in rows)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _comparison_group_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(row["season"]),
        int(row["cutoff_index"]),
        str(row["prior_family"]),
        str(row["population_view"]),
    )


def validate_common_comparison_keys(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Require identical game and strict FBS target keys across candidates."""

    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_comparison_group_key(row)].append(row)
    audits: list[dict[str, object]] = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        hashes = {
            (str(row["game_key_sha256"]), str(row["team_key_sha256"]))
            for row in values
        }
        if len(hashes) != 1:
            raise ValueError(f"comparison support mismatch for {key}")
        audits.append(
            {
                "season": key[0],
                "cutoff_index": key[1],
                "prior_family": key[2],
                "population_view": key[3],
                "candidate_count": len(values),
                "game_key_sha256": values[0]["game_key_sha256"],
                "team_key_sha256": values[0]["team_key_sha256"],
                "matched_fbs_teams": values[0]["matched_fbs_teams"],
            }
        )
    return {"groups": audits, "group_count": len(audits)}


def _future_key_hash(rows: Sequence[Mapping[str, object]]) -> str:
    value = "\n".join(sorted(str(row["game_id"]) for row in rows)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def validate_common_future_keys(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["season"]),
                int(row["cutoff_index"]),
                str(row["prior_family"]),
            )
        ].append(row)
    audits: list[dict[str, object]] = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        hashes = {
            (
                str(row["future_game_key_sha256"]),
                str(row["next_game_key_sha256"]),
            )
            for row in values
        }
        if len(hashes) != 1:
            raise ValueError(f"future comparison support mismatch for {key}")
        audits.append(
            {
                "season": key[0],
                "cutoff_index": key[1],
                "prior_family": key[2],
                "candidate_count": len(values),
                "future_game_key_sha256": values[0]["future_game_key_sha256"],
                "next_game_key_sha256": values[0]["next_game_key_sha256"],
            }
        )
    return {"groups": audits, "group_count": len(audits)}


def _target_pmfs(
    targets: Mapping[tuple[int, str, str], Mapping[str, object]], season: int
) -> dict[str, np.ndarray]:
    return {
        team_id: np.asarray(value["pmf"], dtype=float)
        for (target_season, subdivision, team_id), value in targets.items()
        if target_season == season and subdivision == "fbs"
    }


def posterior_metrics(
    pmfs: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> dict[str, float | int]:
    keys = sorted(set(pmfs) & set(targets))
    if not keys:
        raise ValueError("no common FBS posterior/target keys")
    nll: list[float] = []
    crps: list[float] = []
    expected_mae: list[float] = []
    median_mae: list[float] = []
    coverage: list[float] = []
    width: list[float] = []
    entropy: list[float] = []
    max_probability: list[float] = []
    brier: dict[int, list[float]] = {5: [], 10: [], 25: []}
    for key in keys:
        prediction = np.asarray(pmfs[key], dtype=float)
        target = np.asarray(targets[key], dtype=float)
        if len(prediction) != len(target):
            raise ValueError(f"rank support mismatch for common key {key}")
        ranks = np.arange(1, len(prediction) + 1)
        p_summary = pmf_summaries(prediction)
        q_summary = pmf_summaries(target)
        nll.append(float(-np.sum(target * np.log(np.maximum(prediction, 1e-15)))))
        crps.append(float(np.mean((np.cumsum(prediction) - np.cumsum(target)) ** 2)))
        expected_mae.append(abs(p_summary["expected_rank"] - q_summary["expected_rank"]))
        median_mae.append(abs(p_summary["median_rank"] - q_summary["median_rank"]))
        low, high = p_summary["interval_80_low"], p_summary["interval_80_high"]
        coverage.append(float(target[(ranks >= low) & (ranks <= high)].sum()))
        width.append(high - low)
        entropy.append(float(-np.sum(prediction * np.log(np.maximum(prediction, 1e-15)))))
        max_probability.append(float(np.max(prediction)))
        for threshold, values in brier.items():
            values.append(
                float(
                    (
                        p_summary[f"top{threshold}_probability"]
                        - target[:threshold].sum()
                    )
                    ** 2
                )
            )
    return {
        "matched_fbs_teams": len(keys),
        "nll": float(np.mean(nll)),
        "crps": float(np.mean(crps)),
        "expected_rank_mae": float(np.mean(expected_mae)),
        "median_rank_mae": float(np.mean(median_mae)),
        "interval_80_coverage": float(np.mean(coverage)),
        "interval_80_width": float(np.mean(width)),
        "mean_entropy": float(np.mean(entropy)),
        "mean_max_probability": float(np.mean(max_probability)),
        **{f"top{threshold}_brier": float(np.mean(values)) for threshold, values in brier.items()},
    }


def _student_t_density(
    value: float,
    locations: np.ndarray,
    scale: float,
    degrees_of_freedom: float,
) -> np.ndarray:
    return student_t.pdf((value - locations) / scale, degrees_of_freedom) / scale


def future_margin_score(
    game: Game,
    home: Team,
    away: Team,
    pmfs: Mapping[str, np.ndarray],
    likelihood: LikelihoodV1,
) -> dict[str, float]:
    actual_margin = oriented_margin(
        home.subdivision,
        away.subdivision,
        game.home_points,
        game.away_points,
    )
    cache_key = (
        likelihood.beta.tobytes(),
        likelihood.scale,
        likelihood.degrees_of_freedom,
        home.subdivision,
        away.subdivision,
        len(home.prior),
        len(away.prior),
        game.neutral_site,
        actual_margin,
    )
    cached = _FUTURE_SURFACE_CACHE.get(cache_key)
    if cached is None:
        locations, _ = game_margin_parameters(game, home, away, likelihood)
        density = _student_t_density(
            actual_margin,
            locations,
            likelihood.scale,
            likelihood.degrees_of_freedom,
        )
        win_density = student_t.sf(
            (0.0 - locations) / likelihood.scale,
            likelihood.degrees_of_freedom,
        )
        _FUTURE_SURFACE_CACHE[cache_key] = locations, density, win_density
    else:
        locations, density, win_density = cached
    weights = np.asarray(pmfs[home.team_id])[:, None] * np.asarray(pmfs[away.team_id])[None, :]
    predictive_density = float(np.sum(weights * density))
    expected_margin = float(np.sum(weights * locations))
    win_probability = float(np.sum(weights * win_density))
    actual_win = float(actual_margin > 0)
    return {
        "margin_mae": abs(expected_margin - actual_margin),
        "win_brier": (win_probability - actual_win) ** 2,
        "margin_nll": -math.log(max(predictive_density, 1e-300)),
        "n_games": 1,
    }


def aggregate_future_scores(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    if not rows:
        return {"n_games": 0, "margin_mae": None, "win_brier": None, "margin_nll": None}
    return {
        "n_games": len(rows),
        "margin_mae": float(np.mean([float(row["margin_mae"]) for row in rows])),
        "win_brier": float(np.mean([float(row["win_brier"]) for row in rows])),
        "margin_nll": float(np.mean([float(row["margin_nll"]) for row in rows])),
    }


def _pmf_expected_percentile(pmf: np.ndarray) -> float:
    ranks = np.arange(1, len(pmf) + 1, dtype=float)
    return float(np.dot(ranks - 0.5, pmf) / len(pmf))


def _oriented_ids(row: Mapping[str, object]) -> tuple[str, str]:
    if str(row["home_subdivision"]) == "fcs" and str(row["away_subdivision"]) == "fbs":
        return str(row["away_team_id"]), str(row["home_team_id"])
    return str(row["home_team_id"]), str(row["away_team_id"])


def _inference_pmfs(
    *,
    season: int,
    family: str,
    included_rows: Sequence[Mapping[str, object]],
    variant: str,
    models: Mapping[str, Mapping[str, object]],
    likelihood: LikelihoodV1,
    targets: Mapping[tuple[int, str, str], Mapping[str, object]],
    v1_result: PosteriorResult | None = None,
) -> tuple[PosteriorResult, list[Team]]:
    teams = make_teams(season, family, included_rows, targets)
    games = [row_to_game(row) for row in included_rows]
    ypp_by_game = {
        str(row["game_id"]): (row.get("home_ypp"), row.get("away_ypp"))
        for row in included_rows
    }
    if variant == "y1" and v1_result is not None:
        return v1_result, teams
    if variant == "v1":
        model = None
        allowed_pairings = SUPPORTED_YPP_PAIRINGS
    elif variant == "y1":
        model = models["supported_pairings"]["y1"]
        allowed_pairings = SUPPORTED_YPP_PAIRINGS
    elif variant == Y2_ORIGINAL_VARIANT:
        model = models["all_pairings_original"]["y2"]
        # This is intentionally isolated to the retained pre-correction
        # diagnostic so the old PR result can be quantified.  The corrected
        # candidate never opts into this policy.
        allowed_pairings = None
    elif variant == Y2_SUPPORTED_VARIANT:
        model = models["supported_pairings"]["y2"]
        allowed_pairings = SUPPORTED_YPP_PAIRINGS
    elif variant == "naive":
        model = models["supported_pairings"]["naive"]
        allowed_pairings = SUPPORTED_YPP_PAIRINGS
    else:
        raise ValueError(f"unknown posterior variant: {variant}")
    result = infer_posterior_with_ypp(
        teams,
        games,
        likelihood,
        ypp_by_game,
        model,
        allowed_pairings=allowed_pairings,
        max_iterations=100,
        tolerance=1e-6,
        damping=0.35,
    )
    if not result.converged:
        raise RuntimeError(
            f"posterior did not converge for {season}/{family}/{variant}: "
            f"delta={result.max_message_delta} iterations={result.iterations}"
        )
    return result, teams


def _future_games(
    rows: Sequence[Mapping[str, object]], cutoff: datetime
) -> list[dict[str, object]]:
    return [
        dict(row)
        for row in rows
        if _parse_date(row["start_date"]) > cutoff
    ]


def _next_future_game_ids(
    future_rows: Sequence[Mapping[str, object]],
) -> set[str]:
    """Return the union of each team's first strictly-future game IDs."""

    by_team: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in future_rows:
        by_team[str(row["home_team_id"])].append(row)
        by_team[str(row["away_team_id"])].append(row)
    selected: set[str] = set()
    for team_rows in by_team.values():
        first = min(
            team_rows,
            key=lambda row: (_parse_date(row["start_date"]), str(row["game_id"])),
        )
        selected.add(str(first["game_id"]))
    return selected


def run_posterior_evaluation(
    rows: Sequence[Mapping[str, object]],
    targets: Mapping[tuple[int, str, str], Mapping[str, object]],
    models: Mapping[str, Mapping[str, object]],
    likelihood: LikelihoodV1,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Run matched full/common posterior comparisons and future checks."""

    candidate_rows: list[dict[str, object]] = []
    season_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    future_rows: list[dict[str, object]] = []
    contamination_rows: list[dict[str, object]] = []
    team_impact_rows: list[dict[str, object]] = []
    for season in FINAL_YEARS:
        season_rows_source = [row for row in rows if int(row["season"]) == season]
        cutoffs = standard_cutoffs(rows, season)
        for cutoff_index, (cutoff_label, cutoff) in enumerate(cutoffs):
            for family in ("context", "history"):
                for view in ("full", SUPPORTED_YPP_OBSERVED_VIEW):
                    if view == SUPPORTED_YPP_OBSERVED_VIEW and cutoff_index != len(cutoffs) - 1:
                        continue
                    eligible = select_population_rows(
                        season_rows_source,
                        season=season,
                        cutoff=cutoff,
                        view=view,
                    )
                    game_hash = _game_population_hash(eligible)
                    key_team_ids = [
                        team_id
                        for (target_season, subdivision, team_id) in targets
                        if target_season == season and subdivision == "fbs"
                    ]
                    key_hash = _game_key_hash(key_team_ids)
                    v1_result: PosteriorResult | None = None
                    final_results: dict[str, PosteriorResult] = {}
                    final_teams: dict[str, list[Team]] = {}
                    final_metrics: dict[str, dict[str, float | int]] = {}
                    for variant in POSTERIOR_VARIANTS:
                        if variant == "y1" and v1_result is not None:
                            result = v1_result
                            teams = make_teams(season, family, eligible, targets)
                        else:
                            result, teams = _inference_pmfs(
                                season=season,
                                family=family,
                                included_rows=eligible,
                                variant=variant,
                                models=models,
                                likelihood=likelihood,
                                targets=targets,
                                v1_result=v1_result,
                            )
                        if variant == "v1":
                            v1_result = result
                        metrics = posterior_metrics(
                            result.pmfs,
                            _target_pmfs(targets, season),
                        )
                        row = {
                            "season": season,
                            "cutoff_label": cutoff_label,
                            "cutoff": cutoff.isoformat(),
                            "cutoff_index": cutoff_index,
                            "is_final_cutoff": cutoff_index == len(cutoffs) - 1,
                            "prior_family": family,
                            "population_view": view,
                            "candidate": variant,
                            "game_count": len(eligible),
                            "ypp_observed_game_count": sum(row.get("ypp_diff") is not None for row in eligible),
                            "game_key_sha256": game_hash,
                            "team_key_sha256": key_hash,
                            **metrics,
                            "posterior_converged": result.converged,
                            "posterior_iterations": result.iterations,
                            "posterior_max_message_delta": result.max_message_delta,
                        }
                        candidate_rows.append(row)
                        calibration_rows.append(
                            {
                                "season": season,
                                "cutoff_label": cutoff_label,
                                "prior_family": family,
                                "population_view": view,
                                "candidate": variant,
                                "n_teams": metrics["matched_fbs_teams"],
                                "nll": metrics["nll"],
                                "crps": metrics["crps"],
                                "interval_80_coverage": metrics["interval_80_coverage"],
                                "interval_80_width": metrics["interval_80_width"],
                                "mean_entropy": metrics["mean_entropy"],
                                "mean_max_probability": metrics["mean_max_probability"],
                            }
                        )
                        if cutoff_index == len(cutoffs) - 1:
                            season_rows.append(row.copy())
                            final_results[variant] = result
                            final_teams[variant] = teams
                            final_metrics[variant] = metrics

                        if view == "full":
                            future = _future_games(season_rows_source, cutoff)
                            next_ids = _next_future_game_ids(future)
                            team_by_id = {team.team_id: team for team in teams}
                            scores: list[dict[str, float]] = []
                            scored_future_rows: list[Mapping[str, object]] = []
                            next_scores: list[dict[str, float]] = []
                            scored_next_rows: list[Mapping[str, object]] = []
                            for future_row in future:
                                game = row_to_game(future_row)
                                if game.home_id not in result.pmfs or game.away_id not in result.pmfs:
                                    continue
                                if game.home_id not in team_by_id or game.away_id not in team_by_id:
                                    continue
                                score = future_margin_score(
                                    game,
                                    team_by_id[game.home_id],
                                    team_by_id[game.away_id],
                                    result.pmfs,
                                    likelihood,
                                )
                                scores.append(score)
                                scored_future_rows.append(future_row)
                                if str(future_row["game_id"]) in next_ids:
                                    next_scores.append(score)
                                    scored_next_rows.append(future_row)
                            aggregate = aggregate_future_scores(scores)
                            next_aggregate = aggregate_future_scores(next_scores)
                            future_rows.append(
                                {
                                    "season": season,
                                    "cutoff_label": cutoff_label,
                                    "cutoff_index": cutoff_index,
                                    "prior_family": family,
                                    "candidate": variant,
                                    "future_game_count": aggregate["n_games"],
                                    "future_margin_mae": aggregate["margin_mae"],
                                    "future_win_brier": aggregate["win_brier"],
                                    "future_margin_nll": aggregate["margin_nll"],
                                    "future_game_key_sha256": _future_key_hash(
                                        scored_future_rows
                                    ),
                                    "next_game_count": next_aggregate["n_games"],
                                    "next_margin_mae": next_aggregate["margin_mae"],
                                    "next_win_brier": next_aggregate["win_brier"],
                                    "next_margin_nll": next_aggregate["margin_nll"],
                                    "next_game_key_sha256": _future_key_hash(
                                        scored_next_rows
                                    ),
                                }
                            )
                    if view == "full" and cutoff_index == len(cutoffs) - 1:
                        # The independent diagnostic is deliberately limited to
                        # final full-population cutoffs; it is never included in
                        # the promotion candidate set.
                        result, teams = _inference_pmfs(
                            season=season,
                            family=family,
                            included_rows=eligible,
                            variant="naive",
                            models=models,
                            likelihood=likelihood,
                            targets=targets,
                        )
                        metrics = posterior_metrics(result.pmfs, _target_pmfs(targets, season))
                        diagnostic = {
                            "season": season,
                            "cutoff_label": cutoff_label,
                            "cutoff": cutoff.isoformat(),
                            "cutoff_index": cutoff_index,
                            "is_final_cutoff": True,
                            "prior_family": family,
                            "population_view": view,
                            "candidate": NAIVE_VARIANT,
                            "game_count": len(eligible),
                            "ypp_observed_game_count": sum(row.get("ypp_diff") is not None for row in eligible),
                            "game_key_sha256": game_hash,
                            "team_key_sha256": key_hash,
                            **metrics,
                            "posterior_converged": result.converged,
                            "posterior_iterations": result.iterations,
                            "posterior_max_message_delta": result.max_message_delta,
                        }
                        candidate_rows.append(diagnostic)
                        season_rows.append(diagnostic.copy())
                        final_results[NAIVE_VARIANT] = result
                        final_teams[NAIVE_VARIANT] = teams
                        final_metrics[NAIVE_VARIANT] = metrics
                        calibration_rows.append(
                            {
                                "season": season,
                                "cutoff_label": cutoff_label,
                                "prior_family": family,
                                "population_view": view,
                                "candidate": NAIVE_VARIANT,
                                "n_teams": metrics["matched_fbs_teams"],
                                "nll": metrics["nll"],
                                "crps": metrics["crps"],
                                "interval_80_coverage": metrics["interval_80_coverage"],
                                "interval_80_width": metrics["interval_80_width"],
                                "mean_entropy": metrics["mean_entropy"],
                                "mean_max_probability": metrics["mean_max_probability"],
                            }
                        )
                    if (
                        view == "full"
                        and Y2_ORIGINAL_VARIANT in final_metrics
                        and Y2_SUPPORTED_VARIANT in final_metrics
                    ):
                        original = final_metrics[Y2_ORIGINAL_VARIANT]
                        supported = final_metrics[Y2_SUPPORTED_VARIANT]
                        contamination_rows.append(
                            {
                                "season": season,
                                "prior_family": family,
                                "population_view": view,
                                "original_candidate": Y2_ORIGINAL_VARIANT,
                                "corrected_candidate": Y2_SUPPORTED_VARIANT,
                                "delta_nll_supported_minus_original": float(
                                    supported["nll"] - original["nll"]
                                ),
                                "delta_crps_supported_minus_original": float(
                                    supported["crps"] - original["crps"]
                                ),
                                "delta_expected_rank_mae_supported_minus_original": float(
                                    supported["expected_rank_mae"]
                                    - original["expected_rank_mae"]
                                ),
                                "delta_median_rank_mae_supported_minus_original": float(
                                    supported["median_rank_mae"]
                                    - original["median_rank_mae"]
                                ),
                                "delta_interval_80_coverage_supported_minus_original": float(
                                    supported["interval_80_coverage"]
                                    - original["interval_80_coverage"]
                                ),
                                "delta_interval_80_width_supported_minus_original": float(
                                    supported["interval_80_width"]
                                    - original["interval_80_width"]
                                ),
                                "delta_top5_brier_supported_minus_original": float(
                                    supported["top5_brier"] - original["top5_brier"]
                                ),
                                "delta_top10_brier_supported_minus_original": float(
                                    supported["top10_brier"] - original["top10_brier"]
                                ),
                                "delta_top25_brier_supported_minus_original": float(
                                    supported["top25_brier"] - original["top25_brier"]
                                ),
                                "delta_entropy_supported_minus_original": float(
                                    supported["mean_entropy"] - original["mean_entropy"]
                                ),
                            }
                        )
                        original_result = final_results[Y2_ORIGINAL_VARIANT]
                        supported_result = final_results[Y2_SUPPORTED_VARIANT]
                        original_team_lookup = {
                            team.team_id: team for team in final_teams[Y2_ORIGINAL_VARIANT]
                        }
                        supported_team_lookup = {
                            team.team_id: team for team in final_teams[Y2_SUPPORTED_VARIANT]
                        }
                        for team_id in sorted(
                            set(original_result.pmfs)
                            & set(supported_result.pmfs)
                            & set(key_team_ids)
                        ):
                            original_pmf = original_result.pmfs[team_id]
                            supported_pmf = supported_result.pmfs[team_id]
                            ranks = np.arange(1, len(original_pmf) + 1, dtype=float)
                            original_expected = float(np.dot(ranks, original_pmf))
                            supported_expected = float(np.dot(ranks, supported_pmf))
                            fcs_opponents: set[str] = set()
                            fbs_fcs_games = 0
                            for source in eligible:
                                if source["pairing"] != "fbs-fcs":
                                    continue
                                if str(source["home_team_id"]) == team_id:
                                    fcs_id = str(source["away_team_id"])
                                    fcs_name = str(source.get("away_team", fcs_id))
                                elif str(source["away_team_id"]) == team_id:
                                    fcs_id = str(source["home_team_id"])
                                    fcs_name = str(source.get("home_team", fcs_id))
                                else:
                                    continue
                                fcs_opponents.add(f"{fcs_name} ({fcs_id})")
                                fbs_fcs_games += 1
                            team = supported_team_lookup.get(
                                team_id, original_team_lookup[team_id]
                            )
                            team_impact_rows.append(
                                {
                                    "season": season,
                                    "prior_family": family,
                                    "population_view": view,
                                    "team_id": team_id,
                                    "team_name": team.name,
                                    "subdivision": team.subdivision,
                                    "original_expected_rank": original_expected,
                                    "supported_expected_rank": supported_expected,
                                    "delta_expected_rank_supported_minus_original": supported_expected
                                    - original_expected,
                                    "absolute_delta_expected_rank": abs(
                                        supported_expected - original_expected
                                    ),
                                    "original_expected_percentile": original_expected
                                    / len(original_pmf),
                                    "supported_expected_percentile": supported_expected
                                    / len(supported_pmf),
                                    "fbs_fcs_game_count": fbs_fcs_games,
                                    "fcs_opponents": "; ".join(sorted(fcs_opponents)),
                                }
                            )
    comparison_key_audit = validate_common_comparison_keys(candidate_rows)
    future_key_audit = validate_common_future_keys(future_rows)
    return candidate_rows, season_rows, calibration_rows, {
        "future_game_metrics": future_rows,
        "pairing_contamination": contamination_rows,
        "team_impact": team_impact_rows,
        "comparison_key_audit": comparison_key_audit,
        "future_key_audit": future_key_audit,
        "standard_cutoff_definition": "seven actual-date regular-season quantiles: 0, .20, .35, .55, .72, .87, 1.0",
        "posterior_engine": "production V1 BP semantics with research factor multiplication; unsupported FCS-FCS YPP uses an all-ones factor; 100 iterations, tolerance 1e-6, damping .35",
    }


def _aggregate_metric_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    candidate: str,
    view: str = "full",
    final_only: bool = True,
) -> dict[tuple[str, str], dict[str, float | int]]:
    selected = [
        row
        for row in rows
        if row["candidate"] == candidate
        and row["population_view"] == view
        and (not final_only or bool(row["is_final_cutoff"]))
    ]
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in selected:
        grouped[(str(row["prior_family"]), str(row["season"]))].append(row)
    result: dict[tuple[str, str], dict[str, float | int]] = {}
    for key, values in grouped.items():
        result[key] = {
            "nll": float(np.mean([float(row["nll"]) for row in values])),
            "crps": float(np.mean([float(row["crps"]) for row in values])),
            "expected_rank_mae": float(
                np.mean([float(row["expected_rank_mae"]) for row in values])
            ),
            "median_rank_mae": float(
                np.mean([float(row["median_rank_mae"]) for row in values])
            ),
            "interval_80_coverage": float(
                np.mean([float(row["interval_80_coverage"]) for row in values])
            ),
            "interval_80_width": float(
                np.mean([float(row["interval_80_width"]) for row in values])
            ),
            "mean_entropy": float(np.mean([float(row["mean_entropy"]) for row in values])),
            "mean_max_probability": float(
                np.mean([float(row["mean_max_probability"]) for row in values])
            ),
            "top5_brier": float(np.mean([float(row["top5_brier"]) for row in values])),
            "top10_brier": float(np.mean([float(row["top10_brier"]) for row in values])),
            "top25_brier": float(np.mean([float(row["top25_brier"]) for row in values])),
        }
    return result


def _metric_delta(
    candidate_rows: Sequence[Mapping[str, object]],
    left: str,
    right: str,
    *,
    view: str,
) -> list[dict[str, object]]:
    left_map = _aggregate_metric_rows(candidate_rows, candidate=left, view=view)
    right_map = _aggregate_metric_rows(candidate_rows, candidate=right, view=view)
    output: list[dict[str, object]] = []
    for key in sorted(set(left_map) & set(right_map)):
        output.append(
            {
                "prior_family": key[0],
                "season": int(key[1]),
                "candidate": left,
                "baseline": right,
                "population_view": view,
                "delta_nll": left_map[key]["nll"] - right_map[key]["nll"],
                "delta_crps": left_map[key]["crps"] - right_map[key]["crps"],
                "delta_expected_rank_mae": left_map[key]["expected_rank_mae"]
                - right_map[key]["expected_rank_mae"],
                "delta_median_rank_mae": left_map[key]["median_rank_mae"]
                - right_map[key]["median_rank_mae"],
                "delta_interval_80_coverage": left_map[key]["interval_80_coverage"]
                - right_map[key]["interval_80_coverage"],
                "delta_interval_80_width": left_map[key]["interval_80_width"]
                - right_map[key]["interval_80_width"],
                "delta_mean_entropy": left_map[key]["mean_entropy"]
                - right_map[key]["mean_entropy"],
                "delta_mean_max_probability": left_map[key]["mean_max_probability"]
                - right_map[key]["mean_max_probability"],
                "delta_top5_brier": left_map[key]["top5_brier"]
                - right_map[key]["top5_brier"],
                "delta_top10_brier": left_map[key]["top10_brier"]
                - right_map[key]["top10_brier"],
                "delta_top25_brier": left_map[key]["top25_brier"]
                - right_map[key]["top25_brier"],
            }
        )
    return output


def _choose_disagreement_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Select examples from years with frozen posterior priors available.

    The descriptive corpus reaches back to 2004, but the checked-in frozen
    context/history prior artifact currently contains 2022--2025 rows.  Keep
    the local V1-versus-Y2 effect comparison honest by selecting real games
    only from that supported evaluation era rather than inventing an early
    prior.
    """

    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    for source in rows:
        if int(source["season"]) not in FINAL_YEARS:
            continue
        if not is_supported_ypp_observed(source):
            continue
        row = dict(source)
        row["pairing"] = _row_pairing(source)
        row["ypp_diff"] = _usable_ypp_difference(source)
        margin = oriented_margin(
            str(row["home_subdivision"]),
            str(row["away_subdivision"]),
            int(row["home_points"]),
            int(row["away_points"]),
        )
        ypp = float(row["ypp_diff"])
        if margin >= 28 and ypp <= 0:
            row["disagreement_type"] = "large_win_with_nonpositive_ypp"
            row["selection_score"] = margin - 4 * ypp
            candidates["large_win_with_nonpositive_ypp"].append(row)
        if -14 <= margin < 0 and ypp >= 2:
            row["disagreement_type"] = "close_loss_with_strong_positive_ypp"
            row["selection_score"] = ypp - abs(margin) / 10
            candidates["close_loss_with_strong_positive_ypp"].append(row)
        if 0 < margin <= 14 and ypp <= -2:
            row["disagreement_type"] = "close_win_with_strong_negative_ypp"
            row["selection_score"] = -ypp - margin / 10
            candidates["close_win_with_strong_negative_ypp"].append(row)
    selected: list[dict[str, object]] = []
    for kind in (
        "large_win_with_nonpositive_ypp",
        "close_loss_with_strong_positive_ypp",
        "close_win_with_strong_negative_ypp",
    ):
        values = sorted(
            candidates[kind],
            key=lambda row: (
                -float(row["selection_score"]),
                int(row["season"]),
                str(row["game_id"]),
            ),
        )
        selected.extend(values[:3])
    return selected


def _local_disagreement_effect(
    row: Mapping[str, object],
    all_rows: Sequence[Mapping[str, object]],
    targets: Mapping[tuple[int, str, str], Mapping[str, object]],
    models: Mapping[str, Mapping[str, object]],
    likelihood: LikelihoodV1,
) -> dict[str, object]:
    season = int(row["season"])
    event_date = _parse_date(row["start_date"])
    before = [
        item
        for item in all_rows
        if int(item["season"]) == season
        and _parse_date(item["start_date"]) < event_date
    ]
    after = [*before, dict(row)]
    results: dict[str, tuple[PosteriorResult, list[Team]]] = {}
    after_teams: list[Team] = []
    for variant in ("v1", Y2_SUPPORTED_VARIANT):
        results[f"before_{variant}"], _ = _inference_pmfs(
            season=season,
            family="context",
            included_rows=before,
            variant=variant,
            models=models,
            likelihood=likelihood,
            targets=targets,
        )
        results[f"after_{variant}"], after_teams = _inference_pmfs(
            season=season,
            family="context",
            included_rows=after,
            variant=variant,
            models=models,
            likelihood=likelihood,
            targets=targets,
        )
    first_id, second_id = _oriented_ids(row)

    team_by_id = {team.team_id: team for team in after_teams}

    def expected_percentile(result: PosteriorResult, team_id: str) -> float:
        pmf = result.pmfs.get(team_id, team_by_id[team_id].prior)
        return _pmf_expected_percentile(pmf)

    v1_before_diff = expected_percentile(results["before_v1"], first_id) - expected_percentile(results["before_v1"], second_id)
    v1_after_diff = expected_percentile(results["after_v1"], first_id) - expected_percentile(results["after_v1"], second_id)
    y2_before_diff = expected_percentile(
        results[f"before_{Y2_SUPPORTED_VARIANT}"], first_id
    ) - expected_percentile(results[f"before_{Y2_SUPPORTED_VARIANT}"], second_id)
    y2_after_diff = expected_percentile(
        results[f"after_{Y2_SUPPORTED_VARIANT}"], first_id
    ) - expected_percentile(results[f"after_{Y2_SUPPORTED_VARIANT}"], second_id)
    game = row_to_game(row)
    pre_pmfs = dict(results["before_v1"].pmfs)
    for team_id in (game.home_id, game.away_id):
        pre_pmfs.setdefault(team_id, team_by_id[team_id].prior)
    v1_predictive = future_margin_score(
        game,
        team_by_id[game.home_id],
        team_by_id[game.away_id],
        pre_pmfs,
        likelihood,
    )
    return {
        "disagreement_type": row["disagreement_type"],
        "game_id": row["game_id"],
        "season": season,
        "start_date": row["start_date"],
        "home_team": row.get("home_team", row["home_team_id"]),
        "away_team": row.get("away_team", row["away_team_id"]),
        "home_team_id": row["home_team_id"],
        "away_team_id": row["away_team_id"],
        "site": row["site"],
        "pairing": row["pairing"],
        "home_score": row["home_points"],
        "away_score": row["away_points"],
        "oriented_margin": oriented_margin(
            str(row["home_subdivision"]),
            str(row["away_subdivision"]),
            int(row["home_points"]),
            int(row["away_points"]),
        ),
        "home_ypp": row.get("home_ypp"),
        "away_ypp": row.get("away_ypp"),
        "oriented_ypp_diff": row["ypp_diff"],
        "oriented_first_team": row["home_team"] if first_id == row["home_team_id"] else row["away_team"],
        "oriented_second_team": row["home_team"] if second_id == row["home_team_id"] else row["away_team"],
        "opponent_quality_before_context_percentile": expected_percentile(
            results["before_v1"], second_id
        ),
        "v1_margin_predictive_nll": v1_predictive["margin_nll"],
        "v1_oriented_quality_shift": v1_after_diff - v1_before_diff,
        "y2_oriented_quality_shift": y2_after_diff - y2_before_diff,
        "ypp_augmented_effect_on_oriented_quality": y2_after_diff - v1_after_diff,
        "v1_after_oriented_quality_difference": v1_after_diff,
        "y2_after_oriented_quality_difference": y2_after_diff,
        "pre_game_count": len(before),
    }


def build_disagreement_table(
    rows: Sequence[Mapping[str, object]],
    targets: Mapping[tuple[int, str, str], Mapping[str, object]],
    models: Mapping[str, Mapping[str, object]],
    likelihood: LikelihoodV1,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in _choose_disagreement_rows(rows):
        output.append(_local_disagreement_effect(row, rows, targets, models, likelihood))
    return output


def write_plots(
    coverage_rows: Sequence[Mapping[str, object]],
    signal_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    delta_rows: Sequence[Mapping[str, object]],
) -> None:
    plots = OUT / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 120, "font.size": 9})

    all_coverage = [row for row in coverage_rows if row["pairing"] == "all"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        [int(row["season"]) for row in all_coverage],
        [float(row["team_game_coverage_pct"]) for row in all_coverage],
        marker="o",
        color="#245b8a",
    )
    ax.set(
        title="Usable team-game YPP coverage",
        xlabel="Season",
        ylabel="Team-game rows with both inputs (%)",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "coverage_by_season.png")
    plt.close(fig)

    final_signal = [
        row
        for row in signal_rows
        if row.get("final_quality_advantage") is not None
        and row["period"] in {"development_2018_2021", "final_2022_2025"}
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for axis, period, title in zip(
        axes,
        ("development_2018_2021", "final_2022_2025"),
        ("Development 2018–2021", "Final test 2022–2025"),
    ):
        for pairing, color in (("fbs-fbs", "#245b8a"), ("fbs-fcs", "#b55d2a"), ("fcs-fcs", "#4f7d4f")):
            values = [row for row in final_signal if row["period"] == period and row["pairing"] == pairing]
            if not values:
                continue
            x = np.asarray([float(row["conditional_ypp_residual"]) for row in values])
            y = np.asarray([float(row["final_quality_advantage"]) for row in values])
            axis.scatter(x, y, s=5, alpha=0.18, label=pairing, color=color)
            if len(x) >= 3 and np.var(x) > 1e-12:
                slope = _slope(x, y)
                intercept = float(np.mean(y) - slope * np.mean(x)) if slope is not None else 0.0
                grid = np.linspace(float(np.min(x)), float(np.max(x)), 40)
                axis.plot(grid, intercept + (slope or 0.0) * grid, color=color, linewidth=1.2)
        axis.axhline(0, color="black", linewidth=0.5)
        axis.axvline(0, color="black", linewidth=0.5)
        axis.set(title=title, xlabel="Conditional YPP residual", ylim=None)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Final-quality advantage (oriented percentile)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "conditional_residual_quality.png")
    plt.close(fig)

    final_deltas = [
        row
        for row in delta_rows
        if row["candidate"] == Y2_SUPPORTED_VARIANT
        and row["population_view"] == "full"
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    for family, color in (("context", "#245b8a"), ("history", "#b55d2a")):
        values = [row for row in final_deltas if row["prior_family"] == family]
        values.sort(key=lambda row: int(row["season"]))
        if not values:
            continue
        ax.plot(
            [int(row["season"]) for row in values],
            [float(row["delta_nll"]) for row in values],
            marker="o",
            label=family,
            color=color,
        )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(
        title="Supported Y2 minus V1 final-rank NLL",
        xlabel="Season",
        ylabel="NLL delta (lower is better)",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "final_rank_nll_delta.png")
    plt.close(fig)


def _promotion_assessment(
    candidate_rows: Sequence[Mapping[str, object]],
    season_rows: Sequence[Mapping[str, object]],
    future_rows: Sequence[Mapping[str, object]],
    signal_summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    deltas = _metric_delta(
        candidate_rows, Y2_SUPPORTED_VARIANT, "v1", view="full"
    )
    context = [row for row in deltas if row["prior_family"] == "context"]
    history = [row for row in deltas if row["prior_family"] == "history"]
    all_deltas = [*context, *history]
    mean_nll_delta = float(np.mean([float(row["delta_nll"]) for row in all_deltas])) if all_deltas else None
    mean_crps_delta = float(np.mean([float(row["delta_crps"]) for row in all_deltas])) if all_deltas else None
    mean_coverage_delta = float(np.mean([float(row["delta_interval_80_coverage"]) for row in all_deltas])) if all_deltas else None
    seasons_with_improvement = {
        family: sum(float(row["delta_nll"]) < 0 for row in values)
        for family, values in (("context", context), ("history", history))
    }
    worst_nll = max((float(row["delta_nll"]) for row in all_deltas), default=None)
    worst_crps = max((float(row["delta_crps"]) for row in all_deltas), default=None)
    future_deltas: list[dict[str, object]] = []
    for family in ("context", "history"):
        for season in FINAL_YEARS:
            y2 = [
                row
                for row in future_rows
                if row["prior_family"] == family
                and int(row["season"]) == season
                and row["candidate"] == Y2_SUPPORTED_VARIANT
                and int(row["cutoff_index"]) == 0
            ]
            v1 = [
                row
                for row in future_rows
                if row["prior_family"] == family
                and int(row["season"]) == season
                and row["candidate"] == "v1"
                and int(row["cutoff_index"]) == 0
            ]
            if y2 and v1 and y2[0]["future_margin_mae"] is not None and v1[0]["future_margin_mae"] is not None:
                future_deltas.append(
                    {
                        "prior_family": family,
                        "season": season,
                        "delta_future_margin_mae": float(y2[0]["future_margin_mae"]) - float(v1[0]["future_margin_mae"]),
                        "delta_future_margin_nll": float(y2[0]["future_margin_nll"]) - float(v1[0]["future_margin_nll"]),
                    }
                )
    mean_future_mae = float(np.mean([row["delta_future_margin_mae"] for row in future_deltas])) if future_deltas else None
    mean_future_nll = float(np.mean([row["delta_future_margin_nll"] for row in future_deltas])) if future_deltas else None
    common_quality = [
        row
        for row in signal_summaries
        if row["period"] == "final_2022_2025"
        and row["pairing"] in SUPPORTED_YPP_PAIRINGS
        and row["quality_slope"] is not None
        and int(row["n_games"]) >= 20
    ]
    quality_seasons = sum(
        any(
            int(row["season"]) == season and float(row["quality_slope"]) > 0
            for row in common_quality
        )
        for season in FINAL_YEARS
    )
    checks = {
        "aggregate_nll": bool(
            mean_nll_delta is not None
            and mean_nll_delta <= -float(PROMOTION_CRITERIA["aggregate_final_rank_nll_improvement_nats_per_team"])
            and all(float(row["delta_nll"]) <= -float(PROMOTION_CRITERIA["aggregate_final_rank_nll_improvement_nats_per_team"]) for row in (context, history))
        ),
        "aggregate_crps": bool(
            mean_crps_delta is not None
            and mean_crps_delta <= float(PROMOTION_CRITERIA["maximum_allowed_aggregate_crps_degradation"])
        ),
        "calibration": bool(
            mean_coverage_delta is not None
            and mean_coverage_delta >= -float(PROMOTION_CRITERIA["maximum_allowed_aggregate_interval_coverage_drop"])
        ),
        "majority_seasons": all(
            value >= int(PROMOTION_CRITERIA["minimum_held_out_seasons_with_nll_improvement"])
            for value in seasons_with_improvement.values()
        ),
        "no_catastrophic_season": bool(
            worst_nll is not None
            and worst_nll <= float(PROMOTION_CRITERIA["maximum_allowed_single_season_nll_degradation"])
            and worst_crps is not None
            and worst_crps <= float(PROMOTION_CRITERIA["maximum_allowed_single_season_crps_degradation"])
        ),
        "future_direction": bool(
            mean_future_mae is not None
            and mean_future_mae <= float(PROMOTION_CRITERIA["maximum_allowed_future_margin_mae_degradation_points"])
            and mean_future_nll is not None
            and mean_future_nll <= float(PROMOTION_CRITERIA["maximum_allowed_future_margin_nll_degradation_nats"])
        ),
        "common_subset_signal": quality_seasons >= int(PROMOTION_CRITERIA["minimum_common_subset_seasons_with_positive_quality_slope"]),
    }
    if all(checks.values()):
        recommendation = "A"
    elif common_quality and any(float(row["quality_slope"]) > 0 for row in common_quality):
        recommendation = "B"
    else:
        recommendation = "C"
    return {
        "recommendation": recommendation,
        "checks": checks,
        "aggregate_delta_nll": mean_nll_delta,
        "aggregate_delta_crps": mean_crps_delta,
        "aggregate_delta_interval_80_coverage": mean_coverage_delta,
        "seasons_with_nll_improvement": seasons_with_improvement,
        "worst_single_season_delta_nll": worst_nll,
        "worst_single_season_delta_crps": worst_crps,
        "mean_future_delta_margin_mae": mean_future_mae,
        "mean_future_delta_margin_nll": mean_future_nll,
        "common_subset_positive_quality_slope_seasons": quality_seasons,
    }


def _metric_table_rows(
    rows: Sequence[Mapping[str, object]],
    candidates: Sequence[str],
    *,
    view: str = "full",
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for candidate in candidates:
        grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            if (
                row["candidate"] == candidate
                and row["population_view"] == view
                and bool(row["is_final_cutoff"])
            ):
                grouped[str(row["prior_family"])].append(row)
        for family in ("context", "history"):
            selected = grouped.get(family, [])
            if not selected:
                continue
            output.append(
                {
                    "prior_family": family,
                    "candidate": candidate,
                    "population_view": view,
                    "nll": float(np.mean([float(row["nll"]) for row in selected])),
                    "crps": float(np.mean([float(row["crps"]) for row in selected])),
                    "expected_rank_mae": float(
                        np.mean([float(row["expected_rank_mae"]) for row in selected])
                    ),
                    "median_rank_mae": float(
                        np.mean([float(row["median_rank_mae"]) for row in selected])
                    ),
                    "interval_80_coverage": float(
                        np.mean([float(row["interval_80_coverage"]) for row in selected])
                    ),
                    "interval_80_width": float(
                        np.mean([float(row["interval_80_width"]) for row in selected])
                    ),
                    "top5_brier": float(
                        np.mean([float(row["top5_brier"]) for row in selected])
                    ),
                    "top10_brier": float(
                        np.mean([float(row["top10_brier"]) for row in selected])
                    ),
                    "top25_brier": float(
                        np.mean([float(row["top25_brier"]) for row in selected])
                    ),
                    "mean_entropy": float(
                        np.mean([float(row["mean_entropy"]) for row in selected])
                    ),
                    "mean_max_probability": float(
                        np.mean([float(row["mean_max_probability"]) for row in selected])
                    ),
                    "n_rows": len(selected),
                }
            )
    return output


def _report_number(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(
    summary: Mapping[str, object],
    coverage_rows: Sequence[Mapping[str, object]],
    metric_table: Sequence[Mapping[str, object]],
    season_rows: Sequence[Mapping[str, object]],
    future_rows: Sequence[Mapping[str, object]],
    disagreement_rows: Sequence[Mapping[str, object]],
    signal_summaries: Sequence[Mapping[str, object]],
    contamination_rows: Sequence[Mapping[str, object]],
    team_impact_rows: Sequence[Mapping[str, object]],
) -> None:
    audit = summary["audit"]
    split = summary["data_split"]
    selection = summary["model_selection"]
    assessment = summary["promotion_assessment"]
    candidate_labels = {
        "v1": "V1",
        "y1": "Y1",
        Y2_ORIGINAL_VARIANT: "Y2-all-pairings-original",
        Y2_SUPPORTED_VARIANT: "Y2-supported",
        NAIVE_VARIANT: "naive-independence diagnostic",
    }
    lines = [
        "# YPP conditional-likelihood investigation",
        "",
        "## Question and conclusion",
        "",
        f"This research asks whether yards per play adds latent team-quality information beyond score margin, opponent quality, and site in Historical Likelihood V1. Applying the unchanged predeclared gate to the corrected supported-pairing candidate gives Recommendation **{assessment['recommendation']}**.",
        "",
        "The production V1 margin factor is frozen. The corrected candidate is `Y2-supported`: `p(margin | ranks, site) × p(YPP | margin, ranks, site)` only for historically supported FBS–FBS and FBS–FCS pairings. FCS–FCS uses the exact V1 margin factor because its YPP support is absent before the final test. The independent product remains a separate double-counting diagnostic and is not a promotion candidate.",
        "",
        "## Temporal split and prospective support policy",
        "",
        f"The temporal experiment is unchanged: training is {split['training']}; development is {split['development']}; the final test is {split['final_test']}. 2022–2025 remained untouched for candidate fitting, degrees-of-freedom selection, and promotion selection.",
        "",
        "| Pairing | Candidate fitting | Inference YPP factor |",
        "|:---|:---|:---|",
        "| FBS–FBS | enabled | conditional YPP when usable |",
        "| FBS–FCS | enabled | conditional YPP when usable |",
        "| FCS–FCS | unsupported; excluded from corrected fits | exactly 1; V1 margin only regardless of YPP |",
        "",
        "This is a prospective support boundary, not a post-hoc performance adjustment. `Y2-all-pairings-original` is retained only as the explicitly labelled invalid diagnostic that reproduces the first PR formulation. It is not used for promotion. For supported pairings, missing or unusable YPP also gives exactly the V1 margin factor.",
        "",
        "## Data audit",
        "",
        "The research uses the frozen historical modeling corpus for 2003–2025; 2003 has no usable team-game YPP, so the defensible common-data fit starts in 2004. The primary final-rank evaluation is for FBS teams using strict common FBS keys.",
        "",
        f"CFBD raw `/games/teams` responses contain no direct `yardsPerPlay`, `plays`, `offensivePlays`, or `totalPlays` category in the cached corpus. The derivation remains `plays = rushingAttempts + pass attempts parsed from completionAttempts`, then `totalYards / plays`. On {audit['processed_derivation']['raw_processed_common_rows']} raw/processed common team-game rows, derivation mismatches were {audit['processed_derivation']['raw_processed_derivation_mismatches']}.",
        "",
        f"The raw audit found {audit['stats']['raw_stat_exact_duplicate_rows']} exact duplicate team-game rows from overlapping classification/week queries and {audit['stats']['raw_stat_conflicts']} conflicting duplicates. The schedule audit found {audit['games']['raw_schedule_exact_overlaps']} exact overlaps and {audit['games']['raw_schedule_conflicts']} conflicts.",
        "",
        f"Usable YPP ranges from {_report_number(audit['coverage_summary']['extreme_ypp_values']['minimum'], 3)} to {_report_number(audit['coverage_summary']['extreme_ypp_values']['maximum'], 3)}; negative values={audit['coverage_summary']['extreme_ypp_values']['negative_count']}, zero values={audit['coverage_summary']['extreme_ypp_values']['zero_count']}, values below 1={audit['coverage_summary']['extreme_ypp_values']['below_one_count']}, values above 12={audit['coverage_summary']['extreme_ypp_values']['above_twelve_count']}. Coverage changes materially over time: {audit['coverage_summary']['coverage_materially_changes_over_time']}.",
        "",
        "Representative raw payloads contain the expected `rushingAttempts`, `completionAttempts`, `totalYards`, and `sacks` fields. No direct YPP/plays value was available to compare. In the stored 2024 Tennessee–Arkansas example, CFBD gives 36 rush attempts + 29 pass attempts = 65 plays for Tennessee and 44 + 30 = 74 for Arkansas; the official team notes report total offensive plays of 65 and 74 while listing sacks separately. This supports the current NCAA-style attempt semantics for this audit, without changing the corpus derivation. Because CFBD did not expose a source-wide official-play field in these payloads, this one official cross-check cannot establish a season-wide sacks mismatch rate; that remains a follow-up data-quality check if a direct play field becomes available.",
        "",
        "The current corpus builder has an operational sidecar hazard: a broad `game_stats/*.json` glob will see `.provenance.json` objects if a refreshed corpus is present. The audit excluded sidecars and did not rewrite or silently correct the stored corpus. This does not create a YPP-value mismatch in the common rows above, but it should be fixed in a separate acquisition-maintenance change.",
        "",
        "CFBD API schema reference: https://apinext.collegefootballdata.com/api/games. Official attempt/play cross-check: https://utsports.com/documents/download/2024/11/4/G9_UT_Notes_MSU.pdf.",
        "",
        "### Pairing-specific coverage by era",
        "",
        "The aggregate coverage series is misleading for this question because it is dominated by the much larger FCS–FCS schedule. The pairing-specific audit is the relevant support check:",
        "",
        "| Period | Pairing | Games | Both YPP | Game coverage | Team-row coverage |",
        "|:---|:---|---:|---:|---:|---:|",
    ]
    for row in audit["coverage_summary"]["period_pairing"]:
        if row["period"] in {
            "training_2004_2017",
            "development_2018_2021",
            "final_2022_2025",
        }:
            lines.append(
                f"| {row['period']} | {row['pairing']} | {row['games']} | {row['games_with_both_usable_ypp']} | {_report_number(row['game_coverage_pct'], 1)}% | {_report_number(row['team_game_coverage_pct'], 1)}% |"
            )
    lines.extend(
        [
            "",
            "FBS–FBS YPP is essentially complete throughout most of 2004–2021, and FBS–FCS is generally near-complete. FCS–FCS is essentially absent through 2021, then becomes approximately 98–99% covered in 2022–2025. The appropriate conclusion is that FCS–FCS YPP is unsupported by the training/development data for this experiment; the post-2022 FCS–FCS relationship is not treated as a validated negative or unstable YPP effect.",
            "",
        ]
    )
    lines.extend(
        [
            "",
            "## Conditional signal",
            "",
            f"Y1 uses only the fixed margin/site/pairing basis; Y2 adds the predeclared rank-percentile contrast basis. Student-t degrees of freedom were selected on 2018–2021 equal-game marginalized YPP NLL only, separately for the all-pairings diagnostic and the supported-pairing panel. Selected values: `{json.dumps(selection, sort_keys=True)}`.",
            "",
            "A. Direct residual signal. The residual is observed YPP differential minus the supported-pairing Y1 conditional mean. Positive residual means the V1-oriented side produced more YPP than its margin/site/pairing relationship predicted. This table includes only FBS–FBS and FBS–FCS rows; FCS–FCS rows are not mixed into the direct-signal conclusion. `conditional_signal.csv` reports margin and residual bins; `temporal_stability.csv` reports season/pairing effects.",
            "",
            "| Period | Season | Pairing | Site | N | Quality slope | Quality Spearman | Next-game slope |",
            "|:---|---:|:---|:---|---:|---:|---:|---:|",
        ]
    )
    for row in signal_summaries:
        if row["pairing"] in SUPPORTED_YPP_PAIRINGS:
            lines.append(
                f"| {row['period']} | {row['season']} | {row['pairing']} | {row['site']} | {row['n_games']} | {_report_number(row['quality_slope'])} | {_report_number(row['quality_spearman'])} | {_report_number(row['next_margin_slope'])} |"
            )
    lines.extend(
        [
            "",
            "The descriptive relationship is the direct residual evidence: inspect FBS–FBS and FBS–FCS slopes and correlations rather than treating the extra-variable YPP NLL as comparable with margin-only NLL. A positive quality slope means higher conditional YPP residual was associated with better eventual oriented rank percentile. This direct question is separate from the end-to-end posterior question below; a direct signal can exist while the connected posterior does not improve.",
            "",
            "### Candidate fitting and selection",
            "",
            "The corrected `Y2-supported` and the retained `Y2-all-pairings-original` diagnostic use the already-declared Y1/Y2 formulations, feature basis, Student-t grid, and development selection rule. Only the corrected panel is eligible for promotion. The final test is never used for candidate or df selection.",
            "",
            "| Panel | Candidate | Fit pairings | Selected df | Development marginalized NLL | Final fit seasons |",
            "|:---|:---|:---|---:|---:|:---|",
        ]
    )
    for panel, values in selection.items():
        for candidate in ("y1", "y2"):
            value = values[candidate]
            lines.append(
                f"| {panel} | {candidate} | {', '.join(value['fit_pairings'])} | {_report_number(value['student_t_df'], 1)} | {_report_number(value['development_marginalized_nll'])} | {value['final_fit_seasons']} |"
            )
    original_y2_df = selection["all_pairings_original"]["y2"]["student_t_df"]
    supported_y2_df = selection["supported_pairings"]["y2"]["student_t_df"]
    lines.append(
        f"Y2 selected df was {_report_number(original_y2_df, 1)} for the original panel and {_report_number(supported_y2_df, 1)} for the corrected panel; the selected formulation/df therefore did not change. The supported development NLL differs slightly because the corrected fit removes the unsupported FCS–FCS training observations."
    )
    lines.extend(
        [
            "",
            "## Candidate selection and evaluation",
            "",
            "The primary evaluation is FBS final-rank quality on the untouched 2022–2025 test period. Context and History priors, cutoffs, rank supports, final-rank targets, and strict comparison keys are identical across candidates. `candidate_metrics.csv` retains every matched cutoff row.",
            "",
            "| Prior | Candidate | View | NLL | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width | Top-5 Brier | Top-10 Brier | Top-25 Brier |",
            "|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metric_table:
        lines.append(
            f"| {row['prior_family']} | {candidate_labels.get(str(row['candidate']), row['candidate'])} | {row['population_view']} | {_report_number(row['nll'])} | {_report_number(row['crps'])} | {_report_number(row['expected_rank_mae'])} | {_report_number(row['median_rank_mae'])} | {_report_number(row['interval_80_coverage'])} | {_report_number(row['interval_80_width'], 1)} | {_report_number(row['top5_brier'])} | {_report_number(row['top10_brier'])} | {_report_number(row['top25_brier'])} |"
        )
    lines.extend(
        [
            "",
            "The primary end-to-end question is whether adding supported YPP to the connected ranking network improves FBS posterior quality. Full production-style keeps every eligible game: supported-pairing missing YPP gets exactly V1 margin evidence, and every FCS–FCS game gets V1 margin evidence regardless of YPP. The `supported_ypp_observed` view is the matched diagnostic subset of games with usable YPP and pairing exactly FBS–FBS or FBS–FCS; FCS–FCS is excluded even when YPP values exist. Both V1 and Y2 use that identical selected game population, while the full view remains primary.",
            "",
            "### Y2-supported minus V1 by final-test season",
            "",
            "| Prior | Season | Δ NLL | Δ CRPS | Δ expected-rank MAE | Δ median-rank MAE | Δ 80% coverage | Δ width | Δ Top-5 Brier | Δ Top-10 Brier | Δ Top-25 Brier |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    supported_deltas = _metric_delta(
        season_rows, Y2_SUPPORTED_VARIANT, "v1", view="full"
    )
    for row in supported_deltas:
        lines.append(
            f"| {row['prior_family']} | {row['season']} | {_report_number(row['delta_nll'])} | {_report_number(row['delta_crps'])} | {_report_number(row['delta_expected_rank_mae'])} | {_report_number(row['delta_median_rank_mae'])} | {_report_number(row['delta_interval_80_coverage'])} | {_report_number(row['delta_interval_80_width'], 1)} | {_report_number(row['delta_top5_brier'])} | {_report_number(row['delta_top10_brier'])} | {_report_number(row['delta_top25_brier'])} |"
        )
    aggregate_supported_deltas: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in supported_deltas:
        aggregate_supported_deltas[str(row["prior_family"])].append(row)
    lines.extend(
        [
            "",
            "### Aggregate Y2-supported minus V1",
            "",
            "| Prior | Δ NLL | Δ CRPS | Δ expected-rank MAE | Δ median-rank MAE | Δ 80% coverage | Δ width | Δ Top-5 Brier | Δ Top-10 Brier | Δ Top-25 Brier |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for family in ("context", "history"):
        values = aggregate_supported_deltas[family]
        if not values:
            continue
        mean = {
            field: float(np.mean([float(row[field]) for row in values]))
            for field in (
                "delta_nll",
                "delta_crps",
                "delta_expected_rank_mae",
                "delta_median_rank_mae",
                "delta_interval_80_coverage",
                "delta_interval_80_width",
                "delta_top5_brier",
                "delta_top10_brier",
                "delta_top25_brier",
            )
        }
        lines.append(
            f"| {family} | {_report_number(mean['delta_nll'])} | {_report_number(mean['delta_crps'])} | {_report_number(mean['delta_expected_rank_mae'])} | {_report_number(mean['delta_median_rank_mae'])} | {_report_number(mean['delta_interval_80_coverage'])} | {_report_number(mean['delta_interval_80_width'], 1)} | {_report_number(mean['delta_top5_brier'])} | {_report_number(mean['delta_top10_brier'])} | {_report_number(mean['delta_top25_brier'])} |"
        )
    lines.extend(
        [
            "",
            "Y1 is a semantic null: its factor is rank-invariant, and the posterior rows match V1 up to the deterministic alias used by the research evaluator. The naïve independent diagnostic is shown in `candidate_metrics.csv` only at final full cutoffs; any sharper posterior without commensurate rank scores is double-counting warning evidence.",
            "",
            "### Original all-pairings diagnostic versus corrected supported pairing",
            "",
            "The following differences are `Y2-supported − Y2-all-pairings-original` on the same full-population final cutoffs. They quantify how much the unsupported FCS–FCS factor in the first result changed downstream FBS posterior evaluation; they are diagnostic, not a new model search.",
            "",
            "| Prior | Season | Δ NLL | Δ CRPS | Δ expected-rank MAE | Δ median-rank MAE | Δ 80% coverage | Δ width | Δ Top-10 Brier |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in contamination_rows:
        lines.append(
            f"| {row['prior_family']} | {row['season']} | {_report_number(row['delta_nll_supported_minus_original'])} | {_report_number(row['delta_crps_supported_minus_original'])} | {_report_number(row['delta_expected_rank_mae_supported_minus_original'])} | {_report_number(row['delta_median_rank_mae_supported_minus_original'])} | {_report_number(row['delta_interval_80_coverage_supported_minus_original'])} | {_report_number(row['delta_interval_80_width_supported_minus_original'], 1)} | {_report_number(row['delta_top10_brier_supported_minus_original'])} |"
        )
    full_contamination = [
        row for row in contamination_rows if row["population_view"] == "full"
    ]
    if full_contamination:
        mean_contamination_nll = float(
            np.mean(
                [
                    float(row["delta_nll_supported_minus_original"])
                    for row in full_contamination
                ]
            )
        )
        mean_contamination_coverage = float(
            np.mean(
                [
                    float(row["delta_interval_80_coverage_supported_minus_original"])
                    for row in full_contamination
                ]
            )
        )
        lines.extend(
            [
                "",
                f"Across the full final-cutoff rows, the mean corrected-minus-original NLL change was {_report_number(mean_contamination_nll)} and the mean 80% coverage change was {_report_number(mean_contamination_coverage)}. The full row-level comparison is in `pairing_contamination.csv`.",
                "",
                "The largest FBS expected-rank shifts are listed below. `fbs_fcs_game_count` and `fcs_opponents` are descriptive network exposure fields, not causal attribution.",
                "",
                "| Season | Prior | Team | Δ expected rank | FBS–FCS games | FCS opponents |",
                "|---:|:---|:---|---:|---:|:---|",
            ]
        )
        top_team_rows = sorted(
            [
                row
                for row in team_impact_rows
                if row["population_view"] == "full"
                and row["subdivision"] == "fbs"
            ],
            key=lambda row: (-float(row["absolute_delta_expected_rank"]), str(row["team_id"])),
        )[:12]
        for row in top_team_rows:
            lines.append(
                f"| {row['season']} | {row['prior_family']} | {row['team_name']} ({row['team_id']}) | {_report_number(row['delta_expected_rank_supported_minus_original'])} | {row['fbs_fcs_game_count']} | {row['fcs_opponents'] or 'none'} |"
            )
        lines.extend(
            [
                "",
                "This is a propagation diagnostic through the connected schedule graph. It does not claim that a particular FBS–FCS opponent caused the change; it identifies where the unsupported FCS–FCS evidence reached the FBS posterior most strongly.",
            ]
        )
    lines.extend(
        [
            "",
            "### Future-game check",
            "",
            "Future games are scored with the same frozen V1 margin density from each cutoff posterior. The YPP factor is not used to score future margins; it only changes the state estimate. Future and next-game keys are identical across V1, Y1, Y2-all-pairings-original, and Y2-supported.",
            "",
        ]
    )
    future_summary: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in future_rows:
        future_summary[(str(row["prior_family"]), str(row["candidate"]))].append(row)
    lines.extend(
        [
            "| Prior | Candidate | Future games | Next-game MAE | Future Win Brier | Next-game Brier | Future margin NLL | Next-game NLL |",
            "|:---|:---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for (family, candidate), values in sorted(future_summary.items()):
        usable = [row for row in values if row["future_margin_mae"] is not None]
        if not usable:
            continue
        lines.append(
            f"| {family} | {candidate_labels.get(candidate, candidate)} | {sum(int(row['future_game_count']) for row in usable)} | {_report_number(np.mean([float(row['next_margin_mae']) for row in usable if row['next_margin_mae'] is not None]))} | {_report_number(np.mean([float(row['future_win_brier']) for row in usable]))} | {_report_number(np.mean([float(row['next_win_brier']) for row in usable if row['next_win_brier'] is not None]))} | {_report_number(np.mean([float(row['future_margin_nll']) for row in usable]))} | {_report_number(np.mean([float(row['next_margin_nll']) for row in usable if row['next_margin_nll'] is not None]))} |"
        )
    lines.extend(
        [
            "",
            "## Margin/YPP disagreement games",
            "",
            "The examples below are real corpus FBS–FBS or FBS–FCS games with usable YPP. `v1_margin_predictive_nll` is the pre-game V1 predictive evidence; quality shifts are oriented percentile shifts from a context-prior local update; negative YPP-augmented effect means the supported YPP factor moves the V1-oriented side toward a better latent rank relative to V1 alone.",
            "",
            "| Type | Season | Game | Teams | Score | Margin | YPP diff | V1 NLL | V1 shift | YPP effect |",
            "|:---|---:|---:|:---|:---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in disagreement_rows:
        lines.append(
            f"| {row['disagreement_type']} | {row['season']} | {row['game_id']} | {row['home_team']}–{row['away_team']} | {row['home_score']}–{row['away_score']} | {_report_number(row['oriented_margin'], 0)} | {_report_number(row['oriented_ypp_diff'])} | {_report_number(row['v1_margin_predictive_nll'])} | {_report_number(row['v1_oriented_quality_shift'])} | {_report_number(row['ypp_augmented_effect_on_oriented_quality'])} |"
        )
    lines.extend(
        [
            "",
            "## Temporal and subdivision stability",
            "",
            "The fixed pre-test model is evaluated by season and supported pairing; no adaptive era weighting, NIL break, or post-test refit is introduced. FCS–FCS is excluded from the direct residual-signal interpretation and receives V1-only evidence in the corrected posterior.",
            "",
            "Before reading 2022–2025, the promotion gate was fixed at: aggregate final-rank NLL improvement of at least 0.02 nats/team; CRPS degradation no greater than 0.002; 80% coverage drop no greater than 0.03; improvement in at least 3 of 4 held-out seasons for both prior families; no single-season NLL degradation above 0.10 or CRPS degradation above 0.02; future margin MAE degradation no greater than 0.50 points and NLL degradation no greater than 0.02; and positive common-subset quality slope in at least 3 seasons.",
            "",
            f"The promotion assessment for Y2-supported is `{json.dumps(assessment, sort_keys=True)}`. These are the unchanged predeclared thresholds; the original all-pairings diagnostic is not used for this decision.",
            "",
            "## Production feasibility (not productionized here)",
            "",
            f"The lightweight weekly updater currently fetches only two `/games` schedule responses for the season and deliberately does not fetch `/games/teams`. The historical builder has {audit['stats']['raw_stat_files']} cached team-stat response files across the corpus; the cached 2026 shape contains {audit['stats']['current_season_stat_files']} regular-season `/games/teams` response artifacts ({json.dumps(audit['stats']['current_season_stat_files_by_classification'], sort_keys=True)}). That is the current full-season request estimate if each classification/week response is fetched, with fewer requests for an in-season snapshot. Raw/provenance sidecars, a game-id/team-id join, and exact missing-YPP fallback semantics would be required. Current-season YPP acquisition and weekly snapshot runtime were not changed or claimed production-safe in this PR; a later PR should benchmark request latency, rate limits, raw artifact size, and local factor-construction cost before promotion.",
            "",
            "## Frozen production boundary and reproducibility",
            "",
            "No Historical Likelihood V1, Posterior V1, H 1.1, C 1.2, Performance V1, weekly publication, or website file is modified. The research script reads frozen artifacts and writes only this investigation directory. The input SHA-256 values and output inventory are in `summary.json`.",
            "",
            "The branch is research-only. Recommendation A would mean a future Likelihood V2 productionization PR is justified; B means the residual signal exists but formulation/coverage needs more research; C means YPP does not earn the added game-likelihood complexity.",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT,
        help="Research artifact directory (default: data/processed/ypp_investigation)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip PNG generation; all tabular and markdown artifacts still run.",
    )
    args = parser.parse_args()
    OUT = args.output
    required = [HISTORICAL_ROWS_PATH, RANK_DISTRIBUTIONS_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required frozen research inputs are missing: " + ", ".join(missing))

    production_paths = [
        ROOT / "data/processed/posterior/historical_likelihood_v1.json",
        ROOT / "data/processed/preseason/history/predictions.csv",
        ROOT / "data/processed/preseason/context/predictions.csv",
        ROOT / "data/processed/weekly_updates/2026-09-06.json",
        ROOT / "site/data/manifest.json",
        ROOT / "site/index.html",
    ]
    production_hashes_before = {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in production_paths
        if path.exists()
    }

    games, game_audit = load_raw_games()
    raw_stats, stat_audit = load_raw_stats()
    processed_stats = load_processed_ypp()
    coverage_rows, coverage_summary = build_coverage_audit(games, raw_stats)
    coverage_summary["period_pairing"] = summarise_pairing_coverage(coverage_rows)
    processed_derivation = audit_processed_derivation(raw_stats, processed_stats)
    historical = load_historical_rows()
    targets = load_rank_targets()
    enriched = build_enriched_rows(historical, games, targets)
    attach_next_game_performance(enriched)
    data_all_pairings = build_ypp_data(enriched, allowed_pairings=None)
    data_supported_pairings = build_ypp_data(
        enriched, allowed_pairings=SUPPORTED_YPP_PAIRINGS
    )
    all_models, all_selection_rows, all_selected_models = select_ypp_models(
        data_all_pairings,
        allowed_pairings=None,
        model_panel="all_pairings_original",
    )
    supported_models, supported_selection_rows, supported_selected_models = (
        select_ypp_models(
            data_supported_pairings,
            allowed_pairings=SUPPORTED_YPP_PAIRINGS,
            model_panel="supported_pairings",
        )
    )
    models = {
        "all_pairings_original": all_models,
        "supported_pairings": supported_models,
    }
    selection_rows = [*all_selection_rows, *supported_selection_rows]
    selected_models = {
        "all_pairings_original": all_selected_models,
        "supported_pairings": supported_selected_models,
    }
    supported_signal_rows = [
        row for row in enriched if row["pairing"] in SUPPORTED_YPP_PAIRINGS
    ]
    signal_rows = build_signal_rows(
        supported_signal_rows,
        supported_models["y1"],
        supported_models["y2"],
        targets,
    )
    signal_table, signal_summary = build_conditional_signal_table(signal_rows)
    likelihood = load_likelihood(ROOT / "data/processed/posterior/historical_likelihood_v1.json")
    candidate_rows, season_rows, calibration_rows, evaluation_metadata = run_posterior_evaluation(
        enriched, targets, models, likelihood
    )
    future_rows = evaluation_metadata.pop("future_game_metrics")
    contamination_rows = evaluation_metadata["pairing_contamination"]
    team_impact_rows = evaluation_metadata["team_impact"]
    disagreement_rows = build_disagreement_table(
        enriched, targets, models, likelihood
    )
    delta_rows = _metric_delta(
        candidate_rows, Y2_SUPPORTED_VARIANT, "v1", view="full"
    )
    promotion = _promotion_assessment(candidate_rows, season_rows, future_rows, signal_summary["period_pairing"])

    metric_table = [
        *_metric_table_rows(
            season_rows,
            ("v1", "y1", Y2_ORIGINAL_VARIANT, Y2_SUPPORTED_VARIANT, NAIVE_VARIANT),
            view="full",
        ),
        *_metric_table_rows(
            season_rows,
            ("v1", "y1", Y2_ORIGINAL_VARIANT, Y2_SUPPORTED_VARIANT),
            view=SUPPORTED_YPP_OBSERVED_VIEW,
        ),
    ]
    production_hashes_after = {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in production_paths
        if path.exists()
    }
    summary: dict[str, object] = {
        "research_question": "Does YPP add enough quality information beyond margin to deserve a place in the game likelihood?",
        "recommendation": promotion["recommendation"],
        "ypp_support_policy": {
            "name": YPP_PAIRING_POLICY,
            "supported_pairings": sorted(SUPPORTED_YPP_PAIRINGS),
            "unsupported_pairings": ["fcs-fcs"],
            "unsupported_inference_factor": "all ones; exact Historical Likelihood V1 margin fallback",
            "unsupported_fit_behavior": "excluded from corrected Y1/Y2/naive candidate fitting",
        },
        "data_split": {
            "training": "2004-2017 (2003 has no usable YPP)",
            "development": "2018-2021",
            "final_test": "2022-2025 untouched until candidate fitting, df selection, and promotion selection were complete",
            "final_rank_target": "frozen final constituent-rank PMFs from team_season_rank_distributions.csv; common FBS keys only",
        },
        "candidate_family": {
            "y0": "frozen Historical Likelihood V1 margin factor",
            "y1": "p(YPP_diff | margin basis, site, supported pairing), rank-independent semantic null",
            "y2_all_pairings_original": "retained diagnostic p(YPP_diff | margin basis, restrained rank-percentile basis, site, all pairings); invalid for promotion",
            "y2_supported": "corrected p(YPP_diff | margin basis, restrained rank-percentile basis, site, FBS-FBS/FBS-FCS only)",
            "naive_independent_diagnostic": "p(margin | ranks, site) × p(YPP_diff | ranks, site), never a promotion candidate",
            "margin_basis": "intercept, signed margin/20, abs(margin)/20, fixed hinges at -28,-14,0,14,28, plus V1 site indicators",
            "rank_basis": "same-subdivision odd percentile terms d, d*mean, d*abs(d); cross-subdivision FBS-minus-FCS d and d*mean",
            "fit": "weighted pseudo-observations with equal total weight per game and robust Student-t IRLS",
        },
        "population_views": {
            "full": "all eligible regular-season FBS-FBS, FBS-FCS, and FCS-FCS games through each cutoff; unsupported FCS-FCS YPP still uses the exact V1 margin fallback",
            SUPPORTED_YPP_OBSERVED_VIEW: "eligible games through the final cutoff with usable YPP and pairing exactly FBS-FBS or FBS-FCS; FCS-FCS is excluded even when YPP exists; V1 and Y2 use identical game keys",
        },
        "audit": {
            "games": game_audit,
            "stats": stat_audit,
            "processed_derivation": processed_derivation,
            "coverage_summary": coverage_summary,
        },
        "model_selection": selected_models,
        "fitting_data": {
            "all_pairings_original_pseudo_observations": len(data_all_pairings),
            "supported_pairings_pseudo_observations": len(data_supported_pairings),
            "all_pairings_original_games": len(set(data_all_pairings.game_id.tolist())),
            "supported_pairings_games": len(set(data_supported_pairings.game_id.tolist())),
            "training_seasons": list(TRAIN_YEARS),
            "development_seasons": list(DEVELOPMENT_YEARS),
            "final_test_seasons_excluded_from_selection": list(FINAL_YEARS),
        },
        "signal": signal_summary,
        "promotion_criteria": PROMOTION_CRITERIA,
        "promotion_assessment": promotion,
        "final_metric_table": metric_table,
        "final_metric_deltas_y2_supported_minus_v1": delta_rows,
        "evaluation": evaluation_metadata,
        "input_sha256": {
            "historical_modeling_games.csv": _sha256(HISTORICAL_ROWS_PATH),
            "team_season_rank_distributions.csv": _sha256(RANK_DISTRIBUTIONS_PATH),
            "historical_likelihood_v1.json": _sha256(ROOT / "data/processed/posterior/historical_likelihood_v1.json"),
        },
        "frozen_production_integrity": {
            "hashes_before": production_hashes_before,
            "hashes_after": production_hashes_after,
            "unchanged": production_hashes_before == production_hashes_after,
            "scope": "Historical Likelihood V1, Posterior V1, H 1.1, C 1.2, Performance V1, weekly publication, website",
        },
        "artifact_inventory": {
            "report.md": "human-readable findings",
            "summary.json": "machine-readable configuration, audit, metrics, and recommendation",
            "coverage.csv": "season/pairing YPP coverage and missingness",
            "conditional_signal.csv": "margin/residual bins",
            "candidate_selection.csv": "predeclared Student-t df development selection",
            "candidate_metrics.csv": "matched cutoff posterior metrics for full and supported_ypp_observed populations",
            "season_metrics.csv": "final cutoff metrics by season",
            "future_game_metrics.csv": "frozen V1 future-margin scoring from each posterior",
            "disagreement_games.csv": "real margin/YPP disagreement cases and local effects",
            "calibration.csv": "interval width/coverage and concentration diagnostics",
            "temporal_stability.csv": "season/pairing conditional signal effects",
            "pairing_contamination.csv": "same-cutoff Y2-supported minus original all-pairings diagnostic differences",
            "team_impact.csv": "FBS posterior expected-rank shifts from removing unsupported FCS-FCS YPP",
            "plots/": "deterministic coverage, signal, and final-rank delta PNGs",
        },
    }
    _write_json(OUT / "summary.json", summary)
    _write_csv(OUT / "coverage.csv", coverage_rows)
    _write_csv(OUT / "conditional_signal.csv", signal_table)
    _write_csv(OUT / "candidate_selection.csv", selection_rows)
    _write_csv(OUT / "candidate_metrics.csv", candidate_rows)
    _write_csv(OUT / "season_metrics.csv", season_rows)
    _write_csv(OUT / "future_game_metrics.csv", future_rows)
    _write_csv(OUT / "disagreement_games.csv", disagreement_rows)
    _write_csv(OUT / "calibration.csv", calibration_rows)
    _write_csv(OUT / "temporal_stability.csv", signal_summary["period_pairing"])
    _write_csv(OUT / "metric_deltas.csv", delta_rows)
    _write_csv(OUT / "pairing_contamination.csv", contamination_rows)
    _write_csv(OUT / "team_impact.csv", team_impact_rows)
    if not args.skip_plots:
        write_plots(coverage_rows, signal_rows, candidate_rows, delta_rows)
    write_report(
        summary,
        coverage_rows,
        metric_table,
        season_rows,
        future_rows,
        disagreement_rows,
        signal_summary["period_pairing"],
        contamination_rows,
        team_impact_rows,
    )
    print(json.dumps({"output": str(OUT), "recommendation": promotion["recommendation"], "candidate_rows": len(candidate_rows), "future_rows": len(future_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
