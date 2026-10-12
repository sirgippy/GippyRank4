"""Compare original and corrected Massey rank populations without fitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from gippyrank.data.cfbd_games import (
    apply_historical_game_subdivisions,
    read_cfbd_game_rows,
)
from gippyrank.data.massey import NORMALIZED_FIELDS
from gippyrank.modeling import (
    CONSTITUENT_RANK_PAIR_FIELDS,
    audit_population_coverage,
    build_constituent_rank_pairs,
    build_distributions,
    coverage_table,
)

ROOT = Path(__file__).resolve().parents[1]
CORPUS_VERSION = "massey-rank-population-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_observations(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != NORMALIZED_FIELDS:
            raise ValueError(f"Unexpected normalized Massey schema in {path}")
        return list(reader)


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def legacy_team_ranks(
    observations: list[dict[str, str]], depth: list[dict[str, object]], threshold: float
) -> tuple[dict[tuple[int, str, str], list[tuple[str, int]]], dict[tuple[int, str], int]]:
    """Reconstruct original rank pairs without applying new fail-fast checks."""
    selected = {
        (int(row["season"]), str(row["subdivision"]), str(row["system_code"]))
        for row in depth
        if float(row["coverage_ratio"]) >= threshold
    }
    populations = {
        (int(row["season"]), str(row["subdivision"])): int(row["team_population"])
        for row in depth
    }
    by_team: dict[tuple[int, str, str], list[tuple[str, int]]] = defaultdict(list)
    for row in observations:
        key = (int(row["season"]), row["subdivision"], row["system_code"])
        if row["is_composite"] != "True" and key in selected:
            by_team[(key[0], key[1], row["team_id"])].append(
                (key[2], int(row["ordinal_rank"]))
            )
    return by_team, populations


def legacy_pair_rows(
    games: list[dict[str, str]],
    by_team: dict[tuple[int, str, str], list[tuple[str, int]]],
    populations: dict[tuple[int, str], int],
) -> list[dict[str, object]]:
    rows = []
    for game in games:
        home_sub = game["homeClassification"]
        away_sub = game["awayClassification"]
        if home_sub not in {"fbs", "fcs"} or away_sub not in {"fbs", "fcs"}:
            continue
        season = int(game["season"])
        home_key = (season, home_sub, game["homeId"])
        away_key = (season, away_sub, game["awayId"])
        home_ranks = dict(by_team.get(home_key, []))
        away_ranks = dict(by_team.get(away_key, []))
        for system in sorted(set(home_ranks) & set(away_ranks)):
            home_rank, away_rank = home_ranks[system], away_ranks[system]
            rows.append(
                {
                    "game_id": game["id"],
                    "season": season,
                    "system_code": system,
                    "home_team_id": home_key[2],
                    "home_subdivision": home_sub,
                    "home_rank": home_rank,
                    "home_team_population": populations.get((season, home_sub), 0),
                    "away_team_id": away_key[2],
                    "away_subdivision": away_sub,
                    "away_rank": away_rank,
                    "away_team_population": populations.get((season, away_sub), 0),
                }
            )
    return rows


def pmf_support_metrics(rows: list[dict[str, str]]) -> dict[tuple[int, str], dict[str, float]]:
    metrics: dict[tuple[int, str], dict[str, float]] = defaultdict(
        lambda: {"teams": 0, "unsupported_teams": 0, "unsupported_observations": 0, "unsupported_mass": 0.0}
    )
    for row in rows:
        key = (int(row["season"]), row["subdivision"])
        item = metrics[key]
        item["teams"] += 1
        population = int(row["team_population"])
        unsupported_team = False
        for point in json.loads(row["pmf"]):
            rank = int(point["rank"])
            if rank > population:
                unsupported_team = True
                item["unsupported_observations"] += int(point["count"])
                item["unsupported_mass"] += float(point["probability"])
        if unsupported_team:
            item["unsupported_teams"] += 1
    return metrics


def corpus_accounting(
    label: str,
    before: list[dict[str, str]],
    after: list[dict[str, str]],
    before_targets: list[dict[str, str]],
    after_targets: list[dict[str, str]],
    threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    before_depth, after_depth = coverage_table(before), coverage_table(after)
    before_coverage = audit_population_coverage(
        before, before_depth, before_targets, threshold
    )
    after_coverage = audit_population_coverage(after, after_depth, after_targets, threshold)
    coverage_by_key = {
        (int(row["season"]), str(row["subdivision"])): row
        for row in before_coverage
    }
    after_coverage_by_key = {
        (int(row["season"]), str(row["subdivision"])): row
        for row in after_coverage
    }
    before_support = pmf_support_metrics(before_targets)
    after_support = pmf_support_metrics(after_targets)
    before_target = {
        (int(row["season"]), row["subdivision"], row["team_id"]): row
        for row in before_targets
    }
    after_target = {
        (int(row["season"]), row["subdivision"], row["team_id"]): row
        for row in after_targets
    }
    before_by_cohort: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    after_by_cohort: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    before_cmp_by_cohort: dict[tuple[int, str], set[str]] = defaultdict(set)
    after_cmp_by_cohort: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in before:
        key = (int(row["season"]), row["subdivision"])
        before_by_cohort[key].append(row)
        if row["is_composite"] == "True":
            before_cmp_by_cohort[key].add(row["team_id"])
    for row in after:
        key = (int(row["season"]), row["subdivision"])
        after_by_cohort[key].append(row)
        if row["is_composite"] == "True":
            after_cmp_by_cohort[key].add(row["team_id"])
    moved: list[dict[str, object]] = []
    cohort_keys = sorted(
        {
            (int(row["season"]), row["subdivision"])
            for row in before
            if row["is_composite"] == "True"
        }
        | {
            (int(row["season"]), row["subdivision"])
            for row in after
            if row["is_composite"] == "True"
        }
    )
    summary = []
    team_deltas = []
    for season, subdivision in cohort_keys:
        cohort_key = (season, subdivision)
        old_cmp = before_cmp_by_cohort[cohort_key]
        new_cmp = after_cmp_by_cohort[cohort_key]
        old_rows = before_by_cohort[cohort_key]
        new_rows = after_by_cohort[cohort_key]
        old_selected = {
            (int(row["season"]), row["subdivision"], row["system_code"])
            for row in before_depth
            if int(row["season"]) == season
            and row["subdivision"] == subdivision
            and float(row["coverage_ratio"]) >= threshold
        }
        new_selected = {
            (int(row["season"]), row["subdivision"], row["system_code"])
            for row in after_depth
            if int(row["season"]) == season
            and row["subdivision"] == subdivision
            and float(row["coverage_ratio"]) >= threshold
        }
        old_invalid = [
            row
            for row in old_rows
            if row["is_composite"] != "True"
            and int(row["ordinal_rank"]) > len(old_cmp)
        ]
        new_invalid = [
            row
            for row in new_rows
            if row["is_composite"] != "True"
            and int(row["ordinal_rank"]) > len(new_cmp)
        ]
        old_selected_invalid = [
            row
            for row in old_invalid
            if (season, subdivision, row["system_code"]) in old_selected
        ]
        new_selected_invalid = [
            row
            for row in new_invalid
            if (season, subdivision, row["system_code"]) in new_selected
        ]
        old_cov = coverage_by_key.get((season, subdivision), {})
        new_cov = after_coverage_by_key.get((season, subdivision), {})
        old_pop = len(old_cmp)
        new_pop = len(new_cmp)
        old_targets = {
            key: row
            for key, row in before_target.items()
            if key[:2] == (season, subdivision)
        }
        new_targets = {
            key: row
            for key, row in after_target.items()
            if key[:2] == (season, subdivision)
        }
        common = set(old_targets) & set(new_targets)
        rank_mean_changed = 0
        for key in sorted(set(old_targets) | set(new_targets)):
            old = old_targets.get(key)
            new = new_targets.get(key)
            if old is None or new is None:
                changed = {
                    "corpus": label,
                    "season": season,
                    "subdivision": subdivision,
                    "team_id": key[2],
                    "team_name": (old or new or {}).get("team_name", ""),
                    "change": "added" if old is None else "removed",
                    "population_before": old.get("team_population", "") if old else "",
                    "population_after": new.get("team_population", "") if new else "",
                    "rank_percentile_mean_before": old.get("rank_percentile_mean", "") if old else "",
                    "rank_percentile_mean_after": new.get("rank_percentile_mean", "") if new else "",
                    "rank_observations_equal": False,
                    "pmf_equal": False,
                }
                team_deltas.append(changed)
            else:
                old_mean = float(old["rank_percentile_mean"])
                new_mean = float(new["rank_percentile_mean"])
                rank_mean_changed += not abs(old_mean - new_mean) <= 1e-12
                rank_observations_equal = old["rank_observations"] == new["rank_observations"]
                pmf_equal = old["pmf"] == new["pmf"]
                if (
                    not rank_observations_equal
                    or not pmf_equal
                    or old["team_population"] != new["team_population"]
                ):
                    team_deltas.append(
                        {
                            "corpus": label,
                            "season": season,
                            "subdivision": subdivision,
                            "team_id": key[2],
                            "team_name": new["team_name"],
                            "change": "population_or_target_changed",
                            "population_before": old["team_population"],
                            "population_after": new["team_population"],
                            "rank_percentile_mean_before": old["rank_percentile_mean"],
                            "rank_percentile_mean_after": new["rank_percentile_mean"],
                            "rank_observations_equal": rank_observations_equal,
                            "pmf_equal": pmf_equal,
                        }
                    )
        old_support = before_support.get((season, subdivision), {})
        new_support = after_support.get((season, subdivision), {})
        summary.append(
            {
                "corpus": label,
                "season": season,
                "subdivision": subdivision,
                "observations_before": len(old_rows),
                "observations_after": len(new_rows),
                "cmp_population_before": old_pop,
                "cmp_population_after": new_pop,
                "cmp_members_added": len(new_cmp - old_cmp),
                "cmp_members_removed": len(old_cmp - new_cmp),
                "constituent_systems_before": len({r["system_code"] for r in old_rows if r["is_composite"] != "True"}),
                "constituent_systems_after": len({r["system_code"] for r in new_rows if r["is_composite"] != "True"}),
                "selected_systems_before": len(old_selected),
                "selected_systems_after": len(new_selected),
                "max_source_rank_before": max((int(r["ordinal_rank"]) for r in old_rows if r["is_composite"] != "True"), default=0),
                "max_source_rank_after": max((int(r["ordinal_rank"]) for r in new_rows if r["is_composite"] != "True"), default=0),
                "all_system_out_of_support_rows_before": len(old_invalid),
                "all_system_out_of_support_rows_after": len(new_invalid),
                "selected_out_of_support_rows_before": len(old_selected_invalid),
                "selected_out_of_support_rows_after": len(new_selected_invalid),
                "target_team_seasons_before": int(old_support.get("teams", 0)),
                "target_team_seasons_after": int(new_support.get("teams", 0)),
                "target_unsupported_team_seasons_before": int(old_support.get("unsupported_teams", 0)),
                "target_unsupported_team_seasons_after": int(new_support.get("unsupported_teams", 0)),
                "target_unsupported_observations_before": int(old_support.get("unsupported_observations", 0)),
                "target_unsupported_observations_after": int(new_support.get("unsupported_observations", 0)),
                "target_unsupported_probability_mass_before": float(old_support.get("unsupported_mass", 0.0)),
                "target_unsupported_probability_mass_after": float(new_support.get("unsupported_mass", 0.0)),
                "teams_missing_selected_constituents_before": old_cov.get("teams_missing_selected_constituents", old_pop),
                "teams_missing_selected_constituents_after": new_cov.get("teams_missing_selected_constituents", new_pop),
                "common_target_team_seasons": len(common),
                "common_target_rank_percentile_mean_changed": rank_mean_changed,
                "target_teams_added": len(set(new_targets) - set(old_targets)),
                "target_teams_removed": len(set(old_targets) - set(new_targets)),
            }
        )
        for team_id in sorted(old_cmp | new_cmp):
            if (team_id in old_cmp) == (team_id in new_cmp):
                continue
            name = next(
                (
                    row["team_name"]
                    for row in new_rows + old_rows
                    if row["team_id"] == team_id
                ),
                "",
            )
            moved.append(
                {
                    "corpus": label,
                    "season": season,
                    "subdivision": subdivision,
                    "team_id": team_id,
                    "team_name": name,
                    "change": "added_to_model_population" if team_id in new_cmp else "removed_from_model_population",
                }
            )
    return summary, team_deltas, moved


def pair_accounting(
    label: str,
    before_games: list[dict[str, str]],
    after_games: list[dict[str, str]],
    before_observations: list[dict[str, str]],
    after_observations: list[dict[str, str]],
    threshold: float,
    corrected_pair_path: Path,
) -> dict[str, object]:
    old_depth = coverage_table(before_observations)
    new_depth = coverage_table(after_observations)
    old_by_team, old_populations = legacy_team_ranks(
        before_observations, old_depth, threshold
    )
    old_pairs = legacy_pair_rows(before_games, old_by_team, old_populations)
    _, new_by_team = build_distributions(after_observations, new_depth, threshold)
    new_pairs = build_constituent_rank_pairs(after_games, new_by_team, new_depth)
    with corrected_pair_path.open(newline="", encoding="utf-8-sig") as handle:
        pair_reader = csv.DictReader(handle)
        if tuple(pair_reader.fieldnames or ()) != CONSTITUENT_RANK_PAIR_FIELDS:
            raise ValueError(f"Unexpected corrected rank-pair schema in {corrected_pair_path}")
        corrected_pair_count = 0
        for expected, actual in zip(new_pairs, pair_reader, strict=False):
            if any(str(expected[field]) != actual[field] for field in CONSTITUENT_RANK_PAIR_FIELDS):
                raise ValueError(
                    f"Corrected pair artifact content mismatch in {corrected_pair_path} "
                    f"at row {corrected_pair_count + 2}"
                )
            corrected_pair_count += 1
    if corrected_pair_count != len(new_pairs):
        raise ValueError(
            f"Corrected pair artifact row count mismatch: {corrected_pair_path} "
            f"has {corrected_pair_count}, rebuilt {len(new_pairs)}"
        )
    old_index = {(r["game_id"], r["system_code"]): r for r in old_pairs}
    new_index = {(r["game_id"], r["system_code"]): r for r in new_pairs}
    common = set(old_index) & set(new_index)
    changed = [
        key
        for key in common
        if any(
            str(old_index[key][field]) != str(new_index[key][field])
            for field in ("home_rank", "away_rank", "home_team_population", "away_team_population")
        )
    ]
    old_unsupported = sum(
        int(row["home_rank"]) > int(row["home_team_population"])
        or int(row["away_rank"]) > int(row["away_team_population"])
        for row in old_pairs
    )
    return {
        "corpus": label,
        "before_pair_rows": len(old_pairs),
        "after_pair_rows": corrected_pair_count,
        "pairs_added": len(set(new_index) - set(old_index)),
        "pairs_removed": len(set(old_index) - set(new_index)),
        "common_pairs_changed": len(changed),
        "before_pairs_out_of_population": old_unsupported,
        "after_pairs_out_of_population": 0,
        "corrected_pair_file": corrected_pair_path.as_posix(),
        "corrected_pair_sha256": sha256(corrected_pair_path),
        "corrected_pair_rows_match_rebuild": True,
    }


def source_rank_dispositions(
    label: str,
    version: str,
    observations: list[dict[str, str]],
    threshold: float,
) -> list[dict[str, object]]:
    depth = coverage_table(observations)
    selected = {
        (int(row["season"]), str(row["subdivision"]), str(row["system_code"]))
        for row in depth
        if float(row["coverage_ratio"]) >= threshold
    }
    populations = {
        (int(row["season"]), str(row["subdivision"])): int(row["team_population"])
        for row in depth
    }
    result = []
    coverage_by_key = {
        (int(item["season"]), str(item["subdivision"]), str(item["system_code"])): float(
            item["coverage_ratio"]
        )
        for item in depth
    }
    for row in observations:
        if row["is_composite"] == "True":
            continue
        key = (int(row["season"]), row["subdivision"], row["system_code"])
        population = populations.get(key[:2], 0)
        rank = int(row["ordinal_rank"])
        if population and rank > population:
            is_selected = key in selected
            result.append(
                {
                    "corpus": label,
                    "version": version,
                    "season": key[0],
                    "subdivision": key[1],
                    "team_id": row["team_id"],
                    "team_name": row["team_name"],
                    "system_code": key[2],
                    "source_rank": rank,
                    "cmp_population": population,
                    "system_coverage_ratio": coverage_by_key[key],
                    "selected_for_target": is_selected,
                    "disposition": (
                        "selected_support_error"
                        if is_selected
                        else "preserved_but_excluded_by_coverage_threshold"
                    ),
                    "source_file": row["source_file"],
                }
            )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-final", type=Path, required=True)
    parser.add_argument("--original-final-targets", type=Path, required=True)
    parser.add_argument("--original-pre-bowl", type=Path, required=True)
    parser.add_argument("--original-pre-bowl-targets", type=Path, required=True)
    parser.add_argument("--corrected-final", type=Path, required=True)
    parser.add_argument("--corrected-final-targets", type=Path, required=True)
    parser.add_argument("--corrected-final-pairs", type=Path, required=True)
    parser.add_argument("--corrected-pre-bowl", type=Path, required=True)
    parser.add_argument("--corrected-pre-bowl-targets", type=Path, required=True)
    parser.add_argument("--corrected-pre-bowl-pairs", type=Path, required=True)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--coverage-threshold", type=float, default=0.90)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    before_games, game_sources = read_cfbd_game_rows(
        args.games, apply_cohort_decisions=False
    )
    after_games = apply_historical_game_subdivisions(before_games)
    corpora = [
        (
            "final",
            args.original_final,
            args.original_final_targets,
            args.corrected_final,
            args.corrected_final_targets,
            args.corrected_final_pairs,
        ),
        (
            "pre_bowl",
            args.original_pre_bowl,
            args.original_pre_bowl_targets,
            args.corrected_pre_bowl,
            args.corrected_pre_bowl_targets,
            args.corrected_pre_bowl_pairs,
        ),
    ]
    accounting, target_deltas, membership_deltas, pair_summaries = [], [], [], []
    rank_dispositions = []
    for (
        label,
        old_obs_path,
        old_pmf_path,
        new_obs_path,
        new_pmf_path,
        new_pair_path,
    ) in corpora:
        old_obs, new_obs = read_observations(old_obs_path), read_observations(new_obs_path)
        old_pmfs, new_pmfs = read_table(old_pmf_path), read_table(new_pmf_path)
        summary, target_changes, membership_changes = corpus_accounting(
            label,
            old_obs,
            new_obs,
            old_pmfs,
            new_pmfs,
            args.coverage_threshold,
        )
        accounting.extend(summary)
        target_deltas.extend(target_changes)
        membership_deltas.extend(membership_changes)
        rank_dispositions.extend(
            source_rank_dispositions(
                label, "original", old_obs, args.coverage_threshold
            )
        )
        rank_dispositions.extend(
            source_rank_dispositions(
                label, CORPUS_VERSION, new_obs, args.coverage_threshold
            )
        )
        pair_summaries.append(
            pair_accounting(
                label,
                before_games,
                after_games,
                old_obs,
                new_obs,
                args.coverage_threshold,
                new_pair_path,
            )
        )
    write_csv(
        args.output_root / "cohort_before_after_accounting.csv",
        list(accounting[0]),
        accounting,
    )
    write_csv(
        args.output_root / "target_team_changes.csv",
        list(target_deltas[0]) if target_deltas else ["corpus"],
        target_deltas,
    )
    write_csv(
        args.output_root / "population_membership_changes.csv",
        list(membership_deltas[0]) if membership_deltas else ["corpus"],
        membership_deltas,
    )
    write_csv(
        args.output_root / "rank_pair_accounting.csv",
        list(pair_summaries[0]),
        pair_summaries,
    )
    rank_disposition_fields = [
        "corpus",
        "version",
        "season",
        "subdivision",
        "team_id",
        "team_name",
        "system_code",
        "source_rank",
        "cmp_population",
        "system_coverage_ratio",
        "selected_for_target",
        "disposition",
        "source_file",
    ]
    write_csv(
        args.output_root / "source_rank_support_dispositions.csv",
        rank_disposition_fields,
        rank_dispositions,
    )
    named_inputs = [
        ("original_final_observations", args.original_final, "original/final_observations.csv"),
        (
            "original_final_targets",
            args.original_final_targets,
            "original/team_season_rank_distributions.csv",
        ),
        (
            "original_pre_bowl_observations",
            args.original_pre_bowl,
            "original/pre_bowl_observations.csv",
        ),
        (
            "original_pre_bowl_targets",
            args.original_pre_bowl_targets,
            "original/pre_bowl_target_distributions.csv",
        ),
        (
            "corrected_final_observations",
            args.corrected_final,
            "data/processed/massey_issue_204_v1/final_observations.csv",
        ),
        (
            "corrected_final_targets",
            args.corrected_final_targets,
            "data/processed/massey_prebowl_issue_204_v1/final_target_distributions.csv",
        ),
        (
            "corrected_final_rank_pairs",
            args.corrected_final_pairs,
            "data/processed/massey_prebowl_issue_204_v1/final_constituent_rank_pairs.csv",
        ),
        (
            "corrected_pre_bowl_observations",
            args.corrected_pre_bowl,
            "data/processed/massey_prebowl_issue_204_v1/pre_bowl_observations.csv",
        ),
        (
            "corrected_pre_bowl_targets",
            args.corrected_pre_bowl_targets,
            "data/processed/massey_prebowl_issue_204_v1/pre_bowl_target_distributions.csv",
        ),
        (
            "corrected_pre_bowl_rank_pairs",
            args.corrected_pre_bowl_pairs,
            "data/processed/massey_prebowl_issue_204_v1/pre_bowl_constituent_rank_pairs.csv",
        ),
    ]
    inputs = [
        {
            "name": name,
            "path": logical_path,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path, logical_path in named_inputs
    ]
    for path in sorted(game_sources, key=lambda item: item.as_posix()):
        game_name = (
            path.relative_to(args.games).as_posix()
            if args.games.is_dir()
            else path.name
        )
        inputs.append(
            {
                "name": "cached_cfbd_game_schedule",
                "path": f"data/raw/cfbd/games/{game_name}",
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema_version": 2,
        "corpus_version": CORPUS_VERSION,
        "coverage_threshold": args.coverage_threshold,
        "game_source_root": "data/raw/cfbd/games",
        "model_fitting_performed": False,
        "inputs": inputs,
        "pair_accounting": pair_summaries,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(args.output_root.glob("*.csv"))
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main(parse_args())
